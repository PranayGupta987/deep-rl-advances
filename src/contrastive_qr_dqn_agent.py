"""
Contrastive QR-DQN Agent (Iteration 2).

Integrates Self-Supervised Contrastive Representation Learning (CURL / SPR)
with Distributional Quantile Regression DQN (QR-DQN).

Joint Objective:
  L_total = L_QR + lambda_curl * L_InfoNCE

Key Capabilities:
  1. Shared latent representation backbone regularized by InfoNCE auxiliary loss.
  2. Momentum target encoder with slow EMA updates.
  3. Wang distortion risk-sensitive policy extraction over 51 quantile atoms.
  4. Prioritized Experience Replay (Sum-Tree).
  5. Latent linear probe evaluation for representation disentanglement tracking.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

torch.set_num_threads(1)

from src.contrastive_encoder import ContrastiveRepresentationModule, LatentLinearProbe
from src.replay_buffer import PrioritizedReplayBuffer


class QuantileHead(nn.Module):
    """
    Quantile output projection head mapping latent state representation z to
    quantile atoms for all discrete actions: (B, n_actions, n_quantiles).
    """
    def __init__(self, latent_dim: int, n_actions: int, n_quantiles: int = 51):
        super().__init__()
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles
        self.head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, n_actions * n_quantiles)
        )
        taus = (2 * torch.arange(1, n_quantiles + 1, dtype=torch.float32) - 1) / (2 * n_quantiles)
        self.register_buffer("taus", taus)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raw = self.head(z)
        return raw.view(-1, self.n_actions, self.n_quantiles)

    def q_values(self, z: torch.Tensor) -> torch.Tensor:
        return self.forward(z).mean(dim=2)

    def risk_sensitive_q(self, z: torch.Tensor, risk_param: float = 0.0) -> torch.Tensor:
        if abs(risk_param) < 1e-8:
            return self.q_values(z)
        quantiles = self.forward(z)
        half_width = 1.0 / (2.0 * self.n_quantiles)
        tau_lower = (self.taus - half_width).clamp(1e-6, 1.0 - 1e-6)
        tau_upper = (self.taus + half_width).clamp(1e-6, 1.0 - 1e-6)
        phi_inv_l = torch.erfinv(2 * tau_lower - 1) * np.sqrt(2)
        phi_inv_u = torch.erfinv(2 * tau_upper - 1) * np.sqrt(2)
        g_l = 0.5 * (1 + torch.erf((phi_inv_l + risk_param) / np.sqrt(2)))
        g_u = 0.5 * (1 + torch.erf((phi_inv_u + risk_param) / np.sqrt(2)))
        weights = (g_u - g_l)
        weights = weights / weights.sum()
        return (quantiles * weights.unsqueeze(0).unsqueeze(0)).sum(dim=2)


class ContrastiveQRDQNAgent:
    """
    Combined Self-Supervised Contrastive + Quantile Regression DQN Agent.
    """
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        n_quantiles: int = 51,
        latent_dim: int = 128,
        proj_dim: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        tau: float = 0.005,
        kappa: float = 1.0,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_decay: float = 0.99,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        risk_param: float = 0.0,
        contrastive_coeff: float = 0.1,
        temperature: float = 0.1,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles
        self.latent_dim = latent_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.kappa = kappa
        self.risk_param = risk_param
        self.contrastive_coeff = contrastive_coeff

        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.device = torch.device("cpu")

        # Contrastive representation module (encoder + momentum target + projector)
        self.contrastive_module = ContrastiveRepresentationModule(
            state_dim=state_dim,
            latent_dim=latent_dim,
            proj_dim=proj_dim,
            temperature=temperature,
        ).to(self.device)

        # Quantile prediction head
        self.quantile_head = QuantileHead(
            latent_dim=latent_dim,
            n_actions=n_actions,
            n_quantiles=n_quantiles,
        ).to(self.device)

        # Target quantile head
        self.target_quantile_head = copy.deepcopy(self.quantile_head)
        self.target_quantile_head.eval()

        # Optimizer: joint optimization over encoder, projector, W, and quantile head
        all_params = list(self.contrastive_module.encoder.parameters()) + \
                     list(self.contrastive_module.projector.parameters()) + \
                     [self.contrastive_module.W] + \
                     list(self.quantile_head.parameters())
        self.optimizer = optim.Adam(all_params, lr=lr)

        # Prioritized Experience Replay
        self.buffer = PrioritizedReplayBuffer(
            capacity=buffer_size,
            alpha=per_alpha,
            beta_start=per_beta_start,
            beta_anneal_steps=10_000,
        )
        self.step_count = 0

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        if not eval_mode and np.random.rand() < self.eps:
            return np.random.randint(self.n_actions)

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            z = self.contrastive_module.encoder(state_t)
            if abs(self.risk_param) > 1e-8 and eval_mode:
                q_vals = self.quantile_head.risk_sensitive_q(z, self.risk_param)
            else:
                q_vals = self.quantile_head.q_values(z)
        return int(q_vals.argmax(dim=1).item())

    def act(self, state: np.ndarray) -> int:
        return self.select_action(state, eval_mode=False)

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def decay_eps(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def _quantile_huber_loss(
        self,
        quantiles: torch.Tensor,
        target_quantiles: torch.Tensor,
        taus: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, np.ndarray]:
        diff = target_quantiles.unsqueeze(1) - quantiles.unsqueeze(2)  # (B, N, N)
        abs_diff = diff.abs()
        huber = torch.where(
            abs_diff <= self.kappa,
            0.5 * diff.pow(2),
            self.kappa * (abs_diff - 0.5 * self.kappa),
        )
        indicator = (diff < 0).float()
        tau_weight = (taus.view(1, -1, 1) - indicator).abs()
        qh_loss = tau_weight * huber
        loss_per_sample = qh_loss.sum(dim=2).mean(dim=1)
        loss = (loss_per_sample * weights).mean()
        with torch.no_grad():
            td_errors = abs_diff.mean(dim=(1, 2)).cpu().numpy()
        return loss, td_errors

    def update(self) -> dict | None:
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones, indices, weights = \
            self.buffer.sample(self.batch_size)

        states_t      = torch.tensor(states,      device=self.device)
        actions_t     = torch.tensor(actions,     device=self.device, dtype=torch.long)
        rewards_t     = torch.tensor(rewards,     device=self.device)
        next_states_t = torch.tensor(next_states, device=self.device)
        dones_t       = torch.tensor(dones,       device=self.device)
        weights_t     = torch.tensor(weights,     device=self.device)

        B = states_t.shape[0]
        N = self.n_quantiles

        # ── 1. Latent Representations ────────────────────────────────────
        z_online = self.contrastive_module.encoder(states_t)  # (B, latent_dim)

        # ── 2. Quantile Predictions ─────────────────────────────────────
        all_quantiles = self.quantile_head(z_online)  # (B, n_actions, N)
        action_idx = actions_t.view(B, 1, 1).expand(B, 1, N)
        current_quantiles = all_quantiles.gather(1, action_idx).squeeze(1)  # (B, N)

        # ── 3. Target Quantile Calculation ──────────────────────────────
        with torch.no_grad():
            # Target features from momentum target encoder
            z_target = self.contrastive_module.target_encoder(next_states_t)
            # Online network selects greedy action on target features
            z_next_online = self.contrastive_module.encoder(next_states_t)
            next_q = self.quantile_head.q_values(z_next_online)
            best_actions = next_q.argmax(dim=1)

            next_all_quantiles = self.target_quantile_head(z_target)
            best_action_idx = best_actions.view(B, 1, 1).expand(B, 1, N)
            next_quantiles = next_all_quantiles.gather(1, best_action_idx).squeeze(1)

            # Distributional Bellman target
            target_quantiles = rewards_t.unsqueeze(1) + \
                self.gamma * (1.0 - dones_t.unsqueeze(1)) * next_quantiles

        # ── 4. Quantile Huber Loss ──────────────────────────────────────
        taus = self.quantile_head.taus
        qr_loss, td_errors = self._quantile_huber_loss(
            current_quantiles, target_quantiles, taus, weights_t
        )

        # ── 5. Auxiliary InfoNCE Contrastive Loss ────────────────────────
        infonce_loss, contrast_acc = self.contrastive_module.compute_infonce_loss(
            states_t, next_states_t
        )

        # ── 6. Joint Loss ───────────────────────────────────────────────
        total_loss = qr_loss + self.contrastive_coeff * infonce_loss

        # ── 7. Optimization Step ────────────────────────────────────────
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.contrastive_module.parameters()) + list(self.quantile_head.parameters()),
            10.0
        )
        self.optimizer.step()

        # ── 8. PER Priority Update ──────────────────────────────────────
        self.buffer.update_priorities(indices, td_errors)

        # ── 9. Polyak & Momentum Target Updates ─────────────────────────
        self.contrastive_module.update_momentum()
        for target_p, online_p in zip(
            self.target_quantile_head.parameters(), self.quantile_head.parameters()
        ):
            target_p.data.copy_(
                self.tau * online_p.data + (1.0 - self.tau) * target_p.data
            )

        self.step_count += 1
        with torch.no_grad():
            mean_q = current_quantiles.mean().item()
            quantile_spread = current_quantiles.std(dim=1).mean().item()

        return {
            "loss": total_loss.item(),
            "qr_loss": qr_loss.item(),
            "infonce_loss": infonce_loss.item(),
            "contrast_acc": contrast_acc,
            "mean_q": mean_q,
            "quantile_spread": quantile_spread,
        }

    def evaluate_representation_quality(self, test_states: np.ndarray) -> float:
        """
        Evaluate linear probe R^2 score of frozen representations predicting next-state dynamics.
        """
        with torch.no_grad():
            states_t = torch.tensor(test_states[:-1], dtype=torch.float32, device=self.device)
            z = self.contrastive_module.encoder(states_t).cpu().numpy()
        targets = test_states[1:]
        probe = LatentLinearProbe(latent_dim=self.latent_dim, target_dim=test_states.shape[1])
        return probe.fit_and_evaluate(z, targets, epochs=50)
