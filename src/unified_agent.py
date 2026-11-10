"""
Full Unified Agent (Iteration 3 Apex):
Combines Distributional QR-DQN + Contrastive Representation Learning (CURL) +
Bootstrapped Ensemble Epistemic Uncertainty Exploration.

Grounding:
  1. Dabney et al. (ICML 2018): 51-quantile regression Bellman loss
  2. Srinivas et al. (ICML 2020) & Schwarzer et al. (ICLR 2021): Bilinear InfoNCE loss with momentum target encoder
  3. Osband et al. (NeurIPS 2016, 2018): Epistemic uncertainty exploration bonus
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

torch.set_num_threads(1)

from src.contrastive_encoder import ContrastiveRepresentationModule
from src.replay_buffer import PrioritizedReplayBuffer


class EnsembleQuantileHead(nn.Module):
    """
    Multi-head Quantile Regression head: K independent heads, each predicting
    51 quantile atoms for all actions.
    """
    def __init__(self, latent_dim: int, n_actions: int, num_heads: int = 3, n_quantiles: int = 51):
        super().__init__()
        self.num_heads = num_heads
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, latent_dim // 2),
                nn.ReLU(),
                nn.Linear(latent_dim // 2, n_actions * n_quantiles)
            ) for _ in range(num_heads)
        ])
        taus = (2 * torch.arange(1, n_quantiles + 1, dtype=torch.float32) - 1) / (2 * n_quantiles)
        self.register_buffer("taus", taus)

    def forward(self, z: torch.Tensor, head_idx: int = None) -> torch.Tensor:
        if head_idx is not None:
            raw = self.heads[head_idx](z)
            return raw.view(-1, self.n_actions, self.n_quantiles)
        all_q = [head(z).view(-1, self.n_actions, self.n_quantiles) for head in self.heads]
        return torch.stack(all_q, dim=0)   # (K, B, A, N)

    def q_values(self, z: torch.Tensor) -> torch.Tensor:
        """Ensemble mean Q-values averaged over heads and quantiles: (B, A)."""
        all_quantiles = self.forward(z)   # (K, B, A, N)
        return all_quantiles.mean(dim=(0, 3))

    def compute_epistemic_uncertainty(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute mean Q-value and epistemic standard deviation across heads."""
        all_quantiles = self.forward(z)   # (K, B, A, N)
        head_means = all_quantiles.mean(dim=3)   # (K, B, A)
        mean_q = head_means.mean(dim=0)          # (B, A)
        epistemic_std = head_means.std(dim=0)    # (B, A)
        return mean_q, epistemic_std


class FullUnifiedAgent:
    """
    Unified Conference-Level Agent:
    Contrastive Representation Learning + Quantile Regression + Epistemic Uncertainty Bonus.
    """
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        n_quantiles: int = 51,
        num_heads: int = 3,
        latent_dim: int = 128,
        proj_dim: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        tau: float = 0.005,
        kappa: float = 1.0,
        contrastive_coeff: float = 0.1,
        beta_uncertainty: float = 0.05,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_decay: float = 0.99,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles
        self.num_heads = num_heads
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.kappa = kappa
        self.contrastive_coeff = contrastive_coeff
        self.beta_uncertainty = beta_uncertainty

        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.device = torch.device("cpu")

        # Contrastive encoder module
        self.contrastive_module = ContrastiveRepresentationModule(
            state_dim=state_dim,
            latent_dim=latent_dim,
            proj_dim=proj_dim,
        ).to(self.device)

        # Ensemble quantile head
        self.quantile_head = EnsembleQuantileHead(
            latent_dim=latent_dim,
            n_actions=n_actions,
            num_heads=num_heads,
            n_quantiles=n_quantiles,
        ).to(self.device)
        self.target_quantile_head = copy.deepcopy(self.quantile_head)
        self.target_quantile_head.eval()

        all_params = list(self.contrastive_module.parameters()) + list(self.quantile_head.parameters())
        self.optimizer = optim.Adam(all_params, lr=lr)

        self.buffer = PrioritizedReplayBuffer(
            capacity=buffer_size,
            alpha=0.6,
            beta_start=0.4,
            beta_anneal_steps=10_000,
        )

        self.active_head = np.random.randint(num_heads)
        self.step_count = 0

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        if not eval_mode and np.random.rand() < self.eps:
            return np.random.randint(self.n_actions)

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            z = self.contrastive_module.encoder(state_t)
            if eval_mode:
                mean_q, _ = self.quantile_head.compute_epistemic_uncertainty(z)
                return int(mean_q.argmax(dim=1).item())
            else:
                q_head = self.quantile_head.forward(z, head_idx=self.active_head).mean(dim=2)
                return int(q_head.argmax(dim=1).item())

    def act(self, state: np.ndarray) -> int:
        return self.select_action(state, eval_mode=False)

    def decay_eps(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)
        self.active_head = np.random.randint(self.num_heads)

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

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

        # ── 1. Latent features & Epistemic Uncertainty Bonus ─────────────
        with torch.no_grad():
            z_next = self.contrastive_module.encoder(next_states_t)
            _, epistemic_std = self.quantile_head.compute_epistemic_uncertainty(z_next)
            bonus = self.beta_uncertainty * epistemic_std.max(dim=1).values
            augmented_rewards = rewards_t + bonus

        # ── 2. Quantile predictions for active head ──────────────────────
        z_curr = self.contrastive_module.encoder(states_t)
        all_quantiles = self.quantile_head(z_curr, head_idx=self.active_head)   # (B, A, N)
        action_idx = actions_t.view(B, 1, 1).expand(B, 1, N)
        current_quantiles = all_quantiles.gather(1, action_idx).squeeze(1)     # (B, N)

        # ── 3. Distributional Target calculation ────────────────────────
        with torch.no_grad():
            z_next_target = self.contrastive_module.target_encoder(next_states_t)
            # Greedy action from online ensemble mean
            next_mean_q, _ = self.quantile_head.compute_epistemic_uncertainty(z_next)
            best_actions = next_mean_q.argmax(dim=1)

            next_all_quantiles = self.target_quantile_head(z_next_target, head_idx=self.active_head)
            best_action_idx = best_actions.view(B, 1, 1).expand(B, 1, N)
            next_quantiles = next_all_quantiles.gather(1, best_action_idx).squeeze(1)

            target_quantiles = augmented_rewards.unsqueeze(1) + \
                self.gamma * (1.0 - dones_t.unsqueeze(1)) * next_quantiles

        # ── 4. Quantile Huber Loss ──────────────────────────────────────
        taus = self.quantile_head.taus
        diff = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)
        abs_diff = diff.abs()
        huber = torch.where(abs_diff <= self.kappa, 0.5 * diff.pow(2), self.kappa * (abs_diff - 0.5 * self.kappa))
        indicator = (diff < 0).float()
        tau_w = (taus.view(1, -1, 1) - indicator).abs()
        qh_loss = (tau_w * huber).sum(dim=2).mean(dim=1)
        qr_loss = (qh_loss * weights_t).mean()

        with torch.no_grad():
            td_errors = abs_diff.mean(dim=(1, 2)).cpu().numpy()

        # ── 5. Auxiliary Contrastive InfoNCE Loss ────────────────────────
        infonce_loss, _ = self.contrastive_module.compute_infonce_loss(states_t, next_states_t)

        # ── 6. Joint Loss ───────────────────────────────────────────────
        total_loss = qr_loss + self.contrastive_coeff * infonce_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.contrastive_module.parameters()) + list(self.quantile_head.parameters()),
            10.0
        )
        self.optimizer.step()

        # ── 7. PER Priority Update ──────────────────────────────────────
        self.buffer.update_priorities(indices, td_errors)

        # ── 8. Momentum & Soft Target Updates ───────────────────────────
        self.contrastive_module.update_momentum()
        for tp, op in zip(self.target_quantile_head.parameters(), self.quantile_head.parameters()):
            tp.data.copy_(self.tau * op.data + (1.0 - self.tau) * tp.data)

        self.step_count += 1
        return {
            "loss": total_loss.item(),
            "qr_loss": qr_loss.item(),
            "infonce_loss": infonce_loss.item(),
            "mean_q": current_quantiles.mean().item(),
            "quantile_spread": current_quantiles.std(dim=1).mean().item(),
        }
