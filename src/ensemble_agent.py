r"""
Bootstrapped Ensemble DQN with Epistemic Uncertainty-Guided Exploration (Iteration 3).

Grounding Literature:
  - Osband et al., "Deep Exploration via Bootstrapped DQN", NeurIPS 2016.
  - Osband et al., "Randomized Prior Functions for Deep Reinforcement Learning", NeurIPS 2018.
  - Pathak et al., "Curiosity-driven Exploration by Self-supervised Prediction", ICML 2017.
  - MacKay et al., "Self-Tuning Networks: Bilevel Optimization of Hyperparameters using Meta-Gradients", ICLR 2019.

Mathematical Formulation:
  Ensemble Q-heads: Q_k(s, a),  k \in {1, ..., K}
  Epistemic Uncertainty:
      sigma_epistemic(s, a) = sqrt( 1/K sum_k (Q_k(s, a) - bar{Q}(s, a))^2 )
  Intrinsic Reward Bonus:
      r^+(s, a) = r + beta * sigma_epistemic(s, a)
  Bootstrapped Masking:
      Each transition is assigned a Bernoulli bootstrap mask vector m ~ Bern(p=0.7)^K.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

torch.set_num_threads(1)

from src.replay_buffer import PrioritizedReplayBuffer


class BootstrappedQNetwork(nn.Module):
    """
    Multi-head Bootstrapped Q-Network with randomized prior functions.
    Each of the K heads learns an independent value posterior from bootstrapped data masks.
    """
    def __init__(self, state_dim: int, n_actions: int, num_heads: int = 5, hidden_dim: int = 128):
        super().__init__()
        self.num_heads = num_heads
        self.n_actions = n_actions

        # Shared feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # K independent Q-value heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, n_actions)
            ) for _ in range(num_heads)
        ])

        # Untrainable randomized prior network (Osband et al., 2018) to prevent posterior collapse
        self.prior_feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.prior_heads = nn.ModuleList([
            nn.Linear(hidden_dim, n_actions) for _ in range(num_heads)
        ])
        for p in self.prior_feature_net.parameters():
            p.requires_grad = False
        for head in self.prior_heads:
            for p in head.parameters():
                p.requires_grad = False
        self.prior_scale = 1.0

    def forward(self, x: torch.Tensor, head_idx: int = None) -> torch.Tensor:
        """
        Forward pass.
        If head_idx is specified, returns Q-values for that head: (B, n_actions).
        If head_idx is None, returns Q-values across all heads: (K, B, n_actions).
        """
        feats = self.feature_net(x)
        with torch.no_grad():
            prior_feats = self.prior_feature_net(x)

        if head_idx is not None:
            q = self.heads[head_idx](feats)
            prior = self.prior_heads[head_idx](prior_feats) * self.prior_scale
            return q + prior

        all_q = []
        for k in range(self.num_heads):
            q_k = self.heads[k](feats) + self.prior_heads[k](prior_feats) * self.prior_scale
            all_q.append(q_k)
        return torch.stack(all_q, dim=0)   # (K, B, n_actions)

    def compute_epistemic_uncertainty(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the ensemble mean Q-value and epistemic standard deviation for each action.
        Returns
        -------
        mean_q : (B, n_actions)
        uncertainty : (B, n_actions)
        """
        all_q = self.forward(x)   # (K, B, n_actions)
        mean_q = all_q.mean(dim=0)
        uncertainty = all_q.std(dim=0, unbiased=True)
        return mean_q, uncertainty


class EnsembleBootstrappedAgent:
    """
    Epistemic Uncertainty-Guided Bootstrapped Ensemble DQN Agent (Iteration 3).
    """
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        num_heads: int = 5,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        tau: float = 0.005,
        beta_uncertainty: float = 0.05,
        bootstrap_p: float = 0.7,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_decay: float = 0.99,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.num_heads = num_heads
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.beta_uncertainty = beta_uncertainty
        self.bootstrap_p = bootstrap_p

        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.device = torch.device("cpu")

        # Online ensemble network & target network
        self.q_net = BootstrappedQNetwork(state_dim, n_actions, num_heads, hidden_dim).to(self.device)
        self.target_net = copy.deepcopy(self.q_net)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # Prioritized buffer
        self.buffer = PrioritizedReplayBuffer(
            capacity=buffer_size,
            alpha=0.6,
            beta_start=0.4,
            beta_anneal_steps=10_000,
        )

        # Active head for deep exploration during each episode (Thompson Sampling-style)
        self.active_head = np.random.randint(num_heads)
        self.step_count = 0

    def reset_episode_head(self):
        """Thompson sampling: sample a random head at the start of each episode."""
        self.active_head = np.random.randint(self.num_heads)

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        if not eval_mode and np.random.rand() < self.eps:
            return np.random.randint(self.n_actions)

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if eval_mode:
                # In evaluation: take majority consensus / ensemble mean action
                mean_q, _ = self.q_net.compute_epistemic_uncertainty(state_t)
                return int(mean_q.argmax(dim=1).item())
            else:
                # During training: use the sampled active head (deep exploration)
                q_head = self.q_net(state_t, head_idx=self.active_head)
                return int(q_head.argmax(dim=1).item())

    def act(self, state: np.ndarray) -> int:
        return self.select_action(state, eval_mode=False)

    def decay_eps(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)
        self.reset_episode_head()

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

        # ── 1. Epistemic Uncertainty & Intrinsic Exploration Bonus ──────
        with torch.no_grad():
            _, uncertainty_next = self.q_net.compute_epistemic_uncertainty(next_states_t)
            # Intrinsic bonus based on max potential epistemic uncertainty
            bonus = self.beta_uncertainty * uncertainty_next.max(dim=1).values
            augmented_rewards = rewards_t + bonus

        # ── 2. Multi-Head Bellman Targets with Bootstrap Masking ────────
        # Generate Bernoulli bootstrap mask: (K, B)
        masks = (torch.rand(self.num_heads, B, device=self.device) < self.bootstrap_p).float()

        total_loss = torch.tensor(0.0, device=self.device)
        head_td_errors = []

        # Target action selection using online ensemble mean (Double DQN style)
        with torch.no_grad():
            next_mean_q, _ = self.q_net.compute_epistemic_uncertainty(next_states_t)
            best_actions = next_mean_q.argmax(dim=1, keepdim=True)
            # Target head predictions
            all_target_q = self.target_net(next_states_t)  # (K, B, n_actions)

        all_online_q = self.q_net(states_t)  # (K, B, n_actions)

        for k in range(self.num_heads):
            # Target Q-value for head k
            q_target_k = all_target_q[k].gather(1, best_actions).squeeze(1)
            target_val = augmented_rewards + self.gamma * (1.0 - dones_t) * q_target_k

            # Online Q-value for head k
            current_q_k = all_online_q[k].gather(1, actions_t.unsqueeze(1)).squeeze(1)

            # Masked Huber loss
            diff = target_val - current_q_k
            abs_diff = diff.abs()
            huber = torch.where(abs_diff <= 1.0, 0.5 * diff.pow(2), abs_diff - 0.5)

            # Apply bootstrap mask and PER weights
            masked_loss = (huber * weights_t * masks[k]).sum() / (masks[k].sum() + 1e-8)
            total_loss = total_loss + masked_loss
            head_td_errors.append(abs_diff.detach())

        # Average TD errors across heads for PER priority updates
        mean_td_errors = torch.stack(head_td_errors, dim=0).mean(dim=0).cpu().numpy()

        # ── 3. Gradient Optimization ────────────────────────────────────
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        # ── 4. PER Update ───────────────────────────────────────────────
        self.buffer.update_priorities(indices, mean_td_errors)

        # ── 5. Polyak Soft Target Updates ───────────────────────────────
        for target_p, online_p in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_p.data.copy_(self.tau * online_p.data + (1.0 - self.tau) * target_p.data)

        self.step_count += 1
        with torch.no_grad():
            mean_q_val = all_online_q.mean().item()
            epistemic_var = all_online_q.std(dim=0).mean().item()

        return {
            "loss": total_loss.item(),
            "mean_q": mean_q_val,
            "epistemic_uncertainty": epistemic_var,
            "quantile_spread": epistemic_var,
        }
