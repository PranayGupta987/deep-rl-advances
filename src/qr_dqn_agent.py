"""
Quantile Regression DQN (QR-DQN) Agent with Risk-Sensitive Policy Extraction.

Implements the full QR-DQN algorithm from:
  Dabney et al., "Distributional Reinforcement Learning with Quantile Regression",
  ICML 2018 (https://arxiv.org/abs/1710.10044)

Key features:
  1. Quantile Regression Loss (asymmetric Huber loss) for distributional Bellman updates.
  2. Double DQN-style target computation to mitigate overestimation.
  3. Prioritized Experience Replay (PER) with quantile-derived TD errors.
  4. Soft Polyak target updates (τ-averaging).
  5. Wang distortion risk-sensitive action selection for risk-averse/seeking policies.

The agent maintains a QuantileNetwork that outputs N quantile atoms per action,
and learns the full return distribution Z(s,a) rather than a scalar Q(s,a).
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.distributional_network import QuantileNetwork
from src.replay_buffer import PrioritizedReplayBuffer


class QRDQNAgent:
    """
    Quantile Regression DQN Agent with Double DQN, PER, Soft Updates, and
    risk-sensitive action selection via Wang distortion.

    Parameters
    ----------
    state_dim      : int    - observation dimensionality
    n_actions      : int    - number of discrete actions
    n_quantiles    : int    - number of quantile atoms N (default 51)
    lr             : float  - Adam learning rate
    gamma          : float  - discount factor
    buffer_size    : int    - PER replay buffer capacity
    batch_size     : int    - minibatch size
    tau            : float  - Polyak averaging coefficient for target network
    kappa          : float  - Huber loss threshold κ (1.0 = standard Huber)
    eps_start      : float  - initial exploration epsilon
    eps_end        : float  - final exploration epsilon
    eps_decay_steps: int    - linear epsilon decay schedule length
    hidden_dim     : int    - hidden layer width
    per_alpha      : float  - PER prioritization exponent
    per_beta_start : float  - PER initial importance-sampling exponent
    risk_param     : float  - Wang distortion parameter ξ for risk-sensitive evaluation
                              (0.0 = risk-neutral, >0 = risk-averse, <0 = risk-seeking)
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        n_quantiles: int = 51,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        tau: float = 0.005,
        kappa: float = 1.0,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_decay_steps: int = 10_000,
        hidden_dim: int = 128,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        risk_param: float = 0.0,
    ):
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.kappa = kappa
        self.risk_param = risk_param

        # Exploration schedule
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = (eps_start - eps_end) / eps_decay_steps

        self.device = torch.device("cpu")

        # Online and target quantile networks
        self.q_net = QuantileNetwork(
            state_dim, n_actions, n_quantiles, hidden_dim
        ).to(self.device)
        self.target_net = copy.deepcopy(self.q_net)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # Prioritized Experience Replay
        self.buffer = PrioritizedReplayBuffer(
            capacity=buffer_size,
            alpha=per_alpha,
            beta_start=per_beta_start,
            beta_anneal_steps=10_000,
        )
        self.step_count = 0

        # Diagnostics: track quantile statistics for analysis
        self.quantile_history = []

    # ------------------------------------------------------------------
    # Action Selection
    # ------------------------------------------------------------------
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """
        Select an action using epsilon-greedy exploration with optional
        risk-sensitive Q-value computation.

        During evaluation (eval_mode=True), no exploration noise is added and
        the risk_param is used for distorted action selection.
        """
        if not eval_mode and np.random.rand() < self.eps:
            return np.random.randint(self.n_actions)

        state_t = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            if abs(self.risk_param) > 1e-8 and eval_mode:
                q_vals = self.q_net.risk_sensitive_q(state_t, self.risk_param)
            else:
                q_vals = self.q_net.q_values(state_t)

        return int(q_vals.argmax(dim=1).item())

    def store(self, state, action, reward, next_state, done):
        """Push a transition into the PER buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    def decay_eps(self):
        """Decay exploration epsilon at the end of each episode."""
        self.eps = max(self.eps_end, self.eps * 0.99)

    # ------------------------------------------------------------------
    # Quantile Huber Loss
    # ------------------------------------------------------------------
    def _quantile_huber_loss(
        self,
        quantiles: torch.Tensor,
        target_quantiles: torch.Tensor,
        taus: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple:
        """
        Compute the quantile Huber loss for distributional Bellman updates.

        For each pair (i, j) of current quantile θ_i and target quantile θ'_j,
        the asymmetric Huber loss is:

            ρ^κ_τ(u) = |τ - 𝟙(u < 0)| · L_κ(u)

        where L_κ(u) is the Huber loss with threshold κ, and u = θ'_j - θ_i.

        Parameters
        ----------
        quantiles        : (B, N) — current quantile predictions for selected action
        target_quantiles : (B, N) — target quantile values
        taus             : (N,)   — quantile midpoints
        weights          : (B,)   — PER importance-sampling weights

        Returns
        -------
        loss      : scalar tensor — weighted mean quantile Huber loss
        td_errors : (B,) numpy   — mean absolute TD error per sample (for PER)
        """
        B, N = quantiles.shape

        # Pairwise differences: θ'_j - θ_i for all (i, j) pairs
        # quantiles:        (B, N, 1)
        # target_quantiles: (B, 1, N)
        # diff:             (B, N, N) — diff[b, i, j] = target_j - current_i
        diff = target_quantiles.unsqueeze(1) - quantiles.unsqueeze(2)   # (B, N, N)

        # Huber loss element-wise
        abs_diff = diff.abs()
        huber = torch.where(
            abs_diff <= self.kappa,
            0.5 * diff.pow(2),
            self.kappa * (abs_diff - 0.5 * self.kappa),
        )   # (B, N, N)

        # Asymmetric quantile weighting: |τ_i - 𝟙(diff < 0)|
        # taus shape: (N,) → (1, N, 1) for broadcasting
        indicator = (diff < 0).float()
        tau_weight = (taus.view(1, -1, 1) - indicator).abs()   # (B, N, N)

        # Element-wise quantile Huber loss
        qh_loss = tau_weight * huber   # (B, N, N)

        # Mean over target quantiles j, then over current quantiles i
        loss_per_sample = qh_loss.sum(dim=2).mean(dim=1)   # (B,)

        # Apply PER importance-sampling weights
        loss = (loss_per_sample * weights).mean()

        # TD-error for PER priority updates: mean absolute pairwise difference
        with torch.no_grad():
            td_errors = abs_diff.mean(dim=(1, 2)).cpu().numpy()   # (B,)

        return loss, td_errors

    # ------------------------------------------------------------------
    # Update Step
    # ------------------------------------------------------------------
    def update(self) -> dict | None:
        """
        Perform one gradient step of the QR-DQN algorithm.

        Returns
        -------
        info : dict with 'loss', 'mean_q', 'quantile_spread' or None if buffer too small
        """
        if len(self.buffer) < self.batch_size:
            return None

        # Sample from PER
        states, actions, rewards, next_states, dones, indices, weights = \
            self.buffer.sample(self.batch_size)

        states_t      = torch.tensor(states,      device=self.device)
        actions_t     = torch.tensor(actions,      device=self.device, dtype=torch.long)
        rewards_t     = torch.tensor(rewards,      device=self.device)
        next_states_t = torch.tensor(next_states,  device=self.device)
        dones_t       = torch.tensor(dones,        device=self.device)
        weights_t     = torch.tensor(weights,      device=self.device)

        B = states_t.shape[0]
        N = self.n_quantiles

        # ── Current quantiles for selected actions ──────────────────────
        all_quantiles = self.q_net(states_t)   # (B, A, N)
        # Gather quantiles for the actions that were actually taken
        action_idx = actions_t.view(B, 1, 1).expand(B, 1, N)
        current_quantiles = all_quantiles.gather(1, action_idx).squeeze(1)   # (B, N)

        # ── Target quantiles (Double DQN-style) ────────────────────────
        with torch.no_grad():
            # Action selection: online network picks the best action
            next_q = self.q_net.q_values(next_states_t)   # (B, A)
            best_actions = next_q.argmax(dim=1)            # (B,)

            # Action evaluation: target network provides quantile values
            next_all_quantiles = self.target_net(next_states_t)   # (B, A, N)
            best_action_idx = best_actions.view(B, 1, 1).expand(B, 1, N)
            next_quantiles = next_all_quantiles.gather(1, best_action_idx).squeeze(1)  # (B, N)

            # Distributional Bellman target: T θ_j = r + γ(1-d) θ_j(s', a*)
            target_quantiles = rewards_t.unsqueeze(1) + \
                self.gamma * (1.0 - dones_t.unsqueeze(1)) * next_quantiles   # (B, N)

        # ── Quantile Huber loss ─────────────────────────────────────────
        taus = self.q_net.taus   # (N,)
        loss, td_errors = self._quantile_huber_loss(
            current_quantiles, target_quantiles, taus, weights_t
        )

        # ── Gradient step ───────────────────────────────────────────────
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        # ── PER priority update ─────────────────────────────────────────
        self.buffer.update_priorities(indices, td_errors)

        # ── Soft target update (Polyak averaging) ───────────────────────
        self.step_count += 1

        for target_p, online_p in zip(
            self.target_net.parameters(), self.q_net.parameters()
        ):
            target_p.data.copy_(
                self.tau * online_p.data + (1.0 - self.tau) * target_p.data
            )

        # ── Diagnostics ────────────────────────────────────────────────
        with torch.no_grad():
            mean_q = current_quantiles.mean().item()
            # Spread = mean std across actions → measures distributional variance
            quantile_spread = current_quantiles.std(dim=1).mean().item()

        return {
            "loss": loss.item(),
            "mean_q": mean_q,
            "quantile_spread": quantile_spread,
        }

    # ------------------------------------------------------------------
    # Evaluation Utilities
    # ------------------------------------------------------------------
    def get_return_distribution(self, state: np.ndarray) -> dict:
        """
        Get the full predicted return distribution for all actions at a given state.

        Returns
        -------
        dict with:
            'quantiles'  : ndarray (A, N) — quantile values per action
            'taus'       : ndarray (N,)   — quantile midpoints
            'q_values'   : ndarray (A,)   — mean Q-values (risk-neutral)
            'q_risk'     : ndarray (A,)   — risk-sensitive Q-values (Wang distortion)
        """
        state_t = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            quantiles = self.q_net(state_t).squeeze(0)   # (A, N)
            q_vals = quantiles.mean(dim=1)               # (A,)
            q_risk = self.q_net.risk_sensitive_q(state_t, self.risk_param).squeeze(0)

        return {
            "quantiles": quantiles.cpu().numpy(),
            "taus": self.q_net.taus.cpu().numpy(),
            "q_values": q_vals.cpu().numpy(),
            "q_risk": q_risk.cpu().numpy(),
        }
