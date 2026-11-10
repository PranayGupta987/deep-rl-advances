"""
Distributional Q-Network for Quantile Regression DQN (QR-DQN).

Implements the architecture from:
  Dabney et al., "Distributional Reinforcement Learning with Quantile Regression",
  ICML 2018 (https://arxiv.org/abs/1710.10044)

Instead of outputting a single scalar Q(s,a), the network outputs N quantile
atoms θ_i(s,a) for each action, representing the full return distribution Z(s,a).

The quantile midpoints τ_i = (2i - 1) / (2N) for i = 1,...,N correspond to
evenly spaced cumulative probability levels, ensuring the quantile estimates
uniformly cover the return distribution.
"""

import torch
import torch.nn as nn
import numpy as np


class QuantileNetwork(nn.Module):
    """
    MLP-based Quantile Regression DQN network.

    For each action a ∈ A, the network outputs N quantile values
    {θ_1(s,a), ..., θ_N(s,a)}, so the total output dimension is |A| × N.

    Architecture
    ------------
    state_dim → Linear(hidden) → ReLU → Linear(hidden) → ReLU
              → Linear(hidden/2) → ReLU → Linear(n_actions × n_quantiles)
              → reshape to (batch, n_actions, n_quantiles)

    Parameters
    ----------
    state_dim   : int   - dimensionality of the observation space
    n_actions   : int   - number of discrete actions
    n_quantiles : int   - number of quantile atoms (N), default 51
    hidden_dim  : int   - width of hidden layers, default 128
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        n_quantiles: int = 51,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles

        # Shared feature extraction backbone
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Quantile output head: maps features → (n_actions × n_quantiles) values
        self.quantile_head = nn.Linear(hidden_dim // 2, n_actions * n_quantiles)

        # Pre-compute fixed quantile midpoints τ_i = (2i-1) / (2N) for i=1,...,N
        # These are registered as a buffer so they move with the model to GPU/CPU
        taus = (2 * torch.arange(1, n_quantiles + 1, dtype=torch.float32) - 1) / (2 * n_quantiles)
        self.register_buffer("taus", taus)   # shape: (N,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch_size, state_dim)

        Returns
        -------
        quantiles : Tensor of shape (batch_size, n_actions, n_quantiles)
            The predicted quantile values θ_i(s, a) for every action and quantile.
        """
        features = self.feature_net(x)                          # (B, hidden/2)
        raw = self.quantile_head(features)                      # (B, n_actions * N)
        return raw.view(-1, self.n_actions, self.n_quantiles)   # (B, A, N)

    def q_values(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the mean Q-value for each action by averaging over quantiles.

        Q(s, a) = (1/N) Σ_i θ_i(s, a)

        This is the risk-neutral expected value used for standard action selection.

        Parameters
        ----------
        x : Tensor of shape (batch_size, state_dim)

        Returns
        -------
        q : Tensor of shape (batch_size, n_actions)
        """
        quantiles = self.forward(x)           # (B, A, N)
        return quantiles.mean(dim=2)          # (B, A)

    def risk_sensitive_q(
        self,
        x: torch.Tensor,
        risk_param: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute risk-sensitive Q-values using the Wang distortion risk measure.

        The Wang transform applies a normal CDF distortion to the quantile
        probabilities, shifting weight toward lower (risk-averse, ξ > 0) or
        higher (risk-seeking, ξ < 0) quantiles:

            g(τ) = Φ(Φ⁻¹(τ) + ξ)

        where Φ is the standard normal CDF and ξ ∈ ℝ is the risk parameter.

        Q_ξ(s, a) = Σ_i w_i(ξ) · θ_i(s, a)

        where w_i(ξ) = g(τ_i + 1/(2N)) - g(τ_i - 1/(2N)) are the distorted
        probability weights for each quantile bin.

        Parameters
        ----------
        x          : Tensor of shape (batch_size, state_dim)
        risk_param : float — the Wang distortion parameter ξ.
                     ξ = 0  → risk-neutral (standard mean)
                     ξ > 0  → risk-averse  (overweights lower quantiles / pessimistic)
                     ξ < 0  → risk-seeking  (overweights upper quantiles / optimistic)

        Returns
        -------
        q_risk : Tensor of shape (batch_size, n_actions)
        """
        if abs(risk_param) < 1e-8:
            return self.q_values(x)

        quantiles = self.forward(x)   # (B, A, N)

        # Compute distorted weights using the Wang transform:
        #   g(τ) = Φ(Φ⁻¹(τ) + ξ)
        # where bin edges are τ_i ± 1/(2N)
        half_width = 1.0 / (2.0 * self.n_quantiles)
        tau_lower = self.taus - half_width   # (N,)
        tau_upper = self.taus + half_width   # (N,)

        # Clamp to avoid numerical issues at the boundaries of Φ⁻¹
        tau_lower = tau_lower.clamp(1e-6, 1 - 1e-6)
        tau_upper = tau_upper.clamp(1e-6, 1 - 1e-6)

        # Standard normal inverse CDF (quantile function)
        phi_inv_lower = torch.erfinv(2 * tau_lower - 1) * np.sqrt(2)
        phi_inv_upper = torch.erfinv(2 * tau_upper - 1) * np.sqrt(2)

        # Wang distortion: shift by ξ then apply Φ
        g_lower = 0.5 * (1 + torch.erf((phi_inv_lower + risk_param) / np.sqrt(2)))
        g_upper = 0.5 * (1 + torch.erf((phi_inv_upper + risk_param) / np.sqrt(2)))

        # Distorted bin weights
        weights = (g_upper - g_lower)         # (N,)
        weights = weights / weights.sum()     # normalize to valid probability distribution

        # Weighted sum over quantiles: Q_ξ(s,a) = Σ_i w_i · θ_i(s,a)
        q_risk = (quantiles * weights.unsqueeze(0).unsqueeze(0)).sum(dim=2)   # (B, A)
        return q_risk
