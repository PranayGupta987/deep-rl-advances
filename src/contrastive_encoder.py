r"""
Contrastive Representation Learning Module for Reinforcement Learning (CURL / SPR-inspired).

Grounding Literature:
  - Srinivas et al., "CURL: Contrastive Unsupervised Representations for Reinforcement Learning", ICML 2020.
  - Schwarzer et al., "Data-Efficient Reinforcement Learning with Self-Predictive Representations (SPR)", ICLR 2021.
  - Oord et al., "Representation Learning with Contrastive Predictive Coding (CPC)", arXiv:1807.03748.

Mathematical Formulation:
  Online encoder:        z_t = f_theta(s_t) \in R^d
  Momentum target enc:   z_tgt = f_theta_bar(s_{t+1} + epsilon)
  Projection heads:      q = g_psi(z_t),  k = g_psi_bar(z_tgt)
  InfoNCE loss:
      L_InfoNCE = - 1/B sum_i log [ exp(q_i^T W k_i / tau) / sum_j exp(q_i^T W k_j / tau) ]
  where W in R^{d_proj x d_proj} is a learnable bilinear similarity matrix.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)


class StateEncoder(nn.Module):
    """
    State representation encoder mapping raw observations to latent representations.
    Architecture: MLP with LayerNorm to stabilize contrastive feature scaling.
    """
    def __init__(self, state_dim: int, latent_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
        )
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProjectionHead(nn.Module):
    """
    Non-linear projection head mapping latent features to the hyperspherical contrastive space.
    """
    def __init__(self, latent_dim: int = 128, proj_dim: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, proj_dim)
        )
        self.proj_dim = proj_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        p = self.head(z)
        return F.normalize(p, p=2, dim=-1)


class ContrastiveRepresentationModule(nn.Module):
    """
    Self-Supervised Contrastive Representation Learning module with momentum target encoder
    and learnable bilinear InfoNCE similarity metric.
    """
    def __init__(
        self,
        state_dim: int,
        latent_dim: int = 128,
        proj_dim: int = 64,
        momentum_tau: float = 0.01,
        temperature: float = 0.1,
        noise_std: float = 0.015,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.proj_dim = proj_dim
        self.momentum_tau = momentum_tau
        self.temperature = temperature
        self.noise_std = noise_std

        # Online encoder and projection head
        self.encoder = StateEncoder(state_dim, latent_dim)
        self.projector = ProjectionHead(latent_dim, proj_dim)

        # Momentum target encoder and target projection head (slow-moving EMA)
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_projector = copy.deepcopy(self.projector)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

        # Learnable bilinear similarity matrix: W in R^{proj_dim x proj_dim}
        self.W = nn.Parameter(torch.randn(proj_dim, proj_dim) / np.sqrt(proj_dim))

    def update_momentum(self):
        """Exponential Moving Average (EMA) update for target encoder parameters."""
        with torch.no_grad():
            for p, p_target in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                p_target.data.copy_(
                    (1.0 - self.momentum_tau) * p_target.data + self.momentum_tau * p.data
                )
            for p, p_target in zip(self.projector.parameters(), self.target_projector.parameters()):
                p_target.data.copy_(
                    (1.0 - self.momentum_tau) * p_target.data + self.momentum_tau * p.data
                )

    def compute_infonce_loss(
        self,
        states: torch.Tensor,
        next_states: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """
        Compute InfoNCE contrastive auxiliary loss.
        Positive query-key pair:
          - Query: anchor state s_t passed through online encoder
          - Key: next state s_{t+1} perturbed with stochastic Gaussian jitter,
                 passed through target momentum encoder
        Negative keys: all other transitions in the batch.

        Returns
        -------
        loss : scalar tensor InfoNCE loss
        accuracy : float top-1 contrastive prediction accuracy
        """
        B = states.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=states.device, requires_grad=True), 1.0

        # Query projection: q = g_psi(f_theta(s_t))
        z_query = self.encoder(states)
        q = self.projector(z_query)   # (B, proj_dim)

        # Target key projection: k = g_psi_bar(f_theta_bar(s_{t+1} + noise))
        noise = torch.randn_like(next_states) * self.noise_std
        augmented_next_states = next_states + noise
        with torch.no_grad():
            z_key = self.target_encoder(augmented_next_states)
            k = self.target_projector(z_key)   # (B, proj_dim)

        # Bilinear logits: q_i^T W k_j
        # q W: (B, proj_dim)
        q_W = torch.matmul(q, self.W)   # (B, proj_dim)
        # logits: (B, B)
        logits = torch.matmul(q_W, k.t()) / self.temperature

        # Labels: positive pairs are on the diagonal (i == j)
        labels = torch.arange(B, device=states.device, dtype=torch.long)

        # Symmetric cross-entropy InfoNCE loss
        loss = F.cross_entropy(logits, labels)

        # Contrastive Top-1 Accuracy
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            accuracy = (preds == labels).float().mean().item()

        return loss, accuracy


class LatentLinearProbe(nn.Module):
    """
    Linear Probe for evaluating state disentanglement and semantic representational quality.
    Evaluates whether frozen latent representations linearly encode ground truth physical states.
    """
    def __init__(self, latent_dim: int, target_dim: int):
        super().__init__()
        self.probe = nn.Linear(latent_dim, target_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.probe(z)

    def fit_and_evaluate(
        self,
        latent_features: np.ndarray,
        ground_truth_targets: np.ndarray,
        lr: float = 1e-2,
        epochs: int = 100,
    ) -> float:
        """Fit linear probe on frozen features and compute R^2 score."""
        X = torch.tensor(latent_features, dtype=torch.float32)
        Y = torch.tensor(ground_truth_targets, dtype=torch.float32)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            preds = self(X)
            loss = criterion(preds, Y)
            loss.backward()
            optimizer.step()

        self.eval()
        with torch.no_grad():
            preds = self(X).numpy()
            targets = Y.numpy()
            ss_tot = np.sum((targets - targets.mean(axis=0)) ** 2)
            ss_res = np.sum((targets - preds) ** 2)
            r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))
        return max(-1.0, min(1.0, r2))
