"""
Online Meta-Gradient Hyperparameter Optimizer (Iteration 3).

Grounding Literature:
  - Xu et al., "Meta-Gradient Reinforcement Learning", NeurIPS 2018.
  - MacKay et al., "Self-Tuning Networks: Bilevel Optimization of Hyperparameters using Meta-Gradients", ICLR 2019.
  - Zahavy et al., "Discovering a set of policies for the worst case reward", ICLR 2021.

Mathematical Formulation:
  Inner objective:
      theta' = theta - alpha * nabla_theta L_inner(theta; beta)
      where L_inner incorporates the epistemic intrinsic exploration bonus:
          r^+ = r + beta * sigma_epistemic(s, a)
  Outer objective (validation return / TD error):
      L_outer(theta') evaluated on an independent validation batch B_val.
  Meta-gradient w.r.t exploration weight beta:
      nabla_beta L_outer(theta') = (nabla_theta' L_outer(theta'))^T * (d theta' / d beta)
  Meta-update:
      beta <- clip(beta - eta_meta * nabla_beta L_outer, beta_min, beta_max)
"""

import torch
import torch.nn as nn
import numpy as np

torch.set_num_threads(1)


class MetaGradientHyperparameterOptimizer:
    """
    Online meta-gradient optimizer that dynamically tunes the exploration bonus weight
    beta to maximize outer validation Bellman consistency.
    """
    def __init__(
        self,
        initial_beta: float = 0.05,
        meta_lr: float = 1e-4,
        beta_min: float = 0.001,
        beta_max: float = 0.5,
    ):
        self.meta_lr = meta_lr
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.device = torch.device("cpu")

        # Unconstrained log-parameterization to ensure positivity: beta = softplus(log_beta)
        self.log_beta = nn.Parameter(
            torch.tensor(np.log(np.exp(initial_beta) - 1.0), dtype=torch.float32, device=self.device)
        )
        self.meta_optimizer = torch.optim.Adam([self.log_beta], lr=meta_lr)
        self.beta_history = [initial_beta]

    @property
    def beta(self) -> float:
        with torch.no_grad():
            b = float(torch.nn.functional.softplus(self.log_beta).item())
            return float(np.clip(b, self.beta_min, self.beta_max))

    def step_meta_gradient(
        self,
        td_error_train: torch.Tensor,
        td_error_val: torch.Tensor,
        uncertainty_val: torch.Tensor,
    ) -> float:
        """
        Approximate implicit meta-gradient step using outer validation TD-error correlation
        with epistemic uncertainty.

        If states with higher epistemic uncertainty lead to larger outer validation improvements,
        the meta-gradient increases beta; otherwise it decreases beta.
        """
        # Outer validation loss
        val_loss = td_error_val.mean()

        # Compute surrogate covariance between validation TD error and epistemic uncertainty
        with torch.no_grad():
            centered_td = td_error_val - td_error_val.mean()
            centered_unc = uncertainty_val - uncertainty_val.mean()
            cov = (centered_td * centered_unc).mean()

        # Meta-gradient signal: if uncertainty correlates with remaining validation error, increase exploration
        surrogate_loss = -self.log_beta * cov.detach()

        self.meta_optimizer.zero_grad()
        surrogate_loss.backward()
        self.meta_optimizer.step()

        cur_beta = self.beta
        self.beta_history.append(cur_beta)
        return cur_beta
