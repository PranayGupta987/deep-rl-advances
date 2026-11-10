"""
Neural network Q-function (shared by DQN and Enhanced DQN).

A simple multi-layer perceptron is used because all environments are
low-dimensional (≤ 6-D state spaces).  This is consistent with the
original DQN paper's linear/MLP variant evaluated on classic-control tasks.
"""
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Three-hidden-layer MLP approximating Q(s, a) for all actions at once.

    Architecture
    ------------
    state_dim → Linear(128) → ReLU → Linear(128) → ReLU → Linear(64) → ReLU → Linear(n_actions)

    Parameters
    ----------
    state_dim  : int  – dimensionality of the state space.
    n_actions  : int  – number of discrete actions.
    hidden_dim : int  – width of each hidden layer (default 128).
    """

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
