"""
Vanilla DQN agent — faithful implementation of:
  Mnih et al., "Playing Atari with Deep Reinforcement Learning", 2013.
  https://arxiv.org/abs/1312.5602

Key design choices matched to the paper:
  * Experience replay with uniform sampling.
  * Separate target network updated (hard copy) every C steps.
  * ε-greedy exploration with linear ε decay.
  * MSE loss on Bellman residuals.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.network import QNetwork
from src.replay_buffer import UniformReplayBuffer


class DQNAgent:
    """
    Parameters
    ----------
    state_dim          : int   – env observation dimension.
    n_actions          : int   – env discrete action count.
    lr                 : float – Adam learning rate.
    gamma              : float – discount factor.
    buffer_size        : int   – replay buffer capacity.
    batch_size         : int   – mini-batch size.
    target_update_freq : int   – steps between hard target-network copies.
    eps_start          : float – initial ε for ε-greedy.
    eps_end            : float – final ε.
    eps_decay_steps    : int   – steps to linearly decay ε from start to end.
    hidden_dim         : int   – Q-network hidden layer width.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        target_update_freq: int = 300,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_decay_steps: int = 10_000,
        hidden_dim: int = 128,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = (eps_start - eps_end) / eps_decay_steps

        self.device = torch.device("cpu")

        # Networks
        self.q_net     = QNetwork(state_dim, n_actions, hidden_dim).to(self.device)
        self.target_net = copy.deepcopy(self.q_net)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn   = nn.MSELoss()

        self.buffer = UniformReplayBuffer(buffer_size)
        self.step_count = 0

    # ------------------------------------------------------------------
    def select_action(self, state: np.ndarray) -> int:
        if np.random.rand() < self.eps:
            return np.random.randint(self.n_actions)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.q_net(state_t)
        return int(q_vals.argmax(dim=1).item())

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    # ------------------------------------------------------------------
    def update(self):
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones, _, _ = self.buffer.sample(self.batch_size)

        states_t      = torch.tensor(states,      device=self.device)
        actions_t     = torch.tensor(actions,     device=self.device)
        rewards_t     = torch.tensor(rewards,     device=self.device)
        next_states_t = torch.tensor(next_states, device=self.device)
        dones_t       = torch.tensor(dones,       device=self.device)

        # Current Q(s,a)
        q_current = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target: r + γ * max_a' Q_target(s', a')   — Hard Bootstrap (original DQN)
        with torch.no_grad():
            q_next = self.target_net(next_states_t).max(dim=1).values
            q_target = rewards_t + self.gamma * q_next * (1.0 - dones_t)

        loss = self.loss_fn(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)  # gradient clipping
        self.optimizer.step()

        self.step_count += 1

        # ε decay
        self.eps = max(self.eps_end, self.eps - self.eps_decay)

        # Hard target update every C steps
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()
