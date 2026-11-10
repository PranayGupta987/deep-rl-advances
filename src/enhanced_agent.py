"""
Enhanced DQN Agent combining three major improvements over vanilla DQN:
  1. Double DQN (van Hasselt et al., 2016) — mitigates Q-value overestimation
     by decoupling action selection (online net) from action evaluation (target net).
  2. Prioritized Experience Replay (Schaul et al., 2016) — samples informative
     transitions with high TD-error.
  3. Soft Target Updates (Polyak averaging) — smooth target updates instead of
     infrequent hard copies.
  4. Huber Loss — robust regression against gradient outliers.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.network import QNetwork
from src.replay_buffer import PrioritizedReplayBuffer


class EnhancedDQNAgent:
    """
    Enhanced Q-Learning agent with Double DQN, PER, Soft Updates, and Huber Loss.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        tau: float = 0.005,         # Polyak update rate for soft target update
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_decay_steps: int = 10_000,
        hidden_dim: int = 128,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = (eps_start - eps_end) / eps_decay_steps

        self.device = torch.device("cpu")

        # Networks
        self.q_net = QNetwork(state_dim, n_actions, hidden_dim).to(self.device)
        self.target_net = copy.deepcopy(self.q_net)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss(reduction='none')  # Huber loss with per-element return

        self.buffer = PrioritizedReplayBuffer(
            capacity=buffer_size,
            alpha=per_alpha,
            beta_start=per_beta_start,
            beta_anneal_steps=10_000,
        )
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

        states, actions, rewards, next_states, dones, indices, weights = self.buffer.sample(self.batch_size)

        states_t      = torch.tensor(states,      device=self.device)
        actions_t     = torch.tensor(actions,     device=self.device)
        rewards_t     = torch.tensor(rewards,     device=self.device)
        next_states_t = torch.tensor(next_states, device=self.device)
        dones_t       = torch.tensor(dones,       device=self.device)
        weights_t     = torch.tensor(weights,     device=self.device)

        # Current Q(s,a)
        q_current = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # --------------------------------------------------------------
        # Double DQN target computation:
        #   1. Select best action using online network: a* = argmax_a Q(s', a)
        #   2. Evaluate action using target network: Q_target(s', a*)
        # --------------------------------------------------------------
        with torch.no_grad():
            best_actions = self.q_net(next_states_t).argmax(dim=1, keepdim=True)
            q_next = self.target_net(next_states_t).gather(1, best_actions).squeeze(1)
            q_target = rewards_t + self.gamma * q_next * (1.0 - dones_t)

        # Element-wise Huber loss for PER priority updates
        elementwise_loss = self.loss_fn(q_current, q_target)
        loss = (elementwise_loss * weights_t).mean()

        # Update PER priorities based on absolute TD errors
        with torch.no_grad():
            td_errors = torch.abs(q_current - q_target).cpu().numpy()
        self.buffer.update_priorities(indices, td_errors)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.step_count += 1
        self.eps = max(self.eps_end, self.eps - self.eps_decay)

        # Soft target update (Polyak averaging)
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return loss.item()
