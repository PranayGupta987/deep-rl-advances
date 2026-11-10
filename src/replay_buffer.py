"""
Replay Buffer implementations for DQN experiments.

Provides:
  - UniformReplayBuffer  : standard i.i.d. experience replay (original DQN)
  - PrioritizedReplayBuffer : Proportional Prioritized Experience Replay (PER)
    as in Schaul et al., 2016 (https://arxiv.org/abs/1511.05952)

The PER buffer uses a sum-segment tree for O(log n) sampling and update.
"""
import numpy as np
import random
from collections import deque


# ---------------------------------------------------------------------------
# Uniform replay buffer (original DQN, Mnih et al. 2013)
# ---------------------------------------------------------------------------
class UniformReplayBuffer:
    """Fixed-size circular buffer that samples uniformly at random."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            None,   # indices  (not used for uniform)
            None,   # weights  (not used for uniform)
        )

    def update_priorities(self, indices, priorities):
        pass  # no-op for uniform buffer

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Sum-segment tree (internal helper for PER)
# ---------------------------------------------------------------------------
class _SumTree:
    """Binary segment tree that supports prefix-sum queries and point updates."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float64)   # tree[1] = total sum
        self.data = [None] * capacity
        self._ptr = 0

    # ------------------------------------------------------------------
    def _propagate(self, idx: int, delta: float):
        while idx > 1:
            idx >>= 1
            self.tree[idx] += delta

    def update(self, tree_idx: int, priority: float):
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, delta)

    def add(self, priority: float, data):
        tree_idx = self._ptr + self.capacity
        self.data[self._ptr] = data
        self.update(tree_idx, priority)
        self._ptr = (self._ptr + 1) % self.capacity

    def get(self, query: float):
        """Return (tree_idx, priority, data) for cumulative-sum query."""
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            if query <= self.tree[left]:
                idx = left
            else:
                query -= self.tree[left]
                idx = left + 1
        data_idx = idx - self.capacity
        return idx, self.tree[idx], self.data[data_idx]

    @property
    def total(self) -> float:
        return float(self.tree[1])

    @property
    def max_priority(self) -> float:
        return float(self.tree[self.capacity:].max())


# ---------------------------------------------------------------------------
# Prioritized Experience Replay buffer
# ---------------------------------------------------------------------------
class PrioritizedReplayBuffer:
    """
    Proportional PER (Schaul et al., 2016).

    Transitions are sampled with probability proportional to |TD-error|^alpha.
    Importance-sampling weights correct for the bias introduced by non-uniform
    sampling.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store.
    alpha : float
        How much prioritization to use (0 = uniform, 1 = full).
    beta_start : float
        Starting IS-weight exponent (annealed toward 1 over training).
    beta_end : float
        Final IS-weight exponent.
    beta_anneal_steps : int
        Number of steps over which beta is annealed from beta_start to beta_end.
    epsilon : float
        Small constant added to priorities to ensure every transition is sampled
        with non-zero probability.
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_anneal_steps: int = 100_000,
        epsilon: float = 1e-6,
    ):
        self.tree = _SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_anneal_steps = beta_anneal_steps
        self.epsilon = epsilon
        self._step = 0   # for beta annealing
        self._size = 0

    # ------------------------------------------------------------------
    def _beta(self) -> float:
        t = min(self._step / self.beta_anneal_steps, 1.0)
        return self.beta_start + t * (self.beta_end - self.beta_start)

    def push(self, state, action, reward, next_state, done):
        # New transitions get the current max priority so they are replayed once.
        max_p = self.tree.max_priority
        if max_p == 0:
            max_p = 1.0
        self.tree.add(max_p, (state, action, reward, next_state, done))
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int):
        self._step += 1
        beta = self._beta()

        total = self.tree.total
        segment = total / batch_size

        tree_indices, priorities, batch = [], [], []
        for i in range(batch_size):
            lo, hi = segment * i, segment * (i + 1)
            query = random.uniform(lo, hi)
            idx, p, data = self.tree.get(query)
            tree_indices.append(idx)
            priorities.append(p)
            batch.append(data)

        # Importance-sampling weights
        probabilities = np.array(priorities, dtype=np.float64) / total
        weights = (self._size * probabilities) ** (-beta)
        weights /= weights.max()   # normalise so max weight = 1

        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
            tree_indices,
            np.array(weights,     dtype=np.float32),
        )

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            priority = (abs(float(err)) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)

    def __len__(self):
        return self._size
