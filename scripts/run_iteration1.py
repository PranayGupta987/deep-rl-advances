"""
Multi-Seed Benchmark Training Infrastructure for Iteration 1 Research.

Runs QR-DQN alongside Vanilla DQN and Enhanced DQN across N seeds, producing
statistically rigorous results suitable for international conference reporting.

Features:
  - Deterministic seeding (NumPy + PyTorch + Python random)
  - Multi-seed (N=5) parallel-compatible sequential training
  - Per-seed and aggregated JSON output
  - Progress logging with wall-clock timing

Usage:
    python run_iteration1.py
"""

import sys
import os
import time
import json
import copy
import random
from typing import Dict, List, Any

# Ensure stdout handles UTF-8 gracefully on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# -- Ensure src is importable ------------------------------------------- #
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.qr_dqn_agent import QRDQNAgent


# =========================================================================
#  Inline Vanilla DQN and Enhanced DQN (from existing run_training.py)
#  --- keeps this script fully self-contained for clean reproducibility ---
# =========================================================================

class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, n_actions),
        )
    def forward(self, x):
        return self.net(x)


class UniformReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)
    def push(self, *a): self.buf.append(a)
    def sample(self, bs):
        b = random.sample(self.buf, bs)
        s, a, r, ns, d = zip(*b)
        return (np.array(s, np.float32), np.array(a, np.int64),
                np.array(r, np.float32), np.array(ns, np.float32),
                np.array(d, np.float32), None, None)
    def update_priorities(self, *a): pass
    def __len__(self): return len(self.buf)


class _SumTree:
    def __init__(self, cap):
        self.cap = cap
        self.tree = np.zeros(2 * cap, np.float64)
        self.data = [None] * cap
        self._p = 0
    def _prop(self, i, d):
        while i > 1: i >>= 1; self.tree[i] += d
    def update(self, i, p):
        d = p - self.tree[i]; self.tree[i] = p; self._prop(i, d)
    def add(self, p, data):
        i = self._p + self.cap
        self.data[self._p] = data; self.update(i, p)
        self._p = (self._p + 1) % self.cap
    def get(self, q):
        i = 1
        while i < self.cap:
            l = 2 * i
            if q <= self.tree[l]: i = l
            else: q -= self.tree[l]; i = l + 1
        return i, self.tree[i], self.data[i - self.cap]
    @property
    def total(self): return float(self.tree[1])
    @property
    def max_p(self): return float(self.tree[self.cap:].max())


class _PERBuffer:
    def __init__(self, cap, alpha=0.6, beta0=0.4, beta1=1.0, anneal=10_000, eps=1e-6):
        self.tree = _SumTree(cap); self.cap = cap; self.alpha = alpha
        self.b0 = beta0; self.b1 = beta1; self.anneal = anneal
        self.eps = eps; self._step = 0; self._size = 0
    def _beta(self):
        t = min(self._step / self.anneal, 1.0)
        return self.b0 + t * (self.b1 - self.b0)
    def push(self, *a):
        mp = self.tree.max_p
        self.tree.add(mp if mp > 0 else 1.0, a)
        self._size = min(self._size + 1, self.cap)
    def sample(self, bs):
        self._step += 1; beta = self._beta(); total = self.tree.total
        seg = total / bs
        idxs, pris, batch = [], [], []
        for i in range(bs):
            q = random.uniform(seg * i, seg * (i + 1))
            idx, p, data = self.tree.get(q)
            idxs.append(idx); pris.append(p); batch.append(data)
        probs = np.array(pris) / total
        w = (self._size * probs) ** (-beta); w /= w.max()
        s, a, r, ns, d = zip(*batch)
        return (np.array(s, np.float32), np.array(a, np.int64),
                np.array(r, np.float32), np.array(ns, np.float32),
                np.array(d, np.float32), idxs, np.array(w, np.float32))
    def update_priorities(self, idxs, errs):
        for i, e in zip(idxs, errs):
            self.tree.update(i, (abs(float(e)) + self.eps) ** self.alpha)
    def __len__(self): return self._size


class DQNAgent:
    def __init__(self, sd, na, lr=1e-3, gamma=0.99, buf=50000, bs=64, tuf=200, hd=128):
        self.na = na; self.gamma = gamma; self.bs = bs; self.tuf = tuf
        self.eps = 1.0; self.eps_end = 0.01; self.eps_decay = 0.99
        self.dev = torch.device("cpu")
        self.q = QNetwork(sd, na, hd).to(self.dev)
        self.tq = copy.deepcopy(self.q); self.tq.eval()
        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.buf = UniformReplayBuffer(buf)
        self.updates = 0
    def act(self, s):
        if np.random.rand() < self.eps: return np.random.randint(self.na)
        with torch.no_grad():
            return int(self.q(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).argmax())
    def store(self, *a): self.buf.push(*a)
    def decay_eps(self): self.eps = max(self.eps_end, self.eps * self.eps_decay)
    def update(self):
        if len(self.buf) < self.bs: return None
        s, a, r, ns, d, _, _ = self.buf.sample(self.bs)
        s = torch.tensor(s); a = torch.tensor(a); r = torch.tensor(r)
        ns = torch.tensor(ns); d = torch.tensor(d)
        qc = self.q(s).gather(1, a.unsqueeze(1)).squeeze()
        with torch.no_grad():
            qt = r + self.gamma * self.tq(ns).max(1).values * (1 - d)
        loss = nn.MSELoss()(qc, qt)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0); self.opt.step()
        self.updates += 1
        if self.updates % self.tuf == 0: self.tq.load_state_dict(self.q.state_dict())
        return {"loss": loss.item(), "mean_q": qc.mean().item(), "quantile_spread": 0.0}


class EnhancedDQNAgent:
    def __init__(self, sd, na, lr=1e-3, gamma=0.99, buf=50000, bs=64, tau=0.005, hd=128):
        self.na = na; self.gamma = gamma; self.bs = bs; self.tau = tau
        self.eps = 1.0; self.eps_end = 0.01; self.eps_decay = 0.99
        self.dev = torch.device("cpu")
        self.q = QNetwork(sd, na, hd).to(self.dev)
        self.tq = copy.deepcopy(self.q); self.tq.eval()
        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.buf = _PERBuffer(buf)
    def act(self, s):
        if np.random.rand() < self.eps: return np.random.randint(self.na)
        with torch.no_grad():
            return int(self.q(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).argmax())
    def store(self, *a): self.buf.push(*a)
    def decay_eps(self): self.eps = max(self.eps_end, self.eps * self.eps_decay)
    def update(self):
        if len(self.buf) < self.bs: return None
        s, a, r, ns, d, idxs, w = self.buf.sample(self.bs)
        s = torch.tensor(s); a = torch.tensor(a); r = torch.tensor(r)
        ns = torch.tensor(ns); d = torch.tensor(d); w = torch.tensor(w)
        qc = self.q(s).gather(1, a.unsqueeze(1)).squeeze()
        with torch.no_grad():
            ba = self.q(ns).argmax(1, keepdim=True)
            qt = r + self.gamma * self.tq(ns).gather(1, ba).squeeze() * (1 - d)
        el = nn.SmoothL1Loss(reduction='none')(qc, qt)
        loss = (el * w).mean()
        with torch.no_grad(): tde = (qc - qt).abs().cpu().numpy()
        self.buf.update_priorities(idxs, tde)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0); self.opt.step()
        for tp, p in zip(self.tq.parameters(), self.q.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        return {"loss": loss.item(), "mean_q": qc.mean().item(), "quantile_spread": 0.0}


# =========================================================================
#  Training loop
# =========================================================================

def set_global_seed(seed: int) -> None:
    """Fully deterministic seeding across NumPy, Python random, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Encourage determinism in cuDNN (no-op if not using GPU)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_qrdqn_agent(state_dim: int, n_actions: int, risk_param: float = 0.0) -> QRDQNAgent:
    """Construct a QR-DQN agent with standard hyperparameters for this benchmark."""
    return QRDQNAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        n_quantiles=51,
        lr=1e-3,
        gamma=0.99,
        buffer_size=50_000,
        batch_size=64,
        tau=0.005,
        kappa=1.0,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay_steps=10_000,
        hidden_dim=128,
        per_alpha=0.6,
        per_beta_start=0.4,
        risk_param=risk_param,
    )


def run_single_seed(
    agent_factory,
    env_name: str,
    seed: int,
    n_episodes: int = 300,
    max_steps: int = 500,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train an agent for n_episodes on env_name with a specific random seed.

    Returns
    -------
    dict with keys:
        'rewards'         : List[float]  - per-episode return
        'losses'          : List[float]  - per-episode mean loss
        'steps'           : List[int]    - per-episode step count
        'mean_q_history'  : List[float]  - per-episode mean Q-value
        'spread_history'  : List[float]  - per-episode quantile spread (0 for non-QR agents)
        'time'            : float        - total wall-clock training time (seconds)
        'seed'            : int
    """
    set_global_seed(seed)
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = agent_factory(state_dim, n_actions)

    rewards, losses, steps_list = [], [], []
    mean_q_history, spread_history = [], []
    t0 = time.time()
    env_rng = np.random.default_rng(seed)

    for ep in range(n_episodes):
        s, _ = env.reset(seed=int(env_rng.integers(1_000_000)))
        ep_r, ep_losses, ep_mean_q, ep_spread, ep_s = 0.0, [], [], [], 0

        for _ in range(max_steps):
            a = agent.act(s) if hasattr(agent, "act") else agent.select_action(s)
            ns, r, term, trunc, _ = env.step(a)
            done = term or trunc
            agent.store(s, a, r, ns, float(done))

            info = agent.update()
            if info is not None:
                ep_losses.append(info["loss"])
                ep_mean_q.append(info["mean_q"])
                ep_spread.append(info["quantile_spread"])

            s = ns
            ep_r += r
            ep_s += 1
            if done:
                break

        # End-of-episode epsilon decay
        if hasattr(agent, "decay_eps"):
            agent.decay_eps()
        elif hasattr(agent, "eps"):
            agent.eps = max(agent.eps_end, agent.eps - agent.eps_decay)

        rewards.append(ep_r)
        losses.append(float(np.mean(ep_losses)) if ep_losses else 0.0)
        steps_list.append(ep_s)
        mean_q_history.append(float(np.mean(ep_mean_q)) if ep_mean_q else 0.0)
        spread_history.append(float(np.mean(ep_spread)) if ep_spread else 0.0)

        if verbose and (ep + 1) % 50 == 0:
            avg = np.mean(rewards[-50:])
            sys.stdout.write(
                f"    Seed={seed:4d} | Ep {ep+1:3d}/{n_episodes} | "
                f"Avg50={avg:8.2f} | eps={getattr(agent, 'eps', 0):.4f}\n"
            )
            sys.stdout.flush()

    env.close()
    return {
        "rewards": rewards,
        "losses": losses,
        "steps": steps_list,
        "mean_q_history": mean_q_history,
        "spread_history": spread_history,
        "time": time.time() - t0,
        "seed": seed,
    }


def run_multi_seed(
    agent_factory,
    agent_name: str,
    env_name: str,
    seeds: List[int],
    n_episodes: int = 300,
    max_steps: int = 500,
) -> Dict[str, Any]:
    """
    Run an agent across multiple seeds and return all per-seed results.
    """
    print(f"\n  +-- Agent: {agent_name} | Env: {env_name}")
    all_runs = []
    for seed in seeds:
        print(f"  |  [seed={seed}]", flush=True)
        result = run_single_seed(agent_factory, env_name, seed, n_episodes, max_steps)
        all_runs.append(result)
        avg = np.mean(result["rewards"][-50:])
        print(f"  |  -> Final-50-avg = {avg:.2f}  ({result['time']:.1f}s)\n", flush=True)

    # Stack rewards arrays: shape (n_seeds, n_episodes)
    reward_matrix = np.array([r["rewards"] for r in all_runs])
    loss_matrix = np.array([r["losses"] for r in all_runs])

    return {
        "runs": all_runs,
        "reward_matrix": reward_matrix.tolist(),
        "loss_matrix": loss_matrix.tolist(),
        "agent": agent_name,
        "env": env_name,
        "seeds": seeds,
    }


# =========================================================================
#  Main entry point
# =========================================================================

SEEDS = [42, 100, 2013, 2024, 777]
ENVS  = ["CartPole-v1", "Acrobot-v1"]
N_EP  = 300

# Agent factories keyed by name
AGENT_FACTORIES = {
    "vanilla":   lambda sd, na: DQNAgent(sd, na),
    "enhanced":  lambda sd, na: EnhancedDQNAgent(sd, na),
    "qr_dqn":    lambda sd, na: make_qrdqn_agent(sd, na, risk_param=0.0),
    # Risk-sensitive variants of QR-DQN (evaluated at action-selection time)
    "qr_dqn_risk_averse":  lambda sd, na: make_qrdqn_agent(sd, na, risk_param=0.75),
    "qr_dqn_risk_seeking": lambda sd, na: make_qrdqn_agent(sd, na, risk_param=-0.75),
}


if __name__ == "__main__":
    os.makedirs("results/iteration1", exist_ok=True)
    all_results = {}

    for env_name in ENVS:
        print(f"\n{'-'*60}")
        print(f"  Environment: {env_name}")
        print(f"{'-'*60}")
        all_results[env_name] = {}

        for agent_name, factory in AGENT_FACTORIES.items():
            result = run_multi_seed(
                factory, agent_name, env_name,
                seeds=SEEDS, n_episodes=N_EP,
            )
            all_results[env_name][agent_name] = result

    # -- Serialise all results to JSON -------------------------------- #
    def to_serialisable(obj):
        if isinstance(obj, np.ndarray):   return obj.tolist()
        if isinstance(obj, np.integer):   return int(obj)
        if isinstance(obj, np.floating):  return float(obj)
        if isinstance(obj, dict):         return {k: to_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, list):         return [to_serialisable(v) for v in obj]
        return obj

    out_path = "results/iteration1/multi_seed_results.json"
    with open(out_path, "w") as f:
        json.dump(to_serialisable(all_results), f, indent=2)

    print(f"\n[OK] All results saved to {out_path}")
    print("  Run  python analyze_iteration1.py  to compute statistics and generate plots.")
