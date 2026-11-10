"""
Standalone picklable worker module for multiprocessing on Windows.
"""
import random
import time
from typing import Dict, Any, Tuple
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
torch.set_num_threads(1)
from collections import deque
import copy

from src.qr_dqn_agent import QRDQNAgent
from src.contrastive_qr_dqn_agent import ContrastiveQRDQNAgent
from src.ensemble_agent import EnsembleBootstrappedAgent
from src.unified_agent import FullUnifiedAgent


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


def execute_seed_task(task_args: Tuple) -> Dict[str, Any]:
    """Execute a single (agent, env, seed) training run."""
    agent_type, env_name, seed, n_episodes, max_steps = task_args

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    if agent_type == "vanilla":
        agent = DQNAgent(state_dim, n_actions)
    elif agent_type == "enhanced":
        agent = EnhancedDQNAgent(state_dim, n_actions)
    elif agent_type == "qr_dqn":
        agent = QRDQNAgent(
            state_dim=state_dim, n_actions=n_actions, n_quantiles=51,
            lr=1e-3, gamma=0.99, buffer_size=50_000, batch_size=64,
            tau=0.005, kappa=1.0, eps_start=1.0, eps_end=0.01,
            eps_decay_steps=10_000, hidden_dim=128, per_alpha=0.6,
            per_beta_start=0.4, risk_param=0.0
        )
    elif agent_type == "qr_dqn_risk_averse":
        agent = QRDQNAgent(
            state_dim=state_dim, n_actions=n_actions, n_quantiles=51,
            lr=1e-3, gamma=0.99, buffer_size=50_000, batch_size=64,
            tau=0.005, kappa=1.0, eps_start=1.0, eps_end=0.01,
            eps_decay_steps=10_000, hidden_dim=128, per_alpha=0.6,
            per_beta_start=0.4, risk_param=0.75
        )
    elif agent_type == "qr_dqn_risk_seeking":
        agent = QRDQNAgent(
            state_dim=state_dim, n_actions=n_actions, n_quantiles=51,
            lr=1e-3, gamma=0.99, buffer_size=50_000, batch_size=64,
            tau=0.005, kappa=1.0, eps_start=1.0, eps_end=0.01,
            eps_decay_steps=10_000, hidden_dim=128, per_alpha=0.6,
            per_beta_start=0.4, risk_param=-0.75
        )
    elif agent_type == "contrastive_qr_dqn":
        agent = ContrastiveQRDQNAgent(
            state_dim=state_dim, n_actions=n_actions, n_quantiles=51,
            latent_dim=128, proj_dim=64, lr=1e-3, gamma=0.99,
            buffer_size=50_000, batch_size=64, tau=0.005, kappa=1.0,
            eps_start=1.0, eps_end=0.01, eps_decay=0.99,
            contrastive_coeff=0.1
        )
    elif agent_type == "ensemble_dqn":
        agent = EnsembleBootstrappedAgent(
            state_dim=state_dim, n_actions=n_actions, num_heads=5,
            hidden_dim=128, lr=1e-3, gamma=0.99, buffer_size=50_000,
            batch_size=64, tau=0.005, beta_uncertainty=0.05, bootstrap_p=0.7,
            eps_start=1.0, eps_end=0.01, eps_decay=0.99
        )
    elif agent_type == "full_unified":
        agent = FullUnifiedAgent(
            state_dim=state_dim, n_actions=n_actions, n_quantiles=51,
            num_heads=3, latent_dim=128, proj_dim=64, lr=1e-3,
            gamma=0.99, buffer_size=50_000, batch_size=64, tau=0.005,
            kappa=1.0, contrastive_coeff=0.1, beta_uncertainty=0.05,
            eps_start=1.0, eps_end=0.01, eps_decay=0.99
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

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

        if hasattr(agent, "decay_eps"):
            agent.decay_eps()
        elif hasattr(agent, "eps"):
            agent.eps = max(agent.eps_end, agent.eps - agent.eps_decay)

        rewards.append(ep_r)
        losses.append(float(np.mean(ep_losses)) if ep_losses else 0.0)
        steps_list.append(ep_s)
        mean_q_history.append(float(np.mean(ep_mean_q)) if ep_mean_q else 0.0)
        spread_history.append(float(np.mean(ep_spread)) if ep_spread else 0.0)

    env.close()
    return {
        "agent": agent_type,
        "env": env_name,
        "seed": seed,
        "rewards": rewards,
        "losses": losses,
        "steps": steps_list,
        "mean_q_history": mean_q_history,
        "spread_history": spread_history,
        "time": time.time() - t0,
    }
