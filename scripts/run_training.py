"""
Optimized training script for RL Semester Project.
Compares Vanilla DQN vs Enhanced DQN (Double DQN + PER + Soft Updates)
on CartPole-v1 and Acrobot-v0.
"""
import sys, os, time, json, copy, random
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# ─────────────────────────────── network ─────────────────────────────── #
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

# ─────────────────────────── replay buffers ──────────────────────────── #
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
        self.tree = np.zeros(2*cap, np.float64)
        self.data = [None]*cap
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
            l = 2*i
            if q <= self.tree[l]: i = l
            else: q -= self.tree[l]; i = l+1
        return i, self.tree[i], self.data[i - self.cap]
    @property
    def total(self): return float(self.tree[1])
    @property
    def max_p(self): return float(self.tree[self.cap:].max())

class PrioritizedReplayBuffer:
    def __init__(self, cap, alpha=0.6, beta0=0.4, beta1=1.0, anneal=100000, eps=1e-6):
        self.tree = _SumTree(cap); self.cap = cap; self.alpha = alpha
        self.b0 = beta0; self.b1 = beta1; self.anneal = anneal
        self.eps = eps; self._step = 0; self._size = 0
    def _beta(self):
        t = min(self._step / self.anneal, 1.0)
        return self.b0 + t*(self.b1 - self.b0)
    def push(self, *a):
        mp = self.tree.max_p
        self.tree.add(mp if mp > 0 else 1.0, a)
        self._size = min(self._size+1, self.cap)
    def sample(self, bs):
        self._step += 1; beta = self._beta(); total = self.tree.total
        seg = total / bs
        idxs, pris, batch = [], [], []
        for i in range(bs):
            q = random.uniform(seg*i, seg*(i+1))
            idx, p, data = self.tree.get(q)
            idxs.append(idx); pris.append(p); batch.append(data)
        probs = np.array(pris)/total
        w = (self._size * probs)**(-beta); w /= w.max()
        s, a, r, ns, d = zip(*batch)
        return (np.array(s, np.float32), np.array(a, np.int64),
                np.array(r, np.float32), np.array(ns, np.float32),
                np.array(d, np.float32), idxs, np.array(w, np.float32))
    def update_priorities(self, idxs, errs):
        for i, e in zip(idxs, errs):
            self.tree.update(i, (abs(float(e))+self.eps)**self.alpha)
    def __len__(self): return self._size

# ──────────────────────────────── agents ─────────────────────────────── #
class DQNAgent:
    def __init__(self, sd, na, lr=1e-3, gamma=0.99, buf=50000, bs=64, tuf=200, hd=128):
        self.na = na; self.gamma = gamma; self.bs = bs; self.tuf = tuf
        self.eps = 1.0; self.eps_end = 0.01; self.eps_decay_rate = 0.99
        self.dev = torch.device("cpu")
        self.q = QNetwork(sd, na, hd).to(self.dev)
        self.tq = copy.deepcopy(self.q); self.tq.eval()
        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.buf = UniformReplayBuffer(buf)
        self.steps = 0; self.updates = 0
    def act(self, s):
        if np.random.rand() < self.eps: return np.random.randint(self.na)
        with torch.no_grad():
            return int(self.q(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).argmax())
    def store(self, *a): self.buf.push(*a)
    def decay_eps(self): self.eps = max(self.eps_end, self.eps * self.eps_decay_rate)
    def update(self):
        if len(self.buf) < self.bs: return None
        s, a, r, ns, d, _, _ = self.buf.sample(self.bs)
        s = torch.tensor(s); a = torch.tensor(a); r = torch.tensor(r)
        ns = torch.tensor(ns); d = torch.tensor(d)
        qc = self.q(s).gather(1, a.unsqueeze(1)).squeeze()
        with torch.no_grad():
            qt = r + self.gamma * self.tq(ns).max(1).values * (1-d)
        loss = nn.MSELoss()(qc, qt)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0); self.opt.step()
        self.updates += 1
        if self.updates % self.tuf == 0: self.tq.load_state_dict(self.q.state_dict())
        return loss.item()

class EnhancedDQNAgent:
    def __init__(self, sd, na, lr=1e-3, gamma=0.99, buf=50000, bs=64, tau=0.005, hd=128):
        self.na = na; self.gamma = gamma; self.bs = bs; self.tau = tau
        self.eps = 1.0; self.eps_end = 0.01; self.eps_decay_rate = 0.99
        self.dev = torch.device("cpu")
        self.q = QNetwork(sd, na, hd).to(self.dev)
        self.tq = copy.deepcopy(self.q); self.tq.eval()
        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.buf = PrioritizedReplayBuffer(buf)
        self.steps = 0
    def act(self, s):
        if np.random.rand() < self.eps: return np.random.randint(self.na)
        with torch.no_grad():
            return int(self.q(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).argmax())
    def store(self, *a): self.buf.push(*a)
    def decay_eps(self): self.eps = max(self.eps_end, self.eps * self.eps_decay_rate)
    def update(self):
        if len(self.buf) < self.bs: return None
        s, a, r, ns, d, idxs, w = self.buf.sample(self.bs)
        s = torch.tensor(s); a = torch.tensor(a); r = torch.tensor(r)
        ns = torch.tensor(ns); d = torch.tensor(d); w = torch.tensor(w)
        qc = self.q(s).gather(1, a.unsqueeze(1)).squeeze()
        with torch.no_grad():
            ba = self.q(ns).argmax(1, keepdim=True)
            qt = r + self.gamma * self.tq(ns).gather(1, ba).squeeze() * (1-d)
        el = nn.SmoothL1Loss(reduction='none')(qc, qt)
        loss = (el * w).mean()
        with torch.no_grad(): tde = (qc - qt).abs().cpu().numpy()
        self.buf.update_priorities(idxs, tde)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        self.steps += 1
        for tp, p in zip(self.tq.parameters(), self.q.parameters()):
            tp.data.copy_(self.tau*p.data + (1-self.tau)*tp.data)
        return loss.item()

# ───────────────────────── training loop ─────────────────────────────── #
def run(cls, env_name, n_ep=300, max_steps=500, seed=42, **kw):
    rng = np.random.default_rng(seed)
    env = gym.make(env_name)
    sd = env.observation_space.shape[0]; na = env.action_space.n
    agent = cls(sd, na, **kw)
    rewards, losses, steps_list = [], [], []
    t0 = time.time()
    for ep in range(n_ep):
        s, _ = env.reset(seed=int(rng.integers(1e6)))
        ep_r, ep_l, ep_s = 0.0, [], 0
        for _ in range(max_steps):
            a = agent.act(s)
            ns, r, term, trunc, _ = env.step(a)
            done = term or trunc
            agent.store(s, a, r, ns, float(done))
            lv = agent.update()
            if lv is not None: ep_l.append(lv)
            s = ns; ep_r += r; ep_s += 1
            if done: break
        agent.decay_eps()
        rewards.append(ep_r)
        losses.append(float(np.mean(ep_l)) if ep_l else 0.0)
        steps_list.append(ep_s)
        if (ep+1) % 50 == 0:
            lbl = "Vanilla" if cls is DQNAgent else "Enhanced"
            sys.stdout.write(f"  [{lbl:8s}] Ep {ep+1:3d}/{n_ep}  Avg50={np.mean(rewards[-50:]):8.1f}  eps={agent.eps:.4f}\n")
            sys.stdout.flush()
    env.close()
    return {"rewards": rewards, "losses": losses, "steps": steps_list, "time": time.time()-t0}

# ──────────────────────────── main ───────────────────────────────────── #
if __name__ == "__main__":
    ENVS = ["CartPole-v1", "Acrobot-v1"]
    N_EP = 300
    SEED = 42
    all_results = {}
    os.makedirs("results", exist_ok=True)

    for env_name in ENVS:
        sys.stdout.write(f"\n{'='*55}\n Environment: {env_name}\n{'='*55}\n"); sys.stdout.flush()
        all_results[env_name] = {}
        for cls, tag in [(DQNAgent, "vanilla"), (EnhancedDQNAgent, "enhanced")]:
            sys.stdout.write(f" Training {tag.upper()} DQN on {env_name}...\n"); sys.stdout.flush()
            res = run(cls, env_name, n_ep=N_EP, seed=SEED)
            all_results[env_name][tag] = res
            avg = np.mean(res["rewards"][-50:])
            sys.stdout.write(f" Done. Time={res['time']:.1f}s  Final-50-avg={avg:.1f}\n")
            sys.stdout.flush()

    # Serialize
    safe = {}
    for env, agents in all_results.items():
        safe[env] = {}
        for ag, r in agents.items():
            safe[env][ag] = {k: [float(x) for x in v] if isinstance(v, list) else float(v)
                             for k, v in r.items()}
    with open("results/metrics.json", "w") as f:
        json.dump(safe, f, indent=2)
    sys.stdout.write("\nAll metrics saved to results/metrics.json\n")
    sys.stdout.flush()
