"""DQN agents (PyTorch): independent (one network per intersection) and shared-weight (parameter sharing)."""
import numpy as np
import torch, torch.nn as nn
from neighbors import augment, extra_dim
torch.set_num_threads(1)


class Replay:
    def __init__(self, obs_dim, cap=30000):
        self.o = np.zeros((cap, obs_dim), np.float32); self.o2 = np.zeros((cap, obs_dim), np.float32)
        self.a = np.zeros(cap, np.int64); self.r = np.zeros(cap, np.float32); self.d = np.zeros(cap, np.float32)
        self.cap, self.n, self.i = cap, 0, 0

    def add_many(self, o, a, r, o2, d):
        for k in range(len(a)):
            self.o[self.i], self.a[self.i], self.r[self.i], self.o2[self.i], self.d[self.i] = o[k], a[k], r[k], o2[k], d
            self.i = (self.i + 1) % self.cap; self.n = min(self.n + 1, self.cap)

    def sample(self, b):
        idx = np.random.randint(0, self.n, b)
        return (torch.from_numpy(self.o[idx]), torch.from_numpy(self.a[idx]), torch.from_numpy(self.r[idx]),
                torch.from_numpy(self.o2[idx]), torch.from_numpy(self.d[idx]))


class DQN:
    """Double DQN with target network, Huber loss and gradient clipping."""

    def __init__(self, obs_dim, n_act, lr=1e-3, gamma=0.9, batch=64, target_every=100, hidden=64, min_buf=500):
        mk = lambda: nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_act))
        self.q, self.qt = mk(), mk()
        self.qt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buf = Replay(obs_dim)
        self.n_act, self.gamma, self.batch, self.target_every, self.min_buf, self.steps = n_act, gamma, batch, target_every, min_buf, 0

    def act(self, obs, eps):
        obs = np.atleast_2d(obs)
        if np.random.rand() < eps:
            return np.random.randint(0, self.n_act, len(obs))
        with torch.no_grad():
            return self.q(torch.from_numpy(obs.astype(np.float32))).argmax(1).numpy()

    def learn(self):
        if self.buf.n < self.min_buf:
            return
        o, a, r, o2, d = self.buf.sample(self.batch)
        with torch.no_grad():
            a2 = self.q(o2).argmax(1, keepdim=True)
            y = r + self.gamma * (1 - d) * self.qt(o2).gather(1, a2).squeeze(1)
        loss = nn.functional.smooth_l1_loss(self.q(o).gather(1, a.unsqueeze(1)).squeeze(1), y)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        self.opt.step()
        self.steps += 1
        if self.steps % self.target_every == 0:
            self.qt.load_state_dict(self.q.state_dict())


class IndependentDQN:
    """No coordination: every intersection owns a separate network and learns from its own obs/reward."""
    def __init__(self, env):
        self.ag = env.agents
        self.nets = {a: DQN(env.obs_dim, env.n_act) for a in self.ag}

    def act(self, obs, eps):
        return {a: int(self.nets[a].act(obs[a], eps)[0]) for a in self.ag}

    def store(self, o, act, r, o2, done):
        for a in self.ag:
            self.nets[a].buf.add_many([o[a]], [act[a]], [r[a]], [o2[a]], float(done))

    def learn(self):
        for a in self.ag:
            self.nets[a].learn()

    def state_dict(self): return {a: n.q.state_dict() for a, n in self.nets.items()}

    def load(self, sd):
        for a, n in self.nets.items(): n.q.load_state_dict(sd[a])


class SharedDQN:
    """Parameter sharing: one network for all intersections, trained on pooled experience.
    `level` (0/1/2) = how much neighbour information is appended to the observation (see neighbors.py).
    act/store take the RAW level-0 observations from the env; the augmentation happens inside the agent,
    so the evaluation function can use this agent without knowing about communication."""
    def __init__(self, env, level=0, batch=128):
        self.ag, self.nbrs, self.level = env.agents, env.neighbors, level
        self.obs_dim = env.obs_dim + extra_dim(level)
        self.net = DQN(self.obs_dim, env.n_act, batch=batch)

    def act(self, obs, eps):
        obs = augment(obs, self.nbrs, self.level)
        M = np.stack([obs[a] for a in self.ag])
        acts = self.net.act(M, -1)
        if eps > 0:
            rnd = np.random.rand(len(self.ag)) < eps
            acts = np.where(rnd, np.random.randint(0, self.net.n_act, len(self.ag)), acts)
        return {a: int(acts[i]) for i, a in enumerate(self.ag)}

    def store(self, o, act, r, o2, done):
        o, o2 = augment(o, self.nbrs, self.level), augment(o2, self.nbrs, self.level)
        self.net.buf.add_many([o[a] for a in self.ag], [act[a] for a in self.ag], [r[a] for a in self.ag],
                              [o2[a] for a in self.ag], float(done))

    def learn(self): self.net.learn()

    def state_dict(self): return self.net.q.state_dict()

    def load(self, sd): self.net.q.load_state_dict(sd)


def mix_rewards(env, r, beta):
    """Cooperative reward shaping: own reward + beta * mean reward of neighbours (penalises pushing queues onto neighbours)."""
    if beta == 0:
        return r
    return {a: r[a] + beta * float(np.mean([r[n] for n in env.neighbors[a] if n])) for a in env.agents}