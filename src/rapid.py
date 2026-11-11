"""RAPID-Light: shielded, anticipatory, pressure-informed multi-agent DQN for traffic signal control.

Works on any map made by osm_import.py, through env_map.SumoMapEnv (same 5 s decisions, 2 s yellow, same metrics).

Pieces (each one closes a loophole found in the baseline analysis, see docs/RAPID_light.md):
  Sensing      Each light sees only what a real roadside detector (radar/camera) sees: vehicles within DETECT_M metres
               of its stop line, following the road upstream but never past the previous signalised junction.
               Per vehicle: speed, distance to stop line, time stopped.                                  (knowledge K)
  Messages     Neighbouring lights exchange one number per approach: how many vehicles the upstream light is
               about to release towards us (its queued / near vehicles on movements that are green now and lead
               to us). Messages arrive one decision late (5 s) and can be lost (held at the last value). (belief B)
  State        One token per green phase: queue, approaching vehicles 0-50 m and 50-150 m, delay rate, exit-lane
               occupancy and queue, pressure, longest stop, time since last green, current-phase flag, time in
               phase, message. Any number of phases/lanes -> one shared network for every light of every map.
  Network      Phase-competition network: shared token encoder + self-attention between the phases of a light
               + pooled context -> one Q value per phase. Invalid (padded) phases are masked.
  Reward       r_a = -(sum over vehicles on the light's whole approaches of (1 - v/v_max)
                     + vehicles waiting to enter the map on entries that feed the light) / 10  [+ ALPHA * neighbours]
               The sum is the instantaneous delay rate: integrated over time it is exactly the time lost on the
               approaches, so the reward matches trip time / total travel time (not only stopped cars), and keeping
               vehicles out of the map is charged too (no gaming of in-network metrics). Training-only information
               (centralised training); the policy itself only sees the detectors and messages.
  Learner      Shared Double DQN, n-step returns, gamma 0.95, large replay. Warm start from Advanced-MP
               demonstrations (DQfD-style margin loss), guided exploration (random action = expert action half
               the time), demand randomised over x0.5 .. x2.0 during training.
  Shield       Runtime monitor of two temporal-logic properties, evaluated only on K (own detectors), never on B:
                 phi_1  G( demand_p and red_p for >= T_MAX  ->  green_p next )    bounded liveness, no starvation
                 phi_2  G( not (green_p and every exit lane of p is full) ) unless no phase can move   no gridlock
               The learner always sees the action that was actually executed.

  python src/rapid.py train maps/nyc16 --seed 0 --episodes 300
  python src/rapid.py eval  maps/nyc16 --seeds 0 1 2 --mults 1.0 1.5 2.0
"""
import argparse, collections, copy, json, os, sys, time
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env_map as E

traci = E.traci
DETECT_M = 150.0        # detector range (m) upstream of the stop line
NEAR_M = 50.0           # "about to arrive" band
T_MAX = 60.0            # shield: longest red (s) allowed for a phase with stopped vehicles
FULL_OCC = 0.85         # shield: exit lane counted as full above this occupancy
MAXP = 3                # phases per light padded to this (largest on nyc16 is 3; raised automatically)
F = 15                  # features per phase token
torch.set_num_threads(1)


# ======================================================================================== sensing / features
class Sensor:
    """Detector zones, per-step lane statistics, phase tokens, rewards and neighbour messages for one env."""

    def __init__(self, env, drop=0.0, rng=None):
        self.env, self.drop = env, drop
        self.rng = rng or np.random.default_rng(0)
        net = env.net
        tls_nodes = {n for a in env.agents for n in env.tls_nodes[a]}
        tls_nodes |= {t["id"] for t in env.info.report["traffic_lights"]}
        self.vmax = {}
        self.tls_nodes = tls_nodes
        # zone[l] = [(lane_id, offset_m)]: offset = distance from that lane's end to l's stop line (detector, <= DETECT_M)
        # rzone[l]: the same walk without the distance cap (whole approach up to the previous signal / map edge);
        #           used ONLY for the training reward (centralised training, decentralised execution)
        self.zone, self.boundary, self.rzone = {}, {}, {}
        for a in env.agents:
            for l in env.lanes[a]:
                self.zone[l], self.boundary[l] = self._walk(l, DETECT_M)
                self.rzone[l], _ = self._walk(l, 1e9)
                for z, _ in self.rzone[l]:
                    self.vmax[z] = net.getLane(z).getSpeed()
        # entry edge -> agent whose approach it feeds (for vehicles waiting to be inserted)
        lane_owner = {z: a for a in env.agents for l in env.lanes[a] for z, _ in self.rzone[l]}
        self.entry_agent = {}
        for e0 in env.info.entries:
            frontier, seen = [net.getEdge(e0)], set()
            while frontier and e0 not in self.entry_agent:
                e = frontier.pop(0)
                if e.getID() in seen:
                    continue
                seen.add(e.getID())
                hit = [lane_owner[ln.getID()] for ln in e.getLanes() if ln.getID() in lane_owner]
                if hit:
                    self.entry_agent[e0] = hit[0]
                    break
                frontier += list(e.getOutgoing().keys())
        # message sources: for each agent lane l, the (upstream agent, in-lane, out-lane) movements that feed l
        owner_out = collections.defaultdict(list)
        for b in env.agents:
            for p, links in enumerate(env.phase_links[b]):
                for i, o in links:
                    owner_out[o].append((b, p, i))
        # the lane where the zone stopped is upstream of the signal's node; the upstream signal's OUT lanes are
        # the lanes that end at our zone's far end: take the lanes leaving the boundary node into the zone
        self.feeds = {}
        for a in env.agents:
            for l in env.lanes[a]:
                srcs = set()
                for z, _ in self.zone[l]:
                    for (b, p, i) in owner_out.get(z, []):
                        if b != a:
                            srcs.add((b, p, i))
                self.feeds[l] = sorted(srcs)
        self.n_phase = {a: env.n_acts[a] for a in env.agents}
        self.in_lanes = {a: [sorted({i for i, _ in links}) for links in env.phase_links[a]] for a in env.agents}
        self.out_lanes = {a: [sorted({o for _, o in links}) for links in env.phase_links[a]] for a in env.agents}
        self.reset()

    def _walk(self, l, cap):
        net = self.env.net
        zone, bnd, frontier, seen = [(l, 0.0)], [], [(net.getLane(l), 0.0)], {l}
        while frontier:
            ln, off = frontier.pop()
            reach = off + ln.getLength()
            if reach >= cap:
                continue
            node = ln.getEdge().getFromNode()
            for e_in in node.getIncoming():
                for c in e_in.getConnections(ln.getEdge()):
                    if c.getToLane().getID() != ln.getID():
                        continue
                    up = c.getFromLane()
                    if up.getID() in seen:
                        continue
                    seen.add(up.getID())
                    if node.getID() in self.tls_nodes or node.getType() == "traffic_light":
                        bnd.append(up.getID())        # stop: the upstream lane belongs to another signal
                        continue
                    zone.append((up.getID(), reach))
                    frontier.append((up, reach))
        return zone, bnd

    def reset(self):
        import xml.etree.ElementTree as ET
        self.trip_from = {}
        rf = getattr(self.env, "route_file", None)
        if rf and os.path.exists(rf):
            for t in ET.parse(rf).getroot().iter("trip"):
                self.trip_from[t.get("id")] = t.get("from")
        self.red_age = {a: np.zeros(self.n_phase[a]) for a in self.env.agents}
        self.msg = {a: np.zeros(self.n_phase[a]) for a in self.env.agents}       # delayed messages (belief)
        self.msg_next = {a: np.zeros(self.n_phase[a]) for a in self.env.agents}

    # ------------------------------------------------------------------ one sweep of the detectors
    def sense(self):
        """Read every detector once. Must be called once per decision step (after env.step / env.reset)."""
        env = self.env
        lane_cache = {}

        def raw(z):
            if z not in lane_cache:
                ids = traci.lane.getLastStepVehicleIDs(z)
                lane_cache[z] = [(traci.vehicle.getSpeed(v), traci.vehicle.getLanePosition(v),
                                  traci.vehicle.getWaitingTime(v)) for v in ids]
            return lane_cache[z]

        st = {}
        for a in env.agents:
            for l in env.lanes[a]:
                q = near = far = dl = mw = 0.0
                for z, off in self.zone[l]:
                    length = self._len(z)
                    vm = max(1.0, self.vmax[z])
                    for sp, pos, w in raw(z):
                        d = off + length - pos
                        if d > DETECT_M:
                            continue
                        dl += max(0.0, 1.0 - sp / vm)
                        if sp < 0.1:
                            q += 1; mw = max(mw, w)
                        elif d <= NEAR_M:
                            near += 1
                        else:
                            far += 1
                st[l] = (q, near, far, dl, mw)
        self.st = st
        rd = {}
        for a in env.agents:
            tot = 0.0
            for l in env.lanes[a]:
                for z, _ in self.rzone[l]:
                    vm = max(1.0, self.vmax[z])
                    tot += sum(max(0.0, 1.0 - sp / vm) for sp, _, _ in raw(z))
            rd[a] = tot
        for v in traci.simulation.getPendingVehicles():          # waiting to enter the map: full delay
            a = self.entry_agent.get(self.trip_from.get(v))
            if a is not None:
                rd[a] += 1.0
        self.rdelay = rd
        occ, oq = {}, {}
        for a in env.agents:
            for outs in self.out_lanes[a]:
                for o in outs:
                    if o not in occ:
                        occ[o] = traci.lane.getLastStepOccupancy(o) / 100.0
                        oq[o] = traci.lane.getLastStepHaltingNumber(o)
        self.occ, self.oq = occ, oq
        # red ages (own knowledge)
        for a in env.agents:
            for p in range(self.n_phase[a]):
                dem = sum(st[l][0] for l in self.in_lanes[a][p])
                if p == env.cur[a] or dem == 0:
                    self.red_age[a][p] = 0.0
                else:
                    self.red_age[a][p] += E.DELTA
        # messages: deliver last step's, compute this step's (arrive next step)
        for a in env.agents:
            keep = self.rng.random(self.n_phase[a]) >= self.drop
            self.msg[a] = np.where(keep, self.msg_next[a], self.msg[a])
            m = np.zeros(self.n_phase[a])
            for p in range(self.n_phase[a]):
                tot = 0.0
                for l in self.in_lanes[a][p]:
                    for (b, pb, i) in self.feeds[l]:
                        if env.cur[b] == pb and i in self.st:
                            q, near, far, _, _ = self.st[i]
                            tot += q + near
                m[p] = tot
            self.msg_next[a] = m

    def _len(self, z):
        if not hasattr(self, "_lens"):
            self._lens = {}
        if z not in self._lens:
            self._lens[z] = self.env.net.getLane(z).getLength()
        return self._lens[z]

    # ------------------------------------------------------------------ derived quantities
    def phase_stats(self, a):
        """Per phase: queue, near, far, delay, max wait, exit queue (avg per exit lane x n in-lanes), exit occupancy."""
        out = []
        for p in range(self.n_phase[a]):
            ins, outs = self.in_lanes[a][p], self.out_lanes[a][p]
            q = sum(self.st[l][0] for l in ins); near = sum(self.st[l][1] for l in ins)
            far = sum(self.st[l][2] for l in ins); dl = sum(self.st[l][3] for l in ins)
            mw = max(self.st[l][4] for l in ins)
            oq = np.mean([self.oq[o] for o in outs]) * len(ins)
            oc = np.mean([self.occ[o] for o in outs])
            full = all(self.occ[o] > FULL_OCC for o in outs)
            out.append(dict(q=q, near=near, far=far, delay=dl, mw=mw, oq=oq, occ=oc, full=full, n=len(ins)))
        return out

    def tokens(self):
        """{agent: (MAXP x F array, mask)}"""
        env, res = self.env, {}
        for a in env.agents:
            ps = self.phase_stats(a)
            T = np.zeros((MAXP, F), np.float32)
            M = np.zeros(MAXP, np.float32)
            since = min(env.since[a], 120) / 60.0
            for p, s in enumerate(ps):
                T[p] = [s["q"] / 10, s["near"] / 10, s["far"] / 10, s["delay"] / 10, s["oq"] / 10, s["occ"],
                        (s["q"] + s["near"] - s["oq"]) / 10, min(s["mw"], 180) / 60, min(self.red_age[a][p], 180) / 60,
                        float(p == env.cur[a]), since, self.msg[a][p] / 10, s["n"] / 4, float(s["full"]), 1.0]
                M[p] = 1.0
            res[a] = (T, M)
        return res

    def rewards(self, alpha):
        env = self.env
        own = {a: -self.rdelay[a] / 10.0 for a in env.agents}
        if alpha == 0:
            return own
        return {a: own[a] + alpha * (np.mean([own[b] for b in env.neighbors[a]]) if env.neighbors[a] else 0.0)
                for a in env.agents}


# ======================================================================================== expert + shield
class AdvancedMP:
    """Advanced Max-Pressure (Zhang et al. 2022 style) on detector data: keep the current phase while its queue
    pressure plus the vehicles about to arrive beats the best competing pressure, else switch to that phase."""

    def __init__(self, sensor):
        self.s = sensor

    def act(self):
        env, out = self.s.env, {}
        for a in env.agents:
            ps = self.s.phase_stats(a)
            P = [s["q"] - s["oq"] for s in ps]
            cur, best = env.cur[a], int(np.argmax(P))
            out[a] = cur if ps[cur]["near"] + P[cur] >= P[best] else best
        return out


class Shield:
    """Temporal-logic runtime shield (phi_1 no starvation, phi_2 no green into a blocked exit). Uses only own
    detector knowledge. Returns the executed actions and how many proposals were overridden."""

    def __init__(self, sensor, tmax=T_MAX):
        self.s, self.tmax = sensor, tmax

    def filter(self, acts):
        env, out, n = self.s.env, {}, 0
        for a in env.agents:
            ps = self.s.phase_stats(a)
            act = acts[a]
            red = self.s.red_age[a]
            starving = [p for p in range(len(ps)) if red[p] >= self.tmax and ps[p]["q"] > 0]
            if starving:
                if act not in starving:
                    act = max(starving, key=lambda p: red[p])
            elif ps[act]["full"] and ps[act]["q"] + ps[act]["near"] > 0:
                ok = [p for p in range(len(ps)) if not ps[p]["full"] and ps[p]["q"] + ps[p]["near"] > 0]
                if ok:
                    act = env.cur[a] if env.cur[a] in ok else max(ok, key=lambda p: ps[p]["q"] - ps[p]["oq"])
            n += act != acts[a]
            out[a] = act
        return out, n


# ======================================================================================== network + learner
class PhaseNet(nn.Module):
    def __init__(self, f=F, h=64, heads=4):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(f, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.att = nn.MultiheadAttention(h, heads, batch_first=True)
        self.head = nn.Sequential(nn.Linear(3 * h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x, m):                      # x: B x P x F, m: B x P (1 valid)
        h = self.enc(x)
        pad = m < 0.5
        a, _ = self.att(h, h, h, key_padding_mask=pad)
        ctx = (h * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
        z = torch.cat([h, a, ctx.unsqueeze(1).expand_as(h)], -1)
        q = self.head(z).squeeze(-1)
        return q.masked_fill(pad, -1e9)


class Replay:
    def __init__(self, cap=300000):
        self.cap, self.n, self.i = cap, 0, 0
        self.x = np.zeros((cap, MAXP, F), np.float32); self.m = np.zeros((cap, MAXP), np.float32)
        self.a = np.zeros(cap, np.int64); self.r = np.zeros(cap, np.float32); self.g = np.zeros(cap, np.float32)
        self.x2 = np.zeros((cap, MAXP, F), np.float32); self.m2 = np.zeros((cap, MAXP), np.float32)
        self.demo = np.zeros(cap, np.float32)

    def add(self, x, m, a, r, g, x2, m2, demo):
        k = self.i
        self.x[k], self.m[k], self.a[k], self.r[k], self.g[k], self.x2[k], self.m2[k], self.demo[k] = x, m, a, r, g, x2, m2, demo
        self.i = (k + 1) % self.cap; self.n = min(self.n + 1, self.cap)

    def sample(self, b):
        idx = np.random.randint(0, self.n, b)
        t = lambda v: torch.from_numpy(v[idx])
        return t(self.x), t(self.m), t(self.a), t(self.r), t(self.g), t(self.x2), t(self.m2), t(self.demo)


class RapidAgent:
    def __init__(self, lr=5e-4, gamma=0.95, nstep=3, batch=256, tau=0.005, margin=0.5):
        self.q = PhaseNet(); self.qt = copy.deepcopy(self.q)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buf = Replay()
        self.gamma, self.nstep, self.batch, self.tau, self.margin = gamma, nstep, batch, tau, margin
        self.pending = {}
        self.demo_w = 1.0

    def act(self, toks, agents, eps=0.0, expert=None, rng=np.random):
        X = torch.from_numpy(np.stack([toks[a][0] for a in agents]))
        Mk = torch.from_numpy(np.stack([toks[a][1] for a in agents]))
        with torch.no_grad():
            g = self.q(X, Mk).argmax(1).numpy()
        out = {}
        for k, a in enumerate(agents):
            if rng.random() < eps:
                if expert is not None and rng.random() < 0.5:
                    out[a] = expert[a]
                else:
                    out[a] = int(rng.integers(int(Mk[k].sum()))) if hasattr(rng, "integers") else int(np.random.randint(int(Mk[k].sum())))
            else:
                out[a] = int(g[k])
        return out

    # n-step transitions per agent
    def store(self, toks, acts, rews, toks2, agents, demo=False, last=False):
        for a in agents:
            dq = self.pending.setdefault(a, collections.deque())
            dq.append((toks[a][0], toks[a][1], acts[a], rews[a]))
            if len(dq) == self.nstep or last:
                while dq and (len(dq) == self.nstep or last):
                    R = sum(self.gamma ** k * dq[k][3] for k in range(len(dq)))
                    g = self.gamma ** len(dq)
                    x, m, ac, _ = dq[0]
                    self.buf.add(x, m, ac, R, g, toks2[a][0], toks2[a][1], float(demo))
                    dq.popleft()
                    if not last:
                        break
        if last:
            self.pending = {}

    def learn(self):
        if self.buf.n < 2000:
            return None
        x, m, a, r, g, x2, m2, d = self.buf.sample(self.batch)
        with torch.no_grad():
            a2 = self.q(x2, m2).argmax(1, keepdim=True)
            y = r + g * self.qt(x2, m2).gather(1, a2).squeeze(1)    # time limit != terminal: always bootstrap
        qa = self.q(x, m)
        q = qa.gather(1, a.unsqueeze(1)).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q, y)
        if self.demo_w > 0 and d.sum() > 0:                          # DQfD large-margin loss on expert samples
            marg = torch.full_like(qa, self.margin).scatter(1, a.unsqueeze(1), 0.0)
            marg = marg.masked_fill(m < 0.5, -1e9)
            lm = ((qa + marg).max(1).values - q) * d
            loss = loss + self.demo_w * lm.sum() / d.sum().clamp(min=1)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        with torch.no_grad():
            for p, pt in zip(self.q.parameters(), self.qt.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)
        return float(loss.detach())


# ======================================================================================== episodes
def run_episode(env, sensor, policy, scenario, seed, train_agent=None, alpha=0.0, eps=0.0, demo=False,
                shield=True, learn_every=1, tripinfo=None, rng=None, record=None):
    """policy: 'rapid' (uses train_agent.q), 'amp', or a callable(env, sensor) -> actions."""
    rng = rng or np.random.default_rng(seed)
    env.reset(scenario, seed, tripinfo_file=tripinfo)
    sensor.reset(); sensor.sense()
    sh, amp = Shield(sensor), AdvancedMP(sensor)
    toks = sensor.tokens()
    done, overrides, switches = False, 0, 0
    maxred = 0.0
    while not done:
        expert = amp.act()
        if policy == "amp":
            prop = expert
        elif policy == "rapid":
            prop = train_agent.act(toks, env.agents, eps=eps, expert=expert, rng=rng)
        else:
            prop = policy(env, sensor)
        acts, n = sh.filter(prop) if shield else (prop, 0)
        overrides += n
        switches += sum(acts[a] != env.cur[a] for a in env.agents)
        _, _, done, _ = env.step(acts)
        sensor.sense()
        maxred = max(maxred, max(r.max() for r in sensor.red_age.values()))
        toks2 = sensor.tokens()
        if train_agent is not None and record:
            rews = sensor.rewards(alpha)
            train_agent.store(toks, acts, rews, toks2, env.agents, demo=demo, last=done)
            if env.t % (E.DELTA * learn_every) == 0:
                train_agent.learn()
        toks = toks2
    m = env.metrics()
    m.update(overrides=overrides, switches_per_light=switches / len(env.agents), max_red_s=maxred)
    return m


def extended(m, tripinfo):
    import xml.etree.ElementTree as ET
    ti = ET.parse(tripinfo).getroot().findall("tripinfo")
    w = np.array([float(t.get("waitingTime")) for t in ti]); tl = np.array([float(t.get("timeLoss")) for t in ti])
    m.update(time_loss_s=float(tl.mean()), wait_p95_s=float(np.percentile(w, 95)), wait_max_s=float(w.max()),
             sumo_wait_s=float(w.mean()))
    return m


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def train(map_dir, seed, episodes, alpha, out, demo_eps=12, mult_range=(0.5, 2.0), p_nominal=0.4, drop=0.2,
          gamma=0.95, tag="rapid"):
    np.random.seed(seed); torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    envs = {}

    def get_env(mult):
        k = round(mult, 2)
        if k not in envs:
            envs[k] = E.SumoMapEnv(map_dir, demand_mult=k)
            envs[k]._sensor = Sensor(envs[k], drop=drop, rng=rng)
        return envs[k]

    agent = RapidAgent(gamma=gamma)
    log, t0 = [], time.time()
    total = demo_eps + episodes
    for ep in range(total):
        scn = E.SCENARIOS[rng.integers(len(E.SCENARIOS))]
        mult = 1.0 if rng.random() < p_nominal else float(np.round(rng.uniform(*mult_range), 1))
        env = get_env(mult)
        if ep < demo_eps:
            m = run_episode(env, env._sensor, "amp", scn, 1000 * seed + ep, train_agent=agent, alpha=alpha,
                            demo=True, record=True, rng=rng)
            if ep == demo_eps - 1:                       # pre-train on the demonstrations
                for _ in range(4000):
                    agent.learn()
        else:
            k = ep - demo_eps
            frac = k / max(1, episodes)
            eps = max(0.02, 0.3 * (1 - frac / 0.6))
            agent.demo_w = max(0.0, 1.0 - frac / 0.3)
            m = run_episode(env, env._sensor, "rapid", scn, 1000 * seed + ep, train_agent=agent, alpha=alpha,
                            eps=eps, record=True, rng=rng)
        log.append(dict(ep=ep, scenario=scn, mult=mult, **{k: float(v) for k, v in m.items()}))
        print(f"[{tag} s{seed}] ep {ep+1}/{total} {scn:10s} x{mult:.1f} wait={m['avg_wait_s']:.1f} "
              f"tt={m['total_travel_h']:.1f} thr={m['throughput']} ovr={m['overrides']} ({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(out, exist_ok=True)
    torch.save(agent.q.state_dict(), os.path.join(out, f"{tag}_s{seed}.pt"))
    import pandas as pd
    pd.DataFrame(log).to_csv(os.path.join(out, f"curve_{tag}_s{seed}.csv"), index=False)
    json.dump(dict(method=tag, seed=seed, episodes=episodes, demo_episodes=demo_eps, alpha=alpha, gamma=gamma,
                   detect_m=DETECT_M, t_max=T_MAX, drop=drop, mult_range=mult_range, p_nominal=p_nominal,
                   train_seconds=round(time.time() - t0)), open(os.path.join(out, f"settings_{tag}_s{seed}.json"), "w"), indent=2)
    for e in envs.values():
        e.close()
    return agent


def evaluate(map_dir, policies, mults, out_csv, drop=0.0):
    """policies: {name: ('rapid', state_dict_path) | ('amp', None) | ('callable', fn)}"""
    import tempfile, pandas as pd
    rows = []
    for mult in mults:
        env = E.SumoMapEnv(map_dir, demand_mult=mult)
        sensor = Sensor(env, drop=drop)
        for name, (kind, arg) in policies.items():
            agent = None
            if kind == "rapid":
                agent = RapidAgent(); agent.q.load_state_dict(torch.load(arg))
            for scn in E.SCENARIOS:
                for sd in (9001, 9002, 9003):
                    f = tempfile.mktemp(suffix=".xml")
                    pol = "rapid" if kind == "rapid" else ("amp" if kind == "amp" else arg)
                    m = run_episode(env, sensor, pol, scn, sd, train_agent=agent, tripinfo=f,
                                    shield=(kind != "callable"))
                    env.close()
                    m = extended(m, f)
                    rows.append(dict(method=name, demand_mult=mult, scenario=scn, eval_seed=sd,
                                     **{k: float(v) for k, v in m.items()}))
            print(f"x{mult} {name} done", flush=True)
        env.close()
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("map_dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--tag", default="rapid")
    ap.add_argument("--mults", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    a = ap.parse_args()
    name = os.path.basename(os.path.normpath(a.map_dir))
    out = os.path.join(ROOT, "results", name, "rapid")
    if a.cmd == "train":
        train(a.map_dir, a.seed, a.episodes, a.alpha, out, gamma=a.gamma, tag=a.tag)
    else:
        pols = {"advanced_mp+shield": ("amp", None)}
        for s in a.seeds:
            pols[f"{a.tag}_s{s}"] = ("rapid", os.path.join(out, f"{a.tag}_s{s}.pt"))
        df = evaluate(a.map_dir, pols, a.mults, os.path.join(out, f"eval_{a.tag}.csv"))
        cols = ["avg_wait_s", "avg_trip_s", "throughput", "total_travel_h", "time_loss_s", "wait_p95_s",
                "wait_max_s", "max_red_s", "switches_per_light"]
        print(df.groupby(["demand_mult", "method"])[cols].mean().round(2).to_string())
