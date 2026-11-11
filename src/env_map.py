"""Environment v2: one agent per controllable traffic light of ANY map made by osm_import.py.

  python src/env_map.py maps/nyc16            # print the agents, run one random episode, report the time

Same mechanics as the 3x3 environment (env.py): 5 s decisions, an automatic 2 s yellow when the phase changes,
reward = -(halting cars on the light's incoming lanes)/10, same metrics. What changes is that nothing is assumed
about the junction shape:

  agents       the lights with >= 2 green phases (map_report.json). Other lights keep their own fixed program.
  lanes        each agent's own controlled incoming lanes, ordered by the compass bearing of the approach
               (on a regular grid this is exactly the N, E, S, W order of env.py), then by lane number.
  actions      index of the green phase to serve next, 0 .. n_acts[agent]-1. Lights differ in how many they have:
               use action_mask(agent) / action_masks() when a network has a fixed number of outputs.
  observation  per agent: halting/capacity and cars/capacity for each of ITS lanes, phase one-hot, time in phase / 60.
               Length differs between agents (obs_dims). pad_obs() pads to one common length for shared networks.
  neighbours   from the road graph: the signalised junctions reachable along the roads with no other signal in between.
  traffic      scenario_gen.py, with the calibrated demand scale of the pattern (demand.json).
  baseline     reset(..., own_programs=True) and step(None): the network's own fixed-time programs run, and the same
               metrics are measured, so the baseline is comparable with the agents.
"""
import heapq, math, os, sys, tempfile
import numpy as np
try:                                    # find SUMO inside whichever Python is running (pip install eclipse-sumo)
    import sumo as _sumo
    os.environ.setdefault("SUMO_HOME", _sumo.SUMO_HOME)
except ImportError:
    pass
import libsumo as traci
import sumolib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scenario_gen as G

DELTA, YELLOW = 5, 2          # decision interval and yellow time (seconds), as in env.py
EPISODE_LEN = G.EPISODE_LEN   # 1200 simulated seconds
COMPASS = ["N", "E", "S", "W"]
SCENARIOS = list(G.PATTERNS)


def _bearing_key(edge):
    """Compass position of the approach: angle (clockwise from north, 0-360) of the direction FROM the junction TO
    where the road comes from, shifted by 45 so that north is the range [0, 90) and sorts first."""
    shp = edge.getShape()
    end = shp[-1]
    ref = shp[0]
    for p in reversed(shp[:-1]):                                  # walk back ~15 m: robust for curved roads
        ref = p
        if math.hypot(p[0] - end[0], p[1] - end[1]) >= 15.0:
            break
    ang = math.degrees(math.atan2(ref[0] - end[0], ref[1] - end[1])) % 360.0
    return (ang + 45.0) % 360.0, ang


class SumoMapEnv:
    def __init__(self, map_dir, neighbor_dist=300.0, demand_mult=1.0):
        """map_dir: folder made by osm_import.py (net.net.xml, map_report.json) and calibrated by scenario_gen.py
        (demand.json). neighbor_dist: neighbours are signalised junctions at most this many metres apart along roads.
        demand_mult: multiplies the calibrated demand (for stress tests)."""
        self.map_dir = map_dir
        self.net_file = os.path.join(map_dir, "net.net.xml")
        self.demand_mult = demand_mult
        self.neighbor_limit_m = neighbor_dist
        self.net = sumolib.net.readNet(self.net_file, withPrograms=True)
        self.info = G.MapInfo(map_dir)
        self.tmp = tempfile.mkdtemp()
        self.running = False
        self.route_file = None
        ctrl = [t["id"] for t in self.info.report["traffic_lights"] if t["controllable"]]
        self.agents = sorted(ctrl)
        if not self.agents:
            raise ValueError("this map has no controllable traffic light (>= 2 green phases)")
        self._static()
        self.n_act = max(self.n_acts.values())                       # largest number of actions of any light
        self.max_lanes = max(len(v) for v in self.lanes.values())
        self.max_obs_dim = max(self.obs_dims.values())
        self.padded_dim = 2 * self.max_lanes + self.n_act + 1

    # ------------------------------------------------------------------ static topology (no simulation needed)
    def _static(self):
        self.green, self.lanes, self.lane_cap, self.phase_links = {}, {}, {}, {}
        self.phase_dirs, self.n_acts, self.obs_dims, self.obs_names = {}, {}, {}, {}
        self.tls_nodes, self.centroid = {}, {}
        for a in self.agents:
            tls = self.net.getTLS(a)
            conns = tls.getConnections()                              # (inLane, outLane, linkIndex)
            prog = next(iter(tls.getPrograms().values()))
            self.green[a] = [p.state for p in prog.getPhases()
                             if any(c in "Gg" for c in p.state) and not any(c in "yY" for c in p.state)]
            link = {}
            for inl, outl, idx in conns:
                link[idx] = (inl, outl)
            lane_objs = {inl.getID(): inl for inl, _, _ in conns}
            key = {}
            for lid, ln in lane_objs.items():
                k, ang = _bearing_key(ln.getEdge())
                key[lid] = (k, ln.getEdge().getID(), ln.getIndex(), ang)
            order = sorted(lane_objs, key=lambda l: key[l][:3])
            self.lanes[a] = order
            for l in order:
                self.lane_cap[l] = max(1.0, lane_objs[l].getLength() / 7.5)
            self.phase_links[a], self.phase_dirs[a] = [], []
            for st in self.green[a]:
                pl = [(link[i][0].getID(), link[i][1].getID()) for i, ch in enumerate(st) if ch in "Gg" and i in link]
                self.phase_links[a].append(pl)
                dirs = sorted({COMPASS[int(key[i][3] + 45.0) % 360 // 90] for i, _ in pl}, key=COMPASS.index)
                self.phase_dirs[a].append(dirs)
            self.n_acts[a] = len(self.green[a])
            self.obs_dims[a] = 2 * len(order) + self.n_acts[a] + 1
            names = []
            lane_names = [f"{COMPASS[int(key[l][3] + 45.0) % 360 // 90]}_{l}" for l in order]
            names += [f"{n}_waiting" for n in lane_names] + [f"{n}_cars" for n in lane_names]
            names += [f"phase{p}_green" for p in range(self.n_acts[a])] + ["time_since_change"]
            self.obs_names[a] = names
            self.tls_nodes[a] = sorted({ln.getEdge().getToNode().getID() for ln in lane_objs.values()})
            pts = [self.net.getNode(n).getCoord() for n in self.tls_nodes[a]]
            self.centroid[a] = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
        self._neighbour_graph()

    def _neighbour_graph(self, max_dist=None):
        """Signalised junctions reachable along roads with no other signalised junction in between."""
        max_dist = max_dist or self.neighbor_limit_m
        owner = {n: a for a in self.agents for n in self.tls_nodes[a]}
        down = {a: {} for a in self.agents}
        for a in self.agents:
            heap = [(0.0, n) for n in self.tls_nodes[a]]
            heapq.heapify(heap)
            seen = set()
            while heap:
                d, nid = heapq.heappop(heap)
                if nid in seen:
                    continue
                seen.add(nid)
                node = self.net.getNode(nid)
                if owner.get(nid) not in (None, a):                   # reached another agent: stop here
                    b = owner[nid]
                    down[a][b] = min(d, down[a].get(b, 1e9))
                    continue
                for e in sorted(node.getOutgoing(), key=lambda e: e.getID()):
                    nd = d + e.getLength()
                    if nd <= max_dist and e.getToNode().getID() not in seen:
                        heapq.heappush(heap, (nd, e.getToNode().getID()))
        self.downstream = {a: sorted(down[a]) for a in self.agents}
        self.upstream = {a: sorted(b for b in self.agents if a in down[b]) for a in self.agents}
        self.neighbor_dist = {}
        for a in self.agents:
            for b in set(down[a]) | {x for x in self.agents if a in down[x]}:
                self.neighbor_dist[(a, b)] = min(down[a].get(b, 1e9), down[b].get(a, 1e9))

        def bearing(a, b):
            (ax, ay), (bx, by) = self.centroid[a], self.centroid[b]
            return math.degrees(math.atan2(bx - ax, by - ay)) % 360.0
        self.neighbors = {a: sorted({b for (x, b) in self.neighbor_dist if x == a},
                                    key=lambda b: ((bearing(a, b) + 45.0) % 360.0, b)) for a in self.agents}

    # ------------------------------------------------------------------ actions and padded observations
    def action_mask(self, a):
        m = np.zeros(self.n_act, dtype=np.float32)
        m[:self.n_acts[a]] = 1.0
        return m

    def action_masks(self):
        return {a: self.action_mask(a) for a in self.agents}

    def pad_obs(self, obs):
        """Pad every agent's observation to ONE common length (padded_dim) for a shared network:
        [halting x max_lanes | cars x max_lanes | phase one-hot x n_act | time in phase]. Missing slots are zero."""
        out = {}
        for a, v in obs.items():
            n, k = len(self.lanes[a]), self.n_acts[a]
            p = np.zeros(self.padded_dim, dtype=np.float32)
            p[:n] = v[:n]
            p[self.max_lanes:self.max_lanes + n] = v[n:2 * n]
            p[2 * self.max_lanes:2 * self.max_lanes + k] = v[2 * n:2 * n + k]
            p[-1] = v[-1]
            out[a] = p
        return out

    # ------------------------------------------------------------------ episode control
    def _make_routes(self, scenario, seed):
        if scenario not in G.PATTERNS:
            raise ValueError(f"unknown scenario {scenario!r}; choose from {G.PATTERNS}")
        try:
            scale = G.scale_for(self.map_dir, scenario) * self.demand_mult
        except FileNotFoundError:
            raise FileNotFoundError(f"{self.map_dir}/demand.json not found: run  python src/scenario_gen.py "
                                    f"{self.map_dir} --calibrate-all  first")
        path = os.path.join(self.tmp, "r.rou.xml")
        G.build_route_file(self.map_dir, scenario, seed, path, scale, self.info)
        return path

    def reset(self, scenario="uniform", seed=0, tripinfo_file=None, route_file=None, own_programs=False):
        """Start an episode. scenario: one of SCENARIOS. seed: same scenario + same seed = identical traffic.
        route_file: use this route file instead of generating one. own_programs: do not control the lights,
        let the network's own fixed-time programs run (use step(None))."""
        if self.running:
            traci.close()
            self.running = False
        self.route_file = route_file or self._make_routes(scenario, seed)
        args = ["sumo", "-n", self.net_file, "-r", self.route_file, "--begin", "0", "--no-warnings", "true",
                "--no-step-log", "true", "--seed", str(seed), "--time-to-teleport", "-1"]
        if tripinfo_file:                              # optional: SUMO's own trip output, for metric checks
            args += ["--tripinfo-output", tripinfo_file]
        traci.start(args)
        self.running = True
        self.own_programs = own_programs
        self.t = 0
        self.cur = {a: 0 for a in self.agents}
        self.since = {a: 0 for a in self.agents}
        if not own_programs:
            for a in self.agents:
                traci.trafficlight.setRedYellowGreenState(a, self.green[a][0])
        self.dep_time, self.durations = {}, []
        self.m = dict(halt_vs=0.0, veh_s=0.0, departed=0, arrived=0, qsum=0.0, qn=0, maxq=0.0)
        return self._obs()

    def close(self):
        if self.running:
            traci.close()
            self.running = False

    def _sim(self, secs):
        for _ in range(secs):
            traci.simulationStep()
            self.t += 1
            for v in traci.simulation.getDepartedIDList():
                self.dep_time[v] = self.t
                self.m["departed"] += 1
            for v in traci.simulation.getArrivedIDList():
                self.m["arrived"] += 1
                if v in self.dep_time:
                    self.durations.append(self.t - self.dep_time.pop(v))
            self.m["veh_s"] += traci.vehicle.getIDCount()

    @staticmethod
    def _yellow(cur, new):
        return "".join("y" if (c in "Gg" and n not in "Gg") else c for c, n in zip(cur, new))

    def step(self, actions):
        """actions: {agent: phase index} (every agent, 0 <= action < n_acts[agent]); None in own_programs mode."""
        if self.own_programs:
            sw = []
            self._sim(DELTA)
            for a in self.agents:
                st = traci.trafficlight.getRedYellowGreenState(a)
                if st in self.green[a]:
                    k = self.green[a].index(st)
                    self.since[a] = self.since[a] + DELTA if k == self.cur[a] else DELTA
                    self.cur[a] = k
                else:
                    self.since[a] += DELTA
        else:
            for a in self.agents:
                if a not in actions or not (0 <= actions[a] < self.n_acts[a]):
                    raise ValueError(f"agent {a}: action {actions.get(a)!r} is not in 0..{self.n_acts[a] - 1}")
            sw = [a for a in self.agents if actions[a] != self.cur[a]]
            if sw:
                for a in sw:
                    traci.trafficlight.setRedYellowGreenState(a, self._yellow(self.green[a][self.cur[a]], self.green[a][actions[a]]))
                self._sim(YELLOW)
                for a in sw:
                    traci.trafficlight.setRedYellowGreenState(a, self.green[a][actions[a]])
                    self.cur[a] = actions[a]
                self._sim(DELTA - YELLOW)
            else:
                self._sim(DELTA)
            for a in self.agents:
                self.since[a] = (DELTA - YELLOW) if a in sw else self.since[a] + DELTA
        halt = {a: self._halting(a) for a in self.agents}
        tot = sum(h.sum() for h in halt.values())
        self.m["halt_vs"] += tot * DELTA
        self.m["qsum"] += np.mean([h.sum() for h in halt.values()])
        self.m["qn"] += 1
        self.m["maxq"] = max(self.m["maxq"], max(h.max() for h in halt.values()))
        rewards = {a: -halt[a].sum() / 10.0 for a in self.agents}
        done = self.t >= EPISODE_LEN
        tot_cars, acc_wait = {}, {}
        for a in self.agents:
            tot_cars[a] = int(sum(traci.lane.getLastStepVehicleNumber(l) for l in self.lanes[a]))
            acc_wait[a] = float(sum(traci.vehicle.getAccumulatedWaitingTime(v)
                                    for l in self.lanes[a] for v in traci.lane.getLastStepVehicleIDs(l)))
        queue = {a: float(halt[a].sum()) for a in self.agents}
        info = {"queue": queue,
                "waiting_cars": dict(queue),            # raw values, so other rewards can be tried
                "total_cars": tot_cars,
                "acc_wait_s": acc_wait,                 # SUMO accumulated waiting time of cars on incoming lanes
                "phase": dict(self.cur),
                "yellow": {a: a in sw for a in self.agents},
                "time": self.t}
        return self._obs(), rewards, done, info

    # ------------------------------------------------------------------ observations and metrics
    def _halting(self, a):
        return np.array([traci.lane.getLastStepHaltingNumber(l) for l in self.lanes[a]], dtype=np.float32)

    def _obs(self):
        obs = {}
        for a in self.agents:
            h = self._halting(a)
            c = np.array([traci.lane.getLastStepVehicleNumber(l) for l in self.lanes[a]], dtype=np.float32)
            cap = np.array([self.lane_cap[l] for l in self.lanes[a]], dtype=np.float32)
            ph = np.zeros(self.n_acts[a], dtype=np.float32)
            ph[self.cur[a]] = 1.0
            obs[a] = np.concatenate([np.clip(h / cap, 0, 1), np.clip(c / cap, 0, 1), ph,
                                     [min(self.since[a], 60) / 60.0]]).astype(np.float32)
        return obs

    def metrics(self):
        m = self.m
        dep = max(1, m["departed"])
        return dict(avg_wait_s=m["halt_vs"] / dep, avg_queue=m["qsum"] / max(1, m["qn"]), max_queue=m["maxq"],
                    throughput=m["arrived"], total_travel_h=m["veh_s"] / 3600.0,
                    avg_trip_s=float(np.mean(self.durations)) if self.durations else float("nan"),
                    in_network=traci.vehicle.getIDCount(), departed=m["departed"])

    def describe(self):
        return [dict(agent=a, lanes=len(self.lanes[a]), actions=self.n_acts[a], obs_dim=self.obs_dims[a],
                     serves=self.phase_dirs[a], neighbors=self.neighbors[a]) for a in self.agents]


if __name__ == "__main__":
    import time
    d = sys.argv[1] if len(sys.argv) > 1 else "maps/nyc16"
    env = SumoMapEnv(d)
    print(f"map {d}: {len(env.agents)} agents, up to {env.max_lanes} lanes and {env.n_act} actions, "
          f"observation {min(env.obs_dims.values())}-{env.max_obs_dim} numbers (padded {env.padded_dim})")
    for r in env.describe():
        print(f"  {r['agent'][:34]:>34}: {r['lanes']:>2} lanes, {r['actions']} actions {r['serves']}, neighbours {len(r['neighbors'])}")
    for scn in SCENARIOS:
        t0 = time.time()
        obs, done = env.reset(scn, 9001), False
        rng = np.random.default_rng(0)
        while not done:
            obs, r, done, info = env.step({a: int(rng.integers(env.n_acts[a])) for a in env.agents})
        m = env.metrics()
        print(f"  {scn:>10}: random agent, one episode {time.time() - t0:4.1f} s, avg wait {m['avg_wait_s']:.1f} s, "
              f"throughput {m['throughput']}, in network {m['in_network']}")
    env.close()