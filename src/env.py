"""Multi-agent SUMO environment: one agent per signalised intersection of a 3x3 grid.

Action  : index of the green phase to serve next (yellow inserted automatically when it changes).
Obs     : own lane queues/counts, current phase (one-hot), time in phase, optional neighbour info.
Reward  : negative total queue on the intersection's incoming lanes (scaled).
"""
import os, sys, tempfile
import numpy as np
try:                                    # find SUMO inside whichever Python is running (pip install eclipse-sumo)
    import sumo as _sumo
    os.environ.setdefault("SUMO_HOME", _sumo.SUMO_HOME)
except ImportError:
    pass
if os.environ.get("SUMO_GUI") == "1":
    import traci
    SUMO_BIN = "sumo-gui"
else:
    import libsumo as traci
    SUMO_BIN = "sumo"
import sumolib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scenarios as S

DELTA, YELLOW = 5, 2          # decision interval and yellow time (seconds)
DIRS = ["N", "E", "S", "W"]


class SumoGridEnv:
    def __init__(self, comm_level=0, seed=0):
        """comm_level: 0 = no neighbour info, 1 = neighbour queue, 2 = neighbour queue + phase."""
        S.ensure_network()
        self.comm_level = comm_level
        self.net = sumolib.net.readNet(S.NET_FILE)
        self.agents = sorted(n.getID() for n in self.net.getNodes()
                             if not n.getID().startswith(("left", "right", "top", "bottom")))
        self.tmp = tempfile.mkdtemp()
        self.running = False
        self._static()
        self.n_act = len(self.green[self.agents[0]])
        self.obs_dim = len(self._dummy_obs())
        self.obs_names = self._make_obs_names()
        assert len(self.obs_names) == self.obs_dim

    # ---------- static topology ----------
    def _dir_of(self, node, other):
        (nx, ny), (ox, oy) = node.getCoord(), other.getCoord()
        dx, dy = ox - nx, oy - ny
        if abs(dy) >= abs(dx):
            return 0 if dy > 0 else 2
        return 1 if dx > 0 else 3

    def _static(self):
        self.neighbor_slot, self.neighbors, self.edge_dir = {}, {}, {}
        for a in self.agents:
            node = self.net.getNode(a)
            slots = [None] * 4
            for e in node.getIncoming():
                d = self._dir_of(node, e.getFromNode())
                self.edge_dir[e.getID()] = d
                fid = e.getFromNode().getID()
                slots[d] = fid if fid in self.agents else None
            self.neighbor_slot[a] = slots
            self.neighbors[a] = slots                      # [N, E, S, W], each an agent id or None
        # SUMO-dependent info, needs a running simulation once
        cfg = self._write_cfg("uniform", 0)
        traci.start(["sumo", "-c", cfg, "--no-warnings", "true", "--no-step-log", "true"])
        self.green, self.lanes, self.lane_cap, self.phase_links = {}, {}, {}, {}
        for a in self.agents:
            logic = traci.trafficlight.getAllProgramLogics(a)[0]
            self.green[a] = [p.state for p in logic.phases if ("G" in p.state or "g" in p.state) and "y" not in p.state]
            ctrl = list(dict.fromkeys(traci.trafficlight.getControlledLanes(a)))
            ctrl.sort(key=lambda l: (self.edge_dir[l.rsplit("_", 1)[0]], int(l.rsplit("_", 1)[1])))
            self.lanes[a] = ctrl
            for l in ctrl:
                self.lane_cap[l] = max(1.0, traci.lane.getLength(l) / 7.5)
            links = traci.trafficlight.getControlledLinks(a)
            self.phase_links[a] = []
            for st in self.green[a]:
                pl = [(links[i][0][0], links[i][0][1]) for i, ch in enumerate(st) if ch in "Gg" and links[i]]
                self.phase_links[a].append(pl)
        traci.close()
        # sanity: phase p serves the same approach directions at every intersection (needed for weight sharing)
        sig = {a: [tuple(sorted({self.edge_dir[i.rsplit('_', 1)[0]] for i, _ in pl})) for pl in self.phase_links[a]]
               for a in self.agents}
        assert len({tuple(v) for v in sig.values()}) == 1, f"phase semantics differ across agents: {sig}"
        self.phase_dirs = sig[self.agents[0]]

    def _make_obs_names(self):
        a = self.agents[0]
        lane_names = [f"{DIRS[self.edge_dir[l.rsplit('_', 1)[0]]]}_lane{l.rsplit('_', 1)[1]}" for l in self.lanes[a]]
        names = [f"{n}_waiting" for n in lane_names] + [f"{n}_cars" for n in lane_names]
        names += [f"phase{p}_green" for p in range(self.n_act)] + ["time_since_change"]
        if self.comm_level >= 1:
            for d in DIRS:
                names.append(f"nb_{d}_queue")
                if self.comm_level >= 2:
                    names += [f"nb_{d}_phase{p}" for p in range(self.n_act)]
        return names

    def _write_cfg(self, scn, seed):
        rou = os.path.join(self.tmp, f"{scn}.rou.xml")
        if not os.path.exists(rou):
            S.build_route_file(scn, rou)
        cfg = os.path.join(self.tmp, f"{scn}.sumocfg")
        with open(cfg, "w") as f:
            f.write(f'<configuration><input><net-file value="{S.NET_FILE}"/><route-files value="{rou}"/></input>'
                    f'<time><begin value="0"/></time></configuration>')
        return cfg

    # ---------- episode control ----------
    def reset(self, scenario="uniform", seed=0, tripinfo_file=None):
        if self.running:
            traci.close()
        cfg = self._write_cfg(scenario, seed)
        args = [SUMO_BIN, "-c", cfg, "--no-warnings", "true", "--no-step-log", "true", "--seed", str(seed),
                "--time-to-teleport", "-1"]
        if SUMO_BIN == "sumo-gui":
            args += ["--start", "--delay", os.environ.get("SUMO_GUI_DELAY", "100")]
        if tripinfo_file:                              # optional: SUMO's own trip output, for metric checks
            args += ["--tripinfo-output", tripinfo_file]
        traci.start(args)
        self.running = True
        self.t = 0
        self.cur = {a: 0 for a in self.agents}
        self.since = {a: 0 for a in self.agents}
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

    def _yellow(self, cur, new):
        return "".join("y" if (c in "Gg" and n not in "Gg") else c for c, n in zip(cur, new))

    def step(self, actions):
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
        done = self.t >= S.EPISODE_LEN
        tot_cars, acc_wait = {}, {}
        for a in self.agents:
            tot_cars[a] = int(sum(traci.lane.getLastStepVehicleNumber(l) for l in self.lanes[a]))
            acc_wait[a] = float(sum(traci.vehicle.getAccumulatedWaitingTime(v)
                                    for l in self.lanes[a] for v in traci.lane.getLastStepVehicleIDs(l)))
        queue = {a: float(halt[a].sum()) for a in self.agents}
        info = {"queue": queue,                         # used by the dashboard traces
                "waiting_cars": dict(queue),            # raw values, so other rewards can be tried
                "total_cars": tot_cars,
                "acc_wait_s": acc_wait,                 # SUMO accumulated waiting time of cars on incoming lanes
                "phase": dict(self.cur),
                "yellow": {a: a in sw for a in self.agents},   # True if a 2 s yellow was inserted this step
                "time": self.t}
        return self._obs(), rewards, done, info

    # ---------- observations ----------
    def _halting(self, a):
        return np.array([traci.lane.getLastStepHaltingNumber(l) for l in self.lanes[a]], dtype=np.float32)

    def _own(self, a):
        h = self._halting(a)
        c = np.array([traci.lane.getLastStepVehicleNumber(l) for l in self.lanes[a]], dtype=np.float32)
        cap = np.array([self.lane_cap[l] for l in self.lanes[a]], dtype=np.float32)
        ph = np.zeros(self.n_act, dtype=np.float32)
        ph[self.cur[a]] = 1.0
        return np.concatenate([np.clip(h / cap, 0, 1), np.clip(c / cap, 0, 1), ph,
                               [min(self.since[a], 60) / 60.0]]).astype(np.float32), float(np.clip(h / cap, 0, 1).mean())

    def _obs(self):
        own = {a: self._own(a) for a in self.agents}
        obs = {}
        for a in self.agents:
            vec = [own[a][0]]
            if self.comm_level >= 1:
                for nb in self.neighbor_slot[a]:
                    f = [own[nb][1]] if nb else [0.0]
                    if self.comm_level >= 2:
                        f += list(own[nb][0][16:16 + self.n_act]) if nb else [0.0] * self.n_act
                    vec.append(np.array(f, dtype=np.float32))
            obs[a] = np.concatenate(vec).astype(np.float32)
        return obs

    def _dummy_obs(self):
        n = len(self.lanes[self.agents[0]])
        d = 2 * n + self.n_act + 1
        if self.comm_level >= 1:
            d += 4 * (1 + (self.n_act if self.comm_level >= 2 else 0))
        return np.zeros(d)

    def own_dim(self):
        return 2 * len(self.lanes[self.agents[0]]) + self.n_act + 1

    def metrics(self):
        m = self.m
        dep = max(1, m["departed"])
        return dict(avg_wait_s=m["halt_vs"] / dep, avg_queue=m["qsum"] / max(1, m["qn"]), max_queue=m["maxq"],
                    throughput=m["arrived"], total_travel_h=m["veh_s"] / 3600.0,
                    avg_trip_s=float(np.mean(self.durations)) if self.durations else float("nan"),
                    in_network=traci.vehicle.getIDCount(), departed=m["departed"])
