"""RAPID-Light for mixed traffic: queues and delay weighted by passenger-car units (PCU).

Everything except the detector read-out comes unchanged from rapid.py (network, learner, shield, Advanced-MP expert,
training loop). Differences:
  * MixedMapEnv   env_map.SumoMapEnv whose route files come from scenario_gen_mixed.py (car, two-wheeler,
                  auto-rickshaw, bus; same trips as the single-car-type traffic).
  * MixedSensor   every count the policy and the shield see is a PCU sum: stopped queue, approaching vehicles,
                  exit-lane queue, neighbour messages (a queue of 10 two-wheelers = 4 PCU, one bus = 3 PCU).
                  The delay rate uses each vehicle's own allowed speed (lane limit x its speedFactor, capped by its
                  type's maximum), so a slow auto-rickshaw driving freely is not counted as delayed, and is weighted
                  by PCU:  r_a = -(sum_i pcu_i * (1 - v_i / v_allowed_i) + PCU waiting to enter) / 10.
                  Vehicle types are read from the route file (vehicle id = trip id), as a camera classifier would.

  python src/rapid_mixed.py train maps/nyc16 --seed 0 --episodes 400
Outputs: results/<map>/rapid_mixed/rapid_mixed_s<seed>.pt, curve_*.csv, settings_*.json
"""
import argparse, os, sys
import xml.etree.ElementTree as ET
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env_map as E
import scenario_gen as G
import scenario_gen_mixed as M
import rapid as R

traci = E.traci


class MixedMapEnv(E.SumoMapEnv):
    """Same environment (actions, observations, metrics), mixed-traffic route files."""

    def _make_routes(self, scenario, seed):
        if scenario not in G.PATTERNS:
            raise ValueError(f"unknown scenario {scenario!r}; choose from {G.PATTERNS}")
        scale = G.scale_for(self.map_dir, scenario) * self.demand_mult
        path = os.path.join(self.tmp, "r.rou.xml")
        M.build_route_file(self.map_dir, scenario, seed, path, scale, self.info)
        return path


class MixedSensor(R.Sensor):
    def reset(self):
        super().reset()
        self.trip_pcu = {}
        rf = getattr(self.env, "route_file", None)
        if rf and os.path.exists(rf):
            for t in ET.parse(rf).getroot().iter("trip"):
                self.trip_pcu[t.get("id")] = M.PCU.get(t.get("type"), 1.0)

    def _pcu(self, v):
        if v not in self.trip_pcu:                   # vehicle not from the route file: ask the simulator once
            self.trip_pcu[v] = M.PCU.get(traci.vehicle.getTypeID(v), 1.0)
        return self.trip_pcu[v]

    def sense(self):
        env = self.env
        lane_cache = {}

        def raw(z):
            if z not in lane_cache:
                lane_cache[z] = [(traci.vehicle.getSpeed(v), traci.vehicle.getLanePosition(v),
                                  traci.vehicle.getWaitingTime(v), self._pcu(v),
                                  max(1.0, traci.vehicle.getAllowedSpeed(v)))
                                 for v in traci.lane.getLastStepVehicleIDs(z)]
            return lane_cache[z]

        st = {}
        for a in env.agents:
            for l in env.lanes[a]:
                q = near = far = dl = mw = 0.0
                for z, off in self.zone[l]:
                    length = self._len(z)
                    for sp, pos, w, pcu, va in raw(z):
                        d = off + length - pos
                        if d > R.DETECT_M:
                            continue
                        dl += pcu * max(0.0, 1.0 - sp / va)
                        if sp < 0.1:
                            q += pcu; mw = max(mw, w)
                        elif d <= R.NEAR_M:
                            near += pcu
                        else:
                            far += pcu
                st[l] = (q, near, far, dl, mw)
        self.st = st
        rd = {}
        for a in env.agents:
            tot = 0.0
            for l in env.lanes[a]:
                for z, _ in self.rzone[l]:
                    tot += sum(pcu * max(0.0, 1.0 - sp / va) for sp, _, _, pcu, va in raw(z))
            rd[a] = tot
        for v in traci.simulation.getPendingVehicles():
            a = self.entry_agent.get(self.trip_from.get(v))
            if a is not None:
                rd[a] += self._pcu(v)
        self.rdelay = rd
        occ, oq = {}, {}
        for a in env.agents:
            for outs in self.out_lanes[a]:
                for o in outs:
                    if o not in occ:
                        occ[o] = traci.lane.getLastStepOccupancy(o) / 100.0
                        oq[o] = sum(pcu for sp, _, _, pcu, _ in raw(o) if sp < 0.1)
        self.occ, self.oq = occ, oq
        for a in env.agents:
            for p in range(self.n_phase[a]):
                dem = sum(st[l][0] for l in self.in_lanes[a][p])
                if p == env.cur[a] or dem == 0:
                    self.red_age[a][p] = 0.0
                else:
                    self.red_age[a][p] += E.DELTA
        for a in env.agents:
            keep = self.rng.random(self.n_phase[a]) >= self.drop
            self.msg[a] = np.where(keep, self.msg_next[a], self.msg[a])
            m = np.zeros(self.n_phase[a])
            for p in range(self.n_phase[a]):
                tot = 0.0
                for l in self.in_lanes[a][p]:
                    for (b, pb, i) in self.feeds[l]:
                        if env.cur[b] == pb and i in self.st:
                            q, near, _, _, _ = self.st[i]
                            tot += q + near
                m[p] = tot
            self.msg_next[a] = m


class _MixedE:
    """Stand-in for the env_map module inside rapid.train: identical except that SumoMapEnv builds mixed traffic."""
    SumoMapEnv = MixedMapEnv

    def __getattr__(self, name):
        return getattr(E, name)


def train(map_dir, seed, episodes, out, tag="rapid_mixed", **kw):
    """rapid.train with the mixed environment and the PCU sensor (rebound in this process only; rapid.py unchanged)."""
    old_E, old_S = R.E, R.Sensor
    R.E, R.Sensor = _MixedE(), MixedSensor
    try:
        return R.train(map_dir, seed, episodes, 0.0, out, tag=tag, **kw)
    finally:
        R.E, R.Sensor = old_E, old_S


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("map_dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--tag", default="rapid_mixed")
    a = ap.parse_args()
    name = os.path.basename(os.path.normpath(a.map_dir))
    out = os.path.join(R.ROOT, "results", name, "rapid_mixed")
    train(a.map_dir, a.seed, a.episodes, out, tag=a.tag)
