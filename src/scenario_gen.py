"""Generic synthetic-demand generator for any map made by osm_import.py.

  python src/scenario_gen.py maps/nyc16 --pattern rush_hour --seed 9001 --scale 1.0 --out /tmp/rush.rou.xml
  python src/scenario_gen.py maps/nyc16 --calibrate                 # find the demand scale for this map

Reads  <map_dir>/net.net.xml  and  <map_dir>/map_report.json.  Writes a SUMO route file of explicit trips.
Demand is always synthetic. Same (map, pattern, seed, scale) -> byte-identical file, independent of SUMO's own RNG,
so every controller can be tested on exactly the same vehicles.

Pieces:
  entries / exits   boundary roads only (dead-end side streets inside the area are NOT sources of demand)
  routes            only entry->exit pairs that have a path through at least one controllable light
  patterns          uniform | asymmetric | rush_hour | corridor   (defined by travel direction, not by names)
  rate              veh/h per entry lane x number of lanes x scale; `scale` is found by calibrate()
"""
import argparse, json, math, os, subprocess, sys, tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
import sumolib

EPISODE_LEN = 1200            # simulated seconds per episode
PATTERNS = ["uniform", "asymmetric", "rush_hour", "corridor"]
BASE_RATE = 300.0             # veh/h per entry lane at scale 1.0
HEAVY, LIGHT = 1.7, 0.5       # asymmetric / rush-hour multipliers for the heavy and light axis
CORRIDOR_BOOST, CORRIDOR_SHARE = 2.0, 0.6    # corridor entry rate vs. normal; share of its trips that use the corridor
CALIB_SEED = 424242           # never a training (1000*s+ep) or test (9001-9003) seed
VTYPE = '<vType id="car" accel="2.6" decel="4.5" length="5" minGap="2.5" maxSpeed="13.9" sigma="0.5"/>'


class MapInfo:
    """Everything the generator needs to know about one imported map."""

    def __init__(self, map_dir, margin_m=None):
        self.dir = map_dir
        self.report = json.load(open(os.path.join(map_dir, "map_report.json")))
        self.net_file = os.path.join(map_dir, "net.net.xml")              # relative to the map folder: portable
        self.net = sumolib.net.readNet(self.net_file)
        (x0, y0), (x1, y1) = self.net.getBoundary()[:2], self.net.getBoundary()[2:]
        self.box = (x0, y0, x1, y1)
        self.margin = margin_m if margin_m is not None else max(60.0, 0.1 * min(x1 - x0, y1 - y0))
        # edges that approach a controllable light: a route is only useful if it crosses one
        self.ctrl_edges = {e for t in self.report["traffic_lights"] if t["controllable"] for e in t["approaches"]}
        self.entries = [e for e in self.report["entry_edges"] if self._near_boundary(self.net.getEdge(e).getFromNode())]
        self.exits = [e for e in self.report["exit_edges"] if self._near_boundary(self.net.getEdge(e).getToNode())]
        self.od, self.od_ctrl = self._valid_pairs()          # od[entry] = [exit, ...]; od_ctrl[(entry, exit)] = lights on path

    def _near_boundary(self, node):
        x, y = node.getCoord()
        x0, y0, x1, y1 = self.box
        return min(x - x0, x1 - x, y - y0, y1 - y) <= self.margin

    def _valid_pairs(self):
        od, od_ctrl = defaultdict(list), {}
        for ein in self.entries:
            a = self.net.getEdge(ein)
            for eout in self.exits:
                b = self.net.getEdge(eout)
                if ein == eout or (b.getFromNode() == a.getToNode() and b.getToNode() == a.getFromNode()):
                    continue                                              # no U-turn at the boundary
                path, _ = self.net.getShortestPath(a, b, vClass="passenger")
                if not path:
                    continue
                n_ctrl = sum(1 for e in path if e.getID() in self.ctrl_edges)
                if n_ctrl >= 1 and len(path) >= 3:
                    od[ein].append(eout)
                    od_ctrl[(ein, eout)] = n_ctrl
        return dict(od), od_ctrl

    def heading(self, entry):
        """Compass class of the direction a vehicle travels when it enters: N, E, S or W."""
        shp = self.net.getEdge(entry).getShape()
        dx, dy = shp[-1][0] - shp[0][0], shp[-1][1] - shp[0][1]
        if abs(dy) >= abs(dx):
            return "N" if dy > 0 else "S"
        return "E" if dx > 0 else "W"

    def lanes(self, entry):
        return self.net.getEdge(entry).getLaneNumber()

    def sources(self):
        return sorted(self.od)

    def describe(self):
        return dict(entries=self.sources(), n_entries=len(self.sources()),
                    dropped_entries=sorted(set(self.report["entry_edges"]) - set(self.sources())),
                    n_exits=len({x for v in self.od.values() for x in v}), n_od_pairs=len(self.od_ctrl),
                    margin_m=round(self.margin), headings={e: self.heading(e) for e in self.sources()})


def _axis(h):
    return "NS" if h in ("N", "S") else "EW"


def _corridor_pair(info):
    """The entry/exit pair whose path crosses the most controllable lights (longest on ties): the demand corridor."""
    return max(info.od_ctrl, key=lambda k: (info.od_ctrl[k], k))


def _multiplier(info, pattern, entry, t0):
    ax = _axis(info.heading(entry))
    axes = {_axis(info.heading(e)) for e in info.sources()}
    if len(axes) < 2 or pattern in ("uniform", "corridor"):
        return 1.0 if pattern != "corridor" else 0.6          # corridor: weak background everywhere
    if pattern == "asymmetric":
        return HEAVY if ax == "EW" else LIGHT
    if pattern == "rush_hour":                                # NS-heavy -> balanced -> EW-heavy
        phase = min(2, int(3 * t0 / EPISODE_LEN))
        if phase == 1:
            return 1.0
        heavy_axis = "NS" if phase == 0 else "EW"
        return HEAVY if ax == heavy_axis else LIGHT
    raise ValueError(pattern)


def _segments(pattern, episode_len):
    return [(i * episode_len // 3, (i + 1) * episode_len // 3) for i in range(3)] if pattern == "rush_hour" else [(0, episode_len)]


def _raw_rates(info, pattern, scale, t0, cor):
    """veh/s entering at each source entry in the time segment starting at t0 (before normalisation)."""
    rates = {}
    for ein in info.sources():
        m = CORRIDOR_BOOST if (cor and ein == cor[0]) else _multiplier(info, pattern, ein, t0)
        rates[ein] = BASE_RATE * info.lanes(ein) * scale * m / 3600.0
    return rates


def _expected_total(info, pattern, scale, episode_len):
    cor = _corridor_pair(info) if pattern == "corridor" else None
    return sum((t1 - t0) * sum(_raw_rates(info, pattern, scale, t0, cor).values()) for t0, t1 in _segments(pattern, episode_len))


def generate(info, pattern, seed, scale=1.0, episode_len=EPISODE_LEN):
    """Return (list of (depart_s, entry, exit), stats). Deterministic in (pattern, seed, scale).
    Every pattern is normalised to the same EXPECTED number of vehicles as 'uniform' at the same scale: a pattern
    changes where and when the traffic goes, never how much there is, so one calibrated scale fits all patterns."""
    if pattern not in PATTERNS:
        raise ValueError(f"unknown pattern {pattern!r}; choose from {PATTERNS}")
    if not info.od:
        raise ValueError("no valid entry->exit routes through a controllable light on this map")
    rng = np.random.default_rng([int(seed), PATTERNS.index(pattern), 7])
    cor = _corridor_pair(info) if pattern == "corridor" else None
    norm = _expected_total(info, "uniform", scale, episode_len) / _expected_total(info, pattern, scale, episode_len)
    trips, per_entry = [], defaultdict(int)
    for (t0, t1) in _segments(pattern, episode_len):
        for ein, rate in _raw_rates(info, pattern, scale, t0, cor).items():
            rate *= norm
            dests = info.od[ein]
            w = np.ones(len(dests))
            if cor and ein == cor[0] and len(dests) > 1:                  # most of the corridor entry's traffic goes the long way
                w = np.where(np.array(dests) == cor[1], 0.0, 1.0)
                w = w / w.sum() * (1 - CORRIDOR_SHARE)
                w[dests.index(cor[1])] = CORRIDOR_SHARE
            w = w / w.sum()
            n = rng.poisson(rate * (t1 - t0))
            for d in np.sort(rng.uniform(t0, t1, n)):
                trips.append((round(float(d), 1), ein, dests[rng.choice(len(dests), p=w)]))
                per_entry[ein] += 1
    trips.sort(key=lambda t: (t[0], t[1], t[2]))
    return trips, dict(n_trips=len(trips), per_entry=dict(per_entry), corridor=cor, normalisation=round(norm, 3))


def write_routes(trips, path):
    lines = ["<routes>", "  " + VTYPE]
    for k, (d, a, b) in enumerate(trips):
        lines.append(f'  <trip id="t{k}" type="car" depart="{d:.1f}" from="{a}" to="{b}" departLane="best" departSpeed="max"/>')
    lines.append("</routes>")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def build_route_file(map_dir, pattern, seed, path, scale=1.0, info=None):
    info = info or MapInfo(map_dir)
    trips, st = generate(info, pattern, seed, scale)
    write_routes(trips, path)
    return st


# ---------------------------------------------------------------- calibration
def _find_sumo():
    import shutil
    exe = shutil.which("sumo")
    if exe:
        return exe
    import sumo
    return os.path.join(sumo.SUMO_HOME, "bin", "sumo")


def run_static(info, route_file, end=EPISODE_LEN, seed=1):
    """Run SUMO with the network's own fixed-time programs; return its statistics (no TraCI, no agents)."""
    with tempfile.TemporaryDirectory() as d:
        stat = os.path.join(d, "stat.xml")
        r = subprocess.run([_find_sumo(), "-n", info.net_file, "-r", route_file, "--end", str(end), "--seed", str(seed),
                            "--no-step-log", "true", "--no-warnings", "true", "--time-to-teleport", "-1",
                            "--duration-log.statistics", "true", "--statistic-output", stat], capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(stat):
            raise RuntimeError("SUMO failed on the generated routes:\n" + (r.stderr or r.stdout)[-1200:])
        root = ET.parse(stat).getroot()
    v, t = root.find("vehicles").attrib, root.find("vehicleTripStatistics").attrib
    loaded = int(v["loaded"])
    dur, loss = float(t["duration"]), float(t["timeLoss"])
    return dict(loaded=loaded, inserted=int(v["inserted"]), arrived=int(t["count"]), backlog=int(v["waiting"]),
                backlog_frac=int(v["waiting"]) / max(1, loaded), loss_ratio=loss / dur if dur > 0 else 0.0,
                mean_time_loss_s=loss, mean_duration_s=dur)


def calibrate(map_dir, pattern="uniform", max_backlog=0.02, n_seeds=3, lo=0.05, hi=6.0, iters=9, info=None):
    """Find the demand scale at which the network is busy but not gridlocked, under fixed-time control.

    Rule (the same for every map): the LARGEST scale at which, with the network's own fixed-time programs,
    at most `max_backlog` of the demand is still waiting to enter the map at the end of the episode
    (mean over `n_seeds` calibration seeds). Below that point every vehicle can enter; above it the entry roads
    are blocked, i.e. demand exceeds what the network can carry. Bisection on scale.
    (Not used: 'time lost per trip', because even at near-zero demand red phases cost a lot of time on real maps.)"""
    info = info or MapInfo(map_dir)
    history = []

    def measure(s):
        rows = []
        for k in range(n_seeds):
            with tempfile.TemporaryDirectory() as d:
                rf = os.path.join(d, "r.rou.xml")
                build_route_file(map_dir, pattern, CALIB_SEED + k, rf, s, info)
                rows.append(run_static(info, rf, seed=1 + k))
        m = {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
        history.append(dict(scale=round(s, 4), **{key: round(v, 3) for key, v in m.items()}))
        return m

    ok = lambda s: measure(s)["backlog_frac"] <= max_backlog
    if not ok(lo):
        return dict(scale=lo, ok=False, note="entries are blocked even at the lowest demand: check the map", history=history)
    if ok(hi):
        return dict(scale=hi, ok=True, note="demand cap reached without saturation", history=history)
    a, b = lo, hi
    for _ in range(iters):
        mid = (a + b) / 2
        a, b = (mid, b) if ok(mid) else (a, mid)
    final = measure(a)
    return dict(scale=round(a, 3), ok=True, max_backlog=max_backlog, n_seeds=n_seeds, pattern=pattern,
                at_scale=dict(vehicles=round(final["loaded"]), backlog_frac=round(final["backlog_frac"], 3),
                              mean_time_loss_s=round(final["mean_time_loss_s"], 1)), history=history)


def calibrate_all(map_dir, max_backlog=0.02, n_seeds=3, iters=9, info=None):
    """Calibrate every pattern separately. The patterns do not saturate the network at the same demand
    (on maps/nyc16: uniform 1.14, asymmetric 0.98, rush_hour 1.00, corridor 0.77), so one scale would leave
    some scenarios gridlocked and others light. Each pattern gets the largest scale the network can carry."""
    info = info or MapInfo(map_dir)
    per = {}
    for pat in PATTERNS:
        r = calibrate(map_dir, pat, max_backlog, n_seeds, iters=iters, info=info)
        per[pat] = {k: v for k, v in r.items() if k != "history"}
    return dict(scales={p: v["scale"] for p, v in per.items()}, ok=all(v["ok"] for v in per.values()),
                max_backlog=max_backlog, n_seeds=n_seeds, per_pattern=per)


def scale_for(map_dir, pattern):
    """Demand scale to use for `pattern` on this map (reads demand.json; also accepts the old single-scale file)."""
    d = json.load(open(os.path.join(map_dir, "demand.json")))
    return d["scales"][pattern] if "scales" in d else d["scale"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map_dir")
    ap.add_argument("--pattern", default="uniform", choices=PATTERNS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--out", help="route file to write")
    ap.add_argument("--calibrate", action="store_true", help="find the demand scale for this map and write it to <map_dir>/demand.json")
    ap.add_argument("--calibrate-all", action="store_true", help="calibrate all four patterns separately; writes a scale per pattern to <map_dir>/demand.json")
    ap.add_argument("--max-backlog", type=float, default=0.02, help="calibration: allowed share of demand still waiting to enter")
    ap.add_argument("--margin", type=float, help="metres from the map edge that count as 'boundary' (default max(60, 10%% of size))")
    a = ap.parse_args()
    info = MapInfo(a.map_dir, a.margin)
    print(json.dumps(info.describe(), indent=1))
    if a.calibrate:
        res = calibrate(a.map_dir, a.pattern, a.max_backlog, info=info)
        for h in res["history"]:
            print(h)
        print({k: v for k, v in res.items() if k != "history"})
        json.dump({k: v for k, v in res.items() if k != "history"}, open(os.path.join(a.map_dir, "demand.json"), "w"), indent=1)
    if a.calibrate_all:
        res = calibrate_all(a.map_dir, a.max_backlog, info=info)
        print(json.dumps(res, indent=1))
        json.dump(res, open(os.path.join(a.map_dir, "demand.json"), "w"), indent=1)
    if a.out:
        print(build_route_file(a.map_dir, a.pattern, a.seed, a.out, a.scale, info))