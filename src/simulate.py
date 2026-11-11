"""Visual simulation of our controllers on real cities: map in -> one HTML viewer per map out (double-click it).

  ./simulate_all.sh                                                    # every city -> viz/index.html (opens it)
  python src/simulate.py maps/mumbai                                   # one city: RAPID final + multi-city + DQN + MAPPO + fixed
  python src/simulate.py maps/bangalore maps/jersey maps/mumbai maps/nyc16 maps/pune_test    # all five cities
  python src/simulate.py maps/pune_test --controllers rapid rapid_multicity --compare fixed_program advanced_mp
  python src/simulate.py osm/mumbai.osm --name mumbai_lh                # a new .osm: imported + calibrated first

Per map:
  1. map       an existing maps/<name> folder is used as it is (net.net.xml, map_report.json, demand.json);
               an .osm file is imported with osm_import.py (left-hand traffic auto-detected) and calibrated first
  2. models    RAPID-Light needs NO training per map (one phase-token network for every light of every city):
                 rapid            results/nyc16/rapid_v2_mixed/rapid2_mixedtrained_s0.pt   (final model)
                 rapid_v1         results/nyc16/rapid/rapid_s0.pt
                 rapid_v2         results/nyc16/rapid/rapid2_s0.pt
                 rapid_pcu        results/nyc16/rapid_mixed/rapid_mixed_s0.pt   (PCU sensor of rapid_mixed.py)
                 rapid_multicity  results/multicity/rapid_multicity_s0.pt
               run exactly as in eval_mixed.py: detector Sensor -> greedy Q over phase tokens -> LTL shield.
               Independent DQN / MAPPO are per-map: models/<name>/independent_s0.pt and mappo_s0.pt are loaded, or
               trained on that map first (--episodes / --mappo-episodes) and saved for next time.
  3. simulate  every controller on IDENTICAL traffic (one route file, same SUMO seed): mixed Indian traffic from
               scenario_gen_mixed.py by default, rush hour, test seed 9001 (never used in training)
  4. viewer    viz/<map>_<scenario>_s<seed>.html, plus viz/index.html with every map's numbers and links;
               buildings/parks/rail are drawn when osm/<map>.osm exists, otherwise roads only

PLUGIN: a .py file with make_policy(env, map_dir) returning a function obs -> {light: phase} (or an object with .act).
"""
import argparse, datetime, importlib.util, json, os, sys, tempfile, time, webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)

import numpy as np
import env_map as E
import scenario_gen as G
import osm_import as OI
import viz_export as V

LEFTHAND_BOXES = [  # (min_lat, max_lat, min_lon, max_lon): rough boxes of left-hand-traffic countries
    (6.0, 37.5, 68.0, 97.5),      # India, Sri Lanka, Bangladesh, Nepal
    (49.8, 61.0, -8.7, 1.9),      # UK, Ireland
    (24.0, 46.0, 123.0, 146.0),   # Japan
    (-44.0, -10.0, 112.0, 154.0), # Australia
    (-47.5, -34.0, 166.0, 179.0), # New Zealand
    (-35.0, -22.0, 16.0, 33.0),   # South Africa
    (-11.0, 6.0, 95.0, 141.0),    # Indonesia, Malaysia, Singapore
    (-5.0, 5.5, 33.5, 42.0),      # Kenya, Uganda, Tanzania (north)
    (22.1, 22.6, 113.8, 114.5),   # Hong Kong
]


def osm_center(osm):
    import xml.etree.ElementTree as ET
    for _, el in ET.iterparse(osm, events=("start",)):
        if el.tag == "bounds":
            return ((float(el.get("minlat")) + float(el.get("maxlat"))) / 2, (float(el.get("minlon")) + float(el.get("maxlon"))) / 2)
        if el.tag == "node":
            return float(el.get("lat")), float(el.get("lon"))
    return None


def is_lefthand(osm):
    c = osm_center(osm)
    return bool(c) and any(a <= c[0] <= b and lo <= c[1] <= hi for a, b, lo, hi in LEFTHAND_BOXES)


MAX_CALIB_VEHICLES = 6000     # largest demand tried by the fast calibration (per 1200 s episode)


def log(msg):
    print(f"[simulate {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ pipeline steps
def step_import(osm, map_dir, lefthand, bbox, force):
    net = os.path.join(map_dir, "net.net.xml")
    if os.path.exists(net) and not force:
        log(f"map already imported: {net}")
        return json.load(open(os.path.join(map_dir, "map_report.json")))
    log(f"importing {osm} ({'left' if lefthand else 'right'}-hand traffic) ...")
    limits = dict(max_tls=500, max_approaches=12, max_lanes_per_approach=8)   # the DQN handles any size; only a light-less map fails
    rep = OI.import_osm(osm, map_dir, limits, bbox, ["--lefthand"] if lefthand else None)
    print(OI.summary(rep))
    if rep["n_controllable"] == 0:
        sys.exit("no controllable traffic light on this map: nothing for the agents to control")
    if not rep["entry_edges"] or not rep["exit_edges"]:
        sys.exit("no entry/exit roads found: traffic cannot cross this map")
    return rep


def step_demand(map_dir, full, force):
    path = os.path.join(map_dir, "demand.json")
    if os.path.exists(path) and not force:
        log(f"demand already calibrated: {json.load(open(path)).get('scales')}")
        return
    info = G.MapInfo(map_dir)
    if not info.od:
        sys.exit("no entry->exit route passes a controllable light: cannot generate traffic for this map")
    t0 = time.time()
    if full:
        log("calibrating demand (full, 4 patterns x 3 seeds; this is the slow one) ...")
        res = G.calibrate_all(map_dir, info=info)
    else:
        # Search range sized to THIS map: big real maps have hundreds of entry roads, so the generator's default upper
        # bound (scale 6) would mean ~100k vehicles and one endless gridlocked SUMO run. Cap the test demand instead.
        per_unit = G._expected_total(info, "uniform", 1.0, G.EPISODE_LEN)      # vehicles per episode at scale 1
        hi = min(6.0, MAX_CALIB_VEHICLES / max(1.0, per_unit))
        lo = min(0.05, hi / 40)
        log(f"calibrating demand (fast). {len(info.sources())} entry roads, {per_unit:.0f} vehicles at scale 1; "
            f"testing scale {lo:.3f}..{hi:.3f} (at most {MAX_CALIB_VEHICLES} vehicles per test run)")
        runs = [0]
        orig = G.run_static

        def counted(*args, **kw):                                  # show progress: one line per SUMO test run
            runs[0] += 1
            t1 = time.time()
            out = orig(*args, **kw)
            print(f"      test run {runs[0]:>2}: {out['loaded']:>5} vehicles, {100 * out['backlog_frac']:5.1f}% could not enter "
                  f"({time.time() - t1:.0f} s)", flush=True)
            return out
        G.run_static = counted
        per = {}
        for pat in G.PATTERNS:
            log(f"  pattern {pat} ...")
            if pat == "uniform":
                r = G.calibrate(map_dir, pat, n_seeds=1, lo=lo, hi=hi, iters=6, info=info)
            else:                                                   # other patterns land near the uniform scale
                s = per["uniform"]["scale"]
                r = G.calibrate(map_dir, pat, n_seeds=1, lo=max(lo, 0.4 * s), hi=min(hi, 1.3 * s), iters=4, info=info)
            log(f"  {pat:>10}: scale {r['scale']}  {r.get('at_scale', r.get('note', ''))}")
            per[pat] = {k: v for k, v in r.items() if k != "history"}
        G.run_static = orig
        res = dict(scales={p: v["scale"] for p, v in per.items()}, ok=all(v["ok"] for v in per.values()),
                   max_backlog=0.02, n_seeds=1, per_pattern=per, mode="fast")
    json.dump(res, open(path, "w"), indent=1)
    log(f"demand scales {res['scales']} ({time.time() - t0:.0f} s)")


def train_mappo(map_dir, episodes, seed, level, path):
    """Same training loop as run_mappo_map.py main() (without its baseline evaluation), saves to `path`."""
    import pandas as pd, torch
    import run_map as RM
    from run_experiments import train_seed, check_seed_split
    from mappo_map import MAPPO
    d = RM._dirs(map_dir, None)
    check_seed_split(seed, episodes)
    np.random.seed(seed); torch.manual_seed(seed)
    tenv = E.SumoMapEnv(map_dir)
    agent = MAPPO(tenv, level=level)
    method = "mappo" if level == 0 else f"mappo_c{level}"
    curve, t0 = [], time.time()
    for ep in range(episodes):
        scn = E.SCENARIOS[ep % len(E.SCENARIOS)]
        obs = tenv.reset(scn, train_seed(seed, ep)); done, ret = False, 0.0
        while not done:
            obs, r, done, _ = tenv.step(agent.act(obs, explore=True))
            agent.store(r)
            ret += np.mean(list(r.values()))
        st = agent.update(obs)
        m = tenv.metrics()
        curve.append(dict(episode=ep, scenario=scn, ret=ret, avg_queue=float(m["avg_queue"]), avg_wait_s=float(m["avg_wait_s"]), **st))
        print(f"[{method} s{seed}] ep {ep+1}/{episodes} {scn:10s} return={ret:8.1f} avg_wait={m['avg_wait_s']:.1f} "
              f"ent={st['entropy']:.2f} ({time.time()-t0:.0f}s)", flush=True)
    tenv.close()
    torch.save(agent.state_dict(), path)
    pd.DataFrame(curve).to_csv(os.path.join(d["results"], f"curve_{method}_s{seed}.csv"), index=False)
    log(f"MAPPO trained in {time.time() - t0:.0f} s -> {path}")


PRETTY = {"nyc16": "New York", "pune_test": "Pune (unseen)", "bangalore": "Bangalore", "jersey": "Jersey", "mumbai": "Mumbai"}


def pretty(name):
    return PRETTY.get(name, name.replace("_", " ").title())


RAPID_MODELS = {   # controller name -> (model file relative to traffic_marl/, label, sensor); same files the eval scripts load
    "rapid": ("results/nyc16/rapid_v2_mixed/rapid2_mixedtrained_s{seed}.pt", "RAPID-Light (final)", "plain"),
    "rapid_v1": ("results/nyc16/rapid/rapid_s{seed}.pt", "RAPID-Light v1", "plain"),
    "rapid_v2": ("results/nyc16/rapid/rapid2_s{seed}.pt", "RAPID-Light v2", "plain"),
    "rapid_pcu": ("results/nyc16/rapid_mixed/rapid_mixed_s{seed}.pt", "RAPID-Light PCU", "pcu"),   # needs rapid_mixed.MixedSensor
    "rapid_multicity": ("results/multicity/rapid_multicity_s{seed}.pt", "RAPID-Light multi-city", "plain"),
}


class Ctrl:
    """A controller for record_run: act(obs) -> {light: phase}; optional hooks after env.reset / env.step."""

    def __init__(self, name, label, act=None, own=False, on_reset=None, on_step=None, model=None):
        self.name, self.label, self.act, self.own = name, label, act, own
        self.on_reset, self.on_step, self.model = on_reset, on_step, model


def rapid_parts(env, kind="plain"):
    """RAPID-Light's detector sensor + LTL Shield + Advanced-MP expert for this env (built once per map and sensor kind).
    kind "plain": rapid.Sensor (vehicle counts); "pcu": rapid_mixed.MixedSensor (PCU-weighted counts, as rapid_mixed.py)."""
    cache = env.__dict__.setdefault("_rapid", {})
    if kind not in cache:
        import rapid as R
        R.MAXP = max(3, max(env.n_acts.values()))          # every controllable light is controlled (as eval_multicity.py)
        if kind == "pcu":
            import rapid_mixed as RMX
            sensor = RMX.MixedSensor(env)
        else:
            sensor = R.Sensor(env)
        cache[kind] = (R, sensor, R.Shield(sensor), R.AdvancedMP(sensor))
    return cache[kind]


def load_controller(kind, env, map_dir, name, seed, episodes, level, mappo_episodes=150, rapid_seed=0, rapid_model=None):
    import torch
    import run_map as RM
    mdir = os.path.join(ROOT, "models", name)
    if kind == "fixed_program":
        return Ctrl(kind, "Fixed-time (city signals)", own=True)
    if kind == "max_pressure":
        mp = RM.MaxPressureMap()
        return Ctrl(kind, "Max-Pressure", act=lambda o: mp.act(env, o))
    if kind in RAPID_MODELS or kind == "advanced_mp":
        R, sensor, shield, amp = rapid_parts(env, RAPID_MODELS[kind][2] if kind in RAPID_MODELS else "plain")
        hooks = dict(on_reset=lambda: (sensor.reset(), sensor.sense()), on_step=sensor.sense)
        if kind == "advanced_mp":
            return Ctrl(kind, "Advanced Max-Pressure", act=lambda o: shield.filter(amp.act())[0], **hooks)
        rel, label, _ = RAPID_MODELS[kind]
        path = rapid_model or os.path.join(ROOT, rel.format(seed=rapid_seed))
        if not os.path.exists(path):
            sys.exit(f"RAPID-Light model not found: {path}\n"
                     f"  copy your trained file there (or pass --rapid-model FILE). It is made by:\n"
                     f"    rapid            python src/rapid_v2_mixed.py train maps/nyc16 --seed {rapid_seed}\n"
                     f"    rapid_v1         python src/rapid.py train maps/nyc16 --seed {rapid_seed}\n"
                     f"    rapid_v2         python src/rapid.py train maps/nyc16 --seed {rapid_seed} --tag rapid2\n"
                     f"    rapid_pcu        python src/rapid_mixed.py train maps/nyc16 --seed {rapid_seed}\n"
                     f"    rapid_multicity  python src/rapid_multicity.py train --seed {rapid_seed}")
        agent = R.RapidAgent()
        agent.q.load_state_dict(torch.load(path, map_location="cpu"))
        agent.q.eval()
        # exactly as eval_mixed.run(): greedy Q over the phase tokens, then the shield
        return Ctrl(kind, label, act=lambda o: shield.filter(agent.act(sensor.tokens(), env.agents))[0],
                    model=os.path.relpath(path, ROOT), **hooks)
    if kind == "independent":
        path = os.path.join(mdir, f"independent_s{seed}.pt")
        if not os.path.exists(path):
            log(f"no trained model at {path}: training independent DQN on this map, {episodes} episodes "
                f"(afterwards run_map.py scores it on the 12 test episodes: a few quiet minutes) ...")
            RM.train(map_dir, episodes, seed)
        agent = RM.MapIndependentDQN(env)
        agent.load(torch.load(path, map_location="cpu"))
        return Ctrl(kind, "Independent DQN", act=lambda o: agent.act(o, 0.0), model=os.path.relpath(path, ROOT))
    if kind == "mappo":
        from mappo_map import MAPPO
        tag = "mappo" if level == 0 else f"mappo_c{level}"
        path = os.path.join(mdir, f"{tag}_s{seed}.pt")
        if not os.path.exists(path):
            log(f"no trained model at {path}: training MAPPO (level {level}) on this map, {mappo_episodes} episodes ...")
            os.makedirs(mdir, exist_ok=True)
            train_mappo(map_dir, mappo_episodes, seed, level, path)
        agent = MAPPO(env, level=level)
        agent.load(torch.load(path, map_location="cpu"))
        return Ctrl(kind, f"MAPPO{'' if level == 0 else f' (comm {level})'}", act=lambda o: agent.act(o, explore=False),
                    model=os.path.relpath(path, ROOT))
    if kind.startswith("plugin:"):
        f = kind.split(":", 1)[1]
        spec = importlib.util.spec_from_file_location("viz_plugin", f)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        pol = mod.make_policy(env, map_dir)
        return Ctrl("plugin", getattr(mod, "NAME", os.path.splitext(os.path.basename(f))[0]),
                    act=pol.act if hasattr(pol, "act") else pol, model=f)
    sys.exit(f"unknown controller {kind!r}")


class RecordingEnv(E.SumoMapEnv):
    """SumoMapEnv that lets a Recorder look at every simulated second (same simulation, nothing else changes)."""
    rec = None

    def _sim(self, secs):
        for _ in range(secs):
            E.traci.simulationStep()
            self.t += 1
            for v in E.traci.simulation.getDepartedIDList():
                self.dep_time[v] = self.t
                self.m["departed"] += 1
            for v in E.traci.simulation.getArrivedIDList():
                self.m["arrived"] += 1
                if v in self.dep_time:
                    self.durations.append(self.t - self.dep_time.pop(v))
            self.m["veh_s"] += E.traci.vehicle.getIDCount()
            if self.rec is not None:
                self.rec.snap(E.traci, self.t)


def measure_sensor(env):
    """A separate rapid.Sensor used ONLY to measure red times (Sensor.sense() advances its red timers, so the
    controller's own sensor must never be sensed twice per step)."""
    if not hasattr(env, "_meas"):
        import rapid as R
        R.MAXP = max(3, max(env.n_acts.values()))
        env._meas = (R, R.Sensor(env))
    return env._meas


def run_episode(env, ctrl, route_file, seed, n_trips, rec=None):
    """One 1,200 s episode. Same loop and metrics as eval_mixed.run(), plus the longest red with waiting vehicles
    (max_red_s) and the number of (light, phase) pairs left red > 60 s with waiting vehicles (starvation_viol)."""
    R, meas = measure_sensor(env)
    env.rec = None
    obs = env.reset(seed=seed, route_file=route_file, own_programs=ctrl.own)
    if ctrl.on_reset:
        ctrl.on_reset()
    meas.reset(); meas.sense()
    if rec is not None:
        rec.snap(E.traci, 0)
        env.rec = rec
    steps = dict(t=[], ph=[], q=[], sw=[])
    done, maxred, starved = False, 0.0, set()
    while not done:
        obs, _, done, info = env.step(None if ctrl.own else ctrl.act(obs))
        if ctrl.on_step:
            ctrl.on_step()
        meas.sense()
        for a in env.agents:
            for p, r in enumerate(meas.red_age[a]):
                maxred = max(maxred, r)
                if r > R.T_MAX:
                    starved.add((a, p))
        steps["t"].append(env.t)
        steps["ph"].append([int(info["phase"][a]) for a in env.agents])
        steps["q"].append([int(info["queue"][a]) for a in env.agents])
        steps["sw"].append([1 if info["yellow"].get(a) else 0 for a in env.agents])
    m = env.metrics()
    m["not_entered"] = n_trips - m["departed"]
    # the sensor follows the controlled phase (env.cur); fixed-time programs do not set it, so no value for them
    m["max_red_s"] = float("nan") if ctrl.own else maxred
    m["starvation_viol"] = float("nan") if ctrl.own else len(starved)
    env.rec = None
    return m, steps


def make_routes(map_dir, scenario, seed, traffic, scale, info, path):
    if traffic == "mixed":                                          # the team's Indian mixed traffic (scenario_gen_mixed.py)
        import scenario_gen_mixed as GM
        return GM.build_route_file(map_dir, scenario, seed, path, scale, info)
    return G.build_route_file(map_dir, scenario, seed, path, scale, info)


def record_run(env, scene, origin, ctrl, route_file, seed, n_trips, avg=None):
    rec = V.Recorder(origin, (scene["w"], scene["h"]))
    t0 = time.time()
    m, steps = run_episode(env, ctrl, route_file, seed, n_trips, rec)
    rec.steps = steps
    mr = "–" if ctrl.own else f"{m['max_red_s']:.0f}"
    log(f"  {ctrl.label:<28} wait {m['avg_wait_s']:6.1f} s | trip {m['avg_trip_s']:6.1f} s | arrived {m['throughput']:5d} | "
        f"not entered {m['not_entered']:4d} | total travel {m['total_travel_h']:6.1f} veh-h | longest red {mr:>4} s | {time.time() - t0:.0f} s")
    run = rec.pack(ctrl.name, ctrl.label, m)
    if avg:
        run["avg"] = avg
    return run


AVG_KEYS = ["total_travel_h", "avg_wait_s", "avg_trip_s", "throughput", "not_entered", "max_red_s", "starvation_viol"]


def average_all(env, policies, map_dir, a, scale_of):
    """Every controller on all 12 test episodes (4 patterns x seeds 9001-9003), without recording.
    Returns {controller: {metric: mean}} and {(pattern, seed): {controller: metrics}}."""
    per = {}
    tmp = tempfile.mkdtemp()
    for scn in G.PATTERNS:
        for sd in (9001, 9002, 9003):
            rf = os.path.join(tmp, f"{scn}_{sd}.rou.xml")
            st = make_routes(map_dir, scn, sd, a.traffic, scale_of(scn), env.info, rf)
            per[(scn, sd)] = {c.name: run_episode(env, c, rf, sd, st["n_trips"])[0] for c in policies}
        log(f"  12-episode average: {scn} done")
    avg = {}
    for c in policies:
        avg[c.name] = {}
        for k in AVG_KEYS:
            vals = [per[e][c.name][k] for e in per if per[e][c.name].get(k) is not None]
            vals = [float(v) for v in vals if v == v]                 # drop NaN
            avg[c.name][k] = round(float(np.mean(vals)), 2) if vals else None
    return avg, per


def representative(per, avg, policies):
    """The test episode whose results are closest to the 12-episode averages: smallest sum over controllers of
    |total-travel change vs fixed-time in that episode - average change|."""
    base = next((c.name for c in policies if c.name == "fixed_program"), None)
    def change(m, b):
        return (m["total_travel_h"] - b["total_travel_h"]) / b["total_travel_h"] * 100
    best, bd = None, 1e18
    for e, ms in per.items():
        if base:
            d = sum(abs(change(ms[c.name], ms[base]) - change(avg[c.name], avg[base])) for c in policies if c.name != base)
        else:
            d = sum(abs(ms[c.name]["total_travel_h"] - avg[c.name]["total_travel_h"]) for c in policies)
        if d < bd:
            best, bd = e, d
    return best


def net_is_lefthand(net_file):
    with open(net_file) as f:
        head = f.read(6000)
    return 'lefthand value="true"' in head or 'lefthand value="1"' in head


def find_osm(name, explicit=None):
    if explicit:
        return os.path.abspath(explicit)
    for n in (name, name.replace("_test", ""), name.split("_")[0]):
        p = os.path.join(ROOT, "osm", n + ".osm")
        if os.path.exists(p):
            return p
    return None


def simulate_map(target, a):
    """One map: import/calibrate if needed, load/train every controller, record all on identical traffic, write the viewer."""
    if os.path.isdir(target):                                       # an existing map folder (maps/<name>)
        map_dir = os.path.abspath(target)
        name = os.path.basename(os.path.normpath(map_dir))
        if not os.path.exists(os.path.join(map_dir, "net.net.xml")):
            sys.exit(f"{map_dir} has no net.net.xml")
        osm = find_osm(name, a.osm)
        step_demand(map_dir, a.full_calibration, a.recalibrate)
        lefthand = net_is_lefthand(os.path.join(map_dir, "net.net.xml"))
    else:                                                           # an .osm file: import it first
        osm = os.path.abspath(target)
        name = a.name or "".join(c if c.isalnum() else "_" for c in os.path.splitext(os.path.basename(osm))[0].lower())
        map_dir = os.path.join(ROOT, "maps", name)
        lefthand = (not a.righthand) and (a.lefthand or is_lefthand(osm))
        bbox = [float(v) for v in a.bbox.split(",")] if a.bbox else None
        step_import(osm, map_dir, lefthand, bbox, a.reimport)
        step_demand(map_dir, a.full_calibration, a.reimport or a.recalibrate)

    log(f"===== map '{name}' =====")
    env = RecordingEnv(map_dir, demand_mult=a.demand)
    log(f"{len(env.agents)} agents (controllable lights); buildings layer: {os.path.basename(osm) if osm else 'none (no .osm in osm/)'}")
    scene, origin = V.build_scene(env, osm)
    ctrls = [a.controller] if a.controller else a.controllers
    names = ctrls + [b for b in a.compare if b != "none" and b not in ctrls]
    # train / load every model BEFORE simulating (training runs its own SUMO instances)
    policies = [load_controller(c, env, map_dir, name, a.train_seed, a.episodes, a.level, a.mappo_episodes,
                                a.rapid_seed, a.rapid_model) for c in names]

    scale_of = lambda scn: G.scale_for(map_dir, scn) * a.demand
    scenario, seed, avg, shown = a.scenario, a.seed, None, "chosen"
    if a.average:
        log(f"running all {len(policies)} controllers on the 12 test episodes (4 patterns x seeds 9001-9003) ...")
        avg, per = average_all(env, policies, map_dir, a, scale_of)
        if a.show == "representative":
            scenario, seed = representative(per, avg, policies)
            shown = "representative"
            log(f"episode closest to the 12-episode average: {scenario}, seed {seed}")

    # identical traffic for every controller: one route file, same SUMO seed
    tmp = tempfile.mkdtemp()
    rf = os.path.join(tmp, "viz.rou.xml")
    scale = scale_of(scenario)
    st = make_routes(map_dir, scenario, seed, a.traffic, scale, env.info, rf)
    log(f"traffic: {scenario}, seed {seed}, {st['n_trips']} vehicles over {E.EPISODE_LEN} s, {a.traffic} traffic, scale {scale:.3f}")

    runs = [record_run(env, scene, origin, c, rf, seed, st["n_trips"], avg and avg[c.name]) for c in policies]
    env.close()

    meta = dict(map=name, title=pretty(name), osm=os.path.basename(osm) if osm else None, scenario=scenario, seed=seed, traffic=a.traffic,
                shown=shown, avg_episodes=12 if avg else 0,
                fleet="india" if a.traffic == "mixed" else "cars", demand_scale=round(scale, 3), n_trips=st["n_trips"],
                n_agents=len(env.agents), lefthand=lefthand, controllers=names, label=" vs ".join(r["label"] for r in runs),
                models={c.name: c.model for c in policies}, episode_s=E.EPISODE_LEN, decision_s=E.DELTA, yellow_s=E.YELLOW,
                agent_order=env.agents, created=datetime.datetime.now().isoformat(timespec="seconds"))
    out_dir = a.out_dir or os.path.join(ROOT, "viz")
    out = os.path.join(out_dir, f"{name}.html" if a.average else f"{name}_{scenario}_s{seed}.html")
    V.write_html(out, f"{pretty(name)} - " + ", ".join(c.label for c in policies), scene, runs, meta)
    log(f"viewer written: {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")
    return dict(map=name, file=os.path.basename(out), meta=meta,
                runs=[dict(name=r["name"], label=r["label"], metrics=r.get("avg") or r["metrics"]) for r in runs])


CITY_ORDER = ["nyc16", "pune_test", "mumbai", "bangalore", "jersey"]

HUB_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Traffic-signal simulations</title>
<style>
:root{--bg:#0a0e15;--panel:#121a28;--line:rgba(148,163,184,.16);--text:#e6edf6;--muted:#8b97a8;--acc:#22d3ee}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,Ubuntu,sans-serif}
header{position:fixed;top:0;left:0;right:0;height:52px;display:flex;align-items:center;gap:6px;padding:0 14px;background:#0d131e;
  border-bottom:1px solid var(--line);z-index:5;overflow-x:auto}
header b{margin-right:12px;white-space:nowrap}
.tab{border:0;background:transparent;color:var(--muted);font:inherit;font-weight:600;padding:8px 12px;border-radius:9px;cursor:pointer;white-space:nowrap}
.tab:hover{color:var(--text)}
.tab.on{background:rgba(148,163,184,.14);color:var(--text)}
main{position:absolute;top:52px;left:0;right:0;bottom:0}
iframe{border:0;width:100%;height:100%;display:none;background:var(--bg)}
#overview{height:100%;overflow:auto;padding:24px 16px}
.wrap{max-width:1220px;margin:0 auto}
h1{margin:0 0 4px;font-size:22px}
.sub{color:var(--muted);margin:0 0 20px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px;cursor:pointer}
.card:hover{border-color:var(--acc)}
h2{margin:0 0 4px;font-size:17px}
.card p{margin:0 0 10px;color:var(--muted);font-size:12px}
table{width:100%;border-collapse:collapse;font-size:12px}
td,th{padding:3px 0;text-align:right;white-space:nowrap}
td:first-child,th:first-child{text-align:left}
th{color:var(--muted);font-weight:600;padding-left:8px}
td{font-variant-numeric:tabular-nums;padding-left:8px}
tr.best td{color:var(--acc);font-weight:600}
small{color:var(--muted)}
.go{display:block;margin-top:10px;color:var(--acc);font-weight:600}
</style></head><body>
<header><b>Traffic-signal simulations</b><button class="tab" data-k="">Overview</button></header>
<main><div id="overview"><div class="wrap">
<h1>All cities: RAPID-Light, Independent DQN, MAPPO and fixed-time</h1>
<p class="sub">Pick a city (tabs above or a card) to open its side-by-side simulation. Identical traffic for every controller
(same trips, same SUMO seed). Total travel time is the main metric, % against fixed-time; the best is highlighted.
Max red = longest time a road with waiting vehicles was kept red (RAPID-Light's shield limit: 60 s).</p>
<div class="grid" id="grid"></div></div></div>
<iframe id="frame" title="City simulation"></iframe></main>
<script>
const CITIES = /*__CITIES__*/[];
const $ = id => document.getElementById(id);
const hdr = document.querySelector("header");
CITIES.forEach(c => {
  const b = document.createElement("button"); b.className = "tab"; b.dataset.k = c.key; b.textContent = c.title; hdr.appendChild(b);
  const base = c.runs.find(r => r.name === "fixed_program") || c.runs[c.runs.length - 1];
  const best = c.runs.reduce((a, r) => r.metrics.total_travel_h < a.metrics.total_travel_h ? r : a, c.runs[0]);
  const f = (v, d = 1) => v == null || v !== v ? "&ndash;" : (+v).toFixed(d);
  const rows = c.runs.map(r => {
    const m = r.metrics, bt = base.metrics.total_travel_h;
    const d = r === base || !bt ? "" : ` <small>${((m.total_travel_h - bt) / bt * 100 >= 0 ? "+" : "")}${((m.total_travel_h - bt) / bt * 100).toFixed(0)}%</small>`;
    return `<tr class="${r === best ? "best" : ""}"><td>${r.label.replace(" (city signals)", "")}</td><td>${f(m.total_travel_h)}${d}</td><td>${f(m.avg_wait_s)}</td><td>${f(m.throughput, 0)}</td><td>${f(m.max_red_s, 0)}</td></tr>`;
  }).join("");
  const card = document.createElement("div"); card.className = "card";
  card.innerHTML = `<h2>${c.title}</h2><p>${c.what}</p><table><tr><th>Controller</th><th>Total travel (veh-h)</th><th>Wait (s)</th><th>Arrived</th><th>Max red (s)</th></tr>${rows}</table><span class="go">Open simulation &rarr;</span>`;
  card.onclick = () => { location.hash = c.key; };
  $("grid").appendChild(card);
});
hdr.addEventListener("click", e => { const b = e.target.closest(".tab"); if (b) location.hash = b.dataset.k; });
function show() {
  const k = decodeURIComponent(location.hash.slice(1)), c = CITIES.find(x => x.key === k);
  document.querySelectorAll(".tab").forEach(b => b.classList.toggle("on", (b.dataset.k || "") === (c ? c.key : "")));
  if (c) { if ($("frame").dataset.k !== c.key) { $("frame").src = c.file; $("frame").dataset.k = c.key; } $("frame").style.display = "block"; $("overview").style.display = "none"; }
  else { $("frame").style.display = "none"; $("overview").style.display = "block"; }
}
window.addEventListener("hashchange", show); show();
</script></body></html>"""


def write_index(out_dir, results):
    """viz/index.html: ONE page for every city simulated so far. Overview cards with the numbers, and a tab per city
    that opens that city's side-by-side viewer inside the page (keeps earlier cities between runs)."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "index.html")
    store = path + ".json"
    old = {r["map"]: r for r in json.load(open(store))} if os.path.exists(store) else {}
    for r in results:
        old[r["map"]] = r                                           # newest run of a city replaces the older one
    order = lambda r: (CITY_ORDER.index(r["map"]) if r["map"] in CITY_ORDER else 99, r["map"])
    allr = sorted((r for r in old.values() if os.path.exists(os.path.join(out_dir, r["file"]))), key=order)
    json.dump(allr, open(store, "w"), indent=1)
    cities = []
    for r in allr:
        mt = r["meta"]
        if mt.get("avg_episodes"):
            what = "%d agents &middot; %s traffic &middot; average of 12 test episodes &middot; replay: %s, seed %d%s" % (
                mt["n_agents"], mt["traffic"], mt["scenario"].replace("_", " "), mt["seed"],
                " (closest to the average)" if mt.get("shown") == "representative" else "")
        else:
            what = "%d agents &middot; %s traffic &middot; one episode: %s, seed %d, %d vehicles" % (
                mt["n_agents"], mt["traffic"], mt["scenario"].replace("_", " "), mt["seed"], mt["n_trips"])
        cities.append(dict(key=r["map"], title=pretty(r["map"]), file=r["file"], what=what, runs=r["runs"]))
    with open(path, "w") as f:
        f.write(HUB_HTML.replace("/*__CITIES__*/[]", json.dumps(cities).replace("</", "<\\/")))
    return path


def main(a):
    if len(a.targets) > 1 and (a.osm or a.name):
        sys.exit("--osm / --name only work with a single map")
    results = [simulate_map(t, a) for t in a.targets]
    out_dir = a.out_dir or os.path.join(ROOT, "viz")
    idx = write_index(out_dir, results)
    log(f"index of all simulations: {idx}")
    if a.open:                                                      # always the hub page: every city, one tab each
        webbrowser.open("file://" + os.path.abspath(idx) + ("#" + results[0]["map"] if len(results) == 1 else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="+", help="map folders (maps/mumbai ...) and/or .osm files; one viewer per map")
    ap.add_argument("--osm", help="(one map) .osm file for the buildings layer; default osm/<map>.osm if it exists")
    ap.add_argument("--name", help="(one .osm) map name (folder maps/<name>); default from the file name")
    ap.add_argument("--controllers", nargs="+", default=["rapid", "rapid_multicity", "independent", "mappo"],
                    help="any of: rapid rapid_v1 rapid_v2 rapid_pcu rapid_multicity independent mappo advanced_mp max_pressure plugin:file.py")
    ap.add_argument("--controller", help="a single controller (same as --controllers X)")
    ap.add_argument("--compare", nargs="+", default=["fixed_program"],
                    help="baselines added to the same viewer: fixed_program max_pressure advanced_mp | none")
    ap.add_argument("--scenario", default="rush_hour", choices=G.PATTERNS, help="episode to replay (ignored with --show representative)")
    ap.add_argument("--average", action="store_true",
                    help="also run every controller on all 12 test episodes and show the 12-episode averages (as eval_multicity.py)")
    ap.add_argument("--show", default="representative", choices=["representative", "chosen"],
                    help="with --average: replay the episode closest to the averages (representative) or --scenario/--seed (chosen)")
    ap.add_argument("--seed", type=int, default=9001, help="traffic seed (9001-9003 are the test seeds, never trained on)")
    ap.add_argument("--traffic", default="mixed", choices=["mixed", "cars"],
                    help="mixed = scenario_gen_mixed.py (two-wheeler 52%%, car 28%%, auto 15%%, bus 5%%); cars = scenario_gen.py")
    ap.add_argument("--rapid-seed", type=int, default=0, help="which RAPID-Light training seed: ..._s<rapid-seed>.pt")
    ap.add_argument("--rapid-model", help="use this RAPID-Light .pt file instead of the default path")
    ap.add_argument("--train-seed", type=int, default=0, help="which DQN/MAPPO model: <method>_s<train-seed>.pt")
    ap.add_argument("--episodes", type=int, default=60, help="DQN training episodes if its model must be trained first")
    ap.add_argument("--mappo-episodes", type=int, default=150, help="MAPPO training episodes if its model must be trained first")
    ap.add_argument("--level", type=int, default=0, choices=[0, 1, 2], help="MAPPO communication level")
    ap.add_argument("--demand", type=float, default=1.0, help="multiply the calibrated demand (1.5 = busier)")
    ap.add_argument("--bbox", help="(.osm) WEST,SOUTH,EAST,NORTH: only import this part of the .osm")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--lefthand", action="store_true", help="(.osm) force left-hand traffic")
    g.add_argument("--righthand", action="store_true", help="(.osm) force right-hand traffic")
    ap.add_argument("--reimport", action="store_true", help="(.osm) redo import and calibration")
    ap.add_argument("--recalibrate", action="store_true", help="redo the demand calibration")
    ap.add_argument("--full-calibration", action="store_true", help="exact (slow) calibration of scenario_gen")
    ap.add_argument("--out-dir", help="where the viewers go (default viz/)")
    ap.add_argument("--open", action="store_true", help="open the result in the browser")
    main(ap.parse_args())