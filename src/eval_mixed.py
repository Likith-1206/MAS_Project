"""Evaluation on mixed traffic: the same protocol and metrics as eval_rapid.py, plus delay and waiting per vehicle type.

  python src/eval_mixed.py maps/nyc16 --traffic mixed --mults 1.0
  python src/eval_mixed.py maps/nyc16 --traffic old   --methods rapid_mixed      # RAPID-mixed on single-car traffic
  python src/eval_mixed.py maps/nyc16 --compare                                  # old vs mixed table from the CSVs

Methods: fixed_program, max_pressure, independent (models/<map>/independent_s*.pt), advanced_mp (+shield),
rapid2 (results/<map>/rapid/rapid2_s*.pt, count-based detectors, as trained), rapid_mixed
(results/<map>/rapid_mixed/rapid_mixed_s*.pt, PCU-weighted detectors).
Same 4 scenarios x test seeds 9001-9003. Extra per-type columns (finished vehicles, SUMO tripinfo):
  delay_<type>_s (mean timeLoss), wait_<type>_s (mean waitingTime), n_<type> (finished vehicles of that type).
Writes results/<map>/rapid_mixed/eval_<traffic>_x<mults>.csv (never touches results/<map>/rapid/).
"""
import argparse, glob, os, sys, tempfile
import xml.etree.ElementTree as ET
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env_map as E
import scenario_gen as G
import scenario_gen_mixed as M
import rapid as R
import rapid_mixed as RM
from eval_rapid import HEAD, EXTRA
from run_map import MaxPressureMap, MapIndependentDQN

TYPE_COLS = [f"delay_{t}_s" for t in M.TYPES] + [f"wait_{t}_s" for t in M.TYPES]


def run(env, sensor, kind, obj, scn, seed):
    f = tempfile.mktemp(suffix=".xml")
    own = kind == "own"
    env.reset(scn, seed, tripinfo_file=f, own_programs=own)
    sensor.reset(); sensor.sense()
    sh, amp = R.Shield(sensor), R.AdvancedMP(sensor)
    done, sw, starved, maxred, backlog = False, 0, set(), 0.0, 0.0
    obs = env._obs()
    while not done:
        if own:
            acts = None
        elif kind == "mp":
            acts = obj.act(env, obs)
        elif kind == "idqn":
            acts = obj.act(obs, 0.0)
        elif kind == "amp":
            acts, _ = sh.filter(amp.act())
        else:                                             # rapid2 / rapid_mixed
            acts, _ = sh.filter(obj.act(sensor.tokens(), env.agents))
        if acts is not None:
            sw += sum(acts[a] != env.cur[a] for a in env.agents)
        obs, _, done, _ = env.step(acts)
        sensor.sense()
        backlog += len(E.traci.simulation.getPendingVehicles()) * E.DELTA
        for a in env.agents:
            for p, r in enumerate(sensor.red_age[a]):
                maxred = max(maxred, r)
                if r > R.T_MAX:
                    starved.add((a, p))
    m = env.metrics()
    env.close()
    ti = ET.parse(f).getroot().findall("tripinfo")
    os.remove(f)
    w = np.array([float(t.get("waitingTime")) for t in ti]); tl = np.array([float(t.get("timeLoss")) for t in ti])
    ty = np.array([t.get("vType") for t in ti])
    m.update(time_loss_s=tl.mean(), wait_p95_s=np.percentile(w, 95), wait_max_s=w.max(), sumo_wait_s=w.mean())
    for t in M.TYPES:
        k = ty == t
        m[f"delay_{t}_s"] = tl[k].mean() if k.any() else np.nan
        m[f"wait_{t}_s"] = w[k].mean() if k.any() else np.nan
        m[f"n_{t}"] = int(k.sum())
    sched = len(G.generate(env.info, scn, seed, G.scale_for(env.map_dir, scn) * env.demand_mult)[0])
    m.update(backlog_h=backlog / 3600.0, system_travel_h=m["total_travel_h"] + backlog / 3600.0,
             not_entered=sched - m["departed"], max_red_s=maxred, starvation_viol=len(starved),
             switches_per_light=sw / len(env.agents) if not own else np.nan)
    return m


def main_eval(a):
    name = os.path.basename(os.path.normpath(a.map_dir))
    root = R.ROOT
    out = os.path.join(root, "results", name, "rapid_mixed"); os.makedirs(out, exist_ok=True)
    EnvCls = RM.MixedMapEnv if a.traffic == "mixed" else E.SumoMapEnv
    rows = []
    for mult in a.mults:
        env = EnvCls(a.map_dir, demand_mult=mult)
        plain, pcu = R.Sensor(env), RM.MixedSensor(env)
        jobs = []
        if "fixed_program" in a.methods: jobs.append(("fixed_program", -1, "own", None, plain))
        if "max_pressure" in a.methods: jobs.append(("max_pressure", -1, "mp", MaxPressureMap(), plain))
        if "advanced_mp" in a.methods: jobs.append(("advanced_mp", -1, "amp", None, plain))
        for s in (0, 1, 2):
            p = os.path.join(root, "models", name, f"independent_s{s}.pt")
            if "independent" in a.methods and os.path.exists(p):
                ag = MapIndependentDQN(env); ag.load(torch.load(p)); jobs.append(("independent", s, "idqn", ag, plain))
            p = os.path.join(root, "results", name, "rapid", f"rapid2_s{s}.pt")
            if "rapid2" in a.methods and os.path.exists(p):
                ag = R.RapidAgent(); ag.q.load_state_dict(torch.load(p)); jobs.append(("rapid2", s, "rapid", ag, plain))
            p = os.path.join(out, f"rapid_mixed_s{s}.pt")
            if "rapid_mixed" in a.methods and os.path.exists(p):
                ag = R.RapidAgent(); ag.q.load_state_dict(torch.load(p)); jobs.append(("rapid_mixed", s, "rapid", ag, pcu))
        for meth, s, kind, obj, sensor in jobs:
            for scn in E.SCENARIOS:
                for sd in (9001, 9002, 9003):
                    m = run(env, sensor, kind, obj, scn, sd)
                    rows.append(dict(traffic=a.traffic, method=meth, train_seed=s, demand_mult=mult, scenario=scn,
                                     eval_seed=sd, **{k: float(v) for k, v in m.items()}))
            print(f"{a.traffic} x{mult} {meth} s{s} done", flush=True)
        env.close()
    df = pd.DataFrame(rows)
    mt = "_".join(f"{m:g}" for m in a.mults)
    tag = "" if a.methods == DEFAULT_METHODS else "_" + "-".join(a.methods)
    df.to_csv(os.path.join(out, f"eval_{a.traffic}{tag}_x{mt}.csv"), index=False)
    print(table(df, HEAD + EXTRA + (TYPE_COLS if a.traffic == "mixed" else [])).to_string())


def table(df, cols):
    per = df.groupby(["traffic", "demand_mult", "method", "train_seed"])[cols].mean()
    agg = per.groupby(["traffic", "demand_mult", "method"]).agg(["mean", "std"])
    out = pd.DataFrame(index=agg.index)
    for c in cols:
        mu, sd = agg[(c, "mean")], agg[(c, "std")].fillna(0)
        out[c] = [f"{m:.1f}" if s == 0 else f"{m:.1f}±{s:.1f}" for m, s in zip(mu, sd)]
    return out


def compare(map_dir):
    """Old (single car type) vs mixed traffic, x1.0, every method, from the saved CSVs."""
    name = os.path.basename(os.path.normpath(map_dir))
    rdir, mdir = os.path.join(R.ROOT, "results", name, "rapid"), os.path.join(R.ROOT, "results", name, "rapid_mixed")
    old = [pd.read_csv(os.path.join(rdir, "eval_all_rapid2_x1.csv")),
           pd.read_csv(os.path.join(rdir, "eval_all_base.csv")).query("method == 'fixed_program'")]
    old += [pd.read_csv(f) for f in glob.glob(os.path.join(mdir, "eval_old*_x1.csv"))]
    old = pd.concat(old); old["traffic"] = "old"
    new = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(mdir, "eval_mixed*_x1.csv"))])
    df = pd.concat([old, new])
    df = df[df.demand_mult == 1.0]
    cols = HEAD + ["system_travel_h", "time_loss_s", "wait_p95_s", "starvation_viol"]
    t = table(df, cols).reset_index()
    order = ["fixed_program", "max_pressure", "independent", "advanced_mp", "rapid2", "rapid_mixed"]
    t["o"] = t.method.map({m: i for i, m in enumerate(order)})
    t = t.sort_values(["o", "traffic"], ascending=[True, False]).drop(columns=["o", "demand_mult"])
    print(t.to_string(index=False))
    tt = table(new[new.demand_mult == 1.0], TYPE_COLS).reset_index()
    print(tt.drop(columns=["demand_mult"]).to_string(index=False))
    t.to_csv(os.path.join(mdir, "compare_old_vs_mixed.csv"), index=False)


DEFAULT_METHODS = ["fixed_program", "max_pressure", "independent", "advanced_mp", "rapid2", "rapid_mixed"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map_dir")
    ap.add_argument("--traffic", choices=["mixed", "old"], default="mixed")
    ap.add_argument("--mults", type=float, nargs="+", default=[1.0])
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    compare(a.map_dir) if a.compare else main_eval(a)
