"""Evaluate controllers on any map, on old (single car type) and mixed traffic. Every controllable light is controlled
by the method (rapid.MAXP raised in this process to the map's largest phase count), 4 scenarios x seeds 9001-9003.

  python src/eval_multicity.py --map pune_test --traffic mixed --methods fixed_program max_pressure advanced_mp
  python src/eval_multicity.py --compare

Methods:
  fixed_program, max_pressure, advanced_mp       non-learning baselines
  rapid2                results/nyc16/rapid/rapid2_s{0,1,2}.pt                  (New York, single car type)
  rapid2_mixedtrained   results/nyc16/rapid_v2_mixed/rapid2_mixedtrained_s*.pt   (New York, mixed traffic)
  rapid_multicity       results/multicity/rapid_multicity_s*.pt                  (4 cities, mixed traffic)
Outputs (new folder only): results/multicity/eval/<map>_<traffic>_<methods>.csv, compare_table.csv
"""
import argparse, glob, os, sys
import pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env_map as E
import rapid as R
import rapid_mixed as RM
import eval_mixed as EM
from eval_rapid import HEAD
from run_map import MaxPressureMap

MODELS = {"rapid2": ("nyc16/rapid", "rapid2"),
          "rapid2_mixedtrained": ("nyc16/rapid_v2_mixed", "rapid2_mixedtrained"),
          "rapid_multicity": ("multicity", "rapid_multicity")}
OUT = os.path.join(R.ROOT, "results", "multicity", "eval")


def evaluate(map_name, traffic, methods):
    os.makedirs(OUT, exist_ok=True)
    map_dir = os.path.join(R.ROOT, "maps", map_name)
    env = (RM.MixedMapEnv if traffic == "mixed" else E.SumoMapEnv)(map_dir)
    R.MAXP = max(3, max(env.n_acts.values()))
    sensor = R.Sensor(env)
    rows = []
    for meth in methods:
        if meth in ("fixed_program", "max_pressure", "advanced_mp"):
            kind = {"fixed_program": "own", "max_pressure": "mp", "advanced_mp": "amp"}[meth]
            jobs = [(-1, kind, MaxPressureMap() if kind == "mp" else None)]
        else:
            folder, tag = MODELS[meth]
            jobs = []
            for s in (0, 1, 2):
                ag = R.RapidAgent()
                ag.q.load_state_dict(torch.load(os.path.join(R.ROOT, "results", folder, f"{tag}_s{s}.pt")))
                jobs.append((s, "rapid", ag))
        for s, kind, obj in jobs:
            for scn in E.SCENARIOS:
                for sd in (9001, 9002, 9003):
                    m = EM.run(env, sensor, kind, obj, scn, sd)
                    rows.append(dict(map=map_name, traffic=traffic, method=meth, train_seed=s, demand_mult=1.0,
                                     scenario=scn, eval_seed=sd, **{k: float(v) for k, v in m.items()}))
            print(f"{map_name} {traffic} {meth} s{s} done", flush=True)
    env.close()
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"{map_name}_{traffic}_{'-'.join(methods)}.csv"), index=False)


def compare():
    df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(OUT, "*.csv")) if "compare" not in f])
    cols = HEAD + ["time_loss_s", "wait_p95_s", "starvation_viol"]
    per = df.groupby(["map", "traffic", "method", "train_seed"])[cols].mean()
    agg = per.groupby(["map", "traffic", "method"]).agg(["mean", "std"])
    out = pd.DataFrame(index=agg.index)
    for c in cols:
        mu, sd = agg[(c, "mean")], agg[(c, "std")].fillna(0)
        out[c] = [f"{a:.1f}" if b == 0 else f"{a:.1f}±{b:.1f}" for a, b in zip(mu, sd)]
    out = out.reset_index()
    order = ["fixed_program", "max_pressure", "advanced_mp", "rapid2", "rapid2_mixedtrained", "rapid_multicity"]
    mo = ["pune_test", "nyc16", "bangalore", "jersey", "mumbai"]
    out["o"] = out.method.map({m: i for i, m in enumerate(order)})
    out["mo"] = out["map"].map({m: i for i, m in enumerate(mo)})
    out = out.sort_values(["mo", "traffic", "o"], ascending=[True, False, True]).drop(columns=["o", "mo"])
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    print(out.to_string(index=False))
    out.to_csv(os.path.join(OUT, "compare_table.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map")
    ap.add_argument("--traffic", choices=["old", "mixed"], default="mixed")
    ap.add_argument("--methods", nargs="+", default=list(MODELS) + ["fixed_program", "max_pressure", "advanced_mp"])
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    compare() if a.compare else evaluate(a.map, a.traffic, a.methods)
