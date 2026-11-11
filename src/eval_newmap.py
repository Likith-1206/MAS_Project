"""Zero-shot evaluation of saved RAPID models on a NEW map (no retraining), against the non-learning baselines.

  python src/eval_newmap.py maps/bangalore --traffic old   --methods fixed_program max_pressure advanced_mp
  python src/eval_newmap.py maps/bangalore --traffic mixed --methods rapid2 rapid2_mixedtrained
  python src/eval_newmap.py maps/bangalore --compare

Models are loaded from the nyc16 training runs (never retrained here):
  rapid2               results/nyc16/rapid/rapid2_s{0,1,2}.pt
  rapid2_mixedtrained  results/nyc16/rapid_v2_mixed/rapid2_mixedtrained_s{0,1,2}.pt
Lights with more than 3 green phases (RAPID was trained with at most 3, rapid.MAXP = 3):
  <model>              controls every light; rapid.MAXP is raised IN THIS PROCESS ONLY to the map's largest
                       phase count (the phase-token network accepts any number of phases; those lights are unseen)
  <model>_le3          lights with > 3 green phases keep their own fixed program, RAPID controls the rest
Baselines (fixed_program, max_pressure, advanced_mp) control every light. Metrics: as eval_mixed.py.
Outputs: results/<map>/rapid_eval/eval_<traffic>_<methods>.csv, compare_table.csv (new folder only).
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

MODELS = {"rapid2": ("rapid", "rapid2"), "rapid2_mixedtrained": ("rapid_v2_mixed", "rapid2_mixedtrained")}


def make_env(map_dir, traffic, le3):
    env = (RM.MixedMapEnv if traffic == "mixed" else E.SumoMapEnv)(map_dir)
    if le3:
        env.agents = [a for a in env.agents if env.n_acts[a] <= 3]      # the others run their own program
    return env


def main_eval(a):
    name = os.path.basename(os.path.normpath(a.map_dir))
    out = os.path.join(R.ROOT, "results", name, "rapid_eval"); os.makedirs(out, exist_ok=True)
    rows = []
    for meth in a.methods:
        le3 = meth.endswith("_le3")
        env = make_env(a.map_dir, a.traffic, le3)
        R.MAXP = max(3, max(env.n_acts[x] for x in env.agents))
        sensor = R.Sensor(env)
        if meth in ("fixed_program", "max_pressure", "advanced_mp"):
            kind = {"fixed_program": "own", "max_pressure": "mp", "advanced_mp": "amp"}[meth]
            jobs = [(-1, kind, MaxPressureMap() if kind == "mp" else None)]
        else:
            folder, tag = MODELS[meth.replace("_le3", "")]
            jobs = []
            for s in (0, 1, 2):
                ag = R.RapidAgent()
                ag.q.load_state_dict(torch.load(os.path.join(R.ROOT, "results", "nyc16", folder, f"{tag}_s{s}.pt")))
                jobs.append((s, "rapid", ag))
        for s, kind, obj in jobs:
            for scn in E.SCENARIOS:
                for sd in (9001, 9002, 9003):
                    m = EM.run(env, sensor, kind, obj, scn, sd)
                    rows.append(dict(traffic=a.traffic, method=meth, train_seed=s, demand_mult=1.0, scenario=scn,
                                     eval_seed=sd, n_controlled=len(env.agents), **{k: float(v) for k, v in m.items()}))
            print(f"{name} {a.traffic} {meth} s{s} done", flush=True)
        env.close()
    pd.DataFrame(rows).to_csv(os.path.join(out, f"eval_{a.traffic}_{'-'.join(a.methods)}.csv"), index=False)


def compare(map_dir):
    name = os.path.basename(os.path.normpath(map_dir))
    out = os.path.join(R.ROOT, "results", name, "rapid_eval")
    df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(out, "eval_*.csv"))])
    cols = HEAD + ["time_loss_s", "wait_p95_s", "starvation_viol"]
    t = EM.table(df, cols).reset_index()
    order = ["fixed_program", "max_pressure", "advanced_mp", "rapid2", "rapid2_le3", "rapid2_mixedtrained",
             "rapid2_mixedtrained_le3"]
    t["o"] = t.method.map({m: i for i, m in enumerate(order)})
    t = t.sort_values(["traffic", "o"], ascending=[False, True]).drop(columns=["o", "demand_mult"])
    pd.set_option("display.width", 250)
    print(t.to_string(index=False))
    mixed = df[df.traffic == "mixed"]
    if len(mixed):
        tt = EM.table(mixed, EM.TYPE_COLS[:4]).reset_index().drop(columns=["demand_mult", "traffic"])
        tt["o"] = tt.method.map({m: i for i, m in enumerate(order)}); print(tt.sort_values("o").drop(columns="o").to_string(index=False))
    t.to_csv(os.path.join(out, "compare_table.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map_dir")
    ap.add_argument("--traffic", choices=["old", "mixed"], default="old")
    ap.add_argument("--methods", nargs="+", default=["fixed_program", "max_pressure", "advanced_mp", "rapid2",
                                                     "rapid2_le3", "rapid2_mixedtrained", "rapid2_mixedtrained_le3"])
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    compare(a.map_dir) if a.compare else main_eval(a)
