"""RAPID v2 trained on mixed traffic.

Exactly the RAPID v2 algorithm of rapid.py (plain vehicle counts, delay reward, same network, learner, shield and
settings), but the training episodes use the mixed Indian-style traffic of scenario_gen_mixed.py instead of the
single-car-type traffic. rapid.py is not modified: its environment class is swapped inside this process only.

  python src/rapid_v2_mixed.py train   maps/nyc16 --seed 0 --episodes 400
  python src/rapid_v2_mixed.py eval    maps/nyc16 --traffic mixed      # and --traffic old
  python src/rapid_v2_mixed.py compare maps/nyc16

Outputs (new folder only): results/<map>/rapid_v2_mixed/
  rapid2_mixedtrained_s<seed>.pt, curve_*.csv, settings_*.json, eval_<traffic>_x1.csv, compare_table.csv
"""
import argparse, glob, os, sys
import pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env_map as E
import rapid as R
import rapid_mixed as RM
import eval_mixed as EM
from eval_rapid import HEAD

TAG = "rapid2_mixedtrained"


class _MixedE:
    SumoMapEnv = RM.MixedMapEnv

    def __getattr__(self, name):
        return getattr(E, name)


def out_dir(map_dir):
    d = os.path.join(R.ROOT, "results", os.path.basename(os.path.normpath(map_dir)), "rapid_v2_mixed")
    os.makedirs(d, exist_ok=True)
    return d


def train(map_dir, seed, episodes):
    old = R.E
    R.E = _MixedE()                      # mixed traffic; Sensor stays the plain (count-based) RAPID v2 sensor
    try:
        R.train(map_dir, seed, episodes, 0.0, out_dir(map_dir), tag=TAG)
    finally:
        R.E = old


def evaluate(map_dir, traffic):
    d = out_dir(map_dir)
    env = (RM.MixedMapEnv if traffic == "mixed" else E.SumoMapEnv)(map_dir)
    sensor = R.Sensor(env)
    rows = []
    for s in (0, 1, 2):
        ag = R.RapidAgent(); ag.q.load_state_dict(torch.load(os.path.join(d, f"{TAG}_s{s}.pt")))
        for scn in E.SCENARIOS:
            for sd in (9001, 9002, 9003):
                m = EM.run(env, sensor, "rapid", ag, scn, sd)
                rows.append(dict(traffic=traffic, method=TAG, train_seed=s, demand_mult=1.0, scenario=scn,
                                 eval_seed=sd, **{k: float(v) for k, v in m.items()}))
        print(f"{traffic} s{s} done", flush=True)
    env.close()
    pd.DataFrame(rows).to_csv(os.path.join(d, f"eval_{traffic}_x1.csv"), index=False)


def compare(map_dir):
    name = os.path.basename(os.path.normpath(map_dir))
    rdir = os.path.join(R.ROOT, "results", name, "rapid")
    mdir = os.path.join(R.ROOT, "results", name, "rapid_mixed")
    d = out_dir(map_dir)
    old = pd.concat([pd.read_csv(os.path.join(rdir, "eval_all_rapid2_x1.csv")),
                     pd.read_csv(os.path.join(rdir, "eval_all_base.csv")).query("method == 'fixed_program'"),
                     pd.read_csv(os.path.join(mdir, "eval_old_rapid_mixed_x1.csv"))])
    old["traffic"] = "old"
    new = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(mdir, "eval_mixed*_x1.csv"))])
    mine = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(d, "eval_*_x1.csv"))])
    df = pd.concat([old, new, mine])
    df = df[df.demand_mult == 1.0]
    cols = HEAD + ["time_loss_s", "wait_p95_s", "starvation_viol"]
    t = EM.table(df, cols).reset_index()
    order = ["fixed_program", "max_pressure", "independent", "advanced_mp", "rapid2", "rapid_mixed", TAG]
    t["o"] = t.method.map({m: i for i, m in enumerate(order)})
    t = t.sort_values(["o", "traffic"], ascending=[True, False]).drop(columns=["o", "demand_mult"])
    pd.set_option("display.width", 250)
    print(t.to_string(index=False))
    mixed = df[df.traffic == "mixed"]
    tt = EM.table(mixed, EM.TYPE_COLS[:4]).reset_index().drop(columns=["demand_mult", "traffic"])
    tt["o"] = tt.method.map({m: i for i, m in enumerate(order)}); tt = tt.sort_values("o").drop(columns="o")
    print(tt.to_string(index=False))
    t.to_csv(os.path.join(d, "compare_table.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["train", "eval", "compare"])
    ap.add_argument("map_dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--traffic", choices=["mixed", "old"], default="mixed")
    a = ap.parse_args()
    {"train": lambda: train(a.map_dir, a.seed, a.episodes), "eval": lambda: evaluate(a.map_dir, a.traffic),
     "compare": lambda: compare(a.map_dir)}[a.cmd]()
