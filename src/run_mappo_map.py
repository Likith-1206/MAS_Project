"""Train and evaluate MAPPO on a map (env_map.SumoMapEnv) and compare it with the baselines of run_map.py.

  python src/run_mappo_map.py maps/nyc16 --episodes 100 --seed 0                        # shared actor, own observation (level 0)
  python src/run_mappo_map.py maps/nyc16 --episodes 100 --seed 0 --level 2 --beta 0.5   # + neighbour info, reward mixing
  python src/run_mappo_map.py maps/nyc16 --baselines                                    # only the baselines

Shares everything with run_map.py so the results can be compared: same output folders (results/<map>/, models/<map>/),
same test seeds (run_experiments.EVAL_SEEDS), same training-seed rule, and the SAME baseline file eval_baselines.csv
(fixed_program = netconvert's own programs, max_pressure), which this script asks run_map.py to create if it is missing.
Adds one extra baseline of its own, in eval_fixed30.csv:  fixed_30 = each light cycles its green phases, 30 s each.
Writes  curve_<tag>.csv, eval_<tag>.csv, settings_<tag>.json  (tag = mappo[_c<level>]_s<seed>)  and  models/<map>/<tag>.pt.

Extra column in the comparison:  not_entered = scheduled vehicles that never got into the map (entries blocked). A controller
that jams the entries can look good on waiting time, so read it next to throughput and avg_trip_s. It is computed the same way
for every method (scheduled trips of that scenario+seed, minus `departed`), so it also works for files written by run_map.py.
"""
import argparse, datetime, json, os, sys, time
import numpy as np, pandas as pd, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import env_map as E
import scenario_gen as G
import run_map as RM
from run_experiments import ROOT, EVAL_SEEDS, train_seed, check_seed_split, git_commit
from mappo_map import MAPPO

SHOW = ["avg_wait_s", "avg_trip_s", "throughput", "not_entered", "total_travel_h"]


def mix_rewards(env, r, beta):
    """Own reward + beta * mean reward of the light's neighbours (a light with no neighbour keeps its own reward)."""
    if beta == 0:
        return r
    return {a: r[a] + beta * (float(np.mean([r[n] for n in env.neighbors[a]])) if env.neighbors[a] else 0.0) for a in env.agents}


class FixedCycle:
    """Each light cycles through its own green phases, `green_s` seconds each."""
    def __init__(self, green_s=30):
        self.green_s = green_s
        self.name = f"fixed_{green_s}"

    def act(self, env, obs):
        return {a: int((env.t // self.green_s) % env.n_acts[a]) for a in env.agents}


def ensure_baselines(env, map_dir, d):
    """eval_baselines.csv from run_map.py (created if missing) and our extra fixed_30 in eval_fixed30.csv."""
    if not os.path.exists(os.path.join(d["results"], "eval_baselines.csv")):
        RM.baselines(map_dir)
    f30 = os.path.join(d["results"], "eval_fixed30.csv")
    if not os.path.exists(f30):
        fc = FixedCycle(30)
        pd.DataFrame(RM.eval_policy(fc.name, env, lambda o: fc.act(env, o))).to_csv(f30, index=False)
        print("fixed_30 baseline done", flush=True)


def compare(map_dir, d, tag):
    """Mean over the test seeds, per scenario and overall: this MAPPO run next to the baselines (same seeds only)."""
    parts = [pd.read_csv(os.path.join(d["results"], n)) for n in ("eval_baselines.csv", "eval_fixed30.csv", f"eval_{tag}.csv")
             if os.path.exists(os.path.join(d["results"], n))]
    df = pd.concat(parts, ignore_index=True)
    seeds = set.intersection(*[set(p.eval_seed) for p in parts])
    if any(set(p.eval_seed) != seeds for p in parts):
        print(f"WARNING: the files use different test seeds; comparing only the common ones: {sorted(seeds)}")
    df = df[df.eval_seed.isin(seeds)].copy()
    info = G.MapInfo(map_dir)
    sched = {(s, sd): len(G.generate(info, s, int(sd), G.scale_for(map_dir, s))[0]) for s in E.SCENARIOS for sd in seeds}
    df["scheduled"] = [sched[(s, int(sd))] for s, sd in zip(df.scenario, df.eval_seed)]
    df["not_entered"] = df.scheduled - df.departed
    print(f"\nmean over test seeds {sorted(int(s) for s in seeds)}  (lower is better, except throughput):")
    print(df.groupby(["scenario", "method"])[SHOW].mean().round(1).to_string())
    print("\nall scenarios together:\n" + df.groupby("method")[SHOW].mean().round(1).to_string())


def main(a):
    d = RM._dirs(a.map_dir, None)
    env = E.SumoMapEnv(a.map_dir)
    ensure_baselines(env, a.map_dir, d)
    if a.baselines:
        env.close(); return
    method = "mappo" if a.level == 0 else f"mappo_c{a.level}"
    tag = f"{method}_s{a.seed}"
    check_seed_split(a.seed, a.episodes)
    np.random.seed(a.seed); torch.manual_seed(a.seed)
    agent = MAPPO(env, level=a.level)
    curve, t0 = [], time.time()
    for ep in range(a.episodes):
        scn = E.SCENARIOS[ep % len(E.SCENARIOS)]
        obs = env.reset(scn, train_seed(a.seed, ep)); done, ret = False, 0.0
        while not done:
            act = agent.act(obs, explore=True)
            obs, r, done, _ = env.step(act)
            agent.store(mix_rewards(env, r, a.beta))
            ret += np.mean(list(r.values()))                   # return logged on the raw reward
        st = agent.update(obs)                                 # episode ended by the time limit: bootstrap from the last observation
        m = env.metrics()
        curve.append(dict(episode=ep, scenario=scn, ret=ret, avg_queue=float(m["avg_queue"]), avg_wait_s=float(m["avg_wait_s"]), **st))
        print(f"[{method} s{a.seed}] ep {ep+1}/{a.episodes} {scn:10s} return={ret:8.1f} avg_wait={m['avg_wait_s']:.1f} "
              f"ent={st['entropy']:.2f} ({time.time()-t0:.0f}s)", flush=True)
    train_time = time.time() - t0
    pd.DataFrame(curve).to_csv(os.path.join(d["results"], f"curve_{tag}.csv"), index=False)
    torch.save(agent.state_dict(), os.path.join(d["models"], f"{tag}.pt"))
    rows = RM.eval_policy(method, env, lambda o: agent.act(o, explore=False), train_seed_=a.seed, traces_dir=d["traces"])
    pd.DataFrame(rows).to_csv(os.path.join(d["results"], f"eval_{tag}.csv"), index=False)
    env.close()
    settings = dict(method=method, map=os.path.basename(os.path.normpath(a.map_dir)), seed=a.seed, episodes=a.episodes,
                    comm_level=a.level, reward_mix_beta=a.beta, train_seed_formula="1000*seed+episode", eval_seeds=EVAL_SEEDS,
                    scenarios=list(E.SCENARIOS), n_agents=len(env.agents), demand_scales={p: G.scale_for(a.map_dir, p) for p in E.SCENARIOS},
                    **agent.hyperparams(), git_commit=git_commit(), train_seconds=round(train_time),
                    finished=datetime.datetime.now().isoformat(timespec="seconds"))
    json.dump(settings, open(os.path.join(d["results"], f"settings_{tag}.json"), "w"), indent=2)
    print(f"[{method} s{a.seed}] done in {time.time()-t0:.0f}s", flush=True)
    compare(a.map_dir, d, tag)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map_dir")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--level", type=int, default=0, choices=[0, 1, 2])
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--baselines", action="store_true", help="only create the baseline files")
    main(ap.parse_args())