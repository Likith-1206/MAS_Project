"""Multi-city RAPID v2: one shared model trained on several maps at once, on mixed traffic.

Same algorithm as RAPID v2 (rapid.py: plain count-based detectors, delay reward incl. vehicles waiting to enter,
PhaseNet, shared Double DQN with 3-step returns, Advanced-MP demonstrations, guided exploration, LTL shield).
Differences:
  * each training episode uses a randomly chosen map from TRAIN_MAPS;
  * traffic is the mixed Indian-style traffic of scenario_gen_mixed.py (rapid_mixed.MixedMapEnv);
  * rapid.MAXP is raised (in this process only) to the largest green-phase count over ALL_MAPS, so every
    controllable light of every map is controlled by the model (none left on fixed time).
Pune (maps/pune_test) is a held-out test city and is never used here.

  python src/rapid_multicity.py train --seed 0 --episodes 400
Outputs (new folder only): results/multicity/rapid_multicity_s<seed>.pt, curve_*.csv, settings_*.json
"""
import argparse, json, os, sys, time
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env_map as E
import rapid as R
import rapid_mixed as RM

TRAIN_MAPS = ["nyc16", "bangalore", "jersey", "mumbai"]
ALL_MAPS = TRAIN_MAPS + ["pune_test"]
TAG = "rapid_multicity"
OUT = os.path.join(R.ROOT, "results", "multicity")


def max_phases(maps):
    n = 0
    for m in maps:
        rep = json.load(open(os.path.join(R.ROOT, "maps", m, "map_report.json")))
        n = max(n, max(t["n_green_phases"] for t in rep["traffic_lights"] if t["controllable"]))
    return n


def train(seed, episodes, demo_eps=12, mult_range=(0.5, 2.0), p_nominal=0.4, drop=0.2):
    R.MAXP = max(3, max_phases(ALL_MAPS))
    np.random.seed(seed); torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    envs = {}

    def get_env(m, mult):
        k = (m, round(mult, 2))
        if k not in envs:
            env = RM.MixedMapEnv(os.path.join(R.ROOT, "maps", m), demand_mult=k[1])
            env._sensor = R.Sensor(env, drop=drop, rng=rng)
            envs[k] = env
        return envs[k]

    agent = R.RapidAgent()
    log, t0 = [], time.time()
    total = demo_eps + episodes
    for ep in range(total):
        m = TRAIN_MAPS[rng.integers(len(TRAIN_MAPS))]
        scn = E.SCENARIOS[rng.integers(len(E.SCENARIOS))]
        mult = 1.0 if rng.random() < p_nominal else float(np.round(rng.uniform(*mult_range), 1))
        env = get_env(m, mult)
        if ep < demo_eps:
            res = R.run_episode(env, env._sensor, "amp", scn, 1000 * seed + ep, train_agent=agent, demo=True,
                                record=True, rng=rng)
            if ep == demo_eps - 1:
                for _ in range(4000):
                    agent.learn()
        else:
            frac = (ep - demo_eps) / max(1, episodes)
            agent.demo_w = max(0.0, 1.0 - frac / 0.3)
            res = R.run_episode(env, env._sensor, "rapid", scn, 1000 * seed + ep, train_agent=agent,
                                eps=max(0.02, 0.3 * (1 - frac / 0.6)), record=True, rng=rng)
        env.close()
        log.append(dict(ep=ep, map=m, scenario=scn, mult=mult, **{k: float(v) for k, v in res.items()}))
        print(f"[{TAG} s{seed}] ep {ep+1}/{total} {m:9s} {scn:10s} x{mult:.1f} wait={res['avg_wait_s']:.1f} "
              f"tt={res['total_travel_h']:.1f} thr={res['throughput']} ({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(OUT, exist_ok=True)
    torch.save(agent.q.state_dict(), os.path.join(OUT, f"{TAG}_s{seed}.pt"))
    import pandas as pd
    pd.DataFrame(log).to_csv(os.path.join(OUT, f"curve_{TAG}_s{seed}.csv"), index=False)
    json.dump(dict(method=TAG, seed=seed, episodes=episodes, demo_episodes=demo_eps, train_maps=TRAIN_MAPS,
                   held_out="pune_test", maxp=R.MAXP, traffic="mixed", mult_range=mult_range, p_nominal=p_nominal,
                   drop=drop, train_seconds=round(time.time() - t0)),
              open(os.path.join(OUT, f"settings_{TAG}_s{seed}.json"), "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=400)
    a = ap.parse_args()
    train(a.seed, a.episodes)
