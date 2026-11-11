"""Baselines and independent DQN on ANY imported map (environment v2, env_map.py).

  python src/run_map.py baselines maps/nyc16
  python src/run_map.py train maps/nyc16 --episodes 120 --seed 0
  python src/run_map.py stress maps/nyc16 --mults 1.0 1.5 2.0 --seeds 0 1 2    # same controllers, more traffic

Controllers compared on the same traffic (4 scenarios x test seeds 9001-9003):
  fixed_program   the network's own fixed-time signal programs (netconvert's), no control from us
  max_pressure    every 5 s, serve the phase with the highest pressure (halting in minus halting out, over the
                  phase's real movements), stay on ties
  independent     one DQN per light, each with its own observation size and its own number of phases

Outputs go to results/<map name>/, models/<map name>/, traces/<map name>/ so they never mix with the 3x3 results:
  eval_baselines.csv, curve_independent_s<seed>.csv, eval_independent_s<seed>.csv, settings_independent_s<seed>.json
Same columns as the 3x3 files (method, train_seed, scenario, eval_seed, avg_wait_s, ...).
"""
import argparse, datetime, json, os, sys, time
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env_map as E
from agents import DQN
from run_experiments import ROOT, EVAL_SEEDS, train_seed, check_seed_split, git_commit, EPS_START, EPS_END, EPS_DECAY_FRAC


class MaxPressureMap:
    name = "max_pressure"

    def act(self, env, obs):
        lane = E.traci.lane
        out = {}
        for a in env.agents:
            press = [sum(lane.getLastStepHaltingNumber(i) - lane.getLastStepHaltingNumber(o) for i, o in links)
                     for links in env.phase_links[a]]
            best = int(np.argmax(press))
            out[a] = env.cur[a] if press[best] <= press[env.cur[a]] else best     # stay on ties
        return out


class MapIndependentDQN:
    """One DQN per light. Each network has the input size and the number of outputs of ITS light."""

    def __init__(self, env):
        self.ag = list(env.agents)
        self.nets = {a: DQN(env.obs_dims[a], env.n_acts[a]) for a in self.ag}

    def act(self, obs, eps):
        return {a: int(self.nets[a].act(obs[a], eps)[0]) for a in self.ag}

    def store(self, o, act, r, o2, done):
        for a in self.ag:
            self.nets[a].buf.add_many([o[a]], [act[a]], [r[a]], [o2[a]], float(done))

    def learn(self):
        for a in self.ag:
            self.nets[a].learn()

    def state_dict(self):
        return {a: n.q.state_dict() for a, n in self.nets.items()}

    def load(self, sd):
        for a, n in self.nets.items():
            n.q.load_state_dict(sd[a]); n.qt.load_state_dict(sd[a])

    def hyperparams(self):
        n = next(iter(self.nets.values()))
        return dict(lr=n.opt.param_groups[0]["lr"], gamma=n.gamma, batch=n.batch, target_every=n.target_every,
                    min_buf=n.min_buf, replay_cap=n.buf.cap, hidden=n.q[0].out_features,
                    obs_dims={a: m.q[0].in_features for a, m in self.nets.items()},
                    n_acts={a: m.n_act for a, m in self.nets.items()})


def _dirs(map_dir, out_root):
    name = os.path.basename(os.path.normpath(map_dir))
    root = out_root or ROOT
    d = {k: os.path.join(root, k, name) for k in ("results", "models", "traces")}
    for p in d.values():
        os.makedirs(p, exist_ok=True)
    return d


def run_eval_episode(env, act_fn, scn, seed, own=False, trace=False):
    obs, done, tr = env.reset(scn, seed, own_programs=own), False, []
    while not done:
        obs, r, done, info = env.step(None if own else act_fn(obs))
        if trace:
            tr.append({"t": env.t, "queue": info["queue"], "phase": dict(info["phase"])})
    return env.metrics(), tr


def eval_policy(name, env, act_fn, train_seed_=-1, own=False, trace_scn="rush_hour", traces_dir=None):
    rows = []
    for scn in E.SCENARIOS:
        for sd in EVAL_SEEDS:
            want = traces_dir is not None and scn == trace_scn and sd == EVAL_SEEDS[0]
            m, tr = run_eval_episode(env, act_fn, scn, sd, own=own, trace=want)
            rows.append(dict(method=name, train_seed=train_seed_, scenario=scn, eval_seed=sd,
                             **{k: float(v) for k, v in m.items()}))
            if want:
                json.dump({"method": name, "scenario": scn, "agents": env.agents, "steps": tr},
                          open(os.path.join(traces_dir, f"{name}_s{train_seed_}.json"), "w"))
    return rows


def baselines(map_dir, out_root=None):
    d = _dirs(map_dir, out_root)
    env = E.SumoMapEnv(map_dir)
    mp = MaxPressureMap()
    rows = eval_policy("fixed_program", env, None, own=True, traces_dir=d["traces"])
    rows += eval_policy(mp.name, env, lambda obs: mp.act(env, obs), traces_dir=d["traces"])
    env.close()
    pd.DataFrame(rows).to_csv(os.path.join(d["results"], "eval_baselines.csv"), index=False)
    print("baselines done ->", d["results"])


def train(map_dir, episodes, seed, out_root=None):
    d = _dirs(map_dir, out_root)
    check_seed_split(seed, episodes)
    np.random.seed(seed); torch.manual_seed(seed)
    env = E.SumoMapEnv(map_dir)
    agent = MapIndependentDQN(env)
    curve, t0 = [], time.time()
    for ep in range(episodes):
        eps = max(EPS_END, EPS_START * (1 - ep / (EPS_DECAY_FRAC * episodes)))
        scn = E.SCENARIOS[ep % len(E.SCENARIOS)]
        obs, done, ret = env.reset(scn, train_seed(seed, ep)), False, 0.0
        while not done:
            act = agent.act(obs, eps)
            obs2, r, done, _ = env.step(act)
            agent.store(obs, act, r, obs2, 0.0)       # 1200 s is a time limit, not a real end state
            agent.learn()
            ret += np.mean(list(r.values())); obs = obs2
        m = env.metrics()
        curve.append(dict(episode=ep, scenario=scn, eps=eps, ret=ret, avg_queue=float(m["avg_queue"]),
                          avg_wait_s=float(m["avg_wait_s"])))
        print(f"[independent s{seed}] ep {ep+1}/{episodes} {scn:10s} eps={eps:.2f} return={ret:8.1f} "
              f"avg_wait={m['avg_wait_s']:.1f} ({time.time()-t0:.0f}s)", flush=True)
    tag = f"independent_s{seed}"
    pd.DataFrame(curve).to_csv(os.path.join(d["results"], f"curve_{tag}.csv"), index=False)
    torch.save(agent.state_dict(), os.path.join(d["models"], f"{tag}.pt"))
    train_time = time.time() - t0
    rows = eval_policy("independent", env, lambda o: agent.act(o, 0.0), train_seed_=seed, traces_dir=d["traces"])
    env.close()
    pd.DataFrame(rows).to_csv(os.path.join(d["results"], f"eval_{tag}.csv"), index=False)
    settings = dict(method="independent", map=os.path.basename(os.path.normpath(map_dir)), seed=seed, episodes=episodes,
                    eps_start=EPS_START, eps_end=EPS_END, eps_decay_frac=EPS_DECAY_FRAC,
                    train_seed_formula="1000*seed+episode", eval_seeds=EVAL_SEEDS, scenarios=list(E.SCENARIOS),
                    n_agents=len(env.agents), demand_scales={p: E.G.scale_for(map_dir, p) for p in E.SCENARIOS},
                    **agent.hyperparams(), git_commit=git_commit(), train_seconds=round(train_time),
                    finished=datetime.datetime.now().isoformat(timespec="seconds"))
    json.dump(settings, open(os.path.join(d["results"], f"settings_{tag}.json"), "w"), indent=2)
    print(f"[independent s{seed}] done in {time.time()-t0:.0f}s")


def stress(map_dir, mults, seeds, out_root=None):
    """Evaluate the baselines and the trained independent agents at higher demand (calibrated demand x mult).
    Needs models/<map>/independent_s<seed>.pt from `train`. Writes results/<map>/stress.csv (column demand_mult)."""
    d = _dirs(map_dir, out_root)
    rows = []
    for mult in mults:
        env = E.SumoMapEnv(map_dir, demand_mult=mult)
        mp = MaxPressureMap()
        part = eval_policy("fixed_program", env, None, own=True)
        part += eval_policy(mp.name, env, lambda obs, e=env, m=mp: m.act(e, obs))
        for sd in seeds:
            agent = MapIndependentDQN(env)
            agent.load(torch.load(os.path.join(d["models"], f"independent_s{sd}.pt")))
            part += eval_policy("independent", env, lambda o, a=agent: a.act(o, 0.0), train_seed_=sd)
        for r in part:
            r["demand_mult"] = mult
        rows += part
        env.close()
        print(f"demand x{mult} done", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(d["results"], "stress.csv"), index=False)
    print("stress done ->", d["results"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["baselines", "train", "stress"])
    ap.add_argument("map_dir")
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mults", type=float, nargs="+", default=[1.0, 1.5, 2.0], help="stress: demand multipliers")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="stress: trained models to evaluate")
    a = ap.parse_args()
    if a.cmd == "baselines":
        baselines(a.map_dir)
    elif a.cmd == "train":
        train(a.map_dir, a.episodes, a.seed)
    else:
        stress(a.map_dir, a.mults, a.seeds)