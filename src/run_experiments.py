"""Train / evaluate all controllers.  Examples:
  python src/run_experiments.py baselines
  python src/run_experiments.py train --method coop --episodes 50 --seed 0
Methods: independent | coop (shared net + neighbour obs level 2 + reward mixing) | shared_c0 | shared_c1 | shared_c2 (ablation, no reward mixing)
"""
import argparse, datetime, json, os, subprocess, sys, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import SumoGridEnv
import scenarios as S
from controllers import FixedTime, MaxPressure
from agents import IndependentDQN, SharedDQN, mix_rewards

ROOT = S.ROOT
EVAL_SEEDS = [9001, 9002, 9003]      # fixed test seeds (Interface 3); never used in training
METHODS = {  # name: (agent kind, communication level built by neighbors.py, reward-mixing beta)
    "independent": ("ind", 0, 0.0), "coop": ("shared", 2, 0.5),
    "shared_c0": ("shared", 0, 0.0), "shared_c1": ("shared", 1, 0.0), "shared_c2": ("shared", 2, 0.0)}
EPS_START, EPS_END, EPS_DECAY_FRAC = 0.6, 0.05, 0.6   # linear decay, reaches EPS_END at 60% of the episodes


def train_seed(seed, ep):
    """Traffic seed of training episode `ep` of training run `seed`."""
    return 1000 * seed + ep


def check_seed_split(seed, episodes):
    """Training traffic must never equal a test seed (otherwise results look too good)."""
    used = {train_seed(seed, ep) for ep in range(episodes)}
    clash = used & set(EVAL_SEEDS)
    assert not clash, f"training seeds {sorted(clash)} overlap the test seeds {EVAL_SEEDS}"


def hyperparams(agent):
    """Read the learning settings back from the agent, so the log shows what was really used."""
    d = next(iter(agent.nets.values())) if hasattr(agent, "nets") else agent.net
    return dict(lr=d.opt.param_groups[0]["lr"], gamma=d.gamma, batch=d.batch, target_every=d.target_every,
                min_buf=d.min_buf, replay_cap=d.buf.cap, hidden=d.q[0].out_features, obs_dim=d.q[0].in_features)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def run_eval_episode(env, act_fn, scn, seed, trace=False):
    obs = env.reset(scn, seed); done = False; tr = []
    while not done:
        obs, r, done, info = env.step(act_fn(obs))
        if trace:
            tr.append({"t": env.t, "queue": info["queue"], "phase": dict(env.cur)})
    return env.metrics(), tr


def eval_policy(name, env, act_fn, train_seed=-1, trace_scn="rush_hour"):
    rows = []
    for scn in S.SCENARIOS:
        for sd in EVAL_SEEDS:
            m, tr = run_eval_episode(env, act_fn, scn, sd, trace=(scn == trace_scn and sd == EVAL_SEEDS[0]))
            rows.append(dict(method=name, train_seed=train_seed, scenario=scn, eval_seed=sd, **{k: float(v) for k, v in m.items()}))
            if tr:
                json.dump({"method": name, "scenario": scn, "agents": env.agents, "steps": tr},
                          open(os.path.join(ROOT, "traces", f"{name}_s{train_seed}.json"), "w"))
    return rows


def baselines():
    env = SumoGridEnv(comm_level=0)
    rows = []
    for ctl in (FixedTime(30), MaxPressure()):
        rows += eval_policy(ctl.name, env, lambda obs, c=ctl: c.act(env, obs))
    env.close()
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "eval_baselines.csv"), index=False)
    print("baselines done")


def train(method, episodes, seed):
    kind, level, beta = METHODS[method]
    check_seed_split(seed, episodes)
    for d in ("results", "models", "traces"):
        os.makedirs(os.path.join(ROOT, d), exist_ok=True)
    np.random.seed(seed); torch.manual_seed(seed)
    env = SumoGridEnv(comm_level=0)         # learned agents always get the level-0 env; neighbors.py adds the rest
    agent = IndependentDQN(env) if kind == "ind" else SharedDQN(env, level=level)
    curve, t0 = [], time.time()
    for ep in range(episodes):
        eps = max(EPS_END, EPS_START * (1 - ep / (EPS_DECAY_FRAC * episodes)))
        scn = S.SCENARIOS[ep % len(S.SCENARIOS)]
        obs = env.reset(scn, train_seed(seed, ep)); done, ret = False, 0.0
        while not done:
            act = agent.act(obs, eps)
            obs2, r, done, _ = env.step(act)
            rm = mix_rewards(env, r, beta)
            # `done` at 1200 s is only a time limit, not a real end state: do not cut off the bootstrap
            agent.store(obs, act, rm, obs2, 0.0); agent.learn()
            ret += np.mean(list(r.values())); obs = obs2      # return is logged on the raw (unmixed) reward
        m = env.metrics()
        curve.append(dict(episode=ep, scenario=scn, eps=eps, ret=ret, avg_queue=float(m["avg_queue"]), avg_wait_s=float(m["avg_wait_s"])))
        print(f"[{method} s{seed}] ep {ep+1}/{episodes} {scn:10s} eps={eps:.2f} return={ret:8.1f} avg_wait={m['avg_wait_s']:.1f} ({time.time()-t0:.0f}s)", flush=True)
    tag = f"{method}_s{seed}"
    pd.DataFrame(curve).to_csv(os.path.join(ROOT, "results", f"curve_{tag}.csv"), index=False)
    torch.save(agent.state_dict(), os.path.join(ROOT, "models", f"{tag}.pt"))
    train_time = time.time() - t0
    rows = eval_policy(method, env, lambda obs: agent.act(obs, 0.0), train_seed=seed)
    env.close()
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", f"eval_{tag}.csv"), index=False)
    settings = dict(method=method, seed=seed, episodes=episodes, agent=kind, comm_level=level, reward_mix_beta=beta,
                    eps_start=EPS_START, eps_end=EPS_END, eps_decay_frac=EPS_DECAY_FRAC, learn_steps_per_decision=1,
                    train_seed_formula="1000*seed+episode", eval_seeds=EVAL_SEEDS, scenarios=list(S.SCENARIOS),
                    **hyperparams(agent), git_commit=git_commit(), train_seconds=round(train_time),
                    finished=datetime.datetime.now().isoformat(timespec="seconds"))
    json.dump(settings, open(os.path.join(ROOT, "results", f"settings_{tag}.json"), "w"), indent=2)
    print(f"[{method} s{seed}] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["baselines", "train"])
    ap.add_argument("--method", default="coop", choices=list(METHODS))
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    baselines() if a.mode == "baselines" else train(a.method, a.episodes, a.seed)