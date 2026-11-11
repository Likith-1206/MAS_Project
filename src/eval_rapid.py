"""Head-to-head evaluation of every controller on identical traffic (4 scenarios x test seeds 9001-9003).

  python src/eval_rapid.py maps/nyc16 --mults 1.0 1.5 2.0 [--yellow 3]

Methods: fixed_program, max_pressure, independent DQN (models/<map>/independent_s*.pt), advanced_mp (+shield),
rapid (results/<map>/rapid/rapid_s*.pt), rapid_gate (rapid + per-light Hedge regret gate vs advanced_mp).
Headline metrics exactly as in env_map.metrics(); not_entered = scheduled - departed, as in run_mappo_map.compare().
Extra columns: system_travel_h = total_travel_h + vehicle-hours spent waiting to enter the map (cannot be gamed by
blocking entries), time_loss_s (SUMO tripinfo, finished vehicles), wait_p95_s / wait_max_s (SUMO waitingTime),
max_red_s (longest red of a phase with stopped vehicles), switches_per_light, starvation_viol (phases red > 60 s
with stopped vehicles, counted once per episode-phase).
Writes results/<map>/rapid/eval_all[_y<yellow>].csv and prints the summary table.
"""
import argparse, os, sys, tempfile
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env_map as E
import scenario_gen as G
import rapid as R
from run_map import MaxPressureMap, MapIndependentDQN

HEAD = ["avg_wait_s", "avg_trip_s", "throughput", "not_entered", "total_travel_h"]
EXTRA = ["system_travel_h", "time_loss_s", "wait_p95_s", "wait_max_s", "max_red_s", "starvation_viol", "switches_per_light"]


class HedgeGate:
    """Per-light no-regret selection between two experts (0 = learned RAPID policy, 1 = Advanced-MP).
    Every WINDOW seconds the light is charged the delay its approaches accumulated under the expert it used
    (bandit feedback); EXP3-style importance-weighted Hedge update. Weights persist across episodes, as a deployed
    controller would keep them."""
    WINDOW = 60

    def __init__(self, agents, eta=0.05, gamma_mix=0.05, seed=0):
        self.w = {a: np.zeros(2) for a in agents}
        self.eta, self.mix = eta, gamma_mix
        self.rng = np.random.default_rng(seed)
        self.choice = {a: 0 for a in agents}; self.prob = {a: np.array([1.0, 0.0]) for a in agents}
        self.acc = {a: 0.0 for a in agents}

    def probs(self, a):
        z = np.exp(self.w[a] - self.w[a].max()); p = z / z.sum()
        return (1 - self.mix) * p + self.mix / 2

    def step(self, env, sensor, t):
        for a in env.agents:
            self.acc[a] += sum(sensor.st[l][3] for l in env.lanes[a]) * E.DELTA
        if t % self.WINDOW == 0:
            for a in env.agents:
                k, p = self.choice[a], self.prob[a][self.choice[a]]
                loss = self.acc[a] / (self.WINDOW * 10.0)          # normalised delay in the window
                self.w[a][k] -= self.eta * loss / max(p, 1e-3)
                self.prob[a] = self.probs(a)
                self.choice[a] = int(self.rng.random() < self.prob[a][1])
                self.acc[a] = 0.0


def run(env, sensor, kind, obj, scn, seed, yellow):
    f = tempfile.mktemp(suffix=".xml")
    own = kind == "own"
    env.reset(scn, seed, tripinfo_file=f, own_programs=own)
    sensor.reset(); sensor.sense()
    sh, amp = R.Shield(sensor), R.AdvancedMP(sensor)
    done, sw, starved = False, 0, set()
    maxred, backlog = 0.0, 0.0
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
        elif kind in ("rapid", "gate"):
            agent = obj if kind == "rapid" else obj[0]
            toks = sensor.tokens()
            prop = agent.act(toks, env.agents)
            if kind == "gate":
                gate, e = obj[1], amp.act()
                prop = {a: (e[a] if gate.choice[a] == 1 else prop[a]) for a in env.agents}
            acts, _ = sh.filter(prop)
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
        if kind == "gate":
            obj[1].step(env, sensor, env.t)
    m = env.metrics()
    env.close()
    m = R.extended(m, f)
    os.remove(f)
    sched = len(G.generate(env.info, scn, seed, G.scale_for(env.map_dir, scn) * env.demand_mult)[0])
    m.update(backlog_h=backlog / 3600.0, system_travel_h=m["total_travel_h"] + backlog / 3600.0,
             not_entered=sched - m["departed"], max_red_s=maxred, starvation_viol=len(starved),
             switches_per_light=sw / len(env.agents) if kind != "own" else np.nan)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map_dir")
    ap.add_argument("--mults", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    ap.add_argument("--yellow", type=int, default=2)
    ap.add_argument("--tag", default="rapid")
    ap.add_argument("--drop", type=float, default=0.0, help="probability that a neighbour message is lost")
    ap.add_argument("--methods", nargs="+",
                    default=["fixed_program", "max_pressure", "independent", "advanced_mp", "rapid", "rapid_gate"])
    a = ap.parse_args()
    E.YELLOW = a.yellow
    name = os.path.basename(os.path.normpath(a.map_dir))
    rdir = os.path.join(R.ROOT, "results", name, "rapid")
    mdir = os.path.join(R.ROOT, "models", name)
    rows = []
    for mult in a.mults:
        env = E.SumoMapEnv(a.map_dir, demand_mult=mult)
        sensor = R.Sensor(env, drop=a.drop)
        jobs = []
        if "fixed_program" in a.methods: jobs.append(("fixed_program", -1, "own", None))
        if "max_pressure" in a.methods: jobs.append(("max_pressure", -1, "mp", MaxPressureMap()))
        if "advanced_mp" in a.methods: jobs.append(("advanced_mp", -1, "amp", None))
        for s in (0, 1, 2):
            p = os.path.join(mdir, f"independent_s{s}.pt")
            if "independent" in a.methods and os.path.exists(p):
                ag = MapIndependentDQN(env); ag.load(torch.load(p)); jobs.append(("independent", s, "idqn", ag))
            p = os.path.join(rdir, f"{a.tag}_s{s}.pt")
            if os.path.exists(p) and ("rapid" in a.methods or "rapid_gate" in a.methods):
                ag = R.RapidAgent(); ag.q.load_state_dict(torch.load(p))
                if "rapid" in a.methods: jobs.append((a.tag, s, "rapid", ag))
                if "rapid_gate" in a.methods: jobs.append((a.tag + "_gate", s, "gate", (ag, HedgeGate(env.agents, seed=s))))
        for meth, s, kind, obj in jobs:
            for scn in E.SCENARIOS:
                for sd in (9001, 9002, 9003):
                    m = run(env, sensor, kind, obj, scn, sd, a.yellow)
                    rows.append(dict(method=meth, train_seed=s, demand_mult=mult, scenario=scn, eval_seed=sd,
                                     **{k: float(v) for k, v in m.items()}))
            print(f"x{mult} {meth} s{s} done", flush=True)
        env.close()
    df = pd.DataFrame(rows)
    suffix = ("" if a.yellow == 2 else f"_y{a.yellow}") + ("" if a.drop == 0 else f"_drop{a.drop:g}")
    mt = "_".join(f"{m:g}" for m in a.mults)
    df.to_csv(os.path.join(rdir, f"eval_all_{a.tag}{suffix}_x{mt}.csv"), index=False)
    summarize(df)


def summarize(df):
    # mean over 12 episodes per training seed, then mean +- std over training seeds (as in the project table)
    per = df.groupby(["demand_mult", "method", "train_seed"])[HEAD + EXTRA].mean()
    agg = per.groupby(["demand_mult", "method"]).agg(["mean", "std"])
    out = pd.DataFrame(index=agg.index)
    for c in HEAD + EXTRA:
        mu, sd = agg[(c, "mean")], agg[(c, "std")].fillna(0)
        out[c] = [f"{m:.1f}" if s == 0 else f"{m:.1f}±{s:.1f}" for m, s in zip(mu, sd)]
    pd.set_option("display.width", 250)
    print(out.to_string())
    return out


if __name__ == "__main__":
    main()
