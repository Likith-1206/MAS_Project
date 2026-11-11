"""Replay a trained controller in sumo-gui.

Put this file in traffic_marl/src/ and run, for example:
  python src/watch.py --method coop --train-seed 0 --scenario rush_hour --seed 9001
Methods: independent | coop | shared_c0 | shared_c1 | shared_c2
Needs the small edit in env.py (SUMO_GUI switch), see the instructions.
"""
import argparse, os, sys

os.environ["SUMO_GUI"] = "1"          # must be set before env.py is imported
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
from env import SumoGridEnv
from agents import IndependentDQN, SharedDQN
import scenarios as S

# neighbour-information level of each method (same values as METHODS in run_experiments.py)
LEVEL = {"independent": None, "coop": 2, "shared_c0": 0, "shared_c1": 1, "shared_c2": 2}

ap = argparse.ArgumentParser()
ap.add_argument("--method", default="coop", choices=list(LEVEL))
ap.add_argument("--train-seed", type=int, default=0, help="which trained model: <method>_s<train-seed>.pt")
ap.add_argument("--scenario", default="rush_hour", choices=S.SCENARIOS)
ap.add_argument("--seed", type=int, default=9001, help="traffic seed (9001-9003 are the test seeds)")
ap.add_argument("--delay", type=int, default=100, help="ms of real time per simulated second (higher = slower)")
a = ap.parse_args()
os.environ["SUMO_GUI_DELAY"] = str(a.delay)

path = os.path.join(S.ROOT, "models", f"{a.method}_s{a.train_seed}.pt")
if not os.path.isfile(path):
    sys.exit(f"model not found: {path}")

env = SumoGridEnv(comm_level=0)       # learned agents always use the level-0 env (as in training)
agent = IndependentDQN(env) if a.method == "independent" else SharedDQN(env, level=LEVEL[a.method])
agent.load(torch.load(path, map_location="cpu"))

obs, done = env.reset(a.scenario, a.seed), False
while not done:
    obs, _, done, _ = env.step(agent.act(obs, 0.0))   # epsilon = 0: greedy, no exploration

print({k: round(float(v), 2) for k, v in env.metrics().items()})
input("Episode finished. Press Enter to close the SUMO window...")
env.close()