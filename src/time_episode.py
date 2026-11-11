"""How long does one 1200 s episode take on this computer?"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import SumoGridEnv
env = SumoGridEnv(); env.reset("uniform", 1)
for sc in ["uniform", "asymmetric", "rush_hour", "corridor"]:
    t0 = time.time(); env.reset(sc, 5); done = False
    while not done:
        _, _, done, _ = env.step({a: 0 for a in env.agents})
    print(f"{sc:11s} one episode (240 steps = 1200 sim s): {time.time()-t0:.1f} s")
env.close()
