"""Which directions get green for action 0 and action 1? Asks SUMO itself (no assumptions)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import SumoGridEnv, DIRS
env = SumoGridEnv()
for p, dirs in enumerate(env.phase_dirs):
    print(f"action {p}: green for traffic coming from {[DIRS[d] for d in dirs]}   (SUMO state at B1: {env.green['B1'][p]})")
print("same at all 9 intersections (checked in env.__init__)")
env.close()
