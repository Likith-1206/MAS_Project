"""Compare env.metrics() with SUMO's own trip output (tripinfo) for one episode."""
import os, sys, xml.etree.ElementTree as ET
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import SumoGridEnv
trip = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "tripinfo_check.xml")
env = SumoGridEnv(); env.reset("uniform", 1, tripinfo_file=trip); done = False; i = 0
while not done:
    _, _, done, _ = env.step({a: (i // 6) % 2 for a in env.agents}); i += 1
m = env.metrics(); env.close()
rows = [t.attrib for t in ET.parse(trip).getroot().iter("tripinfo")]; n = len(rows)
dur = sum(float(r["duration"]) for r in rows) / n; wt = sum(float(r["waitingTime"]) for r in rows) / n
print(f"{'':24s}{'ours':>9s}{'SUMO tripinfo':>15s}")
print(f"{'throughput (arrived)':24s}{m['throughput']:>9d}{n:>15d}")
print(f"{'avg trip time (s)':24s}{m['avg_trip_s']:>9.1f}{dur:>15.1f}")
print(f"{'avg waiting (s per car)':24s}{m['avg_wait_s']:>9.1f}{wt:>15.1f}")
os.remove(trip)
