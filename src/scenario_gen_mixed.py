"""Mixed (Indian-style) traffic for any map made by osm_import.py: same demand patterns as scenario_gen.py, plus
vehicle types, a traffic mix from Indian field studies, and driver speed variation. Lane-based driving (no sublane).

  python src/scenario_gen_mixed.py maps/nyc16 --pattern rush_hour --seed 9001 --out /tmp/rush_mixed.rou.xml

Demand patterns: the trips (departure time, entry, exit) are produced by scenario_gen.generate() itself, so for the
same (map, pattern, seed, scale) the mixed file contains EXACTLY the same trips as the single-car-type file; only the
vehicle type of each trip changes. The type is drawn with a separate random stream (seed, pattern, 11), so the trip
list is untouched. This makes "old traffic vs mixed traffic" a like-for-like comparison: same vehicles, same routes,
same times, different vehicles.

-------------------------------------------------------------------------------------------------------------------
Traffic mix (share of vehicles) at Indian urban signalised intersections
-------------------------------------------------------------------------------------------------------------------
  two-wheeler 0.52, car 0.28, auto-rickshaw 0.15, bus 0.05
Field values the mix is built from:
  * Chennai, signalised intersections (Retteri), 12 h counts: ~56 % two-wheelers, 17 % cars, 13 % autos,
    6 % trucks, 6 % others.  (Saturation Flow Model for Signalized Intersection under Mixed Traffic Condition,
    https://www.researchgate.net/publication/327509939)
  * Tiruchirappalli, signalised intersections: two-wheelers 40-50 % of the mix.
    (https://www.researchgate.net/publication/342532741)
  * Hyderabad: two-wheelers 51.6 %, auto-rickshaws 17.2 % (a Hyderabad count found in the literature search; the
    site was not confirmed to be signalised, so it is used only as supporting evidence).
  * Delhi NCR video counts: cars 52-62 %, two-wheelers 18-27 %, autos 8-10 % (a car-heavy upper bound for cars).
    (https://arxiv.org/pdf/1901.06358)
  Two-wheeler 0.52 ~ mean of Chennai / Hyderabad / Tiruchirappalli; auto 0.15 ~ mean of Chennai 13 % and Hyderabad
  17 %; bus 0.05: the studies above give no clean urban bus share (Chennai lists 6 % "others"), so 5 % is an
  assumption; car takes the remainder (0.28, between Chennai 17 % and Delhi 52-62 %). Trucks are left out
  (most Indian cities restrict them in daytime).

-------------------------------------------------------------------------------------------------------------------
Vehicle types (SUMO Krauss model, lane-based)
-------------------------------------------------------------------------------------------------------------------
  type          length  minGap  accel  decel  maxSpeed     speedFactor               vClass
  car           5.0     2.5     2.6    4.5    13.9 (50)    normc(1.00, 0.10, .6, 1.4) passenger  (= project car)
  two_wheeler   1.9     1.0     2.5    4.0    16.7 (60)    normc(1.12, 0.15, .6, 1.6) motorcycle
  auto_rickshaw 2.6     1.5     1.4    3.0    11.1 (40)    normc(0.80, 0.12, .5, 1.2) passenger
  bus           12.0    2.5     1.2    3.5    13.9 (50)    normc(0.90, 0.10, .6, 1.2) bus
  (metres, m/s^2, m/s (km/h))
Sources / reasoning:
  * Lengths: auto-rickshaw 2.6 m from the Chennai mixed-traffic model of https://arxiv.org/pdf/1805.05076
    (Table 1); two-wheeler 1.9 m and 12 m bus are typical dimensions (Indo-HCM 2017 Table 1.3 lists design-vehicle
    dimensions; the exact numbers were not checked against the manual); car = the project's existing car.
  * Acceleration: SUMO "accel" is the MAXIMUM a vehicle can use. Field AVERAGE rates at 24 signalised approaches
    in Hyderabad and Warangal: two-wheeler 0.60, three-wheeler 0.43, car 0.49 m/s^2, deceleration 0.55 / 0.59 /
    0.62 m/s^2 (https://www.academia.edu/95709060); literature range of two-wheeler acceleration 0.55-1.59 and
    auto 0.59-1.16 m/s^2 (https://www.researchgate.net/publication/363231926). The ranking used here follows
    those studies (two-wheeler ~ car > auto > bus); the maxima themselves are assumptions (not calibrated).
    Bus values: close to SUMO's default bus class (no Indian bus value found).
  * Speed variation: the same Hyderabad/Warangal study measured mean speeds two-wheeler 5.87, car 5.24,
    three-wheeler 4.08 m/s, i.e. ratios 1.12 : 1 : 0.78 to the car; these set the speedFactor means. The spread
    (coefficient of variation 0.10-0.15) is an assumption in the usual range for urban speed distributions.
  * sigma (driver imperfection) is SUMO's default 0.5 for all types, as in the project's car type.

-------------------------------------------------------------------------------------------------------------------
Passenger-car units (used by rapid_mixed.py, never by the simulator)
-------------------------------------------------------------------------------------------------------------------
  car 1.0, two-wheeler 0.4, auto-rickshaw 1.2, bus 3.0
  * Two-wheeler 0.4 (range 0.3-0.5) and auto 1.2 (range 1.1-1.3): Indo-HCM 2017 values for signalised
    intersections as reproduced in CSIR-CRRI teaching material found in the literature search (slides, e.g.
    https://www.slideshare.net/slideshow/capacity-level-of-service-highways-signalized-intersections-indohcm/248889022;
    to be checked against the Indo-HCM 2017 manual).
  * Bus 3.0: inside the range 1.02-3.78 measured at Indian signalised intersections ("Dynamic PCU values at
    signalised intersections in India for mixed traffic",
    https://www.researchgate.net/publication/281362799); Indo-HCM's road-segment bus value is 4.5.
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scenario_gen as G
from scenario_gen import MapInfo, PATTERNS, EPISODE_LEN, scale_for   # same patterns, same calibration  # noqa: F401

VTYPES = {
    "car": dict(length=5.0, minGap=2.5, accel=2.6, decel=4.5, maxSpeed=13.9, sigma=0.5,
                speedFactor="normc(1.00,0.10,0.6,1.4)", vClass="passenger", guiShape="passenger"),
    "two_wheeler": dict(length=1.9, minGap=1.0, accel=2.5, decel=4.0, maxSpeed=16.7, sigma=0.5,
                        speedFactor="normc(1.12,0.15,0.6,1.6)", vClass="motorcycle", guiShape="motorcycle"),
    "auto_rickshaw": dict(length=2.6, minGap=1.5, accel=1.4, decel=3.0, maxSpeed=11.1, sigma=0.5,
                          speedFactor="normc(0.80,0.12,0.5,1.2)", vClass="passenger", guiShape="passenger/hatchback"),
    "bus": dict(length=12.0, minGap=2.5, accel=1.2, decel=3.5, maxSpeed=13.9, sigma=0.5,
                speedFactor="normc(0.90,0.10,0.6,1.2)", vClass="bus", guiShape="bus"),
}
MIX = {"two_wheeler": 0.52, "car": 0.28, "auto_rickshaw": 0.15, "bus": 0.05}
PCU = {"car": 1.0, "two_wheeler": 0.4, "auto_rickshaw": 1.2, "bus": 3.0}
TYPES = list(MIX)


def generate(info, pattern, seed, scale=1.0, episode_len=EPISODE_LEN, mix=None):
    """Same trips as scenario_gen.generate, each with a vehicle type. Returns (list of (depart, entry, exit, type), stats)."""
    mix = mix or MIX
    trips, st = G.generate(info, pattern, seed, scale, episode_len)
    rng = np.random.default_rng([int(seed), PATTERNS.index(pattern), 11])
    p = np.array([mix[t] for t in TYPES]); p = p / p.sum()
    kinds = rng.choice(len(TYPES), size=len(trips), p=p)
    out = [(d, a, b, TYPES[k]) for (d, a, b), k in zip(trips, kinds)]
    st = dict(st, types={t: int((kinds == i).sum()) for i, t in enumerate(TYPES)})
    return out, st


def vtype_xml():
    rows = []
    for t, prm in VTYPES.items():
        attrs = " ".join(f'{k}="{v}"' for k, v in prm.items())
        rows.append(f'  <vType id="{t}" {attrs}/>')
    return rows


def write_routes(trips, path):
    lines = ["<routes>"] + vtype_xml()
    for k, (d, a, b, t) in enumerate(trips):
        lines.append(f'  <trip id="t{k}" type="{t}" depart="{d:.1f}" from="{a}" to="{b}" departLane="best" departSpeed="max"/>')
    lines.append("</routes>")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def build_route_file(map_dir, pattern, seed, path, scale=1.0, info=None):
    info = info or MapInfo(map_dir)
    trips, st = generate(info, pattern, seed, scale)
    write_routes(trips, path)
    return st


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map_dir")
    ap.add_argument("--pattern", default="uniform", choices=PATTERNS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=None, help="default: the calibrated scale of the pattern")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    scale = a.scale if a.scale is not None else scale_for(a.map_dir, a.pattern)
    print(build_route_file(a.map_dir, a.pattern, a.seed, a.out, scale))
