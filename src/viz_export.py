"""Building blocks of the visual replay (used by simulate.py).

  scene      static picture of the map: lanes and junctions from the SUMO network, buildings / parks / water / rail
             from the .osm file (projected with the network's own projection, so everything lines up), street names,
             stop-line positions of every signal link, and the agents (controllable lights).
  Recorder   hooks into SumoMapEnv: every simulated second it stores position, heading, speed and type of every vehicle,
             the signal state of every traffic light and a few network-wide numbers (vehicles, speed, halting, arrivals).
  write_html packs one or more recorded runs + the scene into ONE self-contained HTML file (viz_template.html),
             no server and no internet needed: double-click it and it plays.

Vehicle data is stored as compact binary (base64 typed arrays), about 8 bytes per vehicle per second,
so a 20-minute run of a busy map stays a few MB.
"""
import base64, json, math, os, random, zlib
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "viz_template.html")

# vehicle type codes used by the viewer (anything unknown is drawn as a car)
TYPE_CODES = {"car": 0, "moto": 1, "auto": 2, "bus": 3, "truck": 4,
              "two_wheeler": 1, "auto_rickshaw": 2}      # scenario_gen_mixed.py type ids

# Indian-style mixed traffic. All are vClass passenger so every trip can be routed on the imported network
# (osm_import keeps passenger roads); the type changes size, speed and acceleration, and how it is drawn.
FLEET_VTYPES = {
    "car":   '<vType id="car" vClass="passenger" length="4.3" width="1.8" minGap="2.0" accel="2.6" decel="4.5" maxSpeed="13.9" sigma="0.5"/>',
    "moto":  '<vType id="moto" vClass="passenger" length="2.0" width="0.8" minGap="1.0" accel="3.5" decel="6.0" maxSpeed="13.9" sigma="0.6"/>',
    "auto":  '<vType id="auto" vClass="passenger" length="2.8" width="1.4" minGap="1.5" accel="2.0" decel="4.0" maxSpeed="11.0" sigma="0.6"/>',
    "bus":   '<vType id="bus" vClass="passenger" length="11.0" width="2.5" minGap="2.5" accel="1.2" decel="4.0" maxSpeed="11.0" sigma="0.3"/>',
    "truck": '<vType id="truck" vClass="passenger" length="8.0" width="2.4" minGap="2.5" accel="1.3" decel="4.0" maxSpeed="11.0" sigma="0.3"/>',
}
# india: two-wheeler 52 %, car 28 %, auto 15 %, bus 5 % (the field-study mix used in the RAPID-Light experiments)
FLEET_MIX = {"india": [("moto", .52), ("car", .28), ("auto", .15), ("bus", .05)],
             "cars": [("car", 1.0)]}


def apply_fleet(route_file, fleet, seed):
    """Rewrite a route file from scenario_gen.py so its trips use a vehicle mix. Deterministic in seed:
    every controller sees exactly the same vehicles."""
    if fleet == "cars":
        return route_file
    mix = FLEET_MIX[fleet]
    rng = random.Random(f"fleet-{seed}")
    names, w = [m[0] for m in mix], [m[1] for m in mix]
    out = []
    for line in open(route_file):
        s = line.strip()
        if s.startswith("<vType"):
            out += ["  " + v + "\n" for v in FLEET_VTYPES.values()]
        elif s.startswith("<trip"):
            out.append(line.replace('type="car"', f'type="{rng.choices(names, w)[0]}"'))
        else:
            out.append(line)
    with open(route_file, "w") as f:
        f.writelines(out)
    return route_file


# ----------------------------------------------------------------------------------------------- scene
def _r(v):
    return round(float(v), 1)


def _flat(pts, x0, y0):
    out = []
    for x, y in pts:
        out += [_r(x - x0), _r(y - y0)]
    return out


ROAD_CLASS = [("motorway", 5), ("trunk", 4), ("primary", 4), ("secondary", 3), ("tertiary", 2),
              ("residential", 1), ("living_street", 1), ("unclassified", 1), ("service", 0)]


def _road_class(edge_type):
    t = (edge_type or "").lower()
    for k, c in ROAD_CLASS:
        if k in t:
            return c
    return 1


GREEN_TAGS = {("leisure", v) for v in ("park", "garden", "playground", "pitch", "common", "recreation_ground", "golf_course", "nature_reserve")} | \
             {("landuse", v) for v in ("grass", "forest", "meadow", "recreation_ground", "village_green", "cemetery", "orchard")} | \
             {("natural", v) for v in ("wood", "scrub", "grassland")}
WATER_TAGS = {("natural", "water"), ("waterway", "riverbank"), ("landuse", "reservoir"), ("leisure", "swimming_pool"), ("landuse", "basin")}
ZONE_TAGS = {"residential": 1, "commercial": 2, "retail": 2, "industrial": 3, "railway": 4}


def osm_backdrop(osm_path, to_xy, box, margin=80.0):
    """Buildings, green areas, water, zones, rail and canals from the OSM file, in network coordinates.
    to_xy(lon, lat) -> (x, y). box = (xmin, ymin, xmax, ymax); shapes fully outside box+margin are dropped."""
    root = ET.parse(osm_path).getroot()
    nodes = {}
    for n in root.iter("node"):
        nodes[n.get("id")] = (float(n.get("lon")), float(n.get("lat")))
    ways = {}
    for w in root.iter("way"):
        ways[w.get("id")] = ([nd.get("ref") for nd in w.findall("nd")], {t.get("k"): t.get("v") for t in w.findall("tag")})
    cache = {}

    def xy(ref):
        if ref not in cache:
            ll = nodes.get(ref)
            cache[ref] = None if ll is None else to_xy(*ll)
        return cache[ref]

    x0, y0, x1, y1 = box

    def pts(refs):
        p = [xy(r) for r in refs]
        p = [q for q in p if q is not None]
        if len(p) < 2:
            return None
        xs, ys = [q[0] for q in p], [q[1] for q in p]
        if max(xs) < x0 - margin or min(xs) > x1 + margin or max(ys) < y0 - margin or min(ys) > y1 + margin:
            return None
        return p

    out = dict(buildings=[], green=[], water=[], zones=[], rail=[], canals=[])

    def add_area(refs, tags):
        p = pts(refs)
        if not p or len(p) < 3:
            return
        if "building" in tags or "building:part" in tags:
            lv = tags.get("building:levels") or ""
            try:
                h = float(tags["height"].split()[0]) if "height" in tags else float(lv) * 3.2
            except ValueError:
                h = 0
            if not h:   # no height in OSM: a stable pseudo-random 2..8 floors (from the id), bigger footprints a bit taller
                area = abs(sum(p[i][0] * p[i - 1][1] - p[i - 1][0] * p[i][1] for i in range(len(p)))) / 2
                h = 3.2 * (2 + (zlib.crc32(refs[0].encode()) % 5) + min(4, area / 800))
            out["buildings"].append(dict(p=_flat(p, x0, y0), h=round(min(h, 120), 1)))
        elif any((k, tags.get(k)) in WATER_TAGS for k in ("natural", "waterway", "landuse", "leisure")):
            out["water"].append(_flat(p, x0, y0))
        elif any((k, tags.get(k)) in GREEN_TAGS for k in ("leisure", "landuse", "natural")):
            out["green"].append(_flat(p, x0, y0))
        elif tags.get("landuse") in ZONE_TAGS or tags.get("amenity") in ("school", "college", "university", "hospital"):
            z = ZONE_TAGS.get(tags.get("landuse"), 5)
            out["zones"].append(dict(p=_flat(p, x0, y0), z=z))

    for refs, tags in ways.values():
        if tags.get("railway") in ("rail", "subway", "light_rail", "monorail", "narrow_gauge") and tags.get("tunnel") != "yes":
            p = pts(refs)
            if p:
                out["rail"].append(_flat(p, x0, y0))
        elif tags.get("waterway") in ("river", "canal", "drain", "stream", "ditch"):
            p = pts(refs)
            if p:
                out["canals"].append(dict(p=_flat(p, x0, y0), w=8 if tags["waterway"] in ("river", "canal") else 3))
        elif len(refs) >= 4 and refs[0] == refs[-1]:
            add_area(refs, tags)
    # multipolygon relations (big parks, lakes, building complexes): use their outer rings
    for rel in root.iter("relation"):
        tags = {t.get("k"): t.get("v") for t in rel.findall("tag")}
        if tags.get("type") != "multipolygon":
            continue
        for m in rel.findall("member"):
            if m.get("type") == "way" and m.get("role") in ("outer", "") and m.get("ref") in ways:
                refs = ways[m.get("ref")][0]
                if len(refs) >= 4 and refs[0] == refs[-1]:
                    add_area(refs, tags)
    out["buildings"].sort(key=lambda b: -min(b["p"][1::2]))   # far (north) first: correct overlap for 3D buildings
    return out


def build_scene(env, osm_path=None):
    """Static scene of the map for the viewer, in metres relative to the network's lower-left corner."""
    net = env.net
    bx0, by0, bx1, by1 = net.getBoundary()
    x0, y0 = bx0, by0
    lanes, labels, seen_names = [], [], defaultdict(float)
    for e in net.getEdges():
        if e.getFunction() == "internal":
            continue
        rc = _road_class(e.getType())
        for ln in e.getLanes():
            lanes.append(dict(p=_flat(ln.getShape(), x0, y0), w=_r(ln.getWidth()), c=rc,
                              i=ln.getIndex(), n=e.getLaneNumber()))
        name = e.getName() if hasattr(e, "getName") else ""
        if name and e.getLength() > 60:
            shp = e.getShape()
            # label at the middle of the longest segment, readable angle
            k = max(range(len(shp) - 1), key=lambda i: math.dist(shp[i], shp[i + 1]))
            (ax, ay), (bx, by) = shp[k], shp[k + 1]
            ang = math.atan2(by - ay, bx - ax)
            if ang > math.pi / 2 or ang < -math.pi / 2:
                ang += math.pi
            if e.getLength() > seen_names[name] * 0.6:          # avoid repeating the same name on short pieces
                labels.append(dict(t=name, x=_r((ax + bx) / 2 - x0), y=_r((ay + by) / 2 - y0), a=round(ang, 3),
                                   l=_r(e.getLength()), c=rc))
                seen_names[name] = max(seen_names[name], e.getLength())
    labels.sort(key=lambda l: (-l["c"], -l["l"]))
    junctions = []
    for n in net.getNodes():
        shp = n.getShape()
        if shp and len(shp) >= 3:
            junctions.append(_flat(shp, x0, y0))
    # stop lines: per controlled incoming lane, its end point, direction, width and the link indices it carries
    signals = {}
    for tls in net.getTrafficLights():
        per_lane = defaultdict(list)
        lane_obj = {}
        for inl, outl, idx in tls.getConnections():
            per_lane[inl.getID()].append(idx)
            lane_obj[inl.getID()] = inl
        items = []
        for lid, idxs in per_lane.items():
            shp = lane_obj[lid].getShape()
            (ax, ay), (bx, by) = shp[-2], shp[-1]
            items.append(dict(x=_r(bx - x0), y=_r(by - y0), a=round(math.atan2(by - ay, bx - ax), 3),
                              w=_r(lane_obj[lid].getWidth()), k=sorted(set(idxs))))
        signals[tls.getID()] = items
    agents = []
    for a in env.agents:
        cx, cy = env.centroid[a]
        agents.append(dict(id=a, x=_r(cx - x0), y=_r(cy - y0), n=env.n_acts[a], dirs=env.phase_dirs[a],
                           nb=env.neighbors[a], lanes=len(env.lanes[a])))
    scene = dict(w=_r(bx1 - bx0), h=_r(by1 - by0), lanes=lanes, junctions=junctions, labels=labels[:400],
                 signals=signals, agents=agents, backdrop=None)
    if osm_path and os.path.isfile(osm_path):
        try:
            net.convertLonLat2XY(*net.convertXY2LonLat((bx0 + bx1) / 2, (by0 + by1) / 2))   # needs pyproj + geo net
            scene["backdrop"] = osm_backdrop(osm_path, net.convertLonLat2XY, (bx0, by0, bx1, by1))
            scene["backdrop"]["box"] = [0, 0, scene["w"], scene["h"]]
        except Exception as ex:                                     # no projection: draw the roads only
            print(f"[viz] no buildings/parks layer ({type(ex).__name__}: {ex}); install pyproj to get it")
    return scene, (x0, y0)


# ----------------------------------------------------------------------------------------------- recorder
class Recorder:
    """Call snap(traci, t) once per simulated second. Collects everything the viewer animates."""

    def __init__(self, origin, extent):
        self.x0, self.y0 = origin
        self.q = max(extent) / 65000.0 + 1e-9                       # metres per position unit (uint16)
        self.vidx, self.vtype = {}, []
        self.off, self.vid, self.xs, self.ys, self.ang, self.spd = [0], [], [], [], [], []
        self.tls_last, self.tls = {}, defaultdict(list)
        self.kpi = defaultdict(list)
        self.dep, self.arrived, self.halt_vs = 0, 0, 0.0
        self.steps = []                                              # one entry per 5 s decision (filled by simulate.py)

    def snap(self, traci, t):
        V = traci.vehicle
        n, sp_sum, halted = 0, 0.0, 0
        for v in V.getIDList():
            i = self.vidx.get(v)
            if i is None:
                i = self.vidx[v] = len(self.vtype)
                self.vtype.append(TYPE_CODES.get(V.getTypeID(v).split("@")[0], 0))
            x, y = V.getPosition(v)
            s = V.getSpeed(v)
            self.vid.append(i)
            self.xs.append(int(min(65535, max(0, (x - self.x0) / self.q))))
            self.ys.append(int(min(65535, max(0, (y - self.y0) / self.q))))
            self.ang.append(int(round(V.getAngle(v) % 360 * 256 / 360)) % 256)
            self.spd.append(int(min(255, max(0, s * 8))))
            n += 1
            sp_sum += s
            halted += s < 0.1
        self.off.append(len(self.vid))
        for tl in traci.trafficlight.getIDList():
            st = traci.trafficlight.getRedYellowGreenState(tl)
            if self.tls_last.get(tl) != st:
                self.tls[tl].append([t, st])
                self.tls_last[tl] = st
        self.dep += traci.simulation.getDepartedNumber()
        self.arrived += traci.simulation.getArrivedNumber()
        self.halt_vs += halted
        k = self.kpi
        k["n"].append(n)
        k["spd"].append(round(sp_sum / n * 3.6, 1) if n else 0.0)    # km/h
        k["halt"].append(halted)
        k["arr"].append(self.arrived)
        k["wait"].append(round(self.halt_vs / max(1, self.dep), 2))  # halting seconds per departed vehicle so far

    def pack(self, name, label, metrics, extra=None):
        big = len(self.vtype) > 65535
        b64 = lambda a, dt: base64.b64encode(np.asarray(a, dtype=dt).tobytes()).decode()
        return dict(name=name, label=label, T=len(self.off) - 1, q=self.q, vbig=big,
                    off=b64(self.off, "<u4"), vid=b64(self.vid, "<u4" if big else "<u2"),
                    x=b64(self.xs, "<u2"), y=b64(self.ys, "<u2"), a=b64(self.ang, "u1"), s=b64(self.spd, "u1"),
                    vtype=b64(self.vtype, "u1"), tls=dict(self.tls), kpi=dict(self.kpi), steps=self.steps,
                    metrics={k: (None if (isinstance(v, float) and math.isnan(v)) else round(float(v), 2)) for k, v in metrics.items()},
                    **(extra or {}))


def write_html(path, title, scene, runs, meta):
    """One self-contained HTML file: template + scene + runs."""
    data = json.dumps(dict(title=title, scene=scene, runs=runs, meta=meta), separators=(",", ":"))
    html = open(TEMPLATE, encoding="utf-8").read()
    html = html.replace("__TITLE__", title).replace("/*__DATA__*/null", data.replace("</", "<\\/"))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path
