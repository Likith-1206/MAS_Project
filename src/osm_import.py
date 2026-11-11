"""OpenStreetMap -> SUMO network, plus an inspection report and limit checks.

  python src/osm_import.py area.osm --out maps/area1
  python src/osm_import.py area.osm --out maps/area1 --max-tls 16 --max-approaches 5 --max-lanes 4
  python src/osm_import.py area.osm --out maps/full --scan 600          # list 600 m windows with 8-16 controllable lights
  python src/osm_import.py area.osm --out maps/crop1 --bbox=W,S,E,N      # crop (lon/lat) and import only that part

Writes  <out>/net.net.xml  (the SUMO network)  and  <out>/map_report.json  (what is in it, and any problems).
The report is the input for the later stages (scenario generator, environment): for every traffic light it lists
the approaches, lanes and green phases that SUMO built, so nothing has to be assumed about junction shapes.
"""
import argparse, json, os, shutil, subprocess, sys
from collections import defaultdict

DEFAULT_LIMITS = dict(min_tls=1, max_tls=16, max_approaches=10, max_lanes_per_approach=4)   # limits apply to CONTROLLABLE lights

NETCONVERT_OPTS = [
    "--tls.guess-signals", "--tls.join", "--junctions.join",   # signals near a junction control it; merge junction clusters
    "--geometry.remove", "--roundabouts.guess", "--ramps.guess",
    "--keep-edges.by-vclass", "passenger",                      # cars only: drops footpaths, rail, bike paths
    "--remove-edges.isolated", "--keep-edges.components", "1",  # keep only the largest connected road network
    "--tls.default-type", "static",                             # fixed program, our agents set the phase themselves
    "--output.street-names", "--no-warnings",
]


def find_netconvert():
    exe = shutil.which("netconvert")
    if exe:
        return exe
    try:
        import sumo
        cand = os.path.join(sumo.SUMO_HOME, "bin", "netconvert")
        if os.path.exists(cand):
            return cand
    except Exception:
        pass
    raise RuntimeError("netconvert not found: pip install eclipse-sumo")


def convert(osm_path, out_dir, bbox=None, extra_opts=None):
    """Run netconvert on an .osm file; returns the path of the SUMO network.
    extra_opts: more netconvert options, e.g. ["--lefthand"] for left-hand traffic (India, UK, Japan ...)."""
    if not os.path.isfile(osm_path):
        raise FileNotFoundError(
            f"OSM file not found: {osm_path!r}. Download an area first, e.g. "
            "curl -o area.osm \"https://overpass-api.de/api/map?bbox=WEST,SOUTH,EAST,NORTH\" "
            "or use Export on openstreetmap.org.")
    os.makedirs(out_dir, exist_ok=True)
    net = os.path.join(out_dir, "net.net.xml")
    cmd = [find_netconvert(), "--osm-files", osm_path, "-o", net] + NETCONVERT_OPTS
    if bbox:                                                      # crop: keep only edges inside WEST,SOUTH,EAST,NORTH
        cmd += ["--keep-edges.in-geo-boundary", ",".join(str(v) for v in bbox)]
    cmd += list(extra_opts or [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(net):
        raise RuntimeError("netconvert failed:\n" + (r.stderr or r.stdout)[-1500:])
    return net


def _green_phases(program):
    """Phases that give some movement green and no yellow: the ones an agent can choose between."""
    out = []
    for i, ph in enumerate(program.getPhases()):
        s = ph.state
        if any(ch in "Gg" for ch in s) and not any(ch in "yY" for ch in s):
            out.append(i)
    return out


def inspect_network(net_path):
    """Describe every traffic light (approaches, lanes, green phases) and the network as a whole."""
    import sumolib
    net = sumolib.net.readNet(net_path, withPrograms=True)
    tls_list = []
    for tls in net.getTrafficLights():
        conns = tls.getConnections()                      # (inLane, outLane, linkIndex)
        approaches = defaultdict(set)
        for inl, _, _ in conns:
            approaches[inl.getEdge().getID()].add(inl.getID())
        progs = tls.getPrograms()
        prog = next(iter(progs.values())) if progs else None
        phases = [dict(index=i, state=p.state, duration=p.duration) for i, p in enumerate(prog.getPhases())] if prog else []
        green = _green_phases(prog) if prog else []
        nodes = [n.getID() for n in tls.getNodes()] if hasattr(tls, "getNodes") else []
        pts = [inl.getEdge().getToNode().getCoord() for inl, _, _ in conns]
        x = sum(p[0] for p in pts) / len(pts) if pts else 0.0
        y = sum(p[1] for p in pts) / len(pts) if pts else 0.0
        try:
            lon, lat = net.convertXY2LonLat(x, y) if pts else (0.0, 0.0)
        except RuntimeError:                                   # no geo-projection (or pyproj missing): keep x/y only
            lon, lat = None, None
        tls_list.append(dict(
            id=tls.getID(), nodes=nodes, n_links=len(conns),
            controllable=len(green) >= 2,                      # an agent needs a real choice between green phases
            x=round(x, 1), y=round(y, 1), lon=None if lon is None else round(lon, 6), lat=None if lat is None else round(lat, 6),
            n_approaches=len(approaches),
            approaches={e: sorted(l) for e, l in sorted(approaches.items())},
            max_lanes_per_approach=max((len(v) for v in approaches.values()), default=0),
            n_phases=len(phases), green_phases=green, n_green_phases=len(green), phases=phases))
    # Boundary edges: where traffic can enter or leave the map (needed by the scenario generator).
    # A road that leaves the cropped area ends in a dead-end node (one neighbour, both directions), or in a
    # one-way source / sink. Note: dead-end side streets inside the area also count; they act as parking.
    def dead_end(n):
        nb = {e.getFromNode().getID() for e in n.getIncoming()} | {e.getToNode().getID() for e in n.getOutgoing()}
        return len(nb) <= 1 or not n.getIncoming() or not n.getOutgoing()
    def leads_on(e):          # an entry must lead somewhere: its far end has a road other than the way back
        return any(o.getToNode() != e.getFromNode() for o in e.getToNode().getOutgoing())
    def comes_from(e):        # an exit must be reachable: its near end has a road other than the way back
        return any(i.getFromNode() != e.getToNode() for i in e.getFromNode().getIncoming())
    cand_in = [e for e in net.getEdges() if e.allows("passenger") and dead_end(e.getFromNode()) and leads_on(e)]
    cand_out = [e for e in net.getEdges() if e.allows("passenger") and dead_end(e.getToNode()) and comes_from(e)]
    # keep only real boundary roads: an entry must have a path to some exit, and an exit must be reachable from some entry
    # (this drops stubs and tiny cul-de-sacs that no traffic can cross the map through)
    reach = set()
    for a in cand_in:
        for b in cand_out:
            if a is not b and not (b.getFromNode() == a.getToNode() and b.getToNode() == a.getFromNode()):
                path, _ = net.getShortestPath(a, b, vClass="passenger")
                if path:
                    reach.add((a.getID(), b.getID()))
    entries = sorted({a for a, _ in reach})
    exits = sorted({b for _, b in reach})
    x0, y0, x1, y1 = net.getBBoxXY()[0] + net.getBBoxXY()[1]
    n_ctrl = sum(t["controllable"] for t in tls_list)
    return dict(n_edges=len(net.getEdges()), n_nodes=len(net.getNodes()), n_tls=len(tls_list),
                n_controllable=n_ctrl, n_fixed=len(tls_list) - n_ctrl,
                bbox_m=[round(x1 - x0), round(y1 - y0)],
                entry_edges=sorted(entries), exit_edges=sorted(exits), traffic_lights=tls_list)


def check_limits(report, limits=None):
    """Return a list of problems (errors stop the pipeline, warnings are noted).
    The limits apply to controllable lights; fixed-program lights (crossings, one-way stubs) are only counted."""
    L = dict(DEFAULT_LIMITS, **(limits or {}))
    probs = []
    add = lambda level, msg: probs.append(dict(level=level, message=msg))
    ctrl = [t for t in report["traffic_lights"] if t["controllable"]]
    n = len(ctrl)
    if n < L["min_tls"]:
        add("error", f"only {n} controllable traffic light(s) found; need at least {L['min_tls']}. "
                     f"({report['n_fixed']} more stay on a fixed program.) Check the area has signalised junctions.")
    if n > L["max_tls"]:
        add("error", f"{n} controllable traffic lights; the limit is {L['max_tls']}. Choose a smaller area (use --scan / --bbox).")
    if not report["entry_edges"] or not report["exit_edges"]:
        add("error", "no entry or exit edges: traffic cannot enter or leave the map.")
    for t in ctrl:
        if t["n_approaches"] > L["max_approaches"]:
            add("error", f"traffic light {t['id']} has {t['n_approaches']} approaches (limit {L['max_approaches']}).")
        if t["max_lanes_per_approach"] > L["max_lanes_per_approach"]:
            add("error", f"traffic light {t['id']} has an approach with {t['max_lanes_per_approach']} lanes (limit {L['max_lanes_per_approach']}).")
    if report["n_fixed"]:
        add("info", f"{report['n_fixed']} traffic light(s) with fewer than 2 green phases stay on their fixed program.")
    return probs


def scan_windows(net_path, report, size_m=600, step_m=100, lo=8, hi=16, top=8, max_approaches=None):
    """Slide a square window over the map; list non-overlapping windows holding lo..hi controllable lights
    and no light above the approach limit.
    Returns dicts with the lon/lat box to pass to --bbox."""
    try:
        import pyproj  # noqa: F401  (sumolib needs it to convert metres to lon/lat)
    except ImportError:
        raise RuntimeError("--scan needs pyproj: pip install pyproj (and add it to requirements.txt)")
    import sumolib
    net = sumolib.net.readNet(net_path)
    max_app = max_approaches or DEFAULT_LIMITS["max_approaches"]
    pts = [(t["x"], t["y"]) for t in report["traffic_lights"] if t["controllable"]]
    bad = [(t["x"], t["y"]) for t in report["traffic_lights"]        # lights over the approach limit: avoid windows with them
           if t["controllable"] and t["n_approaches"] > max_app]
    (x0, y0), (x1, y1) = net.getBoundary()[:2], net.getBoundary()[2:]
    cands, x = [], x0
    while x <= x1 - size_m + step_m:
        y = y0
        while y <= y1 - size_m + step_m:
            n = sum(1 for (px, py) in pts if x <= px < x + size_m and y <= py < y + size_m)
            has_bad = any(x - 30 <= px < x + size_m + 30 and y - 30 <= py < y + size_m + 30 for (px, py) in bad)
            if lo <= n <= hi and not has_bad:
                cands.append((n, x, y))
            y += step_m
        x += step_m
    cands.sort(key=lambda c: (-c[0], c[1], c[2]))
    out = []
    for n, x, y in cands:
        if all(abs(x - o["x"]) >= size_m or abs(y - o["y"]) >= size_m for o in out):    # non-overlapping
            w, s = net.convertXY2LonLat(x, y)
            e, nn = net.convertXY2LonLat(x + size_m, y + size_m)
            out.append(dict(n_controllable=n, x=x, y=y, bbox=[round(w, 6), round(s, 6), round(e, 6), round(nn, 6)]))
        if len(out) >= top:
            break
    return out


def import_osm(osm_path, out_dir, limits=None, bbox=None, extra_opts=None):
    net = convert(osm_path, out_dir, bbox, extra_opts)
    rep = inspect_network(net)
    rep["problems"] = check_limits(rep, limits)
    rep["ok"] = not any(p["level"] == "error" for p in rep["problems"])
    rep["source"] = os.path.abspath(osm_path)
    rep["bbox"] = bbox
    rep["net_file"] = os.path.abspath(net)
    json.dump(rep, open(os.path.join(out_dir, "map_report.json"), "w"), indent=2)
    return rep


def summary(rep):
    lines = [f"network: {rep['n_edges']} edges, {rep['n_nodes']} nodes, {rep['bbox_m'][0]} x {rep['bbox_m'][1]} m, "
             f"{len(rep['entry_edges'])} entry / {len(rep['exit_edges'])} exit edges",
             f"traffic lights: {rep['n_tls']} ({rep['n_controllable']} controllable, {rep['n_fixed']} fixed-program)"]
    for t in rep["traffic_lights"]:
        if not t["controllable"]:
            continue
        lines.append(f"  {t['id'][:40]:>40}: {t['n_approaches']} approaches, up to {t['max_lanes_per_approach']} lanes each, "
                     f"{t['n_green_phases']} green phases of {t['n_phases']}")
    for p in rep["problems"]:
        lines.append(f"  [{p['level'].upper()}] {p['message']}")
    lines.append("RESULT: " + ("ok" if rep["ok"] else "map rejected (see errors)"))
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("osm"); ap.add_argument("--out", required=True)
    ap.add_argument("--min-tls", type=int); ap.add_argument("--max-tls", type=int)
    ap.add_argument("--max-approaches", type=int); ap.add_argument("--max-lanes", type=int)
    ap.add_argument("--bbox", help="WEST,SOUTH,EAST,NORTH in degrees: crop to this box")
    ap.add_argument("--scan", type=int, metavar="SIZE_M", help="after import, list windows of SIZE_M metres with 8-16 controllable lights")
    a = ap.parse_args()
    bbox = [float(v) for v in a.bbox.split(",")] if a.bbox else None
    lim = {k: v for k, v in dict(min_tls=a.min_tls, max_tls=a.max_tls, max_approaches=a.max_approaches,
                                 max_lanes_per_approach=a.max_lanes).items() if v is not None}
    r = import_osm(a.osm, a.out, lim, bbox)
    if a.scan:
        print(f"network has {r['n_controllable']} controllable lights; windows of {a.scan} m with 8-16 of them:")
        for w in scan_windows(r["net_file"], r, a.scan):
            print(f"  {w['n_controllable']:>2} lights  --bbox={','.join(str(v) for v in w['bbox'])}")
        sys.exit(0)
    print(summary(r)); sys.exit(0 if r["ok"] else 1)