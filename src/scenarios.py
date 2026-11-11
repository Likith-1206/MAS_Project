"""Road network (3x3 grid) and traffic-demand scenario generation for SUMO."""
import os, subprocess, shutil
import sumolib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMO_DIR = os.path.join(ROOT, "sumo_files")
NET_FILE = os.path.join(SUMO_DIR, "grid.net.xml")
EPISODE_LEN = 1200  # simulated seconds per episode

# Demand in vehicles/hour per entry edge. Each entry is a dict side -> rate, optionally piecewise in time.
# sides: left/right = east-west traffic, top/bottom = north-south traffic; "row1" = the middle row.
SCENARIOS = ["uniform", "asymmetric", "rush_hour", "corridor"]


def ensure_network(grid_n=3, length=200):
    if os.path.exists(NET_FILE):
        return NET_FILE
    os.makedirs(SUMO_DIR, exist_ok=True)
    exe = shutil.which("netgenerate") or os.path.join(os.environ.get("SUMO_HOME", ""), "bin", "netgenerate")
    subprocess.check_call([exe, "--grid", f"--grid.number={grid_n}", f"--grid.length={length}",
                           f"--grid.attach-length={length}", "--default.lanenumber=2",
                           "--default-junction-type=traffic_light", "--tls.guess",
                           f"--output-file={NET_FILE}", "--no-warnings"])
    return NET_FILE


def _entry_exit_edges():
    net = sumolib.net.readNet(NET_FILE)
    bnd = ("left", "right", "top", "bottom")
    entries, exits = [], []
    for e in net.getEdges():
        f, t = e.getFromNode().getID(), e.getToNode().getID()
        if f.startswith(bnd):
            entries.append((e.getID(), f))
        if t.startswith(bnd):
            exits.append((e.getID(), t))
    return sorted(entries), sorted(exits)


def _rate(scn, node_id, t0, t1):
    """Vehicles per hour entering at boundary node `node_id` during [t0,t1)."""
    side = "".join(c for c in node_id if c.isalpha())
    idx = int("".join(c for c in node_id if c.isdigit()))
    ew = side in ("left", "right")
    if scn == "uniform":
        return 260
    if scn == "asymmetric":
        return 420 if ew else 120
    if scn == "rush_hour":  # NS-heavy -> balanced -> EW-heavy
        phase = min(2, int(3 * t0 / EPISODE_LEN))
        heavy, light = 420, 130
        if phase == 0:
            return heavy if not ew else light
        if phase == 1:
            return 270
        return heavy if ew else light
    if scn == "corridor":  # one heavy east-west corridor (middle row)
        return 560 if (ew and idx == 1) else 150
    raise ValueError(scn)


DEMAND_SCALE = 2.2


def build_route_file(scn, path, scale=None):
    """Write a SUMO route file with Bernoulli (~Poisson) arrivals from every entry to every other exit."""
    scale = DEMAND_SCALE if scale is None else scale
    ensure_network()
    entries, exits = _entry_exit_edges()
    segs = [(0, EPISODE_LEN)] if scn != "rush_hour" else [(i * EPISODE_LEN // 3, (i + 1) * EPISODE_LEN // 3) for i in range(3)]
    lines = ['<routes>', '  <vType id="car" accel="2.6" decel="4.5" length="5" minGap="2.5" maxSpeed="13.9" sigma="0.5"/>']
    k = 0
    for (t0, t1) in segs:                      # SUMO needs flows sorted by begin time -> segments outermost
        for (eid, node) in entries:
            dests = [d for d in exits if _opposite_ok(node, d[1])]
            rate = _rate(scn, node, t0, t1) * scale
            p = rate / 3600.0 / len(dests)
            for (xid, _) in dests:
                lines.append(f'  <flow id="f{k}" type="car" begin="{t0}" end="{t1}" from="{eid}" to="{xid}" '
                             f'probability="{p:.6f}" departLane="best" departSpeed="max"/>')
                k += 1
    lines.append('</routes>')
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def _opposite_ok(entry_node, exit_node):
    """Exclude the exit attached to the same junction as the entry (U-turn)."""
    return entry_node != exit_node and _attached(entry_node) != _attached(exit_node)


_ATT = None


def _attached(bnode):
    global _ATT
    if _ATT is None:
        net = sumolib.net.readNet(NET_FILE)
        _ATT = {}
        for n in net.getNodes():
            if n.getID().startswith(("left", "right", "top", "bottom")):
                nb = [e.getToNode().getID() for e in n.getOutgoing()] + [e.getFromNode().getID() for e in n.getIncoming()]
                _ATT[n.getID()] = nb[0]
    return _ATT[bnode]
