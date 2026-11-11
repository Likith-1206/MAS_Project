"""Builds a small synthetic OpenStreetMap file with irregular junctions, for testing osm_import.py.

4x4 street grid (~220 m blocks). Interior nodes carry traffic signals, plus one mid-block pedestrian signal that must be dropped. Irregularities on purpose:
  - T-junction with a signal on the western border,
  - a diagonal street that makes one interior junction a 5-way,
  - horizontal streets have 2 lanes per direction, vertical ones 1,
  - one one-way street.
"""
import os


def make_test_osm(path, step=0.002, lat0=10.0, lon0=10.0):
    nid, nodes, tags = 100, {}, {}
    for r in range(4):
        for c in range(4):
            nid += 1
            nodes[(r, c)] = (nid, lat0 + r * step, lon0 + c * step)
    signals = {(1, 1), (1, 2), (2, 1), (2, 2), (1, 0)}          # (1,0) is the border T-junction
    for k in signals:
        tags[nodes[k][0]] = {"highway": "traffic_signals"}
    stub_nodes = {}                       # roads that continue off the map, ending in a dead-end node
    for r in range(4):
        for side, c in (("W", -1), ("E", 4)):
            if (r, side) == (1, "W"):     # keep (1,0) a T-junction
                continue
            nid += 1
            stub_nodes[(r, c)] = (nid, lat0 + r * step, lon0 + c * step)
    for c in range(4):
        for side, r in (("S", -1), ("N", 4)):
            nid += 1
            stub_nodes[(r, c)] = (nid, lat0 + r * step, lon0 + c * step)
    nodes.update(stub_nodes)
    ped_id = 950                          # pedestrian-crossing signal in the middle of a block (not a junction)
    nodes["M"] = (ped_id, lat0 + 3 * step, lon0 + 1.5 * step)
    tags[ped_id] = {"highway": "traffic_signals"}
    diag_id = 900
    nodes["D"] = (diag_id, lat0 + 0.4 * step, lon0 + 0.4 * step)  # between (0,0) and (1,1)

    ways, wid = [], 5000
    for r in range(4):        # horizontal streets
        wid += 1
        cols = [c for c in (-1, 0, 1, 2, 3, 4) if (r, c) in nodes]
        refs = [nodes[(r, c)][0] for c in cols]
        if r == 3:
            refs.insert(refs.index(nodes[(3, 2)][0]), ped_id)     # between (3,1) and (3,2)
        ways.append((wid, refs,
                     {"highway": "secondary", "lanes": "4", "name": f"East Street {r}"}))
    for c in range(4):        # vertical streets
        wid += 1
        t = {"highway": "residential", "name": f"North Street {c}"}
        if c == 3:
            t["oneway"] = "yes"
        ways.append((wid, [nodes[(r, c)][0] for r in (-1, 0, 1, 2, 3, 4)], t))
    wid += 1                  # diagonal: (0,0) -> D -> (1,1)  => (1,1) becomes a 5-way junction
    ways.append((wid, [nodes[(0, 0)][0], diag_id, nodes[(1, 1)][0]], {"highway": "tertiary", "name": "Diagonal"}))

    out = ['<?xml version="1.0" encoding="UTF-8"?>', '<osm version="0.6" generator="make_test_osm">']
    for key, (i, la, lo) in nodes.items():
        t = tags.get(i)
        if t:
            out.append(f'  <node id="{i}" lat="{la:.7f}" lon="{lo:.7f}" version="1">')
            out += [f'    <tag k="{k}" v="{v}"/>' for k, v in t.items()]
            out.append('  </node>')
        else:
            out.append(f'  <node id="{i}" lat="{la:.7f}" lon="{lo:.7f}" version="1"/>')
    for w, refs, t in ways:
        out.append(f'  <way id="{w}" version="1">')
        out += [f'    <nd ref="{r}"/>' for r in refs]
        out += [f'    <tag k="{k}" v="{v}"/>' for k, v in t.items()]
        out.append('  </way>')
    out.append('</osm>')
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    open(path, "w").write("\n".join(out))
    return path


if __name__ == "__main__":
    import sys
    print(make_test_osm(sys.argv[1] if len(sys.argv) > 1 else "test_area.osm"))