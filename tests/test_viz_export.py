"""Tests for the visual-replay export (no SUMO needed): fleet rewrite, recorder packing, HTML writing."""
import base64, json, os, re, sys, types
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import viz_export as V


def fake_traci(vehicles, tls):
    veh = types.SimpleNamespace(getIDList=lambda: list(vehicles), getTypeID=lambda v: vehicles[v][0],
                                getPosition=lambda v: vehicles[v][1], getSpeed=lambda v: vehicles[v][2],
                                getAngle=lambda v: vehicles[v][3])
    tl = types.SimpleNamespace(getIDList=lambda: list(tls), getRedYellowGreenState=lambda i: tls[i])
    sim = types.SimpleNamespace(getDepartedNumber=lambda: 1, getArrivedNumber=lambda: 0)
    return types.SimpleNamespace(vehicle=veh, trafficlight=tl, simulation=sim)


def test_fleet_is_deterministic(tmp_path):
    src = '<routes>\n  <vType id="car"/>\n' + "".join(f'  <trip id="t{i}" type="car" depart="{i}.0" from="a" to="b"/>\n' for i in range(200)) + "</routes>\n"
    a, b = tmp_path / "a.xml", tmp_path / "b.xml"
    a.write_text(src); b.write_text(src)
    V.apply_fleet(str(a), "india", 9001); V.apply_fleet(str(b), "india", 9001)
    assert a.read_text() == b.read_text()
    types_used = set(re.findall(r'type="(\w+)"', a.read_text()))
    assert {"car", "moto", "auto"} <= types_used and a.read_text().count("<vType") == 5


def test_recorder_roundtrip(tmp_path):
    rec = V.Recorder((100.0, 200.0), (500.0, 400.0))
    vs = {"x": ("bus", (150.0, 250.0), 5.0, 90.0), "y": ("car@1", (600.0, 600.0), 0.0, 180.0)}
    tl = {"J1": "GGrr"}
    rec.snap(fake_traci(vs, tl), 0)
    tl["J1"] = "yyrr"; rec.snap(fake_traci(vs, tl), 1)
    run = rec.pack("m", "Model", dict(avg_wait_s=1.0, avg_trip_s=float("nan")))
    off = np.frombuffer(base64.b64decode(run["off"]), "<u4")
    xs = np.frombuffer(base64.b64decode(run["x"]), "<u2")
    assert list(off) == [0, 2, 4] and run["T"] == 2
    assert abs(xs[0] * run["q"] - 50.0) < 0.05
    assert run["tls"]["J1"] == [[0, "GGrr"], [1, "yyrr"]]
    assert list(np.frombuffer(base64.b64decode(run["vtype"]), "u1")) == [3, 0]
    assert run["metrics"]["avg_trip_s"] is None and run["kpi"]["halt"] == [1, 1]


def test_write_html(tmp_path):
    out = V.write_html(str(tmp_path / "v.html"), "T", dict(w=1, h=1, lanes=[], junctions=[], labels=[], signals={}, agents=[], backdrop=None),
                       [], dict(note="</script>"))
    s = open(out).read()
    assert "/*__DATA__*/" not in s and "<\\/script>" in s
    data = json.loads(re.search(r"const DATA = (\{.*?\});\n", s, re.S).group(1).replace("<\\/", "</"))
    assert data["meta"]["note"] == "</script>"
