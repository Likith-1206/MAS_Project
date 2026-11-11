import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from collections import Counter
import pytest
from tests.make_test_osm import make_test_osm
import osm_import as O
import scenario_gen as G

try:
    O.find_netconvert()
    HAVE_SUMO = True
except RuntimeError:
    HAVE_SUMO = False
pytestmark = pytest.mark.skipif(not HAVE_SUMO, reason="netconvert not installed (pip install eclipse-sumo)")

HERE = os.path.dirname(os.path.abspath(__file__))
NYC = os.path.join(HERE, "..", "maps", "nyc16")


@pytest.fixture(scope="module")
def info(tmp_path_factory):
    d = tmp_path_factory.mktemp("gen")
    osm = make_test_osm(str(d / "area.osm"))
    O.import_osm(osm, str(d / "map"))
    return G.MapInfo(str(d / "map"))


def trips(info, pattern, seed=1, scale=1.0):
    return G.generate(info, pattern, seed, scale)


def test_same_inputs_give_identical_file(info, tmp_path):
    a, b, c = (str(tmp_path / n) for n in "abc")
    G.build_route_file(info.dir, "rush_hour", 5, a, 1.0, info)
    G.build_route_file(info.dir, "rush_hour", 5, b, 1.0, info)
    G.build_route_file(info.dir, "rush_hour", 6, c, 1.0, info)
    assert open(a).read() == open(b).read()          # deterministic
    assert open(a).read() != open(c).read()          # a different seed is a different scenario


def test_every_trip_is_a_valid_route(info):
    tr, st = trips(info, "uniform")
    assert tr and st["n_trips"] == len(tr)
    assert [t[0] for t in tr] == sorted(t[0] for t in tr)                       # SUMO needs sorted departures
    assert all(0 <= t[0] < G.EPISODE_LEN for t in tr)
    assert all((a, b) in info.od_ctrl for _, a, b in tr)                         # path exists and crosses >= 1 controllable light
    assert set(a for _, a, _ in tr) <= set(info.report["entry_edges"])
    assert set(b for _, _, b in tr) <= set(info.report["exit_edges"])


def test_no_u_turn_at_the_boundary(info):
    for ein, outs in info.od.items():
        a = info.net.getEdge(ein)
        for eout in outs:
            b = info.net.getEdge(eout)
            assert not (b.getFromNode() == a.getToNode() and b.getToNode() == a.getFromNode())


def test_demand_scales_with_scale(info):
    n1 = trips(info, "uniform", 3, 1.0)[1]["n_trips"]
    n2 = trips(info, "uniform", 3, 2.0)[1]["n_trips"]
    assert 1.6 < n2 / n1 < 2.4


def test_all_patterns_carry_the_same_total_demand(info):
    mean = lambda pat: sum(trips(info, pat, sd)[1]["n_trips"] for sd in range(8)) / 8
    base = mean("uniform")
    for pat in G.PATTERNS:
        assert abs(mean(pat) / base - 1) < 0.08, pat


def _axis_counts(info, pattern, t0, t1, seeds=range(6)):
    """Trips per entry LANE (demand is proportional to lanes; maps rarely have equal entries or lanes on both axes)."""
    c, n_entries = Counter(), Counter()
    for e in info.sources():
        n_entries[G._axis(info.heading(e))] += info.lanes(e)
    for sd in seeds:
        for d, a, _ in trips(info, pattern, sd)[0]:
            if t0 <= d < t1:
                c[G._axis(info.heading(a))] += 1
    return {ax: c[ax] / n_entries[ax] for ax in n_entries}


def test_asymmetric_favours_east_west(info):
    c = _axis_counts(info, "asymmetric", 0, G.EPISODE_LEN)
    assert c["EW"] > 2.5 * c["NS"]


def test_rush_hour_changes_direction_over_time(info):
    third = G.EPISODE_LEN // 3
    first, mid, last = (_axis_counts(info, "rush_hour", i * third, (i + 1) * third) for i in range(3))
    assert first["NS"] > 1.5 * first["EW"]            # NS-heavy
    assert last["EW"] > 1.5 * last["NS"]              # EW-heavy
    assert 0.7 < mid["EW"] / mid["NS"] < 1.4          # balanced in the middle


def test_corridor_concentrates_demand(info):
    tr, st = trips(info, "corridor", 2)
    ein, eout = st["corridor"]
    per = st["per_entry"]
    assert per[ein] > 2 * (sum(per.values()) - per[ein]) / max(1, len(per) - 1)
    from_ein = [t for t in tr if t[1] == ein]
    assert sum(1 for t in from_ein if t[2] == eout) / len(from_ein) > 0.45


def test_unknown_pattern_is_rejected(info):
    with pytest.raises(ValueError):
        G.generate(info, "nonsense", 0)


def test_sumo_accepts_the_routes(info, tmp_path):
    rf = str(tmp_path / "r.rou.xml")
    st = G.build_route_file(info.dir, "uniform", 4, rf, 0.5, info)
    m = G.run_static(info, rf)                                                   # raises if SUMO reports a route error
    assert m["loaded"] == st["n_trips"] and m["inserted"] > 0 and m["arrived"] > 0


def test_calibration_returns_a_working_scale(info):
    res = G.calibrate(info.dir, "uniform", n_seeds=1, iters=4, info=info)
    assert res["ok"] and res["scale"] > 0
    # past the chosen scale the entries must be (more) blocked than at it
    hi = [h for h in res["history"] if h["scale"] > res["scale"]]
    assert hi and max(h["backlog_frac"] for h in hi) >= min(h["backlog_frac"] for h in res["history"] if h["scale"] <= res["scale"])


@pytest.mark.skipif(not os.path.isdir(NYC), reason="maps/nyc16 not present")
def test_interior_dead_ends_are_not_entries():
    info = G.MapInfo(NYC)
    d = info.describe()
    assert d["n_entries"] == 7 and "46721847#0" not in info.report["entry_edges"]   # stub that leads nowhere is no entry
    assert "5704673#0" in d["dropped_entries"]                                  # interior side street, ~190 m from the edge


def test_calibrate_all_gives_a_scale_per_pattern(info, tmp_path):
    res = G.calibrate_all(info.dir, n_seeds=1, iters=2, info=info)
    assert set(res["scales"]) == set(G.PATTERNS) and all(v > 0 for v in res["scales"].values())
    import json, shutil
    d = tmp_path / "m"; shutil.copytree(info.dir, d)
    json.dump(res, open(d / "demand.json", "w"))
    assert G.scale_for(str(d), "corridor") == res["scales"]["corridor"]


def test_scale_for_reads_the_old_single_scale_file(tmp_path):
    import json
    json.dump({"scale": 1.142, "ok": True}, open(tmp_path / "demand.json", "w"))
    assert G.scale_for(str(tmp_path), "rush_hour") == 1.142