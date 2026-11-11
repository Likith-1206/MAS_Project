import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import pytest
from tests.make_test_osm import make_test_osm
import osm_import as O

try:
    O.find_netconvert()
    HAVE_SUMO = True
except RuntimeError:
    HAVE_SUMO = False
pytestmark = pytest.mark.skipif(not HAVE_SUMO, reason="netconvert not installed (pip install eclipse-sumo)")


@pytest.fixture(scope="module")
def imported(tmp_path_factory):
    d = tmp_path_factory.mktemp("osm")
    osm = make_test_osm(str(d / "area.osm"))
    return O.import_osm(osm, str(d / "out")), osm, d


def test_junction_shapes_are_detected(imported):
    rep, _, _ = imported
    assert rep["n_controllable"] == 5 and rep["ok"]
    by_appr = sorted(t["n_approaches"] for t in rep["traffic_lights"] if t["controllable"])
    assert by_appr == [3, 4, 4, 4, 5]                      # T-junction, four 4-way, one 5-way
    ctl = [t for t in rep["traffic_lights"] if t["controllable"]]
    five = next(t for t in ctl if t["n_approaches"] == 5)
    three = next(t for t in ctl if t["n_approaches"] == 3)
    assert five["n_green_phases"] > three["n_green_phases"] >= 2   # phase sets differ per junction: env must not assume 2


def test_entry_exit_edges(imported):
    rep, _, _ = imported
    assert len(rep["entry_edges"]) >= 10 and len(rep["exit_edges"]) >= 10
    assert set(rep["entry_edges"]).isdisjoint(set(rep["exit_edges"]))   # an edge leaves OR enters, not both


def test_limits(imported):
    rep, _, _ = imported
    assert [p for p in O.check_limits(rep) if p["level"] == "error"] == []
    errs = [p["message"] for p in O.check_limits(rep, dict(max_approaches=4)) if p["level"] == "error"]
    assert len(errs) == 1 and "5 approaches" in errs[0]
    assert any("limit is 3" in p["message"] for p in O.check_limits(rep, dict(max_tls=3)))


def test_map_without_signals_is_rejected(tmp_path):
    osm = make_test_osm(str(tmp_path / "a.osm"))
    txt = open(osm).read().replace('v="traffic_signals"', 'v="crossing"')
    open(osm, "w").write(txt)
    rep = O.import_osm(osm, str(tmp_path / "out"))
    assert rep["n_tls"] == 0 and not rep["ok"]
    assert any("traffic light" in p["message"] for p in rep["problems"])


def test_sumo_loads_the_network(imported):
    """The converted network must run in SUMO and expose the same traffic lights as the report."""
    import libsumo as traci
    rep, _, _ = imported
    traci.start(["sumo", "-n", rep["net_file"], "--no-step-log", "true", "--no-warnings", "true"])
    try:
        for _ in range(5):
            traci.simulationStep()
        assert sorted(traci.trafficlight.getIDList()) == sorted(t["id"] for t in rep["traffic_lights"])
        for t in rep["traffic_lights"]:
            assert len(traci.trafficlight.getRedYellowGreenState(t["id"])) == t["n_links"]
    finally:
        traci.close()


def test_pedestrian_signal_stays_as_a_fixed_light(imported):
    """6 signal nodes in OSM: 5 junctions + 1 mid-block crossing. The crossing is kept (it really stops cars) but is not an agent."""
    rep, osm, _ = imported
    assert open(osm).read().count('v="traffic_signals"') == 6
    assert rep["n_tls"] == 6 and rep["n_controllable"] == 5 and rep["n_fixed"] == 1
    fixed = [t for t in rep["traffic_lights"] if not t["controllable"]]
    assert len(fixed) == 1 and fixed[0]["n_green_phases"] < 2


def test_bbox_keeps_only_the_chosen_area(imported, tmp_path):
    _, osm, _ = imported
    # lon 10.0035 lies between the signalised columns (10.002 and 10.004): the eastern two lights fall outside
    rep = O.import_osm(osm, str(tmp_path / "cropped"), bbox=[9.995, 9.99, 10.0035, 10.01])
    assert rep["n_controllable"] == 3 and rep["ok"]


def test_scan_windows_round_trip(imported, tmp_path):
    """A window suggested by --scan, fed back as --bbox, must give a map with a number of lights in the asked range."""
    rep, _, d = imported
    wins = O.scan_windows(rep["net_file"], rep, size_m=700, lo=2, hi=5, top=3)
    assert wins
    again = O.import_osm(rep["source"], str(tmp_path / "w"), bbox=wins[0]["bbox"])
    assert 2 <= again["n_controllable"] <= 5