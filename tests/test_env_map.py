import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pytest
from tests.make_test_osm import make_test_osm
import osm_import as O
import scenario_gen as G
from env_map import SumoMapEnv, SCENARIOS

try:
    O.find_netconvert()
    HAVE_SUMO = True
except RuntimeError:
    HAVE_SUMO = False
pytestmark = pytest.mark.skipif(not HAVE_SUMO, reason="netconvert not installed (pip install eclipse-sumo)")

HERE = os.path.dirname(os.path.abspath(__file__))
NYC = os.path.join(HERE, "..", "maps", "nyc16")


@pytest.fixture(scope="module")
def mapdir(tmp_path_factory):
    d = tmp_path_factory.mktemp("envmap")
    O.import_osm(make_test_osm(str(d / "area.osm")), str(d / "map"))
    json.dump({"scales": {p: 0.6 for p in G.PATTERNS}, "ok": True}, open(d / "map" / "demand.json", "w"))
    return str(d / "map")


@pytest.fixture(scope="module")
def env(mapdir):
    e = SumoMapEnv(mapdir)
    yield e
    e.close()


def rollout(env, scenario, seed, n=60, own=False):
    env.reset(scenario, seed, own_programs=own)
    for i in range(n):
        env.step(None if own else {a: (i // 4) % env.n_acts[a] for a in env.agents})
    return {k: float(v) for k, v in env.metrics().items()}


def test_agents_have_their_own_shape(env):
    """The test map has a T-junction, four 4-ways and a 5-way: lights must differ in lanes and in phases."""
    assert len(env.agents) == 5
    assert len(set(env.n_acts.values())) >= 2 and len(set(env.obs_dims.values())) >= 2
    for a in env.agents:
        assert env.obs_dims[a] == 2 * len(env.lanes[a]) + env.n_acts[a] + 1 == len(env.obs_names[a])
        assert len(env.phase_links[a]) == env.n_acts[a] >= 2


def test_reset_obs_shapes_and_range(env):
    obs = env.reset("uniform", 1)
    assert set(obs) == set(env.agents)
    for a, v in obs.items():
        assert v.shape == (env.obs_dims[a],) and v.dtype == np.float32 and v.min() >= 0 and v.max() <= 1
    pad = env.pad_obs(obs)
    assert all(p.shape == (env.padded_dim,) for p in pad.values())
    a = next(x for x in env.agents if env.n_acts[x] < env.n_act)           # a light with fewer phases than the max
    k = env.n_acts[a]
    assert pad[a][2 * env.max_lanes + k:2 * env.max_lanes + env.n_act].sum() == 0   # padded phase slots are zero
    assert pad[a][-1] == obs[a][-1]


def test_same_scenario_and_seed_same_traffic(env):
    a, b, c = rollout(env, "rush_hour", 3), rollout(env, "rush_hour", 3), rollout(env, "rush_hour", 4)
    assert a == b and a != c


def test_action_masks_and_invalid_actions(env):
    for a, m in env.action_masks().items():
        assert m.shape == (env.n_act,) and m.sum() == env.n_acts[a] and m[:env.n_acts[a]].all()
    env.reset("uniform", 1)
    bad = {a: 0 for a in env.agents}
    bad[env.agents[0]] = env.n_acts[env.agents[0]]                          # one past the last valid phase
    with pytest.raises(ValueError):
        env.step(bad)
    with pytest.raises(ValueError):
        env.step({})


def test_yellow_reward_info_done(env):
    obs = env.reset("corridor", 2)
    a0, a1 = env.agents[0], env.agents[1]
    acts = {a: 0 for a in env.agents}
    acts[a0] = 1
    _, _, _, info = env.step(acts)
    assert info["yellow"][a0] and not info["yellow"][a1]
    assert info["phase"][a0] == 1 and info["phase"][a1] == 0
    done, n = False, 1
    while not done:
        _, r, done, info = env.step({a: 0 for a in env.agents}); n += 1
        for a in env.agents:
            assert r[a] == pytest.approx(-info["waiting_cars"][a] / 10)
            assert info["total_cars"][a] >= info["waiting_cars"][a]
    assert n == 240 and info["time"] == 1200


def test_neighbour_graph(env):
    assert all(set(env.neighbors) == set(env.agents) for _ in [0])
    for a in env.agents:
        assert a not in env.neighbors[a] and env.neighbors[a]               # connected map: everyone has a neighbour
        for b in env.neighbors[a]:
            assert a in env.neighbors[b]                                    # symmetric by construction
    by_approaches = {t["id"]: t["n_approaches"] for t in env.info.report["traffic_lights"]}
    five = next(a for a in env.agents if by_approaches[a] == 5)
    assert len(env.neighbors[five]) == 3                                    # the 5-way touches three signalised junctions


def test_own_programs_baseline_matches_sumo(env):
    """The baseline mode must measure what SUMO itself reports for the network's own fixed-time programs."""
    env.reset("uniform", 7, own_programs=True)
    done = False
    while not done:
        _, _, done, info = env.step(None)
    m, route = env.metrics(), env.route_file
    st = G.run_static(env.info, route, seed=7)
    assert m["departed"] == st["inserted"] and m["throughput"] == st["arrived"]
    assert not any(info["yellow"].values())


def test_missing_demand_file_gives_clear_error(tmp_path, mapdir):
    import shutil
    d = tmp_path / "m"; shutil.copytree(mapdir, d); os.remove(d / "demand.json")
    e = SumoMapEnv(str(d))
    with pytest.raises(FileNotFoundError, match="calibrate-all"):
        e.reset("uniform", 1)


@pytest.mark.skipif(not os.path.isdir(NYC), reason="maps/nyc16 not present")
def test_real_map_smoke():
    e = SumoMapEnv(NYC)
    assert len(e.agents) == 15
    rng = np.random.default_rng(0)
    for scn in SCENARIOS:
        obs, done = e.reset(scn, 9001), False
        while not done:
            obs, r, done, info = e.step({a: int(rng.integers(e.n_acts[a])) for a in e.agents})
            assert all(np.isfinite(v).all() and v.min() >= 0 and v.max() <= 1 for v in obs.values())
        assert e.metrics()["departed"] > 100
    e.close()


def test_neighbour_distance_limit_is_used(mapdir):
    near = SumoMapEnv(mapdir, neighbor_dist=100)       # blocks in the test map are ~220 m: no light is within 100 m
    far = SumoMapEnv(mapdir, neighbor_dist=300)
    assert all(not v for v in near.neighbors.values())
    assert all(v for v in far.neighbors.values())
    assert all(d <= 300 for d in far.neighbor_dist.values())