import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, pytest
from make_test_osm import make_test_osm
import osm_import as O
import scenario_gen as G
import env_map as E
import run_map as R

try:
    O.find_netconvert(); HAVE_SUMO = True
except RuntimeError:
    HAVE_SUMO = False
pytestmark = pytest.mark.skipif(not HAVE_SUMO, reason="netconvert not installed")


@pytest.fixture(scope="module")
def mapdir(tmp_path_factory):
    d = tmp_path_factory.mktemp("runmap")
    O.import_osm(make_test_osm(str(d / "area.osm")), str(d / "testmap"))
    json.dump({"scales": {p: 0.6 for p in G.PATTERNS}, "ok": True}, open(d / "testmap" / "demand.json", "w"))
    return str(d / "testmap")


def test_max_pressure_gives_valid_actions_for_a_full_episode(mapdir):
    env, mp = E.SumoMapEnv(mapdir), R.MaxPressureMap()
    obs, done, n = env.reset("uniform", 9001), False, 0
    while not done:
        act = mp.act(env, obs)
        assert set(act) == set(env.agents) and all(0 <= act[a] < env.n_acts[a] for a in env.agents)
        obs, _, done, _ = env.step(act); n += 1
    assert n == 240 and env.metrics()["departed"] > 50
    env.close()


def test_independent_networks_have_each_lights_own_size(mapdir):
    env = E.SumoMapEnv(mapdir)
    ag = R.MapIndependentDQN(env)
    assert len(set(env.obs_dims.values())) >= 2                       # lights really differ on this map
    for a in env.agents:
        assert ag.nets[a].q[0].in_features == env.obs_dims[a] and ag.nets[a].n_act == env.n_acts[a]
    obs = env.reset("uniform", 1)
    for _ in range(70):                                               # > min_buf (500) for the busiest light? learn() is a no-op below
        act = ag.act(obs, 0.5)
        assert all(0 <= act[a] < env.n_acts[a] for a in env.agents)
        obs2, r, done, _ = env.step(act)
        ag.store(obs, act, r, obs2, 0.0); ag.learn(); obs = obs2
    sd = {a: {k: v.clone() for k, v in s.items()} for a, s in ag.state_dict().items()}
    other = R.MapIndependentDQN(env); other.load(sd)
    assert ag.act(obs, 0.0) == other.act(obs, 0.0)
    env.close()


def test_baselines_and_training_write_the_expected_files(mapdir, tmp_path):
    R.baselines(mapdir, out_root=str(tmp_path))
    b = pd.read_csv(tmp_path / "results" / "testmap" / "eval_baselines.csv")
    assert sorted(b.method.unique()) == ["fixed_program", "max_pressure"]
    assert len(b) == 2 * 4 * 3 and set(b.eval_seed) == {9001, 9002, 9003}
    R.train(mapdir, episodes=2, seed=5, out_root=str(tmp_path))
    r = tmp_path / "results" / "testmap"
    ev = pd.read_csv(r / "eval_independent_s5.csv")
    assert len(ev) == 12 and set(ev.train_seed) == {5} and not ev.avg_wait_s.isna().any()
    cv = pd.read_csv(r / "curve_independent_s5.csv")
    assert list(cv.columns) == ["episode", "scenario", "eps", "ret", "avg_queue", "avg_wait_s"] and len(cv) == 2
    st = json.load(open(r / "settings_independent_s5.json"))
    assert st["n_agents"] == 5 and st["eval_seeds"] == [9001, 9002, 9003] and st["demand_scales"]["corridor"] == 0.6
    assert (tmp_path / "models" / "testmap" / "independent_s5.pt").exists()


def test_stress_reproduces_eval_at_1x_and_adds_traffic_at_2x(mapdir, tmp_path):
    R.train(mapdir, episodes=2, seed=5, out_root=str(tmp_path))
    R.stress(mapdir, [1.0, 2.0], [5], out_root=str(tmp_path))
    r = tmp_path / "results" / "testmap"
    st, ev = pd.read_csv(r / "stress.csv"), pd.read_csv(r / "eval_independent_s5.csv")
    assert sorted(st.method.unique()) == ["fixed_program", "independent", "max_pressure"]
    assert set(st.demand_mult) == {1.0, 2.0} and len(st) == 2 * 3 * 12
    one = st[(st.demand_mult == 1.0) & (st.method == "independent")].sort_values(["scenario", "eval_seed"])
    assert np.allclose(one.avg_wait_s.values, ev.sort_values(["scenario", "eval_seed"]).avg_wait_s.values)   # 1x == the normal eval
    dep = st[st.method == "fixed_program"].groupby("demand_mult").departed.sum()
    assert dep[2.0] > 1.4 * dep[1.0]                                                                        # 2x really is more traffic