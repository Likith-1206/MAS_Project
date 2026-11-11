"""Tests for MAPPO on a map environment (needs torch, SUMO, and maps/nyc16 with demand.json)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pytest
torch = pytest.importorskip("torch")
NYC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "maps", "nyc16")
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(NYC, "demand.json")), reason="maps/nyc16/demand.json missing")
from env_map import SumoMapEnv
from mappo_map import MAPPO
import neighbors_map as NB
import scenario_gen as G


@pytest.fixture(scope="module")
def env():
    e = SumoMapEnv(NYC)
    yield e
    e.close()


def test_map_has_lights_with_different_phase_counts(env):
    assert len(set(env.n_acts.values())) > 1            # this is the case the old MAPPO could not handle
    assert max(len(v) for v in env.neighbors.values()) > 4


@pytest.mark.parametrize("level", [0, 1, 2])
def test_act_store_update_never_picks_an_illegal_phase(env, level):
    torch.manual_seed(0)
    ag = MAPPO(env, level=level)
    obs, done, illegal = env.reset("uniform", 7), False, 0
    while not done:
        act = ag.act(obs, explore=True)
        illegal += sum(act[a] >= env.n_acts[a] for a in env.agents)
        obs, r, done, _ = env.step(act)
        ag.store(r)
    assert illegal == 0
    st = ag.update(obs)
    assert np.isfinite(list(st.values())).all()
    ev = ag.act(env.reset("uniform", 8), explore=False)
    assert all(ev[a] < env.n_acts[a] for a in env.agents)


def test_actor_input_size_matches_the_slots(env):
    ag = MAPPO(env, level=2)
    k = max(len(v) for v in env.neighbors.values())
    assert ag.n_slots == k
    assert ag.obs_dim == NB.base_dim(env.n_act, env.max_lanes) + k * NB.slot_dim(2, env.n_act)
    assert ag._actor_obs(ag._prepare_observations(env.reset("uniform", 1))).shape == (len(env.agents), ag.obs_dim)


def test_masked_probability_of_illegal_actions_is_zero(env):
    ag = MAPPO(env, level=0)
    logits = ag.actor(torch.zeros(len(env.agents), ag.obs_dim)).masked_fill(ag.mask == 0, -1e9)
    p = torch.softmax(logits, dim=1)
    assert float(p[ag.mask == 0].max()) < 1e-6


def test_same_seed_same_episode(env):
    def run():
        torch.manual_seed(3)
        ag = MAPPO(env, level=1)
        obs, done = env.reset("rush_hour", 5), False
        while not done:
            obs, _, done, _ = env.step(ag.act(obs, explore=False))
        return env.metrics()["avg_wait_s"], env.metrics()["throughput"]
    assert run() == run()


def test_scale_for_reads_both_demand_formats(tmp_path):
    import json
    (tmp_path / "demand.json").write_text(json.dumps({"scale": 1.5}))
    assert G.scale_for(str(tmp_path), "corridor") == 1.5
    (tmp_path / "demand.json").write_text(json.dumps({"scales": {"corridor": 0.8, "uniform": 1.1}}))
    assert G.scale_for(str(tmp_path), "corridor") == 0.8
    with pytest.raises(KeyError):
        G.scale_for(str(tmp_path), "rush_hour")
    with pytest.raises(FileNotFoundError):
        G.scale_for(str(tmp_path / "nope"), "uniform")


def test_compare_uses_common_seeds_and_fills_not_entered(tmp_path, capsys):
    """run_map.py baselines (3 seeds, no not_entered) and a MAPPO file with more seeds must still compare fairly."""
    import pandas as pd
    import run_mappo_map as R
    cols = dict(avg_wait_s=10.0, avg_trip_s=100.0, throughput=500.0, total_travel_h=20.0, departed=600.0)
    def table(method, seeds):
        return pd.DataFrame([dict(method=method, train_seed=0, scenario=s, eval_seed=sd, **cols) for s in G.PATTERNS for sd in seeds])
    table("fixed_program", [9001, 9002, 9003]).to_csv(tmp_path / "eval_baselines.csv", index=False)
    table("mappo", [9001, 9002, 9003, 9004, 9005]).to_csv(tmp_path / "eval_mappo_s0.csv", index=False)
    R.compare(NYC, {"results": str(tmp_path)}, "mappo_s0")
    out = capsys.readouterr().out
    assert "WARNING" in out and "9004" not in out.split("mean over test seeds")[1].split("\n")[0]
    assert "NaN" not in out and "not_entered" in out