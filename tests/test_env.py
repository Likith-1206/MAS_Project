import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pytest
from env import SumoGridEnv
import scenarios as S


def run(env, scenario, seed, n_steps=60):
    env.reset(scenario, seed)
    for i in range(n_steps):
        env.step({a: (i // 4) % 2 for a in env.agents})        # fixed, boring policy
    return {k: float(v) for k, v in env.metrics().items()}


def test_same_seed_same_traffic():
    env = SumoGridEnv()
    a, b, c = run(env, "uniform", 3), run(env, "uniform", 3), run(env, "uniform", 4)
    env.close()
    assert a == b          # same scenario + same seed -> identical
    assert a != c          # different seed -> different cars


def test_obs_size_names_range():
    for level, dim in [(0, 19), (1, 23), (2, 31)]:
        env = SumoGridEnv(comm_level=level)
        assert env.obs_dim == dim == len(env.obs_names) and env.n_act == 2
        obs = env.reset("uniform", 1)
        for _ in range(10):
            obs, *_ = env.step({a: 0 for a in env.agents})
        assert len(env.agents) == 9 and set(obs) == set(env.agents)
        for v in obs.values():
            assert v.shape == (dim,) and v.min() >= 0 and v.max() <= 1
        env.close()


def test_phase_meaning():
    env = SumoGridEnv()
    assert env.phase_dirs == [(0, 2), (1, 3)]      # action 0 = N,S green ; action 1 = E,W green
    env.close()


def test_yellow_reward_info_done():
    env = SumoGridEnv()
    env.reset("corridor", 2)
    _, _, _, info = env.step({a: (1 if a == "B1" else 0) for a in env.agents})
    assert info["yellow"]["B1"] and not any(v for a, v in info["yellow"].items() if a != "B1")
    assert info["phase"]["B1"] == 1 and info["phase"]["A0"] == 0
    done, n = False, 1
    while not done:
        _, r, done, info = env.step({a: 0 for a in env.agents}); n += 1
        for a in env.agents:
            assert r[a] == pytest.approx(-info["waiting_cars"][a] / 10)
            assert info["total_cars"][a] >= info["waiting_cars"][a]
    assert n == 240 and info["time"] == 1200
    env.close()


def test_neighbors():
    env = SumoGridEnv()
    assert env.neighbors["B1"] == ["B2", "C1", "B0", "A1"] and env.neighbors["A0"] == ["A1", "B0", None, None]
    for a in env.agents:
        for k, n in enumerate(env.neighbors[a]):
            if n: assert env.neighbors[n][(k + 2) % 4] == a
    env.close()
