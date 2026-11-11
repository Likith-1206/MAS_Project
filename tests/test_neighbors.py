import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from env import SumoGridEnv
from neighbors import augment, extra_dim


def test_extra_dims():
    assert [extra_dim(l) for l in (0, 1, 2)] == [0, 4, 12]


def rollout(level, n_steps=30, scenario="asymmetric", seed=5):
    """Run one episode with a fixed action sequence; return every observation and the neighbour table.
    (libsumo allows only ONE simulation per process, so envs must run one after another.)"""
    env = SumoGridEnv(comm_level=level)
    obs = env.reset(scenario, seed)
    seq = [obs]
    for i in range(n_steps):
        obs, *_ = env.step({a: (i // 3 + k) % 2 for k, a in enumerate(env.agents)})
        seq.append(obs)
    nbrs = env.neighbors
    env.close()
    return seq, nbrs


def test_matches_env_levels():
    """Our module on level-0 obs must equal the env's own level-1/2 obs (same scenario, seed, actions)."""
    base, nbrs = rollout(0)
    for level in (1, 2):
        ref, _ = rollout(level)
        for t, (o0, oL) in enumerate(zip(base, ref)):
            mine = augment(o0, nbrs, level)
            for a in o0:
                assert mine[a].shape == (19 + extra_dim(level),)
                assert np.allclose(mine[a], oL[a], atol=1e-6), (level, t, a)


def test_missing_neighbour_is_zero():
    env = SumoGridEnv(comm_level=0)
    obs = env.reset("uniform", 1)
    v = augment(obs, env.neighbors, 2)["A0"]       # A0 neighbours: [A1, B0, None, None]
    assert env.neighbors["A0"] == ["A1", "B0", None, None]
    assert np.all(v[19 + 6:] == 0)                 # S and W slots (2 x 3 numbers) are zero
    env.close()