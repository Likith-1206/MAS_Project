import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch, pytest
from env import SumoGridEnv
from agents import SharedDQN, IndependentDQN, mix_rewards


@pytest.mark.parametrize("level,dim", [(0, 19), (1, 23), (2, 31)])
def test_shared_agent_levels(level, dim):
    env = SumoGridEnv(comm_level=0)                      # learned agents always use the level-0 env
    agent = SharedDQN(env, level=level)
    assert agent.obs_dim == dim
    obs = env.reset("uniform", 1)
    for _ in range(70):                                  # 70 steps x 9 agents > min_buf (500)
        act = agent.act(obs, 0.5)
        assert set(act) == set(env.agents) and set(act.values()) <= {0, 1}
        obs2, r, done, info = env.step(act)
        agent.store(obs, act, mix_rewards(env, r, 0.5), obs2, 0.0)
        agent.learn()
        obs = obs2
    # save / load round trip gives identical greedy actions
    sd = {k: v.clone() for k, v in agent.state_dict().items()}
    other = SharedDQN(env, level=level); other.load(sd)
    assert agent.act(obs, 0.0) == other.act(obs, 0.0)
    env.close()


def test_independent_agent_and_mixing():
    env = SumoGridEnv(comm_level=0)
    agent = IndependentDQN(env)
    obs = env.reset("corridor", 2)
    act = agent.act(obs, 0.0)
    obs2, r, done, info = env.step(act)
    agent.store(obs, act, r, obs2, 0.0); agent.learn()
    mixed = mix_rewards(env, r, 0.5)                     # corner A0 has 2 neighbours, no crash on None
    nb = [n for n in env.neighbors["A0"] if n]
    assert mixed["A0"] == pytest.approx(r["A0"] + 0.5 * np.mean([r[n] for n in nb]))
    env.close()