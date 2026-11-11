"""Regression benchmark: environment v2, run on the old 3x3 grid, must reproduce environment v1 exactly
(same route file, same seed, same actions -> identical observations, rewards and metrics)."""
import json, os, shutil, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pytest
try:
    import env as v1, scenarios as S
    import osm_import as O
    from env_map import SumoMapEnv
    OK = True
except Exception:                                              # v1 files or SUMO missing
    OK = False
pytestmark = pytest.mark.skipif(not OK, reason="env.py / scenarios.py / SUMO not available")


def actions(i, agents):
    return {a: (i // 3 + k) % 2 for k, a in enumerate(agents)}


def test_v2_reproduces_v1_on_the_3x3_grid(tmp_path):
    n_steps, seed = 50, 11
    e1 = v1.SumoGridEnv(comm_level=0)
    obs1, rew1 = [e1.reset("uniform", seed)], []
    for i in range(n_steps):
        o, r, done, info = e1.step(actions(i, e1.agents)); obs1.append(o); rew1.append(r)
    m1, route, agents1 = e1.metrics(), os.path.join(e1.tmp, "uniform.rou.xml"), list(e1.agents)
    e1.close()                                                  # libsumo: one simulation per process

    d = tmp_path / "grid"; d.mkdir()
    shutil.copy(S.NET_FILE, d / "net.net.xml")
    json.dump(O.inspect_network(str(d / "net.net.xml")), open(d / "map_report.json", "w"))
    e2 = SumoMapEnv(str(d))
    assert e2.agents == agents1 and all(k == 2 for k in e2.n_acts.values()) and all(v == 19 for v in e2.obs_dims.values())
    obs2, rew2 = [e2.reset("uniform", seed, route_file=route)], []
    for i in range(n_steps):
        o, r, done, info = e2.step(actions(i, e2.agents)); obs2.append(o); rew2.append(r)
    m2 = e2.metrics(); e2.close()

    for t, (a, b) in enumerate(zip(obs1, obs2)):
        for ag in agents1:
            assert np.array_equal(a[ag], b[ag]), (t, ag)
    assert rew1 == rew2
    assert {k: float(v) for k, v in m1.items()} == {k: float(v) for k, v in m2.items()}