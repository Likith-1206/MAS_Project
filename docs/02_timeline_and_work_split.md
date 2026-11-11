# 30-day plan and work split (everyone works at every stage)

| Member | Primary module | Secondary role |
|---|---|---|
| M1 | Network, demand, scenarios (`scenarios.py`) | Runs/monitors training & experiment jobs |
| M2 | RL environment (`env.py`: obs, action, reward) | Reward variants and debugging |
| M3 | Independent DQN + training loop (`agents.py`, `run_experiments.py`) | Hyper-parameter tuning for all learners |
| M4 | Cooperative agent, communication scheme | RQ2 ablation |
| M5 | Baselines, metrics, statistics (`controllers.py`, `make_plots.py`) | Results tables and analysis |
| M6 | Dashboard, plots, integration (`dashboard.py`), repo owner | Report assembly |

> **Note:** this repository already contains a working first version of every module (see README). With that head start the
> 30 days shift from "build" to "verify, extend, tune and write": each member re-derives, tests and improves *their* module.

## Week 1 (Days 1-7): Learn, reproduce, understand   Gate: everyone runs the full pipeline and passes the whiteboard check
| Member | Tasks |
|---|---|
| All | Install `requirements.txt`; run `bash run_all.sh` with `EPISODES=5`; read `01_learning_guide.md` core; answer whiteboard questions |
| M1 | Read `scenarios.py`; open the net in `sumo-gui`; add 2 new scenarios (e.g. incident/blocked lane, uniform-low); verify demand with `tripinfo` |
| M2 | Read `env.py`; unit tests for obs/reward/yellow logic; check phase semantics across agents |
| M3 | Re-implement DQN update by hand on CartPole; compare to `agents.py`; log with TensorBoard |
| M4 | Read MARL papers; write design note on communication schemes; test neighbour obs correctness |
| M5 | Re-derive each metric; validate against SUMO `tripinfo.xml`; add Webster-timed fixed baseline |
| M6 | Set up repo/branches/CI; extend dashboard; start report skeleton |

## Week 2 (Days 8-14): Improve environment and independent agents   Gate: design freeze Day 10, learning curves stable
| Member | Tasks |
|---|---|
| M1 | Calibrate demand so fixed-time is congested but not gridlocked; 4 phases (protected left turns) version of the network |
| M2 | Richer observation (speed, waiting time), queue-vs-wait-vs-pressure reward experiments |
| M3 | Tune independent DQN (lr, gamma, eps schedule, target update); longer training (300+ episodes) |
| M4 | Freeze communication design (levels 0/1/2, reward mixing beta); implement 2-hop information |
| M5 | Harness for all controllers x scenarios x >=5 seeds; Actuated-control baseline |
| M6 | Dashboard live comparison view; integration on `main` every 2 days |

## Week 3 (Days 15-21): Cooperation and full comparison   Gate: first full comparison with >=3 training seeds
| Member | Tasks |
|---|---|
| M1 | Run training jobs (seeds x methods); 4x4 network for scalability |
| M2 | Fix environment issues found in training; reward shaping ablation |
| M3 | Equal-budget tuning for coop and independent; Double/Dueling DQN variants |
| M4 | Cooperative agent improvements; QMIX/MAPPO via EPyMARL only if coop DQN is stable by Day 18 |
| M5 | Statistics: mean +- std, CI, paired tests over seeds; results tables |
| M6 | Replay of trained policies in dashboard; final plots; report draft |

## Week 4 (Days 22-30): Final experiments, report, demo
| Days | Activity |
|---|---|
| 22-24 | Final experiments: all scenarios, >=5 training seeds, RQ2 ablation. Each member runs a slice |
| 25 | **Code freeze** |
| 25-28 | Each member writes the report section for their module; dashboard polish |
| 29 | Full demo rehearsal on a clean machine |
| 30 | Buffer / submission |

## Contingency plan
* DQN does not learn by Day 14 -> simplify state, pressure-based reward, or start from `sumo-rl`.
* Training too slow -> `libsumo`, parallel seeds, shorter episodes.
* Cooperative agent does not beat independent -> report as is; the RQ2 ablation is still a valid result.
* SUMO install problems -> shared pinned environment / container.
