# Multi-Agent Traffic Signal Coordination (MAS project) - working prototype

A complete, runnable first version of the project in the proposal: a 3x3 signalised grid in SUMO (libsumo), one agent per
intersection, three approaches compared (fixed-time, independent adaptive DQN, cooperative DQN with neighbour information),
plus a Max-Pressure reference baseline, an RQ2 communication ablation, statistics, plots and a Streamlit dashboard.

**Status:** every module runs end-to-end and produced the results in `results/`. Those results come from a *short* training
budget (about 40 episodes, 1-2 seeds, 1 CPU core) and are **preliminary**. The remaining project time is for longer training, more
seeds, tuning, extensions and the write-up (see `docs/02_timeline_and_work_split.md`).

## Quick start
```bash
pip install -r requirements.txt
bash run_all.sh                 # baselines + all training runs (EPISODES=40 by default, the value used for the delivered results; ~25 min on one core)
python src/make_plots.py        # tables, stat tests, plots -> results/
streamlit run src/dashboard.py  # network replay + results
```
Single runs: `python src/run_experiments.py train --method coop --episodes 100 --seed 0` (methods: `independent`, `coop`, `shared_c0`, `shared_c1`, `shared_c2`).

## Layout
| Path | Content |
|---|---|
| `src/scenarios.py` | 3x3 grid generation (`netgenerate`), 4 demand scenarios (uniform, asymmetric, rush_hour, corridor), Poisson-like arrivals |
| `src/env.py` | Multi-agent environment: obs / action / reward, yellow handling, metrics, optional neighbour information |
| `src/controllers.py` | Baseline 1 fixed-time (30 s per phase); reference Max-Pressure |
| `src/agents.py` | Double-DQN in PyTorch: `IndependentDQN` (one net per intersection) and `SharedDQN` (parameter sharing), reward mixing |
| `src/run_experiments.py` | Training + evaluation (3 eval seeds x 4 scenarios per method) |
| `src/make_plots.py` | Summary tables, paired statistical tests, bar plots, ablation, learning curves |
| `src/dashboard.py` | Streamlit dashboard (replay of queues/phases per intersection, results) |
| `results/`, `models/`, `traces/` | CSVs, plots, trained weights, replay traces |
| `docs/` | Learning guide, 30-day timeline and work split, results/report draft |

## Formulation
* **Observation** (per intersection): queue and vehicle count per incoming lane (8 lanes, normalised), current phase one-hot, time in phase.
  Cooperative agent adds, for each of 4 neighbour slots (N/E/S/W), the neighbour's mean queue (level 1) and current phase (level 2); missing neighbours are zero-padded.
* **Action:** which green phase to serve next (2 phases: north-south / east-west); switching inserts a 2 s yellow. Decision every 5 s.
* **Reward:** minus total queue on the incoming lanes / 10. Cooperative agent: `r_i + 0.5 * mean(r_neighbours)` (discourages pushing congestion onto neighbours).
* **Demand:** Bernoulli arrivals (approximately Poisson) from 12 boundary entries to all other exits, 1200 s episodes, `DEMAND_SCALE = 2.2` in `scenarios.py` (calibrated to be congested but not gridlocked).
* **Methods:** `fixed_time`; `max_pressure` (reference, not in the proposal); `independent` (separate network per intersection, no information exchange);
  `coop` (shared network + neighbour info level 2 + reward mixing); `shared_c0/c1/c2` (shared network, reward not mixed, neighbour info level 0/1/2 -> isolates the effect of communication, RQ2).

## Known limitations (be upfront about these in the report)
* Only 2 green phases (no protected left-turn phases) -> binary action. A 4-phase network is a natural extension.
* Short training, few seeds: differences between the learned methods can be within noise; use `results/stat_tests.csv`.
* `independent` differs from `coop` in more than communication (separate vs shared weights, no reward mixing); the `shared_c*` ablation separates those factors.
* Travel-time metrics count only vehicles that entered the network; vehicles waiting to be inserted are not counted.
* Learned agents are evaluated after training with epsilon = 0 on unseen seeds; scenarios are cycled during training (one policy for all scenarios).

## Animated simulations on every city (`src/simulate.py`)
One HTML file per city (double-click, no server): the city map, mixed Indian traffic, live signal heads, a ring on
every agent showing the phase it chose, and **RAPID-Light, Independent DQN, MAPPO and fixed-time side by side on
identical traffic** (same trips, same SUMO seed), with live KPIs, a junction inspector and a video-record button.

Setup (once; the Quick start above is for the original 3x3 grid, not these city simulations):
```bash
cd traffic_marl
python3 -m venv mas && source mas/bin/activate
pip install -r requirements.txt
ls results/nyc16/rapid_v2_mixed/ models/*/        # trained models must be here (see below)
```
Run (from `traffic_marl`, with `mas` activated):
```bash
./simulate_all.sh            # ALL five cities, all controllers -> opens viz/index.html
./simulate_all.sh --average  # same, numbers = 12-test-episode averages + the episode closest to them
```
`viz/index.html` is one page: an overview with every city's numbers, and a tab per city (New York, Pune, Mumbai,
Bangalore, Jersey). Pick a city and its simulation opens with **RAPID-Light final, RAPID-Light multi-city,
Independent DQN, MAPPO and fixed-time side by side** on identical traffic. Later: just `xdg-open viz/index.html`.
Models already in `results/` and `models/<city>/` are loaded, never retrained; a city without its DQN/MAPPO models
gets them trained first (slow). Other choices: `CTRL="rapid" ./simulate_all.sh` (RAPID final vs fixed only),
`python src/simulate.py maps/mumbai` (re-run one city; it stays in the same index page).
* **RAPID-Light is not trained per city**: the same phase-token network runs every light of every map, exactly as
  in `eval_mixed.py` (detector `Sensor` -> greedy Q -> LTL `Shield`). Put your trained files at
  `results/nyc16/rapid_v2_mixed/rapid2_mixedtrained_s0.pt` (`rapid`, final), `results/nyc16/rapid/rapid_s0.pt` (`rapid_v1`),
  `results/nyc16/rapid/rapid2_s0.pt` (`rapid_v2`), `results/nyc16/rapid_mixed/rapid_mixed_s0.pt` (`rapid_pcu`, runs with the
  PCU `MixedSensor`), `results/multicity/rapid_multicity_s0.pt` (`rapid_multicity`); `--rapid-seed 1` picks the `_s1` files.
* **Independent DQN and MAPPO are per city**: `models/<city>/independent_s0.pt` and `mappo_s0.pt` are loaded; a missing
  one is trained on that city first (`--episodes 60`, `--mappo-episodes 150`) and saved. `models/nyc16/independent_s*.pt`
  (the team's DQN) is included. Note: training them on `pune_test` uses the held-out city (only RAPID keeps it unseen).
* `--average` (recommended for presenting): every controller also runs on all 12 test episodes (4 patterns x seeds
  9001-9003, as `eval_multicity.py`); the viewer and `viz/<city>.html` then show the 12-episode averages and replay the
  test episode closest to them (`--show chosen` keeps `--scenario/--seed`). Every run also reports the longest red with
  waiting vehicles (`max_red_s`, as in `eval_mixed.py`; RAPID's shield limit is 60 s).
* Traffic: `--traffic mixed` (default, `scenario_gen_mixed.py`) or `cars`; `--scenario rush_hour|uniform|asymmetric|corridor`;
  `--seed 9001` (test seeds 9001-9003, never trained on); `--demand 1.5` for heavier traffic.
* Buildings, parks and rail are drawn when `osm/<city>.osm` exists (`osm/mumbai.osm` included; add `bangalore.osm`,
  `jersey.osm`, `pune.osm` (or `pune_test.osm`) and `nyc16.osm` for New York, the same .osm files the maps were made from).
  Otherwise roads only.
* Without `--average` each viewer is ONE episode (default rush hour, seed 9001, `_s0` models), so its numbers differ from
  the averaged evaluation tables; use those (or `--average`) for claims.
* A new city: `python src/simulate.py path/to/area.osm --name newcity` imports and calibrates it first.

Files: `src/simulate.py`, `src/viz_export.py`, `src/viz_template.html`, `simulate_all.sh`, `src/plugins/policy_template.py`,
`tests/test_viz_export.py`.

> **Note:** The simulation numbers differ from the reported results because each viewer replays a single 20-minute episode (by default rush hour, test seed 9001, using only the `_s0` models), while the reported results average 4 traffic patterns × 3 test seeds × 3 training seeds. In addition, Independent DQN and MAPPO are trained separately on each city here, whereas RAPID-Light is never retrained. Run `./simulate_all.sh --average` to see 12-episode averages instead.