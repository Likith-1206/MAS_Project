# RAPID-Light: design, loopholes fixed, results (nyc16)

Code: `src/rapid.py` (algorithm, training), `src/eval_rapid.py` (head-to-head evaluation, extra metrics).
Models: `results/nyc16/rapid/rapid2_s{0,1,2}.pt`. Evaluation CSVs: `results/nyc16/rapid/eval_all_*.csv`.

```bash
python src/rapid.py train maps/nyc16 --seed 0 --episodes 400 --tag rapid2      # ~27 min on one CPU core
python src/eval_rapid.py maps/nyc16 --mults 1.0 --tag rapid2                   # all methods, same traffic
python src/eval_rapid.py maps/nyc16 --mults 1.0 --yellow 3 --tag rapid2        # real-world amber
python src/eval_rapid.py maps/nyc16 --mults 1.0 1.5 --drop 0.3 --tag rapid2 --methods rapid   # 30 % lost messages
```

Same environment as the project table: `env_map.SumoMapEnv`, 5 s decisions, 2 s yellow, 1200 s episodes, scenarios
uniform / asymmetric / rush_hour / corridor, test seeds 9001-9003, metrics computed by `env_map.metrics()`.

## 1. Loopholes found in the baseline (measured, not assumed)

| # | Loophole | Evidence | Fix in RAPID-Light |
|---|---|---|---|
| L1 | Reward = halting cars only | IDQN time loss 27 s, of which only 8 s is stopped time | Reward = delay rate sum(1 - v/v_max): its time integral is exactly the time loss |
| L2 | Reactive: cars must stop before they get green | Even at 5 % demand, max_pressure gives 8 s wait per car | Detector features for vehicles 0-50 m / 50-150 m from the stop line; neighbour release messages |
| L3 | Horizon 50 s (gamma 0.9) shorter than a trip (100 s) | | 3-step returns, gamma 0.95 |
| L4 | Collapse outside training demand | IDQN at x1.5: one seed gridlocks (wait 78.7 s, throughput 655) | Demand randomised x0.5-2.0 in training; expert warm start; shield |
| L5 | No starvation guarantee | IDQN worst red with queued cars: 52 s / 347 s / 665 s at x1 / x1.5 / x2 | LTL shield phi_1, 0 violations at every demand level |
| L6 | 15 separate networks, no transfer | | One shared phase-competition network for any light / any map |
| L7 | MAPPO sample-starved, team reward | MAPPO worse than max_pressure | Off-policy shared DQN with per-light reward |
| L8 | Metrics can be gamed by blocking entries | collapsed IDQN seed shows the *best* avg trip | Reward charges vehicles waiting to enter; new column system_travel_h |

## 2. Algorithm

* **Sensing (real world):** each light sees only vehicles within 150 m of its stop line, never past the previous
  signal (radar/camera range). **Knowledge K.**
* **Communication (real world):** adjacent lights send, per approach, how many vehicles they are releasing towards
  the receiver. One decision late (5 s), 20 % lost in training (held at last value). **Belief B.**
* **State:** one token per green phase: queue, near (0-50 m) and far (50-150 m) approaching vehicles, delay rate,
  exit-lane queue and occupancy, pressure, longest stop, time since last green, current-phase flag, time in phase,
  message, lane count, exit-full flag.
* **Network:** shared token encoder -> self-attention between the phases of the light -> pooled context -> Q per
  phase (FRAP-style phase competition). Works for any number of phases / lanes; one network for all lights.
* **Reward (training only, CTDE):** r_a = -(sum over vehicles on the light's whole approaches of (1 - v/v_max)
  + vehicles waiting to enter the map on entries feeding the light) / 10. Summed over lights = d/dt of total delay
  including time outside the map, i.e. exactly the primary metric (total travel) without the border loophole.
* **Learner:** shared Double DQN, 3-step returns, gamma 0.95, replay 300k, batch 256, soft target (tau 0.005).
  12 Advanced-MP demonstration episodes + DQfD margin loss (decays to 0 over the first 30 % of training);
  epsilon 0.3 -> 0.02, half of the random actions are the expert's. 400 episodes, demand x1.0 with p = 0.4,
  else uniform x0.5-2.0, random scenario.
* **Shield (temporal logic, runs in training and test, uses only K):**
  phi_1 `G(stopped_p and red_p >= 60 s -> green_p)` (no starvation),
  phi_2 `G not(green_p and every exit lane of p full)` unless no phase can move (no gridlock).

Tried and **dropped** because measurements showed no gain:
* reward coupling with neighbours (alpha 0.3): x1.0 wait 5.2 vs 4.9, x2.0 throughput 895 vs 935;
* Hedge/EXP3 regret gate between RAPID and Advanced-MP: x1.0 wait 7.2-8.8 vs 4.9-5.9.
  The learned policy is better than the expert in almost every window, so the gate only pays its exploration cost.
  Regret is kept as an evaluation lens, not as a controller component.

## 3. Results (mean over 4 scenarios x seeds 9001-9003; +- = spread over 3 training seeds)

### Normal demand (the project table's setting)

| Method | avg wait (s) | avg trip (s) | throughput | not entered | total travel (h) |
|---|---|---|---|---|---|
| fixed_program | 86.4 | 196.9 | 528.2 | 12.8 | 34.5 |
| max_pressure | 12.2 | 117.4 | 597.6 | 2.8 | 21.0 |
| mappo (project table) | 22.8 | 145.6 | 556.5 | 13.4 | 26.6 |
| independent DQN (project table) | 6.3 +- 0.3 | 102.7 +- 0.8 | 609.5 +- 0.9 | 1.1 +- 0.1 | 18.5 +- 0.1 |
| Advanced-MP + shield | 10.3 | 108.1 | 607.4 | 1.1 | 19.4 |
| **RAPID-Light** | **4.9 +- 0.2** | **98.5 +- 0.1** | **612.2 +- 0.2** | 1.2 +- 0.1 | **17.8 +- 0.0** |

Not entered has a structural floor of about 1.08: every controller fails to insert the same vehicles that are scheduled
in the last seconds of the episode. RAPID differs from the floor on one episode only.

Extra metrics at x1.0: time loss 22.5 s (IDQN 27.0), 95th-pct wait 23.1 s (IDQN 23.5), starvation violations 0.

### Stress (x1.5 / x2.0 demand, never the exact training condition)

| Demand | Method | avg wait | avg trip | throughput | total travel | travel incl. waiting outside | starvation viol. |
|---|---|---|---|---|---|---|---|
| x1.5 | max_pressure | 27.7 | 155.0 | 785.5 | 39.3 | 48.4 | 0.5 |
| x1.5 | independent DQN | 39.9 +- 33.7 | 140.8 | 763.9 +- 94.3 | 39.1 | 48.2 | 1.6 |
| x1.5 | Advanced-MP | 24.3 | 133.9 | 853.9 | 36.0 | 37.8 | 0 |
| x1.5 | **RAPID-Light** | **18.8 +- 1.0** | **125.2** | **860.9 +- 5.2** | **33.6** | **35.9** | **0** |
| x2.0 | max_pressure | 52.3 | 193.4 | 819.4 | 57.7 | 96.9 | 2.2 |
| x2.0 | independent DQN | 82.1 +- 51.7 | 173.7 | 757.2 +- 137.5 | 59.0 | 100.6 | 4.2 |
| x2.0 | Advanced-MP | 51.3 | 182.2 | **944.1** | 56.6 | 82.7 | 0 |
| x2.0 | **RAPID-Light** | **45.2 +- 6.4** | **163.7** | 934.8 +- 23.1 | **51.0** | **81.3** | **0** |

### Robustness

| Test | RAPID-Light | Best other |
|---|---|---|
| Real-world amber 3 s, x1.0 | wait 5.9, trip 101.7, thr 610.5, travel 18.3 | IDQN 7.2 / 105.7 / 607.6 / 19.0 |
| Real-world amber 3 s, x1.5 | wait 20.2, thr 855.7, travel 34.4 | Advanced-MP 28.0 / 836.1 / 37.0 |
| 30 % of messages lost, x1.0 / x1.5 | wait 4.9 / 17.4, travel 17.8 / 33.0 | (no degradation) |

## 4. Open items

1. Real-world demand from the team's SUMO files (network + routes + vehicle types) instead of `scenario_gen.py`.
2. Mixed Indian traffic: weight queues and delay by PCU once vehicle classes exist; per-class delay metric.
3. At x2.0 Advanced-MP still admits ~9 more vehicles per episode (944 vs 935); RAPID is better on every other
   metric there, including travel time counted with the time spent waiting outside.
