# Preliminary results (prototype run)

**Setup.** 3x3 grid, 4 scenarios, 1200 s episodes, demand scale 2.2. Learned agents: 40 training episodes (scenarios cycled), **one training seed**,
evaluated greedily on 3 unseen seeds x 4 scenarios = 12 episodes per method. Numbers are mean ± std over those 12 episodes
(`results/summary_table.csv`; per-scenario values in `results/summary_per_scenario.csv`; tests in `results/stat_tests.csv`).

| Method | Avg wait / veh (s) | Avg queue / intersection | Max lane queue | Throughput (veh) | Total travel (veh-h) |
|---|---|---|---|---|---|
| Fixed-time (Baseline 1) | 44.0 ± 7.3 | 9.0 ± 2.2 | 21.1 ± 1.8 | 1856 ± 151 | 84.9 ± 14.6 |
| Max-Pressure (reference) | 19.2 ± 5.6 | 4.0 ± 1.4 | 11.5 ± 5.1 | 1886 ± 133 | 79.0 ± 16.1 |
| Independent DQN (Baseline 2) | 18.1 ± 4.7 | 3.7 ± 1.2 | 11.8 ± 4.3 | 1779 ± 100 | 87.1 ± 21.1 |
| Cooperative DQN (proposed) | 18.6 ± 5.9 | 3.8 ± 1.4 | 15.5 ± 7.5 | 1774 ± 73 | 87.9 ± 22.9 |

## What the data supports
1. **Adaptive control lowers waiting time and queues sharply versus fixed-time.** Both learned agents cut average waiting time by about 58% (paired t-test and Wilcoxon p < 0.001 over 12 paired episodes). Max-Pressure achieves the same reduction with no learning.
2. **RQ1 (does coordination help?) - not shown at this budget.** Cooperative vs Independent: +0.5 s waiting time, p = 0.41 (t) / 0.73 (Wilcoxon). No detectable difference.
3. **RQ2 (does more neighbour information help?) - not shown.** With a shared network and no reward mixing: no communication 17.3 s, queue 17.7 s, queue+phase 17.0 s; differences vs no-communication are not significant (p > 0.5).
   The cooperative agent (level-2 information **plus reward mixing**) was slightly worse than the same information without reward mixing (18.6 vs 17.0 s, p = 0.03, single seed), so beta = 0.5 mixing may hurt or may just be noise; it needs tuning and more seeds before any conclusion.
4. **Mixed picture on throughput and total travel time.** Learned agents lower halting-based waiting time, but throughput is about 4-5% below fixed-time/Max-Pressure and total travel time is not lower than fixed-time (87 vs 85 veh-h). Possible reasons to investigate: waiting time counts only vehicles with speed < 0.1 m/s (slow-moving queues are not counted); more phase switching means more yellow (lost) time; the reward is queue-based rather than throughput-based. Do not report the learned agents as better overall on the basis of waiting time alone.

## Caveats
* One training seed and 40 episodes: RL results vary a lot between seeds; the conclusions above may change with 5+ seeds and 300+ episodes.
* Only 2 green phases; independent and cooperative agents differ in more than communication (see README); `shared_c*` isolates communication.
* Std values are across evaluation episodes (scenarios and seeds), not across independent training runs.

## Suggested next experiments (in priority order)
1. Train 5 seeds per method for 300+ episodes; report confidence intervals across training seeds.
2. Tune beta (0, 0.25, 0.5, 1.0) for reward mixing; try pressure-based or throughput-inclusive rewards.
3. Add a 4-phase (protected left) network and a 4x4 grid (scalability, RQ on network size).
4. Add 2-hop information and delayed/limited communication for RQ2.
5. Stretch: QMIX or MAPPO through EPyMARL.
