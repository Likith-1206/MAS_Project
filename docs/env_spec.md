# Environment spec (Interface 1) - Pair A
`from env import SumoGridEnv` (put `src` on the path). Runs with libsumo, no GUI.

| Call | Meaning |
|---|---|
| `env = SumoGridEnv(comm_level=0)` | 0 = own data (19 numbers); 1 / 2 add neighbour queue / queue+phase (23 / 31) |
| `obs = env.reset(scenario, seed)` | `scenario` in `scenarios.SCENARIOS`; returns `{agent: np.float32 vector}`. Same scenario + seed = identical traffic (tested). |
| `obs, rewards, done, info = env.step({agent: 0 or 1})` | one step = 5 s; a 2 s yellow is the first 2 s of the step if the phase changes |
| `env.metrics()` | avg_wait_s, avg_queue, max_queue, throughput, total_travel_h, avg_trip_s, in_network, departed |

**Observation (19, all 0-1):** 1-8 waiting cars per incoming lane / capacity; 9-16 all cars per lane / capacity;
17-18 current phase one-hot; 19 seconds since last phase change / 60 (max 1). Lane order: approach N, E, S, W, then lane 0, 1.
`env.obs_names` lists the names, `env.obs_dim` = 19 / 23 / 31.

**Reward:** `-(waiting cars on the agent's incoming lanes) / 10`.

**info (per-agent dicts):** `queue`, `waiting_cars`, `total_cars`, `acc_wait_s`, `phase`, `yellow`, `time`.

**Actions (checked, see `python src/check_phases.py`):** `env.n_act = 2`.
Action 0 = green for traffic coming from **North and South**; action 1 = green for traffic coming from **East and West**
(same at all 9 intersections; `env.phase_dirs = [(0, 2), (1, 3)]`, direction index 0=N 1=E 2=S 3=W).

**Agents / neighbours:** `env.agents` = A0..C2 (A0 bottom-left). `env.neighbors[a] = [N, E, S, W]`, each an agent id or `None`.

**Episode:** `done` is True after 1200 simulated seconds (240 steps). One episode takes about 2-3 s on one core (run `python src/time_episode.py` for your own number).

**Metric check vs SUMO tripinfo (`python src/check_metrics.py`):** throughput and avg trip time identical; waiting time per car is
~2 s lower in ours (we sample every 5 s, SUMO every 1 s).
