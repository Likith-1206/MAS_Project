# What the team needs to learn

Priority: **Must** = needed to build or defend the project, **Should** = helps with papers/results, **Stretch** = only for QMIX/MAPPO.

## A. Theory and mathematics
| Area | Topics | Priority | Used for |
|---|---|---|---|
| Probability | random variables, expectation, variance; Markov chains; Poisson process / exponential inter-arrival times | Must | vehicle arrival model (`probability=` flows), MDP foundations |
| Statistics | confidence intervals, standard error, paired t-test, Wilcoxon signed-rank, bootstrap | Must | comparing controllers across seeds (`make_plots.py`) |
| Linear algebra / calculus | vectors, matrices, norms, gradients, chain rule, Jacobian | Must | neural networks, backpropagation |
| Optimisation | gradient descent, SGD, Adam, learning rate, batch size, gradient clipping, Huber loss | Must | DQN training |
| Single-agent RL | MDP (S,A,P,R,gamma); return; V and Q; Bellman expectation/optimality equations; TD learning; Q-learning update Q<-Q+alpha(r+gamma max Q'-Q); on/off-policy; epsilon-greedy; function approximation and the deadly triad; replay buffer, target network; Double DQN | Must | agents (`agents.py`) |
| Policy gradients | policy-gradient theorem, advantage, GAE, PPO clipped objective | Should / Stretch | MAPPO |
| Multi-agent RL | Markov games, Dec-POMDP, non-stationarity, credit assignment, independent learners vs parameter sharing vs CTDE, team vs local reward, reward mixing | Must | core of the course and RQ1/RQ2 |
| Game theory | Nash equilibrium, cooperative vs non-cooperative, locally optimal != globally optimal | Should | motivation of coordination |
| QMIX theory | IGM condition, monotonic mixing dQ_tot/dQ_i >= 0 | Stretch | QMIX extension |
| Graph theory | adjacency, degree, Laplacian L=D-A, k-hop neighbourhood, connectivity, diameter; consensus x(k+1)=Wx(k) | Must | neighbour definition, RQ2 (how many hops help) |
| Traffic flow | flow q = density x speed; fundamental diagram; queue dynamics q(t+1)=q(t)+arrivals-departures; Little's law L = lambda W; utilisation rho = lambda/mu and stability | Must | metric interpretation, demand calibration |
| Signal control | phase, cycle, split, yellow/lost time, saturation flow, offset (green wave); Webster C0=(1.5L+5)/(1-Y); Max-Pressure (phase pressure = sum of upstream minus downstream queue) | Must | baselines |
| Experimental method | multiple seeds, identical demand for every controller, equal tuning budget, warm-up, ablation, mean +- std / CI | Must | credible evaluation |

## B. Technical skills
| Skill | Topics |
|---|---|
| SUMO | `netgenerate --grid`, flows/routes (flows must be sorted by `begin`), traffic-light programs, output files, `sumo-gui`, **libsumo** (in-process, fast) vs TraCI (socket) |
| Python / PyTorch | modules, optimisers, training loops, replay buffers, saving models, TensorBoard/W&B |
| RL tooling | Gymnasium and PettingZoo (parallel) APIs; `sumo-rl` and RESCO benchmark as references; EPyMARL for QMIX/MAPPO (stretch) |
| Data / plots | pandas, Matplotlib, Plotly, scipy.stats |
| Dashboard | Streamlit, Plotly, replaying stored traces |
| Engineering | Git branches + pull requests, pinned environments (`requirements.txt`), config files, reproducible seeds, unit tests |

## C. Papers (abstract + method + results)
Max-Pressure (Varaiya 2013), IntelliLight, PressLight, CoLight, RESCO benchmark (Ault & Sharon), QMIX (Rashid 2018), MAPPO (Yu 2021), independent Q-learning (Tan 1993).

## D. Whiteboard check (each member, before Day 14)
Explain without notes: (a) the Q-learning update; (b) why independent learners face non-stationarity; (c) Little's law for queue vs waiting time; (d) the neighbour graph and why hops matter.
