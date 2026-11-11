"""MAPPO agent for multi-intersection traffic signal control (version for env_map.SumoMapEnv).

Changes against mappo.py (which assumed every light has the same number of phases):
  * observations are read with each light's OWN number of phases (env.n_acts) and padded to the largest;
  * ILLEGAL ACTIONS ARE MASKED: lights with fewer phases can never choose a phase they do not have
    (applied when acting, in the PPO ratio and in the entropy);
  * max_lanes defaults to the map's largest light (env.max_lanes), neighbour slots to its largest neighbour count;
  * imports neighbors_map.py (the old neighbors.py is still used by the DQN agents).


Multi-Agent PPO with Centralized Training and Decentralized Execution (CTDE).

Supports:
    - 16 traffic-signal agents (or any number detected by the environment).
    - Variable incoming lanes per intersection.
    - Maximum 10 incoming lanes per intersection.
    - Shared actor across all intersections.
    - Centralized critic using information from all intersections.
    - Optional neighbour communication (levels 0, 1, 2).

Raw observation from env._own():
    [waiting for n lanes,
     vehicle counts for n lanes,
     phase one-hot,
     time since phase change]

Padded observation:
    [10 waiting values,
     10 vehicle-count values,
     10 lane-presence mask values,
     phase one-hot,
     time since phase change]

For two phases, the padded observation contains 33 values.
"""

import numpy as np
import torch
import torch.nn as nn

from neighbors_map import augment, extra_dim, base_dim

torch.set_num_threads(1)


MAX_LANES = 10


def mlp(input_dim, hidden_dim, output_dim, activation=nn.Tanh):
    """Create a simple multilayer perceptron."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        activation(),
        nn.Linear(hidden_dim, hidden_dim),
        activation(),
        nn.Linear(hidden_dim, output_dim)
    )


class MAPPO:

    def __init__(
        self,
        env,
        level=2,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.95,
        lam=0.95,
        clip=0.2,
        epochs=4,
        minibatch=256,
        ent_coef=0.01,
        vf_coef=0.5,
        hidden=64,
        max_grad=0.5,
        max_lanes=None
    ):

        # ---------- Environment information ----------

        self.ag = list(env.agents)
        self.nbrs = env.neighbors

        self.n_ag = len(self.ag)
        self.n_act = env.n_act                      # largest number of actions of any light
        self.n_acts = dict(env.n_acts)              # actions of each light
        self.level = level
        self.max_lanes = env.max_lanes if max_lanes is None else max_lanes
        self.n_slots = max(len(v) for v in self.nbrs.values()) if level else 0
        self.mask = torch.from_numpy(np.stack([env.action_mask(a) for a in self.ag]))   # [n_agents, n_act], 1 = legal

        if self.n_ag == 0:
            raise ValueError("The environment contains no traffic-signal agents.")

        if level not in (0, 1, 2):
            raise ValueError("Communication level must be 0, 1, or 2.")

        # Store the actual incoming lane count of every intersection.
        self.lane_counts = {
            a: len(env.lanes[a]) for a in self.ag
        }

        # Validate the maximum lane count.
        for a, count in self.lane_counts.items():
            if count > self.max_lanes:
                raise ValueError(
                    f"Intersection {a} has {count} incoming lanes. "
                    f"Maximum supported lanes = {self.max_lanes}."
                )

            if count == 0:
                raise ValueError(
                    f"Intersection {a} has no incoming lanes."
                )

        # ---------- Observation dimensions ----------

        # Fixed padded observation size.
        self.base_dim = base_dim(
            n_act=self.n_act,
            max_lanes=self.max_lanes
        )

        # Actor receives own observation plus neighbour features.
        self.obs_dim = (
            self.base_dim + extra_dim(level, self.n_act, self.n_slots)
        )

        # Critic receives all agents' observations plus agent identity.
        self.state_dim = (
            self.n_ag * self.base_dim + self.n_ag
        )

        # ---------- Neural networks ----------

        # Shared actor: one observation -> action logits.
        self.actor = mlp(
            self.obs_dim,
            hidden,
            self.n_act
        )

        # Centralized critic: global state + agent ID -> value.
        self.critic = mlp(
            self.state_dim,
            2 * hidden,
            1
        )

        # ---------- Optimizers ----------

        self.opt_a = torch.optim.Adam(
            self.actor.parameters(),
            lr=lr_actor
        )

        self.opt_c = torch.optim.Adam(
            self.critic.parameters(),
            lr=lr_critic
        )

        # ---------- PPO parameters ----------

        self.gamma = gamma
        self.lam = lam
        self.clip = clip
        self.epochs = epochs
        self.minibatch = minibatch

        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad = max_grad

        # One-hot identity for each agent.
        self.eye = np.eye(
            self.n_ag,
            dtype=np.float32
        )

        self._reset_buffer()

        print("MAPPO initialized successfully.")
        print("Number of agents:", self.n_ag)
        print("Maximum incoming lanes:", self.max_lanes)
        print("Padded observation dimension:", self.base_dim)
        print("Actor input dimension:", self.obs_dim)
        print("Centralized critic input dimension:", self.state_dim)

    # ==========================================================
    # Observation preprocessing
    # ==========================================================

    def _prepare_observations(self, obs):
        """
        Convert variable-length observations into fixed-size vectors.

        Each intersection is padded to:
            10 waiting values
            10 vehicle-count values
            10 lane-presence mask values
            n_act phase values
            1 time value
        """

        prepared = {}

        for agent in self.ag:

            if agent not in obs:
                raise ValueError(
                    f"Observation missing for intersection {agent}."
                )

            raw = np.asarray(
                obs[agent],
                dtype=np.float32
            ).reshape(-1)

            n = self.lane_counts[agent]

            k = self.n_acts[agent]
            expected = 2 * n + k + 1

            if len(raw) != expected:
                raise ValueError(
                    f"Invalid observation size for {agent}: "
                    f"expected {expected}, received {len(raw)}."
                )

            # Extract actual lane information.
            waiting = raw[:n]
            counts = raw[n:2 * n]

            # Extract phase and time information.
            phase = np.zeros(self.n_act, dtype=np.float32)
            phase[:k] = raw[2 * n:2 * n + k]
            time_value = raw[-1:]

            # Create fixed-size lane arrays.
            waiting_pad = np.zeros(
                self.max_lanes,
                dtype=np.float32
            )

            counts_pad = np.zeros(
                self.max_lanes,
                dtype=np.float32
            )

            lane_mask = np.zeros(
                self.max_lanes,
                dtype=np.float32
            )

            # Fill only the positions corresponding to real lanes.
            waiting_pad[:n] = waiting
            counts_pad[:n] = counts
            lane_mask[:n] = 1.0

            # Combine into one fixed-size observation.
            fixed_obs = np.concatenate([
                waiting_pad,
                counts_pad,
                lane_mask,
                phase,
                time_value
            ]).astype(np.float32)

            if len(fixed_obs) != self.base_dim:
                raise ValueError(
                    f"Incorrect padded observation size for {agent}."
                )

            prepared[agent] = fixed_obs

        return prepared

    def _actor_obs(self, prepared):
        """
        Prepare decentralized actor inputs.

        Neighbour information is appended according to
        the selected communication level.
        """

        augmented = augment(
            prepared,
            self.nbrs,
            self.level,
            n_lanes=self.max_lanes,
            n_act=self.n_act,
            n_slots=self.n_slots
        )

        return np.stack([
            augmented[a] for a in self.ag
        ]).astype(np.float32)

    def _critic_obs(self, prepared):
        """
        Construct centralized critic inputs.

        Global state contains all intersections' padded observations.
        Each critic input also contains the identity of its agent.
        """

        global_state = np.concatenate([
            prepared[a] for a in self.ag
        ]).astype(np.float32)

        critic_input = np.concatenate([
            np.tile(global_state, (self.n_ag, 1)),
            self.eye
        ], axis=1)

        return critic_input.astype(np.float32)

    # ==========================================================
    # Experience buffer
    # ==========================================================

    def _reset_buffer(self):

        self.B = dict(
            o=[],
            s=[],
            a=[],
            logp=[],
            v=[],
            r=[]
        )

    # ==========================================================
    # Action selection
    # ==========================================================

    def act(self, obs, eps=0.0, explore=False):
        """
        Select an action for every intersection.

        During training:
            explore=True -> sample from stochastic policy.

        During evaluation:
            explore=False -> select highest-probability action.

        eps is retained for compatibility with the existing
        DQN evaluation interface. PPO does not use epsilon-greedy.
        """

        prepared = self._prepare_observations(obs)

        actor_input = self._actor_obs(prepared)

        O = torch.from_numpy(actor_input)

        with torch.no_grad():

            logits = self.actor(O).masked_fill(self.mask == 0, -1e9)     # illegal actions get ~zero probability

            if not explore:

                actions = logits.argmax(dim=1)

                return {
                    agent: int(actions[i])
                    for i, agent in enumerate(self.ag)
                }

            # Stochastic policy for PPO exploration.
            dist = torch.distributions.Categorical(
                logits=logits
            )

            actions = dist.sample()

            # Centralized critic values.
            critic_input = self._critic_obs(prepared)

            S = torch.from_numpy(critic_input)

            values = self.critic(S).squeeze(1)

            log_probs = dist.log_prob(actions)

            # Save information required for PPO updates.
            self._last = (
                O.numpy().copy(),
                S.numpy().copy(),
                actions.numpy().copy(),
                log_probs.numpy().copy(),
                values.numpy().copy()
            )

        return {
            agent: int(actions[i])
            for i, agent in enumerate(self.ag)
        }

    # ==========================================================
    # Store rewards and transitions
    # ==========================================================

    def store(self, rewards):
        """
        Store rewards immediately after:
            act(explore=True)
            env.step(actions)
        """

        if not hasattr(self, "_last"):
            raise RuntimeError(
                "Call act(explore=True) before store()."
            )

        o, s, a, lp, v = self._last

        r = np.array([
            rewards[agent] for agent in self.ag
        ], dtype=np.float32)

        for key, value in zip(
            ("o", "s", "a", "logp", "v", "r"),
            (o, s, a, lp, v, r)
        ):
            self.B[key].append(value)

        del self._last

    # ==========================================================
    # Generalized Advantage Estimation and PPO update
    # ==========================================================

    def update(self, last_obs):
        """
        Update actor and critic after an episode.

        Uses:
            - Generalized Advantage Estimation (GAE)
            - Clipped PPO objective
            - Advantage normalization
            - Entropy regularization
            - Value-function loss
            - Gradient clipping
        """

        T = len(self.B["r"])

        if T == 0:
            raise RuntimeError(
                "No training transitions available for update."
            )

        rewards = np.stack(self.B["r"])
        values = np.stack(self.B["v"])

        # Estimate value of the final observation.
        final_prepared = self._prepare_observations(last_obs)

        with torch.no_grad():

            final_state = torch.from_numpy(
                self._critic_obs(final_prepared)
            )

            v_last = self.critic(
                final_state
            ).squeeze(1).numpy()

        # ---------- Calculate GAE ----------

        advantages = np.zeros_like(rewards)

        gae = np.zeros(
            self.n_ag,
            dtype=np.float32
        )

        for t in reversed(range(T)):

            if t == T - 1:
                next_value = v_last
            else:
                next_value = values[t + 1]

            delta = (
                rewards[t]
                + self.gamma * next_value
                - values[t]
            )

            gae = (
                delta
                + self.gamma * self.lam * gae
            )

            advantages[t] = gae

        returns = advantages + values

        # ---------- Convert stored data to tensors ----------

        O = torch.from_numpy(
            np.concatenate(self.B["o"], axis=0)
        )

        S = torch.from_numpy(
            np.concatenate(self.B["s"], axis=0)
        )

        A = torch.from_numpy(
            np.concatenate(self.B["a"], axis=0)
        )

        LP = torch.from_numpy(
            np.concatenate(self.B["logp"], axis=0)
        )

        MASK = self.mask.repeat(T, 1)                  # rows are ordered step by step, agent by agent

        ADV = torch.from_numpy(
            advantages.reshape(-1).astype(np.float32)
        )

        RET = torch.from_numpy(
            returns.reshape(-1).astype(np.float32)
        )

        # Normalize advantages.
        ADV = (
            ADV - ADV.mean()
        ) / (
            ADV.std(unbiased=False) + 1e-8
        )

        # ---------- PPO optimization ----------

        n = len(A)

        stats = []

        for _ in range(self.epochs):

            permutation = torch.randperm(n)

            for start in range(0, n, self.minibatch):

                indices = permutation[
                    start:start + self.minibatch
                ]

                # ---------- Actor update ----------

                dist = torch.distributions.Categorical(
                    logits=self.actor(O[indices]).masked_fill(MASK[indices] == 0, -1e9)
                )

                new_log_probs = dist.log_prob(
                    A[indices]
                )

                ratio = torch.exp(
                    new_log_probs - LP[indices]
                )

                surrogate1 = (
                    ratio * ADV[indices]
                )

                surrogate2 = (
                    torch.clamp(
                        ratio,
                        1 - self.clip,
                        1 + self.clip
                    ) * ADV[indices]
                )

                surrogate = torch.min(
                    surrogate1,
                    surrogate2
                )

                entropy = dist.entropy().mean()

                actor_loss = (
                    -surrogate.mean()
                    - self.ent_coef * entropy
                )

                self.opt_a.zero_grad()

                actor_loss.backward()

                nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    self.max_grad
                )

                self.opt_a.step()

                # ---------- Critic update ----------

                predicted_values = self.critic(
                    S[indices]
                ).squeeze(1)

                critic_loss = (
                    self.vf_coef
                    * nn.functional.smooth_l1_loss(
                        predicted_values,
                        RET[indices]
                    )
                )

                self.opt_c.zero_grad()

                critic_loss.backward()

                nn.utils.clip_grad_norm_(
                    self.critic.parameters(),
                    self.max_grad
                )

                self.opt_c.step()

                stats.append((
                    actor_loss.item(),
                    critic_loss.item(),
                    entropy.item()
                ))

        # Clear the experience buffer.
        self._reset_buffer()

        mean_stats = np.mean(stats, axis=0)

        return dict(
            actor_loss=float(mean_stats[0]),
            critic_loss=float(mean_stats[1]),
            entropy=float(mean_stats[2])
        )

    # ==========================================================
    # Save and load model
    # ==========================================================

    def state_dict(self):

        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict()
        }

    def load(self, state):

        self.actor.load_state_dict(
            state["actor"]
        )

        self.critic.load_state_dict(
            state["critic"]
        )

    # ==========================================================
    # Hyperparameters and model information
    # ==========================================================

    def hyperparams(self):

        return dict(
            algo="MAPPO",
            number_of_agents=self.n_ag,
            max_incoming_lanes=self.max_lanes,
            number_of_phases=self.n_act,
            communication_level=self.level,
            neighbour_slots=self.n_slots,
            actions_per_light=sorted(set(self.n_acts.values())),
            action_masking=True,
            lr_actor=self.opt_a.param_groups[0]["lr"],
            lr_critic=self.opt_c.param_groups[0]["lr"],
            gamma=self.gamma,
            gae_lambda=self.lam,
            clip=self.clip,
            ppo_epochs=self.epochs,
            minibatch=self.minibatch,
            ent_coef=self.ent_coef,
            hidden=self.actor[0].out_features,
            base_obs_dim=self.base_dim,
            actor_obs_dim=self.obs_dim,
            critic_state_dim=self.state_dim
        )