"""Neighbour-information features for MAPPO on maps made by osm_import.py (env_map.SumoMapEnv).

Same idea as neighbors.py, with two changes for real maps:
  * the number of neighbour slots is `n_slots` (the largest neighbour count of any light on the map), not always 4.
    env.neighbors[a] lists the neighbours of a light ordered by compass bearing; unused slots are zeros.
  * the observation it reads is the PADDED one built in mappo_map.py (lane mask included).

Padded observation of one light (length base_dim):
    0 : L        halting/capacity per lane slot (L = max lanes of any light, zero padded)
    L : 2L       cars/capacity per lane slot
    2L: 3L       lane-presence mask (1 = real lane, 0 = padding)
    3L: 3L+A     current phase one-hot (A = largest number of actions of any light, zero padded)
    last         time in phase / 60

Communication levels:
    Level 0: own observation only
    Level 1: + for each neighbour slot: its mean queue
    Level 2: + for each neighbour slot: its mean queue and current phase one-hot
Missing neighbours are zeros. (Phase index k of one light does not mean the same movement as phase k of another light;
level 2 gives the phase as the neighbour's own index.)

The old neighbors.py is untouched: agents.py (DQN) still imports it.
"""
import numpy as np


def base_dim(n_act, max_lanes):
    """Size of a padded observation for one light."""
    return 3 * max_lanes + n_act + 1


def slot_dim(level, n_act):
    """Numbers per neighbour slot."""
    if level == 0:
        return 0
    return 1 + (n_act if level >= 2 else 0)


def extra_dim(level, n_act, n_slots):
    """Numbers added by all neighbour slots."""
    return n_slots * slot_dim(level, n_act)


def augment(obs, neighbors, level, n_lanes, n_act, n_slots):
    """obs: {agent: padded observation}; neighbors: env.neighbors ({agent: [neighbour ids]}).
    Returns {agent: observation with n_slots neighbour slots appended}."""
    if level not in (0, 1, 2):
        raise ValueError("Communication level must be 0, 1, or 2.")
    if level == 0:
        return obs
    mask_a, mask_b = 2 * n_lanes, 3 * n_lanes
    ph_a, ph_b = 3 * n_lanes, 3 * n_lanes + n_act
    out = {}
    for agent, own in obs.items():
        nbs = list(neighbors[agent])
        if len(nbs) > n_slots:
            raise ValueError(f"{agent} has {len(nbs)} neighbours but only {n_slots} slots")
        nbs += [None] * (n_slots - len(nbs))
        parts = [own]
        for nb in nbs:
            if nb is None:
                parts.append(np.zeros(slot_dim(level, n_act), dtype=np.float32))
                continue
            o = obs[nb]
            valid = o[mask_a:mask_b] > 0.5
            f = [float(np.mean(o[:n_lanes][valid])) if valid.any() else 0.0]
            if level >= 2:
                f += o[ph_a:ph_b].tolist()
            parts.append(np.asarray(f, dtype=np.float32))
        out[agent] = np.concatenate(parts).astype(np.float32)
    return out