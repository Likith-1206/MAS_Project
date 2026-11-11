"""Team B: neighbour-information features (communication levels 0, 1, 2).

The env is always created with comm_level=0. This module builds the extra features from the
level-0 observations, so changing the communication scheme never touches the control panel.

Level 0: nothing added                         -> 19 numbers
Level 1: + mean queue of each neighbour        -> 19 + 4  = 23
Level 2: + neighbour's current phase (one-hot) -> 19 + 12 = 31
Slots are in the fixed order N, E, S, W; a missing neighbour (edge of the grid) gives zeros.
"""
import numpy as np

N_LANES, N_ACT = 8, 2   # incoming lanes per intersection, phases (see env_spec.md)


def slot_dim(level, n_act=N_ACT):
    """Numbers per neighbour slot."""
    return 0 if level == 0 else 1 + (n_act if level >= 2 else 0)


def extra_dim(level, n_act=N_ACT):
    """Numbers added to the own observation (4 slots)."""
    return 4 * slot_dim(level, n_act)


def augment(obs, neighbors, level, n_lanes=N_LANES, n_act=N_ACT):
    """obs: {agent: level-0 vector}; neighbors: env.neighbors ({agent: [N, E, S, W]}).
    Returns {agent: vector with neighbour features appended}."""
    if level == 0:
        return obs
    out = {}
    for a, own in obs.items():
        parts = [own]
        for nb in neighbors[a]:
            if nb is None:
                parts.append(np.zeros(slot_dim(level, n_act), np.float32))
                continue
            o = obs[nb]
            f = [o[:n_lanes].mean()]                              # neighbour mean queue
            if level >= 2:
                f += list(o[2 * n_lanes: 2 * n_lanes + n_act])    # neighbour phase one-hot
            parts.append(np.array(f, np.float32))
        out[a] = np.concatenate(parts).astype(np.float32)
    return out