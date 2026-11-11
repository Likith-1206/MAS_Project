"""Template: show ANY controller in the visual simulation.

  python src/simulate.py osm/mumbai.osm --controller plugin:src/plugins/policy_template.py

simulate.py calls make_policy(env, map_dir) once, then act(obs) every 5 simulated seconds.
  env   env_map.SumoMapEnv of the map: env.agents (light ids), env.n_acts[a] (green phases of light a),
        env.lanes[a], env.phase_links[a] (incoming->outgoing lane pairs each phase serves), env.neighbors[a],
        env.pad_obs(obs), env.action_masks(), env.t (simulated seconds)
  obs   {light: observation vector} exactly as in training (halting/cap, cars/cap per lane, phase one-hot, time in phase)
  act   must return {light: phase index in 0 .. env.n_acts[light]-1} for EVERY light

To show RAPID-Light: load its network + weights here and translate obs -> actions the same way its evaluation
script does (e.g. build the phase tokens from env.phase_links, apply the safety shield, take the argmax).
This file ships a simple working policy (longest queue first, minimum green 10 s) so the pipeline can be tried.
"""
import numpy as np

NAME = "Longest-queue-first (template)"


def make_policy(env, map_dir):
    lane = None
    min_green = 10

    def act(obs):
        nonlocal lane
        if lane is None:
            lane = env_map_traci().lane
        out = {}
        for a in env.agents:
            if env.since[a] < min_green:
                out[a] = env.cur[a]
                continue
            q = [sum(lane.getLastStepHaltingNumber(i) for i, _ in links) for links in env.phase_links[a]]
            out[a] = int(np.argmax(q))
        return out
    return act


def env_map_traci():
    import env_map
    return env_map.traci
