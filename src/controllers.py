"""Non-learning baselines: fixed-time and Max-Pressure."""
import numpy as np
import libsumo as traci


class FixedTime:
    """Baseline 1: cycle through the green phases with equal, constant green durations."""
    name = "fixed_time"

    def __init__(self, green_s=30):
        self.green_s = green_s

    def reset(self, env): pass

    def act(self, env, obs):
        p = (env.t // self.green_s) % env.n_act
        return {a: int(p) for a in env.agents}


class MaxPressure:
    """Reference controller (Varaiya 2013): serve the phase with the highest pressure
    = sum over its movements of (upstream queue - downstream queue). Keeps the current phase on ties."""
    name = "max_pressure"

    def reset(self, env): pass

    def act(self, env, obs):
        acts = {}
        for a in env.agents:
            press = []
            for pl in env.phase_links[a]:
                press.append(sum(traci.lane.getLastStepHaltingNumber(i) - traci.lane.getLastStepHaltingNumber(o) for i, o in pl))
            best = int(np.argmax(press))
            acts[a] = env.cur[a] if press[best] <= press[env.cur[a]] else best
        return acts
