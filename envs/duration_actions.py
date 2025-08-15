#!/usr/bin/env python3
# DurationActionWrapper: expands action space to (base_action, duration) pairs.
# Returns cumulative reward over the held duration and puts 'tau' and 'dt' in info.

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register


class DurationActionWrapper(gym.Env):
    """
    New action space: Discrete(nA * len(durations)).
    Decoding: a_idx -> base_a, tau in durations.
    Repeats base action for tau steps (or until termination).
    Accumulates *undiscounted* reward R and exposes info['tau']=actual_steps_taken,
    info['dt']=sum of child infos' dt (or tau if absent).
    """

    metadata = {"render_modes": ["rgb_array", "ansi"]}

    def __init__(self, env, durations=(1, 2, 3)):
        super().__init__()
        self.env = env
        self.durations = tuple(int(x) for x in durations)
        self.nA_base = env.action_space.n
        self.action_space = gym.spaces.Discrete(self.nA_base * len(self.durations))
        self.observation_space = env.observation_space
        self.render_mode = getattr(env, "render_mode", None)

    def _decode(self, a_idx: int):
        k = len(self.durations)
        base_a = int(a_idx) // k
        tau = self.durations[int(a_idx) % k]
        return base_a, tau

    def reset(self, **kw):
        return self.env.reset(**kw)

    def step(self, a_idx: int):
        base_a, tau = self._decode(a_idx)
        total_r = 0.0
        total_dt = 0.0
        steps = 0
        obs = None
        terminated = truncated = False
        info_last = {}

        for i in range(tau):
            obs, r, term, trunc, info = self.env.step(base_a)
            total_r += float(r)
            total_dt += float(info.get("dt", 1.0))
            steps += 1
            info_last = info
            if term or trunc:
                terminated, truncated = bool(term), bool(trunc)
                break

        # Merge info and attach SMDP duration/time
        info_out = dict(info_last)
        info_out["tau"] = steps
        info_out["dt"] = total_dt if total_dt > 0 else steps

        return obs, total_r, terminated, truncated, info_out

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


# --- Convenience registrations (idempotent). Import this file to enable them. ---


def register_velocitygrid_hold():
    from . import velocity_grid

    try:
        register(
            id="VelocityGridHold-v0",
            entry_point=lambda **kw: DurationActionWrapper(
                gym.make("VelocityGrid-v0", **kw), durations=(1, 2, 3)
            ),
        )
    except Exception:
        pass


def register_windy_hold():
    from . import windy_gridworld

    try:
        register(
            id="WindyGridHold-v0",
            entry_point=lambda **kw: DurationActionWrapper(
                gym.make("WindyGrid-v0", **kw), durations=(1, 2, 3)
            ),
        )
    except Exception:
        pass


# Auto-register on import
try:
    import gymnasium as gym  # noqa

    register_velocitygrid_hold()
    register_windy_hold()
except Exception:
    pass
