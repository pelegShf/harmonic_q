#!/usr/bin/env python3
# MountainCarTab-v0: tabular wrapper around Gymnasium's MountainCar-v0
# - Discretizes the continuous observation (position, velocity) into bins.
# - Observation space becomes Discrete(n_pos_bins * n_vel_bins), action space stays Discrete(3).
# - Optional goal_bonus to add a positive terminal reward spike when the goal is reached.

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register


class MountainCarTab(gym.Env):
    metadata = {"render_modes": ["rgb_array", "ansi"], "render_fps": 30}

    def __init__(
        self,
        bins: tuple[int, int] = (18, 14),  # (pos_bins, vel_bins) — common choice
        goal_bonus: float = 0.0,  # add to reward on terminal success (default 0 = classic)
        render_mode: str | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self._base = gym.make("MountainCar-v0", render_mode=render_mode)
        if seed is not None:
            # seed underlying env RNG
            try:
                self._base.reset(seed=seed)
            except TypeError:
                pass

        self._bins = (int(bins[0]), int(bins[1]))
        self.goal_bonus = float(goal_bonus)
        self.render_mode = render_mode

        # Build discretization boundaries from base env limits
        low = self._base.observation_space.low.astype(float)
        high = self._base.observation_space.high.astype(float)
        # Create (bins_i - 1) internal cut points for np.digitize
        self._edges = [
            np.linspace(low[0], high[0], self._bins[0] + 1)[1:-1],  # position cuts
            np.linspace(low[1], high[1], self._bins[1] + 1)[1:-1],  # velocity cuts
        ]

        nS = self._bins[0] * self._bins[1]
        self.observation_space = gym.spaces.Discrete(nS)
        self.action_space = self._base.action_space  # Discrete(3)

        self._last_obs = None  # continuous last obs

    # ---------- helpers ----------
    def _to_indices(self, obs: np.ndarray) -> tuple[int, int]:
        # np.digitize returns bin index in [0..n_bins]; with our edges it yields [0..n_bins-1]
        i_pos = int(np.digitize(obs[0], self._edges[0], right=False))
        i_vel = int(np.digitize(obs[1], self._edges[1], right=False))
        # Safety: clip just in case of numeric edge cases
        i_pos = int(np.clip(i_pos, 0, self._bins[0] - 1))
        i_vel = int(np.clip(i_vel, 0, self._bins[1] - 1))
        return i_pos, i_vel

    def _enc(self, i_pos: int, i_vel: int) -> int:
        return i_pos * self._bins[1] + i_vel

    # ---------- Gym API ----------
    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            try:
                obs, info = self._base.reset(seed=seed)
            except TypeError:
                obs, info = self._base.reset()
        else:
            obs, info = self._base.reset()
        self._last_obs = np.asarray(obs, dtype=float)
        i_pos, i_vel = self._to_indices(self._last_obs)
        return self._enc(i_pos, i_vel), info

    def step(self, a: int):
        obs, r, terminated, truncated, info = self._base.step(int(a))
        self._last_obs = np.asarray(obs, dtype=float)

        # Add optional terminal bonus on success
        if terminated:
            r += self.goal_bonus

        # Provide a 'dt' so your Time metric works (equals 1 per step here)
        info = dict(info)
        info.setdefault("dt", 1.0)

        i_pos, i_vel = self._to_indices(self._last_obs)
        return (
            self._enc(i_pos, i_vel),
            float(r),
            bool(terminated),
            bool(truncated),
            info,
        )

    def render(self):
        # Defer to base renderer (rgb_array or ansi)
        return self._base.render()

    def close(self):
        self._base.close()


# -------------------- Register (idempotent) --------------------
def _register():
    try:
        register(id="MountainCarTab-v0", entry_point=lambda **kw: MountainCarTab(**kw))
    except Exception:
        pass


_register()
