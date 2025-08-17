# envs/frozenlake_tab.py
from __future__ import annotations
import gymnasium as gym
from gymnasium.envs.registration import register

class FrozenLakeTab(gym.Env):
    """
    Tabular FrozenLake with reward knobs.
    Defaults match classic: step 0, hole 0, goal +1.
    """
    metadata = {"render_modes": ["rgb_array", "ansi"], "render_fps": 50}

    def __init__(self,
                 map_name: str | None = "8x8",
                 desc: list[str] | None = None,   # custom map rows (optional)
                 is_slippery: bool = True,
                 step_cost: float = 0.0,
                 hole_penalty: float = 0.0,
                 goal_reward: float = 1.0,
                 render_mode: str | None = None,
                 seed: int | None = None):
        super().__init__()
        # Build base env (either by name or custom desc)
        make_kwargs = dict(is_slippery=is_slippery, render_mode=render_mode)
        if desc is not None:
            make_kwargs["desc"] = desc
        else:
            make_kwargs["map_name"] = map_name

        self._base = gym.make("FrozenLake-v1", **make_kwargs)
        self._seed = seed
        self._step_cost = float(step_cost)
        self._hole_penalty = float(hole_penalty)
        self._goal_reward = float(goal_reward)
        self.render_mode = render_mode

        self.observation_space = self._base.observation_space
        self.action_space = self._base.action_space

    # ---- Gym API ----
    def reset(self, *, seed: int | None = None, options=None):
        if seed is None:
            seed = self._seed
        obs, info = self._base.reset(seed=seed, options=options)
        info = dict(info or {})
        info["dt"] = 1.0
        return obs, info

    def step(self, a: int):
        s2, r, terminated, truncated, info = self._base.step(int(a))
        # Classic env returns r=1 on goal, r=0 otherwise.
        if terminated:
            r_new = self._goal_reward if r > 0.0 else self._hole_penalty
        else:
            r_new = self._step_cost

        info = dict(info or {})
        info["dt"] = 1.0
        return s2, float(r_new), bool(terminated), bool(truncated), info

    def render(self):
        return self._base.render()

    def close(self):
        self._base.close()

# Register on import (idempotent)
try:
    register(id="FrozenLakeTab-v0", entry_point=lambda **kw: FrozenLakeTab(**kw))
except Exception:
    pass
