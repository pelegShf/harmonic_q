# envs/cliff_tab.py
from __future__ import annotations
import gymnasium as gym
from gymnasium.envs.registration import register

class CliffWalkingTab(gym.Env):
    """
    Tabular CliffWalking with configurable rewards.
    Defaults match the classic task: step -1, cliff -100, goal 0.
    """
    metadata = {"render_modes": ["rgb_array", "ansi"], "render_fps": 50}

    def __init__(self,
                 step_cost: float = -1.0,
                 cliff_penalty: float = -100.0,
                 goal_reward: float = 0.0,
                 render_mode: str | None = None,
                 seed: int | None = None):
        super().__init__()
        self._base = gym.make("CliffWalking-v0", render_mode=render_mode)
        self._seed = seed
        self._step_cost = float(step_cost)
        self._cliff_penalty = float(cliff_penalty)
        self._goal_reward = float(goal_reward)
        self.render_mode = render_mode

        # spaces are already discrete; reuse base
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
        # Overwrite reward to requested scheme
        if r <= -100.0:                # fell off the cliff
            r_new = self._cliff_penalty
        elif terminated:               # reached goal
            r_new = self._goal_reward
        else:                          # regular move
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
    register(id="CliffWalkingTab-v0", entry_point=lambda **kw: CliffWalkingTab(**kw))
except Exception:
    pass
