#!/usr/bin/env python3
# Custom inertial gridworld with cliffs; registers as "VelocityGrid-v0" on import.

from __future__ import annotations
import numpy as np, gymnasium as gym
from gymnasium.envs.registration import register


class VelocityGrid(gym.Env):
    """
    Discrete grid with velocity (inertia).
    State: (x,y,vx,vy) encoded as a single integer.
    Actions: 0=none, 1=up, 2=down, 3=left, 4=right  (accelerations)
    Dynamics: v <- clip(v + a), pos <- clip(pos + v)
    Rewards: step -1, goal +100, cliff -100
    Layout: start=(0,H-1), goal=(W-1,H-1), cliff bottom row cells 1..W-2
    """

    metadata = {"render_modes": ["rgb_array", "ansi"], "render_fps": 8}

    def __init__(
        self,
        size=8,
        max_speed=4,
        max_steps=200,
        slip_prob=0.05,
        step_cost=-1.0,
        cliff_penalty=-100.0,
        goal_reward=100.0,
        render_mode=None,
        seed=None,
    ):
        super().__init__()
        self.W = self.H = int(size)
        self.max_speed = int(max_speed)
        self.v_levels = 2 * self.max_speed + 1
        self.max_steps = int(max_steps)
        self.slip_prob = float(slip_prob)
        self.step_cost = float(step_cost)
        self.cliff_penalty = float(cliff_penalty)
        self.goal_reward = float(goal_reward)
        self.render_mode = render_mode

        self.action_space = gym.spaces.Discrete(5)
        self.nS = self.W * self.H * self.v_levels * self.v_levels
        self.observation_space = gym.spaces.Discrete(self.nS)

        self.start = (0, self.H - 1)
        self.goal = (self.W - 1, self.H - 1)
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._state_tuple = (0, 0, 0, 0)

    # encoding helpers
    def _enc(self, x, y, vx, vy):
        vx_i = vx + self.max_speed
        vy_i = vy + self.max_speed
        return ((y * self.W + x) * self.v_levels + vx_i) * self.v_levels + vy_i

    def _dec(self, s):
        vy_i = s % self.v_levels
        s //= self.v_levels
        vx_i = s % self.v_levels
        s //= self.v_levels
        xy = s
        x = xy % self.W
        y = xy // self.W
        vx = vx_i - self.max_speed
        vy = vy_i - self.max_speed
        return x, y, vx, vy

    def _is_cliff(self, x, y):
        return (y == self.H - 1) and (1 <= x <= self.W - 2)

    # Gym API
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        x, y = self.start
        vx = vy = 0
        self._state_tuple = (x, y, vx, vy)
        return self._enc(x, y, vx, vy), {}

    def step(self, a):
        x, y, vx, vy = self._state_tuple

        ax = ay = 0
        if a == 1:
            ay = -1
        elif a == 2:
            ay = +1
        elif a == 3:
            ax = -1
        elif a == 4:
            ax = +1

        # slip/noise occasionally
        if self._rng.random() < self.slip_prob:
            ax2, ay2 = self._rng.integers(-1, 2), self._rng.integers(-1, 2)
            if ax == 0:
                ax = int(ax2)
            if ay == 0:
                ay = int(ay2)

        vx = int(np.clip(vx + ax, -self.max_speed, self.max_speed))
        vy = int(np.clip(vy + ay, -self.max_speed, self.max_speed))
        nx = int(np.clip(x + vx, 0, self.W - 1))
        ny = int(np.clip(y + vy, 0, self.H - 1))

        self._t += 1
        r = self.step_cost
        terminated = truncated = False

        if (nx, ny) == self.goal:
            r = self.goal_reward
            terminated = True
        elif self._is_cliff(nx, ny):
            r = self.cliff_penalty
            terminated = True
        elif self._t >= self.max_steps:
            truncated = True

        self._state_tuple = (nx, ny, vx, vy)
        return self._enc(nx, ny, vx, vy), r, terminated, truncated, {}

    def render(self):
        if self.render_mode == "ansi":
            x, y, vx, vy = self._state_tuple
            rows = []
            for j in range(self.H):
                row = []
                for i in range(self.W):
                    if (i, j) == (x, y):
                        row.append("A")
                    elif (i, j) == self.goal:
                        row.append("G")
                    elif self._is_cliff(i, j):
                        row.append("C")
                    else:
                        row.append(".")
                rows.append(" ".join(row))
            return "\n".join(rows) + f"\n(v=({vx},{vy}))"

        elif self.render_mode == "rgb_array":
            cell = 24
            W, H = self.W * cell, self.H * cell
            img = np.ones((H, W, 3), dtype=np.uint8) * 240
            # grid
            img[::cell, :, :] = 180
            img[:, ::cell, :] = 180
            # paint goal/cliff
            for j in range(self.H):
                for i in range(self.W):
                    y0, y1 = j * cell + 1, (j + 1) * cell - 1
                    x0, x1 = i * cell + 1, (i + 1) * cell - 1
                    if (i, j) == self.goal:
                        img[y0:y1, x0:x1] = [120, 220, 120]
                    elif self._is_cliff(i, j):
                        img[y0:y1, x0:x1] = [230, 120, 120]
            # agent
            x, y, vx, vy = self._state_tuple
            y0, y1 = y * cell + 3, (y + 1) * cell - 3
            x0, x1 = x * cell + 3, (x + 1) * cell - 3
            img[y0:y1, x0:x1] = [90, 140, 240]
            return img

        return None


# Register (idempotent)
try:
    register(id="VelocityGrid-v0", entry_point=lambda **kw: VelocityGrid(**kw))
except Exception:
    pass  # already registered
