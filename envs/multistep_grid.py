#!/usr/bin/env python3
# MultiStepGrid-v0: 12 actions (Right/Left/Up/Down × 1,2,3 cells) with "time" and "brake" metrics.
# - time increment per step: dt = 1 / m  (m ∈ {1,2,3})
# - braking penalty (reported via info): max(0, prev_m - m)
# Rewards: step -1, goal +100, cliff -100 (no time/brake in reward by default; see params).

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register


class MultiStepGrid(gym.Env):
    """
    State: (x,y) encoded -> int.
    Actions (0..11): [R1,R2,R3, L1,L2,L3, U1,U2,U3, D1,D2,D3]
    Movement clamps to walls. If the path crosses any cliff cell, episode terminates (cliff penalty).
    Info dict contains:
      - 'dt'            : 1/m
      - 'mag'           : chosen magnitude m
      - 'brake_penalty' : max(0, prev_m - m)
    """

    metadata = {"render_modes": ["rgb_array", "ansi"], "render_fps": 8}

    def __init__(
        self,
        size=8,
        max_steps=200,
        step_cost=-1.0,
        cliff_penalty=-100.0,
        goal_reward=100.0,
        time_cost=0.0,  # if >0, add -time_cost*dt to reward
        brake_cost=0.0,  # if >0, add -brake_cost*brake_penalty to reward
        render_mode=None,
        seed=None,
    ):
        super().__init__()
        self.W = self.H = int(size)
        self.max_steps = int(max_steps)
        self.step_cost = float(step_cost)
        self.cliff_penalty = float(cliff_penalty)
        self.goal_reward = float(goal_reward)
        self.time_cost = float(time_cost)
        self.brake_cost = float(brake_cost)
        self.render_mode = render_mode

        self.action_space = gym.spaces.Discrete(12)
        self.observation_space = gym.spaces.Discrete(self.W * self.H)

        self.start = (0, self.H - 1)
        self.goal = (self.W - 1, self.H - 1)
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._pos = (0, 0)
        self._prev_mag = 0  # previous move magnitude (0 at episode start)

    # ----- helpers -----
    @staticmethod
    def _decode_action(a: int):
        # Order: [R1,R2,R3, L1,L2,L3, U1,U2,U3, D1,D2,D3]
        dir_idx, mag_idx = divmod(int(a), 3)  # dir ∈ {0..3}, mag_idx ∈ {0,1,2}
        m = mag_idx + 1
        if dir_idx == 0:
            dx, dy = +1, 0  # Right
        elif dir_idx == 1:
            dx, dy = -1, 0  # Left
        elif dir_idx == 2:
            dx, dy = 0, -1  # Up
        else:
            dx, dy = 0, +1  # Down
        return dx, dy, m

    def _enc(self, x, y):  # (x,y) -> int
        return y * self.W + x

    def _is_cliff(self, x, y):
        return (y == self.H - 1) and (1 <= x <= self.W - 2)

    def _path_and_end(self, x, y, dx, dy, m):
        """Return list of traversed cells (excluding start), and end cell, clamped at walls."""
        path = []
        cx, cy = x, y
        for _ in range(m):
            nx = int(np.clip(cx + dx, 0, self.W - 1))
            ny = int(np.clip(cy + dy, 0, self.H - 1))
            if (nx, ny) == (cx, cy):  # hit wall, no further movement
                break
            path.append((nx, ny))
            cx, cy = nx, ny
        return path, (cx, cy)

    # ----- Gym API -----
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._pos = self.start
        self._prev_mag = 0
        return self._enc(*self._pos), {}

    def step(self, a):
        x, y = self._pos
        dx, dy, m = self._decode_action(a)
        path, end_pos = self._path_and_end(x, y, dx, dy, m)

        self._t += 1
        r = self.step_cost
        terminated = truncated = False

        # Check cliffs along the path (including final cell)
        for px, py in path:
            if self._is_cliff(px, py):
                r = self.cliff_penalty
                self._pos = (px, py)
                terminated = True
                break

        if not terminated:
            self._pos = end_pos
            if self._pos == self.goal:
                r = self.goal_reward
                terminated = True
            elif self._t >= self.max_steps:
                truncated = True

        # Time and braking metrics (reported via info)
        dt = 1.0 / float(m)  # faster for bigger moves
        brake = max(0, self._prev_mag - m)  # penalty when slowing down
        self._prev_mag = m

        # Optional shaping (off by default)
        if self.time_cost > 0.0:
            r += -self.time_cost * dt
        if self.brake_cost > 0.0:
            r += -self.brake_cost * brake

        obs = self._enc(*self._pos)
        info = {"dt": dt, "mag": m, "brake_penalty": brake}
        return obs, r, terminated, truncated, info

    def render(self):
        if self.render_mode == "ansi":
            x, y = self._pos
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
            return "\n".join(rows)
        elif self.render_mode == "rgb_array":
            cell = 24
            W, H = self.W * cell, self.H * cell
            img = np.ones((H, W, 3), dtype=np.uint8) * 240
            img[::cell, :, :] = 180
            img[:, ::cell, :] = 180
            for j in range(self.H):
                for i in range(self.W):
                    y0, y1 = j * cell + 1, (j + 1) * cell - 1
                    x0, x1 = i * cell + 1, (i + 1) * cell - 1
                    if (i, j) == self.goal:
                        img[y0:y1, x0:x1] = [120, 220, 120]
                    elif self._is_cliff(i, j):
                        img[y0:y1, x0:x1] = [230, 120, 120]
            x, y = self._pos
            y0, y1 = y * cell + 3, (y + 1) * cell - 3
            x0, x1 = x * cell + 3, (x + 1) * cell - 3
            img[y0:y1, x0:x1] = [90, 140, 240]
            return img
        return None


# Register (idempotent)
try:
    register(id="MultiStepGrid-v0", entry_point=lambda **kw: MultiStepGrid(**kw))
except Exception:
    pass
