#!/usr/bin/env python3
# WindyGrid-v0: classic windy gridworld (Sutton & Barto).
# - Grid: width=10, height=7
# - Start=(0,3), Goal=(7,3)
# - Wind (per column): [0,0,0,1,1,1,2,2,1,0]  (pushes upward)
# Variants auto-registered:
#   - "WindyGrid-v0"         : 4 actions (L,R,U,D), deterministic wind
#   - "WindyGridKings-v0"    : 8 actions (King's moves), deterministic wind
#   - "WindyGridStoch-v0"    : 4 actions, stochastic wind (±1 with prob p/2 each)
#
# Rewards: default step_cost=-1, reaching goal terminates (reward += goal_reward, default 0)
# Renderers: "rgb_array" (simple tile art) and "ansi" (ASCII)

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register


class WindyGrid(gym.Env):
    metadata = {"render_modes": ["rgb_array", "ansi"], "render_fps": 8}

    def __init__(
        self,
        width: int = 10,
        height: int = 7,
        wind: list[int] | None = None,
        start: tuple[int, int] = (0, 3),
        goal: tuple[int, int] = (7, 3),
        king_moves: bool = False,         # 4 actions if False, 8 actions if True
        stochastic_wind_p: float = 0.0,   # probability wind is offset by ±1 (split evenly)
        step_cost: float = -1.0,
        goal_reward: float = 0.0,
        max_steps: int = 200,
        render_mode: str | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.W = int(width)
        self.H = int(height)
        self.wind = np.array(wind if wind is not None else [0,0,0,1,1,1,2,2,1,0], dtype=int)
        assert len(self.wind) == self.W, "len(wind) must equal width"
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.king_moves = bool(king_moves)
        self.p_stoch = float(stochastic_wind_p)
        self.step_cost = float(step_cost)
        self.goal_reward = float(goal_reward)
        self.max_steps = int(max_steps)
        self.render_mode = render_mode

        self.action_space = gym.spaces.Discrete(8 if self.king_moves else 4)
        self.observation_space = gym.spaces.Discrete(self.W * self.H)

        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._pos = self.start

    # ---------- helpers ----------
    def _enc(self, x: int, y: int) -> int:
        return y * self.W + x

    def _moves(self):
        if self.king_moves:
            # 8-neighbors (dx,dy): E,W,N,S, NE,NW,SE,SW (order not important)
            return [(+1,0), (-1,0), (0,-1), (0,+1), (+1,-1), (-1,-1), (+1,+1), (-1,+1)]
        else:
            # 4-neighbors: E,W,N,S   (note: y grows downward; wind pushes toward smaller y)
            return [(+1,0), (-1,0), (0,-1), (0,+1)]

    # ---------- Gym API ----------
    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._pos = self.start
        return self._enc(*self._pos), {}

    def step(self, a: int):
        x, y = self._pos
        dx, dy = self._moves()[int(a)]

        # 1) agent's intended move
        nx = int(np.clip(x + dx, 0, self.W - 1))
        ny = int(np.clip(y + dy, 0, self.H - 1))

        # 2) apply column wind (pushes UP: toward smaller y)
        w = int(self.wind[nx])  # wind determined by *new* column
        if self.p_stoch > 0.0:
            if self._rng.random() < self.p_stoch:
                w += int(self._rng.integers(-1, 2))  # add -1,0,+1 uniformly

        ny = int(np.clip(ny - w, 0, self.H - 1))

        self._t += 1
        self._pos = (nx, ny)

        r = self.step_cost
        terminated = (self._pos == self.goal)
        if terminated:
            r += self.goal_reward
        truncated = (self._t >= self.max_steps)

        obs = self._enc(*self._pos)
        info = {"wind_applied": w}
        return obs, r, terminated, truncated, info

    def render(self):
        if self.render_mode == "ansi":
            x, y = self._pos
            rows = []
            for j in range(self.H):
                row = []
                for i in range(self.W):
                    if (i, j) == (x, y): row.append("A")
                    elif (i, j) == self.goal: row.append("G")
                    else:
                        row.append(str(self.wind[i]) if j == 0 else ".")
                rows.append(" ".join(row))
            return "\n".join(rows)
        elif self.render_mode == "rgb_array":
            cell = 24
            Wp, Hp = self.W * cell, self.H * cell
            img = np.ones((Hp, Wp, 3), dtype=np.uint8) * 240
            # grid
            img[::cell, :, :] = 180
            img[:, ::cell, :] = 180
            # draw wind strengths in top row via color shade
            for i in range(self.W):
                y0, y1 = 1, cell - 1
                x0, x1 = i * cell + 1, (i + 1) * cell - 1
                shade = 230 - int(25 * self.wind[i])
                shade = int(np.clip(shade, 120, 230))
                img[y0:y1, x0:x1] = [shade, 230, 230]
            # goal
            gx, gy = self.goal
            y0, y1 = gy * cell + 1, (gy + 1) * cell - 1
            x0, x1 = gx * cell + 1, (gx + 1) * cell - 1
            img[y0:y1, x0:x1] = [120, 220, 120]
            # agent
            x, y = self._pos
            y0, y1 = y * cell + 3, (y + 1) * cell - 3
            x0, x1 = x * cell + 3, (x + 1) * cell - 3
            img[y0:y1, x0:x1] = [90, 140, 240]
            return img
        return None


# -------------------- Register common variants (idempotent) --------------------

def _register():
    try:
        register(id="WindyGrid-v0",
                 entry_point=lambda **kw: WindyGrid(**kw))
    except Exception:
        pass
    try:
        register(id="WindyGridKings-v0",
                 entry_point=lambda **kw: WindyGrid(king_moves=True, **kw))
    except Exception:
        pass
    try:
        register(id="WindyGridStoch-v0",
                 entry_point=lambda **kw: WindyGrid(stochastic_wind_p=0.1, **kw))
    except Exception:
        pass

_register()
