# envs/cartpole_tab.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

class CartPoleTab(gym.Env):
    """
    Tabular wrapper over Gymnasium CartPole-v1 by discretizing (x, x_dot, theta, theta_dot).
    Rewards: +1 per step until termination (same as CartPole-v1).
    """
    metadata = {"render_modes": ["rgb_array", "ansi"], "render_fps": 50}

    def __init__(self,
                 bins=(9, 9, 19, 19),
                 obs_lo=None,
                 obs_hi=None,
                 render_mode=None):
        # Base env
        self._render_mode = render_mode
        self.base = gym.make("CartPole-v1", render_mode=render_mode)

        # Reasonable clip ranges (CartPole thresholds are smaller; we clip a bit wider):
        # x in [-2.4, 2.4], theta in ~[-0.2095, 0.2095] rad; velocities are free
        lo_default = np.array([-2.4,  -3.0, -0.25, -3.5], dtype=np.float32)
        hi_default = np.array([ 2.4,   3.0,  0.25,  3.5], dtype=np.float32)
        self.lo = np.array(obs_lo, dtype=np.float32) if obs_lo is not None else lo_default
        self.hi = np.array(obs_hi, dtype=np.float32) if obs_hi is not None else hi_default

        self.bins = tuple(int(b) for b in bins)
        assert len(self.bins) == 4, "bins must be a 4-tuple for (x, x_dot, theta, theta_dot)"

        # Bin edges for np.digitize (length = n_bins - 1)
        self.edges = [
            np.linspace(self.lo[i], self.hi[i], self.bins[i] + 1, dtype=np.float32)[1:-1]
            for i in range(4)
        ]

        nS = int(np.prod(self.bins))
        self.observation_space = spaces.Discrete(nS)
        self.action_space = spaces.Discrete(2)  # left or right

        self._last_obs = None  # continuous obs cache

    # --------- helpers ---------
    def _to_indices(self, obs):
        # clip, then digitize each dim to [0, n_i-1]
        o = np.clip(np.asarray(obs, dtype=np.float32), self.lo, self.hi)
        idxs = []
        for i in range(4):
            j = int(np.digitize(o[i], self.edges[i]))
            if j < 0: j = 0
            if j >= self.bins[i]: j = self.bins[i] - 1
            idxs.append(j)
        return idxs

    def _encode(self, idxs):
        # mixed-radix encoding to a single integer
        s = 0
        for i, b in zip(idxs, self.bins):
            s = s * b + i
        return int(s)

    # --------- Gym API ---------
    def reset(self, *, seed=None, options=None):
        obs, info = self.base.reset(seed=seed, options=options)
        self._last_obs = obs
        s = self._encode(self._to_indices(obs))
        # provide a small dt for logging symmetry with other envs
        info = dict(info or {})
        info["dt"] = 1.0
        return s, info

    def step(self, action):
        s_next, r, terminated, truncated, info = self.base.step(int(action))
        self._last_obs = s_next
        s_disc = self._encode(self._to_indices(s_next))
        info = dict(info or {})
        info["dt"] = 1.0
        return s_disc, float(r), bool(terminated), bool(truncated), info

    def render(self):
        if self._render_mode == "ansi":
            return None
        return self.base.render()

    def close(self):
        return self.base.close()

# Register on import
try:
    register(id="CartPoleTab-v0",
             entry_point=lambda **kw: CartPoleTab(**kw))
except Exception:
    pass
