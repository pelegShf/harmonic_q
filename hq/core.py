#!/usr/bin/env python3
# Core tabular Q-learning: harmonic update, training loop, greedy eval.
from __future__ import annotations
from typing import Tuple, List
import numpy as np
import gymnasium as gym

def epsilon_greedy(Q: np.ndarray, s: int, eps: float, rng: np.random.Generator) -> int:
    return int(rng.integers(Q.shape[1])) if rng.random() < eps else int(np.argmax(Q[s]))

def safe_harmonic_update(q_sa: float, target: float, alpha: float,
                         eps_div: float = 1e-8, shift_eps: float = 1e-6) -> float:
    # Weighted harmonic step with positivity shift.
    sft = min(q_sa, target, 0.0) - shift_eps
    q_ = q_sa - sft
    y_ = target - sft
    inv_new = (1.0 - alpha) / max(q_, eps_div) + alpha / max(y_, eps_div)
    q_new_ = 1.0 / max(inv_new, eps_div)
    return q_new_ + sft

def train_one(env_id: str,
              variant: str,
              episodes: int,
              alpha: float,
              gamma: float,
              eps_start: float,
              eps_end: float,
              eps_decay_episodes: int,
              max_steps_per_ep: int,
              seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      Q:      [nS, nA]
      returns: per-episode sum of rewards
      steps:   per-episode number of env steps
      times:   per-episode accumulated 'time' (sum of info['dt'] if provided; else ~steps)
    """
    env = gym.make(env_id)
    assert hasattr(env.observation_space, "n") and hasattr(env.action_space, "n"), \
        "This trainer expects discrete state/action spaces."

    nS, nA = env.observation_space.n, env.action_space.n
    Q = np.zeros((nS, nA), dtype=float)
    rng = np.random.default_rng(seed)
    rets: List[float] = []
    steps: List[int] = []
    times: List[float] = []

    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep)
        ret = 0.0
        step_count = 0
        ep_time = 0.0

        t = min(1.0, ep / max(1, eps_decay_episodes))
        eps = eps_start + (eps_end - eps_start) * t

        for _ in range(max_steps_per_ep):
            a = epsilon_greedy(Q, s, eps, rng)
            s2, r, term, trunc, info = env.step(a)
            step_count += 1
            done = term or trunc

            y = r if done else r + gamma * np.max(Q[s2])
            if variant == "standard":
                Q[s, a] = (1.0 - alpha) * Q[s, a] + alpha * y
            elif variant == "harmonic":
                Q[s, a] = safe_harmonic_update(Q[s, a], y, alpha)
            else:
                raise ValueError("variant must be 'standard' or 'harmonic'")

            ret += r
            ep_time += float(info.get("dt", 1.0))  # default 1.0 if env doesn't provide
            s = s2
            if done:
                break

        rets.append(ret)
        steps.append(step_count)
        times.append(ep_time)
        if (ep + 1) % 100 == 0:
            print(f"[{variant}] ep {ep+1}/{episodes} avg_ret={np.mean(rets):.3f} "
                  f"avg_steps={np.mean(steps):.2f} avg_time={np.mean(times):.3f}")

    env.close()
    return Q, np.asarray(rets, float), np.asarray(steps, int), np.asarray(times, float)

def evaluate_greedy(env_id: str, Q: np.ndarray, episodes: int = 20, seed: int = 1337) -> float:
    env = gym.make(env_id)
    total = 0.0
    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep)
        ep_ret = 0.0
        while True:
            a = int(np.argmax(Q[s]))
            s, r, term, trunc, _ = env.step(a)
            ep_ret += r
            if term or trunc:
                break
        total += ep_ret
    env.close()
    return total / max(1, episodes)
