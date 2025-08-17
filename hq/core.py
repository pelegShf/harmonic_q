#!/usr/bin/env python3
# Core tabular Q-learning: harmonic update, training loop, greedy eval.
from __future__ import annotations
from typing import Tuple, List
import numpy as np
import gymnasium as gym


def make_alpha_getter(nS, nA, base_alpha: float,
                      schedule: str = "constant",
                      c: float = 1.0,
                      kappa: float = 0.5,
                      time_scaled: bool = False):
    """
    Returns alpha_t(s,a[,tau]) per-visit.
      schedule:
        - "constant"        : alpha_t = base_alpha
        - "1_over_n"        : alpha_t = 1 / (1 + N(s,a))
        - "c_over_c_plus_n" : alpha_t = c / (c + N(s,a))
        - "power"           : alpha_t = base_alpha / (1 + N(s,a))**kappa
      time_scaled:
        - if True, divides by tau for SMDP macro-actions (durations)
    """
    counts = np.zeros((nS, nA), dtype=np.int64)

    def alpha(s: int, a: int, tau: int = 1) -> float:
        n = counts[s, a]  # visits so far (before this update)
        if schedule == "1_over_n":
            a0 = 1.0 / max(1, n + 1)
        elif schedule == "c_over_c_plus_n":
            a0 = c / (c + n + 1e-9)
        elif schedule == "power":
            a0 = base_alpha / ((1 + n) ** kappa)
        else:  # "constant"
            a0 = base_alpha
        if time_scaled and tau > 1:
            a0 = a0 / float(tau)
        counts[s, a] = n + 1
        return float(a0)

    return alpha



def epsilon_greedy(Q: np.ndarray, s: int, eps: float, rng: np.random.Generator) -> int:
    return int(rng.integers(Q.shape[1])) if rng.random() < eps else int(np.argmax(Q[s]))


def safe_harmonic_update(
    q_sa: float,
    target: float,
    alpha: float,
    eps_div: float = 1e-8,
    shift_eps: float = 1e-6,
) -> float:
    # Weighted harmonic step with positivity shift.
    sft = min(q_sa, target, 0.0) - shift_eps
    q_ = q_sa - sft
    y_ = target - sft
    inv_new = (1.0 - alpha) / max(q_, eps_div) + alpha / max(y_, eps_div)
    q_new_ = 1.0 / max(inv_new, eps_div)
    return q_new_ + sft

def train_one(env_id: str, variant: str, episodes: int, alpha: float, gamma: float,
              eps_start: float, eps_end: float, eps_decay_episodes: int,
              max_steps_per_ep: int, seed: int,
              # schedules:
              alpha_schedule: str = "constant",   # {"constant","1_over_n","c_over_c_plus_n","power"}
              alpha_c: float = 1.0,
              alpha_kappa: float = 0.5,
              env_kwargs: dict | None = None):
    import gymnasium as gym, numpy as np
    rng = np.random.default_rng(seed)
    env = gym.make(env_id, **(env_kwargs or {}))

    nS = env.observation_space.n; nA = env.action_space.n
    Q = np.zeros((nS, nA), dtype=np.float64)

    # per-(s,a) Robbins–Monro stepsize
    _counts = np.zeros((nS, nA), dtype=np.int64)
    def alpha_of(s: int, a: int) -> float:
        n = _counts[s, a]
        if alpha_schedule == "1_over_n":
            a0 = 1.0 / max(1, n + 1)
        elif alpha_schedule == "c_over_c_plus_n":
            a0 = alpha_c / (alpha_c + n + 1e-9)
        elif alpha_schedule == "power":
            a0 = alpha / ((1 + n) ** alpha_kappa)
        else:
            a0 = alpha
        _counts[s, a] = n + 1
        return float(a0)

    rets, steps, times = [], [], []
    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep)
        ret = 0.0; step_count = 0; ep_time = 0.0
        t = min(1.0, ep / max(1, eps_decay_episodes))
        eps = eps_start + (eps_end - eps_start) * t

        for _ in range(max_steps_per_ep):
            a = epsilon_greedy(Q, s, eps, rng)
            s2, r, term, trunc, info = env.step(a)
            step_count += 1
            done = term or trunc

            y = r if done else r + gamma * np.max(Q[s2])
            alpha_eff = alpha_of(s, a)

            if variant == "standard":
                Q[s, a] = (1.0 - alpha_eff) * Q[s, a] + alpha_eff * y
            elif variant == "harmonic":
                Q[s, a] = safe_harmonic_update(Q[s, a], y, alpha_eff)
            else:
                raise ValueError("variant must be 'standard' or 'harmonic'")

            ret += r
            ep_time += float(info.get("dt", 1.0))
            s = s2
            if done:
                break

        rets.append(ret); steps.append(step_count); times.append(ep_time)
        if (ep + 1) % 1000 == 0:
            print(f"[{variant}] ep {ep+1}/{episodes} avg_ret={np.mean(rets):.3f} "
                  f"avg_steps={np.mean(steps):.2f} avg_time={np.mean(times):.3f}")

    env.close()
    return Q, np.asarray(rets, float), np.asarray(steps, int), np.asarray(times, float)


def evaluate_greedy(
    env_id: str, Q: np.ndarray, episodes: int = 20, seed: int = 1337
) -> float:
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
