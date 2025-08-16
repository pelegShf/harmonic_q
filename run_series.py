#!/usr/bin/env python3
from __future__ import annotations

"""
YAML-driven experiment runner for harmonic-vs-standard Q-learning.
- Define experiments in ./experiments/*.yaml
- Define models and the experiment list in ./series.yaml
- Run:  python run_series.py --series series.yaml --task all

Requires: pyyaml
pip install pyyaml
"""
import argparse, os
from typing import Dict, Any, List
import numpy as np
import yaml

# Register envs on import (safe no-ops if unused)
import envs.velocity_grid
import envs.multistep_grid
import envs.windy_gridworld
import envs.mountaincar_tab
import envs.duration_actions

from hq.core import train_one, evaluate_greedy
from hq.plotting import runs_with_mean_plot


def safe_env_key(env_id: str) -> str:
    return env_id.replace("/", "_").replace(":", "_").replace("-", "_")


def parse_seeds(spec: str) -> List[int]:
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            out.extend(range(min(a, b), max(a, b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def deep_update(base: dict, upd: dict) -> dict:
    out = dict(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def save_csv(path: str, arr):
    ensure_dir(os.path.dirname(path))
    np.savetxt(path, np.asarray(arr), delimiter=",")


def load_runs(log_root: str, model_id: str, variant: str, metric: str):
    runs = []
    model_root = os.path.join(log_root, model_id, variant)
    if not os.path.isdir(model_root):
        return runs
    for sd in sorted(os.listdir(model_root)):
        if not sd.startswith("s"):
            continue
        p = os.path.join(model_root, sd, f"{variant}_{metric}.csv")
        if os.path.isfile(p):
            try:
                runs.append(np.loadtxt(p, delimiter=",", dtype=float).reshape(-1))
            except Exception:
                pass
    return runs


def plot_model_aggregates(exp: dict, series: dict, model_id: str, model_cfg: dict):
    env_id = exp["env"]
    env_key = safe_env_key(env_id)
    plots_root = series.get("plots_root", "plots")
    log_root = os.path.join(series.get("log_root", "logs"), exp["id"])
    variant = model_cfg["variant"]
    model_name = model_cfg.get("name", model_id)

    for metric, ylabel, fname in [
        ("returns", "Return", "returns.png"),
        ("steps", "Steps", "steps.png"),
        ("time", "Time (Σ dt)", "time.png"),
    ]:
        runs = load_runs(log_root, model_id, variant, metric)
        if not runs:
            print(f"[plot:skip] {exp['id']}/{model_id}/{metric} (no logs)")
            continue
        out_dir = os.path.join(plots_root, variant, env_key)
        ensure_dir(out_dir)
        out_path = os.path.join(out_dir, f"{exp['id']}__{model_id}__{fname}")
        title = f"{model_name} on {env_id} [{exp['id']}]"
        runs_with_mean_plot(
            runs, series.get("smooth", 1), title, out_path, ylabel=ylabel
        )
        print(f"[plot] {out_path}")

def compare_models_within_experiment(exp: dict, series: dict, models: Dict[str, dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    def moving_average(x: np.ndarray, window: int) -> np.ndarray:
        if window is None or window <= 1 or window > len(x):
            return x
        w = int(window)
        left, right = w // 2, w - 1 - (w // 2)
        xpad = np.pad(np.asarray(x, float), (left, right), mode="edge")  # or "reflect"
        k = np.ones(w, dtype=float) / float(w)
        return np.convolve(xpad, k, mode="valid")

    env_id = exp["env"]
    env_key = safe_env_key(env_id)
    log_root = os.path.join(series.get("log_root", "logs"), exp["id"])
    plots_root = series.get("plots_root", "plots")
    out_dir = os.path.join(plots_root, "compare", env_key, exp["id"])
    ensure_dir(out_dir)

    # allow per-experiment override; else fall back to series-level smooth
    smooth = int(exp.get("smooth", series.get("smooth", 1)))

    for metric, ylabel, fname in [
        ("returns", "Return", "models_returns.png"),
        ("steps",   "Steps",  "models_steps.png"),
        ("time",    "Time (Σ dt)", "models_time.png"),
    ]:
        means = {}
        T = None
        for mid, mcfg in models.items():
            runs = load_runs(log_root, mid, mcfg["variant"], metric)
            if not runs:
                continue
            arr = np.stack(runs, axis=0)
            T = len(arr[0]) if T is None else min(T, arr.shape[1])
            means[mid] = arr

        if not means or T is None:
            print(f"[compare:skip] {exp['id']} {metric} (no data)")
            continue

        x = np.arange(1, T + 1)
        plt.figure(figsize=(10, 5))
        for mid, arr in sorted(means.items()):
            m = arr[:, :T].mean(axis=0)
            se = arr[:, :T].std(axis=0) / np.sqrt(arr.shape[0])  # Standard error
            m = moving_average(m, smooth)  # <-- apply smoothing to mean curve
            se = moving_average(se, smooth)  # <-- apply smoothing to SE curve
            
            plt.plot(x, m, label=mid)
            plt.fill_between(x, m - se, m + se, alpha=0.3)  # Add SE shading

        title = f"Across-seed mean — {env_id} [{exp['id']}]"
        if smooth > 1:
            title += f" (smoothed w={smooth})"
        plt.xlabel("Episode"); plt.ylabel(ylabel); plt.title(title)
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.legend()
        out_path = os.path.join(out_dir, fname)
        plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
        print(f"[compare] {out_path}")


def run_experiment(exp: dict, series: dict, models: Dict[str, dict], task: str):
    import inspect
    env_id = exp["env"]
    seeds = parse_seeds(exp.get("seed_spec", series.get("seed_spec", "0-0")))
    episodes = int(exp.get("episodes", series.get("episodes", 4000)))
    max_steps = int(exp.get("max_steps_per_ep", series.get("max_steps_per_ep", 300)))
    eval_eps = int(exp.get("eval_episodes", series.get("eval_episodes", 50)))
    gamma = float(series.get("gamma", 0.99))

    train_one_params = set(inspect.signature(train_one).parameters.keys())
    supports_sched = {"alpha_schedule", "alpha_c", "alpha_kappa"}.issubset(train_one_params)

    alpha_schedule_default = series.get("alpha_schedule", "constant")
    alpha_c_default       = float(series.get("alpha_c", 1.0))
    alpha_kappa_default   = float(series.get("alpha_kappa", 0.5))

    base_log_root = series.get("log_root", "logs")
    for mid, mcfg in models.items():
        variant = mcfg["variant"]
        alpha = float(mcfg.get("alpha", series.get("alpha", 0.2)))
        eps = mcfg.get("eps", {})
        eps_start = float(eps.get("start", series.get("eps_start", 1.0)))
        eps_end   = float(eps.get("end",   series.get("eps_end", 0.05)))
        eps_decay = int(eps.get("decay_episodes", series.get("eps_decay_episodes", episodes - 500)))
        gamma_m   = float(mcfg.get("gamma", gamma))

        alpha_schedule = mcfg.get("alpha_schedule", alpha_schedule_default)
        alpha_c        = float(mcfg.get("alpha_c", alpha_c_default))
        alpha_kappa    = float(mcfg.get("alpha_kappa", alpha_kappa_default))

        log_root = os.path.join(base_log_root, exp["id"])
        model_root = os.path.join(log_root, mid, variant)

        if task in ("train", "all"):
            for sd in seeds:
                args_common = [env_id, variant, episodes, alpha, gamma_m,
                               eps_start, eps_end, eps_decay, max_steps, sd]
                kwargs = {}
                if supports_sched:
                    kwargs.update(dict(alpha_schedule=alpha_schedule,
                                       alpha_c=alpha_c,
                                       alpha_kappa=alpha_kappa))
                Q, rets, steps, times = train_one(*args_common, **kwargs)
                out_dir = os.path.join(model_root, f"s{sd}")
                ensure_dir(out_dir)
                save_csv(os.path.join(out_dir, f"{variant}_returns.csv"), rets)
                save_csv(os.path.join(out_dir, f"{variant}_steps.csv"), steps)
                save_csv(os.path.join(out_dir, f"{variant}_time.csv"), times)
            print(f"[train] {exp['id']} :: {mid} ({variant}) — seeds={seeds} episodes={episodes}")

        if task in ("plot", "all"):
            plot_model_aggregates(exp, series, mid, mcfg)

    if task in ("compare", "all"):
        compare_models_within_experiment(exp, series, models)
        
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="series.yaml", help="Path to main YAML")
    ap.add_argument(
        "--task", choices=["train", "plot", "compare", "all"], default="all"
    )
    args = ap.parse_args()

    with open(args.series, "r", encoding="utf-8") as f:
        series = yaml.safe_load(f)

    model_defs = series.get("models", {})
    experiments = series.get("experiments", [])
    for exp_entry in experiments:
        if "file" in exp_entry:
            with open(exp_entry["file"], "r", encoding="utf-8") as ef:
                base_exp = yaml.safe_load(ef)
        else:
            base_exp = {}
        exp_cfg = deep_update(base_exp, exp_entry)
        exp_cfg.setdefault("id", exp_entry.get("id", "exp"))

        exp_models = {}
        for mid in exp_cfg.get("models", []):
            if mid not in model_defs:
                raise KeyError(f"Model '{mid}' not found in series.models")
            merged = deep_update(
                model_defs[mid], exp_cfg.get("overrides", {}).get(mid, {})
            )
            merged = dict(merged)
            merged.setdefault("name", mid)
            exp_models[mid] = merged

        run_experiment(exp_cfg, series, exp_models, task=args.task)


if __name__ == "__main__":
    main()
