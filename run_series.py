#!/usr/bin/env python3
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed

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
import envs.cartpole_tab
import envs.cliff_tab
import envs.frozenlake_tab
import envs.duration_actions

from hq.core import train_one, evaluate_greedy
from hq.plotting import runs_with_mean_plot

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

def _train_seed_job(args_tuple):
    """Child-process job: train one (model,seed) and return arrays."""
    (env_id, variant, episodes, alpha, gamma_m,
     eps_start, eps_end, eps_decay, max_steps, sd) = args_tuple

    # Ensure envs are registered in the child too
    import envs.velocity_grid
    import envs.multistep_grid
    import envs.windy_gridworld
    import envs.mountaincar_tab
    import envs.duration_actions
    try:
        import envs.cartpole_tab  # optional
    except Exception:
        pass

    from hq.core import train_one
    return sd, train_one(env_id, variant, episodes, alpha, gamma_m,
                         eps_start, eps_end, eps_decay, max_steps, sd)



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


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def model_id_from_file(path: str, cfg: dict) -> str:
    # Prefer explicit id in file, else stem
    if "id" in cfg and cfg["id"]:
        return str(cfg["id"])
    return os.path.splitext(os.path.basename(path))[0]



def run_experiment(exp: dict, series: dict, models_catalog: Dict[str, dict], task: str):
    # --- 1) Load env config ---
    if "env_file" in exp:
        env_cfg = load_yaml(exp["env_file"])
        env_id = env_cfg["id"]
        env_kwargs = env_cfg.get("env_kwargs", {})

        max_steps_default = int(env_cfg.get("max_steps_per_ep", series.get("max_steps_per_ep", 300)))
        eval_eps_default  = int(env_cfg.get("eval_episodes",   series.get("eval_episodes",   50)))
    else:
        # backward compat (old style)
        env_id = exp["env"]
        max_steps_default = int(series.get("max_steps_per_ep", 300))
        eval_eps_default  = int(series.get("eval_episodes",   50))

    # --- 2) Experiment-level knobs override env defaults ---
    seeds     = parse_seeds(exp.get("seed_spec", series.get("seed_spec", "0-0")))
    episodes  = int(exp.get("episodes", series.get("episodes", 4000)))
    max_steps = int(exp.get("max_steps_per_ep", max_steps_default))
    eval_eps  = int(exp.get("eval_episodes",    eval_eps_default))

    # --- 3) Load models ---
    loaded_models: Dict[str, dict] = {}
    if "model_files" in exp:
        # new style: model files listed per experiment
        for mpath in exp["model_files"]:
            mcfg = load_yaml(mpath)
            mid = model_id_from_file(mpath, mcfg)
            mcfg = dict(mcfg); mcfg.setdefault("name", mid)
            loaded_models[mid] = mcfg
    else:
        # old style: look up by name in series.models
        for mid in exp.get("models", []):
            if mid not in models_catalog:
                avail = ", ".join(sorted(models_catalog.keys()))
                raise KeyError(f"Model '{mid}' not found in series.models. Available: [{avail}]")
            loaded_models[mid] = dict(models_catalog[mid])

    # --- 4) Per-experiment overrides (optional) ---
    for mid, ov in (exp.get("overrides") or {}).items():
        if mid in loaded_models:
            loaded_models[mid] = deep_update(loaded_models[mid], ov)

    # --- 5) Train/plot/compare (unchanged logic) ---
    base_log_root = series.get("log_root", "logs")
    gamma_global  = float(series.get("gamma", 0.99))

    for mid, mcfg in loaded_models.items():
        variant   = mcfg["variant"]
        alpha     = float(mcfg.get("alpha", series.get("alpha", 0.2)))
        gamma_m   = float(mcfg.get("gamma", gamma_global))
        eps       = mcfg.get("eps", {})
        eps_start = float(eps.get("start", series.get("eps_start", 1.0)))
        eps_end   = float(eps.get("end",   series.get("eps_end",   0.05)))
        eps_decay = int(eps.get("decay_episodes", series.get("eps_decay_episodes", episodes - 500)))

        log_root   = os.path.join(base_log_root, exp["id"])
        model_root = os.path.join(log_root, mid, variant)

        if task in ("train", "all"):
            w = int(series.get("workers", 1))
            if w > 1:
                jobs = []
                for sd in seeds:
                    jobs.append((env_id, variant, episodes, alpha, gamma_m,
                                eps_start, eps_end, eps_decay, max_steps, sd))
                with ProcessPoolExecutor(max_workers=w) as ex:
                    fut2seed = {ex.submit(_train_seed_job, j): j[-1] for j in jobs}
                    for fut in as_completed(fut2seed):
                        sd, (Q, rets, steps, times) = fut.result()
                        out_dir = os.path.join(model_root, f"s{sd}")
                        ensure_dir(out_dir)
                        save_csv(os.path.join(out_dir, f"{variant}_returns.csv"), rets)
                        save_csv(os.path.join(out_dir, f"{variant}_steps.csv"),   steps)
                        save_csv(os.path.join(out_dir, f"{variant}_time.csv"),    times)
            else:
                for sd in seeds:
                    Q, rets, steps, times = train_one(
                        env_id, variant, episodes, alpha, gamma_m,
                        eps_start, eps_end, eps_decay, max_steps, sd,
                        env_kwargs=env_kwargs
                    )
                    out_dir = os.path.join(model_root, f"s{sd}")
                    ensure_dir(out_dir)
                    save_csv(os.path.join(out_dir, f"{variant}_returns.csv"), rets)
                    save_csv(os.path.join(out_dir, f"{variant}_steps.csv"),   steps)
                    save_csv(os.path.join(out_dir, f"{variant}_time.csv"),    times)


        if task in ("plot", "all"):
            plot_model_aggregates({"env": env_id, "id": exp["id"]}, series, mid, mcfg)

    if task in ("compare", "all"):
        cmp_models = {mid: {"variant": mcfg["variant"]} for mid, mcfg in loaded_models.items()}
        compare_models_within_experiment({"env": env_id, "id": exp["id"]}, series, cmp_models)

def resolve_experiments(series: dict) -> list[dict]:
    """
    Accepts either:
      - experiments: [ {...}, {...} ]           # current format (list)
      - experiments: {id: {...}, id2: {...}}    # catalog map
    Optionally filters by:
      - experiments_to_run: [id, id2, ...]
    Returns a list of experiment dicts with an 'id' field set.
    """
    exps = series.get("experiments", [])
    run_ids = series.get("experiments_to_run")

    # Case A: catalog map {id: cfg}
    if isinstance(exps, dict):
        catalog = exps
        ids = run_ids or list(catalog.keys())
        out = []
        for eid in ids:
            if eid not in catalog:
                avail = ", ".join(sorted(catalog.keys()))
                raise KeyError(f"Experiment '{eid}' not in catalog. Available: [{avail}]")
            item = dict(catalog[eid])  # shallow copy
            item.setdefault("id", eid)
            out.append(item)
        return out

    # Case B: existing list format
    if isinstance(exps, list):
        if run_ids:
            wanted = set(run_ids)
            return [e for e in exps if e.get("id") in wanted]
        return exps

    # Fallback
    return []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="series.yaml", help="Path to main YAML")
    ap.add_argument("--task", choices=["train", "plot", "compare", "all"], default="all")
    ap.add_argument("--workers", type=int, default=None,
                help="Parallel processes for training seeds (default: series.yaml 'workers' or 1)")

    args = ap.parse_args()

    with open(args.series, "r", encoding="utf-8") as f:
        series = yaml.safe_load(f)
    workers = int(args.workers if args.workers is not None else series.get("workers", 1))
    series["workers"] = workers  # stash for run_experiment

    models_catalog = series.get("models", {})  # used only by old-style experiments
    experiments = resolve_experiments(series)  # <-- use your existing helper

    for exp_entry in experiments:
        # keep support for your old 'file:' style (optional)
        if "file" in exp_entry:
            base_exp = load_yaml(exp_entry["file"])
            exp_cfg  = deep_update(base_exp, exp_entry)
        else:
            exp_cfg  = dict(exp_entry)

        exp_cfg.setdefault("id", exp_entry.get("id", "exp"))
        run_experiment(exp_cfg, series, models_catalog, task=args.task)



if __name__ == "__main__":
    main()
