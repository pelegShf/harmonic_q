#!/usr/bin/env python3
# Orchestrator: train (standard/harmonic), optional videos, new aggregate plots, and cross-seed compare.
# Plots folder structure: plots/{standard|harmonic|compare}/<env_key>/...

import argparse, os, glob, numpy as np
import matplotlib

matplotlib.use("Agg")

from hq.core import train_one, evaluate_greedy
from hq.video import record_video, record_trajectories_grid
from hq.plotting import runs_with_mean_plot, mean_se_plot

# Register custom envs (safe no-ops if unused)
import envs.velocity_grid  # VelocityGrid-v0
import envs.multistep_grid  # MultiStepGrid-v0
import envs.windy_gridworld  # WindyGrid-*
import envs.mountaincar_tab  # MountainCarTab-v0
import envs.duration_actions  # VelocityGridHold-v0, WindyGridHold-v0


def safe_env_key(env_id: str) -> str:
    return env_id.replace("/", "_").replace(":", "_").replace("-", "_")


def parse_seeds(s: str):
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = map(int, part.split("-", 1))
            out += list(range(min(a, b), max(a, b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def load_csvs(log_root: str, variant: str, seeds, metric: str):
    """Load arrays for metric ∈ {'returns','steps','time'} for given variant/seeds."""
    fname = f"{variant}_{metric}.csv"
    paths = [os.path.join(log_root, variant, f"s{sd}", fname) for sd in seeds]
    paths += glob.glob(os.path.join(log_root, variant, "s*", fname))
    uniq = [p for p in dict.fromkeys(paths) if os.path.isfile(p)]
    arrs = []
    for p in sorted(uniq):
        try:
            arrs.append(np.loadtxt(p, delimiter=",", dtype=float).reshape(-1))
        except Exception:
            pass
    return arrs


def task_train(args, variants, seeds):
    for v in variants:
        for sd in seeds:
            print(f"\n== train {v} seed={sd} on {args.env} ==")
            Q, rets, steps, times = train_one(
                args.env,
                v,
                args.episodes,
                args.alpha,
                args.gamma,
                args.eps_start,
                args.eps_end,
                args.eps_decay_episodes,
                args.max_steps_per_ep,
                sd,
            )
            eval_avg = evaluate_greedy(
                args.env, Q, episodes=args.eval_episodes, seed=sd + 10_000
            )

            out_dir = os.path.join(args.log_root, v, f"s{sd}")
            os.makedirs(out_dir, exist_ok=True)
            np.savetxt(os.path.join(out_dir, f"{v}_returns.csv"), rets, delimiter=",")
            np.savetxt(os.path.join(out_dir, f"{v}_steps.csv"), steps, delimiter=",")
            np.savetxt(os.path.join(out_dir, f"{v}_time.csv"), times, delimiter=",")
            if args.save_q:
                np.save(os.path.join(out_dir, f"Q_{v}_seed{sd}.npy"), Q)

            print(
                f"[{v}] avg_ret={np.mean(rets):.3f} | avg_steps={np.mean(steps):.2f} "
                f"| avg_time={np.mean(times):.3f} | eval_avg_ret={eval_avg:.3f} | saved {out_dir}"
            )

            safe_env = safe_env_key(args.env)

            if args.record_video:
                record_video(
                    args.env,
                    Q,
                    out_dir=os.path.join(args.videos_root, v, f"s{sd}"),
                    name_prefix=f"{safe_env}_final_policy_seed{sd}",
                    seed=args.video_seed or sd,
                    max_steps=args.video_max_steps,
                )

            if args.traj_grid:
                grid_dir = os.path.join(args.videos_root, v, f"s{sd}", "grids")
                os.makedirs(grid_dir, exist_ok=True)
                out_path = os.path.join(
                    grid_dir,
                    f"{safe_env}_grid_{args.traj_grid_rows}x{args.traj_grid_cols}.mp4",
                )
                seed0 = (
                    args.traj_grid_seed0
                    if args.traj_grid_seed0 is not None
                    else (sd * 1000 + 777)
                )
                record_trajectories_grid(
                    env_id=args.env,
                    Q=Q,
                    out_path=out_path,
                    rows=args.traj_grid_rows,
                    cols=args.traj_grid_cols,
                    seed0=seed0,
                    max_steps=args.traj_grid_max_steps,
                    fps=args.traj_grid_fps,
                )


def task_plot(args, variants, seeds):
    env_key = safe_env_key(args.env)
    for v in variants:
        # Load all runs for each metric
        for metric, ylabel, fname in [
            ("returns", "Return", "returns.png"),
            ("steps", "Steps", "steps.png"),
            ("time", "Time (Σ dt)", "time.png"),
        ]:
            runs = load_csvs(args.log_root, v, seeds, metric=metric)
            if not runs:
                print(f"[plot] skipping {v}/{metric} (missing logs)")
                continue
            out_dir = os.path.join(args.plots_root, v, env_key)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, fname)
            title = f"{v.capitalize()} Q-learning on {args.env}"
            runs_with_mean_plot(runs, args.smooth, title, out_path, ylabel=ylabel)
            print(f"[saved] {out_path}")


def task_compare(args, variants, seeds):
    # force both for compare
    env_key = safe_env_key(args.env)
    std_ret = load_csvs(args.log_root, "standard", seeds, metric="returns")
    har_ret = load_csvs(args.log_root, "harmonic", seeds, metric="returns")
    std_stp = load_csvs(args.log_root, "standard", seeds, metric="steps")
    har_stp = load_csvs(args.log_root, "harmonic", seeds, metric="steps")
    std_tim = load_csvs(args.log_root, "standard", seeds, metric="time")
    har_tim = load_csvs(args.log_root, "harmonic", seeds, metric="time")

    out_dir = os.path.join(args.plots_root, "compare", env_key)
    os.makedirs(out_dir, exist_ok=True)

    if std_ret and har_ret:
        T = min(min(map(len, std_ret)), min(map(len, har_ret)))
        std_ret = np.stack([a[:T] for a in std_ret])
        har_ret = np.stack([a[:T] for a in har_ret])
        out = os.path.join(out_dir, "mean_se_returns.png")
        mean_se_plot(
            std_ret,
            har_ret,
            smooth=args.smooth,
            title=f"Mean ± SE Returns ({args.env})",
            out_path=out,
            ylabel="Return",
        )
        print(f"[saved] {out}")
    else:
        print("[compare] skipping returns (missing logs)")

    if std_stp and har_stp:
        T = min(min(map(len, std_stp)), min(map(len, har_stp)))
        std_stp = np.stack([a[:T] for a in std_stp])
        har_stp = np.stack([a[:T] for a in har_stp])
        out = os.path.join(out_dir, "mean_se_steps.png")
        mean_se_plot(
            std_stp,
            har_stp,
            smooth=args.smooth,
            title=f"Mean ± SE Steps ({args.env})",
            out_path=out,
            ylabel="Steps",
        )
        print(f"[saved] {out}")
    else:
        print("[compare] skipping steps (missing logs)")

    if std_tim and har_tim:
        T = min(min(map(len, std_tim)), min(map(len, har_tim)))
        std_tim = np.stack([a[:T] for a in std_tim])
        har_tim = np.stack([a[:T] for a in har_tim])
        out = os.path.join(out_dir, "mean_se_time.png")
        mean_se_plot(
            std_tim,
            har_tim,
            smooth=args.smooth,
            title=f"Mean ± SE Time ({args.env})",
            out_path=out,
            ylabel="Time (Σ dt)",
        )
        print(f"[saved] {out}")
    else:
        print("[compare] skipping time (missing logs)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--env",
        default="VelocityGrid-v0",
        help="Env id (VelocityGrid-v0, MultiStepGrid-v0, WindyGrid-v0, MountainCarTab-v0, ...)",
    )
    ap.add_argument(
        "--variants", choices=["standard", "harmonic", "both"], default="both"
    )
    ap.add_argument("--seeds", default="0-4")
    # training
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.05)
    ap.add_argument("--eps_decay_episodes", type=int, default=3500)
    ap.add_argument("--max_steps_per_ep", type=int, default=300)
    ap.add_argument("--eval_episodes", type=int, default=50)
    # io
    ap.add_argument("--log_root", default="logs")
    ap.add_argument("--plots_root", default="plots")
    ap.add_argument("--videos_root", default="videos")
    ap.add_argument("--save_q", action="store_true")
    # single rollout video
    ap.add_argument("--record_video", action="store_true")
    ap.add_argument("--video_seed", type=int, default=None)
    ap.add_argument("--video_max_steps", type=int, default=300)
    # trajectories grid
    ap.add_argument(
        "--traj_grid",
        action="store_true",
        help="Save an R×C grid of sampled greedy rollouts",
    )
    ap.add_argument("--traj_grid_rows", type=int, default=3)
    ap.add_argument("--traj_grid_cols", type=int, default=3)
    ap.add_argument("--traj_grid_fps", type=int, default=8)
    ap.add_argument("--traj_grid_max_steps", type=int, default=300)
    ap.add_argument(
        "--traj_grid_seed0",
        type=int,
        default=None,
        help="Base seed for grid sampling (default uses seed*1000+777)",
    )
    # plots/compare
    ap.add_argument("--smooth", type=int, default=151)
    ap.add_argument(
        "--tag", default=""
    )  # kept for backward compat; not used in paths now
    # task
    ap.add_argument(
        "--task", choices=["train", "plot", "compare", "all"], default="all"
    )
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    variants = ["standard", "harmonic"] if args.variants == "both" else [args.variants]

    if args.task in ("train", "all"):
        task_train(args, variants, seeds)
    if args.task in ("plot", "all"):
        task_plot(args, variants, seeds)
    if args.task in ("compare", "all"):
        task_compare(args, variants, seeds)


if __name__ == "__main__":
    main()
