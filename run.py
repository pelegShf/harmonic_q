#!/usr/bin/env python3
# Orchestrator: train (standard/harmonic), optional videos:
# - single rollout video
# - R×C grid of sampled greedy trajectories
# plus plots (returns/steps/time) and cross-seed comparisons.

import argparse, os, glob, numpy as np
import matplotlib
matplotlib.use("Agg")

from hq.core import train_one, evaluate_greedy
from hq.video import record_video, record_trajectories_grid
from hq.plotting import save_curve, mean_se_plot

# Register custom envs (safe no-ops if unused)
import envs.velocity_grid      # VelocityGrid-v0
import envs.multistep_grid     # MultiStepGrid-v0
import envs.windy_gridworld  # registers WindyGrid-* variants


def parse_seeds(s: str):
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part: continue
        if "-" in part:
            a,b = map(int, part.split("-",1))
            out += list(range(min(a,b), max(a,b)+1))
        else:
            out.append(int(part))
    return sorted(set(out))

def load_csvs(log_root: str, variant: str, seeds, metric: str):
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
            Q, rets, steps, times = train_one(args.env, v, args.episodes, args.alpha, args.gamma,
                                              args.eps_start, args.eps_end, args.eps_decay_episodes,
                                              args.max_steps_per_ep, sd)
            eval_avg = evaluate_greedy(args.env, Q, episodes=args.eval_episodes, seed=sd + 10_000)

            out_dir = os.path.join(args.log_root, v, f"s{sd}")
            os.makedirs(out_dir, exist_ok=True)
            np.savetxt(os.path.join(out_dir, f"{v}_returns.csv"), rets, delimiter=",")
            np.savetxt(os.path.join(out_dir, f"{v}_steps.csv"), steps, delimiter=",")
            np.savetxt(os.path.join(out_dir, f"{v}_time.csv"), times, delimiter=",")
            if args.save_q:
                np.save(os.path.join(out_dir, f"Q_{v}_seed{sd}.npy"), Q)

            print(f"[{v}] avg_ret={np.mean(rets):.3f} | avg_steps={np.mean(steps):.2f} "
                  f"| avg_time={np.mean(times):.3f} | eval_avg_ret={eval_avg:.3f} | saved {out_dir}")

            safe_env = args.env.replace("/", "_").replace(":", "_").replace("-", "_")

            if args.record_video:
                record_video(
                    args.env, Q,
                    out_dir=os.path.join(args.videos_root, v, f"s{sd}"),
                    name_prefix=f"{safe_env}_final_policy_seed{sd}",
                    seed=args.video_seed or sd,
                    max_steps=args.video_max_steps
                )

            if args.traj_grid:
                grid_dir = os.path.join(args.videos_root, v, f"s{sd}", "grids")
                os.makedirs(grid_dir, exist_ok=True)
                out_path = os.path.join(
                    grid_dir, f"{safe_env}_grid_{args.traj_grid_rows}x{args.traj_grid_cols}.mp4"
                )
                seed0 = args.traj_grid_seed0 if args.traj_grid_seed0 is not None else (sd * 1000 + 777)
                record_trajectories_grid(
                    env_id=args.env,
                    Q=Q,
                    out_path=out_path,
                    rows=args.traj_grid_rows,
                    cols=args.traj_grid_cols,
                    seed0=seed0,
                    max_steps=args.traj_grid_max_steps,
                    fps=args.traj_grid_fps
                )

def task_plot(args, variants, seeds):
    for v in variants:
        # returns
        for p in sorted(glob.glob(os.path.join(args.log_root, v, "s*", f"{v}_returns.csv"))):
            try: y = np.loadtxt(p, delimiter=",", dtype=float).reshape(-1)
            except Exception: continue
            sd = os.path.basename(os.path.dirname(p))
            out = os.path.join(args.plots_root, v, f"learning_curve_{sd}.png")
            save_curve(y, out, f"{v.capitalize()} Q-learning ({args.env}, {sd})", args.smooth, ylabel="Return")
            print(f"[saved] {out}")
        # steps
        for p in sorted(glob.glob(os.path.join(args.log_root, v, "s*", f"{v}_steps.csv"))):
            try: y = np.loadtxt(p, delimiter=",", dtype=float).reshape(-1)
            except Exception: continue
            sd = os.path.basename(os.path.dirname(p))
            out = os.path.join(args.plots_root, v, f"steps_curve_{sd}.png")
            save_curve(y, out, f"Steps to finish ({args.env}, {v}, {sd})", args.smooth, ylabel="Steps")
            print(f"[saved] {out}")
        # time
        for p in sorted(glob.glob(os.path.join(args.log_root, v, "s*", f"{v}_time.csv"))):
            try: y = np.loadtxt(p, delimiter=",", dtype=float).reshape(-1)
            except Exception: continue
            sd = os.path.basename(os.path.dirname(p))
            out = os.path.join(args.plots_root, v, f"time_curve_{sd}.png")
            save_curve(y, out, f"Time to finish ({args.env}, {v}, {sd})", args.smooth, ylabel="Time (Σ dt)")
            print(f"[saved] {out}")

def task_compare(args, variants, seeds):
    if "standard" not in variants or "harmonic" not in variants:
        variants = ["standard", "harmonic"]
    tag = args.tag or args.env
    # returns
    std_ret = load_csvs(args.log_root, "standard", seeds, metric="returns")
    har_ret  = load_csvs(args.log_root, "harmonic", seeds, metric="returns")
    if std_ret and har_ret:
        T = min(min(map(len,std_ret)), min(map(len,har_ret)))
        std_ret = np.stack([a[:T] for a in std_ret]); har_ret = np.stack([a[:T] for a in har_ret])
        out = os.path.join("plots","compare", f"{tag.replace('/','_')}_mean_se_returns.png")
        mean_se_plot(std_ret, har_ret, smooth=args.smooth, title=f"Mean ± SE Returns ({tag})", out_path=out, ylabel="Return")
        print(f"[saved] {out}")
    else:
        print("[compare] skipping returns (missing logs)")
    # steps
    std_stp = load_csvs(args.log_root, "standard", seeds, metric="steps")
    har_stp = load_csvs(args.log_root, "harmonic", seeds, metric="steps")
    if std_stp and har_stp:
        T = min(min(map(len,std_stp)), min(map(len,har_stp)))
        std_stp = np.stack([a[:T] for a in std_stp]); har_stp = np.stack([a[:T] for a in har_stp])
        out = os.path.join("plots","compare", f"{tag.replace('/','_')}_mean_se_steps.png")
        mean_se_plot(std_stp, har_stp, smooth=args.smooth, title=f"Mean ± SE Steps ({tag})", out_path=out, ylabel="Steps")
        print(f"[saved] {out}")
    else:
        print("[compare] skipping steps (missing logs)")
    # time
    std_time = load_csvs(args.log_root, "standard", seeds, metric="time")
    har_time = load_csvs(args.log_root, "harmonic", seeds, metric="time")
    if std_time and har_time:
        T = min(min(map(len,std_time)), min(map(len,har_time)))
        std_time = np.stack([a[:T] for a in std_time]); har_time = np.stack([a[:T] for a in har_time])
        out = os.path.join("plots","compare", f"{tag.replace('/','_')}_mean_se_time.png")
        mean_se_plot(std_time, har_time, smooth=args.smooth, title=f"Mean ± SE Time ({tag})", out_path=out, ylabel="Time (Σ dt)")
        print(f"[saved] {out}")
    else:
        print("[compare] skipping time (missing logs)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="MultiStepGrid-v0",
                    help="Env id (MultiStepGrid-v0, VelocityGrid-v0, CliffWalking-v0, FrozenLake-v1, ...)")
    ap.add_argument("--variants", choices=["standard","harmonic","both"], default="both")
    ap.add_argument("--seeds", default="0-4")
    # training
    ap.add_argument("--episodes", type=int, default=8000)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.05)
    ap.add_argument("--eps_decay_episodes", type=int, default=7600)
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
    ap.add_argument("--traj_grid", action="store_true", help="Save an R×C grid of sampled greedy rollouts")
    ap.add_argument("--traj_grid_rows", type=int, default=3)
    ap.add_argument("--traj_grid_cols", type=int, default=3)
    ap.add_argument("--traj_grid_fps", type=int, default=8)
    ap.add_argument("--traj_grid_max_steps", type=int, default=300)
    ap.add_argument("--traj_grid_seed0", type=int, default=None, help="Base seed for grid sampling (default uses seed*1000+777)")
    # plots/compare
    ap.add_argument("--smooth", type=int, default=151)
    ap.add_argument("--tag", default="")
    # task
    ap.add_argument("--task", choices=["train","plot","compare","all"], default="all")
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    variants = ["standard","harmonic"] if args.variants=="both" else [args.variants]

    if args.task in ("train","all"): task_train(args, variants, seeds)
    if args.task in ("plot","all"):  task_plot(args, variants, seeds)
    if args.task in ("compare","all"): task_compare(args, variants, seeds)

if __name__ == "__main__":
    main()
