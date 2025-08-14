#!/usr/bin/env python3
# Minimal all-in-one: train tabular Q-learning (standard|harmonic), optional video,
# per-run plots, and a simple cross-seed comparison. ~150 lines.

import argparse, os, glob, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

# ---------- tiny utils ----------
def parse_seeds(s):
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

def ma(x, w):  # moving average (same length)
    if w<=1 or w>len(x): return x
    k = np.ones(w)/w
    return np.convolve(x, k, mode="same")

def save_curve(y, out_png, title, smooth):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    x = np.arange(1, len(y)+1)
    plt.figure(figsize=(8,4.5))
    plt.plot(x, y); plt.grid(True, ls="--", lw=0.5)
    plt.xlabel("Episode"); plt.ylabel("Return"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
    ys = ma(y, smooth)
    base,ext = os.path.splitext(out_png)
    plt.figure(figsize=(8,4.5)); plt.plot(x, ys)
    plt.grid(True, ls="--", lw=0.5); plt.xlabel("Episode"); plt.ylabel("Return")
    plt.title(f"{title} (smoothed w={smooth})")
    plt.tight_layout(); plt.savefig(f"{base}_smoothed_w{smooth}{ext}", dpi=150); plt.close()

# ---------- q-learning ----------
def eps_greedy(Q, s, eps, rng):
    return rng.integers(Q.shape[1]) if rng.random()<eps else int(np.argmax(Q[s]))

def h_update(q, y, a, eps_div=1e-8, shift=1e-6):  # harmonic step between q and y with weight a
    sft = min(q, y, 0.0) - shift
    q_, y_ = q - sft, y - sft
    inv_new = (1-a)/max(q_, eps_div) + a/max(y_, eps_div)
    return sft + 1.0/max(inv_new, eps_div)

def train_one(env_id, variant, episodes, alpha, gamma, eps0, eps1, eps_decay_episodes, max_steps, seed):
    env = gym.make(env_id)
    nS, nA = env.observation_space.n, env.action_space.n
    Q = np.zeros((nS, nA), float)
    rng = np.random.default_rng(seed)
    rets = []
    for ep in range(episodes):
        s,_ = env.reset(seed=seed+ep); ret = 0.0
        t = min(1.0, ep/max(1,eps_decay_episodes))
        eps = eps0 + (eps1 - eps0)*t
        for _ in range(max_steps):
            a = eps_greedy(Q, s, eps, rng)
            s2, r, term, trunc, _ = env.step(a); done = term or trunc
            y = r if done else r + gamma*np.max(Q[s2])
            Q[s,a] = (1-alpha)*Q[s,a] + alpha*y if variant=="standard" else h_update(Q[s,a], y, alpha)
            ret += r; s = s2
            if done: break
        rets.append(ret)
        if (ep+1)%100==0: print(f"[{variant}] ep {ep+1}/{episodes} avg={np.mean(rets):.3f}")
    env.close()
    return Q, np.array(rets, float)

# ---------- video (simple, RecordVideo only; skip if unsupported) ----------
def record_video(env_id, Q, out_dir, name_prefix, seed, max_steps):
    try:
        os.makedirs(out_dir, exist_ok=True)
        env = gym.make(env_id, render_mode="rgb_array")
        env = RecordVideo(env, video_folder=out_dir, episode_trigger=lambda e: e==0, name_prefix=name_prefix)
        s,_ = env.reset(seed=seed)
        for _ in range(max_steps):
            a = int(np.argmax(Q[s])); s,_,term,trunc,_ = env.step(a)
            if term or trunc: break
        env.close()
        print(f"[video] saved under: {out_dir}")
    except Exception as e:
        print(f"[video] skipped (env lacks rgb render?): {e}")

# ---------- tasks ----------
def task_train(args, variants, seeds):
    for v in variants:
        for sd in seeds:
            print(f"\n== train {v} seed={sd} ==")
            Q, rets = train_one(args.env, v, args.episodes, args.alpha, args.gamma,
                                args.eps_start, args.eps_end, args.eps_decay_episodes,
                                args.max_steps_per_ep, sd)
            out_dir = os.path.join(args.log_root, v, f"s{sd}"); os.makedirs(out_dir, exist_ok=True)
            np.savetxt(os.path.join(out_dir, f"{v}_returns.csv"), rets, delimiter=",")
            if args.save_q: np.save(os.path.join(out_dir, f"Q_{v}_seed{sd}.npy"), Q)
            if args.record_video:
                safe_env = args.env.replace("/","_").replace(":","_").replace("-","_")
                record_video(args.env, Q, os.path.join(args.videos_root, v, f"s{sd}"),
                             f"{safe_env}_final_policy_seed{sd}", args.video_seed or sd, args.video_max_steps)

def task_plot(args, variants, seeds):
    for v in variants:
        for sd in seeds:
            p = os.path.join(args.log_root, v, f"s{sd}", f"{v}_returns.csv")
            if os.path.isfile(p):
                y = np.loadtxt(p, delimiter=",", dtype=float).reshape(-1)
                out = os.path.join(args.plots_root, v, f"learning_curve_s{sd}.png")
                save_curve(y, out, f"{v.capitalize()} Q-learning (s{sd})", args.smooth)

def task_compare(args, variants, seeds):
    if "standard" not in variants or "harmonic" not in variants:
        variants = ["standard", "harmonic"]
    def load_many(v):
        arrs = []
        for p in sorted(glob.glob(os.path.join(args.log_root, v, "s*", f"{v}_returns.csv"))):
            try: arrs.append(np.loadtxt(p, delimiter=",", dtype=float).reshape(-1))
            except: pass
        return arrs
    std, har = load_many("standard"), load_many("harmonic")
    if not std or not har: print("[compare] need logs for both variants"); return
    T = min(min(map(len,std)), min(map(len,har)))
    std = np.stack([a[:T] for a in std]); har = np.stack([a[:T] for a in har])
    x = np.arange(1, T+1)
    std_m, har_m = std.mean(0), har.mean(0)
    std_se, har_se = std.std(0)/np.sqrt(len(std)), har.std(0)/np.sqrt(len(har))
    if args.smooth>1:
        std_m, std_se = ma(std_m, args.smooth), ma(std_se, args.smooth)
        har_m, har_se = ma(har_m, args.smooth), ma(har_se, args.smooth)
    os.makedirs("plots/compare", exist_ok=True)
    plt.figure(figsize=(9,5))
    plt.plot(x, std_m, label="Standard"); plt.fill_between(x, std_m-std_se, std_m+std_se, alpha=0.2)
    plt.plot(x, har_m, label="Harmonic"); plt.fill_between(x, har_m-har_se, har_m+har_se, alpha=0.2)
    plt.xlabel("Episode"); plt.ylabel("Return"); plt.title(f"Mean ± SE ({args.tag or args.env})")
    plt.grid(True, ls="--", lw=0.5); plt.legend()
    out = f"plots/compare/{(args.tag or args.env).replace('/','_')}_mean_se.png"
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close(); print(f"[saved] {out}")

# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="CliffWalking-v0")
    ap.add_argument("--variants", choices=["standard","harmonic","both"], default="both")
    ap.add_argument("--seeds", default="0")  # "0-4" or "0,1,2"
    # training
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.05)
    ap.add_argument("--eps_decay_episodes", type=int, default=2500)
    ap.add_argument("--max_steps_per_ep", type=int, default=500)
    # io
    ap.add_argument("--log_root", default="logs")
    ap.add_argument("--plots_root", default="plots")
    ap.add_argument("--videos_root", default="videos")
    ap.add_argument("--save_q", action="store_true")
    # video
    ap.add_argument("--record_video", action="store_true")
    ap.add_argument("--video_seed", type=int, default=None)
    ap.add_argument("--video_max_steps", type=int, default=500)
    # plots/compare
    ap.add_argument("--smooth", type=int, default=101)
    ap.add_argument("--tag", default="")
    # task
    ap.add_argument("--task", choices=["train","plot","compare","all"], default="all")
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    variants = ["standard","harmonic"] if args.variants=="both" else [args.variants]

    if args.task in ("train","all"): task_train(args, variants, seeds)
    if args.task in ("plot","all"):  task_plot(args, variants, seeds)
    if args.task in ("compare","all"): task_compare(args, variants, seeds)
