#!/usr/bin/env python3
# Video helpers:
# 1) record_video(...) — single-episode video via rgb_array (skips if unsupported)
# 2) record_trajectories_grid(...) — N trajectories tiled into an R×C grid (with rgb/ANSI fallbacks)

from __future__ import annotations
import os
from typing import List, Tuple
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import imageio.v2 as imageio

# ------------------------- simple single-episode recorder -------------------------

def record_video(env_id: str, Q: np.ndarray, out_dir: str, name_prefix: str, seed: int, max_steps: int = 500):
    """Record one greedy rollout using Gymnasium's RecordVideo (rgb_array mode)."""
    try:
        os.makedirs(out_dir, exist_ok=True)
        env = gym.make(env_id, render_mode="rgb_array")
        env = RecordVideo(env, video_folder=out_dir, episode_trigger=lambda ep: ep == 0, name_prefix=name_prefix)
        s, _ = env.reset(seed=seed)
        for _ in range(max_steps):
            a = int(np.argmax(Q[s]))
            s, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        env.close()
        print(f"[video] saved under: {out_dir}")
    except Exception as e:
        print(f"[video] skipped (no rgb render?): {e}")

# ------------------------- grid of trajectories (3x3, etc.) -------------------------

try:
    from PIL import Image, ImageDraw, ImageFont  # for ANSI fallback
    _PIL_OK = True
except Exception:
    _PIL_OK = False

def _ansi_to_image(txt: str, pad: int = 8) -> np.ndarray:
    """Render ANSI text to an RGB image for video fallback."""
    if not _PIL_OK:
        # As a last resort, make a tiny white tile
        return np.ones((64, 64, 3), dtype=np.uint8) * 255
    font = ImageFont.load_default()
    lines = (txt or "").splitlines() or [""]
    # estimate char size
    x0, y0, x1, y1 = font.getbbox("M")
    ch_w, ch_h = (x1 - x0), (y1 - y0)
    max_w = max(len(l) for l in lines)
    W = pad * 2 + max(1, max_w) * ch_w
    H = pad * 2 + max(1, len(lines)) * ch_h
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    y = pad
    for line in lines:
        draw.text((pad, y), line, font=font, fill=(0, 0, 0))
        y += ch_h
    return np.asarray(img, dtype=np.uint8)

def _rollout_frames(env_id: str, Q: np.ndarray, seed: int, max_steps: int) -> Tuple[List[np.ndarray], str]:
    """
    Run one greedy rollout and return list of frames and mode ('rgb' or 'ansi').
    Tries rgb_array first; falls back to ansi text → image.
    """
    # 1) rgb_array path
    try:
        env = gym.make(env_id, render_mode="rgb_array")
        frames: List[np.ndarray] = []
        s, _ = env.reset(seed=seed)
        frame = env.render()
        if isinstance(frame, np.ndarray):
            frames.append(frame)
        for _ in range(max_steps):
            a = int(np.argmax(Q[s]))
            s, _, term, trunc, _ = env.step(a)
            frame = env.render()
            if isinstance(frame, np.ndarray):
                frames.append(frame)
            if term or trunc:
                break
        env.close()
        if frames:
            return frames, "rgb"
    except Exception:
        pass

    # 2) ansi path
    try:
        env = gym.make(env_id, render_mode="ansi")
        frames = []
        s, _ = env.reset(seed=seed)
        txt = env.render()
        frames.append(_ansi_to_image(str(txt)))
        for _ in range(max_steps):
            a = int(np.argmax(Q[s]))
            s, _, term, trunc, _ = env.step(a)
            txt = env.render()
            frames.append(_ansi_to_image(str(txt)))
            if term or trunc:
                break
        env.close()
        return frames, "ansi"
    except Exception as e:
        print(f"[grid] rollout failed for seed={seed}: {e}")
        # make a placeholder tile so grid still works
        return [np.ones((64, 64, 3), dtype=np.uint8) * 200], "none"

def _pad_to(frames: List[np.ndarray], length: int) -> List[np.ndarray]:
    """Pad a trajectory with its last frame to reach a uniform length."""
    if not frames:
        # create a dummy
        frames = [np.ones((64, 64, 3), dtype=np.uint8) * 200]
    last = frames[-1]
    while len(frames) < length:
        frames.append(last)
    return frames

def _tile_frame_grid(frame_rows: List[List[np.ndarray]]) -> np.ndarray:
    """Tile frames in a grid: frame_rows[r][c] must be same H×W×3 per row/col."""
    # normalize sizes per row
    row_imgs = []
    for row in frame_rows:
        # height match by padding/cropping if needed (rare)
        Hs = [img.shape[0] for img in row]
        Ws = [img.shape[1] for img in row]
        H = max(Hs); W = max(Ws)
        fixed = []
        for img in row:
            h, w = img.shape[:2]
            # pad to (H,W) with light gray
            pad = np.ones((H, W, 3), dtype=np.uint8) * 240
            pad[:h, :w] = img
            fixed.append(pad)
        row_imgs.append(np.concatenate(fixed, axis=1))
    return np.concatenate(row_imgs, axis=0)

def record_trajectories_grid(
    env_id: str,
    Q: np.ndarray,
    out_path: str,
    rows: int = 3,
    cols: int = 3,
    seed0: int = 7777,
    max_steps: int = 300,
    fps: int = 8,
):
    """
    Make a tiled video of R×C greedy rollouts using seeds seed0 .. seed0+R*C-1.
    Saves MP4 (or GIF if extension is .gif).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    N = rows * cols
    seeds = [seed0 + i for i in range(N)]
    trajs: List[List[np.ndarray]] = []
    sizes = []
    for sd in seeds:
        frames, mode = _rollout_frames(env_id, Q, sd, max_steps)
        trajs.append(frames)
        if frames and isinstance(frames[0], np.ndarray):
            sizes.append(frames[0].shape[:2])

    # unify length
    T = max(len(fr) for fr in trajs)
    trajs = [_pad_to(fr, T) for fr in trajs]

    # build tiled frames
    tiled_frames: List[np.ndarray] = []
    for t in range(T):
        # gather R×C frames at time t
        grid_rows: List[List[np.ndarray]] = []
        for r in range(rows):
            row_imgs = [trajs[r * cols + c][t] for c in range(cols)]
            grid_rows.append(row_imgs)
        tiled = _tile_frame_grid(grid_rows)
        tiled_frames.append(tiled.astype(np.uint8))

    # write video
    ext = os.path.splitext(out_path)[1].lower()
    if ext == ".gif":
        imageio.mimsave(out_path, tiled_frames, fps=fps, loop=0)
    else:
        imageio.mimsave(out_path, tiled_frames, fps=fps)  # MP4 via imageio-ffmpeg
    print(f"[grid] saved {rows}x{cols} trajectories → {out_path}")
