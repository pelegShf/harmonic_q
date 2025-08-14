#!/usr/bin/env python3
# Minimal plotting helpers (headless) — only save the *smoothed* plot.

from __future__ import annotations
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window is None or window <= 1 or window > len(x):
        return x.copy()
    k = np.ones(window, dtype=float) / float(window)
    return np.convolve(x, k, mode="same")


def save_curve(y: np.ndarray, out_path: str, title: str, smooth: int, ylabel: str = "Return"):
    """Save a single smoothed curve to out_path."""
    x = np.arange(1, len(y) + 1)
    ys = moving_average(y, smooth)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.plot(x, ys)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.xlabel("Episode"); plt.ylabel(ylabel)
    plt.title(f"{title} (smoothed w={smooth})" if smooth and smooth > 1 else title)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def mean_se_plot(std_runs: np.ndarray, harm_runs: np.ndarray, smooth: int, title: str, out_path: str, ylabel: str = "Return"):
    """Comparison plot (mean ± SE), optionally smoothed; saves a single figure."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    T = min(std_runs.shape[1], harm_runs.shape[1])
    std, har = std_runs[:, :T], harm_runs[:, :T]
    x = np.arange(1, T + 1)

    std_m, har_m = std.mean(0), har.mean(0)
    std_se = std.std(0, ddof=1) / max(1, np.sqrt(len(std)))
    har_se = har.std(0, ddof=1) / max(1, np.sqrt(len(har)))

    if smooth and smooth > 1:
        std_m = moving_average(std_m, smooth)
        har_m = moving_average(har_m, smooth)
        std_se = moving_average(std_se, smooth)
        har_se = moving_average(har_se, smooth)

    plt.figure(figsize=(9, 5))
    plt.plot(x, std_m, label="Standard")
    plt.fill_between(x, std_m - std_se, std_m + std_se, alpha=0.2)
    plt.plot(x, har_m, label="Harmonic")
    plt.fill_between(x, har_m - har_se, har_m + har_se, alpha=0.2)
    plt.xlabel("Episode"); plt.ylabel(ylabel)
    plt.title(f"{title} (smoothed w={smooth})" if smooth and smooth > 1 else title)
    plt.grid(True, linestyle="--", linewidth=0.5); plt.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
