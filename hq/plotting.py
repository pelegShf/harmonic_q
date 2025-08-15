#!/usr/bin/env python3
# Headless plotting helpers:
# - runs_with_mean_plot: plot all seed runs (low opacity) + bold mean curve
# - mean_se_plot: mean ± SE comparison for two groups (used in "compare")

from __future__ import annotations
import os, numpy as np, matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window is None or window <= 1:
        return x.copy()
    w = int(min(window, len(x)))
    # pad with edge values to avoid zero-padding artifacts at the ends
    left = w // 2
    right = w - 1 - left
    xpad = np.pad(x.astype(float), (left, right), mode="edge")
    k = np.ones(w, dtype=float) / float(w)
    return np.convolve(xpad, k, mode="valid")


def runs_with_mean_plot(
    runs: list[np.ndarray],
    smooth: int,
    title: str,
    out_path: str,
    ylabel: str = "Return",
    alpha_runs: float = 0.25,
    lw_mean: float = 2.8,
):
    """
    Plot all seed runs with low opacity + bold mean curve (single image).
    Each run and the mean can be optionally smoothed with a moving-average window.
    """
    if not runs:
        return
    # Trim to common length
    T = min(len(r) for r in runs)
    R = np.stack([r[:T] for r in runs], axis=0)
    x = np.arange(1, T + 1)

    # Smooth per-run (for display)
    disp = [moving_average(r, smooth) if smooth and smooth > 1 else r for r in R]

    # Mean over raw, then optionally smooth
    mean_curve = R.mean(axis=0)
    if smooth and smooth > 1:
        mean_curve = moving_average(mean_curve, smooth)

    # Plot
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(9, 5))
    for y in disp:
        plt.plot(x, y, linewidth=1.0, alpha=alpha_runs)
    plt.plot(x, mean_curve, linewidth=lw_mean, label="Mean")
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    if smooth and smooth > 1:
        plt.title(f"{title} (smoothed w={smooth})")
    else:
        plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def mean_se_plot(
    std_runs: np.ndarray,
    harm_runs: np.ndarray,
    smooth: int,
    title: str,
    out_path: str,
    ylabel: str = "Return",
):
    """
    Comparison plot (mean ± SE), optionally smoothed; saves a single figure.
    Inputs are arrays shaped [n_seeds, T].
    """
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
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(f"{title} (smoothed w={smooth})" if smooth and smooth > 1 else title)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
