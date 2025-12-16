#!/usr/bin/env python3
"""
scripts/plot_results.py

Create publication-style figures:
  - One 4-panel figure for Mode A (horizon sweep)
  - One 4-panel figure for Mode B (sample-size sweep)

Inputs:
  results/tables/ResultA.csv
  results/tables/ResultB.csv

Outputs (default):
  results/figures/fig_modeA_4panel.png
  results/figures/fig_modeB_4panel.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=str, default="results")
    p.add_argument("--no-band", action="store_true", help="Disable 95% CI bands")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--fmt", choices=["pdf", "png"], default="png")
    return p.parse_args()


def agg_ci(df: pd.DataFrame, by: str, y: str) -> pd.DataFrame:
    g = df.groupby(by)[y].agg(["mean", "std", "count"]).reset_index()
    g["se"] = g["std"] / np.sqrt(np.maximum(g["count"].to_numpy(), 1))
    g["ci95"] = 1.96 * g["se"]
    return g


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8),
    )


def plot_mode_a(dfA: pd.DataFrame, out_fig: Path, no_band: bool, dpi: int, fmt: str) -> None:
    df = dfA.copy()
    df["episode_len"] = df["episode_len"].astype(int)

    g_err = agg_ci(df, "episode_len", "abs_error")
    g_sec = agg_ci(df, "episode_len", "fit_sec")

    x = g_err["episode_len"].to_numpy()
    err_m = g_err["mean"].to_numpy()
    err_ci = g_err["ci95"].to_numpy()
    sec_m = g_sec["mean"].to_numpy()
    sec_ci = g_sec["ci95"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)

    # (a) abs_error vs T
    ax = axes[0, 0]
    ax.plot(x, err_m, marker="o")
    if not no_band:
        ax.fill_between(x, err_m - err_ci, err_m + err_ci, alpha=0.2)
    ax.set_xlabel("Horizon T")
    ax.set_ylabel("Absolute error")
    ax.grid(True, which="both", linestyle=":")
    panel_label(ax, "(a)")

    # (b) fit_sec vs T
    ax = axes[0, 1]
    ax.plot(x, sec_m, marker="o")
    if not no_band:
        ax.fill_between(x, sec_m - sec_ci, sec_m + sec_ci, alpha=0.2)
    ax.set_xlabel("Horizon T")
    ax.set_ylabel("Fit time (sec)")
    ax.grid(True, which="both", linestyle=":")
    panel_label(ax, "(b)")

    # (c) abs_error vs T (log y)
    ax = axes[1, 0]
    ax.plot(x, err_m, marker="o")
    if not no_band:
        ax.fill_between(x, np.maximum(err_m - err_ci, 1e-12), err_m + err_ci, alpha=0.2)
    ax.set_yscale("log")
    ax.set_xlabel("Horizon T")
    ax.set_ylabel("Absolute error (log)")
    ax.grid(True, which="both", linestyle=":")
    panel_label(ax, "(c)")

    # (d) fit_sec vs T (log y)
    ax = axes[1, 1]
    ax.plot(x, sec_m, marker="o")
    if not no_band:
        ax.fill_between(x, np.maximum(sec_m - sec_ci, 1e-12), sec_m + sec_ci, alpha=0.2)
    ax.set_yscale("log")
    ax.set_xlabel("Horizon T")
    ax.set_ylabel("Fit time (sec, log)")
    ax.grid(True, which="both", linestyle=":")
    panel_label(ax, "(d)")

    fig.suptitle("Mode A: Horizon sweep", fontsize=14, fontweight="bold")
    out_path = out_fig / f"fig_modeA_4panel.{fmt}"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def agg_mode_b(dfB: pd.DataFrame) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    df = dfB.copy()
    df["episode_len"] = df["episode_len"].astype(int)
    df["samp_size"] = df["samp_size"].astype(int)

    out: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for T in sorted(df["episode_len"].unique()):
        sub = df[df["episode_len"] == T]
        g_err = sub.groupby("samp_size")["abs_error"].agg(["mean", "std", "count"]).reset_index()
        g_sec = sub.groupby("samp_size")["fit_sec"].agg(["mean", "std", "count"]).reset_index()

        n = g_err["samp_size"].to_numpy()
        err_m = g_err["mean"].to_numpy()
        err_ci = 1.96 * (g_err["std"].to_numpy() / np.sqrt(np.maximum(g_err["count"].to_numpy(), 1)))
        sec_m = g_sec["mean"].to_numpy()
        sec_ci = 1.96 * (g_sec["std"].to_numpy() / np.sqrt(np.maximum(g_sec["count"].to_numpy(), 1)))

        if not np.array_equal(n, g_sec["samp_size"].to_numpy()):
            tmp = pd.merge(
                g_err,
                g_sec,
                on="samp_size",
                suffixes=("_err", "_sec"),
                how="inner",
            )
            n = tmp["samp_size"].to_numpy()
            err_m = tmp["mean_err"].to_numpy()
            err_ci = 1.96 * (tmp["std_err"].to_numpy() / np.sqrt(np.maximum(tmp["count_err"].to_numpy(), 1)))
            sec_m = tmp["mean_sec"].to_numpy()
            sec_ci = 1.96 * (tmp["std_sec"].to_numpy() / np.sqrt(np.maximum(tmp["count_sec"].to_numpy(), 1)))

        out[int(T)] = (n, err_m, err_ci, sec_m, sec_ci)

    return out


def plot_mode_b(dfB: pd.DataFrame, out_fig: Path, no_band: bool, dpi: int, fmt: str) -> None:
    data = agg_mode_b(dfB)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)

    # (a) abs_error vs n (linear)
    ax = axes[0, 0]
    for T, (n, err_m, err_ci, _, _) in data.items():
        ax.plot(n, err_m, marker="o", label=f"T={T}")
        if not no_band:
            ax.fill_between(n, err_m - err_ci, err_m + err_ci, alpha=0.15)
    ax.set_xlabel("Sample size n")
    ax.set_ylabel("Absolute error")
    ax.grid(True, which="both", linestyle=":")
    ax.legend(frameon=True)
    panel_label(ax, "(a)")

    # (b) fit_sec vs n (linear)
    ax = axes[0, 1]
    for T, (n, _, _, sec_m, sec_ci) in data.items():
        ax.plot(n, sec_m, marker="o", label=f"T={T}")
        if not no_band:
            ax.fill_between(n, sec_m - sec_ci, sec_m + sec_ci, alpha=0.15)
    ax.set_xlabel("Sample size n")
    ax.set_ylabel("Fit time (sec)")
    ax.grid(True, which="both", linestyle=":")
    ax.legend(frameon=True)
    panel_label(ax, "(b)")

    # (c) abs_error vs n (log-log)
    ax = axes[1, 0]
    for T, (n, err_m, err_ci, _, _) in data.items():
        ax.plot(n, err_m, marker="o", label=f"T={T}")
        if not no_band:
            ax.fill_between(n, np.maximum(err_m - err_ci, 1e-12), err_m + err_ci, alpha=0.15)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sample size n (log)")
    ax.set_ylabel("Absolute error (log)")
    ax.grid(True, which="both", linestyle=":")
    ax.legend(frameon=True)
    panel_label(ax, "(c)")

    # (d) fit_sec vs n (log-log)
    ax = axes[1, 1]
    for T, (n, _, _, sec_m, sec_ci) in data.items():
        ax.plot(n, sec_m, marker="o", label=f"T={T}")
        if not no_band:
            ax.fill_between(n, np.maximum(sec_m - sec_ci, 1e-12), sec_m + sec_ci, alpha=0.15)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sample size n (log)")
    ax.set_ylabel("Fit time (sec, log)")
    ax.grid(True, which="both", linestyle=":")
    ax.legend(frameon=True)
    panel_label(ax, "(d)")

    fig.suptitle("Mode B: Sample-size sweep", fontsize=14, fontweight="bold")
    out_path = out_fig / f"fig_modeB_4panel.{fmt}"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    tables = out_root / "tables"
    figs = out_root / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    pathA = tables / "ResultA.csv"
    pathB = tables / "ResultB.csv"

    if pathA.exists():
        dfA = pd.read_csv(pathA)
        plot_mode_a(dfA, figs, args.no_band, args.dpi, args.fmt)

    if pathB.exists():
        dfB = pd.read_csv(pathB)
        plot_mode_b(dfB, figs, args.no_band, args.dpi, args.fmt)

    print(f"[done] figures in {figs}")


if __name__ == "__main__":
    main()
