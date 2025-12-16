#!/usr/bin/env python3
"""
scripts/demo_sweep.py

Self-contained demo workflow for STATS 607:
- Runs a small Mode A+B sweep on CPU with parallelization over seeds (POMDP_N_WORKERS)
- Writes tables to results/tables/
- Generates 4-panel figures to results/figures/

This script is designed to finish quickly on most machines (demo-scale defaults).
You can increase the parameters for paper-scale runs.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["A", "B", "AB"], default="AB")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--epsilon", type=float, default=0.2)

    # Demo-scale defaults (adjust upward if desired)
    p.add_argument("--mc-episodes", type=int, default=30000)
    p.add_argument("--eval-initial", type=int, default=10000)
    p.add_argument("--n-sims", type=int, default=30)

    p.add_argument("--out-root", type=str, default="results")

    # Parallelization (CPU only)
    p.add_argument("--n-workers", type=int, default=4, help="Used when device=cpu")

    # Noise control
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print("[demo] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_root = repo_root / args.out_root
    (out_root / "tables").mkdir(parents=True, exist_ok=True)
    (out_root / "figures").mkdir(parents=True, exist_ok=True)

    # Enable CPU parallelization inside each job
    if args.device.lower() == "cpu":
        os.environ["POMDP_N_WORKERS"] = str(args.n_workers)

    run_grid = repo_root / "scripts" / "run_grid.py"
    plotter = repo_root / "scripts" / "plot_results.py"

    if not run_grid.exists():
        raise FileNotFoundError(f"Missing: {run_grid}")
    if not plotter.exists():
        raise FileNotFoundError(f"Missing: {plotter}")

    sweep_cmd = [
        sys.executable,
        str(run_grid),
        "--mode", args.mode,
        "--device", args.device,
        "--epsilon", str(args.epsilon),
        "--mc-episodes", str(args.mc_episodes),
        "--eval-initial", str(args.eval_initial),
        "--n-sims", str(args.n_sims),
        "--out-root", str(out_root),
    ]
    if args.quiet:
        sweep_cmd.append("--quiet")

    run(sweep_cmd)

    plot_cmd = [
        sys.executable,
        str(plotter),
        "--out-root", str(out_root),
        "--fmt", "png",
        "--dpi", "300",
    ]
    run(plot_cmd)

    print("[demo] Done.")
    print("[demo] Tables:", out_root / "tables")
    print("[demo] Figures:", out_root / "figures")


if __name__ == "__main__":
    main()
