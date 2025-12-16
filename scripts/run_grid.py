#!/usr/bin/env python3
"""
scripts/run_grid.py

Cross-platform grid runner that calls ContSimuOffPolicy.py via CLI.

Outputs:
  results/tables/ResultA.csv
  results/tables/ResultB.csv

This runner prints one line per job and shows a progress bar.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_dirs(out_root: Path) -> Tuple[Path, Path]:
    tables = out_root / "tables"
    figures = out_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return tables, figures


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["A", "B", "AB"], default="AB")
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--mc-episodes", type=int, default=50000)
    p.add_argument("--eval-initial", type=int, default=50000)
    p.add_argument("--n-sims", type=int, default=100)
    p.add_argument("--seed0", type=int, default=0)

    p.add_argument("--gamma-f", type=str, default="auto")
    p.add_argument("--n-gamma-hs", type=int, default=20)
    p.add_argument("--n-alphas", type=int, default=30)
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")

    p.add_argument("--out-root", type=str, default="results")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def _jobs_for_mode(mode: str, n_sims: int) -> Tuple[List[Tuple[int, int, int]], str]:
    jobs: List[Tuple[int, int, int]] = []
    if mode.upper() == "A":
        Ts = [1, 2, 4, 8, 16, 24, 32, 48, 64]
        n = 512
        for T in Ts:
            jobs.append((T, n, n_sims))
        return jobs, "ResultA.csv"
    if mode.upper() == "B":
        Ts = [2, 4, 6]
        ns = [256, 512, 1024, 2048, 4096]
        for T in Ts:
            for n in ns:
                jobs.append((T, n, n_sims))
        return jobs, "ResultB.csv"
    raise ValueError(f"Unknown mode: {mode}")


def _run_one_job(
    T: int,
    n: int,
    n_sims: int,
    args: argparse.Namespace,
    out_csv: Path,
) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "ContSimuOffPolicy.py"),
        "--T", str(T),
        "--n", str(n),
        "--epsilon", str(args.epsilon),
        "--device", str(args.device),
        "--mc-episodes", str(args.mc_episodes),
        "--eval-initial", str(args.eval_initial),
        "--n-sims", str(n_sims),
        "--seed0", str(args.seed0),
        "--out", str(out_csv),

        "--gamma-f", str(args.gamma_f),
        "--n-gamma-hs", str(args.n_gamma_hs),
        "--n-alphas", str(args.n_alphas),
        "--cv", str(args.cv),
        "--dtype", str(args.dtype),
    ]
    if args.quiet:
        cmd.append("--quiet")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Job failed (T={T}, n={n}).\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def main() -> None:
    args = _parse_args()
    out_root = Path(args.out_root)
    tables_dir, _ = _ensure_dirs(out_root)

    modes = ["A", "B"] if args.mode == "AB" else [args.mode]

    for mode in modes:
        jobs, out_name = _jobs_for_mode(mode, args.n_sims)
        out_path = tables_dir / out_name

        # Fresh file per mode
        if out_path.exists():
            out_path.unlink()

        t0 = time.time()
        for i, (T, n, n_sims) in enumerate(tqdm(jobs, desc=f"Mode {mode}", unit="job")):
            t_job = time.time()
            print(f"[job {i+1}/{len(jobs)}] T={T} n={n} sims={n_sims}")
            _run_one_job(T=T, n=n, n_sims=n_sims, args=args, out_csv=out_path)
            print(f"[job {i+1}/{len(jobs)}] done in {time.time()-t_job:.1f}s")

        print(f"[saved] {out_path} in {time.time()-t0:.1f}s")

    print(f"[done] tables in {tables_dir}")


if __name__ == "__main__":
    main()
