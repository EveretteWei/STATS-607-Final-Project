"""
ContSimuOffPolicy.py

Entry point to reproduce the NeurIPS 2022 simulation for confounded POMDP OPE.

Key properties:
  - Sequential NPIV recursion is preserved (backward in t).
  - Vectorized data collection (preallocated arrays) via utils.batch_data_collector.
  - More stable RKHS/linear algebra (handled in rkhs_torch.py).
  - Optional caching of Monte Carlo "true value" estimates.
  - Optional CPU parallelization over seeds (GPU stays single-process by default).

CLI usage:
  python ContSimuOffPolicy.py --T 2 --n 64 --epsilon 0.2 --device cuda:0 \
      --mc-episodes 200 --eval-initial 200 --n-sims 2 --out results/tables/demo.csv

Legacy positional usage (kept for compatibility):
  python ContSimuOffPolicy.py T n epsilon device mc_episodes out_csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor
from scipy.special import expit as scipy_expit
from torch.special import expit as torch_expit
from tqdm import tqdm

from envs import ContinuousEnv
from agents import Policy
from prox_fqe import fit_v0
from rkhs_torch import _to_tensor
from utils import MC_evaluator, batch_data_collector, sample_initial_states, set_seeds


# -------------------------- policy -------------------------- #

class ContPolicy(Policy):
    """
    Epsilon-greedy target policy used in the original repo.
    """

    def __init__(self, env: ContinuousEnv, eps: float, device: str):
        super().__init__(env=env)
        self.device = device
        self.eps = float(eps)

        # CPU versions (numpy) for eps_greedy
        self.kappa_0_np = float(self.env.kappa_0)
        self.kappa_a_np = float(self.env.kappa_a)
        self.kappa_s_np = np.asarray(self.env.kappa_s, dtype=float).reshape(2)
        self.t0tukappa0_np = float(self.env.t_0 + self.env.t_u * self.env.kappa_0)
        self.tstukappas_np = np.asarray(self.env.t_s + self.env.t_u * self.env.kappa_s, dtype=float).reshape(2)

        # Torch versions for prob_torch
        with torch.no_grad():
            self.kappa_0_t = torch.tensor(self.env.kappa_0)
            self.kappa_a_t = torch.tensor(self.env.kappa_a)
            self.kappa_s_t = torch.tensor(self.env.kappa_s)
            self.t0tukappa0_t = torch.tensor(self.env.t_0 + self.env.t_u * self.env.kappa_0)
            self.tstukappas_t = torch.tensor(self.env.t_s + self.env.t_u * self.env.kappa_s)
            self._to_device()

    @torch.no_grad()
    def _to_device(self) -> None:
        self.kappa_0_t = _to_tensor(self.kappa_0_t, self.device)
        self.kappa_a_t = _to_tensor(self.kappa_a_t, self.device)
        self.kappa_s_t = _to_tensor(self.kappa_s_t, self.device)
        self.t0tukappa0_t = _to_tensor(self.t0tukappa0_t, self.device)
        self.tstukappas_t = _to_tensor(self.tstukappas_t, self.device)

    @torch.no_grad()
    def prob_torch(self, a: int, s: torch.Tensor) -> torch.Tensor:
        """
        Vectorized probability for action a in {0,1}.
        """
        # ind is in {0,1} as float
        ind = (
            self.kappa_0_t
            + s @ self.kappa_s_t
            + 1.0
            - 2.0 * self.kappa_a_t * torch_expit(self.t0tukappa0_t + s @ self.tstukappas_t)
            + s[:, 0]
            - 2.0 * s[:, 1]
            > 0
        ).to(dtype=s.dtype)

        if int(a) == 1:
            p = torch.abs(ind - self.eps).reshape(-1, 1)
        else:
            p = (1.0 - torch.abs(ind - self.eps)).reshape(-1, 1)
        return p

    def eps_greedy(self, obs: Dict[str, np.ndarray]) -> int:
        s = np.asarray(obs["S"], dtype=float).reshape(2)
        ind = int(
            self.kappa_0_np
            + s @ self.kappa_s_np
            + 1.0
            - 2.0 * self.kappa_a_np * scipy_expit(self.t0tukappa0_np + s @ self.tstukappas_np)
            + s[0]
            - 2.0 * s[1]
            > 0
        )
        if np.random.rand() < self.eps:
            return 1 - ind
        return ind


# -------------------------- config -------------------------- #

@dataclass(frozen=True)
class SimConfig:
    T: int
    n: int
    epsilon: float
    device: str
    mc_episodes: int
    eval_initial: int
    n_sims: int
    seed0: int
    out_csv: str
    cache_dir: str
    quiet: bool

    # RKHS / CV options
    gamma_f: str
    n_gamma_hs: int
    n_alphas: int
    cv: int

    # dtype for fitting
    dtype: str  # "float32" or "float64"


def _dtype_from_str(s: str) -> torch.dtype:
    s = s.lower().strip()
    if s in ("float32", "fp32"):
        return torch.float32
    if s in ("float64", "fp64"):
        return torch.float64
    raise ValueError(f"Unsupported dtype: {s}")


def _true_value_cache_path(cfg: SimConfig) -> str:
    key = f"T{cfg.T}_eps{cfg.epsilon:.6f}_mc{cfg.mc_episodes}"
    return os.path.join(cfg.cache_dir, f"true_{key}.json")


def _load_true_value(cfg: SimConfig) -> Optional[float]:
    os.makedirs(cfg.cache_dir, exist_ok=True)
    path = _true_value_cache_path(cfg)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return float(obj["true_value"])
    except Exception:
        return None


def _save_true_value(cfg: SimConfig, v: float) -> None:
    os.makedirs(cfg.cache_dir, exist_ok=True)
    path = _true_value_cache_path(cfg)
    tmp = {"true_value": float(v), "T": cfg.T, "epsilon": cfg.epsilon, "mc_episodes": cfg.mc_episodes}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tmp, f)


# -------------------------- workers -------------------------- #

def _make_env_params(T: int, offline: bool) -> Dict[str, Any]:
    return {
        "episode_len": T,
        "offline": offline,
        "alpha_0": 0,
        "alpha_a": 0.5,
        "alpha_s": [0.5, 0.5],
        "mu_0": 0,
        "mu_a": -0.25,
        "mu_s": [0.5, 0.5],
        "kappa_0": 0,
        "kappa_a": -0.5,
        "kappa_s": [0.5, 0.5],
        "t_0": 0,
        "t_u": 1,
        "t_s": [-0.5, -0.5],
    }


def _run_one_seed(seed: int, cfg: SimConfig, reward_mean: float, pfqe_option: Dict[str, Any]) -> Dict[str, Any]:
    set_seeds(seed)

    env = ContinuousEnv(_make_env_params(cfg.T, offline=True))
    pi = ContPolicy(env=env, eps=cfg.epsilon, device=cfg.device)

    ds = batch_data_collector(env, {"episode_num": cfg.n}, policy=None, seed=seed, return_legacy=False)

    t0 = time.time()
    vpi = fit_v0(
        Episodes=ds,
        action_space=env.action_space,
        observation_space=env.observation_space,
        policy=pi,
        option=pfqe_option,
        device=cfg.device,
        dtype=_dtype_from_str(cfg.dtype),
    )
    fit_sec = time.time() - t0

    # Evaluate vpi on sampled initial states
    env.params["offline"] = False
    S0, W0 = sample_initial_states(env, n=cfg.eval_initial, seed=1234)
    W0_t = _to_tensor(W0, cfg.device).reshape(-1, 1)
    S0_t = _to_tensor(S0, cfg.device)

    with torch.no_grad():
        est = float(vpi(W0_t, S0_t).mean().item())

    abs_error = float(abs(reward_mean - est))
    return {
        "seed": int(seed),
        "abs_error": abs_error,
        "value_hat": float(est),
        "true_value": float(reward_mean),
        "fit_sec": float(fit_sec),
    }


def _run_one_seed_job(args):
    seed, cfg_dict, reward_mean, pfqe_option = args
    cfg = SimConfig(**cfg_dict)
    return _run_one_seed(int(seed), cfg, float(reward_mean), pfqe_option)


# -------------------------- CLI -------------------------- #

def parse_args(argv: Optional[List[str]] = None) -> SimConfig:
    import sys
    if argv is None:
        argv = sys.argv[1:]

    # Legacy positional args: T n epsilon device mc_episodes out_csv
    if len(argv) == 6 and not argv[0].startswith("-"):
        T = int(argv[0])
        n = int(argv[1])
        eps = float(argv[2])
        dev = argv[3]
        mc = int(argv[4])
        out = argv[5]
        return SimConfig(
            T=T,
            n=n,
            epsilon=eps,
            device=dev,
            mc_episodes=mc,
            eval_initial=50000,
            n_sims=100,
            seed0=0,
            out_csv=out,
            cache_dir=".cache_pomdp",
            quiet=False,
            gamma_f="auto",
            n_gamma_hs=20,
            n_alphas=30,
            cv=5,
            dtype="float32",
        )

    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--device", type=str, default="cuda:0")

    p.add_argument("--mc-episodes", type=int, default=50000)
    p.add_argument("--eval-initial", type=int, default=50000)
    p.add_argument("--n-sims", type=int, default=100)
    p.add_argument("--seed0", type=int, default=0)

    p.add_argument("--out", type=str, required=True)
    p.add_argument("--cache-dir", type=str, default=".cache_pomdp")
    p.add_argument("--quiet", action="store_true")

    # RKHS / CV knobs (kept simple)
    p.add_argument("--gamma-f", type=str, default="auto")
    p.add_argument("--n-gamma-hs", type=int, default=20)
    p.add_argument("--n-alphas", type=int, default=30)
    p.add_argument("--cv", type=int, default=5)

    p.add_argument("--dtype", type=str, choices=["float32", "float64"], default="float32")

    ns = p.parse_args(argv)
    return SimConfig(
        T=ns.T,
        n=ns.n,
        epsilon=ns.epsilon,
        device=ns.device,
        mc_episodes=ns.mc_episodes,
        eval_initial=ns.eval_initial,
        n_sims=ns.n_sims,
        seed0=ns.seed0,
        out_csv=ns.out,
        cache_dir=ns.cache_dir,
        quiet=bool(ns.quiet),
        gamma_f=str(ns.gamma_f),
        n_gamma_hs=int(ns.n_gamma_hs),
        n_alphas=int(ns.n_alphas),
        cv=int(ns.cv),
        dtype=str(ns.dtype),
    )


def run_config(cfg: SimConfig, *, n_workers: int = 1) -> List[Dict[str, Any]]:
    # Compute or load true value
    env_tv = ContinuousEnv(_make_env_params(cfg.T, offline=False))
    pi_tv = ContPolicy(env=env_tv, eps=cfg.epsilon, device=cfg.device)

    reward_mean = _load_true_value(cfg)
    if reward_mean is None:
        t0 = time.time()
        reward_mean = float(MC_evaluator(env_tv, policy=pi_tv, config={"episode_num": cfg.mc_episodes}, seed=0))
        _save_true_value(cfg, reward_mean)
        if not cfg.quiet:
            print(f"[true] computed in {time.time()-t0:.1f}s value={reward_mean:.6f}")
    else:
        reward_mean = float(reward_mean)
        if not cfg.quiet:
            print(f"[true] cache hit value={reward_mean:.6f}")

    pfqe_option = {
        "gamma_f": cfg.gamma_f,
        "n_gamma_hs": cfg.n_gamma_hs,
        "n_alphas": cfg.n_alphas,
        "cv": cfg.cv,
    }

    seeds = [cfg.seed0 + i for i in range(cfg.n_sims)]

    # GPU: keep single process by default
    if n_workers > 1 and str(cfg.device).startswith("cuda"):
        n_workers = 1

    rows: List[Dict[str, Any]] = []

    if n_workers == 1:
        it = seeds if cfg.quiet else tqdm(seeds, desc="seeds", unit="seed")
        for s in it:
            if not cfg.quiet:
                print(f"[seed] {s} T={cfg.T} n={cfg.n}")
            rows.append(_run_one_seed(int(s), cfg, reward_mean, pfqe_option))
    else:
        cfg_dict = cfg.__dict__.copy()
        jobs = [(int(s), cfg_dict, reward_mean, pfqe_option) for s in seeds]
        with ProcessPoolExecutor(max_workers=min(n_workers, len(seeds))) as ex:
            for out in ex.map(_run_one_seed_job, jobs):
                rows.append(out)

    return rows


def write_rows(cfg: SimConfig, rows: List[Dict[str, Any]]) -> None:
    out_path = cfg.out_csv
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    new_file = not os.path.exists(out_path)
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["episode_len", "samp_size", "epsilon", "seed", "abs_error", "value_hat", "true_value", "fit_sec"])
        for r in rows:
            w.writerow([cfg.T, cfg.n, cfg.epsilon, r["seed"], r["abs_error"], r["value_hat"], r["true_value"], r["fit_sec"]])


def main() -> None:
    cfg = parse_args()
    n_workers = int(os.environ.get("POMDP_N_WORKERS", "1"))
    rows = run_config(cfg, n_workers=n_workers)
    write_rows(cfg, rows)
    if not cfg.quiet:
        print(f"[saved] {cfg.out_csv} (+{len(rows)} rows)")


if __name__ == "__main__":
    main()
