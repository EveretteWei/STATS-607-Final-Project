"""
utils.py (optimized)

Utilities for data collection and evaluation.

Main changes:
  - episodes are collected into preallocated NumPy arrays (no nested dict lists)
  - conversion to torch uses one-shot torch.as_tensor (minimal Python overhead)
  - Gymnasium reset/step signatures supported
  - optional legacy format is still available
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def _reset_env(env, *, seed: Optional[int] = None):
    out = env.reset(seed=seed) if seed is not None else env.reset()
    # gymnasium: (obs, info)
    if isinstance(out, tuple) and len(out) == 2:
        return out[0]
    return out


def _step_env(env, action):
    out = env.step(action)
    if not isinstance(out, tuple):
        raise RuntimeError("env.step must return a tuple")
    # gymnasium: (obs, r, terminated, truncated, info)
    if len(out) == 5:
        obs, r, terminated, truncated, info = out
        done = bool(terminated or truncated)
        return obs, float(r), done
    # gym: (obs, r, done, info)
    if len(out) == 4:
        obs, r, done, info = out
        return obs, float(r), bool(done)
    raise RuntimeError(f"Unexpected env.step return length: {len(out)}")


def batch_data_collector(
    env,
    config: Dict[str, Any],
    policy=None,
    seed: int = 0,
    *,
    return_legacy: bool = False,
) -> Dict[str, np.ndarray] | List[List[Dict[str, Any]]]:
    """
    Collect episodes into arrays.

    Returns a dict with:
      - S: (n, T, 2)
      - U/W/Z: (n, T, 1)
      - action: (n, T) int8
      - reward: (n, T) float32

    If return_legacy=True, returns the original nested list-of-dict format.
    """
    if policy is None:
        env.params["offline"] = True
    else:
        env.params["offline"] = False

    set_seeds(seed)

    n_ep = int(config["episode_num"])
    T = int(env.params["episode_len"])

    S = np.empty((n_ep, T, 2), dtype=np.float32)
    U = np.empty((n_ep, T, 1), dtype=np.float32)
    W = np.empty((n_ep, T, 1), dtype=np.float32)
    Z = np.empty((n_ep, T, 1), dtype=np.float32)
    A = np.empty((n_ep, T), dtype=np.int8)
    R = np.empty((n_ep, T), dtype=np.float32)

    for n in range(n_ep):
        obs = _reset_env(env, seed=seed + 10_000 * n + 123)
        for t in range(T):
            act = int(env.act_last) if policy is None else int(policy.eps_greedy(obs))
            A[n, t] = act

            S[n, t, :] = obs["S"]
            U[n, t, 0] = obs["U"][0]
            W[n, t, 0] = obs["W"][0]
            Z[n, t, 0] = obs["Z"][0]

            obs, r, done = _step_env(env, act)
            R[n, t] = r
            if done and (t < T - 1):
                # pad remaining steps by repeating the terminal obs and zero reward
                for tt in range(t + 1, T):
                    A[n, tt] = act
                    S[n, tt, :] = obs["S"]
                    U[n, tt, 0] = obs["U"][0]
                    W[n, tt, 0] = obs["W"][0]
                    Z[n, tt, 0] = obs["Z"][0]
                    R[n, tt] = 0.0
                break

    if return_legacy:
        Episodes: List[List[Dict[str, Any]]] = []
        for n in range(n_ep):
            ep = []
            for t in range(T):
                state = {"S": S[n, t], "U": U[n, t], "W": W[n, t], "Z": Z[n, t]}
                ep.append({"state": state, "action": int(A[n, t]), "reward": float(R[n, t])})
            Episodes.append(ep)
        return Episodes

    return {"S": S, "U": U, "W": W, "Z": Z, "action": A, "reward": R}


def batch_cat(
    Episodes: Dict[str, np.ndarray] | List[List[Dict[str, Any]]],
    action_space,
    observation_space,
    device: str,
    *,
    dtype: torch.dtype = torch.float32,
    verbose: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Convert collected episodes into torch tensors with the original repo's layout:
      - action: (n, 1, T)
      - reward: (n, 1, T)
      - each obs key: (n, dim, T)
    """
    if isinstance(Episodes, dict):
        S = Episodes["S"]  # (n,T,2)
        W = Episodes["W"]  # (n,T,1)
        Z = Episodes["Z"]  # (n,T,1)
        U = Episodes.get("U", None)
        A = Episodes["action"]  # (n,T)
        R = Episodes["reward"]  # (n,T)

        n, T, _ = S.shape
        out: Dict[str, torch.Tensor] = {}
        out["action"] = torch.as_tensor(A, device=device, dtype=torch.int64).reshape(n, 1, T).to(torch.int8)
        out["reward"] = torch.as_tensor(R, device=device, dtype=dtype).reshape(n, 1, T)
        out["S"] = torch.as_tensor(S, device=device, dtype=dtype).transpose(1, 2).contiguous()
        out["W"] = torch.as_tensor(W, device=device, dtype=dtype).transpose(1, 2).contiguous()
        out["Z"] = torch.as_tensor(Z, device=device, dtype=dtype).transpose(1, 2).contiguous()
        if U is not None:
            out["U"] = torch.as_tensor(U, device=device, dtype=dtype).transpose(1, 2).contiguous()

        if verbose:
            for t in range(T):
                msg = [f"Step{t}"]
                for a in range(action_space.start, action_space.start + action_space.n):
                    cnt = int((out["action"][:, :, t] == a).sum().item())
                    msg.append(f"a{a}:{cnt}")
                print(" | ".join(msg), flush=True)

        return out

    # legacy conversion path (slower)
    obs_spaces = observation_space.spaces if hasattr(observation_space, "spaces") else observation_space
    n = len(Episodes)
    T = len(Episodes[0])
    out: Dict[str, torch.Tensor] = {}
    out["action"] = torch.empty((n, 1, T), dtype=torch.int8, device=device)
    out["reward"] = torch.empty((n, 1, T), dtype=dtype, device=device)
    for key, space in obs_spaces.items():
        shape = tuple(space.shape)
        out[key] = torch.empty((n,) + shape + (T,), dtype=dtype, device=device)

    for i in range(n):
        for t in range(T):
            out["action"][i, 0, t] = int(Episodes[i][t]["action"])
            out["reward"][i, 0, t] = float(Episodes[i][t]["reward"])
            st = Episodes[i][t]["state"]
            for key in obs_spaces.keys():
                out[key][i, :, t] = torch.as_tensor(st[key], device=device, dtype=dtype).reshape(-1)

    return out


def sample_initial_states(env, n: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample (S1, W1) pairs by resetting the env n times.
    This is used to approximate E[v_pi(W1,S1)] without running full trajectories.
    """
    set_seeds(seed)
    S0 = np.empty((n, 2), dtype=np.float32)
    W0 = np.empty((n, 1), dtype=np.float32)
    for i in range(n):
        obs = _reset_env(env, seed=seed + 99_999 + i)
        S0[i] = obs["S"]
        W0[i, 0] = obs["W"][0]
    return S0, W0


def MC_evaluator(env, policy, config: Dict[str, Any], seed: int = 0) -> float:
    """
    Monte Carlo evaluation of a fixed policy. Returns mean cumulative reward.
    """
    set_seeds(seed)
    n_ep = int(config["episode_num"])
    T = int(env.params["episode_len"])

    returns = np.empty((n_ep,), dtype=np.float32)
    for n in range(n_ep):
        obs = _reset_env(env, seed=seed + 1_000_000 + n)
        ret = 0.0
        for t in range(T):
            a = int(policy.eps_greedy(obs))
            obs, r, done = _step_env(env, a)
            ret += float(r)
            if done:
                break
        returns[n] = ret

    mean_reward = float(np.mean(returns))
    return mean_reward
