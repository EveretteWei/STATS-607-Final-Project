"""
prox_fqe.py (optimized)

Sequential NPIV (prox-FQE) recursion for confounded POMDP OPE.

Key properties preserved from the original repository:
  - backward recursion over t (sequential NPIV)
  - per-action NPIV fit at each t using ApproxRKHSIVCV

Key improvements:
  - avoid repeated torch.cat in inner loops by precomputing WS/ZS blocks
  - avoid in-place modification of reward tensors by maintaining V_next explicitly
  - dtype is configurable (float32 recommended on GPU)
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple, Any

import torch
from torch import Tensor

from rkhs_torch import ApproxRKHSIVCV
from utils import batch_cat


def fit_qpi_cv_step(
    *,
    WS_t: Tensor,
    ZS_t: Tensor,
    S_t: Tensor,
    Y_t: Tensor,
    A_t: Tensor,
    action_space,
    policy,
    option: Dict[str, Any],
    device: str,
) -> Tuple[Callable[[Tensor, Tensor], Tensor], List[ApproxRKHSIVCV], Tensor]:
    """
    Fit q_t for each action at a single time step and return:
      - v_t(W,S) callable
      - list of fitted q models (one per action)
      - V_t evaluated on the training samples (used for recursion)
    """
    actions = list(range(action_space.start, action_space.start + action_space.n))
    q_models: List[ApproxRKHSIVCV] = []

    for a in actions:
        idx = (A_t.reshape(-1) == int(a))
        n_a = int(idx.sum().item())
        if n_a < 3:
            # degenerate: fit a trivial model that predicts zero
            m = ApproxRKHSIVCV(
                n_components=3,
                gamma_f=option.get("gamma_f", "auto"),
                n_gamma_hs=int(option.get("n_gamma_hs", 20)),
                n_alphas=int(option.get("n_alphas", 30)),
                cv=int(option.get("cv", 5)),
                device=device,
            )
            # fake fitted attributes for predict
            m.gamma_h = 1.0
            m.best_alpha = 1.0
            m.a = torch.zeros((3, 1), device=WS_t.device, dtype=WS_t.dtype)
            m._basis_X = WS_t[:3].clone()
            m._invsqrt_X = torch.eye(3, device=WS_t.device, dtype=WS_t.dtype)
            m.transX = type("Identity", (), {"transform": lambda self, x: x})()
            q_models.append(m)
            continue

        n_comp = max(int(n_a ** 0.5), 25)
        n_comp = min(n_comp, n_a)

        model = ApproxRKHSIVCV(
            n_components=n_comp,
            gamma_f=option.get("gamma_f", "auto"),
            n_gamma_hs=int(option.get("n_gamma_hs", 20)),
            n_alphas=int(option.get("n_alphas", 30)),
            cv=int(option.get("cv", 5)),
            device=device,
        ).fit(
            WS_t[idx, :],
            Y_t[idx, :],
            ZS_t[idx, :],
        )
        q_models.append(model)

    def v_t(W: Tensor, S: Tensor) -> Tensor:
        WS = torch.cat((W, S), dim=1)
        out = torch.zeros((S.shape[0], 1), device=S.device, dtype=WS.dtype)
        for k, a in enumerate(actions):
            out = out + policy.prob_torch(a, S).to(dtype=out.dtype) * q_models[k].predict(WS)
        return out

    # compute V_t on training samples using the precomputed WS_t
    V_on_train = torch.zeros((S_t.shape[0], 1), device=S_t.device, dtype=WS_t.dtype)
    for k, a in enumerate(actions):
        V_on_train = V_on_train + policy.prob_torch(a, S_t).to(dtype=V_on_train.dtype) * q_models[k].predict(WS_t)

    return v_t, q_models, V_on_train


def fit_v0(
    *,
    Episodes,
    action_space,
    observation_space,
    policy,
    option: Dict[str, Any],
    device: str,
    dtype: torch.dtype = torch.float32,
) -> Callable[[Tensor, Tensor], Tensor]:
    """
    Fit v_1^pi via backward recursion.

    Episodes can be either:
      - dict-of-arrays from utils_optimized.batch_data_collector, or
      - legacy list-of-list-of-dict episodes.
    """
    Episodes_cat = batch_cat(
        Episodes,
        action_space,
        observation_space,
        device,
        dtype=dtype,
        verbose=bool(option.get("verbose", False)),
    )

    S_all = Episodes_cat["S"]  # (n, 2, T)
    W_all = Episodes_cat["W"]  # (n, 1, T)
    Z_all = Episodes_cat["Z"]  # (n, 1, T)
    A_all = Episodes_cat["action"]  # (n, 1, T)
    R_all = Episodes_cat["reward"]  # (n, 1, T)

    n, _, T = S_all.shape

    # precompute concatenations for all t
    WS_all = torch.cat((W_all, S_all), dim=1)  # (n, 3, T)
    ZS_all = torch.cat((Z_all, S_all), dim=1)  # (n, 3, T)

    V_next = torch.zeros((n, 1), device=device, dtype=dtype)
    v0_fn: Callable[[Tensor, Tensor], Tensor] | None = None

    for t in reversed(range(T)):
        Y_t = R_all[:, :, t] + V_next  # (n,1)

        v_t, q_models, V_on_train = fit_qpi_cv_step(
            WS_t=WS_all[:, :, t],
            ZS_t=ZS_all[:, :, t],
            S_t=S_all[:, :, t],
            Y_t=Y_t,
            A_t=A_all[:, :, t],
            action_space=action_space,
            policy=policy,
            option=option,
            device=device,
        )

        V_next = V_on_train
        if t == 0:
            v0_fn = v_t

    assert v0_fn is not None
    return v0_fn
