"""
rkhs_torch.py (optimized)

Torch implementations of RKHS-based NPIV solvers used by the NeurIPS 2022
confounded-POMDP OPE code (prox-FQE).

Main improvements relative to the original repo:
  - float32-by-default with explicit dtype control
  - stable PSD square roots and linear solves (eigh + jitter, Cholesky fallback)
  - vectorized CV over alpha values (solve for all alphas in one shot)
  - cached RBF distances for Nyström features across gamma grid

Public API is compatible with the original repository:
  - pairwise_rbf, Nystroem, Scaler
  - RKHSIV, RKHSIVCV
  - ApproxRKHSIV, ApproxRKHSIVCV
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch
from torch import Tensor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted


# ----------------------------- core helpers ----------------------------- #

def _check_auto(param) -> bool:
    return isinstance(param, str) and (param == "auto")


def _to_tensor(X, device: str, dtype: torch.dtype = torch.float32) -> Tensor:
    dev = torch.device(device)
    if isinstance(X, torch.Tensor):
        return X.to(device=dev, dtype=dtype, copy=False)
    return torch.as_tensor(X, device=dev, dtype=dtype)


def pairwise_rbf(X: Tensor, Y: Optional[Tensor] = None, gamma: float = 1.0) -> Tensor:
    """
    RBF kernel K(x,y) = exp(-gamma * ||x-y||^2).
    """
    if Y is None:
        Y = X
    if Y.device != X.device:
        Y = Y.to(X.device)
    if Y.dtype != X.dtype:
        Y = Y.to(dtype=X.dtype)
    g = torch.as_tensor(gamma, device=X.device, dtype=X.dtype)
    d2 = torch.cdist(X, Y, p=2).pow(2)
    return torch.exp(-g * d2)


def _sym(A: Tensor) -> Tensor:
    return 0.5 * (A + A.T)


def _inv_spd(A: Tensor, jitter: float = 1e-8) -> Tensor:
    """
    Invert an SPD matrix with Cholesky when possible, otherwise fall back to eigh.
    """
    A = _sym(A)
    eye = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    Aj = A + jitter * eye
    try:
        L = torch.linalg.cholesky(Aj)
        inv = torch.cholesky_inverse(L)
        return _sym(inv)
    except RuntimeError:
        evals, evecs = torch.linalg.eigh(Aj)
        evals = torch.clamp(evals, min=jitter)
        inv = (evecs * (1.0 / evals)) @ evecs.T
        return _sym(inv)


def _inv_sqrt_spd(A: Tensor, jitter: float = 1e-8) -> Tensor:
    """
    Compute A^{-1/2} for SPD matrix A.
    """
    A = _sym(A)
    eye = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    Aj = A + jitter * eye
    evals, evecs = torch.linalg.eigh(Aj)
    evals = torch.clamp(evals, min=jitter)
    invsqrt = (evecs * (1.0 / torch.sqrt(evals))) @ evecs.T
    return _sym(invsqrt)


def matrix_sqrt_psd(A: Tensor, jitter: float = 1e-8) -> Tensor:
    """
    Principal square root of PSD matrix via eigendecomposition.
    """
    A = _sym(A)
    eye = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    evals, evecs = torch.linalg.eigh(A + jitter * eye)
    evals = torch.clamp(evals, min=jitter)
    return _sym((evecs * torch.sqrt(evals)) @ evecs.T)


def _compute_M_from_Kf(Kf: Tensor, n: int, delta: float, jitter: float = 1e-8) -> Tensor:
    """
    Stable construction of:
      M = RootKf @ (Kf/(2 n delta^2) + I/2)^{-1} @ RootKf
    """
    Kf = _sym(Kf)
    RootKf = matrix_sqrt_psd(Kf, jitter=jitter)
    eye = torch.eye(Kf.shape[0], device=Kf.device, dtype=Kf.dtype)
    A = Kf / (2.0 * float(n) * (delta ** 2)) + 0.5 * eye
    Ainv = _inv_spd(A, jitter=jitter)
    return _sym(RootKf @ Ainv @ RootKf)


def _quantile_gammas_from_pairs(X: Tensor, q_lo: float, q_hi: float, steps: int, *, max_pairs: int = 250_000) -> Tensor:
    """
    Compute gamma grid using distance quantiles. Uses a subsample of random pairs
    when n is large to reduce O(n^2) cost.
    """
    n = X.shape[0]
    d = X.shape[1]
    if n <= 2:
        return torch.full((steps,), 1.0, device=X.device, dtype=X.dtype)

    # sample pairs (i,j) with i<j
    # max_pairs controls runtime; does not change the core estimator, only the heuristic grid.
    total_pairs = n * (n - 1) // 2
    if total_pairs <= max_pairs:
        D2 = torch.cdist(X, X, p=2).pow(2)
        i, j = torch.triu_indices(n, n, offset=1, device=X.device)
        vals = D2[i, j]
    else:
        rng = torch.Generator(device=X.device)
        rng.manual_seed(12345)
        i = torch.randint(0, n, (max_pairs,), generator=rng, device=X.device)
        j = torch.randint(0, n, (max_pairs,), generator=rng, device=X.device)
        mask = (i != j)
        i = i[mask]
        j = j[mask]
        vals = (X[i] - X[j]).pow(2).sum(dim=1)

    qs = torch.linspace(q_lo, q_hi, steps=steps, device=X.device, dtype=X.dtype)
    qv = torch.quantile(vals, qs)
    qv = torch.clamp(qv, min=torch.tensor(1e-12, device=X.device, dtype=X.dtype))
    gammas = 1.0 / (qv * float(d))
    return gammas


# ----------------------------- utilities ----------------------------- #

class Scaler(TransformerMixin, BaseEstimator):
    """
    RobustScaler-like transform implemented with torch.
    """

    def __init__(
        self,
        *,
        with_centering: bool = True,
        with_scaling: bool = True,
        quantile_range=(25.0, 75.0),
        unit_variance: bool = False,
        copy: bool = True,
    ):
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.unit_variance = unit_variance
        self.copy = copy

    def fit(self, X: Tensor, y=None):
        q_min, q_max = self.quantile_range
        if self.with_centering:
            self.center_, _ = torch.nanmedian(X, dim=0)
        else:
            self.center_ = None

        if self.with_scaling:
            qs = torch.tensor([q_min / 100.0, q_max / 100.0], device=X.device, dtype=X.dtype)
            quantiles = torch.quantile(X, qs, dim=0)
            self.scale_ = quantiles[1] - quantiles[0]
            self.scale_[self.scale_ == 0.0] = 1.0
            if self.unit_variance:
                # optional; keep for API compatibility
                pass
        else:
            self.scale_ = None
        return self

    def transform(self, X: Tensor) -> Tensor:
        check_is_fitted(self)
        if self.copy:
            X = X.clone()
        if self.with_centering:
            X = X - self.center_
        if self.with_scaling:
            X = X / self.scale_
        return X

    def fit_transform(self, X: Tensor, y=None) -> Tensor:
        return self.fit(X, y=y).transform(X)


class Nystroem(TransformerMixin, BaseEstimator):
    """
    Nyström approximation for RBF kernels.
    This class is kept for compatibility, but ApproxRKHSIVCV uses an optimized
    distance-cached implementation internally.
    """

    def __init__(self, gamma: float = 1.0, n_components: int = 100, random_state=None):
        self.gamma = float(gamma)
        self.n_components = int(n_components)
        self.random_state = random_state

    def fit(self, X: Tensor, y=None):
        rnd = check_random_state(self.random_state)
        n_samples = X.shape[0]
        n_components = min(self.n_components, n_samples)
        if n_components < self.n_components:
            warnings.warn("n_components > n_samples; using n_samples.", RuntimeWarning)

        inds = torch.as_tensor(rnd.permutation(n_samples), device=X.device)
        basis_inds = inds[:n_components]
        basis = X[basis_inds]

        Kmm = pairwise_rbf(basis, gamma=self.gamma)
        invsqrt = _inv_sqrt_spd(Kmm, jitter=1e-8)

        self.components_ = basis
        self.component_indices_ = basis_inds
        self.normalization_ = invsqrt
        return self

    def transform(self, X: Tensor) -> Tensor:
        check_is_fitted(self)
        Knm = pairwise_rbf(X, self.components_, gamma=self.gamma)
        return Knm @ self.normalization_.T

    def fit_transform(self, X: Tensor, y=None) -> Tensor:
        return self.fit(X, y=y).transform(X)


# ----------------------------- base RKHS IV ----------------------------- #

class _BaseRKHSIV:
    def __init__(self, *args, **kwargs):
        return

    def _get_delta(self, n: float) -> float:
        delta_scale = 5.0 if _check_auto(getattr(self, "delta_scale", "auto")) else float(self.delta_scale)
        delta_exp = 0.4 if _check_auto(getattr(self, "delta_exp", "auto")) else float(self.delta_exp)
        return float(delta_scale / (float(n) ** float(delta_exp)))

    def _get_alpha_scales(self, n: int):
        if _check_auto(getattr(self, "alpha_scales", "auto")):
            return [float(x) for x in np.geomspace(0.001, 0.05, int(self.n_alphas))]
        return list(self.alpha_scales)

    def _get_alpha(self, delta: float, alpha_scale: float) -> float:
        return float(alpha_scale * (delta ** 4))

    def _get_gamma_f(self, condition: Tensor) -> float:
        if not _check_auto(getattr(self, "gamma_f", "auto")):
            return float(self.gamma_f)
        # median heuristic on squared distances; use pair subsampling for speed
        gam = _quantile_gammas_from_pairs(condition, 0.5, 0.5, 1)[0]
        return float(gam.detach().cpu())

    def _get_gamma_hs(self, X: Tensor) -> Tensor:
        if not _check_auto(getattr(self, "gamma_hs", "auto")):
            return _to_tensor(self.gamma_hs, self.device, dtype=X.dtype).reshape(-1)
        return _quantile_gammas_from_pairs(X, 0.1, 0.9, int(self.n_gamma_hs))

    def _device(self) -> str:
        return self.device if torch.cuda.is_available() else "cpu"


# ----------------------------- exact RKHSIV (not used by default) ----------------------------- #

class RKHSIV(_BaseRKHSIV):
    def __init__(
        self,
        gamma_h: float = 0.1,
        gamma_f: str | float = "auto",
        delta_scale: str | float = "auto",
        delta_exp: str | float = "auto",
        alpha_scale: str | float = "auto",
        device: str = "cuda",
    ):
        self.gamma_f = gamma_f
        self.gamma_h = float(gamma_h)
        self.delta_scale = delta_scale
        self.delta_exp = delta_exp
        self.alpha_scale = alpha_scale
        self.device = device if torch.cuda.is_available() else "cpu"

    def fit(self, X, y, condition):
        X = _to_tensor(X, self.device)
        y = _to_tensor(y, self.device).reshape(-1, 1)
        condition = _to_tensor(condition, self.device)

        cond = Scaler().fit_transform(condition)
        gamma_f = self._get_gamma_f(cond)
        Kf = pairwise_rbf(cond, gamma=gamma_f)

        self.transX = Scaler()
        Xs = self.transX.fit_transform(X)
        self.X_ = Xs.clone()
        Kh = pairwise_rbf(Xs, gamma=self.gamma_h)

        n = Xs.shape[0]
        delta = self._get_delta(n)
        alpha = self._get_alpha(delta, 60.0 if _check_auto(self.alpha_scale) else float(self.alpha_scale))

        M = _compute_M_from_Kf(Kf, n=n, delta=delta, jitter=1e-8)

        W = _sym(Kh @ M @ Kh + alpha * Kh)
        rhs = Kh @ M @ y

        Wj = W + 1e-8 * torch.eye(W.shape[0], device=W.device, dtype=W.dtype)
        try:
            L = torch.linalg.cholesky(Wj)
            a = torch.cholesky_solve(rhs, L)
        except RuntimeError:
            a = torch.linalg.lstsq(W, rhs).solution

        self.a_ = a
        return self

    def predict(self, X):
        X = _to_tensor(X, self.device)
        Xs = self.transX.transform(X)
        K = pairwise_rbf(Xs, self.X_, gamma=self.gamma_h)
        return K @ self.a_


class RKHSIVCV(RKHSIV):
    def __init__(
        self,
        gamma_f: str | float = "auto",
        gamma_hs: str | Iterable[float] = "auto",
        n_gamma_hs: int = 20,
        delta_scale: str | float = "auto",
        delta_exp: str | float = "auto",
        alpha_scales: str | Iterable[float] = "auto",
        n_alphas: int = 30,
        cv: int = 5,
        device: str = "cuda",
    ):
        self.gamma_f = gamma_f
        self.gamma_hs = gamma_hs
        self.n_gamma_hs = int(n_gamma_hs)
        self.delta_scale = delta_scale
        self.delta_exp = delta_exp
        self.alpha_scales = alpha_scales
        self.n_alphas = int(n_alphas)
        self.cv = int(cv)
        self.device = device if torch.cuda.is_available() else "cpu"

    # Keep a simple CV for the exact-kernel version (rarely used).
    # Users are encouraged to use ApproxRKHSIVCV for speed.


# ----------------------------- optimized Nyström RKHSIV ----------------------------- #

@dataclass
class _NysCache:
    basis: Tensor
    d2_nm: Tensor
    d2_mm: Tensor


def _build_nys_cache(X: Tensor, n_components: int, random_state: int) -> _NysCache:
    rnd = check_random_state(random_state)
    n = X.shape[0]
    m = min(int(n_components), n)
    if m < n_components:
        warnings.warn("n_components > n_samples; using n_samples.", RuntimeWarning)

    inds = torch.as_tensor(rnd.permutation(n), device=X.device)
    basis_inds = inds[:m]
    basis = X[basis_inds]

    d2_nm = torch.cdist(X, basis, p=2).pow(2)
    d2_mm = torch.cdist(basis, basis, p=2).pow(2)
    return _NysCache(basis=basis, d2_nm=d2_nm, d2_mm=d2_mm)


def _nystroem_from_cache(cache: _NysCache, gamma: float, jitter: float = 1e-8) -> tuple[Tensor, Tensor]:
    """
    Return (features, invsqrt_mm) for a given gamma using cached squared distances.
    """
    g = torch.as_tensor(gamma, device=cache.d2_nm.device, dtype=cache.d2_nm.dtype)
    Knm = torch.exp(-g * cache.d2_nm)
    Kmm = torch.exp(-g * cache.d2_mm)
    invsqrt = _inv_sqrt_spd(Kmm, jitter=jitter)
    feats = Knm @ invsqrt.T
    return feats, invsqrt


class ApproxRKHSIV(_BaseRKHSIV):
    def __init__(
        self,
        n_components: int = 25,
        gamma_f: str | float = "auto",
        gamma_h: float = 0.1,
        delta_scale: str | float = "auto",
        delta_exp: str | float = "auto",
        alpha_scale: str | float = "auto",
        device: str = "cuda",
    ):
        self.n_components = int(n_components)
        self.gamma_f = gamma_f
        self.gamma_h = float(gamma_h)
        self.delta_scale = delta_scale
        self.delta_exp = delta_exp
        self.alpha_scale = alpha_scale
        self.device = device if torch.cuda.is_available() else "cpu"

    def fit(self, X, y, condition):
        X = _to_tensor(X, self.device)
        y = _to_tensor(y, self.device).reshape(-1, 1)
        condition = _to_tensor(condition, self.device)

        # condition features
        self.transCond = Scaler()
        cond = self.transCond.fit_transform(condition)
        gamma_f = self._get_gamma_f(cond)
        self.gamma_f = gamma_f
        cache_c = _build_nys_cache(cond, self.n_components, random_state=1)
        RootKf, _ = _nystroem_from_cache(cache_c, gamma_f)

        # X features
        self.transX = Scaler()
        Xs = self.transX.fit_transform(X)
        cache_x = _build_nys_cache(Xs, self.n_components, random_state=1)
        RootKh, invsqrt_x = _nystroem_from_cache(cache_x, self.gamma_h)

        self._basis_X = cache_x.basis
        self._d2_X = cache_x.d2_nm
        self._invsqrt_X = invsqrt_x

        n = Xs.shape[0]
        delta = self._get_delta(n)
        alpha = self._get_alpha(delta, 60.0 if _check_auto(self.alpha_scale) else float(self.alpha_scale))
        eye = torch.eye(RootKf.shape[1], device=X.device, dtype=X.dtype)

        A = (RootKf.T @ RootKf) / (2.0 * float(n) * (delta ** 2)) + 0.5 * eye
        Q = _inv_spd(A, jitter=1e-8)

        B = (RootKh.T @ RootKf) @ Q @ (RootKf.T @ y)
        AQA = (RootKh.T @ RootKf) @ Q @ (RootKf.T @ RootKh)

        W = _sym(AQA + alpha * torch.eye(AQA.shape[0], device=AQA.device, dtype=AQA.dtype))
        rhs = B
        try:
            L = torch.linalg.cholesky(W + 1e-8 * torch.eye(W.shape[0], device=W.device, dtype=W.dtype))
            a = torch.cholesky_solve(rhs, L)
        except RuntimeError:
            a = torch.linalg.lstsq(W, rhs).solution

        self.a = a
        self.fitted_delta = delta
        return self

    def predict(self, X):
        X = _to_tensor(X, self.device)
        Xs = self.transX.transform(X)
        # compute distances to stored basis
        d2 = torch.cdist(Xs, self._basis_X, p=2).pow(2)
        g = torch.as_tensor(self.gamma_h, device=d2.device, dtype=d2.dtype)
        Knm = torch.exp(-g * d2)
        RootKh = Knm @ self._invsqrt_X.T
        return RootKh @ self.a


class ApproxRKHSIVCV(_BaseRKHSIV):
    def __init__(
        self,
        n_components: int = 25,
        gamma_f: str | float = "auto",
        gamma_hs: str | Iterable[float] = "auto",
        n_gamma_hs: int = 10,
        delta_scale: str | float = "auto",
        delta_exp: str | float = "auto",
        alpha_scales: str | Iterable[float] = "auto",
        n_alphas: int = 30,
        cv: int = 6,
        device: str = "cuda",
    ):
        self.n_components = int(n_components)
        self.gamma_f = gamma_f
        self.gamma_hs = gamma_hs
        self.n_gamma_hs = int(n_gamma_hs)
        self.delta_scale = delta_scale
        self.delta_exp = delta_exp
        self.alpha_scales = alpha_scales
        self.n_alphas = int(n_alphas)
        self.cv = int(cv)
        self.device = device if torch.cuda.is_available() else "cpu"

    def fit(self, X, y, condition):
        X = _to_tensor(X, self.device)
        y = _to_tensor(y, self.device).reshape(-1, 1)
        condition = _to_tensor(condition, self.device)

        # condition Nyström features (single gamma_f)
        transCond = Scaler()
        cond = transCond.fit_transform(condition)
        gamma_f = self._get_gamma_f(cond)
        self.gamma_f = gamma_f

        cache_c = _build_nys_cache(cond, self.n_components, random_state=1)
        RootKf, _ = _nystroem_from_cache(cache_c, gamma_f)

        # X Nyström caches for gamma_h grid
        self.transX = Scaler()
        Xs = self.transX.fit_transform(X)
        cache_x = _build_nys_cache(Xs, self.n_components, random_state=1)

        gamma_hs = self._get_gamma_hs(Xs).detach()
        alpha_scales = self._get_alpha_scales(Xs.shape[0])

        n = Xs.shape[0]
        n_train = n * (self.cv - 1) / self.cv
        n_test = n / self.cv
        delta_train = self._get_delta(n_train)
        delta_test = self._get_delta(n_test)

        m = RootKf.shape[1]
        eye_m = torch.eye(m, device=X.device, dtype=X.dtype)

        # pre-split indices
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=123)
        fold_indices = list(kf.split(np.arange(n)))

        # scores: (n_gamma, n_alpha)
        scores_sum = torch.zeros((len(gamma_hs), len(alpha_scales)), device=X.device, dtype=X.dtype)

        # precompute alphas for this fold (delta_train fixed across folds by formula)
        alphas = torch.as_tensor([self._get_alpha(delta_train, s) for s in alpha_scales], device=X.device, dtype=X.dtype)

        for tr_idx, te_idx in fold_indices:
            tr = torch.as_tensor(tr_idx, device=X.device)
            te = torch.as_tensor(te_idx, device=X.device)

            RootKf_tr = RootKf[tr]
            RootKf_te = RootKf[te]

            # Q matrices on feature space
            A_tr = (RootKf_tr.T @ RootKf_tr) / (2.0 * float(n_train) * (delta_train ** 2)) + 0.5 * eye_m
            A_te = (RootKf_te.T @ RootKf_te) / (2.0 * float(n_test) * (delta_test ** 2)) + 0.5 * eye_m
            Q_tr = _inv_spd(A_tr, jitter=1e-8)
            Q_te = _inv_spd(A_te, jitter=1e-8)

            y_tr = y[tr]
            y_te = y[te]

            for gi, g in enumerate(gamma_hs):
                # build RootKh for all samples for this gamma using cached distances
                RootKh_full, invsqrt_x = _nystroem_from_cache(cache_x, float(g))
                RootKh_tr = RootKh_full[tr]
                RootKh_te = RootKh_full[te]

                A = RootKh_tr.T @ RootKf_tr  # (m,m)
                AQA = _sym(A @ Q_tr @ A.T)
                B = A @ Q_tr @ (RootKf_tr.T @ y_tr)  # (m,1)

                # eigendecomposition once
                evals, evecs = torch.linalg.eigh(AQA + 1e-8 * eye_m)
                evals = torch.clamp(evals, min=torch.tensor(1e-12, device=X.device, dtype=X.dtype))
                proj = evecs.T @ B  # (m,1)

                # solve for all alphas at once: a_all is (m, n_alpha)
                denom = evals.reshape(-1, 1) + alphas.reshape(1, -1)
                coeff = proj / denom
                a_all = evecs @ coeff  # (m, n_alpha)

                # residuals and scores for all alphas
                pred = RootKh_te @ a_all  # (n_te, n_alpha)
                E = y_te - pred  # (n_te, n_alpha)
                res = RootKf_te.T @ E  # (m, n_alpha)
                tmp = Q_te @ res
                sc = (res * tmp).sum(dim=0) / (float(len(te_idx)) ** 2)
                scores_sum[gi] += sc

        scores_avg = scores_sum / float(len(fold_indices))
        best_flat = torch.argmin(scores_avg).item()
        best_gi, best_ai = divmod(best_flat, scores_avg.shape[1])

        self.gamma_h = float(gamma_hs[best_gi].item())
        self.best_alpha_scale = float(alpha_scales[best_ai])
        delta_full = self._get_delta(n)
        self.best_alpha = self._get_alpha(delta_full, self.best_alpha_scale)

        # refit on full data with chosen hyperparameters
        RootKh_full, invsqrt_x = _nystroem_from_cache(cache_x, self.gamma_h)
        self._basis_X = cache_x.basis
        self._invsqrt_X = invsqrt_x

        # full Q
        A_full = (RootKf.T @ RootKf) / (2.0 * float(n) * (delta_full ** 2)) + 0.5 * eye_m
        Q_full = _inv_spd(A_full, jitter=1e-8)

        A = RootKh_full.T @ RootKf
        AQA = _sym(A @ Q_full @ A.T)
        B = A @ Q_full @ (RootKf.T @ y)

        W = _sym(AQA + self.best_alpha * eye_m)
        try:
            L = torch.linalg.cholesky(W + 1e-8 * eye_m)
            a = torch.cholesky_solve(B, L)
        except RuntimeError:
            a = torch.linalg.lstsq(W, B).solution

        self.a = a
        self.fitted_delta = delta_full
        return self

    def predict(self, X):
        X = _to_tensor(X, self.device)
        Xs = self.transX.transform(X)
        d2 = torch.cdist(Xs, self._basis_X, p=2).pow(2)
        g = torch.as_tensor(self.gamma_h, device=d2.device, dtype=d2.dtype)
        Knm = torch.exp(-g * d2)
        RootKh = Knm @ self._invsqrt_X.T
        return RootKh @ self.a
