"""
envs.py

Gymnasium-compatible continuous-state confounded POMDP simulation environment
from Miao, Qi, Zhang (NeurIPS 2022).

This is a refactor of the original repo's ContinuousEnv to:
  - remove the vendored legacy gym clone
  - adopt Gymnasium reset/step signatures
  - use a single NumPy RNG for reproducible simulation
  - avoid slow scipy.stats samplers by using NumPy primitives

Public behavior is preserved:
  - action space is Discrete(2) with actions {0, 1}
  - the behavior policy for offline data generation is implemented inside the env
    when params["offline"] is True by ignoring the passed-in action.
  - observations are Dict with keys {"S","U","W","Z"}.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


class ContinuousEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, params: Dict[str, Any], *, device: str = "cpu", render_mode: Optional[str] = None):
        super().__init__()
        self.name = "Continuous Env"
        self.device = device
        self.render_mode = render_mode

        # params
        self.params = params
        self.COV = np.asarray([[1.0, 0.25, 0.5], [0.25, 1.0, 0.5], [0.5, 0.5, 1.0]], dtype=np.float64)

        self.alpha_0 = float(params["alpha_0"])
        self.alpha_a = float(params["alpha_a"])
        self.alpha_s = np.asarray(params["alpha_s"], dtype=np.float64).reshape(2)

        self.mu_0 = float(params["mu_0"])
        self.mu_a = float(params["mu_a"])
        self.mu_s = np.asarray(params["mu_s"], dtype=np.float64).reshape(2)

        self.kappa_0 = float(params["kappa_0"])
        self.kappa_a = float(params["kappa_a"])
        self.kappa_s = np.asarray(params["kappa_s"], dtype=np.float64).reshape(2)

        self.t_0 = float(params["t_0"])
        self.t_u = float(params["t_u"])
        self.t_s = np.asarray(params["t_s"], dtype=np.float64).reshape(2)

        self._episode_len = int(params["episode_len"])

        # spaces
        self.action_space = spaces.Discrete(2)  # {0,1}
        self.observation_space = spaces.Dict(
            {
                "S": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
                "U": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                "W": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                "Z": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            }
        )

        # internal RNG/state
        self._rng = np.random.default_rng(None)
        self._t = 0
        self.obs_last: Dict[str, np.ndarray] = {}
        self.act_last: int = 0  # action used by the offline behavior policy

        # precompute Cholesky for the proxy Gaussian
        self._cov_chol = np.linalg.cholesky(self.COV)

    # ---------------- Gymnasium API ----------------
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))

        # sample S_1 from observation_space; keep it float64 internally
        s = self.observation_space["S"].sample().astype(np.float64)

        # generate pseudo action A in {-1,+1} from S to generate Z,U,W
        logit = -(self.t_0 + self.t_u * self.kappa_0 + (self.t_s + self.t_u * self.kappa_s) @ s)
        p = float(_sigmoid(logit))
        A = -1 + 2 * int(self._rng.random() < p)

        # sample (Z, W, U) | (S, A) ~ N(mean, COV)
        mean = np.array(
            [
                self.alpha_0 + self.alpha_a * A + self.alpha_s @ s,
                self.mu_0 + self.mu_a * A + self.mu_s @ s,
                self.kappa_0 + self.kappa_a * A + self.kappa_s @ s,
            ],
            dtype=np.float64,
        )
        zwu = mean + self._cov_chol @ self._rng.normal(size=(3,))

        z = np.array([zwu[0]], dtype=np.float64)
        w = np.array([zwu[1]], dtype=np.float64)
        u = np.array([zwu[2]], dtype=np.float64)

        self.obs_last = {"S": s.astype(np.float32), "Z": z.astype(np.float32), "W": w.astype(np.float32), "U": u.astype(np.float32)}

        # offline behavior policy action is induced by pseudo A
        if bool(self.params.get("offline", False)):
            self.act_last = int((A + 1) // 2)  # {-1,+1} -> {0,1}

        self._t = 0
        info: Dict[str, Any] = {"t": self._t}
        return self.obs_last, info

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError("Invalid action")

        # offline mode ignores caller action
        if bool(self.params.get("offline", False)):
            action = int(self.act_last)

        # reward depends on (U_t, S_t, action)
        u = float(self.obs_last["U"][0])
        s = self.obs_last["S"].astype(np.float64)

        # original reward: expit((action-0.5)*(U + s1 - 2*s2)) + Uniform[-.05,.05]
        lin = (float(action) - 0.5) * (u + float(s[0]) - 2.0 * float(s[1]))
        r = float(_sigmoid(lin) + (self._rng.random() - 0.5) / 10.0)

        # state transition: S_{t+1} = S_t + 2*(action-0.5)*U_t*1 + eps, eps ~ N(0, I2)
        mean_s_next = s + 2.0 * (float(action) - 0.5) * u * np.ones_like(s)
        s_next = mean_s_next + self._rng.normal(size=(2,))

        # generate next proxies from pseudo action induced by S_{t+1}
        logit = -(self.t_0 + self.t_u * self.kappa_0 + (self.t_s + self.t_u * self.kappa_s) @ s_next)
        p = float(_sigmoid(logit))
        A = -1 + 2 * int(self._rng.random() < p)

        mean = np.array(
            [
                self.alpha_0 + self.alpha_a * A + self.alpha_s @ s_next,
                self.mu_0 + self.mu_a * A + self.mu_s @ s_next,
                self.kappa_0 + self.kappa_a * A + self.kappa_s @ s_next,
            ],
            dtype=np.float64,
        )
        zwu = mean + self._cov_chol @ self._rng.normal(size=(3,))

        obs = {"S": s_next.astype(np.float32), "Z": np.array([zwu[0]], dtype=np.float32), "W": np.array([zwu[1]], dtype=np.float32), "U": np.array([zwu[2]], dtype=np.float32)}
        self.obs_last = obs

        if bool(self.params.get("offline", False)):
            self.act_last = int((A + 1) // 2)
        else:
            self.act_last = int(action)

        # time and termination
        terminated = (self._t >= self._episode_len - 1)
        truncated = False
        self._t += 1

        info: Dict[str, Any] = {"t": self._t}
        return obs, r, terminated, truncated, info

    def render(self) -> None:
        if self.render_mode == "human":
            s = self.obs_last.get("S", None)
            print(f"t={self._t} S={s} act_last={self.act_last}")

    def close(self) -> None:
        return
