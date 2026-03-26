"""Regression direct score."""

from dataclasses import dataclass

import numpy as np

from gridcp.typing import ArrayLike
from gridcp.scores._score_helpers import as_obs, inv_sqrtm_pd


@dataclass(slots=True)
class RegressionDirectState:
    n_samples: int = 0
    yx_sum: np.ndarray = None
    xx_sum: np.ndarray = None


@dataclass(frozen=True, slots=True)
class RegressionDirect:
    """Direct regression-change score using covariance-normalized differences."""

    n_regressors: int

    @property
    def n_features(self) -> int:
        return self.n_regressors + 1

    def init_state(self) -> RegressionDirectState:
        q = self.n_regressors
        return RegressionDirectState(
            yx_sum=np.zeros(q, dtype=np.float64),
            xx_sum=np.zeros((q, q), dtype=np.float64),
        )

    def update(
        self,
        state: RegressionDirectState,
        x: ArrayLike,
    ) -> RegressionDirectState:
        obs = as_obs(x, self.n_features)
        y = float(obs[0])
        xx = obs[1:]

        return RegressionDirectState(
            n_samples=state.n_samples + 1,
            yx_sum=state.yx_sum + y * xx,
            xx_sum=state.xx_sum + np.outer(xx, xx),
        )

    def compute_penalised_scores(
        self,
        state: RegressionDirectState,
        grid_states: list[RegressionDirectState],
    ) -> np.ndarray:
        out = np.zeros(len(grid_states), dtype=np.float64)
        t = state.n_samples
        q = self.n_regressors
        p = self.n_features
        penalty = np.sqrt((p - 1) * np.log(t)) + np.log(t)

        for i, st in enumerate(grid_states):
            n1 = st.n_samples
            n2 = t - n1
            if n1 <= q or n2 <= q:
                out[i] = 0.0
                continue

            yx_pre = st.yx_sum
            yx_post = state.yx_sum - yx_pre
            xx_pre = st.xx_sum
            xx_post = state.xx_sum - xx_pre

            m1_inv = inv_sqrtm_pd(xx_pre)
            m2_inv = inv_sqrtm_pd(xx_post)
            diff = m1_inv @ yx_pre - m2_inv @ yx_post
            raw = 0.5 * float(np.dot(diff, diff)) - q
            out[i] = raw / penalty

        return out
