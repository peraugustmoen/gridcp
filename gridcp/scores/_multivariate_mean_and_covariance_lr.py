"""Multivariate mean-or-covariance LR score."""

from dataclasses import dataclass

import numpy as np

from gridcp.typing import ArrayLike
from gridcp.scores._score_helpers import as_obs


@dataclass(slots=True)
class MultivariateMeanAndCovarianceLRState:
    n_samples: int = 0
    sum: np.ndarray = None
    sum_outer: np.ndarray = None


@dataclass(frozen=True, slots=True)
class MultivariateMeanAndCovarianceLR:
    """Multivariate mean-or-covariance LR score."""

    n_features: int

    def init_state(self) -> MultivariateMeanAndCovarianceLRState:
        return MultivariateMeanAndCovarianceLRState(
            sum=np.zeros(self.n_features, dtype=np.float64),
            sum_outer=np.zeros((self.n_features, self.n_features), dtype=np.float64),
        )

    def update(
        self,
        state: MultivariateMeanAndCovarianceLRState,
        x: ArrayLike,
    ) -> MultivariateMeanAndCovarianceLRState:
        x_arr = as_obs(x, self.n_features)
        return MultivariateMeanAndCovarianceLRState(
            n_samples=state.n_samples + 1,
            sum=state.sum + x_arr,
            sum_outer=state.sum_outer + np.outer(x_arr, x_arr),
        )

    def compute_penalised_scores(
        self,
        state: MultivariateMeanAndCovarianceLRState,
        grid_states: list[MultivariateMeanAndCovarianceLRState],
    ) -> np.ndarray:
        out = np.zeros(len(grid_states), dtype=np.float64)
        t = state.n_samples
        p = self.n_features
        df = float(p + (p * (p + 1)) // 2)
        penalty = np.sqrt(df * np.log(t / 0.05)) + np.log(t / 0.05)

        s_tot = state.sum
        sxx_tot = state.sum_outer

        for i, st in enumerate(grid_states):
            n1 = st.n_samples
            n2 = t - n1

            if n1 <= 2 * p or n2 <= 2 * p:
                out[i] = 0.0
                continue

            s1 = st.sum
            sxx1 = st.sum_outer
            s2 = s_tot - s1
            sxx2 = sxx_tot - sxx1

            sigma_tot = (sxx_tot - np.outer(s_tot, s_tot) / t) / t
            sigma_1 = (sxx1 - np.outer(s1, s1) / n1) / n1
            sigma_2 = (sxx2 - np.outer(s2, s2) / n2) / n2

            sign0, logdet0 = np.linalg.slogdet(sigma_tot)
            sign1, logdet1 = np.linalg.slogdet(sigma_1)
            sign2, logdet2 = np.linalg.slogdet(sigma_2)
            if sign0 <= 0 or sign1 <= 0 or sign2 <= 0:
                out[i] = 0.0
                continue

            lr = t * logdet0 - n1 * logdet1 - n2 * logdet2
            out[i] = (lr - df) / penalty

        return out
