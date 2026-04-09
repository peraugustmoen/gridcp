"""Multivariate mean-or-covariance LR score."""

from dataclasses import dataclass, field

import numpy as np

from gridcp.typing import ArrayLike, PenaltyType
from gridcp.scores._score_helpers import as_obs


@dataclass(slots=True)
class MultivariateMeanOrCovarianceState:
    n_samples: int = 0
    sum: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    sum_outer: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )


@dataclass(frozen=True, slots=True)
class MultivariateMeanOrCovariance:
    """Multivariate mean-or-covariance LR score.

    Centering by ``-(p + p(p+1)/2)`` and the default penalty
    ``sqrt(df log t) + log t`` with ``df = p + p(p+1)/2`` use a Wilks-style
    chi-square approximation.
    """

    n_features: int
    penalty: PenaltyType = PenaltyType.TIME_DEPENDENT

    def init_state(self) -> MultivariateMeanOrCovarianceState:
        return MultivariateMeanOrCovarianceState(
            sum=np.zeros(self.n_features, dtype=np.float64),
            sum_outer=np.zeros((self.n_features, self.n_features), dtype=np.float64),
        )

    def update(
        self,
        state: MultivariateMeanOrCovarianceState,
        x: ArrayLike,
    ) -> MultivariateMeanOrCovarianceState:
        x_arr = as_obs(x, self.n_features)
        return MultivariateMeanOrCovarianceState(
            n_samples=state.n_samples + 1,
            sum=state.sum + x_arr,
            sum_outer=state.sum_outer + np.outer(x_arr, x_arr),
        )

    def _compute_centered_scores(
        self,
        state: MultivariateMeanOrCovarianceState,
        grid_states: list[MultivariateMeanOrCovarianceState],
    ) -> np.ndarray:
        """Compute centered (but unpenalised) scores for every active grid candidate."""
        out = np.zeros(len(grid_states), dtype=np.float64)
        t = state.n_samples
        p = self.n_features
        df = float(p + (p * (p + 1)) // 2)

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
            out[i] = lr - df

        return out

    def _get_penalty(self, n_samples: int) -> float:
        """Return the penalty divisor for the current sample size."""
        if self.penalty == PenaltyType.TIME_DEPENDENT:
            p = self.n_features
            df = float(p + (p * (p + 1)) // 2)
            return np.sqrt(df * np.log(n_samples)) + np.log(n_samples)
        return 1.0

    def compute_penalised_scores(
        self,
        state: MultivariateMeanOrCovarianceState,
        grid_states: list[MultivariateMeanOrCovarianceState],
    ) -> np.ndarray:
        return self._compute_centered_scores(state, grid_states) / self._get_penalty(
            state.n_samples
        )
