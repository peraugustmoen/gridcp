"""Multivariate mean LR score with unknown covariance."""

from dataclasses import dataclass, field

import numpy as np

from gridcp.typing import ArrayLike, PenaltyType
from gridcp.scores._score_helpers import as_obs


@dataclass(slots=True)
class MultivariateMeanUnknownCovState:
    n_samples: int = 0
    sum: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    sum_outer: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )


@dataclass(frozen=True, slots=True)
class MultivariateMeanUnknownCov:
    """Multivariate mean-change LR score with unknown covariance.

    Centering by ``-p`` and the default penalty ``sqrt(p log t) + log t`` use a
    Wilks-style ``chi^2_p`` approximation for the LR.
    """

    n_features: int
    penalty: PenaltyType = PenaltyType.TIME_DEPENDENT

    def init_state(self) -> MultivariateMeanUnknownCovState:
        return MultivariateMeanUnknownCovState(
            sum=np.zeros(self.n_features, dtype=np.float64),
            sum_outer=np.zeros((self.n_features, self.n_features), dtype=np.float64),
        )

    def update(
        self,
        state: MultivariateMeanUnknownCovState,
        x: ArrayLike,
    ) -> MultivariateMeanUnknownCovState:
        x_arr = as_obs(x, self.n_features)
        return MultivariateMeanUnknownCovState(
            n_samples=state.n_samples + 1,
            sum=state.sum + x_arr,
            sum_outer=state.sum_outer + np.outer(x_arr, x_arr),
        )

    def _compute_centered_scores(
        self,
        state: MultivariateMeanUnknownCovState,
        grid_states: list[MultivariateMeanUnknownCovState],
    ) -> np.ndarray:
        """Compute centered (but unpenalised) scores for every active grid candidate."""
        out = np.zeros(len(grid_states), dtype=np.float64)
        t = state.n_samples
        p = self.n_features
        df = float(p)

        if t < 2 * p:
            return out

        s_tot = state.sum
        sxx_tot = state.sum_outer

        for i, st in enumerate(grid_states):
            n1 = st.n_samples
            n2 = t - n1

            if n1 == 0 or n2 == 0:
                out[i] = 0.0
                continue

            s1 = st.sum
            sxx1 = st.sum_outer
            s2 = s_tot - s1
            sxx2 = sxx_tot - sxx1

            sigma_null = (sxx_tot - np.outer(s_tot, s_tot) / t) / t
            sigma_alt = (
                (sxx1 - np.outer(s1, s1) / n1) + (sxx2 - np.outer(s2, s2) / n2)
            ) / t

            sign0, logdet0 = np.linalg.slogdet(sigma_null)
            sign1, logdet1 = np.linalg.slogdet(sigma_alt)
            if sign0 <= 0 or sign1 <= 0:
                out[i] = 0.0
                continue

            lr = t * (logdet0 - logdet1)
            out[i] = lr - df

        return out

    def _get_penalty(self, n_samples: int) -> float:
        """Return the penalty divisor for the current sample size."""
        if self.penalty == PenaltyType.TIME_DEPENDENT:
            df = float(self.n_features)
            return np.sqrt(df * np.log(n_samples)) + np.log(n_samples)
        return 1.0

    def compute_penalised_scores(
        self,
        state: MultivariateMeanUnknownCovState,
        grid_states: list[MultivariateMeanUnknownCovState],
    ) -> np.ndarray:
        """Compute penalised LR score at every active grid candidate."""
        return self._compute_centered_scores(state, grid_states) / self._get_penalty(
            state.n_samples
        )
