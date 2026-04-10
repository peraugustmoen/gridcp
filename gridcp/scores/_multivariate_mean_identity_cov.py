"""Multivariate mean LR score under identity covariance."""

from dataclasses import dataclass, field

import numpy as np

from gridcp.typing import ArrayLike, PenaltyType
from gridcp.scores._score_helpers import as_obs


@dataclass(slots=True)
class MultivariateMeanIdentityCovState:
    n_samples: int = 0
    sum: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))


@dataclass(frozen=True, slots=True)
class MultivariateMeanIdentityCov:
    """Multivariate mean-change LR score under identity covariance.

    The sparse coordinate-wise maximum uses centering ``-1`` and penalty
    ``log(t) + log(p)``, while the dense LR statistic uses centering ``-p`` and
    penalty ``sqrt(p log t) + log t``. Both come from non-asymptotic Gaussian
    concentration bounds in this model.
    """

    n_features: int
    penalty: PenaltyType = PenaltyType.TIME_DEPENDENT

    def init_state(self) -> MultivariateMeanIdentityCovState:
        return MultivariateMeanIdentityCovState(
            sum=np.zeros(self.n_features, dtype=np.float64)
        )

    def update(
        self,
        state: MultivariateMeanIdentityCovState,
        x: ArrayLike,
    ) -> MultivariateMeanIdentityCovState:
        x_arr = as_obs(x, self.n_features)
        return MultivariateMeanIdentityCovState(
            n_samples=state.n_samples + 1,
            sum=state.sum + x_arr,
        )

    def _compute_centered_scores(
        self,
        state: MultivariateMeanIdentityCovState,
        grid_states: list[MultivariateMeanIdentityCovState],
    ) -> np.ndarray:
        """Compute centered (but unpenalised) scores for every active grid candidate."""
        out = np.zeros((len(grid_states), 2), dtype=np.float64)
        t = state.n_samples
        p = self.n_features

        for i, st in enumerate(grid_states):
            n1 = st.n_samples
            n2 = t - n1

            if n1 == 0 or n2 == 0:
                out[i, :] = 0.0
                continue

            mean1 = st.sum / n1
            mean2 = (state.sum - st.sum) / n2
            diff = mean1 - mean2
            lr_dense = (n1 * n2 / t) * float(np.dot(diff, diff))
            lr_sparse = (n1 * n2 / t) * np.max(diff * diff)
            out[i, 0] = lr_sparse - 1
            out[i, 1] = lr_dense - p

        return out

    def _get_penalty(self, n_samples: int) -> np.ndarray:
        """Return the penalty divisor for the current sample size."""
        if self.penalty == PenaltyType.TIME_DEPENDENT:
            t = n_samples
            p = self.n_features
            return np.array(
                [
                    np.log(t) + np.log(p),
                    np.sqrt(p * np.log(t)) + np.log(t),
                ]
            )
        return np.ones(2)

    def compute_penalised_scores(
        self,
        state: MultivariateMeanIdentityCovState,
        grid_states: list[MultivariateMeanIdentityCovState],
    ) -> np.ndarray:
        return self._compute_centered_scores(state, grid_states) / self._get_penalty(
            state.n_samples
        )
