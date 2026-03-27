"""Multivariate mean LR score under identity covariance."""

from dataclasses import dataclass

import numpy as np

from gridcp.typing import ArrayLike
from gridcp.scores._score_helpers import as_obs


@dataclass(slots=True)
class MultivariateMeanIdentityCovState:
    n_samples: int = 0
    sum: np.ndarray = None


@dataclass(frozen=True, slots=True)
class MultivariateMeanIdentityCov:
    """Multivariate mean-change LR score under identity covariance."""

    n_features: int

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

    def compute_penalised_scores(
        self,
        state: MultivariateMeanIdentityCovState,
        grid_states: list[MultivariateMeanIdentityCovState],
    ) -> np.ndarray:
        out = np.zeros(len(grid_states), dtype=np.float64)
        t = state.n_samples
        p = self.n_features
        penalty = np.log(t) + np.sqrt(p * np.log(t))

        for i, st in enumerate(grid_states):
            n1 = st.n_samples
            n2 = t - n1
            mean1 = st.sum / n1
            mean2 = (state.sum - st.sum) / n2
            diff = mean1 - mean2
            lr = (n1 * n2 / t) * float(np.dot(diff, diff))
            out[i] = (lr - p) / penalty

        return out
