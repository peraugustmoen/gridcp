"""Variance-change LR score."""

from dataclasses import dataclass

import numpy as np

from gridcp.typing import ArrayLike
from gridcp.scores._score_helpers import as_obs


@dataclass(slots=True)
class VarianceState:
    n_samples: int = 0
    sum_sq: np.ndarray = None


@dataclass(frozen=True, slots=True)
class Variance:
    """Univariate variance-change LR score.

    For `n_features > 1`, scores are computed per feature and the maximum is used.
    """

    n_features: int = 1

    def init_state(self) -> VarianceState:
        return VarianceState(sum_sq=np.zeros(self.n_features, dtype=np.float64))

    def update(self, state: VarianceState, x: ArrayLike) -> VarianceState:
        x_arr = as_obs(x, self.n_features)
        return VarianceState(
            n_samples=state.n_samples + 1,
            sum_sq=state.sum_sq + x_arr * x_arr,
        )

    def compute_penalised_scores(
        self,
        state: VarianceState,
        grid_states: list[VarianceState],
    ) -> np.ndarray:
        out = np.zeros(len(grid_states), dtype=np.float64)
        t = state.n_samples
        p = self.n_features
        penalty = np.log(t) + np.sqrt(np.log(t))

        for i, st in enumerate(grid_states):
            n1 = st.n_samples
            n2 = t - n1
            sumsq1 = st.sum_sq
            sumsq2 = state.sum_sq - sumsq1

            best = -1.0e300
            has_valid = False
            for k in range(p):
                sigma_tot = state.sum_sq[k] / t
                sigma_1 = sumsq1[k] / n1
                sigma_2 = sumsq2[k] / n2
                if sigma_tot <= 0.0 or sigma_1 <= 0.0 or sigma_2 <= 0.0:
                    continue
                has_valid = True
                lr = t * np.log(sigma_tot) - n1 * np.log(sigma_1) - n2 * np.log(sigma_2)
                score = lr - 1.0
                if score > best:
                    best = score

            raw = best if has_valid else 0.0
            out[i] = raw / penalty

        return out
