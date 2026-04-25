"""Variance-change LR score."""

from dataclasses import dataclass, field

import numpy as np

from gridcp.typing import ArrayLike
from gridcp.scores._score_helpers import as_obs


@dataclass(slots=True)
class VarianceState:
    n_samples: int = 0
    sum_sq: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))


@dataclass(frozen=True, slots=True)
class Variance:
    """Univariate variance-change LR score.

    Assumes zero-mean data.  If the data has a nonzero mean, subtract the
    known (or estimated) mean before feeding observations to this score.

    For ``n_features > 1``, scores are computed per feature and the maximum
    is used.

    For ``n_features > 1``, the score is the maximum of the feature-wise LR
    statistics. Centering by ``-1`` and the default penalty
    ``log(t p) + sqrt(log(t p))`` are based on a Wilks-style ``chi^2_1``
    approximation.
    """

    n_features: int = 1
    enable_penalty: bool = True

    def init_state(self) -> VarianceState:
        return VarianceState(sum_sq=np.zeros(self.n_features, dtype=np.float64))

    def update(self, state: VarianceState, x: ArrayLike) -> VarianceState:
        x_arr = as_obs(x, self.n_features)
        return VarianceState(
            n_samples=state.n_samples + 1,
            sum_sq=state.sum_sq + x_arr * x_arr,
        )

    def _compute_centered_scores(
        self,
        state: VarianceState,
        grid_states: list[VarianceState],
    ) -> np.ndarray:
        """Compute centered (but unpenalised) scores for every active grid candidate."""
        out = np.zeros(len(grid_states), dtype=np.float64)
        t = state.n_samples
        p = self.n_features

        for i, st in enumerate(grid_states):
            n1 = st.n_samples
            n2 = t - n1

            if n1 == 0 or n2 == 0:
                out[i] = 0.0
                continue

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

            out[i] = best if has_valid else 0.0

        return out

    def _get_penalty(self, n_samples: int) -> float:
        """Return the penalty divisor for the current sample size."""
        if self.enable_penalty:
            logg = np.log(n_samples * self.n_features)
            return logg + np.sqrt(logg)
        return 1.0

    def compute_penalised_scores(
        self,
        state: VarianceState,
        grid_states: list[VarianceState],
    ) -> np.ndarray:
        """Compute penalised variance-change scores for all active candidates."""
        return self._compute_centered_scores(state, grid_states) / self._get_penalty(
            state.n_samples
        )
