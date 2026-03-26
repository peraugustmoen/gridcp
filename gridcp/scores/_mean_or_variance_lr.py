"""Mean-or-variance LR score."""

from dataclasses import dataclass

import numpy as np

from gridcp.typing import ArrayLike
from gridcp.scores._score_helpers import as_obs


@dataclass(slots=True)
class MeanOrVarianceLRState:
    n_samples: int = 0
    sum: np.ndarray = None
    sum_sq: np.ndarray = None


@dataclass(frozen=True, slots=True)
class MeanOrVarianceLR:
    """Univariate mean-or-variance LR score.

    For `n_features > 1`, scores are computed per feature and the maximum is used.
    """

    n_features: int = 1

    def init_state(self) -> MeanOrVarianceLRState:
        return MeanOrVarianceLRState(
            sum=np.zeros(self.n_features, dtype=np.float64),
            sum_sq=np.zeros(self.n_features, dtype=np.float64),
        )

    def update(
        self, state: MeanOrVarianceLRState, x: ArrayLike
    ) -> MeanOrVarianceLRState:
        x_arr = as_obs(x, self.n_features)
        return MeanOrVarianceLRState(
            n_samples=state.n_samples + 1,
            sum=state.sum + x_arr,
            sum_sq=state.sum_sq + x_arr * x_arr,
        )

    def compute_penalised_scores(
        self,
        state: MeanOrVarianceLRState,
        grid_states: list[MeanOrVarianceLRState],
    ) -> np.ndarray:
        out = np.zeros(len(grid_states), dtype=np.float64)
        t = state.n_samples
        p = self.n_features
        penalty = np.log(2.0 * t) + np.sqrt(np.log(2.0 * t))

        for i, st in enumerate(grid_states):
            n1 = st.n_samples
            n2 = t - n1

            if n1 <= 2 or n2 <= 2:
                out[i] = 0.0
                continue

            best = -1.0e300
            has_valid = False
            for k in range(p):
                s_tot = state.sum[k]
                s1 = st.sum[k]
                s2 = s_tot - s1
                ss_tot = state.sum_sq[k]
                ss1 = st.sum_sq[k]
                ss2 = ss_tot - ss1

                sigma_tot = (ss_tot - s_tot * s_tot / t) / t
                sigma_1 = (ss1 - s1 * s1 / n1) / n1
                sigma_2 = (ss2 - s2 * s2 / n2) / n2
                if sigma_tot <= 0.0 or sigma_1 <= 0.0 or sigma_2 <= 0.0:
                    continue

                has_valid = True
                lr = t * np.log(sigma_tot) - n1 * np.log(sigma_1) - n2 * np.log(sigma_2)
                score = lr - 2.0
                if score > best:
                    best = score

            raw = best if has_valid else 0.0
            out[i] = raw / penalty

        return out
