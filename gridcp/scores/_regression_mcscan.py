"""Regression McScan score."""

from dataclasses import dataclass

import numpy as np

from gridcp.typing import ArrayLike
from gridcp.scores._score_helpers import as_obs


@dataclass(slots=True)
class RegressionMcScanState:
    n_samples: int = 0
    yx_sum: np.ndarray = None


@dataclass(frozen=True, slots=True)
class RegressionMcScan:
    """Regression-change score from McScan (Cho, Kley, Li; 2025)."""

    n_regressors: int

    @property
    def n_features(self) -> int:
        return self.n_regressors + 1

    def init_state(self) -> RegressionMcScanState:
        return RegressionMcScanState(
            yx_sum=np.zeros(self.n_regressors, dtype=np.float64)
        )

    def update(
        self,
        state: RegressionMcScanState,
        x: ArrayLike,
    ) -> RegressionMcScanState:
        obs = as_obs(x, self.n_features)
        y = float(obs[0])
        xx = obs[1:]
        return RegressionMcScanState(
            n_samples=state.n_samples + 1,
            yx_sum=state.yx_sum + y * xx,
        )

    def compute_penalised_scores(
        self,
        state: RegressionMcScanState,
        grid_states: list[RegressionMcScanState],
    ) -> np.ndarray:
        out = np.zeros(len(grid_states), dtype=np.float64)
        t = state.n_samples
        p = self.n_features
        penalty = np.sqrt(np.log(p * t))

        for i, st in enumerate(grid_states):
            n1 = st.n_samples
            n2 = t - n1
            cov1 = st.yx_sum / n1
            cov2 = (state.yx_sum - st.yx_sum) / n2
            dist = np.max(np.abs(cov1 - cov2))
            raw = np.sqrt(n1 * n2 / t) * dist
            out[i] = raw / penalty

        return out
