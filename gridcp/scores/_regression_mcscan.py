"""Regression McScan score."""

from dataclasses import dataclass, field

import numpy as np

from gridcp.typing import ArrayLike
from gridcp.scores._score_helpers import as_obs


@dataclass(slots=True)
class RegressionMcScanState:
    n_samples: int = 0
    yx_sum: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))


@dataclass(frozen=True, slots=True)
class RegressionMcScan:
    """Regression-change score from McScan (Cho, Kley, Li; 2025).

    The default penalty ``sqrt(log(t q))`` is the high-probability bound used
    for this max-type statistic, with ``q = n_regressors``. There is no
    chi-square centering term here.
    """

    n_regressors: int
    enable_penalty: bool = True

    @property
    def n_features(self) -> int:
        return self.n_regressors + 1

    @property
    def n_tests(self) -> int:
        """Number of tests returned by ``compute_penalized_scores``."""
        return 1

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

    def _compute_centered_scores(
        self,
        state: RegressionMcScanState,
        grid_states: list[RegressionMcScanState],
    ) -> np.ndarray:
        """Compute centered (but unpenalized) scores for every active grid candidate."""
        out = np.zeros(len(grid_states), dtype=np.float64)
        t = state.n_samples

        for i, st in enumerate(grid_states):
            n1 = st.n_samples
            n2 = t - n1

            if n1 == 0 or n2 == 0:
                out[i] = 0.0
                continue

            cov1 = st.yx_sum / n1
            cov2 = (state.yx_sum - st.yx_sum) / n2
            dist = np.max(np.abs(cov1 - cov2))
            out[i] = np.sqrt(n1 * n2 / t) * dist

        return out

    def _get_penalty(self, n_samples: int) -> float:
        """Return the penalty divisor for the current sample size."""
        if self.enable_penalty:
            return np.sqrt(np.log(self.n_regressors * n_samples))
        return 1.0

    def compute_penalized_scores(
        self,
        state: RegressionMcScanState,
        grid_states: list[RegressionMcScanState],
    ) -> np.ndarray:
        """Compute penalized McScan regression scores."""
        scores = self._compute_centered_scores(state, grid_states) / self._get_penalty(
            state.n_samples
        )
        return scores[:, np.newaxis]
