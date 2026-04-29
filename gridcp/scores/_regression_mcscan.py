"""Regression McScan score."""

from dataclasses import dataclass, field

import numba as nb
import numpy as np

from gridcp.typing import ArrayLike
from gridcp.scores._score_helpers import as_obs


@nb.njit(cache=True)
def regression_mcscan_score(
    total_yx_sum: np.ndarray,
    before_yx_sums: np.ndarray,
    total_samples: int,
    before_samples: np.ndarray,
) -> np.ndarray:
    """Compute McScan regression-change scores for all grid candidates.

    Parameters
    ----------
    total_yx_sum : np.ndarray
        Total cross-product sum ``Σ yᵢ xᵢ``, shape ``(n_regressors,)``.
    before_yx_sums : np.ndarray
        Per-candidate cross-product sums, shape ``(G, n_regressors)``.
    total_samples : int
        Total number of observations seen so far.
    before_samples : np.ndarray
        Number of observations before each candidate, shape ``(G,)``.

    Returns
    -------
    np.ndarray
        Centered scores for each candidate, shape ``(G,)``.
    """
    n_candidates = before_yx_sums.shape[0]
    n_regressors = before_yx_sums.shape[1]
    out = np.zeros(n_candidates, dtype=np.float64)
    t = total_samples

    for i in range(n_candidates):
        n1 = before_samples[i]
        n2 = t - n1

        if n1 == 0 or n2 == 0:
            out[i] = 0.0
            continue

        max_abs_diff = 0.0
        for j in range(n_regressors):
            cov1_j = before_yx_sums[i, j] / n1
            cov2_j = (total_yx_sum[j] - before_yx_sums[i, j]) / n2
            diff = cov1_j - cov2_j
            if diff < 0.0:
                diff = -diff
            if diff > max_abs_diff:
                max_abs_diff = diff

        out[i] = np.sqrt(n1 * n2 / t) * max_abs_diff

    return out


@dataclass(slots=True)
class RegressionMcScanState:
    """State for the RegressionMcScan score.

    Parameters
    ----------
    n_samples : int
        Number of samples seen so far.
    yx_sum : np.ndarray
        Cumulative cross-product sum ``Σ yᵢ xᵢ``, shape ``(n_regressors,)``.
    """

    n_samples: int = 0
    yx_sum: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))


@dataclass(frozen=True, slots=True)
class RegressionMcScan:
    """Regression-change CUSUM score following Cho, Kley, and Li (2025).

    Under no change, observations (yᵢ, xᵢ) follow the linear regression model
    y = βᵀx + ε with ε ~ N(0, 1) and regressors x ∈ ℝq with identity
    covariance and zero mean; under a change at τ, X_1, ..., X_{τ-1} follow
    the model with coefficient vector β₁ and X_τ, ..., X_t follow the model
    with β₂ ≠ β₁.  Each observation is passed as a vector of length q + 1
    with the response y in position 0 and the regressors x in positions
    1, ..., q.

    **Score.**  For a candidate with n_1 pre-change and n_2 = t - n_1
    post-change observations, the score is

        T = sqrt(n_1 n_2 / t) * max_{j=1,...,q} |ȳx_j^1 - ȳx_j^2|,

    where ȳx_j^k is the empirical cross-product mean of y with regressor j
    in segment k.  Under the null, ȳx_j^k estimates β_j (since x has identity
    covariance), so a change in any regression coefficient produces a
    detectable difference between segments.

    **Aggregation.**  The score is the max over q regressors of the
    per-regressor absolute CUSUM increment, analogous to the max-coordinate
    CUSUM of :class:`MeanCUSUM` but operating on cross-product statistics
    rather than the response mean.

    **Centering and penalty.**  This is a max-type statistic; no chi-squared
    centering constant is subtracted.  When ``enable_penalty=True`` (default),
    the score is divided by ``sqrt(log(q t))``, where q = ``n_regressors``;
    this is a high-probability bound for the max of q sub-Gaussian random
    variables (analogous to a Bonferroni correction for max-type scans).  When
    ``enable_penalty=False``, the divisor is 1.0 and the raw score is returned.

    **Sample size requirement.**  None.

    Parameters
    ----------
    n_regressors : int
        Number of regressors ``q`` (dimension of ``x`` in the model
        ``y = βᵀx + ε``).
    enable_penalty : bool, default=True
        If ``True``, divide the score by ``sqrt(log(q t))``.
        If ``False``, return the raw score with divisor 1.0.
    """

    n_regressors: int
    enable_penalty: bool = True

    @property
    def n_features(self) -> int:
        """Observation width expected by ``update`` (response + regressors)."""
        return self.n_regressors + 1

    @property
    def n_tests(self) -> int:
        """Number of tests returned by ``compute_penalized_scores``."""
        return 1

    def init_state(self) -> RegressionMcScanState:
        """Return a fresh initial state with no observations seen."""
        return RegressionMcScanState(
            yx_sum=np.zeros(self.n_regressors, dtype=np.float64)
        )

    def update(
        self,
        state: RegressionMcScanState,
        x: ArrayLike,
    ) -> RegressionMcScanState:
        """Update the state with a new observation.

        Parameters
        ----------
        state : RegressionMcScanState
            Current state.
        x : ArrayLike
            New observation of length ``n_regressors + 1``, with the response
            ``y`` in position 0 and regressors in positions 1, ...,
            ``n_regressors``.

        Returns
        -------
        RegressionMcScanState
            Updated state.
        """
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
        """Compute unpenalized McScan scores for every active grid candidate."""
        n = len(grid_states)
        before_yx_sums = np.empty((n, self.n_regressors), dtype=np.float64)
        before_samples = np.empty(n, dtype=np.int64)
        for i, st in enumerate(grid_states):
            before_yx_sums[i] = st.yx_sum
            before_samples[i] = st.n_samples
        return regression_mcscan_score(
            state.yx_sum,
            before_yx_sums,
            state.n_samples,
            before_samples,
        )

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
