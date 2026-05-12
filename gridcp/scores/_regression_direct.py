"""Regression direct score."""

from dataclasses import dataclass, field

import numba as nb
import numpy as np

from gridcp.typing import ArrayLike
from gridcp.scores._score_helpers import as_obs, inv_sqrtm_pd_nb


@nb.njit(cache=True)
def regression_direct_score(
    total_yx_sum: np.ndarray,
    total_xx_sum: np.ndarray,
    before_yx_sums: np.ndarray,
    before_xx_sums: np.ndarray,
    total_samples: int,
    before_samples: np.ndarray,
    n_regressors: int,
) -> np.ndarray:
    """Compute direct regression-change scores for all grid candidates.

    Parameters
    ----------
    total_yx_sum : np.ndarray
        Total cross-product sum, shape ``(q,)``.
    total_xx_sum : np.ndarray
        Total Gram matrix, shape ``(q, q)``.
    before_yx_sums : np.ndarray
        Per-candidate cross-product sums, shape ``(G, q)``.
    before_xx_sums : np.ndarray
        Per-candidate Gram matrices, shape ``(G, q, q)``.
    total_samples : int
        Total number of observations.
    before_samples : np.ndarray
        First post-change index (0-based) for each candidate, shape ``(G,)``.
        Equals the pre-change sample count: ``data[0:n1]`` is pre-change.
    n_regressors : int
        Number of regressors ``q``.

    Returns
    -------
    np.ndarray
        Centered scores for each candidate, shape ``(G,)``.
    """
    n_candidates = before_yx_sums.shape[0]
    out = np.zeros(n_candidates, dtype=np.float64)
    t = total_samples
    q = n_regressors

    for i in range(n_candidates):
        n1 = before_samples[i]
        n2 = t - n1

        if n1 < q or n2 < q:
            out[i] = 0.0
            continue

        yx_pre = before_yx_sums[i]
        yx_post = total_yx_sum - yx_pre
        xx_pre = before_xx_sums[i]
        xx_post = total_xx_sum - xx_pre

        m1_inv = inv_sqrtm_pd_nb(xx_pre)
        m2_inv = inv_sqrtm_pd_nb(xx_post)
        diff = m1_inv @ yx_pre - m2_inv @ yx_post
        out[i] = 0.5 * np.dot(diff, diff) - q

    return out


@dataclass(slots=True)
class RegressionDirectState:
    """State for the RegressionDirect score.

    Parameters
    ----------
    n_samples : int
        Number of samples seen so far.
    yx_sum : np.ndarray
        Cumulative cross-product sum ``Σ yᵢ xᵢ``, shape ``(n_regressors,)``.
    xx_sum : np.ndarray
        Cumulative uncentered Gram matrix ``Σ xᵢ xᵢᵀ``, shape
        ``(n_regressors, n_regressors)``.
    """

    n_samples: int = 0
    yx_sum: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    xx_sum: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )


@dataclass(frozen=True, slots=True)
class RegressionDirect:
    """Direct regression-change score using covariance-normalized differences.

    Under no change, (yᵢ, xᵢ) follow the linear model y = βᵀx + ε with
    ε ~ N(0, 1) and fixed regressors x ∈ ℝq; under a change at τ,
    X_1, ..., X_{τ-1} follow the model with coefficient vector β₁ and
    X_τ, ..., X_t follow the model with β₂ ≠ β₁.  Each observation is passed
    as a vector of length q + 1 with the response y in position 0 and the
    regressors x in positions 1, ..., q.

    **Score.**  For a candidate with n_1 pre-change and n_2 = t - n_1
    post-change observations, the score is

        S = (1/2) ‖X_1^{-1/2} Σyx_1 - X_2^{-1/2} Σyx_2‖² - q,

    where X_k is the uncentered Gram matrix of the regressors in segment k,
    X_k^{-1/2} is its inverse square root, and Σyx_k is the cross-product
    vector of y and x in segment k.

    **Aggregation.**  The score produces a single test statistic
    (``n_scores = 1``).  No per-regressor aggregation is performed; the score
    pools all q regressors into one Wald-type statistic.

    **Centering and penalty.**  The statistic is centered by subtracting q
    (the chi-squared(q) mean).  When ``enable_penalty=True`` (default), the
    centered score is divided by ``sqrt(q log t) + log t``; this penalty is
    derived from a non-asymptotic concentration bound, not an asymptotic
    chi-squared approximation.  When ``enable_penalty=False``, the divisor is
    1.0 and the raw centered statistic is returned.

    **Sample size requirement.**  The uncentered Gram matrix X_k has rank
    ``min(n_k, q)``; full rank is achieved at exactly ``n_k = q`` (one fewer
    observation than a centered covariance requires).  The score returns 0
    whenever ``n_1 < q`` or ``n_2 < q``.  :func:`inv_sqrtm_pd_nb`
    additionally clips small eigenvalues to ``1e-15`` for numerical safety at
    the boundary. Unlike the Gaussian covariance scores, this penalty is
    derived from a non-asymptotic concentration bound, so the strict sample
    size requirement is also an appropriate practical guard.

    Parameters
    ----------
    n_regressors : int
        Number of regressors ``q`` (dimension of ``x`` in the model
        ``y = βᵀx + ε``).
    enable_penalty : bool, default=True
        If ``True``, divide the centered score by ``sqrt(q log t) + log t``.
        If ``False``, return the raw centered score with divisor 1.0.
    """

    n_regressors: int
    enable_penalty: bool = True

    @property
    def n_features(self) -> int:
        return self.n_regressors + 1

    @property
    def n_scores(self) -> int:
        """Number of scores returned by ``compute_penalized_scores``."""
        return 1

    def init_state(self) -> RegressionDirectState:
        """Return a fresh initial state with no observations seen."""
        q = self.n_regressors
        return RegressionDirectState(
            yx_sum=np.zeros(q, dtype=np.float64),
            xx_sum=np.zeros((q, q), dtype=np.float64),
        )

    def update(
        self,
        state: RegressionDirectState,
        x: ArrayLike,
    ) -> RegressionDirectState:
        """Update the state with a new observation.

        Parameters
        ----------
        state : RegressionDirectState
            Current state.
        x : ArrayLike
            New observation of length ``n_regressors + 1``, with the response
            ``y`` in position 0 and regressors in positions 1, ...,
            ``n_regressors``.

        Returns
        -------
        RegressionDirectState
            Updated state.
        """
        obs = as_obs(x, self.n_features)
        y = float(obs[0])
        xx = obs[1:]

        return RegressionDirectState(
            n_samples=state.n_samples + 1,
            yx_sum=state.yx_sum + y * xx,
            xx_sum=state.xx_sum + np.outer(xx, xx),
        )

    def _compute_centered_scores(
        self,
        state: RegressionDirectState,
        grid_states: list[RegressionDirectState],
    ) -> np.ndarray:
        """Compute centered (but unpenalized) scores for every active grid candidate."""
        n = len(grid_states)
        q = self.n_regressors
        before_yx_sums = np.empty((n, q), dtype=np.float64)
        before_xx_sums = np.empty((n, q, q), dtype=np.float64)
        before_samples = np.empty(n, dtype=np.int64)
        for i, st in enumerate(grid_states):
            before_yx_sums[i] = st.yx_sum
            before_xx_sums[i] = st.xx_sum
            before_samples[i] = st.n_samples
        return regression_direct_score(
            state.yx_sum,
            state.xx_sum,
            before_yx_sums,
            before_xx_sums,
            state.n_samples,
            before_samples,
            q,
        )

    def _get_penalty(self, n_samples: int) -> float:
        """Return the penalty divisor for the current sample size."""
        if self.enable_penalty:
            p = self.n_features
            return np.sqrt((p - 1) * np.log(n_samples)) + np.log(n_samples)
        return 1.0

    def compute_penalized_scores(
        self,
        state: RegressionDirectState,
        grid_states: list[RegressionDirectState],
    ) -> np.ndarray:
        """Compute penalized direct-regression scores."""
        scores = self._compute_centered_scores(state, grid_states) / self._get_penalty(
            state.n_samples
        )
        return scores[:, np.newaxis]
