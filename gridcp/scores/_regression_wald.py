"""Two-sample (Chow) Wald regression-change score."""

from dataclasses import dataclass, field

import numba as nb
import numpy as np

from gridcp.scores._aggregation import chi2_max_bound
from gridcp.scores._score_helpers import as_obs
from gridcp.typing import ArrayLike


@nb.njit(cache=True)
def _inv_pd_nb(matrix: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Inverse of a symmetric PD matrix with eigenvalue clipping for safety."""
    eigvals, eigvecs = np.linalg.eigh(matrix)
    q = eigvals.shape[0]
    inv_eigvals = np.empty(q, dtype=np.float64)
    for i in range(q):
        v = eigvals[i]
        if v < eps:
            v = eps
        inv_eigvals[i] = 1.0 / v
    return (eigvecs * inv_eigvals) @ eigvecs.T


@nb.njit(cache=True)
def regression_wald_score(
    total_yx_sum: np.ndarray,
    total_xx_sum: np.ndarray,
    before_yx_sums: np.ndarray,
    before_xx_sums: np.ndarray,
    total_samples: int,
    before_samples: np.ndarray,
    n_regressors: int,
) -> np.ndarray:
    """Compute two-sample Wald regression-change scores for all candidates.

    For each candidate the statistic is the standard two-sample (Chow) Wald
    contrast for a change in regression coefficients,

        D̃_g = (β̂₁ - β̂₂)ᵀ (M₁⁻¹ + M₂⁻¹)⁻¹ (β̂₁ - β̂₂) / σ²,

    with σ² = 1 (known unit noise variance), ``β̂_k = M_k⁻¹ S_k``,
    ``M_k = Σ xᵢxᵢᵀ`` and ``S_k = Σ xᵢ yᵢ`` over segment ``k``. Under the null
    this is exactly chi-squared(q), independent of the true coefficients.

    Parameters
    ----------
    total_yx_sum : np.ndarray
        Total cross-product sum ``S = Σ x y``, shape ``(q,)``.
    total_xx_sum : np.ndarray
        Total Gram matrix ``M = Σ x xᵀ``, shape ``(q, q)``.
    before_yx_sums : np.ndarray
        Per-candidate cross-product sums, shape ``(G, q)``.
    before_xx_sums : np.ndarray
        Per-candidate Gram matrices, shape ``(G, q, q)``.
    total_samples : int
        Total number of observations.
    before_samples : np.ndarray
        First post-change index (0-based) for each candidate, shape ``(G,)``.
    n_regressors : int
        Number of regressors ``q``.

    Returns
    -------
    np.ndarray
        Centered scores ``D̃_g - q`` for each candidate, shape ``(G,)``.
        Candidates with ``n1 < q`` or ``n2 < q`` return 0.
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

        s_pre = before_yx_sums[i]
        s_post = total_yx_sum - s_pre
        m_pre = before_xx_sums[i]
        m_post = total_xx_sum - m_pre

        m1_inv = _inv_pd_nb(m_pre)
        m2_inv = _inv_pd_nb(m_post)

        beta1 = m1_inv @ s_pre
        beta2 = m2_inv @ s_post
        diff = beta1 - beta2

        middle = _inv_pd_nb(m1_inv + m2_inv)
        wald = diff @ (middle @ diff)
        out[i] = wald - q

    return out


@dataclass(slots=True)
class RegressionWaldState:
    """State for the :class:`RegressionWald` score.

    Parameters
    ----------
    n_samples : int
        Number of samples seen so far.
    yx_sum : np.ndarray
        Cumulative cross-product sum ``S = Σ yᵢ xᵢ``, shape ``(n_regressors,)``.
    xx_sum : np.ndarray
        Cumulative uncentered Gram matrix ``M = Σ xᵢ xᵢᵀ``, shape
        ``(n_regressors, n_regressors)``.
    """

    n_samples: int = 0
    yx_sum: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    xx_sum: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )


@dataclass(frozen=True, slots=True)
class RegressionWald:
    """Two-sample (Chow) Wald score for a change in regression coefficients.

    Under no change, (yᵢ, xᵢ) follow the linear model y = βᵀx + ε with
    ε ~ N(0, 1) and regressors x ∈ ℝq; under a change at τ the coefficient
    vector changes from β₁ to β₂ ≠ β₁. Each observation is passed as a vector
    of length q + 1 with the response y in position 0 and the regressors x in
    positions 1, ..., q.

    **Score.** For a candidate with n_1 pre-change and n_2 = t - n_1
    post-change observations, the score is the standard two-sample (Chow) Wald
    contrast

        D̃_g = (β̂₁ - β̂₂)ᵀ (M₁⁻¹ + M₂⁻¹)⁻¹ (β̂₁ - β̂₂) / σ²,

    with σ² = 1 (known unit noise variance), ``β̂_k = M_k⁻¹ S_k``,
    ``M_k = Σ xᵢxᵢᵀ`` and ``S_k = Σ xᵢ yᵢ``. Under the null this has an exact
    chi-squared(q) distribution that does not depend on the regression
    coefficients.

    **Aggregation.** This score is self-aggregating: it pools all q regressors
    into one joint Wald statistic (``n_scores = 1``) and does not take an
    ``aggregation`` keyword.

    **Centering and penalty.** The statistic is centered by subtracting q (the
    chi-squared(q) mean). When ``enable_penalty=True`` (default), the centered
    score is divided by ``chi2_max_bound(1, q, t) = sqrt(q log t) + log t``; when
    ``enable_penalty=False`` the divisor is 1.0.

    **Sample size requirement.** The uncentered Gram matrix M_k has full rank at
    ``n_k = q``. The score returns 0 whenever ``n_1 < q`` or ``n_2 < q``;
    eigenvalues are clipped to ``1e-15`` for numerical safety at the boundary.

    Parameters
    ----------
    n_regressors : int
        Number of regressors ``q`` in the model ``y = βᵀx + ε``.
    enable_penalty : bool, default=True
        If ``True``, divide the centered score by ``chi2_max_bound(1, q, t)``.
        If ``False``, return the raw centered score with divisor 1.0.
    """

    n_regressors: int
    enable_penalty: bool = True

    @property
    def n_features(self) -> int:
        """Observation length expected by the score (response + regressors)."""
        return self.n_regressors + 1

    @property
    def n_scores(self) -> int:
        """Number of scores returned by ``compute_penalized_scores``."""
        return 1

    def init_state(self) -> RegressionWaldState:
        """Return a fresh initial state with no observations seen."""
        q = self.n_regressors
        return RegressionWaldState(
            yx_sum=np.zeros(q, dtype=np.float64),
            xx_sum=np.zeros((q, q), dtype=np.float64),
        )

    def update(
        self,
        state: RegressionWaldState,
        x: ArrayLike,
    ) -> RegressionWaldState:
        """Update the state with a new observation.

        Parameters
        ----------
        state : RegressionWaldState
            Current state.
        x : ArrayLike
            New observation of length ``n_regressors + 1``, with the response
            ``y`` in position 0 and regressors in positions 1, ...,
            ``n_regressors``.

        Returns
        -------
        RegressionWaldState
            Updated state.
        """
        obs = as_obs(x, self.n_features)
        y = float(obs[0])
        xx = obs[1:]
        return RegressionWaldState(
            n_samples=state.n_samples + 1,
            yx_sum=state.yx_sum + y * xx,
            xx_sum=state.xx_sum + np.outer(xx, xx),
        )

    def _get_penalty(self, n_samples: int) -> float:
        """Return the penalty divisor for the current sample size."""
        if self.enable_penalty:
            return chi2_max_bound(1, self.n_regressors, n_samples)
        return 1.0

    def compute_penalized_scores(
        self,
        state: RegressionWaldState,
        grid_states: list[RegressionWaldState],
    ) -> np.ndarray:
        """Compute penalized two-sample Wald scores for all active candidates."""
        n = len(grid_states)
        q = self.n_regressors
        before_yx_sums = np.empty((n, q), dtype=np.float64)
        before_xx_sums = np.empty((n, q, q), dtype=np.float64)
        before_samples = np.empty(n, dtype=np.int64)
        for i, st in enumerate(grid_states):
            before_yx_sums[i] = st.yx_sum
            before_xx_sums[i] = st.xx_sum
            before_samples[i] = st.n_samples
        scores = regression_wald_score(
            state.yx_sum,
            state.xx_sum,
            before_yx_sums,
            before_xx_sums,
            state.n_samples,
            before_samples,
            q,
        )
        return (scores / self._get_penalty(state.n_samples))[:, np.newaxis]
