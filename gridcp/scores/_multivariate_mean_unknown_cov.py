"""Multivariate mean LR score with unknown covariance."""

from dataclasses import dataclass, field

import numba as nb
import numpy as np

from gridcp.typing import ArrayLike
from gridcp.scores._score_helpers import as_obs


@nb.njit(cache=True)
def multivariate_mean_unknown_cov_score(
    total_sum: np.ndarray,
    total_sum_outer: np.ndarray,
    before_sums: np.ndarray,
    before_sum_outers: np.ndarray,
    total_samples: int,
    before_samples: np.ndarray,
    n_features: int,
) -> np.ndarray:
    """Compute pooled-within-group LR scores for all grid candidates.

    Parameters
    ----------
    total_sum : np.ndarray
        Sum of all observations, shape ``(n_features,)``.
    total_sum_outer : np.ndarray
        Outer-product sum over all observations, shape ``(n_features, n_features)``.
    before_sums : np.ndarray
        Per-candidate sums, shape ``(G, n_features)``.
    before_sum_outers : np.ndarray
        Per-candidate outer-product sums, shape ``(G, n_features, n_features)``.
    total_samples : int
        Total number of observations seen so far.
    before_samples : np.ndarray
        Number of observations before each candidate, shape ``(G,)``.
    n_features : int
        Dimension ``p`` of each observation vector.

    Returns
    -------
    np.ndarray
        Centered LR scores for each candidate, shape ``(G,)``.  Each value is
        ``2 * LR - p`` where ``2 * LR = t * (logdet Σ̂_null - logdet Σ̂_alt)``.
        Returns the zero vector when ``t < 2*p + 2`` or when the null
        covariance matrix is singular.
    """
    n_candidates = before_sums.shape[0]
    out = np.zeros(n_candidates, dtype=np.float64)
    t = total_samples
    p = n_features
    df = float(p)

    if t < 2 * p + 2:
        return out

    s_tot = total_sum
    sxx_tot = total_sum_outer

    sigma_null = (sxx_tot - np.outer(s_tot, s_tot) / t) / t
    sign_null, logdet_null = np.linalg.slogdet(sigma_null)
    if sign_null <= 0:
        return out

    for i in range(n_candidates):
        n1 = before_samples[i]
        n2 = t - n1

        s1 = before_sums[i]
        sxx1 = before_sum_outers[i]
        s2 = s_tot - s1
        sxx2 = sxx_tot - sxx1

        sigma_alt = (
            (sxx1 - np.outer(s1, s1) / n1) + (sxx2 - np.outer(s2, s2) / n2)
        ) / t
        sign1, logdet1 = np.linalg.slogdet(sigma_alt)
        if sign1 <= 0:
            out[i] = 0.0
            continue

        two_lr = t * (logdet_null - logdet1)
        out[i] = two_lr - df

    return out


@dataclass(slots=True)
class MultivariateMeanUnknownCovState:
    """State for the MultivariateMeanUnknownCov score.

    Parameters
    ----------
    n_samples : int
        Number of samples seen so far.
    sum : np.ndarray
        Cumulative sum of observations, shape ``(n_features,)``.
    sum_outer : np.ndarray
        Cumulative outer-product sum, shape ``(n_features, n_features)``.
    """

    n_samples: int = 0
    sum: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    sum_outer: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )


@dataclass(frozen=True, slots=True)
class MultivariateMeanUnknownCov:
    """Multivariate mean-change LR score with unknown (pooled) covariance.

    Under no change, X_1, ..., X_t are i.i.d. N(μ, Σ) with Σ unknown; under a
    change at τ, X_1, ..., X_{τ-1} ~ N(μ₁, Σ) and X_τ, ..., X_t ~ N(μ₂, Σ)
    with a common unknown covariance Σ and μ₁ ≠ μ₂.

    **Score.**  For a candidate with n_1 pre-change and n_2 = t - n_1
    post-change observations, pooling within-group scatter from both segments
    gives the twice log-likelihood ratio

        2 * LR = t * (logdet Σ̂_null - logdet Σ̂_alt),

    where

        Σ̂_null = (Σᵢ xᵢxᵢᵀ - t x̄x̄ᵀ) / t   (full-window centered covariance)
        Σ̂_alt  = (S_W1 + S_W2) / t            (pooled within-group scatter / t)

    and S_Wk is the within-segment scatter matrix of segment k.  By Wilks'
    theorem, 2 * LR is asymptotically chi-squared(p) under the null.

    **Aggregation.**  The score produces a single test statistic
    (``n_scores = 1``).  No per-feature aggregation is performed; the LR
    operates jointly on the full p-dimensional distribution.

    **Centering and penalty.**  The statistic is centered by subtracting p (the
    chi-squared(p) mean).  When ``enable_penalty=True`` (default), the centered
    score is divided by ``sqrt(p log t) + log t``; this is an asymptotic
    Wilks-style approximation.  When ``enable_penalty=False``, the divisor is
    1.0 and the raw centered statistic is returned.

    **Sample size requirement.**  Because Σ̂_alt pools scatter from both segments,
    its rank is ``min(t - 2, p)`` regardless of how the split is made.  The
    strict minimum for invertibility is ``t >= p + 2``, but a more conservative
    threshold ``t >= 2*p + 2`` is used because near the strict minimum
    ``2 * LR`` deviates noticeably from its chi-squared limit.  The score
    returns ``0`` whenever ``t < 2*p + 2``.

    Parameters
    ----------
    n_features : int
        Dimension ``p`` of each observation vector.
    enable_penalty : bool, default=True
        If ``True``, divide the centered score by ``sqrt(p log t) + log t``.
        If ``False``, return the raw centered score with divisor 1.0.
    """

    n_features: int
    enable_penalty: bool = True

    @property
    def n_scores(self) -> int:
        """Number of scores returned by ``compute_penalized_scores``."""
        return 1

    def init_state(self) -> MultivariateMeanUnknownCovState:
        """Return a fresh initial state with no observations seen."""
        return MultivariateMeanUnknownCovState(
            sum=np.zeros(self.n_features, dtype=np.float64),
            sum_outer=np.zeros((self.n_features, self.n_features), dtype=np.float64),
        )

    def update(
        self,
        state: MultivariateMeanUnknownCovState,
        x: ArrayLike,
    ) -> MultivariateMeanUnknownCovState:
        """Update the state with a new observation.

        Parameters
        ----------
        state : MultivariateMeanUnknownCovState
            Current state.
        x : ArrayLike
            New observation, shape ``(n_features,)``.

        Returns
        -------
        MultivariateMeanUnknownCovState
            Updated state.
        """
        x_arr = as_obs(x, self.n_features)
        return MultivariateMeanUnknownCovState(
            n_samples=state.n_samples + 1,
            sum=state.sum + x_arr,
            sum_outer=state.sum_outer + np.outer(x_arr, x_arr),
        )

    def _compute_centered_scores(
        self,
        state: MultivariateMeanUnknownCovState,
        grid_states: list[MultivariateMeanUnknownCovState],
    ) -> np.ndarray:
        """Compute centered (but unpenalized) scores for every active grid candidate."""
        n = len(grid_states)
        p = self.n_features
        before_sums = np.empty((n, p), dtype=np.float64)
        before_sum_outers = np.empty((n, p, p), dtype=np.float64)
        before_samples = np.empty(n, dtype=np.int64)
        for i, st in enumerate(grid_states):
            before_sums[i] = st.sum
            before_sum_outers[i] = st.sum_outer
            before_samples[i] = st.n_samples
        return multivariate_mean_unknown_cov_score(
            state.sum,
            state.sum_outer,
            before_sums,
            before_sum_outers,
            state.n_samples,
            before_samples,
            self.n_features,
        )

    def _get_penalty(self, n_samples: int) -> float:
        """Return the penalty divisor for the current sample size."""
        if self.enable_penalty:
            df = float(self.n_features)
            return np.sqrt(df * np.log(n_samples)) + np.log(n_samples)
        return 1.0

    def compute_penalized_scores(
        self,
        state: MultivariateMeanUnknownCovState,
        grid_states: list[MultivariateMeanUnknownCovState],
    ) -> np.ndarray:
        """Compute penalized LR score at every active grid candidate."""
        scores = self._compute_centered_scores(state, grid_states) / self._get_penalty(
            state.n_samples
        )
        return scores[:, np.newaxis]
