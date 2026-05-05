"""Multivariate mean-or-covariance LR score."""

from dataclasses import dataclass, field

import numba as nb
import numpy as np

from gridcp.typing import ArrayLike
from gridcp.scores._score_helpers import as_obs


@nb.njit(cache=True)
def multivariate_mean_or_covariance_score(
    total_sum: np.ndarray,
    total_sum_outer: np.ndarray,
    before_sums: np.ndarray,
    before_sum_outers: np.ndarray,
    total_samples: int,
    before_samples: np.ndarray,
    n_features: int,
) -> np.ndarray:
    """Compute per-segment GLR scores for all grid candidates.

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
        ``2 * LR - df`` where ``df = p + p*(p+1)/2`` and
        ``2 * LR = t * logdet Σ̂_tot - n_1 * logdet Σ̂_1 - n_2 * logdet Σ̂_2``.
        Candidates where ``n_1 ≤ 2*p`` or ``n_2 ≤ 2*p`` receive a score of 0.
    """
    n_candidates = before_sums.shape[0]
    out = np.zeros(n_candidates, dtype=np.float64)
    t = total_samples
    p = n_features
    df = float(p + (p * (p + 1)) // 2)

    s_tot = total_sum
    sxx_tot = total_sum_outer

    sigma_tot = (sxx_tot - np.outer(s_tot, s_tot) / t) / t
    sign_tot, logdet_tot = np.linalg.slogdet(sigma_tot)
    if sign_tot <= 0:
        return out

    for i in range(n_candidates):
        n1 = before_samples[i]
        n2 = t - n1

        if n1 <= 2 * p or n2 <= 2 * p:
            out[i] = 0.0
            continue

        s1 = before_sums[i]
        sxx1 = before_sum_outers[i]
        s2 = s_tot - s1
        sxx2 = sxx_tot - sxx1

        sigma_1 = (sxx1 - np.outer(s1, s1) / n1) / n1
        sigma_2 = (sxx2 - np.outer(s2, s2) / n2) / n2

        sign1, logdet1 = np.linalg.slogdet(sigma_1)
        sign2, logdet2 = np.linalg.slogdet(sigma_2)
        if sign1 <= 0 or sign2 <= 0:
            out[i] = 0.0
            continue

        two_lr = t * logdet_tot - n1 * logdet1 - n2 * logdet2
        out[i] = two_lr - df

    return out


@dataclass(slots=True)
class MultivariateMeanOrCovarianceState:
    """State for the MultivariateMeanOrCovariance score.

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
class MultivariateMeanOrCovariance:
    """Multivariate mean-or-covariance LR score.

    Under no change, X_1, ..., X_t are i.i.d. N(μ, Σ) with both parameters
    unknown; under a change at τ, X_1, ..., X_{τ-1} ~ N(μ₁, Σ₁) and
    X_τ, ..., X_t ~ N(μ₂, Σ₂), where the change may be in the mean vector,
    covariance matrix, or both.

    **Score.**  For a candidate with n_1 pre-change and n_2 = t - n_1
    post-change observations, the twice log-likelihood ratio is

        2 * LR = t * logdet Σ̂_tot - n_1 * logdet Σ̂_1 - n_2 * logdet Σ̂_2,

    where Σ̂_tot, Σ̂_1, and Σ̂_2 are the biased sample covariance matrices of
    the full window, left segment, and right segment, respectively.  By
    Wilks' theorem, 2 * LR is asymptotically chi-squared with
    ``df = p + p*(p+1)/2`` degrees of freedom under the null (p for the mean
    and p*(p+1)/2 for the upper triangle of the covariance).

    **Aggregation.**  The score produces a single test statistic
    (``n_scores = 1``).  No per-feature aggregation is performed; the LR
    operates jointly on the full p-dimensional distribution.

    **Centering and penalty.**  The statistic is centered by subtracting df
    (the chi-squared(df) mean).  When ``enable_penalty=True`` (default), the
    centered score is divided by ``sqrt(df log t) + log t``; this is an
    asymptotic Wilks-style approximation.  When ``enable_penalty=False``, the
    divisor is 1.0 and the raw centered statistic is returned.

    **Sample size requirement.**  Each segment's centered sample covariance has
    rank at most ``min(n - 1, p)``; unlike :class:`MultivariateMeanUnknownCov`
    the scatter cannot be pooled, so the constraint applies to each side of
    the split independently.  The strict invertibility minimum is ``n > p``,
    but a more conservative threshold ``n > 2*p`` is used because near the
    strict minimum ``2 * LR`` deviates substantially from its chi-squared
    limit.  The score returns 0 for any candidate that does not meet this
    condition.

    Parameters
    ----------
    n_features : int
        Dimension ``p`` of each observation vector.
    enable_penalty : bool, default=True
        If ``True``, divide the centered score by ``sqrt(df log t) + log t``
        with ``df = p + p*(p+1)/2``.  If ``False``, return the raw centered
        score with divisor 1.0.
    """

    n_features: int
    enable_penalty: bool = True

    @property
    def n_scores(self) -> int:
        """Number of scores returned by ``compute_penalized_scores``."""
        return 1

    def init_state(self) -> MultivariateMeanOrCovarianceState:
        """Return a fresh initial state with no observations seen."""
        return MultivariateMeanOrCovarianceState(
            sum=np.zeros(self.n_features, dtype=np.float64),
            sum_outer=np.zeros((self.n_features, self.n_features), dtype=np.float64),
        )

    def update(
        self,
        state: MultivariateMeanOrCovarianceState,
        x: ArrayLike,
    ) -> MultivariateMeanOrCovarianceState:
        """Update the state with a new observation.

        Parameters
        ----------
        state : MultivariateMeanOrCovarianceState
            Current state.
        x : ArrayLike
            New observation, shape ``(n_features,)``.

        Returns
        -------
        MultivariateMeanOrCovarianceState
            Updated state.
        """
        x_arr = as_obs(x, self.n_features)
        return MultivariateMeanOrCovarianceState(
            n_samples=state.n_samples + 1,
            sum=state.sum + x_arr,
            sum_outer=state.sum_outer + np.outer(x_arr, x_arr),
        )

    def _compute_centered_scores(
        self,
        state: MultivariateMeanOrCovarianceState,
        grid_states: list[MultivariateMeanOrCovarianceState],
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
        return multivariate_mean_or_covariance_score(
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
            p = self.n_features
            df = float(p + (p * (p + 1)) // 2)
            return np.sqrt(df * np.log(n_samples)) + np.log(n_samples)
        return 1.0

    def compute_penalized_scores(
        self,
        state: MultivariateMeanOrCovarianceState,
        grid_states: list[MultivariateMeanOrCovarianceState],
    ) -> np.ndarray:
        """Compute penalized multivariate mean-or-covariance scores."""
        scores = self._compute_centered_scores(state, grid_states) / self._get_penalty(
            state.n_samples
        )
        return scores[:, np.newaxis]
