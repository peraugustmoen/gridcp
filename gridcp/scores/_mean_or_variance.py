"""Mean-or-variance LR score."""

from dataclasses import dataclass, field

import numba as nb
import numpy as np

from gridcp.typing import ArrayLike
from gridcp.scores._score_helpers import as_obs


@nb.njit(cache=True)
def mean_or_variance_score(
    total_sum: np.ndarray,
    total_sum_sq: np.ndarray,
    before_sums: np.ndarray,
    before_sum_sqs: np.ndarray,
    total_samples: int,
    before_samples: np.ndarray,
) -> np.ndarray:
    """Compute mean-or-variance LR scores for all grid candidates.

    Parameters
    ----------
    total_sum : np.ndarray
        Sum of all observations, shape ``(n_features,)``.
    total_sum_sq : np.ndarray
        Sum of squares of all observations, shape ``(n_features,)``.
    before_sums : np.ndarray
        Per-candidate sums, shape ``(G, n_features)``.
    before_sum_sqs : np.ndarray
        Per-candidate sum-of-squares, shape ``(G, n_features)``.
    total_samples : int
        Total number of observations seen so far.
    before_samples : np.ndarray
        Number of observations before each candidate, shape ``(G,)``.

    Returns
    -------
    np.ndarray
        Centered LR scores for each candidate, shape ``(G,)``.
    """
    n_candidates = before_sums.shape[0]
    n_features = before_sums.shape[1]
    out = np.zeros(n_candidates, dtype=np.float64)
    t = total_samples

    for i in range(n_candidates):
        n1 = before_samples[i]
        n2 = t - n1

        if n1 <= 2 or n2 <= 2:
            out[i] = 0.0
            continue

        best = -1.0e300
        has_valid = False
        for k in range(n_features):
            s_tot = total_sum[k]
            s1 = before_sums[i, k]
            s2 = s_tot - s1
            ss_tot = total_sum_sq[k]
            ss1 = before_sum_sqs[i, k]
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

        out[i] = best if has_valid else 0.0

    return out


@dataclass(slots=True)
class MeanOrVarianceState:
    """State for the MeanOrVariance score.

    Parameters
    ----------
    n_samples : int
        Number of samples seen so far.
    sum : np.ndarray
        Cumulative sum of observations, shape ``(n_features,)``.
    sum_sq : np.ndarray
        Cumulative sum of squared observations, shape ``(n_features,)``.
    """

    n_samples: int = 0
    sum: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    sum_sq: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))


@dataclass(frozen=True, slots=True)
class MeanOrVariance:
    """Mean-or-variance LR score for a Gaussian change in mean, variance, or both.

    Under no change, X_1, ..., X_t are i.i.d. N(μ, σ²) with both parameters
    unknown; under a change at τ, X_1, ..., X_{τ-1} ~ N(μ₁, σ₁²) and
    X_τ, ..., X_t ~ N(μ₂, σ₂²), where the change may be in the mean, variance,
    or both.

    **Score.**  For a candidate with n_1 pre-change and n_2 = t - n_1
    post-change observations, the twice log-likelihood ratio is

        2 * LR = t * log σ̂²_tot - n_1 * log σ̂²_1 - n_2 * log σ̂²_2,

    where σ̂²_k is the (biased) sample variance of segment k.  By Wilks' theorem,
    2 * LR is asymptotically chi-squared(2) under the null (two degrees of
    freedom: one for mean, one for variance).

    **Aggregation.**  For ``n_features > 1``, scores are computed per feature
    and the maximum is taken.

    **Centering and penalty.**  The statistic is centered by subtracting 2 (the
    chi-squared(2) mean).  When ``enable_penalty=True`` (default), the centered
    score is divided by ``log(t p) + sqrt(2 log(t p))``, where p = ``n_features``;
    this is an asymptotic Wilks-style approximation for a chi-squared(2)
    statistic.  When ``enable_penalty=False``, the divisor is 1.0 and the raw
    centered statistic is returned.

    **Sample size requirement.**  Candidates with n_1 ≤ 2 or n_2 ≤ 2 return
    0.0 to avoid degenerate per-segment variance estimates.

    Parameters
    ----------
    n_features : int, default=1
        Number of features in the input data.
    enable_penalty : bool, default=True
        If ``True``, divide the centered score by ``log(t p) + sqrt(2 log(t p))``.
        If ``False``, return the raw centered score with divisor 1.0.
    """

    n_features: int = 1
    enable_penalty: bool = True

    @property
    def n_tests(self) -> int:
        """Number of tests returned by ``compute_penalized_scores``."""
        return 1

    def init_state(self) -> MeanOrVarianceState:
        """Return a fresh initial state with no observations seen."""
        return MeanOrVarianceState(
            sum=np.zeros(self.n_features, dtype=np.float64),
            sum_sq=np.zeros(self.n_features, dtype=np.float64),
        )

    def update(self, state: MeanOrVarianceState, x: ArrayLike) -> MeanOrVarianceState:
        """Update the state with a new observation.

        Parameters
        ----------
        state : MeanOrVarianceState
            Current state.
        x : ArrayLike
            New observation, shape ``(n_features,)``.

        Returns
        -------
        MeanOrVarianceState
            Updated state.
        """
        x_arr = as_obs(x, self.n_features)
        return MeanOrVarianceState(
            n_samples=state.n_samples + 1,
            sum=state.sum + x_arr,
            sum_sq=state.sum_sq + x_arr * x_arr,
        )

    def _compute_centered_scores(
        self,
        state: MeanOrVarianceState,
        grid_states: list[MeanOrVarianceState],
    ) -> np.ndarray:
        """Compute centered (but unpenalized) scores for every active grid candidate."""
        n = len(grid_states)
        before_sums = np.empty((n, self.n_features), dtype=np.float64)
        before_sum_sqs = np.empty((n, self.n_features), dtype=np.float64)
        before_samples = np.empty(n, dtype=np.int64)
        for i, st in enumerate(grid_states):
            before_sums[i] = st.sum
            before_sum_sqs[i] = st.sum_sq
            before_samples[i] = st.n_samples
        return mean_or_variance_score(
            state.sum,
            state.sum_sq,
            before_sums,
            before_sum_sqs,
            state.n_samples,
            before_samples,
        )

    def _get_penalty(self, n_samples: int) -> float:
        """Return the penalty divisor for the current sample size."""
        if self.enable_penalty:
            logg = np.log(n_samples * self.n_features)
            return logg + np.sqrt(2.0 * logg)
        return 1.0

    def compute_penalized_scores(
        self,
        state: MeanOrVarianceState,
        grid_states: list[MeanOrVarianceState],
    ) -> np.ndarray:
        """Compute penalized mean-or-variance scores for all active candidates."""
        scores = self._compute_centered_scores(state, grid_states) / self._get_penalty(
            state.n_samples
        )
        return scores[:, np.newaxis]
