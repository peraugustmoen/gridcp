"""Mean-change score with unknown variance.

Note: Supports both univariate and multivariate data. For multivariate data, the
unknown-variance LR score is computed per feature and the maximum feature-wise
score is used (matching the max-across-features behavior of ``MeanCUSUM``).
"""

from dataclasses import dataclass, field

import numba as nb
import numpy as np

from gridcp.typing import ArrayLike


@nb.njit(cache=True)
def mean_unknown_variance_score(
    total_stats: np.ndarray,
    before_stats: np.ndarray,
    total_samples: int,
    before_samples: np.ndarray,
) -> np.ndarray:
    """Compute LR scores for a change in mean with unknown variance.

    Parameters
    ----------
    total_stats : np.ndarray, shape (2, n_features)
        Running sufficient statistics ``[sum(x), sum(x^2)]``.
    before_stats : np.ndarray, shape (n_candidates, 2, n_features)
        Candidate-wise sufficient statistics.
    total_samples : int
        Total number of samples seen.
    before_samples : np.ndarray, shape (n_candidates,)
        Pre-change candidate sample counts.

    Returns
    -------
    np.ndarray, shape (n_candidates,)
        Penalization-free score (LR) per candidate. For multivariate data this is
        the maximum of per-feature LR scores.
    """
    n_candidates = before_samples.shape[0]
    out = np.zeros(n_candidates, dtype=np.float64)
    n_features = total_stats.shape[1]

    for i in range(n_candidates):
        n1 = before_samples[i]
        n2 = total_samples - n1

        # Keep small-sample behavior identical to old API implementation.
        if n1 <= 2 or n2 <= 2:
            out[i] = 0.0
            continue

        best_feature_score = -1.0e300
        has_valid_feature = False

        for k in range(n_features):
            total_sum = total_stats[0, k]
            total_sum2 = total_stats[1, k]
            sum1 = before_stats[i, 0, k]
            sum2_1 = before_stats[i, 1, k]

            sum2 = total_sum - sum1
            sum2_2 = total_sum2 - sum2_1

            sigma_null = (
                total_sum2 - total_sum * total_sum / total_samples
            ) / total_samples
            sigma_alt = (
                (sum2_1 - sum1 * sum1 / n1) + (sum2_2 - sum2 * sum2 / n2)
            ) / total_samples

            # Skip numerically invalid features for this candidate.
            if sigma_null <= 0.0 or sigma_alt <= 0.0:
                continue

            has_valid_feature = True
            feature_lr = total_samples * np.log(sigma_null) - total_samples * np.log(
                sigma_alt
            )
            feature_score = feature_lr - 1.0
            if feature_score > best_feature_score:
                best_feature_score = feature_score

        out[i] = best_feature_score if has_valid_feature else 0.0

    return out


@nb.njit(cache=True)
def mean_unknown_variance_penalty(n_samples: int, n_features: int) -> float:
    """Return the penalty divisor ``log(t p) + sqrt(log(t p))``."""
    logg = np.log(n_samples * n_features)
    return logg + np.sqrt(logg)


@dataclass(slots=True)
class MeanCUSUMUnknownVarianceState:
    """Running sufficient statistics for the MeanCUSUMUnknownVariance score.

    Parameters
    ----------
    n_samples : int
        Number of samples seen so far.
    stats : np.ndarray
        Running sufficient statistics ``[sum(x), sum(x²)]``.  Shape ``(2,)``
        for univariate series or ``(2, n_features)`` for multivariate series.
    """

    n_samples: int = 0
    stats: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))


@dataclass(frozen=True, slots=True)
class MeanCUSUMUnknownVariance:
    """Mean-change score with unknown common variance.

    Under no change, X_1, ..., X_t are i.i.d. N(μ, σ²I_p) with σ² unknown;
    under a change at τ, X_1, ..., X_{τ-1} ~ N(μ₁, σ²I_p) and
    X_τ, ..., X_t ~ N(μ₂, σ²I_p) with μ₁ ≠ μ₂ and common unknown variance σ².

    **Score.**  For a candidate with n_1 pre-change and n_2 = t - n_1
    post-change observations, the twice profile log-likelihood ratio is

        2 * LR = t * (log σ̂²_null - log σ̂²_alt),

    where σ̂²_null is the biased variance of the full window and σ̂²_alt is the
    pooled biased variance of the two segments (both estimated under the
    shared-variance model).  By Wilks' theorem, 2 * LR is asymptotically
    chi-squared(1) under the null.

    **Aggregation.**  For ``n_features > 1``, the above statistic is computed
    independently for each feature and the maximum is taken, matching the
    coordinate-max convention of :class:`MeanCUSUM`.

    **Centering and penalty.**  Each per-feature statistic is centered by
    subtracting 1 (the chi-squared(1) mean), and the maximum is taken after
    centering.  When ``enable_penalty=True`` (default), the centered score is
    divided by ``log(t p) + sqrt(log(t p))``, where p = ``n_features``; this is
    an asymptotic Wilks-style approximation.  When ``enable_penalty=False``, the
    divisor is 1.0 and the raw centered statistic is returned.

    **Sample size requirement.**  Candidates with n_1 ≤ 2 or n_2 ≤ 2 return
    0.0 to avoid degenerate variance estimates.

    Parameters
    ----------
    n_features : int, default=1
        Number of features expected in each observation.
    enable_penalty : bool, default=True
        If ``True``, divide the centered score by ``log(t p) + sqrt(log(t p))``.
        If ``False``, return the raw centered score with divisor 1.0.
    """

    n_features: int = 1
    enable_penalty: bool = True

    @property
    def n_tests(self) -> int:
        """Number of tests returned by ``compute_penalized_scores``."""
        return 1

    def init_state(self) -> MeanCUSUMUnknownVarianceState:
        """Return a fresh initial state with no observations seen."""
        if self.n_features < 1:
            raise ValueError("n_features must be >= 1.")

        if self.n_features == 1:
            stats = np.zeros(2, dtype=np.float64)
        else:
            stats = np.zeros((2, self.n_features), dtype=np.float64)

        return MeanCUSUMUnknownVarianceState(stats=stats)

    def update(
        self,
        state: MeanCUSUMUnknownVarianceState,
        x: ArrayLike,
    ) -> MeanCUSUMUnknownVarianceState:
        """Update the state with a new observation.

        Parameters
        ----------
        state : MeanCUSUMUnknownVarianceState
            Current state.
        x : ArrayLike
            New observation, shape ``(n_features,)``.

        Returns
        -------
        MeanCUSUMUnknownVarianceState
            Updated state.
        """
        x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
        if x_arr.size != self.n_features:
            raise ValueError(
                f"MeanCUSUMUnknownVariance expected observation of size {self.n_features}, got {x_arr.size}."
            )

        next_n_samples = state.n_samples + 1

        if self.n_features == 1:
            val = float(x_arr[0])
            increment = np.array([val, val * val], dtype=np.float64)
        else:
            increment = np.vstack((x_arr, x_arr * x_arr))

        next_stats = state.stats + increment
        return MeanCUSUMUnknownVarianceState(
            n_samples=next_n_samples,
            stats=next_stats,
        )

    def _compute_centered_scores(
        self,
        state: MeanCUSUMUnknownVarianceState,
        grid_states: list[MeanCUSUMUnknownVarianceState],
    ) -> np.ndarray:
        """Compute centered (but unpenalized) scores for every active grid candidate."""
        if len(grid_states) == 0:
            raise ValueError("grid_states is empty.")

        if self.n_features == 1:
            total_stats = state.stats.reshape(2, 1)
            grid_stats = np.stack([st.stats.reshape(2, 1) for st in grid_states])
        else:
            total_stats = state.stats
            grid_stats = np.stack([st.stats for st in grid_states])

        grid_n_samples = np.array([st.n_samples for st in grid_states], dtype=np.int64)
        return mean_unknown_variance_score(
            total_stats=total_stats,
            before_stats=grid_stats,
            total_samples=state.n_samples,
            before_samples=grid_n_samples,
        )

    def _get_penalty(self, n_samples: int) -> float:
        """Return the penalty divisor for the current sample size."""
        if self.enable_penalty:
            return mean_unknown_variance_penalty(n_samples, self.n_features)
        return 1.0

    def compute_penalized_scores(
        self,
        state: MeanCUSUMUnknownVarianceState,
        grid_states: list[MeanCUSUMUnknownVarianceState],
    ) -> np.ndarray:
        """Compute penalized LR score at every active grid candidate."""
        scores = self._compute_centered_scores(state, grid_states) / self._get_penalty(
            state.n_samples
        )
        return scores[:, np.newaxis]
