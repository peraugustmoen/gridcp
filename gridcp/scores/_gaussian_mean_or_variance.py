"""Mean-or-variance LR score (Gaussian change in mean, variance, or both)."""

from dataclasses import dataclass, field

import numba as nb
import numpy as np

from gridcp.scores._aggregation import (
    aggregate_features,
    aggregation_dims,
    aggregation_mode_code,
    chi2_max_bound,
)
from gridcp.scores._score_helpers import as_obs
from gridcp.typing import ArrayLike

# Per-coordinate degrees of freedom of the mean-or-variance LR statistic.
_GAUSSIAN_MEAN_OR_VARIANCE_DF = 2


@nb.njit(cache=True)
def gaussian_mean_or_variance_score(
    total_sum: np.ndarray,
    total_sum_sq: np.ndarray,
    before_sums: np.ndarray,
    before_sum_sqs: np.ndarray,
    total_samples: int,
    before_samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean-or-variance LR for all grid candidates.

    Returns
    -------
    raw : np.ndarray, shape (G, n_features)
        Raw (uncentered, unaggregated) per-feature LR; degenerate features 0.
    valid : np.ndarray, shape (G,)
        ``False`` for candidates with ``n1 <= 2``/``n2 <= 2`` or no valid
        feature (these score 0 after centering).
    """
    n_candidates = before_sums.shape[0]
    n_features = before_sums.shape[1]
    raw = np.zeros((n_candidates, n_features), dtype=np.float64)
    valid = np.zeros(n_candidates, dtype=np.bool_)
    t = total_samples

    for i in range(n_candidates):
        n1 = before_samples[i]
        n2 = t - n1
        if n1 <= 2 or n2 <= 2:
            continue

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
            raw[i, k] = (
                t * np.log(sigma_tot) - n1 * np.log(sigma_1) - n2 * np.log(sigma_2)
            )

        valid[i] = has_valid

    return raw, valid


@dataclass(slots=True)
class GaussianMeanOrVarianceState:
    """State for the :class:`GaussianMeanOrVariance` score.

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
class GaussianMeanOrVariance:
    """Mean-or-variance LR score for a Gaussian change in mean, variance, or both.

    Under no change, X_1, ..., X_t are i.i.d. N(μ, σ²) with both parameters
    unknown; under a change at τ either or both parameters change.

    **Score.** For a candidate with n_1 pre-change and n_2 = t - n_1
    post-change observations, the per-feature twice log-likelihood ratio is
    ``2 * LR = t log σ̂²_tot - n_1 log σ̂²_1 - n_2 log σ̂²_2``, asymptotically
    chi-squared(2) under the null (mean and variance both free).

    **Aggregation.** The per-feature statistics are combined according to the
    ``aggregation`` keyword (``"max"`` by default; see Parameters).

    **Centering and penalty.** Each output column is centered by subtracting its
    degrees of freedom ``df`` inside the score. When ``enable_penalty=True``
    (default) the centered column is divided by ``chi2_max_bound(M, df, t)``;
    when ``enable_penalty=False`` the divisor is 1.0.

    **Sample size requirement.** Candidates with n_1 ≤ 2 or n_2 ≤ 2 return 0.0;
    features with a non-positive variance estimate are skipped.

    Parameters
    ----------
    n_features : int, default=1
        Number of features in the input data.
    aggregation : {"max", "sum", "max-sum", None, "None"}, default="max"
        How to combine the p per-feature statistics across the feature axis.
        ``"max"`` (1 column), ``"sum"`` (1 column), ``"max-sum"`` (2 columns,
        max then sum), ``None``/``"None"`` (p columns, one per feature).
    enable_penalty : bool, default=True
        If ``True``, divide each centered column by ``chi2_max_bound(M, df, t)``;
        if ``False``, return the raw centered statistic with divisor 1.0.
    """

    n_features: int = 1
    aggregation: object = "max"
    enable_penalty: bool = True

    def __post_init__(self):
        """Validate the ``aggregation`` keyword at construction time."""
        aggregation_mode_code(self.aggregation)

    @property
    def n_scores(self) -> int:
        """Number of scores returned by ``compute_penalized_scores``."""
        return len(
            aggregation_dims(
                self.aggregation, self.n_features, _GAUSSIAN_MEAN_OR_VARIANCE_DF
            )
        )

    def init_state(self) -> GaussianMeanOrVarianceState:
        """Return a fresh initial state with no observations seen."""
        return GaussianMeanOrVarianceState(
            sum=np.zeros(self.n_features, dtype=np.float64),
            sum_sq=np.zeros(self.n_features, dtype=np.float64),
        )

    def update(
        self, state: GaussianMeanOrVarianceState, x: ArrayLike
    ) -> GaussianMeanOrVarianceState:
        """Update the state with a new observation.

        Parameters
        ----------
        state : GaussianMeanOrVarianceState
            Current state.
        x : ArrayLike
            New observation, shape ``(n_features,)``.

        Returns
        -------
        GaussianMeanOrVarianceState
            Updated state.
        """
        x_arr = as_obs(x, self.n_features)
        return GaussianMeanOrVarianceState(
            n_samples=state.n_samples + 1,
            sum=state.sum + x_arr,
            sum_sq=state.sum_sq + x_arr * x_arr,
        )

    def compute_penalized_scores(
        self,
        state: GaussianMeanOrVarianceState,
        grid_states: list[GaussianMeanOrVarianceState],
    ) -> np.ndarray:
        """Compute penalized mean-or-variance scores for all active candidates."""
        if len(grid_states) == 0:
            raise ValueError("grid_states is empty.")

        before_sums = np.stack([st.sum for st in grid_states])
        before_sum_sqs = np.stack([st.sum_sq for st in grid_states])
        before_samples = np.array([st.n_samples for st in grid_states], dtype=np.int64)
        t = state.n_samples

        raw, valid = gaussian_mean_or_variance_score(
            total_sum=state.sum,
            total_sum_sq=state.sum_sq,
            before_sums=before_sums,
            before_sum_sqs=before_sum_sqs,
            total_samples=t,
            before_samples=before_samples,
        )
        cols = aggregate_features(raw, aggregation_mode_code(self.aggregation))
        dims = aggregation_dims(
            self.aggregation, self.n_features, _GAUSSIAN_MEAN_OR_VARIANCE_DF
        )

        out = np.empty((cols.shape[0], len(dims)), dtype=np.float64)
        for j, (n_terms, df) in enumerate(dims):
            penalty = chi2_max_bound(n_terms, df, t) if self.enable_penalty else 1.0
            out[:, j] = (cols[:, j] - df) / penalty

        out[~valid, :] = 0.0
        return out
