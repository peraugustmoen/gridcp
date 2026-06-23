"""Mean-change score with unknown common variance (Gaussian LR)."""

from dataclasses import dataclass, field

import numba as nb
import numpy as np

from gridcp.scores._aggregation import (
    aggregate_features,
    aggregation_dims,
    aggregation_mode_code,
    chi2_max_bound,
)
from gridcp.typing import ArrayLike

# Per-coordinate degrees of freedom of the mean-change LR statistic.
_GAUSSIAN_MEAN_DF = 1


@nb.njit(cache=True)
def gaussian_mean_score(
    total_stats: np.ndarray,
    before_stats: np.ndarray,
    total_samples: int,
    before_samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean-change LR (unknown variance) for all grid candidates.

    Parameters
    ----------
    total_stats : np.ndarray, shape (2, n_features)
        Running sufficient statistics ``[sum(x), sum(x^2)]``.
    before_stats : np.ndarray, shape (n_candidates, 2, n_features)
        Candidate-wise sufficient statistics.
    total_samples : int
        Total number of samples seen.
    before_samples : np.ndarray, shape (n_candidates,)
        First post-change index (0-based) for each candidate.

    Returns
    -------
    raw : np.ndarray, shape (n_candidates, n_features)
        Raw (uncentered, unaggregated) per-feature LR.  Features with a
        degenerate variance estimate are set to 0.
    valid : np.ndarray, shape (n_candidates,)
        Boolean flag; ``False`` for candidates with ``n1 <= 2``/``n2 <= 2`` or
        no valid feature (these score 0 after centering).
    """
    n_candidates = before_samples.shape[0]
    n_features = total_stats.shape[1]
    raw = np.zeros((n_candidates, n_features), dtype=np.float64)
    valid = np.zeros(n_candidates, dtype=np.bool_)

    for i in range(n_candidates):
        n1 = before_samples[i]
        n2 = total_samples - n1
        if n1 <= 2 or n2 <= 2:
            continue

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

            if sigma_null <= 0.0 or sigma_alt <= 0.0:
                continue

            has_valid_feature = True
            raw[i, k] = total_samples * np.log(sigma_null) - total_samples * np.log(
                sigma_alt
            )

        valid[i] = has_valid_feature

    return raw, valid


@dataclass(slots=True)
class GaussianMeanState:
    """Running sufficient statistics for the :class:`GaussianMean` score.

    Parameters
    ----------
    n_samples : int
        Number of samples seen so far.
    stats : np.ndarray
        Running sufficient statistics ``[sum(x), sum(x²)]``.  Shape ``(2,)`` for
        univariate series and ``(2, n_features)`` for multivariate series.
    """

    n_samples: int = 0
    stats: np.ndarray = field(
        default_factory=lambda: np.empty((2, 0), dtype=np.float64)
    )


@dataclass(frozen=True, slots=True)
class GaussianMean:
    """Mean-change score with unknown common variance.

    Under no change, X_1, ..., X_t are i.i.d. N(μ, σ²I_p) with σ² unknown;
    under a change at τ the mean shifts from μ₁ to μ₂ with a common unknown
    variance σ².

    **Score.**  For a candidate with n_1 pre-change and n_2 = t - n_1
    post-change observations, the per-feature twice profile log-likelihood ratio
    is ``2 * LR = t * (log σ̂²_null - log σ̂²_alt)``, asymptotically
    chi-squared(1) under the null (Wilks).

    **Aggregation.**  The per-feature statistics are combined according to the
    ``aggregation`` keyword (``"max"`` by default; see Parameters).

    **Centering and penalty.**  Each output column is centered by subtracting its
    degrees of freedom ``df`` inside the score.  When ``enable_penalty=True``
    (default) the centered column is divided by ``chi2_max_bound(M, df, t)``;
    when ``enable_penalty=False`` the divisor is 1.0.

    **Sample size requirement.**  Candidates with n_1 ≤ 2 or n_2 ≤ 2 return 0.0
    (degenerate variance estimates); features with a non-positive variance
    estimate are skipped.

    Parameters
    ----------
    n_features : int, default=1
        Number of features expected in each observation.
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
        """Validate inputs at construction time."""
        if self.n_features < 1:
            raise ValueError("n_features must be >= 1.")
        aggregation_mode_code(self.aggregation)

    @property
    def n_scores(self) -> int:
        """Number of scores returned by ``compute_penalized_scores``."""
        return len(
            aggregation_dims(self.aggregation, self.n_features, _GAUSSIAN_MEAN_DF)
        )

    def init_state(self) -> GaussianMeanState:
        """Return a fresh initial state with no observations seen.

        Univariate series use a flat ``(2,)`` statistic; multivariate series use
        ``(2, n_features)``.
        """
        if self.n_features == 1:
            stats = np.zeros(2, dtype=np.float64)
        else:
            stats = np.zeros((2, self.n_features), dtype=np.float64)
        return GaussianMeanState(stats=stats)

    def update(self, state: GaussianMeanState, x: ArrayLike) -> GaussianMeanState:
        """Update the state with a new observation.

        Parameters
        ----------
        state : GaussianMeanState
            Current state.
        x : ArrayLike
            New observation, shape ``(n_features,)``.

        Returns
        -------
        GaussianMeanState
            Updated state.
        """
        x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
        if x_arr.size != self.n_features:
            raise ValueError(
                f"GaussianMean expected observation of size {self.n_features}, "
                f"got {x_arr.size}."
            )
        if self.n_features == 1:
            val = float(x_arr[0])
            increment = np.array([val, val * val], dtype=np.float64)
        else:
            increment = np.vstack((x_arr, x_arr * x_arr))
        return GaussianMeanState(
            n_samples=state.n_samples + 1,
            stats=state.stats + increment,
        )

    def compute_penalized_scores(
        self,
        state: GaussianMeanState,
        grid_states: list[GaussianMeanState],
    ) -> np.ndarray:
        """Compute penalized LR score at every active grid candidate."""
        if len(grid_states) == 0:
            raise ValueError("grid_states is empty.")

        if self.n_features == 1:
            total_stats = state.stats.reshape(2, 1)
            grid_stats = np.stack([st.stats.reshape(2, 1) for st in grid_states])
        else:
            total_stats = state.stats
            grid_stats = np.stack([st.stats for st in grid_states])
        before_samples = np.array([st.n_samples for st in grid_states], dtype=np.int64)
        t = state.n_samples

        raw, valid = gaussian_mean_score(
            total_stats=total_stats,
            before_stats=grid_stats,
            total_samples=t,
            before_samples=before_samples,
        )
        cols = aggregate_features(raw, aggregation_mode_code(self.aggregation))
        dims = aggregation_dims(self.aggregation, self.n_features, _GAUSSIAN_MEAN_DF)

        out = np.empty((cols.shape[0], len(dims)), dtype=np.float64)
        for j, (n_terms, df) in enumerate(dims):
            penalty = chi2_max_bound(n_terms, df, t) if self.enable_penalty else 1.0
            out[:, j] = (cols[:, j] - df) / penalty

        out[~valid, :] = 0.0
        return out
