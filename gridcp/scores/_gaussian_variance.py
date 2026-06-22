"""Variance-change LR score assuming known zero mean (Gaussian LR)."""

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

# Per-coordinate degrees of freedom of the variance-change LR statistic.
_GAUSSIAN_VARIANCE_DF = 1


@nb.njit(cache=True)
def gaussian_variance_score(
    total_sum_sq: np.ndarray,
    before_sum_sqs: np.ndarray,
    total_samples: int,
    before_samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature variance-change LR for all grid candidates.

    Parameters
    ----------
    total_sum_sq : np.ndarray
        Sum of squares over all observations, shape ``(n_features,)``.
    before_sum_sqs : np.ndarray
        Sum of squares for each grid candidate, shape ``(G, n_features)``.
    total_samples : int
        Total number of observations seen so far.
    before_samples : np.ndarray
        Pre-change sample count for each candidate, shape ``(G,)``.

    Returns
    -------
    raw : np.ndarray, shape (G, n_features)
        Raw (uncentered, unaggregated) per-feature LR; degenerate features 0.
    valid : np.ndarray, shape (G,)
        ``False`` for candidates with ``n1 == 0``/``n2 == 0`` or no valid
        feature (these score 0 after centering).
    """
    n_candidates = before_sum_sqs.shape[0]
    n_features = before_sum_sqs.shape[1]
    raw = np.zeros((n_candidates, n_features), dtype=np.float64)
    valid = np.zeros(n_candidates, dtype=np.bool_)
    t = total_samples

    for i in range(n_candidates):
        n1 = before_samples[i]
        n2 = t - n1
        if n1 == 0 or n2 == 0:
            continue

        has_valid = False
        for k in range(n_features):
            sigma_tot = total_sum_sq[k] / t
            sigma_1 = before_sum_sqs[i, k] / n1
            sigma_2 = (total_sum_sq[k] - before_sum_sqs[i, k]) / n2
            if sigma_tot <= 0.0 or sigma_1 <= 0.0 or sigma_2 <= 0.0:
                continue
            has_valid = True
            raw[i, k] = (
                t * np.log(sigma_tot) - n1 * np.log(sigma_1) - n2 * np.log(sigma_2)
            )

        valid[i] = has_valid

    return raw, valid


@dataclass(slots=True)
class GaussianVarianceState:
    """State for the :class:`GaussianVariance` score.

    Parameters
    ----------
    n_samples : int
        Number of samples seen so far.
    sum_sq : np.ndarray
        Cumulative sum of squared observations, shape ``(n_features,)``.
    """

    n_samples: int = 0
    sum_sq: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))


@dataclass(frozen=True, slots=True)
class GaussianVariance:
    """Variance-change LR score assuming known zero mean.

    Under no change, X_1, ..., X_t are i.i.d. N(0, σ²); under a change at τ the
    variance shifts from σ₁² to σ₂².  If the data has a nonzero mean, subtract
    it before feeding observations to this score.

    **Score.**  For a candidate with n_1 pre-change and n_2 = t - n_1
    post-change observations, the per-feature twice log-likelihood ratio is
    ``2 * LR = t log σ̂²_tot - n_1 log σ̂²_1 - n_2 log σ̂²_2`` (zero-mean
    variance estimates), asymptotically chi-squared(1) under the null.

    **Aggregation.**  The per-feature statistics are combined according to the
    ``aggregation`` keyword (``"max"`` by default; see Parameters).

    **Centering and penalty.**  Each output column is centered by subtracting its
    degrees of freedom ``df`` inside the score.  When ``enable_penalty=True``
    (default) the centered column is divided by ``chi2_max_bound(M, df, t)``;
    when ``enable_penalty=False`` the divisor is 1.0.

    **Sample size requirement.**  Candidates with n_1 == 0 or n_2 == 0 return
    0.0; features with a non-positive variance estimate are skipped.

    Parameters
    ----------
    n_features : int, default=1
        Number of features in the input data.
    aggregation : {"max", "sum", "max-sum", None, "None"}, default="max"
        How to combine the p per-feature statistics across the feature axis.
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
            aggregation_dims(self.aggregation, self.n_features, _GAUSSIAN_VARIANCE_DF)
        )

    def init_state(self) -> GaussianVarianceState:
        """Return a fresh initial state with no observations seen."""
        return GaussianVarianceState(sum_sq=np.zeros(self.n_features, dtype=np.float64))

    def update(
        self, state: GaussianVarianceState, x: ArrayLike
    ) -> GaussianVarianceState:
        """Update the state with a new observation.

        Parameters
        ----------
        state : GaussianVarianceState
            Current state.
        x : ArrayLike
            New observation, shape ``(n_features,)``.

        Returns
        -------
        GaussianVarianceState
            Updated state.
        """
        x_arr = as_obs(x, self.n_features)
        return GaussianVarianceState(
            n_samples=state.n_samples + 1,
            sum_sq=state.sum_sq + x_arr * x_arr,
        )

    def compute_penalized_scores(
        self,
        state: GaussianVarianceState,
        grid_states: list[GaussianVarianceState],
    ) -> np.ndarray:
        """Compute penalized variance-change scores for all active candidates."""
        if len(grid_states) == 0:
            raise ValueError("grid_states is empty.")

        before_sum_sqs = np.stack([st.sum_sq for st in grid_states])
        before_samples = np.array([st.n_samples for st in grid_states], dtype=np.int64)
        t = state.n_samples

        raw, valid = gaussian_variance_score(
            total_sum_sq=state.sum_sq,
            before_sum_sqs=before_sum_sqs,
            total_samples=t,
            before_samples=before_samples,
        )
        cols = aggregate_features(raw, aggregation_mode_code(self.aggregation))
        dims = aggregation_dims(
            self.aggregation, self.n_features, _GAUSSIAN_VARIANCE_DF
        )

        out = np.empty((cols.shape[0], len(dims)), dtype=np.float64)
        for j, (n_terms, df) in enumerate(dims):
            penalty = chi2_max_bound(n_terms, df, t) if self.enable_penalty else 1.0
            out[:, j] = (cols[:, j] - df) / penalty

        out[~valid, :] = 0.0
        return out
