"""Nonparametric FOCuS score for changepoint detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numba as nb
import numpy as np
from numpy.typing import NDArray

from gridcp.scores._score_helpers import as_obs
from gridcp.typing import ArrayLike


@nb.njit(cache=True)
def _scalar_max_ll(k_i: float, n_i: int) -> float:
    """Return the Bernoulli maximized log-likelihood for one count."""
    if k_i <= 0.0 or k_i >= n_i:
        return 0.0

    p_hat = k_i / n_i
    return k_i * np.log(p_hat) + (n_i - k_i) * np.log1p(-p_hat)


@nb.njit(cache=True)
def bernoulli_max_ll(k: np.ndarray, n: int) -> np.ndarray:
    """Vectorize ``_scalar_max_ll`` over a grid of Bernoulli counts."""
    out = np.empty(k.shape, dtype=np.float64)
    for i in range(k.shape[0]):
        out[i] = _scalar_max_ll(k[i], n)
    return out


@nb.njit(cache=True)
def npfocus_score(
    total_sum: np.ndarray,
    before_sums: np.ndarray,
    total_samples: int,
    before_samples: np.ndarray,
) -> np.ndarray:
    """Compute NPFOCuS scores for all active candidate changepoints.

    Returns one row per active candidate and two columns.

    For each feature, NPFOCuS forms the sum and maximum of the Bernoulli LR
    statistics across the user grid. The returned score is then the maximum of
    these per-feature statistics across channels.
    """
    n_candidates = before_sums.shape[0]
    scores = np.zeros((n_candidates, 2), dtype=np.float64)
    if total_samples <= 0:
        return scores

    n_features = total_sum.shape[0]

    for j in range(n_candidates):
        n_pre = before_samples[j]
        n_post = total_samples - n_pre
        if n_pre <= 0 or n_post <= 0:
            continue

        best_sum_score = 0.0
        best_max_score = 0.0

        for feature_idx in range(n_features):
            total_sum_feature = total_sum[feature_idx]
            sum_pre_feature = before_sums[j, feature_idx]
            sum_post_feature = total_sum_feature - sum_pre_feature

            ll_tot = bernoulli_max_ll(total_sum_feature, total_samples)
            ll_pre = bernoulli_max_ll(sum_pre_feature, n_pre)
            ll_post = bernoulli_max_ll(sum_post_feature, n_post)
            lr_scores = 2.0 * (ll_post + ll_pre - ll_tot)

            sum_score = np.sum(lr_scores)
            max_score = np.max(lr_scores)

            if feature_idx == 0 or sum_score > best_sum_score:
                best_sum_score = sum_score
            if feature_idx == 0 or max_score > best_max_score:
                best_max_score = max_score

        scores[j, 0] = best_sum_score
        scores[j, 1] = best_max_score

    return scores


@dataclass(slots=True)
class NPFOCuSState:
    """Running state for the NPFOCuS score."""

    n_samples: int = 0
    n_smaller: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))


@dataclass(frozen=True, slots=True)
class NPFOCuS:
    """Nonparametric FOCuS score for changepoint detection.

    Under no change, X_1, ..., X_t are i.i.d. with unknown distribution F;
    under a change at τ, X_1, ..., X_{τ-1} ~ F₁ and X_τ, ..., X_t ~ F₂ for
    continuous distributions F₁ ≠ F₂.

    **Score.**  The score discretizes each feature's marginal using the
    user-supplied ``value_grid``.  For each threshold v_k in the grid, the
    indicator I(x ≤ v_k) is Bernoulli with parameter F(v_k) under no change.
    The Bernoulli LR statistic is computed for each grid point and candidate
    changepoint.

    **Aggregation.**  The grid LR values are combined into two test statistics
    per candidate (``n_scores = 2``):

    - **Column 0 (sum):** Σₖ 2 * LR_k(b), summing Bernoulli LR values across
      all grid points.
    - **Column 1 (max):** max_k 2 * LR_k(b), the maximum Bernoulli LR across
      grid points.

    For ``n_features > 1``, both columns are computed per feature and the
    per-feature maximum is taken, matching the channelwise-max convention of
    the other univariate-style scores.

    **Centering and penalty.**  No centering constant is subtracted from either
    output column.  ``enable_penalty`` defaults to ``False`` for this score,
    as threshold calibration is typically done empirically.  When
    ``enable_penalty=True``, both columns are divided by
    ``sqrt(2 log(t p)) + log(t p)``, where p = ``n_features``; this follows
    a Bonferroni-corrected penalty analogous to the two-dimensional
    exponential-family GLR case.  When ``enable_penalty=False`` (default), the
    divisor is 1.0 and the raw scores are returned.

    **Sample size requirement.**  None.

    Parameters
    ----------
    value_grid : ArrayLike
        Strictly increasing one-dimensional grid of threshold values used to
        construct the indicator process underlying the score.
    n_features : int, default=1
        Observation dimension expected by the score.
    enable_penalty : bool, default=False
        If ``True``, apply the time-dependent divisor
        ``sqrt(2 log(t p)) + log(t p)`` to both output columns.
        If ``False`` (default), use divisor 1.0.
    """

    value_grid: ArrayLike
    n_features: int = 1
    enable_penalty: bool = False

    @property
    def n_scores(self) -> int:
        """Number of scores returned by ``compute_penalized_scores``."""
        return 2

    @property
    def _value_grid_array(self) -> NDArray[np.float64]:
        """Return the normalized value grid as a float64 ndarray."""
        return cast(NDArray[np.float64], self.value_grid)

    def __post_init__(self) -> None:
        """Validate and normalize the evaluation grid and score config."""
        if self.n_features < 1:
            raise ValueError("n_features must be >= 1.")

        try:
            raw_value_grid = np.asarray(self.value_grid)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "value_grid must be an array-like of real numeric values."
            ) from exc

        if np.iscomplexobj(raw_value_grid):
            raise ValueError("value_grid must contain real values, not complex.")

        try:
            value_grid_arr = np.asarray(raw_value_grid, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "value_grid must be an array-like of real numeric values."
            ) from exc

        if value_grid_arr.ndim != 1:
            raise ValueError("value_grid must be a one-dimensional array.")
        if value_grid_arr.size == 0:
            raise ValueError("value_grid must be non-empty.")
        if not np.all(np.isfinite(value_grid_arr)):
            raise ValueError("value_grid must contain only finite values.")
        if np.any(np.diff(value_grid_arr) <= 0.0):
            raise ValueError("value_grid must be strictly increasing.")

        object.__setattr__(self, "value_grid", value_grid_arr)

    def init_state(self) -> NPFOCuSState:
        """Return a fresh initial state with no observations seen."""
        value_grid_arr = self._value_grid_array
        return NPFOCuSState(
            n_smaller=np.zeros((self.n_features, value_grid_arr.size), dtype=np.int64),
        )

    def update(
        self,
        state: NPFOCuSState,
        x: ArrayLike,
    ) -> NPFOCuSState:
        """Update the state with one observation."""
        x_arr = as_obs(x, self.n_features)
        value_grid_arr = self._value_grid_array

        next_n_samples = state.n_samples + 1
        next_n_smaller = state.n_smaller + (
            x_arr[:, None] <= value_grid_arr[None, :]
        ).astype(np.int64)
        return NPFOCuSState(
            n_samples=next_n_samples,
            n_smaller=next_n_smaller,
        )

    def _compute_centered_scores(
        self,
        state: NPFOCuSState,
        grid_states: list[NPFOCuSState],
    ) -> np.ndarray:
        """Compute centered but unpenalized scores for active candidates."""
        if len(grid_states) == 0:
            raise ValueError("grid_states is empty.")

        grid_sums = np.stack([candidate.n_smaller for candidate in grid_states])
        grid_n_samples = np.array(
            [candidate.n_samples for candidate in grid_states],
            dtype=np.int64,
        )
        return npfocus_score(
            total_sum=state.n_smaller,
            before_sums=grid_sums,
            total_samples=state.n_samples,
            before_samples=grid_n_samples,
        )

    def _get_penalty(self, n_samples: int) -> float:
        """Return the penalty divisor for the current sample size."""
        if self.enable_penalty:
            log_tp = np.log(n_samples * self.n_features)
            return np.sqrt(2.0 * log_tp) + log_tp
        return 1.0

    def compute_penalized_scores(
        self,
        state: NPFOCuSState,
        grid_states: list[NPFOCuSState],
    ) -> np.ndarray:
        """Compute penalized score for every active grid candidate."""
        return self._compute_centered_scores(state, grid_states) / self._get_penalty(
            state.n_samples
        )
