"""Nonparametric FOCuS score for changepoint detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numba as nb
import numpy as np
from numpy.typing import NDArray

from gridcp.scores._aggregation import (
    aggregate_features,
    aggregation_dims,
    aggregation_mode_code,
)
from gridcp.scores._score_helpers import as_obs
from gridcp.typing import ArrayLike

# Per-channel degrees of freedom placeholder; NPFOCuS applies no penalty or
# centering, so this only drives the per-aggregation column count.
_NPFOCUS_DF = 1


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
def npfocus_channel_stats(
    total_sum: np.ndarray,
    before_sums: np.ndarray,
    total_samples: int,
    before_samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the two per-channel NPFOCuS statistics for all candidates.

    For each candidate and feature (channel), the Bernoulli 2*LR is formed
    across the value grid and reduced two ways: a sum over the grid and a max
    over the grid.  No aggregation across channels, no centering, no penalty.

    Parameters
    ----------
    total_sum : np.ndarray, shape (n_features, value_grid.size)
        Per-threshold cumulative counts across all observations.
    before_sums : np.ndarray, shape (G, n_features, value_grid.size)
        Per-candidate cumulative counts.
    total_samples : int
        Total number of observations seen so far.
    before_samples : np.ndarray, shape (G,)
        First post-change index (0-based) for each candidate.

    Returns
    -------
    sum_stat : np.ndarray, shape (G, n_features)
        Per-channel sum-over-grid Bernoulli 2*LR.
    max_stat : np.ndarray, shape (G, n_features)
        Per-channel max-over-grid Bernoulli 2*LR.
    """
    n_candidates = before_sums.shape[0]
    n_features = total_sum.shape[0]
    sum_stat = np.zeros((n_candidates, n_features), dtype=np.float64)
    max_stat = np.zeros((n_candidates, n_features), dtype=np.float64)
    if total_samples <= 0:
        return sum_stat, max_stat

    for j in range(n_candidates):
        n_pre = before_samples[j]
        n_post = total_samples - n_pre
        if n_pre <= 0 or n_post <= 0:
            continue

        for feature_idx in range(n_features):
            total_sum_feature = total_sum[feature_idx]
            sum_pre_feature = before_sums[j, feature_idx]
            sum_post_feature = total_sum_feature - sum_pre_feature

            ll_tot = bernoulli_max_ll(total_sum_feature, total_samples)
            ll_pre = bernoulli_max_ll(sum_pre_feature, n_pre)
            ll_post = bernoulli_max_ll(sum_post_feature, n_post)
            lr_scores = 2.0 * (ll_post + ll_pre - ll_tot)

            sum_stat[j, feature_idx] = np.sum(lr_scores)
            max_stat[j, feature_idx] = np.max(lr_scores)

    return sum_stat, max_stat


@dataclass(slots=True)
class NPFOCuSState:
    """Running state for the NPFOCuS score."""

    n_samples: int = 0
    n_smaller: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))


@dataclass(frozen=True, slots=True)
class NPFOCuS:
    """Nonparametric FOCuS score for changepoint detection.

    Under no change, X_1, ..., X_t are i.i.d. with unknown distribution F;
    under a change at τ the marginal distribution changes.

    **Score.**  The score discretizes each feature's marginal using the
    user-supplied ``value_grid``.  For each threshold v_k, the indicator
    I(x ≤ v_k) is Bernoulli, and the Bernoulli 2*LR statistic is computed for
    each grid point and candidate changepoint.

    **Channel statistics.**  For each feature (channel), the grid 2*LR values are
    reduced two ways: a **sum over the value grid** and a **max over the value
    grid**.  These two per-channel statistics are intrinsic to NPFOCuS and are
    always both produced.

    **Aggregation.**  The ``aggregation`` keyword acts on the **channel axis**,
    independently on each of the two statistics:

    - ``"max"`` (default): 2 columns (channel-max of the grid-sum and of the
      grid-max statistic).
    - ``"sum"``: 2 columns (channel-sum of each).
    - ``"max-sum"``: 4 columns.
    - ``None``/``"None"``: ``2 * n_features`` columns (per-channel, both stats).

    **Centering and penalty.**  None.  The grid-sum statistic sums strongly
    positively correlated indicator terms, so there is no defensible
    chi-squared degrees of freedom; NPFOCuS therefore applies **no penalty and
    no centering** and has no ``enable_penalty`` parameter.  Calibrate the
    detector threshold empirically.

    **Sample size requirement.**  None (candidates with no pre- or post-change
    observations score 0).

    Parameters
    ----------
    value_grid : ArrayLike
        Strictly increasing one-dimensional grid of threshold values used to
        construct the indicator process underlying the score.
    n_features : int, default=1
        Observation dimension expected by the score.
    aggregation : {"max", "sum", "max-sum", None, "None"}, default="max"
        How to combine the per-channel statistics across the feature axis,
        applied independently to the grid-sum and grid-max statistics.
    """

    value_grid: ArrayLike
    n_features: int = 1
    aggregation: object = "max"

    @property
    def n_scores(self) -> int:
        """Number of scores returned by ``compute_penalized_scores``."""
        per_stat = len(aggregation_dims(self.aggregation, self.n_features, _NPFOCUS_DF))
        return 2 * per_stat

    @property
    def _value_grid_array(self) -> NDArray[np.float64]:
        """Return the normalized value grid as a float64 ndarray."""
        return cast(NDArray[np.float64], self.value_grid)

    def __post_init__(self) -> None:
        """Validate and normalize the evaluation grid and score config."""
        if self.n_features < 1:
            raise ValueError("n_features must be >= 1.")
        aggregation_mode_code(self.aggregation)

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

    def compute_penalized_scores(
        self,
        state: NPFOCuSState,
        grid_states: list[NPFOCuSState],
    ) -> np.ndarray:
        """Compute the channel-aggregated raw NPFOCuS statistics (no penalty)."""
        if len(grid_states) == 0:
            raise ValueError("grid_states is empty.")

        grid_sums = np.stack([candidate.n_smaller for candidate in grid_states])
        grid_n_samples = np.array(
            [candidate.n_samples for candidate in grid_states],
            dtype=np.int64,
        )
        sum_stat, max_stat = npfocus_channel_stats(
            total_sum=state.n_smaller,
            before_sums=grid_sums,
            total_samples=state.n_samples,
            before_samples=grid_n_samples,
        )
        mode = aggregation_mode_code(self.aggregation)
        cols_sum = aggregate_features(sum_stat, mode)
        cols_max = aggregate_features(max_stat, mode)
        return np.concatenate((cols_sum, cols_max), axis=1)
