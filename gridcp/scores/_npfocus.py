"""Nonparametric FOCuS score for changepoint detection."""

from __future__ import annotations

from dataclasses import dataclass, field

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

    The score discretizes the sample space using a one-dimensional evaluation
    grid. For each feature and each grid point it tracks the running count of
    observations below that point, then forms a two-statistic score from the
    resulting Bernoulli likelihood-ratio values across the grid:
    the grid-sum and the grid-maximum.

    For ``n_features > 1``, NPFOCuS is applied independently to each feature
    and the final two-component score is the maximum across features, matching
    the channelwise-max convention used by the other univariate-style scores.

    Parameters
    ----------
    value_grid : ArrayLike
        Strictly increasing one-dimensional grid used to construct the
        indicator process underlying the score.
    n_features : int, default=1
        Observation dimension expected by the score.
    enable_penalty : bool, default=False
        If ``True``, apply the time-dependent divisor used in the
        two-dimensional exponential-family GLR case with a Bonferroni-style
        feature correction, ``sqrt(2 log(t p)) + log(t p)`` where
        ``p = n_features``. If ``False``, use constant divisor 1.0.
    """

    value_grid: NDArray[np.float64]
    n_features: int = 1
    enable_penalty: bool = False

    @property
    def n_tests(self) -> int:
        """Number of tests returned by ``compute_penalized_scores``."""
        return 2

    def __post_init__(self) -> None:
        """Validate and normalize the evaluation grid and score config."""
        if self.n_features < 1:
            raise ValueError("n_features must be >= 1.")

        value_grid_arr = np.asarray(self.value_grid, dtype=np.float64)
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
        return NPFOCuSState(
            n_smaller=np.zeros((self.n_features, self.value_grid.size), dtype=np.int64),
        )

    def update(
        self,
        state: NPFOCuSState,
        x: ArrayLike,
    ) -> NPFOCuSState:
        """Update the state with one observation."""
        x_arr = as_obs(x, self.n_features)

        next_n_samples = state.n_samples + 1
        next_n_smaller = state.n_smaller + (
            x_arr[:, None] <= self.value_grid[None, :]
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
