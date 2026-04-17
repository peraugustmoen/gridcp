"""Nonparametric FOCuS score for univariate changepoint detection."""

from __future__ import annotations

from dataclasses import dataclass, field

import numba as nb
import numpy as np

from gridcp.scores._score_helpers import as_obs
from gridcp.typing import ArrayLike, PenaltyType


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

    Returns one row per active candidate and two columns:
    - column 0: sum of Bernoulli LR statistics across the user grid
    - column 1: maximum Bernoulli LR statistic over the user grid
    """
    n_candidates = before_sums.shape[0]
    scores = np.zeros((n_candidates, 2), dtype=np.float64)
    if total_samples <= 0:
        return scores

    ll_tot = bernoulli_max_ll(total_sum, total_samples)

    for j in range(n_candidates):
        n_pre = before_samples[j]
        n_post = total_samples - n_pre
        if n_pre <= 0 or n_post <= 0:
            continue

        sum_pre = before_sums[j]
        sum_post = total_sum - sum_pre
        ll_pre = bernoulli_max_ll(sum_pre, n_pre)
        ll_post = bernoulli_max_ll(sum_post, n_post)
        lr_scores = 2.0 * (ll_post + ll_pre - ll_tot)
        scores[j, 0] = np.sum(lr_scores)
        scores[j, 1] = np.max(lr_scores)

    return scores


@dataclass(slots=True)
class NPFOCuSState:
    """Running state for the NPFOCuS score."""

    n_samples: int = 0
    n_smaller: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))


@dataclass(frozen=True, slots=True)
class NPFOCuS:
    """Nonparametric FOCuS score for univariate changepoint detection.

    The score discretizes the sample space using a one-dimensional evaluation
    grid. For each grid point it tracks the running count of observations below
    that point, then forms a two-statistic multivariate score from the
    resulting Bernoulli likelihood-ratio values across the grid:
    the grid-sum and the grid-maximum.

    Parameters
    ----------
    value_grid : ArrayLike
        Strictly increasing one-dimensional grid used to construct the
        indicator process underlying the score.
    n_features : int, default=1
        Observation dimension expected by the score. NPFOCuS only supports the
        univariate case, so this must be ``1``.
    penalty : PenaltyType, default=PenaltyType.CONSTANT
        Penalty mode used to scale the score. ``PenaltyType.CONSTANT`` uses a
        unit penalty. ``PenaltyType.TIME_DEPENDENT`` uses the same divisor as
        the two-dimensional exponential-family GLR case,
        ``sqrt(2 log(t)) + log(t)``.
    """

    value_grid: ArrayLike
    n_features: int = 1
    penalty: PenaltyType = PenaltyType.CONSTANT

    def __post_init__(self) -> None:
        """Validate and normalize the evaluation grid and score config."""
        if self.n_features != 1:
            raise ValueError("NPFOCuS only supports n_features=1.")

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
            n_smaller=np.zeros(self.value_grid.size, dtype=np.int64),
        )

    def update(
        self,
        state: NPFOCuSState,
        x: ArrayLike,
    ) -> NPFOCuSState:
        """Update the state with one univariate observation."""
        x_arr = as_obs(x, self.n_features)

        next_n_samples = state.n_samples + 1
        next_n_smaller = state.n_smaller + (x_arr[0] <= self.value_grid).astype(
            np.int64
        )
        return NPFOCuSState(
            n_samples=next_n_samples,
            n_smaller=next_n_smaller,
        )

    def _compute_centered_scores(
        self,
        state: NPFOCuSState,
        grid_states: list[NPFOCuSState],
    ) -> np.ndarray:
        """Compute centered but unpenalised scores for active candidates."""
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
        if self.penalty == PenaltyType.TIME_DEPENDENT:
            log_t = np.log(n_samples)
            return np.sqrt(2.0 * log_t) + log_t
        if self.penalty == PenaltyType.CONSTANT:
            return 1.0
        raise ValueError(f"Unsupported penalty mode: {self.penalty!r}")

    def compute_penalised_scores(
        self,
        state: NPFOCuSState,
        grid_states: list[NPFOCuSState],
        n_samples_for_penalty: int,
    ) -> np.ndarray:
        """Compute penalised score for every active grid candidate.

        Parameters
        ----------
        n_samples_for_penalty : int
            Sample count used for the penalty divisor.
        """
        return self._compute_centered_scores(state, grid_states) / self._get_penalty(
            n_samples_for_penalty
        )
