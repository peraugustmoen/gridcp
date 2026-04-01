"""Nonparametric FOCuS score for univariate changepoint detection."""

from __future__ import annotations

from dataclasses import dataclass

import numba as nb
import numpy as np

from gridcp.typing import ArrayLike


@nb.njit(cache=True)
def _scalar_max_ll(k_i: float, n_i: int) -> float:
    """Return the Bernoulli maximized log-likelihood for one count.

    Parameters
    ----------
    k_i : float
        Number of successes.
    n_i : int
        Number of Bernoulli trials.

    Returns
    -------
    float
        Maximized Bernoulli log-likelihood contribution.
    """
    if k_i <= 0.0 or k_i >= n_i:
        return 0.0

    p_hat = k_i / n_i
    return k_i * np.log(p_hat) + (n_i - k_i) * np.log1p(-p_hat)


@nb.njit(cache=True)
def bernoulli_max_ll(k: np.ndarray, n: int) -> np.ndarray:
    """Vectorize `_scalar_max_ll` over a grid of Bernoulli counts.

    Parameters
    ----------
    k : np.ndarray
        One-dimensional array of success counts.
    n : int
        Number of Bernoulli trials associated with each count.

    Returns
    -------
    np.ndarray
        Maximized Bernoulli log-likelihood at each entry in `k`.
    """
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

    Parameters
    ----------
    total_sum : np.ndarray
        Total indicator counts over the current stream prefix.
    before_sums : np.ndarray
        Candidate-wise indicator counts for the pre-change segment.
    total_samples : int
        Number of observations in the current stream prefix.
    before_samples : np.ndarray
        Candidate-wise pre-change segment sizes.

    Returns
    -------
    np.ndarray
        Unpenalized score for each active candidate.
    """
    n_candidates = before_sums.shape[0]
    scores = np.zeros(n_candidates, dtype=np.float64)
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
        scores[j] = np.sum(2.0 * (ll_post + ll_pre - ll_tot))

    return scores


@nb.njit(cache=True)
def npfocus_penalty(n_samples: int) -> float:
    """Return the penalty used to scale the NPFOCuS score."""
    return 1.0


@dataclass(slots=True)
class NPFOCuSState:
    """Running state for the NPFOCuS score.

    Parameters
    ----------
    n_samples : int, default=0
        Number of observations incorporated so far.
    n_smaller : np.ndarray | None, default=None
        Vector whose `j`th entry counts how many observations are less than or
        equal to the `j`th grid point.
    """

    n_samples: int = 0
    n_smaller: np.ndarray = None


@dataclass(frozen=True, slots=True)
class NPFOCuS:
    """Nonparametric FOCuS score for univariate changepoint detection.

    The score discretizes the sample space using a one-dimensional evaluation
    grid. For each grid point it tracks the running count of observations below
    that point, then combines the resulting Bernoulli likelihood-ratio scores
    across the grid.

    Parameters
    ----------
    grid : ArrayLike
        Strictly increasing one-dimensional grid used to construct the
        indicator process underlying the score.
    """

    grid: ArrayLike

    def __post_init__(self) -> None:
        """Validate and normalize the evaluation grid."""
        grid_arr = np.asarray(self.grid, dtype=np.float64)
        if grid_arr.ndim != 1:
            raise ValueError("grid must be a one-dimensional array.")
        if grid_arr.size == 0:
            raise ValueError("grid must be non-empty.")
        if not np.all(np.isfinite(grid_arr)):
            raise ValueError("grid must contain only finite values.")
        if np.any(np.diff(grid_arr) <= 0.0):
            raise ValueError("grid must be strictly increasing.")
        object.__setattr__(self, "grid", grid_arr)

    def init_state(self) -> NPFOCuSState:
        """Return a fresh initial state with no observations seen."""
        return NPFOCuSState(
            n_samples=0,
            n_smaller=np.zeros(self.grid.size, dtype=np.int64),
        )

    def update(
        self,
        state: NPFOCuSState,
        x: ArrayLike,
    ) -> NPFOCuSState:
        """Update the state with one univariate observation.

        Parameters
        ----------
        state : NPFOCuSState
            Current running state.
        x : ArrayLike
            New observation. Must be scalar or length 1 after flattening.

        Returns
        -------
        NPFOCuSState
            Updated state.
        """
        x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
        if x_arr.size != 1:
            raise ValueError(
                f"NPFOCuS expected a univariate observation, got size {x_arr.size}."
            )

        next_n_samples = state.n_samples + 1
        next_n_smaller = state.n_smaller.copy()
        next_n_smaller += (x_arr[0] <= self.grid).astype(np.int64)
        return NPFOCuSState(
            n_samples=next_n_samples,
            n_smaller=next_n_smaller,
        )

    def compute_penalised_scores(
        self,
        state: NPFOCuSState,
        grid_states: list[NPFOCuSState],
    ) -> np.ndarray:
        """Compute a penalized score for every active grid candidate.

        Parameters
        ----------
        state : NPFOCuSState
            Global running state after the latest observation.
        grid_states : list[NPFOCuSState]
            Per-candidate state snapshots, one per active grid point.

        Returns
        -------
        np.ndarray, shape (len(grid_states),)
            Penalized score for each active candidate.
        """
        if len(grid_states) == 0:
            raise ValueError("grid_states is empty.")

        grid_sums = np.stack([candidate.n_smaller for candidate in grid_states])
        grid_n_samples = np.array(
            [candidate.n_samples for candidate in grid_states],
            dtype=np.int64,
        )
        scores = npfocus_score(
            total_sum=state.n_smaller,
            before_sums=grid_sums,
            total_samples=state.n_samples,
            before_samples=grid_n_samples,
        )
        penalty = npfocus_penalty(state.n_samples)
        return scores / penalty
