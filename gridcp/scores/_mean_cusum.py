"""The Mean CUSUM score."""

from dataclasses import dataclass

import numba as nb
import numpy as np

from gridcp.typing import ArrayLike


@nb.njit(cache=True)
def mean_cusum_score(
    total_sum: np.ndarray,
    before_sums: np.ndarray,
    total_samples: int,
    before_samples: np.ndarray,
) -> np.ndarray:
    """
    Calculate the CUSUM score for a change in the mean.

    Compares the mean of the data before and after the split within the interval from
    ``start:end``.

    Parameters
    ----------
    starts : `np.ndarray`
        Start indices of the intervals to test for a change in the mean.
    ends : `np.ndarray`
        End indices of the intervals to test for a change in the mean.
    splits : `np.ndarray`
        Split indices of the intervals to test for a change in the mean.
    sums : `np.ndarray`
        Cumulative sum of the input data, with a row of 0-entries as the first row.

    Returns
    -------
    `np.ndarray`
        CUSUM scores for the intervals and splits.
    """
    after_samples = total_samples - before_samples
    after_sums = total_sum - before_sums
    before_weight = np.sqrt(after_samples / (total_samples * before_samples)).reshape(
        -1, 1
    )
    after_weight = np.sqrt(before_samples / (total_samples * after_samples)).reshape(
        -1, 1
    )
    square_cusum = (before_weight * before_sums - after_weight * after_sums) ** 2

    # Numba compatibility: compute row-wise max explicitly.
    n_candidates = square_cusum.shape[0]
    n_features = square_cusum.shape[1]
    out = np.empty(n_candidates, dtype=np.float64)

    for i in range(n_candidates):
        row_max = square_cusum[i, 0]
        for j in range(1, n_features):
            if square_cusum[i, j] > row_max:
                row_max = square_cusum[i, j]
        out[i] = row_max - 1.0

    return out


@nb.njit(cache=True)
def mean_cusum_penalty(n_samples: int) -> float:
    logg = np.log(n_samples)
    return logg + np.sqrt(logg)


@dataclass(slots=True)
class MeanCUSUMState:
    """State for the MeanCUSUM score.

    This holds the running statistics needed to compute the CUSUM score for a change in
    the mean.

    Parameters
    ----------
    n_samples : int
        Number of samples seen so far.
    sum : np.ndarray
        Cumulative sum of the samples seen so far, shape (n_features,).
    """

    n_samples: int = 0
    sum: np.ndarray = None  # shape (n_features,); set by MeanCUSUM.init_state


@dataclass(frozen=True, slots=True)
class MeanCUSUM:
    """CUSUM score for a change in the mean.

    This score compares the mean of the data before and after the split within the
    interval from ``start:end``. It is based on the cumulative sum of the data and
    includes a penalty term to control for false alarms.

    Parameters
    ----------
    n_features : int, default=1
        Number of features in the input data. This determines the shape of the running
        sum statistic.

    Examples
    --------
    >>> from gridcp.new_api.scores import MeanCUSUM
    >>> score = MeanCUSUM(n_features=2)
    >>> state = score.init_state()
    >>> state = score.update(state, [1.0, 2.0])
    >>> state = score.update(state, [1.5, 2.5])
    >>> state
    MeanCUSUMState(n_samples=2, sum=array([2.5, 4.5]))
    """

    n_features: int = 1

    def init_state(self) -> MeanCUSUMState:
        """Return a fresh initial state with no observations seen."""
        return MeanCUSUMState(sum=np.zeros(self.n_features))

    def update(
        self,
        state: MeanCUSUMState,
        x: ArrayLike,
    ) -> MeanCUSUMState:
        """Update the state with a new observation.

        Parameters
        ----------
        state : MeanCUSUMState
            Current state.
        x : ArrayLike
            New observation, shape (n_features,).

        Returns
        -------
        MeanCUSUMState
            Updated state.
        """
        x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
        if x_arr.size != self.n_features:
            raise ValueError(
                "MeanCUSUM expected observation of size "
                f"{self.n_features}, got {x_arr.size}."
            )

        next_n_samples = state.n_samples + 1
        next_sum = state.sum + x_arr
        return MeanCUSUMState(n_samples=next_n_samples, sum=next_sum)

    def compute_penalised_scores(
        self,
        state: MeanCUSUMState,
        grid_states: list[MeanCUSUMState],
    ) -> np.ndarray:
        """Compute a penalised score for every active grid candidate.

        Parameters
        ----------
        state : MeanCUSUMState
            Global running state after the latest observation.
        grid_states : list[MeanCUSUMState]
            Per-candidate state snapshots, one per active grid point.

        Returns
        -------
        np.ndarray, shape (len(grid_states),)
            Penalised score for each active candidate.
        """
        if len(grid_states) == 0:
            raise ValueError("grid_states is empty.")

        grid_sums = np.stack([st.sum for st in grid_states])
        grid_n_samples = np.array([st.n_samples for st in grid_states], dtype=np.int64)
        scores = mean_cusum_score(
            total_sum=state.sum,
            before_sums=grid_sums,
            total_samples=state.n_samples,
            before_samples=grid_n_samples,
        )
        penalty = mean_cusum_penalty(state.n_samples)
        return scores / penalty
