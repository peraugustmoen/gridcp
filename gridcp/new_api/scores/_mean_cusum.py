"""The Mean CUSUM score."""

from dataclasses import dataclass

import numba as nb
import numpy as np

from gridcp.new_api.typing import ArrayLike


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
    before_weight = np.sqrt(after_samples / (total_samples * before_samples)).reshape(-1, 1)
    after_weight = np.sqrt(before_samples / (total_samples * after_samples)).reshape(-1, 1)
    square_cusum = (before_weight * before_sums - after_weight * after_sums) ** 2
    return np.sum(square_cusum, axis=1) - 1

@nb.njit(cache=True)
def mean_cusum_penalty(n_samples: int) -> float:
    logg = np.log(n_samples)
    return logg + np.sqrt(logg)

@dataclass(slots=True)
class MeanCUSUMState:
    n_samples: int = 0
    sum: np.ndarray = None  # shape (n_features,); set by MeanCUSUM.init_state


@dataclass(frozen=True, slots=True)
class MeanCUSUM:
    n_features: int = 1

    def init_state(self) -> MeanCUSUMState:
        return MeanCUSUMState(sum=np.zeros(self.n_features))

    def update(
        self,
        state: MeanCUSUMState,
        x: ArrayLike,
    ) -> MeanCUSUMState:
        x = np.asarray(x)
        next_n_samples = state.n_samples + 1
        next_sum = state.sum + x
        return MeanCUSUMState(n_samples=next_n_samples, sum=next_sum)

    def compute_penalised_scores(
        self,
        state: MeanCUSUMState,
        grid_states: list[MeanCUSUMState],
    ) -> np.ndarray:
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

