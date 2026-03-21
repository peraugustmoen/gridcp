"""Mean-change score for univariate data with unknown variance."""

from dataclasses import dataclass

import numba as nb
import numpy as np

from gridcp.typing import ArrayLike


@nb.njit(cache=True)
def mean_unknown_variance_score(
    total_stats: np.ndarray,
    before_stats: np.ndarray,
    total_samples: int,
    before_samples: np.ndarray,
) -> np.ndarray:
    """Compute univariate mean-change LR scores under unknown variance.

    Statistics are based on sufficient statistics ``[sum(x), sum(x^2)]``.
    """
    n_candidates = before_samples.shape[0]
    out = np.zeros(n_candidates, dtype=np.float64)

    total_sum = total_stats[0]
    total_sum2 = total_stats[1]

    for i in range(n_candidates):
        n1 = before_samples[i]
        n2 = total_samples - n1

        # Keep small-sample behavior identical to old API implementation.
        if n1 <= 2 or n2 <= 2:
            out[i] = 0.0
            continue

        sum1 = before_stats[i, 0]
        sum2_1 = before_stats[i, 1]

        sum2 = total_sum - sum1
        sum2_2 = total_sum2 - sum2_1

        sigma_null = (
            total_sum2 - total_sum * total_sum / total_samples
        ) / total_samples
        sigma_alt = (
            (sum2_1 - sum1 * sum1 / n1) + (sum2_2 - sum2 * sum2 / n2)
        ) / total_samples

        # Guard numerics: non-positive variance estimates do not produce valid LR.
        if sigma_null <= 0.0 or sigma_alt <= 0.0:
            out[i] = 0.0
            continue

        lr = total_samples * np.log(sigma_null) - total_samples * np.log(sigma_alt)
        out[i] = lr - 1.0

    return out


@nb.njit(cache=True)
def mean_unknown_variance_penalty(n_samples: int) -> float:
    logg = np.log(2.0 * n_samples)
    return logg + np.sqrt(logg)


@dataclass(slots=True)
class MeanCUSUMUnknownVarianceState:
    """Running sufficient statistics for unknown-variance mean-change score."""

    n_samples: int = 0
    stats: np.ndarray = None  # shape (2,): [sum(x), sum(x^2)]


@dataclass(frozen=True, slots=True)
class MeanCUSUMUnknownVariance:
    """Univariate mean-change score under unknown variance.

    Uses Gaussian likelihood-ratio style scoring with unknown variance and
    sufficient statistics ``[sum(x), sum(x^2)]``.
    """

    def init_state(self) -> MeanCUSUMUnknownVarianceState:
        """Return a fresh initial state with no observations seen."""
        return MeanCUSUMUnknownVarianceState(stats=np.zeros(2, dtype=np.float64))

    def update(
        self,
        state: MeanCUSUMUnknownVarianceState,
        x: ArrayLike,
    ) -> MeanCUSUMUnknownVarianceState:
        """Update sufficient statistics with one univariate observation."""
        x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
        if x_arr.size != 1:
            raise ValueError(
                "MeanCUSUMUnknownVariance expects univariate observations (size 1)."
            )

        val = float(x_arr[0])
        next_n_samples = state.n_samples + 1
        next_stats = state.stats + np.array([val, val * val], dtype=np.float64)
        return MeanCUSUMUnknownVarianceState(
            n_samples=next_n_samples,
            stats=next_stats,
        )

    def compute_penalised_scores(
        self,
        state: MeanCUSUMUnknownVarianceState,
        grid_states: list[MeanCUSUMUnknownVarianceState],
    ) -> np.ndarray:
        """Compute penalised LR score at every active grid candidate."""
        if len(grid_states) == 0:
            raise ValueError("grid_states is empty.")

        grid_stats = np.stack([st.stats for st in grid_states])
        grid_n_samples = np.array([st.n_samples for st in grid_states], dtype=np.int64)
        scores = mean_unknown_variance_score(
            total_stats=state.stats,
            before_stats=grid_stats,
            total_samples=state.n_samples,
            before_samples=grid_n_samples,
        )
        penalty = mean_unknown_variance_penalty(state.n_samples)
        return scores / penalty
