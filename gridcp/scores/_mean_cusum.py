"""The Mean CUSUM score."""

from dataclasses import dataclass, field

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

    Compares the mean of the data before and after each candidate split point.

    Parameters
    ----------
    total_sum : np.ndarray
        Cumulative sum over all observations, shape ``(n_features,)``.
    before_sums : np.ndarray
        Per-candidate sums of observations, shape ``(G, n_features)``.
    total_samples : int
        Total number of observations seen so far.
    before_samples : np.ndarray
        Number of observations before each candidate, shape ``(G,)``.

    Returns
    -------
    np.ndarray
        Centered max CUSUM scores for each candidate, shape ``(G,)``.
        Each value is ``max_j C_j - 1`` where
        ``C_j = (n_1 n_2 / t) (x̄_{1,j} - x̄_{2,j})²``.
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
def mean_cusum_penalty(n_samples: int, n_features: int) -> float:
    """Return the penalty divisor ``log(t p) + sqrt(log(t p))``."""
    logg = np.log(n_samples * n_features)
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
    sum: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )  # shape (n_features,); set by MeanCUSUM.init_state


@dataclass(frozen=True, slots=True)
class MeanCUSUM:
    """CUSUM score for a change in the mean with known unit variance.

    Under no change, X_1, ..., X_t are i.i.d. N(μ, I_p); under a change at τ,
    X_1, ..., X_{τ-1} ~ N(μ₁, I_p) and X_τ, ..., X_t ~ N(μ₂, I_p) with μ₁ ≠ μ₂.

    **Score.**  For a candidate with n_1 pre-change and n_2 = t - n_1
    post-change observations, the per-coordinate squared CUSUM statistic is

        C_j = (n_1 n_2 / t) (x̄_{1,j} - x̄_{2,j})²,

    which equals the squared coordinate-j CUSUM and is the exact
    2 * LR under known unit variance.  Under the null, C_j is
    chi-squared(1) for each j.

    **Aggregation.**  For ``n_features > 1``, the score is ``max_j C_j``,
    sensitive to changes concentrated in any single coordinate.  For a dense
    multivariate test that aggregates all coordinates, use
    :class:`MultivariateMeanIdentityCov`, which also returns the coordinate-sum
    statistic.

    **Centering and penalty.**  The statistic is centered by subtracting 1 (the
    chi-squared(1) mean).  When ``enable_penalty=True`` (default), the centered
    score is divided by the time-dependent penalty
    ``log(t p) + sqrt(log(t p))``, where p = ``n_features``; this penalty is
    derived from a non-asymptotic Gaussian concentration bound.  When
    ``enable_penalty=False``, the divisor is 1.0 and the raw centered statistic
    is returned.

    **Sample size requirement.**  None.

    Parameters
    ----------
    n_features : int, default=1
        Number of features in the input data. This determines the shape of the
        running sum statistic.
    enable_penalty : bool, default=True
        If ``True``, divide the centered score by ``log(t p) + sqrt(log(t p))``.
        If ``False``, return the raw centered statistic with divisor 1.0.

    Examples
    --------
    >>> from gridcp.scores import MeanCUSUM
    >>> score = MeanCUSUM(n_features=2)
    >>> state = score.init_state()
    >>> state = score.update(state, [1.0, 2.0])
    >>> state = score.update(state, [1.5, 2.5])
    >>> state
    MeanCUSUMState(n_samples=2, sum=array([2.5, 4.5]))
    """

    n_features: int = 1
    enable_penalty: bool = True

    @property
    def n_scores(self) -> int:
        """Number of scores returned by ``compute_penalized_scores``."""
        return 1

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

    def _compute_centered_scores(
        self,
        state: MeanCUSUMState,
        grid_states: list[MeanCUSUMState],
    ) -> np.ndarray:
        """Compute centered (but unpenalized) scores for every active grid candidate."""
        if len(grid_states) == 0:
            raise ValueError("grid_states is empty.")

        grid_sums = np.stack([st.sum for st in grid_states])
        grid_n_samples = np.array([st.n_samples for st in grid_states], dtype=np.int64)
        return mean_cusum_score(
            total_sum=state.sum,
            before_sums=grid_sums,
            total_samples=state.n_samples,
            before_samples=grid_n_samples,
        )

    def _get_penalty(self, n_samples: int) -> float:
        """Return the penalty divisor for the current sample size."""
        if self.enable_penalty:
            return mean_cusum_penalty(n_samples, self.n_features)
        return 1.0

    def compute_penalized_scores(
        self,
        state: MeanCUSUMState,
        grid_states: list[MeanCUSUMState],
    ) -> np.ndarray:
        """Compute a penalized score for every active grid candidate."""
        scores = self._compute_centered_scores(state, grid_states) / self._get_penalty(
            state.n_samples
        )
        return scores[:, np.newaxis]
