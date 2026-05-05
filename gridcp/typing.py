"""Typing definitions for the grid detector API."""

from typing import Protocol, TypedDict, TypeVar, runtime_checkable
import numpy as np
from numpy.typing import ArrayLike

TScoreState = TypeVar("TScoreState")


class DetectorOutput(TypedDict):
    """Output dictionary returned by ``GridDetector.update()``.

    Attributes
    ----------
    n_samples : int
        Number of observations seen since the most recent reset.
        This is the detector's local time and is the value that returns to 0
        after a reset.
    alarm : bool
        Whether any score exceeded the threshold at this time step.
    max_score : np.ndarray
        Maximum penalized score across grid candidates, shape ``(K,)``.
        Single-test scores use ``K=1``.
    max_split_point : np.ndarray
        Split location (pre-change sample count) achieving the max score,
        computed as ``state.grid[argmax]`` for each test, shape ``(K,)``.
        Single-test scores use ``K=1``.

        For ``n_samples < 2``, no candidate scores are available yet and this
        field is a placeholder zero vector of shape ``(K,)``.
    """

    n_samples: int
    alarm: bool
    max_score: np.ndarray
    max_split_point: np.ndarray


@runtime_checkable
class ScoreModel(Protocol[TScoreState]):
    """Protocol for score computers used within the grid detector.

    A compliant class maintains per-candidate sufficient statistics and computes
    scores for all active candidates after each new observation.

    IMPORTANT: state objects returned by ``init_state``/``update`` are treated as
    immutable snapshots by ``GridDetector``. ``update`` must return a new state
    and must not mutate the input ``state`` in place.

    IMPORTANT: ``GridDetector`` calls ``compute_penalized_scores(state, grid_states)``
    and score implementations should derive any time-dependent penalty scaling
    from the provided state.  Accordingly, ``TScoreState`` must carry whatever
    time information the score needs (typically an ``n_samples`` counter updated
    by ``update``).  No separate ``n_samples_for_penalty`` argument is passed.

    The grid detector calls these methods; the implementation is free to choose
    any backend (NumPy, Numba, pandas, PyTorch, JAX, etc.).
    """

    @property
    def n_features(self) -> int:
        """Observation dimension expected by the score model."""
        ...

    @property
    def n_scores(self) -> int:
        """Number of scores ``K`` returned by ``compute_penalized_scores``.

        This determines the second dimension of the ``(G, K)`` score matrix
        and the length of ``DetectorOutput.max_score`` / ``max_split_point``.
        """
        ...

    def init_state(self) -> TScoreState:
        """Return fresh initial state with no observations seen."""
        ...

    def update(self, state: TScoreState, x: ArrayLike) -> TScoreState:
        """Incorporate a new observation into the global running statistic.

        Parameters
        ----------
        state : TScoreState
            Current state.
        x : ArrayLike
            New observation.

        Returns
        -------
        TScoreState
            Updated state.
        """
        ...

    def compute_penalized_scores(
        self,
        state: TScoreState,
        grid_states: list[TScoreState],
    ) -> np.ndarray:
        """Compute a penalized score for every active grid candidate.

        Parameters
        ----------
        state : TScoreState
            Global running state after the latest observation.  This is the
            authoritative source for any time-dependent penalty scaling; for
            example, built-in scores read ``state.n_samples`` from here.
            Custom ``TScoreState`` types must therefore include an ``n_samples``
            field (or equivalent) if time-dependent penalties are required.
        grid_states : list[TScoreState]
            Per-candidate state snapshots, one per active grid point.

        Returns
        -------
        np.ndarray, shape (len(grid_states), n_scores)
            Penalized score matrix for active candidates. The first dimension is
            the number of candidates ``G`` and the second dimension is the
            number of scores ``K``. Single-score models must return ``(G, 1)``.

        """
        ...
