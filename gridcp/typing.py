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
        after initialization or reset.
    alarm : bool
        Whether any score exceeds the threshold at the current time step.
    max_score : np.ndarray
        Maximum penalized score across grid candidates, shape ``(n_scores,)``.
        ``n_scores`` is the number of penalized scores returned by
        ``compute_penalized_scores``.  Single-score models use ``n_scores=1``.
    max_split_point : np.ndarray
        First post-change index (0-based) achieving the max score, computed as
        ``state.grid[argmax]`` for each score, shape ``(n_scores,)``.

        For a value ``n1``, ``data[0:n1]`` is the pre-change segment and
        ``data[n1:]`` is the post-change segment.

        For ``n_samples == 1``, no candidate scores are available yet and this
        field is a placeholder zero vector of shape ``(n_scores,)``.
    """

    n_samples: int
    alarm: bool
    max_score: np.ndarray
    max_split_point: np.ndarray


@runtime_checkable
class ScoreModel(Protocol[TScoreState]):
    """Protocol for score models used within the grid detector.

    A compliant class maintains per-candidate sufficient statistics and computes
    scores for all active candidates after each new observation.

    IMPORTANT: state objects returned by ``init_state``/``update`` are treated as
    immutable snapshots by ``GridDetector``. ``update`` must return a new state
    and must not mutate the input ``state`` in place.

    IMPORTANT: ``GridDetector`` calls ``compute_penalized_scores(state, grid_states)``
    and score implementations should derive any time-dependent penalty scaling
    from the provided state.  Accordingly, ``TScoreState`` must carry whatever
    time information the score needs (typically an ``n_samples`` counter updated
    by ``update``).
    """

    @property
    def n_features(self) -> int:
        """Observation dimension expected by the score model."""
        ...

    @property
    def n_scores(self) -> int:
        """Number of penalized scores returned by ``compute_penalized_scores``.

        This determines the second dimension of the ``(G, n_scores)`` score
        matrix and the length of ``DetectorOutput.max_score`` /
        ``max_split_point``.
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
        np.ndarray, shape ``(G, n_scores)``
            Penalized score matrix for active candidates, where ``G =
            len(grid_states)`` is the number of active candidates and
            ``n_scores`` is the number of penalized scores.  Single-score
            models must return ``(G, 1)``.

        """
        ...
