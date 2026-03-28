"""Typing definitions for the grid detector API."""

from typing import Protocol, TypeVar, runtime_checkable
import numpy as np
from numpy.typing import ArrayLike

TScoreState = TypeVar("TScoreState")


@runtime_checkable
class ScoreModel(Protocol[TScoreState]):
    """Protocol for score computers used within the grid detector.

    A compliant class maintains per-candidate sufficient statistics and computes
    scores for all active candidates after each new observation. It owns:
            - a state type TScoreState that holds each per-candidate running statistics
      - the logic for initialising, updating, and computing penalised scores from
        these states.

    The grid detector calls these methods; the implementation is free to choose
    any backend (NumPy, Numba, pandas, PyTorch, JAX, etc.).
    """

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

    def compute_penalised_scores(
        self,
        state: TScoreState,
        grid_states: list[TScoreState],
    ) -> np.ndarray:
        """Compute a penalised score for every active grid candidate.

        Parameters
        ----------
        state : TScoreState
            Global running state after the latest observation.
        grid_states : list[TScoreState]
            Per-candidate state snapshots, one per active grid point.

        Returns
        -------
        np.ndarray, shape (len(grid_states),) or (len(grid_states), n_tests)
            Penalised score for each active candidate.  When the score model
            produces a single test statistic the shape is ``(G,)``.  When the
            model produces multiple test statistics (e.g. separate tests for
            mean vs variance), the shape is ``(G, K)`` where ``K`` is the
            number of tests.
        """
        ...
