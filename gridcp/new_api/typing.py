"""Typing definitions for the grid detector API."""

from typing import Protocol, TypeVar, runtime_checkable
import numpy as np
from numpy.typing import ArrayLike

ScoreState = TypeVar("ScoreState")


@runtime_checkable
class GridScore(Protocol[ScoreState]):
    """Protocol for score computers used within the grid detector.

    A compliant class maintains per-candidate sufficient statistics and computes
    scores for all active candidates after each new observation. It owns:
      - a state type ScoreState that holds each per-candidate running statistics
      - the logic for initialising, updating, and computing penalised scores from
        these states.

    The grid detector calls these methods; the implementation is free to choose
    any backend (NumPy, Numba, pandas, PyTorch, JAX, etc.).
    """

    def init_state(self) -> ScoreState:
        """Return a fresh initial state with no observations seen."""
        ...

    def update(self, state: ScoreState, x: ArrayLike) -> ScoreState:
        """Incorporate a new observation into the global running statistic.

        Parameters
        ----------
        state : ScoreState
            Current state.
        x : ArrayLike
            New observation.

        Returns
        -------
        ScoreState
            Updated state.
        """
        ...

    def compute_penalised_scores(
        self,
        state: ScoreState,
        grid_states: list[ScoreState],
    ) -> np.ndarray:
        """Compute a penalised score for every active grid candidate.

        Parameters
        ----------
        state : ScoreState
            Global running state after the latest observation.
        grid_states : list[ScoreState]
            Per-candidate state snapshots, one per active grid point.

        Returns
        -------
        np.ndarray, shape (len(grid_states),)
            Penalised score for each active candidate.
        """
        ...
