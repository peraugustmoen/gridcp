from gridcp import core
import numpy as np
import numba as nb
from numba.typed import List
from numba.core.registry import CPUDispatcher
import pytest


def test_grid():
    grid = List.empty_list(nb.int64)
    T = 1000
    for t in range(1, T + 1):
        _ = core.update_grid_numba(grid, t)

    true_grid = core.get_grid(T)
    assert len(grid) == len(true_grid)

    # reverse true_grid to match the order in grid
    true_grid = true_grid[::-1]
    for g1, g2 in zip(grid, true_grid):
        assert g1 + T + 1 == g2, f"Expected {g2}, got {g1 + T + 1}"


def test_get_grid_edge_cases():
    """Test that get_grid handles edge cases (t <= 1) correctly per documentation."""
    for t in [-100, -1, 0, 1]:
        result = core.get_grid(t)
        assert len(result) == 1, f"Expected length 1 for t={t}, got {len(result)}"
        assert result[0] == 1, f"Expected result[0]=1 for t={t}, got {result[0]}"
        assert (
            result.dtype == np.int64
        ), f"Expected dtype int64 for t={t}, got {result.dtype}"


@pytest.mark.parametrize("t", [2, 5, 10, 100, 1000])
def test_get_grid_normal_cases(t):
    """Test that get_grid computes grid correctly for t > 1."""
    result = core.get_grid(t)
    assert isinstance(result, np.ndarray), f"Expected ndarray for t={t}"
    assert result.dtype == np.int64, f"Expected dtype int64 for t={t}"
    assert len(result) > 0, f"Expected non-empty grid for t={t}"
    assert result[0] == 1, f"Expected first element to be 1 for t={t}"
    assert np.all(result > 0), f"Expected all grid points > 0 for t={t}"
