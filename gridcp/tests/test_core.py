from gridcp import core
import numpy as np
import numba as nb
from numba.typed import List
import math
from numba.core.registry import CPUDispatcher
from gridcp.utils import v2_numba, LOG2


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
