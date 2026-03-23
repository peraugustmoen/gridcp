import numpy as np
import pytest

from gridcp.utils import get_changeloc_grid, v2


def test_v2_matches_expected_trailing_zero_counts():
    assert v2(0) == -1
    assert v2(1) == 0
    assert v2(2) == 1
    assert v2(4) == 2
    assert v2(8) == 3
    assert v2(12) == 2
    assert v2(-8) == 3


def test_get_changeloc_grid_t1_baseline():
    out = get_changeloc_grid(1)
    assert np.array_equal(out, np.array([0], dtype=np.int64))


def test_get_changeloc_grid_raises_for_invalid_t():
    with pytest.raises(ValueError, match="positive integer"):
        get_changeloc_grid(0)
