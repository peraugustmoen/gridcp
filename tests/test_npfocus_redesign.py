"""Reproduction and behavior tests for the reworked NPFOCuS score."""

import numpy as np
import pytest

from gridcp.scores import NPFOCuS
from tests._reference_legacy_kernels import reference_npfocus_score

GRID = np.array([-1.0, 0.0, 1.0])


def _build_states(score, data, splits):
    total = score.init_state()
    for x in data:
        total = score.update(total, x)
    grid = []
    for n1 in splits:
        st = score.init_state()
        for x in data[:n1]:
            st = score.update(st, x)
        grid.append(st)
    return total, grid


def test_max_reproduces_legacy_channel_max_output():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(24, 2))
    splits = [3, 8, 15, 21]
    score = NPFOCuS(value_grid=GRID, n_features=2)  # aggregation="max" default
    total, grid = _build_states(score, data, splits)

    grid_sums = np.stack([st.n_smaller for st in grid])
    before_samples = np.array(splits, dtype=np.int64)
    ref = reference_npfocus_score(
        total.n_smaller, grid_sums, total.n_samples, before_samples
    )

    out = score.compute_penalized_scores(total, grid)
    assert out.shape == (4, 2)
    np.testing.assert_allclose(out, ref)


@pytest.mark.parametrize(
    "aggregation,expected_cols", [("max", 2), ("sum", 2), ("max-sum", 4), (None, 6)]
)
def test_channel_aggregation_shapes(aggregation, expected_cols):
    rng = np.random.default_rng(1)
    data = rng.normal(size=(20, 3))
    splits = [4, 10, 16]
    score = NPFOCuS(value_grid=GRID, n_features=3, aggregation=aggregation)
    assert score.n_scores == expected_cols
    total, grid = _build_states(score, data, splits)
    out = score.compute_penalized_scores(total, grid)
    assert out.shape == (3, expected_cols)


def test_no_enable_penalty_parameter():
    with pytest.raises(TypeError):
        NPFOCuS(value_grid=GRID, enable_penalty=False)  # type: ignore[call-arg]


def test_invalid_aggregation_rejected():
    with pytest.raises(ValueError):
        NPFOCuS(value_grid=GRID, aggregation="biggest")
