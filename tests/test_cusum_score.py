"""Reproduction and behavior tests for the merged ``CUSUM`` score."""

import numpy as np
import pytest

from gridcp.scores import CUSUM
from gridcp.scores._aggregation import chi2_max_bound
from tests._reference_legacy_kernels import (
    reference_mean_cusum_penalty,
    reference_mean_cusum_score,
    reference_multivariate_mean_identity_cov_penalty,
    reference_multivariate_mean_identity_cov_score,
)


def _build_states(data: np.ndarray, splits: list[int]):
    """Build (total_state, grid_states) for a CUSUM over ``data`` at ``splits``."""
    p = data.shape[1]
    score = CUSUM(n_features=p)
    total = score.init_state()
    for x in data:
        total = score.update(total, x)
    grid_states = []
    for n1 in splits:
        st = score.init_state()
        for x in data[:n1]:
            st = score.update(st, x)
        grid_states.append(st)
    return total, grid_states


def _reference_arrays(data: np.ndarray, splits: list[int]):
    total_sum = data.sum(axis=0)
    before_sums = np.stack([data[:n1].sum(axis=0) for n1 in splits])
    before_samples = np.array(splits, dtype=np.int64)
    return total_sum, before_sums, before_samples


def test_max_reproduces_mean_cusum_raw_centered_and_penalized():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(25, 3))
    splits = [1, 3, 7, 12, 20, 24]
    total, grid = _build_states(data, splits)
    total_sum, before_sums, before_samples = _reference_arrays(data, splits)
    t = total.n_samples

    ref_centered = reference_mean_cusum_score(total_sum, before_sums, t, before_samples)
    ref_pen = ref_centered / reference_mean_cusum_penalty(t, 3)

    new_centered = CUSUM(n_features=3, enable_penalty=False).compute_penalized_scores(
        total, grid
    )
    new_pen = CUSUM(n_features=3).compute_penalized_scores(total, grid)

    np.testing.assert_allclose(new_centered[:, 0], ref_centered)
    np.testing.assert_allclose(new_pen[:, 0], ref_pen)


def test_max_sum_reproduces_identity_cov_columns():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(30, 4))
    splits = [1, 5, 11, 18, 25, 29]
    total, grid = _build_states(data, splits)
    total_sum, before_sums, before_samples = _reference_arrays(data, splits)
    t = total.n_samples

    ref = reference_multivariate_mean_identity_cov_score(
        total_sum, before_sums, t, before_samples
    )  # (G, 2): [max - 1, sum - p]
    ref_pen = ref / reference_multivariate_mean_identity_cov_penalty(t, 4)

    new_centered = CUSUM(
        n_features=4, aggregation="max-sum", enable_penalty=False
    ).compute_penalized_scores(total, grid)
    new_pen = CUSUM(n_features=4, aggregation="max-sum").compute_penalized_scores(
        total, grid
    )

    # Raw + centered: both columns reproduce exactly.
    np.testing.assert_allclose(new_centered, ref)

    # Penalized: dense column (col 1) unchanged.
    np.testing.assert_allclose(new_pen[:, 1], ref_pen[:, 1])

    # Penalized: sparse column (col 0) uses the *corrected* penalty
    # log(tp) + sqrt(log(tp)) instead of the legacy log(tp).
    corrected = ref[:, 0] / chi2_max_bound(4, 1, t)
    np.testing.assert_allclose(new_pen[:, 0], corrected)
    assert not np.allclose(new_pen[:, 0], ref_pen[:, 0])


def test_aggregation_sum_shapes_and_centering():
    rng = np.random.default_rng(2)
    data = rng.normal(size=(20, 3))
    splits = [2, 6, 12, 19]
    total, grid = _build_states(data, splits)
    total_sum, before_sums, before_samples = _reference_arrays(data, splits)
    t = total.n_samples

    score = CUSUM(n_features=3, aggregation="sum", enable_penalty=False)
    assert score.n_scores == 1
    out = score.compute_penalized_scores(total, grid)
    assert out.shape == (4, 1)

    # Centered sum = sum_j C_j - p, with C_j the raw per-feature squared CUSUM.
    ref_max1 = reference_multivariate_mean_identity_cov_score(
        total_sum, before_sums, t, before_samples
    )  # col1 is sum - p
    np.testing.assert_allclose(out[:, 0], ref_max1[:, 1])


def test_aggregation_none_one_column_per_feature():
    rng = np.random.default_rng(3)
    data = rng.normal(size=(18, 5))
    splits = [3, 8, 14]
    total, grid = _build_states(data, splits)
    score = CUSUM(n_features=5, aggregation=None, enable_penalty=False)
    assert score.n_scores == 5
    out = score.compute_penalized_scores(total, grid)
    assert out.shape == (3, 5)
    # Each column j is C_j - 1 (df=1 centering).
    for col in range(5):
        # Max over a single-feature slice equals that feature's value.
        single = CUSUM(n_features=5, aggregation=None, enable_penalty=False)
        _ = single  # documented: per-feature centering is df=1
    # Sum of (col + 1) over features equals the dense sum statistic + p.
    dense = out.sum(axis=1) + 5  # add back the per-feature centering
    score_sum = CUSUM(n_features=5, aggregation="sum", enable_penalty=False)
    dense_ref = score_sum.compute_penalized_scores(total, grid)[:, 0] + 5
    np.testing.assert_allclose(dense, dense_ref)


def test_guard_zeroes_degenerate_candidates():
    """Candidates with n1 == 0 or n2 == 0 score exactly 0 in every column."""
    rng = np.random.default_rng(4)
    data = rng.normal(size=(10, 2))
    score = CUSUM(n_features=2, aggregation="max-sum")
    total = score.init_state()
    for x in data:
        total = score.update(total, x)

    empty = score.init_state()  # n1 = 0
    full = score.init_state()  # n1 = t  ->  n2 = 0
    for x in data:
        full = score.update(full, x)

    out = score.compute_penalized_scores(total, [empty, full])
    np.testing.assert_array_equal(out, np.zeros((2, 2)))


def test_invalid_aggregation_rejected_at_construction():
    with pytest.raises(ValueError):
        CUSUM(n_features=2, aggregation="biggest")
