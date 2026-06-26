"""Tests for the shared aggregation/penalty helpers in ``_aggregation.py``."""

import numpy as np
import pytest

from gridcp.scores._aggregation import (
    aggregate_features,
    aggregation_dims,
    aggregation_mode_code,
    chi2_max_bound,
)


def test_chi2_max_bound_max_form():
    """``chi2_max_bound(p, d, t)`` equals ``log(tp) + sqrt(d log(tp))``."""
    for p, d, t in [(3, 1, 50), (1, 4, 7), (5, 2, 1000)]:
        logg = np.log(t * p)
        expected = logg + np.sqrt(d * logg)
        assert chi2_max_bound(p, d, t) == pytest.approx(expected)


def test_chi2_max_bound_single_statistic_form():
    """``chi2_max_bound(1, D, t)`` equals ``log(t) + sqrt(D log(t))``."""
    for d, t in [(1, 10), (7, 200)]:
        logg = np.log(t)
        expected = logg + np.sqrt(d * logg)
        assert chi2_max_bound(1, d, t) == pytest.approx(expected)


def test_chi2_max_bound_matches_legacy_cusum_penalty():
    """CUSUM-max penalty ``log(tp) + sqrt(log(tp))`` is ``chi2_max_bound(p, 1, t)``."""
    t, p = 123, 4
    logg = np.log(t * p)
    assert chi2_max_bound(p, 1, t) == pytest.approx(logg + np.sqrt(logg))


def test_aggregate_features_max_matches_sequential_row_max():
    rng = np.random.default_rng(0)
    stats = rng.normal(size=(8, 5))
    cols = aggregate_features(stats, aggregation_mode_code("max"))
    assert cols.shape == (8, 1)
    np.testing.assert_allclose(cols[:, 0], stats.max(axis=1))


def test_aggregate_features_sum_matches_row_sum():
    rng = np.random.default_rng(1)
    stats = rng.normal(size=(6, 4))
    cols = aggregate_features(stats, aggregation_mode_code("sum"))
    assert cols.shape == (6, 1)
    np.testing.assert_allclose(cols[:, 0], stats.sum(axis=1))


def test_aggregate_features_max_sum_columns():
    rng = np.random.default_rng(2)
    stats = rng.normal(size=(6, 3))
    cols = aggregate_features(stats, aggregation_mode_code("max-sum"))
    assert cols.shape == (6, 2)
    np.testing.assert_allclose(cols[:, 0], stats.max(axis=1))
    np.testing.assert_allclose(cols[:, 1], stats.sum(axis=1))


def test_aggregate_features_none_is_identity():
    rng = np.random.default_rng(3)
    stats = rng.normal(size=(5, 4))
    cols = aggregate_features(stats, aggregation_mode_code(None))
    assert cols.shape == (5, 4)
    np.testing.assert_allclose(cols, stats)


def test_aggregation_dims_table():
    p, d = 4, 2
    assert aggregation_dims("max", p, d) == [(p, d)]
    assert aggregation_dims("sum", p, d) == [(1, p * d)]
    assert aggregation_dims("max-sum", p, d) == [(p, d), (1, p * d)]
    assert aggregation_dims(None, p, d) == [(1, d)] * p


def test_aggregation_none_string_is_identical_to_none():
    p, d = 3, 1
    assert aggregation_dims("None", p, d) == aggregation_dims(None, p, d)
    assert aggregation_mode_code("None") == aggregation_mode_code(None)


def test_invalid_aggregation_raises_value_error():
    for bad in ["Max", "average", 5, "maximum"]:
        with pytest.raises(ValueError):
            aggregation_dims(bad, 3, 1)
        with pytest.raises(ValueError):
            aggregation_mode_code(bad)
