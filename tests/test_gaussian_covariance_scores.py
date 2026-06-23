"""Reproduction tests for the self-aggregating Gaussian covariance scores."""

import numpy as np
import pytest

from gridcp.scores import GaussianMeanFullCovariance, GaussianMeanOrCovariance
from gridcp.scores._aggregation import chi2_max_bound
from tests._reference_legacy_kernels import (
    reference_multivariate_mean_or_covariance_penalty,
    reference_multivariate_mean_or_covariance_score,
    reference_multivariate_mean_unknown_cov_penalty,
    reference_multivariate_mean_unknown_cov_score,
)


def _build_states(score, data: np.ndarray, splits: list[int]):
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


def _ref_arrays(data, splits):
    total_sum = data.sum(axis=0)
    total_outer = data.T @ data
    before_sums = np.stack([data[:n1].sum(axis=0) for n1 in splits])
    before_outers = np.stack([data[:n1].T @ data[:n1] for n1 in splits])
    before_samples = np.array(splits, dtype=np.int64)
    return total_sum, total_outer, before_sums, before_outers, before_samples


def test_full_covariance_reproduces_legacy():
    rng = np.random.default_rng(0)
    p = 2
    data = rng.normal(size=(40, p))
    splits = [8, 16, 24, 32, 39]
    score = GaussianMeanFullCovariance(n_features=p)
    total, grid = _build_states(score, data, splits)
    t = total.n_samples
    ts, to, bs, bo, bn = _ref_arrays(data, splits)

    ref = reference_multivariate_mean_unknown_cov_score(ts, to, bs, bo, t, bn, p)
    ref_pen = ref / reference_multivariate_mean_unknown_cov_penalty(t, p)

    centered = GaussianMeanFullCovariance(
        n_features=p, enable_penalty=False
    ).compute_penalized_scores(total, grid)
    penalized = score.compute_penalized_scores(total, grid)

    np.testing.assert_allclose(centered[:, 0], ref)
    np.testing.assert_allclose(penalized[:, 0], ref_pen)
    # chi2_max_bound(1, p, t) matches the legacy divisor exactly.
    assert chi2_max_bound(1, p, t) == pytest.approx(
        reference_multivariate_mean_unknown_cov_penalty(t, p)
    )


def test_mean_or_covariance_reproduces_legacy():
    rng = np.random.default_rng(1)
    p = 2
    data = rng.normal(size=(40, p))
    splits = [6, 12, 20, 28, 36]
    score = GaussianMeanOrCovariance(n_features=p)
    total, grid = _build_states(score, data, splits)
    t = total.n_samples
    ts, to, bs, bo, bn = _ref_arrays(data, splits)

    ref = reference_multivariate_mean_or_covariance_score(ts, to, bs, bo, t, bn, p)
    ref_pen = ref / reference_multivariate_mean_or_covariance_penalty(t, p)
    df = p + p * (p + 1) // 2

    centered = GaussianMeanOrCovariance(
        n_features=p, enable_penalty=False
    ).compute_penalized_scores(total, grid)
    penalized = score.compute_penalized_scores(total, grid)

    np.testing.assert_allclose(centered[:, 0], ref)
    np.testing.assert_allclose(penalized[:, 0], ref_pen)
    assert chi2_max_bound(1, df, t) == pytest.approx(
        reference_multivariate_mean_or_covariance_penalty(t, p)
    )


@pytest.mark.parametrize(
    "score_cls", [GaussianMeanFullCovariance, GaussianMeanOrCovariance]
)
def test_self_aggregating_scores_reject_aggregation_keyword(score_cls):
    with pytest.raises(TypeError):
        score_cls(n_features=2, aggregation="max")  # type: ignore[call-arg]
