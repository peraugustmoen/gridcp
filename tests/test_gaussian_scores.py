"""Reproduction and behavior tests for the per-feature Gaussian LR scores."""

import numpy as np
import pytest

from gridcp.scores import GaussianMean, GaussianMeanOrVariance, GaussianVariance
from gridcp.scores._aggregation import chi2_max_bound
from tests._reference_legacy_kernels import (
    reference_mean_or_variance_score,
    reference_mean_unknown_variance_score,
    reference_variance_score,
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


def test_gaussian_mean_max_reproduces_legacy():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(30, 3))
    splits = [4, 9, 15, 22, 29]
    score = GaussianMean(n_features=3)
    total, grid = _build_states(score, data, splits)
    t = total.n_samples

    total_stats = np.vstack((data.sum(axis=0), (data**2).sum(axis=0)))
    before_stats = np.stack(
        [
            np.vstack((data[:n1].sum(axis=0), (data[:n1] ** 2).sum(axis=0)))
            for n1 in splits
        ]
    )
    before_samples = np.array(splits, dtype=np.int64)
    ref = reference_mean_unknown_variance_score(
        total_stats, before_stats, t, before_samples
    )

    centered = GaussianMean(
        n_features=3, enable_penalty=False
    ).compute_penalized_scores(total, grid)
    penalized = score.compute_penalized_scores(total, grid)
    np.testing.assert_allclose(centered[:, 0], ref)
    np.testing.assert_allclose(penalized[:, 0], ref / chi2_max_bound(3, 1, t))


def test_gaussian_variance_max_reproduces_legacy():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(28, 2))
    splits = [3, 8, 14, 21, 27]
    score = GaussianVariance(n_features=2)
    total, grid = _build_states(score, data, splits)
    t = total.n_samples

    total_sum_sq = (data**2).sum(axis=0)
    before_sum_sqs = np.stack([(data[:n1] ** 2).sum(axis=0) for n1 in splits])
    before_samples = np.array(splits, dtype=np.int64)
    ref = reference_variance_score(total_sum_sq, before_sum_sqs, t, before_samples)

    centered = GaussianVariance(
        n_features=2, enable_penalty=False
    ).compute_penalized_scores(total, grid)
    penalized = score.compute_penalized_scores(total, grid)
    np.testing.assert_allclose(centered[:, 0], ref)
    np.testing.assert_allclose(penalized[:, 0], ref / chi2_max_bound(2, 1, t))


def test_gaussian_mean_or_variance_max_reproduces_legacy():
    rng = np.random.default_rng(2)
    data = rng.normal(size=(32, 3))
    splits = [4, 10, 17, 24, 31]
    score = GaussianMeanOrVariance(n_features=3)
    total, grid = _build_states(score, data, splits)
    t = total.n_samples

    total_sum = data.sum(axis=0)
    total_sum_sq = (data**2).sum(axis=0)
    before_sums = np.stack([data[:n1].sum(axis=0) for n1 in splits])
    before_sum_sqs = np.stack([(data[:n1] ** 2).sum(axis=0) for n1 in splits])
    before_samples = np.array(splits, dtype=np.int64)
    ref = reference_mean_or_variance_score(
        total_sum, total_sum_sq, before_sums, before_sum_sqs, t, before_samples
    )

    centered = GaussianMeanOrVariance(
        n_features=3, enable_penalty=False
    ).compute_penalized_scores(total, grid)
    penalized = score.compute_penalized_scores(total, grid)
    np.testing.assert_allclose(centered[:, 0], ref)
    np.testing.assert_allclose(penalized[:, 0], ref / chi2_max_bound(3, 2, t))


@pytest.mark.parametrize(
    "score_cls",
    [GaussianMean, GaussianVariance, GaussianMeanOrVariance],
)
def test_new_aggregations_shapes_and_n_scores(score_cls):
    rng = np.random.default_rng(3)
    data = rng.normal(size=(25, 4))
    splits = [5, 12, 20]
    for aggregation, expected_cols in [
        ("max", 1),
        ("sum", 1),
        ("max-sum", 2),
        (None, 4),
    ]:
        score = score_cls(n_features=4, aggregation=aggregation, enable_penalty=False)
        assert score.n_scores == expected_cols
        total, grid = _build_states(score, data, splits)
        out = score.compute_penalized_scores(total, grid)
        assert out.shape == (3, expected_cols)


def test_guard_zeroes_small_and_degenerate_candidates():
    # Small candidates (n1 <= 2) and an all-degenerate (constant) candidate -> 0.
    rng = np.random.default_rng(4)
    data = rng.normal(size=(12, 1))
    score = GaussianMean(n_features=1)
    total, grid = _build_states(score, data, [1, 2, 6])
    out = score.compute_penalized_scores(total, grid)
    # n1 = 1 and n1 = 2 are guarded.
    np.testing.assert_array_equal(out[:2, 0], np.zeros(2))

    # All-zero data -> zero-mean variance estimate 0 -> degenerate -> score 0.
    zeros = np.zeros((12, 1))
    vscore = GaussianVariance(n_features=1)
    vtotal, vgrid = _build_states(vscore, zeros, [3, 6, 9])
    vout = vscore.compute_penalized_scores(vtotal, vgrid)
    np.testing.assert_array_equal(vout, np.zeros((3, 1)))
