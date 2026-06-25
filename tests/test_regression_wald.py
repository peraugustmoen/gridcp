"""Tests for the corrected two-sample Wald regression score."""

import numpy as np
import pytest

from gridcp.scores import RegressionWald
from gridcp.scores._aggregation import chi2_max_bound
from gridcp.scores._regression_wald import regression_wald_score


def _segment_stats(x: np.ndarray, y: np.ndarray):
    """Return (M = XᵀX, S = Xᵀy) for a design block."""
    return x.T @ x, x.T @ y


def _wald_for_split(x: np.ndarray, y: np.ndarray, n1: int, q: int) -> float:
    """Uncentered D̃_g for a single split via the kernel (centered + q)."""
    t = x.shape[0]
    m_tot, s_tot = _segment_stats(x, y)
    m_pre, s_pre = _segment_stats(x[:n1], y[:n1])
    centered = regression_wald_score(
        s_tot,
        m_tot,
        s_pre[np.newaxis, :],
        m_pre[np.newaxis, :, :],
        t,
        np.array([n1], dtype=np.int64),
        q,
    )
    return float(centered[0]) + q


def test_statistic_is_coefficient_free():
    """Same X and ε but different β give an identical statistic (β-free)."""
    rng = np.random.default_rng(0)
    t, q, n1 = 30, 3, 7
    x = rng.normal(size=(t, q))
    eps = rng.normal(size=t)
    beta_a = np.array([1.0, -2.0, 0.5])
    beta_b = np.array([10.0, 4.0, -7.0])
    y_a = x @ beta_a + eps
    y_b = x @ beta_b + eps
    assert _wald_for_split(x, y_a, n1, q) == pytest.approx(
        _wald_for_split(x, y_b, n1, q)
    )


def test_null_distribution_is_chi2_q_unbalanced_split():
    """Monte-Carlo: uncentered D̃_g has mean ≈ q and variance ≈ 2q under H0."""
    rng = np.random.default_rng(1)
    t, q, n1 = 24, 2, 5  # deliberately unbalanced (n1=5, n2=19)
    beta = np.array([3.0, -1.5])  # nonzero true coefficient
    n_sims = 6000
    stats = np.empty(n_sims)
    for s in range(n_sims):
        x = rng.normal(size=(t, q))
        eps = rng.normal(size=t)
        y = x @ beta + eps
        stats[s] = _wald_for_split(x, y, n1, q)

    # chi-squared(q): mean q, variance 2q.
    assert stats.mean() == pytest.approx(q, abs=0.15)
    assert stats.var() == pytest.approx(2 * q, abs=0.6)


def test_guard_returns_zero_below_rank():
    rng = np.random.default_rng(2)
    q = 3
    score = RegressionWald(n_regressors=q)
    data = rng.normal(size=(12, q + 1))
    total = score.init_state()
    for row in data:
        total = score.update(total, row)
    # n1 = 2 < q -> guarded; n1 = 10, n2 = 2 < q -> guarded.
    low = score.init_state()
    for row in data[:2]:
        low = score.update(low, row)
    high = score.init_state()
    for row in data[:10]:
        high = score.update(high, row)
    out = score.compute_penalized_scores(total, [low, high])
    np.testing.assert_array_equal(out, np.zeros((2, 1)))


def test_n_scores_centering_and_penalty():
    rng = np.random.default_rng(3)
    q = 2
    score = RegressionWald(n_regressors=q)
    assert score.n_scores == 1

    data = rng.normal(size=(20, q + 1))
    total = score.init_state()
    for row in data:
        total = score.update(total, row)
    n1 = 8
    grid = score.init_state()
    for row in data[:n1]:
        grid = score.update(grid, row)

    centered = RegressionWald(
        n_regressors=q, enable_penalty=False
    ).compute_penalized_scores(total, [grid])
    penalized = score.compute_penalized_scores(total, [grid])
    np.testing.assert_allclose(
        penalized[:, 0], centered[:, 0] / chi2_max_bound(1, q, total.n_samples)
    )


def test_rejects_aggregation_keyword():
    with pytest.raises(TypeError):
        RegressionWald(n_regressors=2, aggregation="max")  # type: ignore[call-arg]
