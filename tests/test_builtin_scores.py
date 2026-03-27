import numpy as np

from gridcp.detector import GridDetector
from gridcp.scores import (
    MultivariateMeanIdentityCov,
    MultivariateMeanOrCovariance,
    MultivariateMeanUnknownCov,
)
from gridcp.utils import get_changeloc_grid


def _run_stream(detector: GridDetector, x: np.ndarray):
    state = detector.init_state()
    outputs = []
    for row in x:
        state, out = detector.update(state, row)
        outputs.append(out)
    return state, outputs


def test_multivariate_mean_known_var_alarm():
    """No alarm under null, alarm after extreme observation."""
    n = 100
    p = 10

    rng = np.random.default_rng(123)
    x = rng.normal(loc=0.0, scale=1.0, size=(n, p))

    detector = GridDetector(
        score=MultivariateMeanIdentityCov(n_features=p),
        threshold=5.0,
    )
    state, outputs = _run_stream(detector, x)

    assert not any(out["alarm"] for out in outputs)

    state, out = detector.update(state, np.ones(p, dtype=np.float64) * 100000.0)
    assert out["alarm"]


def test_multivariate_mean_known_var_cumsums_and_scores():
    """Prefix sums and raw LR scores match direct formula."""
    n = 100
    p = 10

    rng = np.random.default_rng(123)
    x = rng.normal(loc=0.0, scale=1.0, size=(n, p))

    detector = GridDetector(
        score=MultivariateMeanIdentityCov(n_features=p),
        threshold=100.0,
    )
    state, _ = _run_stream(detector, x)

    cumsums = np.cumsum(x, axis=0)
    true_grid = get_changeloc_grid(n)

    assert np.array_equal(np.asarray(state.grid, dtype=np.int64), true_grid)

    for i, n1 in enumerate(state.grid):
        expected_sum = cumsums[n1 - 1]
        actual_sum = state.candidate_score_states[i].sum
        assert np.allclose(actual_sum, expected_sum)

        n2 = n - n1
        total_sum = state.running_score_state.sum
        mean1 = expected_sum / n1
        mean2 = (total_sum - expected_sum) / n2
        diff = mean1 - mean2
        lr = (n1 * n2 / n) * float(np.dot(diff, diff))
        expected_raw = lr - p

        score = detector.score.compute_penalised_scores(
            state.running_score_state,
            state.candidate_score_states,
        )[i]
        penalty = np.log(n) + np.sqrt(p * np.log(n))
        assert np.isclose(score, expected_raw / penalty)


def test_multivariate_mean_known_var_reset_semantics():
    """Reinitializing state gives a fresh detector run."""
    p = 5
    n = 50

    rng = np.random.default_rng(42)
    x = rng.normal(loc=0.0, scale=1.0, size=(n, p))

    detector = GridDetector(
        score=MultivariateMeanIdentityCov(n_features=p),
        threshold=5.0,
    )
    state, _ = _run_stream(detector, x)
    assert state.n_samples == n

    fresh_state = detector.init_state()
    assert fresh_state.n_samples == 0
    assert fresh_state.grid == []


def test_multivariate_mean_unknown_var_alarm():
    """No alarm under null, alarm after extreme observation."""
    n = 100
    p = 5

    rng = np.random.default_rng(123)
    x = rng.normal(loc=0.0, scale=1.0, size=(n, p))

    detector = GridDetector(
        score=MultivariateMeanUnknownCov(n_features=p),
        threshold=5.0,
    )
    state, outputs = _run_stream(detector, x)

    assert not any(out["alarm"] for out in outputs)

    state, out = detector.update(state, np.ones(p, dtype=np.float64) * 100000.0)
    assert out["alarm"]


def test_multivariate_mean_unknown_var_cumsums():
    """Stored prefix sums and second moments match manual cumulative sums."""
    n = 100
    p = 5

    rng = np.random.default_rng(123)
    x = rng.normal(loc=0.0, scale=1.0, size=(n, p))

    detector = GridDetector(
        score=MultivariateMeanUnknownCov(n_features=p),
        threshold=100.0,
    )
    state, _ = _run_stream(detector, x)

    cumsums = np.cumsum(x, axis=0)
    cum_outers = np.cumsum(
        np.stack([np.outer(x[i], x[i]) for i in range(n)], axis=0),
        axis=0,
    )

    true_grid = get_changeloc_grid(n)
    assert np.array_equal(np.asarray(state.grid, dtype=np.int64), true_grid)

    for i, n1 in enumerate(state.grid):
        expected_sum = cumsums[n1 - 1]
        expected_outer = cum_outers[n1 - 1]
        actual = state.candidate_score_states[i]
        assert np.allclose(actual.sum, expected_sum)
        assert np.allclose(actual.sum_outer, expected_outer)

    assert np.allclose(state.running_score_state.sum, cumsums[-1])
    assert np.allclose(state.running_score_state.sum_outer, cum_outers[-1])


def test_multivariate_mean_unknown_var_no_false_alarm():
    """Under null with high threshold, no alarm should occur."""
    n = 200
    p = 5

    rng = np.random.default_rng(999)
    x = rng.normal(loc=0.0, scale=1.0, size=(n, p))

    detector = GridDetector(
        score=MultivariateMeanUnknownCov(n_features=p),
        threshold=50.0,
    )
    _, outputs = _run_stream(detector, x)

    assert not any(out["alarm"] for out in outputs)


def test_multivariate_mean_or_covariance_alarm():
    """No alarm under null, alarm after extreme observation."""
    n = 100
    p = 5

    rng = np.random.default_rng(123)
    x = rng.normal(loc=0.0, scale=1.0, size=(n, p))

    detector = GridDetector(
        score=MultivariateMeanOrCovariance(n_features=p),
        threshold=5.0,
    )
    state, outputs = _run_stream(detector, x)

    assert not any(out["alarm"] for out in outputs)

    state, out = detector.update(state, np.ones(p, dtype=np.float64) * 100000.0)
    assert out["alarm"]


def test_multivariate_mean_or_covariance_cumsums():
    """Stored prefix sums and second moments match manual cumulative sums."""
    n = 100
    p = 5

    rng = np.random.default_rng(123)
    x = rng.normal(loc=0.0, scale=1.0, size=(n, p))

    detector = GridDetector(
        score=MultivariateMeanOrCovariance(n_features=p),
        threshold=100.0,
    )
    state, _ = _run_stream(detector, x)

    cumsums = np.cumsum(x, axis=0)
    cum_outers = np.cumsum(
        np.stack([np.outer(x[i], x[i]) for i in range(n)], axis=0),
        axis=0,
    )

    true_grid = get_changeloc_grid(n)
    assert np.array_equal(np.asarray(state.grid, dtype=np.int64), true_grid)

    for i, n1 in enumerate(state.grid):
        expected_sum = cumsums[n1 - 1]
        expected_outer = cum_outers[n1 - 1]
        actual = state.candidate_score_states[i]
        assert np.allclose(actual.sum, expected_sum)
        assert np.allclose(actual.sum_outer, expected_outer)

    assert np.allclose(state.running_score_state.sum, cumsums[-1])
    assert np.allclose(state.running_score_state.sum_outer, cum_outers[-1])


def test_multivariate_mean_or_covariance_no_false_alarm():
    """Under null with high threshold, no alarm should occur."""
    n = 200
    p = 5

    rng = np.random.default_rng(999)
    x = rng.normal(loc=0.0, scale=1.0, size=(n, p))

    detector = GridDetector(
        score=MultivariateMeanOrCovariance(n_features=p),
        threshold=50.0,
    )
    _, outputs = _run_stream(detector, x)

    assert not any(out["alarm"] for out in outputs)
