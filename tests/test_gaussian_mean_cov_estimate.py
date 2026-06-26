"""Contract tests for the merged GaussianMean(cov_estimate=...) score."""

import warnings

import numpy as np
import pytest

from gridcp.calibration import calibrate_threshold_false_alarm
from gridcp.scores import BUILTIN_SCORE_TYPES, GaussianMean


def test_default_cov_estimate_is_diagonal():
    assert GaussianMean(n_features=3).cov_estimate == "diagonal"


def test_invalid_cov_estimate_raises():
    with pytest.raises(ValueError):
        GaussianMean(n_features=2, cov_estimate="spherical")


def test_n_scores_full_is_one_regardless_of_aggregation():
    for aggregation in ("max", "sum", "max-sum", None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = GaussianMean(
                n_features=4, cov_estimate="full", aggregation=aggregation
            )
        assert score.n_scores == 1


@pytest.mark.parametrize(
    "aggregation,expected",
    [("max", 1), ("sum", 1), ("max-sum", 2), (None, 4)],
)
def test_n_scores_diagonal_depends_on_aggregation(aggregation, expected):
    score = GaussianMean(n_features=4, cov_estimate="diagonal", aggregation=aggregation)
    assert score.n_scores == expected


def test_output_column_count_matches_n_scores():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(30, 4))
    for cov_estimate, aggregation in [
        ("diagonal", "max"),
        ("diagonal", "max-sum"),
        ("diagonal", None),
        ("full", "max"),
    ]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = GaussianMean(
                n_features=4, cov_estimate=cov_estimate, aggregation=aggregation
            )
        total = score.init_state()
        for x in data:
            total = score.update(total, x)
        grid = []
        for n1 in (6, 14, 22):
            st = score.init_state()
            for x in data[:n1]:
                st = score.update(st, x)
            grid.append(st)
        out = score.compute_penalized_scores(total, grid)
        assert out.shape[1] == score.n_scores


def test_full_with_non_default_aggregation_warns():
    with pytest.warns(UserWarning, match="aggregation is ignored"):
        score = GaussianMean(n_features=3, cov_estimate="full", aggregation="sum")
    assert score.n_scores == 1


def test_full_with_default_aggregation_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes an error
        GaussianMean(n_features=3, cov_estimate="full")  # default aggregation="max"


def test_diagonal_state_is_o_p_no_pxp_matrix():
    """Diagonal mode keeps O(p) state and never allocates the p×p outer product."""
    p = 5
    score = GaussianMean(n_features=p, cov_estimate="diagonal")
    state = score.init_state()
    assert state.sum_outer.size == 0  # no p×p matrix populated
    assert state.stats.shape == (2, p)
    rng = np.random.default_rng(1)
    for x in rng.normal(size=(10, p)):
        state = score.update(state, x)
    assert state.sum_outer.size == 0
    assert state.stats.shape == (2, p)


def test_full_state_carries_pxp_matrix():
    p = 4
    score = GaussianMean(n_features=p, cov_estimate="full")
    state = score.init_state()
    assert state.sum_outer.shape == (p, p)


def test_full_covariance_class_is_removed():
    with pytest.raises(ImportError):
        from gridcp.scores import GaussianMeanFullCovariance  # noqa: F401


def test_builtin_score_types_drops_full_covariance():
    names = {cls.__name__ for cls in BUILTIN_SCORE_TYPES}
    assert "GaussianMeanFullCovariance" not in names
    assert "GaussianMean" in names
    assert len(BUILTIN_SCORE_TYPES) == 9


def _gaussian_sampler(rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=3)


def test_calibration_sizes_threshold_from_n_scores():
    # diagonal max-sum -> n_scores == 2 -> threshold of length 2.
    diag = GaussianMean(n_features=3, cov_estimate="diagonal", aggregation="max-sum")
    th_diag = calibrate_threshold_false_alarm(
        score=diag,
        false_alarm_probability=0.1,
        n_paths=40,
        stream_len=30,
        pre_sampler=_gaussian_sampler,
        rng=0,
        parallel=False,
    )
    assert np.asarray(th_diag).shape == (2,)

    # full -> n_scores == 1.
    full = GaussianMean(n_features=3, cov_estimate="full")
    th_full = calibrate_threshold_false_alarm(
        score=full,
        false_alarm_probability=0.1,
        n_paths=40,
        stream_len=30,
        pre_sampler=_gaussian_sampler,
        rng=0,
        parallel=False,
    )
    assert np.asarray(th_full).reshape(-1).shape == (1,)
