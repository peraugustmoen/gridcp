"""Tests for enable_penalty behavior and apply_bonferroni features."""

import numpy as np
import pytest

from gridcp.calibration import calibrate_threshold_false_alarm
from gridcp.detector import GridDetector
from gridcp.scores import (
    CUSUM,
    GaussianMean,
    GaussianMeanOrVariance,
    GaussianMeanOrCovariance,
    GaussianMeanFullCovariance,
    RegressionWald,
    RegressionMcScan,
    GaussianVariance,
)
from gridcp.scores._aggregation import chi2_max_bound


def _run_stream(detector, x):
    """Feed observations through a detector and return final state + outputs."""
    state = detector.init_state()
    outputs = []
    for row in x:
        state, out = detector.update(state, row)
        outputs.append(out)
    return state, outputs


def _prefix_state(score, data, n1):
    """Build a score state from the first ``n1`` observations of ``data``."""
    st = score.init_state()
    for x in data[:n1]:
        st = score.update(st, x)
    return st


# ---------------------------------------------------------------------------
# enable_penalty tests
# ---------------------------------------------------------------------------


class TestEnablePenalty:
    """Verify that enable_penalty=False works and differs from enabled mode."""

    @pytest.mark.parametrize(
        "score_cls, kwargs",
        [
            (CUSUM, dict(n_features=1)),
            (GaussianMean, dict(n_features=1)),
            (GaussianMeanOrVariance, dict(n_features=1)),
            (GaussianVariance, dict(n_features=1)),
            (CUSUM, dict(n_features=3, aggregation="max-sum")),
            (GaussianMeanFullCovariance, dict(n_features=3)),
            (GaussianMeanOrCovariance, dict(n_features=3)),
            (RegressionWald, dict(n_regressors=2)),
            (RegressionMcScan, dict(n_regressors=2)),
        ],
        ids=lambda x: x.__name__ if isinstance(x, type) else str(x),
    )
    def test_constant_penalty_produces_finite_scores(self, score_cls, kwargs):
        """Disabled penalty mode should produce finite scores without errors."""
        score = score_cls(**kwargs, enable_penalty=False)
        n_features = getattr(score, "n_features")
        detector = GridDetector(score=score, threshold=1e6)

        rng = np.random.default_rng(42)
        n = 50
        x = rng.normal(size=(n, n_features))

        state, outputs = _run_stream(detector, x)

        # Scores should be finite for all time steps
        for out in outputs:
            ms = np.asarray(out["max_score"])
            assert np.all(np.isfinite(ms)), f"Non-finite score: {ms}"

    @pytest.mark.parametrize(
        "score_cls, kwargs",
        [
            (CUSUM, dict(n_features=1)),
            (GaussianMeanOrVariance, dict(n_features=1)),
            (GaussianVariance, dict(n_features=1)),
            (CUSUM, dict(n_features=3, aggregation="max-sum")),
            (RegressionMcScan, dict(n_regressors=2)),
        ],
        ids=lambda x: x.__name__ if isinstance(x, type) else str(x),
    )
    def test_constant_vs_time_dependent_differ(self, score_cls, kwargs):
        """Enabled and disabled penalty modes should produce different scores."""
        score_td = score_cls(**kwargs, enable_penalty=True)
        score_c = score_cls(**kwargs, enable_penalty=False)

        n_features = getattr(score_td, "n_features")
        det_td = GridDetector(score=score_td, threshold=1e6)
        det_c = GridDetector(score=score_c, threshold=1e6)

        rng = np.random.default_rng(7)
        n = 50
        x = rng.normal(size=(n, n_features))

        _, outs_td = _run_stream(det_td, x)
        _, outs_c = _run_stream(det_c, x)

        # After enough observations, scores should differ
        td_final = np.asarray(outs_td[-1]["max_score"])
        c_final = np.asarray(outs_c[-1]["max_score"])

        # Enabled mode divides by a time-varying divisor > 1 for large n,
        # while disabled mode divides by 1.0.
        assert not np.allclose(td_final, c_final), (
            "Expected different scores: "
            f"enable_penalty=True={td_final}, "
            f"enable_penalty=False={c_final}"
        )

    def test_disabled_penalty_returns_centered_statistic(self):
        """With enable_penalty=False the centered statistic is returned (divisor 1)."""
        rng = np.random.default_rng(0)
        data = rng.normal(size=(12, 1))
        off = CUSUM(n_features=1, enable_penalty=False)
        on = CUSUM(n_features=1, enable_penalty=True)
        total = off.init_state()
        for x in data:
            total = off.update(total, x)
        grid = [_prefix_state(off, data, n1) for n1 in (3, 7, 11)]

        off_out = off.compute_penalized_scores(total, grid)
        on_out = on.compute_penalized_scores(total, grid)
        # Disabled divides by 1.0; enabled divides by chi2_max_bound(1, 1, t) > 1,
        # so the enabled magnitudes are strictly smaller.
        assert np.all(np.abs(on_out) < np.abs(off_out) + 1e-12)
        assert chi2_max_bound(1, 1, total.n_samples) > 1.0


# ---------------------------------------------------------------------------
# apply_bonferroni tests
# ---------------------------------------------------------------------------


def _mv_normal_sampler(rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=3)


class TestApplyBonferroni:
    """Verify the apply_bonferroni parameter in calibrate_threshold_false_alarm."""

    def test_multivariate_threshold_returns_array(self):
        """Multivariate score should return an array threshold."""
        score = CUSUM(n_features=3, aggregation="max-sum")
        threshold = calibrate_threshold_false_alarm(
            score=score,
            false_alarm_probability=0.1,
            n_paths=50,
            stream_len=30,
            pre_sampler=_mv_normal_sampler,
            rng=42,
            parallel=False,
        )
        assert isinstance(threshold, np.ndarray)
        assert threshold.shape == (2,)
        assert np.all(threshold > 0)

    def test_bonferroni_higher_than_uncorrected(self):
        """Bonferroni-corrected threshold should be >= uncorrected."""
        score = CUSUM(n_features=3, aggregation="max-sum")
        th_bonf = calibrate_threshold_false_alarm(
            score=score,
            false_alarm_probability=0.1,
            n_paths=80,
            stream_len=30,
            pre_sampler=_mv_normal_sampler,
            rng=42,
            parallel=False,
            apply_bonferroni=True,
        )
        th_no_bonf = calibrate_threshold_false_alarm(
            score=score,
            false_alarm_probability=0.1,
            n_paths=80,
            stream_len=30,
            pre_sampler=_mv_normal_sampler,
            rng=42,
            parallel=False,
            apply_bonferroni=False,
        )

        assert isinstance(th_bonf, np.ndarray)
        assert isinstance(th_no_bonf, np.ndarray)
        # Bonferroni uses alpha/K instead of alpha, giving a higher quantile
        assert np.all(th_bonf >= th_no_bonf), (
            f"Bonferroni {th_bonf} should be >= uncorrected {th_no_bonf}"
        )

    def test_scalar_score_ignores_bonferroni(self):
        """For scalar scores, apply_bonferroni has no effect."""
        score = CUSUM(n_features=1)

        def sampler(rng: np.random.Generator) -> float:
            return float(rng.normal())

        th_bonf = calibrate_threshold_false_alarm(
            score=score,
            false_alarm_probability=0.1,
            n_paths=50,
            stream_len=30,
            pre_sampler=sampler,
            rng=42,
            parallel=False,
            apply_bonferroni=True,
        )
        th_no_bonf = calibrate_threshold_false_alarm(
            score=score,
            false_alarm_probability=0.1,
            n_paths=50,
            stream_len=30,
            pre_sampler=sampler,
            rng=42,
            parallel=False,
            apply_bonferroni=False,
        )

        assert isinstance(th_bonf, np.ndarray)
        assert isinstance(th_no_bonf, np.ndarray)
        assert th_bonf.shape == (1,)
        assert th_no_bonf.shape == (1,)
        assert np.array_equal(th_bonf, th_no_bonf)
