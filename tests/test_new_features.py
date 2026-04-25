"""Tests for enable_penalty behavior and apply_bonferroni features."""

import numpy as np
import pytest

from gridcp.calibration import calibrate_threshold_false_alarm
from gridcp.detector import GridDetector
from gridcp.scores import (
    MeanCUSUM,
    MeanCUSUMUnknownVariance,
    MeanOrVariance,
    MultivariateMeanIdentityCov,
    MultivariateMeanOrCovariance,
    MultivariateMeanUnknownCov,
    RegressionDirect,
    RegressionMcScan,
    Variance,
)


def _run_stream(detector, x):
    """Feed observations through a detector and return final state + outputs."""
    state = detector.init_state()
    outputs = []
    for row in x:
        state, out = detector.update(state, row)
        outputs.append(out)
    return state, outputs


# ---------------------------------------------------------------------------
# enable_penalty tests
# ---------------------------------------------------------------------------


class TestEnablePenalty:
    """Verify that enable_penalty=False works and differs from enabled mode."""

    @pytest.mark.parametrize(
        "score_cls, kwargs",
        [
            (MeanCUSUM, dict(n_features=1)),
            (MeanCUSUMUnknownVariance, dict(n_features=1)),
            (MeanOrVariance, dict(n_features=1)),
            (Variance, dict(n_features=1)),
            (MultivariateMeanIdentityCov, dict(n_features=3)),
            (MultivariateMeanUnknownCov, dict(n_features=3)),
            (MultivariateMeanOrCovariance, dict(n_features=3)),
            (RegressionDirect, dict(n_regressors=2)),
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
            (MeanCUSUM, dict(n_features=1)),
            (MeanOrVariance, dict(n_features=1)),
            (Variance, dict(n_features=1)),
            (MultivariateMeanIdentityCov, dict(n_features=3)),
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

    def test_constant_penalty_returns_one(self):
        """_get_penalty should return 1.0 for disabled mode."""
        score = MeanCUSUM(n_features=1, enable_penalty=False)
        assert score._get_penalty(100) == 1.0

    def test_time_dependent_penalty_greater_than_one(self):
        """_get_penalty should return > 1.0 for enabled mode with large n."""
        score = MeanCUSUM(n_features=1, enable_penalty=True)
        assert score._get_penalty(100) > 1.0


# ---------------------------------------------------------------------------
# apply_bonferroni tests
# ---------------------------------------------------------------------------


def _mv_normal_sampler(rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=3)


class TestApplyBonferroni:
    """Verify the apply_bonferroni parameter in calibrate_threshold_false_alarm."""

    def test_multivariate_threshold_returns_array(self):
        """Multivariate score should return an array threshold."""
        score = MultivariateMeanIdentityCov(n_features=3)
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
        score = MultivariateMeanIdentityCov(n_features=3)
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
        score = MeanCUSUM(n_features=1)

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

        assert isinstance(th_bonf, float)
        assert isinstance(th_no_bonf, float)
        assert th_bonf == th_no_bonf
