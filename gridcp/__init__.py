"""gridcp — online grid-based changepoint detection.

Public API
----------
The main classes are imported from the top-level package:

- :class:`GridDetector` — the meta-detector.
- :class:`DetectorState` — frozen state object returned by the detector.
- :func:`reset_detector_state` — reset a detector state to time 0.
- :class:`DetectorOutput` — typed dict returned per update step.

Score classes (e.g. :class:`~gridcp.scores.MeanCUSUM`,
:class:`~gridcp.scores.ExponentialFamilyGLR`) are **not** re-exported from
this top-level namespace and must be imported from :mod:`gridcp.scores`.

Calibration helpers
-------------------
Two naming patterns exist:

- ``calibrate_threshold_*`` functions accept a *score* object
  (:class:`~gridcp.typing.ScoreModel`) as their first argument.
- ``calibrate_detector_threshold_*`` functions accept an already-constructed
  :class:`GridDetector` as a convenience wrapper.

Both sets are equivalent; use whichever matches the objects you already have.
"""

from gridcp.calibration import (
    calibrate_detector_threshold_arl,
    calibrate_detector_threshold_arl_from_data,
    calibrate_detector_threshold_arl_from_samples,
    calibrate_detector_threshold_false_alarm,
    calibrate_detector_threshold_false_alarm_from_data,
    calibrate_threshold_arl,
    calibrate_threshold_arl_from_data,
    calibrate_threshold_arl_from_samples,
    calibrate_threshold_false_alarm,
    calibrate_threshold_false_alarm_from_data,
    calibrate_threshold_false_alarm_from_samples,
    draw_samples,
    mc_alarm_times,
    mc_max_scores,
    with_calibrated_threshold,
)
from gridcp.detector import DetectorState, GridDetector, reset_detector_state
from gridcp.typing import DetectorOutput

__all__ = [
    "draw_samples",
    "mc_alarm_times",
    "mc_max_scores",
    "calibrate_threshold_false_alarm",
    "calibrate_detector_threshold_false_alarm",
    "calibrate_threshold_false_alarm_from_samples",
    "calibrate_threshold_false_alarm_from_data",
    "calibrate_detector_threshold_false_alarm_from_data",
    "calibrate_threshold_arl",
    "calibrate_detector_threshold_arl",
    "calibrate_threshold_arl_from_samples",
    "calibrate_detector_threshold_arl_from_samples",
    "calibrate_threshold_arl_from_data",
    "calibrate_detector_threshold_arl_from_data",
    "with_calibrated_threshold",
    "GridDetector",
    "DetectorState",
    "reset_detector_state",
    "DetectorOutput",
]
