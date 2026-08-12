# gridcp

**Online grid-based changepoint detection in Python — with logarithmic time and memory.**

[![CI](https://github.com/peraugustmoen/G-CHAD/actions/workflows/python-package.yml/badge.svg)](https://github.com/peraugustmoen/G-CHAD/actions/workflows/python-package.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/peraugustmoen/G-CHAD/blob/main/LICENSE)

<!-- On release, add PyPI version and DOI badges here. -->

## Overview

Online changepoint detection is the problem of detecting a distributional change in a
data stream in real-time. Plenty of *offline* tests exist for detecting a change in a
fixed sample, but rerunning them as new data arrive gets expensive — the cost of each
re-test typically grows with the length of the stream.

`gridcp` lets you run an offline test online without that blow-up. After each new
observation it evaluates your chosen test statistic over a sparse grid of candidate
split points — dense near the present, spreading out into the past — so update time and
memory stay O(log n) in the stream length. The grid is designed so that no true
changepoint is ever far from a candidate.

```
   time ──────────────────────────────────────────────▶  now
   │        │     │     │   │  │  │ │ │ ││││
   ●········●·····●·····●···●··●··●·●·●·●●●●●          candidate split points
   sparse in the distant past        dense near the present
```

The detector comes with a library of built-in, Numba-accelerated test statistics (scores) and
with Monte Carlo tools for calibrating the alarm threshold to a target false alarm
probability or average run length (ARL).

## Installation

```bash
pip install gridcp
```

## Quick start

```python
import numpy as np
from gridcp import GridDetector
from gridcp.scores import CUSUM

# A detector for a change in mean, using the CUSUM statistic.
# The detector is stateless — the evolving state is a separate object.
detector = GridDetector(score=CUSUM(n_features=1), threshold=5.0)
state = detector.init_state()

# Feed observations one at a time; a change is injected at t = 100.
rng = np.random.default_rng(0)
for t in range(200):
    x = rng.normal(0.0, 1.0) if t < 100 else rng.normal(3.0, 1.0)
    state, output = detector.update(state, [x])
    if output["alarm"]:
        print(f"Change detected at observation {t}")
        break
```

The threshold above is just for illustration. To control the false-alarm rate, calibrate
it first — see [Calibrating thresholds](#calibrating-thresholds).

## Key features

- **Bring your own statistic** — any offline test implementing the `ScoreModel` protocol
  works with the detector. You never need to touch detector internals.
- **Logarithmic cost** — update time and memory scale as O(log n) in the stream length.
- **Explicit, immutable state** — the detector is stateless. Each stream's state is a
  separate object you thread through `update`, so one detector can drive many streams and
  a state can be copied or pickled.
- **Built-in detectors** — changes in mean, variance, covariance, regression
  coefficients, and exponential-family parameters, plus a non-parametric detector.
- **Numba-accelerated** — the hot paths in the built-in scores are JIT-compiled.
- **Calibrated thresholds** — Monte Carlo calibration to a target false-alarm
  probability or average run length, optionally in parallel.

## Built-in detectors

Each test is a *score* imported from `gridcp.scores` and plugged into `GridDetector`.

| Score | Detects a change in |
| --- | --- |
| `CUSUM` | mean (known unit variance), uni- or multivariate |
| `GaussianMean` | mean (unknown variance/covariance) |
| `GaussianVariance` | variance |
| `GaussianMeanOrVariance` | mean and/or variance |
| `GaussianMeanOrCovariance` | mean and/or covariance |
| `RegressionMcScan` | regression coefficients (McScan) |
| `RegressionWald` | regression coefficients (Wald) |
| `ExponentialFamilyGLR` | exponential-family parameters (GLR test) |
| `NPFOCuS` | distribution (non-parametric) |

`ExponentialFamilyGLR.from_family(name)` gives ready-made GLR detectors for the
`bernoulli`, `poisson`, `exponential`, `gamma_rate`, `gaussian_mean`,
`gaussian_variance`, `gaussian_mean_variance`, and `gaussian_covariance` families.

## Calibrating thresholds

Calibrate a threshold to a target false-alarm probability under a null model you specify,
then build the detector with it:

```python
import numpy as np
from gridcp import GridDetector
from gridcp.scores import CUSUM
from gridcp.calibration import calibrate_threshold_false_alarm

def normal_sampler(rng, size, loc, scale):
    return rng.normal(size=size, loc=loc, scale=scale)

score = CUSUM(n_features=1)
threshold = calibrate_threshold_false_alarm(
    score=score,
    false_alarm_probability=0.05,
    stream_len=150,
    n_paths=3000,
    pre_sampler=normal_sampler,
    pre_kwargs={"size": 1, "loc": 0.0, "scale": 1.0},
    rng=123,
)

detector = GridDetector(score=score, threshold=threshold)
```

For an average run length instead, use `calibrate_threshold_arl`. Both can run in
parallel and can calibrate directly from data. 

## Custom detectors

Any object that implements the `ScoreModel` protocol — the methods `init_state`,
`update`, and `compute_penalized_scores`, plus the `n_features` and `n_scores`
properties — works with `GridDetector`. There is no base class to inherit from. This is
how you bring your own offline test statistic online. 



## Citation

If you use `gridcp` in your research, please cite:

```bibtex
@misc{gridcp,
  title  = {{gridcp}: Grid-based online changepoint detection in {Python}},
  author = {Moen, Per August J., Nielsen, Sebastian G., Urheim, Espen B., Tveten, Martin, Glad, Ingrid K.},
  year   = {2026},
  note   = {Working paper}
}
```
<!-- Complete the author list and publication details before release. -->

## Contributing

Contributions are welcome. See
[CONTRIBUTING.md](https://github.com/peraugustmoen/G-CHAD/blob/main/CONTRIBUTING.md)
for the development setup, coding conventions, and the contract for adding a new
detector.

## License

`gridcp` is released under the MIT License. See
[LICENSE](https://github.com/peraugustmoen/G-CHAD/blob/main/LICENSE).
