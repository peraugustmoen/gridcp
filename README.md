# gridcp

**Online grid-based changepoint detection in Python — with logarithmic time and memory.**

[![CI](https://github.com/peraugustmoen/G-CHAD/actions/workflows/python-package.yml/badge.svg)](https://github.com/peraugustmoen/G-CHAD/actions/workflows/python-package.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/peraugustmoen/G-CHAD/blob/main/LICENSE)

<!-- On release, add PyPI version and DOI badges here. -->

## Overview

**Online changepoint detection** is the problem of detecting distributional changes
in a data stream in real time. A rich toolbox of *offline* (fixed-sample) tests
already exists, but rerunning them as data accumulate becomes infeasible: cost grows
with the length of the stream.

`gridcp` makes offline tests usable online. It evaluates a chosen test statistic over a
**sparse, geometric grid of candidate split points** — dense near the present and
increasingly spread out into the past — so that update time and memory stay **O(log n)**
in the length of the stream, while guaranteeing that no true changepoint is ever far from
a grid point.

```
   time ──────────────────────────────────────────────▶  now
   │        │     │     │   │  │  │ │ │ ││││
   ●········●·····●·····●···●··●··●·●·●·●●●●●          candidate split points
   sparse in the distant past        dense near the present
```

The package pairs this grid detector with a library of ready-to-use, Numba-accelerated
detectors and with Monte Carlo tools for calibrating alarm thresholds to a target
false-alarm probability or average run length (ARL).

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

The `threshold` above is illustrative. To control the false-alarm rate, calibrate it
first — see [Calibrating thresholds](#calibrating-thresholds).

## Key features

- **Bring your own statistic.** Any offline test that implements the lightweight
  `ScoreModel` protocol works with the detector — no detector internals required.
- **Logarithmic cost.** Update time and memory scale as O(log n) in the stream length.
- **Built-in tests included.** A library of built-in detectors for changes in mean,
  variance, covariance, regression coefficients, and exponential-family parameters,
  plus a non-parametric detector.
- **Numba-accelerated.** Hot paths in the built-in scores are JIT-compiled.
- **Principled thresholds.** Monte Carlo calibration targets a false-alarm probability
  or an average run length, with optional parallel execution.

## Built-in detectors

Each implemented test is a *score* imported from `gridcp.scores` and plugged into `GridDetector`.

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

`ExponentialFamilyGLR.from_family(name)` provides ready-made GLR detectors for the
`bernoulli`, `poisson`, `exponential`, `gamma_rate`, `gaussian_mean`,
`gaussian_variance`, `gaussian_mean_variance`, and `gaussian_covariance` families.

## Calibrating thresholds

Pick a threshold that achieves a target false-alarm probability under a null model you
specify, then build the detector with it:

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

To target an average run length instead, use `calibrate_threshold_arl`. Both helpers
support parallel execution and can calibrate directly from data. See the
[demo notebook](https://github.com/peraugustmoen/G-CHAD/blob/main/notebooks/new_api_demo.ipynb)
for an end-to-end calibration and benchmarking workflow.

## Custom detectors

Any object implementing the `ScoreModel` protocol — `init_state`, `update`, and
`compute_penalized_scores` — can be used with `GridDetector`; there is no base class to
inherit from. This is how you bring your own offline test statistic online. The
[demo notebook](https://github.com/peraugustmoen/G-CHAD/blob/main/notebooks/new_api_demo.ipynb)
walks through a complete custom score, and
[CONTRIBUTING.md](https://github.com/peraugustmoen/G-CHAD/blob/main/CONTRIBUTING.md)
documents the full contract.

## Documentation and examples

- **Worked examples:** the [`notebooks/`](https://github.com/peraugustmoen/G-CHAD/tree/main/notebooks)
  directory contains runnable tours of detection, calibration, and benchmarking.
- **API reference:** every public class and function carries a NumPy-style docstring;
  use `help(...)` or your editor's tooltips.

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
