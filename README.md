# gridcp

**Online changepoint detection in Python with logarithmic update and storage costs.**

[![PyPI](https://img.shields.io/pypi/v/gridcp.svg)](https://pypi.org/project/gridcp/)
[![CI](https://github.com/peraugustmoen/gridcp/actions/workflows/python-package.yml/badge.svg)](https://github.com/peraugustmoen/gridcp/actions/workflows/python-package.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/peraugustmoen/gridcp/blob/main/LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2608.18695-blue.svg)](https://doi.org/10.48550/arXiv.2608.18695)

## Overview

Online changepoint detection is the problem of detecting a distributional change in a
data stream in real time. Plenty of *offline* tests exist for a fixed sample, such as the CUSUM and other likelihood-ratio tests, but
rerunning them on all candidate split points as new data arrive is computationally expensive, with update and storage costs typically
growing at least linearly with the length of the stream.

`gridcp` lets you run an offline test online with logarithmic update and storage costs. After each new
observation arrives, it evaluates your chosen test statistic over a sparse grid of candidate
split points, so update time and memory stay O(log n) in the stream length. The grid is designed so that no true
changepoint is never far from a candidate, hence the statistic maintains power to detect any change despite being evaluated on fewer candidates. Visually, it looks something like this:

```
   time ────────────────────────────────────▶  now
   │        │     │     │   │  │  │ │ │ ││││
   ●········●·····●·····●···●··●··●·●·●·●●●●   candidate split points
   sparse in the distant past        dense near the present
```

The package comes with a library of built-in test statistics
(called scores) and Monte Carlo tools for calibrating the alarm threshold to a target
false alarm probability or average run length (ARL). For a full explanation of available utilities, see the companion paper ([arXiv:2608.18695](https://arxiv.org/abs/2608.18695)).

## Installation

```bash
pip install gridcp
```

## Example of use
Below is a simple code example showing how to define a score and a corresponding detector in gridcp, and run it sequentially on a data stream with a change at $\tau=100$.
```python
import numpy as np
from gridcp import GridDetector
from gridcp.scores import CUSUM

# A detector for a change in mean, using the CUSUM statistic.
# The detector is stateless, and the evolving state is a separate object.
detector = GridDetector(score=CUSUM(n_features=1), threshold=5.0)
state = detector.init_state()

# Iterate over observations one at a time. At t=100, a change in mean occurs.
rng = np.random.default_rng(0)
for t in range(200):
    x = rng.normal(0.0, 1.0) if t < 100 else rng.normal(3.0, 1.0)
    state, output = detector.update(state, [x])
    if output["alarm"]:
        print(f"Change detected at observation {t}")
        break
```

The threshold above is set manually just for illustration. In practice, it should be set to control the false alarm rate, explained here: [Calibrating thresholds](#calibrating-thresholds).

## Built-in detectors

Each test is a *score* imported from `gridcp.scores`, which is plugged into a `GridDetector`-object before it is run sequentially on data.

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

`gridcp` provides functions for calibrating the detector threshold to a target false alarm probability at a given stream length or to a target average run length by the use of Monte Carlo simulations. The example below illustrates how to calibrate to a target false alarm probability of 5% when the null distribution is standard Gaussian.

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

To target an average run length, use instead `calibrate_threshold_arl`. If the null distribution is not known but a data set with no change is available, `calibrate_threshold_false_alarm_from_data` and `calibrate_threshold_arl_from_data` can be used. For more information, read Section 5 of the companion paper ([arXiv:2608.18695](https://arxiv.org/abs/2608.18695)).

## Custom detectors

Any object that implements the `ScoreModel` protocol works with `GridDetector`. More information is provided in the Appendix of the companion paper.

## Citation

If you use `gridcp` in your research, please cite:

```bibtex
@misc{gridcp,
  title         = {{gridcp}: Fast Online Changepoint Detection in {Python}},
  author        = {Moen, Per August Jarval and Nielsen, Sebastian Grau and Urheim, Espen Bj{\o}rge and Tveten, Martin and Glad, Ingrid Kristine},
  year          = {2026},
  eprint        = {2608.18695},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ME},
  url           = {https://arxiv.org/abs/2608.18695}
}
```

## Reproducing the paper

The notebooks and data reproducing every figure, table, and timing result in the
paper live in a separate repository:
[gridcp-paper](https://github.com/espenurheim/gridcp-paper).

## Contributing

Contributions are welcome. See
[CONTRIBUTING.md](https://github.com/peraugustmoen/gridcp/blob/main/CONTRIBUTING.md)
for the development setup, coding conventions, and the contract for adding a new
score.

## License

`gridcp` is released under the MIT License. See
[LICENSE](https://github.com/peraugustmoen/gridcp/blob/main/LICENSE).
