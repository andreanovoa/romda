# romda — real-time reduced-order modelling and bias-aware data assimilation

[![PyPI](https://img.shields.io/pypi/v/romda)](https://pypi.org/project/romda/)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://andreanovoa.github.io/real-time-bias-aware-DA/)

`romda` is an open-source Python package for real-time data assimilation with
reduced-order models: ensemble Kalman filters (including the regularized bias-aware
EnKF), physical and data-driven forecast models (ESN, POD-ESN), bias estimators, and
POD/SPOD decompositions — with applications to thermoacoustics and fluid flows.
romda builds on three packages, which `pip` installs with it:
[`dynamodels`](https://github.com/andreanovoa/dynamodels) (the physical models and the time integrators),
[`echostatenetwork`](https://github.com/andreanovoa/EchoStateNetwork) (the echo state network) and
[`ntsa`](https://github.com/andreanovoa/ntsa) (time-series analysis).

> This repository was formerly the `real-time-DA` package. The code used in the papers listed
> below is preserved at the release tags [v1.0](https://github.com/andreanovoa/real-time-bias-aware-DA/releases/tag/v1.0), [v1.1](https://github.com/andreanovoa/real-time-bias-aware-DA/releases/tag/v1.1) and [v2.1](https://github.com/andreanovoa/real-time-bias-aware-DA/releases/tag/v2.1).


---

## 🚀 Getting started

1. **Install from PyPI**
```
pip install romda
```

or, to work on the source, clone and install in editable mode:
```
git clone https://github.com/andreanovoa/real-time-bias-aware-DA
cd real-time-bias-aware-DA
conda create -n romda python=3.12 && conda activate romda
pip install -e ".[dev,notebooks]" --use-pep517   # dev = pytest and ruff, notebooks = jupyter
```

2. (Optional) **Run the test suite**
```
python -m pytest tests/                 # unit and integration tests
python -m pytest tutorials/test_tutorials.py           # execute the tutorial notebooks
```

Quick example — bias-aware state and parameter estimation on a Van der Pol twin experiment:
```python
import numpy as np
from romda.bias_estimators import ConstantBias
from romda.estimators import EnsembleFilter
from romda.models.physical import VdP
from romda.observations import Observations

truth = Observations(model=VdP, t_start=0.6, t_stop=0.8, Nt_obs=30,
                     add_noise=True, manual_bias='linear')
Cdd = np.diag((0.05 * np.max(abs(truth.y_obs), axis=0)) ** 2)

ensemble = EnsembleFilter(parent_model=VdP(dt=truth.dt), parent_bias=ConstantBias,
                          type='stochastic',             # EnKF
                          parameter_estimation='joint',  # state and parameters in one analysis
                          bias_regularization=1.0,       # regularized bias-aware update
                          m=10, std_phi=0.1, std_alpha=dict(zeta=(40., 60.)))

for d, t_d in zip(truth.y_obs, truth.t_obs):
    ensemble.forecast_step(t_end=t_d)     # advance model (and bias) to the observation time
    ensemble.analysis_step(d=d, Cdd=Cdd)  # Bayesian update
```
This filter is the regularized bias-aware EnKF of Nóvoa, Racca & Magri (2023), which
`romda.estimators.rBA_EnKF` also builds under its name in the paper. The ensemble forecast runs in a
multiprocessing pool, so, in a script, keep the code under `if __name__ == "__main__":`.

## 🧭 Where to start

1. **The tutorials** ([`tutorials/`](tutorials/README.md)) are numbered notebooks, topic by topic: the classes
   of the package (0), data assimilation (1), thermoacoustics (2), reduced-order models (3), and data
   assimilation with reduced-order models (4). Read them in order.
2. **The examples** ([`examples/`](examples/README.md)) are complete experiments, one folder per topic. Each
   case is a configuration file, with a script next to it when the case needs one; the README of each
   topic lists the papers that it supports:

   ```bash
   python examples/lorenz/lorenz63.py --save-figs
   python -m romda.experiments examples/thermoacoustics/rijke.yml --seed 1
   ```
3. **The documentation** ([andreanovoa.github.io/real-time-bias-aware-DA](https://andreanovoa.github.io/real-time-bias-aware-DA/))
   describes the classes and their options.


---


## 🌟 What is available?
   Data assimilation methods [`romda.estimators`](src/romda/estimators/), with the classes and keywords of [`biasda`](https://github.com/andreanovoa/biasda)
   * `EnsembleFilter` — ensemble Kalman filter, configured by keywords: `type='stochastic'` (EnKF) or `type='sqrt'` (symmetric square-root EnSRKF, i.e. the ETKF); `parameter_estimation='joint'` (state and parameters in one analysis); `inflation` (multiplicative covariance inflation factor); and `parent_bias` with `bias_regularization` (the regularized bias-aware EnKF of Nóvoa, Racca & Magri 2023)
   * `KalmanFilter` — linear Kalman filter
   * Named entry points of `EnsembleFilter`: `EnKF`, `EnSRKF`, `rEnKF`, and `rBA_EnKF` (the name of the regularized bias-aware EnKF in the 2023 paper)

   romda keeps only the filters that its papers employ. The other methods (the deterministic and the other square-root ensemble updates, the dual scheme, covariance localization, the bias-estimation routes, the extended and unscented Kalman filters, the particle filters and the smoothers) live in [`biasda`](https://github.com/andreanovoa/biasda).

   Physical models `romda.models.physical` (from the [`dynamodels`](https://andreanovoa.github.io/dynamodels/) package)
   * `Rijke` — Rijke tube, a longitudinal thermoacoustic model (dimensional, with Galerkin projection)
   * `VdP` — Van der Pol oscillator
   * `Annular` — azimuthal thermoacoustic model of an annular combustor
   * `Lorenz63`, `Lorenz96` — Lorenz 63 and Lorenz 96 systems
   * `KS`, `KS2D` — Kuramoto-Sivashinsky equation in one and two dimensions
   * `Kuznetsov` — Kuznetsov oscillator, with quasiperiodic oscillations

   Data-driven models [`romda.models.data_driven`](src/romda/models/data_driven/)
   * `ESN_model` — echo state network as a forecast model
   * `POD_ESN` — POD dimensionality reduction + ESN forecaster
   * `LatentROM` — base class of the reduced-order models that pair a projector with a forecaster, such as `POD_ESN`
   * `LinearModel` — linear map with process noise (Kalman-filter tutorials)
   * Projectors ([`autoencoders/`](src/romda/models/data_driven/autoencoders/)): `POD`, `SPOD` — (Spectral) Proper Orthogonal Decomposition (Sieber 2016, Towne 2018)

   Reservoir core: `ESN_model` subclasses `EchoStateNetwork` of the external [`echostatenetwork`](https://github.com/andreanovoa/EchoStateNetwork) package; `pod_utils` (in `autoencoders/`) holds the standalone POD/SPOD algorithms

   Bias estimators [`romda.bias_estimators`](src/romda/bias_estimators/)
   * `ESN_bias` — echo state network
   * `ConstantBias` — constant bias
   * `NoBias` — no bias (unbiased limit)


---
## 📂 Structure
```
.
├── src/romda/                   # the package: `from romda.<module> import ...`
│   ├── estimators/              # data assimilation: EnsembleFilter (EnKF, EnSRKF, rEnKF), KalmanFilter
│   ├── bias_estimators/         # bias models: ESN_bias, ConstantBias, NoBias
│   ├── models/                  # physical models (from dynamodels) and data_driven/ (ESN_model, POD_ESN, LatentROM, projectors)
│   ├── plotting/                # figure helpers
│   ├── observations.py          # truth and observations
│   ├── data_assimilation.py     # run_da_loop: the sequential data-assimilation driver
│   ├── experiments.py           # run_experiment: one YAML configuration, one experiment
│   └── metrics.py, utils.py, verbose.py   # verbose.py: logging, silent unless romda.set_verbose()
├── tutorials/                   # numbered notebooks, topic by topic: start here
├── examples/                    # complete experiments: lorenz/, thermoacoustics/, rom_da/cylinder/
├── tests/                       # test suite (pytest)
├── docs/                        # documentation site, built by mkdocs.yml and hooks.py, and the figures of the notebooks
├── data/, results/              # downloaded datasets and cached results, not tracked by git
└── pyproject.toml               # package metadata; CITATION.cff, CHANGELOG.md
```


---

## 📚 Main publications from this repository

##### Journal papers

- [x] Nóvoa, Noiray, Dawson & Magri (2024). A real-time digital twin of azimuthal thermoacoustic instabilities. Journal of Fluid Mechanics. [Published paper](https://doi.org/10.1017/jfm.2024.1052) |  🏷️ [v1.0](https://github.com/andreanovoa/real-time-bias-aware-DA/releases/tag/v1.0) | 📁 [`examples/thermoacoustics`](examples/thermoacoustics/README.md). 
- [x] Nóvoa, Racca & Magri (2023). Inferring unknown unknowns. Computer Methods in Applied Mechanics and Engineering. [Published paper](https://doi.org/10.1016/j.cma.2023.116502) | [_Legacy_ repository](https://github.com/MagriLab/rBA-EnKF) | 📁 [`examples/thermoacoustics`](examples/thermoacoustics/README.md).
- [x] Nóvoa & Magri (2022). Real-time thermoacoustic data assimilation. Journal of Fluid Mechanics. [Published paper](https://doi.org/10.1017/jfm.2022.653) | [_Legacy_ repository](https://github.com/MagriLab/Real-time-TA-DA) | 📁 [`examples/thermoacoustics`](examples/thermoacoustics/README.md).

##### Conference papers and proceedings
- [x] Nóvoa & Magri (2025). Online model learning with data-assimilated reservoir computers. [Preprint](https://doi.org/10.48550/arXiv.2504.16767) | 🏷️ [v1.1](https://github.com/andreanovoa/real-time-bias-aware-DA/releases/tag/v1.1) | 📁 [`examples/rom_da/cylinder`](examples/rom_da/cylinder/README.md).
- [x] Nóvoa & Magri (2024). Real-time digital twins of multiphysics and turbulent flows. [Paper](https://web.stanford.edu/group/ctr/ctrsp24/ii11_NOVOA.pdf).
- [x] Nóvoa & Magri (2022). Bias-aware thermoacoustic data assimilation. In_ 51st International Congress and Exposition on Noise Control Engineering. [Paper](https://az659834.vo.msecnd.net/eventsairwesteuprod/production-inconference-public/808b4f8c38f944d188db8a326a98c65c). | [_Legacy_ repository](https://github.com/MagriLab/IN22-Bias-aware-TADA) | 📁 [`examples/thermoacoustics`](examples/thermoacoustics/README.md).

##### PhD thesis
- [x] Nóvoa (2024). Real-time data assimilation in nonlinear dynamical systems. University of Cambridge. [Thesis](https://doi.org/10.17863/CAM.113001). 

--- 
## 🤝 Contributing

Contributions, bug reports, and feature requests are welcome! Please open an issue or submit a pull request. For questions or collaborations, please reach out to [A. Nóvoa](https://andreanovoa.github.io/).

## 📄 Licence

romda is distributed under the MIT licence ([LICENSE](LICENSE)).
