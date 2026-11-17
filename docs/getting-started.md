# Getting started

`romda` combines physical and data-driven forecast models with real-time data assimilation.
The shortest route is to install the package, run the Van der Pol example, and then choose a
model or tutorial from the navigation.

## Install

From PyPI:

```bash
pip install romda
```

For a source checkout:

```bash
git clone https://github.com/andreanovoa/romda
cd romda
conda create -n romda python=3.12
conda activate romda
pip install -e ".[dev,notebooks]" --use-pep517
```

The convolutional autoencoders and the CAE-ESN reduced-order models (`CAE`, `CAE1D`, `MultiScaleCAE`, `CAE1D_ESN` and `MultiScaleCAE_ESN`) need PyTorch, the optional extra `torch`; POD, SPOD, the echo state network and `POD_ESN` do not:

```bash
pip install "romda[torch]"              # from PyPI
pip install -e ".[torch]" --use-pep517  # from a source checkout
```

Extras combine, e.g. `".[torch,dev,notebooks]"`.


## Quick example

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

for data, time in zip(truth.y_obs, truth.t_obs):
    ensemble.forecast_step(t_end=time)
    ensemble.analysis_step(d=data, Cdd=Cdd)
```

This filter is the regularized bias-aware EnKF of Nóvoa, Racca & Magri (2023), which
`romda.estimators.rBA_EnKF` also builds under its name in the paper. The ensemble forecast runs in a
multiprocessing pool, so, in a script, keep the code under `if __name__ == "__main__":`.

romda runs silently. For debugging, `romda.set_verbose()` prints its diagnostic
messages, and those of `dynamodels` and `echostatenetwork`, on stderr (e.g. a cached
network loaded, the noise added to the observations, the multiprocessing pool of an
ensemble, a network trained again), and
`romda.set_verbose(False)` silences them again. Warnings are shown either way.

## Run an example

The complete experiments of the repository are documented on the
[Examples](experiments.md) page:

```bash
python -m romda.run scripts/thermoacoustics/rijke.yml --seed 1
```

## Continue

- [Physical models](api/models_physical.md)
- [Data-driven models](api/models_data_driven.md)
- [Data assimilation estimators](api/estimators.md)
- [Tutorials](tutorials.md)
