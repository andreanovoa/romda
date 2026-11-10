"""Sensitivity sweeps of the Lorenz63 twin experiment.

Usage
-----
    python examples/lorenz/sweeps_lorenz63.py --study ensemble_size [--seeds 5] [--quick] [--plot]

Studies
-------
- ``ensemble_size`` : number of members ``m`` for EnKF vs EnSRKF.
- ``assimilation_frequency`` : time steps between observations ``Nt_obs``.
- ``inflation`` : multiplicative inflation factor.
- ``observation_noise`` : truth noise level vs assumed observation error.
- ``regularization`` : bias-aware r-EnKF regularization factor (biased truth).

Results are appended to ``results/lorenz63_<study>.jsonl`` next to this script and can be
re-plotted at any time with ``--plot`` (no new runs if the sweep is complete).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # examples/: common, sweep

import lorenz63 as case
from common import output_folders
from sweep import main

results_folder, figs_folder = output_folders(__file__)   # gitignored, next to this script


# ------------------------- Experiment adapter ------------------------- #

def experiment(seed=0,
               da_method='EnSRKF',
               m=50,
               Nt_obs=20,
               inflation_factor=1.02,
               noise_level=0.05,
               std_obs=0.05,
               regularization_factor=None,
               manual_bias=None):
    """Map a flat grid point onto ``lorenz63.run_experiment`` kwargs."""
    truth_kwargs = dict(Nt_obs=Nt_obs, noise_level=noise_level, manual_bias=manual_bias)
    ensemble_kwargs = dict(m=m, inflation=inflation_factor, seed=seed)
    if regularization_factor is not None:
        ensemble_kwargs['bias_regularization'] = regularization_factor

    _, _, metrics = case.run_experiment(da_method=da_method,
                                        std_obs=std_obs,
                                        truth_kwargs=truth_kwargs,
                                        ensemble_kwargs=ensemble_kwargs)
    return metrics


# ------------------------- Study definitions ------------------------- #

STUDIES = dict(
    ensemble_size=dict(
        axes=dict(m=[5, 10, 20, 50, 100],
                  da_method=['EnKF', 'EnSRKF']),
        x_key='m', group_key='da_method',
    ),
    assimilation_frequency=dict(
        axes=dict(Nt_obs=[5, 10, 20, 40, 80]),
        x_key='Nt_obs',
    ),
    inflation=dict(
        axes=dict(inflation_factor=[1.0, 1.01, 1.02, 1.05, 1.1]),
        x_key='inflation_factor',
    ),
    observation_noise=dict(
        axes=dict(noise_level=[0.01, 0.05, 0.1, 0.2],
                  std_obs=[0.01, 0.05, 0.1]),
        x_key='noise_level', group_key='std_obs',
    ),
    regularization=dict(
        axes=dict(regularization_factor=list(np.linspace(0., 5., 6)),
                  da_method=['rBA_EnKF'],
                  manual_bias=['linear']),
        x_key='regularization_factor',
    ),
)


# ------------------------------- Main -------------------------------- #

if __name__ == '__main__':
    main(STUDIES, experiment, f'{results_folder}lorenz63_{{study}}.jsonl',
         os.path.join(figs_folder, 'lorenz63_{study}.png'), 'Lorenz63', doc=__doc__)
