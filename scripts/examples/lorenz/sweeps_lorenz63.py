"""Sensitivity sweeps of the Lorenz63 twin experiment.

Usage
-----
    python scripts/examples/lorenz/sweeps_lorenz63.py --study ensemble_size [--seeds 5] [--quick] [--plot]

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # scripts/: common

from common import load_main_config, output_folders
from romda.sweep import config_experiment, main

results_folder, figs_folder = output_folders(__file__)   # gitignored, next to this script

CFG = load_main_config('examples/lorenz/lorenz63')

#: the grid points over the configuration, whose std_obs is 0.01
experiment = config_experiment(CFG, std_obs=0.05)


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
                  manual_bias=['linear_offset']),
        x_key='regularization_factor',
    ),
)


# ------------------------------- Main -------------------------------- #

if __name__ == '__main__':
    main(STUDIES, experiment, f'{results_folder}lorenz63_{{study}}.jsonl',
         os.path.join(figs_folder, 'lorenz63_{study}.png'), context=CFG, doc=__doc__)
