"""Sensitivity sweeps of bias-aware assimilation of the annular experiment data.

Requires the azimuthal dataset (downloaded automatically from
https://zenodo.org/records/15609832 on first run).

Usage
-----
    python examples/thermoacoustics/sweeps_annular.py --study regularization [--seeds 3] [--quick] [--plot]

Studies
-------
- ``regularization`` : bias regularization factor of the rBA-EnKF (zero is the
  bias-blind limit; the EnKF baseline is in ``ensemble_size``).
- ``ensemble_size`` : number of members ``m``.
- ``assimilation_frequency`` : time steps between observations ``Nt_obs``.
- ``equivalence_ratio`` : robustness across the four available operating
  conditions (ER = 0.4875 ... 0.5625).

Results are appended to ``results/annular_<study>.jsonl`` next to this script and can be
re-plotted at any time with ``--plot``.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # examples/: common, sweep

import annular as case
from common import output_folders
from sweep import main

results_folder, figs_folder = output_folders(__file__)   # gitignored, next to this script

ERs = 0.4875 + np.arange(0, 4) * 0.025    # available equivalence ratios (the data grid)


# ------------------------- Experiment adapter ------------------------- #

def experiment(seed=0,
               da_method='rBA_EnKF',
               ER=ERs[1],
               regularization_factor=5.,
               m=20,
               Nt_obs=35,
               inflation_factor=1.0,
               std_obs=0.05):
    """Map a flat grid point onto ``annular.run_experiment`` kwargs.

    The ESN bias estimator is trained once per (ER, Nt_obs) and cached on disk
    by ``romda.experiments``, so sweeping m/regularization reuses it.
    """
    truth_kwargs = dict(Nt_obs=Nt_obs)
    ensemble_kwargs = dict(m=m, inflation=inflation_factor, seed=seed)

    _, _, metrics = case.run_experiment(ER=ER,
                                        regularization_factor=regularization_factor,
                                        da_method=da_method,
                                        std_obs=std_obs,
                                        truth_kwargs=truth_kwargs,
                                        ensemble_kwargs=ensemble_kwargs)
    return metrics


# ------------------------- Study definitions ------------------------- #

STUDIES = dict(
    regularization=dict(
        axes=dict(regularization_factor=list(np.linspace(0., 10., 11)),
                  da_method=['rBA_EnKF']),
        x_key='regularization_factor',
    ),
    ensemble_size=dict(
        axes=dict(m=[10, 20, 40, 60, 80],
                  da_method=['rBA_EnKF', 'EnKF']),
        x_key='m', group_key='da_method',
    ),
    assimilation_frequency=dict(
        axes=dict(Nt_obs=[15, 25, 35, 50, 70]),
        x_key='Nt_obs',
    ),
    equivalence_ratio=dict(
        axes=dict(ER=list(ERs),
                  da_method=['rBA_EnKF', 'EnKF']),
        x_key='ER', group_key='da_method',
    ),
)


# ------------------------------- Main -------------------------------- #

if __name__ == '__main__':
    main(STUDIES, experiment, f'{results_folder}annular_{{study}}.jsonl',
         os.path.join(figs_folder, 'annular_{study}.png'), 'Annular', doc=__doc__,
         seeds=3, quick_seeds=1)
