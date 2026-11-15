"""Sensitivity sweeps of the Lorenz96 twin experiment.

Usage
-----
    python scripts/examples/lorenz/sweeps_lorenz96.py --study ensemble_size [--seeds 5] [--quick] [--plot]

Studies
-------
- ``ensemble_size`` : number of members ``m`` for EnKF vs EnSRKF.
- ``observation_sparsity`` : time steps between observations ``Nt_obs`` x
  observing every k-th of the ``Nx = 40`` variables (one combined figure).
- ``inflation`` : multiplicative inflation factor (undersampled regime, small m).

Results are appended to ``results/lorenz96_<study>.jsonl`` next to this script and can be
re-plotted at any time with ``--plot``.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # scripts/: common

from common import load_main_config, output_folders
from romda.sweep import main, run_point

results_folder, figs_folder = output_folders(__file__)   # gitignored, next to this script

CFG = load_main_config('examples/lorenz/lorenz96')
Nx = CFG['truth']['params']['Nx']


def experiment(seed=0, obs_every=CFG['truth']['observe_every'], **point):
    """The metrics of a grid point over the configuration, with the number of observed sites ``Nq``."""
    metrics = run_point(CFG, seed=seed, obs_every=obs_every, **point)[2]
    metrics['Nq'] = len(range(0, Nx, obs_every))
    return metrics


# ------------------------- Study definitions ------------------------- #

STUDIES = dict(
    ensemble_size=dict(
        axes=dict(m=[10, 20, 50, 100],
                  da_method=['EnKF', 'EnSRKF']),
        x_key='m', group_key='da_method',
    ),
    observation_sparsity=dict(
        # Temporal (Nt_obs) x spatial (obs_every) observation sparsity in one figure
        axes=dict(Nt_obs=[5, 10, 20, 40],
                  obs_every=[1, 2, 4, 8]),
        x_key='Nt_obs', group_key='obs_every',
    ),
    inflation=dict(
        axes=dict(inflation_factor=[1.0, 1.02, 1.05, 1.1],
                  m=[20]),      # inflation matters most for small ensembles
        x_key='inflation_factor',
    ),
)


# ------------------------------- Main -------------------------------- #

if __name__ == '__main__':
    main(STUDIES, experiment, f'{results_folder}lorenz96_{{study}}.jsonl',
         os.path.join(figs_folder, 'lorenz96_{study}.png'), context=CFG, doc=__doc__)
