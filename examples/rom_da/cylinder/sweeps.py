"""Sensitivity sweeps of the POD-ESN cylinder wake assimilation.

Requires the cylinder wake dataset (downloaded automatically on first run).
The POD-ESN reduced-order model is trained once per configuration and cached
on disk, so repeated sweeps reuse it.

Usage
-----
    python examples/rom_da/cylinder/sweeps.py --study ensemble_size [--seeds 3] [--quick] [--plot]

Studies
-------
- ``ensemble_size`` : number of members ``m``.
- ``assimilation_frequency`` : time steps between observations ``Nt_obs``.
- ``sensors`` : number of point sensors (QR-pivot placement on the flow field).
- ``wout_estimation`` : state-only vs joint state + ``Wout`` estimation, over
  the parameter uncertainty ``std_alpha`` and initial state uncertainty
  ``std_phi``.
- ``inflation`` : multiplicative inflation factor (the periodic wake keeps the
  seed spread tiny; inflation counteracts the spread collapse).

Results are appended to ``results/cylinder_<study>.jsonl`` next to this script and can be
re-plotted at any time with ``--plot``.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # examples/: common, sweep

import pod_esn as case
from common import output_folders
from sweep import main

results_folder, figs_folder = output_folders(__file__)   # gitignored, next to this script


# ------------------------- Experiment adapter ------------------------- #

def experiment(seed=0,
               m=50,
               Nt_obs=25,
               N_sensors=None,
               std_phi=1.5,
               est_Wout=False,
               std_alpha=0.02,
               inflation_factor=None,
               std_obs=0.1):
    """Map a flat grid point onto ``pod_esn.run_experiment`` kwargs.

    ``N_sensors=None`` keeps the default (down-sampled QR) sensor placement of
    the trained ROM; an integer re-selects that many QR-pivot sensors.
    """
    ensemble_kwargs = dict(m=m, std_phi=std_phi, seed=seed)
    if inflation_factor is not None:
        ensemble_kwargs['inflation'] = inflation_factor
    if est_Wout:
        ensemble_kwargs.update(est_alpha=['Wout'], std_alpha=std_alpha)

    _, _, metrics = case.run_experiment(N_sensors=N_sensors, std_obs=std_obs,
                                        truth_kwargs=dict(Nt_obs=Nt_obs),
                                        ensemble_kwargs=ensemble_kwargs)
    return metrics


# ------------------------- Study definitions ------------------------- #

STUDIES = dict(
    ensemble_size=dict(
        axes=dict(m=[10, 50, 100, 200]),
        x_key='m',
    ),
    assimilation_frequency=dict(
        axes=dict(Nt_obs=[10, 20, 30, 50]),
        x_key='Nt_obs',
    ),
    sensors=dict(
        axes=dict(N_sensors=[1, 2, 3, 4, 8]),
        x_key='N_sensors',
    ),
    inflation=dict(
        axes=dict(inflation_factor=[1.0, 1.01, 1.02, 1.05, 1.1, 1.2]),
        x_key='inflation_factor',
    ),
    wout_estimation=dict(
        axes=dict(est_Wout=[False, True],
                  std_phi=[0.1, 0.5, 1.1],
                  std_alpha=[0.01, 0.05, 0.1, 0.2]),
        x_key='std_alpha', group_key='est_Wout',
    ),
)


# ------------------------------- Main -------------------------------- #

if __name__ == '__main__':
    main(STUDIES, experiment, f'{results_folder}cylinder_{{study}}.jsonl',
         os.path.join(figs_folder, 'cylinder_{study}.png'), 'POD-ESN cylinder', doc=__doc__,
         seeds=3, quick_seeds=1)
