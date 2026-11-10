"""Bias-aware data assimilation of raw azimuthal thermoacoustic data.

The truth comes from experimental azimuthal pressure data (downloaded from
https://zenodo.org/records/15609832), the forecast model is the annular
low-order model, and the model bias is estimated with an ESN so that the
regularized bias-aware EnKF (rBA-EnKF) can assimilate the raw observations.
The whole experiment is described by ``annular.yml`` (next to this script) and
runs through `romda.experiments` — the same flow as

    python -m romda.experiments examples/thermoacoustics/annular.yml [--set truth.ER=0.5375]

Run as a script for a default equivalence ratio and regularization sweep
(``python examples/thermoacoustics/annular.py [--save-figs]``), or import
`run_experiment` from the sweeps in ``sweeps_annular.py``.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # examples/: common

from common import load_main_config, output_folders, parse_main_args, print_metrics, show_or_save_figs
from romda import experiments
from romda.utils import save_to_pickle_file

results_folder, figs_folder = output_folders(__file__)   # gitignored, next to this script

CFG = load_main_config('thermoacoustics/annular')


def run_experiment(ER=None, regularization_factor=None, **kwargs):
    """`romda.experiments.run_experiment` on the annular config; `ER` and
    `regularization_factor` map onto the truth/ensemble overrides, and the same
    kwargs apply (`da_method`, `std_obs`, `truth_kwargs`, `ensemble_kwargs`,
    `bias_kwargs`, `t_extra`, `seed`).

    Returns
    -------
    (EnsembleEstimator, Observations, dict)
        Filtered ensemble, truth, and the error metrics.
    """
    truth_kwargs = dict(kwargs.pop('truth_kwargs', None) or {})
    if ER is not None:
        truth_kwargs['ER'] = ER
    ensemble_kwargs = dict(kwargs.pop('ensemble_kwargs', None) or {})
    if regularization_factor is not None:
        ensemble_kwargs['bias_regularization'] = regularization_factor
    return experiments.run_experiment(CFG, truth_kwargs=truth_kwargs,
                                      ensemble_kwargs=ensemble_kwargs, **kwargs)


# --------------------------------------- Main ---------------------------------------- #

if __name__ == '__main__':

    args = parse_main_args(description=__doc__)

    ER = CFG['truth']['ER']
    out = []

    for kk in [0., 5.]:   # regularization factors (kk = 0 recovers the bias-blind EnKF update)
        filter_ens, truth, metrics = run_experiment(ER=ER, regularization_factor=kk)
        print_metrics(metrics, header=f'Annular ER={ER}, regularization={kk}')
        out.append(filter_ens)

        filter_ens.visualize_history(truth=truth, plot_members=False, dims=[0, 1])
        filter_ens.bias.visualize_bias_and_innovations(plot_members=True)

    results_dir = f'{results_folder}ER{ER}/'
    os.makedirs(results_dir, exist_ok=True)
    save_to_pickle_file(f'{results_dir}rBA_EnKF_regularization_sweep', truth, out)

    show_or_save_figs(args.save_figs, prefix='annular', figs_folder=figs_folder)
