"""Real-time data assimilation of the cylinder wake with a POD-ESN model.

A POD-ESN reduced-order model is trained on noisy snapshots of the cylinder
wake (Re = 100), sparse sensors are placed on the flow field (QR pivoting) or
directly on the POD coefficients, and the remaining snapshots provide the
observations for sequential filtering with the EnSRKF. Optionally the ESN
output weights (``Wout`` singular values) are appended to the augmented state
and estimated online. Defaults live in ``pod_esn.yml`` (next to this script):
a data-driven forecast model, no true model required.

Run as a script for a single default experiment
(``python scripts/podesn_cylinder/pod_esn.py [--save-figs]``), or import the
builder functions from the sweeps in ``sweeps.py``.
"""

import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts/: common

from common import load_main_config, main_parser, output_folders, print_metrics, show_or_save_figs
from romda.data_assimilation import run_da_loop
from romda.estimators import EnsembleEstimator
from romda.run import DA_METHODS
from romda.metrics import ensemble_metrics
from romda.models.data_driven import POD_ESN
from romda.observations import Observations
from romda.utils import load_cylinder_dataset, load_from_pickle_file, save_to_pickle_file

results_folder, figs_folder = output_folders(__file__)   # gitignored, next to this script

CFG = load_main_config('podesn_cylinder/pod_esn')

# Sensors of `run_experiment(N_sensors=...)`, unless the configuration has a `sensors` section
SENSORS = dict(domain_of_measurement=[4, 7, -1, 1],   # x_min, x_max, y_min, y_max (wake)
               down_sample_measurement=(10, 10),
               qr_selection=True)


# ------------------------------------- Builders -------------------------------------- #

def load_data(cfg=None, **overrides):
    """Load the (noisy, true) cylinder snapshots and split train/filter sets.

    The first ``N_train + N_val`` snapshots of the configuration `cfg` (default: ``pod_esn.yml``) train the
    reduced-order model, and the rest are held out for the assimilation."""
    params = {**(cfg or CFG)['data'], **overrides}
    N_train, N_val = params.pop('N_train'), params.pop('N_val')
    X_true, X_noisy, simulation_dir = load_cylinder_dataset(**params)

    X_train, X_train_true = [yy[:N_train + N_val].transpose(-1, 0, 1, 2) for yy in [X_noisy, X_true]]
    X_filter, X_filter_true = [yy[N_train + N_val:].transpose(-1, 0, 1, 2) for yy in [X_noisy, X_true]]

    return (X_train, X_train_true), (X_filter, X_filter_true), simulation_dir


def build_rom(X_train, simulation_dir, use_cache=True, cfg=None, N_val=None, **overrides) -> POD_ESN:
    """Train (or load from cache) the POD-ESN reduced-order model.

    The last `N_val` snapshots of `X_train` (default: that of the configuration) validate the network.
    The cache is named after the sizes of the model and a digest of its options, so that two
    configurations on the same record do not share a network."""
    cfg = cfg or CFG
    params = {**cfg['rom'], **overrides}
    N_val = cfg['data']['N_val'] if N_val is None else N_val
    params.setdefault('t_train', (X_train.shape[1] - N_val) * params['dt'])
    params.setdefault('t_val', N_val * params['dt'])
    digest = hashlib.sha1(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:8]
    case_filename = (f"{simulation_dir}POD{params['N_modes']}_ESN{params['N_units']}"
                     f"_Ntrain{X_train.shape[1]}_{digest}")

    if use_cache and os.path.isfile(case_filename):
        rom = load_from_pickle_file(case_filename)
        print(f'Loaded POD-ESN case from {case_filename}')
    else:
        rom = POD_ESN(data=X_train, figs_folder=simulation_dir, **params)
        rom.name = case_filename
        if use_cache:
            save_to_pickle_file(case_filename, rom)

    return rom


def build_truth(rom: POD_ESN, X_filter, X_filter_true, cfg=None, **overrides) -> Observations:
    """Sample the measurements from the held-out snapshots: the sensors of `rom`, or the POD coefficients
    of the snapshots when `rom` measures its modes (full fields)."""
    params = {**(cfg or CFG)['truth'], **overrides}

    N_test = X_filter.shape[1]
    if rom.measure_modes:
        y_raw, y_true = (rom.encode(X).T for X in (X_filter, X_filter_true))  # Nt x N_modes
    else:
        y_raw, y_true = (X.transpose(1, 0, 2, 3).reshape(N_test, -1)[:, rom.sensor_locations]
                         for X in (X_filter, X_filter_true))                  # Nt x Nq

    t_true = np.arange(N_test) * rom.dt
    params['t_stop'] = min(params['t_stop'], t_true[-10])

    return Observations(y_raw=y_raw, y_true=y_true, t_true=t_true, **params)


def build_ensemble(rom: POD_ESN, da_method=None, cfg=None, **overrides) -> EnsembleEstimator:
    params = {**(cfg or CFG)['da'], **overrides}
    params.pop('std_obs', None)
    method = params.pop('method')
    return DA_METHODS[da_method or method](parent_model=rom, **params)


def run_experiment(da_method=None,
                   std_obs=None,
                   data_kwargs=None,
                   rom_kwargs=None,
                   truth_kwargs=None,
                   ensemble_kwargs=None,
                   t_extra=None,
                   N_sensors=None,
                   cfg=None,
                   return_case=False):
    """Train/load the POD-ESN, build observations from held-out snapshots, filter.

    ``N_sensors=None`` keeps the (down-sampled QR) sensor placement of the trained
    ROM; an integer re-selects that many QR-pivot sensors in the wake, and 0 measures
    the POD coefficients (the full fields). ``ensemble_kwargs=dict(update_reservoir=False)``
    leaves the reservoir out of the analysis, which then updates the POD coefficients
    only. `cfg` replaces the configuration ``pod_esn.yml``.

    Returns
    -------
    (EnsembleEstimator, Observations, dict)
        Filtered ensemble, truth, and the error metrics; with `return_case`, also the
        reduced-order model and the held-out ``(noisy, true)`` snapshots.
    """
    cfg = cfg or CFG
    (X_train, _), (X_filter, X_filter_true), simulation_dir = load_data(cfg=cfg, **(data_kwargs or {}))

    rom = build_rom(X_train, simulation_dir, cfg=cfg, N_val=(data_kwargs or {}).get('N_val'),
                    **(rom_kwargs or {}))
    if N_sensors is not None:
        rom = rom.copy()
    if N_sensors == 0:
        rom.select_sensors(measure_modes=True)
    elif N_sensors is not None:
        rom.select_sensors(N_sensors=N_sensors, **{**SENSORS, **cfg.get('sensors', {})})
    truth = build_truth(rom, X_filter, X_filter_true, cfg=cfg, **(truth_kwargs or {}))
    ensemble = build_ensemble(rom, da_method=da_method, cfg=cfg, **(ensemble_kwargs or {}))

    if t_extra is None:
        t_extra = 2. * rom.t_CR

    filter_ens = run_da_loop(ensemble, truth,
                             std_obs=cfg['da']['std_obs'] if std_obs is None else std_obs,
                             t_extra=t_extra)
    out = filter_ens, truth, ensemble_metrics(filter_ens, truth)
    return (*out, rom, (X_filter, X_filter_true)) if return_case else out


# --------------------------------------- Main ---------------------------------------- #

if __name__ == '__main__':

    args = main_parser(__doc__).parse_args()

    # State estimation only
    filter_ens, truth, metrics = run_experiment()
    print_metrics(metrics, header='POD-ESN cylinder wake (state estimation)')
    filter_ens.visualize_history(truth=truth, plot_members=True)

    # State estimation + online update of the Wout singular values
    filter_ens_W, truth_W, metrics_W = run_experiment(
        ensemble_kwargs=dict(est_alpha=['Wout'], std_alpha=0.02))
    print_metrics(metrics_W, header='POD-ESN cylinder wake (state + Wout estimation)')
    filter_ens_W.visualize_history(truth=truth_W, plot_members=True)

    os.makedirs(results_folder, exist_ok=True)
    save_to_pickle_file(f'{results_folder}cylinder_main_results',
                        truth, filter_ens, filter_ens_W)

    show_or_save_figs(args.save_figs, prefix='cylinder', figs_folder=figs_folder)
