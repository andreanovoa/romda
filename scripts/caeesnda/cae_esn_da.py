"""Real-time data assimilation with the convolutional autoencoder echo state network (CAE-ESN).

The cases of Özalp, Nóvoa & Magri (2026), as in tutorials 45 (Kuramoto-Sivashinsky) and 46 (Kolmogorov flow):
a trained autoencoder compresses the snapshots, an echo state network is trained on the reduced coordinates, and
the EnKF assimilates sparse, noisy measurements of the field into an ensemble of CAE-ESNs that starts unrelated to
the true state. The settings are in ``ks.yml`` and ``kolmogorov.yml`` next to this script:

    python scripts/caeesnda/cae_esn_da.py ks --save-figs
    python scripts/caeesnda/cae_esn_da.py kolmogorov --set da.m=20     # needs the record in $DATA_ROOT

The CAE-ESN is a data-driven ROM trained on snapshots, so these cases run through this script, not through
``romda.run``. The script prints the error of the decoded ensemble mean and saves it, with the spread, to
``results/<case>.json``.
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts/: common

import romda.models.data_driven as dd
from common import main_parser, output_folders, print_metrics, show_or_save_figs
from romda.data_assimilation import run_da_loop
from romda.observations import Observations
from romda.plotting import Palette
from romda.run import _apply_override, load_config, resolve_da_method

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
results_folder, figs_folder = output_folders(__file__)   # gitignored, next to this script


def load_record(rec):
    """The snapshots (N_t, Nx, Ny) of the record after the transient, and their time step."""
    n = rec['N_trans'] + rec['N_train'] + rec['N_val'] + rec['N_test']
    if rec['source'] == 'ks':                    # integrated here: anyone can run this case
        from dynamodels.physical import KS
        ks = KS(Nx=rec['Nx'], L=rec['L_over_pi'] * np.pi, dt=rec['dt'], seed=rec['seed'])
        psi, _ = ks.time_integrate(Nt=n)
        U = np.fft.irfft(psi[rec['N_trans']:, :, 0], axis=1)[..., None]     # the state is in Fourier space
    else:                                        # a record of snapshots in $DATA_ROOT, not distributed
        import h5py
        path = os.path.join(os.environ.get('DATA_ROOT', 'data'), rec['path'])
        if not os.path.isfile(path):
            sys.exit(f'{path} not found: this record is not distributed with romda (see scripts/caeesnda/README.md)')
        with h5py.File(path, 'r') as f:
            U = np.asarray(f[rec['dataset']][rec['N_trans']:n])
    return U, rec['dt']


def load_projector(cfg, grid_shape):
    """The trained autoencoder: the state dictionary of the paper (.pth), on the grid `grid_shape` (Nu, Nx, Ny), or a
    romda save (.json options + .npz arrays)."""
    import torch
    cls, weights = getattr(dd, cfg['class']), os.path.join(ROOT, cfg['weights'])
    if weights.endswith('.pth'):
        return cls.load_state_dicts(torch.load(weights, map_location='cpu'), grid_shape=grid_shape, scale=cfg['scale'])
    with open(weights + '.json') as f:
        options = json.load(f)
    return cls.from_trained(options, dict(np.load(weights + '.npz')),
                            device='cuda' if torch.cuda.is_available() else 'cpu')


def sensor_grid(n_sensors, shape):
    """`n_sensors` equispaced grid points of a ring (Ny = 1), or a regular square grid of them on a plane,
    as raw grid indices x * Ny + y."""
    Nx, Ny = shape
    if Ny == 1:
        return np.linspace(0, Nx - 1, n_sensors).round().astype(int)
    k = int(round(np.sqrt(n_sensors)))
    xx, yy = np.meshgrid(np.linspace(0, Nx - 1, k, dtype=int), np.linspace(0, Ny - 1, k, dtype=int), indexing='ij')
    return np.sort((xx * Ny + yy).ravel())


def run_case(cfg):
    """Train the CAE-ESN, assimilate, and return the filtered ensemble, the truth, the test record and the scores."""
    fc, tr = cfg['forecast'], cfg['truth']
    rec, esn = fc['record'], dict(fc['esn'])
    U, dt = load_record(rec)
    N_fit = rec['N_train'] + rec['N_val']
    U_test = U[N_fit:]
    projector = load_projector(fc['projector'], grid_shape=(1, *U.shape[1:]))
    sensors = sensor_grid(fc['rom']['n_sensors'], U.shape[1:])

    # the ESN trains on the reduced coordinates of the training (and validation) snapshots
    N_val_esn = esn.pop('N_val', None)
    data = U[:rec['N_train']] if N_val_esn else U[:N_fit]
    N_val_esn = N_val_esn or rec['N_val']
    rom = getattr(dd, fc['rom']['class'])(
        data=data[None], dt=dt, projector=projector, sensor_locations=sensors,
        t_train=(len(data) - N_val_esn) * dt, t_val=N_val_esn * dt, t_test=0., upsample=1,
        hyperparameters_to_optimize=[], bias_in=np.array([1.]), plot_training=False, **esn)

    # measurements of the test snapshots at the sensors
    U_flat = U_test.reshape(len(U_test), -1)
    scale = {'max': np.abs(U[:N_fit]).max(axis=0), 'std': U[:N_fit].std(axis=0)}[tr['noise_type']]
    sigma = tr['noise_level'] * scale.reshape(-1)[sensors]
    rng = np.random.default_rng(tr['seed'])
    t_lyap, n_obs = fc['t_lyap'], tr['Nt_obs']
    truth = Observations(y_raw=U_flat[:, sensors] + sigma * rng.standard_normal((len(U_test), len(sensors))),
                         y_true=U_flat[:, sensors], t_true=np.arange(len(U_test)) * dt,
                         t_start=n_obs * dt, t_stop=tr['window_lyap'] * t_lyap, Nt_obs=n_obs)

    da = cfg['da']
    ensemble = resolve_da_method(da['method'])(parent_model=rom, m=da['m'], inflation=da['inflation'])
    filt = run_da_loop(ensemble, truth, Cdd=np.diag(sigma ** 2), t_extra=fc['t_extra_lyap'] * t_lyap)
    return filt, truth, U_flat, scores(filt, rom, U_flat, truth, t_lyap)


def scores(filt, rom, U_flat, truth, t_lyap):
    """Error of the decoded ensemble mean and ensemble spread, normalized by the root-mean-square of the true field,
    against time; scalar summaries over the assimilation window (after one Lyapunov time) and the forecast."""
    model = filt.model
    t = np.asarray(model.hist_t)
    keep = np.r_[t[1:] != t[:-1], True]                         # at the observation times, keep the analysis
    t, Z = t[keep], model.hist[keep, :rom.N_latent]
    idx = np.rint(t / rom.dt).astype(int)
    members = np.stack([rom.decode(Z[:, :, j].T).T for j in range(Z.shape[-1])])
    rms = np.sqrt(np.mean(np.sum(U_flat ** 2, axis=1)))
    error = np.linalg.norm(members.mean(0) - U_flat[idx], axis=1) / rms
    spread = np.sqrt(members.var(0).sum(-1)) / rms
    window, after = (t > t_lyap) & (t <= truth.t_obs[-1]), t > truth.t_obs[-1]
    return dict(t_lyap=(t / t_lyap).tolist(), error=error.tolist(), spread=spread.tolist(),
                error_window=float(error[window].mean()), spread_window=float(spread[window].mean()),
                error_forecast=float(error[after].mean()), error_initial=float(error[0]))


def plot_scores(s, truth, t_lyap):
    fig, ax = plt.subplots(figsize=(7, 3), num='error')
    ax.plot(s['t_lyap'], s['error'], color=Palette.UNBIASED, label='error of the ensemble mean')
    ax.plot(s['t_lyap'], s['spread'], '--', color=Palette.BIASED, label='ensemble spread')
    ax.axvspan(truth.t_obs[0] / t_lyap, truth.t_obs[-1] / t_lyap, color=Palette.OBS, alpha=0.1, label='measurements')
    ax.set(xlabel='time [Lyapunov times]', ylabel='normalized error', xlim=[0, s['t_lyap'][-1]])
    ax.legend(fontsize=8)
    fig.tight_layout()


if __name__ == '__main__':
    parser = main_parser(__doc__)
    parser.add_argument('case', choices=['ks', 'kolmogorov'], help='configuration <case>.yml next to this script')
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                        help='override one value of the configuration, e.g. --set da.m=20')
    args = parser.parse_args()

    cfg = load_config(os.path.join(HERE, f'{args.case}.yml'))
    for item in args.set:
        key, value = item.split('=', 1)
        _apply_override(cfg, key, yaml.safe_load(value))

    filt, truth, _, s = run_case(cfg)
    print_metrics({k: v for k, v in s.items() if not isinstance(v, list)}, header=f'CAE-ESN, {args.case}')
    os.makedirs(results_folder, exist_ok=True)
    with open(f'{results_folder}{args.case}.json', 'w') as f:
        json.dump({'config': cfg, **s}, f)
    plot_scores(s, truth, cfg['forecast']['t_lyap'])
    show_or_save_figs(args.save_figs, prefix=args.case, figs_folder=figs_folder)
