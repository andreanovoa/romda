"""Figures 3-5 of Nóvoa & Magri (2025), ICCS 2025: online learning of a POD-ESN of the cylinder wake.

The settings are those of ``iccs2025.yml`` (next to this script): the paper's, and those of its code
(release v1.1) where the paper states none. Three estimation problems are compared with the EnSRKF:

- physical state estimation: the analysis updates the POD coefficients;
- two-fold estimation: the POD coefficients and the reservoir state;
- three-fold estimation: both, and the singular values of the output weights ``Wout``.

Figure 3 compares the first two with one sensor, Fig. 4 varies the number of sensors (1 to 4 by QR
pivoting, and the full fields) and Fig. 5 compares the two-fold and three-fold estimation with a
network trained on a truncated record, for three times between observations.

Usage
-----
    python scripts/podesn_cylinder/iccs2025.py [--figs 3 4 5] [--rerun] [--save-figs]

Each run is stored in ``results/iccs2025/`` next to this script, so the figures are redrawn without
filtering again; ``--rerun`` filters again.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts/: common

import pod_esn as case
from common import load_main_config, main_parser, output_folders, show_or_save_figs

results_folder, figs_folder = output_folders(__file__)   # gitignored, next to this script
results_folder = os.path.join(results_folder, 'iccs2025', '')

CFG = load_main_config('podesn_cylinder/iccs2025')
N_SHORT = CFG['data'].pop('N_short')
M_LARGE, STD_ALPHA = CFG['da'].pop('m_large'), CFG['da'].pop('std_alpha')

SENSORS = (1, 2, 3, 4, 0)            # 0: the full fields, through their POD coefficients
DELTAS = (10, 20, 30)                # time steps between observations of Fig. 5
T_EXTRA = 0.5                        # forecast after the last observation


# ----------------------------------- Experiments ------------------------------------ #

def mse(rom, Z, X_true):
    """Mean squared error of the fields decoded from the POD coefficients `Z`, shape ``(Nt, N_modes)``,
    against the flat true fields ``(N, Nt)``, over the two components and every grid point (Eq. 5)."""
    return np.sum((rom.decode(Z.T) - X_true) ** 2, axis=0) / np.prod(rom.grid_shape)


def run(key, N_sensors, short=False, three_fold=False, rerun=False, **kwargs):
    """Filter one case and store its errors (or load them): the squared error of each member, of the
    ensemble mean and of the noisy data, and the histories of the first POD coefficient and of the
    observables."""
    filename = f'{results_folder}{key}.npz'
    if os.path.isfile(filename) and not rerun:
        return dict(np.load(filename))

    data_kwargs = dict(N_train=N_SHORT - N_SHORT // 10, N_val=N_SHORT // 10) if short else None
    ensemble_kwargs = dict(kwargs.pop('ensemble_kwargs', {}))
    if three_fold:
        ensemble_kwargs.update(est_alpha=['Wout'], std_alpha=STD_ALPHA)

    filter_ens, truth, _, rom, (X_noisy, X_true) = case.run_experiment(
        cfg=CFG, N_sensors=N_sensors, data_kwargs=data_kwargs, ensemble_kwargs=ensemble_kwargs,
        t_extra=T_EXTRA, return_case=True, **kwargs)

    hist, t = filter_ens.model.hist, filter_ens.model.hist_t
    Nt = min(len(t), X_true.shape[1])                    # the model starts at the first held-out snapshot
    Z = hist[:Nt, :rom.N_modes]                          # Nt x N_modes x m
    X_true, X_noisy = (rom._to_flat(X[:, :Nt]) for X in (X_true, X_noisy))

    out = dict(t=t[:Nt],
               mse_members=np.stack([mse(rom, Z[..., j], X_true) for j in range(Z.shape[-1])], axis=-1),
               mse_mean=mse(rom, Z.mean(-1), X_true),
               mse_data=np.sum((X_noisy - X_true) ** 2, axis=0) / np.prod(rom.grid_shape),
               t_obs=truth.t_obs, y_obs=truth.y_obs,
               phi_1=Z[:, 0], y=filter_ens.model.get_observable_hist()[:Nt],
               y_raw=truth.y_raw[:Nt].reshape(Nt, -1), rho=rom.rho, sigma_in=rom.sigma_in, tikh=rom.tikh)
    os.makedirs(results_folder, exist_ok=True)
    np.savez(filename, **out)
    print(f'{key}: MSE of the ensemble mean after the last observation '
          f'{out["mse_mean"][t[:Nt] > truth.t_obs[-1]].mean():.2e} (noisy data {out["mse_data"].mean():.2e})')
    return out


# -------------------------------------- Figures -------------------------------------- #

def sensor_label(n):
    return 'full fields' if n == 0 else f'$N_d = {n}$'


def sensor_colours():
    return dict(zip(SENSORS, plt.get_cmap('viridis')(np.linspace(0.9, 0., len(SENSORS)))))


def mark_observations(ax, t_obs):
    for tk in t_obs:
        ax.axvline(tk, color='0.8', lw=0.5, zorder=-10)


def panel_letters(axs):
    for letter, ax in zip('abcdef', np.ravel(axs)):
        ax.text(0.01, 0.97, f'({letter})', transform=ax.transAxes, va='top')


def fig3(rerun=False):
    """State estimation against two-fold estimation: one sensor, m = 10."""
    runs = [run('fig3_state', 1, ensemble_kwargs=dict(update_reservoir=False), rerun=rerun),
            run('fig3_two_fold', 1, rerun=rerun)]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3), sharey=True, layout='constrained', num='fig3')
    for ax, r in zip(axs, runs):
        ax.semilogy(r['t'], r['mse_members'], color='C0', lw=0.5, alpha=0.6)
        ax.semilogy(r['t'], r['mse_members'].mean(-1), 'C0--', label='ensemble mean of the MSE')
        ax.semilogy(r['t'], r['mse_data'], 'k', lw=1, label='noisy data')
        mark_observations(ax, r['t_obs'])
        ax.set(xlabel='$t$', xlim=r['t'][[0, -1]])
    axs[0].set(ylabel='MSE')
    axs[0].legend(loc='lower left', fontsize='small')
    panel_letters(axs)


def fig4(rerun=False):
    """Number of sensors, two-fold estimation, m = 10."""
    runs = {n: run(f'fig4_Nd{n}', n, rerun=rerun) for n in SENSORS}
    fig = plt.figure(figsize=(9, 6), layout='constrained', num='fig4')
    ax_a, ax_b, ax_c = fig.subplot_mosaic([['a', 'b'], ['c', 'c']]).values()

    for ax, r, hist, raw, obs, label in [
            (ax_a, runs[0], runs[0]['phi_1'], runs[0]['y_raw'][:, 0], runs[0]['y_obs'][:, 0], '$\\phi_1$'),
            (ax_b, runs[1], runs[1]['y'][:, 1], runs[1]['y_raw'][:, 1], runs[1]['y_obs'][:, 1], '$u_y(x_1)$')]:
        ax.plot(r['t'], raw, color='0.6', lw=2, label='reference data')
        ax.plot(r['t'], hist, color='C0', lw=0.5)
        ax.plot(r['t_obs'], obs, 'r.', label='assimilated data')
        ax.set(xlabel='$t$', ylabel=label, xlim=r['t'][[0, -1]])
    ax_a.legend(loc='lower left', fontsize='small', ncols=2)

    for n, colour in sensor_colours().items():
        ax_c.semilogy(runs[n]['t'], runs[n]['mse_members'], color=colour, lw=0.5)
        ax_c.plot([], [], color=colour, label=sensor_label(n))
    ax_c.semilogy(runs[0]['t'], runs[0]['mse_data'], 'k', lw=1, label='noisy data')
    mark_observations(ax_c, runs[0]['t_obs'])
    ax_c.set(xlabel='$t$', ylabel='MSE', xlim=runs[0]['t'][[0, -1]])
    ax_c.legend(loc='upper right', fontsize='small', ncols=6)
    panel_letters([ax_a, ax_b, ax_c])


def fig5(rerun=False):
    """Two-fold against three-fold estimation with the network trained on the truncated record, m = 50."""
    fig, axs = plt.subplots(2, 3, figsize=(11, 5), sharex=True, sharey=True, layout='constrained', num='fig5')
    for row, three_fold in zip(axs, (False, True)):
        for ax, delta in zip(row, DELTAS):
            for n, colour in sensor_colours().items():
                r = run(f'fig5_{"three" if three_fold else "two"}_fold_Delta{delta}_Nd{n}', n, short=True,
                        three_fold=three_fold, truth_kwargs=dict(Nt_obs=delta),
                        ensemble_kwargs=dict(m=M_LARGE), rerun=rerun)
                ax.semilogy(r['t'], r['mse_mean'], color=colour, label=sensor_label(n))
            ax.semilogy(r['t'], r['mse_data'], 'k', lw=1, label='noisy data')
            mark_observations(ax, r['t_obs'])
            ax.set(xlim=r['t'][[0, -1]])
    [ax.set(xlabel='$t$') for ax in axs[-1]]
    [ax.set(ylabel='MSE') for ax in axs[:, 0]]
    axs[0, 0].legend(loc='lower left', fontsize='small', ncols=3)
    panel_letters(axs)


FIGURES = {3: fig3, 4: fig4, 5: fig5}


# --------------------------------------- Main ---------------------------------------- #

if __name__ == '__main__':

    parser = main_parser(__doc__)
    parser.add_argument('--figs', type=int, nargs='+', default=list(FIGURES), choices=list(FIGURES),
                        help='figures of the paper to make (default: all)')
    parser.add_argument('--rerun', action='store_true', help='filter again instead of loading the stored runs')
    args = parser.parse_args()

    for number in args.figs:
        FIGURES[number](rerun=args.rerun)

    show_or_save_figs(args.save_figs, prefix='iccs2025', figs_folder=figs_folder)
