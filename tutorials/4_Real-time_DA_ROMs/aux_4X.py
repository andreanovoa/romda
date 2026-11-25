"""Helpers of tutorials 40 and 41, the washout-free deployment of echo state networks: the configurations of the study in
`scripts/esnda`, the closed-loop forecasts after a washout, and the figures.

Colours (Okabe-Ito): the true state is grey or black, the ensemble filter blue, the substitution of the data
vermillion, the reference without data green. Errors are normalized by `romda.metrics.error_scale` of the truth.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from romda.metrics import error_scale, state_error, valid_time
from romda.plotting import COLORS, plot_error_spread, plot_scores, space_time, use_tutorial_style
from romda.run import load_config

BLUE, VERMILLION, GREEN, ORANGE, GREY = (COLORS[k] for k in ('blue', 'vermillion', 'green', 'orange', 'grey'))
use_tutorial_style()


def config(name):
    """The configuration `scripts/esnda/configs/<name>.yml`, from the tutorial folder or the repository root."""
    folder = next(p for p in (Path('../../scripts/esnda/configs'), Path('scripts/esnda/configs')) if p.is_dir())
    return load_config(folder / f'{name}.yml')


def washout_forecasts(esn, record, clean, starts, n_steps):
    """Closed-loop forecasts of `n_steps` after a washout on `record` from each of `starts`: the last forecast, its
    target (from `clean`) and the normalized error of every forecast."""
    scale, errors = error_scale(clean), []
    for i0 in starts:
        target = clean[i0::esn.upsample][esn.N_wash:esn.N_wash + n_steps]
        forecast, _ = esn.closed_loop(record[i0::esn.upsample][:esn.N_wash + n_steps], n_steps)
        errors.append(np.linalg.norm(forecast - target, axis=1) / scale)
    return forecast, target, errors


def valid_times(errors, t):
    """Valid time (error above 0.5) of each error series, in the units of `t`."""
    return np.array([valid_time(e, t) for e in errors])


def _title(ax, s):
    ax.set_title(s, loc='left', fontsize=10)


def _errors_panel(ax, t, errors):
    ax.semilogy(t, np.array(errors).T, color=BLUE, lw=0.5, alpha=0.5)
    ax.axhline(0.5, color='k', ls=':')
    ax.set(xlabel=r'$t/T_\lambda$ after the washout', ylabel='error', ylim=[1e-3, 3], xlim=[0, t[-1]])


def plot_washout(t, target, forecast, errors, space=False):
    """One closed-loop forecast against the truth, as x (or as space-time diagrams when `space`), and the error of
    every forecast."""
    if space:
        fig, axs = plt.subplots(3, 1, figsize=(6.5, 4.2), sharex=True)
        vmax = np.abs(target).max()
        sites = np.arange(target.shape[1])
        space_time(axs[0], t, sites, target, vmax)
        fig.colorbar(space_time(axs[1], t, sites, forecast, vmax), ax=axs[:2], label='$u_i$')
        for ax, s in zip(axs, ['true', 'ESN, closed loop']):
            ax.set(ylabel='site')
            _title(ax, s)
    else:
        fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.4), width_ratios=[1.6, 1])
        axs[0].plot(t, target[:, 0], color=GREY, lw=2, label='true')
        axs[0].plot(t, forecast[:, 0], '--', color=BLUE, label='ESN, closed loop')
        axs[0].set(xlabel=r'$t/T_\lambda$ after the washout', ylabel='$x$', xlim=[0, 8])
        axs[0].legend(ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1))
        t = t[t <= 8]
        errors = [e[:len(t)] for e in errors]
    _errors_panel(axs[-1], t, errors)
    plt.show()


def plot_members(run, truth, t, clean, t_lyap, measured=(0, 1, 2), n_lyap=6):
    """The members of a deployment against the truth (grey) and the data (dots), over the first `n_lyap` Lyapunov
    times of data."""
    fig, axs = plt.subplots(3, 1, figsize=(6.5, 4), sharex=True)
    members = run['ensemble'].model.hist[:, :3] if run['ensemble'] is not None else run['y_members']
    t0, t1 = truth.t_obs[0] - 0.5 * t_lyap, truth.t_obs[0] + n_lyap * t_lyap
    w, w_true, w_obs = (run['t'] >= t0) & (run['t'] <= t1), (t >= t0) & (t <= t1), truth.t_obs <= t1
    for k, (ax, name) in enumerate(zip(axs, 'xyz')):
        ax.plot(t[w_true] / t_lyap, clean[w_true, k], color=GREY, lw=2)
        ax.plot(run['t'][w] / t_lyap, members[w, k], color=BLUE, lw=0.4, alpha=0.5)
        if k in measured:
            ax.plot(truth.t_obs[w_obs] / t_lyap, truth.y_obs[w_obs, list(measured).index(k)], '.k', ms=4)
        lim = np.abs(clean[:, k] - clean[:, k].mean()).max()
        ax.set(ylabel=f'${name}$', ylim=clean[:, k].mean() + 1.3 * lim * np.array([-1, 1]), xlim=[t0 / t_lyap, t1 / t_lyap])
    axs[-1].set_xlabel(r'$t/T_\lambda$')
    plt.show()


def plot_errors(runs, truth, t, clean, t_lyap):
    """Error of the ensemble mean (lines) and spread (dashed) of each deployment, {label: (run, colour)}; the data
    window is shaded."""
    fig, ax = plt.subplots(figsize=(6.5, 2.5))
    ax.axvspan(truth.t_obs[0] / t_lyap, truth.t_obs[-1] / t_lyap, color='k', alpha=0.06, lw=0)
    for label, (run, color) in runs.items():
        t_run, error, spread = state_error(run, t, clean, per_member=False)
        plot_error_spread(ax, t_run / t_lyap, error, spread, color=color, label=label)
    ax.plot([], [], '--k', lw=0.8, label='spread')
    ax.set(xlabel=r'$t/T_\lambda$', ylabel='error', ylim=[3e-3, 5], xlim=[0, t_run[-1] / t_lyap])
    ax.legend(ncol=len(runs) + 1, loc='lower center', bbox_to_anchor=(0.5, 1))
    plt.show()


def plot_sweep(x, results, xlabel, ylabels, logy=(True, False), logx=False):
    """Scores against a setting, {label: (array (len(x), windows, n_scores), colour)}: medians (lines) and
    interquartile ranges (bands), one panel per score."""
    n = len(ylabels)
    fig, axs = plt.subplots(1, n, figsize=(6.5 if n > 1 else 4.5, 2.5), squeeze=False)
    for label, (res, color) in results.items():
        for j, ax in enumerate(axs[0]):
            plot_scores(ax, x, res[..., j], color=color, label=label or None)
    for ax, ylabel, log in zip(axs[0], ylabels, logy):
        ax.set(xlabel=xlabel, ylabel=ylabel, yscale='log' if log else 'linear', xscale='log' if logx else 'linear')
    if any(results):
        axs[0, 0].legend(fontsize=9)
    plt.show()


def plot_space_time(rows, time, t_first):
    """Space-time diagrams, {title: field (N_t, N_sites)}; the last row is an absolute error. The first datum is
    dotted."""
    fig, axs = plt.subplots(len(rows), 1, figsize=(6.5, 1.4 * len(rows)), sharex=True)
    fields = list(rows.values())
    vmax, sites = np.abs(fields[0]).max(), np.arange(fields[0].shape[1])
    for ax, (title, field) in zip(axs, rows.items()):
        im = (space_time(ax, time, sites, field, vmax / 2, vmin=0, cmap='Reds') if ax is axs[-1]
              else space_time(ax, time, sites, field, vmax))
        ax.axvline(t_first, color='k', lw=0.8, ls=':')
        ax.set(ylabel='site')
        _title(ax, title)
        if ax in axs[-2:]:
            fig.colorbar(im, ax=ax if ax is axs[-1] else axs[:-1])
    axs[-1].set_xlabel(r'$t/T_\lambda$')
    plt.show()


def plot_components(runs, truth, t, clean, t_lyap, measured, n_lyap=8):
    """x, y, z of each estimate, {label: (run, colour)}, against the truth (thick grey) and the data (dots)."""
    t0, t1 = truth.t_obs[0] - 0.5 * t_lyap, truth.t_obs[0] + n_lyap * t_lyap
    fig, axs = plt.subplots(3, 1, figsize=(6.5, 4.6), sharex=True)
    w_true, w_obs = (t >= t0) & (t <= t1), truth.t_obs <= t1
    for k, (ax, name) in enumerate(zip(axs, 'xyz')):
        ax.plot(t[w_true] / t_lyap, clean[w_true, k], color='k', lw=2.5, alpha=0.35, label='true')
        for label, (run, color) in runs.items():
            w = (run['t'] >= t0) & (run['t'] <= t1)
            thin = dict(lw=0.4, alpha=0.5, zorder=0) if color == VERMILLION else dict(lw=0.9)
            ax.plot(run['t'][w] / t_lyap, run['y'][w, k], color=color, label=label, **thin)
        if k in measured:
            ax.plot(truth.t_obs[w_obs] / t_lyap, truth.y_obs[w_obs, measured.index(k)], '.k', ms=4, label='data')
        lim = np.abs(clean[:, k] - clean[:, k].mean()).max()
        ax.set(ylabel=f'${name}$' + ('' if k in measured else ' (not measured)'), xlim=[t0 / t_lyap, t1 / t_lyap],
               ylim=clean[:, k].mean() + 1.3 * lim * np.array([-1, 1]))
    axs[0].legend(ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1), fontsize=9)
    axs[-1].set_xlabel(r'$t/T_\lambda$')
    plt.show()


def plot_dot_rows(table, labels, colors, legend=('three components', '$y$')):
    """Two scores per window (filled and open symbols), one row per method: table (methods, windows, 2)."""
    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(labels) + 0.6))
    for k, color in enumerate(colors):
        ax.plot(table[k, :, 0], np.full(table.shape[1], k) + 0.12, 'o', color=color, ms=3.5, alpha=0.6)
        ax.plot(table[k, :, 1], np.full(table.shape[1], k) - 0.12, 's', color=color, ms=3.5, mfc='none')
    ax.plot([], [], 'ok', ms=3.5, label=legend[0])
    ax.plot([], [], 'sk', ms=3.5, mfc='none', label=legend[1])
    ax.set(xscale='log', xlabel='error', yticks=range(len(labels)), yticklabels=labels, ylim=[len(labels) - 0.5, -0.5])
    ax.legend(ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1))
    plt.show()


def plot_parameter(run, runs, truth, t, clean, t_lyap, rho_true, rho_train):
    """The parameter of the members, their mean and the truth (top); the error of each deployment (bottom),
    {label: (run, colour)}."""
    model = run['ensemble'].model
    t_run, rho = model.hist_t / t_lyap, model.hist[:, model.Nphi]
    fig, axs = plt.subplots(2, 1, figsize=(6.5, 4), sharex=True, height_ratios=[1.2, 1])
    axs[0].axhspan(rho_train.min(), rho_train.max(), color='k', alpha=0.06, lw=0, label='training range')
    axs[0].plot(t_run, rho, color=BLUE, lw=0.3, alpha=0.4)
    axs[0].plot(t_run, rho.mean(axis=1), color=BLUE, lw=1.5, label='ensemble mean')
    axs[0].axhline(rho_true, color='k', ls='--', lw=1, label='true')
    axs[0].set(ylabel=r'$\rho$', ylim=[22, 40])
    axs[0].legend(ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1))
    for label, (r, color) in runs.items():
        t_e, error, _ = state_error(r, t, clean, per_member=False)
        axs[1].semilogy(t_e / t_lyap, error, color=color, lw=0.8, label=label)
    axs[1].axvline(truth.t_obs[-1] / t_lyap, color='k', ls=':', lw=0.8)
    axs[1].set(xlabel=r'$t/T_\lambda$', ylabel='error', ylim=[3e-3, 5], xlim=[0, t_e[-1] / t_lyap])
    axs[1].legend(ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1))
    plt.show()


def plot_parameter_scan(rho_cases, scan, rho_train):
    """Estimated against true parameter, with bars of two ensemble standard deviations (left); the error of the
    state (right). `scan` (cases, 3): mean, std and state error. The training range is shaded."""
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.5))
    axs[0].plot([20, 44], [20, 44], 'k--', lw=0.8)
    axs[0].errorbar(rho_cases, scan[:, 0], yerr=2 * scan[:, 1], fmt='o', color=BLUE, ms=4, capsize=2)
    axs[0].set(ylabel=r'estimated $\rho$', ylim=[20, 44])
    axs[1].semilogy(rho_cases, scan[:, 2], 'o', color=BLUE, ms=4)
    axs[1].set(ylabel='error of the state', ylim=[1e-2, 1])
    for ax in axs:
        ax.axvspan(rho_train.min(), rho_train.max(), color='k', alpha=0.06, lw=0)
        ax.set(xlabel=r'true $\rho$', xlim=[20, 44])
    plt.show()
