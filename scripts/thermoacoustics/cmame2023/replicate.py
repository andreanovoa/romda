"""Main results of Nóvoa, Racca & Magri (2023), Comput. Methods Appl. Mech. Eng. 418, 116502, from romda runs.

The paper infers the state, the parameters and the model bias of a thermoacoustic low-order model with the
regularized bias-aware ensemble Kalman filter (r-EnKF), in which an echo state network estimates the bias. This
script repeats its two test cases with the settings of the paper (``vdp.yml`` and ``rijke.yml``, next to this
script) and redraws their main figures:

- the Van der Pol oscillator with the bias b = cos(eta): figure 6 (the error after the assimilation for four
  training sets of the network and three regularization factors), figure 7 (the time series for
  gamma = 0, 10 and 20 with L = 50) and figure 8 (the errors against gamma for L = 1, 10 and 50);
- the time-delayed Rijke tube with a linear and a nonlinear bias: figure 10 (the error after the assimilation
  over L and gamma), figure 11 (the errors against gamma at L = 70), figure 12 (the time series at the minimum
  of figure 10) and Table 2 (the errors at the minima).

    python scripts/thermoacoustics/cmame2023/replicate.py vdp [--save-figs] [--quick]
    python scripts/thermoacoustics/cmame2023/replicate.py rijke [--save-figs] [--quick] [--bias linear]

Every run is recorded in ``results/<case>.jsonl`` as it finishes, so that an interrupted call resumes; with
``--plot`` the figures are drawn from the records, and only their time series are run again (a run is
deterministic). The numbers of Table 2 are printed and written to ``results/table2.md``. romda implements the
r-EnKF of the erratum of the paper, whereas the published numbers of the Rijke tube, which has six observables,
come from the form before the erratum: the conclusions are compared, not the digits (see the README).
"""

import copy
import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # scripts/: common

import matplotlib.pyplot as plt
import numpy as np
from common import load_main_config, main_parser, output_folders, show_or_save_figs
from matplotlib.colors import LogNorm, Normalize
from romda.metrics import relative_error
from romda.sweep import dedupe, filter_context, run_point, run_sweep
from romda.utils import interpolate
from scipy.interpolate import interp1d

results_folder, figs_folder = output_folders(__file__)   # gitignored, next to this script

CASES = dict(vdp=load_main_config('thermoacoustics/cmame2023/vdp'),
             rijke=load_main_config('thermoacoustics/cmame2023/rijke'))

#: A variant of the Van der Pol case that is not in the paper: a prior 50 % from the truth (75, 55, 3.4), to test
#: the inference of the parameters. beta and zeta start 50 % below the truth (37.5, 27.5), which keeps the growth
#: rate beta - zeta of the prior model positive and both away from their upper limit of 120, and kappa 50 % above
#: (5.1). The standard deviation is 50 % of the mean for beta and zeta, so that the truth is two standard deviations
#: away (with the 25 % of the paper it would be four), and 25 % for kappa. The ranges are [mean - 2 sigma,
#: mean + 2 sigma]; the training members of the network are uniform within 50 % of the prior mean, as in the paper.
FAR_PRIOR = dict(da=dict(beta=[0., 75.], zeta=[0., 55.], kappa=[2.55, 7.65]),
                 bias=dict(beta=[18.75, 56.25], zeta=[13.75, 41.25], kappa=[2.55, 7.65]))
CASES['vdp_far_prior'] = copy.deepcopy(CASES['vdp'])
CASES['vdp_far_prior']['name'] = 'vdp_cmame2023_far_prior'
CASES['vdp_far_prior']['da']['std_alpha'] = FAR_PRIOR['da']
CASES['vdp_far_prior']['bias']['std_alpha'] = FAR_PRIOR['bias']

#: figure 6: the training sets of the network, (L, data augmentation), and the regularization factors
FIG6_SETS = ((1, False), (1, True), (10, True), (50, True))
FIG6_GAMMAS = (0., 10., 20.)
#: figure 8: the number of training members and the regularization factors
FIG8_LS = (1, 10, 50)
FIG8_GAMMAS = tuple(float(g) for g in range(21))
#: figure 10: the paper sweeps L = 10, 20, ..., 100 and gamma = 0, 0.25, ..., 10 (41 values); here gamma has a
#: step of 0.25 or 0.5 up to 5, where the minima are, with the minima of the paper, and a step of 1 beyond
RIJKE_LS = tuple(range(10, 101, 10))
RIJKE_GAMMAS = (0., 0.5, 1., 1.25, 1.5, 1.75, 2., 2.25, 2.5, 2.75, 3., 3.5, 4., 4.5, 5., 6., 7., 8., 9., 10.)
RIJKE_L_FIG11 = 70
#: the biases of the Rijke tube: the name in the paper and the `manual_bias` of `romda.observations.Observations`
BIASES = dict(linear='linear', nonlinear='periodic')
#: Table 2 of the paper: the error of the true biased signal, and (L, gamma, biased error, unbiased error) at the
#: minimum of the sum of the two errors at the end of the assimilation ('DA') and after it ('post')
TABLE_2 = dict(linear=dict(true=0.2623, DA=(10, 3.50, 0.1761, 0.0244), post=(100, 1.75, 0.1817, 0.0157)),
               nonlinear=dict(true=0.2217, DA=(60, 2.75, 0.2303, 0.0799), post=(60, 2.75, 0.2279, 0.0792)))

# Paul Tol's colour-blind-safe palette
C = dict(truth='#BBBBBB', obs='#EE6677', biased='#66CCEE', unbiased='#332288', bias='#AA3377')
C_PARAM = dict(beta='#4477AA', zeta='#CC6677', kappa='#117733', tau='#DDAA33')
LABEL = dict(beta=r'\beta', zeta=r'\zeta', kappa=r'\kappa', tau=r'\tau')
#: the axis of the parameters over their true values, for the cases whose prior is far from the truth
PARAMETER_LIMITS = dict(vdp_far_prior=(0., 2.))


def letter(ax, text, x=-0.02, y=1.03):
    """The panel letter, bold, above the top-left corner of `ax` (axes fraction)."""
    ax.text(x, y, f'({text})', transform=ax.transAxes, fontweight='bold', ha='right', va='bottom')


def legend_above(fig, axs):
    """One legend above the panels of `fig`, with the labelled lines of the axes `axs`."""
    handles, labels = [], []
    for ax in axs:
        h, text = ax.get_legend_handles_labels()
        handles, labels = handles + h, labels + text
    fig.legend(handles, labels, loc='outside upper center', ncols=len(labels), frameon=False, fontsize='small')


# --------------------------------------------- runs and errors --------------------------------------------- #

def training_ranges(config, L):
    """The options of the bias block for `L` training members. A single member takes the mean of the prior of the
    parameters (the middle of their ranges in the ``da`` block) and the state of the model, as in the paper; the
    others are drawn from the ranges of the configuration."""
    if L > 1:
        return dict(L=L)
    prior = config['da']['std_alpha']
    return dict(L=1, std_phi=0., std_alpha={name: [float(np.mean(prior[name]))] * 2 for name in config['bias']['std_alpha']})


def run(case, L, gamma, augment=True, bias=None, da_method=None, seed=0):
    """One assimilation of the case `case` ('vdp' or 'rijke') with `L` training members of the network and the
    regularization factor `gamma`; `bias` is the `manual_bias` of the truth (default: that of the configuration).

    Returns
    -------
    (EnsembleEstimator, Observations)
        The filtered ensemble, forecast for two `t_CR` after the last observation, and the truth.
    """
    config = CASES[case]
    model = getattr(__import__('romda.models.physical', fromlist=['physical']), config['truth']['model'])
    filter_ens, truth, _ = run_point(config, regularization_factor=gamma, manual_bias=bias, seed=seed,
                                     da_method=da_method, t_extra=2 * model.t_CR,
                                     bias_kwargs=dict(augment_data=augment, **training_ranges(config, L)))
    return filter_ens, truth


def signals(filter_ens, truth):
    """The signals of a run on the time steps of the model: the time, the truth `d` and its bias `b_true` (Nt, Nq),
    the model observables `q` and the bias-corrected observables `y` of every member (Nt, Nq, m). The network
    estimates the bias every `upsample` time steps; between them the bias is a cubic spline, as in the code of
    the paper (the history of the bias in romda is linear between the network steps, which leaves an error of
    4.7 % in the Van der Pol case, against 2.2 % for the spline)."""
    t = filter_ens.model.hist_t
    q = filter_ens.model.get_observable_hist()
    y, bias = q, filter_ens.bias
    if getattr(filter_ens, 'is_bias_aware', False):
        b = bias.get_bias_hist()                                            # (Nt, Nq, 1), on the model steps
        steps = np.arange(len(bias.hist_t) - 1, -1, -int(bias.upsample))[::-1]     # the steps of the network
        y = q + interp1d(bias.hist_t[steps], b[steps], kind='cubic', axis=0, bounds_error=False,
                         fill_value=(b[steps[0]], b[steps[-1]]))(t)
    d = interpolate(truth.t_true, truth.y_true[..., 0], t)
    b_true = interpolate(truth.t_true, truth.b_true[..., 0], t)
    return dict(t=t, d=d, b_true=b_true, q=q, y=y)


def scores(filter_ens, truth):
    """The errors of a run, as the paper defines them (section 6.1 and Table 1): the root-mean-square (RMS) error
    is `romda.metrics.relative_error` between the truth, which includes the bias, and an estimate. 'biased' is the
    ensemble mean of the model observables and 'unbiased' adds the bias estimate. The windows last one `t_CR`:
    'DA' ends at the last observation and 'post' starts there; 'pre' runs from the start of the ensemble to the
    first observation. ``R_true`` is the error of the true unbiased signal after the assimilation, i.e. the norm
    of the true bias over that of the truth. The mean absolute errors, ``mae_*``, are averaged over consecutive
    windows of one `t_CR` and over the observables, and normalized by the maximum of the truth."""
    model, s = filter_ens.model, signals(filter_ens, truth)
    t, d, t_cr = s['t'], s['d'], filter_ens.model.t_CR
    q, y = np.mean(s['q'], axis=-1), np.mean(s['y'], axis=-1)
    t0, t1 = truth.t_obs[0], truth.t_obs[-1]
    windows = dict(pre=t < t0 - 1e-12, DA=(t >= t1 - t_cr - 1e-12) & (t < t1 - 1e-12),
                   post=(t >= t1 - 1e-12) & (t < t1 + t_cr - 1e-12))
    out = dict(rejected=len(filter_ens.rejected_analysis.times),
               R_true=float(relative_error(d[windows['post']], (d - s['b_true'])[windows['post']])))
    for name, w in windows.items():
        out[f'R_biased_{name}'] = float(relative_error(d[w], q[w]))
        out[f'R_unbiased_{name}'] = float(relative_error(d[w], y[w]))
    # each member after the assimilation: the bars of figure 6 and their spread
    w = windows['post']
    for name, members in (('biased', s['q']), ('unbiased', s['y'])):
        errors = relative_error(d[w], members[w], axis=(0, 1))
        out[f'R_{name}_post_members'], out[f'R_{name}_post_std'] = float(errors.mean()), float(errors.std())
    edges = np.arange(0., t[-1] + 1e-12, t_cr)
    scale = np.max(abs(d), axis=0)
    out['mae_t'] = [float(e) for e in edges[:-1] + t_cr / 2]
    for name, estimate in (('biased', q), ('unbiased', y)):
        out[f'mae_{name}'] = [float(np.mean(abs(d - estimate)[(t >= a) & (t < b)] / scale))
                              for a, b in zip(edges[:-1], edges[1:])]
    i0, i1 = np.argmin(abs(t - t0)) - 1, np.argmin(abs(t - t1))      # the forecast before the first analysis
    for k, name in enumerate(filter_ens.est_alpha):
        for tag, i in (('start', i0), ('end', i1)):
            row = model.hist[i, model.Nphi + k]
            out[f'{name}_{tag}'], out[f'std_{name}_{tag}'] = float(row.mean()), float(row.std(ddof=1))
    return out


def experiment(case, seed=0, L=None, regularization_factor=None, augment=True, bias=None, da_method=None):
    """One run of a sweep, for `romda.sweep.run_sweep`: the `scores` of the run."""
    filter_ens, truth = run(case, L, regularization_factor, augment=augment, bias=bias, da_method=da_method,
                            seed=seed)
    out = scores(filter_ens, truth)
    filter_ens.close()
    del filter_ens, truth
    gc.collect()                       # the ensembles sit in reference cycles, which fill the memory in a sweep
    return out


def sweep(case, axes, name=None, **fixed):
    """The runs of `axes` (``L``, ``regularization_factor``, ``augment``) for the case `case` with the keywords
    `fixed` of `experiment`, recorded in ``results/<name>.jsonl`` and stamped with the configuration: the runs
    found there under the same configuration are not repeated. Returns ``{(L, augment, gamma): scores}``."""
    context = dict(config=CASES[case], **fixed)
    axes = {'augment': [True], **axes}
    records = run_sweep(lambda seed=0, **point: experiment(case, seed=seed, **fixed, **point), axes,
                        os.path.join(results_folder, f'{name or case}.jsonl'), n_seeds=1, context=context)
    table = {}
    for record in dedupe(filter_context(records, context), axes):
        if record['status'] != 'ok':
            raise RuntimeError(f'The run { {key: record[key] for key in axes} } failed; see the traceback above.')
        table[(record['L'], record['augment'], record['regularization_factor'])] = record
    return table


def stored(name, context):
    """The records of ``results/<name>.jsonl`` made under `context`, ``{(L, augment, gamma): scores}``."""
    from romda.sweep import load_results
    records = [r for r in filter_context(load_results(os.path.join(results_folder, f'{name}.jsonl')), context)
               if r.get('status') == 'ok']
    return {(r['L'], r['augment'], r['regularization_factor']): r for r in records}


# ------------------------------------------------- figures ------------------------------------------------- #

def plot_training_sets(table):
    """Figure 6: the RMS errors after the assimilation for the training sets `FIG6_SETS` and the factors
    `FIG6_GAMMAS`. Wide bars: biased; narrow bars: unbiased; error bars: standard deviation over the members."""
    fig, ax = plt.subplots(figsize=(7.5, 3.), layout='constrained')
    colours = plt.get_cmap('viridis')(np.linspace(0.15, 0.85, len(FIG6_GAMMAS)))
    width = 0.8 / len(FIG6_GAMMAS)
    for i, (L, augment) in enumerate(FIG6_SETS):
        for j, (gamma, colour) in enumerate(zip(FIG6_GAMMAS, colours)):
            r, x = table[(L, augment, gamma)], i + (j - (len(FIG6_GAMMAS) - 1) / 2) * width
            ax.bar(x, r['R_biased_post_members'], width=0.9 * width, color=colour, alpha=0.45,
                   yerr=r['R_biased_post_std'], capsize=2, label=rf'$\gamma = {gamma:g}$, biased' if i == 0 else None)
            ax.bar(x, r['R_unbiased_post_members'], width=0.35 * width, color=colour, ec='k', lw=0.5,
                   yerr=r['R_unbiased_post_std'], capsize=2,
                   label=rf'$\gamma = {gamma:g}$, unbiased' if i == 0 else None)
    reference = table[(FIG6_SETS[0][0], FIG6_SETS[0][1], FIG6_GAMMAS[0])]
    ax.axhline(reference['R_true'], color=C['truth'], lw=3, label='true bias', zorder=0)
    ax.axhline(reference['R_biased_pre'], color='k', lw=1, label='before the assimilation', zorder=0)
    ax.set(xticks=range(len(FIG6_SETS)), ylabel='RMS error after the assimilation',
           xticklabels=[f'({"abcd"[i]}) $L = {L}$' + ('\naugmented' if augment else '')
                        for i, (L, augment) in enumerate(FIG6_SETS)])
    ax.legend(frameon=False, fontsize='small', loc='center left', bbox_to_anchor=(1., 0.5))
    fig.set_label('fig06_training_sets')


def plot_time_series(case, runs, name, mic=0):
    """Figures 7 and 12: one row for each run of `runs`, ``[(label, filter_ens, truth), ...]``. Columns: the
    observable `mic` at the start and at the end of the assimilation (truth, observations, biased and unbiased
    estimates, with the standard deviation of the ensemble), the bias (the truth minus the model estimate, and
    the estimate of the network), and the parameters over their true values (mean and standard deviation).
    Dashed vertical lines: the first and the last observation. The time is that of the paper."""
    t_shift = CASES[case]['truth'].get('t_min', 0.)
    fig, axs = plt.subplots(len(runs), 4, figsize=(13., 2.3 * len(runs)), layout='constrained', squeeze=False,
                            width_ratios=[1, 1, 1.3, 1.3])
    letters = iter('abcdefghijklmnop')
    for row, (label, filter_ens, truth) in zip(axs, runs):
        s, model = signals(filter_ens, truth), filter_ens.model
        t, t_cr = s['t'] + t_shift, model.t_CR
        t0, t1 = truth.t_obs[0] + t_shift, truth.t_obs[-1] + t_shift
        scale = np.max(abs(s['d'][:, mic]))
        d, q, y = s['d'][:, mic] / scale, np.mean(s['q'][:, mic], axis=-1) / scale, np.mean(s['y'][:, mic], axis=-1) / scale
        std = np.std(s['q'][:, mic], axis=-1) / scale
        for ax, window in zip(row[:2], ((max(t[0], t0 - t_cr), t0 + 1.5 * t_cr), (t1 - t_cr, t1 + t_cr))):
            ax.plot(t, d, color=C['truth'], lw=4, label='truth')
            ax.fill_between(t, q - std, q + std, color=C['biased'], alpha=0.3, lw=0)
            ax.plot(t, q, '--', color=C['biased'], lw=1.2, label='biased')
            ax.plot(t, y, color=C['unbiased'], lw=1, label='unbiased')
            ax.plot(truth.t_obs + t_shift, truth.y_obs[:, mic] / scale, '.', color=C['obs'], ms=4, label='data')
            ax.set(xlim=window, ylim=(-1.6, 1.6), xlabel='$t$ [s]')
        row[0].set_ylabel(f'{label}\n' + (r'$\eta$' if case.startswith('vdp') else r"$p'(x_\mathrm{f})$") + ' / max')
        row[2].plot(t, d - q, color=C['bias'], lw=3, alpha=0.35, label='truth minus model')
        row[2].plot(t, y - q, color=C['bias'], lw=0.8, label='network')
        row[2].set(xlim=(max(t[0], t0 - 0.05 * (t1 - t0)), t[-1]), xlabel='$t$ [s]', ylabel='bias / max', ylim=(-1.6, 1.6))
        for k, name_k in enumerate(filter_ens.est_alpha):
            alpha = model.hist[:, model.Nphi + k] / CASES[case]['truth']['params'][name_k]
            mean, spread = alpha.mean(axis=-1), alpha.std(axis=-1)
            row[3].fill_between(t, mean - spread, mean + spread, color=C_PARAM[name_k], alpha=0.25, lw=0)
            row[3].plot(t, mean, color=C_PARAM[name_k], label=rf'${LABEL[name_k]}/{LABEL[name_k]}^\mathrm{{t}}$')
        row[3].axhline(1., color='k', lw=0.6, ls=':')
        row[3].set(xlim=(max(t[0], t0 - 0.05 * (t1 - t0)), t[-1]), xlabel='$t$ [s]', ylabel='parameters',
                   ylim=PARAMETER_LIMITS.get(case, (0.5, 1.5)))
        for ax in row:
            for edge in (t0, t1):
                ax.axvline(edge, color='k', lw=0.7, ls='--')
            letter(ax, next(letters))
    legend_above(fig, axs[0, 1:])
    fig.set_label(name)


def plot_gamma_sweep(case, tables, name):
    """Figures 8 and 11: one row for each entry of `tables`, ``{label: {gamma: scores}}``. Columns: the RMS errors
    against the regularization factor (circles: last `t_CR` of the assimilation; crosses: first `t_CR` after it;
    lines: the true bias and the error before the assimilation); the parameters over their true values at the end
    of the assimilation (mean and standard deviation); the mean absolute errors of the biased and of the unbiased
    estimates in time, in per cent of the maximum of the truth, coloured by the regularization factor."""
    t_shift = CASES[case]['truth'].get('t_min', 0.)
    true = CASES[case]['truth']['params']
    fig, axs = plt.subplots(len(tables), 4, figsize=(13., 2.5 * len(tables)), layout='constrained', squeeze=False)
    letters = iter('abcdefghijklmnop')
    for row, (label, table) in zip(axs, tables.items()):
        gammas = sorted(table)
        first = table[gammas[0]]
        for kind in ('biased', 'unbiased'):
            for window, marker in (('DA', 'o'), ('post', 'x')):
                row[0].plot(gammas, [table[g][f'R_{kind}_{window}'] for g in gammas], marker, color=C[kind], ms=4,
                            mfc='none', label=f'{kind}, {"end" if window == "DA" else "after"}')
        row[0].axhline(first['R_true'], color=C['truth'], lw=3, zorder=0, label='true bias')
        row[0].axhline(first['R_biased_pre'], color='k', lw=1, zorder=0, label='before')
        row[0].set(xlabel=r'$\gamma$', ylabel=f'{label}\nRMS error', ylim=(0., 1.5 * max(first['R_biased_pre'], 0.5)))
        names = [key[:-4] for key in first if key.endswith('_end') and not key.startswith('std_')]
        for name_k in names:
            row[1].errorbar(gammas, [table[g][f'{name_k}_end'] / true[name_k] for g in gammas],
                            yerr=[table[g][f'std_{name_k}_end'] / true[name_k] for g in gammas], fmt='x', ms=4,
                            capsize=2, lw=0.8, color=C_PARAM[name_k],
                            label=rf'${LABEL[name_k]}/{LABEL[name_k]}^\mathrm{{t}}$')
        row[1].axhline(1., color='k', lw=0.6, ls=':')
        row[1].set(xlabel=r'$\gamma$', ylabel='parameters', ylim=PARAMETER_LIMITS.get(case, (0.4, 1.6)))
        norm = Normalize(min(gammas), max(gammas))
        cmap = plt.get_cmap('viridis')
        for ax, kind in zip(row[2:], ('biased', 'unbiased')):
            for g in gammas:
                ax.plot(np.array(table[g]['mae_t']) + t_shift, 100 * np.array(table[g][f'mae_{kind}']),
                        color=cmap(norm(g)), lw=0.9)
            ax.set(xlabel='$t$ [s]', ylabel=f'{kind} error [%]', yscale='log')
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=row[3], label=r'$\gamma$')
        for ax in row:
            letter(ax, next(letters))
    legend_above(fig, axs[0, :2])
    fig.set_label(name)


def minimum(table, window):
    """The run of smallest sum of the biased and unbiased errors over `window` ('DA' or 'post'), as the paper
    selects the optimum: ``(L, gamma, scores)``."""
    (L, _, gamma), record = min(table.items(), key=lambda item: item[1][f'R_biased_{window}'] + item[1][f'R_unbiased_{window}'])
    return L, gamma, record


def plot_contours(tables):
    """Figure 10: the sum of the biased and unbiased RMS errors after the assimilation over the regularization
    factor and the number of training members, one panel for each bias of `tables`. Star: the minimum."""
    fig, axs = plt.subplots(1, len(tables), figsize=(5.2 * len(tables), 3.6), layout='constrained', squeeze=False)
    norm = LogNorm(0.2, 2.)
    for ax, text, (bias, table) in zip(axs[0], 'abcd', tables.items()):
        Ls, gammas = sorted({key[0] for key in table}), sorted({key[2] for key in table})
        total = np.array([[table[(L, True, g)]['R_biased_post'] + table[(L, True, g)]['R_unbiased_post']
                           if (L, True, g) in table else np.nan for g in gammas] for L in Ls])
        mesh = ax.pcolormesh(gammas, Ls, np.clip(total, norm.vmin, norm.vmax), norm=norm, cmap='RdBu_r',
                             shading='nearest')
        L, gamma, _ = minimum(table, 'post')
        ax.plot(gamma, L, '*', color='w', mec='k', ms=14)
        ax.set(xlabel=r'$\gamma$', ylabel='$L$')
        ax.text(0.98, 0.04, f'{bias} bias', transform=ax.transAxes, ha='right', fontsize='small',
                bbox=dict(fc='w', ec='k', lw=0.6, pad=2))
        letter(ax, text)
    fig.colorbar(mesh, ax=axs[0], label='biased + unbiased RMS error\nafter the assimilation', extend='both')
    fig.set_label('fig10_L_gamma')


def table_2(tables):
    """Table 2 of the paper against these runs, as Markdown: for each bias, the error of the true unbiased signal
    and, at the minimum of the sum of the two errors at the end of the assimilation and after it, the number of
    training members, the regularization factor and the biased and unbiased errors."""
    lines = ['| bias | source | true biased RMS | end: L, gamma | end: biased | end: unbiased | after: L, gamma '
             '| after: biased | after: unbiased |', '|---|---|---|---|---|---|---|---|---|']
    for bias, table in tables.items():
        paper = TABLE_2[bias]
        row = f'| {bias} | paper | {paper["true"]:.4f} |'
        for window in ('DA', 'post'):
            L, gamma, biased, unbiased = paper[window]
            row += f' {L}, {gamma:.2f} | {biased:.4f} | {unbiased:.4f} |'
        lines.append(row)
        row = f'| {bias} | romda | {next(iter(table.values()))["R_true"]:.4f} |'
        for window in ('DA', 'post'):
            L, gamma, record = minimum(table, window)
            row += f' {L}, {gamma:.2f} | {record[f"R_biased_{window}"]:.4f} | {record[f"R_unbiased_{window}"]:.4f} |'
        lines.append(row)
        # the runs of romda at the minima of the paper
        row = f'| {bias} | romda at the minima of the paper | |'
        for window in ('DA', 'post'):
            L, gamma = paper[window][:2]
            record = table.get((L, True, gamma))
            row += (f' {L}, {gamma:.2f} | {record[f"R_biased_{window}"]:.4f} | {record[f"R_unbiased_{window}"]:.4f} |'
                    if record else f' {L}, {gamma:.2f} | not run | not run |')
        lines.append(row)
    return '\n'.join(lines)


def plot_seed_spread(records, name='fig08_seeds'):
    """The Van der Pol study over the seed of the run. One row for each `L`; columns: the RMS errors after the
    assimilation, the parameters at the end of the assimilation over their initial mean (as figure F.20 of the
    paper), and (beta - zeta) / kappa over its true value. One marker for each seed; the line is seed 0."""
    config = CASES['vdp']
    prior = {key: float(np.mean(lims)) for key, lims in config['da']['std_alpha'].items()}
    true = config['truth']['params']
    ratio_true = (true['beta'] - true['zeta']) / true['kappa']
    Ls = sorted({r['L'] for r in records})
    fig, axs = plt.subplots(len(Ls), 3, figsize=(11, 2.9 * len(Ls)), layout='constrained', squeeze=False, num=name)
    for row, L in zip(axs, Ls):
        rows = [r for r in records if r['L'] == L]
        for r in rows:
            g = r['regularization_factor']
            for kind in ('biased', 'unbiased'):
                row[0].plot(g, r[f'R_{kind}_post'], 'x', color=C[kind], ms=3, alpha=0.6)
            for name_k in prior:
                row[1].plot(g, r[f'{name_k}_end'] / prior[name_k], 'x', color=C_PARAM[name_k], ms=3, alpha=0.6)
            row[2].plot(g, (r['beta_end'] - r['zeta_end']) / r['kappa_end'] / ratio_true, 'kx', ms=3, alpha=0.6)
        first = sorted((r for r in rows if r['seed'] == 0), key=lambda r: r['regularization_factor'])
        gammas = [r['regularization_factor'] for r in first]
        for kind in ('biased', 'unbiased'):
            row[0].plot(gammas, [r[f'R_{kind}_post'] for r in first], color=C[kind], lw=1, label=kind)
        for name_k in prior:
            row[1].plot(gammas, [r[f'{name_k}_end'] / prior[name_k] for r in first], color=C_PARAM[name_k], lw=1,
                        label=rf'${LABEL[name_k]}/\bar{{{LABEL[name_k]}}}^0$')
            row[1].axhline(true[name_k] / prior[name_k], color=C_PARAM[name_k], lw=0.8, ls=':')
        row[2].plot(gammas, [(r['beta_end'] - r['zeta_end']) / r['kappa_end'] / ratio_true for r in first], 'k', lw=1)
        row[0].axhline(first[0]['R_true'], color=C['truth'], lw=3, zorder=0, label='true bias')
        row[2].axhline(1., color='k', lw=0.8, ls=':')
        row[0].set(xlabel=r'$\gamma$', ylabel=f'$L = {L}$\nRMS error', ylim=(0., 0.75))
        row[1].set(xlabel=r'$\gamma$', ylabel='parameters', ylim=(0., 2.5))
        row[2].set(xlabel=r'$\gamma$', ylabel=r'$(\beta - \zeta)/\kappa$ over the truth', ylim=(0., 2.))
    for k, ax in enumerate(axs.ravel()):
        letter(ax, chr(97 + k))
    legend_above(fig, axs[0])
    return fig


def plot_ratio_spread(records, name='fig08_ratio'):
    """The combination (beta - zeta) / kappa at the end of the assimilation over its true value, against the
    regularization factor: one panel for each `L`, one marker for each seed; the line is the median over the
    seeds and the band spans the seeds. The dotted line is the truth and the dashed line the initial mean."""
    config = CASES['vdp']
    prior = {key: float(np.mean(lims)) for key, lims in config['da']['std_alpha'].items()}
    true = config['truth']['params']
    ratio_true = (true['beta'] - true['zeta']) / true['kappa']
    ratio_prior = (prior['beta'] - prior['zeta']) / prior['kappa'] / ratio_true
    Ls = sorted({r['L'] for r in records})
    fig, axs = plt.subplots(1, len(Ls), figsize=(3.6 * len(Ls), 3.), sharey=True, layout='constrained', num=name)
    for k, (ax, L) in enumerate(zip(np.atleast_1d(axs), Ls)):
        by_gamma = {}
        for r in records:
            if r['L'] == L:
                by_gamma.setdefault(r['regularization_factor'], []).append(
                    (r['beta_end'] - r['zeta_end']) / r['kappa_end'] / ratio_true)
        gammas = sorted(by_gamma)
        for g in gammas:
            ax.plot([g] * len(by_gamma[g]), by_gamma[g], 'x', color='0.35', ms=3.5, label='seeds' if g == gammas[0] else None)
        ax.fill_between(gammas, [min(by_gamma[g]) for g in gammas], [max(by_gamma[g]) for g in gammas],
                        color=C['biased'], alpha=0.25, lw=0)
        ax.plot(gammas, [np.median(by_gamma[g]) for g in gammas], color=C['biased'], lw=1.5, label='median')
        ax.axhline(1., color='k', lw=0.8, ls=':', label='truth')
        ax.axhline(ratio_prior, color='k', lw=0.8, ls='--', label='initial mean')
        ax.set(xlabel=r'$\gamma$', ylim=(0., 2.))
        ax.text(0.97, 0.05, f'$L = {L}$', transform=ax.transAxes, ha='right', va='bottom',
                bbox=dict(fc='w', ec='k', lw=0.6, pad=2))
        letter(ax, chr(97 + k))
    np.atleast_1d(axs)[0].set(ylabel=r'$(\beta - \zeta)/\kappa$ over the truth')
    legend_above(fig, np.atleast_1d(axs)[:1])
    return fig


# -------------------------------------------------- cases -------------------------------------------------- #

def vdp_seeds(plot_only=False, n_seeds=6):
    """The Van der Pol study of figure 8 over `n_seeds` seeds of the run (not in the paper). The code of the paper
    does not reseed its generator between the runs of a sweep, so each of its regularization factors has its own
    realization of the perturbed observations; here each run is seeded, and the seeds show that variability."""
    from romda.sweep import load_results
    axes = dict(augment=[True], L=list(FIG8_LS), regularization_factor=list(FIG8_GAMMAS))
    context = dict(config=CASES['vdp'])
    path = os.path.join(results_folder, 'vdp_seeds.jsonl')
    records = load_results(path) if plot_only else run_sweep(
        lambda seed=0, **point: experiment('vdp', seed=seed, **point), axes, path, n_seeds=n_seeds, context=context)
    records = [r for r in dedupe(filter_context(records, context), axes) if r.get('status') == 'ok']
    plot_seed_spread(records)
    plot_ratio_spread(records)
    for L in FIG8_LS:
        for gamma in (5., 10., 20.):
            rows = [r for r in records if r['L'] == L and r['regularization_factor'] == gamma]
            line = ', '.join(f"{key} {min(r[f'{key}_end'] for r in rows):.1f}-{max(r[f'{key}_end'] for r in rows):.1f}"
                             for key in ('beta', 'zeta', 'kappa'))
            print(f'L = {L}, gamma = {gamma:g}, {len(rows)} seeds: {line}; rejected up to {max(r["rejected"] for r in rows)}')



def vdp(quick=False, plot_only=False, case='vdp'):
    """Figures 6, 7 and 8 of the paper, or the same study for the variant `case` ('vdp_far_prior')."""
    gammas8 = FIG8_GAMMAS[::10] if quick else FIG8_GAMMAS
    name = f'{case}_quick' if quick else case
    if plot_only:
        table = stored(name, dict(config=CASES[case]))
    else:
        table = sweep(case, dict(L=list(FIG8_LS), regularization_factor=list(gammas8)), name=name)
        table.update(sweep(case, dict(L=[1], augment=[False], regularization_factor=list(FIG6_GAMMAS)), name=name))
    plot_training_sets(table)
    runs = [(rf'$\gamma = {gamma:g}$', *run(case, 50, gamma)) for gamma in FIG6_GAMMAS]
    plot_time_series(case, runs, 'fig07_time_series')
    plot_gamma_sweep(case, {f'$L = {L}$': {g: table[(L, True, g)] for g in gammas8} for L in FIG8_LS},
                     'fig08_gamma')
    print('\nVan der Pol: RMS errors after the assimilation (biased, unbiased), rejected analyses')
    reference = table[(1, True, 0.)]
    print(f'  true bias {reference["R_true"]:.3f}, before the assimilation {reference["R_biased_pre"]:.3f}')
    for (L, augment, gamma), r in sorted(table.items()):
        print(f'  L = {L:3d}{"" if augment else " (not augmented)":16s} gamma = {gamma:4.1f}: '
              f'{r["R_biased_post"]:.3f}  {r["R_unbiased_post"]:.3f}  rejected {r["rejected"]}')
    return table


def rijke(quick=False, plot_only=False, biases=tuple(BIASES)):
    """Figures 10, 11 and 12 and Table 2 of the paper."""
    Ls, gammas = ((10, RIJKE_L_FIG11), (0., 2., 4.)) if quick else (RIJKE_LS, RIJKE_GAMMAS)
    tables = {}
    for bias in biases:
        name = f'rijke_{bias}' + ('_quick' if quick else '')
        if plot_only:
            tables[bias] = stored(name, dict(config=CASES['rijke'], bias=BIASES[bias]))
        else:
            tables[bias] = sweep('rijke', dict(L=list(Ls), regularization_factor=list(gammas)), name=name,
                                 bias=BIASES[bias])
    plot_contours(tables)
    plot_gamma_sweep('rijke', {f'{bias} bias': {key[2]: r for key, r in table.items() if key[0] == RIJKE_L_FIG11}
                               for bias, table in tables.items()}, 'fig11_gamma')
    runs = []
    for bias, table in tables.items():
        L, gamma, _ = minimum(table, 'post')
        runs.append((f'{bias} bias\n$L = {L}$, $\\gamma = {gamma:g}$', *run('rijke', L, gamma, bias=BIASES[bias])))
    plot_time_series('rijke', runs, 'fig12_time_series')
    text = table_2(tables)
    print('\nTable 2: RMS errors at the minima of the sum of the biased and unbiased errors\n' + text)
    os.makedirs(results_folder, exist_ok=True)
    with open(os.path.join(results_folder, 'table2_quick.md' if quick else 'table2.md'), 'w') as f:
        f.write(text + '\n')
    return tables


if __name__ == '__main__':
    parser = main_parser(__doc__)
    parser.add_argument('case', choices=('vdp', 'vdp_far_prior', 'vdp_seeds', 'rijke'),
                        help="vdp_far_prior: the Van der Pol study with a prior 50 %% from the truth (not in the paper)")
    parser.add_argument('--quick', action='store_true', help='a few runs of each sweep, as a smoke test')
    parser.add_argument('--plot', action='store_true', help='draw the figures from the recorded runs')
    parser.add_argument('--bias', choices=tuple(BIASES), default=None, help='rijke: one bias (default: both)')
    args = parser.parse_args()

    if args.case == 'vdp_seeds':
        vdp_seeds(plot_only=args.plot)
    elif args.case.startswith('vdp'):
        vdp(quick=args.quick, plot_only=args.plot, case=args.case)
    else:
        rijke(quick=args.quick, plot_only=args.plot, biases=(args.bias,) if args.bias else tuple(BIASES))
    show_or_save_figs(args.save_figs, prefix=args.case + ('_quick' if args.quick else ''), figs_folder=figs_folder)
