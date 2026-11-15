"""Figure of the washout-free paper: the global and the localized filter on the parallel network, against N_e.

Reads results/localization/localization_parallel_lorenz96_w20.npz (L96-40p, N_WASH=20: test windows; run_parallel_localization.py run) and writes
figs/localization.pdf, one column of the paper: root-mean-square error
over the last Lyapunov time of the data, the final error (a), above the valid time (b), against the number of members, with the innovation-based inflation of
the paper (the the *_innov files of run_parallel_localization.py run); the
curves are medians and interquartile ranges over the test windows. Grey band, named in (b): the washout of
the full state over the same duration as the filter, median and interquartile range.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from plot_aux import BLUE, RESULTS, GLOBAL, NOISE_CDD, NOISE_LEVEL, TOP, load, reference_lines, save, selection, VERMILION

# ponytail: the half-width of 4 sites and the single reservoir with the LETKF (it diverges in every window) are in the
# text only; the figure keeps the three curves that the argument needs
STYLE = {np.nan: (VERMILION, 'o', 'parallel, EnSRKF'), 2.0: (BLUE, 's', 'parallel, LETKF')}
GLOBAL_STYLE = ('0.15', 'D', 'single, EnSRKF')
NOISE1_STYLE = ('#CC79A7', 'P', r'L96-40p, LETKF, $\mathbf{C}_{dd}$ = noise')   # (CDD_STD=0.02 TAG=_innov_cdd1x run)
PANELS = (('feed_error', 'wash_feed', r'$e_\mathrm{end}$', 'log'),)   # filter key, washout key, label, scale (the valid time: text only)


def table_legend(ax, sensors, dashes):
    """The legend as a table: the filter in the columns (global, local), the network in the rows (single ESN,
    i.e. L96-40s, and parallel ESN, L96-40p); the single ESN with the local filter is not drawn. The line styles of
    the sensor layouts on the right."""
    ax.set(xlim=(0, 1), ylim=(0, 3))
    ax.axis('off')
    cols, rows = (0.37, 0.63), (1.5, 0.5)                             # cell centres
    for x, title in zip(cols, ('global filter', 'local filter')):
        ax.text(x, 2.5, title, ha='center', va='center', fontsize=9)
    for y, name in zip(rows, ('single ESN', 'parallel ESN')):
        ax.text(0.0, y, name, ha='left', va='center', fontsize=9)
    cells = {(0, 0): GLOBAL_STYLE, (1, 0): STYLE[np.nan], (1, 1): STYLE[2.0]}
    for (i, j), (c, mk, _) in cells.items():
        ax.plot([cols[j] - 0.05, cols[j] + 0.05], [rows[i]] * 2, color=c, lw=1)
        ax.plot(cols[j], rows[i], color=c, marker=mk, ms=3.5)
    for y, s in zip(rows, sensors):                                   # N_q: one row each, right of the table
        ax.plot([0.8, 0.87], [y] * 2, color='k', lw=1, ls=dashes[s])
        ax.text(0.885, y, f'$N_q = {s:g}$', ha='left', va='center', fontsize=9)


def main():
    z = load(NOISE_CDD)
    sensors = [s for s in sorted(set(z['n_sensors']))[::-1] if s != 10]   # N_q = 10 (_more10) does not synchronize: text only
    dashes = dict(zip(sensors, ('-', '--', ':')))                   # every site solid, fewer sites dashed
    fig, axs = plt.subplots(len(PANELS) + 1, 1, figsize=(3.4, 2.9), layout='constrained', squeeze=False,
                            height_ratios=(0.42, *[1] * len(PANELS)))
    legend, axs = axs[0, 0], axs[1:, 0]                              # one column of the paper; the legend on top
    curves = [(z, radius, *style) for radius, style in STYLE.items()]
    if os.path.exists(os.path.join(RESULTS, 'localization', GLOBAL + '_innov.npz')):
        curves.append((load(GLOBAL), np.nan, *GLOBAL_STYLE))
    if os.environ.get('CDD_CURVES'):                    # CDD_CURVES=1: the LETKF with C_dd equal to the noise (text only)
        curves.append((load(NOISE_CDD, '_innov_cdd1x'), 2.0, *NOISE1_STYLE))
    members = sorted(set().union(*(set(c[0]['members']) for c in curves)))
    for ax in axs:                                   # the reference lines of Fig. 3
        reference_lines(ax, NOISE_LEVEL)
    for s in sensors:
        for zc, radius, c, mk, label in curves:
            pts = []
            for m in sorted(set(zc['members'])):
                rows = selection(zc, s, m, radius)
                print(f'{s:2.0f} sensors, {label:32s} N_e {m:3.0f}: valid '
                      f'{np.median(zc["valid_lyap"][rows]):.2f} (IQR {np.percentile(zc["valid_lyap"][rows], 25):.2f}-'
                      f'{np.percentile(zc["valid_lyap"][rows], 75):.2f}), first error {np.median(zc["first_error"][rows]):.3g}')
                pts.append((m, [np.percentile(zc[k][rows], [25, 50, 75]) for k, *_ in PANELS]))
            m_ = np.array([p[0] for p in pts])
            for ax, (*_, scale) in zip(axs, PANELS):
                q = np.array([p[1][axs.tolist().index(ax)] for p in pts])
                if scale == 'log':
                    q = np.minimum(q, TOP)                                 # errors above sit on the top edge
                ax.plot(m_, q[:, 1], color=c, marker=mk, ms=3.5, lw=1, ls=dashes[s], clip_on=False,
                        label=label if s == sensors[0] else None)
                ax.fill_between(m_, q[:, 0], q[:, 2], color=c, alpha=0.12, lw=0)
    for ax, (_, _, label, scale) in zip(axs, PANELS):
        ax.set(xscale='log', yscale=scale, ylabel=label, xlabel=r'members, $m$' if ax is axs[-1] else None)
        if scale == 'log':
            ax.set_ylim(1e-2, TOP)                                        # the range of Fig. 3: shows the synchronized network
        ax.set_xticks(members, [f'{m:g}' for m in members])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    table_legend(legend, sensors, dashes)
    save(fig, 'localization.pdf')


if __name__ == '__main__':
    main()
