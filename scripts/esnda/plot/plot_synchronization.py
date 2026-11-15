"""Figure of the washout-free paper: final error against the data a synchronization consumes.

Reads results/synchronization/synchronization_<case>.npz and the extra sensor layouts in results/synchronization/synchronization_<case>_s*.npz
(run_synchronization.py; run_parallel_localization.py sync for the 40-site parallel network) and writes
figs/synchronization.pdf, one column per
system: root-mean-square error over the last Lyapunov time of the data, e_W, against the number of scalar data consumed; medians and interquartile ranges over the windows. The filter
curves are coloured by the fraction of the components observed, on one colour scale for all the systems.
"""

import glob

import os

import matplotlib.pyplot as plt
import numpy as np
from fractions import Fraction

from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from plot_aux import RESULTS, noise_level, panel_letter, reference_lines, save

import run_synchronization as SYNC  # noqa: E402


TITLES = {'lorenz63': 'Lorenz-63', 'lorenz96': 'L96-10s'}   # L96-10p ('lorenz96_parallel_g5_h2_n500_nx10_w20') in the text only
# scalar data of the training washout, T_w network steps times N_u (configs/*.yml N_wash; N_WASH of
# run_parallel_localization.py)
SHADES = {'washout': 'Oranges', 'filter': 'Blues', 'noise': 'Greens'}   # one map per method, darker = more observed


def shade(method, f):
    return plt.get_cmap(SHADES[method])(0.35 + 0.6 * float(f))   # no pale end
TOP = 4.0                            # errors above sit on the top edge
TRAIN_WASH = {'lorenz63': 20 * 3, 'lorenz96': 20 * 10}


def series(z, method, n_sensors):
    sel = (z['method'] == method) & (z['n_sensors'] == n_sensors)
    lengths = sorted(set(z['length_lyap'][sel]))
    out = []
    for L in lengths:
        s = sel & (z['length_lyap'] == L)
        if method != 'washout' and np.median(z['n_data'][s]) < 2 * n_sensors:   # one analysis: e_end is that analysis alone
            continue
        if any(abs(np.median(z['n_data'][s]) - o[0]) <= 0.05 * o[0] for o in out):   # a data count already shown
            continue
        out.append((np.median(z['n_data'][s]), np.nanpercentile(z['feed_error'][s], [25, 50, 75])))
    return out


def load(case):
    """Rows of the base file and of every extra sensor layout of the case."""
    files = [os.path.join(RESULTS, 'synchronization', f'synchronization_{case}.npz')]
    files += sorted(f for f in glob.glob(os.path.join(RESULTS, 'synchronization', f'synchronization_{case}_s*.npz')) if '_cdd' not in f and '_n1' not in f)   # the same lengths for every curve
    zs = [np.load(f) for f in files]
    return {k: np.concatenate([z[k] for z in zs]) for k in zs[0].files}


def diverged(ax, n, med, c):
    """An arrow in the colour of a curve where it first leaves the top of the panel: that case diverged."""
    out = np.flatnonzero(med > TOP)
    if out.size:
        x = n[out[0]]
        ax.annotate('', xy=(x, TOP), xytext=(x, TOP / 2.2), arrowprops=dict(arrowstyle='->', color=c, lw=1.2),
                    annotation_clip=False)


def main():
    cases = [c for c in TITLES if os.path.exists(os.path.join(RESULTS, 'synchronization', f'synchronization_{c}.npz'))]
    fig = plt.figure(figsize=(3.4, 2.0 * len(cases) + 0.6), layout='constrained')   # one column of the paper
    gs = fig.add_gridspec(len(cases) + 1, 1, height_ratios=[0.4] + [1] * len(cases))
    top = fig.add_subplot(gs[0])                        # legend and colour bar
    top.axis('off')
    axs = [fig.add_subplot(gs[1])]
    axs += [fig.add_subplot(gs[i + 1], sharex=axs[0]) for i in range(1, len(cases))]
    fractions = set()                                   # the observed fractions, for the discrete colour bar
    for col, case in enumerate(cases):
        z = load(case)
        full = max(z['n_sensors'])
        SHOWN = {3, 2, 1} if full == 3 else {10, 5, 3}      # three layouts per system: all, about half, about a third
        # the noise: std of the noise / std of each component (Table 1) = its normalized error, because every
        # component carries the same fraction; climatology named in (a) alone
        reference_lines(axs[col], noise_level(SYNC.CASES[case]['cfg']), climatology=col == 0)
        layouts = [('washout', full, shade('washout', 1), 'o', 'washout, full state')]
        layouts += [('filter', ns, shade('filter', ns / full), 's', None) for ns in sorted(set(z['n_sensors']) & SHOWN, reverse=True)]
        for method, ns, c, mk, label in layouts:
            rows = series(z, method, ns)
            if not rows:
                continue
            n = np.array([r[0] for r in rows]) / TRAIN_WASH[case]   # in units of the data of T_w
            q = np.array([r[1] for r in rows])
            axs[col].plot(n, q[:, 1], color=c, marker=mk, ms=3.5, lw=1, label=label)
            if method != 'washout':                    # the washout from r = 0 starts above, it does not diverge
                diverged(axs[col], n, q[:, 1], c)
            ok = q[:, 2] <= TOP; axs[col].errorbar(n[ok], q[ok, 1], yerr=[q[ok, 1] - q[ok, 0], q[ok, 2] - q[ok, 1]], fmt='none', ecolor=c, elinewidth=0.7, capsize=1.5, alpha=0.8)   # interquartile range
        noise = [np.load(f) for f in sorted(glob.glob(os.path.join(RESULTS, 'synchronization', f'synchronization_{case}_s*_cddnoise.npz')))]
        noise = {k: np.concatenate([z[k] for z in noise]) for k in noise[0].files} if noise else None   # TAG=_cddnoise
        fractions |= {Fraction(int(ns), int(full)) for ns in set(z['n_sensors']) & SHOWN}
        for ns in sorted(set(noise['n_sensors']) & SHOWN, reverse=True) if noise else ():   # C_dd at the noise level of the data
            rows = series(noise, 'filter', ns)
            n = np.array([r[0] for r in rows]) / TRAIN_WASH[case]   # in units of the data of T_w
            q = np.array([r[1] for r in rows])
            axs[col].plot(n, q[:, 1], color=shade('noise', ns / full), marker='^', mfc='white', ms=3.5, lw=1, ls='--',
                          label=r'filter, $\mathbf{C}_{dd}$ = noise' if ns == full else None)
            ok = q[:, 2] <= TOP; axs[col].errorbar(n[ok], q[ok, 1], yerr=[q[ok, 1] - q[ok, 0], q[ok, 2] - q[ok, 1]], fmt='none', ecolor=shade('noise', ns / full), elinewidth=0.7, capsize=1.5, alpha=0.8)   # interquartile range
            diverged(axs[col], n, q[:, 1], shade('noise', ns / full))
        axs[col].axvline(1.0, color='0.3', ls=':', lw=0.9)
        if col == 0:
            axs[col].annotate('$T_w$', xy=(1.0, 0.74), xytext=(0.4, 0.74), xycoords=('data', 'axes fraction'), ha='right', va='center',
                              color='0.35', fontsize=9, arrowprops=dict(arrowstyle='->', color='0.35', lw=0.6))
        axs[col].set(xscale='log', yscale='log', ylabel=r'$e_\mathrm{end}$')
        axs[col].set_ylim(top=1.6 if case == 'lorenz63' else TOP)   # (a): no curve above climatology but the washout
        panel_letter(axs[col], col, fontsize=12,   # systems named in the caption; flush left
                     xycoords=blended_transform_factory(fig.transFigure, axs[col].transAxes))
    h = [Line2D([], [], color=shade('washout', 1), marker='o', ms=3.5, lw=1),          # short labels; the colours
         Line2D([], [], color=shade('filter', 1), marker='s', ms=3.5, lw=1),           # and the fractions are in
         Line2D([], [], color=shade('noise', 1), marker='^', mfc='white', ms=3.5, lw=1, ls='--')]   # the key
    l = ['washout', r'filter, $\mathbf{C}_{dd}$ = 4–10$\times$noise', r'filter, $\mathbf{C}_{dd}$ = noise']
    axs[-1].set_xlabel(r'scalar data consumed / those of $T_w$')
    for ax in axs[:-1]:
        ax.tick_params(labelbottom=False)
    top.legend(h, l, loc='center left', frameon=False, borderaxespad=0, handlelength=1.8, handletextpad=0.5,
               labelspacing=0.3)
    cax = top.inset_axes([0.72, 0.05, 0.06, 0.9])        # right of the legend, clear of it: one strip per method, one row per
    fr = sorted(fractions)[::-1]                        # observed fraction (discrete)
    cax.imshow(np.array([[shade(m, f) for m in ('filter', 'noise')] for f in fr]), aspect='auto')
    cax.set(xticks=[], yticks=range(len(fr)))
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position('right')
    cax.set_yticklabels([str(f) for f in fr], fontsize=8)
    cax.tick_params(length=0)
    cax.set_ylabel('fraction\nobserved')
    save(fig, 'synchronization.pdf')
    print(cases)


if __name__ == '__main__':
    main()
