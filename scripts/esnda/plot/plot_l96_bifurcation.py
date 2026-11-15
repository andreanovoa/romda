"""Figure of the washout-free paper (`fig:bifurcation`): the bifurcation diagrams of the ten-site Lorenz-96 system (top),
of the parametric L96-10s (middle, the single ESN of the paper) and of the parallel parametric ESN (bottom) in closed
loop, with the training and the test forcings of run_parametric.py.

Reads results/parameter/lorenz96_bifurcation_Nx10.npz (run_parametric.py truth: local extrema of x_k - F against F,
four observables; the first is drawn) and, for each ESN, the local extrema of the closed loops of run_parametric.py
bifurcation at the forcings of the true diagram, u_1 - F over the last 35 of 400 time units, from washouts on random
windows of the validation data (max_random, min_random; the first start per forcing whose range stays within 100 is
drawn): results/readout_parameter/bifurcation_single1000_readout_peaks.npz (run_parametric.py bifurcation) and
results/readout_parameter/bifurcation_parallel_readout_peaks.npz (run_parametric.py bifurcation readout parallel;
without this file, the figure has two rows). Writes $PAPER/l96_bifurcation.pdf, one column of the paper.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from plot_aux import BLUE, RESULTS, TOL_ORANGE, save, style

from run_parametric import F_TRUE, VALUES  # noqa: E402

style(**{'legend.fontsize': 9})


def main():
    d = np.load(os.path.join(RESULTS, 'parameter', 'lorenz96_bifurcation_Nx10.npz'), allow_pickle=True)
    par_file = os.path.join(RESULTS, 'readout_parameter', 'bifurcation_parallel_readout_peaks.npz')   # the parallel parametric ESN
    rows = 3 if os.path.exists(par_file) else 2
    fig, axs = plt.subplots(rows, 1, figsize=(3.4, 0.6 + 1.25 * rows), layout='constrained', sharex=True, sharey=True)
    ax, net = axs[:2]
    for key, color, label in (('peaks_max', '0.15', 'maxima'), ('peaks_min', BLUE, 'minima')):
        for F, peaks in zip(d['F_vals'], d[key][0]):
            ax.plot(np.full(len(peaks), F), peaks, '.', color=color, ms=1.0, alpha=0.5, rasterized=True)
        ax.plot([], [], '.', color=color, ms=5, label=label)
    single = os.path.join(RESULTS, 'readout_parameter', 'bifurcation_single1000_readout_peaks.npz')   # the single ESN
    panels = [(net, np.load(single, allow_pickle=True))]             # washouts on random validation windows
    if rows == 3:
        panels.append((axs[2], np.load(par_file, allow_pickle=True)))
    for panel, q in panels:
        Fn = q['F']
        # runs that leave the attractor are not drawn; a range cut of the drawing, the text counts 3 max|u| (run_parametric)
        bounded = q['span_random'] <= 100
        first = [np.flatnonzero(bounded & (Fn == F))[:1] for F in np.unique(Fn)]   # one start per forcing, as (a)
        for key, color in (('max_random', '0.15'), ('min_random', BLUE)):
            peaks = q[key]                                            # a fixed point: its value
            for j in np.concatenate(first):
                p = peaks[j]
                panel.plot(np.full(len(p), Fn[j]), p, '.', color=color, ms=1.0, alpha=0.5, rasterized=True)
    for a in axs:
        for F in F_TRUE:
            a.axvline(F, color=TOL_ORANGE, lw=0.8, ls=(0, (3, 5)), zorder=0.5)   # wide gaps: the training line shows
        for F in VALUES:
            a.axvline(F, color='0.55', lw=0.6, zorder=0.4)
    names = ('True', 'ESN') if rows == 2 else ('True', 'Single ESN', 'Parallel ESN')
    for a, name in zip(axs, names):                                  # in the empty lower left corner
        a.text(0.03, 0.06, name, transform=a.transAxes, ha='left', va='bottom', fontsize=10)
    ax.plot(F_TRUE, np.ones(len(F_TRUE)), 'v', mfc='none', mec=TOL_ORANGE, mew=0.6, ms=3.5, transform=ax.get_xaxis_transform(),
            clip_on=False, zorder=3)                                  # over the training markers
    ax.plot([], [], color=TOL_ORANGE, lw=0.8, ls=(0, (3, 5)), marker='v', mfc='none', mec=TOL_ORANGE, mew=0.6, ms=3.5,
            label='test forcings')
    ax.plot(VALUES, np.ones(len(VALUES)), 'v', color='0.35', mew=0.6, ms=3.5, transform=ax.get_xaxis_transform(),
            clip_on=False)
    ax.plot([], [], color='0.55', lw=0.6, marker='v', mfc='0.35', mec='0.35', mew=0.6, ms=3.5, label='training forcings')
    ax.set(xlim=(0, 10), ylim=(-20, None))                          # shared: crops both panels
    axs[-1].set(xlabel='$F$')
    fig.supylabel(r'local extrema of $u_1 - F$', fontsize=10)      # one label for both panels
    h, l = ax.get_legend_handles_labels()
    order = [l.index(k) for k in ('maxima', 'training forcings', 'minima', 'test forcings')]
    fig.legend([h[i] for i in order], [l[i] for i in order], frameon=False, loc='outside upper center', ncol=2,
               handlelength=1.4, columnspacing=1.0)                 # rows: forcings, extrema
    save(fig, 'l96_bifurcation.pdf', dpi=300)


if __name__ == '__main__':
    main()
