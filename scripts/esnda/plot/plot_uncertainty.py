"""Figure of the washout-free paper (uncertainty quantification): the ensemble spread against the error of the ensemble
mean for four levels of the assumed observation error, sigma_obs, per test case, with the band of an ensemble that
is calibrated within a factor of two. Each point is one run that does not fail: circles with every component observed,
triangles with partial observations; the label of a colour is sigma_obs in units of the noise in the data.

Reads results/uncertainty/uq_levels_<case>_cdd<s>[_q1|_q5].npz (run_uq_levels.py l63, 10s and 10p) and
results/localization/parallel_forecast_windows_cdd<s>.npz (CDD_STD=<s> python run_parallel_localization.py forecast;
the columns SCORE_COLUMNS).

    python plot_uncertainty.py   -> $PAPER/uncertainty.pdf
"""
import functools
import os

import matplotlib.pyplot as plt
import numpy as np
from plot_aux import RESULTS, noise_level, panel_letter, save, style
from romda.metrics import VALID_THRESHOLD  # noqa: E402

from run_parallel_localization import SCORE_COLUMNS, window_scores_path  # noqa: E402

style(**{'legend.fontsize': 8.5})


def levels(case, sigma, partial=False):
    """(error, spread) of the runs of run_uq_levels.py for one sigma_obs; partial: y alone (Lorenz-63), every other
    site (Lorenz-96)."""
    q = ('_q1' if case == 'l63' else '_q5') if partial else ''
    z = np.load(os.path.join(RESULTS, 'uncertainty', f'uq_levels_{case}_cdd{sigma:g}{q}.npz'))
    return np.column_stack([z['error'], z['spread']])


def lorenz96_40(sigma, partial=False):
    """L96-40p with the LETKF (run_parallel_localization.py forecast: ten windows, 8 T_lambda of data, 48 analyses): every site,
    or every other site."""
    sc = np.load(window_scores_path(sigma))['scores']
    col = SCORE_COLUMNS.index
    return sc[int(partial)][:, [col('error'), col('spread')]]


# case: (name, colour map (darker: larger sigma_obs), loader, noise, sigma_obs levels)
CASES = (('L63', 'Blues', functools.partial(levels, 'l63'), noise_level('lorenz63'), (0.05, 0.1, 0.25, 0.5)),
         ('L96-10s', 'Greens', functools.partial(levels, '10s'), noise_level('lorenz96_small'), (0.02, 0.04, 0.1, 0.2)),
         ('L96-10p', 'Purples', functools.partial(levels, '10p'), noise_level('lorenz96_parallel'), (0.02, 0.04, 0.1, 0.2)),
         ('L96-40p', 'Oranges', lorenz96_40, noise_level('lorenz96_parallel'), (0.02, 0.04, 0.1, 0.2)))


def main():
    fig, axs = plt.subplots(2, 2, figsize=(3.4, 3.5), layout='constrained', sharex=True, sharey=True)
    lim = np.array([6e-3, 1.5])
    for i, (ax, (label, cmap, load, noise, sigmas)) in enumerate(zip(axs.ravel(), CASES)):
        ax.fill_between(lim, 0.5 * lim, 2 * lim, color='0.94', lw=0, zorder=0)   # calibrated within a factor of two
        ax.plot(lim, lim, color='0.25', lw=0.7, zorder=1)
        shades = plt.get_cmap(cmap)(np.linspace(0.4, 0.95, len(sigmas)))        # light: sigma_obs at the noise
        for partial, marker in ((False, 'o'), (True, '^')):
            out = []
            for sigma, c in zip(sigmas, shades):
                v = load(sigma, partial)
                v = v[np.isfinite(v).all(1) & (v[:, 0] <= VALID_THRESHOLD)]     # the runs that do not fail
                e, sp = np.median(v, axis=0)
                out.append((sigma, f'e {e:.3f}', f's {sp:.3f}', f'ratio {sp / e:.2f}', len(v)))
                ax.plot(v[:, 0], v[:, 1], ls='none', marker=marker, ms=3.6 if partial else 3.2, mfc=c, alpha=0.7,
                        mec='k' if partial else 'none', mew=0.35, zorder=3 if partial else 2,
                        rasterized=True)                                        # every run; triangles: outlined
                if not partial:
                    ax.plot([], [], ls='none', marker='o', ms=5, mfc=c, mec='none', label=f'{sigma / noise:g}$\\times$')
            print(label, 'partial' if partial else 'full', out)
        ax.set(xscale='log', yscale='log', xlim=lim, ylim=lim)
        ax.set_box_aspect(1)
        panel_letter(ax, i)   # bold, top left
        ax.text(0.96, 0.04, label, transform=ax.transAxes, fontsize=8, ha='right', va='bottom')   # the case, bottom right
        ax.legend(frameon=False, loc='upper left', fontsize=7, handletextpad=0.0, borderaxespad=0.1, labelspacing=0.15,
                  ncol=2, columnspacing=0.4)
    fig.supxlabel('error, $e$', fontsize=10)
    fig.supylabel('spread, $s$', fontsize=10)
    save(fig, 'uncertainty.pdf', dpi=300)


if __name__ == '__main__':
    main()
