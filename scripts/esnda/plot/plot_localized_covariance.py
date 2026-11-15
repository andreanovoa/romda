"""Figure of the washout-free paper: the forecast correlations that the LETKF localizes, on L96-40p.

The LETKF (half-width of two sites, innovation-based inflation) synchronizes L96-40p from every other site for
2 T_lambda in test window 0, with m = 10 and with m = 1000 members (the reference). At the last analysis, the
forecast ensemble gives the correlation between every row of the state and the predicted data at the observed
sites: the input rows (column I) and the reservoir rows, as the root mean square over the units of each reservoir
(column II). Rows: (a) m = 10 sampled; (b-d) m = 10 tapered by the Gaspari-Cohn weight of Eq. (letkf_taper) with
half-widths of two (the filter), four and eight sites; (e) m = 1000 sampled, when the record holds it.

Reads results/localization/localized_covariance.npz and localized_covariance_positions.npz
(run_parallel_localization.py covariance).

    python plot_localized_covariance.py   -> PAPER/localized_covariance.pdf
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from plot_aux import save, style
from romda.estimators.localization import localization_matrix

import run_parallel_localization as P  # noqa: E402

WIDTH_B = 0.45   # width of the reservoir column (8 reservoirs) relative to the input column (40 sites)
SHOWN_RADII = (2.0, 4.0, 8.0)   # half-widths of the tapered rows: the filter's, 2, and two wider ones
VMIN = 3e-2   # the colours are logarithmic, from VMIN to 1: a tapered correlation below VMIN is white
style(**{'font.size': 8, 'legend.fontsize': 8, 'axes.spines.top': True, 'axes.spines.right': True})


def taper(positions, sensors, radius=P.SHOWN_RADIUS):
    return localization_matrix(positions, sensors, radius, period=float(P.NX))   # distance on the ring


def main():
    z = np.load(P.COV_PATH)
    sensors = z['sensors']
    pos_r = np.load(P.COV_POSITIONS_PATH)['reservoir']
    small, *big = P.COV_MEMBERS
    big = [m for m in big if f'u{m}' in z]   # the reference, when it was run (COV_MEMBERS)
    panels = [(z[f'u{small}'], z[f'r{small}'])] + \
             [(taper(np.arange(P.NX, dtype=float), sensors, h) * z[f'u{small}'], taper(pos_r, sensors, h) * z[f'r{small}'])
              for h in SHOWN_RADII] + \
             [(z[f'u{m}'], z[f'r{m}']) for m in big]
    names = ['sampled m=10'] + [f'tapered m=10, h={h:g}' for h in SHOWN_RADII] + [f'sampled m={m}' for m in big]
    far = taper(np.arange(P.NX, dtype=float), sensors) == 0
    for name, (cu, _) in zip(names, panels):
        print(f'{name}: RMS correlation of the inputs beyond 2h = {np.sqrt(np.mean(cu[far] ** 2)):.3f}')

    fig, axs = plt.subplots(len(panels), 2, figsize=(2.55, 1.04 * len(panels) + 0.42), layout='constrained',
                            sharex='col', sharey=True, width_ratios=(1, WIDTH_B))   # y: the observed site, shared
    fig.get_layout_engine().set(h_pad=0.01, hspace=0)   # the rows close together
    ext_u = (0.5, P.NX + 0.5, 0, P.NX)       # y in sites, from 0 at the bottom: observed site s covers [s - 1, s + 1]
    ext_r = (0.5, P.N_P + 0.5, 0, P.NX)
    for j, (cu, cr) in enumerate(panels):     # rows: (a) to (e); columns: I inputs, II reservoirs
        im = axs[j, 0].imshow(np.abs(cu).T, cmap='Blues', norm=LogNorm(VMIN, 1), extent=ext_u, aspect='auto', origin='lower',
                              interpolation='none')    # magnitude, as the RMS of the reservoirs
        imr = axs[j, 1].imshow(cr.T, cmap='Purples', norm=LogNorm(VMIN, 1), extent=ext_r, aspect='auto', origin='lower', interpolation='none')
        for ax, aspect in zip(axs[j], (1, 1 / WIDTH_B)):            # (a) square, (b) as tall and WIDTH_B as wide
            ax.grid(False)
            ax.set_box_aspect(aspect)
        axs[j, 0].annotate(f'({"abcdef"[j]})', xy=(0, 1), xycoords='axes fraction', xytext=(-22, 2),
                           textcoords='offset points', ha='right', va='bottom', fontsize=8, fontweight='bold')
        axs[j, 0].set_ylabel('observed site, $k$')
    for i, ax in enumerate(axs[0]):                                   # columns: I, II above the first row
        ax.text(1.0, 1.02, ('I', 'II')[i], transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
                fontweight='bold')
    axs[0, 0].set_yticks((0, 10, 20, 30, 40))
    axs[-1, 0].set(xticks=(10, 20, 30, 40), xlabel='input site, $i$')
    axs[-1, 1].set(xticks=(2, 4, 6, 8), xlabel='reservoir, $p$')
    for row, (mappable, label, y0) in enumerate(((im, '|corr.|', 0.0), (imr, 'RMS(corr.)', 0.0))):   # right of column II
        cax = axs[row, 1].inset_axes([1.6, y0, 0.22, 0.84])            # I beside (a), II beside (b); the same size
        cb = fig.colorbar(mappable, cax=cax, ticks=[VMIN, 1e-1, 1])
        cb.ax.set_yticklabels([f'{VMIN:g}', '0.1', '1'])
        cb.ax.minorticks_off()
        cax.text(0.5, 1.04, label, transform=cax.transAxes, ha='center', va='bottom', fontsize=7)   # above the bar
    save(fig, 'localized_covariance.pdf')


if __name__ == '__main__':
    main()
