"""Figure of the washout-free paper: one LETKF deployment of the parallel network on the 40-site Lorenz-96 ring,
with the uncertainty that the ensemble covariance returns.

The test window closest to the median valid time of the ten test windows, N_e = 10, Gaspari-Cohn half-width 2 sites,
the innovation-based inflation of the paper (INNOVATION of run_parallel_localization.py); every site, every other site,
every other site half as often and every fourth site half as often observed (columns), for 8 T_lambda.
Rows: true state, ensemble mean and ensemble standard deviation (square root of the diagonal of the ensemble
covariance) over the sites; sites 1 and 2 in time with the members, the mean and its 2-sigma band, the true state
and the observations; error of the ensemble mean and ensemble spread. The orange dashed line marks the
synchronization time t_s, the first time the normalized ensemble spread falls below SPREAD_SYNC, which needs no true
state; the black dashed line marks t_e, the first time the normalized error of the ensemble mean falls below the same
level, which needs the true state.

Reads the scores of every test window, results/localization/parallel_forecast_windows_cdd0.2.npz, and the deployments
drawn, results/localization/parallel_forecast_shown.pkl (run_parallel_localization.py forecast), and writes
PAPER/parallel_forecast.pdf.

    python plot_parallel_forecast.py
"""

import pickle

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from plot_aux import C_MEAN, C_OBS, C_RES, C_SPREAD, C_TRUE, SPREAD_SYNC, panel_letter, save
from romda.metrics import VALID_THRESHOLD  # noqa: E402

import run_parallel_localization as P  # noqa: E402

SITES = (1,)                                  # site 2 is not observed when every other site is
LAYOUT_X = 0.93                               # right end of the panels and colour bars; the legends overhang it
# SPREAD_SYNC (plot_aux), the level drawn: 0.3 under the former amplitude norm, times its ratio to the variance
# norm (1.19); 0.3 is reached in 5 and 8 of the 10 windows of the two every-other-site layouts, whose mean does
# synchronize
TITLES = ('every site', 'every other site', 'every other site,\nhalf as often', 'every fourth site,\nhalf as often')


def main():
    cases = [(*c, title) for c, title in zip(P.FORECAST_CASES, TITLES)]
    sc = np.load(P.window_scores_path())['scores']
    window = P.shown_window(sc)
    names = ('std unobserved/observed', 'spread', 'error', 'coverage', 'smallest spread')
    for (n, nt_obs, a, title), v, t_s, f, u, t_e in zip(cases, sc[:, :, 0], sc[:, :, 1:4], sc[:, :, 4], sc[:, :, 5:12],
                                                        sc[:, :, 12:15]):
        print(f'{title} ({a:g} T_lambda): valid median {np.median(v):.2f} (IQR {np.percentile(v, 25):.2f}-'
              f'{np.percentile(v, 75):.2f}); '
              + ', '.join(f'{name} < {lev}: median {np.nanmedian(b):.2f} ({np.nanmin(b):.2f}-{np.nanmax(b):.2f}) '
                          f'T_lambda, reached in {np.isfinite(b).sum()}/10'
                          for name, x in (('t_s: spread', t_s), ('t_e: error', t_e))
                          for lev, b in zip(P.SPREAD_LEVELS, x.T)))
        print(f'   last T_lambda of the data: spread {np.median(u[:, 5]):.2f} ({u[:, 5].min():.2f}-{u[:, 5].max():.2f}), '
              f'error e_W {np.median(u[:, 6]):.2f} ({u[:, 6].min():.2f}-{u[:, 6].max():.2f}), '
              f'failed (e_W > {VALID_THRESHOLD}) in {(u[:, 6] > VALID_THRESHOLD).sum()}/10')
        print(f'   mean inflation factor sqrt(lambda): median {np.median(f):.3f} ({f.min():.3f}-{f.max():.3f})')
        print('   per window: valid', np.round(v, 2), f't_s (spread < {SPREAD_SYNC})', np.round(t_s[:, -1], 2),
              f't_e (error < {SPREAD_SYNC})', np.round(t_e[:, -1], 2))
        print(f'   [2, {a:g}] T_lambda, medians:', ', '.join(f'{nm} {np.nanmedian(x):.2f} ({np.nanmin(x):.2f}-'
                                                         f'{np.nanmax(x):.2f})' for nm, x in zip(names, u.T)))
    print('window shown', window, 'valid', sc[:, window, 0])
    fig, axs = plt.subplots(6, len(cases), figsize=(7.4, 6.8), sharex='col', sharey='row', layout='constrained',
                            height_ratios=[0.825, 0.825, 0.825, 1, 1, 0.9], width_ratios=[c[2] + P.FORECAST_LYAP for c in cases])
    fig.get_layout_engine().set(wspace=0.01, hspace=0.01, rect=(0, 0, LAYOUT_X, 1))
    shown = pickle.load(open(P.SHOWN_PATH, 'rb'))   # the deployments drawn
    r_lim = []
    for col, (n_sensors, nt_obs, assimilation_lyap, title) in enumerate(cases):
        r = shown[P.shown_key((n_sensors, nt_obs, assimilation_lyap), window)]
        err, spread, valid, t_s, t_e = P.deployment_scores(r)          # normalized error and spread of the mean
        t_s, t_e = (x[P.SPREAD_LEVELS.index(SPREAD_SYNC)] for x in (t_s, t_e))
        t, U, truth = r['t'], r['U'], r['truth']
        mean, std = U.mean(-1), U.std(-1, ddof=1)
        vmax = np.abs(truth).max()
        sat = 0.6 * vmax                                                # saturated colours; the extremes clip
        sites = np.arange(1, P.NX + 1)
        for row, (field, label, cmap, lim) in enumerate(((truth, r'$u^\dagger$', 'PRGn', (-sat, sat)),
                                                         (mean, r'$\bar{u}$', 'PRGn', (-sat, sat)),
                                                         (std, r'$\sigma$', 'magma_r', (0, 0.5 * vmax)))):
            ax = axs[row, col]
            im = ax.pcolormesh(t, sites, field.T, cmap=cmap, vmin=lim[0], vmax=lim[1], shading='nearest')
            for t_x in (0.0, r['t_last']):                              # the assimilation window
                ax.axvline(t_x, color='k', lw=0.8, ls=':')
            for t_x, c in ((t_e, 'k'), (t_s, C_SPREAD)):                # arrows under the map: lines hide in it
                if np.isfinite(t_x):
                    ax.annotate('', xy=(t_x, 0.0), xycoords=('data', 'axes fraction'), xytext=(0, -9),
                                textcoords='offset points', annotation_clip=False,
                                arrowprops=dict(arrowstyle='-|>', color=c, lw=1.2, mutation_scale=7))
            ax.set(ylabel='site' if col == 0 else None, yticks=[1, 20, 40])
            if row == 0:
                ax.annotate(f'$N_q{{=}}{n_sensors}$, $\\Delta t_\\mathrm{{obs}}{{=}}{nt_obs / P.STEPS_LYAP:.2f}\\,T_\\lambda$',
                            (0, 1), xycoords='axes fraction', xytext=(21, 7), textcoords='offset points',
                            fontsize=8, va='baseline').set_in_layout(False)   # just right of the bold letter
                panel_letter(ax, col, fontsize=11.25, offset=(0, 1), points=(0, 7), va='baseline')   # bold, top left
                for k in r['sensors']:                              # where: the observed sites
                    ax.plot(t[0] - 0.006 * t[-1], k + 1, '>', color=C_OBS, ms=2.5, mec='k', mew=0.3, clip_on=False)
                for to in r['t_obs']:                               # when: the analysis times
                    ax.plot(to, P.NX + 2.2, 'v', color=C_OBS, ms=2.5, mec='k', mew=0.3, clip_on=False)
            if col == len(cases) - 1:
                cb = fig.colorbar(im, ax=axs[row, :].ravel().tolist(), shrink=0.9, pad=0.01, extend='both' if row < 2 else 'neither')
                cb.set_label(label, rotation=0, labelpad=8, va='center')   # upright: a symbol
                cb.solids.set_rasterized(False)                  # vector, as the rest of the figure
        ax = axs[4, col]                                                # V: one unit of reservoir 1 (sites 1-5)
        Rm, Rs = r['R'].mean(-1), r['R'].std(-1, ddof=1)
        ax.plot(t, r['R'], color=C_RES, lw=0.3, alpha=0.3)
        ax.fill_between(t, Rm - 2 * Rs, Rm + 2 * Rs, color=C_RES, alpha=0.25, lw=0)
        ax.plot(t, Rm, color=C_RES, lw=1.2, label=r'$\bar{r}$')
        r_lim.append(np.abs(Rm[t <= r['t_last']]).max())             # the mean during the data sets the axis
        for t_x in (0.0, r['t_last']):
            ax.axvline(t_x, color='k', lw=0.8, ls=':')
        for t_x, c in ((t_e, 'k'), (t_s, C_SPREAD)):
            if np.isfinite(t_x):
                ax.axvline(t_x, color=c, lw=0.8, ls='--')
        ax.set(ylabel=f'$r_{{1,{r["unit"] + 1}}}$' if col == 0 else None)
        for j, k in enumerate(SITES):                                   # IV: site 2
            ax = axs[3 + j, col]
            ax.plot(t, U[:, k, :], color=C_MEAN, lw=0.3, alpha=0.3)
            ax.fill_between(t, mean[:, k] - 2 * std[:, k], mean[:, k] + 2 * std[:, k], color=C_MEAN, alpha=0.25,
                            lw=0, label=r'$\pm 2\sigma$')
            ax.plot(t, mean[:, k], color=C_MEAN, lw=1.2, label=r'$\bar{u}$')
            ax.plot(t, truth[:, k], color=C_TRUE, lw=1.0, label=r'$u^\dagger$')
            if k in r['sensors']:                                       # the data at the analysis times, on top
                ax.plot(r['t_obs'], r['d'][:, list(r['sensors']).index(k)], 'o', color=C_OBS, ms=3.5, mec='k',
                        mew=0.4, zorder=5, label='$d$')
            for t_x in (0.0, r['t_last']):                              # the assimilation window
                ax.axvline(t_x, color='k', lw=0.8, ls=':')
            if np.isfinite(t_e):
                ax.axvline(t_e, color='k', lw=0.8, ls='--')
            if np.isfinite(t_s):
                ax.axvline(t_s, color=C_SPREAD, lw=0.8, ls='--')
            ax.set(ylabel=f'$u_{{{k + 1}}}$' if col == 0 else None, ylim=(-1.1 * vmax, 1.1 * vmax))
        ax = axs[5, col]
        ax.semilogy(t, err, color='k', lw=0.9, label='$e$')
        ax.semilogy(t, spread, color=C_SPREAD, lw=1.4, ls='--', label='$s$')
        ax.axvline(0.0, color='k', lw=0.8, ls=':')
        ax.axvline(r['t_last'], color='k', lw=0.8, ls=':')   # the assimilation window: named in the caption
        ax.axhline(SPREAD_SYNC, color='k', lw=0.6, ls=':')
        if np.isfinite(t_e):
            ax.axvline(t_e, color='k', lw=0.8, ls='--', label=r'$\tau_\mathrm{e}$')
        if np.isfinite(t_s):
            ax.axvline(t_s, color=C_SPREAD, lw=0.8, ls='--', label=r'$\tau_\mathrm{s}$')
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set(ylabel=r'$e$, $s$' if col == 0 else None, xlabel=r'$t / T_\lambda$')
        sync = (t >= 1.0) & (t <= r['t_last'])                         # after the synchronization transient
        ratio = np.sqrt(np.mean(spread[sync] ** 2) / np.mean(err[sync] ** 2)) * np.sqrt((P.SHOWN_M + 1) / (P.SHOWN_M - 1))
        print(f'{title}: error {np.sqrt(np.mean(err[sync] ** 2)):.3f} and spread-skill {ratio:.2f} '
              f'over [1, {r["t_last"]:.0f}] T_lambda, forecast valid {valid:.2f} T_lambda')
    axs[5, 0].set_ylim(bottom=min(0.1, axs[5, 0].get_ylim()[0]))       # a labelled decade on the log axis
    for ax in axs[5]:                                                  # a diverged forecast leaves the panel
        ax.set_ylim(top=2.0)
    for ax in axs[4]:                                                  # the unit: the range of its mean in the
        ax.set_ylim(-2.2 * min(r_lim), 2.2 * min(r_lim))                # layout that synchronizes best
    for row, numeral in enumerate(('I', 'II', 'III', 'IV', 'V', 'VI')):    # the rows, named in the caption
        axs[row, 0].text(-0.3, 1.0, numeral, transform=axs[row, 0].transAxes, ha='right', va='top', fontweight='bold')
    # two legends right of the last column, under the colour bars of I-III, each beside the rows that it explains:
    # IV-V (the sites; centred on the boundary between them) and VI (error and spread). 10.3 pt: 9 pt, the caption,
    # once scaled to the text width (7.4 in -> 468 pt)
    # (outside the constrained layout, which would shrink every column to fit them)
    kw = dict(loc='center left', frameon=False, fontsize=10.3, handlelength=1.4, handletextpad=0.4, borderaxespad=0)
    fig.canvas.draw()                                                  # the final axes positions
    for row, y in ((3, 0.0), (5, 0.5)):
        x, y = fig.transFigure.inverted().transform(axs[row, -1].transAxes.transform((1.0, y)))
        h, l = axs[row, 0].get_legend_handles_labels()
        if row == 3:                                                   # IV-V: the site's entries and the reference
            (h3, l3), (h4, l4) = axs[3, 0].get_legend_handles_labels(), axs[4, 0].get_legend_handles_labels()
            h, l = h3 + h4, l3 + l4
        h = [(hi, matplotlib.lines.Line2D([], [], color=hi.get_color(), ls='', marker='^', ms=4.5))   # + the arrow under I-III
             if li in (r'$\tau_\mathrm{e}$', r'$\tau_\mathrm{s}$') else hi for hi, li in zip(h, l)]
        leg = fig.legend(h, l, bbox_to_anchor=(x + 0.012, y), **kw)
        leg.set_in_layout(False)
    save(fig, 'parallel_forecast.pdf', dpi=200, pad_inches=0.02)


if __name__ == '__main__':
    main()
