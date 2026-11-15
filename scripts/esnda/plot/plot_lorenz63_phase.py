"""Figure 2 of the washout-free paper: the three deployments of the Lorenz-63 network from y alone (light blue) and
from z alone (dark blue), one row per deployment (MODES): partial inputs, the filter
that corrects only the input, and the filter that corrects the input and the reservoir. Each row shows, for each
case,
* the z-x plane (x vertical, on the scale of the time series beside it): every member from the last WINDOW_LYAP
  of the data to the end of the closed loop (thin; red: those that leave the attractor, drawn up to that point and
  left out of the mean, the band and the scores) and the true state over that last WINDOW_LYAP (grey), over the
  attractor of the training record (light grey). From z alone, (x, y, z) and (-x, -y, z) give the same data, so
  the members of the filter split between the two lobes, above and below in this plane;
* the unmeasured component x in time, over the data and the closed loop of LOOP_LYAP after the last observation
  (dotted): the members, the ensemble mean (dashed) with its 2-sigma band, and the true state; the percentage of
  the members of the ensemble drawn that leave the attractor, when any does (for the partial inputs, the
  realizations driven by the same data), and, printed only, that over N_ENSEMBLES independent ensembles;
and, for both cases in one panel, the error of the ensemble mean (solid) and the spread (dotted) in time, medians
over the N_ENSEMBLES ensembles.
Reads results/partial/lorenz63_phase.npz (run_partial_table.py phase), whose docstring describes the deployments:
the data of Table 1 (every 10 steps, sigma_obs = 0.1, m = 10), the members started from training states one
observation interval before the first datum; from z alone, both filters run the Gaussian-mixture analysis of up to two
components, and every member closer to the mirror of the true state, (-x, -y, z), than to the true state over the last
Lyapunov time of the data is mirrored in the time series, the mean, the band and e and s (the phase panels show the
members unmirrored). A member that leaves twice the range of the training data (detectable without the true state) is
left out of the mean, the band, the spread and the scores. The legend lists symbols only, each in both shades.

    python plot_lorenz63_phase.py    -> PAPER/lorenz63_phase.pdf
"""

import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plot_aux import C_DIVERGED, C_OBS, C_TRUE, ORANGE, RESULTS, SPREAD_SYNC, VERMILION, letter, save

from romda.metrics import first_below  # noqa: E402
from run_partial_table import FORECAST_LYAP as LOOP_LYAP, N_ENSEMBLES  # noqa: E402  (closed loop after the data)

#: (case, title, colour): shades of blue, light to dark as the analysis corrects more
MODES = (('none', 'partial inputs', matplotlib.colors.to_hex(plt.cm.Blues(0.45))),
         ('input', 'filter, input only', matplotlib.colors.to_hex(plt.cm.Blues(0.7))),
         ('reservoir', 'filter, input and reservoir', matplotlib.colors.to_hex(plt.cm.Blues(0.95))))

#: measured components, title and shade of each case: a light and a dark blue, far apart
CASES = (([1], r'from $y$', '#4BA3E3'), ([2], r'from $z$', '#0A3D91'))
C_LOOP = {'#4BA3E3': '#5AAE61', '#0A3D91': '#1B7837'}                 # the forecast attractor by case: light, dark green
C_ES = {'#4BA3E3': ('k', ORANGE), '#0A3D91': ('k', VERMILION)}   # (e, s) by case: e black (grey is the true
# state), s light from y, dark from z
PLANE = (2, 0)                                     # z horizontal, x vertical: the lobes above and below; x shared with the time series
WINDOW_LYAP = 1.3                                  # phase panels: the last 1.3 T_lambda of the data, about two turns on a wing
ESCAPE_LYAP = 0.3                                  # phase panels: a diverged member over its last 0.3 T_lambda before it leaves
C_ATTRACTOR = '0.88'
SCORE_FROM = 3.5                                   # printed medians of e and s: from here to the last observation
plt.rcParams['legend.fontsize'] = 10.3             # 9 pt once the 7.4 in figure is scaled to the text width


def case_runs(dims):
    """The record of the deployments with `dims` measured: the ensemble drawn of each deployment, with the error,
    the spread and the fraction of diverged members of its N_ENSEMBLES ensembles, and the attractor of the training
    record."""
    z, c = np.load(os.path.join(RESULTS, 'partial', 'lorenz63_phase.npz')), 'xyz'[dims[0]]
    shown = {}
    for mode, _, _ in MODES:
        r = {k[len(f'{c}_{mode}_'):]: z[k] for k in z.files if k.startswith(f'{c}_{mode}_')}
        r['t_last'], r['m'] = float(r['t_last']), int((~r['bad']).sum())
        shown[mode] = r
    return shown, z[f'{c}_attractor']


def phase(ax, r, attractor, colour):
    """Every member and the true state over the same window, the last WINDOW_LYAP of the data, in the z-x plane turned
    a quarter turn anticlockwise (-x right, z up);
    a member that leaves the attractor (red, thin) over its last ESCAPE_LYAP before it leaves."""
    i, k = PLANE
    ax.plot(-attractor[::2, k], attractor[::2, i], color=C_ATTRACTOR, lw=0.3, zorder=0)
    t, t_o, U = r['t'], r['t_last'], r['U_raw']                       # unmirrored: from z, both lobes
    win = (t >= t_o - WINDOW_LYAP) & (t <= t_o)
    lo, hi = attractor.min(0), attractor.max(0)
    pad = 0.1 * (hi - lo)
    for j in range(U.shape[-1]):
        if r['bad'][j]:                                                 # diverged: its last ESCAPE_LYAP on the attractor
            n_out = int(np.argmax(r['out'][:, j]))
            inside = np.all((U[:, :, j] >= lo - pad) & (U[:, :, j] <= hi + pad), axis=1)   # within the panel
            seg = (t >= t[n_out] - ESCAPE_LYAP) & (np.arange(t.size) < n_out) & inside
            ax.plot(-U[seg, k, j], U[seg, i, j], lw=0.4, alpha=0.8, color=C_DIVERGED, zorder=2)
        else:
            ax.plot(-U[win, k, j], U[win, i, j], lw=1.0, alpha=0.8, color=colour, zorder=2)
    ax.plot(-r['truth'][win, k], r['truth'][win, i], color=C_TRUE, lw=0.8, zorder=3)   # thin, over the members
    ax.set(xlim=(-hi[k] - pad[k], -lo[k] + pad[k]), ylim=(lo[i] - pad[i], hi[i] + pad[i]))


def phase_forecast(ax, r, attractor, colour):
    """Every member that stays on the attractor over the closed loop after the last observation (LOOP_LYAP), in the
    z-x plane turned a quarter turn
    anticlockwise: the longer run, free of the jumps of the analyses."""
    i, k = PLANE
    ax.plot(-attractor[::2, k], attractor[::2, i], color=C_ATTRACTOR, lw=0.3, zorder=0)
    t, t_o, U = r['t'], r['t_last'], r['U_raw']                       # unmirrored
    loop = t > t_o + 1e-9
    for j in np.flatnonzero(~r['bad']):
        ax.plot(-U[loop, k, j], U[loop, i, j], lw=0.4, alpha=0.6, color=colour, zorder=2)
    lo, hi = attractor.min(0), attractor.max(0)
    pad = 0.1 * (hi - lo)
    ax.set(xlim=(-hi[k] - pad[k], -lo[k] + pad[k]), ylim=(lo[i] - pad[i], hi[i] + pad[i]))


def x_time(ax, r, colour, k, dims, off):
    """Component k in time, over the data and the closed loop of the ensemble drawn: the members, the ensemble mean
    (dashed) with its 2-sigma band, the true state (grey) and, when k is measured (`dims`), the observations;
    diverged members in red; `off` shifts the component vertically."""
    t, U, truth = r['t'], r['U'] + off * np.eye(3)[k][None, :, None], r['truth'] + off * np.eye(3)[k]
    ax.plot(t, np.where(r['out'], np.nan, U[:, k, :])[:, r['bad']], color=C_DIVERGED, lw=0.4)
    if r['m']:
        mean, std = r['mean'] + off * np.eye(3)[k], r['std']
        ax.plot(t, U[:, k, ~r['bad']], color=colour, lw=0.3, alpha=0.3)
        ax.fill_between(t, mean[:, k] - 2 * std[:, k], mean[:, k] + 2 * std[:, k], color=colour, alpha=0.25, lw=0)
    ax.plot(t, truth[:, k], color=C_TRUE, lw=1.0, zorder=0.5)         # under the members and the mean
    if r['m']:
        ax.plot(t, mean[:, k], color=colour, lw=1.1, ls='--')
    if k in dims:                                                       # the data, on top
        ax.plot(r['t_obs'], r['d'][:, list(dims).index(k)] + off, 'o', color=C_OBS, ms=2.2, mec='k', mew=0.3, zorder=5)
    ax.axvline(r['t_last'], color='k', lw=0.8, ls=':')


def error_spread(ax, runs):
    """Error (grey/black) and spread (gold/orange) in time for each case, medians over the ensembles: `runs` lists
    (ensemble drawn, with the error and the spread of its N_ENSEMBLES ensembles, shade)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)                  # all-nan columns once a run has left
        for r, colour in runs:
            for i, ls, lw in ((0, '-', 1.0), (1, '-', 1.2)):          # e and s: told apart by colour
                q = r[('error', 'spread')[i]]                           # (N_ENSEMBLES, Nt)
                if not np.isnan(q).all():
                    ax.semilogy(r['t'], np.nanmedian(q, axis=0), color=C_ES[colour][i], lw=lw, ls=ls)
    ax.axvline(runs[0][0]['t_last'], color='k', lw=0.8, ls=':')
    ax.axhline(SPREAD_SYNC, color='k', lw=0.6, ls=':')              # the threshold of t_e and t_s, as in Fig. 5
    ax.set_ylim(1e-2, 10)                                               # a collapsed spread leaves at the bottom
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def scores(r):
    """Printed check of the text: diverged members, synchronization times and the medians of e and s."""
    t, E, S = r['t'], r['error'], r['spread']
    t_e, t_s = (np.nanmedian([first_below(t, q, SPREAD_SYNC, r['t_last']) for q in Q]) for Q in (E, S))
    score = (t >= SCORE_FROM) & (t <= r['t_last'])
    return (f'{r["bad"].sum()} of {r["bad"].size} members diverge (ensemble drawn), '
            f'{100 * np.mean(r["diverged"]):.0f}% over {N_ENSEMBLES} ensembles; '
            f't_e {t_e:.2f}, t_s {t_s:.2f}; medians over [{SCORE_FROM}, {r["t_last"]:.1f}] T_lambda: '
            f'e {np.nanmedian(E[:, score]):.3f}, s {np.nanmedian(S[:, score]):.3f}')


# the layout, in inches: margins, time series (y, z above e, s), attractors (data above forecast), gaps
FIG_W, LEFT, TOP, BOTTOM = 7.4, 0.55, 0.22, 0.72
TS_H, ES_H, TS_GAP, ROW_GAP, COL_GAP = 0.95, 0.52, 0.08, 0.35, 0.22
ATT_W, ATT_H, ATT_GAP = 0.55, 0.6, 0.04
LABEL_PAD = 31                                     # points from the axes to the names y, z and e, s
TS_W = (FIG_W - LEFT - 0.02 - 2 * COL_GAP) / 3 - ATT_GAP - ATT_W


def main():
    """Columns: the deployments (a)-(c). Rows: from y and from z, each with y and z in time (attached, z shifted
    down) above their e and s, and the members in the z-x plane on their right, data above forecast."""
    results = {tuple(dims): case_runs(dims) for dims, _, _ in CASES}
    n_cases = len(CASES)
    fig_h = TOP + n_cases * (TS_H + TS_GAP + ES_H) + (n_cases - 1) * ROW_GAP + BOTTOM
    fig = plt.figure(figsize=(FIG_W, fig_h))
    box = lambda x, y, w, h: fig.add_axes([x / FIG_W, 1 - (y + h) / fig_h, w / FIG_W, h / fig_h])   # from the top left
    phases, comps, ess = [], [], []
    for col, (mode, title, _) in enumerate(MODES):
        first = col == 0
        x0 = LEFT + col * (TS_W + ATT_GAP + ATT_W + COL_GAP)
        for b, (dims, case, colour) in enumerate(CASES):
            last = b == n_cases - 1
            shown, attractor = results[tuple(dims)]
            r = shown[mode]
            y0 = TOP + b * (TS_H + TS_GAP + ES_H + ROW_GAP)
            ax_t = box(x0, y0, TS_W, TS_H)                               # y above z, attached: z shifted down
            ax_e = box(x0, y0 + TS_H + TS_GAP, TS_W, ES_H)              # e and s of this case
            ax_p = box(x0 + TS_W + ATT_GAP, y0, ATT_W, ATT_H)           # the data, above
            ax_f = box(x0 + TS_W + ATT_GAP, y0 + ATT_H + 0.05, ATT_W, ATT_H)   # the closed loop after the data
            error_spread(ax_e, [(r, colour)])
            ax_e.set(xlabel=r'$t / T_\lambda$' if last else None, yticks=[1e-2, 1e0])
            ax_e.tick_params(labelbottom=last, labelleft=first)
            ess.append(ax_e)
            phase(ax_p, r, attractor, colour)
            ax_p.set_axis_off()                                           # the attractor: no axes
            phase_forecast(ax_f, r, attractor, C_LOOP[colour])
            ax_f.set_axis_off()
            phases.append(ax_f)
            pct = 100 * r['bad'].mean()                                  # of the ensemble drawn: its red members
            if round(pct):                                               # members that leave the attractor
                ax_p.text(0.0, 0.0, f'{pct:.0f}% diverge', transform=ax_p.transAxes, ha='left', va='top',
                          fontsize=8)                                   # under the panel
            lo, hi = attractor.min(0), attractor.max(0)
            pad = 0.08 * (hi - lo)
            off = (lo[1] - pad[1]) - (hi[2] + pad[2])                      # the top of z's band at the bottom of y's
            for k, o in ((1, 0.0), (2, off)):
                x_time(ax_t, r, colour, k, dims, o)
            ax_t.axhline(lo[1] - pad[1], color='k', lw=0.6)
            ax_t.set_ylim(lo[2] - pad[2] + off, hi[1] + pad[1])
            ax_t.set_yticks([-20, 20, 10 + off, 30 + off], ['$-20$', '$20$', '$10$', '$30$'])   # z's values, shifted
            ax_t.tick_params(axis='y', length=2, pad=1, labelleft=first)
            ax_t.tick_params(labelbottom=False)
            if first:
                for ax, c, y in ((ax_t, '$y$', 0.5 * (lo[1] + hi[1])), (ax_t, '$z$', 0.5 * (lo[2] + hi[2]) + off),
                                 (ax_e, '$e$, $s$', 0.5)):              # the names, one distance from the axes
                    ax.annotate(c, xy=(0, y), xytext=(-LABEL_PAD, 0), textcoords='offset points', ha='center',
                                va='center', rotation=90, xycoords=blended_transform_factory(
                                    ax.transAxes, ax.transAxes if ax is ax_e else ax.transData))
            if b == 0:                                                   # the deployment: its letter, above
                ax_t.text(0.0 if first else x0 / FIG_W - 0.02, 1 - 0.02 / fig_h, letter(col),
                          transform=fig.transFigure, ha='left', va='top', fontsize=12)
            phases.append(ax_p)
            comps.append(ax_t)
            print(f'{case:10s} {title:28s}: {scores(r)}')
    for ax in phases[1:]:                                                # one scale per kind of panel
        ax.sharex(phases[0])
        ax.sharey(phases[0])
    for ax in comps[1:] + ess:
        ax.sharex(comps[0])
    for ax in ess[1:]:
        ax.sharey(ess[0])
    comps[0].set_xlim(right=8.0)                                        # the time axes up to 8 T_lambda
    comps[0].set_xticks(np.arange(0, 9, 2))
    light, dark = (colour for _, _, colour in CASES)
    pair = lambda **kw: tuple(Line2D([], [], color=c, **kw) for c in (light, dark))   # both shades, side by side
    handles = [Line2D([], [], color=C_TRUE, lw=1.3), pair(lw=1.1, ls='--'),
               tuple(Patch(color=c, alpha=0.25, lw=0) for c in (light, dark)),
               Line2D([], [], ls='', marker='o', color=C_OBS, ms=3, mec='k', mew=0.3), Line2D([], [], color='k', lw=1.0),
               tuple(Line2D([], [], color=C_ES[c][1], lw=1.2) for c in (light, dark)), tuple(Line2D([], [], color=C_LOOP[c], lw=1.2) for c in (light, dark))]
    labels = [r'$u^\dagger$', r'$\bar{u}$', r'$\pm2\sigma$', '$d$', '$e$', '$s$',
              rf'forecast after DA (${LOOP_LYAP:.0f}\,T_\lambda$)']
    fig.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=2, pad=0)}, loc='lower center', bbox_to_anchor=(0.5, 0.0),
               ncol=len(handles), frameon=False, columnspacing=1.5, handlelength=2.6, handletextpad=0.5)
    save(fig, 'lorenz63_phase.pdf', dpi=200, bbox_inches=None)


if __name__ == '__main__':
    main()
