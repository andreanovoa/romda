"""fig:param and fig:kld of the washout-free paper: the state and parameter estimation of the parametric L96-10s.
Reads the scans of run_parametric.py (scan, with and without FIX_F=1; attractor, with and
without FIX_F=1, and the KLD of the perfect model that attractor writes) and writes param_scan_lorenz96p.pdf (fig:param) and param_kld_lorenz96p.pdf (fig:kld).

(a) The estimated against the true forcing after the same number of analyses for every interval, with the trained
    range shaded: the estimate follows the truth inside the range and saturates outside it.
(b) The ensemble mean and 2-sigma band of F against time at F = 8.5, beyond the trained range, up to 10 T_lambda.
(c) Attractor Kullback-Leibler divergence of each member after the last observation, which measures whether the
    reconstructed dynamics, and not only the parameter, are right; the attractors of the extrapolation cases
    (`phase`) below it.
"""
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Patch, Rectangle
from plot_aux import RESULTS, VERMILION, panel_letter, save, style
from scipy.ndimage import binary_dilation


#: the style of this figure: 9 pt, grid off, ticks out
SCAN_STYLE = {'font.serif': ['TeX Gyre Termes', 'STIXGeneral', 'DejaVu Serif'], 'font.size': 9, 'axes.labelsize': 9,
              'legend.fontsize': 8.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
              'axes.linewidth': 0.6, 'grid.color': '#e1e0d9', 'grid.linewidth': 0.5, 'axes.axisbelow': True,
              'axes.grid': False, 'xtick.direction': 'out', 'ytick.direction': 'out'}
KLD_FLOOR = 1e-4
F_HISTORY = 8.5                                         # the history of (b): beyond the trained range


F_MID = 5.75


def attractor_runs():
    """The closed loops of the extrapolation cases (F = 3 and 8.5, beyond the trained range) after the data: the true
    system and one member of the network at its estimated forcing and at the forcing held at the centre of the trained
    range (run_parametric.py attractor, with and without FIX_F=1)."""
    return (np.load(os.path.join(RESULTS, 'parameter', 'param_attractor_lorenz96p.npz')),
            np.load(os.path.join(RESULTS, 'parameter', 'param_attractor_lorenz96p_fixF.npz')))


def phase(ax, i, est, fix, runs_drawn):
    """Case i (0: F = 3, 1: F = 8.5) in the (u_1, u_2) plane of ax over the true attractor, shaded, and the networks
    of runs_drawn (label, colour, index) as lines."""
    runs = (est[f'truth_{i}'], est[f'free_{i}'][:, :, 0], fix[f'free_{i}'][:, :, 0])   # member 0 of each
    lim = np.percentile(np.abs(np.concatenate([r[:, :2].ravel() for r in runs])), 99.5) * 1.1
    u = runs[0]                                                  # the true attractor: cells visited by any pair
    pairs = np.concatenate([u[:, [k, (k + 1) % u.shape[1]]] for k in range(u.shape[1])])   # of neighbouring sites
    H, xe, ye = np.histogram2d(pairs[:, 0], pairs[:, 1], bins=120, range=((-lim, lim), (-lim, lim)))
    H = binary_dilation(H.T > 0, iterations=2) if i == 0 else H.T > 0   # a limit cycle is a curve: widened
    ax.contourf((xe[1:] + xe[:-1]) / 2, (ye[1:] + ye[:-1]) / 2, H.astype(float), levels=[0.5, 1.5], colors=['0.6'])
    for label, c, k in runs_drawn:
        ax.plot(runs[k][:, 0], runs[k][:, 1], color=c, lw=(1.0 if i == 0 else 0.5) if k == 1 else 0.3)
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect='equal')


def attractor_handles(runs_drawn):
    """Legend entries of the attractors: the true system, the estimated and the fixed forcing."""
    return ([Patch(color='0.6')] + [Line2D([], [], color=c, lw=1.2) for _, c, k in runs_drawn],   # also the violins
                                                                                                  # of Fig. param(c)
            ['true system'] + [label for label, _, _ in runs_drawn])


def main():
    style(**SCAN_STYLE)
    os.makedirs(os.path.join(RESULTS, 'parameter'), exist_ok=True)
    import run_parametric as parallel                # the runs of run_parametric.py scan
    F_TRUE, INTERVALS = parallel.F_TRUE, parallel.INTERVALS   # the true forcings and the observation intervals
    train = np.asarray(parallel.VALUES)
    lo, hi = train.min(), train.max()
    lo_ax = min(lo, min(F_TRUE) - 0.3)                  # the axes reach a truth below the trained range
    span = 0.12 * (hi - lo)
    gap = min(np.diff(sorted(F_TRUE)))                  # the violins and dodges scale with the grid

    cache = os.path.join(RESULTS, 'parameter', 'param_scan_lorenz96p_gammainnovation.npz')
    if not os.path.exists(cache):
        sys.exit(f'{cache} is missing: run run/run_parametric.py scan first')
    rows = list(np.load(cache, allow_pickle=True)['rows'])
    # (a): four intervals across the range, (b) and (c) all
    drawn = [r for r in rows if r['rho_true'] == F_HISTORY and r['interval'] in (0.024, 0.3, 1.0, 3.0)]

    # two figures of one column: (a) with its colour bar above (b); (c) above the attractors of the two extrapolated
    # forcings
    left = fig = plt.figure(figsize=(3.43, 4.3), layout='constrained')
    right = plt.figure(figsize=(3.43, 3.2), layout='constrained')   # no blank band below the attractors
    fig_c = right                                # the figure that joins the boxes of (c) to the attractors
    top_l, bottom_l = left.subfigures(2, 1, height_ratios=(1.4, 1))   # (b) shorter than (a)
    top_r, bottom_r = right.subfigures(2, 1, height_ratios=(1.8, 1))
    b, key = top_l.subplots(1, 2, width_ratios=(1, 0.12))     # the estimates (a) and their colour bar
    a = bottom_l.subplots()                                       # the history (b)
    c = top_r.subplots()                                          # (c) on top, its x axis along the top edge,
    cells = bottom_r.subplots(1, 2)                               # the attractors below it, smaller than their cells,
    for cell in cells:                                            # each with its forcings on its right
        cell.axis('off')
    att = [cell.inset_axes((0.06, 0.04, 0.6, 0.92)) for cell in cells]
    axs = [a, b, c]
    c.xaxis.tick_top()
    c.xaxis.set_label_position('top')
    c.spines['top'].set_visible(True)
    c.spines['bottom'].set_visible(False)
    panels = list(axs)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(INTERVALS)))   # densest data darkest
    # the networks drawn over the true attractor (shaded), and their violins: the estimated forcing, in the colour of
    # its interval, then the fixed one on top, thinner
    runs_drawn = [('estimated $F$', colors[INTERVALS.index(0.1)], 1), (f'$F$ fixed at {F_MID:g}', VERMILION, 2)]
    t_end = max(r['trace']['t'][-1] for r in drawn)                 # the last analysis of any interval
    for r in sorted(drawn, key=lambda r: -r['interval']):           # the densest data drawn last, on top
        h, c = dict(r['trace']), colors[INTERVALS.index(r['interval'])]
        if h['t'][-1] < t_end:                                       # held at its last analysis up to t_end
            h.update({k: np.append(h[k], h[k][-1] if k != 't' else t_end) for k in ('t', 'mean', 'std')})
        axs[0].fill_between(h['t'], h['mean'] - 2 * h['std'], h['mean'] + 2 * h['std'], color=c,
                            alpha=0.15, lw=0, step='post')   # F is constant between
        axs[0].plot(h['t'], h['mean'], color=c, lw=1.0, drawstyle='steps-post')   # analyses
    axs[0].fill_between([], [], [], color='0.5', alpha=0.3, lw=0, label=r'mean $\pm\,2\sigma$')
    axs[0].axhline(F_HISTORY, color='k', ls='--', lw=0.9, label='true value')
    axs[0].set(xlabel=r'$t / T_\lambda$', ylabel='estimated $F$')
    axs[0].set_xlim(-0.3, 10)                       # the first 10 T_lambda of the data
    axs = axs[1:]
    axs[0].fill_between([lo, hi], lo, hi, color='0.5', alpha=0.12, lw=0)   # the trained range in x and y
    # labelled on the plot
    axs[0].text(lo + 0.1, hi - 0.1, 'trained\nrange', fontsize=7, color='0.35', ha='left', va='top')
    axs[0].plot([lo_ax - span, hi + 2 * span], [lo_ax - span, hi + 2 * span], 'k--', lw=0.9)
    n_an = min(len(r['trace']['mean']) for r in rows) - 1
    for ii, interval in enumerate(INTERVALS):      # n_an: the same number of analyses for every interval
        pts = [r for r in rows if r['interval'] == interval]
        x = np.array([r['rho_true'] for r in pts]) + 0.08 * gap * (ii - (len(INTERVALS) - 1) / 2)
        y, e = (np.array([r['trace'][k][n_an] for r in pts]) for k in ('mean', 'std'))
        seen = np.isin([r['rho_true'] for r in pts], train)
        axs[0].errorbar(x, y, yerr=e, fmt='none', lw=1, capsize=2, color=colors[ii],
                        label=f'$\\Delta t_\\mathrm{{obs}} = {interval:.1f}\\,T_\\lambda$')
        out = np.array([not lo <= r['rho_true'] <= hi for r in pts])   # beyond the trained range: triangles
        axs[0].plot(x[seen], y[seen], 'o', ms=4, mfc=colors[ii], mec=colors[ii])
        axs[0].plot(x[~seen & ~out], y[~seen & ~out], 'o', ms=4, mfc='w', mec=colors[ii], mew=1)
        axs[0].plot(x[out], y[out], '^', ms=4.5, mfc='w', mec=colors[ii], mew=1)
    # a key of the markers
    marks = [Line2D([], [], ls='', marker='o', ms=4, mfc='0.4', mec='0.4'),
             Line2D([], [], ls='', marker='o', ms=4, mfc='w', mec='0.4', mew=1),
             Line2D([], [], ls='', marker='^', ms=4.5, mfc='w', mec='0.4', mew=1)]
    axs[0].add_artist(axs[0].legend(marks, ['training', 'interpolation', 'extrapolation'],
                                    loc='lower right', frameon=False, fontsize=7, handletextpad=0.2,
                                    borderaxespad=0.2))
    axs[0].plot(train, train, 'k|', ms=7)
    axs[0].text(0.03, 0.97, f'after {n_an} analyses', transform=axs[0].transAxes, fontsize=8, ha='left', va='top')
    axs[0].set(xlabel='true $F$', ylabel='estimated $F$',
               xlim=(lo_ax - span, hi + 2 * span), ylim=(lo_ax - span, hi + 2 * span))

    axs[1].axvspan(lo, hi, color='0.5', alpha=0.12, lw=0)
    fixed = cache.replace('.npz', '_fixF.npz')                     # F held at the middle of the range
    ref = list(np.load(fixed, allow_pickle=True)['rows']) if os.path.exists(fixed) else None
    # per forcing, over the members and the intervals: a violin of
    for rs, (_, c, _), side in zip((rows, ref), runs_drawn, (-1, 1)):   # the estimated forcing, left, and
        if rs is None:                                                       # of the fixed one, right
            continue
        vals = [np.log10(np.clip(np.concatenate([r['kld'] for r in rs if r['rho_true'] == v]), KLD_FLOOR, None))
                for v in F_TRUE]
        vp = axs[1].violinplot(vals, positions=np.asarray(F_TRUE) + side * 0.17 * gap, widths=0.32 * gap,
                               showmedians=True, showextrema=False)
        for body in vp['bodies']:
            body.set(facecolor=c, alpha=0.7, edgecolor='none')
        vp['cmedians'].set(color='k', lw=0.8)
    # the sides of (c) left to the attractors
    axs[1].set_xlim(2.3, max(F_TRUE) + 0.5 * gap)    # the violins only; the attractors sit below (c)
    axs[1].set(xlabel='true $F$', ylabel=r'$\log_{10}$ attractor KLD per member', xticks=F_TRUE,
               xticklabels=[f'{v:g}' for v in F_TRUE])
    # the x axis of (c) along its top edge
    axs[1].xaxis.set_label_position('top')
    axs[1].set_ylabel(r'$\log_{10}\,$KLD')    # the details in the caption
    # the perfect model: Lorenz-96 at the true forcing (run_parametric.py attractor, kld_floor)
    floor = np.load(os.path.join(RESULTS, 'parameter', 'param_kld_perfect_lorenz96p.npz'))['floor']
    axs[1].hlines(np.log10(floor), np.array(F_TRUE) - 0.4 * gap, np.array(F_TRUE) + 0.4 * gap, color='k', lw=1.2,
                  label='true $F$')
    est, fix = attractor_runs()
    for i, ax in enumerate(att):
        phase(ax, i, est, fix, runs_drawn)
        u = np.concatenate([est[f'truth_{i}'][:, :2], est[f'free_{i}'][:, :2, 0], fix[f'free_{i}'][:, :2, 0]])
        lo_u, hi_u = u.min(axis=0), u.max(axis=0)   # the attractor centred in its box, with a margin
        half = 0.6 * np.max(hi_u - lo_u)
        ax.set(xlim=(lo_u[0] + hi_u[0]) / 2 + np.array([-half, half]),
               ylim=(lo_u[1] + hi_u[1]) / 2 + np.array([-half, half]), xticks=[], yticks=[])
        ls, grey = ':', '0.6'                   # the box of the attractor and of its violins, light
        for spine in ax.spines.values():
            spine.set(visible=True, edgecolor=grey, lw=0.8, ls=ls)
        F = float(est[f'F_{i}'])                 # a box around the violins of this forcing
        y = np.log10(np.clip(np.concatenate([r['kld'] for rs in (rows, ref) if rs for r in rs
                                             if r['rho_true'] == F]), KLD_FLOOR, None))
        x0, x1, y0, y1 = F - 0.4 * gap, F + 0.4 * gap, y.min() - 0.1, y.max() + 0.1
        axs[1].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=grey, lw=0.8, ls=ls))
        for xb, xa in ((x0, 0), (x1, 1)):        # the box joined to its attractor below, corner to corner
            fig_c.add_artist(ConnectionPatch((xb, y0), (xa, 1), coordsA=axs[1].transData, coordsB=ax.transAxes,
                                           color=grey, lw=0.8, ls=ls))
        for k, (value, c) in enumerate(((f'{F:.1f}', '0.5'), (f'{np.mean(est[f"F_est_{i}"]):.2f}', runs_drawn[0][1]),
                                        (f'{F_MID:g}', runs_drawn[1][1]))):   # on the right, in the colours of the
            ax.text(1.05, 0.95 - 0.14 * k, f'$F={value}$', transform=ax.transAxes, color=c, fontsize=7,
                    ha='left', va='top')                                # true system, the estimate, the fixed F
    extra = attractor_handles(runs_drawn)

    handles, labels = zip(*[hl for ax in panels for hl in zip(*ax.get_legend_handles_labels())])
    handles, labels = [*handles, Patch(facecolor='0.5', alpha=0.25), *extra[0]], [*labels, 'trained range', *extra[1]]
    # the references above (b) and a colour bar of the intervals in the key
    ref = [j for j, l in enumerate(labels) if 'Delta' not in l and l not in (*extra[1], 'trained range', 'true $F$')]
    panels[0].legend([handles[j] for j in ref], [labels[j] for j in ref], loc='lower right', bbox_to_anchor=(1, 1),
                     frameon=False, fontsize=8, ncol=len(ref), columnspacing=1.0, handlelength=1.8,
                     borderaxespad=0.1)          # the references in one row above (b); the colour bar alone in the key
    cmap = matplotlib.colors.ListedColormap(colors)
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(matplotlib.colors.BoundaryNorm(np.arange(len(INTERVALS) + 1),
                      cmap.N), cmap), cax=key.inset_axes((0.4, 0.08, 0.45, 0.84)), orientation='vertical')
    cb.set_ticks(np.arange(len(INTERVALS)) + 0.5, labels=[f'{v:g}' for v in INTERVALS])
    cb.ax.tick_params(labelsize=8, length=0)
    cb.ax.set_title(r'$\Delta t_\mathrm{obs} / T_\lambda$', fontsize=8)   # on top, horizontal
    key.axis('off')
    h_c, l_c = axs[1].get_legend_handles_labels()   # the key of (c) inside it: the violins and the true forcing
    axs[1].legend([*[Patch(facecolor=c, alpha=0.7) for _, c, _ in runs_drawn], *[h for h, l in zip(h_c, l_c) if l == 'true $F$']],
                  [*[label for label, _, _ in runs_drawn], 'true $F$'], loc='lower center', bbox_to_anchor=(0.6, 0),
                  frameon=False, fontsize=8)          # inside the trained range, clear of the box of F = 8.5
    for ax, i in zip(panels[:2], (1, 0)):          # (b), (a), flush left of
        panel_letter(ax, i, fontsize=12, offset=(0, 1), points=(-32, 6))   # each panel's labels; (c) alone: no letter
    save(fig, 'param_scan_lorenz96p.pdf', bbox_inches=None)
    save(right, 'param_kld_lorenz96p.pdf', bbox_inches=None)


if __name__ == '__main__':
    main()
