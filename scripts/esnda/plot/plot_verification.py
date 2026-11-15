"""Appendix A of the washout-free paper (fig:verification): do the trained networks reproduce the dynamics and
statistics of the systems? One system per column (L63, L96-10s and L96-10p, L96-40s and L96-40p), both architectures in
the same panels; rows: probability density, autocorrelation and the leading Lyapunov exponents of the closed-loop
networks against the true system.

Reads results/verification/esn_verification_runs.pkl (run_verification.py, whose docstring describes the runs).

    python plot_verification.py   -> PAPER/esn_verification.pdf
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from plot_aux import C_TRUE_DARK, panel_letter, save

from run_verification import load_runs  # noqa: E402

LW_TRUE = 2.2                                       # the truth under the networks' dashes


def main():
    plot(load_runs())


ARCH = {'one': dict(ls='--', marker='x', label='single reservoir'),
        'par': dict(ls='-.', marker='+', label='parallel')}
COLOR = {'Lorenz-63': '#0088DD',                                                  # one hue per row (Paul Tol):
         'Lorenz-96, 10 sites': '#A50026', 'Lorenz-96, 10 sites, parallel': '#FDAE61',   # reds, dark: one reservoir
         'Lorenz-96, 40 sites, one reservoir': '#117733', 'Lorenz-96, 40 sites, parallel': '#BBCC33'}   # green, pear
PANELS = [('L63', [('Lorenz-63', 'one')]),
          ('L96-10s\nL96-10p', [('Lorenz-96, 10 sites', 'one'), ('Lorenz-96, 10 sites, parallel', 'par')]),
          ('L96-40s\nL96-40p', [('Lorenz-96, 40 sites, one reservoir', 'one'),
                                         ('Lorenz-96, 40 sites, parallel', 'par')])]


def plot(done):
    """One system per column, both architectures in the same panels; rows: density, autocorrelation, Lyapunov exponents
    (one column of the paper). The Lorenz-63 components are offset in every row but the exponents. axs[system, row]."""
    from matplotlib.lines import Line2D
    plt.rcParams.update({'xtick.labelsize': 7, 'ytick.labelsize': 7, 'xtick.major.pad': 1.5, 'ytick.major.pad': 1.5})
    fig, axs = plt.subplots(3, len(PANELS), figsize=(0.3 + 3.1 * len(PANELS) / 3, 4.4), layout='constrained',
                            height_ratios=[1, 1, 1.5], squeeze=False)
    axs = axs.T                                                     # axs[system, quantity]
    for ax in axs[1:, 1]:
        ax.sharex(axs[0, 1])
    for row, (label, nets) in enumerate(PANELS):
        components = done[nets[0][0]]['fc'].shape[1] == 3          # Lorenz-63: y and z in one panel, no offset
        for j, (name, arch) in enumerate(nets):
            c, st, a = done[name], done[name]['st'], dict(ARCH[arch], color=COLOR[name])
            for s in (st[1:] if components else st):                # one group (pooled), or y and z
                centres = 0.5 * (s['edges'][1:] + s['edges'][:-1])
                axs[row, 0].plot(centres, s['py'], color=C_TRUE_DARK, lw=LW_TRUE, zorder=1)
                axs[row, 0].plot(centres, s['pz'], color=a['color'], ls=a['ls'], lw=1.0)
                axs[row, 1].plot(s['lags'], s['acf_y'], color=C_TRUE_DARK, lw=LW_TRUE, zorder=1)
                axs[row, 1].plot(s['lags'], s['acf_z'], color=a['color'], ls=a['ls'], lw=1.0)
            ax = axs[row, 2]                         # mean and std over the starts; one truth per row
            i = np.arange(1, c['lam_true'].shape[1] + 1)
            if j == 0:
                ax.errorbar(i, c['lam_true'].mean(0), c['lam_true'].std(0), color=C_TRUE_DARK, marker='o', ms=5, lw=0, zorder=1,
                            elinewidth=0.8, mfc='none')
            ax.errorbar(i, c['lam_net'].mean(0), c['lam_net'].std(0), color=a['color'], marker=a['marker'], ms=3.5,
                        lw=0, elinewidth=0.8)
        panel_letter(axs[row, 0], row)
        if components:                                              # name each curve where it is alone
            sy, sz = done[nets[0][0]]['st'][1:]
            arrow = dict(arrowstyle='-|>', lw=0.5, color='0.3', shrinkA=1, shrinkB=1, mutation_scale=5)
            for s, name, q, text in ((sy, 'y', 0.12, (0.08, 0.6)), (sz, 'z', 0.88, (0.95, 0.75))):   # outer tails
                centres = 0.5 * (s['edges'][1:] + s['edges'][:-1])
                n = int(q * (len(centres) - 1))
                axs[row, 0].annotate(f'${name}$', (centres[n], s['py'][n]), xytext=text, textcoords='axes fraction',
                                     fontsize=8, ha='center', va='center', arrowprops=arrow)
            a, lags = sz['acf_y'], sz['lags']                       # the peaks of the oscillating z
            peaks = np.flatnonzero((a[1:-1] > a[:-2]) & (a[1:-1] > a[2:]) & (lags[1:-1] > 0.2)) + 1
            n_y, n_z = peaks[1], peaks[-1]                          # y under the 2nd peak of z, z above its last
            axs[row, 1].annotate('$y$', (lags[n_y], sy['acf_y'][n_y]), xytext=(lags[n_y], -0.4), ha='center',
                                 va='center', fontsize=8, arrowprops=arrow)   # below y, where z is up
            axs[row, 1].annotate('$z$', (lags[n_z], a[n_z]), xytext=(lags[n_z] - 0.25, 0.85), ha='center',
                                 va='center', fontsize=8, arrowprops=arrow)   # above z
        axs[row, 2].axhline(0, color='0.7', lw=0.5, zorder=0)
        axs[row, 2].margins(x=0.1)
        axs[row, 2].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3, integer=True, steps=[1, 2, 5, 10]))
    for ax, numeral in zip(axs[0], [r'$\mathbf{I}$', r'$\mathbf{II}$', r'$\mathbf{III}$']):
        ax.annotate(numeral, (0, 1), xycoords=('figure fraction', 'axes fraction'),   # left of every label
                    fontsize=9, ha='left', va='top')
    for ax_row, xlabel in zip(axs.T, ['$u$', r'lag $/ T_\lambda$', '$i$']):   # every panel of each row
        for ax in ax_row:
            ax.set_xlabel(xlabel, fontsize=8, labelpad=1)
    for ax, ylabel in zip(axs[0], ['$p(u)$', 'ACF', r'$\lambda_i$']):
        ax.set_ylabel(ylabel, fontsize=8, labelpad=2)
    h = [Line2D([], [], color=C_TRUE_DARK, lw=LW_TRUE, marker='o', mfc='none', ms=5)] + \
        [Line2D([], [], color='0.5', ls=ARCH[k]['ls'], lw=1.0, marker=ARCH[k]['marker'], ms=3.5) for k in ARCH]
    fig.legend(h, ['true system'] + [ARCH[k]['label'] for k in ARCH], loc='outside lower center',
               ncol=3, frameon=False, fontsize=7.5, handlelength=2.5, columnspacing=1.0)
    return save(fig, 'esn_verification.pdf', pad_inches=0.02)


if __name__ == '__main__':
    main()
