"""Helpers of tutorial 17, the Gaussian-mixture ensemble filter: a Lorenz-63 integrator vectorized over many states,
the scores of the experiments, and the figures.

Colours: the true state is black; the mirror image, the Gaussian filter and the mixture filter take the validated
categorical slots orange, blue and aqua of the romda figures; the forecast members are dark grey; the attractor is
light grey; the reference posterior is one purple hue, darker for larger weights. In the two-dimensional example, the
exact prior is grey, the exact posterior is black, the priors of the filters take a lighter shade of their colour
than their analyses, and the likelihood is red (`romda.plotting.Palette.OBS`). The figures have panel letters and no
titles.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.patheffects import withStroke
from scipy.stats import multivariate_normal, norm

S = np.array([-1., -1., 1.])                 # the mirror symmetry, (x, y, z) -> (-x, -y, z)
SIGMA, RHO, BETA = 10., 28., 8. / 3.         # parameters of the Lorenz-63 equations

C_TRUE, C_MIRROR, C_FORECAST, C_ATTRACTOR = '#000000', '#eb6834', '#4d4d4d', '#dddddd'
C_FILTER = {1: '#2a78d6', 2: '#1baf7a'}     # Gaussian, Gaussian mixture
C_PRIOR, C_DATA = '#8c8c8c', '#d62728'      # two-dimensional example: the exact prior and the data
C_LIGHT = {1: '#7aaae8', 2: '#5ec59c'}      # the priors of the filters, lighter than their analyses
C_DARK = {1: '#174f99', 2: '#0b7550'}       # the labels of the analyses of the filters
MASSES = (0.99, 0.9, 0.5)                    # probability inside the contours of the two-dimensional example
Z_MASSES = tuple(norm.ppf(0.5 + q / 2) for q in MASSES)    # half-widths, in standard deviations, of the same masses
SHADES = {'Purples': [plt.cm.Purples(v) for v in (0.25, 0.5, 0.8)],      # regions of 99, 90, 50% probability
          **{k: [to_rgba(c, a) for a in (0.15, 0.32, 0.55)] for k, c in C_FILTER.items()}}
NAME = {1: 'Gaussian EnSRKF', 2: 'Gaussian-mixture EnSRKF'}
DASH = (0, (4, 2))                           # the mirror image is dashed
HALF = np.array([1.8, 3.0])                  # half-widths in x and y of the zooms on the modes

plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                     'font.size': 10, 'axes.labelsize': 11})


# ── Lorenz-63 for many states ────────────────────────────────────────────────

def rk4(u, n, h=0.005):
    """`n` fourth-order Runge-Kutta steps of size `h` of the Lorenz-63 states `u` (3, N), vectorized over the states."""
    def rhs(u):
        x, y, z = u
        return np.array([SIGMA * (y - x), x * (RHO - z) - y, x * y - BETA * z])

    for _ in range(n):
        k1 = rhs(u)
        k2 = rhs(u + h / 2 * k1)
        k3 = rhs(u + h / 2 * k2)
        k4 = rhs(u + h * k3)
        u = u + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return u


def sample_attractor(rng, n_traj=8000, n_samples=100):
    """States on the attractor and their mirror images, (3, 2 n_traj n_samples): `n_traj` trajectories after a transient
    of 10 time units, sampled every 0.5 time units."""
    u = rk4(rng.normal(0., 10., (3, n_traj)) + np.array([[0.], [0.], [25.]]), 2000)
    states = []
    for _ in range(n_samples):
        u = rk4(u, 100)
        states.append(u)
    states = np.hstack(states)
    return np.hstack([states, S[:, None] * states])


# ── Scores ───────────────────────────────────────────────────────────────────

def weights(log_w):
    """Normalized importance weights from their logarithms."""
    w = np.exp(log_w - log_w.max())
    return w / w.sum()


def mode_means(samples, log_w):
    """Means of the two modes of a weighted sample: {+1: the mode with x > 0, -1: the mode with x < 0}."""
    w = weights(log_w)
    means = {}
    for s in (1, -1):
        side = np.sign(samples[0]) == s
        means[s] = w[side] @ samples[:, side].T / w[side].sum()
    return means


def distance_to_modes(E, modes):
    """Median distance of the members `E` (3, m) to the mode on their side of x = 0."""
    return float(np.median([np.linalg.norm(E[:, j] - modes[np.sign(E[0, j])]) for j in range(E.shape[1])]))


def twin_scores(filt, t_true, u_true, t_last, window, scale):
    """Scores of a twin experiment: the times, the members (Nt, 3, m) and the true state; the error of the mean on z
    over `window` before the last observation, `t_last`, normalized by `scale`; and the fraction of the members that are
    nearer the mirror image of the true state than the true state, in time and at `t_last`."""
    H, t = np.asarray(filt.model.hist)[:, :3, :], np.asarray(filt.model.hist_t)
    ref = np.array([np.interp(t, t_true, u_true[:, j]) for j in range(3)]).T
    last = (t > t_last - window) & (t <= t_last)
    e_z = float(np.sqrt(np.mean((H[last, 2].mean(-1) - ref[last, 2]) ** 2)) / scale)
    mirror = (np.linalg.norm(H - (S * ref)[..., None], axis=1) < np.linalg.norm(H - ref[..., None], axis=1)).mean(-1)
    return dict(t=t, H=H, ref=ref, e_z=e_z, mirror=mirror, mirror_last=float(mirror[np.argmin(np.abs(t - t_last))]))


def mixture_pdf(X, w, mu, C):
    """Probability density of the Gaussian mixture sum_k w_k N(mu_k, C_k) at the points `X` (..., N)."""
    return sum(wk * multivariate_normal(mk, Ck).pdf(X) for wk, mk, Ck in zip(w, mu, C))


def hpd_level(p, mass):
    """Density of a grid `p` above which the grid holds the fraction `mass` of the probability, i.e., the contour of
    the highest-density region with that probability."""
    v = np.sort(p.ravel())[::-1]
    return v[min(np.searchsorted(np.cumsum(v) / v.sum(), mass), v.size - 1)]


# ── Figures ──────────────────────────────────────────────────────────────────

def _letter(ax, s):
    ax.text(-0.02, 1.02, f'({s})', transform=ax.transAxes, ha='right', va='bottom', fontweight='bold')


def _stars(ax, u_true):
    """The true state and its mirror image in the x-y plane."""
    for p, c in ((u_true, C_TRUE), (S * u_true, C_MIRROR)):
        ax.plot(p[0], p[1], '*', color=c, ms=15, mec='w', mew=0.8, zorder=6)


def _posterior(ax, samples, log_w, s=4):
    """The weighted samples that carry weight, darker for larger weights, the heaviest on top."""
    w = weights(log_w)
    keep = np.flatnonzero(w > 1e-4 * w.max())
    keep = keep[np.argsort(w[keep])]
    ax.scatter(samples[0, keep], samples[1, keep], c=w[keep] / w.max(), cmap='Purples', norm=PowerNorm(0.4, 0, 1),
               s=s, lw=0, zorder=1)


def _attractor(ax, attractor):
    ax.scatter(attractor[0], attractor[1], s=0.3, color=C_ATTRACTOR, lw=0, zorder=0, rasterized=True)


def _zoom(ax, centre):
    lo = centre[:2] - HALF
    ax.set(xlim=(lo[0], lo[0] + 2 * HALF[0]), ylim=(lo[1], lo[1] + 2 * HALF[1]))
    return lo


def _box(ax, centre, label=None):
    lo = centre[:2] - HALF
    ax.add_patch(Rectangle(lo, *(2 * HALF), fill=False, lw=0.8, color='0.3', zorder=4))
    if label:
        ax.text(lo[0] + 2 * HALF[0] + 0.8, centre[1], label, va='center', fontsize=9)


def _legend(fig, handles, labels, **kw):
    fig.legend(handles, labels, loc='outside upper center', **{'ncol': len(handles), **kw})


def _star_handles():
    return [Line2D([], [], ls='', marker='*', ms=13, color=c, mec='w') for c in (C_TRUE, C_MIRROR)]


STAR_LABELS = [r'$\boldsymbol{\phi}^\dagger$', r'$\mathcal{S}\boldsymbol{\phi}^\dagger$']
DASHED = [(0, (4, 2))]                       # dashed contour lines: the priors of the filters


def _regions(ax, phi1, phi2, p, shades='Purples'):
    """The regions of the grid density `p` with 99, 90 and 50% of the probability, darker inside, in the `shades` of
    `SHADES`."""
    ax.contourf(phi1, phi2, p, levels=[hpd_level(p, q) for q in MASSES] + [1.01 * p.max()], colors=SHADES[shades],
                zorder=0)


def _contours(ax, phi1, phi2, p, color, linestyles='solid'):
    ax.contour(phi1, phi2, p, levels=[hpd_level(p, q) for q in MASSES], colors=color, linestyles=linestyles,
               linewidths=1.1, zorder=2)


def _nearest(phi1, phi2, p, xy, mass=0.9):
    """The grid point of the region of `p` with probability `mass` nearest to `xy`, i.e., a point of its contour when
    `xy` lies outside the region."""
    P1, P2 = np.meshgrid(phi1, phi2)
    dist = np.where(p >= hpd_level(p, mass), np.hypot(P1 - xy[0], P2 - xy[1]), np.inf)
    j = np.argmin(dist)
    return P1.flat[j], P2.flat[j]


def _label(ax, text, xy_text, target, color, ha='center'):
    """`text` at `xy_text`, with an arrow to `target` (data coordinates)."""
    ax.annotate(text, xy=target, xytext=xy_text, color=color, fontsize=9.5, ha=ha, va='center', zorder=6,
                path_effects=[withStroke(linewidth=3, foreground='w')],
                arrowprops=dict(arrowstyle='->', color=color, lw=0.9, shrinkA=3, shrinkB=1))


def _label_density(ax, phi1, phi2, p, text, frac, color, ha='center'):
    """`text` at the fraction `frac` of the axes, with an arrow to the contour of `p` that holds 90% of the
    probability."""
    xy = (phi1[0] + frac[0] * (phi1[-1] - phi1[0]), phi2[0] + frac[1] * (phi2[-1] - phi2[0]))
    _label(ax, text, xy, _nearest(phi1, phi2, p, xy), color, ha)


PRIOR, POST = r'$p(\phi_1, \phi_2)$', r'$p(\phi_1, \phi_2 \mid d)$'


def plot_bayes_2d(phi1, phi2, prior, post, d, Cdd, gaussian_f, gaussian_a, mixture_f, mixture_a):
    """Bayes' rule in the phi1-phi2 plane, on the grid (`phi1`, `phi2`), with the exact prior `prior` (grey lines, in
    a) and posterior `post` (black dashed lines): (a) the likelihood of the data `d` with variance `Cdd`, observed in
    phi2 (bands of 50, 90 and 99% in phi2); (b) the
    Gaussian with the mean and covariance of the prior, `gaussian_f` = (mean, cov) (lines), and its Kalman analysis,
    `gaussian_a` (regions), whose mean is the cross; (c) the prior mixture, `mixture_f` = (weights, means, covs)
    (lines), and its analysis, `mixture_a` (regions), with the weights. (d-f) The densities of phi1 alone, below each
    panel, with the labels of the densities. The lines and the regions hold 50, 90 and 99% of the probability."""
    X = np.stack(np.meshgrid(phi1, phi2), axis=-1)
    filters = {1: (multivariate_normal(*gaussian_f).pdf(X), multivariate_normal(*gaussian_a).pdf(X)),
               2: (mixture_pdf(X, *mixture_f), mixture_pdf(X, *mixture_a))}     # (prior, analysis) on the grid

    def marginal(p):                                                              # the density of phi1 alone
        return p.sum(0) * (phi2[1] - phi2[0])

    fig, axs = plt.subplots(2, 3, figsize=(11, 5.8), layout='constrained', sharex=True,
                            gridspec_kw=dict(height_ratios=(1.5, 1)))
    s = np.sqrt(Cdd[0, 0])
    for z, alpha in zip(Z_MASSES, (0.08, 0.16, 0.3)):        # the likelihood: 99, 90 and 50% of it in phi2
        axs[0, 0].axhspan(d[0] - z * s, d[0] + z * s, color=C_DATA, alpha=alpha, lw=0, zorder=1)
    axs[0, 0].axhline(d[0], color=C_DATA, lw=0.8, zorder=1)                      # the data
    _contours(axs[0, 0], phi1, phi2, prior, C_PRIOR)
    axs[1, 0].plot(phi1, marginal(prior), color=C_PRIOR, lw=1.3)
    for k, (p_f, p_a) in filters.items():                                        # column k: filter k
        _regions(axs[0, k], phi1, phi2, p_a, k)
        _contours(axs[0, k], phi1, phi2, p_f, C_LIGHT[k])
        axs[1, k].fill_between(phi1, marginal(p_a), color=SHADES[k][1], lw=0)
        axs[1, k].plot(phi1, marginal(p_f), color=C_LIGHT[k], lw=1.3)
    for j in range(3):
        _contours(axs[0, j], phi1, phi2, post, C_TRUE, DASHED)
        axs[1, j].plot(phi1, marginal(post), color=C_TRUE, lw=1.3, ls=DASH)
        axs[0, j].set(ylabel=r'$\phi_2$' if j == 0 else None, ylim=(phi2[0], phi2[-1]))
        axs[1, j].set(xlabel=r'$\phi_1$', ylabel='probability density' if j == 0 else None, xlim=(phi1[0], phi1[-1]),
                      ylim=(0, 1.1 * marginal(post).max()))
        for ax in axs[:, j]:
            ax.tick_params(labelleft=j == 0)
    axs[0, 1].plot(*gaussian_a[0], 'x', color=C_FILTER[1], ms=8, mew=2, zorder=3,
                   path_effects=[withStroke(linewidth=4, foreground='w')])                # the analysis mean
    w_f, w_a, means = mixture_f[0], mixture_a[0], mixture_f[1]
    for k in range(len(w_f)):                                    # the weights, beside each Gaussian
        right = means[k][0] > 0
        axs[0, 2].text(0.98 if right else 0.03, 0.98, f'$w^\\mathrm{{f}}_{k + 1} = {w_f[k]:.2f}$\n'
                       f'$w^\\mathrm{{a}}_{k + 1} = {w_a[k]:.2f}$', ha='right' if right else 'left', va='top',
                       fontsize=9, transform=axs[0, 2].transAxes)
    # The labels of the densities of phi1: the prior on the left, the posterior or the analysis on the right, at the
    # fraction `at` of the axis, on a curve (height 1) or inside a filled density (height 0.5)
    gauss = lambda s: rf'$\mathcal{{N}}(\bar{{\phi}}^\mathrm{{{s}}}_1, C^\mathrm{{{s}}}_{{11}})$'      # noqa: E731
    mix = lambda s: rf'$\Sigma_k\, w^\mathrm{{{s}}}_k\,\mathcal{{N}}^\mathrm{{{s}}}_k$'                # noqa: E731
    labels = (((prior, r'$p(\phi_1)$', 0.23, C_PRIOR, 1.), (post, r'$p(\phi_1 \mid d)$', 0.75, C_TRUE, 1.)),
              ((filters[1][0], gauss('f'), 0.17, C_LIGHT[1], 1.), (filters[1][1], gauss('a'), 0.8, C_DARK[1], 0.5)),
              ((filters[2][0], mix('f'), 0.23, C_LIGHT[2], 1.), (filters[2][1], mix('a'), 0.76, C_DARK[2], 0.5)))
    for ax, pair in zip(axs[1], labels):
        for (p, text, at, color, height), frac in zip(pair, (0.03, 0.97)):
            i = int(at * (len(phi1) - 1))
            xy = (phi1[0] + frac * (phi1[-1] - phi1[0]), 0.85 * ax.get_ylim()[1])
            _label(ax, text, xy, (phi1[i], height * marginal(p)[i]), color, 'left' if frac < 0.5 else 'right')
    for ax, s in zip(axs.flat, 'abcdef'):
        _letter(ax, s)
    handles = [Line2D([], [], color=C_PRIOR, lw=1.3), Line2D([], [], color=C_TRUE, lw=1.3, ls=DASH)]
    for k in (1, 2):
        handles += [Line2D([], [], color=C_LIGHT[k], lw=1.3), Patch(color=SHADES[k][1])]
    handles.append(Patch(color=C_DATA, alpha=0.3, lw=0))
    _legend(fig, handles, ['exact prior', "exact posterior (Bayes' rule)", 'Gaussian prior', 'Gaussian analysis',
                           'mixture prior', 'mixture analysis', 'likelihood'], ncol=4)
    plt.show()


def plot_members_2d(phi1, phi2, prior, post, rows):
    """Members in the phi1-phi2 plane, one row per ensemble, `rows` = [(Ef, labels, analyses, outside)], with the
    exact prior as grey lines and the exact posterior as black dashed lines: (a, d) the forecast `Ef` (2, m), marked by
    the Gaussian `labels` of the fit, with the prior `prior`; (b, c, e, f) the analyses {mixture: (2, m)} with the
    posterior `post`, and the number of members outside the region with 99% of the posterior probability, `outside`
    {mixture: int}. Along the bottom of every panel, on one scale: the histogram of phi1 of the members (filled), and
    the density of phi1 of the prior (a, d) or of the posterior (dashed; b, c, e, f). The densities are labelled in
    the last row. The lines hold 50, 90 and 99% of the probability."""
    fig, axs = plt.subplots(len(rows), 3, figsize=(11, 0.6 + 3.3 * len(rows)), layout='constrained', sharex=True,
                            sharey=True, squeeze=False)

    def marginal(p):                                                     # the density of phi1 alone
        return p.sum(0) * (phi2[1] - phi2[0])

    scale = 0.3 * (phi2[-1] - phi2[0]) / marginal(post).max()           # densities of phi1 up to 30% of the height
    edges = np.linspace(phi1[0], phi1[-1], 21)                           # bins of the histograms

    def histogram(ax, x, face, edge):                                    # of the members, as a density
        ax.hist(x, bins=edges, weights=np.full(len(x), scale / (len(x) * (edges[1] - edges[0]))), bottom=phi2[0],
                histtype='stepfilled', facecolor=face, edgecolor=edge, lw=0.8, zorder=1)

    for i, (row, (Ef, labels, analyses, outside)) in enumerate(zip(axs, rows)):
        small = Ef.shape[1] <= 40                                        # markers: smaller for many members
        _contours(row[0], phi1, phi2, prior, C_PRIOR)
        for k, marker in enumerate('os'):
            row[0].plot(*Ef[:, labels == k], marker, mfc='none', mec=C_FORECAST, mew=1.3 if small else 0.8,
                        ms=6.5 if small else 3.5, zorder=3)
        row[0].text(1., 1.02, f'$m = {Ef.shape[1]}$', fontsize=9, ha='right', va='bottom', transform=row[0].transAxes)
        histogram(row[0], Ef[0], to_rgba(C_FORECAST, 0.25), C_FORECAST)
        row[0].plot(phi1, phi2[0] + scale * marginal(prior), color=C_PRIOR, lw=1.1, zorder=2)
        for ax, mixture in zip(row[1:], (1, 2)):
            E = analyses[mixture]
            _contours(ax, phi1, phi2, post, C_TRUE, DASHED)
            histogram(ax, E[0], SHADES[mixture][1], C_DARK[mixture])
            ax.plot(phi1, phi2[0] + scale * marginal(post), color=C_TRUE, lw=1.1, ls=DASH, zorder=2)
            ax.plot(*E, 'o', color=C_FILTER[mixture], mec='w', mew=0.6 if small else 0.3, ms=7 if small else 3.5,
                    zorder=3)
            ax.text(1., 1.02, f'{outside[mixture]} of {E.shape[1]} members outside the 99% region', fontsize=9,
                    ha='right', va='bottom', transform=ax.transAxes)    # above the panel, beside its letter
        for j, ax in enumerate(row):
            ax.set(xlabel=r'$\phi_1$' if i == len(rows) - 1 else None, ylabel=r'$\phi_2$' if j == 0 else None,
                   xlim=(phi1[0], phi1[-1]), ylim=(phi2[0], phi2[-1]))
            _letter(ax, 'abcdefghi'[3 * i + j])
    row = axs[-1]                                                        # the labels, in the last row
    _label_density(row[0], phi1, phi2, prior, PRIOR, (0.98, 0.93), C_PRIOR, 'right')
    for ax in row[1:]:
        _label_density(ax, phi1, phi2, post, POST, (0.98, 0.93), C_TRUE, 'right')
    handles = [Line2D([], [], ls='', marker=mk, mfc='none', mec=C_FORECAST, mew=1.3, ms=6.5) for mk in 'os'] + \
              [Line2D([], [], ls='', marker='o', color=C_FILTER[k], mec='w', ms=7) for k in (1, 2)] + \
              [Line2D([], [], color=C_PRIOR, lw=1.1), Line2D([], [], color=C_TRUE, lw=1.1, ls=DASH)]
    _legend(fig, handles, ['forecast, Gaussian 1 of the fit', 'forecast, Gaussian 2 of the fit', 'Gaussian analysis',
                           'mixture analysis', 'exact prior', 'exact posterior'], ncol=3)
    plt.show()


def plot_symmetry(t, u_true, u_mirror, shade=(3, 5), t_max=10):
    """(a) x and (b) z of the true trajectory and of its mirror image against time, `t` in Lyapunov times; (c) both in
    the x-z plane over the interval `shade`, and the two states at its end, at the same z."""
    fig = plt.figure(figsize=(10, 4), layout='constrained')
    gs = fig.add_gridspec(2, 2, width_ratios=(2.3, 1))
    ax_x, ax_p = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[:, 1])
    ax_z = fig.add_subplot(gs[1, 0], sharex=ax_x)
    for ax, j in ((ax_x, 0), (ax_z, 2)):
        ax.axvspan(*shade, color='0.95', zorder=0)
        ax.plot(t, u_true[:, j], color=C_TRUE, lw=1.3)
        ax.plot(t, u_mirror[:, j], color=C_MIRROR, lw=1.3, ls=DASH)
    ax_x.set(ylabel='$x$')
    ax_x.tick_params(labelbottom=False)
    ax_z.set(ylabel='$z$', xlabel=r'$t/T_\lambda$', xlim=(0, t_max))
    seg = (t >= shade[0]) & (t <= shade[1])
    end = np.flatnonzero(seg)[-1]
    ax_p.plot(u_true[:, 0], u_true[:, 2], color=C_ATTRACTOR, lw=0.4, zorder=0)
    ax_p.plot(u_true[seg, 0], u_true[seg, 2], color=C_TRUE, lw=1.3)
    ax_p.plot(u_mirror[seg, 0], u_mirror[seg, 2], color=C_MIRROR, lw=1.3, ls=DASH)
    ax_p.plot([u_true[end, 0], u_mirror[end, 0]], [u_true[end, 2]] * 2, ':', color='0.4', lw=1)
    for u, c in ((u_true, C_TRUE), (u_mirror, C_MIRROR)):
        ax_p.plot(u[end, 0], u[end, 2], 'o', color=c, ms=7, mec='w', zorder=5)
    ax_p.set(xlabel='$x$', ylabel='$z$')
    for ax, s in ((ax_x, 'a'), (ax_z, 'b'), (ax_p, 'c')):
        _letter(ax, s)
    _legend(fig, [Line2D([], [], color=C_TRUE, lw=1.3), Line2D([], [], color=C_MIRROR, lw=1.3, ls=DASH)],
            [r'true state, $\boldsymbol{\phi}^\dagger$', r'mirror image, $\mathcal{S}\boldsymbol{\phi}^\dagger$'])
    plt.show()


def plot_posterior(post_last, post_all, u_true, attractor):
    """Posterior in the x-y plane given (a) the last observation, `post_last`, and (b) all the observations,
    `post_all`, both (samples, log-weights), over the attractor; (c, d) zooms on the two modes of (b), with the
    probability of each."""
    fig = plt.figure(figsize=(11, 4.8), layout='constrained')
    gs = fig.add_gridspec(2, 3, width_ratios=(1, 1, 0.62))
    ax_a, ax_b = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[:, 1])
    for ax, post in ((ax_a, post_last), (ax_b, post_all)):
        _attractor(ax, attractor)
        _posterior(ax, *post)
        _stars(ax, u_true)
        ax.set(xlabel='$x$', ylabel='$y$', xlim=(-22, 22), ylim=(-30, 30))
    samples, log_w = post_all
    w = weights(log_w)
    for row, (sy, s) in enumerate(((1, 'c'), (-1, 'd'))):            # the modes in the upper and lower half of (b)
        ax = fig.add_subplot(gs[row, 2])
        side = np.sign(samples[1]) == sy
        centre = w[side] @ samples[:, side].T / w[side].sum()
        _box(ax_b, centre, f'({s})')
        _zoom(ax, centre)
        _posterior(ax, samples, log_w, s=10)
        _stars(ax, u_true)
        ax.text(0.04, 0.95, f'probability {w[side].sum():.2f}', transform=ax.transAxes, va='top', fontsize=9)
        ax.set(xlabel='$x$', ylabel='$y$')
        _letter(ax, s)
    _letter(ax_a, 'a')
    _letter(ax_b, 'b')
    _legend(fig, _star_handles() + [Line2D([], [], ls='', marker='o', ms=5, color=plt.cm.Purples(0.75)),
                                    Line2D([], [], ls='', marker='o', ms=4, color=C_ATTRACTOR)],
            STAR_LABELS + ['posterior samples (darker: larger weight)', 'attractor'])
    plt.show()


def plot_analysis(Ef, analyses, post_all, u_true, attractor):
    """The forecast members `Ef` (3, m) and the analyses {mixture: (3, m)} in the x-y plane, over the reference
    posterior `post_all` (samples, log-weights): (a-c) over the attractor, (d-i) zoomed on the two modes (the boxes)."""
    samples, log_w = post_all
    modes = mode_means(samples, log_w)
    centres = (None, modes[np.sign(-u_true[0])], modes[np.sign(u_true[0])])     # full view, mirror mode, true mode
    fig, axs = plt.subplots(3, 3, figsize=(10.5, 10), layout='constrained',
                            gridspec_kw=dict(height_ratios=(1.25, 1, 1)))
    for i, centre in enumerate(centres):
        for j, (E, mixture) in enumerate(((Ef, None), (analyses[1], 1), (analyses[2], 2))):
            ax = axs[i, j]
            if centre is None:
                _attractor(ax, attractor)
                ax.set(xlim=(-22, 22), ylim=(-30, 30))
                for c in modes.values():
                    _box(ax, c)
            else:
                lo = _zoom(ax, centre)
                inside = np.sum(np.all((E[:2] >= lo[:, None]) & (E[:2] <= lo[:, None] + 2 * HALF[:, None]), axis=0))
                ax.text(0.04, 0.95, f'{inside} of {E.shape[1]} members', transform=ax.transAxes, va='top', fontsize=9)
            _posterior(ax, samples, log_w, s=6 if centre is None else 10)
            if mixture is None:
                ax.plot(E[0], E[1], 'o', mfc='none', mec=C_FORECAST, mew=1.4, ms=7, zorder=3)
            else:                                                     # the forecast, faint, for reference
                ax.plot(Ef[0], Ef[1], 'o', mfc='none', mec=C_FORECAST, mew=0.8, ms=6, alpha=0.4, zorder=2)
                ax.plot(E[0], E[1], 'o', color=C_FILTER[mixture], mec='w', mew=0.6, ms=7, zorder=3)
            _stars(ax, u_true)
            ax.set(xlabel='$x$' if i == 2 else None, ylabel='$y$' if j == 0 else None)
            _letter(ax, 'abcdefghi'[3 * i + j])
    handles = [Line2D([], [], ls='', marker='o', mfc='none', mec=C_FORECAST, mew=1.4, ms=7)] + \
              [Line2D([], [], ls='', marker='o', color=C_FILTER[k], mec='w', ms=7) for k in (1, 2)] + \
              _star_handles() + [Line2D([], [], ls='', marker='o', ms=5, color=plt.cm.Purples(0.75))]
    _legend(fig, handles, ['forecast members', 'Gaussian analysis', 'mixture analysis'] + STAR_LABELS +
            ['reference posterior'], handletextpad=0.3, columnspacing=1.2)
    plt.show()


def plot_twin(scores, t_data, t_lyap):
    """Twin experiment, {mixture: twin_scores}: x of the members of (a) the Gaussian and (b) the mixture filter, with
    the true state and its mirror image; (c) the fraction of the members nearer the mirror image. The data span
    `t_data` (shaded)."""
    t0, t1 = (t / t_lyap for t in t_data)
    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True, layout='constrained',
                            gridspec_kw=dict(height_ratios=(1, 1, 0.75)))
    for ax, mixture in zip(axs[:2], (1, 2)):
        sc = scores[mixture]
        t = sc['t'] / t_lyap
        ax.plot(t, sc['H'][:, 0], color=C_FILTER[mixture], lw=0.5, alpha=0.45)
        ax.plot(t, sc['ref'][:, 0], color=C_TRUE, lw=1.2)
        ax.plot(t, -sc['ref'][:, 0], color=C_MIRROR, lw=1.2, ls=DASH)
        ax.set(ylabel='$x$', ylim=(-28, 28))
        axs[2].step(t, sc['mirror'], where='post', color=C_FILTER[mixture], lw=1.4)
    axs[2].set(xlabel=r'$t/T_\lambda$', ylabel='fraction of members\n' + r'nearer $\mathcal{S}\boldsymbol{\phi}^\dagger$',
               ylim=(-0.05, 1.05), yticks=(0, 0.5, 1))
    for ax, s in zip(axs, 'abc'):
        ax.axvspan(t0, t1, color='0.95', zorder=0)
        _letter(ax, s)
    _legend(fig, [Line2D([], [], color=C_TRUE, lw=1.2), Line2D([], [], color=C_MIRROR, lw=1.2, ls=DASH)] +
            [Line2D([], [], color=C_FILTER[k], lw=1.4) for k in (1, 2)],
            [r'$x^\dagger$', r'$-x^\dagger$ (mirror image)', NAME[1], NAME[2]])
    plt.show()
