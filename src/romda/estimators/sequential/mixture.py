r"""Gaussian-mixture analysis of the ensemble filters: one analysis per mode of the forecast ensemble.

When the forecast probability density function has several modes, a single Gaussian analysis moves every member
towards the mean of the ensemble, which lies between the modes. For example, observations of $z$ alone do not
distinguish the Lorenz-63 state $(x, y, z)$ from its mirror image $(-x, -y, z)$, so that $p(x, y \mid z)$ has two
modes. With ``mixture`` $= K_\mathrm{max} > 1$, `EnsembleFilter.analysis_step`

1. fits a mixture of $K \le K_\mathrm{max}$ Gaussians with one diagonal covariance to the physical rows of the forecast
   ensemble by expectation maximization, chooses $K$ by the Bayesian information criterion (BIC; Sondergaard &
   Lermusiaux 2013), and assigns each member to its most likely component (Smith 2007). One more component must
   decrease the BIC by more than 10, i.e. very strong evidence (Kass & Raftery 1995): with tens of members, a sample
   of one Gaussian has gaps by chance, and without the margin the BIC splits 2 to 20% of such samples (10 to 30
   members, three rows), against 1% or fewer with it. The rows are scaled by their climatological standard
   deviation (``climatology``), and the variance of a component is at least $10^{-2}$ in these units: a synchronized
   ensemble, narrower than a tenth of the climatological standard deviation, is one component. Scaled by the spread
   of the ensemble instead, the fit splits a narrow ensemble along its curvature, because the BIC does not depend
   on the scale of the rows, and every split halves the members that estimate each covariance;
2. updates each component with the analysis of the filter (``type``, inflation, rejection), from the members of that
   component only;
3. updates the weight of each component with its evidence (the Gaussian-sum filter of Alspach & Sorenson 1972),

   $$w^a_k \propto w^f_k\,\mathcal{N}\big(\vect{d};\ \matr{M}\bar{\vect{\psi}}^f_k,\
     \matr{M}\matr{C}^f_k\matr{M}^\T + \matr{C}_{dd}\big),$$

   in which the forecast weight $w^f_k = m_k/m$ is the fraction of the members in component $k$;
4. redistributes the members among the components in proportion to $w^a_k$, so that every member keeps the weight
   $1/m$. A component that loses members drops them at random; a component that gains members draws them from the
   Gaussian of its analysis, in the span of its deviations. Members keep their columns where possible, so that the
   history of a member stays continuous.

A component needs at least ``mixture_min_members`` members: a fit with a smaller component is discarded, and a
component whose weight would give it fewer members is emptied. With $K = 1$ the analysis is the Gaussian one.

References
----------
Alspach & Sorenson (1972). Nonlinear Bayesian estimation using Gaussian sum approximations.
*IEEE Trans. Autom. Control*, 17(4), 439-448.

Bengtsson, Snyder & Nychka (2003). Toward a nonlinear ensemble filter for high-dimensional systems.
*J. Geophys. Res.*, 108(D24), 8775. [DOI: 10.1029/2002JD002900](https://doi.org/10.1029/2002JD002900).

Kass & Raftery (1995). Bayes factors. *J. Am. Stat. Assoc.*, 90(430), 773-795.

Smith (2007). Cluster ensemble Kalman filter. *Tellus A*, 59(5), 749-757.
[DOI: 10.1111/j.1600-0870.2007.00246.x](https://doi.org/10.1111/j.1600-0870.2007.00246.x).

Sondergaard & Lermusiaux (2013). Data assimilation with Gaussian mixture models using the dynamically orthogonal
field equations. Part I: Theory and scheme. *Mon. Weather Rev.*, 141(6), 1737-1760.
[DOI: 10.1175/MWR-D-11-00295.1](https://doi.org/10.1175/MWR-D-11-00295.1).
"""

from collections import namedtuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp

#: Record of the mixture analyses: per analysis, the time, the component of each forecast member, the forecast and
#: analysis weights of the components, and the number of members of each component after the redistribution.
MixtureData = namedtuple('MixtureData', ['times', 'labels', 'weights_f', 'weights_a', 'counts'])

VAR_FLOOR = 1e-2     # lowest variance of the components, in units of the (climatological) variance of the row
BIC_MARGIN = 10.     # decrease of the BIC that one more component must give: very strong evidence (Kass & Raftery 1995)
N_INIT = 5           # starts of the expectation maximization, from different members; the best fit is kept
N_ITER = 200


def _em(Z, K, rng, n_iter=N_ITER, tol=1e-8):
    """Log-likelihood, responsibilities (K, m), means (K, p) and variances (p,) of the fit of K Gaussians with one
    diagonal covariance to the samples Z (m, p). The shared covariance keeps the fit from collapsing a component onto
    a few members, which separate variances do with ensembles of tens of members."""
    m, p = Z.shape
    mu = Z[rng.choice(m, K, replace=False)]
    var = np.ones(p)
    log_w = np.full(K, -np.log(K))
    ll_old = -np.inf
    for _ in range(n_iter):
        log_p = log_w[:, None] - 0.5 * (np.log(2 * np.pi * var).sum()
                                         + (((Z[None] - mu[:, None]) ** 2) / var).sum(-1))         # (K, m)
        ll_i = logsumexp(log_p, axis=0)
        R = np.exp(log_p - ll_i)
        ll = float(ll_i.sum())
        n_k = R.sum(1)
        if np.any(n_k < 1e-8):                       # an empty component: the fit has K - 1 components
            return -np.inf, R, mu, var
        log_w = np.log(n_k / m)
        mu = R @ Z / n_k[:, None]
        var = np.maximum(np.einsum('km,kmp->p', R, (Z[None] - mu[:, None]) ** 2) / m, VAR_FLOOR)
        if ll - ll_old <= tol * abs(ll):
            break
        ll_old = ll
    return ll, R, mu, var


def fit_mixture(X, max_components, rng, min_members=3, scale=None):
    """Component of each member, from the mixture of at most `max_components` Gaussians with one diagonal covariance
    and the lowest BIC.

    Parameters
    ----------
    X : ndarray (p, m)
        Features of the m members (e.g. the physical rows of the forecast ensemble).
    max_components : int
    rng : numpy Generator
        Starts of the expectation maximization.
    min_members : int
        A fit in which a component has fewer members is discarded.
    scale : ndarray (p,), optional
        Scale of each row (e.g. its climatological standard deviation), in whose units the variances are at least
        `VAR_FLOOR`. None: the standard deviation of the row over the members.

    Returns
    -------
    labels : ndarray (m,) of int
        Component of each member, numbered from 0 in the order of their first member.
    bic : ndarray (max_components,)
        BIC of the fit with each number of components (inf where discarded); K components replace fewer when they
        decrease the BIC by more than `BIC_MARGIN`.
    """
    X = np.asarray(X, dtype=float)
    m = X.shape[1]
    scale = X.std(axis=1) if scale is None else np.asarray(scale, dtype=float)
    rows = scale > 0
    bic = np.full(max_components, np.inf)
    labels = np.zeros(m, dtype=int)
    if not rows.any():
        return labels, bic
    Z = ((X[rows] - X[rows].mean(1, keepdims=True)) / scale[rows, None]).T     # (m, p)
    p = Z.shape[1]
    best = 0
    for K in range(1, max_components + 1):
        if K > 1 and K * min_members > m:
            break
        fits = [_em(Z, K, rng) for _ in range(N_INIT if K > 1 else 1)]
        ll, R, _, _ = max(fits, key=lambda f: f[0])
        lab = R.argmax(0)
        if not np.isfinite(ll) or (K > 1 and np.bincount(lab, minlength=K).min() < min_members):
            continue
        bic[K - 1] = -2 * ll + (p * K + p + K - 1) * np.log(m)
        if K > 1 and bic[K - 1] < bic[best] - BIC_MARGIN:
            best, labels = K - 1, lab
    _, labels = np.unique(labels, return_inverse=True)
    order = np.argsort([np.flatnonzero(labels == k)[0] for k in range(labels.max() + 1)])
    return np.argsort(order)[labels], bic


def log_evidence(S, z, Cdd):
    """log N(z; 0, S S^T / (n - 1) + C_dd) of the innovation z (Nq, 1) of a component with mapped deviations
    S (Nq, n); the component's prediction of the data is Gaussian with this covariance."""
    n = S.shape[1]
    C = S @ S.T / max(n - 1, 1) + Cdd
    c, low = cho_factor(C, lower=True)
    a = z.ravel()
    return float(-0.5 * a @ cho_solve((c, low), a) - np.log(np.diag(c)).sum() - 0.5 * len(a) * np.log(2 * np.pi))


def allocate(weights, m, min_members):
    """Members of each component: m w_k, rounded by the largest remainders, after emptying, one at a time, the
    component of lowest weight while any non-empty component would have fewer than `min_members`."""
    w = np.asarray(weights, dtype=float).copy()
    while True:
        raw = m * w / w.sum()
        n = np.floor(raw).astype(int)
        n[np.argsort(n - raw)[:m - n.sum()]] += 1
        small = (n > 0) & (n < min_members)
        if not small.any() or (w > 0).sum() == 1:
            return n
        w[np.flatnonzero(small)[np.argmin(w[small])]] = 0.


def redistribute(parts, columns, counts, rng):
    """The analysis ensemble (N, m) from the analysis of each component, `parts[k]` (N, m_k) in the columns
    `columns[k]`, with `counts[k]` members per component. A component that loses members keeps a random subset in
    its own columns; a component that gains members fills the freed columns with draws mean + Psi' xi /
    sqrt(m_k - 1), xi ~ N(0, I), which have the mean and covariance of the component's analysis in expectation."""
    N = parts[0].shape[0]
    out = np.empty((N, sum(len(c) for c in columns)))
    free = []
    for A, cols, n in zip(parts, columns, counts):
        keep = np.sort(rng.choice(len(cols), size=min(n, len(cols)), replace=False))
        out[:, cols[keep]] = A[:, keep]
        free.extend(np.setdiff1d(cols, cols[keep]))
    free = np.array(sorted(free), dtype=int)
    i = 0
    for A, cols, n in zip(parts, columns, counts):
        extra = n - len(cols)
        if extra <= 0:
            continue
        mean = A.mean(1, keepdims=True)
        xi = rng.standard_normal((A.shape[1], extra))
        out[:, free[i:i + extra]] = mean + (A - mean) @ xi / np.sqrt(max(A.shape[1] - 1, 1))
        i += extra
    return out


def mixture_update(est, Af_aug, Af_fc, labels, d, Cdd, Af_params):
    """Gaussian-mixture analysis (steps 2-4 of the module) of the forecast `Af_aug` (N, m), whose members belong to
    the components `labels`; `Af_fc` is the forecast without the prior inflation (the rejection compares with it)."""
    d1, C1 = np.atleast_1d(d), np.atleast_2d(Cdd)
    m = Af_aug.shape[1]
    columns = [np.flatnonzero(labels == k) for k in range(labels.max() + 1)]
    parts, log_w = [], []
    for cols in columns:
        A = Af_aug[:, cols]
        mean = A.mean(1, keepdims=True)
        S, z, C = est._stacked_innovation(est._MA(A - mean), est._MA(mean), d1, C1)
        log_w.append(np.log(len(cols) / m) + log_evidence(S, z, C))
        parts.append(est._gaussian_update(A, Af_fc[:, cols], d, Cdd, Af_params))
    log_w = np.array(log_w)
    w_a = np.exp(log_w - logsumexp(log_w))
    counts = allocate(w_a, m, est.mixture_min_members)
    est.mixture_history.weights_f.append(np.array([len(c) / m for c in columns]))
    est.mixture_history.weights_a.append(w_a)
    est.mixture_history.counts.append(counts)
    return redistribute(parts, columns, counts, est.rng)
