r"""Covariance localization for ensemble filters.

A finite ensemble produces spurious long-range correlations between state
components and predicted measurements. Localization damps them by a Schur
(Hadamard, i.e. element-wise) product of the sample covariances with a compactly supported
correlation matrix,

$$
\matr{C}_{\psi q} \leftarrow \matr{L}_{\psi q} \circ \matr{C}_{\psi q},
\qquad
\matr{C}_{qq} \leftarrow \matr{L}_{qq} \circ \matr{C}_{qq},
$$

where each entry of $\matr{L}$ is the Gaspari-Cohn function of the distance
between the two components. The module mirrors ``biasda.localization``, whose
estimators apply these matrices when configured with ``localization``.
In romda, ``EnsembleFilter(type='sqrt', localization=...)`` is the
local ensemble transform Kalman filter (LETKF; Hunt et al. 2007), which tapers the
inverse of $\matr{C}_{dd}$ row by row instead (see `romda.estimators.sequential.ensemble`).

References
----------
Hunt, Kostelich & Szunyogh (2007). Efficient data assimilation for spatiotemporal
chaos: a local ensemble transform Kalman filter. *Physica D*, 230, 112-126.

Gaspari & Cohn (1999). Construction of correlation functions in two and three
dimensions. *Q. J. R. Meteorol. Soc.*, 125, 723-757.

Evensen, Vossepoel & van Leeuwen (2022). *Data Assimilation Fundamentals.*
Springer, Chap. 10 (localization and inflation).
"""

from __future__ import annotations

import numpy as np

__all__ = ['gaspari_cohn', 'distance_matrix', 'localization_matrix']


def gaspari_cohn(r: np.ndarray) -> np.ndarray:
    r"""Gaspari-Cohn fifth-order piecewise-rational correlation function.

    The function is 1 at $r = 0$, decays monotonically, and is exactly zero for
    $r \ge 2$ (compact support), where $r = \mathrm{distance} / c$ and $c$ is the
    localization half-radius. Gaspari & Cohn (1999), Eq. (4.10).

    Parameters
    ----------
    r : ndarray
        Non-negative normalized distances, ``distance / c``.

    Returns
    -------
    ndarray
        Correlation values in $[0, 1]$, same shape as `r`.
    """
    r = np.abs(np.asarray(r, dtype=float))
    out = np.zeros_like(r)

    near = r <= 1.0
    x = r[near]
    out[near] = -0.25 * x**5 + 0.5 * x**4 + 0.625 * x**3 - (5.0 / 3.0) * x**2 + 1.0

    far = (r > 1.0) & (r < 2.0)
    x = r[far]
    out[far] = (x**5 / 12.0 - 0.5 * x**4 + 0.625 * x**3 + (5.0 / 3.0) * x**2
                - 5.0 * x + 4.0 - (2.0 / 3.0) / x)

    return out


def distance_matrix(pos_a: np.ndarray, pos_b: np.ndarray, period: float | None = None) -> np.ndarray:
    """Pairwise distances between two sets of scalar positions.

    Parameters
    ----------
    pos_a : ndarray, shape (Na,)
        Positions of the first set of components (e.g. state grid points).
    pos_b : ndarray, shape (Nb,)
        Positions of the second set (e.g. observation locations).
    period : float, optional
        Domain period for a ring topology (e.g. ``Nx`` for the Lorenz-96 lattice,
        the domain length for the Kuramoto-Sivashinsky equation); distances then
        wrap around. Default None (open domain).

    Returns
    -------
    ndarray, shape (Na, Nb)
    """
    pos_a = np.asarray(pos_a, dtype=float).ravel()
    pos_b = np.asarray(pos_b, dtype=float).ravel()
    dists = np.abs(pos_a[:, None] - pos_b[None, :])
    if period is not None:
        dists = np.minimum(dists, period - dists)
    return dists


def localization_matrix(pos_a: np.ndarray, pos_b: np.ndarray, radius: float,
                        period: float | None = None) -> np.ndarray:
    """Gaspari-Cohn localization matrix between two sets of positions.

    Entry ``(i, j)`` is ``gaspari_cohn(distance(i, j) / radius)``: 1 at zero
    distance, zero beyond ``2 * radius``.

    Parameters
    ----------
    pos_a, pos_b : ndarray
        Component positions, see `distance_matrix`.
    radius : float
        Localization half-radius $c$ (same units as the positions).
    period : float, optional
        Domain period for a ring topology; see `distance_matrix`.

    Returns
    -------
    ndarray, shape (len(pos_a), len(pos_b))
    """
    if radius <= 0:
        raise ValueError(f"Localization radius must be positive, got {radius}.")
    return gaspari_cohn(distance_matrix(pos_a, pos_b, period=period) / radius)
