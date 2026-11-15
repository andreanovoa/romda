"""Covariance inflation methods (Evensen 2009, Chap. 15).

The analysis step of an ensemble filter reduces the ensemble variance not only
where the data justify it, but also through *spurious correlations* between the
state and the predicted measurements that a finite ensemble cannot average to
zero (Sect. 15.1). Inflation counteracts that underestimation of the spread.

- `multiplicative_inflation` — fixed multiplicative inflation about the ensemble mean
  (Anderson & Anderson 1999; Evensen 2009, Eq. 15.3, Sect. 15.2), controlled by
  the estimator's ``inflation``.
- `adaptive_inflation_factor` — factor estimated at each analysis from a random
  ensemble pushed through the same update (Evensen 2009, Sect. 15.3). It restores
  the spread lost to sampling error only.
- `innovation_inflation_factor` — factor of the forecast covariance estimated from
  the innovation statistics (Wang & Bishop 2003; Li, Kalnay & Miyoshi 2009), the
  estimator's ``inflation='innovation'``. It restores the spread lost to model error
  as well; on a bias-aware filter, it measures the random model error that the bias
  model does not explain.

References
----------
Evensen (2009). *Data Assimilation: The Ensemble Kalman Filter*, 2nd ed.,
Springer, Chap. 15.

Evensen, Vossepoel & van Leeuwen (2022). *Data Assimilation Fundamentals.*
Springer.

Wang & Bishop (2003). A comparison of breeding and ensemble transform Kalman
filter ensemble forecast schemes. *J. Atmos. Sci.*, 60, 1140-1158.

Li, Kalnay & Miyoshi (2009). Simultaneous estimation of covariance inflation and
observation errors within an ensemble Kalman filter. *Q. J. R. Meteorol. Soc.*,
135, 523-533.

Tandeo et al. (2020). A review of innovation-based methods to jointly estimate
model and observation error covariance matrices in ensemble data assimilation.
*Mon. Wea. Rev.*, 148, 3973-3994.
"""

import numpy as np

__all__ = ['multiplicative_inflation']


def multiplicative_inflation(A: np.ndarray, rho: float) -> np.ndarray:
    r"""Multiplicative inflation of an ensemble about its mean.

    $\vect{\psi}_j \leftarrow \rho\,(\vect{\psi}_j -
    \bar{\vect{\psi}}) + \bar{\vect{\psi}}$ — Evensen (2009),
    Eq. (15.3) — with $\rho$ typically slightly greater than one (e.g. 1.01).

    Parameters
    ----------
    A : (N, m) ndarray
        Ensemble matrix.
    rho : float or (N, 1) ndarray
        Inflation factor, or one factor per row.

    Returns
    -------
    (N, m) ndarray
    """
    A_mean = np.mean(A, axis=-1, keepdims=True)
    return A_mean + rho * (A - A_mean)


__all__ += ['adaptive_inflation_factor']
def adaptive_inflation_factor(
    transform: np.ndarray,
    rng: np.random.Generator,
    n_rand: int = 100,
) -> float:
    r"""Adaptive inflation factor from a random-ensemble update (Evensen 2009, Sect. 15.3).

    A dynamics-free random ensemble $\matr{B} \in \mathbb{R}^{n_\mathrm{rand}
    \times m}$ is sampled from $\mathcal{N}(0, 1)$ and each row is rescaled to
    exactly zero mean and unit variance. It is then passed through the same
    member-space analysis update as the state ensemble,
    $[\matr{\Psi}^\mathrm{a};\,\matr{B}^\mathrm{a}] =
    [\matr{\Psi}^\mathrm{f};\,\matr{B}^\mathrm{f}]\,\hat{\matr{X}}$
    (Eq. 15.1). Since $\matr{B}$ is uncorrelated with the predicted
    measurements, any variance it loses is purely the spurious-correlation
    sampling error, and

    $\rho = 1 \,/\, \overline{\mathrm{std}(\matr{B}^\mathrm{a})}$

    (one over the row-averaged standard deviation) is the factor that restores it.

    Parameters
    ----------
    transform : (m, m) ndarray
        Member-space analysis transform $\hat{\matr{X}}$ such that
        $\matr{\Psi}^\mathrm{a} = \matr{\Psi}^\mathrm{f}\hat{\matr{X}}$
        (stashed by the ensemble filter kernels as ``_analysis_transform``).
    rng : numpy Generator
        Random number generator for sampling B.
    n_rand : int
        Number of rows of B; more rows reduce the sampling error of the
        estimate, especially for small m.

    Returns
    -------
    float
        The inflation factor rho.
    """
    m = transform.shape[0]
    B = rng.standard_normal((n_rand, m))
    B -= np.mean(B, axis=1, keepdims=True)
    B /= np.std(B, axis=1, ddof=1, keepdims=True)
    Ba = B @ transform
    return float(1.0 / np.mean(np.std(Ba, axis=1, ddof=1)))


__all__ += ['innovation_inflation_factor']
def innovation_inflation_factor(innovation: np.ndarray, Cdd: np.ndarray, Cyy: np.ndarray) -> float:
    r"""Inflation of the forecast covariance from one innovation (Wang & Bishop 2003; Li et al. 2009).

    If the ensemble underestimates the forecast covariance by a factor $\lambda$, the forecast
    innovation $\vect{\nu} = \vect{d} - \bar{\vect{y}}^\mathrm{f}$ has the covariance
    $\matr{C}_{dd} + \lambda\,\matr{C}_{yy}$, in which $\matr{C}_{yy}$ is the ensemble
    covariance of the predicted observables. The trace of
    $\vect{\nu}\vect{\nu}^\mathrm{T}$ then estimates $\lambda$,

    $$
    \tilde{\lambda} = \frac{\vect{\nu}^\mathrm{T}\vect{\nu} - \mathrm{tr}\,\matr{C}_{dd}}
    {\mathrm{tr}\,\matr{C}_{yy}},
    $$

    which is noisy, and may be negative, for one innovation. The estimator averages it over the
    analyses, weighted by $\mathrm{tr}\,\matr{C}_{yy}$ (``inflation_rate``), and bounds the factor
    it applies below by one.

    In the bias-aware analysis, the predicted observables are
    $\vect{y} = \vect{q} + \vect{b}(\vect{q})$, so the innovation is the bias-corrected one,
    $\vect{\nu} = \vect{d} - \bar{\vect{q}}^\mathrm{f} - \vect{b}$, and, linearized about
    the forecast mean, $\matr{C}_{yy} = (\mathbb{I}+\matr{J})\matr{C}_{qq}(\mathbb{I}+\matr{J})^\mathrm{T}$.
    The regularized analysis (r-EnKF) assimilates, besides $\vect{d}$, the pseudo-observation
    $\vect{0} = \sqrt{\gamma}\,\vect{b}$ with covariance $\matr{C}_{dd}$. Its stacked innovation,
    $[\vect{\nu}; -\sqrt{\gamma}\,\vect{b}]$, has the predicted covariance
    $\mathbb{I}_2 \otimes \matr{C}_{dd} + \lambda\,\tilde{\matr{M}}\matr{C}_{qq}\tilde{\matr{M}}^\mathrm{T}$,
    with $\tilde{\matr{M}} = [\mathbb{I}+\matr{J};\, \sqrt{\gamma}\,\matr{J}]$, whose block on the
    rows of $\vect{d}$ is $\matr{C}_{dd} + \lambda\,\matr{C}_{yy}$ for every $\gamma$: the
    regularization changes the weight of the innovation in the gain,
    $(\mathbb{I}+\matr{J})^\mathrm{T}(\mathbb{I}+\matr{J}) + \gamma\,\matr{J}^\mathrm{T}\matr{J}$,
    not the statistics of $\vect{\nu}$. The estimate employs the rows of $\vect{d}$ only. The
    rows of the pseudo-observation would add $\gamma\,\vect{b}^\mathrm{T}\vect{b} -
    \mathrm{tr}\,\matr{C}_{dd}$ to the numerator, but $\vect{b}$ is the systematic error itself,
    not a forecast error, so a large bias would inflate $\lambda$ through these rows for any spread
    of the forecast.

    A bias estimate that captures the systematic model error therefore leaves $\tilde{\lambda}$ to
    measure the random model error only, and $\lambda \approx 1$ indicates that the bias model
    explains the model error. Without the bias correction, the systematic error enters
    $\vect{\nu}^\mathrm{T}\vect{\nu}$, and $\lambda$ grows to absorb it, as Li et al.
    (2009) report for a model with a large systematic error.

    Parameters
    ----------
    innovation : (Nq,) or (Nq, 1) ndarray
        Forecast innovation $\vect{\nu}$, bias-corrected in a bias-aware analysis.
    Cdd : (Nq, Nq) ndarray
        Error covariance of $\vect{d}$, which the inflation does not scale.
    Cyy : (Nq, Nq) ndarray
        Ensemble covariance of the predicted observables, which the inflation scales.

    Returns
    -------
    float
        The one-sample estimate $\tilde{\lambda}$ of the variance factor; the deviations are
        inflated by $\sqrt{\lambda}$.
    """
    v = np.ravel(innovation)
    return float((v @ v - np.trace(Cdd)) / np.trace(Cyy))
