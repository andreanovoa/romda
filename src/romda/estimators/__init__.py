"""
estimators/__init__.py
======================
State/parameter estimators: the filters of romda's papers. The layout, the keywords and the
equations mirror those of ``biasda`` (``biasda.base``, ``biasda.sequential``, ``biasda.inflation``),
which romda will import once it is released.

Taxonomy
--------
An estimator is classified by **what it estimates** and by **how it assimilates**.

What is estimated (a configuration of the augmented state, not a class tree):

- state (phi), the default;
- state + parameters (alpha): name the parameters in ``est_alpha``; they are appended to the
  state and updated by the same analysis (``parameter_estimation='joint'``);
- state + parameters + bias (b): attach a `Bias` estimator (`romda.bias_estimators`) to any
  estimator; the analysis then corrects the forecast observables with the current bias estimate
  (the un-regularized, gamma = 0 bias-aware update). Like inflation, this is a
  configuration, not a class per filter; the gamma > 0 bias-norm regularization is a configuration
  too (``bias_regularization``), with `rEnKF` as its named stochastic entry point.

How it assimilates (class hierarchy)
------------------------------------
romda keeps the filters that its papers employ: the washout-free echo state network (EnSRKF, EnKF,
multiplicative inflation, joint parameters), the thermoacoustic papers (rEnKF with an ESN bias)
and the qlROM data assimilation (EnKF, EnSRKF); the linear Kalman filter serves the introductory
tutorials. biasda holds the rest (see `romda.estimators.sequential`).

Estimator                       (root: configuration and bias machinery)  ->  base.py
└── SequentialEstimator         (forecast/analysis cycle)                 ->  base.py
    ├── KalmanFilter            (type = linear, parameter_estimation = None | joint)  ->  sequential/kalman.py
    └── EnsembleFilter          (type = stochastic | sqrt,
        │                        parameter_estimation = None | joint)     ->  sequential/ensemble.py
        ├── EnKF                (type='stochastic')
        │   └── rEnKF           (EnKF with the gamma-regularized bias-aware update)
        │       └── rBA_EnKF    (the name of rEnKF in Nóvoa, Racca & Magri 2023)
        └── EnSRKF              (type='sqrt')

Auxiliary: `romda.estimators.inflation` (multiplicative covariance inflation) and
`romda.estimators.localization` (Gaspari-Cohn correlation functions, as in ``biasda.localization``).
With ``localization``, ``EnsembleFilter(type='sqrt')`` is the LETKF, which tapers with these functions;
``inflation='adaptive'`` and ``inflation='innovation'`` estimate the inflation factor.

Forecast strategy
-----------------
The ensemble estimators use a `romda.models.Model` instance for the forecast step; the
forecast is the model's ``time_integrate()``. The Kalman filter can instead run from a linear
transition matrix ``F``.

Observation operator
--------------------
For all estimators the observation operator M maps the state to the model observables:
    q = M @ psi    if M is an ndarray (Nq, N)
    q = M(psi)     if M is a callable
If M is not supplied explicitly, it is read from model.M.

Notation
--------
N      state dimension (Nphi + Na + Nq for augmented ensemble)
Na     number of estimated parameters
Nphi   number of model state variables
Nq     number of observables
m      ensemble size
psi    state vector / ensemble matrix (N,) or (N, m)
M      measurement operator (Nq, N) or callable
d      observation vector (Nq,)
Cdd    observation noise covariance (Nq, Nq)
Cpp    model (prior/forecast) covariance (N, N)
K      Kalman gain matrix (N, Nq)

References
----------
Kalman (1960) KF
Evensen (2009) EnKF, EnSRKF
Evensen, Vossepoel & van Leeuwen (2022) Data Assimilation Fundamentals
Nóvoa, Racca & Magri (2023) rBA-EnKF (rEnKF)
"""

from romda.estimators.base import Estimator, SequentialEstimator
from romda.estimators.inflation import multiplicative_inflation
from romda.estimators.sequential import EnKF, EnsembleEstimator, EnsembleFilter, EnSRKF, KalmanFilter, rBA_EnKF, rEnKF

__all__ = [
    "Estimator",
    "SequentialEstimator",
    "EnsembleEstimator",
    "EnsembleFilter",
    "EnKF",
    "EnSRKF",
    "rEnKF",
    "rBA_EnKF",
    "KalmanFilter",
    "multiplicative_inflation",
]
from romda.estimators.inflation import adaptive_inflation_factor, innovation_inflation_factor  # noqa: E402

__all__ += ["adaptive_inflation_factor", "innovation_inflation_factor"]
