"""Abstract base class shared by all state/parameter estimators.

See the package docstring (`romda.estimators`) for the estimator taxonomy, the forecast
strategy, and the notation table. The layout, the keywords and the equations mirror
those of ``biasda`` (``biasda.base``), which romda will import once it is released.

The forecast-model protocol
---------------------------
The ensemble estimators own a forecast model (`romda.models.Model`, from ``dynamodels``)
and require of it:

- attributes ``dt``, ``precision_t``, ``Nphi``, ``Na``, ``Nq``, ``est_alpha``,
  ``alpha_lims``, ``rng``, ``M`` (measurement operator over the augmented state
  ``[phi; alpha; q]``, a matrix or a callable) and ``m`` (ensemble size);
- methods ``time_integrate(Nt)``, ``update_history(psi, t, reset=,
  modify_saved_states=)``, ``get_observables()``, ``get_observable_hist(Nt=)``,
  ``init_ensemble(m=, std_phi=, std_alpha=, ...)``, ``copy()`` and ``close()``;
- history accessors ``hist`` (shape ``(Nt, N, m)``), ``hist_t``,
  ``current_state`` and ``current_time``.

The Kalman filter (`KalmanFilter`) can run without a model, from a linear transition
matrix ``F`` instead.
"""

from abc import ABC, abstractmethod
from collections import namedtuple
from copy import deepcopy

import numpy as np
from romda.bias_estimators import Bias, NoBias
from romda.models import Model

AssimilatedData = namedtuple("AssimilatedData", ["data", "times"])

#: The attributes that a keyword sets, with that keyword: one spelling per option.
_STORED = {'regularization_factor': 'bias_regularization', 'inflation_factor': 'inflation'}

# ══════════════════════════════════════════════════════════════════════════════
# Abstract base — Estimator
# ══════════════════════════════════════════════════════════════════════════════

class Estimator(ABC):
    r"""Root of the estimators: the configuration and the bias machinery shared by every family.

    romda keeps the sequential estimators that its papers employ (`SequentialEstimator`: the
    ensemble Kalman filters and the linear Kalman filter); biasda adds the particle filters,
    nudging, the smoothers and the variational estimators. Every concrete estimator owns a forecast model or a one-step
    map for the forecast step and, optionally, a `Bias` instance for bias-aware assimilation. The observation operator maps state
    to the model observables, $\vect{q} = \matr{M}\vect{\psi}$ if $\matr{M}$
    is a matrix, or $\vect{q} = \matr{M}(\vect{\psi})$ if $\matr{M}$ is
    callable; if not supplied explicitly it is read from `model.M`.

    What is estimated is a configuration of the augmented state, not a separate
    class tree: the state is always estimated, `est_alpha` names the model
    parameters appended to the state, and a `Bias` instance (``parent_bias=`` on
    the ensemble filters, ``bias=`` on the others) makes the estimator bias-aware.

    Subclasses must implement `analysis_step(d, Cdd, return_analysis)`: the Bayesian update
    given observation $\vect{d}$ and observation-noise covariance $\matr{C}_{dd}$.
    Implementations build the forecast state internally, run the filter update,
    validate the result, and update the model history in-place.

    **Notation**

    | Symbol | Meaning | Shape |
    | --- | --- | --- |
    | $N$ | augmented state dimension ($N_\phi{+}N_\alpha$, or $+N_q$ once observables are appended) | |
    | $N_\phi$ | model state dimension | |
    | $N_\alpha$ | number of estimated parameters | |
    | $N_q$ | number of observables | |
    | $m$ | ensemble size | |
    | $\vect{\psi}$ | state vector | $(N,)$ |
    | $\bar{\vect{\psi}}$ | ensemble mean | $(N,)$ |
    | $\matr{\Psi}$ | ensemble | $(N, m)$ |
    | $\matr{\Psi}'$ | ensemble deviations, $\matr{\Psi} - \bar{\vect{\psi}}$ | $(N, m)$ |
    | $\matr{M}$ | measurement operator (matrix or callable) | $(N_q, N)$ |
    | $\vect{d}$ | observation vector | $(N_q,)$ |
    | $\matr{C}_{dd}$ | observation-noise covariance | $(N_q, N_q)$ |
    | $\matr{C}_{\psi\psi}$ | forecast (prior) covariance; ensemble approximation $\matr{\Psi}'\matr{\Psi}'^\mathrm{T}/(m{-}1)$ | $(N, N)$ |
    | $\matr{K}$ | Kalman gain | $(N, N_q)$ |

    Parameters
    ----------
    psi0 : ndarray, shape (N,), optional
        Initial state mean of the estimator that owns its state (`KalmanFilter`); it sets $N$.
        The ensemble filters read the state, the operator and the parameters from their forecast
        model (``parent_model``) instead.
    Cdd : ndarray, shape (Nq, Nq), optional
        Default observation-noise covariance; it sets $N_q$.
    M : ndarray, shape (Nq, N), or callable, optional
        Measurement operator (default ``model.M``).
    model : forecast model, optional
        Forecast model of the mean (`KalmanFilter`).
    F : ndarray, shape (N, N), optional
        Linear transition matrix.
    bias : Bias instance, optional
        Bias estimator; any `Bias` other than `NoBias` makes the analysis bias-aware
        (``parent_bias`` on the ensemble filters).
    est_alpha : sequence of str, optional
        Names of the estimated parameters, the trailing components of the state (default none;
        ``model.est_alpha`` with a model).

    Attributes
    ----------
    est_alpha : list of str
        Names of model parameters to estimate (default ``[]``).
    start_param : int
        Analysis step at which parameter estimation starts; parameters are
        frozen in earlier analyses (0 = active from the first analysis).
    start_bias : int
        Analysis step at which a bias-aware filter starts its bias-aware
        update; a plain EnKF update is applied in earlier analyses
        (0 = active from the first analysis).
    inflation_factor : float
        Covariance/ensemble inflation factor (default 1.0), set by the keyword ``inflation``.
    inflate_parameters : bool
        Whether the inflation also acts on the estimated parameters (default
        True). A parameter has no forecast dynamics, so an inflated parameter
        performs a random walk; False keeps the analysis spread of the parameters
        (their forecast spread after a rejected analysis).
    inflation_factor_rejection : float
        Inflation applied after a rejected analysis (default 1.002).

    Other Parameters
    ----------------
    bias_regularization : float, optional
        Bias-regularization factor $\gamma$ (stored as ``regularization_factor``); requires a bias.
    inflation : float, optional
        Multiplicative inflation factor (stored as ``inflation_factor``).

    Notes
    -----
    ``bias_regularization`` and ``inflation`` are the only spellings: the stored attributes are not
    accepted as keywords.

    ``bias_estimation`` (the bias-estimation routes) and ``localization`` are options of biasda's
    estimators and raise NotImplementedError here.

    References
    ----------
    Kalman (1960). A new approach to linear filtering and prediction problems.
    *J. Basic Eng.*, 82(1), 35-45.

    Evensen, Vossepoel & van Leeuwen (2022). *Data Assimilation Fundamentals: A
    Unified Formulation of the State and Parameter Estimation Problem.* Springer.
    [DOI: 10.1007/978-3-030-96709-3](https://doi.org/10.1007/978-3-030-96709-3).

    Nóvoa, Racca & Magri (2023). Inferring unknown unknowns: Regularized bias-aware
    ensemble Kalman filter. *Comput. Methods Appl. Mech. Eng.*, 418, 116502.
    """

    # ── General DA class-level defaults ──────────────────────────────────────

    # Warm-up windows, counted in analysis steps (0 = active from the start).
    start_param: int = 0   # parameter estimation starts at this analysis step
    start_bias: int = 0    # bias-aware update starts at this analysis step
    activate_parameter_estimation: bool = True

    inflation_factor: float = 1.0
    inflate_parameters: bool = True   # False: inflate the state rows only
    inflation_factor_rejection: float = 1.002
    # State rejection: an analysis that moves the ensemble mean of the physical state (not the reservoir of an
    # ESN, not the parameters) by more than ``max_mean_shift`` climatological standard deviations is rejected,
    # and the forecast is kept (None: never). ``climatology``: the standard deviation of each physical state
    # component; None computes it from a free run of the model (`EnsembleEstimator.model_climatology`).
    max_mean_shift: float | None = None
    climatology = None
    # Gaspari-Cohn localization (the LETKF of `EnsembleFilter(type='sqrt')`): the half-radius, set by the keyword
    # ``localization`` (None: global analysis), the positions of the physical state components and of the
    # observables (default: the index of each component; every component observed), and the period of a ring.
    loc_radius: float | None = None
    state_positions = None
    obs_positions = None
    loc_period: float | None = None
    #: Whether the estimator implements ``localization``.
    _localization = False
    # Bias-regularization factor gamma of the bias-aware update (None is 0;
    # see `gamma`); a Bias instance, not this factor, makes an estimator bias-aware.
    regularization_factor: float | None = None

    _keys_to_print = [
        'm', 'est_alpha', 'Na', 'regularization_factor', 'inflation_factor',
        'inflate_parameters', 'inflation_factor_rejection',
    ]

    def __init__(self, *, psi0=None, Cdd=None, M=None, model=None, F=None, bias=None, est_alpha=(),
                 **kwargs) -> None:
        self._assimilated = AssimilatedData(data=[], times=[])
        # Record what was consumed so subclasses can exclude these keys from
        # whatever they forward the remaining kwargs to (e.g. the model ctor).
        self._consumed_kwargs: set = set()
        # The keywords shared by the filters set the configuration attributes; the attributes
        # themselves are not keywords, so that every option has one spelling.
        for attr, keyword in _STORED.items():
            if attr in kwargs:
                raise TypeError(f"{attr}= is not a keyword argument; use {keyword}.")
        if kwargs.get('loc_radius') is not None:
            raise TypeError("loc_radius= is not a keyword argument; use localization.")
        for key in ('bias_estimation', 'f') + (() if self._localization else ('localization', 'state_positions',
                                                                              'obs_positions', 'loc_period')):
            if kwargs.get(key) is not None:
                raise NotImplementedError(f"{key}= is an option of biasda's estimators, not romda's.")
            kwargs.pop(key, None)
        self.loc_radius = kwargs.pop('localization', None)
        if self.loc_radius is not None and self.loc_radius <= 0:
            raise ValueError(f"localization={self.loc_radius}: the half-radius must be positive.")
        for key in ('state_positions', 'obs_positions', 'loc_period'):
            if key in kwargs:
                setattr(self, key, kwargs.pop(key))
        self._consumed_kwargs.update(('localization', 'state_positions', 'obs_positions', 'loc_period'))
        bias_regularization = kwargs.pop('bias_regularization', None)
        inflation = kwargs.pop('inflation', None)
        self._consumed_kwargs.update(('bias_regularization', 'inflation'))
        self._bias_regularization_keyword = bool(bias_regularization)     # a zero factor regularizes nothing
        if bias_regularization is not None:
            self.regularization_factor = bias_regularization
        if inflation in ('adaptive', 'innovation'):
            enabled = False             # the public build has neither
            if not enabled:
                raise NotImplementedError(f"inflation={inflation!r} is an option of biasda's estimators; pass a fixed factor.")
        elif isinstance(inflation, str):
            raise ValueError(f"inflation={inflation!r}: expected a factor, 'adaptive' or 'innovation'.")
        elif inflation is not None:
            self.inflation_factor = inflation
        keys = list(kwargs.keys())
        for attr in keys:
            # Derived read-only properties (e.g. EnsembleEstimator.m, which reads
            # the model's history) must not be assigned; they are consumed
            # downstream, e.g. by Model.init_ensemble.
            class_attr = getattr(self.__class__, attr, None)
            if isinstance(class_attr, property) and class_attr.fset is None:
                continue
            # hasattr, not __class__.__dict__: the config fields are declared on
            # Estimator, so a leaf class (EnSRKF, EnKF, ...) has none of them in its
            # own __dict__ and every config kwarg was silently dropped.
            if hasattr(self.__class__, attr):
                setattr(self, attr, kwargs.pop(attr))
                self._consumed_kwargs.add(attr)
        self._consumed_kwargs.update(('psi0', 'Cdd', 'M', 'model', 'F', 'bias', 'est_alpha'))
        if psi0 is not None:
            self._init_state(psi0, M, Cdd, model=model, F=F, bias=bias, est_alpha=est_alpha)
        elif any(v is not None for v in (Cdd, M, model, F, bias)) or est_alpha:
            raise TypeError(f"{self.name}: Cdd, M, model, F, bias and est_alpha configure the state of psi0; "
                            "pass psi0.")

    # ── Identity / introspection ──────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self.__class__.__name__

    # ── Model / Bias accessors ───────────────────────────────────────────────

    _model = None
    _rng = None

    @property
    def model(self) -> Model | None:
        """The forecast model used for the forecast step (None for a Kalman filter run from ``F``)."""
        return self._model

    @property
    def bias(self) -> Bias | None:
        """Bias instance, or None if not configured."""
        return getattr(self, "_bias", None)

    @bias.setter
    def bias(self, value: Bias | None) -> None:
        """Set the bias model (a Bias instance or None); romda attaches the ESN bias after construction."""
        if value is not None and not isinstance(value, Bias):
            raise ValueError("Bias must be a Bias instance or None.")
        self._bias = value

    # ── The state of the estimator that owns it (Kalman filter) ───────────────
    #
    # The ensemble estimators read the state, the observation operator and the parameters from
    # their forecast model; the others own a state mean, set by `_init_state`, and the accessors
    # below return it when it is set.
    _psi = None
    _M = None
    _Nq = None
    _est_alpha = None

    def _init_state(self, psi0, M=None, Cdd=None, *, model=None, F=None, bias=None, est_alpha=()) -> None:
        """Set the state mean, the observation operator and noise, the forecast matrix and the bias.

        $N$ is the size of `psi0`, and $N_q$ that of `Cdd`, or of a matrix `M` (left to the subclass
        when both are missing and `M` is callable). Without a model, `current_time` counts steps.
        """
        if M is None:
            if model is None:
                raise ValueError("M must be supplied when no model is provided.")
            M = model.M
        self._psi = np.array(psi0, dtype=float).ravel()
        self.N = self._psi.size
        self.Cdd = None if Cdd is None else np.atleast_2d(Cdd)
        if self.Cdd is not None:
            self._Nq = self.Cdd.shape[0]
        elif not callable(M):
            self._Nq = np.atleast_2d(M).shape[0]
        self._M = M
        self._model, self._F = model, (None if F is None else np.atleast_2d(F))
        self._t = float(model.current_time) if model is not None else 0.0
        self._est_alpha = list(model.est_alpha) if model is not None else list(est_alpha)
        self.bias = bias

    def _advance(self, psi: np.ndarray) -> np.ndarray:
        """Advance a state (or a matrix of state columns) by one step via `F`; the parameters persist."""
        out = np.array(self._F @ psi, dtype=float)
        out[self.Nphi:] = psi[self.Nphi:]
        return out

    # ──  Current state accessor ───────────────────────────────────
    @property
    def current_state(self) -> np.ndarray:
        """Current state: the estimator's own mean, or the model's state."""
        return self._psi if self._psi is not None else self.model.current_state

    @property
    def current_time(self) -> float:
        """Current time: the estimator's own clock, or the model's."""
        return self._t if self._psi is not None else self.model.current_time

    @property
    def current_bias_estimate(self) -> np.ndarray:
        """Current (ensemble-mean) bias estimate, shape (Nq, 1); zero if no bias model."""
        if self.bias is not None:
            return self.bias.current_bias
        return np.zeros((self.model.Nq, 1))

    # ── Model config helper ───────────────────────────────────────────────────

    @property
    def est_alpha(self) -> list[str]:
        """Names of the estimated parameters (for an own state, its trailing `Na` components)."""
        return self._est_alpha if self._est_alpha is not None else self.model.est_alpha

    @property
    def Na(self) -> int:
        return len(self._est_alpha) if self._est_alpha is not None else self.model.Na

    @property
    def Nphi(self) -> int:
        """Size of the state vector without the parameters."""
        return self.N - self.Na if self._psi is not None else self.model.Nphi

    @property
    def Nq(self) -> int:
        """Number of observable dimensions."""
        return self._Nq if self._Nq is not None else self.model.Nq

    @property
    def M(self) -> np.ndarray:
        return self._M if self._M is not None else self.model.M

    @property
    def rng(self) -> np.random.Generator:
        if self._rng is None:
            self._rng = self.model.rng
        return self._rng

    @rng.setter
    def rng(self, value: np.random.Generator) -> None:
        self._rng = value

    # ── Measurement-operator wrapper ───────────────────────────────────

    def _MA(self, A: np.ndarray) -> np.ndarray:
        """Map ensemble A (N, m) through M --> (Nq, m)."""
        if callable(self.M) and not isinstance(self.M, np.ndarray):
            return np.asarray(
                np.column_stack([self.M(A[:, i]) for i in range(A.shape[1])])
            )
        M = self.M
        if A.shape[0] != M.shape[1]:
            # Parameters excluded from the analysis: drop the alpha columns of M,
            # keeping the state and the trailing observable ones.
            Nq = M.shape[0]
            M = np.hstack((M[:, :A.shape[0] - Nq], M[:, -Nq:]))
        return M @ A

    # ── History update ────────────────────────────────────────────────────────

    def update_history(
        self,
        psi: np.ndarray,
        t=None,
        modify_saved_states: bool = False,
        reset: bool = False,
    ) -> None:
        """Update the model history."""
        self.model.update_history(psi, t, reset=reset, modify_saved_states=modify_saved_states)

    # ── Bias-awareness helpers ────────────────────────────────────────────────

    @property
    def is_bias_aware(self) -> bool:
        """True when the estimator carries an explicit bias correction.

        Bias-aware assimilation is a *configuration* of every estimator, not a
        separate class tree (like inflation): passing a `Bias`
        instance makes the analysis correct the forecast observables with the
        current bias estimate, which is the $\\gamma = 0$ limit of the
        regularized bias-aware update. The $\\gamma > 0$ bias-norm
        regularization is a configuration too (``bias_regularization``), and
        `rEnKF` is its named stochastic entry point.
        """
        return self.bias is not None and not isinstance(self.bias, NoBias)

    def _positions(self, n_state: int, Nq: int):
        """Positions of the `n_state` state components and of the `Nq` observables."""
        pos_x = np.arange(n_state) if self.state_positions is None else np.asarray(self.state_positions)
        if self.obs_positions is not None:
            return pos_x, np.asarray(self.obs_positions)
        if Nq == len(pos_x):
            return pos_x, pos_x
        raise ValueError('obs_positions is required when not every state component is observed.')

    @property
    def gamma(self) -> float:
        """Bias-regularization factor; 0 when `regularization_factor` is unset."""
        return self.regularization_factor or 0.0

    def _check_parameter_estimation(self) -> None:
        """Raise ValueError unless ``parameter_estimation`` is None exactly when `est_alpha` is empty."""
        if self.parameter_estimation == 'dual':
            raise NotImplementedError("parameter_estimation='dual' is a scheme of biasda's estimators; romda has 'joint'.")
        if self.parameter_estimation not in (None, 'joint'):
            raise ValueError(f"parameter_estimation={self.parameter_estimation!r}: expected None or 'joint'.")
        if (self.parameter_estimation is None) != (self.Na == 0):
            raise ValueError(f"parameter_estimation={self.parameter_estimation!r} with est_alpha={list(self.est_alpha)}: "
                             "estimate the parameters ('joint') if and only if est_alpha is set.")

    def _bias_active(self) -> bool:
        """Bias-aware and past the bias-blind warm-up of `start_bias` analyses.

        A ``bias_regularization`` keyword without a bias raises here, at the first analysis: biasda
        raises at construction, but romda attaches the ESN bias to a built estimator.
        """
        if self._bias_regularization_keyword and self.bias is None:
            raise ValueError("bias_regularization regularizes the bias-aware update: pass a bias "
                             "(parent_bias=... or bias=..., or estimator.bias = ... before the first analysis).")
        return (self.is_bias_aware
                and len(self.assimilated_data.times) >= self.start_bias)

    def _bias_terms(self, y_pred: np.ndarray, d: np.ndarray):
        """Current bias estimate for the analysis: ``(b, J, d)``.

        Returns the bias column ``b`` (Nq, 1), its Jacobian ``J`` (Nq, Nq) with
        respect to the observables, and the observation ``d`` shifted by the
        mean bias-innovation gap when the bias model flags the observations as
        biased. Returns ``(0, 0, d)``-shaped zeros when the estimator is
        bias-blind. The bias forms of romda are driven by the innovation, so the
        prediction `y_pred` does not change them (biasda's observable-driven forms
        are evaluated at it).
        """
        Nq = np.atleast_1d(d).shape[0]
        if not self.is_bias_aware:
            return np.zeros((Nq, 1)), np.zeros((Nq, Nq)), d
        b = np.asarray(self.current_bias_estimate)
        J = self.bias.state_derivative()
        if self.bias.biased_observations:
            d = d + np.mean(b - self.bias.current_innovations, axis=-1)
        return b.reshape(Nq, 1), np.atleast_2d(J), d

    def _stacked_innovation(self, S, y, d, Cdd):
        """``(S, z, Cdd)`` of the update: the mapped deviations (or operator) `S`, the innovation
        ``z = d - y`` (column) and `Cdd`. Bias-aware, `S` is (I + J)-mapped and `z` bias-corrected,
        with the pseudo-observation 0 = sqrt(gamma) b(q) stacked under them when gamma > 0, so the
        regularized update is the plain one on the stacked quantities. `y` is (Nq, 1)."""
        if not self._bias_active():
            return S, d[:, None] - y, Cdd
        b, J, d = self._bias_terms(y, d)
        S_eff, z = (np.eye(len(d)) + J) @ S, d[:, None] - y - b
        if self.gamma > 0.0:
            S_eff = np.vstack([S_eff, np.sqrt(self.gamma) * (J @ S)])
            z = np.vstack([z, -np.sqrt(self.gamma) * b])
            Cdd = np.kron(np.eye(2), Cdd)
        return S_eff, z, Cdd

    def _update_bias_from_analysis(self, d: np.ndarray, Ya: np.ndarray) -> None:
        """Update the bias estimator with the analysis innovation ``d - Ya``."""
        if self.bias is None:
            return
        ia = np.atleast_1d(d)[:, np.newaxis] - Ya   # (Nq, m) or (Nq, 1)
        self.bias.update_state(self.bias.update_state_from_innovation(ia))

    # ── Assimilated data log ──────────────────────────────────────────────────

    @property
    def assimilated_data(self):
        """Namedtuple with fields *data* and *times* of all assimilated obs."""
        return self._assimilated

    @assimilated_data.setter
    def assimilated_data(self, value: tuple) -> None:
        """Append (y_obs, t_obs) to the assimilation log."""
        y_obs, t_obs = value
        self._assimilated.data.append(y_obs)
        self._assimilated.times.append(t_obs)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def copy(self):
        return deepcopy(self)

    def close(self) -> None:
        """Close the model (if any) and the bias estimator resources, e.g. integrator pools."""
        if getattr(self, '_model', None) is not None:
            self._model.close()
        if self.bias is not None:
            self.bias.close()

    def print_parameters(self, indent: int = 0) -> None:
        print(f'\n{self.name}')
        print('=' * len(self.name))
        for key in self._keys_to_print:
            print(f'{" " * indent}{key} = {getattr(self, key, "N/A")}')

    # ── Bias initialisation (general DA, not ensemble-specific) ──────────────

    def _init_bias(self, parent_bias: Bias | type[Bias] | None = None) -> None:
        """Initialise the bias model and store it as ``self._bias``.

        Parameters
        ----------
        parent_bias : Bias instance, Bias subclass, or None
            If a subclass, it is instantiated using the model's current
            observable mean as the initial innovation.
        """
        if isinstance(parent_bias, Bias):
            self._bias = parent_bias.copy()
        elif parent_bias is None:
            self._bias = None
        else:
            assert isinstance(parent_bias, type) and issubclass(parent_bias, Bias), \
                "parent_bias must be a Bias instance or a subclass of Bias"
            pm = self.model
            try:
                y0_all = pm.get_observables()
                y0 = np.mean(y0_all, axis=-1, keepdims=True)
            except (AttributeError, IndexError):
                y0 = np.zeros((1, pm.Nq, 1))

            self._bias = parent_bias(
                innovation=y0,
                t=pm.current_time,
                dt=pm.dt,
                initial_capacity=pm.history._initial_capacity,
                rom=pm,
            )

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def analysis_step(
        self,
        d: np.ndarray,
        Cdd: np.ndarray,
        return_analysis: bool = False,
    ) -> np.ndarray | None:
        """Perform the Bayesian analysis step given observation *d*.

        Implementations are responsible for building the augmented forecast
        state, running the filter update, validating the result, and updating
        the model history in-place.  Optionally return the analysed state when
        ``return_analysis=True``.
        """


# ══════════════════════════════════════════════════════════════════════════════
# Sequential estimators — the forecast/analysis cycle
# ══════════════════════════════════════════════════════════════════════════════

class SequentialEstimator(Estimator):
    r"""Family of the sequential estimators: filters that alternate a forecast and an analysis step.

    `forecast_step` advances the forecast model (and the bias estimator) to the next observation time,
    and `analysis_step` corrects the state with the observation. The concrete filters are
    `KalmanFilter` and `EnsembleFilter`.
    """

    # ── Forecast step (shared implementation) ────────────────────────────────

    def forecast_step(self, t_end=None, close: bool = False, **kwargs) -> None:
        """Advance model (and bias, if present) in time.

        Parameters
        ----------
        t_end : float, optional
            Target time; Nt is derived from it if provided.
        close : bool
            Close the estimator (`close`) after the forecast, e.g. after the last forecast of a run.
        """
        pm = self.model

        if t_end is not None:
            t_end = round(t_end, pm.precision_t)
            # round, not truncate: the division is one short whenever it is inexact
            # in binary (1.18 / 0.01 = 117.999...)
            Nt = int(np.round((t_end - pm.current_time) / pm.dt))
        else:
            Nt = kwargs.get("Nt", -1)

        if Nt == 0:
            return  # Already at requested time; nothing to advance.
        assert Nt > 0, "Must specify positive Nt or t_end for forecast_step."

        psi, t = pm.time_integrate(Nt=Nt, averaged=kwargs.get("averaged", False))

        if t_end is not None:
            assert abs(t[-1] - t_end) < pm.dt, (
                f"Final time {t[-1]} does not match requested t_end {t_end}."
            )

        pm.update_history(psi, t)

        if self.bias is not None:
            self.bias.advance(Nt=Nt)
            # The two clocks round to their own decimals; they agree to within a fraction of a
            # step, and the bias clock is then set to the model's so that the rounding does not drift.
            if abs(pm.current_time - self.bias.current_time) > 0.5 * pm.dt:
                raise AssertionError(
                    f"Time mismatch: model {pm.current_time} vs bias {self.bias.current_time}"
                )
            self.bias.update_state(self.bias.current_state, t=pm.current_time)

        if close:
            self.close()
