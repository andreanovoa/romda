"""Ensemble Kalman filters: the stochastic and the square-root updates of biasda, and the bias-aware filter.

The class hierarchy is in `romda.estimators`. romda keeps the filters that its papers employ (the
washout-free echo state network, the thermoacoustic rBA-EnKF papers and the qlROM data assimilation),
with the classes, keywords and equations of ``biasda`` (``biasda.sequential.ensemble``), which romda will import once it is released.
The other updates of ``biasda.EnsembleFilter`` (``'deterministic'``, ``'sqrt-random'``,
``'sqrt-serial'``, ``'sqrt-half'``), the dual scheme, localization, the bias-estimation routes, the
particle filters and the smoothers live in biasda only.
The exception is the LETKF (``type='sqrt'`` with ``localization``), which romda has too.

Bias-aware assimilation is a configuration, not a class tree: passing
``parent_bias=<Bias>`` to any filter corrects the forecast observables with the
current bias estimate (the un-regularized, gamma = 0 update), and setting
``bias_regularization`` > 0 adds the gamma-weighted bias-norm terms — on the
stochastic path through the r-EnKF gain, on the square-root path through the
stacked pseudo-observation of zero bias. `rEnKF` is the named entry point for the
regularized stochastic filter.

Usage
-----
::
    from romda.estimators import EnSRKF

    enkf = EnSRKF(
        parent_model=Lorenz63,   # class or instance
        inflation=1.02,          # multiplicative inflation factor
        # parent_bias=ESN_bias, bias_regularization=0.5,   # bias-aware, regularized
        m=50,
        std_phi=0.2,
        std_alpha={'rho': (25, 35), 'beta': (2, 4)},
        dt=0.015,                # forwarded to Lorenz63(...)
        observe_dims=[0, 2],
    )
    enkf.forecast_step(t_end=t_d)
    enkf.analysis_step(d=d, Cdd=Cdd)
"""

import logging
from abc import abstractmethod
from collections import namedtuple
from functools import cached_property
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import cholesky, inv
from romda.bias_estimators import Bias, NoBias
from romda.estimators.base import SequentialEstimator
from romda.estimators.inflation import multiplicative_inflation
from romda.estimators.localization import localization_matrix
from romda.models import Model
from romda.observations import Observations
from romda.plotting import Palette
from romda.utils import (
    allowed_kwargs_for_func,
    canonical_name,
    cut_signals,
    given,
    interpolate,
    normalized_alpha,
    normalized_time,
    normalized_y,
)

__all__ = ['EnsembleEstimator', 'EnsembleFilter', 'EnKF', 'EnSRKF', 'rBA_EnKF', 'rEnKF']

logger = logging.getLogger(__name__)

InflationData = namedtuple('InflationData', ['times', 'factors'])
RejectedData = namedtuple('RejectedData', ['times', 'reasons'])


# ══════════════════════════════════════════════════════════════════════════════
# EnsembleEstimator — concrete base for ensemble Kalman filter family
# ══════════════════════════════════════════════════════════════════════════════

def _reservoir_size(model):
    """Rows of reservoir state after the physical state: `N_r` (every patch of a parallel ESN), else `N_units`, else 0."""
    return getattr(model, 'N_r', None) or getattr(model, 'N_units', 0)


def _sqrt_factors(S, z, Cdd):
    r"""Factors of the symmetric square-root analysis (Tippett et al. 2003), as in biasda.

    The arguments are the mapped forecast deviations $\matr{Q}' = \matr{M}\matr{\Psi}'$
    (``S``, shape (Nq, m)), the innovation $\vect{\nu}$ (``z``) and $\matr{C}_{dd}$. With
    $\matr{C}_{\nu\nu} = \matr{C}_{dd} + \matr{Q}'\matr{Q}'^\mathrm{T}/(m-1)$ the ensemble
    covariance of the innovation, the analysed mean is $\bar{\vect{\psi}}^\mathrm{a} =
    \bar{\vect{\psi}}^\mathrm{f} + \matr{\Psi}'\vect{w}$ with $\vect{w} =
    \matr{Q}'^\mathrm{T}\matr{C}_{\nu\nu}^{-1}\vect{\nu}/(m-1)$, and the analysed deviations
    are $\matr{\Psi}'^\mathrm{a} = \matr{\Psi}'\matr{T}^{1/2}$ with
    $\matr{T} = \mathbb{I} - \matr{Q}'^\mathrm{T}\matr{C}_{\nu\nu}^{-1}\matr{Q}'/(m-1)$.
    $\matr{T}$ is the identity plus a correction of rank at most $N_q$: with
    $\hat{\matr{Q}} = \matr{L}^{-1}\matr{Q}'/\sqrt{m-1}$ and
    $\matr{C}_{dd} = \matr{L}\matr{L}^\mathrm{T}$,
    $\matr{T} = (\mathbb{I} + \hat{\matr{Q}}^\mathrm{T}\hat{\matr{Q}})^{-1}$, so the thin SVD
    $\hat{\matr{Q}} = \matr{U}\vect{s}\matr{V}^\mathrm{T}$ gives its symmetric square root,
    $\matr{T}^{1/2} = \mathbb{I} + \matr{V}\,\mathrm{diag}((1 + s^2)^{-1/2} - 1)\,\matr{V}^\mathrm{T}$,
    in $\mathcal{O}(m N_q^2)$ operations instead of the $\mathcal{O}(m^3)$ of an $m \times m$
    eigendecomposition.

    Returns `w` (m, 1); `sqrt_T(Psi)`, which applies the square root to deviations without
    forming $\matr{T}$; and `transform()`, which builds the member-space transform $\matr{X}$,
    $\matr{\Psi}^\mathrm{a} = \matr{\Psi}^\mathrm{f}\matr{X}$, an $m \times m$ matrix
    that the analysis does not need.
    """
    m = S.shape[1]
    w = S.T @ inv((m - 1) * Cdd + S @ S.T) @ z
    S_hat = np.linalg.solve(cholesky(Cdd), S) / np.sqrt(m - 1)
    _, s, Vt = np.linalg.svd(S_hat, full_matrices=False)
    shrink = 1.0 / np.sqrt(1.0 + s**2) - 1.0

    def sqrt_T(Psi):
        return Psi + (Psi @ Vt.T) @ (shrink[:, None] * Vt)

    def transform():
        ones_m = np.ones((m, m)) / m
        return ones_m + (np.eye(m) - ones_m) @ (w @ np.ones((1, m)) + sqrt_T(np.eye(m)))

    return w, sqrt_T, transform


class EnsembleEstimator(SequentialEstimator):
    r"""Abstract base for the ensemble Kalman filters.

    Owns the model, ensemble, and bias.  Subclasses only need to implement
    ``_analysis_kernel(Af, d, Cdd) -> Aa``.

    Attributes
    ----------
    m : int
        Number of ensemble members.
    activate_parameter_estimation : bool
        Whether to include parameter rows in the analysis update.
    regularization_factor : float or None
        Bias-regularization factor $\gamma$ of the bias-aware update (None is 0).
    update_reservoir : bool
        Whether the analysis corrects the reservoir block of an ESN model
        (``model.N_dim : model.N_dim + model.N_units``). False leaves it at its
        forecast value, a physical-state-only update; no-op for a model without ``N_units``.
    inflation_history : InflationData
        Times and factors of every applied inflation.
    rejected_analysis : RejectedData
        Times and reasons of every rejected (non-physical) analysis.

    The keyword arguments of the model's ``init_ensemble`` (e.g. ``std_phi``, ``std_alpha``,
    ``distribution_phi``, ``distribution_alpha``, ``ensemble_psi0``, a pre-built initial ensemble,
    or ``ensure_mean_at_init``) are forwarded to it, with its defaults.
    """

    # ── Ensemble-specific class-level defaults ────────────────────────────────
    update_reservoir: bool = True
    # Gaussian-mixture analysis (`romda.estimators.sequential.mixture`): the largest number of components of the
    # forecast ensemble (1: the Gaussian analysis), and the fewest members of a component.
    mixture: int = 1
    mixture_min_members: int = 3
    _mixture_cache = None

    # ── Constructor ───────────────────────────────────────────────────────────

    def __init__(
        self,
        parent_model: Model | type[Model],
        parent_bias: Bias | type[Bias] | None = None,
        **kwargs,
    ):
        """Initialise ensemble estimator.

        Parameters
        ----------
        parent_model : Model instance or Model subclass
            If a class is passed, remaining *kwargs* are forwarded to its
            constructor (e.g. ``dt``, ``psi0``, model-specific parameters).
        parent_bias : Bias instance, Bias subclass, or None
        **kwargs
            Ensemble config keys (``m``, ``std_phi``, ``std_alpha``, …) and/or
            model constructor keys are accepted here.
        """
        # The arguments of an own state (psi0, M, F, f, est_alpha, ...) configure the forecast model
        # here: they go to its constructor, not to Estimator.
        for key in ('model', 'bias'):
            if key in kwargs:
                raise TypeError(f"{key}= is not a keyword argument of {self.name}; use parent_{key}.")
        model_args = {key: kwargs.pop(key) for key in ('psi0', 'Cdd', 'M', 'F', 'f', 'est_alpha') if key in kwargs}
        super().__init__(**kwargs)

        # Split the remaining kwargs: ensemble-generation keys (e.g. est_alpha) go to
        # init_ensemble; everything else (dt, psi0, model parameters, ...) to
        # the model constructor. Keys consumed by Estimator.__init__ are
        # forwarded to neither.
        ensemble_kwargs = allowed_kwargs_for_func(parent_model.init_ensemble, kwargs | model_args)
        model_kwargs = {k: v for k, v in kwargs.items() if k not in self._consumed_kwargs and k not in ensemble_kwargs}
        model_kwargs |= {k: v for k, v in model_args.items() if k not in ensemble_kwargs}

        # Instantiate or copy the model. A model instance takes no constructor keywords: whatever remains is a keyword
        # that neither the filter nor init_ensemble takes (a misspelling, or an option of another build), which the
        # copy would drop silently.
        if isinstance(parent_model, type):
            parent_model = parent_model(**model_kwargs)
        else:
            if model_kwargs:
                raise TypeError(f"unexpected keyword arguments: {sorted(model_kwargs)} (parent_model is an instance "
                                f"of {type(parent_model).__name__}, whose constructor they would have gone to)")
            parent_model = parent_model.copy()

        parent_model.init_ensemble(**ensemble_kwargs)

        self._model = parent_model

        self._init_bias(parent_bias)
        self.inflation_history = InflationData(times=[], factors=[])
        self.rejected_analysis = RejectedData(times=[], reasons=[])
        from romda.estimators.sequential.mixture import MixtureData
        if not (isinstance(self.mixture, (int, np.integer)) and self.mixture >= 1):
            raise ValueError(f"mixture={self.mixture!r}: expected the largest number of components, an integer >= 1.")
        self.mixture_history = MixtureData(times=[], labels=[], weights_f=[], weights_a=[], counts=[])

    # ── Ensemble-specific observable helpers ─────────────────────────────────
    @property
    def m(self) -> int:
        """Ensemble size."""
        return self._model.m

    def get_observable_hist(
        self, Nt: int = 0
    ) -> tuple[np.ndarray | None, np.ndarray]:
        """Return ``(y_unbiased, y_model)`` history, interpolating bias if needed."""
        pb = self.bias
        y_model = self.model.get_observable_hist(Nt=Nt)

        if pb is None or isinstance(pb, NoBias):
            return None, y_model

        t_model = self.model.hist_t[-Nt:] if Nt else self.model.hist_t
        # hist carries the estimator's full state (an ESN's reservoir included);
        # only the bias components are commensurate with the observables
        b = pb.get_bias_hist()
        if b.shape[-1] == 1 and y_model.shape[-1] > 1:
            b = np.repeat(b, y_model.shape[-1], axis=-1)
        if len(pb.hist_t) != len(t_model):
            b = interpolate(pb.hist_t, b, t_model, fill_values=None)
        return y_model + b, y_model

    # ── Analysis step orchestration ───────────────────────────────────────────

    def analysis_step(
        self,
        d: np.ndarray,
        Cdd: np.ndarray,
        return_analysis: bool = False,
    ) -> np.ndarray | None:
        """Bayesian analysis step.

        Builds the augmented forecast ensemble, delegates the filter update to
        ``_analysis_kernel``, applies inflation, validates parameters, and
        updates the model history in-place.
        With ``mixture`` > 1, a forecast ensemble with several modes receives one
        update per mode, and the members are redistributed among the modes by their
        evidence (`romda.estimators.sequential.mixture`).

        Parameters
        ----------
        d : ndarray (Nq,)
            Observation vector.
        Cdd : ndarray (Nq, Nq)
            Observation noise covariance.
        return_analysis : bool
            If True, return the analysed state array.
        """
        Af_state = self.current_state          # (Nphi+Na, m)

        # State-estimation-only warm-up.
        if self.start_param > 0:
            self.activate_parameter_estimation = (
                len(self.assimilated_data.times) >= self.start_param
            )

        # Optionally exclude parameters from the analysis update. The forecast
        # parameters are kept aside and re-appended below, so that the analysis
        # written to history always has the model's full Nphi+Na rows.
        Af_params = None
        if self.Na > 0 and not self.activate_parameter_estimation:
            Af_params = Af_state[self.Nphi:self.Nphi + self.Na, :].copy()
            Af_state = Af_state[:self.Nphi, :]

        # Append observables to form augmented state.
        y = self.model.get_observables()       # (Nq, m)
        Af_aug = np.vstack((Af_state, y))      # (Nphi+Na+Nq, m)  or  (Nphi+Nq, m)
        Af_fc = Af_aug                         # the forecast without the prior inflation

        Aa = None
        labels = np.zeros(Af_aug.shape[1], dtype=int)
        if self.mixture > 1:                   # the component of each member
            labels = self.mixture_labels(Af_state)
            self.mixture_history.times.append(self.current_time)
            self.mixture_history.labels.append(labels)

        # ── Prior inflation from the innovation (Li, Kalnay & Miyoshi 2009) ──
        # The analysis starts from the inflated forecast; a mixture inflates each component about its own mean.
        if self.innovation_inflation:
            rho = self._innovation_inflation_factor(Af_aug, d, Cdd)
            Af_aug = self._inflate_components(Af_aug, self._inflation_rows(rho, len(Af_aug), Af_params), labels)
            self.inflation_history.times.append(self.current_time)
            self.inflation_history.factors.append(rho)

        if labels.max() > 0:                   # one analysis per component (romda.estimators.sequential.mixture)
            from romda.estimators.sequential.mixture import mixture_update
            Aa = mixture_update(self, Af_aug, Af_fc, labels, d, Cdd, Af_params)
        elif self.mixture > 1:
            m = Af_aug.shape[1]
            self.mixture_history.weights_f.append(np.ones(1))
            self.mixture_history.weights_a.append(np.ones(1))
            self.mixture_history.counts.append(np.array([m]))
        if Aa is None:
            Aa = self._gaussian_update(Af_aug, Af_fc, d, Cdd, Af_params)

        # ── Update model history ─────────────────────────────────────────────
        # Only the state rows go to the model; the observables are derived.
        Aa_psi = Aa[:Af_state.shape[0], :]
        if Af_params is not None:
            Aa_psi = np.vstack((Aa_psi, Af_params))   # parameters left at their forecast

        self.update_history(
            Aa_psi,
            self.current_time,
            modify_saved_states=True,
        )
        self.assimilated_data = (d, self.current_time)

        # ── Update the bias estimator with the analysis innovation ───────────
        # i^a = d - q^a, evaluated on the *analysed* observables.
        if self.bias is not None:
            self._update_bias_from_analysis(d, self.model.get_observables())

        if return_analysis:
            # The analysed *state* rows, matching model.hist and the convention in
            # KalmanFilter.analysis_step. The trailing rows of Aa are the
            # augmented observables, which are derived rather than stored.
            return Aa_psi

    def _gaussian_update(self, Af_aug, Af_fc, d, Cdd, Af_params):
        """The analysis of the forecast `Af_aug` (N, m), with one Gaussian for the forecast: the kernel of the filter,
        the posterior inflation, the parameter checks, the rejection of the analysis (which keeps `Af_fc`, the
        forecast without the prior inflation) and the reservoir rows of an input-only update."""
        alpha_rows = slice(self.Nphi, self.Nphi + self.Na)

        # ── Call the subclass filter kernel ──────────────────────────────────
        Aa = self._analysis_kernel(Af_aug, d, Cdd)

        # ── Covariance inflation (Evensen 2009, Chap. 15) ────────────────────
        rho = self.inflation_factor
        if self.adaptive_inflation:
            from romda.estimators.inflation import adaptive_inflation_factor
            rho = adaptive_inflation_factor(self.member_space_transform(), rng=self.rng,
                                            n_rand=self.inflation_n_rand)
        if rho != 1.0:
            Aa = multiplicative_inflation(Aa, self._inflation_rows(rho, len(Aa), Af_params))
            self.inflation_history.times.append(self.current_time)
            self.inflation_history.factors.append(rho)

        # ── Parameter validity checks ─────────────────────────────────────────
        # Frozen parameters (Af_params set) are not in Aa: nothing to validate.
        if self.Na > 0 and Af_params is None and self.alpha_limits_matrix is not None:
            is_physical, idx_alpha = self.has_valid_params(Aa[alpha_rows], self.alpha_limits_matrix)
            if not is_physical:
                self._reject(self.current_time, f'Non-physical parameters at indices {idx_alpha}')
                Aa = multiplicative_inflation(Af_aug, self._inflation_rows(self.inflation_factor_rejection,
                                                                           len(Af_aug), Af_params))
                if self.inflate_parameters and not self.has_valid_params(Aa[alpha_rows], self.alpha_limits_matrix)[0]:
                    # The inflation itself takes a parameter out of its limits: keep the forecast,
                    # as the original rBA-EnKF code does (DA.analysisStep). Otherwise every further
                    # rejection inflates the ensemble again, and the spread grows without bound.
                    Aa = Af_aug.copy()

        # ── State rejection: the analysis moves the mean of the physical state too far ───
        if self.max_mean_shift is not None:
            rows = self._physical_rows()
            shift = np.linalg.norm(Aa[rows].mean(1) - Af_fc[rows].mean(1)) / np.linalg.norm(self.model_climatology())
            if shift > self.max_mean_shift:
                self._reject(self.current_time, f'State mean moves {shift:.2f} climatological std')
                Aa = Af_fc.copy()                      # the forecast, without the prior inflation

        # An EnKF-type gain corrects each state row from its own covariance with y,
        # independent of what other rows are present -- so leaving the reservoir out
        # of the analysis is the same as running it and discarding its correction.
        if not self.update_reservoir:
            N_units = _reservoir_size(self.model)
            if N_units > 0:
                reservoir_rows = slice(self.model.N_dim, self.model.N_dim + N_units)
                Aa[reservoir_rows, :] = Af_aug[reservoir_rows, :]
        return Aa

    def mixture_labels(self, Af_state: np.ndarray) -> np.ndarray:
        """Component of each member of the forecast `Af_state` (N, m), from the Gaussian mixture fitted to its physical
        rows in units of their climatological standard deviation (`model_climatology`;
        `romda.estimators.sequential.mixture.fit_mixture`). The fit is computed once per analysis time, so that an
        inflation applied before `analysis_step` (e.g. by a harness) and the analysis use the same components."""
        from romda.estimators.sequential.mixture import fit_mixture
        if self._mixture_cache is not None and self._mixture_cache[0] == self.current_time:
            return self._mixture_cache[1]
        labels, _ = fit_mixture(Af_state[self._physical_rows()], self.mixture, self.rng, self.mixture_min_members,
                                scale=self.model_climatology())
        self._mixture_cache = (self.current_time, labels)
        return labels

    @staticmethod
    def _inflate_components(A: np.ndarray, rho, labels: np.ndarray) -> np.ndarray:
        """Multiplicative inflation of each component of the ensemble `A` (N, m) about its own mean."""
        out = A.copy()
        for k in range(labels.max() + 1):
            cols = labels == k
            out[:, cols] = multiplicative_inflation(A[:, cols], rho)
        return out

    _inflation_moments = None   # running means of lambda~ tr(C_yy) and tr(C_yy), inflation='innovation'

    def _innovation_inflation_factor(self, Af: np.ndarray, d: np.ndarray, Cdd: np.ndarray) -> float:
        """Deviation factor sqrt(lambda) of ``inflation='innovation'`` for the forecast `Af`.

        The innovation and the predicted covariance are those of the analysis that follows (see
        `innovation_inflation_factor`): the bias-corrected innovation at the forecast mean with
        ``(I + J)``-mapped deviations. The estimate is averaged over the analyses with the weight
        ``inflation_rate``, and the returned variance factor is bounded by one and ``inflation_max``
        squared.
        """
        from romda.estimators.inflation import innovation_inflation_factor
        d, Cdd = np.atleast_1d(d), np.atleast_2d(Cdd)
        Y = self._MA(Af)
        y = Y.mean(1, keepdims=True)
        S, nu = Y - y, d[:, None] - y
        if self._bias_active():
            b, J, d = self._bias_terms(y.ravel(), d)
            S, nu = (np.eye(len(d)) + J) @ S, d[:, None] - y - b
        Cyy = S @ S.T / (Af.shape[1] - 1)
        weight = np.trace(Cyy)
        upper = np.inf if self.inflation_max is None else self.inflation_max**2
        if self.inflation_average == 'plain':
            if self.inflation_guard and weight > np.trace(Cdd):
                one = 1.0
            elif weight > 0:
                one = float(np.clip(innovation_inflation_factor(nu, Cdd, Cyy), 1.0, upper))
            else:                               # no spread at the sensors: the limit of the ratio, without 0/0
                one = upper if np.ravel(nu) @ np.ravel(nu) > np.trace(Cdd) else 1.0
            lam = 1.0 if self._inflation_moments is None else self._inflation_moments
            self._inflation_moments = (1.0 - self.inflation_rate) * lam + self.inflation_rate * one
            return float(np.sqrt(self._inflation_moments))
        if self.inflation_average != 'weighted':
            raise ValueError(f"inflation_average={self.inflation_average!r}: expected 'weighted' or 'plain'.")
        # Running mean of lambda~ weighted by tr C_yy, i.e. the ratio of the running means of
        # nu^T nu - tr C_dd and tr C_yy, from lambda = 1: the one-sample estimate is noisiest when the
        # spread is small, and its plain mean is biased upward (lambda ~ 1.1 on a perfect model).
        # The bounds act on the factor it returns, not on the mean, which a lower bound would bias too.
        # With no spread at the sensors, lambda~ is 0/0: its weighted term is the numerator itself.
        excess = (innovation_inflation_factor(nu, Cdd, Cyy) * weight if weight > 0
                  else np.ravel(nu) @ np.ravel(nu) - np.trace(Cdd))
        if self.inflation_guard and weight > np.trace(Cdd):
            excess = weight                     # lambda~ = 1 before the ensemble synchronizes
        if self._inflation_moments is None:
            self._inflation_moments = np.array([weight, weight])
        r = self.inflation_rate
        self._inflation_moments = (1.0 - r) * self._inflation_moments + r * np.array([excess, weight])
        if self._inflation_moments[1] <= 0:     # no spread yet: a factor would scale zero deviations
            return 1.0
        return float(np.sqrt(np.clip(self._inflation_moments[0] / self._inflation_moments[1], 1.0, upper)))

    def __getstate__(self):
        # The lazy member-space transform is a local closure, which pickle refuses; it is rebuilt by the next analysis.
        state = self.__dict__.copy()
        state.pop('_analysis_transform', None)
        return state

    def member_space_transform(self):
        """The member-space transform X of the last analysis, Aa = Af X, an m x m matrix.

        The kernels build it on demand, because the analysis does not need it.
        """
        X = getattr(self, '_analysis_transform', None)
        return X() if callable(X) else X

    def _perturbed_observations(self, d: np.ndarray, Cdd: np.ndarray, m: int) -> np.ndarray:
        """Perturbed-observation ensemble, ``d 1^T + chol(Cdd) E`` with ``E ~ N(0, I)`` of shape (Nq, m)."""
        return d[:, None] + cholesky(Cdd) @ self.rng.standard_normal((len(d), m))

    @staticmethod
    def _real_or_forecast(Aa: np.ndarray, Af: np.ndarray) -> np.ndarray:
        """`Aa`, or the forecast `Af` when a real-state forecast gave a complex analysis.

        Complex-state models (e.g. KS Fourier modes) legitimately produce a
        complex analysis; the realness guard only applies to real-state models.
        """
        return Aa if np.iscomplexobj(Af) or np.isreal(Aa).all() else Af

    # ── Shared bias-aware covariance-form kernel ──────────────────────────────

    def _bias_covariance_kernel(self, Af: np.ndarray, d: np.ndarray, Cdd: np.ndarray, gamma: float) -> np.ndarray:
        r"""Bias-aware stochastic update of Nóvoa, Racca & Magri (2023).

        The forecast observables are corrected with the current bias estimate,
        $\matr{Y} = \matr{M}\matr{\Psi}^\mathrm{f} + \matr{B}$, and the
        gain carries the bias Jacobian $\matr{J}$ and the regularization
        factor $\gamma$; ``gamma = 0`` is the un-regularized bias-aware update
        that any filter applies when configured with a `Bias` instance, and
        ``gamma > 0`` is `rEnKF`.
        """
        m   = Af.shape[1]
        d   = np.atleast_1d(d)
        Cdd = np.atleast_2d(Cdd)
        Nq  = len(d)

        MEf = self._MA(Af)
        b, J, d = self._bias_terms(np.mean(MEf, axis=-1), d)

        Y   = MEf + b
        D   = self._perturbed_observations(d, Cdd, m)

        Af_p  = Af - Af.mean(1, keepdims=True)
        MEf_p = MEf - MEf.mean(1, keepdims=True)

        Cpp_xy = Af_p @ MEf_p.T / (m - 1)
        Cqq    = MEf_p @ MEf_p.T / (m - 1)

        # Cbb is fixed to Cdd, so the weight W = Cdd @ inv(Cbb) is the identity
        # and drops out of both the covariance and the update.
        IpJ = np.eye(Nq) + J
        Cyy = Cdd + IpJ.T @ IpJ @ Cqq + gamma * J.T @ J @ Cqq

        Cyy_inv = inv(Cyy)
        K  = Cpp_xy @ Cyy_inv
        G  = IpJ.T @ (D - Y) - gamma * (J.T @ b)   # (Nq, m)
        Aa = Af + K @ G

        # Member-space transform Aa = Af X_hat (for the adaptive inflation):
        # K = Af' MEf'.T inv(Cyy) / (m-1) and Af' = Af (I - 11.T/m), so
        # Aa = Af + K G = Af [I + (I - 11.T/m) MEf'.T inv(Cyy) G / (m-1)],
        # for any Cyy (the scaling change enters through Cyy_inv alone).
        def transform():
            ones_m = np.ones((m, m)) / m
            return np.eye(m) + (np.eye(m) - ones_m) @ (MEf_p.T @ Cyy_inv @ G) / (m - 1)

        self._analysis_transform = transform
        return self._real_or_forecast(Aa, Af)

    # ── Abstract filter kernel (implemented by every ensemble filter) ─────

    @abstractmethod
    def _analysis_kernel(self, Af: np.ndarray, d: np.ndarray, Cdd: np.ndarray) -> np.ndarray:
        """Pure mathematical filter update.

        Parameters
        ----------
        Af  : (N, m)    augmented forecast ensemble ``[phi; alpha; q]``
        d   : (Nq,)     observation vector
        Cdd : (Nq, Nq)  observation noise covariance

        Returns
        -------
        Aa : (N, m)  analysis ensemble
        """

    # ── Ensemble utilities ────────────────────────────────────────────────────

    @staticmethod
    def has_valid_params(A_alpha: np.ndarray, alpha_limits_matrix: np.ndarray) -> tuple[bool, list[int]]:
        """Whether every parameter member lies within its bounds, and the indices of those that do not.

        Parameters
        ----------
        A_alpha : (Na, m)
        alpha_limits_matrix : (2, Na, 1)  rows are [lower_bounds, upper_bounds]
        """
        low_limits, high_limits = alpha_limits_matrix
        oob = np.any((A_alpha < low_limits) | (A_alpha > high_limits), axis=1)
        return not np.any(oob), np.where(oob)[0].tolist()

    @cached_property
    def alpha_limits_matrix(self) -> np.ndarray | None:
        """Parameter bounds, shape (2, Na, 1), in the row order of `est_alpha`; None if all unbounded."""
        lims = self.model.alpha_lims
        alpha_lims = np.array(
            [[-np.inf if lo is None else lo, np.inf if hi is None else hi]
             for lo, hi in (lims.get(key, (None, None)) for key in self.model.est_alpha)],
            dtype=float,
        ).T  # (2, Na)
        return None if np.all(np.isinf(alpha_lims)) else alpha_lims[:, :, np.newaxis]

    def _physical_rows(self):
        """The rows of the physical state: all of Nphi, but the reservoir of an ESN."""
        return slice(0, self.Nphi - getattr(self.model, 'N_units', 0))

    def _inflation_rows(self, rho, n_rows, Af_params):
        """Row factors (n_rows, 1) of the inflation: `rho`, and 1 on the parameter rows (unless
        `inflate_parameters`) and on the reservoir rows that the analysis leaves at their forecast."""
        rho_f = np.full((n_rows, 1), float(rho))
        # In the state-only warm-up (Af_params set), the rows after Nphi are the observables.
        if not self.inflate_parameters and Af_params is None:
            rho_f[self.Nphi:self.Nphi + self.Na] = 1.0
        N_units = 0 if self.update_reservoir else _reservoir_size(self.model)
        if N_units > 0:
            rho_f[self.model.N_dim:self.model.N_dim + N_units] = 1.0
        return rho_f

    def model_climatology(self):
        """Standard deviation of each physical state component (`climatology`), by default from a free run of
        a copy of the model from the current ensemble: 10 ``t_transient`` long, as dynamodels'
        ``create_long_timeseries`` (2000 steps for a model without a transient), at most 5000 steps, the first
        tenth dropped, over the time steps and the members. Computed once."""
        if self.climatology is None:
            model = self.model.copy()
            t_transient = getattr(model, 't_transient', 0.)
            n = int(10 * t_transient / model.dt) if t_transient > 0 else 2000
            n = min(n, 5000)        # ponytail: fixed cap (an ESN's t_transient is its training washout); a keyword if needed
            psi, _ = model.time_integrate(Nt=n)                            # (n, rows, m)
            self.climatology = psi[n // 10:, self._physical_rows()].std(axis=(0, 2))
            if hasattr(model, 'close'):
                model.close()
        return np.asarray(self.climatology, dtype=float)

    def _reject(self, time, reason: str) -> None:
        """Log a rejected analysis (`rejected_analysis`) and its running count (debug log)."""
        self.rejected_analysis.times.append(time)
        self.rejected_analysis.reasons.append(reason)
        n_total = len(self.assimilated_data.times) + 1
        logger.debug('Non-physical analysis: %d/%d (%s)', len(self.rejected_analysis.times), n_total, reason)

    # ── Printing and plotting ─────────────────────────────────────────────────

    def print_parameters(self, indent: int = 0) -> None:
        super().print_parameters(indent=indent)
        print(f'{" " * indent}=== Model parameters: ===')
        self.model.print_parameters(show_header=False, indent=indent*2)
        print(f'{" " * indent}=== Bias parameters: ===')
        if self.bias is not None:
            self.bias.print_bias_parameters(indent=indent*2)
        else:
            print(f'{" " * indent*2}No bias model.')

    def visualize_history(self, **kwargs) -> None:
        """Plot observable and parameter histories."""
        kwargs_obs = allowed_kwargs_for_func(plot_observable_history, kwargs)
        plot_observable_history(ensemble=self, **kwargs_obs)
        if self.Na > 0:
            kwargs_alpha = allowed_kwargs_for_func(plot_alpha_history, kwargs)
            plot_alpha_history(ensemble=self, **kwargs_alpha)

    def visualize_state(self, **kwargs) -> None:
        """Plot ensemble state distributions."""
        func = self.model.visualize_state
        kwargs_state = allowed_kwargs_for_func(func, kwargs)
        func(**kwargs_state)


# ══════════════════════════════════════════════════════════════════════════════
# EnsembleFilter — the ensemble Kalman filter, by update type
# ══════════════════════════════════════════════════════════════════════════════

#: The updates of biasda's EnsembleFilter that romda leaves to biasda.
_BIASDA_TYPES = {k: k for k in ('deterministic', 'sqrtrandom', 'sqrtserial', 'sqrthalf')}
#: Accepted ``type`` names (lower case, alphanumerics only), with those of biasda.
_TYPES = {'stochastic': 'stochastic', 'sqrt': 'sqrt'} | _BIASDA_TYPES


class EnsembleFilter(EnsembleEstimator):
    r"""Ensemble Kalman filter, configured by its analysis update (``type``), as ``biasda.EnsembleFilter``.

    Writing $\matr{\Psi}' = \matr{\Psi}^\mathrm{f} - \bar{\vect{\psi}}^\mathrm{f}$
    for the forecast-ensemble deviations and ${\matr{Q}'} = \matr{M}\matr{\Psi}'$
    for the mapped deviations, the ensemble Kalman gain is

    $$
    \matr{K} = \matr{\Psi}'{\matr{Q}'}^\mathrm{T}\matr{C}^{-1}, \qquad
    \matr{C} = (m-1)\,\matr{C}_{dd} + {\matr{Q}'}{\matr{Q}'}^\mathrm{T},
    $$

    which is the Kalman gain $\matr{C}_{\psi q}(\matr{C}_{qq} + \matr{C}_{dd})^{-1}$
    with the sample covariances.

    ``type='stochastic'`` (default). Each member assimilates a
    perturbed copy of the observation,
    $\matr{D} = \vect{d}\vect{1}^\mathrm{T} + \mathrm{chol}(\matr{C}_{dd})\,\vect{\varepsilon}$
    with $\vect{\varepsilon} \sim \mathcal{N}(\vect{0}, \mathbb{I})$, so that

    $$
    \matr{\Psi}^\mathrm{a} = \matr{\Psi}^\mathrm{f}
    + \matr{K}\left(\matr{D} - \matr{M}\matr{\Psi}^\mathrm{f}\right).
    $$

    ``type='sqrt'``. The square-root update (Tippett et al. 2003): the gain updates the mean,
    and the symmetric square root of
    $\matr{T} = \mathbb{I}_m - {\matr{Q}'}^\mathrm{T}\matr{C}^{-1}{\matr{Q}'}$ updates
    the deviations, ${\matr{\Psi}'}^\mathrm{a} = \matr{\Psi}'\,\matr{T}^{1/2}$, with no
    perturbed observations (see `_sqrt_factors`); this is the ensemble transform Kalman filter
    (ETKF; Bishop et al. 2001; Hunt et al. 2007).

    ``parameter_estimation`` matches the model: None (default) when it estimates no
    parameters, and ``'joint'`` when it does (``est_alpha`` non-empty); a mismatch raises
    ValueError. ``'joint'`` updates the augmented state
    $[\vect{\phi}; \vect{\alpha}; \vect{q}]$ in one analysis.

    A bias estimator (``parent_bias``, any `Bias` other than `NoBias`) makes every update
    bias-aware. With $\vect{b}$ the current bias estimate, its Jacobian
    $\matr{J} = \mathrm{d}\vect{b}/\mathrm{d}(\matr{M}\vect{\psi})$, the
    bias-corrected forecast observables
    $\matr{Y} = \matr{M}\matr{\Psi}^\mathrm{f} + \vect{b}\vect{1}^\mathrm{T}$, and
    the bias-regularization factor $\gamma \ge 0$ (``bias_regularization``), the stochastic
    update employs the regularized bias-aware gain (r-EnKF),

    $$
    \matr{C}_{yy} = \matr{C}_{dd}
    + \left[(\mathbb{I}+\matr{J})^\mathrm{T}(\mathbb{I}+\matr{J})
    + \gamma\, \matr{J}^\mathrm{T}\matr{J}\right]\matr{C}_{qq},
    \qquad
    \matr{\Psi}^\mathrm{a} = \matr{\Psi}^\mathrm{f}
    + \matr{C}_{\psi q}\matr{C}_{yy}^{-1}
    \left[(\mathbb{I}+\matr{J})^\mathrm{T}(\matr{D} - \matr{Y})
    - \gamma\, \matr{J}^\mathrm{T}\vect{b}\vect{1}^\mathrm{T}\right],
    $$

    where $\matr{C}_{\psi q}$ and $\matr{C}_{qq}$ are the sample forecast covariances of
    the state and of the mapped observables. The bias covariance is fixed to
    $\matr{C}_{bb} = \matr{C}_{dd}$, so the weight $\matr{C}_{dd}\matr{C}_{bb}^{-1}$ of
    Nóvoa, Racca & Magri (2023) is the identity; with $\gamma = 0$ and $\vect{b} = \vect{0}$,
    the update is the bias-blind update. The square-root update assimilates instead the
    pseudo-observation of zero bias, $\vect{0} = \sqrt{\gamma}\,\vect{b}$, stacked under the
    observation. If the observations are biased (`Bias.biased_observations`), $\vect{d}$ is
    shifted by the mean bias-innovation gap, and the first ``start_bias`` analyses are bias-blind.

    The other updates, schemes and options of ``biasda.EnsembleFilter`` (``'deterministic'``,
    ``'sqrt-random'``, ``'sqrt-serial'``, ``'sqrt-half'``, ``parameter_estimation='dual'``,
    ``bias_estimation`` and ``localization``) live in biasda and raise NotImplementedError here.

    The exception is ``localization`` (a Gaspari-Cohn half-radius) with ``type='sqrt'``, the local ensemble transform
    Kalman filter (LETKF; Hunt et al. 2007): every augmented row takes its own square-root transform, with
    $\matr{C}_{dd}^{-1}$ tapered by the distance between the row and each observable
    (`_local_transform`). ``state_positions`` (default: the index of each physical component),
    ``obs_positions`` (required unless every component is observed) and ``loc_period`` (a ring) set the
    distances; the parameter rows are not localized.

    Parameters
    ----------
    type : str, optional
        Analysis update: ``'stochastic'`` (default) or ``'sqrt'``. Case, hyphens, underscores and
        spaces are ignored.
    parameter_estimation : None or 'joint', optional
        None (default) for a model without ``est_alpha``, ``'joint'`` for the augmented-state analysis.
    parent_bias : Bias instance, Bias subclass or None, optional
        Bias estimator; any `Bias` other than `NoBias` makes the analysis bias-aware.
    inflation : float, optional
        Multiplicative inflation factor (default 1.0).
        ``'adaptive'`` estimates the factor at every analysis.
        ``'innovation'`` inflates the forecast deviations, before the analysis, by
        $\sqrt{\lambda}$, with the variance factor $\lambda$ estimated from the bias-corrected
        innovation (Li, Kalnay & Miyoshi 2009; `romda.estimators.inflation.innovation_inflation_factor`).
        ``inflation_rate`` (default 0.1) is the weight of every new estimate in $\lambda$, which is
        bounded below by one and above by ``inflation_max`` squared (default None, no bound).
        ``inflation_history`` records the deviation factors, $\sqrt{\lambda}$ for ``'innovation'``.

    Other Parameters
    ----------------
    bias_regularization : float or None, optional
        Bias-regularization factor $\gamma$. None (default) for a bias-unaware filter or the
        un-regularized update ($\gamma = 0$); a float requires a bias, passed as ``parent_bias`` or
        attached before the first analysis (``estimator.bias = ...``, as the ESN bias of the
        thermoacoustic mains is).

    References
    ----------
    Evensen (2003). The ensemble Kalman filter: theoretical formulation and practical
    implementation. *Ocean Dynamics*, 53, 343-367.

    Tippett, Anderson, Bishop, Hamill & Whitaker (2003). Ensemble square root
    filters. *Mon. Wea. Rev.*, 131, 1485-1490.

    Bishop, Etherton & Majumdar (2001). Adaptive sampling with the ensemble transform
    Kalman filter. Part I: theoretical aspects. *Mon. Wea. Rev.*, 129, 420-436.

    Hunt, Kostelich & Szunyogh (2007). Efficient data assimilation for
    spatiotemporal chaos: a local ensemble transform Kalman filter. *Physica D*,
    230, 112-126.

    Nóvoa, Racca & Magri (2023). Inferring unknown unknowns: Regularized bias-aware
    ensemble Kalman filter. *Comput. Methods Appl. Mech. Eng.*, 418, 116502.
    [DOI: 10.1016/j.cma.2023.116502](https://doi.org/10.1016/j.cma.2023.116502).

    Li, Kalnay & Miyoshi (2009). Simultaneous estimation of covariance inflation and observation
    errors within an ensemble Kalman filter. *Q. J. R. Meteorol. Soc.*, 135, 523-533.
    """

    type: str = 'stochastic'
    parameter_estimation: str | None = None
    _localization = True
    #: The named entry points (EnKF, EnSRKF, ...) estimate the parameters jointly when the
    #: model has ``est_alpha``; `EnsembleFilter` asks for ``parameter_estimation`` explicitly.
    _infer_parameter_estimation = False

    def __init__(self, parent_model, parent_bias=None, *, type: Literal['stochastic', 'sqrt'] | None = None,
                 parameter_estimation: Literal['joint'] | None = None,
                 inflation: float | Literal['adaptive', 'innovation'] | None = None, **kwargs):
        type = canonical_name(self.type if type is None else type, _TYPES)
        if type in _BIASDA_TYPES:
            raise NotImplementedError(f"type={type!r} is an update of biasda.EnsembleFilter; romda "
                                      "has 'stochastic' and 'sqrt'.")
        kwargs.update(given(type=type, parameter_estimation=parameter_estimation, inflation=inflation))
        super().__init__(parent_model, parent_bias, **kwargs)
        if self.loc_radius is not None:
            # romda localizes the square-root update only (the LETKF); the localized stochastic update, the
            # bias-aware local updates and the adaptive factor, which needs one member-space transform, are biasda's.
            if self.type != 'sqrt':
                raise NotImplementedError("localization= with type='stochastic' is an option of biasda.EnsembleFilter; "
                                          "romda localizes type='sqrt' (the LETKF).")
            biasda_only = self.bias is not None and not isinstance(self.bias, NoBias)
            biasda_only = biasda_only or self.adaptive_inflation
            if biasda_only:
                raise NotImplementedError("localization= with inflation='adaptive' or a bias is an option of "
                                          "biasda.EnsembleFilter.")
        if self._infer_parameter_estimation and self.parameter_estimation is None and self.Na > 0:
            self.parameter_estimation = 'joint'
        self._check_parameter_estimation()

    # ── Analysis kernels ──────────────────────────────────────────────────────

    def _analysis_kernel(self, Af, d, Cdd):
        """Analysis of every row of `Af` with the update of `type`."""
        if self.type == 'sqrt':
            return self._sqrt_kernel(Af, d, Cdd)
        # Bias-aware assimilation is a configuration, not a subclass: with a
        # Bias instance set, apply the bias-corrected update (gamma = 0 unless
        # `bias_regularization` is set) after the bias-blind warm-up window
        # of `start_bias` analyses.
        if self._bias_active():
            return self._bias_covariance_kernel(Af, d, Cdd, gamma=self.gamma)
        return self._plain_kernel(Af, d, Cdd)

    def _plain_kernel(self, Af: np.ndarray, d: np.ndarray, Cdd: np.ndarray) -> np.ndarray:
        m   = Af.shape[1]
        d   = np.atleast_1d(d)
        Cdd = np.atleast_2d(Cdd)

        # Mapped forecast matrix M(Af) and mapped deviations M(Af')
        Psi_f   = Af - Af.mean(1, keepdims=True)
        S       = self._MA(Psi_f)
        Y       = self._MA(Af)
        D       = self._perturbed_observations(d, Cdd, m)

        C = (m - 1) * Cdd + np.dot(S, S.T)
        G = inv(C) @ (D - Y)                                  # (Nq, m)

        # Aa = Af (I + S^T G) = Af + (Af S^T) G, in O(N m N_q) without an m x m matrix.
        # The member-space transform I + S^T G is built only when the adaptive inflation
        # asks for it, to push a random ensemble through the SAME update
        # ([Aa; Ba] = [Af; Bf] X, Evensen 2009, Eq. 15.1).
        self._analysis_transform = lambda: np.eye(m) + S.T @ G
        return self._real_or_forecast(Af + (Af @ S.T) @ G, Af)

    def _sqrt_kernel(self, Af, d, Cdd):
        d   = np.atleast_1d(d)
        Cdd = np.atleast_2d(Cdd)

        psi_f_m = np.mean(Af, 1, keepdims=True)   # (N, 1)
        Psi_f   = Af - psi_f_m                    # (N, m)

        y = self._MA(psi_f_m)   # (Nq, 1)
        S = self._MA(Psi_f)     # (Nq, m)

        S_eff, z, Cdd_eff = self._stacked_innovation(S, y, d, Cdd)

        if self.loc_radius is not None:
            self._analysis_transform = None
            return self._real_or_forecast(self._local_transform(psi_f_m, Psi_f, S_eff, z, Cdd_eff, S.shape[0]), Af)

        w, sqrt_T, self._analysis_transform = _sqrt_factors(S_eff, z, Cdd_eff)
        psi_a_m = psi_f_m + Psi_f @ w       # analysed mean: the Kalman update of the forecast mean
        Psi_a = sqrt_T(Psi_f)               # analysed deviations: Psi_f T^{1/2}
        return self._real_or_forecast(psi_a_m + Psi_a, Af)

    def _loc_matrices(self, N_rows: int, Nq: int) -> tuple[np.ndarray, np.ndarray]:
        """Gaspari-Cohn localization matrices of the augmented rows, ``(L_xy, L_yy)``, as in biasda."""
        pos_x, pos_q = self._positions(self.Nphi, Nq)
        L_xy = np.ones((N_rows, Nq))
        L_xy[:self.Nphi, :] = localization_matrix(pos_x, pos_q, self.loc_radius, period=self.loc_period)
        # Parameter rows (between Nphi and N_rows - Nq) and state rows with no position (nan, e.g. a reservoir unit fed
        # by the bias alone) keep 1: no position, no localization.
        L_xy[:self.Nphi][np.isnan(np.asarray(pos_x, dtype=float))] = 1.0
        L_yy = localization_matrix(pos_q, pos_q, self.loc_radius, period=self.loc_period)
        L_xy[N_rows - Nq:, :] = L_yy
        return L_xy, L_yy

    def _local_transform(self, psi_f_m, Psi_f, S_eff, z, Cdd_eff, Nq):
        r"""LETKF analysis (Hunt et al. 2007): one ensemble transform per augmented row, as in biasda.

        Row $i$ assimilates the data with $\matr{C}_{dd}^{-1}$ tapered by the square root of its row of
        $\matr{L}_{\psi q}$, $\matr{R}_i^{-1} = \matr{D}_i^{1/2}\matr{C}_{dd}^{-1}\matr{D}_i^{1/2}$. With
        $\matr{V}\matr{\Lambda}\matr{V}^\mathrm{T}$ the eigendecomposition of
        $(m-1)\mathbb{I} + {\matr{Q}'}^\mathrm{T}\matr{R}_i^{-1}\matr{Q}'$, the mean weights are
        $\vect{w}_i = \matr{V}\matr{\Lambda}^{-1}\matr{V}^\mathrm{T}{\matr{Q}'}^\mathrm{T}\matr{R}_i^{-1}\vect{\nu}$
        and the deviation transform is $\matr{T}_i^{1/2} = \matr{V}((m-1)\matr{\Lambda}^{-1})^{1/2}\matr{V}^\mathrm{T}$.
        With no taper, every row takes the global square-root update of `_sqrt_factors`. Rows with the same taper
        have the same transform, which is computed once: the state of an ESN has many rows per site (the reservoir
        of a parallel network has one position per patch). `_row_transforms` holds the distinct transforms and
        `_row_index` the one of each row, ``T_i = _row_transforms[_row_index[i]]``.
        """
        m = Psi_f.shape[1]
        L_xy, _ = self._loc_matrices(Psi_f.shape[0], Nq)
        L_xy, index = np.unique(L_xy, axis=0, return_inverse=True)
        sqrt_rho = np.sqrt(np.tile(L_xy, (1, S_eff.shape[0] // Nq)))
        Cdd_inv = inv(Cdd_eff)
        ones_m = np.full((m, m), 1.0 / m)
        T = np.empty((len(L_xy), m, m))
        for i, r in enumerate(sqrt_rho):
            R_inv = r[:, None] * Cdd_inv * r[None, :]
            ev, V = np.linalg.eigh((m - 1) * np.eye(m) + S_eff.T @ R_inv @ S_eff)
            w = V @ ((V.T @ (S_eff.T @ R_inv @ z)) / ev[:, None])         # (m, 1)
            W = V @ np.diag(np.sqrt((m - 1) / ev)) @ V.T                  # T^{1/2}
            T[i] = ones_m + (np.eye(m) - ones_m) @ (w + W)
        self._row_transforms, self._row_index = T, index.ravel()
        return np.einsum('im,imn->in', psi_f_m + Psi_f, T[self._row_index])


# ══════════════════════════════════════════════════════════════════════════════
# Named entry points
# ══════════════════════════════════════════════════════════════════════════════

class EnKF(EnsembleFilter):
    """Stochastic ensemble Kalman filter, `EnsembleFilter` with ``type='stochastic'``.

    The parameters of a model with ``est_alpha`` are estimated jointly by default.
    """

    _infer_parameter_estimation = True


class EnSRKF(EnsembleFilter):
    """Ensemble square-root Kalman filter, `EnsembleFilter` with ``type='sqrt'`` (ETKF).

    The parameters of a model with ``est_alpha`` are estimated jointly by default.
    """

    type = 'sqrt'
    _infer_parameter_estimation = True


class rEnKF(EnKF):  # noqa: N801
    """Regularized bias-aware ensemble Kalman filter (r-EnKF), `EnKF` with ``parent_bias`` and ``bias_regularization=1``.

    `EnsembleFilter` gives the update.

    References
    ----------
    Nóvoa, Racca & Magri (2023). Inferring unknown unknowns: Regularized bias-aware
    ensemble Kalman filter. *Comput. Methods Appl. Mech. Eng.*, 418, 116502.
    """

    def __init__(self, parent_model, parent_bias=None, **kwargs):
        if 'gamma' in kwargs:
            raise TypeError("gamma= is not a keyword argument; use bias_regularization.")
        kwargs.setdefault('bias_regularization', 1.0)
        super().__init__(parent_model, parent_bias, **kwargs)


class rBA_EnKF(rEnKF):  # noqa: N801
    """The regularized bias-aware EnKF under its name in Nóvoa, Racca & Magri (2023): `rEnKF`."""


# ══════════════════════════════════════════════════════════════════════════════
# Auxiliary plotting functions  (used in visualisation methods)
# ══════════════════════════════════════════════════════════════════════════════

def plot_observable_history(
    ensemble: EnsembleEstimator,
    truth: Observations | None = None,
    plot_members: bool = False,
    reference_y=1.0,
    reference_t: float = 1.0,
    max_time=None,
    dims='all',
) -> None:
    C = Palette()
    pm = ensemble.model

    y_unbiased, y_model = ensemble.get_observable_hist()

    (y_unbiased, y_model), y_labels = normalized_y(
        reference_y, pm.obs_labels, y_unbiased, y_model
    )

    (t, t_margin), t_label = normalized_time(reference_t, pm.hist_t.copy(), np.array(pm.t_CR))

    t_obs = None
    if len(ensemble.assimilated_data.times) > 0:
        t_obs = normalized_time(reference_t, np.array(ensemble.assimilated_data.times))[0][0]
        max_time = max_time if max_time is not None else t[-1]
        min_time = t_obs[0] - 0.25 * t_margin
    else:
        min_time, max_time = t[0], t[-1]

    t, (y_model, y_unbiased) = cut_signals(t, y_model, y_unbiased, min_time=min_time, max_time=max_time)

    Nq = pm.Nq

    # Partial observation: map each model observable to its column in the observed
    # data. `ensemble.M` holds rows of the model's [0 | I] operator, so the position
    # of each row's 1 in the trailing Nq block is the model dim that row measures.
    # A callable M offers no mapping; assume full observation there.
    M = ensemble.M
    obs_col = {}
    if isinstance(M, np.ndarray) and M.ndim == 2:
        for j, row in enumerate(M):
            p = int(np.flatnonzero(row)[-1]) - (M.shape[1] - Nq)
            if 0 <= p < Nq:
                obs_col[p] = j
    else:
        obs_col = {qi: qi for qi in range(Nq)}

    # y-limits from the truth for every row when one is given (a diverging prediction
    # would squash the panels); dims without their own truth column get the truth's
    # global range. Only a truth-less call falls back to the model.
    y_margin = 0.15 * np.mean(abs(y_model), axis=(0, 2))
    max_y = np.max(y_model, axis=(0, 2))
    min_y = np.min(y_model, axis=(0, 2))

    if truth is not None:
        y_raw   = np.asarray(truth.y_raw.copy())
        y_true  = np.asarray(truth.y_true.copy())
        t_true  = normalized_time(reference_t, np.asarray(truth.t_true.copy()))[0][0]
        (y_raw, y_true), _ = normalized_y(reference_y, pm.obs_labels, y_raw, y_true)
        t_true, (y_raw, y_true) = cut_signals(t_true, y_raw, y_true, min_time=min_time, max_time=max_time)
        if len(t) != len(t_true):
            y_raw  = interpolate(t_true, y_raw,  t)
            y_true = interpolate(t_true, y_true, t)
        # truth arrays carry either every observable or exactly the measured ones,
        # in the operator's order
        truth_col = (lambda qi: qi) if y_true.shape[1] == Nq else obs_col.get
        y_margin = np.full(Nq, 0.15 * np.mean(abs(y_true)))
        max_y = np.full(Nq, np.max(y_true))
        min_y = np.full(Nq, np.min(y_true))
        for qi in range(Nq):
            jt = truth_col(qi)
            if jt is not None:
                y_margin[qi] = 0.15 * np.mean(abs(y_true[:, jt]))
                max_y[qi] = np.max(y_true[:, jt])
                min_y[qi] = np.min(y_true[:, jt])
    else:
        y_true, y_raw = None, None
        truth_col = obs_col.get

    if t_obs is not None:
        y_obs = normalized_y(reference_y, pm.obs_labels,
                             np.array(ensemble.assimilated_data.data)[..., np.newaxis])[0][0]

    dims_list: list[int] = (
        list(range(Nq)) if dims == 'all' else
        [dims] if isinstance(dims, int) else
        [int(d) for d in dims]
    )

    fig1 = plt.figure(figsize=(12, 2 * len(dims_list)), layout='constrained')
    ax_all = fig1.subplots(
        nrows=len(dims_list), ncols=3, sharey='row', sharex='col', width_ratios=[1, 1, 3]
    )
    if len(dims_list) == 1:
        ax_all = ax_all[np.newaxis, :]

    if t_obs is not None:
        x_lims = [
            [t_obs[0] - 0.25 * t_margin, t_obs[0] + 0.75 * t_margin],
            [t_obs[-1] - 0.25 * t_margin, min(t_obs[-1] + 0.75 * t_margin, max_time)],
            [t[0], max_time],
        ]
    else:
        x_lims = [
            [min_time, min_time + t_margin],
            [max_time - t_margin, max_time],
            [t[0], max_time],
        ]

    for row_i, qi in enumerate(dims_list):
        j = obs_col.get(qi)
        jt = truth_col(qi) if y_true is not None else None
        yl = [min_y[qi] - y_margin[qi], max_y[qi] + y_margin[qi]]
        for col_i, (ax, xl) in enumerate(zip(ax_all[row_i], x_lims)):
            if y_true is not None and jt is not None:
                ax.plot(t, y_true[:, jt, :], **C.true_props)
                if y_raw is not None:
                    ax.plot(t, y_raw[:, jt], **C.true_noisy_props)

            if isinstance(y_unbiased, np.ndarray) and y_unbiased.ndim == 3:
                if plot_members:
                    ax.plot(t, y_unbiased[:, qi, :], **C.y_unbias_props)
                else:
                    ax.plot(t, np.mean(y_unbiased[:, qi], axis=-1), **C.y_unbias_props)

            m_mean = np.mean(y_model[:, qi], axis=-1)
            ax.plot(t, m_mean, **C.y_biased_mean_props)

            # always show the ensemble spread; members are drawn on top when requested
            s = np.std(y_model[:, qi], axis=-1)
            ax.fill_between(t, m_mean + s, m_mean - s, color=C.get_color('BIASED', 0.5))
            if plot_members:
                ax.plot(t, y_model[:, qi, :], **C.y_biased_props)

            if t_obs is not None and j is not None:
                ax.plot(t_obs, y_obs[:, j], **C.obs_props)

            if col_i == 0:
                ax.set(ylabel=y_labels[qi])
                if row_i == 0 and any(ax_.get_legend_handles_labels()[1] for ax_ in fig1.axes):
                    fig1.legend(loc='center', bbox_to_anchor=(0.5, 1.05), ncol=6, frameon=False)
            ax.set(ylim=yl, xlim=xl)
            if row_i == len(dims_list) - 1:
                ax.set(xlabel=t_label)


def plot_alpha_history(
    ensemble: EnsembleEstimator,
    plot_members: bool = False,
    reference_a=None,
    reference_t: float = 1.0,
    max_time=None,
) -> None:
    pm = ensemble.model
    C = Palette()
    c1 = C.get_color_params(n=ensemble.Na, alpha=1)
    c2 = C.get_color_params(n=ensemble.Na, alpha=0.2)

    (t, t_margin), t_lbl = normalized_time(reference_t, pm.hist_t, np.array(pm.t_CR))
    hist_alpha, alpha_lbls = normalized_alpha(
        pm.hist[:, -pm.Na:],
        alpha_keys=ensemble.est_alpha,
        alpha_labels=pm.alpha_labels,
        reference_a=reference_a,
    )
    mean_alpha = np.mean(hist_alpha, axis=-1)
    std_alpha  = np.std(hist_alpha,  axis=-1)

    t_obs = np.array(ensemble.assimilated_data.times)
    if len(t_obs) > 0:
        t_obs = normalized_time(reference_t, t_obs)[0][0]
        max_time = max_time if max_time is not None else t[-1]
        min_time = t_obs[0] - 0.25 * t_margin
        x_lims = [
            [t_obs[0] - 0.25 * t_margin, t_obs[0] + 0.75 * t_margin],
            [t_obs[-1] - 0.25 * t_margin, min(t_obs[-1] + 0.75 * t_margin, max_time)],
            [min_time, max_time],
        ]
    else:
        min_time, max_time = t[0], t[-1]
        x_lims = [
            [min_time, min_time + t_margin],
            [max_time - t_margin, max_time],
            [min_time, max_time],
        ]

    fig = plt.figure(figsize=(12, 2 * ensemble.Na), layout='constrained')
    axs = fig.subplots(ensemble.Na, 3, sharex='col', sharey='row', width_ratios=[1, 1, 3])
    if ensemble.Na == 1:
        axs = [axs]

    for row_i, axs_row, p in zip(range(ensemble.Na), axs, ensemble.est_alpha):
        avg, s, all_h = [xx[:, row_i] for xx in [mean_alpha, std_alpha, hist_alpha]]
        for col_i, ax in enumerate(axs_row):
            if plot_members:
                ax.plot(t, all_h, color=c2[row_i], lw=1.0)
            ax.fill_between(t, avg + 2 * abs(s), avg - 2 * abs(s), alpha=0.2, color=c2[row_i])
            ax.plot(t, avg, color=c1[row_i], lw=2, dashes=(5, 1))
            if col_i == 0:
                ax.set(ylabel=alpha_lbls[p],
                       ylim=[min(avg) - 3 * max(s), max(avg) + 3 * max(s)])
            if row_i == ensemble.Na - 1:
                ax.set(xlabel=t_lbl, xlim=x_lims[col_i])
