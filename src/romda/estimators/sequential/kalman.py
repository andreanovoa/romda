"""The linear Kalman filter, as ``biasda.KalmanFilter`` with ``type='linear'``."""


from functools import cached_property
from typing import Literal

import numpy as np
from numpy.linalg import inv
from romda.estimators.base import SequentialEstimator
from romda.utils import canonical_name, given

__all__ = ['KalmanFilter']

#: Accepted ``type`` names (lower case, alphanumerics only); romda has 'linear', biasda the others.
_TYPES = {'linear': 'linear', 'extended': 'extended', 'unscented': 'unscented'}


class KalmanFilter(SequentialEstimator):
    r"""Linear Kalman filter, configured as ``biasda.KalmanFilter(type='linear')``.

    Unlike the ensemble filters, the Kalman filter owns an explicit mean $\vect{\psi}$ and
    covariance $\matr{C}_{\psi\psi}$; the (optional) model only advances the mean. With
    ``model=None``, the linear map `F` advances the mean, one application per step, and
    `current_time` counts steps. The analysis is

    $$
    \matr{K} = \matr{C}_{\psi y}\matr{C}_{yy}^{-1}, \qquad
    \vect{\psi}^\mathrm{a} = \vect{\psi}^\mathrm{f} + \matr{K}(\vect{d} - \bar{\vect{y}}), \qquad
    \matr{C}^\mathrm{a}_{\psi\psi} = \matr{C}^\mathrm{f}_{\psi\psi} - \matr{K}\matr{C}_{yy}\matr{K}^\mathrm{T},
    $$

    and the covariance follows a fixed one-step Jacobian,
    $\matr{C}^\mathrm{f}_{\psi\psi} = \matr{F}_\mathrm{jac}\matr{C}_{\psi\psi}\matr{F}_\mathrm{jac}^\mathrm{T}
    + \matr{Q}$ once per step, with $\matr{C}_{\psi y} = \matr{C}^\mathrm{f}_{\psi\psi}\matr{M}^\mathrm{T}$ and
    $\matr{C}_{yy} = \matr{M}\matr{C}^\mathrm{f}_{\psi\psi}\matr{M}^\mathrm{T} + \matr{C}_{dd}$.
    $\matr{F}_\mathrm{jac}$ defaults to `F`, to ``model.F``, or to the identity, in which case the
    covariance grows by $\matr{Q}$ only.

    A bias (``bias``, any `Bias` other than `NoBias`) corrects the predicted observables through its
    linearization, $\matr{H} = (\mathbb{I} + \matr{J})\matr{M}$; ``bias_regularization``
    $\gamma > 0$ stacks the pseudo-observation of zero bias, $\vect{0} = \sqrt{\gamma}\,\vect{b}$.

    ``parameter_estimation`` matches ``est_alpha``, the names of the trailing components of the state
    that are parameters: None when it is empty, ``'joint'`` otherwise (a mismatch raises ValueError).
    The forecast holds the parameters constant, so the propagated covariance carries their
    cross-covariance with the state, and the analysis updates the whole state at once. ``inflation``
    scales the analysis covariance by $\rho^2$ (the state block only with ``inflate_parameters=False``).

    The extended and unscented types, the dual scheme, localization and the Kalman smoother live in
    biasda (``biasda.KalmanFilter(type='extended' | 'unscented')``, ``biasda.KalmanSmoother``).

    Parameters
    ----------
    Cpp0 : ndarray, shape (N, N)
        Initial state covariance $\matr{C}_{\psi\psi,\,0|0}$.
    Q : ndarray, shape (N, N), optional
        Process-noise covariance added at every step (default zeros).
    type : str, optional
        ``'linear'`` (default; the only type in romda).
    parameter_estimation : None or 'joint', optional
        Parameter estimation, as above.
    F_jac : ndarray, optional
        One-step Jacobian of the forecast.
    **kwargs
        The arguments of `Estimator`: ``psi0``, ``Cdd``, ``M``, ``bias``, ``est_alpha`` and one of
        ``model`` and ``F`` (the forecast of the mean), and the shared configuration
        (``bias_regularization``, ``inflation``, ...); the sizes $N$ and $N_q$ are read from ``psi0``
        and ``Cdd``.

    References
    ----------
    Kalman (1960). A new approach to linear filtering and prediction problems.
    *J. Basic Eng.*, 82(1), 35-45.

    Evensen, Vossepoel & van Leeuwen (2022). *Data Assimilation Fundamentals.*
    Springer, Chap. 6 (the Kalman filter).
    """

    type: str = 'linear'
    parameter_estimation: str | None = None

    def __init__(
        self,
        Cpp0: np.ndarray,
        Q: np.ndarray | None = None,
        F_jac=None,
        *,
        type: Literal['linear'] | None = None,
        parameter_estimation: Literal['joint'] | None = None,
        inflation: float | None = None,
        **kwargs,
    ):
        type = canonical_name(self.type if type is None else type, _TYPES)
        if type != 'linear':
            raise NotImplementedError(f"type={type!r} is a type of biasda.KalmanFilter; romda has 'linear'.")
        kwargs.update(given(type=type, parameter_estimation=parameter_estimation, inflation=inflation))
        super().__init__(**kwargs)
        if set(kwargs) - self._consumed_kwargs:
            raise TypeError(f"unexpected keyword arguments: {sorted(set(kwargs) - self._consumed_kwargs)}")
        if self._psi is None:
            raise TypeError(f"{self.name} needs the initial state, psi0=...")
        model, N = self._model, self.N
        if model is None and self._F is None:
            raise ValueError(f"{self.name}: provide a model instance (model=...) or a linear transition matrix (F=...).")
        self.Q = np.atleast_2d(Q) if Q is not None else np.zeros((N, N))
        self._Cpp = np.array(Cpp0, dtype=float)          # posterior covariance P_{k|k}
        self._check_parameter_estimation()
        if F_jac is None:
            F_jac = (self._F if self._F is not None
                     else model.F if model is not None and hasattr(model, 'F') else np.eye(N))
        self._F_jac = self._hold_parameter_rows(np.atleast_2d(np.asarray(F_jac, dtype=float)))

    @property
    def Cpp(self) -> np.ndarray:
        """Current state covariance (forecast after `forecast_step`, posterior after `analysis_step`)."""
        return self._Cpp

    @cached_property
    def M_mat(self) -> np.ndarray:
        """Measurement matrix over the N-dimensional state (the columns of M applied to the identity)."""
        return self._MA(np.eye(self.N))

    def _hold_parameter_rows(self, J: np.ndarray) -> np.ndarray:
        """A one-step Jacobian with the identity rows of the persistent parameters."""
        J = np.array(J, dtype=float)
        J[self.Nphi:] = np.eye(self.N)[self.Nphi:]
        return J

    # ── Forecast ──────────────────────────────────────────────────────────────

    def forecast_step(self, t_end=None, **kwargs) -> None:
        r"""Advance the mean and the covariance.

        With a model, the model advances the mean (`SequentialEstimator.forecast_step`) and the
        covariance follows the Jacobian once per step spanned; without one, ``F`` is applied ``Nt``
        times (default 1), `current_time` counts steps, and `t_end` is not supported.
        """
        if self._model is not None:
            t_prev = self._t
            super().forecast_step(t_end=t_end, **kwargs)
            psi = np.asarray(self.model.current_state, dtype=float)
            self._psi = psi.mean(axis=-1) if psi.ndim > 1 else psi
            self._t = float(self.model.current_time)
            for _ in range(max(1, int(round((self._t - t_prev) / self._model.dt)))):
                self._Cpp = self._F_jac @ self._Cpp @ self._F_jac.T + self.Q
            return

        if t_end is not None:
            raise ValueError("t_end requires a model; F-only mode advances Nt steps (default 1).")
        Nt = kwargs.get("Nt", 1)
        for _ in range(Nt):
            self._psi = self._advance(self._psi)
            self._Cpp = self._F_jac @ self._Cpp @ self._F_jac.T + self.Q
        self._t += Nt
        if self.bias is not None:
            self.bias.advance(Nt=Nt)

    # ── Analysis ──────────────────────────────────────────────────────────────

    def _innovation_stats(self, d, Cdd):
        """``(P, Cxy, Cyy, z)`` of the update at the current forecast: the state covariance that the
        update corrects, the state-observable and observable covariances, and the innovation."""
        # Bias-aware, the linearized bias model maps H = (I + J) M, with the gamma rows stacked
        # under it when the estimator regularizes (exact covariances).
        H, z, Cdd_eff = self._stacked_innovation(self.M_mat, (self.M_mat @ self._psi)[:, None], d, Cdd)
        z = z.ravel()
        Cpp_f = self._Cpp                                        # forecast covariance P_{k|k-1}
        Cxy = Cpp_f @ H.T
        return Cpp_f, Cxy, H @ Cxy + Cdd_eff, z

    def _update_rows(self, rows, P, Cxy, Cyy, z) -> None:
        """Kalman update of the given rows; the other rows keep their forecast.

        With the gain ``K`` of the rows and zero elsewhere, the posterior covariance is the Joseph
        form, ``P - K Cxy^T - Cxy K^T + K Cyy K^T``, which is ``P - K Cyy K^T`` when every row is updated.
        In exact arithmetic, this equals ``(I - K H) P``, the inverse Hessian of the (stacked) cost.
        In floating point, the two forms treat the antisymmetric round-off ``A`` of ``P`` differently:
        ``(I - K H) P`` contracts it, whereas this form returns ``A + K H A H^T K^T``, which grows at every
        analysis. Symmetrizing the posterior removes ``A`` at every analysis.
        """
        K = np.zeros_like(Cxy)
        K[rows] = Cxy[rows] @ inv(Cyy)
        KC = K @ Cxy.T
        self._psi = self._psi + K @ z
        C = P - KC - KC.T + K @ Cyy @ K.T
        self._Cpp = 0.5 * (C + C.T)

    def analysis_step(self, d: np.ndarray, Cdd: np.ndarray, return_analysis: bool = False) -> np.ndarray | None:
        """Kalman analysis of the joint state (the parameters stay at their forecast during the
        ``start_param`` warm-up). Returns the analysed mean when ``return_analysis`` is True."""
        d, Cdd = np.atleast_1d(d), np.atleast_2d(Cdd)
        s, a = slice(0, self.Nphi), slice(self.Nphi, self.N)
        if self.Na and self.start_param > 0:
            self.activate_parameter_estimation = len(self.assimilated_data.times) >= self.start_param
        rows = s if self.Na and not self.activate_parameter_estimation else slice(None)
        self._update_rows(rows, *self._innovation_stats(d, Cdd))

        if self.inflation_factor != 1.0:
            D = np.full(self.N, self.inflation_factor)
            if not self.inflate_parameters:
                D[a] = 1.0
            self._Cpp = D[:, None] * self._Cpp * D

        if self._model is not None:
            self.update_history(self._psi[:, np.newaxis], self._t, modify_saved_states=True)
        self.assimilated_data = (d, self._t)
        if self.bias is not None:
            self._update_bias_from_analysis(d, self._MA(self._psi[:, np.newaxis]))

        if return_analysis:
            return self._psi.copy()
