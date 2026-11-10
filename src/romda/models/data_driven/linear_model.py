# %% =================================== LINEAR MODEL ============================================== %% #
import numpy as np
from dynamodels import DiscreteIntegrator, Model


class LinearModel(Model):
    r"""Simple linear state-space forecast model.

    $$
    \vect{\psi}_{t+1} = \matr{F}\vect{\psi}_t + \vect{\eta}_t,
    $$

    where $\matr{F}$ is the $(N_\phi \times N_\phi)$ state transition matrix and
    $\vect{\eta}_t \sim \mathcal{N}(\vect{0}, \matr{Q})$ is optional
    zero-mean Gaussian process noise with covariance $\matr{Q}$ (`Q_noise`).

    Uses `DiscreteIntegrator` (a fixed-step map), so it only requires a `time_step`
    method instead of a continuous `time_derivative`.
    """

    t_CR = 0.0
    # Q_noise is a covariance matrix, not a scalar estimable parameter, so it does
    # not belong in `params` (which feeds alpha0/filename/DA parameter estimation).
    fixed_params = ['F']

    def __init__(self, F, M_obs=None, psi0=None, dt=1.0, Q=None, **model_dict):
        """Build a `LinearModel` from a fixed state transition matrix `F`.

        Parameters
        ----------
        F : np.ndarray
            State transition matrix, shape ``(Nphi, Nphi)`` (or scalar/1D,
            promoted to a square 2D array via `numpy.atleast_2d`).
        M_obs : np.ndarray, optional
            Observation (measurement) operator, shape ``(Nq, Nphi)``. Defaults to
            the identity (full-state observation).
        psi0 : np.ndarray, optional
            Initial state, shape ``(Nphi,)`` or ``(Nphi, m)`` for an ensemble.
            Defaults to zeros.
        dt : float
            Time step. Default 1.0.
        Q : np.ndarray, optional
            Process noise covariance `Q_noise`, shape ``(Nphi, Nphi)``. Defaults to
            the zero matrix (no noise).
        **model_dict
            Additional `Model` options, forwarded to `Model.__init__`.

        Raises
        ------
        AssertionError
            If `F` is not square.
        """
        F = np.atleast_2d(np.array(F, dtype=float))
        Nphi = F.shape[0]
        assert F.shape == (Nphi, Nphi), f"F must be square, got {F.shape}"

        self.F = F
        self.Q_noise = np.zeros((Nphi, Nphi)) if Q is None else np.array(Q, dtype=float)
        self._has_noise = not np.allclose(self.Q_noise, 0)

        if psi0 is None:
            psi0 = np.zeros(Nphi)
        psi0 = np.atleast_1d(np.array(psi0, dtype=float))

        # Measurement operator M  (Nq x Nphi)
        if M_obs is None:
            M_obs = np.eye(Nphi)
        self._M_obs = np.atleast_2d(np.array(M_obs, dtype=float))
        self.Nq = self._M_obs.shape[0]

        super().__init__(psi0=psi0, dt=dt,
                         integrator_class=DiscreteIntegrator,
                         **model_dict)

    # ---- labels ----
    @property
    def state_labels(self):
        r"""list of str: LaTeX labels for the state vector, $x_0, \dots, x_{N_\phi-1}$."""
        return [f'$x_{{{i}}}$' for i in range(self.Nphi)]

    @property
    def obs_labels(self):
        r"""list of str: LaTeX labels for the observed outputs, $y_0, \dots, y_{N_q-1}$."""
        return [f'$y_{{{i}}}$' for i in range(self.Nq)]

    # ---- discrete map ----
    def time_step(self, Nt):
        r"""Propagate the state forward `Nt` steps,
        $\vect{\psi}_{k+1} = \matr{F}\vect{\psi}_k + \vect{\eta}_k$
        with $\vect{\eta}_k \sim \mathcal{N}(\vect{0}, \matr{Q})$ added if
        `Q_noise` is non-zero.

        Parameters
        ----------
        Nt : int
            Number of steps to propagate.

        Returns
        -------
        psi_out : np.ndarray
            State trajectory, shape ``(Nt + 1, Nphi, m)`` (``psi_out[0]`` is the
            current state).
        t_out : np.ndarray
            Corresponding time points, shape ``(Nt + 1,)``.
        """
        psi0 = self.current_state
        t0   = self.current_time
        dt   = self.dt

        t_out = np.round(t0 + np.arange(Nt + 1) * dt, self.precision_t)

        m = psi0.shape[1] if psi0.ndim == 2 else 1
        psi_out = np.empty((Nt + 1, self.Nphi, m))
        psi_out[0] = psi0

        for k in range(1, Nt + 1):
            psi_out[k] = self.F @ psi_out[k - 1]
            if self._has_noise:
                psi_out[k] += self.rng.multivariate_normal(
                    np.zeros(self.Nphi), self.Q_noise, size=m
                ).T

        # shape expected by DiscreteIntegrator: (Nt+1, Nphi, m=1)
        return psi_out, t_out

    # ---- observables ----
    def get_observables(self, Nt=1, **kwargs):
        r"""Map the trailing states to observables, $\vect{y} = \matr{M}_\mathrm{obs}\vect{\psi}$.

        Parameters
        ----------
        Nt : int
            Number of trailing time steps to return (same convention as
            `Model.get_observables`). If 1 (default), the leading ``Nt`` axis
            is dropped; 0 returns the full history.
        **kwargs
            Unused; accepted for interface compatibility.

        Returns
        -------
        np.ndarray
            Observed outputs, shape ``(Nq, m)`` if ``Nt == 1``, else
            ``(Nt, Nq, m)``.
        """
        if Nt == 1:
            return self._M_obs @ self.hist[-1, :self.Nphi, :]
        psi_hist = self.hist[-Nt:, :self.Nphi, :]              # (Nt, Nphi, m)
        return np.einsum('qp,tpm->tqm', self._M_obs, psi_hist)
