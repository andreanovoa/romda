"""A fixed-step integrator for the physical models, on the `Integrator` protocol of dynamodels."""

import numpy as np
from dynamodels import Integrator


class RK4Integrator(Integrator):
    r"""Classical fourth-order Runge-Kutta scheme at the output step of the model, `model.dt`.

    The default integrator of the physical models, `IVPIntegrator`, calls ``scipy.integrate.solve_ivp`` with
    RK45 at its default tolerances (``rtol = 1e-3``), which is cheap but loses the phase of a weakly damped
    oscillator: on the annular model at 1/51200 s it drifts by 100 Pa in 0.2 s at an amplitude of 250 Pa,
    whereas this scheme drifts by 1 Pa. A parameter inferred with the default integrator absorbs that error
    (the angular frequency of the annular model, by about 0.05 %). Select this integrator after the model is
    built::

        model.integrator = RK4Integrator(model)

    The right-hand side, ``model.time_derivative(t, psi, **parameters)``, is evaluated on the whole ensemble
    at once, with `psi` of shape ``(N, m)`` and one array of ``m`` values per parameter, so it must be written
    with array operations, as those of `Annular`, `VdP` and `Lorenz63` are. The first forecast checks this
    against a member-by-member evaluation; ``vectorized=False`` evaluates member by member (slower). The
    parameter rows of an augmented state are held constant.

    Parameters
    ----------
    model_instance : Model
        A model that defines ``time_derivative``.
    substeps : int, default=1
        Runge-Kutta steps within one output step.
    vectorized : bool, default=True
        Whether `time_derivative` accepts the ensemble as one array.
    """

    def __init__(self, model_instance, substeps: int = 1, vectorized: bool = True):
        super().__init__(model_instance)
        if int(substeps) < 1:
            raise ValueError(f'substeps={substeps}: at least one step within an output step.')
        self.substeps = int(substeps)
        self.vectorized = bool(vectorized)
        self._checked = False

    # ── Right-hand side ───────────────────────────────────────────────────────

    def _parameters(self, psi):
        """Parameters of the members of `psi` (N, m): one list of per-member dicts, and the same as arrays."""
        pm = self.model
        get_alpha = getattr(pm, 'get_alpha', None)
        members = get_alpha(psi) if get_alpha is not None else [pm.alpha0] * psi.shape[-1]
        arrays = {key: np.array([member[key] for member in members], dtype=float) for key in members[0]}
        return members, arrays

    def _rhs(self, psi):
        """The function ``f(t, psi)`` of the ensemble `psi` (N, m), with the parameters of its members."""
        pm = self.model
        fixed = pm.governing_eqns_params
        members, arrays = self._parameters(psi)
        Nphi, m = pm.Nphi, psi.shape[-1]

        def by_member(t, y):
            out = np.zeros_like(y)
            for j in range(m):
                out[:, j] = np.asarray(pm.time_derivative(t, y[:, j], **fixed, **members[j]), dtype=float)
            out[Nphi:] = 0.
            return out

        def all_members(t, y):
            out = np.stack([np.broadcast_to(np.asarray(row, dtype=float), (m,))
                            for row in pm.time_derivative(t, y, **fixed, **arrays)])
            out[Nphi:] = 0.
            return out

        if not self.vectorized:
            return by_member
        if not self._checked:           # once: the array evaluation must be the member-by-member one
            t0 = float(pm.current_time)
            try:
                same = np.allclose(all_members(t0, psi), by_member(t0, psi), rtol=1e-10, atol=0., equal_nan=True)
            except Exception as error:      # noqa: BLE001  (any failure means: not written for arrays)
                raise TypeError(f'{type(pm).__name__}.time_derivative does not accept the ensemble as one array '
                                f'({error}); build the integrator with vectorized=False.') from error
            if not same:
                raise TypeError(f'{type(pm).__name__}.time_derivative gives another result for the ensemble as '
                                'one array than member by member; build the integrator with vectorized=False.')
            self._checked = True
        return all_members

    # ── Forecast ──────────────────────────────────────────────────────────────

    def _integrate(self, psi, Nt):
        """`Nt` output steps from `psi` (N, m); returns (Nt, N, m), without the initial state."""
        pm = self.model
        f = self._rhs(psi)
        h = pm.dt / self.substeps
        t = float(pm.current_time)
        out = np.empty((Nt,) + psi.shape)
        y = np.array(psi, dtype=float)
        for n in range(Nt):
            for _ in range(self.substeps):
                k1 = f(t, y)
                k2 = f(t + h / 2, y + h / 2 * k1)
                k3 = f(t + h / 2, y + h / 2 * k2)
                k4 = f(t + h, y + h * k3)
                y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                t += h
            out[n] = y
        if not np.isfinite(out[-1]).all():
            bad = np.flatnonzero(~np.isfinite(out[-1]).all(axis=0)).tolist()
            raise RuntimeError(f'{type(pm).__name__}: the forecast of member(s) {bad} is not finite (the state '
                               'diverged); check the parameters of these members.')
        return out

    def advance_single(self, Nt: int = 100, **kwargs):
        """Forecast the single member `Nt` output steps (see `advance_ensemble`)."""
        return self.advance_ensemble(Nt=Nt)

    def advance_ensemble(self, Nt: int = 100, averaged: bool = False, alpha=None):
        """Forecast every member `Nt` output steps.

        With ``averaged=True`` only the ensemble mean is integrated, with the parameters of the mean, and each
        member keeps its initial deviation from it, as in `IVPIntegrator`. `alpha` is accepted for the
        interface and not used: the parameters are those of the members.

        Returns
        -------
        psi : ndarray, shape (Nt, N, m)
            Forecast, without the initial state.
        t : ndarray, shape (Nt,)
            Time stamps, `model.dt` apart.
        """
        pm = self.model
        psi0 = np.array(pm.current_state, dtype=float)
        if not np.isfinite(psi0).all():
            bad = np.flatnonzero(~np.isfinite(psi0).all(axis=0)).tolist()
            raise ValueError(f'{type(pm).__name__}: the initial state of member(s) {bad} is not finite.')
        Nt = int(Nt)
        if averaged and psi0.shape[-1] > 1:
            mean = psi0.mean(axis=-1, keepdims=True)
            psi = self._integrate(mean, Nt) + (psi0 - mean)[np.newaxis]
        else:
            psi = self._integrate(psi0, Nt)
        t = np.round(pm.current_time + np.arange(1, Nt + 1) * pm.dt, pm.precision_t)
        return psi, t
