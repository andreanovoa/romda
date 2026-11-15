"""`romda.models`: the fixed-step integrator of the physical models (`RK4Integrator`) and the shim that re-exports
dynamodels, with the pickle paths of the files saved before the split."""

import io
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from romda.models import RK4Integrator
from romda.models.physical import Annular, Lorenz63
from scipy.integrate import solve_ivp

DT = 1. / 51200


PARAMETERS = dict(nu=10., c2beta=30., kappa=1.5e-4, epsilon=6e-3, omega=2 * np.pi * 1092, theta_b=0.6, theta_e=0.5)


def test_rk4_keeps_the_phase_that_the_default_integrator_loses():
    """Over 0.1 s of the annular model, RK4 at the sampling interval stays within 1 Pa of a tight-tolerance
    solution; scipy's RK45 at its default tolerances, the integrator of dynamodels, drifts by tens of pascals."""
    Nt = int(0.1 / DT)
    t = np.arange(Nt + 1) * DT
    reference = solve_ivp(lambda tt, y: Annular.time_derivative(tt, y, **PARAMETERS), (0, t[-1]),
                          Annular(dt=DT, **PARAMETERS).psi0.ravel(), t_eval=t, rtol=1e-11, atol=1e-9, method='DOP853').y.T

    model = Annular(dt=DT, **PARAMETERS)
    model.integrator = RK4Integrator(model)
    psi, times = model.time_integrate(Nt)
    np.testing.assert_allclose(times, t[1:], atol=1e-12)
    assert np.abs(psi[:, 0, 0] - reference[1:, 0]).max() < 1.

    default, _ = Annular(dt=DT, **PARAMETERS).time_integrate(Nt)
    assert np.abs(default[:, 0, 0] - reference[1:, 0]).max() > 10.


def test_rk4_forecasts_each_member_with_its_own_parameters():
    """The ensemble as one array gives the forecast of each member alone, and holds the parameter rows."""
    model = Lorenz63()
    model.init_ensemble(m=4, std_phi=0.1, std_alpha={'rho': [26., 30.], 'beta': [2.5, 2.8]})
    state = np.array(model.current_state)
    model.integrator = RK4Integrator(model)
    psi, _ = model.time_integrate(50)
    np.testing.assert_array_equal(psi[:, model.Nphi:], np.broadcast_to(state[model.Nphi:], psi[:, model.Nphi:].shape))

    by_member = Lorenz63()
    by_member.init_ensemble(m=4, std_phi=0.1, std_alpha={'rho': [26., 30.], 'beta': [2.5, 2.8]})
    by_member.update_history(psi=state[np.newaxis], t=model.hist_t[[0]], reset=True)
    by_member.integrator = RK4Integrator(by_member, vectorized=False)
    np.testing.assert_allclose(by_member.time_integrate(50)[0], psi, rtol=1e-12)
    model.close(), by_member.close()

    # two half steps are more accurate than one step: the two differ, by the error of the scheme
    fine = Lorenz63()
    fine.init_ensemble(m=4, std_phi=0.1, std_alpha={'rho': [26., 30.], 'beta': [2.5, 2.8]})
    fine.update_history(psi=state[np.newaxis], t=model.hist_t[[0]], reset=True)
    fine.integrator = RK4Integrator(fine, substeps=2)
    difference = np.abs(fine.time_integrate(50)[0] - psi).max()
    assert 0. < difference < 1e-2


def test_rk4_averaged_forecast_keeps_the_deviations():
    model = Lorenz63()
    model.init_ensemble(m=3, std_phi=0.1)
    state = np.array(model.current_state)
    model.integrator = RK4Integrator(model)
    psi, _ = model.time_integrate(5, averaged=True)
    np.testing.assert_allclose(psi - psi.mean(axis=-1, keepdims=True),
                               np.broadcast_to(state - state.mean(axis=-1, keepdims=True), psi.shape), atol=1e-10)


def test_rk4_refuses_a_right_hand_side_that_is_not_written_for_arrays():
    def scalar_only(t, psi, a):
        return [-a * float(psi[0])]                      # float() fails on the array of the members

    model = SimpleNamespace(time_derivative=scalar_only, current_state=np.ones((1, 3)), current_time=0., dt=0.1,
                            precision_t=3, Nphi=1, alpha0=dict(a=1.), governing_eqns_params={})
    with pytest.raises(TypeError, match='vectorized=False'):
        RK4Integrator(model).advance_ensemble(Nt=2)
    psi, t = RK4Integrator(model, vectorized=False).advance_ensemble(Nt=2)
    np.testing.assert_allclose(psi[-1, 0], np.exp(-0.2), rtol=1e-5)
    np.testing.assert_allclose(t, [0.1, 0.2])
    with pytest.raises(ValueError, match='substeps'):
        RK4Integrator(model, substeps=0)


def test_the_shim_re_exports_dynamodels_and_resolves_old_pickle_paths():
    # stock pickle stores 'romda.models.physical.lorenz63' + 'Lorenz63' and resolves
    # them with find_class (import module, getattr) — exactly what the sys.modules
    # aliases in romda.models must serve for pre-split .pkl files
    import dynamodels
    from romda.models import HistoryTracker, Model

    assert Model is dynamodels.Model
    assert HistoryTracker is dynamodels.HistoryTracker
    assert Lorenz63 is dynamodels.physical.Lorenz63
    for mod in ('romda.models.model', 'romda.models.history', 'romda.models.integrator',
                'romda.models.physical.lorenz63', 'romda.models.physical.rijke'):
        assert mod in sys.modules, f'{mod} alias missing'
    cls = pickle.Unpickler(io.BytesIO()).find_class('romda.models.physical.lorenz63', 'Lorenz63')
    assert cls is dynamodels.physical.Lorenz63


def test_modify_saved_states_overwrites_the_last_steps_alone():
    """analysis_step passes one state (N, m): it must overwrite the last step, not N steps; a window of states
    overwrites that window. (A check of dynamodels' HistoryTracker that its tests do not hold yet.)"""
    from romda.models.history import HistoryTracker

    for new in (np.full((6, 5), -1.0), np.full((3, 6, 5), -1.0)):
        h = HistoryTracker()
        hist = np.arange(10 * 6 * 5, dtype=float).reshape(10, 6, 5)
        h.update_history(hist, np.arange(10) * 0.1, reset=True)
        n = 1 if new.ndim == 2 else new.shape[0]
        h.update_history(new, t=0.9 if new.ndim == 2 else None, modify_saved_states=True)
        assert np.allclose(h.hist[-n:], new)
        assert np.allclose(h.hist[:-n], hist[:-n])
        assert np.allclose(h.hist_t, np.arange(10) * 0.1)
