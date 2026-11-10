"""Checks for `romda.estimators.inflation` (Evensen 2009, Chap. 15).

The load-bearing invariant is that each filter kernel's stashed member-space
transform reproduces its own analysis exactly (Aa = Af X̂).
"""

import numpy as np
import pytest
from romda.estimators import EnsembleFilter, EnSRKF
from romda.estimators.inflation import multiplicative_inflation
from romda.models.physical import Lorenz63, VdP


def test_multiplicative_inflation_scales_deviations():
    A = np.random.default_rng(0).standard_normal((5, 20))
    Aa = multiplicative_inflation(A, 1.5)
    np.testing.assert_allclose(np.mean(Aa, axis=1), np.mean(A, axis=1))
    np.testing.assert_allclose(np.std(Aa, axis=1), 1.5 * np.std(A, axis=1))


def _filter(type, m=12, seed=0, **kw):
    """A Lorenz-63 ensemble filter of `type` with its own random stream."""
    est = EnsembleFilter(parent_model=Lorenz63(dt=0.01), m=m, std_phi=0.2, type=type, **kw)
    est.rng = np.random.default_rng(seed)
    return est


@pytest.mark.parametrize('type', ['stochastic', 'sqrt'])
def test_analysis_transform_reproduces_kernel(type):
    est = _filter(type)
    Af = np.vstack((est.current_state, est.model.get_observables()))
    Aa = est._analysis_kernel(Af.copy(), np.array([1.0, 2.0, 3.0]), 0.1 * np.eye(3))
    np.testing.assert_allclose(Aa, Af @ est.member_space_transform(), atol=1e-10)
    est.close()


def _vdp_analysis(zeta_lims=None, **kwargs):
    """One forecast and one analysis of a VdP ensemble with one estimated parameter, zeta."""
    ens = EnSRKF(parent_model=VdP(dt=1e-3), m=8, std_phi=0.1, std_alpha=dict(zeta=(40., 60.)),
                 inflation=1.5, **kwargs)
    if zeta_lims:
        ens.model.alpha_lims = dict(zeta=zeta_lims)
    alpha0 = ens.current_state[ens.Nphi:].copy()  # constant over the forecast
    ens.forecast_step(t_end=0.05)
    d = ens.model.get_observables().mean(-1) * 1.1
    Aa = ens.analysis_step(d=d, Cdd=0.05 * np.diag(d ** 2), return_analysis=True)
    ens.model.close()
    return ens, Aa, alpha0


def _deviations(A):
    return A - A.mean(axis=-1, keepdims=True)


def test_inflate_parameters_false_inflates_the_state_only():
    ens, Aa_all, _ = _vdp_analysis()
    _, Aa_state, _ = _vdp_analysis(inflate_parameters=False)
    Nphi = ens.Nphi
    assert ens.Na == 1 and ens.inflate_parameters
    np.testing.assert_array_equal(Aa_state[:Nphi], Aa_all[:Nphi])
    np.testing.assert_allclose(Aa_all[Nphi:].mean(-1), Aa_state[Nphi:].mean(-1))
    np.testing.assert_allclose(_deviations(Aa_all[Nphi:]), 1.5 * _deviations(Aa_state[Nphi:]))


def test_inflate_parameters_false_in_state_only_warm_up():
    """With start_param > 0 the rows after Nphi are the observables, not the parameters."""
    ens, Aa, alpha0 = _vdp_analysis(inflate_parameters=False, start_param=2)
    assert ens.inflation_history.factors == [1.5]
    np.testing.assert_array_equal(Aa[ens.Nphi:], alpha0)
    # the observable rows must not be validated against the parameter limits
    assert not ens.rejected_analysis.times


def test_inflate_parameters_false_in_rejected_analysis():
    """A rejected analysis returns the parameters to their forecast, which is not inflated."""
    ens, Aa, alpha0 = _vdp_analysis(inflate_parameters=False, zeta_lims=(49., 51.))
    assert len(ens.rejected_analysis.times) == 1
    np.testing.assert_array_equal(Aa[ens.Nphi:], alpha0)


def test_rejected_analysis_keeps_the_forecast_when_the_inflation_leaves_the_limits():
    """The forecast parameters are already outside the limits, so the inflated forecast is too:
    the rejected analysis returns the forecast unchanged (as DA.analysisStep of the original
    code), not an inflated one."""
    ens = EnSRKF(parent_model=VdP(dt=1e-3), m=8, std_phi=0.1, std_alpha=dict(zeta=(40., 60.)),
                 inflation=1.5, inflation_factor_rejection=1.05)
    ens.model.alpha_lims = dict(zeta=(49., 51.))
    ens.forecast_step(t_end=0.05)
    Af = ens.current_state.copy()
    d = ens.model.get_observables().mean(-1) * 1.1
    Aa = ens.analysis_step(d=d, Cdd=0.05 * np.diag(d ** 2), return_analysis=True)
    ens.model.close()
    assert len(ens.rejected_analysis.times) == 1
    np.testing.assert_array_equal(Aa, Af)


@pytest.mark.parametrize('type', ['stochastic', 'sqrt'])
def test_low_rank_kernels_equal_the_m_by_m_transform_forms(type):
    # The EnKF gain form Af + (Af S^T) G and the EnSRKF thin-SVD square root must equal
    # the m x m transform forms they replace, for an ensemble larger than the
    # observation dimension and a correlated Cdd.
    rng = np.random.default_rng(1)
    est = _filter(type, m=30, seed=5)
    Af = np.vstack((est.current_state, est.model.get_observables()))
    m, Nq, M = Af.shape[1], est.Nq, est.M
    B = rng.standard_normal((Nq, Nq))
    Cdd = B @ B.T / Nq + 0.1 * np.eye(Nq)
    d = rng.standard_normal(Nq)
    Aa = est._analysis_kernel(Af.copy(), d, Cdd)
    mean = Af.mean(1, keepdims=True)
    Psi, S = Af - mean, M @ (Af - mean)
    Cinv = np.linalg.inv((m - 1) * Cdd + S @ S.T)
    if type == 'stochastic':
        D = d[:, None] + np.linalg.cholesky(Cdd) @ np.random.default_rng(5).standard_normal((Nq, m))
        reference = Af @ (np.eye(m) + S.T @ Cinv @ (D - M @ Af))
    else:
        ev, evec = np.linalg.eigh(np.eye(m) - S.T @ Cinv @ S)
        reference = mean + Psi @ S.T @ Cinv @ (d[:, None] - M @ mean) \
            + Psi @ (evec * np.sqrt(np.maximum(ev, 0.0))) @ evec.T
    np.testing.assert_allclose(Aa, reference, atol=1e-10)
    est.close()
