"""Inflation of the ensemble filters (`romda.estimators.inflation`): multiplicative (Evensen 2009, Chap. 15),
adaptive (Evensen 2009, Sect. 15.3) and innovation inflation, ``inflation='innovation'`` (Li, Kalnay & Miyoshi 2009).

The adaptive method applies the stashed member-space transform of the analysis (checked in test_ensemble.py) to a
random ensemble. The innovation tests mirror biasda's tests/test_innovation_inflation.py for romda's updates. Their
truth is a 10-site Lorenz-96 ring, observed at every site every two steps with noise of standard deviation 0.5; the
filters have 20 members. The random model error is an additive noise in the dynamics of the truth, and the systematic
error is a sensor offset of 3 at every site. The factor is the median of the logged deviation factors over the second
half of 300 analyses.
"""

import numpy as np
import pytest
from conftest import obs_covariance, vdp_ensemble, vdp_twin
from romda.bias_estimators import ConstantBias
from romda.estimators import EnsembleFilter, EnSRKF
from romda.estimators.inflation import (
    adaptive_inflation_factor,
    innovation_inflation_factor,
    multiplicative_inflation,
)
from romda.models import Integrator
from romda.models.physical import Lorenz63, Lorenz96, VdP

NX, DT, EVERY, SIGMA, M = 10, 0.05, 2, 0.5, 20


def test_multiplicative_inflation_scales_deviations():
    A = np.random.default_rng(0).standard_normal((5, 20))
    Aa = multiplicative_inflation(A, 1.5)
    np.testing.assert_allclose(np.mean(Aa, axis=1), np.mean(A, axis=1))
    np.testing.assert_allclose(np.std(Aa, axis=1), 1.5 * np.std(A, axis=1))


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
    # with start_param > 0 the rows after Nphi are the observables, not the parameters
    ens, Aa, alpha0 = _vdp_analysis(inflate_parameters=False, start_param=2)
    assert ens.inflation_history.factors == [1.5]
    np.testing.assert_array_equal(Aa[ens.Nphi:], alpha0)
    # the observable rows must not be validated against the parameter limits
    assert not ens.rejected_analysis.times


def test_rejected_analysis_keeps_the_forecast():
    """A rejected analysis returns the parameters to their forecast, which is not inflated. When the forecast
    parameters are already outside the limits, the inflated forecast is too: the rejected analysis returns the
    forecast unchanged (as DA.analysisStep of the original code), not an inflated one."""
    ens, Aa, alpha0 = _vdp_analysis(inflate_parameters=False, zeta_lims=(49., 51.))
    assert len(ens.rejected_analysis.times) == 1
    np.testing.assert_array_equal(Aa[ens.Nphi:], alpha0)

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


# ── adaptive inflation

def test_adaptive_factor_from_the_transform_of_the_analysis():
    rho = adaptive_inflation_factor(np.eye(10), rng=np.random.default_rng(1))
    assert rho == pytest.approx(1.0, abs=1e-12)                 # the identity update
    for type in ('stochastic', 'sqrt'):
        est = EnsembleFilter(parent_model=Lorenz63(dt=0.01), m=12, std_phi=0.2, type=type)
        est.rng = np.random.default_rng(0)
        Af = np.vstack((est.current_state, est.model.get_observables()))
        est._analysis_kernel(Af, np.array([1.0, 2.0, 3.0]), 0.1 * np.eye(3))
        rho = adaptive_inflation_factor(est.member_space_transform(), rng=np.random.default_rng(2))
        assert np.isfinite(rho) and rho > 0
        if type == 'sqrt':
            # the square-root deviation transform is a contraction, so rho >= 1
            assert rho >= 1.0
        est.close()


def test_adaptive_inflation_da_loop():
    truth = vdp_twin()
    ens = vdp_ensemble(truth, EnSRKF, inflation='adaptive')
    assert ens.adaptive_inflation and ens.inflation_factor == 1.0      # inflation='adaptive' switches it on
    Cdd = obs_covariance(truth)
    for d, t_d in list(zip(truth.y_obs, truth.t_obs))[:4]:
        ens.forecast_step(t_end=t_d)
        ens.analysis_step(d=d, Cdd=Cdd.copy())
    ens.model.close()
    assert np.isfinite(ens.model.hist).all()
    assert len(ens.inflation_history.factors) == 4
    assert all(r >= 1.0 for r in ens.inflation_history.factors)


# ── innovation inflation

def _rk4(x):
    def f(x):
        return (np.roll(x, -1, 0) - np.roll(x, 2, 0)) * np.roll(x, 1, 0) - x + 8.0
    k1 = f(x)
    k2 = f(x + 0.5 * DT * k1)
    k3 = f(x + 0.5 * DT * k2)
    k4 = f(x + DT * k3)
    return x + DT / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


class _RK4(Integrator):
    """RK4 on every member at once; the IVP integrator of Lorenz96 runs one process per member,
    which makes 300 analyses take a minute."""

    def advance_single(self, Nt=100, **kwargs):
        return self.advance_ensemble(Nt=Nt)

    def advance_ensemble(self, Nt=100, averaged=False, alpha=None):
        model, x = self.model, self.model.current_state.copy()
        out = np.empty((Nt,) + x.shape)
        for n in range(Nt):
            x = out[n] = _rk4(x)
        return out, np.round(model.current_time + np.arange(1, Nt + 1) * model.dt, model.precision_t)


def _l96():
    model = Lorenz96(Nx=NX, dt=DT, psi0=8.0 + np.sin(np.arange(NX)), observed_idx=list(range(NX)))
    model.integrator = _RK4(model)
    psi, t = model.time_integrate(Nt=200)
    model.update_history(psi[-1:], t=0.0, reset=True)
    return model


def _twin(model_error=0.0, offset=0.0, n_an=300, seed=0, **kw):
    """Mean analysis error and median deviation factor over the second half of the analyses."""
    rng = np.random.default_rng(seed)
    truth = _l96()
    x = truth.current_state[:, 0].copy()
    est = EnsembleFilter(parent_model=truth, m=M, std_phi=0.5, **kw)
    est.rng = np.random.default_rng(seed)
    err = []
    for _ in range(n_an):
        for _ in range(EVERY):
            x = _rk4(x) + model_error * np.sqrt(DT) * rng.normal(size=NX)
        d = x + offset + SIGMA * rng.normal(size=NX)
        est.forecast_step(Nt=EVERY)
        est.analysis_step(d=d, Cdd=SIGMA**2 * np.eye(NX))
        err.append(np.sqrt(np.mean((est.current_state.mean(-1) - x) ** 2)))
    factors = est.inflation_history.factors
    return np.mean(err[n_an // 2:]), np.median(factors[len(factors) // 2:]) if factors else 1.0


class _KnownOffset(ConstantBias):
    """The sensor offset, known: a constant bias that the analysis innovation does not reset."""

    def update_state_from_innovation(self, *args, **kwargs):
        return self.current_state


def _true_offset(c):
    return _KnownOffset(innovation=np.zeros(NX), t=0.0, dt=DT, k=c)


def test_the_options_of_the_inflation_are_checked():
    # The dual scheme and localization, which have no single transform for the adaptive factor, live in biasda.
    with pytest.raises(NotImplementedError, match='biasda'):
        EnsembleFilter(parent_model=VdP(dt=1e-3), m=8, std_phi=0.1, std_alpha=dict(zeta=(40., 60.)),
                       parameter_estimation='dual', inflation='adaptive')
    with pytest.raises(NotImplementedError, match='biasda'):
        EnSRKF(parent_model=Lorenz63(dt=0.01), m=8, std_phi=0.2, inflation='adaptive', localization=1.0)
    # One spelling: inflation='adaptive' or 'innovation'; the attributes are not keywords.
    with pytest.raises(TypeError):
        EnSRKF(adaptive_inflation=True, parent_model=Lorenz63, dt=0.015, m=5, std_phi=0.2)
    kw = dict(parent_model=_l96(), m=M, std_phi=0.5)
    with pytest.raises(TypeError):
        EnsembleFilter(**kw, innovation_inflation=True)
    with pytest.raises(ValueError):
        EnsembleFilter(**kw, inflation='innovations')
    with pytest.raises(ValueError, match='inflation_average'):
        _one_analysis(1.0, 0.1, inflation='innovation', inflation_average='median')


def test_one_sample_estimate_is_the_trace_identity():
    # nu nu^T has the expectation C_dd + lambda C_yy: the estimate recovers lambda from the traces.
    nu = np.array([2.0, 1.0])
    assert innovation_inflation_factor(nu, np.eye(2), np.diag([1.0, 0.5])) == pytest.approx((5.0 - 2.0) / 1.5)


def test_the_factor_stays_at_one_on_a_perfect_model_and_grows_with_random_model_error():
    # Measured over seeds 0-4, perfect model: median 1.000 for both types, mean errors 0.10-0.14. Random model error:
    # median factors 1.44-1.64 and mean errors 0.34-0.41, against 2.7-4.7 for the uninflated filter, which loses the
    # truth in every run.
    for type in ('stochastic', 'sqrt'):
        err, rho = _twin(inflation='innovation', type=type)
        assert rho < 1.1, type
        assert err < SIGMA, type
        err, rho = _twin(model_error=1.0, inflation='innovation', type=type)
        err_1, _ = _twin(model_error=1.0, type=type)
        assert rho > 1.2, type
        assert err < SIGMA < err_1, type


def test_the_bias_corrected_innovation_keeps_the_factor_of_the_random_error():
    # A sensor offset of 3 (six times the noise) on top of the random model error. The raw
    # innovation of the bias-blind filter carries the offset, and the factor grows to absorb it;
    # the innovation corrected by the true offset is that of the unbiased twin. Measured over seeds
    # 0-4 (stochastic; sqrt): unbiased and true offset 1.55-1.64; 1.44-1.49, bias-blind 3.9-4.3;
    # 3.7-4.0.
    for type in ('stochastic', 'sqrt'):
        kw = dict(model_error=1.0, inflation='innovation', type=type)
        _, rho_random = _twin(**kw)
        _, rho_raw = _twin(offset=3.0, **kw)
        _, rho_true = _twin(offset=3.0, parent_bias=_true_offset(3.0), **kw)
        assert rho_raw > rho_random + 2.0, type
        assert rho_true == pytest.approx(rho_random, abs=0.01), type


def test_update_reservoir_false_keeps_the_reservoir_at_its_forecast():
    # As in test_ensemble: N_dim and N_units mark the trailing rows as a 'reservoir', which the
    # analysis and the prior inflation leave at its forecast. With inflation_rate = 1 the factor is the
    # one-sample estimate of an innovation of two forecast standard deviations at every site.
    est = EnsembleFilter(parent_model=_l96(), m=M, std_phi=0.5, type='sqrt', inflation='innovation',
                         inflation_rate=1.0, update_reservoir=False)
    est.model.N_dim, est.model.N_units = 2, NX - 2
    est.forecast_step(Nt=EVERY)
    Af = est.current_state.copy()
    Cyy = np.cov(Af)
    est.analysis_step(d=Af.mean(-1) + 2.0 * np.sqrt(np.diag(Cyy)), Cdd=SIGMA**2 * np.eye(NX))
    assert est.inflation_history.factors[-1] == pytest.approx(np.sqrt(4.0 - NX * SIGMA**2 / np.trace(Cyy)))
    np.testing.assert_allclose(est.current_state[2:], Af[2:])
    assert not np.allclose(est.current_state[:2], Af[:2])


def test_identical_members_give_a_unit_factor_not_nan():
    # With no spread at the sensors, the one-sample estimate is 0/0: the factor stays one and the
    # running means stay finite, so that the factor responds once the ensemble spreads.
    est = EnsembleFilter(parent_model=_l96(), m=M, std_phi=0.5, type='sqrt', inflation='innovation')
    X = np.ones((NX, M))           # identical members, whose mean is exact (rounding leaves no spread)
    d, Cdd = np.full(NX, 2.0), SIGMA**2 * np.eye(NX)    # every site observed
    assert [est._innovation_inflation_factor(np.vstack((X, X)), d, Cdd) for _ in range(3)] == [1.0] * 3
    assert np.all(np.isfinite(est._inflation_moments))
    X = X + 0.1 * np.random.default_rng(0).standard_normal(X.shape)
    assert est._innovation_inflation_factor(np.vstack((X, X)), d, Cdd) > 1.0


# ── The guarded, plainly averaged variant of the washout-free ESN-DA paper (the da block of scripts/esnda/configs) ──

PAPER = dict(inflation='innovation', inflation_guard=True, inflation_average='plain', inflation_max=3.0)


def _one_analysis(offset, sigma, collapse=False, **kw):
    """The deviation factor of one analysis (inflation_rate = 1: the one-sample estimate) whose innovation is
    `offset` forecast standard deviations at every site, with C_dd = sigma^2 I."""
    est = EnsembleFilter(parent_model=_l96(), m=M, std_phi=0.5, type='sqrt', inflation_rate=1.0, **kw)
    est.forecast_step(Nt=EVERY)
    Af = est.current_state.copy()
    if collapse:                                        # every member at the mean: no spread at the sensors
        Af[:] = Af.mean(-1, keepdims=True)
        est.model.update_history(Af[np.newaxis], None, modify_saved_states=True)
    std = np.sqrt(np.diag(np.cov(Af))) if not collapse else np.ones(NX)
    est.analysis_step(d=Af.mean(-1) + offset * std, Cdd=sigma**2 * np.eye(NX))
    return est.inflation_history.factors[-1], Af


def test_the_guard_and_the_plain_average():
    # forecast variance at the sensors (std 0.5 per site) above tr C_dd (sigma 0.1): lambda~ = 1 with the guard
    plain = dict(inflation='innovation', inflation_average='plain', inflation_max=3.0)
    assert _one_analysis(3.0, 0.1, **PAPER)[0] == 1.0
    assert _one_analysis(3.0, 0.1, **plain)[0] > 1.0
    # an innovation of two forecast standard deviations, C_dd small against the spread once the guard is off:
    # lambda~ = 4 - tr C_dd / tr C_yy, clipped to inflation_max^2 = 9 and below by one
    rho, Af = _one_analysis(2.0, 0.1, **plain)
    assert rho == pytest.approx(np.sqrt(4.0 - NX * 0.1**2 / np.trace(np.cov(Af))))
    assert _one_analysis(10.0, 0.1, **plain)[0] == 3.0
    assert _one_analysis(0.0, 0.1, **plain)[0] == 1.0


def test_a_collapsed_ensemble_takes_the_limit_of_the_ratio():
    # no spread at the sensors: the bound when the innovation exceeds C_dd, one otherwise (no 0/0)
    assert _one_analysis(10.0, 0.1, collapse=True, **PAPER)[0] == 3.0
    assert _one_analysis(0.0, 0.1, collapse=True, **PAPER)[0] == 1.0
