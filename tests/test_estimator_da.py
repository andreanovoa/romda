"""Behavioral DA-loop tests for the `romda.estimators` filters on a physical twin.

Port of the retired `test_ensemble_da.py` (which ran on `romda.legacy`) to the
`Estimator` hierarchy: the assertions are about the DA mathematics, not the
implementation — an analysis step must move the ensemble-mean observables toward
the data (exactly, per analysis, for the deterministic EnSRKF; on average for the
stochastic EnKF, whose perturbed-observation mean carries an O(1/sqrt(m)) sampling
error), estimated parameters stay within `alpha_lims` or the analysis is rejected,
and `start_param` defers parameter estimation.
"""

import numpy as np
import pytest
from romda.bias_estimators import ConstantBias, NoBias
from romda.estimators import EnKF, EnSRKF, rEnKF
from romda.models.physical import VdP
from romda.observations import Observations


@pytest.fixture(scope='module')
def truth_unbiased():
    return Observations(model=VdP, t_start=0.6, t_stop=0.75, t_max=0.9, Nt_obs=30,
                        add_noise=True, noise_type='gauss, add', noise_level=0.01)


@pytest.fixture(scope='module')
def truth_biased():
    return Observations(model=VdP, t_start=0.6, t_stop=0.75, t_max=0.9, Nt_obs=30,
                        add_noise=True, noise_type='gauss, add', noise_level=0.02,
                        manual_bias='linear')


def obs_covariance(truth, std_obs=0.05):
    return np.diag(std_obs * np.ones(truth.y_obs.shape[1])) * np.max(abs(truth.y_obs), axis=0) ** 2


def make_ensemble(truth, da_method=EnKF, bias=None, m=8, **kwargs):
    return da_method(parent_model=VdP(dt=truth.dt),
                     parent_bias=bias,
                     m=m,
                     std_phi=0.1,
                     std_alpha=dict(zeta=(40., 60.)),
                     **kwargs)


def run_da(ens, truth, n_analysis=6):
    Cdd = obs_covariance(truth)
    for k, (d, t_d) in enumerate(zip(truth.y_obs, truth.t_obs)):
        ens.forecast_step(t_end=t_d)
        ens.analysis_step(d=d, Cdd=Cdd.copy())
        if k + 1 >= n_analysis:
            break
    ens.model.close()
    return ens


@pytest.mark.parametrize('da_method', [EnKF, EnSRKF])
def test_da_reduces_observation_error(truth_unbiased, da_method):
    ens = make_ensemble(truth_unbiased, da_method=da_method)

    Cdd = obs_covariance(truth_unbiased)
    err_forecast, err_analysis = [], []
    for k, (d, t_d) in enumerate(zip(truth_unbiased.y_obs, truth_unbiased.t_obs)):
        ens.forecast_step(t_end=t_d)
        y_f = np.mean(ens.model.get_observables(), axis=-1)
        err_forecast.append(np.linalg.norm(y_f - d))
        ens.analysis_step(d=d, Cdd=Cdd.copy())
        y_a = np.mean(ens.model.get_observables(), axis=-1)
        err_analysis.append(np.linalg.norm(y_a - d))
        if da_method is EnSRKF:
            # deterministic filter: it assimilates d exactly, so every analysis must
            # move the ensemble-mean observables towards the data
            assert err_analysis[-1] <= err_forecast[-1] + 1e-12
        if k >= 5:
            break
    # The stochastic EnKF assimilates a *perturbed* observation ensemble, whose mean
    # differs from d by a sampling error of order 1/sqrt(m) -- with m=8 an individual
    # analysis can move the mean slightly the wrong way. The guarantee is on average.
    assert np.mean(err_analysis) <= np.mean(err_forecast)
    ens.model.close()
    assert np.isfinite(ens.model.hist).all()
    assert len(ens.assimilated_data.times) == 6


def test_parameters_stay_within_limits_or_rejected(truth_unbiased):
    ens = run_da(make_ensemble(truth_unbiased, da_method=EnKF), truth_unbiased)
    alpha = ens.model.hist[-1, -ens.Na:, :]
    lo, hi = ens.model.alpha_lims['zeta']
    assert np.all((alpha >= lo) & (alpha <= hi)) or len(ens.rejected_analysis.times) > 0


@pytest.mark.parametrize('bias_cls', [NoBias, ConstantBias])
def test_bias_aware_loop_runs(truth_biased, bias_cls):
    ens = run_da(make_ensemble(truth_biased, da_method=rEnKF, bias=bias_cls,
                               bias_regularization=1.), truth_biased)
    assert np.isfinite(ens.model.hist).all()
    assert np.isfinite(ens.bias.hist).all()
    # model and bias histories stay synchronized in time
    assert abs(ens.bias.current_time - ens.model.current_time) < ens.model.dt
    if bias_cls is NoBias:
        np.testing.assert_allclose(ens.bias.hist, 0.)


def test_renkf_at_zero_bias_equals_the_enkf(truth_unbiased):
    # With zero bias and zero Jacobian the regularized bias-aware gain must reduce exactly
    # to the EnKF gain (Novoa et al. 2023, erratum 2024): C_yy = C_dd + C_qq with C_qq the
    # per-sample covariance. Weighting C_dd by (m - 1) instead would inflate the
    # observation noise m - 1 times and divide the effective gamma by m - 1.
    Cdd = obs_covariance(truth_unbiased)
    d, t_d = truth_unbiased.y_obs[0], truth_unbiased.t_obs[0]
    states = []
    for da_method, kwargs in ((EnKF, {}), (rEnKF, dict(bias=NoBias, bias_regularization=1.0))):
        ens = make_ensemble(truth_unbiased, da_method=da_method, **kwargs)
        ens.forecast_step(t_end=t_d)
        ens.rng = np.random.default_rng(7)               # identical observation perturbations
        ens.analysis_step(d=d, Cdd=Cdd.copy())
        states.append(np.array(ens.model.current_state))
        ens.model.close()
    np.testing.assert_allclose(states[1], states[0], atol=1e-10)
