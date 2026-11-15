"""`romda.metrics`: the statistics of an attractor, Lyapunov exponents of a map, the valid time of a forecast, the
error of the full state (`state_metrics`, `state_error`) and of the observables of a filter (`ensemble_metrics`,
`relative_error`, `window_errors`)."""

import types

import numpy as np
import pytest
from conftest import obs_covariance, vdp_ensemble, vdp_twin
from romda.metrics import (
    attractor_kld,
    error_scale,
    first_below,
    forecast_scores,
    lyapunov_spectrum_map,
    state_error,
    state_metrics,
    valid_time,
)


def test_attractor_kld_is_small_for_the_same_statistics_and_bounded_off_the_attractor():
    rng = np.random.default_rng(0)
    y = rng.standard_normal((20000, 2))
    members = np.stack([rng.standard_normal((20000, 2)), 50 + rng.standard_normal((20000, 2)),
                        np.full((20000, 2), np.nan)], axis=-1)
    kld = attractor_kld(y, members)
    assert kld[0] < 1e-2                                    # the same distribution: sampling error only
    assert kld[1] == kld[2] and 10 < kld[1] < np.log(1 / 1e-6) + 1   # off the bins, or blown up: the bounded maximum


def test_lyapunov_spectrum_map_of_a_linear_map_gives_the_log_of_its_multipliers():
    # exponents log(1.5), log(0.8), log(0.2) per step, from the fixed point at the origin (a growing reference
    # would swamp the finite-difference perturbations)
    A = np.diag([1.5, 0.8, 0.2])
    lam = lyapunov_spectrum_map(lambda x: A @ x, np.zeros(3), dt=0.5, n=200, n_transient=10, n_exp=3)
    np.testing.assert_allclose(lam, np.log([1.5, 0.8, 0.2]) / 0.5, rtol=1e-6)
    assert np.isnan(lyapunov_spectrum_map(lambda x: A @ x, np.ones(3), 0.5, 200, 10, 2, bound=10.0)).all()   # leaves


def test_valid_time_forecast_scores_and_first_below():
    # valid_time: the time of the first error above the threshold, nan counted as above
    t = np.arange(6) / 3.0
    assert valid_time([0.1, 0.2, 0.6, 0.1, 0.9, 0.9], t) == t[2]
    assert valid_time([0.6, 0.1, 0.1, 0.1, 0.1, 0.1], t) == 0.0                 # fails at the first sample
    assert valid_time(np.full(6, 0.1), t) == t[-1]                              # never: the last time
    assert valid_time(np.full(6, 0.1), t, t_max=2.0) == 2.0                     # never: the horizon
    assert valid_time([0.1, np.nan, 0.1, 0.1, 0.1, 0.1], t) == t[1]             # a blow-up fails
    assert valid_time([0.1, 0.3, 0.1, 0.1, 0.1, 0.1], t, threshold=0.25) == t[1]
    # forecast_scores: the first error, the valid time and the rms over one Lyapunov time
    t = np.arange(1, 7) / 2.0                                                  # 0.5 ... 3.0, from the last datum
    err = np.array([0.1, 0.2, 0.3, 0.6, 0.1, 0.1])
    first, valid, window = forecast_scores(err, t, t_lyap=1.0)
    assert first == 0.1 and valid == 2.0                                        # the first error above 0.5
    assert np.isclose(window, np.sqrt((0.1 ** 2 + 0.2 ** 2) / 2))                # t <= 1
    assert forecast_scores(err, t, t_lyap=2.0)[1] == 1.0                        # in Lyapunov times
    assert forecast_scores(np.full(6, 0.1), t, 1.0)[1] == 3.0                   # never fails: the last time
    # first_below: the first time a series falls below a level, within a window, or nan
    t = np.arange(6) / 2.0                                                     # 0 ... 2.5
    q = np.array([0.1, 0.9, 0.2, 0.9, 0.1, 0.1])
    assert first_below(t, q, 0.5) == 0.0
    assert first_below(t, q, 0.5, t_start=0.5) == 1.0                           # from t_start
    assert np.isnan(first_below(t, q, 0.15, t_end=1.5, t_start=0.5))            # never within the window
    assert first_below(t, q, 0.15, t_start=0.5) == 2.0


def test_state_metrics_scores_the_full_state():
    t = np.linspace(0., 10., 101)
    y_true = np.stack([np.sin(t), np.cos(t), np.sin(2 * t)], axis=1)
    truth = types.SimpleNamespace(t_obs=np.array([0., 5.]), true_parameters=None)

    perfect = dict(t=t, y=y_true.copy(), ensemble=None)
    assert state_metrics(perfect, truth, t, y_true)['rmse_da'] < 1e-12

    # an error on a component that is never measured still counts
    off = y_true.copy()
    off[:, 1] += 1.0
    metrics = state_metrics(dict(t=t, y=off, ensemble=None), truth, t, y_true)
    scale = error_scale(y_true[t <= 5.]) / np.sqrt(3)     # deviation from the time average, per component
    assert np.isclose(metrics['rmse_da'], np.sqrt(1 / 3) / scale)

    # the two windows are split at the last observation
    assert metrics['rmse_forecast'] > 0
    t_late, err, spread = state_error(dict(t=t, y=off, ensemble=None), t, y_true)
    assert t_late is t and err.shape == t.shape
    assert np.isnan(spread).all()          # open loop: no ensemble, NaN spread

    # a model record longer than the truth is refused
    with pytest.raises(ValueError, match='truth record ends'):
        state_metrics(dict(t=t + 5., y=y_true, ensemble=None), truth, t, y_true)


def test_state_metrics_carries_the_ensemble_spread():
    t = np.linspace(0., 10., 101)
    y_true = np.stack([np.sin(t), np.cos(t), np.sin(2 * t)], axis=1)
    truth = types.SimpleNamespace(t_obs=np.array([0., 5.]),
                                  true_parameters=dict(rho=2.0))

    # members = mean ± 0.5 on every state component -> spread 0.5 everywhere
    hist = np.concatenate([y_true[:, :, None] - 0.5, y_true[:, :, None] + 0.5], axis=-1)
    alpha = np.concatenate([hist, np.full((len(t), 1, 2), [[1.5, 2.5]])], axis=1)
    ens = types.SimpleNamespace(
        Na=1, est_alpha=['rho'],
        model=types.SimpleNamespace(hist=alpha, N_dim=3, Nphi=3))

    m = state_metrics(dict(t=t, y=y_true, ensemble=ens), truth, t, y_true)
    scale = error_scale(y_true[t <= 5.]) / np.sqrt(3)     # deviation from the time average, per component
    # per-member: every member is off by exactly 0.5 -> mean error 0.5, zero std
    assert np.isclose(m['rmse_da'], 0.5 / scale) and np.isclose(m['rmse_forecast'], 0.5 / scale)
    assert np.isclose(m['std_da'], 0.0) and np.isclose(m['std_forecast'], 0.0)
    # mean-trajectory convention kept alongside: the ensemble mean IS the truth here
    assert np.isclose(m['rmse_mean_da'], 0.0, atol=1e-12)
    assert np.isclose(m['spread_da'], 0.5 / scale) and np.isclose(m['spread_forecast'], 0.5 / scale)
    assert np.isclose(m['alpha_rho'], 2.0) and np.isclose(m['alpha_std_rho'], 0.5)
    assert np.isclose(m['alpha_error_rho'], 0.0)

    scale_full = error_scale(y_true) / np.sqrt(3)
    _, err, spread = state_error(dict(t=t, y=y_true, ensemble=ens), t, y_true)
    assert np.allclose(err, 0.5 / scale_full) and np.allclose(spread, 0.0)
    _, err_m, spread_m = state_error(dict(t=t, y=y_true, ensemble=ens), t, y_true,
                                       per_member=False)
    assert np.allclose(err_m, 0.0, atol=1e-12) and np.allclose(spread_m, 0.5 / scale_full)


def test_state_metrics_scores_the_calibration_after_synchronization():
    rng = np.random.default_rng(0)
    t = np.linspace(0., 10., 2001)
    y_true = np.stack([np.sin(t), np.cos(t), np.sin(2 * t)], axis=1)
    truth = types.SimpleNamespace(t_obs=np.array([0., 10.]), true_parameters=dict(rho=2.0))
    sigma = 0.1
    calibration = ('rmse_mean_sync', 'spread_sync', 'spread_skill', 'coverage_2sigma', 'rank_outliers')

    def score(state_members, rho_members, **kwargs):
        state_members[t < 2.] += 50.           # members off the attractor before the lock
        rho = np.broadcast_to(rho_members, (len(t), 1, len(rho_members)))
        ens = types.SimpleNamespace(
            Na=1, est_alpha=['rho'],
            model=types.SimpleNamespace(hist=np.concatenate([state_members, rho], axis=1),
                                        N_dim=3, Nphi=3))
        return state_metrics(dict(t=t, y=state_members.mean(axis=-1), ensemble=ens),
                             truth, t, y_true, **kwargs)

    # coverage of a calibrated ensemble: P(|t_(m-1)| <= 2 sqrt((m-1)/(m+1)))
    for n_members, coverage in ((8, 0.879), (64, 0.947)):
        # calibrated: the truth and the members are draws of one distribution, N(centre, sigma)
        centre = y_true + sigma * rng.standard_normal(y_true.shape)
        noise = rng.standard_normal(y_true.shape + (n_members,))
        m = score(centre[:, :, None] + sigma * noise, 2.0 + sigma * noise[0, 0], t_lyap=1.)
        assert abs(m['spread_skill'] - 1.) < 0.05              # t_sync = 2 t_lyap = 2
        m1 = score(centre[:, :, None] + sigma * noise, 2.0 + sigma * noise[0, 0], t_lyap=1., ddof=1)
        assert np.isclose(m1['spread_sync'], m['spread_sync'] * np.sqrt(n_members / (n_members - 1)), rtol=1e-6)
        assert np.isclose(m1['spread_skill'], m['spread_skill'], rtol=1e-6)   # the sample std: the same ratio
        assert abs(m['coverage_2sigma'] - coverage) < 0.02
        assert abs(m['rank_outliers'] / (2 / (n_members + 1)) - 1.) < 0.3
        assert m['alpha_cover_rho'] == 1.0
        assert m['rmse_mean_da'] > 100 * m['rmse_mean_sync']   # the transient is left out

        # under-dispersed: every member shares a bias ten times the spread
        biased, rho_biased = y_true[:, :, None] + sigma + 0.1 * sigma * noise, 3.0 + 0.1 * sigma * noise[0, 0]
        m = score(biased.copy(), rho_biased, t_lyap=1.)
        assert m['spread_skill'] < 0.2 and m['coverage_2sigma'] < 0.5
        assert m['rank_outliers'] > 0.9 and m['alpha_cover_rho'] == 0.0

    # no t_lyap: the whole assimilation window; t_sync past the last observation: no window
    m = score(biased.copy(), rho_biased)
    assert m['rmse_mean_sync'] == m['rmse_mean_da'] and m['spread_sync'] == m['spread_da']
    assert m['rmse_mean_last'] == m['rmse_mean_da'] and m['spread_last'] == m['spread_da']
    m = score(biased.copy(), rho_biased, t_lyap=1., t_sync=11.)
    assert all(np.isnan(m[k]) for k in calibration) and np.isfinite(m['rmse_mean_da'])

    # single-network open loop: no ensemble, the same keys with NaN calibration
    m_open = state_metrics(dict(t=t, y=y_true, ensemble=None), truth, t, y_true, t_lyap=1.)
    assert all(np.isnan(m_open[k]) for k in calibration[1:])
    assert set(m) - set(m_open) == {'alpha_rho', 'alpha_std_rho', 'alpha_error_rho', 'alpha_cover_rho'}


def test_relative_error_is_eq_6_1_at_each_step_and_member():
    from romda.metrics import relative_error

    t = np.arange(0, 0.01, 1 / 51200)
    w = np.stack([np.cos(2 * np.pi * 1090. * t + k) for k in range(4)], axis=1) * 300.
    assert relative_error(w, w) == 0.
    assert relative_error(w, 0 * w) == pytest.approx(1.)  # a fixed point has error 1
    assert relative_error(w, 2 * w) == pytest.approx(1.)
    assert relative_error(w, w + 30.) == pytest.approx(np.sqrt(w.size * 900. / np.sum(w ** 2)))

    rng = np.random.default_rng(0)
    w, z = rng.normal(size=(50, 4)), rng.normal(size=(50, 4, 3))
    steps = relative_error(w, z, axis=1)
    assert steps.shape == (50, 3)
    for k in (0, 17, 49):
        for j in range(3):
            assert steps[k, j] == pytest.approx(relative_error(w[k], z[k, :, j]))
    assert relative_error(w, z[..., 0], axis=1).shape == (50,)
    assert np.allclose(relative_error(w, 0 * w, axis=1), 1.)  # a fixed point: 1 at every step


# ── the metrics of a filter on a Van der Pol twin

@pytest.fixture(scope='module')
def truth_biased():
    return vdp_twin('linear', noise_level=0.02)


def run_da(truth, bias=None, n_analysis=6):
    """Six analyses of an EnKF (rEnKF with a bias) of 8 members with zeta estimated."""
    from romda.estimators import rEnKF

    ens = vdp_ensemble(truth, rEnKF, bias, bias_regularization=1.) if bias else vdp_ensemble(truth)
    Cdd = obs_covariance(truth)
    for d, t_d in list(zip(truth.y_obs, truth.t_obs))[:n_analysis]:
        ens.forecast_step(t_end=t_d)
        ens.analysis_step(d=d, Cdd=Cdd.copy())
    ens.model.close()
    return ens


def test_ensemble_metrics_score_the_estimate_where_the_truth_exists(truth_biased):
    """A bias-aware filter estimates the truth with model plus bias: `ensemble_metrics` reports the error of
    that estimate (`rmse_corrected_*`) next to the error of the model observables (`rmse_*`). Past the end of the truth
    record the interpolation repeats its last value, so a default forecast that ran past the record was scored against
    a frozen truth."""
    from romda.bias_estimators import ConstantBias
    from romda.metrics import ensemble_metrics
    from romda.utils import interpolate

    ens = run_da(truth_biased, bias=ConstantBias)
    metrics = ensemble_metrics(ens, truth_biased)

    t = ens.model.hist_t
    y_ref = interpolate(truth_biased.t_true, truth_biased.y_true[..., 0], t)
    window = (t >= truth_biased.t_obs[0]) & (t <= truth_biased.t_obs[-1])
    scale = np.sqrt(np.mean(y_ref[window] ** 2))
    corrected, model = (np.mean(y, axis=-1) for y in ens.get_observable_hist())
    for key, estimate in (('rmse_corrected_da', corrected), ('rmse_da', model)):
        assert metrics[key] == pytest.approx(np.sqrt(np.mean((estimate[window] - y_ref[window]) ** 2)) / scale)
    assert metrics['rmse_corrected_da'] != metrics['rmse_da']

    truth_unbiased = vdp_twin()
    assert 'rmse_corrected_da' not in ensemble_metrics(run_da(truth_unbiased), truth_unbiased)

    t = np.linspace(0., 3., 31)                                  # the model runs to t = 3
    y = np.repeat(np.sin(t)[:, None, None], 2, axis=-1)          # two members on the truth ...
    y[t > 2.] += 100.                                            # ... that leave it after the record ends at t = 2
    ensemble = types.SimpleNamespace(model=types.SimpleNamespace(hist_t=t, get_observable_hist=lambda: y), Na=0)
    t_true = t[t <= 2.]
    truth = types.SimpleNamespace(t_true=t_true, y_true=np.sin(t_true)[:, None], t_obs=np.array([0.5, 1.]))
    assert ensemble_metrics(ensemble, truth)['rmse_forecast'] == pytest.approx(0., abs=1e-12)


def test_window_errors_score_one_window(truth_biased):
    """`window_errors` is Eq. (6.1) of Nóvoa et al. (2024) over a window: the model observables and the
    bias-corrected estimate against the truth, normalized by the truth."""
    from romda.bias_estimators import ConstantBias
    from romda.metrics import window_errors
    from romda.utils import interpolate

    ens = run_da(truth_biased, bias=ConstantBias)
    t = ens.model.hist_t
    t0, t1 = truth_biased.t_obs[1], truth_biased.t_obs[3]
    errors = window_errors(ens, truth_biased, t0, t1)

    window = (t >= t0 - 1e-12) & (t < t1 - 1e-12)
    y_ref = interpolate(truth_biased.t_true, truth_biased.y_true[..., 0], t[window])
    corrected, model = (np.mean(y[window], axis=-1) for y in ens.get_observable_hist())
    assert errors['model'] == pytest.approx(np.sqrt(np.sum((y_ref - model) ** 2) / np.sum(y_ref ** 2)))
    assert errors['corrected'] == pytest.approx(np.sqrt(np.sum((y_ref - corrected) ** 2) / np.sum(y_ref ** 2)))
    assert errors['bias'] == pytest.approx(np.sqrt(np.mean((corrected - model) ** 2)))
    with pytest.raises(ValueError, match='no step'):
        window_errors(ens, truth_biased, t[-1] + 1., t[-1] + 2.)
