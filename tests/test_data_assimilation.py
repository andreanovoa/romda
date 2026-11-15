"""`romda.data_assimilation`: the DA loop (its hook after each analysis and the washout of an ESN bias) and the
summary of the inflation history."""
import types

import numpy as np
import pytest

Nq = 2


def test_run_da_loop_washout_feeds_the_data_before_the_first_observation():
    """`run_da_loop(washout=True)` forecasts to the first observation and washes the network out on the raw data
    minus the ensemble-mean observables of the N_wash network steps before it."""
    from romda.data_assimilation import _washout_bias
    SimpleNamespace = types.SimpleNamespace

    n_wash, step, first, m = 3, 2, 40, 4
    rng = np.random.default_rng(1)
    q_hist = rng.normal(size=(first + 1, Nq, m))                    # model observables at steps 0, ..., first
    y_raw = rng.normal(size=(100, Nq, 1))
    seen = {}
    bias = SimpleNamespace(upsample=step, forecaster=SimpleNamespace(N_wash=n_wash),
                           washout=lambda innovations: seen.update(innovations=innovations))
    model = SimpleNamespace(get_observable_hist=lambda Nt: q_hist[-Nt:])
    filter_ens = SimpleNamespace(bias=bias, model=model, forecast_step=lambda t_end: seen.update(t_end=t_end))
    truth = SimpleNamespace(obs_idx=np.array([first, first + 10]), t_obs=np.array([0.4, 0.5]), y_raw=y_raw)

    _washout_bias(filter_ens, truth)
    steps = first - step * np.arange(n_wash, 0, -1)                  # 34, 36, 38: one network step apart
    assert seen['t_end'] == 0.4
    np.testing.assert_allclose(seen['innovations'], y_raw[steps, :, 0] - q_hist[steps].mean(axis=-1))

    with pytest.raises(ValueError, match='before the first observation'):
        _washout_bias(filter_ens, SimpleNamespace(obs_idx=np.array([4]), t_obs=np.array([0.04]), y_raw=y_raw))
    with pytest.raises(ValueError, match='ESN bias'):
        _washout_bias(SimpleNamespace(bias=None), truth)


def test_run_da_loop_calls_after_analysis_on_the_filtered_ensemble():
    # The hook runs after every analysis on the ensemble that run_da_loop filters (a copy of the input), and an edit
    # of its state carries into the next forecast, as in the hand-written loop.
    from romda.data_assimilation import run_da_loop
    from romda.estimators import EnSRKF
    from romda.models.physical import Lorenz63

    rng = np.random.default_rng(4)
    E0 = np.array([1.0, 1.0, 20.0])[:, None] + rng.standard_normal((3, 12))
    d, Cdd, steps = np.array([1.0, 2.0, 21.0]) + rng.standard_normal((4, 3)), 0.1 * np.eye(3), np.array([5, 10, 15, 20])
    loop = EnSRKF(parent_model=Lorenz63(dt=0.01), m=12, ensemble_psi0=E0)
    for k, n in enumerate(np.diff(np.r_[0, steps])):
        loop.forecast_step(Nt=int(n))
        loop.analysis_step(d=d[k], Cdd=Cdd)
        loop.model.current_state[0] *= 1.01
    est = EnSRKF(parent_model=Lorenz63(dt=0.01), m=12, ensemble_psi0=E0)
    seen = []

    def edit(ens):
        seen.append(ens)
        ens.model.current_state[0] *= 1.01
    out = run_da_loop(est, types.SimpleNamespace(t_obs=steps * 0.01, y_obs=d), Cdd=Cdd, t_extra=0., close=False,
                      after_analysis=edit)
    assert len(seen) == 4 and all(e is out for e in seen) and out is not est
    np.testing.assert_array_equal(out.model.hist, loop.model.hist)


def test_inflation_metrics_of_an_empty_history_is_a_factor_of_one():
    from romda.data_assimilation import inflation_metrics

    def ens(factors, rejected=()):
        return types.SimpleNamespace(inflation_history=types.SimpleNamespace(factors=list(factors)),
                                     rejected_analysis=types.SimpleNamespace(times=list(rejected)))

    assert inflation_metrics(ens([])) == dict(rho_mean=1., rho_first=1., rho_last=1., n_rejected=0)
    m = inflation_metrics(ens([1.08, 1.02, 1.005], rejected=[0.4]))
    assert np.isclose(m['rho_mean'], 1.035) and (m['rho_first'], m['rho_last']) == (1.08, 1.005)
    assert m['n_rejected'] == 1
