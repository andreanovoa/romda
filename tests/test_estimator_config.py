"""Config kwargs must reach the estimator, and the parameter-freeze path must run."""
import numpy as np
import pytest
from romda.bias_estimators import ConstantBias
from romda.estimators import EnsembleFilter, EnSRKF, rBA_EnKF, rEnKF
from romda.models.physical import Lorenz63

KW = dict(parent_model=Lorenz63, dt=0.015, m=5, std_phi=0.2,
          std_alpha=dict(rho=(25., 35.), beta=(2, 4), sigma=(5, 15)),
          observe_dims=[0, 1, 2])


@pytest.fixture
def Cdd():
    return np.eye(3) * 0.5


def test_base_class_config_kwargs_are_applied():
    """These live on Estimator, not on the leaf class, and used to be dropped."""
    e = EnSRKF(inflation=1.01, start_param=3, start_bias=2, **KW)
    assert (e.inflation_factor, e.start_param, e.start_bias) == (1.01, 3, 2)
    # The named entry point is the square-root type with the parameters estimated jointly.
    assert (e.type, e.parameter_estimation) == ('sqrt', 'joint')
    e.close()


def test_ensemble_filter_asks_for_the_parameter_scheme():
    with pytest.raises(ValueError):
        EnsembleFilter(**KW)                       # est_alpha set, no scheme
    e = EnsembleFilter(type='sqrt', parameter_estimation='joint', **KW)
    assert (e.type, e.parameter_estimation) == ('sqrt', 'joint')
    e.close()


def test_attribute_spellings_are_not_keywords():
    """One spelling per option, as in biasda: the stored attributes are not keywords."""
    for bad in (dict(inflation_factor=1.01), dict(regularization_factor=2.0)):
        with pytest.raises(TypeError):
            EnSRKF(**bad, **KW)
    for cls in (rEnKF, rBA_EnKF):
        with pytest.raises(TypeError):
            cls(parent_bias=ConstantBias, gamma=1.0, **KW)
    e = rBA_EnKF(parent_bias=ConstantBias, **KW)           # the paper's name of rEnKF, gamma = 1 by default
    assert isinstance(e, rEnKF) and e.gamma == 1.0 and e.type == 'stochastic'
    e.close()


def test_frozen_parameters_survive_the_analysis(Cdd):
    e = EnSRKF(activate_parameter_estimation=False, **KW)
    alpha0 = e.current_state[-e.Na:, :].copy()
    e.forecast_step(t_end=0.15)
    alpha_f = e.current_state[-e.Na:, :].copy()

    Aa = e.analysis_step(d=e.model.get_observables().mean(-1), Cdd=Cdd, return_analysis=True)
    e.model.close()

    assert Aa.shape[0] == e.Nphi + e.Na, 'analysis must carry the full state'
    np.testing.assert_allclose(e.current_state[-e.Na:, :], alpha_f)
    np.testing.assert_allclose(alpha_f, alpha0)          # parameters are static in the forecast
    assert not np.allclose(e.current_state[:e.Nphi, :], Aa[:e.Nphi, :] * 0), 'state was updated'


def test_update_reservoir_false_freezes_non_physical_rows(Cdd):
    """`update_reservoir=False` leaves the 'reservoir' rows exactly at their
    forecast value; only the physical rows should move.

    Lorenz63 has no real reservoir -- N_dim/N_units are monkeypatched onto the
    model purely to exercise the row-split `analysis_step` applies for an ESN.
    """
    e = EnSRKF(update_reservoir=False, **KW)
    e.model.N_dim, e.model.N_units = 1, e.Nphi - 1

    e.forecast_step(t_end=0.15)
    forecast = e.current_state.copy()
    e.analysis_step(d=e.model.get_observables().mean(-1), Cdd=Cdd)

    reservoir_rows = slice(e.model.N_dim, e.model.N_dim + e.model.N_units)
    np.testing.assert_allclose(e.current_state[reservoir_rows, :], forecast[reservoir_rows, :])
    assert not np.allclose(e.current_state[:e.model.N_dim, :], forecast[:e.model.N_dim, :])
    e.model.close()


def test_update_reservoir_false_freezes_every_patch_of_a_parallel_esn(Cdd):
    """A parallel ESN carries `N_r` = patches x `N_units` reservoir rows: all of them stay at the forecast."""
    e = EnSRKF(update_reservoir=False, **KW)
    e.model.N_dim, e.model.N_units, e.model.N_r = 1, 1, e.Nphi - 1

    e.forecast_step(t_end=0.15)
    forecast = e.current_state.copy()
    e.analysis_step(d=e.model.get_observables().mean(-1), Cdd=Cdd)

    np.testing.assert_allclose(e.current_state[1:e.Nphi, :], forecast[1:e.Nphi, :])
    e.model.close()


def test_start_bias_blind_window_uses_plain_enkf(Cdd):
    e = rEnKF(parent_bias=ConstantBias, bias_regularization=1.0, start_bias=1, **KW)
    e.forecast_step(t_end=0.15)
    Af = np.vstack((e.current_state, e.model.get_observables()))
    d = e.model.get_observables().mean(-1)

    e.rng = np.random.default_rng(7)
    Aa_blind = e._analysis_kernel(Af, d, Cdd)       # 0 analyses done < start_bias
    e.rng = np.random.default_rng(7)
    Aa_enkf = e._plain_kernel(Af, d, Cdd)
    np.testing.assert_allclose(Aa_blind, Aa_enkf)   # blind window == plain EnKF

    e.assimilated_data = (d, 0.15)                  # window over
    e.rng = np.random.default_rng(7)
    Aa_aware = e._analysis_kernel(Af, d, Cdd)
    assert not np.allclose(Aa_aware, Aa_enkf)       # bias-aware update kicked in
    e.model.close()


def test_bias_attached_after_construction_is_checked_and_regularized(Cdd):
    """romda builds the ESN bias from the estimator's model, so the bias may arrive after
    construction; the regularized update then needs it by the first analysis."""
    e = EnSRKF(bias_regularization=1.0, **KW)
    assert not e.is_bias_aware
    e.forecast_step(t_end=0.15)
    d = e.model.get_observables().mean(-1)
    with pytest.raises(ValueError):
        e.analysis_step(d=d, Cdd=Cdd)
    e.bias = ConstantBias(innovation=np.zeros(3), t=e.current_time, dt=e.model.dt)
    assert e.is_bias_aware
    e.analysis_step(d=d, Cdd=Cdd)
    assert len(e.assimilated_data.times) == 1
    # the analysis resets the bias to the innovation of the analysed observables
    np.testing.assert_allclose(e.bias.current_bias.ravel(), d - e.model.get_observables().mean(-1), atol=1e-12)
    with pytest.raises(ValueError):
        e.bias = 'not a bias'
    e.model.close()


def test_start_param_freezes_then_releases_parameters(Cdd):
    e = EnSRKF(start_param=2, **KW)
    alpha0 = e.current_state[-e.Na:, :].copy()
    for k in range(3):
        e.forecast_step(t_end=round(0.15 * (k + 1), 6))
        e.analysis_step(d=e.model.get_observables().mean(-1), Cdd=Cdd)
        frozen = bool(np.allclose(e.current_state[-e.Na:, :], alpha0))
        assert frozen == (k < 2), f'analysis {k}: expected frozen={k < 2}'
    e.model.close()
