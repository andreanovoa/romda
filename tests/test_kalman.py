"""The Kalman filter (`romda.estimators.KalmanFilter`, the estimator that owns a mean and a covariance) and
`LinearModel`, its linear model: convergence, a model in the loop, joint parameter estimation, inflation and the
bias-aware update.

The extended and unscented types, the dual scheme, localization and the Kalman smoother live in biasda
and raise NotImplementedError here.
"""

import numpy as np
import pytest
from romda.bias_estimators import ConstantBias
from romda.estimators import KalmanFilter
from romda.models.data_driven import LinearModel

THETA = 2 * np.pi / 40
F = 0.998 * np.array([[np.cos(THETA), -np.sin(THETA)],
                      [np.sin(THETA), np.cos(THETA)]])


def _rotation(theta=0.3, damping=0.98):
    c, s = np.cos(theta), np.sin(theta)
    return damping * np.array([[c, -s], [s, c]])


def test_kf_f_only_converges():
    """Linear KF without a model: track a decaying rotation from noisy obs."""
    rng = np.random.default_rng(0)
    theta = 0.1
    F = 0.99 * np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta),  np.cos(theta)]])
    M = np.array([[1.0, 0.0]])          # observe first component only
    Cdd = np.array([[0.05]])

    kf = KalmanFilter(Cdd=Cdd, psi0=np.zeros(2), Cpp0=np.eye(2), M=M, F=F, Q=1e-4 * np.eye(2))
    assert (kf.N, kf.Nq) == (2, 1)

    x = np.array([1.0, 0.0])            # truth
    for _ in range(100):
        x = F @ x
        kf.forecast_step()
        d = M @ x + rng.normal(0, np.sqrt(Cdd[0, 0]))
        kf.analysis_step(d=d, Cdd=Cdd)

    assert np.linalg.norm(kf.current_state - x) < 0.5
    assert np.all(np.linalg.eigvalsh(kf.Cpp) > 0)          # covariance stays PD
    assert kf.current_time == 100                            # F-only time counts steps
    assert len(kf.assimilated_data.times) == 100

    with pytest.raises(ValueError):
        kf.forecast_step(t_end=1.0)     # t_end unsupported without a model


def test_kf_with_model_roundtrip():
    """KF wrapping a nonlinear model: analysis is written back to the model."""
    from romda.models.physical import Lorenz63

    model = Lorenz63(dt=0.01)
    N = model.Nphi
    kf = KalmanFilter(Cdd=0.1 * np.eye(model.Nq),
                      psi0=np.asarray(model.current_state, float).ravel()[:N],
                      Cpp0=np.eye(N), model=model, Q=1e-3 * np.eye(N))

    kf.forecast_step(Nt=10)
    psi_f = kf.current_state.copy()
    d = psi_f[:model.Nq] + 0.5
    kf.analysis_step(d=d, Cdd=0.1 * np.eye(model.Nq))

    assert not np.allclose(kf.current_state, psi_f)          # update moved the mean
    model_state = np.asarray(model.current_state, float)
    model_state = model_state.mean(axis=-1) if model_state.ndim > 1 else model_state
    np.testing.assert_allclose(model_state, kf.current_state)  # written back to history
    kf.close()


def test_the_options_of_the_kalman_filter_are_checked():
    """N and Nq follow from psi0 and Cdd, they are not keywords; the options of biasda raise NotImplementedError,
    and the parameter scheme needs est_alpha, and est_alpha a scheme."""
    kw = dict(Cdd=np.eye(1), psi0=np.zeros(2), Cpp0=np.eye(2), M=np.array([[1.0, 0.0]]), F=np.eye(2))
    assert (KalmanFilter(**kw).N, KalmanFilter(**kw).Nq) == (2, 1)
    assert KalmanFilter(type='Linear', **kw).type == 'linear'
    for bad in (dict(N=2, Nq=1), dict(unknown=1), dict(inflation_factor=1.1)):
        with pytest.raises(TypeError):
            KalmanFilter(**bad, **kw)
    with pytest.raises(ValueError):
        KalmanFilter(type='ensemble', **kw)
    for type in ('extended', 'unscented', 'Unscented'):
        with pytest.raises(NotImplementedError, match='biasda'):
            KalmanFilter(type=type, **kw)
    with pytest.raises(NotImplementedError, match='biasda'):
        KalmanFilter(**kw, est_alpha=['a'], parameter_estimation='dual')
    for option in (dict(localization=1.0), dict(bias_estimation='two-stage-analysis'), dict(f=lambda x: x)):
        with pytest.raises(NotImplementedError, match='biasda'):
            KalmanFilter(**kw, **option)
    # the adaptive and innovation inflations need an ensemble (the Kalman filter carries no sampling error)
    for inflation in ('adaptive', 'innovation'):
        with pytest.raises(NotImplementedError):
            KalmanFilter(**kw, inflation=inflation)
    with pytest.raises(ValueError):
        KalmanFilter(**KW3, parameter_estimation='joint')       # no est_alpha to estimate
    with pytest.raises(ValueError):
        KalmanFilter(**KW3, est_alpha=['a'])                    # est_alpha without a scheme
    with pytest.raises(ValueError):                             # the regularization needs a bias
        KalmanFilter(**KW3, bias_regularization=0.5).analysis_step(d=np.zeros(3), Cdd=np.eye(3))


def test_callable_measurement_operator_and_close_without_a_model():
    M = np.array([[1.0, 0.0]])
    kw = dict(Cdd=np.eye(1), psi0=np.array([1.0, -1.0]), Cpp0=np.eye(2), F=np.eye(2))
    kf_matrix, kf_callable = KalmanFilter(M=M, **kw), KalmanFilter(M=lambda x: M @ x, **kw)
    assert np.allclose(kf_callable.M_mat, kf_matrix.M_mat)
    for kf in (kf_matrix, kf_callable):
        kf.forecast_step()
        kf.analysis_step(d=np.array([0.5]), Cdd=np.eye(1))
    assert np.array_equal(kf_callable.current_state, kf_matrix.current_state)
    assert np.array_equal(kf_callable.Cpp, kf_matrix.Cpp)
    kf_matrix.close()


def test_kf_forecast_advances_the_bias_model():
    bias = ConstantBias(innovation=np.ones(2), t=0.0, dt=1.0)
    kf = KalmanFilter(Cdd=np.eye(2), psi0=np.zeros(2), Cpp0=np.eye(2), M=np.eye(2), F=np.eye(2), bias=bias)
    assert kf.is_bias_aware
    kf.forecast_step(Nt=5)
    assert kf.bias.current_time == 5.0 and kf.bias.hist.shape[0] == 6
    assert np.allclose(kf.bias.current_bias.ravel(), 1.0)
    kf.analysis_step(d=np.array([0.5, 0.5]), Cdd=np.eye(2))
    # the bias holds the analysis innovation: the corrected observables match the observation
    assert np.allclose(kf.M_mat @ kf.current_state + kf.bias.current_bias.ravel(), [0.5, 0.5])


def test_bias_aware_kf_with_a_constant_bias_assimilates_the_corrected_observation():
    # A constant bias has a zero Jacobian, H = (I + J) M = M, and its gamma rows carry no state
    # covariance, so the bias-aware analysis is the bias-blind analysis of d - b.
    b0, d, Cdd = np.array([0.3, -0.2]), np.array([1.0, 2.0]), 0.1 * np.eye(2)
    kw = dict(Cdd=Cdd, psi0=np.array([0.5, 0.5, 0.0]), Cpp0=np.diag([1.0, 2.0, 0.5]),
              M=np.hstack([np.eye(2), np.zeros((2, 1))]), F=np.eye(3))
    for gamma in (None, 0.5):
        aware = KalmanFilter(**kw, bias=ConstantBias(innovation=b0, t=0.0, dt=1.0), bias_regularization=gamma)
        blind = KalmanFilter(**kw)
        aware.analysis_step(d=d, Cdd=Cdd)
        blind.analysis_step(d=d - b0, Cdd=Cdd)
        assert np.allclose(aware.current_state, blind.current_state, atol=1e-12)
        assert np.allclose(aware.Cpp, blind.Cpp, atol=1e-12)


KW3 = dict(Cdd=np.eye(3), psi0=np.ones(3), Cpp0=np.eye(3), M=np.eye(3), F=np.eye(3))


F_FORCED = np.array([[0.9, 0.1, 0.5], [-0.1, 0.9, 0.0], [0.0, 0.0, 1.0]])


M_FORCED = np.hstack([np.eye(2), np.zeros((2, 1))])


def _analysis_cov(**kw):
    est = KalmanFilter(**{**KW3, 'F': F_FORCED}, **kw)
    est.forecast_step()
    est.analysis_step(d=np.zeros(3), Cdd=np.eye(3))
    return est.Cpp


def test_kalman_inflation_scales_the_analysis_covariance():
    rho = 1.1
    assert np.allclose(_analysis_cov(inflation=rho), rho**2 * _analysis_cov())
    # inflate_parameters=False: the parameter block keeps its analysis covariance.
    kw = dict(est_alpha=['a'], parameter_estimation='joint')
    P, P_inf = _analysis_cov(**kw), _analysis_cov(**kw, inflation=rho, inflate_parameters=False)
    assert np.allclose(P_inf[:2, :2], rho**2 * P[:2, :2])
    assert np.allclose(P_inf[:2, 2], rho * P[:2, 2]) and np.isclose(P_inf[2, 2], P[2, 2])


def _forced_twin(n_obs=40, seed=0, **kw):
    rng = np.random.default_rng(seed)
    truth = np.array([1.0, 0.0, 2.0])
    est = KalmanFilter(Cdd=0.04 * np.eye(2), psi0=np.array([0.0, 0.0, 0.0]),
                       Cpp0=np.diag([1.0, 1.0, 4.0]), M=M_FORCED, F=F_FORCED, **kw)
    for _ in range(n_obs):
        truth = F_FORCED @ truth
        est.forecast_step()
        est.analysis_step(d=M_FORCED @ truth + rng.normal(scale=0.2, size=2), Cdd=0.04 * np.eye(2))
    return est


def test_joint_kalman_parameter_estimation_is_the_augmented_filter():
    # Joint estimation of a parameter that the forecast holds constant is the Kalman filter of
    # the augmented state; it recovers the forcing. start_param holds the parameter at its forecast.
    joint = _forced_twin(est_alpha=['alpha'], parameter_estimation='joint')
    assert np.allclose(joint.current_state, _forced_twin().current_state, atol=1e-12)
    assert np.allclose(joint.Cpp, _forced_twin().Cpp, atol=1e-12)
    assert abs(joint.current_state[2] - 2.0) < 0.2
    frozen = _forced_twin(n_obs=3, est_alpha=['alpha'], parameter_estimation='joint', start_param=5)
    assert frozen.current_state[2] == 0.0
    released = _forced_twin(n_obs=6, est_alpha=['alpha'], parameter_estimation='joint', start_param=5)
    assert released.current_state[2] != 0.0


def test_kalman_filter_keeps_a_symmetric_positive_definite_covariance():
    # The Joseph form P - K Cxy^T - Cxy K^T + K Cyy K^T does not contract the antisymmetric round-off of P,
    # and an unstable forecast amplifies it; the analysis symmetrizes the posterior (biasda 491b936, where
    # the unsymmetrized covariance of the regularized EKF turned indefinite after 85 cycles). Here a growing
    # rotation, a joint parameter with a small prior variance and a regularized constant bias.
    F = np.eye(3)
    F[:2, :2] = _rotation(theta=0.7, damping=1.15)
    F[:2, 2] = [0.5, -0.3]
    M = np.hstack([np.eye(2), np.zeros((2, 1))])
    noise, b0 = 0.3, np.array([1.0, 2.0])
    kf = KalmanFilter(Cdd=noise**2 * np.eye(2), psi0=np.array([1.3, -0.7, 1.0]), Cpp0=np.diag([25.0, 25.0, 1e-6]),
                      M=M, F=F, Q=np.diag([1e-3, 1e-3, 0.0]), est_alpha=['alpha'], parameter_estimation='joint',
                      bias=ConstantBias(innovation=b0, t=0.0, dt=1.0), bias_regularization=0.5)
    rng, x = np.random.default_rng(0), np.array([1.0, 0.0, 1.2])
    asymmetry, smallest = [], []
    for _ in range(500):
        x = F @ x
        x[:2] /= max(1.0, np.linalg.norm(x[:2]) / 10.0)   # keep the true state bounded
        kf.forecast_step()
        kf.analysis_step(d=M @ x + b0 + rng.normal(scale=noise, size=2), Cdd=noise**2 * np.eye(2))
        P = kf.Cpp
        asymmetry.append(np.max(np.abs(P - P.T)))
        smallest.append(np.linalg.eigvalsh(P).min())
    assert max(asymmetry) == 0.0 and min(smallest) > 0.0


def test_linear_model_regressions():
    """Fixes of LinearModel surfaced by tutorial 15."""
    # Q != 0 used to crash: multivariate_normal was called with a 2-D mean
    model = LinearModel(F=F, psi0=np.zeros((2, 5)), Q=0.1 * np.eye(2), seed=0)
    psi, _ = model.time_integrate(Nt=50)
    assert psi.shape == (50, 2, 5)
    assert psi[-1].std() > 0                      # noise actually entered
    assert not np.allclose(psi[:, :, 0], psi[:, :, 1])  # per-member draws differ
    # get_observable_hist returned only the last state
    M_obs = np.array([[1.0, 0.0]])
    model = LinearModel(F=F, M_obs=M_obs, psi0=np.array([1.0, 0.0]))
    model.update_history(*model.time_integrate(Nt=10))
    y = model.get_observable_hist()
    assert y.shape == (model.hist.shape[0], 1, 1)
    np.testing.assert_allclose(y[:, 0, 0], model.hist[:, 0, 0])
    assert model.get_observables().shape == (1, 1)
    model = LinearModel(F=F, psi0=np.array([1.0, 0.0]))
    assert model.filename                          # crashed: Q_noise had no class default
    model.init_ensemble(m=4, std_phi=0.1)          # crashed: t_transient == 0 -> psi[-1] on empty
    assert model.hist.shape[-1] == 4


def test_kf_covariance_uses_model_F_per_step():
    Nt_obs, Cpp0 = 5, 0.5 * np.eye(2)
    model = LinearModel(F=F, M_obs=np.array([[1.0, 0.0]]), psi0=np.array([1.0, 0.0]), dt=1.0)
    kf = KalmanFilter(Cdd=np.eye(1), psi0=np.array([1.0, 0.0]), Cpp0=Cpp0,
                      M=np.array([[1.0, 0.0]]), model=model)   # no F_jac: defaults to model.F
    kf.forecast_step(t_end=model.current_time + Nt_obs * model.dt)
    Fk = np.linalg.matrix_power(F, Nt_obs)
    np.testing.assert_allclose(kf.Cpp, Fk @ Cpp0 @ Fk.T, atol=1e-12)  # public accessor, applied per step


def test_kf_matches_hand_rolled_update():
    M = np.array([[1.0, 0.0]])
    Cdd, Cpp0, psi0 = np.array([[0.04]]), 0.5 * np.eye(2), np.array([1.0, 0.0])
    model = LinearModel(F=F, M_obs=M, psi0=psi0)
    kf = KalmanFilter(Cdd=Cdd, psi0=psi0, Cpp0=Cpp0, M=M, model=model)
    kf.forecast_step(t_end=1.0)
    d = np.array([0.9])
    psi_a = kf.analysis_step(d, Cdd, return_analysis=True)

    Cf = F @ Cpp0 @ F.T
    K = Cf @ M.T @ np.linalg.inv(M @ Cf @ M.T + Cdd)
    expected = F @ psi0 + K @ (d - M @ (F @ psi0))
    np.testing.assert_allclose(psi_a, expected)
