"""The linear Kalman filter: convergence, joint parameter estimation, inflation and the bias-aware update.

The extended and unscented types, the dual scheme, localization and the Kalman smoother live in biasda
and raise NotImplementedError here.
"""

import numpy as np
import pytest
from romda.bias_estimators import ConstantBias
from romda.estimators import KalmanFilter


def _rotation(theta=0.3, damping=0.98):
    c, s = np.cos(theta), np.sin(theta)
    return damping * np.array([[c, -s], [s, c]])


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


def test_the_options_of_biasda_raise_not_implemented():
    kw = dict(Cdd=np.eye(1), psi0=np.zeros(2), Cpp0=np.eye(2), M=np.array([[1.0, 0.0]]), F=np.eye(2))
    assert KalmanFilter(type='Linear', **kw).type == 'linear'
    for type in ('extended', 'unscented', 'Unscented'):
        with pytest.raises(NotImplementedError, match='biasda'):
            KalmanFilter(type=type, **kw)
    with pytest.raises(ValueError):
        KalmanFilter(type='ensemble', **kw)
    with pytest.raises(NotImplementedError, match='biasda'):
        KalmanFilter(**kw, est_alpha=['a'], parameter_estimation='dual')
    for option in (dict(localization=1.0), dict(bias_estimation='two-stage-analysis'), dict(f=lambda x: x)):
        with pytest.raises(NotImplementedError, match='biasda'):
            KalmanFilter(**kw, **option)


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


@pytest.mark.parametrize('gamma', [None, 0.5])
def test_bias_aware_kf_with_a_constant_bias_assimilates_the_corrected_observation(gamma):
    # A constant bias has a zero Jacobian, H = (I + J) M = M, and its gamma rows carry no state
    # covariance, so the bias-aware analysis is the bias-blind analysis of d - b.
    b0, d, Cdd = np.array([0.3, -0.2]), np.array([1.0, 2.0]), 0.1 * np.eye(2)
    kw = dict(Cdd=Cdd, psi0=np.array([0.5, 0.5, 0.0]), Cpp0=np.diag([1.0, 2.0, 0.5]),
              M=np.hstack([np.eye(2), np.zeros((2, 1))]), F=np.eye(3))
    aware = KalmanFilter(**kw, bias=ConstantBias(innovation=b0, t=0.0, dt=1.0), bias_regularization=gamma)
    blind = KalmanFilter(**kw)
    aware.analysis_step(d=d, Cdd=Cdd)
    blind.analysis_step(d=d - b0, Cdd=Cdd)
    assert np.allclose(aware.current_state, blind.current_state, atol=1e-12)
    assert np.allclose(aware.Cpp, blind.Cpp, atol=1e-12)


KW3 = dict(Cdd=np.eye(3), psi0=np.ones(3), Cpp0=np.eye(3), M=np.eye(3), F=np.eye(3))
# A forced linear system, phi_{k+1} = A phi_k + g alpha, with the constant forcing alpha as the third component.
F_FORCED = np.array([[0.9, 0.1, 0.5], [-0.1, 0.9, 0.0], [0.0, 0.0, 1.0]])
M_FORCED = np.hstack([np.eye(2), np.zeros((2, 1))])


def test_kalman_parameter_estimation_configuration_errors():
    with pytest.raises(ValueError):
        KalmanFilter(**KW3, parameter_estimation='joint')       # no est_alpha to estimate
    with pytest.raises(ValueError):
        KalmanFilter(**KW3, est_alpha=['a'])                    # est_alpha without a scheme
    with pytest.raises(ValueError):                             # the regularization needs a bias
        KalmanFilter(**KW3, bias_regularization=0.5).analysis_step(d=np.zeros(3), Cdd=np.eye(3))


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
    # the augmented state; it recovers the forcing.
    joint = _forced_twin(est_alpha=['alpha'], parameter_estimation='joint')
    assert np.allclose(joint.current_state, _forced_twin().current_state, atol=1e-12)
    assert np.allclose(joint.Cpp, _forced_twin().Cpp, atol=1e-12)
    assert abs(joint.current_state[2] - 2.0) < 0.2


def test_joint_kalman_start_param_holds_the_parameter_at_its_forecast():
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
