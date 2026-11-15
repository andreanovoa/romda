"""`romda.observations`: the Observations (truth/data generation) class, the observation covariances and the sensor
index sets."""
import types

import numpy as np
import pytest
from romda.models.physical import VdP
from romda.observations import Observations, esn_observed_idx, measured_idx


def make_truth(**kwargs):
    defaults = dict(model=VdP, t_start=0.6, t_stop=0.8, t_max=1.0, Nt_obs=30)
    defaults.update(kwargs)
    return Observations(**defaults)


def test_a_model_record():
    truth = make_truth()
    assert truth.y_true.ndim == 3
    assert truth.y_raw.shape == truth.y_true.shape
    assert len(truth.t_true) == truth.y_true.shape[0]
    assert truth.y_obs.shape[0] == len(truth.t_obs)
    assert truth.t_obs[0] >= truth.t_start - truth.dt
    assert truth.t_obs[-1] <= truth.t_stop + truth.dt
    # no bias, no noise: the raw data are the truth
    np.testing.assert_allclose(truth.y_raw, truth.y_true)
    np.testing.assert_allclose(truth.b_true, 0.)
    # frozen after init
    with pytest.raises(AttributeError):
        truth.y_raw = np.zeros(3)
    n_obs_before = len(truth.t_obs)
    truth.update_obs_idx(Nt_obs=60)
    assert len(truth.t_obs) < n_obs_before


def test_manual_bias_types():
    for bias_name in ('linear', 'linear_offset', 'periodic', 'cosine', 'cosine_squared', 'time'):
        truth = make_truth(manual_bias=bias_name)
        assert truth.name_bias == bias_name
        assert np.isfinite(truth.b_true).all()
    # regression: y_raw must be the biased truth when add_noise=False
    truth = make_truth(manual_bias='linear')
    assert truth.y_raw.shape == truth.y_true.shape
    assert np.any(truth.b_true != 0.)
    np.testing.assert_allclose(truth.y_raw, truth.y_true)


def test_manual_bias_formulas():
    for bias_name, formula in (('linear_offset', lambda y, y_max: .3 * y_max + .3 * y),
                               ('cosine_squared', lambda y, y_max: .3 * y_max * (np.cos((y / y_max) ** 2) + 1))):
        truth = make_truth(manual_bias=bias_name)
        y = truth.y_true - truth.b_true
        np.testing.assert_allclose(truth.b_true, formula(y, np.max(y, axis=0)), rtol=1e-12)

    def my_bias(y, t):
        return 0.1 * y, 'my_bias'
    truth = make_truth(manual_bias=my_bias)
    assert truth.name_bias == 'my_bias'
    np.testing.assert_allclose(truth.b_true, (truth.y_true - truth.b_true) * 0.1, rtol=1e-10)


def test_additive_gaussian_noise():
    truth = make_truth(add_noise=True, noise_level=0.05, noise_type='gauss, add')
    noise = truth.y_raw - truth.y_true
    assert np.std(noise) > 0.
    # noise std should be roughly noise_level * max|y|
    expected = 0.05 * np.max(abs(truth.y_true))
    assert np.std(noise) == pytest.approx(expected, rel=0.5)


def test_provided_data():
    t = np.linspace(0., 1., 101)
    y = np.sin(2 * np.pi * t)[:, np.newaxis]
    truth = Observations(model=None, y_true=y, t_true=t, Nt_obs=10)
    assert truth.y_true.shape == (101, 1, 1)
    assert len(truth.t_obs) > 0


def test_a_record_from_a_file_is_cut_at_t_min_and_t_max(tmp_path):
    """A measured record is kept between t_min and t_max, as a model record is, or whole without t_max. The bias
    estimators train on the last steps of the record: uncut, the annular case trained on the end of the 120 s of
    data, where the mean of the raw pressure differs from that of the assimilation window."""
    import scipy.io as sio

    t = np.arange(1000) * 1e-3
    y = np.stack([t, 2. * t], axis=1)                       # the value is the time: easy to check
    sio.savemat(tmp_path / 'annular_record.mat', dict(t=t, y_raw=y + 5., y_filtered=y))
    for t_min in (0., 0.2):
        truth = Observations(model=str(tmp_path / 'annular_record'), t_min=t_min, t_start=0.3, t_stop=0.4,
                             t_max=0.6, Nt_obs=10)
        n = round((0.6 - t_min) / 1e-3)
        assert truth.y_true.shape == truth.y_raw.shape == (n, 2, 1)
        np.testing.assert_allclose(truth.t_true, np.arange(n) * 1e-3)          # the clock starts at t_min
        np.testing.assert_allclose(truth.y_true[:, 0, 0], t_min + np.arange(n) * 1e-3)
        np.testing.assert_allclose(truth.y_raw[..., 0], truth.y_true[..., 0] + 5.)
        assert truth.t_obs[0] == pytest.approx(0.3 - t_min) and truth.t_obs[-1] <= 0.4 - t_min + 1e-9

    truth = Observations(model=str(tmp_path / 'annular_record'), t_start=0.5, t_stop=0.6, Nt_obs=10)
    assert len(truth.t_true) == 1000


def test_observation_covariances():
    """`std_fraction_covariance`: `fraction` of the standard deviation of the record, per observed component, or one
    value from every component of the record (the Lorenz-96 sites share their statistics) on the observed ones.
    `variance_fraction_covariance`: C_dd = fraction max|d|^2, the maximum of each observable (v1.0, the default of
    run_da_loop), or one maximum for every observable (the paper)."""
    from romda.data_assimilation import _legacy_observation_covariance
    from romda.observations import std_fraction_covariance, variance_fraction_covariance

    y = np.random.default_rng(0).standard_normal((400, 4)) * np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(np.diag(std_fraction_covariance(y, 0.5)), (0.5 * y.std(axis=0)) ** 2)
    np.testing.assert_allclose(np.diag(std_fraction_covariance(y, 0.5, idx=[1, 3])), (0.5 * y[:, [1, 3]].std(0)) ** 2)
    C = std_fraction_covariance(y, 0.2, idx=[0, 2], identical=True)
    assert C.shape == (2, 2) and np.allclose(np.diag(C), (0.2 * y.std()) ** 2) and C[0, 1] == 0.0
    assert std_fraction_covariance(y[:, :, None], 0.5).shape == (4, 4)          # (Nt, N, 1) records are flattened

    truth = types.SimpleNamespace(y_obs=np.array([[1., -4.], [-2., 3.]]))
    np.testing.assert_allclose(variance_fraction_covariance(truth, 0.1), np.diag([0.4, 1.6]))
    np.testing.assert_allclose(variance_fraction_covariance(truth, 0.1), _legacy_observation_covariance(truth, 0.1))
    np.testing.assert_allclose(variance_fraction_covariance(truth, 0.1, identical=True), 1.6 * np.eye(2))


def test_sensor_index_sets():
    assert measured_idx(None, 4) == [0, 1, 2, 3]
    assert measured_idx(3, 3) == [0, 1, 2]
    # line topology includes the endpoints ...
    assert measured_idx(2, 3) == [0, 2]
    assert measured_idx(3, 10) == [0, 4, 9]
    # ... a ring excludes the far end (site N_dim IS site 0)
    assert measured_idx(2, 10, ring=True) == [0, 5]
    assert measured_idx(5, 10, ring=True) == [0, 2, 4, 6, 8]
    assert measured_idx(10, 10, ring=True) == list(range(10))
    # explicit indices pass through as ints, order preserved
    assert measured_idx([2, 0], 5) == [2, 0]
    # digit strings are index sets, not int counts ('01' must not collapse to int('01'))
    assert measured_idx('01', 3) == [0, 1]
    assert measured_idx('12', 3) == [1, 2]
    assert measured_idx('02', 3) == [0, 2]
    assert measured_idx('0', 3) == [0]
    # the inputs of the network: with the Bayesian update it keeps every input regardless of the sensors; in open
    # loop it is trained to live without the unmeasured ones
    assert esn_observed_idx(2, 5, bayesian_update=True) == [0, 1, 2, 3, 4]
    assert esn_observed_idx(2, 5, bayesian_update=False) == [0, 4]
    assert esn_observed_idx(None, 3, bayesian_update=False) == [0, 1, 2]
