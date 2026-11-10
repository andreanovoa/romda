"""Unit tests for the bias estimators (ESN_bias: its Jacobian only; the rest is covered by the integration tests)."""
import numpy as np
import pytest
from romda.bias_estimators import ConstantBias, NoBias

Nq = 2


class TestConstantBias:

    def test_init_from_innovation(self):
        cb = ConstantBias(innovation=np.array([1., 2.]), t=0.0, dt=0.01)
        assert cb.Nq == Nq
        assert cb.N_dim == Nq  # unbiased observations by default
        np.testing.assert_allclose(cb.current_bias.ravel(), [1., 2.])

    def test_init_with_constant_k(self):
        cb = ConstantBias(innovation=np.zeros(Nq), t=0.0, dt=0.01, k=0.5)
        np.testing.assert_allclose(cb.current_bias.ravel(), 0.5)

    def test_forecast_is_constant(self):
        cb = ConstantBias(innovation=np.array([1., -1.]), t=0.0, dt=0.01, N_ens=3)
        state, t = cb.time_integrate(Nt=10)
        cb.update_history(state, t)
        assert state.shape == (10, cb.N_dim, 3)
        np.testing.assert_allclose(cb.current_bias, cb.get_bias(state, mean=True)[0])  # current_bias is the ensemble mean
        np.testing.assert_allclose(state, np.broadcast_to(state[0], state.shape))  # constant in time
        assert cb.current_time == pytest.approx(0.1)

    def test_state_derivative_is_zero(self):
        cb = ConstantBias(innovation=np.ones(Nq), t=0.0, dt=0.01)
        J = cb.state_derivative()
        assert J.shape == (Nq, Nq)
        np.testing.assert_allclose(J, 0.)

    def test_innovation_update(self):
        cb = ConstantBias(innovation=np.zeros(Nq), t=0.0, dt=0.01)
        new_innovation = np.array([[0.3], [0.7]]) * np.ones((1, 5))  # (Nq, 5) member innovations
        updated = cb.update_state_from_innovation(new_innovation)
        cb.update_history(updated, t=cb.current_time, modify_saved_states=True)
        np.testing.assert_allclose(cb.current_bias.ravel(), [0.3, 0.7])

    def test_member_innovations_with_biased_observations(self):
        # one bias state (N_ens = 1), N_dim = 2 Nq, and an (Nq, m) innovation ensemble: the mean
        # innovation goes to the innovation rows and the bias rows are left at their forecast
        cb = ConstantBias(innovation=np.array([1., 2.]), t=0.0, dt=0.01, biased_observations=True)
        members = np.array([[0.2, 0.4, 0.6], [1.0, 2.0, 3.0]])           # (Nq, m = 3)
        updated = cb.update_state_from_innovation(members)
        assert updated.shape == (2 * Nq, 1)
        np.testing.assert_allclose(updated[cb.observed_idx, 0], [0.4, 2.0])
        np.testing.assert_allclose(updated[cb.bias_idx, 0], cb.current_state[cb.bias_idx, 0])

    def test_biased_observations_state_is_doubled(self):
        cb = ConstantBias(innovation=np.ones(Nq), t=0.0, dt=0.01, biased_observations=True)
        assert cb.N_dim == 2 * Nq
        assert cb.current_state.shape[0] == 2 * Nq
        assert cb.current_bias.shape[0] == Nq
        assert cb.current_innovations.shape[0] == Nq


class TestNoBias:

    def test_bias_is_always_zero(self):
        nb = NoBias(innovation=np.ones(Nq), t=0.0, dt=0.01)
        np.testing.assert_allclose(nb.current_bias, 0.)

        state, t = nb.time_integrate(Nt=5)
        nb.update_history(state, t)
        np.testing.assert_allclose(nb.current_bias, 0.)

        updated = nb.update_state_from_innovation(np.ones((Nq, 1)))
        nb.update_history(updated, t=nb.current_time, modify_saved_states=True)
        np.testing.assert_allclose(nb.current_bias, 0.)


class TestBiasBase:

    def test_format_state_shapes(self):
        cb = ConstantBias(innovation=np.zeros(Nq), t=0.0, dt=0.01, N_ens=3)
        assert cb._format_state(np.zeros(Nq)).shape == (1, Nq, 3)
        assert cb._format_state(np.zeros((Nq, 3))).shape == (1, Nq, 3)
        assert cb._format_state(np.zeros((5, Nq, 3))).shape == (5, Nq, 3)

    def test_update_history_time_consistency(self):
        cb = ConstantBias(innovation=np.zeros(Nq), t=0.0, dt=0.1)
        state, t = cb.time_integrate(Nt=7)
        cb.update_history(state, t)
        assert len(cb.hist_t) == 8  # initial condition + 7 steps
        assert cb.current_time == pytest.approx(0.7)

    def test_history_grows_beyond_initial_capacity(self):
        cb = ConstantBias(innovation=np.zeros(Nq), t=0.0, dt=0.1, initial_capacity=5)
        for _ in range(4):
            state, t = cb.time_integrate(Nt=3)
            cb.update_history(state, t)
        assert len(cb.hist_t) == 13
        assert cb.current_time == pytest.approx(1.2)

    def test_innovation_reset_with_one_innovation_per_member(self):
        cb = ConstantBias(innovation=np.zeros(Nq), t=0.0, dt=0.1, N_ens=6)
        innovation = np.random.default_rng(2).normal(size=(Nq, 6))
        updated = cb.update_state_from_innovation(innovation)
        assert updated.shape == (cb.N_dim, 6)
        np.testing.assert_array_equal(updated[cb.observed_idx], innovation)
        np.testing.assert_array_equal(cb.current_bias, 0.0)        # the reset returns the state, it does not store it
        with pytest.raises(ValueError):                           # neither one per member nor a mean
            cb.update_state_from_innovation(np.ones((Nq, 4)))
        with pytest.raises(AssertionError):                       # one row per observable
            cb.update_state_from_innovation(np.ones((Nq + 1, 6)))

    @pytest.mark.parametrize('innovation', [np.array([0.3, 0.7]), np.array([[0.3], [0.7]]),
                                            np.array([[[0.3], [0.7]]])], ids=['(Nq,)', '(Nq, 1)', '(1, Nq, 1)'])
    def test_innovation_reset_with_a_mean_innovation(self, innovation):
        # One innovation for every member of the bias ensemble.
        cb = ConstantBias(innovation=np.ones(Nq), t=0.0, dt=0.1, N_ens=4)
        updated = cb.update_state_from_innovation(innovation)
        np.testing.assert_allclose(updated, np.repeat([[0.3], [0.7]], 4, axis=1))
        # One bias state and an innovation ensemble: the state takes the mean innovation.
        cb1 = ConstantBias(innovation=np.ones(Nq), t=0.0, dt=0.1)
        members = np.array([[0.2, 0.4, 0.6], [1.0, 2.0, 3.0]])
        np.testing.assert_allclose(cb1.update_state_from_innovation(members), [[0.4], [2.0]])

    def test_innovation_reset_feeds_update_state(self):
        cb = ConstantBias(innovation=np.zeros(Nq), t=0.0, dt=0.1, N_ens=3)
        cb.advance(Nt=2)
        cb.update_state(cb.update_state_from_innovation(np.array([0.5, -0.5])))
        assert cb.hist.shape[0] == 3 and cb.current_time == pytest.approx(0.2)    # overwrites, does not append
        np.testing.assert_allclose(cb.current_bias.ravel(), [0.5, -0.5])
        cb.advance(Nt=1)                                                        # the forecast holds the reset bias
        np.testing.assert_allclose(cb.current_bias.ravel(), [0.5, -0.5])

    def test_advance_and_update_state_wrap_the_history(self):
        cb = ConstantBias(innovation=np.ones(Nq), t=0.0, dt=0.1, N_ens=3)
        cb.advance(Nt=4)
        assert cb.current_time == pytest.approx(0.4) and cb.hist.shape[0] == 5
        cb.update_state(2.0 * np.ones((cb.N_dim, 3)))
        assert cb.hist.shape[0] == 5 and np.allclose(cb.current_bias, 2.0)


@pytest.mark.parametrize('biased_observations, inputs', [
    (True, 'all'),           # network input [b; i] (the romda default)
    (True, 'innovations'),   # network input i alone (as in biasda)
    (False, 'all'),          # network input i = b
])
def test_esn_bias_jacobian_matches_finite_differences(biased_observations, inputs):
    """J = d b_{k+1} / d q through one closed-loop step after the analysis sets i = d - q."""
    from types import SimpleNamespace

    from echostatenetwork import EchoStateNetwork
    from romda.bias_estimators.esn import ESN_bias

    Nq = 3
    N_dim = 2 * Nq if biased_observations else Nq
    rng = np.random.default_rng(0)
    data = rng.normal(size=(80, N_dim))
    kw = dict(observed_idx=np.arange(Nq, 2 * Nq)) if inputs == 'innovations' else {}
    esn = EchoStateNetwork(data.T, dt=1, N_units=12, upsample=1, t_train=40, t_val=15,
                           t_test=10, N_wash=5, hyperparameters_to_optimize=[], **kw)
    esn.train(data, plot_training=False)

    bias = SimpleNamespace(Nq=Nq, biased_observations=biased_observations, N_dim=N_dim,
                           N_hidden=esn.N_units, forecaster=esn)
    bias.bias_idx = ESN_bias.bias_idx.fget(bias)
    bias.observed_idx = ESN_bias.observed_idx.fget(bias)

    u = data[10].reshape(-1, 1)
    r = 0.1 * rng.normal(size=(esn.N_units, 1))
    d = rng.normal(size=(Nq, 1))

    def analysis(q):                                      # the analysis sets i = d - q
        u_a = u.copy()
        u_a[bias.observed_idx] = d - q
        return u_a

    def b_next(q):                                        # one closed-loop step
        return esn.step(esn.outputs_to_inputs(analysis(q)), r)[0][bias.bias_idx]

    q = rng.normal(size=(Nq, 1))
    bias.current_state = np.vstack([analysis(q), r])
    J = ESN_bias.state_derivative(bias)

    eps = 1e-5
    J_fd = np.hstack([(b_next(q + eps * e) - b_next(q - eps * e)) / (2 * eps)
                      for e in np.eye(Nq)[:, :, None]])
    assert J.shape == (Nq, Nq)
    np.testing.assert_allclose(J, J_fd, rtol=1e-5, atol=1e-8)
