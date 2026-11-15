"""Unit tests for the bias estimators (ESN_bias: its Jacobian, the input of its network and its training members)."""
import numpy as np
import pytest
from romda.bias_estimators import ConstantBias, NoBias

Nq = 2


def test_constant_bias_init():
    cb = ConstantBias(innovation=np.array([1., 2.]), t=0.0, dt=0.01)
    assert cb.Nq == Nq
    assert cb.N_dim == Nq  # unbiased observations by default
    np.testing.assert_allclose(cb.current_bias.ravel(), [1., 2.])
    cb = ConstantBias(innovation=np.zeros(Nq), t=0.0, dt=0.01, k=0.5)
    np.testing.assert_allclose(cb.current_bias.ravel(), 0.5)
    # biased observations: the state is [b; i]
    cb = ConstantBias(innovation=np.ones(Nq), t=0.0, dt=0.01, biased_observations=True)
    assert cb.N_dim == 2 * Nq
    assert cb.current_state.shape[0] == 2 * Nq
    assert cb.current_bias.shape[0] == Nq
    assert cb.current_innovations.shape[0] == Nq


def test_constant_bias_forecast_and_history():
    cb = ConstantBias(innovation=np.array([1., -1.]), t=0.0, dt=0.01, N_ens=3)
    assert cb._format_state(np.zeros(Nq)).shape == (1, Nq, 3)
    assert cb._format_state(np.zeros((Nq, 3))).shape == (1, Nq, 3)
    assert cb._format_state(np.zeros((5, Nq, 3))).shape == (5, Nq, 3)
    state, t = cb.time_integrate(Nt=10)
    cb.update_history(state, t)
    assert state.shape == (10, cb.N_dim, 3)
    np.testing.assert_allclose(cb.current_bias, cb.get_bias(state, mean=True)[0])  # current_bias is the ensemble mean
    np.testing.assert_allclose(state, np.broadcast_to(state[0], state.shape))  # constant in time
    assert cb.current_time == pytest.approx(0.1)
    assert len(cb.hist_t) == 11  # initial condition + 10 steps
    J = cb.state_derivative()
    assert J.shape == (Nq, Nq)
    np.testing.assert_allclose(J, 0.)
    # advance and update_state wrap the history
    cb = ConstantBias(innovation=np.ones(Nq), t=0.0, dt=0.1, N_ens=3)
    cb.advance(Nt=4)
    assert cb.current_time == pytest.approx(0.4) and cb.hist.shape[0] == 5
    cb.update_state(2.0 * np.ones((cb.N_dim, 3)))
    assert cb.hist.shape[0] == 5 and np.allclose(cb.current_bias, 2.0)


def test_innovation_reset_shapes():
    # one innovation per member
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
    # one innovation for every member of the bias ensemble, in any of its shapes
    for innovation in (np.array([0.3, 0.7]), np.array([[0.3], [0.7]]), np.array([[[0.3], [0.7]]])):
        cb = ConstantBias(innovation=np.ones(Nq), t=0.0, dt=0.1, N_ens=4)
        np.testing.assert_allclose(cb.update_state_from_innovation(innovation), np.repeat([[0.3], [0.7]], 4, axis=1))
    # one bias state and an innovation ensemble: the state takes the mean innovation
    members = np.array([[0.2, 0.4, 0.6], [1.0, 2.0, 3.0]])           # (Nq, m = 3)
    cb1 = ConstantBias(innovation=np.ones(Nq), t=0.0, dt=0.1)
    np.testing.assert_allclose(cb1.update_state_from_innovation(members), [[0.4], [2.0]])
    # with biased observations (N_dim = 2 Nq) the mean innovation goes to the innovation rows and the bias rows are
    # left at their forecast
    cb = ConstantBias(innovation=np.array([1., 2.]), t=0.0, dt=0.01, biased_observations=True)
    updated = cb.update_state_from_innovation(members)
    assert updated.shape == (2 * Nq, 1)
    np.testing.assert_allclose(updated[cb.observed_idx, 0], [0.4, 2.0])
    np.testing.assert_allclose(updated[cb.bias_idx, 0], cb.current_state[cb.bias_idx, 0])


def test_innovation_reset_feeds_the_history():
    cb = ConstantBias(innovation=np.zeros(Nq), t=0.0, dt=0.01)
    updated = cb.update_state_from_innovation(np.array([[0.3], [0.7]]) * np.ones((1, 5)))   # (Nq, 5) member innovations
    cb.update_history(updated, t=cb.current_time, modify_saved_states=True)
    np.testing.assert_allclose(cb.current_bias.ravel(), [0.3, 0.7])
    cb = ConstantBias(innovation=np.zeros(Nq), t=0.0, dt=0.1, N_ens=3)
    cb.advance(Nt=2)
    cb.update_state(cb.update_state_from_innovation(np.array([0.5, -0.5])))
    assert cb.hist.shape[0] == 3 and cb.current_time == pytest.approx(0.2)    # overwrites, does not append
    np.testing.assert_allclose(cb.current_bias.ravel(), [0.5, -0.5])
    cb.advance(Nt=1)                                                        # the forecast holds the reset bias
    np.testing.assert_allclose(cb.current_bias.ravel(), [0.5, -0.5])


def test_no_bias_is_always_zero():
    nb = NoBias(innovation=np.ones(Nq), t=0.0, dt=0.01)
    np.testing.assert_allclose(nb.current_bias, 0.)

    state, t = nb.time_integrate(Nt=5)
    nb.update_history(state, t)
    np.testing.assert_allclose(nb.current_bias, 0.)

    updated = nb.update_state_from_innovation(np.ones((Nq, 1)))
    nb.update_history(updated, t=nb.current_time, modify_saved_states=True)
    np.testing.assert_allclose(nb.current_bias, 0.)


def test_esn_bias_jacobian_matches_finite_differences():
    """J = d b_{k+1} / d q through one closed-loop step after the analysis sets i = d - q, for the network inputs
    [b; i] (observed_idx = every row), i alone (the default of ESN_bias, as in biasda) and i = b."""
    from types import SimpleNamespace

    from echostatenetwork import EchoStateNetwork
    from romda.bias_estimators.esn import ESN_bias

    for biased_observations, inputs in ((True, 'all'), (True, 'innovations'), (False, 'all')):
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
        np.testing.assert_allclose(J, J_fd, rtol=1e-5, atol=1e-8, err_msg=f'{biased_observations}, {inputs}')


def _record_resets(monkeypatch, rom, resets):
    """Record the states that `rom` resets its history to."""
    update_history = type(rom).update_history

    def record(self, psi=None, t=None, reset=False, **kwargs):
        if reset:
            resets.append(np.array(psi[0]))
        return update_history(self, psi=psi, t=t, reset=reset, **kwargs)

    monkeypatch.setattr(type(rom), 'update_history', record)


def test_sample_model_states_keeps_the_ranges_and_the_spread(monkeypatch):
    """The parameter rows of an ensemble follow the sorted est_alpha; a std_alpha dict in another order must
    still give each parameter its own range (the annular bias data drew epsilon from the c2beta range). Without
    std_phi the training members take the spread of the current ensemble: the standard deviation was passed as it
    is to `mean_vector_to_ensemble`, which reads a relative spread, so a component with mean 20 and standard
    deviation 5 was drawn in 20 (1 +- 5) instead of 20 +- 5."""
    from romda.bias_estimators.aux import sample_model_states
    from romda.models.physical import Lorenz63

    ranges = {'sigma': [9., 11.], 'rho': [27., 29.], 'beta': [2.5, 2.8]}     # not in sorted order
    rom = Lorenz63()
    rom.init_ensemble(m=3, std_alpha=ranges)
    assert list(rom.est_alpha) == sorted(ranges)
    sampled = []
    _record_resets(monkeypatch, rom, sampled)
    sample_model_states(rom, L=8, minimum_training_steps=10, std_alpha=ranges)
    rows = sampled[0][rom.Nphi:]
    for key, row in zip(rom.est_alpha, rows):
        low, high = ranges[key]
        assert np.all((row >= low) & (row <= high)), f'{key} drawn outside {ranges[key]}: {row}'

    monkeypatch.undo()
    rom = Lorenz63()
    rom.init_ensemble(m=4, std_alpha={'rho': [27., 29.]})
    state = np.array(rom.current_state)
    state[:rom.Nphi] = np.array([[8., 9., 10., 11.], [-9., -8., -7., -6.], [20., 25., 30., 35.]])
    rom.update_history(psi=state[None], t=0.0, reset=True)
    mean, std = state[:rom.Nphi].mean(axis=1), state[:rom.Nphi].std(axis=1)
    sampled = []
    _record_resets(monkeypatch, rom, sampled)
    sample_model_states(rom, L=200, minimum_training_steps=10, std_alpha={'rho': [27., 29.]})
    members = sampled[0][:rom.Nphi]
    assert np.all(np.abs(members - mean[:, None]) <= std[:, None] * (1 + 1e-12))
    np.testing.assert_allclose(members.std(axis=1), std / np.sqrt(3.), rtol=0.2)     # uniform in mean +- std


def test_esn_bias_network_reads_the_innovation_alone():
    """With biased observations the bias state is [b; i] and the analysis resets i. The network must read i
    alone: with its own bias forecast as an input too, it forecasts the bias from the bias and the data do not
    reach it (annular case: |d b / d i| = 0.002, so the regularization of the rBA-EnKF did nothing)."""
    from romda.bias_estimators import ESN_bias
    from romda.models.physical import Lorenz63
    from romda.observations import Observations

    rom = Lorenz63()
    rom.init_ensemble(m=3, std_alpha={'rho': [27., 29.]})
    truth = Observations(model=Lorenz63, t_start=1., t_stop=2., t_max=12., Nt_obs=10)
    bias = ESN_bias(rom=rom, reference_data=truth, biased_observations=True, L=2, N_units=12, N_wash=3,
                    upsample=1, t_train=4., t_val=1., t_test=1., hyperparameters_to_optimize=[],
                    plot_training=False, auto_save=False, force_retrain=True)
    rom.close()

    esn, Nq = bias.forecaster, rom.Nq
    assert bias.N_dim == 2 * Nq
    np.testing.assert_array_equal(esn.observed_idx, bias.observed_idx)        # the innovation rows
    assert esn.N_dim_in == Nq
    assert bias.state_derivative().shape == (Nq, Nq)

    def forecast(state):                                # one closed-loop step of the network
        return esn.step(esn.outputs_to_inputs(state[:bias.N_dim]), state[bias.N_dim:])[0]

    state = bias.current_state
    other_bias, other_innovation = state.copy(), state.copy()
    other_bias[bias.bias_idx] += 1.
    other_innovation[bias.observed_idx] += 1.
    np.testing.assert_array_equal(forecast(other_bias), forecast(state))      # the bias rows are not an input
    assert np.abs(forecast(other_innovation) - forecast(state)).max() > 0.


def test_bias_keeps_a_time_step_that_needs_many_decimals():
    """dt = 1/51200 s (the annular data) is 1.953125e-5: rounded to two digits past its leading one it became
    2e-5, and the bias clock ran 2.4 % ahead of the model's (0.512 s against 0.5 s at the first observation)."""
    dt = 1. / 51200
    cb = ConstantBias(innovation=np.zeros(Nq), t=0.0, dt=dt)
    assert cb.dt == dt
    cb.advance(Nt=25600)
    assert cb.current_time == pytest.approx(0.5, abs=1e-9)


# ------------------------------------------------- the options of the training data and of the network start

def _reference(n_steps=400, n_obs_start=300):
    """A reference record whose value is its time step: easy to locate a window."""
    from romda.observations import Observations

    t = np.arange(n_steps) * 0.01
    y = np.stack([np.arange(n_steps, dtype=float), -np.arange(n_steps, dtype=float)], axis=1)
    return Observations(model=None, y_true=y, y_raw=y + 0.5, t_true=t, t_start=t[n_obs_start], t_stop=t[-1], Nt_obs=10)


def test_training_record_takes_the_end_or_the_start_of_the_reference():
    """The bias network trains on the last steps of the record (v1.0) or on the first, before the assimilation
    (Nóvoa et al. 2024); a start that runs into the assimilation window is refused."""
    from romda.bias_estimators.aux import prepare_reference_data

    truth = _reference()
    y_raw, y_true = prepare_reference_data(truth, 50)
    np.testing.assert_array_equal(y_true[0][:, 0, 0], np.arange(350, 400))
    y_raw, y_true = prepare_reference_data(truth, 50, training_record='start')
    np.testing.assert_array_equal(y_true[0][:, 0, 0], np.arange(50))
    np.testing.assert_array_equal(y_raw[0][:, 0, 0], np.arange(50) + 0.5)
    with pytest.raises(ValueError, match='after the first observation'):
        prepare_reference_data(truth, 350, training_record='start')
    with pytest.raises(ValueError, match='training_record'):
        prepare_reference_data(truth, 50, training_record='middle')


def test_lag_criterion_distance_selects_the_lag_of_the_closest_model_record():
    """With 'distance' the best lag minimizes the squared distance to the data, and the worst maximizes it; with
    'correlation' a record of half the amplitude has the same best lag, with 'distance' it need not."""
    from romda.bias_estimators.aux import correlate_data

    n_min, n_corr, lag = 200, 40, 13
    phase = 2 * np.pi * np.arange(n_min + n_corr) / n_corr
    data = np.sin(phase[:n_min] + 2 * np.pi * lag / n_corr)[:, None, None]          # the model, 13 steps later
    model = np.stack([np.sin(phase), 3. * np.sin(phase)], axis=-1)[:, None, :]      # (Nt, Nq = 1, L = 2)

    out = correlate_data(model, data, 2, n_min, lag_criterion='distance')
    np.testing.assert_allclose(out[:, 0, 0], data[:, 0, 0], atol=1e-12)             # the best lag of the first record
    distance = [np.sum((model[k:k + n_corr, 0, 1] - data[:n_corr, 0, 0]) ** 2) for k in range(n_corr)]
    np.testing.assert_array_equal(out[:, 0, 2], model[int(np.argmin(distance)):int(np.argmin(distance)) + n_min, 0, 1])
    mid = int(np.mean([np.argmin(distance), np.argmax(distance)]))
    np.testing.assert_array_equal(out[:, 0, 3], model[mid:mid + n_min, 0, 1])       # the augmentation: the mid lag
    np.testing.assert_allclose(correlate_data(model, data, 1, n_min)[:, 0, 0], data[:, 0, 0], atol=1e-12)
    with pytest.raises(ValueError, match='lag_criterion'):
        correlate_data(model, data, 1, n_min, lag_criterion='phase')



def test_fixed_points_of_the_training_members(monkeypatch):
    """Members that decay to the fixed point in the transient: one is kept and the others are drawn again
    (default), or a fifth of the members plus one is kept and the others come from a normal distribution fitted
    to the remaining members, as the v1.0 code does."""
    from romda.bias_estimators.aux import sample_model_states
    from romda.models import RK4Integrator
    from romda.models.physical import Annular

    class FastAnnular(Annular):                 # a short transient: the members grow or decay within it
        t_transient = 0.02
        t_CR = 0.005

    for keep, n_kept in ((None, 1), (0.2, 5)):
        L, ranges = 20, {'nu': [-900., 900.]}
        rom = FastAnnular(dt=1. / 51200)
        rom.integrator = RK4Integrator(rom)
        rom.t_transient = 0.                        # the ensemble is drawn about the default initial condition
        rom.init_ensemble(m=4, std_phi=0.1, std_alpha=ranges)
        del rom.t_transient

        resets = []
        _record_resets(monkeypatch, rom, resets)
        sample_model_states(rom, L=L, minimum_training_steps=20, std_alpha=ranges, keep_fixed_points=keep)
        monkeypatch.undo()

        drawn, restart = resets[0], resets[2]        # the members drawn, and those that restart after the transient
        assert np.mean(drawn[rom.Nphi] < 0) >= 0.2    # a fifth or more decay, so the rule applies
        at_fixed_point = np.abs(restart[[0, 2]]).max(axis=0) < 1e-3
        assert restart.shape == (rom.Nphi + 1, L) and at_fixed_point.sum() == n_kept


def _small_esn_bias(**options):
    from romda.bias_estimators import ESN_bias
    from romda.models.physical import Lorenz63
    from romda.observations import Observations

    rom = Lorenz63()
    rom.init_ensemble(m=3, std_alpha={'rho': [27., 29.]})
    truth = Observations(model=Lorenz63, t_start=8., t_stop=9., t_max=12., Nt_obs=10)
    bias = ESN_bias(rom=rom, reference_data=truth, biased_observations=True, L=2, N_units=12, N_wash=3, upsample=2,
                    t_train=4., t_val=1., t_test=1., hyperparameters_to_optimize=[], plot_training=False,
                    auto_save=False, force_retrain=True, **options)
    rom.close()
    return bias, rom, truth


def test_esn_bias_options_of_the_training_data_reach_the_dataset():
    """`training_record`, `lag_criterion` and `keep_fixed_points` are options of ESN_bias, stored with the
    dataset; a dataset cached before they existed counts as built with the defaults."""
    from romda.bias_estimators.aux import load_bias_training_dataset
    from romda.utils import save_to_pickle_file

    bias, _, _ = _small_esn_bias(training_record='start', lag_criterion='distance', keep_fixed_points=0.2)
    assert bias.config['training_record'] == 'start' and bias.config['lag_criterion'] == 'distance'
    assert bias.config['keep_fixed_points'] == 0.2

    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        old = dict(upsample=2, biased_observations=True, L=2, augment_data_length=2, data=np.zeros((4, 700, 6)))
        save_to_pickle_file(f'{folder}/data', old)
        load = dict(filename=f'{folder}/data', minimum_training_steps=600, augment_data_length=2, L=2, expected_Ndim=6,
                    defaults=type(bias)._data_defaults)
        defaults = dict(upsample=2, biased_observations=True, L=2, augment_data_length=2, **type(bias)._data_defaults)
        assert load_bias_training_dataset(necessary_properties=defaults, **load) is not None
        assert load_bias_training_dataset(necessary_properties=bias.config, **load) is None


def test_esn_bias_washout_synchronizes_the_network_on_innovations():
    """The washout runs the network in open loop on the innovations from a reservoir at rest; every member then
    holds its prediction for the current time. A wrong input size is refused."""
    bias, rom, _ = _small_esn_bias(N_ens=3)
    esn, Nq = bias.forecaster, rom.Nq
    innovations = np.random.default_rng(0).normal(size=(4, Nq))

    u, r = np.zeros((2 * Nq, 1)), np.zeros((esn.N_units, 1))
    for innovation in innovations:
        u, r = esn.step(innovation[:, None], r)
    time = bias.current_time
    bias.washout(innovations)
    state = bias.current_state
    assert state.shape[1] == 3 and bias.current_time == time
    np.testing.assert_allclose(state[:2 * Nq], np.repeat(u, 3, axis=1))
    np.testing.assert_allclose(state[2 * Nq:], np.repeat(r, 3, axis=1))
    with pytest.raises(ValueError, match='washout'):
        bias.washout(np.zeros((4, 2 * Nq)))

