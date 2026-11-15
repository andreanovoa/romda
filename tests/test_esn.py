"""`ESN_model` (rnns/esn.py) and its builders (rnns/builders.py): the history reserve, the loops of
echostatenetwork (the washouts of initialize_from_val_data and regime_matched_states, closed_loop, replacement_loop),
the ensemble options that the washout-free ESN-DA paper employs (the read-out singular values in logarithms,
`Wout_log`; the read-out corrections along fixed directions, `Wout_dirs` with ``est_alpha=['Wdir']``; the per-member
reservoir hyperparameters, ``est_alpha=['sigma_in', 'rho']``; a parameter read out by W_out), the measured rows of
the observation operator, and the builders of the training data.

The networks are small and train without a hyperparameter search (a fraction of a second each).
"""

import types
from unittest import mock

import numpy as np
import pytest
from romda.models.physical import Lorenz63, Lorenz96

N_UNITS, M = 30, 4
DATA = Lorenz63(dt=0.01).time_integrate(Nt=3000)[0][None, :, :, 0]     # (1, Nt, 3)


def _l96(F=8.0, Nx=6, n=900):
    return Lorenz96(Nx=Nx, F=F, dt=0.01).time_integrate(Nt=n)[0][..., 0]


def test_a_large_ensemble_reserves_little_and_keeps_every_step():
    """The history of `ESN_model` reserves a few time steps, not dynamodels' default of 1000. The state of a network
    holds its reservoir: with the 1000 steps that dynamodels' `HistoryTracker` reserves at every reset of the history, a
    reservoir of 6000 units on 40 sites with 160 members reserved 7.8 GB at `init_ensemble`, before the first forecast
    step; the L96-40s sweeps of the washout-free paper held 40 to 44 GB per worker. The reserve is a keyword."""
    from romda.models.data_driven import ESN_model

    data = _l96(Nx=40)[None]                                                   # (1, Nt, 40)
    kw = dict(dt=0.01, plot_training=False, N_wash=10, t_train=5.0, t_val=2.0, t_test=0.0,
              hyperparameters_to_optimize=[], rho=0.5, sigma_in=0.1, tikh=1e-6, seed=0)
    esn = ESN_model(data=data, N_units=6000, **kw)
    assert esn.history._initial_capacity == 16
    ens = esn.copy()
    ens.init_ensemble(m=160)                                                  # the L96-40s ensemble of Sect. 4.4
    state_bytes = ens.hist[0].nbytes
    assert ens.hist.shape[1:] == (esn.N_dim + esn.N_units, 160) or ens.hist.shape[-1] == 160
    assert ens.history.capacity == 16 and ens.history.capacity * state_bytes < 0.2e9    # 1000 steps: 7.8 GB
    assert ens.copy().history.capacity == 16                                   # copies keep the reserve
    states = []
    for _ in range(3):                                                         # 3 x 10 steps: grows past 16
        psi, t = ens.time_step(Nt=10)
        ens.update_history(psi[1:], t=t[1:])
        states.append(psi[1:])
    assert ens.hist.shape[0] == 31 and np.array_equal(ens.hist[1:], np.concatenate(states))
    assert ens.history.capacity < 4 * ens.hist.shape[0]                        # doubling: bounded overallocation

    ens = ESN_model(data=data[..., :3], N_units=20, history_capacity=4, **kw).copy()
    ens.init_ensemble(m=8)
    assert ens.history.capacity == 4


# ── the loops of echostatenetwork on a parametric network (F in the reservoir)

@pytest.fixture(scope='module')
def parametric_F():
    from romda.models.data_driven import ESN_model
    data = np.stack([_l96(F) for F in (6.0, 7.0, 8.0)])
    return ESN_model(dt=0.01, data=data, plot_training=False, N_units=40, N_wash=10, t_train=5.0, t_val=2.0,
                     t_test=0.0, hyperparameters_to_optimize=[], rho=0.9, sigma_in=0.1, tikh=1e-8, seed=0,
                     input_parameters=np.array([[6.0, 7.0, 8.0]]), param_names=('F',), param_values=(7.0,))


def _manual_washout(esn, U, F):
    """The washout that the methods replaced: r = 0, one step per datum at the parameter F."""
    u, r = np.zeros((esn.N_dim, 1)), np.zeros((esn.N_r, 1))
    for u_in in U:
        u, r = esn.step(np.concatenate([u_in[esn.observed_idx], [F]])[:, None], r)
    return u[:, 0], r[:, 0]


def test_one_segment_needs_no_swap_of_the_validation_data(parametric_F):
    esn = parametric_F
    regimes, own = np.array([[6.0, 7.0, 8.0]]), esn.input_parameters
    esn.input_parameters = np.full((1, 4), 99.0)               # what a closed loop leaves behind
    psi = esn.initialize_from_val_data(N_ens=4, segments=[1], input_parameters=regimes)
    t_ids = np.random.default_rng(esn.seed).choice(esn.validation_data.shape[1] - esn.N_wash, size=4, replace=False)
    for j, t in enumerate(t_ids):                              # every member on segment 1, at F = 7
        u, r = _manual_washout(esn, esn.validation_data[1, t:t + esn.N_wash], 7.0)
        np.testing.assert_array_equal(psi[:esn.N_dim, j], u)
        np.testing.assert_array_equal(psi[esn.N_dim:esn.N_dim + esn.N_r, j], r)
    assert np.all(esn.input_parameters == 99.0)
    esn.input_parameters = own


def test_regime_matched_states_and_the_old_name(parametric_F):
    esn = parametric_F
    regimes = np.array([[6.0, 7.0, 8.0]])
    psi, F = esn.regime_matched_states(5, regimes, seed=2)
    np.testing.assert_array_equal(F, regimes[:, [0, 1, 2, 0, 1]])
    np.testing.assert_array_equal(esn._regime_matched_init(5, regimes, seed=2)[0], psi)
    rng = np.random.default_rng(2)
    for i in range(5):
        k0 = rng.integers(0, esn.validation_data.shape[1] - esn.N_wash)
        u, _ = _manual_washout(esn, esn.validation_data[i % 3, k0:k0 + esn.N_wash], F[0, i])
        np.testing.assert_array_equal(psi[:esn.N_dim, i], u)


def test_closed_loop_starts_from_the_last_washout_output(parametric_F):
    esn = parametric_F
    data = esn.validation_data[2]
    pred, target = esn.closed_loop(data, 15, input_parameters=[8.0])
    u, r = _manual_washout(esn, data[:esn.N_wash], 8.0)
    u, r = u[:, None], r[:, None]
    ref = [u[:, 0]]
    for _ in range(14):
        u, r = esn.step(np.vstack([u, [[8.0]]]), r)
        ref.append(u[:, 0])
    np.testing.assert_array_equal(pred, np.array(ref))
    np.testing.assert_array_equal(target, data[esn.N_wash:esn.N_wash + 15])
    assert esn.input_parameters.shape == (1, 3)


def test_the_data_replace_the_observed_components_at_the_observation_steps():
    """`ESN_model.replacement_loop`: the closed loop with the observed components replaced by the data at the
    observation times (direct replacement in the washout-free ESN-DA paper). One network returns a two-dimensional
    output, and observation times that collide are refused."""
    from romda.models.data_driven import ESN_model
    esn = ESN_model(dt=0.01, data=_l96(Nx=10)[None], plot_training=False, N_units=50, N_wash=10, t_train=5.0,
                    t_val=2.0, t_test=0.0, hyperparameters_to_optimize=[], rho=0.5, sigma_in=0.1, tikh=1e-6, seed=0)
    idx, steps, n_extra, m = [0, 2], np.array([0, 3, 6, 9]), 5, 3
    t_obs = 1.0 + steps * esn.dt_ESN
    y_obs = np.random.default_rng(0).standard_normal((len(steps), len(idx)))
    t, y = esn.replacement_loop(t_obs, y_obs, idx, n_extra=n_extra, m=m)

    u, r = esn.unbuild_psi(esn.initialize_from_val_data(N_ens=m))     # the same start: washouts on the same windows
    ref = []
    for i in range(steps[-1] + 1 + n_extra):
        ref.append(u.copy())
        if i in steps:
            u[idx, :] = y_obs[list(steps).index(i)][:, None]
        u, r = esn._single_step(u, r)
    assert y.shape == (len(ref), esn.N_dim, m)
    np.testing.assert_array_equal(y, np.array(ref))                 # the output before each replacement
    np.testing.assert_allclose(t, 1.0 + np.arange(len(ref)) * esn.dt_ESN)

    t, y = esn.replacement_loop(np.array([0.0, 0.1]), np.zeros((2, esn.N_dim)))   # every component, by default
    assert y.shape == (len(t), esn.N_dim)
    with pytest.raises(AssertionError, match='collide'):
        esn.replacement_loop(np.array([0.0, 0.1 * esn.dt_ESN]), np.zeros((2, esn.N_dim)))


# ── the ensemble options of the washout-free ESN-DA paper

def _esn(**kw):
    from romda.models.data_driven import ESN_model
    return ESN_model(dt=0.01, data=DATA, plot_training=False, N_units=N_UNITS, N_wash=10, t_train=20.0, t_val=5.0,
                     t_test=0.0, hyperparameters_to_optimize=[], rho=0.9, sigma_in=0.1, tikh=1e-8, seed=0, **kw)


def _forecast(esn, Nt=20):
    psi, _ = esn.time_step(Nt=Nt)
    return psi[:, :esn.N_dim]                  # (Nt + 1, N_dim, m): the outputs u


def _set_alpha(esn, name, values):
    """Overwrite the row of the estimated parameter `name` in the current augmented state."""
    psi = esn.current_state.copy()
    psi[-len(esn.est_alpha) + esn.est_alpha.index(name)] = values
    esn.update_history(psi[None], t=esn.current_time, reset=True)


def test_logarithmic_singular_values_read_out_the_same_network():
    # Wout_log: the augmented state carries log(s_i). With no spread the members read out with the trained
    # singular values, as with Wout_log = False; with a spread, every member's s_i is positive and within the
    # log-uniform band s_i exp([-std, std]).
    log, lin = _esn(Wout_log=True), _esn(Wout_svd=True)       # est_alpha=['Wout'] switches Wout_svd on too
    for esn in (log, lin):
        esn.init_ensemble(m=M, est_alpha=['Wout'], std_alpha=0.0)
    s = log.Wout_Sigma0
    assert np.allclose(np.asarray(log.current_state)[-3:], np.log(s)[:, None])
    assert np.allclose(_forecast(log), _forecast(lin))

    band = _esn(Wout_log=True)
    band.init_ensemble(m=M, est_alpha=['Wout'], std_alpha=0.5)
    vals = np.exp(np.asarray(band.current_state)[-3:].reshape(3, M))
    assert np.all(vals > 0) and np.all(vals >= s[:, None] * np.exp(-0.5) - 1e-12) \
        and np.all(vals <= s[:, None] * np.exp(0.5) + 1e-12)


def test_readout_directions_add_their_coordinates_times_the_projected_reservoir():
    # est_alpha=['Wdir']: member j reads out with Wout + Wout_dirs C_j. With zero coordinates the forecast is
    # the plain network's; with coordinates C, the first output step differs by C (Wout_dirs^T [r; b_out]),
    # because the reservoir update of that step does not depend on the read-out.
    k = 2
    dirs = np.random.default_rng(1).standard_normal((N_UNITS + 1, k))
    wdir, plain = _esn(Wout_dirs=dirs, Wout_dir_std=np.ones((k, 3))), _esn()
    wdir.init_ensemble(m=M, est_alpha=['Wdir'], std_alpha=0.0)
    plain.init_ensemble(m=M)
    assert wdir.est_alpha == [f'wdir_{i}_{q}' for i in range(k) for q in range(3)]
    assert np.allclose(_forecast(wdir), _forecast(plain))

    wdir, plain = _esn(Wout_dirs=dirs, Wout_dir_std=np.ones((k, 3))), _esn()
    wdir.init_ensemble(m=M, est_alpha=['Wdir'], std_alpha=0.0)
    plain.init_ensemble(m=M)
    C = np.random.default_rng(2).standard_normal((k, 3, M))
    for i in range(k):
        for q in range(3):
            _set_alpha(wdir, f'wdir_{i}_{q}', C[i, q])
    psi_w, _ = wdir.time_step(Nt=1)
    psi_p, _ = plain.time_step(Nt=1)
    r1 = psi_p[1, 3:3 + N_UNITS]                                         # (N_units, m), the same in both
    assert np.allclose(psi_w[1, 3:3 + N_UNITS], r1)
    proj = dirs.T @ np.vstack([r1, plain.bias_out * np.ones((1, M))])    # (k, m)
    assert np.allclose(psi_w[1, :3] - psi_p[1, :3], np.einsum('kqm,km->qm', C, proj))


def test_per_member_reservoir_hyperparameters():
    # est_alpha=['sigma_in', 'rho']: each member steps with its own value; the nominal attributes are restored
    # after the forecast. Member j forecasts as an ensemble in which every member carries member j's values.
    for names in (['sigma_in'], ['rho'], ['sigma_in', 'rho']):
        esn = _esn()
        esn.init_ensemble(m=M, est_alpha=names, std_alpha=0.05)
        nominal = {a: getattr(esn, a) for a in names}
        values = {a: np.asarray(esn.current_state)[-len(names) + names.index(a)].ravel().copy() for a in names}
        assert all(np.ptp(v) > 0 for v in values.values())
        u = _forecast(esn)
        assert all(getattr(esn, a) == v for a, v in nominal.items())

        j = 2
        same = _esn()
        same.init_ensemble(m=M, est_alpha=names, std_alpha=0.05)
        for a in names:
            _set_alpha(same, a, np.full(M, values[a][j]))
        assert np.allclose(_forecast(same)[:, :, j], u[:, :, j]), names
        assert not np.allclose(u[:, :, 0], u[:, :, j]), names


def test_explicit_est_alpha_names_its_rows_with_a_dict_of_ranges():
    # The parameter rows follow est_alpha. With a std_alpha dict, `Model.init_ensemble` re-derives est_alpha as
    # the sorted keys of the dict, which named the rows of an unsorted est_alpha in another order: the
    # reservoir then stepped with the input scaling as its spectral radius.
    ranges = {'sigma_in': [0.09, 0.11], 'rho': [0.85, 0.95], 'unused': [0., 1.]}
    esn = _esn()
    esn.init_ensemble(m=M, est_alpha=['sigma_in', 'rho'], std_alpha=ranges)
    assert list(esn.est_alpha) == ['sigma_in', 'rho']
    for name, row in zip(esn.est_alpha, np.asarray(esn.current_state)[-2:]):
        low, high = ranges[name]
        assert np.all((row >= low) & (row <= high)), f'{name} drawn outside {ranges[name]}: {row}'


def _parametric(data, values, **kw):
    """A small parametric network (one record per value of the parameter `a`) with `a` in the readout only."""
    from romda.models.data_driven import ESN_model
    return ESN_model(dt=0.01, data=data, plot_training=False, N_units=N_UNITS, N_wash=10, t_train=10.0, t_val=2.0,
                     t_test=0.0, hyperparameters_to_optimize=[], rho=0.9, sigma_in=0.1, tikh=1e-8, seed=0,
                     input_parameters=np.array([values]), param_names=('a',), param_values=(float(np.mean(values)),),
                     param_in_reservoir=False, readout_input='params', **kw)


def test_readout_parameter_reads_each_members_own_value():
    # readout_input='params': W_out reads [r; a; 1]. reservoir_to_physical(r) without u reads out with the
    # parameter of every member (the estimated one in the ensemble), as the closed-loop step does; with patches the
    # reservoir state has N_r = N_patches * N_units rows and reset_forecaster sizes it so.
    for layout in ({}, dict(patch_size=4, halo=1, periodic=True)):
        Nx = 8 if layout else 3
        data = (np.stack([_l96(F, Nx, n=1300) for F in (6.0, 8.0)]) if layout
                else np.stack([DATA[0, :1300], DATA[0, 1300:2600]]))
        esn = _parametric(data, [6.0, 8.0], **layout)
        assert esn.N_r == (Nx // 4 if layout else 1) * N_UNITS
        esn.init_ensemble(m=M, est_alpha=['a'], regimes=np.array([[6.0, 8.0]]), seed=0)
        psi, _ = esn.time_step(Nt=3)
        u, r = psi[-1, :esn.N_dim], psi[-1, esn.N_dim:esn.N_dim + esn.N_r]
        np.testing.assert_allclose(esn.reservoir_to_physical(r), u, atol=1e-12)
        a = np.asarray(esn.current_state)[-1]
        shifted = esn.reservoir_to_physical(r, u=np.vstack([np.zeros((Nx, M)), a + 1.0]))
        assert not np.allclose(shifted, u)                                   # the readout reads a
        esn.reset_forecaster(u0=u)
        assert np.asarray(esn.current_state).shape[0] == esn.N_dim + esn.N_r + 1
        assert len(esn.forecaster_state_labels) == esn.N_r


def test_readout_corrections_are_rejected_with_a_readout_parameter():
    esn = _parametric(np.stack([DATA[0, :1300], DATA[0, 1300:2600]]), [6.0, 8.0])
    for est in (['Wout'], ['Wdir']):
        if est == ['Wdir']:
            esn.Wout_dirs, esn.Wout_dir_std = np.ones((N_UNITS + 2, 1)), np.ones((1, 3))
        with pytest.raises(ValueError, match='not implemented with readout_input'):
            esn.init_ensemble(m=M, est_alpha=est, std_alpha=0.1)


# ── the measured rows of the observation operator, the readout and the builders

def _restricted_M(observed_idx, measured, Na=0, N_dim=3, N_units=5):
    """The observation operator after `ESN_model.init_ensemble(measured=...)` —
    the estimator deployment's M-row selection — on an untrained stub network."""
    from romda.models import Model
    from romda.models.data_driven import ESN_model

    esn = ESN_model.__new__(ESN_model)                 # no training: attributes only
    esn.observed_idx = list(observed_idx)
    Nq = len(esn.observed_idx)
    # the repo's M layout: observables appended at the bottom of [phi; alpha; y]
    esn._M = np.hstack([np.zeros((Nq, N_dim + N_units + Na)), np.eye(Nq)])

    with mock.patch.object(Model, 'init_ensemble', return_value=None):
        ESN_model.init_ensemble(esn, m=2, measured=measured,
                                ensemble_psi0=np.zeros((N_dim + N_units, 2)))
    return esn.M


def test_measured_restriction_selects_M_rows():
    # full-input network, x and z measured: two rows of the trailing identity
    M = _restricted_M([0, 1, 2], [0, 2])
    assert M.shape == (2, 3 + 5 + 3)
    assert np.array_equal(M[:, -3:], np.array([[1., 0., 0.], [0., 0., 1.]]))
    assert not M[:, :-3].any()                                 # nothing on state/params

    # with estimated parameters the zero block widens, and the identity stays put
    M_a = _restricted_M([0, 1, 2], [0, 2], Na=4)
    assert M_a.shape == (2, 3 + 5 + 4 + 3)
    assert np.array_equal(M_a[:, -3:], M[:, -3:])

    # partial-input network: `measured` still indexes the physical state, and the row
    # is looked up in observed_idx rather than used directly
    M_p = _restricted_M([0, 2], [2])
    assert M_p.shape == (1, 3 + 5 + 2)
    assert np.array_equal(M_p[:, -2:], np.array([[0., 1.]]))

    # measuring exactly the network's inputs leaves the operator untouched
    M_id = _restricted_M([0, 2], [0, 2])
    assert M_id.shape == (2, 3 + 5 + 2)
    assert np.array_equal(M_id[:, -2:], np.eye(2))


def test_svd_readout_reaches_the_closed_loop():
    """`EchoStateNetwork.step` reads out through `_readout`; with `Wout_svd` it must be the
    per-member SVD read-out, or est_alpha=['Wout'] has no effect on the forecast."""
    from romda.models.data_driven import ESN_model
    esn = ESN_model.__new__(ESN_model)
    esn.Wout_svd, esn.Wout_U = True, np.eye(5, 2)          # the SVD factors exist once Wout is trained
    with mock.patch.object(ESN_model, 'reservoir_to_physical', return_value='svd') as svd:
        assert esn._readout(np.zeros((5, 2)), None) == 'svd'
    svd.assert_called_once()


def test_twin_data_is_sampled_at_the_network_step():
    from romda.models.data_driven.rnns import twin_data

    rng = np.random.default_rng(0)
    clean = rng.normal(size=(400, 3)) * [1., 5., 10.]
    noisy = clean + 0.05 * clean.std(axis=0) * rng.normal(size=clean.shape)
    data = dict(clean_data=clean, noisy_data=noisy, t=0.01 * np.arange(400), t_lyap=0.1, i_test=100)

    def closed_loop(held_out, n_steps):         # a stub network: its closed loop counts the steps
        assert len(held_out) == 150             # the held-out record, at the network step
        return np.arange(n_steps)[:, None] * [1., 5., 10.], None

    esn = types.SimpleNamespace(upsample=2, dt_ESN=0.02, closed_loop=closed_loop)
    twin = twin_data(esn, data, n_spin_lyap=5)

    # the 25 network steps of spin-up are dropped, and the record spans the held-out time
    assert twin['clean_data'][0, 0] == 25 and twin['clean_data'].shape == (150, 3)
    assert twin['i_test'] == 0 and twin['dt'] == 0.02 and np.allclose(np.diff(twin['t']), 0.02)
    assert np.isclose(twin['t'][-1], data['t'][-1] - data['t'][100], atol=0.01)
    # the noise level of the dataset, relative to each component, with a fixed seed
    level = (twin['noisy_data'] - twin['clean_data']).std(axis=0) / twin['clean_data'].std(axis=0)
    assert np.allclose(level, 0.05, rtol=0.25)
    assert np.array_equal(twin['noisy_data'], twin_data(esn, data)['noisy_data'])


def test_parametric_segments_carry_the_noise_of_the_dataset():
    """The parametric network trains on noisy records, as a single network: each segment carries Gaussian noise of
    the configured level times the standard deviation of each of its components, the same noise for the same seed,
    and the record digest of the store key changes with the noise."""
    from romda.models.data_driven.rnns import builders as build
    from romda.models.physical import Lorenz63

    scale = np.array([1., 5., 10.])

    def trajectory(model_class, seed=0, n_steps=1000, **kwargs):    # a clean segment of its own per seed
        return np.random.default_rng(100 + seed).normal(size=(n_steps, 3)) * scale

    keys = []

    def store(data=None, training_data_filename=None, **kwargs):     # the store: records the key, trains nothing
        keys.append(training_data_filename)
        return 'esn'

    train = dict(param='rho', est_name='mu', label=None, values=[28.0, 32.0, 36.0], dt=0.02,
                 t_train_lyap=20, t_val_lyap=5, t_test_lyap=5)
    with mock.patch.multiple(build, trajectory=trajectory, auto_load_or_create=store):
        _, noisy, regimes = build.parametric_esn(Lorenz63, train, dict(N_units=10), 0.05)
        _, again, _ = build.parametric_esn(Lorenz63, train, dict(N_units=10), 0.05)
        _, clean, _ = build.parametric_esn(Lorenz63, train, dict(N_units=10), 0.0)

    assert noisy.shape == clean.shape and np.array_equal(regimes, [[28.0, 32.0, 36.0]])
    assert np.array_equal(clean[1], trajectory(Lorenz63, seed=1, n_steps=clean.shape[1]))   # noise 0: the clean segments
    level = (noisy - clean).std(axis=1) / clean.std(axis=1)          # (segment, component)
    assert np.allclose(level, 0.05, rtol=0.1)
    assert np.array_equal(noisy, again)                             # same seed, same noise
    assert not np.allclose(noisy[0] - clean[0], noisy[1] - clean[1])   # the segments do not share one draw
    assert keys[0] == keys[1] != keys[2]                            # the digest of the noisy record keys the store


def test_network_summary_of_a_network_loaded_from_the_store():
    """`train` of run_parallel_localization.py and run_parametric.py prints the summary of the network: a
    network loaded from the store has no training summary (echostatenetwork raises), so its hyperparameters stand in."""
    from romda.models.data_driven.rnns import network_summary

    loaded = types.SimpleNamespace(rho=np.array(0.3), sigma_in=np.array([0.919]), tikh=np.array(1e-10),
                                   sigma_param=None,
                                   training_summary=mock.Mock(side_effect=RuntimeError('no training summary yet')))
    assert network_summary(loaded) == 'loaded from the store | rho=0.3, sigma_in=0.919, tikh=1e-10'
    loaded.sigma_param = np.array([0.0141])
    assert network_summary(loaded).endswith(', sigma_param=0.0141')
    trained = types.SimpleNamespace(training_summary=lambda: 'trained: rho=0.3')
    assert network_summary(trained) == 'trained: rho=0.3'


def test_log_nmse_is_the_error_over_the_input_range():
    from types import SimpleNamespace

    from romda.models.data_driven import ESN_model
    from romda.models.data_driven.rnns.esn import log_nMSE

    case = SimpleNamespace(norm=np.array([10., 10.]))
    truth, prediction = np.zeros((5, 2)), np.ones((5, 2))
    assert log_nMSE(case, truth, prediction) == -2.                         # log10(1 / 100)
    assert log_nMSE(case, truth, np.full((5, 2), np.inf)) == 10.            # a diverged forecast
    with pytest.raises(ValueError, match='validation_metric'):
        ESN_model.validation_metric.fset(SimpleNamespace(), 'unknown')
