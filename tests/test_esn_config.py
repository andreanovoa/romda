"""The network store (`romda.models.data_driven.rnns.esn_config`): its hash must tell networks and hyperparameter
searches apart, without moving any network cached before the search settings were hashed, and key the training data
by their name, so that a path finds the network from every clone, and by the digest of their content."""

from pathlib import Path
from unittest import mock

import numpy as np
from romda.models.data_driven.rnns import (
    ESN_model,
    ESNConfig,
    auto_load_or_create,
    find_matching_config,
    rekey_esn_store,
)

CACHED = dict(N_dim=3, N_folds=4, N_func_evals=20, N_grid=4, N_initial_rand=0, N_split=5,
              N_units=300, N_wash=20, Win_type='sparse', Wout_svd=False, bias_in=[0.1],
              bias_out=[1.0], connect=3, dt=0.02, noise=0.01, norm_method='range',
              observed_idx=[0, 2], rho_range=[0.2, 1.0], seed=0, sigma_in_range=[-2, 1],
              t_test=5.5, t_train=110.0, t_val=5.5, tikh_range=[1e-6, 1e-9, 1e-12],
              training_data_filename='Lorenz63_default_Nlyap300_noise0.02_seed0',
              update_reservoir=True, update_state=True, upsample=2)


CACHED_HASH = '1afa29689edcdc37'


def test_default_search_keeps_the_cached_hash():
    assert ESNConfig(**CACHED).to_hash() == CACHED_HASH
    # the defaults spelled out, in the container types a caller or a YAML file uses
    explicit = dict(CACHED, hyperparameters_to_optimize=['rho', 'sigma_in', 'tikh'],
                    bias_in_range=[0, 2], regularize_bias_out=True)
    assert ESNConfig(**explicit).to_hash() == CACHED_HASH
    # the Bayesian search and parametric-normalization options at their
    # echostatenetwork 0.2.3 defaults
    explicit.update(stability_steps=0, bho_noise=1e-10, grid_midpoints=False,
                    optimize_parameter_normalization=False, param_shift_range=[-1, 1],
                    param_norm_range=[0.1, 10])
    assert ESNConfig(**explicit).to_hash() == CACHED_HASH
    assert ESNConfig.from_init_params(**explicit).to_hash() == CACHED_HASH


def test_the_settings_of_a_network_key_the_hash():
    """Every setting of the network, of its search and of its training data enters the hash; the defaults leave it
    at the hash of a network cached before they existed."""
    def hashes(base, changes):
        return [ESNConfig(**dict(base, **change)).to_hash() for change in ({},) + changes]

    def apart(base, *changes):
        found = hashes(base, changes)
        assert len(set(found)) == len(found), changes

    # the search settings
    apart(CACHED, dict(hyperparameters_to_optimize=('rho', 'sigma_in', 'tikh', 'bias_in')),
          dict(bias_in_range=(0.0, 1.0)), dict(regularize_bias_out=False), dict(stability_steps=1350),
          dict(bho_noise='gaussian'), dict(grid_midpoints=True), dict(optimize_parameter_normalization=True),
          dict(param_shift_range=(3.5, 8.0)), dict(param_norm_range=(1.0, 20.0)))
    # `from_init_params` is how `auto_load_or_create` keys a query: the options must reach the hash
    assert ESNConfig.from_init_params(**CACHED).to_hash() == CACHED_HASH
    assert ESNConfig.from_init_params(**dict(CACHED, stability_steps=1350)).to_hash() != CACHED_HASH
    # the input subset: two subsets never share one entry, whatever the container of the indices
    apart(CACHED, dict(observed_idx=[0, 1, 2]), dict(observed_idx=[0]))
    assert ESNConfig(**dict(CACHED, observed_idx=np.array([0, 2]))).to_hash() == CACHED_HASH
    # input centring, the validation metric and a parameter in the readout, at their defaults or not
    assert ESNConfig(**dict(CACHED, centre_input=True, validation_metric=None)).to_hash() == CACHED_HASH
    assert ESNConfig(**dict(CACHED, param_in_reservoir=True, readout_input=False)).to_hash() == CACHED_HASH
    apart(CACHED, dict(centre_input=False), dict(validation_metric='log_nMSE'))
    # the parallel layout at its defaults
    assert ESNConfig(**dict(CACHED, patch_size=None, halo=0, periodic=True, shared=True)).to_hash() == CACHED_HASH

    # Without a search, rho, sigma_in and tikh are settings: networks that differ in them are different networks.
    # With the default search, their values are training results and do not enter the hash.
    fixed = dict(CACHED, hyperparameters_to_optimize=[])
    found = {ESNConfig(**dict(fixed, rho=r, sigma_in=0.5, tikh=t)).to_hash() for r in (0.4, 0.8) for t in (1e-6, 1e-10)}
    assert len(found) == 4
    assert ESNConfig(**dict(CACHED, rho=0.4, sigma_in=0.5, tikh=1e-6)).to_hash() == CACHED_HASH
    # rho searched, sigma_in fixed: only sigma_in keys the network
    partial = dict(CACHED, hyperparameters_to_optimize=['rho'])
    assert hashes(dict(partial, sigma_in=0.5), (dict(rho=0.4), dict(rho=0.9)))[1:] == \
        [ESNConfig(**dict(partial, sigma_in=0.5)).to_hash()] * 2
    apart(partial, dict(sigma_in=0.5), dict(sigma_in=0.7))
    assert ESNConfig.from_init_params(**dict(fixed, tikh=1e-6)).to_hash() != \
        ESNConfig.from_init_params(**dict(fixed, tikh=1e-8)).to_hash()


def test_the_training_data_are_keyed_by_their_name_and_content():
    """A path to the training data hashes as its name: the experiment driver and the tutorials pass an absolute
    path, which differs between clones and machines. Two records with one file name (e.g. the same options
    integrated by two environments) may never share a network: `record_key` tags the name with a digest of the
    record, round-off differences included."""
    from romda.utils import record_key

    name = CACHED['training_data_filename']
    for filename in (f'/storage0/user/romda/data/Lorenz63/{name}',
                     f'/Users/user/codes/worktrees/romda-x/results/tutorials/Lorenz63//{name}',
                     f'data/Lorenz63/{name}', Path('data') / 'Lorenz63' / name):
        assert ESNConfig(**dict(CACHED, training_data_filename=filename)).to_hash() == CACHED_HASH, filename
    # the name still tells the datasets apart
    assert ESNConfig(**dict(CACHED, training_data_filename=f'/data/{name}_seed1')).to_hash() != CACHED_HASH
    assert ESNConfig(**dict(CACHED, training_data_filename=None)).to_hash() != CACHED_HASH

    record = np.random.default_rng(0).normal(size=(200, 3))
    perturbed = record.copy()
    perturbed[-1, -1] += 1e-15 * abs(perturbed[-1, -1])
    assert (ESNConfig(**dict(CACHED, training_data_filename=record_key(name, record))).to_hash()
            != ESNConfig(**dict(CACHED, training_data_filename=record_key(name, perturbed))).to_hash())


def test_esn_defaults_match_the_store_defaults():
    """The defaults dropped from the hash are the class defaults of echostatenetwork:
    a network trained without the option is the network trained with its default."""
    from echostatenetwork import EchoStateNetwork
    for key in ('stability_steps', 'stability_starts', 'stability_penalty', 'bho_noise', 'grid_midpoints', 'bias_in_range',
                'regularize_bias_out', 'optimize_parameter_normalization', 'param_shift_range',
                'param_norm_range'):
        assert getattr(EchoStateNetwork, key) == getattr(ESNConfig, key), key


def _small_network(**kwargs):
    """The settings of a network that trains in well under a second."""
    t = np.arange(400) * 0.1
    data = np.stack([np.sin(t), np.cos(t)], axis=-1)[np.newaxis]
    return dict(data=data, dt=0.1, t_train=30., t_val=5., t_test=0., N_units=20,
                N_wash=5, N_folds=2, N_grid=2, N_func_evals=8, perform_test=False,
                plot_training=False, **kwargs)


def test_search_options_round_trip_through_the_store(tmp_path):
    """The searched bias_in and the options of the search reload as they were, from the same hash (load ->
    to_esn_model), not trained again. `centre_input=False` scales the input by its range without its mean, and
    `validation_metric` names the metric of the search."""
    kwargs = _small_network(config_dir=tmp_path / 'bias_in',
                            hyperparameters_to_optimize=['rho', 'sigma_in', 'tikh', 'bias_in'],
                            bias_in_range=(0.0, 1.0), regularize_bias_out=False,
                            stability_steps=20, grid_midpoints=True)
    trained = auto_load_or_create(**kwargs)
    (saved,) = (tmp_path / 'bias_in').iterdir()

    config = ESNConfig.load(saved)
    assert config.hyperparameters_to_optimize == kwargs['hyperparameters_to_optimize']
    assert tuple(config.bias_in_range) == kwargs['bias_in_range']
    assert config.regularize_bias_out is False
    assert config.stability_steps == 20 and config.grid_midpoints is True

    with mock.patch.object(ESN_model, 'train', side_effect=AssertionError('retrained')):
        loaded = auto_load_or_create(**kwargs)
    np.testing.assert_array_equal(loaded.Wout, trained.Wout)
    np.testing.assert_array_equal(loaded.bias_in, trained.bias_in)    # the searched value
    assert loaded.hyperparameters_to_optimize == kwargs['hyperparameters_to_optimize']
    assert loaded.regularize_bias_out is False

    t = np.arange(400) * 0.1
    data = np.column_stack([3. + np.sin(t), -2. + np.cos(t)])[None]          # (1, Nt, 2), with non-zero means
    kw = dict(dt=0.1, data=data, training_data_filename='offset_wave', N_units=20, N_wash=5, upsample=1,
              t_train=30., t_val=5., t_test=5., N_folds=2, N_func_evals=2, N_grid=1, tikh_range=[1e-8],
              plot_training=False, seed=0)

    centred = ESN_model(**kw)
    np.testing.assert_allclose(centred.shift, [3., -2.], atol=0.2)
    assert centred.validation_metric is None

    plain = auto_load_or_create(config_dir=tmp_path, centre_input=False, validation_metric='log_nMSE', **kw)
    np.testing.assert_array_equal(plain.shift, [0., 0.])
    np.testing.assert_allclose(plain.norm, centred.norm)                     # the range is that of the data
    assert plain.validation_metric.__name__ == 'log_nMSE' and plain.validation_metric_name == 'log_nMSE'
    assert ESNConfig.from_esn_model(plain).validation_metric == 'log_nMSE'

    with mock.patch.object(ESN_model, 'train', side_effect=AssertionError('retrained instead of loading')):
        loaded = auto_load_or_create(config_dir=tmp_path, centre_input=False, validation_metric='log_nMSE', **kw)
    np.testing.assert_array_equal(loaded.shift, [0., 0.])
    assert loaded.centre_input is False and loaded.validation_metric_name == 'log_nMSE'
    np.testing.assert_allclose(loaded.Wout, plain.Wout)


def test_layouts_round_trip_through_the_store(tmp_path):
    """The store keeps the parallel layout of echostatenetwork, and a parametric network that reads its parameter in
    the readout only (W_out on [r; p; 1], no parameter column in W_in), which is a different network from the one that
    reads it in the reservoir: each reloads as itself, with the same matrices and the same closed loop."""
    t = np.arange(400) * 0.1
    data = np.stack([np.sin(t + k) for k in range(8)], axis=-1)[np.newaxis]   # a travelling wave on 8 sites
    kwargs = dict(data=data, dt=0.1, t_train=30., t_val=5., t_test=0., N_units=20, N_wash=5,
                  patch_size=2, halo=1, periodic=True, hyperparameters_to_optimize=[], tikh=1e-6,
                  perform_test=False, plot_training=False, config_dir=tmp_path / 'parallel')
    trained = auto_load_or_create(**kwargs)
    with mock.patch.object(ESN_model, 'train', side_effect=AssertionError('retrained')):
        loaded = auto_load_or_create(**kwargs)
    assert (loaded.patch_size, loaded.halo, loaded.periodic, loaded.N_r) == (2, 1, True, 80)
    np.testing.assert_array_equal(loaded.Wout, trained.Wout)
    u, r = data[0, :1].T, np.zeros((80, 1))
    for _ in range(5):
        (u1, _), (u, r) = trained.step(u, r), loaded.step(u, r)
        np.testing.assert_array_equal(u1, u)

    data = np.stack([np.stack([np.sin(a * t), np.cos(a * t)], axis=-1) for a in (1.0, 1.5)])   # two regimes
    kwargs = dict(data=data, dt=0.1, t_train=30., t_val=5., t_test=0., N_units=20, N_wash=5,
                  hyperparameters_to_optimize=[], tikh=1e-6, perform_test=False, plot_training=False,
                  input_parameters=np.array([[1.0, 1.5]]), param_names=('a',), param_values=(1.25,),
                  config_dir=tmp_path / 'readout')
    readout = dict(kwargs, param_in_reservoir=False, readout_input='params')
    hashes = {ESNConfig.from_init_params(**k).to_hash() for k in (kwargs, readout)}
    assert len(hashes) == 2
    trained = auto_load_or_create(**readout)
    assert trained.Wout.shape[0] == 20 + 1 + 1 and trained.Win[:, 2].nnz == 0   # [r; a; 1], a not in W_in
    with mock.patch.object(ESN_model, 'train', side_effect=AssertionError('retrained')):
        loaded = auto_load_or_create(**readout)
    assert (loaded.param_in_reservoir, loaded.readout_input) == (False, 'params')
    np.testing.assert_array_equal(loaded.Wout, trained.Wout)
    loaded.input_parameters = trained.input_parameters = np.array([[1.2]])
    u, r = data[0, :1].T, np.zeros((20, 1))
    for _ in range(5):
        (u1, _), (u, r) = trained.step(np.vstack([u, [[1.2]]]), r), loaded.step(np.vstack([u, [[1.2]]]), r)
        np.testing.assert_array_equal(u1, u)


def test_a_network_is_found_from_another_clone(tmp_path):
    """A network trained in one clone, with the training data given by its absolute
    path, is loaded (not trained again) in a clone whose path to the data differs."""
    kwargs = _small_network(config_dir=tmp_path / 'store', centre_input=False, validation_metric='log_nMSE')
    trained = auto_load_or_create(
        training_data_filename=tmp_path / 'clone_a' / 'results' / 'ESN_train_data_vdp', **kwargs)
    with mock.patch.object(ESN_model, 'train', side_effect=AssertionError('retrained')):
        loaded = auto_load_or_create(
            training_data_filename='/storage0/user/clone_b/results/ESN_train_data_vdp', **kwargs)
    np.testing.assert_array_equal(loaded.Wout, trained.Wout)
    assert loaded.centre_input is False and loaded.validation_metric_name == 'log_nMSE'
    (saved,) = (tmp_path / 'store').iterdir()
    assert saved.name == ESNConfig.from_init_params(
        training_data_filename='ESN_train_data_vdp', **kwargs).to_hash()
    # the path is kept in the saved configuration (where the data were), the name in the hash
    assert ESNConfig.load(saved).training_data_filename == str(tmp_path / 'clone_a' / 'results' / 'ESN_train_data_vdp')


def test_rekey_esn_store_moves_the_path_keyed_networks(tmp_path):
    """A network stored under the hash of a path (the rule before the name-keyed one)
    moves to the hash of the name, where the lookup now finds it; the rest stays put."""
    name = CACHED['training_data_filename']
    path_keyed = ESNConfig(**dict(CACHED, training_data_filename=f'/storage0/user/data/Lorenz63/{name}'))
    with mock.patch('romda.models.data_driven.rnns.esn_config.training_data_name', lambda f: f):
        legacy = path_keyed.to_hash()  # the hash of the path
    assert legacy != CACHED_HASH
    other = ESNConfig(**dict(CACHED, seed=1))

    def save(entry, config):
        config.save(tmp_path / entry)
        np.savez_compressed(tmp_path / entry / 'trained_matrices.npz', Wout=np.zeros(2))

    save(legacy, path_keyed)              # keyed by the path: moves
    save('0123456789abcdef', path_keyed)  # a query hash its configuration does not give: stays
    save(other.to_hash(), other)          # keyed by the name: stays
    path_keyed.save(tmp_path / 'no_matrices')  # not a complete entry: ignored

    assert rekey_esn_store(tmp_path, dry_run=True) == [(legacy, CACHED_HASH)]
    assert find_matching_config(tmp_path, legacy)  # a dry run moves nothing
    assert rekey_esn_store(tmp_path) == [(legacy, CACHED_HASH)]
    assert {p.name for p in tmp_path.iterdir()} == {CACHED_HASH, '0123456789abcdef', other.to_hash(), 'no_matrices'}
    moved = ESNConfig.load(find_matching_config(tmp_path, CACHED_HASH))
    assert moved.training_data_filename == path_keyed.training_data_filename
    assert rekey_esn_store(tmp_path) == []

    # the same network keyed by another path: its target is taken, so it stays
    twin = ESNConfig(**dict(CACHED, training_data_filename=f'/Users/user/data/Lorenz63/{name}'))
    with mock.patch('romda.models.data_driven.rnns.esn_config.training_data_name', lambda f: f):
        twin_legacy = twin.to_hash()
    save(twin_legacy, twin)
    assert rekey_esn_store(tmp_path) == []
    assert find_matching_config(tmp_path, twin_legacy) and find_matching_config(tmp_path, CACHED_HASH)

    # two path-keyed copies of one network: the first in name order moves, in a dry run too
    copies = [ESNConfig(**dict(CACHED, seed=2, training_data_filename=f'{root}/{name}'))
              for root in ('/storage0/user/data', '/Users/user/data')]
    with mock.patch('romda.models.data_driven.rnns.esn_config.training_data_name', lambda f: f):
        copy_legacy = [copy.to_hash() for copy in copies]
    for entry, copy in zip(copy_legacy, copies):
        save(entry, copy)
    expected = [(min(copy_legacy), copies[0].to_hash())]
    assert rekey_esn_store(tmp_path, dry_run=True) == expected
    assert rekey_esn_store(tmp_path) == expected
    assert find_matching_config(tmp_path, copies[0].to_hash()) and find_matching_config(tmp_path, max(copy_legacy))
