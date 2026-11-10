"""The network store's hash must tell hyperparameter searches apart, without moving
any network cached before the search settings were hashed."""

from unittest import mock

import numpy as np
from romda.models.data_driven.rnns import ESN_model, ESNConfig, auto_load_or_create

# results/esn_configs/1afa29689edcdc37/esn_config.yaml: a Lorenz-63 network cached
# before the search settings joined the hash
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


def test_search_settings_change_the_hash():
    hashes = {ESNConfig(**dict(CACHED, **change)).to_hash() for change in (
        {},
        dict(hyperparameters_to_optimize=('rho', 'sigma_in', 'tikh', 'bias_in')),
        dict(bias_in_range=(0.0, 1.0)),
        dict(regularize_bias_out=False),
        dict(stability_steps=1350),
        dict(bho_noise='gaussian'),
        dict(grid_midpoints=True),
        dict(optimize_parameter_normalization=True),
        dict(param_shift_range=(3.5, 8.0)),
        dict(param_norm_range=(1.0, 20.0)))}
    assert len(hashes) == 10


def test_init_params_keep_the_bho_options():
    """`from_init_params` is how `auto_load_or_create` keys a query: the options must
    reach the hash, not be dropped on the way."""
    default = ESNConfig.from_init_params(**CACHED).to_hash()
    gated = ESNConfig.from_init_params(**dict(CACHED, stability_steps=1350)).to_hash()
    assert default == CACHED_HASH and gated != CACHED_HASH


def test_esn_defaults_match_the_store_defaults():
    """The defaults dropped from the hash are the class defaults of echostatenetwork:
    a network trained without the option is the network trained with its default."""
    from echostatenetwork import EchoStateNetwork
    for key in ('stability_steps', 'bho_noise', 'grid_midpoints', 'bias_in_range',
                'regularize_bias_out', 'optimize_parameter_normalization', 'param_shift_range',
                'param_norm_range'):
        assert getattr(EchoStateNetwork, key) == getattr(ESNConfig, key), key


def test_searched_bias_in_round_trips_through_the_store(tmp_path):
    t = np.arange(400) * 0.1
    data = np.stack([np.sin(t), np.cos(t)], axis=-1)[np.newaxis]
    kwargs = dict(data=data, dt=0.1, t_train=30., t_val=5., t_test=0., N_units=20,
                  N_wash=5, N_folds=2, N_grid=2, N_func_evals=8, perform_test=False,
                  plot_training=False, config_dir=tmp_path,
                  hyperparameters_to_optimize=['rho', 'sigma_in', 'tikh', 'bias_in'],
                  bias_in_range=(0.0, 1.0), regularize_bias_out=False,
                  stability_steps=20, grid_midpoints=True)
    trained = auto_load_or_create(**kwargs)
    (saved,) = tmp_path.iterdir()

    config = ESNConfig.load(saved)
    assert config.hyperparameters_to_optimize == kwargs['hyperparameters_to_optimize']
    assert tuple(config.bias_in_range) == kwargs['bias_in_range']
    assert config.regularize_bias_out is False
    assert config.stability_steps == 20 and config.grid_midpoints is True

    # found under the same hash (load -> to_esn_model), not trained again
    with mock.patch.object(ESN_model, 'train', side_effect=AssertionError('retrained')):
        loaded = auto_load_or_create(**kwargs)
    np.testing.assert_array_equal(loaded.Wout, trained.Wout)
    np.testing.assert_array_equal(loaded.bias_in, trained.bias_in)    # the searched value
    assert loaded.hyperparameters_to_optimize == kwargs['hyperparameters_to_optimize']
    assert loaded.regularize_bias_out is False
