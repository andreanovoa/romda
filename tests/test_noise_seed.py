"""The shared measurement-noise generator (`romda.utils.noise_rng`): a seed fixes the noise of a run."""
import os

import numpy as np
import yaml
from romda import observations
from romda.experiments import run_experiment
from romda.observations import Observations
from romda.utils import add_noise_to_flow, noise_generator, noise_rng, seed_noise

VDP = os.path.join(os.path.dirname(__file__), '..', 'examples', 'thermoacoustics', 'vdp.yml')


def _noisy_observations():
    t = np.linspace(0., 1., 401)
    y = np.column_stack([np.sin(2 * np.pi * 5 * t), np.cos(2 * np.pi * 3 * t)])
    return Observations(y_raw=y, y_true=y, t_true=t, t_start=0.1, t_stop=0.9, Nt_obs=20,
                        add_noise=True, noise_level=0.1).y_obs


def test_seed_noise_restarts_the_generator_in_place():
    held = observations.rng                  # a module holds the generator itself
    assert held is noise_rng
    seed_noise(3)
    first = held.normal(size=5)
    seed_noise(3)
    assert np.array_equal(held.normal(size=5), first)


def test_observation_noise_follows_the_seed():
    seed_noise(1)
    first = _noisy_observations()
    seed_noise(1)
    assert np.array_equal(_noisy_observations(), first)
    seed_noise(2)
    assert not np.allclose(_noisy_observations(), first)


def test_flow_noise_takes_the_shared_or_a_given_generator():
    U = np.ones((2, 4, 6, 5))
    seed_noise(0)
    shared = add_noise_to_flow(U, noise_level=0.1)
    seed_noise(0)
    assert np.array_equal(add_noise_to_flow(U, noise_level=0.1), shared)
    own = add_noise_to_flow(U, noise_level=0.1, rng=np.random.default_rng(5))
    assert np.array_equal(add_noise_to_flow(U, noise_level=0.1, rng=np.random.default_rng(5)), own)


def test_a_run_is_reproducible_from_its_seed():
    small = dict(m=5)
    same = [run_experiment(VDP, seed=1, ensemble_kwargs=small)[2] for _ in range(2)]
    other = run_experiment(VDP, seed=2, ensemble_kwargs=small)[2]
    assert same[0] == same[1]
    assert other['rmse_da'] != same[0]['rmse_da']


def test_the_noise_stream_is_not_the_stream_of_a_model_seed():
    # a Model with seed s draws its ensemble from default_rng(s): the noise must not repeat it
    assert not np.allclose(noise_generator(3).normal(size=8), np.random.default_rng(3).normal(size=8))


def test_a_seed_in_the_da_block_seeds_the_noise_too():
    with open(VDP) as stream:
        cfg = yaml.safe_load(stream)
    cfg['da'] = {**cfg['da'], 'seed': 2}
    small = dict(m=5)
    assert run_experiment(cfg, ensemble_kwargs=small)[2] == run_experiment(VDP, seed=2, ensemble_kwargs=small)[2]


def test_an_esn_ensemble_follows_the_seed_it_is_given():
    from romda.estimators import EnSRKF
    from romda.models.data_driven import ESN_model
    from romda.models.physical import Lorenz63

    data = Lorenz63(dt=0.01).time_integrate(Nt=3000)[0][None, :, :, 0]
    esn = ESN_model(dt=0.01, data=data, plot_training=False, N_units=30, N_wash=10, t_train=20.0, t_val=5.0,
                    t_test=0.0, hyperparameters_to_optimize=[], rho=0.9, sigma_in=0.1, tikh=1e-8, seed=0)

    def start(**seed):
        return EnSRKF(parent_model=esn, m=4, **seed).model.current_state.copy()

    assert np.array_equal(start(seed=1), start(seed=1))
    assert not np.allclose(start(seed=1), start(seed=2))
    assert np.array_equal(start(), start(seed=0))        # without a seed, the copy keeps the seed of its network
