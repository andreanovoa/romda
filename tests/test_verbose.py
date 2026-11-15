"""Verbosity: romda is silent by default; romda.set_verbose shows its diagnostic messages and those of dynamodels and
echostatenetwork."""
import logging
import re

import dynamodels
import echostatenetwork
import numpy as np
import pytest
import romda
from conftest import vdp_twin
from echostatenetwork import EchoStateNetwork
from romda.models.physical import Lorenz63

#: dynamodels: the pool of an ensemble forecast ('Initializing multiprocessing pool ...' in 0.6.0,
#: '<model>: starting a pool of n worker processes ...' later)
POOL = re.compile(r'\[dynamodels\] (Initializing multiprocessing pool|\w+: starting a pool of)')
NOISE = 'Adding noise'                          # romda: the noise of the observations
#: dynamodels >= the release with set_verbose logs the pool instead of printing it
LOGGING_DYNAMODELS = hasattr(dynamodels, 'set_verbose')
RETRAIN = 'training the trained network again'  # echostatenetwork: a trained network trained with force=True
LOGGING_ESN = hasattr(echostatenetwork, 'set_verbose')


def ensemble_forecast():
    """A forecast of an ensemble of 3 members, which starts the pool of the IVP integrator."""
    model = Lorenz63()
    model.init_ensemble(m=3, std_phi=0.1)
    psi, _ = model.time_integrate(Nt=5)
    model.close()
    return psi


def noisy_observations():
    return vdp_twin('linear')


def esn_trained_twice():
    """A small echo state network trained twice (the second time with force=True), which logs the retraining."""
    data = np.random.default_rng(0).normal(size=(200, 3)).cumsum(axis=0) * 0.1
    esn = EchoStateNetwork(data.T, dt=1, N_units=15, upsample=1, t_train=120, t_val=30, t_test=40, N_wash=5,
                           hyperparameters_to_optimize=[], seed=0)
    for force in (False, True):
        esn.train(data, plot_training=False, force=force)


@pytest.fixture(autouse=True)
def quiet():
    yield
    romda.set_verbose(False)


def test_silent_by_default(capsys, caplog):
    if LOGGING_DYNAMODELS:
        assert ensemble_forecast().shape == (5, 3, 3)
    noisy_observations()
    out = capsys.readouterr()
    assert out.out == '' and out.err == ''
    assert 'pool' not in caplog.text and NOISE not in caplog.text


def test_set_verbose_on_and_off(capsys, caplog):
    romda.set_verbose(True)
    romda.set_verbose(True)                                         # idempotent: one handler per package
    if LOGGING_DYNAMODELS:
        dynamodels.set_verbose(True)                                # one handler, with dynamodels' switch too
    for package in ('romda', 'dynamodels', 'echostatenetwork'):
        handlers = logging.getLogger(package).handlers
        assert sum(h.get_name() == f'{package}.verbose' for h in handlers) == 1
    ensemble_forecast()
    noisy_observations()
    assert NOISE in caplog.text
    err = capsys.readouterr().err
    assert '[romda] ' + NOISE in err
    if LOGGING_DYNAMODELS:
        assert POOL.search(err) and 'pool' in caplog.text

    romda.set_verbose(False)
    caplog.clear()
    ensemble_forecast()
    noisy_observations()
    assert 'pool' not in caplog.text and NOISE not in caplog.text
    assert capsys.readouterr().err == ''


@pytest.mark.skipif(not LOGGING_ESN, reason='echostatenetwork without set_verbose')
def test_set_verbose_switches_echostatenetwork(capsys, caplog):
    esn_trained_twice()
    assert RETRAIN not in caplog.text and capsys.readouterr().err == ''
    romda.set_verbose(True)
    echostatenetwork.set_verbose(True)                              # one handler, as with dynamodels
    handlers = logging.getLogger('echostatenetwork').handlers
    assert sum(h.get_name() == 'echostatenetwork.verbose' for h in handlers) == 1
    esn_trained_twice()
    assert RETRAIN in caplog.text and '[echostatenetwork] ' in capsys.readouterr().err
    romda.set_verbose(False)
    caplog.clear()
    esn_trained_twice()
    assert RETRAIN not in caplog.text
