"""`romda.utils` and `romda.plotting`: the project-root search, the shared measurement-noise generator, the key of a
record, the collection of a pool job and the Times font of the figures."""
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import romda
from romda import observations, utils
from romda.utils import add_noise_to_flow, get_project_root, noise_generator, noise_rng, seed_noise

PACKAGE_ROOT = Path(romda.__file__).resolve().parents[2]       # <root>/src/romda/__init__.py
#: romda installed from a wheel (site-packages) has no clone around it: the tests of the clone do not apply
from_clone = pytest.mark.skipif(not (PACKAGE_ROOT / 'src' / 'romda').is_dir(), reason='romda is not run from a clone')


def test_the_project_root_is_the_closest_folder_with_src(tmp_path, monkeypatch):
    """The project-root search must not stop at the filesystem root: a project around the working directory wins,
    and without one and without a clone the search fails."""
    (tmp_path / 'src').mkdir()
    start = tmp_path / 'a' / 'b'
    start.mkdir(parents=True)
    assert get_project_root(str(start)) == str(tmp_path.resolve())
    assert get_project_root(str(tmp_path)) == str(tmp_path.resolve()) != str(PACKAGE_ROOT)
    # romda installed outside a clone (no src/romda above the package) and no project around the directory
    outside = tmp_path / 'a' / 'outside'
    outside.mkdir()
    (tmp_path / 'src').rmdir()
    monkeypatch.setattr(utils, '__file__', str(tmp_path / 'site-packages' / 'romda' / 'utils.py'))
    with pytest.raises(FileNotFoundError):
        get_project_root(str(outside))


@from_clone
def test_outside_a_project_the_clone_of_romda_is_the_root(tmp_path):
    """From outside a project the root is the clone that holds romda, so that the model stores under results/ are
    shared instead of falling back on the working directory."""
    # no src folder in tmp_path or its parents (the filesystem root is not searched): the package's own clone
    assert get_project_root(str(tmp_path)) == str(PACKAGE_ROOT)
    # MODEL_STORE is set at import: from a working directory outside any project it is results/model_configs of
    # the clone, not of the working directory (the ESN store sits next to it)
    code = 'from romda.models.data_driven.model_config import MODEL_STORE; print(MODEL_STORE)'
    out = subprocess.run([sys.executable, '-c', code], cwd=tmp_path, capture_output=True, text=True, check=True)
    assert Path(out.stdout.strip()) == PACKAGE_ROOT / 'results' / 'model_configs'


def test_seed_noise_restarts_the_shared_generator_in_place():
    held = observations.rng                  # a module holds the generator itself
    assert held is noise_rng
    seed_noise(3)
    first = held.normal(size=5)
    seed_noise(3)
    assert np.array_equal(held.normal(size=5), first)
    # a Model with seed s draws its ensemble from default_rng(s): the noise must not repeat it
    assert not np.allclose(noise_generator(3).normal(size=8), np.random.default_rng(3).normal(size=8))


def test_observation_and_flow_noise_follow_the_seed():
    from romda.observations import Observations

    def noisy_observations():
        t = np.linspace(0., 1., 401)
        y = np.column_stack([np.sin(2 * np.pi * 5 * t), np.cos(2 * np.pi * 3 * t)])
        return Observations(y_raw=y, y_true=y, t_true=t, t_start=0.1, t_stop=0.9, Nt_obs=20,
                            add_noise=True, noise_level=0.1).y_obs

    seed_noise(1)
    first = noisy_observations()
    seed_noise(1)
    assert np.array_equal(noisy_observations(), first)
    seed_noise(2)
    assert not np.allclose(noisy_observations(), first)
    # the flow noise takes the shared or a given generator
    U = np.ones((2, 4, 6, 5))
    seed_noise(0)
    shared = add_noise_to_flow(U, noise_level=0.1)
    seed_noise(0)
    assert np.array_equal(add_noise_to_flow(U, noise_level=0.1), shared)
    own = add_noise_to_flow(U, noise_level=0.1, rng=np.random.default_rng(5))
    assert np.array_equal(add_noise_to_flow(U, noise_level=0.1, rng=np.random.default_rng(5)), own)


def test_record_key_holds_the_record_content():
    """Two records with one file name (e.g. the same options integrated by two environments) may never share a
    network: `record_key` tags the name with a digest of the record, round-off differences included."""
    from romda.utils import record_key

    name = 'Lorenz63_default_Nlyap300_noise0.05_seed0'
    record = np.random.default_rng(0).normal(size=(200, 3))
    perturbed = record.copy()
    perturbed[-1, -1] += 1e-15 * abs(perturbed[-1, -1])
    key = record_key(name, record)
    assert key.startswith(name + '@')
    assert key == record_key(name, record.copy(order='F'))        # content, not layout
    assert key != record_key(name, perturbed)
    assert key != record_key(name, record.reshape(300, 2))        # shape counts too


def test_a_collected_job_frees_the_cycles_of_the_earlier_jobs():
    """A pool worker reuses its process: the ensembles of its earlier jobs sit in reference cycles (a model and its
    integrator), and numpy's allocations do not trigger a collection. A job decorated with `collected` frees them
    before it runs."""
    import gc
    import weakref

    from romda.utils import collected

    class Ensemble:                                   # a cycle that holds a large array, as a model and its integrator
        def __init__(self):
            self.integrator, self.history = types.SimpleNamespace(model=self), np.zeros(10 ** 6)

    alive = []

    def job():
        e = Ensemble()
        alive.append(weakref.ref(e))
        return [ref() is not None for ref in alive[:-1]]

    gc.disable()
    try:
        job()
        assert job() == [True]                        # the first ensemble outlives its job
        assert collected(job)() == [False, False]     # freed before the next collected job runs
    finally:
        gc.enable()


def test_use_times_font_sets_a_serif_times_style():
    """The figures are drawn in Times with STIX mathematics."""
    import matplotlib.pyplot as plt
    from romda.plotting.utils_plotting import use_times_font
    with plt.rc_context():
        use_times_font()
        assert plt.rcParams['font.family'] == ['serif']
        assert plt.rcParams['font.serif'][0] == 'Times New Roman'
        assert plt.rcParams['mathtext.fontset'] == 'stix'
