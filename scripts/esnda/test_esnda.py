"""The run scripts of the washout-free ESN-DA paper (scripts/esnda/run) as the plot scripts read them.

A plot script reads the namespace of a run script (``runpy.run_path(...)`` or ``import run_... as``) at import, where a
name that the run script defines only under ``__main__`` or no longer defines fails with a KeyError or an
AttributeError when the script draws the figure. The layout of run_parallel_localization.py sets the training washout
and the names of the outputs that reproduce.sh waits for.

    pytest scripts/esnda
"""

import glob
import os
import re
import runpy
import sys

import matplotlib

matplotlib.use('Agg')

import numpy as np  # noqa: E402
import pytest  # noqa: E402

ESN_DA = os.path.dirname(os.path.abspath(__file__))
RUN, PLOT = os.path.join(ESN_DA, 'run'), os.path.join(ESN_DA, 'plot')
LAYOUT_KEYS = ('NX', 'G_SITES', 'HALO', 'N_R', 'N_WASH', 'TRAIN_LYAP', 'TAG', 'ESN_CONFIG')


@pytest.fixture
def scripts(monkeypatch):
    """Load a run script by name, with the layout of the environment `env` (the defaults otherwise)."""
    for p in (ESN_DA, RUN, PLOT):
        monkeypatch.syspath_prepend(p)
    monkeypatch.chdir(ESN_DA)
    monkeypatch.setattr(sys, 'argv', ['test'])

    def load(name, **env):
        for k in LAYOUT_KEYS:
            monkeypatch.delenv(k, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, str(v))
        return runpy.run_path(os.path.join(RUN, name + '.py'), run_name='test')
    return load


def test_scripts_read_only_names_that_the_run_scripts_define(scripts):
    spaces = {name: scripts(name) for name in ('run_parallel_localization', 'run_parametric',
                                                'run_synchronization')}
    missing = []
    for path in sorted(glob.glob(os.path.join(PLOT, '*.py')) + glob.glob(os.path.join(RUN, '*.py'))):
        src = open(path).read()
        names = {m.group(1): m.group(2) for m in re.finditer(
            r"^(\w+)\s*=\s*runpy\.run_path\(os\.path\.join\(\w+, '(run_\w+)\.py'\)", src, re.M)}
        if 'def run_script' in src:                      # run_verification.py: run_script() is a runpy dict
            names.update(PAR='run_parallel_localization')
        names.update({m.group(2): m.group(1) for m in re.finditer(r"^\s*import (run_\w+) as (\w+)", src, re.M)})
        for m in re.finditer(r"^\s*from (run_\w+) import ([\w, ]+)", src, re.M):
            missing += [(path, n) for n in m.group(2).replace(' ', '').split(',')
                        if m.group(1) in spaces and n not in spaces[m.group(1)]]
        for var, script in names.items():
            if script in spaces:
                used = set(re.findall(rf"\b{var}\[['\"](\w+)['\"]\]", src)) | set(re.findall(rf"\b{var}\.(\w+)", src))
                missing += [(path, k) for k in used if k not in spaces[script] and not k.startswith('__')]
    assert not missing, missing


def test_layouts_of_the_paper(scripts):
    """The single reservoir L96-40s trains after a washout of 100 steps and the parallel networks after 20, and the
    names of their outputs are those that notebook 03 and plot_aux.py read."""
    for env, wash, suffix, steps, patches in (
            ({}, 20, '_w20', 30, 8),                                                              # L96-40p
            (dict(G_SITES=40, HALO=0, N_R=6000, TRAIN_LYAP=800), 100, '_g40_h0_n6000', 30, 1),   # L96-40s: Table 1
            (dict(NX=10, G_SITES=5, HALO=2, N_R=500), 20, '_g5_h2_n500_nx10_w20', 42, 2)):       # L96-10p
        P = scripts('run_parallel_localization', **env)
        assert (P['N_WASH'], P['SUFFIX'], P['STEPS_LYAP'], P['N_P']) == (wash, suffix, steps, patches), env


def test_l96_40s_reads_out_with_the_tikhonov_factor_of_the_paper(scripts):
    """L96-40s searches its readout at the fixed Tikhonov factor of the paper, 1e-4 (Sect. 4.4: the LETKF fails on a
    single reservoir with a weakly regularized readout); the parallel networks keep the grid of the configuration, and
    dev/esn_da/run_readout_control.py still replaces the factor through OVERRIDES."""

    class TrainedError(Exception):
        pass

    def esn_settings(env, overrides=None):
        P = scripts('run_parallel_localization', **env)
        g, seen = P['network'].__wrapped__.__globals__, {}

        def model_esn(model, truth, esn, **kwargs):
            seen.update(esn)
            raise TrainedError
        g['model_esn'] = model_esn
        g['OVERRIDES'] = overrides or {}
        with pytest.raises(TrainedError):
            P['network'].__wrapped__()
        return seen

    single = dict(G_SITES=40, HALO=0, N_R=6000)
    assert esn_settings(dict(single, TRAIN_LYAP=800))['tikh_range'] == [1e-4]
    assert esn_settings(single)['tikh_range'] == [1e-4]
    assert len(esn_settings({})['tikh_range']) > 1                                    # L96-40p: the grid
    assert len(esn_settings(dict(NX=10, G_SITES=5, HALO=2, N_R=500))['tikh_range']) > 1   # L96-10p
    control = esn_settings(single, dict(tikh=1e-12, hyperparameters_to_optimize=[]))
    assert control['tikh'] == 1e-12 and control['hyperparameters_to_optimize'] == []


def test_parametric_constants_and_pool_jobs(scripts):
    """The constants of run_parametric.py follow its configuration; every pool job that builds a filter without
    romda.data_assimilation.deploy is decorated with romda.utils.collected."""
    import yaml

    P = scripts('run_parametric')
    cfg = yaml.safe_load(open(os.path.join(ESN_DA, 'configs', 'lorenz96_parametric.yml')))
    assert P['VALUES'] == cfg['forecast']['train']['values']
    assert P['DT'] == pytest.approx(0.02) and P['STEPS_LYAP'] == 42 and P['NOISE'] == cfg['truth']['noise_level']
    assert P['STEPS_LYAP'] == round(P['T_LYAP'] / P['DT'])
    jobs = {'run_parallel_localization': ('filter_run',), 'run_parametric': ('filter_run',)}
    for script, names in jobs.items():
        P = scripts(script)
        for name in names:
            assert P[name].__name__ == name and hasattr(P[name], '__wrapped__'), (script, name)


def test_the_gate_rejects_a_readout_that_moves_with_the_forcing(scripts):
    """The gate of the parametric networks: from the state of a training regime with the input of another forcing, a
    readout that moves by 20 per unit of F leaves 3 times the training range in one step; one that moves by 0.5 does
    not. The training forcings of the network are restored."""
    from echostatenetwork import EchoStateNetwork

    for sensitivity, leaves in ((20.0, True), (0.5, False)):
        P = scripts('run_parametric')
        nx = P['NX']

        class Network:                               # one step: u = sensitivity (F - 3.5), whatever the state
            N_r, N_dim = 6, nx
            shift, norm = np.zeros(nx), np.full(nx, 4.0)        # |u| <= 4 on the training records: the bound is 12
            input_parameters = np.array([[3.5, 3.75]])
            _n_param = staticmethod(EchoStateNetwork._n_param)
            _parameters_of_columns = EchoStateNetwork._parameters_of_columns
            closed_loop_step, forecast = EchoStateNetwork.closed_loop_step, EchoStateNetwork.forecast

            def regime_matched_states(self, m, regimes, seed=0):
                return np.zeros((nx + self.N_r, m)), regimes[:, :m]

            def outputs_to_inputs(self, u):
                return u

            def step(self, u, r, s=sensitivity):
                return s * (self.input_parameters - 3.5) * np.ones((nx, 1)), r

        net = Network()
        P['gate'].__globals__['network'] = lambda *a, **k: (net, np.array([[3.5, 3.75]]))
        left = P['gate']('readout')
        assert bool(left) == leaves
        if leaves:
            assert 3.5 not in left and 4.0 not in left and {0.5, 4.5, 10.0} <= set(left)
        assert np.array_equal(net.input_parameters, [[3.5, 3.75]])


def test_the_parameter_scan_resumes_only_under_its_own_settings(scripts, tmp_path):
    """run_parametric.py scan stores its settings with the rows, and refuses to resume a file of other settings."""
    P = scripts('run_parametric')
    g = P['scan'].__globals__
    runs = []
    g.update(HERE=str(tmp_path), F_TRUE=[6.5], INTERVALS=[0.1, 0.3],
             _pool=lambda fn, jobs: [runs.append(j) or dict(rho_true=j[0], interval=j[1], trace={}) for j in jobs])
    os.makedirs(tmp_path / 'results' / 'parameter')
    P['scan']()
    P['scan']()                                              # the same settings: nothing left to run
    assert len(runs) == 2
    g['CDD_STD'] = 2 * g['CDD_STD']
    with pytest.raises(SystemExit, match='settings'):
        P['scan']()


def test_run_lorenz96_small_takes_the_yml_defaults(monkeypatch):
    import importlib

    monkeypatch.syspath_prepend(RUN)
    script = importlib.import_module('run_lorenz96_small')
    # every `da` key of the yml is a parameter of `experiment`, and a misspelt one is refused
    assert set(script.DEFAULTS['da']) <= set(script.DEFAULT_PARAMS)   # the resolved configuration
    with pytest.raises(AssertionError, match='unknown parameter'):
        script.experiment(not_a_parameter=1)
