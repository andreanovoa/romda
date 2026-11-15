"""The sweep harness of the scripts (romda.sweep): the flat keys of a grid point, and the resume of a sweep only
under the configuration that stamped its records."""

import json
import sys

import matplotlib

matplotlib.use('Agg')

import pytest  # noqa: E402
from romda.sweep import config_kwargs, filter_context, main, run_sweep  # noqa: E402


def test_context_keeps_only_the_records_stamped_with_the_same_defaults(tmp_path):
    now = dict(da=dict(m=20, adaptive_inflation=False), dataset=dict(dt=0.02))
    old = dict(m=20, seed=0, status='ok', context=dict(da=dict(m=20)))
    new = dict(m=20, seed=1, status='ok', context=now)

    assert filter_context([old, new], now) == [new]
    assert filter_context([dict(m=20, seed=0)], now) == []     # never stamped

    # the resume of `run_sweep` compares in the same way: only the seed of the old record runs again
    path = f'{tmp_path}/records.jsonl'
    with open(path, 'w') as f:
        f.write(json.dumps(old) + '\n' + json.dumps(new) + '\n')
    runs = []
    run_sweep(lambda **kw: runs.append(kw) or {}, dict(m=[20]), path, n_seeds=2, context=now)
    assert runs == [dict(m=20, seed=0)]


def test_config_kwargs_routes_the_flat_keys_of_a_grid_point():
    """The flat keys of a sweep go to the blocks of `romda.run.run_config`; None keeps the configuration."""
    kwargs = config_kwargs(seed=3, da_method='EnKF', std_obs=0.05, m=20, inflation_factor=1.02, Nt_obs=10,
                           obs_every=2, ER=0.5125, regularization_factor=None, bias_kwargs=dict(L=10))
    assert kwargs == dict(da_method='EnKF', std_obs=0.05, bias_kwargs=dict(L=10),
                          truth_kwargs=dict(Nt_obs=10, observe_every=2, ER=0.5125),
                          ensemble_kwargs=dict(seed=3, m=20, inflation=1.02))
    assert config_kwargs(regularization_factor=2.)['ensemble_kwargs'] == dict(bias_regularization=2.)
    with pytest.raises(ValueError, match='unknown_key'):
        config_kwargs(unknown_key=1)


def test_sweep_main_resumes_only_under_the_same_configuration(monkeypatch, tmp_path):
    """`main` stamps the records with the configuration of the case: a second call resumes, a changed default
    re-runs the grid."""
    runs = []
    studies = dict(size=dict(axes=dict(m=[10, 20]), x_key='m'))

    def sweep(cfg):
        monkeypatch.setattr(sys, 'argv', ['sweeps', '--study', 'size', '--seeds', '1'])
        main(studies, lambda **kw: runs.append(kw) or dict(rmse_da=0.1, rmse_forecast=0.2),
             str(tmp_path / '{study}.jsonl'), str(tmp_path / '{study}.png'), context=cfg)

    sweep(dict(da=dict(inflation=1.02)))
    sweep(dict(da=dict(inflation=1.02)))
    assert len(runs) == 2
    sweep(dict(da=dict(inflation=1.05)))
    assert len(runs) == 4
