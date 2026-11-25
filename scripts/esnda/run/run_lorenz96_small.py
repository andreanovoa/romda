"""L96-10s of the washout-free paper: one ESN trained at the single forcing F = 8 (where the ten-site Lorenz-96 system
is chaotic), the truth from the held-out end of its own training record (configs/lorenz96_small.yml).
`experiment` deploys the network on one window and scores the full state (run_and_score).

    python run_lorenz96_small.py partial N   -> results/partial/partial_lorenz96_<N>.jsonl (Sect. 4.2): the filter, with the
        innovation-based inflation, and the open-loop substitution from N = 2, 3 or 5 sensors, seeds 0-4,
        20 T_lambda of assimilation; one JSON line per (deployment, seed) with the run_and_score metrics
        (rmse_mean_last = e_W, over the last T_lambda of the data).
    python run_lorenz96_small.py uq          -> results/uncertainty/uq_lorenz96_small.jsonl (Sect. 4.5): spread against error of the
        global filter with every site observed, m = 64, with the innovation-based inflation and without (inflation 1),
        seeds (windows) 0-4. Spread and error of the ensemble mean from 2 T_lambda after the first observation to the
        last (spread_sync, rmse_mean_sync); the spread with ddof = 1, as run_parallel_localization.py uq10.

C_dd is 0.2 of the standard deviation of each observable (CDD_STD, the paper) unless the environment sets it.
"""

import json
import multiprocessing
import os
import sys

from pathlib import Path  # noqa: E402

from romda.run import innovation_options, load_config  # noqa: E402
from romda.data_assimilation import run_and_score  # noqa: E402
from romda.models.data_driven.rnns import model_esn, twin_data  # noqa: E402
from romda.models.physical import Lorenz96  # noqa: E402
from romda.observations import measured_idx  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # scripts/esnda: results/
CONFIGS = Path(__file__).resolve().parents[1] / 'configs'   # the configurations of the experiments
DEFAULTS = load_config(CONFIGS / 'lorenz96_small.yml')


# ------------------------- Default parameters ------------------------- #

#: configs/lorenz96_small.yml defines *all* defaults of this experiment
#: (dataset, ESN, DA, window) -- edit values there.
NX = DEFAULTS['truth']['params']['Nx']
#: `est_wout`, `start_param` and `twin` are options of this folder only, not `da` keys of
#: `romda.run.config`, so the yml cannot hold them.
DEFAULT_PARAMS = {'est_wout': False, 'start_param': 0, 'twin': False, 'ddof': 0, **DEFAULTS['da']}   # ddof: of the spread
DEFAULT_ESN = DEFAULTS['forecast']['esn']
WINDOW_LYAP = DEFAULTS['window_lyap']

#: Relative spread of the initial readout singular values (tutorial 40).
STD_WOUT = 0.04


def experiment(**overrides):
    """One (grid point, seed) run: build the truth window, deploy, score the full state.

    ``est_wout`` appends ``'Wout'`` to `est_alpha`: each member carries its own readout
    singular values, which the filter estimates alongside the state. ``twin`` replaces
    the held-out record with the closed-loop forecast of the network itself (`twin_data`),
    so that the forecast model has no model error."""
    assert not (bad := set(overrides) - set(DEFAULT_PARAMS)), f'unknown parameter(s): {bad}'
    p = {**DEFAULT_PARAMS, **overrides}
    esn, data = model_esn(Lorenz96, DEFAULTS['truth'], DEFAULT_ESN,
                          measured=measured_idx(p['n_measured'], NX, ring=True),
                          bayesian_update=p['bayesian_update'])
    measured = measured_idx(p['n_measured'], esn.N_dim, ring=True)
    wout = dict(est_alpha=['Wout'], std_alpha=STD_WOUT) if p['est_wout'] else {}
    Nt_obs = p['Nt_obs']
    if p['twin']:
        # the twin record is sampled at dt_ESN = upsample * dt, and Nt_obs counts record steps
        assert Nt_obs % esn.upsample == 0, 'twin: Nt_obs must be a multiple of the upsample'
        data, Nt_obs = twin_data(esn, data), Nt_obs // esn.upsample
    return run_and_score(esn, data, measured, p['seed'], p['da_method'], p['m'],
                         Nt_obs, p['std_obs'], p['inflation'], **innovation_options(p),
                         cdd_fraction=float(os.environ.get('CDD_STD', p['cdd_fraction'])),
                         window_lyap=WINDOW_LYAP, bayesian_update=p['bayesian_update'],
                         adaptive_inflation=p['adaptive_inflation'],
                         inflate_parameters=p['inflate_parameters'],
                         start_param=p['start_param'], ddof=p['ddof'], **wout)


def partial(n):
    out = os.path.join(HERE, 'results', 'partial', f'partial_lorenz96_{n}.jsonl')
    open(out, 'w').close()
    for dep, kw in (('filter', dict(bayesian_update=True, inflation='innovation')),
                    ('substitution', dict(bayesian_update=False))):
        for seed in range(5):
            r = experiment(n_measured=n, seed=seed, **kw)
            r = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in r.items()
                 if not isinstance(v, (dict, list))}
            with open(out, 'a') as f:
                f.write(json.dumps({**r, 'n_measured': n, 'seed': seed, 'deployment': dep,
                                    'inflation': kw.get('inflation')}) + '\n')
            print(dep, n, seed, r.get('rmse_mean_last'), r.get('valid_time_forecast'), flush=True)


M_UQ = 64


def uq_job(j):
    inflation, seed = j
    r = experiment(n_measured=10, m=M_UQ, seed=seed, inflation=inflation, ddof=1)
    ratio = r['spread_sync'] / r['rmse_mean_sync']
    return dict(inflation=str(inflation), seed=seed, error=r['rmse_mean_sync'], ratio=float(ratio),
                e_end=r['rmse_mean_last'], valid_time=r['valid_time_mean_forecast'])


def uq():
    jobs = [(g, s) for g in ('innovation', 1.0) for s in range(5)]
    # N_WORKERS: explicit, small by default; a worker of the L96-40 sweeps can hold tens of GB
    with multiprocessing.Pool(int(os.environ.get('N_WORKERS', 2))) as pool:
        rows = pool.map(uq_job, jobs)
    with open(os.path.join(HERE, 'results', 'uncertainty', 'uq_lorenz96_small.jsonl'), 'w') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
            print(row, flush=True)


if __name__ == '__main__':
    # Python 3.14 defaults to forkserver on Linux, whose pool workers die silently
    # (dataset creation integrates through IVPIntegrator); same fix as conftest.py.
    multiprocessing.set_start_method('fork', force=True)
    partial(int(sys.argv[2])) if sys.argv[1] == 'partial' else uq()
