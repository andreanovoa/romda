"""Table 2 of the washout-free paper: every measured subset of Lorenz-63, the open-loop substitution against the filter
that corrects the input alone and the filter that corrects the input and the reservoir, with the configuration of
Table 1 (configs/lorenz63.yml), over N_INIT initializations.

Initialization i draws the start of the window uniformly over the test record (rng(i)) and the starting reservoir
states (esn.seed = i, `initialize_from_val_data`): one network for the open loop, the m members for the filters.
The same draw serves every subset and deployment. Scores of `run_and_score` (romda.metrics.state_metrics, one
normalization for all components): error over the last Lyapunov time of the data, e_W, and valid time of the
forecast after the last observation. A run fails when e_W exceeds 0.5, the threshold of the valid time;
the medians are over the runs that do not diverge. The filters keep every analysis, as in the paper (the runs before
7 Nov 2026 rejected an analysis that moved the mean of the state by more than one climatological standard deviation,
max_mean_shift of romda.estimators). The 5-window version is results/partial/partial_table_lorenz63_5windows.json.

    python run_partial_table.py [N_INIT]    -> results/partial/partial_table_lorenz63.json, and the LaTeX rows
    python run_partial_table.py table       -> the LaTeX rows of the stored runs

M=10 sets the members of the filters (default: the yml), into results/partial/partial_table_lorenz63_m10.json.
SUBSETS=2 runs only those subsets (2: z alone); TAG=_x appends _x to the output name.
MIXTURE=2 runs the filters with the Gaussian-mixture analysis of up to two components (`mixture` of
romda.estimators, branch gaussian-mixture-filter), into _mixture2.json; the paper runs it from z alone.
Every filter run records n_escaped, the members that leave the attractor: |u| above ESCAPE times the largest |u| of the
noisy training data (detectable without the true state), as in Fig. 2.
From z alone (SYMMETRIC), (x, y, z) and its mirror state (-x, -y, z) give the same data, so each run is also scored up
to the mirror state (`mirrored`): error_sym, valid_sym, spread_sym and n_mirrored.

    MIXTURE=2 SUBSETS=2 TAG=_z M=10 python run_partial_table.py 100   -> partial_table_lorenz63_m10_mixture2_z.json

Fig. 2 (fig:l63_forecast, plot_lorenz63_phase.py): the three deployments of Lorenz-63 from y alone and from z alone
on test window PHASE_WINDOW (`deploy_window`), the ensemble drawn (seed 0, m = PHASE_M) and N_ENSEMBLES independent
ensembles, whose normalized error and spread the figure draws as medians:

    python run_partial_table.py phase   -> results/partial/lorenz63_phase.npz
"""

import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

from pathlib import Path

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # scripts/esnda: results/
CONFIGS = Path(__file__).resolve().parents[1] / 'configs'   # the configurations of the experiments

import numpy as np  # noqa: E402
from romda.run import innovation_options, load_config  # noqa: E402
from romda import data_assimilation as deployments  # noqa: E402  (its state_metrics is wrapped below)
from romda.data_assimilation import run_and_score  # noqa: E402
from romda.models.data_driven.rnns import model_esn  # noqa: E402
from romda.metrics import VALID_THRESHOLD, error_scale, interpolate  # noqa: E402
from romda.observations import Observations  # noqa: E402
from romda.models.physical import Lorenz63  # noqa: E402

from run_synchronization import test_window_time  # noqa: E402

DEFAULTS = load_config(CONFIGS / 'lorenz63.yml')   # truth, ESN, DA and window of Table 1
#: `est_wout`, `start_param` and `twin` are options of the runs, not `da` keys of the yml
DEFAULT_PARAMS = {'est_wout': False, 'start_param': 0, 'twin': False, **DEFAULTS['da']}
WINDOW_LYAP = DEFAULTS['window_lyap']
#: 'none' is one partial-input network; 'none_ensemble' is m of them (the same members as the filtered cases,
#: substitution instead of analysis, `deploy(open_loop_ensemble=)`); 'input'/'reservoir' are the filtered
#: deployments, differing in what the analysis updates.
CASE_KWARGS = dict(none=dict(bayesian_update=False),
                   none_ensemble=dict(bayesian_update=False, open_loop_ensemble=True),
                   input=dict(bayesian_update=True, update_reservoir=False),
                   reservoir=dict(bayesian_update=True, update_reservoir=True))
MEASURED_SUBSETS = ['0', '1', '2', '01', '02', '12', '012']
MIXTURE = int(os.environ.get('MIXTURE', 1))   # components of the Gaussian-mixture analysis (1: one Gaussian)
SYMMETRIC = ('2',)         # subsets whose data the mirror symmetry S leaves unchanged: scored also up to S
S = np.array([-1., -1., 1.])   # the mirror symmetry of Lorenz-63, (x, y, z) -> (-x, -y, z)
# twice, as the caption of Fig. 2; the appendices count three times (run_parametric.ESCAPE)
ESCAPE = 2.0               # a member leaves the attractor once |u| exceeds ESCAPE times the largest |u| of the training data
CASES = {'none': 'Open-loop substitution', 'none_ensemble': 'Open-loop substitution, ensemble',
         'input': 'Filter, input only', 'reservoir': 'Filter, input and reservoir'}
# none_ensemble: m networks with the filter's starting states, the data substituted into each (its spread for Table 2)


def mirror(U, y_ref, last):
    """The members of U (Nt, 3, m) that are closer to the mirror of the reference y_ref (Nt, 3), S y_ref, than to
    y_ref over the steps `last`, replaced by their mirror, and the mask of those members (nan, escaped: kept)."""
    d_id = np.sqrt(((U[last] - y_ref[last][..., None]) ** 2).mean(axis=(0, 1)))
    d_mi = np.sqrt(((U[last] - (S * y_ref[last])[..., None]) ** 2).mean(axis=(0, 1)))
    flip = d_mi < d_id
    return U * np.where(flip, S[:, None], 1.)[None], flip


def mirrored(result, truth, t_true, y_true, t_lyap, **_):   # _: the other keywords of state_metrics (ddof)
    """`result` with every member (or the one trajectory of the partial-input network) that is closer to the
    mirror of the true state, S u, than to u over the last Lyapunov time of the data replaced by its mirror, and
    the number of members replaced. `state_metrics` of the result scores the deployment up to the symmetry."""
    t = np.asarray(result['t'])
    ens = result.get('ensemble')
    U = ens.model.hist[:, :ens.model.N_dim, :] if ens is not None else result.get('y_members')
    U = np.asarray(result['y'])[..., None] if U is None else np.asarray(U)        # (Nt, 3, m)
    y_ref = interpolate(t_true, y_true, t)
    t_obs = truth.t_obs
    last = (t >= t_obs[0]) & (t <= t_obs[-1]) & (t > t_obs[-1] - t_lyap)
    A, flip = mirror(U, y_ref, last)
    out = dict(t=t, y=A.mean(-1), ensemble=None)
    if ens is not None or result.get('y_members') is not None:
        out['y_members'] = A
    return out, int(flip.sum())


@lru_cache(maxsize=None)
def network(measured, bayesian_update):
    return model_esn(Lorenz63, DEFAULTS['truth'], DEFAULTS['forecast']['esn'], measured=[int(c) for c in measured],
                     bayesian_update=bayesian_update)


def one(job):
    measured, case, i = job
    kw, p = CASE_KWARGS[case], dict(DEFAULT_PARAMS)    # the paper: innovation-based
    esn, data = network(measured, kw['bayesian_update'])
    if MIXTURE > 1 and kw['bayesian_update']:                     # the mixture is fitted in units of the standard
        kw = {**kw, 'mixture': MIXTURE,                           # deviation of each component of the training data
              'climatology': data['noisy_data'][data['i_train']:data['i_test']].std(axis=0)}
    if 'M' in os.environ:
        p['m'] = int(os.environ['M'])
    esn.seed = i                                                  # the starting reservoir states
    t0, t1, extra = WINDOW_LYAP
    record = (data['clean_data'].shape[0] - data['i_test']) * data['dt'] / data['t_lyap']
    off = np.random.default_rng(i).uniform(0, record - t1 - extra - 0.5)    # the start of the window
    captured = {}
    score = deployments.state_metrics

    def capture(result, *a, **k):                                 # the ensemble, for the members that escape
        captured['ensemble'] = result['ensemble']
        if measured in SYMMETRIC:                                 # the scores up to the mirror state
            sym, captured['n_mirrored'] = mirrored(result, *a, **k)
            captured['sym'] = score(sym, *a, **k)
        return score(result, *a, **k)

    deployments.state_metrics = capture
    try:
        r = run_and_score(esn, data, [int(c) for c in measured], 0, p['da_method'], p['m'], p['Nt_obs'], p['std_obs'],
                          p['inflation'], window_lyap=(t0 + off, t1 + off, extra), **innovation_options(p),
                          cdd_fraction=float(p['cdd_fraction']),
                          adaptive_inflation=p['adaptive_inflation'], inflate_parameters=p['inflate_parameters'],
                          start_param=p['start_param'], **kw)
    finally:
        deployments.state_metrics = score
    n_escaped = 0
    if captured.get('ensemble') is not None:
        U = np.asarray(captured['ensemble'].model.hist)[:, :3]          # (Nt, 3, m)
        lim = ESCAPE * np.abs(data['noisy_data'][data['i_train']:data['i_test']]).max()
        n_escaped = int((~(np.abs(U) <= lim)).any(axis=(0, 1)).sum())   # nan: escaped
    row = dict(n_escaped=n_escaped, measured=measured, case=case, init=i, offset_lyap=off, error=float(r['rmse_mean_last']),
               valid=float(r['valid_time_mean_forecast']), spread=float(r.get('spread_last', np.nan)),
               n_rejected=r.get('n_rejected', 0), mixture=MIXTURE if kw['bayesian_update'] else 1)
    if 'sym' in captured:
        q = captured['sym']
        row.update(error_sym=float(q['rmse_mean_last']), valid_sym=float(q['valid_time_mean_forecast']),
                   spread_sym=float(q.get('spread_last', np.nan)), n_mirrored=captured['n_mirrored'])
    return row


def table(rows):
    """LaTeX rows: diverged runs (%), and the medians of the error and the valid time over the other runs."""
    def cell(measured, case, key):
        rs = [r for r in rows if r['measured'] == measured and r['case'] == case]
        ok = [r for r in rs if r['error'] <= VALID_THRESHOLD]
        if key == 'diverged':
            return f'${100 * (len(rs) - len(ok)) / len(rs):.0f}$'
        return f'${np.median([r[key] for r in ok]):.2g}$' if ok else '--'
    for key, head in (('diverged', 'Failed runs (\\%)'), ('error', 'Error at the end of the data, $e_W$'),
                      ('valid', 'Valid time ($T_\\lambda$)')):
        print(f'    \\multicolumn{{8}}{{l}}{{\\emph{{{head}}}}} \\\\')
        for case, name in CASES.items():
            print(f'    {name} & ' + ' & '.join(cell(m, case, key) for m in MEASURED_SUBSETS) + ' \\\\')


def paper_table(runs, members=(5, 10, 20, 30)):
    """The body of tab:partial: failed runs (%), valid time and spread--error ratio for partial inputs (case none;
    its spread from none_ensemble, pooled over the member counts) and the two filters at each member count.
    `runs[m]`: the records of partial_table_lorenz63_m<m>.json. Bold: the best per subset (fewest failures,
    longest valid time, ratio closest to one, partial inputs aside)."""
    def ok(rs):
        return [r for r in rs if r['error'] <= VALID_THRESHOLD]

    def pick(m, case, sub):
        return [r for r in runs[m] if r['case'] == case and r['measured'] == sub]

    def fmt(x, key):
        if x is None:
            return 'failed' if key == 'ratio' else '--'
        if key == 'failed':
            return f'{x:.0f}'
        if key == 'ratio' and x < 1e-2:
            e = int(np.floor(np.log10(x)))
            return f'{x / 10 ** e:.0f}{{\\cdot}}10^{{{e}}}'
        return f'{x:#.2g}'.rstrip('.')                              # two significant figures, trailing zeros kept

    def value(rs, key):
        if key == 'failed':
            return 100 * (len(rs) - len(ok(rs))) / len(rs)
        good = ok(rs)
        if not good:
            return None
        if key == 'valid':
            return float(np.median([r['valid'] for r in good]))
        return float(np.median([r['spread'] / r['error'] for r in good]))

    lines = []
    for key, head in (('failed', r'Failed\\runs (\%)'), ('valid', r'Valid time,\\$t_\mathrm{v}$ ($T_\lambda$)'),
                      ('ratio', r'Spread--error\\ratio')):
        cells = {('none', None): [value(pick(members[0], 'none', s), key) if key != 'ratio' else
                                  value(sum((pick(m, 'none_ensemble', s) for m in members), []), key)
                                  for s in MEASURED_SUBSETS]}
        for case in ('input', 'reservoir'):
            for m in members:
                cells[case, m] = [value(pick(m, case, s), key) for s in MEASURED_SUBSETS]
        best = []
        for j in range(len(MEASURED_SUBSETS)):
            col = {k: float(fmt(v[j], key).split('{')[0]) for k, v in cells.items()        # on the printed values
                   if v[j] is not None and not (key == 'ratio' and k[0] == 'none') and '{' not in fmt(v[j], key)}
            if min(value(pick(members[0], 'none', MEASURED_SUBSETS[j]), 'failed'),
                   *(value(pick(m, c, MEASURED_SUBSETS[j]), 'failed') for c in ('input', 'reservoir') for m in members)) >= 50:
                col = {}                                                 # every deployment fails: no best
            score = {'failed': lambda v: v, 'valid': lambda v: -v, 'ratio': lambda v: abs(np.log(v))}[key]
            top = min(score(v) for v in col.values()) if col else None
            best.append({k for k, v in col.items() if score(v) == top})

        def row(k):
            return ' & '.join((f'$\\bm{{{fmt(v, key)}}}$' if k in best[j] else f'${fmt(v, key)}$') if v is not None
                              else fmt(v, key) for j, v in enumerate(cells[k]))
        lines.append(f'    \\multirow[t]{{9}}{{*}}{{\\makecell[lt]{{{head}}}}} & Partial inputs &  & {row(("none", None))} \\\\')
        for case, name in (('input', 'Filter, input only'), ('reservoir', 'Filter, input and reservoir')):
            lines.append('    \\cmidrule(l){2-10}')
            for i, m in enumerate(members):
                lines.append(f'     & {name if i == 0 else ""} & ${m}$ & {row((case, m))} \\\\')
        lines[-1] += '\\midrule' if key != 'ratio' else ''
    return '\n'.join(lines)


# ── Fig. 2: the deployments drawn ───────────────────────────────────────────────────────────────────────────────────

PHASE_CASES = ([1], [2])      # measured component: y alone, z alone
PHASE_MODES = ('none', 'input', 'reservoir')   # partial inputs (m copies, substitution), the filter correcting the
# input only, and the filter correcting the input and the reservoir (the paper's)
PHASE_M = 10                  # members of each ensemble
ASSIMILATION_LYAP = 6.0       # the data: 6 T_lambda of the measured component
PHASE_WINDOW = 3              # the test window shown (13 Nov 2026, retrained L63): of windows 0-9, the one where the
                              # filter that corrects the reservoir wins from y and from z alike
PHASE_MIXTURE = 2             # components of the mixture analysis from a SYMMETRIC subset (one Gaussian otherwise)
N_ENSEMBLES = 20              # independent ensembles on the window shown: the error and the spread get a band
FORECAST_LYAP = 6.0           # closed loop after the last observation


def deploy_window(mode, dims, seed=0, m=PHASE_M, w=PHASE_WINDOW):
    """Deployment `mode` (PHASE_MODES) of the network with `dims` measured on test window w, with the data of Table 1
    (every Nt_obs steps, sigma_obs); the clock starts one observation interval before the first datum. `seed` draws
    the training reservoir states from which the members start. From z alone, every member closer to the mirror of
    the true state than to the true state over the last Lyapunov time of the data is mirrored in U (`mirror`; U_raw
    keeps it); a member leaves (`out`) once |u| exceeds ESCAPE times the largest |u| of the training data."""
    subset = ''.join(map(str, dims))
    esn, data = network(subset, mode != 'none')
    esn.seed = seed                                                     # read by initialize_from_val_data
    da, t_lyap = DEFAULTS['da'], data['t_lyap']
    noisy, clean, i_test = data['noisy_data'], data['clean_data'], data['i_test']
    t_rec = data['t'][i_test:] - data['t'][i_test]
    t_start = test_window_time(w, t_lyap)
    sh = t_start - da['Nt_obs'] * (t_rec[1] - t_rec[0])
    obs = Observations(y_raw=noisy[i_test:][:, dims], y_true=clean[i_test:][:, dims], t_true=t_rec - sh,
                       t_start=t_start - sh, t_stop=t_start + ASSIMILATION_LYAP * t_lyap - sh, Nt_obs=da['Nt_obs'])
    if mode == 'none':
        res = deployments.deploy(esn, obs, measured=dims, m=m, bayesian_update=False, open_loop_ensemble=True,
                                 t_extra=FORECAST_LYAP * t_lyap)        # the filter's members, substitution only
        t, U = np.asarray(res['t']), np.asarray(res['y_members'])
    else:
        mixture = PHASE_MIXTURE if subset in SYMMETRIC else 1
        res = deployments.deploy(esn, obs, measured=dims, m=m, da_method=da['da_method'], std_obs=da['std_obs'],
                                 t_extra=FORECAST_LYAP * t_lyap, inflation=da['inflation'], **innovation_options(da),
                                 cdd_fraction=da['cdd_fraction'], update_reservoir=mode == 'reservoir',
                                 **({'mixture': mixture, 'climatology': noisy[data['i_train']:i_test].std(axis=0)}
                                    if mixture > 1 else {}))            # fitted in units of the std of the training data
        t, U = np.asarray(res['ensemble'].model.hist_t), np.asarray(res['ensemble'].model.hist)[:, :3]
    truth = np.array([np.interp(t, t_rec - sh, clean[i_test:, i]) for i in range(3)]).T
    t_obs = np.asarray(obs.t_obs)
    U_raw = U
    if subset in SYMMETRIC:                                             # z alone: up to the mirror state
        U = mirror(U, truth, (t > t_obs[-1] - t_lyap) & (t <= t_obs[-1]))[0]
    lim = ESCAPE * np.abs(noisy[data['i_train']:i_test]).max()          # needs no true state
    out = np.maximum.accumulate((np.abs(U) > lim).any(axis=1), axis=0)  # (Nt, m): the member has left
    return dict(t=t / t_lyap, U=U, U_raw=U_raw, truth=truth, t_obs=t_obs / t_lyap, d=np.asarray(obs.y_obs),
                t_last=t_obs[-1] / t_lyap, norm=error_scale(clean[i_test:]), out=out, bad=out[-1],
                m=int((~out[-1]).sum()), ensemble=res.get('ensemble'))


def phase_summary(r):
    """Ensemble mean and standard deviation, and the normalized error of the mean and spread in time, over the
    members of `deploy_window` that stay on the attractor (nan when none does)."""
    U = r['U'][..., ~r['bad']]
    if not r['m']:
        nan = np.full(r['truth'].shape, np.nan)
        return nan, nan, nan[:, 0], nan[:, 0]
    mean = U.mean(-1)
    std = U.std(-1, ddof=1) if r['m'] > 1 else np.zeros_like(mean)
    err = np.linalg.norm(mean - r['truth'], axis=1) / r['norm']
    return mean, std, err, np.sqrt(np.sum(std ** 2, axis=1)) / r['norm']


def phase():
    """The record of Fig. 2: per case (y, z) and deployment, the ensemble drawn (its members, raw and mirrored, the
    true state, the data, the members that leave, the mean and the standard deviation), the error, the spread and the
    fraction of diverged members of N_ENSEMBLES ensembles, and the attractor of the training record."""
    out = {}
    for dims in PHASE_CASES:
        c = 'xyz'[dims[0]]
        for mode in PHASE_MODES:
            r = deploy_window(mode, dims)
            out.update({f'{c}_{mode}_{k}': v for k, v in r.items() if k not in ('ensemble', 'norm', 'm')})
            out[f'{c}_{mode}_mean'], out[f'{c}_{mode}_std'] = phase_summary(r)[:2]
            reps = [deploy_window(mode, dims, seed=k) for k in range(N_ENSEMBLES)]
            out[f'{c}_{mode}_error'], out[f'{c}_{mode}_spread'] = (np.array([phase_summary(q)[i] for q in reps])
                                                                   for i in (2, 3))
            out[f'{c}_{mode}_diverged'] = np.array([q['bad'].mean() for q in reps])
        data = network(''.join(map(str, dims)), True)[1]
        out[f'{c}_attractor'] = data['clean_data'][data['i_train']:data['i_test']]
    path = os.path.join(HERE, 'results', 'partial', 'lorenz63_phase.npz')
    np.savez(path, **out)
    print('wrote', path)


if __name__ == '__main__' and sys.argv[1:2] == ['phase']:
    phase()
    sys.exit()

if __name__ == '__main__' and sys.argv[1:2] == ['paper']:      # the body of tab:partial from the M=5,10,20,30 runs
    folder = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, 'results', 'partial')
    print(paper_table({m: json.load(open(os.path.join(folder, f'partial_table_lorenz63_m{m}.json'))) for m in (5, 10, 20, 30)}))
    sys.exit()

if __name__ == '__main__' and sys.argv[1:2] == ['table']:      # reprint the rows from the stored runs
    table(json.load(open(os.path.join(HERE, 'results', 'partial', 'partial_table_lorenz63' + (f"_m{os.environ['M']}" if 'M' in os.environ else '') + '.json'))))
elif __name__ == '__main__':
    n_init = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    subsets = os.environ['SUBSETS'].split(',') if 'SUBSETS' in os.environ else MEASURED_SUBSETS
    jobs = [(m, c, i) for m in subsets for c in CASES for i in range(n_init)]
    # N_WORKERS: explicit, small by default; a worker of the L96-40 sweeps can hold tens of GB
    with ProcessPoolExecutor(int(os.environ.get('N_WORKERS', 2)), mp_context=multiprocessing.get_context('fork')) as ex:
        rows = list(ex.map(one, jobs, chunksize=10))
    with open(os.path.join(HERE, 'results', 'partial', 'partial_table_lorenz63' + (f"_m{os.environ['M']}" if 'M' in os.environ else '') + (f'_mixture{MIXTURE}' if MIXTURE > 1 else '') + os.environ.get('TAG', '') + '.json'), 'w') as f:
        json.dump(rows, f)
    if "SUBSETS" not in os.environ:   # the LaTeX rows need every subset
        table(rows)
