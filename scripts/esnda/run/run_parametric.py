"""The parametric ESNs of the washout-free paper on the ten-site Lorenz-96 system: Sect. 5.2 (fig:bifurcation, fig:param,
fig:kld) and Appendix B (the forcing F in the input or in the readout).

The networks are parametric ESN_models of configs/lorenz96_parametric.yml (romda.models.data_driven.rnns.parametric_esn: one noisy record
per training forcing, VALUES), kept in the ESN store. VARIANTS sets where F enters:

    readout      : F in the readout only (param_in_reservoir=False, readout_input='params'), tikh 1e-8 (the paper)
    paper        : F an input of every unit, sigma_param searched (Appendix B)
    readout_t12, paper_t12 : the same with tikh 1e-12 (Appendix B)

The single ESN (the parametric L96-10s) has one reservoir of N_UNITS = 1000 units; with `parallel`, the network has the
layout of L96-10p (two patches of five sites, halo of two, 500 units each). The filter (romda.estimators.EnsembleFilter,
type='sqrt', global, the innovation-based inflation of the paper on u and r, not on F) runs on [u; r; F], in which F has
no dynamics. The members start from washouts on random windows of random validation records with F drawn uniformly over
the trained range, independent of the window (start 'decoupled'), the input-F networks from washouts on the training
records at their own forcing ('matched').

    python run_parametric.py truth                     -> results/parameter/lorenz96_bifurcation_Nx10.npz (the true
        system of fig:bifurcation: every F integrated at once from the same initial condition, the local extrema of
        x_0 - F, x_1 - F, x_2 - F and of E = 1/2 sum_i (x_i - F)^2 over 60 T_lambda after a transient of 60 T_lambda)
    python run_parametric.py train [VARIANT] [parallel] -> the network, its gate, valid times and dU/dF:
        results/readout_parameter/single1000/train_<VARIANT>.npz (parallel: readout_parameter/parallel_<VARIANT>.npz)
    python run_parametric.py bifurcation [VARIANT] [parallel] -> results/readout_parameter/bifurcation_single1000_<VARIANT>.npz
        (parallel: bifurcation_parallel_<VARIANT>.npz): 165 forcings x 5 starts x 3 kinds of washout, 400 time units;
        and the local extrema of every loop in bifurcation_<...>_peaks.npz (fig:bifurcation)
    [FIX_F=1] [CDD_STD=0.02 TAG=_cdd002] DATA_LYAP=30 python run_parametric.py scan
        -> results/parameter/param_scan_lorenz96p_gammainnovation[_fixF][TAG].npz (fig:param, fig:kld; plot_parameter_scan.py)
    [FIX_F=1] DATA_LYAP=30 python run_parametric.py attractor -> results/parameter/param_attractor_lorenz96p[_fixF][TAG].npz
        and, without FIX_F, results/parameter/param_kld_perfect_lorenz96p.npz (the perfect model of fig:kld, kld_floor)
    DATA_LYAP=30 python run_parametric.py one_record   -> results/readout_parameter/single1000/one_record.npz: every member
        washed out on the training record farthest from the true F, against the decoupled start (F = 6.5, 7.5, 8.5)
    [START=matched|decoupled] DATA_LYAP=30 python run_parametric.py appscan VARIANT
        -> results/readout_parameter/single1000/scan_<VARIANT>_<START>.npz (Appendix B)

N_WORKERS sets the workers of the scans (default 2; a worker holds a few GB).
"""

import gc
import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import numpy as np
from scipy.signal import find_peaks

from romda import estimators as da  # noqa: E402
from romda.metrics import attractor_kld, error_scale, valid_time  # noqa: E402
from romda.observations import std_fraction_covariance  # noqa: E402
from romda.data_assimilation import filter_record  # noqa: E402
from romda.run import innovation_options, load_config, resolve_model  # noqa: E402
from romda.models.data_driven.rnns import network_summary, parametric_esn  # noqa: E402
from romda.utils import collected, trajectory  # noqa: E402
from romda.models.physical import Lorenz96  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # scripts/esnda: results/
CONFIG = os.path.join(HERE, 'configs', 'lorenz96_parametric.yml')
TRUTH = os.path.join(HERE, 'results', 'parameter', 'lorenz96_bifurcation_Nx10.npz')
OUT = os.path.join(HERE, 'results', 'readout_parameter')
N_UNITS = 1000                                   # the single ESN: the units of the two reservoirs of L96-10p together
OUT_SINGLE = os.path.join(OUT, f'single{N_UNITS}')
NX, LAYOUT = 10, dict(N_units=500, patch_size=5, halo=2, periodic=True,   # L96-10p
                      rho_range=[0.2, 1.2], sigma_in_range=[-1.5, 1.0],   # as configs/lorenz96_parallel.yml
                      tikh_range=[1e-8],   # fixed (2026-11-12, noisy records, 20 windows at F = 6.5/7/8, 40 loops of 400 time
                      # units from F = 0.5 to 10): 1e-8 forecasts for 4.00/3.88/2.21 T_lambda, its readout moves by 1.95 per
                      # unit of F at a fixed reservoir state, and no loop leaves the attractor from a washout at F or from the
                      # state of another regime. The grid down to 1e-10 picked 1e-10: a sensitivity of 12.6 to 21, and 24 to
                      # 32 of the 40 loops from the state of another regime leave in the first steps; 1e-6 and 1e-4 lose
                      # 0.5 to 2 T_lambda of valid time and 6 and 10 of the 40 loops from a washout leave
                      noise=0.0,   # no added input noise: the training segments carry the 2% noise of truth.noise_level; with
                      # 0.01 on top, the network forecasts for less long in 12 of 60 windows at F = 6.5, 7 and 8, and longer
                      # in 30 (Wilcoxon p = 0.018, 2026-11-11)
                      stability_steps=4200, stability_starts=39, stability_penalty=10.0,   # the gate, 100 T_lambda,
                      N_grid=5, N_func_evals=50,                     # from 3 starts per training forcing (one start
                      # per forcing, 50 T_lambda, passed a network whose loops leave the attractor from F = 6.25,
                      # half of them after 50 T_lambda); a 3 x 3 x 3 grid
                      optimize_parameter_normalization='data',       # and 23 Bayesian points. F normalized like the
                      hyperparameters_to_optimize=['rho', 'sigma_in', 'sigma_param', 'tikh'],   # state, with an input
                      sigma_param_range=[-2, 1.0])                   # scaling of its own (as the former sigma_F)
_READOUT = dict(param_in_reservoir=False, readout_input='params', hyperparameters_to_optimize=['rho', 'sigma_in', 'tikh'])
VARIANTS = dict(paper={}, readout=_READOUT, paper_t12=dict(tikh_range=[1e-12]), readout_t12=dict(_READOUT, tikh_range=[1e-12]))
F_TRUE = [3.0, 3.6, 4.3, 5.5, 6.5, 7.5, 8.5]      # the test forcings: unseen; 3.0 and 8.5 outside the trained range
_CFG = load_config(CONFIG)
VALUES = list(_CFG['forecast']['train']['values'])   # the training forcings
T_LYAP = resolve_model(_CFG['truth']['model'])(**_CFG['truth']['params']).t_lyap   # ten sites, F = 8
DT = _CFG['forecast']['train']['dt'] * _CFG['forecast']['esn']['upsample']          # network step
STEPS_LYAP = int(round(T_LYAP / DT))              # 42
NOISE = _CFG['truth']['noise_level']              # noise of the records, relative to their standard deviation
INTERVALS = [0.024, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]   # observation intervals, T_lambda (0.024: every network step)
#: data from T0 to T1 T_lambda, then T_FREE of closed loop; DATA_LYAP=30 gives every interval up to 3 T_lambda at least
#: 10 analyses (the paper)
T0, T_FREE = 2.0, 30
T1 = T0 + float(os.environ.get('DATA_LYAP', 20))
M = 64
FIX_F = os.environ.get('FIX_F') == '1'            # FIX_F=1: F held at the middle of the trained range, not estimated
CDD_STD = float(os.environ.get('CDD_STD', _CFG['da']['cdd_fraction']))   # C_dd: this fraction of the std of each site
LIMS = (2.0, 9.0)                                 # F of every member kept in this range
SEEDS = [0, 1, 2, 3, 4]
MISMATCH_FORCINGS = np.round(np.arange(0.5, 10.01, 0.5), 2)   # the forcings of the gate
ESCAPE = 3.0   # a loop leaves the attractor once |u| exceeds ESCAPE times the largest |u| of the record (Appendix B)


# ── the networks ─────────────────────────────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def network(variant='readout', parallel=False):
    """(net, regimes, segments, error scale of the training records) of `variant`, trained or loaded from the store."""
    cfg = load_config(CONFIG)
    model = resolve_model(cfg['truth']['model'])
    esn = {**cfg['forecast']['esn'], **LAYOUT, **VARIANTS[variant]}
    if not parallel:                                                 # one reservoir
        esn['N_units'] = N_UNITS
        for key in ('patch_size', 'halo', 'periodic'):
            esn.pop(key)
    if VARIANTS[variant].get('readout_input'):                       # no parameter columns in W_in
        esn.pop('sigma_param_range')
    net, segments, regimes = parametric_esn(model, cfg['forecast']['train'], esn, cfg['truth']['noise_level'],
                                            model_kwargs=dict(cfg['truth']['params']), plot_training=False)
    assert int(round(T_LYAP / net.dt_ESN)) == STEPS_LYAP
    return net, regimes, segments, error_scale(segments.reshape(-1, NX))


def record(F, seed, n):
    """Clean and noisy record of n network steps at the forcing F (noise of the truth block of CONFIG)."""
    up, dt = _CFG['forecast']['esn']['upsample'], _CFG['forecast']['train']['dt']   # those of every network
    X = trajectory(resolve_model(_CFG['truth']['model']), seed=seed, n_steps=n * up, dt=dt,
                   constructor_defaults=dict(_CFG['truth']['params']), F=F)[::up]
    return X, X + NOISE * X.std() * np.random.default_rng(seed).standard_normal(X.shape)


def gate(variant, parallel=False):
    """The forcings whose closed loop leaves |u| <= 3 max|u| of the records within 50 steps from the washouts on two
    training records with the input switched to each forcing (a state of one regime with the input of another)."""
    net, regimes = network(variant, parallel)[:2]
    psi, _ = net.regime_matched_states(2, regimes, seed=0)
    F = np.repeat(MISMATCH_FORCINGS, 2)
    r = np.tile(psi[NX:NX + net.N_r], len(MISMATCH_FORCINGS))
    u = np.tile(psi[:NX], len(MISMATCH_FORCINGS))
    bound = ESCAPE * np.max(np.abs(np.ravel(net.shift)[:NX]) + np.abs(np.ravel(net.norm)[:NX]))
    left = ~(np.abs(net.forecast(u, r, 50, input_parameters=F[None])[0]) <= bound).all(axis=(0, 1))   # any step, any site
    return sorted({float(f) for f in F[left]})


def valid_times(variant, F, n_windows=10, horizon=10, parallel=False):
    """Valid times (T_lambda) of the closed loop from a washout on a fresh noisy record at the forcing F."""
    net, _, _, norm = network(variant, parallel)
    n_loop, gap = horizon * STEPS_LYAP, net.N_wash + horizon * STEPS_LYAP
    X, Y = record(F, 2000, n_windows * gap + 1)
    out = []
    for k in range(n_windows):
        s = k * gap
        U, R = net.open_loop(Y[s:s + net.N_wash], input_parameters=F)
        pred = np.vstack([U[-1:], net.forecast(U[-1], R[-1], n_loop - 1, input_parameters=F)[0][..., 0]])
        err = np.linalg.norm(pred - X[s + net.N_wash:s + gap], axis=1) / norm
        out.append(valid_time(err, np.arange(len(err)) / STEPS_LYAP, t_max=horizon))
    return np.array(out)


def sensitivity(variant, dF=0.01, parallel=False):
    """RMS change per site of the closed-loop output per unit of F after 1 step, 0.3 and 1 T_lambda from the same
    [u; r] (washouts on the training records, two per record), F switched by dF. (len(horizons), members)."""
    net, regimes = network(variant, parallel)[:2]
    horizons = (1, int(round(0.3 * STEPS_LYAP)), STEPS_LYAP)
    psi, F = net.regime_matched_states(2 * regimes.shape[1], regimes, seed=0)
    u0, r0 = psi[:NX], psi[NX:NX + net.N_r]
    runs = [net.forecast(u0, r0, max(horizons), input_parameters=F + d)[0][np.array(horizons) - 1] for d in (0.0, dF)]
    return np.sqrt(np.mean((runs[1] - runs[0]) ** 2, axis=1)) / dF


def train(variant='readout', parallel=False):
    """Train (or load) the network and print its gate, valid times and dU/dF."""
    net = network(variant, parallel)[0]
    path = (os.path.join(OUT, f'parallel_{variant}.npz') if parallel
            else os.path.join(OUT_SINGLE, f'train_{variant}.npz'))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f'--- {variant}:', network_summary(net), f'| F shift/scale {np.ravel(net.shift)[-1]:.3g}/'
          f'{np.ravel(net.norm)[-1]:.3g}', flush=True)
    left = gate(variant, parallel)
    print(f'{variant}: gate (state of another regime, input switched to F): loops leave at F =', left or 'none',
          flush=True)
    vt = {F: valid_times(variant, F, parallel=parallel) for F in VALUES + F_TRUE}
    for F, t in sorted(vt.items()):
        print(f'{variant} F={F:5.2f} valid time median {np.median(t):5.2f} (p25 {np.quantile(t, .25):.2f}, '
              f'p75 {np.quantile(t, .75):.2f}) T_lambda', flush=True)
    sens = sensitivity(variant, parallel=parallel)
    print(f'{variant} dU/dF (RMS per site, median over 26 starts) after 1 step / 0.3 / 1 T_lambda: '
          + ' / '.join(f'{np.median(s):.3g}' for s in sens), flush=True)
    hp = {k: float(np.ravel(getattr(net, k))[0]) for k in ('rho', 'sigma_in', 'tikh')}
    np.savez(path, gate=np.array(left), sens=sens, hp=np.array(hp, dtype=object),
             valid=np.array({str(k): v for k, v in vt.items()}, dtype=object))


# ── bifurcation diagrams ─────────────────────────────────────────────────────────────────────────────────────────────

STARTS = ('nearest', 'random', 'truth')
START_N = 5
WASH = int(round(2.0 * STEPS_LYAP))                       # 2 T_lambda
N_LOOP = int(round(400 / DT))
N_KEEP = int(round(35 / DT))                              # the last 35 time units
TRANSIENT = 20.0                                          # transient of the truth before its washout (time units)


def truth():
    """The bifurcation diagram of the true system (fig:bifurcation, top)."""
    from scipy.integrate import solve_ivp
    Fmin, Fmax, nF, t_transient_lyap, t_sample_lyap, seed = 0.0, 12.0, 200, 60, 60, 0
    F_vals = np.linspace(Fmin, Fmax, nF)
    t_lyap, dt = Lorenz96.t_lyap, 0.01

    def rhs(t, y):                                                   # every forcing a column of the ring
        return Lorenz96.time_derivative(t, y.reshape(nF, NX).T, NX, F_vals).T.ravel()

    y0 = np.tile(8. + np.random.default_rng(seed).standard_normal(NX), nF)   # same initial condition for every F
    y1 = solve_ivp(rhs, (0., t_transient_lyap * t_lyap), y0).y[:, -1]
    sol = solve_ivp(rhs, (0., t_sample_lyap * t_lyap), y1, t_eval=np.arange(0., t_sample_lyap * t_lyap, dt))
    dev = sol.y.reshape(nF, NX, -1) - F_vals[:, None, None]             # deviation from the equilibrium x_i = F
    series = np.concatenate((dev[:, :3], 0.5 * np.sum(dev ** 2, axis=1, keepdims=True)), axis=1)   # (nF, 4, Nt)
    peaks = {ext: [[series[k, i][find_peaks(s * series[k, i])[0]] for k in range(nF)] for i in range(4)]
             for ext, s in (('max', 1), ('min', -1))}
    os.makedirs(os.path.dirname(TRUTH), exist_ok=True)
    np.savez(TRUTH, F_vals=F_vals, peaks_max=np.array(peaks['max'], dtype=object),
             peaks_min=np.array(peaks['min'], dtype=object), allow_pickle=True)
    print('wrote', TRUTH)


def extrema(X):
    """The local maxima and minima (find_peaks, as `truth`) of every column of X (Nt, n), as two (n,) object arrays;
    a column without any (a fixed point) gives its last value. A top of more than one sample (equal float32 values
    near a fixed point) is not an extremum."""
    out = []
    for sign in (1, -1):
        peaks = np.empty(X.shape[1], dtype=object)
        for j, x in enumerate(X.T):
            i = find_peaks(sign * x, plateau_size=(1, 1))[0]
            peaks[j] = x[i] if i.size else x[-1:]
        out.append(peaks)
    return out


def washout_inputs(net, regimes, F, kind, seed=0):
    """(WASH, NX, n) open-loop inputs and the (n,) forcing the network reads during the washout: on a random window of
    the validation record whose training forcing is nearest F ('nearest'), of a random record ('random'), or on the
    true trajectory at F ('truth')."""
    rng = np.random.default_rng(seed)
    data, train_F = net.validation_data, regimes[0]
    n = len(F)
    if kind == 'truth':
        x = F + 0.1 * np.random.default_rng(1).standard_normal((NX, n))
        for _ in range(int(round(TRANSIENT / net.dt_ESN))):
            x = Lorenz96.rk4_step(x, net.dt_ESN, F, h=0.01)
        U = np.empty((WASH, NX, n))
        for k in range(WASH):
            x = Lorenz96.rk4_step(x, net.dt_ESN, F, h=0.01)
            U[k] = x
        return U, F
    rec = (np.argmin(np.abs(train_F[None] - F[:, None]), axis=1) if kind == 'nearest'
           else rng.integers(0, data.shape[0], n))
    k0 = rng.integers(0, data.shape[1] - WASH, n)
    U = np.stack([data[j, k:k + WASH] for j, k in zip(rec, k0)], axis=-1)
    return U, train_F[rec]


def bifurcation(variant='readout', parallel=False):
    """Every loop held at one forcing F of the true diagram (0 < F <= 10) in closed loop for 400 time units after a
    washout of 2 T_lambda of each kind (STARTS, START_N per forcing); the closed loop reads F, the washout on a record
    reads the record's forcing. u_1 - F over the last 35 time units, and the loops that leave |u| <= 3 max|u| of the
    training records; in the file of the peaks, the local extrema (`extrema`) and the range of u_1 - F of every loop
    over those 35 time units."""
    net, regimes, segments = network(variant, parallel)[:3]
    bound = ESCAPE * np.abs(segments).max()
    F_vals = np.load(TRUTH, allow_pickle=True)['F_vals']
    F_grid = F_vals[(F_vals > 0) & (F_vals <= 10)]
    F = np.repeat(F_grid, START_N)
    out = dict(F_grid=F_grid, bound=bound)
    peaks = dict(F=F)
    name = f'{"parallel" if parallel else f"single{N_UNITS}"}_{variant}'
    for kind in STARTS:
        U, F_wash = washout_inputs(net, regimes, F, kind, seed=STARTS.index(kind))
        Y, R = net.open_loop(U, input_parameters=F_wash)
        u, r = Y[-1], R[-1]
        del Y, R
        umax = np.abs(u).max(axis=0)
        X = np.empty((N_KEEP, len(F)), dtype=np.float32)
        for k in range(N_LOOP):
            u, r = net.closed_loop_step(u, r, input_parameters=F)
            umax = np.maximum(umax, np.nan_to_num(np.abs(u), nan=np.inf).max(axis=0))
            if k >= N_LOOP - N_KEEP:
                X[k - (N_LOOP - N_KEEP)] = u[0] - F
        out[f'X_{kind}'], out[f'left_{kind}'] = X, umax > bound
        peaks[f'max_{kind}'], peaks[f'min_{kind}'] = extrema(X)
        peaks[f'span_{kind}'] = np.ptp(X, axis=0)
        left = sorted(set(np.round(F[umax > bound], 2)))
        print(f'{name} {kind:8s}: {int((umax > bound).sum())}/{len(F)} loops leave the attractor'
              + (f', at F = {left}' if left else ''), flush=True)
        del U, X, r, u
        gc.collect()
    out['F'] = F
    np.savez(os.path.join(OUT, f'bifurcation_{name}.npz'), **out)
    np.savez(os.path.join(OUT, f'bifurcation_{name}_peaks.npz'), **peaks)


# ── state and parameter estimation ───────────────────────────────────────────────────────────────────────────────────

def Filter(**kwargs):
    """The global filter with the innovation-based inflation of the paper on u and r only: F, which has no dynamics,
    would random-walk under it (an error burst of the network inflates every analysis that follows, and data at short
    intervals barely constrain F)."""
    return da.EnsembleFilter(**{'inflation': _CFG['da']['inflation'], **innovation_options(_CFG['da']),
                                'inflate_parameters': False, **kwargs})


def decoupled_start(est, regimes, seed, record=None):
    """Replace the matched start of `est` (member i washed out on record i % L at its forcing) by washouts on random
    windows of random validation records (ESN_model.initialize_from_val_data, seeded by the member copy's seed), or of
    the one `record`, with F drawn uniformly over the trained range, independent of the window. Returns the RMS over
    the reservoir units of corr(F, r_i) across the members (nan with `record`)."""
    model = est.model
    # the washout reads out at the forcing of its record, not at whatever F a previous closed loop left
    phi = model.initialize_from_val_data(N_ens=model.m, segments=None if record is None else [record],
                                         input_parameters=regimes)
    F = np.random.default_rng(10_000 + seed).uniform(regimes.min(), regimes.max(), model.m)
    psi = model.current_state
    psi[:NX + model.N_r] = phi[:NX + model.N_r]
    psi[-1] = F
    if record is not None:
        return np.nan
    dr = phi[NX:NX + model.N_r] - phi[NX:NX + model.N_r].mean(-1, keepdims=True)
    c = (dr @ (F - F.mean())) / (np.linalg.norm(dr, axis=1) * np.linalg.norm(F - F.mean()) + 1e-30)
    return float(np.sqrt(np.mean(c[np.linalg.norm(dr, axis=1) > 0] ** 2)))


@collected
def filter_run(job):
    """One deployment: the filter on [u; r; F] from T0 to T1 every `interval` T_lambda on a fresh record at F_true, then
    T_FREE of closed loop of every member at its own F. start: 'decoupled', 'matched' or 'one_record' (every member
    washed out on the training record farthest from F_true). Returns every score and series of the run."""
    variant, start, fix, F_true, interval, seed = job
    net, regimes, _, norm = network(variant)
    nt_obs = max(1, int(round(interval * STEPS_LYAP)))
    s0, n_obs = int(T0 * STEPS_LYAP), int(round((T1 - T0) * STEPS_LYAP / nt_obs))
    t_last = s0 + n_obs * nt_obs
    X, Y = record(F_true, 1000 + seed, t_last + T_FREE * STEPS_LYAP + 1)   # a fresh truth (seeds after training)
    est = Filter(parent_model=net, m=M, type='sqrt', seed=seed, regimes=regimes, est_alpha=['F'],
                 parameter_estimation='joint', measured=list(range(NX)))
    c0 = np.nan
    if start != 'matched':
        c0 = decoupled_start(est, regimes, seed,
                             record=int(np.argmax(np.abs(regimes[0] - F_true))) if start == 'one_record' else None)
    F_mid = float(np.mean([regimes.min(), regimes.max()]))
    if fix:
        est.model.current_state[-1] = F_mid
    Cdd = std_fraction_covariance(X, CDD_STD, identical=True)   # the sites share statistics
    k = np.arange(1, n_obs + 1)                                      # the analyses

    def bound(ens):                                                  # F of every member kept in the trained range
        psi = ens.model.current_state
        psi[-1] = F_mid if fix else np.clip(psi[-1], *LIMS)
    est, H = filter_record(est, Y[s0 + k * nt_obs], k * nt_obs * net.upsample, Cdd, after_analysis=bound)
    F = H[np.r_[0, k * nt_obs * net.upsample], -1]                   # F of the members at the start and each analysis
    t, mean, std = np.r_[0.0, k * nt_obs / STEPS_LYAP], F.mean(-1), F.std(-1)
    psi = H[-1]
    mu_F, std_F = float(psi[-1].mean()), float(psi[-1].std())
    off = np.abs(np.array(mean) - F_true) > 0.02 * F_true            # converged: within 2% from then on
    t_conv = float(t[np.flatnonzero(off)[-1] + 1]) if not off[-1] and off.any() else (0.0 if not off.any() else np.nan)
    psi = np.array(psi, copy=True)
    del est
    gc.collect()
    free = net.forecast(psi[:NX], psi[NX:NX + net.N_r], T_FREE * STEPS_LYAP, input_parameters=psi[-1:])[0]
    truth_ = X[t_last + 1:t_last + 1 + len(free)]
    err = np.linalg.norm(free.mean(-1) - truth_, axis=1) / norm
    valid = float(valid_time(err, np.arange(len(err)) / STEPS_LYAP, t_max=T_FREE))
    kld = attractor_kld(X[t_last + 1:], free)
    blown = int((~np.isfinite(free) | (np.abs(free) > ESCAPE * np.abs(X).max())).any(axis=(0, 1)).sum())
    print(f'{variant:11s} {start:10s} {"fixF" if fix else "est "} F={F_true:4.1f} dt={interval:5.3f} seed={seed}: '
          f'F {mu_F:5.2f} +/- {std_F:5.3f}, conv {t_conv:5.2f}, valid {valid:5.2f}, KLD {np.median(kld):.3f}, '
          f'blown {blown}', flush=True)
    return dict(variant=variant, start=start, fix=fix, F=F_true, interval=interval, seed=seed, mu=mu_F, std=std_F,
                t_conv=t_conv, valid=valid, kld=kld, blown=blown, corr_Fr0=c0,
                t=np.array(t), mean=np.array(mean), sd=np.array(std), F_est=psi[-1], truth=truth_, free=free,
                n_data=NX * int(round(t_conv * STEPS_LYAP / nt_obs)) if np.isfinite(t_conv) else np.nan)


def paper_row(r):
    """A row of the scan of fig:param (plot_parameter_scan.py)."""
    h = dict(t=r['t'], mean=r['mean'], std=r['sd'], t_last=T1 - T0)
    return dict(rho_true=r['F'], interval=r['interval'], mu=r['mu'], std=r['std'], kld=r['kld'], t_conv=r['t_conv'],
                valid=r['valid'], n_data=r['n_data'], history=h, trace=h)   # history: panel (a); trace: the same (b)


def app_row(r):
    """A row of the scans of Appendix B and of one_record: the scores, the median KLD, no closed loops."""
    return {k: (float(np.median(v)) if k == 'kld' else v) for k, v in r.items() if k not in ('F_est', 'truth', 'free')}


def _pool(fn, jobs):
    with ProcessPoolExecutor(int(os.environ.get('N_WORKERS', 2))) as ex:   # a worker holds a few GB
        return list(ex.map(fn, jobs))


def _paper_run(job):
    return paper_row(filter_run(('readout', 'decoupled', FIX_F, *job)))


def _attractor_run(F):
    r = filter_run(('readout', 'decoupled', FIX_F, F, 0.1, 0))
    return dict(F=F, F_est=r['F_est'], truth=r['truth'], free=r['free'])


def _app_run(job):
    return app_row(filter_run(job))


def scan():
    """fig:param and fig:kld: the parametric L96-10s at the test forcings and the observation intervals, seed 0;
    resumes from its output file, only if that file was written with the same settings (stored with the rows; a
    record without them predates the stamp and is resumed as it is). A complete record is left untouched."""
    out = os.path.join(HERE, 'results', 'parameter', 'param_scan_lorenz96p_gammainnovation' + ('_fixF' if FIX_F else '')
                       + os.environ.get('TAG', '') + '.npz')   # TAG=_cdd002 with CDD_STD=0.02: C_dd at the noise
    settings = json.dumps(dict(cdd_std=CDD_STD, data_lyap=T1 - T0, t_free=T_FREE, m=M, fix_f=FIX_F, n_units=N_UNITS,
                               noise=NOISE), sort_keys=True)
    done = []
    if os.path.exists(out):
        z = np.load(out, allow_pickle=True)
        found = str(z['settings']) if 'settings' in z.files else settings   # unstamped: the paper's records
        if found != settings:
            sys.exit(f'{out} was written with the settings {found}, not {settings}: '
                     'set another TAG (or move the file) instead of resuming it')
        done = [r for r in z['rows'] if 'trace' in r]
    jobs = [j for j in itertools.product(F_TRUE, INTERVALS, [0])
            if not any((r['rho_true'], r['interval']) == j[:2] for r in done)]
    if done and not jobs:
        print('complete:', out)
        return
    rows = done + _pool(_paper_run, jobs)
    for r in rows:                                                   # panel (a): every interval at F = 6.5
        if r['rho_true'] != 6.5:
            r['history'] = None
    np.savez(out, rows=np.array(rows, dtype=object), settings=settings)
    print('wrote', out)


def attractor():
    """The closed loops after the data of the extrapolation cases (data every 0.1 T_lambda), for the attractors, and
    the KLD of the perfect model (kld_floor)."""
    out = _pool(_attractor_run, [min(F_TRUE), max(F_TRUE)])
    path = os.path.join(HERE, 'results', 'parameter', 'param_attractor_lorenz96p' + ('_fixF' if FIX_F else '')
                        + os.environ.get('TAG', '') + '.npz')
    np.savez(path, **{f'{k}_{i}': v for i, o in enumerate(out) for k, v in o.items()})
    print('wrote', path)
    if not FIX_F:                                                    # the perfect model does not depend on F of the ESN
        path = os.path.join(HERE, 'results', 'parameter', 'param_kld_perfect_lorenz96p.npz')
        np.savez(path, F=F_TRUE, floor=kld_floor())
        print('wrote', path)


def kld_floor():
    """The perfect model of fig:kld: per test forcing, the median attractor KLD of Lorenz-96 at the true F over T_FREE,
    from 10 noisy copies of the true state at the last analysis of the scan of each interval (the truth of the scan,
    seed 1000; the noise of the records), over the starts and the intervals, as the violins."""
    n = T_FREE * STEPS_LYAP
    floor = []
    for F in F_TRUE:
        kl = []
        for interval in INTERVALS:
            nt = max(1, int(round(interval * STEPS_LYAP)))
            t_last = int(T0 * STEPS_LYAP) + int(round((T1 - T0) * STEPS_LYAP / nt)) * nt   # as filter_run
            X = record(F, 1000, t_last + 1 + n)[0]
            x = X[t_last][:, None] + NOISE * X.std() * np.random.default_rng(int(round(interval * 1000))).standard_normal((NX, 10))
            free = np.array([x := Lorenz96.rk4_step(x, DT, F=F) for _ in range(n)])   # dynamodels' RK4, one step of DT
            kl.append(attractor_kld(X[t_last + 1:], free))
        floor.append(np.median(np.concatenate(kl)))
    return np.array(floor)


def one_record():
    """The states of the initial ensemble: every member washed out on the record farthest from F, against the
    decoupled start."""
    os.makedirs(OUT_SINGLE, exist_ok=True)
    jobs = [('readout', start, False, F, dt, s)
            for start, F, dt, s in itertools.product(['one_record', 'decoupled'], (6.5, 7.5, 8.5), (0.1, 1.0), range(5))]
    path = os.path.join(OUT_SINGLE, 'one_record.npz')
    np.savez(path, rows=np.array(_pool(_app_run, jobs), dtype=object))
    print('wrote', path)


def appscan(variant):
    """Appendix B: the scan over the test forcings, the intervals and SEEDS, from the start of the variant (decoupled
    for F in the readout, matched for F in the input; START overrides it)."""
    os.makedirs(OUT_SINGLE, exist_ok=True)
    start = os.environ.get('START', 'decoupled' if VARIANTS[variant].get('readout_input') else 'matched')
    jobs = [(variant, start, False, F, dt, s) for F, dt, s in itertools.product(F_TRUE, INTERVALS, SEEDS)]
    path = os.path.join(OUT_SINGLE, f'scan_{variant}_{start}.npz')
    np.savez(path, rows=np.array(_pool(_app_run, jobs), dtype=object))
    print('wrote', path)


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('fork', force=True)
    command, args = sys.argv[1], sys.argv[2:]
    variant = next((a for a in args if a in VARIANTS), 'readout')
    parallel = 'parallel' in args
    if command not in ('truth',):
        network(variant, parallel)                                   # trained before the workers fork
    dict(truth=truth, train=lambda: train(variant, parallel), bifurcation=lambda: bifurcation(variant, parallel),
         scan=scan, attractor=attractor, one_record=one_record, appscan=lambda: appscan(variant))[command]()
