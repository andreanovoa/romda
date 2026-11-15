"""Washout-free synchronization of a parallel echo state network on a 40-site Lorenz-96 ring, with the global
and the localized square-root filter (LETKF) of romda.estimators.

The network has one reservoir per patch of G_SITES neighbouring sites; each reservoir reads its patch and HALO
sites on either side, and predicts its patch (Pathak et al. 2018). The Lorenz-96 equations are invariant under
a shift of the sites, so all patches share the input, reservoir and readout matrices, which are trained on all
patches at once. Every reservoir row of the network state takes the position of its patch, so the LETKF analyses
all the units that a readout reads with one transform. A single reservoir (G_SITES=40 HALO=0) places each unit at
the site of its input (unit_positions); the analysed reservoir is then no longer a combination of the member
states, which a weakly regularized readout amplifies (run_localization.py, run_readout_control.py).

The network is a romda ESN_model with the parallel layout of echostatenetwork (patch_size, halo, periodic), trained
and validated by the package and kept in the ESN store (romda.models.data_driven.rnns.model_esn) on the Lorenz-96 record of
configs/lorenz96_parallel.yml. The filters are romda.estimators.EnsembleFilter(type='sqrt'): the global ensemble
transform and, with `localization`, the local ensemble transform Kalman filter (LETKF), on the state [u; r] of the
network; the members start from washouts on validation windows (ESN_model.init_ensemble).

    python run_parallel_localization.py train   -> the network of the layout, in the ESN store
    python run_parallel_localization.py sync    -> results/synchronization/synchronization_lorenz96_parallel<layout>.npz
    python run_parallel_localization.py run     -> results/localization/localization_parallel_lorenz96.npz (test windows 0-9)
    python run_parallel_localization.py forecast -> results/localization/parallel_forecast_windows_cdd<CDD_STD>.npz (the
        scores of every test window of fig:parallel_forecast, columns SCORE_COLUMNS; with another CDD_STD, for
        fig:uncertainty) and, at the C_dd of the configuration, parallel_forecast_shown.pkl (the deployments drawn)
    python run_parallel_localization.py covariance -> results/localization/localized_covariance.npz (the forecast
        correlations of fig:localized_covariance) and localized_covariance_positions.npz (the reservoir positions)

The layout comes from the environment (G_SITES, HALO, N_R; a non-default layout writes its own files, suffixed
_g<G>_h<H>_n<N>; the ten-site ring always names its layout). The global network is G_SITES=40 HALO=0 N_R=6000: one
reservoir for the 40 sites, and TRAIN_LYAP=800 trains it on twice the data; MEMBERS=10,20,40,80,160 sets the
ensemble sizes; ESN_CONFIG and ESN_STORE replace the config and the store (tests).
"""

import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import numpy as np

from romda import estimators as da  # noqa: E402
from romda.metrics import error_scale, first_below, valid_time  # noqa: E402
from romda.observations import std_fraction_covariance  # noqa: E402
from romda.data_assimilation import filter_record  # noqa: E402
from romda.run import innovation_options, load_config, resolve_model  # noqa: E402
from romda.models.data_driven.rnns import model_esn, network_summary  # noqa: E402
from romda.utils import collected  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # scripts/esnda: configs/, results/

NX = int(os.environ.get('NX', 40))                # sites (NX=10: the ten-site ring)
G_SITES, HALO, N_R = (int(os.environ.get(k, v)) for k, v in (('G_SITES', 5), ('HALO', 2), ('N_R', 500)))
# sites per patch, halo on each side, units per reservoir; G_SITES=NX HALO=0: one reservoir for the whole ring
N_WASH = int(os.environ.get('N_WASH', 100 if G_SITES == NX else 20))   # training washout, network steps: 100 for
# a single reservoir (L96-40s, Table 1 of the paper), 20 for the parallel networks
TRAIN_LYAP = int(os.environ.get('TRAIN_LYAP', 400))   # length of the training record, T_lambda
SUFFIX = ('' if (G_SITES, HALO, N_R, NX) == (5, 2, 500, 40) else f'_g{G_SITES}_h{HALO}_n{N_R}') + \
         ('' if NX == 40 else f'_nx{NX}') + \
         ('' if N_WASH == 100 else f'_w{N_WASH}')   # the paper's networks: _w20 (the names of the former 100 default)
TAG = os.environ.get('TAG', '')                 # appended to the output names (e.g. _infl for an extra grid)
CONFIG = os.environ.get('ESN_CONFIG', os.path.join(HERE, 'configs', 'lorenz96_parallel.yml'))
_DA = load_config(os.path.join(HERE, 'configs', 'lorenz96_parallel.yml'))['da']
INNOVATION = dict(inflation='innovation', **innovation_options(_DA))   # the innovation-based inflation of the paper
CDD_STD = os.environ.get('CDD_STD', str(_DA['cdd_fraction']))   # C_dd, fraction of the standard deviation
SINGLE_TIKH_RANGE = [1e-4]   # L96-40s (one reservoir, 40 sites): the Tikhonov factor is fixed at 1e-4, that of the
# paper. Sect. 4.4: the LETKF analyses each unit of a single reservoir with the data near its input site, so the
# analysed reservoir is no longer a combination of the member states, and the readout maps that departure to the
# output in proportion to its norm. With the grid of CONFIG down to 1e-10, the search picked 1e-10 (rerun of
# 12 Nov 2026, TRAIN_LYAP=800): |Wout| 1.7e6 against 604 at 1e-4 in the paper, and the LETKF failed in 38 of the
# 38 test windows at every ensemble size from 5 to 160, from 40 and from 20 sites
OVERRIDES = {}      # settings of the network that replace those of CONFIG (run_readout_control.py: a fixed tikh)
N_P = NX // G_SITES                       # patches (1: a single reservoir)


def _steps_lyap():
    """Network steps per T_lambda = 1 / lambda_1 of the system of the layout (network()[3]): 30 for 40 sites and 42
    for 10, as Table 1 of the paper. Not the Lyapunov time of the record, N_lyap dt with N_lyap = int(T_lambda / dt),
    which truncates T_lambda of the 40 sites to 0.59 and gives 29 steps."""
    cfg = load_config(CONFIG)
    model = resolve_model(cfg['truth']['model'])(**{**cfg['truth']['params'], 'Nx': NX})
    return int(round(model.t_lyap / (model.dt * cfg['forecast']['esn']['upsample'])))


STEPS_LYAP = _steps_lyap()


def sensor_sites(n):
    """n evenly spaced sites of the ring; = arange(0, NX, NX // n) when n divides NX."""
    return np.unique(np.round(np.arange(n) * NX / n).astype(int))


# ── the network (romda ESN_model, echostatenetwork parallel layout) ──────────

@lru_cache(maxsize=None)
def network():
    """The ESN_model of the layout, trained (or loaded from the store) on the Lorenz-96 record of CONFIG, and the
    record in network steps: X clean, Y noisy (the data), the steps per T_lambda and the error scale. The test
    windows lie in the held-out end of the record, from step `net.held_out` (window_start)."""
    cfg = load_config(CONFIG)
    truth, esn = cfg['truth'], dict(cfg['forecast']['esn'])
    esn.update(N_units=N_R, N_wash=N_WASH, patch_size=None if G_SITES == NX else G_SITES, halo=HALO,
               periodic=G_SITES != NX)
    if G_SITES == NX == 40:                       # L96-40s: the Tikhonov factor of the paper (SINGLE_TIKH_RANGE)
        esn.update(tikh_range=SINGLE_TIKH_RANGE)
    esn.update(OVERRIDES)
    split = (truth['split_lyap'][0], TRAIN_LYAP, *truth['split_lyap'][2:])
    n_lyap = truth['num_lyap_times'] - sum(truth['split_lyap']) + sum(split)
    record = {**truth, 'params': {**truth['params'], 'Nx': NX}, 'num_lyap_times': n_lyap, 'split_lyap': split}
    net, data = model_esn(resolve_model(truth['model']), record, esn, measured=list(range(NX)),
                          config_dir=os.environ.get('ESN_STORE'))
    i0, up = data['i_test'], net.upsample
    X, Y = data['clean_data'][i0 % up::up], data['noisy_data'][i0 % up::up]
    net.held_out = i0 // up                       # first network step of the held-out record (test window 0) in X, Y
    return net, X, Y, STEPS_LYAP, error_scale(data['clean_data'])


def forecast(psi, n):
    """The outputs (n, NX, m) of the closed-loop forecast from the network state psi = [u; r] (columns: members), its
    first output u included."""
    return np.concatenate([psi[None, :NX], network()[0].forecast(psi[:NX], psi[NX:], n - 1)[0]])


def window_error(psi, X, start, norm, steps_lyap):
    """RMS over the first T_lambda of the normalized error of the ensemble-mean forecast whose first output is X[start]."""
    e2 = np.sum((forecast(psi, steps_lyap).mean(-1) - X[start:start + steps_lyap]) ** 2, axis=1) / norm ** 2
    return float(np.sqrt(np.mean(e2)))


def valid_times(psi, X, starts, norm, steps_lyap, n_lyap=4):
    """Valid time (T_lambda) of the closed-loop forecasts from psi (columns), whose first output is X[starts]."""
    n = n_lyap * steps_lyap
    U = forecast(psi, n)
    err = np.array([np.linalg.norm(U[k] - X[starts + k].T, axis=0) / norm for k in range(n)])   # (n, columns)
    return np.array([valid_time(e, np.arange(n) / steps_lyap, t_max=float(n_lyap)) for e in err.T])


# ── synchronization ──────────────────────────────────────────────────────────

NT_OBS = int(os.environ.get('NT_OBS', 5))   # steps between observations (0.17 T_lambda; NT_OBS=4 for L96-10p: 0.095 T_lambda, as L96-10s)
T_A = 2.0                            # assimilation window, T_lambda
SIGMA_OBS = float(os.environ.get('SIGMA_OBS', 0.1))   # assumed observation error, relative to the largest amplitude of each observable
MEMBERS = tuple(int(m) for m in os.environ.get('MEMBERS', '10,20,40').split(','))
RADII = (tuple(None if r == 'global' else float(r) for r in os.environ['RADII'].split(',')) if 'RADII' in os.environ
         else (None, 2.0, 4.0) if G_SITES < NX else (None,))   # Gaspari-Cohn half-width, sites; None: global transform
# (one reservoir: RADII=2,4 places each unit at the site of its input, unit_positions)
INFLATIONS = tuple(g if g == 'innovation' else float(g) for g in os.environ.get('INFLATIONS', '1.0,1.05,1.1,1.2').split(','))
# 'innovation': the innovation-based inflation of the paper (INNOVATION); the stored factor
# of such a run is the mean sqrt(lambda) it applied
SENSORS = tuple(int(s) for s in os.environ.get('SENSORS', '40,20').split(','))   # observed sites, evenly spaced (run, sync)
TUNING_WINDOW0 = 20                  # windows 20-29 of the held-out record, apart from the test windows 0-19 and 30-47
TEST_WINDOWS = [w for a in os.environ.get('TEST_WINDOWS', '0-9').split(',')   # run: e.g. '0-19,30-47'
                for w in range(int(a.split('-')[0]), int(a.split('-')[-1]) + 1)]   # (the paper's L96-40p: 38 windows)


def observation_cdd(X, Y, t_obs, sensors):
    """Diagonal C_dd: CDD_STD (default: cdd_fraction of the configuration) times the standard deviation of the record (every site has the same
    statistics); CDD_STD= (empty): SIGMA_OBS times the largest magnitude of each observable over the observations
    of the window, as romda.observations.observation_covariance."""
    if CDD_STD:                                                      # the paper: 0.2 of the standard deviation
        return std_fraction_covariance(X, float(CDD_STD), sensors, identical=True)
    return np.diag((SIGMA_OBS * np.abs(Y[np.asarray(t_obs)][:, sensors]).max(0)) ** 2)


def window_start(w, steps_lyap):
    """First record step of test window w: the windows are 4 T_lambda apart from step `held_out`, with no offset
    (run_synchronization.test_window_time is a time, 2 T_lambda into the test record)."""
    return network()[0].held_out + w * 4 * steps_lyap


@collected
def filter_run(job, return_filter=False):
    """The scores of one deployment of the filter; with `return_filter`, also the filter, whose history the caller
    scores."""
    n_sensors, w, m, radius, gamma, *rest = job
    nt_obs = rest[0] if rest else NT_OBS                              # steps between observations
    t_a = rest[1] if len(rest) > 1 else T_A                           # assimilation window, T_lambda
    net, X, Y, steps_lyap, norm = network()
    sensors = sensor_sites(n_sensors)
    if os.environ.get('SENSOR_LAYOUT') == 'half':                    # the first n_sensors sites, contiguous (20: half the ring)
        sensors = np.arange(n_sensors)
    s0 = window_start(w, steps_lyap)
    n_obs = max(1, int(round(t_a * steps_lyap / nt_obs)))
    pos = net.state_positions
    loc = {} if radius is None else dict(localization=radius, loc_period=float(NX), state_positions=pos,
                                         obs_positions=sensors.astype(float))
    infl = INNOVATION if gamma == 'innovation' else dict(inflation=gamma)
    # the members start from washouts on validation windows (ESN_model.init_ensemble), drawn with the seed w
    est = da.EnsembleFilter(parent_model=net, m=m, type='sqrt', seed=w, measured=list(sensors), **infl, **loc)
    t_last = s0 + n_obs * nt_obs
    Cdd = observation_cdd(X, Y, s0 + nt_obs * np.arange(1, n_obs + 1), sensors)
    fed = lambda j: j >= s0 + nt_obs and j > t_last - steps_lyap      # the last T_lambda of the data, from the first
    state = lambda: est.model.current_state[:NX + net.N_r]
    rsd = lambda s: np.sqrt(np.mean(s[NX:].reshape(-1, net.N_units, s.shape[-1]).std(-1, ddof=1) ** 2, axis=1))
    k = np.arange(1, n_obs + 1)
    # the model outputs every record step (dt) and interpolates between network steps (dt_ESN = upsample dt): the
    # ensemble at every network step, the forecast between the observations and the analysis at each of them
    est, H = filter_record(est, Y[s0 + k * nt_obs][:, sensors], k * nt_obs * net.upsample, Cdd)
    H = H[::net.upsample, :NX + net.N_r]
    traj, rtraj = list(H[:, :NX]), [rsd(h) for h in H]                # SAVE_TRAJ: the ensemble, and the spread of
    # each reservoir (RMS over units), at every step
    e2 = [np.sum((H[j, :NX].mean(-1) - X[s0 + j]) ** 2)               # squared errors of the ensemble mean over the
          for j in range(1, len(H)) if fed(s0 + j)]                   # last T_lambda of the data
    feed = float(np.sqrt(np.mean(e2)) / norm)
    if os.environ.get('SAVE_TRAJ'):                                   # + a 2 T_lambda forecast of the ensemble mean
        u, r = state()[:NX], state()[NX:]
        for _ in range(2 * steps_lyap):
            u, r = net.closed_loop_step(u, r)
            traj.append(u)
            rtraj.append(rsd(np.vstack([u, r])))
        np.savez(os.environ['SAVE_TRAJ'] + f'_s{n_sensors}_w{w}_m{m}_r{radius}.npz', mean=np.array(traj).mean(-1), std=np.array(traj).std(-1, ddof=1), rstd=np.array(rtraj),
                 true=X[s0:s0 + len(traj)], sensors=sensors, n_data=n_obs * nt_obs, steps_lyap=steps_lyap)
    psi = np.vstack(net.closed_loop_step(state()[:NX], state()[NX:]))   # first forecast step after the last datum
    first = np.linalg.norm(psi[:NX].mean(-1) - X[t_last + 1]) / norm
    window = window_error(psi, X, t_last + 1, norm, steps_lyap)
    # the ensemble-mean forecast: advance every member, score the mean
    err = np.linalg.norm(forecast(psi, 4 * steps_lyap).mean(-1) - X[t_last + 1:t_last + 1 + 4 * steps_lyap],
                         axis=1) / norm
    valid = valid_time(err, np.arange(len(err)) / steps_lyap, t_max=4.0)
    gamma = float(np.mean(est.inflation_history.factors or [1.0])) if gamma == 'innovation' else gamma
    row = (n_sensors, w, m, np.nan if radius is None else radius, gamma, n_obs * len(sensors), first, valid, window,
           feed)
    print(*row, flush=True)
    return (row, est) if return_filter else row


def washout_run(job):
    w, n_lyap = job
    net, X, Y, steps_lyap, norm = network()
    s0 = window_start(w, steps_lyap)
    n_w = max(1, int(round(n_lyap * steps_lyap)))
    U, R = net.open_loop(Y[s0 + 1:s0 + 1 + n_w])                       # output j: the estimate of X[s0 + j + 2]
    j = np.arange(n_w - 1)
    j = j[s0 + j + 2 > s0 + n_w - steps_lyap]                         # the last T_lambda of the data
    feed = float(np.sqrt(np.mean(np.sum((U[j] - X[s0 + j + 2]) ** 2, 1))) / norm) if j.size else np.nan
    psi = np.concatenate([U[-1], R[-1]])[:, None]
    t_last = s0 + n_w                                                  # the last datum is Y[s0 + n_w]
    first = np.linalg.norm(psi[:NX, 0] - X[t_last + 1]) / norm
    valid = valid_times(psi, X, np.array([t_last + 1]), norm, steps_lyap)[0]
    return 'washout', w, n_lyap, n_w * NX, first, valid, window_error(psi, X, t_last + 1, norm, steps_lyap), feed


# ── fig:parallel_forecast: one LETKF deployment per layout of the data, and its scores over the test windows ───────

SHOWN_M, SHOWN_RADIUS = 10, 2.0     # members, Gaspari-Cohn half-width (sites)
REF_WASH_LYAP = 10.0          # the reference reservoir state: the network driven by the true state for this long
FORECAST_WINDOWS = range(10)  # the test windows scored
FORECAST_LYAP = 2.0           # closed loop after the data
LEAD_LYAP = 1.0               # closed loop before the data, in the deployments shown, from t = -1
ASSIMILATION_LYAP = 8.0       # longer than the 2 T_lambda of the sweep: one deployment, shown in time
#: (observed sites, steps between data, T_lambda of data): every site, every other site, every other site half as
#: often, every fourth site half as often
FORECAST_CASES = ((40, NT_OBS, ASSIMILATION_LYAP), (20, NT_OBS, ASSIMILATION_LYAP), (20, 2 * NT_OBS, ASSIMILATION_LYAP),
                  (10, 2 * NT_OBS, ASSIMILATION_LYAP))
SPREAD_LEVELS = (0.2, 0.3, 0.35)    # the first time the spread (t_s) and the error (t_e) fall below each
#: the scores of every window (window_score), by name
SCORE_COLUMNS = ('valid', *(f't_s{lev:g}' for lev in SPREAD_LEVELS), 'factor', 'std_ratio', 'spread', 'error', 'coverage',
                 'min_spread', 'spread_W', 'error_W', *(f't_e{lev:g}' for lev in SPREAD_LEVELS))
FORECAST_OUT = os.path.join(HERE, 'results', 'localization')
SHOWN_PATH = os.path.join(FORECAST_OUT, 'parallel_forecast_shown.pkl')   # the deployments drawn


def letkf(net, sensors, m, seed):
    """The LETKF of the paper on `net`: square root, innovation-based inflation, Gaspari-Cohn half-width SHOWN_RADIUS
    on the periodic sites, data at `sensors`."""
    return da.EnsembleFilter(parent_model=net, m=m, type='sqrt', seed=seed, **INNOVATION, measured=list(sensors),
                             localization=SHOWN_RADIUS, loc_period=float(NX), state_positions=net.state_positions,
                             obs_positions=sensors.astype(float))


def deployment(n_sensors, nt_obs, window, assimilation_lyap=ASSIMILATION_LYAP, lead_lyap=0.0):
    """The LETKF of the sweep, step by step: input rows of every member at every step, then the forecast; with
    lead_lyap > 0, the members first run in closed loop for that long, without data, from t = -lead_lyap. The
    inflation is the innovation-based one of the paper (Wang & Bishop 2003; Li et al. 2009)."""
    net, X, Y, S, norm = network()
    up = net.upsample
    sensors = sensor_sites(n_sensors)
    s0 = window_start(window, S)
    n_obs = int(round(assimilation_lyap * S / nt_obs))
    assert s0 + n_obs * nt_obs + FORECAST_LYAP * S < window_start(TUNING_WINDOW0, S), \
        'the deployment runs into the tuning windows'
    est = letkf(net, sensors, SHOWN_M, window)
    Cdd = observation_cdd(X, Y, s0 + nt_obs * np.arange(1, n_obs + 1), sensors)   # per observable, as the filter runs
    L = int(round(lead_lyap * S))                                      # steps of closed loop before the data
    k = np.arange(-L, n_obs * nt_obs + int(FORECAST_LYAP * S) + 1)     # t = 0 at the start of the data
    # reference: the reservoir driven in open loop by the true state from REF_WASH_LYAP before the first step
    # shown; the state at step k has read the input of step k - 1
    a = s0 + k[0] - 1 - int(REF_WASH_LYAP * S)
    ref = net.open_loop(X[a:s0 + k[-1]])[1][k + s0 - 1 - a]
    unit = int(np.argmax(ref[:, :net.N_units].std(axis=0)))           # the unit of reservoir 1 that moves most
    ref = ref[:, unit]
    j = nt_obs * np.arange(1, n_obs + 1)                               # the observations, network steps from t = 0
    est, H = filter_record(est, Y[s0 + j][:, sensors], (L + j) * up, Cdd)
    H = H[::up, :NX + net.N_r]                                         # every network step
    U, R = list(H[:, :NX]), list(H[:, NX + unit])
    u, r = H[-1, :NX], H[-1, NX:]
    for _ in range(int(FORECAST_LYAP * S)):
        u, r = net.closed_loop_step(u, r)
        U.append(u)
        R.append(r[unit].copy())
    U = np.array(U)                                                     # (Nt, NX, M)
    t_obs = np.arange(1, n_obs + 1) * nt_obs
    return dict(t=k / S, U=U, R=np.array(R), r_ref=ref, unit=unit, truth=X[s0 + k], t_obs=t_obs / S,
                d=Y[s0 + t_obs][:, sensors], sensors=sensors, t_last=n_obs * nt_obs / S, norm=float(norm),
                factor=float(np.mean(est.inflation_history.factors)))   # mean sqrt(lambda)


def deployment_scores(r):
    """Normalized error and spread, valid time after the last datum, and the first times the spread (t_s) and the
    error (t_e) fall below each of SPREAD_LEVELS."""
    mean, std = r['U'].mean(-1), r['U'].std(-1, ddof=1)
    err = np.linalg.norm(mean - r['truth'], axis=1) / r['norm']
    spread = np.sqrt(np.sum(std ** 2, axis=1)) / r['norm']
    after = r['t'] > r['t_last']
    valid = valid_time(err[after], r['t'][after] - r['t_last'], t_max=FORECAST_LYAP)
    first = lambda x, lev: first_below(r['t'], x, lev, r['t_last'], 0.0)   # from the first datum to the last
    return err, spread, valid, [first(spread, lev) for lev in SPREAD_LEVELS], [first(err, lev) for lev in SPREAD_LEVELS]


def uncertainty(r, t0, t1):
    """From t0 to t1: standard deviation at the unobserved over that at the observed sites (each normalized by the
    RMS amplitude of the site), RMS of the normalized spread and error, coverage of the 2-sigma band (population
    std; 0.896 for a calibrated ensemble of M = 10), and the smallest spread."""
    err, spread, _, _, _ = deployment_scores(r)
    win = (r['t'] >= t0) & (r['t'] <= t1)
    U, truth = r['U'][win], r['truth'][win]
    sd = U.std(-1, ddof=1) / np.sqrt(np.mean(truth ** 2, axis=0))
    obs = np.isin(np.arange(NX), r['sensors'])
    rms = lambda a: np.sqrt(np.mean(a ** 2))
    ratio = rms(sd[:, ~obs]) / rms(sd[:, obs]) if (~obs).any() else np.nan
    cover = np.mean(np.abs(U.mean(-1) - truth) <= 2 * U.std(-1))
    return [ratio, rms(spread[win]), rms(err[win]), cover, spread[win].min()]


@collected
def window_score(job):
    """The scores SCORE_COLUMNS of one deployment (n_sensors, nt_obs, assimilation_lyap, window)."""
    n_sensors, nt_obs, assimilation_lyap, w = job
    r = deployment(n_sensors, nt_obs, w, assimilation_lyap)
    _, _, valid, t_s, t_e = deployment_scores(r)
    last = uncertainty(r, r['t_last'] - 1 + 0.5 / STEPS_LYAP, r['t_last'])   # e_W: the last T_lambda of the data
    return [valid] + t_s + [r['factor']] + uncertainty(r, 2.0, r['t_last']) + last[1:3] + t_e


def window_scores_path(cdd=CDD_STD):
    return os.path.join(FORECAST_OUT, f'parallel_forecast_windows_cdd{float(cdd):g}.npz')


def window_scores():
    """(case, window, SCORE_COLUMNS) of FORECAST_CASES over FORECAST_WINDOWS; resumes from its output file, and
    runs only the cases missing from it."""
    path = window_scores_path()
    key = np.array([str((*c, 'innovation', 't_s spread, t_e')) for c in FORECAST_CASES])   # sensors, steps between data, T_lambda
    old = dict(zip(z['cases'], z['scores'])) if os.path.exists(path) and 'cases' in (z := np.load(path)) else {}
    todo = [c for c, k in zip(FORECAST_CASES, key) if k not in old]
    if todo:
        jobs = [(*c, w) for c in todo for w in FORECAST_WINDOWS]
        # N_WORKERS: explicit, small by default; a worker of the L96-40 sweeps can hold tens of GB
        with ProcessPoolExecutor(max_workers=min(len(jobs), int(os.environ.get('N_WORKERS', 2)))) as ex:
            new = np.array(list(ex.map(window_score, jobs))).reshape(len(todo), len(FORECAST_WINDOWS), -1)
        old.update(zip(key[[k not in old for k in key]], new))
    out = np.array([old[k] for k in key])
    assert out.shape[-1] == len(SCORE_COLUMNS)
    if todo:                                     # only new results: the file may be the record of the paper
        np.savez(path, scores=out, cases=key, columns=np.array(SCORE_COLUMNS))
        print('wrote', path)
    return out


def shown_window(scores):
    """The test window closest to the median valid time of every case, summed over the cases."""
    med = np.median(scores[:, :, 0], axis=1)
    return int(np.argmin(np.abs(scores[:, :, 0] - med[:, None]).sum(axis=0)))


def shown_key(case, window):
    """The key of a deployment drawn in parallel_forecast_shown.pkl."""
    n_sensors, nt_obs, assimilation_lyap = case
    return n_sensors, 'innovation', nt_obs, window, assimilation_lyap, LEAD_LYAP


def shown_deployments(window):
    """The deployments of FORECAST_CASES drawn by fig:parallel_forecast on `window`, from LEAD_LYAP before the data,
    in parallel_forecast_shown.pkl, a dict keyed by `shown_key`; runs only the deployments missing from it."""
    shown = pickle.load(open(SHOWN_PATH, 'rb')) if os.path.exists(SHOWN_PATH) else {}
    todo = [c for c in FORECAST_CASES if 'R' not in shown.get(shown_key(c, window), {})]
    for c in todo:
        shown[shown_key(c, window)] = deployment(c[0], c[1], window, c[2], LEAD_LYAP)
    if todo:                                     # only new results: the file may be the record of the paper
        pickle.dump(shown, open(SHOWN_PATH, 'wb'))
        print('wrote', SHOWN_PATH)


# ── fig:localized_covariance: the forecast correlations that the LETKF localizes ────────────────────────────────────

COV_SENSORS, COV_WINDOW = 20, 0                     # every other site, test window 0
COV_MEMBERS = tuple(int(m) for m in os.environ.get('COV_MEMBERS', '10,1000').split(','))   # the filter, the reference
# (m = 1000 peaks at about 41 GB: COV_MEMBERS=10 runs the filter alone, and the figure then omits row (e))
COV_PATH = os.path.join(FORECAST_OUT, 'localized_covariance.npz')
COV_POSITIONS_PATH = os.path.join(FORECAST_OUT, 'localized_covariance_positions.npz')   # sites of the reservoirs


def forecast_ensemble(m):
    """The forecast ensemble before the last analysis of the LETKF (half-width SHOWN_RADIUS, innovation-based
    inflation) with m members, from COV_SENSORS sites for T_A in test window COV_WINDOW, and the observed sites."""
    net, X, Y, steps_lyap, _ = network()
    sensors = sensor_sites(COV_SENSORS)
    s0 = window_start(COV_WINDOW, steps_lyap)
    n_obs = max(1, int(round(T_A * steps_lyap / NT_OBS)))
    est = letkf(net, sensors, m, COV_WINDOW)
    Cdd = observation_cdd(X, Y, None, sensors)
    k = np.arange(1, n_obs)                                           # every analysis but the last
    est = filter_record(est, Y[s0 + k * NT_OBS][:, sensors], k * NT_OBS * net.upsample, Cdd)[0]
    est.forecast_step(Nt=NT_OBS * net.upsample)
    return est.model.current_state[:NX + net.N_r].copy(), sensors


def correlations(psi, sensors):
    """Correlation of every row of psi with the observed input rows: (N_u, N_q) inputs, (P, N_q) reservoirs (RMS)."""
    A = psi - psi.mean(-1, keepdims=True)
    A /= np.maximum(A.std(-1, keepdims=True), 1e-300)
    C = A @ A[sensors].T / A.shape[1]
    net = network()[0]
    reservoir = np.sqrt(np.mean(C[NX:].reshape(net.N_patches, net.N_units, -1) ** 2, axis=1))
    return C[:NX], reservoir


def covariance():
    """The correlations of the forecast ensembles of COV_MEMBERS in COV_PATH (u<m>: inputs, r<m>: reservoirs, and
    the observed sites), and the position of every reservoir in COV_POSITIONS_PATH; runs only what is missing."""
    out = dict(np.load(COV_PATH)) if os.path.exists(COV_PATH) else {}
    todo = [m for m in COV_MEMBERS if f'u{m}' not in out]
    for m in todo:
        psi, out['sensors'] = forecast_ensemble(m)
        out[f'u{m}'], out[f'r{m}'] = correlations(psi, out['sensors'])
    if todo:                                     # only new results: the file may be the record of the paper
        np.savez(COV_PATH, **out)
        print('wrote', COV_PATH)
    if not os.path.exists(COV_POSITIONS_PATH):   # the site of every reservoir, so that the figure needs no network
        net = network()[0]
        np.savez(COV_POSITIONS_PATH, reservoir=net.state_positions[NX::net.N_units])
        print('wrote', COV_POSITIONS_PATH)


if __name__ == '__main__':
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    if sys.argv[1] == 'train':                          # train (or load) the network of the layout
        print(network_summary(network()[0]))
    elif sys.argv[1] == 'sync':                       # Fig. 2, third column: data consumed, as run_synchronization.py
        # LETKF, N_e = 10, half-width 2 sites, no inflation (the paper)
        gamma = dict.fromkeys(SENSORS, 'innovation')
        lengths = {s: (0.25, 0.5, 1.0, 2.0, 4.0, 8.0) for s in SENSORS}
        _, X, _, STEPS_LYAP, _ = network()
        fits = lambda w, T: (window_start(w, STEPS_LYAP) + round(T * STEPS_LYAP) + 1 + 4 * STEPS_LYAP
                             <= len(X))                  # the forecast after the last datum stays in the record
        jobs = [(s, w, 10, 2.0, gamma[s], NT_OBS, T) for s in SENSORS for w in range(10)
                for T in lengths[s] if fits(w, T)]
        # N_WORKERS: explicit, small by default; a worker of the L96-40 sweeps can hold tens of GB
        with ProcessPoolExecutor(max_workers=int(os.environ.get('N_WORKERS', 2))) as ex:
            rows = list(ex.map(filter_run, jobs))
            wash = list(ex.map(washout_run, [(w, L) for w in range(10) for L in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
                                             if NX in SENSORS]))   # full state only
        rows = [('filter', r[0], T, r[1], r[5], r[6], r[7], r[8], r[9]) for r, (*_, T) in zip(rows, jobs)]
        rows += [('washout', NX, L, w, n, e0, v, e1, ef) for _, w, L, n, e0, v, e1, ef in wash]
        arr = np.array([r[1:] for r in rows], float)
        tag = '' if SENSORS == (NX, NX // 2) else '_s' + '_'.join(map(str, SENSORS))
        np.savez(os.path.join(HERE, 'results', 'synchronization', f'synchronization_lorenz96_parallel{SUFFIX}{tag}{TAG}.npz'),
                 method=np.array([r[0] for r in rows]), n_sensors=arr[:, 0], length_lyap=arr[:, 1], window=arr[:, 2],
                 n_data=arr[:, 3], first_error=arr[:, 4], valid_lyap=arr[:, 5], window_error=arr[:, 6], feed_error=arr[:, 7])
    elif sys.argv[1] == 'covariance':                 # fig:localized_covariance
        covariance()
    elif sys.argv[1] == 'forecast':                   # fig:parallel_forecast, and L96-40p of fig:uncertainty
        sc = window_scores()
        if float(CDD_STD) == _DA['cdd_fraction']:     # the figure draws the assumed observation error of the paper
            shown_deployments(shown_window(sc))
    else:
        windows = range(int(sys.argv[2])) if len(sys.argv) > 2 else TEST_WINDOWS
        jobs = [(s, w, m, c, g) for s in SENSORS for m in MEMBERS for c in RADII for g in INFLATIONS
                for w in windows]
        with ProcessPoolExecutor(max_workers=int(os.environ.get('N_WORKERS', 2))) as ex:
            rows = np.array(list(ex.map(filter_run, jobs)), float)
            wash = list(ex.map(washout_run, [(w, L) for w in windows for L in (0.1, 0.25, 0.5, 1.0, 2.0)]))
        wash = np.array([r[1:] for r in wash], float)
        np.savez(os.path.join(HERE, 'results', 'localization', f'localization_parallel_lorenz96{SUFFIX}{TAG}.npz'),
                 n_sensors=rows[:, 0], window=rows[:, 1], members=rows[:, 2], radius=rows[:, 3], inflation=rows[:, 4],
                 n_data=rows[:, 5], first_error=rows[:, 6], valid_lyap=rows[:, 7], window_error=rows[:, 8], feed_error=rows[:, 9],
                 wash_window=wash[:, 0], wash_lyap=wash[:, 1], wash_data=wash[:, 2], wash_first=wash[:, 3],
                 wash_valid=wash[:, 4], wash_error=wash[:, 5], wash_feed=wash[:, 6])
