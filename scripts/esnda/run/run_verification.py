"""Appendix A of the washout-free paper: do the trained networks reproduce the dynamics and statistics of the systems?

For the networks that the filter synchronizes (Lorenz-63, Lorenz-96 with 10 sites, the parallel networks of Lorenz-96
with 10 and 40 sites) and the single reservoir of the 40 sites (Sect. 4.4), against the true system:
* short-term dynamics: valid time of closed-loop forecasts after a washout of 2 T_lambda, test windows;
* statistics: probability density and autocorrelation of every component, over LONG_LYAP of closed loop against
  LONG_LYAP of the true system, and the Kullback-Leibler divergence of the densities (`statistics`);
* stability: the leading Lyapunov exponents of the closed-loop network and of the true system, by the same
  finite-difference QR method (Benettin, re-orthonormalized every step) on each map, over SEG_LYAP from the
  starts that stay on the attractor;
* long closed loop: from the washouts of ten test windows, the time at which the closed loop leaves the
  attractor (largest component above ESCAPE times the largest of the record), within ESCAPE_LYAP.
The statistics and the exponents start from the first of these states that stays on the attractor for
LONG_LYAP; the text reports the escape times.

    python run_verification.py [L63] [L96-10] [L96-40]   (default: all three)
        -> results/verification/esn_verification_runs.pkl (the runs of every network, resumed: a network in the file
           is not run again) and results/verification/esn_verification.npz (the summary of tab:esns)

plot_verification.py draws fig:verification from these runs.
"""

import os
import pickle
import runpy
import sys

import numpy as np

from romda.metrics import attractor_kld, error_scale, lyapunov_spectrum_map
from romda.models.physical import Lorenz63, Lorenz96

import run_synchronization as SYNC
from run_parametric import ESCAPE   # three times, as Appendix B; Fig. 2 counts twice (run_partial_table)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # scripts/esnda: results/
RUN = os.path.join(HERE, 'run')
RUNS = os.path.join(HERE, 'results', 'verification', 'esn_verification_runs.pkl')
SUMMARY = os.path.join(HERE, 'results', 'verification', 'esn_verification.npz')
LONG_LYAP, SEG_LYAP, TRANSIENT_LYAP, ACF_LYAP, N_WINDOWS = 400, 50, 5, 3.0, 10
ESCAPE_LYAP = 500


def run_script(script='run_parallel_localization.py', **env):
    """The namespace of a run/ script with `env` set in the environment, then restored; for
    run_parallel_localization.py, one layout (NX, G_SITES, HALO, N_R; the environment selects its network)."""
    old = {k: os.environ.get(k) for k in env}
    os.environ.update({k: str(v) for k, v in env.items()})
    try:
        return runpy.run_path(os.path.join(RUN, script), run_name=script[:-3])
    finally:
        for k, v in old.items():
            os.environ.pop(k) if v is None else os.environ.__setitem__(k, v)


def closed_loop_map(net):
    """The closed-loop step of the network state [u; r] (columns: members), the map of `lyapunov_spectrum_map`, `run`
    and `escape_time`."""
    return lambda p: np.vstack(net.closed_loop_step(p[:net.N_dim], p[net.N_dim:]))


def run(step, x0, n):
    """n steps of the map from x0 (N, 1); returns the states (n, N)."""
    x, out = x0, np.empty((n, x0.shape[0]))
    for k in range(n):
        x = step(x)
        out[k] = x[:, 0]
    return out


def escape_time(step, psi0, N, steps_lyap, bound):
    """First time (T_lambda) at which the closed loop from psi0 leaves |u| <= bound; inf if never."""
    x = psi0.copy()
    for k in range(ESCAPE_LYAP * steps_lyap):
        x = step(x)
        if np.abs(x[:N]).max() > bound:
            return k / steps_lyap
    return np.inf


def stay_and_stats(net_step, true_map, starts, truths, N, S, dt, bound, K):
    """Escape times from every start; long run, statistics and exponents from the first start that stays. The truth
    does not depend on the network: its long run from the first start, its exponents from every start."""
    esc = np.array([escape_time(net_step, p, N, S, bound) for p in starts])
    i = int(np.argmax(esc >= LONG_LYAP + TRANSIENT_LYAP))
    long_lyap = LONG_LYAP
    if esc[i] < LONG_LYAP + TRANSIENT_LYAP:          # no start stays that long: the longest stay, up to its escape
        i = int(np.argmax(esc))
        long_lyap = int(esc[i] - TRANSIENT_LYAP)
        print(f'no start stays on the attractor for {LONG_LYAP} T_lambda; statistics over {long_lyap} T_lambda', flush=True)
    Z = run(net_step, starts[i], long_lyap * S)[:, :N]
    Y = run(true_map, truths[0][:, None], LONG_LYAP * S)
    # spectra over SEG_LYAP from the starts whose closed loop stays on the attractor that long
    ok = [w for w in range(len(starts)) if esc[w] >= SEG_LYAP + TRANSIENT_LYAP]
    seg = SEG_LYAP
    if not ok:                                       # no start stays that long: the longest stay, up to its escape
        ok, seg = [int(np.argmax(esc))], int(esc.max() - TRANSIENT_LYAP)
        print(f'no start stays on the attractor for {SEG_LYAP} T_lambda; exponents over {seg} T_lambda', flush=True)
    lam_net = np.array([lyapunov_spectrum_map(net_step, starts[w][:, 0], dt, seg * S, TRANSIENT_LYAP * S, K, bound, N)
                        for w in ok])
    lam_true = np.array([lyapunov_spectrum_map(true_map, x, dt, SEG_LYAP * S, TRANSIENT_LYAP * S, K, bound, N)
                         for x in truths])
    return dict(escape=esc, start=i, Y=Y, Z=Z, lam_net=lam_net, lam_true=lam_true)


def washouts(net, windows, n_w, n_ref):
    """For each test window, (noisy, clean) in network steps from the first input of its washout: the network state
    [u; r] (one column) after the open loop on the first n_w noisy states, the true state that it estimates,
    clean[n_w], and the n_ref true states after it."""
    starts, truths, refs = [], [], []
    for noisy, clean in windows:
        U, R = net.open_loop(noisy[:n_w])
        starts.append(np.concatenate([U[-1], R[-1]])[:, None])
        truths.append(clean[n_w])
        refs.append(clean[n_w + 1 + np.arange(n_ref)])
    return starts, truths, refs


def verify(net, windows, n_w, S, true_map, K, bound, valid, t_lyap):
    """Escape times, statistics and K exponents (`stay_and_stats`) from the washouts of `windows`, with the valid
    times of the test windows (`valid`, or a function of the washout states that returns them) and one forecast for
    the figure (first test window)."""
    starts, truths, refs = washouts(net, windows, n_w, 4 * S)
    valid = valid(starts) if callable(valid) else valid
    net_step = closed_loop_map(net)
    out = stay_and_stats(net_step, true_map, starts, truths, net.N_dim, S, net.dt_ESN, bound, K)
    fc = run(net_step, starts[0], 4 * S)[:, :net.N_dim]
    return dict(out, valid=np.array(valid), S=S, fc=fc, ref=refs[0], t_lyap=t_lyap)


def test_windows(data, dt, t_lyap):
    """First record step of the washout of each test window of `run_synchronization` (2 + 4 w T_lambda in)."""
    return [data['i_test'] + int(round(SYNC.test_window_time(w, t_lyap) / dt)) for w in range(N_WINDOWS)]


def single(case, n_sensors, truth_step, substeps, K):
    """A romda network (run_synchronization.network): forecast windows, long run, Lyapunov exponents."""
    esn, data, _, _ = SYNC.network(case, n_sensors)
    t_lyap, dt = data['t_lyap'], esn.dt_ESN
    S = int(round(t_lyap / dt))
    valid = []
    for w in range(N_WINDOWS):
        try:
            valid.append(SYNC.washout_run(esn, data, SYNC.test_window_time(w, t_lyap), 2.0 * t_lyap, t_lyap)[1][1])
        except (IndexError, ValueError):
            break
    up = esn.upsample
    windows = [(data['noisy_data'][i0::up], data['clean_data'][i0::up]) for i0 in test_windows(data, esn.dt, t_lyap)]
    true_map = lambda x: [x := truth_step(x, dt / substeps) for _ in range(substeps)][-1]
    return verify(esn, windows, int(round(2 * t_lyap / dt)), S, true_map, K, ESCAPE * np.abs(data['clean_data']).max(),
                  valid, t_lyap)


def parallel(K, PAR):
    """A network of run_parallel_localization.py (run_script) on its own record: parallel, or one reservoir with
    G_SITES = NX."""
    net, X, Y, S, _ = PAR['network']()
    DT = net.dt_ESN
    valid = [PAR['washout_run']((w, 2.0))[5] for w in range(N_WINDOWS)]   # (..., first, valid, ...)
    windows = [(Y[s0 + 1:], X[s0 + 1:]) for s0 in (PAR['window_start'](w, S) for w in range(N_WINDOWS))]
    true_map = lambda x: Lorenz96.rk4_step(x, DT, h=DT / 2)   # two substeps of the record's dt
    return verify(net, windows, 2 * S, S, true_map, K, ESCAPE * np.abs(X).max(), valid, S * DT)


def on_record(K, PAR, case):
    """parallel() on the test record and windows of single(case): the same washout data, truth and normalization, so
    that the network is scored on the same data as the single network of that case."""
    esn, data, _, _ = SYNC.network(case, PAR['NX'])
    noisy, clean, up, t_lyap = data['noisy_data'], data['clean_data'], esn.upsample, data['t_lyap']
    net, _, _, S, _ = PAR['network']()
    DT = net.dt_ESN
    assert np.isclose(esn.dt_ESN, DT) and clean.shape[1] == PAR['NX'], 'the records differ in time step or sites'
    norm, n_w = error_scale(clean[data['i_test']:]), 2 * S
    windows = [(noisy[i0::up], clean[i0::up]) for i0 in test_windows(data, esn.dt, t_lyap)]
    windows = [(Yw, Xw) for Yw, Xw in windows if n_w + 1 + 4 * S <= len(Xw)]   # the forecast stays in the record
    valid = lambda starts: [PAR['valid_times'](s, Xw, np.array([n_w]), norm, S)[0]   # from the washouts of verify
                            for s, (_, Xw) in zip(starts, windows)]
    true_map = lambda x: Lorenz96.rk4_step(x, DT, h=DT / 2)   # as single(case): two substeps
    return verify(net, windows, n_w, S, true_map, K, ESCAPE * np.abs(clean).max(), valid, t_lyap)


#: (name, run, pooled statistics) of every network, and the networks of each system (one column of the figure)
ROWS = [('Lorenz-63', lambda: single('lorenz63', 3, Lorenz63.rk4_step, 4, 3), False),
        ('Lorenz-96, 10 sites', lambda: single('lorenz96', 10, Lorenz96.rk4_step, 2, 10), True),
        ('Lorenz-96, 10 sites, parallel', lambda: on_record(10, run_script(NX=10, G_SITES=5, HALO=2, N_R=500),
                                                            'lorenz96'), True),
        ('Lorenz-96, 40 sites, parallel', lambda: parallel(20, run_script()), True),
        ('Lorenz-96, 40 sites, one reservoir',
         lambda: parallel(20, run_script(G_SITES=40, HALO=0, N_R=6000, TRAIN_LYAP=800)), True)]
SYSTEMS = {'L63': ('Lorenz-63',), 'L96-10': ('Lorenz-96, 10 sites', 'Lorenz-96, 10 sites, parallel'),
           'L96-40': ('Lorenz-96, 40 sites, parallel', 'Lorenz-96, 40 sites, one reservoir')}


def statistics(Y, Z, steps_lyap, pooled):
    """Densities, autocorrelations and the KL divergence of network (Z) against truth (Y), both (n, N)."""
    groups = [np.arange(Y.shape[1])] if pooled else [[i] for i in range(Y.shape[1])]
    out = []
    for g in groups:
        y, z = Y[:, g].ravel(), Z[:, g].ravel()
        edges = np.linspace(min(y.min(), z.min()), max(y.max(), z.max()), 81)
        py = np.histogram(y, edges, density=True)[0] + 1e-12                # the densities of the figure
        pz = np.histogram(z, edges, density=True)[0] + 1e-12
        kl = float(attractor_kld(y[:, None], z[:, None, None])[0])          # on the bins of the truth, as fig:kld
        lags = np.arange(int(ACF_LYAP * steps_lyap) + 1)
        acf = lambda A: np.mean([[np.corrcoef(a[:len(a) - l], a[l:])[0, 1] if l else 1.0 for l in lags]
                                 for a in A[:, g].T], axis=0)
        out.append(dict(edges=edges, py=py, pz=pz, kl=kl, lags=lags / steps_lyap, acf_y=acf(Y), acf_z=acf(Z),
                        std_ratio=float(z.std() / y.std())))
    return out


def load_runs():
    """{name: the run of the network, with its statistics (`st`) and the exponents of the starts that stay on the
    attractor over SEG_LYAP} of every network in RUNS."""
    done = {}
    for name, c, pooled in pickle.load(open(RUNS, 'rb')):
        stay = np.isfinite(c['lam_net']).all(1)      # a start can leave within the exponents' segment (spectrum: nan)
        if stay.any() and not stay.all():
            print(f'{name}: {np.sum(~stay)} of {stay.size} starts leave the attractor within the {SEG_LYAP} T_lambda '
                  f'of the exponents; exponents from the other {np.sum(stay)}', flush=True)
            c = dict(c, lam_net=c['lam_net'][stay])
        done[name] = dict(c, st=statistics(c['Y'], c['Z'], c['S'], pooled))
    return done


def main(systems):
    """Run the networks of `systems` that RUNS does not hold, then write the summary of every network in RUNS."""
    names = [n for s in systems for n in SYSTEMS[s]]
    done = {name: c for name, c, _ in pickle.load(open(RUNS, 'rb'))} if os.path.exists(RUNS) else {}
    todo = [(name, compute) for name, compute, _ in ROWS if name in names and name not in done]
    for name, compute in todo:
        done[name] = compute()
        pickle.dump([(n, done[n], pooled) for n, _, pooled in ROWS if n in done], open(RUNS, 'wb'))
    if not todo and os.path.exists(SUMMARY):     # nothing new: the summary may be the record of the paper
        print('complete:', RUNS)
        return
    summary = {}
    for name, c in load_runs().items():
        summary[name] = dict(escape=c['escape'].tolist(), start=c['start'], valid_median=float(np.median(c['valid'])),
                             valid_iqr=np.percentile(c['valid'], [25, 75]).tolist(), lam_net=c['lam_net'].tolist(),
                             lam_true=c['lam_true'].tolist(), kl=float(np.mean([s['kl'] for s in c['st']])),
                             std_ratio=[s['std_ratio'] for s in c['st']])
        print(name, summary[name], flush=True)
    np.savez(SUMMARY, summary=summary)
    print('wrote', SUMMARY)


if __name__ == '__main__':
    main(sys.argv[1:] or list(SYSTEMS))
