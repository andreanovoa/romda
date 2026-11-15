"""Synchronization without a washout: how much data a trained ESN needs before it forecasts.

Two ways to bring the reservoir of a trained network onto the trajectory of the data, from the same point of
the held-out record (2 Lyapunov times in, shifted by 4 per window), followed by the same closed-loop forecast:

* the washout: one network, r = 0, driven open loop by the full noisy state at every network step for T_w;
* the filter: the EnSRKF of the paper (members from reservoir states visited in training, Table 1 of the
  paper), assimilating the measured components every Nt_obs record steps for T_a.

Each run records the number of scalar data it consumed, the error of the first forecast step, the error over the
first Lyapunov time of the forecast, the error over the last Lyapunov time of the data (from the first datum at
the latest: the network output during the washout; the ensemble mean, analysis at the observation times and
forecast between them, during the filter; root mean square over each window) and the valid time of the forecast
after the last datum (normalized error, one normalization for all components, first
above 0.5), in Lyapunov times. The normalization is `romda.metrics.error_scale` of the test record, the
root-mean-square deviation of the true state from its time average.

    python run_synchronization.py [lorenz63|lorenz96] [N_WINDOWS, default 20]   -> results in results/synchronization/synchronization_<case>.npz

SENSORS=2 (comma-separated) runs those layouts alone, into results/synchronization/synchronization_<case>_s2.npz; the washout runs only
with the full state. N_ANALYSES=1,2 runs the filter for that many analyses alone; CDD_STD sets C_dd (a fraction of
the standard deviation of each component) and TAG is appended to the output name.
"""

import os
import sys

from pathlib import Path

import numpy as np
from romda.run import innovation_options, load_config
from romda.data_assimilation import deploy
from romda.models.data_driven.rnns import model_esn
from romda.metrics import error_scale, forecast_scores
from romda.models.physical import Lorenz63, Lorenz96
from romda.observations import Observations, measured_idx

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # scripts/esnda: results/
CONFIGS = Path(HERE) / 'configs'   # the configurations of the experiments
FORECAST_LYAP = 4.0

CASES = {
    'lorenz63': dict(model=Lorenz63, cfg='lorenz63', sensors=(3, 1), gamma=dict.fromkeys((3, 2, 1), 'innovation'),
                     washout_lyap=(0.1, 0.25, 0.5, 1.0, 2.0, 4.0), filter_lyap=(0.25, 0.5, 1.0, 2.0, 4.0, 8.0)),
    # gamma: 'innovation' is the innovation-based inflation of the paper (the options of the da block of the configuration)
    'lorenz96': dict(model=Lorenz96, cfg='lorenz96_small', sensors=(10, 5), gamma=dict.fromkeys((10, 5, 3, 2), 'innovation'),
                     washout_lyap=(0.1, 0.25, 0.5, 1.0, 2.0, 4.0), filter_lyap=(0.25, 0.5, 1.0, 2.0, 4.0, 8.0)),
}


def network(case, n_sensors):
    c = CASES[case]
    defaults = load_config(CONFIGS / f"{c['cfg']}.yml")
    if case == 'lorenz63':                  # the paper: 30 members (the best of 5-30, Table 2); the yml has 20
        defaults['da']['m'] = int(os.environ.get('M', 30))   # M: other members (a quicker check)
    dims = (measured_idx(n_sensors, 10, ring=True) if case == 'lorenz96' else measured_idx(n_sensors, 3))
    esn, data = model_esn(c['model'], defaults['truth'], defaults['forecast']['esn'], measured=dims)
    return esn, data, dims, defaults


def test_window_time(w, t_lyap):
    """Start time of test window w after the start of the test record (step ``data['i_test']``): 2 T_lambda in,
    4 T_lambda apart. Not run_parallel_localization.window_start, a record step from `held_out` with no offset."""
    return (2.0 + 4.0 * w) * t_lyap


def scores(t, y, t_truth, truth, t_last, t_lyap):
    """Error of the first forecast step, valid time (T_lambda) and error over the first T_lambda after t_last."""
    norm = error_scale(truth)                              # over the test record
    sel = t > t_last + 1e-9
    tt, yy = t[sel], y[sel]
    ref = np.array([np.interp(tt, t_truth, truth[:, i]) for i in range(truth.shape[1])]).T
    return forecast_scores(np.linalg.norm(yy - ref, axis=1) / norm, tt - t_last, t_lyap)


def feed_error(t, y, t_truth, truth, t_first, t_last, t_lyap):
    """RMS normalized error of the estimate y(t) over the last T_lambda of the data, from the first datum t_first."""
    sel = (t >= t_first - 1e-9) & (t > t_last - t_lyap + 1e-9) & (t <= t_last + 1e-9)
    ref = np.array([np.interp(t[sel], t_truth, truth[:, i]) for i in range(truth.shape[1])]).T
    return float(np.sqrt(np.mean(np.sum((y[sel] - ref) ** 2, axis=1)) / error_scale(truth) ** 2))


def washout_run(esn, data, t_start, T_w, t_lyap):
    """Classic washout: r = 0, the full noisy state at every network step, then the closed loop."""
    noisy, clean, t_rec = data['noisy_data'], data['clean_data'], data['t']
    i_test, up = data['i_test'], esn.upsample
    i0 = i_test + int(round(t_start / esn.dt))
    n_w = max(1, int(round(T_w / esn.dt_ESN)))
    fed, R = esn.open_loop(noisy[i0:i0 + n_w * up:up])          # output after input k: the estimate at k + 1
    t_last = t_rec[i0 + (n_w - 1) * up] - t_rec[i_test]       # time of the last datum
    n_fc = int(round(FORECAST_LYAP * t_lyap / esn.dt_ESN))
    ys = np.vstack([fed[-1:], esn.forecast(fed[-1], R[-1], n_fc)[0][..., 0]])
    t = t_last + esn.dt_ESN * np.arange(1, n_fc + 2)
    t_truth = t_rec[i_test:] - t_rec[i_test]
    t_fed = t_rec[i0 + up * np.arange(1, n_w)] - t_rec[i_test]          # during the washout: k + 1 <= n_w - 1
    feed = feed_error(t_fed, fed[:-1], t_truth, clean[i_test:],
                      t_fed[0] if n_w > 1 else t_last, t_last, t_lyap) if n_w > 1 else np.nan
    return n_w * clean.shape[1], scores(t, ys, t_truth, clean[i_test:], t_last, t_lyap) + (feed,)


def filter_run(esn, data, dims, defaults, t_start, T_a, gamma, t_lyap):
    """The EnSRKF of the paper on [t_start, t_start + T_a], then the closed-loop forecast."""
    da = defaults['da']
    noisy, clean, i_test = data['noisy_data'], data['clean_data'], data['i_test']
    t_truth = data['t'][i_test:] - data['t'][i_test]
    truth = Observations(y_raw=noisy[i_test:][:, dims], y_true=clean[i_test:][:, dims], t_true=t_truth,
                         t_start=t_start, t_stop=t_start + T_a, Nt_obs=da['Nt_obs'])
    res = deploy(esn, truth, measured=dims, m=da['m'], da_method=da['da_method'], std_obs=da['std_obs'],
                 t_extra=FORECAST_LYAP * t_lyap, inflation=gamma, **innovation_options(da),
                 cdd_fraction=float(os.environ.get('CDD_STD', da['cdd_fraction'])))
    t_obs = np.asarray(truth.t_obs)
    n_data = t_obs.size * len(dims)
    t, y = np.asarray(res['t']), np.asarray(res['y'])
    feed = feed_error(t, y, t_truth, clean[i_test:], t_obs[0], t_obs[-1], t_lyap)
    return n_data, scores(t, y, t_truth, clean[i_test:], t_obs[-1], t_lyap) + (feed,)


if __name__ == '__main__':
    case = sys.argv[1] if len(sys.argv) > 1 else 'lorenz63'
    n_windows = int(sys.argv[2]) if len(sys.argv) > 2 else 20   # the paper: 20 test windows
    c = CASES[case]
    sensors = tuple(int(s) for s in os.environ['SENSORS'].split(',')) if 'SENSORS' in os.environ else c['sensors']
    tag = '_s' + '_'.join(map(str, sensors)) if 'SENSORS' in os.environ else ''
    tag += os.environ.get('TAG', '')  # e.g. _cddnoise: C_dd at the noise level (CDD_STD=0.05 L63, 0.02 L96)
    rows = []                         # (method, n_sensors, length_lyap, window, n_data, first error, valid time)
    for n_sensors in sensors:
        esn, data, dims, defaults = network(case, n_sensors)
        t_lyap = data['t_lyap']
        lengths = c['filter_lyap']
        if 'N_ANALYSES' in os.environ:   # N_ANALYSES=1: that many analyses (TAG=_n1 SENSORS=1: one scalar datum)
            d_obs = defaults['da']['Nt_obs'] * esn.dt / t_lyap
            lengths = [(int(k) - 0.5) * d_obs for k in os.environ['N_ANALYSES'].split(',')]
        for w in range(n_windows):
            t_start = test_window_time(w, t_lyap)
            if n_sensors == esn.N_dim and 'N_ANALYSES' not in os.environ:   # the washout needs the full state
                for T in c['washout_lyap']:
                    n, (e0, tv, e1, ef) = washout_run(esn, data, t_start, T * t_lyap, t_lyap)
                    rows.append(('washout', n_sensors, T, w, n, e0, tv, e1, ef))
                    print(f'{case} washout   {n_sensors:2d} sensors T = {T:5.2f}: data {n:5d}, '
                          f'first error {e0:.3f}, valid {tv:.2f} T_lambda', flush=True)
            for T in lengths:
                n, (e0, tv, e1, ef) = filter_run(esn, data, dims, defaults, t_start, T * t_lyap, c['gamma'][n_sensors],
                                                 t_lyap)
                rows.append(('filter', n_sensors, T, w, n, e0, tv, e1, ef))
                print(f'{case} filter    {n_sensors:2d} sensors T = {T:5.2f}: data {n:5d}, '
                      f'first error {e0:.3f}, valid {tv:.2f} T_lambda', flush=True)
    arr = np.array([r[1:] for r in rows], float)
    np.savez(os.path.join(HERE, 'results', 'synchronization', f'synchronization_{case}{tag}.npz'), method=np.array([r[0] for r in rows]),
             n_sensors=arr[:, 0], length_lyap=arr[:, 1], window=arr[:, 2], n_data=arr[:, 3],
             first_error=arr[:, 4], valid_lyap=arr[:, 5], window_error=arr[:, 6], feed_error=arr[:, 7])
