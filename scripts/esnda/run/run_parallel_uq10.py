"""Sect. 4.4 of the washout-free paper: the inflation that the ten-site Lorenz-96 ring needs depends on the model
error of the network. A parallel network of two reservoirs of 500 units (patches of five sites, a halo of two sites;
NX=10 G_SITES=5 HALO=2 N_R=500 of run_parallel_localization.py), the global filter with m = 64, observed every
0.095 T_lambda for 20 T_lambda, with the innovation-based inflation and without (rho_I = 1), on ten windows.
Scores after the synchronization transient (2 T_lambda to the last observation): error and spread of the ensemble
mean, normalized as the error, and coverage of the 2-sigma band, as Eq. (eq:error_spread) of the paper; plus the
mean sqrt(lambda) and the valid time. Writes results/uncertainty/uq_parallel10.npz.

    python run_parallel_uq10.py        (from this folder; trains nothing: run_parallel_localization.py train first)
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

os.environ.update(NX='10', G_SITES='5', HALO='2', N_R='500')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import run_parallel_localization as p  # noqa: E402

M, NT_OBS, T = 64, 4, 20.0


def job(j):
    n_sensors, w, gamma = j
    row, est = p.filter_run((n_sensors, w, M, None, gamma, NT_OBS, T), return_filter=True)
    net, X_rec, _, steps_lyap, norm = p.network()
    H = est.model.hist[::net.upsample, :p.NX]                    # (network steps, sites, members), from s0
    s0 = p.window_start(w, steps_lyap)
    steps = s0 + np.arange(H.shape[0])
    sel = steps >= s0 + 2 * steps_lyap
    U, X = H[sel], X_rec[steps[sel]]
    mean, std = U.mean(-1), U.std(-1, ddof=1)
    err = np.sqrt(np.mean(np.sum((mean - X) ** 2, 1))) / norm
    spread = np.sqrt(np.mean(np.sum(std ** 2, 1))) / norm
    cover = np.mean(np.abs(mean - X) <= 2 * np.sqrt((M - 1) / M) * std)
    t_last = s0 + int(round(T * steps_lyap / NT_OBS)) * NT_OBS
    last = steps > t_last - steps_lyap                                    # e_W: the last T_lambda of the data
    Ul, Xl = H[last], X_rec[steps[last]]
    err_last = np.sqrt(np.mean(np.sum((Ul.mean(-1) - Xl) ** 2, 1))) / norm
    spread_last = np.sqrt(np.mean(np.sum(Ul.var(-1, ddof=1), 1))) / norm
    return (n_sensors, w, 0.0 if gamma == 'innovation' else 1.0, row[4], row[7], err, spread, cover, row[6], row[9],
            err_last, spread_last)


if __name__ == '__main__':
    jobs = [(s, w, g) for s in (10, 5) for g in ('innovation', 1.0) for w in range(10)]
    # N_WORKERS: explicit, small by default; a worker of the L96-40 sweeps can hold tens of GB
    with ProcessPoolExecutor(max_workers=int(os.environ.get('N_WORKERS', 2))) as ex:
        rows = np.array(list(ex.map(job, jobs)), float)
    np.savez(os.path.join(p.HERE, 'results', 'uncertainty', 'uq_parallel10.npz'), rows=rows,
             columns=np.array(['sensors', 'window', 'fixed', 'factor', 'valid', 'error', 'spread', 'coverage', 'first', 'last', 'error_last', 'spread_last']))
    # error/spread/coverage: 2 T_lambda to the last datum; last: e_W of filter_run; error_last/spread_last: the
    # error and spread of the stored ensemble over the last T_lambda of the data (e_W, s_W)
    for s in (10, 5):
        for g in (0.0, 1.0):
            q = rows[(rows[:, 0] == s) & (rows[:, 2] == g)]
            print(f"{s} sites, {'innovation' if g == 0 else 'rho_I = 1'}: sqrt(lambda) {np.median(q[:, 3]):.3f} "
                  f"({q[:, 3].min():.3f}-{q[:, 3].max():.3f}), error {np.median(q[:, 5]):.3f}, spread {np.median(q[:, 6]):.3f}, "
                  f"coverage {np.median(q[:, 7]):.2f}, valid {np.median(q[:, 4]):.2f}, e_W {np.median(q[:, 9]):.3f} ({np.median(q[:, 10]):.3f}), spread_last {np.median(q[:, 11]):.3f}, "
                  f"fail (e_W > 0.5) {int(np.sum(~(q[:, 9] <= 0.5)))}/10")
