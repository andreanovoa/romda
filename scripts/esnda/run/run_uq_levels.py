"""Spread against error for one level of the assumed observation error (Fig. `uncertainty` of the washout-free
paper): the runs of run_lorenz96_small.py uq (L96-10s) and run_parallel_uq10.py (L96-10p) with the innovation-based
inflation, over N seeds or windows (default 40) instead of 5 and 10, and Lorenz-63 with the protocol of these runs.
Every component is observed unless Q is set.

    CDD_STD=0.1 [Q=1|5] python run_uq_levels.py l63|10s|10p
        -> results/uncertainty/uq_levels_<case>_cdd<CDD_STD>[_q<Q>].npz (error, spread)
"""
import multiprocessing
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # scripts/esnda: configs/, results/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

N = int(os.environ.get('N', 40))
Q = os.environ.get('Q')     # partial observations: Q=1 (Lorenz-63: y alone), Q=5 (ten sites: every other site)
CASE = sys.argv[1]
if CASE == '10s':
    import run_lorenz96_small as R  # noqa: E402

    def job(seed):
        r = R.experiment(n_measured=int(Q or 10), m=R.M_UQ, seed=seed, inflation='innovation', ddof=1)
        return r['rmse_mean_sync'], r['spread_sync']
elif CASE == 'l63':    # Lorenz-63 with the protocol of the ten-site runs: N_AN analyses, 3 network steps apart
    from romda.run import innovation_options, load_config  # noqa: E402
    from romda.data_assimilation import run_and_score  # noqa: E402
    from romda.models.data_driven.rnns import model_esn  # noqa: E402
    from romda.models.physical import Lorenz63  # noqa: E402
    D = load_config(os.path.join(HERE, 'configs', 'lorenz63.yml'))
    M63, NT_OBS, N_AN = 10, 6, 210     # members (Table 2); record steps between observations (0.11 T_lambda); analyses

    def job(seed):
        measured = [int(c) for c in (Q or '012')]
        esn, data = model_esn(Lorenz63, D['truth'], D['forecast']['esn'], measured=measured, bayesian_update=True)
        t_data = N_AN * NT_OBS * data['dt'] / data['t_lyap']
        r = run_and_score(esn, data, measured, seed, D['da']['da_method'], M63, NT_OBS, D['da']['std_obs'],
                          D['da']['inflation'], window_lyap=(2., 2. + t_data, 4.), seed_shift_lyap=3.5,
                          cdd_fraction=float(os.environ.get('CDD_STD', D['da']['cdd_fraction'])), **innovation_options(D['da']),
                          bayesian_update=True, adaptive_inflation=D['da']['adaptive_inflation'],
                          inflate_parameters=D['da']['inflate_parameters'], ddof=1)
        return r['rmse_mean_sync'], r['spread_sync']
else:
    import run_parallel_uq10 as R  # noqa: E402

    def job(window):
        row = R.job((int(Q or 10), window, 'innovation'))
        return row[5], row[6]

if __name__ == '__main__':
    multiprocessing.set_start_method('fork', force=True)
    # a fresh process per run: a run of L96-10s holds about 7 GB, which the worker would keep
    with multiprocessing.Pool(int(os.environ.get('N_WORKERS', 1)), maxtasksperchild=1) as pool:
        rows = np.array(pool.map(job, range(N), chunksize=1), float)
    out = os.path.join(HERE, 'results', 'uncertainty', f"uq_levels_{CASE}_cdd{os.environ['CDD_STD']}{'_q' + Q if Q else ''}.npz")
    np.savez(out, error=rows[:, 0], spread=rows[:, 1])
    print('wrote', out, '| median error', np.median(rows[:, 0]), 'spread', np.median(rows[:, 1]))
