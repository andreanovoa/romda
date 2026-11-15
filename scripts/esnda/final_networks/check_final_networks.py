"""Check the bundle of final networks (build_final_networks.py) against the ESN store and the paper.

For each <short name>.npz, reloaded with EchoStateNetwork.from_arrays:
  (a) its steps are bit-identical to those of the store network (networks.load): 51 closed-loop steps from 16 random
      states;
  (b) the readout-forcing L96-10p keeps param_in_reservoir False, readout_input 'params' and no forcing column in W_in;
  (c) the valid time of the reloaded network, computed with the paper's own functions on its test windows (a washout
      of 2 T_lambda at the start of 10 test windows, as run_verification.py), equals that of the store network and
      the median of Appendix A (results/verification/esn_verification.npz of the paper's branch); the parametric
      network is scored at the unseen F = 6.5 (run_parametric.valid_times, 10 windows).

    python check_final_networks.py
"""

import os
import sys

import numpy as np
from echostatenetwork import EchoStateNetwork

import networks as N

#: Appendix A of the paper: median valid time (T_lambda) of the 10 test windows (esn_verification.npz)
APPENDIX_A = {'L63': 2.22, 'L96-10s': 2.71, 'L96-10p': 3.69, 'L96-40p': 3.65, 'L96-40s': 0.72}   # 13 Nov 2026: L63 and
#: L96-10s retrained (ranges of lorenz96_parallel.yml; L96-10s on the 400 T_lambda record of L96-10p), L96-10p scored on that record


def identical_steps(store, npz, seed=0):
    rng = np.random.default_rng(seed)
    m = 16
    shift, norm = np.ravel(store.shift), np.ravel(store.norm)
    u = shift[:store.N_dim, None] + 0.3 * norm[:store.N_dim, None] * rng.standard_normal((store.N_dim, m))
    r = np.tanh(rng.standard_normal((store.N_r, m)))
    F = None if store.input_parameters is None else rng.uniform(3.5, 8.0, (1, m))
    (Y1, _, r1), (Y2, _, r2) = (net.forecast(u, r, 51, input_parameters=F) for net in (store, npz))
    return max(np.abs(Y1 - Y2).max(), np.abs(r1 - r2).max())


def valid_time(name, net, ns):
    """Median valid time of the 10 test windows with the paper's functions, net in place of the store network."""
    if name in N.SYNC_CASE:
        case, n = N.SYNC_CASE[name]
        _, data, _, _ = ns['network'](case, n)
        t_lyap = data['t_lyap']
        return float(np.median([ns['washout_run'](net, data, ns['test_window_time'](w, t_lyap), 2.0 * t_lyap, t_lyap)[1][1]
                                for w in range(10)]))
    if name == 'L96-10s-parametric':
        import run_parametric as R
        original = R.network
        R.network = lambda variant='readout', parallel=False: (net, *original('readout')[1:])
        try:
            return float(np.median(R.valid_times('readout', 6.5)))
        finally:
            R.network = original
    _, X, Y, S, norm = ns['network']()
    patched = lambda: (net, X, Y, S, norm)
    ns['network'] = patched
    if name == 'L96-10p':          # run_verification.on_record: the test record and windows of L96-10s
        sync = N.run_script('run_synchronization.py')
        _, data, _, _ = sync['network']('lorenz96', 10)
        noisy, clean, up, t_lyap = data['noisy_data'], data['clean_data'], net.upsample, data['t_lyap']
        from romda.metrics import error_scale
        norm, n_w, valid = error_scale(clean[data['i_test']:]), 2 * S, []
        for w in range(10):
            i0 = data['i_test'] + int(round(sync['test_window_time'](w, t_lyap) / data['dt']))
            U, R_ = net.open_loop(noisy[i0::up][:n_w])
            psi = np.concatenate([U[-1], R_[-1]])[:, None]
            valid.append(ns['valid_times'](psi, clean[i0::up], np.array([n_w]), norm, S)[0])
        return float(np.median(valid))
    return float(np.median([ns['washout_run']((w, 2.0))[5] for w in range(10)]))


def main():
    ok = True
    for name in N.NAMES:
        store, info, ns = N.load(name)
        npz = EchoStateNetwork.from_arrays(np.load(os.path.join(N.HERE, f'{name}.npz'), allow_pickle=False))
        if not hasattr(npz, 'dt'):         # the step of the data, which the paper's functions read: from_arrays keeps
            npz.dt = npz.dt_ESN / npz.upsample   # dt_ESN and upsample
        d = identical_steps(store, npz)
        line = [f'{name:19s} store {info["store_hash"]}: (a) max |step difference| {d:.1e}']
        ok &= d == 0.0
        if name == 'L96-10s-parametric':
            Win = npz.Win
            col = Win[:, -2]                    # columns: the ten sites, F, the input bias
            col = col.toarray() if hasattr(col, 'toarray') else col
            no_F = not np.any(col)
            line.append(f'(b) param_in_reservoir {npz.param_in_reservoir}, readout_input {npz.readout_input!r}, '
                        f'F column of W_in zero: {no_F}')
            ok &= (npz.param_in_reservoir is False or npz.param_in_reservoir == False) and no_F \
                and npz.readout_input == 'params'
        v_npz = valid_time(name, npz, ns)
        store2, _, ns2 = N.load(name)
        v_store = valid_time(name, store2, ns2)
        ref = APPENDIX_A.get(name, 4.65)   # parametric: the 10-window median at F = 6.5 of run_parametric.py train
        line.append(f'(c) valid time {v_npz:.2f} T_lambda (store {v_store:.2f}, paper {ref:.2f})')
        ok &= v_npz == v_store and abs(v_npz - ref) < 0.01
        print(' | '.join(line), flush=True)
    print('ALL CHECKS PASS' if ok else 'SOME CHECKS FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
