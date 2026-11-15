"""The six trained networks of the washout-free ESN-DA paper (Table 1), loaded from the ESN store exactly as the
paper's scripts load them, never trained.

    L63, L96-10s        run_synchronization.network(case, n_sensors) with all components measured (the full-state
                        networks; configs/lorenz63.yml, configs/lorenz96_small.yml)
    L96-10p             run_parallel_localization.network() with NX=10 G_SITES=5 HALO=2 N_R=500
    L96-40s             run_parallel_localization.network() with G_SITES=40 HALO=0 N_R=6000 TRAIN_LYAP=800 (Tikhonov
                        factor fixed at 1e-4, SINGLE_TIKH_RANGE)
    L96-40p             run_parallel_localization.network() with its defaults (NX=40 G_SITES=5 HALO=2 N_R=500)
    L96-10s-parametric  run_parametric.network('readout'): one reservoir of 1000 units with the forcing in the readout
                        only (configs/lorenz96_parametric.yml, tikh 1e-8)

`load(name)` returns the network, the query hash of the store entry it came from, the training record (the array
handed to the store and its key) and the namespace of the script, for the checks. The store hash is the one that the
script's own call computes: `romda.models.data_driven.rnns.builders.auto_load_or_create` is wrapped to record it, and `ESN_model.train` raises,
so a network that the store does not hold is reported, not retrained.
"""

import hashlib
import os
import sys
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ESN_DA = os.path.dirname(HERE)
RUN = os.path.join(ESN_DA, 'run')
for p in (ESN_DA, RUN):
    if p not in sys.path:
        sys.path.insert(0, p)

import romda.models.data_driven.rnns.builders as build  # noqa: E402
from romda.models.data_driven.rnns import ESN_model  # noqa: E402
from romda.models.data_driven.rnns.esn_config import ESN_STORE, ESNConfig, find_matching_config  # noqa: E402
from run_verification import run_script  # noqa: E402  (any run/ script, env set)

NAMES = ('L63', 'L96-10s', 'L96-10p', 'L96-40s', 'L96-40p', 'L96-10s-parametric')
PARALLEL_ENV = {'L96-10p': dict(NX=10, G_SITES=5, HALO=2, N_R=500),
                'L96-40s': dict(G_SITES=40, HALO=0, N_R=6000, TRAIN_LYAP=800),
                'L96-40p': {}}
SYNC_CASE = {'L63': ('lorenz63', 3), 'L96-10s': ('lorenz96', 10)}


def load(name):
    """(network, info, namespace): info holds the store hash, the store folder, the training record (its key, shape
    and sha256) and, for model_esn networks, the dataset entry."""
    seen = {}
    original = build.auto_load_or_create

    def recorded(config_dir=ESN_STORE, data=None, **kwargs):
        query = ESNConfig.from_init_params(**{**kwargs, 'data': data}).to_hash()
        if not find_matching_config(config_dir, query):
            raise LookupError(f'{name}: no store entry {query} in {config_dir}')
        seen.update(hash=query, store=str(config_dir), data=data,
                    training_data_filename=kwargs.get('training_data_filename'))
        return original(config_dir=config_dir, data=data, **kwargs)

    with mock.patch.object(build, 'auto_load_or_create', recorded), \
            mock.patch.object(ESN_model, 'train', side_effect=RuntimeError('the paper network is not in the store')):
        if name in SYNC_CASE:
            ns = run_script('run_synchronization.py')
            case, n = SYNC_CASE[name]
            esn, data, _, defaults = ns['network'](case, n)
            truth = defaults['truth']                # the dataset of the manifest: the record options and the parameters
            dataset = dict(noise_level=truth['noise_level'], num_lyap_times=truth['num_lyap_times'],
                           split_lyap=tuple(truth['split_lyap']), **truth['params'])
            record = dict(dataset=dataset, dt=float(data['dt']), t_lyap=float(data['t_lyap']),
                          split_steps=[int(s) for s in data['split']])
        elif name in PARALLEL_ENV:
            ns = run_script('run_parallel_localization.py', **PARALLEL_ENV[name])
            esn = ns['network']()[0]
            cfg = ns['load_config'](ns['CONFIG'])
            truth = cfg['truth']
            split = (truth['split_lyap'][0], ns['TRAIN_LYAP'], *truth['split_lyap'][2:])
            record = dict(dataset={**truth['params'], 'Nx': ns['NX'], 'seed': truth['seed'],
                                   'noise_level': truth['noise_level'],
                                   'num_lyap_times': truth['num_lyap_times'] - sum(truth['split_lyap']) + sum(split),
                                   'split_lyap': list(split)}, config=os.path.relpath(ns['CONFIG'], ESN_DA))
        else:
            import run_parametric as ns_mod
            ns = vars(ns_mod)
            ns_mod.network.cache_clear()    # a fresh load from the store, recorded
            esn, regimes, segments, _ = ns_mod.network('readout')
            cfg = ns_mod.load_config(ns_mod.CONFIG)
            tr = cfg['forecast']['train']
            record = dict(system=cfg['truth']['model'], params=dict(cfg['truth']['params']), parameter=tr['param'],
                          values=list(tr['values']), dt=tr['dt'], noise_level=cfg['truth']['noise_level'],
                          segment_seeds=list(range(len(tr['values']))), noise_seed=0,
                          t_train_lyap=tr['t_train_lyap'], t_val_lyap=tr['t_val_lyap'], t_test_lyap=tr['t_test_lyap'],
                          segments_shape=list(segments.shape), config=os.path.relpath(ns_mod.CONFIG, ESN_DA))
    data = seen['data']
    record.update(training_data_filename=seen['training_data_filename'], training_array_shape=list(data.shape),
                  training_array_sha256=hashlib.sha256(data.astype('float64').tobytes()).hexdigest())
    return esn, dict(store_hash=seen['hash'], store=seen['store'], record=record), ns
