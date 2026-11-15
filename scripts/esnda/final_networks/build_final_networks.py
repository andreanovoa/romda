"""Write the six trained networks of the washout-free ESN-DA paper (networks.py) as one npz each, through the
package's own EchoStateNetwork.to_arrays, and manifest.json with what produced them.

    python build_final_networks.py      -> final_networks/<short name>.npz, manifest.json, SHA256SUMS
"""

import datetime
import hashlib
import json
import os
import subprocess

import echostatenetwork
import numpy as np

import networks as N

HP = ('N_units', 'N_r', 'N_dim', 'N_wash', 'upsample', 'dt', 'dt_ESN', 'rho', 'sigma_in', 'sigma_param', 'tikh',
      'leak_rate', 'connect', 'Win_type', 'bias_in', 'bias_out', 'noise', 'norm_method', 'seed')
LAYOUT = ('patch_size', 'halo', 'periodic', 'shared', 'readout_input', 'param_in_reservoir', 'observed_idx')
PAPER_STORE = os.environ.get('PAPER_STORE', '')   # a second ESN store, searched first (the paper's networks were
# trained in two working trees); empty: the store of this repository alone


def plain(v):
    """JSON value of an attribute: arrays of one value as a number, other arrays as lists."""
    if v is None or isinstance(v, (bool, str)):
        return v
    a = np.asarray(v)
    if a.dtype == object:
        return str(v)
    if a.size == 1:
        x = a.ravel()[0]
        return bool(x) if a.dtype == bool else (int(x) if np.issubdtype(a.dtype, np.integer) else float(x))
    return a.tolist()


def git(*args, cwd=None):
    return subprocess.run(['git', *args], cwd=cwd or N.HERE, capture_output=True, text=True).stdout.strip()


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    entries = []
    for name in N.NAMES:
        esn, info, _ = N.load(name)
        arrays = esn.to_arrays()
        path = os.path.join(N.HERE, f'{name}.npz')
        np.savez(path, **arrays)
        h = info['store_hash']
        store_dir = os.path.join(PAPER_STORE if os.path.isdir(os.path.join(PAPER_STORE, h)) else info['store'], h)
        entry = dict(
            short_name=name, file=f'{name}.npz', bytes=os.path.getsize(path), sha256=sha256(path),
            hyperparameters={k: plain(getattr(esn, k, None)) for k in HP},
            layout={k: plain(getattr(esn, k, None)) for k in LAYOUT},
            parameters=None if esn.input_parameters is None else dict(
                names=list(getattr(esn, 'param_names', ()) or ()), training_values=plain(esn.input_parameters),
                shift=plain(np.ravel(esn.shift)[esn.N_dim:]), norm=plain(np.ravel(esn.norm)[esn.N_dim:])),
            training_record=info['record'],
            store=dict(hash=h, entry=f'results/esn_configs/{h}',   # relative to the root of romda
                       saved=datetime.datetime.fromtimestamp(os.path.getmtime(store_dir)).isoformat(timespec='seconds')),
        )
        entries.append(entry)
        print(f'{name}: {h}, {entry["bytes"] / 1e6:.1f} MB', flush=True)
    worktree = os.path.dirname(os.path.dirname(N.ESN_DA))
    paper_tree = os.path.dirname(os.path.dirname(PAPER_STORE)) if PAPER_STORE else worktree
    manifest = dict(
        paper='Washout-free synchronization, parameter estimation and uncertainty quantification of echo state networks (manuscript.tex, Table A.1)',
        echostatenetwork=echostatenetwork.__version__,
        romda=dict(written_by=dict(branch=git('rev-parse', '--abbrev-ref', 'HEAD', cwd=worktree),
                                   commit=git('rev-parse', 'HEAD', cwd=worktree)),
                   paper_networks_trained_on=dict(branch=git('rev-parse', '--abbrev-ref', 'HEAD', cwd=paper_tree),
                                                  commit=git('rev-parse', 'HEAD', cwd=paper_tree)),
                   # the single readout-forcing network (run_parametric.py, formerly run_single_parametric_readout.py, 1000 units), trained on
                   # 13 Nov 2026 from the working tree of this branch; the script was committed afterwards, in 86240b5
                   parametric_network_trained_on=dict(branch='readout-parameter',
                                                      script='scripts/esnda/run/run_single_parametric_readout.py',
                                                      commit=git('rev-parse', 'HEAD', cwd=worktree),
                                                      script_committed_in='86240b591f419bdf85abca924bd5daf0e292cd93')),
        load='EchoStateNetwork.from_arrays(np.load(<file>, allow_pickle=False))',
        networks=entries)
    with open(os.path.join(N.HERE, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1)
    with open(os.path.join(N.HERE, 'SHA256SUMS'), 'w') as f:
        for e in entries:
            f.write(f'{e["sha256"]}  {e["file"]}\n')
        f.write(f'{sha256(os.path.join(N.HERE, "manifest.json"))}  manifest.json\n')
    print('wrote manifest.json and SHA256SUMS')


if __name__ == '__main__':
    main()
