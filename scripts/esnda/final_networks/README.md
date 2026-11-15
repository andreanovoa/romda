# Trained echo state networks of the washout-free ESN-DA paper

This folder describes the six echo state networks (ESNs) of Appendix A of the paper (`tab:esns`), as trained and deployed in its experiments. The `.npz` files are not in the repository: they come with the data of the paper, which unzipped at the root of romda puts them here, or `build_final_networks.py` exports them from a store that holds the networks. Each file contains the matrices, the normalization and the hyperparameters of one network. The networks are stored with `EchoStateNetwork.to_arrays` of the package `echostatenetwork` (version 0.5.0), so they load without pickle and without romda.

| File | Network | Reservoirs × units | Sites per patch; halo | Spectral radius | Input scaling | Tikhonov factor |
|---|---|---|---|---|---|---|
| `L63.npz` | Lorenz-63, single | 1 × 100 | – | 0.5 | 0.562 | 1e-9 |
| `L96-10s.npz` | Lorenz-96, 10 sites, single | 1 × 500 | – | 0.295 | 0.544 | 1e-9 |
| `L96-10p.npz` | Lorenz-96, 10 sites, parallel | 2 × 500 | 5; 2 | 0.300 | 0.919 | 1e-10 |
| `L96-40s.npz` | Lorenz-96, 40 sites, single | 1 × 6000 | – | 0.211 | 2.23 | 1e-4 |
| `L96-40p.npz` | Lorenz-96, 40 sites, parallel | 8 × 500 | 5; 2 | 0.231 | 0.623 | 1e-10 |
| `L96-10s-parametric.npz` | Lorenz-96, 10 sites, single, trained at 13 forcings, F = 3.5 to 8 | 1 × 1000 | – | 0.2 | 0.557 | 1e-8 |

The parametric network reads the forcing, F, in its readout only, `u = W_out [r; F~; 1]`, where `F~` is the normalized forcing. The column of its input matrix for the forcing is zero (`param_in_reservoir = False`, `readout_input = 'params'`).

`manifest.json` records, for every network, the hyperparameters and the layout, the training record (system, parameters, length, time step, noise and seeds, with the SHA-256 digest of the training array), the entry of the ESN store from which the network was exported, and the romda branch and commit that trained it.

## Loading a network and forecasting in closed loop

The network advances in steps of `dt_ESN`, which is `upsample` times the time step of the training data. The input is the physical state. The reservoir is first driven by data (open loop, the washout) and then by its own output (closed loop).

```python
import numpy as np
from echostatenetwork import EchoStateNetwork

net = EchoStateNetwork.from_arrays(np.load('L96-10s-parametric.npz', allow_pickle=False))
net.input_parameters = np.array([[6.5]])     # parametric network only: the forcing of the forecast

# X: a record of the system, (time, sites), sampled every net.dt_ESN
r = np.zeros((net.N_r, 1))
for x in X[:84]:                             # washout of 84 steps, two Lyapunov times
    u, r = net.step(net.outputs_to_inputs(x[:, None]), r)

forecast = []                                # forecast[k] is the estimate of X[84 + k]
for _ in range(400):
    forecast.append(u[:, 0])
    u, r = net.step(net.outputs_to_inputs(u), r)
```

Every column of `u` and `r` is one realization, so an ensemble is advanced by giving `u` and `r` several columns. The networks without a parameter skip the line `net.input_parameters = ...`.

## Verification

`check_final_networks.py` reloads every file and compares it with the store network: the steps are bit-identical, and the valid time on the test windows, computed with the functions of the paper, equals that of Appendix A (2.22, 2.71, 3.69, 0.72 and 3.65 Lyapunov times for L63, L96-10s, L96-10p, L96-40s and L96-40p) and, for the parametric network, 4.65 Lyapunov times at the unseen forcing F = 6.5. `build_final_networks.py` writes the files from the store, and `networks.py` resolves each network to its store entry through the call that the paper's scripts make.

## Checksums (SHA-256)

```
2ed53dd130e613ba671de58e78d8afa71b55626622957e635565ec5fbaf2d794  L63.npz
7b371e1958fc81048d813851b44850508ba65ff2f2dda4d815d8b936f8708dc8  L96-10s.npz
12e22a7e2e87697f98c335e1bf0eca1f99aaa7a2cd83a27318938bd1d06ccd47  L96-10p.npz
a11eee19c720730d11f8208ea219f8cc4ccc5203192bb910686a600f2df281db  L96-40s.npz
090eab3e04b297dc840836dd6f2056d8e696bad9a674da5de0d90d6629493635  L96-40p.npz
20a232fec19badb646c50b9841c8c32be151b330fe0c3be5a0714ae0fa31f5b6  L96-10s-parametric.npz
8ce2dc5814ba5cbcfb493f5e02849daeca65eac01a99f9ed0d583d9b34e80ef1  manifest.json
```

`SHA256SUMS` lists the same digests (`shasum -a 256 -c SHA256SUMS`).
