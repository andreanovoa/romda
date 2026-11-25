# Real-time data assimilation with the CAE-ESN

## What the example shows

Real-time data assimilation with a convolutional autoencoder echo state network (CAE-ESN), a data-driven
reduced-order model of a chaotic field. A convolutional autoencoder compresses each snapshot into a few reduced
coordinates, and an echo state network (ESN) forecasts them in time.

1. The autoencoder is trained beforehand: its weights are loaded from `tutorials/weights/`.
2. The ESN is trained on the reduced coordinates of the training snapshots, with the hyperparameters of the paper.
3. An ensemble of CAE-ESNs starts from reservoir states visited during training, unrelated to the true state, and
   the EnKF assimilates sparse, noisy measurements of the test snapshots. The observation operator is the decoder,
   read at the sensors.
4. After the last measurement, the ensemble forecasts in closed loop.

The script prints the error of the decoded ensemble mean and the ensemble spread, normalized by the
root-mean-square of the true field, over the assimilation window and the forecast.

| Configuration | Case | Autoencoder | Data |
|---|---|---|---|
| `ks.yml` | Kuramoto–Sivashinsky equation, $L = 20\pi$, 128 points, $\Delta t = 0.25$ | `CAE1D`, the weights of the paper | integrated by the script with dynamodels (`KS`, ETDRK4) |
| `kolmogorov.yml` | Kolmogorov flow, $Re = 30$, $48 \times 48$, vorticity, $\Delta t = 0.1$ | `MultiScaleCAE`, trained in romda on this record with the settings of the paper | a jax-cfd record, not distributed with romda |

## Published works

- Özalp, Nóvoa & Magri (2026). Real-time forecasting of chaotic dynamics from sparse data and autoencoders.
  *Computer Methods in Applied Mechanics and Engineering* 450, 118600.
  [doi:10.1016/j.cma.2025.118600](https://doi.org/10.1016/j.cma.2025.118600). The code of the paper is
  [github.com/MagriLab/DA-CAE-ESN](https://github.com/MagriLab/DA-CAE-ESN).
- Racca, Doan & Magri (2023). Predicting turbulent dynamics with the convolutional autoencoder echo state network.
  *Journal of Fluid Mechanics* 975, A2. The multi-scale autoencoder of the Kolmogorov flow.

This folder accompanies the tutorials `43_DA_CAE_ESN_KS.ipynb` and `44_DA_CAE_ESN_Kolmogorov.ipynb` (topic 4 of
the [tutorials](../../tutorials/README.md)), which explain each step.

## Data

- `ks.yml`: the script integrates the record, so anyone can run this case.
- `kolmogorov.yml`: the record was computed with jax-cfd and is not distributed with romda. The script reads it
  from `$DATA_ROOT/Kolmogorov/jax-cfd/Kolmogorov_Re30_drag0.0_jaxcfd_res48.h5` (HDF5, dataset `vorticity`,
  snapshots $\Delta t = 0.1$ apart); compute a record of the same flow with jax-cfd to run this case.

Both cases need torch, the optional extra `romda[torch]`, and the Kolmogorov case needs `h5py`.

## How to run

From the repository root:

```bash
python scripts/caeesnda/cae_esn_da.py ks --save-figs
DATA_ROOT=/path/to/data python scripts/caeesnda/cae_esn_da.py kolmogorov --save-figs
python scripts/caeesnda/cae_esn_da.py ks --set da.m=50 --set truth.Nt_obs=25    # override settings
```

The CAE-ESN is trained on snapshots, not built from a model of romda, so these cases run only through
`cae_esn_da.py`, not through `run_config`. Edit the configuration, or pass `--set key=value`, to change the
record, the ESN, the sensors or the filter.

## Outputs

In this folder, not tracked by git:

- `figs/`: the error and spread of the ensemble against time (`<case>_error.png`);
- `results/`: the configuration and the scores of each run (`<case>.json`).
