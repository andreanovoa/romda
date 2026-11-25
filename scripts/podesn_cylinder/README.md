# Cylinder wake: data assimilation with a POD-ESN

## What the example shows

Real-time data assimilation with a data-driven reduced-order model, when no model of the truth is
available. The flow is the two-dimensional wake behind a circular cylinder at a Reynolds number of 100.

1. A POD-ESN reduced-order model is trained on a short record of noisy snapshots: proper orthogonal
   decomposition (POD) projects the flow onto a few modes, and an echo state network (ESN) forecasts
   their amplitudes.
2. Sparse sensors are placed on the flow field by QR pivoting, and the held-out snapshots provide the
   observations.
3. The square-root EnKF assimilates the observations into the reduced-order model in real time, first
   estimating the state only, and then estimating the state together with the singular values of the
   ESN output weights (`Wout`), so that the model is updated online.

The sweep, around the defaults of the `experiment` function of `sweeps.py`, shows how the filter responds to the ensemble size, the time between observations, the
number of sensors, the covariance inflation and the online estimation of the output weights.

## Published works

- Nóvoa & Magri (2025). Online model learning with data-assimilated reservoir computers. In
  *Computational Science – ICCS 2025 Workshops*, Lecture Notes in Computer Science 15910, 57–64.
  [doi:10.1007/978-3-031-97567-7_5](https://doi.org/10.1007/978-3-031-97567-7_5),
  [arXiv:2504.16767](https://arxiv.org/abs/2504.16767). The code of the paper is the release
  [v1.1](https://github.com/andreanovoa/romda/releases/tag/v1.1); `iccs2025.py` replicates its Figs. 3–5 with
  the current code (see [Replication of the paper](#replication-of-the-paper)).

This case accompanies the tutorials `31_POD_SPOD.ipynb`, `35_POD_ESN_cylinder.ipynb` and
`42_DA_POD_ESN_cylinder.ipynb` (topics 3 and 4 of the [tutorials](../../../tutorials/README.md)).

## Data

The wake snapshots of Zenodo record [15623774](https://zenodo.org/records/15623774) (`circle_re_100.mat`).
The first run downloads them into `data/wakes/` at the repository root, adds the noise of `pod_esn.yml`
once, and caches the noisy record under `results/wakes/` at the repository root.

## How to run

From the repository root:

```bash
python scripts/podesn_cylinder/pod_esn.py --save-figs
```

The trained POD-ESN is cached next to the noisy record, so later runs, and the sweep, reload it. Edit
`pod_esn.yml` to change the record, the reduced-order model or the filter; this case runs only through
its script, because its truth is a set of snapshots, not a model.

The sweep takes `--study <name>`, and `--quick` runs two values of each setting as a smoke test:

```bash
python scripts/podesn_cylinder/sweeps.py --study sensors --quick
```

| Study | Setting varied | What to look for |
|---|---|---|
| `ensemble_size` | members `m` | the smallest ensemble within reach of the saturated error |
| `assimilation_frequency` | time steps between observations `Nt_obs`, as fractions of a shedding period | how sparse in time the data can be for a periodic flow |
| `sensors` | number of point sensors placed by QR pivoting | a sharp improvement once there are about as many well-placed sensors as POD modes |
| `inflation` | multiplicative inflation factor | the periodic wake keeps the spread of the ensemble small, and inflation counteracts its collapse |
| `wout_estimation` | state only against state and `Wout`, over the prior uncertainty of the weights `std_alpha` | whether the online update of the output weights compensates for the noisy training data |

## Replication of the paper

`iccs2025.py` makes Figs. 3–5 of Nóvoa & Magri (2025) with the settings of `iccs2025.yml`:

```bash
python scripts/podesn_cylinder/iccs2025.py --save-figs
```

| Setting | Value | Source |
|---|---|---|
| Noise of the snapshots | Gaussian, standard deviation 0.1 of the maximum velocity, smoothed by a Gaussian kernel of standard deviation 0.1 | paper |
| POD modes, reservoir size | 4, 40 | paper |
| Training snapshots | 250 (Figs. 3–4) and 90 (Fig. 5); the last tenth validates the network | paper; v1.1 |
| Spectral radius, input scaling | 0.976, 0.890, fixed; the Tikhonov factor is searched | paper |
| Filter, ensemble size | EnSRKF; 10 (Figs. 3–4) and 50 (Fig. 5) | paper; v1.1 for Fig. 4 |
| Time steps between observations | 25 (Figs. 3–4); 10, 20 and 30 (Fig. 5) | paper; v1.1 for Fig. 4 |
| Sensors | 1 to 4 by QR pivoting in the wake, $4 \leq x \leq 7$, $|y| \leq 1$, and the full fields through their POD coefficients | paper; v1.1 for the region |
| Observation error, prior of the singular values of `Wout` | 0.01, 0.1 | v1.1 |
| Assimilation window | from $t = 0.5$ to $t = 5$, then a forecast of 0.5 | v1.1 |

The error is the mean squared error of the reconstructed velocity fields with respect to the noise-free fields
(Eq. 5 of the paper). The conclusions of the paper hold with the current code.

1. Fig. 3: with one sensor, the two-fold estimation (POD coefficients and reservoir state) brings the error below
   that of the noisy data at the first observation, and one order of magnitude below it within four observations.
   The estimation of the POD coefficients alone needs about eight observations to stay below the error of the data.
2. Fig. 4: with the full fields, the error falls to its final level at the first observation. With one to four
   sensors, the ensemble converges to the same level within five observations.
3. Fig. 5: with the network trained on 90 snapshots, the two-fold estimation does not prevent the forecast from
   drifting, and the final error exceeds that of the noisy data in 4 of the 15 cases. The three-fold estimation, which
   also updates the singular values of `Wout`, keeps the final error below that of the data in the 15 cases.

The curves are not those of the paper digit for digit, for three reasons: (i) the random matrices of the network
and the realization of the noise differ from those of v1.1; (ii) the members start from reservoir states visited in
training (`ESN_model.init_ensemble`), whereas v1.1 perturbed the state after a transient; and (iii) the stride of
the candidate sensor locations follows the v1.1 call, whose axis order is assumed.

## Outputs

In this folder, not tracked by git:

- `figs/`: the figures of the case (`cylinder_01.png`, ...), of the sweep (`cylinder_<study>.png`) and of the
  replication (`iccs2025_fig3.png`, ...);
- `results/`: the filtered ensembles of both runs (`cylinder_main_results`), the records of the sweep
  (`cylinder_<study>.jsonl`) and the errors of each run of the replication (`iccs2025/`).
