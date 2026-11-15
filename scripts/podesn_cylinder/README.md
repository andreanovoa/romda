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
  [v1.1](https://github.com/andreanovoa/romda/releases/tag/v1.1).

This case accompanies the tutorials `31_POD_SPOD.ipynb`, `35_POD_ESN_cylinder.ipynb` and
`44_DA_POD_ESN_cylinder.ipynb` (topics 3 and 4 of the [tutorials](../../../tutorials/README.md)).

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

## Outputs

In this folder, not tracked by git:

- `figs/`: the figures of the case (`cylinder_01.png`, ...) and of the sweep (`cylinder_<study>.png`);
- `results/`: the filtered ensembles of both runs (`cylinder_main_results`) and the records of the sweep
  (`cylinder_<study>.jsonl`).
