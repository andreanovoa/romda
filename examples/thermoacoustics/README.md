# Thermoacoustics

## What the example shows

Real-time data assimilation of thermoacoustic oscillations, the self-excited acoustic waves driven by
unsteady heat release in combustors.

- **`rijke`**: a twin experiment on the Rijke tube, a Galerkin low-order model of longitudinal
  thermoacoustics. The heat-release gain `beta` and the flame time delay `tau` are estimated together
  with the acoustic state.
- **`vdp`**: a twin experiment on the Van der Pol oscillator, the simplest model of a thermoacoustic
  limit cycle, estimating the parameters `beta` and `zeta`.
- **`annular`**: the real-time digital twin of an annular combustor. The truth is raw experimental
  pressure data, the forecast model is a low-order model of azimuthal thermoacoustics, and an echo state
  network estimates the model bias, so that the regularized bias-aware EnKF (rBA-EnKF) assimilates the
  raw data. The script compares the bias-aware filter with the bias-blind limit (regularization zero).

The sweep of the annular case shows how the bias-aware filter responds to its regularization, to the
ensemble size, to the time between observations and to the operating condition.

## Published works

- Nóvoa, Noiray, Dawson & Magri (2024). A real-time digital twin of azimuthal thermoacoustic
  instabilities. *Journal of Fluid Mechanics*. [doi:10.1017/jfm.2024.1052](https://doi.org/10.1017/jfm.2024.1052).
  The `annular` case; the code of the paper is the release
  [v1.0](https://github.com/andreanovoa/real-time-bias-aware-DA/releases/tag/v1.0).
- Nóvoa, Racca & Magri (2024). Inferring unknown unknowns: regularized bias-aware ensemble Kalman filter.
  *Computer Methods in Applied Mechanics and Engineering* 418, 116502.
  [doi:10.1016/j.cma.2023.116502](https://doi.org/10.1016/j.cma.2023.116502). The rBA-EnKF and the Rijke
  tube ([legacy repository](https://github.com/MagriLab/rBA-EnKF)).
- Nóvoa & Magri (2022). Real-time thermoacoustic data assimilation. *Journal of Fluid Mechanics*.
  [doi:10.1017/jfm.2022.653](https://doi.org/10.1017/jfm.2022.653)
  ([legacy repository](https://github.com/MagriLab/Real-time-TA-DA)).
- Nóvoa & Magri (2022). Bias-aware thermoacoustic data assimilation. *51st International Congress and
  Exposition on Noise Control Engineering*
  ([legacy repository](https://github.com/MagriLab/IN22-Bias-aware-TADA)).

These cases accompany topic 2 of the [tutorials](../../tutorials/README.md).

## Data

- `rijke` and `vdp`: none; the truth is generated on the fly. These two cases have no script or sweep of
  their own: they run through `run_config.py` or `python -m romda.experiments`.
- `annular`: the azimuthal pressure measurements of [Zenodo record 15609832](https://zenodo.org/records/15609832),
  one file per equivalence ratio. The first run downloads them into `data/annular/` at the repository
  root.

## How to run

From the repository root:

```bash
python examples/run_config.py thermoacoustics/rijke --save-figs
python examples/run_config.py thermoacoustics/vdp --members           # also plot the ensemble members
python examples/thermoacoustics/annular.py --save-figs
python -m romda.experiments examples/thermoacoustics/annular.yml --set truth.ER=0.5375
```

The first annular run trains the echo state network of the bias, which takes much longer than the other
cases. It caches the training data under `results/annular/` and the trained network under
`results/esn_configs/`, both at the repository root; later runs, and the sweeps, reload them.

The sweep takes `--study <name>`, and `--quick` runs two values of each setting as a smoke test:

```bash
python examples/thermoacoustics/sweeps_annular.py --study regularization --quick
```

| Study | Setting varied | What to look for |
|---|---|---|
| `regularization` | rBA-EnKF regularization factor | with no regularization, the analysis follows the biased observations; a moderate value separates the bias from the state error, and too large a value freezes the update |
| `ensemble_size` | members `m`, rBA-EnKF and EnKF | the smallest ensemble within reach of the saturated error, which sets the cost of the digital twin |
| `assimilation_frequency` | time steps between observations `Nt_obs` | a limit cycle degrades gracefully with sparse data, unlike the chaotic Lorenz cases |
| `equivalence_ratio` | the four operating conditions, rBA-EnKF and EnKF | whether the conclusions, and the trained bias estimator, hold away from a single operating point |

## Outputs

In this folder, not tracked by git:

- `figs/`: the figures of the cases (`annular_01.png`, `rijke_01.png`, ...) and of the sweep
  (`annular_<study>.png`);
- `results/`: the regularization comparison of the annular script (`ER<value>/rBA_EnKF_regularization_sweep`)
  and the records of the sweep (`annular_<study>.jsonl`).
