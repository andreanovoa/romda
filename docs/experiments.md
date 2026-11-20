# Examples

The [`scripts/`](https://github.com/andreanovoa/romda/tree/main/scripts) folder of the
repository holds complete experiments, one folder per topic. Each case is a YAML configuration, with a
script next to it when the case needs figures or steps of its own, and most cases have a sweep script that
repeats them over a grid of settings. Every case writes its outputs next to its own configuration, and the
README of each topic lists the published works that it supports.

| Topic | Cases | What it shows |
|---|---|---|
| [`lorenz/`](https://github.com/andreanovoa/romda/tree/main/scripts/examples/lorenz) | `lorenz63`, `lorenz96` | Twin experiments on chaotic systems: stochastic and square-root EnKF, combined state and parameter estimation, the rBA-EnKF on biased observations. |
| [`thermoacoustics/`](https://github.com/andreanovoa/romda/tree/main/scripts/thermoacoustics) | `annular`, `rijke`, `vdp` | Twin experiments on the Rijke tube and the Van der Pol oscillator, and the real-time digital twin of raw experimental data from an annular combustor. |
| [`podesn_cylinder/`](https://github.com/andreanovoa/romda/tree/main/scripts/podesn_cylinder) | `pod_esn` | Real-time data assimilation with a POD-ESN reduced-order model of the cylinder wake, with online learning of the ESN output weights. |
| [`caeesnda/`](https://github.com/andreanovoa/romda/tree/main/scripts/caeesnda) | `ks`, `kolmogorov` | Real-time data assimilation with the convolutional autoencoder echo state network (CAE-ESN) of Özalp, Nóvoa & Magri (2026), on the Kuramoto–Sivashinsky equation and the Kolmogorov flow. Both run through `python scripts/caeesnda/cae_esn_da.py <case>` and need torch. `ks.yml` integrates its record with dynamodels and loads the autoencoder of the paper; `kolmogorov.yml` reads a jax-cfd record from `$DATA_ROOT`, which is not distributed with romda. |
| [`esnda/`](https://github.com/andreanovoa/romda/tree/main/scripts/esnda) | scripts of a paper | Washout-free deployment of echo state networks with ensemble filters on the Lorenz-63 and Lorenz-96 systems: synchronization, parameter estimation, localization with parallel networks and uncertainty quantification, with the trained networks of the paper. |

Two published works are replicated with the current code, each with a page that sets the figures beside the
conclusions of the paper:

- [Nóvoa & Magri (2025), ICCS](replications/iccs2025.md): online learning of a POD-ESN of the cylinder wake.
- [Nóvoa, Racca & Magri (2023), CMAME](replications/cmame2023.md): regularized bias-aware EnKF in thermoacoustics.

Install `romda` in editable mode from a clone of the repository, and run the commands from its root:

```bash
pip install -e ".[dev,notebooks]" --use-pep517
```

## Run a configuration

Use `romda.run.run_config` from Python:

```python
from romda.run import run_config

filter_ensemble, truth, metrics = run_config(
    "scripts/thermoacoustics/rijke.yml",
    seed=1,
)
print(metrics)
```

The same runner is available from the command line:

```bash
python -m romda.run scripts/examples/lorenz/lorenz63.yml --seed 3
python -m romda.run scripts/thermoacoustics/rijke.yml --seed 1
python -m romda.run scripts/thermoacoustics/annular.yml --set truth.ER=0.5375
```

The `--seed` option redraws the initial ensemble and the observation noise on the same physical truth;
the same seed gives the same numbers. The noise comes from one shared generator, `romda.utils.noise_rng`,
which `romda.utils.seed_noise` restarts.
Use `--set key=value` for temporary dotted-key overrides. Numbers, booleans, lists, and
`null` are parsed as JSON. The final output line contains the experiment metrics as JSON.

The scripts of the cases add the figures, for example:

```bash
python scripts/examples/lorenz/lorenz63.py --save-figs
python scripts/run_config.py thermoacoustics/vdp --save-figs
python scripts/podesn_cylinder/pod_esn.py --save-figs
python scripts/caeesnda/cae_esn_da.py ks --save-figs
```

The annular and cylinder cases download their data on first use, into `data/` at the repository root.

## Available configurations

| Configuration | Experiment |
|---|---|
| `scripts/examples/lorenz/lorenz63.yml` | Lorenz 63 twin |
| `scripts/examples/lorenz/lorenz96.yml` | Lorenz 96 twin |
| `scripts/thermoacoustics/vdp.yml` | Van der Pol twin |
| `scripts/thermoacoustics/rijke.yml` | Rijke-tube twin |
| `scripts/thermoacoustics/annular.yml` | Annular experimental data |
| `scripts/podesn_cylinder/pod_esn.yml` | POD-ESN cylinder wake (through its script only) |
| `scripts/caeesnda/ks.yml` | CAE-ESN, Kuramoto–Sivashinsky equation (through its script only) |
| `scripts/caeesnda/kolmogorov.yml` | CAE-ESN, Kolmogorov flow; the jax-cfd record is not distributed (through its script only) |

The Python API returns the filtered ensemble, truth observations, and a metrics dictionary,
which can be used to create custom plots or further analysis.
