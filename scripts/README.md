# Scripts

Complete experiments, run from the command line at the repository root after `pip install -e ".[dev]"`. The
[tutorials](../tutorials/README.md) explain the concepts step by step; these scripts run whole cases.

| Folder | Cases | Content |
|---|---|---|
| [`examples/lorenz/`](examples/lorenz/README.md) | `lorenz63`, `lorenz96` | Usage examples: EnKF and square-root EnKF, state and parameter estimation, the rBA-EnKF on biased observations |
| [`thermoacoustics/`](thermoacoustics/README.md) | `rijke`, `vdp`, `annular` | Twin experiments on the Rijke tube and the Van der Pol oscillator; digital twin of an annular combustor from raw data, with the settings of Nóvoa, Noiray, Dawson & Magri (2024) |
| [`podesn_cylinder/`](podesn_cylinder/README.md) | `pod_esn` | Nóvoa & Magri (2025): POD-ESN of the cylinder wake with online learning of the output weights |
| [`esnda/`](esnda/README.md) | scripts of a paper | Washout-free deployment of echo state networks: synchronization, parameter estimation, localization, uncertainty quantification |

Each case is a YAML configuration, with a script next to it when it needs figures or steps of its own; each topic
README lists its papers. Outputs go to `results/` and `figs/` next to the script (untracked); datasets download to
`data/` and trained models are cached under `results/` at the repository root.

## Run

```bash
python scripts/examples/lorenz/lorenz63.py --save-figs              # a case script, with its figures
python scripts/run_config.py thermoacoustics/rijke                  # a case without a script
python -m romda.run scripts/examples/lorenz/lorenz96.yml --seed 3 --set da.m=20   # metrics as JSON
```

```python
from romda.run import run_config
filter_ensemble, truth, metrics = run_config('scripts/thermoacoustics/rijke.yml', seed=1)
```

`--seed` changes the initial ensemble and the observation noise; the same seed gives the same numbers.
`--set key=value` overrides one value, and a misspelt key raises an error. The cylinder case runs only through
`pod_esn.py`, because its truth is a set of flow snapshots, not a model.

## Sweeps

A sweep varies one setting around the defaults of its script and repeats it over seeds; runs are appended to
`results/` as they finish, so an interrupted sweep resumes. The harness is `romda.sweep`.

```bash
python scripts/examples/lorenz/sweeps_lorenz63.py --study ensemble_size --quick   # smoke test
python scripts/examples/lorenz/sweeps_lorenz63.py --study ensemble_size --plot    # plot the stored runs
for study in ensemble_size assimilation_frequency inflation observation_noise regularization; do
    MPLBACKEND=Agg python scripts/examples/lorenz/sweeps_lorenz63.py --study $study   # every study, one at a time
done
```

Run the studies one after another, not at once: each run holds an ensemble (and, for the bias-aware
filter, an echo state network) in memory.
