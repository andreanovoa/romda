# Examples

Each folder of `examples/` is one topic. A topic holds a few cases. Each case is a YAML configuration;
the cases that need figures or steps of their own also have a script next to it, and the others run
through `run_config.py`. Most cases also have a sweep script that repeats them over a grid of settings.
The Lorenz, Rijke, Van der Pol and cylinder cases run in minutes on a laptop; the first annular run
trains the echo state network of the bias and takes much longer. Every case writes its outputs next to
its own configuration. Each topic README lists the published works that the topic supports.

The examples complement the [tutorials](../tutorials/README.md): a tutorial explains one concept step by
step in a notebook; an example runs a complete experiment from the command line.

| Topic | Cases | What it shows |
|---|---|---|
| [`lorenz/`](lorenz/README.md) | `lorenz63`, `lorenz96` | Twin experiments on chaotic systems: the stochastic and the square-root EnKF, combined state and parameter estimation, and the regularized bias-aware EnKF (rBA-EnKF) on biased observations. |
| [`thermoacoustics/`](thermoacoustics/README.md) | `annular`, `rijke`, `vdp` | Thermoacoustic oscillations: twin experiments on the Rijke tube and the Van der Pol oscillator, and the real-time digital twin of raw experimental data from an annular combustor. |
| [`rom_da/cylinder/`](rom_da/cylinder/README.md) | `pod_esn` | Real-time data assimilation with a data-driven reduced-order model (POD-ESN) of the wake behind a cylinder, with online learning of the ESN output weights. |

## Before you start

Install `romda` from a clone of the repository, in editable mode, and run every command from the
repository root:

```bash
pip install -e ".[dev,notebooks]" --use-pep517
```

- **Data.** The Lorenz, Rijke and Van der Pol cases generate their own truth. The annular and cylinder cases
  download their datasets from Zenodo on the first run, into `data/` at the repository root.
- **Outputs.** A case writes its records and pickles to `results/`, and its figures to `figs/`, both in the
  folder of its script. These folders are not tracked by git, and every file in them can be regenerated.
- **Caches.** Trained bias estimators and reduced-order models are cached under `results/` at the
  repository root, so that a second run, or a sweep, reloads them instead of training them again.

## Run a case

There are three equivalent ways to run a case.

1. **The case script**, which also draws the figures:

   ```bash
   python examples/lorenz/lorenz63.py                   # opens the figures
   python examples/lorenz/lorenz63.py --save-figs       # saves them to examples/lorenz/figs/
   ```

   The cases without a script of their own run through `run_config.py`, with the configuration path
   relative to `examples/`:

   ```bash
   python examples/run_config.py lorenz/lorenz96
   python examples/run_config.py thermoacoustics/rijke --members
   ```

2. **The command line of the package**, which prints the metrics as the last line, in JSON:

   ```bash
   python -m romda.experiments examples/lorenz/lorenz63.yml --seed 3
   python -m romda.experiments examples/lorenz/lorenz96.yml --set da.m=20 --set da.method=EnKF
   python -m romda.experiments examples/thermoacoustics/annular.yml --set truth.ER=0.5375
   ```

   `--seed` draws another initial ensemble and another observation noise on the same truth; the same
   seed gives the same numbers. `--set key=value` overrides one value of
   the configuration for this run; the value is parsed as JSON when possible.

3. **Python**, which returns the filtered ensemble, the truth and the metrics:

   ```python
   from romda.experiments import run_experiment

   filter_ensemble, truth, metrics = run_experiment('examples/thermoacoustics/rijke.yml', seed=1)
   ```

The cylinder case is the exception: its truth is a set of held-out flow snapshots, not a model, so it
runs only through its script, `python examples/rom_da/cylinder/pod_esn.py`.

## Run a sweep

A sweep varies one setting at a time around fixed defaults (the defaults of the `experiment`
function of the sweep script) and repeats each setting over several random seeds:

```bash
python examples/lorenz/sweeps_lorenz63.py --study ensemble_size --quick   # two values, few seeds: a smoke test
python examples/lorenz/sweeps_lorenz63.py --study ensemble_size           # the full sweep
python examples/lorenz/sweeps_lorenz63.py --study ensemble_size --plot    # plot the records already on disk
examples/run_parallel.sh examples/lorenz/sweeps_lorenz63.py               # every study, in parallel
```

Each run is appended to a JSON-lines record in `results/` as soon as it finishes, so an interrupted
sweep resumes where it stopped; failed runs are recorded and left out of the statistics. The figure
shows the distribution over the seeds, with each run drawn on top, so that a diverged run appears as an
outlier instead of being hidden in a mean. The topic READMEs list the studies of each sweep and what to
look for in them. The harness is [`sweep.py`](sweep.py).

The errors are normalized by the root-mean-square amplitude of the unbiased truth over the assimilation
window, and are reported separately for the assimilation window (`rmse_da`) and the forecast after it
(`rmse_forecast`), together with the ensemble spread (see `romda.metrics.ensemble_metrics`). In the twin
configurations, part of the forecast window runs past the end of the truth record (`window_lyap` or
`window`), so `rmse_forecast` is a coarse measure; compare it between runs of the same case only.

## Reproducibility

Every run is reproducible. The observation noise comes from one shared generator,
`romda.utils.noise_rng`, which is seeded with 0 at import; `romda.experiments.run_experiment` restarts it
from the seed of the run (`romda.utils.seed_noise`). The same configuration with the same seed therefore
gives the same numbers, and the seeds of a sweep give different noise realizations as well as different
ensembles. The noise of the cylinder record has a fixed seed of its own, because the record is noised once
and cached; a record cached before this seed existed keeps its noise until `results/wakes/` is deleted.

## Anatomy of a configuration

```yaml
name: lorenz63_main             # a label only; the values below define the experiment
truth:
  model: Lorenz63               # class name in romda.models.physical
  params: {dt: 0.015, rho: 28., sigma: 10., beta: 2.6666666666666665}
  observe_dims: [0, 2]          # the measured state variables
  window_lyap: [10, 60, 80]     # t_start, t_stop, t_max in Lyapunov times (or absolute: window: [...])
  Nt_obs: 20                    # time steps between observations
  noise_level: 0.05
forecast:
  kind: physical                # the forecast model is the model of the truth
da:
  method: EnSRKF                # EnKF | EnSRKF | rBA_EnKF
  m: 50                         # ensemble size
  std_obs: 0.05                 # assumed observation error: covariance std_obs * max|y|^2 (a variance)
  std_phi: 0.2                  # initial spread of the state
  std_alpha: {rho: [25., 35.], beta: [2., 4.]}   # estimated parameters and their prior ranges
  inflation: 1.02
bias:                           # rBA_EnKF only: the echo state network that estimates the bias
  t_train_lyap: 40              # keys ending in _lyap are converted with the t_lyap of the model
  N_units: 50
```

`romda.experiments` does not check the keys: a misspelt key, in the file or in `--set`, is ignored
without a warning, so check the spelling against the configuration. The prior ranges in `std_alpha` must
lie inside the hard limits `alpha_lims` of the model (for example, the Rijke-tube `beta` is capped at 5):
members initialized at or beyond a limit make the filter reject every analysis.

## Add a case

Copy the closest configuration into its topic folder and edit the values. A new physical model needs only
a class in `romda.models.physical`. Add a script next to the configuration when the case needs figures
or steps of its own, following `lorenz/lorenz63.py`, and a README section with the published works it
supports.

## Published works

Each topic README lists the papers of its cases. This repository started with: 

- Nóvoa (2024). *Real-time data assimilation in nonlinear dynamical systems*. PhD thesis, University of
  Cambridge. [doi:10.17863/CAM.113001](https://doi.org/10.17863/CAM.113001)
