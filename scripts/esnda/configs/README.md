# ESN-DA configurations

The YAML files in this folder configure the experiments of the washout-free ESN-DA paper, run by `romda.run.run_esn_config`: the truth, the forecast
model, the data-assimilation method and the assimilation window. Run every command from `scripts/esnda` in
the `romda` environment. The public cases (the Lorenz twins with their own physical model, and the thermoacoustic
cases) live next to their scripts in `scripts/` and run through `python -m romda.run`; see
`scripts/README.md`.

## Set up the environment

Install the package in editable mode from a source checkout:

```bash
conda activate romda
pip install -e ".[dev,notebooks]" --use-pep517
```

The package requires Python 3.10 or newer. The optional `dev` dependencies install the test and lint tools;
`notebooks` adds Jupyter and notebook execution.

## Cases

| Folder | Cases | `forecast.kind` | Forecast model | Runner |
|---|---|---|---|---|
| `configs/` | `lorenz63`, `lorenz96_small`, `lorenz63_parametric`, `lorenz96_parametric`, `lorenz63_forecast`, `lorenz96_forecast` | `esn` / `parametric_esn` | an ESN trained on the truth's own record, or across regimes (`romda.models.data_driven.rnns`) | `python -m romda.run` |

The `forecast.kind` key selects the runner. These files are the single source of the ESN-DA defaults: the
runs (`run/`) and the tutorials read them with
`romda.run.load_config`, and every value of the `esn` block enters the hash of the trained network,
so that editing a value trains or loads another network instead of reusing one silently.

## Run a configuration

```bash
python -m romda.run configs/<case>.yml [--seed N] [--set key=value]
```

Examples:

```bash
# ESN-DA twin experiment; the seed slides the truth window by 4 Lyapunov times
python -m romda.run configs/lorenz96_small.yml --seed 3

# override resolved values for one run; dotted keys, values parsed as JSON when possible
python -m romda.run configs/lorenz63.yml --set da.m=20 --set da.method=EnKF
python -m romda.run configs/lorenz96_small.yml --set da.n_measured=5 --set da.ring=true

# the same experiment from Python (mains, notebooks, sweeps)
python -c "
from romda.run import run_esn_config
print(run_esn_config('configs/lorenz63.yml', seed=1, **{'da.m': 20}))"
```

For an ESN forecast, `--seed` selects a new truth window (the ensemble is not reseeded: `initialize_from_val_data`
draws the members from the network's own seed). Integers, floats, booleans, lists and `null` given with `--set` are
parsed as JSON; other values stay strings. The metrics dictionary is printed as the last JSON line on standard
output; other messages come before it, so a script parses the last line.

## The POD-ESN cylinder case

The configuration of the POD-ESN cylinder case is `scripts/podesn_cylinder/pod_esn.yml`, next to its
script (see `scripts/podesn_cylinder/README.md`).

## Caching and outputs

The first ESN run trains and caches the matching network under `results/esn_configs/<hash>`. The ESN and training
settings, the observed components and the dataset identity enter the hash; the assimilation settings, the truth
window and `--seed` do not, so a parameter sweep reuses one trained network. Generated data and experiment outputs
go under `results/`, which is gitignored. Do not run two sweeps of the same study at once: the sweep harness
appends to JSON-lines records, and concurrent runs duplicate points.

## Add an experiment

Copy the nearest YAML file in `configs/` and edit its values. `truth.model` must resolve to a class exposed by
`romda.models.physical`. A new `forecast.kind` needs its own branch in `romda.run.run_esn_config`. Unknown configuration
keys fail validation, so a misspelt key does not pass silently. `tests/test_run.py` loads every `configs/`
file and pins the store hash of the `lorenz96_small` network (`tests/fixtures/lorenz96_small_pinned.yml`).
