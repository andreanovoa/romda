# Washout-free deployment of echo state networks

A recurrent network forecasts only after a washout, i.e., an initial phase in which the full state drives the network until its internal state synchronizes with the system. The scripts of this folder replace the washout with an ensemble filter. They train echo state networks (ESNs), deploy them from sparse and noisy data, estimate a physical parameter with the state, and quantify the uncertainty of the forecast. The test cases are the Lorenz-63 system (L63) and the Lorenz-96 system with ten and 40 sites, forecast by single ESNs (L96-10s, L96-40s), parallel ESNs (L96-10p, L96-40p) and a parametric ESN (parametric L96-10s).

| Folder or file | Contents |
|---|---|
| `configs/` | The YAML configurations of the single ESNs and of their filters (`configs/README.md`): the defaults of the runs, including the innovation-based inflation and `cdd_fraction` (the assumed observation error, a fraction of the standard deviation; `CDD_STD` overrides it). The code that is not specific to these studies is in romda: the builders of the trained ESNs (`romda.models.data_driven.rnns`), the deployments of the filters (`romda.data_assimilation`) and the YAML runner (`python -m romda.run configs/<case>.yml`). |
| `run/` | The runs: training, filter sweeps and the records that the figures read. |
| `plot/` | One `plot_*.py` script per figure; `plot_aux.py` holds the paths, the style and the loaders that the figures share. |
| `reproduce.sh` | Every run and every figure, from synchronization to uncertainty; it skips a run whose outputs exist, so a second call only draws the figures (`DRY_RUN=1` lists the steps). |
| `final_networks/` | The manifest and the checksums of the six trained ESNs, and the scripts that build and check them (`final_networks/README.md`); the `.npz` files are distributed separately (see Trained networks). |

The runs write their records to `results/<study>/` and the figures to `figs/` (or to the folder in the environment variable `PAPER`). The release holds the empty folders of `results/` and none of their records: the figures are reproduced by repeating the runs.

## How to run

Install romda from the root of the repository (`pip install -e .`), which also installs the package `echostatenetwork` (version 0.6.0 or later) that trains the ESNs. `./reproduce.sh` runs every study, or one group of them (`synchronization`, `parameter`, `localization`).

Run every script from this folder, e.g.

```bash
python run/run_synchronization.py lorenz63 5
```

```bash
python plot/plot_localization.py
```

Each figure script imports `plot_aux` first, which puts this folder and `run/` on the path, so that a figure imports the run whose functions it needs. Each script documents its modes and its environment variables in its docstring. The single ESNs are cached in the store of romda (`results/esn_configs/` at the root of the repository), so a network is trained once.

The longest runs hold several gigabytes of memory per worker (about 7 GB for one run of L96-10s with 64 members). The scripts read the number of workers from the environment variable `N_WORKERS`.

## Studies

Each row is one study: the runs that produce its records, and the script that draws its figure or prints its numbers. The assumed observation error is 0.2 for every case (`configs/*.yml`).

| Study | Runs (`run/`) | Figure or numbers (`plot/`) |
|---|---|---|
| Deployment of L63 from one observed component ($y$ or $z$) | `run_partial_table.py phase` (from $z$ alone, the Gaussian-mixture analysis, `mixture=2`) | `plot_lorenz63_phase.py` |
| Partial observations of L63 and L96-10s: failed runs, valid time and spread | `run_partial_table.py` (from $z$ alone: `MIXTURE=2 SUBSETS=2 TAG=_z M=<m> python run/run_partial_table.py 100`); `run_lorenz96_small.py partial N` | |
| Data required for synchronization, against a washout | `run_synchronization.py lorenz63` and `lorenz96` | `plot_synchronization.py` |
| Bifurcation diagram of Lorenz-96 against the forcing $F$: true system, single and parallel parametric ESNs | `run_parametric.py truth` (true system), `bifurcation` (parametric L96-10s) and `bifurcation readout parallel` (parallel parametric ESN) | `plot_l96_bifurcation.py` |
| State and parameter estimation, and the forecast after it | `DATA_LYAP=30 python run/run_parametric.py scan` and `attractor`, each also with `FIX_F=1`, and `scan` with `CDD_STD=0.02 TAG=_cdd002`; `train` for the network | `plot_parameter_scan.py` saves the two figures; `DATA_LYAP=30 python run/run_parametric.py one_record` gives the test of the initial ensemble |
| Global filter and LETKF on L96-40 | `run_parallel_localization.py train` and `run`; L96-40s with `G_SITES=40 HALO=0 N_R=6000 TRAIN_LYAP=800` | `plot_localization.py` |
| Deployment of L96-40p and its uncertainty | `run_parallel_localization.py train` and `forecast` | `plot_parallel_forecast.py`, then `pdfcrop --margins 2` |
| Spread against error for several $\sigma_\mathrm{obs}$ | `CDD_STD=<s> python run/run_uq_levels.py l63`, `10s` and `10p` (`Q=1` or `Q=5` for partial observations); `CDD_STD=<s> python run/run_parallel_localization.py forecast` for L96-40p | `plot_uncertainty.py` |
| Verification of the trained ESNs: forecast, statistics and Lyapunov exponents | `run_verification.py` (`L63`, `L96-10` and `L96-40`; default: all) on the networks above | `plot_verification.py` |
| The forcing in the input or in the readout of the parametric ESN | `run_parametric.py train`, `bifurcation` and `appscan` with the variants `readout`, `paper`, `readout_t12` and `paper_t12` | |
| Localized correlations of the ensemble on L96-40p | `run_parallel_localization.py train` and `covariance` (row e, m = 1000: `FULL=1`, about 41 GB) | `plot_localized_covariance.py` |
| Gaussian-mixture analysis | `romda.estimators.sequential.mixture`, selected with `mixture=2` of `EnsembleFilter` | `dev/esn_da/plot_lorenz63_mixture.py` (larger ensembles) |

`plot_parameter_scan.py` saves its two figures as `param_scan_lorenz96p.pdf` and `param_kld_lorenz96p.pdf`. The files of the parametric L96-10s keep the names `param_scan_lorenz96p_*` of the first, parallel version of the study. A scan of `run_parametric.py` resumes from its output file when the file exists, whatever the value of `CDD_STD`; give each value of `CDD_STD` its own `TAG`.

Further studies (the latent L96, the single-ESN LETKF, the half-ring, the Gaussian-mixture figure of L63, the rerun comparison) live in `dev/esn_da/` of the repository.

## Methods in the code


| Method | Code |
|---|---|
| Ensemble square-root Kalman filter on the input and the reservoir state | `romda.estimators.EnsembleFilter` (`type='sqrt'`), deployed by `romda.data_assimilation.deploy` |
| Innovation-based inflation | `inflation='innovation'` of `romda.estimators.EnsembleFilter`, with the options of the `da` block of each configuration (`inflation_guard: true`, `inflation_average: plain`, `inflation_rate`, `inflation_max`; `romda.run.innovation_options`); `inflation='innovation'` in a run selects it |
| Gaussian-mixture analysis for a bimodal posterior | `romda/estimators/sequential/mixture.py`; tutorial 17 of romda introduces it on L63 observed through $z$ alone |
| Local ensemble transform Kalman filter (LETKF) | `EnsembleFilter(type='sqrt', localization=...)` |
| Parallel ESN | the parallel layout of `echostatenetwork` (`patch_size`, `halo`), through `romda.models.data_driven.rnns.ESN_model` |
| Parametric ESN with the forcing in the readout | `ESN_model` with `param_in_reservoir=False` and `readout_input='params'` |

## Trained networks

The six trained ESNs are distributed as a data archive, not with this repository: unzipped at the root of romda, the archive puts them in `final_networks/` as `.npz` files and in the network store (`results/esn_configs/`), so that `reproduce.sh` loads them instead of training them. The `.npz` files load with `EchoStateNetwork.from_arrays` and need neither romda nor pickle. `final_networks/README.md` gives the hyperparameters, a closed-loop forecast in a few lines and the SHA-256 checksums; `manifest.json` records the training record and the provenance of each network.
