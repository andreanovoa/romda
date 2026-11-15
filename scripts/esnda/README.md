# Washout-free deployment of echo state networks

This folder holds the code and the trained echo state networks (ESNs) of the paper *Washout-free synchronization, parameter estimation and uncertainty quantification of echo state networks* (submitted to *Neural Networks*). The scripts train the ESNs, deploy them with ensemble filters from sparse and noisy data, and draw the figures of the paper. The test cases are the Lorenz-63 system (L63) and the Lorenz-96 system with ten and 40 sites, forecast by single ESNs (L96-10s, L96-40s), parallel ESNs (L96-10p, L96-40p) and a parametric ESN (parametric L96-10s).

| Folder or file | Contents |
|---|---|
| `configs/` | The YAML configurations of the single ESNs and of their filters (`configs/README.md`): the defaults of the runs, including the innovation-based inflation and `cdd_fraction` (the assumed observation error, a fraction of the standard deviation; `CDD_STD` overrides it). The code that is not specific to the paper is in romda: the builders of the trained ESNs (`romda.models.data_driven.rnns`), the deployments of the filters (`romda.data_assimilation`) and the YAML runner (`python -m romda.run configs/<case>.yml`). |
| `run/` | The runs: training, filter sweeps and the records that the figures read. |
| `plot/` | One `plot_*.py` script per figure of the paper; `plot_aux.py` holds the paths, the style and the loaders that the figures share. |
| `reproduce.sh` | Every run and every figure in the order of the paper; it skips a run whose outputs exist, so a second call only draws the figures (`DRY_RUN=1` lists the steps). |
| `final_networks/` | The manifest and the checksums of the six trained ESNs of the paper, and the scripts that build and check them (`final_networks/README.md`); the `.npz` files come with the data of the paper. |

The runs write their records to `results/<study>/` and the figures to `figs/` (or to the folder in the environment variable `PAPER`). The release holds the empty folders of `results/` and none of their records: the figures are reproduced by repeating the runs.

## How to run

Install romda from the root of the repository (`pip install -e .`), which also installs the package `echostatenetwork` (version 0.6.0 or later) that trains the ESNs. `./reproduce.sh` runs the paper, or one of its sections (`synchronization`, `parameter`, `localization`).

Run every script from this folder, e.g.

```bash
python run/run_synchronization.py lorenz63 5
```

```bash
python plot/plot_localization.py
```

Each figure script imports `plot_aux` first, which puts this folder and `run/` on the path, so that a figure imports the run whose functions it needs. Each script documents its modes and its environment variables in its docstring. The single ESNs are cached in the store of romda (`results/esn_configs/` at the root of the repository), so a network is trained once.

The longest runs hold several gigabytes of memory per worker (about 7 GB for one run of L96-10s with 64 members). The scripts read the number of workers from the environment variable `N_WORKERS`.

## From the paper to the scripts

The labels are those of the manuscript. The schematics (`fig:schematic`, `fig:parallel_schematic`) are drawn by hand.

| Paper (`\label`) | Runs (`run/`) | Figure or numbers (`plot/`) |
|---|---|---|
| `tab:cases` (data and filters of the test cases) | `configs/*.yml`; the assumed observation error is 0.2 for every case | |
| `fig:l63_forecast` (deployment of L63 from $y$ and from $z$) | `run_partial_table.py phase` (from $z$ alone, the Gaussian-mixture analysis, `mixture=2`) | `plot_lorenz63_phase.py` |
| `sec:sync`, partial observations of L63 and L96-10s | `run_partial_table.py` (from $z$ alone: `MIXTURE=2 SUBSETS=2 TAG=_z M=<m> python run/run_partial_table.py 100`); `run_lorenz96_small.py partial N` | |
| `fig:sync` (data required for synchronization) | `run_synchronization.py lorenz63` and `lorenz96` | `plot_synchronization.py` |
| `fig:bifurcation` (Lorenz-96, single and parallel parametric ESNs against $F$) | `run_parametric.py truth` (true system), `bifurcation` (parametric L96-10s) and `bifurcation readout parallel` (parallel parametric ESN) | `plot_l96_bifurcation.py` |
| `fig:param`, `fig:kld` (state and parameter estimation, and the forecast after it) | `DATA_LYAP=30 python run/run_parametric.py scan` and `attractor`, each also with `FIX_F=1`, and `scan` with `CDD_STD=0.02 TAG=_cdd002`; `train` for the network | `plot_parameter_scan.py` saves the two figures; `DATA_LYAP=30 python run/run_parametric.py one_record` gives the test of the initial ensemble |
| `fig:localization` (global filter and LETKF on L96-40) | `run_parallel_localization.py train` and `run`; L96-40s with `G_SITES=40 HALO=0 N_R=6000 TRAIN_LYAP=800` | `plot_localization.py` |
| `fig:parallel_forecast` (deployment of L96-40p and its uncertainty) | `run_parallel_localization.py train` and `forecast` | `plot_parallel_forecast.py`, then `pdfcrop --margins 2` |
| `fig:uncertainty` (spread against error for several $\sigma_\mathrm{obs}$) | `CDD_STD=<s> python run/run_uq_levels.py l63`, `10s` and `10p` (`Q=1` or `Q=5` for partial observations); `CDD_STD=<s> python run/run_parallel_localization.py forecast` for L96-40p | `plot_uncertainty.py` |
| `tab:esns`, `fig:verification` (Appendix A) | `run_verification.py` (`L63`, `L96-10` and `L96-40`; default: all) on the networks above | `plot_verification.py` |
| `app:readout_param` (Appendix B: the forcing in the input or in the readout) | `run_parametric.py train`, `bifurcation` and `appscan` with the variants `readout`, `paper`, `readout_t12` and `paper_t12` | |
| `fig:localized_covariance` (Appendix C) | `run_parallel_localization.py train` and `covariance` (row e, m = 1000: `FULL=1`, about 41 GB) | `plot_localized_covariance.py` |
| `app:mixture` (Appendix D: Gaussian-mixture analysis) | `romda.estimators.sequential.mixture`, selected with `mixture=2` of `EnsembleFilter` | `dev/esn_da/plot_lorenz63_mixture.py` (larger ensembles, not in the paper) |

The paper names `param_scan_lorenz96p.pdf` and `param_kld_lorenz96p.pdf` of `plot_parameter_scan.py` as `param_estimation.pdf` and `param_kld.pdf`. The files of the parametric L96-10s keep the names `param_scan_lorenz96p_*` of the first, parallel version of the study. A scan of `run_parametric.py` resumes from its output file when the file exists, whatever the value of `CDD_STD`; give each value of `CDD_STD` its own `TAG`.

The studies that the paper does not use (the latent L96, the single-ESN LETKF, the half-ring, the Gaussian-mixture figure of L63, the rerun comparison) live in `dev/esn_da/` of the repository.

## Methods of the paper in the code

| Method | Code |
|---|---|
| Ensemble square-root Kalman filter on the input and the reservoir state | `romda.estimators.EnsembleFilter` (`type='sqrt'`), deployed by `romda.data_assimilation.deploy` |
| Innovation-based inflation | `inflation='innovation'` of `romda.estimators.EnsembleFilter`, with the options of the `da` block of each configuration (`inflation_guard: true`, `inflation_average: plain`, `inflation_rate`, `inflation_max`; `romda.run.innovation_options`); `inflation='innovation'` in a run selects it |
| Gaussian-mixture analysis for a bimodal posterior | `romda/estimators/sequential/mixture.py`; tutorial 17 of romda introduces it on L63 observed through $z$ alone |
| Local ensemble transform Kalman filter (LETKF) | `EnsembleFilter(type='sqrt', localization=...)` |
| Parallel ESN | the parallel layout of `echostatenetwork` (`patch_size`, `halo`), through `romda.models.data_driven.rnns.ESN_model` |
| Parametric ESN with the forcing in the readout | `ESN_model` with `param_in_reservoir=False` and `readout_input='params'` |

## Trained networks

The six ESNs of Appendix A of the paper (`tab:esns`) are distributed with the data of the paper, not with this repository: unzipped at the root of romda, the archive puts them in `final_networks/` as `.npz` files and in the network store (`results/esn_configs/`), so that `reproduce.sh` loads them instead of training them. The `.npz` files load with `EchoStateNetwork.from_arrays` and need neither romda nor pickle. `final_networks/README.md` gives the hyperparameters, a closed-loop forecast in a few lines and the SHA-256 checksums; `manifest.json` records the training record and the provenance of each network.
