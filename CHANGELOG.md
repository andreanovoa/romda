# Changelog

## Unreleased

### Added

- `scripts/thermoacoustics/cmame2023/` and the documentation page of the replication of Nóvoa, Racca & Magri
  (2023) are public: the Van der Pol oscillator and the Rijke tube with the regularized bias-aware EnKF.

### Changed

- Tutorials 40, 41 and 42 are one notebook, `40_DA_washout_free_ESN.ipynb`, in three parts: the Lorenz-63
  system, partial observations of the same system, and the ten-site Lorenz-96 system.
- The documentation has a Publications section, after the tutorials: the works produced with romda (the list of the
  README) and the replication pages of CMAME 2023 and ICCS 2025, which leave the tutorial folders. The old addresses
  redirect to the new pages.
- Tutorials 43–46 of topic 4 are renumbered 41–44, so the numbers have no gap after the merge of 40–42:
  `41_DA_parameter_estimation`, `42_DA_POD_ESN_cylinder`, `43_DA_CAE_ESN_KS` and `44_DA_CAE_ESN_Kolmogorov`. The
  old addresses of the documentation redirect to the new pages.
- `ESN_model` and the store of trained networks work with the echostatenetwork whose readout is
  $\mathbf{W}_\mathrm{out}^T \mathbf{r} + \mathbf{w}_0$ (the intercept outside `Wout`) and with the releases up
  to 0.6.0, whose `Wout` holds the constant in its last row. The store keeps one layout, `[Wout; constant row]`,
  so the networks stored before load to the same network with either. The singular values and the directions of
  the read-out corrections (`est_alpha=['Wout']`, `['Wdir']`) are those of the readout with its constant, as before.
- The defaults of `ESN_model` and `ESNConfig` are the conventions that the hyperparameter study favours:
  `Win_type='sparse_dense_bias'` (every unit reads the input bias; `'sparse'` before), `scale_bias_in=False`
  (`sigma_in` does not multiply the input bias; new option) and `regularize_bias_out=False` (the Tikhonov factor
  does not act on the constant of the readout; `True` before). They are options of romda, so they do not follow
  the defaults of the installed echostatenetwork. A configuration that leaves them at their defaults trains a new
  network: its hash in the store differs from that of the networks of the former conventions, which a
  configuration finds by pinning `Win_type`, `scale_bias_in: true` and `regularize_bias_out: true`, as the
  configurations of the published studies now do (`scripts/esnda`, `scripts/thermoacoustics`,
  `scripts/podesn_cylinder/iccs2025.yml`, `scripts/caeesnda`). A network stored or pickled before the change loads
  with the conventions that it was trained with.
- echostatenetwork 0.6.0 has no `scale_bias_in`: there a network is trained, hashed and stored with
  `scale_bias_in=True`, with a warning. The echostatenetwork with the intercept runs a stored network of
  `regularize_bias_out=True` but does not train one.

## 4.1.0 (2026-11-17)

### Added

- The convolutional autoencoders `CAE`, `CAE1D` and `MultiScaleCAE`, and the CAE-ESN reduced-order models
  `CAE_ESN`, `CAE1D_ESN` and `MultiScaleCAE_ESN` (Racca, Doan & Magri 2023; Özalp, Nóvoa & Magri 2026), are public.
  They need torch, the optional extra `romda[torch]`; `import romda` does not import it.
- Tutorial `45_DA_CAE_ESN_KS`: real-time data assimilation with the CAE-ESN on the Kuramoto–Sivashinsky equation,
  with the autoencoder weights of Özalp, Nóvoa & Magri (2026).
- `scripts/caeesnda/`: the CAE-ESN cases `ks.yml` and `kolmogorov.yml`, run by `cae_esn_da.py`.
- `scripts/podesn_cylinder/iccs2025.py` and `iccs2025.yml`: Figs. 3–5 of Nóvoa & Magri (2025) with the current
  code. `pod_esn.run_experiment` takes a configuration (`cfg`), measures the POD coefficients with `N_sensors=0`,
  and names the cached network after a digest of its options.
- Documentation: the page *Replications* sets the figures of the replication of Nóvoa & Magri (2025) beside the
  conclusions of the paper.

### Changed

- The Kolmogorov CAE-ESN tutorial is renamed `46_DA_CAE_ESN_Kolmogorov` and is public. Its record, computed with
  jax-cfd, is not distributed; the notebook reads it from `$DATA_ROOT`.
- `tutorials/README.md` lists the tutorials only; the notes on running them as tests are in the docstring of
  `tutorials/test_tutorials.py`.
- The figures of the documentation site follow its colour scheme: the build makes the white background of every
  PNG transparent, and `docs/stylesheets/figures.css` inverts the colours in the dark scheme. The class
  `no-invert` keeps the colours of an image.
- The documentation site renders every tutorial of topics 3 and 4 as a page with its saved outputs, instead of
  linking to the notebook on GitHub.

## 4.0.0 (2026-11-15)

### Changed

- The sweep harness moves from `scripts/sweep.py` to the module `romda.sweep`, and its tests to `tests/`.
- The package lives in `src/romda/` (a standard src layout). setuptools finds the subpackages, so a new
  subpackage no longer needs an entry in `pyproject.toml`. Import paths are unchanged (`romda.*`).
- `scripts/mains/`, `scripts/ablations/` and `configs/` are regrouped in `scripts/`, one folder per topic: the usage
  examples (`scripts/examples/lorenz/`) and the code of the papers (`scripts/thermoacoustics/`,
  `scripts/podesn_cylinder/`, `scripts/esnda/`). Each case keeps its script, its YAML configuration and its sweeps
  together, and each topic README lists the published works it supports. The commands
  change accordingly, e.g. `python -m romda.run scripts/examples/lorenz/lorenz63.yml`.
- The runner of the YAML configurations is the module `romda.run` (`load_config`, `run_config`, `run_esn_config`;
  `python -m romda.run <config>.yml`). A configuration writes a frequency in Hz with the tag `!hz`, read as
  2 pi f, e.g. `omega: !hz [1090, 1095]`.
- The scripts write their records and figures to `results/` and `figs/` next to each script, not to
  `results/ablations/` at the repository root; these folders are not tracked by git.
- The tutorials move from `scripts/tutorials/` to `tutorials/`, and `publish_public.py` moves to `tools/`.
- The sweep harness moves to `scripts/sweep.py`, next to the sweeps that run it.
- The configuration of the POD-ESN cylinder case (`scripts/podesn_cylinder/pod_esn.yml`) is public, next
  to its script.
- `scripts/thermoacoustics/annular.yml` holds the settings of Nóvoa, Noiray, Dawson & Magri (2024) at an equivalence
  ratio of 0.5125, and `annular_release.yml` those of the tutorial `32_DA_annular_raw` of the release v1.0.
- romda requires `echostatenetwork` 0.6 (open- and closed-loop methods of the network) and `dynamodels` 0.8
  (`Lorenz63.rk4_step` and `Lorenz96.rk4_step`).

### Fixed

- `scripts/thermoacoustics/vdp.yml` estimates `beta` alone. On the limit cycle, the data fix only the difference
  `beta - zeta`: with both estimated, the members spread along that line (correlation 0.996), the inflation pushed
  them past the parameter limits, a third of the analyses were rejected, and both settled at about 0.8 of the truth.
  With `zeta` known, `beta` ends within 0.5 % of the truth over seeds 0 to 4.
- `scripts/examples/lorenz/lorenz63.yml`: `std_obs` is 0.01, not 0.05. `std_obs` is a fraction of a variance, so 0.05 assumed an
  observation error with a standard deviation of 22 % of the largest observation, against a true noise of 5 %. The
  analysis barely moved the parameters, which drifted to 0.76 (beta) and 0.92 (rho) of their true values. With 0.01
  (10 %), both end within 4.3 % of the truth over seeds 0 to 4.
- `ESN_bias` with biased observations: the echo state network reads the innovation alone, as in Nóvoa et al.
  (2024). Its input was the whole state [bias; innovation], so the network forecast the bias from its own
  forecast, the data barely reached the bias, and the regularization of the rBA-EnKF had no effect. The
  networks of biased observations have a new hash and are trained again.
- `Observations` cuts a record loaded from a file at `t_min` and `t_max`, as it does for a model. The bias
  network of the annular case was trained on the last steps of the 120 s of data, where the mean of the raw
  pressure differs from that of the assimilation window.
- The training data of the annular bias network are cached for each equivalence ratio, assimilation window
  and set of parameter ranges; one file, and so one network, served every equivalence ratio.
- `sample_model_states` draws each parameter of the bias training members from its own range (the ranges
  followed the order of the `std_alpha` dict, the rows that of the sorted `est_alpha`), and gives the members
  the spread of the ensemble (its standard deviation was read as a relative spread).
- `Bias` keeps a time step that needs many decimals (1/51200 s was rounded to 2e-5 s).
- `ESN_model.init_ensemble` names the parameter rows of an explicit `est_alpha` correctly when `std_alpha` is
  a dict.

### Added

- `scripts/esnda/`: the code of the paper *Washout-free synchronization, parameter estimation and uncertainty
  quantification of echo state networks*. The folder holds the configurations (whose `da` blocks set the inflation and `cdd_fraction`,
  the fraction of the standard deviation in `C_dd`), the runs, one script per figure, `reproduce.sh` (every run and figure in the order of the paper) and the six trained echo state
  networks of the paper (`final_networks/`, as `.npz` files that load with `EchoStateNetwork.from_arrays`). Its
  README maps each figure of the paper to the scripts that produce it.
- The general code of the ESN-DA paper is in the package:
  - `romda.models.data_driven.rnns`: `model_esn` and `parametric_esn` build the dataset of a physical model and the
    network trained on it, from the blocks of a configuration (`romda.run.load_config`: `truth`,
    `forecast.esn`, `forecast.train`), and load the network from the ESN store when it holds it; `twin_data`,
    `network_summary`.
  - `romda.data_assimilation`: `deploy` (a filter, or the direct replacement of the measured components),
    `run_and_score`, `assimilate` (with `cdd_fraction`), `filter_record`, `as_result`, `inflation_metrics`;
    `run_da_loop(..., after_analysis=)` calls a function on the filtered ensemble after every analysis.
  - `romda.run.run_esn_config` runs a configuration whose forecast model is an echo state network
    (`forecast.kind: esn` or `parametric_esn`); `python -m romda.run` runs either kind of configuration.
  - `romda.utils`: `trajectory`, `noisy_copy`, `record_key`, `collected`.
  - `romda.metrics`: `valid_time(err, t, threshold, t_max)` (a nan error fails) and `forecast_valid_time(pred,
    target, norm, threshold, dt)`, which keeps the former `valid_time(pred, target, norm, threshold, dt)`, `attractor_kld` (on the bins of the
    truth), `lyapunov_spectrum_map` (the Lyapunov exponents of a map, by finite differences), and `ddof` of
    `state_metrics`.
  - `romda.observations.std_fraction_covariance`: `Cdd` as a fraction of the standard deviation of the record.
  - `ESN_model.replacement_loop`, `ESN_model.regime_matched_states` and `initialize_from_val_data(segments=)`, on
    the open- and closed-loop methods of `echostatenetwork` 0.6.
  - `EnsembleFilter` with `localization`: the state rows with no position (nan) are not tapered, and the transform
    of the LETKF is computed once per distinct taper.
  - `EnsembleFilter(inflation='innovation', inflation_guard=True, inflation_average='plain')`: the innovation-based
    inflation of the paper; `romda.run.innovation_options` reads its options from the `da` block of a
    configuration.
- The adaptive (`inflation='adaptive'`) and innovation-based (`inflation='innovation'`) inflation of
  `EnsembleFilter`, and its LETKF (`type='sqrt'` with `localization`), are public: the public example of the
  washout-free ESN-DA paper runs on them. They were options of the private build only.
- Gaussian-mixture analysis of the ensemble filters (`romda.estimators.sequential.mixture`), selected with
  `mixture=K` of `EnsembleFilter`: one analysis per mode of a forecast ensemble with up to `K` modes, e.g. the
  two modes of Lorenz-63 observed through z alone. Tutorial 17 introduces it. With `mixture=1`, the default, the
  filter is unchanged.
- `training_ERs` in the `bias` block of an experimental configuration: the network of the bias is trained on the
  records of several equivalence ratios instead of the assimilated one, e.g. the unified network of section 6.4
  of Nóvoa et al. (2024), trained on 0.4875, 0.5125 and 0.5375 and tested on 0.5625. The training data and the
  network are cached under the joined names of the records.
- `ensemble_metrics` reports `rmse_corrected_da` and `rmse_corrected_forecast` for a bias-aware filter: the
  error of the bias-corrected estimate, model plus bias.
- `RK4Integrator` (`romda.models`): a fourth-order Runge-Kutta scheme at the output step of a physical model, on
  the `Integrator` protocol of dynamodels. The default integrator, scipy's RK45 at its default tolerances, drifts
  by 100 Pa in 0.2 s on the annular model at an amplitude of 250 Pa; this scheme drifts by 1 Pa.
- Options of the bias training data of `ESN_bias`, which default to the earlier behaviour: `training_record`
  ('end', or 'start' for the first steps of the record, before the assimilation), `lag_criterion`
  ('correlation', or 'distance' for the lag of smallest distance to the data) and `keep_fixed_points` (the
  fraction of the training members at a fixed point that is kept).
- Options of `ESN_model`: `centre_input=False` scales the input of the network by its range only, and
  `validation_metric='log_nMSE'` selects the hyperparameters with the metric of the release v1.0.
- `ESN_bias.washout` and `run_da_loop(..., washout=True)` synchronize the network of the bias with the innovations
  of the `N_wash` network steps before the first observation, as Nóvoa et al. (2024) do. It is off by default:
  the default start of the network needs no data before the assimilation.
- `romda.run` reads `integrator` ('rk4'), `spinup_cr` (the initial state: the default model after that
  many characteristic times, `draw_state_after_spinup`) and `Cdd_entries` ('identical': one variance for every
  observable, `variance_fraction_covariance(..., identical=True)`) from the `da` block of a configuration, and
  `washout` from its `bias` block.
- `romda.metrics.window_errors`: the errors of the model estimate and of the bias-corrected estimate over a time
  window, e.g. after the last observation, to select a regularization factor.
- The measurement noise is seeded: one shared generator, `romda.utils.noise_rng` (seed 0 at import), feeds
  `Observations` and `add_noise_to_flow`; `romda.utils.seed_noise(seed)` restarts it, and
  `romda.run.run_config` does so with the seed of the run. The same seed now gives the same
  numbers. The noise has a stream of its own (`noise_generator`), apart from the stream of a model seed.
  Assigning `romda.observations.rng` still overrides it.
- A seed given to an estimator built on an `ESN_model` (or `POD_ESN`) now seeds its ensemble; before, only
  `regimes` used it, so the seeds of the cylinder sweep repeated one run.
- `ModelConfig`, `auto_load_or_create`, `save_model_to_config` and `load_model_from_config` are public
  (`romda.models.data_driven.model_config`): the store of trained models, by the hash of their configuration.
  The public wheel imports again (`esn_config` depends on it).
- Continuous integration of the tests and the linter (`.github/workflows/tests.yml`).
- `CITATION.cff` and this changelog.

### Fixed

- The stores of trained networks key the training data by their file name: `ESNConfig.to_hash` and
  `ModelConfig.to_hash` hash the name of `training_data_filename`, not its path. The experiment driver and
  the tutorials pass an absolute path, so a network trained in one clone was stored under a hash that no
  other clone or machine computed, and was trained again there. A name is its own name, so the networks
  stored under one keep their hash; `romda.models.data_driven.rnns.rekey_esn_store()` moves those stored
  under a path to the hash of the name.
- `get_project_root` (and with it `set_working_directories` and the model stores `MODEL_STORE` and
  `ESN_STORE`) falls back on the clone that holds romda when no folder above the working directory has a
  `src` folder, so a run script or notebook started outside the repository finds the shared `results/`
  stores instead of creating them under the working directory.
