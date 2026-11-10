# Changelog

## Unreleased

### Changed

- The package lives in `src/romda/` (a standard src layout). setuptools finds the subpackages, so a new
  subpackage no longer needs an entry in `pyproject.toml`. Import paths are unchanged (`romda.*`).
- `scripts/mains/`, `scripts/ablations/` and `configs/` become `examples/`, one folder per topic
  (`lorenz/`, `thermoacoustics/`, `rom_da/cylinder/`). Each case keeps its script, its YAML configuration
  and its sweeps together, and each topic README lists the published works it supports. The commands
  change accordingly, e.g. `python -m romda.experiments examples/lorenz/lorenz63.yml`.
- The examples write their records and figures to `results/` and `figs/` next to each script, not to
  `results/ablations/` at the repository root; these folders are not tracked by git.
- The tutorials move from `scripts/tutorials/` to `tutorials/`, and `publish_public.py` moves to `tools/`.
- The sweep harness moves to `examples/sweep.py`, next to the sweeps that run it.
- The configuration of the POD-ESN cylinder case (`examples/rom_da/cylinder/pod_esn.yml`) is public, next
  to its script.

### Added

- The measurement noise is seeded: one shared generator, `romda.utils.noise_rng` (seed 0 at import), feeds
  `Observations` and `add_noise_to_flow`; `romda.utils.seed_noise(seed)` restarts it, and
  `romda.experiments.run_experiment` does so with the seed of the run. The same seed now gives the same
  numbers. The noise has a stream of its own (`noise_generator`), apart from the stream of a model seed.
  Assigning `romda.observations.rng` still overrides it.
- A seed given to an estimator built on an `ESN_model` (or `POD_ESN`) now seeds its ensemble; before, only
  `regimes` used it, so the seeds of the cylinder sweep repeated one run.
- `ModelConfig`, `auto_load_or_create`, `save_model_to_config` and `load_model_from_config` are public
  (`romda.models.data_driven.model_config`): the store of trained models, by the hash of their configuration.
  The public wheel imports again (`esn_config` depends on it).
- Continuous integration of the tests and the linter (`.github/workflows/tests.yml`).
- `CITATION.cff` and this changelog.
