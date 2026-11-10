# Ensemble filters (ensemble Kalman filters)

## Summary

**File:** `src/romda/estimators/sequential/ensemble.py`. `EnsembleEstimator` is the abstract base of the
ensemble estimators (the forecast model, the ensemble, the bias, the inflation and the rejected
analyses); `EnsembleFilter` is the ensemble Kalman filter, configured by its analysis update
(`type='stochastic'` or `'sqrt'`) and its parameter scheme (`parameter_estimation=None` or
`'joint'`). The named classes are presets.

| Class | Configuration | Reference |
|---|---|---|
| `EnKF` | `type='stochastic'`: perturbed observations | Evensen (2003) |
| `EnSRKF` | `type='sqrt'`: symmetric square root (ETKF) | Tippett et al. (2003) |
| `rEnKF` (`rBA_EnKF`) | `EnKF` with a bias and `bias_regularization=1` | Nóvoa, Racca & Magri (2023) |

`rBA_EnKF` is the name of `rEnKF` in Nóvoa, Racca & Magri (2023); the two classes are the same filter.
The named filters estimate the parameters jointly when the model has `est_alpha`. A bias estimator
(`parent_bias=`) makes both updates bias-aware: the stochastic update employs the regularized
bias-aware gain, and the square-root update assimilates the pseudo-observation of zero bias stacked
under the observation. With an echo state network as the forecast model, `update_reservoir` sets
whether the analysis also updates the reservoir state.

romda keeps only these updates. The deterministic update (DEnKF) and the other square-root
variants (`type='deterministic'`, `'sqrt-random'`, `'sqrt-serial'`, `'sqrt-half'`), the dual scheme
(`parameter_estimation='dual'`), covariance localization and the LETKF (`localization=`), the
bias-estimation routes (`bias_estimation=`) and the particle filters live in
[`biasda`](https://github.com/andreanovoa/biasda); in romda, these options raise `NotImplementedError`.

**Covariance inflation** (`src/romda/estimators/inflation.py`, Evensen 2009 Chap. 15): fixed
multiplicative inflation via `inflation=`. Applied factors are logged in
`estimator.inflation_history`. With `inflate_parameters=False`, the inflation acts on the
state only, and the estimated parameters keep their analysis spread.

**Rejected analyses**: an analysis that takes a parameter outside its limits (`alpha_lims`) is
rejected: the forecast ensemble, inflated by `inflation_factor_rejection`, replaces the analysis,
and the time and the reason are logged in `estimator.rejected_analysis`.

------

::: romda.estimators.EnsembleEstimator

::: romda.estimators.EnsembleFilter

::: romda.estimators.EnKF

::: romda.estimators.EnSRKF

::: romda.estimators.rEnKF

::: romda.estimators.rBA_EnKF

::: romda.estimators.inflation
