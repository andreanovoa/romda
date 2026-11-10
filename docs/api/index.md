# API reference

`romda`'s primary data-assimilation API is the [`romda.estimators`](estimators.md)
hierarchy, in the layout of [`biasda`](https://github.com/andreanovoa/biasda). romda keeps only the
filters that its papers employ: the [ensemble filters](filter_stochastic.md) (`EnsembleFilter`:
`EnKF`, `EnSRKF`, `rEnKF`, `rBA_EnKF`) and the linear [Kalman filter](filter_deterministic.md)
(`KalmanFilter`). The other methods (smoothers, particle filters, localization, the extended and
unscented Kalman filters, ...) live in [`biasda`](https://github.com/andreanovoa/biasda).

- [Estimators (DA filters) — overview](estimators.md)
    - [Ensemble filters](filter_stochastic.md)
    - [Kalman filters](filter_deterministic.md)
- [Bias estimators](bias_estimators.md)
- [Models — overview](models.md)
    - [Physical models](models_physical.md)
    - [Data-driven models](models_data_driven.md)
- [Observations](observations.md)
