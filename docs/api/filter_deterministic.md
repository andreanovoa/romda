# Kalman filters

## Summary

**File:** `src/romda/estimators/sequential/kalman.py`. `KalmanFilter` is the linear Kalman filter
(`type='linear'`, the only type in romda). Unlike the ensemble filters, it owns the state mean `ψ`
and the covariance `Cpp` directly (no ensemble). A model (`model=`) or a linear transition matrix
(`F=`) advances the mean, and a fixed one-step Jacobian, `F_jac` (default `F`, `model.F` or the
identity), advances the covariance.

| Option | Configuration |
|---|---|
| `parameter_estimation` | None, or `'joint'` for the trailing components of the state named in `est_alpha` |
| `bias` | a `Bias` estimator corrects the predicted observables through its linearization |
| `bias_regularization` | $\gamma > 0$ stacks the pseudo-observation of zero bias |
| `inflation` | scales the analysis covariance by $\rho^2$ (the state block only with `inflate_parameters=False`) |

romda keeps only the linear filter, which the introductory tutorials employ. The extended and
unscented Kalman filters (`type='extended'`, `'unscented'`), the dual scheme, covariance localization
and the Kalman smoother live in [`biasda`](https://github.com/andreanovoa/biasda); in romda, these
options raise `NotImplementedError`.

------

::: romda.estimators.KalmanFilter
