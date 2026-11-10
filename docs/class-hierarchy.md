# Class hierarchy

How the four core class families — models, estimators, bias estimators and observations —
relate, before diving into the [API reference](api/index.md).

## Models

#### Physical models — [`dynamodels.physical`](https://andreanovoa.github.io/dynamodels/)

All use `IVPIntegrator` (scipy `solve_ivp`) except KS.

| Class | Dim | Key parameters | Integrator |
|---|---|---|---|
| `Lorenz63` | 3 | `rho`, `sigma`, `beta` | IVP |
| `Lorenz96` | Nx | `F`, `Nx` | IVP |
| `VdP` | 2 | `beta`, `zeta`, `kappa`, `law`, `omega` | IVP |
| `Annular` | 4 | `omega`, `nu`, `c2beta`, `kappa`, `epsilon` | IVP |
| `KS` | Nx | `nu`, `L`, `Nx` | Discrete (ETDRK4) |
| `Rijke` | 2Nm+Nc | `beta`, `tau`, `C1`, `C2`, `kappa` | IVP |

#### Data-driven models — `src/romda/models/data_driven/`

All use `DiscreteIntegrator`. The ML base classes `EchoStateNetwork` (from the external `echostatenetwork` package) and the `Projector` hierarchy (`POD`/`SPOD`, in `src/romda/models/data_driven/autoencoders/` — see [Data-driven models](api/models_data_driven.md)) are mixed in via multiple inheritance.

| Class | Key parameters | Notes |
|---|---|---|
| `ESN_model` | `N_units`, `rho`, `sigma_in`, `N_wash` | Inherits `EchoStateNetwork`; requires training `data` at init |
| `LatentROM` | `Nq`, `measure_modes`, `qr_selection`, `sensor_locations` | Base of every projector + forecaster ROM: fits the projector, trains the forecaster on the latent trajectory, places sensors by QR on the decoder Jacobian |
| `POD_ESN` | `N_modes`, `sensor_locations` | `LatentROM` + `ESN_model` + `POD`; sensor placement via QR on `Psi` |
| `LinearModel` | `F` (transition matrix), `Q_noise` | $\vect{\psi}_{t+1} = \matr{F}\vect{\psi}_t + \vect{\eta}_t$ |



---

## Estimator

**Package:** `src/romda/estimators/` (the layout of [`biasda`](https://github.com/andreanovoa/biasda), which romda will import once it is released).

romda keeps only the filters that its papers employ: the ensemble Kalman filters (`EnKF`, `EnSRKF`, `rEnKF`) and the linear Kalman filter of the introductory tutorials. The other ensemble updates, the dual scheme, localization, the bias-estimation routes, the extended and unscented Kalman filters, the particle filters and the smoothers live in [`biasda`](https://github.com/andreanovoa/biasda).

Abstract base class (ABC) that wraps a `Model` and an optional `Bias` into a full DA loop. `SequentialEstimator` adds the shared `forecast_step()` that advances both the model and the bias in time.

**Key attributes:** `est_alpha`, `inflation_factor`, `regularization_factor`, `inflate_parameters`, `start_param`, `start_bias`

**Key concrete methods:** `forecast_step()`, `_init_bias()`, `_MA()` (applies measurement operator `M`)

Subclasses must implement:
- `analysis_step(d, Cdd)` — Bayesian update given observation vector `d` and noise covariance `Cdd`

What is estimated (state, state + parameters, state + parameters + bias) is a configuration of the augmented state; how it assimilates is the class:

```
Estimator  (ABC)
└── SequentialEstimator
    ├── KalmanFilter            type = linear
    └── EnsembleFilter          type = stochastic | sqrt
        ├── EnKF
        │   └── rEnKF
        │       └── rBA_EnKF
        └── EnSRKF
```

The relationship between the three main classes — attributes, not subclasses:

```
Estimator instance
  .model  →  Model instance   (used for the forecast)
  .bias   →  Bias instance    (optional; None if bias-unaware)
             .forecaster  →  Model instance   (usually an ESN_model)
```

---

### Available estimators

#### Ensemble filters — `src/romda/estimators/sequential/ensemble.py`

`EnsembleEstimator` is the abstract base of the ensemble estimators; `EnsembleFilter` is the ensemble Kalman filter, configured by `type` (`'stochastic'` or `'sqrt'`) and `parameter_estimation` (None or `'joint'`), and the named classes are presets.

**Key attributes:** `m` (ensemble size), `type`, `parameter_estimation`, `regularization_factor`, `update_reservoir`

| Class | Configuration | Reference |
|---|---|---|
| `EnKF` | `type='stochastic'` (perturbed observations) | Evensen (2003) |
| `EnSRKF` | `type='sqrt'` (symmetric square root, ETKF) | Tippett et al. (2003) |
| `rEnKF` (`rBA_EnKF`) | bias-aware, regularized by `bias_regularization` | Nóvoa, Racca & Magri (2023) |

Covariance inflation (`inflation=`, `src/romda/estimators/inflation.py`) is a configuration of every filter.

#### Kalman filters — `src/romda/estimators/sequential/kalman.py`

`KalmanFilter` owns the state mean `ψ` and covariance `Cpp` directly (no ensemble).

| Class | Notes |
|---|---|
| `KalmanFilter` | `type='linear'`; a model or `F` advances the mean, and the Jacobian `F_jac` propagates the covariance |

---

## Bias

**File:** `src/romda/bias_estimators/bias.py`

Base class for observation-bias estimators. Mixes in `HistoryTracker`. Wraps a `forecaster` model to produce bias corrections at each assimilation step.

**Key attributes:** `innovation`, `dt`, `forecaster`, `Nq`, `N_dim`, `upsample`, `biased_observations`

Subclasses must implement:
- `init_forecaster(**kwargs)` — build/train the internal forecasting model
- `state_derivative()` — return the Jacobian `J = db/dy` used by bias-aware filters

```
Bias
├── ESN_bias
└── ConstantBias
    └── NoBias
```

| Class | File | Notes |
|---|---|---|
| `ESN_bias` | `src/romda/bias_estimators/esn.py` | Correlation-based training; `biased_observations = True`; the workhorse |
| `ConstantBias` | `src/romda/bias_estimators/constantbias.py` | Fixed bias estimate; `NoBias` subclass for bias-blind DA |

---

### Forecaster

The `forecaster` attribute of a `Bias` instance is a `Model` subclass — typically a data-driven model trained on the residual between observations and model output.

```
Bias instance
  .forecaster  →  Model instance   (usually ESN_model)
```

The forecaster is initialised inside `init_forecaster()` and stepped forward in sync with the main model during `Estimator.forecast_step()`. Its output is the predicted bias `b(t)`, which enters the analysis step as a correction to the observation.

---

## Observations

**File:** `src/romda/observations.py`

Standalone class (no inheritance). Generates or loads the reference solution, applies a configurable manual bias, adds noise, and exposes observation time indices for the DA loop.

**Key attributes:**

| Attribute | Description |
|---|---|
| `y_true` | Clean biased truth signal `(Nt, Nq, 1)` |
| `y_raw` | Noisy observed signal `(Nt, Nq, 1)` |
| `b_true` | Applied bias `(Nt, Nq, 1)` |
| `t_true` | Full time vector |
| `y_obs`, `t_obs` | Observations at assimilation times |
| `obs_idx` | Indices into `t_true` at which observations are taken |
| `Nt_obs` | Subsampling rate (every `Nt_obs` steps) |

**Noise options** (`noise_type`): `'gauss, add'`, `'gauss, mult'`, coloured noise variants.

**Manual bias options** (`manual_bias`): `'linear'`, `'periodic'`, `'time'`, `'cosine'`, or any callable `f(y_true, t_true) -> (b, name)`.

**Key method:** `plot_truth(case)` — five-panel figure (raw, truth, PDF, PSD, difference).
