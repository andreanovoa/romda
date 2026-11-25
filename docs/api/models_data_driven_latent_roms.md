# Latent reduced-order models

## Summary

**Files:** `src/romda/models/data_driven/latent_rom.py` (the base class and the sensor placement),
`pod_esn.py` and `autoencoder_roms.py` (the combinations) and `model_config.py` (the store of
trained models).

A latent reduced-order model combines two parts. A projector maps a flow snapshot to a few
reduced coordinates (known in machine learning as the latent space), and a forecaster advances
these coordinates in time. The model observes the flow at sensors by decoding the forecast
reduced coordinates. Every combination is a `dynamodels.Model`, so the ensemble filters of
`romda.estimators` take it as their forecast model. `LatentROM` and the generalization of the
`Projector` interface were written by Lukas Campbell (autoencoder-rom, 2026), who
generalized romda's original `POD_ESN` to any projector and any forecaster.

| Object | Role |
|---|---|
| `LatentROM` | Fits the projector, trains the forecaster, places the sensors and computes the observables |
| `POD_ESN`, `CAE_ESN`, `CAE1D_ESN`, `MultiScaleCAE_ESN` | One class per projector and forecaster pair (see [the class table](#available-reduced-order-models)) |
| `ModelConfig`, `auto_load_or_create` | Save, find and reload trained projectors by a hash of their options |

## Projector and forecaster

Notation: $\vect{q} \in \mathbb{R}^{N_x}$ is a snapshot with every field component stacked
on the fluid points, $\vect{z} = \mathcal{E}(\vect{q}) \in \mathbb{R}^{r}$ are its reduced coordinates
($r$ = `N_latent`, `N_modes` for POD), $\mathcal{E}$ is the encoder (`encode`) and $\mathcal{D}$ the decoder
(`decode`), which restores the temporal mean. For POD, $\mathcal{D}(\vect{z}) = \matr{\Psi}\vect{z} +
\bar{\vect{q}}$, in which $\matr{\Psi}$ are the spatial modes and $\bar{\vect{q}}$ is the
temporal mean ([projectors page](models_data_driven_projectors.md)); for a convolutional
autoencoder, $\mathcal{E}$ and $\mathcal{D}$ are networks
([Convolutional autoencoders](convolutional_autoencoders.md)).

The forecaster advances the reduced coordinates with its own internal state, $\vect{r}$,

$$
\left(\vect{z}_{t+1}, \vect{r}_{t+1}\right) = F\left(\vect{z}_{t}, \vect{r}_{t}\right),
$$

in which $F$ is one step of the echo state network (ESN, $\vect{r}$ is the reservoir state;
[ESN page](models_data_driven_esn.md)).
The state of the reduced-order model is

$$
\vect{\psi} = \begin{bmatrix} \vect{z} \\ \vect{r} \end{bmatrix},
$$

which is what the ensemble filters forecast and update. The internal state of the forecaster
is part of the state, as for the echo state network in the bias-aware filter of Nóvoa, Racca &
Magri (2024).

**Composition by inheritance.** A combination lists `LatentROM`, then the forecaster, then the
projector, and has no body:

```python
class MultiScaleCAE_ESN(LatentROM, ESN_model, MultiScaleCAE): ...
```

`LatentROM.__init_subclass__` reads the forecaster and projector classes from the three bases
into `_forecaster_cls` and `_projector_cls`, and sets `extra_print_params` to the forecaster's
followed by `Nq`, `measure_modes` and the latent size (`_size_param`). The method resolution
order makes the projector provide `fit`, `encode`, `decode` and `decode_at`, and the forecaster
provide `time_step`, `build_psi` and `reset_forecaster`. `LatentROM` overrides `spatial_basis`
(to linearize the decoder about $\bar{\vect{z}}$), `get_observables`, `obs_labels`,
`state_labels` and `reset_case`.

**Construction.** `LatentROM.__init__(data, dt, **kwargs)` takes snapshots of shape
`(Nu, N_t, Nx, Ny)`, with NaN at solid points, and performs four steps:

1. It fits the projector on `data`, or adopts a fitted one passed as `projector=`.
2. It encodes the training snapshots into the reduced trajectory,
   $\matr{Z} = [\vect{z}_1, \dots, \vect{z}_{N_t}]$ (`latent_training_trajectory`), and stores
   its temporal mean, $\bar{\vect{z}}$.
3. It trains the forecaster on $\matr{Z}$, reshaped to the `(L, N_t, N_latent)` layout of
   `phi_to_esn_layout`.
4. It places `Nq` sensors ([sensor placement](#sensor-placement)).

**One option namespace.** A combination is one object, so the options of `LatentROM`, of the
projector and of the forecaster share one namespace, and the keywords are routed by name. The
options of `LatentROM` (`Nq`, `measure_modes`, `sensor_locations`, `qr_selection`,
`perform_test`, `latent_symbol`) are set on the model. The latent size (`n_latent`, `n_modes`)
and, when no fitted projector is passed, every option that the projector class defines go to the
projector. The remaining keywords go to the forecaster. In a `CAE1D_ESN` or a
`MultiScaleCAE_ESN`, `channels`, `kernel_sizes`, `periodic_padding`, `activation`,
`learning_rate`, `batch_size`, `val_fraction`, `patience`, `device` and the other options of the
[convolutional autoencoders](convolutional_autoencoders.md) therefore configure the
autoencoder, whereas `N_units`, `rho`, `sigma_in`, `N_wash`, `upsample` and the other options of
the [ESN](models_data_driven_esn.md) configure the forecaster. The autoencoder options are named
so that they do not shadow the ESN ones: `periodic_padding` and `decoder_upsampling` are spatial,
whereas the ESN's `periodic` and `upsample` act on the reservoir layout and on time. An option
that both classes define, such as `seed`, goes to the projector. The training epochs are given
per part, because `epochs` would be ambiguous. The options that change the construction are

| Option | Default | Effect |
|---|---|---|
| `projector` | `None` | A fitted projector of `_projector_cls`, such as an autoencoder restored with `auto_load_or_create` or `load_state_dicts`; `data` then trains only the forecaster, and the remaining keywords go to the forecaster. An `n_latent` or `n_modes` that differs from the projector's raises `ValueError` |
| `projector_epochs`, `forecaster_epochs` | `None` | Training epochs of each part (`epochs` raises `TypeError`). `None` keeps each class's default |
| `train_forecaster` (alias `train_ESN`) | `True` | `False` fits only the projector and places the sensors |
| `skip_sensor_placement` | `False` | Observes the reduced coordinates, as `measure_modes=True` does |

`projector_history` and `forecaster_history` hold the training histories of the two parts.
`reset_case(reset_forecaster=True, Z0=...)` restarts the forecaster from `Z0` (default: the
reduced coordinates of the first training snapshot) with zero reservoir states, and keeps its
trained parameters. With the default `reset_forecaster=False`, the call changes nothing.

The example below builds the CAE-ESN of the Kuramoto–Sivashinsky equation in two ways: in one
call, and from an autoencoder trained beforehand. The convolutional autoencoders need the
optional extra `romda[torch]`.

```python
from romda.models.data_driven import CAE1D, CAE1D_ESN

# U: snapshots of the Kuramoto–Sivashinsky equation, shape (1, N_t, 128, 1), time step dt
rom = CAE1D_ESN(data=U, dt=dt, n_latent=8, projector_epochs=500,   # autoencoder
                N_units=500, N_wash=20,                            # echo state network
                Nq=16)                                             # sensor points

cae = CAE1D(n_latent=8, epochs=500).fit(U)                         # trained once, reused
rom = CAE1D_ESN(data=U, dt=dt, projector=cae, N_units=500, N_wash=20, Nq=16)

psi, t = rom.time_integrate(Nt=200)                 # closed-loop forecast of [z; r]
rom.update_history(psi, t)
y = rom.get_observable_hist()                       # decoded field at the sensors
```

::: romda.models.data_driven.latent_rom.LatentROM
    options:
      docstring_style: google

## Sensor placement

The sensors are placed where the decoded field is most informative about the reduced
coordinates. For a nonlinear decoder, `LatentROM.spatial_basis` linearizes the decoder about
the mean reduced state, $\vect{z}_0 = \bar{\vect{z}}$, by central differences

$$
\matr{J}_{:,j} = \frac{\mathcal{D}(\vect{z}_0 + \Delta_j \vect{e}_j) - \mathcal{D}(\vect{z}_0 - \Delta_j
\vect{e}_j)}{2\Delta_j}, \qquad \Delta_j = 10^{-4}\max\left(|z_{0,j}|, 1\right),
$$

in which $\matr{J} \in \mathbb{R}^{N_x \times r}$ is the decoder Jacobian, $\vect{e}_j$ is the
$j$-th unit vector and $\Delta_j$ is the step. For POD, the basis is exact,
$\matr{J} = \matr{\Psi}$, and independent of $\vect{z}_0$. The Jacobian of an autoencoder is a
local basis, so the placement is a heuristic away from $\bar{\vect{z}}$.

With $\matr{J}_\mathrm{c} \in \mathbb{R}^{N_\mathrm{c} \times r}$ the rows of the basis at the $N_\mathrm{c}$ candidate
locations, `qr_sensor_points` computes the QR factorization with column pivoting
(Businger & Golub 1965)

$$
\matr{J}_\mathrm{c}^\mathrm{T}\matr{\Pi} = \matr{O}\matr{R} \quad (N_\mathrm{pt} \le r), \qquad
\left(\matr{J}_\mathrm{c}\matr{J}_\mathrm{c}^\mathrm{T}\right)\matr{\Pi} = \matr{O}\matr{R} \quad (N_\mathrm{pt} > r),
$$

in which $\matr{\Pi}$ is the permutation matrix, $\matr{O}$ is orthogonal, $\matr{R}$ is upper
triangular and $N_\mathrm{pt}$ is the number of sensor points. At each step, the pivoting selects the
candidate whose row of $\matr{J}_\mathrm{c}$ has the largest norm after the rows already selected are
projected out, which is a greedy maximization of the volume of the selected rows. This is the
sensor placement of Manohar et al. (2018), in which the second form oversamples when there are
more sensors than modes. For $N_\mathrm{pt} = r$, the pivots are the Q-DEIM interpolation points of
Drmač & Gugercin (2016). Each pivot is mapped to its grid point, $g = g_\mathrm{raw} \bmod N_\mathrm{g}$, in
which $g_\mathrm{raw}$ is the raw grid index of the pivot and $N_\mathrm{g}$ = `Nx * Ny` is the number of
grid points per component. Duplicate points are dropped (keeping the pivot order) and every
point records every field component, so `Nq` $= N_u N_\mathrm{pt}$ after construction, in which $N_u$ is
the number of field components. A warning reports fewer distinct points than requested.

| Option | Default | Effect |
|---|---|---|
| `Nq` | `10` | Number of sensor points to place; after construction, the number of observables |
| `qr_selection` | `True` | QR pivoting; `False` draws the points at random (seeded by the model's `rng`) |
| `domain_of_measurement` | `None` | Extent `[x0, x1, y0, y1]` the candidates are restricted to (needs the projector's `domain`) |
| `down_sample_measurement` | `None` | Stride over the candidate grid, one integer or one per axis |
| `sensor_locations` | `None` | Prescribed raw grid indices `u * Nx * Ny + g`; skips the placement |
| `measure_modes` | `False` | Observes $\vect{z}$ instead of sensors |

`select_sensors(...)` replaces the sensors of a trained model; `qr_selection=None` keeps the
current setting. Sensor locations are raw grid indices, $i N_\mathrm{g} + g$, in which $i$ is
the field component; `sensor_rows` maps them to rows of the flat field, which holds only the fluid points.
QR placement is designed for a number of sensors of the order of the number of modes; the random
placement is a baseline.



::: romda.models.data_driven.latent_rom.qr_sensor_points
    options:
      docstring_style: google

::: romda.models.data_driven.latent_rom.measurement_grid
    options:
      docstring_style: google

::: romda.models.data_driven.latent_rom.sensor_rows
    options:
      docstring_style: google

## Observables and data assimilation

The observables are the decoded field at the sensor rows, $\mathcal{S}$,

$$
\vect{y} = \left[\mathcal{D}(\vect{z})\right]_{\mathcal{S}}, \qquad \text{or} \qquad
\vect{y} = \matr{\Psi}_{\mathcal{S}}\vect{z} + \bar{\vect{q}}_{\mathcal{S}} \quad
\text{for POD},
$$

in which $[\cdot]_{\mathcal{S}}$ keeps the rows in $\mathcal{S}$. `get_observables` computes
them with `decode_at`, which decodes the whole field for an autoencoder and only the sensor rows
for POD. With `measure_modes=True`, the observables are the reduced coordinates,
$\vect{y} = \vect{z}$. The observation operator is therefore nonlinear for an autoencoder and
linear for POD; the ensemble filters need no Jacobian, because they evaluate the observables of
each member.

Any ensemble filter takes a latent reduced-order model as its forecast model. The analysis
updates the full state, $\vect{\psi} = [\vect{z}; \vect{r}]$: the reduced coordinates and the
reservoir states. The reference solution is the held-out snapshots read at the sensor
locations, as in `tests/test_latent_rom.py`. With the model `rom` trained on the first
`N_train` snapshots of a record `X`,

```python
import numpy as np
from romda.data_assimilation import run_da_loop
from romda.estimators import EnSRKF
from romda.observations import Observations

X_test = X[:, N_train:]                                       # snapshots the ROM has not seen
flat = X_test.transpose(1, 0, 2, 3).reshape(X_test.shape[1], -1)
y_true = flat[:, rom.sensor_locations]
t_true = np.arange(X_test.shape[1]) * rom.dt
truth = Observations(y_raw=y_true, y_true=y_true, t_true=t_true, t_start=0.5, t_stop=3.0, Nt_obs=5)

ensemble = EnSRKF(parent_model=rom, m=10, std_phi=0.1)
filter_ens = run_da_loop(ensemble, truth, std_obs=0.05, t_extra=0.5)
```

The filter can only correct what the forecaster can represent, so measure the closed-loop
forecast skill of the trained model before data assimilation. The tutorials
[43_DA_CAE_ESN_KS.ipynb](../tutorials/4_Real-time_DA_ROMs/43_DA_CAE_ESN_KS.md)
(Kuramoto–Sivashinsky) and
[44_DA_CAE_ESN_Kolmogorov.ipynb](../tutorials/4_Real-time_DA_ROMs/44_DA_CAE_ESN_Kolmogorov.md)
(Kolmogorov flow) assimilate sensor data into a CAE-ESN, and
[`scripts/caeesnda`](https://github.com/andreanovoa/romda/tree/main/scripts/caeesnda) holds
their configurations.

## Available reduced-order models

| Class | Projector | Forecaster | Module | Reference |
|---|---|---|---|---|
| `POD_ESN` | `POD` | `ESN_model` | `pod_esn` ([page](models_data_driven_pod_esn.md)) | Nóvoa & Magri (2025) |
| `CAE_ESN` | `CAE` (two-dimensional convolutional) | `ESN_model` | `autoencoder_roms` | Racca, Doan & Magri (2023) |
| `CAE1D_ESN` | `CAE1D` (one-dimensional, tanh) | `ESN_model` | `autoencoder_roms` | Özalp, Nóvoa & Magri (2026) |
| `MultiScaleCAE_ESN` | `MultiScaleCAE` (multi-scale, tanh) | `ESN_model` | `autoencoder_roms` | Racca, Doan & Magri (2023); Özalp, Nóvoa & Magri (2026) |

The POD classes take the latent size as `n_modes` (or `N_modes`) and the autoencoder classes as
`n_latent`. `autoencoder_roms` imports torch (the optional extra `romda[torch]`); the
`romda.models.data_driven` package loads it on first access. The CAE-ESN classes reproduce the
convolutional autoencoder echo state network of Racca, Doan & Magri (2023) and of Özalp, Nóvoa
& Magri (2026); the autoencoders and their options are documented in
[Convolutional autoencoders](convolutional_autoencoders.md).
Choose POD when the field is close to a low-rank linear subspace, and an autoencoder when POD
needs many more modes than the dimension of the attractor.

::: romda.models.data_driven.autoencoder_roms.CAE_ESN
    options:
      docstring_style: google

::: romda.models.data_driven.autoencoder_roms.CAE1D_ESN
    options:
      docstring_style: google

::: romda.models.data_driven.autoencoder_roms.MultiScaleCAE_ESN
    options:
      docstring_style: google

## Saving and reloading trained models

`model_config` stores trained models whose class implements `Configurable` (the dataclass
autoencoders `CAE`, `CAE1D` and `MultiScaleCAE`).

Each saved model is a directory of the store (default `results/model_configs`) named by a hash
of its configuration:

```
<store>/<hash>/model_config.yaml      the class, its options and the training-data name
<store>/<hash>/trained_matrices.npz   the trained arrays
```

`ModelConfig.to_hash` serializes the import path of the class, the constructor options (with
the defaults filled in and the options inferred from the data, such as `N_dim_in`) and
`training_data_filename` to sorted JSON, rounds floats to 12 decimals (to 12 significant digits
when the rounding would set them to zero), writes tuples as lists, and keeps the first 16
hexadecimal digits of the SHA-256 digest. `auto_load_or_create(model_class, data, **options)`
computes this hash, loads the model if the store holds both files, and otherwise trains it with
`model_class.from_data(data, **options)` and saves it under the hash (`auto_save=True`).
`force_create=True` retrains and overwrites. The hash contains the name of the training data,
not the data, so a model retrained on new data must be given a new `training_data_filename`.

A latent reduced-order model is not stored (`configurable = False`), because its construction
needs more than its options. Store its parts instead: restore the projector with
`auto_load_or_create` and pass it as `projector=`. The echo state network keeps its own store,
`rnns.esn_config` (`results/esn_configs`).

```python
from romda.models.data_driven import MultiScaleCAE, MultiScaleCAE_ESN, auto_load_or_create

cae = auto_load_or_create(MultiScaleCAE, data=W, training_data_filename="kolmogorov.h5", n_latent=16)
rom = MultiScaleCAE_ESN(data=W, dt=0.1, projector=cae, Nq=8, N_units=1000)
```

::: romda.models.data_driven.model_config.ModelConfig
    options:
      docstring_style: google

::: romda.models.data_driven.model_config.auto_load_or_create
    options:
      docstring_style: google

::: romda.models.data_driven.model_config.save_model_to_config
    options:
      docstring_style: google

::: romda.models.data_driven.model_config.load_model_from_config
    options:
      docstring_style: google

::: romda.models.data_driven.model_config.find_matching_config
    options:
      docstring_style: google

::: romda.models.data_driven.model_config.list_saved_configs
    options:
      docstring_style: google

## References

- Businger, P. & Golub, G. H. (1965). Linear least squares solutions by Householder
  transformations. *Numerische Mathematik*, 7, 269–276. [doi:10.1007/BF01436084](https://doi.org/10.1007/BF01436084)
- Drmač, Z. & Gugercin, S. (2016). A new selection operator for the discrete empirical
  interpolation method—improved a priori error bound and extensions. *SIAM Journal on
  Scientific Computing*, 38(2), A631–A648. [doi:10.1137/15M1019271](https://doi.org/10.1137/15M1019271),
  [arXiv:1505.00370](https://arxiv.org/abs/1505.00370)
- Manohar, K., Brunton, B. W., Kutz, J. N. & Brunton, S. L. (2018). Data-driven sparse sensor
  placement for reconstruction: demonstrating the benefits of exploiting known patterns.
  *IEEE Control Systems Magazine*, 38(3), 63–86. [doi:10.1109/MCS.2018.2810460](https://doi.org/10.1109/MCS.2018.2810460),
  [arXiv:1701.07569](https://arxiv.org/abs/1701.07569)
- Nóvoa, A. & Magri, L. (2025). Online model learning with data-assimilated reservoir
  computers. In *Computational Science – ICCS 2025 Workshops*, Lecture Notes in Computer
  Science 15910, 57–64. Springer, Cham. [doi:10.1007/978-3-031-97567-7_5](https://doi.org/10.1007/978-3-031-97567-7_5),
  [arXiv:2504.16767](https://arxiv.org/abs/2504.16767)
- Nóvoa, A., Racca, A. & Magri, L. (2024). Inferring unknown unknowns: regularized bias-aware
  ensemble Kalman filter. *Computer Methods in Applied Mechanics and Engineering*, 418, 116502.
  [doi:10.1016/j.cma.2023.116502](https://doi.org/10.1016/j.cma.2023.116502),
  [arXiv:2306.04315](https://arxiv.org/abs/2306.04315)
- Özalp, E., Nóvoa, A. & Magri, L. (2026). Real-time forecasting of chaotic dynamics from
  sparse data and autoencoders. *Computer Methods in Applied Mechanics and Engineering*, 450,
  118600. [doi:10.1016/j.cma.2025.118600](https://doi.org/10.1016/j.cma.2025.118600),
  [arXiv:2508.08729](https://arxiv.org/abs/2508.08729)
- Racca, A., Doan, N. A. K. & Magri, L. (2023). Predicting turbulent dynamics with the
  convolutional autoencoder echo state network. *Journal of Fluid Mechanics*, 975, A2.
  [doi:10.1017/jfm.2023.716](https://doi.org/10.1017/jfm.2023.716),
  [arXiv:2211.11379](https://arxiv.org/abs/2211.11379)
