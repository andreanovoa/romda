# Data-driven models

## Summary

**File:** `src/romda/models/data_driven/`. All use `DiscreteIntegrator`. `ESN_model` (and its
save/load config, `ESNConfig`) lives in the `rnns/` subpackage — future RNN
architectures join it there. `ESN_model` mixes in `EchoStateNetwork` — the reservoir
core from the external [`echostatenetwork`](https://andreanovoa.github.io/EchoStateNetwork/)
package (re-exported by `romda.models.data_driven` for convenience; see its own
documentation for the reservoir API) — and `POD_ESN` mixes in `ESN_model` and
[`POD`](models_data_driven_projectors.md).

| Class | Key parameters | Notes |
|---|---|---|
| `ESN_model` | `N_units`, `rho`, `sigma_in`, `N_wash` | Inherits `EchoStateNetwork`; requires training `data` at init |
| `LatentROM` | `Nq`, `measure_modes`, `qr_selection` | Base of every projector + forecaster ROM (any `Projector`, any forecaster `Model`) |
| `POD_ESN` | `N_modes`, `sensor_locations` | `LatentROM` + `ESN_model` + `POD`; sensor placement via QR |
| `LinearModel` | `F` (transition matrix), `Q_noise` | $\vect{\psi}_{t+1} = \matr{F}\vect{\psi}_t + \vect{\eta}_t$ |

## Latent reduced-order models

A latent ROM is a projector, which reduces the snapshots to latent coefficients,
combined with a forecaster, which advances them in time. `LatentROM`
(`latent_rom.py`) holds what every combination shares: it fits the projector, trains
the forecaster on the latent training trajectory, places sensors by QR column pivoting
on the decoder Jacobian (`Projector.spatial_basis`, exactly `Psi` for POD), and reads the
observables as the decoded field at the sensors (`Projector.decode_at`). Every ROM is
therefore a forecast model for the estimators in the same way as `POD_ESN`.
`LatentROM` and the generalized `Projector` interface were written by Lukas Campbell
(autoencoder-rom, 2026).

The `Projector` hierarchy (`POD`/`SPOD`) and the standalone decomposition functions
(`pod_utils.py`) live in the `autoencoders/` subpackage. Import everything from
`romda.models.data_driven` (or its `autoencoders` subpackage).

## Model pages

- [Echo State Network model](models_data_driven_esn.md)
- [POD-ESN model](models_data_driven_pod_esn.md)
- [Linear model](models_data_driven_linear.md)
- [Projectors and modal decompositions](models_data_driven_projectors.md)
