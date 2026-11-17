"""Data-driven reduced-order models.

A latent ROM pairs a projector, which reduces flow snapshots to latent
coefficients, with a forecaster, which advances those coefficients in time:

    autoencoders/   projectors (`Projector`): POD, SPOD and the autoencoders
    rnns/           forecasters: ESN_model (and its config store)
    latent_rom.py   LatentROM, the base class every ROM shares
    pod_esn.py      POD_ESN
    autoencoder_roms.py  CAE_ESN, CAE1D_ESN and MultiScaleCAE_ESN, the CAE-ESNs (torch)

Every ROM is a `dynamodels.Model`, so it is a forecast model for romda's
estimators. Importing this package does not import torch: the classes that
need it (the optional extra `romda[torch]`) load on first access.
"""

import importlib
from typing import Any

from . import autoencoders, rnns
from .autoencoders import POD, SPOD, Projector
from .latent_rom import LatentROM, measurement_grid, qr_sensor_points, sensor_rows
from .linear_model import LinearModel
from .pod_esn import POD_ESN, synthetic_field
from .rnns import ESN_model, phi_to_esn_layout
from .training import TrainingHistory

__all__ = [
    "ESN_model",
    "phi_to_esn_layout",
    "synthetic_field",
    "LinearModel",
    "POD_ESN",
    "LatentROM",
    "measurement_grid",
    "qr_sensor_points",
    "sensor_rows",
    "TrainingHistory",
    # projector building blocks (the autoencoders subpackage)
    "Projector",
    "POD",
    "SPOD",
]

#: Lazily loaded classes, by name: the module that defines them. The projectors
#: and forecasters of the subpackages load through their own maps.
_LAZY: dict[str, str] = {
    # save/load of any `Configurable` model
    'ModelConfig': '.model_config',
    'auto_load_or_create': '.model_config',
    'save_model_to_config': '.model_config',
    'load_model_from_config': '.model_config',
    # the CAE-ESNs, among them those of Racca, Doan & Magri (2023) and Özalp, Nóvoa & Magri (2026); torch
    'CAE_ESN': '.autoencoder_roms',
    'CAE1D_ESN': '.autoencoder_roms',
    'MultiScaleCAE_ESN': '.autoencoder_roms',
}
__all__ += [*_LAZY, *autoencoders._LAZY]



def __getattr__(name: str) -> Any:
    """Load an optional class (most need torch or JAX) the first time it is accessed."""
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    for sub in (autoencoders, rnns):
        if name in getattr(sub, '_LAZY', ()):
            return getattr(sub, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
