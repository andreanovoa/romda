"""
romda.models.data_driven.autoencoders
=====================================

Dimensionality-reduction building blocks (projectors) for data-driven ROMs.

All projectors share the same sklearn-style API (see `Projector`):

    p.fit(X)         -- learn the representation from data X, grid (Nu, N_t, Nx, Ny) or flat (N_x, N_t)
    p.encode(X)      -- X --> Z (N_latent, N_t)
    p.decode(Z)      -- Z (N_latent, N_t) --> X_hat (N_x, N_t)
    p.reconstruct(X) -- full round trip
    p.score(X)       -- mean squared reconstruction error
    p.N_latent       -- size of the latent (bottleneck) space

Class hierarchy
---------------

    Projector (ABC)               shared interface, layouts, sensor readout   [base.py]
    └── POD(Projector)            Proper Orthogonal Decomposition (linear)    [pod.py]
        └── SPOD(POD)             Spectral POD (Sieber et al. JFM 2016)       [pod.py]

The autoencoders (`CAE`, `CAE1D`, `MultiScaleCAE`) need torch, the optional
extra `romda[torch]`. They load their module on first access, so importing
this package does not import torch.

These are pure dimensionality-reduction tools, with no temporal forecaster.
Combine one with an ESN to build a complete ROM (`romda.models.data_driven.LatentROM`).
"""

from __future__ import annotations

import importlib
from typing import Any

from .base import Projector
from .pod import POD, SPOD
from .pod_utils import print_spod_towne_summary, spod_towne

__all__ = ['Projector', 'POD', 'SPOD', 'spod_towne', 'print_spod_towne_summary']

#: Lazily loaded classes, by name: the module that defines them.
_LAZY: dict[str, str] = {
    'Autoencoder': '.autoencoder_base',  # options and save/restore of the trained projectors
    'conv_stages': '.autoencoder_base',
    'TorchAutoencoder': '.torch_utils',   # torch training loop shared by the autoencoders
    'CAE': '.cae',                        # convolutional autoencoder, torch
    'CAE1D': '.cae_esn',                  # the CAE-ESN autoencoders (MagriLab/DA-CAE-ESN), torch
    'MultiScaleCAE': '.cae_esn',
}

__all__ += list(_LAZY)


def __getattr__(name: str) -> Any:
    """Load a torch or JAX projector the first time it is accessed."""
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
