"""Convolutional autoencoder reduction with an echo state network forecaster: the CAE-ESN.

Importing this module imports torch.

Typical usage example:

  cae = CAE1D.load_state_dicts(torch.load('DA-CAE-ESN_ks_l20pi.pth'), grid_shape=(1, 128, 1), scale=1.0)
  rom = CAE1D_ESN(data=X, dt=0.25, projector=cae, N_units=1000)
  psi, t = rom.time_integrate(Nt=200)

Ported from autoencoder-rom; co-author: Lukas Campbell (2026).
"""

from __future__ import annotations

from .autoencoders.cae import CAE
from .autoencoders.cae_esn import CAE1D, MultiScaleCAE
from .latent_rom import LatentROM
from .rnns import ESN_model

__all__ = ["CAE_ESN", "CAE1D_ESN", "MultiScaleCAE_ESN"]


class CAE_ESN(LatentROM, ESN_model, CAE):
    """Convolutional autoencoder with an echo state network.

    `LatentROM` documents construction and observables.
    """


class CAE1D_ESN(LatentROM, ESN_model, CAE1D):
    """The CAE-ESN of Özalp, Nóvoa & Magri (2026): `CAE1D` with an echo state network.

    `LatentROM` documents construction and observables.
    """


class MultiScaleCAE_ESN(LatentROM, ESN_model, MultiScaleCAE):
    """The CAE-ESN of Racca, Doan & Magri (2023): `MultiScaleCAE` with an echo state network.

    `LatentROM` documents construction and observables.
    """


