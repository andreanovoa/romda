"""The convolutional autoencoders of the CAE-ESN (MagriLab/DA-CAE-ESN).

`MultiScaleCAE` is the autoencoder of the two-dimensional Kolmogorov flow of Racca,
Doan & Magri (2023), Predicting turbulent dynamics with the convolutional
autoencoder echo state network, JFM 975, A2, and of Özalp, Nóvoa & Magri (2026),
Real-time forecasting of chaotic dynamics from sparse data and autoencoders, CMAME
450, 118600 (network `multi_scale_cae`, run `kolmogorov_re30`). `CAE1D` is the
autoencoder of the Kuramoto-Sivashinsky equation of the second paper (network
`cae_ks_l66`, run `ks_l20pi`). Both are reimplemented from the layer sizes of
github.com/MagriLab/DA-CAE-ESN; the layer names match that repository, so its
trained state dictionaries load with `load_state_dicts`.

Typical usage example:

  cae = MultiScaleCAE(n_latent=16).fit(W)      # W: vorticity, (1, N_t, 48, 48)
  Z = cae.encode(W)
  W_hat = cae.decode(Z)
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from .base import FloatArray
from .cae import CAE
from .torch_utils import PeriodicConvTranspose2d

__all__ = ["CAE1D", "MultiScaleCAE"]


@dataclass(eq=False, repr=False)
class _TanhCAE(CAE):
    """Shared parts of the CAE-ESN autoencoders: an `encoder` and a `decoder` network,
    `activation` on every stage (the output included). The default activation,
    tanh, bounds the output, so the fields are divided by their maximum absolute
    value and the temporal mean is not removed. Training, early
    stopping and the saved arrays are those of `CAE`, whose `channels`,
    `kernel_size`, `stride`, `pad`, `hidden` and `latent_channels` are unused here."""

    _encoder: nn.Module | None = field(default=None, init=False)
    _decoder: nn.Module | None = field(default=None, init=False)

    def _networks_by_name(self) -> dict[str, nn.Module]:
        """The encoder and the decoder.

        Raises:
          RuntimeError: If the networks have not been built.
        """
        if self._encoder is None or self._decoder is None:
            raise RuntimeError(f"{type(self).__name__} is not fitted; call fit() first")
        return {"encoder": self._encoder, "decoder": self._decoder}

    def decoder_networks(self) -> list[nn.Module]:
        """The decoder.

        Raises:
          RuntimeError: If the networks have not been built.
        """
        return [self.networks()[1]]

    # ── Scaling: maximum absolute value, mean kept ────────────────────────────

    def _fit_geometry(self, X: npt.NDArray[Any], subtract_mean: bool = True) -> FloatArray:
        """Records the geometry and a zero mean: the output layer (tanh by default) reconstructs the full field."""
        del subtract_mean
        return super()._fit_geometry(X, subtract_mean=False)

    def _field_scale(self, Q: FloatArray) -> FloatArray:
        """The maximum absolute value of the training fields, the same for every point."""
        return np.full((Q.shape[0], 1), float(np.abs(Q).max()), dtype=Q.dtype)

    # ── Weights of the reference implementation ───────────────────────────────

    @classmethod
    def _options_from_state(cls, state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        """The constructor options a reference state dictionary fixes."""
        key = next(k for k in state if k.endswith("enc_linear_dense.weight"))
        return {"n_latent": state[key].shape[0]}

    @classmethod
    def load_state_dicts(
        cls,
        state: Mapping[str, torch.Tensor],
        grid_shape: tuple[int, int, int],
        scale: float,
        **options: Any,
    ) -> _TanhCAE:
        """Builds a fitted autoencoder from a state dictionary of the reference repository.

        Args:
          state: The `torch.load` of a `best_model.pth`, with `encoder.*` and
            `decoder.*` entries.
          grid_shape: The grid `(Nu, Nx, Ny)`.
          scale: The maximum absolute value the reference divided the data by.
          **options: Constructor options, such as `device`.

        Returns:
          The fitted autoencoder.
        """
        model = cls(**cls._options_from_state(state), **options)
        model.grid_shape = tuple(grid_shape)
        model.fluid_mask_flat = np.ones(int(np.prod(grid_shape)), dtype=bool)
        model.Q_mean = np.zeros((model.fluid_mask_flat.size, 1))
        model._scale = np.full((model.fluid_mask_flat.size, 1), float(scale))
        model._build_networks()
        for name, net in model._networks_by_name().items():
            net.load_state_dict({k.removeprefix(f"{name}."): v for k, v in state.items() if k.startswith(f"{name}.")})
        model.fitted = True
        return model


@dataclass(eq=False, repr=False)
class CAE1D(_TanhCAE):
    """The CAE-ESN autoencoder of a periodic one-dimensional field.

    The encoder applies four convolutions (2, 4, 8 and 16 channels; kernels 8,
    8, 3, 3; strides 1, 2, 2, 2), flattens, and maps to `n_latent` coefficients.
    The decoder maps back to the bottleneck and applies three stride-2
    transposed convolutions (kernels 3, 5, 8), each followed by a stride-1
    one (kernel 3). The layers restore the grid length for some Nx only (96 and
    128 among them). Snapshots are grids `(Nu, N_t, Nx, 1)`; each component is
    an input channel.

    Raises:
      ValueError: If the grid is not one-dimensional, or its length does not
        give back the same length through the decoder.
    """

    def encode_grid(self, G: torch.Tensor) -> torch.Tensor:
        """Encodes scaled snapshots `(B, Nu, Nx, 1)` into latent codes `(B, n_latent)`."""
        return self.networks()[0](G[..., 0])

    def decode_grid(self, Z: torch.Tensor) -> torch.Tensor:
        """Decodes latent codes `(B, n_latent)` into scaled fields `(B, Nu, Nx, 1)`."""
        return self.networks()[1](Z)[..., None]

    def _build_networks(self) -> None:
        """Creates freshly initialized networks for the recorded grid.

        Raises:
          ValueError: If the grid is not one-dimensional or the decoder does not
            restore its length.
        """
        if self.periodic_padding:
            raise NotImplementedError("periodic_padding=True is an option of CAE and MultiScaleCAE, not of CAE1D")
        nu, nx, ny = self._grid()
        if ny != 1:
            raise ValueError(f"CAE1D convolves one-dimensional fields: grid (Nu, Nx, 1), got {(nu, nx, ny)}")
        act = self._activation
        n_red = nx + 2 - 8 + 1                               # enc_conv0: kernel 8, padding 1
        for k in (8, 3, 3):                                  # enc_conv1-3: stride 2
            n_red = (n_red - k) // 2 + 1
        n_flat = 16 * n_red
        n_out = n_red
        for k in (3, 5, 8):                                  # dec_tconv1-3 then their kernel-3 partners
            n_out = (n_out - 1) * 2 + k
            n_out += 2 if k == 3 else 0                      # dec_tconv1_2 has no padding
        if n_out != nx:
            raise ValueError(f"CAE1D's layers map {nx} grid points to {n_out}; choose Nx that they restore, e.g. 96 or 128")

        self._encoder = nn.Sequential(OrderedDict([
            ("enc_conv0", nn.Conv1d(nu, 2, kernel_size=8, stride=1, padding=1)), ("tanh0", act()),
            ("enc_conv1", nn.Conv1d(2, 4, kernel_size=8, stride=2)), ("tanh1", act()),
            ("enc_conv2", nn.Conv1d(4, 8, kernel_size=3, stride=2)), ("tanh2", act()),
            ("enc_conv3", nn.Conv1d(8, 16, kernel_size=3, stride=2)), ("tanh3", act()),
            ("enc_flat", nn.Flatten()),
            ("enc_linear_dense", nn.Linear(n_flat, self.N_latent)), ("tanh4", act()),
        ])).to(self.device)
        self._decoder = nn.Sequential(OrderedDict([
            ("dec_linear_dense", nn.Linear(self.N_latent, n_flat)), ("dec_tanh_dense", act()),
            ("dec_unflat", nn.Unflatten(1, (16, n_red))),
            ("dec_tconv1", nn.ConvTranspose1d(16, 8, kernel_size=3, stride=2)),
            ("dec_tconv1_2", nn.ConvTranspose1d(8, 8, kernel_size=3, stride=1, padding=0)), ("dec_tanh1", act()),
            ("dec_tconv2", nn.ConvTranspose1d(8, 4, kernel_size=5, stride=2)),
            ("dec_tconv2_2", nn.ConvTranspose1d(4, 4, kernel_size=3, stride=1, padding=1)), ("dec_tanh2", act()),
            ("dec_tconv3", nn.ConvTranspose1d(4, 2, kernel_size=8, stride=2)),
            ("dec_tconv3_2", nn.ConvTranspose1d(2, nu, kernel_size=3, stride=1, padding=1)), ("dec_tanh3", act()),
        ])).to(self.device)
        # the reference initializes the encoder weights with Xavier-normal (its `xavier_normal` run)
        for module in self._encoder:
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_normal_(module.weight)


@dataclass(eq=False, repr=False)
class MultiScaleCAE(_TanhCAE):
    """The multi-scale CAE-ESN autoencoder of a two-dimensional field.

    One encoder-decoder branch per entry of `kernel_sizes`; the latent codes and
    the reconstructions of the branches are summed. Each encoder applies four
    convolutions (`channels`, by default 4, 8, 16 and 32; strides 1, 2, 2, 2; the first one
    padded by 1), flattens, and maps to `n_latent` coefficients. Each decoder maps
    back to the bottleneck and applies three stride-2 transposed convolutions and
    a stride-1 one with kernel `k + 1` and padding 1. The layers restore the grid
    for some sizes only (48 x 48 among them). Snapshots are grids
    `(Nu, N_t, Nx, Ny)`; each component is an input channel.

    Attributes:
      kernel_sizes: Kernel size of each branch (odd with `periodic_padding=True`).
      channels: Output channels of the four encoder convolutions of each branch; the
        decoder mirrors them. The default (4, 8, 16, 32) is the reference architecture.
      periodic_padding: Pad periodically, for a field that is periodic along both axes: every
        convolution pads circularly by `(k - 1) // 2`, so the grid halves exactly at each
        stride-2 stage (sides must be multiples of 8), the transposed convolutions are
        `PeriodicConvTranspose2d`, and the last layer is a periodic convolution of kernel
        `k` instead of the transposed convolution of kernel `k + 1`. False (default)
        reproduces the reference architecture, which pads with zeros.

    Raises:
      ValueError: If a branch does not give back the grid size.
    """

    kernel_sizes: Sequence[int] = (3, 5, 7)
    channels: Sequence[int] = (4, 8, 16, 32)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.kernel_sizes = tuple(int(k) for k in self.kernel_sizes)
        self.channels = tuple(int(c) for c in self.channels)
        if self.channels == CAE.channels:  # configurations saved before this option hold the unused CAE default
            self.channels = (4, 8, 16, 32)
        if len(self.channels) != 4 or min(self.channels) < 1:
            raise ValueError(f"channels must be four entries >= 1, got {self.channels}")

    def encode_grid(self, G: torch.Tensor) -> torch.Tensor:
        """Encodes scaled snapshots `(B, Nu, Nx, Ny)` into latent codes `(B, n_latent)`."""
        return sum(branch(G) for branch in self.networks()[0])

    def decode_grid(self, Z: torch.Tensor) -> torch.Tensor:
        """Decodes latent codes `(B, n_latent)` into scaled fields `(B, Nu, Nx, Ny)`."""
        return sum(branch(Z) for branch in self.networks()[1])

    @staticmethod
    def _sizes(n: int, k: int) -> tuple[int, int]:
        """Bottleneck size along one axis of `n` points, and the size the decoder gives back."""
        red = n - k + 3                                      # enc_conv0: stride 1, padding 1
        for _ in range(3):
            red = (red - k) // 2 + 1
        out = red
        for _ in range(3):
            out = (out - 1) * 2 + k
        return red, out + k - 2                              # dec_deconv0: kernel k + 1, padding 1

    def _build_networks(self) -> None:
        """Creates freshly initialized networks for the recorded grid.

        Raises:
          ValueError: If a branch does not give back the grid size.
        """
        nu, nx, ny = self._grid()
        act = self._activation
        c0, c1, c2, c3 = self.channels
        encoders, decoders = [], []
        if self.periodic_padding:
            if nx % 8 or ny % 8 or any(k % 2 == 0 for k in self.kernel_sizes):
                raise ValueError(f"periodic_padding=True needs grid sides that are multiples of 8 and odd kernels; "
                                 f"got a {nx} x {ny} grid and kernel_sizes={self.kernel_sizes}")
            for k in self.kernel_sizes:
                encoders.append(self._periodic_encoder(nu, k, c3 * (nx // 8) * (ny // 8)))
                decoders.append(self._periodic_decoder(nu, k, nx // 8, ny // 8))
        for k in ([] if self.periodic_padding else self.kernel_sizes):
            (hx, ox), (hy, oy) = self._sizes(nx, k), self._sizes(ny, k)
            if (ox, oy) != (nx, ny):
                raise ValueError(f"the kernel-{k} branch maps a {nx} x {ny} grid to {ox} x {oy}; "
                                 f"choose a grid it restores, e.g. 48 x 48")
            n_flat = c3 * hx * hy
            encoders.append(nn.Sequential(OrderedDict([
                ("enc_conv0", nn.Conv2d(nu, c0, kernel_size=k, stride=1, padding=1)), ("tanh0", act()),
                ("enc_conv1", nn.Conv2d(c0, c1, kernel_size=k, stride=2)), ("tanh1", act()),
                ("enc_conv2", nn.Conv2d(c1, c2, kernel_size=k, stride=2)), ("tanh2", act()),
                ("enc_conv3", nn.Conv2d(c2, c3, kernel_size=k, stride=2)), ("tanh3", act()),
                ("enc_flat", nn.Flatten()),
                ("enc_linear_dense", nn.Linear(n_flat, self.N_latent)), ("tanh4", act()),
            ])))
            decoders.append(nn.Sequential(OrderedDict([
                ("dec_linear_dense", nn.Linear(self.N_latent, n_flat)), ("tanh0", act()),
                ("dec_unflat", nn.Unflatten(1, (c3, hx, hy))),
                ("dec_deconv3", nn.ConvTranspose2d(c3, c2, kernel_size=k, stride=2)), ("tanh1", act()),
                ("dec_deconv2", nn.ConvTranspose2d(c2, c1, kernel_size=k, stride=2)), ("tanh2", act()),
                ("dec_deconv1", nn.ConvTranspose2d(c1, c0, kernel_size=k, stride=2)), ("tanh3", act()),
                ("dec_deconv0", nn.ConvTranspose2d(c0, nu, kernel_size=k + 1, stride=1, padding=1)), ("tanh4", act()),
            ])))
        self._encoder = nn.ModuleList(encoders).to(self.device)
        self._decoder = nn.ModuleList(decoders).to(self.device)
        # the reference initializes the weights of every layer with Kaiming-uniform
        for module in [*self._encoder.modules(), *self._decoder.modules()]:
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_uniform_(module.weight)

    def _periodic_encoder(self, nu: int, k: int, n_flat: int) -> nn.Sequential:
        """Encoder branch of kernel `k` with periodic padding: the grid halves at each stride-2 stage."""
        act, p = self._activation, (k - 1) // 2
        c0, c1, c2, c3 = self.channels
        conv = lambda c_in, c_out, s: nn.Conv2d(c_in, c_out, k, s, p, padding_mode="circular")  # noqa: E731
        return nn.Sequential(OrderedDict([
            ("enc_conv0", conv(nu, c0, 1)), ("tanh0", act()),
            ("enc_conv1", conv(c0, c1, 2)), ("tanh1", act()),
            ("enc_conv2", conv(c1, c2, 2)), ("tanh2", act()),
            ("enc_conv3", conv(c2, c3, 2)), ("tanh3", act()),
            ("enc_flat", nn.Flatten()),
            ("enc_linear_dense", nn.Linear(n_flat, self.N_latent)), ("tanh4", act()),
        ]))

    def _periodic_decoder(self, nu: int, k: int, hx: int, hy: int) -> nn.Sequential:
        """Decoder branch of kernel `k` with periodic padding: the grid doubles at each stride-2 stage."""
        act, p = self._activation, (k - 1) // 2
        c0, c1, c2, c3 = self.channels
        return nn.Sequential(OrderedDict([
            ("dec_linear_dense", nn.Linear(self.N_latent, c3 * hx * hy)), ("tanh0", act()),
            ("dec_unflat", nn.Unflatten(1, (c3, hx, hy))),
            ("dec_deconv3", PeriodicConvTranspose2d(c3, c2, k, 2, p)), ("tanh1", act()),
            ("dec_deconv2", PeriodicConvTranspose2d(c2, c1, k, 2, p)), ("tanh2", act()),
            ("dec_deconv1", PeriodicConvTranspose2d(c1, c0, k, 2, p)), ("tanh3", act()),
            ("dec_deconv0", nn.Conv2d(c0, nu, k, 1, p, padding_mode="circular")), ("tanh4", act()),
        ]))

    @classmethod
    def _options_from_state(cls, state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        n = 1 + max(int(k.split(".")[1]) for k in state if k.startswith("encoder."))
        ks = tuple(int(state[f"encoder.{i}.enc_conv0.weight"].shape[-1]) for i in range(n))
        # the periodic last layer has kernel k; the reference transposed one has k + 1
        periodic = int(state["decoder.0.dec_deconv0.weight"].shape[-1]) == ks[0]
        channels = tuple(int(state[f"encoder.0.enc_conv{i}.weight"].shape[0]) for i in range(4))
        return super()._options_from_state(state) | {"kernel_sizes": ks, "periodic_padding": periodic, "channels": channels}
