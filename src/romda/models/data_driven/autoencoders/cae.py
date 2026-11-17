"""Convolutional autoencoder in PyTorch.

Typical usage example:

  cae = CAE(n_latent=8, channels=(16, 32, 64)).fit(X)
  Z = cae.encode(X)
  Q_hat = cae.decode(Z)
  mse = cae.score(X)

Ported from autoencoder-rom; co-author: Lukas Campbell (2026).
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from .autoencoder_base import conv_stages
from .base import FloatArray
from .torch_utils import PeriodicConvTranspose2d, TorchAutoencoder

__all__ = ["CAE"]


@dataclass(eq=False, repr=False)
class CAE(TorchAutoencoder):
    """A convolutional autoencoder on the snapshot grid.

    The encoder applies one convolution per entry of `channels`, each followed
    by `activation`, then flattens and maps to `n_latent` coefficients, linearly
    or through the dense layers `hidden`; with `latent_channels`, it keeps the
    bottleneck grid instead. The decoder maps back to the bottleneck grid in the
    same way and applies the mirrored transposed convolutions, with the last one
    linear. Solid points enter as zeros and are excluded from the
    loss. An axis of length 1 is not convolved (kernel 1, stride 1, no padding),
    so a grid `(Nx, 1)` is a one-dimensional field. `Autoencoder` documents the
    training options.

    Attributes:
      channels: Output channels of each encoder convolution.
      kernel_size: Convolution kernel size.
      stride: Convolution stride, or one per entry of `channels`. A stage with stride 1 keeps
        its grid: `stride=(1, 2, 2)` convolves the field at full resolution before halving it
        twice, which the decoder mirrors with a last convolution at full resolution.
      pad: Convolution padding.
      periodic_padding: Pad periodically instead of with zeros, for a field that is periodic along
        both axes (e.g. the Kolmogorov flow) or along the one axis of a grid `(Nx, 1)` (e.g. the
        Lorenz-96 ring): the encoder convolutions pad circularly and the decoder applies
        `PeriodicConvTranspose2d`. Every encoder stage must divide its grid exactly by its stride.
      decoder_upsampling: How each decoder stage enlarges the grid. `"transposed"` applies a strided
        transposed convolution; `"resize"` repeats every point (nearest-neighbour upsampling)
        and applies a stride-1 convolution, which avoids the checkerboard artefacts of a
        transposed convolution whose kernel is not a multiple of its stride (Odena, Dumoulin
        & Olah 2016, Distill). Both have the same weights per stage.
      hidden: Widths of the dense layers between the flattened bottleneck grid and the latent
        coefficients, each followed by `activation`; the decoder uses them in reverse order.
        Empty (the default): one linear map each way.
      latent_channels: Channels of a convolutional bottleneck; 0 (the default) keeps the dense one.
        With `latent_channels > 0` the reduced coordinates are `latent_channels` channels on the
        bottleneck grid, reached by a linear 1 x 1 convolution and mirrored in the decoder. The
        weights are shared along the grid, so their number does not grow with it; `n_latent` is set
        to `latent_channels` times the number of bottleneck grid points when the networks are
        built. Excludes `hidden`.

    Raises:
      ValueError: If an option is out of range, `activation` is
        unknown, or `device` is not a valid torch device name.
    """

    channels: Sequence[int] = (16, 32, 64)
    kernel_size: int = 3
    stride: int | Sequence[int] = 2
    pad: int = 1
    periodic_padding: bool = False
    decoder_upsampling: str = "transposed"
    hidden: Sequence[int] = ()
    latent_channels: int = 0

    _enc_conv: nn.Sequential | None = field(default=None, init=False)
    _enc_fc: nn.Module | None = field(default=None, init=False)
    _dec_fc: nn.Module | None = field(default=None, init=False)
    _dec_conv: nn.Sequential | None = field(default=None, init=False)
    _red_shape: tuple[int, int, int] = field(default=(0, 0, 0), init=False)

    def __post_init__(self) -> None:
        """Validates the options.

        Raises:
          ValueError: If an option is out of range, `activation` is
            unknown, or `device` is not a valid torch device name.
        """
        super().__post_init__()
        self.channels = tuple(self.channels)
        if not self.channels or any(c < 1 for c in self.channels):
            raise ValueError(f"channels must be non-empty with entries >= 1, got {self.channels}")
        self.hidden = tuple(self.hidden)
        if any(d < 1 for d in self.hidden):
            raise ValueError(f"hidden must all be >= 1, got {self.hidden}")
        if self.latent_channels < 0:
            raise ValueError(f"latent_channels must be >= 0, got {self.latent_channels}")
        if self.latent_channels and self.hidden:
            raise ValueError(f"hidden={self.hidden} needs the dense bottleneck, latent_channels=0")
        if self.decoder_upsampling not in ("transposed", "resize"):
            raise ValueError(f"decoder_upsampling must be 'transposed' or 'resize', got {self.decoder_upsampling!r}")
        if not isinstance(self.stride, int):
            self.stride = tuple(self.stride)
            if len(self.stride) != len(self.channels):
                raise ValueError(f"stride needs one entry per stage of channels={self.channels}, got {self.stride}")
        if self.kernel_size < 1 or min(self._strides()) < 1 or self.pad < 0:
            raise ValueError(
                f"need kernel_size >= 1, stride >= 1, pad >= 0; got {self.kernel_size}, {self.stride}, {self.pad}"
            )

    # ── Networks ──────────────────────────────────────────────────────────────

    @property
    def enc_conv(self) -> nn.Sequential:
        """The convolutional encoder stages.

        Raises:
          RuntimeError: If the networks have not been built.
        """
        if self._enc_conv is None:
            raise RuntimeError("CAE is not fitted; call fit() first")
        return self._enc_conv

    @property
    def enc_fc(self) -> nn.Module:
        """The map from the bottleneck grid to the latent space: linear without `hidden`, a 1 x 1
        convolution with `latent_channels`.

        Raises:
          RuntimeError: If the networks have not been built.
        """
        if self._enc_fc is None:
            raise RuntimeError("CAE is not fitted; call fit() first")
        return self._enc_fc

    @property
    def dec_fc(self) -> nn.Module:
        """The map from the latent space to the bottleneck grid: linear without `hidden`, a 1 x 1
        convolution with `latent_channels`.

        Raises:
          RuntimeError: If the networks have not been built.
        """
        if self._dec_fc is None:
            raise RuntimeError("CAE is not fitted; call fit() first")
        return self._dec_fc

    @property
    def dec_conv(self) -> nn.Sequential:
        """The transposed-convolution decoder stages.

        Raises:
          RuntimeError: If the networks have not been built.
        """
        if self._dec_conv is None:
            raise RuntimeError("CAE is not fitted; call fit() first")
        return self._dec_conv

    @property
    def n_params(self) -> int:
        """Number of trainable parameters, or 0 before the networks are built."""
        try:
            nets = self.networks()
        except RuntimeError:
            return 0
        return sum(p.numel() for net in nets for p in net.parameters())

    def decoder_networks(self) -> list[nn.Module]:
        """The decoder's linear map and its transposed convolutions.

        Raises:
          RuntimeError: If the networks have not been built.
        """
        return [self.dec_fc, self.dec_conv]

    def encode_grid(self, G: torch.Tensor) -> torch.Tensor:
        """Encodes a batch of scaled grid snapshots, differentiably.

        Args:
          G: Scaled snapshots with zeros at solid points, shape
            `(B, Nu, Nx, Ny)`.

        Returns:
          Latent codes, shape `(B, n_latent)`.

        Raises:
          RuntimeError: If the networks have not been built.
        """
        R = self.enc_conv(G)
        return self.enc_fc(R).flatten(1) if self.latent_channels else self.enc_fc(R.flatten(1))

    def decode_grid(self, Z: torch.Tensor) -> torch.Tensor:
        """Decodes a batch of latent codes onto the grid, differentiably.

        Args:
          Z: Latent codes, shape `(B, n_latent)`.

        Returns:
          Scaled fields, shape `(B, Nu, Nx, Ny)`.

        Raises:
          RuntimeError: If the networks have not been built.
        """
        c, h, w = self._red_shape
        if self.latent_channels:
            return self.dec_conv(self.dec_fc(Z.view(-1, self.latent_channels, h, w)))
        return self.dec_conv(self.dec_fc(Z).view(-1, c, h, w))

    def _networks_by_name(self) -> dict[str, nn.Module]:
        """Lists the networks by name, in the order the encoder and decoder apply them.

        Raises:
          RuntimeError: If the networks have not been built.
        """
        return {"enc_conv": self.enc_conv, "enc_fc": self.enc_fc, "dec_fc": self.dec_fc, "dec_conv": self.dec_conv}

    def _build_networks(self) -> None:
        """Creates freshly initialized encoder and decoder networks for the recorded grid.

        Raises:
          ValueError: If the grid is too small for the convolution stages.
        """
        c_in, nx, ny = self._grid()
        encoder, decoder = conv_stages((c_in, nx, ny), self.channels, self.kernel_size, self.stride, self.pad)
        c_red, (h_red, w_red) = encoder[-1].c_out, encoder[-1].size_out
        self._red_shape = (c_red, h_red, w_red)
        flat = c_red * h_red * w_red
        # an axis of length 1 is not convolved, so a grid (Nx, 1) is a one-dimensional field
        line = [n == 1 for n in (nx, ny)]
        k = tuple(1 if l else self.kernel_size for l in line)
        p = tuple(0 if l else self.pad for l in line)
        r = tuple(n // 2 for n in k)                                  # padding of the stride-1 resize convolutions
        strides = [tuple(1 if l else si for l in line) for si in self._strides()]   # per stage, per axis

        if self.periodic_padding and any(n_in != si * n_out for st, s in zip(encoder, strides)
                                         for n_in, n_out, si in zip(st.size_in, st.size_out, s)):
            raise ValueError(f"periodic_padding=True needs every stage to divide the grid by stride={self.stride}; "
                             f"a {nx} x {ny} grid with {len(self.channels)} stages does not")
        enc: list[nn.Module] = []
        for stage, s in zip(encoder, strides):
            conv = nn.Conv2d(stage.c_in, stage.c_out, k, s, p, padding_mode="circular" if self.periodic_padding else "zeros")
            enc += [conv, self._activation()]
        self._enc_conv = nn.Sequential(*enc).to(self.device)
        if self.latent_channels:
            self.n_latent = self.N_latent = self.latent_channels * h_red * w_red
            self._enc_fc = nn.Conv2d(c_red, self.latent_channels, 1).to(self.device)
            self._dec_fc = nn.Conv2d(self.latent_channels, c_red, 1).to(self.device)
        else:
            self._enc_fc = self._dense([flat, *self.hidden, self.N_latent]).to(self.device)
            self._dec_fc = self._dense([self.N_latent, *reversed(self.hidden), flat]).to(self.device)

        dec: list[nn.Module] = []
        for stage, s in zip(decoder, reversed(strides)):
            if dec:
                dec.append(self._activation())
            if self.decoder_upsampling == "resize":
                dec += [nn.Upsample(size=tuple(stage.size_out), mode="nearest"),
                        nn.Conv2d(stage.c_in, stage.c_out, k, 1, r,
                                  padding_mode="circular" if self.periodic_padding else "zeros")]
            else:
                dec.append(PeriodicConvTranspose2d(stage.c_in, stage.c_out, k, s, p) if self.periodic_padding else
                           nn.ConvTranspose2d(stage.c_in, stage.c_out, k, s, p, output_padding=stage.output_padding))
        self._dec_conv = nn.Sequential(*dec).to(self.device)

    def _strides(self) -> tuple[int, ...]:
        """The stride of each encoder stage."""
        return self.stride if isinstance(self.stride, tuple) else (self.stride,) * len(self.channels)

    def _dense(self, dims: Sequence[int]) -> nn.Module:
        """Builds linear layers with `activation` between consecutive ones.

        Args:
          dims: Layer widths, input first.

        Returns:
          A single `nn.Linear` for two widths, so that fits saved before `hidden` existed load
          unchanged; otherwise an `nn.Sequential` whose last layer is linear.
        """
        layers: list[nn.Module] = []
        for n_in, n_out in itertools.pairwise(dims):
            if layers:
                layers.append(self._activation())
            layers.append(nn.Linear(n_in, n_out))
        return layers[0] if len(layers) == 1 else nn.Sequential(*layers)

    # ── Projector ─────────────────────────────────────────────────────────────

    def fit(self, X: npt.NDArray[Any], X_val: npt.NDArray[Any] | None = None) -> CAE:
        """Trains the encoder and decoder on snapshots.

        Args:
          X: Snapshots on the grid, shape `(Nu, N_t, Nx, Ny)`, with NaN at solid
            points.
          X_val: Held-out snapshots in any accepted layout. When given, they
            are the validation block (centred and scaled with the training
            statistics) and every snapshot of `X` trains; otherwise the last
            `val_fraction` of `X` is held out.

        Returns:
          This autoencoder, fitted.

        Raises:
          ValueError: If `X` is not a valid grid record, `val_fraction` leaves no
            training snapshots, or the grid is too small for the convolution
            stages.
        """
        if np.ndim(X) != 4:
            raise ValueError(f"CAE convolves the grid: fit it on grid snapshots (Nu, N_t, Nx, Ny), got shape {np.shape(X)}")
        data_tr, data_val = self._split_fit(X, X_val)
        Nu, Nx, Ny = self._grid()
        if not self._resuming:
            self._build_networks()
        mask = torch.as_tensor(self._fluid_mask().reshape(Nx, Ny), dtype=torch.float32, device=self.device)[None, None]

        def loss_fn(batch: torch.Tensor, idx: torch.Tensor | None = None) -> torch.Tensor:
            """Computes the mean squared reconstruction error over fluid points."""
            recon = self.decode_grid(self.encode_grid(batch))
            return ((recon - batch) ** 2 * mask).sum() / (mask.sum() * batch.shape[0] * Nu)

        history = self._train(self.networks(), loss_fn, data_tr, data_val)
        self._finish_fit(history)
        return self

    def encode_torch(self, T: torch.Tensor) -> torch.Tensor:
        """Encodes scaled grid snapshots `(B, Nu, Nx, Ny)` into latent codes `(B, n_latent)`."""
        return self.encode_grid(T)

    def decode_torch(self, Z: torch.Tensor) -> torch.Tensor:
        """Decodes latent codes `(B, n_latent)` into scaled flat fields `(B, N_x)` at the fluid points.

        The flat layout groups the rows by component, as `_to_flat` does.
        """
        G = self.decode_grid(Z)
        Nu, Nx, Ny = self._grid()
        idx = (np.arange(Nu)[:, None] * (Nx * Ny) + np.flatnonzero(self._fluid_mask())[None, :]).ravel()
        return G.flatten(1)[:, torch.as_tensor(idx, device=G.device)]

    # ── Layouts ───────────────────────────────────────────────────────────────

    def _to_tensor(self, Q: FloatArray) -> torch.Tensor:
        """Scales zero-mean flat fields and places them on the grid as a tensor.

        Args:
          Q: Zero-mean flat fields, shape `(N_x, N_t)`.

        Returns:
          Scaled float32 fields with zeros at solid points, shape
          `(N_t, Nu, Nx, Ny)`.
        """
        Nu, Nx, Ny = self._grid()
        mask = self._fluid_mask()
        n_t = Q.shape[1]
        values = (Q / self.scale).reshape(Nu, int(mask.sum()), n_t).transpose(0, 2, 1)
        grid: FloatArray = np.zeros((Nu, n_t, Nx * Ny), dtype=Q.dtype)
        grid[:, :, mask] = values
        G = grid.reshape(Nu, n_t, Nx, Ny).transpose(1, 0, 2, 3)
        return torch.as_tensor(G, dtype=torch.float32, device=self.device)
