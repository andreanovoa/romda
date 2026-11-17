"""Base class, training loop, and weight saving shared by the PyTorch autoencoders.

Typical usage example:

  @dataclass(eq=False, repr=False)
  class MyAE(TorchAutoencoder):
      def fit(self, X, X_val=None):
          X_tr, X_val = self._split_fit(X, X_val)
          ...
          self._finish_fit(self._train(nets, loss_fn, X_tr, X_val))
          return self

Ported from autoencoder-rom; co-author: Lukas Campbell (2026).
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from ..training import TrainingHistory
from .autoencoder_base import Autoencoder
from .base import FloatArray

__all__ = ["ACTIVATIONS", "PeriodicConvTranspose2d", "TorchAutoencoder"]

#: Hidden layer activations, by name.
ACTIVATIONS: dict[str, type[nn.Module]] = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "identity": nn.Identity,
}


class PeriodicConvTranspose2d(nn.ConvTranspose2d):
    """Transposed convolution on a periodic grid: each axis grows from n to `stride * n`.

    The input is padded periodically by ceil(kernel / stride) samples on each side, the
    transposed convolution runs without padding, and the `stride * n` samples that a zero-padded
    layer with padding `pad` would place at 0, ..., stride * n - 1 are kept. Every periodic image
    of an input sample then contributes once, so the layer is the transposed convolution of the
    periodic signal. The weights are those of `nn.ConvTranspose2d`, so a saved state loads in either.
    `kernel_size`, `stride` and `pad` are integers or per-axis pairs.
    """

    def __init__(self, c_in: int, c_out: int, kernel_size: int | tuple[int, int], stride: int | tuple[int, int],
                 pad: int | tuple[int, int]) -> None:
        super().__init__(c_in, c_out, kernel_size, stride, padding=0)
        self.pad = (pad, pad) if isinstance(pad, int) else tuple(pad)
        self.wrap = tuple(-(-k // s) for k, s in zip(self.kernel_size, self.stride))

    def forward(self, x: torch.Tensor, output_size: Any = None) -> torch.Tensor:
        (ah, aw), (sh, sw), (ph, pw) = self.wrap, self.stride, self.pad
        h, w = x.shape[-2:]
        if ah <= h and aw <= w:
            x = nn.functional.pad(x, (aw, aw, ah, ah), mode="circular")
        else:                                                         # pad wider than the grid: periodic indices
            x = x[..., torch.arange(-ah, h + ah, device=x.device) % h, :][..., torch.arange(-aw, w + aw, device=x.device) % w]
        y = super().forward(x)
        oh, ow = ph + sh * ah, pw + sw * aw
        return y[..., oh:oh + sh * h, ow:ow + sw * w]


#: Maps a batch of scaled snapshots, and during training their row indices into
#: the training block, to the scalar loss. Validation calls it without indices.
LossFn = Callable[..., torch.Tensor]


@dataclass(eq=False, repr=False)
class TorchAutoencoder(Autoencoder):
    """Options, training loop and torch-side projector shared by the torch autoencoders.

    Subclasses define `encode_torch`, `decode_torch` and `decoder_networks`;
    `encode` and `decode` wrap them in NumPy.

    Attributes:
      activation: Hidden layer activation. One of `"tanh"`, `"relu"`,
        `"elu"`, `"gelu"`, or `"identity"`.
      device: Torch device name, such as `"cpu"` or `"cuda"`. Saved
        configurations leave it out; pass it when restoring.
      compile: Compile the training loss, i.e. the forward pass through
        every network, with `torch.compile`. Faster for small networks on a
        GPU, after a first epoch spent compiling; the trained weights are the
        same networks. Saved configurations leave it out.
      warm_start: Continue training a fitted autoencoder from its current
        weights, at its last learning rate, instead of rebuilding the
        networks; `epochs` then counts the additional epochs and the history
        grows. Saved configurations leave it out.

    Raises:
      ValueError: If an option is out of range, `activation` is
        unknown, or `device` is not a valid torch device name.
    """

    activation: str = "tanh"
    device: str = "cpu"
    compile: bool = False
    warm_start: bool = False

    _config_exclude: ClassVar[tuple[str, ...]] = ("grid_shape", "device", "compile", "warm_start")

    def __post_init__(self) -> None:
        """Validates the options.

        Raises:
          ValueError: If an option is out of range, `activation` is
            unknown, or `device` is not a valid torch device name.
        """
        super().__post_init__()
        if self.activation not in ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {sorted(ACTIVATIONS)}, got {self.activation!r}"
            )
        try:
            torch.device(self.device)
        except RuntimeError as e:
            raise ValueError(f"device {self.device!r} is not a valid torch device") from e

    def _prepare_fit(
        self, X: npt.NDArray[Any], X_val: npt.NDArray[Any] | None = None
    ) -> tuple[FloatArray, int]:
        """Records the geometry and field scale, then seeds torch.

        See `Autoencoder._prepare_fit`.
        """
        prepared = super()._prepare_fit(X, X_val)
        torch.manual_seed(self.seed)
        return prepared

    def _split_fit(
        self, X: npt.NDArray[Any], X_val: npt.NDArray[Any] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepares the fit and splits the scaled snapshots into training and validation blocks.

        Args:
          X: Training snapshots. Without `X_val`, the last `val_fraction` of
            them is the validation block.
          X_val: Validation snapshots in any accepted layout, or `None`.

        Returns:
          The training and validation blocks, as `_to_tensor` lays them out.
        """
        # whether `fit` continues from the current weights; read before `_prepare_fit` clears `fitted`
        self._resuming = self.warm_start and self.fitted
        Q, n_val = self._prepare_fit(X, X_val)
        data = self._to_tensor(Q)
        n_tr = Q.shape[1] - n_val
        return data[:n_tr], data[n_tr:]

    @abstractmethod
    def _networks_by_name(self) -> dict[str, nn.Module]:
        """Lists the networks by name.

        Raises:
          RuntimeError: If the networks have not been built.
        """

    @abstractmethod
    def _build_networks(self) -> None:
        """Creates freshly initialized networks for the recorded geometry."""

    def _activation(self) -> nn.Module:
        """Creates one instance of the hidden layer activation."""
        return ACTIVATIONS[self.activation]()

    # ── Torch-side projector ──────────────────────────────────────────────────

    def networks(self) -> list[nn.Module]:
        """Lists every network, in the order of `_networks_by_name`.

        Raises:
          RuntimeError: If the networks have not been built.
        """
        return list(self._networks_by_name().values())

    @abstractmethod
    def _to_tensor(self, Q: FloatArray) -> torch.Tensor:
        """Scales zero-mean flat fields `(N_x, N_t)` into the batch layout `encode_torch` takes."""

    @abstractmethod
    def encode_torch(self, T: torch.Tensor) -> torch.Tensor:
        """Encodes a batch of scaled snapshots, differentiably.

        Args:
          T: Scaled snapshots, as `_to_tensor` lays them out.

        Returns:
          Latent codes, shape `(B, n_latent)`.
        """

    @abstractmethod
    def decode_torch(self, Z: torch.Tensor) -> torch.Tensor:
        """Decodes a batch of latent codes into scaled flat fields, differentiably.

        Args:
          Z: Latent codes, shape `(B, n_latent)`.

        Returns:
          Scaled zero-mean fields, shape `(B, N_x)`.
        """

    @abstractmethod
    def decoder_networks(self) -> list[nn.Module]:
        """Lists the networks `decode_torch` applies.

        Raises:
          RuntimeError: If the networks have not been built.
        """

    def encode(self, X: npt.NDArray[Any]) -> FloatArray:
        """Encodes snapshots into latent coefficients.

        Args:
          X: Snapshots in the grid, snapshot, or flat layout.

        Returns:
          Latent coefficients, shape `(n_latent, N_t)`.

        Raises:
          RuntimeError: If `fit` has not completed.
          ValueError: If `X` does not match the fitted grid.
        """
        self._check_fitted()
        T = self._to_tensor(self.preprocess_snapshot(X))
        for net in self.networks():
            net.eval()
        with torch.no_grad():
            Z = self.encode_torch(T)
        return Z.cpu().numpy().T

    def decode(self, Z: npt.NDArray[Any]) -> FloatArray:
        """Decodes latent coefficients into flat fields.

        Args:
          Z: Latent coefficients, shape `(n_latent, N_t)`.

        Returns:
          Flat fields with the temporal mean restored, shape `(N_x, N_t)`.

        Raises:
          RuntimeError: If `fit` has not completed.
          ValueError: If `Z` is not two-dimensional with `n_latent` rows.
        """
        self._check_fitted()
        Z = self._check_latent(Z)
        for net in self.networks():
            net.eval()
        with torch.no_grad():
            Q_hat = self.decode_torch(torch.as_tensor(Z.T, dtype=torch.float32, device=self.device))
        return Q_hat.cpu().numpy().T * self.scale + self.Q_mean

    def _train(
        self, nets: Sequence[nn.Module], loss_fn: LossFn, X_tr: torch.Tensor, X_val: torch.Tensor
    ) -> TrainingHistory:
        """Trains networks and restores the weights with the lowest validation loss.

        Args:
          nets: Networks whose parameters the loss depends on.
          loss_fn: Maps a batch of scaled snapshots to the scalar loss.
          X_tr: Training snapshots, one per entry along the first axis.
          X_val: Validation snapshots, in the same layout. May be empty.

        Returns:
          The training history.
        """
        if self.compile:
            loss_fn = torch.compile(loss_fn)
        params = [p for net in nets for p in net.parameters()]
        opt = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay,
                               fused=True if params and params[0].is_cuda else None)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            factor=self.lr_factor,
            patience=self.lr_patience,
            threshold=self.threshold,
            min_lr=self.min_lr,
        )

        best: list[dict[str, torch.Tensor]] | None = None
        best_val, wait = float("inf"), 0
        history = TrainingHistory()
        if self._resuming:  # continue the stored history; only an improvement on its best replaces the weights
            prev = self.training_history
            history = TrainingHistory(list(prev.train), list(prev.val), list(prev.lr))
            if prev.lr:
                for group in opt.param_groups:
                    group["lr"] = prev.lr[-1]
            if prev.val:
                best_val = min(prev.val)

        for _ in range(self.epochs):
            history.lr.append(float(opt.param_groups[0]["lr"]))
            history.train.append(self._train_epoch(nets, loss_fn, opt, X_tr))
            if X_val.shape[0] == 0:
                continue

            for net in nets:
                net.eval()
            with torch.no_grad():
                v = loss_fn(X_val).item()
            history.val.append(v)
            sched.step(v)
            if v < best_val * (1.0 - self.threshold):
                best_val, wait = v, 0
                best = [{k: t.detach().clone() for k, t in net.state_dict().items()} for net in nets]
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best is not None:
            for net, state in zip(nets, best, strict=True):
                net.load_state_dict(state)
        return history

    def _train_epoch(
        self,
        nets: Sequence[nn.Module],
        loss_fn: LossFn,
        opt: torch.optim.Optimizer,
        X_tr: torch.Tensor,
    ) -> float:
        """Runs one shuffled pass over the training snapshots.

        Args:
          nets: Networks whose parameters the loss depends on.
          loss_fn: Maps a batch of scaled snapshots to the scalar loss.
          opt: Optimizer over the networks' parameters.
          X_tr: Training snapshots, one per entry along the first axis.

        Returns:
          The mean loss over the pass.
        """
        for net in nets:
            net.train()
        # Drawn on the CPU, then moved with a blocking copy.
        order = torch.randperm(X_tr.shape[0]).to(X_tr.device)

        total = 0.0
        for start in range(0, X_tr.shape[0], self.batch_size):
            idx = order[start : start + self.batch_size]
            batch = X_tr[idx]
            opt.zero_grad()
            loss = loss_fn(batch, idx)
            loss.backward()
            opt.step()
            total += loss.item() * batch.shape[0]
        return total / X_tr.shape[0]

    def _weight_arrays(self) -> dict[str, npt.NDArray[Any]]:
        """Collects each network's state dictionary.

        Returns:
          Each tensor as an array, named `<network>.<key>`.
        """
        return {
            f"{name}.{key}": value.detach().cpu().numpy()
            for name, net in self._networks_by_name().items()
            for key, value in net.state_dict().items()
        }

    def _load_weight_arrays(self, arrays: Mapping[str, npt.NDArray[Any]]) -> None:
        """Builds the networks and loads their state dictionaries.

        Args:
          arrays: Arrays holding the entries `_weight_arrays` returned.

        Raises:
          RuntimeError: If a network's stored state is missing a key or holds an
            unexpected one.
        """
        self._build_networks()
        for name, net in self._networks_by_name().items():
            prefix = f"{name}."
            state = {
                k.removeprefix(prefix): torch.from_numpy(np.array(v)) for k, v in arrays.items() if k.startswith(prefix)
            }
            net.load_state_dict(state)

