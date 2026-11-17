"""Options, training state and save/restore shared by the trained projectors,
and the layer geometry of the convolutional ones.

Ported from autoencoder-rom; co-author: Lukas Campbell (2026).
"""

from __future__ import annotations

import warnings
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

import numpy as np
import numpy.typing as npt

from ..configurable import Configurable, restore_tuples
from ..training import TrainingHistory
from .base import FloatArray, Projector

if TYPE_CHECKING:
    from typing import Self

__all__ = ["Autoencoder", "DecoderStage", "EncoderStage", "conv_stages"]


@dataclass(eq=False, repr=False)
class Autoencoder(Projector, Configurable):
    """Options, validation, and training state shared by the autoencoders.

    `fit` scales each field component by its standard deviation, trains on the
    mean squared reconstruction error with Adam, holds out the last
    `val_fraction` of the snapshots, reduces the learning rate when the
    validation loss plateaus, stops early, and keeps the parameters with the
    lowest validation loss.

    Attributes:
      n_latent: Dimension of the latent space. `N_latent` holds the same value.
      learning_rate: Initial Adam step size.
      epochs: Maximum number of training epochs.
      batch_size: Minibatch size.
      val_fraction: Fraction of the snapshots held out for validation, taken
        from the end of the record. 0 trains on every snapshot and disables
        early stopping; the learning rate then stays at `learning_rate` (no
        plateau decay). The temporal mean and the field scale come from the
        training snapshots only.
      weight_decay: L2 penalty coefficient.
      patience: Epochs without improvement before training stops.
      threshold: Relative improvement in the validation loss that counts as
        progress, for both early stopping and the learning-rate schedule.
      lr_factor: Factor the learning rate decays by on a plateau.
      lr_patience: Plateau epochs before the learning rate decays.
      min_lr: Lower bound on the learning rate.
      seed: Random seed for initialization and shuffling.
      scale_fields: Divide each field component by its standard deviation
        before training. False trains on the zero-mean fields themselves, so
        the loss is the plain reconstruction MSE (a linear autoencoder then
        recovers the POD subspace of the data).
      grid_shape: Grid shape `(Nu, Nx, Ny)`. `fit` sets it from the data.
      domain: Physical extent `[x0, x1, y0, y1]` of the grid.
      training_history: Losses and learning rates of the last `fit`.

    Raises:
      ValueError: If an option is out of range or `grid_shape` does not have
        three entries.
    """

    n_latent: int = 10
    learning_rate: float = 1e-3
    epochs: int = 500
    batch_size: int = 32
    val_fraction: float = 0.2
    weight_decay: float = 0.0
    patience: int = 50
    threshold: float = 1e-4
    lr_factor: float = 0.5
    lr_patience: int = 10
    min_lr: float = 1e-6
    seed: int = 0
    scale_fields: bool = True
    grid_shape: tuple[int, int, int] | None = None
    domain: list[float] | None = None

    training_history: TrainingHistory = field(default_factory=TrainingHistory, init=False)
    _scale: FloatArray | None = field(default=None, init=False)

    _config_exclude: ClassVar[tuple[str, ...]] = ("grid_shape",)

    def __post_init__(self) -> None:
        """Validates the options.

        Raises:
          ValueError: If an option is out of range or `grid_shape` does not have
            three entries.
        """
        self.N_latent = self.n_latent
        if self.grid_shape is not None and len(self.grid_shape) != 3:
            raise ValueError(f"grid_shape must be (Nu, Nx, Ny), got {self.grid_shape}")
        if self.n_latent < 1:
            raise ValueError(f"n_latent must be >= 1, got {self.n_latent}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in [0, 1), got {self.val_fraction}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if self.patience < 1:
            raise ValueError(f"patience must be >= 1, got {self.patience}")
        if self.threshold < 0:
            raise ValueError(f"threshold must be >= 0, got {self.threshold}")
        if not 0.0 < self.lr_factor < 1.0:
            raise ValueError(f"lr_factor must be in (0, 1), got {self.lr_factor}")
        if self.lr_patience < 0:
            raise ValueError(f"lr_patience must be >= 0, got {self.lr_patience}")
        if self.min_lr < 0:
            raise ValueError(f"min_lr must be >= 0, got {self.min_lr}")

    @property
    def scale(self) -> FloatArray:
        """The per-field scale inputs are divided by, shape `(N_x, 1)`.

        Raises:
          RuntimeError: If `fit` has not been called.
        """
        if self._scale is None:
            raise RuntimeError(f"{type(self).__name__} is not fitted; call fit() first")
        return self._scale

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Number of trainable parameters, or 0 before `fit`."""

    def _prepare_fit(
        self, X: npt.NDArray[Any], X_val: npt.NDArray[Any] | None = None
    ) -> tuple[FloatArray, int]:
        """Records the geometry and field scale of the training snapshots.

        The temporal mean and the field scale come from the training snapshots
        only; the validation block is centred with that mean.

        Args:
          X: Snapshots on the grid, shape `(Nu, N_t, Nx, Ny)`, or flat
            `(N_x, N_t)`.
          X_val: Validation snapshots in any accepted layout. `None` holds out
            the last `val_fraction` of `X`.

        Returns:
          The zero-mean flat fields, training snapshots first and the
          validation block last, shape `(N_x, N_t)`, and the number of
          validation snapshots.

        Raises:
          ValueError: If `X` is not a valid record, the validation block holds
            non-finite values at fluid points, or `val_fraction` leaves no
            training snapshots.
        """
        self.fitted = False
        X = np.asarray(X)
        if X.ndim not in (2, 4):
            raise ValueError(f"expected grid snapshots (Nu, N_t, Nx, Ny) or flat (N_x, N_t), got shape {X.shape}")
        n_t = X.shape[1]
        n_val = int(round(self.val_fraction * n_t)) if X_val is None else 0
        if n_t - n_val < 1:
            raise ValueError(f"val_fraction={self.val_fraction} leaves no training snapshots out of {n_t}")
        if X_val is None and self.val_fraction > 0 and n_val == 0:
            warnings.warn(f"val_fraction={self.val_fraction} holds out no snapshots out of {n_t}; "
                          "training without validation, early stopping or learning-rate decay", stacklevel=3)
        Q = self._fit_geometry(X[:, : n_t - n_val])
        self._scale = self._field_scale(Q) if self.scale_fields else np.ones((Q.shape[0], 1), dtype=Q.dtype)
        if X_val is None:
            if not n_val:
                return Q, 0
            X_val = X[:, n_t - n_val :]
        Q_val = self._flat(X_val) - self.Q_mean
        if not np.isfinite(Q_val).all():
            raise ValueError("validation snapshots hold non-finite values at fluid points")
        return np.concatenate([Q, Q_val], axis=1), Q_val.shape[1]

    def _finish_fit(self, history: TrainingHistory) -> None:
        """Records the training history and marks the autoencoder fitted.

        Args:
          history: The history the training loop returned.
        """
        self.training_history = history
        self.fitted = True

    def _check_fitted(self) -> None:
        """Raises if `fit` has not completed.

        Raises:
          RuntimeError: If `fit` has not completed.
        """
        if not self.fitted or self._scale is None:
            raise RuntimeError(f"{type(self).__name__} is not fitted; call fit() first")

    def _check_latent(self, Z: npt.NDArray[Any]) -> FloatArray:
        """Checks the shape of latent coefficients.

        Args:
          Z: Latent coefficients.

        Returns:
          `Z` as an array, shape `(n_latent, N_t)`.

        Raises:
          ValueError: If `Z` is not two-dimensional with `n_latent` rows.
        """
        Z = np.asarray(Z)
        if Z.ndim != 2 or Z.shape[0] != self.N_latent:
            raise ValueError(f"Z must be (n_latent={self.N_latent}, N_t), got shape {Z.shape}")
        return Z

    # ── Saving and restoring ──────────────────────────────────────────────────

    def trained_arrays(self) -> dict[str, npt.NDArray[Any]]:
        """Collects the geometry, field statistics, history, and weights.

        Returns:
          The arrays, by name.

        Raises:
          RuntimeError: If `fit` has not completed.
        """
        self._check_fitted()
        arrays: dict[str, npt.NDArray[Any]] = {
            "Q_mean": self.Q_mean,
            "scale": self.scale,
            **({"grid_shape": np.asarray(self.grid_shape)} if self.grid_shape is not None else {}),
            "fluid_mask_flat": self._fluid_mask(),
            **self.training_history.to_arrays(),
        }
        if self._TKE is not None:
            arrays["TKE"] = np.asarray(self._TKE)
        return arrays | self._weight_arrays()

    @classmethod
    def from_trained(cls, options: Mapping[str, Any], arrays: Mapping[str, npt.NDArray[Any]], **overrides: Any) -> Self:
        """Rebuilds a fitted autoencoder.

        Args:
          options: Options from `config_options`. Options the class no longer
            defines are ignored.
          arrays: Arrays from `trained_arrays`.
          **overrides: Options to use instead of the stored ones, such as
            `device`.

        Returns:
          The fitted autoencoder.
        """
        stored = {k: v if k == "domain" else restore_tuples(v) for k, v in options.items()}
        model = cls(**cls._known_options(stored, overrides))
        model.Q_mean = np.asarray(arrays["Q_mean"])
        model._scale = np.asarray(arrays["scale"])
        if "grid_shape" in arrays:  # absent for a fit on flat (N_x, N_t) data
            Nu, Nx, Ny = (int(v) for v in arrays["grid_shape"])
            model.grid_shape = (Nu, Nx, Ny)
        model.fluid_mask_flat = np.asarray(arrays["fluid_mask_flat"], dtype=bool)
        if "TKE" in arrays:
            model._TKE = float(arrays["TKE"])
        model._load_weight_arrays(arrays)
        model._finish_fit(TrainingHistory.from_arrays(arrays))
        return model

    @abstractmethod
    def _weight_arrays(self) -> dict[str, npt.NDArray[Any]]:
        """Collects the trained weights.

        Returns:
          The weights, by name.
        """

    @abstractmethod
    def _load_weight_arrays(self, arrays: Mapping[str, npt.NDArray[Any]]) -> None:
        """Builds the networks from the recorded geometry and loads trained weights.

        Args:
          arrays: Arrays holding the entries `_weight_arrays` returned.
        """


# ── Convolutional geometry ────────────────────────────────────────────────────


class EncoderStage(NamedTuple):
    """One convolution of the convolutional encoder.

    Attributes:
      c_in: Input channels.
      c_out: Output channels.
      size_in: Input grid size `(Nx, Ny)`.
      size_out: Output grid size.
    """

    c_in: int
    c_out: int
    size_in: tuple[int, int]
    size_out: tuple[int, int]


class DecoderStage(NamedTuple):
    """One transposed convolution of the convolutional decoder.

    Attributes:
      c_in: Input channels.
      c_out: Output channels.
      output_padding: Extra rows and columns that restore the encoder's input
        size.
      size_out: Output grid size `(Nx, Ny)`.
    """

    c_in: int
    c_out: int
    output_padding: tuple[int, int]
    size_out: tuple[int, int]


def conv_stages(
    grid_shape: tuple[int, int, int],
    channels: Sequence[int],
    kernel_size: int,
    stride: int | Sequence[int],
    pad: int,
) -> tuple[tuple[EncoderStage, ...], tuple[DecoderStage, ...]]:
    """Computes the layer sizes of a convolutional autoencoder.

    The decoder mirrors the encoder, so each transposed convolution restores the
    grid size and channel count of the matching convolution's input.

    Args:
      grid_shape: Input grid shape `(Nu, Nx, Ny)`.
      channels: Output channels of each encoder convolution.
      kernel_size: Convolution kernel size.
      stride: Convolution stride, or one per entry of `channels`.
      pad: Convolution padding.

    Returns:
      The encoder stages, input first, and the decoder stages, bottleneck first.

    Raises:
      ValueError: If a convolution's input is smaller than its padded kernel.
    """
    k, p = kernel_size, pad
    strides = (stride,) * len(channels) if isinstance(stride, int) else tuple(stride)
    c, h, w = grid_shape
    encoder: list[EncoderStage] = []
    for i, (c_out, s) in enumerate(zip(channels, strides, strict=True)):
        if h + 2 * p < k or w + 2 * p < k:
            raise ValueError(
                f"stage {i} input {h}x{w} is smaller than kernel_size={k} with pad={p}; use fewer entries in channels"
            )
        size_out = ((h + 2 * p - k) // s + 1, (w + 2 * p - k) // s + 1)
        encoder.append(EncoderStage(c, c_out, (h, w), size_out))
        c, (h, w) = c_out, size_out

    decoder: list[DecoderStage] = []
    for stage, s in zip(reversed(encoder), reversed(strides)):
        (h, w), (th, tw) = stage.size_out, stage.size_in
        padding = (th - ((h - 1) * s - 2 * p + k), tw - ((w - 1) * s - 2 * p + k))
        decoder.append(DecoderStage(stage.c_out, stage.c_in, padding, stage.size_in))
    return tuple(encoder), tuple(decoder)
