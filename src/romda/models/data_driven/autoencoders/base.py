r"""Base classes shared by every projector.

`Projector` is the interface every dimensionality-reduction building block
implements, linear (`POD`, `SPOD`) or nonlinear.

Every projector works on snapshots in one of three layouts:

    grid      (Nu, N_t, Nx, Ny)   NaN at solid points
    snapshot  (Nu, Nx, Ny)        one grid snapshot
    flat      (N_x, N_t)          fluid points only, N_x = Nu * N_fluid

The flat layout groups rows by component: row ``u * N_fluid + f`` holds
component ``u`` at the ``f``-th fluid point. `fit` accepts the grid layout, or an
already-flat data matrix (no grid, no solid mask).

The projector API is sklearn-like:

    p.fit(X)          learn the representation from data X
    p.encode(X)       X --> Z, shape (N_latent, N_t)
    p.decode(Z)       Z --> X_hat, flat, shape (N_x, N_t), mean restored
    p.reconstruct(X)  full round trip
    p.score(X)        mean squared reconstruction error
    p.decode_at(Z, i) decoded rows i only (sensor readout)
    p.spatial_basis() decoder Jacobian, the local spatial basis (Psi for POD)

The `Projector` interface and the layouts merge romda's original projector base
with the generalization by its co-author Lukas Campbell (autoencoder-rom, 2026).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ["FloatArray", "Projector"]

#: A real-valued array. Each parameter documents its own shape.
FloatArray = npt.NDArray[np.floating[Any]]
#: A boolean mask over the grid points.
BoolArray = npt.NDArray[np.bool_]


class Projector(ABC):
    r"""Abstract base for all dimensionality-reduction building blocks.

    Every projector shares the same sklearn-style interface: `fit` learns the
    representation from data, `encode` maps state space to the latent space,
    and `decode` maps back. `reconstruct`, `score`, `decode_at` and
    `spatial_basis` are concrete methods built on top of `encode`/`decode`.

    Attributes
    ----------
    N_latent : int
        Size of the latent (bottleneck) space. For `POD`/`SPOD` this is the
        number of modes retained.
    fitted : bool
        Whether `fit` has completed.
    grid_shape : tuple of int or None
        Grid shape ``(Nu, Nx, Ny)``; None when fitted on flat data.
    domain : list of float or None
        Physical extent ``[x0, x1, y0, y1]`` of the grid.
    fluid_mask_flat : np.ndarray or None
        Mask over the ``Nx * Ny`` grid points, True at fluid points (over the
        ``N_x`` rows when fitted on flat data).
    Q_mean : np.ndarray
        Temporal mean removed from the data during preprocessing, shape
        $(N_x, 1)$. Raises ``AttributeError`` if accessed before `fit`.
    """

    N_latent: int = 20
    fitted: bool = False
    grid_shape: tuple[int, int, int] | None = None
    domain: list[float] | None = None
    fluid_mask_flat: BoolArray | None = None
    _Q_mean: FloatArray | None = None
    _TKE: float | None = None

    # ── Interface ─────────────────────────────────────────────────────────────

    @abstractmethod
    def fit(self, X: npt.NDArray[Any]) -> Projector:
        r"""Learn the projection from snapshots.

        Parameters
        ----------
        X : np.ndarray
            Snapshots on the grid, shape $(N_u, N_t, N_x, N_y)$, or a flat data
            matrix $(N_x, N_t)$.

        Returns
        -------
        Projector
            The fitted instance (``self``).
        """

    @abstractmethod
    def encode(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        r"""Map snapshots to latent coefficients.

        Parameters
        ----------
        X : np.ndarray
            Snapshots in the grid, snapshot or flat layout.

        Returns
        -------
        np.ndarray
            Latent coefficients $\mathbf{Z}$, shape $(N_\mathrm{latent}, N_t)$.
        """

    @abstractmethod
    def decode(self, Z: npt.NDArray[Any]) -> npt.NDArray[Any]:
        r"""Map latent coefficients back to flat fields.

        Parameters
        ----------
        Z : np.ndarray
            Latent coefficients, shape $(N_\mathrm{latent}, N_t)$.

        Returns
        -------
        np.ndarray
            Flat fields with the temporal mean restored, shape $(N_x, N_t)$.
        """

    @property
    def Q_mean(self) -> FloatArray:
        """np.ndarray: Temporal mean of the training fields, shape ``(N_x, 1)``."""
        if self._Q_mean is None:
            raise AttributeError("Not fitted — call fit() first.")
        return self._Q_mean

    @Q_mean.setter
    def Q_mean(self, value: FloatArray) -> None:
        self._Q_mean = value

    def reconstruct(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Full round trip: `encode` then `decode` (flat fields)."""
        return self.decode(self.encode(X))

    def score(self, X: npt.NDArray[Any]) -> float:
        r"""Mean squared reconstruction error over every fluid point and snapshot,

        $$
        \mathrm{MSE} = \frac{1}{N_x N_t}\,
        \lVert \mathbf{X} - \mathrm{decode}(\mathrm{encode}(\mathbf{X})) \rVert_F^2 ,
        $$

        equal to the error of the zero-mean fields since the mean cancels.

        Parameters
        ----------
        X : np.ndarray
            Snapshots in the grid, snapshot or flat layout.

        Returns
        -------
        float
            Mean squared reconstruction error.
        """
        X_hat = self.reconstruct(X)
        return float(np.mean((self._flat(X) - X_hat) ** 2))

    # ── Sensor readout and local basis ────────────────────────────────────────

    def decode_at(self, Z: npt.NDArray[Any], idx: npt.NDArray[np.intp] | None = None) -> npt.NDArray[Any]:
        """Decode latent coefficients and keep the requested rows.

        Parameters
        ----------
        Z : np.ndarray
            Latent coefficients, shape ``(N_latent, N_t)``.
        idx : np.ndarray, optional
            Rows of the flat field to keep, such as sensor locations. None
            keeps every row.

        Returns
        -------
        np.ndarray
            The selected rows of the decoded field, shape ``(len(idx), N_t)``.
        """
        X_hat = self.decode(Z)
        return X_hat if idx is None else X_hat[idx]

    def spatial_basis(self, z0: npt.NDArray[Any] | None = None) -> FloatArray:
        """Decoder Jacobian at a latent state, by central differences on `decode`.

        Each column is the change in the decoded field per unit change in one
        latent coefficient. Linear projectors override it with their exact basis.

        Parameters
        ----------
        z0 : np.ndarray, optional
            Latent state to linearize about, shape ``(N_latent,)``. None uses
            the origin.

        Returns
        -------
        np.ndarray
            The Jacobian, shape ``(N_x, N_latent)``.

        Raises
        ------
        ValueError
            If `z0` does not have `N_latent` entries.
        """
        n = self.N_latent
        z: FloatArray = np.zeros(n) if z0 is None else np.asarray(z0, dtype=np.float64).ravel()
        if z.size != n:
            raise ValueError(f"z0 has {z.size} entries, expected N_latent={n}")

        eps: FloatArray = 1e-4 * np.maximum(np.abs(z), 1.0)
        cols: list[FloatArray] = []
        for k in range(n):
            zp, zm = z.copy(), z.copy()
            zp[k] += eps[k]
            zm[k] -= eps[k]
            dp: FloatArray = self.decode(zp[:, np.newaxis])
            dm: FloatArray = self.decode(zm[:, np.newaxis])
            cols.append((dp - dm).ravel() / (2.0 * eps[k]))
        return np.column_stack(cols)

    # ── Layouts ───────────────────────────────────────────────────────────────

    def preprocess_snapshot(self, X: npt.NDArray[Any], subtract_mean: bool = True) -> FloatArray:
        r"""Build the zero-mean flat data matrix from snapshots.

        Before `fit` completes, also records the grid shape, the fluid mask
        (NaN-masked solid points are detected and removed) and the temporal mean.

        Parameters
        ----------
        X : np.ndarray
            Before `fit`: grid snapshots $(N_u, N_t, N_x, N_y)$ or a flat data
            matrix $(N_x, N_t)$. After `fit`: the grid, snapshot or flat layout.
        subtract_mean : bool
            Record the temporal mean of `X` rather than zeros (before `fit` only).
            Default True.

        Returns
        -------
        np.ndarray
            Zero-mean data matrix $\mathbf{Q}$, shape
            $(N_\mathrm{fluid} \cdot N_u, N_t)$.

        Raises
        ------
        ValueError
            If `X` does not match an accepted layout or the fitted grid, or
            holds non-finite values at fluid points.
        """
        if not self.fitted:
            return self._fit_geometry(X, subtract_mean)
        return self._flat(X) - self.Q_mean

    def _fit_geometry(self, X: npt.NDArray[Any], subtract_mean: bool = True) -> FloatArray:
        """Record the grid shape, fluid mask and temporal mean of snapshots.

        Parameters
        ----------
        X : np.ndarray
            Grid snapshots ``(Nu, N_t, Nx, Ny)`` with NaN at solid points, or a
            flat data matrix ``(N_x, N_t)``.
        subtract_mean : bool
            Record the temporal mean of `X` rather than zeros.

        Returns
        -------
        np.ndarray
            Zero-mean flat fields, shape ``(N_x, N_t)``.

        Raises
        ------
        ValueError
            If `X` is neither two- nor four-dimensional, or holds non-finite
            values at fluid points.
        """
        X = np.asarray(X)
        if X.ndim == 2:
            # Already-flat data matrix (N_x, N_t): no grid, no solid mask.
            self.fluid_mask_flat = np.ones(X.shape[0], dtype=bool)
            X_flat = X
        elif X.ndim == 4:
            Nu, _, Nx, Ny = X.shape
            self.grid_shape = (Nu, Nx, Ny)
            self.fluid_mask_flat = ~np.isnan(X[0, 0]).ravel()
            X_flat = self._to_flat(X)
        else:
            raise ValueError(f"expected grid snapshots (Nu, N_t, Nx, Ny) or flat (N_x, N_t), got shape {X.shape}")
        if not np.isfinite(X_flat).all():
            raise ValueError("snapshots hold non-finite values at points that are fluid in the first snapshot")
        self.Q_mean = X_flat.mean(axis=1, keepdims=True) if subtract_mean else np.zeros_like(X_flat[:, :1])
        Q = X_flat - self.Q_mean
        # total kinetic energy, for relative error metrics
        self._TKE = 0.5 * float(np.sum(np.mean(Q**2, axis=1)))
        return Q

    def _flat(self, X: npt.NDArray[Any]) -> FloatArray:
        """Convert snapshots in any accepted layout to flat fields (mean included).

        Raises
        ------
        ValueError
            If `X` does not match an accepted layout or the fitted grid.
        """
        X = np.asarray(X)
        n_x = self.Q_mean.shape[0]
        if X.ndim == 2:
            if X.shape[0] != n_x:
                raise ValueError(f"flat snapshots must have N_x={n_x} rows, got shape {X.shape}")
            return X
        if X.ndim == 3:
            X = X[:, np.newaxis]
        if X.ndim != 4:
            raise ValueError(f"expected grid, snapshot, or flat input, got shape {X.shape}")
        Nu, _, Nx, Ny = X.shape
        if (Nu, Nx, Ny) != self.grid_shape:
            raise ValueError(f"expected grid shape {self.grid_shape}, got {(Nu, Nx, Ny)}")
        return self._to_flat(X)

    def _to_flat(self, X: FloatArray) -> FloatArray:
        """Select the fluid points of grid snapshots ``(Nu, N_t, Nx, Ny)``.

        Returns the flat fields ``(Nu * N_fluid, N_t)``, grouped by component.
        """
        mask = self._fluid_mask()
        X_fluid = X.reshape(X.shape[0], X.shape[1], -1)[:, :, mask]
        return X_fluid.transpose(0, 2, 1).reshape(-1, X.shape[1])

    def _to_physical_grid(self, X_hat: FloatArray) -> FloatArray:
        """Place flat fields back on the grid; exact inverse of `_to_flat`.

        Parameters
        ----------
        X_hat : np.ndarray
            Flat fields, shape ``(N_x, N_t)`` or ``(N_x,)``.

        Returns
        -------
        np.ndarray
            Fields on the grid with NaN at solid points, shape
            ``(Nu, N_t, Nx, Ny)``, or ``(Nu, Nx, Ny)`` for a single snapshot.
        """
        Nu, Nx, Ny = self._grid()
        mask = self._fluid_mask()
        if X_hat.ndim == 1:
            X_hat = X_hat[:, np.newaxis]
        n_t = X_hat.shape[1]

        values = X_hat.reshape(Nu, int(mask.sum()), n_t).transpose(0, 2, 1)
        out: FloatArray = np.full((Nu, n_t, Nx * Ny), np.nan)
        out[:, :, mask] = values
        out = out.reshape(Nu, n_t, Nx, Ny)
        return out[:, 0] if n_t == 1 else out

    def grid_index_to_flat_rows(self, grid_idx) -> np.ndarray:
        """Map raw-grid indices to rows of the masked flat representation.

        Both layouts use variable-block ordering: raw index
        ``var * Nx * Ny + g`` (``g`` the flattened ``(x, y)`` position) maps to
        flat row ``var * N_fluid + fluid_pos``.

        Parameters
        ----------
        grid_idx : array-like of int
            Raw-grid indices (e.g. sensor locations) at fluid points.

        Returns
        -------
        np.ndarray of int
            Row indices into `Psi` / `Q_mean`.

        Raises
        ------
        ValueError
            If some requested grid points are not fluid points.
        """
        from ..latent_rom import sensor_rows  # latent_rom imports this module

        return sensor_rows(np.asarray(grid_idx).ravel(), self._grid(), self._fluid_mask())

    def _field_scale(self, Q: FloatArray) -> FloatArray:
        """Standard deviation of each field component, shape ``(N_x, 1)``.

        A component with zero variance gets 1. Without a grid (flat fit) the
        whole matrix counts as one component.
        """
        Nu = self.grid_shape[0] if self.grid_shape is not None else 1
        n_fluid = Q.shape[0] // Nu
        scale: FloatArray = np.ones((Q.shape[0], 1), dtype=Q.dtype)
        for u in range(Nu):
            rows = slice(u * n_fluid, (u + 1) * n_fluid)
            s = float(Q[rows].std())
            scale[rows, 0] = s if s > 0 else 1.0
        return scale

    def _grid(self) -> tuple[int, int, int]:
        """The grid shape; raises RuntimeError if unknown (flat fit)."""
        if self.grid_shape is None:
            raise RuntimeError("grid_shape is unknown; fit on grid snapshots first")
        return self.grid_shape

    def _fluid_mask(self) -> BoolArray:
        """The fluid mask; raises RuntimeError if unknown."""
        if self.fluid_mask_flat is None:
            raise RuntimeError("fluid_mask_flat is unknown; fit on grid snapshots first")
        return self.fluid_mask_flat
