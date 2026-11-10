"""
pod.py
======
Linear projectors: `POD` and `SPOD` (Sieber). Thin sklearn-style wrappers
around the decomposition algorithms in `.pod_utils`.
"""

from __future__ import annotations

import warnings

import numpy as np

from .base import Projector
from .pod_utils import energy_fraction, snapshot_pod, snapshot_pod_randomized, spod_sieber

__all__ = ['POD', 'SPOD']


class POD(Projector):
    r"""
    Snapshot POD.

    Inherits the shared interface from `Projector` (`fit`/`encode`/`decode`/
    `reconstruct`/`score`/`N_latent`) and adds linear-specific attributes,
    geometry helpers, and plotting utilities.

    Two solvers are available via ``method``:

    - ``'exact'`` — full eigendecomposition of the temporal correlation matrix
      $\matr{C} = \matr{Q}^\mathrm{T}\matr{Q} / N_t$ (Sirovich 1987, snapshot
      method: see `snapshot_pod`). Exact but $\mathcal{O}(N_t^3)$.
    - ``'randomized'`` (default) — randomized SVD of $\matr{Q}$ (Halko, Martinsson
      & Tropp 2011: see `snapshot_pod_randomized`). Returns only the leading
      ``n_modes`` modes; fast and memory-efficient.

    Either way, writing $\matr{Q} = \matr{X} - \bar{\matr{Q}}$ for the zero-mean
    data matrix, ``fit(X)`` stores the orthonormal spatial modes $\matr{\Psi}$
    and singular values $\matr{\Sigma}$ of $\matr{Q}$, together with the
    temporal coefficients $\matr{\Phi} = \matr{\Psi}^\mathrm{T}\matr{Q}$,
    so that

    $$
    \matr{X} \approx \matr{\Psi}\matr{\Phi} + \bar{\matr{Q}},
    $$

    with equality when all $N_t$ modes are retained (``n_modes >= N_t``, ``method='exact'``).

    Parameters
    ----------
    n_modes : int, optional
        Number of modes retained (the latent-space size). ``None`` uses the
        rows of a pre-set ``Phi``, or 20.
    method : str
        ``'exact'`` or ``'randomized'``. Default ``'randomized'``.
    n_iter : int
        Power-iteration steps for the randomized solver. Default 4.
    random_state : int, optional
        Seed for reproducibility of the randomized solver.
    grid_shape : tuple, optional
        Grid shape ``(Nu, Nx, Ny)`` used to map flat vectors back to the grid.
    domain : list, optional
        Physical domain ``[x0, x1, y0, y1]`` used by the plotting utilities.
    **kwargs
        Pre-set any instance attribute (e.g. pre-computed ``Sigma``, ``Psi``,
        ``Phi``, ``Q_mean``), or pass ``X=...`` to fit directly at construction.

    Raises
    ------
    TypeError
        If a keyword is not an attribute of the class.

    Attributes
    ----------
    Sigma : np.ndarray
        Singular values (descending), shape $(N_\mathrm{latent},)$.
    Psi : np.ndarray
        Spatial modes with orthonormal columns, shape $(N_x, N_\mathrm{latent})$.
    Phi : np.ndarray
        Temporal coefficients, shape $(N_\mathrm{latent}, N_t)$.
    Q_mean : np.ndarray
        Temporal mean $\bar{\matr{Q}}$, shape $(N_x, 1)$.

    References
    ----------
    Sirovich (1987). Turbulence and the dynamics of coherent structures.
    Part I: Coherent structures. *Quart. Appl. Math.*, 45(3), 561-571.
    doi:10.1090/qam/910462.

    Halko, Martinsson & Tropp (2011). Finding structure with randomness.
    *SIAM Review*, 53(2), 217-288.
    """

    # ── class-level defaults ─────────────────────────────────────────────────
    _Sigma:       np.ndarray | None = None
    _Psi:         np.ndarray | None = None
    _Phi:         np.ndarray | None = None
    grid_shape:   tuple | None      = None   # (Nu, Nx, Ny) for grid <-> flat mapping
    domain:       list | None       = None   # [x0, x1, y0, y1]
    field_labels: list                 = ['$u_x$', '$u_y$']
    _TKE:         float | None      = None
    indices_to_original_grid: np.ndarray | None = None
    method:       str           = 'randomized'
    n_iter:       int           = 4
    random_state: int | None = None

    @property
    def Sigma(self) -> np.ndarray:
        if self._Sigma is None:
            raise AttributeError("Not fitted — call fit() first.")
        return self._Sigma

    @Sigma.setter
    def Sigma(self, v: np.ndarray) -> None:
        self._Sigma = v

    @property
    def Psi(self) -> np.ndarray:
        if self._Psi is None:
            raise AttributeError("Not fitted — call fit() first.")
        return self._Psi

    @Psi.setter
    def Psi(self, v: np.ndarray) -> None:
        self._Psi = v

    @property
    def Phi(self) -> np.ndarray:
        if self._Phi is None:
            raise AttributeError("Not fitted — call fit() first.")
        return self._Phi

    @Phi.setter
    def Phi(self, v: np.ndarray) -> None:
        self._Phi = v


    def __init__(self,
                 n_modes:      int | None     = None,
                 method:       str            = 'randomized',
                 n_iter:       int            = 4,
                 random_state: int | None  = None,
                 grid_shape:   tuple | None = None,
                 domain:       list | None  = None,
                 **kwargs):
        self.N_latent     = 20 if n_modes is None else n_modes
        self.grid_shape   = grid_shape
        self.domain       = domain
        self.method       = method
        self.n_iter       = n_iter
        self.random_state = random_state
        for key, val in kwargs.items():
            if key == 'X':
                continue
            if not (hasattr(type(self), key) or key in ('Sigma', 'Psi', 'Phi', 'Q_mean', '_TKE',
                                                         'indices_to_original_grid')):
                raise TypeError(f"{type(self).__name__} got an unexpected keyword {key!r}")
            setattr(self, key, val)
        # infer latent size from pre-loaded Phi if provided
        if self._Phi is not None and n_modes is None:
            self.N_latent = self._Phi.shape[0]
        # backward compat: auto-fit if raw data provided as kwarg 'X'
        if 'X' in kwargs and kwargs['X'] is not None:
            self.fit(kwargs['X'])

    # ── N_modes backward-compat alias ─────────────────────────────────────────

    @property
    def N_modes(self) -> int:
        """Backward-compatible alias for N_latent."""
        return self.N_latent

    @N_modes.setter
    def N_modes(self, value: int) -> None:
        self.N_latent = value

    # ── decomposition hook (overridden by SPOD) ────────────────────────────────

    def _decompose(self, Q: np.ndarray) -> tuple:
        """
        Run the SVD/eigendecomposition on the zero-mean data matrix Q,
        dispatching to `snapshot_pod` or `snapshot_pod_randomized`
        depending on `method`. Overridden by `SPOD` to filter the
        correlation matrix first.

        Returns
        -------
        tuple of np.ndarray
            ``(Sigma, Psi, Phi)`` with shapes ``(N_latent,)``,
            ``(N_x, N_latent)`` and ``(N_latent, N_t)`` respectively.
        """
        if self.method == 'exact':
            Sigma, Psi, Phi, _ = snapshot_pod(Q)
            return Sigma[:self.N_latent], Psi[:, :self.N_latent], Phi[:self.N_latent]
        elif self.method == 'randomized':
            return snapshot_pod_randomized(Q, n_modes=self.N_latent,
                                           n_iter=self.n_iter,
                                           random_state=self.random_state)
        else:
            raise ValueError(f"Unknown method '{self.method}'. "
                             "Choose 'exact' or 'randomized'.")

    # ── Projector interface ────────────────────────────────────────────────────


    def fit(self, X: np.ndarray) -> POD:
        r"""
        Fit the POD to data X.

        Accepts a flat matrix $(N_x, N_t)$ or a raw grid array
        $(N_u, N_t, N_x, N_y)$ / $(N_t, N_x, N_y)$. For raw grid input the
        NaN solid-body mask is detected and stored automatically.

        Parameters
        ----------
        X : np.ndarray
            Snapshot data, flat or raw grid (see above).

        Returns
        -------
        POD
            The fitted instance (``self``).
        """

        Q = self.preprocess_snapshot(X)

        result = self._decompose(Q)
        self.Sigma = result[0]
        self.Psi   = result[1]
        self.Phi   = result[2]
        assert self._Sigma is not None and self._Psi is not None and self._Phi is not None, \
            "Decomposition must return Sigma, Psi, Phi."
        self.N_latent = self.Sigma.shape[0]
        self.fitted = True
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        r"""
        Project X onto the spatial modes,

        $$
        \matr{Z} = \matr{\Psi}^\mathrm{T}(\matr{X} - \bar{\matr{Q}}).
        $$

        Parameters
        ----------
        X : np.ndarray
            Snapshot data, shape $(N_x, N_t)$.

        Returns
        -------
        np.ndarray
            Latent (POD) coefficients $\matr{Z}$, shape $(N_\mathrm{latent}, N_t)$.
        """
        Q = self.preprocess_snapshot(X)
        return self.Psi.T @ Q

    def decode(self, Z: np.ndarray, idx: np.ndarray | None = None) -> np.ndarray:
        r"""
        Reconstruct the state in the original space from latent coefficients,

        $$
        \hat{\matr{Q}} = \matr{\Psi}\matr{Z} + \bar{\matr{Q}}.
        $$

        Parameters
        ----------
        Z : np.ndarray
            Latent coefficients, shape $(N_\mathrm{latent}, N_t)$.
        idx : np.ndarray, optional
            Indices selecting a subset of rows of `Psi` and `Q_mean`
            (e.g. sensor locations) to reconstruct only those entries.

        Returns
        -------
        np.ndarray
            Reconstructed state, shape $(N_x, N_t)$, or ``(len(idx), N_t)``
            if `idx` is given.
        """

        if idx is not None:
            return self.Psi[idx, :] @ Z + self.Q_mean[idx, :]
        else:
            return self.Psi @ Z + self.Q_mean

    def decode_at(self, Z: np.ndarray, idx: np.ndarray | None = None) -> np.ndarray:
        """Sensor readout by the linear route, ``Psi[idx] @ Z + Q_mean[idx]``,
        without forming the full field (overrides `Projector.decode_at`)."""
        return self.decode(Z, idx=idx)

    def spatial_basis(self, z0: np.ndarray | None = None) -> np.ndarray:
        r"""The spatial modes $\boldsymbol{\Psi}$, shape $(N_x, N_\mathrm{latent})$.

        A linear decoder is its own Jacobian, so the local basis that
        `Projector.spatial_basis` approximates by finite differences is exact
        here and independent of `z0` (accepted for interface parity). Sensor
        placement by QR pivoting on this basis is the POD-ESN placement.
        """
        return self.Psi

    @property
    def n_params(self) -> int:
        """int: Number of stored basis entries (the size of `Psi`)."""
        return int(self.Psi.size)


    def reconstruct(self, X: np.ndarray | None = None,
                    n_modes: int | None = None,
                    Phi: np.ndarray | None = None) -> np.ndarray:
        r"""
        Full round-trip: encode, then decode.

        Parameters
        ----------
        X : np.ndarray, optional
            Raw input, grid or flat. If None, the stored `Phi` is used.
        n_modes : int, optional
            Retain only the first ``n_modes`` modes. Default: all fitted modes.
        Phi : np.ndarray, optional
            Pre-computed latent coefficients, shape $(N_\mathrm{latent}, N_t)$;
            skips the `encode` step.

        Returns
        -------
        np.ndarray
            Reconstructed flat state, shape $(N_x, N_t)$.
        """
        if Phi is not None:
            Z = Phi
        elif X is not None:
            Z = self.encode(X)
        else:
            Z = self.Phi

        nm = n_modes if n_modes is not None else self.N_latent
        if nm < self.N_latent:
            X_hat = self.Psi[:, :nm] @ Z[:nm] + self.Q_mean
        else:
            X_hat = self.decode(Z)
        return X_hat


    # ── utilities ─────────────────────────────────────────────────────────────

    def energy_fraction(self):
        r"""
        Relative and cumulative energy per mode, from the eigenvalues
        $\lambda_j = \Sigma_j^2$ of the temporal correlation matrix
        $\matr{C}$:

        $$
        \mathrm{rel}_j = \frac{\lambda_j}{\sum_k \lambda_k}, \qquad
        \mathrm{cum}_j = \sum_{k \le j} \mathrm{rel}_k.
        $$

        Returns
        -------
        rel : np.ndarray
            Relative energy per mode, shape $(N_\mathrm{latent},)$.
        cum : np.ndarray
            Cumulative energy fraction captured by the first $j$ modes,
            shape $(N_\mathrm{latent},)$.
        """
        return energy_fraction(self.Sigma)


    def truncate(self, n_modes: int) -> POD:
        """Truncate to the first n_modes modes in-place."""
        if n_modes >= self.N_latent:
            warnings.warn(f'Requested n_modes={n_modes} >= N_latent={self.N_latent}: no truncation applied.',
                          stacklevel=2)
            return self
        self.Psi      = self.Psi[:, :n_modes]
        self.Phi      = self.Phi[:n_modes, :]
        self.Sigma    = self.Sigma[:n_modes]
        self.N_latent = n_modes
        return self


    @property
    def domain_mesh(self):
        """
        Meshgrid for the spatial domain.

        Returns
        -------
        X1 : np.ndarray
            First coordinate array, shape ``(Nx, Ny)``.
        X2 : np.ndarray
            Second coordinate array, shape ``(Nx, Ny)``.
        """
        if self.domain is None or self.grid_shape is None:
            raise ValueError('Both domain and grid_shape must be set.')
        x1 = np.linspace(*self.domain[:2], num=self.grid_shape[-2])
        x2 = np.linspace(*self.domain[2:], num=self.grid_shape[-1])
        return np.meshgrid(x1, x2, indexing='ij')

    # ── metrics ───────────────────────────────────────────────────────────────

    @staticmethod
    def compute_MSE(ROM_data: np.ndarray, original_data: np.ndarray,
                    time_evolution: bool = False):
        """Mean Squared Error between ROM reconstruction and original data."""
        ROM_data, original_data = POD.flatten(ROM_data, original_data)
        original_data = np.where(np.isnan(original_data), 0., original_data)
        if time_evolution:
            return np.mean((original_data - ROM_data) ** 2, axis=0)
        return float(np.mean((original_data - ROM_data) ** 2))

    @staticmethod
    def compute_RMS(ROM_data: np.ndarray, original_data: np.ndarray):
        """Pointwise absolute error field, with NaN in `original_data` read as 0."""
        original_data = np.where(np.isnan(original_data), 0., original_data)
        return np.abs(original_data - ROM_data)

    @staticmethod
    def flatten(*args):
        """Flatten multi-dimensional arrays to 2-D (space × time)."""
        return [a.reshape(-1, a.shape[-1]) if a.ndim > 2 else a.copy()
                for a in args]



# ─────────────────────────────────────────────────────────────────────────────
# SPOD (Sieber)  — inherits from POD, only overrides _decompose
# ─────────────────────────────────────────────────────────────────────────────

class SPOD(POD):
    r"""
    Spectral POD via the filtered correlation matrix (Sieber, Paschereit &
    Oberleithner, 2016).

    Inherits the `POD` interface; `_decompose` delegates to `spod_sieber` and
    `encode` projects by least squares. The snapshot correlation matrix
    $\matr{C} = \matr{Q}^\mathrm{T}\matr{Q}/N_t$ is filtered along its
    diagonals (periodic continuation) before the eigensolve,

    $$
    S_{ij} = \sum_{k=-N_f}^{N_f} g_k\, C_{i+k,\,j+k},
    $$

    with $g_k$ a normalised 1-D kernel of half-width ``Nf`` (``filter_kind``
    selects ``'gaussian'``, ``'box'`` or ``'hann'``). The temporal coefficients
    $\matr{\Phi}$ are orthogonal, but for ``Nf > 0`` the spatial modes
    $\matr{\Psi}$ are not orthonormal, so `encode` solves
    $\min_{\matr{Z}} \|\matr{\Psi}\matr{Z} - (\matr{X} - \bar{\matr{Q}})\|$
    instead of applying $\matr{\Psi}^\mathrm{T}$. For a truncated fit, ``Phi``
    holds Sieber's coefficients of the training snapshots, which differ from
    ``encode(X_train)`` because the modes are not orthogonal. Setting ``Nf=0``
    skips the filtering step and exactly recovers snapshot POD; as ``Nf`` grows
    towards $N_t/2$ the SPOD modes approach Fourier (DFT) modes.

    Parameters
    ----------
    Nf : int
        Filter half-width (0 recovers POD; $N_t/2$ approaches the DFT limit).
    filter_kind : str
        ``'gaussian'``, ``'box'`` or ``'hann'``. Default ``'gaussian'``.
    n_modes : int
        Number of modes to retain. Default 20.
    grid_shape : tuple, optional
        Grid shape ``(Nu, Nx, Ny)`` for grid mapping.
    domain : list, optional
        Physical domain ``[x0, x1, y0, y1]`` for the plotting utilities.
    **kwargs
        Forwarded to `POD.__init__` (e.g. pre-set ``Sigma``, ``Psi``, ``Phi``).

    Attributes
    ----------
    C_tilde : np.ndarray
        Filtered correlation matrix $\matr{S}$, shape $(N_t, N_t)$
        (stored after ``fit``).

    References
    ----------
    Sieber, Paschereit & Oberleithner (2016). Spectral proper orthogonal
    decomposition. *J. Fluid Mech.*, 792, 798–828.
    """

    def __init__(self,
                 Nf:          int            = 0,
                 filter_kind: str            = 'gaussian',
                 n_modes:     int            = 20,
                 grid_shape:  tuple | None = None,
                 domain:      list | None  = None,
                 **kwargs):
        super().__init__(n_modes=n_modes,
                         grid_shape=grid_shape,
                         domain=domain,
                         **kwargs)
        self.Nf                 = Nf
        self.filter_kind        = filter_kind
        self._n_modes_requested = n_modes   # None = keep all after fit

    def _decompose(self, Q: np.ndarray) -> tuple:
        """Filtered-correlation-matrix eigendecomposition, via `spod_sieber`."""
        Sigma, Psi, Phi, C_tilde = spod_sieber(Q, self.Nf, self.filter_kind)
        self.C_tilde = C_tilde
        if self._n_modes_requested is not None:
            nm = min(self._n_modes_requested, Sigma.shape[0])
            return Sigma[:nm], Psi[:, :nm], Phi[:nm]
        self.N_latent = Sigma.shape[0]
        return Sigma, Psi, Phi

    def encode(self, X: np.ndarray) -> np.ndarray:
        r"""Least-squares projection onto the (non-orthonormal) spatial modes,
        $\matr{Z} = \matr{\Psi}^{+}(\matr{X} - \bar{\matr{Q}})$.

        Parameters
        ----------
        X : np.ndarray
            Snapshot data, shape $(N_x, N_t)$.

        Returns
        -------
        np.ndarray
            SPOD coefficients $\matr{Z}$, shape $(N_\mathrm{latent}, N_t)$.
        """
        return np.linalg.lstsq(self.Psi, self.preprocess_snapshot(X), rcond=None)[0]
