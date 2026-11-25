"""Base class for latent reduced-order models.

A latent ROM is a projector, which reduces flow snapshots to latent
coefficients, combined with a forecaster, which advances those coefficients in
time. Each ROM class inherits from `LatentROM`, a forecaster, and a projector,
in that order:

  class POD_ESN(LatentROM, ESN_model, POD): ...

Every ROM built this way is a `dynamodels.Model` whose state is the latent
coefficients followed by the forecaster state, and whose observables are the
decoded field at the sensors (or the latent coefficients), so it plugs into
romda's estimators and `romda.data_assimilation.run_da_loop` as `POD_ESN` does.

`LatentROM` fits the projector, trains the forecaster on the latent training
trajectory, places sensors, and reads sensor observables off the forecast.

Sensor locations are raw grid indices `u * Nx * Ny + g`, where `u` is the field
component and `g` the flattened `(x, y)` position, so every sensor records every
component. `sensor_rows` maps them to rows of the flat field.

Typical usage example:

  rom = POD_ESN(data=X, dt=0.01, N_modes=8, Nq=10, N_units=100)
  psi, t = rom.time_integrate(Nt=200)
  rom.update_history(psi, t)
  observables = rom.get_observable_hist()

Ported from autoencoder-rom; co-author: Lukas Campbell (2026).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
import scipy.linalg as sla

from .autoencoders.base import FloatArray, Projector
from .rnns import phi_to_esn_layout
from .training import TrainingHistory

__all__ = ["LatentROM", "measurement_grid", "qr_sensor_points", "sensor_rows"]

# Written by Lukas Campbell (autoencoder-rom, 2026) as the generalization of
# romda's POD_ESN to any projector and forecaster.

#: Integer indices into a grid or a flat field.
IndexArray = npt.NDArray[np.intp]

#: Names of the field components, used in observable labels.
_COMPONENTS = ("u_x", "u_y", "u_z")


# ── Sensor placement ──────────────────────────────────────────────────────────


def measurement_grid(
    grid_shape: tuple[int, int, int],
    fluid_mask: npt.NDArray[np.bool_],
    domain: Sequence[float] | None = None,
    region: Sequence[float] | None = None,
    stride: tuple[int, int] | None = None,
) -> IndexArray:
    """Lists the grid points sensors may occupy.

    Args:
      grid_shape: Grid shape `(Nu, Nx, Ny)`.
      fluid_mask: Mask over the `Nx * Ny` grid points, `True` at fluid points.
      domain: Physical extent `[x0, x1, y0, y1]` of the grid.
      region: Extent `[x0, x1, y0, y1]` to restrict sensors to. `None` allows
        the whole grid.
      stride: Keep every `stride[0]`-th x index and `stride[1]`-th y index of
        the region. `None` keeps all of them.

    Returns:
      Raw grid indices of the fluid points in the region, for every component,
      grouped by component.

    Raises:
      ValueError: If `region` is given without `domain`, or covers no grid
        points.
    """
    Nu, Nx, Ny = grid_shape
    if region is None or (domain is not None and list(region) == list(domain)):
        x_idx: IndexArray = np.arange(Nx)
        y_idx: IndexArray = np.arange(Ny)
    else:
        if domain is None:
            raise ValueError("domain_of_measurement needs the grid's domain to be set")
        x0, x1, y0, y1 = domain
        rx0, rx1, ry0, ry1 = region
        x = np.linspace(x0, x1, Nx)
        y = np.linspace(y0, y1, Ny)
        x_idx = np.flatnonzero((x >= rx0) & (x <= rx1))
        y_idx = np.flatnonzero((y >= ry0) & (y <= ry1))

    if x_idx.size == 0 or y_idx.size == 0:
        raise ValueError(f"domain_of_measurement {region} covers no grid points of domain {domain}")
    if stride is not None:
        x_idx, y_idx = x_idx[:: stride[0]], y_idx[:: stride[1]]

    points = np.ravel_multi_index(np.ix_(x_idx, y_idx), dims=(Nx, Ny))
    fluid_points = np.intersect1d(np.flatnonzero(fluid_mask), points)
    return np.concatenate([fluid_points + Nx * Ny * u for u in range(Nu)])


def qr_sensor_points(
    basis_at_candidates: FloatArray, candidates: IndexArray, n_sensors: int, n_points: int
) -> IndexArray:
    """Picks sensor points by QR column pivoting on a spatial basis.

    Args:
      basis_at_candidates: The basis at the candidate locations, shape
        `(n_candidates, r)`.
      candidates: Raw grid indices of the candidate locations.
      n_sensors: Number of sensor points to pick.
      n_points: Grid points per component, `Nx * Ny`.

    Returns:
      Distinct grid point indices `g`, in pivot order (most informative first).
      Holds `n_sensors` entries unless the candidates cover fewer distinct
      points.
    """
    A = basis_at_candidates
    if n_sensors > A.shape[1]:
        A = A @ A.T
    _, pivots = sla.qr(A.T, mode="r", pivoting=True)
    pts = candidates[pivots] % n_points
    _, first = np.unique(pts, return_index=True)
    points: IndexArray = pts[np.sort(first)][:n_sensors]
    if points.size < n_sensors:
        warnings.warn(f"QR placement found {points.size} distinct sensor points, fewer than the {n_sensors} requested")
    return points


def sensor_rows(
    locations: IndexArray,
    grid_shape: tuple[int, int, int],
    fluid_mask: npt.NDArray[np.bool_],
) -> IndexArray:
    """Maps raw grid indices to rows of the flat field.

    The flat field groups rows by component, so the row of component `u` at the
    `f`-th fluid point is `u * N_fluid + f`.

    Args:
      locations: Raw grid indices `u * Nx * Ny + g`.
      grid_shape: Grid shape `(Nu, Nx, Ny)`.
      fluid_mask: Mask over the `Nx * Ny` grid points, `True` at fluid points.

    Returns:
      The flat field row of each location.

    Raises:
      ValueError: If a location is outside the grid or at a solid point.
    """
    Nu, Nx, Ny = grid_shape
    if locations.size and (locations.min() < 0 or locations.max() >= Nu * Nx * Ny):
        raise ValueError(f"sensor locations must lie in [0, {Nu * Nx * Ny}), got {locations.min()}..{locations.max()}")
    component, point = np.divmod(locations, Nx * Ny)
    if not fluid_mask[point].all():
        raise ValueError("some sensor locations are solid points")
    fluid_points = np.flatnonzero(fluid_mask)
    return component * fluid_points.size + np.searchsorted(fluid_points, point)


# ── Base class ────────────────────────────────────────────────────────────────


class LatentROM(Projector):
    """A projector and a forecaster combined into a reduced-order model.

    Subclasses list `LatentROM` first, then a forecaster (a `dynamodels.Model`
    such as `ESN_model`), then a projector; `__init_subclass__` sets
    `_forecaster_cls`, `_projector_cls` and `extra_print_params` from them.

    Attributes:
      Nq: Number of observables. Before construction, the number of sensor
        points to place; afterwards, one observable per sensor and component.
      measure_modes: Observe the latent coefficients instead of sensors.
      qr_selection: Place sensors by QR pivoting rather than at random.
      perform_test: Hold out a test block when training.
      latent_symbol: Symbol for the latent coefficients in labels.
      projector_history: Training history of the projector, or `None` for a
        projector without one.
      forecaster_history: Training history of the forecaster, or `None` for a
        forecaster without one.
    """

    Nq: int = 10
    measure_modes: bool = False
    qr_selection: bool = True
    perform_test: bool = False
    latent_symbol: str = "z"
    projector_history: TrainingHistory | None = None
    forecaster_history: TrainingHistory | None = None
    configurable: ClassVar[bool] = False

    _projector_cls: ClassVar[type[Any]]
    _forecaster_cls: ClassVar[type[Any]]
    _size_param: ClassVar[str] = "N_latent"
    _options: ClassVar[tuple[str, ...]] = (
        "Nq",
        "measure_modes",
        "sensor_locations",
        "qr_selection",
        "perform_test",
        "latent_symbol",
    )

    _sensor_locations: IndexArray | None = None
    _sensor_rows: IndexArray | None = None
    _domain_of_measurement: list[float] | None = None
    _down_sample_measurement: tuple[int, int] | None = None
    _latent_training_trajectory: FloatArray | None = None
    _z_mean: FloatArray | None = None
    _constructing: type[Any] | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Takes the forecaster and projector classes from the bases `(LatentROM, F, P)`."""
        super().__init_subclass__(**kwargs)
        if cls.__bases__[0] is LatentROM and len(cls.__bases__) == 3:
            cls._forecaster_cls, cls._projector_cls = cls.__bases__[1:3]
            if "extra_print_params" not in cls.__dict__:
                cls.extra_print_params = [*cls._forecaster_cls.extra_print_params, "Nq", "measure_modes", cls._size_param]

    def __init__(
        self,
        data: npt.NDArray[Any],
        dt: float,
        *,
        train_forecaster: bool = True,
        train_ESN: bool | None = None,
        skip_sensor_placement: bool = False,
        domain_of_measurement: Sequence[float] | None = None,
        down_sample_measurement: int | Sequence[int] | None = None,
        projector_epochs: int | None = None,
        forecaster_epochs: int | None = None,
        projector: Any = None,
        **kwargs: Any,
    ) -> None:
        """Fits the projector, trains the forecaster, and places sensors.

        Args:
          data: Snapshots on the grid, shape `(Nu, N_t, Nx, Ny)`, with NaN at
            solid points; or a list of such records of the same shape (e.g. a
            trajectory and its mirror image), which the projector fits together
            and the forecaster takes as separate segments, so that no step is
            learnt across the joins.
          dt: Time step between snapshots.
          train_forecaster: Train the forecaster. `False` fits only the
            projector and places sensors.
          train_ESN: Alias for `train_forecaster`. Takes precedence when given.
          skip_sensor_placement: Observe the latent coefficients, so `Nq`
            becomes `N_latent`; the same as `measure_modes=True`.
          domain_of_measurement: Extent `[x0, x1, y0, y1]` to restrict sensors
            to. `None` allows the whole grid.
          down_sample_measurement: Stride over the measurement grid, one value
            for both axes or one per axis.
          projector_epochs: Training epochs of the projector. `None` keeps the
            projector's default.
          forecaster_epochs: Training epochs of the forecaster. `None` keeps
            the forecaster's default.
          projector: A fitted projector of `_projector_cls` to adopt instead of
            fitting one on `data`, such as a trained autoencoder restored from
            its weights. `data` then only trains the forecaster.
          **kwargs: Options of this class (see Attributes), of the projector,
            and of the forecaster. Each goes to the class that defines it; the
            rest go to the forecaster.

        Raises:
          TypeError: If `kwargs` holds `epochs`, an epoch count is given for
            a projector or forecaster that does not train in epochs, or
            `projector_epochs` is given with a fitted `projector`.
          ValueError: If a latent size is given that differs from the fitted
            `projector`'s.
        """
        if train_ESN is not None:
            train_forecaster = train_ESN
        for key in self._options:
            if key in kwargs:
                setattr(self, key, kwargs.pop(key))
        if "epochs" in kwargs:
            raise TypeError(f"{type(self).__name__} takes projector_epochs and forecaster_epochs, not epochs")
        # with a fitted projector, its options are fixed: the rest (e.g. `seed`) go to the forecaster
        projector_kwargs = {
            k: kwargs.pop(k)
            for k in list(kwargs)
            if k in ("n_latent", "n_modes") or (projector is None and hasattr(self._projector_cls, k))
        }
        for count, cls, role in (
            (projector_epochs, self._projector_cls, "projector"),
            (forecaster_epochs, self._forecaster_cls, "forecaster"),
        ):
            if count is not None and not hasattr(cls, "epochs"):
                raise TypeError(f"{cls.__name__} does not train in epochs; drop {role}_epochs")
        if projector_epochs is not None:
            projector_kwargs["epochs"] = projector_epochs
        if forecaster_epochs is not None:
            kwargs["epochs"] = forecaster_epochs

        segments = list(data) if isinstance(data, (list, tuple)) else None
        if segments is not None:
            if len({np.shape(s) for s in segments}) != 1:
                raise ValueError("the records in data must have the same shape (Nu, N_t, Nx, Ny)")
            data = np.concatenate(segments, axis=1)

        projector_cls: Any = self._projector_cls
        self._constructing = projector_cls
        if projector is None:
            projector_cls.__init__(self, **projector_kwargs)
            self.fit(data)
        else:
            if not isinstance(projector, projector_cls):
                raise TypeError(f"projector must be a fitted {projector_cls.__name__}, got {type(projector).__name__}")
            size = projector_kwargs.get("n_latent", projector_kwargs.get("n_modes"))
            if size is not None and size != projector.N_latent:
                raise ValueError(f"latent size {size} differs from the fitted projector's N_latent={projector.N_latent}")
            if projector_epochs is not None:
                raise TypeError("projector_epochs has no effect on a fitted projector; drop it")
            self.__dict__.update(vars(projector))
        self.projector_history = getattr(self, "training_history", None)
        Z = self._training_latents(data)
        self._latent_training_trajectory = Z
        self._z_mean = Z.mean(axis=1)

        if train_forecaster:
            forecaster: Any = self._forecaster_cls
            self._constructing = forecaster
            Z_fit = Z if segments is None else np.stack([self._training_latents(s) for s in segments])
            forecaster.__init__(self, data=phi_to_esn_layout(Z_fit), dt=dt, **kwargs)
            if self.projector_history is not getattr(self, "training_history", None):
                self.forecaster_history = getattr(self, "training_history", None)
        self._constructing = None

        locations = self.sensor_locations
        if self.measure_modes or skip_sensor_placement:
            self.measure_modes = True
            self.Nq = self.N_latent
            return
        if locations is None:
            self.domain_of_measurement = domain_of_measurement
            self.down_sample_measurement = down_sample_measurement
            locations = self.define_sensors(N_sensors=self.Nq, z0=self._z_mean)
            self.sensor_locations = locations
        self.Nq = locations.size

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore a pickled ROM, including those pickled before `LatentROM` existed.

        A `POD_ESN` pickled by romda <= 3.1 stores `sensor_locations`,
        `domain_of_measurement` and `down_sample_measurement` as plain instance
        attributes, which the properties of this class now shadow; they are
        moved to the private names the properties read.
        """
        if "sensor_locations" in state:
            value = state.pop("sensor_locations")
            state["_sensor_locations"] = None if value is None else np.asarray(value, dtype=np.intp).ravel()
            state["_sensor_rows"] = None
        if "domain_of_measurement" in state:
            value = state.pop("domain_of_measurement")
            state["_domain_of_measurement"] = None if value is None else list(value)
        if "down_sample_measurement" in state:
            state["_down_sample_measurement"] = state.pop("down_sample_measurement")
        dsm = state.get("_down_sample_measurement")
        if isinstance(dsm, int | np.integer):
            state["_down_sample_measurement"] = (int(dsm), int(dsm))
        elif dsm is not None:
            state["_down_sample_measurement"] = tuple(int(step) for step in dsm)
        restore = getattr(super(), "__setstate__", None)    # the forecaster's own (`ESN_model.__setstate__`)
        if restore is None:
            self.__dict__.update(state)
        else:
            restore(state)

    def __post_init__(self) -> None:
        """Runs the `__post_init__` of the base class being constructed."""
        # A dataclass `__init__` calls `self.__post_init__`, which would resolve
        # to the forecaster's when the projector is a dataclass too.
        post_init = getattr(self._constructing, "__post_init__", None)
        if post_init is not None:
            post_init(self)

    # ── Latent state ──────────────────────────────────────────────────────────

    @property
    def latent_training_trajectory(self) -> FloatArray:
        """The latent coefficients the forecaster trains on, shape `(N_latent, N_t)`.

        Raises:
          RuntimeError: If construction has not fitted the projector.
        """
        if self._latent_training_trajectory is None:
            raise RuntimeError(f"{type(self).__name__} has no training trajectory; construct it with data")
        return self._latent_training_trajectory

    def _training_latents(self, data: npt.NDArray[Any]) -> FloatArray:
        """Computes the latent coefficients of the training snapshots.

        Args:
          data: The training snapshots.

        Returns:
          Latent coefficients, shape `(N_latent, N_t)`.
        """
        Z: FloatArray = self.encode(data)
        return Z

    def spatial_basis(self, z0: npt.NDArray[Any] | None = None) -> FloatArray:
        """Computes the projector's spatial basis.

        Args:
          z0: Latent state to linearize about. `None` uses the mean of the
            training trajectory.

        Returns:
          The basis, shape `(N_x, N_latent)`.
        """
        return super().spatial_basis(self._z_mean if z0 is None else z0)

    def get_latent_coefficients(self, Nt: int = 1) -> FloatArray:
        """Reads the latest latent coefficients from the model history.

        Args:
          Nt: Number of most recent time steps.

        Returns:
          Coefficients, shape `(N_latent, m)` for `Nt=1`, otherwise
          `(Nt, N_latent, m)`.
        """
        if Nt == 1:
            return self.hist[-1, : self.N_latent]
        return self.hist[-Nt:, : self.N_latent]

    def get_observables(self, Nt: int = 1, Z: FloatArray | None = None, **kwargs: Any) -> FloatArray:
        """Computes the observables of the latest or given latent coefficients.

        Args:
          Nt: Number of most recent time steps, when `Z` is `None`.
          Z: Latent coefficients, shape `(N_latent, m)` or `(Nt, N_latent, m)`.
            `None` reads them from the model history.
          **kwargs: Ignored.

        Returns:
          The latent coefficients when `measure_modes` is set, otherwise the
          decoded field at the sensors, shape `(Nq, m)` or `(Nt, Nq, m)`.
        """
        del kwargs
        if self.measure_modes:
            return self.get_latent_coefficients(Nt=Nt) if Z is None else Z
        if Z is None:
            Z = self.get_latent_coefficients(Nt=Nt)
        if Z.ndim == 2:
            return self.decode_at(Z, idx=self.sensor_rows)

        n_t, n_latent, m = Z.shape
        flat = Z.transpose(1, 0, 2).reshape(n_latent, n_t * m)
        observables = self.decode_at(flat, idx=self.sensor_rows)
        return observables.reshape(-1, n_t, m).transpose(1, 0, 2)

    @property
    def obs_labels(self) -> list[str]:
        """Labels of the observables."""
        if self.measure_modes:
            return [f"${self.latent_symbol}_{j + 1}$" for j in range(self.N_latent)]
        Nu = self._grid()[0]
        names = _COMPONENTS[:Nu] if Nu <= len(_COMPONENTS) else tuple(f"u_{u}" for u in range(Nu))
        return [f"${{{name}}}_{j}$" for name in names for j in range(self.N_sensors)]

    @property
    def state_labels(self) -> list[str]:
        """Labels of the state vector: the latent coefficients, then the forecaster's state."""
        latent = [f"${self.latent_symbol}_{j + 1}$" for j in range(self.N_latent)] if self.update_state else []
        return latent + self.forecaster_state_labels

    # ── Sensors ───────────────────────────────────────────────────────────────

    @property
    def N_sensors(self) -> int:
        """Number of sensor points, or 0 when `measure_modes` is set."""
        if self.measure_modes:
            return 0
        return self.Nq // self._grid()[0]

    @property
    def sensor_locations(self) -> IndexArray | None:
        """Raw grid indices of the sensors, grouped by component, or `None`."""
        return self._sensor_locations

    @sensor_locations.setter
    def sensor_locations(self, locations: npt.ArrayLike | None) -> None:
        self._sensor_locations = None if locations is None else np.asarray(locations, dtype=np.intp).ravel()
        self._sensor_rows = None

    @property
    def sensor_rows(self) -> IndexArray | None:
        """Rows of the flat field the sensors read, or `None` without sensors."""
        if self._sensor_locations is None:
            return None
        if self._sensor_rows is None:
            self._sensor_rows = sensor_rows(self._sensor_locations, self._grid(), self._fluid_mask())
        return self._sensor_rows

    @property
    def domain_of_measurement(self) -> list[float] | None:
        """Extent `[x0, x1, y0, y1]` sensors are restricted to, or `None` for the whole grid."""
        return self._domain_of_measurement

    @domain_of_measurement.setter
    def domain_of_measurement(self, region: Sequence[float] | None) -> None:
        if region is not None and len(region) != 4:
            raise ValueError(f"domain_of_measurement must be [x0, x1, y0, y1], got {region}")
        self._domain_of_measurement = None if region is None else list(region)

    @property
    def down_sample_measurement(self) -> tuple[int, int] | None:
        """Stride over the measurement grid along x and y, or `None`."""
        return self._down_sample_measurement

    @down_sample_measurement.setter
    def down_sample_measurement(self, stride: int | Sequence[int] | None) -> None:
        if stride is None:
            self._down_sample_measurement = None
            return
        steps = (stride, stride) if isinstance(stride, int) else tuple(stride)
        if len(steps) != 2 or min(steps) < 1:
            raise ValueError(f"down_sample_measurement must be a positive int or a pair of them, got {stride!r}")
        self._down_sample_measurement = (int(steps[0]), int(steps[1]))

    @property
    def grid_of_measurement(self) -> IndexArray:
        """Raw grid indices sensors may occupy, for every component."""
        return measurement_grid(
            self._grid(),
            self._fluid_mask(),
            self.domain,
            self.domain_of_measurement,
            self.down_sample_measurement,
        )

    def select_sensors(
        self,
        measure_modes: bool = False,
        domain_of_measurement: Sequence[float] | None = None,
        down_sample_measurement: int | Sequence[int] | None = None,
        N_sensors: int | None = None,
        qr_selection: bool | None = None,
    ) -> None:
        """Replaces the observables with the latent coefficients or new sensors.

        Args:
          measure_modes: Observe the latent coefficients.
          domain_of_measurement: Extent to restrict sensors to.
          down_sample_measurement: Stride over the measurement grid.
          N_sensors: Number of sensor points. `None` uses `N_sensors`.
          qr_selection: Place sensors by QR pivoting rather than at random.
            `None` keeps the current setting.
        """
        self.measure_modes = measure_modes
        if measure_modes:
            self.Nq = self.N_latent
            self.sensor_locations = None
            return
        self.domain_of_measurement = domain_of_measurement
        self.down_sample_measurement = down_sample_measurement
        if qr_selection is not None:
            self.qr_selection = qr_selection
        locations = self.define_sensors(N_sensors=N_sensors)
        self.sensor_locations = locations
        self.Nq = locations.size

    def define_sensors(self, N_sensors: int | None = None, z0: npt.NDArray[Any] | None = None) -> IndexArray:
        """Chooses sensor points in the measurement grid.

        With `qr_selection`, picks the points by QR column pivoting on the
        spatial basis at the candidates; otherwise picks them at random.

        Args:
          N_sensors: Number of sensor points. `None` uses `N_sensors`.
          z0: Latent state to linearize the basis about. `None` uses the mean
            of the training trajectory.

        Returns:
          Raw grid indices of the sensors, for every component, grouped by
          component.
        """
        Nu, Nx, Ny = self._grid()
        n_points = Nx * Ny
        n_sensors = self.N_sensors if N_sensors is None else N_sensors
        candidates = self.grid_of_measurement
        if n_sensors > candidates.size:
            warnings.warn(f"{n_sensors} sensors requested but the measurement grid holds {candidates.size} locations")

        if self.qr_selection:
            rows = sensor_rows(candidates, (Nu, Nx, Ny), self._fluid_mask())
            points = qr_sensor_points(self.spatial_basis(z0)[rows], candidates, n_sensors, n_points)
        else:
            first = candidates[candidates < n_points]
            if n_sensors < first.size:
                points = np.sort(self.rng.choice(first, size=n_sensors, replace=False))
            else:
                points = first.copy()
        return np.concatenate([points + n_points * u for u in range(Nu)])

    # ── Resets ────────────────────────────────────────────────────────────────

    def reset_case(
        self,
        reset_forecaster: bool = False,
        Z0: FloatArray | None = None,
        *,
        reset_ESN: bool | None = None,
        Phi0: FloatArray | None = None,
        **kwargs: Any,
    ) -> None:
        """Retrains the forecaster.

        Args:
          reset_forecaster: Reset the forecaster on the training trajectory.
          Z0: Initial latent state. `None` uses the first training snapshot.
          reset_ESN: Alias for `reset_forecaster`. Takes precedence when given.
          Phi0: Alias for `Z0` (the former `POD_ESN` name).
          **kwargs: Passed to `reset_forecaster`.
        """
        if reset_ESN is not None:
            reset_forecaster = reset_ESN
        if Z0 is None:
            Z0 = Phi0
        if reset_forecaster:
            Z = self.latent_training_trajectory
            self.reset_forecaster(phi_to_esn_layout(Z), u0=Z[:, 0] if Z0 is None else Z0, **kwargs)
