"""The latent ROMs (latent_rom.py, pod_esn.py and, privately, the autoencoder ROMs): sensor placement on the decoder
Jacobian, the observables, the segments of several records, the projector + forecaster wiring of tutorial 30, the
ntsa model protocol, and a latent ROM as the forecast model of romda's data-assimilation loop.

Parts ported from autoencoder-rom (Lukas Campbell, tests/test_ae_esn.py).
`test_pod_sensor_placement_unchanged` checks that placing sensors on
`spatial_basis` reproduces the original QR pivoting on `Psi` for POD.

Slow cases (anything that trains a model for longer) run only with `pytest -m slow`.
"""

import importlib.util

import numpy as np
import pytest
from conftest import DOMAIN, GRID, ks_like, synthetic_flow
from romda.data_assimilation import run_da_loop
from romda.estimators import EnsembleEstimator, EnSRKF
from romda.models.data_driven import POD_ESN, ESN_model, phi_to_esn_layout, synthetic_field
from romda.models.data_driven.autoencoders import POD
from romda.observations import Observations

slow = pytest.mark.slow
torch_ok = importlib.util.find_spec("torch") is not None
needs_torch = pytest.mark.skipif(not torch_ok, reason="torch is not installed")

DT = 0.05


# ── sensor placement and observables (no forecaster training)

def _untrained_pod_esn(data, **kw):
    return POD_ESN(data=data, dt=0.01, n_modes=6, method="exact", grid_shape=GRID, domain=DOMAIN,
                   Nq=8, seed=0, train_ESN=False, **kw)


def _reference_placement(case, N_sensors):
    """The original placement algorithm, pivoting explicitly on ``Psi`` on the grid."""
    import scipy.linalg as sla

    Nu, Nx, Ny = case.grid_shape
    measure_grid_idx = np.asarray(case.grid_of_measurement)

    Psi = case._to_physical_grid(case.Psi).transpose(1, 0, 2, 3)
    Psi = np.nan_to_num(Psi, nan=0.0)
    Psi = Psi.reshape(Psi.shape[0], Nu, Nx * Ny)
    A = Psi.reshape(Psi.shape[0], -1)[:, measure_grid_idx].T
    if N_sensors > A.shape[1]:
        A = np.dot(A, A.T)
    qr_idx = sla.qr(A.T, pivoting=True)[-1]
    sensor_idx = measure_grid_idx[qr_idx[:N_sensors]].ravel() % (Nx * Ny)
    return np.array(
        [sensor_idx + Nx * Ny * i for i in range(Nu)]
    ).reshape((-1,))


def test_pod_sensor_placement_unchanged(data):
    """Generalising placement must be a no-op for POD.

    `POD.spatial_basis` returns `Psi`, so the mixin pivots QR on exactly the
    matrix `POD_ESN` used to pivot on. Asserted against a reference
    implementation of the pre-lift algorithm rather than a recorded array, so
    the check survives changes to the test data.
    """
    case = POD_ESN(data=data, dt=0.01, n_modes=6, method="exact",
                   random_state=0, grid_shape=GRID, domain=DOMAIN, Nq=8, seed=0,
                   N_units=30, N_wash=5, N_func_evals=4, N_grid=2,
                   perform_test=False, train_ESN=False,
                   skip_sensor_placement=True)
    case.domain_of_measurement = None
    case.down_sample_measurement = None

    np.testing.assert_array_equal(
        case.define_sensors(N_sensors=8), _reference_placement(case, 8)
    )


def _check_protocol(m):
    """The attributes and methods of ntsa's model protocol (https://andreanovoa.github.io/ntsa/protocol/)."""
    for attr in ("Nphi", "Nq", "psi0", "alpha0", "params", "fixed_params",
                 "dt", "t_transient", "t_CR", "obs_labels", "alpha_labels", "name"):
        assert hasattr(m, attr), f"{m.name} lacks protocol attribute {attr!r}"
    for meth in ("time_integrate", "update_history", "get_observable_hist",
                 "close", "time_step"):
        assert callable(getattr(m, meth)), f"{m.name} lacks protocol method {meth!r}"
    assert len(m.obs_labels) == m.Nq
    assert np.asarray(m.psi0).ndim == 2  # (Nphi, m)


def test_sensor_selection_options(data):
    """Regression: skip_sensor_placement set Nq = N_latent but left measure_modes off, so the observables were the
    full field, and a given Z was ignored; select_sensors defaulted qr_selection=False and switched QR placement off."""
    m = _untrained_pod_esn(data, skip_sensor_placement=True)
    assert m.measure_modes and m.Nq == m.N_latent == len(m.obs_labels)
    Z = m.latent_training_trajectory[:, :3]
    assert m.get_observables(Z=Z) is Z
    m = _untrained_pod_esn(data)
    m.select_sensors(N_sensors=4)
    assert m.qr_selection and m.Nq == 4 * GRID[0]


def test_qr_sensor_points_are_distinct_in_pivot_order():
    """Regression: pivots on the u and v rows of one point gave duplicate sensors."""
    import scipy.linalg as sla
    from romda.models.data_driven import qr_sensor_points

    rng = np.random.default_rng(0)
    B = rng.normal(size=(3, 3))
    basis = np.vstack([B, B + 1e-6 * rng.normal(size=B.shape)])  # u and v rows nearly equal
    candidates = np.arange(6)
    pts = candidates[sla.qr(basis.T, mode="r", pivoting=True)[1]] % 3
    with pytest.warns(UserWarning, match="3 distinct"):
        out = qr_sensor_points(basis, candidates, 4, 3)
    np.testing.assert_array_equal(out, list(dict.fromkeys(pts)))


def test_legacy_pickle_state_is_migrated(data):
    """Regression: __setstate__ left legacy domain/stride attributes hidden behind the properties."""
    m = _untrained_pod_esn(data)
    state = dict(m.__dict__, sensor_locations=[1, 2], domain_of_measurement=[0.0, 1.0, 0.0, 1.0],
                 down_sample_measurement=2)
    m.__setstate__(state)
    assert m.domain_of_measurement == [0.0, 1.0, 0.0, 1.0] and m.down_sample_measurement == (2, 2)
    np.testing.assert_array_equal(m.sensor_locations, [1, 2])


def test_latent_rom_adopts_a_fitted_projector(data, pod):
    """projector= takes a fitted projector of the ROM's class and of its size (regression: n_modes given with
    projector= was silently ignored), and the seed still reaches the ESN."""
    assert POD_ESN._projector_cls is POD and POD_ESN.extra_print_params[-1] == "N_modes"
    with pytest.raises(ValueError, match="N_latent=6"):
        POD_ESN(data=data, dt=0.01, projector=pod, n_modes=4, train_ESN=False)

    X = ks_like(N_t=300)
    pod = POD(n_modes=3).fit(X)
    rom = POD_ESN(data=X, dt=0.1, projector=pod, sensor_locations=[0, 40, 80], N_units=20, N_wash=5,
                  t_train=20., t_val=5., t_test=0., hyperparameters_to_optimize=[], plot_training=False)
    assert np.array_equal(rom.Psi, pod.Psi)
    other = POD_ESN(data=X, dt=0.1, projector=pod, sensor_locations=[0, 40, 80], N_units=20, N_wash=5, seed=3,
                    t_train=20., t_val=5., t_test=0., hyperparameters_to_optimize=[], plot_training=False)
    assert other.seed == 3 and not np.array_equal(other.Win.toarray(), rom.Win.toarray())   # the seed reaches the ESN
    with pytest.raises(TypeError, match="fitted POD"):
        POD_ESN(data=X, dt=0.1, projector=object(), N_units=20)


# ── a trained POD_ESN on a small synthetic travelling-wave 'flow'

@pytest.fixture(scope='module')
def pod_esn_case():
    X = synthetic_flow(N_t=260)
    dt = 0.05
    return POD_ESN(data=X, domain=[0, 2 * np.pi, 0, np.pi], N_modes=4, skip_sensor_placement=False, qr_selection=True,
                   down_sample_measurement=[3, 2], t_val=20 * dt, t_train=200 * dt, t_test=0., perform_test=False,
                   N_units=30, train_ESN=True, N_wash=5, noise=1e-4, N_func_evals=4, N_grid=2, rho_range=(0.2, 0.9),
                   upsample=2, Nq=6, dt=dt, plot_case=False), X


def test_sensors_sit_on_fluid_points_and_read_the_data(pod_esn_case):
    """Regression test for the sensor-placement bug: the model observables computed from the POD coefficients must
    equal the (POD-reconstructed) flow at the sensor locations, indexed on the RAW grid like the tutorials do."""
    case, X = pod_esn_case
    Nu, Nx, Ny = case.grid_shape
    locs = np.asarray(case.sensor_locations)
    assert case.Nq == len(locs)
    assert np.all(locs >= 0) and np.all(locs < Nu * Nx * Ny)
    # every sensor must sit on a fluid point
    assert np.all(case.fluid_mask_flat[locs % (Nx * Ny)])

    # observables from the training coefficients at the last snapshot
    Phi_last = case.Phi[:, -1][:, np.newaxis]              # (N_modes, 1)
    y_model = case.get_observables(Z=Phi_last)           # (Nq, 1)
    # ground truth: raw data at the sensor grid locations (variable-block flattening)
    X_flat = X.transpose(1, 0, 2, 3).reshape(X.shape[1], -1)  # (N_t, Nu*Nx*Ny)
    y_data = X_flat[-1, case.sensor_locations]
    # POD truncation error bound: compare against the N_modes reconstruction instead
    rows = case.grid_index_to_flat_rows(case.sensor_locations)
    y_recon = (case.Psi[rows] @ Phi_last + case.Q_mean[rows]).ravel()
    np.testing.assert_allclose(y_model.ravel(), y_recon, atol=1e-8)
    # and the reconstruction itself must be close to the true data at the sensors
    scale = np.max(abs(X_flat[-1][case.fluid_mask_flat[
        np.arange(X_flat.shape[1]) % (case.grid_shape[1] * case.grid_shape[2])]]))
    assert np.max(abs(y_recon - y_data)) < 0.15 * scale


def test_forecast_observables_and_measure_modes(pod_esn_case):
    case, _ = pod_esn_case
    _check_protocol(case)
    model = case.copy()
    psi, t = model.time_integrate(Nt=100)
    model.update_history(psi, t)
    assert np.isfinite(psi).all()
    y = model.get_observables()
    assert y.shape[0] == model.Nq
    assert np.isfinite(y).all()
    model = case.copy()
    model.select_sensors(measure_modes=True)
    assert model.Nq == model.N_modes
    y = model.get_observables()
    assert y.shape[0] == model.N_modes


# ── several records: `LatentROM.__init__` takes `data` as one record ``(Nu, N_t, Nx, Ny)`` or a list of records of
# that shape (e.g. a trajectory and its mirror image): the projector fits the records together and the forecaster
# trains on each as its own segment.

ESN_SEGMENTS = dict(N_units=30, N_wash=5, noise=1e-4, upsample=1, t_train=150 * DT, t_val=40 * DT, t_test=0.,
                    perform_test=False, plot_training=False, seed=0, hyperparameters_to_optimize=[], rho=0.5,
                    sigma_in=1.0, tikh=1e-3, tikh_range=[1e-3], skip_sensor_placement=True)


def field(N_t=200, Nx=12, Ny=8, phase=0.):
    t = np.arange(N_t) * DT
    x, y = np.linspace(0, 2 * np.pi, Nx), np.linspace(-1, 1, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    u = np.sin(X - 2 * t[:, None, None] + phase) * np.cos(np.pi * Y / 2) + 0.3 * np.sin(2 * X + Y)
    v = np.cos(X + Y - 2 * t[:, None, None] + phase)
    return np.array([u, v])


def basis(rom):
    q0 = rom.decode(np.zeros((rom.N_modes, 1)))
    return rom.decode(np.eye(rom.N_modes)) - q0, q0


def test_segments_fit_the_projector_on_all_records():
    A, B = field(), field(phase=1.)
    seg = POD_ESN(data=[A, B], dt=DT, N_modes=4, train_ESN=False, skip_sensor_placement=True)
    cat = POD_ESN(data=np.concatenate([A, B], axis=1), dt=DT, N_modes=4, train_ESN=False,
                  skip_sensor_placement=True)
    (_, m1), (_, m2) = basis(seg), basis(cat)
    np.testing.assert_allclose(m1, m2, atol=1e-10)
    # the same subspace (modes of equal energy may rotate within their pair): the same reconstruction
    np.testing.assert_allclose(seg.decode(seg.encode(A)), cat.decode(cat.encode(A)), atol=1e-8)
    with pytest.raises(ValueError, match='same shape'):
        POD_ESN(data=[field(), field(N_t=150)], dt=DT, N_modes=4, train_ESN=False, skip_sensor_placement=True)


def test_segments_train_the_forecaster_on_each_record():
    A, B = field(), field(phase=1.)
    rom = POD_ESN(data=[A, B], dt=DT, N_modes=4, **ESN_SEGMENTS)
    assert rom.trained
    assert rom.validation_data.shape[0] == 2                  # one segment per record
    one = POD_ESN(data=np.concatenate([A, B], axis=1), dt=DT, N_modes=4, **ESN_SEGMENTS)
    assert one.validation_data.shape[0] == 1
    assert not np.allclose(rom.Wout, one.Wout)               # no training pair across the join
    psi, _ = rom.time_integrate(Nt=20)
    assert np.all(np.isfinite(np.asarray(psi)))


def test_projector_forecaster_pipeline_runs_end_to_end():
    """The projector + forecaster wiring of tutorial 30: encode -> ESN_model -> closed-loop forecast -> decode, on
    small synthetic data with an untuned ESN (no Bayesian search), so it checks that the wiring runs end to end, not
    the accuracy of the forecast or of the reconstruction. Generic over the fitted projector."""
    X = synthetic_field(N_t=200)
    projectors = [POD(n_modes=4, method='exact')]
    for projector in projectors:
        Phi = projector.fit(X).encode(X)
        N_train, n_x = 160, Phi.shape[1]
        esn = ESN_model(data=phi_to_esn_layout(Phi[:, :N_train]), dt=1.0, N_units=20,
                        perform_test=False, hyperparameters_to_optimize=[], plot_training=False)
        n_steps = n_x - N_train
        wash = Phi[:, N_train - esn.N_wash:N_train].T          # open-loop washout, then the closed loop
        pred, _ = esn.closed_loop(wash, n_steps)
        assert pred.shape == (n_steps, esn.N_dim)
        assert projector.decode(pred.T).shape[1] == n_steps


# ── every latent ROM is a forecast model for romda's data-assimilation loop: `POD_ESN` is assimilated in
# scripts/podesn_cylinder/pod_esn.py through an `EnsembleEstimator` (EnSRKF) and `run_da_loop`. The same path must run
# with every latent ROM: the ensemble is built from the ROM, the truth is the held-out snapshots read at the ROM's
# sensors, and the filter runs forecast and analysis steps. Tiny synthetic 2D fields and a few epochs keep this fast.

def wake_like_field(N_t=300, Nx=12, Ny=8, seed=0):
    """Two travelling-wave components (Nu, N_t, Nx, Ny) with a NaN body."""
    rng = np.random.default_rng(seed)
    t = np.arange(N_t) * DT
    x = np.linspace(0, 2 * np.pi, Nx)
    y = np.linspace(0, np.pi, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    u = (np.sin(X)[None] * np.cos(2 * t)[:, None, None]
         + 0.5 * np.sin(2 * X + Y)[None] * np.sin(4 * t)[:, None, None])
    v = np.cos(X + Y)[None] * np.sin(2 * t)[:, None, None]
    U = np.array([u + 0.01 * rng.standard_normal(u.shape), v + 0.01 * rng.standard_normal(v.shape)])
    U[:, :, 2:4, 3:5] = np.nan
    return U


ESN_KW = dict(N_units=30, N_wash=5, N_func_evals=4, N_grid=2, noise=1e-4, rho_range=(0.2, 0.9),
              upsample=1, t_test=0., perform_test=False, plot_training=False, seed=0)


def build_rom(name, X_train):
    import romda.models.data_driven as dd

    common = dict(data=X_train, dt=DT, domain=DOMAIN, Nq=3, qr_selection=True,
                  t_train=200 * DT, t_val=40 * DT, **ESN_KW)
    if name == 'CAE_ESN':
        return dd.CAE_ESN(n_latent=4, channels=(4, 8), projector_epochs=30, val_fraction=0.1, **common)
    return dd.POD_ESN(N_modes=4, **common)


@pytest.mark.parametrize('name', ['POD_ESN',
                                  pytest.param('CAE_ESN', marks=needs_torch),
                                  ])
def test_rom_runs_through_the_da_loop(name):
    X = wake_like_field()
    X_train, X_filter = X[:, :250], X[:, 250:]
    rom = build_rom(name, X_train)
    assert rom.Nq == 3 * GRID[0] == len(rom.sensor_locations)

    # truth: the held-out snapshots at the sensors, as pod_esn.build_truth reads them
    N_test = X_filter.shape[1]
    X_flat = X_filter.transpose(1, 0, 2, 3).reshape(N_test, -1)
    t_true = np.arange(N_test) * rom.dt
    truth = Observations(y_raw=X_flat[:, rom.sensor_locations], y_true=X_flat[:, rom.sensor_locations],
                         t_true=t_true, t_start=0.5, t_stop=2.0, Nt_obs=5)

    ensemble = EnSRKF(parent_model=rom, m=6)
    assert isinstance(ensemble, EnsembleEstimator)
    assert ensemble.model.get_observables().shape == (rom.Nq, 6)

    filter_ens = run_da_loop(ensemble, truth, std_obs=0.05, t_extra=0.2)

    hist = filter_ens.model.hist
    assert np.isfinite(hist).all()
    assert hist.shape[1:] == (rom.Nphi, 6)
    y = filter_ens.model.get_observable_hist()
    assert y.shape[1:] == (rom.Nq, 6) and np.isfinite(y).all()
    # the forecast covers every observation time and the extra time after the last one
    assert filter_ens.model.hist_t[-1] >= truth.t_obs[-1] + 0.2 - 1e-9


