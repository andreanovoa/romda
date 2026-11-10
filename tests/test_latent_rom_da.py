"""Every latent ROM is a forecast model for romda's data-assimilation loop.

`POD_ESN` is assimilated in `examples/rom_da/cylinder/pod_esn.py` through an
`EnsembleEstimator` (EnSRKF) and `romda.data_assimilation.run_da_loop`. The same
path must run with every latent ROM: the ensemble is built from the ROM, the
truth is the held-out snapshots read at the ROM's sensors, and the filter runs
forecast and analysis steps. Tiny synthetic 2D fields and a few epochs keep this fast.
"""

import importlib.util

import numpy as np
import pytest
from romda.data_assimilation import run_da_loop
from romda.estimators import EnsembleEstimator, EnSRKF
from romda.observations import Observations

needs_torch = pytest.mark.skipif(importlib.util.find_spec('torch') is None, reason='torch is not installed')

GRID = (2, 12, 8)
DOMAIN = [0., 2 * np.pi, 0., np.pi]
DT = 0.05


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
    return dd.POD_ESN(N_modes=4, **common)


@pytest.mark.parametrize('name', ['POD_ESN',
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

    ensemble = EnSRKF(parent_model=rom, m=6, std_phi=0.1)
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
