"""
Shared configuration for the test suite.

The tests import the installed package as ``romda`` (``pip install -e .``).
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')  # never open figure windows during tests

from romda.estimators import EnKF  # noqa: E402
from romda.models.data_driven import synthetic_field  # noqa: E402
from romda.models.data_driven.autoencoders import POD  # noqa: E402
from romda.models.physical import VdP  # noqa: E402
from romda.observations import Observations  # noqa: E402

GRID = (2, 16, 8)
DOMAIN = [0.0, 2 * np.pi, 0.0, np.pi]


def synthetic_flow(N_t=200, Nx=24, Ny=12, noise=0.0, body=True, seed=0):
    """The packaged `synthetic_field` ``(2, N_t, Nx, Ny)``, with a NaN 'body' mask unless ``body=False``."""
    X = synthetic_field(N_t=N_t, Nx=Nx, Ny=Ny, noise=noise, seed=seed)
    if body:
        X[:, :, 2:4, 2:4] = np.nan
    return X


def ks_like(N_t=200, Nx=128, seed=0):
    """(1, N_t, Nx, 1) travelling waves of amplitude ~3, as the KS field."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 2 * np.pi, Nx, endpoint=False)
    t = np.arange(N_t)[:, None] * 0.1
    U = sum(rng.uniform(0.5, 1.5) * np.sin(k * x[None] - rng.uniform(0.5, 2) * t + rng.uniform(0, 6)) for k in (3, 5, 8))
    return U[None, :, :, None]


@pytest.fixture(scope="module")
def data():
    return synthetic_field(N_t=300)


@pytest.fixture(scope="module")
def pod(data):
    return POD(n_modes=6, method="exact", grid_shape=GRID, domain=DOMAIN).fit(data)


# ── a Van der Pol twin for the DA loops

def vdp_twin(manual_bias=None, noise_level=0.01):
    """Thirty noisy observations of a Van der Pol truth between t = 0.6 and 0.75, biased by `manual_bias`."""
    return Observations(model=VdP, t_start=0.6, t_stop=0.75, t_max=0.9, Nt_obs=30, add_noise=True,
                        noise_type='gauss, add', noise_level=noise_level, manual_bias=manual_bias)


def obs_covariance(truth, std_obs=0.05):
    """The diagonal C_dd of the twin: `std_obs` times the squared maximum of each observable."""
    return np.diag(std_obs * np.ones(truth.y_obs.shape[1])) * np.max(abs(truth.y_obs), axis=0) ** 2


def vdp_ensemble(truth, da_method=EnKF, bias=None, m=8, **kwargs):
    """A filter of `m` Van der Pol members on the time step of `truth`, with zeta estimated."""
    return da_method(parent_model=VdP(dt=truth.dt), parent_bias=bias, m=m, std_phi=0.1,
                     std_alpha=dict(zeta=(40., 60.)), **kwargs)
