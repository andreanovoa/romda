"""
Shared configuration for the test suite.

The tests import the installed package as ``romda`` (``pip install -e .``).
"""

import matplotlib
import numpy as np

matplotlib.use('Agg')  # never open figure windows during tests


def synthetic_flow(N_t=200, Nx=24, Ny=12, noise=0.0, body=True, seed=0):
    """Two travelling-wave fields ``(2, N_t, Nx, Ny)``, three coherent structures plus
    `noise`, with a NaN 'body' mask unless ``body=False``. The packaged
    `romda.models.data_driven.synthetic_field` is the ``noise=0.02, body=False`` case."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 8 * np.pi, N_t)
    x = np.linspace(0, 2 * np.pi, Nx)
    y = np.linspace(0, np.pi, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    u = (np.sin(X)[None] * np.cos(t)[:, None, None]
         + 0.5 * np.sin(2 * X + Y)[None] * np.sin(2 * t)[:, None, None])
    v = np.cos(X + Y)[None] * np.sin(t)[:, None, None]
    u = u + noise * rng.standard_normal(u.shape)
    v = v + noise * rng.standard_normal(v.shape)
    if body:
        u[:, 2:4, 2:4] = v[:, 2:4, 2:4] = np.nan
    return np.array([u, v])

