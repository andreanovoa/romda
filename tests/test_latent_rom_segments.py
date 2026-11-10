"""A latent ROM trained on several records (e.g. a trajectory and its mirror image).

`LatentROM.__init__` takes `data` as one record ``(Nu, N_t, Nx, Ny)`` or a list of records of that shape: the
projector fits the records together and the forecaster trains on each as its own segment.
"""

import numpy as np
import pytest

DT = 0.05
ESN_KW = dict(N_units=30, N_wash=5, noise=1e-4, upsample=1, t_train=150 * DT, t_val=40 * DT, t_test=0.,
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
    from romda.models.data_driven import POD_ESN
    A, B = field(), field(phase=1.)
    seg = POD_ESN(data=[A, B], dt=DT, N_modes=4, train_ESN=False, skip_sensor_placement=True)
    cat = POD_ESN(data=np.concatenate([A, B], axis=1), dt=DT, N_modes=4, train_ESN=False,
                  skip_sensor_placement=True)
    (_, m1), (_, m2) = basis(seg), basis(cat)
    np.testing.assert_allclose(m1, m2, atol=1e-10)
    # the same subspace (modes of equal energy may rotate within their pair): the same reconstruction
    np.testing.assert_allclose(seg.decode(seg.encode(A)), cat.decode(cat.encode(A)), atol=1e-8)


def test_segments_train_the_forecaster_on_each_record():
    from romda.models.data_driven import POD_ESN
    A, B = field(), field(phase=1.)
    rom = POD_ESN(data=[A, B], dt=DT, N_modes=4, **ESN_KW)
    assert rom.trained
    assert rom.validation_data.shape[0] == 2                  # one segment per record
    one = POD_ESN(data=np.concatenate([A, B], axis=1), dt=DT, N_modes=4, **ESN_KW)
    assert one.validation_data.shape[0] == 1
    assert not np.allclose(rom.Wout, one.Wout)               # no training pair across the join
    psi, _ = rom.time_integrate(Nt=20)
    assert np.all(np.isfinite(np.asarray(psi)))


def test_segments_must_have_the_same_shape():
    from romda.models.data_driven import POD_ESN
    with pytest.raises(ValueError, match='same shape'):
        POD_ESN(data=[field(), field(N_t=150)], dt=DT, N_modes=4, train_ESN=False, skip_sensor_placement=True)
