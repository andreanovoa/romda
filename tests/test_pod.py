"""Tests for the POD / SPOD decompositions (models.data_driven.autoencoders: pod, pod_utils) and the projector
additions that sensor placement is built on (autoencoders/base.py: `spatial_basis`, `decode_at`)."""
import numpy as np
import pytest
from conftest import synthetic_flow
from romda.models.data_driven.autoencoders import POD, SPOD, Projector, spod_towne
from romda.models.data_driven.autoencoders.pod_utils import snapshot_pod, spod_sieber


def test_fit_exact():
    X = synthetic_flow(noise=0.02)  # noise makes the data full-rank
    pod = POD(n_modes=10, method='exact').fit(X)
    assert pod.fitted
    assert pod.Psi.shape[1] == 10
    assert pod.Phi.shape == (10, X.shape[1])
    assert pod.Sigma.shape == (10,)
    # orthonormal spatial modes
    np.testing.assert_allclose(pod.Psi.T @ pod.Psi, np.eye(10), atol=1e-6)
    # singular values descending
    assert np.all(np.diff(pod.Sigma) <= 1e-12)
    # the synthetic flow has 3 coherent structures: 4 modes capture >99% energy
    pod = POD(n_modes=8, method='exact').fit(synthetic_flow())
    _, cum = pod.energy_fraction()
    assert cum[3] > 0.99
    pod.truncate(3)
    assert pod.N_latent == 3
    assert pod.Psi.shape[1] == 3
    assert pod.Phi.shape[0] == 3


def test_encode_decode_roundtrip():
    X = synthetic_flow()
    pod = POD(n_modes=20, method='exact').fit(X)
    Q = pod.preprocess_snapshot(X)
    Z = pod.encode(X)
    np.testing.assert_allclose(Z, pod.Phi, atol=1e-8)
    Q_hat = pod.decode(Z) - pod.Q_mean
    # low-rank data: 20 modes should reconstruct nearly exactly
    assert np.mean((Q - Q_hat) ** 2) < 1e-10 * np.mean(Q ** 2) + 1e-12
    # POD can be fitted directly on a flat (N_x, N_t) data matrix
    Q_raw = np.random.default_rng(5).normal(size=(30, 80)) + 3.
    pod = POD(n_modes=5, method='exact').fit(Q_raw)
    assert pod.Psi.shape == (30, 5)
    np.testing.assert_allclose(pod.encode(Q_raw), pod.Phi, atol=1e-8)


def test_randomized_close_to_exact():
    X = synthetic_flow(noise=0.01)
    pod_e = POD(n_modes=4, method='exact').fit(X)
    pod_r = POD(n_modes=4, method='randomized', random_state=0).fit(X)
    # the leading (energetic) singular values must agree
    np.testing.assert_allclose(pod_e.Sigma[:3], pod_r.Sigma[:3], rtol=5e-2)


def test_pod_checks_its_options():
    """An unfitted POD has no modes. Regression: POD dropped typos silently, and an explicit n_modes=20 was
    overridden by the rows of a pre-set Phi."""
    with pytest.raises(AttributeError):
        _ = POD(n_modes=3).Psi
    with pytest.raises(TypeError, match="n_mode"):
        POD(n_mode=3)
    assert POD(n_modes=20, Phi=np.zeros((5, 10))).N_latent == 20
    assert POD(Phi=np.zeros((5, 10))).N_latent == 5


def test_variable_block_ordering():
    """Rows [0:N_fluid] of the flat representation must be the FIRST field:
    regression test for the documented Psi_ux = Psi[:N_fluid] convention."""
    X = synthetic_flow()
    X[1] = 0. * X[1]  # kill the second field (keep its NaN mask)
    X[1, :, 2:4, 2:4] = np.nan
    prep = POD()
    Q = prep.preprocess_snapshot(X, subtract_mean=False)
    N_fluid = int(prep.fluid_mask_flat.sum())
    assert np.any(Q[:N_fluid] != 0.)                       # first block: u field
    np.testing.assert_allclose(Q[N_fluid:], 0.)            # second block: zeroed field


def test_grid_index_to_flat_rows_roundtrip():
    """decode(idx=grid_index_to_flat_rows(...)) must return the data at those points."""
    X = synthetic_flow()
    pod = POD(n_modes=30, method='exact').fit(X)
    Nu, Nx, Ny = pod.grid_shape

    fluid_idx = np.flatnonzero(pod.fluid_mask_flat)
    picks = fluid_idx[[3, 17, 101]]
    grid_idx = np.concatenate([picks, picks + Nx * Ny])   # same points in both fields

    rows = pod.grid_index_to_flat_rows(grid_idx)
    Q = pod.preprocess_snapshot(X)
    rec = pod.decode(pod.Phi, idx=rows) - pod.Q_mean[rows]
    np.testing.assert_allclose(rec, Q[rows], atol=1e-8)

    # non-fluid points must be rejected
    solid = np.flatnonzero(~pod.fluid_mask_flat)[:1]
    with pytest.raises(ValueError):
        pod.grid_index_to_flat_rows(solid)


def test_spod_with_no_filter_is_pod():
    X = synthetic_flow(noise=0.02)
    pod = POD(n_modes=6, method='exact').fit(X)
    spod = SPOD(Nf=0, n_modes=6).fit(X)
    np.testing.assert_allclose(pod.Sigma, spod.Sigma, rtol=1e-8)
    # the functional route
    rng = np.random.default_rng(3)
    Q = rng.normal(size=(20, 40))
    Q -= Q.mean(axis=1, keepdims=True)
    np.testing.assert_allclose(snapshot_pod(Q)[0], spod_sieber(Q, Nf=0)[0], atol=1e-10)


def test_spod_encode_inverts_decode_and_cannot_beat_pod():
    """Regression: for Nf > 0 the SPOD spatial modes are not orthonormal, so
    Psi^T encoding did not invert decode (reconstruction error ~800 %)."""
    X = synthetic_flow(noise=0.05)
    spod = SPOD(Nf=10, filter_kind='gaussian', n_modes=6).fit(X)
    assert spod.Psi.shape[1] == 6
    assert np.isfinite(spod.Sigma).all()
    assert hasattr(spod, 'C_tilde')
    Z = spod.Phi[:, :7]
    np.testing.assert_allclose(spod.encode(spod.decode(Z)), Z, atol=1e-8)
    Q = spod.preprocess_snapshot(X)
    pod = POD(n_modes=6, method='exact').fit(X)
    err_pod, err_spod = (np.linalg.norm(m.decode(m.encode(X)) - m.Q_mean - Q) for m in (pod, spod))
    assert err_pod <= err_spod + 1e-10 <= np.linalg.norm(Q)


def test_spod_sieber_matches_hand_built_diagonal_filter():
    """Regression: spod_sieber filtered C as G^T C G (POD of time-filtered data)
    instead of Sieber's diagonal filter S_ij = sum_k g_k C_{i+k, j+k} (periodic)."""
    from romda.models.data_driven.autoencoders.pod_utils import _filter_kernel
    rng = np.random.default_rng(5)
    Q = rng.normal(size=(15, 30))
    Q -= Q.mean(axis=1, keepdims=True)
    Nf, N_t = 4, Q.shape[1]
    C, g = Q.T @ Q / N_t, _filter_kernel(Nf, 'gaussian')
    S = np.zeros_like(C)
    for i in range(N_t):
        for j in range(N_t):
            S[i, j] = sum(g[k + Nf] * C[(i + k) % N_t, (j + k) % N_t] for k in range(-Nf, Nf + 1))
    Sigma, Psi, Phi, C_tilde = spod_sieber(Q, Nf)
    np.testing.assert_allclose(C_tilde, S, atol=1e-12)
    np.testing.assert_allclose(Sigma ** 2, np.sort(np.linalg.eigvalsh(S))[::-1], atol=1e-10)
    np.testing.assert_allclose(Phi @ Phi.T, N_t * np.diag(Sigma ** 2), atol=1e-8)
    np.testing.assert_allclose(Psi @ Phi, Q, atol=1e-8)


def test_spod_towne_runs():
    """Regression test: spod_towne previously crashed with NameError
    (missing get_window / gammaincinv imports)."""
    t = np.linspace(0, 40 * np.pi, 512)
    Q = np.array([np.sin(t + phi) for phi in np.linspace(0, 1, 12)])
    Q += 0.01 * np.random.default_rng(4).standard_normal(Q.shape)
    Q -= Q.mean(axis=1, keepdims=True)
    L, Psi, f, Lc, info = spod_towne(Q, dt=t[1] - t[0], n_fft=128)
    assert L.shape[0] == len(f)
    assert np.isfinite(L).all()
    # regression: gammaincinv took (p, n_blks) in MATLAB order, so Lc was all NaN
    assert np.isfinite(Lc).all()
    assert (Lc[..., 0] <= Lc[..., 1]).all()
    # the dominant frequency should be ~1/(2 pi)
    f_peak = f[np.argmax(L[:, 0])]
    assert f_peak == pytest.approx(1 / (2 * np.pi), rel=0.2)


def test_error_metrics_do_not_mutate_their_inputs():
    """Regression: compute_RMS (and compute_MSE on >2-D input) set NaN to 0 in the caller's array."""
    truth = np.ones((2, 3, 4))
    truth[0, 0, 0] = np.nan
    POD.compute_MSE(np.zeros_like(truth), truth)
    POD.compute_RMS(np.zeros_like(truth), truth)
    assert np.isnan(truth[0, 0, 0])
    np.testing.assert_array_equal(POD.compute_RMS(np.zeros(2), np.array([-2.0, np.nan])), [2.0, 0.0])


# ── the projector additions that sensor placement is built on

def test_spatial_basis_is_the_jacobian_of_the_decoder(pod):
    """A linear decoder's Jacobian is its own matrix: POD must hand back Psi itself, not a copy or an approximation,
    or sensor placement would move. The base-class finite-difference Jacobian is validated against the one projector
    whose answer is known exactly."""
    assert pod.spatial_basis() is pod.Psi
    exact = pod.spatial_basis()
    generic = Projector.spatial_basis(pod)
    assert generic.shape == exact.shape
    assert np.linalg.norm(generic - exact) / np.linalg.norm(exact) < 1e-10
    with pytest.raises(ValueError, match="expected N_latent"):
        Projector.spatial_basis(pod, np.zeros(pod.N_latent + 1))


def test_decode_at_matches_decoding_then_indexing(pod, data):
    """POD overrides decode_at with the cheap `Psi[idx] @ Z` route; it must agree with the generic
    decode-everything-then-index path, and decode everything without idx."""
    Z = pod.encode(data)[:, :5]
    idx = np.array([3, 17, 44, 100])
    np.testing.assert_allclose(pod.decode_at(Z, idx=idx), Projector.decode_at(pod, Z, idx=idx), rtol=1e-12)
    np.testing.assert_allclose(pod.decode_at(Z[:, :4]), pod.decode(Z[:, :4]), rtol=1e-12)
