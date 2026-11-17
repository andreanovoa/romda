"""The convolutional autoencoders (autoencoders/cae.py, cae_esn.py and torch_utils.py): the periodic padding and the
resize decoder of `CAE` and `MultiScaleCAE`, `PeriodicConvTranspose2d`, the stride per stage, the convolutional
bottleneck, `CAE1D` and `MultiScaleCAE`, the autoencoders of the CAE-ESN (MagriLab/DA-CAE-ESN), and the CAE-ESN
reduced-order models `CAE1D_ESN` and `MultiScaleCAE_ESN`. torch is optional: without it, the module is skipped.

The weights in tutorials/weights are those of the reference runs `ks_l20pi`
and `kolmogorov_re30` of MagriLab/DA-CAE-ESN; a test skips when its file is absent.
"""
import os

import numpy as np
import pytest
from conftest import ks_like

torch = pytest.importorskip("torch")

from romda.models.data_driven import CAE, CAE1D, CAE1D_ESN, MultiScaleCAE, MultiScaleCAE_ESN  # noqa: E402
from romda.models.data_driven.autoencoders.torch_utils import PeriodicConvTranspose2d  # noqa: E402

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "tutorials", "weights")


def periodic_record(n_t=12, n=48, seed=0):
    """Random periodic fields (1, n_t, n, n): a few Fourier modes with random phases."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 2 * np.pi, n, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    fields = [sum(np.sin(kx * X + ky * Y + rng.uniform(0, 2 * np.pi)) for kx, ky in ((1, 0), (0, 2), (3, 1)))
              for _ in range(n_t)]
    return np.asarray(fields, dtype=np.float32)[None]


def ring_record(n_t=12, n=40, seed=0):
    """Random periodic fields on a ring, as a grid (1, n_t, n, 1)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 2 * np.pi, n, endpoint=False)
    fields = [sum(np.sin(k * x + rng.uniform(0, 2 * np.pi)) for k in (1, 3, 8)) for _ in range(n_t)]
    return np.asarray(fields, dtype=np.float32)[None, :, :, None]


def test_periodic_transposed_convolution_equals_tiled_reference():
    """The layer equals the zero-padded transposed convolution of the 3 x 3 tiled input, cropped to the middle tile."""
    for k, s, p in ((3, 2, 1), (5, 2, 2), (7, 2, 3), (4, 2, 1), (5, 1, 2)):
        torch.manual_seed(0)
        layer = PeriodicConvTranspose2d(3, 2, k, s, p)
        ref = torch.nn.ConvTranspose2d(3, 2, k, s, padding=0)
        ref.load_state_dict(layer.state_dict())
        x = torch.randn(2, 3, 6, 5)
        y = layer(x)
        tiled = ref(x.repeat(1, 1, 3, 3))
        h, w = x.shape[-2:]
        o_h, o_w = s * h + p, s * w + p                    # where the middle tile starts in the unpadded output
        expected = tiled[..., o_h:o_h + s * h, o_w:o_w + s * w]
        assert y.shape[-2:] == (s * h, s * w)
        torch.testing.assert_close(y, expected)


def test_periodic_networks_are_shift_equivariant():
    """A shift of the field by one bottleneck cell (8 grid points) shifts the bottleneck map, and back, exactly: the
    CAE and the MultiScaleCAE with transposed convolutions, and the CAE with the resize decoder."""
    G = torch.as_tensor(periodic_record(n_t=3, seed=1)[0][:, None])           # (B, 1, 48, 48)
    for cls, upsampling in ((CAE, "transposed"), (MultiScaleCAE, "transposed"), (CAE, "resize")):
        kw = dict(decoder_upsampling=upsampling) if upsampling == "resize" else {}
        model = cls(n_latent=4, periodic_padding=True, epochs=1, batch_size=4, val_fraction=0, seed=0,
                    **kw).fit(periodic_record())
        with torch.no_grad():
            if cls is CAE:
                enc, dec = model._enc_conv, model._dec_conv
            else:
                enc = model.networks()[0][0][:8]                               # convolutions of the first branch
                dec = model.networks()[1][0][3:]                               # transposed convolutions onwards
            for net, x, shift_in, shift_out in ((enc, G, 8, 1), (dec, enc(G), 1, 8)):
                torch.testing.assert_close(net(torch.roll(x, (shift_in, shift_in), dims=(-2, -1))),
                                           torch.roll(net(x), (shift_out, shift_out), dims=(-2, -1)), atol=1e-5,
                                           rtol=1e-5)


def test_periodic_multiscale_cae_restores_from_its_state():
    """`_options_from_state` recognizes the periodic last layer (kernel k instead of k + 1)."""
    model = MultiScaleCAE(n_latent=4, periodic_padding=True, epochs=1, val_fraction=0, seed=0).fit(periodic_record(n_t=4))
    state = {f"{name}.{k}": v for name, net in model._networks_by_name().items() for k, v in net.state_dict().items()}
    assert MultiScaleCAE._options_from_state(state)["periodic_padding"] is True
    restored = MultiScaleCAE.load_state_dicts(state, (1, 48, 48), scale=1.0)
    G = torch.as_tensor(periodic_record(n_t=2, seed=3)[0][:, None])
    with torch.no_grad():
        torch.testing.assert_close(restored.decode_grid(restored.encode_grid(G)), model.decode_grid(model.encode_grid(G)))


def test_resize_decoder_restores_the_grid():
    """The resize decoder (upsampling + stride-1 convolution) restores the grid, odd sizes included, and round-trips."""
    for periodic, n in ((True, 48), (False, 48), (False, 45)):
        X = periodic_record(n_t=6, n=n)
        model = CAE(n_latent=4, periodic_padding=periodic, decoder_upsampling="resize", epochs=1, batch_size=4,
                    val_fraction=0, seed=0).fit(X)
        assert model.reconstruct(X).shape == (n * n, 6)
        assert not any(isinstance(m, torch.nn.ConvTranspose2d) for m in model._dec_conv)
        back = CAE.from_trained(model.config_options(), model.trained_arrays())
        np.testing.assert_allclose(back.reconstruct(X), model.reconstruct(X), atol=1e-6)


def test_periodic_cae_on_a_ring_is_shift_equivariant():
    """A grid (40, 1) is a one-dimensional periodic field: kernels (k, 1), and a shift by one bottleneck cell
    (8 sites after three stages) shifts the bottleneck map, and back, exactly."""
    for upsampling in ("transposed", "resize"):
        model = CAE(n_latent=4, periodic_padding=True, decoder_upsampling=upsampling, epochs=1, batch_size=4,
                    val_fraction=0, seed=0).fit(ring_record())
        assert all(m.kernel_size == (3, 1) for m in model._enc_conv if isinstance(m, torch.nn.Conv2d))
        assert model._red_shape == (64, 5, 1)
        G = torch.as_tensor(ring_record(n_t=3, seed=1)[0][:, None])                  # (B, 1, 40, 1)
        with torch.no_grad():
            x = model._enc_conv(G)
            torch.testing.assert_close(model._enc_conv(torch.roll(G, 8, dims=-2)), torch.roll(x, 1, dims=-2),
                                       atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(model._dec_conv(torch.roll(x, 1, dims=-2)),
                                       torch.roll(model._dec_conv(x), 8, dims=-2), atol=1e-5, rtol=1e-5)
        back = CAE.from_trained(model.config_options(), model.trained_arrays())
        X = ring_record(n_t=5, seed=2)
        np.testing.assert_allclose(back.reconstruct(X), model.reconstruct(X), atol=1e-6)


def test_hidden_layers_round_trip_on_a_ring():
    """`hidden` puts dense layers between the bottleneck grid and the latent coefficients, mirrored in the decoder;
    the default keeps the single linear maps, whose saved weights load unchanged (`enc_fc.weight`)."""
    model = CAE(n_latent=4, periodic_padding=True, decoder_upsampling="resize", hidden=(16,), epochs=1, batch_size=4,
                val_fraction=0, seed=0).fit(ring_record())
    assert [m.out_features for m in model.enc_fc if isinstance(m, torch.nn.Linear)] == [16, 4]
    assert [m.out_features for m in model.dec_fc if isinstance(m, torch.nn.Linear)] == [16, 64 * 5]
    back = CAE.from_trained(model.config_options(), model.trained_arrays())
    X = ring_record(n_t=5, seed=2)
    np.testing.assert_allclose(back.reconstruct(X), model.reconstruct(X), atol=1e-6)
    plain = CAE(n_latent=4, epochs=1, batch_size=4, val_fraction=0, seed=0).fit(ring_record())
    assert {"enc_fc.weight", "dec_fc.weight"} <= set(plain.trained_arrays())
    # zero padding is the default
    assert not plain.periodic_padding
    assert not any(isinstance(m, PeriodicConvTranspose2d) for m in plain._dec_conv)
    assert all(m.padding_mode == "zeros" for m in plain._enc_conv if isinstance(m, torch.nn.Conv2d))


def test_per_stage_stride_keeps_a_full_resolution_stage():
    """`stride=(1, 2, 2)` convolves the 40 sites before halving them twice (a periodic autoencoder of
    the Lorenz 96 ring); the network stays shift-equivariant by one bottleneck cell (4 sites)
    and round-trips."""
    for upsampling in ("transposed", "resize"):
        model = CAE(n_latent=4, channels=(8, 8, 8), stride=(1, 2, 2), kernel_size=5, pad=2, activation="gelu",
                    hidden=(16,), periodic_padding=True, decoder_upsampling=upsampling, epochs=1, batch_size=4,
                    val_fraction=0, seed=0).fit(ring_record())
        assert [m.stride for m in model.enc_conv if isinstance(m, torch.nn.Conv2d)] == [(1, 1), (2, 1), (2, 1)]
        assert model._red_shape == (8, 10, 1)
        G = torch.as_tensor(ring_record(n_t=3, seed=1)[0][:, None])
        with torch.no_grad():
            x = model.enc_conv(G)
            torch.testing.assert_close(model.enc_conv(torch.roll(G, 4, dims=-2)), torch.roll(x, 1, dims=-2),
                                       atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(model.dec_conv(torch.roll(x, 1, dims=-2)),
                                       torch.roll(model.dec_conv(x), 4, dims=-2), atol=1e-5, rtol=1e-5)
        back = CAE.from_trained(model.config_options(), model.trained_arrays())
        X = ring_record(n_t=5, seed=2)
        np.testing.assert_allclose(back.reconstruct(X), model.reconstruct(X), atol=1e-6)
        zero = CAE(n_latent=4, channels=(4, 4), stride=(1, 2), decoder_upsampling=upsampling, epochs=1,
                   val_fraction=0, seed=0).fit(periodic_record(n_t=4, n=45))
        assert zero.reconstruct(periodic_record(n_t=4, n=45)).shape == (45 * 45, 4)


def test_convolutional_bottleneck_keeps_the_grid():
    """`latent_channels` keeps the bottleneck grid: the reduced coordinates are its channels (n_latent follows the
    grid), the weights do not grow with the grid, a periodic network shifts its reduced coordinates with the field
    (by one bottleneck cell, 4 points), and the fit round-trips."""
    X = periodic_record(n_t=6, n=48)
    for periodic in (False, True):
        options = dict(n_latent=4, channels=(4, 8), latent_channels=2, periodic_padding=periodic, epochs=1,
                       batch_size=4, val_fraction=0, seed=0)
        model = CAE(**options).fit(X)
        assert model.n_latent == model.N_latent == 2 * 12 * 12
        assert model.encode(X).shape == (2 * 12 * 12, 6)
        assert CAE(**options).fit(periodic_record(n_t=6, n=96)).n_params == model.n_params
        back = CAE.from_trained(model.config_options(), model.trained_arrays())
        np.testing.assert_allclose(back.reconstruct(X), model.reconstruct(X), atol=1e-6)
        if periodic:
            G = torch.as_tensor(X[0][:, None])
            with torch.no_grad():
                Z = model.encode_grid(G).view(-1, 2, 12, 12)
                torch.testing.assert_close(model.encode_grid(torch.roll(G, 4, dims=-1)).view(-1, 2, 12, 12),
                                           torch.roll(Z, 1, dims=-1), atol=1e-5, rtol=1e-5)


def reference_weights(name):
    """The state dictionary of a reference run in tutorials/weights; skips the test when the file is absent."""
    path = os.path.join(WEIGHTS_DIR, f"DA-CAE-ESN_{name}.pth")
    if not os.path.isfile(path):
        pytest.skip(f"{path} is not distributed")
    return torch.load(path, map_location="cpu")


def test_reference_ks_weights_load():
    state = reference_weights("ks_l20pi")
    cae = CAE1D.load_state_dicts(state, grid_shape=(1, 128, 1), scale=3.58)
    X = ks_like()
    Z = cae.encode(X)
    assert Z.shape == (24, X.shape[1])
    assert np.all(np.abs(Z) <= 1)                       # tanh latent layer
    assert cae.reconstruct(X).shape == (128, X.shape[1])
    assert cae.n_params == sum(v.numel() for v in state.values())


def test_reference_kolmogorov_weights_load_with_their_branches():
    state = reference_weights("kolmogorov_re30")
    cae = MultiScaleCAE.load_state_dicts(state, grid_shape=(1, 48, 48), scale=8.76)
    assert cae.N_latent == 16 and cae.kernel_sizes == (3, 5, 7)
    assert cae.n_params == sum(v.numel() for v in state.values())
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1, 5, 48, 48))
    Z = cae.encode(X)
    assert Z.shape == (16, 5) and np.all(np.abs(Z) <= 3)        # three tanh branches, summed
    assert cae.reconstruct(X).shape == (48 * 48, 5)


def test_the_cae_esn_roms_forecast_and_observe_the_decoded_field():
    """With a trained autoencoder (`projector=`), the CAE-ESN trains the ESN alone on the reduced coordinates, forecasts
    them in closed loop and observes the decoded field at the sensors."""
    esn = dict(N_units=30, N_wash=5, t_test=0., upsample=1, rho=0.5, sigma_in=0.5, tikh=1e-6,
               hyperparameters_to_optimize=[], plot_training=False, seed=0, verbose=False)
    X = ks_like(N_t=300)
    cae = CAE1D(n_latent=4, epochs=2, val_fraction=0.).fit(X)
    sensors = np.arange(0, 128, 16)
    rom = CAE1D_ESN(data=X, dt=0.1, projector=cae, sensor_locations=sensors, t_train=20., t_val=4., **esn)
    assert (rom.N_latent, rom.Nq, rom.Nphi) == (4, 8, 4 + 30)
    z = rom.current_state[:rom.N_latent]
    np.testing.assert_allclose(rom.get_observables(), rom.decode(z)[sensors])
    Z_f, _ = rom.closed_loop(cae.encode(X).T[:50], 20)
    assert Z_f.shape == (20, 4) and np.all(np.isfinite(Z_f))
    psi, t = rom.time_integrate(Nt=10)
    assert psi.shape == (10, rom.Nphi, 1) and t.shape == (10,)

    W = periodic_record(n_t=80)
    ms = MultiScaleCAE(n_latent=4, kernel_sizes=(3,), epochs=1, val_fraction=0.).fit(W)
    rom = MultiScaleCAE_ESN(data=W, dt=0.1, projector=ms, sensor_locations=np.arange(0, 48 * 48, 300), t_train=6.,
                            t_val=1., **esn)
    assert rom.Nq == 8 and rom.time_integrate(Nt=5)[0].shape == (5, 4 + 30, 1)


def test_the_cae_esn_autoencoders_keep_the_mean_and_divide_by_the_maximum():
    X = ks_like(N_t=40)
    cae = CAE1D(n_latent=4, epochs=2, val_fraction=0.).fit(X)
    assert np.allclose(cae.Q_mean, 0)
    assert np.allclose(cae.scale, np.abs(X).max())
    x = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    X = np.array([np.sin(4 * x[None] + 0.1 * t) * np.cos(x[:, None]) for t in range(20)])[None]
    cae = MultiScaleCAE(n_latent=4, kernel_sizes=(3, 5), epochs=2, val_fraction=0.).fit(X)
    assert np.allclose(cae.Q_mean, 0) and np.allclose(cae.scale, np.abs(X).max())
    assert len(cae.networks()[0]) == 2


def test_the_options_and_grids_are_checked():
    """A periodic network needs a grid or a ring that halves at every stage; CAE convolves the grid, so a flat data
    matrix is refused with a clear message; CAE1D and MultiScaleCAE restore the grids of the reference runs alone."""
    with pytest.raises(ValueError, match="periodic_padding=True"):
        CAE(n_latent=4, periodic_padding=True, epochs=1, val_fraction=0).fit(periodic_record(n_t=4, n=50))
    with pytest.raises(ValueError, match="periodic_padding=True"):
        CAE(n_latent=4, periodic_padding=True, epochs=1, val_fraction=0).fit(ring_record(n_t=4, n=36))
    with pytest.raises(ValueError, match="decoder_upsampling"):
        CAE(n_latent=4, decoder_upsampling="bilinear")
    with pytest.raises(ValueError, match="hidden"):
        CAE(n_latent=4, hidden=(0,))
    with pytest.raises(ValueError, match="hidden"):
        CAE(latent_channels=2, hidden=(8,))
    with pytest.raises(ValueError, match="one entry per stage"):
        CAE(n_latent=4, channels=(8, 8), stride=(1, 2, 2))
    with pytest.raises(ValueError, match='grid snapshots'):
        CAE().fit(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="choose Nx"):
        CAE1D(n_latent=4, epochs=1, val_fraction=0.).fit(ks_like(N_t=10, Nx=100))
    with pytest.raises(ValueError, match="one-dimensional"):
        CAE1D(n_latent=4, epochs=1, val_fraction=0.).fit(np.zeros((1, 10, 128, 2)))
    with pytest.raises(ValueError, match="choose a grid"):
        MultiScaleCAE(n_latent=4, epochs=1, val_fraction=0.).fit(np.zeros((1, 5, 50, 50)))
