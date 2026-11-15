"""Gaussian-mixture analysis of the ensemble filters (`romda.estimators.sequential.mixture`).

Identities: one component gives the Gaussian analysis bit for bit; with two mirror components and data of z alone
(equal evidence), each component receives the analysis that the filter gives to its members alone, and the analysis
commutes with the symmetry (x, y, z) -> (-x, -y, z); data of x move every member to the component that they favour.
"""
import numpy as np
import pytest
from romda.estimators import EnSRKF
from romda.estimators.sequential.mixture import allocate, fit_mixture, redistribute
from romda.models.physical import Lorenz63

S = np.array([-1., -1., 1.])[:, None]
DT = 0.015
CLIMATOLOGY = np.array([7.9, 9.0, 8.6])     # standard deviation of x, y, z on the attractor


def lobe(m, seed=0, centre=(8., 8., 27.), std=(1.5, 2., 3.)):
    """m states about the centre of the x > 0 lobe of the Lorenz-63 attractor, (3, m)."""
    rng = np.random.default_rng(seed)
    return np.array(centre)[:, None] + np.array(std)[:, None] * rng.standard_normal((3, m))


def mirror_ensemble(m, seed=0):
    """m/2 members in the x > 0 lobe and their mirror images, interleaved."""
    A = lobe(m // 2, seed)
    E = np.empty((3, m))
    E[:, 0::2], E[:, 1::2] = A, S * A
    return E


def filt(E, observe_dims, **kw):
    return EnSRKF(parent_model=Lorenz63, dt=DT, m=E.shape[1], ensemble_psi0=E, observe_dims=observe_dims,
                  climatology=CLIMATOLOGY, **kw)


def test_fit_mixture_finds_the_two_lobes_and_one_gaussian():
    rng = np.random.default_rng(1)
    E = mirror_ensemble(20)
    labels, bic = fit_mixture(E, 2, rng)
    assert np.array_equal(labels, np.arange(20) % 2)            # interleaved: the first member is component 0
    assert bic[1] < bic[0]
    labels, bic = fit_mixture(lobe(20, seed=2), 2, rng)
    assert np.all(labels == 0) and bic[0] < bic[1]
    # a component needs min_members
    E = mirror_ensemble(10)
    E[:, [1, 3, 5, 7]] = E[:, [0, 2, 4, 6]]                      # one member left in the x < 0 lobe
    labels, _ = fit_mixture(E, 2, np.random.default_rng(0), min_members=3)
    assert np.all(labels == 0)


def test_allocate():
    """m w_k rounded by the largest remainders; a component below three members is emptied ((8, 2) -> (10, 0))."""
    for w, counts in (((0.5, 0.5), (5, 5)), ((0.7, 0.3), (7, 3)), ((0.95, 0.05), (10, 0)), ((0.75, 0.25), (10, 0))):
        assert tuple(allocate(np.array(w), 10, 3)) == counts, w


def test_redistribute_keeps_columns_and_draws_in_the_span():
    rng = np.random.default_rng(0)
    A, B = rng.standard_normal((4, 5)), rng.standard_normal((4, 5)) + 10
    cols = [np.array([0, 2, 4, 6, 8]), np.array([1, 3, 5, 7, 9])]
    out = redistribute([A, B], cols, np.array([5, 5]), rng)
    assert np.array_equal(out[:, cols[0]], A) and np.array_equal(out[:, cols[1]], B)
    out = redistribute([A, B], cols, np.array([7, 3]), rng)
    kept_B = [j for j in cols[1] if any(np.array_equal(out[:, j], b) for b in B.T)]
    assert len(kept_B) == 3
    new = [j for j in cols[1] if j not in kept_B]
    dev = out[:, new] - A.mean(1, keepdims=True)                 # the draws lie in the span of A's deviations
    Ad = A - A.mean(1, keepdims=True)
    coef = np.linalg.lstsq(Ad, dev, rcond=None)[0]
    assert np.allclose(Ad @ coef, dev)


def test_one_component_is_the_gaussian_analysis():
    E = lobe(20, seed=3)
    d, Cdd = np.array([25.]), np.eye(1) * 4.
    a, b = filt(E, [2]), filt(E, [2], mixture=2)
    Aa = a.analysis_step(d, Cdd, return_analysis=True)
    Ab = b.analysis_step(d, Cdd, return_analysis=True)
    assert np.array_equal(Aa, Ab)
    assert b.mixture_history.counts[-1].tolist() == [20]


def test_mirror_components_from_z_alone():
    """Equal evidence: each component is analysed as by a filter with its members alone, and the analysis of the
    x < 0 component is the mirror image of the x > 0 one; a single Gaussian moves the members towards x = 0."""
    E = mirror_ensemble(20)
    d, Cdd = np.array([22.]), np.eye(1) * 4.
    mix = filt(E, [2], mixture=2)
    Aa = mix.analysis_step(d, Cdd, return_analysis=True)
    assert mix.mixture_history.counts[-1].tolist() == [10, 10]
    assert np.allclose(mix.mixture_history.weights_a[-1], 0.5)
    for cols in (slice(0, None, 2), slice(1, None, 2)):
        alone = filt(E[:, cols], [2]).analysis_step(d, Cdd, return_analysis=True)
        assert np.allclose(Aa[:, cols], alone)
    assert np.allclose(Aa[:, 1::2], S * Aa[:, 0::2])
    assert np.all(Aa[0, 0::2] > 3) and np.all(Aa[0, 1::2] < -3)     # every member stays in its lobe
    gauss = filt(E, [2]).analysis_step(d, Cdd, return_analysis=True)
    assert np.mean(np.abs(gauss[0])) < np.mean(np.abs(Aa[0]))


def test_data_of_x_select_one_component():
    E = mirror_ensemble(20)
    mix = filt(E, [0], mixture=2)
    Aa = mix.analysis_step(np.array([8.]), np.eye(1), return_analysis=True)
    assert mix.mixture_history.counts[-1].tolist() == [20, 0]
    assert mix.mixture_history.weights_a[-1][0] > 1 - 1e-6
    assert np.all(Aa[0] > 3)


def test_mixture_keyword(monkeypatch):
    """mixture= takes a number of components; in a build without the mixture analysis (the public one strips it), it
    is an unknown keyword: the filter raises instead of running one Gaussian silently."""
    from romda.estimators.sequential.ensemble import EnsembleEstimator

    for bad in (0, 2.5, -1):
        with pytest.raises(ValueError, match='mixture'):
            filt(lobe(10), [2], mixture=bad)
    monkeypatch.delattr(EnsembleEstimator, 'mixture')
    with pytest.raises(TypeError, match="unexpected keyword arguments: \\['mixture'\\]"):
        EnSRKF(parent_model=Lorenz63(dt=0.01), m=10, std_phi=0.2, mixture=2)


def test_mixture_with_the_stochastic_filter_and_rejection():
    from romda.estimators import EnKF
    E = mirror_ensemble(20)
    est = EnKF(parent_model=Lorenz63, dt=DT, m=20, ensemble_psi0=E, observe_dims=[2], mixture=2,
               max_mean_shift=1.0, climatology=np.array([7.9, 9.0, 8.6]))
    Aa = est.analysis_step(np.array([22.]), np.eye(1) * 4., return_analysis=True)
    assert Aa.shape == (3, 20) and np.all(np.isfinite(Aa))
    assert len(est.mixture_history.times) == 1
