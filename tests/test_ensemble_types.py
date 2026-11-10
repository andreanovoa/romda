"""The analysis updates of `EnsembleFilter` (``type``) and the joint parameter scheme, as in biasda's tests.

The square-root update is checked against the m x m eigendecomposition, bias-blind and bias-aware. The
other updates, schemes and options of ``biasda.EnsembleFilter`` live in biasda and raise
NotImplementedError here.
"""

import numpy as np
import pytest
from romda.bias_estimators import ConstantBias
from romda.estimators import EnKF, EnsembleFilter, EnSRKF
from romda.models.data_driven import LinearModel
from romda.models.physical import Lorenz63

TYPES = ('stochastic', 'sqrt')


def _square_root_case(bias_aware, n=12, m=50, gamma=0.5):
    """One forecast ensemble of a linear model, analysed by a new square-root filter.

    The observation noise is correlated and, bias-aware, a persistent bias with gamma > 0 stacks the
    pseudo-observation rows (its Jacobian is zero, so they carry the bias alone). Returns
    ``analyse(seed)``, which gives the analysis and the filter, the forecast ensemble, and the mapped
    deviations, the observation-noise covariance and the innovation of the stacked update.
    """
    rng = np.random.default_rng(0)
    B = rng.normal(size=(n, n))
    Cdd = B @ B.T / n + 0.1 * np.eye(n)
    psi0, d = rng.normal(size=n), rng.normal(size=n)
    b0 = rng.normal(size=n)

    def build(seed=1):
        bias = ConstantBias(innovation=b0, t=0.0, dt=1.0) if bias_aware else None
        est = EnsembleFilter(parent_model=LinearModel(F=np.eye(n), psi0=psi0), parent_bias=bias, m=m, std_phi=0.3,
                             type='sqrt', bias_regularization=gamma if bias_aware else None)
        est.rng = np.random.default_rng(seed)
        return est

    est = build()
    Af = np.vstack((est.current_state, est.model.get_observables()))
    mean = Af.mean(1, keepdims=True)
    y, S = est._MA(mean), est._MA(Af - mean)
    S_eff, z, C_eff = S, d[:, None] - y, Cdd
    if bias_aware:
        b, J, d_b = est._bias_terms(y, d)
        assert np.allclose(b.ravel(), b0) and not np.any(J)
        S_eff = np.vstack([(np.eye(n) + J) @ S, np.sqrt(gamma) * (J @ S)])
        z = np.vstack([d_b[:, None] - y - b, -np.sqrt(gamma) * b])
        C_eff = np.kron(np.eye(2), Cdd)

    def analyse(seed=1):
        est = build(seed)
        return est._analysis_kernel(Af.copy(), d, Cdd), est

    return analyse, Af, S_eff, C_eff, z


@pytest.mark.parametrize('bias_aware', [False, True], ids=['bias-blind', 'bias-aware'])
def test_ensrkf_low_rank_square_root_equals_the_eigendecomposition(bias_aware):
    # sqrt(I - S^T C^-1 S) from the thin SVD of the whitened deviations must equal the
    # symmetric square root by eigendecomposition of the m x m matrix, for an ensemble
    # larger than the observation dimension, a correlated Cdd, and the stacked gamma rows.
    analyse, Af, S_eff, C_eff, z = _square_root_case(bias_aware)
    m = Af.shape[1]
    Aa, est = analyse()
    mean, dev = Af.mean(1, keepdims=True), Af - Af.mean(1, keepdims=True)
    Cinv = np.linalg.inv((m - 1) * C_eff + S_eff @ S_eff.T)
    ev, evec = np.linalg.eigh(np.eye(m) - S_eff.T @ Cinv @ S_eff)
    reference = mean + dev @ S_eff.T @ Cinv @ z + dev @ (evec * np.sqrt(np.maximum(ev, 0.0))) @ evec.T
    assert np.max(np.abs(Aa - reference)) < 1e-10
    assert np.max(np.abs(Af @ est.member_space_transform() - Aa)) < 1e-10     # the transform, built on demand
    # The square-root update draws no random numbers.
    assert np.array_equal(Aa, analyse(seed=2)[0])


def test_type_names_and_errors():
    kw = dict(parent_model=Lorenz63(dt=0.01), m=10, std_phi=0.2)
    # Case, hyphens, underscores and spaces are ignored; no other spelling is accepted.
    for spelling, type in (('SQRT', 'sqrt'), ('Stochastic', 'stochastic'), ('Sq_Rt', 'sqrt')):
        assert EnsembleFilter(type=spelling, **kw).type == type, spelling
    for alias in ('sqrt-symmetric', 'square-root', 'ETKF', 'SEIK', 'EnSRF', 'EAKF', 'DEnKF', 'PertObs'):
        with pytest.raises(ValueError):
            EnsembleFilter(type=alias, **kw)
    assert EnsembleFilter(**kw).type == 'stochastic'
    assert EnSRKF(**kw).type == 'sqrt'
    assert EnKF(**kw).type == 'stochastic'


def test_the_options_of_biasda_raise_not_implemented():
    # The other updates, the dual scheme, localization and the bias-estimation routes live in biasda.
    kw = dict(parent_model=Lorenz63(dt=0.01), m=10, std_phi=0.2)
    for type in ('deterministic', 'sqrt-random', 'sqrt-serial', 'sqrt-half', 'Sqrt_Half'):
        for cls in (EnsembleFilter, EnKF, EnSRKF):
            with pytest.raises(NotImplementedError, match='biasda'):
                cls(type=type, **kw)
    with pytest.raises(NotImplementedError, match='biasda'):
        EnsembleFilter(parent_model=Lorenz63, dt=0.01, m=10, std_phi=0.2, std_alpha={'rho': (26.0, 38.0)},
                       parameter_estimation='dual')
    # romda localizes the square-root update without a bias (the LETKF); the rest is biasda's.
    with pytest.raises(NotImplementedError, match='biasda'):
        EnKF(localization=2.0, **kw)
    with pytest.raises(TypeError):
        EnSRKF(loc_radius=2.0, **kw)
    for option in (dict(localization=2.0), dict(bias_estimation='two-stage-analysis'), dict(bias_estimation='lms')):
        for cls in (EnsembleFilter, EnSRKF):
            with pytest.raises(NotImplementedError, match='biasda'):
                cls(parent_bias=ConstantBias, **option, **kw)


@pytest.mark.parametrize('type', TYPES)
def test_every_type_runs_the_joint_scheme(type):
    rng = np.random.default_rng(2)
    Cdd = np.diag([0.04, 0.09, 0.01])
    est = EnsembleFilter(parent_model=Lorenz63, dt=0.01, type=type, m=20, std_phi=0.2,
                         std_alpha={'rho': (26.0, 38.0)}, parameter_estimation='joint', inflation=1.02,
                         parent_bias=ConstantBias, bias_regularization=0.5)
    assert est.Na == 1 and est.est_alpha == ['rho'] and est.parameter_estimation == 'joint'
    rho0 = est.current_state[-1].copy()
    for _ in range(3):
        est.forecast_step(Nt=10)
        est.analysis_step(d=est.model.get_observables().mean(-1) + rng.normal(scale=0.2, size=3), Cdd=Cdd)
    assert np.all(np.isfinite(est.current_state))
    assert not np.allclose(est.current_state[-1], rho0)          # the analysis updates the parameter
    assert len(est.inflation_history.factors) == 3 and len(est.assimilated_data.times) == 3
    est.close()


@pytest.mark.parametrize('kind', TYPES)
def test_filter_pickles_after_an_analysis(kind):
    """The lazy member-space transform is a closure; pickling the filter must drop it, not fail."""
    import pickle

    analyse, *_ = _square_root_case(bias_aware=False)
    _, est = analyse()
    if kind == 'stochastic':
        est.type = 'stochastic'
        est._analysis_kernel(np.vstack((est.current_state, est.model.get_observables())),
                             np.zeros(est.Nq), np.eye(est.Nq))
    assert callable(est._analysis_transform)
    back = pickle.loads(pickle.dumps(est))
    assert back.member_space_transform() is None


