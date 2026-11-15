"""The ensemble filters of `romda.estimators` (sequential/ensemble.py, base.py and localization.py), as in biasda's
tests: the analysis kernels against their m x m forms, the constructor, the joint parameter scheme, the LETKF, the
options of the analysis (frozen parameters and reservoir, the bias-blind window), state rejection and DA loops on a
physical twin.

The other updates, schemes and options of ``biasda.EnsembleFilter`` live in biasda and raise NotImplementedError here.
"""

import numpy as np
import pytest
from conftest import obs_covariance, vdp_ensemble, vdp_twin
from romda.bias_estimators import ConstantBias, NoBias
from romda.estimators import EnKF, EnsembleFilter, EnSRKF, rBA_EnKF, rEnKF
from romda.models.data_driven import LinearModel
from romda.models.physical import Lorenz63

TYPES = ('stochastic', 'sqrt')
KW = dict(parent_model=Lorenz63, dt=0.015, m=5, std_phi=0.2,
          std_alpha=dict(rho=(25., 35.), beta=(2, 4), sigma=(5, 15)),
          observe_dims=[0, 1, 2])
CDD = np.eye(3) * 0.5


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



def test_analysis_kernels_equal_the_m_by_m_forms():
    # The EnSRKF thin-SVD square root, sqrt(I - S^T C^-1 S), must equal the symmetric square root by
    # eigendecomposition of the m x m matrix, for an ensemble larger than the observation dimension, a correlated
    # Cdd, and the stacked gamma rows (bias-aware); it draws no random numbers.
    for bias_aware in (False, True):
        analyse, Af, S_eff, C_eff, z = _square_root_case(bias_aware)
        m = Af.shape[1]
        Aa, est = analyse()
        mean, dev = Af.mean(1, keepdims=True), Af - Af.mean(1, keepdims=True)
        Cinv = np.linalg.inv((m - 1) * C_eff + S_eff @ S_eff.T)
        ev, evec = np.linalg.eigh(np.eye(m) - S_eff.T @ Cinv @ S_eff)
        reference = mean + dev @ S_eff.T @ Cinv @ z + dev @ (evec * np.sqrt(np.maximum(ev, 0.0))) @ evec.T
        assert np.max(np.abs(Aa - reference)) < 1e-10
        assert np.max(np.abs(Af @ est.member_space_transform() - Aa)) < 1e-10     # the transform, built on demand
        assert np.array_equal(Aa, analyse(seed=2)[0])

    # The EnKF gain form Af + (Af S^T) G equals the m x m transform form, and its member-space transform (what the
    # adaptive inflation applies) reproduces the analysis.
    rng = np.random.default_rng(1)
    est = EnsembleFilter(parent_model=Lorenz63(dt=0.01), m=30, std_phi=0.2, type='stochastic')
    est.rng = np.random.default_rng(5)
    Af = np.vstack((est.current_state, est.model.get_observables()))
    m, Nq, M = Af.shape[1], est.Nq, est.M
    B = rng.standard_normal((Nq, Nq))
    Cdd = B @ B.T / Nq + 0.1 * np.eye(Nq)
    d = rng.standard_normal(Nq)
    Aa = est._analysis_kernel(Af.copy(), d, Cdd)
    S = M @ (Af - Af.mean(1, keepdims=True))
    Cinv = np.linalg.inv((m - 1) * Cdd + S @ S.T)
    D = d[:, None] + np.linalg.cholesky(Cdd) @ np.random.default_rng(5).standard_normal((Nq, m))
    np.testing.assert_allclose(Aa, Af @ (np.eye(m) + S.T @ Cinv @ (D - M @ Af)), atol=1e-10)
    np.testing.assert_allclose(Aa, Af @ est.member_space_transform(), atol=1e-10)
    est.close()


def test_the_constructor_reads_its_options():
    kw = dict(parent_model=Lorenz63(dt=0.01), m=10, std_phi=0.2)
    # Case, hyphens, underscores and spaces are ignored.
    for spelling, type in (('SQRT', 'sqrt'), ('Stochastic', 'stochastic'), ('Sq_Rt', 'sqrt')):
        assert EnsembleFilter(type=spelling, **kw).type == type, spelling
    assert EnsembleFilter(**kw).type == 'stochastic'
    assert EnSRKF(**kw).type == 'sqrt'
    assert EnKF(**kw).type == 'stochastic'
    # A model class takes its constructor keywords.
    for cls in (EnsembleFilter, EnKF, EnSRKF):
        assert cls(parent_model=Lorenz63, m=10, std_phi=0.2, dt=0.02).model.dt == 0.02
    # The options of Estimator, not of the leaf class, reach the filter (they used to be dropped); the named entry
    # point is the square-root type with the parameters estimated jointly.
    e = EnSRKF(inflation=1.01, start_param=3, start_bias=2, **KW)
    assert (e.inflation_factor, e.start_param, e.start_bias) == (1.01, 3, 2)
    assert (e.type, e.parameter_estimation) == ('sqrt', 'joint')
    e.close()
    e = EnsembleFilter(type='sqrt', parameter_estimation='joint', **KW)
    assert (e.type, e.parameter_estimation) == ('sqrt', 'joint')
    e.close()
    e = rBA_EnKF(parent_bias=ConstantBias, **KW)           # the paper's name of rEnKF, gamma = 1 by default
    assert isinstance(e, rEnKF) and e.gamma == 1.0 and e.type == 'stochastic'
    e.close()


def test_the_constructor_refuses_what_it_does_not_take():
    kw = dict(parent_model=Lorenz63(dt=0.01), m=10, std_phi=0.2)
    for alias in ('sqrt-symmetric', 'square-root', 'ETKF', 'SEIK', 'EnSRF', 'EAKF', 'DEnKF', 'PertObs'):
        with pytest.raises(ValueError):
            EnsembleFilter(type=alias, **kw)
    with pytest.raises(ValueError):
        EnsembleFilter(**KW)                       # est_alpha set, no scheme
    # The other updates, the dual scheme, localization and the bias-estimation routes live in biasda.
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
    # A model instance takes no constructor keywords: a keyword that neither the filter nor init_ensemble takes
    # raises, as the Kalman filters do, instead of being dropped with the copy of the model.
    for cls in (EnsembleFilter, EnKF, EnSRKF):
        with pytest.raises(TypeError, match='unexpected keyword'):
            cls(parent_model=Lorenz63(dt=0.01), m=10, std_phi=0.2, inflaton=1.05)       # misspelt
        with pytest.raises(TypeError, match='unexpected keyword'):
            cls(parent_model=Lorenz63(dt=0.01), m=10, std_phi=0.2, dt=0.02)            # a constructor keyword
    # One spelling per option, as in biasda: the stored attributes are not keywords.
    for bad in (dict(inflation_factor=1.01), dict(regularization_factor=2.0)):
        with pytest.raises(TypeError):
            EnSRKF(**bad, **KW)
    for cls in (rEnKF, rBA_EnKF):
        with pytest.raises(TypeError):
            cls(parent_bias=ConstantBias, gamma=1.0, **KW)


def test_every_type_runs_the_joint_scheme():
    for type in TYPES:
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


def test_filter_pickles_after_an_analysis():
    """The lazy member-space transform is a closure; pickling the filter must drop it, not fail."""
    import pickle

    for kind in TYPES:
        analyse, *_ = _square_root_case(bias_aware=False)
        _, est = analyse()
        if kind == 'stochastic':
            est.type = 'stochastic'
            est._analysis_kernel(np.vstack((est.current_state, est.model.get_observables())),
                                 np.zeros(est.Nq), np.eye(est.Nq))
        assert callable(est._analysis_transform)
        back = pickle.loads(pickle.dumps(est))
        assert back.member_space_transform() is None


def test_letkf_with_a_wide_taper_is_the_global_square_root_filter():
    # Gaspari-Cohn is one at zero distance: with a half-radius far beyond the domain, every row of the LETKF
    # (Hunt et al. 2007) takes the transform of the global ensemble transform filter.
    rng = np.random.default_rng(0)
    E0 = np.array([1.0, 1.0, 20.0])[:, None] + rng.standard_normal((3, 12))
    out = []
    for loc in (dict(), dict(localization=1e6)):
        est = EnSRKF(parent_model=Lorenz63(dt=0.01), m=12, ensemble_psi0=E0, **loc)
        est.forecast_step(Nt=10)
        est.analysis_step(d=np.array([1.0, 2.0, 21.0]), Cdd=0.1 * np.eye(3))
        out.append(est.model.current_state.copy())
    assert np.allclose(out[0], out[1], atol=1e-10)


def test_letkf_leaves_the_rows_beyond_the_taper_at_the_forecast():
    # Lorenz-63 components at positions 0, 1, 2; one observation of x at 0 with a half-radius of 0.5: z, two units
    # away (beyond 2 x 0.5), keeps its forecast, and x moves. A row with no position (nan, as a reservoir unit fed by
    # the bias alone) takes the global update.
    rng = np.random.default_rng(1)
    E0 = np.array([1.0, 1.0, 20.0])[:, None] + rng.standard_normal((3, 12))
    for positions, moves in ((None, 0), ([0.0, np.nan, 2.0], 1)):
        est = EnSRKF(parent_model=Lorenz63(dt=0.01), m=12, ensemble_psi0=E0, localization=0.5, obs_positions=[0.0],
                     **(dict(state_positions=positions) if positions else {}))
        est.forecast_step(Nt=10)
        forecast = est.model.current_state.copy()
        est.analysis_step(d=np.array([5.0]), Cdd=0.1 * np.eye(1))
        analysis = est.model.current_state
        assert np.allclose(analysis[2], forecast[2]) and not np.allclose(analysis[moves], forecast[moves])
        if positions:
            L_xy = est._loc_matrices(4, 1)[0]
            assert L_xy[1, 0] == 1.0 and L_xy[2, 0] == 0.0


def _letkf_rows(psi_f_m, Psi_f, S_eff, z, Cdd_eff, L_xy, Nq):
    """The LETKF of biasda, one transform per row (the reference of the grouped computation)."""
    m = Psi_f.shape[1]
    sqrt_rho = np.sqrt(np.tile(L_xy, (1, S_eff.shape[0] // Nq)))
    Cdd_inv, ones_m = np.linalg.inv(Cdd_eff), np.full((m, m), 1.0 / m)
    T = np.empty((Psi_f.shape[0], m, m))
    for i, r in enumerate(sqrt_rho):
        R_inv = r[:, None] * Cdd_inv * r[None, :]
        ev, V = np.linalg.eigh((m - 1) * np.eye(m) + S_eff.T @ R_inv @ S_eff)
        w = V @ ((V.T @ (S_eff.T @ R_inv @ z)) / ev[:, None])
        T[i] = ones_m + (np.eye(m) - ones_m) @ (w + V @ np.diag(np.sqrt((m - 1) / ev)) @ V.T)
    return np.einsum('im,imn->in', psi_f_m + Psi_f, T)


def test_letkf_computes_one_transform_per_distinct_taper_and_matches_every_row():
    # x and y at site 0, z at site 2: x and y (and their observables) share the taper, so two transforms serve the
    # six rows; the analysis is bit for bit that of one transform per row.
    rng = np.random.default_rng(3)
    E0 = np.array([1.0, 1.0, 20.0])[:, None] + rng.standard_normal((3, 12))
    est = EnSRKF(parent_model=Lorenz63(dt=0.01), m=12, ensemble_psi0=E0, localization=1.0,
                 state_positions=[0.0, 0.0, 2.0], obs_positions=[0.0, 0.0, 2.0])
    seen = {}
    local = est._local_transform

    def record(*args):
        seen['args'] = args
        return local(*args)
    est._local_transform = record
    est.forecast_step(Nt=10)
    est.analysis_step(d=np.array([2.0, 2.0, 21.0]), Cdd=0.1 * np.eye(3))
    psi_f_m, Psi_f, S_eff, z, Cdd_eff, Nq = seen['args']
    L_xy = est._loc_matrices(Psi_f.shape[0], Nq)[0]
    assert len(est._row_transforms) == len(np.unique(L_xy, axis=0)) < Psi_f.shape[0]
    assert est._row_index[0] == est._row_index[1] != est._row_index[2]
    np.testing.assert_array_equal(local(psi_f_m, Psi_f, S_eff, z, Cdd_eff, Nq),
                                  _letkf_rows(psi_f_m, Psi_f, S_eff, z, Cdd_eff, L_xy, Nq))


def test_frozen_parameters_survive_the_analysis():
    e = EnSRKF(activate_parameter_estimation=False, **KW)
    alpha0 = e.current_state[-e.Na:, :].copy()
    e.forecast_step(t_end=0.15)
    alpha_f = e.current_state[-e.Na:, :].copy()

    Aa = e.analysis_step(d=e.model.get_observables().mean(-1), Cdd=CDD, return_analysis=True)
    e.model.close()

    assert Aa.shape[0] == e.Nphi + e.Na, 'analysis must carry the full state'
    np.testing.assert_allclose(e.current_state[-e.Na:, :], alpha_f)
    np.testing.assert_allclose(alpha_f, alpha0)          # parameters are static in the forecast
    assert not np.allclose(e.current_state[:e.Nphi, :], Aa[:e.Nphi, :] * 0), 'state was updated'

    # start_param freezes the parameters for its first analyses, then releases them
    e = EnSRKF(start_param=2, **KW)
    alpha0 = e.current_state[-e.Na:, :].copy()
    for k in range(3):
        e.forecast_step(t_end=round(0.15 * (k + 1), 6))
        e.analysis_step(d=e.model.get_observables().mean(-1), Cdd=CDD)
        frozen = bool(np.allclose(e.current_state[-e.Na:, :], alpha0))
        assert frozen == (k < 2), f'analysis {k}: expected frozen={k < 2}'
    e.model.close()


def test_update_reservoir_false_freezes_the_reservoir_rows():
    """`update_reservoir=False` leaves the 'reservoir' rows exactly at their forecast value; only the physical rows
    move. Lorenz63 has no real reservoir: N_dim/N_units are monkeypatched onto the model to exercise the row split
    that `analysis_step` applies for an ESN. A parallel ESN carries `N_r` = patches x `N_units` reservoir rows: all
    of them stay at the forecast."""
    for parallel in (False, True):
        e = EnSRKF(update_reservoir=False, **KW)
        e.model.N_dim, e.model.N_units = 1, (1 if parallel else e.Nphi - 1)
        if parallel:
            e.model.N_r = e.Nphi - 1
        e.forecast_step(t_end=0.15)
        forecast = e.current_state.copy()
        e.analysis_step(d=e.model.get_observables().mean(-1), Cdd=CDD)
        np.testing.assert_allclose(e.current_state[1:e.Nphi, :], forecast[1:e.Nphi, :])
        assert not np.allclose(e.current_state[:1, :], forecast[:1, :])
        e.model.close()


def test_start_bias_blind_window_uses_plain_enkf():
    e = rEnKF(parent_bias=ConstantBias, bias_regularization=1.0, start_bias=1, **KW)
    e.forecast_step(t_end=0.15)
    Af = np.vstack((e.current_state, e.model.get_observables()))
    d = e.model.get_observables().mean(-1)

    e.rng = np.random.default_rng(7)
    Aa_blind = e._analysis_kernel(Af, d, CDD)       # 0 analyses done < start_bias
    e.rng = np.random.default_rng(7)
    Aa_enkf = e._plain_kernel(Af, d, CDD)
    np.testing.assert_allclose(Aa_blind, Aa_enkf)   # blind window == plain EnKF

    e.assimilated_data = (d, 0.15)                  # window over
    e.rng = np.random.default_rng(7)
    Aa_aware = e._analysis_kernel(Af, d, CDD)
    assert not np.allclose(Aa_aware, Aa_enkf)       # bias-aware update kicked in
    e.model.close()


def test_bias_attached_after_construction_is_checked_and_regularized():
    """romda builds the ESN bias from the estimator's model, so the bias may arrive after
    construction; the regularized update then needs it by the first analysis."""
    e = EnSRKF(bias_regularization=1.0, **KW)
    assert not e.is_bias_aware
    e.forecast_step(t_end=0.15)
    d = e.model.get_observables().mean(-1)
    with pytest.raises(ValueError):
        e.analysis_step(d=d, Cdd=CDD)
    e.bias = ConstantBias(innovation=np.zeros(3), t=e.current_time, dt=e.model.dt)
    assert e.is_bias_aware
    e.analysis_step(d=d, Cdd=CDD)
    assert len(e.assimilated_data.times) == 1
    # the analysis resets the bias to the innovation of the analysed observables
    np.testing.assert_allclose(e.bias.current_bias.ravel(), d - e.model.get_observables().mean(-1), atol=1e-12)
    with pytest.raises(ValueError):
        e.bias = 'not a bias'
    e.model.close()


# ── DA loops on a Van der Pol twin: an analysis moves the ensemble-mean observables toward the data (exactly, per
# analysis, for the deterministic EnSRKF; on average for the stochastic EnKF, whose perturbed-observation mean carries
# an O(1/sqrt(m)) sampling error), and the estimated parameters stay within `alpha_lims` or the analysis is rejected.

@pytest.fixture(scope='module')
def truth_unbiased():
    return vdp_twin()


@pytest.fixture(scope='module')
def truth_biased():
    return vdp_twin('linear', noise_level=0.02)


def test_da_reduces_observation_error(truth_unbiased):
    for da_method in (EnKF, EnSRKF):
        ens = vdp_ensemble(truth_unbiased, da_method=da_method)
        Cdd = obs_covariance(truth_unbiased)
        err_forecast, err_analysis = [], []
        for k, (d, t_d) in enumerate(zip(truth_unbiased.y_obs, truth_unbiased.t_obs)):
            ens.forecast_step(t_end=t_d)
            y_f = np.mean(ens.model.get_observables(), axis=-1)
            err_forecast.append(np.linalg.norm(y_f - d))
            ens.analysis_step(d=d, Cdd=Cdd.copy())
            y_a = np.mean(ens.model.get_observables(), axis=-1)
            err_analysis.append(np.linalg.norm(y_a - d))
            if da_method is EnSRKF:
                # deterministic filter: it assimilates d exactly, so every analysis must
                # move the ensemble-mean observables towards the data
                assert err_analysis[-1] <= err_forecast[-1] + 1e-12
            if k >= 5:
                break
        # The stochastic EnKF assimilates a *perturbed* observation ensemble, whose mean
        # differs from d by a sampling error of order 1/sqrt(m) -- with m=8 an individual
        # analysis can move the mean slightly the wrong way. The guarantee is on average.
        assert np.mean(err_analysis) <= np.mean(err_forecast)
        ens.model.close()
        assert np.isfinite(ens.model.hist).all()
        assert len(ens.assimilated_data.times) == 6
        alpha = ens.model.hist[-1, -ens.Na:, :]
        lo, hi = ens.model.alpha_lims['zeta']
        assert np.all((alpha >= lo) & (alpha <= hi)) or len(ens.rejected_analysis.times) > 0


def test_bias_aware_loop_runs(truth_biased):
    for bias_cls in (NoBias, ConstantBias):
        ens = vdp_ensemble(truth_biased, da_method=rEnKF, bias=bias_cls, bias_regularization=1.)
        Cdd = obs_covariance(truth_biased)
        for d, t_d in list(zip(truth_biased.y_obs, truth_biased.t_obs))[:6]:
            ens.forecast_step(t_end=t_d)
            ens.analysis_step(d=d, Cdd=Cdd.copy())
        ens.model.close()
        assert np.isfinite(ens.model.hist).all()
        assert np.isfinite(ens.bias.hist).all()
        # model and bias histories stay synchronized in time
        assert abs(ens.bias.current_time - ens.model.current_time) < ens.model.dt
        if bias_cls is NoBias:
            np.testing.assert_allclose(ens.bias.hist, 0.)


def test_renkf_at_zero_bias_equals_the_enkf(truth_unbiased):
    # With zero bias and zero Jacobian the regularized bias-aware gain must reduce exactly
    # to the EnKF gain (Novoa et al. 2023, erratum 2024): C_yy = C_dd + C_qq with C_qq the
    # per-sample covariance. Weighting C_dd by (m - 1) instead would inflate the
    # observation noise m - 1 times and divide the effective gamma by m - 1.
    Cdd = obs_covariance(truth_unbiased)
    d, t_d = truth_unbiased.y_obs[0], truth_unbiased.t_obs[0]
    states = []
    for da_method, kwargs in ((EnKF, {}), (rEnKF, dict(bias=NoBias, bias_regularization=1.0))):
        ens = vdp_ensemble(truth_unbiased, da_method=da_method, **kwargs)
        ens.forecast_step(t_end=t_d)
        ens.rng = np.random.default_rng(7)               # identical observation perturbations
        ens.analysis_step(d=d, Cdd=Cdd.copy())
        states.append(np.array(ens.model.current_state))
        ens.model.close()
    np.testing.assert_allclose(states[1], states[0], atol=1e-10)


# ── State rejection (`max_mean_shift`): an analysis that moves the ensemble mean of the physical state by more than
# `max_mean_shift` climatological standard deviations is rejected, and the forecast is kept.

def _one_analysis(shift, **kw):
    """One forecast of a Lorenz-63 ensemble, then one analysis of data `shift` away from the forecast mean."""
    ens = EnSRKF(parent_model=Lorenz63(dt=0.01), m=12, std_phi=0.5, **kw)
    ens.rng = np.random.default_rng(0)
    ens.forecast_step(t_end=0.5)
    Af = ens.current_state.copy()
    d = ens.model.get_observables().mean(-1) + shift
    ens.analysis_step(d=d, Cdd=0.01 * np.eye(3))
    Aa = ens.current_state.copy()
    ens.close()
    return ens, Af, Aa


def test_far_analysis_is_rejected_and_the_forecast_kept():
    ens, Af, Aa = _one_analysis(100., max_mean_shift=1.0)
    assert len(ens.rejected_analysis.times) == 1
    np.testing.assert_array_equal(Aa, Af)
    for shift, kw in ((0.5, dict(max_mean_shift=1.0)), (100., {}),          # near; off by default
                      (100., dict(max_mean_shift=1.0, climatology=[1e3, 1e3, 1e3]))):   # a tenth of the climatology
        ens, Af, Aa = _one_analysis(shift, **kw)
        assert not ens.rejected_analysis.times
        assert not np.allclose(Aa, Af)


def test_climatology_from_a_free_run_of_the_model():
    ens = EnSRKF(parent_model=Lorenz63(dt=0.01), m=12, std_phi=0.5, max_mean_shift=1.0)
    sd = ens.model_climatology()
    ens.close()
    np.testing.assert_allclose(sd, [7.9, 9.0, 8.6], rtol=0.15)   # the Lorenz-63 attractor (sigma 10, rho 28, beta 8/3)


def test_an_esn_ensemble_follows_the_seed_it_is_given():
    from romda.estimators import EnSRKF
    from romda.models.data_driven import ESN_model
    from romda.models.physical import Lorenz63

    data = Lorenz63(dt=0.01).time_integrate(Nt=3000)[0][None, :, :, 0]
    esn = ESN_model(dt=0.01, data=data, plot_training=False, N_units=30, N_wash=10, t_train=20.0, t_val=5.0,
                    t_test=0.0, hyperparameters_to_optimize=[], rho=0.9, sigma_in=0.1, tikh=1e-8, seed=0)

    def start(**seed):
        return EnSRKF(parent_model=esn, m=4, **seed).model.current_state.copy()

    assert np.array_equal(start(seed=1), start(seed=1))
    assert not np.allclose(start(seed=1), start(seed=2))
    assert np.array_equal(start(), start(seed=0))        # without a seed, the copy keeps the seed of its network
