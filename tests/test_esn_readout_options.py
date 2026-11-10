"""The ensemble options of `ESN_model` that the washout-free ESN-DA paper employs: the read-out singular values in
logarithms (`Wout_log`), the read-out corrections along fixed directions (`Wout_dirs`, ``est_alpha=['Wdir']``) and
the per-member reservoir hyperparameters (``est_alpha=['sigma_in', 'rho']``).

Each test trains a small network on a Lorenz-63 record without a hyperparameter search (a fraction of a second).
"""

import numpy as np
import pytest
from romda.models.physical import Lorenz63

N_UNITS, M = 30, 4


DATA = Lorenz63(dt=0.01).time_integrate(Nt=3000)[0][None, :, :, 0]     # (1, Nt, 3)


def _esn(**kw):
    from romda.models.data_driven import ESN_model
    return ESN_model(dt=0.01, data=DATA, plot_training=False, N_units=N_UNITS, N_wash=10, t_train=20.0, t_val=5.0,
                     t_test=0.0, hyperparameters_to_optimize=[], rho=0.9, sigma_in=0.1, tikh=1e-8, seed=0, **kw)


def _forecast(esn, Nt=20):
    psi, _ = esn.time_step(Nt=Nt)
    return psi[:, :esn.N_dim]                  # (Nt + 1, N_dim, m): the outputs u


def _set_alpha(esn, name, values):
    """Overwrite the row of the estimated parameter `name` in the current augmented state."""
    psi = esn.current_state.copy()
    psi[-len(esn.est_alpha) + esn.est_alpha.index(name)] = values
    esn.update_history(psi[None], t=esn.current_time, reset=True)


def test_logarithmic_singular_values_read_out_the_same_network():
    # Wout_log: the augmented state carries log(s_i). With no spread the members read out with the trained
    # singular values, as with Wout_log = False; with a spread, every member's s_i is positive and within the
    # log-uniform band s_i exp([-std, std]).
    log, lin = _esn(Wout_log=True), _esn(Wout_svd=True)       # est_alpha=['Wout'] switches Wout_svd on too
    for esn in (log, lin):
        esn.init_ensemble(m=M, est_alpha=['Wout'], std_alpha=0.0)
    s = log.Wout_Sigma0
    assert np.allclose(np.asarray(log.current_state)[-3:], np.log(s)[:, None])
    assert np.allclose(_forecast(log), _forecast(lin))

    band = _esn(Wout_log=True)
    band.init_ensemble(m=M, est_alpha=['Wout'], std_alpha=0.5)
    vals = np.exp(np.asarray(band.current_state)[-3:].reshape(3, M))
    assert np.all(vals > 0) and np.all(vals >= s[:, None] * np.exp(-0.5) - 1e-12) \
        and np.all(vals <= s[:, None] * np.exp(0.5) + 1e-12)


def test_readout_directions_add_their_coordinates_times_the_projected_reservoir():
    # est_alpha=['Wdir']: member j reads out with Wout + Wout_dirs C_j. With zero coordinates the forecast is
    # the plain network's; with coordinates C, the first output step differs by C (Wout_dirs^T [r; b_out]),
    # because the reservoir update of that step does not depend on the read-out.
    k = 2
    dirs = np.random.default_rng(1).standard_normal((N_UNITS + 1, k))
    wdir, plain = _esn(Wout_dirs=dirs, Wout_dir_std=np.ones((k, 3))), _esn()
    wdir.init_ensemble(m=M, est_alpha=['Wdir'], std_alpha=0.0)
    plain.init_ensemble(m=M)
    assert wdir.est_alpha == [f'wdir_{i}_{q}' for i in range(k) for q in range(3)]
    assert np.allclose(_forecast(wdir), _forecast(plain))

    wdir, plain = _esn(Wout_dirs=dirs, Wout_dir_std=np.ones((k, 3))), _esn()
    wdir.init_ensemble(m=M, est_alpha=['Wdir'], std_alpha=0.0)
    plain.init_ensemble(m=M)
    C = np.random.default_rng(2).standard_normal((k, 3, M))
    for i in range(k):
        for q in range(3):
            _set_alpha(wdir, f'wdir_{i}_{q}', C[i, q])
    psi_w, _ = wdir.time_step(Nt=1)
    psi_p, _ = plain.time_step(Nt=1)
    r1 = psi_p[1, 3:3 + N_UNITS]                                         # (N_units, m), the same in both
    assert np.allclose(psi_w[1, 3:3 + N_UNITS], r1)
    proj = dirs.T @ np.vstack([r1, plain.bias_out * np.ones((1, M))])    # (k, m)
    assert np.allclose(psi_w[1, :3] - psi_p[1, :3], np.einsum('kqm,km->qm', C, proj))


@pytest.mark.parametrize('names', [['sigma_in'], ['rho'], ['sigma_in', 'rho']])
def test_per_member_reservoir_hyperparameters(names):
    # est_alpha=['sigma_in', 'rho']: each member steps with its own value; the nominal attributes are restored
    # after the forecast. Member j forecasts as an ensemble in which every member carries member j's values.
    esn = _esn()
    esn.init_ensemble(m=M, est_alpha=names, std_alpha=0.05)
    nominal = {a: getattr(esn, a) for a in names}
    values = {a: np.asarray(esn.current_state)[-len(names) + names.index(a)].ravel().copy() for a in names}
    assert all(np.ptp(v) > 0 for v in values.values())
    u = _forecast(esn)
    assert all(getattr(esn, a) == v for a, v in nominal.items())

    j = 2
    same = _esn()
    same.init_ensemble(m=M, est_alpha=names, std_alpha=0.05)
    for a in names:
        _set_alpha(same, a, np.full(M, values[a][j]))
    assert np.allclose(_forecast(same)[:, :, j], u[:, :, j])
    assert not np.allclose(u[:, :, 0], u[:, :, j])
