"""State rejection (`max_mean_shift`): an analysis that moves the ensemble mean of the physical state by more than
`max_mean_shift` climatological standard deviations is rejected, and the forecast is kept."""

import numpy as np
from romda.estimators import EnSRKF
from romda.models.physical import Lorenz63


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


def test_far_analysis_rejected_and_forecast_kept():
    ens, Af, Aa = _one_analysis(100., max_mean_shift=1.0)
    assert len(ens.rejected_analysis.times) == 1
    np.testing.assert_array_equal(Aa, Af)


def test_near_analysis_accepted():
    ens, Af, Aa = _one_analysis(0.5, max_mean_shift=1.0)
    assert not ens.rejected_analysis.times
    assert not np.allclose(Aa, Af)


def test_off_by_default():
    ens, Af, Aa = _one_analysis(100.)
    assert not ens.rejected_analysis.times
    assert not np.allclose(Aa, Af)


def test_climatology_from_a_free_run_of_the_model():
    ens = EnSRKF(parent_model=Lorenz63(dt=0.01), m=12, std_phi=0.5, max_mean_shift=1.0)
    sd = ens.model_climatology()
    ens.close()
    np.testing.assert_allclose(sd, [7.9, 9.0, 8.6], rtol=0.15)   # the Lorenz-63 attractor (sigma 10, rho 28, beta 8/3)


def test_given_climatology_is_used():
    ens, Af, Aa = _one_analysis(100., max_mean_shift=1.0, climatology=[1e3, 1e3, 1e3])
    assert not ens.rejected_analysis.times                  # 100 is a tenth of this climatology
