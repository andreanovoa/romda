"""The sequential data assimilation driver shared by every entry point.

`run_da_loop` alternates ``forecast_step`` / ``analysis_step`` over the
observations of a truth record. It is duck-typed on purpose: it accepts any
`romda.estimators.Estimator` — all it needs is ``copy()``,
``forecast_step(t_end, close=...)``, ``analysis_step(d, Cdd)`` and
``model.t_CR``.
"""

import gc
import types

import numpy as np
from romda.estimators import EnSRKF
from romda.metrics import state_metrics
from romda.observations import Observations, measured_idx, observation_covariance, std_fraction_covariance


def _legacy_observation_covariance(truth, std_obs):
    """The historical default ``Cdd`` of the mains (variance ``std_obs * max|y|^2``).

    Kept verbatim from the former ``scripts/mains/common.observation_covariance`` so that
    calling `run_da_loop` without an explicit ``Cdd`` behaves exactly as the
    published sweeps (now ``scripts/*/sweeps_*.py``) did. New work should pass
    ``Cdd=romda.observations.observation_covariance(truth, std_obs)`` — the
    corrected ``(std_obs * max|y|)^2`` variant — explicitly.
    """
    Nq = truth.y_obs.shape[1]
    return np.diag(std_obs * np.ones(Nq)) * np.max(abs(truth.y_obs), axis=0) ** 2


def _washout_bias(filter_ens, truth):
    """Forecast to the first observation and synchronize the bias network with the innovations of the
    `N_wash` network steps before it: the raw data minus the ensemble mean of the model observables."""
    bias = filter_ens.bias
    if bias is None or not hasattr(bias, 'washout'):
        raise ValueError('washout=True synchronizes the network of an ESN bias; this estimator has '
                         f'{"no bias" if bias is None else type(bias).__name__}.')
    step, n_wash = int(bias.upsample), int(bias.forecaster.N_wash)
    first = int(truth.obs_idx[0])
    if first < step * n_wash:
        raise ValueError(f'washout=True needs {step * n_wash} steps of data before the first observation; the '
                         f'record has {first}.')
    filter_ens.forecast_step(t_end=truth.t_obs[0])
    q = np.mean(filter_ens.model.get_observable_hist(Nt=step * n_wash + 1), axis=-1)[:-1:step]     # (N_wash, Nq)
    data = truth.y_raw[first - step * n_wash:first:step, :, 0]
    bias.washout(data - q)


def run_da_loop(ensemble,
                truth,
                std_obs: float = 0.1,
                Cdd: np.ndarray = None,
                t_extra: float = None,
                close: bool = True,
                washout: bool = False,
                after_analysis=None):
    """Run sequential data assimilation over all observations in ``truth``.

    Parameters
    ----------
    ensemble : Estimator or romda.legacy.ensemble.Ensemble
        Initialized ensemble (copied, so the input is left untouched).
    truth : Observations
        Provides ``y_obs`` and ``t_obs``.
    std_obs : float
        Fractional observation error used to build ``Cdd`` when not provided
        (legacy variance semantics, see `_legacy_observation_covariance`).
    Cdd : np.ndarray, optional
        Observation error covariance. Overrides ``std_obs`` if given.
    t_extra : float, optional
        Extra forecast time beyond the last observation (default ``10 t_CR``).
    close : bool
        Close the model's multiprocessing pool after the final forecast.
    washout : bool
        Synchronize the network of an ESN bias with the data of the `N_wash` network steps before the first
        observation (`ESN_bias.washout`), as Nóvoa et al. (2024) do. The default start of the network, from
        states of its validation data, needs no data before the assimilation.
    after_analysis : callable, optional
        Called as ``after_analysis(filtered_ensemble)`` after every analysis, e.g. to bound a parameter of the
        analysed members in place (``filtered_ensemble.model.current_state``).

    Returns
    -------
    Estimator or Ensemble
        The filtered ensemble with full model (and bias) histories.
    """
    filter_ens = ensemble.copy()

    if Cdd is None:
        Cdd = _legacy_observation_covariance(truth, std_obs)

    if washout:
        _washout_bias(filter_ens, truth)

    for d, t_d in zip(truth.y_obs, truth.t_obs):
        filter_ens.forecast_step(t_end=t_d)
        filter_ens.analysis_step(d=d, Cdd=Cdd.copy())
        if after_analysis is not None:
            after_analysis(filter_ens)

    if t_extra is None:
        t_extra = 10. * filter_ens.model.t_CR

    filter_ens.forecast_step(t_end=truth.t_obs[-1] + t_extra, close=close)
    return filter_ens


def assimilate(ensemble, truth, std_obs=0.05, cdd_fraction=None, **kwargs):
    """`run_da_loop` with the corrected `Cdd` (`observation_covariance`), or, with `cdd_fraction` = f (the
    washout-free ESN-DA paper: f = 0.2), `Cdd` with f times the standard deviation of each observable over the truth
    record (`std_fraction_covariance`)."""
    if cdd_fraction:
        return run_da_loop(ensemble, truth, Cdd=std_fraction_covariance(truth.y_true, float(cdd_fraction)), **kwargs)
    return run_da_loop(ensemble, truth, Cdd=observation_covariance(truth, std_obs), **kwargs)


def filter_record(ensemble, d, steps, Cdd, after_analysis=None):
    """`run_da_loop` of the data `d` (K, Nq) at `steps` (K,) model steps after the current time, with no forecast
    after the last datum: the filtered ensemble and its history from the current time, the forecast between the
    data and the analysis at each datum."""
    start = ensemble.model.hist.shape[0] - 1
    truth = types.SimpleNamespace(t_obs=ensemble.model.current_time + np.asarray(steps) * ensemble.model.dt, y_obs=d)
    ensemble = run_da_loop(ensemble, truth, Cdd=Cdd, t_extra=0., close=False, after_analysis=after_analysis)
    return ensemble, ensemble.model.hist[start:]


def as_result(ensemble):
    """A `deploy`-shaped result from an ensemble estimator that was run some other way.

    Lets the free-running and washout ablations be scored by `state_metrics` /
    `state_error`, i.e. on the same full-state footing as either deployment.
    """
    return dict(t=ensemble.model.hist_t,
                y=ensemble.model.hist[:, :ensemble.model.N_dim].mean(axis=-1),
                ensemble=ensemble)


def inflation_metrics(ensemble):
    """Mean, first and last inflation factor applied at the analyses, and the number
    of rejected analyses.

    `inflation_history` logs only the factors that differ from 1, so an empty
    history is a factor of 1."""
    rho = np.asarray(ensemble.inflation_history.factors, dtype=float)
    if not rho.size:
        rho = np.ones(1)
    return dict(rho_mean=float(rho.mean()), rho_first=float(rho[0]), rho_last=float(rho[-1]),
                n_rejected=len(ensemble.rejected_analysis.times))


def deploy(esn, truth, measured=None, bayesian_update=True,
           m=64, da_method=EnSRKF, std_obs=0.05, t_extra=0., cdd_fraction=None,
           open_loop_ensemble=False, **kwargs):
    """Run whichever deployment `bayesian_update` selects, with one return shape.

    `truth.y_obs` must hold exactly the `measured` components, in that order; extra
    kwargs go to the estimator/`init_ensemble`. ``inflation`` is a factor, ``'adaptive'``, ``'innovation'`` or a
    dict of the inflation keywords of the estimator; `cdd_fraction` selects the `Cdd` of `assimilate`. ``open_loop_ensemble=True`` runs the same `m`
    members as the Bayesian branch with substitution instead of analysis
    (trajectories under ``y_members``). Returns ``dict(t, y, ensemble)`` — `y` is
    the full-state estimate (ensemble mean, or the pre-substitution output in open
    loop), `ensemble` is ``None`` in the open-loop branch."""
    # the ensembles of earlier deployments sit in reference cycles, which the incremental collector of Python 3.14
    # frees too late: a pool worker of run_partial_table.py grew by 0.25 GB per job (m = 20) and filled the memory
    gc.collect()
    if isinstance(da_method, str):
        from romda.run import DA_METHODS  # romda.run imports this module
        da_method = DA_METHODS[da_method]
    if bayesian_update:
        # `adaptive_inflation` is the harness's own on/off axis (the fixed factor stays in the record as
        # the reference); the estimator takes inflation='adaptive'.
        inflation = kwargs.pop('inflation', 1.0)
        if kwargs.pop('adaptive_inflation', False):
            inflation = 'adaptive'
        # a dict holds the inflation keywords of the estimator (e.g. inflation='innovation' with its options)
        inflation = dict(inflation) if isinstance(inflation, dict) else dict(inflation=inflation)
        ens = da_method(parent_model=esn, m=m, measured=measured, **inflation, **kwargs)
        return as_result(assimilate(ens, truth, std_obs=std_obs, cdd_fraction=cdd_fraction, t_extra=t_extra))

    idx = list(esn.observed_idx) if measured is None else measured_idx(measured, esn.N_dim)
    t, y = esn.replacement_loop(truth.t_obs, truth.y_obs, idx, n_extra=int(round(t_extra / esn.dt_ESN)),
                                m=m if open_loop_ensemble else 1)
    if y.ndim == 3:
        return dict(t=t, y=y.mean(axis=-1), y_members=y, ensemble=None)
    return dict(t=t, y=y, ensemble=None)


def run_and_score(esn, data, measured, seed=0, da_method='EnSRKF', m=64, Nt_obs=8,
                  std_obs=0.1, inflation=1.0, window_lyap=(2., 22., 4.),
                  seed_shift_lyap=4.0, ddof=0, cdd_fraction=None, **deploy_kwargs):
    """Build the truth window from the held-out record, deploy, and score the full state.

    The experiment body of the twin experiments of an ESN (e.g. the washout-free ESN-DA example); `window_lyap` = (start, stop,
    extra forecast) in Lyapunov times. The seed shifts the window along the record
    by `seed_shift_lyap` Lyapunov times per seed — it deliberately does not reseed
    the ensemble, so only the trajectory varies. `deploy_kwargs` reach the estimator
    (`adaptive_inflation`, `inflate_parameters`, `start_param`, `est_alpha`,
    `std_alpha`, ...); an ensemble deployment adds `inflation_metrics` to the scores."""
    t_start_lyap, t_stop_lyap, t_extra_lyap = window_lyap
    clean, noisy = data['clean_data'], data['noisy_data']
    t_lyap, i_test = data['t_lyap'], data['i_test']

    offset = seed * seed_shift_lyap * t_lyap
    t_truth = data['t'][i_test:] - data['t'][i_test]
    truth_full = clean[i_test:]

    truth = Observations(y_raw=noisy[i_test:][:, measured],
                         y_true=truth_full[:, measured],
                         t_true=t_truth,
                         t_start=(t_start_lyap * t_lyap) + offset,
                         t_stop=(t_stop_lyap * t_lyap) + offset,
                         Nt_obs=Nt_obs)

    result = deploy(esn, truth, measured=measured, m=m, da_method=da_method,
                    std_obs=std_obs, t_extra=t_extra_lyap * t_lyap, cdd_fraction=cdd_fraction,
                    inflation=inflation, **deploy_kwargs)

    metrics = state_metrics(result, truth, t_truth, truth_full, t_lyap=t_lyap, ddof=ddof)
    metrics['n_measured'] = len(measured)
    if result['ensemble'] is not None:
        metrics.update(inflation_metrics(result['ensemble']))
    return metrics
