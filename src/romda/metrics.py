"""Scalar and time-resolved error metrics for deployed forecast models.

Full-state scoring of a deployed forecast result against a known truth
record: `valid_time` (time to the threshold of one error series), `state_metrics`
(scalar summary comparable across deployments) and `state_error` (the error
curve behind it). Ported from `dev/ESN-DA/esnda_tools.py` with metric keys
unchanged. Since 8 Oct 2026 the errors are normalized by the deviation of the truth
from its time average (`error_scale`), not by its amplitude: the jsonl records
written before that date keep the amplitude normalization and do not compare
with new ones.
"""

import numpy as np
from romda.utils import interpolate

VALID_THRESHOLD = 0.5   # the normalized error at which a forecast stops being valid (`valid_time`)


def error_scale(y):
    """Normalization of the errors, (<||u - <u>||^2>)^(1/2), of a true record `y` (Nt, N).

    The root-mean-square distance of the true state from its time average <u>, i.e. the
    error of the forecast that returns the time-averaged state, so that a normalized
    error of one means no skill on every system, whatever the mean of its state."""
    y = np.asarray(y, dtype=float)
    return float(np.sqrt(np.mean(np.sum((y - y.mean(axis=0)) ** 2, axis=1))))


def valid_time(err, t, threshold=VALID_THRESHOLD, t_max=None):
    """Valid time of a forecast: the time `t` of the first error of the series `err` above `threshold`, or `t_max`
    (default ``t[-1]``) if the error never exceeds it. A nan error counts as above the threshold: a forecast that
    blows up has no skill.

    Parameters
    ----------
    err : (n,) ndarray
        The error of the forecast, e.g. ``||u - u_true|| / error_scale``.
    t : (n,) ndarray
        The time of each error, from the origin of the valid time (e.g. ``t - t_last``, or ``k / steps_lyap``).
    threshold : float
        The error above which the forecast fails.
    t_max : float, optional
        The valid time of a forecast that never fails (the horizon).
    """
    err = np.asarray(err)
    over = ~(err <= threshold)
    if over.any():
        return t[int(np.argmax(over))]
    return t[-1] if t_max is None else t_max


def forecast_scores(err, t, t_lyap):
    """The scores of one forecast from its error series `err` at the times `t` from the last datum (e.g.
    ``t - t_last``): the first error, the valid time (`valid_time`) in Lyapunov times and the root mean square of the
    error over the first Lyapunov time, ``t <= t_lyap``."""
    err = np.asarray(err)
    window = np.sqrt(np.mean(err[t <= t_lyap + 1e-9] ** 2))
    return float(err[0]), float(valid_time(err, t) / t_lyap), float(window)


def first_below(t, q, level, t_end=np.inf, t_start=-np.inf):
    """First time `t` in [t_start, t_end] at which the series `q` falls below `level` (e.g. the synchronization time
    of an error or a spread, up to the last datum `t_end`); nan if it never does."""
    hit = t[(q < level) & (t >= t_start) & (t <= t_end)]
    return float(hit[0]) if hit.size else np.nan


def forecast_valid_time(pred, target, norm, threshold=0.3, dt=1.0):
    """Valid time of the forecast `pred` (n, N) of `target`: the error is the root mean square over the components of
    ``(pred - target) / norm`` (e.g. `norm` the standard deviation of each component), and the time of step k is
    ``k * dt``, up to ``n * dt`` (`valid_time`). Returns the valid time and the error series."""
    err = np.sqrt((((np.asarray(pred) - target) / norm) ** 2).mean(axis=1))
    return valid_time(err, np.arange(len(err)) * dt, threshold, t_max=len(err) * dt), err


def attractor_kld(y_true, y_model, n_bins=32, eps=1e-6):
    """Kullback-Leibler divergence KL(truth || model) of the statistics of each component, averaged over the
    components, for every member of an ensemble.

    The histograms have `n_bins` bins over the range of the truth: the probability that a member puts outside that
    range is missing from its histogram, so a member that leaves the attractor of the truth, or blows up (non-finite
    values are dropped), scores a large divergence, bounded by about ``log(1 / eps)`` per bin by the floor `eps` added
    to both probabilities.

    Parameters
    ----------
    y_true : (n, N) ndarray
        The true record.
    y_model : (n_model, N, m) ndarray
        The records of the `m` members (or networks), of any length.

    Returns
    -------
    (m,) ndarray
    """
    kld = np.zeros(y_model.shape[-1])
    for d in range(y_true.shape[1]):
        edges = np.histogram_bin_edges(y_true[:, d], bins=n_bins)
        p = np.histogram(y_true[:, d], bins=edges)[0] / len(y_true)
        for j in range(kld.size):
            x = y_model[:, d, j]
            q = np.histogram(x[np.isfinite(x)], bins=edges)[0] / len(y_model)
            kld[j] += np.sum(p * np.log((p + eps) / (q + eps)))
    return kld / y_true.shape[1]


def lyapunov_spectrum_map(step, x0, dt, n, n_transient, n_exp, bound=np.inf, n_bound=None, eps=1e-7, seed=0):
    """Leading `n_exp` Lyapunov exponents, per unit time, of the map `step` (Benettin et al. 1980), for a forecast
    model that has no right-hand side, such as an echo state network in closed loop, or for the time-`dt` map of a
    flow, so that both are estimated by the same procedure.

    `n_exp` perturbations of size `eps`, advanced by `step` with the reference state and re-orthonormalized by QR at
    every step, give the tangent dynamics by finite differences; the exponents are the mean logarithmic stretching over
    `n` steps after `n_transient` steps, a fixed length (no convergence test).

    Parameters
    ----------
    step : callable
        One step of the map, for states as columns, (N_state, k) -> (N_state, k).
    x0 : (N_state,) ndarray
        The reference state at the start.
    dt : float
        The time of one step.
    bound : float
        The exponents are nan if the first `n_bound` components of the reference (default all) leave |x| <= bound,
        e.g. a network that leaves the attractor.

    Returns
    -------
    (n_exp,) ndarray
    """
    x = np.asarray(x0, dtype=float)[:, None].copy()
    n_bound = x.shape[0] if n_bound is None else n_bound
    Q = np.linalg.qr(np.random.default_rng(seed).standard_normal((x.shape[0], n_exp)))[0]
    total = np.zeros(n_exp)
    for k in range(n_transient + n):
        out = step(np.hstack([x, x + eps * Q]))
        x = out[:, :1]
        if np.abs(x[:n_bound]).max() > bound:
            return np.full(n_exp, np.nan)
        Q, R = np.linalg.qr((out[:, 1:] - x) / eps)
        if k >= n_transient:
            total += np.log(np.abs(np.diag(R)))
    return total / (n * dt)


def _check_truth_covers(t, t_true):
    """Refuse to score a model run that extends past the end of the truth record —
    the interpolation would silently extrapolate garbage there."""
    if t[-1] > t_true[-1] + 1e-9:
        raise ValueError(
            f'The model history runs to t = {t[-1]:.2f} but the truth record ends at '
            f't = {t_true[-1]:.2f}: more assimilation cycles (N_da) and/or t_extra than '
            'the available truth covers. Run the truth for longer (hold out more test '
            'data), or reduce N_da / Nt_obs / t_extra.')


def state_metrics(result, truth, t_true, y_true, t_lyap=None, vt_threshold=VALID_THRESHOLD, t_sync=None, ddof=0):
    """Full-state metrics of a `deploy` result, comparable across both deployments.

    Scored on the full clean state, normalized by the truth's RMS deviation from its
    time average over the assimilation window (`error_scale`, per component). `rmse_*`/`std_*` are
    the mean/std over each member's own RMSE; `rmse_mean_*` scores the ensemble-mean
    trajectory and `spread_*` the RMS spread; estimated parameters add
    `alpha_<name>` (+std). `valid_time_*` is the free-forecast valid time (error
    below `vt_threshold` after the last observation, censored at the end of the
    run; in Lyapunov times when `t_lyap` is given).

    The ensemble calibration is scored on the post-synchronization window
    `t_obs[0] + t_sync <= t <= t_obs[-1]` (`t_sync` defaults to `2 * t_lyap`, or to 0
    without `t_lyap`), because the members that leave the attractor before the
    ensemble locks dominate the `*_da` scores. `rmse_mean_sync`/`spread_sync` are
    `rmse_mean_da`/`spread_da` on that window, and `rmse_mean_last`/`spread_last` on the last Lyapunov time
    of the data (`t > t_obs[-1] - t_lyap`; the whole window without `t_lyap`). `spread_skill` is their ratio, spread
    over error, times the finite-ensemble factor sqrt((m+1)/(m-1)) (Fortin et al.
    2014), so that a calibrated ensemble (the truth is statistically one more member)
    gives 1 for every `m`. `coverage_2sigma` is the fraction of (time, component)
    entries with |ensemble mean - truth| <= 2 member std. This band has no
    finite-ensemble factor: a calibrated ensemble gives 0.88 for m = 8, 0.93 for
    m = 20, 0.95 for m = 64, and 0.954 as m -> inf. `rank_outliers` is the fraction of
    entries in which the truth falls outside the ensemble range, i.e. in the two
    extreme bins of the rank histogram (2/(m+1) for a calibrated ensemble).
    `alpha_cover_<name>` is 1.0 when the true parameter is within 2 std of the final
    ensemble mean, and 0.0 otherwise. `ddof` is that of the member std of the spreads and of the 2-sigma band
    (1: the sample std); `spread_skill` does not depend on it."""
    t, y = result['t'], result['y']
    _check_truth_covers(t, t_true)
    t_obs = truth.t_obs
    y_ref = interpolate(t_true, y_true, t)
    win_da = (t >= t_obs[0]) & (t <= t_obs[-1])
    win_fc = t > t_obs[-1]
    if t_sync is None:
        t_sync = 2 * t_lyap if t_lyap else 0.
    win_sync = win_da & (t >= t_obs[0] + t_sync)
    win_last = win_da & (t > t_obs[-1] - (t_lyap or np.inf))         # the last Lyapunov time of the data
    scale = error_scale(y_ref[win_da]) / np.sqrt(y_ref.shape[1])   # per component, as the errors

    def nrmse(w):
        return float(np.sqrt(np.mean((y[w] - y_ref[w]) ** 2)) / scale) if w.any() else np.nan

    metrics = dict(rmse_mean_da=nrmse(win_da), rmse_mean_forecast=nrmse(win_fc),
                   rmse_mean_sync=nrmse(win_sync), rmse_mean_last=nrmse(win_last))

    def first_exceed(err_fc):
        """Valid time of one error series over `win_fc`, clock at the last observation."""
        if not win_fc.any():
            return np.nan
        return float((valid_time(err_fc, t[win_fc], vt_threshold) - t_obs[-1]) / (t_lyap or 1.0))

    err_fc_mean = (np.sqrt(((y - y_ref) ** 2).mean(axis=1)) / scale)[win_fc]
    metrics['valid_time_mean_forecast'] = first_exceed(err_fc_mean)

    ensemble = result.get('ensemble')
    members = result.get('y_members')                          # open-loop ensemble
    if ensemble is not None:
        members = ensemble.model.hist[:, :ensemble.model.N_dim, :]
    if members is None:
        metrics.update(rmse_da=metrics['rmse_mean_da'],
                       rmse_forecast=metrics['rmse_mean_forecast'],
                       valid_time_forecast=metrics['valid_time_mean_forecast'],
                       std_da=np.nan, std_forecast=np.nan, valid_time_std=np.nan,
                       spread_da=np.nan, spread_forecast=np.nan,
                       spread_sync=np.nan, spread_last=np.nan, spread_skill=np.nan,
                       coverage_2sigma=np.nan, rank_outliers=np.nan)
        return metrics

    e = members - y_ref[:, :, None]                            # (Nt, N_dim, m)
    spread = members.std(axis=-1, ddof=ddof)

    def per_member(w):
        if not w.any():
            return np.nan, np.nan
        e_m = np.sqrt(np.mean(e[w] ** 2, axis=(0, 1))) / scale   # (m,) member RMSEs
        return float(e_m.mean()), float(e_m.std())

    def nspread(w):
        return float(np.sqrt(np.mean(spread[w] ** 2)) / scale) if w.any() else np.nan

    metrics['rmse_da'], metrics['std_da'] = per_member(win_da)
    metrics['rmse_forecast'], metrics['std_forecast'] = per_member(win_fc)
    metrics.update(spread_da=nspread(win_da), spread_forecast=nspread(win_fc))

    # calibration after the ensemble locks: spread against error, 2-sigma band, rank extremes
    def fraction(flag):
        return float(flag[win_sync].mean()) if win_sync.any() else np.nan

    rmse_sync, m = metrics['rmse_mean_sync'], members.shape[-1]
    metrics.update(spread_sync=nspread(win_sync), spread_last=nspread(win_last),
                   coverage_2sigma=fraction(np.abs(y - y_ref) <= 2 * spread),
                   rank_outliers=fraction((e.min(axis=-1) > 0) | (e.max(axis=-1) < 0)))
    # E[error^2] = (1 + 1/m) sigma^2 and E[spread^2] = (1 - (1 - ddof)/m) sigma^2 when calibrated
    metrics['spread_skill'] = (float(np.sqrt((m + 1) / (m - 1 + ddof)) * metrics['spread_sync'] / rmse_sync)
                               if rmse_sync > 0 else np.nan)

    err_fc_members = (np.sqrt((e ** 2).mean(axis=1)) / scale)[win_fc]     # (Nt_fc, m)
    vts = [first_exceed(err_fc_members[:, j]) for j in range(members.shape[-1])]
    metrics['valid_time_forecast'] = float(np.mean(vts))
    metrics['valid_time_std'] = float(np.std(vts))

    if ensemble is not None and ensemble.Na > 0:
        model = ensemble.model
        alpha_hist = model.hist[-1, model.Nphi:model.Nphi + ensemble.Na, :]
        true_params = getattr(truth, 'true_parameters', None) or {}
        for key, val, s in zip(ensemble.est_alpha,
                               alpha_hist.mean(axis=-1), alpha_hist.std(axis=-1)):
            metrics[f'alpha_{key}'] = float(val)
            metrics[f'alpha_std_{key}'] = float(s)
            if key in true_params:
                a_true = true_params[key]
                metrics[f'alpha_error_{key}'] = float(abs(val - a_true) / max(abs(a_true), 1e-12))
                metrics[f'alpha_cover_{key}'] = float(abs(val - a_true) <= 2 * s)

    return metrics


def state_error(result, t_true, y_true, per_member=True):
    """Normalized full-state error of a `deploy` result over time, as `(t, err, spread)`,
    normalized by `error_scale` of the truth over the whole run.

    `per_member=True` scores each member's own curve (mean/std across members);
    otherwise `err` is the ensemble-mean trajectory's error and `spread` the RMS
    spread. `spread` is all-NaN for the single-network open loop."""
    t, y = result['t'], result['y']
    _check_truth_covers(t, t_true)
    y_ref = interpolate(t_true, y_true, t)
    scale = error_scale(y_ref) / np.sqrt(y_ref.shape[1])

    ensemble = result.get('ensemble')
    hist = result.get('y_members')                            # open-loop ensemble
    if ensemble is not None:
        hist = ensemble.model.hist[:, :ensemble.model.N_dim, :]
    if hist is None:
        err = np.sqrt(((y - y_ref) ** 2).mean(axis=1)) / scale
        return t, err, np.full_like(err, np.nan)
    if per_member:
        e = np.sqrt(((hist - y_ref[:, :, None]) ** 2).mean(axis=1)) / scale   # (Nt, m)
        return t, e.mean(axis=-1), e.std(axis=-1)

    err = np.sqrt(((y - y_ref) ** 2).mean(axis=1)) / scale
    spread = np.sqrt((hist.std(axis=-1) ** 2).mean(axis=1)) / scale
    return t, err, spread


def relative_error(reference, estimate, axis=None):
    """Eq. (6.1) of Nóvoa et al. (2024): the root of the sum of the squared differences between `estimate` and
    `reference` over `axis` (all axes by default), over the root of the sum of the squared `reference`.

    An `estimate` with one more axis than `reference` holds members on its last axis, scored one by one; e.g. a
    reference (Nt, Nq) and an ensemble (Nt, Nq, m) with ``axis=1`` give the error of each step and member (Nt, m).
    """
    reference, estimate = np.asarray(reference), np.asarray(estimate)
    if estimate.ndim == reference.ndim + 1:
        reference = reference[..., None]
    return np.sqrt(np.sum((reference - estimate) ** 2, axis=axis) / np.sum(reference ** 2, axis=axis))


def window_errors(filter_ens, truth, t_start, t_end) -> dict:
    """Errors of a filtered ensemble against the truth over the window ``t_start <= t < t_end``.

    The error is that of Eq. (6.1) of Nóvoa et al. (2024), `relative_error` over the window and the observables. A window after the last
    observation scores the forecast without data, e.g. to select a regularization factor.

    Returns
    -------
    dict
        ``model`` : error of the ensemble mean of the model observables; ``corrected`` : error of the
        bias-corrected estimate, model plus bias (equal to ``model`` without a bias estimator); ``bias`` :
        root-mean-square of the bias estimate.
    """
    model = filter_ens.model
    t = model.hist_t
    window = (t >= t_start - 1e-12) & (t < t_end - 1e-12)
    if not window.any():
        raise ValueError(f'The run has no step in [{t_start}, {t_end}): it covers t = {t[0]} to {t[-1]}.')
    y_true = truth.y_true
    if y_true.ndim == 3:
        y_true = np.mean(y_true, axis=-1)
    y_ref = interpolate(truth.t_true, y_true, t[window])
    q = np.mean(model.get_observable_hist()[window], axis=-1)
    y = np.mean(filter_ens.get_observable_hist()[0][window], axis=-1) if getattr(filter_ens, 'is_bias_aware', False) else q

    return dict(model=float(relative_error(y_ref, q)), corrected=float(relative_error(y_ref, y)),
                bias=float(np.sqrt(np.mean((y - q) ** 2))))


def ensemble_metrics(filter_ens, truth) -> dict:
    """Scalar error metrics of a filtered ensemble against the unbiased truth.

    The physical-twin counterpart of `state_metrics` (which scores `deploy`
    results): works off the estimator's model histories. All RMS errors are
    computed on the ensemble-mean observables and normalized by the RMS
    amplitude of the truth over the assimilation window.

    Returns
    -------
    dict
        ``rmse_da`` / ``rmse_forecast`` : normalized RMSE of the model observables over
        the assimilation window and the post-assimilation forecast;
        ``rmse_corrected_da`` / ``rmse_corrected_forecast`` (bias-aware estimators only) :
        the same for the bias-corrected estimate, model plus bias, which is the estimate
        of the truth of a bias-aware filter (the model observables alone carry the model
        bias); ``spread_da`` / ``spread_forecast`` : normalized mean ensemble spread;
        ``alpha_<name>`` (+ ``alpha_error_<name>`` when the truth stores
        ``true_parameters``) : final ensemble-mean parameter estimates.
    """
    model = filter_ens.model
    t = model.hist_t
    y = model.get_observable_hist()          # (Nt, Nq, m)
    y_mean = np.mean(y, axis=-1)
    spread = np.std(y, axis=-1)

    y_true = truth.y_true
    if y_true.ndim == 3:
        y_true = np.mean(y_true, axis=-1)
    y_ref = interpolate(truth.t_true, y_true, t)

    t_first, t_last = truth.t_obs[0], truth.t_obs[-1]
    win_da = (t >= t_first) & (t <= t_last)
    # past the end of the truth record the interpolation repeats its last value: the forecast is scored where
    # the truth exists only
    win_fc = (t > t_last) & (t <= truth.t_true[-1] + 1e-9)

    scale = np.sqrt(np.mean(y_ref[win_da] ** 2))

    def _nrmse(window, estimate=y_mean):
        return float(np.sqrt(np.mean((estimate[window] - y_ref[window]) ** 2)) / scale)

    def _nspread(window):
        return float(np.mean(spread[window]) / scale)

    metrics = dict(rmse_da=_nrmse(win_da),
                   rmse_forecast=_nrmse(win_fc) if win_fc.any() else np.nan)

    # Bias-aware: the estimate of the truth is the bias-corrected one, model plus bias
    if getattr(filter_ens, 'is_bias_aware', False):
        y_corrected = np.mean(filter_ens.get_observable_hist()[0], axis=-1)
        metrics.update(rmse_corrected_da=_nrmse(win_da, y_corrected),
                       rmse_corrected_forecast=_nrmse(win_fc, y_corrected) if win_fc.any() else np.nan)

    metrics.update(spread_da=_nspread(win_da),
                   spread_forecast=_nspread(win_fc) if win_fc.any() else np.nan)

    # Parameter estimates (rows Nphi..Nphi+Na of the state augmentation)
    if filter_ens.Na > 0:
        alpha_hist = model.hist[:, model.Nphi:model.Nphi + filter_ens.Na, :]
        alpha_final = np.mean(alpha_hist[-1], axis=-1)
        for key, val in zip(filter_ens.est_alpha, alpha_final):
            metrics[f'alpha_{key}'] = float(val)
            if truth.true_parameters and key in truth.true_parameters:
                a_true = truth.true_parameters[key]
                metrics[f'alpha_error_{key}'] = float(abs(val - a_true) / max(abs(a_true), 1e-12))

    return metrics
