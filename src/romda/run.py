"""Data-assimilation experiments described by YAML files.

:func:`run_config` runs a physical twin experiment or the annular experimental-data case (``scripts/examples/lorenz``,
``scripts/thermoacoustics``); :func:`run_esn_config` runs a twin experiment whose forecast model is an echo state
network (``forecast.kind: esn`` or ``parametric_esn``, ``scripts/esnda/configs``). From the command line,
``python -m romda.run <config>.yml [--seed N] [--set dotted.key=value ...]`` runs either and prints the metrics.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import difflib
import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np
import yaml
from romda.bias_estimators import ESN_bias
from romda.data_assimilation import deploy, run_and_score, run_da_loop
from romda.estimators import EnKF, EnSRKF, rBA_EnKF, rEnKF
from romda.metrics import ensemble_metrics, state_metrics
from romda.models import RK4Integrator, physical
from romda.observations import Observations, measured_idx, variance_fraction_covariance
from romda.utils import (
    convert_to_python_type,
    mean_vector_to_ensemble,
    noisy_copy,
    seed_noise,
    set_working_directories,
    trajectory,
)

DA_METHODS = {'EnKF': EnKF, 'EnSRKF': EnSRKF, 'rEnKF': rEnKF, 'rBA_EnKF': rBA_EnKF}
_DA_ONLY = ('measured', 'n_measured', 'ring', 'bayesian_update', 'case')
#: integrators that the ``da`` block names (default: that of the model, scipy's RK45 for the physical models)
INTEGRATORS = {'rk4': RK4Integrator}

#: keys that this module reads itself, by block; every other key must name an attribute of the class it configures
_OWN_KEYS = {
    'truth': {'model', 'params', 'ER', 't_stop_cr', 'observe_every', 'window', 'window_lyap', 'observe_dims',
              'observed_idx'},
    # the estimator keywords set attributes of another name (inflation -> inflation_factor, ...)
    'da': {'method', 'da_method', 'std_obs', 'seed', 'integrator', 'spinup_cr', 'Cdd_entries', *_DA_ONLY,
           'inflation', 'bias_regularization', 'localization', 'state_positions', 'obs_positions', 'loc_period'},
    # the ranges of the training members, and the options of the assimilation loop and of this module
    'bias': {'washout', 'training_ERs', 't_train_lyap', 't_val_lyap', 't_test_lyap', 'std_alpha', 'std_phi'},
}


def _unknown_keys(block, cfg, receivers, accepted=()):
    """The keys of block `block` that neither this module nor any of the `receivers` reads, each with the
    closest known key. The classes of romda and dynamodels take a keyword only when they have an attribute of
    that name, and drop any other in silence, so a misspelt key would leave its default in place."""
    known = set(_OWN_KEYS[block]) | set(accepted)
    unknown = [key for key in cfg if key not in known and not any(hasattr(cls, key) for cls in receivers)]
    if not unknown:
        return []
    names = known | {name for cls in receivers for name in dir(cls) if not name.startswith('_')}
    return [(block, key, difflib.get_close_matches(key, names, n=1, cutoff=0.7)) for key in unknown]


def check_config_keys(truth_cfg, da_cfg, bias_cfg, model_class, da_method):
    """Raise a `ValueError` that names every key of the configuration that no part of the experiment reads.

    Parameters
    ----------
    truth_cfg, da_cfg, bias_cfg : dict
        The ``truth``, ``da`` and ``bias`` blocks, with their overrides; `bias_cfg` is checked only for a
        bias-aware method.
    model_class : type
        The physical model.
    da_method : type
        The estimator class.
    """
    from romda.models.data_driven.rnns.esn import ESN_model
    from romda.models.data_driven.rnns.esn_config import ESNConfig

    ensemble_keywords = set(inspect.signature(model_class.init_ensemble).parameters) - {'self'}
    problems = _unknown_keys('truth', truth_cfg, (Observations, model_class))
    problems += _unknown_keys('da', da_cfg, (da_method, model_class), accepted=ensemble_keywords)
    if issubclass(da_method, rEnKF):
        problems += _unknown_keys('bias', bias_cfg or {}, (ESN_bias, ESN_model),
                                  accepted={field.name for field in dataclasses.fields(ESNConfig)})
    if problems:
        lines = [f"  {block}.{key}" + (f" (did you mean '{match[0]}'?)" if match else '')
                 for block, key, match in problems]
        raise ValueError('These configuration keys are read by no part of the experiment, so they would be '
                         'ignored:\n' + '\n'.join(lines))


def _every_nth(constructor, step):
    return list(range(0, constructor['Nx'], step))


def build_truth(model_class, cfg, overrides=None):
    """The `Observations` of the truth block `cfg` of a configuration, with `overrides` over it: the record of the
    equivalence ratio ``ER`` of the annular data set, or else the model `model_class` integrated over ``window``
    (or ``window_lyap``, in Lyapunov times), observed every ``observe_every`` site when given."""
    params = {**cfg, **(overrides or {})}
    params.pop('model', None)
    constructor = params.pop('params', None) or {}

    if 'ER' in params:
        from romda.utils import get_annular_data

        ER = params.pop('ER')
        data_folder = set_working_directories(model_class.__name__.lower())[0]
        get_annular_data(data_folder)
        params.setdefault('t_start', model_class.t_transient)
        params['t_stop'] = params['t_start'] + model_class.t_CR * params.pop('t_stop_cr')
        params.setdefault('t_max', params['t_stop'] + model_class.t_transient)
        return Observations(model=os.path.join(data_folder, f'ER_{ER}'), **params)

    if 'observe_every' in params:
        params['observed_idx'] = _every_nth(constructor, params.pop('observe_every'))
    window = params.pop('window', None)
    if window is None:
        window = [value * model_class.t_lyap for value in params.pop('window_lyap')]
    t_start, t_stop, t_max = window
    return Observations(model=model_class, t_start=t_start, t_stop=t_stop, t_max=t_max,
                        **constructor, **params)


def _build_ensemble(model_class, cfg, truth, truth_cfg, method=None, overrides=None):
    params = {k: v for k, v in {**cfg, **(overrides or {})}.items()
              if k not in _DA_ONLY and k not in ('method', 'std_obs')}
    integrator, spinup_cr = params.pop('integrator', None), params.pop('spinup_cr', None)
    params.pop('Cdd_entries', None)
    if integrator is not None and integrator not in INTEGRATORS:
        raise ValueError(f'integrator={integrator!r}: expected one of {sorted(INTEGRATORS)}, or no key for the '
                         'integrator of the model.')
    da_method = method or cfg.get('method', cfg.get('da_method'))
    da_method = DA_METHODS[da_method] if isinstance(da_method, str) else da_method
    if not issubclass(da_method, rEnKF):
        params.pop('bias_regularization', None)

    constructor = dict(truth_cfg.get('params') or {})
    layout = {k: truth_cfg[k] for k in ('observe_dims', 'observed_idx') if k in truth_cfg}
    if 'observe_every' in truth_cfg:  # the ensemble must observe what the truth observes
        layout['observed_idx'] = _every_nth(constructor, truth_cfg['observe_every'])
    estimated = set(params.get('std_alpha') or ())
    shared = {k: v for k, v in constructor.items()
              if k not in estimated and k != 'dt'}
    ensemble = da_method(parent_model=model_class, dt=truth.dt, **shared, **layout, **params)
    model = ensemble.model
    if integrator is not None:
        model.integrator.close()
        model.integrator = INTEGRATORS[integrator](model)
    if spinup_cr is not None:
        reference = model_class(dt=truth.dt, **shared)
        if integrator is not None:
            reference.integrator = INTEGRATORS[integrator](reference)
        draw_state_after_spinup(model, reference, spinup_cr, std_phi=params.get('std_phi', 0.001),
                                distribution_phi=params.get('distribution_phi', 'normal'))
    return ensemble


def draw_state_after_spinup(model, reference, spinup_cr=1., std_phi=0.001, distribution_phi='normal'):
    """Replace the state rows of the initial ensemble of `model`: the members are drawn about the state that
    the `reference` model, a single member with the default parameters, reaches after ``spinup_cr * t_CR`` from
    its initial condition, with the relative spread `std_phi`, as the v1.0 code does (``create_ensemble``, with
    one `t_CR`). `Model.init_ensemble` forecasts `t_transient` instead, in which a model that is stable at its
    default parameters (the annular model) decays to its fixed point. The parameter rows keep the draw of
    `init_ensemble`, and `reference` is left as it was."""
    psi, _ = reference.time_integrate(Nt=int(round(spinup_cr * reference.t_CR / reference.dt)))
    reference.close()
    phi = mean_vector_to_ensemble(model.rng, psi[-1, :model.Nphi, 0], std_phi, model.m, method=distribution_phi)
    state = np.array(model.current_state)
    state[:model.Nphi] = phi
    model.update_history(psi=state[np.newaxis], t=model.hist_t[[0]], reset=True)


#: options of the bias block, and of the initial ensemble, that change the bias training data
_DATA_OPTIONS = ('training_record', 'lag_criterion', 'keep_fixed_points', 'L', 'augment_data',
                 'correlation_based_training', 'integrator', 'spinup_cr')


def _training_data_key(truth, params):
    """What the bias training data of an experimental record depend on, for the name of their cache: the
    record (e.g. ``ER_0.5125``), the end of the record (the data are its last steps) and the ranges that the
    training members are drawn from. The trained network is stored under a hash of this name, so that a second
    record, window or set of ranges does not reuse the data and the network of the first. A list of records
    (``training_ERs``) joins their names with '+'."""
    records = truth if isinstance(truth, (list, tuple)) else [truth]
    record = '+'.join(item.name_truth.removeprefix('Truth_Exp_') for item in records)
    truth = records[0]
    options = {key: params.get(key) for key in ('std_alpha', 'std_phi')}
    # the options of the data added later enter the digest only when given, so that the names of the datasets
    # cached before them do not change
    options.update({key: params[key] for key in _DATA_OPTIONS if key in params})
    digest = hashlib.sha256(json.dumps(convert_to_python_type(options), sort_keys=True).encode()).hexdigest()[:8]
    return f'{record}_t{truth.t_true[-1]:.4g}_{digest}'


def _build_bias(cfg, ensemble, truth, model_class, experimental=False, overrides=None, data_options=None,
                truth_cfg=None):
    params = {**(cfg or {}), **(overrides or {})}
    params.pop('washout', None)                   # an option of the assimilation loop (`run_config`)
    # training_ERs: the network is trained on the records of these equivalence ratios instead of the assimilated
    # one (one network for several records, Nóvoa et al. 2024, section 6.4)
    training_ERs = params.pop('training_ERs', None)
    reference = truth
    if training_ERs is not None:
        if not experimental:
            raise ValueError('training_ERs needs an experimental record (a truth block with ER).')
        reference = [build_truth(model_class, truth_cfg, dict(ER=ER)) for ER in training_ERs]
    for key in ('t_train', 't_val', 't_test'):
        if f'{key}_lyap' in params:
            params[key] = params.pop(f'{key}_lyap') * model_class.t_lyap
    name = model_class.__name__.lower()
    if experimental:
        params.setdefault('t_train', model_class.t_transient / 3.)
        params.setdefault('t_test', model_class.t_CR * 2)
        params.setdefault('t_val', model_class.t_CR * 2)
        tag = f'ESN_train_data_{name}_raw_{_training_data_key(reference, {**params, **(data_options or {})})}'
    else:
        tag = f'ESN_train_data_{name}_{truth.name_bias}'
    results_folder = set_working_directories(name)[1]
    os.makedirs(results_folder, exist_ok=True)
    params.setdefault('N_ens', ensemble.m)        # one bias state for each member; N_ens: 1 carries their mean
    return ESN_bias(rom=ensemble.model, reference_data=reference,
                    training_data_filename=f'{results_folder}{tag}',
                    **params)


def run_config(config, seed=0, da_method=None, std_obs=None,
                   truth_kwargs=None, ensemble_kwargs=None, bias_kwargs=None,
                   t_extra=None):
    """Run a physical twin or thermoacoustic experiment.

    Parameters
    ----------
    config : dict or path-like
        Resolved experiment mapping or a YAML configuration path.
    seed : int, default=0
        Seed of the run: it draws the physical-model ensemble and restarts the shared
        observation-noise generator (`romda.utils.seed_noise`), so the same seed gives the
        same run. A ``seed`` in `ensemble_kwargs`, then one in the ``da`` block of the
        configuration, takes precedence.
    std_obs : float, optional
        Overrides ``da.std_obs``. Despite its name, ``std_obs`` is a fraction of a VARIANCE, the
        convention of Nóvoa et al. (2024) that the configurations and their results follow: the
        observation error covariance is diagonal with the variance ``std_obs * max|d|^2``
        (`romda.observations.variance_fraction_covariance`), so ``std_obs = 0.01`` is a standard
        deviation of 10 % of the largest observation, and ``0.1`` one of 32 %.

    Returns
    -------
    tuple
        Filtered ensemble, truth observations, and scalar metrics.

    Raises
    ------
    ValueError
        If a key of the ``truth``, ``da`` or ``bias`` block is read by no part of the experiment
        (a misspelt key would otherwise leave its default in place), or if an explicit forecast
        length after the last observation, `t_extra`, runs past the end of the truth record, which
        the metrics cannot score. Both are checked before any assimilation. The default forecast
        length may run past the record; the metrics then score the part that the record covers.
    """
    if isinstance(config, (str, Path)):
        with open(config) as stream:
            config = yaml.safe_load(stream)
    truth_cfg = {**config['truth'], **(truth_kwargs or {})}
    model_class = getattr(physical, truth_cfg['model'])
    experimental = 'ER' in truth_cfg
    ensemble_options = dict(ensemble_kwargs or {})
    if seed:
        ensemble_options.setdefault('seed', seed)
    method = da_method or config['da'].get('method', config['da'].get('da_method'))
    check_config_keys(truth_cfg, {**config['da'], **ensemble_options},
                      {**(config.get('bias') or {}), **(bias_kwargs or {})}, model_class,
                      DA_METHODS[method] if isinstance(method, str) else method)
    # the noise of the truth follows the seed of the ensemble: ensemble_kwargs, else the da block of the config
    seed_noise({**config['da'], **ensemble_options}.get('seed', 0))
    truth = build_truth(model_class, config['truth'], truth_kwargs)
    ensemble = _build_ensemble(model_class, config['da'], truth, truth_cfg,
                               method=da_method, overrides=ensemble_options)
    da_cfg = {**config['da'], **ensemble_options}
    if isinstance(ensemble, rEnKF):
        # the training members start from the initial ensemble, which the integrator and the spin-up change
        data_options = {key: da_cfg[key] for key in ('integrator', 'spinup_cr') if key in da_cfg}
        ensemble.bias = _build_bias(config.get('bias'), ensemble, truth, model_class,
                                    experimental=experimental, overrides=bias_kwargs, data_options=data_options,
                                    truth_cfg=truth_cfg)
    # an explicit forecast length that the truth record cannot score is refused before the assimilation; the
    # default one may run past the record, and the metrics then score only the part the record covers
    if t_extra is not None and truth.t_obs[-1] + t_extra > truth.t_true[-1] + truth.dt:
        raise ValueError(f'The forecast after the last observation runs to t = {truth.t_obs[-1] + t_extra:.4g}, but the '
                         f'truth record ends at t = {truth.t_true[-1]:.4g}: shorten t_extra, or end the assimilation '
                         'earlier, or extend the truth (t_max, window or window_lyap).')
    if t_extra is None and experimental:
        t_extra = 5. * model_class.t_CR
    std_obs = config['da']['std_obs'] if std_obs is None else std_obs
    entries = da_cfg.get('Cdd_entries', 'per_observable')
    if entries not in ('per_observable', 'identical'):
        raise ValueError(f"Cdd_entries={entries!r}: expected 'per_observable' or 'identical'.")
    Cdd = variance_fraction_covariance(truth, std_obs, identical=True) if entries == 'identical' else None
    # the bias block may ask for the washout of its network on the data before the first observation
    washout = isinstance(ensemble, rEnKF) and bool({**(config.get('bias') or {}), **(bias_kwargs or {})}.get('washout'))
    filtered = run_da_loop(ensemble, truth, std_obs=std_obs, Cdd=Cdd, t_extra=t_extra, washout=washout)
    return filtered, truth, ensemble_metrics(filtered, truth)


# ── Twin experiments of an echo state network (forecast.kind: esn or parametric_esn) ───────────────────────────────
# A configuration of scripts/esnda/configs/*.yml: the truth (a `romda.models.physical` class name, its constructor
# parameters and dataset options), the forecast model (``kind: esn`` trained on the record of the truth, ``kind:
# parametric_esn`` trained across regimes, or ``kind: physical``, the model of the truth itself, see `run_config`),
# the deployment and the assimilation window. Package defaults, then the file, then the overrides are applied before
# the store hash of the network is computed, so one resolved configuration maps to one network of the store.

#: Allowed keys per validated block. `forecast.esn` / `forecast.train` are left
#: free-form (they are the `ESNConfig` / parametric-training surfaces, validated
#: downstream); `da` additionally accepts '<param>_true' keys — the true value of
#: the truth's swept parameter in a parametric twin experiment (e.g. ``F_true``).
TOP_KEYS = ('name', 'truth', 'forecast', 'da', 'bias', 'window_lyap', 'seed_shift_lyap')
TRUTH_KEYS = ('model', 'params', 'noise_level', 'num_lyap_times', 'split_lyap', 'seed',
              # physical-twin / experimental truths (forecast.kind: physical)
              'window', 'window_lyap', 'Nt_obs', 'add_noise', 'noise_type', 'manual_bias',
              'ER', 't_stop_cr', 't_start', 't_stop', 't_max',
              'observe_dims', 'observed_idx', 'observe_every')
DA_KEYS = ('method', 'da_method', 'seed', 'm', 'Nt_obs', 'std_obs', 'inflation',
           'adaptive_inflation', 'inflation_n_rand', 'inflate_parameters',
           'est_alpha', 'measured', 'n_measured', 'ring', 'bayesian_update', 'case',
           # ensemble generation (forecast.kind: physical)
           'std_phi', 'std_alpha', 'distribution_alpha',
           # the estimator's own spelling, for the physical configs (configs/*.yml)
           'bias_regularization',
           # Cdd as a fraction of the standard deviation of each observable (romda.data_assimilation.assimilate)
           'cdd_fraction',
           # the options of inflation='innovation' (EnsembleFilter)
           'inflation_guard', 'inflation_average', 'inflation_rate', 'inflation_max')

#: Package defaults, applied by `resolve_config` beneath the file's values.
DA_DEFAULTS = dict(seed=0, bayesian_update=True, measured=None, case=None,
                   ring=False, est_alpha=[], inflation=1.0)
SEED_SHIFT_LYAP = 4.0   # truth-window slide per seed, in Lyapunov times


def _hertz(loader, node):
    """YAML tag ``!hz``: a frequency, or a list of them, in Hz, read as an angular frequency 2 pi f (rad/s),
    e.g. ``omega: !hz [1090, 1095]``."""
    if isinstance(node, yaml.SequenceNode):
        return [float(2 * np.pi * float(f)) for f in loader.construct_sequence(node)]
    return float(2 * np.pi * float(loader.construct_scalar(node)))


yaml.SafeLoader.add_constructor('!hz', _hertz)


def resolve_model(name):
    """Truth-model registry: a `romda.models.physical` class name in, the class out."""
    from romda.models import physical
    cls = getattr(physical, name, None)
    if not isinstance(cls, type):
        available = sorted(k for k in vars(physical)
                           if isinstance(getattr(physical, k), type) and not k.startswith('_'))
        raise ValueError(f"Unknown truth model '{name}'. Available: {available}")
    return cls


def resolve_da_method(name):
    """DA-method registry: 'EnKF' / 'EnSRKF' / 'rBA_EnKF' in, the estimator class out."""
    if isinstance(name, type):
        return name
    if name not in DA_METHODS:
        raise ValueError(f"Unknown DA method '{name}'. Available: {sorted(DA_METHODS)}")
    return DA_METHODS[name]


def validate_config(cfg: dict) -> dict:
    """Reject unknown top-level, `truth` or `da` keys (typos fail loudly, not silently)."""
    if not isinstance(cfg, dict):
        raise ValueError(f'Config must be a mapping, got {type(cfg).__name__}')

    def _unknown(block, allowed, extra_ok=()):
        return sorted(k for k in block
                      if k not in allowed and not any(fn(k) for fn in extra_ok))

    bad = _unknown(cfg, TOP_KEYS)
    if bad:
        raise ValueError(f'Unknown config key(s) {bad}; allowed: {sorted(TOP_KEYS)}')
    bad = _unknown(cfg.get('truth') or {}, TRUTH_KEYS)
    if bad:
        raise ValueError(f'Unknown truth key(s) {bad}; allowed: {sorted(TRUTH_KEYS)}')
    bad = _unknown(cfg.get('da') or {}, DA_KEYS,
                   extra_ok=(lambda k: isinstance(k, str) and k.endswith('_true'),))
    if bad:
        raise ValueError(f"Unknown da key(s) {bad}; allowed: {sorted(DA_KEYS)} "
                         "plus '<param>_true'")
    return cfg


def resolve_config(cfg: dict) -> dict:
    """Package defaults beneath the given config; returns a deep copy."""
    cfg = copy.deepcopy(cfg)
    truth = cfg.setdefault('truth', {})
    truth.setdefault('params', {})
    truth.setdefault('seed', 0)

    forecast = cfg.setdefault('forecast', {})
    forecast.setdefault('kind', 'esn')
    forecast.setdefault('esn', {})

    if 'da' in cfg and cfg['da'] is not None:
        for key, val in DA_DEFAULTS.items():
            cfg['da'].setdefault(key, copy.deepcopy(val))

    cfg.setdefault('seed_shift_lyap', SEED_SHIFT_LYAP)
    return cfg


def load_config(path) -> dict:
    """Load, validate and resolve (fill package defaults into) a YAML config file."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return resolve_config(validate_config(cfg))


def _apply_override(cfg: dict, dotted_key: str, value):
    parts = dotted_key.split('.')
    block = cfg
    for p in parts[:-1]:
        nxt = block.setdefault(p, {})
        if not isinstance(nxt, dict):
            raise ValueError(f"Cannot override '{dotted_key}': '{p}' is not a mapping")
        block = nxt
    block[parts[-1]] = value


INNOVATION_OPTIONS = ('inflation_guard', 'inflation_average', 'inflation_rate', 'inflation_max')


def innovation_options(da: dict) -> dict:
    """The options of ``inflation='innovation'`` that a ``da`` block sets (`EnsembleFilter`: `inflation_guard`,
    `inflation_average`, `inflation_rate`, `inflation_max`). They act only with ``inflation='innovation'``, so a run can
    pass them whatever its inflation."""
    return {k: da[k] for k in INNOVATION_OPTIONS if k in da}


def _inflation_kwargs(da: dict) -> dict:
    """Optional inflation keys, forwarded only when the config sets them
    (absent keys leave existing configs' behaviour bit-identical)."""
    return {**{k: da[k] for k in ('adaptive_inflation', 'inflation_n_rand', 'inflate_parameters') if k in da},
            **innovation_options(da)}


def _da_method_name(da: dict):
    name = da.get('method', da.get('da_method'))
    if name is None:
        raise ValueError("da block needs a 'method' (or legacy 'da_method') entry")
    return name


def run_esn_config(cfg_or_path, seed=0, config_dir=None, force_create=False,
                       **overrides) -> dict:
    """Run the twin experiment a config describes and return its metrics dict.

    ``kind: esn`` builds the network on the truth's own record (`model_esn`) and
    scores the deployment on the held-out end (`run_and_score`, truth window slid
    by `seed_shift_lyap` Lyapunov times per seed). ``kind: parametric_esn``
    synthesizes a *fresh* truth trajectory at the unseen parameter value
    ``da.<param>_true``, estimating that parameter online alongside the state

    `overrides` are applied on top of the resolved config; keys may be dotted
    paths into the nested blocks (``run_config(cfg, **{'da.m': 32})``).
    `config_dir` / `force_create` go to the ESN store lookup
    (`romda.models.data_driven.rnns.auto_load_or_create`). ``da.cdd_fraction`` = f sets `Cdd` to f times the
    standard deviation of each observable over the truth record (`romda.data_assimilation.assimilate`).
    """
    from romda.models.data_driven.rnns.builders import model_esn, parametric_esn  # the networks, when needed
    if isinstance(cfg_or_path, (str, Path)):
        cfg = load_config(cfg_or_path)
    else:
        cfg = resolve_config(validate_config(cfg_or_path))

    for key, value in overrides.items():
        _apply_override(cfg, key, value)
    validate_config(cfg)

    truth_cfg = cfg['truth']
    model_class = resolve_model(truth_cfg['model'])
    truth_params = truth_cfg['params']
    fc = cfg['forecast']
    kind = fc['kind']

    da = cfg.get('da')
    if not da:
        raise ValueError(f"Config '{cfg.get('name')}' has no da block: nothing to run")
    da_method = resolve_da_method(_da_method_name(da))

    if kind == 'physical':
        # the truth's own physical model forecasts (twin / experimental-data DA);
        # windows and Nt_obs live in the truth block, not window_lyap
        for key in ('m', 'std_obs'):
            if key not in da:
                raise ValueError(f"da block is missing '{key}'")
        return run_config(cfg, seed=seed, da_method=da_method)[2]

    for key in ('m', 'Nt_obs', 'std_obs'):
        if key not in da:
            raise ValueError(f"da block is missing '{key}'")
    if 'window_lyap' not in cfg:
        raise ValueError("Config is missing 'window_lyap' (start, stop, extra forecast)")
    window_lyap = tuple(cfg['window_lyap'])
    seed_shift_lyap = cfg['seed_shift_lyap']

    if kind == 'esn':
        for key in ('noise_level', 'num_lyap_times', 'split_lyap'):
            if key not in truth_cfg:
                raise ValueError(f"kind 'esn' needs truth.{key}")
        n_dim = model_class(**truth_params).Nphi
        spec = da['measured'] if da.get('measured') is not None else da.get('n_measured')
        measured = measured_idx(spec, n_dim, ring=da['ring'])

        esn, data = model_esn(model_class, truth_cfg, fc['esn'],
                              measured=measured, bayesian_update=da['bayesian_update'],
                              plot_training=False,
                              config_dir=config_dir, force_create=force_create)
        extra = dict(est_alpha=list(da['est_alpha'])) if da.get('est_alpha') else {}
        extra.update(_inflation_kwargs(da))
        return run_and_score(esn, data, measured, seed=seed, da_method=da_method,
                             m=da['m'], Nt_obs=da['Nt_obs'], std_obs=da['std_obs'],
                             inflation=da['inflation'], cdd_fraction=da.get('cdd_fraction'),
                             window_lyap=window_lyap, seed_shift_lyap=seed_shift_lyap,
                             bayesian_update=da['bayesian_update'], **extra)

    if kind == 'parametric_esn':
        tr = fc.get('train')
        if not tr:
            raise ValueError("kind 'parametric_esn' needs a forecast.train block")
        if 'noise_level' not in truth_cfg:
            raise ValueError("kind 'parametric_esn' needs truth.noise_level")
        param, est_name, values = tr['param'], tr['est_name'], tr['values']

        true_value = da.get(f'{param}_true')
        if true_value is None:
            raise ValueError(f"kind 'parametric_esn' needs da.{param}_true "
                             '(the unseen truth value of the swept parameter)')

        esn, _, regimes = parametric_esn(model_class, tr, fc['esn'], truth_cfg['noise_level'],
                                         model_kwargs=truth_params,
                                         plot_training=False,
                                         config_dir=config_dir, force_create=force_create)
        # reject analyses outside the sensible range around the training regimes
        # (the (min-0.5, max+1.0) margins reproduce the (3, 9) used for Lorenz-96)
        esn.alpha_lims = {est_name: (float(min(values)) - 0.5, float(max(values)) + 1.0)}
        esn.alpha0[est_name] = float(np.mean(regimes))

        spec = da['measured'] if da.get('measured') is not None else da.get('n_measured')
        measured = measured_idx(spec, esn.N_dim, ring=da['ring'])

        t_lyap = model_class.t_lyap
        t_start_lyap, t_stop_lyap, t_extra_lyap = window_lyap
        n_truth = int(round((t_stop_lyap + t_extra_lyap + 1) * t_lyap / esn.dt))
        traj = trajectory(model_class, seed=seed, n_steps=n_truth, dt=esn.dt,
                          constructor_defaults=truth_params, **{param: true_value})
        t_truth = np.arange(n_truth) * esn.dt
        noise_level = truth_cfg['noise_level']
        noisy = noisy_copy(traj, noise_level, seed)

        truth = Observations(y_raw=noisy[:, measured], y_true=traj[:, measured],
                             t_true=t_truth,
                             t_start=t_start_lyap * t_lyap, t_stop=t_stop_lyap * t_lyap,
                             Nt_obs=da['Nt_obs'])
        truth.true_parameters = {est_name: true_value}

        est_alpha = list(da['est_alpha']) or [est_name]
        result = deploy(esn, truth, measured=measured, m=da['m'], da_method=da_method,
                        std_obs=da['std_obs'], t_extra=t_extra_lyap * t_lyap,
                        inflation=da['inflation'], cdd_fraction=da.get('cdd_fraction'),
                        regimes=regimes, est_alpha=est_alpha, **_inflation_kwargs(da))

        # no t_lyap here on purpose: the parametric ablation records its valid times in
        # absolute time units, and existing jsonl records must stay comparable
        metrics = state_metrics(result, truth, t_truth, traj)
        metrics['n_measured'] = len(measured)
        return metrics

    raise ValueError(f"Unknown forecast.kind '{kind}': expected 'esn', 'parametric_esn' or 'physical'")


def _main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='path to an experiment YAML file (physical model or echo state network)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--set', dest='overrides', action='append', default=[], metavar='KEY=VALUE')
    args = parser.parse_args(argv)
    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    for item in args.overrides:
        key, _, raw = item.partition('=')
        if not key or not _:
            parser.error(f'--set expects KEY=VALUE, got {item!r}')
        block = config
        parts = key.split('.')
        for part in parts[:-1]:
            block = block.setdefault(part, {})
        try:
            block[parts[-1]] = json.loads(raw)
        except json.JSONDecodeError:
            block[parts[-1]] = raw
    if (config.get('forecast') or {}).get('kind', 'physical') != 'physical':    # an echo state network
        metrics = run_esn_config(config, seed=args.seed)
    else:
        _, _, metrics = run_config(config, seed=args.seed)
    print(json.dumps(metrics))


if __name__ == '__main__':
    _main()
