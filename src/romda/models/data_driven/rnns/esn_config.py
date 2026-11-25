import hashlib
import json
import logging
from dataclasses import asdict, dataclass, fields
from functools import partial
from pathlib import Path

import numpy as np
import yaml
from romda.utils import convert_to_python_type

from ..model_config import MODEL_STORE, training_data_name
from ..model_config import find_matching_config as _find_matching_config
from .esn import LEGACY_CONVENTIONS, ESN_model, has_intercept, trained_scale_bias_in

logger = logging.getLogger(__name__)

#: the trained-network store is a project-local cache under results/ (gitignored)
ESN_STORE = MODEL_STORE.with_name('esn_configs')

#: INIT_KEYS added after networks were cached: `to_hash` drops each one at the value that those networks were
#: trained with, so every hash taken before they joined still finds its network. That value is the default of
#: the option, or, for the conventions whose default changed afterwards, that of `LEGACY_CONVENTIONS`: a default
#: configuration then has another hash than the networks of the former conventions
HASHED_IF_NOT_DEFAULT = ('hyperparameters_to_optimize', 'bias_in_range', 'regularize_bias_out',
                         'stability_steps', 'bho_noise', 'grid_midpoints',
                         'optimize_parameter_normalization', 'param_shift_range',
                         'param_norm_range', 'patch_size', 'halo', 'periodic', 'shared', 'stability_starts',
                         'stability_penalty', 'sigma_param_range', 'centre_input', 'validation_metric',
                         'param_in_reservoir', 'readout_input', 'scale_bias_in')

#: left unset when building a network at their default: EchoStateNetwork then searches the
#: parameter normalization whenever input_parameters are given, over ranges sized from the
#: parameters, which the parametric networks of stores without these keys rely on
PARAM_NORM_KEYS = ('optimize_parameter_normalization', 'param_shift_range', 'param_norm_range')

TRAINED_KEYS = ['norm', 'shift',
                'rho', 'sigma_in', 'sigma_param', 'tikh',
                'Win', 'W', 'Wout', 'validation_data',
                # parametric ESN: saved with the matrices, not hashed — the regime set
                # is already identified by `training_data_filename`
                'param_names', 'param_values', 'param_labels', 'input_parameters']


@dataclass
class ESNConfig:
    """Configuration class for ESN_model to enable easy save/load without retraining."""

    # Model dimensions
    N_dim: int = 0
    observed_idx: list | None = None
    update_reservoir: bool = True
    update_state: bool = True
    Wout_svd: bool = False
    # the data the network was trained on, a name or a path: the hash keeps the name
    # (`training_data_name`), so a path finds the network from every clone and machine
    training_data_filename: str | None = None
    plot_training: bool = False

    # Time parameters.
    dt: float = 0.1
    t_train: float | None = None
    t_val: float | None = None
    t_test: float | None = None

    # ESN architecture parameters
    N_units: int = 50
    norm_method: str = 'range'
    # the conventions of the network (see `ESN_model`); the published studies pin `LEGACY_CONVENTIONS`
    Win_type: str = 'sparse_dense_bias'
    scale_bias_in: bool = False
    N_wash: int = 5
    upsample:  int = 1
    bias_in: float = 0.1
    # options of the echostatenetwork whose Wout holds the constant of the readout in its last row (<= 0.6): the
    # constant that the row multiplies, and whether the Tikhonov factor acts on the row
    bias_out: float = 1.0
    regularize_bias_out: bool = False
    connect: float = 3
    # parallel layout (echostatenetwork): one reservoir per patch of patch_size sites, reading halo sites on either
    # side; set before the trained matrices, whose shapes depend on it (independent patches, shared=False, are not
    # stored)
    patch_size: int | None = None
    halo: int = 0
    periodic: bool = True
    shared: bool = True

    # Training parameters
    noise:  float = 1e-2
    noise_type: str = 'gauss'
    N_func_evals: int = 40
    N_grid:  int = 5
    N_folds: int = 8
    N_split: int = 5
    N_initial_rand: int = 0

    # Hyperparameter optimization ranges
    rho_range: tuple[float, float] = (0.2, 0.8)
    sigma_in_range: tuple[float, float] = (-2, 2)
    sigma_param_range: tuple[float, float] = (-3, 0.477)
    tikh_range: tuple[float, ... ] = (1e-6, 1e-9, 1e-12)
    bias_in_range: tuple[float, float] = (0.0, 2.0)
    hyperparameters_to_optimize: tuple[str, ...] = ('rho', 'sigma_in', 'tikh')
    # Bayesian search options: the defaults of echostatenetwork 0.2.3
    stability_steps: int = 0
    stability_starts: int = 1
    stability_penalty: float = 10.0   # echostatenetwork >= 0.3.1: the score of a diverged probe
    bho_noise: float | str = 1e-10
    grid_midpoints: bool = False
    # ESN_model options: the input scaled by its range without its mean (False), and the name of the
    # validation metric (None: the default of echostatenetwork); see `ESN_model`
    centre_input: bool = True
    validation_metric: str | None = None
    # parametric ESN: search of the parameter shift and scale (echostatenetwork defaults;
    # `to_esn_model` leaves them unset at these values, see PARAM_NORM_KEYS)
    optimize_parameter_normalization: bool = False
    param_shift_range: tuple[float, float] = (-1.0, 1.0)
    param_norm_range: tuple[float, float] = (0.1, 10.0)
    # parametric ESN: the parameters enter the reservoir (W_in) and/or the readout (readout_input='params' reads
    # W_out on [r; p; 1]); echostatenetwork defaults
    param_in_reservoir: bool = True
    readout_input: bool | str = False

    # Additional metadata
    seed: int | None = 0


    # Parametric ESN fields — declared BEFORE the trained matrices: `asdict` order is
    # the order `ESN_model` applies them on load, and `input_parameters` must be set
    # before `norm`/`Win` so their (N_dim_in,)-shape validation sees the parameter rows
    param_names: tuple = ()
    param_values: tuple | None = None
    param_labels: tuple | None = None
    input_parameters: np.ndarray | None = None

    # Trained data keys (loaded separately)
    norm: np.ndarray | None = None
    shift: np.ndarray | None = None
    rho: float | None = None
    sigma_in: float | None = None
    sigma_param: float | None = None
    tikh: float | None = None
    Win: np.ndarray | None = None
    W: np.ndarray | None = None
    Wout: np.ndarray | None = None
    validation_data: np.ndarray | None = None

    @staticmethod
    def _init_config_dict(case):
        # default for keys an older echostatenetwork does not define
        config = {key: getattr(case, key, getattr(ESNConfig, key)) for key in INIT_KEYS}
        if callable(config['validation_metric']):       # an ESN_model holds the function: its name is hashed
            config['validation_metric'] = getattr(case, 'validation_metric_name', None)
        return config

    def to_hash(self):
        """Compute a hash of the initial configuration parameters.

        Uses INITIAL parameters only (before any optimization), so configs match
        even if hyperparameter optimization produces slightly different results
        due to randomness. The training data count by the name of
        `training_data_filename`, not by its path (`training_data_name`), so a
        network is found from every clone and machine.

        Returns
        -------
        str
            The first 16 hex characters of the SHA256 digest of the config.
        """
        return ESNConfig._digest(self._hashed_params())

    def _hashed_params(self):
        """The initial parameters as `to_hash` digests them: types normalized, the
        training data keyed by their name, and the options of HASHED_IF_NOT_DEFAULT
        left out at their defaults."""
        hash_params = ESNConfig._init_config_dict(self)

        # Normalize types to ensure consistent hashing
        # Convert all numpy arrays and lists to consistent format
        for key in ['bias_in', 'bias_out']:
            if hash_params[key] is not None:
                if isinstance(hash_params[key], (float, int)):
                    hash_params[key] = [float(hash_params[key])]
                elif isinstance(hash_params[key], np.ndarray):
                    hash_params[key] = hash_params[key].tolist()

        # a path to the training data hashes as its name: the store is shared by
        # clones and machines whose data folders differ
        hash_params['training_data_filename'] = training_data_name(hash_params['training_data_filename'])

        # observed_idx and tikh_range: arrays, tuples and lists all become lists
        hash_params = convert_to_python_type(hash_params)
        for key in HASHED_IF_NOT_DEFAULT:
            if hash_params[key] == convert_to_python_type(LEGACY_CONVENTIONS.get(key, getattr(ESNConfig, key))):
                del hash_params[key]

        # A hyperparameter that is not searched is a setting, not a training result: its value keys the network.
        # rho and sigma_in are fixed when not in hyperparameters_to_optimize; tikh is picked from tikh_range
        # whenever a search runs, and fixed only without one. The default search adds nothing, so the hashes of
        # the networks cached before this rule hold.
        searched = list(convert_to_python_type(self.hyperparameters_to_optimize) or [])
        fixed = [k for k in ('rho', 'sigma_in', 'sigma_param') if k not in searched] + ([] if searched else ['tikh'])
        for key in fixed:
            value = getattr(self, key, None)
            if value is not None:
                hash_params[f'fixed_{key}'] = float(np.ravel(value)[0])
        return hash_params

    @staticmethod
    def _digest(hash_params):
        """The first 16 hex characters of the SHA256 digest of `hash_params` (JSON, sorted keys)."""
        return hashlib.sha256(json.dumps(hash_params, sort_keys=True).encode()).hexdigest()[:16]


    @classmethod
    def from_esn_model(cls, esn_model):
        """Create a config from an existing `ESN_model` instance."""
        return cls(**cls._init_config_dict(esn_model))


    @classmethod
    def from_init_params(cls, **kwargs):
        """Create a config from initialization parameters (before training).

        Useful for checking if a matching config already exists.

        Parameters
        ----------
        **kwargs
            ESN initialization parameters.

        Returns
        -------
        ESNConfig
        """

        config_dict = {key: kwargs[key] for key in INIT_KEYS if key in kwargs}
        # the fixed values of the hyperparameters that are not searched key the network (see to_hash)
        config_dict.update({key: kwargs[key] for key in ('rho', 'sigma_in', 'sigma_param', 'tikh') if key in kwargs})

        # If N_test, N_val, N_train are not provided, compute them from t_test, t_val, t_train and dt
        for key in ['train', 'val', 'test']:
            if f'N_{key}' in kwargs.keys():
                config_dict[f't_{key}'] = kwargs[f'N_{key}'] * kwargs['dt']

        init_config = cls(**config_dict)
        # the convention that the network is trained with (echostatenetwork <= 0.6.0 has one)
        init_config.scale_bias_in = trained_scale_bias_in(init_config.scale_bias_in)
        if init_config.N_dim == 0:
            assert 'data' in kwargs and kwargs['data'] is not None, "N_dim not provided and data not available to infer it."
            init_config.N_dim = kwargs['data'].shape[1]

        if init_config.observed_idx is None:
            init_config.observed_idx = list(range(init_config.N_dim))

        return init_config


    def to_esn_model(self):
        """Create the pre-trained `ESN_model` this config (with its trained matrices) describes."""
        if self.N_dim is None:
            raise ValueError("N_dim must be set to load a pre-trained model")

        config = asdict(self)
        config['y0'] = np.zeros((self.N_dim, 1))  # Dummy initial state
        if has_intercept():
            # the stored Wout is [Wout; constant row], which multiplies [r; bias_out]: the setter of Wout takes the
            # last row as the intercept, once it carries bias_out, which the network has no more.
            bias_out = float(np.ravel(config.pop('bias_out'))[0])
            if config['Wout'] is not None and bias_out != 1.0:
                config['Wout'] = np.array(config['Wout'], dtype=float)
                config['Wout'][-1] *= bias_out
        # Parametric fields are absent/empty on non-parametric networks (and on
        # stores saved before they existed) — drop them so ESN_model's defaults apply
        for key in PARAM_NORM_KEYS:
            if convert_to_python_type(config[key]) == convert_to_python_type(getattr(ESNConfig, key)):
                del config[key]
        for key in ('param_names', 'param_values', 'param_labels', 'input_parameters'):
            v = config.get(key)
            if v is None or (np.ndim(v) == 0 and not np.any(v)) or \
                    (np.ndim(v) > 0 and len(v) == 0):
                config.pop(key, None)
        return ESN_model(**config)


    def save(self, save_dir:  Path):
        """Save configuration to YAML file."""
        filepath = Path(save_dir / "esn_config.yaml")
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Get initial parameters only
        config_dict = ESNConfig._init_config_dict(self)

        # Convert tuples to lists for YAML compatibility and  sort keys for consistent ordering in YAML file
        config_dict = convert_to_python_type(config_dict)
        config_dict = dict(sorted(config_dict.items()))# type: ignore


        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        # Print the last 3 folders and name of the save directory for confirmation
        logger.debug('Configuration saved to .../%s/esn_config.yaml', '/'.join(list(save_dir.parts)[-3:]))


    @classmethod
    def load(cls, load_dir: Path, verbose=1):
        """Load configuration from YAML file."""
        with open(Path(load_dir / 'esn_config.yaml')) as f:
            config_dict = yaml.safe_load(f)

        for key in config_dict.keys():
            if config_dict[key] == "none":
                config_dict[key] = None

        # a configuration saved before the conventions were options is that of a network trained with the former ones
        config_dict = {**{key: LEGACY_CONVENTIONS[key] for key in ('scale_bias_in', 'regularize_bias_out')},
                       **config_dict}

        if verbose:
            logger.debug('Configuration loaded from .../%s/esn_config.yaml', '/'.join(list(load_dir.parts)[-3:]))
        return cls(**config_dict)



    @staticmethod
    def save_trained_matrices(esn_model, save_dir: Path):
        """
        Save large matrices and other metadata defined after training the network.
        """
        data_to_save = {key: getattr(esn_model, key, False) for key in TRAINED_KEYS }
        # one layout of the store for every echostatenetwork: the constant of the readout is the last row of Wout
        data_to_save['Wout'] = esn_model._readout_matrix()
        # the input bias at full precision: the configuration file rounds it, and a network that was not given one
        # is trained with a value taken from the data, which the reloaded network must run with to the last bit
        data_to_save['bias_in'] = np.asarray(esn_model.bias_in, dtype=float)

        np.savez_compressed(save_dir / 'trained_matrices.npz', **data_to_save)
        logger.debug('Trained matrices saved to .../%s/trained_matrices.npz', '/'.join(list(save_dir.parts)[-3:]))


    @staticmethod
    def load_trained_matrices(save_dir: Path):
        """Load saved matrices from a single ``.npz`` file.

        Parameters
        ----------
        save_dir : Path
            Directory containing the ``.npz`` file (e.g. ``'saved_configs/esn_abc123/'``).

        Returns
        -------
        dict
            Loaded matrices.
        """
        trained_dict = {}
        with np.load(Path(save_dir) / 'trained_matrices.npz', allow_pickle=True) as data:
            for name in TRAINED_KEYS:
                item = data.get(name)
                if isinstance(item, np.ndarray) and item.dtype == object:
                    item = item.item()

                trained_dict[name] = item
            if 'bias_in' in data:                   # absent from the stores written before it was saved here
                trained_dict['bias_in'] = data['bias_in']

        return trained_dict


#: the hashed (and saved) settings: every field but the trained ones and the run options
INIT_KEYS = [f.name for f in fields(ESNConfig)
             if f.name not in (*TRAINED_KEYS, 'plot_training', 'noise_type')]


# Convenience functions
def save_esn_model_to_config(esn_model, save_dir: str | Path =  ESN_STORE, name: str | None = None):
    """Save an ESN model's configuration and trained matrices to disk.

    Parameters
    ----------
    esn_model : ESN_model
        Instance to save.
    save_dir : str or Path, optional
        Directory to save the configuration under.
    name : str, optional
        Subdirectory name for the saved files. Defaults to the config hash.

    Returns
    -------
    tuple
        ``(config, save_path)``.
    """

    if isinstance(save_dir, str):
        save_dir = Path(save_dir)


    config = ESNConfig.from_esn_model(esn_model)

    if name is None:
        name = f"{config.to_hash()}"

    save_path = save_dir / name
    config.save(save_path)

    ESNConfig.save_trained_matrices(esn_model, save_path)

    return config, save_path



def load_esn_model_from_config(q: str, load_dir: str | Path = ESN_STORE):
    """Load the `ESN_model` saved under the config hash `q` in `load_dir`, or
    return `None` if there is none."""
    matching_path = find_matching_config(load_dir, q)
    if not matching_path:
        logger.debug('No matching config %s found in %s.', q, load_dir)
        return None

    config = ESNConfig.load(matching_path)
    vars(config).update(ESNConfig.load_trained_matrices(matching_path))
    return config.to_esn_model()


#: `model_config.find_matching_config` on the ESN store's file name
find_matching_config = partial(_find_matching_config, config_file="esn_config.yaml")


def auto_load_or_create(config_dir: str | Path = ESN_STORE,
                        auto_save: bool = True,
                        force_create: bool = False,
                        query_hash: str | None = None,
                        data: np.ndarray | None = None,
                        **kwargs):
    """Load a matching saved config, or create and train a new `ESN_model`.

    Parameters
    ----------
    config_dir : str or Path, optional
        Directory to search for/save configs.
    auto_save : bool, optional
        If True, save newly trained models. Default True.
    force_create : bool, optional
        If True, skip the search and always train a new model. Default False.
    query_hash : str, optional
        Hash to search for directly, skipping config-based hash computation.
    data : np.ndarray, optional
        Training data.
    **kwargs
        ESN initialization parameters.

    Returns
    -------
    ESN_model
        Loaded or newly trained model.
    """


    # If query_hash is provided, skip config creation and search directly for matching hash
    if query_hash is None:
        initial_params = kwargs.copy()
        initial_params['data'] = data
        query_config = ESNConfig.from_init_params(**initial_params)
        query_hash = query_config.to_hash()

    # Try to find matching config
    if force_create:
        logger.debug('Force creating new model (skipping search)...')
    elif find_matching_config(config_dir, query_hash):
        return load_esn_model_from_config(query_hash, config_dir)

    # Train new model
    model = ESN_model(data=data, **kwargs)
    if auto_save:
        logger.debug('Saving new model to %s', config_dir)
        save_esn_model_to_config(model, save_dir=config_dir, name=f"{query_hash}")
    return model


def rekey_esn_store(store_dir: str | Path = ESN_STORE, dry_run: bool = False):
    """Move the networks stored under the hash of a path to their training data to the
    hash of its name, where the store lookup now finds them.

    Before the training data were keyed by name, a network trained with an absolute
    `training_data_filename` (as the experiment driver and the tutorials pass it) was
    stored under a hash of that path, which no other clone or machine computes. An
    entry moves when its stored configuration gives its directory name under the old
    rule; an entry stored under a query hash that its configuration does not give, or
    whose new hash is taken (by a copy of the same network, say), stays where it is,
    with a warning.

    Parameters
    ----------
    store_dir : str or Path, optional
        The store, `ESN_STORE` by default.
    dry_run : bool, optional
        If True, report the moves without renaming anything.

    Returns
    -------
    list of tuple
        ``(old_name, new_name)`` of every entry moved (to move, if `dry_run`).
    """
    store_dir = Path(store_dir)
    moves, taken = [], set()
    for path in sorted(store_dir.iterdir()) if store_dir.is_dir() else ():
        if not find_matching_config(store_dir, path.name):
            continue
        config = ESNConfig.load(path, verbose=0)
        hash_params = config._hashed_params()
        new_name = ESNConfig._digest(hash_params)
        stored = convert_to_python_type(config.training_data_filename)
        if new_name == path.name or stored == hash_params['training_data_filename']:
            continue  # keyed by the name already
        hash_params['training_data_filename'] = stored
        if ESNConfig._digest(hash_params) != path.name:
            logger.warning('%s: stored under a hash its configuration does not give; left in place',
                           path.name)
            continue
        if new_name in taken or (store_dir / new_name).exists():
            logger.warning('%s: %s already holds this network; left in place', path.name, new_name)
            continue
        if not dry_run:
            path.rename(store_dir / new_name)
        taken.add(new_name)
        moves.append((path.name, new_name))
    return moves
