import hashlib
import json
import logging
from dataclasses import asdict, dataclass, fields
from functools import partial
from pathlib import Path

import numpy as np
import yaml
from romda.utils import convert_to_python_type

from ..model_config import MODEL_STORE
from ..model_config import find_matching_config as _find_matching_config
from .esn import ESN_model

logger = logging.getLogger(__name__)

#: the trained-network store is a project-local cache under results/ (gitignored)
ESN_STORE = MODEL_STORE.with_name('esn_configs')

#: INIT_KEYS added after networks were cached: `to_hash` drops each one at its
#: default, so every hash taken before they joined still finds its network
HASHED_IF_NOT_DEFAULT = ('hyperparameters_to_optimize', 'bias_in_range', 'regularize_bias_out',
                         'stability_steps', 'bho_noise', 'grid_midpoints',
                         'optimize_parameter_normalization', 'param_shift_range',
                         'param_norm_range')

#: left unset when building a network at their default: EchoStateNetwork then searches the
#: parameter normalization whenever input_parameters are given, over ranges sized from the
#: parameters, which the parametric networks of stores without these keys rely on
PARAM_NORM_KEYS = ('optimize_parameter_normalization', 'param_shift_range', 'param_norm_range')

TRAINED_KEYS = ['norm', 'shift',
                'rho', 'sigma_in', 'tikh',
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
    Win_type: str = 'sparse'
    N_wash: int = 5
    upsample:  int = 1
    bias_in: float = 0.1
    bias_out: float = 1.0
    regularize_bias_out: bool = True
    connect: float = 3

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
    tikh_range: tuple[float, ... ] = (1e-6, 1e-9, 1e-12)
    bias_in_range: tuple[float, float] = (0.0, 2.0)
    hyperparameters_to_optimize: tuple[str, ...] = ('rho', 'sigma_in', 'tikh')
    # Bayesian search options: the defaults of echostatenetwork 0.2.3
    stability_steps: int = 0
    bho_noise: float | str = 1e-10
    grid_midpoints: bool = False
    # parametric ESN: search of the parameter shift and scale (echostatenetwork defaults;
    # `to_esn_model` leaves them unset at these values, see PARAM_NORM_KEYS)
    optimize_parameter_normalization: bool = False
    param_shift_range: tuple[float, float] = (-1.0, 1.0)
    param_norm_range: tuple[float, float] = (0.1, 10.0)

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
    tikh: float | None = None
    Win: np.ndarray | None = None
    W: np.ndarray | None = None
    Wout: np.ndarray | None = None
    validation_data: np.ndarray | None = None

    @staticmethod
    def _init_config_dict(case):
        # default for keys an older echostatenetwork does not define
        return {key: getattr(case, key, getattr(ESNConfig, key)) for key in INIT_KEYS}

    def to_hash(self):
        """Compute a hash of the initial configuration parameters.

        Uses INITIAL parameters only (before any optimization), so configs match
        even if hyperparameter optimization produces slightly different results
        due to randomness.

        Returns
        -------
        str
            The first 16 hex characters of the SHA256 digest of the config.
        """

        hash_params = ESNConfig._init_config_dict(self)

        # Normalize types to ensure consistent hashing
        # Convert all numpy arrays and lists to consistent format
        for key in ['bias_in', 'bias_out']:
            if hash_params[key] is not None:
                if isinstance(hash_params[key], (float, int)):
                    hash_params[key] = [float(hash_params[key])]
                elif isinstance(hash_params[key], np.ndarray):
                    hash_params[key] = hash_params[key].tolist()

        # observed_idx and tikh_range: arrays, tuples and lists all become lists
        hash_params = convert_to_python_type(hash_params)
        for key in HASHED_IF_NOT_DEFAULT:
            if hash_params[key] == convert_to_python_type(getattr(ESNConfig, key)):
                del hash_params[key]

        # Convert to JSON string (sorted keys for consistency)
        hash_string = json.dumps(hash_params, sort_keys=True)

        # Compute SHA256 hash
        return hashlib.sha256(hash_string.encode()).hexdigest()[:16]


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

        # If N_test, N_val, N_train are not provided, compute them from t_test, t_val, t_train and dt
        for key in ['train', 'val', 'test']:
            if f'N_{key}' in kwargs.keys():
                config_dict[f't_{key}'] = kwargs[f'N_{key}'] * kwargs['dt']

        init_config = cls(**config_dict)
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

        if verbose:
            logger.debug('Configuration loaded from .../%s/esn_config.yaml', '/'.join(list(load_dir.parts)[-3:]))
        return cls(**config_dict)



    @staticmethod
    def save_trained_matrices(esn_model, save_dir: Path):
        """
        Save large matrices and other metadata defined after training the network.
        """
        data_to_save = {key: getattr(esn_model, key, False) for key in TRAINED_KEYS }

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
