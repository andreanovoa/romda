"""Saved configurations of the trained models that implement `Configurable`
(`romda.models.data_driven.configurable`).

Each saved model is a directory named by the hash of its configuration:

    <store>/<hash>/model_config.yaml      the class and its options
    <store>/<hash>/trained_matrices.npz   the trained arrays

The echo state network keeps its own store, `romda.models.data_driven.rnns.esn_config`
(`ESNConfig`, ``results/esn_configs``); this module is the generic counterpart for
every other trained model (``results/model_configs``). Written by Lukas Campbell
(autoencoder-rom, 2026).

Typical usage example:

  model = auto_load_or_create(ModelClass, data=X, training_data_filename="wake.h5", n_latent=8)
  config, path = save_model_to_config(lstm, training_data_filename="wake.h5")
  lstm = load_model_from_config(q=config.to_hash())

Ported from autoencoder-rom; co-author: Lukas Campbell (2026).
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeVar, cast

import numpy as np
import numpy.typing as npt
import yaml
from romda.utils import get_project_root

from .configurable import Configurable

__all__ = [
    "MODEL_STORE",
    "ModelConfig",
    "auto_load_or_create",
    "find_matching_config",
    "list_saved_configs",
    "load_model_from_config",
    "save_model_to_config",
]

#: Default store, ``results/model_configs`` in the project (next to the ESN store).
try:
    MODEL_STORE = Path(get_project_root()) / "results" / "model_configs"
except FileNotFoundError:  # imported from outside a project tree
    MODEL_STORE = Path.cwd() / "results" / "model_configs"

M = TypeVar("M", bound=Configurable)
CONFIG_FILE = "model_config.yaml"
TRAINED_FILE = "trained_matrices.npz"
#: Digits floats are rounded to before hashing.
_FLOAT_DIGITS = 12


@dataclass
class ModelConfig:
    """The class and options of a saved model.

    Attributes:
      model_class: Import path of the model class, `module:QualifiedName`.
      options: Constructor options, as `Configurable.config_options` reports
        them.
      training_data_filename: Name of the data the model was trained on.
    """

    model_class: str
    options: dict[str, Any] = field(default_factory=dict[str, Any])
    training_data_filename: str | None = None

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_model(cls, model: Configurable, training_data_filename: str | None = None) -> ModelConfig:
        """Creates the configuration of a trained model.

        Args:
          model: The model.
          training_data_filename: Name of the data the model was trained on.

        Returns:
          The configuration.

        Raises:
          TypeError: If the model's class cannot be stored.
        """
        model_class = type(model)
        _check_configurable(model_class)
        return cls(_class_path(model_class), model.config_options(), training_data_filename)

    @classmethod
    def from_init_params(
        cls,
        model_class: type[Configurable],
        data: Any = None,
        training_data_filename: str | None = None,
        **options: Any,
    ) -> ModelConfig:
        """Creates the configuration a model built from options would have.

        Args:
          model_class: The model class.
          data: The training data, for options the class infers from it.
          training_data_filename: Name of the training data.
          **options: Constructor options. Missing options take their defaults.

        Returns:
          The configuration.

        Raises:
          TypeError: If `model_class` cannot be stored.
        """
        _check_configurable(model_class)
        return cls(_class_path(model_class), model_class.resolve_options(data, **options), training_data_filename)

    def to_hash(self) -> str:
        """Hashes the configuration.

        Floats are rounded to 12 decimals (12 significant digits where that
        would zero them) and tuples compared as lists.

        Returns:
          The first 16 hexadecimal digits of the SHA-256 digest.
        """
        text = json.dumps(_canonical(asdict(self)), sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    # ── Models ────────────────────────────────────────────────────────────────

    def resolve_class(self) -> type[Configurable]:
        """Imports the model class.

        Returns:
          The class.

        Raises:
          TypeError: If the import path does not name a class that can be
            stored.
        """
        module_name, _, qualname = self.model_class.partition(":")
        obj: Any = importlib.import_module(module_name)
        for part in qualname.split("."):
            obj = getattr(obj, part)
        if not isinstance(obj, type) or not issubclass(obj, Configurable):
            raise TypeError(f"{self.model_class} is not a Configurable class")
        _check_configurable(obj)
        return obj

    def to_model(self, trained: Mapping[str, npt.NDArray[Any]], **overrides: Any) -> Configurable:
        """Rebuilds the trained model.

        Args:
          trained: The trained arrays.
          **overrides: Options to use instead of the stored ones.

        Returns:
          The trained model.
        """
        return self.resolve_class().from_trained(self.options, trained, **overrides)

    # ── Files ─────────────────────────────────────────────────────────────────

    def save(self, save_dir: Path) -> Path:
        """Writes the configuration to `model_config.yaml`.

        Args:
          save_dir: Directory to write into. Created if missing.

        Returns:
          The path of the file.
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / CONFIG_FILE
        with path.open("w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)
        return path

    @classmethod
    def load(cls, load_dir: Path) -> ModelConfig:
        """Reads a configuration written by `save`.

        Args:
          load_dir: Directory holding `model_config.yaml`, or the file itself.

        Returns:
          The configuration.
        """
        path = load_dir if load_dir.suffix == ".yaml" else load_dir / CONFIG_FILE
        with path.open() as f:
            stored: dict[str, Any] = yaml.safe_load(f)
        return cls(**stored)

    @staticmethod
    def save_trained_matrices(model: Configurable, save_dir: Path) -> Path:
        """Writes a model's trained arrays to `trained_matrices.npz`.

        Args:
          model: The trained model.
          save_dir: Directory to write into. Created if missing.

        Returns:
          The path of the file.
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / TRAINED_FILE
        np.savez_compressed(path, allow_pickle=False, **model.trained_arrays())
        return path

    @staticmethod
    def load_trained_matrices(save_dir: Path) -> dict[str, npt.NDArray[Any]]:
        """Reads arrays written by `save_trained_matrices`.

        Args:
          save_dir: Directory holding `trained_matrices.npz`.

        Returns:
          The arrays, by name.
        """
        with np.load(save_dir / TRAINED_FILE, allow_pickle=False) as z:
            return {name: z[name] for name in z.files}


# ── Store ─────────────────────────────────────────────────────────────────────


def save_model_to_config(
    model: Configurable,
    save_dir: str | Path = MODEL_STORE,
    name: str | None = None,
    training_data_filename: str | None = None,
) -> tuple[ModelConfig, Path]:
    """Saves a trained model's configuration and arrays.

    Args:
      model: The trained model.
      save_dir: The store.
      name: Directory name within the store. `None` uses the configuration
        hash.
      training_data_filename: Name of the data the model was trained on.

    Returns:
      The configuration and the directory it was saved in.

    Raises:
      TypeError: If the model's class cannot be stored.
      RuntimeError: If the model is not trained.
    """
    config = ModelConfig.from_model(model, training_data_filename)
    path = Path(save_dir) / (config.to_hash() if name is None else name)
    ModelConfig.save_trained_matrices(model, path)
    config.save(path)
    return config, path


def load_model_from_config(
    q: str | None = None,
    config: ModelConfig | None = None,
    load_dir: str | Path = MODEL_STORE,
    **overrides: Any,
) -> Configurable | None:
    """Loads a saved model by name or by configuration.

    Args:
      q: Directory name within the store, such as a configuration hash. Takes
        precedence over `config`.
      config: Configuration whose hash names the directory.
      load_dir: The store.
      **overrides: Options to use instead of the stored ones, such as `device`.

    Returns:
      The trained model, or `None` if the store holds no such model.

    Raises:
      ValueError: If neither `q` nor `config` is given.
    """
    if q is None:
        if config is None:
            raise ValueError("pass q or config to load a model")
        q = config.to_hash()
    path = find_matching_config(load_dir, q)
    if path is None:
        return None
    return ModelConfig.load(path).to_model(ModelConfig.load_trained_matrices(path), **overrides)


def find_matching_config(search_dir: str | Path, query_hash: str, config_file: str = CONFIG_FILE) -> Path | None:
    """Finds a saved model by directory name.

    Args:
      search_dir: The store.
      query_hash: Directory name, such as a configuration hash.
      config_file: Name of the configuration file, `model_config.yaml` or
        `esn_config.yaml`.

    Returns:
      The directory, or `None` if it does not hold a complete saved model.
    """
    path = Path(search_dir) / query_hash
    if (path / config_file).is_file() and (path / TRAINED_FILE).is_file():
        return path
    return None


def auto_load_or_create(
    model_class: type[M],
    data: Any,
    config_dir: str | Path = MODEL_STORE,
    auto_save: bool = True,
    force_create: bool = False,
    query_hash: str | None = None,
    training_data_filename: str | None = None,
    **options: Any,
) -> M:
    """Loads a saved model with matching options, or trains and saves a new one.

    Args:
      model_class: The model class.
      data: The training data.
      config_dir: The store.
      auto_save: Save a newly trained model.
      force_create: Train a new model even if one is saved.
      query_hash: Directory name to look up. `None` uses the hash of the
        configuration `options` and `training_data_filename` give.
      training_data_filename: Name of the training data.
      **options: Constructor options.

    Returns:
      The trained model.

    Raises:
      TypeError: If `model_class` cannot be stored, or the saved model is of
        another class.
    """
    if query_hash is None:
        config = ModelConfig.from_init_params(model_class, data, training_data_filename, **options)
        query_hash = config.to_hash()

    path = None if force_create else find_matching_config(config_dir, query_hash)
    if path is not None:
        model = ModelConfig.load(path).to_model(ModelConfig.load_trained_matrices(path))
        if not isinstance(model, model_class):
            raise TypeError(f"{path} holds a {type(model).__name__}, not a {model_class.__name__}")
        return model

    model = model_class.from_data(data, **options)
    if auto_save:
        save_model_to_config(model, config_dir, query_hash, training_data_filename)
    return model


def list_saved_configs(search_dir: str | Path = MODEL_STORE) -> list[dict[str, Any]]:
    """Lists the saved models in a store.

    Args:
      search_dir: The store.

    Returns:
      For each saved model, its `directory`, `name`, `model_class`, and
      `training_data_filename`. Empty if the store does not exist.
    """
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(search_dir).glob(f"*/{CONFIG_FILE}")):
        config = ModelConfig.load(path)
        rows.append(
            {
                "directory": path.parent,
                "name": path.parent.name,
                "model_class": config.model_class,
                "training_data_filename": config.training_data_filename,
            }
        )
    return rows


# ── Helpers ───────────────────────────────────────────────────────────────────


def _class_path(model_class: type) -> str:
    """Returns the import path of a class, `module:QualifiedName`."""
    return f"{model_class.__module__}:{model_class.__qualname__}"


def _check_configurable(model_class: type[Configurable]) -> None:
    """Raises if a class cannot be stored.

    Raises:
      TypeError: If `model_class.configurable` is false.
    """
    if not model_class.configurable:
        raise TypeError(f"{model_class.__name__} cannot be stored; store its components instead")


def _canonical(value: Any) -> Any:
    """Normalizes a configuration value for hashing.

    Args:
      value: A configuration value.

    Returns:
      `value` with floats rounded and tuples as lists.
    """
    if isinstance(value, float):
        rounded = round(value, _FLOAT_DIGITS)  # significant digits only where rounding would zero the value
        return rounded if rounded or not value else float(f"{value:.{_FLOAT_DIGITS}g}")
    if isinstance(value, list | tuple):
        return [_canonical(v) for v in cast(list[Any] | tuple[Any, ...], value)]
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in cast(dict[Any, Any], value).items()}
    return value
