"""
Created on Wed May 11 09:45:48 2022

@author: Andrea Nóvoa @andrea_novoa
"""
import functools
import gc
import hashlib
import logging
import os
import pickle
import re
import urllib.request
import zipfile
from functools import partial
from pathlib import Path

import matplotlib.backends.backend_pdf as plt_pdf
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage

# shared model-layer helpers now live in (and are re-exported from) dynamodels
from dynamodels.utils import (  # noqa: F401
    allowed_kwargs_for_func,
    interpolate,
    load_from_mat_file,
    mean_vector_to_ensemble,
    normalized_alpha,
    normalized_time,
    normalized_y,
    save_to_mat_file,
)
from dynamodels.utils import create_dataset as _dm_create_dataset
from matplotlib import colors
from matplotlib.animation import FuncAnimation

logger = logging.getLogger(__name__)


def noise_generator(seed=0) -> np.random.Generator:
    """A generator of measurement noise for `seed`.

    Its stream is keyed apart from ``np.random.default_rng(seed)``, the stream of a `Model` (and of its
    ensemble) with the same seed, so that the noise of the observations never repeats the random
    numbers of the filter.
    """
    return np.random.default_rng([seed, 1])


#: The shared generator of the measurement noise: every noise draw of the observation operators
#: (`romda.observations.Observations` and, unless given another generator, `add_noise_to_flow`)
#: comes from it. It is seeded with 0 at import, so a script draws the same noise on every run;
#: `seed_noise` restarts it from another seed (`romda.run.run_config` does so with the
#: seed of the run).
noise_rng = noise_generator(0)


def seed_noise(seed=0) -> None:
    """Restart the shared noise generator `noise_rng` from `seed`.

    The generator is reset in place, so every module that holds `noise_rng` draws the new sequence.
    """
    noise_rng.bit_generator.state = noise_generator(seed).bit_generator.state


def given(**options) -> dict:
    """The options that were given (not None), to forward as keyword arguments; None keeps the class default."""
    return {key: value for key, value in options.items() if value is not None}


def canonical_name(name, names: dict, label: str = 'type') -> str:
    """The canonical value of `name` in `names`, ignoring case, hyphens, underscores and spaces."""
    key = ''.join(c for c in str(name).lower() if c.isalnum())
    if key not in names:
        raise ValueError(f"{label}={name!r}: expected one of {sorted(set(names.values()))}.")
    return names[key]


def convert_to_python_type(obj, *, float_ndigits=12):
    """Convert numpy types to native Python types, with canonical float rounding."""
    if obj is None:
        return "none"

    if isinstance(obj, np.generic):
        if np.issubdtype(type(obj), np.integer):
            return int(obj)
        elif np.issubdtype(type(obj), np.floating):
            return round(float(obj), float_ndigits)
        elif np.issubdtype(type(obj), np.bool_):
            return bool(obj)
        elif np.issubdtype(type(obj), np.complexfloating):
            c = complex(obj)
            return (round(c.real, float_ndigits), round(c.imag, float_ndigits))
        else:
            return obj.item()

    elif isinstance(obj, float):
        return round(obj, float_ndigits)

    elif isinstance(obj, np.ndarray):
        return [convert_to_python_type(x, float_ndigits=float_ndigits) for x in obj.tolist()]
    elif isinstance(obj, tuple):
        return [convert_to_python_type(item, float_ndigits=float_ndigits) for item in obj]
    elif isinstance(obj, list):
        return [convert_to_python_type(item, float_ndigits=float_ndigits) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_python_type(v, float_ndigits=float_ndigits) for k, v in obj.items()}

    # if Path, change to sttring
    elif isinstance(obj, os.PathLike):
        return str(obj)

    return obj


def cut_signals(t, *signals, min_time=None, max_time=None):
    """Trim time array `t` and any number of aligned `signals` to ``[min_time, max_time]``."""
    i0 = 0 if min_time is None else np.argmin(abs(t - min_time))
    i1 = len(t) - 1 if max_time is None else np.argmin(abs(t - max_time))
    t_cut = t[i0:i1]
    signals_cut = [sig[i0:i1].copy() if sig is not None else None for sig in signals]
    return t_cut, signals_cut


def add_noise_to_flow(U, noise_level=0.05, noise_type="gauss", spatial_smooth=0., rng=None):
    """Add noise to a velocity field.

    Parameters
    ----------
    U : np.ndarray
        3D array ``(Nt, Nx, Ny)`` for a single velocity component, or 4D array
        ``(2, Nt, Nx, Ny)`` for both components (U, V). NaNs (e.g. inside a
        cylinder) are excluded from the noise-amplitude scaling.
    noise_level : float, optional
        Standard deviation of the noise, as a fraction of the maximum absolute
        velocity. Default 0.05.
    noise_type : str, optional
        Only ``'gauss'`` (Gaussian white noise) is supported. Default ``'gauss'``.
    spatial_smooth : float, optional
        Standard deviation for Gaussian spatial smoothing of the noise (0
        disables smoothing). Default 0.
    rng : np.random.Generator, optional
        Generator of the noise. Default: the shared `noise_rng` (see `seed_noise`).

    Returns
    -------
    np.ndarray
        Noisy velocity field, same shape as `U`.
    """
    if noise_type != "gauss":
        raise ValueError(f"noise_type={noise_type!r}: only 'gauss' is supported.")
    # mask nan values (e.g., inside cylinder) to avoid affecting noise scaling
    fluid_mask = ~np.isnan(U)
    U = np.where(fluid_mask, U, 0.)

    if U.ndim == 4:
        assert U.shape[0] == 2, "Expected first dimension of size 2 for (U, V) components."
    elif U.ndim == 3:
        U = U[np.newaxis, ...]  # Add component dimension for uniform processing
    else:
        raise ValueError("Input U must be a 3D array (Nt x Nx x Ny) or a 4D array (2 x Nt x Nx x Ny).")

    # Compute noise amplitude
    noise_amp = noise_level * np.max(np.abs(U[fluid_mask]))
    noise_U = (noise_rng if rng is None else rng).normal(scale=noise_amp, size=U.shape)

    # Apply optional spatial smoothing
    if spatial_smooth > 0:
        sigma = (0, 0, spatial_smooth, spatial_smooth)
        noise_U = ndimage.gaussian_filter(noise_U, sigma=sigma)


    U_noisy = U + noise_U
    U_noisy[~fluid_mask] = np.nan

    return U_noisy



def get_project_root(root='.'):
    """The first of `root` and its parents that contains a ``src`` folder, else the project that holds romda
    itself (the clone of an editable install: the folder whose ``src/romda`` is this package).

    The search from `root` (the working directory by default) keeps a script inside a project, a worktree or a
    user project, on that project; the fallback keeps a script run from anywhere else (a notebook in another
    folder, a run script called by its path) on romda's own clone, so that the model stores under
    ``results/`` are shared instead of falling back on the working directory.

    The filesystem root is not searched: its system folders would otherwise mark
    '/' as the project root of any directory.
    """
    start = Path(root).resolve()
    for p in [start, *start.parents]:
        if p != p.parent and (p / 'src').is_dir():
            return str(p)
    package_root = Path(__file__).resolve().parents[2]          # <root>/src/romda/utils.py
    if (package_root / 'src' / 'romda').is_dir():
        return str(package_root)
    raise FileNotFoundError(f"Project root directory not found: no folder of {start} or its parents holds a src "
                            "folder, and romda is not installed from a clone (src/romda).")


def set_working_directories(subfolder='', root='.'):
    """Resolve the data/results/figures folders for a given case `subfolder`.

    Parameters
    ----------
    subfolder : str, optional
        Case-specific subfolder appended to each base directory.
    root : str, optional
        Directory to start searching for the project root from (see
        `get_project_root`). Default ``'.'``.

    Returns
    -------
    tuple of str
        ``(data_folder, results_folder, figs_folder)``.
    """


    if subfolder[-1] != '/':
        subfolder += '/'

    # Get project root, i.e., where real_time_DA is
    project_root = get_project_root(root)

    #  Set results and fgures folders
    if 'tutorials' in os.getcwd():
        results_folder = f'{project_root}/results/tutorials/{subfolder}'
        figs_folder = f'{project_root}/docs/figs/{subfolder}'
    else:
        results_folder = f'{project_root}/results/{subfolder}'
        figs_folder = f'{project_root}/results/figs/{subfolder}'

    #  Set data folder. Note: some data is not provided in the repository.

    data_folder = f'{project_root}/data/{subfolder}'

    return data_folder, results_folder, figs_folder


def colour_noise(dims, noise_colour='pink', beta=2):
    r"""Generate a spectral filter for coloured noise, for use with `np.fft.irfft`.

    Parameters
    ----------
    dims : int or sequence of int
        Number of time steps or spatial points along each dimension.
    noise_colour : str, optional
        Type of noise: ``'white'`` (flat spectrum), ``'pink'`` ($1/f^\beta$,
        long-range correlation; a trailing number overrides `beta`, e.g.
        ``'pink1.5'``), ``'brown'`` ($1/f^2$), ``'blue'`` ($\sqrt{f}$) or
        ``'violet'`` ($f$). Default ``'pink'``.
    beta : float, optional
        Decay exponent used for ``'pink'`` noise (overridden if `noise_colour`
        has a trailing number). Default 2.

    Returns
    -------
    np.ndarray
        Spectral filter, real-FFT shaped: length ``dims // 2 + 1`` if `dims` is an
        int, or the same shape as `dims` with the last axis halved if `dims` is a
        sequence. The zero-frequency component is zero and each axis has unit RMS.
    """
    noise_colour = noise_colour.lower()
    numbers = re.findall(r'\d+\.*\d*', noise_colour)
    # spectrum |f|^e per axis
    exponents = {'white': 0., 'blue': .5, 'violet': 1.,
                 'pink': -1 / (float(numbers[0]) if numbers else beta), 'brown': -1.}
    e = next((v for k, v in exponents.items() if k in noise_colour), None)
    if e is None:
        raise ValueError(f"Unknown noise type: {noise_colour}")

    if isinstance(dims, int):
        ff_dims = [np.fft.rfftfreq(dims)]
    else:
        ff_dims = [np.fft.fftfreq(Nt) for Nt in dims[:-1]] + [np.fft.rfftfreq(dims[-1])]

    S = 1.
    for ii, f in enumerate(ff_dims):
        with np.errstate(divide='ignore'):
            s = np.where(f == 0, 0., np.abs(f) ** e)
        if np.any(s):
            s = s / np.sqrt(np.mean(s[f != 0] ** 2))
        S = S * s.reshape([-1 if jj == ii else 1 for jj in range(len(ff_dims))])
    return S




def check_valid_file(load_case, params_dict):
    """True if every `params_dict` value equals the same key/attribute of `load_case`; missing keys pass."""
    get = load_case.get if isinstance(load_case, dict) else partial(getattr, load_case)
    return all(np.array_equal(get(key, val), val) for key, val in params_dict.items())


def save_to_pickle_file(filename, *args):
    """Pickle each of `args` sequentially into `filename`, creating parent directories."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        for arg in args:
            pickle.dump(arg, f)


def load_from_pickle_file(filename):
    """Load all objects sequentially pickled into `filename` (see `save_to_pickle_file`).

    Returns
    -------
    object, list or False
        The single unpickled object, a list of objects if more than one was
        stored, or False if `filename` does not exist.
    """
    if not os.path.exists(filename):
        return False

    args = []
    with open(filename, 'rb') as f:
        while True:
            try:
                arg = pickle.load(f)
                args.append(arg)
            except EOFError:
                break
    if len(args) == 1:
        return args[0]
    else:
        return args


def save_figs_to_pdf(pdf_name, figs=None):
    """Save `figs` (default: all open figures) to a multi-page PDF, closing each after saving."""
    pdf_file = plt_pdf.PdfPages(pdf_name)
    if figs is None:
        figs = [plt.figure(ii) for ii in plt.get_fignums()]
    elif not isinstance(figs, list):
        figs = [figs]

    for fig in figs:
        pdf_file.savefig(fig, dpi=300)  # Save figure to PDF
        plt.close(fig)

    pdf_file.close()  # Close results pdf


def correlation(y_true, y_est):
    """Pearson correlation of each member of `y_est` (Nt, Nq, N_ens) with `y_true` (Nt, Nq[, 1]); 0 where undefined."""
    y_true = np.asarray(y_true)
    if y_true.ndim == 2:
        y_true = y_true[..., np.newaxis]
    a = y_true - y_true.mean(axis=0, keepdims=True)
    b = np.asarray(y_est) - np.mean(y_est, axis=0, keepdims=True)
    num = np.sum(a * b, axis=(0, 1))
    den = np.sqrt(np.sum(a ** 2, axis=(0, 1)) * np.sum(b ** 2, axis=(0, 1)))
    return np.divide(num, den, out=np.zeros_like(num), where=den >= 1e-10)


def create_dataset_from_model(model_class, num_lyap_times=300, noise_level=0.02, seed=0, **kwargs):
    """`dynamodels.utils.create_dataset_from_model` cached under this repo's
    ``data/<model class name>/`` folder."""
    data_folder = set_working_directories(f'{model_class.__name__}/')[0]
    return _dm_create_dataset(model_class, data_folder, num_lyap_times=num_lyap_times,
                              noise_level=noise_level, seed=seed, **kwargs)




def record_key(name, record):
    """`name` tagged with a SHA-256 digest of the bytes of `record`.

    The file name of a record fixes its options (noise level, length, seed) but not
    its content: two environments can integrate the same chaotic system to records
    that differ by round-off amplified over the Lyapunov times. Passed as the
    `training_data_filename` of a network, the tag keys the store by the data the
    network actually trained on, so a regenerated record never loads a network
    trained on an older one."""
    record = np.ascontiguousarray(record, dtype=np.float64)
    digest = hashlib.sha256(record.tobytes() + str(record.shape).encode()).hexdigest()
    return f'{name}@{digest[:12]}'


def noisy_copy(clean, noise_level, seed=0):
    """`clean` plus Gaussian noise of `noise_level` times the standard deviation of each component (over the first
    axis), as `dynamodels.utils.create_dataset` draws the noisy record of a dataset. `seed` is a seed or a
    `numpy.random.Generator`, whose draws continue from call to call."""
    rng = np.random.default_rng(seed)
    return clean + rng.normal(scale=noise_level * clean.std(axis=0), size=clean.shape)


def trajectory(model_class, seed=0, n_steps=1000, t_transient=None,
               constructor_defaults=None, **params):
    """`n_steps` on-attractor states of any physical `Model` at the given parameters.

    `constructor_defaults` holds the per-model constructor fallbacks (`dt`, `Nx`, ...); explicit `params` win.
    """
    params = {**(constructor_defaults or {}), **params}
    rng = np.random.default_rng(seed)
    psi0 = np.ravel(model_class(**params).psi0)
    model = model_class(psi0=psi0 + rng.normal(scale=.1, size=psi0.size), **params)

    psi, _ = model.time_integrate(Nt=int(round((t_transient or model.t_transient) / model.dt)))
    model.update_history(psi[-1:], t=np.array([0.]), reset=True)
    psi, _ = model.time_integrate(Nt=n_steps - 1)
    model.update_history(psi, t=None)

    return model.hist[:, :model.Nphi, 0]


def collected(job):
    """A job of a pool of workers that first collects the garbage of the earlier jobs of its worker, as `deploy` does.

    Models and estimators sit in reference cycles (a model and its integrator, a filter and the closures that replace
    its methods), which Python frees only at a collection, and the allocations of numpy do not trigger one: without
    it, a worker of the L96-40s sweeps of the washout-free ESN-DA example kept the ensemble of every job that it had run,
    1.1 GB each, and grew to 40 to 44 GB. Decorate every job that builds a filter without `deploy`."""
    @functools.wraps(job)
    def collecting_job(*args, **kwargs):
        gc.collect()
        return job(*args, **kwargs)
    return collecting_job


def download_zenodo_file(download_url, data_folder='./', filename=None):
    """Download `download_url` into `data_folder` (as `filename`, default the URL's), unless already there.

    A downloaded zip archive is also unzipped in place (see `unzip_file`).
    """
    os.makedirs(data_folder, exist_ok=True)
    filename = filename or download_url.split('/')[-1].split('?')[0]
    file_path = os.path.join(data_folder, filename)
    if os.path.exists(file_path) and (not filename.endswith('.zip') or zipfile.is_zipfile(file_path)):
        logger.debug("File '%s' already exists. Skipping download.", filename)
        return
    logger.info('Downloading %s from Zenodo...', filename)
    urllib.request.urlretrieve(download_url, file_path)
    if zipfile.is_zipfile(file_path):
        unzip_file(file_path, output_folder=data_folder)


def unzip_file(file_path, output_folder=None):
    """Extract a zip archive into `output_folder`, stripping the archive's top-level folder."""
    with zipfile.ZipFile(file_path) as zf:
        members = [m for m in zf.infolist()
                   if not m.is_dir() and not m.filename.startswith(('__MACOSX/', '.DS_Store'))]
        if not members:
            return
        top = Path(members[0].filename).parts[0]
        for m in members:
            m.filename = str(Path(m.filename).relative_to(top))
            zf.extract(m, output_folder)


def get_zenodo_data(record, files, data_folder):
    """Download `files` ({remote name: local name}) and the README of Zenodo `record` into `data_folder`."""
    url = f"https://zenodo.org/records/{record}/files"
    for remote, local in {**files, 'README.md': 'README.md'}.items():
        download_zenodo_file(f'{url}/{remote}?download=1', data_folder, filename=local)


def get_annular_data(data_folder: str | None = None):
    """Download and unzip the annular data from Zenodo if not already present."""
    get_zenodo_data(15609832, {'annular.zip': 'annular_data.zip'},
                    data_folder or set_working_directories('annular/')[0])


def get_wake_data(data_folder: str | None = None, case='circle_re_100'):
    """Download the bluff-body wake flow data from Zenodo if not already present."""
    get_zenodo_data(15623774, {f'{case}.mat': f'{case}.mat'},
                    data_folder or set_working_directories('wakes/')[0])




def load_cylinder_dataset(noise_type = 'gauss', noise_level = 0.1, smoothing = 0.1,
                          root_folder='.', visualize=False):
    """Load (downloading if needed) the cylinder-wake flow dataset, adding noise once and caching the result.

    Downloads the ``circle_re_100`` wake dataset (see `get_wake_data`) on first
    use, adds noise via `add_noise_to_flow`, and caches the noisy/clean pair to a
    ``.mat`` file so subsequent calls with the same parameters just reload it.

    Parameters
    ----------
    noise_type : str, optional
        Forwarded to `add_noise_to_flow`. Default ``'gauss'``.
    noise_level : float, optional
        Forwarded to `add_noise_to_flow`. Default 0.1.
    smoothing : float, optional
        Spatial smoothing, forwarded to `add_noise_to_flow` as `spatial_smooth`.
        Default 0.1.
    root_folder : str, optional
        Passed to `set_working_directories` to locate the data/results folders.
        Default ``'.'``.
    visualize : bool, optional
        If True, also generate a comparison animation via `visualize_flow_data`.
        Default False.

    Returns
    -------
    all_data : np.ndarray
        Clean velocity data, shape ``(Nt, Nx, Ny, 2)``.
    all_data_noisy : np.ndarray
        Noisy velocity data, same shape.
    new_results_dir : str
        Directory where the cached dataset (and optional animation) is stored.
    """
    data_folder, results_folder = set_working_directories('wakes/', root=root_folder)[:2]


    new_results_dir = f'{results_folder}/data_noise{noise_level}{noise_type}_smoothing{smoothing}/'

    # new_data_dir = f'{data_folder}/data_noise{noise_level}{noise_type}_smoothing{smoothing}/'
    os.makedirs(new_results_dir, exist_ok=True)

    data_name = f'{new_results_dir}00_data.mat'

    if not os.path.exists(data_name):

        if not os.path.exists(data_folder + 'circle_re_100.mat'):
            # Download the dataset if it does not exist

            # pri(f'folder/file not found {data_folder}')
            get_wake_data(data_folder, case='circle_re_100')

        # Load dataset
        mat = load_from_mat_file(data_folder + 'circle_re_100.mat')

        U, V = [mat[key] for key in ['ux', 'uy']]  # Nu, Nt, Ny, Nx

        # the record is noised once and cached, so its noise has a fixed seed of its own: drawing it from
        # the shared generator would shift every later draw of the run that builds the cache
        U_noisy, V_noisy = add_noise_to_flow(np.array([U, V]),
                                             noise_level=noise_level,
                                             noise_type=noise_type,
                                             spatial_smooth=smoothing,
                                             rng=noise_generator(0))

        all_data = np.array([U, V])                     # Nu, Nt, Ny, Nx
        all_data_noisy = np.array([U_noisy, V_noisy])   # Nu, Nt, Ny, Nx

        #  Change order of dimensions
        all_data = all_data.transpose(1, 2, 3, 0)               # Nt, Nx, Ny, Nu
        all_data_noisy = all_data_noisy.transpose(1, 2, 3, 0)   # Nt, Nx, Ny, Nu

        save_to_mat_file(data_name, dict(all_data=all_data,
                                         all_data_noisy=all_data_noisy))
    else:
        logger.debug('Loading %s', data_name)
        dataset = load_from_mat_file(data_name)
        all_data, all_data_noisy = [dataset[key] for key in ['all_data', 'all_data_noisy']]

    if visualize:
        visualize_flow_data(all_data, all_data_noisy, simulation_dir=new_results_dir)


    return all_data, all_data_noisy, new_results_dir




def visualize_flow_data(X_true, X_noisy, simulation_dir=''):
    """Build (if missing) and display a GIF comparing true and noisy flow fields.

    Parameters
    ----------
    X_true : np.ndarray
        True flow field, shape ``(..., 2)`` with the last axis indexing the
        (u, v) velocity components.
    X_noisy : np.ndarray
        Noisy flow field, same shape as `X_true`.
    simulation_dir : str, optional
        Directory where the GIF (``00_data.gif``) is saved/read from.
    """

    # Define the GIF name

    gif_name = f'{simulation_dir}00_data.gif'

    # Visualize the flow fields
    if not os.path.exists(gif_name):
        datasets = {
            '$u_x$': X_true[...,0],
            '$u_y$': X_true[...,1],
            '$\\tilde{u}_x$': X_noisy[...,0],
            '$\\tilde{u}_y$': X_noisy[...,1]
        }
        anim = animate_flowfields(datasets, n_frames=200, step=2, figsize=(6, 4))
        anim.save(gif_name)

    # Display in notebook (IPython imported lazily so the package does not
    # require IPython outside notebook contexts, e.g. in the docs CI build)
    from IPython.display import Image, display
    display(Image(filename=gif_name))


def animate_flowfields(datsets,
                       time=None,
                       n_frames=40, cmaps=None, step=1,
                       figsize=None, aspect=None):
    """Create a `matplotlib.animation.FuncAnimation` of side-by-side flow fields.

    Parameters
    ----------
    datsets : dict
        Flow-field datasets, each of shape ``(Ny, Nx, Nt)`` (transposed in place
        if needed so the vertical dimension is the larger one, and required to
        share spatial dimensions). Keys are used as subplot titles; a title
        containing ``'RMS'`` or ``'std'`` is drawn in ``'Reds'`` or ``'Blues'``.
    time : np.ndarray, optional
        Ignored; frames are counted in snapshots.
    n_frames : int, optional
        Number of snapshots to animate. Default 40.
    cmaps : list of str, optional
        Colormap per dataset. Defaults to ``'viridis'`` for all.
    step : int, optional
        Frame stride. Default 1.
    figsize : tuple of float, optional
        Figure size. Defaults to a size scaled by the number of datasets.
    aspect : float, optional
        Physical height/width of one displayed grid cell (rows/columns after the
        automatic transpose), passed to `Axes.set_aspect` -- e.g. ``dx / dy`` on a
        uniform but anisotropic grid. Default None (the panel shape sets it).

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animation object (figure is closed; call `.save` or display it).
    """

    if cmaps is None:
        cmaps = ['viridis'] * len(datsets)

    if figsize is None:
        figsize = (4, 1.5 * len(datsets))
    fig, axs = plt.subplots(1, len(datsets), sharex=True, sharey=True,
                            figsize=figsize, layout='constrained', squeeze=False)
    axs = axs[0]

    # Transpose datasets if needed so vertical dim > horizontal dim
    for key, D in datsets.items():
        if D.shape[0] < D.shape[1]:
            datsets[key] = D.transpose(1, 0, 2)

    for key, D in datsets.items():
        ref = list(datsets.values())[0]
        if D.shape[0] != ref.shape[0] or D.shape[1] != ref.shape[1]:
            raise ValueError("All datasets must have the same spatial dimensions.")

    Nt = list(datsets.values())[0].shape[-1]
    frame_indices = list(range(0, min(n_frames, Nt), step))

    # ------------------------------------------------------------------ #
    #  Build initial pcolormesh artists                                   #
    # ------------------------------------------------------------------ #
    ims = []
    for ax, (ttl, D), cmap in zip(axs, datsets.items(), cmaps):
        if 'RMS' in ttl:
            ims.append(ax.pcolormesh(D[..., 0], rasterized=True,
                                     cmap=plt.get_cmap('Reds'), vmin=0, vmax=1))
        elif 'std' in ttl.lower():
            norm = colors.Normalize(vmin=np.nanmin(D), vmax=np.nanmax(D))
            ims.append(ax.pcolormesh(D[..., 0], rasterized=True,
                                     cmap=plt.get_cmap('Blues'), norm=norm))
        else:
            norm = colors.Normalize(vmin=np.nanmin(D), vmax=np.nanmax(D))
            ims.append(ax.pcolormesh(D[..., 0], rasterized=True,
                                     cmap=plt.get_cmap(cmap), norm=norm))

        ax.set(xticks=[], yticks=[])
        if aspect is not None:
            ax.set_aspect(aspect)
        fig.colorbar(ims[-1], ax=ax, orientation='horizontal', label=ttl)

    # ------------------------------------------------------------------ #
    #  Animation update function                                          #
    # ------------------------------------------------------------------ #
    def animate(ti):
        frame = frame_indices[ti]
        for im, D in zip(ims, datsets.values()):
            im.set_array(D[..., frame])
        logger.debug('Frame %d/%d', ti + 1, len(frame_indices))
        return ims

    plt.close(fig)
    return FuncAnimation(fig, animate, frames=len(frame_indices), cache_frame_data=False)




# --------------------------- notebook figure collector ---------------------------

_savefig_figures = {}


def savefig(fig, name, figs_dir=None, **kwargs):
    """Collect a notebook's figures into one PDF, ``figs/<nn>.pdf`` (from the ``NN_``
    name prefix), rewriting the file each call so re-runs replace pages, not append.

    ``figs_dir`` defaults to ``figs/`` next to the caller (the current directory)."""
    figs = figs_dir or os.path.join(os.getcwd(), 'figs')
    os.makedirs(figs, exist_ok=True)
    nb = name.split('_')[0]
    _savefig_figures.setdefault(nb, {})[name] = fig

    path = os.path.join(figs, f'{nb}.pdf')
    with plt_pdf.PdfPages(path) as pdf:
        for f in _savefig_figures[nb].values():
            pdf.savefig(f, bbox_inches='tight', **kwargs)
    return path
