

import numpy as np

from .autoencoders import POD
from .latent_rom import LatentROM
from .rnns import ESN_model


def synthetic_field(N_t=200, Nx=16, Ny=8, seed=0):
    """Two travelling-wave snapshot fields, ``(2, N_t, Nx, Ny)`` — the shape
    convention `Projector.fit` expects — low-rank enough that a handful of latent
    modes reconstructs them almost exactly. For demos and self-checks (same
    construction as ``tests/test_pod_spod.py``); real cases load actual snapshots."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 8 * np.pi, N_t)
    x = np.linspace(0, 2 * np.pi, Nx)
    y = np.linspace(0, np.pi, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    u = (np.sin(X)[None] * np.cos(t)[:, None, None]
         + 0.5 * np.sin(2 * X + Y)[None] * np.sin(2 * t)[:, None, None])
    v = np.cos(X + Y)[None] * np.sin(t)[:, None, None]
    u = u + 0.02 * rng.standard_normal(u.shape)  # keep the data full-rank
    v = v + 0.02 * rng.standard_normal(v.shape)
    return np.array([u, v])  # (2, N_t, Nx, Ny)


class POD_ESN(LatentROM, ESN_model, POD):
    r"""POD-projected echo state network: a [`POD`][romda.models.data_driven.autoencoders.POD]
    decomposition reduces the (spatial) field to a handful of temporal
    coefficients, and an [`ESN_model`][romda.models.data_driven.rnns.esn.ESN_model] is
    trained to forecast those coefficients in time.

    Following the [`POD`][romda.models.data_driven.autoencoders.POD] convention (see its
    docstring), writing $\matr{Q} = \matr{X} - \bar{\matr{Q}}$ for the
    zero-mean data, the field is approximated as

    $$
    \matr{X} \approx \matr{\Psi}\matr{\Phi} + \bar{\matr{Q}},
    $$

    with $\matr{\Psi}$ (`Psi`) the orthonormal *spatial* modes and
    $\matr{\Phi}$ (`Phi`) the *temporal* coefficients
    ($\matr{\Phi} = \matr{\Psi}^\mathrm{T}\matr{Q}$, already scaled by the
    singular values `Sigma`). `POD_ESN.__init__` runs the POD decomposition first,
    then trains the ESN on $\matr{\Phi}$ (transposed to the
    ``(L, Nt, N_modes)`` layout `ESN_model` expects) to forecast the temporal
    coefficients forward in time; `get_observables` maps the ESN's closed-loop
    `Phi` forecast back to physical sensor readings (`decode_at`) or returns the modal
    coefficients directly if `measure_modes`.

    Construction, sensor placement and observables are those of
    `romda.models.data_driven.latent_rom.LatentROM`, which generalizes
    this class to any projector and forecaster; `POD_ESN` keeps its former
    keywords (``N_modes``, ``plot_case``, ``train_ESN``, ``figs_folder``) and
    method `plot_case`.
    """

    _options = (*LatentROM._options, 'figs_folder')
    latent_symbol = '\\Phi'

    _size_param = 'N_modes'
    figs_folder: str = 'figs/POD-ESN/'

    def __init__(self,
                 data,
                 dt,
                 plot_case=False,
                 **kwargs):
        """Run the POD decomposition, train the ESN on the resulting temporal
        coefficients, and select sensor locations for observation.

        Parameters
        ----------
        data : np.ndarray
            Snapshots for the POD decomposition and ESN training, shape
            ``(Nu, Nt, Nx, Ny)`` (NaN at solid points), or a list of such records
            (see `LatentROM.__init__`).
        dt : float
            Time step of `data`; forwarded to `ESN_model.__init__`.
        plot_case : bool
            Whether to plot the POD modes/spectrum/reconstruction and the ESN
            training process. Default False.
        **kwargs
            Keywords of `LatentROM.__init__` (``train_ESN``,
            ``skip_sensor_placement``, ``domain_of_measurement``,
            ``down_sample_measurement``, ``Nq``, ``sensor_locations``,
            ``measure_modes``, ``qr_selection``, ``figs_folder``), of `POD`
            (``N_modes``, ``domain``, ``grid_shape``, ``method``, ...) and of
            `ESN_model` (``t_train``, ``t_val``, ``N_units``, ...).
        """
        kwargs.setdefault('plot_training', plot_case)
        LatentROM.__init__(self, data=data, dt=dt, **kwargs)

        if plot_case:
            data = data[-1] if isinstance(data, (list, tuple)) else data   # several records: plot the last
            rec = self._to_physical_grid(self.reconstruct(X=data[:, -1]))
            err = np.abs(rec - data[:, -1]) / np.nanmax(np.abs(data[:, -1]))
            datasets = {'Input': data[:, -1],
                        f'POD {self.N_modes} modes': rec,
                        'Error': err}
            POD_ESN.plot_case(case=self, num_modes=self.N_modes, datasets=datasets)

    # ========================================== PLOTS =======================================================


    @staticmethod
    def plot_case(case, datasets: dict | None=None, num_modes=None):
        """Plot the POD modes, temporal coefficients, spectrum and (if `datasets`
        is given) flow/reconstruction/error fields with sensor locations overlaid.

        Parameters
        ----------
        case : POD_ESN
            Instance to plot.
        datasets : dict, optional
            Named fields to pass to `romda.plotting.pod.plot_flows_rms` (e.g.
            ``{'Input': ..., 'Reconstruction': ..., 'Error': ...}``).
        num_modes : int, optional
            Number of POD modes to show. Defaults to `case.N_modes`.

        Returns
        -------
        None
        """
        from romda.plotting.pod import plot_flows_rms, plot_modes, plot_spectrum, plot_time_coefficients

        if num_modes is None:
            num_modes = case.N_modes

        plot_modes(case=case, num_modes=num_modes, cmap='viridis', n_col=2)
        plot_time_coefficients(case=case, num_modes=num_modes)
        plot_spectrum(case=case, max_mode=num_modes)


        if datasets is not None:
            display_sensors = case.sensor_locations is not None
            plot_flows_rms(case=case, datasets=datasets, display_sensors=display_sensors)
