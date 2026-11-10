import inspect
import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sla
from dynamodels import DiscreteIntegrator, Model
from echostatenetwork import EchoStateNetwork
from romda.utils import mean_vector_to_ensemble

logger = logging.getLogger(__name__)


def phi_to_esn_layout(Z):
    """``(N_latent, N_t)`` latent coefficients (a projector's ``encode()`` output,
    or POD's ``Phi``) -> the ``(L, N_t, N_latent)`` layout `ESN_model` expects as
    ``data``; a 3-D input is taken as ``(L, N_latent, N_t)`` segments."""
    Z = np.asarray(Z)
    if Z.ndim == 2:
        Z = Z[np.newaxis, ...]  # (1, N_latent, N_t)
    return Z.transpose(0, 2, 1)  # (L, N_t, N_latent)


class ESN_model(EchoStateNetwork, Model):
    r"""Echo state network as a data-driven forecast model.

    Wraps the [`EchoStateNetwork`](https://andreanovoa.github.io/EchoStateNetwork/) reservoir
    with the [`Model`][dynamodels.model.Model] interface (state history, discrete
    integrator, observation operator), so a trained ESN can be used as the forecast
    model of an `Ensemble` — or as the forecaster inside
    [`ESN_bias`][romda.bias_estimators.esn.ESN_bias]. The model state is
    $[\vect{u}; \vect{r}]$: the physical outputs and the reservoir state.
    Training data is mandatory at construction (the network trains itself unless a
    cached configuration is found).

    Parameters
    ----------
    dt : float
        Output time step (the internal ESN step is ``dt * upsample``).
    **kwargs
        Supported keys include:

        - ``data`` : np.ndarray, training data, shape $(L, N_t, N_\mathrm{dim})$
          ($L$ segments/experiments, $N_t$ time steps spanning
          train + validation + test, $N_\mathrm{dim}$ state dimensions). Required
          unless ``y0`` is given (with pre-trained matrices, e.g. ``Wout``).
        - ``y0`` : np.ndarray, initial state, used only if ``data`` is not given.
        - ``plot_training`` : bool, whether to plot the training data and
          convergence. Default True.
        - ESN hyperparameters (``N_units``, ``N_wash``, ``rho_range``, ...) and
          `Model` options.

    Notes
    -----
    Besides the parameters of a parametric ESN, `init_ensemble` accepts in ``est_alpha`` the
    following network quantities, which each member then carries in the augmented state and the
    filter estimates with the reservoir state:

    - ``'Wout'``: the singular values of the read-out, $\matr{W}_\mathrm{out} =
      \matr{U}\matr{\Sigma}\matr{V}^\mathrm{h}$, one ``svd_i`` per output (this sets
      ``Wout_svd``). With ``Wout_log = True`` the state carries $\log\sigma_i$, so that every member
      keeps positive singular values, and a scalar ``std_alpha`` is the half-width of the log-uniform
      initial spread, $\sigma_i\exp(U[-\mathrm{std}, \mathrm{std}])$.
    - ``'Wdir'``: read-out corrections along $k$ fixed directions, the columns of ``Wout_dirs``,
      shape $(N_\mathrm{units} + 1, k)$: member $j$ reads out with $\matr{W}_\mathrm{out} +
      \matr{W}_\mathrm{dirs}\matr{C}_j$, with the coordinates $\matr{C}_j$, shape
      $(k, N_\mathrm{dim})$, stored as ``wdir_<i>_<q>`` and started from
      $\mathcal{N}(0, (\mathrm{std\_alpha}\,\mathrm{Wout\_dir\_std}_{iq})^2)$.
    - ``'sigma_in'``, ``'rho'``: the input scaling and the spectral radius; each member steps with
      its own value, and the nominal attributes are restored after every forecast.

    `tests/test_esn_readout_options.py` checks each option.
    """

    update_reservoir = True
    update_state = True
    training_data_filename = None # Filename of the data used for training (for config saving/loading only, not used in the actual training)

    Wout_svd = False
    # True: the augmented state carries log(svd_i), so members stay positive; with 'Wout' in est_alpha, a float
    # std_alpha is then the half-width of a log-uniform initial spread (members at s_i * exp(U[-std, std])).
    Wout_log = False
    # Readout corrections along fixed directions (est_alpha=['Wdir']): each member reads out with
    # Wout + Wout_dirs @ C_j, C_j = (k, N_dim) coordinates 'wdir_<i>_<q>' in the augmented state, started from
    # N(0, (std_alpha * Wout_dir_std)^2); Wout_dirs is (N_units + 1, k), Wout_dir_std is (k, N_dim).
    Wout_dirs = None
    Wout_dir_std = None
    validation_data = None

    t_train, t_val, t_test = None, None, 0.

    perform_test = True

    upsample = 1

    N_wash = 5  # Number of washout steps i.e., open-loop initialization
    N_units = 50  # Number of neurons
    N_func_evals = 40
    N_grid = 5
    noise = 1e-2
    N_folds = 8
    N_split = 5

    # Hyperparameter optimization ranges
    rho_range = (.2, .8)
    sigma_in_range = (-2, 2)
    tikh_range = [1E-6, 1E-9, 1E-12]

    # params = ['Wout']
    extra_print_params = ['rho', 'sigma_in', 'N_units', 'N_wash', 'upsample',
                          'update_reservoir', 'update_state']

    #: Estimable input parameters of a *parametric* ESN (empty for a plain one).
    param_names = ()

    def __init__(self,
                 dt,
                 **kwargs):
        # See the class docstring for the meaning of dt and **kwargs.
        data = kwargs.pop('data', None)
        y0 = kwargs.pop('y0', None)
        plot_training = kwargs.pop('plot_training', True)

        # =================== STEP 0: ESTIMABLE INPUT PARAMETERS ======================
        # A parametric ESN is conditioned on a physical parameter vector that is
        # normally *given* at forecast time (`EchoStateNetwork.input_parameters`).
        # Naming the entries in `param_names` mirrors each one as a `Model` parameter:
        # it can be listed in `est_alpha`, joins the augmented state alongside the
        # reservoir (and the `Wout` singular values), and is read back per member
        # before every forecast (see `time_step`). `param_values` are the nominal
        # values (the ensemble mean at initialization). Avoid the names of the ESN's
        # own hyperparameters (`rho`/`sigma_in`/`tikh`).
        self.param_names = tuple(kwargs.pop('param_names', ()))
        param_labels = kwargs.pop('param_labels', None)
        values = np.atleast_1d(np.asarray(
            kwargs.pop('param_values', np.zeros(len(self.param_names))), dtype=float))
        # kept as attributes so the config store can save/rebuild a parametric ESN
        self.param_values = tuple(float(v) for v in values)
        self.param_labels = tuple(param_labels) if param_labels else None
        for name, val in zip(self.param_names, values):
            setattr(self, name, float(val))
        # `Model.__init__` builds alpha0 from `params`, so register them before it runs
        self.params = list(self.params) + list(self.param_names)

        # =================== STEP 1: EchoStateNetwork INITIALIZATION ======================

        # options that are class attributes of ESN_model; a keyword that names a method is left to Model.__init__, which
        # ignores it with a warning, instead of replacing the method
        own = vars(ESN_model)
        [setattr(self, key, kwargs.pop(key)) for key in list(kwargs.keys())
         if key in own and not (callable(own[key]) or isinstance(own[key], (staticmethod, classmethod)))]

        if data is not None:
            assert isinstance(data, np.ndarray), f"Expected data to be a numpy array, got {type(data)}"
            data = self._process_initialization_data(data, dt, kwargs) # type: np.ndarray # with shape (L, Nt, Ndim)
            y0 = data[0, 0]
        elif y0 is None:
            raise ValueError('Either training data or initial state y0 must be provided to initialize the ESN_model.')

        initial_dict = {key: kwargs.pop(key) for key in list(kwargs.keys()) if key in vars(EchoStateNetwork)}
        EchoStateNetwork.__init__(self,
                                y=y0,
                                dt=dt,
                                **initial_dict)

        # what is left goes to `train` (its own options) and `Model.__init__` (the rest); name the
        # keys that neither knows, so a misspelt or unsupported option is not lost silently
        train_args = inspect.signature(EchoStateNetwork.train).parameters
        unknown = sorted(key for key in kwargs if key not in train_args and not hasattr(self, key))
        if unknown:
            warnings.warn(f'{type(self).__name__}: unknown keyword(s) {unknown} ignored', stacklevel=2)


        # =================== STEP 2: EchoStateNetwork TRAINING ======================
        # Train the network if not already trained
        if not self.trained:
            logger.debug('Training ESN model...')
            if plot_training:
                self.plot_training_data(case=self, train_data=data, dt=dt)

            self.train(train_data=data, plot_training=plot_training,
                       **{key: val for key, val in kwargs.items() if key in train_args})

            # save validation data for initialization
            Y_wtv = self._split_and_format_data(data)[1]
            self.validation_data = Y_wtv[-(self.N_wash + self.N_val):]


        # ================== STEP 3: DEFINE INITIAL STATE & PARAMS ======================
        # `Model.Nq = 1` is a class attribute, so `hasattr` is always True -- check the
        # instance dict instead, otherwise every ESN_model silently ends up with Nq=1
        # (POD_ESN sets its own Nq as an instance attribute before calling this).
        if 'Nq' not in vars(self):
            self.Nq = len(self.observed_idx)  # Number of observed dimensions (for the physical state)

        psi0 = self.initialize_from_val_data()  # shape (Ndim + N_units + Na, m)

        # Initialise SVD Wout terms if required
        if self.Wout_svd:
            [self.Wout_U, self.Wout_Sigma0, self.Wout_Vh] = sla.svd(self.Wout, full_matrices=False)

        # =================== STEP 4: Model INITIALIZATION ======================
        Model.__init__(self,
                       dt=dt,
                       psi0=psi0,
                       integrator_class=DiscreteIntegrator,
                       **{key: val for key, val in kwargs.items() if key not in train_args})

        if self.param_names:
            # `Model.alpha_labels` defaults to a label->name mapping, so plotting an
            # estimated parameter by name raises KeyError until real labels are set.
            self.alpha_labels = dict(zip(self.param_names,
                                         param_labels or [f'${n}$' for n in self.param_names]))

    @property
    def t_transient(self):
        """float: Total time spanning training + validation + test, ``t_train + t_val + t_test``."""
        return self.t_train + self.t_val + self.t_test

    def _process_initialization_data(self, data, dt, kwargs) -> np.ndarray:
        """Reshape `data` to ``(L, Nt, N_dim)`` and infer `t_train`/`t_val`/`t_test`
        (from ``N_train``/``N_val``/``N_test`` in `kwargs` if given, else defaulting
        to an 80/20/remainder split of the data length) if not already set.
        """
        # Increase ndim if there is only one set of parameters
        if data.ndim == 1:
            data = data[np.newaxis, :, np.newaxis]
        elif data.ndim == 2:
            data = data[np.newaxis, :]

        # Check that the times are provided and not in time steps
        Nt = data.shape[1]
        for key in ["train", "val", "test"]:
            if f"N_{key}" in kwargs.keys():
                setattr(self, f"t_{key}", kwargs.pop(f"N_{key}") * dt)

        # Set other ESN_model attributes provided in kwargs

        #  Set time attributes  #
        t_total = Nt * dt
        self.t_train = self.t_train or t_total * 0.8
        self.t_val = self.t_val or self.t_train * 0.2

        if self.perform_test:
            self.t_test = self.t_test or t_total - self.t_train - self.t_val

            # The dataset is sized as ceil((t_train + t_val + t_test) / dt) steps
            # (see ESN_bias.minimum_training_steps), so Nt * dt can exceed the
            # requested times by up to one dt; absorb the rounding into t_test.
            sum_t = self.t_train + self.t_val + self.t_test
            if abs(sum_t - t_total) > dt / 2.:
                self.t_test += t_total - sum_t

            assert abs((ts := sum([self.t_train, self.t_val, self.t_test])) - t_total) <= dt, \
                f"t_train + t_val + t_test = {ts} does not match the data length t_total = {t_total}"

        return data


    # ______________________ New class attributes ______________________ #
    def modify_settings(self):
        """Switch to the SVD parametrization of `Wout` (see `Wout_svd`) if ``'Wout'``
        is requested as an ensemble parameter to estimate (`est_alpha`). Called by
        `Model.init_ensemble`; updates `est_alpha`/`alpha_labels`/`alpha_lims`/`M`
        in place."""
        if self.ensemble_cfg is not None:
            # If Wout is being estimated, we need to update the est_alpha list to include the SVD components
            # and remove Wout. We do not directly estimate Wout, but rather its singular values.
            if 'Wout' in self.est_alpha:
                self.est_alpha = [a for a in self.est_alpha if a != 'Wout'] + \
                                 [f'svd_{qi}' for qi in range(self.N_dim)]

            # The switch must key on the svd names, not on 'Wout': callers that build
            # the initial ensemble themselves pass est_alpha already
            # expanded, and without Wout_svd the read-out keeps using the fixed Wout —
            # the estimated singular values never act on the forecast.
            dir_keys = [a for a in self.est_alpha if a.startswith('wdir_')]
            if dir_keys:
                new = [key for key in dir_keys if key not in self.params]       # registered like the svd_i
                for key in new:
                    setattr(self, key, 0.0)
                    if hasattr(self, '_alpha0'):
                        self._alpha0[key] = 0.0
                self.params = list(self.params) + new
                self.alpha_labels = {key: f'$c_{{{key[5:]}}}$' for key in dir_keys}
                self.alpha_lims = {key: (None, None) for key in dir_keys}
            svd_keys = [a for a in self.est_alpha if a.startswith('svd_')]
            if svd_keys:
                if not self.Wout_svd:
                    self.Wout_svd = True
                    [self.Wout_U, self.Wout_Sigma0, self.Wout_Vh] = sla.svd(self.Wout, full_matrices=False)
                # the setters merge, so labels/lims of other estimated parameters survive
                self.alpha_labels = {key: f'$\\sigma_{{{key.split("_")[1]}}}$' for key in svd_keys}
                self.alpha_lims = {key: (None, None) for key in svd_keys}
        self.M = None

    @property
    def dt_step(self):
        """float: Integrator time step, `dt_ESN` (the ESN advances in closed loop at
        its own upsampled time step, not `dt`)."""
        return self.dt_ESN

    @property
    def t_CR(self):
        """float: Characteristic response time used by the base `Model`, aliased to `t_val`."""
        return self.t_val

    @property
    def Wout_Sigma(self):
        r"""np.ndarray: Per-member diagonal singular-value matrices used to
        reconstruct `Wout` as $\matr{W}_\mathrm{out} \approx \matr{U}\,\matr{\Sigma}\,\matr{V}^\mathrm{h}$
        (see `reservoir_to_physical`), shape ``(m, N_dim, N_dim)``. Built on every
        access from the current ``svd_i`` parameter estimates (`get_alpha_matrix`),
        falling back to the corresponding `Wout_Sigma0` singular value for any
        ``svd_i`` not in `est_alpha`.
        """
        alpha_matrix = self.get_alpha_matrix

        eigs = np.zeros((self.m, self.N_dim, self.N_dim))

        for qi in range(self.N_dim):
            key = f'svd_{qi}'
            if key in self.est_alpha:
                ai = self.est_alpha.index(key)
                vals = np.exp(alpha_matrix[ai]) if self.Wout_log else alpha_matrix[ai]
            else:
                vals = self.Wout_Sigma0[qi] * np.ones(self.m)

            eigs[:, qi, qi] = vals

        return eigs

    @property
    def get_alpha_matrix(self):
        """np.ndarray: Current ensemble parameter estimates (`est_alpha`), shape
        ``(len(est_alpha), m)``: the last `Na` rows of the augmented state."""
        state = self.current_state
        return np.array(state[len(state) - self.Na:], dtype=float)


    @property
    def Wout_Sigma0(self):
        """np.ndarray: Reference (initial) singular values of `Wout`, shape
        ``(N_dim,)``, as computed by ``scipy.linalg.svd`` when `Wout_svd` is enabled.
        """
        return self._Wout_Sigma0

    @Wout_Sigma0.setter
    def Wout_Sigma0(self, eigs):
        """Set `Wout_Sigma0` and register one ``svd_i`` scalar attribute/parameter
        per singular value (so they can be estimated like any other `params` entry).
        """
        self._Wout_Sigma0 = eigs
        params = self.params.copy()

        for eig_i, val in enumerate(eigs):
            setattr(self, f'svd_{eig_i}', val)
            params.append(f'svd_{eig_i}')
            # Keep alpha0 in sync if the model is already initialized
            if hasattr(self, '_alpha0'):
                self._alpha0[f'svd_{eig_i}'] = val

        self.params = params

    # ______________________ Changed EchoStateNetwork class attributes ______________________ #

    def init_ensemble(self, m=10, est_alpha=[], std_alpha=0.001,
                      distribution_alpha='uniform', regimes=None, seed=None,
                      measured=None, ensemble_psi0=None, **kwargs):
        """ESN override of `Model.init_ensemble`: members start from reservoir states
        visited during training (`initialize_from_val_data`) — the generic transient +
        multiplicative `std_phi` path pushes `r` outside the tanh range and the closed
        loop blows up.

        `regimes` (`(N_param, L)`, parametric ESN): washes each member out on one
        training segment *and* starts it from that segment's parameters, keeping state
        and parameter consistent — paired independently, members leave the learned
        attractor and diverge. ``'Wout'`` in `est_alpha` estimates the read-out
        singular values (one ``svd_i`` per output dimension); `measured` restricts
        the observation operator to those state components.

        Parameter perturbations draw from *this* instance's `rng`; an estimator
        builds the ensemble on its own copy (`EnsembleEstimator` copies
        `parent_model` first), so repeated builds from one parent network draw
        identical perturbations — the parent's `rng` is deliberately untouched.
        A given `seed` becomes the seed of this copy: it selects the washout windows,
        the parameter perturbations and the training segments of `regimes`. Without
        it, the copy keeps its own seed (and `regimes` seed 0)."""
        if seed is not None:
            self.seed = seed
        if ensemble_psi0 is None:
            nominal = {}
            if isinstance(std_alpha, dict) and 'Wout' in std_alpha:
                std_alpha = dict(std_alpha)
                std_alpha.update({f'svd_{qi}': std_alpha.pop('Wout')
                                  for qi in range(self.N_dim)})
            if isinstance(std_alpha, dict) and not est_alpha:
                # base-class convention: an empty est_alpha with a dict std_alpha
                # estimates every parameter in the dict; an explicit est_alpha is
                # honored (the dict may carry spreads for more parameters)
                est_alpha = sorted(std_alpha)
            if 'Wout' in est_alpha:
                svd_names = [f'svd_{qi}' for qi in range(self.N_dim)]
                est_alpha = [a for a in est_alpha if a != 'Wout'] + svd_names
                nominal = dict(zip(svd_names, sla.svd(self.Wout, full_matrices=False)[1]))

            dir_names = []
            if 'Wdir' in est_alpha:
                k = self.Wout_dirs.shape[1]
                dir_names = [f'wdir_{i}_{q}' for i in range(k) for q in range(self.N_dim)]
                est_alpha = [a for a in est_alpha if a != 'Wdir'] + dir_names

            if regimes is None:
                phi, alpha = self.initialize_from_val_data(N_ens=m), {}
            else:
                phi, values = self._regime_matched_init(m, np.atleast_2d(regimes), seed=0 if seed is None else seed)
                alpha = dict(zip(self.param_names, values))

            if dir_names:
                assert np.isscalar(std_alpha), 'Wdir: std_alpha scales Wout_dir_std, a float'
                alpha.update({f'wdir_{i}_{q}': self.rng.normal(0, std_alpha * self.Wout_dir_std[i, q], m)
                              for i in range(self.Wout_dirs.shape[1]) for q in range(self.N_dim)})
            if self.Wout_log and nominal:
                assert np.isscalar(std_alpha), 'Wout_log: std_alpha is the half-width of log(svd_i), a float'
                alpha.update({a: np.log(v) + self.rng.uniform(-std_alpha, std_alpha, m) for a, v in nominal.items()})
            other = [a for a in est_alpha if a not in alpha]
            if other:
                means = [nominal.get(a, getattr(self, a, None)) for a in other]
                assert not any(v is None for v in means), \
                    f'init_ensemble: no nominal value to seed the ensemble for {other}'
                sub_std = ({a: std_alpha[a] for a in other}
                           if isinstance(std_alpha, dict) else std_alpha)
                alpha.update(zip(other, mean_vector_to_ensemble(
                    self.rng, np.array(means, dtype=float), sub_std, m,
                    method=distribution_alpha)))

            ensemble_psi0 = phi if not est_alpha else np.concatenate(
                [phi, np.stack([alpha[name] for name in est_alpha])], axis=0)

        out = super().init_ensemble(m=m, est_alpha=est_alpha, std_alpha=std_alpha,
                                    distribution_alpha=distribution_alpha,
                                    ensemble_psi0=ensemble_psi0, **kwargs)

        if measured is not None:
            if np.isscalar(measured):   # int count or digit string: resolve to indices
                from romda.observations import measured_idx
                idx = measured_idx(measured, self.N_dim)
            else:
                idx = [int(k) for k in measured]
            if idx != list(self.observed_idx):
                # Rows of `M` for the measured components only. Written to the private
                # attribute: the public setter requires exactly Nq rows, and a
                # restricted operator deliberately has fewer.
                obs = list(self.observed_idx)
                self._M = self.M[[obs.index(k) for k in idx]]
        return out

    def _regime_matched_init(self, m, regimes, seed=0):
        """Wash out member `i` on training segment `i % L`, conditioned on that
        segment's own parameters. Returns the `(Nphi, m)` state and `(N_param, m)`
        parameters."""
        data = self.validation_data                         # (L, Nt, N_dim)
        rng = np.random.default_rng(seed)
        segments = np.arange(m) % data.shape[0]

        u = np.empty((self.N_dim, m))
        r = np.empty((self.N_r, m))
        original = self.input_parameters
        try:
            for i, j in enumerate(segments):
                self.input_parameters = regimes[:, j:j + 1]
                k0 = rng.integers(0, data.shape[1] - self.N_wash)
                u_i, r_i = np.zeros((self.N_dim, 1)), np.zeros((self.N_r, 1))
                for u_in in data[j, k0:k0 + self.N_wash]:
                    u_i, r_i = self.step(self.outputs_to_inputs(u_in[:, np.newaxis]), r_i)
                u[:, i], r[:, i] = u_i[:, 0], r_i[:, 0]
        finally:
            self.input_parameters = original

        return self.build_psi(u=u, r=r), regimes[:, segments]

    def initialize_from_val_data(self, N_ens=1, seed=0):
        """Initialize the ESN state (physical output and reservoir) from an
        open-loop washout over a random time window of `validation_data`, so a fresh
        ensemble starts from an on-attractor reservoir state rather than zeros.

        Parameters
        ----------
        N_ens : int
            Number of ensemble members (random washout windows) to draw. Default 1.
        seed : int
            Random seed for selecting the washout windows. Overridden by `self.seed`
            if set.

        Returns
        -------
        np.ndarray
            Initial full state (built via `build_psi`), shape
            ``(N_dim + N_units [+ Na], N_ens)``.

        Raises
        ------
        AssertionError
            If `validation_data` has not been set (i.e. before training).
        """
        assert self.validation_data is not None

        data = self.validation_data.copy()

        if hasattr(self, 'seed'):
            seed = self.seed
        rng0 = np.random.default_rng(seed)


        # initialise state with a random sample from test data
        u_init, r_init = np.empty((self.N_dim, N_ens)), np.empty((self.N_r, N_ens))

        # Random time windows and dimension
        if data.shape[0] == 1:
            dim_ids = [0] * N_ens
        else:
            # Choose a random dimension from the data
            replace = N_ens >= data.shape[0]
            dim_ids = rng0.choice(data.shape[0], size=N_ens, replace=replace)

        # Choose random time indices from the data (with replacement when there are
        # fewer washout windows than members, e.g. a short validation record)
        n_windows = data.shape[1] - self.N_wash
        t_ids = rng0.choice(n_windows, size=N_ens, replace=N_ens >= n_windows)

        # validation_data is indexed by segment (dim_i matches a column of
        # input_parameters); condition the washout of each segment on its own
        # parameter vector, then restore the full (N_param, L) array afterwards.
        original_input_parameters = self.input_parameters

        for ii, ti, dim_i in zip(range(N_ens), t_ids, dim_ids):
            u_wash = data[dim_i, ti:ti+self.N_wash]
            r_open = np.zeros((self.N_r, 1))
            u_open = np.zeros((self.N_dim, 1))
            if original_input_parameters is not None:
                self.input_parameters = original_input_parameters[:, dim_i]
            # Open-loop reservoir
            for u_in in u_wash:
                u_open, r_open = self._single_step(u_in, r_open)

            #store final state into the initialization arrays
            u_init[:, ii] = u_open.squeeze()
            r_init[:, ii] = r_open.squeeze()

        self.input_parameters = original_input_parameters

        # Set physical and reservoir states as ensembles
        return self.build_psi(u=u_init, r=r_init)


    def closed_loop(self, data, n_steps, input_parameters=None):
        """Open-loop washout on `data[:N_wash]` (sampled at `dt_ESN`), then a closed-loop
        forecast; returns ``(prediction, target)``, both `(n_steps, N_dim)`.

        The ESN maps `u_t` to `u_{t+1}`, so the *last* washout step already predicts
        `data[N_wash]` — an off-by-one here costs a full `dt_ESN` of drift."""
        original = self.input_parameters
        if input_parameters is not None:
            self.input_parameters = np.asarray(input_parameters, dtype=float).reshape(-1, 1)

        try:
            r = np.zeros((self.N_r, 1))
            u = np.zeros((self.N_dim, 1))
            for u_in in data[:self.N_wash]:
                u, r = self.step(self.outputs_to_inputs(np.asarray(u_in)[:, np.newaxis]), r)

            pred = np.empty((n_steps, self.N_dim))
            pred[0] = u[:, 0]
            for i in range(1, n_steps):
                u, r = self._single_step(u, r)
                pred[i] = u[:, 0]
        finally:
            # the training-time (N_param, L) array must survive a single-regime forecast
            self.input_parameters = original

        return pred, np.asarray(data[self.N_wash:self.N_wash + n_steps])


    def reset_forecaster(self, data=None, u0=None, r0=None, **kwargs):
        """Restart the model from `u0`, keeping the trained network.

        Nothing is retrained: the history is replaced by a single state at the
        current time. This is the call `LatentROM.reset_case`
        makes on every forecaster.

        Parameters
        ----------
        data : np.ndarray, optional
            Ignored; accepted for the `LatentROM.reset_case` call signature.
        u0 : np.ndarray, optional
            Initial physical state, ``N_dim`` values or shape ``(N_dim, m)``.
            Defaults to the current physical state.
        r0 : np.ndarray, optional
            Initial reservoir state, shape ``(N_units, m)``. Defaults to zeros,
            the state training starts from, so the first ~`N_wash` closed-loop
            steps are a transient; `initialize_from_val_data` gives a start on
            the attractor instead.
        **kwargs
            Ignored.
        """
        del data, kwargs
        if u0 is None:
            u = self.unbuild_psi(self.current_state)[0]
        else:
            u = np.asarray(u0, dtype=float).reshape(self.N_dim, -1)
        r = np.zeros((self.N_units, u.shape[1])) if r0 is None else np.asarray(r0, dtype=float)
        psi = self.build_psi(u=u, r=r)
        self.update_history(psi[np.newaxis], t=np.array([self.current_time]), reset=True)

    # ______________________ Changed Model class attributes ______________________ #

    @property
    def forecaster_state_labels(self):
        r"""list of str: Labels of the reservoir units, $r_1, \dots, r_{N_\mathrm{units}}$,
        or none when `update_reservoir` is False (they are then not part of the state)."""
        if not self.update_reservoir:
            return []
        return [f'$r_{{{j+1}}}$' for j in np.arange(self.N_units)]

    @property
    def state_labels(self):
        r"""list of str: LaTeX labels for the state vector, $u_1, \dots, u_{N_\mathrm{dim}}$
        (physical outputs) followed by $r_1, \dots, r_{N_\mathrm{units}}$ (reservoir units)."""
        return [f'$u_{{{j+1}}}$' for j in np.arange(self.N_dim)] + [f'$r_{{{j+1}}}$' for j in np.arange(self.N_r)]

    @property
    def obs_labels(self):
        """list of str: LaTeX labels for the observed physical outputs (`observed_idx`)."""
        return [f'$u_{{{j+1}}}$' for j in self.observed_idx]

    def get_observables(self, Nt=1, **kwargs):
        """Observables are the *observed* physical outputs, which need not be the
        leading rows of psi (the base Model assumes `psi[:Nq]`).

        Parameters
        ----------
        Nt : int
            Number of trailing history steps to return. Default 1.
        **kwargs
            Unused; accepted for interface compatibility.

        Returns
        -------
        np.ndarray
            Observed outputs, shape ``(Nq, m)`` if ``Nt == 1`` else ``(Nt, Nq, m)``.
        """
        if Nt == 1:
            return self.hist[-1, self.observed_idx, :]
        return self.hist[-Nt:, self.observed_idx, :]



    @property
    def reservoir_state(self):
        """np.ndarray: Reservoir-state block of `current_state`, shape ``(N_units, m)``."""
        return self.current_state[self.N_dim:self.N_dim+self.N_r, :]


    def reservoir_to_physical(self, r):
        r"""Convert reservoir states to physical outputs via `Wout` (overrides
        `EchoStateNetwork.reservoir_to_physical` to also support the SVD
        parametrization of `Wout`, see `Wout_svd`).

        When `Wout_svd` is False: $\vect{u} = \matr{W}_\mathrm{out}^\mathrm{T}[\vect{r}; b_\mathrm{out}]$,
        as in the base class. When True, $\matr{W}_\mathrm{out}$ is reconstructed
        (per ensemble member, if `r` has `m` members) from
        $\matr{U}\,\matr{\Sigma}\,\matr{V}^\mathrm{h}$ (`Wout_U`, `Wout_Sigma`,
        `Wout_Vh`) before the same read-out is applied; if `r` does not have exactly
        `m` members (e.g. a single averaged state), `Wout_Sigma` is averaged over the
        ensemble first.

        Parameters
        ----------
        r : np.ndarray
            Reservoir state, shape ``(N_units, N_ens)``.

        Returns
        -------
        np.ndarray
            Physical output, shape ``(N_dim, N_ens)``.
        """
        if not self.Wout_svd:
            return super().reservoir_to_physical(r)   # also the parallel layout
        else:
            bias_out = self.bias_out * np.ones((1, r.shape[-1]))
            r_aug = np.concatenate((r, bias_out), axis=0)

            if r.shape[-1] == self.m:
                Wout = np.einsum('ij,kjl,lm->imk', self.Wout_U, self.Wout_Sigma, self.Wout_Vh)
                return np.einsum('ij,ikj->kj', r_aug, Wout)
            else:
                # average the alpha values
                logger.debug('Averaging Wout_Sigma for reservoir_to_physical')
                Wout_Sigma_avg = np.mean(self.Wout_Sigma, axis=0)
                Wout = np.dot(self.Wout_U, np.dot(Wout_Sigma_avg, self.Wout_Vh))
                return np.dot(r_aug.T, Wout).T


    def _readout(self, r, u_norm):
        # `EchoStateNetwork.step` reads out through `_readout` (echostatenetwork >= 0.1.2),
        # not through `reservoir_to_physical`: without this hook the per-member SVD
        # read-out (est_alpha=['Wout']) never reaches the closed loop.
        if self.Wout_svd and 'Wout_U' in vars(self):   # not while training, before Wout has its SVD
            return self.reservoir_to_physical(r)
        u = super()._readout(r, u_norm)
        if self.Wout_dirs is not None and self.est_alpha and self.est_alpha[-1].startswith('wdir_') \
                and r.shape[-1] == self.m:
            k = self.Wout_dirs.shape[1]
            A = self.get_alpha_matrix
            C = np.stack([[A[self.est_alpha.index(f'wdir_{i}_{q}')] for q in range(self.N_dim)]
                          for i in range(k)])                                   # (k, N_dim, m)
            proj = self.Wout_dirs.T @ np.vstack([r, self.bias_out * np.ones((1, r.shape[-1]))])  # (k, m)
            u = u + np.einsum('kqm,km->qm', C, proj)
        return u

    def time_step(self, Nt=10, averaged=False):
        """Advance the ESN in closed loop.

        Parameters
        ----------
        Nt : int
            Number of forecast steps (in physical time steps, not ``dt_ESN``).
        averaged : bool
            If True, the ensemble is forecast as its mean plus frozen deviations;
            otherwise each member is forecast individually.

        Returns
        -------
        tuple
            ``(psi, t)`` — forecasted state of shape ``(Nt+1, N, m)`` and the
            corresponding times.
        """

        assert self.trained, 'ESN model not trained'

        if self.param_names:
            # Condition each member's forecast on its own current parameter value.
            # `get_alpha` returns one dict per member: the estimated value when the
            # parameter is in `est_alpha`, otherwise the nominal `alpha0` one (change
            # it in place with ``esn.alpha0[name] = ...`` to forecast at a fixed
            # parameter). Either way the `(N_param, m)` array replaces the
            # `(N_param, L)` one left over from training.
            alpha = self.get_alpha()
            self.input_parameters = np.array([[a[name] for a in alpha]
                                              for name in self.param_names])

        # 1. get initial condition


        t = np.round(self.current_time + np.arange(0, Nt + 1) * self.dt_ESN, self.precision_t)
        psi0 = self.current_state
        u, r_out = np.empty((Nt + 1, self.N_dim, self.m)), np.empty((Nt + 1, self.N_r, self.m))
        u[0], r_out[0] = self.unbuild_psi(psi0)

        if averaged:
            # Mean state
            u_m, r_m = (np.mean(yy[0], axis=-1, keepdims=True) for yy in [u, r_out])
            u_dev, r_dev = u[0] - u_m[0], r_out[0] - r_m[0]


            for i in range(Nt):
                u_m, r_m = self._single_step(u_m, r_m)
                u[i+1] = u_m + u_dev
                r_out[i+1] = r_m + r_dev

        else:
            # Estimated reservoir hyperparameters (e.g. est_alpha=['sigma_in']): each member steps with its own
            # value; `step` broadcasts the (m,) array over the ensemble columns.
            hyper = [a for a in ('sigma_in', 'rho') if a in self.est_alpha]
            nominal = {a: getattr(self, a) for a in hyper}
            for a in hyper:
                setattr(self, a, np.array([al[a] for al in self.get_alpha()]))
            try:
                for i in range(Nt):
                    u[i+1], r_out[i+1] = self._single_step(u[i], r_out[i])
            finally:
                for a, v in nominal.items():
                    setattr(self, a, v)


        psi = self.build_psi(u=u, r=r_out)

        return psi, t


    def _single_step(self, u, r):
        u_input = self.outputs_to_inputs(full_state=u)
        return self.step(u_input, r)


    def build_psi(self, u=None, r=None):
        """Assemble the full model state from physical output `u` and reservoir
        state `r`: ``concatenate([u, r, alpha])`` along the state axis, keeping only
        `u` or only `r` if `update_state`/`update_reservoir` is False, and appending
        the ensemble parameter block (`get_alpha_matrix`) if `Na > 0`.

        Parameters
        ----------
        u : np.ndarray, optional
            Physical output, shape ``(N_dim, m)`` or ``(Nt, N_dim, m)``. Defaults to
            `reservoir_to_physical(r)`.
        r : np.ndarray, optional
            Reservoir state, shape ``(N_units, m)`` or ``(Nt, N_units, m)``. Defaults
            to `reservoir_state`.

        Returns
        -------
        np.ndarray
            Full state ``psi``, shape ``(N [+ Na], m)`` or ``(Nt, N [+ Na], m)``.

        Raises
        ------
        ValueError
            If `u` and `r` have incompatible number of dimensions, or (for 3D
            inputs) a mismatched number of time steps.
        """
        if r is None:
            r = self.reservoir_state
        if u is None:
            u = self.reservoir_to_physical(r)


        if u.ndim == 2 and r.ndim == 2:
            ax_dim = 0
        elif u.ndim == 3 and r.ndim == 3:
            ax_dim = 1
            if u.shape[0] != r.shape[0]:
                raise ValueError(f'Incompatible time steps for u ({u.shape[0]}) and r ({r.shape[0]})')
        else:
            raise ValueError(f'Incompatible dimensions for u ({u.ndim}) and r ({r.ndim})')

        if self.update_state and self.update_reservoir:
            phi = np.concatenate((u, r), axis=ax_dim)
        elif self.update_state:
            phi = u
        else:
            phi = r


        if self.Na > 0:
            alph = self.get_alpha_matrix
            if u.ndim == 3:
                alph = np.tile(alph, reps=(u.shape[0], 1, 1)) # repeat for all time steps (alpha is constant in time)
            return np.concatenate((phi, alph), axis=ax_dim)
        else:
            return phi


    def unbuild_psi(self, psi=None):
        """Inverse of `build_psi`: split a full state vector into its physical
        (`u`) and reservoir (`r`) blocks, assuming they occupy the leading
        ``N_dim + N_units`` rows of `psi` (as `build_psi` lays them out when both
        `update_state` and `update_reservoir` are True).

        Parameters
        ----------
        psi : np.ndarray, optional
            Full state, shape ``(N, m)`` or ``(Nt, N, m)``. Defaults to
            `current_state`.

        Returns
        -------
        u : np.ndarray
            Physical state, shape ``(N_dim, m)`` (or ``(Nt, N_dim, m)``).
        r : np.ndarray
            Reservoir state, shape ``(N_units, m)`` (or ``(Nt, N_units, m)``).

        Raises
        ------
        AssertionError
            If `psi` has fewer than `N_units` + 1 rows.
        """
        if psi is None:
            psi = self.current_state
        if psi.ndim == 2:
            psi = np.expand_dims(psi, axis=0)
            squeeze = True
        else:
            squeeze = False

        assert psi.shape[1] > self.N_r, f"Expected psi shape (N x m) with N > {self.N_r}, got {psi.shape}"
        u = psi[:, :self.N_dim]
        r = psi[:, self.N_dim:self.N_dim+self.N_r]

        if squeeze:
            u = u.squeeze(axis=0)
            r = r.squeeze(axis=0)

        return u, r



    # ______________________________ Plotting functions ______________________________ #
    @staticmethod
    def plot_training_data(case, train_data, dt=None):
        """Plot each dimension of `train_data`, shading the training/validation/test
        windows (`case.t_train`/`t_val`/`t_test`).

        Parameters
        ----------
        case : ESN_model
            Instance providing `t_train`, `t_val`, `t_test` (and `dt` if not given).
        train_data : np.ndarray
            Data to plot, shape ``(L, Nt, N_dim)`` (or 1D/2D, reshaped accordingly).
        dt : float, optional
            Time step for the x-axis. Defaults to ``case.dt``.

        Returns
        -------
        None
            Displays the figure with `matplotlib.pyplot.show`.
        """
        if train_data.ndim == 1:
            train_data = train_data[np.newaxis, :, np.newaxis]
        elif train_data.ndim == 2:
            train_data = train_data[np.newaxis, :]

        L, Nt, Ndim = train_data.shape
        if dt is None:
            dt = case.dt
        t_data = np.arange(0, Nt) * dt
        nrows = min(Ndim*L, 10)


        _, axs = plt.subplots(nrows=nrows, ncols=1,
                                figsize=(8, nrows), sharex=True,
                                layout='constrained')
        if nrows * L > 1 and isinstance(axs, np.ndarray):
            axs = axs.T.flatten()
        else:
            axs = [axs]


        for l, data_l in enumerate(train_data):
            axs_dim = axs[l*Ndim:(l+1)*Ndim]

            for kk, ax in enumerate(axs_dim):

                ax.plot(t_data, data_l[:, kk], lw=1., color='k')
                ax.axvspan(0, case.t_train, facecolor='orange',
                           alpha=0.3, zorder=-100, label='Train')
                ax.axvspan(case.t_train, case.t_train + case.t_val,
                           facecolor='red', alpha=0.3, zorder=-100, label='Validation')
                ax.axvspan(case.t_train + case.t_val,
                           case.t_train + case.t_val + case.t_test, facecolor='navy',
                           alpha=0.2, zorder=-100, label='Test')

                ax.legend(ncols=1, loc='upper left', bbox_to_anchor=(1., 1.), frameon=False, title=f'L={l}, dim={kk}', fontsize='x-small', title_fontsize='small')
        axs[-1].set(xlabel='time')
        plt.show()

    def visualize_config(self):
        """Plot the trained read-out matrix (`plot_Wout`).

        Returns
        -------
        None
        """
        self.plot_Wout()

    def visualize_spatiotemporal_hist(self, y_hist=None, t=None, **kwargs):
        """Space-time diagrams of the state history: `Model.visualize_spatiotemporal_hist`
        on the physical outputs and, if present, on the reservoir state. `y_hist`
        defaults to the last `t_CR` of `hist` and `t` to the matching `hist_t`;
        `kwargs` (``nrows``, ``averaged``, ``reference_y``, ``reference_t``) go to the
        base method."""
        if y_hist is None:
            y_hist = self.hist[-int(self.t_CR // self.dt):, :self.Nphi]
        if t is None:
            t = self.hist_t[-len(y_hist):]
        super().visualize_spatiotemporal_hist(y_hist[:, :self.N_dim], t, **kwargs)
        if y_hist.shape[1] > self.N_dim:
            super().visualize_spatiotemporal_hist(y_hist[:, self.N_dim:self.N_dim + self.N_units], t, **kwargs)

    def plot_Wout(self):
        """Visualize the trained read-out matrix `Wout` (overrides
        `EchoStateNetwork.plot_Wout`): a single heat-map if `Wout_svd` is False, or
        the (ensemble-averaged) SVD factors `Wout_U`, `Wout_Sigma`, `Wout_Vh` and
        their reconstruction if True.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if not self.Wout_svd:
            return super().plot_Wout()
        else:
            fig, axs = plt.subplots(1, 4, figsize=(15, 15), width_ratios=[1, 1, 1, 1])
            eigs = self.Wout_Sigma
            if eigs.ndim >2:
                eigs = np.mean(eigs, axis=0)

            Wout = np.dot(self.Wout_U, np.dot(eigs, self.Wout_Vh))

            for W, ax, title in zip([Wout, self.Wout_U, eigs, self.Wout_Vh], axs,
                                    ['$\\bar{\\mathbf{W_{out}}} = $', '$\\mathbf{U}$', '$\\bar{\\Sigma}$', '$\\mathbf{V}^\\mathrm{T}$']):
                cmap = 'PuOr'
                im = ax.imshow(W, cmap=cmap, vmin=-np.max(W), vmax=np.max(W))
                ax.set(title=title)
                # set the same colorbar for all the matrices
                fig.colorbar(im, ax=ax, shrink=.9, orientation='horizontal')


        return fig
