from copy import deepcopy
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from romda.models import HistoryTracker, Integrator, Model
from romda.plotting import categorical_cmap


class Bias:
    r"""Base class for the model-bias estimators used in bias-aware data assimilation.

    A bias estimator provides three things to the assimilation loop:

    1. a **forecast** of the bias between analyses (`advance`), driven by its
       internal forecaster (an ESN, a constant map, a linear model, ...);
    2. the **Jacobian** of the bias with respect to the observables
       (`state_derivative`), $\matr{J} = \mathrm{d}\vect{b}/\mathrm{d}\vect{q}$,
       required by the regularized bias-aware EnKF;
    3. an **update rule** from the analysis innovation
       (`update_state_from_innovation`): a reset of the observed components to the innovation.

    The estimator state has $N_\mathrm{dim}$ components: $[\vect{b}]$ if the
    observations are unbiased, or $[\vect{b}; \vect{i}]$ (bias and innovations,
    $N_\mathrm{dim} = 2 N_q$) if `biased_observations` is set. Child classes may add
    hidden components (e.g., the ESN reservoir). Unlike ``biasda.bias.Bias``, which holds
    only the current state, the romda bias keeps its history (`hist`, `hist_t`) for the plots.

    Parameters
    ----------
    innovation : np.ndarray
        Initial innovation/bias estimate, used to set the observable dimension $N_q$.
    t : float
        Initial time.
    dt : float
        Time step of the output history.
    **kwargs
        Class-attribute overrides (see Attributes) and forecaster options.

    Notes
    -----
    The bias-estimation routes of biasda (``bias_estimation='two-stage-analysis'``, ``'joint-...'``, ...) are not
    in romda: the bias state is reset to the analysis innovation.

    Attributes
    ----------
    upsample : int
        Upsampling factor of the internal forecaster time step relative to ``dt``.
    L : int
        Number of trajectories in the training dataset (data-driven estimators).
    augment_data : bool or int
        Whether (and how much) to augment the training data.
    biased_observations : bool
        If True, the observations themselves are assumed biased and the estimator
        tracks bias and innovations separately.
    force_retrain : bool
        If True, retrain the forecaster even if a cached configuration exists.
    """

    upsample = 1
    L = 1
    N_ens = 1
    augment_data = False

    biased_observations = False  # Whether observations are biased or not
    force_retrain = False

    keys_to_print = ['upsample', 'biased_observations', 'L', 'augment_data_length']
    #: Integrator of the default forecaster (`init_forecaster`).
    integrator_class = None

    def __init__(self, innovation, t, dt, **kwargs):

        # ===================== ASSIGN PROVIDED KWARGS ======================= ##
        keys = list(kwargs.keys())
        [setattr(self, key, kwargs.pop(key)) for key in keys if hasattr(self, key)]

        # ================== Setup dimensions ================= ##

        # As dynamodels Model.dt: keep the decimals that dt itself needs (1/51200 = 1.953125e-5 needs 11),
        # up to 12; rounding to 2 extra digits made that step 2e-5 and the bias clock drift from the model's.
        self.precision_t = int(np.ceil(-np.log10(dt) + 2))
        while self.precision_t < 12 and not np.isclose(round(dt, self.precision_t), dt, rtol=1e-12, atol=0):
            self.precision_t += 1
        self.dt = np.round(dt, self.precision_t)
        self._b0 = np.asarray(innovation, dtype=float)
        self.Nq = self._format_state(innovation).shape[1]


        # ================== Initialize Forecaster & HISTORY ================= ##

        self.init_forecaster(**kwargs)

        bias_state = self.initialize_bias_state

        assert bias_state.shape[-2] == self.N, f"Bias state shape {bias_state.shape} does not match expected (Nt, N = {self.N}, Nens)."
        self.update_history(bias_state, t=t, reset=True)


    @property
    def name(self):
        return self.__class__.__name__

    def advance(self, Nt):
        """Forecast the bias state `Nt` steps forward, appending the forecast to the history."""
        b, t_b = self.time_integrate(Nt=Nt)
        self.update_history(b, t_b)

    def update_state(self, state, t=None):
        """Overwrite the current estimator state (e.g., with the analysis innovation)."""
        self.update_history(state, t=t, modify_saved_states=True)

    @property
    def augment_data_length(self):
        augment_data = self.augment_data
        if augment_data:
            if isinstance(augment_data, int) and augment_data > 1:
                return augment_data
            return 2
        return 1

    @property
    def initialize_bias_state(self):
        """Initial state (N_dim, N_ens) from the construction innovation; [bias; innovations] starts both blocks from it."""
        b0 = self._b0
        if b0.ndim == 3:
            b0 = b0[0]
        elif b0.ndim == 1:
            b0 = b0[:, np.newaxis]

        if b0.shape[-1] != self.N_ens:
            b0 = np.repeat(np.mean(b0, axis=-1, keepdims=True), self.N_ens, axis=1)

        if self.biased_observations and b0.shape[0] == self.Nq:
            b0 = np.concatenate([b0, b0], axis=0)

        assert b0.shape[0] == self.N_dim, \
            f'Initial bias state has {b0.shape[0]} components, expected N_dim={self.N_dim}.'
        return b0

    @property
    def history(self) -> HistoryTracker:
        return self.forecaster.history  # type: ignore

    @property
    def integrator(self) -> Integrator:
        """Integrator of the forecaster."""
        return self.forecaster.integrator  # type: ignore

    @property
    def N_dim(self):
        if not self.biased_observations:
            return self.Nq
        else:
            return 2 * self.Nq

    @property
    def N(self):
        return self.N_dim + self.N_hidden

    @property
    def N_hidden(self):
        # Number of hidden units in the bias model, e.g., for ESN bias model. This is added to the state dimension N_dim to get the total state dimension N.
        return 0


    @property
    def bias_idx(self):
        return np.arange(self.Nq)

    @property
    def observed_idx(self):
        if self.biased_observations:
            return self.bias_idx + self.Nq
        else:
            return self.bias_idx

    def init_forecaster(self, **kwargs):
        """Set `self.forecaster`, which holds the `history` and the `integrator`; by default, a namespace
        with an empty history and an `integrator_class` integrator."""
        self.forecaster = SimpleNamespace(history=HistoryTracker(initial_capacity=kwargs.pop('initial_capacity', 1000)))
        self.forecaster.integrator = self.integrator_class(self)

    def state_derivative(self):
        """
        Returns the derivative of the bias state, which is used for time integration.
        This is computed by the forecaster model.
        """
        raise NotImplementedError('Bias child classes must implement state_derivative property, typically computed by the forecaster model.')

    def _format_state(self, b):
        """
        Ensure b has shape (nt, nb, nens)
        """
        if b.ndim == 3:
            return b # already (nt, nb, nens)
        elif b.ndim == 1:
            b_repeated = np.repeat(b[:, np.newaxis], self.N_ens, axis=1)  # (nb,) -> (nb, nens)
            return b_repeated[np.newaxis, :, :]  # (nb, nens) -> (1, nb, nens) Add extra dimension for time

        elif b.ndim == 2 and (b.shape[-1] == self.N_ens or b.shape[0] == self.N_dim):
            return b[np.newaxis, :, :]  # (nb, nens) -> (1, nb, nens) # Add extra dimension for time
        elif b.ndim == 2 and b.shape[-1] != self.N_ens:
            return np.repeat(b[:, :, np.newaxis], self.N_ens, axis=2)  # (nt, nb) -> (nt, nb, nens)
        else:
            raise AssertionError(f'b must have 1, 2 or 3 dimensions, got {b.ndim}=({b.shape})')

    @property
    def hist(self):
        """Returns only the valid (non-empty) portion of the history buffer."""
        return self.history.hist

    @property
    def hist_t(self):
        """Returns only the valid portion of the time history."""
        return self.history.hist_t

    @property
    def current_state(self):
        """Returns the current state (last entry in history)."""
        return self.history.current_state.copy()

    @property
    def current_time(self):
        """Returns the current time (last entry in time history)."""
        return self.history.current_time


    @property
    def config(self):
        _config = dict()
        for key in self.keys_to_print:
             if hasattr(self, key):
                _config[key] = getattr(self, key)

        return _config

    @property
    def current_bias(self):
        """Returns the current (ensemble-mean) bias computed from the current state.
        Shape: (Nq, 1) -- the bias is defined on the ensemble mean."""
        return self.get_bias(state=self.current_state, mean=True)[0, :, :]

    @property
    def current_innovations(self):
        """Returns the current (ensemble-mean) innovations computed from the current state.
        Shape: (Nq, 1)."""
        return self.get_innovations(state=self.current_state, mean=True)[0, :, :]


    def get_bias(self, state, mean=False):
        state = self._format_state(state)
        if mean:
            state = np.mean(state, axis=-1, keepdims=True)

        return state[:, self.bias_idx, :]


    def get_innovations(self, state, mean=False):
        state = self._format_state(state)
        if mean:
            state = np.mean(state, axis=-1, keepdims=True)

        return state[:, self.observed_idx, :]


    def get_bias_hist(self, mean=False):
        return self.get_bias(state=self.hist, mean=mean)

    def time_integrate(self, Nt):
        return self.integrator.advance(Nt=Nt)


    def update_history(self, state, t=None, reset=False, modify_saved_states=False, **kwargs):
        state = self._format_state(state) # Ensure shape (nt, nstate, nens)

        # Ensure time array matches nt
        if t is None:
            t = (np.arange(state.shape[0]) * self.dt).round(self.precision_t) + self.current_time

        if modify_saved_states:
            assert state.shape[0] == 1, "When modify_saved_states is True, state must have only one time step (shape[0] == 1)."

        self.history.update_history(state, t=t, reset=reset, modify_saved_states=modify_saved_states)


    def update_state_from_innovation(self, input_innovation):
        """Update the estimator state from an innovation: assign it to the observed components.

        Parameters
        ----------
        input_innovation : np.ndarray
            Innovation ensemble, shape ``(Nq, m)``, ``(Nq, 1)`` or ``(Nq,)``.

        Returns
        -------
        np.ndarray
            Updated estimator state, shape ``(N, N_ens)``.
        """
        # The innovation is (Nq, m), (Nq, 1) or (Nq,). `_format_state` guesses the layout from
        # N_dim and N_ens, and with biased observations (N_dim = 2 Nq) and one bias state it read
        # an (Nq, m) ensemble as m time steps of an Nq = m state, so the shape is fixed here.
        input_innovation = np.asarray(input_innovation, dtype=float)
        if input_innovation.ndim == 3:
            assert input_innovation.shape[0] == 1, "The innovation must have one time step."
            input_innovation = input_innovation[0]
        if input_innovation.ndim == 1:
            input_innovation = input_innovation[:, np.newaxis]
        assert input_innovation.shape[0] == len(self.observed_idx), \
            f"Innovation has {input_innovation.shape[0]} rows, expected Nq = {len(self.observed_idx)}."

        updated_state = self.current_state.copy()
        if input_innovation.shape[-1] == self.N_ens:
            # One innovation per bias-ensemble member: assign directly
            updated_state[self.observed_idx, :] = input_innovation
        elif self.N_ens == 1 or input_innovation.shape[-1] == 1:
            # Assign the same mean innovation to all ensemble members
            updated_state[self.observed_idx, :] = np.repeat(np.mean(input_innovation, axis=-1, keepdims=True),
                                                            self.N_ens, axis=-1)
        else:
            raise ValueError(f'Innovation ensemble of size {input_innovation.shape[-1]} does not '
                             f'match the bias ensemble of size {self.N_ens}.')
        return updated_state

    def print_bias_parameters(self, indent=2):
        print(f'\n{self.__class__.__name__}')
        print(f'{"=" * len(self.__class__.__name__)}')
        for key in sorted(set(self.keys_to_print)):
            if hasattr(self, key):
                val = getattr(self, key)
                if type(val) is float:
                    print(f'{" " * indent}{key} = {val:.6}')
                else:
                    print(f'{" " * indent}{key} = {val}')
        if isinstance(self.forecaster, Model):
            print(f'{" " * indent}====Forecaster model ====')
            self.forecaster.print_parameters(show_header=False,
                                             indent=indent*2)


    def copy(self):
        return deepcopy(self)

    def close(self):
        """Close any resources used by the bias model, e.g., forecaster model."""
        if hasattr(getattr(self, 'forecaster', None), 'close'):
            self.forecaster.close()


    # ================= Visualization methods ================== #
    def visualize_bias_and_innovations(self, state=None, t=None, plot_members=True, ylims=None):
        if state is None:
            state = self.hist

        inn = self.get_innovations(state=state, mean=not plot_members)
        b = self.get_bias(state=state, mean=not plot_members)

        if t is None:
            t = self.hist_t[-len(inn):]

        lbls =( [f'inn_{ii}' for ii in range(inn.shape[1])],
              [f'b_{ii}' for ii in range(b.shape[1])])


        t_mid = len(t) - len(t)//8
        lims = [[0, t_mid ],  [t_mid, len(t)-1]  ]

        Nq = inn.shape[1]
        nens = inn.shape[-1]
        cols = categorical_cmap(nc=2, nsc=nens)
        cols = [cols[ii * nens:(ii + 1) * nens] for ii in range(2)]
        # Plot the time evolution of the observables
        for y, lbl, cs in zip([inn, b], lbls, cols):

            fig = plt.figure(figsize=(8, Nq+1), layout="constrained")

            axs = fig.subplots(Nq, 2, sharey='row', sharex='col')
            if self.Nq == 1:
                axs = [axs]


            for ii, ax in enumerate(axs):
                for jj, lim in enumerate(lims):
                    lines = ax[jj].plot(t[lim[0]:lim[1]], y[lim[0]:lim[1], ii])
                    for line, color in zip(lines, cs):
                        line.set_color(color)
                    if ii == self.Nq-1:
                        ax[jj].set(xlabel='$t$', xlim=[t[lim[0]], t[lim[1]]])

                ax[0].set(ylabel=lbl[ii])
                if ylims is not None:
                    ax[0].set(ylim=ylims)
