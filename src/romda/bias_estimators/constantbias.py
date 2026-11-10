import numpy as np
from romda.models import ConstantIntegrator

from .bias import Bias


class ConstantBias(Bias):
    r"""
    Constant (persistent) bias estimator.

    The bias is held constant between analysis steps, i.e., the forecast model of the
    bias is $\dot{\vect{b}} = \vect{0}$. At each analysis step, the bias state is
    reset to the latest innovation (see `Bias.update_state_from_innovation`). This is
    the classic persistent-bias assumption.

    Parameters
    ----------
    innovation : np.ndarray
        Initial innovation/bias estimate, shape (Nq,), (Nq, N_ens) or (1, Nq, N_ens).
    t : float
        Initial time.
    dt : float
        Time step of the output history.
    k : float or np.ndarray, optional
        If provided, the initial bias state is set to the constant value(s) k instead of
        the provided innovation.
    """

    integrator_class = ConstantIntegrator   # the bias is held in time

    def __init__(self, innovation, t, dt, k=None, **kwargs):

        if k is not None:
            innovation = np.ones(np.shape(innovation)) * k  # Constant initial bias

        super().__init__(innovation=innovation, t=t, dt=dt, **kwargs)

    def state_derivative(self):
        n_bias = self.Nq
        return np.zeros([n_bias, n_bias])


class NoBias(ConstantBias):
    """
    Placeholder bias estimator that always returns zero bias.
    Useful to run the bias-aware machinery in its unbiased limit.
    """

    def __init__(self, innovation, t, dt, **kwargs):
        kwargs.pop('k', None)
        super().__init__(innovation=innovation, t=t, dt=dt, k=0., **kwargs)

    def update_state_from_innovation(self, input_innovation):
        # NoBias never updates: the bias remains identically zero
        return self.current_state

