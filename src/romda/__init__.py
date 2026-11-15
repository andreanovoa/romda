"""
romda — real-time reduced-order modelling and bias-aware data assimilation.

Main entry points
-----------------
- ``romda.models``            : physical and data-driven forecast models
- ``romda.estimators``        : Estimator hierarchy in the biasda layout — EnsembleFilter
                                (EnKF, EnSRKF, rEnKF, ...) and KalmanFilter
- ``romda.bias_estimators``   : ESN, constant, drift-linear and no-bias estimators
- ``romda.observations``      : truth/observation generation and loading

The ML building blocks (POD / SPOD / autoencoders) live in
``romda.models.data_driven`` (``autoencoders``); the ESN reservoir core comes
from the external ``echostatenetwork`` package.

romda is silent by default; ``romda.set_verbose()`` shows its diagnostic
messages, and those of dynamodels, for debugging.

"""

__version__ = "4.0.0"

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

from romda import bias_estimators, estimators, models  # noqa: E402
from romda.observations import Observations  # noqa: E402
from romda.verbose import set_verbose  # noqa: E402

__all__ = [
    "models",
    "bias_estimators",
    "estimators",
    "Observations",
    "set_verbose",
]
