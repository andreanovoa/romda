# Linear model

`LinearModel` advances reduced coordinates with a linear map and optional process noise,
which makes it useful for Kalman-filter examples and baseline forecasts.

$$
\vect{\psi}_{t+1} = \matr{F}\vect{\psi}_t + \vect{\eta}_t.
$$

::: romda.models.data_driven.linear_model.LinearModel
