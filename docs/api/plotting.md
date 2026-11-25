# Plotting

## Summary

**Files:** `src/romda/plotting/tutorials.py`, `pod.py`, `plot_results.py`, `utils_plotting.py`

`romda.plotting.tutorials` holds the figures that several tutorials share, in one light style: a sans-serif font,
no top and right axes, frameless legends and the Okabe-Ito colours (`COLORS`). The layouts of a single tutorial stay
in the helper module of its topic (`tutorials/<topic>/aux_<n>X.py`).

| Function | Draws |
|---|---|
| `use_tutorial_style()` | the style, for the rest of a notebook |
| `space_time(ax, t, x, field)` | a field against time and space, with a symmetric colour range (or from `vmin`, for an error) |
| `plot_error_spread(ax, t, error, spread)` | the error of an estimate and the spread of its ensemble against time |
| `plot_scores(ax, x, scores, band='iqr')` | a score against a setting: the median of the runs and their interquartile range, full range or dots |
| `plot_grouped_dots(ax, categories, groups)` | a score per category for several groups, one dot per run and a bar at the median |

```python
import matplotlib.pyplot as plt
from romda.plotting import COLORS, plot_scores, use_tutorial_style

use_tutorial_style()
fig, ax = plt.subplots()
plot_scores(ax, dt_obs, errors, color=COLORS['blue'], label='EnSRKF')   # errors: (len(dt_obs), n_runs)
```

## API

::: romda.plotting.tutorials
