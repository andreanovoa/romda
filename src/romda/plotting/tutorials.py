"""Figures shared by the tutorials: one light style and the plots that several tutorials draw.

The tutorials use a sans-serif font and no panel letters; the figures of the papers have their own style. Colours
are the Okabe-Ito palette, which stays legible for colour-blind readers: the truth is black or grey, and the
estimators take `COLORS`.

- `use_tutorial_style` sets the style for the rest of a notebook.
- `space_time` draws a field against time and space (Lorenz-96 sites, Kuramoto-Sivashinsky grid).
- `plot_error_spread` draws the error of an estimate and the spread of its ensemble against time.
- `plot_scores` draws a score against a setting: the median of the runs and their spread (band or dots).
- `plot_grouped_dots` draws a score per category for several groups, one dot per run and a bar at the median.
"""

import matplotlib.pyplot as plt
import numpy as np

__all__ = ['COLORS', 'TUTORIAL_STYLE', 'use_tutorial_style', 'space_time', 'plot_error_spread', 'plot_scores', 'plot_grouped_dots']

#: Okabe-Ito colours, by name.
COLORS = dict(blue='#0072B2', vermillion='#D55E00', green='#009E73', orange='#E69F00', sky='#56B4E9',
              purple='#CC79A7', yellow='#F0E442', grey='#808080', black='#000000')


#: The style of the tutorial figures, as matplotlib rcParams: sans serif, no top and right axes, frameless legends,
#: constrained layout. `use_tutorial_style` sets it globally; ``with plt.rc_context(TUTORIAL_STYLE):`` sets it for
#: the figures of one function only.
TUTORIAL_STYLE = {'font.family': 'sans-serif', 'mathtext.fontset': 'dejavusans', 'font.size': 10.,
                  'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                  'figure.constrained_layout.use': True}


def use_tutorial_style(font_size: float = 10.):
    """Set the style of the tutorial figures (`TUTORIAL_STYLE`) for the rest of a notebook."""
    plt.rcParams.update({**TUTORIAL_STYLE, 'font.size': font_size})


def space_time(ax, t, x, field, vmax=None, vmin=None, cmap='RdBu_r', **kwargs):
    """Space-time diagram of `field` (len(t), len(x)) on `ax`, with time on the horizontal axis.

    The colour range is symmetric, (-vmax, vmax), unless `vmin` is given (e.g. 0 for an error); `vmax` defaults to
    the largest absolute value. Returns the mesh, for a colour bar."""
    field = np.asarray(field)
    vmax = float(np.nanmax(np.abs(field))) if vmax is None else vmax
    return ax.pcolormesh(t, x, field.T, cmap=cmap, vmin=-vmax if vmin is None else vmin, vmax=vmax,
                         shading='gouraud', rasterized=True, **kwargs)


def plot_error_spread(ax, t, error, spread=None, color=COLORS['blue'], label=None, log=True):
    """The error of an estimate (line) and, if given, the spread of its ensemble (dashed, same colour) against `t`."""
    plot = ax.semilogy if log else ax.plot
    plot(t, error, color=color, label=label)
    if spread is not None:
        plot(t, spread, '--', color=color, lw=0.8)


def plot_scores(ax, x, scores, color=COLORS['blue'], label=None, band='iqr', marker='o'):
    """A score against a setting: `scores` (len(x), n_runs). The line is the median over the runs; `band='iqr'`
    shades the interquartile range, `band='range'` the smallest to the largest run, and `band='points'` draws
    every run as a dot."""
    scores = np.asarray(scores, dtype=float)
    ax.plot(x, np.nanmedian(scores, axis=1), marker=marker, ms=4, color=color, label=label)
    if band == 'points':
        ax.plot(np.repeat(x, scores.shape[1]), scores.ravel(), '.', color=color, alpha=0.5)
    elif band in ('iqr', 'range'):
        lo, hi = (np.nanquantile(scores, [0.25, 0.75], axis=1) if band == 'iqr'
                  else (np.nanmin(scores, axis=1), np.nanmax(scores, axis=1)))
        ax.fill_between(x, lo, hi, color=color, alpha=0.2, lw=0)
    elif band is not None:
        raise ValueError(f"band must be 'iqr', 'range', 'points' or None; got {band!r}")


def plot_grouped_dots(ax, categories, groups, colors=None, width=0.25):
    """A score per category for several groups: `groups` maps a label to an array (len(categories), n_runs). Each run
    is a dot and the median a horizontal bar; the groups sit side by side around each category."""
    colors = colors or list(COLORS.values())
    n = len(groups)
    for j, ((label, values), color) in enumerate(zip(groups.items(), colors)):
        values = np.asarray(values, dtype=float)
        x = np.arange(len(categories)) + width * (j - (n - 1) / 2)
        ax.plot(np.repeat(x, values.shape[1]), values.ravel(), '.', color=color, alpha=0.5)
        ax.plot(x, np.nanmedian(values, axis=1), '_', color=color, ms=14, mew=2, label=label)
    ax.set_xticks(range(len(categories)), [f'{c:g}' if isinstance(c, (int, float)) else str(c) for c in categories])
