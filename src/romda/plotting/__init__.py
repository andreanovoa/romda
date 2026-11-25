from .plot_results import plot_train_dataset
from .tutorials import (
           COLORS,
           TUTORIAL_STYLE,
           plot_error_spread,
           plot_grouped_dots,
           plot_scores,
           space_time,
           use_tutorial_style,
)
from .utils_plotting import Palette, categorical_cmap

__all__ = ['COLORS', 'Palette', 'TUTORIAL_STYLE', 'categorical_cmap', 'plot_error_spread', 'plot_grouped_dots', 'plot_scores',
           'plot_train_dataset', 'space_time', 'use_tutorial_style']
