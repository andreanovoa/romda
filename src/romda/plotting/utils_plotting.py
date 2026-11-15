
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors


class Palette:
    TRUE = "#808080"
    UNBIASED = "#000080"
    BIASED = "#20B2AA"

    OBS = "#D62728"

    PARAMS = ["#2CA02C", "#F4A460", "#B3B3DE", "#1F77B4", "#D62728", "#FFD700", "#00BFFF"]

    @staticmethod
    def get_color(name: str, alpha: float = 1.0):
        val = getattr(Palette, name.upper())
        if isinstance(val, list):
            return [colors.to_rgba(c, alpha) for c in val]
        return colors.to_rgba(val, alpha)

    @staticmethod
    def get_color_params(n=-1, alpha: float = 1.0):
        return [colors.to_rgba(c, alpha) for c in Palette.PARAMS[:n]]

    y_unbias_props = dict(marker='none', linestyle='--', dashes=(10, 1), lw=.8, color=colors.to_rgba(UNBIASED))
    y_biased_props = dict(marker='none', linestyle='-', lw=.8, color=colors.to_rgba(BIASED, .7))
    y_biased_mean_props = dict(marker='none', linestyle='--', dashes=(2, .5), lw=1, color=colors.to_rgba(BIASED))
    true_noisy_props = dict(marker='none', linestyle='-', lw=1.2, color=colors.to_rgba(TRUE, .7))
    true_props = dict(marker='none', linestyle='-', lw=2, color=colors.to_rgba(TRUE, .7))
    obs_props = dict(marker='.', linestyle='none', markersize=7, markeredgecolor='none', color=colors.to_rgba(OBS))


def use_times_font():
    """Set the text of matplotlib figures in Times, with STIX mathematics to match (the style of the figures of
    the papers). Falls back on STIX and then the default serif where Times is not installed."""
    plt.rcParams.update({'font.family': 'serif',
                         'font.serif': ['Times New Roman', 'Times', 'STIXGeneral', 'DejaVu Serif'],
                         'mathtext.fontset': 'stix'})


def categorical_cmap(nc, nsc):
    """Colours, shape ``(nc * nsc, 3)``, for `nc` tab10 categories of `nsc` lightening shades each."""
    if nc > plt.get_cmap('tab10').N:
        raise ValueError("Too many categories for colormap.")
    ccolors = plt.get_cmap('tab10')(np.arange(nc, dtype=int))
    cols = np.zeros((nc * nsc, 3))
    for i, c in enumerate(ccolors):
        chsv = colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv, nsc).reshape(nsc, 3)
        arhsv[:, 1] = np.linspace(chsv[1], 0.25, nsc)
        arhsv[:, 2] = np.linspace(chsv[2], 1, nsc)
        rgb = colors.hsv_to_rgb(arhsv)
        cols[i * nsc:(i + 1) * nsc, :] = rgb
    return cols
