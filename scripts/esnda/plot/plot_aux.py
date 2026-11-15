"""Shared set-up of the figures of the washout-free paper: paths, style, colours and the loaders of more than one figure.

Every plot_*.py imports this module first. It puts run/ (the run scripts) on the path, so a figure
imports the run script whose functions or constants it needs, selects the Agg backend and applies STYLE.

PAPER is the folder that receives the PDFs: $PAPER, else the paper's figs/ on the workstation that ran the paper,
else figs/ here (reproduce.sh sets PAPER to figs/ here).
"""

import glob
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from romda.run import load_config  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/esnda
RUN = os.path.join(HERE, 'run')                                      # the run scripts
FIGS = os.path.join(HERE, 'figs')                                    # the figures when $PAPER is unset
RESULTS = os.path.join(HERE, 'results')                              # the records of the runs, one folder per study
CONFIGS = os.path.join(HERE, 'configs')                              # the configurations of the experiments
if RUN not in sys.path:
    sys.path.insert(0, RUN)
PAPER = os.environ.get('PAPER', FIGS)                                 # where the figures are saved

STYLE = {'font.family': 'serif', 'font.serif': ['TeX Gyre Termes', 'STIXGeneral'], 'mathtext.fontset': 'stix',
         'pdf.fonttype': 42, 'font.size': 10, 'legend.fontsize': 10, 'axes.spines.top': False,
         'axes.spines.right': False}
plt.rcParams.update(STYLE)

#: colours shared by the figures: the true state (grey), the data and observed sites (the pink of Fig. 1), the
#: ensemble mean (blue), the spread (orange), the reservoir (purple) and a member off the attractor (Paul Tol red)
BLUE, ORANGE, VERMILION, TOL_ORANGE = '#0072B2', '#E69F00', '#D55E00', '#EE7733'   # Okabe-Ito; Paul Tol's orange
C_TRUE, C_OBS, C_MEAN, C_SPREAD, C_RES, C_DIVERGED = '0.45', '#EE3075', BLUE, ORANGE, '#6a51a3', '#CC3311'
C_TRUE_DARK = '0.2'    # the true system under the dashed networks of fig:verification
SPREAD_SYNC = 0.35     # the synchronization level: the first time the normalized spread (t_s) or error (t_e) falls below it
TOP = 2.0              # error panels: errors above sit on the top edge


def noise_level(config):
    """The noise of the data of configs/<config>.yml (truth.noise_level), normalized as the error."""
    return load_config(os.path.join(CONFIGS, f'{config}.yml'))['truth']['noise_level']


NOISE_LEVEL = noise_level('lorenz96_parallel')   # the noise of the Lorenz-96 data


def style(**overrides):
    """Apply STYLE with the given rcParams changed."""
    plt.rcParams.update({**STYLE, **overrides})


def reference_lines(ax, noise, climatology=True):
    """The reference levels of the error panels: climatology (e = 1 by the normalization), the noise of the data,
    normalized as the error, and the synchronization level SPREAD_SYNC, each named; `climatology`: name it."""
    ax.axhline(1.0, color='k', lw=0.6, ls=':')
    ax.axhline(noise, color='0.3', lw=0.6, ls='--')
    ax.axhline(SPREAD_SYNC, color='0.3', lw=0.6, ls='-.')
    labels = [(0.98, 1.0, 'climatology', 'right')] if climatology else []
    for x, y, name, ha in labels + [(0.02, noise, 'noise', 'left'), (0.98, SPREAD_SYNC, 'sync', 'right')]:
        ax.text(x, y, name, transform=ax.get_yaxis_transform(), ha=ha, va='bottom', color='0.35', fontsize=9)


def letter(i):
    """Bold panel label (a), (b), ... of panel i."""
    return rf'$\mathbf{{({"abcdefgh"[i]})}}$'


def panel_letter(ax, i, fontsize=9, offset=(0, 1.04), points=(0, 0), xycoords='axes fraction', va='bottom'):
    """Write the bold label of panel i on ax, left-aligned at `offset` (in `xycoords`, axes fraction by default),
    shifted by `points` (in points)."""
    return ax.annotate(letter(i), offset, xycoords=xycoords, xytext=points, textcoords='offset points',
                       fontsize=fontsize, va=va, ha='left')


def save(fig, name, **kwargs):
    """Write PAPER/name (bbox_inches='tight' unless given) and print the path."""
    os.makedirs(PAPER, exist_ok=True)
    path = os.path.join(PAPER, name)
    fig.savefig(path, **{'bbox_inches': 'tight', **kwargs})
    print('wrote', path)
    return path


# ── the L96-40p sweeps of run/run_parallel_localization.py (Figs. 6 and 10), in results/localization ─────────────────────────────────

NOISE_CDD = 'localization_parallel_lorenz96_w20'           # L96-40p: 5;2;500 with the 20-step training washout
GLOBAL = 'localization_parallel_lorenz96_g40_h0_n6000'     # L96-40s: one reservoir for the 40 sites
FILTER_KEYS = ('n_sensors', 'window', 'members', 'radius', 'inflation', 'n_data', 'first_error', 'valid_lyap',
               'window_error', 'feed_error')


def selection(z, s, m, radius):
    """Rows of sensor layout s, N_e = m and the given radius (nan: global)."""
    return (z['n_sensors'] == s) & (z['members'] == m) & (np.isnan(z['radius']) if np.isnan(radius)
                                                          else z['radius'] == radius)


def load(base, tag='_innov'):
    """Test rows of a sweep (windows 0-9) with the innovation-based inflation, with the filter rows of the extra
    ensemble sizes of <base><tag>_more*.npz (TAG=_innov_more MEMBERS=80,160 ... run; TAG=_innov_more5 MEMBERS=5)
    appended."""
    z = dict(np.load(os.path.join(RESULTS, 'localization', base + tag + '.npz')))
    for more in sorted(glob.glob(os.path.join(RESULTS, 'localization', base + tag + '_more*.npz'))):
        e = np.load(more)
        z.update({k: np.concatenate([z[k], e[k]]) for k in FILTER_KEYS})
    return z
