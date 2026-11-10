"""Shared helpers of the examples.

These utilities load the YAML configuration of a case, print its metrics, and
show or save its figures, for every topic (Lorenz twin experiments,
thermoacoustics, POD-ESN cylinder wake). Every case writes its outputs next to
its own script, in the gitignored folders ``results/`` and ``figs/``.
"""

import argparse
import os

import matplotlib.pyplot as plt
import yaml

#: this folder: configurations are named relative to it, e.g. 'lorenz/lorenz63'
EXAMPLES_FOLDER = os.path.dirname(os.path.abspath(__file__))


def load_main_config(name: str) -> dict:
    """The configuration ``examples/<name>.yml`` of a case (e.g. 'thermoacoustics/annular')."""
    name = name.removesuffix('.yml')
    with open(os.path.join(EXAMPLES_FOLDER, f'{name}.yml')) as f:
        return yaml.safe_load(f)


def output_folders(script: str) -> tuple:
    """The ``(results, figs)`` folders next to `script` (pass ``__file__``), with a trailing separator."""
    here = os.path.dirname(os.path.abspath(script))
    return os.path.join(here, 'results', ''), os.path.join(here, 'figs', '')


def print_metrics(metrics: dict, header: str = 'Metrics') -> None:
    print(f'\n ------------------ {header} ------------------ ')
    for key, val in metrics.items():
        print(f'\t {key} = {val:.4f}' if isinstance(val, float) else f'\t {key} = {val}')


def parse_main_args(description: str = None) -> argparse.Namespace:
    """Common command-line interface of the case scripts."""
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--save-figs', action='store_true',
                        help='save the figures to figs/ next to the script instead of showing them')
    return parser.parse_args()


def show_or_save_figs(save: bool, prefix: str, figs_folder: str) -> None:
    """Either open the interactive figures or save them as PNGs in `figs_folder`."""
    if not save:
        plt.show()
        return

    os.makedirs(figs_folder, exist_ok=True)
    for num in plt.get_fignums():
        fig = plt.figure(num)
        path = os.path.join(figs_folder, f'{prefix}_{num:02d}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'Saved {path}')
