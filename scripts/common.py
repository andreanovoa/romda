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

import romda.run  # noqa: F401  (the !hz tag of the configurations)

#: this folder: configurations are named relative to it, e.g. 'examples/lorenz/lorenz63'
SCRIPTS_FOLDER = os.path.dirname(os.path.abspath(__file__))


def load_main_config(name: str) -> dict:
    """The configuration ``scripts/<name>.yml`` of a case (e.g. 'thermoacoustics/annular')."""
    name = name.removesuffix('.yml')
    with open(os.path.join(SCRIPTS_FOLDER, f'{name}.yml')) as f:
        return yaml.safe_load(f)


def output_folders(script: str) -> tuple:
    """The ``(results, figs)`` folders next to `script` (pass ``__file__``), with a trailing separator."""
    here = os.path.dirname(os.path.abspath(script))
    return os.path.join(here, 'results', ''), os.path.join(here, 'figs', '')


def print_metrics(metrics: dict, header: str = 'Metrics') -> None:
    print(f'\n ------------------ {header} ------------------ ')
    for key, val in metrics.items():
        print(f'\t {key} = {val:.4f}' if isinstance(val, float) else f'\t {key} = {val}')


def main_parser(description: str = None) -> argparse.ArgumentParser:
    """The command line of the case scripts, ``--save-figs``, to which a script adds its own arguments."""
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--save-figs', action='store_true',
                        help='save the figures to figs/ (next to the script or the config) instead of showing them')
    return parser


def show_or_save_figs(save: bool, prefix: str, figs_folder: str) -> None:
    """Either open the interactive figures or save them as PNGs in `figs_folder`, named
    ``<prefix>_<label>.png`` after the label of the figure, or its number when it has none."""
    if not save:
        plt.show()
        return

    os.makedirs(figs_folder, exist_ok=True)
    for num in plt.get_fignums():
        fig = plt.figure(num)
        path = os.path.join(figs_folder, f'{prefix}_{fig.get_label() or f"{num:02d}"}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'Saved {path}')
