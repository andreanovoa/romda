"""Verbosity of romda.

romda writes its diagnostic messages (e.g. an ESN configuration loaded from the store, the noise added to the
observations, a rejected analysis) to the logger ``'romda'`` and its children, which are silent by default; its
physical models and their integrators log to ``'dynamodels'`` (e.g. the start of the multiprocessing pool of an
ensemble), and its echo state networks to ``'echostatenetwork'`` (e.g. the hyperparameter search and the summary of
a training). `set_verbose` shows the three on stderr; an application that configures `logging` itself can set the
levels of these loggers instead (with both, its own handlers print every message a second time).
"""

import importlib
import logging

__all__ = ['set_verbose']

PACKAGES = ('romda', 'dynamodels', 'echostatenetwork')


class _PrefixFormatter(logging.Formatter):
    """Prefix every line of a record with the name of the package, so that a multi-line report reads as one block."""

    def __init__(self, package):
        super().__init__('%(message)s')
        self.prefix = f'[{package}] '

    def format(self, record):
        return '\n'.join(self.prefix + line for line in super().format(record).split('\n'))


def _by_name(package, on):
    """Switch the logger of `package` by name: one named stream handler on stderr and DEBUG, or the default level."""
    logger = logging.getLogger(package)
    name = f'{package}.verbose'                         # the handler of <package>.set_verbose has the same name
    handlers = [h for h in logger.handlers if h.get_name() == name]
    if on:
        if not handlers:
            handler = logging.StreamHandler()
            handler.set_name(name)
            handler.setFormatter(_PrefixFormatter(package))
            logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    else:
        for handler in handlers:
            logger.removeHandler(handler)
        logger.setLevel(logging.NOTSET)


def set_verbose(on: bool = True) -> None:
    """Show (``on=True``) or silence (``on=False``) the reports and diagnostic messages of romda, dynamodels and
    echostatenetwork.

    dynamodels and echostatenetwork are switched by their own ``set_verbose`` when the installed version has one, and
    by the name of their logger otherwise. On: each logger takes the level DEBUG and one stream handler on stderr,
    which prefixes every line with ``[<package>]``; calling it again, or calling the ``set_verbose`` of dynamodels or
    echostatenetwork, adds no second handler. Off: the handlers are removed and the levels return to the default.
    """
    for package in PACKAGES:
        try:
            own = None if package == 'romda' else getattr(importlib.import_module(package), 'set_verbose', None)
        except ImportError:
            own = None
        if own is not None:
            own(on)
        else:
            _by_name(package, on)
