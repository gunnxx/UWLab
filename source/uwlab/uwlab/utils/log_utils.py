# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Console logging for the UW Lab packages, inside and outside Omniverse Kit."""

import logging
import sys

_HANDLER_NAME = "uwlab_console"
"""Name of the stdout handler, so repeated calls do not stack handlers."""

_PACKAGE_LOGGERS = ("uwlab", "uwlab_assets", "uwlab_tasks", "__main__")
"""Loggers that receive the handler: the three packages and the running script (``logging.getLogger(__name__)``)."""


def configure_console_logging(level: int = logging.INFO):
    """Show the log records of the UW Lab packages on stdout.

    Under Omniverse Kit the root logger carries carb's handler, which shows WARNING and above, and Isaac Lab's
    INFO handler, which only passes records of ``isaaclab*`` loggers, so INFO records of the ``uwlab*`` packages
    and of the running script are dropped. Outside Kit nothing is configured at all. This attaches one stdout
    handler to those loggers and stops them propagating to the root logger, so a warning is shown once. Calling
    it again has no effect.

    Args:
        level: Threshold applied to the loggers and the handler.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.name = _HANDLER_NAME
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("[%(levelname)s]: %(message)s"))
    for name in _PACKAGE_LOGGERS:
        package_logger = logging.getLogger(name)
        if not any(getattr(existing, "name", None) == _HANDLER_NAME for existing in package_logger.handlers):
            package_logger.addHandler(handler)
        if package_logger.level == logging.NOTSET or package_logger.level > level:
            package_logger.setLevel(level)
        package_logger.propagate = False
