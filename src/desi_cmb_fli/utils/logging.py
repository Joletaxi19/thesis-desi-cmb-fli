"""Small helper for consistent, idempotent logging setup.

We deliberately avoid global side effects and only attach a single stream
handler per logger instance. Repeated calls to :func:`setup_logger` will not
stack multiple handlers, which prevents duplicate log lines in notebooks or
test runs.
"""

from __future__ import annotations

import logging


def setup_logger(name: str = "desi_cmb_fli", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger with a single stream handler.

    Parameters
    ----------
    name
        Logger name. Use sub-names (e.g. ``"desi_cmb_fli.analysis"``) to get a
        hierarchical logger that inherits level and handlers.
    level
        Logging level for the created handler, defaults to ``INFO``.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Notes
    -----
    If the named logger already has handlers, this function leaves them alone
    to avoid attaching duplicate handlers. This makes it safe to call from
    multiple modules.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        ch = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(levelname)s] %(asctime)s - %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger
