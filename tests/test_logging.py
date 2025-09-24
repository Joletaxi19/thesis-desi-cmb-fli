"""Minimal test for the shared logging helper."""

import logging

from desi_cmb_fli.utils.logging import setup_logger


def test_setup_logger_idempotent():
    logger = setup_logger("desi_cmb_fli.test", level=logging.DEBUG)
    # Calling twice should not add duplicate handlers
    setup_logger("desi_cmb_fli.test", level=logging.DEBUG)
    assert len(logger.handlers) == 1
    assert logger.level == logging.DEBUG
