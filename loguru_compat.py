"""
loguru_compat.py — stdlib logging shim that mimics the loguru API.

Installed as a fallback when the `loguru` package is not available.
Covers the subset of loguru used in this codebase:
    from loguru import logger
    logger.info / debug / warning / error / critical
"""

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_log = logging.getLogger("xaisdn")


class _LoguruCompat:
    """Drop-in replacement for loguru.logger with stdlib logging backend."""

    def info(self, msg, *args, **kwargs):
        _log.info(str(msg), *args)

    def debug(self, msg, *args, **kwargs):
        _log.debug(str(msg), *args)

    def warning(self, msg, *args, **kwargs):
        _log.warning(str(msg), *args)

    def warn(self, msg, *args, **kwargs):
        _log.warning(str(msg), *args)

    def error(self, msg, *args, **kwargs):
        _log.error(str(msg), *args)

    def critical(self, msg, *args, **kwargs):
        _log.critical(str(msg), *args)

    def exception(self, msg, *args, **kwargs):
        _log.exception(str(msg), *args)

    def success(self, msg, *args, **kwargs):
        _log.info(str(msg), *args)

    def add(self, *args, **kwargs):
        pass  # no-op; loguru add() for sinks

    def remove(self, *args, **kwargs):
        pass

    def bind(self, **kwargs):
        return self

    def opt(self, **kwargs):
        return self


logger = _LoguruCompat()
