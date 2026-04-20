"""Startup logging helpers for noisy third-party dependencies."""

from __future__ import annotations

import logging
import warnings

_NOISY_LOGGERS = ("pyjabber", "winloop", "asyncio")
_IGNORED_ROOT_MESSAGES = {"Unknown stanza interface: id"}
_INSTALLED = False


class _ExactMessageFilter(logging.Filter):
    """Drop only known-harmless log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            return record.getMessage() not in _IGNORED_ROOT_MESSAGES
        except Exception:
            return True


def install_startup_logging_filters() -> None:
    """Suppress dependency noise without hiding unrelated failures."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    for logger_name in _NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)

    warnings.filterwarnings("ignore", message="Unknown stanza interface")
    logging.getLogger().addFilter(_ExactMessageFilter())
