"""Error taxonomy (CONTRACTS §0) + a single logger factory.

One base class so callers can catch everything from this package with
``except ScaleForecastError``. Subclasses are intentionally few and boring — add one
only when a caller would branch on it, not for decoration.
"""

from __future__ import annotations

import logging


class ScaleForecastError(Exception):
    """Base class for every error raised by this package."""


class ConfigError(ScaleForecastError):
    """The run config is missing, malformed, or internally inconsistent."""


class DataError(ScaleForecastError):
    """The input series data violates the contract the config declares.

    Raised by the pre-flight validator (``validation.py``) *before* any compute fans
    out, so a shape problem in the source (missing column, gap in a series, wrong
    freq) surfaces as one clear message naming the offender instead of thousands of
    failed cells.
    """


class ModelError(ScaleForecastError):
    """A model failed to fit or predict.

    Note: inside a worker cell this is *captured* into the CellResult, never raised
    out of ``run_cell`` (CONTRACTS §3.3). It is raised only in direct/unit use.
    """


class RegistryError(ScaleForecastError):
    """A BigQuery/GCS registry operation failed (table, write, or artifact)."""


class EngineError(ScaleForecastError):
    """A compute engine (Spark/Ray/BigQuery) failed to launch or collect results."""


def get_logger(name: str) -> logging.Logger:
    """Return a package logger.

    We attach a single stream handler once so library use doesn't duplicate lines,
    and leave the level to the root/app config (default WARNING) unless overridden.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.propagate = False
    return logger
