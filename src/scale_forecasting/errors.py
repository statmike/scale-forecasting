"""Error taxonomy + a single logger factory.

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
    out of ``run_cell``. It is raised only in direct/unit use.
    """


class RegistryError(ScaleForecastError):
    """A BigQuery/GCS registry operation failed (table, write, or artifact)."""


class EngineError(ScaleForecastError):
    """A compute engine (Spark/Ray/BigQuery) failed to launch or collect results."""


class JobIdTaken(EngineError):
    """The platform already holds the job id this submit asked for.

    Raised by every runtime submitter that names its own job, in place of the platform's raw
    ``ALREADY_EXISTS``. Two callers branch on it, which is why it exists rather than being a bare
    message: `registry.lifecycle.run_job` stamps a distinct ``failure_reason`` token so the clash is
    legible in the registry rather than only in the launcher's stdout, and the operator needs to be
    told which of the two ways out applies.

    **Why it happens at all.** The attempt counter is derived from the registry
    (`registry.jobs.next_job_attempt` → ``MAX(attempt)``), while uniqueness is enforced by the
    platform. Those agree only as long as every job that reached a platform also wrote a
    ``run_jobs`` row — and the emitted-command path (`launch_plan.stage_run`, the Airflow emitter)
    hands a runnable command to something that is not the launcher. Observed live 2026-09-22 on a
    ``--force`` re-run whose id had been created four hours earlier by a pasted command; see
    ``docs/validation.md``. `launch_plan.stage_run` now files an ``EMITTED`` row precisely so the
    counter can see that launch, and `job_launch` walks forward past a taken id on the force path,
    so reaching this error means both of those missed — a job created by something with no access
    to this registry at all.
    """


def get_logger(name: str) -> logging.Logger:
    """Return a package logger.

    We attach a single stream handler once so library use doesn't duplicate lines,
    and leave the level to the root/app config (default WARNING) unless overridden.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    return logger
