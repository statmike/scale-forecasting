"""Error taxonomy + a single logger factory.

One base class so callers can catch everything from this package with
``except ScaleForecastError``. Subclasses are intentionally few and boring — add one
only when a caller would branch on it, not for decoration.
"""

from __future__ import annotations

import logging
import os


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


class MissingExtraError(ScaleForecastError, ImportError):
    """A feature needs an optional-dependency extra that is not installed.

    Both a `ScaleForecastError` and an ``ImportError``, so ``except ScaleForecastError`` and the
    idiomatic ``except ImportError`` each catch it. The message is the fix: the ``pip install``
    line naming the extra. Raised by `require_extra`.
    """


# What each extra must make importable, one probe per distribution — the module that *proves* the
# distribution is present, not every module it ships. The composed extras (`[spark]`, `[ray]`,
# `[models-automl]` all include `[gcp]`) list their base's probes too, so the install line in the
# error is sufficient for whichever module turned out to be missing.
_GCP_MODULES: tuple[str, ...] = (
    "google.cloud.bigquery",
    "google.cloud.bigquery_storage",
    "google.cloud.storage",
    "google.cloud.dataproc_v1",
    "google.cloud.aiplatform",
)
EXTRA_MODULES: dict[str, tuple[str, ...]] = {
    "gcp": _GCP_MODULES,
    "notebook": ("matplotlib",),
    "spark": (*_GCP_MODULES, "pyspark"),
    "ray": (*_GCP_MODULES, "ray"),
    "submit": (*_GCP_MODULES, "ray"),
    "models-automl": (*_GCP_MODULES, "google_cloud_pipeline_components"),
}


def is_importable(module: str) -> bool:
    """Is ``module`` installed? Probed without importing it, so this costs microseconds.

    ``find_spec`` on a dotted name imports the *parents*, and raises ``ModuleNotFoundError`` when a
    parent is absent (``google.cloud.bigquery`` with no ``google`` at all) — which is just another
    way of saying "not installed".
    """
    import importlib.util

    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def require_extra(extra: str, *, purpose: str) -> None:
    """Raise `MissingExtraError` naming the install command unless ``extra`` is installed.

    The package has zero top-level ``google.*`` imports and a bare ``pip install scale-forecasting``
    is the pure forecasting layer (models, metrics, backtests, the playground). Everything that
    reaches Google Cloud sits behind the ``gcp`` extra, so a core-only install that calls
    `main.run` would otherwise die on ``ModuleNotFoundError: No module named 'google'`` somewhere
    below the registry. This is called once at each entry point that needs the extra — `main.run`,
    the `Forecaster`, the CLI mains, the plotting helpers — with ``purpose`` saying what the caller
    was trying to do, so the failure is one sentence with the fix in it.
    """
    missing = [m for m in EXTRA_MODULES[extra] if not is_importable(m)]
    if missing:
        raise MissingExtraError(
            f"{purpose} needs the '{extra}' extra, which is not installed "
            f"(missing: {', '.join(missing)}). Install it with:\n"
            f'    pip install "scale-forecasting[{extra}]"\n'
            f"or, from a clone, `uv sync --extra {extra}`."
        )


PACKAGE_LOGGER = "scale_forecasting"
CLI_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"

# The library's whole opinion about where its records go: nowhere, by default. A single
# ``NullHandler`` on the package logger stops the stdlib's "no handlers could be found" last-resort
# path, and every record propagates to whatever the host configured — Airflow's task handler, a
# notebook's root handler, the CLI handler `configure_cli_logging` attaches. Added at import so it
# is there before the first ``get_logger`` call and never duplicated.
logging.getLogger(PACKAGE_LOGGER).addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    """Return the package logger for ``name`` — ``__name__`` at every call site.

    No handler is attached here and propagation is left on, which is what lets an application
    route this package's records with its own configuration. Until 1.0 each logger carried its own
    ``StreamHandler`` with propagation off, and an Airflow task log never saw a line this package
    wrote; the CLIs that relied on that handler now call `configure_cli_logging` instead.
    """
    return logging.getLogger(name)


def configure_cli_logging(default_level: str = "INFO") -> None:
    """Attach a root handler for a command-line entry point, unless the host already has one.

    Every ``python -m scale_forecasting.<module>`` verb reports through ``_log.info`` — the resolved
    run_id, the fanout, "submitted" — and the root logger ships with no handler and a WARNING
    threshold, so as a library that is right and as a CLI it means the documented commands print
    nothing at all (``--dry-run``, whose whole job is to say what a run would do, once exited 0 in
    silence). Guarded on the root's handlers so a process that has already configured logging
    (Airflow, a notebook) does not get a second copy of every line. ``SF_LOG_LEVEL`` overrides the
    level.
    """
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=os.environ.get("SF_LOG_LEVEL", default_level).upper(), format=CLI_LOG_FORMAT
    )
