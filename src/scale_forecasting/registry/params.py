"""Registry column types — how a Python value is bound into a parameterized query.

One table per registry table (``run_registry``, ``run_jobs``) mapping column name to its
BigQuery type, plus the binder that turns a name/value pair into a query parameter. The two
binders live together because they are the same idea applied twice; they live apart from
`registry.ddl` because that renders the schema while this consumes it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

# run_registry columns that may be set by write_header / update_header, with their BQ types.
_HEADER_PARAM_TYPES: dict[str, str] = {
    "run_id": "STRING",
    "created_at": "TIMESTAMP",
    "snapshot_millis": "INT64",
    "user_id": "STRING",
    "git_sha": "STRING",
    "python_runtime": "STRING",
    "bq_models": "ARRAY<STRING>",
    "backtest_on": "BOOL",
    "decision_metric": "STRING",
    "ensemble_strategies": "ARRAY<STRING>",
    "raw_config": "JSON",
    "status": "STRING",
    "n_series": "INT64",
    "n_models": "INT64",
    "runtime_seconds": "FLOAT64",
    "job_telemetry": "JSON",
}


def _header_param(name: str, value: Any) -> Any:
    """Build a scalar or array query parameter for a run_registry column."""
    from google.cloud import bigquery

    bq_type = _HEADER_PARAM_TYPES[name]
    if bq_type.startswith("ARRAY<"):
        element_type = bq_type[len("ARRAY<") : -1]
        return bigquery.ArrayQueryParameter(name, element_type, list(value or []))
    return bigquery.ScalarQueryParameter(name, bq_type, value)


# run_jobs columns that may be set by write_job / update_job, with their BQ types.
_JOB_PARAM_TYPES: dict[str, str] = {
    "job_id": "STRING",
    "run_id": "STRING",
    "family": "STRING",
    "attempt": "INT64",
    "runtime": "STRING",
    "spark_mode": "STRING",
    "hardware": "STRING",
    "gpu_type": "STRING",
    "system_job_id": "STRING",
    "status": "STRING",
    "created_at": "TIMESTAMP",
    "started_at": "TIMESTAMP",
    "ended_at": "TIMESTAMP",
    "runtime_seconds": "FLOAT64",
    # Why a FAILED row failed, as a short machine-readable token (`capacity.CAPACITY_EXHAUSTED` is
    # the first). A column rather than another JSON path because this is the field an operator
    # filters a whole registry on — "show me every job that ran out of regions" has to be a WHERE
    # clause, not something you need to know a JSON path to find. NULL for every other failure and
    # for every row written before it existed.
    "failure_reason": "STRING",
    "job_telemetry": "JSON",
}


def _job_param(name: str, value: Any) -> Any:
    """Build a scalar query parameter for a run_jobs column."""
    from google.cloud import bigquery

    return bigquery.ScalarQueryParameter(name, _JOB_PARAM_TYPES[name], value)


# The parameter name both status-guarded UPDATEs bind their protected-status list to.
_STATUS_GUARD_PARAM = "unless_status_in"


def render_status_guard(unless_status_in: Sequence[str]) -> str:
    """The ``AND status …`` tail that makes an UPDATE skip rows already in a protected state (pure).

    Empty sequence → empty string, so an unguarded call renders exactly the SQL it always did. The
    ``status IS NULL`` arm is deliberate: SQL three-valued logic makes ``NULL NOT IN (…)`` unknown,
    which would silently drop the row from the update, and a row with no status is precisely one
    that has nothing worth protecting.
    """
    if not unless_status_in:
        return ""
    return f" AND (status IS NULL OR status NOT IN UNNEST(@{_STATUS_GUARD_PARAM}))"


def _status_guard_param(unless_status_in: Sequence[str]) -> Any:
    """Bind the protected-status list for `render_status_guard`'s tail."""
    from google.cloud import bigquery

    return bigquery.ArrayQueryParameter(_STATUS_GUARD_PARAM, "STRING", list(unless_status_in))
