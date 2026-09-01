"""Registry column types — how a Python value is bound into a parameterized query.

One table per registry table (``run_registry``, ``run_jobs``) mapping column name to its
BigQuery type, plus the binder that turns a name/value pair into a query parameter. The two
binders live together because they are the same idea applied twice; they live apart from
`registry.ddl` because that renders the schema while this consumes it.
"""

from __future__ import annotations

from typing import Any

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
    "job_telemetry": "JSON",
}


def _job_param(name: str, value: Any) -> Any:
    """Build a scalar query parameter for a run_jobs column."""
    from google.cloud import bigquery

    return bigquery.ScalarQueryParameter(name, _JOB_PARAM_TYPES[name], value)
