"""The ``run_jobs`` table — one row per family job, per attempt.

Per-job identity and trace: the INSERT at RUNNING, the in-place UPDATE at terminal, the attempt
numbering a retry needs, and the read the DAG trace, the review layer and the probes all share.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .params import (
    _JOB_PARAM_TYPES,
    _job_param,
    _status_guard_param,
    render_status_guard,
    render_telemetry_merge,
    telemetry_merge_params,
)
from .tables import _resolve_settings

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ..settings import Settings


def write_job(
    row: dict[str, Any], *, settings: Settings | None = None
) -> None:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Insert one ``run_jobs`` row (a job at RUNNING) via a parameterized single-row INSERT.

    Takes an assembled row (`assemble_job_row`), mirroring `write_header`. Raises `RegistryError`
    on failure.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    resolved = _resolve_settings(settings)
    columns = list(row)
    placeholders = ", ".join(f"@{col}" for col in columns)
    sql = (
        f"INSERT INTO `{resolved.registry_table_ref('run_jobs')}` "
        f"({', '.join(columns)}) VALUES ({placeholders})"
    )
    params = [_job_param(col, row[col]) for col in columns]
    client = bigquery.Client(project=resolved.project_id)
    try:
        client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RegistryError(f"write_job failed for job {row.get('job_id')}: {exc}") from exc


def update_job(
    job_id: str,
    *,
    settings: Settings | None = None,
    unless_status_in: Sequence[str] = (),
    merge_telemetry: Mapping[str, Any] | None = None,
    **fields: Any,
) -> None:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Update named columns on a job's ``run_jobs`` row, e.g. status/runtime_seconds/telemetry.

    ``update_job(job_id, status="COMPLETED", runtime_seconds=42.0)`` → a parameterized
    ``UPDATE … WHERE job_id=@job_id``. The ``job_id`` is 1:1 with a row, so exactly one row is
    touched. Unknown column names raise `RegistryError`; a no-op call (no fields, no patch) returns
    without touching BigQuery.

    ``merge_telemetry`` is the accreting alternative to ``job_telemetry=…``: a ``{dotted.path:
    value}`` patch merged in place (`params.render_telemetry_merge`) so the paths it does not name
    survive. Use it whenever the row may already carry telemetry someone else wrote — a family that
    walked regions for capacity holds its whole attempt ledger under ``$.capacity``, and a
    whole-column write erases the record of every region tried. Passing both forms raises, because
    a statement that replaces the column *and* merges into it has no honest meaning.

    ``unless_status_in`` adds a status guard to the WHERE (`render_status_guard`), so a row already
    in one of those states is left exactly as it is — the write is skipped, not merged. That makes
    the update conditional inside the one statement rather than read-then-write, which matters
    because the state being protected is written by a *different process*: see `registry.lifecycle`
    and `probes.settle` for the two callers and why each needs it. A guarded skip is silent, so a
    caller that must know whether its write landed re-reads the row rather than assuming.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    patch = dict(merge_telemetry or {})
    if not fields and not patch:
        return
    unknown = set(fields) - set(_JOB_PARAM_TYPES)
    if unknown:
        raise RegistryError(f"update_job: unknown run_jobs column(s): {sorted(unknown)}")
    if patch and "job_telemetry" in fields:
        raise RegistryError(
            "update_job: pass either job_telemetry= (replace) or merge_telemetry= (merge), not both"
        )

    assignments = [f"{col} = @{col}" for col in fields]
    params = [_job_param(col, value) for col, value in fields.items()]
    if patch:
        # Bound before the settings are resolved, so an illegal path is a caller bug that raises
        # the same way whether or not this process can reach a project.
        params.extend(telemetry_merge_params(patch, caller="update_job"))
        assignments.append(render_telemetry_merge(list(patch)))
    resolved = _resolve_settings(settings)
    table = resolved.registry_table_ref("run_jobs")
    sql = (
        f"UPDATE `{table}` SET {', '.join(assignments)} WHERE job_id=@job_id"
        f"{render_status_guard(unless_status_in)}"
    )
    params.append(_job_param("job_id", job_id))
    if unless_status_in:
        params.append(_status_guard_param(unless_status_in))
    client = bigquery.Client(project=resolved.project_id)
    try:
        client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RegistryError(f"update_job failed for job {job_id}: {exc}") from exc


def latest_job_attempt(
    run_id: str, family: str, *, settings: Settings | None = None
) -> int | None:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Return the highest ``attempt`` recorded for ``(run_id, family)``, or ``None`` if no job.

    The registry read that feeds the re-run policy (`registry.ids.decide_attempt`): a non-``None``
    result means this family has already run under this run, so an unforced re-run reuses that job
    and a forced one takes ``max + 1``. Raises `RegistryError` on failure.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    resolved = _resolve_settings(settings)
    sql = (
        f"SELECT MAX(attempt) AS max_attempt FROM `{resolved.registry_table_ref('run_jobs')}` "
        "WHERE run_id=@run_id AND family=@family"
    )
    params = [_job_param("run_id", run_id), _job_param("family", family)]
    client = bigquery.Client(project=resolved.project_id)
    try:
        rows = list(
            client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RegistryError(f"latest_job_attempt failed for {run_id}/{family}: {exc}") from exc
    return rows[0]["max_attempt"] if rows and rows[0]["max_attempt"] is not None else None


def next_job_attempt(
    run_id: str, family: str, *, force: bool = False, settings: Settings | None = None
) -> tuple[int, bool]:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Resolve the ``(attempt, is_new_job)`` a submission should use for ``(run_id, family)``.

    Reads the current max attempt (`latest_job_attempt`) and applies the pure policy
    (`registry.ids.decide_attempt`): first run → ``(1, True)``; unforced re-run of an existing job →
    ``(max, False)`` (reuse, no new job); ``force`` → ``(max + 1, True)`` (a distinct new attempt).
    """
    from .ids import decide_attempt

    current_max = latest_job_attempt(run_id, family, settings=settings)
    return decide_attempt(current_max, force=force)


def read_run_jobs(
    run_id: str, *, settings: Settings | None = None
) -> list[dict[str, Any]]:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Return the current job per family for ``run_id`` from ``v_run_jobs`` — the forward trace.

    The view already keeps one row per ``(run_id, family)`` (highest attempt wins), so this is the
    run's DAG as executed: which families ran, on what runtime/hardware, with what status. Ordered
    by ``family`` for stable output. Raises `RegistryError` on failure.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    resolved = _resolve_settings(settings)
    sql = (
        f"SELECT * FROM `{resolved.registry_table_ref('v_run_jobs')}` "
        "WHERE run_id=@run_id ORDER BY family"
    )
    params = [_job_param("run_id", run_id)]
    client = bigquery.Client(project=resolved.project_id)
    try:
        rows = list(
            client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RegistryError(f"read_run_jobs failed for run {run_id}: {exc}") from exc
    return [dict(r) for r in rows]
