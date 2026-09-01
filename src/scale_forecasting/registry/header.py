"""The run header — one ``run_registry`` row per run, written once and updated in place.

The whole life of that row: the input-data snapshot every job pins its source read to
(`resolve_snapshot_millis` / `snapshot_millis_for`), the opening INSERT and the later UPDATEs, the
accreting ``job_telemetry`` merge (several jobs of one run each record their own sizing without
overwriting each other), and the status read the pollers go through.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from ..errors import get_logger
from .params import _HEADER_PARAM_TYPES, _header_param
from .rows import assemble_header_row
from .tables import _resolve_settings

if TYPE_CHECKING:
    from ..config import RunConfig
    from ..settings import Settings

_log = get_logger(__name__)


# Pull the pinned snapshot a hair behind the BigQuery clock so the instant every reader
# time-travels to is unambiguously in the committed past (not a moment BigQuery is still
# stamping), avoiding any read-your-writes edge on a source table touched right before a run.
_SNAPSHOT_SAFETY_MARGIN_MS = 2000


def resolve_snapshot_millis(
    *, settings: Settings | None = None
) -> int | None:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Resolve the input-data snapshot for a run — one instant every job pins its source read to.

    Queries the **BigQuery clock** (``SELECT UNIX_MILLIS(CURRENT_TIMESTAMP())``) so the snapshot is
    a single authoritative epoch-millis value independent of any driver's local clock, then steps it
    back a small `_SNAPSHOT_SAFETY_MARGIN_MS` margin. Called once per run when the header is written
    (`write_header`); the value is stored on ``run_registry.snapshot_millis`` and read back by every
    family job via `snapshot_millis_for`, so a Spark batch, a Ray job, and the BigQuery-native
    models all read the *identical* source state — the "every job in a run sees the same input"
    guarantee, uniform across native and managed-Iceberg tables (both are read through BigQuery).

    Best-effort: on any failure it logs and returns ``None`` (the reads fall back to unpinned rather
    than failing the run). Kept out of the config so it never perturbs the config-derived run_id.
    """
    from google.cloud import bigquery

    resolved = _resolve_settings(settings)
    client = bigquery.Client(project=resolved.project_id)
    try:
        rows = list(client.query("SELECT UNIX_MILLIS(CURRENT_TIMESTAMP()) AS ms").result())
        return int(rows[0]["ms"]) - _SNAPSHOT_SAFETY_MARGIN_MS
    except Exception as exc:  # noqa: BLE001 - best-effort; unpinned read is the safe fallback
        _log.warning("resolve_snapshot_millis failed; run will read unpinned: %s", exc)
        return None


def snapshot_millis_for(
    run_id: str, *, settings: Settings | None = None
) -> int | None:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Return the pinned input-data snapshot for ``run_id`` from its header, or ``None``.

    The reader-side counterpart to `resolve_snapshot_millis`: each family job derives its ``run_id``
    from its own config (`registry.ids.make_run_id`, a pure function) and calls this to fetch the
    one snapshot the run recorded, so it time-travels its source read to the exact instant every
    other job in the run does — without the value ever being threaded through submitters or args.
    Reads the latest header row for the id (a ``--force`` re-run appends a fresh header with its own
    snapshot; latest ``created_at`` wins, matching the rest of the read-side dedupe).

    Best-effort: returns ``None`` (→ unpinned read) if no header, a NULL snapshot, or a query
    error — so a missing snapshot degrades gracefully to the pre-snapshot behavior rather than
    crashing a read. The owner path always writes a snapshot, so ``None`` means an old run.
    """
    from google.cloud import bigquery

    resolved = _resolve_settings(settings)
    sql = (
        f"SELECT snapshot_millis FROM `{resolved.registry_table_ref('run_registry')}` "
        "WHERE run_id=@run_id ORDER BY created_at DESC LIMIT 1"
    )
    params = [_header_param("run_id", run_id)]
    client = bigquery.Client(project=resolved.project_id)
    try:
        rows = list(
            client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
        )
    except Exception as exc:  # noqa: BLE001 - best-effort; unpinned read is the safe fallback
        _log.warning("snapshot_millis_for(%s) failed; reading unpinned: %s", run_id, exc)
        return None
    return rows[0]["snapshot_millis"] if rows and rows[0]["snapshot_millis"] is not None else None


def write_header(
    cfg: RunConfig, run_id: str, *, settings: Settings | None = None
) -> None:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Insert the run's ``run_registry`` header row (status RUNNING) from its config.

    A single-row parameterized INSERT (not the Write API — no benefit for one row, and the
    header is updated in place later by `update_header`). Resolves the run's input-data snapshot
    once here (`resolve_snapshot_millis`) and stamps it on the header so every family job pins the
    same source state (`snapshot_millis_for`). Also resolves the launching principal
    (`identity.resolve_principal`, best-effort) into ``user_id`` so *launch* is attributable in the
    audit trail — alongside the cancel actor recorded by the P5 cancel path. Raises `RegistryError`
    on failure.
    """
    from datetime import UTC, datetime

    from google.cloud import bigquery

    from ..errors import RegistryError
    from ..identity import resolve_principal

    resolved = _resolve_settings(settings)
    snapshot_millis = resolve_snapshot_millis(settings=resolved)
    row = assemble_header_row(
        cfg,
        run_id,
        datetime.now(UTC),
        snapshot_millis=snapshot_millis,
        user_id=resolve_principal(resolved),
    )
    columns = list(row)
    placeholders = ", ".join(f"@{col}" for col in columns)
    sql = (
        f"INSERT INTO `{resolved.registry_table_ref('run_registry')}` "
        f"({', '.join(columns)}) VALUES ({placeholders})"
    )
    params = [_header_param(col, row[col]) for col in columns]
    client = bigquery.Client(project=resolved.project_id)
    try:
        client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RegistryError(f"write_header failed for run {run_id}: {exc}") from exc


def update_header(
    run_id: str, *, settings: Settings | None = None, **fields: Any
) -> None:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Update named columns on a run's header row, e.g. status/runtime_seconds.

    ``update_header(run_id, status="COMPLETED", runtime_seconds=42.0)`` → a parameterized
    ``UPDATE … SET … WHERE run_id=@run_id``. Unknown column names raise `RegistryError`;
    a no-op call (no fields) returns without touching BigQuery.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    if not fields:
        return
    unknown = set(fields) - set(_HEADER_PARAM_TYPES)
    if unknown:
        raise RegistryError(f"update_header: unknown run_registry column(s): {sorted(unknown)}")

    resolved = _resolve_settings(settings)
    set_clause = ", ".join(f"{col} = @{col}" for col in fields)
    table = resolved.registry_table_ref("run_registry")
    sql = f"UPDATE `{table}` SET {set_clause} WHERE run_id=@run_id"
    params = [_header_param(col, value) for col, value in fields.items()]
    params.append(_header_param("run_id", run_id))
    client = bigquery.Client(project=resolved.project_id)
    try:
        client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RegistryError(f"update_header failed for run {run_id}: {exc}") from exc


# A `job_telemetry` merge path: dot-separated lower-snake segments, rendered as ``$.a.b``. The
# charset is enforced rather than escaped because every caller is our own code writing a known
# key — a path that needs quoting is a bug in the caller, not an input to accommodate.
_TELEMETRY_PATH_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+)*$")


def render_header_telemetry_merge(table_ref: str, paths: Sequence[str]) -> str:
    """The ``UPDATE … JSON_SET(…)`` that merges ``paths`` into a header's ``job_telemetry`` (pure).

    Separated from the client call so the SQL is readable and testable offline. ``JSON_SET`` writes
    each path independently and leaves the rest of the document alone, which is the whole point:
    the run header's telemetry is written by *several* jobs of one run (a Serverless batch, a
    cluster job, a Ray job), and a whole-column write means whichever finishes last is the only one
    that leaves a trace. ``IFNULL(…, JSON '{}')`` covers the first writer, whose column is still
    NULL; nested paths create their parent objects.

    Parameters are named ``@t0…@tN`` positionally against ``paths``; the caller binds them in the
    same order.
    """
    sets = ", ".join(f"'$.{path}', @t{i}" for i, path in enumerate(paths))
    return (
        f"UPDATE `{table_ref}` "
        f"SET job_telemetry = JSON_SET(IFNULL(job_telemetry, JSON '{{}}'), {sets}) "
        "WHERE run_id=@run_id"
    )


def sizing_telemetry_path(sizing: Mapping[str, Any]) -> str:
    """Where one sizing record (`resources.audit.sizing_telemetry`) is filed on the header (pure).

    ``sizing.<family>`` — because a run's families are sized separately, on separate runtimes and
    separate hardware, and the question "why is the deep-learning job this shape" is not answerable
    from a field that holds whichever family stamped last. The family label is slugged (it may be a
    ``+``-joined union when several families share one cluster) so it is a legal path segment;
    a record with no plan to take a family from files under ``sizing.run``.
    """
    family = str(sizing.get("family") or "").lower()
    slug = re.sub(r"[^a-z0-9_]+", "_", family).strip("_")
    return f"sizing.{slug or 'run'}"


def merge_header_telemetry(
    run_id: str, patch: Mapping[str, Any], *, settings: Settings | None = None
) -> None:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Merge ``{path: value}`` into a run header's ``job_telemetry``, leaving the rest untouched.

    The accreting counterpart to ``update_header(job_telemetry=…)``, which replaces the column
    whole. Keys are dotted paths (``"total_wall_s"``, ``"sizing.deep_learning"``); values are any
    JSON-able object and are bound as ``JSON`` parameters, so a dict lands as an object rather than
    as a string. A no-op call (empty patch) returns without touching BigQuery; an illegal path
    raises `RegistryError` rather than being escaped into SQL.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    if not patch:
        return
    bad = [path for path in patch if not _TELEMETRY_PATH_RE.match(path)]
    if bad:
        raise RegistryError(f"merge_header_telemetry: illegal telemetry path(s): {sorted(bad)}")

    resolved = _resolve_settings(settings)
    paths = list(patch)
    sql = render_header_telemetry_merge(resolved.registry_table_ref("run_registry"), paths)
    params: list[Any] = [
        bigquery.ScalarQueryParameter(f"t{i}", "JSON", patch[path]) for i, path in enumerate(paths)
    ]
    params.append(_header_param("run_id", run_id))
    client = bigquery.Client(project=resolved.project_id)
    try:
        client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RegistryError(f"merge_header_telemetry failed for run {run_id}: {exc}") from exc


def header_status(
    run_id: str, *, settings: Settings | None = None
) -> str | None:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Return the status of a run's ``run_registry`` header, or ``None`` if it has never run.

    Reads the most recent header row for ``run_id`` (a forced re-run of the same config appends
    another row under the same id; the latest ``created_at`` wins, matching the read-side dedupe in
    ``v_run_summary``). Because ``run_id`` is a pure digest of the config, this is the pre-submit
    existence check: a non-``None`` status means this exact config has already run. Raises
    `RegistryError` on failure (including when the registry table does not exist yet).
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    resolved = _resolve_settings(settings)
    sql = (
        f"SELECT status FROM `{resolved.registry_table_ref('run_registry')}` "
        "WHERE run_id=@run_id ORDER BY created_at DESC LIMIT 1"
    )
    params = [_header_param("run_id", run_id)]
    client = bigquery.Client(project=resolved.project_id)
    try:
        rows = list(
            client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RegistryError(f"header_status failed for run {run_id}: {exc}") from exc
    return rows[0]["status"] if rows else None
