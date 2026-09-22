"""The run header — one ``run_registry`` row per *attempt*, written once and updated in place.

The whole life of that row: the input-data snapshot every job pins its source read to
(`resolve_snapshot_millis` / `snapshot_millis_for`), the opening INSERT and the later UPDATEs, the
accreting ``job_telemetry`` merge (several jobs of one run each record their own sizing without
overwriting each other), and the status read the pollers go through.

**One row per attempt, not one per run, and every statement here has to mean it.** ``run_id`` is a
pure digest of the config, so a ``--force`` re-run appends a *second* header under the same id with
its own ``created_at``. The read side has always known that and takes the newest. The write side did
not, until `render_latest_header_guard` — see its docstring for the live case where a re-run erased
the attempt before it.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from ..errors import get_logger
from .params import (
    _HEADER_PARAM_TYPES,
    _header_param,
    _status_guard_param,
    render_status_guard,
    render_telemetry_merge,
    telemetry_merge_params,
)
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
    cfg: RunConfig, run_id: str, *, settings: Settings | None = None, status: str = "RUNNING"
) -> None:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Insert the run's ``run_registry`` header row from its config.

    A single-row parameterized INSERT (not the Write API — no benefit for one row, and the
    header is updated in place later by `update_header`). Resolves the run's input-data snapshot
    once here (`resolve_snapshot_millis`) and stamps it on the header so every family job pins the
    same source state (`snapshot_millis_for`). Also resolves the launching principal
    (`identity.resolve_principal`, best-effort) into ``user_id`` so *launch* is attributable in the
    audit trail — alongside the cancel actor recorded by the P5 cancel path. Raises `RegistryError`
    on failure.

    ``status`` opens the run and defaults to ``RUNNING``, which is right for every caller that is
    about to do work. `launch_plan.stage_run` passes ``STAGED`` instead: it opens a header so the
    repair verbs can see the job rows it filed, for a run it is not going to launch.
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
        status=status,
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


def render_latest_header_guard(table_ref: str) -> str:
    """The ``AND created_at = (SELECT MAX(…))`` tail that narrows an UPDATE to one attempt (pure).

    ``run_id`` is a pure digest of the config, so it is **not** unique in ``run_registry``: a
    ``--force`` re-run of the same config appends a second header with its own ``created_at`` and
    its own ``snapshot_millis``. Every reader already knows this and takes the newest —
    `header_status`, `snapshot_millis_for`, `reads.read_run_config` and ``v_run_summary`` all order
    by ``created_at`` descending and keep one row. This is the write-side mirror of that rule, and
    it exists because the writers did **not** have it.

    Found 2026-09-16 on a re-run of ``all_families_10k``: the finalize matched on ``run_id`` alone,
    so the second attempt's 7,054.6 s and its terminal status landed on *both* headers and the
    first attempt's 13,134.2 s was gone. It survived only because the validation ledger had written
    it down by hand. The per-cell tables were never affected — they are append-only and separable by
    ``created_at`` — so the damage was confined to exactly the columns a finalize touches, which is
    also the set an operator reads first.

    The job layer never had this problem, and the contrast is the clearest statement of the fix.
    A ``job_id`` is ``sf-<run_id>-<family>-a<attempt>``; `ids.decide_attempt` reads the family's
    current max out of the registry and returns ``current_max + 1`` under ``--force``, so
    `jobs.update_job` can match one id and honestly claim one row. The header has no attempt
    counter and adding one would be a schema change, a DDL migration, and a new identity concept.
    Selecting the newest ``created_at`` gets the same guarantee out of a column that is already
    there, and — the part that matters more — states it in exactly the words the read side already
    uses, so the two sides cannot drift into disagreeing about which attempt is *the* attempt.

    A tie needs two headers stamped in the same microsecond by `write_header`'s
    ``datetime.now(UTC)``, which is two launches of one config inside a microsecond; the
    pre-submit existence check (`header_status`) stops that long before it reaches here.
    """
    return f" AND created_at = (SELECT MAX(created_at) FROM `{table_ref}` WHERE run_id=@run_id)"


def render_header_update(
    table_ref: str, columns: Sequence[str], unless_status_in: Sequence[str] = ()
) -> str:
    """The whole ``UPDATE … SET … WHERE`` a `update_header` call issues (pure).

    Separated from the client call for the same reason `render_header_telemetry_merge` is: a
    statement that only exists inside a ``# pragma: no cover`` GCP function can only be checked by
    reading it. The overwrite this module's `render_latest_header_guard` now prevents lived in
    exactly that blind spot — the WHERE clause was assembled inline, no test could see it, and the
    bug was found on a live run instead.

    ``columns`` renders ``col = @col`` in the order given; the caller binds a parameter per name
    plus ``@run_id``. The two tails are the latest-attempt guard (always) and the status guard
    (only when ``unless_status_in`` is non-empty).
    """
    set_clause = ", ".join(f"{col} = @{col}" for col in columns)
    return (
        f"UPDATE `{table_ref}` SET {set_clause} WHERE run_id=@run_id"
        f"{render_latest_header_guard(table_ref)}{render_status_guard(unless_status_in)}"
    )


def update_header(
    run_id: str,
    *,
    settings: Settings | None = None,
    unless_status_in: Sequence[str] = (),
    **fields: Any,
) -> None:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Update named columns on a run's **latest** header row, e.g. status/runtime_seconds.

    ``update_header(run_id, status="COMPLETED", runtime_seconds=42.0)`` → a parameterized
    ``UPDATE … SET … WHERE run_id=@run_id AND created_at = (SELECT MAX(…))``. Unknown column names
    raise `RegistryError`; a no-op call (no fields) returns without touching BigQuery.

    The ``created_at`` tail is `render_latest_header_guard`, and it is what keeps a forced re-run
    from overwriting the attempt before it — read that docstring for the case that motivated it.

    ``unless_status_in`` adds a status guard to the WHERE (`render_status_guard`), leaving a header
    already in one of those states untouched. Same reason as `registry.jobs.update_job`: the state
    being protected was written by another process, so the condition belongs in the statement.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    if not fields:
        return
    unknown = set(fields) - set(_HEADER_PARAM_TYPES)
    if unknown:
        raise RegistryError(f"update_header: unknown run_registry column(s): {sorted(unknown)}")

    resolved = _resolve_settings(settings)
    table = resolved.registry_table_ref("run_registry")
    sql = render_header_update(table, list(fields), unless_status_in)
    params = [_header_param(col, value) for col, value in fields.items()]
    params.append(_header_param("run_id", run_id))
    if unless_status_in:
        params.append(_status_guard_param(unless_status_in))
    client = bigquery.Client(project=resolved.project_id)
    try:
        client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RegistryError(f"update_header failed for run {run_id}: {exc}") from exc


def render_header_telemetry_merge(table_ref: str, paths: Sequence[str]) -> str:
    """The ``UPDATE … JSON_SET(…)`` that merges ``paths`` into a header's ``job_telemetry`` (pure).

    Separated from the client call so the SQL is readable and testable offline. The SET assignment
    itself is `params.render_telemetry_merge`, shared with the job-row writer; this wraps it in the
    header's statement. Parameters are named ``@t0…@tN`` positionally against ``paths``; the caller
    binds them in the same order.

    Carries the same `render_latest_header_guard` tail as `update_header`, for the same reason and
    with one extra edge of its own: ``JSON_SET`` merges rather than replaces, so without the guard a
    re-run would not overwrite the first attempt's telemetry but *blend into* it — one document
    holding two runs' sizing under the same paths, with nothing in it to say so.
    """
    return (
        f"UPDATE `{table_ref}` SET {render_telemetry_merge(paths)} WHERE run_id=@run_id"
        f"{render_latest_header_guard(table_ref)}"
    )


def sizing_telemetry_path(sizing: Mapping[str, Any]) -> str:
    """Where one sizing record (`resources.audit.sizing_telemetry`) is filed on the header (pure).

    ``sizing.<family>`` — because a run's families are sized separately, on separate runtimes and
    separate hardware, and the question "why is the deep-learning job this shape" is not answerable
    from a field that holds whichever family stamped last. The family label is slugged (it may be a
    ``+``-joined union when several families share one cluster) so it is a legal path segment;
    a record with no plan to take a family from files under ``sizing.run``.
    """
    return f"sizing.{_family_slug(sizing.get('family'))}"


def executed_sizing_path(family: str | None) -> str:
    """Where the *executed* shape is filed on the header — ``sizing_executed.<family>`` (pure).

    A sibling of `sizing_telemetry_path`, and deliberately a different key rather than a merge into
    the same one. ``sizing.<family>`` is what was decided before anything ran, from a config and a
    past run's measurements; this is what the shape turned out to be once there was a live session
    or a live cluster to ask. The two disagreeing is the finding — a fan-out widened to match a
    ceiling nobody passed, a Ray pool re-planned against a device that measured differently — and
    filing the second on top of the first would erase exactly that.

    Same slugging as its sibling, so a union family (``statistical+ml`` sharing one cluster) files
    under the same segment on both sides and the pair can be read together.
    """
    return f"sizing_executed.{_family_slug(family)}"


def _family_slug(family: object) -> str:
    """A family label as a legal telemetry path segment; empty or missing → ``run`` (pure)."""
    slug = re.sub(r"[^a-z0-9_]+", "_", str(family or "").lower()).strip("_")
    return slug or "run"


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
    params = telemetry_merge_params(patch, caller="merge_header_telemetry")
    resolved = _resolve_settings(settings)
    sql = render_header_telemetry_merge(resolved.registry_table_ref("run_registry"), list(patch))
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
