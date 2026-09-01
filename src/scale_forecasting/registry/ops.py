"""Registry lifecycle operations — the manage-only operator surface.

A deployment's registry is not write-once. Runs accumulate, experiments end, a bad run wants
deleting, and the operator wants to know what is actually in there before touching any of it. This
module is that surface: **six bounded verbs** over exactly one registry — the one
`Settings.registry_dataset_ref` resolves to.

**Manage only, by design.** There is no wipe here. A full teardown is `bq rm -r -f <dataset>` or the
BigQuery console — nobody needs us to write that, and shipping it invites the accident. Every verb
below touches only what the caller named:

| verb | what it does |
|---|---|
| `init` | create this registry's tables + views (idempotent; does not touch the source panel) |
| `doctor` | read-only health report — row counts, runs stuck ``RUNNING``, orphaned artifacts |
| `drop_run` | delete named run(s) from every tier: rows, GCS artifacts, and BQML model objects |
| `sweep_orphans` | delete artifact prefixes under *this* registry with no ``run_registry`` row |
| `snapshot` | BigQuery table snapshots of the five registry tables (cheap, expirable) |
| `export` | dump the registry to GCS (Parquet or newline-delimited JSON) for offline analysis |

**The ordering rule.** A registry row is the *only* index of which GCS objects exist. Delete the
rows first and the artifacts become unidentifiable garbage forever — which is exactly how the old
``reset`` path accumulated an unbounded orphan pile. So every destructive verb goes:

    1. enumerate artifact prefixes   <- FIRST, while the index still exists
    2. delete the GCS objects
    3. delete the rows / model objects

`sweep_orphans` is the cleanup for every prefix that was stranded before that rule existed. It is
correctly scopeable only because the artifact root now carries the registry key
(`Settings.artifact_root`): a prefix under *this* root with no ``run_registry`` row is garbage by
definition, and no other registry sharing the bucket can be caught by it.

**Safety model** (the shape `reset` and the probe's cancel path already established): preview is the
default, ``yes=True`` executes, and the preview prints the exact objects — run ids, row counts,
object counts, byte totals — that will be touched. A mutating verb also refuses to run against a
registry with an in-flight run unless forced; use `monitor(probe=True)` to tell a live run from a
dead row still marked ``RUNNING``.

**This module is the policy, not the plumbing.** Step 1 of the ordering rule — reading the GCS
layout back to find out which objects a run owns — is `artifacts`, which owns that layout in both
directions (it is the module that composed the paths on the way in). What lives here is the
decision of *when* to enumerate, *what* the answer blocks, and *in which order* the deletes go.

Split along the pure/I-O seam, like the rest of the registry package: the planners and SQL renderers
below are pure strings and dataclasses (tested offline, no client), and the six verbs are thin I/O
wrappers that execute what a planner produced.

Public surface: the verbs ``init``, ``doctor``, ``drop_run``, ``sweep_orphans``, ``snapshot``,
``export``; the plan/report types ``DropPlan``, ``SweepPlan``, ``DoctorReport``, ``TableStat``;
and the pure helpers ``blocking_runs``, ``render_delete_rows``, ``render_snapshot_sql``,
``render_export_sql``, ``format_plan``, ``format_doctor``. The artifact-prefix helpers the
destructive verbs run on (``ArtifactPrefix``, ``split_gcs_uri``, ``run_id_from_blob``,
``orphan_run_ids``, ``list_prefixes``, ``delete_prefixes``) are `artifacts`'.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..errors import get_logger
from . import artifacts
from .artifacts import ArtifactPrefix
from .ddl import REGISTRY_TABLE_NAMES

if TYPE_CHECKING:
    from ..settings import Settings

_log = get_logger(__name__)

# A run whose header sits in one of these is in flight; a mutating verb refuses to touch it unless
# forced. Anything else (COMPLETED / FAILED / PARTIAL / CANCELLED) is terminal and safe.
LIVE_STATUSES: frozenset[str] = frozenset({"RUNNING", "PENDING"})

# Export formats we render. PARQUET is the default (typed, compact, reads straight into pandas);
# newline-delimited JSON is the escape hatch for a consumer with no Parquet reader — and the
# honest choice for the registry's JSON columns (`raw_config`, `job_telemetry`, `quantiles`,
# `best_params`), which a Parquet export flattens to strings.
EXPORT_FORMATS: tuple[str, ...] = ("PARQUET", "JSON")


# --- pure: types ----------------------------------------------------------------


@dataclass(frozen=True)
class TableStat:
    """One registry table's row count, or ``None`` rows if the table is absent."""

    table: str
    rows: int | None

    @property
    def exists(self) -> bool:
        return self.rows is not None


@dataclass(frozen=True)
class DropPlan:
    """Exactly what `drop_run` would delete — the preview, and the thing `drop_run` executes.

    ``blocked`` is the subset of ``run_ids`` whose header is still in a `LIVE_STATUSES` state. A
    non-empty ``blocked`` stops the whole plan (not just those runs): deleting half a batch because
    the other half was busy is a worse outcome than doing nothing and saying why.
    """

    registry: str
    run_ids: tuple[str, ...] = ()
    prefixes: tuple[ArtifactPrefix, ...] = ()
    models: tuple[str, ...] = ()
    blocked: tuple[str, ...] = ()
    unknown: tuple[str, ...] = ()

    @property
    def object_count(self) -> int:
        return sum(p.object_count for p in self.prefixes)

    @property
    def byte_total(self) -> int:
        return sum(p.byte_total for p in self.prefixes)

    @property
    def is_empty(self) -> bool:
        """Nothing to do — every named run is unknown to this registry."""
        return not self.run_ids


@dataclass(frozen=True)
class SweepPlan:
    """Artifact prefixes under this registry's root with no ``run_registry`` row."""

    registry: str
    root: str
    prefixes: tuple[ArtifactPrefix, ...] = ()
    known_runs: int = 0

    @property
    def object_count(self) -> int:
        return sum(p.object_count for p in self.prefixes)

    @property
    def byte_total(self) -> int:
        return sum(p.byte_total for p in self.prefixes)


@dataclass(frozen=True)
class DoctorReport:
    """A read-only picture of one registry: what is in it, and what looks wrong."""

    registry: str
    artifact_root: str
    tables: tuple[TableStat, ...] = ()
    views: tuple[str, ...] = ()
    live_runs: tuple[tuple[str, str], ...] = ()  # (run_id, status)
    orphans: tuple[ArtifactPrefix, ...] = ()
    notes: tuple[str, ...] = field(default=())

    @property
    def missing_tables(self) -> tuple[str, ...]:
        return tuple(t.table for t in self.tables if not t.exists)

    @property
    def healthy(self) -> bool:
        """No missing tables, no in-flight runs, no orphaned artifacts."""
        return not self.missing_tables and not self.live_runs and not self.orphans


# --- pure: helpers ---------------------------------------------------------------


def blocking_runs(status_by_run: dict[str, str | None]) -> tuple[str, ...]:
    """The subset of runs whose header is in a `LIVE_STATUSES` state, sorted.

    A ``None`` status (no header row) is *not* blocking: an unknown run cannot be in flight, and
    `DropPlan.unknown` reports it separately.
    """
    return tuple(
        sorted(r for r, s in status_by_run.items() if s is not None and s.upper() in LIVE_STATUSES)
    )


def human_bytes(n: int) -> str:
    """``1536`` → ``1.5 KiB``. Preview text only — never parsed."""
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"  # pragma: no cover - unreachable, the loop returns at TiB


# --- pure: SQL renderers ----------------------------------------------------------


def render_delete_rows(
    dataset: str, *, tables: Sequence[str] = REGISTRY_TABLE_NAMES
) -> dict[str, str]:
    """``{table: DELETE ... WHERE run_id IN UNNEST(@run_ids)}`` for a per-run delete.

    Every registry table carries ``run_id``, so one parameterized statement per table covers the
    whole tier. The ids are bound as an **array query parameter**, never interpolated.

    A caveat worth surfacing rather than hiding: BigQuery rejects a ``DELETE`` that matches rows
    still in the Storage Write API streaming buffer (~90 minutes after the append). Dropping a run
    that just finished can therefore fail; `drop_run` re-raises that with a message saying to wait.
    """
    return {
        name: f"DELETE FROM `{dataset}.{name}` WHERE run_id IN UNNEST(@run_ids);" for name in tables
    }


def render_snapshot_sql(
    source_dataset: str,
    target_dataset: str,
    suffix: str,
    *,
    tables: Sequence[str] = REGISTRY_TABLE_NAMES,
    expiration_days: int | None = None,
) -> dict[str, str]:
    """``{table: CREATE SNAPSHOT TABLE ...}`` — a point-in-time copy of the registry.

    A BigQuery **table snapshot**, not a ``CREATE TABLE AS SELECT``: a snapshot is metadata plus the
    delta from its base, so it costs almost nothing at creation and only accrues storage as the base
    diverges. It is read-only and can carry an expiration, which is what makes "snapshot before the
    risky thing" a habit rather than a budget decision.

    ``suffix`` is appended to each table name (``run_registry`` → ``run_registry_20260831``), so a
    snapshot can live in the registry's own dataset without colliding; pass a different
    ``target_dataset`` to keep the registry uncluttered.
    """
    options = ""
    if expiration_days is not None:
        if expiration_days <= 0:
            raise ValueError(f"expiration_days must be positive, got {expiration_days}")
        options = (
            "\nOPTIONS (expiration_timestamp = "
            f"TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL {expiration_days} DAY))"
        )
    return {
        name: (
            f"CREATE SNAPSHOT TABLE IF NOT EXISTS `{target_dataset}.{name}_{suffix}`\n"
            f"CLONE `{source_dataset}.{name}`{options};"
        )
        for name in tables
    }


def render_export_sql(
    dataset: str,
    destination_root: str,
    *,
    fmt: str = "PARQUET",
    tables: Sequence[str] = REGISTRY_TABLE_NAMES,
) -> dict[str, str]:
    """``{table: EXPORT DATA ...}`` — dump the registry to GCS for offline analysis.

    A different job from `render_snapshot_sql`, and both earn their place: a snapshot stays inside
    BigQuery (fast, queryable, cheap) while an export leaves it (readable by anyone with the bucket
    and no BigQuery access at all). Each table lands under
    ``<destination_root>/<table>/`` with a sharded wildcard, which is how BigQuery writes an export.

    ``fmt`` is one of `EXPORT_FORMATS`. Prefer ``JSON`` when the registry's ``JSON`` columns matter
    downstream — a Parquet export flattens them to strings.
    """
    fmt_up = fmt.upper()
    if fmt_up not in EXPORT_FORMATS:
        raise ValueError(f"unsupported export format {fmt!r}; expected one of {EXPORT_FORMATS}")
    bq_format = "NEWLINE_DELIMITED_JSON" if fmt_up == "JSON" else "PARQUET"
    ext = "json" if fmt_up == "JSON" else "parquet"
    root = destination_root.rstrip("/")
    return {
        name: (
            "EXPORT DATA OPTIONS (\n"
            f"  uri = '{root}/{name}/{name}-*.{ext}',\n"
            f"  format = '{bq_format}',\n"
            "  overwrite = true\n"
            ") AS\n"
            f"SELECT * FROM `{dataset}.{name}`;"
        )
        for name in tables
    }


# --- pure: preview formatting -----------------------------------------------------


def format_plan(plan: DropPlan | SweepPlan) -> str:
    """Render a plan as the operator-facing preview — the exact text a dry run prints."""
    lines: list[str] = []
    if isinstance(plan, DropPlan):
        lines.append(f"drop-run against registry {plan.registry}")
        if plan.unknown:
            lines.append(f"  not in this registry (skipped): {', '.join(plan.unknown)}")
        if plan.blocked:
            lines.append(f"  IN FLIGHT — refusing: {', '.join(plan.blocked)}")
        if plan.run_ids:
            lines.append(f"  runs ({len(plan.run_ids)}): {', '.join(plan.run_ids)}")
            lines.append(f"  rows: DELETE from {len(REGISTRY_TABLE_NAMES)} tables")
        if plan.models:
            lines.append(f"  BQML model objects ({len(plan.models)}): {', '.join(plan.models)}")
    else:
        lines.append(f"sweep-orphans against registry {plan.registry}")
        lines.append(f"  artifact root: {plan.root}")
        lines.append(f"  runs known to the registry: {plan.known_runs}")
        lines.append(
            f"  orphan prefixes ({len(plan.prefixes)}): "
            f"{', '.join(p.run_id for p in plan.prefixes) or '(none)'}"
        )
    lines.append(f"  GCS: {plan.object_count} objects, {human_bytes(plan.byte_total)}")
    return "\n".join(lines)


def format_doctor(report: DoctorReport) -> str:
    """Render a `DoctorReport` as readable text."""
    lines = [f"registry {report.registry}", f"  artifacts: {report.artifact_root}"]
    for stat in report.tables:
        rows = "MISSING" if stat.rows is None else f"{stat.rows:,} rows"
        lines.append(f"  {stat.table:<22} {rows}")
    lines.append(f"  views: {', '.join(report.views) or '(none)'}")
    if report.live_runs:
        pairs = ", ".join(f"{r} ({s})" for r, s in report.live_runs)
        lines.append(f"  IN FLIGHT ({len(report.live_runs)}): {pairs}")
    if report.orphans:
        objects = sum(p.object_count for p in report.orphans)
        total = human_bytes(sum(p.byte_total for p in report.orphans))
        lines.append(
            f"  ORPHANED artifacts: {len(report.orphans)} prefixes, {objects} objects, {total}"
            " — run sweep_orphans"
        )
    lines.extend(f"  note: {n}" for n in report.notes)
    if report.healthy:
        lines.append("  healthy")
    return "\n".join(lines)


# --- I/O: shared plumbing ----------------------------------------------------------


def _resolved(settings: Settings | None) -> Settings:
    """The passed settings, or a fresh resolve from the ``SF_*`` environment."""
    from .tables import _resolve_settings

    return _resolve_settings(settings)


def _known_run_ids(settings: Settings) -> set[str]:  # pragma: no cover - GCP I/O, @gcp smoke
    """Every ``run_id`` in this registry's ``run_registry`` (one query)."""
    from google.cloud import bigquery

    from ..errors import RegistryError

    sql = f"SELECT DISTINCT run_id FROM `{settings.registry_table_ref('run_registry')}`"
    client = bigquery.Client(project=settings.project_id)
    try:
        return {str(row["run_id"]) for row in client.query(sql).result()}
    except Exception as exc:  # noqa: BLE001 - re-raised with registry context
        raise RegistryError(
            f"could not read run ids from {settings.registry_dataset_ref}: {exc}"
        ) from exc


def _statuses(
    settings: Settings, run_ids: Sequence[str]
) -> dict[str, str | None]:  # pragma: no cover - GCP I/O, @gcp smoke
    """``{run_id: latest header status or None}`` for the named runs (one query)."""
    from google.cloud import bigquery

    from ..errors import RegistryError

    sql = (
        "SELECT run_id, ARRAY_AGG(status ORDER BY created_at DESC LIMIT 1)[OFFSET(0)] AS status\n"
        f"FROM `{settings.registry_table_ref('run_registry')}`\n"
        "WHERE run_id IN UNNEST(@run_ids)\nGROUP BY run_id"
    )
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("run_ids", "STRING", list(run_ids))]
    )
    client = bigquery.Client(project=settings.project_id)
    try:
        found = {str(r["run_id"]): r["status"] for r in client.query(sql, job_config).result()}
    except Exception as exc:  # noqa: BLE001 - re-raised with registry context
        raise RegistryError(
            f"could not read run statuses from {settings.registry_dataset_ref}: {exc}"
        ) from exc
    return {r: found.get(r) for r in run_ids}


def _run_models(
    settings: Settings, run_ids: Sequence[str]
) -> tuple[str, ...]:  # pragma: no cover - GCP I/O, @gcp smoke
    """The BQML model object ids in this registry belonging to any of ``run_ids``.

    Nothing records these names, so the only way to find them is to list the dataset's models and
    match each back to a run (`bigquery_names.model_object_matches_run`).
    """
    from google.cloud import bigquery

    from ..engines.bigquery_names import model_object_matches_run

    client = bigquery.Client(project=settings.project_id)
    try:
        listed = [m.model_id for m in client.list_models(settings.registry_dataset_ref)]
    except Exception as exc:  # noqa: BLE001 - a dataset with no models is fine; a real error is not
        _log.warning("could not list BQML models in %s: %s", settings.registry_dataset_ref, exc)
        return ()
    return tuple(sorted(m for m in listed if any(model_object_matches_run(m, r) for r in run_ids)))


# --- verbs -------------------------------------------------------------------------


def init(
    *, settings: Settings | None = None, create_dataset: bool = False
) -> str:  # pragma: no cover - GCP I/O, @gcp smoke
    """Create this registry's five tables + three views in `Settings.registry_dataset_ref`.

    Idempotent (``CREATE TABLE IF NOT EXISTS`` + ``CREATE OR REPLACE VIEW``) and **registry-only**:
    the source panel is a separate concern with a separate lifetime and is never created here. Point
    ``SF_REGISTRY_DATASET_ID`` at a new dataset and this is how you stand up a second registry.

    Terraform owns dataset creation, so by default this expects the dataset to exist and fails with
    BigQuery's own 404 if it does not. ``create_dataset=True`` is the opt-in escape hatch for a
    registry outside the Terraform state — convenient, and it means the dataset's location and
    labels are not what Terraform would have given it. Returns the registry ref.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError
    from .ddl import render_create_tables, render_migrations
    from .tables import ensure_views

    resolved = _resolved(settings)
    client = bigquery.Client(project=resolved.project_id)
    if create_dataset:
        _log.warning(
            "creating dataset %s outside Terraform — its location/labels will not match the "
            "Terraform-managed datasets in this deployment",
            resolved.registry_dataset_ref,
        )
        client.create_dataset(resolved.registry_dataset_ref, exists_ok=True)

    # The registry half of what `tables.ensure_tables` does — CREATE, then the additive ALTERs, so
    # this both stands up a new registry and brings an older one up to the current column set. The
    # five run-collection tables are always native BigQuery, hence `iceberg=False` and no
    # connection. Two passes, not one merged dict: both renderers key on the table name, so merging
    # would drop a CREATE for every table that also has a migration.
    creates = render_create_tables(
        resolved.registry_dataset_ref, iceberg=False, tables=REGISTRY_TABLE_NAMES
    )
    migrations = render_migrations(resolved.registry_dataset_ref, tables=REGISTRY_TABLE_NAMES)
    for stage, statements in (("creating", creates), ("migrating", migrations)):
        for name, statement in statements.items():
            try:
                client.query(statement).result()
            except Exception as exc:  # noqa: BLE001 - re-raised with table context
                raise RegistryError(f"init failed {stage} {name}: {exc}") from exc
    ensure_views(settings=resolved)
    _log.info("registry ready: %s", resolved.registry_dataset_ref)
    return resolved.registry_dataset_ref


def doctor(*, settings: Settings | None = None) -> DoctorReport:  # pragma: no cover - GCP I/O
    """A read-only health report for this registry. Touches nothing.

    Answers the three questions an operator actually has: is it *there* (tables present, row counts
    per tier), is anything *stuck* (headers still in a `LIVE_STATUSES` state), and is anything
    *leaking* (artifact prefixes under this registry's root with no ``run_registry`` row). The last
    one is the state every pre-isolation reset left behind, and it is only answerable now that the
    artifact path carries the registry key.
    """
    from google.cloud import bigquery

    from .views import render_create_views

    resolved = _resolved(settings)
    client = bigquery.Client(project=resolved.project_id)
    notes: list[str] = []

    stats: list[TableStat] = []
    for name in REGISTRY_TABLE_NAMES:
        ref = resolved.registry_table_ref(name)
        try:
            rows = list(client.query(f"SELECT COUNT(*) AS n FROM `{ref}`").result())
            stats.append(TableStat(name, int(rows[0]["n"])))
        except Exception:  # noqa: BLE001 - an absent table is a finding, not a crash
            stats.append(TableStat(name, None))

    live: tuple[tuple[str, str], ...] = ()
    orphans: tuple[ArtifactPrefix, ...] = ()
    if all(s.exists for s in stats):
        statuses = ", ".join(f"'{s}'" for s in sorted(LIVE_STATUSES))
        sql = (
            "SELECT run_id, status FROM (\n"
            "  SELECT run_id, status, ROW_NUMBER() OVER "
            "(PARTITION BY run_id ORDER BY created_at DESC) AS rn\n"
            f"  FROM `{resolved.registry_table_ref('run_registry')}`\n"
            f") WHERE rn = 1 AND status IN ({statuses}) ORDER BY run_id"
        )
        live = tuple((str(r["run_id"]), str(r["status"])) for r in client.query(sql).result())
        known = _known_run_ids(resolved)
        prefixes = artifacts.list_prefixes(resolved)
        orphans = tuple(prefixes[r] for r in artifacts.orphan_run_ids(prefixes, known))
    else:
        notes.append("skipped the run/orphan checks — run init first, some tables are missing")

    return DoctorReport(
        registry=resolved.registry_dataset_ref,
        artifact_root=resolved.artifact_root,
        tables=tuple(stats),
        views=tuple(render_create_views(resolved.registry_dataset_ref)),
        live_runs=live,
        orphans=orphans,
        notes=tuple(notes),
    )


def plan_drop_run(
    run_ids: Sequence[str], *, settings: Settings | None = None
) -> DropPlan:  # pragma: no cover - GCP I/O, @gcp smoke
    """Everything `drop_run` would delete, without deleting any of it.

    Step 1 of the ordering rule: enumerate while the index still exists. A run absent from
    ``run_registry`` lands in ``unknown`` and is dropped from the plan — including its artifacts,
    because an unattributable prefix is `sweep_orphans`' job, not this verb's.
    """
    resolved = _resolved(settings)
    statuses = _statuses(resolved, list(run_ids))
    unknown = tuple(sorted(r for r, s in statuses.items() if s is None))
    live = blocking_runs(statuses)
    targets = tuple(sorted(r for r in run_ids if statuses.get(r) is not None))

    prefixes = artifacts.list_prefixes(resolved, targets) if targets else {}
    return DropPlan(
        registry=resolved.registry_dataset_ref,
        run_ids=targets,
        prefixes=tuple(prefixes[r] for r in targets if r in prefixes),
        models=_run_models(resolved, targets) if targets else (),
        blocked=live,
        unknown=unknown,
    )


def drop_run(
    run_ids: Sequence[str],
    *,
    settings: Settings | None = None,
    yes: bool = False,
    force: bool = False,
) -> DropPlan:  # pragma: no cover - GCP I/O, @gcp smoke
    """Delete named run(s) from every tier: GCS artifacts, BQML models, then registry rows.

    The daily verb, and the one that absorbs "prune" — pass as many ids as you like. Preview is the
    default: without ``yes=True`` this plans, logs the preview, and returns without touching
    anything. ``force=True`` overrides the in-flight refusal; check with `sdk.Forecaster.monitor`
    (``probe=True``) first, because a registry row marked ``RUNNING`` can mean either a live job or
    a dead one that never got to write its status.

    Order is artifacts → models → rows, never the reverse: the rows are the only index of which
    objects belong to the run, so deleting them first strands the artifacts permanently.

    Returns the executed plan. Raises `RegistryError` if the delete fails — most usefully when
    BigQuery rejects a ``DELETE`` against rows still in the streaming buffer, which happens for
    roughly 90 minutes after a run finishes writing.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    resolved = _resolved(settings)
    plan = plan_drop_run(run_ids, settings=resolved)
    _log.warning("%s", format_plan(plan))

    if plan.blocked and not force:
        raise RegistryError(
            f"refusing to drop: {', '.join(plan.blocked)} still in flight. Check with "
            "monitor(probe=True) — a RUNNING row can also be a dead job — then pass force=True."
        )
    if plan.is_empty:
        _log.warning("nothing to drop — none of the named runs are in %s", plan.registry)
        return plan
    if not yes:
        _log.warning("DRY RUN — nothing deleted. Re-run with yes=True to execute.")
        return plan

    # 1+2. Artifacts first, while the rows still say which objects exist.
    deleted = artifacts.delete_prefixes(resolved, plan.run_ids)
    _log.warning("deleted %d artifact objects", deleted)

    # 3a. BQML model objects (invisible to the registry — matched by name).
    client = bigquery.Client(project=resolved.project_id)
    for model_id in plan.models:
        ref = f"{resolved.registry_dataset_ref}.{model_id}"
        try:
            client.query(f"DROP MODEL IF EXISTS `{ref}`;").result()
        except Exception as exc:  # noqa: BLE001 - best-effort, reported and continued
            _log.warning("could not drop model %s: %s", ref, exc)

    # 3b. Rows.
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("run_ids", "STRING", list(plan.run_ids))]
    )
    for name, statement in render_delete_rows(resolved.registry_dataset_ref).items():
        try:
            client.query(statement, job_config=job_config).result()
        except Exception as exc:  # noqa: BLE001 - re-raised with table context
            raise RegistryError(
                f"drop_run failed deleting from {name}: {exc}. If the run just finished, its rows "
                "may still be in the Storage Write API streaming buffer (~90 min), which BigQuery "
                "will not let a DELETE touch — wait and retry."
            ) from exc
    _log.warning("dropped %d run(s) from %s", len(plan.run_ids), plan.registry)
    return plan


def plan_sweep_orphans(
    *, settings: Settings | None = None
) -> SweepPlan:  # pragma: no cover - GCP I/O, @gcp smoke
    """Artifact prefixes under this registry's root with no ``run_registry`` row."""
    resolved = _resolved(settings)
    known = _known_run_ids(resolved)
    prefixes = artifacts.list_prefixes(resolved)
    return SweepPlan(
        registry=resolved.registry_dataset_ref,
        root=resolved.artifact_root,
        prefixes=tuple(prefixes[r] for r in artifacts.orphan_run_ids(prefixes, known)),
        known_runs=len(known),
    )


def sweep_orphans(
    *, settings: Settings | None = None, yes: bool = False
) -> SweepPlan:  # pragma: no cover - GCP I/O, @gcp smoke
    """Delete artifact prefixes under *this* registry with no ``run_registry`` row.

    The cleanup for everything stranded before the artifact path carried a registry key. Scoping is
    what makes it safe: the sweep only ever looks under `Settings.artifact_root`, so another
    registry's objects in the same bucket are outside its world, and an object that cannot be
    attributed to a run at all is skipped rather than guessed at.

    Preview by default; ``yes=True`` executes. Returns the plan.
    """
    resolved = _resolved(settings)
    plan = plan_sweep_orphans(settings=resolved)
    _log.warning("%s", format_plan(plan))
    if not plan.prefixes:
        return plan
    if not yes:
        _log.warning("DRY RUN — nothing deleted. Re-run with yes=True to execute.")
        return plan
    deleted = artifacts.delete_prefixes(resolved, [p.run_id for p in plan.prefixes])
    _log.warning("swept %d orphan prefixes (%d objects)", len(plan.prefixes), deleted)
    return plan


def snapshot(
    suffix: str,
    *,
    settings: Settings | None = None,
    into: str | None = None,
    expiration_days: int | None = None,
) -> dict[str, str]:  # pragma: no cover - GCP I/O, @gcp smoke
    """Take a BigQuery table snapshot of each registry table; return ``{table: snapshot ref}``.

    Non-destructive, so there is no dry run — the whole point is to take one *before* the risky
    thing. ``suffix`` names the snapshot set (a date, a ticket, ``before_migration``);
    ``into`` redirects to another dataset so the registry itself stays uncluttered;
    ``expiration_days`` sets a TTL so a habit of snapshotting does not become a storage bill.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    resolved = _resolved(settings)
    target = into or resolved.registry_dataset_ref
    statements = render_snapshot_sql(
        resolved.registry_dataset_ref, target, suffix, expiration_days=expiration_days
    )
    client = bigquery.Client(project=resolved.project_id)
    out: dict[str, str] = {}
    for name, statement in statements.items():
        try:
            client.query(statement).result()
        except Exception as exc:  # noqa: BLE001 - re-raised with table context
            raise RegistryError(f"snapshot failed for {name}: {exc}") from exc
        out[name] = f"{target}.{name}_{suffix}"
    _log.info("snapshotted %d tables into %s", len(out), target)
    return out


def export(
    destination_root: str, *, settings: Settings | None = None, fmt: str = "PARQUET"
) -> dict[str, str]:  # pragma: no cover - GCP I/O, @gcp smoke
    """Export each registry table to GCS; return ``{table: destination prefix}``.

    Non-destructive but **overwriting** — each table's destination prefix is replaced, which is what
    makes re-exporting to a stable path idempotent. Use a run-stamped ``destination_root`` if you
    want a history of exports rather than a current one.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    resolved = _resolved(settings)
    statements = render_export_sql(resolved.registry_dataset_ref, destination_root, fmt=fmt)
    client = bigquery.Client(project=resolved.project_id)
    root = destination_root.rstrip("/")
    out: dict[str, str] = {}
    for name, statement in statements.items():
        try:
            client.query(statement).result()
        except Exception as exc:  # noqa: BLE001 - re-raised with table context
            raise RegistryError(f"export failed for {name}: {exc}") from exc
        out[name] = f"{root}/{name}/"
    _log.info("exported %d tables to %s", len(out), root)
    return out


# --- CLI ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI: ``python -m scale_forecasting.registry.ops <verb> [...]``.

    One subcommand per verb, the same six the SDK exposes (G1 — one implementation, three entry
    points). Destructive verbs preview by default and need ``--yes``.
    """
    import argparse

    p = argparse.ArgumentParser(
        prog="registry-ops",
        description="Manage one scale-forecasting registry (the dataset SF_REGISTRY_DATASET_ID, "
        "or SF_DATASET_ID, resolves to). No full-wipe verb — delete the dataset in BigQuery.",
    )
    sub = p.add_subparsers(dest="verb", required=True)

    p_init = sub.add_parser("init", help="create this registry's tables + views (idempotent)")
    p_init.add_argument(
        "--create-dataset",
        action="store_true",
        help="also create the dataset if absent (Terraform normally owns this)",
    )
    sub.add_parser("doctor", help="read-only health report")

    p_drop = sub.add_parser("drop-run", help="delete named run(s): artifacts, models, rows")
    p_drop.add_argument("run_ids", nargs="+", help="one or more run ids")
    p_drop.add_argument("--yes", action="store_true", help="execute (default is a preview)")
    p_drop.add_argument("--force", action="store_true", help="drop even if a run looks in-flight")

    p_sweep = sub.add_parser("sweep-orphans", help="delete artifacts with no run_registry row")
    p_sweep.add_argument("--yes", action="store_true", help="execute (default is a preview)")

    p_snap = sub.add_parser("snapshot", help="BigQuery table snapshots of the registry")
    p_snap.add_argument("suffix", help="names the snapshot set, e.g. 20260831")
    p_snap.add_argument("--into", default=None, help="target dataset (default: the registry's own)")
    p_snap.add_argument("--expiration-days", type=int, default=None, help="snapshot TTL in days")

    p_exp = sub.add_parser("export", help="dump the registry to GCS")
    p_exp.add_argument("destination", help="gs:// prefix to export under")
    p_exp.add_argument("--format", default="PARQUET", choices=EXPORT_FORMATS, dest="fmt")

    ns = p.parse_args(argv)
    if ns.verb == "init":
        init(create_dataset=ns.create_dataset)
    elif ns.verb == "doctor":
        _log.warning("%s", format_doctor(doctor()))
    elif ns.verb == "drop-run":
        drop_run(ns.run_ids, yes=ns.yes, force=ns.force)
    elif ns.verb == "sweep-orphans":
        sweep_orphans(yes=ns.yes)
    elif ns.verb == "snapshot":
        snapshot(ns.suffix, into=ns.into, expiration_days=ns.expiration_days)
    elif ns.verb == "export":
        export(ns.destination, fmt=ns.fmt)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
