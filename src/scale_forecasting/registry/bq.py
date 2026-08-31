"""Registry writers — the bulk BigQuery lineage layer.

Split into two halves along the pure/I-O seam:

- **Pure row assembly** (tested offline): turn a `CellResult` and
  a `RunConfig` into the exact ``list[dict]`` rows each table expects — column
  mapping, stamping ``run_id``/``ts_id``/``model_type``/``compute_engine``, JSON
  serialization, and the per-cell idempotency key.
- **I/O** (GCP-verified by the ``@gcp`` round-trip test):
  ``ensure_tables``, ``write_header``, ``update_header``, ``write_cells`` — execute DDL,
  INSERT/UPDATE the single-row header, and stream the three cell tables via the Storage
  Write API. Idempotency is **append-only + dedupe-on-read**: ``write_cells`` only appends
  (never DELETEs — a DELETE against rows still in the ~90-min streaming buffer is rejected),
  and serving views dedupe with ``DISTINCT``/``GROUP BY`` on ``run_id`` (+ cell keys).

The infra identity (project / dataset / connection / warehouse) is not on ``RunConfig`` —
it is resolved from the environment via `Settings`, so the identical writer code runs locally
under ADC and on Composer under the runner SA. Each
writer accepts an optional ``settings=`` for tests/callers that already hold one; otherwise it
resolves from ``SF_*`` env vars. GCP client libraries are imported lazily inside the writers so
the pure layer (and its offline tests) never need them installed.

Public surface: ``ensure_tables``, ``ensure_views``, ``write_header``, ``update_header``,
``merge_header_telemetry`` / ``sizing_telemetry_path`` (the accreting ``job_telemetry`` write —
several jobs of one run each record their own sizing without overwriting each other),
``run_header`` (the header-lifecycle context manager), ``resolve_snapshot_millis`` /
``snapshot_millis_for`` (the run's input-data snapshot: resolve once, look up per job),
``write_cells``, the ``run_jobs`` per-job
writers/readers (``write_job``, ``update_job``, ``latest_job_attempt``, ``next_job_attempt``,
``read_run_jobs``) and ``run_job`` (the per-job lifecycle context manager), the read surface
(``header_status``, ``read_run_summary``, ``read_leaderboard``, ``read_prediction_counts``,
``read_cell_timing``, ``parse_ts`` (the shared registry-timestamp coercion the review/probe age
arithmetic both read through), the compute-harvest reads ``read_compute_harvest`` /
``discover_harvest_run`` (what `compute.profile.source` resolves against), and the review-layer
reads ``read_run_config`` / ``read_progress`` /
``read_metric_aggregates`` / ``read_cell_metrics``), plus the pure assemblers used by the writers
and the tests.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, get_args

from ..config import DecisionMetric
from ..errors import get_logger

_log = get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..config import RunConfig
    from ..settings import Settings
    from ..worker import CellResult

# The full metric panel, in table-column order — derived from the config's DecisionMetric
# literal so there is exactly one source of truth.
METRIC_COLUMNS: tuple[str, ...] = get_args(DecisionMetric)


# --- pure row assembly ---------------------------------------------------------


def cell_dedup_key(result: CellResult) -> dict[str, str]:
    """The run-scoped identity anchor for a cell's rows.

    Idempotency is **append-only + dedupe-on-read**, anchored on ``run_id``. `write_cells`
    never DELETEs — a DELETE that matches rows still in the Storage Write API streaming buffer
    is rejected for the whole buffer window (~90 min), so a clear-then-append is not viable
    against the default stream. Instead we rely on
    ``run_id`` being a pure function of the config (``make_run_id``): the same ``run_id`` implies
    the same config implies byte-identical rows, so a re-run's "duplicates" are exact copies.
    Serving views dedupe with ``DISTINCT``/``GROUP BY`` on ``run_id`` (+ cell keys); no write-time
    delete is needed. ``model_hash`` uniquely identifies the cell on ``forecast_metadata`` for
    lineage.
    """
    return {"run_id": result.run_id}


def assemble_prediction_rows(result: CellResult) -> list[dict[str, Any]]:
    """Canonical prediction frame → ``forecast_predictions`` rows.

    Stamps run/series/model/engine onto each row and maps ``ds`` → ``forecast_date``.
    ``quantiles`` is serialized to a JSON string (or None).
    """
    rows: list[dict[str, Any]] = []
    for rec in result.predictions.to_dict("records"):
        rows.append(
            {
                "run_id": result.run_id,
                "ts_id": result.ts_id,
                "model_type": result.model_type,
                "compute_engine": result.compute_engine,
                "forecast_date": _as_date(rec["ds"]),
                "yhat": _as_float(rec.get("yhat")),
                "yhat_lower": _as_float(rec.get("yhat_lower")),
                "yhat_upper": _as_float(rec.get("yhat_upper")),
                "quantiles": _as_json(rec.get("quantiles")),
            }
        )
    return rows


def assemble_oof_rows(result: CellResult) -> list[dict[str, Any]]:
    """Canonical OOF frame → ``backtest_oof`` rows. Empty if no backtest."""
    if result.oof is None:
        return []
    rows: list[dict[str, Any]] = []
    for rec in result.oof.to_dict("records"):
        rows.append(
            {
                "run_id": result.run_id,
                "ts_id": result.ts_id,
                "model_type": result.model_type,
                "fold_id": int(rec["fold_id"]),
                "forecast_date": _as_date(rec["ds"]),
                "y_true": _as_float(rec.get("y_true")),
                "yhat": _as_float(rec.get("yhat")),
            }
        )
    return rows


def assemble_metadata_row(
    result: CellResult, created_at: datetime, model_artifact: str | None = None
) -> dict[str, Any]:
    """One full-fit ``forecast_metadata`` row: metrics panel + artifact link.

    ``fold_id`` is None (this is the full-fit summary row). ``model_artifact`` is the
    ObjectRef/URI filled in by the writer after the artifact upload. ``worker_id`` and the
    ``cell_started_at``/``cell_ended_at`` wall-clock bracket come off the cell (the Python worker
    stamps them); they are None for cells produced outside `run_cell` (native SQL / ensemble).
    """
    row: dict[str, Any] = {
        "run_id": result.run_id,
        "ts_id": result.ts_id,
        "model_type": result.model_type,
        "compute_engine": result.compute_engine,
        "model_hash": result.model_hash,
        "fold_id": None,
        "fit_seconds": _as_float(result.fit_seconds),
        "best_params": _as_json(result.best_params),
        "model_artifact": model_artifact,
        "created_at": created_at,
        "worker_id": result.worker_id,
        "cell_started_at": result.cell_started_at,
        "cell_ended_at": result.cell_ended_at,
        # Harvested compute measurement (compute.profile.measure). All None when measurement is
        # off, which is also how rows written before these columns existed read back — so
        # `profiling.harvest_profile` needs no version check, only a NULL check.
        "cpu_seconds": _as_float(result.cpu_seconds),
        "process_rss_bytes": result.process_rss_bytes,
        "peak_gpu_bytes": result.peak_gpu_bytes,
        "intraop_threads": result.intraop_threads,
        "n_obs": result.n_obs,
    }
    for name in METRIC_COLUMNS:
        row[name] = _as_float(result.metrics.get(name))
    return row


def assemble_header_row(
    cfg: RunConfig,
    run_id: str,
    created_at: datetime,
    *,
    snapshot_millis: int | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Build the ``run_registry`` header row from a config.

    ``raw_config`` is the validated config as a **dict** — the config *is* the record.
    ``run_registry.raw_config`` is a native ``JSON`` column, and the client's JSON query
    parameter serializes the value itself (``json.dumps``), so the row must carry the dict, not
    a pre-serialized string (a string would be double-encoded). ``bq_models`` is left empty here
    and filled by the router once model runtimes are known; status starts RUNNING.

    ``user_id`` is the principal that launched the run (the ADC identity resolved by `write_header`
    via `identity.resolve_principal` — a runner SA under Composer/CI, a user's email on a laptop),
    stamped so *launch* is attributable in the audit trail. ``None`` leaves it NULL (the pre-audit
    behavior, and when the principal couldn't be resolved cheaply).

    ``snapshot_millis`` is the input-data snapshot the run pins every read to (epoch millis on the
    BigQuery clock, resolved once by `resolve_snapshot_millis`): stored on the header so every
    family job — whichever runtime — can look it up by ``run_id`` (`snapshot_millis_for`) and read
    the *identical* source state. It is deliberately **not** part of the config (it would perturb
    the config-derived ``run_id``), so it is passed in here, not derived. ``None`` leaves it NULL —
    the reads fall back to unpinned (the pre-snapshot behavior).
    """
    return {
        "run_id": run_id,
        "created_at": created_at,
        "snapshot_millis": snapshot_millis,
        "user_id": user_id,
        "git_sha": None,
        "python_runtime": cfg.python_runtime,
        "bq_models": [],
        "backtest_on": cfg.backtest.enabled,
        "decision_metric": cfg.backtest.decision_metric,
        "ensemble_strategies": list(cfg.ensemble.strategies) if cfg.ensemble.enabled else [],
        "raw_config": cfg.model_dump(mode="json"),
        "status": "RUNNING",
        "n_series": cfg.data.series_limit,
        "n_models": len(cfg.models),
        "runtime_seconds": None,
        # Dataproc-level job telemetry (executor sizing, wall/startup split, DCU usage): a native
        # JSON column filled in after the batch finishes by the submitter (extract_job_telemetry →
        # update_header) as a **dict** (the JSON query param serializes it). NULL here at RUNNING
        # and for any run whose telemetry couldn't be read (best-effort — never blocks a run).
        # See the run_registry DDL.
        "job_telemetry": None,
    }


def assemble_job_row(
    run_id: str,
    family: str,
    attempt: int,
    created_at: datetime,
    *,
    runtime: str | None = None,
    spark_mode: str | None = None,
    hardware: str | None = None,
    gpu_type: str | None = None,
    system_job_id: str | None = None,
    status: str = "RUNNING",
    started_at: datetime | None = None,
    probe_handle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a ``run_jobs`` row for one family's job under a run.

    The ``job_id`` is derived here (`registry.ids.make_job_key`) from ``(run_id, family, attempt)``
    so the row's identity always matches the id a submitter hands the platform — the two can't
    drift. The resolved compute fields (``runtime``/``spark_mode``/``hardware``/``gpu_type``) are
    passed in, not re-derived, so this stays a pure mapping: the orchestrator resolves them
    (`config.RunConfig.resolve_family_compute` for a model family, the ensemble node's own config
    for ``ensemble``) and hands them over. ``status`` starts RUNNING; ``runtime_seconds`` and
    ``job_telemetry`` are NULL until the job finishes and the submitter updates the row.

    ``started_at`` is the job's execution start (defaults to ``created_at`` when not given); the
    matching ``ended_at`` is NULL here and stamped by `run_job` at exit — together they give the
    trace an absolute wall-clock lane per job, alongside the measured ``runtime_seconds``.
    """
    from .ids import make_job_key

    return {
        "job_id": make_job_key(run_id, family, attempt),
        "run_id": run_id,
        "family": family,
        "attempt": attempt,
        "runtime": runtime,
        "spark_mode": spark_mode,
        "hardware": hardware,
        "gpu_type": gpu_type,
        "system_job_id": system_job_id,
        "status": status,
        "created_at": created_at,
        "started_at": started_at if started_at is not None else created_at,
        "ended_at": None,
        "runtime_seconds": None,
        # The probe handle (runtime coordinates for reconciliation) is stamped at RUNNING entry so a
        # reader can check a live job; NULL when no handle was captured (a pre-feature run).
        "job_telemetry": {"probe_handle": probe_handle} if probe_handle is not None else None,
    }


# --- small pure coercers -------------------------------------------------------


def _as_float(value: Any) -> float | None:
    """Coerce to float, mapping missing/non-finite to None (BQ NULL).

    The BigQuery Storage Write API rejects NaN and ±Inf for a FLOAT64 column, and a single
    rejected row fails the whole ``append_rows`` request — which, in a Spark/Ray worker, kills
    the task and cascades to the entire run. A non-finite forecast is a per-series pathology
    (e.g. ``log1p``'s ``expm1`` inverse overflowing to ``+Inf`` on a runaway series), so it must
    not take the fleet down: coerce it to NULL here, at the one boundary every engine's rows flow
    through, so the bad cell lands as a missing value and the run completes.
    """
    if value is None:
        return None
    f = float(value)
    return f if math.isfinite(f) else None  # NaN and ±Inf → NULL


def _as_json(value: Any) -> str | None:
    """Serialize a dict (or None/empty) to a JSON string, or None.

    Non-finite values (NaN/±Inf) are dropped from a dict before serializing: Python's
    ``json.dumps`` emits the bare literals ``NaN``/``Infinity`` by default, which are invalid
    JSON — and BigQuery's ``JSON`` column parser rejects them ("syntax error while parsing value
    - invalid literal"), failing the whole Storage Write API append. A quantile dict on a runaway
    series (``log1p``'s ``expm1`` overflow) can carry such values; dropping the offending keys —
    parity with `_as_float`'s scalar NULL — keeps the row writable. An all-non-finite dict
    collapses to NULL.
    """
    if value is None or (isinstance(value, dict) and not value):
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        clean = {
            k: v
            for k, v in value.items()
            if not (isinstance(v, float) and not math.isfinite(v))
        }
        if not clean:
            return None
        return json.dumps(clean, sort_keys=True)
    return json.dumps(value, sort_keys=True)


def _as_date(value: Any) -> Any:
    """Normalize a timestamp-ish value to a ``date`` for BQ DATE columns."""
    if hasattr(value, "date"):
        return value.date()
    return value


# --- I/O: Storage Write API encoding (proto per table) -------------------------
#
# The Storage Write API serializes rows as protobuf. Each cell table has a fixed column
# order (its "spec"): the field number is the write order and the type char selects the
# proto scalar type. These specs mirror the DDL and the keys the pure assemblers emit,
# so a spec and its assembler must be edited together.
#
# Type chars → proto scalar (and how the value is encoded):
#   "S" TYPE_STRING  — str as-is; a date/datetime is ``.isoformat()``-ed (DATE→"YYYY-MM-DD",
#                      TIMESTAMP→RFC3339). Both are accepted by the Write API as strings.
#   "D" TYPE_DOUBLE  — FLOAT64.
#   "I" TYPE_INT64   — INT64.
# A None value leaves the (proto2-optional) field unset, which the Write API writes as NULL.

_PRED_SPEC: tuple[tuple[str, str], ...] = (
    ("run_id", "S"),
    ("ts_id", "S"),
    ("model_type", "S"),
    ("compute_engine", "S"),
    ("ensemble_id", "S"),
    ("forecast_date", "S"),
    ("yhat", "D"),
    ("yhat_lower", "D"),
    ("yhat_upper", "D"),
    ("quantiles", "S"),
)

_OOF_SPEC: tuple[tuple[str, str], ...] = (
    ("run_id", "S"),
    ("ts_id", "S"),
    ("model_type", "S"),
    ("fold_id", "I"),
    ("forecast_date", "S"),
    ("y_true", "D"),
    ("yhat", "D"),
)

_META_SPEC: tuple[tuple[str, str], ...] = (
    ("run_id", "S"),
    ("ts_id", "S"),
    ("model_type", "S"),
    ("compute_engine", "S"),
    ("model_hash", "S"),
    ("ensemble_id", "S"),
    ("fold_id", "I"),
    *((name, "D") for name in METRIC_COLUMNS),
    ("fit_seconds", "D"),
    ("best_params", "S"),
    ("model_artifact", "S"),
    ("created_at", "S"),
    # Per-cell wall-clock lane + worker identity for the run trace (SDK trace()). None on rows the
    # native/ensemble engines emit (no per-cell worker); those families trace at the run_jobs grain.
    ("worker_id", "S"),
    ("cell_started_at", "S"),
    ("cell_ended_at", "S"),
    # What the fit cost, harvested from the fit the run was doing anyway (compute.profile.measure).
    # `fit_seconds` above is the wall-clock half; together these let a completed run be read back
    # as a `ComputeProfile` that sizes a later one — see profiling.harvest_profile.
    ("cpu_seconds", "D"),
    ("process_rss_bytes", "I"),
    ("peak_gpu_bytes", "I"),
    ("intraop_threads", "I"),
    ("n_obs", "I"),
)

# Which assembler feeds which table, and its column spec. Driven in this one place so
# write_cells stays a loop, not three near-identical blocks.
_CELL_TABLES: tuple[str, ...] = ("forecast_predictions", "backtest_oof", "forecast_metadata")

# Write API request limits: batch serialized rows under ~10MB and a sane row cap per request.
_MAX_REQUEST_BYTES = 9 * 1024 * 1024
_MAX_REQUEST_ROWS = 10_000

# Retry-on-transient for the Storage Write API append. The service returns transient 500/503/429s
# (e.g. a 500 "error while verifying authorization") that Google's own guidance says to retry with
# backoff — over a multi-hour 100k run we make thousands of appends, so ≥1 blip is likely and must
# not fail an otherwise-complete run. Safe by construction: the default stream is at-least-once and
# the registry dedupes-on-read under a deterministic run_id, so re-sending a whole append never
# double-counts. Naming mirrors ray_submit.py's manual-retry idiom.
_WRITE_RETRY_ATTEMPTS = 5  # total attempts per append (1 initial + 4 retries)
_WRITE_RETRY_BACKOFF_SECONDS = 2.0  # exponential base: 2, 4, 8, 16s


def _proto_for(table_name: str, spec: tuple[tuple[str, str], ...]) -> tuple[Any, Any]:
    """Build a protobuf message class + descriptor matching a table's column spec.

    Fields are proto2-optional (so an unset field → BigQuery NULL) and numbered by write
    order. A private `DescriptorPool` isolates the registration so repeated calls (or
    two tables in one process) never collide on a duplicate proto file name.
    """
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

    type_map = {
        "S": descriptor_pb2.FieldDescriptorProto.TYPE_STRING,
        "D": descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE,
        "I": descriptor_pb2.FieldDescriptorProto.TYPE_INT64,
    }
    optional = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL

    dp = descriptor_pb2.DescriptorProto()
    dp.name = "Row"
    for number, (col, type_char) in enumerate(spec, start=1):
        field = dp.field.add()
        field.name = col
        field.number = number
        field.type = type_map[type_char]
        field.label = optional

    file_dp = descriptor_pb2.FileDescriptorProto()
    file_dp.name = f"{table_name}.proto"
    file_dp.package = "scale_forecasting.registry"
    file_dp.message_type.add().CopyFrom(dp)

    pool = descriptor_pool.DescriptorPool()
    file_desc = pool.Add(file_dp)
    msg_desc = file_desc.message_types_by_name["Row"]
    msg_cls = message_factory.GetMessageClass(msg_desc)

    proto_descriptor = descriptor_pb2.DescriptorProto()
    msg_desc.CopyToProto(proto_descriptor)
    return msg_cls, proto_descriptor


def _encode_rows(
    msg_cls: Any, spec: tuple[tuple[str, str], ...], rows: list[dict[str, Any]]
) -> list[bytes]:
    """Serialize assembled row dicts to protobuf bytes per the column spec.

    None values are left unset (→ NULL). ``"S"`` columns holding a date/datetime are
    ``.isoformat()``-ed; everything else is set directly.
    """
    serialized: list[bytes] = []
    for row in rows:
        msg = msg_cls()
        for col, type_char in spec:
            value = row.get(col)
            if value is None:
                continue
            if type_char == "S":
                setattr(msg, col, value.isoformat() if hasattr(value, "isoformat") else value)
            elif type_char == "D":
                setattr(msg, col, float(value))
            else:  # "I"
                setattr(msg, col, int(value))
        serialized.append(msg.SerializeToString())
    return serialized


def _chunk_rows(serialized: list[bytes]) -> list[list[bytes]]:
    """Split serialized rows into request-sized batches (Write API ~10MB/request)."""
    batches: list[list[bytes]] = []
    batch: list[bytes] = []
    size = 0
    for row in serialized:
        row_bytes = len(row)
        if batch and (size + row_bytes > _MAX_REQUEST_BYTES or len(batch) >= _MAX_REQUEST_ROWS):
            batches.append(batch)
            batch, size = [], 0
        batch.append(row)
        size += row_bytes
    if batch:
        batches.append(batch)
    return batches


def _append_via_write_api(
    write_client: Any,
    project: str,
    dataset: str,
    table: str,
    proto_descriptor: Any,
    serialized: list[bytes],
) -> None:
    """Append serialized rows to a table's default stream via the Storage Write API.

    Uses the direct ``append_rows(requests=...)`` bidi call (not the ``AppendRowsStream``
    wrapper, which masks the underlying gRPC error): the first request carries the
    stream + writer schema, each subsequent request carries only rows.

    Transient service errors (500/503/429/deadline) are retried with exponential backoff — safe
    because the default stream is at-least-once and the registry dedupes-on-read (re-sending the
    whole append can't double-count). A permanent error (bad schema/data, a response-level row
    error, or a non-transient API error) is re-raised as `RegistryError` with the real
    error attached, unchanged.
    """
    from google.api_core.exceptions import (
        DeadlineExceeded,
        GoogleAPICallError,
        InternalServerError,
        ServiceUnavailable,
        TooManyRequests,
    )
    from google.cloud.bigquery_storage_v1 import types

    from ..errors import RegistryError

    # Transient server-side errors the append should retry rather than fail the run on.
    transient = (InternalServerError, ServiceUnavailable, TooManyRequests, DeadlineExceeded)

    def _is_retryable(exc: GoogleAPICallError) -> bool:
        """Whether an append error is a transient blip worth retrying (vs a real data/config fault).

        Covers the transient status types plus one intermittent 400 the Storage Write API emits
        under long, high-volume append streams: ``Cannot route on empty project id ''`` — a gRPC
        routing-header race, NOT a real empty project (the same client wrote fine for hours before
        it). A fresh ``append_rows`` call repopulates the routing header. Genuine 400s (bad schema,
        proto mismatch) don't mention routing, so they still fail fast.
        """
        if isinstance(exc, transient):
            return True
        msg = str(exc).lower()
        return "empty project id" in msg or "cannot route" in msg

    parent = write_client.table_path(project, dataset, table)
    stream = f"{parent}/_default"  # default stream = at-least-once, no stream management
    batches = _chunk_rows(serialized)

    def requests() -> Any:
        for i, batch in enumerate(batches):
            request = types.AppendRowsRequest()
            proto_data = types.AppendRowsRequest.ProtoData()
            if i == 0:
                request.write_stream = stream
                proto_data.writer_schema = types.ProtoSchema(proto_descriptor=proto_descriptor)
            proto_rows = types.ProtoRows()
            proto_rows.serialized_rows.extend(batch)
            proto_data.rows = proto_rows
            request.proto_rows = proto_data
            yield request

    # Rebuild the request iterator per attempt: requests() is a generator *function*, so each call
    # yields a fresh, complete iterator (a retry re-sends the whole append).
    for attempt in range(1, _WRITE_RETRY_ATTEMPTS + 1):
        try:
            for response in write_client.append_rows(requests=requests()):
                if response.error.code != 0:
                    # Surface the per-row detail: the top-level message only says "N Errors
                    # found"; row_errors names the offending field/index/reason, which is what a
                    # caller needs to diagnose a data pathology (e.g. a non-finite float) without
                    # reverse-engineering it from the landed rows. A response-level error is a
                    # data/schema problem, not a transient blip — fail fast, don't retry.
                    detail = "; ".join(
                        f"row {re.index}: {re.message}"
                        for re in getattr(response, "row_errors", [])
                    )
                    raise RegistryError(
                        f"Storage Write API append to {table} failed: "
                        f"{response.error.code} {response.error.message}"
                        + (f" [{detail}]" if detail else "")
                    )
            return
        except GoogleAPICallError as exc:
            # Non-retryable API error (real bad schema/data) — same behavior as before the retry
            # loop existed: fail fast with the real error attached.
            if not _is_retryable(exc):
                raise RegistryError(
                    f"Storage Write API append to {table} failed: {exc}"
                ) from exc
            if attempt == _WRITE_RETRY_ATTEMPTS:
                raise RegistryError(
                    f"Storage Write API append to {table} failed after {attempt} attempts: {exc}"
                ) from exc
            backoff = _WRITE_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
            _log.warning(
                "transient Storage Write API error on %s (attempt %d/%d), retrying in %.0fs: %s",
                table,
                attempt,
                _WRITE_RETRY_ATTEMPTS,
                backoff,
                exc,
            )
            time.sleep(backoff)


# --- I/O: the four registry writers --------------------------------------------


def _resolve_settings(settings: Settings | None) -> Settings:
    """Return the passed settings, or resolve from the ``SF_*`` environment."""
    if settings is not None:
        return settings
    from ..settings import Settings as _Settings

    return _Settings.resolve()


def ensure_tables(
    cfg: RunConfig | None = None, *, settings: Settings | None = None
) -> None:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Create every registry + source table if absent (idempotent DDL).

    Renders the deployment DDL for the resolved dataset — native registry (native ``JSON``
    columns) plus both source variants, ``source_series_iceberg`` (managed Iceberg) and
    ``source_series_native`` (plain) — and executes each statement. ``cfg`` is accepted for
    signature symmetry with the other writers but is unused — the schema is fixed, not
    config-driven. Raises `RegistryError` on a DDL failure.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError
    from .ddl import render_deployment_ddl, render_migrations

    resolved = _resolve_settings(settings)
    ddl = render_deployment_ddl(
        resolved.dataset_ref,
        connection=resolved.connection,
        warehouse_uri=resolved.warehouse_uri,
    )
    client = bigquery.Client(project=resolved.project_id)
    for name, statement in ddl.items():
        try:
            client.query(statement).result()
        except Exception as exc:  # noqa: BLE001 - re-raised with table context
            raise RegistryError(f"ensure_tables failed creating {name}: {exc}") from exc

    # Additive schema evolution: bring tables created under an older schema up to the current
    # column set (ADD COLUMN IF NOT EXISTS). A fresh CREATE already has every column, so these
    # ALTERs are no-ops on it; on a pre-existing table they back-fill new nullable columns.
    migrations = render_migrations(resolved.dataset_ref)
    for name, statement in migrations.items():
        try:
            client.query(statement).result()
        except Exception as exc:  # noqa: BLE001 - re-raised with table context
            raise RegistryError(f"ensure_tables failed migrating {name}: {exc}") from exc

    # Curated analyst views sit on top of the tables — create them in the same setup pass so the
    # reviewable read surface (v_run_summary / v_model_leaderboard) exists after any run.
    ensure_views(settings=resolved)


def ensure_views(
    *, settings: Settings | None = None
) -> None:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Create/replace the analyst views over the registry (idempotent).

    Renders the ``CREATE OR REPLACE VIEW`` statements (`registry.views.render_create_views`)
    for the resolved dataset and executes each. Called by `ensure_tables`; safe to call on its
    own to refresh view definitions after a change. Raises `RegistryError` on failure.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError
    from .views import render_create_views

    resolved = _resolve_settings(settings)
    views = render_create_views(resolved.dataset_ref)
    client = bigquery.Client(project=resolved.project_id)
    for name, statement in views.items():
        try:
            client.query(statement).result()
        except Exception as exc:  # noqa: BLE001 - re-raised with view context
            raise RegistryError(f"ensure_views failed creating {name}: {exc}") from exc


def drop_all(
    *, settings: Settings | None = None
) -> None:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Drop all seven registry + source tables and the analyst views (the reset path).

    **Destructive.** Renders `registry.ddl.render_drop_tables` and executes each
    ``DROP TABLE IF EXISTS`` (plus ``DROP VIEW IF EXISTS`` for the two analyst views), so a
    subsequent `ensure_tables` recreates everything in the current native/dual-format
    shape — the Iceberg→native registry switch is a drop-and-recreate, not an ``ALTER``. Callers
    are responsible for confirming intent before invoking. Raises `RegistryError` on
    failure.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError
    from .ddl import render_drop_tables
    from .views import render_create_views

    resolved = _resolve_settings(settings)
    client = bigquery.Client(project=resolved.project_id)

    # Views depend on the tables — drop them first so the table drops don't trip a dependency.
    for name in render_create_views(resolved.dataset_ref):
        statement = f"DROP VIEW IF EXISTS `{resolved.table_ref(name)}`;"
        try:
            client.query(statement).result()
        except Exception as exc:  # noqa: BLE001 - re-raised with view context
            raise RegistryError(f"drop_all failed dropping view {name}: {exc}") from exc

    for name, statement in render_drop_tables(resolved.dataset_ref).items():
        try:
            client.query(statement).result()
        except Exception as exc:  # noqa: BLE001 - re-raised with table context
            raise RegistryError(f"drop_all failed dropping {name}: {exc}") from exc


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
        f"SELECT snapshot_millis FROM `{resolved.table_ref('run_registry')}` "
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
        f"INSERT INTO `{resolved.table_ref('run_registry')}` "
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
    sql = f"UPDATE `{resolved.table_ref('run_registry')}` SET {set_clause} WHERE run_id=@run_id"
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
    """Where one sizing record (`resources.sizing_telemetry`) is filed on the header (pure).

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
    sql = render_header_telemetry_merge(resolved.table_ref("run_registry"), paths)
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
        f"SELECT status FROM `{resolved.table_ref('run_registry')}` "
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


def parse_ts(value: Any) -> datetime | None:
    """Coerce a registry timestamp to a timezone-aware UTC ``datetime``, or ``None`` if it isn't.

    A registry row's timestamp arrives as a ``datetime`` from the BigQuery client but as an ISO
    string from a JSON-shaped reader dict (and from every offline test), so both readers that do
    age arithmetic — `review._assemble_progress`'s quiet-time and `probes._is_stale`'s escalation
    grace — need the same coercion. Pure and defensive: anything unparseable comes back ``None``
    rather than raising, so a malformed timestamp costs a *signal*, never a monitor.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def read_run_summary(
    run_id: str, *, settings: Settings | None = None
) -> dict[str, Any] | None:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Return the ``v_run_summary`` row for ``run_id`` (status + scaling/efficiency), or ``None``.

    The view already keeps one row per run (latest header wins), so this is a single-row lookup —
    the run-level answer to "how did this run go, and how efficiently?". Raises `RegistryError` on
    failure.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    resolved = _resolve_settings(settings)
    sql = f"SELECT * FROM `{resolved.table_ref('v_run_summary')}` WHERE run_id=@run_id"
    params = [_header_param("run_id", run_id)]
    client = bigquery.Client(project=resolved.project_id)
    try:
        rows = list(
            client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RegistryError(f"read_run_summary failed for run {run_id}: {exc}") from exc
    return dict(rows[0]) if rows else None


def read_leaderboard(
    run_id: str, *, settings: Settings | None = None
) -> list[dict[str, Any]]:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Return the ``v_model_leaderboard`` rows for ``run_id`` — one per (model, ensemble).

    Ordered by the mean decision-metric error (WAPE) ascending so the best model is first; a model
    with no scored metric (NULL) sorts last. Raises `RegistryError` on failure.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    resolved = _resolve_settings(settings)
    sql = (
        f"SELECT * FROM `{resolved.table_ref('v_model_leaderboard')}` "
        "WHERE run_id=@run_id ORDER BY mean_wape ASC NULLS LAST"
    )
    params = [_header_param("run_id", run_id)]
    client = bigquery.Client(project=resolved.project_id)
    try:
        rows = list(
            client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RegistryError(f"read_leaderboard failed for run {run_id}: {exc}") from exc
    return [dict(r) for r in rows]


def read_prediction_counts(
    run_id: str, *, settings: Settings | None = None
) -> dict[str, int]:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Return ``{model_type: forecast row count}`` for ``run_id`` from ``forecast_predictions``.

    The direct proof that fits produced forecasts (not just metadata): a model whose cells all
    failed writes metadata rows but **zero** predictions, so it still shows on the leaderboard yet
    has a count of 0 here. Raises `RegistryError` on failure.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    resolved = _resolve_settings(settings)
    sql = (
        f"SELECT model_type, COUNT(*) AS n FROM `{resolved.table_ref('forecast_predictions')}` "
        "WHERE run_id=@run_id GROUP BY model_type"
    )
    params = [_header_param("run_id", run_id)]
    client = bigquery.Client(project=resolved.project_id)
    try:
        rows = list(
            client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RegistryError(f"read_prediction_counts failed for run {run_id}: {exc}") from exc
    return {str(r["model_type"]): int(r["n"]) for r in rows}


# --- I/O: run_jobs (per-job identity + trace) ----------------------------------

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
        f"INSERT INTO `{resolved.table_ref('run_jobs')}` "
        f"({', '.join(columns)}) VALUES ({placeholders})"
    )
    params = [_job_param(col, row[col]) for col in columns]
    client = bigquery.Client(project=resolved.project_id)
    try:
        client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RegistryError(f"write_job failed for job {row.get('job_id')}: {exc}") from exc


def update_job(
    job_id: str, *, settings: Settings | None = None, **fields: Any
) -> None:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Update named columns on a job's ``run_jobs`` row, e.g. status/runtime_seconds/telemetry.

    ``update_job(job_id, status="COMPLETED", runtime_seconds=42.0)`` → a parameterized
    ``UPDATE … WHERE job_id=@job_id``. The ``job_id`` is 1:1 with a row, so exactly one row is
    touched. Unknown column names raise `RegistryError`; a no-op call (no fields) returns without
    touching BigQuery.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    if not fields:
        return
    unknown = set(fields) - set(_JOB_PARAM_TYPES)
    if unknown:
        raise RegistryError(f"update_job: unknown run_jobs column(s): {sorted(unknown)}")

    resolved = _resolve_settings(settings)
    set_clause = ", ".join(f"{col} = @{col}" for col in fields)
    sql = f"UPDATE `{resolved.table_ref('run_jobs')}` SET {set_clause} WHERE job_id=@job_id"
    params = [_job_param(col, value) for col, value in fields.items()]
    params.append(_job_param("job_id", job_id))
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
        f"SELECT MAX(attempt) AS max_attempt FROM `{resolved.table_ref('run_jobs')}` "
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
        f"SELECT * FROM `{resolved.table_ref('v_run_jobs')}` "
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


def read_cell_timing(
    run_id: str, *, limit: int = 5000, settings: Settings | None = None
) -> list[dict[str, Any]]:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Return per-cell wall-clock brackets for ``run_id`` — the fine grain under the per-job trace.

    Reads ``forecast_metadata`` directly (no view — this is a fine-grained, bounded read, not a
    curated roll-up): one row per full-fit cell (``fold_id IS NULL``) that recorded a wall-clock
    bracket (``cell_started_at IS NOT NULL`` — older rows predate the trace columns, so skipped),
    carrying ``ts_id`` / ``model_type`` / ``compute_engine`` / ``worker_id`` and the
    ``cell_started_at`` / ``cell_ended_at`` stamps. Writes are append-only + at-least-once, so it
    collapses to one row per cell (latest write wins) with the same grain as ``v_model_leaderboard``
    before returning. Ordered by ``cell_started_at`` and capped at ``limit`` rows (a 100k-cell run
    would swamp a trace plot; the cap keeps the read bounded — the per-job trace stays the
    whole-run view). Raises `RegistryError` on failure.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    resolved = _resolve_settings(settings)
    sql = (
        "SELECT ts_id, model_type, compute_engine, worker_id, cell_started_at, cell_ended_at "
        f"FROM `{resolved.table_ref('forecast_metadata')}` "
        "WHERE run_id=@run_id AND fold_id IS NULL AND cell_started_at IS NOT NULL "
        "QUALIFY ROW_NUMBER() OVER ("
        "PARTITION BY run_id, ts_id, model_type, fold_id, ensemble_id "
        "ORDER BY created_at DESC) = 1 "
        "ORDER BY cell_started_at LIMIT @limit"
    )
    params = [
        bigquery.ScalarQueryParameter("run_id", "STRING", run_id),
        bigquery.ScalarQueryParameter("limit", "INT64", limit),
    ]
    client = bigquery.Client(project=resolved.project_id)
    try:
        rows = list(
            client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RegistryError(f"read_cell_timing failed for run {run_id}: {exc}") from exc
    return [dict(r) for r in rows]


def read_run_config(
    run_id: str, *, settings: Settings | None = None
) -> dict[str, Any] | None:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Return the validated ``raw_config`` a run landed under, as a dict — or ``None`` if never run.

    Reads the latest ``run_registry`` header's ``raw_config`` JSON column for ``run_id`` (the config
    *is* the experiment record). The review layer rebuilds a `config.RunConfig` from this to recover
    the run's plan — models per family, decision metric, ensemble strategies — so a monitor keyed on
    a bare ``run_id`` knows the *expected* work, not just what has landed. ``TO_JSON_STRING`` +
    ``json.loads`` sidesteps the client's native-JSON decoding so the shape is a plain dict. Raises
    `RegistryError` on failure.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    resolved = _resolve_settings(settings)
    sql = (
        f"SELECT TO_JSON_STRING(raw_config) AS raw_config "
        f"FROM `{resolved.table_ref('run_registry')}` "
        "WHERE run_id=@run_id ORDER BY created_at DESC LIMIT 1"
    )
    params = [_header_param("run_id", run_id)]
    client = bigquery.Client(project=resolved.project_id)
    try:
        rows = list(
            client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RegistryError(f"read_run_config failed for run {run_id}: {exc}") from exc
    if not rows or rows[0]["raw_config"] is None:
        return None
    return json.loads(rows[0]["raw_config"])


# The most cells one harvest read will pull back. A profiling run is meant to be *small* — §3.10's
# "an ordinary small run of the same config" — so the cap is not the expected case; it is what keeps
# pointing `source` at a finished 100k run from materialising half a million dicts on a submit host
# that was deliberately given no model stack and not much memory. When it trips, the read is
# truncated and says so, and the resulting profile's signature honestly reports the smaller panel.
_MAX_HARVEST_CELLS = 50_000

# How far back `discover_harvest_run` looks. Two jobs at once: it prunes the `created_at` partition
# scan (the whole table is otherwise in play), and it puts a floor under staleness — evidence older
# than a quarter is evidence about a fleet, a package set and a data volume that have all moved.
_HARVEST_LOOKBACK_DAYS = 90

# The full-fit filter every harvest read shares. Fold rows are bracketed by the full-fit row that
# already covers them and ensemble rows are arithmetic rather than fits, so both would corrupt a
# cost model; `cpu_seconds IS NOT NULL` is what distinguishes a measured cell from one written
# before the columns existed or under `measure="off"`.
_HARVEST_WHERE = (
    "fold_id IS NULL AND ensemble_id IS NULL AND cpu_seconds IS NOT NULL"
)


def read_compute_harvest(
    run_id: str, *, limit: int = _MAX_HARVEST_CELLS, settings: Settings | None = None
) -> tuple[list[dict[str, Any]], str | None] | None:  # pragma: no cover - GCP I/O
    """Return ``(harvest rows, the table the run read)`` for ``run_id``, or ``None`` if unmeasured.

    The read half of "a profile is a query over a `run_id`". Pulls the per-cell measurements
    `worker.run_cell` recorded — `profiling.harvest_profile` aggregates them into the same
    `ComputeProfile` the in-run pre-pass builds, so there is one aggregation and this is only a
    reader. ``None`` (rather than an empty list) means the run has no measurements at all, which is
    what lets the caller fall through to the next source instead of sizing off nothing.

    ``source_table`` comes back alongside because ``forecast_metadata`` does not record one — the
    run header's ``raw_config`` does — and it is the axis `profiling.compare_signatures` cares most
    about: the same fits on a different table are not evidence about this run.

    Writes are append-only and at-least-once, so the rows are deduped to one per cell (latest write
    wins) before returning, the same grain ``v_model_leaderboard`` serves. Ordering is by a
    fingerprint of ``ts_id`` rather than by time or by cost: when ``limit`` truncates, a
    deterministic pseudo-random slice of the panel is a usable sample, while the first N by arrival
    would be whichever bucket happened to finish first. Raises `RegistryError` on failure.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    resolved = _resolve_settings(settings)
    sql = (
        "SELECT ts_id, model_type, fold_id, ensemble_id, fit_seconds, cpu_seconds, "
        "process_rss_bytes, peak_gpu_bytes, intraop_threads, n_obs, created_at "
        f"FROM `{resolved.table_ref('forecast_metadata')}` "
        f"WHERE run_id=@run_id AND {_HARVEST_WHERE} "
        "QUALIFY ROW_NUMBER() OVER ("
        "PARTITION BY run_id, ts_id, model_type, fold_id, ensemble_id "
        "ORDER BY created_at DESC) = 1 "
        "ORDER BY FARM_FINGERPRINT(ts_id) LIMIT @limit"
    )
    params = [
        bigquery.ScalarQueryParameter("run_id", "STRING", run_id),
        bigquery.ScalarQueryParameter("limit", "INT64", limit),
    ]
    client = bigquery.Client(project=resolved.project_id)
    try:
        rows = [
            dict(r)
            for r in client.query(
                sql, job_config=bigquery.QueryJobConfig(query_parameters=params)
            ).result()
        ]
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RegistryError(f"read_compute_harvest failed for run {run_id}: {exc}") from exc
    if not rows:
        return None
    if len(rows) == limit:
        _log.warning(
            "harvest for %s truncated at %d cells; the profile is sized from a sample of the "
            "panel, so its memory bound may under-state the true worst case",
            run_id,
            limit,
        )
    config = read_run_config(run_id, settings=settings) or {}
    return rows, (config.get("data") or {}).get("source_table")


def discover_harvest_run(
    *,
    source_table: str | None,
    freq: str | None,
    lookback_days: int = _HARVEST_LOOKBACK_DAYS,
    settings: Settings | None = None,
) -> str | None:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """The newest completed run carrying a harvest of ``source_table`` at ``freq``, or ``None``.

    What ``compute.profile.source = "auto"`` resolves against. Only the two *identity* axes are
    filtered here — same table, same frequency — because those are the ones where a mismatch means
    "not evidence about this run at all". The scale axes (series count, history length) are checked
    after the rows are loaded, by `profiling.compare_signatures`, where a mismatch is a warning
    rather than a disqualification: a 1k-series run is imperfect but usable evidence for a 100k one,
    and preferring nothing over it would leave the common case unsized.

    Restricted to ``COMPLETED`` runs. A run that died partway measured only the cells that finished,
    which on a bucketed engine is a biased slice, and biased evidence sizes a fleet badly in a
    direction nobody can see. Raises `RegistryError` on failure.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    resolved = _resolve_settings(settings)
    # run_registry is append-only (the header is written, then updated on finalize), so it is
    # deduped to the latest row per run before the join — otherwise a finalized run fans out.
    sql = (
        "WITH headers AS ("
        "  SELECT run_id, raw_config, status FROM `{registry}` "
        "  QUALIFY ROW_NUMBER() OVER (PARTITION BY run_id ORDER BY created_at DESC) = 1"
        ") "
        "SELECT m.run_id AS run_id, MAX(m.created_at) AS measured_at "
        "FROM `{metadata}` AS m JOIN headers AS h USING (run_id) "
        "WHERE m.created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @lookback DAY) "
        "  AND {harvest_where} "
        "  AND h.status = 'COMPLETED' "
        "  AND (@source_table IS NULL OR "
        "       JSON_VALUE(h.raw_config, '$.data.source_table') = @source_table) "
        "  AND (@freq IS NULL OR JSON_VALUE(h.raw_config, '$.data.freq') = @freq) "
        "GROUP BY m.run_id ORDER BY measured_at DESC LIMIT 1"
    ).format(
        registry=resolved.table_ref("run_registry"),
        metadata=resolved.table_ref("forecast_metadata"),
        harvest_where=_HARVEST_WHERE.replace("fold_id", "m.fold_id")
        .replace("ensemble_id", "m.ensemble_id")
        .replace("cpu_seconds", "m.cpu_seconds"),
    )
    params = [
        bigquery.ScalarQueryParameter("lookback", "INT64", lookback_days),
        bigquery.ScalarQueryParameter("source_table", "STRING", source_table),
        bigquery.ScalarQueryParameter("freq", "STRING", freq),
    ]
    client = bigquery.Client(project=resolved.project_id)
    try:
        rows = list(
            client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RegistryError(f"discover_harvest_run failed: {exc}") from exc
    return str(rows[0]["run_id"]) if rows else None


def read_progress(
    run_id: str, *, settings: Settings | None = None
) -> list[dict[str, Any]]:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Return per-model landed-cell counts + mean fit time for ``run_id`` — the live-progress read.

    One row per ``(model_type, ensemble_id)`` with ``n_cells_done`` (deduped full-fit cells that
    have landed) and ``mean_fit_seconds``. Deliberately light (COUNT + AVG, no quantiles) so a
    monitor can poll it cheaply while a run is in flight, at any scale. Progress is coarse-grained:
    cells land when a job's ``write_cells`` runs (often at job end), so counts step up per job
    rather than ticking per series — pair with the per-job status from `read_run_jobs` for the live
    picture. Raises `RegistryError` on failure.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    resolved = _resolve_settings(settings)
    sql = (
        "WITH deduped AS ("
        "  SELECT * FROM `" + resolved.table_ref("forecast_metadata") + "`"
        "  WHERE run_id=@run_id AND fold_id IS NULL"
        "  QUALIFY ROW_NUMBER() OVER ("
        "    PARTITION BY run_id, ts_id, model_type, fold_id, ensemble_id"
        "    ORDER BY created_at DESC) = 1"
        ") "
        "SELECT model_type, ensemble_id, "
        "COUNT(*) AS n_cells_done, AVG(fit_seconds) AS mean_fit_seconds "
        "FROM deduped GROUP BY model_type, ensemble_id"
    )
    params = [_header_param("run_id", run_id)]
    client = bigquery.Client(project=resolved.project_id)
    try:
        rows = list(
            client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RegistryError(f"read_progress failed for run {run_id}: {exc}") from exc
    return [dict(r) for r in rows]


def read_metric_aggregates(
    run_id: str, *, settings: Settings | None = None
) -> list[dict[str, Any]]:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Return the full metric panel aggregated across series, per model — the finished-run stats.

    One row per ``(model_type, ensemble_id)``: ``n_series`` (deduped full-fit cells),
    ``mean_fit_seconds``, and for **every** metric in `METRIC_COLUMNS` its cross-series ``mean_``
    plus ``p10_``/``p50_``/``p90_`` (via ``APPROX_QUANTILES``). The aggregation runs server-side so
    it holds at 100k+ series without pulling per-series rows to the client — the review layer reads
    distribution shape from here, falling back to `read_cell_metrics` only for a bounded drill-down.
    The projection is generated from `METRIC_COLUMNS`, so it tracks the one metric source of truth.
    Raises `RegistryError` on failure.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    resolved = _resolve_settings(settings)
    metric_aggs = ", ".join(
        f"AVG({m}) AS mean_{m}, "
        f"APPROX_QUANTILES({m}, 10)[OFFSET(1)] AS p10_{m}, "
        f"APPROX_QUANTILES({m}, 10)[OFFSET(5)] AS p50_{m}, "
        f"APPROX_QUANTILES({m}, 10)[OFFSET(9)] AS p90_{m}"
        for m in METRIC_COLUMNS
    )
    sql = (
        "WITH deduped AS ("
        "  SELECT * FROM `" + resolved.table_ref("forecast_metadata") + "`"
        "  WHERE run_id=@run_id AND fold_id IS NULL"
        "  QUALIFY ROW_NUMBER() OVER ("
        "    PARTITION BY run_id, ts_id, model_type, fold_id, ensemble_id"
        "    ORDER BY created_at DESC) = 1"
        ") "
        "SELECT model_type, ensemble_id, ANY_VALUE(compute_engine) AS compute_engine, "
        "COUNT(*) AS n_series, AVG(fit_seconds) AS mean_fit_seconds, " + metric_aggs + " "
        "FROM deduped GROUP BY model_type, ensemble_id"
    )
    params = [_header_param("run_id", run_id)]
    client = bigquery.Client(project=resolved.project_id)
    try:
        rows = list(
            client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RegistryError(f"read_metric_aggregates failed for run {run_id}: {exc}") from exc
    return [dict(r) for r in rows]


def read_cell_metrics(
    run_id: str, *, limit: int = 20000, settings: Settings | None = None
) -> list[dict[str, Any]]:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Return per-series metric rows for ``run_id`` — the bounded drill-down under the aggregates.

    One deduped full-fit row per ``(ts_id, model_type, ensemble_id)`` carrying ``compute_engine`` /
    ``fit_seconds`` and the whole `METRIC_COLUMNS` panel — the raw material for per-series
    distribution plots and outlier hunts. Capped at ``limit`` rows (a 100k×N run would swamp the
    client and any plot); use `read_metric_aggregates` for the whole-run distribution shape and this
    for a bounded sample or a small run. Raises `RegistryError` on failure.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    resolved = _resolve_settings(settings)
    metric_cols = ", ".join(METRIC_COLUMNS)
    sql = (
        "SELECT ts_id, model_type, ensemble_id, compute_engine, fit_seconds, " + metric_cols + " "
        "FROM `" + resolved.table_ref("forecast_metadata") + "` "
        "WHERE run_id=@run_id AND fold_id IS NULL "
        "QUALIFY ROW_NUMBER() OVER ("
        "PARTITION BY run_id, ts_id, model_type, fold_id, ensemble_id "
        "ORDER BY created_at DESC) = 1 "
        "LIMIT @limit"
    )
    params = [
        bigquery.ScalarQueryParameter("run_id", "STRING", run_id),
        bigquery.ScalarQueryParameter("limit", "INT64", limit),
    ]
    client = bigquery.Client(project=resolved.project_id)
    try:
        rows = list(
            client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise RegistryError(f"read_cell_metrics failed for run {run_id}: {exc}") from exc
    return [dict(r) for r in rows]


class HeaderFinalizer:
    """Mutable finalize state a `run_header` body fills in before a clean exit.

    A run's terminal ``status`` (default ``COMPLETED``) plus any extra header columns to stamp on
    success (``n_series``, ``n_models``, ``bq_models``, …). Left untouched, the block finalizes a
    plain ``COMPLETED`` with only the wall-clock ``runtime_seconds`` `run_header` measures.
    """

    def __init__(self) -> None:
        self.status: str = "COMPLETED"
        self.extra: dict[str, Any] = {}

    def finalize(self, *, status: str | None = None, **fields: Any) -> None:
        """Set the terminal ``status`` (if given) and merge extra columns for the success write."""
        if status is not None:
            self.status = status
        self.extra.update(fields)


@contextmanager
def run_header(
    cfg: RunConfig,
    run_id: str,
    *,
    settings: Settings | None = None,
    manage: bool = True,
) -> Iterator[HeaderFinalizer]:
    """Own a run's ``run_registry`` header for the duration of a block (the one lifecycle seam).

    In **owner mode** (``manage=True``): on entry ``ensure_tables`` + ``write_header`` (RUNNING);
    on a clean exit ``update_header`` with the finalizer's ``status`` (default COMPLETED), the
    measured wall-clock ``runtime_seconds``, and any extra columns the body set via
    `HeaderFinalizer.finalize`; on an exception ``update_header(status=FAILED, runtime_seconds=…)``
    then re-raise, so a crashed run records a terminal status instead of stranding at RUNNING.

    In **contributor mode** (``manage=False``): touches no header at all — `main.run` owns the
    single shared row — so this only yields the finalizer for uniform call shape. The body may
    still populate it; nothing is written.
    """
    fin = HeaderFinalizer()
    if manage:
        ensure_tables(cfg, settings=settings)
        write_header(cfg, run_id, settings=settings)
    started = time.perf_counter()
    try:
        yield fin
    except BaseException:
        if manage:
            update_header(
                run_id,
                settings=settings,
                status="FAILED",
                runtime_seconds=time.perf_counter() - started,
            )
        raise
    if manage:
        update_header(
            run_id,
            settings=settings,
            status=fin.status,
            runtime_seconds=time.perf_counter() - started,
            **fin.extra,
        )


class JobFinalizer:
    """Mutable finalize state a `run_job` body fills in before a clean exit.

    A job's terminal ``status`` (default ``COMPLETED``) plus any extra ``run_jobs`` columns to stamp
    on success — notably ``system_job_id`` (once the platform assigns/accepts it) and
    ``job_telemetry`` (the per-job sizing/wall/DCU overlay). Left untouched, the block finalizes a
    plain ``COMPLETED`` with only the wall-clock ``runtime_seconds`` `run_job` measures.
    """

    def __init__(self) -> None:
        self.status: str = "COMPLETED"
        self.extra: dict[str, Any] = {}

    def finalize(self, *, status: str | None = None, **fields: Any) -> None:
        """Set the terminal ``status`` (if given) and merge extra columns for the success write."""
        if status is not None:
            self.status = status
        self.extra.update(fields)


@contextmanager
def run_job(
    run_id: str,
    family: str,
    attempt: int,
    *,
    runtime: str | None = None,
    spark_mode: str | None = None,
    hardware: str | None = None,
    gpu_type: str | None = None,
    system_job_id: str | None = None,
    probe_handle: dict[str, Any] | None = None,
    settings: Settings | None = None,
    manage: bool = True,
) -> Iterator[JobFinalizer]:
    """Own one family's ``run_jobs`` row for the duration of a block (the per-job lifecycle seam).

    The `run_header` analog for the per-job tier: on entry write the job row (RUNNING) with its
    deterministic id (`assemble_job_row` → `registry.ids.make_job_key`) and resolved compute; on a
    clean exit ``update_job`` with the finalizer's ``status`` (default COMPLETED), the measured
    wall-clock ``runtime_seconds``, and any extra columns set via `JobFinalizer.finalize` (e.g. the
    platform ``system_job_id`` and ``job_telemetry``); on an exception ``update_job(status=FAILED,
    runtime_seconds=…)`` then re-raise, so a crashed job records a terminal status instead of
    stranding at RUNNING. The run header is owned separately by `run_header`; a job row sits *under*
    it. ``manage=False`` yields the finalizer without touching ``run_jobs`` (uniform call shape for
    a caller that records the job elsewhere). Assumes the tables exist (the header owner ran
    `ensure_tables`), so it does not re-create them.
    """
    from .ids import make_job_key

    fin = JobFinalizer()
    job_id = make_job_key(run_id, family, attempt)
    if manage:
        from datetime import UTC, datetime

        row = assemble_job_row(
            run_id,
            family,
            attempt,
            datetime.now(UTC),
            runtime=runtime,
            spark_mode=spark_mode,
            hardware=hardware,
            gpu_type=gpu_type,
            system_job_id=system_job_id,
            probe_handle=probe_handle,
        )
        write_job(row, settings=settings)
    started = time.perf_counter()
    try:
        yield fin
    except BaseException:
        if manage:
            update_job(
                job_id,
                settings=settings,
                status="FAILED",
                runtime_seconds=time.perf_counter() - started,
                ended_at=datetime.now(UTC),
            )
        raise
    if manage:
        update_job(
            job_id,
            settings=settings,
            status=fin.status,
            runtime_seconds=time.perf_counter() - started,
            ended_at=datetime.now(UTC),
            **fin.extra,
        )


def write_cells(
    results: list[CellResult], *, settings: Settings | None = None
) -> None:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Persist a run's cell results to the three cell tables.

    Idempotency is **append-only + dedupe-on-read** (see `cell_dedup_key`): this appends
    and never DELETEs. A DELETE that matches rows still in the Storage Write API streaming buffer
    is rejected for the whole buffer window (~90 min), so a clear-then-append against the default
    stream is not viable. Instead, ``run_id`` is a pure function of the
    config, so a re-run of the same config writes byte-identical rows; serving views dedupe on
    ``run_id`` (+ cell keys). Steps:

    1. Assemble rows via the pure assemblers; upload each cell's serialized model bytes (if any,
       when the run set ``persist_models``) and stamp the returned URI onto its
       ``forecast_metadata`` row.
    2. **Append** each table's rows via the Storage Write API default stream.

    ``write_cells`` may be called once per run (driver-side collect) or many times per run (per
    Spark/Ray partition) — appends compose, so both are safe. Empty input is a no-op. Raises
    `RegistryError` on any BigQuery/GCS failure.
    """
    from datetime import UTC, datetime

    from google.cloud import bigquery_storage_v1

    from ..errors import RegistryError
    from . import artifacts

    if not results:
        return
    run_ids = {r.run_id for r in results}
    if len(run_ids) != 1:
        raise RegistryError(f"write_cells expects one run_id per call, got {sorted(run_ids)}")

    resolved = _resolve_settings(settings)
    created_at = datetime.now(UTC)

    # 1. Assemble rows. Upload artifacts first so the metadata row carries the URI.
    pred_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    for result in results:
        pred_rows.extend(assemble_prediction_rows(result))
        oof_rows.extend(assemble_oof_rows(result))
        model_artifact: str | None = None
        if result.artifact_bytes is not None:
            model_artifact = artifacts.upload_artifact_bytes(
                result.artifact_bytes,
                f"{result.model_hash}.pkl",
                result.run_id,
                resolved.warehouse_uri,
            )
        meta_rows.append(assemble_metadata_row(result, created_at, model_artifact=model_artifact))

    rows_by_table: dict[str, list[dict[str, Any]]] = {
        "forecast_predictions": pred_rows,
        "backtest_oof": oof_rows,
        "forecast_metadata": meta_rows,
    }
    spec_by_table: dict[str, tuple[tuple[str, str], ...]] = {
        "forecast_predictions": _PRED_SPEC,
        "backtest_oof": _OOF_SPEC,
        "forecast_metadata": _META_SPEC,
    }

    # 2. Append via the Storage Write API (no DELETE — append-only, dedupe-on-read).
    write_client = bigquery_storage_v1.BigQueryWriteClient()
    for table in _CELL_TABLES:
        rows = rows_by_table[table]
        if not rows:
            continue
        msg_cls, proto_descriptor = _proto_for(table, spec_by_table[table])
        serialized = _encode_rows(msg_cls, spec_by_table[table], rows)
        _append_via_write_api(
            write_client,
            resolved.project_id,
            resolved.dataset_id,
            table,
            proto_descriptor,
            serialized,
        )
