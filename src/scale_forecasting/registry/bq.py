"""Registry writers — the bulk BigQuery lineage layer (CONTRACTS §3.4, §4).

Split into two halves along the pure/I-O seam (CONTRACTS §0):

- **Pure row assembly** (tested offline now, BUILD 1.4): turn a :class:`CellResult` and
  a :class:`RunConfig` into the exact ``list[dict]`` rows each table expects — column
  mapping, stamping ``run_id``/``ts_id``/``model_type``/``compute_engine``, JSON
  serialization, and the per-cell idempotency key.
- **I/O** (implemented in Arc B step B1, GCP-verified by the ``@gcp`` round-trip test):
  ``ensure_tables``, ``write_header``, ``update_header``, ``write_cells`` — execute DDL,
  INSERT/UPDATE the single-row header, and stream the three cell tables via the Storage
  Write API. Idempotency is **append-only + dedupe-on-read**: ``write_cells`` only appends
  (never DELETEs — a DELETE against rows still in the ~90-min streaming buffer is rejected),
  and serving views dedupe with ``DISTINCT``/``GROUP BY`` on ``run_id`` (+ cell keys).

The infra identity (project / dataset / connection / warehouse) is not on ``RunConfig`` —
it is resolved from the environment via :class:`~scale_forecasting.settings.Settings`, so the
identical writer code runs locally under ADC and on Composer under the runner SA (G1). Each
writer accepts an optional ``settings=`` for tests/callers that already hold one; otherwise it
resolves from ``SF_*`` env vars. GCP client libraries are imported lazily inside the writers so
the pure layer (and its offline tests) never need them installed.

Public surface: ``ensure_tables``, ``ensure_views``, ``write_header``, ``update_header``,
``write_cells``, plus the pure assemblers used by the writers and the tests.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, get_args

from ..config import DecisionMetric

if TYPE_CHECKING:
    from datetime import datetime

    from ..config import RunConfig
    from ..settings import Settings
    from ..worker import CellResult

# The full metric panel, in table-column order — derived from the config's DecisionMetric
# literal so there is exactly one source of truth (CONTRACTS §2.3 / DESIGN §5.1).
METRIC_COLUMNS: tuple[str, ...] = get_args(DecisionMetric)


# --- pure row assembly ---------------------------------------------------------


def cell_dedup_key(result: CellResult) -> dict[str, str]:
    """The run-scoped identity anchor for a cell's rows (CONTRACTS §3.4).

    Idempotency is **append-only + dedupe-on-read**, anchored on ``run_id``. :func:`write_cells`
    never DELETEs — a DELETE that matches rows still in the Storage Write API streaming buffer
    is rejected for the whole buffer window (~90 min), so a clear-then-append is not viable
    against the default stream (B1 live-gate finding, foreshadowed by B0.3). Instead we rely on
    ``run_id`` being a pure function of the config (``make_run_id``): the same ``run_id`` implies
    the same config implies byte-identical rows, so a re-run's "duplicates" are exact copies.
    Serving views dedupe with ``DISTINCT``/``GROUP BY`` on ``run_id`` (+ cell keys); no write-time
    delete is needed. ``model_hash`` uniquely identifies the cell on ``forecast_metadata`` for
    lineage.
    """
    return {"run_id": result.run_id}


def assemble_prediction_rows(result: CellResult) -> list[dict[str, Any]]:
    """Canonical prediction frame (§2.1) → ``forecast_predictions`` rows (§4).

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
    """Canonical OOF frame (§2.2) → ``backtest_oof`` rows (§4). Empty if no backtest."""
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
    """One full-fit ``forecast_metadata`` row (§4): metrics panel + artifact link.

    ``fold_id`` is None (this is the full-fit summary row). ``model_artifact`` is the
    ObjectRef/URI filled in by the writer after the artifact upload.
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
    }
    for name in METRIC_COLUMNS:
        row[name] = _as_float(result.metrics.get(name))
    return row


def assemble_header_row(cfg: RunConfig, run_id: str, created_at: datetime) -> dict[str, Any]:
    """Build the ``run_registry`` header row from a config (§4, §8.2).

    ``raw_config`` is the validated config serialized verbatim — the config *is* the
    record (G3). ``bq_models`` is left empty here and filled by the router once model
    runtimes are known (Arc B); status starts RUNNING.
    """
    return {
        "run_id": run_id,
        "created_at": created_at,
        "user_id": None,
        "git_sha": None,
        "python_runtime": cfg.python_runtime,
        "spark_method": cfg.spark_method,
        "bq_models": [],
        "backtest_on": cfg.backtest.enabled,
        "decision_metric": cfg.backtest.decision_metric,
        "ensemble_strategies": list(cfg.ensemble.strategies) if cfg.ensemble.enabled else [],
        "raw_config": json.dumps(cfg.model_dump(mode="json"), sort_keys=True),
        "status": "RUNNING",
        "n_series": cfg.data.series_limit,
        "n_models": len(cfg.models),
        "runtime_seconds": None,
        # Dataproc-level job telemetry (executor sizing, wall/startup split, DCU usage): a JSON
        # STRING filled in after the batch finishes by the submitter (extract_job_telemetry →
        # update_header). NULL here at RUNNING and for any run whose telemetry couldn't be read
        # (best-effort — never blocks a run). See run_registry DDL / DESIGN §8.2.
        "job_telemetry": None,
    }


# --- small pure coercers -------------------------------------------------------


def _as_float(value: Any) -> float | None:
    """Coerce to float, mapping missing/NaN to None (BQ NULL)."""
    if value is None:
        return None
    f = float(value)
    return None if f != f else f  # NaN check


def _as_json(value: Any) -> str | None:
    """Serialize a dict (or None/empty) to a JSON string, or None."""
    if value is None or (isinstance(value, dict) and not value):
        return None
    if isinstance(value, str):
        return value
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
# proto scalar type. These specs mirror the DDL (§4) and the keys the pure assemblers emit,
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
    ("fold_id", "I"),
    *((name, "D") for name in METRIC_COLUMNS),
    ("fit_seconds", "D"),
    ("best_params", "S"),
    ("model_artifact", "S"),
    ("created_at", "S"),
)

# Which assembler feeds which table, and its column spec. Driven in this one place so
# write_cells stays a loop, not three near-identical blocks.
_CELL_TABLES: tuple[str, ...] = ("forecast_predictions", "backtest_oof", "forecast_metadata")

# Write API request limits: batch serialized rows under ~10MB and a sane row cap per request.
_MAX_REQUEST_BYTES = 9 * 1024 * 1024
_MAX_REQUEST_ROWS = 10_000


def _proto_for(table_name: str, spec: tuple[tuple[str, str], ...]) -> tuple[Any, Any]:
    """Build a protobuf message class + descriptor matching a table's column spec.

    Fields are proto2-optional (so an unset field → BigQuery NULL) and numbered by write
    order. A private :class:`DescriptorPool` isolates the registration so repeated calls (or
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
    wrapper, which masks the underlying gRPC error — B0.3): the first request carries the
    stream + writer schema, each subsequent request carries only rows. Any failure is
    re-raised as :class:`RegistryError` with the real error attached.
    """
    from google.api_core.exceptions import GoogleAPICallError
    from google.cloud.bigquery_storage_v1 import types

    from ..errors import RegistryError

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

    try:
        for response in write_client.append_rows(requests=requests()):
            if response.error.code != 0:
                raise RegistryError(
                    f"Storage Write API append to {table} failed: "
                    f"{response.error.code} {response.error.message}"
                )
    except GoogleAPICallError as exc:
        raise RegistryError(f"Storage Write API append to {table} failed: {exc}") from exc


# --- I/O: the four registry writers --------------------------------------------


def _resolve_settings(settings: Settings | None) -> Settings:
    """Return the passed settings, or resolve from the ``SF_*`` environment (G1)."""
    if settings is not None:
        return settings
    from ..settings import Settings as _Settings

    return _Settings.resolve()


def ensure_tables(
    cfg: RunConfig | None = None, *, settings: Settings | None = None
) -> None:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Create every registry + source table if absent (idempotent DDL, §4).

    Renders the managed-Iceberg DDL for the resolved dataset and executes each statement.
    ``cfg`` is accepted for signature symmetry with the other writers but is unused — the
    schema is fixed, not config-driven. Raises :class:`RegistryError` on a DDL failure.
    """
    from google.cloud import bigquery

    from ..errors import RegistryError
    from .ddl import render_create_tables

    resolved = _resolve_settings(settings)
    ddl = render_create_tables(
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

    # Curated analyst views sit on top of the tables — create them in the same setup pass so the
    # reviewable read surface (v_run_summary / v_model_leaderboard) exists after any run.
    ensure_views(settings=resolved)


def ensure_views(
    *, settings: Settings | None = None
) -> None:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Create/replace the analyst views over the registry (idempotent, §4).

    Renders the ``CREATE OR REPLACE VIEW`` statements (:func:`registry.views.render_create_views`)
    for the resolved dataset and executes each. Called by :func:`ensure_tables`; safe to call on its
    own to refresh view definitions after a change. Raises :class:`RegistryError` on failure.
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


# run_registry columns that may be set by write_header / update_header, with their BQ types.
_HEADER_PARAM_TYPES: dict[str, str] = {
    "run_id": "STRING",
    "created_at": "TIMESTAMP",
    "user_id": "STRING",
    "git_sha": "STRING",
    "python_runtime": "STRING",
    "spark_method": "STRING",
    "bq_models": "ARRAY<STRING>",
    "backtest_on": "BOOL",
    "decision_metric": "STRING",
    "ensemble_strategies": "ARRAY<STRING>",
    "raw_config": "STRING",
    "status": "STRING",
    "n_series": "INT64",
    "n_models": "INT64",
    "runtime_seconds": "FLOAT64",
    "job_telemetry": "STRING",
}


def _header_param(name: str, value: Any) -> Any:
    """Build a scalar or array query parameter for a run_registry column."""
    from google.cloud import bigquery

    bq_type = _HEADER_PARAM_TYPES[name]
    if bq_type.startswith("ARRAY<"):
        element_type = bq_type[len("ARRAY<") : -1]
        return bigquery.ArrayQueryParameter(name, element_type, list(value or []))
    return bigquery.ScalarQueryParameter(name, bq_type, value)


def write_header(
    cfg: RunConfig, run_id: str, *, settings: Settings | None = None
) -> None:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Insert the run's ``run_registry`` header row (status RUNNING) from its config (§8.2).

    A single-row parameterized INSERT (not the Write API — no benefit for one row, and the
    header is updated in place later by :func:`update_header`). Raises :class:`RegistryError`
    on failure.
    """
    from datetime import UTC, datetime

    from google.cloud import bigquery

    from ..errors import RegistryError

    resolved = _resolve_settings(settings)
    row = assemble_header_row(cfg, run_id, datetime.now(UTC))
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
    """Update named columns on a run's header row (§8.2), e.g. status/runtime_seconds.

    ``update_header(run_id, status="COMPLETED", runtime_seconds=42.0)`` → a parameterized
    ``UPDATE … SET … WHERE run_id=@run_id``. Unknown column names raise :class:`RegistryError`;
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


def write_cells(
    results: list[CellResult], *, settings: Settings | None = None
) -> None:  # pragma: no cover - GCP I/O, covered by the @gcp round-trip test
    """Persist a run's cell results to the three cell tables (§3.4).

    Idempotency is **append-only + dedupe-on-read** (see :func:`cell_dedup_key`): this appends
    and never DELETEs. A DELETE that matches rows still in the Storage Write API streaming buffer
    is rejected for the whole buffer window (~90 min), so a clear-then-append against the default
    stream is not viable (B1 live-gate finding). Instead, ``run_id`` is a pure function of the
    config, so a re-run of the same config writes byte-identical rows; serving views dedupe on
    ``run_id`` (+ cell keys). Steps:

    1. Assemble rows via the pure assemblers; upload each cell's serialized model bytes (if any,
       when the run set ``persist_models``) and stamp the returned URI onto its
       ``forecast_metadata`` row.
    2. **Append** each table's rows via the Storage Write API default stream.

    ``write_cells`` may be called once per run (driver-side collect) or many times per run (per
    Spark/Ray partition) — appends compose, so both are safe. Empty input is a no-op. Raises
    :class:`RegistryError` on any BigQuery/GCS failure.
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
