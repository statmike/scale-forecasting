"""Storage Write API transport — protobuf encoding, chunking and retry for the cell tables.

The high-volume write path. Rows arrive already assembled (`registry.rows`); this turns them into
protobuf, splits them into requests that fit the API's byte and row ceilings, and appends them
under a bounded retry. `registry.cells` is the only caller — the three cell tables are the only
writes big enough to be worth the Write API rather than a parameterized INSERT.
"""

from __future__ import annotations

import time
from typing import Any

from ..errors import get_logger
from .rows import METRIC_COLUMNS

_log = get_logger(__name__)


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
    # as a `ComputeProfile` that sizes a later one — see profiling.cost.harvest_profile.
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
# double-counts. Naming mirrors ray_jobs.py's manual-retry idiom.
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
                raise RegistryError(f"Storage Write API append to {table} failed: {exc}") from exc
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
