"""The three cell tables — the bulk per-(series, model) write.

Idempotency is **append-only + dedupe-on-read**: this only appends (never DELETEs — a DELETE
against rows still in the ~90-min streaming buffer is rejected), and the serving views dedupe with
``DISTINCT``/``GROUP BY`` on ``run_id`` (+ cell keys). Assembly is `registry.rows`, transport is
`registry.write_api`; what is here is the per-table orchestration between them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .rows import assemble_metadata_row, assemble_oof_rows, assemble_prediction_rows
from .tables import _resolve_settings
from .write_api import (
    _CELL_TABLES,
    _META_SPEC,
    _OOF_SPEC,
    _PRED_SPEC,
    _append_via_write_api,
    _encode_rows,
    _proto_for,
)

if TYPE_CHECKING:
    from ..settings import Settings
    from ..worker import CellResult


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
                resolved.artifact_root,
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
            resolved.registry_dataset_id,
            table,
            proto_descriptor,
            serialized,
        )
