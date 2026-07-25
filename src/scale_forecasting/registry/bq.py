"""Registry writers — the bulk BigQuery lineage layer (CONTRACTS §3.4, §4).

Split into two halves along the pure/I-O seam (CONTRACTS §0):

- **Pure row assembly** (tested offline now, BUILD 1.4): turn a :class:`CellResult` and
  a :class:`RunConfig` into the exact ``list[dict]`` rows each table expects — column
  mapping, stamping ``run_id``/``ts_id``/``model_type``/``compute_engine``, JSON
  serialization, and the per-cell idempotency key.
- **I/O** (structured now, GCP-verified in Arc B step B1): ``ensure_tables``,
  ``write_header``, ``update_header``, ``write_cells`` — execute DDL and stream rows via
  the Storage Write API, idempotent per ``model_hash``.

Public surface: ``ensure_tables``, ``write_header``, ``update_header``, ``write_cells``,
plus the pure assemblers used by the writers and the tests.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, get_args

from ..config import DecisionMetric

if TYPE_CHECKING:
    from datetime import datetime

    from ..config import RunConfig
    from ..worker import CellResult

# The full metric panel, in table-column order — derived from the config's DecisionMetric
# literal so there is exactly one source of truth (CONTRACTS §2.3 / DESIGN §5.1).
METRIC_COLUMNS: tuple[str, ...] = get_args(DecisionMetric)


# --- pure row assembly ---------------------------------------------------------


def cell_dedup_key(result: CellResult) -> dict[str, str]:
    """The idempotency key for one cell (CONTRACTS §3.4).

    Re-running a cell must overwrite, not duplicate. ``model_hash`` already encodes
    ``(run_id, ts_id, model_type, config)`` deterministically, so the writer deletes
    rows matching this key before inserting.
    """
    return {"run_id": result.run_id, "model_hash": result.model_hash}


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


# --- I/O (structured now; GCP-verified in Arc B step B1) -----------------------


def ensure_tables(cfg: RunConfig) -> None:  # pragma: no cover - Arc B (B1)
    raise NotImplementedError("registry.bq.ensure_tables — BUILD step B1")


def write_header(cfg: RunConfig, run_id: str) -> None:  # pragma: no cover - Arc B (B1)
    raise NotImplementedError("registry.bq.write_header — BUILD step B1")


def update_header(run_id: str, **fields: Any) -> None:  # pragma: no cover - Arc B (B1)
    raise NotImplementedError("registry.bq.update_header — BUILD step B1")


def write_cells(results: list[CellResult]) -> None:  # pragma: no cover - Arc B (B1)
    raise NotImplementedError("registry.bq.write_cells — BUILD step B1")
