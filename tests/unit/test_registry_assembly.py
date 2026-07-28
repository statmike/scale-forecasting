"""Tests for the pure registry row-assembly layer (CONTRACTS §3.4, §4, BUILD 1.4).

Offline only: a :class:`CellResult` (plus a :class:`RunConfig` for the header) maps to the
exact row dicts each table expects. No BigQuery client — the I/O writers are Arc B (B1).
Covers frame→rows mapping, run/series/model/engine stamping, dtype/NaN coercion, JSON
serialization, and the model_hash idempotency key.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from scale_forecasting.config import RunConfig
from scale_forecasting.registry import artifacts, bq
from scale_forecasting.worker import CellResult

_CREATED = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def _cfg(**over: Any) -> RunConfig:
    base: dict[str, Any] = {
        "run_name": "my run",
        "data": {"source_table": "p.d.source_series", "series_limit": 5},
        "models": ["theta", "sarimax"],
    }
    base.update(over)
    return RunConfig(**base)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ds": pd.to_datetime(["2026-02-01", "2026-02-02"]),
            "yhat": [10.0, 11.5],
            "yhat_lower": [8.0, 9.0],
            "yhat_upper": [12.0, 14.0],
            "quantiles": [{"0.5": 10.0}, None],
        }
    )


def _oof() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold_id": [0, 0, 1],
            "ds": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-15"]),
            "y_true": [9.0, 10.0, 11.0],
            "yhat": [8.5, 10.5, 10.0],
        }
    )


def _result(**over: Any) -> CellResult:
    base: dict[str, Any] = {
        "run_id": "my-run-abc123def456",
        "ts_id": "series-7",
        "model_type": "theta",
        "compute_engine": "spark",
        "model_hash": "f" * 64,
        "status": "ok",
        "error": None,
        "predictions": _predictions(),
        "oof": _oof(),
        "metrics": {"wape": 0.12, "mae": 3.4},
        "best_params": {"alpha": 0.5},
        "fit_seconds": 1.25,
    }
    base.update(over)
    return CellResult(**base)


# --- prediction rows -----------------------------------------------------------


def test_prediction_rows_map_and_stamp() -> None:
    rows = bq.assemble_prediction_rows(_result())
    assert len(rows) == 2
    first = rows[0]
    # stamped identity on every row
    assert first["run_id"] == "my-run-abc123def456"
    assert first["ts_id"] == "series-7"
    assert first["model_type"] == "theta"
    assert first["compute_engine"] == "spark"
    # ds -> forecast_date, as a date (not timestamp)
    assert first["forecast_date"].isoformat() == "2026-02-01"
    assert first["yhat"] == 10.0


def test_prediction_quantiles_serialized_to_json_or_none() -> None:
    rows = bq.assemble_prediction_rows(_result())
    assert rows[0]["quantiles"] == '{"0.5": 10.0}'
    assert rows[1]["quantiles"] is None


# --- oof rows ------------------------------------------------------------------


def test_oof_rows_carry_fold_and_truth() -> None:
    rows = bq.assemble_oof_rows(_result())
    assert len(rows) == 3
    assert [r["fold_id"] for r in rows] == [0, 0, 1]
    assert all(isinstance(r["fold_id"], int) for r in rows)
    assert rows[0]["y_true"] == 9.0
    assert rows[0]["forecast_date"].isoformat() == "2026-01-01"


def test_oof_rows_empty_when_no_backtest() -> None:
    assert bq.assemble_oof_rows(_result(oof=None)) == []


# --- metadata row --------------------------------------------------------------


def test_metadata_row_has_full_metric_panel() -> None:
    row = bq.assemble_metadata_row(_result(), _CREATED)
    # every declared metric column is present...
    for name in bq.METRIC_COLUMNS:
        assert name in row
    # ...populated where provided, NULL (None) where absent
    assert row["wape"] == 0.12
    assert row["mae"] == 3.4
    assert row["rmse"] is None
    assert row["fold_id"] is None  # full-fit summary row
    assert row["best_params"] == '{"alpha": 0.5}'
    assert row["fit_seconds"] == 1.25
    assert row["created_at"] is _CREATED
    assert row["model_hash"] == "f" * 64


def test_metadata_row_carries_artifact_link() -> None:
    row = bq.assemble_metadata_row(_result(), _CREATED, model_artifact="gs://wh/artifacts/x/m.pkl")
    assert row["model_artifact"] == "gs://wh/artifacts/x/m.pkl"


def test_metadata_metric_columns_match_ddl() -> None:
    # The assembled metric keys must be exactly the metric columns in the DDL (§4).
    ddl_metrics = {
        "mae",
        "rmse",
        "mse",
        "mape",
        "smape",
        "wape",
        "mase",
        "rmsse",
        "bias",
        "coverage",
        "pinball",
    }
    assert set(bq.METRIC_COLUMNS) == ddl_metrics


# --- coercion ------------------------------------------------------------------


def test_nan_metric_coerced_to_none() -> None:
    row = bq.assemble_metadata_row(_result(metrics={"wape": float("nan")}), _CREATED)
    assert row["wape"] is None


def test_empty_best_params_serialized_to_none() -> None:
    row = bq.assemble_metadata_row(_result(best_params={}), _CREATED)
    assert row["best_params"] is None


# --- idempotency key -----------------------------------------------------------


def test_dedup_key_is_run_scoped() -> None:
    # Idempotency is enforced at the run grain (run-level clear + append), so the key is
    # run_id alone — not model_hash (B0.3: can't DELETE the streaming buffer per-cell).
    key = bq.cell_dedup_key(_result())
    assert key == {"run_id": "my-run-abc123def456"}


# --- header row ----------------------------------------------------------------


def test_header_row_snapshots_config() -> None:
    cfg = _cfg()
    row = bq.assemble_header_row(cfg, "my-run-abc123def456", _CREATED)
    assert row["run_id"] == "my-run-abc123def456"
    assert row["status"] == "RUNNING"
    assert row["python_runtime"] == "spark"
    assert row["spark_method"] == "explode"  # normalized default
    assert row["n_models"] == 2
    assert row["n_series"] == 5
    assert row["backtest_on"] is False
    assert row["created_at"] is _CREATED
    # raw_config is the verbatim validated config as a JSON string
    assert isinstance(row["raw_config"], str)
    assert '"run_name": "my run"' in row["raw_config"]
    assert '"models": ["theta", "sarimax"]' in row["raw_config"]


def test_header_ensemble_strategies_empty_when_disabled() -> None:
    row = bq.assemble_header_row(_cfg(), "rid", _CREATED)
    assert row["ensemble_strategies"] == []


def test_header_ensemble_strategies_listed_when_enabled() -> None:
    cfg = _cfg(ensemble={"enabled": True, "strategies": ["mean", "median"]})
    row = bq.assemble_header_row(cfg, "rid", _CREATED)
    assert row["ensemble_strategies"] == ["mean", "median"]


# --- artifact uri --------------------------------------------------------------


def test_artifact_uri_is_run_scoped_and_deterministic() -> None:
    uri = artifacts.artifact_gcs_uri("/tmp/model.pkl", "my-run-abc123", "gs://bucket/warehouse/")
    assert uri == "gs://bucket/warehouse/artifacts/my-run-abc123/model.pkl"
    # deterministic
    assert uri == artifacts.artifact_gcs_uri(
        "/tmp/model.pkl", "my-run-abc123", "gs://bucket/warehouse"
    )
