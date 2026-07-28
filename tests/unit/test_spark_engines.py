"""Offline tests for the pure Spark-engine helpers (BUILD B2, ``engines.spark_io``).

Everything here runs without Spark or BigQuery — it exercises the grouped-UDF body
(:func:`run_group`), the run-level status roll-up (:func:`aggregate_status`), and the bucketing
policy that is the crux of the explode-vs-naive scaling story. The Spark shell (connector read,
cross-join, applyInPandas) is covered by the ``@spark``/``@gcp`` gates in B2.2.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from scale_forecasting.config import RunConfig
from scale_forecasting.engines import spark_io
from scale_forecasting.engines.spark_io import (
    STATUS_COLUMNS,
    aggregate_status,
    bucket_key_cols,
    default_bucket_count,
    run_group,
)

HORIZON = 7


def _series(ts_id: str, n: int = 90) -> pd.DataFrame:
    """One ts_id's rows: deterministic trend + weekly seasonality, columns [ts_id, ds, y]."""
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    trend = np.linspace(10.0, 30.0, n)
    weekly = 3.0 * np.sin(np.arange(n) * 2 * np.pi / 7)
    return pd.DataFrame({"ts_id": ts_id, "ds": idx, "y": trend + weekly})


def _panel(ids: list[str]) -> pd.DataFrame:
    return pd.concat([_series(i) for i in ids], ignore_index=True)


def _cfg(**over: Any) -> RunConfig:
    base: dict[str, Any] = {
        "run_name": "spark io test",
        "data": {"source_table": "t", "freq": "D", "horizon": HORIZON},
        "models": ["theta"],
    }
    base.update(over)
    return RunConfig(**base)


# --- bucket_key_cols: the explode-vs-naive crux --------------------------------


def test_explode_buckets_on_cell_naive_on_series() -> None:
    explode = _cfg(spark_method="explode", models=["theta", "holtwinters"])
    naive = _cfg(spark_method="naive", models=["theta", "holtwinters"])
    # explode isolates each (series, model) cell; naive keeps a whole series in one task.
    assert bucket_key_cols(explode) == ["ts_id", spark_io._MODEL_COL]
    assert bucket_key_cols(naive) == ["ts_id"]


def test_bucket_key_honors_custom_ts_id_col() -> None:
    cfg = _cfg(data={"source_table": "t", "ts_id_col": "series_key"}, spark_method="naive")
    assert bucket_key_cols(cfg) == ["series_key"]


# --- default_bucket_count ------------------------------------------------------


def test_bucket_count_is_per_cell_for_explode() -> None:
    cfg = _cfg(
        spark_method="explode",
        models=["theta", "holtwinters", "sarimax"],
        data={"source_table": "t", "series_limit": 100},
    )
    # explode: one bucket per (series × model) cell.
    assert default_bucket_count(cfg) == 300


def test_bucket_count_is_per_series_for_naive() -> None:
    cfg = _cfg(
        spark_method="naive",
        models=["theta", "holtwinters"],
        data={"source_table": "t", "series_limit": 100},
    )
    # naive: one bucket per series (models run sequentially inside the task).
    assert default_bucket_count(cfg) == 100


def test_bucket_count_clamped_to_max_parallelism() -> None:
    cfg = _cfg(
        spark_method="explode",
        data={"source_table": "t", "series_limit": 10_000},
        compute={"max_parallelism": 50},
    )
    assert default_bucket_count(cfg) == 50


def test_bucket_count_defaults_to_cap_when_unlimited() -> None:
    cfg = _cfg(spark_method="explode", compute={"max_parallelism": 123})
    # series_limit unset → cell count unknown offline → fall back to the parallelism cap.
    assert default_bucket_count(cfg) == 123


# --- run_group: explode path (cross-joined, model column present) ---------------


def _with_model_col(panel: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Emulate the cross-join: replicate each series row once per model, tagged _sf_model."""
    parts = [panel.assign(**{spark_io._MODEL_COL: m}) for m in models]
    return pd.concat(parts, ignore_index=True)


def test_run_group_explode_one_result_per_cell() -> None:
    cfg = _cfg(spark_method="explode", models=["theta", "holtwinters"])
    pdf = _with_model_col(_panel(["s0", "s1"]), ["theta", "holtwinters"])
    results, status = run_group(pdf, cfg)

    # 2 series × 2 models = 4 cells.
    assert len(results) == 4
    assert {(r.ts_id, r.model_type) for r in results} == {
        ("s0", "theta"),
        ("s0", "holtwinters"),
        ("s1", "theta"),
        ("s1", "holtwinters"),
    }
    assert all(r.status == "ok" for r in results)
    # Helper columns never reach run_cell (would break feature building) — all cells succeeded.
    assert list(status.columns) == list(STATUS_COLUMNS)
    assert len(status) == 4


def test_run_group_naive_loops_models_per_series() -> None:
    cfg = _cfg(spark_method="naive", models=["theta", "holtwinters"])
    pdf = _panel(["s0", "s1"])  # no model column — naive groups by ts_id only
    results, status = run_group(pdf, cfg)

    assert len(results) == 4  # 2 series × 2 models, run sequentially
    assert {(r.ts_id, r.model_type) for r in results} == {
        ("s0", "theta"),
        ("s0", "holtwinters"),
        ("s1", "theta"),
        ("s1", "holtwinters"),
    }
    assert all(r.status == "ok" for r in results)


def test_run_group_error_cell_becomes_status_row() -> None:
    cfg = _cfg(spark_method="explode", models=["nonexistent_model"])
    pdf = _with_model_col(_panel(["s0"]), ["nonexistent_model"])
    results, status = run_group(pdf, cfg)

    assert len(results) == 1
    assert results[0].status == "error"
    assert results[0].error is not None
    # The batch survives (CONTRACTS §3.3): an error is a status row, not an exception.
    assert status.iloc[0]["status"] == "error"


def test_run_group_status_frame_has_fit_seconds() -> None:
    cfg = _cfg(spark_method="explode", models=["theta"])
    pdf = _with_model_col(_panel(["s0"]), ["theta"])
    _results, status = run_group(pdf, cfg)
    assert status["fit_seconds"].dtype == np.float64
    assert (status["fit_seconds"] >= 0).all()


# --- aggregate_status: the driver's header roll-up -----------------------------


def _status(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame([(t, m, s, 0.1) for t, m, s in rows], columns=list(STATUS_COLUMNS))


def test_aggregate_all_ok_is_completed() -> None:
    out = aggregate_status(_status([("s0", "theta", "ok"), ("s1", "theta", "ok")]))
    assert out.status == "COMPLETED"
    assert out.n_series == 2
    assert out.n_cells == 2
    assert out.n_ok == 2
    assert out.n_error == 0


def test_aggregate_mixed_is_partial() -> None:
    out = aggregate_status(_status([("s0", "theta", "ok"), ("s1", "theta", "error")]))
    assert out.status == "PARTIAL"
    assert out.n_ok == 1
    assert out.n_error == 1


def test_aggregate_all_error_is_failed() -> None:
    out = aggregate_status(_status([("s0", "theta", "error"), ("s1", "theta", "error")]))
    assert out.status == "FAILED"
    assert out.n_ok == 0


def test_aggregate_empty_is_failed() -> None:
    out = aggregate_status(pd.DataFrame(columns=list(STATUS_COLUMNS)))
    assert out.status == "FAILED"
    assert out.n_cells == 0
    assert out.n_series == 0
