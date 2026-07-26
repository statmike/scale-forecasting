"""Tests for ``run_cell`` — the G1 unit of work (CONTRACTS §3.1–3.3, BUILD 2.6).

``run_cell`` wires features → (optional backtest) → fit → predict into a ``CellResult`` and
must **never raise**: a failing cell comes back as ``status="error"`` so one bad series can't
sink a batch. These tests cover the happy path (complete CellResult, canonical predictions),
the backtest toggle (OOF present/absent, metrics populated/NaN), and the error path.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from scale_forecasting.config import RunConfig
from scale_forecasting.metrics import METRIC_NAMES
from scale_forecasting.models.base_model import PREDICTION_COLUMNS
from scale_forecasting.worker import CellResult, run_cell

HORIZON = 7


def _series(n: int = 120, ts_id: str = "series-a") -> pd.DataFrame:
    """One ts_id's rows: deterministic trend + weekly seasonality, columns [ts_id, ds, y]."""
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    trend = np.linspace(10.0, 30.0, n)
    weekly = 3.0 * np.sin(np.arange(n) * 2 * np.pi / 7)
    return pd.DataFrame({"ts_id": ts_id, "ds": idx, "y": trend + weekly})


def _cfg(**over: Any) -> RunConfig:
    base: dict[str, Any] = {
        "run_name": "worker test",
        "data": {"source_table": "t", "freq": "D", "horizon": HORIZON},
        "models": ["theta"],
    }
    base.update(over)
    return RunConfig(**base)


# --- happy path ----------------------------------------------------------------


def test_ok_cell_is_complete() -> None:
    res = run_cell(_series(), "theta", _cfg())
    assert isinstance(res, CellResult)
    assert res.status == "ok"
    assert res.error is None
    assert res.ts_id == "series-a"
    assert res.model_type == "theta"
    assert res.compute_engine == "spark"  # default python_runtime
    assert res.run_id and res.model_hash
    assert res.fit_seconds >= 0.0


def test_ok_cell_predictions_are_canonical() -> None:
    res = run_cell(_series(), "theta", _cfg())
    df = res.predictions
    assert list(df.columns) == list(PREDICTION_COLUMNS)
    assert len(df) == HORIZON
    assert df["ds"].dtype == np.dtype("datetime64[ns]")
    assert (df["yhat_lower"] <= df["yhat"] + 1e-6).all()
    assert (df["yhat"] <= df["yhat_upper"] + 1e-6).all()


def test_run_id_deterministic_for_same_config() -> None:
    a = run_cell(_series(), "theta", _cfg())
    b = run_cell(_series(), "theta", _cfg())
    assert a.run_id == b.run_id
    assert a.model_hash == b.model_hash


# --- backtest toggle -----------------------------------------------------------


def test_backtest_off_has_no_oof_and_nan_metrics() -> None:
    res = run_cell(_series(), "theta", _cfg())
    assert res.oof is None
    assert set(res.metrics) == set(METRIC_NAMES)
    assert all(math.isnan(v) for v in res.metrics.values())


def test_backtest_on_populates_oof_and_metrics() -> None:
    cfg = _cfg(
        backtest={"enabled": True, "n_folds": 2, "horizon": HORIZON, "step": HORIZON,
                  "min_train": 30},
    )
    res = run_cell(_series(), "theta", cfg)
    assert res.status == "ok"
    assert res.oof is not None
    assert list(res.oof.columns) == ["ds", "fold_id", "y_true", "yhat"]
    assert res.oof["fold_id"].nunique() == 2
    # at least the decision metric rolled up to a finite value
    assert not math.isnan(res.metrics["wape"])


# --- native model routing ------------------------------------------------------


def test_bigquery_native_routes_to_bigquery_engine() -> None:
    # arima_plus raises NotImplementedError in Arc A → error cell, but engine is bigquery.
    res = run_cell(_series(), "arima_plus", _cfg(models=["arima_plus"]))
    assert res.status == "error"
    assert res.compute_engine == "bigquery"
    assert res.error is not None


# --- error path (never raises) -------------------------------------------------


def test_unknown_model_is_error_not_raise() -> None:
    res = run_cell(_series(), "nope", _cfg(models=["theta"]))
    assert res.status == "error"
    assert res.error is not None
    assert res.predictions.empty
    assert res.oof is None
    assert set(res.metrics) == set(METRIC_NAMES)


def test_bad_series_is_error_not_raise() -> None:
    # Missing the target column → build_features raises inside, caught as an error cell.
    bad = _series().drop(columns=["y"])
    res = run_cell(bad, "theta", _cfg())
    assert res.status == "error"
    assert res.error is not None
    assert res.predictions.empty


def test_error_cell_still_carries_identity() -> None:
    res = run_cell(_series(ts_id="series-z"), "nope", _cfg())
    assert res.ts_id == "series-z"
    assert res.model_type == "nope"
    assert res.run_id  # ids computed before the failure
