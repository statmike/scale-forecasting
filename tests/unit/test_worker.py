"""Tests for ``run_cell`` — the unit of work that runs the same locally and in the cloud.

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


def test_ok_cell_stamps_trace_timing_and_worker() -> None:
    # The cell carries a wall-clock bracket + worker identity for the run trace.
    res = run_cell(_series(), "theta", _cfg())
    assert res.worker_id and ":" in res.worker_id  # hostname:pid
    assert res.cell_started_at is not None and res.cell_ended_at is not None
    assert res.cell_ended_at >= res.cell_started_at


def test_error_cell_also_stamps_trace_timing() -> None:
    # An error cell still gets timed + attributed, so failures show on the trace too.
    res = run_cell(_series(), "nope", _cfg(models=["theta"]))
    assert res.status == "error"
    assert res.worker_id and res.cell_started_at is not None and res.cell_ended_at is not None


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
        backtest={
            "enabled": True,
            "n_folds": 2,
            "horizon": HORIZON,
            "step": HORIZON,
            "min_train": 30,
        },
    )
    res = run_cell(_series(), "theta", cfg)
    assert res.status == "ok"
    assert res.oof is not None
    assert list(res.oof.columns) == ["ds", "fold_id", "y_true", "yhat"]
    assert res.oof["fold_id"].nunique() == 2
    # at least the decision metric rolled up to a finite value
    assert not math.isnan(res.metrics["wape"])


# --- artifact persistence (persist_models gate) --------------------------------


def test_persist_off_by_default_yields_no_artifact() -> None:
    # Default config: persistence off, so no bytes to upload (model_artifact stays null).
    res = run_cell(_series(), "theta", _cfg())
    assert res.artifact_bytes is None


def test_persist_on_serializes_the_fitted_model() -> None:
    import pickle

    cfg = _cfg(compute={"persist_models": True})
    res = run_cell(_series(), "theta", cfg)
    assert res.status == "ok"
    assert isinstance(res.artifact_bytes, bytes) and res.artifact_bytes
    # Round-trips back to a fitted model of the right type (default pickle serialize).
    from scale_forecasting.models import get_model

    restored = pickle.loads(res.artifact_bytes)
    assert isinstance(restored, get_model("theta"))


def test_persist_failure_degrades_to_no_artifact(monkeypatch: Any) -> None:
    # A serialize() that raises must not sink the forecast — cell stays ok, artifact is None.
    from scale_forecasting.models.base_model import BaseModel

    def _boom(self: BaseModel) -> bytes | None:
        raise RuntimeError("cannot pickle")

    monkeypatch.setattr(BaseModel, "serialize", _boom)
    res = run_cell(_series(), "theta", _cfg(compute={"persist_models": True}))
    assert res.status == "ok"
    assert res.artifact_bytes is None


# --- native model routing ------------------------------------------------------


def test_bigquery_native_routes_to_bigquery_engine() -> None:
    # arima_plus is executed as SQL in BigQuery; its in-process fit/predict raises (never called
    # on the real path) → error cell here, but compute_engine is still tagged bigquery.
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


# --- HPO params threading into the cell -----------------------------------------


def test_pre_resolved_params_land_in_best_params() -> None:
    # The fleetwide path: the driver tuned xgboost and hands run_cell the winning params directly.
    # get_params (→ forecast_metadata.best_params) must reflect them, not an empty {}.
    params = {"n_estimators": 123, "max_depth": 4, "learning_rate": 0.07}
    res = run_cell(_series(), "xgboost", _cfg(models=["xgboost"]), params)
    assert res.status == "ok"
    assert res.best_params == params


def test_default_no_params_is_empty_best_params() -> None:
    # No params + HPO off → today's behavior: the model runs with its own defaults ({}).
    res = run_cell(_series(), "xgboost", _cfg(models=["xgboost"]))
    assert res.status == "ok"
    assert res.best_params == {}


def test_pre_resolved_params_do_not_change_the_run_id() -> None:
    # The invariant that forces the fleetwide seam placement: params must NOT enter cfg, so the same
    # cfg yields the same run_id whether or not tuned params are passed (reproducibility).
    cfg = _cfg(models=["xgboost"])
    a = run_cell(_series(), "xgboost", cfg, {"n_estimators": 200, "max_depth": 5})
    b = run_cell(_series(), "xgboost", cfg)
    assert a.run_id == b.run_id
    assert a.model_hash == b.model_hash


def test_per_series_hpo_tunes_and_records_best_params() -> None:
    # The per_series granularity: run_cell tunes on THIS series (no pre-resolved params) and records
    # the winner. theta's space is {deseasonalize}, so best_params carries that key.
    cfg = _cfg(
        models=["theta"],
        backtest={
            "enabled": True,
            "n_folds": 2,
            "horizon": HORIZON,
            "step": HORIZON,
            "min_train": 60,
        },
        hpo={"enabled": True, "n_trials": 4, "granularity": "per_series"},
    )
    res = run_cell(_series(), "theta", cfg)
    assert res.status == "ok"
    assert set(res.best_params) == {"deseasonalize"}


# --- harvested measurement (compute.profile.measure) -----------------------------


def test_a_successful_cell_records_what_it_cost() -> None:
    """Harvest is the default: a completed run is the evidence a later run is sized from."""
    result = run_cell(_series(), "theta", _cfg())
    assert result.status == "ok"
    assert result.cpu_seconds is not None and result.cpu_seconds >= 0.0
    assert result.n_obs == 120
    # Absolute footprint, not this cell's increment — the number that sizes a slot.
    assert result.process_rss_bytes is None or result.process_rss_bytes > 0


def test_measurement_off_leaves_every_axis_null_rather_than_zero() -> None:
    """``0`` is a measurement; NULL is the absence of one, and only one of them is true here."""
    result = run_cell(_series(), "theta", _cfg(compute={"profile": {"measure": "off"}}))
    assert result.status == "ok"
    assert result.cpu_seconds is None
    assert result.process_rss_bytes is None
    assert result.peak_gpu_bytes is None
    assert result.intraop_threads is None
    assert result.n_obs is None
    # The wall clock is not part of the opt-in — it always was, and the trace needs it.
    assert result.fit_seconds > 0.0


def test_profiling_off_vetoes_measurement_even_when_measure_asks_for_it() -> None:
    """One switch turns the whole feature off; ``measure`` cannot re-enable it behind it."""
    cfg = _cfg(compute={"profile": {"mode": "off", "measure": "controlled"}})
    assert run_cell(_series(), "theta", cfg).cpu_seconds is None


def test_a_failed_cell_carries_no_measurement_because_it_never_fit_anything() -> None:
    """An error cell's zero elapsed is exactly how the harvest reader infers failure."""
    result = run_cell(_series(), "no_such_model", _cfg(models=["no_such_model"]))
    assert result.status == "error"
    assert result.cpu_seconds is None
    assert result.fit_seconds == 0.0


def test_the_thread_cap_in_force_is_recorded_so_effective_cores_can_be_read_honestly(
    monkeypatch: Any,
) -> None:
    """cpu/wall under a cap reports the cap back; without the cap recorded that is invisible."""
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    assert run_cell(_series(), "theta", _cfg()).intraop_threads == 3
    monkeypatch.delenv("OMP_NUM_THREADS")
    assert run_cell(_series(), "theta", _cfg()).intraop_threads is None
