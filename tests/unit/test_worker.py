"""Tests for ``run_cell`` — the unit of work that runs the same locally and in the cloud.

``run_cell`` wires features → (optional backtest) → fit → predict into a ``CellResult`` and
must **never raise**: a failing cell comes back as ``status="error"`` so one bad series can't
sink a batch. These tests cover the happy path (complete CellResult, canonical predictions),
the backtest toggle (OOF present/absent, metrics populated/NaN), and the error path.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scale_forecasting import worker
from scale_forecasting.backtest import OOF_COLUMNS
from scale_forecasting.calibration import CALIBRATED_COLUMNS
from scale_forecasting.config import RunConfig
from scale_forecasting.errors import ConfigError, DataError, ModelError
from scale_forecasting.metrics import METRIC_NAMES
from scale_forecasting.resources.catalog import _INTRAOP_ENV_VARS
from scale_forecasting.worker import ERROR_CLASSES, CellResult, classify_error, run_cell

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
    assert list(df.columns) == list(CALIBRATED_COLUMNS)
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
    assert list(res.oof.columns) == list(OOF_COLUMNS)
    assert res.oof["fold_id"].nunique() == 2
    # at least the decision metric rolled up to a finite value
    assert not math.isnan(res.metrics["wape"])
    # And so did the interval metrics, which every Python cell reported as NaN before the folds
    # were scored on the bounds the model had already returned.
    assert not math.isnan(res.metrics["coverage"])
    # theta computes its own prediction interval, so its coverage is a claim about theta's
    # uncertainty rather than about the spread of its residuals. The column says which.
    assert res.interval_source == "native"


def test_a_model_without_its_own_interval_records_the_residual_provenance() -> None:
    # xgboost has no notion of a prediction interval, so `BaseModel.residual_intervals` builds one
    # from the spread of its in-sample residuals. Both kinds of band land in the same two columns,
    # which is exactly why the row has to say which kind it is — a coverage number means something
    # different when the band came from the model than when it came from its leftovers.
    res = run_cell(_series(), "xgboost", _cfg(models=["xgboost"]))
    assert res.status == "ok"
    assert res.interval_source == "residual"


# --- which number ships in `yhat` ----------------------------------------------
#
# End to end through `run_cell`, because the unit tests in `test_calibration.py` exercise the
# arithmetic on hand-built frames and this is the seam where a config field has to reach it.


def _flat_series(n: int = 120, level: float = 50.0) -> pd.DataFrame:
    """Constant level plus symmetric noise that sums to exactly zero — so the mean is known."""
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    noise = np.tile([1.0, -1.0], n // 2)
    return pd.DataFrame({"ts_id": "flat", "ds": idx, "y": level + noise})


def test_raw_arm_on_constant_plus_noise_returns_the_analytic_value() -> None:
    """`naive_mean` on a level with zero-sum noise forecasts the level. Under `raw` it says so.

    The check that `raw` really is raw: any residual shift leaking in would move this off 50.0, and
    it is the one case where the right answer is known without running anything.
    """
    cfg = _cfg(models=["naive_mean"], output={"point_forecast": "raw"})
    res = run_cell(_flat_series(), "naive_mean", cfg)
    assert res.status == "ok"
    assert np.allclose(res.predictions["yhat"], 50.0)
    assert np.allclose(res.predictions["yhat_raw"], 50.0)
    assert res.point_forecast_source == "raw"


def test_both_arms_are_written_whichever_one_ships() -> None:
    """The choice is never destructive: the arm not taken is still in the row."""
    for arm in ("raw", "median"):
        res = run_cell(_series(), "theta", _cfg(output={"point_forecast": arm}))
        df = res.predictions
        assert res.point_forecast_source == arm
        assert df["yhat_raw"].notna().all() and df["yhat_adjusted"].notna().all()
        shipped = "yhat_raw" if arm == "raw" else "yhat_adjusted"
        assert np.allclose(df["yhat"], df[shipped])


def test_without_a_backtest_the_band_is_the_models_own_and_the_row_says_so() -> None:
    res = run_cell(_series(), "theta", _cfg())
    assert res.interval_calibration == "in-sample"
    assert res.point_forecast_margin is None  # nothing held out to grade the correction on


def test_a_backtest_recalibrates_the_band_and_grades_the_arm() -> None:
    res = run_cell(_series(200), "theta", _bt_cfg(n_folds=3))
    assert res.status == "ok"
    assert res.interval_calibration in ("oof-per-step", "oof-flat")
    assert res.point_forecast_margin is not None  # a number, sign not asserted — it is measured


def _auto_cfg(n_folds: int) -> RunConfig:
    """A backtested config that asks for per-cell arm selection rather than naming an arm."""
    return _cfg(
        backtest={
            "enabled": True,
            "n_folds": n_folds,
            "horizon": HORIZON,
            "step": HORIZON,
            "min_train": 30,
        },
        output={"point_forecast": "auto"},
    )


def test_a_fixed_arm_records_that_nothing_was_decided() -> None:
    """`point_forecast_decision` separates "you asked for this" from "the folds picked it"."""
    for arm in ("raw", "median"):
        res = run_cell(_series(200), "theta", _cfg(output={"point_forecast": arm}))
        assert res.point_forecast_source == arm
        assert res.point_forecast_decision == "configured"


def test_auto_picks_an_arm_per_cell_and_records_how() -> None:
    """End-to-end: the arm shipped in the row is the one the cell's own held-out folds chose."""
    res = run_cell(_series(300), "theta", _auto_cfg(4))
    assert res.status == "ok"
    assert res.point_forecast_source in ("raw", "median")
    assert res.point_forecast_decision in ("auto-raw", "auto-corrected")
    # Whichever way it went, the shipped column has to match the arm the row claims.
    shipped = "yhat_raw" if res.point_forecast_source == "raw" else "yhat_adjusted"
    assert np.allclose(res.predictions["yhat"], res.predictions[shipped])
    # The diagnostic is still corrected-vs-raw regardless of what was selected, so a fleetwide
    # average over cells that chose differently stays a single comparable number.
    assert res.point_forecast_margin is not None


def test_auto_on_an_unscorable_series_falls_back_and_says_so() -> None:
    """Item 2.1 kept the forecast when scoring fails; `auto` has to keep it too, not error."""
    res = run_cell(_series(30 + HORIZON - 1), "theta", _auto_cfg(2))
    assert res.status == "ok"
    assert len(res.predictions) == HORIZON
    assert res.point_forecast_source == "median"  # the fleetwide arm for the default metric
    assert res.point_forecast_decision == "auto-no-backtest"


# --- a scoring shortfall must never cost the forecast --------------------------
#
# The whole point of this section: backtesting *scores* a model, it does not produce the forecast.
# Every failure mode below used to surface as `status="error"` with no predictions at all, which is
# how short history became the largest error class in the registry.


def _bt_cfg(**over: Any) -> RunConfig:
    bt = {"enabled": True, "n_folds": 2, "horizon": HORIZON, "step": HORIZON, "min_train": 30}
    bt.update(over)
    return _cfg(backtest=bt)


def test_a_series_too_short_to_score_still_returns_its_forecast() -> None:
    """The exit-gate case: one observation short of a single fold."""
    res = run_cell(_series(30 + HORIZON - 1), "theta", _bt_cfg())
    assert res.status == "ok"
    assert res.error is None
    assert len(res.predictions) == HORIZON
    # Not an empty frame — that would write zero rows and read back as "not asked".
    assert res.oof is None
    assert res.backtest_status == "unscored"
    assert res.n_folds_achieved == 0
    assert all(math.isnan(v) for v in res.metrics.values())


def test_an_unscored_cell_forecasts_exactly_what_the_same_cell_forecasts_unscored() -> None:
    """Degrading must not perturb the forecast — the fit was never in question."""
    short = _series(30 + HORIZON - 1)
    degraded = run_cell(short, "theta", _bt_cfg())
    never_asked = run_cell(short, "theta", _cfg())
    pd.testing.assert_frame_equal(degraded.predictions, never_asked.predictions)


def test_a_series_that_supports_some_folds_is_scored_on_them_and_says_so() -> None:
    res = run_cell(_series(30 + HORIZON), "theta", _bt_cfg())  # room for exactly one of two folds
    assert res.status == "ok"
    assert res.backtest_status == "reduced"
    assert res.n_folds_achieved == 1
    assert res.oof is not None and res.oof["fold_id"].nunique() == 1
    assert not math.isnan(res.metrics["wape"])  # a reduced backtest still scores


def test_a_long_series_is_untouched_by_the_clamp_and_reports_a_full_backtest() -> None:
    res = run_cell(_series(), "theta", _bt_cfg())
    assert res.backtest_status == "full"
    assert res.n_folds_achieved == 2
    assert res.backtest_note is None  # nothing to explain when nothing was short
    assert res.oof is not None and res.oof["fold_id"].nunique() == 2


def test_the_note_names_the_arithmetic_so_a_reader_knows_how_much_history_was_needed() -> None:
    """Restating the status would be useless; the actionable part is the shortfall itself."""
    res = run_cell(_series(30 + HORIZON - 1), "theta", _bt_cfg())
    note = res.backtest_note or ""
    assert "0 of 2 folds" in note
    assert "min_train=30" in note and f"horizon={HORIZON}" in note and f"step={HORIZON}" in note


@pytest.mark.parametrize(
    "policy",
    [
        {"short_series": "adapt"},
        {"short_series": "overlap"},
        {"short_series": "shrink_train", "min_train_floor": 10},
        {"short_series": "skip"},
        {"short_series": "error"},
    ],
    ids=lambda p: str(p["short_series"]),
)
def test_no_short_series_policy_can_cost_a_cell_its_forecast(policy: dict[str, Any]) -> None:
    """`error` refuses the run at plan time and nowhere else — this is the "nowhere else" half.

    `launch_plan.preflight_short_series` and `assert_panel_supports_folds` are where a shortfall
    stops a run, and they run before any cell exists. A cell that reaches `run_cell` anyway — a
    staged config, a unit call, an engine that skipped the pre-flight — must still fit and forecast,
    because the fit was never what fell short. Every policy is swept rather than just `error`, since
    the guarantee is about `run_cell`, not about which policy is selected.

    **The forecast is only required to be *identical* where the cell went unscored.**
    ``shrink_train`` rescues this series — it spends training history to reach a fold — and a cell
    that gets folds picks its point-forecast arm from them, so its shipped numbers move. That is
    the policy working, not a leak: the guarantee is that a scoring shortfall never costs a
    forecast, not that scoring never changes one.
    """
    short = _series(30 + HORIZON - 1)  # one observation short of a single fold
    res = run_cell(short, "theta", _bt_cfg(**policy))
    assert res.status == "ok" and res.error is None
    assert len(res.predictions) == HORIZON

    unscored = run_cell(short, "theta", _cfg()).predictions
    if res.n_folds_achieved:
        assert policy["short_series"] == "shrink_train", "only that policy can rescue this series"
        assert not res.predictions["yhat"].equals(unscored["yhat"])
    else:
        pd.testing.assert_frame_equal(res.predictions, unscored)


def test_a_backtest_that_raises_loses_the_score_and_nothing_else(monkeypatch: Any) -> None:
    """The catch-all arm: whatever scoring does, the final fit below it still runs."""

    def boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("scoring exploded")

    monkeypatch.setattr(worker, "backtest_cell", boom)
    res = run_cell(_series(), "theta", _bt_cfg())
    assert res.status == "ok"
    assert len(res.predictions) == HORIZON
    assert res.oof is None
    assert res.backtest_status == "failed"
    assert res.n_folds_achieved == 0
    assert "scoring exploded" in (res.backtest_note or "")
    assert all(math.isnan(v) for v in res.metrics.values())


def test_the_context_a_model_gets_carries_the_largest_horizon_it_will_be_asked_for() -> None:
    """One context is shared by the backtest folds and the final fit, so it must mean the larger.

    Carrying only the forward horizon made the field a trap: a model sizing anything off it would
    be right on the final fit and short on every fold, in a run where both numbers are legal and
    different.
    """
    assert worker._model_context(_bt_cfg(horizon=90)).horizon == 90  # backtest asks for more
    assert worker._model_context(_cfg()).horizon == HORIZON  # forward only


def test_backtesting_switched_off_is_the_one_case_with_no_scoring_verdict() -> None:
    """All three NULL is what distinguishes "never asked" from "asked and could not"."""
    res = run_cell(_series(), "theta", _cfg())
    assert res.backtest_status is None
    assert res.n_folds_achieved is None
    assert res.backtest_note is None


# --- why a cell failed: the error classifier -----------------------------------
#
# The tokens are a fixed vocabulary so a reader can GROUP BY them. What each test really pins is
# the *ordering* of `ERROR_CLASSES`, because first match wins and several exceptions match more
# than one row.


class CapacityExhausted(Exception):
    """Stands in for `capacity.CapacityExhausted`, which needs a ledger to construct.

    Named identically on purpose: the table matches type *names* rather than classes, so that the
    classifier never imports `capacity` into the executor hot path. That makes the name the
    contract, and `test_the_capacity_row_names_the_class_that_actually_exists` is what keeps this
    stand-in honest if the real one is ever renamed.
    """


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (MemoryError("Unable to allocate 4.2 GiB for an array"), "OOM"),
        (RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"), "OOM"),
        (CapacityExhausted("no bounds left"), "CAPACITY"),
        (RuntimeError("429 Quota exceeded: too many concurrent queries"), "CAPACITY"),
        (RuntimeError("503 Service Unavailable: backend error"), "TRANSIENT_INFRA"),
        (ConnectionResetError("connection reset by peer"), "TRANSIENT_INFRA"),
        (TimeoutError("deadline exceeded"), "TRANSIENT_INFRA"),
        (DataError("series 's_001' has only 5 observations, needs >= 12"), "SHORT_HISTORY"),
        (ConfigError("not enough data for 3 folds"), "SHORT_HISTORY"),
        (DataError("series 's_001' has duplicate timestamp 2026-01-04"), "BAD_DATA"),
        (DataError("target column has non-numeric values"), "BAD_DATA"),
        (ModelError("unknown model 'nope'; registered models: theta"), "CONFIG_REPAIRABLE"),
        (ConfigError("ensemble needs at least two base models"), "CONFIG_REPAIRABLE"),
        (ModelError("fit failed: optimization did not converge"), "MODEL_ERROR"),
        (np.linalg.LinAlgError("Singular matrix"), "MODEL_ERROR"),
        (ValueError("something nobody has seen before"), "UNKNOWN"),
    ],
)
def test_the_classifier_files_each_failure_under_the_token_a_reader_would_act_on(
    exc: BaseException, expected: str
) -> None:
    assert classify_error(exc) == expected


def test_the_capacity_row_names_the_class_that_actually_exists() -> None:
    """Matching by name buys a cheap classifier and costs a compile-time check — a rename in
    `capacity.py` would leave the table matching a class nobody raises any more, and every
    exhausted-capacity cell would silently file as `UNKNOWN`. This is that check, moved to a test.
    """
    from scale_forecasting.capacity import CapacityExhausted as real

    named = {name for _token, types, _needles in ERROR_CLASSES for name in types}
    assert real.__name__ in named


def test_short_history_outranks_bad_data_because_both_are_the_same_exception_type() -> None:
    """Order is the policy. "Too short" is actionable; "bad data" is a shrug."""
    assert classify_error(DataError("has only 5 observations, needs >= 12")) == "SHORT_HISTORY"
    assert classify_error(DataError("gap at 2026-01-15")) == "BAD_DATA"


def test_a_model_name_that_is_not_registered_is_a_config_bug_not_a_model_bug() -> None:
    """`get_model` raises ModelError for a typo, which would send a reader to debug a model that
    was never constructed. The `unknown model '` row has to precede the ModelError row."""
    res = run_cell(_series(), "nope", _cfg(models=["theta"]))
    assert res.status == "error"
    assert res.error_class == "CONFIG_REPAIRABLE"


def test_every_token_the_table_can_return_is_in_the_documented_vocabulary() -> None:
    # A token invented per stack trace would make the column unqueryable.
    vocabulary = {
        "OOM",
        "TRANSIENT_INFRA",
        "CAPACITY",
        "BAD_DATA",
        "SHORT_HISTORY",
        "MODEL_ERROR",
        "CONFIG_REPAIRABLE",
    }
    assert {token for token, _, _ in ERROR_CLASSES} == vocabulary
    assert len({token for token, _, _ in ERROR_CLASSES}) == len(ERROR_CLASSES)  # no duplicate rows


def test_an_ok_cell_has_no_error_class_because_there_is_nothing_to_classify() -> None:
    res = run_cell(_series(), "theta", _cfg())
    assert res.status == "ok"
    assert res.error is None
    assert res.error_class is None


def test_a_failed_cell_carries_both_the_class_and_the_text_it_came_from() -> None:
    # One is for grouping, the other for reading. Losing either makes the row less useful than the
    # log line it replaced.
    res = run_cell(_series().drop(columns=["y"]), "theta", _cfg())
    assert res.status == "error"
    assert res.error_class is not None
    assert res.error and res.error_class != res.error


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


def _pin(monkeypatch: Any, **caps: str | None) -> None:
    """Set every intra-op variable to ``caps[name]``, deleting the ones passed as None."""
    for name in _INTRAOP_ENV_VARS:
        value = caps.get(name, "1")
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)


def test_the_thread_cap_in_force_is_recorded_so_effective_cores_can_be_read_honestly(
    monkeypatch: Any,
) -> None:
    """cpu/wall under a cap reports the cap back; without the cap recorded that is invisible."""
    _pin(monkeypatch, **dict.fromkeys(_INTRAOP_ENV_VARS, "3"))
    assert run_cell(_series(), "theta", _cfg()).intraop_threads == 3
    _pin(monkeypatch, **dict.fromkeys(_INTRAOP_ENV_VARS, None))
    assert run_cell(_series(), "theta", _cfg()).intraop_threads is None


def test_one_uncapped_pool_uncaps_the_process_however_many_of_the_others_are_pinned(
    monkeypatch: Any,
) -> None:
    """Reading OMP alone reported the pin back to itself and called an eight-thread fit a one.

    These five variables cap different native thread pools, and a fit uses whichever library sits
    underneath it. ``OMP_NUM_THREADS=1`` next to an unset ``OPENBLAS_NUM_THREADS`` is not a
    single-threaded process — the OpenBLAS matrix work still spreads across the node. Recording 1
    there made ``cpu_seconds / fit_seconds`` come out at clean single-threaded occupancy on a run
    that was oversubscribed, which is the measurement confirming the assumption rather than
    testing it. So any one unset variable yields None for the whole process.
    """
    for uncapped in _INTRAOP_ENV_VARS:
        _pin(monkeypatch, **{uncapped: None})
        assert worker._intraop_threads() is None, uncapped


def test_the_widest_cap_wins_because_the_loosest_pool_is_the_one_that_spreads(
    monkeypatch: Any,
) -> None:
    """Four pools pinned to 1 and a fifth at 4 is a process that can reach four threads."""
    _pin(monkeypatch, MKL_NUM_THREADS="4")
    assert worker._intraop_threads() == 4


def test_an_unparseable_cap_is_not_a_cap(monkeypatch: Any) -> None:
    """``OMP_NUM_THREADS=all`` caps nothing; guessing a number from it would invent evidence."""
    _pin(monkeypatch, OPENBLAS_NUM_THREADS="all")
    assert worker._intraop_threads() is None


# --- what the cell says about the search behind its params ---------------------


def _hpo_cfg(**over: Any) -> RunConfig:
    base: dict[str, Any] = {
        "backtest": {"enabled": True, "n_folds": 3, "horizon": 7, "step": 7, "min_train": 30},
        "hpo": {"enabled": True, "granularity": "fleetwide", "n_trials": 2},
    }
    base.update(over)
    return _cfg(**base)


def test_a_cell_that_never_searched_claims_nothing() -> None:
    # The common case, and the reason the column is nullable: no search ran, so there is no
    # honesty question to answer and "holdout" would be a claim about something that never
    # happened. Passing params with HPO off is the fleetwide map absent, not a silent search.
    assert worker.hpo_scoring_basis(500, _cfg(), None) is None
    assert worker.hpo_scoring_basis(500, _cfg(), {"n_estimators": 10}) is None


def test_fleetwide_tuning_reports_the_runs_fold_geometry() -> None:
    # The cell did not run the search — the driver did, on a sample it never saw — so the only
    # honest thing it can report is whether the run had an inner fold at all.
    cfg = _hpo_cfg()
    assert worker.hpo_scoring_basis(500, cfg, {"n_estimators": 10}) == "holdout"
    one = cfg.model_copy(update={"backtest": cfg.backtest.model_copy(update={"n_folds": 1})})
    assert worker.hpo_scoring_basis(500, one, {"n_estimators": 10}) == "in_sample"


def test_per_series_tuning_reports_this_series_own_fold_count() -> None:
    # Here the search really did run on this series, so a series too short for two folds has
    # tuned on the very window it will be scored on — and says so, per series.
    cfg = _hpo_cfg(hpo={"enabled": True, "granularity": "per_series", "n_trials": 2})
    assert worker.hpo_scoring_basis(200, cfg, None) == "holdout"
    assert worker.hpo_scoring_basis(38, cfg, None) == "in_sample"  # exactly one fold fits


def test_a_tuned_cell_carries_the_basis_onto_its_registry_row() -> None:
    from scale_forecasting.registry.rows import assemble_metadata_row

    cfg = _hpo_cfg(hpo={"enabled": True, "granularity": "per_series", "n_trials": 2})
    res = run_cell(_series(n=200), "naive_mean", cfg)
    assert res.status == "ok"
    assert res.hpo_scoring == "holdout"
    row = assemble_metadata_row(res, datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC))
    assert row["hpo_scoring"] == "holdout"
