"""Tests for the BigQuery-native SQL builders (CONTRACTS §5, DESIGN §3.3).

Pure-string assertions on the rendered CREATE MODEL / forecast INSERT / eval / history SQL plus a
full-script snapshot. No GCP — the ``run`` engine path is exercised live by the ``@gcp`` smoke test.
Covers: model-type routing, the one-statement-all-series id column, ARIMA vs TimesFM shape,
output-column aliasing, custom-holiday CTE presence/absence + name sanitization, the deterministic
series_limit subset, and ``@run_id`` binding for the written run_id column.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from scale_forecasting.config import RunConfig
from scale_forecasting.engines import bigquery_engine as be

SNAPSHOT = Path(__file__).parent / "snapshots" / "bigquery_native.sql"

_DS = "proj.scale_forecasting"


def _cfg(
    models: list[str],
    *,
    holidays: list[str] | None = None,
    exog: list[str] | None = None,
    series_limit: int | None = 100,
    freq: str = "D",
) -> RunConfig:
    over: dict[str, Any] = {
        "run_name": "bq test",
        "data": {
            "source_table": "source_series_native",
            "series_limit": series_limit,
            "freq": freq,
        },
        "models": models,
        "features": {"holidays": holidays or [], "exog": exog or []},
    }
    return RunConfig(**over)


# --- CREATE MODEL --------------------------------------------------------------


def test_create_model_arima_plus_options_and_id_col() -> None:
    sql = be.build_create_model_sql(_cfg(["arima_plus"]), "arima_plus", _DS)
    assert "CREATE OR REPLACE MODEL" in sql
    assert "model_type = 'ARIMA_PLUS'" in sql
    # The one-statement-all-series lever: a time_series_id_col trains every series at once.
    assert "time_series_id_col = 'ts_id'" in sql
    assert "time_series_data_col = 'y'" in sql
    assert "horizon = 28" in sql
    assert "data_frequency = 'DAILY'" in sql


def test_create_model_final_trains_on_all_history() -> None:
    # C2 alignment: the final (true-future) model trains on ALL history — no held-out cutoff — so
    # its ML.FORECAST(horizon) lands beyond MAX(ds), parity with the Spark final fit.
    sql = be.build_create_model_sql(_cfg(["arima_plus"], series_limit=None), "arima_plus", _DS)
    assert "DATE_SUB(MAX(ds)" not in sql
    assert "ds <=" not in sql
    assert "WHERE" not in sql  # no date bound, no series filter → no WHERE at all


def test_create_model_backtest_fold_trains_pre_cutoff() -> None:
    # A backtest fold trains on ds <= cutoff (MAX(ds) - back_steps) into a fold-suffixed object.
    cfg = _cfg(["arima_plus"], series_limit=None)
    sql = be.build_create_model_sql(cfg, "arima_plus", _DS, back_steps=28, fold_id=0)
    assert "ds <= (SELECT DATE_SUB(MAX(ds), INTERVAL 28 DAY)" in sql
    # Fold-suffixed model object so folds + the final model never clobber each other.
    assert "_f0`" in sql


def test_create_model_sliding_fold_has_fixed_window() -> None:
    # scheme='sliding' adds a lower bound so the training window is fixed-width (min_train).
    cfg = RunConfig(
        run_name="bq test",
        data={"source_table": "src", "series_limit": None},
        models=["arima_plus"],
        backtest={
            "enabled": True,
            "scheme": "sliding",
            "min_train": 180,
            "horizon": 28,
            "step": 28,
        },
    )
    sql = be.build_create_model_sql(cfg, "arima_plus", _DS, back_steps=28, fold_id=0)
    assert "ds <= (SELECT DATE_SUB(MAX(ds), INTERVAL 28 DAY)" in sql
    # lower bound = cutoff - min_train = MAX(ds) - (28 + 180)
    assert "ds > (SELECT DATE_SUB(MAX(ds), INTERVAL 208 DAY)" in sql


def test_create_model_name_embeds_run_id_and_is_sanitized() -> None:
    cfg = _cfg(["arima_plus"])
    sql = be.build_create_model_sql(cfg, "arima_plus", _DS)
    # Model object name embeds the config-pinned run_id, hyphens folded to underscores.
    assert "`proj.scale_forecasting.sf_model_arima_plus_" in sql
    assert "-" not in sql.split("sf_model_arima_plus_")[1].split("`")[0]


# --- custom holidays -----------------------------------------------------------


def test_custom_holiday_cte_present_when_configured() -> None:
    cte = be.build_custom_holiday_cte(_cfg(["arima_plus"], holidays=["US"]))
    assert cte.startswith("custom_holiday AS (")
    assert "UNNEST([" in cte
    assert "AS region" in cte and "AS holiday_name" in cte and "AS primary_date" in cte
    assert "preholiday_days" in cte and "postholiday_days" in cte


def test_custom_holiday_names_are_valid_identifiers() -> None:
    cte = be.build_custom_holiday_cte(_cfg(["arima_plus"], holidays=["US"]))
    # Every holiday_name literal must be space-free (valid column name for ML.EXPLAIN_FORECAST).
    import re

    for name in re.findall(r"'([^']*)' AS holiday_name", cte):
        assert " " not in name and name


def test_custom_holiday_cte_absent_without_holidays() -> None:
    assert be.build_custom_holiday_cte(_cfg(["arima_plus"], holidays=[])) == ""
    # And the CREATE MODEL then uses the plain training query, no named subqueries.
    sql = be.build_create_model_sql(_cfg(["arima_plus"], holidays=[]), "arima_plus", _DS)
    assert "custom_holiday" not in sql
    assert "training_data AS" not in sql


# --- forecast INSERT -----------------------------------------------------------


def test_forecast_insert_aliases_and_engine_literal() -> None:
    sql = be.build_forecast_insert_sql(_cfg(["arima_plus"]), "arima_plus", _DS)
    assert "INSERT INTO `proj.scale_forecasting.forecast_predictions`" in sql
    assert "@run_id" in sql  # run_id column bound as a parameter, not interpolated
    assert "'arima_plus'" in sql
    assert "'bigquery'" in sql
    assert "DATE(forecast_timestamp)" in sql
    assert "forecast_value" in sql
    assert "prediction_interval_lower_bound" in sql
    assert "prediction_interval_upper_bound" in sql
    assert "ML.FORECAST(MODEL" in sql


def test_forecast_insert_timesfm_uses_ai_forecast_no_model() -> None:
    sql = be.build_forecast_insert_sql(_cfg(["timesfm"]), "timesfm", _DS)
    assert "AI.FORECAST(" in sql
    assert "ML.FORECAST" not in sql  # TimesFM is serverless — no trained model object
    assert "data_col => 'y'" in sql
    assert "timestamp_col => 'ds'" in sql
    assert "id_cols => ['ts_id']" in sql
    assert "horizon => 28" in sql


def test_forecast_insert_is_true_future_not_held_out() -> None:
    # C2: the final forecast INSERT reads from the all-history model — no held-out cutoff — so it
    # extrapolates beyond MAX(ds). ARIMA_PLUS owns its time axis (ML.FORECAST(horizon) suffices);
    # no future-dates input table is needed for the univariate natives.
    sql = be.build_forecast_insert_sql(_cfg(["arima_plus"]), "arima_plus", _DS)
    assert "DATE_SUB(MAX(ds)" not in sql
    # TimesFM forecasts from all history too (no ds <= cutoff bound on its inline history).
    tsql = be.build_forecast_insert_sql(_cfg(["timesfm"]), "timesfm", _DS)
    assert "DATE_SUB(MAX(ds)" not in tsql


# --- fold plan -----------------------------------------------------------------


def test_fold_plan_mirrors_make_folds_geometry() -> None:
    # back_steps = horizon + (n_folds-1-k)*step, fold 0 = largest step-back (earliest fold),
    # matching backtest.make_folds so native + Python OOF fold ids line up.
    cfg = RunConfig(
        run_name="bq test",
        data={"source_table": "src"},
        models=["arima_plus"],
        backtest={"enabled": True, "n_folds": 3, "horizon": 28, "step": 28},
    )
    assert be.fold_plan(cfg) == [(0, 84), (1, 56), (2, 28)]


def test_fold_create_and_drop_target_the_same_object() -> None:
    # Every fold trains a fold-suffixed object; the matching DROP must name that exact object so
    # backtest runs leave no orphaned sf_model_*_f{k} models behind.
    cfg = _cfg(["arima_plus"])
    create = be.build_fold_create_statements(cfg, "arima_plus", _DS, fold_id=1, back_steps=56)
    drop = be.build_fold_drop_statements(cfg, "arima_plus", _DS, fold_id=1)
    assert len(create) == 1 and len(drop) == 1
    obj = be._model_ref(cfg, "arima_plus", _DS, fold_id=1)
    assert obj in create[0] and obj in drop[0]
    assert drop[0].startswith("DROP MODEL IF EXISTS ")  # safe if the fold CREATE failed


def test_fold_drop_never_targets_the_final_model() -> None:
    # The final true-future model (fold_id=None) backs forecast_predictions and must survive; only
    # fold-suffixed objects are dropped.
    cfg = _cfg(["arima_plus"])
    final_obj = be._model_ref(cfg, "arima_plus", _DS)  # no fold suffix
    for k in range(3):
        drop = be.build_fold_drop_statements(cfg, "arima_plus", _DS, fold_id=k)
        assert final_obj not in drop[0]
        assert f"_f{k}`" in drop[0]


def test_timesfm_has_no_fold_create_or_drop() -> None:
    # TimesFM trains no model object (AI.FORECAST reads history directly), so it has neither a fold
    # CREATE nor a DROP — nothing to clean up.
    cfg = _cfg(["timesfm"])
    assert be.build_fold_create_statements(cfg, "timesfm", _DS, fold_id=0, back_steps=28) == []
    assert be.build_fold_drop_statements(cfg, "timesfm", _DS, fold_id=0) == []


# --- eval + history read-back --------------------------------------------------


def test_eval_query_joins_fold_forecast_to_actuals_with_intervals() -> None:
    sql = be.build_eval_query(_cfg(["arima_plus"]), "arima_plus", _DS, back_steps=28, fold_id=0)
    assert "AS y_true" in sql
    assert "AS yhat" in sql
    assert "AS yhat_lower" in sql and "AS yhat_upper" in sql
    assert "JOIN `proj.scale_forecasting.source_series_native`" in sql
    assert "DATE(f.forecast_timestamp)" in sql
    # Fold eval reads the fold-suffixed model over its held-out window (ds <= cutoff).
    assert "_f0`" in sql


def test_history_query_is_all_history() -> None:
    # C2: MASE/RMSSE scale comes from the full series history (natives train on all of it), so the
    # history read is no longer clipped to a pre-cutoff window.
    sql = be.build_history_query(_cfg(["arima_plus"], series_limit=None), _DS)
    assert "AS ts_id" in sql and "AS y" in sql
    assert "DATE_SUB(MAX(ds)" not in sql


def test_series_ids_query_lists_the_subset() -> None:
    sql = be.build_series_ids_query(_cfg(["arima_plus"], series_limit=100), _DS)
    assert "SELECT DISTINCT ts_id AS ts_id" in sql
    assert "ORDER BY ts_id LIMIT 100" in sql


# --- series_limit subset -------------------------------------------------------


def test_series_limit_subset_present_and_omitted() -> None:
    limited = be.build_create_model_sql(_cfg(["arima_plus"], series_limit=100), "arima_plus", _DS)
    assert "ORDER BY ts_id LIMIT 100" in limited
    unlimited = be.build_create_model_sql(
        _cfg(["arima_plus"], series_limit=None), "arima_plus", _DS
    )
    assert "LIMIT" not in unlimited


# --- bqml_options / best_params ------------------------------------------------


def test_bqml_options_maps_columns() -> None:
    opts = be.bqml_options(_cfg(["arima_plus"]), "arima_plus")
    assert opts["model_type"] == "ARIMA_PLUS"
    assert opts["time_series_id_col"] == "ts_id"
    assert opts["time_series_timestamp_col"] == "ds"
    assert opts["time_series_data_col"] == "y"
    assert opts["horizon"] == 28


def test_bqml_options_timesfm_returns_ai_forecast_params() -> None:
    # TimesFM has no CREATE MODEL, but run() still stamps best_params for every model — so
    # bqml_options must resolve for it (not KeyError) and describe the AI.FORECAST call instead.
    opts = be.bqml_options(_cfg(["timesfm"]), "timesfm")
    assert "TimesFM" in opts["model_type"]
    assert opts["id_cols"] == ["ts_id"]
    assert opts["horizon"] == 28
    assert opts["confidence_level"] == 0.8


# --- snapshot ------------------------------------------------------------------


def test_setup_sql_snapshot() -> None:
    cfg = _cfg(
        ["arima_plus", "timesfm"],
        holidays=["US"],
        series_limit=100,
    )
    rendered = "\n\n-- ===== next model =====\n\n".join(
        be.render_setup_sql(cfg, m, _DS) for m in cfg.models
    )
    if os.environ.get("SF_UPDATE_SNAPSHOTS") == "1":
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(rendered)
    assert SNAPSHOT.exists(), "snapshot missing; run with SF_UPDATE_SNAPSHOTS=1 to create"
    assert rendered == SNAPSHOT.read_text()
