"""Tests for the BigQuery-native SQL builders (CONTRACTS §5, DESIGN §3.3).

Pure-string assertions on the rendered CREATE MODEL / forecast INSERT / eval / history SQL plus a
full-script snapshot. No GCP — the ``run`` engine path is exercised live by the ``@gcp`` smoke test.
Covers: model-type routing, the one-statement-all-series id column, ARIMA vs XREG vs TimesFM shape,
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
        "data": {"source_table": "source_series", "series_limit": series_limit, "freq": freq},
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
    # Held-out cut: train up to and including the cutoff (last training date).
    assert "ds <= (SELECT DATE_SUB(MAX(ds), INTERVAL 28 DAY)" in sql


def test_create_model_xreg_selects_exog_into_training() -> None:
    sql = be.build_create_model_sql(
        _cfg(["arima_plus_xreg"], exog=["price_index"]), "arima_plus_xreg", _DS
    )
    assert "model_type = 'ARIMA_PLUS_XREG'" in sql
    assert "SELECT ts_id, ds, y, price_index" in sql


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


def test_forecast_insert_xreg_supplies_future_features() -> None:
    sql = be.build_forecast_insert_sql(
        _cfg(["arima_plus_xreg"], exog=["price_index"]), "arima_plus_xreg", _DS
    )
    # XREG ML.FORECAST takes the held-out window's real future features before the STRUCT.
    assert "ML.FORECAST(MODEL" in sql
    assert "SELECT ts_id, ds, price_index" in sql
    assert "ds > (SELECT DATE_SUB(MAX(ds)" in sql
    assert "STRUCT(28 AS horizon, 0.8 AS confidence_level)" in sql


def test_forecast_insert_timesfm_uses_ai_forecast_no_model() -> None:
    sql = be.build_forecast_insert_sql(_cfg(["timesfm"]), "timesfm", _DS)
    assert "AI.FORECAST(" in sql
    assert "ML.FORECAST" not in sql  # TimesFM is serverless — no trained model object
    assert "data_col => 'y'" in sql
    assert "timestamp_col => 'ds'" in sql
    assert "id_cols => ['ts_id']" in sql
    assert "horizon => 28" in sql


# --- eval + history read-back --------------------------------------------------


def test_eval_query_joins_forecast_to_actuals_with_intervals() -> None:
    sql = be.build_eval_query(_cfg(["arima_plus"]), "arima_plus", _DS)
    assert "AS y_true" in sql
    assert "AS yhat" in sql
    assert "AS yhat_lower" in sql and "AS yhat_upper" in sql
    assert "JOIN `proj.scale_forecasting.source_series`" in sql
    assert "DATE(f.forecast_timestamp)" in sql


def test_history_query_is_pre_cutoff_training_window() -> None:
    sql = be.build_history_query(_cfg(["arima_plus"]), _DS)
    assert "AS ts_id" in sql and "AS y" in sql
    assert "ds <= (SELECT DATE_SUB(MAX(ds), INTERVAL 28 DAY)" in sql


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


# --- snapshot ------------------------------------------------------------------


def test_setup_sql_snapshot() -> None:
    cfg = _cfg(
        ["arima_plus", "arima_plus_xreg", "timesfm"],
        holidays=["US"],
        exog=["price_index"],
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
