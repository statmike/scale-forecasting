"""Tests for the BigQuery-native SQL builders (`engines.bigquery_sql`).

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

from scale_forecasting.backtest import make_folds
from scale_forecasting.config import RunConfig
from scale_forecasting.engines.bigquery_names import _model_ref
from scale_forecasting.engines.bigquery_sql import (
    bqml_options,
    build_create_model_sql,
    build_custom_holiday_cte,
    build_eval_query,
    build_fold_create_statements,
    build_fold_drop_statements,
    build_forecast_insert_sql,
    build_history_query,
    build_series_ids_query,
    build_setup_statements,
    fold_plan,
    render_setup_sql,
)

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
    sql = build_create_model_sql(_cfg(["arima_plus"]), "arima_plus", _DS)
    assert "CREATE OR REPLACE MODEL" in sql
    assert "model_type = 'ARIMA_PLUS'" in sql
    # The one-statement-all-series lever: a time_series_id_col trains every series at once.
    assert "time_series_id_col = 'ts_id'" in sql
    assert "time_series_data_col = 'y'" in sql
    assert "horizon = 28" in sql
    assert "data_frequency = 'DAILY'" in sql


def test_create_model_final_trains_on_all_history() -> None:
    # The final (true-future) model trains on ALL history — no held-out cutoff — so
    # its ML.FORECAST(horizon) lands beyond MAX(ds), parity with the Spark final fit.
    sql = build_create_model_sql(_cfg(["arima_plus"], series_limit=None), "arima_plus", _DS)
    assert "DATE_SUB(MAX(ds)" not in sql
    assert "ds <=" not in sql
    assert "WHERE" not in sql  # no date bound, no series filter → no WHERE at all


def test_create_model_backtest_fold_trains_pre_cutoff() -> None:
    # A backtest fold trains on ds <= cutoff (MAX(ds) - back_steps) into a fold-suffixed object.
    cfg = _cfg(["arima_plus"], series_limit=None)
    sql = build_create_model_sql(cfg, "arima_plus", _DS, back_steps=28, fold_id=0)
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
    sql = build_create_model_sql(cfg, "arima_plus", _DS, back_steps=28, fold_id=0)
    assert "ds <= (SELECT DATE_SUB(MAX(ds), INTERVAL 28 DAY)" in sql
    # lower bound = cutoff - min_train = MAX(ds) - (28 + 180)
    assert "ds > (SELECT DATE_SUB(MAX(ds), INTERVAL 208 DAY)" in sql


def test_create_model_name_embeds_run_id_and_is_sanitized() -> None:
    cfg = _cfg(["arima_plus"])
    sql = build_create_model_sql(cfg, "arima_plus", _DS)
    # Model object name embeds the config-pinned run_id, hyphens folded to underscores.
    assert "`proj.scale_forecasting.sf_model_arima_plus_" in sql
    assert "-" not in sql.split("sf_model_arima_plus_")[1].split("`")[0]


# --- custom holidays -----------------------------------------------------------


def test_custom_holiday_cte_present_when_configured() -> None:
    cte = build_custom_holiday_cte(_cfg(["arima_plus"], holidays=["US"]))
    assert cte.startswith("custom_holiday AS (")
    assert "UNNEST([" in cte
    assert "AS region" in cte and "AS holiday_name" in cte and "AS primary_date" in cte
    assert "preholiday_days" in cte and "postholiday_days" in cte


def test_custom_holiday_names_are_valid_identifiers() -> None:
    cte = build_custom_holiday_cte(_cfg(["arima_plus"], holidays=["US"]))
    # Every holiday_name literal must be space-free (valid column name for ML.EXPLAIN_FORECAST).
    import re

    for name in re.findall(r"'([^']*)' AS holiday_name", cte):
        assert " " not in name and name


def test_custom_holiday_cte_absent_without_holidays() -> None:
    assert build_custom_holiday_cte(_cfg(["arima_plus"], holidays=[])) == ""
    # And the CREATE MODEL then uses the plain training query, no named subqueries.
    sql = build_create_model_sql(_cfg(["arima_plus"], holidays=[]), "arima_plus", _DS)
    assert "custom_holiday" not in sql
    assert "training_data AS" not in sql


# --- forecast INSERT -----------------------------------------------------------


def test_forecast_insert_aliases_and_engine_literal() -> None:
    sql = build_forecast_insert_sql(_cfg(["arima_plus"]), "arima_plus", _DS)
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
    sql = build_forecast_insert_sql(_cfg(["timesfm"]), "timesfm", _DS)
    assert "AI.FORECAST(" in sql
    assert "ML.FORECAST" not in sql  # TimesFM is serverless — no trained model object
    assert "data_col => 'y'" in sql
    assert "timestamp_col => 'ds'" in sql
    assert "id_cols => ['ts_id']" in sql
    assert "horizon => 28" in sql


def test_forecast_insert_is_true_future_not_held_out() -> None:
    # The final forecast INSERT reads from the all-history model — no held-out cutoff — so it
    # extrapolates beyond MAX(ds). ARIMA_PLUS owns its time axis (ML.FORECAST(horizon) suffices);
    # no future-dates input table is needed for the univariate natives.
    sql = build_forecast_insert_sql(_cfg(["arima_plus"]), "arima_plus", _DS)
    assert "DATE_SUB(MAX(ds)" not in sql
    # TimesFM forecasts from all history too (no ds <= cutoff bound on its inline history).
    tsql = build_forecast_insert_sql(_cfg(["timesfm"]), "timesfm", _DS)
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
    assert fold_plan(cfg) == [(0, 84), (1, 56), (2, 28)]


def test_the_native_plans_holdout_is_the_fold_the_python_engines_reserve() -> None:
    # The holdout is a pure function of the config (`backtest.holdout_fold_id`), and the native
    # plan does not carry a role of its own — it just numbers folds the same way. That agreement
    # is the whole reason a run can reserve one fold across two runtimes without comparing dates,
    # so it is asserted here rather than assumed by both sides.
    from scale_forecasting.backtest import holdout_fold_id

    for n_folds in (1, 2, 5):
        cfg = RunConfig(
            run_name="bq test",
            data={"source_table": "src"},
            models=["arima_plus"],
            backtest={"enabled": True, "n_folds": n_folds, "horizon": 28, "step": 28},
        )
        plan = fold_plan(cfg)
        assert plan[-1][0] == holdout_fold_id(cfg)
        # …and it is the newest window, not merely the last row of the list.
        assert plan[-1][1] == min(steps for _k, steps in plan)


def _bt_cfg(**backtest: Any) -> RunConfig:
    return RunConfig(
        run_name="bq test",
        data={"source_table": "src", "series_limit": None},
        models=["arima_plus"],
        backtest={"enabled": True, "n_folds": 3, "horizon": 28, "step": 28, **backtest},
    )


def test_the_native_cutoff_lands_exactly_where_the_python_fold_stops_training() -> None:
    """The parity that makes the embargo one feature rather than two implementations of it.

    Both engines are told "stop training ``gap`` observations before the validation window". The
    Python path says so in positions and the native path in a date offset from ``MAX(ds)``, and the
    two are the same statement exactly when ``back_steps == n - fold.train_end``. Asserting the
    identity over several gaps is cheaper than reasoning about it twice, and it is the assertion
    that fails if either side later grows an off-by-one.
    """
    n = 400
    for gap in (0, 1, 7, 30):
        cfg = _bt_cfg(gap=gap)
        folds = make_folds(n, cfg)
        assert len(folds) == 3  # long enough that none is clamped away
        assert fold_plan(cfg) == [(f.fold_id, n - f.train_end) for f in folds]


def test_a_native_fold_forecasts_across_the_embargo_and_scores_only_what_is_past_it() -> None:
    """BQML's forecast origin is the model's last training date and cannot be moved.

    So a fold under an embargo asks for ``gap + horizon`` steps and throws the first ``gap`` away —
    the same thing `backtest._forecast_validation` does in Python. Dropping them matters here in a
    way it does not there: the eval query *inner joins* the forecast to actuals on the date, so the
    embargo rows would find their actuals and be scored, quietly, as if they were the window.
    """
    sql = build_eval_query(_bt_cfg(gap=3), "arima_plus", _DS, back_steps=31, fold_id=0)
    assert "STRUCT(31 AS horizon" in sql  # gap + backtest.horizon
    assert "WHERE DATE_DIFF(DATE(f.forecast_timestamp), c.cutoff_date, DAY) > 3" in sql
    # …and what survives is numbered from the validation window, matching the Python OOF rows.
    assert "DATE_DIFF(DATE(f.forecast_timestamp), c.cutoff_date, DAY) - 3 AS horizon_step" in sql


def test_without_an_embargo_the_eval_query_is_byte_for_byte_what_it_always_was() -> None:
    """`gap=0` is the default, so the whole shipped corpus runs down this branch."""
    sql = build_eval_query(_bt_cfg(), "arima_plus", _DS, back_steps=28, fold_id=0)
    assert "STRUCT(28 AS horizon" in sql
    assert "WHERE DATE_DIFF" not in sql
    assert "DATE_DIFF(DATE(f.forecast_timestamp), c.cutoff_date, DAY) AS horizon_step" in sql


def test_a_native_sliding_window_is_window_wide_not_min_train_wide() -> None:
    cfg = _bt_cfg(scheme="sliding", min_train=180, window=60)
    sql = build_create_model_sql(cfg, "arima_plus", _DS, back_steps=28, fold_id=0)
    # lower bound = cutoff - window = MAX(ds) - (28 + 60), not - (28 + 180)
    assert "ds > (SELECT DATE_SUB(MAX(ds), INTERVAL 88 DAY)" in sql


def test_fold_create_and_drop_target_the_same_object() -> None:
    # Every fold trains a fold-suffixed object; the matching DROP must name that exact object so
    # backtest runs leave no orphaned sf_model_*_f{k} models behind.
    cfg = _cfg(["arima_plus"])
    create = build_fold_create_statements(cfg, "arima_plus", _DS, fold_id=1, back_steps=56)
    drop = build_fold_drop_statements(cfg, "arima_plus", _DS, fold_id=1)
    assert len(create) == 1 and len(drop) == 1
    obj = _model_ref(cfg, "arima_plus", _DS, fold_id=1)
    assert obj in create[0] and obj in drop[0]
    assert drop[0].startswith("DROP MODEL IF EXISTS ")  # safe if the fold CREATE failed


def test_fold_drop_never_targets_the_final_model() -> None:
    # The final true-future model (fold_id=None) backs forecast_predictions and must survive; only
    # fold-suffixed objects are dropped.
    cfg = _cfg(["arima_plus"])
    final_obj = _model_ref(cfg, "arima_plus", _DS)  # no fold suffix
    for k in range(3):
        drop = build_fold_drop_statements(cfg, "arima_plus", _DS, fold_id=k)
        assert final_obj not in drop[0]
        assert f"_f{k}`" in drop[0]


def test_timesfm_has_no_fold_create_or_drop() -> None:
    # TimesFM trains no model object (AI.FORECAST reads history directly), so it has neither a fold
    # CREATE nor a DROP — nothing to clean up.
    cfg = _cfg(["timesfm"])
    assert build_fold_create_statements(cfg, "timesfm", _DS, fold_id=0, back_steps=28) == []
    assert build_fold_drop_statements(cfg, "timesfm", _DS, fold_id=0) == []


# --- eval + history read-back --------------------------------------------------


def test_eval_query_joins_fold_forecast_to_actuals_with_intervals() -> None:
    sql = build_eval_query(_cfg(["arima_plus"]), "arima_plus", _DS, back_steps=28, fold_id=0)
    assert "AS y_true" in sql
    assert "AS yhat" in sql
    assert "AS yhat_lower" in sql and "AS yhat_upper" in sql
    assert "JOIN `proj.scale_forecasting.source_series_native`" in sql
    assert "DATE(f.forecast_timestamp)" in sql
    # Fold eval reads the fold-suffixed model over its held-out window (ds <= cutoff).
    assert "_f0`" in sql


def test_history_query_is_all_history() -> None:
    # MASE/RMSSE scale comes from the full series history (natives train on all of it), so the
    # history read is no longer clipped to a pre-cutoff window.
    sql = build_history_query(_cfg(["arima_plus"], series_limit=None), _DS)
    assert "AS ts_id" in sql and "AS y" in sql
    assert "DATE_SUB(MAX(ds)" not in sql


def test_series_ids_query_lists_the_subset() -> None:
    sql = build_series_ids_query(_cfg(["arima_plus"], series_limit=100), _DS)
    assert "SELECT DISTINCT ts_id AS ts_id" in sql
    assert "ORDER BY ts_id LIMIT 100" in sql


def test_series_count_query_counts_exactly_what_the_ids_query_lists() -> None:
    """The repair path's denominator. Two queries that disagree would invent or hide missing work.

    Pinned as a shared-fragment property rather than a string comparison: both render the same
    ``series_limit`` subquery and the same snapshot pin, so the only difference is the projection.
    """
    cfg = _cfg(["arima_plus"], series_limit=100)
    from scale_forecasting.engines.bigquery_sql import build_series_count_query

    count = build_series_count_query(cfg, _DS, snapshot_millis=_SNAP_MS)
    ids = build_series_ids_query(cfg, _DS, snapshot_millis=_SNAP_MS)
    assert "SELECT COUNT(DISTINCT ts_id) AS n_series" in count
    assert "ORDER BY ts_id LIMIT 100" in count
    # Same source, same pin, same subset filter — and the count never orders or lists.
    assert count.count(f"`{_SRC}{_SNAP}") == ids.count(f"`{_SRC}{_SNAP}") == 2
    assert "ORDER BY ts_id;" not in count


def test_series_count_query_drops_the_subset_when_there_is_no_limit() -> None:
    from scale_forecasting.engines.bigquery_sql import build_series_count_query

    sql = build_series_count_query(_cfg(["arima_plus"], series_limit=None), _DS)
    assert "LIMIT" not in sql and "FOR SYSTEM_TIME AS OF" not in sql


# --- series_limit subset -------------------------------------------------------


def test_series_limit_subset_present_and_omitted() -> None:
    limited = build_create_model_sql(_cfg(["arima_plus"], series_limit=100), "arima_plus", _DS)
    assert "ORDER BY ts_id LIMIT 100" in limited
    unlimited = build_create_model_sql(_cfg(["arima_plus"], series_limit=None), "arima_plus", _DS)
    assert "LIMIT" not in unlimited


# --- bqml_options / best_params ------------------------------------------------


def test_bqml_options_maps_columns() -> None:
    opts = bqml_options(_cfg(["arima_plus"]), "arima_plus")
    assert opts["model_type"] == "ARIMA_PLUS"
    assert opts["time_series_id_col"] == "ts_id"
    assert opts["time_series_timestamp_col"] == "ds"
    assert opts["time_series_data_col"] == "y"
    assert opts["horizon"] == 28


def test_bqml_options_timesfm_returns_ai_forecast_params() -> None:
    # TimesFM has no CREATE MODEL, but run() still stamps best_params for every model — so
    # bqml_options must resolve for it (not KeyError) and describe the AI.FORECAST call instead.
    opts = bqml_options(_cfg(["timesfm"]), "timesfm")
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
        render_setup_sql(cfg, m, _DS) for m in cfg.models
    )
    if os.environ.get("SF_UPDATE_SNAPSHOTS") == "1":
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(rendered)
    assert SNAPSHOT.exists(), "snapshot missing; run with SF_UPDATE_SNAPSHOTS=1 to create"
    assert rendered == SNAPSHOT.read_text()


# --- snapshot pinning (FOR SYSTEM_TIME AS OF) ----------------------------------
#
# Every source-table read in a run must time-travel to the one snapshot the run recorded, so all
# jobs read byte-identical input. The clause attaches OUTSIDE the backticks, once per source read;
# model objects and the registry tables are never time-travelled. Default (no snapshot) → no clause.

_SNAP_MS = 1_724_000_000_000
# Leading space + attached right after the source's closing backtick.
_SNAP = f"` FOR SYSTEM_TIME AS OF TIMESTAMP_MILLIS({_SNAP_MS})"
_SRC = "proj.scale_forecasting.source_series_native"


def test_snapshot_clause_absent_by_default() -> None:
    # No snapshot passed → SQL is byte-identical to the pre-snapshot behavior (no time-travel).
    cfg = _cfg(["arima_plus", "timesfm"], series_limit=100)
    for sql in (
        build_series_ids_query(cfg, _DS),
        build_history_query(cfg, _DS),
        build_create_model_sql(cfg, "arima_plus", _DS),
        build_forecast_insert_sql(cfg, "timesfm", _DS),
        build_eval_query(cfg, "arima_plus", _DS, back_steps=28, fold_id=0),
    ):
        assert "FOR SYSTEM_TIME AS OF" not in sql


def test_snapshot_clause_pins_series_ids_and_history_reads() -> None:
    cfg = _cfg(["arima_plus"], series_limit=100)
    # Both builders read the source twice with a limit (outer scan + the series-filter subquery).
    ids = build_series_ids_query(cfg, _DS, snapshot_millis=_SNAP_MS)
    hist = build_history_query(cfg, _DS, snapshot_millis=_SNAP_MS)
    assert ids.count(f"`{_SRC}{_SNAP}") == 2
    assert hist.count(f"`{_SRC}{_SNAP}") == 2


def test_snapshot_clause_pins_create_model_but_not_the_model_object() -> None:
    cfg = _cfg(["arima_plus"], series_limit=100)
    sql = build_create_model_sql(cfg, "arima_plus", _DS, snapshot_millis=_SNAP_MS)
    # Training scan + series-filter subquery both time-travel the source...
    assert sql.count(f"`{_SRC}{_SNAP}") == 2
    # ...and nothing else does: exactly those two source reads carry the clause, so the persisted
    # model object (sf_model_arima_plus_...) is never time-travelled.
    assert "sf_model_arima_plus" in sql
    assert sql.count("FOR SYSTEM_TIME AS OF") == 2


def test_snapshot_clause_pins_timesfm_forecast_source() -> None:
    # TimesFM reads history at forecast time (no CREATE MODEL), so its insert pins the source.
    cfg = _cfg(["timesfm"], series_limit=100)
    sql = build_forecast_insert_sql(cfg, "timesfm", _DS, snapshot_millis=_SNAP_MS)
    assert sql.count(f"`{_SRC}{_SNAP}") == 2  # inner history SELECT + series-filter subquery
    # The registry output table is never time-travelled — the clause follows the source only.
    assert "`proj.scale_forecasting.forecast_predictions` FOR SYSTEM_TIME" not in sql


def test_snapshot_clause_pins_arima_insert_reads_via_model_only() -> None:
    # ARIMA's final forecast reads from the (already-snapshotted) model object, not the source, so
    # its insert has no source read to pin — parity with the pre-snapshot SQL.
    cfg = _cfg(["arima_plus"], series_limit=100)
    sql = build_forecast_insert_sql(cfg, "arima_plus", _DS, snapshot_millis=_SNAP_MS)
    assert "FOR SYSTEM_TIME AS OF" not in sql


def test_snapshot_clause_pins_eval_join_actuals() -> None:
    cfg = _cfg(["arima_plus"], series_limit=100)
    sql = build_eval_query(
        cfg, "arima_plus", _DS, back_steps=28, fold_id=0, snapshot_millis=_SNAP_MS
    )
    # The actuals join reads the source and must time-travel with the rest of the run.
    assert f"`{_SRC}{_SNAP}" in sql
    # Twice, once for each source read: the actuals join and the cutoff-date scalar. Both are
    # reads of the source table, and a cutoff computed off an un-pinned `MAX(ds)` would name a
    # different fold from the one the model was trained for the moment a row lands mid-run.
    assert sql.count(f"`{_SRC}{_SNAP}") == 2
    # ML.FORECAST(MODEL ...) reads the fold model object, which is not time-travelled — so the
    # count above is the *total*: no third occurrence has crept in on the model reference.
    assert sql.count("FOR SYSTEM_TIME AS OF") == 2


def test_snapshot_clause_threads_through_setup_and_fold_builders() -> None:
    cfg = _cfg(["arima_plus"], series_limit=100)
    setup = build_setup_statements(cfg, "arima_plus", _DS, snapshot_millis=_SNAP_MS)
    fold = build_fold_create_statements(cfg, "arima_plus", _DS, 0, 28, snapshot_millis=_SNAP_MS)
    assert any("FOR SYSTEM_TIME AS OF" in s for s in setup)
    assert any(f"`{_SRC}{_SNAP}" in s for s in fold)
