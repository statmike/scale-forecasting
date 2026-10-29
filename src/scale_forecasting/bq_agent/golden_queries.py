"""Verified Golden Natural-Language <-> BigQuery SQL pairs for the BigQuery Registry Agent.

Every query in `build_golden_queries` is parameterized to a specific ``{project_id}.{dataset_id}``
registry (and optional primary ``source_table``) and follows three core discipline rules:

1. **Default to the latest COMPLETED run** when the question does not name a specific ``run_id`` or
   ask for cross-run history.
2. **Query the 11 pre-deduplicated ``v_*`` analytical views first**, and apply ``QUALIFY
   ROW_NUMBER()`` deduplication whenever reading raw ``forecast_predictions``,
   ``forecast_metadata``, or ``backtest_oof`` directly.
3. **Respect ``snapshot_millis`` temporal lineage**: use ``backtest_oof.y_true`` for historical
   backtest evaluation and ``source_table`` for post-run realized actuals arriving after launch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GoldenQuery:
    """One verified natural-language question and BigQuery Standard SQL answer template."""

    category: str
    question: str
    sql: str


def build_golden_queries(
    project_id: str,
    dataset_id: str,
    *,
    default_source_table: str | None = None,
) -> list[GoldenQuery]:
    """Render 32 verified `GoldenQuery` examples bound to ``{project_id}.{dataset_id}``."""
    ds = f"{project_id}.{dataset_id}"
    src = default_source_table or f"{ds}.source_series_covariates_native"

    latest_run_cte = (
        "WITH latest_run AS (\n"
        f"  SELECT run_id FROM `{ds}.v_run_summary`\n"
        "  ORDER BY IF(status = 'COMPLETED', 0, 1), created_at DESC LIMIT 1\n"
        ")\n"
    )

    return [
        # --- 1. Executive S&OP Summary & Run Overview (4) ---
        GoldenQuery(
            category="executive_summary",
            question="Give me the executive S&OP summary for the latest forecast run.",
            sql=(
                f"SELECT * FROM `{ds}.v_executive_sop_summary`\n"
                "ORDER BY IF(status = 'COMPLETED', 0, 1), created_at DESC\n"
                "LIMIT 1;"
            ),
        ),
        GoldenQuery(
            category="executive_summary",
            question=(
                "List the 10 most recent forecast runs with their status, series count, "
                "winning model, and runtime."
            ),
            sql=(
                "SELECT run_id, created_at, status, source_table, n_series, n_models, "
                "champion_model, champion_pooled_wape, fva_relative_lift_pct, runtime_seconds\n"
                f"FROM `{ds}.v_executive_sop_summary`\n"
                "ORDER BY created_at DESC\n"
                "LIMIT 10;"
            ),
        ),
        GoldenQuery(
            category="executive_summary",
            question=(
                "How much orchestration overhead did the latest run have compared to its "
                "longest family job?"
            ),
            sql=(
                "SELECT run_id, status, runtime_seconds, n_jobs, longest_job_seconds, "
                "jobs_span_seconds, overhead_seconds, overhead_fraction\n"
                f"FROM `{ds}.v_run_summary`\n"
                "ORDER BY IF(status = 'COMPLETED', 0, 1), created_at DESC\n"
                "LIMIT 1;"
            ),
        ),
        GoldenQuery(
            category="executive_summary",
            question=(
                "Which source tables have been forecasted in this registry and how many "
                "runs used each?"
            ),
            sql=(
                "SELECT source_table, COUNT(*) AS n_runs, MAX(created_at) AS latest_run_at, "
                "MIN(champion_pooled_wape) AS best_pooled_wape\n"
                f"FROM `{ds}.v_executive_sop_summary`\n"
                "GROUP BY source_table\n"
                "ORDER BY n_runs DESC;"
            ),
        ),
        # --- 2. Model & Ensemble Leaderboards (5) ---
        GoldenQuery(
            category="leaderboard",
            question="Which model or ensemble won on the latest run?",
            sql=(
                f"{latest_run_cte}"
                "SELECT c.run_id, c.model_type, c.n_series, c.pooled_wape, c.pooled_mae, "
                "c.first_forecast_date, c.last_forecast_date, c.refit_modes\n"
                f"FROM `{ds}.v_model_leaderboard_comparable` AS c\n"
                "JOIN latest_run USING (run_id)\n"
                "ORDER BY c.pooled_wape ASC NULLS LAST;"
            ),
        ),
        GoldenQuery(
            category="leaderboard",
            question="Compare ensemble strategies against single base models on the latest run.",
            sql=(
                f"{latest_run_cte}"
                "SELECT c.model_type, "
                "IF(c.ensemble_id IS NOT NULL OR STARTS_WITH(c.model_type, 'ensemble_'), "
                "'ensemble', 'base_model') AS tier, "
                "c.n_series, c.pooled_wape, c.pooled_mae\n"
                f"FROM `{ds}.v_model_leaderboard_comparable` AS c\n"
                "JOIN latest_run USING (run_id)\n"
                "ORDER BY c.pooled_wape ASC NULLS LAST;"
            ),
        ),
        GoldenQuery(
            category="leaderboard",
            question=(
                "Show per-model fit speed, training mode, and covariates used on the latest run."
            ),
            sql=(
                f"{latest_run_cte}"
                "SELECT l.model_type, l.compute_engine, l.training_mode, l.cov_profile, "
                "l.covariates_used, l.n_cells, l.median_fit_seconds, l.mean_wape, l.mean_mae\n"
                f"FROM `{ds}.v_model_leaderboard` AS l\n"
                "JOIN latest_run USING (run_id)\n"
                "ORDER BY l.mean_wape ASC NULLS LAST;"
            ),
        ),
        GoldenQuery(
            category="leaderboard",
            question="Did all models achieve full backtest fold coverage on the latest run?",
            sql=(
                f"{latest_run_cte}"
                "SELECT b.model_type, b.backtest_status, b.n_folds_achieved, b.backtest_refit, "
                "b.n_series, b.series_share, b.mean_staleness_gap\n"
                f"FROM `{ds}.v_backtest_coverage` AS b\n"
                "JOIN latest_run USING (run_id)\n"
                "ORDER BY b.model_type, b.n_folds_achieved DESC;"
            ),
        ),
        GoldenQuery(
            category="leaderboard",
            question="Which cells borrowed fitted results from a prior run via reuse_prior_runs?",
            sql=(
                f"{latest_run_cte}"
                "SELECT l.run_id, l.model_type, l.borrowed_from_run_id, l.n_borrowed, l.n_cells\n"
                f"FROM `{ds}.v_model_leaderboard` AS l\n"
                "JOIN latest_run USING (run_id)\n"
                "WHERE l.n_borrowed > 0\n"
                "ORDER BY l.n_borrowed DESC;"
            ),
        ),
        # --- 3. Forecast Value Added Waterfall (3) ---
        GoldenQuery(
            category="fva",
            question=(
                "Show the Forecast Value Added (FVA) waterfall from naive baseline to "
                "reconciled ensemble on the latest run."
            ),
            sql=(
                f"{latest_run_cte}"
                "SELECT f.stage_order, f.stage_name, f.model_type, f.pooled_wape, "
                "f.baseline_model, f.baseline_wape, f.fva_wape_delta, f.fva_relative_lift_pct\n"
                f"FROM `{ds}.v_fva_waterfall` AS f\n"
                "JOIN latest_run USING (run_id)\n"
                "ORDER BY f.stage_order ASC, f.pooled_wape ASC;"
            ),
        ),
        GoldenQuery(
            category="fva",
            question=(
                "Did adding exogenous covariates improve accuracy over univariate models "
                "on the latest run?"
            ),
            sql=(
                f"{latest_run_cte}"
                "SELECT f.stage_name, MIN(f.pooled_wape) AS best_stage_wape, "
                "AVG(f.pooled_wape) AS avg_stage_wape, "
                "MAX(f.fva_relative_lift_pct) AS max_fva_lift_pct\n"
                f"FROM `{ds}.v_fva_waterfall` AS f\n"
                "JOIN latest_run USING (run_id)\n"
                "WHERE f.stage_order IN (1, 2, 3, 4)\n"
                "GROUP BY f.stage_order, f.stage_name\n"
                "ORDER BY f.stage_order;"
            ),
        ),
        GoldenQuery(
            category="fva",
            question="Which models failed to beat the naive baseline on the latest run?",
            sql=(
                f"{latest_run_cte}"
                "SELECT f.model_type, f.stage_name, f.pooled_wape, f.baseline_model, "
                "f.baseline_wape, f.fva_relative_lift_pct\n"
                f"FROM `{ds}.v_fva_waterfall` AS f\n"
                "JOIN latest_run USING (run_id)\n"
                "WHERE f.fva_wape_delta < 0\n"
                "ORDER BY f.fva_wape_delta ASC;"
            ),
        ),
        # --- 4. Planner Bias & Exception Triage (5) ---
        GoldenQuery(
            category="exceptions",
            question="Which series need planner attention on the latest run and why?",
            sql=(
                "WITH champion AS (\n"
                "  SELECT run_id, champion_model\n"
                f"  FROM `{ds}.v_executive_sop_summary`\n"
                "  ORDER BY IF(status = 'COMPLETED', 0, 1), created_at DESC LIMIT 1\n"
                ")\n"
                "SELECT e.ts_id, e.model_type, e.sbc_class, e.exception_reason, "
                "e.tracking_signal, e.signed_bias, e.wape, e.mase\n"
                f"FROM `{ds}.v_bias_and_exceptions` AS e\n"
                "JOIN champion AS c ON e.run_id = c.run_id AND e.model_type = c.champion_model\n"
                "WHERE e.is_exception = TRUE\n"
                "ORDER BY ABS(e.tracking_signal) DESC NULLS LAST, e.wape DESC NULLS LAST\n"
                "LIMIT 50;"
            ),
        ),
        GoldenQuery(
            category="exceptions",
            question=(
                "Show the top 20 chronically over-forecasted series (tracking signal > 4) "
                "on the latest run."
            ),
            sql=(
                f"{latest_run_cte}"
                "SELECT e.ts_id, e.model_type, e.sbc_class, e.tracking_signal, e.signed_bias, "
                "e.mean_actual, e.mean_forecast, e.wape\n"
                f"FROM `{ds}.v_bias_and_exceptions` AS e\n"
                "JOIN latest_run USING (run_id)\n"
                "WHERE e.exception_reason = 'CHRONIC_OVER_FORECAST'\n"
                "ORDER BY e.tracking_signal DESC\n"
                "LIMIT 20;"
            ),
        ),
        GoldenQuery(
            category="exceptions",
            question=(
                "Show chronically under-forecasted series at risk of stockout on the latest run."
            ),
            sql=(
                f"{latest_run_cte}"
                "SELECT e.ts_id, e.model_type, e.sbc_class, e.tracking_signal, e.signed_bias, "
                "e.mean_actual, e.mean_forecast, e.wape\n"
                f"FROM `{ds}.v_bias_and_exceptions` AS e\n"
                "JOIN latest_run USING (run_id)\n"
                "WHERE e.exception_reason = 'CHRONIC_UNDER_FORECAST'\n"
                "ORDER BY e.tracking_signal ASC\n"
                "LIMIT 20;"
            ),
        ),
        GoldenQuery(
            category="exceptions",
            question=(
                "Summarize forecast accuracy and exception counts by SBC demand regime "
                "on the latest run."
            ),
            sql=(
                "WITH champion AS (\n"
                "  SELECT run_id, champion_model\n"
                f"  FROM `{ds}.v_executive_sop_summary`\n"
                "  ORDER BY IF(status = 'COMPLETED', 0, 1), created_at DESC LIMIT 1\n"
                ")\n"
                "SELECT COALESCE(e.sbc_class, 'unclassified') AS sbc_class, "
                "COUNT(*) AS n_series, COUNTIF(e.is_exception) AS n_exceptions, "
                "AVG(e.wape) AS avg_wape, AVG(e.mase) AS avg_mase\n"
                f"FROM `{ds}.v_bias_and_exceptions` AS e\n"
                "JOIN champion AS c ON e.run_id = c.run_id AND e.model_type = c.champion_model\n"
                "GROUP BY sbc_class\n"
                "ORDER BY n_series DESC;"
            ),
        ),
        GoldenQuery(
            category="exceptions",
            question=(
                "Were there any cell fit errors on the latest run and what error classes occurred?"
            ),
            sql=(
                f"{latest_run_cte}"
                "SELECT m.model_type, m.error_class, COUNT(*) AS n_failed_cells, "
                "ANY_VALUE(m.error_detail) AS sample_error\n"
                f"FROM `{ds}.forecast_metadata` AS m\n"
                "JOIN latest_run USING (run_id)\n"
                "WHERE m.fold_id IS NULL AND m.cell_status = 'error'\n"
                "GROUP BY m.model_type, m.error_class\n"
                "ORDER BY n_failed_cells DESC;"
            ),
        ),
        # --- 5. Forward Forecasts & Probabilistic Quantile Fans (4) ---
        GoldenQuery(
            category="forecasts",
            question=(
                "Show daily total forecasted demand and prediction intervals across the "
                "forward horizon for the winning model on the latest run."
            ),
            sql=(
                "WITH champion AS (\n"
                "  SELECT run_id, champion_model\n"
                f"  FROM `{ds}.v_executive_sop_summary`\n"
                "  ORDER BY IF(status = 'COMPLETED', 0, 1), created_at DESC LIMIT 1\n"
                "), deduped AS (\n"
                "  SELECT p.*\n"
                f"  FROM `{ds}.forecast_predictions` AS p\n"
                "  JOIN champion AS c ON p.run_id = c.run_id AND p.model_type = c.champion_model\n"
                "  QUALIFY ROW_NUMBER() OVER (\n"
                "    PARTITION BY p.run_id, p.ts_id, p.model_type, p.forecast_date, p.ensemble_id\n"
                "    ORDER BY p.created_at DESC NULLS LAST\n"
                "  ) = 1\n"
                ")\n"
                "SELECT forecast_date, COUNT(DISTINCT ts_id) AS n_series, "
                "SUM(yhat) AS total_yhat, SUM(yhat_lower) AS total_yhat_lower, "
                "SUM(yhat_upper) AS total_yhat_upper\n"
                "FROM deduped\n"
                "GROUP BY forecast_date\n"
                "ORDER BY forecast_date;"
            ),
        ),
        GoldenQuery(
            category="forecasts",
            question=(
                "Extract the P10, P50, and P90 quantile forecast fan for each series "
                "on the latest run."
            ),
            sql=(
                "WITH champion AS (\n"
                "  SELECT run_id, champion_model\n"
                f"  FROM `{ds}.v_executive_sop_summary`\n"
                "  ORDER BY IF(status = 'COMPLETED', 0, 1), created_at DESC LIMIT 1\n"
                ")\n"
                "SELECT p.ts_id, p.forecast_date, p.yhat, "
                "SAFE_CAST(JSON_VALUE(p.quantiles, '$.\"0.1\"') AS FLOAT64) AS p10, "
                "SAFE_CAST(JSON_VALUE(p.quantiles, '$.\"0.5\"') AS FLOAT64) AS p50, "
                "SAFE_CAST(JSON_VALUE(p.quantiles, '$.\"0.9\"') AS FLOAT64) AS p90\n"
                f"FROM `{ds}.forecast_predictions` AS p\n"
                "JOIN champion AS c ON p.run_id = c.run_id AND p.model_type = c.champion_model\n"
                "QUALIFY ROW_NUMBER() OVER (\n"
                "  PARTITION BY p.run_id, p.ts_id, p.model_type, p.forecast_date, p.ensemble_id\n"
                "  ORDER BY p.created_at DESC NULLS LAST\n"
                ") = 1\n"
                "ORDER BY p.ts_id, p.forecast_date\n"
                "LIMIT 200;"
            ),
        ),
        GoldenQuery(
            category="forecasts",
            question=(
                "Which 15 series have the highest total forecasted volume over the forward "
                "horizon on the latest run?"
            ),
            sql=(
                "WITH champion AS (\n"
                "  SELECT run_id, champion_model\n"
                f"  FROM `{ds}.v_executive_sop_summary`\n"
                "  ORDER BY IF(status = 'COMPLETED', 0, 1), created_at DESC LIMIT 1\n"
                "), deduped AS (\n"
                "  SELECT p.*\n"
                f"  FROM `{ds}.forecast_predictions` AS p\n"
                "  JOIN champion AS c ON p.run_id = c.run_id AND p.model_type = c.champion_model\n"
                "  QUALIFY ROW_NUMBER() OVER (\n"
                "    PARTITION BY p.run_id, p.ts_id, p.model_type, p.forecast_date, p.ensemble_id\n"
                "    ORDER BY p.created_at DESC NULLS LAST\n"
                "  ) = 1\n"
                ")\n"
                "SELECT ts_id, SUM(yhat) AS horizon_total_yhat, "
                "AVG(yhat_upper - yhat_lower) AS avg_interval_width\n"
                "FROM deduped\n"
                "GROUP BY ts_id\n"
                "ORDER BY horizon_total_yhat DESC\n"
                "LIMIT 15;"
            ),
        ),
        GoldenQuery(
            category="forecasts",
            question=(
                "Which series have the widest relative forecast uncertainty (interval width "
                "divided by yhat) on the latest run?"
            ),
            sql=(
                "WITH champion AS (\n"
                "  SELECT run_id, champion_model\n"
                f"  FROM `{ds}.v_executive_sop_summary`\n"
                "  ORDER BY IF(status = 'COMPLETED', 0, 1), created_at DESC LIMIT 1\n"
                "), deduped AS (\n"
                "  SELECT p.*\n"
                f"  FROM `{ds}.forecast_predictions` AS p\n"
                "  JOIN champion AS c ON p.run_id = c.run_id AND p.model_type = c.champion_model\n"
                "  QUALIFY ROW_NUMBER() OVER (\n"
                "    PARTITION BY p.run_id, p.ts_id, p.model_type, p.forecast_date, p.ensemble_id\n"
                "    ORDER BY p.created_at DESC NULLS LAST\n"
                "  ) = 1\n"
                ")\n"
                "SELECT ts_id, AVG(yhat) AS mean_yhat, "
                "AVG(yhat_upper - yhat_lower) AS mean_width, "
                "SAFE_DIVIDE(AVG(yhat_upper - yhat_lower), NULLIF(ABS(AVG(yhat)), 0.0)) "
                "AS relative_uncertainty\n"
                "FROM deduped\n"
                "GROUP BY ts_id\n"
                "ORDER BY relative_uncertainty DESC NULLS LAST\n"
                "LIMIT 20;"
            ),
        ),
        # --- 6. Explainability & Feature Attributions (3) ---
        GoldenQuery(
            category="explainability",
            question=(
                "What are the top exogenous covariates and lag features driving forecasts "
                "on the latest run?"
            ),
            sql=(
                f"{latest_run_cte}"
                "SELECT a.model_type, a.attribution_scope, a.feature_name, "
                "AVG(a.mean_attribution) AS avg_signed_impact, "
                "AVG(a.mean_abs_attribution) AS avg_abs_importance, "
                "COUNT(DISTINCT a.ts_id) AS n_series\n"
                f"FROM `{ds}.v_feature_attributions` AS a\n"
                "JOIN latest_run USING (run_id)\n"
                "GROUP BY a.model_type, a.attribution_scope, a.feature_name\n"
                "ORDER BY avg_abs_importance DESC\n"
                "LIMIT 25;"
            ),
        ),
        GoldenQuery(
            category="explainability",
            question=(
                "Show the local horizon feature attributions per series on the latest "
                "explainable run."
            ),
            sql=(
                "WITH latest_explainable AS (\n"
                f"  SELECT run_id FROM `{ds}.v_feature_attributions`\n"
                "  WHERE attribution_scope = 'horizon_local'\n"
                "  LIMIT 1\n"
                ")\n"
                "SELECT a.run_id, a.ts_id, a.model_type, a.feature_name, "
                "a.mean_attribution, a.mean_abs_attribution, a.n_horizon_steps\n"
                f"FROM `{ds}.v_feature_attributions` AS a\n"
                "JOIN latest_explainable USING (run_id)\n"
                "WHERE a.attribution_scope = 'horizon_local'\n"
                "ORDER BY a.mean_abs_attribution DESC\n"
                "LIMIT 50;"
            ),
        ),
        GoldenQuery(
            category="explainability",
            question=(
                "What ensemble blending weights or tuned best_params were learned on the "
                "latest run?"
            ),
            sql=(
                f"{latest_run_cte}"
                "SELECT m.ts_id, m.model_type, TO_JSON_STRING(m.best_params) AS best_params_json, "
                "m.wape, m.mae\n"
                f"FROM `{ds}.forecast_metadata` AS m\n"
                "JOIN latest_run USING (run_id)\n"
                "WHERE m.fold_id IS NULL AND m.best_params IS NOT NULL\n"
                "QUALIFY ROW_NUMBER() OVER (\n"
                "  PARTITION BY m.run_id, m.ts_id, m.model_type, m.fold_id, m.ensemble_id\n"
                "  ORDER BY m.created_at DESC NULLS LAST\n"
                ") = 1\n"
                "ORDER BY m.model_type, m.ts_id\n"
                "LIMIT 50;"
            ),
        ),
        # --- 7. Horizon Decay, Refit Cadence & Historical Anomalies (4) ---
        GoldenQuery(
            category="horizon_and_anomalies",
            question=(
                "How does forecast WAPE and interval coverage degrade across lead-time "
                "horizon steps on the latest run?"
            ),
            sql=(
                "WITH champion AS (\n"
                "  SELECT run_id, champion_model\n"
                f"  FROM `{ds}.v_executive_sop_summary`\n"
                "  ORDER BY IF(status = 'COMPLETED', 0, 1), created_at DESC LIMIT 1\n"
                ")\n"
                "SELECT d.model_type, d.horizon_step, d.n_series, d.pooled_wape, d.pooled_mae, "
                "d.signed_bias, d.empirical_coverage, d.mean_interval_width\n"
                f"FROM `{ds}.v_horizon_decay` AS d\n"
                "JOIN champion AS c ON d.run_id = c.run_id AND d.model_type = c.champion_model\n"
                "ORDER BY d.horizon_step ASC;"
            ),
        ),
        GoldenQuery(
            category="horizon_and_anomalies",
            question=(
                "At which horizon step does prediction interval coverage drop below 80% "
                "on the latest run?"
            ),
            sql=(
                f"{latest_run_cte}"
                "SELECT d.model_type, d.horizon_step, d.empirical_coverage, "
                "d.mean_interval_width, d.pooled_wape\n"
                f"FROM `{ds}.v_horizon_decay` AS d\n"
                "JOIN latest_run USING (run_id)\n"
                "WHERE d.empirical_coverage < 0.80\n"
                "ORDER BY d.model_type, d.horizon_step ASC;"
            ),
        ),
        GoldenQuery(
            category="horizon_and_anomalies",
            question=(
                "Show the largest historical demand anomalies and spikes detected during "
                "backtesting on the latest run."
            ),
            sql=(
                "WITH champion AS (\n"
                "  SELECT run_id, champion_model\n"
                f"  FROM `{ds}.v_executive_sop_summary`\n"
                "  ORDER BY IF(status = 'COMPLETED', 0, 1), created_at DESC LIMIT 1\n"
                ")\n"
                "SELECT a.ts_id, a.forecast_date, a.anomaly_type, a.y_true, a.yhat, "
                "a.residual, a.residual_zscore, a.outside_interval\n"
                f"FROM `{ds}.v_anomalies` AS a\n"
                "JOIN champion AS c ON a.run_id = c.run_id AND a.model_type = c.champion_model\n"
                "ORDER BY ABS(a.residual_zscore) DESC NULLS LAST\n"
                "LIMIT 30;"
            ),
        ),
        GoldenQuery(
            category="horizon_and_anomalies",
            question=(
                "Which models lose the most accuracy when frozen without per-fold refitting "
                "(highest staleness gap)?"
            ),
            sql=(
                "SELECT run_id, model_type, mean_staleness_gap, refit_modes, mean_wape\n"
                f"FROM `{ds}.v_model_leaderboard`\n"
                "WHERE mean_staleness_gap IS NOT NULL\n"
                "ORDER BY mean_staleness_gap DESC\n"
                "LIMIT 20;"
            ),
        ),
        # --- 8. Cloud FinOps Telemetry & Realized Accuracy vs. Source Tables (4) ---
        GoldenQuery(
            category="finops_and_realized",
            question=(
                "Break down cloud provisioning time versus active execution time by model "
                "family and runtime on the latest run."
            ),
            sql=(
                f"{latest_run_cte}"
                "SELECT j.family, j.runtime, j.spark_mode, j.hardware, j.gpu_type, j.status, "
                "j.runtime_seconds, j.provision_seconds, j.execute_seconds, j.device_verdict\n"
                f"FROM `{ds}.v_run_jobs` AS j\n"
                "JOIN latest_run USING (run_id)\n"
                "ORDER BY j.runtime_seconds DESC NULLS LAST;"
            ),
        ),
        GoldenQuery(
            category="finops_and_realized",
            question=(
                "Did any GPU jobs across the registry waste accelerator hardware without "
                "utilizing CUDA?"
            ),
            sql=(
                "SELECT run_id, family, runtime, gpu_type, runtime_seconds, device_verdict, "
                "TO_JSON_STRING(device_use) AS device_use_json\n"
                f"FROM `{ds}.v_run_jobs`\n"
                "WHERE hardware = 'gpu' AND device_verdict != 'ENGAGED_UTILISED'\n"
                "ORDER BY created_at DESC;"
            ),
        ),
        GoldenQuery(
            category="finops_and_realized",
            question=(
                "Roll up forward forecasts on the latest run by source hierarchy attributes "
                "(region and category)."
            ),
            sql=(
                "WITH champion AS (\n"
                "  SELECT run_id, champion_model\n"
                f"  FROM `{ds}.v_executive_sop_summary`\n"
                "  ORDER BY IF(status = 'COMPLETED', 0, 1), created_at DESC LIMIT 1\n"
                "), attrs AS (\n"
                f"  SELECT DISTINCT ts_id, archetype FROM `{src}`\n"
                "), deduped AS (\n"
                "  SELECT p.*\n"
                f"  FROM `{ds}.forecast_predictions` AS p\n"
                "  JOIN champion AS c ON p.run_id = c.run_id AND p.model_type = c.champion_model\n"
                "  QUALIFY ROW_NUMBER() OVER (\n"
                "    PARTITION BY p.run_id, p.ts_id, p.model_type, p.forecast_date, p.ensemble_id\n"
                "    ORDER BY p.created_at DESC NULLS LAST\n"
                "  ) = 1\n"
                ")\n"
                "SELECT COALESCE(a.archetype, 'aggregate_or_unknown') AS segment, "
                "COUNT(DISTINCT d.ts_id) AS n_series, SUM(d.yhat) AS total_forecast_yhat\n"
                "FROM deduped AS d\n"
                "LEFT JOIN attrs AS a USING (ts_id)\n"
                "GROUP BY segment\n"
                "ORDER BY total_forecast_yhat DESC;"
            ),
        ),
        GoldenQuery(
            category="finops_and_realized",
            question=(
                "Compare forward forecasts from a completed run against fresh realized "
                "actuals in the source table."
            ),
            sql=(
                "WITH target_run AS (\n"
                "  SELECT run_id, snapshot_millis\n"
                f"  FROM `{ds}.run_registry`\n"
                "  WHERE status = 'COMPLETED'\n"
                "  QUALIFY ROW_NUMBER() OVER (PARTITION BY run_id ORDER BY created_at DESC) = 1\n"
                "  ORDER BY created_at DESC LIMIT 1\n"
                "), preds AS (\n"
                "  SELECT p.run_id, p.ts_id, p.model_type, p.forecast_date, p.yhat\n"
                f"  FROM `{ds}.forecast_predictions` AS p\n"
                "  JOIN target_run AS r USING (run_id)\n"
                "  QUALIFY ROW_NUMBER() OVER (\n"
                "    PARTITION BY p.run_id, p.ts_id, p.model_type, p.forecast_date, p.ensemble_id\n"
                "    ORDER BY p.created_at DESC NULLS LAST\n"
                "  ) = 1\n"
                ")\n"
                "SELECT p.model_type, COUNT(*) AS n_realized_points, "
                "SAFE_DIVIDE(SUM(ABS(s.y - p.yhat)), SUM(ABS(s.y))) AS realized_wape, "
                "AVG(ABS(s.y - p.yhat)) AS realized_mae\n"
                "FROM preds AS p\n"
                f"JOIN `{src}` AS s ON p.ts_id = s.ts_id AND p.forecast_date = s.ds\n"
                "GROUP BY p.model_type\n"
                "ORDER BY realized_wape ASC NULLS LAST;"
            ),
        ),
    ]


def format_example_queries_for_context(
    queries: list[GoldenQuery],
) -> list[dict[str, Any]]:
    """Format `GoldenQuery` list into `geminidataanalytics` ``Context.exampleQueries`` JSON."""
    return [
        {
            "naturalLanguageQuestion": q.question,
            "sqlQuery": q.sql,
        }
        for q in queries
    ]
