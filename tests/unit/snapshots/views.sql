CREATE OR REPLACE VIEW `proj.scale_forecasting.v_run_summary` AS
SELECT
  run_id,
  created_at,
  status,
  python_runtime,
  n_series,
  n_models,
  backtest_on,
  runtime_seconds,
  CAST(JSON_VALUE(job_telemetry, '$.total_wall_s') AS FLOAT64) AS total_wall_s,
  CAST(JSON_VALUE(job_telemetry, '$.total_wall_s') AS FLOAT64)
    - runtime_seconds AS overhead_seconds,
  SAFE_DIVIDE(
    CAST(JSON_VALUE(job_telemetry, '$.total_wall_s') AS FLOAT64) - runtime_seconds,
    CAST(JSON_VALUE(job_telemetry, '$.total_wall_s') AS FLOAT64)
  ) AS overhead_fraction,
  CAST(JSON_VALUE(job_telemetry, '$.executor_instances') AS INT64) AS executor_instances,
  CAST(JSON_VALUE(job_telemetry, '$.executor_cores') AS INT64) AS executor_cores,
  CAST(JSON_VALUE(job_telemetry, '$.max_executors') AS INT64) AS max_executors,
  JSON_VALUE(job_telemetry, '$.executor_memory') AS executor_memory,
  JSON_VALUE(job_telemetry, '$.executor_memory_overhead') AS executor_memory_overhead,
  CAST(JSON_VALUE(job_telemetry, '$.dcu_milli_seconds') AS INT64) AS dcu_milli_seconds,
  JSON_VALUE(job_telemetry, '$.runtime_version') AS runtime_version,
  JSON_QUERY(job_telemetry, '$.sizing') AS sizing,
  JSON_QUERY(job_telemetry, '$.capacity') AS capacity
FROM `proj.scale_forecasting.run_registry`
QUALIFY ROW_NUMBER() OVER (PARTITION BY run_id ORDER BY created_at DESC NULLS LAST) = 1;

CREATE OR REPLACE VIEW `proj.scale_forecasting.v_run_jobs` AS
SELECT
  run_id,
  family,
  job_id,
  attempt,
  runtime,
  spark_mode,
  hardware,
  gpu_type,
  system_job_id,
  status,
  created_at,
  started_at,
  ended_at,
  runtime_seconds,
  failure_reason,
  CAST(JSON_VALUE(job_telemetry, '$.total_wall_s') AS FLOAT64) AS total_wall_s,
  CAST(JSON_VALUE(job_telemetry, '$.dcu_milli_seconds') AS INT64) AS dcu_milli_seconds,
  JSON_VALUE(job_telemetry, '$.device_use.verdict') AS device_verdict,
  JSON_QUERY(job_telemetry, '$.device_use') AS device_use,
  JSON_QUERY(job_telemetry, '$.probe_handle') AS probe_handle,
  JSON_QUERY(job_telemetry, '$.capacity') AS capacity
FROM `proj.scale_forecasting.run_jobs`
QUALIFY ROW_NUMBER() OVER (
  PARTITION BY run_id, family ORDER BY attempt DESC, created_at DESC
) = 1;

CREATE OR REPLACE VIEW `proj.scale_forecasting.v_model_leaderboard` AS
WITH deduped AS (
  SELECT *
  FROM `proj.scale_forecasting.forecast_metadata`
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY run_id, ts_id, model_type, fold_id, ensemble_id
    ORDER BY created_at DESC NULLS LAST
  ) = 1
)
SELECT
  run_id,
  model_type,
  ensemble_id,
  ANY_VALUE(compute_engine) AS compute_engine,
  COUNT(*) AS n_cells,
  COUNTIF(model_artifact IS NULL) AS n_no_artifact,
  SAFE_DIVIDE(COUNTIF(model_artifact IS NULL), COUNT(*)) AS no_artifact_rate,
  APPROX_QUANTILES(fit_seconds, 2)[OFFSET(1)] AS median_fit_seconds,
  AVG(wape) AS mean_wape,
  AVG(mae) AS mean_mae,
  AVG(staleness_gap) AS mean_staleness_gap,
  STRING_AGG(DISTINCT backtest_refit ORDER BY backtest_refit) AS refit_modes
FROM deduped
WHERE fold_id IS NULL
GROUP BY run_id, model_type, ensemble_id;

CREATE OR REPLACE VIEW `proj.scale_forecasting.v_backtest_coverage` AS
WITH deduped AS (
  SELECT *
  FROM `proj.scale_forecasting.forecast_metadata`
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY run_id, ts_id, model_type, fold_id, ensemble_id
    ORDER BY created_at DESC NULLS LAST
  ) = 1
)
SELECT
  run_id,
  model_type,
  ensemble_id,
  backtest_status,
  n_folds_achieved,
  backtest_refit,
  COUNT(*) AS n_series,
  AVG(staleness_gap) AS mean_staleness_gap,
  SAFE_DIVIDE(
    COUNT(*),
    SUM(COUNT(*)) OVER (PARTITION BY run_id, model_type, ensemble_id)
  ) AS series_share
FROM deduped
WHERE fold_id IS NULL
GROUP BY run_id, model_type, ensemble_id, backtest_status, n_folds_achieved, backtest_refit;

CREATE OR REPLACE VIEW `proj.scale_forecasting.v_model_leaderboard_comparable` AS
WITH deduped AS (
  SELECT *
  FROM `proj.scale_forecasting.backtest_oof`
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY run_id, ts_id, model_type, fold_id, forecast_date, ensemble_id
    ORDER BY created_at DESC NULLS LAST
  ) = 1
),
holdout AS (
  SELECT *
  FROM deduped
  QUALIFY fold_id = MAX(fold_id) OVER (PARTITION BY run_id)
),
refit AS (
  SELECT
    run_id,
    model_type,
    ensemble_id,
    STRING_AGG(DISTINCT backtest_refit ORDER BY backtest_refit) AS refit_modes
  FROM `proj.scale_forecasting.forecast_metadata`
  WHERE fold_id IS NULL
  GROUP BY run_id, model_type, ensemble_id
)
SELECT
  h.run_id,
  h.model_type,
  h.ensemble_id,
  ANY_VALUE(h.fold_id) AS holdout_fold_id,
  COUNT(DISTINCT h.ts_id) AS n_series,
  COUNT(*) AS n_points,
  SAFE_DIVIDE(SUM(ABS(h.y_true - h.yhat)), SUM(ABS(h.y_true))) AS pooled_wape,
  AVG(ABS(h.y_true - h.yhat)) AS pooled_mae,
  MIN(h.forecast_date) AS first_forecast_date,
  MAX(h.forecast_date) AS last_forecast_date,
  ANY_VALUE(r.refit_modes) AS refit_modes
FROM holdout AS h
LEFT JOIN refit AS r
  ON h.run_id = r.run_id
  AND h.model_type = r.model_type
  AND COALESCE(h.ensemble_id, '') = COALESCE(r.ensemble_id, '')
WHERE h.y_true IS NOT NULL AND h.yhat IS NOT NULL
GROUP BY h.run_id, h.model_type, h.ensemble_id;