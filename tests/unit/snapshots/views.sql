CREATE OR REPLACE VIEW `proj.scale_forecasting.v_run_summary` AS
SELECT
  run_id,
  created_at,
  status,
  spark_method,
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
  CAST(JSON_VALUE(job_telemetry, '$.dcu_milli_seconds') AS INT64) AS dcu_milli_seconds,
  JSON_VALUE(job_telemetry, '$.runtime_version') AS runtime_version
FROM `proj.scale_forecasting.run_registry`;

CREATE OR REPLACE VIEW `proj.scale_forecasting.v_model_leaderboard` AS
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
  AVG(mae) AS mean_mae
FROM `proj.scale_forecasting.forecast_metadata`
WHERE fold_id IS NULL
GROUP BY run_id, model_type, ensemble_id;