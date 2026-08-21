CREATE TABLE IF NOT EXISTS `proj.scale_forecasting.run_registry` (
  run_id            STRING NOT NULL,
  created_at        TIMESTAMP NOT NULL,
  user_id           STRING,
  git_sha           STRING,
  python_runtime    STRING,
  bq_models         ARRAY<STRING>,
  backtest_on       BOOL,
  decision_metric   STRING,
  ensemble_strategies ARRAY<STRING>,
  raw_config        JSON NOT NULL,
  status            STRING,
  n_series          INT64,
  n_models          INT64,
  runtime_seconds   FLOAT64,
  job_telemetry     JSON
)
PARTITION BY DATE(created_at)
CLUSTER BY run_id;

CREATE TABLE IF NOT EXISTS `proj.scale_forecasting.run_jobs` (
  job_id           STRING NOT NULL,
  run_id           STRING NOT NULL,
  family           STRING NOT NULL,
  attempt          INT64 NOT NULL,
  runtime          STRING,
  spark_mode       STRING,
  hardware         STRING,
  gpu_type         STRING,
  system_job_id    STRING,
  status           STRING,
  created_at       TIMESTAMP NOT NULL,
  runtime_seconds  FLOAT64,
  job_telemetry    JSON
)
PARTITION BY DATE(created_at)
CLUSTER BY run_id, family;

CREATE TABLE IF NOT EXISTS `proj.scale_forecasting.forecast_metadata` (
  run_id         STRING NOT NULL,
  ts_id          STRING NOT NULL,
  model_type     STRING NOT NULL,
  compute_engine STRING NOT NULL,
  model_hash     STRING NOT NULL,
  ensemble_id    STRING,
  fold_id        INT64,
  mae FLOAT64, rmse FLOAT64, mse FLOAT64, mape FLOAT64, smape FLOAT64,
  wape FLOAT64, mase FLOAT64, rmsse FLOAT64, bias FLOAT64,
  coverage FLOAT64, pinball FLOAT64,
  fit_seconds    FLOAT64,
  best_params    JSON,
  model_artifact STRING,
  created_at     TIMESTAMP NOT NULL
)
PARTITION BY DATE(created_at)
CLUSTER BY run_id, model_type;

CREATE TABLE IF NOT EXISTS `proj.scale_forecasting.forecast_predictions` (
  run_id        STRING NOT NULL,
  ts_id         STRING NOT NULL,
  model_type    STRING NOT NULL,
  compute_engine STRING,
  ensemble_id   STRING,
  forecast_date DATE NOT NULL,
  yhat          FLOAT64,
  yhat_lower    FLOAT64,
  yhat_upper    FLOAT64,
  quantiles     JSON
)
PARTITION BY forecast_date
CLUSTER BY run_id, ts_id;

CREATE TABLE IF NOT EXISTS `proj.scale_forecasting.backtest_oof` (
  run_id        STRING NOT NULL,
  ts_id         STRING NOT NULL,
  model_type    STRING NOT NULL,
  fold_id       INT64 NOT NULL,
  forecast_date DATE NOT NULL,
  y_true        FLOAT64,
  yhat          FLOAT64
)
PARTITION BY forecast_date
CLUSTER BY run_id, ts_id;

CREATE TABLE IF NOT EXISTS `proj.scale_forecasting.source_series_iceberg` (
  ts_id       STRING NOT NULL,
  ds          DATE NOT NULL,
  y           FLOAT64,
  archetype   STRING,
  is_holiday  BOOL
)
PARTITION BY ds
CLUSTER BY ts_id
WITH CONNECTION `proj.us-central1.sf-conn`
OPTIONS (
  file_format = 'PARQUET',
  table_format = 'ICEBERG',
  storage_uri = 'gs://proj-wh/warehouse/source_series_iceberg'
);

CREATE TABLE IF NOT EXISTS `proj.scale_forecasting.source_series_native` (
  ts_id       STRING NOT NULL,
  ds          DATE NOT NULL,
  y           FLOAT64,
  archetype   STRING,
  is_holiday  BOOL
)
PARTITION BY ds
CLUSTER BY ts_id;