CREATE TABLE IF NOT EXISTS `proj.scale_forecasting.run_registry` (
  run_id            STRING NOT NULL,
  created_at        TIMESTAMP NOT NULL,
  user_id           STRING,
  git_sha           STRING,
  python_runtime    STRING,
  spark_method      STRING,
  bq_models         ARRAY<STRING>,
  backtest_on       BOOL,
  decision_metric   STRING,
  ensemble_strategies ARRAY<STRING>,
  raw_config        STRING NOT NULL,
  status            STRING,
  n_series          INT64,
  n_models          INT64,
  runtime_seconds   FLOAT64
)
PARTITION BY DATE(created_at)
CLUSTER BY run_id
WITH CONNECTION `proj.us-central1.sf-conn`
OPTIONS (
  file_format = 'PARQUET',
  table_format = 'ICEBERG',
  storage_uri = 'gs://proj-wh/warehouse/run_registry'
);

CREATE TABLE IF NOT EXISTS `proj.scale_forecasting.forecast_metadata` (
  run_id         STRING NOT NULL,
  ts_id          STRING NOT NULL,
  model_type     STRING NOT NULL,
  compute_engine STRING NOT NULL,
  model_hash     STRING NOT NULL,
  fold_id        INT64,
  mae FLOAT64, rmse FLOAT64, mse FLOAT64, mape FLOAT64, smape FLOAT64,
  wape FLOAT64, mase FLOAT64, rmsse FLOAT64, bias FLOAT64,
  coverage FLOAT64, pinball FLOAT64,
  fit_seconds    FLOAT64,
  best_params    STRING,
  model_artifact STRING,
  created_at     TIMESTAMP NOT NULL
)
PARTITION BY DATE(created_at)
CLUSTER BY run_id, model_type
WITH CONNECTION `proj.us-central1.sf-conn`
OPTIONS (
  file_format = 'PARQUET',
  table_format = 'ICEBERG',
  storage_uri = 'gs://proj-wh/warehouse/forecast_metadata'
);

CREATE TABLE IF NOT EXISTS `proj.scale_forecasting.forecast_predictions` (
  run_id        STRING NOT NULL,
  ts_id         STRING NOT NULL,
  model_type    STRING NOT NULL,
  compute_engine STRING,
  forecast_date DATE NOT NULL,
  yhat          FLOAT64,
  yhat_lower    FLOAT64,
  yhat_upper    FLOAT64,
  quantiles     STRING
)
PARTITION BY forecast_date
CLUSTER BY run_id, ts_id
WITH CONNECTION `proj.us-central1.sf-conn`
OPTIONS (
  file_format = 'PARQUET',
  table_format = 'ICEBERG',
  storage_uri = 'gs://proj-wh/warehouse/forecast_predictions'
);

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
CLUSTER BY run_id, ts_id
WITH CONNECTION `proj.us-central1.sf-conn`
OPTIONS (
  file_format = 'PARQUET',
  table_format = 'ICEBERG',
  storage_uri = 'gs://proj-wh/warehouse/backtest_oof'
);

CREATE TABLE IF NOT EXISTS `proj.scale_forecasting.source_series` (
  ts_id       STRING NOT NULL,
  ds          DATE NOT NULL,
  y           FLOAT64,
  archetype   STRING,
  price_index FLOAT64,
  is_holiday  BOOL
)
PARTITION BY ds
CLUSTER BY ts_id
WITH CONNECTION `proj.us-central1.sf-conn`
OPTIONS (
  file_format = 'PARQUET',
  table_format = 'ICEBERG',
  storage_uri = 'gs://proj-wh/warehouse/source_series'
);