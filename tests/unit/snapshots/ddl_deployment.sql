CREATE TABLE IF NOT EXISTS `proj.scale_forecasting.run_registry` (
  run_id            STRING NOT NULL OPTIONS(description='Deterministic content-addressed run identifier (<run_name_slug>-<12hex>) hashed from semantic RunConfig inputs.'),
  created_at        TIMESTAMP NOT NULL OPTIONS(description='UTC timestamp when this run header row was written (partition key; latest row per run_id wins).'),
  snapshot_millis   INT64 OPTIONS(description='BigQuery epoch milliseconds snapshot anchor recorded at run start for reproducible FOR SYSTEM_TIME AS OF reads.'),
  user_id           STRING OPTIONS(description='Identity of the principal or service account that launched the run.'),
  git_sha           STRING OPTIONS(description='Git commit SHA of the scale-forecasting code that launched the run.'),
  python_runtime    STRING OPTIONS(description='Python version string of the launching orchestrator.'),
  bq_models         ARRAY<STRING> OPTIONS(description='BQML model FQDNs created during native BigQuery runs (tracked for drop-run and orphan cleanup).'),
  backtest_on       BOOL OPTIONS(description='True when rolling/expanding walk-forward backtesting was enabled for this run.'),
  decision_metric   STRING OPTIONS(description='Primary metric name (from METRIC_NAMES, e.g. wape, mase) used for HPO, champion selection, and inverse-error weights.'),
  ensemble_strategies ARRAY<STRING> OPTIONS(description='Configured ensemble blending/stacking strategies (e.g. mean, inverse_error, champion, nnls, ridge, xgb).'),
  raw_config        JSON NOT NULL OPTIONS(description='Complete validated RunConfig JSON payload. Key paths: $.data.source_table, $.data.horizon, $.models, $.hierarchy.cols.'),
  status            STRING OPTIONS(description='Lifecycle status of the run: STAGED, RUNNING, COMPLETED, PARTIAL, FAILED, or CANCELLED.'),
  n_series          INT64 OPTIONS(description='Total number of distinct time series (ts_id) scored in this run.'),
  n_models          INT64 OPTIONS(description='Number of configured base models/variants in this run.'),
  runtime_seconds   FLOAT64 OPTIONS(description='End-to-end orchestrator wall-clock duration in seconds.'),
  job_telemetry     JSON OPTIONS(description='Run-level telemetry JSON: $.sizing (per-family worker/machine decisions), $.capacity (failover ledger), $.version.')
)
PARTITION BY DATE(created_at)
CLUSTER BY run_id
OPTIONS (
  description = 'scale-forecasting run header ledger (append-only, partitioned by created_at). Records deterministic run_id, BigQuery snapshot_millis time-travel anchor, verbatim RunConfig JSON ($.data.source_table, $.models, $.backtest), status, panel scale, and run-level sizing/capacity telemetry. Deduplicate to latest row per run_id via QUALIFY ROW_NUMBER() OVER (PARTITION BY run_id ORDER BY created_at DESC) = 1, or query v_run_summary.'
);

CREATE TABLE IF NOT EXISTS `proj.scale_forecasting.run_jobs` (
  job_id           STRING NOT NULL OPTIONS(description='Deterministic job identifier (<run_id>-<family>-a<attempt>) used to deduplicate cloud submissions.'),
  run_id           STRING NOT NULL OPTIONS(description='Parent run_id joining to run_registry.run_id.'),
  family           STRING NOT NULL OPTIONS(description='Model family or barrier stage executed by this job: statistical, ml, deep_learning, automl, native, or ensemble.'),
  attempt          INT64 NOT NULL OPTIONS(description='1-based execution attempt counter for (run_id, family); highest attempt is current.'),
  runtime          STRING OPTIONS(description='Compute engine that executed this family: spark, ray, vertex, gce, gke, vertex_automl, or bigquery.'),
  spark_mode       STRING OPTIONS(description='Spark execution mode when runtime=\'spark\' (serverless, cluster, connect, or local); NULL otherwise.'),
  hardware         STRING OPTIONS(description='Requested hardware tier for this job: cpu or gpu.'),
  gpu_type         STRING OPTIONS(description='Accelerator type when hardware=\'gpu\': T4, L4, A100, or A100_80GB; NULL on CPU.'),
  system_job_id    STRING OPTIONS(description='Cloud provider resource name/ID (Dataproc batch ID, Vertex CustomJob, GCE instance, GKE Job, or BQ job prefix).'),
  status           STRING OPTIONS(description='Job execution status: EMITTED, RUNNING, COMPLETED, PARTIAL, FAILED, or CANCELLED.'),
  created_at       TIMESTAMP NOT NULL OPTIONS(description='UTC timestamp when this job trace row was written.'),
  started_at       TIMESTAMP OPTIONS(description='UTC timestamp when job submission began.'),
  ended_at         TIMESTAMP OPTIONS(description='UTC timestamp when the job reached a terminal state.'),
  runtime_seconds  FLOAT64 OPTIONS(description='Launcher-measured wall-clock seconds from submission to job completion.'),
  failure_reason   STRING OPTIONS(description='Structured failure code and diagnostic summary when status=\'FAILED\' (e.g. CAPACITY_EXHAUSTED); NULL on success.'),
  job_telemetry    JSON OPTIONS(description='Per-job telemetry JSON: $.platform.provision_s, $.platform.execute_s, $.device_use.verdict, $.probe_handle, $.capacity.')
)
PARTITION BY DATE(created_at)
CLUSTER BY run_id, family
OPTIONS (
  description = 'Per-family cloud job execution trace (append-only, 1 row per attempt of statistical, ml, deep_learning, automl, native, or ensemble). Records resolved compute runtime, hardware/GPU shape, platform job ID, wall-clock timestamps, failure_reason, and job_telemetry JSON ($.platform.provision_s, $.platform.execute_s, $.device_use.verdict). Query v_run_jobs for the current attempt per (run_id, family).'
);

CREATE TABLE IF NOT EXISTS `proj.scale_forecasting.forecast_metadata` (
  run_id         STRING NOT NULL OPTIONS(description='Parent run_id joining to run_registry.run_id.'),
  ts_id          STRING NOT NULL OPTIONS(description='Time-series identifier scored in this cell.'),
  model_type     STRING NOT NULL OPTIONS(description='Model or variant identifier (e.g. theta, lightgbm@monotone, ensemble_champion, lightgbm_cross_temporal_mint).'),
  compute_engine STRING NOT NULL OPTIONS(description='Compute runtime that produced this cell (spark, ray, vertex, gce, gke, vertex_automl, bigquery, or ensemble).'),
  model_hash     STRING NOT NULL OPTIONS(description='Deterministic hash of model_type and hyperparameters.'),
  ensemble_id    STRING OPTIONS(description='Configuration digest for ensemble_* pseudo-models; NULL for base models.'),
  fold_id        INT64 OPTIONS(description='NULL for the full-fit summary row (one per cell); 0..n_folds-1 for individual walk-forward backtest fold rows.'),
  mae FLOAT64 OPTIONS(description='Mean Absolute Error: MEAN(|yhat - y_true|) in target units (lower is better).'),
  rmse FLOAT64 OPTIONS(description='Root Mean Squared Error: SQRT(MEAN((yhat - y_true)^2)) in target units (lower is better).'),
  mse FLOAT64 OPTIONS(description='Mean Squared Error: MEAN((yhat - y_true)^2) in squared target units (lower is better).'),
  mape FLOAT64 OPTIONS(description='Mean Absolute Percentage Error: MEAN(|yhat - y_true| / |y_true|); NULL/NaN when any y_true = 0 (lower is better).'),
  smape FLOAT64 OPTIONS(description='Symmetric Mean Absolute Percentage Error: MEAN(2 * |yhat - y_true| / (|y_true| + |yhat|)), bounded in [0, 2] (lower is better).'),
  wape FLOAT64 OPTIONS(description='Weighted Absolute Percentage Error: SUM(|yhat - y_true|) / SUM(|y_true|) across backtest folds (lower is better; 0.08 = 8% error).'),
  mase FLOAT64 OPTIONS(description='Mean Absolute Scaled Error vs. one-step naive random walk on training history (< 1.0 beats naive; lower is better).'),
  rmsse FLOAT64 OPTIONS(description='Root Mean Squared Scaled Error (M5 metric) vs. one-step naive squared step on training history (< 1.0 beats naive; lower is better).'),
  bias FLOAT64 OPTIONS(description='Mean signed forecast error MEAN(yhat - y_true); positive = over-forecasting, negative = under-forecasting (closer to 0 is better).'),
  coverage FLOAT64 OPTIONS(description='Empirical prediction interval coverage: fraction of y_true inside [yhat_lower, yhat_upper] (closer to nominal confidence_level e.g. 0.80 is better).'),
  pinball FLOAT64 OPTIONS(description='Mean pinball (quantile) loss across the lower and upper prediction interval bounds (lower is better).'),
  mase_seasonal FLOAT64 OPTIONS(description='Seasonal Mean Absolute Scaled Error vs. seasonal naive (lag = seasonal_period) on training history (< 1.0 beats seasonal naive; lower is better).'),
  maape FLOAT64 OPTIONS(description='Mean Arctangent Absolute Percentage Error: MEAN(ARCTAN(|yhat - y_true| / |y_true|)), finite even when y_true = 0 (lower is better).'),
  interval_score FLOAT64 OPTIONS(description='Mean Winkler interval score penalizing both interval width and out-of-bounds exceedance (lower is better).'),
  interval_width FLOAT64 OPTIONS(description='Mean prediction interval width MEAN(yhat_upper - yhat_lower) in target units (diagnostic; read alongside coverage).'),
  ope FLOAT64 OPTIONS(description='Overall Percentage Error: |SUM(yhat) - SUM(y_true)| / |SUM(y_true)| measuring cumulative horizon bias magnitude (lower is better).'),
  rmsle FLOAT64 OPTIONS(description='Root Mean Squared Logarithmic Error: SQRT(MEAN((LN(1 + yhat) - LN(1 + y_true))^2)) for non-negative series (lower is better).'),
  msse FLOAT64 OPTIONS(description='Mean Squared Scaled Error: MSE divided by mean squared one-step naive error on training history (lower is better).'),
  msis FLOAT64 OPTIONS(description='Mean Scaled Interval Score (M4/M5 interval metric): Winkler interval_score scaled by seasonal naive MAE (< 1.0 is sharp and calibrated; lower is better).'),
  r2 FLOAT64 OPTIONS(description='Coefficient of determination: 1 - SUM((y_true - yhat)^2) / SUM((y_true - MEAN(y_true))^2) (higher is better; 1.0 is perfect).'),
  cv FLOAT64 OPTIONS(description='Coefficient of Variation of RMSE: RMSE / |MEAN(y_true)| (lower is better).'),
  fit_seconds    FLOAT64 OPTIONS(description='Wall-clock seconds spent fitting and backtesting this (ts_id, model_type) cell.'),
  best_params    JSON OPTIONS(description='Fitted or Optuna-tuned model parameters and conformal calibration metadata as native JSON.'),
  model_artifact STRING OPTIONS(description='GCS URI (gs://...) of the serialized fitted model artifact when artifact persistence is enabled; NULL otherwise.'),
  created_at     TIMESTAMP NOT NULL OPTIONS(description='UTC timestamp when this metadata row was appended (newest write wins on deduplication).'),
  worker_id      STRING OPTIONS(description='Identifier of the distributed worker/executor/pod that processed this cell.'),
  cell_started_at TIMESTAMP OPTIONS(description='UTC timestamp when worker execution began for this cell.'),
  cell_ended_at  TIMESTAMP OPTIONS(description='UTC timestamp when worker execution completed for this cell.'),
  cpu_seconds    FLOAT64 OPTIONS(description='Process CPU seconds consumed while fitting this cell.'),
  process_rss_bytes INT64 OPTIONS(description='Peak worker resident set size (RSS) in bytes observed during cell execution.'),
  peak_gpu_bytes INT64 OPTIONS(description='Peak CUDA GPU memory allocated in bytes during cell execution (NULL or 0 on CPU).'),
  intraop_threads INT64 OPTIONS(description='BLAS/OpenMP/PyTorch intra-op thread cap enforced during cell execution.'),
  n_obs          INT64 OPTIONS(description='Number of historical training observations available for this series.'),
  cell_status    STRING OPTIONS(description='Cell completion outcome: \'ok\' when fit succeeded or \'error\' when fitting failed.'),
  error_class    STRING OPTIONS(description='Python exception class name when cell_status=\'error\'; NULL on success.'),
  error_detail   STRING OPTIONS(description='Sanitized error message when cell_status=\'error\'; NULL on success.'),
  backtest_status STRING OPTIONS(description='Backtest fold achievement status: full, reduced, unscored, failed, or NULL when backtest is disabled.'),
  backtest_note  STRING OPTIONS(description='Diagnostic explanation when backtest_status is reduced, unscored, or failed.'),
  n_folds_achieved INT64 OPTIONS(description='Number of walk-forward validation folds actually evaluated for this series.'),
  achieved_step  INT64 OPTIONS(description='Effective step stride (in periods) between consecutive validation fold cutoffs.'),
  achieved_min_train INT64 OPTIONS(description='Minimum training window length (in periods) achieved on the earliest fold.'),
  first_val_date DATE OPTIONS(description='Earliest validation date evaluated across backtest folds for this cell.'),
  last_val_date  DATE OPTIONS(description='Latest validation date evaluated across backtest folds for this cell.'),
  interval_source STRING OPTIONS(description='Origin of raw prediction intervals prior to conformal calibration (e.g. native, residual).'),
  point_forecast_source STRING OPTIONS(description='Origin of operational yhat (e.g. raw, median_bias_adjusted, target_quantile).'),
  point_forecast_decision STRING OPTIONS(description='Audit decision explaining whether OOF residual bias adjustment was applied or kept raw.'),
  interval_calibration STRING OPTIONS(description='Interval calibration method applied (e.g. conformal_oof, aci, raw).'),
  point_forecast_margin FLOAT64 OPTIONS(description='OOF metric change produced by point-forecast bias adjustment.'),
  ensemble_scoring STRING OPTIONS(description='Scoring window mode used for ensemble weight estimation.'),
  hpo_scoring    STRING OPTIONS(description='HPO validation discipline: \'held_out\' (newest fold reserved from tuning) or \'in_sample\'.'),
  n_fits         INT64 OPTIONS(description='Total number of model fit calls executed for this cell across backtest folds and final fit.'),
  train_rows_total INT64 OPTIONS(description='Cumulative training rows processed across all fits for this cell.'),
  n_hpo_fits     INT64 OPTIONS(description='Number of Optuna hyperparameter trial fits executed for this cell.'),
  fit_diagnostics JSON OPTIONS(description='Structured JSON diagnostics: $.base_model, $.training_mode, $.cov_profile, $.covariates_used, $.feature_attributions (global feature importances), $.demand_profile.sbc_class (smooth/erratic/intermittent/lumpy), $.borrowed_from_run_id.'),
  device_requested STRING OPTIONS(description='Requested accelerator device for this cell (cpu or cuda).'),
  device_available STRING OPTIONS(description='Accelerator device detected on the worker host.'),
  device_used    STRING OPTIONS(description='Actual device used by the model fit (cpu or cuda).'),
  device_name    STRING OPTIONS(description='Hardware model name of the CPU/GPU device (e.g. NVIDIA L4).'),
  backtest_refit STRING OPTIONS(description='Refit behavior executed across backtest folds: per_fold, recondition (frozen), or unsupported.'),
  staleness_gap  FLOAT64 OPTIONS(description='Accuracy loss (in decision_metric units) of a never-refit frozen model vs. per-fold refitting.')
)
PARTITION BY DATE(created_at)
CLUSTER BY run_id, model_type
OPTIONS (
  description = 'Per-(run_id, ts_id, model_type) fit diagnostics, resource telemetry, and all 21 evaluation metrics. Rows with fold_id IS NULL summarize the full-panel backtest and final production fit; rows with a non-null fold_id record per-fold scores. Deduplicate retries via QUALIFY ROW_NUMBER() OVER (PARTITION BY run_id, ts_id, model_type, fold_id, ensemble_id ORDER BY created_at DESC NULLS LAST) = 1.'
);

CREATE TABLE IF NOT EXISTS `proj.scale_forecasting.forecast_predictions` (
  run_id        STRING NOT NULL OPTIONS(description='Parent run_id joining to run_registry.run_id.'),
  ts_id         STRING NOT NULL OPTIONS(description='Time-series identifier.'),
  model_type    STRING NOT NULL OPTIONS(description='Model, variant, ensemble (ensemble_*), or reconciled model identifier.'),
  compute_engine STRING OPTIONS(description='Compute runtime that generated this forecast.'),
  ensemble_id   STRING OPTIONS(description='Ensemble configuration digest for ensemble_* rows; NULL for base models.'),
  forecast_date DATE NOT NULL OPTIONS(description='Future target date being forecasted (partition key).'),
  yhat          FLOAT64 OPTIONS(description='Primary operational point forecast (incorporating calibration, bounds, and rounding when configured).'),
  yhat_raw      FLOAT64 OPTIONS(description='Unadjusted raw point forecast emitted directly by the underlying model.'),
  yhat_adjusted FLOAT64 OPTIONS(description='OOF-calibrated point forecast prior to physical floor/cap/rounding constraints.'),
  yhat_lower    FLOAT64 OPTIONS(description='Calibrated lower prediction interval bound (e.g. P10 at 80% confidence_level).'),
  yhat_upper    FLOAT64 OPTIONS(description='Calibrated upper prediction interval bound (e.g. P90 at 80% confidence_level).'),
  quantiles     JSON OPTIONS(description='Monotone multi-quantile forecast fan as native JSON keyed by probability string, e.g. JSON_VALUE(quantiles, \'$."0.1"\'), \'$."0.5"\', \'$."0.9"\'.'),
  created_at    TIMESTAMP OPTIONS(description='UTC timestamp when this prediction row was written (latest write wins on deduplication).'),
  explanations  JSON OPTIONS(description='Per-horizon-step local feature attributions JSON: $.baseline_score and $.attributions.<feature_name>.')
)
PARTITION BY forecast_date
CLUSTER BY run_id, ts_id
OPTIONS (
  description = 'Forward horizon point and probabilistic forecasts per (run_id, ts_id, model_type, forecast_date). Carries operational yhat, unadjusted yhat_raw, bias-corrected yhat_adjusted, calibrated [yhat_lower, yhat_upper] bounds, multi-percentile quantiles JSON (e.g. JSON_VALUE(quantiles, \'$."0.5"\')), and per-step explanations JSON.'
);

CREATE TABLE IF NOT EXISTS `proj.scale_forecasting.backtest_oof` (
  run_id        STRING NOT NULL OPTIONS(description='Parent run_id joining to run_registry.run_id.'),
  ts_id         STRING NOT NULL OPTIONS(description='Time-series identifier.'),
  model_type    STRING NOT NULL OPTIONS(description='Model, variant, ensemble (ensemble_*), or reconciled model identifier.'),
  fold_id       INT64 NOT NULL OPTIONS(description='0-based walk-forward validation fold index; MAX(fold_id) per run_id is the newest holdout fold.'),
  forecast_date DATE NOT NULL OPTIONS(description='Historical validation date scored in this backtest fold (partition key).'),
  y_true        FLOAT64 OPTIONS(description='Snapshot-locked ground-truth actual observed at forecast_date as of run launch.'),
  yhat          FLOAT64 OPTIONS(description='Operational out-of-fold point forecast evaluated at forecast_date.'),
  yhat_raw      FLOAT64 OPTIONS(description='Unadjusted raw out-of-fold model prediction.'),
  yhat_adjusted FLOAT64 OPTIONS(description='Bias-adjusted out-of-fold prediction prior to output clamping.'),
  cutoff_date   DATE OPTIONS(description='Last training timestamp visible to the model in this fold (forecast origin).'),
  horizon_step  INT64 OPTIONS(description='1-based lead-time step ahead from cutoff_date (1 = 1 step ahead, H = full horizon).'),
  yhat_lower    FLOAT64 OPTIONS(description='Out-of-fold lower prediction interval bound.'),
  yhat_upper    FLOAT64 OPTIONS(description='Out-of-fold upper prediction interval bound.'),
  created_at    TIMESTAMP OPTIONS(description='UTC timestamp when this OOF row was written (latest write wins on deduplication).'),
  ensemble_id   STRING OPTIONS(description='Ensemble configuration digest for ensemble_* rows; NULL for base models.'),
  yhat_stale    FLOAT64 OPTIONS(description='Counterfactual frozen-model prediction when backtest.control_arm=true (used to measure staleness_gap).')
)
PARTITION BY forecast_date
CLUSTER BY run_id, ts_id
OPTIONS (
  description = 'Snapshot-locked out-of-fold (OOF) historical backtest predictions and ground-truth actuals (y_true) per (run_id, ts_id, model_type, fold_id, forecast_date). Preserves exact as-of-launch actuals even after BigQuery time-travel windows expire; drives learned stacking ensembles, conformal calibration, and v_model_leaderboard_comparable.'
);

CREATE TABLE IF NOT EXISTS `proj.scale_forecasting.source_series_iceberg` (
  ts_id       STRING NOT NULL OPTIONS(description='Unique time-series identifier (e.g. SKU, store-SKU, meter, well, or lane ID).'),
  ds          DATE NOT NULL OPTIONS(description='Observation date/timestamp in regular panel frequency (partition key on native tables).'),
  y           FLOAT64 OPTIONS(description='Observed historical target value (e.g. unit demand, load MW, production volume).'),
  archetype   STRING OPTIONS(description='Synthetic or domain demand archetype label describing series dynamics.'),
  is_holiday  BOOL OPTIONS(description='Boolean calendar flag indicating whether ds falls on a holiday event.')
)
PARTITION BY ds
CLUSTER BY ts_id
WITH CONNECTION `proj.us-central1.sf-conn`
OPTIONS (
  description = 'Shipped univariate benchmark source panel stored as a BigQuery-managed Apache Iceberg table over Parquet in GCS (ts_id, ds, y, archetype, is_holiday).',
  file_format = 'PARQUET',
  table_format = 'ICEBERG',
  storage_uri = 'gs://proj-wh/warehouse/source_series_iceberg'
);

CREATE TABLE IF NOT EXISTS `proj.scale_forecasting.source_series_native` (
  ts_id       STRING NOT NULL OPTIONS(description='Unique time-series identifier (e.g. SKU, store-SKU, meter, well, or lane ID).'),
  ds          DATE NOT NULL OPTIONS(description='Observation date/timestamp in regular panel frequency (partition key on native tables).'),
  y           FLOAT64 OPTIONS(description='Observed historical target value (e.g. unit demand, load MW, production volume).'),
  archetype   STRING OPTIONS(description='Synthetic or domain demand archetype label describing series dynamics.'),
  is_holiday  BOOL OPTIONS(description='Boolean calendar flag indicating whether ds falls on a holiday event.')
)
PARTITION BY ds
CLUSTER BY ts_id
OPTIONS (
  description = 'Shipped univariate benchmark source panel stored as a native BigQuery table partitioned by ds and clustered by ts_id (ts_id, ds, y, archetype, is_holiday).'
);