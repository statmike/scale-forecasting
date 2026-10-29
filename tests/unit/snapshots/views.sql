CREATE OR REPLACE VIEW `proj.scale_forecasting.v_run_summary` (
  run_id OPTIONS(description='Deterministic run identifier (1 deduplicated row per run).'),
  created_at OPTIONS(description='UTC timestamp of the latest header write for this run.'),
  status OPTIONS(description='Current run lifecycle status (COMPLETED, PARTIAL, FAILED, RUNNING, STAGED, CANCELLED).'),
  python_runtime OPTIONS(description='Python runtime version of the orchestrator.'),
  n_series OPTIONS(description='Number of distinct time series in the run.'),
  n_models OPTIONS(description='Number of configured base models/variants.'),
  backtest_on OPTIONS(description='Whether walk-forward backtesting was enabled.'),
  runtime_seconds OPTIONS(description='Total orchestrator wall-clock seconds.'),
  n_jobs OPTIONS(description='Number of non-repair family jobs executed in this launch.'),
  longest_job_seconds OPTIONS(description='Critical-path duration (seconds) of the slowest family job.'),
  jobs_seconds OPTIONS(description='Sum of wall-clock seconds across all parallel family jobs.'),
  jobs_span_seconds OPTIONS(description='Elapsed seconds from earliest job start to latest job end.'),
  overhead_seconds OPTIONS(description='Non-critical-path orchestration/ensemble/native overhead (jobs_span_seconds - longest_job_seconds >= 0).'),
  overhead_fraction OPTIONS(description='Share of total job span spent outside the single longest family job.'),
  executor_instances OPTIONS(description='Dataproc Serverless initial executor count when stamped on header; NULL otherwise.'),
  executor_cores OPTIONS(description='Cores per Spark executor when applicable.'),
  max_executors OPTIONS(description='Maximum autoscaling Spark executors when applicable.'),
  executor_memory OPTIONS(description='Memory per Spark executor when applicable.'),
  executor_memory_overhead OPTIONS(description='Off-heap memory overhead per Spark executor when applicable.'),
  dcu_milli_seconds OPTIONS(description='Dataproc Serverless Data Compute Unit (DCU) milli-seconds consumed.'),
  runtime_version OPTIONS(description='Cloud runtime image/platform version.'),
  sizing OPTIONS(description='Per-family auto-sizing audit JSON ($.<family> machine_type, workers, memory rationale).'),
  capacity OPTIONS(description='Shared cluster regional capacity failover ledger JSON.')
)
OPTIONS (
  description = 'Curated one-row-per-run summary deduplicating run_registry to the latest header and rolling up the active launch job time ledger (n_jobs, longest_job_seconds, jobs_seconds, jobs_span_seconds, overhead_seconds, overhead_fraction) plus Dataproc/sizing telemetry.'
) AS
WITH header AS (
  SELECT *
  FROM `proj.scale_forecasting.run_registry`
  QUALIFY ROW_NUMBER() OVER (PARTITION BY run_id ORDER BY created_at DESC NULLS LAST) = 1
),
current_jobs AS (
  SELECT run_id, family, created_at, started_at, ended_at
  FROM `proj.scale_forecasting.run_jobs`
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY run_id, family ORDER BY attempt DESC, created_at DESC
  ) = 1
),
launch_jobs AS (
  SELECT
    j.run_id,
    j.started_at,
    j.ended_at,
    TIMESTAMP_DIFF(j.ended_at, j.started_at, MILLISECOND) / 1000 AS job_seconds
  FROM current_jobs AS j
  JOIN header AS h USING (run_id)
  WHERE j.created_at >= h.created_at
    AND NOT ENDS_WITH(j.family, '_repair')
),
ledger AS (
  SELECT
    run_id,
    COUNT(*) AS n_jobs,
    MAX(job_seconds) AS longest_job_seconds,
    SUM(job_seconds) AS jobs_seconds,
    TIMESTAMP_DIFF(MAX(ended_at), MIN(started_at), MILLISECOND) / 1000 AS jobs_span_seconds
  FROM launch_jobs
  GROUP BY run_id
)
SELECT
  h.run_id,
  h.created_at,
  h.status,
  h.python_runtime,
  h.n_series,
  h.n_models,
  h.backtest_on,
  h.runtime_seconds,
  l.n_jobs,
  l.longest_job_seconds,
  l.jobs_seconds,
  l.jobs_span_seconds,
  l.jobs_span_seconds - l.longest_job_seconds AS overhead_seconds,
  SAFE_DIVIDE(l.jobs_span_seconds - l.longest_job_seconds, l.jobs_span_seconds)
    AS overhead_fraction,
  CAST(JSON_VALUE(h.job_telemetry, '$.executor_instances') AS INT64) AS executor_instances,
  CAST(JSON_VALUE(h.job_telemetry, '$.executor_cores') AS INT64) AS executor_cores,
  CAST(JSON_VALUE(h.job_telemetry, '$.max_executors') AS INT64) AS max_executors,
  JSON_VALUE(h.job_telemetry, '$.executor_memory') AS executor_memory,
  JSON_VALUE(h.job_telemetry, '$.executor_memory_overhead') AS executor_memory_overhead,
  CAST(JSON_VALUE(h.job_telemetry, '$.dcu_milli_seconds') AS INT64) AS dcu_milli_seconds,
  JSON_VALUE(h.job_telemetry, '$.runtime_version') AS runtime_version,
  JSON_QUERY(h.job_telemetry, '$.sizing') AS sizing,
  JSON_QUERY(h.job_telemetry, '$.capacity') AS capacity
FROM header AS h
LEFT JOIN ledger AS l USING (run_id);

CREATE OR REPLACE VIEW `proj.scale_forecasting.v_run_jobs` (
  run_id OPTIONS(description='Parent run identifier.'),
  family OPTIONS(description='Model family or ensemble stage (1 deduplicated row per current attempt).'),
  job_id OPTIONS(description='Deterministic job identifier for the current attempt.'),
  attempt OPTIONS(description='Current (highest) attempt number for (run_id, family).'),
  runtime OPTIONS(description='Resolved cloud compute runtime (spark, ray, vertex, gce, gke, vertex_automl, bigquery).'),
  spark_mode OPTIONS(description='Spark deployment mode when runtime=\'spark\'.'),
  hardware OPTIONS(description='Hardware tier (cpu or gpu).'),
  gpu_type OPTIONS(description='GPU model (T4, L4, A100, A100_80GB) when hardware=\'gpu\'.'),
  system_job_id OPTIONS(description='Cloud platform native job/instance identifier.'),
  status OPTIONS(description='Current job status.'),
  created_at OPTIONS(description='UTC timestamp when the job trace row was written.'),
  started_at OPTIONS(description='UTC timestamp when job submission started.'),
  ended_at OPTIONS(description='UTC timestamp when job execution finished.'),
  runtime_seconds OPTIONS(description='Total wall-clock seconds for this job.'),
  provision_seconds OPTIONS(description='Cloud VM/containerstartup and provisioning wait seconds before code execution began.'),
  execute_seconds OPTIONS(description='Active model training/scoring execution seconds once compute provisioned.'),
  failure_reason OPTIONS(description='Failure classification token and message when status=\'FAILED\'.'),
  device_verdict OPTIONS(description='GPU utilization audit verdict (e.g. ENGAGED_UTILISED vs IDLE_WASTE); NULL for CPU jobs.'),
  device_use OPTIONS(description='Detailed accelerator usage counts and peak GPU memory JSON.'),
  probe_handle OPTIONS(description='Cloud coordinates JSON used by live runtime probes and cancellation.'),
  capacity OPTIONS(description='Per-job regional/zonal capacity failover attempt ledger JSON.')
)
OPTIONS (
  description = 'Curated per-(run_id, family) execution trace keeping only the latest attempt per family and unpacking provision_seconds, execute_seconds, device_verdict (ENGAGED_UTILISED vs IDLE_WASTE), probe_handle, and regional capacity failover history.'
) AS
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
  CAST(JSON_VALUE(job_telemetry, '$.platform.provision_s') AS FLOAT64) AS provision_seconds,
  CAST(JSON_VALUE(job_telemetry, '$.platform.execute_s') AS FLOAT64) AS execute_seconds,
  failure_reason,
  JSON_VALUE(job_telemetry, '$.device_use.verdict') AS device_verdict,
  JSON_QUERY(job_telemetry, '$.device_use') AS device_use,
  JSON_QUERY(job_telemetry, '$.probe_handle') AS probe_handle,
  JSON_QUERY(job_telemetry, '$.capacity') AS capacity
FROM `proj.scale_forecasting.run_jobs`
QUALIFY ROW_NUMBER() OVER (
  PARTITION BY run_id, family ORDER BY attempt DESC, created_at DESC
) = 1;

CREATE OR REPLACE VIEW `proj.scale_forecasting.v_model_leaderboard` (
  run_id OPTIONS(description='Run identifier.'),
  model_type OPTIONS(description='Model, variant, ensemble, or reconciled model name.'),
  ensemble_id OPTIONS(description='Ensemble configuration hash for ensemble_* rows; NULL for base models.'),
  base_model OPTIONS(description='Underlying plugin name stripped of @variant suffixes.'),
  scope_capability OPTIONS(description='Model training scope capability (local_only, global_only, multi_scope).'),
  training_mode OPTIONS(description='Executed training scope (local, global, hybrid).'),
  cov_profile OPTIONS(description='Active covariate profile tag (univariate, cov_all, cov_drop_*, etc.).'),
  is_univariate OPTIONS(description='True when the model ran without exogenous covariates.'),
  covariates_used OPTIONS(description='Comma-separated list of exogenous covariate columns consumed by the model.'),
  model_spec_hash OPTIONS(description='Cross-run reusable specification hash for snapshot-verified cell borrowing.'),
  borrowed_from_run_id OPTIONS(description='Source run_id when cells were borrowed via reuse_prior_runs; NULL when fit fresh.'),
  n_borrowed OPTIONS(description='Count of series cells borrowed from a prior run.'),
  compute_engine OPTIONS(description='Compute engine that produced the model\'s cells.'),
  n_cells OPTIONS(description='Number of deduplicated series cells landed for this model.'),
  n_no_artifact OPTIONS(description='Count of cells with NULL model_artifact.'),
  no_artifact_rate OPTIONS(description='Fraction of cells without a persisted GCS artifact.'),
  median_fit_seconds OPTIONS(description='Median per-series fit duration in seconds.'),
  mean_wape OPTIONS(description='Unweighted mean of per-series WAPEs (for volume-weighted panel WAPE use v_model_leaderboard_comparable.pooled_wape).'),
  mean_mae OPTIONS(description='Mean of per-series MAEs across scored cells.'),
  mean_staleness_gap OPTIONS(description='Mean accuracy degradation under frozen refit vs. per-fold refit.'),
  refit_modes OPTIONS(description='Distinct backtest refit modes observed across the model\'s cohort.')
)
OPTIONS (
  description = 'Deduplicated per-(run_id, model_type, ensemble_id) model summary over full-fit metadata (fold_id IS NULL): cell counts, missing-artifact error rate, median fit seconds, series-averaged mean_wape and mean_mae, mean_staleness_gap, and training/covariate tags.'
) AS
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
  COALESCE(
    JSON_VALUE(ANY_VALUE(fit_diagnostics), '$.base_model'),
    SPLIT(model_type, '@')[SAFE_OFFSET(0)]
  ) AS base_model,
  JSON_VALUE(ANY_VALUE(fit_diagnostics), '$.scope_capability') AS scope_capability,
  JSON_VALUE(ANY_VALUE(fit_diagnostics), '$.training_mode') AS training_mode,
  JSON_VALUE(ANY_VALUE(fit_diagnostics), '$.cov_profile') AS cov_profile,
  CAST(JSON_VALUE(ANY_VALUE(fit_diagnostics), '$.is_univariate') AS BOOL) AS is_univariate,
  ARRAY_TO_STRING(JSON_VALUE_ARRAY(ANY_VALUE(fit_diagnostics), '$.covariates_used'), ',')
    AS covariates_used,
  JSON_VALUE(ANY_VALUE(fit_diagnostics), '$.model_spec_hash') AS model_spec_hash,
  JSON_VALUE(ANY_VALUE(fit_diagnostics), '$.borrowed_from_run_id') AS borrowed_from_run_id,
  COUNTIF(JSON_VALUE(fit_diagnostics, '$.borrowed_from_run_id') IS NOT NULL) AS n_borrowed,
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

CREATE OR REPLACE VIEW `proj.scale_forecasting.v_backtest_coverage` (
  run_id OPTIONS(description='Run identifier.'),
  model_type OPTIONS(description='Model, variant, or ensemble identifier.'),
  ensemble_id OPTIONS(description='Ensemble configuration hash; NULL for base models.'),
  backtest_status OPTIONS(description='Cohort fold status: full, reduced, unscored, or failed.'),
  n_folds_achieved OPTIONS(description='Exact number of walk-forward folds achieved by series in this cohort.'),
  backtest_refit OPTIONS(description='Refit discipline executed for this cohort (per_fold, recondition, unsupported).'),
  n_series OPTIONS(description='Count of distinct series belonging to this coverage cohort.'),
  mean_staleness_gap OPTIONS(description='Mean frozen-vs-refit staleness gap for this cohort.'),
  series_share OPTIONS(description='Fraction of the model\'s total series panel represented by this cohort (sums to 1.0 per model).')
)
OPTIONS (
  description = 'Achieved backtest fold coverage cohort table per (run_id, model_type, ensemble_id, backtest_status, n_folds_achieved, backtest_refit) showing series counts and panel share.'
) AS
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

CREATE OR REPLACE VIEW `proj.scale_forecasting.v_model_leaderboard_comparable` (
  run_id OPTIONS(description='Run identifier.'),
  model_type OPTIONS(description='Model, variant, ensemble (ensemble_*), or reconciled model identifier.'),
  ensemble_id OPTIONS(description='Ensemble configuration hash; NULL for base models.'),
  holdout_fold_id OPTIONS(description='Fold ID of the newest holdout fold evaluated across all models in the run.'),
  n_series OPTIONS(description='Number of distinct series evaluated on the holdout fold (equal n_series confirms apples-to-apples comparison).'),
  n_points OPTIONS(description='Total (ts_id, forecast_date) evaluation points pooled on the holdout fold.'),
  pooled_wape OPTIONS(description='Volume-weighted panel WAPE on the holdout fold: SUM(|y_true - yhat|) / SUM(|y_true|) (lower is better).'),
  pooled_mae OPTIONS(description='Pooled Mean Absolute Error AVG(|y_true - yhat|) on the holdout fold (lower is better).'),
  first_forecast_date OPTIONS(description='Start date of the holdout fold evaluation window.'),
  last_forecast_date OPTIONS(description='End date of the holdout fold evaluation window.'),
  refit_modes OPTIONS(description='Distinct backtest refit modes across the model\'s cells.')
)
OPTIONS (
  description = 'Apples-to-apples model leaderboard restricted to the newest holdout fold in backtest_oof and computed via panel-pooled error: pooled_wape = SUM(|y_true - yhat|) / SUM(|y_true|) and pooled_mae = AVG(|y_true - yhat|). Preferred view for ranking models and ensembles.'
) AS
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

CREATE OR REPLACE VIEW `proj.scale_forecasting.v_fva_waterfall` (
  run_id OPTIONS(description='Run identifier.'),
  stage_order OPTIONS(description='Integer pipeline complexity order: 1=Baseline Naive, 2=Statistical, 3=Univariate ML/DL, 4=Covariate ML/DL/AutoML, 5=Ensemble, 6=Reconciled.'),
  stage_name OPTIONS(description='Human-readable FVA pipeline stage label corresponding to stage_order.'),
  model_type OPTIONS(description='Model, variant, ensemble, or reconciled model name.'),
  ensemble_id OPTIONS(description='Ensemble configuration hash; NULL for base models.'),
  n_series OPTIONS(description='Number of series scored on the holdout fold.'),
  n_points OPTIONS(description='Number of OOF observations pooled on the holdout fold.'),
  pooled_wape OPTIONS(description='Model\'s volume-weighted holdout-fold WAPE.'),
  pooled_mae OPTIONS(description='Model\'s holdout-fold pooled MAE.'),
  baseline_model OPTIONS(description='Reference naive baseline model in this run (best naive* model, or highest-WAPE univariate baseline if no naive model ran).'),
  baseline_wape OPTIONS(description='Holdout-fold pooled_wape of baseline_model.'),
  fva_wape_delta OPTIONS(description='Absolute WAPE reduction vs. baseline (baseline_wape - pooled_wape; positive means this model beat the baseline).'),
  fva_relative_lift_pct OPTIONS(description='Percentage relative error reduction vs. baseline: 100 * (baseline_wape - pooled_wape) / baseline_wape.')
)
OPTIONS (
  description = 'Forecast Value Added (FVA) waterfall per (run_id, model_type, ensemble_id) on the comparable holdout fold. Classifies each model into pipeline stages (1_baseline_naive -> 2_statistical -> 3_univariate_ml_dl -> 4_covariate_ml_dl -> 5_ensemble -> 6_reconciled) and measures absolute WAPE reduction and relative % FVA lift vs. the run\'s naive baseline.'
) AS
WITH oof_deduped AS (
  SELECT *
  FROM `proj.scale_forecasting.backtest_oof`
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY run_id, ts_id, model_type, fold_id, forecast_date, ensemble_id
    ORDER BY created_at DESC NULLS LAST
  ) = 1
),
holdout AS (
  SELECT *
  FROM oof_deduped
  WHERE y_true IS NOT NULL AND yhat IS NOT NULL
  QUALIFY fold_id = MAX(fold_id) OVER (PARTITION BY run_id)
),
meta_deduped AS (
  SELECT
    run_id,
    model_type,
    COALESCE(ensemble_id, '') AS ens_key,
    ANY_VALUE(compute_engine) AS compute_engine,
    COALESCE(
      JSON_VALUE(ANY_VALUE(fit_diagnostics), '$.base_model'),
      SPLIT(model_type, '@')[SAFE_OFFSET(0)]
    ) AS base_model,
    COALESCE(
      CAST(JSON_VALUE(ANY_VALUE(fit_diagnostics), '$.is_univariate') AS BOOL),
      TRUE
    ) AS is_univariate
  FROM (
    SELECT *
    FROM `proj.scale_forecasting.forecast_metadata`
    WHERE fold_id IS NULL
    QUALIFY ROW_NUMBER() OVER (
      PARTITION BY run_id, ts_id, model_type, fold_id, ensemble_id
      ORDER BY created_at DESC NULLS LAST
    ) = 1
  )
  GROUP BY run_id, model_type, ens_key
),
model_scores AS (
  SELECT
    h.run_id,
    h.model_type,
    h.ensemble_id,
    COALESCE(h.ensemble_id, '') AS ens_key,
    COUNT(DISTINCT h.ts_id) AS n_series,
    COUNT(*) AS n_points,
    SAFE_DIVIDE(SUM(ABS(h.y_true - h.yhat)), SUM(ABS(h.y_true))) AS pooled_wape,
    AVG(ABS(h.y_true - h.yhat)) AS pooled_mae
  FROM holdout AS h
  GROUP BY h.run_id, h.model_type, h.ensemble_id
),
classified AS (
  SELECT
    s.run_id,
    CASE
      WHEN REGEXP_CONTAINS(
        s.model_type,
        r'_(bottom_up|wls_struct|mint_shrink|mint_nonneg|thief|cross_temporal_mint)$'
      ) THEN 6
      WHEN STARTS_WITH(s.model_type, 'ensemble_') OR s.ensemble_id IS NOT NULL THEN 5
      WHEN STARTS_WITH(
        LOWER(COALESCE(m.base_model, SPLIT(s.model_type, '@')[SAFE_OFFSET(0)])),
        'naive'
      ) THEN 1
      WHEN LOWER(COALESCE(m.base_model, SPLIT(s.model_type, '@')[SAFE_OFFSET(0)])) IN (
        'arima_plus', 'auto_arima', 'auto_ces', 'auto_ets', 'bats', 'croston',
        'dynamic_optimized_theta', 'dynamicoptimizedtheta', 'ets', 'fft', 'holtwinters',
        'kalman', 'mstl', 'prophet', 'sarimax', 'stl_bagging', 'tbats', 'theta', 'ucm'
      ) THEN 2
      WHEN COALESCE(m.is_univariate, TRUE)
        AND COALESCE(m.compute_engine, '') != 'vertex_automl' THEN 3
      ELSE 4
    END AS stage_order,
    s.model_type,
    s.ensemble_id,
    s.n_series,
    s.n_points,
    s.pooled_wape,
    s.pooled_mae
  FROM model_scores AS s
  LEFT JOIN meta_deduped AS m
    ON s.run_id = m.run_id AND s.model_type = m.model_type AND s.ens_key = m.ens_key
),
staged AS (
  SELECT
    c.*,
    CASE c.stage_order
      WHEN 1 THEN '1_baseline_naive'
      WHEN 2 THEN '2_statistical'
      WHEN 3 THEN '3_univariate_ml_dl'
      WHEN 4 THEN '4_covariate_ml_dl'
      WHEN 5 THEN '5_ensemble'
      ELSE '6_reconciled'
    END AS stage_name
  FROM classified AS c
),
baseline AS (
  SELECT
    run_id,
    model_type AS baseline_model,
    pooled_wape AS baseline_wape
  FROM staged
  WHERE stage_order <= 4 AND pooled_wape IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY run_id
    ORDER BY
      IF(stage_order = 1, 0, 1) ASC,
      IF(stage_order = 1, pooled_wape, -pooled_wape) ASC,
      model_type ASC
  ) = 1
)
SELECT
  s.run_id,
  s.stage_order,
  s.stage_name,
  s.model_type,
  s.ensemble_id,
  s.n_series,
  s.n_points,
  s.pooled_wape,
  s.pooled_mae,
  b.baseline_model,
  b.baseline_wape,
  b.baseline_wape - s.pooled_wape AS fva_wape_delta,
  100.0 * SAFE_DIVIDE(b.baseline_wape - s.pooled_wape, b.baseline_wape)
    AS fva_relative_lift_pct
FROM staged AS s
LEFT JOIN baseline AS b USING (run_id);

CREATE OR REPLACE VIEW `proj.scale_forecasting.v_horizon_decay` (
  run_id OPTIONS(description='Run identifier.'),
  model_type OPTIONS(description='Model, variant, ensemble, or reconciled model identifier.'),
  ensemble_id OPTIONS(description='Ensemble configuration hash; NULL for base models.'),
  horizon_step OPTIONS(description='1-based forecast lead-time step ahead from fold cutoff_date (1 = 1 step ahead, H = max horizon).'),
  n_series OPTIONS(description='Distinct series evaluated at this horizon_step.'),
  n_points OPTIONS(description='Total OOF observations pooled at this horizon_step across folds.'),
  pooled_wape OPTIONS(description='Volume-weighted WAPE at this exact lead-time step ahead.'),
  pooled_mae OPTIONS(description='Mean Absolute Error at this exact lead-time step ahead.'),
  signed_bias OPTIONS(description='Mean signed error AVG(yhat - y_true) at this step (+ = over-forecast, - = under-forecast).'),
  empirical_coverage OPTIONS(description='Fraction of actuals inside [yhat_lower, yhat_upper] at this horizon step.'),
  mean_interval_width OPTIONS(description='Mean width (yhat_upper - yhat_lower) of the prediction interval at this horizon step.')
)
OPTIONS (
  description = 'Lead-time accuracy decay curve per (run_id, model_type, ensemble_id, horizon_step) computed from deduplicated backtest_oof: pooled_wape, pooled_mae, signed_bias, empirical_coverage, and mean_interval_width at each step h = 1..H ahead.'
) AS
WITH deduped AS (
  SELECT *
  FROM `proj.scale_forecasting.backtest_oof`
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY run_id, ts_id, model_type, fold_id, forecast_date, ensemble_id
    ORDER BY created_at DESC NULLS LAST
  ) = 1
),
stepped AS (
  SELECT
    run_id,
    ts_id,
    model_type,
    ensemble_id,
    COALESCE(
      horizon_step,
      ROW_NUMBER() OVER (
        PARTITION BY run_id, ts_id, model_type, fold_id, COALESCE(ensemble_id, '')
        ORDER BY forecast_date ASC
      )
    ) AS horizon_step,
    y_true,
    yhat,
    yhat_lower,
    yhat_upper,
    IF(
      yhat_lower IS NOT NULL AND yhat_upper IS NOT NULL,
      IF(y_true BETWEEN yhat_lower AND yhat_upper, 1.0, 0.0),
      NULL
    ) AS in_bounds
  FROM deduped
  WHERE y_true IS NOT NULL AND yhat IS NOT NULL
)
SELECT
  run_id,
  model_type,
  ensemble_id,
  horizon_step,
  COUNT(DISTINCT ts_id) AS n_series,
  COUNT(*) AS n_points,
  SAFE_DIVIDE(SUM(ABS(y_true - yhat)), SUM(ABS(y_true))) AS pooled_wape,
  AVG(ABS(y_true - yhat)) AS pooled_mae,
  AVG(yhat - y_true) AS signed_bias,
  AVG(in_bounds) AS empirical_coverage,
  AVG(yhat_upper - yhat_lower) AS mean_interval_width
FROM stepped
GROUP BY run_id, model_type, ensemble_id, horizon_step;

CREATE OR REPLACE VIEW `proj.scale_forecasting.v_bias_and_exceptions` (
  run_id OPTIONS(description='Run identifier.'),
  ts_id OPTIONS(description='Time-series identifier.'),
  model_type OPTIONS(description='Model, variant, ensemble, or reconciled model identifier.'),
  ensemble_id OPTIONS(description='Ensemble configuration hash; NULL for base models.'),
  sbc_class OPTIONS(description='Syntetos-Boylan demand regime from fit_diagnostics (smooth, erratic, intermittent, lumpy, or cold_start).'),
  cell_status OPTIONS(description='Cell fit status from forecast_metadata (\'ok\' or \'error\').'),
  n_oof_points OPTIONS(description='Number of out-of-fold validation points scored for this series and model.'),
  mean_actual OPTIONS(description='Mean historical actual y_true across the backtest evaluation window.'),
  mean_forecast OPTIONS(description='Mean out-of-fold forecast yhat across the backtest evaluation window.'),
  signed_bias OPTIONS(description='Mean signed error AVG(yhat - y_true); positive indicates over-forecasting.'),
  mad OPTIONS(description='Mean Absolute Deviation AVG(|y_true - yhat|) in target units.'),
  wape OPTIONS(description='Series-level Weighted Absolute Percentage Error.'),
  mase OPTIONS(description='Series-level Mean Absolute Scaled Error (< 1.0 beats naive).'),
  staleness_gap OPTIONS(description='Accuracy loss when freezing parameters vs. refitting per fold.'),
  tracking_signal OPTIONS(description='Trigg\'s cumulative signed error divided by MAD: SUM(yhat - y_true) / AVG(|y_true - yhat|). |tracking_signal| > 4.0 flags systematic bias.'),
  is_exception OPTIONS(description='True when the series requires planner review (cell error, |tracking_signal| > 4.0, MASE > 1.0, or staleness_gap > 0.05).'),
  exception_reason OPTIONS(description='Primary actionable triage code: CELL_FIT_ERROR, CHRONIC_OVER_FORECAST, CHRONIC_UNDER_FORECAST, WORSE_THAN_NAIVE, STALE_REFIT_DRIFT, or OK.')
)
OPTIONS (
  description = 'Planner exception triage queue per (run_id, ts_id, model_type, ensemble_id). Computes Trigg\'s tracking_signal (cumulative signed error / MAD), series WAPE, MASE, staleness_gap, and SBC demand regime, flagging CHRONIC_OVER_FORECAST, CHRONIC_UNDER_FORECAST, WORSE_THAN_NAIVE, STALE_REFIT_DRIFT, and CELL_FIT_ERROR.'
) AS
WITH meta_deduped AS (
  SELECT
    run_id,
    ts_id,
    model_type,
    ensemble_id,
    COALESCE(ensemble_id, '') AS ens_key,
    cell_status,
    wape,
    mase,
    staleness_gap,
    COALESCE(
      JSON_VALUE(fit_diagnostics, '$.demand_profile.sbc_class'),
      JSON_VALUE(fit_diagnostics, '$.sbc_class')
    ) AS sbc_class
  FROM `proj.scale_forecasting.forecast_metadata`
  WHERE fold_id IS NULL
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY run_id, ts_id, model_type, fold_id, ensemble_id
    ORDER BY created_at DESC NULLS LAST
  ) = 1
),
oof_deduped AS (
  SELECT *
  FROM `proj.scale_forecasting.backtest_oof`
  WHERE y_true IS NOT NULL AND yhat IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY run_id, ts_id, model_type, fold_id, forecast_date, ensemble_id
    ORDER BY created_at DESC NULLS LAST
  ) = 1
),
oof_agg AS (
  SELECT
    run_id,
    ts_id,
    model_type,
    COALESCE(ensemble_id, '') AS ens_key,
    COUNT(*) AS n_oof_points,
    AVG(y_true) AS mean_actual,
    AVG(yhat) AS mean_forecast,
    AVG(yhat - y_true) AS signed_bias,
    AVG(ABS(y_true - yhat)) AS mad,
    SAFE_DIVIDE(SUM(ABS(y_true - yhat)), SUM(ABS(y_true))) AS oof_wape,
    SAFE_DIVIDE(SUM(yhat - y_true), NULLIF(AVG(ABS(y_true - yhat)), 0.0)) AS tracking_signal
  FROM oof_deduped
  GROUP BY run_id, ts_id, model_type, ens_key
),
joined AS (
  SELECT
    m.run_id,
    m.ts_id,
    m.model_type,
    m.ensemble_id,
    m.sbc_class,
    m.cell_status,
    COALESCE(o.n_oof_points, 0) AS n_oof_points,
    o.mean_actual,
    o.mean_forecast,
    o.signed_bias,
    o.mad,
    COALESCE(m.wape, o.oof_wape) AS wape,
    m.mase,
    m.staleness_gap,
    o.tracking_signal,
    CASE
      WHEN m.cell_status = 'error' THEN 'CELL_FIT_ERROR'
      WHEN o.tracking_signal > 4.0 THEN 'CHRONIC_OVER_FORECAST'
      WHEN o.tracking_signal < -4.0 THEN 'CHRONIC_UNDER_FORECAST'
      WHEN m.mase > 1.0 THEN 'WORSE_THAN_NAIVE'
      WHEN m.staleness_gap > 0.05 THEN 'STALE_REFIT_DRIFT'
      ELSE 'OK'
    END AS exception_reason
  FROM meta_deduped AS m
  LEFT JOIN oof_agg AS o
    ON m.run_id = o.run_id
    AND m.ts_id = o.ts_id
    AND m.model_type = o.model_type
    AND m.ens_key = o.ens_key
)
SELECT
  j.run_id,
  j.ts_id,
  j.model_type,
  j.ensemble_id,
  j.sbc_class,
  j.cell_status,
  j.n_oof_points,
  j.mean_actual,
  j.mean_forecast,
  j.signed_bias,
  j.mad,
  j.wape,
  j.mase,
  j.staleness_gap,
  j.tracking_signal,
  j.exception_reason != 'OK' AS is_exception,
  j.exception_reason
FROM joined AS j;

CREATE OR REPLACE VIEW `proj.scale_forecasting.v_anomalies` (
  run_id OPTIONS(description='Run identifier.'),
  ts_id OPTIONS(description='Time-series identifier.'),
  model_type OPTIONS(description='Model, variant, or ensemble used as the expected baseline.'),
  ensemble_id OPTIONS(description='Ensemble configuration hash; NULL for base models.'),
  forecast_date OPTIONS(description='Historical date where the anomalous actual occurred.'),
  fold_id OPTIONS(description='Backtest fold index containing forecast_date.'),
  y_true OPTIONS(description='Observed actual value on forecast_date.'),
  yhat OPTIONS(description='Model expected out-of-fold point forecast on forecast_date.'),
  yhat_lower OPTIONS(description='Lower calibrated prediction bound on forecast_date.'),
  yhat_upper OPTIONS(description='Upper calibrated prediction bound on forecast_date.'),
  residual OPTIONS(description='Signed surprise (y_true - yhat); positive = unexpected spike, negative = unexpected drop.'),
  residual_zscore OPTIONS(description='Standardized residual z-score relative to this (run_id, ts_id, model_type) series\' OOF residual stddev.'),
  outside_interval OPTIONS(description='True when y_true fell outside [yhat_lower, yhat_upper].'),
  anomaly_type OPTIONS(description='Direction of anomaly: POSITIVE_SPIKE (y_true > yhat) or NEGATIVE_DIP (y_true < yhat).')
)
OPTIONS (
  description = 'Historical out-of-fold demand anomalies and interval breaches per (run_id, ts_id, model_type, ensemble_id, forecast_date) where |residual_zscore| >= 2.5 or y_true falls outside calibrated [yhat_lower, yhat_upper] bounds.'
) AS
WITH deduped AS (
  SELECT *
  FROM `proj.scale_forecasting.backtest_oof`
  WHERE y_true IS NOT NULL AND yhat IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY run_id, ts_id, model_type, fold_id, forecast_date, ensemble_id
    ORDER BY created_at DESC NULLS LAST
  ) = 1
),
scored AS (
  SELECT
    run_id,
    ts_id,
    model_type,
    ensemble_id,
    forecast_date,
    fold_id,
    y_true,
    yhat,
    yhat_lower,
    yhat_upper,
    y_true - yhat AS residual,
    SAFE_DIVIDE(
      (y_true - yhat) - AVG(y_true - yhat) OVER (
        PARTITION BY run_id, ts_id, model_type, COALESCE(ensemble_id, '')
      ),
      NULLIF(
        STDDEV_POP(y_true - yhat) OVER (
          PARTITION BY run_id, ts_id, model_type, COALESCE(ensemble_id, '')
        ),
        0.0
      )
    ) AS residual_zscore,
    IF(
      yhat_lower IS NOT NULL AND yhat_upper IS NOT NULL,
      y_true < yhat_lower OR y_true > yhat_upper,
      FALSE
    ) AS outside_interval,
    IF(y_true >= yhat, 'POSITIVE_SPIKE', 'NEGATIVE_DIP') AS anomaly_type
  FROM deduped
)
SELECT
  s.run_id,
  s.ts_id,
  s.model_type,
  s.ensemble_id,
  s.forecast_date,
  s.fold_id,
  s.y_true,
  s.yhat,
  s.yhat_lower,
  s.yhat_upper,
  s.residual,
  s.residual_zscore,
  s.outside_interval,
  s.anomaly_type
FROM scored AS s
WHERE ABS(s.residual_zscore) >= 2.5 OR s.outside_interval = TRUE;

CREATE OR REPLACE VIEW `proj.scale_forecasting.v_feature_attributions` (
  run_id OPTIONS(description='Run identifier.'),
  ts_id OPTIONS(description='Time-series identifier.'),
  model_type OPTIONS(description='Explainability-capable model or reconciled/variant identifier.'),
  attribution_scope OPTIONS(description='Granularity source: \'global_fit\' (from forecast_metadata.fit_diagnostics.feature_attributions) or \'horizon_local\' (averaged from forecast_predictions.explanations).'),
  feature_name OPTIONS(description='Covariate, calendar feature, lag, or similarity feature name (e.g. price_index, promo_flag, temperature, lag_7).'),
  mean_attribution OPTIONS(description='Mean attribution score (signed impact in target units for horizon_local; non-negative importance weight for global_fit).'),
  mean_abs_attribution OPTIONS(description='Mean absolute attribution magnitude |attribution| for ranking top drivers regardless of sign.'),
  n_horizon_steps OPTIONS(description='Number of forward forecast_date steps aggregated for horizon_local; NULL for global_fit.')
)
OPTIONS (
  description = 'Unified exogenous covariate and lag driver attributions per (run_id, ts_id, model_type, attribution_scope, feature_name), unpacking both global model importances (fit_diagnostics.feature_attributions) and horizon-averaged signed SHAP attributions (forecast_predictions.explanations.attributions).'
) AS
WITH meta_deduped AS (
  SELECT
    run_id,
    ts_id,
    model_type,
    TO_JSON_STRING(JSON_QUERY(fit_diagnostics, '$.feature_attributions')) AS attr_json
  FROM `proj.scale_forecasting.forecast_metadata`
  WHERE fold_id IS NULL
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY run_id, ts_id, model_type, fold_id, ensemble_id
    ORDER BY created_at DESC NULLS LAST
  ) = 1
),
global_attrs AS (
  SELECT
    m.run_id,
    m.ts_id,
    m.model_type,
    'global_fit' AS attribution_scope,
    REGEXP_EXTRACT(kv, r'^"([^"]+)"') AS feature_name,
    SAFE_CAST(REGEXP_EXTRACT(kv, r':\s*(.+)$') AS FLOAT64) AS mean_attribution,
    ABS(SAFE_CAST(REGEXP_EXTRACT(kv, r':\s*(.+)$') AS FLOAT64)) AS mean_abs_attribution,
    CAST(NULL AS INT64) AS n_horizon_steps
  FROM meta_deduped AS m,
  UNNEST(
    REGEXP_EXTRACT_ALL(
      m.attr_json,
      r'"[^"]+"\s*:\s*-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?'
    )
  ) AS kv
  WHERE m.attr_json IS NOT NULL
),
pred_deduped AS (
  SELECT
    run_id,
    ts_id,
    model_type,
    forecast_date,
    TO_JSON_STRING(JSON_QUERY(explanations, '$.attributions')) AS attr_json
  FROM `proj.scale_forecasting.forecast_predictions`
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY run_id, ts_id, model_type, forecast_date, ensemble_id
    ORDER BY created_at DESC NULLS LAST
  ) = 1
),
local_unpacked AS (
  SELECT
    p.run_id,
    p.ts_id,
    p.model_type,
    REGEXP_EXTRACT(kv, r'^"([^"]+)"') AS feature_name,
    SAFE_CAST(REGEXP_EXTRACT(kv, r':\s*(.+)$') AS FLOAT64) AS attr_val
  FROM pred_deduped AS p,
  UNNEST(
    REGEXP_EXTRACT_ALL(
      p.attr_json,
      r'"[^"]+"\s*:\s*-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?'
    )
  ) AS kv
  WHERE p.attr_json IS NOT NULL
),
local_attrs AS (
  SELECT
    run_id,
    ts_id,
    model_type,
    'horizon_local' AS attribution_scope,
    feature_name,
    AVG(attr_val) AS mean_attribution,
    AVG(ABS(attr_val)) AS mean_abs_attribution,
    COUNT(*) AS n_horizon_steps
  FROM local_unpacked
  WHERE feature_name IS NOT NULL AND attr_val IS NOT NULL
  GROUP BY run_id, ts_id, model_type, feature_name
),
combined AS (
  SELECT * FROM global_attrs WHERE feature_name IS NOT NULL AND mean_attribution IS NOT NULL
  UNION ALL
  SELECT * FROM local_attrs
)
SELECT
  c.run_id,
  c.ts_id,
  c.model_type,
  c.attribution_scope,
  c.feature_name,
  c.mean_attribution,
  c.mean_abs_attribution,
  c.n_horizon_steps
FROM combined AS c;

CREATE OR REPLACE VIEW `proj.scale_forecasting.v_executive_sop_summary` (
  run_id OPTIONS(description='Run identifier (1 executive S&OP scorecard row per run).'),
  created_at OPTIONS(description='UTC timestamp of the run.'),
  status OPTIONS(description='Run lifecycle status.'),
  source_table OPTIONS(description='Input source table read by this run (JSON_VALUE(raw_config, \'$.data.source_table\')).'),
  n_series OPTIONS(description='Total time series forecasted in the run.'),
  n_models OPTIONS(description='Number of candidate models evaluated.'),
  runtime_seconds OPTIONS(description='End-to-end run wall-clock duration in seconds.'),
  champion_model OPTIONS(description='Lowest-pooled_wape winning model or ensemble on the comparable holdout fold.'),
  champion_pooled_wape OPTIONS(description='Holdout-fold pooled WAPE of champion_model.'),
  champion_pooled_mae OPTIONS(description='Holdout-fold pooled MAE of champion_model.'),
  baseline_wape OPTIONS(description='Holdout-fold pooled WAPE of the run\'s naive baseline model.'),
  fva_relative_lift_pct OPTIONS(description='Percentage relative WAPE reduction achieved by champion_model over the naive baseline.'),
  horizon_start_date OPTIONS(description='First forward forecast_date generated by champion_model.'),
  horizon_end_date OPTIONS(description='Last forward forecast_date generated by champion_model.'),
  total_forecast_units OPTIONS(description='Sum of forward yhat across all series and horizon dates for champion_model.'),
  mean_interval_width OPTIONS(description='Mean forward uncertainty band width (yhat_upper - yhat_lower) for champion_model.'),
  n_exception_series OPTIONS(description='Count of series flagged with is_exception = TRUE for champion_model.'),
  exception_rate OPTIONS(description='Fraction of scored series requiring planner exception review for champion_model.'),
  n_gpu_waste_jobs OPTIONS(description='Count of GPU family jobs in v_run_jobs where device_verdict != \'ENGAGED_UTILISED\' (0 = clean FinOps).')
)
OPTIONS (
  description = 'One-row-per-run Executive S&OP Scorecard combining panel scale, winning model from v_model_leaderboard_comparable, FVA % lift over naive baseline, forward horizon total forecasted volume and uncertainty band width, planner exception rate, and GPU audit count.'
) AS
WITH header AS (
  SELECT
    run_id,
    created_at,
    status,
    JSON_VALUE(raw_config, '$.data.source_table') AS source_table,
    n_series,
    n_models,
    runtime_seconds
  FROM `proj.scale_forecasting.run_registry`
  QUALIFY ROW_NUMBER() OVER (PARTITION BY run_id ORDER BY created_at DESC NULLS LAST) = 1
),
oof_deduped AS (
  SELECT *
  FROM `proj.scale_forecasting.backtest_oof`
  WHERE y_true IS NOT NULL AND yhat IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY run_id, ts_id, model_type, fold_id, forecast_date, ensemble_id
    ORDER BY created_at DESC NULLS LAST
  ) = 1
),
holdout_scores AS (
  SELECT
    run_id,
    model_type,
    COALESCE(ensemble_id, '') AS ens_key,
    SAFE_DIVIDE(SUM(ABS(y_true - yhat)), SUM(ABS(y_true))) AS pooled_wape,
    AVG(ABS(y_true - yhat)) AS pooled_mae
  FROM (
    SELECT *
    FROM oof_deduped
    QUALIFY fold_id = MAX(fold_id) OVER (PARTITION BY run_id)
  )
  GROUP BY run_id, model_type, ens_key
),
pred_deduped AS (
  SELECT
    run_id,
    ts_id,
    model_type,
    COALESCE(ensemble_id, '') AS ens_key,
    forecast_date,
    yhat,
    yhat_lower,
    yhat_upper
  FROM `proj.scale_forecasting.forecast_predictions`
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY run_id, ts_id, model_type, forecast_date, ensemble_id
    ORDER BY created_at DESC NULLS LAST
  ) = 1
),
fallback_models AS (
  SELECT
    run_id,
    model_type,
    ens_key
  FROM pred_deduped
  GROUP BY run_id, model_type, ens_key
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY run_id
    ORDER BY
      IF(model_type = 'ensemble_champion', 0, IF(STARTS_WITH(model_type, 'ensemble_'), 1, 2)),
      model_type ASC
  ) = 1
),
scored_champion AS (
  SELECT
    run_id,
    model_type,
    ens_key,
    pooled_wape,
    pooled_mae
  FROM holdout_scores
  WHERE pooled_wape IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY run_id ORDER BY pooled_wape ASC, model_type ASC
  ) = 1
),
champion AS (
  SELECT
    h.run_id,
    COALESCE(sc.model_type, fm.model_type) AS champion_model,
    COALESCE(sc.ens_key, fm.ens_key, '') AS ens_key,
    sc.pooled_wape AS champion_pooled_wape,
    sc.pooled_mae AS champion_pooled_mae
  FROM header AS h
  LEFT JOIN scored_champion AS sc USING (run_id)
  LEFT JOIN fallback_models AS fm USING (run_id)
),
baseline AS (
  SELECT
    run_id,
    pooled_wape AS baseline_wape
  FROM holdout_scores
  WHERE NOT STARTS_WITH(model_type, 'ensemble_') AND ens_key = '' AND pooled_wape IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY run_id
    ORDER BY
      IF(STARTS_WITH(LOWER(SPLIT(model_type, '@')[SAFE_OFFSET(0)]), 'naive'), 0, 1) ASC,
      IF(
        STARTS_WITH(LOWER(SPLIT(model_type, '@')[SAFE_OFFSET(0)]), 'naive'),
        pooled_wape,
        -pooled_wape
      ) ASC,
      model_type ASC
  ) = 1
),
forward_rollup AS (
  SELECT
    p.run_id,
    MIN(p.forecast_date) AS horizon_start_date,
    MAX(p.forecast_date) AS horizon_end_date,
    SUM(p.yhat) AS total_forecast_units,
    AVG(p.yhat_upper - p.yhat_lower) AS mean_interval_width
  FROM pred_deduped AS p
  JOIN champion AS c
    ON p.run_id = c.run_id AND p.model_type = c.champion_model AND p.ens_key = c.ens_key
  GROUP BY p.run_id
),
meta_deduped AS (
  SELECT
    run_id,
    ts_id,
    model_type,
    COALESCE(ensemble_id, '') AS ens_key,
    cell_status,
    mase,
    staleness_gap
  FROM `proj.scale_forecasting.forecast_metadata`
  WHERE fold_id IS NULL
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY run_id, ts_id, model_type, fold_id, ensemble_id
    ORDER BY created_at DESC NULLS LAST
  ) = 1
),
oof_ts AS (
  SELECT
    run_id,
    ts_id,
    model_type,
    COALESCE(ensemble_id, '') AS ens_key,
    SAFE_DIVIDE(SUM(yhat - y_true), NULLIF(AVG(ABS(y_true - yhat)), 0.0)) AS tracking_signal
  FROM oof_deduped
  GROUP BY run_id, ts_id, model_type, ens_key
),
exceptions_rollup AS (
  SELECT
    m.run_id,
    COUNTIF(
      m.cell_status = 'error'
      OR ABS(t.tracking_signal) > 4.0
      OR m.mase > 1.0
      OR m.staleness_gap > 0.05
    ) AS n_exception_series,
    SAFE_DIVIDE(
      COUNTIF(
        m.cell_status = 'error'
        OR ABS(t.tracking_signal) > 4.0
        OR m.mase > 1.0
        OR m.staleness_gap > 0.05
      ),
      COUNT(*)
    ) AS exception_rate
  FROM meta_deduped AS m
  JOIN champion AS c
    ON m.run_id = c.run_id AND m.model_type = c.champion_model AND m.ens_key = c.ens_key
  LEFT JOIN oof_ts AS t
    ON m.run_id = t.run_id
    AND m.ts_id = t.ts_id
    AND m.model_type = t.model_type
    AND m.ens_key = t.ens_key
  GROUP BY m.run_id
),
gpu_audit AS (
  SELECT
    run_id,
    COUNTIF(
      hardware = 'gpu'
      AND JSON_VALUE(job_telemetry, '$.device_use.verdict') IS NOT NULL
      AND JSON_VALUE(job_telemetry, '$.device_use.verdict') != 'ENGAGED_UTILISED'
    ) AS n_gpu_waste_jobs
  FROM (
    SELECT *
    FROM `proj.scale_forecasting.run_jobs`
    QUALIFY ROW_NUMBER() OVER (
      PARTITION BY run_id, family ORDER BY attempt DESC, created_at DESC
    ) = 1
  )
  GROUP BY run_id
)
SELECT
  h.run_id,
  h.created_at,
  h.status,
  h.source_table,
  h.n_series,
  h.n_models,
  h.runtime_seconds,
  c.champion_model,
  c.champion_pooled_wape,
  c.champion_pooled_mae,
  b.baseline_wape,
  100.0 * SAFE_DIVIDE(b.baseline_wape - c.champion_pooled_wape, b.baseline_wape)
    AS fva_relative_lift_pct,
  f.horizon_start_date,
  f.horizon_end_date,
  f.total_forecast_units,
  f.mean_interval_width,
  COALESCE(e.n_exception_series, 0) AS n_exception_series,
  e.exception_rate,
  COALESCE(g.n_gpu_waste_jobs, 0) AS n_gpu_waste_jobs
FROM header AS h
LEFT JOIN champion AS c USING (run_id)
LEFT JOIN baseline AS b USING (run_id)
LEFT JOIN forward_rollup AS f USING (run_id)
LEFT JOIN exceptions_rollup AS e USING (run_id)
LEFT JOIN gpu_audit AS g USING (run_id);