"""Canonical table, view, and column descriptions for BigQuery & Knowledge Catalog.

Every table created by `ddl.py` (and `data_gen/seed_spark.py`) and every analytical SQL view
created by `views.py` draws its ``OPTIONS(description='...')`` metadata from this single module.

Why one dictionary module instead of inline prose scattered across SQL strings:

1. **Zero drift between new and existing registries:** New deployments embed these exact
   descriptions directly inside ``CREATE TABLE IF NOT EXISTS`` and ``CREATE OR REPLACE VIEW`` DDL,
   while `sync_existing_descriptions` (called by `tables.ensure_tables` and `tables.ensure_views`)
   backfills missing or updated table, view, and column descriptions onto pre-existing registries
   created by earlier releases.
2. **Strict 1-to-1 column alignment:** `column_description` raises ``KeyError`` when a table or
   view declares a column with no entry here, and unit tests in `test_ddl.py` and `test_views.py`
   verify exact bidirectional parity so positional view headers can never shift or misalign.
3. **Rich JSON path guidance for SQL analysts & BigQuery Data Agents:** Native ``JSON`` columns
   (`raw_config`, `job_telemetry`, `best_params`, `fit_diagnostics`, `quantiles`, `explanations`)
   document their exact `JSON_VALUE` / `JSON_QUERY` paths right in the column description so
   Knowledge Catalog and Conversational Analytics API agents generate valid BigQuery SQL on the
   first turn.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..metrics import METRIC_DIRECTION, METRIC_NAMES

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Table & View Top-Level Descriptions
# ---------------------------------------------------------------------------

TABLE_DESCRIPTIONS: dict[str, str] = {
    "run_registry": (
        "scale-forecasting run header ledger (append-only, partitioned by created_at). "
        "Records deterministic run_id, BigQuery snapshot_millis time-travel anchor, "
        "verbatim RunConfig JSON ($.data.source_table, $.models, $.backtest), status, "
        "panel scale, and run-level sizing/capacity telemetry. Deduplicate to latest row per "
        "run_id via QUALIFY ROW_NUMBER() OVER (PARTITION BY run_id ORDER BY created_at DESC) = 1, "
        "or query v_run_summary."
    ),
    "run_jobs": (
        "Per-family cloud job execution trace (append-only, 1 row per attempt of statistical, "
        "ml, deep_learning, automl, native, or ensemble). Records resolved compute runtime, "
        "hardware/GPU shape, platform job ID, wall-clock timestamps, failure_reason, and "
        "job_telemetry JSON ($.platform.provision_s, $.platform.execute_s, $.device_use.verdict). "
        "Query v_run_jobs for the current attempt per (run_id, family)."
    ),
    "forecast_metadata": (
        "Per-(run_id, ts_id, model_type) fit diagnostics, resource telemetry, and all 21 "
        "evaluation metrics. Rows with fold_id IS NULL summarize the full-panel backtest and "
        "final production fit; rows with a non-null fold_id record per-fold scores. "
        "Deduplicate retries via QUALIFY ROW_NUMBER() OVER (PARTITION BY run_id, ts_id, "
        "model_type, fold_id, ensemble_id ORDER BY created_at DESC NULLS LAST) = 1."
    ),
    "forecast_predictions": (
        "Forward horizon point and probabilistic forecasts per (run_id, ts_id, model_type, "
        "forecast_date). Carries operational yhat, unadjusted yhat_raw, bias-corrected "
        "yhat_adjusted, calibrated [yhat_lower, yhat_upper] bounds, multi-percentile quantiles "
        "JSON (e.g. JSON_VALUE(quantiles, '$.\"0.5\"')), and per-step explanations JSON."
    ),
    "backtest_oof": (
        "Snapshot-locked out-of-fold (OOF) historical backtest "
        "predictions and ground-truth actuals (y_true) per (run_id, "
        "ts_id, model_type, fold_id, forecast_date). Preserves exact "
        "as-of-launch actuals even after BigQuery time-travel windows "
        "expire; drives learned stacking ensembles, conformal "
        "calibration, and v_model_leaderboard_comparable."
    ),
    "source_series_iceberg": (
        "Shipped univariate benchmark source panel stored as a BigQuery-managed Apache Iceberg "
        "table over Parquet in GCS (ts_id, ds, y, archetype, is_holiday)."
    ),
    "source_series_native": (
        "Shipped univariate benchmark source panel stored as a native BigQuery table "
        "partitioned by ds and clustered by ts_id (ts_id, ds, y, archetype, is_holiday)."
    ),
    "source_series_covariates_iceberg": (
        "Shipped 10-column multivariate & hierarchical benchmark source panel stored as a "
        "BigQuery-managed Apache Iceberg table (ts_id, ds, y, archetype, is_holiday, region, "
        "category, promo_flag, price_index, temperature)."
    ),
    "source_series_covariates_native": (
        "Shipped 10-column multivariate & hierarchical benchmark source panel stored as a "
        "native BigQuery table partitioned by ds and clustered by ts_id (ts_id, ds, y, archetype, "
        "is_holiday, region, category, promo_flag, price_index, temperature)."
    ),
}

VIEW_DESCRIPTIONS: dict[str, str] = {
    "v_run_summary": (
        "Curated one-row-per-run summary deduplicating run_registry to the latest header and "
        "rolling up the active launch job time ledger (n_jobs, longest_job_seconds, jobs_seconds, "
        "jobs_span_seconds, overhead_seconds, overhead_fraction) plus Dataproc/sizing telemetry."
    ),
    "v_run_jobs": (
        "Curated per-(run_id, family) execution trace keeping only the latest attempt per family "
        "and unpacking provision_seconds, execute_seconds, device_verdict (ENGAGED_UTILISED vs "
        "IDLE_WASTE), probe_handle, and regional capacity failover history."
    ),
    "v_model_leaderboard": (
        "Deduplicated per-(run_id, model_type, ensemble_id) model summary over full-fit metadata "
        "(fold_id IS NULL): cell counts, missing-artifact error rate, median fit seconds, "
        "series-averaged mean_wape and mean_mae, mean_staleness_gap, and training/covariate tags."
    ),
    "v_backtest_coverage": (
        "Achieved backtest fold coverage cohort table per (run_id, model_type, ensemble_id, "
        "backtest_status, n_folds_achieved, backtest_refit) showing series counts and panel share."
    ),
    "v_model_leaderboard_comparable": (
        "Apples-to-apples model leaderboard restricted to the newest holdout fold in backtest_oof "
        "and computed via panel-pooled error: pooled_wape = SUM(|y_true - yhat|) / SUM(|y_true|) "
        "and pooled_mae = AVG(|y_true - yhat|). Preferred view for ranking models and ensembles."
    ),
    "v_fva_waterfall": (
        "Forecast Value Added (FVA) waterfall per (run_id, model_type, ensemble_id) on the "
        "comparable holdout fold. Classifies each model into pipeline stages (1_baseline_naive -> "
        "2_statistical -> 3_univariate_ml_dl -> 4_covariate_ml_dl -> 5_ensemble -> 6_reconciled) "
        "and measures absolute WAPE reduction and relative % FVA lift vs. the run's naive baseline."
    ),
    "v_horizon_decay": (
        "Lead-time accuracy decay curve per (run_id, model_type, ensemble_id, horizon_step) "
        "computed from deduplicated backtest_oof: pooled_wape, pooled_mae, signed_bias, "
        "empirical_coverage, and mean_interval_width at each step h = 1..H ahead."
    ),
    "v_bias_and_exceptions": (
        "Planner exception triage queue per (run_id, ts_id, model_type, "
        "ensemble_id). Computes Trigg's tracking_signal (cumulative "
        "signed error / MAD), series WAPE, MASE, staleness_gap, and SBC "
        "demand regime, flagging CHRONIC_OVER_FORECAST, "
        "CHRONIC_UNDER_FORECAST, WORSE_THAN_NAIVE, STALE_REFIT_DRIFT, "
        "and CELL_FIT_ERROR."
    ),
    "v_anomalies": (
        "Historical out-of-fold demand anomalies and interval breaches per (run_id, ts_id, "
        "model_type, ensemble_id, forecast_date) where |residual_zscore| >= 2.5 or y_true falls "
        "outside calibrated [yhat_lower, yhat_upper] bounds."
    ),
    "v_feature_attributions": (
        "Unified exogenous covariate and lag driver attributions per (run_id, ts_id, model_type, "
        "attribution_scope, feature_name), unpacking both global model importances "
        "(fit_diagnostics.feature_attributions) and horizon-averaged signed SHAP attributions "
        "(forecast_predictions.explanations.attributions)."
    ),
    "v_executive_sop_summary": (
        "One-row-per-run Executive S&OP Scorecard combining panel scale, winning model from "
        "v_model_leaderboard_comparable, FVA % lift over naive baseline, forward horizon total "
        "forecasted volume and uncertainty band width, planner exception rate, and GPU audit count."
    ),
}

# ---------------------------------------------------------------------------
# 2. Metric Column Descriptions (21 Panel Metrics on forecast_metadata)
# ---------------------------------------------------------------------------

METRIC_COLUMN_DESCRIPTIONS: dict[str, str] = {
    "mae": "Mean Absolute Error: MEAN(|yhat - y_true|) in target units (lower is better).",
    "rmse": (
        "Root Mean Squared Error: SQRT(MEAN((yhat - y_true)^2)) in target units (lower is better)."
    ),
    "mse": "Mean Squared Error: MEAN((yhat - y_true)^2) in squared target units (lower is better).",
    "mape": (
        "Mean Absolute Percentage Error: MEAN(|yhat - y_true| / "
        "|y_true|); NULL/NaN when any y_true = 0 (lower is better)."
    ),
    "smape": (
        "Symmetric Mean Absolute Percentage Error: MEAN(2 * |yhat - "
        "y_true| / (|y_true| + |yhat|)), bounded in [0, 2] (lower is better)."
    ),
    "wape": (
        "Weighted Absolute Percentage Error: SUM(|yhat - y_true|) / "
        "SUM(|y_true|) across backtest folds (lower is better; 0.08 = 8% "
        "error)."
    ),
    "mase": (
        "Mean Absolute Scaled Error vs. one-step naive random walk on "
        "training history (< 1.0 beats naive; lower is better)."
    ),
    "rmsse": (
        "Root Mean Squared Scaled Error (M5 metric) vs. one-step naive "
        "squared step on training history (< 1.0 beats naive; lower is "
        "better)."
    ),
    "bias": (
        "Mean signed forecast error MEAN(yhat - y_true); positive = "
        "over-forecasting, negative = under-forecasting (closer to 0 is "
        "better)."
    ),
    "coverage": (
        "Empirical prediction interval coverage: fraction of y_true "
        "inside [yhat_lower, yhat_upper] (closer to nominal "
        "confidence_level e.g. 0.80 is better)."
    ),
    "pinball": (
        "Mean pinball (quantile) loss across the lower and upper "
        "prediction interval bounds (lower is better)."
    ),
    "mase_seasonal": (
        "Seasonal Mean Absolute Scaled Error vs. seasonal naive (lag = "
        "seasonal_period) on training history (< 1.0 beats seasonal "
        "naive; lower is better)."
    ),
    "maape": (
        "Mean Arctangent Absolute Percentage Error: MEAN(ARCTAN(|yhat - "
        "y_true| / |y_true|)), finite even when y_true = 0 (lower is better)."
    ),
    "interval_score": (
        "Mean Winkler interval score penalizing both interval width and "
        "out-of-bounds exceedance (lower is better)."
    ),
    "interval_width": (
        "Mean prediction interval width MEAN(yhat_upper - yhat_lower) in "
        "target units (diagnostic; read alongside coverage)."
    ),
    "ope": (
        "Overall Percentage Error: |SUM(yhat) - SUM(y_true)| / "
        "|SUM(y_true)| measuring cumulative horizon bias magnitude "
        "(lower is better)."
    ),
    "rmsle": (
        "Root Mean Squared Logarithmic Error: SQRT(MEAN((LN(1 + yhat) - "
        "LN(1 + y_true))^2)) for non-negative series (lower is better)."
    ),
    "msse": (
        "Mean Squared Scaled Error: MSE divided by mean squared one-step "
        "naive error on training history (lower is better)."
    ),
    "msis": (
        "Mean Scaled Interval Score (M4/M5 interval metric): Winkler "
        "interval_score scaled by seasonal naive MAE (< 1.0 is sharp and "
        "calibrated; lower is better)."
    ),
    "r2": (
        "Coefficient of determination: 1 - SUM((y_true - yhat)^2) / "
        "SUM((y_true - MEAN(y_true))^2) (higher is better; 1.0 is perfect)."
    ),
    "cv": "Coefficient of Variation of RMSE: RMSE / |MEAN(y_true)| (lower is better).",
}


def metric_description(name: str) -> str:
    """Return the column description for ``name`` in `METRIC_NAMES`."""
    if name in METRIC_COLUMN_DESCRIPTIONS:
        return METRIC_COLUMN_DESCRIPTIONS[name]
    direction = METRIC_DIRECTION.get(name, "lower")
    return f"Evaluation metric '{name}' across held-out backtest folds ({direction} is better)."


# ---------------------------------------------------------------------------
# 3. Column-Level Descriptions for Every Table & View
# ---------------------------------------------------------------------------

_SOURCE_BASE_COLS: dict[str, str] = {
    "ts_id": "Unique time-series identifier (e.g. SKU, store-SKU, meter, well, or lane ID).",
    "ds": "Observation date/timestamp in regular panel frequency (partition key on native tables).",
    "y": "Observed historical target value (e.g. unit demand, load MW, production volume).",
    "archetype": "Synthetic or domain demand archetype label describing series dynamics.",
    "is_holiday": "Boolean calendar flag indicating whether ds falls on a holiday event.",
}

_SOURCE_COV_COLS: dict[str, str] = {
    **_SOURCE_BASE_COLS,
    "region": (
        "Geographic hierarchy attribute (e.g. North, South, East, West) "
        "for spatial rollups and static covariates."
    ),
    "category": (
        "Product/asset category hierarchy attribute for hierarchical "
        "reconciliation and static covariates."
    ),
    "promo_flag": (
        "Binary future-known promotional indicator (1 = active promotion on ds, 0 = baseline)."
    ),
    "price_index": (
        "Relative unit price index covariate (1.0 = regular baseline price, < 1.0 = discount)."
    ),
    "temperature": (
        "Observed or forecasted ambient temperature covariate for weather-sensitive demand."
    ),
}

COLUMN_DESCRIPTIONS: dict[str, dict[str, str]] = {
    # --- 5 Registry Tables ---
    "run_registry": {
        "run_id": (
            "Deterministic content-addressed run identifier "
            "(<run_name_slug>-<12hex>) hashed from semantic RunConfig inputs."
        ),
        "created_at": (
            "UTC timestamp when this run header row was written (partition "
            "key; latest row per run_id wins)."
        ),
        "snapshot_millis": (
            "BigQuery epoch milliseconds snapshot anchor recorded at run "
            "start for reproducible FOR SYSTEM_TIME AS OF reads."
        ),
        "user_id": "Identity of the principal or service account that launched the run.",
        "git_sha": "Git commit SHA of the scale-forecasting code that launched the run.",
        "python_runtime": "Python version string of the launching orchestrator.",
        "bq_models": (
            "BQML model FQDNs created during native BigQuery runs (tracked "
            "for drop-run and orphan cleanup)."
        ),
        "backtest_on": (
            "True when rolling/expanding walk-forward backtesting was enabled for this run."
        ),
        "decision_metric": (
            "Primary metric name (from METRIC_NAMES, e.g. wape, mase) used "
            "for HPO, champion selection, and inverse-error weights."
        ),
        "ensemble_strategies": (
            "Configured ensemble blending/stacking strategies (e.g. mean, "
            "inverse_error, champion, nnls, ridge, xgb)."
        ),
        "raw_config": (
            "Complete validated RunConfig JSON payload. Key paths: "
            "$.data.source_table, $.data.horizon, $.models, $.hierarchy.cols."
        ),
        "status": (
            "Lifecycle status of the run: STAGED, RUNNING, COMPLETED, "
            "PARTIAL, FAILED, or CANCELLED."
        ),
        "n_series": "Total number of distinct time series (ts_id) scored in this run.",
        "n_models": "Number of configured base models/variants in this run.",
        "runtime_seconds": "End-to-end orchestrator wall-clock duration in seconds.",
        "job_telemetry": (
            "Run-level telemetry JSON: $.sizing (per-family worker/machine "
            "decisions), $.capacity (failover ledger), $.version."
        ),
    },
    "run_jobs": {
        "job_id": (
            "Deterministic job identifier (<run_id>-<family>-a<attempt>) "
            "used to deduplicate cloud submissions."
        ),
        "run_id": "Parent run_id joining to run_registry.run_id.",
        "family": (
            "Model family or barrier stage executed by this job: "
            "statistical, ml, deep_learning, automl, native, or ensemble."
        ),
        "attempt": (
            "1-based execution attempt counter for (run_id, family); highest attempt is current."
        ),
        "runtime": (
            "Compute engine that executed this family: spark, ray, vertex, "
            "gce, gke, vertex_automl, or bigquery."
        ),
        "spark_mode": (
            "Spark execution mode when runtime='spark' (serverless, cluster, "
            "connect, or local); NULL otherwise."
        ),
        "hardware": "Requested hardware tier for this job: cpu or gpu.",
        "gpu_type": (
            "Accelerator type when hardware='gpu': T4, L4, A100, or A100_80GB; NULL on CPU."
        ),
        "system_job_id": (
            "Cloud provider resource name/ID (Dataproc batch ID, Vertex "
            "CustomJob, GCE instance, GKE Job, or BQ job prefix)."
        ),
        "status": (
            "Job execution status: EMITTED, RUNNING, COMPLETED, PARTIAL, FAILED, or CANCELLED."
        ),
        "created_at": "UTC timestamp when this job trace row was written.",
        "started_at": "UTC timestamp when job submission began.",
        "ended_at": "UTC timestamp when the job reached a terminal state.",
        "runtime_seconds": (
            "Launcher-measured wall-clock seconds from submission to job completion."
        ),
        "failure_reason": (
            "Structured failure code and diagnostic summary when "
            "status='FAILED' (e.g. CAPACITY_EXHAUSTED); NULL on success."
        ),
        "job_telemetry": (
            "Per-job telemetry JSON: $.platform.provision_s, "
            "$.platform.execute_s, $.device_use.verdict, $.probe_handle, "
            "$.capacity."
        ),
    },
    "forecast_metadata": {
        "run_id": "Parent run_id joining to run_registry.run_id.",
        "ts_id": "Time-series identifier scored in this cell.",
        "model_type": (
            "Model or variant identifier (e.g. theta, lightgbm@monotone, "
            "ensemble_champion, lightgbm_cross_temporal_mint)."
        ),
        "compute_engine": (
            "Compute runtime that produced this cell (spark, ray, vertex, "
            "gce, gke, vertex_automl, bigquery, or ensemble)."
        ),
        "model_hash": "Deterministic hash of model_type and hyperparameters.",
        "ensemble_id": "Configuration digest for ensemble_* pseudo-models; NULL for base models.",
        "fold_id": (
            "NULL for the full-fit summary row (one per cell); 0..n_folds-1 "
            "for individual walk-forward backtest fold rows."
        ),
        **{m: metric_description(m) for m in METRIC_NAMES},
        "fit_seconds": (
            "Wall-clock seconds spent fitting and backtesting this (ts_id, model_type) cell."
        ),
        "best_params": (
            "Fitted or Optuna-tuned model parameters and conformal "
            "calibration metadata as native JSON."
        ),
        "model_artifact": (
            "GCS URI (gs://...) of the serialized fitted model artifact when "
            "artifact persistence is enabled; NULL otherwise."
        ),
        "created_at": (
            "UTC timestamp when this metadata row was appended (newest write "
            "wins on deduplication)."
        ),
        "worker_id": "Identifier of the distributed worker/executor/pod that processed this cell.",
        "cell_started_at": "UTC timestamp when worker execution began for this cell.",
        "cell_ended_at": "UTC timestamp when worker execution completed for this cell.",
        "cpu_seconds": "Process CPU seconds consumed while fitting this cell.",
        "process_rss_bytes": (
            "Peak worker resident set size (RSS) in bytes observed during cell execution."
        ),
        "peak_gpu_bytes": (
            "Peak CUDA GPU memory allocated in bytes during cell execution (NULL or 0 on CPU)."
        ),
        "intraop_threads": (
            "BLAS/OpenMP/PyTorch intra-op thread cap enforced during cell execution."
        ),
        "n_obs": "Number of historical training observations available for this series.",
        "cell_status": (
            "Cell completion outcome: 'ok' when fit succeeded or 'error' when fitting failed."
        ),
        "error_class": "Python exception class name when cell_status='error'; NULL on success.",
        "error_detail": "Sanitized error message when cell_status='error'; NULL on success.",
        "backtest_status": (
            "Backtest fold achievement status: full, reduced, unscored, "
            "failed, or NULL when backtest is disabled."
        ),
        "backtest_note": (
            "Diagnostic explanation when backtest_status is reduced, unscored, or failed."
        ),
        "n_folds_achieved": (
            "Number of walk-forward validation folds actually evaluated for this series."
        ),
        "achieved_step": (
            "Effective step stride (in periods) between consecutive validation fold cutoffs."
        ),
        "achieved_min_train": (
            "Minimum training window length (in periods) achieved on the earliest fold."
        ),
        "first_val_date": "Earliest validation date evaluated across backtest folds for this cell.",
        "last_val_date": "Latest validation date evaluated across backtest folds for this cell.",
        "interval_source": (
            "Origin of raw prediction intervals prior to conformal "
            "calibration (e.g. native, residual)."
        ),
        "point_forecast_source": (
            "Origin of operational yhat (e.g. raw, median_bias_adjusted, target_quantile)."
        ),
        "point_forecast_decision": (
            "Audit decision explaining whether OOF residual bias adjustment "
            "was applied or kept raw."
        ),
        "interval_calibration": (
            "Interval calibration method applied (e.g. conformal_oof, aci, raw)."
        ),
        "point_forecast_margin": "OOF metric change produced by point-forecast bias adjustment.",
        "ensemble_scoring": "Scoring window mode used for ensemble weight estimation.",
        "hpo_scoring": (
            "HPO validation discipline: 'held_out' (newest fold reserved "
            "from tuning) or 'in_sample'."
        ),
        "n_fits": (
            "Total number of model fit calls executed for this cell across "
            "backtest folds and final fit."
        ),
        "train_rows_total": "Cumulative training rows processed across all fits for this cell.",
        "n_hpo_fits": "Number of Optuna hyperparameter trial fits executed for this cell.",
        "fit_diagnostics": (
            "Structured JSON diagnostics: $.base_model, $.training_mode, "
            "$.cov_profile, $.covariates_used, $.feature_attributions "
            "(global feature importances), $.demand_profile.sbc_class "
            "(smooth/erratic/intermittent/lumpy), $.borrowed_from_run_id."
        ),
        "device_requested": "Requested accelerator device for this cell (cpu or cuda).",
        "device_available": "Accelerator device detected on the worker host.",
        "device_used": "Actual device used by the model fit (cpu or cuda).",
        "device_name": "Hardware model name of the CPU/GPU device (e.g. NVIDIA L4).",
        "backtest_refit": (
            "Refit behavior executed across backtest folds: per_fold, "
            "recondition (frozen), or unsupported."
        ),
        "staleness_gap": (
            "Accuracy loss (in decision_metric units) of a never-refit "
            "frozen model vs. per-fold refitting."
        ),
    },
    "forecast_predictions": {
        "run_id": "Parent run_id joining to run_registry.run_id.",
        "ts_id": "Time-series identifier.",
        "model_type": "Model, variant, ensemble (ensemble_*), or reconciled model identifier.",
        "compute_engine": "Compute runtime that generated this forecast.",
        "ensemble_id": "Ensemble configuration digest for ensemble_* rows; NULL for base models.",
        "forecast_date": "Future target date being forecasted (partition key).",
        "yhat": (
            "Primary operational point forecast (incorporating calibration, "
            "bounds, and rounding when configured)."
        ),
        "yhat_raw": "Unadjusted raw point forecast emitted directly by the underlying model.",
        "yhat_adjusted": (
            "OOF-calibrated point forecast prior to physical floor/cap/rounding constraints."
        ),
        "yhat_lower": (
            "Calibrated lower prediction interval bound (e.g. P10 at 80% confidence_level)."
        ),
        "yhat_upper": (
            "Calibrated upper prediction interval bound (e.g. P90 at 80% confidence_level)."
        ),
        "quantiles": (
            "Monotone multi-quantile forecast fan as native JSON keyed by "
            "probability string, e.g. JSON_VALUE(quantiles, '$.\"0.1\"'), "
            "'$.\"0.5\"', '$.\"0.9\"'."
        ),
        "created_at": (
            "UTC timestamp when this prediction row was written (latest "
            "write wins on deduplication)."
        ),
        "explanations": (
            "Per-horizon-step local feature attributions JSON: "
            "$.baseline_score and $.attributions.<feature_name>."
        ),
    },
    "backtest_oof": {
        "run_id": "Parent run_id joining to run_registry.run_id.",
        "ts_id": "Time-series identifier.",
        "model_type": "Model, variant, ensemble (ensemble_*), or reconciled model identifier.",
        "fold_id": (
            "0-based walk-forward validation fold index; MAX(fold_id) per "
            "run_id is the newest holdout fold."
        ),
        "forecast_date": "Historical validation date scored in this backtest fold (partition key).",
        "y_true": "Snapshot-locked ground-truth actual observed at forecast_date as of run launch.",
        "yhat": "Operational out-of-fold point forecast evaluated at forecast_date.",
        "yhat_raw": "Unadjusted raw out-of-fold model prediction.",
        "yhat_adjusted": "Bias-adjusted out-of-fold prediction prior to output clamping.",
        "cutoff_date": (
            "Last training timestamp visible to the model in this fold (forecast origin)."
        ),
        "horizon_step": (
            "1-based lead-time step ahead from cutoff_date (1 = 1 step ahead, H = full horizon)."
        ),
        "yhat_lower": "Out-of-fold lower prediction interval bound.",
        "yhat_upper": "Out-of-fold upper prediction interval bound.",
        "created_at": (
            "UTC timestamp when this OOF row was written (latest write wins on deduplication)."
        ),
        "ensemble_id": "Ensemble configuration digest for ensemble_* rows; NULL for base models.",
        "yhat_stale": (
            "Counterfactual frozen-model prediction when "
            "backtest.control_arm=true (used to measure staleness_gap)."
        ),
    },
    # --- 4 Source Tables ---
    "source_series_iceberg": dict(_SOURCE_BASE_COLS),
    "source_series_native": dict(_SOURCE_BASE_COLS),
    "source_series_covariates_iceberg": dict(_SOURCE_COV_COLS),
    "source_series_covariates_native": dict(_SOURCE_COV_COLS),
    # --- 5 Core Analytical SQL Views ---
    "v_run_summary": {
        "run_id": "Deterministic run identifier (1 deduplicated row per run).",
        "created_at": "UTC timestamp of the latest header write for this run.",
        "status": (
            "Current run lifecycle status (COMPLETED, PARTIAL, FAILED, RUNNING, STAGED, CANCELLED)."
        ),
        "python_runtime": "Python runtime version of the orchestrator.",
        "n_series": "Number of distinct time series in the run.",
        "n_models": "Number of configured base models/variants.",
        "backtest_on": "Whether walk-forward backtesting was enabled.",
        "runtime_seconds": "Total orchestrator wall-clock seconds.",
        "n_jobs": "Number of non-repair family jobs executed in this launch.",
        "longest_job_seconds": "Critical-path duration (seconds) of the slowest family job.",
        "jobs_seconds": "Sum of wall-clock seconds across all parallel family jobs.",
        "jobs_span_seconds": "Elapsed seconds from earliest job start to latest job end.",
        "overhead_seconds": (
            "Non-critical-path orchestration/ensemble/native overhead "
            "(jobs_span_seconds - longest_job_seconds >= 0)."
        ),
        "overhead_fraction": "Share of total job span spent outside the single longest family job.",
        "executor_instances": (
            "Dataproc Serverless initial executor count when stamped on header; NULL otherwise."
        ),
        "executor_cores": "Cores per Spark executor when applicable.",
        "max_executors": "Maximum autoscaling Spark executors when applicable.",
        "executor_memory": "Memory per Spark executor when applicable.",
        "executor_memory_overhead": "Off-heap memory overhead per Spark executor when applicable.",
        "dcu_milli_seconds": "Dataproc Serverless Data Compute Unit (DCU) milli-seconds consumed.",
        "runtime_version": "Cloud runtime image/platform version.",
        "sizing": (
            "Per-family auto-sizing audit JSON ($.<family> machine_type, "
            "workers, memory rationale)."
        ),
        "capacity": "Shared cluster regional capacity failover ledger JSON.",
    },
    "v_run_jobs": {
        "run_id": "Parent run identifier.",
        "family": "Model family or ensemble stage (1 deduplicated row per current attempt).",
        "job_id": "Deterministic job identifier for the current attempt.",
        "attempt": "Current (highest) attempt number for (run_id, family).",
        "runtime": (
            "Resolved cloud compute runtime (spark, ray, vertex, gce, gke, "
            "vertex_automl, bigquery)."
        ),
        "spark_mode": "Spark deployment mode when runtime='spark'.",
        "hardware": "Hardware tier (cpu or gpu).",
        "gpu_type": "GPU model (T4, L4, A100, A100_80GB) when hardware='gpu'.",
        "system_job_id": "Cloud platform native job/instance identifier.",
        "status": "Current job status.",
        "created_at": "UTC timestamp when the job trace row was written.",
        "started_at": "UTC timestamp when job submission started.",
        "ended_at": "UTC timestamp when job execution finished.",
        "runtime_seconds": "Total wall-clock seconds for this job.",
        "provision_seconds": (
            "Cloud VM/containerstartup and provisioning wait seconds before code execution began."
        ),
        "execute_seconds": (
            "Active model training/scoring execution seconds once compute provisioned."
        ),
        "failure_reason": "Failure classification token and message when status='FAILED'.",
        "device_verdict": (
            "GPU utilization audit verdict (e.g. ENGAGED_UTILISED vs "
            "IDLE_WASTE); NULL for CPU jobs."
        ),
        "device_use": "Detailed accelerator usage counts and peak GPU memory JSON.",
        "probe_handle": "Cloud coordinates JSON used by live runtime probes and cancellation.",
        "capacity": "Per-job regional/zonal capacity failover attempt ledger JSON.",
    },
    "v_model_leaderboard": {
        "run_id": "Run identifier.",
        "model_type": "Model, variant, ensemble, or reconciled model name.",
        "ensemble_id": "Ensemble configuration hash for ensemble_* rows; NULL for base models.",
        "base_model": "Underlying plugin name stripped of @variant suffixes.",
        "scope_capability": (
            "Model training scope capability (local_only, global_only, multi_scope)."
        ),
        "training_mode": "Executed training scope (local, global, hybrid).",
        "cov_profile": "Active covariate profile tag (univariate, cov_all, cov_drop_*, etc.).",
        "is_univariate": "True when the model ran without exogenous covariates.",
        "covariates_used": (
            "Comma-separated list of exogenous covariate columns consumed by the model."
        ),
        "model_spec_hash": (
            "Cross-run reusable specification hash for snapshot-verified cell borrowing."
        ),
        "borrowed_from_run_id": (
            "Source run_id when cells were borrowed via reuse_prior_runs; NULL when fit fresh."
        ),
        "n_borrowed": "Count of series cells borrowed from a prior run.",
        "compute_engine": "Compute engine that produced the model's cells.",
        "n_cells": "Number of deduplicated series cells landed for this model.",
        "n_no_artifact": "Count of cells with NULL model_artifact.",
        "no_artifact_rate": "Fraction of cells without a persisted GCS artifact.",
        "median_fit_seconds": "Median per-series fit duration in seconds.",
        "mean_wape": (
            "Unweighted mean of per-series WAPEs (for volume-weighted panel "
            "WAPE use v_model_leaderboard_comparable.pooled_wape)."
        ),
        "mean_mae": "Mean of per-series MAEs across scored cells.",
        "mean_staleness_gap": "Mean accuracy degradation under frozen refit vs. per-fold refit.",
        "refit_modes": "Distinct backtest refit modes observed across the model's cohort.",
    },
    "v_backtest_coverage": {
        "run_id": "Run identifier.",
        "model_type": "Model, variant, or ensemble identifier.",
        "ensemble_id": "Ensemble configuration hash; NULL for base models.",
        "backtest_status": "Cohort fold status: full, reduced, unscored, or failed.",
        "n_folds_achieved": "Exact number of walk-forward folds achieved by series in this cohort.",
        "backtest_refit": (
            "Refit discipline executed for this cohort (per_fold, recondition, unsupported)."
        ),
        "n_series": "Count of distinct series belonging to this coverage cohort.",
        "mean_staleness_gap": "Mean frozen-vs-refit staleness gap for this cohort.",
        "series_share": (
            "Fraction of the model's total series panel represented by this "
            "cohort (sums to 1.0 per model)."
        ),
    },
    "v_model_leaderboard_comparable": {
        "run_id": "Run identifier.",
        "model_type": "Model, variant, ensemble (ensemble_*), or reconciled model identifier.",
        "ensemble_id": "Ensemble configuration hash; NULL for base models.",
        "holdout_fold_id": (
            "Fold ID of the newest holdout fold evaluated across all models in the run."
        ),
        "n_series": (
            "Number of distinct series evaluated on the holdout fold (equal "
            "n_series confirms apples-to-apples comparison)."
        ),
        "n_points": "Total (ts_id, forecast_date) evaluation points pooled on the holdout fold.",
        "pooled_wape": (
            "Volume-weighted panel WAPE on the holdout fold: SUM(|y_true - "
            "yhat|) / SUM(|y_true|) (lower is better)."
        ),
        "pooled_mae": (
            "Pooled Mean Absolute Error AVG(|y_true - yhat|) on the holdout fold (lower is better)."
        ),
        "first_forecast_date": "Start date of the holdout fold evaluation window.",
        "last_forecast_date": "End date of the holdout fold evaluation window.",
        "refit_modes": "Distinct backtest refit modes across the model's cells.",
    },
    # --- 6 New Diagnostic Analytical SQL Views ---
    "v_fva_waterfall": {
        "run_id": "Run identifier.",
        "stage_order": (
            "Integer pipeline complexity order: 1=Baseline Naive, "
            "2=Statistical, 3=Univariate ML/DL, 4=Covariate ML/DL/AutoML, "
            "5=Ensemble, 6=Reconciled."
        ),
        "stage_name": "Human-readable FVA pipeline stage label corresponding to stage_order.",
        "model_type": "Model, variant, ensemble, or reconciled model name.",
        "ensemble_id": "Ensemble configuration hash; NULL for base models.",
        "n_series": "Number of series scored on the holdout fold.",
        "n_points": "Number of OOF observations pooled on the holdout fold.",
        "pooled_wape": "Model's volume-weighted holdout-fold WAPE.",
        "pooled_mae": "Model's holdout-fold pooled MAE.",
        "baseline_model": (
            "Reference naive baseline model in this run (best naive* model, "
            "or highest-WAPE univariate baseline if no naive model ran)."
        ),
        "baseline_wape": "Holdout-fold pooled_wape of baseline_model.",
        "fva_wape_delta": (
            "Absolute WAPE reduction vs. baseline (baseline_wape - "
            "pooled_wape; positive means this model beat the baseline)."
        ),
        "fva_relative_lift_pct": (
            "Percentage relative error reduction vs. baseline: 100 * "
            "(baseline_wape - pooled_wape) / baseline_wape."
        ),
    },
    "v_horizon_decay": {
        "run_id": "Run identifier.",
        "model_type": "Model, variant, ensemble, or reconciled model identifier.",
        "ensemble_id": "Ensemble configuration hash; NULL for base models.",
        "horizon_step": (
            "1-based forecast lead-time step ahead from fold cutoff_date (1 "
            "= 1 step ahead, H = max horizon)."
        ),
        "n_series": "Distinct series evaluated at this horizon_step.",
        "n_points": "Total OOF observations pooled at this horizon_step across folds.",
        "pooled_wape": "Volume-weighted WAPE at this exact lead-time step ahead.",
        "pooled_mae": "Mean Absolute Error at this exact lead-time step ahead.",
        "signed_bias": (
            "Mean signed error AVG(yhat - y_true) at this step (+ = "
            "over-forecast, - = under-forecast)."
        ),
        "empirical_coverage": (
            "Fraction of actuals inside [yhat_lower, yhat_upper] at this horizon step."
        ),
        "mean_interval_width": (
            "Mean width (yhat_upper - yhat_lower) of the prediction interval at this horizon step."
        ),
    },
    "v_bias_and_exceptions": {
        "run_id": "Run identifier.",
        "ts_id": "Time-series identifier.",
        "model_type": "Model, variant, ensemble, or reconciled model identifier.",
        "ensemble_id": "Ensemble configuration hash; NULL for base models.",
        "sbc_class": (
            "Syntetos-Boylan demand regime from fit_diagnostics (smooth, "
            "erratic, intermittent, lumpy, or cold_start)."
        ),
        "cell_status": "Cell fit status from forecast_metadata ('ok' or 'error').",
        "n_oof_points": "Number of out-of-fold validation points scored for this series and model.",
        "mean_actual": "Mean historical actual y_true across the backtest evaluation window.",
        "mean_forecast": "Mean out-of-fold forecast yhat across the backtest evaluation window.",
        "signed_bias": "Mean signed error AVG(yhat - y_true); positive indicates over-forecasting.",
        "mad": "Mean Absolute Deviation AVG(|y_true - yhat|) in target units.",
        "wape": "Series-level Weighted Absolute Percentage Error.",
        "mase": "Series-level Mean Absolute Scaled Error (< 1.0 beats naive).",
        "staleness_gap": "Accuracy loss when freezing parameters vs. refitting per fold.",
        "tracking_signal": (
            "Trigg's cumulative signed error divided by MAD: SUM(yhat - "
            "y_true) / AVG(|y_true - yhat|). |tracking_signal| > 4.0 flags "
            "systematic bias."
        ),
        "is_exception": (
            "True when the series requires planner review (cell error, "
            "|tracking_signal| > 4.0, MASE > 1.0, or staleness_gap > 0.05)."
        ),
        "exception_reason": (
            "Primary actionable triage code: CELL_FIT_ERROR, "
            "CHRONIC_OVER_FORECAST, CHRONIC_UNDER_FORECAST, "
            "WORSE_THAN_NAIVE, STALE_REFIT_DRIFT, or OK."
        ),
    },
    "v_anomalies": {
        "run_id": "Run identifier.",
        "ts_id": "Time-series identifier.",
        "model_type": "Model, variant, or ensemble used as the expected baseline.",
        "ensemble_id": "Ensemble configuration hash; NULL for base models.",
        "forecast_date": "Historical date where the anomalous actual occurred.",
        "fold_id": "Backtest fold index containing forecast_date.",
        "y_true": "Observed actual value on forecast_date.",
        "yhat": "Model expected out-of-fold point forecast on forecast_date.",
        "yhat_lower": "Lower calibrated prediction bound on forecast_date.",
        "yhat_upper": "Upper calibrated prediction bound on forecast_date.",
        "residual": (
            "Signed surprise (y_true - yhat); positive = unexpected spike, "
            "negative = unexpected drop."
        ),
        "residual_zscore": (
            "Standardized residual z-score relative to this (run_id, ts_id, "
            "model_type) series' OOF residual stddev."
        ),
        "outside_interval": "True when y_true fell outside [yhat_lower, yhat_upper].",
        "anomaly_type": (
            "Direction of anomaly: POSITIVE_SPIKE (y_true > yhat) or NEGATIVE_DIP (y_true < yhat)."
        ),
    },
    "v_feature_attributions": {
        "run_id": "Run identifier.",
        "ts_id": "Time-series identifier.",
        "model_type": "Explainability-capable model or reconciled/variant identifier.",
        "attribution_scope": (
            "Granularity source: 'global_fit' (from "
            "forecast_metadata.fit_diagnostics.feature_attributions) or "
            "'horizon_local' (averaged from forecast_predictions.explanations)."
        ),
        "feature_name": (
            "Covariate, calendar feature, lag, or similarity feature name "
            "(e.g. price_index, promo_flag, temperature, lag_7)."
        ),
        "mean_attribution": (
            "Mean attribution score (signed impact in target units for "
            "horizon_local; non-negative importance weight for global_fit)."
        ),
        "mean_abs_attribution": (
            "Mean absolute attribution magnitude |attribution| for ranking "
            "top drivers regardless of sign."
        ),
        "n_horizon_steps": (
            "Number of forward forecast_date steps aggregated for "
            "horizon_local; NULL for global_fit."
        ),
    },
    "v_executive_sop_summary": {
        "run_id": "Run identifier (1 executive S&OP scorecard row per run).",
        "created_at": "UTC timestamp of the run.",
        "status": "Run lifecycle status.",
        "source_table": (
            "Input source table read by this run (JSON_VALUE(raw_config, '$.data.source_table'))."
        ),
        "n_series": "Total time series forecasted in the run.",
        "n_models": "Number of candidate models evaluated.",
        "runtime_seconds": "End-to-end run wall-clock duration in seconds.",
        "champion_model": (
            "Lowest-pooled_wape winning model or ensemble on the comparable holdout fold."
        ),
        "champion_pooled_wape": "Holdout-fold pooled WAPE of champion_model.",
        "champion_pooled_mae": "Holdout-fold pooled MAE of champion_model.",
        "baseline_wape": "Holdout-fold pooled WAPE of the run's naive baseline model.",
        "fva_relative_lift_pct": (
            "Percentage relative WAPE reduction achieved by champion_model over the naive baseline."
        ),
        "horizon_start_date": "First forward forecast_date generated by champion_model.",
        "horizon_end_date": "Last forward forecast_date generated by champion_model.",
        "total_forecast_units": (
            "Sum of forward yhat across all series and horizon dates for champion_model."
        ),
        "mean_interval_width": (
            "Mean forward uncertainty band width (yhat_upper - yhat_lower) for champion_model."
        ),
        "n_exception_series": (
            "Count of series flagged with is_exception = TRUE for champion_model."
        ),
        "exception_rate": (
            "Fraction of scored series requiring planner exception review for champion_model."
        ),
        "n_gpu_waste_jobs": (
            "Count of GPU family jobs in v_run_jobs where device_verdict != "
            "'ENGAGED_UTILISED' (0 = clean FinOps)."
        ),
    },
}


def sql_escape_description(text: str) -> str:
    """Escape ``text`` for a single-quoted BigQuery ``OPTIONS(description='...')`` literal."""
    cleaned = " ".join(text.split())
    return cleaned.replace("\\", "\\\\").replace("'", "\\'")


# Historical columns retained only on pre-v1.0 BigQuery registries (never emitted in fresh DDL,
# documented by `sync_existing_descriptions` so analysts & DataAgents ignore retired fields).
LEGACY_COLUMN_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "run_registry": {
        "spark_method": (
            "Deprecated pre-v1.0 column (unused on current runs); query run_jobs.runtime or "
            "v_run_jobs.runtime instead."
        ),
    },
}


def object_description(name: str) -> str:
    """Return the canonical top-level description for a table or view ``name``.

    Raises ``KeyError`` if ``name`` has no registered description.
    """
    if name in TABLE_DESCRIPTIONS:
        return TABLE_DESCRIPTIONS[name]
    if name in VIEW_DESCRIPTIONS:
        return VIEW_DESCRIPTIONS[name]
    raise KeyError(f"Missing top-level description in registry/metadata.py for {name!r}")


def column_description(object_name: str, column_name: str) -> str:
    """Return the canonical description for ``(object_name, column_name)``.

    Raises ``KeyError`` if either ``object_name`` or ``column_name`` is missing so DDL/view
    rendering and unit tests catch unannotated columns immediately.
    """
    cols = COLUMN_DESCRIPTIONS.get(object_name)
    if cols is None:
        raise KeyError(
            f"Missing column description map in registry/metadata.py for {object_name!r}"
        )
    if column_name in cols:
        return cols[column_name]
    if object_name == "forecast_metadata" and column_name in METRIC_NAMES:
        return metric_description(column_name)
    raise KeyError(
        f"Missing column description in registry/metadata.py for {object_name}.{column_name}"
    )


def sync_existing_descriptions(
    client: Any,
    dataset_ref: str,
    object_names: Sequence[str],
) -> list[str]:
    """Idempotently backfill or refresh table/view and column descriptions in BigQuery.

    ``CREATE TABLE IF NOT EXISTS`` is a no-op when a table already exists from an earlier release
    (e.g. ``v1.0.0``–``v1.9.1``). This helper inspects each existing table or view in
    ``dataset_ref`` and patches its top-level ``description`` and ``SchemaField.description``
    entries in a single metadata ``update_table`` call whenever any description is missing or out
    of date. Non-existent tables (such as optional covariate source tables not yet seeded) are
    skipped cleanly.

    Returns the list of table/view names whose metadata was updated.
    """
    from google.api_core import exceptions as gcp_exceptions
    from google.cloud import bigquery

    updated: list[str] = []
    for name in object_names:
        if name not in TABLE_DESCRIPTIONS and name not in VIEW_DESCRIPTIONS:
            continue
        fqn = f"{dataset_ref}.{name}"
        try:
            table_obj = client.get_table(fqn)
        except gcp_exceptions.NotFound:
            continue

        desired_desc = " ".join(object_description(name).split())
        col_map = {
            **LEGACY_COLUMN_DESCRIPTIONS.get(name, {}),
            **COLUMN_DESCRIPTIONS.get(name, {}),
        }

        needs_update = (table_obj.description or "") != desired_desc
        new_schema: list[Any] = []
        for field in table_obj.schema:
            target_col_desc = col_map.get(field.name)
            if (
                target_col_desc is None
                and name == "forecast_metadata"
                and field.name in METRIC_NAMES
            ):
                target_col_desc = metric_description(field.name)
            if target_col_desc is not None:
                clean_col_desc = " ".join(target_col_desc.split())
                if (field.description or "") != clean_col_desc:
                    needs_update = True
                field_dict = field.to_api_repr()
                field_dict["description"] = clean_col_desc
                new_schema.append(bigquery.SchemaField.from_api_repr(field_dict))
            else:
                new_schema.append(field)

        if needs_update:
            table_obj.description = desired_desc
            table_obj.schema = new_schema
            try:
                client.update_table(table_obj, ["description", "schema"])
                updated.append(name)
            except gcp_exceptions.GoogleAPICallError as exc:
                logger.warning(
                    "Metadata description patch via update_table skipped on %s (%s)", fqn, exc
                )
    return updated
