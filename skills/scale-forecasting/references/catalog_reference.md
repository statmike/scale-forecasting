# Canonical Platform Catalogs (Models, Metrics, Runtimes, Extras, Views & Configs)

> **Auto-generated from code registries by `python -m scale_forecasting.agent_surfaces --write`.** Do not edit by hand; pre-commit (`test_agent_surfaces.py`) enforces zero drift.

---

## 1. Forecasting Models (37 Models Across 5 Families)

| Model (`models` key) | Family | Runtime | Upstream Package | Install Extra | Future Cov. | Past Cov. | Static Cov. | Explainability | Global / Hybrid | GPU Capable |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `vertex_l2l` | `automl` | `vertex_automl` | `google-cloud-pipeline-components` | `models-automl` | Yes | Yes | Yes | Yes | `local/global` | Yes |
| `vertex_seq2seq` | `automl` | `vertex_automl` | `google-cloud-pipeline-components` | `models-automl` | Yes | Yes | Yes | Yes | `local/global` | Yes |
| `vertex_tft` | `automl` | `vertex_automl` | `google-cloud-pipeline-components` | `models-automl` | Yes | Yes | Yes | Yes | `local/global` | Yes |
| `vertex_tide` | `automl` | `vertex_automl` | `google-cloud-pipeline-components` | `models-automl` | Yes | Yes | Yes | Yes | `local/global` | Yes |
| `chronos2` | `deep_learning` | `python` | `chronos-forecasting` | `models-foundation` | Yes | Yes | Yes | No | `local/global` | Yes |
| `chronos_bolt` | `deep_learning` | `python` | `chronos-forecasting` | `models-foundation` | No | No | No | No | `local/global` | Yes |
| `neuralprophet` | `deep_learning` | `python` | `neuralprophet` | `models-dl` | No | No | No | No | `local/global/hybrid` | Yes |
| `patchtst` | `deep_learning` | `python` | `neuralforecast` | `models-dl` | No | No | No | No | `local/global` | Yes |
| `tft` | `deep_learning` | `python` | `neuralforecast` | `models-dl` | Yes | Yes | Yes | No | `local/global` | Yes |
| `tide` | `deep_learning` | `python` | `neuralforecast` | `models-dl` | Yes | Yes | Yes | No | `local/global` | Yes |
| `timesfm_torch` | `deep_learning` | `python` | `timesfm` | `models-foundation` | Yes | Yes | Yes | No | `local/global` | Yes |
| `tsmixer` | `deep_learning` | `python` | `neuralforecast` | `models-dl` | Yes | Yes | Yes | No | `local/global` | Yes |
| `catboost` | `ml` | `python` | `catboost` | `models-trees` | Yes | Yes | No | Yes | `local` | No |
| `lightgbm` | `ml` | `python` | `lightgbm` | `models-trees` | Yes | Yes | No | Yes | `local` | No |
| `random_forest` | `ml` | `python` | `scikit-learn` | `core` | Yes | Yes | No | Yes | `local` | No |
| `regression_lags` | `ml` | `python` | `numpy` | `core` | Yes | Yes | No | Yes | `local` | No |
| `xgboost` | `ml` | `python` | `xgboost` | `models-trees` | Yes | Yes | No | Yes | `local` | No |
| `arima_plus` | `native` | `bigquery` | `bigquery-ml` | `gcp` | Yes | No | No | Yes | `local` | No |
| `timesfm` | `native` | `bigquery` | `bigquery-ml` | `gcp` | Yes | Yes | No | No | `local` | No |
| `auto_arima` | `statistical` | `python` | `statsforecast` | `models-stats` | Yes | Yes | No | No | `local` | No |
| `auto_ces` | `statistical` | `python` | `statsforecast` | `models-stats` | No | No | No | No | `local` | No |
| `auto_theta` | `statistical` | `python` | `statsforecast` | `models-stats` | No | No | No | No | `local` | No |
| `autoets` | `statistical` | `python` | `statsmodels` | `core` | No | No | No | No | `local` | No |
| `croston` | `statistical` | `python` | `numpy` | `core` | No | No | No | No | `local` | No |
| `fft` | `statistical` | `python` | `scipy` | `core` | No | No | No | No | `local` | No |
| `holtwinters` | `statistical` | `python` | `statsmodels` | `core` | No | No | No | No | `local` | No |
| `kalman` | `statistical` | `python` | `statsmodels` | `core` | Yes | Yes | No | No | `local` | No |
| `naive_drift` | `statistical` | `python` | `numpy` | `core` | No | No | No | No | `local` | No |
| `naive_mean` | `statistical` | `python` | `numpy` | `core` | No | No | No | No | `local` | No |
| `naive_moving_average` | `statistical` | `python` | `numpy` | `core` | No | No | No | No | `local` | No |
| `naive_seasonal` | `statistical` | `python` | `numpy` | `core` | No | No | No | No | `local` | No |
| `prophet` | `statistical` | `python` | `prophet` | `models-prophet` | Yes | Yes | No | No | `local` | No |
| `sarimax` | `statistical` | `python` | `statsmodels` | `core` | Yes | Yes | No | No | `local` | No |
| `stl_bagging` | `statistical` | `python` | `statsmodels` | `core` | No | No | No | No | `local` | No |
| `tbats` | `statistical` | `python` | `statsforecast` | `models-stats` | No | No | No | No | `local` | No |
| `theta` | `statistical` | `python` | `statsmodels` | `core` | No | No | No | No | `local` | No |
| `ucm` | `statistical` | `python` | `statsmodels` | `core` | Yes | Yes | No | No | `local` | No |

---

## 2. Evaluation Metrics (21 Metrics: 16 Point + 5 Interval)

| Metric (`decision_metric`) | Kind | Direction | Needs Intervals | Needs Train History | Needs Seasonal Period | Optimal Point Arm |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `mae` | `point` | `lower` | No | No | No | `median` |
| `rmse` | `point` | `lower` | No | No | No | `mean` |
| `mse` | `point` | `lower` | No | No | No | `mean` |
| `mape` | `point` | `lower` | No | No | No | `median` |
| `smape` | `point` | `lower` | No | No | No | `median` |
| `wape` | `point` | `lower` | No | No | No | `median` |
| `mase` | `point` | `lower` | No | Yes | No | `median` |
| `rmsse` | `point` | `lower` | No | Yes | No | `mean` |
| `bias` | `point` | `zero` | No | No | No | `mean` |
| `coverage` | `interval` | `higher` | Yes | No | No | `median` |
| `pinball` | `interval` | `lower` | Yes | No | No | `median` |
| `mase_seasonal` | `point` | `lower` | No | Yes | Yes | `median` |
| `maape` | `point` | `lower` | No | No | No | `median` |
| `interval_score` | `interval` | `lower` | Yes | No | No | `median` |
| `interval_width` | `interval` | `lower` | Yes | No | No | `median` |
| `ope` | `point` | `lower` | No | No | No | `median` |
| `rmsle` | `point` | `lower` | No | No | No | `median` |
| `msse` | `point` | `lower` | No | Yes | No | `mean` |
| `msis` | `interval` | `lower` | Yes | Yes | Yes | `median` |
| `r2` | `point` | `higher` | No | No | No | `mean` |
| `cv` | `point` | `lower` | No | No | No | `mean` |

---

## 3. Compute Runtimes (7 Runtimes)

| Runtime | Sub-Modes | Supported Families | GPU Support | Scaling Model | Submitter Module |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `spark` | spark_mode: 'serverless' (default) \| 'cluster' (plus interactive Spark Connect) | `statistical`, `ml`, `deep_learning` | `L4 (serverless/cluster)`, `T4, A100, A100_80GB (cluster only)` | Dynamic executor allocation (max_executors / min_workers / max_workers) | `scale_forecasting.submit` |
| `ray` | ray_mode: 'vertex' (default) \| 'gke' | `statistical`, `ml`, `deep_learning` | `T4`, `L4`, `A100`, `A100_80GB` | Autoscaling CPU + GPU worker pools (ray_autoscale, min_workers, max_workers) | `scale_forecasting.ray_submit` |
| `vertex` | Managed Vertex AI CustomJob worker pool | `statistical`, `ml`, `deep_learning` | `T4`, `L4`, `A100`, `A100_80GB` | Fixed multi-VM worker pool (workers >= 1, auto-expands for multi-model DL) | `scale_forecasting.vertex_submit` |
| `gce` | Single-VM Container-Optimized OS instance | `statistical`, `ml`, `deep_learning` | `T4`, `L4`, `A100`, `A100_80GB` | Strictly single-VM (workers = 1) with triple-redundant self-deletion | `scale_forecasting.gce_submit` |
| `gke` | gke_mode: 'job' (default, K8s Indexed Job) \| 'ray' (KubeRay) | `statistical`, `ml`, `deep_learning` | `T4`, `L4`, `A100`, `A100_80GB` | Multi-pod Indexed Job (workers >= 1) or autoscaling Ray pods on GKE | `scale_forecasting.gke_submit` |
| `vertex_automl` | automl_mode: 'tabular_workflow' (default) \| 'training_job' | `automl` | `T4`, `L4`, `A100`, `A100_80GB` | Managed Vertex AI Pipelines / AutoML worker pools (min_workers, max_workers) | `scale_forecasting.automl_submit` |
| `bigquery` | In-warehouse BigQuery ML SQL | `native` | CPU / SQL only | Managed BigQuery slots (no VM provisioning) | `scale_forecasting.engines.bigquery_engine` |

---

## 4. Dependency Extras (13 Extras in `pyproject.toml`)

| Extra | Install Command | Key Probe Modules | Purpose |
| :--- | :--- | :--- | :--- |
| `core` (base) | `pip install scale-forecasting` | `pydantic`, `pandas`, `statsmodels`, `sklearn`, `optuna` | Pure offline layer: 14 core models, 21 metrics, backtesting, ensembling, reconciliation, playground, dry-run, MCP server |
| `[gcp]` | `pip install "scale-forecasting[gcp]"` | `google.cloud.bigquery`, `google.cloud.bigquery_storage`, `google.cloud.storage`, `google.cloud.dataproc_v1` (+1 more) | Google Cloud clients (BigQuery, Storage Read/Write APIs, GCS, Dataproc, Vertex AI) |
| `[notebook]` | `pip install "scale-forecasting[notebook]"` | `matplotlib` | Interactive notebook kernel and matplotlib plotting helpers |
| `[spark]` | `pip install "scale-forecasting[spark]"` | `google.cloud.bigquery`, `google.cloud.bigquery_storage`, `google.cloud.storage`, `google.cloud.dataproc_v1` (+2 more) | PySpark runtime client plus [gcp] |
| `[ray]` | `pip install "scale-forecasting[ray]"` | `google.cloud.bigquery`, `google.cloud.bigquery_storage`, `google.cloud.storage`, `google.cloud.dataproc_v1` (+2 more) | Ray cluster/job submission client plus [gcp] |
| `[submit]` | `pip install "scale-forecasting[submit]"` | `google.cloud.bigquery`, `google.cloud.bigquery_storage`, `google.cloud.storage`, `google.cloud.dataproc_v1` (+2 more) | Alias for [ray] (thin launch-host client for all cloud runtimes) |
| `[models-stats]` | `pip install "scale-forecasting[models-stats]"` | `statsforecast` | Nixtla statsforecast and pmdarima statistical models (5 models) |
| `[models-trees]` | `pip install "scale-forecasting[models-trees]"` | `xgboost`, `lightgbm`, `catboost` | Gradient-boosted tree models: XGBoost, LightGBM, CatBoost (3 models) |
| `[models-prophet]` | `pip install "scale-forecasting[models-prophet]"` | `prophet` | Prophet additive/multiplicative decomposable model (1 model) |
| `[models-dl]` | `pip install "scale-forecasting[models-dl]"` | `torch`, `neuralprophet`, `neuralforecast` | PyTorch, NeuralProphet, and NeuralForecast deep-learning models (5 models) |
| `[models-foundation]` | `pip install "scale-forecasting[models-foundation]"` | `torch`, `chronos`, `timesfm` | Open-weight zero-shot foundation models: Chronos-Bolt, Chronos-2, TimesFM (3 models) |
| `[models-automl]` | `pip install "scale-forecasting[models-automl]"` | `google.cloud.bigquery`, `google.cloud.bigquery_storage`, `google.cloud.storage`, `google.cloud.dataproc_v1` (+2 more) | Vertex AI Pipelines / KFP components for Tabular Workflows (4 models) |
| `[models]` | `pip install "scale-forecasting[models]"` | `statsforecast`, `xgboost`, `lightgbm`, `catboost` (+13 more) | All 6 model family extras combined |
| `[all]` | `pip install "scale-forecasting[all]"` | `google.cloud.bigquery`, `google.cloud.bigquery_storage`, `google.cloud.storage`, `google.cloud.dataproc_v1` (+15 more) | Complete installation (all cloud clients, runtimes, notebooks, and model families) |

---

## 5. BigQuery Source Tables, Registry Tables & Analytical SQL Views

| Surface | Canonical Names | Description |
| :--- | :--- | :--- |
| **Source Tables (4)** | `source_series_iceberg`, `source_series_native`, `source_series_wide_native`, `source_series_hierarchical_native` | Input time-series panels (Apache Iceberg on GCS via BigLake + native BigQuery tables). |
| **Registry Tables (5)** | `run_registry`, `run_jobs`, `forecast_metadata`, `forecast_predictions`, `backtest_oof` | Append-only experiment header, per-family job ledger, cell metadata/metrics, forward predictions, and out-of-fold backtests. |
| **Analytical SQL Views (5)** | `v_run_summary`, `v_run_jobs`, `v_model_leaderboard`, `v_backtest_coverage`, `v_model_leaderboard_comparable` | Deduplicated leaderboards, comparable cohort rankings, backtest fold coverage, run time ledger (`overhead_seconds >= 0`), and current job state. |

---

## 6. Shipped Configuration Catalog (`configs/` & `configs/smokes/`)

### Root Demonstration & Scale Configs (`configs/*.json`)

| Config File | `run_name` | Default Runtime | Models | Backtest | Ensemble |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `configs/all_families_10k.json` | `all_families_10k` | `ray` | `theta`, `holtwinters`, `sarimax`, `xgboost` (+3) | off | off |
| `configs/all_families_10k_full.json` | `all_families_10k_full` | `ray` | `theta`, `holtwinters`, `sarimax`, `xgboost` (+3) | `expanding` (2f) | off |
| `configs/bq_native_demo.json` | `bq_native_demo` | `spark` | `arima_plus`, `timesfm` | off | off |
| `configs/ensemble_demo.json` | `ensemble_demo` | `spark` | `theta`, `arima_plus`, `timesfm` | `expanding` (2f) | `mean`, `median`, `inverse_error` |
| `configs/explode_100k.json` | `explode_100k` | `spark` | `theta`, `holtwinters`, `sarimax`, `xgboost` | off | off |
| `configs/explode_demo.json` | `explode_demo` | `spark` | `theta`, `holtwinters`, `sarimax`, `xgboost` | off | off |
| `configs/gke_demo.json` | `gke_demo` | `gke` | `theta`, `holtwinters`, `arima_plus`, `timesfm` | `expanding` (2f) | off |
| `configs/mixed_demo.json` | `mixed_demo` | `spark` | `theta`, `arima_plus`, `timesfm` | `expanding` (2f) | off |
| `configs/neuralprophet_ab_cluster_cpu.json` | `neuralprophet_ab_cluster_cpu` | `spark` | `neuralprophet` | `expanding` (2f) | off |
| `configs/neuralprophet_ab_cluster_gpu.json` | `neuralprophet_ab_cluster_gpu` | `spark` | `neuralprophet` | `expanding` (2f) | off |
| `configs/neuralprophet_ab_cpu.json` | `neuralprophet_ab_cpu` | `ray` | `neuralprophet` | `expanding` (2f) | off |
| `configs/neuralprophet_ab_gpu.json` | `neuralprophet_ab_gpu` | `ray` | `neuralprophet` | `expanding` (2f) | off |
| `configs/per_family_runtimes_cpu_demo.json` | `per_family_runtimes_cpu_demo` | `spark` | `theta`, `holtwinters`, `xgboost`, `neuralprophet` (+1) | off | off |
| `configs/per_family_runtimes_demo.json` | `per_family_runtimes_demo` | `spark` | `theta`, `holtwinters`, `xgboost`, `neuralprophet` (+1) | off | off |
| `configs/ray_100k.json` | `ray_100k` | `ray` | `theta`, `holtwinters`, `sarimax`, `xgboost` | off | off |
| `configs/ray_autoscale_demo.json` | `ray_autoscale_demo` | `ray` | `theta`, `holtwinters`, `sarimax` | off | off |
| `configs/ray_cpu_demo.json` | `ray_cpu_demo` | `ray` | `theta`, `holtwinters`, `arima_plus`, `timesfm` | `expanding` (2f) | off |
| `configs/ray_gpu_demo.json` | `ray_gpu_demo` | `ray` | `neuralprophet`, `theta`, `arima_plus`, `timesfm` | `expanding` (2f) | off |
| `configs/repair_demo.json` | `repair_demo` | `spark` | `theta`, `holtwinters`, `xgboost` | off | off |
| `configs/repair_retry_demo.json` | `repair_retry_demo` | `spark` | `theta`, `holtwinters`, `xgboost` | off | off |

### Live Smoke Test Configs (`configs/smokes/*.json`)

| Smoke Config | `run_name` | Default Runtime | Family Overrides | Models |
| :--- | :--- | :--- | :--- | :--- |
| `configs/smokes/01_serverless_cpu.json` | `smoke_01_serverless_cpu` | `spark` | — | `theta`, `holtwinters`, `xgboost` |
| `configs/smokes/02_bq_native.json` | `smoke_02_bq_native` | `spark` | — | `arima_plus`, `timesfm` |
| `configs/smokes/03_serverless_gpu.json` | `smoke_03_serverless_gpu` | `spark` | `deep_learning:spark` | `neuralprophet` |
| `configs/smokes/04_cluster_cpu.json` | `smoke_04_cluster_cpu` | `spark` | `ml:spark`, `statistical:spark` | `theta`, `holtwinters`, `xgboost` |
| `configs/smokes/05_cluster_reuse.json` | `smoke_05_cluster_reuse` | `spark` | `ml:spark`, `statistical:spark` | `theta`, `xgboost` |
| `configs/smokes/06_cluster_gpu.json` | `smoke_06_cluster_gpu` | `spark` | `deep_learning:spark` | `neuralprophet` |
| `configs/smokes/07_ray_cpu.json` | `smoke_07_ray_cpu` | `ray` | — | `theta`, `holtwinters`, `xgboost` |
| `configs/smokes/08_ray_gpu.json` | `smoke_08_ray_gpu` | `ray` | `deep_learning:ray` | `neuralprophet` |
| `configs/smokes/09_shared_ray.json` | `smoke_09_shared_ray` | `ray` | `deep_learning:ray` | `theta`, `xgboost`, `neuralprophet` |
| `configs/smokes/10_mixed_runtimes.json` | `smoke_10_mixed_runtimes` | `spark` | `deep_learning:ray` | `theta`, `xgboost`, `neuralprophet`, `arima_plus` (+1) |
| `configs/smokes/11_ensemble_barrier.json` | `smoke_11_ensemble_barrier` | `spark` | — | `theta`, `holtwinters`, `arima_plus`, `timesfm` |
| `configs/smokes/12_ensemble_microbatch.json` | `smoke_12_ensemble_microbatch` | `spark` | — | `theta`, `holtwinters`, `arima_plus`, `timesfm` |
| `configs/smokes/13_native_format.json` | `smoke_13_native_format` | `spark` | — | `theta`, `xgboost`, `arima_plus`, `timesfm` |
| `configs/smokes/14_full_dag.json` | `smoke_14_full_dag` | `spark` | `deep_learning:spark` | `theta`, `holtwinters`, `xgboost`, `neuralprophet` (+2) |
| `configs/smokes/15_airflow_multi_engine.json` | `smoke_15_airflow_multi_engine` | `spark` | `deep_learning:ray` | `theta`, `xgboost`, `neuralprophet`, `arima_plus` (+1) |
| `configs/smokes/16_cluster_split_hardware.json` | `smoke_16_cluster_split_hardware` | `spark` | `deep_learning:spark`, `statistical:spark` | `theta`, `holtwinters`, `neuralprophet` |
| `configs/smokes/17_gpu_absent_serverless.json` | `smoke_17_gpu_absent_serverless` | `spark` | `deep_learning:spark` | `neuralprophet` |
| `configs/smokes/18_gpu_absent_cluster.json` | `smoke_18_gpu_absent_cluster` | `spark` | `deep_learning:spark` | `neuralprophet` |
| `configs/smokes/19_gpu_absent_ray.json` | `smoke_19_gpu_absent_ray` | `ray` | `deep_learning:ray` | `neuralprophet` |
| `configs/smokes/20_gpu_intent_cpu_family.json` | `smoke_20_gpu_intent_cpu_family` | `ray` | `deep_learning:ray` | `neuralprophet` |
| `configs/smokes/21_full_catalogue.json` | `smoke_21_full_catalogue` | `spark` | — | `autoets`, `croston`, `holtwinters`, `naive_drift` (+14) |
| `configs/smokes/22_backtest_sliding_overlap.json` | `smoke_22_backtest_sliding_overlap` | `spark` | — | `theta`, `holtwinters`, `naive_seasonal` |
| `configs/smokes/23_backtest_frozen_shrink.json` | `smoke_23_backtest_frozen_shrink` | `spark` | — | `theta`, `holtwinters`, `naive_seasonal` |
| `configs/smokes/24_backtest_stale.json` | `smoke_24_backtest_stale` | `spark` | — | `theta`, `holtwinters`, `naive_seasonal` |
| `configs/smokes/25_backtest_skip.json` | `smoke_25_backtest_skip` | `spark` | — | `theta`, `holtwinters`, `naive_seasonal` |
| `configs/smokes/26_hpo_fleetwide.json` | `smoke_26_hpo_fleetwide` | `spark` | — | `theta`, `holtwinters`, `naive_moving_average` |
| `configs/smokes/27_hpo_per_series.json` | `smoke_27_hpo_per_series` | `spark` | — | `theta`, `holtwinters`, `naive_moving_average` |
| `configs/smokes/28_features_off.json` | `smoke_28_features_off` | `spark` | — | `prophet`, `sarimax`, `ucm`, `regression_lags` (+2) |
| `configs/smokes/29_features_on.json` | `smoke_29_features_on` | `spark` | — | `prophet`, `sarimax`, `ucm`, `regression_lags` (+2) |
| `configs/smokes/30_features_boxcox.json` | `smoke_30_features_boxcox` | `spark` | — | `prophet`, `sarimax`, `ucm`, `regression_lags` (+2) |
| `configs/smokes/31_features_exog.json` | `smoke_31_features_exog` | `spark` | — | `prophet`, `sarimax`, `ucm`, `regression_lags` (+2) |
| `configs/smokes/32_covariates_three_tier.json` | `smoke_32_covariates_three_tier` | `spark` | — | `prophet`, `sarimax`, `ucm`, `regression_lags` (+4) |
| `configs/smokes/33_expanded_stats_ml.json` | `smoke_33_expanded_stats_ml` | `spark` | — | `auto_arima`, `auto_ces`, `auto_theta`, `tbats` (+4) |
| `configs/smokes/34_global_hybrid_dl.json` | `smoke_34_global_hybrid_dl` | `ray` | `deep_learning:ray` | `tide`, `tft`, `tsmixer`, `patchtst` (+1) |
| `configs/smokes/35_hierarchy_reconciliation.json` | `smoke_35_hierarchy_reconciliation` | `spark` | — | `theta`, `holtwinters`, `regression_lags`, `lightgbm` |
| `configs/smokes/36_covariate_fallback_multi_runtime.json` | `smoke_36_covariate_fallback_multi_runtime` | `spark` | `deep_learning:ray` | `theta`, `sarimax`, `xgboost`, `catboost` (+4) |
| `configs/smokes/37_hierarchy_covariates_ensemble.json` | `smoke_37_hierarchy_covariates_ensemble` | `spark` | — | `theta`, `sarimax`, `xgboost`, `lightgbm` |
| `configs/smokes/38_vertex_custom_job.json` | `smoke_38_vertex_custom_job` | `vertex` | `deep_learning:vertex` | `theta`, `sarimax`, `xgboost`, `lightgbm` (+3) |
| `configs/smokes/39_gce_single_vm.json` | `smoke_39_gce_single_vm` | `gce` | `deep_learning:gce` | `theta`, `sarimax`, `xgboost`, `tide` |
| `configs/smokes/40_gke_indexed_job.json` | `smoke_40_gke_indexed_job` | `gke` | `deep_learning:gke`, `statistical:gke` | `theta`, `sarimax`, `xgboost`, `tide` |
| `configs/smokes/41_gke_ray.json` | `smoke_41_gke_ray` | `gke` | `deep_learning:ray`, `ml:gke` | `theta`, `sarimax`, `xgboost`, `tide` |
| `configs/smokes/42_vertex_automl_tabular_workflow.json` | `smoke_42_vertex_automl_tabular_workflow` | `vertex_automl` | `automl:vertex_automl`, `ml:vertex` | `xgboost`, `vertex_tide`, `arima_plus` |

---

## 7. Auto-Forecaster Exposure Tiers, Scale Regimes & Synthetic DGP Benchmarks

### Observational Exposure Tiers (`EXPOSURE_TIERS` in `auto_policy.py`)

| Tier (`tier`) | Title & Typical Duration | Description |
| :--- | :--- | :--- |
| `first_light` | First Light (~20s local / ~55s BQ / ~6m cloud) | Rapid baseline exposure: seasonal naive + fast statistical + gradient boosted trees with a 1-fold backtest to verify data plumbing and establish the FVA baseline. |
| `wide_field` | Wide Field (~2m local / ~2m BQ / ~7-16m cloud, Default) | Balanced multi-family survey across Statistical, ML Trees, and Zero-Shot Foundation models with 2-fold CV, covariate FVA baselines, and champion selection. |
| `deep_sky` | Deep Sky (~6m local / ~4.5m BQ / ~12-22m cloud) | Deep cross-series global + local survey adding CatBoost, TiDE/TSMixer neural architectures, 3-fold CV, fleetwide HPO, and multi-branch Tournament DAG autofocus. |
| `deep_field` | Deep Field (~15m local / ~8m BQ / ~20-35m cloud) | Exhaustive observatory exposure across all eligible families (Statistical, ML, Deep Learning, Foundation, and BigQuery ML SQL), 3-fold CV, Optuna HPO, and learned stacking ensembles. |

### Series-Count Scale Regimes (`SCALE_REGIMES` in `auto_policy.py`)

| Scale Regime (`scale_regime`) | Series Count | Routing Strategy |
| :--- | :--- | :--- |
| `micro` | Micro Panel (1–14 series) | Per-series local statistical, tree, and zero-shot foundation models lead; trained @global neural models deferred unless cross-series correlation is high. |
| `small` | Small Panel (15–199 series) | Hybrid local + @global crossover: runs per-series and pooled models side-by-side to measure empirical pooling FVA. |
| `medium` | Medium Panel (200–2,499 series) | Global trees, neural nets, and foundation models lead; prunes slow O(T^2) per-series iterative statistical models on local CPU. |
| `large` | Large Fleet (2,500–24,999 series) | Global neural/foundation + fast linear-time statistical models with multi-worker cloud sharding and top-K ensemble parsimony. |
| `hyperscale` | Hyperscale Fleet (25,000+ series) | Hyperscale distributed portfolio (Spark Serverless / Ray / Vertex GPU / BigQuery) with 1–2 fold CV and top-3 parsimony ensembling. |

### Ground-Truth Synthetic DGP Benchmarks (20 Panels in `benchmarks.py`)

| Benchmark Name (`benchmark`) | Regime | Freq / H | Ground-Truth Description | Expected `autofocus` Actions |
| :--- | :--- | :---: | :--- | :--- |
| `dgp_01_smooth_seasonal` | `smooth` | `D` / `14` | 12 daily series with smooth linear trend, two weekly harmonics, low Gaussian noise (CV ~ 0.10), and strictly positive demand. | `floor=0.0`, `seasonal_harmonics` |
| `dgp_02_multiplicative_log` | `smooth` | `D` / `14` | 12 daily series with compounding exponential growth where seasonal amplitude and shock variance scale strictly with the series level (sigma proportional to mu). | `inject_transform_log1p`, `champion_selection` |
| `dgp_03_intermittent_lumpy` | `intermittent` | `D` / `14` | 12 daily integer count series mixing 60% intermittent/lumpy SKUs (ADI > 1.6, CV^2 > 0.6) with 40% smooth/erratic SKUs. | `regime_routing`, `croston_sba_tsb`, `cumulative_round` |
| `dgp_04_true_vs_decoy_covs` | `smooth` | `D` / `14` | 12 daily retail series driven by true future covariates ('promo_depth', 'price_index') alongside 3 uncorrelated decoy noise columns ('noise_a', 'noise_b', 'noise_c') that degrade unregularized models. | `prune_decoy_covariates`, `retain_true_drivers` |
| `dgp_05_horizon_crossover` | `smooth` | `D` / `14` | 12 daily series with persistent AR(1) momentum dominating steps 1..5 and deterministic calendar/promo structure dominating steps 6..14. | `inject_direct_horizon_models`, `horizon_specialist_ensemble` |
| `dgp_06_structural_break_spikes` | `erratic` | `D` / `14` | 12 daily series experiencing a permanent +65% step-change level shift at 65% of history plus 3 isolated one-day anomalous spikes. | `enable_level_shift`, `enable_stl_resid_outlier_scrubbing` |
| `dgp_07_censored_stockouts` | `smooth` | `D` / `14` | 12 daily series where true latent demand is clipped at a warehouse capacity ceiling on dates flagged by 'stockout_flag == 1'. | `enable_stockout_imputation` |
| `dgp_08_hierarchical_nonneg` | `smooth` | `D` / `14` | 12 daily series across 3 regions x 4 stores where coherent cross-level aggregation and non-negativity require hierarchical reconciliation. | `enable_hierarchy_reconciliation`, `mint_nonneg` |
| `dgp_09_cross_series_global` | `smooth` | `D` / `14` | 16 moderate-length daily series sharing a common nonlinear price/promo elasticity surface where global cross-series models outperform per-series local fits. | `prefer_global_pooling`, `include_static_covariates` |
| `dgp_10_newsvendor_service` | `smooth` | `D` / `14` | 12 daily SKU demand series with 7-day supplier lead time and 9:1 underage/overage cost ratio (critical fractile q* = 0.90). | `quantile_grid_service_level`, `conformal_calibration` |
| `dgp_11_cold_start_mixed` | `cold_start` | `D` / `14` | 12 mature daily series (T=140) alongside 4 newly launched cold-start series (T=21) requiring short-series fallback and zero-shot/simple baseline routing. | `short_series_fallback`, `regime_routing` |
| `dgp_12_already_optimal_white_noise` | `smooth` | `D` / `14` | 12 daily series consisting of a clean stationary seasonal wave plus irreducible i.i.d. Gaussian noise. Pass 1 captures 100% of the predictable signal; autofocus must recognize low continuation confidence and stop after Pass 1. | `stop_after_pass_1` |
| `dgp_13_all_decoy_univariate_wins` | `smooth` | `D` / `14` | 12 daily series driven strictly by trend and weekly harmonics, accompanied by 3 pure-noise decoy covariates ('noise_a', 'noise_b', 'noise_c'). Autofocus must detect zero covariate value, fall back to plain univariate forecasting, and highlight beaten multivariate challengers. | `fallback_to_plain_univariate`, `champion_is_univariate` |
| `dgp_14_large_hierarchy_intermittent_leaves` | `intermittent` | `D` / `14` | 16 daily count series across 4 regions x 4 stores where bottom-level leaves exhibit sparse intermittent demand while regional and total rollups are smooth. | `enable_hierarchy_reconciliation`, `top_down`, `inject_intermittent_specialists` |
| `dgp_15_scale_crossover_panel` | `smooth` | `D` / `14` | 24 daily series sharing synchronized seasonal harmonics and macro trend where scale-aware policy activates hybrid local + @global cross-series pooling. | `scale_regime_small`, `global_cross_series_pooling` |
| `dgp_16_monthly_trading_days` | `smooth` | `MS` / `6` | 12 monthly series (freq='MS', T=48, H=6) where monthly billing totals fluctuate with the number of weekday business days vs. weekend days in each calendar month. | `enable_trading_days` |
| `dgp_17_missing_gaps_and_leading_zeros` | `smooth` | `D` / `14` | 12 daily series where half of the SKUs have 28 pre-launch leading zeros and internal missing date gaps requiring automatic grid_policy and trim_leading_zeros. | `enable_trim_leading_zeros`, `regularize_grid_policy` |
| `dgp_18_promo_lag_pullforward` | `smooth` | `D` / `14` | 12 daily series where promotions trigger a +85 unit contemporaneous spike on day t followed by a -52 unit pantry-loading dip on day t+1, requiring exog_lags synthesis. | `synthesize_exog_lags` |
| `dgp_19_hourly_diurnal_weekly_load` | `smooth` | `h` / `24` | 12 hourly series (freq='h', T=336, H=24) exhibiting dual 24-hour diurnal double-peak and 168-hour weekend load cycles plus ambient temperature cooling load. | `fourier_multi_seasonal`, `frequency_matched_lags` |
| `dgp_20_monthly_regime_drift_damped` | `smooth` | `MS` / `12` | 12 monthly series (freq='MS', T=60, H=12) with 12-month annual harmonics, a mid-history structural slope shift, and damped long-horizon trend requiring frequency-matched monthly lags. | `enable_level_shift`, `inject_direct_horizon_models` |

### Public Open-Data Benchmarks (9 Real-World Datasets in `benchmarks.py` via `fetch_public_benchmark`)

| Benchmark Name (`name`) | Domain | Freq / H | License | Official Source URL |
| :--- | :--- | :---: | :--- | :--- |
| `m5_walmart` | `Retail / Hierarchical Intermittent & Smooth SKU Demand` | `D` / `28` | M5 Competition Academic & Research Terms (MOFC / Kaggle) | https://github.com/Mcompetitions/M5-methods |
| `favorita_grocery` | `Grocery Retail / Promotional & Macroeconomic Covariates` | `D` / `16` | Competition Research Terms (Corporación Favorita / Kaggle) | https://www.kaggle.com/competitions/store-sales-time-series-forecasting |
| `m4_daily` | `Multi-Domain Daily (M4 Competition)` | `D` / `14` | CC-BY-4.0 (Mcompetitions) | https://github.com/Mcompetitions/M4-methods |
| `m4_weekly` | `Multi-Domain Weekly (M4 Competition)` | `W` / `13` | CC-BY-4.0 (Mcompetitions) | https://github.com/Mcompetitions/M4-methods |
| `m4_monthly` | `Multi-Domain Monthly (M4 Competition)` | `MS` / `18` | CC-BY-4.0 (Mcompetitions) | https://github.com/Mcompetitions/M4-methods |
| `monash_tourism_monthly` | `Tourism / Strong Multiplicative Seasonality` | `MS` / `24` | CC-BY-4.0 (Zenodo / Monash Time Series Forecasting Archive) | https://zenodo.org/records/4656096 |
| `monash_hospital` | `Healthcare / Monthly Patient Count Series` | `MS` / `12` | CC-BY-4.0 (Zenodo / Monash Time Series Forecasting Archive) | https://zenodo.org/records/4656014 |
| `monash_car_parts` | `Automotive / Sparse Intermittent Spare Parts` | `MS` / `12` | CC-BY-4.0 (Zenodo / Monash Time Series Forecasting Archive) | https://zenodo.org/records/4656021 |
| `etth1` | `Energy / Hourly Transformer Load & Oil Temperature` | `h` / `24` | Apache-2.0 (zhouhaoyi/ETDataset) | https://github.com/zhouhaoyi/ETDataset |
