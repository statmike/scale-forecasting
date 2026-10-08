# Changelog

All notable changes to `scale-forecasting` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [1.0.0] - 2026-10-08

### Added

- **Declarative Execution Contract (`RunConfig`):** One JSON/dict contract ([`src/scale_forecasting/config.py`](./src/scale_forecasting/config.py)) drives execution across Local CLI, Python SDK (`Forecaster`), Interactive Notebooks, Cloud Composer 3 DAGs (`--emit-airflow`), and Headless Cloud Batch containers, with deterministic content-addressed `run_id` hashing ([`src/scale_forecasting/registry/ids.py`](./src/scale_forecasting/registry/ids.py)).
- **34 Forecasting Models Across 5 Families:**
  - `statistical` (`18`): `theta`, `auto_theta`, `holtwinters`, `autoets`, `auto_arima`, `sarimax`, `tbats`, `auto_ces`, `stl_bagging`, `ucm`, `kalman`, `prophet`, `croston`, `fft`, `naive_mean`, `naive_seasonal`, `naive_drift`, `naive_moving_average`.
  - `ml` (`5`): `lightgbm`, `xgboost`, `catboost`, `random_forest`, `regression_lags`.
  - `deep_learning` (`5`): `tide`, `tft`, `tsmixer`, `patchtst`, `neuralprophet` across `local`, `global`, and `hybrid` panel regimes.
  - `automl` (`4`): `vertex_l2l`, `vertex_tide`, `vertex_tft`, `vertex_seq2seq` via Vertex AI Tabular Workflows and managed AutoML training jobs.
  - `native` (`2`): `arima_plus` and `timesfm` (`AI.FORECAST`) executed natively in BigQuery SQL.
- **21 Evaluation Metrics & Conformal Calibration:** 16 point metrics (`wape`, `smape`, `mape`, `maape`, `ope`, `mae`, `rmse`, `mse`, `rmsle`, `bias`, `mase`, `mase_seasonal`, `rmsse`, `msse`, `r2`, `cv`) and 5 interval metrics (`coverage`, `pinball`, `interval_score`, `interval_width`, `msis`), plus empirical conformal residual calibration ([`src/scale_forecasting/calibration.py`](./src/scale_forecasting/calibration.py)).
- **7 Google Cloud Compute Runtimes & Per-Family DAG Router:** Parallel per-family job dispatch ([`src/scale_forecasting/dag.py`](./src/scale_forecasting/dag.py)) across Dataproc Serverless / Cluster / Connect (`spark`), Ray on Vertex AI or GKE (`ray`), Vertex AI CustomJob (`vertex`), self-deleting Compute Engine Single-VM (`gce`), GKE Indexed Job (`gke`), Vertex AI AutoML (`vertex_automl`), and BigQuery ML (`bigquery`), supporting CPU and NVIDIA `T4`, `L4`, `A100`, and `A100_80GB` GPUs with BigQuery Storage Read API `[start_id, end_id]` worker sharding and LPT chunk scheduling.
- **Hierarchical Reconciliation, Stacked Ensembles & Explainability:** All 7 Hyndman FPP3 reconciliation methods (`mint_shrink`, `wls_var`, `wls_struct`, `ols`, `bottom_up`, `top_down`, `middle_out`), 6 ensemble strategies (`mean`, `median`, `inverse_error`, `nnls`, `ridge`, `xgb`), cross-run ensembling (`ensemble_cross_runs`), and two-tier feature attributions (`feature_attributions()` and `explain()`).
- **BigQuery Registry & Analytical SQL Views:** 4 source tables, 5 registry tables (`run_registry`, `run_jobs`, `forecast_metadata`, `forecast_predictions`, `backtest_oof`), and 5 analytical SQL views (`v_model_leaderboard`, `v_model_leaderboard_comparable`, `v_backtest_coverage`, `v_run_summary` with per-job time ledger, and `v_run_jobs`), plus surgical per-cell repair ([`src/scale_forecasting/retry_run.py`](./src/scale_forecasting/retry_run.py)) and `Registry.doctor()`.
- **Two-Stage Terraform Infrastructure, 11 Notebooks & 42 Live Smoke Configs:** Complete bootstrap + main Terraform modules ([`terraform/`](./terraform/README.md)), 11 interactive notebooks across 4 learning tracks ([`notebooks/`](./notebooks/README.md)), and 42 live-validated smoke configurations recorded in [`docs/validation.md`](./docs/validation.md).
- **Agent-First Surfaces (`SKILL.md`, Built-in `stdio` MCP Server, JSON Schema, `llms.txt` & Cloud Shell Tutorial):** Portable Agent Skill ([`skills/scale-forecasting/SKILL.md`](./skills/scale-forecasting/SKILL.md)), zero-extra JSON-RPC 2.0 MCP server ([`src/scale_forecasting/mcp.py`](./src/scale_forecasting/mcp.py)) exposing 7 `forecast://*` resources and 9 tools with `--allow-launch` safety gating, published `RunConfig` JSON Schema ([`docs/schemas/run_config.schema.json`](./docs/schemas/run_config.schema.json)), `llms.txt` / `llms-full.txt`, [`gemini-extension.json`](./gemini-extension.json), [`cloudshell_tutorial.md`](./cloudshell_tutorial.md), and pre-commit zero-drift synthesis ([`src/scale_forecasting/agent_surfaces.py`](./src/scale_forecasting/agent_surfaces.py)).

### Changed

- **Modular Dependency Extras (`pyproject.toml`):** Core `pip install scale-forecasting` installs only the pure offline forecasting layer with zero Google Cloud clients. All Google Cloud SDKs live in `[gcp]`, with composable `[notebook]`, `[spark]`, `[ray]`, `[submit]`, `[models-stats]`, `[models-trees]`, `[models-prophet]`, `[models-dl]`, `[models-automl]`, `[models]`, and `[all]` extras and actionable `MissingExtraError` guidance at every cloud and plotting entry point.
- **Static Typing, Logging & Test Ratchet:** Added PEP 561 `py.typed` marker with zero `mypy` errors across all 178 source modules, library-safe `NullHandler` logging (`configure_cli_logging` on CLI entry points), widened Ruff lint families (`SIM`, `C4`, `PIE`, `PERF`, `RUF`, `BLE`), and enforced an 85 % offline line-coverage ratchet in CI.

[Unreleased]: https://github.com/statmike/scale-forecasting/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/statmike/scale-forecasting/releases/tag/v1.0.0
