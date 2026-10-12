# Changelog

All notable changes to `scale-forecasting` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [1.1.0] - 2026-10-12

### Added

- **BigQuery ML `ARIMA_PLUS_XREG` Exogenous Covariate & `ML.EXPLAIN_FORECAST` Explainability Parity (`native` family):**
  - `arima_plus` ([`src/scale_forecasting/models/bigquery_native.py`](./src/scale_forecasting/models/bigquery_native.py)) now declares `supports_exog = True`, `supports_future_covariates = True`, and `supports_explainability = True`, bringing `future_covariates`-capable models to 18 and explainability-capable models to 10.
  - When `features.future_covariates` (`known_future_covariates` / `exog`) is configured on `arima_plus`, [`src/scale_forecasting/engines/bigquery_sql.py`](./src/scale_forecasting/engines/bigquery_sql.py) automatically compiles `CREATE OR REPLACE MODEL ... OPTIONS(model_type = 'ARIMA_PLUS_XREG', ...)` with the exogenous regressors in `SELECT` and passes the future regressor table as the 3rd argument to `ML.FORECAST` and `ML.EXPLAIN_FORECAST`. Univariate configs and `arima_plus@univariate` ablation variants continue to compile byte-identical `model_type = 'ARIMA_PLUS'` SQL, enabling direct Covariate FVA (`arima_plus` vs. `arima_plus@univariate`) in `compute_fva`.
  - [`src/scale_forecasting/engines/bigquery_engine.py`](./src/scale_forecasting/engines/bigquery_engine.py) executes `ML.EXPLAIN_FORECAST` (`build_explain_forecast_query` + `extract_bqml_explanations`) to populate **Tier 1** global feature attributions (`forecast_metadata.fit_diagnostics["feature_attributions"]` across `trend`, `seasonal`, `holiday_effect`, and each `attribution_<col>` exogenous regressor) and **Tier 2** per-step forecast attributions (`forecast_predictions.explanations`).
- **Semantic Versioning Charter (`MAJOR.MINOR.PATCH`):** Documented the formal versioning policy in [`AGENTS.md`](./AGENTS.md) and [`CONTRIBUTING.md`](./CONTRIBUTING.md) defining `PATCH` (`1.0.x`), `MINOR` (`1.x.0`), and `MAJOR` (`x.0.0`) boundaries and release precheck gates.

### Changed

- **PyPI Long-Description Mermaid Rendering (`hatch-fancy-pypi-readme`):** Configured `[tool.hatch.metadata.hooks.fancy-pypi-readme]` in [`pyproject.toml`](./pyproject.toml) so `README.md` retains its interactive ` ```mermaid ` architecture flowchart on GitHub and MkDocs while substituting the diagram with the high-resolution architecture card (`docs/assets/social_card.png`) in built wheel and sdist `METADATA` for `pypi.org`.

---

## [1.0.0] - 2026-10-12

### Added

- **Declarative Execution Contract (`RunConfig`):** One JSON/dict contract ([`src/scale_forecasting/config.py`](./src/scale_forecasting/config.py)) drives execution across Local CLI, Python SDK (`Forecaster`), Interactive Notebooks, Cloud Composer 3 DAGs (`--emit-airflow`), and Headless Cloud Batch containers, with deterministic content-addressed `run_id` and per-model `model_spec_hash` hashing ([`src/scale_forecasting/registry/ids.py`](./src/scale_forecasting/registry/ids.py)).
- **34 Forecasting Models Across 5 Families + Multi-Instance `<base_model>@<tag>` Variants:**
  - `statistical` (`18`): `theta`, `auto_theta`, `holtwinters`, `autoets`, `auto_arima`, `sarimax`, `tbats`, `auto_ces`, `stl_bagging`, `ucm`, `kalman`, `prophet`, `croston` (`classic`, `sba`, `tsb`), `fft`, `naive_mean`, `naive_seasonal`, `naive_drift`, `naive_moving_average`.
  - `ml` (`5`): `lightgbm`, `xgboost`, `catboost`, `random_forest`, `regression_lags` with both `"recursive"` and `"direct"` multi-step horizon strategies and customizable target `lags`.
  - `deep_learning` (`5`): `tide`, `tft`, `tsmixer`, `patchtst`, `neuralprophet` across `local`, `global`, and `hybrid` panel regimes.
  - `automl` (`4`): `vertex_l2l`, `vertex_tide`, `vertex_tft`, `vertex_seq2seq` via Vertex AI Tabular Workflows and managed AutoML training jobs.
  - `native` (`2`): `arima_plus` and `timesfm` (`AI.FORECAST`) executed natively in BigQuery SQL with SQL-enforced `output` bounds and integer rounding.
- **21 Evaluation Metrics & Conformal Calibration:** 16 point metrics (`wape`, `smape`, `mape`, `maape`, `ope`, `mae`, `rmse`, `mse`, `rmsle`, `bias`, `mase`, `mase_seasonal`, `rmsse`, `msse`, `r2`, `cv`) and 5 interval metrics (`coverage`, `pinball`, `interval_score`, `interval_width`, `msis`), plus empirical conformal residual calibration ([`src/scale_forecasting/calibration.py`](./src/scale_forecasting/calibration.py)).
- **7 Google Cloud Compute Runtimes & Per-Family DAG Router:** Parallel per-family job dispatch ([`src/scale_forecasting/dag.py`](./src/scale_forecasting/dag.py)) across Dataproc Serverless / Cluster / Connect (`spark`), Ray on Vertex AI or GKE (`ray`), Vertex AI CustomJob (`vertex`), self-deleting Compute Engine Single-VM (`gce`), GKE Indexed Job (`gke`), Vertex AI AutoML (`vertex_automl`), and BigQuery ML (`bigquery`), supporting CPU and NVIDIA `T4`, `L4`, `A100`, and `A100_80GB` GPUs with BigQuery Storage Read API `[start_id, end_id]` worker sharding and LPT chunk scheduling.
- **Hierarchical Reconciliation, Smart Ensembles & Two-Tier Explainability:** All 8 Hyndman FPP3 reconciliation methods (`mint_shrink`, `mint_nonneg` active-set non-negative QP, `wls_var`, `wls_struct`, `ols`, `bottom_up`, `top_down`, `middle_out`), 9 ensemble strategies (`mean`, `median`, `trimmed_mean`, `inverse_error`, `champion`, `horizon_specialist`, `nnls`, `ridge`, `xgb`) with `top_k`, `parsimony_margin`, and `variant_policy` (`"all"`, `"best_per_base_model"`, `"primary_only"`), calibrated ensemble prediction intervals, cross-run ensembling (`ensemble_cross_runs`), and two-tier feature attributions (`feature_attributions()` and `explain()`).
- **Forecaster Decision Workbench (`Phases G.1–G.5`):**
  - **Physical Bounds & Quantile/Newsvendor Targets (`output`):** Scalar and per-series bounds (`floor`, `cap`, `floor_col`, `cap_col`), integer rounding (`"none"`, `"round"`, `"ceil"`, `"floor"`, `"cumulative_round"`), monotone multi-percentile fans (`output.quantiles`), and Newsvendor asymmetric cost optimization (`underage_cost`, `overage_cost`, `target_quantile`).
  - **Pre-Fit Data Hygiene & SBC Regime Routing (`data`, `features`):** Grid regularization (`grid_policy`), `trim_leading_zeros`, X-13 `trading_days`, fold-safe outlier despiking (`outlier_policy` / `outlier_handling`), censored stockout recovery (`censored_col`), and Syntetos–Boylan–Croston demand classification (`compute_demand_profile`) with optional `regime_routing`.
  - **Econometric Diagnostics, Scenarios, Lifecycle Governance & HTML Reports (`diagnostics.py`, `scenarios.py`, `report.py`):** Multi-baseline Forecast Value Added (`compute_fva` with Total, Covariate, Drop-One, and Pooling FVA + Harvey–Leybourne–Newbold modified Diebold–Mariano & Wilcoxon $p$-values), horizon decay (`compute_horizon_decay`), directional bias & Trigg Tracking Signal audits (`compute_bias_report`), AR(1) autocorrelation-aware temporal rollups (`rollup_time`), date-scoped promo/price scenario simulation with revenue impact (`simulate_scenarios`), realized accuracy vs. new actuals (`compute_realized_accuracy`), cross-run revision stability (`compute_revision_stability`), Top-$N$ exception triage queues (`compute_exceptions_queue`), and single-file interactive HTML/JSON executive reports (`export_report`).
  - **Ablation Ladders & Snapshot-Verified Cross-Run Cell Borrowing (`reuse_prior_runs`):** Declarative `ablation_modes` (`cov_ladder`, `cov_by_tier`, `cov_by_column`, `cov_drop_one`, `scope_ladder`, `transform_ladder`) and server-side BigQuery cell borrowing (`ops.find_reusable_models`, `ops.borrow_model_results`) that skips cloud compute for invariant models.
- **BigQuery Registry & Analytical SQL Views:** 4 source tables, 5 registry tables (`run_registry`, `run_jobs`, `forecast_metadata`, `forecast_predictions`, `backtest_oof`), and 5 analytical SQL views (`v_model_leaderboard`, `v_model_leaderboard_comparable`, `v_backtest_coverage`, `v_run_summary` with per-job time ledger, and `v_run_jobs`), plus surgical per-cell repair ([`src/scale_forecasting/retry_run.py`](./src/scale_forecasting/retry_run.py)) and `Registry.doctor()`.
- **Two-Stage Terraform Infrastructure, 11 Notebooks & 42 Live Smoke Configs:** Complete bootstrap + main Terraform modules ([`terraform/`](./terraform/README.md)), 11 interactive notebooks across 4 learning tracks ([`notebooks/`](./notebooks/README.md)), and 42 live-validated smoke configurations + 4 full-breadth multi-runtime verification runs recorded in [`docs/validation.md`](./docs/validation.md).
- **Agent-First Surfaces (`SKILL.md`, Built-in `stdio` MCP Server, JSON Schema, `llms.txt` & Cloud Shell Tutorial):** Portable Agent Skill ([`skills/scale-forecasting/SKILL.md`](./skills/scale-forecasting/SKILL.md)) with workspace auto-discovery ([`.agents/skills.json`](./.agents/skills.json) and `.agents/skills/scale-forecasting`), zero-extra JSON-RPC 2.0 MCP server ([`src/scale_forecasting/mcp.py`](./src/scale_forecasting/mcp.py)) exposing 7 `forecast://*` resources and 9 tools with `--allow-launch` safety gating, published `RunConfig` JSON Schema ([`docs/schemas/run_config.schema.json`](./docs/schemas/run_config.schema.json)), `llms.txt` / `llms-full.txt`, Google Antigravity [`plugin.json`](./plugin.json) + [`mcp_config.json`](./mcp_config.json), portable [`.mcp.json`](./.mcp.json), [`cloudshell_tutorial.md`](./cloudshell_tutorial.md), and pre-commit zero-drift synthesis ([`src/scale_forecasting/agent_surfaces.py`](./src/scale_forecasting/agent_surfaces.py)).

### Changed

- **Modular Dependency Extras (`pyproject.toml`):** Core `pip install scale-forecasting` installs only the pure offline forecasting layer with zero Google Cloud clients. All Google Cloud SDKs live in `[gcp]`, with composable `[notebook]`, `[spark]`, `[ray]`, `[submit]`, `[models-stats]`, `[models-trees]`, `[models-prophet]`, `[models-dl]`, `[models-automl]`, `[models]`, and `[all]` extras and actionable `MissingExtraError` guidance at every cloud and plotting entry point.
- **Static Typing, Logging & Test Ratchet:** Added PEP 561 `py.typed` marker with zero `mypy` errors across all 181 source modules, library-safe `NullHandler` logging (`configure_cli_logging` on CLI entry points), widened Ruff lint families (`SIM`, `C4`, `PIE`, `PERF`, `RUF`, `BLE`), and enforced an 85 % offline line-coverage ratchet in CI (`4,960` offline unit tests).
- **Python 3.11–3.12 Package Support:** `requires-python` widened from `>=3.11,<3.12` to `>=3.11,<3.13` with `3.12` classifiers and a per-interpreter CI `core-install` matrix; the cloud stack stays pinned to `3.11.15` (`.python-version`) for Spark Connect and Ray on Vertex pickle parity ([`docs/version_matrix.md`](./docs/version_matrix.md)). 3.13 waits on the `numpy<2` pin `neuralprophet` still needs.
- **Per-Job Provisioning vs. Execution Telemetry:** `run_jobs.job_telemetry.$.platform` (`provision_s`, `execute_s`, `total_wall_s`) is stamped from one probe read of each finished Dataproc Serverless batch or Vertex `CustomJob` (`batch_telemetry.platform_phases`), and `v_run_jobs` projects it as `provision_seconds` / `execute_seconds` beside `runtime_seconds`; the Serverless header telemetry carries the same split.
- **One BigQuery Client Factory:** Every `bigquery.Client(...)` in the package (73 sites) now goes through `registry.tables.bigquery_client`, which guards on `google.cloud.bigquery` alone via the new `errors.require_modules` so an on-cluster worker that never holds the launch clients still reaches the registry, and a core-only install gets the `[gcp]` install line instead of a `ModuleNotFoundError`.
- **`DataConfig` Identifier Validation:** `source_table` must be `table`, `dataset.table`, or `project.dataset.table` with no backticks, whitespace, or control characters, and `ts_id_col` / `date_col` / `target_col` must be BigQuery column names — checked once at config load, where the values are interpolated into backticked SQL. No field is added, so no `run_id` moves.

### Security

- **HTML Report Hardening (`report.py`):** The inline `EXPLORER_DATA` JSON escapes `<`, `>`, and `&` as `\u003c` / `\u003e` / `\u0026` so a series id containing `</script>` cannot close the script block, and the fan-chart label is HTML-escaped (`escHtml`) before `innerHTML` assignment.
- **FIPS-Safe Content Addressing:** `code_delivery` tags staged code archives with `hashlib.md5(..., usedforsecurity=False)` (a content-address, not a digest), so staging works on FIPS-enforced hosts.

[Unreleased]: https://github.com/statmike/scale-forecasting/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/statmike/scale-forecasting/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/statmike/scale-forecasting/releases/tag/v1.0.0
