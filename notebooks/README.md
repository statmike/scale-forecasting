# Interactive Notebooks (`notebooks/`)

<p align="center">
  <b>A Complete Interactive Curriculum for Enterprise Scale Forecasting on Google Cloud</b><br>
  <i>From local zero-GCP prototyping across all 37 models and 21 metrics to distributed execution across BigQuery ML, GCE, Vertex AI, GKE (Indexed Jobs & Ray on GKE), Vertex AI AutoML Tabular Workflows, Spark, and Ray — plus 3-tier covariates, time-series similarity & digital-twin analog search (@similarity_global, @similarity_cold_start, similarity_hierarchy), causal & econometric guardrails (@monotone, DML promo elasticity, moving-holiday windows, full-chain margin/inventory scenarios), two-tier explainability, 8-method FPP3 reconciliation, cross-run ensembling, custom 1-file plugins, and 100,000-series operations.</i>
</p>

```mermaid
flowchart TD
    subgraph Track1["Track 1: Local Sandbox, Auto-Forecaster & Custom Plugins (Zero GCP Required)"]
        NB00["00_model_playground.ipynb<br/>37 Models · 21 Metrics · Local/Global/Hybrid/@similarity_global<br/>3-Tier Covariates · @monotone · 9 Hierarchy/THieF · Similarity & Track L"]
        NB11["11_auto_forecaster_and_autofocus.ipynb<br/>Glass-Box Auto-Forecaster · 4 Astronomy Tiers<br/>Bounded autofocus · Forecaster.evolve · 20 DGPs"]
        NB09["09_custom_models_and_metrics.ipynb<br/>1-File BaseModel & BaseMetric Plugins<br/>+ Zero-Rebuild Dynamic Code Shipping"]
    end

    subgraph Track2["Track 2: Cloud Runtimes & Distributed Engines"]
        NB01["01_bigquery_native_sql.ipynb<br/>Serverless BigQuery SQL (ARIMA_PLUS · ARIMA_PLUS_XREG · TimesFM 2.5/3.0)"]
        NB02["02_vertex_and_gce_vms.ipynb<br/>GCE Single-VM, Vertex AI & GKE Indexed Jobs (CPU + L4 GPU)"]
        NB03["03_spark_serverless_and_connect.ipynb<br/>Dataproc Serverless Batch, Cluster & Spark Connect"]
        NB04["04_ray_on_vertex_gpu.ipynb<br/>Autoscaling Ray on Vertex AI & GKE + Fractional GPU Packing"]
    end

    subgraph Track3["Track 3: Covariates, Hierarchy, HPO & Cross-Run Ensembles"]
        NB05["05_covariates_and_global_models.ipynb<br/>3-Tier Covariates · @similarity_global · @monotone<br/>DML Elasticity · Holiday Windows · Margin/Inventory What-If"]
        NB06["06_hierarchical_reconciliation.ipynb<br/>Coherent Hierarchy & THieF Rollups (cross_temporal_mint)<br/>+ Compositional Simplex & Top-Down Goal-Seek QP"]
        NB07["07_hpo_backtesting_and_ensembles.ipynb<br/>Optuna HPO + In-Run, Post-Run & Cross-Run Ensembling"]
    end

    subgraph Track4["Track 4: Master Multi-Family DAG, Registry Ops & Run Review"]
        NB08["08_multi_engine_master_workflow.ipynb<br/>Multi-Family Parallel DAG + Foundation Models + Stacked Ensembles<br/>+ Online ACI, Copula VaR/CVaR, Portfolio Inventory & Scenario Opt"]
        NB10["10_registry_operations_and_scale.ipynb<br/>Registry Doctor, Live Probes, Composer DAG & Run Review"]
    end

    NB00 --> NB11 --> NB09
    Track1 --> Track2 --> Track3 --> Track4
```

---

## Twelve Focused & Combination Workflow Notebooks

Every cloud workflow notebook (`01`–`08`) is a **complete, self-contained 5-act (or 6-act) journey**:
1. **Setup & Architecture Context:** Bootstrap, `Settings.resolve()`, and visual Mermaid architecture map.
2. **Configure & Explain Plan:** Author `RunConfig` and inspect the pre-flight execution table via `forecaster.explain()`.
3. **Launch & Monitor Live:** Execute asynchronously with live in-place progress bars and infrastructure probes via `forecaster.run_live()`.
4. **Retrieve & Review Results:** Inspect `forecaster.leaderboard_df()`, `plot_leaderboard()`, `plot_metric_distribution()`, `forecaster.plot_forecasts()`, `forecaster.cohorts_df()`, `forecaster.jobs_df()`, `forecaster.attributions_df()`, `forecaster.plot_attributions()`, and `forecaster.trace()`.
5. **Deep Dive / Follow-Up Action:** Inspect 4-panel forecast decomposition (`forecaster.plot_forecast_explanation()`), two-tier feature attributions (`forecaster.attributions_df()`, `forecaster.plot_attributions()`), empirical interval calibration by horizon step (`forecaster.plot_calibration()`), learned ensemble stacking weights (`forecaster.plot_ensemble_weights()`), post-run `reensemble()`, cross-run `ensemble_runs()`, direct Spark/Ray engine embedding (`run_group`, `chunk_cells`), causal & econometric guardrails (`@monotone`, `promo_elasticity_df()`, `holiday_window`, and full-chain `simulate_scenarios()`), cross-temporal `THieF` reconciliation (`temporal_reconcile_df()`, `cross_temporal_mint`), compositional simplex (`@transform_clr`, `@transform_ilr`), top-down Goal-Seek QP (`allocate_target()`, `plot_goal_seek()`), Track L cold-start transfer (`cold_start_transfer_df()`), online ACI (`aci_df()`), copula tail risk (`sample_trajectories()`, `tail_risk_df()`), portfolio inventory optimization (`optimize_inventory_df()`), prescriptive price/promo optimization (`optimize_scenario()`), or Track P time-series similarity (`similarity_df()`, `find_similar_series_df()`, `cluster_by_similarity_df()`, `plot_similarity_map()`, `@similarity_global`, `@similarity_cold_start`, and `similarity_hierarchy=True`).

### Track 1: Foundations, Local Prototyping & Extensibility (Zero GCP Setup)
| Notebook | Target Environment | What You Will Learn |
| :--- | :--- | :--- |
| [`00_model_playground.ipynb`](./00_model_playground.ipynb) | Local Python (Offline) | Explore the full 37-model (5-family) and 21-metric catalogs (`model_catalog()`, `metric_catalog()`), generate synthetic panels (`sample_data()`), run 3-fold rolling-origin backtesting and conformal interval calibration (`run_model()`, `bakeoff()`), decompose forecasts into trend, regime level shift, seasonality, and exogenous covariate attribution (`explain_forecast_frame()`, `plot_forecast_explanation()`), inspect Two-Tier Feature Attributions (`build_attributions_frame()`, `plot_attributions()`), compare `local` vs. `global` vs. `hybrid` vs. `@similarity_global` training modes, run all 9 hierarchical & cross-temporal reconciliation methods, and evaluate the Forecaster Workbench (`fva_df()`, `horizon_decay_df()`, `bias_report_df()`, `rollup_df()`, `inventory_policy_df()`, `@monotone` sign/monotonicity constraints, Double/Debiased ML `promo_elasticity_df()` & `plot_promo_elasticity()`, moving-holiday `holiday_window` kernels, full-chain `simulate_scenarios(..., unit_cost=..., reconcile=True, include_inventory_impact=True)` with `compare_scenario_margin_curves()` & `plot_scenarios()`, `THieF` `temporal_reconcile_df()`, `@transform_clr`/`@transform_ilr` `forecast_simplex_panel()`, constrained `allocate_target()` + `plot_goal_seek()`, **Act 8 Track L**: `cold_start_transfer_df()`, online `aci_df()`, joint copula `sample_trajectories()` & `tail_risk_df()`, portfolio `optimize_inventory_df()`, and prescriptive `optimize_scenario()`, and **Act 11 (Act 9 Track P)**: 7-lens `similarity_df()` across `"hybrid"`, `"physics_attractor"`, `"spectral_phase"`, `"elastic_dtw"`, `"minirocket"`, `"structural_signature"`, and `"foundation_latent"`, `find_similar_series_df(mode="launch_prefix")` zero-metadata analog matching, `cluster_by_similarity_df()`, `plot_similarity_map()`, and `@similarity_global` / `@similarity_cold_start`) with zero cloud credentials. |
| [`09_custom_models_and_metrics.ipynb`](./09_custom_models_and_metrics.ipynb) | Local Python (Offline) | Author a custom 1-file `BaseModel` subclass (`@register`) and custom `BaseMetric` subclass, test them immediately in `run_model()` and `bakeoff()`, and inspect automatic BigQuery schema migration (`render_migrations`) and zero-rebuild `src.zip` shipping. |
| [`11_auto_forecaster_and_autofocus.ipynb`](./11_auto_forecaster_and_autofocus.ipynb) | Local Python (Offline) | Explore the **20-panel Synthetic DGP Benchmark Bank** (`list_benchmarks()`, `load_benchmark()`), compile transparent Glass-Box configurations across the **4 observational astronomy exposure tiers** (`first_light`, `wide_field`, `deep_sky`, `deep_field`) and **5 `n_series` scale regimes** (`micro` to `hyperscale`) via `Forecaster.build_config()`, predict wall-clock duration and cloud cost across all **7 runtimes** (`predict_run_duration()`), execute autonomous closed-loop `Forecaster.auto(..., autofocus=True)` with zero-waste cell borrowing (`autofocus_ledger_df()`, `baseline_value_df()`, `hierarchy_value_df()`, `plot_autofocus_progression()`), step through human-in-the-loop `step_by_step=True` (`preview_next_loop()`, `step()`), verify the white-noise stopping gate on `dgp_12_already_optimal_white_noise`, and run **post-run & multi-run evolutionary super-ensembling (`f.evolve()`, `Forecaster.evolve(from_runs=[...])`) + `+1e6` future-leakage verification (`verify_no_future_leakage`)**. |

---

### Track 2: Cloud Runtimes & Distributed Engines
| Notebook | Target Environment | What You Will Learn |
| :--- | :--- | :--- |
| [`01_bigquery_native_sql.ipynb`](./01_bigquery_native_sql.ipynb) | Serverless BigQuery ML | Execute `arima_plus` (`ARIMA_PLUS`), `arima_plus_xreg` (`ARIMA_PLUS_XREG` + `ML.EXPLAIN_FORECAST` Two-Tier attributions), and Google Research `timesfm@univariate` (`TimesFM 2.5`) vs. `timesfm` (`TimesFM 3.0` multivariate covariates via `AI.FORECAST`) directly inside BigQuery SQL with zero cluster provisioning, blend them into stacked ensembles, evaluate Covariate FVA (`fva_df()`), detect anomalies and structural change points (`detect_anomalies_df()`, `detect_change_points_df()`), and compute $(s, S)$ inventory replenishment policies (`inventory_policy_df()`, `plot_inventory_policy()`). |
| [`02_vertex_and_gce_vms.ipynb`](./02_vertex_and_gce_vms.ipynb) | GCE Single-VM, Vertex AI CustomJob & GKE Indexed Jobs | Compare all three direct container runtimes powered by `vertex_engine.py`: ephemeral GCE Single-VM (`runtime="gce"`, triple-redundant auto-delete), multi-worker Vertex AI CustomJob (`runtime="vertex"`), and Google Kubernetes Engine Indexed Jobs (`runtime="gke", gke_mode="job"`, per-model GPU pod auto-expansion and independent node scale-down), plus inspect Two-Tier Feature Attributions (`attributions_df()`, `plot_attributions()`). |
| [`03_spark_serverless_and_connect.ipynb`](./03_spark_serverless_and_connect.ipynb) | Dataproc Spark (`serverless` / `cluster` / `connect`) | Run the distributed Arrow-backed `groupBy(bucket).applyInPandas` fan-out on Dataproc Serverless in parallel with BigQuery SQL, compare with Dataproc Standard Clusters and interactive Spark Connect (`DataprocSparkSession`), and embed the pure Spark UDF directly via `spark_io.run_group()`. |
| [`04_ray_on_vertex_gpu.ipynb`](./04_ray_on_vertex_gpu.ipynb) | Ray on Vertex AI & Ray on GKE (CPU & GPU) | Provision ephemeral autoscaling Ray clusters on Vertex AI (`ray_mode="vertex"`) or GKE (`ray_mode="gke"` / `runtime="gke", gke_mode="ray"`), pack PyTorch deep learning tasks fractionally onto GPUs (`gpu_fraction=0.25`), compare `local`, `global`, and `hybrid` neural network regimes, and embed Ray task chunking directly via `ray_io.chunk_cells()`. |

---

### Track 3: Advanced Modeling — Covariates, Hierarchy, HPO & Cross-Run Ensembles
| Notebook | Target Environment | What You Will Learn |
| :--- | :--- | :--- |
| [`05_covariates_and_global_models.ipynb`](./05_covariates_and_global_models.ipynb) | Vertex AI / Spark + Exog Table | Configure all 3 exogenous covariate tiers (`static_covariates`, `future_covariates`, lag-shifted `past_covariates`) on `source_series_covariates_native`, train global cross-series deep learning (`tide`, `tsmixer`) and ML models (`lightgbm`, `xgboost`), visualize both structural exogenous covariate attribution (`plot_forecast_explanation`) and Two-Tier TreeSHAP feature attributions (`attributions_df`, `plot_attributions`), enforce **Causal & Structural Econometric Guardrails (Act 6)** across `@monotone` sign/monotonicity constraints (`regression_lags@monotone`, `ridge_lags@monotone`, `lightgbm@monotone`, `xgboost@monotone`, `catboost@monotone`), Double/Debiased ML price/promo elasticity (`promo_elasticity_df()`, `plot_promo_elasticity()`), moving-holiday lead/lag windows (`holiday_window`), and full-chain gross-margin, hierarchical reconciliation (`reconcile=True`), and inventory (`include_inventory_impact=True`) what-if scenario simulation (`simulate_scenarios()`, `summarize_scenarios()`, `compare_scenario_margin_curves()`, `plot_scenarios()`), and run **Cluster-Pooled `@similarity_global`, Digital-Twin Analog Search & Similarity-Synthesized Hierarchy Reconciliation (Act 7)** (`similarity_df()`, `cluster_by_similarity_df()`, `find_similar_series_df()`, `plot_similarity_map()`, `tide@similarity_global`, `lightgbm@similarity_global`, `@similarity_cold_start`, and `hierarchy.similarity_hierarchy=True` + `mint_shrink`). |
| [`06_hierarchical_reconciliation.ipynb`](./06_hierarchical_reconciliation.ipynb) | Vertex AI / GCE / Spark + Hierarchy | Configure multi-level business hierarchies (`region` $\rightarrow$ `category` $\rightarrow$ `ts_id`), reconcile forecasts across `bottom_up`, `wls_struct`, `mint_shrink`, and simultaneous `cross_temporal_mint` (`THieF` temporal hierarchy reconciliation via `forecaster.temporal_reconcile_df()`, `verify_temporal_coherence()`, `verify_cross_temporal_coherence()`), forecast compositional market-share panels with `@transform_clr` and `@transform_ilr` (`forecast_simplex_panel()`, `verify_simplex_coherence()`), and disaggregate executive top-down targets via constrained Minimum-Trace QP (`forecaster.allocate_target()` with `locked_nodes`, bounds, `price_col`, `unit_cost`, `include_inventory_impact=True`, and `forecaster.plot_goal_seek()`). |
| [`07_hpo_backtesting_and_ensembles.ipynb`](./07_hpo_backtesting_and_ensembles.ipynb) | BigQuery + Vertex AI | Tune hyperparameters with Optuna (`hpo.enabled=True`, `forecaster.best_params_df()`), inspect backtest cohort health (`cohorts_df()`), run **in-run ensembling**, add new stacked strategies in seconds with **post-run re-ensembling** (`forecaster.reensemble()`), combine base models across two separate runs using **cross-run ensembling** (`forecaster.ensemble_runs()`, plus how to extend to evolutionary super-ensembling via `Forecaster.evolve(from_run_ids=[...])`), and plot learned base-model weights (`plot_ensemble_weights()`). |

---

### Track 4: Master Multi-Family Workflow, Registry Operations & Run Review
| Notebook | Target Environment | What You Will Learn |
| :--- | :--- | :--- |
| [`08_multi_engine_master_workflow.ipynb`](./08_multi_engine_master_workflow.ipynb) | Spark $\parallel$ Vertex / GKE / AutoML $\parallel$ BigQuery | Dispatch active model families (`statistical`, `ml`, `deep_learning` including `[models-foundation]` `chronos_bolt`, and `native`) concurrently across their optimal runtimes from a single `RunConfig` (**zero-idle per-family compute** — including a 5-family execution plan preview with `automl` on `vertex_automl`), join them in a stacked ensemble, and run the complete diagnostic, explainability, anomaly/change-point, online ACI (`aci_df`), copula VaR/CVaR tail risk (`sample_trajectories`, `tail_risk_df`), portfolio inventory (`optimize_inventory_df`, `plot_inventory_frontier`), and prescriptive price/promo (`optimize_scenario`, `plot_optimal_scenario`) workbench (`leaderboard_df`, `fva_df`, `cohorts_df`, `jobs_df`, `attributions_df`, `plot_attributions`, `plot_calibration`, `plot_ensemble_weights`, `plot_forecasts`, `plot_forecast_explanation`, `plot_trace`, `detect_anomalies_df`, `detect_change_points_df`, `inventory_policy_df`, `plot_inventory_policy`). |
| [`10_registry_operations_and_scale.ipynb`](./10_registry_operations_and_scale.ipynb) | BigQuery Registry & Ops | Zero-SQL registry health audit (`reg.doctor()`, `reg.runs_df()`), historical run reattachment (`Forecaster.from_run_id()`, `attributions_df()`), pre-flight fold feasibility (`feasibility()`), preview-safe Day-2 verbs (`retry`, `settle`, `cancel`), **all 5 production launch pathways** (including live execution of the Cloud Composer 3 `airflow_tasks` DAG sequence against a GCS-staged config), and a cross-platform review of every completed run in the registry (the 100,000-series benchmark `run_id`s themselves are recorded in [Quota & Scale](../quota_and_scale.md)). |

---

## How to Run the Notebooks

### Option 1: 1-Click in Google Cloud Colab Enterprise (Recommended)

When you deploy the platform via Terraform (`create_colab_runtimes = true`, enabled by default), the **`sf-main`** Python 3.11 runtime template is pre-configured with all required environment variables (`SF_PROJECT_ID`, `SF_DATASET_ID`, `SF_CONNECTION`, `SF_WAREHOUSE_URI`, `SF_CODE_BUCKET`, `SF_CONTAINER_IMAGE`, `SF_COMPUTE_SA`, `SF_SUBNETWORK_URI`).

1. Open any notebook and click the **Run in Colab Enterprise** badge at the top.
2. Select the **`sf-main`** runtime template.
3. Click **Run all** — the environment bootstrap cell automatically sets up the locked dependencies from `uv.lock` and executes smoothly.

---

### Option 2: Local Jupyter / VS Code

1. Install the pinned environment and register the Jupyter kernel:
   ```bash
   uv sync --all-extras
   uv run python -m ipykernel install --user --name scale-forecasting
   ```
2. Export your deployment's environment variables (not required for `00_model_playground.ipynb`, `09_custom_models_and_metrics.ipynb`, or `11_auto_forecaster_and_autofocus.ipynb`):
   ```bash
   eval "$(cd terraform/main && terraform output -raw sf_env_exports)"
   ```
3. Open any notebook in your local IDE and select the `scale-forecasting` kernel.

---

## Automated Verification

All twelve notebooks are verified end-to-end using the headless acceptance test runner:

```bash
uv run python -m scale_forecasting.notebook_acceptance --tier all
```

For technical details on all 7 cloud compute runtimes and 4-tier scaling, see [`docs/runtimes_reference.md`](../runtimes_reference.md). For Colab Enterprise runtime templates and interpreter configurations, see [`docs/notebook_runtimes.md`](../notebook_runtimes.md).
