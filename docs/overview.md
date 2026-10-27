# Platform overview

`scale-forecasting` is a declarative, multi-engine time-series forecasting platform for Google Cloud. You describe *what* to forecast in one JSON configuration — the BigQuery source table, the horizon, the models, the evaluation metrics, and the ensemble strategies — and the platform uses BigQuery as the single data surface (with zero data replication) while compiling your declaration into a parallel per-family execution DAG across **7 Google Cloud runtimes**, evaluating **37 models** across **21 evaluation metrics**, extracting two-tier feature attributions, reconciling spatial and temporal hierarchies, and streaming governed results back into BigQuery tables and **5 Analytical SQL Views**.

```mermaid
flowchart LR
    subgraph PREP["1. Data, Config & Launch"]
        direction TB
        SRC["Your Data (BigQuery or DataFrame)<br/>series · timestamps · covariates · hierarchies"]
        CFG["Create a Config File (JSON)<br/>Hand-crafted OR use Auto-Forecaster"]
        LAUNCH["Launch & Orchestrate<br/>SDK · CLI · Notebooks · Airflow · Batch · MCP"]
        R["Parallel Family DAG Router<br/>Routes active families (1 job or pool per family)<br/>statistical · ml · deep_learning · automl · native"]
        SRC --> CFG --> LAUNCH --> R
    end
    subgraph RUN["2. Zero-Idle Compute on Google Cloud (Chosen per Family)"]
        direction LR
        S["Serverless Spark or Spark Cluster"]
        RY["Ray on Vertex AI or GKE"]
        V["Vertex AI CustomJob"]
        G["Compute Engine Single VM (Self-Deleting)"]
        K["GKE (Indexed Job or Ray · Scale-to-Zero)"]
        A["Vertex AI AutoML & Tabular Workflows"]
        B["BigQuery ML (CREATE MODEL · AI.FORECAST)"]
    end
    subgraph OUT["3. Governed Results & Trust in BigQuery"]
        direction TB
        REG["5 BigQuery Registry Tables<br/>forecasts · backtests · metadata<br/>runs · jobs"]
        VIEWS["5 Analytical SQL Views<br/>Leaderboards · Backtest Coverage<br/>Run & Job Summaries"]
        USE["Practitioner Trust & Consumption<br/>7-Question Assessment · FVA Ladder<br/>Attributions · SDK · Looker"]
        REG --> VIEWS --> USE
    end

    PREP --> RUN --> OUT

    classDef data fill:#E8F0FE,stroke:#4285F4,stroke-width:2px,color:#174EA6
    classDef config fill:#FEF7E0,stroke:#F9AB00,stroke-width:2px,color:#7A5A00
    classDef route fill:#FCE8E6,stroke:#EA4335,stroke-width:2px,color:#A50E0E
    classDef compute fill:#E6F4EA,stroke:#34A853,stroke-width:2px,color:#0D652D
    class SRC,REG,VIEWS,USE data
    class CFG,LAUNCH config
    class R route
    class S,RY,V,G,K,A,B compute
```

New to the project? Start with [Getting started](./getting_started.md) for a 5-minute local forecast and your first cloud run.

---

## Why `scale-forecasting`?

Traditional forecasting workflows break down when scaled to tens or hundreds of thousands of series across retail, supply chain, energy, or financial hierarchies:

| Challenge | Traditional approach | The `scale-forecasting` solution | Deep dive |
| :--- | :--- | :--- | :--- |
| **Library fragmentation** | Separate, incompatible codebases for Statsmodels, StatsForecast, Prophet, PyTorch, Foundation models, Vertex AI AutoML, and SQL models. | **Unified model contract:** Single [`BaseModel`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/models/base_model.py) interface. All 37 models (`local`, `global`, and `hybrid`) run with identical inputs, outputs, and metrics. | [models_reference.md](./models_reference.md) |
| **Compute scaling limits** | Single-node memory exhaustion; slow sequential loops. | **Per-family distributed execution:** Automatic fan-out across Dataproc Serverless, Vertex AI CustomJob, Compute Engine (`gce`), Google Kubernetes Engine (`gke`), Ray on Vertex AI or GKE, Vertex AI AutoML, and BigQuery ML. | [runtimes_reference.md](./runtimes_reference.md) |
| **Infrastructure lock-in** | Forced choice between pure Spark, pure Ray, or pure SQL. | **Multi-engine DAG:** Run Spark, Ray, Vertex AI CustomJob, Compute Engine, GKE, Vertex AI AutoML, and BigQuery ML concurrently under one `run_id`, bounded by the slowest family rather than their sum. | [architecture.md](./architecture.md) |
| **Uncertainty calibration** | Gaussian assumptions that fail on skewed real-world distributions. | **Conformal residual intervals:** Empirical, distribution-free prediction intervals calibrated against rolling out-of-fold backtest errors. | [backtesting.md](./backtesting.md) |
| **Hierarchical & temporal incoherence** | Bottom-level, upper-level, and multi-frequency (Daily/Weekly/Monthly) forecasts do not add up, or market shares fail to sum to 100%. | **3-mode coherent reconciliation & compositional simplex:** Built-in spatial (`mint_nonneg`, `mint_shrink`, `wls_var`, `wls_struct`, `ols`, `bottom_up`, `top_down`, `middle_out`), temporal (`THieF`), and joint cross-temporal (`cross_temporal_mint`) reconciliation plus `@transform_ilr` / `@transform_clr` simplex closure and top-down Goal-Seek QP (`allocate_target`). | [hierarchical_forecasting.md](./hierarchical_forecasting.md) · [features_and_covariates.md](./features_and_covariates.md) |
| **Operational opacity** | Disconnected log files and ad-hoc evaluation tables. | **BigQuery registry & explainability:** Streaming ingestion via the Storage Write API, two-tier feature attributions (`attributions_df` / `plot_attributions`), and 5 analytical SQL views. | [output_schemas.md](./output_schemas.md) |
| **Brittle batch failures** | One failed series aborts an entire multi-hour distributed job. | **Surgical cell repair & live probes:** Re-runs only failed cells without recomputing successful ones and reconciles platform job state automatically. | [operations.md](./operations.md) |
| **Manual trial-and-error vs. black-box AutoML** | Days of hand-tuning configs or an opaque black box that hides why models were picked. | **Glass-box auto-forecaster, `autofocus` & opt-in `Forecaster.evolve`:** `Forecaster.build_config()` and `Forecaster.auto()` (evolved with **AlphaEvolve on Gemini Enterprise** into pure Python) explain every choice, honor user pins, and borrow finished models across passes; `Forecaster.evolve()` adds opt-in runtime `# EVOLVE-BLOCK` evolution under a 3-Gate Anti-Overfitting Ratchet. | [auto_forecaster.md](./auto_forecaster.md) · [alphaevolve_runner.md](./alphaevolve_runner.md) |

---

## Technology stack on Google Cloud

Every capability maps cleanly to managed Google Cloud data and AI services:

| Layer | Google Cloud service | Role in `scale-forecasting` | Reference |
| :--- | :--- | :--- | :--- |
| **Data warehouse & lakehouse** | BigQuery & BigLake Apache Iceberg | Stores source panels (`source_series_native`, `source_series_iceberg`), the 5 registry tables, and the 5 analytical SQL views. | [reading_source_data.md](./reading_source_data.md) · [output_schemas.md](./output_schemas.md) |
| **SQL-native forecasting** | BigQuery ML | Executes `ARIMA_PLUS`, `ARIMA_PLUS_XREG`, and zero-shot foundation models via `AI.FORECAST` (`TimesFM`) directly in SQL. | [runtimes_reference.md](./runtimes_reference.md) |
| **Managed AutoML & pipelines** | Vertex AI AutoML & Tabular Workflows (`vertex_automl`) | Runs managed Tabular Workflow pipelines and AutoML training jobs (`vertex_l2l`, `vertex_tide`, `vertex_tft`, `vertex_seq2seq`) with Stage-1 HPO reuse and baseline attributions. | [models_reference.md](./models_reference.md) |
| **Serverless & cluster Spark** | Dataproc Serverless & Dataproc on GCE (`spark`) | Runs Arrow-backed `applyInPandas` fan-out across serverless batches, managed clusters, or interactive Spark Connect sessions. | [runtimes_reference.md](./runtimes_reference.md) |
| **Single-VM & worker-pool jobs** | Vertex AI Custom Training (`vertex`) & Compute Engine (`gce`) | Runs serverless single-VM (`workers=1` or `runtime="gce"` with triple-redundant self-deletion) and multi-VM worker pools (`CPU`, `T4`, `L4`, `A100`, `A100_80GB`) with Storage Read API row-range pushdown. | [runtimes_reference.md](./runtimes_reference.md) |
| **Kubernetes batch & Ray** | Google Kubernetes Engine (`gke`) & Vertex AI Ray (`ray`) | Runs Kubernetes `batch/v1` Indexed Jobs (`gke_mode="job"`) and autoscaling Ray pools (`ray_mode="vertex"` or `"gke"`) with fractional GPU packing. | [runtimes_reference.md](./runtimes_reference.md) · [quota_and_scale.md](./quota_and_scale.md) |
| **Notebooks & orchestration** | Colab Enterprise & Cloud Composer 3 | Hosted notebooks pre-wired to the `sf-main` runtime template and optional scheduled Airflow DAGs (`--emit-airflow`). | [notebook_runtimes.md](./notebook_runtimes.md) · [deploying_on_gcp.md](./deploying_on_gcp.md) |

---

## How runs work: declarative JSON configurations

A single JSON file (or Python dictionary validated by `RunConfig`) defines an experiment end to end and hashes to a deterministic `<slug>-<12hex>` `run_id`:

```json
{
  "run_name": "hybrid_stacked_ensemble",
  "data": {
    "source_table": "source_series_iceberg",
    "series_limit": 100,
    "horizon": 14
  },
  "models": ["theta", "holtwinters", "xgboost", "arima_plus"],
  "compute": {
    "families": {
      "statistical": {"runtime": "spark"},
      "ml": {"runtime": "spark"}
    }
  },
  "backtest": {
    "enabled": true,
    "scheme": "expanding",
    "n_folds": 3,
    "decision_metric": "wape"
  },
  "ensemble": {
    "enabled": true,
    "strategies": ["mean", "inverse_error", "nnls", "xgb"]
  }
}
```

| Section | What it controls |
| :--- | :--- |
| `data` | Source table (native BigQuery or BigLake Iceberg), `series_limit`, series/date/target columns, frequency, and `horizon`. |
| `models` | Model identifiers to fit, grouped automatically into `statistical`, `ml`, `deep_learning`, `automl`, and `native` families. |
| `compute` | Default and per-family `runtime`, mode (`spark_mode`, `gke_mode`, `ray_mode`, `automl_mode`), `machine_type`, `workers`, GPUs, and regional fallback. |
| `backtest` | Cross-validation scheme (`expanding`, `sliding`, `expanding_frozen`, `sliding_frozen`), fold count, `decision_metric`, and conformal / online ACI interval calibration (`output.calibration_method`). |
| `features` | Country holidays, multi-day `holiday_window` (`pre_holiday_days`, `post_holiday_days`), Fourier seasonality, level-shift indicators, `static_covariates`, `future_covariates`, `past_covariates`, `exog_lags`, piecewise weather splines (`degree_day_cols`: `hdd_<col>`, `cdd_<col>`), `monotonic_constraints` (`@monotone`), compositional simplex (`clr` / `ilr`), `similarity_cluster_feature`, product generation splicing (`supersession_map`), and cold-start / NPI analog transfer (`cold_start_transfer`, `cold_start_min_obs`, `cold_start_method` including `"arps_decline"` / `@arps_decline`, `cold_start_matching`). |
| `hpo` | Optuna trial budget, search spaces, and fleet-wide vs. per-series tuning. |
| `ensemble` | Consensus (`mean`, `median`, `inverse_error`, `champion`, `horizon_specialist`), digital-twin OOF loss smoothing (`similarity_smoothing`), and learned stackers (`nnls`, `ridge`, `xgb`), plus `barrier` or `microbatch` execution. |
| `hierarchy` | Aggregation levels (`levels`), latent similarity cluster hierarchy discovery (`similarity_hierarchy`, `similarity_n_clusters`), and coherent reconciliation methods (`reconciliation_methods`). |

Full field-by-field specification: [Configuration reference](./configuration_reference.md).

---

## Five ways to run

The same orchestration path (`main.run`) drives every launch surface:

```mermaid
flowchart TD
    src["Author via Hand-Crafted RunConfig JSON<br/>OR Glass-Box Auto-Forecaster (Forecaster.build_config / Forecaster.auto)"] --> cfg["Validated RunConfig (<slug>-<12hex> run_id)"]
    cfg --> E1["1. Python SDK (Forecaster.run / Forecaster.auto)"]
    cfg --> E2["2. CLI (python -m scale_forecasting.main --config / --auto)"]
    cfg --> E3["3. Staged Native CLI (launch_plan.stage_run)"]
    cfg --> E4["4. Cloud Composer 3 (--emit-airflow single-pass or autofocus DAG)"]
    cfg --> E5["5. Direct Cluster Embed (make_group_runner / make_chunk_runner)"]
```

1. **Python SDK (`Forecaster`):** High-level facade for notebooks and services (`dry_run()`, `feasibility()`, `run()`, `run_live()`, `review_run()`).
2. **Command-line interface (`scale_forecasting.main`):** Direct terminal and CI execution (`python -m scale_forecasting.main --config configs/ensemble_demo.json`).
3. **Staged native commands (`launch_plan.stage_run`):** Stages `src/` and config to Cloud Storage and prints copy-pasteable `gcloud` / `kubectl` / `bq` / `ray` commands.
4. **Cloud Composer 3 DAG generation (`--emit-airflow`):** Compiles a `RunConfig` into a self-contained Apache Airflow DAG with parallel per-family tasks.
5. **Direct cluster embedding:** Embed `make_group_runner` (PySpark `applyInPandas`) or `make_chunk_runner` (Ray) inside an existing pipeline.

**AI coding agents & MCP clients:** Every execution surface above is also exposed through the built-in Model Context Protocol server (`python -m scale_forecasting.mcp`, 12 tools including `compute_time_series_similarity`), the portable [`SKILL.md`](https://github.com/statmike/scale-forecasting/blob/main/skills/scale-forecasting/SKILL.md), and the [`RunConfig` JSON Schema](./schemas/run_config.schema.json) — see [AI Agents, Portable Skills & Built-in MCP Server](./agent_and_mcp_guide.md).

Details: [Using the SDK](./using_the_sdk.md) and [Running and reviewing](./running_and_reviewing.md).

---

## Per-family DAG and automated resource sizing

When a run starts, `dag.plan_dag` splits active models into up to five family jobs (`statistical`, `ml`, `deep_learning`, `automl`, `native`) and dispatches them in parallel. Fast CPU families finish and release their machines immediately without waiting for multi-epoch GPU trainers; once the base families complete (or continuously in `microbatch` mode), the `ensemble` node blends their predictions.

Before each family launches, the resource planner ([`scale_forecasting.resources`](./api/resources.md)) combines empirical per-cell memory and wall-clock profiles (`ComputeProfile`) with your target wall-clock budget to size executors, worker pools, thread counts, and fractional GPU shares automatically — and records the full sizing decision (`$.sizing.<family>`) on the run header for inspection in `v_run_summary`. Run `--dry-run` for an offline fan-out plan, `--feasibility` for live panel length and fold-coverage checks, or `--quota` to verify regional vCPU and GPU quota ahead of submission.

Deep dives: [Compute runtimes reference](./runtimes_reference.md) and [Quota and scale guide](./quota_and_scale.md).

---

## Models, explainability, reconciliation, and ensembles

### 37 models across 5 families

Every model inherits from [`BaseModel`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/models/base_model.py), lives in its own module under [`src/scale_forecasting/models/`](https://github.com/statmike/scale-forecasting/tree/main/src/scale_forecasting/models), and imports its upstream library lazily inside `fit()`:

| Family | Count | Representative models | Training modes | Runtimes |
| :--- | :---: | :--- | :--- | :--- |
| `statistical` | **18** | `theta`, `auto_theta`, `holtwinters`, `autoets`, `auto_arima`, `sarimax`, `tbats`, `auto_ces`, `stl_bagging`, `ucm`, `kalman`, `prophet`, `croston`, `fft`, `naive_*` | `local` | `spark`, `ray`, `vertex`, `gce`, `gke` |
| `ml` | **5** | `lightgbm`, `xgboost`, `catboost`, `random_forest`, `regression_lags` (`ridge_lags` alias) | `local`, `@global`, `@similarity_global`, `@monotone` | `spark`, `ray`, `vertex`, `gce`, `gke` |
| `deep_learning` | **8** | `tide`, `tft`, `tsmixer`, `patchtst`, `neuralprophet`, `chronos_bolt`, `chronos2`, `timesfm_torch` | `local`, `global`, `@similarity_global`, `hybrid` | `spark`, `ray`, `vertex`, `gce`, `gke` |
| `automl` | **4** | `vertex_l2l`, `vertex_tide`, `vertex_tft`, `vertex_seq2seq` | `global` | `vertex_automl` |
| `native` | **2** | `arima_plus`, `timesfm` / `bq_timesfm` (`AI.FORECAST`) | `local` (SQL / zero-shot) | `bigquery` |

### Platform capability tree

- **Feature Engineering, Causal Guardrails, Compositional Simplex, Similarity & Cold-Start Transfer** — [Features, covariates & simplex](./features_and_covariates.md) · [Similarity API](./api/similarity.md)
  - **3-tier covariates, `@monotone` sign guardrails & `holiday_window`:** Future (`price`, `promo`), past-only, and static covariates, country holidays, automatic Fourier harmonics, structural level-shift detection, downward price (`-1`) / upward promo (`+1`) sign constraints (`features.monotonic_constraints`), and pre/post-holiday ramps (`pre_holiday_days`, `post_holiday_days`).
  - **Multi-lens & multi-modal time-series similarity (`similarity.py`), `@similarity_global` pooling & merchandising intelligence:** Compute symmetric PSD similarity matrices across 7 lenses (`"hybrid"`, `"physics_attractor"`, `"spectral_phase"`, `"elastic_dtw"`, `"minirocket"`, `"structural_signature"`, `"foundation_latent"`) fused with multi-modal metadata (`text_cols`, `attribute_cols`, `hierarchy_cols`), discover Kuramoto phase synchronization and lead-lag shifts ($\tau_{i \to j}^*$), pool global models within dynamical clusters (`<model>@similarity_global`), and generate inventory-aware SKU substitutes (`recommend_substitutes_df`) and emerging trend breakouts (`detect_emerging_trends_df`).
  - **Compositional market-share simplex (`@transform_ilr`, `@transform_clr`) & Day-0 to Day-14 Cold-Start Analog Transfer (`@similarity_cold_start`):** Forecast brand or channel shares in Aitchison log-ratio space (`simplex_sum`) and synthesize Day-0 (`0` obs) through Day-14 NPI series (`n < min_obs`) via smooth semantic-to-kinematic launch-prefix handoff (`cold_start_matching="similarity" | "hybrid"`, `cold_start_handoff="auto"`) or Bass/Weibull lifecycle curves (`cold_start_transfer_df`).
- **3-Mode Hierarchical Reconciliation (`THieF`, Cross-Temporal & `similarity_hierarchy`)** — [Hierarchical & cross-temporal forecasting](./hierarchical_forecasting.md)
  - **Mode 1 — Spatial & latent similarity hierarchies:** Reconcile explicit trees (`hierarchy.levels`) or auto-discovered `similarity_cluster_id` groups on flat panels (`hierarchy.similarity_hierarchy: true`) via all 8 Hyndman FPP3 methods (`mint_nonneg`, `mint_shrink`, `wls_var`, `wls_struct`, `ols`, `bottom_up`, `top_down`, `middle_out`).
  - **Mode 2 — Temporal hierarchies (`THieF`):** Opt-in `hierarchy.reconcile_temporal: true` reconciles any flat panel across calendar frequencies (`Daily <-> Weekly <-> Monthly <-> Annual`) on both forward horizons and backtest folds.
  - **Mode 3 — Simultaneous cross-temporal (`cross_temporal_mint`):** Enforce spatial and temporal coherence simultaneously (`0.000000` discrepancy on both axes) with OOF-learned `spatial_temporal_weight="auto"`, raced head-to-head against pure spatial methods on the leaderboard.
  - **Coherent Monte Carlo trajectories (`sample_coherent_paths`):** Draw joint multi-step sample paths where every individual path satisfies exact spatial and temporal additivity.
- **Smart Ensembling, Explainability & Modular Plugins** — [Models and ensembles reference](./models_reference.md) · [Adding a model](./adding_a_model.md)
  - **8 ensemble strategies, similarity OOF smoothing & zero-refit re-ensembling:** Blend via heuristic/winner rules (`mean`, `median`, `inverse_error`, `champion`, `horizon_specialist`) with digital-twin OOF loss smoothing (`ensemble.similarity_smoothing`) or out-of-fold stackers (`nnls`, `ridge`, `xgb`) with `top_k` and `parsimony_margin` pruning, and re-ensemble completed runs (`forecaster.reensemble()`, `forecaster.ensemble_runs()`).
  - **Two-tier explainability:** Global driver importance in `forecast_metadata.fit_diagnostics["feature_attributions"]` (Tier 1) and per-step signed attributions in `forecast_predictions.explanations` (Tier 2) across `ml`, `automl`, and `arima_plus` (`ML.EXPLAIN_FORECAST`), plus structural decomposition (`forecaster.explain_forecast()`).
  - **Modular installation & 1-file plugins:** `pip install scale-forecasting` installs the pure offline layer (15 models on NumPy/SciPy/statsmodels/scikit-learn, all 21 metrics, playground, `--dry-run`); optional extras (`[gcp]`, `[spark]`, `[ray]`, `[notebook]`, `[models]`, `[all]`) add cloud clients and third-party model libraries.
- **Glass-Box Auto-Forecaster, Multi-Branch `autofocus` & Opt-In `Forecaster.evolve`** — [Auto-forecaster & `autofocus`](./auto_forecaster.md) · [AlphaEvolve Runtime Harness](./alphaevolve_runner.md)
  - **4 exposure tiers & 7-Question Practitioner Assessment:** Compile transparent `ConfigBlueprint` configs (`Forecaster.build_config()` / `--init-config`) or run multi-branch `autofocus=True` tournaments (`Forecaster.auto()`) across `first_light`, `wide_field`, `deep_sky`, and `deep_field`, evolved with **AlphaEvolve on Gemini Enterprise** across 20 synthetic DGPs and 9 public open-data benchmarks.
  - **Runtime `# EVOLVE-BLOCK` evolution (`Forecaster.evolve`):** Evolve custom `"features"`, `"ensemble"`, `"model"`, `"policy"`, or `"similarity_kernel"` logic via Gemini Enterprise AlphaEvolve (`backend="alphaevolve"` default, `enable_alphaevolve = true`) or standard Vertex AI Gemini (`backend="vertex_gemini"` backup) under an AST security sandbox, `+1e6` lookahead probe, and 3-Gate Anti-Overfitting Ratchet.
- **Forecaster Decision Workbench, Online ACI, Copula Risk, Portfolio Inventory & Prescriptive Optimization** — [Forecaster workbench](./forecaster_workbench.md)
  - **Double/Debiased ML promo & price decomposition:** Robinson/Chernozhukov partialing-out (`promo_elasticity_df`) into `baseline_yhat`, `incremental_promo_lift`, `pull_forward_dip`, similarity-prior `cannibalization_effect` (`use_similarity_priors=True`), and `net_causal_elasticity`.
  - **Online Adaptive Conformal Inference (`aci.py`) & Joint Copula Tail Risk (`copula.py`):** Non-exchangeable online coverage updates (`output.calibration_method = "aci" | "aci_normalized"`, `aci_df`) and similarity-regularized Gaussian/Student-$t$ copula trajectory fans (`sample_trajectories(..., similarity_shrinkage=True)`) with portfolio VaR and CVaR (`tail_risk_df`).
  - **5-stage What-If scenarios, Lifecycle Clearance MILP (`clearance_optimizer.py`), Assortment Transference (`optimize_assortment_df`) & Price/Promo Calendar Optimizer (`scenario_optimizer.py`):** Propagate price/promo shocks (`simulate_scenarios`), solve monotone markdown schedules with digital-twin elasticity borrowing and shelf/warehouse `target_space_freed` constraints (`optimize_clearance_df`), hit supplier rebate tiers or Take-or-Pay commitments (`optimize_contract_volume_df`), rationalize SKU assortments with similarity-driven demand transference and `walk_rate` (`optimize_assortment_df`), or solve for the gross-margin / CVaR-optimal forward price and promotional schedule (`optimize_scenario`, `plot_optimal_scenario`).
  - **Top-down Goal-Seek QP, Multi-SKU Portfolio Inventory, Simplex Pre-Packs & Lateral Rebalancing (`inventory_optimizer.py`):** Solve covariance-weighted target disaggregations (`allocate_target`), multi-SKU replenishment with similarity-pooled residual shrinkage (`similarity_pooling=True`) under shared working-capital (`max_budget`), warehouse volume (`max_volume`), aggregate fill-rate, and MOQ constraints (`optimize_inventory_df`, `plot_inventory_frontier`), Aitchison CLR simplex size-curve and integer pre-pack bundle allocation (`optimize_prepacks_df`, `estimate_simplex_size_curves`), and Hitchcock–Koopmans lateral store-to-store/depot rebalancing (`optimize_rebalancing_df`), plus interactive HTML reports (`export_report`).

Full catalog: [Models and ensembles reference](./models_reference.md), [Forecaster workbench](./forecaster_workbench.md), and [Runtime dependencies](./runtime_dependencies.md).

---

## 21 evaluation metrics

Every model, reconciled hierarchy node, and ensemble is scored in Python (`metrics.compute_metrics`) across a uniform **21-metric evaluation panel** stored in `forecast_metadata`:

| Category | Count | Metrics |
| :--- | :---: | :--- |
| **Point forecast accuracy** | **16** | Relative (`wape`, `smape`, `mape`, `maape`, `ope`), scale-dependent (`mae`, `rmse`, `mse`, `rmsle`, `bias`), scaled against naive baselines (`mase`, `mase_seasonal`, `rmsse`, `msse`), goodness-of-fit & dispersion (`r2`, `cv`). |
| **Prediction interval quality** | **5** | Empirical `coverage`, quantile `pinball` loss, Winkler `interval_score`, `interval_width`, and M4 scaled `msis` over the 80% $(q_{0.10}, q_{0.90})$ prediction band. |

Formulas, calibration modes, and custom metrics: [Evaluation metrics reference](./metrics_reference.md) and [Adding a metric](./adding_a_metric.md).

---

## Packaged synthetic data & scale seeding

The deterministic generator ([`scale_forecasting.data_gen`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/data_gen/README.md)) synthesizes five archetypes (`smooth_seasonal`, `intermittent`, `trending`, `promo_spiky`, `noisy`) in either a 5-column univariate mode or a 10-column mode with 3-tier covariates (`region`, `category`, `is_holiday`, `promo_flag`, `price_index`, `temperature`) and a 3-level hierarchy. Because each series is seeded solely by `(master_seed, series_index)`, generating 3 series in memory via `playground.sample_data()` produces the exact same values for those series as seeding 100,000 series across Spark executors into `source_series_native` and `source_series_iceberg`. See [data_gen API](./api/data_gen.md).

---

## BigQuery registry, views, and operations

Workers stream Arrow batches through the BigQuery Storage Write API into five append-only tables (`run_registry`, `run_jobs`, `forecast_metadata`, `forecast_predictions`, `backtest_oof`), exposed through **5 Analytical SQL Views**:

- `v_model_leaderboard` — per-run model and ensemble ranking across accuracy, fit duration, and error rate.
- `v_model_leaderboard_comparable` — holdout-fold pooled WAPE and MAE (`SUM(|y - yhat|) / SUM(|y|)`) so models are compared on identical series and fold windows.
- `v_backtest_coverage` — achieved fold histogram and backtest status (`full`, `reduced`, `unscored`, `failed`) per model.
- `v_run_summary` — one row per run with scaling knobs, the job-row time ledger (`n_jobs`, `longest_job_seconds`, `jobs_seconds`, `jobs_span_seconds`, `overhead_seconds`, `overhead_fraction`), and sizing/capacity telemetry.
- `v_run_jobs` — deduplicated per-family execution trace with runtime, hardware, platform job ID, wall-clock bracket, and GPU `device_verdict`.

Day-2 management is built into `python -m scale_forecasting.registry.ops` and `sf.Registry`: `doctor` (health and quota audit), `close-runs` (reconcile stale headers), `retry_run` / `Forecaster.retry()` (surgical cell repair), `drop-run`, `sweep-orphans`, `reap-clusters`, `snapshot`, and `export`. Schemas and runbooks: [Output schemas](./output_schemas.md) and [Operations guide](./operations.md).

---

## Persona journeys, notebooks, and deployment

```mermaid
flowchart TD
    subgraph DS["Data scientist & forecaster"]
        direction TB
        DS1["Local playground<br/>00_model_playground.ipynb"] --> DS2["Custom plugins<br/>09_custom_models_and_metrics.ipynb"] --> DS3["HPO & ensembles<br/>07_hpo_backtesting_and_ensembles.ipynb"]
    end
    subgraph AR["Cloud & data architect"]
        direction TB
        AR1["Storage format<br/>BigQuery native vs BigLake Iceberg"] --> AR2["Runtime placement<br/>Spark · Ray · Vertex · GCE · GKE · AutoML · BQ"] --> AR3["Private networking & IAM<br/>VPC · PSC-I · least-privilege SAs"]
    end
    subgraph OP["MLOps & platform engineer"]
        direction TB
        OP1["Two-stage Terraform<br/>terraform/README.md"] --> OP2["Scheduled DAGs<br/>Cloud Composer 3 (--emit-airflow)"] --> OP3["Resilience & repair<br/>regional fallback · probes · retry_run"]
    end
    DS3 & AR3 --> OP1
```

- **12 interactive notebooks across 4 tracks:** From zero-GCP local exploration and glass-box auto-forecasting (`00`, `11`, `09`) to cloud runtimes (`01`–`04`), covariates/hierarchy/HPO/ensembles (`05`–`07`), and the multi-family master DAG & registry operations (`08`, `10`). Browse the [Notebook tour](./notebooks/README.md) or the guided [Hands-on workshop](./workshop.md).
- **Two-stage Terraform deployment (~15 minutes):** Stage 1 (`terraform/bootstrap`) provisions the remote state bucket; Stage 2 (`terraform/main`) provisions the VPC, service accounts, buckets, BigQuery dataset, container image (Cloud Build + Artifact Registry), Colab Enterprise runtime template, and the 100,000-series seed batch on Dataproc Serverless.
- **Estimated deployment and runtime cost:** The one-time image build and 100,000-series seed batch typically cost an estimated **~\$0.25–\$1.00** in Cloud Build, Dataproc Serverless, and BigQuery Storage Write API usage. At rest, buckets, datasets, VPC networks, and service accounts have no compute charge (only standard Cloud Storage and BigQuery storage rates for stored bytes). Forecast runs bill per job for the duration of the batch, VM, pod, or query. **Ongoing cost disclosure:** Cloud Composer 3 (`create_composer = false` by default) and standing GKE clusters (`create_gke = false` by default) run continuously while provisioned and incur ongoing environment/cluster charges until disabled (and deleting a Composer environment leaves behind its Cloud Storage bucket until removed manually). See [Cost estimates & controls](./cost_estimates.md), [Choosing a runtime](./choosing_a_runtime.md), [Why Google Cloud](./why_google_cloud.md), [FAQ](./faq.md), [Glossary](./glossary.md), and [Deploying on GCP](./deploying_on_gcp.md).
