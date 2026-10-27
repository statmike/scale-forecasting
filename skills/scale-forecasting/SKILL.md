---
name: scale-forecasting
description: Configure, validate, dry-run, execute, review, and repair enterprise time-series forecasting and prescriptive optimization pipelines on Google Cloud or offline using scale-forecasting (37 models across statistical, ML, deep learning, Vertex AI AutoML, and BigQuery ML; 21 metrics; 7 runtimes; 5 execution paths; Glass-Box Auto-Forecaster across 4 exposure tiers, multi-branch autofocus, AlphaEvolve/Vertex Gemini code evolution, 20 synthetic DGP + 9 public benchmarks, and 7-Question Practitioner Assessment; causal @monotone guardrails & Double-ML promo elasticity; 3-mode spatial/temporal THieF & cross_temporal_mint reconciliation, @transform_ilr/@transform_clr compositional simplex, and top-down Goal-Seek QP; cold-start NPI analog/Bass/Weibull transfer, 7-lens multi-modal time-series similarity (hybrid, physics_attractor, spectral_phase, elastic_dtw, minirocket, structural_signature, foundation_latent, @similarity_global, @similarity_cold_start, recommend_substitutes_df, detect_emerging_trends_df), online Adaptive Conformal Inference ACI, joint Gaussian/Student-t copula trajectories & VaR/CVaR tail risk, multi-SKU portfolio inventory optimization, lifecycle markdown & perishable clearance MILP (optimize_clearance_df), vendor contract volume & rebate tier optimization (optimize_contract_volume_df), and prescriptive price/promo scenario optimization). Use whenever a user asks to forecast time series, profile a dataset, compute time-series similarity, recommend SKU substitutes, detect emerging trends, or cluster analog series, auto-compile or hand-author a RunConfig JSON, run autofocus or Forecaster.evolve(), answer practitioner trust questions, analyze promo/price elasticity, reconcile spatial/temporal hierarchies or market-share simplexes, disaggregate top-down targets, handle cold-start NPI launches, simulate copula tail risk, optimize inventory budgets/MOQs, clearance markdowns, vendor contract volumes, or promo/price calendars, launch jobs on Spark/Ray/Vertex/GCE/GKE/AutoML/BigQuery, emit an Airflow DAG, inspect BigQuery forecast registries, or surgically repair failed cells.
license: Apache-2.0
metadata:
  author: statmike
  version: 1.9.0
---

# `scale-forecasting` — Agent Skill & Workflow Guide

`scale-forecasting` is a declarative, config-driven enterprise time-series forecasting platform for Google Cloud and local Python environments. Everything is driven by a single Pydantic [`RunConfig`](../../src/scale_forecasting/config.py) JSON/dict contract that executes identically across all 5 execution paths (Offline Playground & Dry-Run, Cloud Preflight & Staging, Unified Multi-Family DAG Execution, Direct Per-Family Submitters, and Cloud Composer 3 Airflow DAG Emission).

---

## Progressive Disclosure References (Auto-Generated & Drift-Locked)

Read these companion reference files in [`references/`](./references/) whenever you need exhaustive field-by-field or catalog-level detail:

1. **[`references/config_reference.md`](./references/config_reference.md):** Every field, type, default, constraint, and cross-field rule across `RunConfig` and its 14 nested Pydantic blocks (`data`, `features`, `backtest`, `output`, `hpo`, `ensemble`, `hierarchy`, `compute`, `compute.families.<family>`, `compute.ensemble`, `compute.profile`, `compute.capacity`), GPU-to-VM shape resolution, and `run_id` digest exclusions.
2. **[`references/catalog_reference.md`](./references/catalog_reference.md):** All **37 models** (with family, runtime, package, `[extra]`, covariate support, explainability, training modes, and GPU capability), **21 evaluation metrics** (16 point + 5 interval), **7 compute runtimes**, **13 dependency extras**, **4 source tables / 5 registry tables / 5 analytical SQL views**, **62 shipped configs** (`20` root demos + `42` smoke configs `01`–`42`), and **Auto-Forecaster exposure tiers, `n_series` scale regimes, all 20 synthetic DGP benchmarks, and 9 public open-data benchmarks**.
3. **[`references/execution_paths_and_ops.md`](./references/execution_paths_and_ops.md):** Step-by-step CLI commands and Python SDK patterns for `SF_*` environment variables, **all 5 Execution Paths**, **Glass-Box Auto-Forecaster & Multi-Branch `autofocus`**, **7-Question Practitioner Assessment**, **Pre-Flight Verification** (`--dry-run`, `--feasibility`, `--quota`, `--stage-only`, `--emit-airflow`), **Post-Run Review & Surgical Repair** (`review_run`, `calibration_report`, `retry_run`, `--probe`, `--settle`, `--cancel`), and **Registry Lifecycle Operations** (`scale_forecasting.registry.ops`).

> **Built-in MCP Server Available:** If your agent client supports the Model Context Protocol (MCP), run `python -m scale_forecasting.mcp` (or `python -m scale_forecasting.mcp --allow-launch` to unlock cloud job submission and registry repair). It exposes 7 structured `forecast://*` resources and 13 tools (including `build_auto_config`, `run_auto_forecast`, `compute_time_series_similarity`, and `optimize_inventory_clearance`) directly over `stdio`.

---

## Mandatory 4-Step Agent Workflow

```mermaid
flowchart TD
    S0["Step 0: Probe Environment & Profile Data<br/>probe_environment() · Forecaster.profile_data()"] --> S1["Step 1: Compile via Auto-Forecaster or Author RunConfig<br/>Forecaster.build_config(tier=...) or manual RunConfig JSON"]
    S1 --> S2["Step 2: Pre-Flight & Execute (Single Pass or autofocus=True)<br/>5 Execution Paths · Forecaster.auto() · Dry-Run/Quota"]
    S2 --> S3["Step 3: 7-Question Practitioner Assessment, Review & Repair<br/>f.practitioner_assessment_df() · f.compare_models() · retry_run"]
```

### Step 0: Probe the Environment First (Never Guess Capabilities)

Before recommending an install command, model family, or execution path, inspect what is installed and configured in the active environment:

```bash
python -m scale_forecasting.agent_surfaces --probe-env
```

Or in Python / MCP (`probe_environment` tool or `forecast://environment` resource):
```python
from scale_forecasting.agent_surfaces import probe_environment

env = probe_environment()
```

Use the returned report to route your workflow:
- **`extras`** and **`models`**: Shows which of the 13 extras (`gcp`, `notebook`, `spark`, `ray`, `submit`, `models-stats`, `models-trees`, `models-prophet`, `models-dl`, `models-foundation`, `models-automl`, `models`, `all`) and which of the 37 models are importable right now, along with exact `pip install "scale-forecasting[<extra>]"` commands for any missing package.
  - Pure core (`pip install scale-forecasting`) includes 14 statistical and ML models (`croston`, `fft`, `holtwinters`, `kalman`, `naive_drift`, `naive_mean`, `naive_moving_average`, `naive_seasonal`, `random_forest`, `regression_lags`, `sarimax`, `stl_bagging`, `theta`, `ucm`) plus all 21 metrics, backtesting, ensembling, hierarchical reconciliation, and the MCP server — **zero model extras or GCP credentials required**.
- **`execution_paths_ready`**:
  - `path_1_offline_playground_and_dry_run` (`True` on every install): Use `scale_forecasting.playground` (`sample_data`, `run_model`, `bakeoff`, `summarize`) and `Forecaster.dry_run()` for instant zero-GCP benchmarking and config validation.
  - `path_5_airflow_dag_emission` (`True` on every install): Render Cloud Composer 3 / Airflow DAG Python source offline (`--emit-airflow`).
  - `path_2_cloud_plan_feasibility_and_stage`, `path_3_cloud_multi_family_dag_run`, and `path_4_direct_family_submitters`: True when `[gcp]` is installed and `SF_PROJECT_ID`, `SF_CONNECTION`, and `SF_WAREHOUSE_URI` are set (or loadable from `.env.infra`).

---

### Step 1: Compile via Glass-Box Auto-Forecaster OR Hand-Author `RunConfig`

You have two co-equal ways to produce a validated `RunConfig`:

#### Option 1A — Glass-Box Auto-Forecaster (`Forecaster.profile_data` & `Forecaster.build_config` / MCP `build_auto_config`)
When starting on a new DataFrame, BigQuery table, or any of the 20 synthetic DGP benchmarks (`dgp_01_smooth_seasonal` .. `dgp_20_monthly_regime_drift_damped`) or 9 public open-data benchmarks (`fetch_public_benchmark`), let the AlphaEvolve-calibrated compiler inspect the panel across 8 dimensions (`scale_regime`, SBC demand regime, seasonality/`log1p`, structural breaks/spikes, grid hygiene, covariates, hierarchy, similarity/clustering) and emit a transparent `ConfigBlueprint`:
```python
from scale_forecasting import Forecaster, load_benchmark

ds = load_benchmark("dgp_04_true_vs_decoy_covs", n_series=6)
profile_df = Forecaster.profile_data(ds.panel_df, horizon=14)  # Step 0 8-row readiness scan
bp = Forecaster.build_config(
    ds.panel_df,
    tier="wide_field",  # "first_light" | "wide_field" | "deep_sky" | "deep_field"
    where="auto",  # "auto" | "local" | "bigquery" | "cloud_cpu" | "cloud_all"
    goal="accuracy",  # "accuracy" | "probabilistic" | "service_level" | "fast"
    baselines="auto",  # 4-anchor ladder (naive, univariate stat, foundation, same-model univariate)
)
print(bp.explain())  # Plain-English rationale table + P50/P90 runtime & cost estimate
bp.to_file("/tmp/compiled_config.json")  # Standard editable RunConfig JSON
```
- Pin any user constraint (`models`, `include_models`, `exclude_models`, `allowed_families`, `decision_metric`, `n_folds`, `hpo`, `ensemble`, `holidays`, `baselines`, `max_minutes`, `max_usd`) — pinned rules are marked `[PINNED]` and respected across every pass. When `price` or `discount` covariates are present, the compiler automatically synthesizes downward-sloping `monotonic_constraints` (`{"price": -1}`) and `<model>@monotone` variants.

#### Option 1B — Hand-Author `RunConfig` Directly
Always include `"$schema": "https://statmike.github.io/scale-forecasting/schemas/run_config.schema.json"` at the top of generated JSON files (it provides IDE autocomplete and is stripped before `run_id` hashing).

#### Key Rules When Constructing a `RunConfig`
1. **Start from the closest shipped template:** Check the 20 root demo configs in `configs/*.json` and 42 smoke configs in `configs/smokes/*.json` (cataloged in [`references/catalog_reference.md`](./references/catalog_reference.md)).
2. **Match `models` to family and runtime rules:**
   - Every config block enforces `extra="forbid"` — unknown keys fail immediately with `ConfigError`.
   - Statistical models (`statistical` family, 18 models) and Tree/ML models (`ml` family, 5 models) run on Python runtimes (`spark`, `ray`, `vertex`, `gce`, `gke`).
   - Deep learning and open-weight foundation models (`deep_learning` family: `chronos2`, `chronos_bolt`, `neuralprophet`, `patchtst`, `tft`, `tide`, `timesfm_torch`, `tsmixer`) run on Python runtimes and support GPU acceleration (`hardware: "gpu"`).
   - Vertex AI AutoML models (`automl` family: `vertex_l2l`, `vertex_seq2seq`, `vertex_tft`, `vertex_tide`) require `compute.families.automl.runtime = "vertex_automl"`.
   - BigQuery-native models (`native` family: `arima_plus`, `timesfm` / alias `bq_timesfm`) execute as in-warehouse SQL on `bigquery`.
3. **Learned Ensembles, Cross-Run Blending & HPO Require Backtesting:**
   - `hpo.enabled = true` requires `backtest.enabled = true`.
   - If `ensemble.strategies` includes learned stackers (`"nnls"`, `"ridge"`, or `"xgb"`), you **must** set `"backtest": {"enabled": true, ...}` so out-of-fold (`backtest_oof`) predictions exist to train the combiner weights; otherwise `RunConfig` drops the learned strategies down to `"mean"` with a warning. Use `ensemble.from_run_ids` to blend models across completed historical runs without refitting.
4. **Covariate Declarations (`features`):**
   - Known-in-advance future drivers (promotions, planned prices) go in `features.future_covariates` (`list[str]`) and `data.future_covariates_table`.
   - Historical-only drivers (observed weather, foot traffic) go in `features.past_covariates` (`list[str]`).
   - Time-invariant series attributes go in `features.static_covariates` (`list[str]`). All three lists must be mutually disjoint.
5. **Composable `<model>@<tag>` Variants & 3-Mode Hierarchy Reconciliation:**
   - Any model in `models` can append `@` tags to race side-by-side variants in one run: scope (`@local`, `@global`, `@hybrid`), covariates (`@univariate`, `@cov_all`, `@cov_<col>`, `@cov_drop_<col>`), transforms (`@transform_log1p`, `@transform_boxcox`, `@transform_clr`, `@transform_ilr`), causal sign constraints (`@monotone`), cold-start NPI & lifecycle archetypes (`@cold_start`, `@analog_transfer`, `@bass_diffusion`, `@weibull_lifecycle`, `@arps_decline`), similarity clustering & donor transfer (`@similarity_global`, `@similarity_cold_start`), and online adaptive conformal intervals (`@aci`, `@aci_norm`, `@aci_normalized`).
   - Set `hierarchy.levels` for Mode 1 spatial reconciliation (`bottom_up`, `wls_struct`, `mint_shrink`, `mint_nonneg`), `hierarchy.reconcile_temporal = true` for Mode 2 `THieF` temporal reconciliation on flat panels, `hierarchy.cross_temporal = true` (with `"cross_temporal_mint"` in `hierarchy.methods` and `spatial_temporal_weight = "auto"`) for Mode 3 simultaneous cross-temporal coherence, or `hierarchy.similarity_hierarchy = true` to auto-discover latent cluster hierarchies on flat panels.
6. **Per-Family Compute Routing (`compute.families.<family>`):**
   - `scale-forecasting` dispatches **1 independent job per active model family in parallel** (`statistical`, `ml`, `deep_learning`, `automl`, `native`) so fast CPU families tear down immediately without waiting for GPU jobs.
   - Keep `machine_type: "auto"` unless pinning a specific shape from `resources/catalog.py`. Only `deep_learning` and `automl` families may set `hardware: "gpu"` or `gpu_type`.
   - On single-VM `gce`, `workers` must be `1`. On `vertex` and `gke` Indexed Jobs, `workers > 1` shards `ts_id` ranges across workers via BigQuery Storage Read API `row_restriction`.

#### Always Validate & Plan Offline Before Proceeding
```python
from scale_forecasting.config import load_config
from scale_forecasting.dag import gpu_usefulness_report, plan_dag
from scale_forecasting.registry.ids import make_run_id

cfg = load_config("configs/ensemble_demo.json")
run_id = make_run_id(cfg)
dag = plan_dag(cfg)
gpu_warnings = gpu_usefulness_report(cfg, dag.jobs)
```

---

### Step 2: Choose the Right Execution Path & Run Pre-Flight Checks

Select the execution path that matches the user's goal and `probe_environment()` readiness (full details in [`references/execution_paths_and_ops.md`](./references/execution_paths_and_ops.md)):

| Path | When to Use | Command / Entry Point |
| :--- | :--- | :--- |
| **Path 1: Offline Playground & Dry-Run** | Zero-GCP exploration, comparing models on synthetic or in-memory DataFrames, validating configs in seconds | `python -m scale_forecasting.playground --model holtwinters --backtest` or `from scale_forecasting.playground import sample_data, run_model, bakeoff` |
| **Path 2: Cloud Preflight & Staging** | Checking live BigQuery series lengths against backtest geometry, checking regional CPU/GPU quotas, or staging artifacts to GCS | `python -m scale_forecasting.main --config <path>.json [--dry-run --feasibility \| --quota \| --stage-only]` |
| **Path 3: Unified Multi-Family DAG (`main.run` / `Forecaster`)** | Running the full parallel multi-family DAG + ensemble node end-to-end from CLI, Python, or notebooks | `python -m scale_forecasting.main --config <path>.json` or `Forecaster.from_file("<path>.json").run()` |
| **Path 4: Direct Per-Family Submitters** | Submitting a single family or runtime (`submit`, `ray_submit`, `vertex_submit`, `gce_submit`, `gke_submit`, `automl_submit`, `ensemble_run`) | `python -m scale_forecasting.vertex_submit --config <path>.json --family ml` |
| **Path 5: Cloud Composer 3 / Airflow DAG Emission** | Rendering a self-contained Airflow DAG (`dag_<run_id>.py`) with per-family tasks, optional `--with-retry` repair node, or multi-branch `--auto --autofocus` DAG | `python -m scale_forecasting.main --config <path>.json --emit-airflow --with-retry` or `bp.emit_airflow_dag(autofocus=True)` |
| **Auto-Forecaster & Multi-Branch `autofocus`** | Automated multi-pass tournament (`Pass 1 -> Branches 2A/2B/2C -> Pass 3 Synthesis`) with `model_spec_hash` cell borrowing and Champion Ratchet across any surface | `Forecaster.auto(source, tier="first_light", autofocus=True)` or `python -m scale_forecasting.main --auto --tier first_light --autofocus` or MCP `run_auto_forecast` |
| **Opt-In AlphaEvolve Runtime Evolution (`Forecaster.evolve`)** | Evolving `# EVOLVE-BLOCK` Python code (`"features"`, `"ensemble"`, `"model"`, `"policy"`, `"similarity_kernel"`) on top of `Forecaster.auto()` under an AST sandbox, `+1e6` lookahead probe, and 3-Gate Anti-Overfitting Ratchet (`backend="alphaevolve"` default requiring Gemini Enterprise + `enable_alphaevolve = true`, or `backend="vertex_gemini"` backup on standard Vertex AI) | `Forecaster.evolve(source, target="features", backend="alphaevolve" \| "vertex_gemini")` or `python -m scale_forecasting.evolve --target features --backend alphaevolve` or MCP `run_auto_forecast(evolve_target="features")` |

**Mandatory Pre-Flight Habit Before Any Live Cloud Run:**
1. Run `--dry-run` (or `Forecaster.dry_run()` / MCP `plan_execution(mode="dry_run")`) to inspect the exact `LaunchPlan`, resolved machine shapes, `run_id`, and idempotency status in `run_registry`.
2. Run `--feasibility` and `--quota` when testing a new dataset or GPU family to catch series-length shortfalls or regional quota limits before spinning up clusters.

---

### Step 3: Post-Run Monitoring, Review, Calibration & Surgical Repair

Every cloud run persists structured lineage and outputs to 5 BigQuery tables (`run_registry`, `run_jobs`, `forecast_metadata`, `forecast_predictions`, `backtest_oof`) and 5 SQL views (`v_model_leaderboard`, `v_model_leaderboard_comparable`, `v_backtest_coverage`, `v_run_summary`, `v_run_jobs`).

1. **Verify Run Completion & Job Telemetry:**
   - Check `run_registry` (or `v_run_summary`) for `status = 'COMPLETED'` and `overhead_seconds`.
   - Inspect `run_jobs` (or `v_run_jobs`) for per-family wall-clock duration, `machine_type`, and `$.sizing` / `$.sizing_executed` JSON telemetry.
2. **Run the 7-Question Practitioner Assessment, Data-Science Review, Calibration & Supply-Chain Policy Translators:**
   ```python
   from scale_forecasting import Forecaster
   from scale_forecasting.review import calibration_report, review_run

   # For Forecaster.auto() sessions — answer the 7 recurring practitioner trust questions:
   # (1) model_vs_baseline, (2) ensemble_vs_single, (3) covariates_vs_overfitting,
   # (4) global_vs_local, (5) hierarchy_reconciliation, (6) horizon_stability, (7) forecast_trust
   # f_auto.practitioner_assessment_df() | f_auto.assess() | f_auto.compare_models()
   # f_auto.baseline_value_df() | f_auto.hierarchy_value_df() | f_auto.autofocus_ledger_df()

   review = review_run("ensemble-demo-8ca68173be52")
   cal = calibration_report("ensemble-demo-8ca68173be52")
   f = Forecaster.from_run_id("ensemble-demo-8ca68173be52")
   policy_df = f.inventory_policy_df(lead_time=7.0, service_level=0.95)
   ```

   | Workbench Surface | SDK / Config Entrypoint | Purpose |
   | :--- | :--- | :--- |
   | **Promo Elasticity & What-If Scenarios** | `f.promo_elasticity_df()`, `f.simulate_scenarios(scenarios)` | Double-ML causal price/promo decomposition, pull-forward dip, cannibalization, and counterfactual lift. |
   | **Temporal & Cross-Temporal (`THieF`) + Simplex** | `f.temporal_reconcile_df(method="cross_temporal_mint")`, `"<model>@transform_clr"`, `"<model>@transform_ilr"` | Multi-frequency temporal/cross-temporal coherence and compositional simplex closure (`sum = S`). |
   | **Top-Down Goal-Seek QP** | `f.allocate_target(target_total, locked_nodes=..., smoothness_weight=...)` | Variance-weighted QP target disaggregation with hard locks/bounds (`GoalSeekInfeasibleError`). |
   | **Cold-Start, Similarity & Online ACI** | `f.cold_start_transfer_df()`, `f.similarity_df()`, `f.find_similar_series_df()`, `f.cluster_by_similarity_df()`, `f.recommend_substitutes_df()`, `f.emerging_trends_df()`, `f.aci_df()`, `"<model>@cold_start"`, `"<model>@similarity_global"`, `"<model>@aci_norm"` | Empirical-Bayes analog/Bass/Weibull NPI transfer, 7-lens multi-modal similarity (`hybrid`, `physics_attractor`, `spectral_phase`, `elastic_dtw`, `minirocket`, `structural_signature`, `foundation_latent`), SKU substitutes & emerging trend breakouts, and Gibbs-Candès online adaptive conformal intervals. |
   | **Joint Copula Risk & Optimizers** | `f.sample_trajectories()`, `f.tail_risk_df()`, `f.optimize_inventory_df()`, `f.optimize_clearance_df()`, `f.optimize_contract_volume_df()`, `f.optimize_scenario()` | Gaussian/Student-$t$ trajectories, portfolio VaR/CVaR, KKT+MILP replenishment, lifecycle markdown & space-reclamation MILP (with FWL covariate partialling, aggregate-node filtering & `hierarchy_rollup_df`), vendor rebate/Take-or-Pay volume optimization, and MILP+SLSQP price/promo calendars. |

   Or via MCP: `review_run(run_id="...", include_inventory_policy=True, inventory_params={"lead_time": 7.0, "service_level": 0.95}, include_cold_start=True, include_aci=True, include_tail_risk=True, include_similarity=True, include_substitutes=True, include_emerging_trends=True, include_clearance=True, include_contract_volume=True)`. Returns `best_overall`, `best_per_family`, `ensemble_lift`, full model leaderboard (`score`, `pooled_wape`, `n_comparable_series`, `cohort`), point-forecast arm win-rate (`arms`), empirical vs. nominal interval coverage (`coverage`, `mean_coverage`, `worst_step`), and optional Forecaster Workbench diagnostics (`fva`, `horizon_decay`, `bias_report`, `exceptions`, `inventory_policy`, `scenarios`, `cold_start`, `aci`, `trajectories`, `tail_risk`, `inventory_optimization`, `scenario_optimization`, `clearance_optimization`, `contract_volume_optimization`, `similarity`, `substitutes`, `emerging_trends`).
3. **Surgical Repair of Failed or Partial Runs (Never Re-Run Everything Blindly):**
   - **Preview or submit surgical cell repair (`retry_run`):** Re-runs only missing or failed `(ts_id, model_type)` cells and re-runs the ensemble barrier if any cell was repaired:
     ```bash
     python -m scale_forecasting.main --run-id <run_id> --retry
     python -m scale_forecasting.main --run-id <run_id> --retry --force --reason "Repair transient worker OOM"
     ```
   - **Inspect, reconcile, or clean up registry state (`registry.ops`):**
     ```bash
     python -m scale_forecasting.registry.ops doctor
     python -m scale_forecasting.registry.ops close-runs --yes
     python -m scale_forecasting.registry.ops drop-run <run_id> --yes
     python -m scale_forecasting.registry.ops sweep-orphans --yes
     ```

