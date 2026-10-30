# Visual system walkthrough — progressive architecture

This visual atlas unpacks **`scale-forecasting`** from the **10,000-foot system map (`Level 0`)** down through **5 enterprise personas** and **8 progressive zoom levels**, using a single **6-color semantic key** that maps every box to source modules, guides, notebooks, and tests.

## Level 0 — The 10,000-foot system map

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
        VIEWS["11 Analytical SQL Views<br/>Leaderboards · FVA · Exceptions<br/>Anomalies · S&OP Scorecard"]
        USE["Practitioner Trust & Consumption<br/>BigQuery DataAgent · Knowledge Catalog<br/>7-Question Assessment · SDK · Looker"]
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

### Visual 6-color key (shared across every diagram)

```mermaid
flowchart LR
    C_CFG["Config, Infra & Launch (config)<br/>Soft Amber · 12 Terraform Modules<br/>RunConfig · autofocus · evolve() · 5 Paths"]
    C_RTE["Routing, Sizing & Lifecycle (route)<br/>Warm Coral · Family DAG · --quota<br/>src.zip · retry_run · CI Tripwires"]
    C_CMP["Zero-Idle Compute & Science (compute)<br/>Mint Green · 7 Cloud Runtimes<br/>37 Models · 21 Metrics · 8 Ensembles"]
    C_DATA["BigQuery Data & Views (data)<br/>Light Blue · Source Panels<br/>5 Registry Tables · 11 Always-On Views"]
    C_SEM["Knowledge Catalog & Agents (semantic)<br/>Soft Purple · 44 Glossary Terms<br/>DataProfile · 32 Golden SQL · DataAgent · MCP"]
    C_DEC["Workbench & Optimization (decision)<br/>Deep Teal · 7-Q assess() · FVA · Double-ML<br/>Similarity · Copula · 7 LP/MILP Solvers"]

    C_CFG --> C_RTE --> C_CMP --> C_DATA --> C_SEM & C_DEC

    classDef data fill:#E8F0FE,stroke:#4285F4,stroke-width:2px,color:#174EA6
    classDef config fill:#FEF7E0,stroke:#F9AB00,stroke-width:2px,color:#7A5A00
    classDef route fill:#FCE8E6,stroke:#EA4335,stroke-width:2px,color:#A50E0E
    classDef compute fill:#E6F4EA,stroke:#34A853,stroke-width:2px,color:#0D652D
    classDef semantic fill:#F3E8FD,stroke:#9334E6,stroke-width:2px,color:#5B1799
    classDef decision fill:#E4F7FB,stroke:#12B5CB,stroke-width:2px,color:#005A66
    class C_CFG config
    class C_RTE route
    class C_CMP compute
    class C_DATA data
    class C_SEM semantic
    class C_DEC decision
```

**Jump to zoom level:** [Persona Map](#persona-map--5-enterprise-personas--entry-surfaces) · [Level 1: Cloud Infra & `src.zip`](#level-1--stage-0-two-stage-cloud-deployment-100k-seeding--zero-rebuild-code-delivery) · [Level 2: `RunConfig`, Auto & Evolve](#level-2--stage-1a-authoring-auto-compiling-ablating--evolving-a-runconfig) · [Level 3: 5 Execution Paths](#level-3--stage-1b-the-5-execution-paths-preflight--durationcost-predictor) · [Level 4: 7 Cloud Runtimes](#level-4--stage-2a-parallel-per-family-routing-sizing-gpu-audit--7-cloud-runtimes) · [Level 5: Models, Covariates, CV & Metrics (`5A`–`5E`)](#level-5--stage-2b-forecasting-science-inside-the-worker-5a5e) · [Level 6: Registry & Knowledge Catalog](#level-6--stage-3a-self-documenting-registry-erdiagram-11-always-on-views--knowledge-catalog) · [Level 7: Workbench & Solvers](#level-7--stage-3b-the-forecaster-workbench--6-pillars--prescriptive-solvers) · [Level 8: Agents, Repair & CI](#level-8--stage-3c-ai-agent-ecosystem-5-chat-surfaces-day-2-repair--3-tier-verification) · [Appendix: Index & Auto-Synced Inventory](#appendix--zoom-index--auto-synced-repository-inventory)

---

## Persona Map — 5 enterprise personas & entry surfaces

Different practitioners enter `scale-forecasting` at different layers while sharing the **exact same `RunConfig`, `run_id`, and BigQuery registry**:

```mermaid
flowchart LR
    subgraph P_ROLES["5 Enterprise Personas"]
        direction TB
        P_DS["1. Data Scientist & Quant Forecaster<br/>Model selection · HPO · custom plugins · ablations"]
        P_DP["2. Demand Planner & Merchandiser<br/>Exceptions · promos · substitutes · inventory · markdowns"]
        P_EX["3. Executive & S&OP Leader<br/>Portfolio accuracy · FVA lift · targets · natural-language Q&A"]
        P_CA["4. Cloud Architect & FinOps Lead<br/>VPC/IAM · BigLake Iceberg · runtime sizing · GPU waste audit"]
        P_ML["5. MLOps Engineer & AI Coding Agent<br/>Airflow DAGs · live probes · surgical repair · MCP & SKILL.md"]
    end

    subgraph P_DOORS["Primary Platform Entry Surfaces"]
        direction TB
        D_MOD["Offline Playground, Auto-Forecaster & Plugins<br/>playground · Forecaster.auto(autofocus=True) · Forecaster.evolve()<br/>BaseModel / BaseMetric · @tag Variant Matrix"]
        D_WB["6-Pillar Forecaster Workbench (workbench.py)<br/>FVA · Double-ML Elasticity · 5-Stage Scenarios · Similarity<br/>Copula Tail Risk · Goal-Seek QP · 7 LP/MILP Solvers"]
        D_SOP["Governed S&OP Read & Chat Layer<br/>11 Always-On SQL Views (v_executive_sop_summary)<br/>BigQuery Studio & SDK DataAgent · 12-Tab HTML Report"]
        D_INF["Declarative Infra, Sizing & Cost Controls<br/>Two-Stage Terraform (12 Modules) · --quota · predict_run_duration<br/>7 Zero-Idle Runtimes · v_run_jobs GPU device_verdict"]
        D_OPS["Production Orchestration, Day-2 Ops & Agent Protocol<br/>--emit-airflow · --stage-only · Zero-Rebuild src.zip<br/>doctor · probes · retry_run · 14-Tool MCP Server"]
    end

    P_DS --> D_MOD & D_WB
    P_DP --> D_WB & D_SOP
    P_EX --> D_SOP & D_WB
    P_CA --> D_INF & D_OPS
    P_ML --> D_OPS & D_INF

    classDef data fill:#E8F0FE,stroke:#4285F4,stroke-width:2px,color:#174EA6
    classDef config fill:#FEF7E0,stroke:#F9AB00,stroke-width:2px,color:#7A5A00
    classDef route fill:#FCE8E6,stroke:#EA4335,stroke-width:2px,color:#A50E0E
    classDef compute fill:#E6F4EA,stroke:#34A853,stroke-width:2px,color:#0D652D
    classDef semantic fill:#F3E8FD,stroke:#9334E6,stroke-width:2px,color:#5B1799
    classDef decision fill:#E4F7FB,stroke:#12B5CB,stroke-width:2px,color:#005A66
    class P_DS,D_MOD compute
    class P_DP,D_WB decision
    class P_EX,D_SOP semantic
    class P_CA,D_INF config
    class P_ML,D_OPS route
```

| Persona | Key Entrypoints (SDK / CLI / SQL) | Recommended Notebooks | Zoom Levels to Read First |
| :--- | :--- | :--- | :--- |
| **1. Data Scientist & Quant Forecaster** | `playground.bakeoff()`, `Forecaster.auto(..., autofocus=True)`, `Forecaster.evolve()`, `variant_matrix()`, `@register` plugins | [`00`](./notebooks/00_model_playground.ipynb), [`05`](./notebooks/05_covariates_and_global_models.ipynb), [`07`](./notebooks/07_hpo_backtesting_and_ensembles.ipynb), [`09`](./notebooks/09_custom_models_and_metrics.ipynb), [`11`](./notebooks/11_auto_forecaster_and_autofocus.ipynb) | [Level 2](#level-2--stage-1a-authoring-auto-compiling-ablating--evolving-a-runconfig), [Level 5](#level-5--stage-2b-forecasting-science-inside-the-worker-5a5e), [Level 7](#level-7--stage-3b-the-forecaster-workbench--6-pillars--prescriptive-solvers) |
| **2. Demand Planner & Merchandiser** | `exceptions_df()`, `promo_elasticity_df()`, `simulate_scenarios()`, `recommend_substitutes_df()`, `emerging_trends_df()`, `optimize_inventory_df()`, `optimize_clearance_df()`, `optimize_assortment_df()` | [`05`](./notebooks/05_covariates_and_global_models.ipynb), [`06`](./notebooks/06_hierarchical_reconciliation.ipynb), [`08`](./notebooks/08_multi_engine_master_workflow.ipynb), [`12`](./notebooks/12_conversational_analytics_and_knowledge_catalog.ipynb) | [Level 6](#level-6--stage-3a-self-documenting-registry-erdiagram-11-always-on-views--knowledge-catalog), [Level 7](#level-7--stage-3b-the-forecaster-workbench--6-pillars--prescriptive-solvers) |
| **3. Executive & S&OP Leader** | `v_executive_sop_summary`, `Registry.ask()` / BigQuery Studio *Agents* tab, `f.assess()`, `allocate_target()`, `export_report("briefing.html")` | [`11`](./notebooks/11_auto_forecaster_and_autofocus.ipynb), [`12`](./notebooks/12_conversational_analytics_and_knowledge_catalog.ipynb), [workshop.md](./workshop.md) | [Level 0](#level-0--the-10000-foot-system-map), [Level 6](#level-6--stage-3a-self-documenting-registry-erdiagram-11-always-on-views--knowledge-catalog), [Level 7](#level-7--stage-3b-the-forecaster-workbench--6-pillars--prescriptive-solvers) |
| **4. Cloud Architect & FinOps Lead** | `terraform/bootstrap` & `terraform/main`, `--quota`, `predict_run_duration()`, `v_run_summary`, `v_run_jobs.device_verdict` | [`01`](./notebooks/01_bigquery_native_sql.ipynb), [`02`](./notebooks/02_vertex_and_gce_vms.ipynb), [`03`](./notebooks/03_spark_serverless_and_connect.ipynb), [`04`](./notebooks/04_ray_on_vertex_gpu.ipynb) | [Level 1](#level-1--stage-0-two-stage-cloud-deployment-100k-seeding--zero-rebuild-code-delivery), [Level 4](#level-4--stage-2a-parallel-per-family-routing-sizing-gpu-audit--7-cloud-runtimes) |
| **5. MLOps Engineer & AI Coding Agent** | `--emit-airflow`, `--stage-only`, `Registry.doctor()`, `Forecaster.retry()`, `python -m scale_forecasting.mcp`, `SKILL.md` | [`08`](./notebooks/08_multi_engine_master_workflow.ipynb), [`10`](./notebooks/10_registry_operations_and_scale.ipynb), [`12`](./notebooks/12_conversational_analytics_and_knowledge_catalog.ipynb) | [Level 1](#level-1--stage-0-two-stage-cloud-deployment-100k-seeding--zero-rebuild-code-delivery), [Level 3](#level-3--stage-1b-the-5-execution-paths-preflight--durationcost-predictor), [Level 8](#level-8--stage-3c-ai-agent-ecosystem-5-chat-surfaces-day-2-repair--3-tier-verification) |

---

## Level 1 — Stage 0: Two-Stage Cloud Deployment, 100k Seeding & Zero-Rebuild Code Delivery

Before the first cloud forecast runs, `scale-forecasting` separates **immutable dependency substrates** from **instant-iteration Python source code**:

1. **Pure-Core Packaging (`13` Extras in `pyproject.toml`):** Base `pip install scale-forecasting` installs **zero Google Cloud SDKs**; `[gcp]` is the single home of every Google client, composed cleanly into `spark`, `ray`, `submit`, and the 6 `models-*` extras.
2. **Two-Stage Terraform (`terraform/bootstrap` + `12` Modules in `terraform/main`):** Provisions remote state, APIs, private VPC/PSC-I networking, least-privilege service accounts, GCS buckets, BigQuery + BigLake Apache Iceberg tables, Artifact Registry container + packed worker `venv`, Colab Enterprise `sf-main` runtime template, budget alerts, and an automated **100,000-series seed batch** (`seed_spark.py` across 5 archetypes).
3. **Zero-Rebuild Code Delivery (`code_delivery.py`):** `src/scale_forecasting/` is **never baked into the Docker image** (`test_code_delivery.py`). Submitters zip `src/` and upload it per run, so local edits or new 1-file model plugins execute on the very next cloud job in seconds without rebuilding containers.

```mermaid
flowchart LR
    subgraph PKG["1. Modular Packaging (13 Packaging Extras)"]
        direction TB
        CORE_PKG["Pure Offline Core (pyproject.toml)<br/>0 GCP SDKs · Playground · Workbench · MCP"]
        EXT_PKG["Composable Cloud & Model Extras<br/>[gcp] · [spark] · [ray] · [notebook]<br/>[models-stats/trees/prophet/dl/foundation/automl] · [all]"]
        CORE_PKG --> EXT_PKG
    end

    subgraph TF["2. Two-Stage Terraform (bootstrap + 12 Terraform Modules)"]
        direction TB
        TF1["Stage 1: terraform/bootstrap<br/>Remote GCS State Bucket"]
        TF2["Stage 2 Core: apis · network (VPC/PSC-I) · iam<br/>storage · bigquery (Native + Iceberg) · budget · colab (sf-main)"]
        TF3["Stage 2 Compute & Opt-In: container (Cloud Build + Venv)<br/>seed (100k Series) · opt-in composer · gke · smoke"]
        TF1 --> TF2 --> TF3
    end

    subgraph SHIP["3. Seeded Lakehouse & Zero-Rebuild Shipping"]
        direction TB
        SEED["Deterministic Scale Seeder (data_gen/seed_spark.py)<br/>5 Archetypes · 100,000 Series -> Native & BigLake Iceberg"]
        ZIP["Dynamic Code Delivery (code_delivery.py)<br/>Docker Holds Locked Deps Only · Fresh src.zip Shipped per Run"]
    end

    EXT_PKG --> TF3
    TF2 & TF3 --> SEED & ZIP

    classDef data fill:#E8F0FE,stroke:#4285F4,stroke-width:2px,color:#174EA6
    classDef config fill:#FEF7E0,stroke:#F9AB00,stroke-width:2px,color:#7A5A00
    classDef route fill:#FCE8E6,stroke:#EA4335,stroke-width:2px,color:#A50E0E
    class CORE_PKG,EXT_PKG,TF1,TF2,TF3 config
    class SEED data
    class ZIP route
```

| Diagram Block | Source Modules | Guides & Notebooks | Tests & Smokes |
| :--- | :--- | :--- | :--- |
| **Pure Core & 13 Packaging Extras** | [`pyproject.toml`](https://github.com/statmike/scale-forecasting/blob/main/pyproject.toml), [`errors.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/errors.py), [`settings.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/settings.py) | [runtime_dependencies.md](./runtime_dependencies.md), [version_matrix.md](./version_matrix.md), [notebook_runtimes.md](./notebook_runtimes.md) | [`test_packaging_extras.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_packaging_extras.py), [`test_core_install.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_core_install.py), [`test_launch_point_lean.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_launch_point_lean.py) |
| **Two-Stage Terraform (12 Modules)** | [`terraform/bootstrap/`](https://github.com/statmike/scale-forecasting/tree/main/terraform/bootstrap), [`terraform/main/modules/`](https://github.com/statmike/scale-forecasting/tree/main/terraform/main/modules), [`docker/Dockerfile`](https://github.com/statmike/scale-forecasting/blob/main/docker/Dockerfile) | [deploying_on_gcp.md](./deploying_on_gcp.md), [cost_estimates.md](./cost_estimates.md), [why_google_cloud.md](./why_google_cloud.md) | [`13_native_format.json`](https://github.com/statmike/scale-forecasting/blob/main/configs/smokes/13_native_format.json), [`explode_100k.json`](https://github.com/statmike/scale-forecasting/blob/main/configs/explode_100k.json), [`ray_100k.json`](https://github.com/statmike/scale-forecasting/blob/main/configs/ray_100k.json) |
| **100k Deterministic Data Generator** | [`data_gen/generator.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/data_gen/generator.py), [`data_gen/seed_spark.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/data_gen/seed_spark.py) | [api/data_gen.md](./api/data_gen.md), [reading_source_data.md](./reading_source_data.md) | [`test_generator.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_generator.py), [`test_seed_transform.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_seed_transform.py) |
| **Zero-Rebuild `src.zip` Delivery** | [`code_delivery.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/code_delivery.py), [`cluster_deps.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/cluster_deps.py) | [editing_code_without_rebuilding.md](./editing_code_without_rebuilding.md), [Notebook 09](./notebooks/09_custom_models_and_metrics.ipynb) | [`test_code_delivery.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_code_delivery.py) |

---

## Level 2 — Stage 1A: Authoring, Auto-Compiling, Ablating & Evolving a `RunConfig`

Zooming into **`Create a Config File (JSON)`**: whether hand-authored across the **12 root `RunConfig` fields**, compiled automatically by the **Glass-Box Auto-Forecaster** (`Forecaster.build_config()` / `Forecaster.auto()` — calibrated offline with AlphaEvolve across `20` synthetic DGPs and `9` public benchmarks, running with **0 runtime LLM calls**), or mutated via opt-in **`Forecaster.evolve()`**, every path produces the **exact same validated `RunConfig`**, expands `<base_model>@<tag>` ablation variants (`variant_matrix()`), and hashes to a **deterministic `<slug>-<12hex>` `run_id`**.

```mermaid
flowchart TD
    SRC["Source Panel & Optional Covariates<br/>BigQuery Native · BigLake Iceberg · pandas DataFrame · 29 Built-In Benchmarks"]

    subgraph A["Path A · Hand-Crafted JSON & @tag Ablations"]
        direction TB
        SCHEMA["IDE JSON Schema Autocomplete<br/>run_config.schema.json ($schema)"]
        MANUAL["12 Root RunConfig Blocks + Variant Expansion<br/><model>@<tag> · ablation_modes · include_univariate_baselines · variant_matrix()"]
        SCHEMA --> MANUAL
    end

    subgraph B["Path B · Glass-Box Auto-Forecaster (0 Runtime LLM Calls)"]
        direction TB
        PROF["Step 0 Data Profiler (profile_data)<br/>8 Dimensions: scale (5 Regimes) · SBC zeros · seasonality · gaps · simplex · hierarchy"]
        TIER["Compiler (build_config · 4 Exposure Tiers)<br/>first_light · wide_field · deep_sky · deep_field + [PINNED] Scope Dials"]
        AF["Multi-Branch autofocus DAG + Snapshot Cell Borrowing (reuse_prior_runs)<br/>Pass 1 (4-Anchor Ladder) -> Pass 2A / 2B / 2C -> Step-by-Step or Champion Ratchet"]
        PROF --> TIER --> AF
    end

    subgraph C["Path C · Opt-In Evolutionary Harness (Forecaster.evolve)"]
        direction TB
        EV["AlphaEvolve or Vertex Gemini Code Mutator<br/>5 # EVOLVE-BLOCK Targets: features · ensemble · model · policy · similarity_kernel"]
        GUARD["AST Security Sandbox + +1e6 Future-Leakage Sentinel<br/>+ 3-Gate Lockdown Fold Anti-Overfitting Ratchet"]
        EV --> GUARD
    end

    VALID["Frozen Validated RunConfig (config.py)<br/>extra='forbid' · Logged Verbatim to run_registry.raw_config"]
    RUNID["Deterministic Content-Addressed run_id (registry/ids.py)<br/>slug-12hex · Strips Plumbing (_RUN_ID_EXCLUDED) · model_spec_hash per Variant"]

    SRC --> MANUAL & PROF & EV
    MANUAL & AF & GUARD --> VALID --> RUNID

    classDef data fill:#E8F0FE,stroke:#4285F4,stroke-width:2px,color:#174EA6
    classDef config fill:#FEF7E0,stroke:#F9AB00,stroke-width:2px,color:#7A5A00
    classDef route fill:#FCE8E6,stroke:#EA4335,stroke-width:2px,color:#A50E0E
    class SRC data
    class SCHEMA,MANUAL,PROF,TIER,AF,EV,VALID config
    class GUARD,RUNID route
```

| Diagram Block | Source Modules | Guides & Notebooks | Tests & Smokes |
| :--- | :--- | :--- | :--- |
| **12-Block `RunConfig`, Schema & `@tag` Ablations** | [`config.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/config.py), [`agent_surfaces.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/agent_surfaces.py) | [configuration_reference.md](./configuration_reference.md), [forecaster_workbench.md](./forecaster_workbench.md) | [`test_config.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_config.py), [`test_config_coverage.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_config_coverage.py), [`test_phase_g5_variants_and_borrowing.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_phase_g5_variants_and_borrowing.py) |
| **Auto-Forecaster, `autofocus` & 29 Benchmarks** | [`auto.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/auto.py), [`auto_policy.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/auto_policy.py), [`benchmarks.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/benchmarks.py) | [auto_forecaster.md](./auto_forecaster.md), [auto_forecaster_benchmarks.md](./auto_forecaster_benchmarks.md), [auto_forecaster_cloud_scale.md](./auto_forecaster_cloud_scale.md), [Notebook 11](./notebooks/11_auto_forecaster_and_autofocus.ipynb) | [`test_auto_forecaster.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_auto_forecaster.py) |
| **Opt-In Evolutionary Harness (`evolve`)** | [`evolve/`](https://github.com/statmike/scale-forecasting/tree/main/src/scale_forecasting/evolve) | [alphaevolve_runner.md](./alphaevolve_runner.md), [api/evolve.md](./api/evolve.md), [Notebook 11](./notebooks/11_auto_forecaster_and_autofocus.ipynb) | [`test_evolve.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_evolve.py) |
| **Deterministic `run_id` & Cell Borrowing** | [`registry/ids.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/registry/ids.py), [`registry/harvest.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/registry/harvest.py) | [architecture.md](./architecture.md), [faq.md](./faq.md) | [`test_ids.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_ids.py), [`test_prebreak_snapshots.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_prebreak_snapshots.py) |

---

## Level 3 — Stage 1B: The 5 Execution Paths, Preflight & Duration/Cost Predictor

Zooming into **`Launch & Orchestrate`**: one `RunConfig` executes identically across **5 launch paths** with zero code forks — backed by offline wall-clock/cost estimation (`predict_run_duration`), read-only cloud `--feasibility` and `--quota` checks, and staged native CLI emission (`--stage-only`).

```mermaid
flowchart LR
    CFG["Validated RunConfig<br/>+ Deterministic run_id"]

    subgraph P1["Path 1 · Pure-Offline Sandbox"]
        OFF["playground.run_one / bakeoff<br/>Forecaster.dry_run() · predict_run_duration()<br/>0 GCP Imports · In-Memory DataFrames"]
    end

    subgraph P2["Path 2 · Cloud Preflight & Staged CLI"]
        PRE["--feasibility (Fold vs Series Lengths)<br/>--quota (Live Regional vCPU & GPU)<br/>--stage-only (Uploads GCS + Prints CLI)"]
    end

    subgraph P3["Path 3 · Unified Multi-Family DAG"]
        MAIN["main.run / Forecaster.run() & run_live()<br/>Colab Notebooks · Cloud Batch<br/>MCP Server (--allow-launch)"]
    end

    subgraph P4["Path 4 · Direct Family Submitters"]
        SUBS["submit · ray_submit · vertex_submit<br/>gce_submit · gke_submit · automl_submit<br/>Contributor or Owner Header Mode"]
    end

    subgraph P5["Path 5 · Cloud Composer 3 (Airflow)"]
        AIR["--emit-airflow (airflow_emit.py)<br/>open_run_header -> family tasks<br/>-> ensemble_run -> finalize_run"]
    end

    JL["Shared Execution Spine (job_launch.py + lifecycle.py)<br/>1 run_registry Header · Pinned snapshot_millis · 1 run_jobs Row per Family"]
    LOCAL_OUT["In-Memory BakeOff / ModelResult<br/>69 Workbench Methods & Offline BakeOff.ask()"]

    CFG --> OFF & PRE & MAIN & SUBS & AIR
    OFF --> LOCAL_OUT
    PRE -.->|"Operator / CI runs emitted commands"| SUBS
    MAIN & SUBS & AIR --> JL

    classDef config fill:#FEF7E0,stroke:#F9AB00,stroke-width:2px,color:#7A5A00
    classDef route fill:#FCE8E6,stroke:#EA4335,stroke-width:2px,color:#A50E0E
    classDef compute fill:#E6F4EA,stroke:#34A853,stroke-width:2px,color:#0D652D
    classDef decision fill:#E4F7FB,stroke:#12B5CB,stroke-width:2px,color:#005A66
    class CFG,MAIN,SUBS,AIR config
    class PRE,JL route
    class OFF compute
    class LOCAL_OUT decision
```

| Diagram Block | Source Modules | Guides & Notebooks | Tests & Smokes |
| :--- | :--- | :--- | :--- |
| **Path 1 — Pure-Offline Sandbox & Duration Predictor** | [`playground.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/playground.py), [`launch_plan.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/launch_plan.py), [`profiling/duration.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/profiling/duration.py) | [getting_started.md](./getting_started.md), [cost_estimates.md](./cost_estimates.md), [Notebook 00](./notebooks/00_model_playground.ipynb) | [`test_playground.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_playground.py), [`test_core_install.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_core_install.py), [`test_launch_plan.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_launch_plan.py) |
| **Path 2 — Feasibility, Quota & `--stage-only`** | [`validation.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/validation.py), [`quota.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/quota.py), [`staging.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/staging.py), [`commands.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/commands.py) | [quota_and_scale.md](./quota_and_scale.md), [running_and_reviewing.md](./running_and_reviewing.md), [Notebook 10](./notebooks/10_registry_operations_and_scale.ipynb) | [`test_validation.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_validation.py), [`test_quota.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_quota.py), [`test_commands.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_commands.py) |
| **Path 3 — Unified Orchestrator, `run_live()` & MCP** | [`main.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/main.py), [`sdk.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/sdk.py), [`mcp.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/mcp.py), [`job_launch.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/job_launch.py) | [using_the_sdk.md](./using_the_sdk.md), [Notebook 08](./notebooks/08_multi_engine_master_workflow.ipynb) | [`test_main.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_main.py), [`test_job_launch.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_job_launch.py), [`14_full_dag.json`](https://github.com/statmike/scale-forecasting/blob/main/configs/smokes/14_full_dag.json) |
| **Path 4 — Direct Per-Family Submitters** | [`submitters.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/submitters.py), [`submit.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/submit.py), [`ray_submit.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/ray_submit.py), [`vertex_submit.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/vertex_submit.py), [`gce_submit.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/gce_submit.py), [`gke_submit.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/gke_submit.py), [`automl_submit.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/automl_submit.py) | [runtimes_reference.md](./runtimes_reference.md) | [`test_submitters.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_submitters.py), [`test_submit.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_submit.py), [`test_ray_submit.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_ray_submit.py), [`test_gke_submit.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_gke_submit.py) |
| **Path 5 — Cloud Composer 3 / Airflow DAG** | [`airflow_emit.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/airflow_emit.py), [`airflow_tasks.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/airflow_tasks.py) | [deploying_on_gcp.md](./deploying_on_gcp.md), [operations.md](./operations.md), [Notebook 10](./notebooks/10_registry_operations_and_scale.ipynb) | [`test_airflow_emit.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_airflow_emit.py), [`test_airflow_dagbag.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_airflow_dagbag.py), [`15_airflow_multi_engine.json`](https://github.com/statmike/scale-forecasting/blob/main/configs/smokes/15_airflow_multi_engine.json) |

---

## Level 4 — Stage 2A: Parallel Per-Family Routing, Sizing, GPU Audit & 7 Cloud Runtimes

Zooming into **`Parallel Family DAG Router -> 2. Zero-Idle Compute on Google Cloud`**: `dag.plan_dag` fans active families out concurrently (`statistical`, `ml`, `deep_learning`, `automl`, `native`) so fast CPU batches tear down immediately without idling behind GPU trainers. `resources/` sizes pools across `CPU`, `T4`, `L4`, `A100`, and `A100_80GB`; `capacity.py` + `compute_fallback.py` walk regions on stockouts; workers read `snapshot_millis`-pinned Arrow slices directly via Storage Read API `row_restriction` (`order_chunks_lpt`); and `device_audit.py` stamps `device_verdict` (`ENGAGED_UTILISED` vs. unused GPU warning) onto `run_jobs`.

```mermaid
flowchart TD
    SRC_PIN["BigQuery Source Panel (Pinned at snapshot_millis)<br/>Zero Data Replication · Storage Read API Arrow Streams + Contiguous [start_id, end_id] Pushdown"]

    DAG["Parallel Family DAG Planner, Sizer & Regional Capacity Router<br/>dag.plan_dag · resources/ (ResourceSlot & UnitShape) · shared_clusters.py · capacity.py · compute_fallback.py"]

    subgraph FAMS["5 Independent Parallel Model Families (37 Models Total)"]
        direction LR
        F_STAT["statistical (18)<br/>ARIMA · ETS · Theta · Croston · MSTL..."]
        F_ML["ml (5)<br/>LightGBM · XGBoost · CatBoost · Ridge..."]
        F_DL["deep_learning (8)<br/>TiDE · TFT · PatchTST · Chronos · TimesFM..."]
        F_AML["automl (4)<br/>vertex_l2l · vertex_tide · vertex_tft · vertex_seq2seq"]
        F_NAT["native (2)<br/>arima_plus · timesfm (AI.FORECAST)"]
    end

    subgraph RTS["7 Zero-Idle Google Cloud Compute Runtimes (Chosen per Family)"]
        direction LR
        RT_SP["1. spark<br/>Serverless Batch · GCE Cluster · Connect"]
        RT_RY["2. ray<br/>Vertex AI Ray · Ray on GKE (Fractional GPU)"]
        RT_VX["3. vertex<br/>Vertex AI CustomJob (1..N Worker Pool VMs)"]
        RT_GC["4. gce<br/>Single COS VM (3x Zero-Orphan Self-Delete)"]
        RT_GK["5. gke<br/>batch/v1 Indexed Job or Ray · Scale-to-Zero"]
        RT_AM["6. vertex_automl<br/>Tabular Workflow KFP or AutoML TrainingJob"]
        RT_BQ["7. bigquery<br/>Pure SQL BQML + AI.FORECAST"]
    end

    AUDIT["Telemetry & GPU Utilization Audit (device_audit.py + *_telemetry.py)<br/>Stamps sizing, provision/execute split & GPU device_verdict onto run_registry / run_jobs"]

    DAG --> F_STAT & F_ML & F_DL & F_AML & F_NAT
    F_STAT & F_ML & F_DL --> RT_SP & RT_RY & RT_VX & RT_GC & RT_GK
    F_AML --> RT_AM
    F_NAT --> RT_BQ
    SRC_PIN -.->|"Arrow Read Streams / SQL"| RT_SP & RT_RY & RT_VX & RT_GC & RT_GK & RT_AM & RT_BQ
    RT_SP & RT_RY & RT_VX & RT_GC & RT_GK & RT_AM & RT_BQ --> AUDIT

    classDef data fill:#E8F0FE,stroke:#4285F4,stroke-width:2px,color:#174EA6
    classDef route fill:#FCE8E6,stroke:#EA4335,stroke-width:2px,color:#A50E0E
    classDef compute fill:#E6F4EA,stroke:#34A853,stroke-width:2px,color:#0D652D
    class SRC_PIN data
    class DAG,F_STAT,F_ML,F_DL,F_AML,F_NAT,AUDIT route
    class RT_SP,RT_RY,RT_VX,RT_GC,RT_GK,RT_AM,RT_BQ compute
```

| Diagram Block | Source Modules | Guides & Notebooks | Tests & Smokes |
| :--- | :--- | :--- | :--- |
| **Family DAG, Sizing, Fallback & GPU Audit** | [`dag.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/dag.py), [`router.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/router.py), [`shared_clusters.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/shared_clusters.py), [`capacity.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/capacity.py), [`compute_fallback.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/compute_fallback.py), [`device_audit.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/device_audit.py), [`resources/`](https://github.com/statmike/scale-forecasting/tree/main/src/scale_forecasting/resources) | [choosing_a_runtime.md](./choosing_a_runtime.md), [quota_and_scale.md](./quota_and_scale.md), [cost_estimates.md](./cost_estimates.md) | [`test_dag.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_dag.py), [`test_resources.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_resources.py), [`test_capacity.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_capacity.py), [`test_compute_fallback.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_compute_fallback.py), [`test_device_audit.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_device_audit.py), [`10_mixed_runtimes.json`](https://github.com/statmike/scale-forecasting/blob/main/configs/smokes/10_mixed_runtimes.json) |
| **`spark` & `ray` Distributed Engines** | [`engines/spark_explode.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/engines/spark_explode.py), [`engines/spark_io.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/engines/spark_io.py), [`engines/ray_engine.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/engines/ray_engine.py), [`engines/ray_io.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/engines/ray_io.py) | [runtimes_reference.md](./runtimes_reference.md), [Notebook 03](./notebooks/03_spark_serverless_and_connect.ipynb), [Notebook 04](./notebooks/04_ray_on_vertex_gpu.ipynb) | [`test_spark_engines.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_spark_engines.py), [`test_ray_engine.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_ray_engine.py), [`01_serverless_cpu.json`](https://github.com/statmike/scale-forecasting/blob/main/configs/smokes/01_serverless_cpu.json), [`08_ray_gpu.json`](https://github.com/statmike/scale-forecasting/blob/main/configs/smokes/08_ray_gpu.json) |
| **`vertex`, `gce` & `gke` Worker-Shard Engines** | [`engines/vertex_engine.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/engines/vertex_engine.py), [`vertex_submit.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/vertex_submit.py), [`gce_submit.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/gce_submit.py), [`gke_submit.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/gke_submit.py) | [reading_source_data.md](./reading_source_data.md), [Notebook 02](./notebooks/02_vertex_and_gce_vms.ipynb) | [`test_vertex_runtime.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_vertex_runtime.py), [`38_vertex_custom_job.json`](https://github.com/statmike/scale-forecasting/blob/main/configs/smokes/38_vertex_custom_job.json), [`39_gce_single_vm.json`](https://github.com/statmike/scale-forecasting/blob/main/configs/smokes/39_gce_single_vm.json), [`40_gke_indexed_job.json`](https://github.com/statmike/scale-forecasting/blob/main/configs/smokes/40_gke_indexed_job.json) |
| **`vertex_automl` & `bigquery` SQL Engines** | [`engines/automl_engine.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/engines/automl_engine.py), [`engines/bigquery_engine.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/engines/bigquery_engine.py), [`engines/bigquery_sql.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/engines/bigquery_sql.py) | [models_reference.md](./models_reference.md), [Notebook 01](./notebooks/01_bigquery_native_sql.ipynb) | [`test_automl_engine.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_automl_engine.py), [`test_bigquery_engine.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_bigquery_engine.py), [`02_bq_native.json`](https://github.com/statmike/scale-forecasting/blob/main/configs/smokes/02_bq_native.json), [`42_vertex_automl_tabular_workflow.json`](https://github.com/statmike/scale-forecasting/blob/main/configs/smokes/42_vertex_automl_tabular_workflow.json) |

---

## Level 5 — Stage 2B: Forecasting Science Inside the Worker (`5A`–`5E`)

Zooming inside an executing task: **the unit of work is byte-identical across laptop, Spark, Ray, Vertex, GCE, and GKE**. The top spine below shows the end-to-end execution order inside every run, and sub-diagrams **`5A`–`5E`** unpack our entire forecasting methodology (models, covariates, backtesting/HPO/calibration, metrics, and hierarchical/temporal reconciliation + ensembling):

```mermaid
flowchart LR
    subgraph PREP_W["1. Fold-Safe Hygiene & Cross-Industry Primitives"]
        direction TB
        F1["Grid Policy · Leading-Zero Trim · Trading Days<br/>Stockout censored_col · Outliers · SBC regime_routing"]
        F2["3-Tier Covariates · Moving Holiday Windows<br/>Weather HDD/CDD Splines · @monotone Sign Constraints"]
        F3["7-Lens Similarity (@similarity_global / cold_start)<br/>Simplex (clr/ilr) · Bass/Weibull · @arps_decline · supersession_map"]
    end

    subgraph FIT_W["2. Fit, Backtest, HPO, Calibrate & Explain"]
        direction TB
        W1["worker.run_cell & run_panel_model<br/>37 Models · Local, Global, Hybrid & Custom 1-File Plugins"]
        W2["4 Walk-Forward CV Schemes + Optuna HPO<br/>Scored Uniformly Across 21 Metrics"]
        W3["Split Conformal or Online ACI Intervals<br/>Chernozhukov Monotone Quantiles · Newsvendor · 2-Tier Attribution"]
    end

    WRITE_W["Storage Write API Append<br/>registry/cells.write_cells<br/>Base Predictions, OOF & Metadata"]

    subgraph POST_W["3. Coherence & 3-Scope Consensus"]
        direction TB
        P1["9 Reconciliation Methods Across 3 Modes<br/>Spatial (8) · THieF · cross_temporal_mint · similarity_hierarchy"]
        P2["8 Ensemble Strategies (In-Run · reensemble · ensemble_runs)<br/>mean · median · inverse_error · champion · horizon_specialist<br/>nnls · ridge · xgb (+ top_k · parsimony_margin · similarity_smoothing)"]
    end

    WRITE_POST["Storage Write API Append<br/>Reconciled Nodes & Ensemble Rows<br/>-> 5 Registry Tables"]

    PREP_W --> FIT_W --> WRITE_W --> POST_W --> WRITE_POST

    classDef data fill:#E8F0FE,stroke:#4285F4,stroke-width:2px,color:#174EA6
    classDef compute fill:#E6F4EA,stroke:#34A853,stroke-width:2px,color:#0D652D
    class WRITE_W,WRITE_POST data
    class F1,F2,F3,W1,W2,W3,P1,P2 compute
```

### 5A. All 37 models, 5 model families & the 31 `<model>@<tag>` variant modifiers

Every model inherits from [`BaseModel`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/models/base_model.py), declares its covariate flags (`supports_future_covariates`, `supports_past_covariates`, `supports_static_covariates`), training modes (`local`, `global`, `hybrid`), GPU affinity (`gpu_useful`), fast origin updates (`recondition`), and 2-tier explainability (`feature_attributions` + horizon-step `explain`), and composes orthogonally with **31 `@tag` modifiers**:

```mermaid
flowchart TD
    BM["BaseModel 1-File Plugin Contract (models/base_model.py)<br/>fit / predict · fit_panel / predict_panel · search_space(trial)<br/>recondition · feature_attributions (Tier 1) · explain (Tier 2)"]

    subgraph PY_STAT["1. Family: statistical (18 Python Models — CPU Parallel)"]
        direction TB
        ST1["Baselines, Spectral & Intermittent (6)<br/>naive_mean · naive_seasonal · naive_drift · naive_moving_average · croston · fft"]
        ST2["Exponential Smoothing & Decomposition (7)<br/>theta · auto_theta · holtwinters · autoets · auto_ces · tbats · stl_bagging"]
        ST3["ARIMA, State-Space & Additive (5 — Exog Capable)<br/>auto_arima · sarimax · ucm · kalman · prophet"]
        ST1 --> ST2 --> ST3
    end

    subgraph PY_MLDL["2 & 3. Families: ml (5) & deep_learning (8 — CPU or T4/L4/A100 GPU)"]
        direction TB
        ML1["ml (5 — Local, @global, @hybrid · @direct & @recursive · @monotone · TreeSHAP)<br/>regression_lags · random_forest · lightgbm · xgboost · catboost"]
        DL1["deep_learning Supervised PyTorch (5 — Local, @global & @similarity_global)<br/>neuralprophet · tide · tft · tsmixer · patchtst"]
        DL2["deep_learning Open-Weight Foundation (3 — Zero-Shot & Covariate Context)<br/>chronos_bolt · chronos2 · timesfm_torch"]
        ML1 --> DL1 --> DL2
    end

    subgraph MGD_FAMS["4 & 5. Managed Cloud Families: automl (4) & native (2)"]
        direction TB
        AM1["automl (4 — Vertex AI Tabular Workflows & AutoML TrainingJob)<br/>vertex_l2l · vertex_tide · vertex_tft · vertex_seq2seq"]
        NT1["native (2 — Pure Serverless SQL in BigQuery)<br/>arima_plus (ARIMA_PLUS / XREG) · timesfm (AI.FORECAST)"]
        AM1 --> NT1
    end

    TAGS["31 Composable @tag Modifiers (VALID_VARIANT_TAGS in config.py)<br/>Scope & Step (6): @local · @global · @hybrid · @similarity_global · @direct · @recursive<br/>Covariate Ablation (9): @univariate · @cov_none · @cov_future_only · @cov_past_only · @cov_static_only · @cov_dynamic · @cov_future_static · @cov_past_static · @cov_all<br/>Transforms (5): @transform_none · @transform_log1p · @transform_boxcox · @transform_clr · @transform_ilr<br/>Causal & Lifecycle (8): @monotone · @arps_decline · @cold_start · @analog_transfer · @bass_diffusion · @weibull_lifecycle · @cold_start_arps · @similarity_cold_start<br/>Online Calibration (3): @aci · @aci_norm · @aci_normalized"]

    BM --> ST1 & ML1 & AM1
    ST3 & DL2 & NT1 --> TAGS

    classDef config fill:#FEF7E0,stroke:#F9AB00,stroke-width:2px,color:#7A5A00
    classDef route fill:#FCE8E6,stroke:#EA4335,stroke-width:2px,color:#A50E0E
    classDef compute fill:#E6F4EA,stroke:#34A853,stroke-width:2px,color:#0D652D
    class BM,TAGS config
    class AM1,NT1 route
    class ST1,ST2,ST3,ML1,DL1,DL2 compute
```

### 5B. Three-tier covariates across the temporal cutoff (`T_cutoff`) & domain feature hygiene

Exogenous drivers are partitioned strictly by **what is observable after `T_cutoff`**, while pre-fit hygiene and target transforms fit strictly inside `ds <= T_cutoff` on every backtest fold:

```mermaid
flowchart LR
    subgraph HYG["1. Fold-Safe Hygiene & Domain Regimes (Fit on ds <= T_cutoff Only)"]
        direction TB
        H_GRID["Grid & Lineage Hygiene<br/>grid_policy (strict · fill_zero · ffill · linear)<br/>trim_leading_zeros · supersession_map (V1 -> V2)"]
        H_CLEAN["Stockouts, Outliers & Intermittent SBC<br/>censored_col imputation · outlier_policy (winsorize · interpolate · flag · stl_resid)<br/>regime_routing (smooth · erratic · intermittent · lumpy)"]
        H_XFORM["Target Transforms & Cold-Start Physics<br/>transform: none · log1p · boxcox · clr · ilr (simplex_sum)<br/>cold_start_method: analog_transfer · bass_diffusion · weibull_lifecycle · arps_decline"]
        H_GRID --> H_CLEAN --> H_XFORM
    end

    subgraph COVS["2. Three-Tier Exogenous Covariate Cutoff Contract"]
        direction TB
        T_STAT["Tier 1 · Static Covariates (static_covariates)<br/>Time-invariant entity metadata (store_format, brand, region)<br/>Broadcast to @global models, Gower similarity & AutoML"]
        T_PAST["Tier 3 · Past-Only Covariates (past_covariates + exog_lags)<br/>Observed ONLY for ds <= T_cutoff (traffic, macro, realized temp)<br/>Masked > T_cutoff · Auto-lagged or fed to past encoders (TFT/TiDE/Chronos-2)"]
        T_FUT["Tier 2 · Known-Future Covariates (future_covariates · ds <= T_cutoff + H)<br/>Planned price & promo · holidays & holiday_window ramps<br/>fourier · level_shift · trading_days · degree_day_cols (HDD/CDD)"]
        T_STAT --> T_PAST --> T_FUT
    end

    subgraph GUARD_COV["3. Routing & Economic Guardrails"]
        direction TB
        G_CAP["Capability Router (on_unsupported_covariates)<br/>fallback (graceful tier drop) OR error<br/>+ include_univariate_baselines FVA control"]
        G_MON["Causal Sign Guardrails (@monotone)<br/>monotonic_constraints = {price: -1, promo: +1}<br/>Enforced in LightGBM, XGBoost, CatBoost & Bounded Ridge"]
        G_CAP --> G_MON
    end

    HYG & COVS --> GUARD_COV

    classDef data fill:#E8F0FE,stroke:#4285F4,stroke-width:2px,color:#174EA6
    classDef route fill:#FCE8E6,stroke:#EA4335,stroke-width:2px,color:#A50E0E
    classDef compute fill:#E6F4EA,stroke:#34A853,stroke-width:2px,color:#0D652D
    class T_STAT,T_PAST,T_FUT data
    class G_CAP,G_MON route
    class H_GRID,H_CLEAN,H_XFORM compute
```

### 5C. Rolling-origin backtesting (`4` schemes), Optuna HPO & uncertainty calibration

Backtesting scores models out-of-sample across `n_folds` walk-forward cutoffs while a **separate full-history fit** produces the shipped production forecast:

```mermaid
flowchart LR
    subgraph BT_CFG["1. Walk-Forward Fold Geometry (backtest.py)"]
        direction TB
        B_SCH["4 CV Schemes (backtest.scheme)<br/>expanding · sliding (window)<br/>expanding_frozen (recondition) · expanding_stale"]
        B_STEP["Cutoff Spacing & Publication Lag<br/>n_folds · horizon · step · gap · min_train · control_arm"]
        B_SHORT["Short-Series Policy (backtest.short_series)<br/>adapt · overlap · shrink_train · skip · error<br/>-> Status: full · reduced · unscored · failed"]
    end

    subgraph HPO_FIT["2. Nested Optuna HPO & Dual Fit Path (hpo.py · worker.py)"]
        direction TB
        HPO_N["Optuna TPE Hyperparameter Search<br/>granularity = fleetwide (sample_size pre-pass) OR per_series<br/>n_trials · Optimizes backtest.decision_metric inside cutoffs"]
        FOLD_N["Per-Fold Holdout Scoring<br/>Emits backtest_oof rows (y_true, yhat_raw, yhat, horizon_step)<br/>+ 21-Metric summary in forecast_metadata"]
        FULL_N["Separate 100%-History Production Fit<br/>Produces shipped forecast_predictions (1..H)<br/>Always written even when short series is unscored"]
    end

    subgraph CAL_OUT["3. Empirical Calibration & Physical Guardrails (calibration.py · aci.py)"]
        direction TB
        CAL_M["Distribution-Free Interval Calibration<br/>calibration_method = conformal (per-horizon) · aci · aci_normalized (aci_gamma)<br/>+ Chernozhukov Monotone Rearrangement across output.quantiles"]
        CAL_G["Physical Bounds, Rounding & Newsvendor Target<br/>floor / cap (or floor_col / cap_col) · integer_rounding (cumulative_round)<br/>underage_cost / overage_cost -> Asymmetric Newsvendor q*"]
    end

    BT_CFG --> HPO_FIT --> CAL_OUT

    classDef config fill:#FEF7E0,stroke:#F9AB00,stroke-width:2px,color:#7A5A00
    classDef compute fill:#E6F4EA,stroke:#34A853,stroke-width:2px,color:#0D652D
    classDef decision fill:#E4F7FB,stroke:#12B5CB,stroke-width:2px,color:#005A66
    class B_SCH,B_STEP,B_SHORT config
    class HPO_N,FOLD_N,FULL_N compute
    class CAL_M,CAL_G decision
```

### 5D. All 21 evaluation metrics & how `backtest.decision_metric` steers execution

Every model family and ensemble strategy is scored through the same pure-Python `metrics.compute_metrics` contract across **16 point metrics** and **5 interval metrics**:

```mermaid
flowchart LR
    CTX["Out-of-Fold MetricContext<br/>y_true · yhat · y_train<br/>yhat_lower · yhat_upper · seasonal_period"]

    subgraph M21["All 21 Registered Metrics (METRIC_NAMES in metrics/__init__.py)"]
        direction TB
        M_REL["Relative & Percentage Error (5)<br/>wape (default decision_metric) · smape · mape · maape · ope"]
        M_ABS["Scale-Dependent, Log & Signed Bias (5)<br/>mae · rmse · mse · rmsle · bias"]
        M_SCL["Training-Scaled Point Error (4)<br/>mase · mase_seasonal · rmsse · msse"]
        M_FIT["Goodness-of-Fit & Dispersion (2)<br/>r2 · cv"]
        M_INT["Prediction-Interval & Quantile Quality (5)<br/>coverage · pinball · interval_score · interval_width · msis"]
    end

    subgraph STEER["How backtest.decision_metric Steers Downstream Decisions"]
        direction TB
        S_OPT["Model Tuning & Variant Selection<br/>Optuna HPO Objective · @tag Ablation Winner · autofocus Champion Ratchet"]
        S_ENS["Consensus Blending & Point Alignment<br/>champion · inverse_error · nnls / ridge / xgb Stackers · point_forecast='auto'"]
        S_SQL["Governed Leaderboards & FVA Views<br/>forecast_metadata (21 cols) · v_model_leaderboard_comparable · v_fva_waterfall"]
    end

    CTX --> M_REL & M_ABS & M_SCL & M_FIT & M_INT
    M_REL & M_ABS & M_SCL & M_FIT & M_INT --> S_OPT & S_ENS & S_SQL

    classDef data fill:#E8F0FE,stroke:#4285F4,stroke-width:2px,color:#174EA6
    classDef route fill:#FCE8E6,stroke:#EA4335,stroke-width:2px,color:#A50E0E
    classDef compute fill:#E6F4EA,stroke:#34A853,stroke-width:2px,color:#0D652D
    classDef decision fill:#E4F7FB,stroke:#12B5CB,stroke-width:2px,color:#005A66
    class CTX,S_SQL data
    class S_OPT route
    class M_REL,M_ABS,M_SCL,M_FIT,M_INT compute
    class S_ENS decision
```

### 5E. Hierarchy construction, 3-mode reconciliation (`9` methods) & 3-scope ensembling (`8` strategies)

Post-worker consensus guarantees **additive coherence across trees and calendars** (`verify_coherence < 1e-6`) and **leak-free multi-model blending** across three operational scopes:

```mermaid
flowchart LR
    BASE_IN["Base & OOF Predictions<br/>Single Run (barrier / microbatch)<br/>OR Past Registry Runs"]

    subgraph COH["3-Mode Hierarchical & Temporal Reconciliation (9 Methods)"]
        direction TB
        H_TREE["Tree Construction (HierarchyConfig)<br/>Explicit hierarchy.levels (middle_level) OR<br/>Data-Driven similarity_hierarchy (7 Lenses, Ward S_cs)"]
        H_SPAT["Mode 1 · Spatial Cross-Sectional (8 Methods in reconciliation.py)<br/>bottom_up · top_down · middle_out · ols<br/>wls_struct · wls_var · mint_shrink · mint_nonneg (Active-Set >= 0)"]
        H_TEMP["Modes 2 & 3 · Temporal THieF & Joint Cross-Temporal<br/>reconcile_temporal (temporal_aggregations)<br/>cross_temporal_mint (spatial_temporal_weight='auto')"]
        H_TREE --> H_SPAT --> H_TEMP
    end

    subgraph ENS_ALL["3-Scope Ensemble Engine (8 Strategies in ensembler.py)"]
        direction TB
        E_SCOPE["3 Execution Scopes<br/>1. In-Run (barrier / Spark microbatch) · 2. Zero-Refit reensemble()<br/>3. Cross-Run ensemble_runs([run_id_1, run_id_2, ...])"]
        E_CALC["5 Calculated & Selection Combiners<br/>mean · median · inverse_error<br/>champion (per-series winner) · horizon_specialist (3 horizon buckets)"]
        E_STACK["3 OOF-Learned Meta-Stackers + Guardrails<br/>nnls · ridge · xgb<br/>top_k · prune_threshold · parsimony_margin · variant_policy · similarity_smoothing"]
        E_SCOPE --> E_CALC --> E_STACK
    end

    OUT_COH["Coherent, Calibrated Registry Rows<br/>Propagates Conformal Intervals & All 21 Metrics"]

    BASE_IN --> H_TREE & E_SCOPE
    H_TEMP & E_STACK --> OUT_COH

    classDef data fill:#E8F0FE,stroke:#4285F4,stroke-width:2px,color:#174EA6
    classDef compute fill:#E6F4EA,stroke:#34A853,stroke-width:2px,color:#0D652D
    classDef decision fill:#E4F7FB,stroke:#12B5CB,stroke-width:2px,color:#005A66
    class BASE_IN,OUT_COH data
    class H_TREE,H_SPAT,H_TEMP compute
    class E_SCOPE,E_CALC,E_STACK decision
```

| Diagram Block | Source Modules | First-Visual Reference Guides & Notebooks | Tests & Smokes |
| :--- | :--- | :--- | :--- |
| **`5A` — 37 Models, 5 Families & 31 `@tag` Modifiers** | [`models/`](https://github.com/statmike/scale-forecasting/tree/main/src/scale_forecasting/models), [`models/base_model.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/models/base_model.py), [`config.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/config.py) | [models_reference.md](./models_reference.md), [adding_a_model.md](./adding_a_model.md), [Notebook 00](./notebooks/00_model_playground.ipynb), [Notebook 09](./notebooks/09_custom_models_and_metrics.ipynb) | [`test_models_contract.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_models_contract.py), [`test_worker_model_matrix.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_worker_model_matrix.py), [`test_phase_g5_variants_and_borrowing.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_phase_g5_variants_and_borrowing.py), [`21_full_catalogue.json`](https://github.com/statmike/scale-forecasting/blob/main/configs/smokes/21_full_catalogue.json) |
| **`5B` — 3-Tier Covariates (`T_cutoff`) & Domain Hygiene** | [`features.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/features.py), [`seasonality.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/seasonality.py), [`holiday_window.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/holiday_window.py), [`similarity.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/similarity.py), [`simplex.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/simplex.py), [`cold_start.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/cold_start.py), [`monotonicity.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/monotonicity.py) | [features_and_covariates.md](./features_and_covariates.md), [api/similarity.md](./api/similarity.md), [Notebook 05](./notebooks/05_covariates_and_global_models.ipynb) | [`test_features.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_features.py), [`test_phase_g2_hygiene_and_quantiles.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_phase_g2_hygiene_and_quantiles.py), [`test_similarity.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_similarity.py), [`test_track_p2_cross_industry.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_track_p2_cross_industry.py), [`32_covariates_three_tier.json`](https://github.com/statmike/scale-forecasting/blob/main/configs/smokes/32_covariates_three_tier.json) |
| **`5C` — 4-Scheme Backtesting, Optuna HPO & Calibration** | [`worker.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/worker.py), [`backtest.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/backtest.py), [`hpo.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/hpo.py), [`calibration.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/calibration.py), [`aci.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/aci.py), [`registry/cells.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/registry/cells.py) | [backtesting.md](./backtesting.md), [writing_results.md](./writing_results.md), [Notebook 07](./notebooks/07_hpo_backtesting_and_ensembles.ipynb) | [`test_backtest.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_backtest.py), [`test_hpo.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_hpo.py), [`test_calibration.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_calibration.py), [`test_aci.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_aci.py), [`22_backtest_sliding_overlap.json`](https://github.com/statmike/scale-forecasting/blob/main/configs/smokes/22_backtest_sliding_overlap.json), [`26_hpo_fleetwide.json`](https://github.com/statmike/scale-forecasting/blob/main/configs/smokes/26_hpo_fleetwide.json) |
| **`5D` — 21 Evaluation Metrics & `decision_metric`** | [`metrics/`](https://github.com/statmike/scale-forecasting/tree/main/src/scale_forecasting/metrics), [`metrics/base_metric.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/metrics/base_metric.py) | [metrics_reference.md](./metrics_reference.md), [adding_a_metric.md](./adding_a_metric.md), [Notebook 09](./notebooks/09_custom_models_and_metrics.ipynb) | [`test_metrics_contract.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_metrics_contract.py), [`test_registry_column_parity.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_registry_column_parity.py) |
| **`5E` — 3-Mode Reconciliation (`9`) & 3-Scope Ensembles (`8`)** | [`reconciliation.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/reconciliation.py), [`temporal_reconciliation.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/temporal_reconciliation.py), [`ensembler.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/ensembler.py), [`ensemble_run.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/ensemble_run.py) | [hierarchical_forecasting.md](./hierarchical_forecasting.md), [models_reference.md](./models_reference.md#ensemble-blending--stacking-strategies), [api/temporal_reconciliation.md](./api/temporal_reconciliation.md), [Notebook 06](./notebooks/06_hierarchical_reconciliation.ipynb), [Notebook 07](./notebooks/07_hpo_backtesting_and_ensembles.ipynb) | [`test_reconciliation.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_reconciliation.py), [`test_temporal_reconciliation.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_temporal_reconciliation.py), [`test_ensembler.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_ensembler.py), [`test_ensemble_run.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_ensemble_run.py), [`11_ensemble_barrier.json`](https://github.com/statmike/scale-forecasting/blob/main/configs/smokes/11_ensemble_barrier.json), [`12_ensemble_microbatch.json`](https://github.com/statmike/scale-forecasting/blob/main/configs/smokes/12_ensemble_microbatch.json), [`35_hierarchy_reconciliation.json`](https://github.com/statmike/scale-forecasting/blob/main/configs/smokes/35_hierarchy_reconciliation.json), [`37_hierarchy_covariates_ensemble.json`](https://github.com/statmike/scale-forecasting/blob/main/configs/smokes/37_hierarchy_covariates_ensemble.json) |

---

## Level 6 — Stage 3A: Self-Documenting Registry (`erDiagram`), 11 Always-On Views & Knowledge Catalog

Zooming into **`5 BigQuery Registry Tables -> 11 Analytical SQL Views -> Knowledge Catalog & DataAgent`**.

> [!IMPORTANT]
> **Architectural Boundary: Always-On Core Registry vs. Opt-In Conversational Layer**
>
> - **Always Created on Every Deployment & Run (Blue `data` — `0` Extra APIs, `\$0` View Storage):** All **5 native registry tables**, all **11 analytical SQL views** (`5` core operational/leaderboard views + `6` server-side diagnostic & executive S&OP views), and all inline BigQuery table, view, and column `OPTIONS(description='...')` are created and updated automatically by [`tables.ensure_tables()`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/registry/tables.py) $\rightarrow$ [`tables.ensure_views()`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/registry/tables.py) on **every forecast run**, **Terraform deploy**, and **`Registry().init()`** (which also backfills missing descriptions onto older registries via `sync_existing_descriptions`). SQL analysts, Looker dashboards, and SDK users get all 11 views automatically — **even if Conversational Analytics is never enabled**.
> - **Opt-In Governance & Conversational Layer (Purple `semantic` — `scale_forecasting.bq_agent`):** Only when you call `Registry().ensure_bq_agent()`, pass `--ensure-agent`, or set `"output": {"ensure_bq_agent": true}` does the platform provision the external **Knowledge Catalog** Business Glossary, **Knowledge Catalog** `DataProfile` scans, Golden SQL examples, and the stateful BigQuery Conversational Analytics `DataAgent` (`sf-registry-<dataset>`).

### 6A. Entity-relationship schema (`erDiagram`) across source panels, 5 registry tables & 11 views

```mermaid
erDiagram
    SOURCE_PANELS ||--o{ RUN_REGISTRY : "pinned_by_snapshot_millis"
    RUN_REGISTRY ||--o{ RUN_JOBS : "run_id"
    RUN_REGISTRY ||--o{ FORECAST_METADATA : "run_id"
    RUN_REGISTRY ||--o{ FORECAST_PREDICTIONS : "run_id"
    RUN_REGISTRY ||--o{ BACKTEST_OOF : "run_id"
    FORECAST_METADATA ||--o{ FORECAST_PREDICTIONS : "run_id_ts_id_model_type"
    FORECAST_METADATA ||--o{ BACKTEST_OOF : "run_id_ts_id_model_type"
    RUN_REGISTRY ||--o{ CORE_VIEWS_5 : "dedupe_and_rollup"
    RUN_JOBS ||--o{ CORE_VIEWS_5 : "job_ledger_and_gpu_audit"
    FORECAST_METADATA ||--o{ CORE_VIEWS_5 : "21_metrics_and_fold_status"
    FORECAST_METADATA ||--o{ DIAGNOSTIC_VIEWS_6 : "sbc_and_tier1_attributions"
    FORECAST_PREDICTIONS ||--o{ DIAGNOSTIC_VIEWS_6 : "horizon_units_and_tier2_drivers"
    BACKTEST_OOF ||--o{ DIAGNOSTIC_VIEWS_6 : "pooled_fva_decay_bias_anomalies"

    SOURCE_PANELS {
        string ts_id PK "Series identifier"
        timestamp ds PK "Observation or future timestamp"
        float64 y "Historical target value"
        string covariates "Static, historical & future drivers"
    }
    RUN_REGISTRY {
        string run_id PK "Deterministic config digest"
        timestamp created_at "Partition key DATE(created_at)"
        int64 snapshot_millis "Pinned source read epoch"
        json raw_config "Full validated RunConfig"
        string status "STAGED, RUNNING, COMPLETED, FAILED"
        json job_telemetry "Per-family sizing & capacity JSON"
    }
    RUN_JOBS {
        string run_id PK "Join key to run_registry"
        string family PK "statistical, ml, deep_learning, automl, native, ensemble"
        int64 attempt PK "Monotonic attempt counter"
        string runtime "Resolved compute runtime"
        string device_verdict "GPU utilization audit verdict"
    }
    FORECAST_METADATA {
        string run_id PK "Join key to run_registry"
        string ts_id PK "Series or hierarchy node id"
        string model_type PK "Model, variant, or ensemble name"
        float64 metrics_21 "16 point + 5 interval metric columns"
        json fit_diagnostics "SBC regime, Trigg signal & Tier-1 attributions"
    }
    FORECAST_PREDICTIONS {
        string run_id PK "Join key to run_registry"
        string ts_id PK "Series or hierarchy node id"
        string model_type PK "Model, variant, or ensemble name"
        timestamp forecast_date PK "Future horizon timestamp"
        float64 yhat "Point forecast +yhat_lower/+yhat_upper"
        json quantiles_and_explanations "Full quantile grid & Tier-2 drivers"
    }
    BACKTEST_OOF {
        string run_id PK "Join key to run_registry"
        string ts_id PK "Series identifier"
        string model_type PK "Model, variant, or ensemble name"
        int64 fold_id PK "Walk-forward fold index"
        timestamp ds PK "Holdout timestamp with actual y and yhat"
    }
    CORE_VIEWS_5 {
        string v_run_summary "Run wall-clock, critical-path & overhead ledger"
        string v_run_jobs "Current attempt per family + GPU verdict"
        string v_model_leaderboard "Mean/median metrics, fit time & error_rate"
        string v_model_leaderboard_comparable "Pooled SUM(|y-yhat|)/SUM(|y|) on shared folds"
        string v_backtest_coverage "Fold histogram: full, reduced, unscored, failed"
    }
    DIAGNOSTIC_VIEWS_6 {
        string v_fva_waterfall "6-stage lift from 1_naive to 6_reconciled"
        string v_horizon_decay "Per-step WAPE, MAE, bias & 80% coverage"
        string v_bias_and_exceptions "Planner queue: |Trigg| >= 0.6 or high WAPE"
        string v_anomalies "Out-of-fold spikes & stockout dips (|z| >= 2.5)"
        string v_feature_attributions "Unnested Tier-1 global & Tier-2 step drivers"
        string v_executive_sop_summary "Single-row S&OP champion, FVA, units & exceptions"
    }
```

### 6B. Always-on BigQuery semantic foundation vs. opt-in Knowledge Catalog & `DataAgent`

```mermaid
flowchart LR
    subgraph CORE_ON["Always-On Core Registry (Created on Every Run & Registry.init — 0 Extra APIs)"]
        direction TB
        DDL["5 Native Tables + Inline OPTIONS(description=...)<br/>ddl.py · metadata.py · sync_existing_descriptions"]
        V5["5 Core Operational & Leaderboard Views<br/>v_run_summary · v_run_jobs · v_model_leaderboard<br/>v_model_leaderboard_comparable · v_backtest_coverage"]
        V6["6 Diagnostic & Executive S&OP Views<br/>v_fva_waterfall · v_horizon_decay · v_bias_and_exceptions<br/>v_anomalies · v_feature_attributions · v_executive_sop_summary"]
        DDL --> V5 & V6
    end

    subgraph OPT_IN["Opt-In Knowledge Catalog & Conversational Layer (bq_agent.ensure_bq_agent)"]
        direction TB
        SCOPE["Strict Allowlist Scope (discover_registry_scope)<br/>5 Tables + 11 Views + Verified $.data.source_table Panels<br/>Excludes _sf_automl_* Staging & Unrelated Dataset Tables"]
        KC_GLOSS["Knowledge Catalog Business Glossary<br/>44 Domain Terms across 5 Categories · <= 8 EntryLinks/Table"]
        KC_SCAN["Knowledge Catalog DataProfile Scans<br/>Auto-Published Column Cardinality, Nulls & Distributions"]
        GOLDEN["Published Context Set<br/>32 Parameterized Golden SQL Queries + Join Edges + Latest-Run Rule"]
        AGENT["Stateful BigQuery DataAgent<br/>sf-registry-<dataset> (geminidataanalytics v1beta)"]
        SCOPE --> KC_GLOSS & KC_SCAN & GOLDEN --> AGENT
    end

    DDL & V5 & V6 -->|"Queried directly by SQL, Looker, SDK & Workbench"| SQL_DIRECT["Immediate SQL, Looker & SDK Access<br/>(No Conversational Setup Needed)"]
    DDL & V5 & V6 -->|"Published when opted in"| SCOPE

    classDef data fill:#E8F0FE,stroke:#4285F4,stroke-width:2px,color:#174EA6
    classDef semantic fill:#F3E8FD,stroke:#9334E6,stroke-width:2px,color:#5B1799
    classDef decision fill:#E4F7FB,stroke:#12B5CB,stroke-width:2px,color:#005A66
    class DDL,V5,V6 data
    class SCOPE,KC_GLOSS,KC_SCAN,GOLDEN,AGENT semantic
    class SQL_DIRECT decision
```

| Diagram Block | Source Modules | Guides & Notebooks | Tests & Smokes |
| :--- | :--- | :--- | :--- |
| **5 Registry Tables & Self-Documenting Metadata** | [`registry/ddl.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/registry/ddl.py), [`registry/metadata.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/registry/metadata.py), [`registry/tables.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/registry/tables.py) | [output_schemas.md](./output_schemas.md), [api/registry_ops.md](./api/registry_ops.md) | [`test_ddl.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_ddl.py), [`test_bq_agent.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_bq_agent.py), [`ddl_deployment.sql`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/snapshots/ddl_deployment.sql) |
| **11 Always-On Analytical SQL Views** | [`registry/views.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/registry/views.py), [`registry/reads.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/registry/reads.py) | [output_schemas.md](./output_schemas.md), [api/registry_views.md](./api/registry_views.md), [Notebook 10](./notebooks/10_registry_operations_and_scale.ipynb) | [`test_views.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_views.py), [`views.sql`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/snapshots/views.sql) |
| **Strict Allowlist Scope, Knowledge Catalog Glossary & Scans** | [`bq_agent/context.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/bq_agent/context.py), [`bq_agent/glossary.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/bq_agent/glossary.py), [`bq_agent/scans.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/bq_agent/scans.py) | [agent_and_mcp_guide.md](./agent_and_mcp_guide.md), [Notebook 12](./notebooks/12_conversational_analytics_and_knowledge_catalog.ipynb) | [`test_bq_agent.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_bq_agent.py), [`test_bq_agent_live.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/integration/test_bq_agent_live.py) |
| **32 Golden SQL Queries & Stateful `DataAgent`** | [`bq_agent/golden_queries.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/bq_agent/golden_queries.py), [`bq_agent/client.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/bq_agent/client.py), [`bq_agent/rest.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/bq_agent/rest.py) | [running_and_reviewing.md](./running_and_reviewing.md), [api/bq_agent.md](./api/bq_agent.md) | [`test_bq_agent.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_bq_agent.py), [`test_bq_agent_live.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/integration/test_bq_agent_live.py) |

---

## Level 7 — Stage 3B: The Forecaster Workbench — 6 Pillars & Prescriptive Solvers

Zooming into **`Forecaster Workbench`** ([`workbench.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/workbench.py)): **every diagnostic, causal decomposition, similarity search, copula simulator, and LP/MILP optimizer runs identically on both live cloud `Forecaster` objects and zero-GCP `playground.bakeoff()` results**, turning forecasts into auditable commercial and supply-chain decisions:

```mermaid
flowchart LR
    IN_WB["Governed Predictions, OOF & Metadata<br/>Cloud Forecaster (BigQuery) OR Offline BakeOff (In-Memory)"]

    subgraph DIAGNOSTICS["Pillars 1–3 · Trust, Live Drift & Similarity Merchandising"]
        direction TB
        WB1["1. Practitioner Trust, FVA & Attribution<br/>7-Question assess() / practitioner_assessment_df · fva_df / fva_cascade_df<br/>Diebold-Mariano & Wilcoxon · horizon_decay_df (h*) · Trigg bias_df · attributions_df"]
        WB2["2. Post-Publication Drift, Whipsaw & Reporting<br/>score_realized (Live Drift Flag) · revision_stability_df (Whipsaw)<br/>detect_anomalies_df · detect_change_points_df · exceptions_df"]
        WB3["3. 7-Lens + Multi-Modal Similarity & Merchandising<br/>similarity_df (TF-IDF + Gower + LCA) · find_similar_series_df<br/>recommend_substitutes_df (Stock-Aware) · detect_emerging_trends_df"]
    end

    subgraph PRESCRIPTIVE["Pillars 4–6 · Causal Econometrics, Tail Risk & Solvers"]
        direction TB
        WB4["4. Causal Econometrics & 5-Stage Counterfactual What-If<br/>Double-ML promo_elasticity_df (Lift · Pull-Forward Dip · Ad-Stock · Halo)<br/>audit_scenario_monotonicity · simulate_scenarios (Margin + Delta ROP/SS)"]
        WB5["5. Joint Tail Risk, Coherent Rollups & Smooth Goal-Seek QP<br/>AR(1) rollup_time · Online aci_df · Copula sample_trajectories & tail_risk_df (VaR/CVaR)<br/>sample_coherent_paths · allocate_target QP (Locks + Bounds + Smoothness)"]
        WB6["6. Single-SKU ROP/SS + 7 SciPy/HiGHS LP & MILP Optimizers<br/>inventory_policy_df · optimize_inventory_df · optimize_scenario<br/>optimize_clearance_df · optimize_contract_volume_df<br/>optimize_assortment_df · optimize_prepacks_df · optimize_rebalancing_df"]
    end

    REPORT["12-Tab Self-Contained HTML & JSON Executive Report<br/>export_report / python -m scale_forecasting.report"]

    IN_WB --> WB1 & WB2 & WB3 & WB4 & WB5 & WB6
    WB1 & WB2 & WB3 & WB4 & WB5 & WB6 --> REPORT

    classDef data fill:#E8F0FE,stroke:#4285F4,stroke-width:2px,color:#174EA6
    classDef compute fill:#E6F4EA,stroke:#34A853,stroke-width:2px,color:#0D652D
    classDef decision fill:#E4F7FB,stroke:#12B5CB,stroke-width:2px,color:#005A66
    class IN_WB data
    class WB1,WB2,WB3 compute
    class WB4,WB5,WB6,REPORT decision
```

| Workbench Pillar | Source Modules | Guides & Notebooks | Tests & Smokes |
| :--- | :--- | :--- | :--- |
| **1. Trust, FVA Waterfall & Statistical Tests** | [`workbench.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/workbench.py), [`diagnostics.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/diagnostics.py), [`review.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/review.py) | [forecaster_workbench.md](./forecaster_workbench.md), [auto_forecaster.md](./auto_forecaster.md), [Notebook 11](./notebooks/11_auto_forecaster_and_autofocus.ipynb) | [`test_phase_g1_workbench.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_phase_g1_workbench.py), [`test_phase_g3_diagnostics_and_scenarios.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_phase_g3_diagnostics_and_scenarios.py), [`test_review.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_review.py) |
| **2. Realized Drift, Whipsaw, Change-Points & HTML Report** | [`diagnostics.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/diagnostics.py), [`report.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/report.py) | [forecaster_workbench.md](./forecaster_workbench.md), [Notebook 01](./notebooks/01_bigquery_native_sql.ipynb), [Notebook 08](./notebooks/08_multi_engine_master_workflow.ipynb) | [`test_phase_g4_lifecycle_and_report.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_phase_g4_lifecycle_and_report.py) |
| **3. 7-Lens Similarity, Substitutes & Emerging Trends** | [`similarity.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/similarity.py) | [forecaster_workbench.md](./forecaster_workbench.md), [api/similarity.md](./api/similarity.md), [Notebook 00](./notebooks/00_model_playground.ipynb), [Notebook 05](./notebooks/05_covariates_and_global_models.ipynb) | [`test_similarity.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_similarity.py), [`test_track_p_e2e.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_track_p_e2e.py), [`test_track_p2_cross_industry.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_track_p2_cross_industry.py) |
| **4. Double-ML Promo Elasticity & 5-Stage What-If** | [`elasticity.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/elasticity.py), [`scenarios.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/scenarios.py), [`monotonicity.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/monotonicity.py) | [forecaster_workbench.md](./forecaster_workbench.md), [api/elasticity.md](./api/elasticity.md), [Notebook 05](./notebooks/05_covariates_and_global_models.ipynb) | [`test_elasticity.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_elasticity.py), [`test_scenarios.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_scenarios.py), [`test_track_j_e2e.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_track_j_e2e.py) |
| **5. Online ACI, Copula `VaR`/`CVaR` & Goal-Seek QP** | [`aci.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/aci.py), [`copula.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/copula.py), [`goal_seek.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/goal_seek.py) | [forecaster_workbench.md](./forecaster_workbench.md), [api/copula.md](./api/copula.md), [api/goal_seek.md](./api/goal_seek.md), [Notebook 06](./notebooks/06_hierarchical_reconciliation.ipynb), [Notebook 08](./notebooks/08_multi_engine_master_workflow.ipynb) | [`test_aci.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_aci.py), [`test_copula.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_copula.py), [`test_goal_seek.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_goal_seek.py), [`test_track_k_e2e.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_track_k_e2e.py) |
| **6. Single-SKU `ROP`/`SS` + 7 LP & MILP Solvers** | [`inventory_optimizer.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/inventory_optimizer.py), [`scenario_optimizer.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/scenario_optimizer.py), [`clearance_optimizer.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/clearance_optimizer.py) | [forecaster_workbench.md](./forecaster_workbench.md), [api/inventory_optimizer.md](./api/inventory_optimizer.md), [api/scenario_optimizer.md](./api/scenario_optimizer.md), [api/clearance_optimizer.md](./api/clearance_optimizer.md), [Notebook 06](./notebooks/06_hierarchical_reconciliation.ipynb), [Notebook 08](./notebooks/08_multi_engine_master_workflow.ipynb) | [`test_inventory_policy.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_inventory_policy.py), [`test_inventory_optimizer.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_inventory_optimizer.py), [`test_scenario_optimizer.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_scenario_optimizer.py), [`test_clearance_optimizer.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_clearance_optimizer.py), [`test_track_l_e2e.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_track_l_e2e.py) |

---

## Level 8 — Stage 3C: AI Agent Ecosystem, 5 Chat Surfaces, Day-2 Repair & 3-Tier Verification

Zooming into **AI Agent Integration, Day-2 Operations, and Repository Drift Guards**:

1. **5 Repository AI Agent Surfaces + 5 Conversational `DataAgent` Surfaces:** Connects IDE/CLI coding agents (**Google Antigravity `agy`**, **Claude Code**, **Cursor**, **VS Code Copilot**, **Windsurf**) via `SKILL.md`, the pure-Python `stdio` **MCP server (`7` resources, `14` tools)**, JSON Schema, and `llms.txt` — while exposing the BigQuery `DataAgent` across **5 chat surfaces** (BigQuery Studio UI, check-and-run SDK/Notebooks `probe_bq_agent` + `ask()`, CLI `--ask`, MCP `ask_forecast_registry`, and `output.ensure_bq_agent = true`).
2. **Day-2 Lifecycle & Surgical Repair:** Live cloud probes (`probes/`) reconcile mid-flight batch/cluster/pod states, while `retry_run` re-fits **only failed or missing `(ts_id, model)` cells** without recomputing healthy ones.
3. **3-Tier Test & Automated Drift-Lock Gate:** Pre-commit/pre-push hooks (`make hooks`), 12-case pathological torture & SQL-injection suites, live `@gcp`/`@spark`/`@ray` + 13-notebook headless Colab acceptance, and the 42-smoke live infrastructure harness keep code, schemas, and docs permanently aligned.

```mermaid
flowchart LR
    CORE_SRC["Canonical Python Registries & Governed BigQuery Dataset"]

    subgraph AGENTS["1. AI Agent & Conversational Ecosystem"]
        direction TB
        AG_REPO["5 Repository Agent Surfaces (agent_surfaces.py + mcp.py)<br/>skills.sh (npx skills add statmike/scale-forecasting) · SKILL.md · MCP (7 Resources, 14 Tools)<br/>JSON Schema ($schema) · llms.txt & llms-full.txt · Antigravity plugin.json"]
        AG_CHAT["5 Conversational Analytics Surfaces (bq_agent/)<br/>1. BigQuery Studio UI · 2. SDK/Notebooks probe_bq_agent() + ask() / BakeOff.ask()<br/>3. CLI --ask · 4. MCP Tool #14 ask_forecast_registry · 5. output.ensure_bq_agent"]
    end

    subgraph DAY2["2. Day-2 Probes, Repair & Hygiene"]
        direction TB
        OP1["Live Platform Probes & Settlement (probes/)<br/>monitor(probe=True) · --settle · close-runs · doctor · cancel"]
        OP2["Surgical Repair & Scoped Cleanup (retry_run.py · registry/ops.py)<br/>retry_run (Failed Cells Only) · reap-clusters · drop-run · sweep-orphans · snapshot · export"]
    end

    subgraph VERIFY["3. 3-Tier Verification & Drift Locks"]
        direction TB
        T_OFF["Tier 1: Offline Gate + Pre-Commit Tripwires (make test)<br/>85%+ Coverage · 12-Case Torture & SQL Safety · make agent-surfaces · make social-card"]
        T_LIVE["Tier 2 & Tier 3: Live Cloud Proof (validation.md)<br/>@gcp / @spark / @ray · 13-Notebook Colab Acceptance · 42-Smoke Live Suite"]
    end

    CORE_SRC --> AG_REPO & AG_CHAT & OP1 & OP2 & T_OFF & T_LIVE

    classDef data fill:#E8F0FE,stroke:#4285F4,stroke-width:2px,color:#174EA6
    classDef semantic fill:#F3E8FD,stroke:#9334E6,stroke-width:2px,color:#5B1799
    classDef route fill:#FCE8E6,stroke:#EA4335,stroke-width:2px,color:#A50E0E
    class CORE_SRC data
    class AG_REPO,AG_CHAT semantic
    class OP1,OP2,T_OFF,T_LIVE route
```

| Diagram Block | Source Modules | Guides & Notebooks | Tests & Smokes |
| :--- | :--- | :--- | :--- |
| **5 Repository AI Agent Surfaces & MCP** | [`agent_surfaces.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/agent_surfaces.py), [`mcp.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/mcp.py), [`skills/scale-forecasting/SKILL.md`](https://github.com/statmike/scale-forecasting/blob/main/skills/scale-forecasting/SKILL.md), [`plugin.json`](https://github.com/statmike/scale-forecasting/blob/main/plugin.json) | [agent_and_mcp_guide.md](./agent_and_mcp_guide.md), [api/agent_surfaces.md](./api/agent_surfaces.md), [api/mcp.md](./api/mcp.md) | [`test_agent_surfaces.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_agent_surfaces.py), [`test_mcp_server.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_mcp_server.py) |
| **5 Conversational Analytics Surfaces** | [`bq_agent/probe.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/bq_agent/probe.py), [`bq_agent/client.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/bq_agent/client.py), [`bq_agent/offline.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/bq_agent/offline.py), [`bq_agent/__main__.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/bq_agent/__main__.py) | [running_and_reviewing.md](./running_and_reviewing.md), [Notebook 10](./notebooks/10_registry_operations_and_scale.ipynb), [Notebook 12](./notebooks/12_conversational_analytics_and_knowledge_catalog.ipynb) | [`test_bq_agent.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_bq_agent.py), [`test_bq_agent_live.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/integration/test_bq_agent_live.py) |
| **Live Probes, Surgical `retry_run` & Hygiene** | [`probes/`](https://github.com/statmike/scale-forecasting/tree/main/src/scale_forecasting/probes), [`retry_run.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/retry_run.py), [`retry_policy.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/retry_policy.py), [`registry/ops.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/registry/ops.py), [`ray_reaper.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/ray_reaper.py) | [operations.md](./operations.md), [troubleshooting.md](./troubleshooting.md), [Notebook 10](./notebooks/10_registry_operations_and_scale.ipynb) | [`test_probes.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_probes.py), [`test_retry_run.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_retry_run.py), [`test_registry_ops.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_registry_ops.py), [`repair_retry_demo.json`](https://github.com/statmike/scale-forecasting/blob/main/configs/repair_retry_demo.json) |
| **3-Tier Test Suite & Drift Tripwires** | [`notebook_acceptance.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/notebook_acceptance.py), [`smoke_run.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/smoke_run.py), [`tests/smokes/smoke_harness.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/smokes/smoke_harness.py) | [validation.md](./validation.md), [smoke_testing.md](./smoke_testing.md) | [`test_docs_integrity.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_docs_integrity.py), [`test_validation_ledger.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_validation_ledger.py), [`test_pathological_torture_suite.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_pathological_torture_suite.py), [`test_sql_injection_safety.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/unit/test_sql_injection_safety.py) |

---

## Appendix — Zoom index & auto-synced repository inventory

### Progressive zoom index

| System Stage | Section | What Is Unpacked |
| :--- | :--- | :--- |
| **Role Routing** | [**Persona Map — 5 Enterprise Personas & Entry Surfaces**](#persona-map--5-enterprise-personas--entry-surfaces) | Which SDK methods, CLI commands, SQL views, notebooks (`00`–`12`), and zoom levels serve each team role. |
| **0. Foundation** | [**Level 1 — Stage 0: Two-Stage Cloud Deployment, 100k Seeding & Zero-Rebuild Code Delivery**](#level-1--stage-0-two-stage-cloud-deployment-100k-seeding--zero-rebuild-code-delivery) | `terraform/bootstrap` + `12` `terraform/main` modules, `13` packaging extras, 100k deterministic `data_gen`, and dynamic `src.zip` shipping. |
| **1. Data, Config & Launch** | [**Level 2 — Stage 1A: Authoring, Auto-Compiling, Ablating & Evolving a `RunConfig`**](#level-2--stage-1a-authoring-auto-compiling-ablating--evolving-a-runconfig) | `12` root `RunConfig` blocks, `@tag` ablation ladders, Glass-Box Auto-Forecaster (`autofocus` + `29` benchmarks), `Forecaster.evolve()`, and `run_id`. |
| **1. Data, Config & Launch** | [**Level 3 — Stage 1B: The 5 Execution Paths, Preflight & Duration/Cost Predictor**](#level-3--stage-1b-the-5-execution-paths-preflight--durationcost-predictor) | Pure-offline sandbox, `--feasibility` / `--quota` / `predict_run_duration`, `--stage-only`, `main.run`, direct submitters, and Cloud Composer 3. |
| **2. Zero-Idle Compute** | [**Level 4 — Stage 2A: Parallel Per-Family Routing, Sizing, GPU Audit & 7 Cloud Runtimes**](#level-4--stage-2a-parallel-per-family-routing-sizing-gpu-audit--7-cloud-runtimes) | Parallel `FamilyJob` fan-out, `ResourceSlot`/`UnitShape` sizing, regional fallback, Arrow `row_restriction` pushdown, 7 runtimes, and `device_verdict`. |
| **2. Zero-Idle Compute** | [**Level 5 — Stage 2B: Forecasting Science Inside the Worker (`5A`–`5E`)**](#level-5--stage-2b-forecasting-science-inside-the-worker-5a5e) | `5A` all `37` models & `31` `@tag` modifiers · `5B` 3-tier covariates across `T_cutoff` & domain hygiene · `5C` 4-scheme backtesting, Optuna HPO & calibration · `5D` all `21` metrics · `5E` 3-mode coherence (`9` methods) & 3-scope ensembles (`8` strategies). |
| **3. Governed Results** | [**Level 6 — Stage 3A: Self-Documenting Registry (`erDiagram`), 11 Always-On Views & Knowledge Catalog**](#level-6--stage-3a-self-documenting-registry-erdiagram-11-always-on-views--knowledge-catalog) | Primary/join keys across 5 tables & **11 always-created views** vs. the **opt-in** Knowledge Catalog glossary, `DataProfile` scans, and `DataAgent`. |
| **3. Governed Results** | [**Level 7 — Stage 3B: The Forecaster Workbench — 6 Pillars & Prescriptive Solvers**](#level-7--stage-3b-the-forecaster-workbench--6-pillars--prescriptive-solvers) | FVA/DM tests, live drift & whipsaw, 7-lens similarity merchandising, Double-ML & 5-stage What-If, Copula `VaR`/`CVaR`, Goal-Seek QP, and **7 LP/MILP solvers**. |
| **3. Governed Results** | [**Level 8 — Stage 3C: AI Agent Ecosystem, 5 Chat Surfaces, Day-2 Repair & 3-Tier Verification**](#level-8--stage-3c-ai-agent-ecosystem-5-chat-surfaces-day-2-repair--3-tier-verification) | 5 repository AI surfaces (`SKILL.md`, MCP, JSON Schema, `llms.txt`), 5 `DataAgent` chat surfaces, surgical `retry_run`, and the 3-tier test gate. |

### Auto-synced repository inventory

> [!NOTE]
> The table below (and all dynamic counts inside the Mermaid diagrams on this page) is regenerated automatically from live Python reflection via `make agent-surfaces` (`python -m scale_forecasting.agent_surfaces --write`), and `Level 0` is synced from `README.md` via `make social-card`.

<!-- BEGIN AUTO-SYNCED VISUAL INVENTORY -->
| System Surface | Live Code Count & Breakdown | Canonical Source of Truth |
| :--- | :--- | :--- |
| **Forecasting Models** | **37 models** across **5 families** (`18` `statistical`, `5` `ml`, `8` `deep_learning`, `4` `automl`, `2` `native`) + **31 `@tag` modifiers** | [`models/__init__.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/models/__init__.py), [`config.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/config.py) |
| **Evaluation Metrics** | **21 metrics** (`16` point accuracy + `5` prediction interval) | [`metrics/__init__.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/metrics/__init__.py) |
| **Cloud Runtimes & GPUs** | **7 runtimes** (`spark`, `ray`, `vertex`, `gce`, `gke`, `vertex_automl`, `bigquery`) · **4 GPU types** (`T4`, `L4`, `A100`, `A100_80GB`) | [`config.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/config.py), [`resources/catalog.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/resources/catalog.py) |
| **Config, Auto-Compiler & Benchmarks** | **12 root `RunConfig` fields** · **4 exposure tiers** · **5 scale regimes** · **20 synthetic DGPs + 9 public benchmarks** | [`auto.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/auto.py), [`auto_policy.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/auto_policy.py), [`benchmarks.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/benchmarks.py) |
| **Similarity, Ensembles & Coherence** | **7 similarity lenses** · **8 ensemble strategies** · **9 reconciliation methods** (3-mode `THieF` & `cross_temporal_mint` + `clr`/`ilr` simplex) | [`similarity.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/similarity.py), [`ensembler.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/ensembler.py), [`temporal_reconciliation.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/temporal_reconciliation.py) |
| **Always-On BigQuery Registry & Views** | **5 native registry tables** · **11 always-created SQL views** (`5` core + `6` diagnostic/S&OP) · `20` object & `326` column descriptions | [`registry/ddl.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/registry/ddl.py), [`registry/views.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/registry/views.py), [`registry/metadata.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/registry/metadata.py) |
| **Opt-In Knowledge Catalog & `DataAgent`** | **44 Business Glossary terms** · `DataProfile` scans · **32 Golden SQL queries** · **5 chat surfaces** | [`bq_agent/`](https://github.com/statmike/scale-forecasting/tree/main/src/scale_forecasting/bq_agent) |
| **Forecaster Workbench & Solvers** | **6 Workbench pillars** · **7-Question `assess()`** · **Single-SKU `ROP`/`SS` + 7 SciPy/HiGHS LP & MILP optimizers** · **12-tab HTML/JSON report** | [`workbench.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/workbench.py), [`inventory_optimizer.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/inventory_optimizer.py), [`clearance_optimizer.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/clearance_optimizer.py) |
| **AI Agent & MCP Ecosystem** | **6 generated agent files** · **1 portable `SKILL.md`** · **MCP server (`7` resources, `14` tools)** across 5 IDE/CLI clients | [`agent_surfaces.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/agent_surfaces.py), [`mcp.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/mcp.py) |
| **Infrastructure, Notebooks & Validation** | **12 Terraform modules** · **13 packaging extras** · **13 notebooks** · **42 smoke + 20 demo configs** | [`terraform/`](https://github.com/statmike/scale-forecasting/tree/main/terraform), [`notebooks/`](https://github.com/statmike/scale-forecasting/tree/main/notebooks), [`configs/`](https://github.com/statmike/scale-forecasting/tree/main/configs) |
<!-- END AUTO-SYNCED VISUAL INVENTORY -->

---

## Where to go next

- **Run your first forecast in 5 minutes:** [Getting started](./getting_started.md) or [`notebooks/00_model_playground.ipynb`](./notebooks/00_model_playground.ipynb).
- **Read the function-by-function call spine:** [Software architecture (`architecture.md`)](./architecture.md).
- **Dive into all 6 Workbench pillars & 7 LP/MILP solvers:** [Forecaster workbench (`forecaster_workbench.md`)](./forecaster_workbench.md).
- **Explore the 11 SQL views & Knowledge Catalog:** [Output schemas (`output_schemas.md`)](./output_schemas.md), [AI agents & MCP (`agent_and_mcp_guide.md`)](./agent_and_mcp_guide.md), and [`notebooks/12_conversational_analytics_and_knowledge_catalog.ipynb`](./notebooks/12_conversational_analytics_and_knowledge_catalog.ipynb).
- **Audit live cloud proofs across all 7 runtimes:** [System validation ledger (`validation.md`)](./validation.md).
