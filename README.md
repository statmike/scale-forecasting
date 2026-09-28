# scale-forecasting

<p align="center">
  <b>Enterprise-Grade, Massively Parallel Time-Series Forecasting on Google Cloud</b><br>
  <i>One declarative configuration. 18 models. Hybrid distributed execution across BigQuery ML, Dataproc Spark, and Vertex AI Ray.</i>
</p>

<p align="center">
  <a href="https://console.cloud.google.com/vertex-ai/colab/import/https%3A%2F%2Fraw.githubusercontent.com%2Fstatmike%2Fscale-forecasting%2Fmain%2Fnotebooks%2Fmodel_playground.ipynb"><img src="https://img.shields.io/badge/Colab%20Enterprise-Launch%20Notebooks-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white" alt="Colab Enterprise"></a>
  <a href="https://statmike.github.io/scale-forecasting/"><img src="https://img.shields.io/badge/Docs-Product%20Documentation-0F9D58?style=for-the-badge&logo=materialformkdocs&logoColor=white" alt="Docs"></a>
  <a href="./docs/workshop.md"><img src="https://img.shields.io/badge/Workshop-Hands--On%20Lab-F4B400?style=for-the-badge&logo=google&logoColor=white" alt="Workshop"></a>
  <a href="./terraform/README.md"><img src="https://img.shields.io/badge/Terraform-1--Click%20Deploy-7B42BC?style=for-the-badge&logo=terraform&logoColor=white" alt="Terraform"></a>
</p>

<p align="center">
  <a href="#quickstart-local-in-5-minutes">⚡ 5-Minute Quickstart</a> •
  <a href="#why-scale-forecasting">💡 Why Scale Forecasting</a> •
  <a href="#the-technology-stack">🛠️ The Stack</a> •
  <a href="#how-runs-work-declarative-json-configurations">⚙️ Configurations</a> •
  <a href="#extending-the-platform-custom-models--metrics-in-1-file">🔌 Extensibility</a> •
  <a href="#architecture-overview">🏛️ Architecture</a> •
  <a href="#interactive-notebook-suite">📓 Notebooks</a> •
  <a href="#deploy-on-google-cloud-in-15-minutes">☁️ Deploy on GCP</a>
</p>

---

## What Is `scale-forecasting`?

`scale-forecasting` brings the modeling flexibility of modern time-series ecosystems (Prophet, Statsmodels, LightGBM, XGBoost, NeuralProphet) to **enterprise Google Cloud scale**. It allows data science and engineering teams to forecast **100,000+ time series** concurrently, perform rigorous rolling-origin backtesting, stack models into learned ensembles, and capture complete experiment lineage in BigQuery — all orchestrated from a single JSON configuration.

The entire platform deploys with 1-click Terraform, pre-seeded with a 100,000-series dataset across both native BigQuery and BigLake Apache Iceberg tables.

---

## Why Scale Forecasting?

Traditional forecasting workflows break down when scaled to hundreds of thousands of series across retail, supply chain, energy, or financial hierarchies:

| Challenge | Traditional Approach | The `scale-forecasting` Solution |
| :--- | :--- | :--- |
| **Library Fragmentation** | Separate, incompatible codebases for Statsmodels, Prophet, PyTorch, and SQL models. | **Unified Model Contract:** Single [`BaseModel`](./src/scale_forecasting/models/base_model.py) interface. 18 models run with identical inputs, outputs, and metrics. |
| **Compute Scaling Limits** | Single-node memory exhaustion (OOMs); slow sequential loops. | **Hybrid Distributed Execution:** Automatic fan-out across Managed Service for Apache Spark (Dataproc Serverless), Gemini Enterprise (Managed Ray on Vertex AI), and BigQuery ML. |
| **Infrastructure Lock-In** | Forced choice between pure Spark or pure SQL. | **Multi-Engine DAG:** Run Spark, Ray, and BigQuery ML *concurrently under one `run_id`*, bounded by the slowest family rather than their sum. |
| **Uncertainty & Calibration** | Gaussian assumptions that fail on real-world skewed distributions. | **Conformal Residual Intervals:** Empirical, distribution-free prediction intervals calibrated against rolling backtest errors. |
| **Operational Opacity** | Disconnected log files and missing evaluation tracking. | **Real-Time BigQuery Registry:** Streaming telemetry via the Storage Write API into 12 analytical SQL views and interactive dashboards. |

---

## The Technology Stack

`scale-forecasting` integrates best-of-breed open-source forecasting algorithms with Google Cloud's data and AI services:

| Component / Layer | Google Cloud Service & Architecture | Primary Role in Platform | Documentation |
| :--- | :--- | :--- | :--- |
| **Data Warehouse & Lakehouse** | **[BigQuery](https://cloud.google.com/bigquery/docs)** & **[BigLake Apache Iceberg](https://cloud.google.com/bigquery/docs/iceberg-tables)** | Stores input time series, acts as the central run registry (`run_registry`, `forecast_predictions`, `forecast_metadata`), and exposes 12 analytical SQL views. | [BigQuery Overview](https://cloud.google.com/bigquery/docs) |
| **SQL-Native Machine Learning** | **[BigQuery ML](https://cloud.google.com/bigquery/docs/bqml-introduction)** | Executes `ARIMA_PLUS`, `ARIMA_PLUS_XREG`, and zero-shot foundation models via `AI.FORECAST` (`TimesFM`) directly in SQL. | [BigQuery ML Guide](https://cloud.google.com/bigquery/docs/bqml-introduction) |
| **Distributed Big Data Engine** | **[Managed Service for Apache Spark (Dataproc)](https://cloud.google.com/dataproc/docs)** | Executes massively parallel cross-joins and pandas UDFs (`applyInPandas`) on Dataproc Serverless or GCE clusters. | [Dataproc Serverless Docs](https://cloud.google.com/dataproc-serverless/docs) |
| **Distributed AI & Ray Compute** | **[Gemini Enterprise / Vertex AI (Managed Ray)](https://cloud.google.com/vertex-ai/docs/open-source/ray/overview)** | Dynamic autoscaling Ray actor pools with fractional GPU packing (NVIDIA L4/T4) for deep learning models like `NeuralProphet`. | [Managed Ray on Vertex AI](https://cloud.google.com/vertex-ai/docs/open-source/ray/overview) |
| **Interactive Analytics** | **[Colab Enterprise](https://cloud.google.com/colab/docs/enterprise-overview)** | Hosted, collaborative Jupyter notebooks pre-wired to the deployment runtime template (`sf-main`) with zero client configuration. | [Colab Enterprise Overview](https://cloud.google.com/colab/docs/enterprise-overview) |
| **Workflow Orchestration** | **[Managed Service for Apache Airflow (Cloud Composer)](https://cloud.google.com/composer/docs)** | Automated end-to-end DAG scheduling, fan-out orchestration across engines, and SLA monitoring. | [Managed Airflow Docs](https://cloud.google.com/composer/docs) |
| **Secure Networking** | **[Virtual Private Cloud (VPC)](https://cloud.google.com/vpc/docs)** & **[Private Service Connect (PSC-I)](https://cloud.google.com/vpc/docs/private-service-connect)** | Private worker subnet, Cloud NAT for outbound dependency resolution, and PSC interface attachments for secure Ray cluster access. | [Private Service Connect](https://cloud.google.com/vpc/docs/private-service-connect) |
| **Infrastructure as Code** | **[Terraform (Google Provider)](https://registry.terraform.io/providers/hashicorp/google/latest/docs)** | 1-click automated deployment of all buckets, datasets, networking, service accounts, and seed datasets. | [Terraform Provider](https://registry.terraform.io/providers/hashicorp/google/latest/docs) |

---

## Quickstart (Local in 5 Minutes)

Experiment locally with zero cloud setup, zero credentials, and zero compute costs.

```bash
# 1. Clone the repository and install dependencies with uv
git clone https://github.com/statmike/scale-forecasting.git && cd scale-forecasting
uv sync

# 2. Explore available models and fit your first forecast
uv run python -m scale_forecasting.playground --list
uv run python -m scale_forecasting.playground --model theta --backtest --horizon 14
```

### Python SDK Quickstart

The high-level [`Forecaster`](./docs/using_the_sdk.md) SDK unifies validation, execution, and review:

```python
import scale_forecasting as sf

# 1. Initialize from a declarative config file or dictionary
forecaster = sf.Forecaster.from_file("configs/quickstart_100.json")

# 2. Preflight validation & cost sizing (offline, zero GCP calls)
dry_run = forecaster.dry_run()
print(f"Planned Run ID : {dry_run.run_id}")
print(f"Total Fits     : {dry_run.fanout.n_series} series × {len(dry_run.python_models) + len(dry_run.bq_models)} models")

# 3. Execute locally or across Google Cloud (Spark, Ray, BigQuery ML)
result = forecaster.run()

# 4. Interactive review: leaderboard, quantile distributions, and ensemble lift
review = sf.review_run(result.run_id)
sf.plot_leaderboard(review)
```

---

## Three Persona Journeys

Whether you are building models, architecting cloud platforms, or managing production operations, `scale-forecasting` provides a tailored path:

```mermaid
flowchart TD
    subgraph DataScientist["🧑‍🔬 Data Scientist & Forecaster"]
        direction LR
        DS1["Interactive Sandbox<br/>notebooks/model_playground.ipynb"] --> DS2["Custom Models & Metrics<br/>docs/adding_a_model.md"] --> DS3["HPO & Stacking Ensembles<br/>Optuna · NNLS / XGBoost"]
    end

    subgraph Architect["🏛️ Enterprise Cloud & Data Architect"]
        direction LR
        AR1["Storage Strategy<br/>BigQuery vs Iceberg on GCS"] --> AR2["Multi-Engine Placement<br/>Spark vs Ray vs BigQuery ML"] --> AR3["Private Networking & IAM<br/>PSC-I · Least-Privilege SAs"]
    end

    subgraph MLOps["⚙️ MLOps & Platform Engineer"]
        direction LR
        OP1["1-Click Terraform<br/>terraform/README.md"] --> OP2["Scheduled Orchestration<br/>Managed Airflow (Composer 3)"] --> OP3["Fleet Resilience<br/>Multi-Region Fallback & Probes"]
    end

    DataScientist --> Architect --> MLOps
```

---

## How Runs Work: Declarative JSON Configurations

In `scale-forecasting`, **a single declarative JSON file defines the entire experiment**. The configuration specifies which data to read, which models to fit, how to route each model family across cloud engines, how to backtest and calibrate prediction intervals, and how to combine forecasts with stacked ensembles.

Every configuration automatically receives a deterministic, content-addressed `<slug>-<12hex>` **`run_id`** calculated from its contents, guaranteeing complete provenance and idempotent re-runs.

### Example Configuration: Hybrid Multi-Engine Forecasting

Here is an example configuration ([`configs/ensemble_demo.json`](./configs/ensemble_demo.json)) that mixes statistical models on Spark with SQL models in BigQuery and trains a stacked ensemble:

```json
{
  "run_name": "hybrid_stacked_ensemble",
  "data": {
    "table": "source_series_iceberg",
    "limit_series": 100,
    "horizon": 14
  },
  "models": [
    "theta",
    "holtwinters",
    "xgboost",
    "arima_plus"
  ],
  "compute": {
    "families": {
      "statistical": {"runtime": "spark"},
      "ml": {"runtime": "spark"},
      "native": {"runtime": "bigquery"}
    }
  },
  "backtest": {
    "scheme": "expanding",
    "n_folds": 3,
    "horizon": 14,
    "decision_metric": "wape"
  },
  "ensemble": {
    "strategies": ["mean", "inverse_error", "nnls", "xgb"]
  }
}
```

### What Each Section Controls
- **`data`:** Target table (BigLake Iceberg or native BigQuery), series count limit, target column, date column, frequency, and forecast horizon.
- **`models`:** List of model identifiers to run. Models are automatically grouped into execution families (`statistical`, `ml`, `deep_learning`, `native`).
- **`compute`:** Runtime engine selection per family (`spark`, `ray`, `bigquery`), machine types, executor counts, and multi-region fallback options.
- **`backtest`:** Cross-validation scheme (`expanding`, `sliding`), fold counts, evaluation metric selection, and conformal interval calibration.
- **`features`:** Automated country holidays, Fourier seasonality terms, structural level-shift detection, and exogenous covariate lags.
- **`hpo`:** Optuna hyperparameter optimization settings (trial counts, search spaces, and fleet-wide vs. per-series tuning).
- **`ensemble`:** Blending strategies (`mean`, `median`, `inverse_error`, `nnls`, `ridge`, `xgb`) and execution trigger (`barrier` or `microbatch`).

➡️ **Explore all configuration options in the [Configuration Reference Guide (`docs/configuration_reference.md`)](./docs/configuration_reference.md).**

---

## Extending the Platform: Custom Models & Metrics in 1 File

`scale-forecasting` is built to be easily extended by data scientists and machine learning engineers without touching cluster infrastructure.

> [!TIP]
> **Zero Container Image Rebuilds:**
> Third-party dependencies are pre-compiled into the container image (`docker/requirements.txt`). Your Python code in `src/scale_forecasting` is dynamically zipped and distributed at job submission time. Any code edit, new model, or new metric takes effect immediately on the very next run without rebuilding a Docker image!

### 1. Adding a Custom Model in 1 File
Every model implements the lightweight [`BaseModel`](./src/scale_forecasting/models/base_model.py) interface (`fit(series)` and `predict(steps, quantiles)`):
1. Copy [`docs/model_template.py`](./docs/model_template.py) to `src/scale_forecasting/models/my_custom_model.py`.
2. Implement your model's training and forecasting logic using any library (e.g. Scikit-learn, Statsforecast, PyTorch).
3. Export the class in `src/scale_forecasting/models/__init__.py`.
4. Your model is instantly available in the CLI, Python SDK, notebooks, and configuration files!

➡️ **Step-by-step walkthrough: [Adding a Model Guide (`docs/adding_a_model.md`)](./docs/adding_a_model.md).**

### 2. Adding a Custom Evaluation Metric in 1 File
1. Copy [`docs/metric_template.py`](./docs/metric_template.py) to `src/scale_forecasting/metrics/my_custom_metric.py`.
2. Implement the point or interval loss calculation (`compute(y_true, y_pred)`).
3. Add the metric name to `METRIC_NAMES` in `src/scale_forecasting/metrics/__init__.py`.
4. The BigQuery table schema, `ADD COLUMN` migration, Storage Write API protobuf field, and leaderboard views update automatically!

➡️ **Step-by-step walkthrough: [Adding a Metric Guide (`docs/adding_a_metric.md`)](./docs/adding_a_metric.md).**

---

## Architecture Overview

A run begins with a declarative [`RunConfig`](./src/scale_forecasting/config.py). The orchestrator resolves the experiment into an execution DAG: each model family runs as an independent parallel job, streaming results into BigQuery.

```mermaid
flowchart TB
    cfg["RunConfig (JSON)<br/>data · models · compute · backtest · features · hpo · ensemble"]
    orch["Orchestrator: Forecaster.run() / main.run()<br/>plan_dag: resolves runtime, hardware & multi-region placement"]
    cfg --> orch

    subgraph compute["Distributed Compute Engines (Parallel Execution)"]
        spark["Dataproc Spark (Serverless or GCE Cluster)<br/>Cross-join (series × model) → applyInPandas Tasks<br/>Statistical & ML Families (CPU / L4 GPU)"]
        ray["Ray on Vertex AI (Autoscaling Worker Pools)<br/>Dynamic task chunks & fractional GPU packing<br/>Deep Learning & ML Families (CPU / T4 GPU)"]
        bq["BigQuery ML (Native SQL Execution)<br/>CREATE MODEL ... ARIMA_PLUS & AI.FORECAST (TimesFM)<br/>Native Family (Parallel BigQuery Queries)"]
    end

    orch -->|"statistical / ml"| spark
    orch -->|"deep_learning / ml"| ray
    orch -->|"native"| bq

    unit["worker.run_cell(series, model, cfg)<br/>Identical unit of work: Local Python · Spark Pandas UDF · Ray Task"]
    spark --> unit
    ray --> unit

    source[("Enterprise Source Panel<br/>source_series_iceberg or source_series_native<br/>BigQuery Storage Read API (Arrow)")]
    source -.->|snapshot-pinned read| spark
    source -.->|snapshot-pinned read| ray
    source -.->|native SQL read| bq

    subgraph registry["BigQuery Run Registry (Storage Write API)"]
        direction TB
        r_meta["forecast_metadata (15 metrics, fit duration, best params)<br/>forecast_predictions (horizon forecasts + conformal intervals)<br/>backtest_oof (out-of-fold historical predictions)"]
        r_trace["run_registry (lineage, config hash, status)<br/>run_jobs (per-family platform execution trace)"]
    end

    unit -->|streamed Arrow batches| registry
    bq -->|SQL insert| registry

    ens["Ensemble Engine (Driver Pandas / Storage Write API)<br/>Calculated (mean, median, inverse_error) & Learned (nnls, ridge, xgb)"]
    registry --> ens
    ens --> registry

    views["12 Analytical SQL Views<br/>v_model_leaderboard · v_forecast_results · v_run_summary · v_run_jobs"]
    registry --> views
```

---

## Model & Ensemble Catalog

Every model lives in its own self-contained file under [`src/scale_forecasting/models/`](./src/scale_forecasting/models/README.md).

| Model | Family | Runtime Engine | Capabilities & Methodology |
| :--- | :--- | :--- | :--- |
| **`naive_mean`** | `statistical` | Spark / Ray | Historical mean baseline with analytical Gaussian intervals. |
| **`naive_seasonal`** | `statistical` | Spark / Ray | Repeats historical seasonal cycles (weekly/monthly/annual). |
| **`naive_drift`** | `statistical` | Spark / Ray | Linear drift extrapolation between first and last observations. |
| **`naive_moving_average`** | `statistical` | Spark / Ray | Trailing moving average with tunable window lengths. |
| **`theta`** | `statistical` | Spark / Ray | Assimakopoulos-Nikolopoulos decomposition method (`statsmodels`). |
| **`holtwinters`** | `statistical` | Spark / Ray | Additive and multiplicative Holt-Winters seasonal exponential smoothing. |
| **`autoets`** | `statistical` | Spark / Ray | Automated Error-Trend-Seasonal state-space model. |
| **`croston`** | `statistical` | Spark / Ray | Intermittent-demand forecaster (Croston / SBA / TSB) for sparse data. |
| **`sarimax`** | `statistical` | Spark / Ray | Seasonal ARIMA with exogenous calendar & economic covariates. |
| **`ucm`** | `statistical` | Spark / Ray | Unobserved Components state-space model with cycle/trend decomposition. |
| **`stl_bagging`** | `statistical` | Spark / Ray | STL decomposition with block-bootstrapped bagged ETS ensembles. |
| **`prophet`** | `statistical` | Spark / Ray | Piecewise trend, multi-period Fourier seasonality, and holiday events. |
| **`regression_lags`** | `ml` | Spark / Ray | Regularized Ridge regression with autoregressive target and covariate lags. |
| **`lightgbm`** | `ml` | Spark / Ray | Gradient boosted decision trees (`LightGBM`) with recursive multi-step forecasting. |
| **`xgboost`** | `ml` | Spark / Ray | Gradient boosted decision trees (`XGBoost`) on CPU or GPU (`device="cuda"`). |
| **`neuralprophet`** | `deep_learning` | Spark / Ray | PyTorch AR-Net; supports fractional GPU allocation on Ray worker pools. |
| **`arima_plus`** | `native` | BigQuery ML | Pure BigQuery SQL: automated pipeline with anomaly detection & holiday modeling. |
| **`arima_plus_xreg`** | `native` | BigQuery ML | BigQuery ML ARIMA with user-supplied exogenous feature tables. |
| **`timesfm`** | `native` | BigQuery ML | Zero-shot foundation model forecasting via BigQuery `AI.FORECAST`. |

### Ensembling Strategies (`ensemble.strategies`)
- **Calculated (Fast, Backtest-Free):** `mean` (uniform average), `median` (robust consensus), `inverse_error` (weighted inversely by backtest validation loss).
- **Learned (Stacking on Out-of-Fold Predictions):** `nnls` (non-negative constrained weights), `ridge` (L2-regularized linear blend), `xgb` (gradient-boosted meta-learner).

---

## Interactive Notebook Suite

The [`notebooks/`](./notebooks/README.md) directory provides a structured learning curriculum with direct **Run in Colab Enterprise** integrations:

```mermaid
flowchart TD
    subgraph Track1["Track 1: Foundations & Local Prototyping"]
        NB0["model_playground.ipynb<br/>Single-series sandbox · 18 models · conformal intervals (Zero GCP Setup)"]
    end

    subgraph Track2["Track 2: Cloud Runtimes & Distributed Engines"]
        direction TB
        subgraph Single["Single-Engine Execution"]
            direction LR
            NB1["01_spark_via_connect.ipynb<br/>Dataproc Spark Connect & Serverless"]
            NB2["02_bigquery_native.ipynb<br/>Serverless BigQuery ML (Pure SQL)"]
        end
        subgraph Multi["Multi-Engine & GPU Scaling"]
            direction LR
            NB3["03_combo_and_ensemble.ipynb<br/>Hybrid Spark ∥ BQ + Stacking Ensembles"]
            NB4["04_ray_on_vertex.ipynb<br/>Autoscaling Ray on Vertex AI (CPU/GPU)"]
        end
        Single --> Multi
    end

    subgraph Track3["Track 3: Operations, Live Monitoring & Scale Benchmarking"]
        direction LR
        NB8["08_run_and_monitor.ipynb<br/>Background Launch & Live Progress Bar"] --> NB9["09_review_run.ipynb<br/>Post-Run Leaderboards & Ensemble Lift"] --> NB7["07_scale_review.ipynb<br/>100k Cross-Platform Benchmark"]
    end

    Track1 --> Track2 --> Track3
```

| Notebook | Focus Area | Runtime Environment | What You Will Learn |
| :--- | :--- | :--- | :--- |
| [`model_playground.ipynb`](./notebooks/model_playground.ipynb) | Foundations | Local Python (In-Memory) | Fit, score, and plot any of the 16 Python models on synthetic data with zero cloud credentials. |
| [`01_spark_via_connect.ipynb`](./notebooks/01_spark_via_connect.ipynb) | Distributed Spark | Dataproc Spark Connect / Batch | Drive distributed Spark fan-out interactively, compare with serverless batch execution, and inspect write speed. |
| [`02_bigquery_native.ipynb`](./notebooks/02_bigquery_native.ipynb) | Cloud SQL | BigQuery ML (`ARIMA_PLUS`, `TimesFM`) | Execute SQL-native forecasting over native and Iceberg tables without provisioning any compute clusters. |
| [`03_combo_and_ensemble.ipynb`](./notebooks/03_combo_and_ensemble.ipynb) | Multi-Engine Hybrid | Spark Serverless $\parallel$ BigQuery ML | Run Spark and BigQuery concurrently under one `run_id`, blend models with stacking ensembles, and evaluate lift. |
| [`04_ray_on_vertex.ipynb`](./notebooks/04_ray_on_vertex.ipynb) | Distributed Ray | Ray on Vertex AI (CPU & T4 GPU) | Provision an ephemeral Ray-on-Vertex cluster via Private Service Connect, pack GPUs fractionally, and observe auto-teardown. |
| [`08_run_and_monitor.ipynb`](./notebooks/08_run_and_monitor.ipynb) | Operations & Telemetry | Background Thread + BigQuery | Launch a cloud job and monitor real-time cell completion via `Forecaster.monitor()` with automated probe escalation. |
| [`09_review_run.ipynb`](./notebooks/09_review_run.ipynb) | Post-Run Evaluation | BigQuery Registry Views | Generate the model leaderboard, inspect per-series error quantiles (`p10`/`p50`/`p90`), and trace the job timeline. |
| [`07_scale_review.ipynb`](./notebooks/07_scale_review.ipynb) | Enterprise Benchmark | BigQuery Registry (100k Runs) | Compare 100,000-series runs across Spark, Ray, and BigQuery: wall-clock time, cluster overhead, and numerical parity. |

---

## Deploy on Google Cloud in 15 Minutes

Deploy the entire platform into your Google Cloud project using Terraform.

### 1-Click Deployment Architecture

```mermaid
flowchart TD
    subgraph Stage1["Stage 1 · terraform/bootstrap/ (Local State, Run Once)"]
        direction LR
        B1["GCP Project (Optional)<br/>+ Billing Link"] --> B2["GCS Remote State Bucket<br/>&lt;project_id&gt;-tfstate"]
    end

    subgraph Stage2["Stage 2 · terraform/main/ (Remote Backend State)"]
        direction TB
        subgraph Found["Foundation & Lakehouse Storage"]
            direction LR
            M1["APIs & IAM<br/>Least-privilege SAs"]
            M2["Networking<br/>VPC · Subnet · PSC-I"]
            M3["Storage & BigQuery<br/>3 Buckets · Dataset · Iceberg"]
        end
        subgraph Runtimes["Interfaces & Seed Data"]
            direction LR
            M4["Container Runtime<br/>Cloud Build · Artifact Registry"]
            M5["Colab Enterprise<br/>sf-main Runtime Template"]
            M6["100k Seed Dataset<br/>Dataproc Serverless Seed Batch"]
        end
        Found --> Runtimes
    end

    Stage1 --> Stage2
```

### Cost Transparency

- **Free at Rest:** Empty GCS buckets, BigQuery datasets, service accounts, and VPC networks have **zero idle cost**.
- **One-Time Provisioning:** The runtime container image build and 100,000-series data seed execute once during initial deployment (~8.5 minutes, **~\$0.15** total).
- **Pay-as-You-Go Compute:** Dataproc Serverless batches and Vertex AI Ray clusters bill strictly for the duration of your forecast run.
- **Optional Services:** Managed Service for Apache Airflow (Cloud Composer 3) is disabled by default (`create_composer = false`).

### Quick Deployment Steps (from Cloud Shell)

```bash
# 1. Clone repository in Google Cloud Shell
cd ~ && git clone https://github.com/statmike/scale-forecasting.git && cd scale-forecasting

# 2. Stage 1: Bootstrap project and remote state bucket
cd terraform/bootstrap
terraform init
terraform apply -var="project_id=YOUR_PROJECT_ID" -var="billing_account=YOUR_BILLING_ID"

# 3. Stage 2: Provision platform, build image, and seed 100,000 series
cd ../main
terraform init
terraform apply -var="project_id=YOUR_PROJECT_ID"
```

For enterprise VPC integration, custom service accounts, and security controls, see [`terraform/README.md`](./terraform/README.md) and [`docs/deploying_on_gcp.md`](./docs/deploying_on_gcp.md).

---

## Hands-On Workshop

Planning to run a proof-of-concept, team hackathon, or training workshop?

Follow our step-by-step **[Hands-On Workshop Guide](./docs/workshop.md)**. It walks through pre-workshop project setup, permissions for attendee Google Groups, and 6 guided lab exercises from initial local modeling to 100,000-series cross-platform benchmarking.

---

## Repository Documentation Map

- **System Architecture & Design:** [`docs/architecture.md`](./docs/architecture.md)
- **Comprehensive Configuration Reference:** [`docs/configuration_reference.md`](./docs/configuration_reference.md)
- **Hands-On Workshop:** [`docs/workshop.md`](./docs/workshop.md)
- **Python SDK & Developer Guide:** [`docs/using_the_sdk.md`](./docs/using_the_sdk.md)
- **Running, Monitoring & Reviewing:** [`docs/running_and_reviewing.md`](./docs/running_and_reviewing.md)
- **GCP Deployment & IAM Architecture:** [`docs/deploying_on_gcp.md`](./docs/deploying_on_gcp.md) & [`terraform/README.md`](./terraform/README.md)
- **Backtesting & Conformal Calibration:** [`docs/backtesting.md`](./docs/backtesting.md)
- **Quota, Sizing & 100k Scale Guide:** [`docs/quota_and_scale.md`](./docs/quota_and_scale.md)
- **BigQuery Registry & View Schemas:** [`docs/output_schemas.md`](./docs/output_schemas.md)
- **Extending the Platform:** [`docs/adding_a_model.md`](./docs/adding_a_model.md) & [`docs/adding_a_metric.md`](./docs/adding_a_metric.md)
- **Operations & Production Runbook:** [`docs/operations.md`](./docs/operations.md)
- **System Validation Ledger:** [`docs/validation.md`](./docs/validation.md) & [`docs/smoke_testing.md`](./docs/smoke_testing.md)
- **Auto-Generated API Reference:** [`docs/api/index.md`](./docs/api/index.md)
- **Troubleshooting:** [`docs/troubleshooting.md`](./docs/troubleshooting.md)

---

## License

Apache-2.0 — see [`LICENSE`](./LICENSE).
