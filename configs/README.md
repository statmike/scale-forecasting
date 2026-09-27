# Run Configurations (`configs/`)

In `scale-forecasting`, **a single JSON file is the complete experiment record**. It declares which source table to read, which models to fit, how to route each model family across compute runtimes, how to backtest and calibrate prediction intervals, and how to blend base forecasts into ensembles.

When you load a config via [`RunConfig`](./../src/scale_forecasting/config.py), the platform computes a deterministic, content-addressed `<slug>-<12hex>` **`run_id`** from the authored configuration and stores the verbatim JSON in `run_registry.raw_config`.

```mermaid
flowchart LR
    json["JSON Config File<br/>(configs/*.json)"]
    cfg["RunConfig Validation<br/>Pydantic schema + cross-field rules"]
    id["Deterministic run_id<br/>&lt;run_name&gt;-&lt;12-char digest&gt;"]
    dag["Family Execution DAG<br/>plan_dag(cfg)"]

    subgraph jobs["Parallel Per-Family Jobs"]
        stat["statistical family<br/>(Spark or Ray)"]
        ml["ml family<br/>(Spark or Ray)"]
        dl["deep_learning family<br/>(Ray GPU or Spark)"]
        nat["native family<br/>(BigQuery SQL)"]
    end

    ens["ensemble node<br/>(barrier or microbatch)"]

    json --> cfg --> id --> dag
    dag --> stat & ml & dl & nat
    stat & ml & dl & nat --> ens
```

---

## Quick Usage

### From the CLI

```bash
# 1. Preview the resolved execution DAG, run_id, and per-family launch commands (offline, no spend)
uv run python -m scale_forecasting.main --config configs/per_family_runtimes_demo.json --dry-run

# 2. Run a preflight feasibility check against the live source table and regional quotas
uv run python -m scale_forecasting.main --config configs/explode_100k.json --feasibility

# 3. Execute the run end-to-end
uv run python -m scale_forecasting.main --config configs/mixed_demo.json
```

### From Python (`Forecaster` SDK)

```python
from scale_forecasting import Forecaster

fc = Forecaster.from_config_file("configs/ensemble_demo.json")
print(fc.plan())          # Inspect the planned DAG and deterministic run_id
run_id = fc.run()         # Launch and wait for completion
print(fc.review_run())    # Inspect leaderboard, metric distributions, and ensemble lift
```

---

## Shipped Configurations

### 1. Interactive & Demo Configs

Small, fast configurations used by the [interactive notebooks](../notebooks/README.md) and quickstart guides to demonstrate specific runtimes and features:

| File | Series | Models | Runtimes Exercised | Purpose |
| :--- | ---: | :--- | :--- | :--- |
| [`explode_demo.json`](./explode_demo.json) | 10 | `theta`, `holtwinters`, `sarimax`, `xgboost` | Spark Serverless (CPU) | Fast Spark cross-join/explode demo with model persistence (`persist_models: true`). |
| [`bq_native_demo.json`](./bq_native_demo.json) | 100 | `arima_plus`, `timesfm` | BigQuery SQL | Pure SQL-native forecasting with zero Python compute provisioning. |
| [`mixed_demo.json`](./mixed_demo.json) | 10 | `theta`, `arima_plus`, `timesfm` | Spark $\parallel$ BigQuery | Minimal multi-engine run pairing a Python model with BigQuery native models. |
| [`ensemble_demo.json`](./ensemble_demo.json) | 10 | `theta`, `arima_plus`, `timesfm` | Spark $\parallel$ BigQuery + Ensemble | End-to-end multi-engine run with calculated (`mean`, `median`, `inverse_error`) and learned (`nnls`) ensembles. |
| [`ray_cpu_demo.json`](./ray_cpu_demo.json) | 6 | `theta`, `holtwinters`, `arima_plus`, `timesfm` | Ray CPU $\parallel$ BigQuery | Quick Ray-on-Vertex CPU run alongside BigQuery native models. |
| [`ray_gpu_demo.json`](./ray_gpu_demo.json) | 6 | `neuralprophet`, `theta`, `arima_plus`, `timesfm` | Ray GPU + CPU $\parallel$ BigQuery | Demonstrates fractional-GPU packing (`gpu_fraction: "auto"`) for `neuralprophet` on Vertex AI T4s. |
| [`per_family_runtimes_demo.json`](./per_family_runtimes_demo.json) | 50 | `theta`, `holtwinters`, `xgboost`, `neuralprophet`, `arima_plus` | Spark CPU + Ray GPU + BigQuery | Routes `statistical` and `ml` to Spark Serverless, `deep_learning` to Ray GPU, and `native` to BigQuery in one run. |
| [`per_family_runtimes_cpu_demo.json`](./per_family_runtimes_cpu_demo.json) | 50 | `theta`, `holtwinters`, `xgboost`, `neuralprophet`, `arima_plus` | Spark CPU + Ray CPU + BigQuery | Quota-free CPU twin of `per_family_runtimes_demo.json` (runs `neuralprophet` on Ray CPU workers). |
| [`repair_demo.json`](./repair_demo.json) | 3,000 | `theta`, `holtwinters`, `xgboost` | Spark Serverless (CPU) | Medium-scale multi-family run used to demonstrate run inspection, cancellation, and cell-level repair (`--retry`). |
| [`repair_retry_demo.json`](./repair_retry_demo.json) | 300 | `theta`, `holtwinters`, `xgboost` | Spark Serverless (CPU) | Compact 300-series target for testing automated `--with-retry` / `Forecaster.retry()` workflows. |

### 2. Scale Benchmarks (10k – 100k Series)

Configurations designed for fleet-scale execution and reviewed in [`notebooks/07_scale_review.ipynb`](../notebooks/07_scale_review.ipynb) and [`docs/quota_and_scale.md`](../docs/quota_and_scale.md):

| File | Series | Models | Runtimes Exercised | Purpose |
| :--- | ---: | :--- | :--- | :--- |
| [`ray_autoscale_demo.json`](./ray_autoscale_demo.json) | 10,000 | `theta`, `holtwinters`, `sarimax` | Ray CPU (Autoscaling 1–8 nodes) | 10k-series statistical run on an autoscaling Ray-on-Vertex cluster with multi-region fallback. |
| [`all_families_10k.json`](./all_families_10k.json) | 10,000 | `theta`, `holtwinters`, `sarimax`, `xgboost`, `neuralprophet`, `arima_plus`, `timesfm` | Ray (CPU + T4 GPU) $\parallel$ BigQuery | All 4 model families at 10k scale with `gpu_fraction: "auto"` calibration and learned + calculated ensembles. |
| [`all_families_10k_full.json`](./all_families_10k_full.json) | 10,000 | `theta`, `holtwinters`, `sarimax`, `xgboost`, `neuralprophet`, `arima_plus`, `timesfm` | Ray (CPU + T4 GPU) $\parallel$ BigQuery | Full-featured 10k run adding `features.holidays: "US"`, `transform: "log1p"`, `persist_models: true`, and `xgb` ensembling. |
| [`explode_100k.json`](./explode_100k.json) | 100,000 | `theta`, `holtwinters`, `sarimax`, `xgboost` | Spark Serverless (CPU) | Full 100k-series benchmark across `statistical` and `ml` families on Dataproc Serverless (`max_executors: 20`). |
| [`ray_100k.json`](./ray_100k.json) | 100,000 | `theta`, `holtwinters`, `sarimax`, `xgboost` | Ray CPU (Autoscaling 1–20 nodes) | Full 100k-series Ray counterpart to `explode_100k.json` for cross-engine accuracy and throughput comparison. |

### 3. Controlled CPU vs. GPU A/B Comparisons

Paired configurations that hold data, seeds, and hyperparameters constant while varying only the hardware target (`cpu` vs. `gpu`):

| File Pair | Series | Surface | Purpose |
| :--- | ---: | :--- | :--- |
| [`neuralprophet_ab_cpu.json`](./neuralprophet_ab_cpu.json) & [`neuralprophet_ab_gpu.json`](./neuralprophet_ab_gpu.json) | 10,000 | Ray on Vertex AI | Compares `neuralprophet` throughput, cost, and accuracy on 12 CPU workers vs. 12 T4 GPU workers (`gpu_fraction: 0.125`). |
| [`neuralprophet_ab_cluster_cpu.json`](./neuralprophet_ab_cluster_cpu.json) & [`neuralprophet_ab_cluster_gpu.json`](./neuralprophet_ab_cluster_gpu.json) | 3,000 | Dataproc GCE Cluster | Compares `neuralprophet` on an ephemeral 4-worker CPU Dataproc cluster vs. a 4-worker T4 GPU Dataproc cluster. |

### 4. Infrastructure Fallback Policy (`compute_fallback.json`)

- **[`compute_fallback.json`](./compute_fallback.json)** is **not** a `RunConfig` file — it is the regional and machine-type fallback table consumed by [`compute_fallback.py`](../src/scale_forecasting/compute_fallback.py) when you pass `--compute-fallback configs/compute_fallback.json` to the CLI. It defines ordered regional/zone and accelerator fallbacks when a primary region encounters a stockout or quota limit.

---

## Smoke Configuration Suite (`configs/smokes/`)

The **[`configs/smokes/`](./smokes/README.md)** subdirectory contains 31 numbered configurations (`01` through `31`) that systematically exercise every runtime, hardware mode, backtest scheme, HPO granularity, feature transform, and ensemble strategy in the platform. See **[`configs/smokes/README.md`](./smokes/README.md)** for the full index.

---

## Reference Links

- **Every configuration field, type, and default:** [`docs/configuration_reference.md`](../docs/configuration_reference.md)
- **Pydantic schema source:** [`src/scale_forecasting/config.py`](../src/scale_forecasting/config.py)
- **Live validation results for every shipped config:** [`docs/validation.md`](../docs/validation.md)
