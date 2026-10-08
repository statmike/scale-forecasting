# Frequently asked questions (FAQ)

Quick answers to the most common questions about installing, running, sizing, and operating `scale-forecasting`.

---

## Installation & local execution

### Can I run `scale-forecasting` without a Google Cloud project or billing account?
**Yes.** The core package (`pip install scale-forecasting`) has zero Google Cloud dependencies and runs entirely on your local CPU. Run `python -m scale_forecasting.playground` or open [`notebooks/00_model_playground.ipynb`](./notebooks/00_model_playground.ipynb) to generate synthetic panels, fit statistical/tree/deep-learning models, run rolling-origin backtests, and inspect leaderboards with no credentials or cloud project. See **[Getting started](./getting_started.md)**.

### Which Python version is supported?
**Python 3.11** (`>=3.11,<3.12`). Python 3.11 is the single supported version across local development, CI, the shared Artifact Registry runtime container, Dataproc Serverless runtime 2.3 (Spark Connect driver↔executor parity), and Vertex AI Ray 2.47 client↔cluster parity. See **[Version matrix](./version_matrix.md)**.

### Why did I get `MissingExtraError` when calling a cloud runtime or model?
`scale-forecasting` keeps its default install lightweight (10 pure-Python scientific packages). Cloud clients and heavy third-party model libraries live in optional extras (`[gcp]`, `[models]`, `[spark]`, `[ray]`, `[notebook]`, `[all]`). The `MissingExtraError` message prints the exact `pip install` or `uv sync` command needed for the feature you called. See **[Runtime dependencies & extras](./runtime_dependencies.md)**.

---

## Runtimes & architecture

### Do I need Apache Spark to use `scale-forecasting`?
**No.** Spark (`runtime = "spark"`) is one of seven supported execution runtimes. You can run every Python model family on serverless **Vertex AI `CustomJob`** (`runtime = "vertex"`), a self-deleting **Compute Engine VM** (`runtime = "gce"`), **Ray** (`runtime = "ray"`), or **Google Kubernetes Engine** (`runtime = "gke"`), and run SQL/foundation models directly inside **BigQuery ML** (`native` family) or **Vertex AI AutoML** (`automl` family). See **[Choosing a runtime](./choosing_a_runtime.md)**.

### Which runtime should I use for GPU deep-learning models?
Use **`vertex`** (serverless Vertex AI `CustomJob`), **`gce`** (single self-deleting Compute Engine VM), **`ray`** (`ray_mode = "vertex"` or `"gke"`), or **`gke`** (`gke_mode = "job"` or `"ray"`). Set `"use_gpu": true` and choose `"gpu_type"` from `"T4"`, `"L4"`, `"A100"`, or `"A100_80GB"` (note: Dataproc Serverless Spark is CPU-only). When `compute.machine_type` is `"auto"`, the platform automatically resolves a valid GPU-attached GCE machine shape (such as `g2-standard-8` for `L4` or `a2-highgpu-1g` for `A100`).

### Can I mix multiple runtimes in a single run?
**Yes.** Set a default `compute.runtime` (for example `"spark"`) and override specific model families under `compute.families.<family>.runtime` (for example routing `deep_learning` to `"gce"` with an `"L4"` GPU and `ml` to `"vertex"`). The DAG router launches one parallel job per active model family so fast CPU jobs tear down immediately without waiting for GPU jobs.

### How is `run_id` computed, and why does it ignore `project_id` and `machine_type = "auto"`?
`run_id` (`<slug>-<12hex>`, implemented in [`registry/ids.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/registry/ids.py)) is a deterministic SHA-256 content hash of the **semantic** forecast specification: source data, horizon, frequency, models, metrics, backtest windows, reconciliation, and ensemble configuration. Operational plumbing (`compute.project_id`, `compute.region`, and `"auto"` machine/worker sizing) is excluded so the same logical forecast produces the same `run_id` across dev/stage/prod projects and supports idempotent `--resume` and deduplicated SQL views.

---

## Cost & operations

### What does a forecast run cost, and does anything bill when idle?
By default (`create_composer = false`, `create_gke = false`), **nothing bills compute when idle** — only standard BigQuery table storage, Cloud Storage buckets, and the Artifact Registry container image remain between runs (estimated pennies to a few dollars per month). During a run, you pay only for the ephemeral compute and BigQuery bytes used by that run (estimated `~\$0.05–\$0.50` for a 100-series smoke test, `~\$2–\$10` for a 10k-series multi-family benchmark, and `~\$5–\$35` for a 100k-series production run). See **[Cost estimates & controls](./cost_estimates.md)** for full service-by-service details.

### Does Cloud Composer 3 bill when no DAG is running?
**Yes.** Cloud Composer 3 (`create_composer = true`, off by default) is an always-on managed Airflow environment with an estimated ongoing cost of `~\$300–\$400/month` while provisioned. Turn it off anytime by setting `create_composer = false` in `terraform/main/terraform.tfvars` and running `terraform apply`. Note that deleting a Composer environment leaves behind its Cloud Storage bucket (`<region>-<env-name>-<hash>-bucket`), which you should delete with `gcloud storage rm -r` if you no longer need the DAG files or logs.

### How do I repair a single failed `(ts_id, model_type)` cell without re-running 100,000 series?
Use the surgical repair CLI ([`retry_run.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/retry_run.py)):
```bash
uv run python -m scale_forecasting.retry_run --run-id <RUN_ID>
```
It queries `forecast_metadata` for rows where `status = 'FAILED'` (or missing cells), re-executes only those `(ts_id, model_type)` pairs locally or on your chosen runtime, appends the repaired rows under a `-repair` job suffix, and leaves all successful series untouched. The analytical views automatically pick up the repaired rows via `ROW_NUMBER() OVER (PARTITION BY run_id, ts_id, model_type ORDER BY created_at DESC)`.

### How do I clean up old runs or tear down everything?
- **Delete a single run across BigQuery, Cloud Storage, and BQML:**
  ```bash
  uv run python -m scale_forecasting.registry.ops drop-run <RUN_ID> --yes
  ```
- **Reap any orphaned Vertex Ray clusters from interrupted runs:**
  ```bash
  uv run python -m scale_forecasting.registry.ops reap-clusters --yes
  ```
- **Tear down the entire Google Cloud deployment:**
  Run `terraform destroy` in `terraform/main/` (and optionally remove the state bucket in `terraform/bootstrap/`). See **[Operations runbook](./operations.md)**.

---

## Modeling, covariates & extensibility

### Can I bring my own historical, future, and static covariates?
**Yes.** Declare `future_covariates` (known ahead of time, such as promotions or holidays), `past_covariates` (observed only in history, such as weather or foot traffic), and `static_covariates` (series-level attributes, such as store format or region) in `data` inside your `RunConfig`. Each model plugin declares `supports_future_covariates`, `supports_past_covariates`, and `supports_static_covariates` in **[Models reference](./models_reference.md)**, and the worker automatically routes supported features to each model.

### How do I add a custom model or evaluation metric?
- **Custom model:** Inherit from [`BaseModel`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/models/base_model.py) in a single file under `src/scale_forecasting/models/` (starting from [`docs/model_template.py`](https://github.com/statmike/scale-forecasting/blob/main/docs/model_template.py)) and register it in `models/__init__.py`. See **[Adding a model](./adding_a_model.md)**.
- **Custom metric:** Inherit from [`BaseMetric`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/metrics/base_metric.py) in a single file under `src/scale_forecasting/metrics/` (starting from [`docs/metric_template.py`](https://github.com/statmike/scale-forecasting/blob/main/docs/metric_template.py)) and register it in `metrics/__init__.py`. See **[Adding a metric](./adding_a_metric.md)**.
