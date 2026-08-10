# scale-forecasting

**Massively-parallel time-series forecasting on Google Cloud — Spark, Ray, and BigQuery, one config away.**

Forecast tens of thousands of time series in parallel, backtest and ensemble many
methods, and capture every run's lineage in BigQuery — from a local notebook or from
Airflow, with the *same* code. Deploy the whole thing into a fresh project with one
`terraform apply`, complete with 100k example series to run against immediately.

> **Status.** The local dev loop and the GCP Terraform deploy are both working end-to-end
> (Spark, Ray, and BigQuery engines all run against a live deployment). An architecture
> diagram lands in the final polish phase.

---

## Why it exists

Large-scale forecasting usually means a pile of bespoke cluster code that only its
author understands. This project is the opposite bet: a small, readable codebase —
**one capability per file** — that still scales to 100k+ series, so a data scientist
can open any file, understand it in one read, and fork it to their needs.

## What it does

**Univariate time-series forecasting at scale** — fit many methods to tens of thousands of series in
parallel, with backtesting, custom holidays, and ensembling built in. One JSON config describes the
whole run.

- **Three runtimes, your pick.** Two **Python** runtimes — Dataproc Serverless (**Spark**) and
  **Ray** on Vertex AI — run the identical per-series unit of work (pick one per run); **BigQuery**
  runs its SQL-only native models **in parallel** with either, no Python compute. See
  [Methods](#methods) and [The three runtimes](#the-three-runtimes).
- **Ray packs fractional GPUs.** Ray's reason to exist here is **NeuralProphet on fractional GPUs** —
  many series share one T4 (auto-profiled `gpu_fraction`), so a deep-learning model scales without a
  GPU per series. See [The three runtimes](#the-three-runtimes).
- **Backtesting + ensembling out of the box** — expanding/sliding folds with a full metric panel,
  and both calculated and learned ensembles.
- **Custom holidays and transforms** — add holiday country codes and a target transform (`log1p` /
  `boxcox`) from config; the generic exog seam is there when you outgrow univariate.
- **BigQuery lineage** — every run's config, per-model metrics, forecasts, and artifact links are
  captured in native BigQuery tables, written via the **Storage Write API** for high-speed updates
  (config, telemetry, and quantiles as native `JSON` columns). Layout:
  [output_schemas.md](./docs/output_schemas.md).
- **Pick your input storage** — the example series ship in **both** managed-Iceberg and
  native BigQuery, so you can benchmark the identical data on either format by name.
- **Config-driven** — one JSON file describes the whole run; the same file runs locally
  and under Composer. Every setting: [configuration_reference.md](./docs/configuration_reference.md).

## Methods

One model per file (`src/scale_forecasting/models/`); each ends in `register(...)` so it shows up in
`playground --list` automatically. Add your own with [`docs/adding_a_model.md`](./docs/adding_a_model.md).

| Model | Runtime | Family | Notes |
|-------|---------|--------|-------|
| [`theta`](./src/scale_forecasting/models/theta.py) | Python | statistical | Simple, strong baseline. |
| [`holtwinters`](./src/scale_forecasting/models/holtwinters.py) | Python | statistical | Holt-Winters exponential smoothing. |
| [`sarimax`](./src/scale_forecasting/models/sarimax.py) | Python | statistical | Seasonal ARIMA; supports exog. |
| [`ucm`](./src/scale_forecasting/models/ucm.py) | Python | statistical | Unobserved-components (structural) state-space. |
| [`stl_bagging`](./src/scale_forecasting/models/stl_bagging.py) | Python | statistical | STL decomposition + bagged base forecasts. |
| [`prophet`](./src/scale_forecasting/models/prophet_model.py) | Python | statistical | Additive trend/seasonality/holidays. |
| [`neuralprophet`](./src/scale_forecasting/models/neuralprophet_model.py) | Python | deep learning | **The GPU model** — fractional-GPU packing on Ray. |
| [`lightgbm`](./src/scale_forecasting/models/lightgbm_model.py) | Python | ML | Gradient boosting on lag/calendar features. |
| [`xgboost`](./src/scale_forecasting/models/xgboost_model.py) | Python | ML | Gradient boosting on lag/calendar features. |
| [`arima_plus`](./src/scale_forecasting/models/bigquery_native.py) | BigQuery | native | SQL-only `ARIMA_PLUS`; runs ∥ the Python runtime. |
| [`timesfm`](./src/scale_forecasting/models/bigquery_native.py) | BigQuery | native | SQL-only foundation-model forecaster. |

Plus **ensembles** across the base models (calculated: `mean`/`median`/`inverse_error`; learned:
`nnls`/`ridge`/`xgb`) — see [configuration_reference.md](./docs/configuration_reference.md#ensemble--ensembleconfig).

## Architecture

A run is one validated JSON config. It selects the data, the model list, one Python
runtime, backtest and ensemble settings — and is persisted verbatim into the run
registry, so the config *is* the experiment record.

<a name="the-three-runtimes"></a>

- **The three runtimes.** The Python model suite runs on **Spark _xor_ Ray** (one per run).
  BigQuery-native models are additive and run **in parallel** with either. So a run is
  "Spark + optional BQ models" or "Ray + optional BQ models" — never Spark + Ray.
  - **Spark** (Dataproc Serverless) is the 100k CPU workhorse, with three fan-out **methods** you
    pick by config — same answers, different speed:

    | `spark_method` | Fans out by | Best for | Watch out |
    |----------------|-------------|----------|-----------|
    | `explode` (default) | one task per `(series, model)` cell | max parallelism at scale — finishes in ~its slowest *cell* | many small tasks |
    | `multi` | one child `explode` batch per model family | isolating a heavy model family | more orchestration |
    | `naive` | one task per series (its models run sequentially) | tiny runs / a baseline to beat | drags on its slowest *series* — the straggler anti-pattern |

    Notebooks [05](./notebooks/05_spark_naive.ipynb)/[06](./notebooks/06_spark_multi.ipynb) isolate
    `naive`/`multi`; [07](./notebooks/07_scale_review.ipynb) compares all three at 100k.
  - **Ray** (on Vertex AI) is for **fractional-GPU packing**: NeuralProphet's per-series fit is small,
    so many series share one T4 via an auto-profiled `gpu_fraction` — a deep-learning model at fleet
    scale without a GPU per series. (For CPU-only work at 100k, Spark is the workhorse; Ray earns its
    place on the GPU path.)
  - **BigQuery** runs `arima_plus` / `timesfm` in SQL, in parallel, under the same `run_id`.
- **The one unit of work.** `worker.run_cell(series, model, cfg) -> CellResult` fits,
  (optionally) backtests, and predicts one `(series, model)` cell. The *same* function
  runs locally, inside a Spark Pandas UDF, and inside a Ray task. Engines differ only in
  how they fan it out and collect results — that's what makes "same code everywhere" real.
- **Ray vs Spark, for the PySpark crowd.** If you fan `(series, model)` work out today with
  `applyInPandas` + pandas UDFs, Ray runs the **identical** `worker.run_cell` (G1) over the
  **same** input table (Iceberg or native — read through the BigQuery Storage Read API), so
  the storage format is transparent to engine code. You get Ray ∥ BigQuery under one `run_id`
  with no Spark session to stand up or tune — and because the Ray ecosystem can also host
  Spark workloads (via RayDP), a team can consolidate many frameworks on one cluster.
  *(Today the Ray engine reads via the BQ Storage Read API; Ray-native Iceberg reads
  (`ray.data`) and Spark-on-Ray (RayDP) are a documented future direction, not a current
  claim.)*
- **Scale without a bottleneck.** Workers return data, not RPCs; results are written to
  BigQuery in bulk (Storage Write API). Parallelism is bounded by compute, not a tracking
  server's QPS.
- **Lineage.** Three native-BigQuery tiers — `run_registry` (config) →
  `forecast_metadata` (metrics + GCS artifact links) → `forecast_predictions` (values),
  plus `backtest_oof` for learned ensembling. These run-collection tables are always native
  (native `JSON` columns, `WRITE_TRUNCATE` reseed) and are written via the **Storage Write API**;
  the *input* table ships in both Iceberg and native so you can compare storage formats on the same
  run shape. Full column-by-column layout: [output_schemas.md](./docs/output_schemas.md).

Read `src/scale_forecasting/config.py` (the run contract) and
`src/scale_forecasting/worker.py` (the unit of work) to see the whole shape.

## Quickstart

No GCP needed — run a real model on sample data in under a minute.

```bash
uv sync                                            # install into a local venv
uv run python -m scale_forecasting.playground --list           # see every model
uv run python -m scale_forecasting.playground --model theta --backtest
```

That runs the **same** `worker.run_cell` the cluster runs, on a small generated
panel, and prints the forecast horizon plus the backtest metric panel. For an
interactive version — pick a model, plot the forecast and its interval — open
[`notebooks/model_playground.ipynb`](./notebooks/model_playground.ipynb).

**Add your own model** in one file: copy [`docs/model_template.py`](./docs/model_template.py)
into `src/scale_forecasting/models/`, add one import line, and it shows up in the
list above automatically. Full walkthrough: [`docs/adding_a_model.md`](./docs/adding_a_model.md).

**Demo notebooks** (run + review against a live deployment) live in [`notebooks/`](./notebooks):
[`01_spark_via_connect`](./notebooks/01_spark_via_connect.ipynb) drives the Spark UDF fan-out over a
Dataproc **Spark Connect** endpoint; [`02_bigquery_native`](./notebooks/02_bigquery_native.ipynb)
runs the BigQuery-native models; [`03_combo_and_ensemble`](./notebooks/03_combo_and_ensemble.ipynb)
runs Spark ∥ BigQuery under one `run_id` with ensembles, then reviews base + ensemble models side by
side on `v_model_leaderboard`; [`04_ray_on_vertex`](./notebooks/04_ray_on_vertex.ipynb) runs the
Python-runtime models on a fixed-size Ray-on-Vertex cluster ∥ the BigQuery natives (job submission
works from any authenticated client — local or in-GCP — because the cluster is provisioned on a
PSC-I network attachment with a dashboard-capable head node, wired by the Terraform network module).

The last two run notebooks isolate the remaining Spark fan-out **methods** on the same 100 series:
[`05_spark_naive`](./notebooks/05_spark_naive.ipynb) shows the `naive` straggler anti-pattern (a
series' models run sequentially in one task), and [`06_spark_multi`](./notebooks/06_spark_multi.ipynb)
shows `multi` fanning out one child `explode` batch per model family — all under **one** `run_id`.

Finally, [`07_scale_review`](./notebooks/07_scale_review.ipynb) runs nothing — point it at one
`run_id` per approach (e.g. the four `configs/*_100k.json` runs submitted via
`python -m scale_forecasting.submit`) and it renders the **cross-approach comparison**: wall-clock
and provisioning overhead from `v_run_summary`, and accuracy parity (same model, same answer across
engines — G1) from `v_model_leaderboard`.

Every notebook has a one-click **Run in Colab Enterprise** header — the Terraform-deployed runtime
templates carry the `SF_*` run identity in their env, so it's open → pick a runtime → **Run all**,
no environment cell. The same notebooks run green headless via the acceptance harness
(`pytest -m gcp tests/integration/test_notebook_acceptance.py`, or
`python -m scale_forecasting.notebook_acceptance`), which the deploy and any notebook change are
verified against. See [`docs/notebook_runtimes.md`](./docs/notebook_runtimes.md).

## Documentation

- [`docs/running_and_reviewing.md`](./docs/running_and_reviewing.md) — submit a run (Spark / Ray /
  BigQuery), watch it land, review the leaderboard, and re-ensemble a completed run — the full
  operator loop with every entrypoint and its flags.
- [`docs/configuration_reference.md`](./docs/configuration_reference.md) — every config field, its
  type, default, and constraint, section by section (the run *is* the config).
- [`docs/output_schemas.md`](./docs/output_schemas.md) — the output tables' layout: every column of
  `run_registry` / `forecast_metadata` / `forecast_predictions` / `backtest_oof`, what it collects,
  and the two analyst views over them.
- [`docs/notebook_runtimes.md`](./docs/notebook_runtimes.md) — which Python version each notebook
  needs (3.11 vs 3.12) and how it behaves under each, locally (ADC) and on Colab Enterprise, plus the
  two runtime templates Terraform ships, the one-click open path, and the headless acceptance harness.
- [`docs/editing_code_without_rebuilding.md`](./docs/editing_code_without_rebuilding.md) — why a code
  edit ships on the next run with **no** image rebuild, and the loop you actually use.
- [`docs/adding_a_model.md`](./docs/adding_a_model.md) — add a model in one file.
- [`docs/deploying_on_gcp.md`](./docs/deploying_on_gcp.md) — a reviewer's guide to the Terraform.

## Deploy on GCP

The whole platform deploys into a Google Cloud project with Terraform, in **two stages**:

```bash
# Stage 1 — bootstrap (run once): creates the project (optional) + the Terraform state bucket.
cd terraform/bootstrap
cp terraform.tfvars.example terraform.tfvars      # edit: project_id, billing_account, org_id
terraform init && terraform apply
terraform output backend_config                   # note the state bucket for stage 2

# Stage 2 — main: everything else (dataset, buckets, SAs, network, connection, budget).
cd ../main
cp terraform.tfvars.example terraform.tfvars      # edit: project_id, billing_account
terraform init -backend-config="bucket=<project_id>-tfstate"
terraform plan                                    # review — nothing is created until apply
terraform apply
```

That single first apply also **builds the shared Spark/Ray runtime image for you** (Cloud Build, a
few minutes) and pushes it to Artifact Registry, so the seed job and every forecast engine have an
image to run — no separate build step. It's content-addressed on `docker/`, so it rebuilds only when
the Dockerfile or locked dependencies change, never on a code edit. Set `build_image = false` if you
build and push the image yourself.

The infrastructure is **effectively free at rest** — empty buckets, an empty dataset, service
accounts, and network plumbing cost nothing until compute runs. Two things cost money:

- **The example dataset is created on the first apply** (`run_seed = true`, **on by default**): a
  one-time Dataproc Serverless batch generates **100,000 deterministic time series** and writes them
  to **both** source tables — `source_series_iceberg` (managed Apache Iceberg) and
  `source_series_native` (native BigQuery) — from a *single generated panel*, so the series are
  identical across formats and you can benchmark storage on the same data. This is the
  "solution-in-a-box" promise: a fresh deploy has data to forecast against immediately. It's cheap —
  the 100k seed is measured at **~$0.15 and ~8.5 min of compute** — and content-addressed, so it runs
  once and does **not** re-run on later applies unless you change the series count / label / seed
  code. Set `run_seed = false` to skip it (bring your own source table), or `seed_num_series = 100`
  to smoke-test first.
- **Composer** (`create_composer`) is **off by default** — the only real at-rest cost
  (~$300–400/mo); turn it on for scheduled DAG runs.

**Greenfield or brownfield.** Defaults create everything (the 5-minute path). For a locked-down org,
flip `create_service_accounts` / `create_network` / `enable_apis` off and pass your existing SAs,
subnet, and pre-enabled APIs in by variable — the modules then create nothing and thread your values
through.

**Reviewing the Terraform before you run it?** [`docs/deploying_on_gcp.md`](./docs/deploying_on_gcp.md)
is a full walkthrough: what each module builds, which GCP services it uses and how, **why each
permission is granted and who uses it** (including the three custom least-privilege IAM roles), and
the greenfield-vs-brownfield toggles. The operator runbook (exact commands, cost notes) is
[`terraform/README.md`](./terraform/README.md).

## License

Apache-2.0 — see [`LICENSE`](./LICENSE).
