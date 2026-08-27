# Smoke testing

A **smoke** is a small, real run — ~100 series — that proves one runtime/hardware/ensemble
combination of the platform works end to end against live Google Cloud. The smoke library under
[`configs/smokes/`](https://github.com/statmike/scale-forecasting/tree/main/configs/smokes) covers
every combination once; the driver
([`tests/smokes/smoke_harness.py`](https://github.com/statmike/scale-forecasting/blob/main/tests/smokes/smoke_harness.py))
runs one config through the full lifecycle and checks the result; and this page is both the
**runbook** for executing them and the **living results log** of what has been validated.

The point is confidence: a green smoke means that combination submits, runs, lands its predictions,
scores onto the leaderboard, re-runs idempotently, and reverse-traces to a clickable job — on real
infrastructure, not a mock.

## What each smoke proves

Run them cheap → expensive; each is numbered in that order.

| # | Config | Proves |
|---|--------|--------|
| 01 | `01_serverless_cpu.json` | Spark on Dataproc Serverless, CPU (statistical + ML families) |
| 02 | `02_bq_native.json` | BigQuery-native models (`arima_plus`, `timesfm`) in BigQuery |
| 03 | `03_serverless_gpu.json` | Serverless GPU (deep-learning family on an L4) |
| 04 | `04_cluster_cpu.json` | Spark on an ephemeral Dataproc cluster, CPU |
| 05 | `05_cluster_reuse.json` | Spark reusing a standing Dataproc cluster by name |
| 06 | `06_cluster_gpu.json` | Dataproc cluster GPU (deep-learning family on a T4) |
| 07 | `07_ray_cpu.json` | Ray on Vertex, CPU |
| 08 | `08_ray_gpu.json` | Ray on Vertex, GPU (deep-learning family on a T4) |
| 09 | `09_shared_ray.json` | Several families on one shared Ray cluster (CPU + GPU pools) |
| 10 | `10_mixed_runtimes.json` | Spark + Ray + BigQuery families concurrently under one run_id |
| 11 | `11_ensemble_barrier.json` | Ensembling in barrier mode (blend once after all base jobs) |
| 12 | `12_ensemble_microbatch.json` | Ensembling in microbatch mode (drain series as they complete) |
| 13 | `13_native_format.json` | Reading the **native** BigQuery source table (dual-format) |
| 14 | `14_full_dag.json` | The flagship: all families + native + ensemble, one run_id |
| 15 | `15_airflow_multi_engine.json` | The whole DAG **orchestrated by Composer/Airflow** — three engines (Spark + Ray GPU + BigQuery) under a microbatch ensemble |

Every other smoke reads the managed-Iceberg source table, so 13 gives the native-format read its own
proof; together they validate both source formats.

Smokes 01–14 launch the run directly (`main.run`); smoke 15 is the one that proves the **Airflow
layer** actually orchestrates the same building blocks — see
[Orchestrating on Composer](#orchestrating-on-composer-airflow-smoke) below.

## Prerequisites

- **Identity** — the `SF_*` environment plus Application Default Credentials
  (`gcloud auth application-default login`). The harness **submits** (not just reads), so it needs the
  full set, not only the five `Settings` vars from [running & reviewing](./running_and_reviewing.md):
  `Settings.resolve()` reads `SF_PROJECT_ID` / `SF_CONNECTION` / `SF_WAREHOUSE_URI` (+ optional
  `SF_DATASET_ID` / `SF_REGION`), **and** `BatchInfra.resolve()` reads `SF_CODE_BUCKET` /
  `SF_CONTAINER_IMAGE` / `SF_COMPUTE_SA` / `SF_SUBNETWORK_URI`, **and** the Ray smokes' `RayInfra.resolve()`
  reads `SF_RAY_NETWORK_ATTACHMENT` (or `SF_RAY_NETWORK`) plus optional `SF_RAY_VERSION` /
  `SF_RUNTIME_VERSION`. Wire them straight from Terraform:

  ```bash
  cd terraform/main
  eval "$(terraform output -json | python -c 'import json,sys
  o=json.load(sys.stdin); g=lambda k: o[k]["value"]
  print(f"export SF_PROJECT_ID={g(\"project_id\")}")
  print(f"export SF_CONNECTION={g(\"iceberg_connection\")}")
  print(f"export SF_WAREHOUSE_URI={g(\"warehouse_uri\")}")
  print(f"export SF_DATASET_ID={g(\"dataset_id\")}")
  print(f"export SF_CODE_BUCKET={g(\"code_bucket\")}")
  print(f"export SF_CONTAINER_IMAGE={g(\"runtime_image_repo\")}:latest")
  print(f"export SF_COMPUTE_SA={g(\"compute_sa\")}")
  print(f"export SF_SUBNETWORK_URI={g(\"subnetwork_uri\")}")
  print(f"export SF_RAY_NETWORK_ATTACHMENT={g(\"network_attachment_id\")}")
  print(f"export SF_VENV_ARCHIVE={g(\"venv_archive_uri\")}")
  gpu=g("gpu_image_uri")
  print(f"export SF_GPU_IMAGE={gpu}") if gpu else None')"
  export SF_REGION=us-central1   # or your deploy region
  ```
  `SF_GPU_IMAGE` is exported only when the deploy built the pre-baked GPU cluster image
  (`build_gpu_image = true`); without it, GPU cluster smokes install the driver at cluster-create
  time instead (slower). See [runtime_dependencies.md](./runtime_dependencies.md#gpu-clusters--the-pre-baked-driver-image).
- **Source tables** — both `source_series_iceberg` and `source_series_native` must exist in the
  deployment dataset (they are created by the Terraform + seed step).
- **Deep-learning smokes (03, 06, 08, 09, 10, 14)** — the container must carry the `models` extra so
  `neuralprophet` can fit; the run image built by the deploy already includes it.
- **GPU smokes (03, 06, 08, 09, 10, 14)** — GPU quota in the run's region: L4 for Serverless
  (03, 14), T4 for cluster/Ray (06, 08, 09, 10).
- **GPU cluster smoke (06)** — a Dataproc *cluster* is a bare set of VMs, so the host NVIDIA driver
  is the deploy's to supply. With `build_gpu_image = true` the deploy bakes a custom VM image with
  the driver pre-baked and exports it as `SF_GPU_IMAGE`, so the cluster boots ready. Without it the
  cluster installs the driver at create time — a source compile that can exceed Dataproc's
  cluster-create window. Prefer the pre-baked image for this smoke.
- **Cluster capacity failover** — GPU (and sometimes CPU) capacity is *zonal* and can stock out
  transiently even with quota to spare, so an ephemeral cluster create walks a candidate list: the
  deploy region's auto-zone first (unchanged), then that region's other zones, then — opt-in — other
  regions. The candidate list is the user-editable
  [`configs/compute_fallback.json`](https://github.com/statmike/scale-forecasting/blob/main/configs/compute_fallback.json)
  (or `SF_COMPUTE_FALLBACK`), prepopulated with the US regions/zones; same-region zone failover works
  with no edits, while cross-region failover activates only for regions you give a `subnetwork_uri`
  (a subnet with Cloud NAT + Private Google Access in that region). Non-capacity errors still fail
  fast, and the list never affects `run_id`.
- **Cluster smokes (04, 05, 06)** — a Dataproc **cluster** can't use the custom container, so it
  gets its dependencies from the **self-contained venv archive** instead. `SF_VENV_ARCHIVE` must point
  at it (the `venv_archive_uri` Terraform output, wired above); the deploy's Cloud Build packs + uploads
  it alongside the image. See [runtime_dependencies.md](./runtime_dependencies.md#dataproc-cluster--self-contained-venv-archive).
- **Cluster-reuse smoke (05)** — a standing Dataproc cluster named `sf-smoke-cluster` must already
  exist; the run submits to it rather than creating one. Delete it when the campaign is done.

## Running one smoke

Each smoke is one command; it blocks until the run is terminal, then prints a report.

```bash
.venv/bin/python tests/smokes/smoke_harness.py configs/smokes/01_serverless_cpu.json
```

The driver walks the lifecycle a reviewer would run by hand:

1. **dry** (`plan_run`) — resolve the run_id, the per-runtime model split, the fanout, and the
   exists-vs-new verdict; touches no GCS.
2. **stage** (`stage_run`) — upload the config (and, for Spark, the code zip) and write the
   reproducibility manifest `runs/<run_id>.plan.json`; capture the runnable launch commands.
3. **run** (`main.run`) — submit every family on its runtime under one run_id and block to terminal.
4. **verify** — read the registry views back and check the run reached `COMPLETED`, every expected
   family ran and succeeded with a real platform job id, and every configured model (plus the
   ensembles, when enabled) scored onto the leaderboard.
5. **rerun / collision** — re-run the same config with no `--force`; it must resolve the **same**
   run_id and, via append-only + dedupe-on-read, leave the leaderboard counts unchanged.
6. **reverse-trace** — print each family's stored `system_job_id` and the service it resolves to
   (Dataproc batch / Dataproc cluster job / Vertex Ray submission / BigQuery job).

Flags:

- `--force` — bump the attempt (a fresh job under the same run_id), instead of the default re-run.
- `--no-rerun` — run once and skip the rerun/collision check.

The command exits non-zero if any check fails, so it drops straight into CI or a `for` loop.

## Orchestrating on Composer (Airflow smoke)

Smokes 01–14 launch the run directly. Smoke 15 instead drives the run **through Composer**, proving
the emitted Airflow DAG orchestrates the same building blocks the direct smokes launch by hand. It
takes the identical config, resolves its `run_id`, stages its artifacts, emits `dag_<run_id>.py`
([`airflow_emit.emit_airflow_dag`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/airflow_emit.py)),
imports it into the environment, triggers it, and waits for the run to land in the registry — the
same terminal signal the direct harness polls. Because the `run_id` is a digest of the config, a
Composer-orchestrated run writes the registry under the **identical** id a local run would, so
success is direct proof of *same code local↔Composer*. Verification reuses the direct harness's
checkers (`verify_run_jobs` / `verify_leaderboard` / `verify_predictions`), holding both smokes to
one standard.

Two levels of proof, cheap → expensive:

- **Parse-under-Airflow (always-on, free).** The offline emitter tests `compile()` the DAG and walk
  its `ast`; they never import Airflow. `tests/unit/test_airflow_dagbag.py` closes that gap — it
  loads an emitted DAG through a real `airflow.models.DagBag` and asserts no import errors, catching
  operator-kwarg / import-chain mistakes the string checks can't. Airflow is heavy and conflicts
  with our torch/ray/spark pins, so it is **not** in `uv.lock`; the test is marked `@airflow` and
  skips cleanly when Airflow is absent (like `@spark`/`@ray`). A dedicated CI job (`airflow-parse`)
  installs Airflow isolated against its official constraints and runs it on every push, so it can
  never destabilize the main offline gate.
- **Live on Composer (gated).** Provisioning Composer costs money and time, so it is gated behind
  `create_composer=true` and run only on explicit go.

**How the code and config reach Composer.** A Composer worker is just another *launch point* — it
runs the same driver code a local launch does. It never runs the model code (that ships per-job as
the `src/` zip and executes in Dataproc/Ray/BigQuery), but it does need three things to *be* a launch
point, and none of them come from GitHub at runtime:

- **The `SF_*` identity** — set as environment variables on the environment, wired from the Terraform
  outputs (`terraform/main` builds this map and passes it to the composer module). This is infra, so
  `terraform apply -var create_composer=true` sets it.
- **The submit-side dependencies** — the `pypi_packages` the driver imports to talk to the services
  (BigQuery registry, Dataproc/Vertex submit, the Ray `JobSubmissionClient` handshake). **Not** the
  model stack (torch/darts/neuralprophet/pyspark) — that runs in-service. Also set by the apply.
- **The code** (`src/`) — delivered by `make composer-sync`, which `gsutil rsync`s **this working
  tree's** `src/` into the environment's plugins prefix (on the workers' `PYTHONPATH`). The worker
  then imports the driver **and** re-zips that same `src/` to ship code to the jobs — so your
  clone/fork/customizations flow through with no image rebuild. GitHub is only the origin: you pull
  and modify locally, and `composer-sync` is what carries the result to the environment. This is a
  bootstrap step (code changes more often than infra), not baked into Terraform.

**Prerequisite — a running Composer environment.** Composer is off by default
([`terraform/main/modules/composer`](https://github.com/statmike/scale-forecasting/tree/main/terraform/main/modules/composer)).
Provision it, deliver the code, run the smoke, then turn the meter back off:

```bash
# 1. Provision (~25 min build; ~$300–400/mo while up — the smallest env). This also sets the
#    workers' SF_* env + submit-side pypi packages.
cd terraform/main
terraform apply -var create_composer=true

# 2. Deliver this working tree's src/ to the workers (the code-delivery step; re-run after edits).
cd ..
make composer-sync

# 3. Run the Airflow smoke end-to-end (stage → emit → import → trigger → wait → verify).
.venv/bin/python tests/smokes/airflow_smoke.py configs/smokes/15_airflow_multi_engine.json \
    --composer-env scale-forecasting --location "$SF_REGION"

# 4. Stop the meter — destroys just the environment; data/registry/buckets untouched.
cd terraform/main
terraform apply -var create_composer=false
```

The deep-learning family on Ray GPU over 200 series × 10 folds is the long pole and can approach the
~60-min bearer-token expiry (a known limit); the config is the knob — drop `backtest.n_folds` or move
`deep_learning` to CPU if a live run runs long.

## Verifying by hand in BigQuery

The harness reads the same views you can query directly for the run_id it prints:

```sql
-- how did the run go, and how efficiently?
SELECT * FROM `PROJECT.DATASET.v_run_summary`     WHERE run_id = 'RUN_ID';
-- which families ran, on what runtime/hardware, and their real job ids (reverse-trace)
SELECT * FROM `PROJECT.DATASET.v_run_jobs`        WHERE run_id = 'RUN_ID';
-- which model won, per run
SELECT * FROM `PROJECT.DATASET.v_model_leaderboard` WHERE run_id = 'RUN_ID' ORDER BY mean_wape;
```

The `system_job_id` in `v_run_jobs` is the real, console-resolvable id for each family's job — paste
it into the Dataproc / Vertex / BigQuery job history to click straight through.

## Offline guardrails

The config library is checked in the offline gate so a broken config never reaches a live submit:

- `tests/smokes/test_smoke_configs.py` — every config loads, validates, and plans a DAG; the library
  still spans every runtime/hardware/ensemble combination; both source formats are exercised.
- `tests/smokes/test_harness.py` — the harness's verify/trace logic (what decides PASS/FAIL) is
  unit-tested with fixture rows.
- `tests/smokes/test_airflow_smoke.py` — the Airflow smoke's pure command-builders (which `gcloud
  composer` argv it shells out) and its `dag_id` derivation, so a typo can't point the live smoke at
  the wrong environment or DAG.

Both run in the standard offline suite (`pytest -m "not gcp and not spark and not ray"`). The
parse-under-Airflow test (`tests/unit/test_airflow_dagbag.py`) is separate: marked `@airflow`, it
skips unless Airflow is installed and runs in its own CI job (see
[Orchestrating on Composer](#orchestrating-on-composer-airflow-smoke)).

## Results log

One row per live execution. Fill it in as the campaign runs.

| # | Config | Date | run_id | Status | Notes |
|---|--------|------|--------|--------|-------|
| 01 | `01_serverless_cpu.json` | 2026-08-22 | `smoke-01-serverless-cpu-2ca2c0f48bd0` | ✅ PASS | 2 families (statistical + ML) on Serverless CPU; 3 models × 100 cells; `mean_wape` null (no backtest); rerun no-op'd, reverse-traced to Dataproc batches. Surfaced + fixed the rerun-collision guard. |
| 02 | `02_bq_native.json` | 2026-08-22 | `smoke-02-bq-native-7b34cfd9eb98` | ✅ PASS | Native family (arima_plus + timesfm) in BigQuery; 5,600 predictions (100×28×2), 200 metadata rows; `mean_wape` null (no backtest). Surfaced that BQML `CREATE MODEL` can't time-travel a BigLake Iceberg source — native path now reads Iceberg un-pinned (see [CONSIDERATIONS.md](https://github.com/statmike/scale-forecasting/blob/main/CONSIDERATIONS.md), C1). |
| 03 | `03_serverless_gpu.json` | 2026-08-22 | `smoke-03-serverless-gpu-a1adfc48d5d3` | ✅ PASS | Deep-learning family (`neuralprophet`) on Serverless GPU (L4); 2,800 predictions (100×28), 100 metadata rows; `mean_wape` null (no backtest). Fixed the Serverless GPU property set (Dataproc-managed accelerator + premium compute/disk tiers; the Spark-level GPU scheduling props are unsupported). Reverse-traced to the Serverless batch (attempt `a2` after an env-policy failure on `a1`). |
| 04 | `04_cluster_cpu.json` | 2026-08-23 | `smoke-04-cluster-cpu-88fddc72b8a1` | ✅ PASS | Ephemeral Dataproc cluster, CPU; 2 families (statistical + ML), 3 models (theta, holtwinters, xgboost) × 100 cells; `mean_wape` null (no backtest); rerun no-op'd, both families reverse-traced to Dataproc cluster jobs. The cluster gets its model libraries from the self-contained venv archive, delivered by a **node init action** that unpacks it to an absolute path on every node — a job-attached archive reaches only executors, never the client-mode driver, so the driver needs the venv on-node. |
| 05 | `05_cluster_reuse.json` | 2026-08-23 | `smoke-05-cluster-reuse-2a7edf806a52` | ✅ PASS | Standing Dataproc cluster reused by name (create/teardown skipped); 2 families (statistical + ML), theta + xgboost × 100 cells; `mean_wape` null (no backtest); rerun no-op'd, reverse-traced to cluster jobs. The standing cluster carries the same venv init action, so reuse runs the identical locked env. |
| 06 | `06_cluster_gpu.json` | 2026-08-24 | `smoke-06-cluster-gpu-a510512f507a` | ✅ PASS | Dataproc cluster GPU (T4), deep-learning family (`neuralprophet`) × 100 cells; `mean_wape` null (no backtest); rerun no-op'd (same id, board unchanged), reverse-traced to a Dataproc cluster job. Boots from the pre-baked NVIDIA-driver VM image (`build_gpu_image`/`SF_GPU_IMAGE`, no driver init action) with Secure Boot disabled. **Transient T4 capacity in `us-central1-a` was rescued by the zone failover** (`configs/compute_fallback.json`): several CREATE ops failed on capacity, then the create landed and ran in `us-central1-b` — no config edits, same run_id. Cluster torn down after (no orphan spend). |
| 07 | `07_ray_cpu.json` | _pending_ | | | |
| 08 | `08_ray_gpu.json` | _pending_ | | | |
| 09 | `09_shared_ray.json` | _pending_ | | | |
| 10 | `10_mixed_runtimes.json` | _pending_ | | | |
| 11 | `11_ensemble_barrier.json` | _pending_ | | | |
| 12 | `12_ensemble_microbatch.json` | _pending_ | | | |
| 13 | `13_native_format.json` | _pending_ | | | |
| 14 | `14_full_dag.json` | _pending_ | | | |
| 15 | `15_airflow_multi_engine.json` | _pending_ | | | Orchestrated by Composer/Airflow, not direct-launch — gated on `create_composer=true`. |
