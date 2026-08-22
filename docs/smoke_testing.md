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

Every other smoke reads the managed-Iceberg source table, so 13 gives the native-format read its own
proof; together they validate both source formats.

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
  print(f"export SF_VENV_ARCHIVE={g(\"venv_archive_uri\")}")')"
  export SF_REGION=us-central1   # or your deploy region
  ```
- **Source tables** — both `source_series_iceberg` and `source_series_native` must exist in the
  deployment dataset (they are created by the Terraform + seed step).
- **Deep-learning smokes (03, 06, 08, 09, 10, 14)** — the container must carry the `models` extra so
  `neuralprophet` can fit; the run image built by the deploy already includes it.
- **GPU smokes (03, 06, 08, 09, 10, 14)** — GPU quota in the run's region: L4 for Serverless
  (03, 14), T4 for cluster/Ray (06, 08, 09, 10).
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

Both run in the standard offline suite (`pytest -m "not gcp and not spark and not ray"`).

## Results log

One row per live execution. Fill it in as the campaign runs.

| # | Config | Date | run_id | Status | Notes |
|---|--------|------|--------|--------|-------|
| 01 | `01_serverless_cpu.json` | 2026-08-22 | `smoke-01-serverless-cpu-2ca2c0f48bd0` | ✅ PASS | 2 families (statistical + ML) on Serverless CPU; 3 models × 100 cells; `mean_wape` null (no backtest); rerun no-op'd, reverse-traced to Dataproc batches. Surfaced + fixed the rerun-collision guard. |
| 02 | `02_bq_native.json` | 2026-08-22 | `smoke-02-bq-native-7b34cfd9eb98` | ✅ PASS | Native family (arima_plus + timesfm) in BigQuery; 5,600 predictions (100×28×2), 200 metadata rows; `mean_wape` null (no backtest). Surfaced that BQML `CREATE MODEL` can't time-travel a BigLake Iceberg source — native path now reads Iceberg un-pinned (see [CONSIDERATIONS.md](../CONSIDERATIONS.md), C1). |
| 03 | `03_serverless_gpu.json` | 2026-08-22 | `smoke-03-serverless-gpu-a1adfc48d5d3` | ✅ PASS | Deep-learning family (`neuralprophet`) on Serverless GPU (L4); 2,800 predictions (100×28), 100 metadata rows; `mean_wape` null (no backtest). Fixed the Serverless GPU property set (Dataproc-managed accelerator + premium compute/disk tiers; the Spark-level GPU scheduling props are unsupported). Reverse-traced to the Serverless batch (attempt `a2` after an env-policy failure on `a1`). |
| 04 | `04_cluster_cpu.json` | 2026-08-22 | `smoke-04-cluster-cpu-88fddc72b8a1` | ⚠️ BLOCKED | Shared-cluster orchestration validated (one ephemeral cluster, both families' jobs on it, single teardown — the DELETING race is fixed). But **0 predictions landed**: the Dataproc **cluster** runtime installs no model libraries (the container delivers deps for serverless + Ray only; clusters can't use it), so every fit errors instantly (`fit_seconds=0.0`, metadata written, no forecast). Fix in progress: implement the packed-venv archive (`compute.spark_deps`, the design default). Blocks 04/05/06. |
| 05 | `05_cluster_reuse.json` | _pending_ | | | |
| 06 | `06_cluster_gpu.json` | _pending_ | | | |
| 07 | `07_ray_cpu.json` | _pending_ | | | |
| 08 | `08_ray_gpu.json` | _pending_ | | | |
| 09 | `09_shared_ray.json` | _pending_ | | | |
| 10 | `10_mixed_runtimes.json` | _pending_ | | | |
| 11 | `11_ensemble_barrier.json` | _pending_ | | | |
| 12 | `12_ensemble_microbatch.json` | _pending_ | | | |
| 13 | `13_native_format.json` | _pending_ | | | |
| 14 | `14_full_dag.json` | _pending_ | | | |
