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

- **Identity** — the same `SF_*` environment the run loop uses (see
  [running & reviewing](./running_and_reviewing.md)) plus Application Default Credentials
  (`gcloud auth application-default login`). The harness resolves `Settings` from this environment.
- **Source tables** — both `source_series_iceberg` and `source_series_native` must exist in the
  deployment dataset (they are created by the Terraform + seed step).
- **Deep-learning smokes (03, 06, 08, 09, 10, 14)** — the container must carry the `models` extra so
  `neuralprophet` can fit; the run image built by the deploy already includes it.
- **GPU smokes (03, 06, 08, 09, 10, 14)** — GPU quota in the run's region: L4 for Serverless
  (03, 14), T4 for cluster/Ray (06, 08, 09, 10).
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
| 01 | `01_serverless_cpu.json` | _pending_ | | | |
| 02 | `02_bq_native.json` | _pending_ | | | |
| 03 | `03_serverless_gpu.json` | _pending_ | | | |
| 04 | `04_cluster_cpu.json` | _pending_ | | | |
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
