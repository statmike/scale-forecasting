# Validation ledger

**What has actually been proven on live infrastructure, and on which architecture.**

This is the single source of truth for live-validation evidence. Nothing else — not a commit
message, not a chat log, not a code comment — counts as a record that something was proven. If a
live run isn't in this file, treat it as never having happened.

## Why this file exists

A live result is only meaningful *relative to the architecture it ran on*. On 2026-08-25 the full
smoke suite passed; three days later `822ae25` replaced the Ray dependency-delivery path wholesale
(custom container image → stock prebuilt image + `runtime_env` uv plugin). Four of those passing
smokes had exercised the path that no longer existed, but the results were recorded in commit
bodies with no note of what they depended on — so two of them silently became claims about a
system that had been deleted, and were still being cited as proof days later.

The fix is structural, not a matter of discipline: every entry declares the **architecture axes**
it depends on and the value each axis had at proof time. When an axis changes, every entry pinned
to the old value is mechanically stale, and the tripwire
(`tests/unit/test_validation_ledger.py`, part of the offline gate) refuses to let it keep claiming
`CURRENT`. You either re-run it or mark it `STALE` — you cannot quietly do neither.

## Architecture axes

The facts a live result depends on. Change one of these in the code and every entry recording its
old value goes stale by definition.

| Axis | Current value | Set by | Previous value |
|------|---------------|--------|----------------|
| `ray_deps` | `stock-image+uv-runtime-env` | `822ae25` (2026-08-28) | `custom-container-image` |
| `cluster_deps` | `packed-venv-init-action` | `eae3874` | job-attached archive (driver never saw it) |
| `serverless_deps` | `container-image` | long-standing | — |
| `gpu_cluster_image` | `prebaked-driver-image` | `254fe4f` | driver install via init action |
| `native_source_pin` | `unpinned-all-sources` | `9af322a` (2026-08-25) | `unpinned-iceberg-only` |
| `python` | `3.11` | `515ecb0` | mixed per surface |

`native_source_pin` governs **native BigQuery table** reads on the BQML `CREATE MODEL` path only;
Iceberg sources were already un-pinned before the change, so entries that read Iceberg do not
declare this axis.

## Status values

| Status | Meaning |
|--------|---------|
| `CURRENT` | Passed live, and every axis it depends on still holds its proof-time value. |
| `STALE` | Passed live, but an axis it depended on has since changed. **The claim no longer stands.** |
| `NEVER_RUN` | Has never been executed against live infrastructure. |
| `NEEDS_RECHECK` | Was run, but the evidence isn't traceable well enough to stand behind. |

## Smoke suite

Configs live in `configs/smokes/`; see [Smoke testing](smoke_testing.md) for how to run them. The
tripwire enforces that this table has exactly one row per config — no ghosts, no gaps.

| # | Config | Proves | Status | Date | run_id | Axes at proof |
|---|--------|--------|--------|------|--------|---------------|
| 01 | `01_serverless_cpu.json` | Spark on Dataproc Serverless, CPU (statistical + ML) | CURRENT | 2026-08-22 | `smoke-01-serverless-cpu-2ca2c0f48bd0` | `serverless_deps=container-image`, `python=3.11` |
| 02 | `02_bq_native.json` | BigQuery-native models (`arima_plus`, `timesfm`) | CURRENT | 2026-08-22 | `smoke-02-bq-native-7b34cfd9eb98` | `python=3.11` |
| 03 | `03_serverless_gpu.json` | Serverless GPU (deep-learning on an L4) | CURRENT | 2026-08-22 | `smoke-03-serverless-gpu-a1adfc48d5d3` | `serverless_deps=container-image`, `python=3.11` |
| 04 | `04_cluster_cpu.json` | Spark on an ephemeral Dataproc cluster, CPU | CURRENT | 2026-08-23 | `smoke-04-cluster-cpu-88fddc72b8a1` | `cluster_deps=packed-venv-init-action`, `python=3.11` |
| 05 | `05_cluster_reuse.json` | Reusing a standing Dataproc cluster by name | CURRENT | 2026-08-23 | `smoke-05-cluster-reuse-2a7edf806a52` | `cluster_deps=packed-venv-init-action`, `python=3.11` |
| 06 | `06_cluster_gpu.json` | Dataproc cluster GPU (T4), incl. zone failover | CURRENT | 2026-08-24 | `smoke-06-cluster-gpu-a510512f507a` | `cluster_deps=packed-venv-init-action`, `gpu_cluster_image=prebaked-driver-image`, `python=3.11` |
| 07 | `07_ray_cpu.json` | Ray on Vertex, CPU | CURRENT | 2026-08-28 | not recorded | `ray_deps=stock-image+uv-runtime-env`, `python=3.11` |
| 08 | `08_ray_gpu.json` | Ray on Vertex, GPU T4 (neuralprophet) | CURRENT | 2026-08-28 | not recorded | `ray_deps=stock-image+uv-runtime-env`, `python=3.11` |
| 09 | `09_shared_ray.json` | Several families on one shared Ray cluster (CPU + GPU pools) | STALE | 2026-08-25 | not recorded | `ray_deps=custom-container-image`, `python=3.11` |
| 10 | `10_mixed_runtimes.json` | Spark + Ray + BigQuery families concurrently under one run_id | STALE | 2026-08-25 | not recorded | `ray_deps=custom-container-image`, `serverless_deps=container-image`, `python=3.11` |
| 11 | `11_ensemble_barrier.json` | Ensembling in barrier mode | CURRENT | 2026-08-25 | not recorded | `serverless_deps=container-image`, `python=3.11` |
| 12 | `12_ensemble_microbatch.json` | Ensembling in microbatch mode | CURRENT | 2026-08-25 | not recorded | `serverless_deps=container-image`, `python=3.11` |
| 13 | `13_native_format.json` | Reading the native BigQuery source table | CURRENT | 2026-08-25 | not recorded | `native_source_pin=unpinned-all-sources`, `serverless_deps=container-image`, `python=3.11` |
| 14 | `14_full_dag.json` | Flagship: all families + native + ensemble, one run_id (DL on Spark L4) | CURRENT | 2026-08-25 | not recorded | `serverless_deps=container-image`, `python=3.11` |
| 15 | `15_airflow_multi_engine.json` | The whole DAG orchestrated by Composer/Airflow | NEVER_RUN | — | — | — |

### Why 09 and 10 are stale

Both ran the Ray path on the custom container image. `822ae25` deleted that path — the custom image
fails Vertex Ray GPU provisioning, so all Ray moved to the stock prebuilt image with dependencies
delivered by Ray 2.47's `runtime_env` uv plugin. Smokes 07 and 08 were re-run on the new path and
pass; **09 and 10 were not.** Smoke 10 is the significant loss: it was the strongest proof in the
suite — all four families across all three runtimes under a single `run_id` — and that claim does
not currently stand on the shipped architecture.

## Notebooks

All eight notebooks were executed headless against a live deployment and committed with their
output cells at `ff1f8bf` (2026-08-28), which lands **after** the Ray re-architecture — so the Ray
notebook reflects the current path.

| Notebook | Status | Date | Axes at proof |
|----------|--------|------|---------------|
| `01_spark_via_connect.ipynb` | CURRENT | 2026-08-28 | `serverless_deps=container-image`, `python=3.11` |
| `02_bigquery_native.ipynb` | CURRENT | 2026-08-28 | `python=3.11` |
| `03_combo_and_ensemble.ipynb` | CURRENT | 2026-08-28 | `serverless_deps=container-image`, `python=3.11` |
| `04_ray_on_vertex.ipynb` | CURRENT | 2026-08-28 | `ray_deps=stock-image+uv-runtime-env`, `python=3.11` |
| `07_scale_review.ipynb` | CURRENT | 2026-08-28 | `python=3.11` |
| `08_run_and_monitor.ipynb` | CURRENT | 2026-08-28 | `serverless_deps=container-image`, `python=3.11` |
| `09_review_run.ipynb` | CURRENT | 2026-08-28 | `python=3.11` |
| `model_playground.ipynb` | CURRENT | 2026-08-28 | `python=3.11` |

## Other capabilities

| Capability | Status | Evidence |
|------------|--------|----------|
| Run-inspection layer (`review.py`) | CURRENT | Exercised live through notebooks 08 + 09 at `ff1f8bf`. Its `@gcp` registry readers ran against a real deployment. |
| Airflow DAG emitter (`airflow_emit`) | NEVER_RUN | The renderer is offline-proven (emitted source compiles; `DagBag` parse test). No run has ever been orchestrated by Composer — that is smoke 15. |
| RuntimeProbe read path (P1–P4) | NEVER_RUN | Offline only, against fakes. No probe has called a live Dataproc/Ray/BigQuery status API. |
| RuntimeProbe cancel (P5) | NEVER_RUN | Offline only. No cancel has stopped a real job. |
| Run audit + IAM roles (P6) | NEVER_RUN | `identity.resolve_principal` is `pragma: no cover` — its ADC and userinfo paths have never executed. The two custom roles are `validate`-clean but never `apply`-ed. |

## Known validation gaps

Things that are true today and that no entry above covers. Keep this list short and act on it.

- **`ray_autoscale` defaults to `True` (`config.py`) but all four Ray smokes pin it `false`.**
  Introduced by `4c988bc`, when a per-pool `AutoscalingSpec` crashed the Vertex Ray head at
  provisioning. The shipped default has therefore never passed a live smoke. That crash may have
  been the custom image — since deleted — so autoscaling may simply work now, but nobody has
  checked.
- **No `run_id` was recorded for smokes 07–14.** Smokes 01–06 have them. Without one there is no
  reverse-trace from the ledger to the platform job, which is the whole point of recording a
  result. Any re-run must capture it.

## Provenance confidence

Entries dated before 2026-08-29 were **reconstructed** during a reconciliation on that date, from
the results log and commit history. Their axis values are inferred from what the code did at the
time, not recorded when the run happened, and the missing `run_id`s cannot be recovered. Entries
from 2026-08-29 onward are recorded at run time and are authoritative.
