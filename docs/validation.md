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
| `run_id_inputs` | `+compute.profile.source` | W11a (2026-08-31) | `+compute.profile.measure` (W10) |
| `fleet_sizing` | `derived-overlay` | W7b `6f4638f` + W8 `be78bec` (2026-08-31) | `platform-defaults` |

`native_source_pin` governs **native BigQuery table** reads on the BQML `CREATE MODEL` path only;
Iceberg sources were already un-pinned before the change, so entries that read Iceberg do not
declare this axis.

`fleet_sizing` governs **how a Spark fleet's shape is decided**. Until W7b/W8 we stated a worker or
executor *count* and let the platform choose everything else: Dataproc Serverless picked its own
executor cores, memory and dynamic-allocation band, and a Dataproc cluster ran its default two
4-core executors per worker with nothing bounding a GPU. Now `resources.translate_serverless` /
`translate_cluster` derive an explicit shape — executor cores, memoryOverhead, the dynamic-allocation
min/initial/max, `spark.task.cpus`, the thread pins, and a derived worker count — and submit it as a
properties overlay. **This is a different fleet**, so any Spark result proven on the old one is a
claim about a machine shape that no longer exists.

Two runtimes do *not* declare it. **BigQuery-native** work has no fleet of ours to shape. **Ray** is
unmoved in practice: `plan_pool(profile=None)` reproduces the pre-profiler arithmetic exactly, W1's
autoscale-ceiling derivation only fires when `ray_autoscale` is true and all four Ray smokes pin it
`false`, and W2's device catalog left T4 at 16 GiB (only L4 moved). Smoke 10 declares the axis
because it submits Serverless work alongside its Ray families.

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
| 01 | `01_serverless_cpu.json` | Spark on Dataproc Serverless, CPU (statistical + ML) | STALE | 2026-08-22 | `smoke-01-serverless-cpu-2ca2c0f48bd0` | `serverless_deps=container-image`, `python=3.11`, `fleet_sizing=platform-defaults` |
| 02 | `02_bq_native.json` | BigQuery-native models (`arima_plus`, `timesfm`) | CURRENT | 2026-08-22 | `smoke-02-bq-native-7b34cfd9eb98` | `python=3.11` |
| 03 | `03_serverless_gpu.json` | Serverless GPU (deep-learning on an L4) | STALE | 2026-08-22 | `smoke-03-serverless-gpu-a1adfc48d5d3` | `serverless_deps=container-image`, `python=3.11`, `fleet_sizing=platform-defaults` |
| 04 | `04_cluster_cpu.json` | Spark on an ephemeral Dataproc cluster, CPU | STALE | 2026-08-23 | `smoke-04-cluster-cpu-88fddc72b8a1` | `cluster_deps=packed-venv-init-action`, `python=3.11`, `fleet_sizing=platform-defaults` |
| 05 | `05_cluster_reuse.json` | Reusing a standing Dataproc cluster by name | STALE | 2026-08-23 | `smoke-05-cluster-reuse-2a7edf806a52` | `cluster_deps=packed-venv-init-action`, `python=3.11`, `fleet_sizing=platform-defaults` |
| 06 | `06_cluster_gpu.json` | Dataproc cluster GPU (T4), incl. zone failover | STALE | 2026-08-24 | `smoke-06-cluster-gpu-a510512f507a` | `cluster_deps=packed-venv-init-action`, `gpu_cluster_image=prebaked-driver-image`, `python=3.11`, `fleet_sizing=platform-defaults` |
| 07 | `07_ray_cpu.json` | Ray on Vertex, CPU | CURRENT | 2026-08-28 | not recorded | `ray_deps=stock-image+uv-runtime-env`, `python=3.11` |
| 08 | `08_ray_gpu.json` | Ray on Vertex, GPU T4 (neuralprophet) | CURRENT | 2026-08-28 | not recorded | `ray_deps=stock-image+uv-runtime-env`, `python=3.11` |
| 09 | `09_shared_ray.json` | Several families on one shared Ray cluster (CPU + GPU pools) | STALE | 2026-08-25 | not recorded | `ray_deps=custom-container-image`, `python=3.11` |
| 10 | `10_mixed_runtimes.json` | Spark + Ray + BigQuery families concurrently under one run_id | STALE | 2026-08-25 | not recorded | `ray_deps=custom-container-image`, `serverless_deps=container-image`, `python=3.11`, `fleet_sizing=platform-defaults` |
| 11 | `11_ensemble_barrier.json` | Ensembling in barrier mode | STALE | 2026-08-25 | not recorded | `serverless_deps=container-image`, `python=3.11`, `fleet_sizing=platform-defaults` |
| 12 | `12_ensemble_microbatch.json` | Ensembling in microbatch mode | STALE | 2026-08-25 | not recorded | `serverless_deps=container-image`, `python=3.11`, `fleet_sizing=platform-defaults` |
| 13 | `13_native_format.json` | Reading the native BigQuery source table | STALE | 2026-08-25 | not recorded | `native_source_pin=unpinned-all-sources`, `serverless_deps=container-image`, `python=3.11`, `fleet_sizing=platform-defaults` |
| 14 | `14_full_dag.json` | Flagship: all families + native + ensemble, one run_id (DL on Spark L4) | STALE | 2026-08-25 | not recorded | `serverless_deps=container-image`, `python=3.11`, `fleet_sizing=platform-defaults` |
| 15 | `15_airflow_multi_engine.json` | The whole DAG orchestrated by Composer/Airflow | NEVER_RUN | — | — | — |

### Why almost everything Spark is stale

Both ran the Ray path on the custom container image. `822ae25` deleted that path — the custom image
fails Vertex Ray GPU provisioning, so all Ray moved to the stock prebuilt image with dependencies
delivered by Ray 2.47's `runtime_env` uv plugin. Smokes 07 and 08 were re-run on the new path and
pass; **09 and 10 were not.** Smoke 10 is the significant loss: it was the strongest proof in the
suite — all four families across all three runtimes under a single `run_id` — and that claim does
not currently stand on the shipped architecture.

**The Spark rows (01, 03, 04, 05, 06, 10–14 and notebooks 03, 08) went stale on 2026-08-31** when
W7b/W8 moved `fleet_sizing`. Every one of them ran on a fleet the platform shaped; today's code
states the shape itself. Concretely, a run that used to get Dataproc's default two 4-core executors
per worker now gets one 7-core executor with a declared heap and overhead, a `spark.task.cpus`
bounded by the accelerator, `OMP_NUM_THREADS` pinned to match, and a worker count derived from the
fan-out rather than a flat two.

This is a bigger downgrade than the Ray one, and it is meant to be. The staled claims are mostly
*correctness* claims ("Serverless GPU works", "cluster reuse works"), and it is tempting to argue
that a different executor shape cannot break them. It can. W7 pins `spark.executor.cores` and
`spark.executor.memoryOverhead` to values snapped from the legal-value tables in the design doc, and
an illegal or unsatisfiable pair fails the batch at submit rather than degrading quietly; W8's
whole-worker executor is the shape that made YARN's `DominantResourceCalculator` leave the
ApplicationMaster unplaceable until an AM reserve was carved out. Those are exactly the failures
that only appear live. **None of the new arithmetic has run on real infrastructure even once** — the
offline gate proves it is self-consistent, not that Dataproc accepts it.

The campaign that re-earns these rows is profiler **W12**, which also does the `off`-vs-profiled A/B
and captures the measurements the shipped baseline will carry.

## Notebooks

All eight notebooks were executed headless against a live deployment and committed with their
output cells at `ff1f8bf` (2026-08-28), which lands **after** the Ray re-architecture — so the Ray
notebook reflects the current path.

| Notebook | Status | Date | Axes at proof |
|----------|--------|------|---------------|
| `01_spark_via_connect.ipynb` | CURRENT | 2026-08-28 | `serverless_deps=container-image`, `python=3.11` |
| `02_bigquery_native.ipynb` | CURRENT | 2026-08-28 | `python=3.11` |
| `03_combo_and_ensemble.ipynb` | STALE | 2026-08-28 | `serverless_deps=container-image`, `python=3.11`, `fleet_sizing=platform-defaults` |
| `04_ray_on_vertex.ipynb` | CURRENT | 2026-08-28 | `ray_deps=stock-image+uv-runtime-env`, `python=3.11` |
| `07_scale_review.ipynb` | CURRENT | 2026-08-28 | `python=3.11` |
| `08_run_and_monitor.ipynb` | STALE | 2026-08-28 | `serverless_deps=container-image`, `python=3.11`, `fleet_sizing=platform-defaults` |
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
- **The recorded `run_id`s for smokes 01–06 are no longer re-derivable.** W5 added
  `compute.profile` to `ComputeConfig`, and `run_id` is a digest of the whole config, so feeding
  those same config files to today's code yields different ids. **No row was marked `STALE` for
  this, deliberately**: `run_id_inputs` is not an axis any of those claims rests on. Smoke 01 proved
  that Spark-on-Serverless-CPU works, and it still did; nothing about the run changed, because in
  W5 nothing yet *reads* `compute.profile`. What was lost is narrower and worth naming — the ids
  above remain valid pointers *into* the registry, but you can no longer recompute one *from* its
  config to find it. Re-running any of 01–06 will record a new id, at which point the old one
  becomes purely historical. W10 moved every id a second time by adding `compute.profile.measure`,
  and W11a a third by adding `compute.profile.source`; the same reasoning applies unchanged to both,
  and no row is stale for either.
  The change that *did* make live results stale arrived at W7b/W8, and it was not the one predicted
  here. This note used to say the staleness event would be W6, "when `profile.mode='auto'` starts
  actually sizing fleets from measurement." That never happened and now never will in that form —
  see the next gap. What moved the fleets was the **static** arithmetic W7/W8 wired in with the
  profile argument left as `None`: no measurement involved, and every Spark fleet reshaped anyway.

- **No live run has ever taken a compute measurement, on any runtime.** The profiler has exactly one
  production call site — `engines/ray_engine` calls `profiling.resolve_profile` — and it has never
  fired in a live smoke. `mode` defaults to `"auto"` with `min_cells = 1000`, no smoke config sets
  `profile`, and the gate compares series × profilable models: smoke 07 offers 300 cells and smoke 08
  offers 100, so both take the `None` path. Both Spark paths pass `None` unconditionally and
  structurally must — `spark.executor.cores` and `spark.task.cpus` are fixed at submit or at create,
  before any of our code runs on the cluster. So `profiling.measure_fit` and `build_profile` are
  unit-tested against injected measurements and have never measured anything real.

  **W10 changed what the next live run will do, but not this gap.** Harvest is now on by default:
  every cell records its CPU time, absolute process memory, peak device bytes, thread cap and
  `n_obs` onto `forecast_metadata`, and `profiling.harvest_profile` aggregates those rows into the
  same `ComputeProfile` the pre-pass would have built. All of it is offline-proven only. Nothing has
  yet run on a real executor, so three things stay unverified until a live run: that the probes
  return sane numbers on Dataproc and Vertex rather than zeros or nulls, that the five columns
  auto-migrate onto the existing deployed `forecast_metadata`, and that the per-cell overhead is
  genuinely negligible at 100k scale. W12 is where that is settled; W13 (shipping a baseline)
  depends on it.

  **W11a built the consumer against the same unproven evidence.** `compute.profile.source` defaults
  to `"auto"`, and `profiling.resolve_profile_source` implements the whole precedence chain —
  named run, discovered run, shipped baseline, static config — with every loader injected, so the
  chain is tested offline end to end with no BigQuery. It cannot yet change a live run: nothing
  calls it (that is W11b), there is no baseline to load (W13), and there is no harvested run
  anywhere to discover, because of the gap above. The consequence worth stating plainly: **the first
  live run that resolves a source will be the first one whose sizing came from a measurement**, and
  its `provenance` block — basis, `run_id`, timestamp, signature, warnings — is the artifact to
  check when that happens.

  **W11b wired it, and it is now reachable on a live run — but still unexercised.** `plan_run` /
  `stage_run` pin `source: "auto"` to a concrete `run_id` before the digest, both Spark sizing call
  sites take a resolved profile instead of `None`, and `registry/bq.read_compute_harvest` /
  `discover_harvest_run` are the two queries behind it. Neither query has ever run against real
  BigQuery, and neither can return anything until a live run has harvested — so on today's
  deployment `auto` discovers nothing, pins `"baseline"`, finds no baseline, and sizes from static
  config: **behaviour identical to before the profiler existed.** The first live campaign therefore
  has a second-order thing to check beyond the probes themselves — that run A's harvest is
  discoverable by run B, that the pinned `run_id` lands in the staged config, and that the memory
  properties B emits actually differ from A's.

  **W9b made the decision auditable, with one assumption still unproven.** The whole sizing
  decision — the fleet plan, its translation to platform settings, and the profile behind it — is
  now stamped into the run header's `job_telemetry` under `sizing.<family>` and surfaced by
  `v_run_summary`. That write is a BigQuery `JSON_SET` **merge** rather than a whole-column write,
  so the several family jobs of one run each record their own sizing instead of overwriting one
  another (previously the last job to finish was the only one that left a trace). Three things
  about it have never run against real BigQuery and belong in the W12 checklist: that `JSON_SET`
  auto-creates the parent object for a nested path (`'$.sizing.deep_learning'` written into a
  document with no `sizing` key), that two families of one run genuinely coexist under `$.sizing`
  rather than racing, and that the **cluster** path's stamp lands at all — `submit_cluster_job` had
  never written header telemetry before this, so a cluster run's `v_run_summary` row was blank.
  Every one of these writes is best-effort (logged and swallowed), so a wrong assumption degrades
  to "no telemetry", never to a failed run.

## Provenance confidence

Entries dated before 2026-08-29 were **reconstructed** during a reconciliation on that date, from
the results log and commit history. Their axis values are inferred from what the code did at the
time, not recorded when the run happened, and the missing `run_id`s cannot be recovered. Entries
from 2026-08-29 onward are recorded at run time and are authoritative.
