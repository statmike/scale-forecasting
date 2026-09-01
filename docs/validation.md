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
| `run_id_inputs` | `authored-config-only` | 2026-09-01, after the fork below | `+compute.profile.source` (W11a) |
| `fleet_sizing` | `derived-overlay` | W7b `6f4638f` + W8 `be78bec` (2026-08-31) | `platform-defaults` |
| `horizon_features` | `computed-at-future-dates` | `cb7d15f` (2026-08-31) | `first-rows-of-history` |

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

`horizon_features` governs **what an exog-aware model is handed for the forecast horizon**. Until
`cb7d15f` the horizon's design matrix was the first `horizon` rows of *history*. Holiday flags and
Fourier phase are functions of the date, so a model was given the seasonal phase from the start of
its history for the dates it was forecasting. `features.build_future_features` now computes those
columns at the future dates.

Every smoke config sets `features.holidays: ["US"]`, so every run builds an `X` frame — but only the
models that *consume* it are affected, which today means the lag forecasters (`xgboost`, `lightgbm`,
`regression_lags`) plus `sarimax`, `ucm` and `prophet`. Rows whose model list is confined to
`theta` / `holtwinters` / `neuralprophet` / the BigQuery natives do not declare this axis, because
nothing in them reads the frame. This is a **forecast-value** change, not a plumbing change: the
affected runs would produce different numbers today.

It cost one of the three remaining `CURRENT` rows. Smoke 07 (Ray CPU) runs `xgboost`, and so does
notebook 01. Both are downgraded here rather than argued around — the point of this file is that
"the mechanism still works" and "the result still stands" are different claims, and only the second
one is what a `CURRENT` row asserts.

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
| 01 | `01_serverless_cpu.json` | Spark on Dataproc Serverless, CPU (statistical + ML) | CURRENT | 2026-09-01 | `smoke-01-serverless-cpu-439b5350249b` | `serverless_deps=container-image`, `python=3.11`, `fleet_sizing=derived-overlay`, `horizon_features=computed-at-future-dates` |
| 02 | `02_bq_native.json` | BigQuery-native models (`arima_plus`, `timesfm`) | CURRENT | 2026-09-01 | `smoke-02-bq-native-0ffcc1f22d54` | `python=3.11` |
| 03 | `03_serverless_gpu.json` | Serverless GPU (deep-learning on an L4) | STALE | 2026-08-22 | `smoke-03-serverless-gpu-a1adfc48d5d3` | `serverless_deps=container-image`, `python=3.11`, `fleet_sizing=platform-defaults` |
| 04 | `04_cluster_cpu.json` | Spark on an ephemeral Dataproc cluster, CPU | STALE | 2026-08-23 | `smoke-04-cluster-cpu-88fddc72b8a1` | `cluster_deps=packed-venv-init-action`, `python=3.11`, `fleet_sizing=platform-defaults`, `horizon_features=first-rows-of-history` |
| 05 | `05_cluster_reuse.json` | Reusing a standing Dataproc cluster by name | STALE | 2026-08-23 | `smoke-05-cluster-reuse-2a7edf806a52` | `cluster_deps=packed-venv-init-action`, `python=3.11`, `fleet_sizing=platform-defaults`, `horizon_features=first-rows-of-history` |
| 06 | `06_cluster_gpu.json` | Dataproc cluster GPU (T4), incl. zone failover | STALE | 2026-08-24 | `smoke-06-cluster-gpu-a510512f507a` | `cluster_deps=packed-venv-init-action`, `gpu_cluster_image=prebaked-driver-image`, `python=3.11`, `fleet_sizing=platform-defaults` |
| 07 | `07_ray_cpu.json` | Ray on Vertex, CPU | STALE | 2026-08-28 | not recorded | `ray_deps=stock-image+uv-runtime-env`, `python=3.11`, `horizon_features=first-rows-of-history` |
| 08 | `08_ray_gpu.json` | Ray on Vertex, GPU T4 (neuralprophet) | CURRENT | 2026-08-28 | not recorded | `ray_deps=stock-image+uv-runtime-env`, `python=3.11` |
| 09 | `09_shared_ray.json` | Several families on one shared Ray cluster (CPU + GPU pools) | STALE | 2026-08-25 | not recorded | `ray_deps=custom-container-image`, `python=3.11`, `horizon_features=first-rows-of-history` |
| 10 | `10_mixed_runtimes.json` | Spark + Ray + BigQuery families concurrently under one run_id | STALE | 2026-08-25 | not recorded | `ray_deps=custom-container-image`, `serverless_deps=container-image`, `python=3.11`, `fleet_sizing=platform-defaults`, `horizon_features=first-rows-of-history` |
| 11 | `11_ensemble_barrier.json` | Ensembling in barrier mode | STALE | 2026-08-25 | not recorded | `serverless_deps=container-image`, `python=3.11`, `fleet_sizing=platform-defaults` |
| 12 | `12_ensemble_microbatch.json` | Ensembling in microbatch mode | STALE | 2026-08-25 | not recorded | `serverless_deps=container-image`, `python=3.11`, `fleet_sizing=platform-defaults` |
| 13 | `13_native_format.json` | Reading the native BigQuery source table | STALE | 2026-08-25 | not recorded | `native_source_pin=unpinned-all-sources`, `serverless_deps=container-image`, `python=3.11`, `fleet_sizing=platform-defaults`, `horizon_features=first-rows-of-history` |
| 14 | `14_full_dag.json` | Flagship: all families + native + ensemble, one run_id (DL on Spark L4) | STALE | 2026-08-25 | not recorded | `serverless_deps=container-image`, `python=3.11`, `fleet_sizing=platform-defaults`, `horizon_features=first-rows-of-history` |
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
that only appear live. The offline gate proves the arithmetic is self-consistent, not that Dataproc
accepts it.

**Dataproc accepts it.** On 2026-09-01 smoke 01 was re-run and is the first row back
(`smoke-01-serverless-cpu-439b5350249b`). The batch was submitted with an explicit
`spark.executor.cores=4`, a `2 / 2 / 7` dynamic-allocation band, and all four thread-pin
`executorEnv` variables at 1, and it was accepted and ran to `SUCCEEDED`. The pin is verifiable from
the other end too: every harvested cell of that run reports `intraop_threads=1`, so the property the
overlay submitted is the one the model actually fitted under. That is the single most load-bearing
untested mechanism in the suite cleared; the remaining Spark rows are stale for want of a re-run,
not for want of a working overlay.

The campaign that re-earns the rest is profiler **W12**, which also does the `off`-vs-profiled A/B
and captures the measurements the shipped baseline will carry.

## Demonstration and scale configs

Configs live in `configs/`; these are what a *user* runs — the demo path on day one, and the four
`*_100k*` runs behind [the workshop](workshop.md)'s Act 1. The tripwire enforces one row per config
here exactly as it does for the smokes. `compute_fallback.json` is excluded: it is a zone-failover
map consumed at submit time, not a run config.

**Every row started `NEVER_RUN` when this table was added, and that was a deliberate reading of the
evidence, not a claim that none of these had ever executed.** Several of them certainly had, during
the demo and build work that produced the figures quoted in `docs/workshop.md`. But no run of any of
them recorded a `run_id`, a date, or the architecture it ran on — and this file's first rule is that
an unrecorded run is not a result. So the demonstration surface entered the ledger empty, which is
the honest starting position and the reason for adding the table at all: it is the surface with the
*widest* gap between what we believe works and what we can cite. Rows fill in as the live campaign
([`docs/smoke_testing.md`](smoke_testing.md) for how each is run) reaches them.

| Config | Proves | Status | Date | run_id | Axes at proof |
|--------|--------|--------|------|--------|---------------|
| `bq_native_demo.json` | The BigQuery-native family alone — no cluster of any kind (100 series) | CURRENT | 2026-09-01 | `bq-native-demo-b374041fdd1e` | `python=3.11` |
| `explode_demo.json` | The Spark `explode` fan-out, statistical + ML, artifacts persisted (10) | NEVER_RUN | — | — | — |
| `mixed_demo.json` | One Spark model and the natives under one `run_id`, backtested (10) | NEVER_RUN | — | — | — |
| `ensemble_demo.json` | The same mix with three ensemble strategies on (10) | NEVER_RUN | — | — | — |
| `per_family_runtimes_demo.json` | Per-family runtime split — deep learning to Ray GPU, the rest on Spark (50) | NEVER_RUN | — | — | — |
| `ray_cpu_demo.json` | Ray on Vertex, CPU, alongside the natives, backtested (6) | NEVER_RUN | — | — | — |
| `ray_gpu_demo.json` | Ray on Vertex, GPU T4 (`neuralprophet`), alongside the natives (6) | NEVER_RUN | — | — | — |
| `ray_autoscale_demo.json` | **The shipped `ray_autoscale=true` default**, 1→8 CPU nodes at 10,000 series | NEVER_RUN | — | — | — |
| `explode_100k.json` | The headline: Spark `explode` over 100,000 series | NEVER_RUN | — | — | — |
| `ray_100k.json` | The same work on Ray — the runtime-parity half of the scale review | NEVER_RUN | — | — | — |
| `all_families_100k.json` | Every family at 100,000 series under one `run_id` (Ray + BigQuery, T4) | NEVER_RUN | — | — | — |
| `all_families_100k_full.json` | As above, plus backtesting and persisted artifacts | NEVER_RUN | — | — | — |

### What this surface will exercise that the smoke suite cannot

- **`ray_autoscale=true`.** Every Ray *smoke* pins it `false`; five configs here leave it `true`,
  which is the shipped default. See the gap below.
- **Scale.** The smokes run 100 series. The fleet arithmetic W7b/W8 introduced is only under real
  pressure at 100k, and `explode_100k.json` is the one config that overrides the bucket sizing
  (`bucket_target_cells: 200`) because the default OOM'd at that scale.
- **Cross-run reading.** `07_scale_review` compares the four scale runs *to each other*; nothing in
  the smoke suite produces a set of runs meant to be read side by side.

## Notebooks

All eight notebooks were executed headless against a live deployment and committed with their
output cells at `ff1f8bf` (2026-08-28), which lands **after** the Ray re-architecture — so the Ray
notebook reflects the current path.

| Notebook | Status | Date | Axes at proof |
|----------|--------|------|---------------|
| `01_spark_via_connect.ipynb` | STALE | 2026-08-28 | `serverless_deps=container-image`, `python=3.11`, `horizon_features=first-rows-of-history` |
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
| Workshop Act 1 (100k history, Cloud Shell / VM) | NEVER_RUN | The four scale configs above, run as the workshop instructs them and in that order. Not the same claim as "the configs work": Act 1 is followed by someone who has just deployed, from a shell with a session limit, and its failure modes are disk, quota and session death. |
| Workshop Act 2 (pre-rendered notebook tour) | NEVER_RUN | Headless execution of the tour notebooks against a fresh deployment. The notebook rows above were proven by the acceptance harness, which is not the same path. |
| Workshop Act 3 (live Colab Enterprise tour) | NEVER_RUN | The six notebooks opened and run interactively on the `sf-main` runtime, reading Act 1's runs. |
| Run-inspection layer (`review.py`) | CURRENT | Exercised live through notebooks 08 + 09 at `ff1f8bf`. Its `@gcp` registry readers ran against a real deployment. |
| Airflow DAG emitter (`airflow_emit`) | NEVER_RUN | The renderer is offline-proven (emitted source compiles; `DagBag` parse test). No run has ever been orchestrated by Composer — that is smoke 15. |
| RuntimeProbe read path (P1–P4) | NEVER_RUN | Offline only, against fakes. No probe has called a live Dataproc/Ray/BigQuery status API. |
| RuntimeProbe cancel (P5) | NEVER_RUN | Offline only. No cancel has stopped a real job. |
| Custom IAM roles (P6) | CURRENT | Applied live 2026-09-01: `projects/statmike-scale-forecasting/roles/sfProbeReader` and `roles/sfJobCanceller` now exist. Until then they had only ever been `validate`-clean. Creation is not use — that the permission sets are *sufficient* for a probe or a cancel is the P1–P5 rows below, not this one. |
| Run audit principal (P6) | NEVER_RUN | `identity.resolve_principal` is `pragma: no cover` — its ADC and userinfo paths have never executed. |

## Known validation gaps

Things that are true today and that no entry above covers. Keep this list short and act on it.

- **`ray_autoscale` defaults to `True` (`config.py`) but all four Ray smokes pin it `false`.**
  Introduced by `4c988bc`, when a per-pool `AutoscalingSpec` crashed the Vertex Ray head at
  provisioning. The shipped default has therefore never passed a live smoke. That crash may have
  been the custom image — since deleted — so autoscaling may simply work now, but nobody has
  checked. The smoke suite cannot close this on its own: five of the configs in the demonstration
  table leave the default `true` (`ray_autoscale_demo`, `ray_gpu_demo`, and all three Ray-runtime
  scale configs), so proving it is a demonstration-surface result, not a smoke result.
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
  and no row is stale for either. **The 2026-09-01 fix moved them a fourth and final time**, by
  *removing* `compute.profile.source` again — so the ids recorded that same day for smokes 01 and
  02 (`…-439b5350249b`, `…-0ffcc1f22d54`) are already in this category, hours after being written.
  They remain valid pointers into the registry and the results they point at are unaffected; they
  are simply no longer recomputable from their configs. Unlike the three before it, this move makes
  identity *stop* drifting rather than start: `run_id_inputs` is now `authored-config-only` and
  there is no resolved value left in the digest to move it again.
  The change that *did* make live results stale arrived at W7b/W8, and it was not the one predicted
  here. This note used to say the staleness event would be W6, "when `profile.mode='auto'` starts
  actually sizing fleets from measurement." That never happened and now never will in that form —
  see the next gap. What moved the fleets was the **static** arithmetic W7/W8 wired in with the
  profile argument left as `None`: no measurement involved, and every Spark fleet reshaped anyway.

- **`compute.profile.source` in the digest forked run identity two different ways. Fixed
  2026-09-01; live confirmation pending.** Both halves were found live, in the campaign's first two
  waves, and neither had an offline analogue — the offline suite contained a test asserting the
  *forking* behaviour was correct.

  - **On failure.** Smoke 02 resolved `smoke-02-bq-native-d2d37cd657e8`, and an immediate re-run
    resolved `…-0ffcc1f22d54` from a byte-identical config. The discovery query ran a moment before
    its own schema migration and raised; `lock_profile_source` caught it and returned the config
    still carrying `source: "auto"` (confirmed in that run's staged manifest) rather than the
    `"baseline"` it would otherwise have pinned. The trigger was one-off, the mechanism was not: the
    same branch catches a timeout, a quota error or a permissions blip, and forks the id with only
    a `debug`-level line to show for it.
  - **On success.** Smoke 01 reported `FAIL` for this and nothing else. Run 1 (`…-439b5350249b`)
    harvested; the re-run found that harvest, pinned
    `source: "smoke-01-serverless-cpu-439b5350249b"`, and resolved `…-8f602110b7ea`. Run 2 harvested
    in turn, so run 3 would have pinned run 2. **Identity never converged.** A "re-run" submitted
    two more Dataproc batches and wrote a second complete result set, so every Spark smoke would
    have reported the same FAIL — and the rerun guard, dedupe-on-read, and any retried Airflow task
    all rest on the id being stable.

  The fix is one exclusion: `registry.ids._canonical_config` drops `compute.profile.source` before
  hashing. Pinning still happens and is still written into the staged manifest, and the sizing
  telemetry still records the full provenance block naming the run the measurements came from. What
  changes is the principle, which is worth stating once: **a run's identity is what was asked for;
  its provenance is what answered.** `source` is resolved by the launcher, not authored by the
  user, so it belongs to the second. That also disposes of the failure half — if the field cannot
  move the id when discovery succeeds, it cannot move it when discovery raises either.

  Offline-proven only so far: the inverted test plus a new one asserting the id is identical whether
  or not discovery reached the registry. **The live confirmation is a re-run of smoke 01 passing its
  `verify_rerun` check**, which is the exact assertion that failed. Until that happens this stays a
  gap rather than a closed item.
- **The measurement path is live — closed 2026-09-01, and what is left of the gap is narrow.** This
  entry used to read "no live run has ever taken a compute measurement, on any runtime." Smoke 01
  ended that in a single wave, and did it twice over. Run 1 harvested; run 2 was sized from run 1.

  The measured fleet is not a cosmetic difference. Run 1's statistical batch was submitted with
  `spark.executor.memory=9600m` and no explicit overhead — the static arithmetic's answer. Run 2's
  was submitted with `spark.executor.memory=2048m` and `spark.executor.memoryOverhead=4093m`,
  derived from run 1's harvested `process_rss_bytes` under the 1.3 memory margin. **That is the
  first fleet in this product's history whose shape came from a measurement rather than a
  constant**, and it ran to `SUCCEEDED` on the same work, producing an identical leaderboard
  (`theta`, `holtwinters`, `xgboost`, 100 cells each). Sizing from evidence is not merely emitted;
  it is sufficient.

  The probes return sane numbers on a real Dataproc executor, which was the other open question:
  per cell, `cpu_seconds` 0.48–0.86 by model, `process_rss_bytes` ≈ 750 MiB (absolute, as intended,
  not a delta), `n_obs` 1460 matching the seeded history, `peak_gpu_bytes` NULL on CPU, and
  `intraop_threads` 1 — agreeing with the `executorEnv` pins the same run submitted.

  **What is still unproven** is smaller than it was and worth keeping separate: the probes have not
  run on **Vertex Ray** (only Dataproc), the `measure: "controlled"` A/B has not been done, and the
  per-cell overhead has not been checked at 100k, where it is the only place it could matter. The
  Ray pre-pass call site remains unfired for the reason below.

  The Ray pre-pass has exactly one production call site — `engines/ray_engine` calls
  `profiling.source.resolve_profile` — and it has still never fired in a live smoke. `mode` defaults to
  `"auto"` with `min_cells = 1000`, no smoke config sets `profile`, and the gate compares series ×
  profilable models: smoke 07 offers 300 cells and smoke 08 offers 100, so both take the `None`
  path. Both Spark paths pass `None` to the *pre-pass* and structurally must — `spark.executor.cores`
  and `spark.task.cpus` are fixed at submit or at create, before any of our code runs on the cluster.
  What smoke 01 proved is the harvest route to the same `ComputeProfile`, not the pre-pass route.

  **W10's harvest is what made that possible.** Harvest is on by default:
  every cell records its CPU time, absolute process memory, peak device bytes, thread cap and
  `n_obs` onto `forecast_metadata`, and `profiling.cost.harvest_profile` aggregates those rows into the
  same `ComputeProfile` the pre-pass would have built. The schema question it raised is also
  **settled live**: on 2026-09-01 the first run of the campaign
  (`smoke-02-bq-native-0ffcc1f22d54`) drove `ensure_tables` against the deployed
  `forecast_metadata`, and all five nullable harvest columns — `cpu_seconds`, `process_rss_bytes`,
  `peak_gpu_bytes`, `intraop_threads`, `n_obs` — were added by the additive ALTER without touching
  the existing rows. The self-migration works on a table that predates it. W13 (shipping a baseline)
  now has real measurements to be built from.

  **W11a's precedence chain and W11b's wiring both executed.** `compute.profile.source` defaults to
  `"auto"`; `profiling.source.resolve_profile_source` walks named run → discovered run → shipped
  baseline → static config; `plan_run` / `stage_run` pin the result before the digest; and
  `registry/bq.read_compute_harvest` / `discover_harvest_run` are the two queries behind it. All of
  that was offline-only until smoke 01, and all three of the second-order checks this section asked
  for came back green in one run: **run A's harvest was discoverable by run B**, **the pinned
  `run_id` landed in the staged config** (`source: "smoke-01-serverless-cpu-439b5350249b"` in
  `runs/smoke-01-serverless-cpu-8f602110b7ea.json`), and **the memory properties B emitted differed
  from A's**. The chain is no longer hypothetical.

  Run 2's `provenance` block has been read, and it is complete: `basis: "measured"`, `source` and
  `run_id` both naming run 1, a `measured_at` timestamp, a signature of
  `{source_table: source_series_iceberg, n_series: 100, median_n_obs: 1460}`, and no warnings. The
  `slot` it produced records `basis: "measured"`, `measured: ["cores", "memory_bytes"]` and an empty
  `assumed` list, against run 1's `basis: "static"`, `measured: []`,
  `assumed: ["cores", "memory_bytes"]`. The audit trail distinguishes the two, which is the point of
  having one.

  One wrinkle to know about before relying on discovery: the recorded signature has `freq: null`,
  because these configs do not set a frequency. Discovery matches on the signature, so a manual
  `discover_harvest_run(..., freq="D")` finds nothing for these runs even though `auto` resolves
  them correctly. It is a query-argument trap, not a defect — but it will mislead anyone probing by
  hand.

  **W9b's merge works; one of its three assumptions is still open.** The whole sizing decision — the
  fleet plan, its translation to platform settings, and the profile behind it — is stamped into the
  run header's `job_telemetry` under `sizing.<family>` and surfaced by `v_run_summary`. That write
  is a BigQuery `JSON_SET` **merge** rather than a whole-column write, so the several family jobs of
  one run each record their own sizing instead of the last one to finish overwriting the rest. Two
  of the three things listed here as never having run against real BigQuery ran on 2026-09-01, in
  smoke 01: `JSON_SET` **did** auto-create the parent object for a nested path, and the run's two
  families **do** coexist — `sizing.ml` and `sizing.statistical` are both present and complete under
  one header, with no sign of a race. The third is untouched, because smoke 01 is a Serverless run:
  whether the **cluster** path's stamp lands at all is still unknown — `submit_cluster_job` had
  never written header telemetry before W9b, so a cluster run's `v_run_summary` row was blank. Wave
  5 answers it. Every one of these writes is best-effort (logged and swallowed), so a wrong
  assumption degrades to "no telemetry", never to a failed run.

## Provenance confidence

Entries dated before 2026-08-29 were **reconstructed** during a reconciliation on that date, from
the results log and commit history. Their axis values are inferred from what the code did at the
time, not recorded when the run happened, and the missing `run_id`s cannot be recovered. Entries
from 2026-08-29 onward are recorded at run time and are authoritative.
