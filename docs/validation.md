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
| `run_id_inputs` | `authored-config-only` | `a22e94c` (2026-09-01), after the fork below | `+compute.profile.source` (W11a) |
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
`false` (the demonstration surface covers that path — see `ray_autoscale_demo`, which reached the
derived ceiling of 8), and W2's device catalog left T4 at 16 GiB (only L4 moved). Smoke 10 declares the axis
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
| 01 | `01_serverless_cpu.json` | Spark on Dataproc Serverless, CPU (statistical + ML) | CURRENT | 2026-09-01 | `smoke-01-serverless-cpu-5af5de1accf2` | `serverless_deps=container-image`, `python=3.11`, `fleet_sizing=derived-overlay`, `run_id_inputs=authored-config-only`, `horizon_features=computed-at-future-dates` |
| 02 | `02_bq_native.json` | BigQuery-native models (`arima_plus`, `timesfm`) | CURRENT | 2026-09-01 | `smoke-02-bq-native-0ffcc1f22d54` | `python=3.11` |
| 03 | `03_serverless_gpu.json` | Serverless GPU (deep-learning on an L4) | CURRENT | 2026-09-01 | `smoke-03-serverless-gpu-a918f22d7970` | `serverless_deps=container-image`, `python=3.11`, `fleet_sizing=derived-overlay`, `run_id_inputs=authored-config-only` |
| 04 | `04_cluster_cpu.json` | Spark on an ephemeral Dataproc cluster, CPU | CURRENT | 2026-09-01 | `smoke-04-cluster-cpu-c5b992778fd1` | `cluster_deps=packed-venv-init-action`, `python=3.11`, `fleet_sizing=derived-overlay`, `horizon_features=computed-at-future-dates`, `run_id_inputs=authored-config-only` |
| 05 | `05_cluster_reuse.json` | Reusing a standing Dataproc cluster by name | CURRENT | 2026-09-01 | `smoke-05-cluster-reuse-596268ab32a7` | `cluster_deps=packed-venv-init-action`, `python=3.11`, `fleet_sizing=derived-overlay`, `horizon_features=computed-at-future-dates`, `run_id_inputs=authored-config-only` |
| 06 | `06_cluster_gpu.json` | Dataproc cluster GPU (T4), incl. zone failover | STALE | 2026-08-24 | `smoke-06-cluster-gpu-a510512f507a` | `cluster_deps=packed-venv-init-action`, `gpu_cluster_image=prebaked-driver-image`, `python=3.11`, `fleet_sizing=platform-defaults` |
| 07 | `07_ray_cpu.json` | Ray on Vertex, CPU | CURRENT | 2026-09-01 | `smoke-07-ray-cpu-782bcec2718f` | `ray_deps=stock-image+uv-runtime-env`, `python=3.11`, `run_id_inputs=authored-config-only`, `horizon_features=computed-at-future-dates` |
| 08 | `08_ray_gpu.json` | Ray on Vertex, GPU T4 (neuralprophet) | NEEDS_RECHECK | 2026-08-28 | not recorded | `ray_deps=stock-image+uv-runtime-env`, `python=3.11` |
| 09 | `09_shared_ray.json` | Several families on one shared Ray cluster (CPU + GPU pools) | STALE | 2026-08-25 | not recorded | `ray_deps=custom-container-image`, `python=3.11`, `horizon_features=first-rows-of-history` |
| 10 | `10_mixed_runtimes.json` | Spark + Ray + BigQuery families concurrently under one run_id | STALE | 2026-08-25 | not recorded | `ray_deps=custom-container-image`, `serverless_deps=container-image`, `python=3.11`, `fleet_sizing=platform-defaults`, `horizon_features=first-rows-of-history` |
| 11 | `11_ensemble_barrier.json` | Ensembling in barrier mode | STALE | 2026-08-25 | not recorded | `serverless_deps=container-image`, `python=3.11`, `fleet_sizing=platform-defaults` |
| 12 | `12_ensemble_microbatch.json` | Ensembling in microbatch mode | STALE | 2026-08-25 | not recorded | `serverless_deps=container-image`, `python=3.11`, `fleet_sizing=platform-defaults` |
| 13 | `13_native_format.json` | Reading the native BigQuery source table | STALE | 2026-08-25 | not recorded | `native_source_pin=unpinned-all-sources`, `serverless_deps=container-image`, `python=3.11`, `fleet_sizing=platform-defaults`, `horizon_features=first-rows-of-history` |
| 14 | `14_full_dag.json` | Flagship: all families + native + ensemble, one run_id (DL on Spark L4) | STALE | 2026-08-25 | not recorded | `serverless_deps=container-image`, `python=3.11`, `fleet_sizing=platform-defaults`, `horizon_features=first-rows-of-history` |
| 15 | `15_airflow_multi_engine.json` | The whole DAG orchestrated by Composer/Airflow | NEVER_RUN | — | — | — |

### The cluster path re-proved itself on 2026-09-01, and the reuse smoke checks the thing that matters

Smokes 04 and 05 both passed under the derived fleet-sizing overlay. What makes 05 worth running
separately from 04 is not that it forecasts — 04 already did — but the **lifecycle asymmetry**: an
ephemeral cluster must be deleted when its run ends, a named one must not. Both halves held. After
04, `sf-cluster-smoke-04-…` was gone; after 05 submitted two family jobs to `sf-smoke-cluster` and
reached `COMPLETED`, that cluster was still `RUNNING`. A reuse path that tore down a cluster it did
not create would pass every leaderboard assertion in the harness and still be badly wrong, so the
survival check is the assertion, and it is made against `clusters list`, outside the harness.

The standing cluster is a **campaign fixture, not infrastructure**: created here for smoke 05 and
deleted at teardown. It is built through `dataproc_cluster.build_cluster` with the same
`_resolve_cluster_deps` / `_stage_cluster_init` pair `provision_shared_cluster` uses, so it is shaped
like an ephemeral cluster in every respect but its name. Standing it up with a hand-written
`gcloud dataproc clusters create` would have made smoke 05 a test of *that command's* fidelity to
the product rather than of the reuse path.

### Smoke 08 is blocked on quota, not broken — and finding that out took two fixes

**GPU on Vertex Ray is unavailable to this project — in every region, on either accelerator.** On
2026-09-01, smoke 08 was attempted six times across `us-central1`, `us-east1` and `us-west1`, first
on T4 and then on L4 (`g2-standard-8`), and never once provisioned. `us-east1` on T4 gave the only
message that named anything: *"The following quotas are exceeded:
`CustomModelTrainingT4GPUsPerProjectPerRegion`"*. The rest were *"An internal error occurred on your
cluster"* and *"Unexpected response."* — which, given the one region that did explain itself, are
almost certainly the same ceiling wearing different masks.

The row is `NEEDS_RECHECK` rather than `STALE`: nothing here suggests the code is wrong, but
2026-08-28's pass recorded no `run_id`, and it now cannot be re-earned to fix that. **This is a
project-entitlement blocker, not an architecture one** — it needs a Vertex GPU quota grant, and
until then every Ray-GPU config in this repo (`ray_gpu_demo`, `all_families_100k`,
`all_families_100k_full`) fails at provisioning.

There is a config-level route around it, and it is worth stating because it is the product's own
answer: **deep learning does not have to run on Ray.** Smoke 03 passed the same day on Dataproc
Serverless L4, and `14_full_dag.json` already puts the DL family there. A deployment without Vertex
Ray GPU entitlement can run every family by pointing `compute.families.deep_learning` at
`runtime: spark`, which is a config edit, not a code change.

The trap worth carrying: **Compute Engine quota does not tell you whether a Ray GPU run can start.**
`NVIDIA_T4_GPUS` reads 4-of-4 free in `us-central1`. The quota a Vertex Ray cluster actually spends
is `CustomModelTrainingT4GPUsPerProjectPerRegion`, a different meter, and it is the one that says no.
Checking the former before a run is worse than not checking — it gives a green light that means
nothing.

Two defects in the region fallback surfaced only because a region actually ran out, and both were
fixed and confirmed live the same day. They are the same defect twice: **an error the classifier
cannot parse was treated as an error it had diagnosed.**

- **An opaque failure disabled the fallback that exists for it.** `_is_generic_cluster_error`
  already reasoned that the SDK's contentless *exception* means "hop", but it only applied when the
  resource carried no message — and `"An internal error occurred on your cluster"` *is* a message.
  So a config listing three regions tried exactly one and re-raised. Vertex's own advice for that
  string is "try recreating"; hopping is a strictly better version of it.
- **A real quota error went unrecognised because of word order.** `us-east1` said "quotas **are**
  exceeded" and the marker list held `"quota exceeded"`, `"exceeds quota"`, `"exceeded quota"`,
  `"quota limit"` — the textbook hoppable case, misread as a config fault. The classifier now
  composes ("quota" near a word of exhaustion) instead of enumerating phrasings.

Neither is visible offline for the usual reason: the unit tests asserted the classifiers correctly
matched the strings someone had thought of. Only a live region running out produces a string nobody
thought of. The fallback now walks all three and raises an `EngineError` naming them, which is what
the plan's abort path expects — a defer, not a block.

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
| `explode_demo.json` | The Spark `explode` fan-out, statistical + ML, artifacts persisted (10) | CURRENT | 2026-09-01 | `explode-demo-d1b57690dc96` | `serverless_deps=container-image`, `python=3.11`, `fleet_sizing=derived-overlay`, `run_id_inputs=authored-config-only`, `horizon_features=computed-at-future-dates` |
| `mixed_demo.json` | One Spark model and the natives under one `run_id`, backtested (10) | CURRENT | 2026-09-01 | `mixed-demo-405983dddf0a` | `serverless_deps=container-image`, `python=3.11`, `fleet_sizing=derived-overlay`, `run_id_inputs=authored-config-only` |
| `ensemble_demo.json` | The same mix with three ensemble strategies on (10) | CURRENT | 2026-09-01 | `ensemble-demo-9849a2f73669` | `serverless_deps=container-image`, `python=3.11`, `fleet_sizing=derived-overlay`, `run_id_inputs=authored-config-only` |
| `per_family_runtimes_demo.json` | Per-family runtime split — deep learning to Ray GPU, the rest on Spark (50) | NEVER_RUN | — | — | — |
| `ray_cpu_demo.json` | Ray on Vertex, CPU, alongside the natives, backtested (6) | CURRENT | 2026-09-01 | `ray-cpu-demo-f6b6fbdb83a5` | `ray_deps=stock-image+uv-runtime-env`, `python=3.11`, `run_id_inputs=authored-config-only` |
| `ray_gpu_demo.json` | Ray on Vertex, GPU T4 (`neuralprophet`), alongside the natives (6) | NEVER_RUN | — | — | — |
| `ray_autoscale_demo.json` | **The shipped `ray_autoscale=true` default**, 1→8 CPU nodes at 10,000 series | CURRENT | 2026-09-01 | `ray-autoscale-demo-886a053c374c` | `ray_deps=stock-image+uv-runtime-env`, `python=3.11`, `run_id_inputs=authored-config-only`, `horizon_features=computed-at-future-dates` |
| `explode_100k.json` | The headline: Spark `explode` over 100,000 series | CURRENT | 2026-09-01 | `explode-100k-1c59265062aa` | `serverless_deps=container-image`, `python=3.11`, `fleet_sizing=derived-overlay`, `run_id_inputs=authored-config-only`, `horizon_features=computed-at-future-dates` |
| `ray_100k.json` | The same work on Ray — the runtime-parity half of the scale review | NEVER_RUN | — | — | — |
| `all_families_100k.json` | Every family at 100,000 series under one `run_id` (Ray + BigQuery, T4) | NEVER_RUN | — | — | — |
| `all_families_100k_full.json` | As above, plus backtesting and persisted artifacts | NEVER_RUN | — | — | — |

**Four demonstration configs are blocked on the same thing, and it is not the code.**
`ray_gpu_demo`, `per_family_runtimes_demo`, `all_families_100k` and `all_families_100k_full` all
put a family on Vertex Ray GPU, which this project cannot provision in any region on either
accelerator (see smoke 08 above). They stay `NEVER_RUN` rather than being quietly re-pointed at
Serverless: the whole point of `per_family_runtimes_demo` is the *split*, and a version of it that
runs everything on Spark would prove something else while keeping the name.

The three Spark demo rows landed together on 2026-09-01, and two of them are worth reading past the
`CURRENT`:

- **`mixed_demo` is the cross-runtime comparability claim, live.** One `run_id`, one leaderboard,
  `theta` from a Dataproc Serverless batch ranked against `arima_plus` and `timesfm` from a BigQuery
  job on backtested WAPE (0.451 / 0.418 / 0.391). Two runtimes, one ranking, no manual join.
- **`ensemble_demo` adds the ensemble node as a third job** (BigQuery), and its three strategies
  rank *inside* the same board: `inverse_error` 0.408, `mean` 0.411, `median` 0.413 — all three
  beating both `arima_plus` and `theta`, none beating `timesfm` at 0.391. Recorded as-is. The claim
  the product makes is that ensembles are produced, ranked and comparable, not that they win.

**`explode_100k` is the headline claim, and it is now a citation.** 100,000 series × 4 models =
**400,000 cells**, `COMPLETED`, all four models on the board at `n_cells=100000` each, and the
re-run resolved the same `run_id` and deduped — dedupe-on-read holds at scale, not just at 100
series. Two families, one `run_id`: `statistical` (theta / holtwinters / sarimax) ran 117.6 min,
`ml` (xgboost) 55.5 min.

Read the wall time with the ceiling in mind. This run was deliberately capped at
`max_executors: 20` — 80 cores per family against a 200-core project — so ~2 hours is what 400k
cells cost *on a fifth of the fleet the arithmetic asked for*, not what the architecture costs.
`sarimax` is the long pole by a wide margin; the ~0.5 s/cell measured for theta/holtwinters/xgboost
does not describe it. A project with quota to spare should expect the uncapped fleet to be several
times faster, and that comparison is exactly what wave 8's A/B is for. What this row establishes is
the claim the product actually makes — 100k series, four models, one run, one leaderboard, and it
finishes.

**`ray_autoscale_demo` proved the shipped default, and the proof is in the audit log rather than the
leaderboard.** `ray_autoscale=true` is what every config gets unless it says otherwise, and until
this run nothing had ever exercised it: all four Ray smokes pin it `false`. The cluster came up with
`autoscalingSpec {minReplicaCount: 1, maxReplicaCount: 8}` and Vertex drove it to **8 worker
replicas** — six `UpdatePersistentResource` events over twenty minutes are the scale-up. 3 models ×
10,000 series, `COMPLETED`, re-run same id and board unchanged, cluster deleted cleanly at the end.

Three operational numbers worth carrying out of it. Cluster provisioning took **10 min 10 s**
(create 19:47:51 → start 19:58:01) — the Ray equivalent of the ~30-minute fixed Serverless batch
overhead, and it is charged before any work starts. The job itself ran ~79 minutes and **did not hit
the bearer-token expiry** that a Ray run over ~60 minutes is documented to risk; that limit is
narrower than assumed, but one run is not enough to call it closed. And `wape` is `None` across the
board because this config does not backtest — the row proves autoscaling and scale, not accuracy.

### What this surface will exercise that the smoke suite cannot

- **`ray_autoscale=true`.** Every Ray *smoke* pins it `false`; five configs here leave it `true`,
  which is the shipped default. Proven once, by `ray_autoscale_demo` above; the gap below is now a
  narrower one about the four smokes still pinning it off.
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
  provisioning. **Resolved on the demonstration surface 2026-09-01**, and the suspected cause was
  right: `ray_autoscale_demo` scaled 1→8 workers and completed, so the `4c988bc` crash was the
  custom image — since deleted — not the autoscaling spec. What remains is smaller and is a
  *hygiene* gap rather than an unknown: the four Ray smokes still pin `false`, so the cheap
  fifteen-minute path does not cover the shipped default, and a regression in it would only surface
  on a demonstration run. Unpinning them is the fix; it is not urgent, because the default is now
  proven at 10,000 series, which is a harder case than any smoke poses.
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
  02 (`…-439b5350249b`, `…-0ffcc1f22d54`) joined this category hours after being written. They
  remain valid pointers into the registry and the results they point at are unaffected; they are
  simply no longer recomputable from their configs. Smoke 01's row has since been re-earned by a
  post-fix run and carries a recomputable id again (`…-5af5de1accf2`); smoke 02's has not, and its
  id stays a pointer-only. Unlike the three before it, this move makes
  identity *stop* drifting rather than start: `run_id_inputs` is now `authored-config-only` and
  there is no resolved value left in the digest to move it again.
  The change that *did* make live results stale arrived at W7b/W8, and it was not the one predicted
  here. This note used to say the staleness event would be W6, "when `profile.mode='auto'` starts
  actually sizing fleets from measurement." That never happened and now never will in that form —
  see the next gap. What moved the fleets was the **static** arithmetic W7/W8 wired in with the
  profile argument left as `None`: no measurement involved, and every Spark fleet reshaped anyway.

- **`compute.profile.source` in the digest forked run identity two different ways. Fixed and
  confirmed live 2026-09-01.** Kept here rather than deleted, because how it was found is the
  point: both halves were found live, in the campaign's first two waves, and neither had an offline
  analogue — the offline suite contained a test asserting the *forking* behaviour was correct.

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

  **Confirmed live the same day, on the exact assertion that failed.** Smoke 01 re-ran post-fix as
  `smoke-01-serverless-cpu-5af5de1accf2` and reported `rerun: checked (same id, board unchanged)` —
  `RESULT: PASS`. The re-run resolved the same id, deduped on read, and submitted no second pair of
  batches (the Dataproc batch list for that config shows two for run 1 and none for the re-run,
  against four for the pre-fix invocation). Discovery was *working* during that run, not silently
  degraded: it found run `…-8f602110b7ea`'s harvest and pinned it, and the id still did not move.
  That is the case that matters — an exclusion is only proven by a config whose resolved value is
  non-trivial. Offline, this is held by the inverted test plus one asserting the id is identical
  whether or not discovery reached the registry.
- **The derived fleet had no infrastructure ceiling, and at 100k that was fatal rather than slow.
  Fixed and confirmed live 2026-09-01.** `explode_100k`'s
  statistical batch was rejected at submit: *"Insufficient 'CPUS' quota. Requested 380.0, available
  200.0."* Nothing was mis-sized — the arithmetic correctly answered *how wide would this run like
  to be*, which at 400,000 cells is 95 executors. It has no way to know *how wide may it be*, and
  the two never met.

  What made this a product defect rather than a small project's quota problem is where the existing
  ceiling lived. `--max-executors` / `submit(max_executors=…)` has always existed, but **every job
  the DAG launches is launched from a config**, so a ceiling reachable only through a CLI flag is
  one an orchestrated run can never set. `ComputeConfig` had `max_parallelism` — documented as a
  cost guardrail, and named as an override in `architecture.md` — but it only ever fed bucket
  sizing and Ray's fallback basis, and never reached the Spark fleet. The fix adds
  `compute.max_executors`, defaulted to `None` (unchanged behaviour) and consulted by both the
  Serverless and the cluster sizing paths, with an explicit argument still winning over it.

  Worth noting for anyone reading the arithmetic: this is not a scale wall. 400,000 cells at the
  ~0.5 s/cell the same campaign measured is about 200,000 CPU-seconds, or ~20 minutes across a
  200-core quota. The fleet wanted to be twice the size of the project, not twice the size of the
  problem. Budget for concurrency when setting the knob — a run's families submit simultaneously, so
  the two-family `explode_100k` at 20 executors × 4 cores needs ~168 cores including drivers.

  **Confirmed live the same day.** The capped `explode_100k` was accepted at submit, peaked at 152
  of the project's 200 cores, and ran to `COMPLETED`; the batch carries
  `spark.dynamicAllocation.maxExecutors: "20"` from the config alone, with no CLI flag anywhere in
  the path. The uncapped attempt is the control: its ML batch, which *was* accepted, held 192 of
  200 cores by itself — the whole project — which is the same defect seen from the other side.

  A second-order observation from the same failure, recorded rather than fixed: **`discover_harvest_run`
  selects the most recent harvest, not the best-matched one.** The 100k run discovered the 10-series
  demo harvest that had just been written and warned `10000x (10 measured vs 100000 planned)`, when a
  100-series harvest from smoke 01 was sitting in the same table. The degradation to
  `basis: "reference"` behaved correctly, so this cost accuracy rather than correctness — but
  "most recent" is a weak selector once more than one run has harvested, and the signature it already
  computes is the obvious thing to rank on.

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

  A third measured run (`…-5af5de1accf2`, the post-fix re-run) added the observation the first two
  could not, because they moved only one family's memory: **the two families in one run derive
  different slot shapes from the same profile, and the overlay follows.** From one harvest,
  `statistical` resolved `slot_cores=2` → `spark.task.cpus=2`, thread pins at `2`, and a
  `maxExecutors=13` band; `ml` resolved `slot_cores=1` → no `spark.task.cpus` at all, thread pins at
  `1`, `maxExecutors=4`, and `memoryOverhead=3834m`. Per-family sizing is real rather than a
  per-run constant wearing a family label, and the two overlays were written under one `run_id`
  without either clobbering the other.

  **The signature check fired live too, and it degraded rather than lied.** The three demo configs
  run 10 series and discovered smoke 01's 100-series harvest. `explode-demo-d1b57690dc96`'s
  provenance records `basis: "reference"` — *not* `"measured"` — with
  `warnings: ["series count differs by 10x (100 measured vs 10 planned)"]`, while still naming the
  source run and its signature. That is the designed path for a profile that is *informative but
  not representative*, and it had never executed before. Two things make it worth a paragraph: the
  warning reached the **audit trail**, not just the driver log, so a reader of the registry a month
  later can see the fleet was sized from mismatched evidence; and the demotion is visible in a
  field (`basis`) that distinguishes three states now rather than two.

  One thing to watch, recorded as an observation and not a defect: `statistical` measured
  `max_effective_cores` of **1.05** and was sized to a **2-core** slot. The ceiling is doing what it
  was written to do, but at 100k series a 2× slot from a 5% overshoot is the kind of rounding that
  is cheap here and expensive there. Wave 10 is where that becomes measurable.

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
  one header, with no sign of a race. The third — whether the **cluster** path's stamp lands at all,
  which smoke 01 could not answer because it is a Serverless run — closed on 2026-09-01 in smoke 04:
  `smoke-04-cluster-cpu-c5b992778fd1`'s header carries `sizing.ml` and `sizing.statistical`, both
  complete, both with `plans[].runtime = "cluster"` and a full `translation` block (7 executor cores,
  `maxExecutors` 2, a derived `worker_count` of 2, and the thread pins). `submit_cluster_job` had
  never written header telemetry before W9b, so this row used to be blank. **All three of W9b's
  assumptions now hold against real BigQuery.** Every one of these writes is best-effort (logged and
  swallowed), so a wrong assumption degrades to "no telemetry", never to a failed run.

  **The same payload is the clearest evidence yet for the `discover_harvest_run` defect**, listed
  below as an unfixed observation and until now only reasoned about. Smoke 04 runs `statistical` and
  `ml`; the profile it picked up is from `smoke-03-serverless-gpu-a918f22d7970`, the most recent
  harvest, which measured **only** `deep_learning` (neuralprophet, 100 fits). So the sizing that
  actually shaped the run fell back to `slot.basis = "static"` with `cores`/`memory_bytes` under
  `assumed` and `measured` empty — a harvest was found, carried, and stamped, and it contributed
  nothing. The run is not wrong (static is the documented fallback and 100 series is trivially
  sized), but "a profile was discovered" reads as "the fleet was sized from measurements" in the
  telemetry, and here it was not. Selecting the most recent harvest instead of the best-matched one
  is what produces that gap.

## Provenance confidence

Entries dated before 2026-08-29 were **reconstructed** during a reconciliation on that date, from
the results log and commit history. Their axis values are inferred from what the code did at the
time, not recorded when the run happened, and the missing `run_id`s cannot be recovered. Entries
from 2026-08-29 onward are recorded at run time and are authoritative.
