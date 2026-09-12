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
| `gpu_cluster_image` | `driver-init-action` | owner decision (2026-09-09), reverting `254fe4f` | `prebaked-driver-image` |
| `native_source_pin` | `unpinned-all-sources` | `9af322a` (2026-08-25) | `unpinned-iceberg-only` |
| `python` | `3.11` | `515ecb0` | mixed per surface |
| `run_id_inputs` | `authored-config-only-v3` | 6.5+6.6 (2026-09-09) | `authored-config-only-v2` (P3, 2026-09-05), before that `authored-config-only` (`a22e94c`, after the fork below), before that `+compute.profile.source` (W11a) |
| `fleet_sizing` | `derived-overlay-three-way-min` | P6 (2026-09-07) | `derived-overlay` (W7b `6f4638f` + W8 `be78bec`, 2026-08-31), before that `platform-defaults` |
| `horizon_features` | `computed-at-future-dates` | `cb7d15f` (2026-08-31) | `first-rows-of-history` |
| `ray_pool_shape` | `autoscaling` | F5 (2026-09-03) | `fixed-size` (pinned by `4c988bc`) |
| `ray_slot_memory` | `harvest-only` | `efecb4c` (2026-09-04) | `driver-rss-prepass` |
| `dl_gpu_routing` | `resolved-per-family` | P1 (2026-09-05) | `flat-compute.use_gpu` |
| `backtest_scoring` | `holdout-reserved+embargo-aware+auto-refit` | Phase 6 (`11401bf`, `fae9cee`, `834ad6a`, 2026-09-09/10) | `holdout-fold-reserved` (3.2, 2026-09-08, after 2.3/2.4/2.6/3.1), before that `unreserved-full-history` |
| `gpu_device_probe` | `trainer-root-device` | Tier 1 campaign (2026-09-09) | `parameter-tensor-after-fit` |
| `serverless_gpu_allocator` | `rapids-pool-released` | Tier 1 campaign (2026-09-09) | `rapids-default-pool` |
| `job_status` | `derived-from-cell-tallies` | `bf12e96` (2026-09-10) | `launch-call-returned` |
| `gpu_batch_churn` | `executor-failure-budget+stall-watchdog` | 2026-09-10 | `unbounded-executor-replacement` |
| `gpu_fault_injection` | `probe-mode-default` | 2026-09-10 | `cuda-visible-devices-emptied` |
| `ray_poll_recovery` | `transient-transport+auth` | 2026-09-10 | `auth-expiry-only` |
| `serverless_cancel` | `operation-cancel` | Tier 5 campaign (2026-09-11) | `batch-delete` (could not stop a live batch at all) |
| `ensemble_weighting` | `per-series-calculated+batch-fit-learned` | Tier 6 wave A (2026-09-11) | `gather-order-dependent` (registered at its broken value the same day, then flipped by the fix; see the note below) |

**`backtest_scoring` is the axis nothing else can see.** The others move something a reader could
notice on their own — a different image, a different `run_id`, a different node count. This one
moves what a metric column *means* while leaving the `run_id`, the row counts and the schema
exactly as they were. A run scored before it moved and a run scored after it produce the same
number of rows in the same tables under the same identity, and the numbers inside are not
comparable: the newest fold is now reserved from every fit, and MASE and RMSSE are scaled by the
fold's own training window rather than by the whole series. Declaring it on a row is the only
mechanism that will ever say so.

**It carries three changes, and the reason they share one axis is that a reader has one question.**
Phase 6 moved it again on 2026-09-09/10: folds now forecast *across* the embargo instead of stopping
at it (three places in the code disagreed about that, `11401bf`), `auto` became the refit scheme a
backtest earns by default, and refitting schemes gained an optional never-refreshed control arm
(`fae9cee`). Bundling them costs precision — a future reader cannot tell from the axis alone which
of the three moved a particular number — and buys the thing that matters more: there is exactly one
value to check before trusting any accuracy figure in this document, rather than three that must all
be read together to mean anything. The commits are named in the table for anyone who needs to go
further.

**`gpu_device_probe` is the second of that kind, and it invalidated more than it broke.** It names
how a fit answers the question "which device did you actually run on" — the answer that
`device_audit` turns into a verdict. Until 2026-09-09 the answer was read off a parameter tensor
after the fit had finished, and PyTorch Lightning moves the module back to the CPU on its way out,
so the answer was always `cpu` and the verdict was always `MISSING_DEVICE`. Nothing about a run
looked different; the rows, the timings and the identity were all normal. What it means is that
**no GPU-contract result recorded before that date was earnable**, because the measurement could not
produce a passing answer. The section on the 2026-09-09 wave below has the detail.

**`job_status` is declared by one row, on purpose.** It names where a `run_jobs` row's terminal
status comes from: until 2026-09-10 it came from the launch call returning without raising, and now
it comes from that attempt's cell tallies in `forecast_metadata`. Only the three negative arms
declare it, because only their *claims* depend on it — they are the entries asserting that a run
which forecast nothing closes `FAILED`. Every other row was recorded from a run whose cells did all
succeed, so the old derivation and the new one agree on it: `COMPLETED` was the right answer under
both, and the old one merely reached it without checking. Rows will pick the axis up as they are
re-run.

**`gpu_fault_injection` names how the negative arms take the card away, and it decides what they can
prove.** `cuda-visible-devices-emptied` is the faithful imitation of a lost accelerator and the
useless test: on Ray and on Serverless something below our code — the worker holding the GPU slot,
the RAPIDS executor plugin — reacts to the missing device first, so the fault never arrives at the
check it was aimed at. `probe-mode-default` (`SF_HIDE_DEVICES=probe`) makes our own device probe
report `cpu` and leaves CUDA alone, which delivers the fault to `_require_device` on all three
services. Only the three negative arms declare the axis; no other row arms the switch at all.

**`gpu_batch_churn` is declared by one row for the same reason.** It names what stops a Dataproc
Serverless GPU batch that cannot make progress. Until 2026-09-10 nothing did: Spark replaced a dying
executor without limit and the batch ran to its four-hour TTL, which is what smoke 17 recorded when
its RAPIDS plugin lost the card. A GPU batch now carries a bounded executor-failure budget
(`spark.executor.maxNumFailures`, scaled to the fleet with a floor of eight, over a 30-minute
window), and the driver additionally cancels a batch that has written no forecast row within
`SF_STALL_GRACE_S` (45 minutes by default, off when set to `0`). Only the smoke 17 row declares the
axis, because only that row's claim is about a batch that never progresses — for every other run the
budget is never spent and the watchdog stands down the moment the first cell lands, so both values
of the axis produce the identical run. The bound lives on `BatchInfra`, not on `ComputeConfig`, so
arming or changing it cannot move a `run_id`.

**`ray_poll_recovery` names which failures of the monitoring channel a Ray run can survive**, and
long runs are the only ones that can tell the two values apart. A Ray run is watched by a driver
that asks the Vertex dashboard proxy for the job's status every fifteen seconds; over a two-hour run
that is several hundred small HTTPS requests, and none of them are the job. Until 2026-09-10 the
poll recovered from exactly one failure — an expired OAuth token — and re-raised everything else,
which meant a single dropped request was treated as a verdict on a run it could not actually see.
`transient-transport+auth` widens that to the transport faults the connect path had already
classified as transient. The reason this is an axis rather than a bug note is that it changes what a
completed long Ray run *proves*: under `auth-expiry-only`, finishing meant the work succeeded **and**
several hundred consecutive network requests happened to survive, so a failure was ambiguous between
the two. Only the rows whose runs are long enough to be exposed declare it; a six-series Ray smoke
polls a handful of times and would finish under either value.

> ### Every row in this document is STALE, on purpose, as of 2026-09-05
>
> `run_id_inputs` moved to `authored-config-only-v2` and every row declares that axis, so all 28
> config rows and all 8 notebook rows went stale in one commit. That is the intended outcome of a
> planned change, not a discovery: six new fields landed on `RunConfig` together — five `backtest`
> knobs and `model_params` — and since `run_id` is a digest of the whole config, every recorded
> identity moved with them.
>
> **Read this the right way.** Nothing here has been shown to be *wrong*. Smoke 13 demonstrated that
> the BigQuery-native path reads a native source table, and an inert `backtest.gap` field does not
> unprove it. What every row has lost is narrower: the `run_id` it records is now a pointer into the
> registry that no config reproduces, so a result can no longer be tied back to the exact
> configuration that produced it. For a document whose entire job is to say *what has been proven,
> and on what*, that is enough to withhold the word CURRENT until a run re-earns it.
>
> **This reverses a precedent, which is why it is written down.** Three earlier changes (W5, W10,
> W11a) moved every id the same way, and each time this file said the opposite: "No row was marked
> STALE for this, deliberately — `run_id_inputs` is not an axis any of those claims rests on." That
> reasoning was locally correct and cumulatively wrong. It let identity drift four times with no
> mark anywhere, and the result was six rows quietly recording ids their configs no longer produced
> for four days before anyone noticed. Grading it as staleness costs a re-run; not grading it cost
> the ability to tell which rows were affected at all.
>
> The fields were batched into one break for the same reason. Adding them one per release would
> have moved every identity six times, and the re-validation campaign that follows is the moment
> the whole surface is proven once, together, on identities that are final.
>
> ### A second break landed on 2026-09-09, and the sentence above was premature
>
> `run_id_inputs` is now `authored-config-only-v3`. Two changes in one commit moved every identity
> again: `backtest.control_arm` is a new field, and an unset `output.point_forecast` now resolves to
> `auto` under a backtest instead of to the fleetwide rule the decision metric implies. Either one
> alone would have done it — a defaulted field appears in every config's dump, and so does a
> resolved default, so both reach the digest for configs that never mention them.
>
> **What this costs and what it does not.** Nothing above becomes any less proven than it already
> was; every row was STALE from the first break and stays STALE for the same narrow reason, that the
> `run_id` it records is a pointer no config reproduces. The cost is to the *plan*: 2026-09-05
> claimed the identities were final and they were not, so anyone who re-ran a config between the two
> dates got an id that is already obsolete again. If you did, the run itself is fine — re-derive its
> id from the config with `plan_run` rather than trusting a transcript.
>
> **The freeze held; the enforcement moved.** `run_ids_prebreak.json` cannot catch this, because
> after `_BREAK_LANDED` it only asserts that ids *differ* from their pre-break values, and they
> still do. What failed the gate was `run_ids.json`, the post-break snapshot, which pins the current
> digest of all thirty-one shipped configs. That is the file that guards the surface from here on,
> and it is the one a future addition has to answer to.

`native_source_pin` governs **native BigQuery table** reads on the BQML `CREATE MODEL` path only;
Iceberg sources were already un-pinned before the change, so entries that read Iceberg do not
declare this axis.

`dl_gpu_routing` is narrower than it looks, and the scoping is deliberate rather than an omission.
It governs how `engines/ray_engine` decides whether *this* job has a device, so only a row with a
deep-learning family **on Ray** declares it. Two groups are therefore left alone. Spark rows never
enter that file at all. Rows whose config asks for a GPU through the flat `compute.use_gpu` field —
`ray_gpu_demo`, `all_families_10k`, `all_families_10k_full` — are unaffected because the old code
and the new code return the same answer for a flat config: the old path read `compute.use_gpu`
directly, the new one asks `resolve_family_compute`, which for a config with no family override
resolves to exactly that flat field. Only the per-family shape made the two disagree.

`fleet_sizing` governs **how a fleet's shape is decided** — how wide a unit is, how many cells fit
on it, and how many units the fan-out therefore needs.

It began as a Spark-only axis and has moved twice. Until W7b/W8 we stated a worker or executor
*count* and let the platform choose everything else: Dataproc Serverless picked its own executor
cores, memory and dynamic-allocation band, and a Dataproc cluster ran its default two 4-core
executors per worker with nothing bounding a GPU. `derived-overlay` replaced that with an explicit
shape from `resources.translate_serverless` / `translate_cluster` — executor cores, memoryOverhead,
the dynamic-allocation min/initial/max, `spark.task.cpus`, the thread pins, and a derived worker
count — submitted as a properties overlay.

`derived-overlay-three-way-min` (P6) changes the **packing arithmetic underneath both runtimes**, so
the axis is no longer Spark-only. Two things moved together. A unit now holds back one core for
itself, because the raylet, the executor JVM, the log shipper and the OS all want somewhere to run
and the cell scheduled onto the last core does not fail — it time-slices against the agent that is
supposed to be reporting its progress. And density is now the smallest of three bounds (devices,
cores, memory) rather than the memory bound overlaid onto whichever single axis defined the slot.
The device axis in particular used to stand alone: a GPU slot's density was
`accelerators x floor(1 / gpu_fraction)` with no core term at all, so a calibrated fraction could
promise more concurrent cells than the node had cores to run them on. **Both effects change how
many cells land on a node and therefore how many nodes the fan-out derives**, which is the quantity
most rows below are measuring, so a result proven on the old arithmetic is a claim about a fleet
that no longer exists.

**BigQuery-native** work still does not declare it — there is no fleet of ours to shape. **Ray** now
does, which is the reversal P6 forces. The old text excluded Ray on the grounds that the axis was
about a Spark properties overlay, and added that `plan_pool(profile=None)` reproduced the
pre-profiler arithmetic exactly. That second claim is what P6 retires: an unprofiled Ray CPU pool
now lands one cell per node below the machine's nameplate core count, so every Ray row is sized by
arithmetic it did not run under. Ray rows are therefore re-declared with the axis at the value they
*did* run on, `derived-overlay`, which is exactly the mark that says a re-run is owed.

The Ray sizing history the old note recorded is worth keeping. `ray_100k` is the first Ray run whose
*slot* came from a measurement (`basis: measured`, 2 cores and 1.29 GiB per task, from a prior Ray
harvest); every Ray row above it was sized from the constants. W1's autoscale-ceiling derivation
only fires when `ray_autoscale` is true, which until 2026-09-03 no Ray smoke did — the demonstration
surface covered that path alone (`ray_autoscale_demo`, which reached the derived ceiling of 8) — and
the four smokes dropped the pin and re-ran on 2026-09-03/04. W2's device catalog left T4 at 16 GiB
(only L4 moved). Smoke 10 declared the axis even under the Spark-only reading, because it submits
Serverless work alongside its Ray families.

`ray_pool_shape` is the Ray-side counterpart: **whether a worker pool is provisioned with an
`AutoscalingSpec` or at a fixed `node_count`.** These are two different provisioning calls, not two
settings of one, and the difference has bitten before — a per-pool `AutoscalingSpec` is what crashed
the Vertex Ray head in `4c988bc`. Only Ray entries declare it. A pool that autoscales also makes
W1's derived ceiling live, so the fan-out arithmetic reaches the cluster instead of sitting inert.

`ray_slot_memory` is the other half of the Ray sizing story, and it is about **density inside a
node** rather than the number of nodes. `ray_engine` runs a driver-side measurement pre-pass
(`profiling.source.resolve_profile`) and turned what it measured into the task's Ray `memory`
request. Ray treats `memory` as a hard scheduling resource, so that request decides how many cells
a node runs at once. The pre-pass kept the fits' absolute `process_rss_bytes` — the right axis for a
slot, but only when the process measured is the process the slot describes, and the driver is not.
It holds the whole source panel resident on a head node a class larger than the workers. `efecb4c`
drops the memory axis from the pre-pass (`_without_driver_rss`); a real memory bound is now earned
only from a **prior run's harvest**, whose rows were written by the worker processes themselves,
and with no basis the slot requests nothing and packs on cores.

**This changes how many cells a Ray node runs concurrently, so it changes wall-clock at fixed
quota** — which is what the affected rows below were measuring. It reaches node *counts* too, not
just density: `ray_autoscale_demo`'s headline is "1→8 CPU nodes at 10,000 series", and a fleet that
packs seven cells per node instead of one has a different reason to reach 8. The mechanism claims
those rows also carry — that autoscaling provisions elastically, that Ray reaches 100,000 series —
are untouched; it is the numbers that are stale. Only Ray rows whose fan-out reached
`compute.profile.min_cells` (1000 cells, the `mode="auto"` threshold) declare it: below that the
pre-pass returns `None` and never sized anything, which is why the Ray smokes and the 6–50-series
demos are untouched. The rows that do declare it are the three large ones — `ray_autoscale_demo`,
`ray_100k` and `all_families_10k`. Two are marked from the *mechanism* rather than from a
measurement of what each one lost — the pre-pass ran on all three and what it feeds the slot has
changed. On the third the cost is measured, and it is large: `all_families_10k` was re-run under the
fix the same day and came in **3.8x faster end to end**, with the deep-learning family 4.1x faster.

**All three have now been re-earned, and the mechanism scoping held on every one.**
`ray_autoscale_demo` re-ran on 2026-09-05 in **2,513 s against 5,223 s — 2.08x** on the same config,
same `run_id`, same eight-node ceiling. The dashboard read mid-run is the cleanest picture of the
defect and its fix anywhere in this file: every one of the eight workers at `CPU [7.0, 7.0]`, 56
cells in flight and 1,732 queued, where the pre-fix regime put roughly one cell on a node with seven
idle cores. Nothing about the pool changed — same machine type, same `max`, same autoscaler. Only
what a task claimed it needed.

`ray_100k` followed the same day and is the one that matters most, because 11c's original
0.97-cells-per-node measurement was taken on it:

| Family | Cells | Before (2026-09-03) | After (2026-09-05) | |
|---|---|---|---|---|
| `ml` (`xgboost`) | 100,000 | 17,666 s (4 h 54 m) | **3,681 s** (1 h 01 m) | **4.80x** |
| `statistical` (3 models) | 300,000 | 19,069 s (5 h 18 m) | **6,290 s** (1 h 45 m) | **3.03x** |
| whole run | 400,000 | ~19,800 s (5 h 30 m) | **6,923 s** (1 h 55 m) | **2.86x** |

At full scale the pool held **139–140 of 140 cores busy** across 20 nodes with 38,000 tasks queued
behind it — against ~20 concurrent cells before. That is the 7x density the fix predicted, arriving
at 3.0x wall clock for the same reason the GPU run did: cells contend once they are actually packed
together. This is also the run [quota and scale](quota_and_scale.md) plans from: its CPU throughput
anchor moves from **72 to 191 cells/min per 8-vCPU node** on the strength of it. Downstream, 100,000
series x 6 models on a stock 200-vCPU project goes from a seven-hour job to a two-and-a-half-hour
one, and the quota to finish it inside an hour falls from ~1,150 vCPUs to **~460** — the difference
between a conversation with an account team and a routine request.

This is the second time the Ray memory axis has cost a run. On 2026-09-03 `ray_100k` sat at zero
cells for 57 minutes behind an unschedulable ~21 GiB per-task request, recorded at the end of this
file.
That one asked for more than a node had and never placed; this one asked for 97 % of a node, which
places perfectly well and then fits exactly once. A schedulability guard catches the first and
cannot catch the second, which is why `efecb4c` also added `RuntimeResourcePlan.binding_axis`.

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
| 01 | `01_serverless_cpu.json` | Spark on Dataproc Serverless, CPU (statistical + ML) | CURRENT | 2026-09-09 | `smoke-01-serverless-cpu-7a3d4234e0e1` | `serverless_deps=container-image`, `python=3.11`, `fleet_sizing=derived-overlay-three-way-min`, `run_id_inputs=authored-config-only-v3`, `horizon_features=computed-at-future-dates` |
| 02 | `02_bq_native.json` | BigQuery-native models (`arima_plus`, `timesfm`) | CURRENT | 2026-09-11 | `smoke-02-bq-native-e354a8652712` | `native_source_pin=unpinned-all-sources`, `python=3.11`, `run_id_inputs=authored-config-only-v3` |
| 03 | `03_serverless_gpu.json` | Serverless GPU (deep-learning on an L4) | CURRENT | 2026-09-09 | `smoke-03-serverless-gpu-92763e0f2242` | `serverless_deps=container-image`, `serverless_gpu_allocator=rapids-pool-released`, `gpu_device_probe=trainer-root-device`, `dl_gpu_routing=resolved-per-family`, `python=3.11`, `fleet_sizing=derived-overlay-three-way-min`, `run_id_inputs=authored-config-only-v3`, `horizon_features=computed-at-future-dates` |
| 04 | `04_cluster_cpu.json` | Spark on an ephemeral Dataproc cluster, CPU — and the create-then-delete half of the lifecycle: its cluster was `NOT_FOUND` the moment the run ended | CURRENT | 2026-09-12 | `smoke-04-cluster-cpu-9196365250ac` | `cluster_deps=packed-venv-init-action`, `python=3.11`, `fleet_sizing=derived-overlay-three-way-min`, `horizon_features=computed-at-future-dates`, `run_id_inputs=authored-config-only-v3` |
| 05 | `05_cluster_reuse.json` | Reusing a standing Dataproc cluster by name — both family jobs ran on `sf-smoke-cluster` and it was still `RUNNING` afterwards | CURRENT | 2026-09-12 | `smoke-05-cluster-reuse-adbe6bd63644` | `cluster_deps=packed-venv-init-action`, `python=3.11`, `fleet_sizing=derived-overlay-three-way-min`, `horizon_features=computed-at-future-dates`, `run_id_inputs=authored-config-only-v3` |
| 06 | `06_cluster_gpu.json` | Dataproc cluster GPU (T4), incl. zone failover | CURRENT | 2026-09-09 | `smoke-06-cluster-gpu-eea70f834c66` | `cluster_deps=packed-venv-init-action`, `gpu_cluster_image=driver-init-action`, `gpu_device_probe=trainer-root-device`, `dl_gpu_routing=resolved-per-family`, `python=3.11`, `fleet_sizing=derived-overlay-three-way-min`, `horizon_features=computed-at-future-dates`, `run_id_inputs=authored-config-only-v3` |
| 07 | `07_ray_cpu.json` | Ray on Vertex, CPU — two families sharing one cluster, and the row that replaces the un-rederivable `run_id` the note below records | CURRENT | 2026-09-12 | `smoke-07-ray-cpu-ed27e03a2083` | `ray_pool_shape=autoscaling`, `ray_deps=stock-image+uv-runtime-env`, `ray_slot_memory=harvest-only`, `python=3.11`, `fleet_sizing=derived-overlay-three-way-min`, `run_id_inputs=authored-config-only-v3`, `horizon_features=computed-at-future-dates` |
| 08 | `08_ray_gpu.json` | Ray on Vertex, GPU T4 (neuralprophet) | CURRENT | 2026-09-09 | `smoke-08-ray-gpu-497c57c3ad2c` | `ray_pool_shape=autoscaling`, `ray_deps=stock-image+uv-runtime-env`, `ray_slot_memory=harvest-only`, `gpu_device_probe=trainer-root-device`, `dl_gpu_routing=resolved-per-family`, `python=3.11`, `fleet_sizing=derived-overlay-three-way-min`, `run_id_inputs=authored-config-only-v3`, `horizon_features=computed-at-future-dates` |
| 09 | `09_shared_ray.json` | Several families on one shared Ray cluster (CPU + GPU pools) — all three landed on one cluster, the two CPU families in ~6 min each and the GPU one in ~17, and the device audit called the T4 `ENGAGED_IDLE` unprompted | CURRENT | 2026-09-12 | `smoke-09-shared-ray-859750fc97a5` | `ray_pool_shape=autoscaling`, `ray_deps=stock-image+uv-runtime-env`, `ray_slot_memory=harvest-only`, `gpu_device_probe=trainer-root-device`, `dl_gpu_routing=resolved-per-family`, `python=3.11`, `fleet_sizing=derived-overlay-three-way-min`, `run_id_inputs=authored-config-only-v3`, `horizon_features=computed-at-future-dates` |
| 10 | `10_mixed_runtimes.json` | Spark + Ray + BigQuery families concurrently under one run_id | STALE | 2026-09-04 | `smoke-10-mixed-runtimes-a39f0fb4f3fa` | `ray_pool_shape=autoscaling`, `ray_deps=stock-image+uv-runtime-env`, `serverless_deps=container-image`, `native_source_pin=unpinned-all-sources`, `python=3.11`, `fleet_sizing=derived-overlay`, `horizon_features=computed-at-future-dates`, `run_id_inputs=authored-config-only`, `dl_gpu_routing=flat-compute.use_gpu` |
| 11 | `11_ensemble_barrier.json` | Ensembling in barrier mode — the ensemble node waits for every member family, then runs once (23 s against a 23-minute run); re-run under the weighting fix as attempt 2, and its four ensembles now match microbatch's on all 2,800 cells | CURRENT | 2026-09-11 | `smoke-11-ensemble-barrier-834f770ba38b` (attempt 2) | `ensemble_weighting=per-series-calculated+batch-fit-learned`, `backtest_scoring=holdout-reserved+embargo-aware+auto-refit`, `serverless_deps=container-image`, `native_source_pin=unpinned-all-sources`, `python=3.11`, `fleet_sizing=derived-overlay-three-way-min`, `horizon_features=computed-at-future-dates`, `run_id_inputs=authored-config-only-v3` |
| 12 | `12_ensemble_microbatch.json` | Ensembling in microbatch mode — the ensemble node runs alongside the members for the whole 24 minutes, gathering as they land; re-run under the weighting fix as attempt 2, and the barrier pair of this row is the proof that gather mode is now a scheduling choice only | CURRENT | 2026-09-11 | `smoke-12-ensemble-microbatch-9bcd1d294437` (attempt 2) | `ensemble_weighting=per-series-calculated+batch-fit-learned`, `backtest_scoring=holdout-reserved+embargo-aware+auto-refit`, `serverless_deps=container-image`, `native_source_pin=unpinned-all-sources`, `python=3.11`, `fleet_sizing=derived-overlay-three-way-min`, `horizon_features=computed-at-future-dates`, `run_id_inputs=authored-config-only-v3` |
| 13 | `13_native_format.json` | Reading the native BigQuery source table | CURRENT | 2026-09-11 | `smoke-13-native-format-0995c922faab` | `native_source_pin=unpinned-all-sources`, `serverless_deps=container-image`, `python=3.11`, `fleet_sizing=derived-overlay-three-way-min`, `horizon_features=computed-at-future-dates`, `run_id_inputs=authored-config-only-v3` |
| 14 | `14_full_dag.json` | Flagship: all four families + native + ensemble under one run_id (DL on a Serverless L4). Also the row that proves the ensemble write timestamp: 11,200 blended prediction rows, none NULL | CURRENT | 2026-09-11 | `smoke-14-full-dag-2cef0feb95da` | `ensemble_weighting=per-series-calculated+batch-fit-learned`, `backtest_scoring=holdout-reserved+embargo-aware+auto-refit`, `serverless_deps=container-image`, `serverless_gpu_allocator=rapids-pool-released`, `gpu_device_probe=trainer-root-device`, `dl_gpu_routing=resolved-per-family`, `native_source_pin=unpinned-all-sources`, `python=3.11`, `fleet_sizing=derived-overlay-three-way-min`, `horizon_features=computed-at-future-dates`, `run_id_inputs=authored-config-only-v3` |
| 15 | `15_airflow_multi_engine.json` | The whole DAG orchestrated by Composer/Airflow | STALE | 2026-09-03 | `smoke-15-airflow-multi-engine-5ec2924b3374` | `ray_deps=stock-image+uv-runtime-env`, `serverless_deps=container-image`, `native_source_pin=unpinned-all-sources`, `python=3.11`, `fleet_sizing=derived-overlay`, `horizon_features=computed-at-future-dates`, `run_id_inputs=authored-config-only`, `dl_gpu_routing=flat-compute.use_gpu` |
| 16 | `16_cluster_split_hardware.json` | One run needing **two** Dataproc clusters at once — a CPU one and a GPU one. **A 2026-09-12 re-run attempt never reached submit** (`smoke-16-cluster-split-hardware-8a15339afe9b`, header closed `FAILED`); see the narrative below | STALE | 2026-09-02 | `smoke-16-cluster-split-hardware-5e05307425e4` | `cluster_deps=packed-venv-init-action`, `python=3.11`, `fleet_sizing=derived-overlay`, `run_id_inputs=authored-config-only` || 17 | `17_gpu_absent_serverless.json` | **Negative arm:** a Serverless L4 job with the device hidden fails every cell with the contract message naming the service, and the batch stops instead of churning executors | CURRENT | 2026-09-10 | `smoke-17-gpu-absent-serverless-ea3341fa9fd5` | `gpu_fault_injection=probe-mode-default`, `gpu_batch_churn=executor-failure-budget+stall-watchdog`, `job_status=derived-from-cell-tallies`, `serverless_deps=container-image`, `serverless_gpu_allocator=rapids-pool-released`, `gpu_device_probe=trainer-root-device`, `dl_gpu_routing=resolved-per-family`, `python=3.11`, `fleet_sizing=derived-overlay-three-way-min`, `run_id_inputs=authored-config-only-v3`, `horizon_features=computed-at-future-dates` |
| 17 | `17_gpu_absent_serverless.json` | **Negative arm:** a Serverless L4 job with the device hidden fails every cell with the contract message naming the service, and the batch stops instead of churning executors | CURRENT | 2026-09-10 | `smoke-17-gpu-absent-serverless-ea3341fa9fd5` | `gpu_fault_injection=probe-mode-default`, `gpu_batch_churn=executor-failure-budget+stall-watchdog`, `job_status=derived-from-cell-tallies`, `serverless_deps=container-image`, `serverless_gpu_allocator=rapids-pool-released`, `gpu_device_probe=trainer-root-device`, `dl_gpu_routing=resolved-per-family`, `python=3.11`, `fleet_sizing=derived-overlay-three-way-min`, `run_id_inputs=authored-config-only-v3`, `horizon_features=computed-at-future-dates` |
| 18 | `18_gpu_absent_cluster.json` | **Negative arm:** a cluster T4 job with the device hidden fails every cell with the contract message, naming the service — and the run closes `FAILED` on both registry tiers, counting only its own attempt's cells | CURRENT | 2026-09-10 | `smoke-18-gpu-absent-cluster-ef1858b8b83d` | `gpu_fault_injection=probe-mode-default`, `job_status=derived-from-cell-tallies`, `cluster_deps=packed-venv-init-action`, `gpu_cluster_image=driver-init-action`, `gpu_device_probe=trainer-root-device`, `dl_gpu_routing=resolved-per-family`, `python=3.11`, `fleet_sizing=derived-overlay-three-way-min`, `run_id_inputs=authored-config-only-v3`, `horizon_features=computed-at-future-dates` |
| 19 | `19_gpu_absent_ray.json` | **Negative arm:** a Ray T4 job with the device hidden fails every cell with the contract message naming the service, instead of crashing the worker that holds the GPU slot | CURRENT | 2026-09-10 | `smoke-19-gpu-absent-ray-1c033f10707b` | `gpu_fault_injection=probe-mode-default`, `job_status=derived-from-cell-tallies`, `ray_pool_shape=autoscaling`, `ray_deps=stock-image+uv-runtime-env`, `ray_slot_memory=harvest-only`, `gpu_device_probe=trainer-root-device`, `dl_gpu_routing=resolved-per-family`, `python=3.11`, `fleet_sizing=derived-overlay-three-way-min`, `run_id_inputs=authored-config-only-v3`, `horizon_features=computed-at-future-dates` |
| 20 | `20_gpu_intent_cpu_family.json` | **Disagreement arm:** `use_gpu: true` with the deep-learning family overridden to `cpu` completes on CPU and buys no accelerator — the Ray pool comes up with no `acceleratorType` at all | CURRENT | 2026-09-10 | `smoke-20-gpu-intent-cpu-family-239ba1e33242` | `dl_gpu_routing=resolved-per-family`, `ray_deps=stock-image+uv-runtime-env`, `ray_pool_shape=autoscaling`, `gpu_device_probe=trainer-root-device`, `python=3.11`, `fleet_sizing=derived-overlay-three-way-min`, `run_id_inputs=authored-config-only-v3`, `horizon_features=computed-at-future-dates` |

### Why three configs exist that are designed to fail

Smokes 17–19 are six-cell runs of the same shape as 03, 06 and 08, and they are the reason those
three prove anything. A GPU rung that comes back green tells you the check did not object; it does
not tell you the check *can* object. Run each of these with `SF_HIDE_DEVICES` armed and the
accelerator is provisioned and then taken away from the workers, so a run that still reaches
`COMPLETED` is reporting a contract that isn't enforced. The switch has two depths:
`SF_HIDE_DEVICES=probe` hides the device from our own probe and leaves CUDA alone, which is the mode
that reaches the contract check on every service; `SF_HIDE_DEVICES=cuda` (and the older `=1`) empties
`CUDA_VISIBLE_DEVICES`, which takes the card from every library on the box. The wave described below
predates `probe` and was run at the blunter depth.

Six cells because the expected outcome is an immediate refusal — there is no reason to buy a
hundred series' worth of fleet to watch a job stop. **All three were run on 2026-09-10 and all three
did withhold the forecast, but only one of them refused immediately or said why**; the section below
on that wave has what each service actually did.

### The 2026-09-09 GPU wave: four defects found, fixed, and re-proven the same day

The first wave of the live campaign ran the four positive rungs — smoke 01 on Serverless CPU, then
the three GPU services: smoke 03 on a Serverless L4, smoke 06 on a Dataproc cluster T4, and smoke 08
on Ray T4. Smoke 01 passed. The other three came back looking green and were not, for three
independent reasons — and fixing the second of those exposed a fourth defect, which blocked submit
outright until it was fixed too.

**All four were fixed and all three rungs were then re-run, and that second pass is what the rows
above record.** Each of 03, 06 and 08 completed 100 of 100 cells with `device_used='cuda'` on every
one, and each stamped `device_use.verdict = ENGAGED_IDLE` with `cells_on_device = 100` — which is
what the campaign plan predicted for a positive arm. Nothing here should be read as "the platform
was broken": every one of these runs really did have its accelerator. What was broken was our
ability to say so, and in one case our ability to keep the fits alive alongside a library that
wanted the same card.

**1. The device probe was reading the weights after the library had already moved them.** All three
services reported `device_used='cpu'` on every single GPU cell, on the same day, for cells that had
50–68 KB genuinely allocated on the card. `device_audit` did exactly what it is supposed to do with
that input and stamped `MISSING_DEVICE` — the verdict that means the accelerator was billed and
never touched. The runs were fine; the probe was wrong. `NeuralProphetModel.device_used` read a
parameter tensor from the fitted module, and PyTorch Lightning ends every run by moving the module
back to the CPU (`Strategy.teardown` calls `self.lightning_module.cpu()`), so the tensor a caller
sees afterwards is on the CPU no matter where the arithmetic happened. The fix reads
`trainer.strategy.root_device` instead, which survives teardown and is what Lightning's accelerator
connector *resolved* the request to against the hardware it actually found — not the request
restated. The parameter read stays as a fallback for the cases where nothing moved the weights. This
is the new `gpu_device_probe` axis, and it is the reason **no GPU-contract positive result recorded
before 2026-09-09 was ever earnable**: the measurement could not return the answer that would have
passed.

The natural implementation — a Lightning callback that reads the device mid-fit, while the weights
are still on it — is not available to us. Handing NeuralProphet 0.9.0 any `callbacks` key sends its
`configure_trainer` down a branch that dereferences `pl.callbacks.ProgressBarBase`, a class removed
from the Lightning version we pin, and the fit dies with an `AttributeError` before training starts.

**2. Serverless GPU lost 37 of 100 cells to the RAPIDS memory pool, and the harness called it a
PASS.** Every failed cell raised `CUDA error: out of memory` while asking for a model that needs
64 KB. The cause is in the driver log rather than in our code:
`Initializing RMM ASYNC pool size = 21632.125 MB on gpuId 0`. Serverless GPU runtimes ship the
RAPIDS accelerator switched on, and its default allocator reserves nearly the whole L4 for Spark SQL
before a single fit runs. The fits do not live in that JVM — they run in the PySpark Python workers,
which are left to share a few hundred megabytes, and the ones that lose the race die. The contrast
case is decisive: the same config on a cluster T4 and on Ray T4, neither of which loads RAPIDS, lost
nothing at all. The fix sets `spark.rapids.memory.gpu.pool=NONE` on GPU batches, which drops the
reservation and leaves RAPIDS allocating on demand, so SQL still runs on the card and the fits can
reach it too. This is the new `serverless_gpu_allocator` axis. On the re-run, smoke 03 completed all
100 cells with no OOM at all.

That fix had a sequel worth knowing about, because it is not discoverable by reading anything Google
publishes. **Naming any `spark.rapids.*` property makes Serverless stop deriving GPU executor
memory.** Left alone the service resolves a 4-core L4 executor to `spark.executor.memory=9560m` and
derives the overhead from it; supply one RAPIDS property and it falls back to
`spark.executor.memory=3346m` with `spark.executor.memoryOverhead=0m`, then rejects its own default
at submit, because 0 is below the 256m-per-core floor it validates against. So a batch that releases
the pool has to restate the memory the platform would have chosen. That number is derivable rather
than copied: the per-config maximum of 3346m per core bounds memory *and* overhead summed, and
overhead is service-owned at 40% of memory, so the memory we may name is `cores × 3346 / 1.4` —
9560m at four cores, exactly what the service picked for itself. The same arithmetic was already
wrong one layer up, in the sizing overlay, which clamped the GPU inversion to the maximum itself and
so guaranteed an illegal batch once the service added its 40%. It had never fired because it needs a
measured deep-learning memory footprint and the profile source had none. Both are fixed, and the
sweep invariant now reads "memory plus the overhead derived from it fits" rather than "memory alone
fits".

**3. The harness had no check that could see a partial run.** Losing a third of the cells satisfied
every verifier it had. The run reached `COMPLETED`, because a failed cell is recorded rather than
fatal. The leaderboard listed the model, because the leaderboard counts metadata rows and a failure
writes one. `verify_predictions` was satisfied, because it asks only that the count be non-zero, and
63 is non-zero. A new `verify_cells` closes this: a smoke is a hundred well-formed series with
nothing adversarial in them, so the bar is *zero* failed cells rather than a tolerance. Under this
check, smoke 03 would have printed FAIL. On the re-run it printed PASS with `n_cells=100`, and that
is the first time the harness has been able to tell those two outcomes apart.

Two smaller facts from the same wave, both recorded here because they will otherwise be rediscovered:

- **The pre-baked GPU cluster image is gone from this deploy.** Smoke 06 first failed with
  *"Selected software image version … can no longer be used to create new clusters"* against a
  hand-set `SF_GPU_IMAGE`. This deploy has Terraform's `build_gpu_image` switched off, so no image
  was ever built and the environment variable was a stale value carried forward. Re-run on the
  fallback — a stock image plus the driver init action — the smoke got its T4 and completed all 100
  cells. **The owner's call, taken the same day, is that the fallback is now the supported path**,
  so `gpu_cluster_image` moved back to `driver-init-action` and `254fe4f` is reverted as a default.
  The reasoning is that the pre-baked image is what broke: a custom image ages out, and when it does
  the failure lands on whoever next asks for a GPU cluster, in a message about image versions rather
  than about anything they did. Every fresh deploy already gets the fallback, because
  `build_gpu_image` defaults off, so this makes the supported path and the default path the same
  one. Terraform can still build the image for an operator who creates GPU clusters often enough to
  care about the few minutes the boot-time driver install costs; it is an opt-in optimisation with
  no ledger row behind it, not the shape we test. Ray reached the same conclusion earlier and for
  the same reason — see the smoke-07 head-crash note.
- **`devices=1` is a request the library discards.** NeuralProphet's `configure_trainer`
  unconditionally overwrites it with `-1` — all visible devices — on every accelerator it resolves
  to `"gpu"`. Every shape we provision puts one card in front of a worker, where `-1` and `1` mean
  the same thing, so nothing is broken; but the pin is documentation of intent, not control.

**One thing the wave did prove.** Smoke 08 wrote a non-NULL `peak_gpu_bytes` on all 100 Ray cells
(50176–77824 bytes, `Tesla T4`), where that column had previously been NULL on 100% of Ray rows.
The GPU memory measurement works on Ray. That is independent of the device probe — it is read by the
worker from `torch.cuda` while the fit is live — and it is the evidence that sits beside
`device_used` in the same row and made it possible to tell that the probe, not the platform, was
wrong.

### The 2026-09-10 negative arms: two passes, and the second one is what the rows record

Smokes 17, 18 and 19 are the other half of the GPU wave — the same three services with the
accelerator provisioned and then taken away from the workers. Without these arms the three green
rungs above are indistinguishable from a check that always says yes.

**They were run twice on the same day, and the two passes are not interchangeable.** The first used
`SF_HIDE_DEVICES=1`, which empties `CUDA_VISIBLE_DEVICES`; it established the property below and
found two defects, one of them a genuine production risk. Both were fixed, the second pass used
`SF_HIDE_DEVICES=probe`, and it is the second pass the ledger rows above record. The first pass is
written out here because what it found is the reason the second one could say anything.

**The property they were run to establish does hold.** None of the three produced a forecast. Every
one of them wrote zero rows to `forecast_predictions`, and the smoke harness printed `RESULT: FAIL`
on all three. A GPU job that loses its device does not quietly finish on the CPU and hand back
numbers, which is the thing that would have been genuinely dangerous.

**Two things around that property do not hold**, and both are worth more than the arms cost.

**1. A run that produced nothing closes `COMPLETED`.** Smoke 18 wrote six `forecast_metadata` rows,
all of them `cell_status='error'`, and no predictions at all; smoke 19 wrote nothing whatsoever and
recorded four crashed chunks in its telemetry. Both of them closed with `run_jobs.status =
COMPLETED` and `run_registry.status = COMPLETED`. The rule that should have applied is in
`aggregate_status`, which has always said that no successful cells means `FAILED` and a mix means
`PARTIAL` — and the Ray driver's own log even says `run will close PARTIAL`. The reason the rule
never fires is structural rather than a slipped condition: under the per-family DAG,
`launch_family_job` wraps the submitter in `run_job` and the submitter returns a *probe handle*, not
an outcome. Nothing carries the remote engine's cell tallies back to the driver that owns the row,
so the row goes terminal on the only fact the driver has — the launch call returned without raising
— and the header rolls up job rows that all say the same thing. **The practical consequence is that
the registry will report a green run that forecast nothing**, which is exactly the reading an
unattended pipeline acts on. The three positive rungs are unaffected: they wrote 100 good cells
each, so `COMPLETED` was the right answer there for the wrong reason.

**This one is fixed, and the fix does not fight the structure.** Nothing tries to carry an outcome
back through a submitter that has none to give. Instead the driver asks the same way the device
audit already asks its own question: once the job finishes, `job_outcome.audit_cells` reads that
attempt's tallies out of `forecast_metadata` — one aggregate, the same for all three services,
because all three write the same rows — and the row's status comes from those. Zero cells is
`FAILED`, not a quiet `COMPLETED`. The header then rolls those statuses up through the *same*
function the Airflow `finalize_run` task calls; there had been two copies of that roll-up and the
local one folded exceptions rather than statuses, which is why neither tier caught the other's
blind spot. Since a run can now finish badly without anything raising, `main.run` also raises at the
end so the process still exits non-zero.

**And it is proven live.** Attempt 2 of `smoke-18-gpu-absent-cluster-ef1858b8b83d`, 2026-09-10,
1196 s on the same cluster T4 with the device hidden again — long enough that it is a real
provision-and-run, not a short-circuit. The same six cells errored the same way, and this time the
`run_jobs` row reads `status=FAILED` with `job_telemetry.cells = {"cells": 6, "errors": 6, "status":
"FAILED"}`, the header reads `FAILED`, and the process exited non-zero on the manufactured
`EngineError`. The paragraph above describes what attempt 1 did; the table row now records attempt
2, which is the behaviour the code ships.

**Six, not twelve** — which is the second thing this re-run had to show. `forecast_metadata` is
append-only and carries no attempt column, so an unbounded read would have counted attempt 1's six
error rows as well and reported `cells: 12`. Both audits now take a launch-time bound, and both
reported six. That closes the cross-attempt counting defect the 2026-09-09 re-runs exposed, where
smokes 03, 06 and 08 each reported `cells: 200` against `cells_on_device: 100` and so read as half
the fleet missing its device when in truth the first attempt's hundred CPU cells were still there.

**2. Only one of the three services let the contract speak.** `_require_device` raises a
`ConfigError` naming the family, the engine and what the worker saw, and on the Dataproc cluster that
is precisely what landed — six rows of `error_class='CONFIG_REPAIRABLE'` carrying *"family
'deep_learning' is set to hardware='gpu' and this spark job was provisioned onto GPU hardware, but
torch is installed here and reports no CUDA device"*. That is the negative arm working as designed.
The other two services never reached the check, for two different reasons:

- **On Ray, the worker process dies first.** The task holds `GPU: 0.5`, and a Ray worker that holds a
  GPU slot with no visible device does not raise — it crashes. The driver saw four
  `WorkerCrashedError()` chunks and nothing else, so the only diagnosis in the registry is the crash
  class. Ray retried each one, because `WorkerCrashedError` is on the deliberate retry list as an
  infrastructure fault, which is the right default and the wrong outcome here.
- **On Serverless, the RAPIDS plugin dies first, and it does not fail fast at all.** The executor
  plugin hit `cudaErrorNoDevice` and shut down (`ai.rapids.cudf.CudaFatalException … 100
  cudaErrorNoDevice`), Spark replaced the executor, and the replacement did the same thing. The batch
  was still churning executors 49 minutes in — against 28 minutes for the equivalent *successful*
  run — with no forecast and no end in sight, and it was cancelled rather than left to the four-hour
  TTL. The `FAILED` on its ledger row is that cancellation. Operationally this is worse than a fast
  refusal: an unattended GPU batch that loses its device burns fleet until something stops it.

Both of those are properties of the fault injection meeting the platform, not of the contract itself.
Emptying `CUDA_VISIBLE_DEVICES` is a blunter instrument on a runtime that loads a CUDA library of its
own than it is on one that does not: Ray's worker and the RAPIDS plugin both react to the missing
card before any of our code runs, so on those two services the injection never delivered its fault
to the check it was aimed at. What the *first pass* established, precisely, is: **the contract is
enforced and named on Dataproc clusters, and on the other two services the forecast is withheld but
the reason is not recorded.** That was a smaller claim than the campaign plan predicted, and it was
the one that afternoon's evidence supported. The next section is the pass that closed the gap.

#### The second pass, the same afternoon: all three services now say why

Both findings were fixed and all three arms re-run under `SF_HIDE_DEVICES=probe`. The injection's
new mode makes our own device probe report `cpu` while leaving CUDA untouched, so the fault arrives
at `_require_device` instead of at the library underneath it. Separately, and independently of
testing, a GPU batch no longer replaces executors without limit: `spark.executor.maxNumFailures` is
bounded and a driver-side watchdog cancels a batch that has written nothing (see `gpu_batch_churn`).

```
17  Serverless L4   FAILED  1068 s   6 cells   CONFIG_REPAIRABLE ×6   "this spark job"
18  Cluster T4      FAILED  1070 s   6 cells   CONFIG_REPAIRABLE ×6   "this spark job"
19  Ray T4          FAILED  1067 s   6 cells   CONFIG_REPAIRABLE ×6   "this ray job"
```

**All three now reach the contract and record what it said**, in the same words the cluster had been
producing alone: *"family 'deep_learning' is set to hardware='gpu' and this &lt;service&gt; job was
provisioned onto GPU hardware, but torch is installed here and reports no CUDA device, so the
accelerator did not attach to this worker."* Each stamped `MISSING_DEVICE` with
`cells = cells_no_device = 6` and `cells_on_device = 0`, and each closed `FAILED` on both registry
tiers. The Ray worker no longer crashes — holding a GPU slot is fine when the device is really
there; it was the empty `CUDA_VISIBLE_DEVICES` that killed it.

**The churn is gone and the Serverless properties are accepted.** `gcloud dataproc batches describe`
read back `spark.executor.maxNumFailures: 8` and `spark.executor.failuresValidityInterval: 30m` on
the live batch, so Dataproc Serverless takes both — a rejected property would have failed the batch
at submit. Smoke 17 ended in **17.8 minutes against 49 minutes of executor-churning on the first
pass**, and against 28 minutes for the equivalent *successful* GPU run. The stall watchdog was not
what stopped it and did not need to be; it stands as the backstop for the case where a batch neither
progresses nor dies.

The three durations landing within three seconds of each other is coincidence, not an artifact —
they ran on independent clocks (17 from 14:30:18, 18 and 19 from 14:40) and each is roughly fifteen
minutes of provisioning followed by an immediate refusal. Provisioning dominates because that is the
point: the accelerator is bought before it is taken away.

#### Smoke 20, the mirror image: an accelerator that is asked for and correctly not bought

`smoke-20-gpu-intent-cpu-family-239ba1e33242`, 2026-09-10. Where 17–19 ask what a job does when it
loses a device it was promised, 20 asks the opposite: the config says `use_gpu: true` and
`gpu_type: "T4"` at the top level, and then overrides `compute.families.deep_learning.hardware` to
`"cpu"`. The family override must win, and the run must cost nothing in accelerators.

**The strongest evidence is not that the run completed — it is the shape of the pool it ran on.**
The Vertex persistent resource came up as `n1-standard-16` and `n1-standard-8` with **no
`acceleratorType` field on either pool**, read straight off the live v1beta1 resource while the run
was in flight. A `use_gpu: true` that leaked past the family override would have shown a T4 there,
and no amount of reading the registry afterwards would distinguish "ran on CPU" from "bought a card
and ignored it". The run then completed all 100 cells `cell_status='ok'` with `device_used='cpu'` on
every one, `hardware='cpu'` and a NULL `gpu_type` on the `run_jobs` row, and a NULL `device_verdict`
because there is no GPU contract to audit when none was requested for that family. 1688 s, pool torn
down and the resource list read back empty.

### Airflow orchestrated the whole DAG, and the two bugs it found are both invisible from a checkout

`smoke-15-airflow-multi-engine-5ec2924b3374`, 2026-09-03: Composer 3 / Airflow 2.10.5 ran the
emitted `dag_<run_id>.py` end to end — 200 series, five models, three backtest folds, a microbatch
ensemble — and reached `COMPLETED` in 80 minutes.

```
statistical    spark/cpu       Dataproc Serverless batch    1607 s
ml             spark/cpu       Dataproc Serverless batch    1653 s
deep_learning  ray/gpu/T4      Vertex Ray submission        3137 s
native         bigquery/cpu    BigQuery job                  340 s
ensemble       bigquery/cpu    BigQuery job (microbatch)    4801 s
```

The assertion that makes this smoke worth its cost is not the leaderboard — it is that the `run_id`
Airflow produced is **the same string** the local planner resolved before anything was submitted.
Same config, same code, two orchestrators, one identity. That is the local↔Composer constraint
stated as a test rather than as an intention. The ensemble node, running microbatch, spanned the
whole 80 minutes and gathered every family as it landed; all four strategies scored over the full
200 series, and `ensemble_nnls` (0.3590) came second only to `timesfm` (0.3474).

**It found two deployment bugs on its first live run, and neither can be reproduced from a repo
checkout.** Both come from the same root fact: a Composer environment is a *src-only delivery*, not
a repository, and Airflow puts inner plugin directories on `sys.path`.

1. `code_delivery` resolved the locked cluster requirements as `<repo root>/docker/requirements.txt`.
   On a checkout that path always exists. On the plugins prefix there is no repo root above `src/`,
   so building a Ray `runtime_env` died on `FileNotFoundError: /home/airflow/gcs/docker/requirements.txt`.
   Fixed by searching both launch-point shapes and having `make composer-sync` deliver the file
   beside `src/` — one source of truth, not a second copy inside the package.
2. `profiling/numbers.py` **shadowed the stdlib `numbers` module**. Inside a package that is
   harmless, because only the package root is ever on `sys.path`; under Airflow it is not, and an
   unrelated third-party library's `import numbers` resolved to our file. Renamed to `numeric.py`,
   with a static check in `tests/unit/test_source_conventions.py` that fails any module named after
   a stdlib module.

The second one is worth dwelling on because of **how it presented**: the failure the operator saw
was `Ray cluster … could not be created in any of ['us-central1']`, which reads as a capacity or
quota problem in the region. The `ImportError` was only in the tail of the wrapped message. A
region-failover error string that can be produced by a typo in our own import graph is a diagnostic
trap, and it cost the first three attempts of this run.

The repair was made **in place**: after re-syncing the fixed code, only the `deep_learning` task was
cleared (`only_failed`, no downstream), so the three families that had already succeeded were not
re-billed. Duplicate `run_jobs` rows across those retries are expected — `v_run_jobs` dedupes on
read by `(run_id, family)`.

### Smoke 14 is the flagship, and it is green again

`smoke-14-full-dag-c8664f7a2d23`, 2026-09-02: five families under one `run_id`, across three
different execution surfaces, with an ensemble node gathering them.

```
deep_learning  spark/gpu/L4    Dataproc Serverless batch
statistical    spark/cpu       Dataproc Serverless batch
ml             spark/cpu       Dataproc Serverless batch
native         bigquery/cpu    BigQuery job
ensemble       bigquery/cpu    BigQuery job
```

Every one of the ten leaderboard entries carries a real `wape` over 100 cells — the six base models
and the four ensemble strategies — so the DAG did not merely finish, it scored. `timesfm` (0.3554)
leads, followed by `ensemble_nnls` (0.3652); the ensembles land above every model they are built
from except the one BigQuery foundation model, which is the sort of ordering that suggests the
ensemble is doing arithmetic rather than copying. The rerun check confirmed the same `run_id` and an
unchanged board.

This run predates the metric-encoding fix recorded below, but is unaffected by it: backtesting is on
here, so nothing was unscored and there were no NaNs to mis-sort.

### Two Dataproc clusters under one run_id, and the custom GPU image had quietly expired

`smoke-16-cluster-split-hardware-5e05307425e4`, 2026-09-02, PASS. This is the first live run of the
hardware-split branch: a Dataproc cluster has one worker machine type, so a run whose ephemeral
cluster families disagree about hardware needs two clusters, created and torn down independently
under one `run_id`.

```
deep_learning  spark/gpu/T4    Dataproc cluster job   1040s
statistical    spark/cpu       Dataproc cluster job    163s
```

Both clusters existed at once (`…-5e05-cpu` RUNNING while `…-5e05-gpu` was still CREATING), each got
its own name and its own sizing, both reached `COMPLETED`, and both are gone — verified by
`describe` returning `NOT_FOUND`, not by any SDK success line. All three models produced 100 cells.
Wall clock 39m18s, of which roughly two thirds is cluster provisioning.

**The first attempt failed, and the failure is the more useful result.** The CPU cluster created
normally; the GPU cluster was refused outright:

```
google.api_core.exceptions.InvalidArgument: 400 Selected software image version
'2.2.86-debian12' can no longer be used to create new clusters.
```

That version string appears in no config, no Terraform variable and no part of the run. It is baked
into the custom GPU image the deployment built nine days earlier, on 2026-08-24, and recorded in its
`goog-dataproc-version=2-2-86-debian12` label. Its CPU sibling was created from the `2.2-debian12`
*alias*, which resolves forward on every create and was therefore unaffected.

So a prebaked image has an invisible expiry. It is built from whatever sub-minor is current that
day; Google retires sub-minors on its own schedule; the image cannot move, because moving is the one
thing baking it prevents. Nothing in the deployment ages it, warns about it, or rebuilds it. The
window here was **nine days**.

The rerun that passed had `SF_GPU_IMAGE` unset, taking the documented fallback — stock image plus
the GPU-driver init action, which compiles the driver at create time. That path cost about 14
minutes of cluster-create for the GPU node and worked without any change. **This row therefore does
not declare the `gpu_cluster_image` axis: it did not run the current value of it.** It is a proof of
the two-cluster branch, not a re-proof of smoke 06's prebaked-image path.

Two things follow, one shipped and one still open. Shipped: `_explain_create_failure` in
`dataproc_cluster` now rewrites this error when a custom image was in play, naming the image, the
`SF_GPU_IMAGE` knob that disables it and the fallback that still works, and keeping the original
text — and `compute_fallback.is_retired_image_error` deliberately classifies it as *not* capacity,
because hopping zones would spend the entire failover walk on an image no zone has. Still open: what
to do about the image itself, which is a cost decision rather than a bug — see the gap below.

### The cluster path re-proved itself on 2026-09-01, and the reuse smoke checks the thing that matters

Smokes 04 and 05 both passed under the derived fleet-sizing overlay. What makes 05 worth running
separately from 04 is not that it forecasts — 04 already did — but the **lifecycle asymmetry**: an
ephemeral cluster must be deleted when its run ends, a named one must not. Both halves held. After
04, `sf-cluster-smoke-04-…` was gone; after 05 submitted two family jobs to `sf-smoke-cluster` and
reached `COMPLETED`, that cluster was still `RUNNING`. A reuse path that tore down a cluster it did
not create would pass every leaderboard assertion in the harness and still be badly wrong, so the
survival check is the assertion, and it is made against `clusters list`, outside the harness.

Smoke 06 followed on 2026-09-02 — `neuralprophet` on a cluster T4 in `us-central1-b`, on the
pre-baked driver image. **One precision about its row: "incl. zone failover" describes the config,
not what happened.** The first candidate (deployment region, auto-zone placement) succeeded, so
`_create_cluster_across_candidates` never walked to a second zone. The failover path therefore
remains proven only offline; what smoke 06 proves is the GPU cluster itself — the driver image, the
accelerator attachment, and a DL model actually fitting on the device.

**`compute.machine_family` is wired end to end, and the proof is a side-by-side.** No smoke config
sets it, so it was run once from a throwaway config —
`wave-54-cluster-machine-family-ab6c818831e8`, 2026-09-02, `COMPLETED`, three models on two cluster
families — with `"machine_family": "n2"` as the only difference from `04_cluster_cpu.json`. Its
cluster came up `n2-standard-8` workers on an `n2-standard-4` master while `sf-smoke-cluster`, built
from the same code path at `auto`, sat in the same `clusters list` at `n1-standard-8`/`n1-standard-4`.
The field reaches GCE. There is **no row** for this in any table above and that is deliberate: the
config is not in the repo (a file under `configs/smokes/` would trip the ledger tripwire, which
requires exactly one row per config), so it lives here as prose with its `run_id`.

Note what this run does *not* prove. `machine_family` is documented as ignored on GPU — an
accelerator dictates its own machine — and this was a CPU run, so that branch is still offline-only.

The standing cluster is a **campaign fixture, not infrastructure**: created here for smoke 05 and
deleted at teardown. It is built through `dataproc_cluster.build_cluster` with the same
`_resolve_cluster_deps` / `_stage_cluster_init` pair `provision_shared_cluster` uses, so it is shaped
like an ephemeral cluster in every respect but its name. Standing it up with a hand-written
`gcloud dataproc clusters create` would have made smoke 05 a test of *that command's* fidelity to
the product rather than of the reuse path.

### Barrier and microbatch ensembling are not interchangeable, and running them back to back showed it

Smokes 11 and 12 differ only in `compute.ensemble.mode`, so running them on the same data on
2026-09-02 is a controlled comparison the suite had never actually made. The three **calculated**
strategies agreed to float noise — identical to 15 significant figures for `inverse_error` and
`median`, last-digit for `mean`. `ensemble_nnls` did not: **0.36516 under barrier, 0.36398 under
microbatch**, a fourth-decimal difference against neighbours agreeing in the fifteenth.

The cause is in the code and is not a bug so much as an unstated property. `_ensemble_batch` is the
shared core behind both triggers, and it calls `fit_learned` on *whatever OOF it was handed*. Barrier
hands it every series once. Microbatch hands it one ready-batch at a time, so the learned strategies
are re-fit per batch on a subset. The calculated strategies are per-series and so are unaffected by
the partitioning; a learned strategy trains across series, and a different training sample gives
different weights. `_ensemble_batch`'s docstring claimed "identical logic and rows" for both
triggers; it now says exactly where that stops being true.

**The consequence worth carrying: `ensemble_nnls` under microbatch is not reproducible the way the
rest of a run is.** How series batch depends on when base jobs finish, so a re-run with different
timing can fit different weights. Whether learned strategies should defer to a final global fit is a
design question and is left open — the campaign's job here was to notice it, and it took a live
side-by-side to do that, because the offline tests exercise each trigger against fixtures rather
than the two against each other.

### Smoke 13 passed, and its leaderboard was wrong — two encodings for "no metric"

Smoke 13 reads a **native BigQuery table** rather than Iceberg, and on 2026-09-02 it did so
correctly: run `smoke-13-native-format-8e67fd137515`, all verifiers green. What was wrong was the
report it printed. `arima_plus` and `timesfm` headed the leaderboard, above two models that had
actually been scored — because the run has backtesting off, so nothing was scored at all, and the
two native models' `wape` was **NaN** while `theta`'s and `xgboost`'s was **NULL**. BigQuery sorts
NaN ahead of every real number, so the unscored models won a ranking they had not entered.

Both encodings came from the same in-memory value. `registry/rows.py::_as_float` is the coercion
that turns a non-finite metric into `None`, and its docstring calls itself the one boundary every
engine's rows flow through — but `bigquery_engine._meta_row` and `ensemble_run._ensemble_meta_row`
each built their metric columns by passing the panel through raw. Both now route through
`_as_float`, and `tests/unit/test_metric_null_encoding.py` covers all three writers together so a
fourth one fails there rather than re-splitting the encoding quietly.

**This is not an architecture axis, and the row above still stands.** No forecast value changed and
no mechanism was replaced; what changed is how the absence of a score is spelled in one column. A
reader auditing rows dated before this fix should expect NaN rather than NULL in the native and
ensemble metric columns of those runs, and should not trust an `ORDER BY <metric>` taken across
engines on them. Only a live run finds this: every offline test asserts against one writer at a
time, and each writer is self-consistent.

### `features.level_shift` changes forecast values, and 8 of 100 series prove the detector isn't firing blindly

`features.level_shift` defaults to `False` and no smoke config turns it on, so the whole feature was
`NEVER_RUN` on live infrastructure. On 2026-09-02 two runs settled it: identical configs — `xgboost`
alone, 100 series, horizon 28, `holidays: ["US"]`, `transform: "log1p"` — differing in that one
boolean. `wave-68-level-shift-off-5f5e05d8ac1b` and `wave-68-level-shift-on-800462340da5`, both
COMPLETED, both a single Serverless `ml` job.

Joined on `(ts_id, forecast_date)`, **2576 of 2800 forecast values differ**, mean absolute difference
4.76 and mean relative difference 23%. This is a forecast-value feature, not plumbing, and it now has
live evidence of that.

The 224 identical values are the more interesting half. They are **8 whole series, unchanged across
all 28 horizon dates** — the other 92 changed at every date. That is exactly what
`level_shift_step`'s contract predicts: it returns all zeros when no split clears
`_LEVEL_SHIFT_SIGMA`, so those series get a constant column the tree cannot split on and the two runs
must agree to the bit. A detector that fired on everything, or a column silently dropped, would both
have shown up as 100/100 or 0/100. Split 92/8, with per-series all-or-nothing, is the signature of
the detector actually discriminating.

Recorded as prose rather than a table row: the two configs are throwaways outside
`configs/smokes/`, so the ledger tripwire has nothing to bind them to. The claim is the comparison,
not either run.

### Smoke 08 passes, and the "Vertex GPU entitlement" that blocked it for two days was half wrong

`smoke-08-ray-gpu-c41ecf2d5d52`, 2026-09-02: `neuralprophet` on a Vertex Ray cluster with **seven
T4s** in `us-central1`, 100 cells, rerun idempotent, cluster torn down and confirmed gone by
`describe`. Provisioning took 12 min 35 s (create 18:09:16 → `RUNNING` 18:21:51), which is the number
to budget for a Ray *GPU* cluster the way ~10 min is the number for a CPU one.

**This row was `NEEDS_RECHECK` and the section under it said the project needed a quota grant. That
was wrong, and the way it was wrong is the same mistake this campaign made twice.** The evidence for
"entitlement blocker" was six failed provisions on 2026-09-01, of which **exactly one named a
cause**. `us-east1` and `us-west1` said *"The following quotas are exceeded:
`CustomModelTrainingT4GPUsPerProjectPerRegion`"*; `us-central1` gave the contentless *"An internal
error occurred on your cluster"* and was read as "the same ceiling wearing a mask." It was not a
mask. It was the [provisioning outage](#the-ray-outage-resolved-itself-and-the-fix-we-nearly-shipped-for-it)
that took every Ray cluster in the region down for two days, GPU and CPU alike, and that this page
already records as having clouded the `us-central1` leg of wave 6.1 on the same day.

**The meter was readable the whole time, and reading it settles both halves at once.** The
per-region Vertex training limits for this project:

| Metric | us-central1 | us-east1 | us-west1 |
|---|---|---|---|
| `custom_model_training_nvidia_t4_gpus` | **12** | 2 | 2 |
| `custom_model_training_nvidia_l4_gpus` | 28 | 28 | 28 |

Smoke 08 asks for seven T4s. Seven does not fit under 2, so `us-east1` and `us-west1` were genuinely
quota-blocked and said so accurately. Seven fits comfortably under 12, so `us-central1` never was.
The regional split explains every observation without needing an entitlement story: **the platform
named the quota failure where there was one, and the region where it stayed silent is the region
that had the headroom.** Wave 11 independently established that this platform is explicit about
quota when quota is the problem; that generalisation should have been applied here and was not.

**The mistake, stated as a rule.** A diagnosis was inferred for `us-central1` from its *neighbours'*
error messages rather than from anything `us-central1` itself reported, at a moment when a
region-wide fault was independently active. That is the confounded-control failure again, in a third
costume: a contentless error is not weak evidence for the nearest available explanation, it is
**absence of evidence**, and the neighbouring regions differed on the very dimension being inferred
across. The check that would have caught it costs one API call — read the meter for the region in
question instead of borrowing a conclusion from a region with a different limit.

**What survives unchanged.** Two things from the original diagnosis are still true and still worth
having:

- **Compute Engine quota does not tell you whether a Ray GPU run can start.** `NVIDIA_T4_GPUS` read
  4-of-4 free in `us-central1` throughout. A Vertex Ray cluster does not spend that meter; it spends
  `custom_model_training_nvidia_t4_gpus`. Checking the former before a run is worse than not
  checking, because it answers confidently and about the wrong thing. What is new is the *right*
  meter's name and the fact that it is per-region — `gcloud alpha services quota list
  --service=aiplatform.googleapis.com` returns it, bucketed by region.
- **Deep learning does not have to run on Ray**, and that remains the answer for a deployment that
  genuinely lacks the quota. `14_full_dag.json` puts the DL family on Serverless L4; pointing
  `compute.families.deep_learning` at `runtime: spark` is a config edit, not a code change. All four
  GPU paths in the product are now live-proven: Serverless L4 (smoke 03), cluster T4 (smoke 06),
  BigQuery-native, and — as of this row — Vertex Ray T4.

**The multi-region part of the original claim stands, for a different reason.** Smoke 08's config
lists three `ray_regions`, but under PSC-I only the deployed region has a network attachment, so the
other two are unreachable regardless of their GPU quota — see the region-failover section below.
`us-east1`/`us-west1` T4 quota of 2 is therefore not worth requesting an increase for; the network
attachment is the binding constraint there, not the accelerator.

Two defects in the region fallback surfaced only because a region actually ran out, and both were
fixed and confirmed live the same day. They are the same defect twice: **an error the classifier
cannot parse was treated as an error it had diagnosed.**

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

One workaround that looks obvious and was not — move the DL family to `hardware: cpu` — failed in
the worst way available: a silent indefinite hang. Fixed and proven live 2026-09-02; see the
zero-worker section two below.

### The Ray region failover cannot leave the deployed region, and a CPU run is what proved it

Wave 6.1's stand-in — a Ray CPU run with **no GPU anywhere in the config** — walked all three
`compute.ray_regions` and failed in each. `us-central1` gave the opaque internal error; `us-east1`
and `us-west1` both gave something unambiguous:

```
The resource 'projects/307701787156/regions/us-east1/networkAttachments/scale-forecasting-ray'
was not found
```

That is structural, not transient. `terraform/main/modules/network/main.tf` creates exactly one
`google_compute_network_attachment`, in `var.region`, and exports it as a fully-qualified
**regional** resource ID. `ray_cluster.py` passes that ID to `PscIConfig` verbatim — it never
rewrites the region for the candidate it is currently trying. So on a PSC-I deployment,
`compute.ray_regions` can only ever succeed in the one region the deployment was built in; the other
entries are guaranteed failures that cost a provisioning attempt each before the walk gives up.

**The value of proving it on CPU is that it separates two blockers that had been reading as one.** A
GPU run that fails in three regions looks like a GPU story. Take the GPU out and the same walk still
fails in two of the three, for a reason that has nothing to do with accelerators. That is what
forced the L4 correction above.

Not fixed here — recorded. The fix is a product decision with more than one defensible answer (make
the attachment multi-region in Terraform; derive the ID per candidate region and skip candidates
without one; or narrow `ray_regions` to the deployed region and drop the pretence of failover), and
choosing it mid-campaign would move an axis under runs already recorded. Until then, treat
`ray_regions` beyond the deployed region as **advertised but non-functional under PSC-I**.

### A deep-learning family on Ray with `hardware: cpu` builds a cluster with no workers and hangs forever

The other wave-6 stand-in — smoke 10's config with the `deep_learning` family moved from
`{"runtime":"ray","hardware":"gpu","gpu_type":"T4"}` to `{"runtime":"ray","hardware":"cpu"}` — did
not fail. It provisioned, submitted, and then sat. The other three families finished; the Ray one
was still running **1h34m** later, with its autoscaler repeating:

```
No available node types can fulfill resource request {'CPU': 1.0}
```

The cluster was up and healthy. It had no workers to run anything on.

**The chain is three correct-looking decisions that compose into a dead run.**
`engines/ray_io.split_gpu_cpu_models` partitions the model list by each model's **registered
family**, so `neuralprophet` is in `gpu_models` no matter what hardware was asked for. `hardware:
cpu` makes `effective_use_gpu` false, and the sizing call then passes `n_gpu_cells` as **0**. With
every model in `gpu_models`, `cpu_models` is empty, so `n_cpu_cells` is **0** too. `_build_workers`
omits any pool with zero planned nodes — correctly, since Vertex rejects a zero-node worker type —
and both pools are omitted. Vertex accepts a head-only cluster, Ray accepts the job, and the job
waits for a worker that will never be created. There is no timeout: the submitter polls until
terminal, so the harness blocked indefinitely and the run had to be stopped by hand.

**This is config-reachable and it is the obvious thing to try.** Anyone whose project lacks Vertex
GPU quota — or who simply does not want to pay for accelerators — reaches for exactly this edit to
get the DL family running on something. It costs a cluster-hour and produces no error message. This
project turned out not to be in that state after all, which lowers how often *we* will hit it and
changes nothing about the defect.

Not fixed at the time, for the same reason as the failover above: there is more than one defensible
answer (fold DL models into the CPU pool when `use_gpu` is false, so a CPU Ray run of a DL family
simply runs slowly; or reject a zero-worker plan before provisioning; or both), and the sizing path
feeds `run_id`-relevant config. The zero-worker plan is the detectable signal — no valid run ever
wants one.

**Fixed since, in two steps.** `split_gpu_cpu_models` took the first option and grew a `use_gpu`
parameter: with no GPU pool, deep-learning cells belong to the CPU pool rather than to a pool that
will not exist. That left one gap, because the engine was still computing `use_gpu` from the flat
`compute.use_gpu` field — so a config combining flat `use_gpu: true` with a family override of
`hardware: "cpu"` still routed cells at a device the submitter had not bought. P1 closes that by
having the engine resolve its GPU decision from the same per-family resolver the submitter
provisions from (see the section above). Both directions are covered offline by
`tests/unit/test_gpu_routing_coherence.py`; **neither has been re-proven live**, and a Ray
deep-learning family on `hardware: cpu` remains an untested live shape.

### Five Ray rows bought accelerators that no cell ever saw

There are two ways to ask for a GPU. The flat `compute.use_gpu` / `compute.gpu_type` pair is the
original one; `compute.families.deep_learning.hardware` is the per-family one, and it is what every
example config written since the multi-runtime DAG landed uses.
`config.RunConfig.resolve_family_compute` reconciles the two, and the submitter provisions from its
answer. `engines/ray_engine` did not ask it — it read the flat field directly, in five places. For a
per-family config the flat field is `False`, so the submitter provisioned accelerators and the
engine sent every deep-learning cell to the CPU pool. Nothing raised, nothing warned, and the run
finished green.

The registry says so unambiguously, and it says so on the recorded runs rather than on a
reconstruction. Across the five per-family Ray rows — smokes 08, 09, 10, 15 and
`per_family_runtimes_demo` — **550 of 550 `neuralprophet` cells report `peak_gpu_bytes` as NULL,
and all 550 report `cpu_seconds`.** That second half is what makes the first half readable: a NULL
`peak_gpu_bytes` means either "no CUDA device was visible" or "profiling was off", and a recorded
`cpu_seconds` rules out the second. The three Ray rows that ask through the flat field recorded a
device on every cell in the same period — `ray_gpu_demo` at 56,320 bytes, both `all_families_10k`
runs at 77,824 — so this is not a gap in the probe.

So the five rows are STALE for the plainest possible reason: what they claim to prove is the thing
that did not happen. Smoke 08's whole purpose is "Ray on Vertex, GPU T4"; smoke 09's is a shared
cluster with a GPU pool beside the CPU one. Their other claims — the DAG shape, the shared cluster,
Airflow orchestrating five models across three runtimes — all held, and the runs are real. Only the
accelerator half is void, and the ledger has no way to stale half a row.

Two details worth keeping. First, the numbers those runs produced are still correct: NeuralProphet
falls back to CPU inside the cell, so the forecasts are forecasts, just slower and on hardware
nobody meant to pay for. Second, the bug ran in the other direction too — flat `use_gpu: true` with
a family override of `hardware: "cpu"` had the engine asking Ray for `num_gpus` against a cluster
with no GPU nodes, which presents as a permanent hang rather than an error. No shipped config is in
that shape, so no row is affected, but it is the same one-line cause.

P1 gives the engine one source of truth, `ray_io.resolve_job_gpu`, which asks the same resolver the
submitter provisions from. A static test walks `ray_engine`'s AST and fails if any read of
`...compute.use_gpu` or `...compute.gpu_type` comes back, because fixing five call sites does not
stop a sixth from being added; the behavioural tests cover both directions. The `peak_gpu_bytes`
probe is now unconditional rather than gated on profiling, so a future run of this shape leaves
evidence either way instead of leaving nothing. **Re-running these five is the point of the live
campaign, and the bar for a re-run is not a green PASS — it is a non-NULL `peak_gpu_bytes` on the
deep-learning cells.**

### Why almost everything Spark is stale

Both ran the Ray path on the custom container image. `822ae25` deleted that path — the custom image
fails Vertex Ray GPU provisioning, so all Ray moved to the stock prebuilt image with dependencies
delivered by Ray 2.47's `runtime_env` uv plugin. All four Ray rows (07–10) have since been re-run on
the new path and passed, including smoke 10 — the strongest proof in the suite, all four families
across all three runtimes under a single `run_id`. They are stale again for a different and narrower
reason, recorded below: they stopped pinning `ray_autoscale`.

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

The campaign that re-earns the rest is profiler **W12**, which also does the `off`-vs-profiled A/B.
It no longer has to capture the baseline's measurements — the 100k Ray run already carried them, so
the baseline shipped ahead of it (see below).

### Smokes 07–10 stopped pinning `ray_autoscale`, which was the point (07 re-earned 2026-09-03)

All four pinned `"ray_autoscale": false` from `4c988bc` (2026-08-25), when a per-pool
`AutoscalingSpec` crashed the Vertex Ray head at provisioning. That crash was later attributed to
the **custom node image**, which no longer exists — every Ray path moved to the stock image plus the
`runtime_env` uv plugin. So the pin was a workaround for a component that has been deleted, and
while it stood, the smoke suite — the part of the record that is supposed to cover the shipped
defaults — was the one part of it not exercising this one.

The autoscaling path itself is not unproven: `ray_autoscale_demo` reached the derived ceiling of 8
on CPU, `ray_gpu_demo` scaled a T4 pool, and `ray_100k` ran at scale, all with `ray_autoscale: true`.
Those four rows now declare `ray_pool_shape=autoscaling` so the record says where the proof lives.
What was unproven is **these four configs** on it — including the two that stand up GPU pools and
the one that shares a cluster across families, which is where a provisioning-time crash would land.
All four have since re-run (below), including the three that stand up T4 pools — the ones the
original crash would actually have hit.

Removing the key (rather than setting it to `true`) is the deliberate edit: the config then runs
whatever the product ships, which is the thing under test. Both spellings hash identically —
identity is `authored-config-only`, and an explicitly-authored `true` and an absent key resolve to
the same value — so this is a genuine behaviour change, not a cosmetic one.

**Four `run_id`s move and the new `ray_pool_shape` axis moves under them**, so four rows go stale by
construction; they were re-graded in the same commit as the config edit:

| # | Proven id | The id its config now resolves to | Re-run |
|---|---|---|---|
| 07 | `smoke-07-ray-cpu-782bcec2718f` | `smoke-07-ray-cpu-2cb4115312b1` | **done 2026-09-03 — CURRENT** |
| 08 | `smoke-08-ray-gpu-c41ecf2d5d52` | `smoke-08-ray-gpu-38e33f02fd6d` | **done 2026-09-03 — CURRENT** |
| 09 | `smoke-09-shared-ray-1d308b8a712c` | `smoke-09-shared-ray-f42e5785f6b9` | **done 2026-09-03 — CURRENT** |
| 10 | `smoke-10-mixed-runtimes-f6f98f70eb80` | `smoke-10-mixed-runtimes-a39f0fb4f3fa` | **done 2026-09-04 — CURRENT** |

**All four re-ran, all four landed on the id this table predicted, and all four passed.** 10 closed
it out with an autoscaling Ray GPU pool running alongside two Dataproc Serverless batches and a
BigQuery job under one `run_id`. Every cluster was confirmed gone afterwards by `describe`/REST, not
by the SDK's farewell line. `ray_pool_shape` therefore has no `fixed-size` rows left anywhere in this
document: the pin from `4c988bc` is fully retired, in the record as well as in the configs.

**07 re-ran on 2026-09-03 and landed on exactly the predicted id**, which is the first thing worth
recording: the table above was arithmetic against uncommitted code, and the cluster agreed with it.
The run provisioned, executed both CPU families and tore down; teardown was confirmed by the
v1beta1 `persistentResources` endpoint returning no clusters, not by the SDK's "Successfully deleted"
line, which is not evidence.

The header's `job_telemetry` carries the actual proof of the axis: `autoscale: true` with
`cpu_min_nodes: 1` against `cpu_max_nodes: 4`. A fixed-size pool has no such spec at all, so this is
the pool shape itself and not a config echo. Its row also picks up `fleet_sizing=derived-overlay` —
the sizing plan recorded `derived_units: 2` under `max_units: 2` on a `basis: measured` slot, so W1's
derived ceiling reached a cluster that could actually act on it. That is the coupling the axis prose
predicted, observed rather than argued. `gpu_node_count: 0`, so the run consumed none of the four
T4s and could not have contended with anything.

The re-run also disposes of the id discrepancy described below, in the only way that was ever going
to: row 07 now names an id that re-derives from the config beside it.

**08 followed, and it is the one that actually retires the `4c988bc` fear.** 07 autoscales a CPU
pool; 08 autoscales a **GPU** pool, which is the precise shape that crashed the Vertex Ray head and
caused the pin. It provisioned, trained `neuralprophet` on 100 series, and tore down —
`total_wall_s: 873.8`, `job_status: SUCCEEDED`, teardown confirmed by the REST endpoint. The
workaround has now outlived both the component it worked around and the failure it prevented.

**And it took seven T4s, which is more than this project is documented to have.** See below — that
turned out to be the more valuable finding of the two.

**09 then autoscaled both pools on one shared cluster** (`reuse: true`, `autoscale: true`,
`total_wall_s: 627.8`), running `statistical`, `ml` and `deep_learning` under a single
`persistentResource`. This is the case where a provisioning-time crash had the most surface — two
`AutoscalingSpec`s on one cluster — and it is now proven. It also took a 7-node GPU pool.

#### `NVIDIA_T4_GPUS` is not the quota that binds a Ray run

`quota_and_scale.md` names `NVIDIA_T4_GPUS` (Compute Engine, default **4**) as the metric limiting
deep-learning workers, and builds its deep-learning scale table on "the default 4 T4s". That is
correct for the **Dataproc** path and wrong for the **Vertex Ray** path, which is a different
service drawing on a different pool. Measured on this project, 2026-09-03:

| Pool | Metric | us-central1 | us-east1 | us-west1 |
|---|---|---|---|---|
| Compute Engine (Dataproc clusters) | `NVIDIA_T4_GPUS` | 4 | — | — |
| Vertex AI (Ray on Vertex) | `aiplatform.../custom_model_training_nvidia_t4_gpus` | **12** | 2 | 2 |

Smoke 08 derived a 7-node GPU pool and got all seven. Nothing is over-committed and nothing is
broken: `ray_max_nodes` (default 16) is the product's only guardrail here, 7 sits under it, and 7
sits under the Vertex limit of 12. The mistake is entirely in the documentation — and it is the
expensive direction of wrong, because a reader sizing a Ray run against the doc will believe they
have a third of the headroom they actually have, and the `gcloud compute regions describe` recipe
the doc gives them cannot show the number that binds them.

Two consequences worth stating plainly, neither of them acted on in that commit (the first has been
since):

- **~~`all_families_10k.json` pins `ray_gpu_max_nodes: 4`~~ — raised to 12 and run.** It had been
  sized to the Compute Engine number while running `python_runtime: "ray"`, so it asked for a third
  of the pool it actually draws from. It now asks for 12, and on 2026-09-04 it got all 12. The
  "~10 hours" this table quotes for it was a 4-GPU estimate; the run took 7 h 45 m on 12 GPUs, which
  is not the vindication it looks like — see `ray_slot_memory` above for why 12 GPUs performed like
  one.
- **11b's `HARD_CEILING` fixture survives, and is now exact.** `us-east1` allows 2 Vertex T4s, so a
  Ray GPU config pointed there asking for 3 hits a real ceiling on demand. That was previously a
  hope about an unmeasured limit; it is now a number.

**One thing the arithmetic turned up that is worth recording.** For 08, 09 and 10 the "proven id"
column is exactly what their committed config produced before this edit — checked against both
today's code and the last pushed commit. For 07 it is not: `07_ray_cpu.json` resolves to
`smoke-07-ray-cpu-af6d7d054b5b` under either, never to the `…-782bcec2718f` its row names. That run
is real and `COMPLETED` in the registry on 2026-09-01, and it is not the
not-recomputable-from-config category the gaps section describes — that one moved *every* id
together, and its three siblings match. The likeliest reading is a hand-edited variant, of the kind
wave 6.1 used to separate the Ray region-failover blocker from the GPU one, whose id the row then
inherited; the difference has not been reconstructed and is not worth the archaeology, because the
re-run replaces the pointer either way. Recorded rather than quietly overwritten: a `run_id` column
whose entries are not re-derivable from the config beside them is the one failure this table cannot
absorb, and it took arithmetic — not reading — to notice.

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
| `bq_native_demo.json` | The BigQuery-native family alone — no cluster of any kind (100 series) | STALE | 2026-09-01 | `bq-native-demo-b374041fdd1e` | `python=3.11`, `run_id_inputs=+compute.profile.source` |
| `explode_demo.json` | The Spark `explode` fan-out, statistical + ML, artifacts persisted (10) | CURRENT | 2026-09-10 | `explode-demo-088f172ad2f5` | `serverless_deps=container-image`, `python=3.11`, `fleet_sizing=derived-overlay-three-way-min`, `run_id_inputs=authored-config-only-v3`, `horizon_features=computed-at-future-dates` |
| `mixed_demo.json` | One Spark model and the natives under one `run_id`, backtested and ranked on one leaderboard (10) | CURRENT | 2026-09-10 | `mixed-demo-db2dfb2f675d` | `backtest_scoring=holdout-reserved+embargo-aware+auto-refit`, `serverless_deps=container-image`, `native_source_pin=unpinned-all-sources`, `python=3.11`, `fleet_sizing=derived-overlay-three-way-min`, `run_id_inputs=authored-config-only-v3`, `horizon_features=computed-at-future-dates` |
| `ensemble_demo.json` | The same mix with three ensemble strategies on, ranked inside the same board (10) — re-run under the weighting fix as attempt 2, which also made it the clearest demonstration of the write-timestamp tiebreak (see below) | CURRENT | 2026-09-12 | `ensemble-demo-b2ff15a4d418` (attempt 2) | `ensemble_weighting=per-series-calculated+batch-fit-learned`, `backtest_scoring=holdout-reserved+embargo-aware+auto-refit`, `serverless_deps=container-image`, `native_source_pin=unpinned-all-sources`, `python=3.11`, `fleet_sizing=derived-overlay-three-way-min`, `run_id_inputs=authored-config-only-v3`, `horizon_features=computed-at-future-dates` |
| `per_family_runtimes_demo.json` | Per-family runtime split — deep learning to Ray GPU, statistical and ml to Serverless Spark, native to BigQuery, all four under one `run_id` (50) | CURRENT | 2026-09-10 | `per-family-runtimes-demo-8fe8f224a7e1` | `serverless_deps=container-image`, `ray_deps=stock-image+uv-runtime-env`, `ray_pool_shape=autoscaling`, `native_source_pin=unpinned-all-sources`, `gpu_device_probe=trainer-root-device`, `dl_gpu_routing=resolved-per-family`, `python=3.11`, `fleet_sizing=derived-overlay-three-way-min`, `run_id_inputs=authored-config-only-v3`, `horizon_features=computed-at-future-dates` |
| `ray_cpu_demo.json` | Ray on Vertex, CPU, alongside the natives, backtested (6) | STALE | 2026-09-01 | `ray-cpu-demo-f6b6fbdb83a5` | `ray_pool_shape=autoscaling`, `ray_deps=stock-image+uv-runtime-env`, `python=3.11`, `fleet_sizing=derived-overlay`, `run_id_inputs=authored-config-only` |
| `ray_gpu_demo.json` | Ray on Vertex, GPU T4 (`neuralprophet`), alongside the natives (6) | STALE | 2026-09-02 | `ray-gpu-demo-e2dcbef4a373` | `ray_pool_shape=autoscaling`, `ray_deps=stock-image+uv-runtime-env`, `python=3.11`, `fleet_sizing=derived-overlay`, `native_source_pin=unpinned-all-sources`, `run_id_inputs=authored-config-only` |
| `ray_autoscale_demo.json` | **The shipped `ray_autoscale=true` default**, 1→8 CPU nodes at 10,000 series | CURRENT | 2026-09-10 | `ray-autoscale-demo-9728c900963a` | `ray_pool_shape=autoscaling`, `ray_deps=stock-image+uv-runtime-env`, `python=3.11`, `fleet_sizing=derived-overlay-three-way-min`, `run_id_inputs=authored-config-only-v3`, `horizon_features=computed-at-future-dates`, `ray_slot_memory=harvest-only` |
| `explode_100k.json` | The headline: Spark `explode` over 100,000 series | CURRENT | 2026-09-10 | `explode-100k-ef602ea229b4` | `serverless_deps=container-image`, `python=3.11`, `fleet_sizing=derived-overlay-three-way-min`, `run_id_inputs=authored-config-only-v3`, `horizon_features=computed-at-future-dates` |
| `ray_100k.json` | The same work on Ray — the runtime-parity half of the scale review | CURRENT | 2026-09-10 | `ray-100k-3fbc82fe3b6d` | `ray_pool_shape=autoscaling`, `ray_deps=stock-image+uv-runtime-env`, `python=3.11`, `fleet_sizing=derived-overlay-three-way-min`, `run_id_inputs=authored-config-only-v3`, `horizon_features=computed-at-future-dates`, `ray_slot_memory=harvest-only`, `ray_poll_recovery=transient-transport+auth` |
| `all_families_10k.json` | Every family under one `run_id` — all four on Ray + BigQuery at 10,000 series, on the 12 T4s this project's Vertex quota allows | STALE | 2026-09-04 | `all-families-10k-eb01dcfecfab` | `ray_pool_shape=autoscaling`, `ray_deps=stock-image+uv-runtime-env`, `python=3.11`, `fleet_sizing=derived-overlay`, `run_id_inputs=authored-config-only`, `horizon_features=computed-at-future-dates`, `ray_slot_memory=harvest-only` |
| `all_families_10k_full.json` | As above, plus backtesting and persisted artifacts | STALE | 2026-09-05 | `all-families-10k-full-e68d9341ce01` | `ray_pool_shape=autoscaling`, `ray_deps=stock-image+uv-runtime-env`, `python=3.11`, `fleet_sizing=derived-overlay`, `run_id_inputs=authored-config-only`, `horizon_features=computed-at-future-dates`, `ray_slot_memory=harvest-only` |
| `repair_demo.json` | The repair ladder's refusal — a family lost *mid-write*, leaving two models partly landed, which `--retry` classifies correctly and then declines to submit (3,000 series, two families) | CURRENT | 2026-09-11 | `repair-demo-55119d4c6f7c` | `serverless_deps=container-image`, `python=3.11`, `fleet_sizing=derived-overlay-three-way-min`, `run_id_inputs=authored-config-only-v3`, `horizon_features=computed-at-future-dates` |
| `repair_retry_demo.json` | The repair ladder end to end — a family lost *during provisioning* lands nothing, and `--retry` re-submits exactly it under a `statistical_repair` token while a second, deliberately cancelled family is left alone (300 series, two families) | CURRENT | 2026-09-11 | `repair-retry-demo-59310436a6fd` | `serverless_deps=container-image`, `serverless_cancel=operation-cancel`, `python=3.11`, `fleet_sizing=derived-overlay-three-way-min`, `run_id_inputs=authored-config-only-v3`, `horizon_features=computed-at-future-dates` |
| `neuralprophet_ab_gpu.json` | The GPU arm of the accelerator A/B — 10,000 NeuralProphet cells on twelve T4 nodes | CURRENT | 2026-09-10 | `neuralprophet-ab-gpu-e530eea3a755` | `ray_deps=stock-image+uv-runtime-env`, `ray_pool_shape=autoscaling`, `ray_slot_memory=harvest-only`, `dl_gpu_routing=resolved-per-family`, `gpu_device_probe=trainer-root-device`, `backtest_scoring=holdout-reserved+embargo-aware+auto-refit`, `python=3.11`, `fleet_sizing=derived-overlay-three-way-min`, `run_id_inputs=authored-config-only-v3`, `horizon_features=computed-at-future-dates`, `ray_poll_recovery=transient-transport+auth` |
| `neuralprophet_ab_cpu.json` | The CPU arm of the same A/B — the identical config with the deep-learning family on CPU | CURRENT | 2026-09-11 | `neuralprophet-ab-cpu-f4bfff3b39e9` | `ray_deps=stock-image+uv-runtime-env`, `ray_pool_shape=autoscaling`, `ray_slot_memory=harvest-only`, `dl_gpu_routing=resolved-per-family`, `gpu_device_probe=trainer-root-device`, `backtest_scoring=holdout-reserved+embargo-aware+auto-refit`, `python=3.11`, `fleet_sizing=derived-overlay-three-way-min`, `run_id_inputs=authored-config-only-v3`, `horizon_features=computed-at-future-dates`, `ray_poll_recovery=transient-transport+auth` |

#### 2026-09-10, `ray_autoscale_demo`: the first side-by-side of what was planned and what ran

The re-run is `ray-autoscale-demo-9728c900963a`, `COMPLETED` in 2,160 s against the prior pass's
2,513 s, with 30,000 cells and no errors. Two things in it are worth more than the pass.

**The node curve still reaches eight.** The run-level telemetry records `derived_units: 8`,
`cpu_node_count: 8` and `total_worker_nodes: 8`, against an authored ceiling of `ray_cpu_max_nodes:
8`. That is the check this rung exists for: the derived fleet size holds back one unit as a reserve,
and if that reserve had been scoped to the wrong number the pool would come up at seven and nothing
would fail — the run would simply be quietly smaller than the config asked for. It comes up at eight.

**`sizing` and `sizing_executed` were compared here for the first time, and they differ in exactly
one place.** The planned slot measured both axes and carried a memory reservation:

```
sizing:           slot.measured = ["cores", "memory_bytes"]   memory_bytes = 2153924199
                  task_options  = {"num_cpus": 1, "memory": 2153924199}
sizing_executed:  slot.measured = ["cores"]   slot.assumed = ["memory_bytes"]   memory_bytes = null
                  task_options  = {"num_cpus": 1}
```

The executed plan drops the memory request before it reaches the scheduler. That is the
`ray_slot_memory=harvest-only` axis doing precisely what its name says, and it is deliberate: Ray
enforces a memory request as hard as a core request, so pinning the measured figure per task packs
one cell onto a node and costs a factor of 3.8 in throughput. Harvesting the number without imposing
it keeps the measurement — which is what feeds later sizing decisions — while letting cores set
concurrency. The proof that nothing was lost is that both structures land on `slots_per_unit: 7` and
`total_slots: 56`. Same concurrency, reached by cores alone. Until this run the two structures had
only ever been read one at a time, so "the executed plan matches the planned one except where the
axis says otherwise" was a reasonable belief rather than a recorded observation. It is now recorded.

The GPU half of the plan is present and empty, which is also correct: this config authors no
deep-learning model, so the deep-learning pool plans `n_cells: 0` and `derived_units: 0`, and the
cluster came up with `gpu_node_count: 0`. No accelerator was bought for a family with nothing to run.

#### 2026-09-10, `explode_demo`: the artifact claim checked rather than inherited

`explode-demo-088f172ad2f5`, `COMPLETED`, both the statistical and ml jobs green, 40 of 40 cells
`ok`. This row's claim ends in "artifacts persisted", and `explode_demo` is the only config in the
tree that sets `persist_models: true`, so that clause was verified on its own rather than allowed to
ride along on the harness `PASS`. All 40 `forecast_metadata` rows carry a non-null `model_artifact`
reference, and the bucket prefix for the run holds exactly 40 objects totalling 645.66 MiB. Both
halves matter: rows with references and no objects, or objects with no rows, would each still let
the harness report a pass. The leaderboard shows `wape=None` for all four models, which is correct —
this config authors no `backtest` block, so there is nothing to score.

#### 2026-09-10, `explode_100k`: the hundred-thousand rung, re-proven under the current architecture

`explode-100k-ef602ea229b4`, `COMPLETED`, both jobs green, 400,000 distinct
(`ts_id`, `model_type`, `forecast_date`) prediction cells — exactly 100,000 series × 4 models × a
28-day horizon, with nothing dropped. This config is not workshop material. The configs a reader is
walked through top out at 10,000 series, because 10,000 is the largest scale a stock project can
reproduce without asking for quota. `explode_100k` and `ray_100k` are kept for one purpose: to show
that the same code, unchanged, runs an order of magnitude larger.

Against the 2026-09-01 run of the identical config, both families finished faster:

| Family | 2026-09-01 | 2026-09-10 | |
|--------|-----------|-----------|---|
| `statistical` | 7,053.7 s (117.6 m) | 5,847.7 s (97.5 m) | 1.21x |
| `ml` | 3,329.3 s (55.5 m) | 2,599.7 s (43.3 m) | 1.28x |

Read that as a sanity check, not as a measurement. The config pins `max_executors: 20` in both runs,
so the ceiling on parallelism was the same, but three architecture axes moved between the two dates
(`fleet_sizing`, `run_id_inputs`, and the horizon-feature fix), and Serverless does not promise the
same machines twice. What the pair establishes is that nothing regressed at scale while those axes
were changing — the kind of claim a headline A/B is built to make properly, and this is not one.

#### 2026-09-10, `ray_100k`: a dropped HTTPS request killed a healthy twenty-node run

The first attempt at `ray-100k-3fbc82fe3b6d` died 79.5 minutes in, with 224,967 of its 400,000 cells
already written to BigQuery and climbing steadily. Both jobs were marked `FAILED` and the fleet was
torn down. Nothing had gone wrong with the run.

What happened is that the driver polls Ray for job status every fifteen seconds, over a public HTTPS
proxy, for as long as the run lasts — several thousand requests on a run this size. One of them died
in transit:

```
SSLError(5, '[SSL: UNEXPECTED_EOF_WHILE_READING] unexpected eof while reading')
```

That is the connection ending mid-response. It says nothing whatsoever about the job on the other
end. But `_submit_and_poll` forgave exactly one kind of poll failure — an expired bearer token,
because a long run outliving its token had already cost us a run once — and
re-raised everything else as a job failure. So a transport blip was reported as two failed jobs, and
the ordinary teardown path then removed twenty nodes of healthy, paid-for work.

The evidence that the jobs were fine is circumstantial but consistent: `failure_reason` is null on
both rows, Ray never reported a `FAILED` state, and cells were landing in `forecast_predictions`
right up to the cut. It cannot be more than circumstantial, because our own teardown destroyed the
cluster we would need to ask. That asymmetry is the whole argument for the fix — being wrong in the
forgiving direction costs a few minutes of a fleet already paid for, and being wrong in the strict
direction destroys a multi-hour run.

The odd part is that the code already knew this error was transient. `_is_dashboard_warmup_error`
classifies exactly this string, and had done since the dashboard-handshake work; the poll loop
simply never consulted it. The fix (`_is_recoverable_poll_error`, `_status_with_recovery`) makes the
poll loop forgive both shapes — expired auth and dropped transport — reconnecting and retrying up to
four consecutive times with a fifteen-second wait, and re-raising anything else untouched. The
budget is per-poll, so a run that hiccups once an hour never accumulates toward it. Eight offline
tests cover the classifier and the retry loop, including the verbatim live error string above, since
the classifier reads the message text and a paraphrase would test something the proxy never sends.

This is what the `ray_poll_recovery` axis names, and why finishing a long Ray run under the old
value proved something weaker than it looked: it proved no request happened to drop, not that a
dropped request was survivable.

**The re-run passed, and it is worth being exact about what that does and does not show.** Submitted
with `--force`, so it kept the same `run_id` and topped up rather than starting over: 400,000 new
`forecast_metadata` rows on top of the 224,967 the first attempt left, `COMPLETED` on both jobs at
attempt 2, all four models at 100,000 cells each. Statistical took 6,620.0 s (110.3 minutes) and ml
4,559.1 s (76.0 minutes) — the whole run ran half an hour past the point where the first attempt
died. Teardown was confirmed by the v1beta1 REST endpoint returning `{}`, not by the SDK's success
line.

But the recovery path logged **zero** retries. No request dropped this time, so the fix was never
exercised on live infrastructure. What this run proves is that a 110-minute Ray run at 100,000
series completes and lands every cell; what proves the recovery path is the eight offline tests,
which drive it with the verbatim error string the proxy actually sent. Those are two different
claims and the ledger should not blur them — which is the same distinction that made the old axis
value misleading in the first place.

**`all_families_10k` ran twice on 2026-09-04, and the pair is the `ray_slot_memory` A/B.** The first
pass is the run that found the defect; the second is the identical config under the fix, submitted
with `--force` so it kept the same `run_id`. Both reached the same result — all four families under
one `run_id`, `COMPLETED`, `n_series 10000`, `n_models 7`, and 1,960,000 distinct
(`ts_id`, `model_type`, `forecast_date`) prediction cells, exactly 10,000 × 7 × 28, with 10,000
`forecast_metadata` rows per model. (`forecast_predictions` holds 3,920,000 raw rows for the run —
both attempts, deduped on read, which is the documented idempotency model.) What changed is
everything about how long it took:

| Family | Runtime | Before | After | |
|--------|---------|--------|-------|---|
| `native` | BigQuery | 673 s | 417 s | 1.6x |
| `ml` | Ray CPU | 2,385 s | 1,637 s | 1.5x |
| `statistical` | Ray CPU | 3,088 s | 1,575 s | 2.0x |
| `deep_learning` | Ray GPU (T4) | 27,245 s (7 h 34 m) | **6,582 s** (1 h 50 m) | **4.1x** |
| whole run | — | 27,932 s (7 h 45 m) | **7,353 s** (2 h 02 m) | **3.8x** |

**Before.** All 12 T4 nodes were provisioned and all 12 worked the whole run (727–906 cells each,
evenly spread) — and each ran **exactly one cell at a time**: 1.0 of 7.0 CPUs, 0.1 of 1.0 GPU, and
17.6 of 18.1 GiB of memory held per node. The driver's pre-pass had measured 13.58 GiB where the
workers' own `forecast_metadata.process_rss_bytes` rows for `neuralprophet` peak at 1.51 GiB, a ~9x
overstatement that the 1.3 memory margin turned into a 17.65 GiB request against 18.12 GiB
schedulable.

**After.** The same live read, taken mid-run: **0.0 of 18.1 GiB** of memory requested on every node,
**7.0 of 7.0 CPUs busy** on every node, 126.0 of 126.0 cluster-wide, T4s at up to 0.7 of 1.0, and a
queue of 3,213 tasks all asking for `{"CPU": 1.0}` and nothing else. Cores are the binding axis, as
predicted. The quota preflight had already said so before a single node was created — *"10000 cells
/ 10 per node"*, against the one cell per node the first pass achieved.

**The two numbers that matter are not the same number, and the gap between them is the interesting
part.** Density went from 0.93 to 5.5 concurrent cells per node (`SUM(fit_seconds) / wall / nodes`),
a **5.9x** improvement — but wall-clock improved 4.1x, because the average `neuralprophet` fit
*slowed* from 30.25 s to 43.48 s. That is contention, and it is the expected shape: seven cells
sharing a T4 and seven cores each finish more slowly than one cell with the node to itself. A fleet
that looks 44 % slower per cell is doing 4.1x the work. Per-device throughput went from 1.8 to
**7.6 `neuralprophet` cells/min per T4**, which is the number `quota_and_scale.md` now plans from.

Two operational notes from the second pass. The Ray jobs client crossed the bearer-token TTL at
21:40 and refreshed itself rather than 401-ing, so the long-run failure mode from earlier campaigns
did not recur. Teardown was REST-verified: 404 on the resource and `{}` on the collection, not the
SDK's "Successfully deleted the cluster" line.

#### 2026-09-10/11, the NeuralProphet accelerator A/B: the T4 is 18 % *slower*, not merely not-faster

Two runs, identical but for one line. `neuralprophet-ab-gpu-e530eea3a755` and
`neuralprophet-ab-cpu-f4bfff3b39e9`, 10,000 NeuralProphet cells each on a twelve-node Ray fleet,
backtested at two folds, both `COMPLETED` at attempt 1, teardown REST-verified `{}` on both. The
configs differ only in `compute.families.deep_learning.hardware`. Both pin `gpu_fraction: 0.125` and
`compute.profile.source: "baseline"` — that second pin is worth recording here because it is excluded
from the `run_id` digest, so it is not recoverable from the identifiers above and a future reader
cannot check it any other way.

| | GPU arm | CPU arm |
|---|---|---|
| Wall clock | 19,054.3 s (5.29 h) | 14,828.0 s (4.12 h) |
| Cells landed / ok | 10,000 / 10,000 | 10,000 / 10,000 |
| Fits (3 per cell — two folds refit plus full history) | 30,000 | 30,000 |
| Mean seconds per cell / per fit | 126.5 / 42.2 | 97.5 / 32.5 |
| WAPE | 0.35006025869087765 | 0.35006025868556495 |
| `device_share` of fit time | 1.0 | 0.0 |
| Cost per 1,000 fits (CPU node-seconds, r = 1.92) | 110,937 | 47,354 |

All four pre-registered controls passed before the decision was read: identical landed cell counts
and distinct series, device telemetry present on the GPU arm and absent on the CPU arm,
`cpu_seconds/fit_seconds` at most 1.005 on every cell with a single `intraop_threads` value per arm,
and measured wave density within 1.0E-4 across 119 waves. The decision rule then fired cleanly:
**s = 0.8196, r = 1.92, `device_share` = 1.0 → CPU, the accelerator does not pay for itself.**

**The prediction was wrong, and in the interesting direction.** Phase 0 pre-registered s in
[0.95, 1.07] — the expectation was rough parity, an accelerator that neither helps nor hurts on a
model that allocates 87 KB of device memory. The measurement is 0.82. Attaching a T4 did not fail to
help; it made every fit about 30 seconds slower. The most likely reading is that host-to-device
transfer and kernel-launch overhead on a network this small exceed the arithmetic they replace, so
the device is pure overhead on the critical path. That is a stronger result than parity would have
been, and it is stronger against us: it says the GPU path costs 2.34x per fit rather than the ~1.9x
the surcharge alone implies.

**The conclusion does not depend on how the fleet is counted.** The query divides by
`COUNT(DISTINCT worker_id)`, which came out 102 on the GPU arm and 108 on the CPU arm — a scheduling
outcome, not what was purchased. Dividing instead by the twelve nodes actually paid for on both arms
gives s = 0.774 and a cost ratio of 2.48. Either denominator lands in the same branch of the same
rule, so the verdict is not an artifact of that choice.

**The pre-registered query needed a repair, and it must be read with that in mind.** Its denominator
was `SUM(n_fits)`, and `n_fits` is a column the `forecast_metadata` row spec declares and no writer
has ever populated — nor `train_rows_total` beside it. So the sum was NULL, the throughput ratio was
NULL, and the `CASE` fell through to `INCONCLUSIVE`. That verdict agreed with the shipped default and
with every prior expectation, which is precisely why it would have been comfortable to accept. It was
a defect, not a result.

The repair derives the fit count from the backtest columns on each cell row — with
`backtest_refit='per_fold'` a cell fits once per fold plus once on full history, so
`n_folds_achieved + 1`, which is 3 here and 30,000 fits per arm. It is written into
[`docs/sql/neuralprophet_ab.sql`](sql/neuralprophet_ab.sql) as a dated note rather than applied
silently. The argument that it cannot have favoured an arm is arithmetic, not assurance: both arms
landed exactly 10,000 cells under the identical backtest block, so whatever per-cell constant one
chooses is the same on both sides and cancels in s, which is a ratio between the arms. Getting it
wrong rescales both cost columns by one shared factor and leaves the decision alone — which is what
happened on the first pass at the repair, where a flat `COUNT(*)` put both costs 3x high and s at
exactly the 0.8196 reported above.

**One thing the diagnosis got wrong on the way, and correcting it is the point.** The original SQL
said section 5 "reads the fold rows too", and no row in `forecast_metadata` carries a non-null
`fold_id` — which read at first like a backtest writing no folds despite reporting
`backtest_status='full'`. It is not. `forecast_metadata` is one row per cell by design and the
per-fold rows live in `backtest_oof`, 560,000 of them per arm: two folds by ten thousand series by a
twenty-eight-step horizon. Nothing was missing; the comment named the wrong table. What remains
genuinely open is smaller and duller — `n_fits` and `train_rows_total` are declared in the row spec
and written by nobody, and should either be wired up or removed.

**On placement versus utilisation.** The GPU arm's `device_audit` recorded that the deep-learning
family "used its device but barely touched it (peak 87,040 bytes on a T4)". Both halves of that are
the point. Placement works — the routing, the fraction, the probe and the audit all did their jobs,
and `device_share` of 1.0 says every second of GPU-arm fit time had a device holding memory.
Utilisation is 87 KB against 16 GB, roughly 0.0005 % of the card. The contract this project added
was that a GPU run must actually reach a device; it was never that reaching one is worth paying for.
This A/B is the measurement that separates the two, and it says `use_gpu: False` stays the default.

#### 2026-09-11, the repair ladder: `--cancel` had never once stopped a Serverless batch

Tier 5 exists to run the repair ladder against live infrastructure rather than against its own unit
tests, all eight of whose verdicts were already covered offline. It found two defects on its first
two attempts, and neither was in the classifier.

**The first was `--cancel` itself.** `repair_demo` was submitted to create a partial run by stopping
one of its two families mid-flight. The verb refused:

```
Cancelled run repair-demo-55119d4c6f7c: 0 of 1 in-flight job(s) stopped
  statistical  NOT cancelled  FailedPrecondition: 400 Cannot delete non-terminal batch
```

The implementation called `delete_batch`, under a comment asserting that "Dataproc Serverless has no
separate cancel — deleting a running batch stops it". The comment was half true and the code was
wrong in the half that mattered. There is no `cancel_batch` RPC — `BatchControllerClient` exposes
only create/get/list/delete, and REST `.../batches/{id}:cancel` is a 404 — but what `gcloud dataproc
batches cancel` actually does, visible under `--log-http`, is `POST .../operations/{id}:cancel`
against the long-running operation the batch names in its own `operation` field. So the verb whose
entire purpose is stopping in-flight work had never been able to stop any of it on this runtime, and
deleting would have destroyed the telemetry the registry reads even where it did apply.

No offline test caught it because the fake batch client's `delete_batch` always succeeded — the test
pinned that a delete was *issued*, which was exactly the wrong assertion. The replacement asserts the
operation is cancelled *and* that the batch is not deleted, plus the terminal-batch and
not-yet-assigned-operation edges. Live, on the next run: `1 of 1 in-flight job(s) stopped`, both
batches confirmed `CANCELLED` by `describe`.

**The second was that `--retry` could not launch at all.** `retry_run` takes `settings` optionally
and the CLI never passes it. Every read on the way to the plan tolerates `None`, because the registry
helpers resolve settings themselves — so the preview was flawless and the submit handed `None` all
the way down to `job_launch.launch_family_job`, which died on `settings.region`:

```
  statistical_repair FAILED to launch: 'NoneType' object has no attribute 'region'
```

The three sibling verbs — `cancel_run`, `settle_run`, `reconcile` — all resolve settings on their
first line. `retry_run` was the one that did not. The bug lived entirely in the gap between a preview
that resolves lazily and a submit that does not, which is why no preview test could see it; the
regression test now asserts the launcher receives a resolved `Settings`, and fails with
`the launcher was handed None` against the old code.

**What the two rungs prove, and why it took two.** The obvious induction — cancel a family mid-flight
— produces the one shape v1 deliberately cannot repair. `repair_demo` lost `statistical` after it had
written 1,968 theta and 1,998 holtwinters cells of 3,000 each, and `--retry` classified that exactly
right and then declined:

```
  verdicts: RETRY_AS_IS=2034, SKIP_ALREADY_DONE=6966
  NOT submittable: holtwinters, theta — these models already have landed predictions, and v1
  re-submits a whole model over the whole series universe, so a repair would append a duplicate
  beside every finished cell.
```

That is the no-overlap invariant enforced *above* its own grain: the invariant is per cell, the
submission unit is per model, so any landed prediction blocks the whole model. Correct, and the
refusal is the product working. But it means the submittable case is a model that landed **nothing**,
and stopping a family mid-write never produces one.

`repair_retry_demo` produces one deliberately, by cancelling during provisioning — before any Python
runs — and it carries two different inductions on purpose:

| family | how it was stopped | `run_jobs.status` | retry verdict |
|--------|--------------------|-------------------|---------------|
| `ml` | the product's `--cancel --job ml` | `CANCELLED` | `UNKNOWN` × 300 — left alone |
| `statistical` | out-of-band, so no cancellation was ever recorded | `FAILED` | `RETRY_AS_IS` × 600 — repaired |

The split is the design, not an accident of timing. A deliberately cancelled family is not something
a repair may quietly resurrect (`retry_policy.classify_cell`: "someone stopped that work on purpose"),
while a family that simply died is. It also re-proves the sticky-cancellation rule from
[the 2026-09-02 cancel run](#the-cancel-reached-its-job-and-then-the-launcher-overwrote-the-cancellation-with-failed)
under a second runtime: the driver's own unwind wrote `FAILED` for both families, and `ml` stayed
`CANCELLED` anyway because the guard held.

The repair then ran end to end. `statistical_repair` filed **its own** attempt-1 row beside the
original `statistical` attempt-1 `FAILED` row rather than over it, so the record of what went wrong
survives the fix:

| family | attempt | status | batch | runtime |
|--------|---------|--------|-------|---------|
| `ml` | 1 | `CANCELLED` | `…-ml-a1` | 143.8 s |
| `statistical` | 1 | `FAILED` | `…-statistical-a1` | 320.5 s |
| `statistical_repair` | 1 | `COMPLETED` | `…-statistical-repair-a1` | 1,965.7 s |

600 cells landed — 300 theta, 300 holtwinters — against a worklist of exactly 600, with **zero**
duplicate `(ts_id, model_type)` pairs and 8,400 prediction rows per model (300 × 28). The leaderboard
resolves both models at 300 cells. The audit blob landed on the repair's own row carrying the counts
it decided from (`RETRY_AS_IS: 600`, `UNKNOWN: 300`), the models, the empty `blocked` list, the
launching user's identity, and the pinned snapshot the universe was counted at.

**What Tier 5 says about the ladder.** The classifier was never the risk — it was fully covered
offline and it was right both times, including the counts, at the first attempt. Both defects were in
the machinery around it, and both had the same shape: a path that only executes when someone actually
stops or repairs something live. `--cancel` shipped a wrong implementation behind a test that
asserted the wrong verb, and `--retry` shipped a launch path that no preview could reach.

#### 2026-09-11, Tier 6 wave A: `inverse_error` was weighted per batch, not per series

Smokes 11 and 12 are the same configuration run two ways. Barrier mode holds the ensemble until every
member family has finished and then computes once; microbatch mode runs the ensemble node alongside
the members for the whole run, gathering each one as it lands. Both are meant to be a **scheduling**
choice — when the arithmetic happens, not what it computes. Running them back to back caught a place
where that was not true.

The unweighted ensembles agree perfectly. `ensemble_mean` and `ensemble_median` are bit-identical
across the two runs on all 2,800 forecast rows. Both **weighted** ensembles differ on every row.

| ensemble | rows differing | series affected | median relative difference | p95 |
|---|---|---|---|---|
| `ensemble_mean` | 0 / 2,800 | 0 | 0 | 0 |
| `ensemble_median` | 0 / 2,800 | 0 | 0 | 0 |
| `ensemble_inverse_error` | 2,800 / 2,800 | 100 | 0.07% | 3.0% |
| `ensemble_nnls` | 2,800 / 2,800 | 100 | 0.58% | 35.2% |

The inputs really are identical, which is what makes the difference meaningful. All 22,400 member
out-of-fold rows — every `(ts_id, model_type, fold_id, forecast_date)` — join across the two runs
with a maximum absolute difference of exactly zero, and both runs combined the same four members
(`mean` and `median` matching bit-for-bit proves it, and the stored NNLS weights name all four in
both).

**The NNLS half is old news, and known.** `ensemble_run._ensemble_batch` has documented it since
2026-09-02, from this same pair of smokes: a non-negative least squares fit over collinear members
has a solution *face* rather than a unique point, an active-set solver walks to whichever vertex it
reaches first, and microbatch feeds the members in arrival order where barrier feeds them in one
canonical pass. Two vertices of the same optimal face score the same on the holdout, so this is a
deliberately open design question rather than an error, and it stays open.

**The `inverse_error` half was new.** Its weights are a closed-form function of per-model error, so
they have no solver to wander — they should have matched, and they did not. The cause was that the
*future* blend pooled its metrics across series: `1/mean(decision_metric)` computed with a run-wide
`groupby("model_type").mean()` over whatever metric rows the call happened to be given. Microbatch
filters those rows to the batch's series, so the pool differed, so every weight differed, so every
one of the 2,800 shipped forecast rows differed. Its out-of-fold counterpart
(`ensembler._inverse_error_blend`) has always looped per series, so the leaderboard the two runs
produced looked clean while the numbers underneath it disagreed — which is precisely why this
survived to be caught by a re-run rather than by a metric.

The fix (`ensembler._inverse_error_weight_matrix`) estimates those weights per series, from that
series' own metadata, through the same `inverse_error_weights` helper the out-of-fold path uses so
the zero-error and non-finite branches cannot drift apart. A series with no metadata of its own falls
back to uniform, degrading to `mean` — pooled fallback would have quietly reintroduced the batch
dependence. Three unit tests pin it, the load-bearing one being that splitting the series across two
`combine_calculated` calls must return byte-identical rows to one call over all of them; all three
fail against the pooled code.

The axis `ensemble_weighting` was registered the same day at its **broken** value,
`gather-order-dependent`, and then moved to `per-series-calculated+batch-fit-learned` by the fix. The
two-step is deliberate and follows the mechanism this page has used since P0: an axis added at its
old value flips the declaring rows to STALE automatically the moment the value changes, where an axis
added afterwards silently leaves behind every row somebody forgot to hand-mark. It worked as
intended — it re-STALEd smokes 11, 12 and `ensemble_demo` without anyone marking them. The new value
names both halves honestly: the calculated strategies are now partition-invariant, the learned ones
are still fit per batch.

**Proven live the same day.** Both smokes were re-run with `--force` under the fix, as attempt 2 of
the same two `run_id`s, and the table settles it without needing to trust either run's own report.
Every cell now holds two values per run — the pre-fix one and the post-fix one — so the question is
which values the two *modes* share:

| ensemble | distinct values per cell, run 11 | per cell, run 12 | cells where barrier and microbatch share a value |
|---|---|---|---|
| `ensemble_mean` | 1 | 1 | 2,800 / 2,800 |
| `ensemble_median` | 1 | 1 | 2,800 / 2,800 |
| `ensemble_inverse_error` | **2** | **2** | 2,800 / 2,800 |
| `ensemble_nnls` | 1 | 2 | 2,800 / 2,800 |

Two values inside each run is the fix moving the numbers; one value shared across the runs is the
two gather modes agreeing on every cell. `ensemble_nnls` landing on the barrier answer this time is
the arrival order happening to match, not a guarantee — that half is still open, and run 12 having
two values while run 11 has one is exactly what a solver that follows arrival order looks like.

#### 2026-09-11: the ensemble forecast rows had no write timestamp at all

Running that comparison is what turned this up, because the obvious way to separate two attempts of
one `run_id` is `created_at` — and on `forecast_predictions` every ensemble row has a NULL one. Not
some of them: all 5,600 across the two smokes, against zero NULLs on the 22,400 member prediction
rows, the 44,800 ensemble out-of-fold rows and the 1,600 metadata rows in the same runs.

That is a contract this page and the code both state. `forecast_predictions` is read
newest-write-wins (`QUALIFY ROW_NUMBER() … ORDER BY created_at DESC NULLS LAST`), and
`ensemble_run`'s own module docstring said "every ensemble row carries a `created_at`" as the
resolution to the append-only design. The engines stamp their prediction rows; `assemble_ensemble_oof_rows`
stamps the blended OOF; `assemble_metadata_row` stamps the metadata. The one writer that did not was
the inline loop in `_ensemble_batch` that stamps `run_id` and `ensemble_id` onto the blended rows —
`@gcp`-only code no unit test reaches, which is the whole reason it went unnoticed. So a second pass
over any run — a `--force` re-ensemble, a repair, a re-score — left two well-formed rows per cell
with nothing to order them by, and the reader got whichever one the scan returned first.

The fix moves the stamping into `registry.rows.stamp_ensemble_prediction_rows`, one function both
blender paths go through, next to the OOF assembler that always did this correctly. Two tests pin
every column it owns, on both blender shapes. Deliberately **not** a new architecture axis: no
forecast number changes, and no result on this page makes a claim that depends on ensemble re-run
idempotency, so nothing already proven has moved under it. Rows written before today keep their NULL
and lose to any later row — which is what the `NULLS LAST` in every one of those reads is for.

Proven live on wave B rather than by a third re-run of 11 and 12, since smoke 14 carries an ensemble
anyway. In `smoke-14-full-dag-2cef0feb95da`, all **11,200** ensemble prediction rows carry a
`created_at` and **none is NULL**, alongside 16,800 member rows that were already stamped. Every
ensemble row shares one timestamp, `20:33:01`, which is the intended behaviour and not a rounding
artifact: `_ensemble_batch` captures `datetime.now(UTC)` once when the job starts and reuses it for
every drain, so one ensemble job is one write generation no matter how many microbatches it gathers
in.

`ensemble_demo`, re-run the next day under the weighting fix, then showed the tiebreak doing its
actual job. That config had already run once before the fix, so its `run_id` now holds two attempts:
**560 raw prediction rows per ensemble strategy over 280 cells — 280 with a NULL `created_at` and
280 stamped.** Every cell therefore resolves to the post-fix row, because `NULLS LAST` sorts the old
attempt underneath the new one. Before the fix both attempts would have been NULL and the reader
would have got whichever the scan happened to return first, with no way to tell which.

#### 2026-09-12, Tier 6 wave C: the cluster lifecycle re-proved itself, and the GPU cluster did not

Smokes 04 and 05 both passed, both matching their pre-registered `run_id`s. What they are worth
running for is the **lifecycle asymmetry** — an ephemeral cluster must be deleted when its run ends,
a named one must not — and both halves held again. Right after 04 finished,
`sf-cluster-smoke-04-cluster-cpu-9196365250ac` was already `NOT_FOUND`; right after 05 finished, both
of its family jobs reported `sf-smoke-cluster` as their placement with state `DONE`, and that cluster
was still `RUNNING`. As before, the survival check is made against `clusters list`, outside the
harness, because a reuse path that tore down a cluster it did not create would pass every assertion
the harness makes.

**One correction to how that check has to be read.** `sf-smoke-cluster` disappeared about half an
hour later, which looks alarming and is not: every cluster the product builds carries a
`LifecycleConfig` whose `idle_delete_ttl` defaults to 1800 s, and the deletion landed at 02:27:43
against a last job finishing 01:57:43 — 1800 s to the second. The reuse path did not delete it;
Dataproc reclaimed it. The practical consequence is that **the survival assertion has a
thirty-minute window**, so it has to be made promptly after the run rather than whenever the
campaign next looks.

**Smoke 16 never reached submit, and the reason is worth recording in two parts.**

The proximate cause is external. The GPU cluster's second init action is Google's published
`gpu/install_gpu_driver.sh`, and on both workers it looked for a prebuilt kernel-module tarball for
this image's kernel (`kmod_debian12_550.142.tar.gz` for `6.1.0-52-cloud-amd64`), got a 404, fell
back to compiling the NVIDIA open kernel modules from source, and was killed partway through
`make -j8 modules`. Both workers reported `Initialization action failed` about six and a half
minutes in. Dataproc then held the create operation in `RUNNING` /
`CREATE_VMS_AND_MANAGED_GROUP_DONE` for another forty minutes collecting diagnostics, so the
product's blocking create call had no failure to react to and simply waited. Nothing here is a
product defect, but it does mean **the cluster-GPU path is currently blocked by an upstream cache
miss** — which is precisely the failure the pre-baked driver image was built to avoid, and that
image was reverted by owner decision on 2026-09-09 as a cost call. Smoke 06 is the other config on
this path; its CURRENT row predates the miss.

The second part *is* ours, and only a two-cluster run could have exposed it.
`shared_clusters.shared_spark_cluster` provisions one cluster **per hardware kind, sequentially**,
and submits nothing until every one of them is up. So the CPU cluster came up at 02:01, then sat
completely idle while the GPU cluster tried and failed to provision — and at 02:39, exactly 1800 s
after it was created, Dataproc's own idle TTL reclaimed it. **A cluster that has never run a job is
idle from the moment it exists**, so on any run where a second cluster takes more than the idle TTL
to provision, the first one is destroyed before it is ever used. Under normal conditions the GPU
create finishes in a few minutes and this is invisible; it took a create that hung to make it
reachable. Nothing was lost — no family had been submitted, so there were no job rows, and
`registry.ops.close_runs` closed the orphaned header `RUNNING → FAILED` with the reason "no job rows
— the run never recorded a family". Both clusters and all three VMs were confirmed gone afterwards.

Neither problem has been fixed yet, so smoke 16's row stays STALE against its 2026-09-02 `run_id`.

### `all_families_10k_full` — the last NEVER_RUN config, and it corrected the arithmetic on this page

Ran 2026-09-05, `all-families-10k-full-e68d9341ce01`, `COMPLETED` in **19,035 s (5 h 17 m)**. Same
seven models and four families as `all_families_10k`, plus the two things that row does not cover:
**`backtest.n_folds: 2`** and **`persist_models: true`**.

| Family | Runtime | Wall | Cells |
|---|---|---|---|
| `native` (`arima_plus`, `timesfm`) | BigQuery | 1,084 s | 20,000 |
| `statistical` (3 models) | Ray CPU | 3,025 s | 30,000 |
| `ml` (`xgboost`) | Ray CPU | 3,177 s | 10,000 |
| `deep_learning` (`neuralprophet`) | Ray GPU (12 x T4) | **18,373 s (5 h 06 m)** | 10,000 |

Everything verified: 1,960,000 prediction rows, **1,960,000 distinct** `(ts_id, model_type,
forecast_date)` — a first run of this `run_id`, so raw and distinct agree exactly. `backtest_oof`
holds **560,000 rows per model for all seven**, folds 1 and 2, 10,000 series each: 3,920,000
out-of-fold rows, and every family produced them, BigQuery natives included. `no_artifact_rate` is
**0.0 across all five Python models — 50,000 GCS artifacts, zero misses**, which is the first
`persist_models: true` proof at this scale. The two natives report 1.0 by design: a BQML model lives
in BigQuery and has no GCS ObjectRef. Header attributed to a principal. Teardown REST-verified.

**It found an error in `quota_and_scale.md`'s core formula.** That page said `cells = series x models
x folds` and that 2-fold backtesting "doubles the run". It does not. Each fold is a fit on truncated
history and then the model is fitted *once more* on all of it to produce the shipped forecast, so
`n_folds: 2` is three fits per cell. Measured against `all_families_10k` on the identical config
minus backtesting, the deep-learning family went **6,582 s → 18,373 s, 2.79x**. The formula is now
`fits = series x models x (folds + 1)` with that measurement beside it.

**And it separated the GPU anchor from the GPU pool's real limit.** At 3 fits per cell this run
delivers **8.2 fits/min/T4** against the 7.6 measured the day before — the same number, which
confirms the anchor is per *fit* rather than per cell. But the pool never used the allowance it was
given: `CPU [84.0, 91.0]` and `GPU [8.4, 12.0]`, held flat for five hours. A deep-learning task asks
for one vCPU as well as a GPU fraction, and 12 `n1-standard-8` workers offer only 84 usable cores, so
**84 concurrent fits is the ceiling and 30 % of the T4 allowance is unreachable** — not the memory
defect from the row above, just the machine shape. Raising `ray_gpu_max_nodes` would buy quota that
cannot be fed; the fix is a GPU worker with more vCPUs. Written up in
[quota and scale](quota_and_scale.md).

The run also crossed the bearer-token TTL **four times** (06:12, 07:42, 08:28, 09:13) and the Ray
jobs client refreshed itself each time without a 401 — a five-hour unattended run is the strongest
evidence yet for that mechanism.

**The config was edited on 2026-09-09, so this row is now stale for a second, separate reason.**
Everything above describes `gpu_fraction: "auto"`, and the config now pins `0.125` and
`profile.source: "baseline"`. The reason is the GPU-utilisation finding: `auto` sizes the fraction
from NeuralProphet's measured peak device memory, that peak is about 75 KB, and `_clamp_fraction`
therefore lands on `_MIN_FRACTION` — ten cells packed onto a card, which is a packing decision
derived from a model that is not really using the accelerator at all. This config is the GPU arm of
the CPU-vs-GPU A/B, and its CPU twin gets eight cells per eight-core node, so an unpinned GPU arm
would start with a 25 % concurrency advantage that has nothing to do with the accelerator. Pinning
`0.125` makes both arms report `slots_per_unit == 8`. `profile.source` is excluded from the digest,
so it does not appear in any `run_id` and has to be recorded here instead; `baseline` is pinned
because the shipped baseline carries no deep-learning family, which means it cannot resolve a
`slot_cores` that would halve one arm's concurrency and not the other's. The `gpu_fraction` change
does move the id. Nothing above is retracted — it describes a run that happened — but the config
that produced it is no longer the config in the tree.

**On 2026-09-02 the whole Ray track stopped provisioning, and the elimination is the useful part.**
`ray_100k` was attempted and never reached a job: Vertex returned the contentless
`"An internal error occurred on your cluster. Please try recreating one in a few minutes."` in
`us-central1`, and the two failover regions returned the missing-`networkAttachment` error recorded
below. Seven creation attempts across five different cluster specs all failed identically:

| Attempt | Head | CPU max nodes | Result |
|---------|------|---------------|--------|
| `ray_100k` as shipped | `n1-highmem-32` | 20 | internal error |
| reduced fleet | `n1-highmem-32` | 10 | internal error |
| default head | `n1-standard-16` | 20 | internal error |
| **exactly `ray_autoscale_demo`'s compute block** | `n1-standard-16` | 8 | internal error |
| **exactly smoke 07's compute block** | default (`n1-standard-16`) | 5 | internal error |
| the same block again, hours later, fresh cluster name | default | 5 | internal error |
| the same block a third time, ~3 h later, fresh name, autoscale off | default | 5 | internal error, 171 s |

Every attempt ran a head node of `n1-standard-16` or larger — `n1-standard-16` is the shipped
default (`config.ComputeConfig.ray_head_machine_type`) and two attempts exceeded it — so an
undersized head is not the explanation.

The last three are the ones that matter: **that compute block provisioned successfully on 2026-09-01
and does not provision now**, with nothing changed between them but the day. That eliminates the
config, the head machine type, the fleet size, and the autoscaling spec in one pass, and it
eliminates quota too — the 10-node attempt asked for 112 vCPU against 180 available. What is left is
the environment.

**The failure is at create, not at submit**, and that distinction is worth keeping straight because
Ray has failed at submit before on this project (the dashboard-handshake 524, where the cluster came
up fine). Here `vertex_ray.create_ray_cluster` itself returns the error and no job is ever
submitted.

Recorded here rather than as a status change, because **no row above becomes false**: the Ray rows
were proven on infrastructure that worked, and an outage is not an architecture axis moving. What it
does mean is operational — **the Ray half of this campaign is stalled**, which blocks `ray_100k`,
the sizing half of the profiler A/B (the profiler is only wired on the Ray path), and any notebook
that provisions a cluster. Retry before concluding anything about Ray from this date.

The sixth and seventh attempts are the ones that set expectations. The platform's own advice is
*"please try recreating one in a few minutes"*; that was taken literally — twice, hours apart,
each with a cluster name that had never been used — and the failure came back byte-identical both
times, spanning most of a working day. So it outlasts the retry the error message asks for by a wide
margin, and at that point a support case looked like the next step rather than another probe.

**It cleared on its own at ~17:00 UTC the same day.** See "The Ray outage resolved itself, and the
fix we nearly shipped for it" below — the retry advice was right in the end, just off by about six
hours, and the draft support case was withdrawn unfiled.

The seventh attempt did tear itself down cleanly (verified by `describe` returning `NOT_FOUND`, not
by trusting the teardown log line — see the leak below), so the outage does not leak a cluster
*every* time. It leaks intermittently, which is worse: a config that fails and leaks is
unretryable, and a config that fails and cleans up looks the same in the logs.

It also retro-explains the `us-central1` leg of the region-failover finding below, which had been
left as "the opaque one". It was the same outage, one day early.

**A failed provision leaks a cluster, and because the name is derived from the `run_id`, the same
config can then never be retried.** This was found by running into it, not by reading the code. After
the fifth failed attempt the product logged its ordinary success line —

    deleted ephemeral Ray cluster …/persistentResources/sf-ray-wave10-ray-availability-probe-b352a2a2cb54

— and the resource was still there, in `PROVISIONING`, fifty minutes later. Retrying that config did
not create a second cluster; it failed outright:

    AlreadyExists('There is an existing PersistentResource with the same ID
    "sf-ray-wave10-ray-availability-probe-b352a2a2cb54" created or being created.
    Please use a different ID.')

Deleting it by hand was refused for the same reason the product's own teardown could not take
effect:

    FAILED_PRECONDITION: PersistentResource "…/sf-ray-wave10-ray-availability-probe-b352a2a2cb54"
    is being created thus can not be deleted now. Please try again later after it's active.

So there is a window — a resource that failed to come up but has not yet been marked failed — in
which Vertex will accept neither a create nor a delete for that name. It closed on its own: the
state moved `PROVISIONING` → `ERROR`, and a delete against the `ERROR`-state resource was accepted
immediately and left the region clean.

Two things to carry from this, kept separate because they are not equally certain. **Certain:** a
run whose cluster fails to provision can leave a resource behind that blocks every retry of that
same config until someone removes it by hand, and `gcloud ai persistent-resources list` reports `[]`
even while it exists (use `describe`), so the thing blocking you is invisible from the obvious
command. **Not established:** exactly why the teardown reported success. `_delete_cluster` logs at
`info` only on the no-exception path and downgrades any failure to a `warning`, so the success line
means the SDK call returned without raising while the resource stayed `PROVISIONING` — but whether
the SDK swallowed a rejection or Vertex accepted a delete it then did not perform was not
determined, and this outage is the wrong conditions to determine it in.

**The sixth attempt did not leak, and that is the more useful half of the finding.** Same code, same
region, same failure — and `describe` on its cluster name returns `NOT_FOUND`. So the leak is *not*
unconditional: teardown works sometimes and silently fails other times, which is exactly the shape
that makes it dangerous. It is a race, not a broken code path, and it will not reproduce on demand.
Two details from that run sharpen where the race lives. The product's teardown logged success two
seconds after the provisioning error, far too fast to have waited on anything. And the `vertex_ray`
SDK prints its *own* `Successfully deleted the cluster` line — so **two independent deleters run
against the same resource**, and because one writes to stdout and the other to stderr, their real
order is not recoverable from a redirected log. A second teardown arriving while the first is in
flight is a plausible way to produce a `PROVISIONING` resource that both parties believe they
removed, but it is a hypothesis; nothing here tests it.

The practical consequence is unchanged and worth stating plainly: **you cannot tell from the logs
whether a failed Ray run left a cluster behind.** The success line is not evidence. Check with
`describe`.

The operational recovery, if a config starts failing with `AlreadyExists`:

    gcloud ai persistent-resources describe sf-ray-<run_id> --region=<region>   # list shows []
    gcloud ai persistent-resources delete   sf-ray-<run_id> --region=<region>   # wait out PROVISIONING

### The Ray outage resolved itself, and the fix we nearly shipped for it

**Resolved 2026-09-02 ~17:00 UTC. No code change. No support case.** The Ray track is unblocked.

The recovery was spotted by accident. A Console-created cluster (`cluster-20260902-120337`) came up
`RUNNING` at 16:26 UTC with the same project, region, PSC-I attachment, service account,
`ray-cpu.2-47.py311` image and machine types our client had failed on seven times. Read at the time,
that said the *service* was healthy and the fault was in what our client sends — and diffing the
Console's resource against our payload left exactly two differences: `boot_disk_type` (`pd-standard`
vs the SDK dataclass default `pd-ssd`, which we inherit without setting) and worker count (2 vs 5).

Bisected with `create_ray_cluster` called directly, everything held still but the field under test:

| Arm | Boot disk | Workers | Result |
|-----|-----------|---------|--------|
| A | `pd-standard` | 5 | **PROVISIONED**, 816 s |
| B | `pd-ssd` | 2 | **PROVISIONED**, 696 s |
| C | `pd-ssd` | **5** — the exact spec that failed 7× | **PROVISIONED**, 666 s |

Arm C is the finding. The identical configuration that failed seven consecutive times at ~171 s
provisioned normally about two hours later with nothing changed on our side. **The failure was
transient and service-side** — not the disk type, not the fleet size, not our payload. All three
arms tore down clean, verified by `describe` returning `NOT_FOUND` rather than by the SDK's own
success line, per the leak finding above.

**Arm A alone would have shipped the wrong fix, and the reason is worth more than the outage.** It
passed first, and `pd-ssd` then explained every fact available: a contentless error (tenant-side SSD
capacity is invisible to us — our own `SSD_TOTAL_GB` reads 0 used of 20480 and is not the binding
quota), a fast pre-flight-shaped failure rather than a provisioning timeout, and a regression
appearing overnight as other tenants' usage grew. A `pd-standard` pin was written into
`ray_cluster.py` with a helper and five tests before arms B and C reversed it. All of it was
reverted; `boot_disk_type` is back to the SDK default, which is what has always been proven live.

The structural error: **arm A changed the hypothesis *and* let two hours pass.** Against an
intermittent fault those are confounded, and "it recovered" is always the competing explanation —
the one that needs its own arm. Re-running the *original failing configuration* is that arm. It cost
25 minutes here and inverted the conclusion. Run it before shipping a fix, not after.

One thing was kept, unrelated to Ray but surfaced by it: the `_fake_vertex_ray` fixture in
`tests/unit/test_ray_submit.py` patched only `sys.modules`, so `from google.cloud.aiplatform import
vertex_ray` bypassed the double as soon as anything else in the session imported the real lazy
submodule — two tests failed in a full run while passing in isolation. Both bindings are patched
now. Same theme as the rest of this file: a guard whose correctness depends on conditions nobody
checks is indistinguishable from one that works.

**Four demonstration configs were held back by the Ray GPU blocker, which turned out to be an
outage rather than an entitlement** (see smoke 08 above for how that was settled). `ray_gpu_demo`,
`per_family_runtimes_demo`, `all_families_10k` and `all_families_10k_full` all put a family on
Vertex Ray GPU. None of them was ever re-pointed at Serverless to get a green row, and that
restraint is the reason the eventual rows mean anything: the whole point of
`per_family_runtimes_demo` is the *split*, and a version of it that ran everything on Spark would
have proved something else while keeping the name.

**`ray_gpu_demo` is also the only live proof of autoscaling on a GPU pool.** It ships
`ray_autoscale: true` with `ray_gpu_min_nodes: 1` / `ray_gpu_max_nodes: 2`, so the T4 pool it
provisions is elastic, not fixed — `ray_autoscale_demo` proves the same mechanism only on CPU. Its
four models ranked on backtested WAPE across two runtimes and two families: `timesfm` 0.279 and
`arima_plus` 0.280 from BigQuery, `neuralprophet` 0.290 from the T4 pool, `theta` 0.339 from the Ray
CPU pool. Provisioning took 9m26s.

**`per_family_runtimes_demo` is the three-runtime split under one `run_id`, and it ran as authored.**
Four family jobs, three runtimes: `statistical` (`theta`, `holtwinters`) and `ml` (`xgboost`) as
Dataproc Serverless batches, `deep_learning` (`neuralprophet`) on a Vertex Ray T4 pool, `native`
(`arima_plus`) as a BigQuery job — all four COMPLETED, 50 cells each, one `run_id`, one reverse
trace naming all four system job ids. The config asks for `hardware: "gpu"` with no `gpu_type`, so
the T4 in the trace is the default resolving correctly rather than a value copied from the config.
This is the row the restraint above was protecting: the split is the claim, and it is now the thing
that was proven.

**Re-run 2026-09-10 as `per-family-runtimes-demo-8fe8f224a7e1`, and it held under a much-changed
architecture** — per-family GPU routing, three-way fleet sizing, the trainer-root-device probe and
the new `run_id` inputs have all landed since the first pass. Same four jobs, same three runtimes,
same 250 cells: `native` home in 35 s, `ml` in 1509 s, `statistical` in 1558 s, `deep_learning` in
1636 s. The Ray pool was torn down and the persistent-resource list read back empty. **Read the
leaderboard's empty WAPE column correctly:** this config authors no `backtest` block, so its 250
cells are forecasts into the future with nothing to score against. The row proves routing and
identity, not accuracy — `mixed_demo` and `ensemble_demo` are the rows that carry scored numbers.
The `deep_learning` job again stamped `ENGAGED_IDLE` at 62,464 peak bytes on a T4, which is the
known "the GPU is attached and has nothing to do at these hyperparameters" finding, not a new one.

The three Spark demo rows landed together on 2026-09-01, and two of them are worth reading past the
`CURRENT`:

- **`mixed_demo` is the cross-runtime comparability claim, live.** One `run_id`, one leaderboard,
  `theta` from a Dataproc Serverless batch ranked against `arima_plus` and `timesfm` from a BigQuery
  job on backtested WAPE (0.451 / 0.418 / 0.391). Two runtimes, one ranking, no manual join.
- **`ensemble_demo` adds the ensemble node as a third job** (BigQuery), and its three strategies
  rank *inside* the same board: `inverse_error` 0.408, `mean` 0.411, `median` 0.413 — all three
  beating both `arima_plus` and `theta`, none beating `timesfm` at 0.391. Recorded as-is. The claim
  the product makes is that ensembles are produced, ranked and comparable, not that they win.

**Both were re-run on 2026-09-10 and every number came back bit-identical.** `mixed-demo-db2dfb2f675d`
and `ensemble-demo-b2ff15a4d418` reproduced the figures above to full float precision — `timesfm`
0.39154588800396195, `arima_plus` 0.4183748578576073, `theta` 0.4510803985450681, and the three
ensembles at 0.4084785200757231 / 0.41166952120260025 / 0.4133315235193376. That deserves a second
look rather than a victory lap, because `backtest_scoring` moved *twice* in between: 3.2 reserved
the newest fold on 2026-09-08, and Phase 6 changed fold geometry and refit on 2026-09-09/10. An axis
that moves without moving a number is either a strong result or a change that never reached the
config, and the two are worth telling apart.

**It is the first.** These configs author no `gap`, and a `gap` of zero reproduces the pre-embargo
fold layout exactly — that is not an inference, it is pinned literally by
`golden_panel_prebreak.json`, which records `[fold_id, train_start, train_end, val_start, val_end]`
for all nine shipped backtesting configs from pre-break code and is compared on every offline gate.
The refit half checks out too: `backtest_refit` reads `per_fold` on all thirty cells, so `auto`
resolved to what these three models were already doing. Same folds and same refit give the same
arithmetic, and the run reproduced it to the last digit across nine days and two architecture moves.
The `run_id` did change — Phase 6 broke the config digest — which is exactly the distinction the
axes are for: the identity moved, the measurement did not.

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

### `ray_100k` reached a cluster and then did nothing at all for an hour, and the number in the error was ours

**2026-09-03. `ray-100k-dcc77a9d1e9b`, cancelled after 58 minutes with zero cells written.** The
cluster provisioned. The job submitted. The driver started. And then nothing — no cells, no
failures, no progress, and a job that would have sat there until the four-hour TTL killed it.

The autoscaler was saying why the whole time, once a second:

    Error: No available node types can fulfill resource request
    {'CPU': 1.0, 'memory': 22548578304.0}.
    Add suitable node types to this cluster to resolve this issue.

22,548,578,304 bytes is not an arbitrary number. It is exactly `0.7 × 30 GiB` — our own
`_SCHEDULABLE_MEMORY_FRACTION` applied to the nameplate RAM of an `n1-standard-8`, which is the
worker type this run asked for. **The product computed a per-task memory request equal to the
largest amount it believed a node could offer, handed it to Ray, and Ray could not place it.** An
unplaceable Ray task does not fail; it queues. So the run's failure mode was silence.

The irony is on the record in the code. `resources/slot.py` clamps a slot to its unit precisely
because *"a task asking for more cores than any node has is not slow, it is unschedulable, and Ray
will sit on it forever rather than fail. Clamping and recording beats hanging."* The clamp that
exists to prevent the hang produced one.

Two independent defects had to line up, and both are worth separating because they fail differently.

**1. The clamp had no headroom.** `schedulable_memory_bytes` *estimates* the scheduler's ceiling
from a machine type's nameplate RAM. Ray computes its real ceiling from what the container's OS
reports, which is a percent or two below nameplate — so a request sized at exactly our estimate
lands just *above* the real ceiling. Not approximately at it; above it. And clamping to exactly the
ceiling would have been a bad plan even where it was a legal one: one cell per node, seven cores
idle. Fixed by adding `_MAX_SLOT_MEMORY_FRACTION = 0.85` and a separate `max_slot_memory_bytes()`
used at the three sites that clamp a slot *to fit*; the sites that *divide up* a node's capacity
still use the schedulable figure, because that is a different question.

**2. The evidence was the wrong kind of measurement.** The ~21 GiB came from `slot_rss_bytes`,
harvested from `explode-100k-1c59265062aa` — a **Spark** run. `process_rss_bytes` is the absolute
footprint of the process that ran the cell: on Ray that process is one task, on Spark it is an
executor running many cells concurrently. Reading a Spark number as a Ray per-task bound is not an
over-estimate, it is a measurement of something else. `rank_harvest_candidates` now ranks runtime
comparability *above* scale: all-target-runtime beats mixed (or unrecorded), which beats none.

**The fix for the previous finding is what exposed this one.** The scale-ranking fix landed hours
earlier did exactly its job — it preferred a 100k-series Spark harvest over a 1k-series Ray one,
which on the scale axis is unambiguously the better evidence. That correct choice on one axis
surfaced a latent category error on an axis nobody had thought to rank. The lesson is not that the
scale fix was wrong; it is that **the profile's provenance had one axis and needed two**, and a
one-axis ranker had been silently getting the right answer only because the corpus was small.

Both fixes are offline-proven at `17e1221` (2406 passed, 2 skipped) and include a regression test
that pins the property directly: a slot may never be clamped to a node's entire schedulable memory.

The cancellation itself is a small good-news footnote: `--cancel --force` on a hung two-family run
tore it down cleanly — header `CANCELLED`, both job rows `CANCELLED`, cluster gone — which is the
second live proof of the sticky-cancel guard and the first on a multi-family run.

**Re-run the same day, and it completed.** `ray-100k-dcc77a9d1e9b` attempt 2, `COMPLETED`, **400,000
cells** — `theta`, `holtwinters`, `sarimax` and `xgboost` at 100,000 each, one `run_id`, two family
jobs on one shared Ray cluster. Cluster provisioning 15 min; `statistical` 317.8 min, `ml` 294.4
min; **330.3 min end to end**. Teardown verified by `describe` returning 404, not by the SDK's
success line. Throughput held flat at ~1,450 cells/min for five hours with no stalls, and the
`~60-minute bearer-token TTL` never surfaced — the proactive refresh in `ray_jobs` carried a
five-and-a-half-hour poll, which is by a wide margin the longest run this product has completed.

What the run proves, in the order it was in doubt:

1. **The headroom fix works.** Zero clamps fired. `statistical` got 2 cores and 1.29 GiB per task
   against an `n1-standard-8`'s 21 GiB schedulable — nowhere near the ceiling, because the evidence
   was finally the right kind of evidence.
2. **The runtime-ranked harvest works.** `auto` resolved to `ray-autoscale-demo-886a053c374c`, a
   **Ray** run at 10,000 series, in preference to `explode-100k-1c59265062aa`, a Spark run at
   100,000 — the exact scale match that hung the previous attempt. Runtime beat scale, which is the
   whole of the fix.
3. **Ray autoscaling reached its ceiling under real load.** The worker pool went 1 → **20**, its
   configured `ray_cpu_max_nodes`, and stayed there. `saturating_units` records what the arithmetic
   actually wanted: **12,500**. The run was throttled by the ceiling, not by the work, and the
   record says so.
4. **A Ray fleet was sized from a measurement for the first time.** `basis: measured` on the
   `statistical` plan — 4 slots per node from a harvested 1.29 GiB `slot_rss_bytes`, versus the 8
   the static path would have assumed.

One honest qualification on point 4. **The `ml` family fell back to `basis: static`**, because
`ray-autoscale-demo` measured three statistical models and no `xgboost` — so the best-ranked Ray
harvest was a *partial* match, and the profile had nothing to say about the family that turned out
to be the faster of the two. The ranker chose correctly on the axes it has; "does this harvest cover
the families I am about to run" is a fourth axis, unranked, and it cost nothing here only because
the static fallback for `ml` was adequate. Recorded, not fixed.

And this run is itself the artifact W13 was waiting for: **the first 100,000-series Ray harvest**,
measured across 400,000 fits, which is what a shipped baseline profile should be cut from rather
than the 10,000-series demo run it would have had to use yesterday.

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
notebook reflects the current path. **Seven were re-executed on 2026-09-02** against current code and
re-committed with their new outputs, clearing the last `STALE` row in this table. The executed
notebooks were diffed against the committed ones first: source cells were byte-identical in all
seven, so only outputs changed.

| Notebook | Status | Date | Axes at proof |
|----------|--------|------|---------------|
| `01_spark_via_connect.ipynb` | STALE | 2026-09-02 | `serverless_deps=container-image`, `python=3.11`, `horizon_features=computed-at-future-dates`, `run_id_inputs=authored-config-only` |
| `02_bigquery_native.ipynb` | STALE | 2026-09-02 | `python=3.11`, `run_id_inputs=authored-config-only` |
| `03_combo_and_ensemble.ipynb` | STALE | 2026-09-02 | `serverless_deps=container-image`, `python=3.11`, `fleet_sizing=derived-overlay`, `run_id_inputs=authored-config-only` |
| `04_ray_on_vertex.ipynb` | STALE | 2026-08-28 | `ray_deps=stock-image+uv-runtime-env`, `python=3.11`, `fleet_sizing=derived-overlay`, `run_id_inputs=authored-config-only` |
| `07_scale_review.ipynb` | STALE | 2026-09-02 | `python=3.11`, `run_id_inputs=authored-config-only` |
| `08_run_and_monitor.ipynb` | STALE | 2026-09-02 | `serverless_deps=container-image`, `python=3.11`, `fleet_sizing=derived-overlay`, `run_id_inputs=authored-config-only` |
| `09_review_run.ipynb` | STALE | 2026-09-02 | `python=3.11`, `run_id_inputs=authored-config-only` |
| `model_playground.ipynb` | STALE | 2026-09-02 | `python=3.11`, `run_id_inputs=authored-config-only` |

**`03` took three attempts, and the two failures had two different causes.** Neither was a defect in
the notebook — it reported **zero cell errors** every time.

The first attempt died before it started: `Quota 'CPUS' exceeded. Limit: 200.0 in region
us-central1`, because six sibling notebooks held the region's Colab runtimes at that moment. The
region measured 36/200 once the wave drained, so this was contention inside the harness's own
fan-out, not a standing shortage. Worth recording for a second reason: **Vertex names a quota
failure explicitly when that is what happened**, which independently strengthens the elimination
above — the Ray outage's contentless "internal error" really was not quota.

The second attempt, re-run alone against a quiet region, hit `Job deadline exceeded` at its 1800 s
ceiling. **The work had actually succeeded** — Dataproc batch
`sf-nb03-combo-ensemble-1788329058-c4a5e6db54a1-statistical-a1` reports `SUCCEEDED` — but the
notebook process was killed while still waiting, so the run header is stranded at `RUNNING`
forever. That is the failure mode that matters: `03` and `08` both block on a Dataproc Serverless
batch, which carries ~30 min of fixed provisioning overhead before any work runs, so a 30-minute
ceiling gave them close to zero margin, and **the run's finalizer lives in the notebook process** —
a deadline kill lands after the batch succeeds and before the header closes. Both timeouts are now
3600 s, matching `01`. A ceiling is not a duration, so the headroom costs nothing unless it is
needed, and what it prevents needs a human to clean up.

The third attempt — same notebook, same region, alone, changing nothing but the ceiling — passed,
and took roughly the 30 minutes the old limit allowed. **That is the confirmation of the diagnosis,
not just the fix for it:** if the deadline had been a symptom rather than the cause, widening it
would have produced a longer failure instead of a pass. It also exercised the new `--only` selection
path end to end, which is what made a one-notebook retry affordable enough to run three times.

## Other capabilities

| Capability | Status | Evidence |
|------------|--------|----------|
| Workshop Act 1 (100k history, Cloud Shell / VM) | NEVER_RUN | The four scale configs above, run as the workshop instructs them and in that order. Not the same claim as "the configs work": Act 1 is followed by someone who has just deployed, from a shell with a session limit, and its failure modes are disk, quota and session death. **Every command and SQL block in the act was walked verbatim on 2026-09-02, and three of them were wrong** — `--dry-run` printed nothing at all, the runbook told you to look for a `SUCCEEDED` run status the registry never emits, and it advertised a `--wait-timeout` flag the documented entrypoint rejects. All fixed; see below. The SQL all ran as written. Still `NEVER_RUN` because the act's real failure modes — disk, quota, session death on a fresh deploy — are untouched by walking it from a working environment. That is the whole argument for running the act rather than the configs. |
| Workshop Act 2 (pre-rendered notebook tour) | NEVER_RUN | Headless execution of the tour notebooks against a fresh deployment. The notebook rows above were proven by the acceptance harness, which is not the same path. Its documented `--tier` table was walked on 2026-09-02 and was two notebooks out of date (3/5/6 against the registry's 4/7/8); corrected. |
| Workshop Act 3 (live Colab Enterprise tour) | NEVER_RUN | The tour notebooks opened and run interactively on the `sf-main` runtime, reading Act 1's runs. The tour table listed six on 2026-09-02 while Act 2 pre-rendered eight; `08_run_and_monitor` and `09_review_run` were added, so the count is now eight. |
| Run-inspection layer (`review.py`) | CURRENT | Exercised live through notebooks 08 + 09 at `ff1f8bf`. Its `@gcp` registry readers ran against a real deployment. |
| Airflow DAG emitter (`airflow_emit`) | CURRENT | Smoke 15, 2026-09-03: an emitted `dag_<run_id>.py` was parsed by a real Composer 3 / Airflow 2.10.5 scheduler (`has_import_errors: false`) and orchestrated a five-family run across Serverless Spark, Ray-on-Vertex and BigQuery to `COMPLETED`. The Airflow-produced `run_id` equalled the locally-resolved one — the same-code local↔Composer claim. **Narrowed 2026-09-05:** the Ray family in that run asked for a T4 per-family and got no device (see the `dl_gpu_routing` section), so what Airflow is proven to orchestrate is Ray-on-Vertex, not Ray-on-Vertex *GPU*. Orchestration is what this row claims and orchestration held. See below. |
| RuntimeProbe read path (P1–P4) | CURRENT | First live probe 2026-09-02 against `wave-62-mixed-runtimes-cpu-a7d04b6a9c8e` mid-flight: correct `TRUST_REGISTRY` + done/expected for the three terminal families, and a correct refusal on the running Ray one. `RayProbe.check` was then driven live out-of-process against that job and returned `RUNNING` — after the missing `_init_vertex` was fixed. The handle fix then landed and was re-proven live the same day: `--probe` against `ray-dl-on-cpu-probe-2e8a9f3f5c8d`, a single-family ephemeral Ray run, escalated out-of-process and returned `RUNNING_CONFIRMED`. Scope is now the whole verb, on every runtime. |
| RuntimeProbe cancel (P5) | CURRENT | A real Ray job was stopped live 2026-09-02 (`RayProbe.cancel` → `stopped: True`, job reached `STOPPED`, the run's own poll loop saw it and unwound). The **data-integrity property is proven by a genuine failure**: when `--cancel --force` could not reach that family, the registry was *not* marked CANCELLED. **Re-run live 2026-09-02 against a purpose-built single-family Ray job and the verb now reaches it** (`deep_learning cancelled — ray job stop issued`, count line correct at `1 of 1`, launcher unwound and tore the cluster down, teardown REST-verified). **New scope: the cancellation does not survive.** The launcher finalized the run `FAILED` 17 s after the cancel wrote `header=CANCELLED`, so the registry cannot distinguish a deliberate stop from a crash. See below. |
| Custom IAM roles (P6) | CURRENT | Applied live 2026-09-01: `projects/statmike-scale-forecasting/roles/sfProbeReader` and `roles/sfJobCanceller` now exist. Until then they had only ever been `validate`-clean. Creation is not use — that the permission sets are *sufficient* for a probe or a cancel is the P1–P5 rows below, not this one. |
| Registry ops (`registry.ops`) | CURRENT | All six `@gcp` tests in `tests/integration/test_registry_ops_live.py` pass 2026-09-02 — artifact-prefix delete correctly scoped in real GCS, `CREATE SNAPSHOT TABLE` valid against the real schema (native `JSON` columns included), `doctor`, `drop_run` preview, `drop_run` execute across every tier. One of the six had rotted and had to be repaired first — see below. **Scope: six of the seven verbs.** |
| Registry ops — `close_runs` (7th verb) | CURRENT | Executed live 2026-09-02 against the real registry: closed 9 of the 10 stuck headers to `FAILED` and skipped the tenth with its reason, leaving `doctor` reporting exactly one in-flight run. **The first live call failed** on a column that does not exist, which no offline test could have caught — see below. |
| Shipped baseline profile (`profiling.baseline`) | CURRENT | The numbers committed in `src/scale_forecasting/profiling/baseline.py` were harvested on 2026-09-03 from `ray-100k-dcc77a9d1e9b` — the `ray_100k` row above, a real 100,000-series Ray run — through the ordinary `read_compute_harvest` path. **This row is a claim about the numbers' provenance and nothing else.** No run has yet been *sized* from the baseline on live infrastructure; that needs a deployment with an empty registry, which this project no longer is. See below. |
| Run audit principal (P6) | CURRENT | The `actor=None` on 2026-09-02's live cancel was a **defect**, resolved 2026-09-04: the userinfo lookup was sending the ADC quota project as `x-goog-user-project` and getting a 403 for `serviceusage.services.use` on a project unrelated to the run. Fixed by stripping the quota project (`identity._without_quota_project`) and verified live under the same ADC credential — `resolve_principal()` returns the user's email. Proven end-to-end on a real run 2026-09-05: `ray-100k-dcc77a9d1e9b`'s header row carries `user_id = <the launching user email>`, where its three pre-fix attempts are blank. That is the *launch* path; cancel-with-attribution has not been re-exercised live since the fix, though the audit *write* was already proven on 2026-09-02. See below. |

### The probe's first live run found that its Ray escalation cannot reach a single-family Ray run

`--probe` was pointed at `wave-62-mixed-runtimes-cpu-a7d04b6a9c8e` while its Ray family was still
running — the exact situation the verb exists for. It printed:

```
run wave-62-...  status=RUNNING  escalated=True  disagreement=False
  statistical    spark    COMPLETED  -        TRUST_REGISTRY  100/100
  ml             spark    COMPLETED  -        TRUST_REGISTRY  100/100
  deep_learning  ray      RUNNING    UNKNOWN  UNKNOWN         0/100   handle missing resource_name
  native         bigquery COMPLETED  -        TRUST_REGISTRY  200/200
```

Three of four families are exactly right, and the fourth is a **well-behaved refusal** rather than a
wrong answer: the probe said it could not tell, and said why. That distinction is the design working
— but the family it could not read is the only one it was asked to escalate to.

**The cause is an ordering problem, not a bug in the probe.** `job_launch` writes an *entry* handle
before submitting, and populates `resource_name` (the Ray cluster path) only when a shared cluster
name is already known. `shared_clusters.shared_ray_inputs` returns `None` for a run with **one**
ephemeral Ray family, so nothing is provisioned up front — the submitter creates the cluster itself.
The corrected handle does get stamped back with the real resource path, but only after `launch`
returns, and `launch` blocks until the Ray job is terminal. So for the whole window in which a probe
is useful, the handle is incomplete; by the time it is complete, the registry alone would answer.

**Which means the Ray probe works only for the shapes that need it least** — a run with two or more
ephemeral Ray families, or one reusing a standing `compute.ray_cluster_name`. A single-family Ray
run, which is the common shape and was the shape here, is unreachable. Offline tests could not catch
this: they construct handles directly and never exercise the launch-ordering that decides whether
`resource_name` is present.

**Fixed offline the same day.** The entry handle now derives the cluster path from `ray_io`'s
name rule rather than waiting to be handed one — the name is a pure function of the `run_id`, so it
is knowable before anything exists to be named. The offline test that had pinned the old behaviour
asserted the absence of `resource_name`, which is worth noting: a test can lock in a defect just as
faithfully as a property, and this one did until a live run said otherwise.

**Re-proven live the same day, on the shape that had been unreachable.** A single-family ephemeral
Ray run (`ray-dl-on-cpu-probe-2e8a9f3f5c8d`, `deep_learning` on Ray CPU — one family, no shared
cluster, nothing to be handed a name by) was probed mid-flight:

```
run ray-dl-on-cpu-probe-2e8a9f3f5c8d  status=RUNNING  escalated=True  disagreement=False
  deep_learning  ray  RUNNING  RUNNING  RUNNING_CONFIRMED  0/6  Job is currently running.
```

`RUNNING_CONFIRMED` is the whole fix in one word: the probe left the registry, reached the Ray job,
and got an answer back, in exactly the configuration that previously returned `UNKNOWN  handle
missing resource_name`. The `escalated=True` matters as much as the verdict — it says the escalation
was attempted rather than skipped. Note the probe was run from a **separate process** from the
launcher, so this also re-proves the `_init_vertex` fix below under the conditions that exposed it.

**Behind it sat a second defect, which the first one had been hiding.** `RayProbe.check` calls
`ray_cluster._get_cluster`, and `vertex_ray.get_ray_cluster` takes no project or location — it reads
the Vertex SDK's *global* config. The launching process sets that while creating the cluster, so
in-process everything works; a probe process never does. Isolated live against the same running
cluster, one call, two outcomes:

```
without _init_vertex:  ValueError: Failed in getting the cluster ... MethodNotImplemented (404)
with    _init_vertex:  OK RUNNING
```

The probe's catch-all would have turned that 404 into `UNKNOWN` — a knowable state reported as
unknowable. `RayProbe.check` and `.cancel` now pin the SDK to **the handle's** region first (not
`settings.region`: a cluster may have hopped on a stockout), and two ordering tests assert the pin
precedes the cluster read, since no stubbed return value can show it.

With that fixed, the Ray probe was driven live out-of-process against the still-running job with a
hand-completed handle, and returned `native_state=RUNNING, exists=True, detail="Job is currently
running."` — **so the read path itself is proven; the only thing still standing between `--probe` and
a Ray family is the missing `resource_name`.**

That one was not fixed in the same sitting. The options looked like provisioning the cluster before
the entry handle is written, or stamping the resource path back the moment the submitter has it
instead of at job end — a change to launch ordering rather than a missing call, and worth designing
rather than patching mid-campaign. **The design that landed was neither: the path is *derived*, not
observed** (see "Fixed offline the same day" above), which removes the ordering question instead of
answering it. Proven live on 2026-09-02.

### A cancel that could not reach its job refused to say it had, and that is the property worth having

The deadlocked Ray family above gave the cancel path something no offline test can construct: a real
in-flight job, on a real cluster, that the verb could not actually reach.

The preview was right about everything it could see:

```
run wave-62-mixed-runtimes-cpu-a7d04b6a9c8e  status=RUNNING
  would cancel:  deep_learning (ray, RUNNING)
  unaffected:    statistical, ml, native (already COMPLETED)
  partial results are RETAINED
```

Correct blast radius, correct per-family effect, and an explicit statement of what happens to data
already written. Then `--cancel --force`:

```
deep_learning  NOT cancelled  handle missing resource_name
audit: actor=None  reason=...  header=None
```

**Three things came out of that, and the middle one is the point.**

**It did not lie about the registry.** `run_jobs.deep_learning` stayed `RUNNING` and the run header
stayed `RUNNING`. The product refused to record a cancellation it had not achieved. That is the
cancel data-integrity property from the RuntimeProbe design, and it has now been proven the only way
that really counts — by a cancel that failed. A run marked CANCELLED while its job kept burning a
cluster would have been the worst available outcome, and it is the easy one to write.

**The verb could not reach the family, for the same reason the probe could not** — the entry handle
has no `resource_name` on a single-family ephemeral Ray run. So `--cancel` inherits the gap recorded
above; fixing the handle fixes both. **Re-observed live on 2026-09-02 and it does** — see the next
section, which also found something the reaching gap had been hiding.

**One wording defect.** The summary line read `1 in-flight job(s) stopped` when zero were. The
per-family line immediately above it says `NOT cancelled`, so the output contradicts itself. Cosmetic
in isolation, not cosmetic in an operational verb someone runs when they are trying to stop spend.

The job was then stopped out of band by calling `RayProbe().cancel()` directly with a hand-completed
handle — `stopped: True | already_gone: False | detail: ray job stop issued` — which closes the
mechanism end to end: the Ray job moved to `STOPPED`, the run's own poll loop observed the terminal
state and raised `EngineError`, the harness unwound, and its `finally` tore the cluster down. The
registry finished `deep_learning FAILED` with the other three `COMPLETED`, and
`gcloud ai persistent-resources describe` returned `NOT_FOUND`. **So the stop, the propagation, and
the teardown all work; only the path from the CLI to the handle does not.**

`actor=None` is the P6 finding. `identity.resolve_principal` ran for the first time — live, under
ADC — and returned nothing, so the audit line for a real cancel attempt names no one. It was recorded
as `NEEDS_RECHECK` rather than a defect because it had not been established whether ADC user
credentials are expected to yield a principal here.

**Resolved 2026-09-04, and it was a defect.** Reproduced under the same credential type and traced to
the userinfo call: `AuthorizedSession` attaches the credential's quota project as an
`x-goog-user-project` header, and the endpoint then answers **403 — caller lacks
`serviceusage.services.use`** on that project. The header is the whole problem. Userinfo is an
*identity* endpoint; it needs no project, and the one it was being billed against was simply whatever
ADC happened to point at, which for a laptop is routinely unrelated to the deployment. The credential
was fine, the scopes were fine, and `resolve_principal` swallowed the 403 exactly as designed — best
effort, never raise — so the failure surfaced only as a blank field.

The fix strips the quota project before the call (`identity._without_quota_project`). Verified live
under the same ADC credential that produced the blank: `resolve_principal()` now returns the user's
email. Two properties worth keeping: it is a *copy* of the credential, so nothing else in the process
is affected, and a credential type that cannot strip goes out unchanged rather than failing —
attribution stays advisory and never blocks the operation it annotates.

**And then a real run proved it end to end, unprompted.** The `ray_100k` re-run the next day carries
`user_id = <the launching user email>` on its `run_registry` header row — the first run in this
project's history to be attributed to a person. The three earlier attempts at the same `run_id` are
blank in that column, which makes the four rows a before/after of the fix sitting inside the registry
itself. That is *launch* attribution; the cancel path writes its actor through the same resolver but
has not been re-exercised live since the fix.

**What this generalizes to.** The audit trail failed on an IAM permission in a project that has
nothing to do with the run — the same shape as the Cloud Billing API check that failed against the
ADC quota project during a Cloud Shell deploy. Any call made with ADC that does not *need* a quota
project should not send one.

### The cancel reached its job, and then the launcher overwrote the cancellation with `FAILED`

`ray-cancel-probe-e22e6fe9a830` was built to be a cancel target and nothing else: one Ray CPU family,
200 series, deliberately long. Once the Ray job was submitted, `--cancel` was run from a separate
process. The preview was right:

```
Cancel run ray-cancel-probe-e22e6fe9a830: 1 in-flight job(s) will be stopped; partial results are RETAINED
  deep_learning  ray  RUNNING  will cancel; 0/200 series landed (retained)
Confirm with --force (CLI) / confirm=True (SDK) to stop these jobs.
```

and `--force` **reached the family**, which is the thing that was owed:

```
Cancelled run ray-cancel-probe-e22e6fe9a830: 1 of 1 in-flight job(s) stopped; partial results are RETAINED
actor=None  reason=-  header=CANCELLED
  deep_learning  cancelled      ray job stop issued
```

Three of the four open items on this verb closed at once. `cancelled  ray job stop issued` is the
CLI reaching a single-family ephemeral Ray job, which it could not do before. `1 of 1` is the count
line telling the truth — the earlier `1 in-flight job(s) stopped` when zero were is gone. And the
launching process observed the stop by itself and unwound properly: `EngineError: ray job
sf-ray-cancel-probe-e22e6fe9a830-deep_learning-a1 terminal state STOPPED`, then its `finally`
tore the cluster down — verified by REST, zero persistent resources. **That is the verified-teardown
path proven on the cancel route, which is the harder one**: the unwind happens through an exception,
which is exactly where a teardown gets skipped.

The registry also shows the handle fix directly rather than by inference. `job_telemetry.probe_handle`
was persisted complete at entry:

```json
{"id_kind":"exact","runtime":"ray","region":"us-central1",
 "native_id":"sf-ray-cancel-probe-e22e6fe9a830-deep_learning-a1",
 "resource_name":"projects/…/locations/us-central1/persistentResources/sf-ray-ray-cancel-probe-e22e6fe9a830"}
```

**Then the fourth thing: the run does not end up recorded as cancelled.** The final state is

| where | value |
|-------|-------|
| `run_registry.status` (header) | `FAILED` |
| `run_jobs.deep_learning.status` | `FAILED` |
| `job_telemetry.cancel.cancelled_at` | `2026-09-02T21:38:08Z` |
| `run_jobs.ended_at` | `2026-09-02 21:38:25` |

The cancel wrote `header=CANCELLED` at 21:38:08 and the launching process finalized the run `FAILED`
seventeen seconds later, because from inside that process a job that went `STOPPED` is a job that
died. **So the verb printed an outcome that did not survive a quarter of a minute.** It was not lying
when it printed it, which is a different failure from the one this file worried about, and in some
ways a worse one: the earlier defect refused to record a cancellation it had not achieved, whereas
this one achieves the cancellation, records it, and then loses the record.

What survives is `job_telemetry.cancel` — `cancelled_at`, `native_state_at_cancel: "RUNNING"`,
`n_done_at_cancel: 0`. So the evidence is not destroyed, it is *demoted* into a JSON column. Every
summary surface — the header, `doctor`, any leaderboard-adjacent view that filters on `status` —
shows a run that failed, with no way to tell a deliberate stop from a crash without opening the
telemetry of each job. For the one verb whose entire purpose is deliberate intervention, that is the
wrong default.

**The rule the fix needs is one sentence: a cancellation is sticky against the failure it caused.**
`CANCELLED` is terminal, and a later `FAILED` arriving from the launcher's own unwind of that same
cancellation must not overwrite it. That is stateable, small, and offline-testable once stated — but
no offline test would have found it, because it needs two processes disagreeing about one run, and
the previous cancel could not reach far enough to create the disagreement. **The reaching gap was
hiding it.**

**Fixed offline the same day.** `update_header` and `update_job` take an `unless_status_in` guard
that renders into the WHERE clause, and `registry.lifecycle` passes `("CANCELLED",)` on every
non-green finalize — both the exception path and the clean path, since `main.run` captures family
errors and finalizes through the *clean* exit with a computed `FAILED`/`PARTIAL`. Two details worth
stating, because both were choices:

- **The condition is in the statement, not a read-then-write.** Reading the row first would leave a
  window, and the whole problem is that the other writer is in another process. One conditional
  UPDATE has no window.
- **`PARTIAL` is guarded and `COMPLETED` is not.** A run whose other families finished is finalized
  `PARTIAL`, which erases a cancellation just as completely as `FAILED` does. `COMPLETED` is left
  alone: if the work genuinely finished, the cancellation lost a race it had no claim to win.

**Re-proven live the same day, on a rerun of the same shape.** `ray-cancel-sticky-824b4822945c`, a
single `deep_learning` family on Ray CPU, 200 series. `--cancel` previewed one in-flight job at
`0/200 series landed (retained)`; `--cancel --force` stopped it and wrote `header=CANCELLED` at
23:06:53. The launcher then did exactly what it did before — saw its own Ray job go `STOPPED`,
raised `EngineError: ray job … terminal state STOPPED`, and unwound — and this time the write it
attempted did not land:

| Where | Before the fix | Now |
|---|---|---|
| `run_registry.status` | `FAILED` | `CANCELLED` |
| `run_jobs…deep_learning.status` | `FAILED` | `CANCELLED` |
| `job_telemetry.cancel.cancelled_at` | `21:38:08Z` (retained) | `23:06:53Z` (retained) |
| `job_telemetry.cancel.reason` | — | `re-proving the sticky-cancellation guard` |

The rest of the unwind is unchanged and still correct: the launcher exited, and the Ray cluster was
torn down — the v1beta1 `persistentResources` endpoint returns `{}`, which is the only teardown
check worth quoting. So the guard suppresses one specific write and nothing else, which is what a
conditional UPDATE in one statement should do.

`cancelled_by: null` in the persisted telemetry is the same P6 `actor=None` finding as above,
confirmed on both runs to propagate into stored state rather than only into the console line — which
is what made it worth chasing rather than dismissing as a cosmetic gap. Root-caused and fixed
2026-09-04 (quota project on the userinfo call); these two rows keep their nulls, since a stored
audit line is a historical record and is not rewritten.

### The workshop's first command printed nothing, and every offline test passed anyway

`docs/workshop.md` opens Act 1 with an offline sanity check:

```bash
uv run python -m scale_forecasting.main --config configs/explode_100k.json --dry-run
```

Run verbatim, it exited **0 with no output whatsoever**. Nothing in the package calls
`logging.basicConfig`, and every verb in the CLI reports through `_log.info` — the resolved
`run_id`, the fanout, the per-family node names, the portable launch commands, `submitted: <id>`.
Python's root logger ships with no handler at WARNING, so all of it was discarded. Correct for a
library; useless for a CLI whose entire job in this command is to *tell you what a run would do*.

`_main` now installs a handler when the root logger has none (guarded, so importing it from Airflow
or a notebook does not double every line; `SF_LOG_LEVEL` overrides). The same command now prints the
run id, `fanout=Fanout(n_series=100000, n_models=4, …)`, both DAG nodes and both launch commands.

**Why no test caught it, and why the new one is written the way it is.** pytest attaches its own
handler to the root logger, so a `caplog` assertion passes against the broken code — the records
exist, they simply have nowhere to go in a real process. The regression test therefore asserts on
the *handler* and the *level*, having first stripped the root logger, which is the only way to
observe the actual defect from inside a test runner.

### Walking the rest of the workshop found three more drifts, and the pattern in them is the same

The silent `--dry-run` was the first command of Act 1. Walking the remaining Act 1–3 commands and
SQL verbatim — checking every claim against the code that implements it or the registry it queries —
turned up three more. None is a code defect; all three are the documentation describing a version of
the system that no longer exists, and all three would mislead someone following the runbook
literally rather than merely confuse them.

**Act 1 told you to look for `SUCCEEDED` runs, and the registry never emits that word.** After the
three 100k submits the runbook said "you want three `SUCCEEDED` rows" from `v_run_summary`. Live, the
column holds only:

| status | rows |
|---|---|
| `COMPLETED` | 78 |
| `FAILED` | 14 |
| `RUNNING` | 9 |
| `PARTIAL` | 7 |

`SUCCEEDED` is the *platform's* vocabulary — what Dataproc and Ray call a finished job, and what
`probes/vocabulary.py` normalises *away from* on the way into the registry. A reader who ran Act 1
correctly would have found no `SUCCEEDED` row and reasonably concluded the runs had not landed. Fixed
by naming the registry's actual vocabulary and saying explicitly which word belongs to which layer.
The same sentence appeared a second time in Act 2's "pre-render only once they're `SUCCEEDED`" and
was fixed with it. One other `SUCCEEDED` in the workshop — the deploy smoke in the opening paragraph
— was checked and left, because that one really is a Dataproc batch state.

**Act 1 advertised a flag that the documented entrypoint does not have.** The note on the two-hour
wait offered `--wait-timeout <seconds>` to change it. That flag exists on
`python -m scale_forecasting.submit`, not on `main` — the entrypoint every command in the runbook
uses. Run as documented it fails outright:

    main: error: unrecognized arguments: --wait-timeout 7200

The 2 h figure itself is right (`_WAIT_TIMEOUT_SECONDS = 7200.0`), and `main.run` never threads a
timeout through, so **there is no knob at all from the documented path** — the doc now says so and
points at the persistent-VM route instead of implying a longer wait is available.

**Act 2's tier table was two notebooks out of date, and Act 3's tour never mentioned them.** The doc
described `smoke` = 3, `batch` = 5, `full` = "all 6". The harness registry has **8**: `smoke` = 4
(`09_review_run` joined it, being registry-read-only), `batch` = 7 (`08_run_and_monitor`), `full` = 8.
The code's own docstring says "all 8", so this is doc drift from when the review-layer pair landed,
not disagreement inside the product. It mattered beyond arithmetic: Act 2 pre-renders whatever the
tier contains, so a presenter running `--tier full` got two rendered notebooks that Act 3's tour
table did not list and gave them no reason to open. Both are now in the tour, in the place the
narrative wants them — `08` launches a mixed Spark + BigQuery run and watches it land, `09` reviews
what `08` just produced — between the engine notebooks and `07`'s cross-run payoff.

**What did check out.** All three Act 1 SQL blocks run as written against the live registry and
return exactly the columns they name (`v_run_summary`: `run_id, created_at, status, python_runtime,
n_series, n_models`; `v_run_jobs`: `run_id, family, runtime, hardware, status, runtime_seconds`; the
`forecast_metadata` progress query). The `submit` extra Act 1 installs exists. `explode_100k`'s live
`v_run_jobs` rows independently reproduce the wall-clocks quoted above — `statistical` 7053.7 s
(117.6 min) and `ml` 3329.3 s (55.5 min) — so the headline row and the registry agree. And Act 2's
`sf-demo-…` job-name prefix is correct: the fan-out path really does use a different prefix
(`sf-demo`) from the blocking harness (`sf-accept`), which looked like a fourth drift until it was
checked.

The generalisation worth keeping: **every one of these four was found by executing the documented
command rather than reading it**, and none was reachable by any test the repo has. Docs drift is
invisible to a test suite that tests code.

### Code the offline gate does not run had rotted in two places, and only running it live showed that

The registry-ops verbs were the last unexercised capability, and getting to them took two repairs
that have nothing to do with the verbs and everything to do with **what the gate covers**. The
offline gate deselects `@gcp`, and it has never had any reason to look at the control-tower tools at
all. Both categories drifted behind refactors that were themselves clean.

**`test_drop_run_deletes_every_tier_of_a_real_run` no longer constructed a valid `CellResult`.**
`model_hash` and `error` became required fields; the test predates them and had not been run since.
It failed at `TypeError` before reaching a single assertion — so the most destructive verb in the
product had the *appearance* of a live test and none of the coverage. Repaired; the six now pass in
135s.

**Both control-tower tools crashed on import-time API drift.** `split_gcs_uri` moved from
`registry.ops` to `registry.artifacts` in the artifacts-before-rows split, and `wipe_registry.py` and
`rebuild_source.py` both still called `ops.split_gcs_uri`. Neither is in any test suite by design —
they are dev tooling, not product — but that is exactly why they rot. Both repaired and both previews
now render against the real deployment.

**The wipe tool's safety interlock then refused, on real data, which is the proof worth having:**

```
REFUSING: 9 run(s) still PENDING/RUNNING — naive-100k-7530d9b41ebb, nb01-spark-connect-…
```

Nine non-terminal run headers are sitting in the registry from interrupted work across the whole
build. A tool whose entire job is irreversible deletion looked at them and stopped. Nothing was
wiped — and nothing should be: the registry holds every `run_id` this ledger's reverse-traces point
at, so a wipe would invalidate the provenance of the campaign that proved the wipe works.

The general lesson is worth stating because it will recur: **a test marked `@gcp` and a tool kept
outside the package are both invisible to the gate, and both silently accumulate drift that only a
live invocation reveals.** Neither failure was a product defect. Both would have been, the first time
someone reached for them in anger.

### `close_runs` worked on the first live try except for the half that only BigQuery can check

The verb's pure half — the status roll-up, the plan, the formatter — was fully offline-tested and
was correct live on the first attempt. The verb still failed on the first attempt, at
`400 Unrecognized name: job_key`, because its I/O half deduped `run_jobs` on a column that does not
exist.

**The wrong column came from correctly applying the wrong table's rule.** `run_registry` is
append-only, so every reader of it takes the latest row per key; I carried that habit to `run_jobs`,
which is not append-only — `jobs.update_job` moves a job to its terminal status with an
`UPDATE … WHERE job_id=@job_id`, in place. The identity column is `job_id`, and there is no
`job_key` anywhere in the schema.

Checking the premise against the live table rather than just fixing the name found that the dedupe
is nonetheless required, for a different reason than the one I had assumed: **197 rows for 166
distinct `job_id`s.** A re-run of an identical config derives the same `run_id` and therefore the
same deterministic `job_id`, and inserts a second row instead of updating the first. So the
latest-per-key roll-up stays — on `job_id`, and justified by re-runs rather than by append-only
writes. Without it, an older `RUNNING` copy would sit beside a newer `COMPLETED` one and the verb
would refuse a run that is perfectly closable.

This is the same shape as the two rots above: **the I/O half of a pure/I-O seam is exactly as
unproven as the seam is clean.** Splitting the pure logic out is what let the roll-up be right on
the first live call; it is also what let a nonexistent column reach production, because everything
either side of the seam tested green.

### The one run `close_runs` refused is a second gap, at the job-row level

Probing `nb03-combo-ensemble-1788329058-c4a5e6db54a1` — the tenth header, the only one with job rows
— returned `STALE_REGISTRY` for its `statistical` family: the Dataproc batch **`SUCCEEDED`**, all
10 of 10 cells landed, and the registry row still says `RUNNING`. The finalize write was lost. Its
`native` family is `COMPLETED` (20/20); its `ensemble` node never ran (0 of 30).

`close_runs` was right to refuse — a non-terminal job row is precisely what it will not guess at.
But nothing else settles this row correctly either. The documented move, `--cancel --force`, writes
`CANCELLED` over a family that demonstrably **succeeded**. That is the *job-row analogue of the
header problem `close_runs` was built to solve*, and it is not fixed: we can now close a header from
its rows, but we cannot close a row from its runtime's own verdict. The probe already computes that
verdict (`native_state='SUCCEEDED'`, `n_done == n_expected`); nothing writes it back.

Left deliberately unclosed rather than papered over with a `CANCELLED` that would be false. It is
also the last remaining in-flight run in the registry, so it is a standing, visible reminder.

**The verb now exists; it has not been run against this row.** `main --settle` /
`Forecaster.settle()` writes a job row from the probe's own verdict — `STALE_REGISTRY` +
`SUCCEEDED` + all cells landed ⇒ `COMPLETED` — and refuses everything ambiguous. That is offline
work only: **no line of it has touched live infrastructure**, so it gets no row in the table above.
This run is the fixture reserved to prove it, and there is exactly one of it. Draft the ledger row
before the command runs.

### The shipped baseline is measured numbers, and the axis it does *not* move is the interesting part

`load_baseline()` returned `None` until 2026-09-03, with a docstring saying why: *"a baseline whose
provenance is a chat message is the exact failure that ledger exists to prevent."* The measurements
now exist, so it ships one.

**What it is.** `src/scale_forecasting/profiling/baseline.py` holds a `ComputeProfile.to_dict()`
payload harvested from `ray-100k-dcc77a9d1e9b` — 100,000 daily series of 1,460 observations from
`source_series_iceberg`, fitted on Ray on Vertex across `theta` / `holtwinters` / `sarimax` /
`xgboost`. It was pulled through `registry.harvest.read_compute_harvest`, whose 50,000-cell cap
makes it a deterministic `FARM_FINGERPRINT(ts_id)` slice of 12,500 series × 4 models out of the
400,000 fits that run performed. That cap is load-bearing rather than a compromise: it makes the
committed payload byte-identical to what pinning `compute.profile.source:
"ray-100k-dcc77a9d1e9b"` resolves to, so the baseline is not a second kind of artifact — it is a
harvest, frozen.

It is a **Python module and not a JSON file** because `code_delivery.build_package_zip` walks
`*.py` and nothing else. A data file would never reach a Dataproc worker or a Ray `working_dir`,
and `resolve_profile_source` swallows a loader failure by design — so the miss would degrade sizing
on exactly the clusters that matter and say nothing.

**It does not move `fleet_sizing`, and that was checked rather than assumed.** Shipping a baseline
changes what a run gets when precedence step 4 is reached — no pin, and discovery finds nothing.
Every config in this repo uses the defaults (`mode: auto`, `source: auto`), so the question is
whether `discover_harvest_run` comes back empty for any of them. It does not: on 2026-09-03 the
90-day window held **32 completed, measured candidate runs** on `source_series_iceberg`/`D` and one
on `source_series_native`/`D`, which between them cover every `data.source_table` any config here
names. Step 4 is unreachable for every ledger-tracked row, so no row's fleet changes and no row
goes stale. Beyond that, the axis is about the *derivation* — the Spark properties overlay — and a
baseline changes only what evidence the derivation is handed.

**What is not proven, and cannot be proven here.** No run has been sized *from* the baseline on
live infrastructure, because reaching it requires a registry with no matching harvest and this
project has 32 of them. The claim in the capabilities table is therefore narrow and deliberate: the
numbers are real and traceable to a live run. Exercising the consumption path is a fresh-deployment
test, and it belongs with the other fresh-deploy gaps below.

**What it deliberately omits.** No `deep_learning` family and no GPU bound — that run had neither.
`for_family` returns `None` for an unmeasured family and `None` means "fall back to static config",
so a run with a deep-learning family gets today's arithmetic for it and measured numbers for the
rest. A fabricated device bound would be worse than an absent one. The number the baseline is
really for is `slot_cores: 1`: `max_effective_cores` measured 1.01–1.05 across all four models, so
every one of these fits is single-threaded. That is a property of the libraries, not of the panel,
which is why it transfers to a user's data in a way a memory bound does not.

## Known validation gaps

Things that are true today and that no entry above covers. Keep this list short and act on it.

- **A run that paid for a GPU and never touched one is now *named*, but still not failed.** The
  verdict this bullet used to ask for exists: `device_audit` runs at the end of every
  GPU-provisioned job and stamps `job_telemetry.device_use.verdict`, and all three services were
  seen producing `ENGAGED_IDLE` on 2026-09-09 and `MISSING_DEVICE` on 2026-09-10. What it does not
  do is change the run's outcome, deliberately — a device that sat idle produced a *correct* run and
  an expensive one, which is a cost finding rather than a fault. The open part is that nobody is
  told. The verdict sits in a JSON column that an operator has to know to query, so "the GPU earned
  its cost" is still a manual check on every GPU run; it wants a place in the review surface, not a
  new failure mode.

- **Two of the three services could not say *why* a GPU job lost its device. Fixed 2026-09-10 and
  proven live the same day.** The forecast was withheld on all three, but only the Dataproc cluster
  reached `_require_device` and recorded the contract message: on Ray the worker holding a GPU slot
  crashed before the check ran, and on Serverless the RAPIDS plugin died first and Spark replaced
  the executor indefinitely — the worse of the two, because an unattended batch then burns fleet
  until something stops it. Two fixes, and the second is a production change rather than a test one:
  `SF_HIDE_DEVICES=probe` injects the fault at our own probe so it reaches the check instead of the
  CUDA library underneath it, and a bounded executor-failure budget plus a driver-side stall
  watchdog make the batch give up. All three arms were re-run under `probe` and all three recorded
  `CONFIG_REPAIRABLE` on six cells, naming their own service; smoke 17 ended in 17.8 minutes against
  49 of churning. **What this gap leaves behind** is that the deeper `cuda` mode still cannot be
  used to test the check on Ray or Serverless — the crash and the plugin abort are real platform
  behaviour and are unchanged. A genuine card failure in production would still present as those,
  not as a named contract error.

- **A per-task memory clamp with no headroom is unschedulable, and no offline test could have
  caught it. Fixed 2026-09-03 at `17e1221` and proven live the same day.** `ray_100k` held at zero cells for
  57 minutes on a request of exactly `0.7 × nameplate RAM` — our own ceiling, refused by Ray because
  Ray derives its ceiling from what the container's OS reports rather than from the machine type's
  nameplate. The gap the unit tests could not close is that **the true ceiling is not knowable from
  a machine-type name**, so every test of "does the clamp fit" was testing our estimate against
  itself. What is testable, and now is, is the weaker invariant that actually prevents the hang: a
  slot may never be clamped to a node's *entire* schedulable memory. The re-run settled it:
  `ray-100k-dcc77a9d1e9b` COMPLETED with **400,000 cells** in 330 minutes, and **no clamp fired at
  all** — the runtime-ranked harvest supplied 1.29 GiB per task, which is not close enough to the
  ceiling for the headroom to matter. What is still open is narrower than the bullet was: the 0.85
  figure has been *exercised* only on `n1-standard-8`, and the run that proved the ranker did so
  without ever reaching the clamp. The hang is fixed twice over (right evidence, and headroom if the
  evidence is ever wrong); the second belt has not itself been pulled tight by a live run.
- **The scale ceiling this project has been proving against is a *quota* ceiling, not an
  architectural one, and no row above said so.** Measured 2026-09-03: `us-central1` allows **200
  CPUs** and **4 NVIDIA T4s**. `ray_100k` used 192 of the 200 — an `n1-highmem-32` head plus 20
  `n1-standard-8` workers — so `ray_cpu_max_nodes: 20` was never a tuning choice, it is simply what
  fits. Its own telemetry recorded `saturating_units: 12500`, meaning the arithmetic wanted **625x**
  what the project is allowed to ask for. Every "N minutes at 100k" figure in this document is
  therefore a statement about a 200-core allowance, not about the product, and raising a
  `max_nodes` knob cannot change any of them.

  The GPU side is where the ceiling stops being a cost question. The two runtimes draw on different
  allowances — **4** `NVIDIA_T4_GPUS` for Dataproc, **12** `custom_model_training_nvidia_t4_gpus`
  for Ray on Vertex — and `neuralprophet` measures at **7.6 fits/min per T4** (2026-09-04, revised
  up from an extrapolated 4; confirmed at 8.2 the next day on a 3-fit-per-cell run). A single-fold
  deep-learning pass is therefore ~1 h 50 m at 10,000 series on the
  Ray allowance and **~18 hours at 100,000** — or ~55 hours on Dataproc's four, and roughly three
  times each of those with `n_folds: 2`. Not expensive so
  much as unfinishable. **A default project runs out of deep-learning headroom somewhere around
  10,000–20,000 series, an order of magnitude before it runs out of CPU headroom.** The 100k
  all-families
  configs were therefore retired in favour of `all_families_10k` / `all_families_10k_full`: 10,000
  series is the largest scale a stock project can actually reproduce, which is the only scale a
  demonstration config should claim. The 100k rows that remain above (`explode_100k`, `ray_100k`)
  are kept because they are *proven* — they are the evidence that the architecture reaches 100k,
  and `docs/quota_and_scale.md` is where the arithmetic for going beyond it now lives.
- **~~No deep-learning run has ever exceeded 100 series, anywhere in this ledger.~~ Closed
  2026-09-04 by `all_families_10k`** — 10,000 `neuralprophet` series on 12 T4s. Until then every
  `neuralprophet` row — smokes 03, 06, 09, 10, 14, 16, `ray_gpu_demo`, `per_family_runtimes_demo` —
  was at 100 cells or fewer, so the GPU code path was well proven and its *throughput at scale* was
  not measured at all. **The measurement is now taken and it is worse than the extrapolation, for a
  reason that was fixed the same day.** The first pass took 27,245 s across 12 T4s: 22.0 cells/min
  fleet-wide, or **1.8 cells/min per T4** against the 4 cells/min/T4 published in
  `quota_and_scale.md`. The per-cell fit averaged 30.25 s, which is 2.0 cells/min in a *single*
  stream — so the fleet of 12 delivered almost exactly what one serial GPU would, the signature of
  one cell per node (`ray_slot_memory`, above). **The second pass, under the fix, is the number to
  use: 6,582 s, 91 fits/min fleet-wide, `7.6 fits/min per T4`** — nearly twice the figure that had
  been extrapolated from 100-series runs, and now measured at 10,000. `quota_and_scale.md` is
  rewritten from it.

  **Re-confirmed 2026-09-05 by `all_families_10k_full`, which also showed why the pool never
  saturates.** At 3 fits per cell that run delivered 8.2 fits/min/T4 — the same anchor, which settles
  that it is a per-*fit* rate. But it held `GPU [8.4, 12.0]` and `CPU [84.0, 91.0]` flat for five
  hours: a deep-learning task asks for a vCPU as well as a GPU fraction, and 12 `n1-standard-8`
  workers have only 84 usable cores, so **84 concurrent fits is the ceiling and 30 % of the T4
  allowance cannot be reached at that machine shape.** Distinct from `ray_slot_memory` — nothing is
  over-requesting here, there simply are not enough cores to feed the devices. The lever is
  `ray_gpu_machine_type`, not `ray_gpu_max_nodes`.
- **The Ray fleet ran at roughly one busy core in eight, and the resource plan did not predict it.**
  Measured 2026-09-03 from `ray-100k-dcc77a9d1e9b`: 37,500 chunk tasks were queued against 20
  `n1-standard-8` workers, and the plan expected 4–8 concurrent cells per node. Actual concurrency,
  computed as `SUM(fit_seconds) / wall` per 30-minute bucket, was **0.97 cells per node** — flat, at
  exactly one, for eight consecutive buckets. `cell_started_at`/`cell_ended_at` bracket the same
  duration as `fit_seconds`, so the idle time is not inside the cell; it is between cells or between
  chunks. Two candidate causes, not yet separated: Ray placing one task per node despite `num_cpus`
  leaving room for four, or per-chunk overhead dominating a chunk of only 8 cells
  (`_DEFAULT_TARGET_CELLS_PER_SLOT = 8` produced 37,500 separate tasks and 37,500 separate
  registry writes for 300,000 cells). **Separating them needs a live `ray status` during a run,
  which is why this is a gap and not a fix.** It is not a correctness problem — every published
  wall-clock figure is real and reproducible — but it means the throughput numbers are a floor, and
  a fleet four to eight times faster may be available at identical quota.

  **Separated 2026-09-04, and it was the first cause.** The live cluster read this gap asked for was
  taken mid-flight against `all-families-10k-eb01dcfecfab` — the Ray dashboard's
  `/api/cluster_status`, `loadMetricsReport.usageByNode` — and every worker showed the same three
  numbers: 1.0 of 7.0 CPUs, 0.1 of 1.0 GPU, and **17.6 of 18.1 GiB of memory**. Ray schedules on
  `memory` as hard as it schedules on `num_cpus`, so a task holding 97 % of a node's memory holds
  the node, and the seven idle cores were never reachable. Chunk overhead is not exonerated but it
  is not what was binding. The cause of the memory request is `ray_slot_memory`, above, and the fix
  is `efecb4c`. **The "four to eight times faster at identical quota" guess was close.** The same
  config re-run under the fix hours later went from 0.93 to 5.5 concurrent cells per node — 5.9x
  density, 4.1x wall-clock on the GPU family, 3.8x on the whole run, the difference between the two
  being per-cell contention. This item is closed on both halves: the cause is identified and the
  improvement is measured.
- **The prebaked GPU cluster image expires, and nothing in the deployment notices.** Found live
  2026-09-02 by smoke 16, analysed above: an image built nine days earlier was refused because the
  Dataproc sub-minor baked into it had been retired. **Smoke 06 is the row that rests on this.** Its
  proof is real and its code path is unchanged, but the specific artifact it ran on can no longer
  create a cluster, so it cannot be reproduced as written until the image is rebuilt. The error is
  now legible (`_explain_create_failure`) and the fallback is proven (smoke 16), so nothing is
  *blocked* — what is undecided is the default. Rebuilding on a schedule keeps ~14 minutes off every
  GPU cluster create and adds a maintenance job that nobody will remember to run; dropping the image
  and always using the init action removes a whole class of silent expiry and pays that 14 minutes
  every time. **This is an owner call, not a bug fix**, and it should be made before the next
  greenfield deploy rather than discovered by it again.
- **~~A run that needs both a CPU and a GPU Dataproc cluster has never been executed.~~ Closed
  2026-09-02 by smoke 16** — two clusters, two names, two sizings, two teardowns, one `run_id`,
  analysed above. The live half named below is now done except for one clause: **the per-cluster
  region after a capacity hop is still unproven**, because neither cluster hopped. A Dataproc
  cluster has one worker machine type, so as of 2026-09-02 a run's ephemeral cluster families are
  grouped by hardware and get one right-sized cluster each — `sf-cluster-<run_id>-cpu` alongside
  `sf-cluster-<run_id>-gpu`. **No row above goes stale, and until smoke 16 none of them reached the
  new branch either**: smoke 04 was the only config with two ephemeral cluster families and both are
  CPU, so it takes the single-group path and is byte-identical to what it was — same one cluster,
  same unsuffixed name, same sizing. `16_cluster_split_hardware.json` was added to close the config
  half of the gap (a `statistical` CPU family and a `deep_learning` T4 family, both
  `spark_mode: cluster`), and `tests/smokes/test_smoke_configs.py` now fails if the library ever
  stops containing one. Offline tests pin the partial-create unwind, which the live run did not
  exercise because neither create failed after the other had succeeded.
- **`ray_autoscale` defaults to `True` (`config.py`) but all four Ray smokes pin it `false`.**
  Introduced by `4c988bc`, when a per-pool `AutoscalingSpec` crashed the Vertex Ray head at
  provisioning. **Resolved on the demonstration surface 2026-09-01**, and the suspected cause was
  right: `ray_autoscale_demo` scaled 1→8 workers and completed, so the `4c988bc` crash was the
  custom image — since deleted — not the autoscaling spec. What remains is smaller and is a
  *hygiene* gap rather than an unknown: the four Ray smokes still pin `false`, so the cheap
  fifteen-minute path does not cover the shipped default, and a regression in it would only surface
  on a demonstration run. Unpinning them is the fix; it is not urgent, because the default is now
  proven at 10,000 series, which is a harder case than any smoke poses.
- **~~No `run_id` was recorded for smokes 07–14.~~ Closed 2026-09-02.** Every re-run in this
  campaign captured one, so 01–08 and 11–14 all carry a reverse-trace. Only 09 and 10 still say "not
  recorded", and both are now runnable — the Ray GPU blocker they were held behind turned out not to
  exist (above), so what remains is spend, not entitlement.
- **~~The Ray probe cannot escalate to a single-family Ray run.~~ Closed 2026-09-02 — fixed offline
  and re-proven live the same day.** `resource_name` was absent from the entry handle for the shape that has
  no shared cluster, and the corrected handle landed only after the job was terminal; `--cancel`
  inherited it. Found live 2026-09-02, analysed above. The fix is that the cluster name is a pure
  function of the `run_id` — `ray_io.cluster_name` — so the entry handle now *predicts* the path the
  submitter is about to create instead of waiting to be told. The one guess in it is the region: a
  capacity hop would move the cluster and the predicted path would miss, degrading the probe to
  registry-only, which is exactly where it was before. Re-proven by a live `--probe` against
  `ray-dl-on-cpu-probe-2e8a9f3f5c8d` while its one Ray family was running: `RUNNING_CONFIRMED`,
  from a separate process, and by `--cancel --force` reaching the family on
  `ray-cancel-probe-e22e6fe9a830`. `job_telemetry.probe_handle` on that run carries a complete
  `resource_name`, so the fix is visible in stored state and not only in console output.
- **~~A cancellation does not survive the launcher's unwind: the run ends `FAILED`, not
  `CANCELLED`.~~ Closed 2026-09-02 — fixed offline and re-proven live the same day** on
  `ray-cancel-sticky-824b4822945c`, where the identical unwind left the header `CANCELLED`.
  Found live 2026-09-02 on `ray-cancel-probe-e22e6fe9a830`, analysed above. `--cancel --force` stops
  the job and writes `header=CANCELLED`; the launching process then sees its own job go `STOPPED`,
  treats that as death, and finalizes the run `FAILED` 17 s later. The evidence is not destroyed —
  `job_telemetry.cancel` keeps `cancelled_at`, `native_state_at_cancel` and `n_done_at_cancel` — but
  it is demoted out of every surface that reads `status`, so a deliberate stop is indistinguishable
  from a crash in the header, in `doctor`, and in any status filter. The fix is one rule —
  **a cancellation is sticky against the failure it caused** — and it is offline-testable once
  stated, though not offline-*findable*: it needs two processes disagreeing about one run, and until
  the handle fix landed the CLI could not reach far enough to create the disagreement. The fix is a
  status guard in the UPDATE's WHERE clause, applied by `registry.lifecycle` to every non-green
  finalize. That live cancel has now been run.
- **~~A `deep_learning` family on Ray with `hardware: cpu` hangs indefinitely.~~ Closed 2026-09-02
  — fixed by `9eeb154` and proven live the same day.** Zero-worker cluster, no timeout, no error —
  analysed above. The fix made `split_gpu_cpu_models` hardware-aware, so with no GPU pool the
  deep-learning cells are sized into the CPU pool instead of falling between the two. The live proof
  used the exact config the commit body names as reachable —
  `{"python_runtime": "ray", "models": ["neuralprophet"]}` — which ran to COMPLETED as
  `ray-dl-on-cpu-probe-2e8a9f3f5c8d`, `deep_learning ray/cpu`, 6 cells. That run was the probe target
  above, so one cheap Ray-CPU run closed both gaps: **the hang was the reason the probe fix had
  nothing safe to be tested against.**
- **~~Vertex Ray will not provision in `us-central1` for this project as of 2026-09-02.~~ Cleared
  the same day, ~17:00 UTC.** Six specs failed with the contentless internal error, including a
  deliberate re-probe hours later under a fresh cluster name; then three bisect arms and smoke 08
  all provisioned normally with no change on our side. Environment, not code, and no support case
  was filed. The lasting form of this gap is narrower: **a Ray conclusion is only as good as the day
  it was reached.** The fault was transient once and can be transient again, so re-probe before
  trusting any Ray *negative* result, and prefer running Ray work early in a window rather than last.
- **A failed Ray provision can leak a cluster that blocks every retry of that config.** The name is
  derived from the `run_id`, so the retry collides with `AlreadyExists` and cannot succeed until the
  leaked resource is deleted by hand — and `list` reports `[]` while it exists, so it is invisible
  from the obvious command. Analysed above. **Intermittent:** a second failed provision under
  identical conditions tore down cleanly, so the teardown success line tells you nothing either way
  and `describe` is the only check. **Both product-side fixes have since landed offline** (2026-09-02):
  `_delete_cluster` now polls the resource until it reads `NOT_FOUND` and logs a named, still-billing
  leak when it does not, and `_clear_stale_resource` deletes an `ERROR`-state same-named resource
  before each create while deliberately leaving a `RUNNING`/`PROVISIONING` one alone. Unit-tested,
  **not yet proven live** — and it cannot be proven on demand, because the leak is intermittent: the
  live evidence will be the absence of a hand-deleted cluster over the next several Ray failures,
  which is the weakest kind of proof there is. Recorded as offline-only until a failed provision
  happens to exercise it.
- **Ten run headers are stuck non-terminal, and no verb closes them.** Left by interrupted work
  across the whole build (`naive-100k`, several `nb01-spark-connect`, `nb03`, `nb06`, …), most
  recently `nb03-combo-ensemble-1788329058-c4a5e6db54a1` on 2026-09-02. Harmless to reads, but they
  block the dev wipe tool's interlock and they make "is anything running?" unanswerable at a glance.
  **An earlier version of this bullet named `sweep_orphans` as the fix, and that was wrong** —
  `sweep_orphans` deletes artifact prefixes that have *no* registry row, which is the opposite
  direction. Checked against the actual verb list (`init`, `doctor`, `drop_run`, `sweep_orphans`,
  `snapshot`, `export`, plus `--probe`/`--cancel`), **nothing finalizes a non-terminal header whose
  jobs have all already succeeded**: `--cancel` would stamp `CANCELLED` on work that completed,
  `drop_run` would throw away real predictions, and `--probe` only reads. So this is a genuine
  coverage gap, not a chore. The missing verb is a reconcile-and-close: take the reconciled per-job
  truth `--probe` already computes and *write* the resulting terminal status to the header.
  **The verb is now built** (`registry.ops.close_runs`, seventh verb, CLI `close-runs`, SDK
  `Registry.close_runs`): it writes a header status and nothing else, and refuses any run whose job
  rows are not already all terminal. Two facts from querying the live registry shaped it — 9 of the
  10 stuck runs have **no job rows at all** (they died in the submit path, so they close as `FAILED`
  by an explicit rule rather than by falling through a roll-up), and the tenth
  (`nb03-combo-ensemble-1788329058-c4a5e6db54a1`) still has a `RUNNING` family, so it is *skipped
  with a reason* rather than guessed at. **Closed live 2026-09-02** — nine headers went to `FAILED`,
  the tenth was skipped, and `doctor` now reports one in-flight run instead of ten. Two things the
  live call taught that the offline gate could not; both are recorded below.
- **Nothing runs the `@gcp` tests or the control-tower tools on a schedule.** Both rotted (above).
  A cheap mitigation is import-only smoke coverage for the tools and a periodic `-m gcp` collection
  pass (`--collect-only` catches neither of these; the `CellResult` break needed execution).
- **The `--cancel` summary line miscounts.** It reported `1 in-flight job(s) stopped` on a run where
  the per-family line said `NOT cancelled`. One-line fix; not made mid-campaign only because it sits
  in the same function as the handle fix. **Fixed offline** — the headline now counts outcomes
  (`n of N stopped`) after executing, and the plan count only in preview, with tests; the live
  re-observation still belongs to the next spend window.
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
  post-fix run, submitted at 16:02 — one minute after the fix landed — so it declares the new
  `run_id_inputs=authored-config-only` honestly. Smoke 02 never was re-run, and neither was
  `bq_native_demo`, which ran in the same pre-fix window; both rows were carrying no
  `run_id_inputs` value at all, which read as "unaffected" when the truth is "proven under the old
  one". They now declare `run_id_inputs=+compute.profile.source` and are graded **STALE**, which is
  what the axis is for. Unlike the three moves before it, this one makes identity *stop* drifting
  rather than start: there is no resolved value left in the digest to move it again.

  **But "no resolved value left" is not "no drift left", and the distinction cost us six rows.**
  That sentence is only about values the *launcher* fills in. A plain new field with a default
  still moves every id, because the digest hashes the dumped payload and a defaulted field appears
  in every config's dump whether or not the config mentions it. One landed the same evening —
  `d6fe690` at 18:11 added `compute.max_executors` — and its own commit body says so outright
  ("the new config field moving every run_id"). What nobody did was reconcile the rows already
  written earlier that day. Six of them record a `run_id` their config no longer produces:

  | Row | Proven id | The id its config resolves to today |
  |---|---|---|
  | smoke 02 | `smoke-02-bq-native-0ffcc1f22d54` | `…-58478a846e89` |
  | `bq_native_demo` | `bq-native-demo-b374041fdd1e` | `…-a54e5afe2ba4` |
  | smoke 01 | `smoke-01-serverless-cpu-5af5de1accf2` | `…-95b59d88fbe8` |
  | `explode_demo` | `explode-demo-d1b57690dc96` | `…-2226cd3f0780` |
  | `mixed_demo` | `mixed-demo-405983dddf0a` | `…-d87f8cdab605` |
  | `ensemble_demo` | `ensemble-demo-9849a2f73669` | `…-c89b6f04afed` |

  Every one of the other twenty-two matches, and the reason is simply that they were re-earned on
  2026-09-02 or later. The six are exactly the runs of 2026-09-01 that finished before 18:11.

  Smoke 01 is the interesting one, because it is **CURRENT and every axis it declares is correct**,
  and its id still does not reproduce. That is the gap in the grading model: an axis records a
  decision the result depends on, and adding a config field is not a decision — it moves identity
  without touching any claim. The ledger cannot see it, and for four days nothing else could either.

  Two things close it. The narrow one is that the digest-input *set* is itself worth versioning:
  the planned break bumped this axis to `authored-config-only-v2` the same day, and every row went
  stale by construction rather than by inspection — see the note under the axis table. The durable
  one is mechanical:
  `tests/unit/snapshots/run_ids_prebreak.json` pins today's digest for all twenty-eight shipped
  configs, and `tests/unit/test_prebreak_snapshots.py` fails the offline gate the moment an
  unplanned field moves one. Discovering this by hand, once, was the last time that should be
  necessary.
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

  **Seen a second time on 2026-09-02, which is what moves it from a plausible reading to a habit.**
  `ray-cancel-probe-e22e6fe9a830` (200 series) was sized off the 6-series probe run finished an hour
  earlier — `compute profile: series count differs by 33x (6 measured vs 200 planned)` — with better
  matches available. The pattern is now clear: because a campaign writes small harvests often and
  large ones rarely, "newest" reliably selects the *least* representative evidence in the table. Note
  that the lookback window already does the freshness job, so recency is being paid for twice; the
  ordering is the part with nothing to defend it. The complication is that `discover_harvest_run`
  takes only the two identity axes and would need the target's scale to rank on proximity — a
  signature change reaching the `source = "auto"` caller, which is why this stays recorded rather
  than squeezed into a campaign gap.

  **A third sighting the same day settled it, and it is now fixed offline.**
  `ray-cancel-sticky-824b4822945c` (200 series) resolved to `smoke-16-cluster-split-hardware-…`,
  which had finished twenty minutes earlier and measured six series. Three sightings in one
  campaign, and the next runs due are the 100k long poles — where "newest" would hand a
  100,000-series plan a six-series profile. The deferral reasoning above was about *cost*, and it
  was wrong on the facts: both callers already hold the signature, so passing `want.n_series` is
  two lines, not a refactor. `discover_harvest_run` now returns every candidate in the window with
  its measured series count and a pure `rank_harvest_candidates` chooses — closeness of scale
  first, recency only as a tie-break, distance in log space, symmetric, rounded so near-ties are
  ties. The policy left SQL specifically so it could be tested; eight offline tests pin it.
  **Proven live on the very next run.** `ray_100k` (100,000 series, `source="auto"`) resolved to
  `explode-100k-1c59265062aa` — a 100,000-series harvest, an exact scale match. The candidate
  recency would have chosen was `smoke-16-cluster-split-hardware-…`, 100 series, measured
  46 minutes earlier: a **1000x** mismatch, on the most expensive run in the campaign. Both
  candidates were in the same window and the same table, so this is the two rules disagreeing on
  identical evidence, not a change in what was available.

  One honest qualification. Discovery ranks on the **true** measured count, but the loading read is
  still capped at `_MAX_HARVEST_CELLS = 50_000`, so what actually got loaded is a deterministic
  fingerprint-ordered slice of the 400k cells — roughly an eighth of the panel — and the profile's
  signature reports that smaller number. The pick is exact; the sample is not. That is the cap
  doing its job (a submit host must not materialise half a million dicts), and it means the
  remaining signature warning on a 100k run is ~8x rather than the 1000x it would have been.

  **And then the exact-scale pick hung the run, which is the part worth keeping.** The harvest it
  correctly chose was a *Spark* run and the plan it sized was a *Ray* one, and
  `process_rss_bytes` does not mean the same thing on both: one Ray task versus one Spark executor
  running many cells. `ray_100k` took a ~21 GiB per-task memory bound from it, Ray could not place
  the task on any node, and the run sat at zero cells for 57 minutes — analysed in full above. So
  the fix was right and incomplete in the same breath: **scale was never the only axis, and a
  one-axis ranker had been getting away with it only because the corpus was small.**
  `rank_harvest_candidates` now ranks runtime comparability first (all-target-runtime → mixed or
  unrecorded → none), scale second, recency last; discovery aggregates each candidate's
  `compute_engine` set to feed it. Seven more offline tests, and **live-proven the same day**: the
  `ray_100k` re-run resolved `auto` to `ray-autoscale-demo-886a053c374c` — a Ray run at 10,000
  series — in preference to the 100,000-series Spark harvest that is still the better scale match,
  and completed 400,000 cells. Runtime beat scale, which is the whole of the fix.

  One axis remains unranked, and the same run showed it: `ray-autoscale-demo` measured three
  statistical models and no `xgboost`, so the `ml` family fell back to `basis: static` while
  `statistical` got `basis: measured`. The ranker has no notion of *coverage* — "does this harvest
  contain the families I am about to run" — and picked a partial match over nothing. That was the
  right call here and cost nothing, because the static fallback for `ml` was adequate. It is a
  fourth axis, not a defect, and it is cheap to add when a run makes it matter.

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
