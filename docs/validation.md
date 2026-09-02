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
| 06 | `06_cluster_gpu.json` | Dataproc cluster GPU (T4), incl. zone failover | CURRENT | 2026-09-02 | `smoke-06-cluster-gpu-2f7296ef8839` | `cluster_deps=packed-venv-init-action`, `gpu_cluster_image=prebaked-driver-image`, `python=3.11`, `fleet_sizing=derived-overlay`, `horizon_features=computed-at-future-dates`, `run_id_inputs=authored-config-only` |
| 07 | `07_ray_cpu.json` | Ray on Vertex, CPU | CURRENT | 2026-09-01 | `smoke-07-ray-cpu-782bcec2718f` | `ray_deps=stock-image+uv-runtime-env`, `python=3.11`, `run_id_inputs=authored-config-only`, `horizon_features=computed-at-future-dates` |
| 08 | `08_ray_gpu.json` | Ray on Vertex, GPU T4 (neuralprophet) | CURRENT | 2026-09-02 | `smoke-08-ray-gpu-c41ecf2d5d52` | `ray_deps=stock-image+uv-runtime-env`, `python=3.11`, `run_id_inputs=authored-config-only` |
| 09 | `09_shared_ray.json` | Several families on one shared Ray cluster (CPU + GPU pools) | CURRENT | 2026-09-02 | `smoke-09-shared-ray-1d308b8a712c` | `ray_deps=stock-image+uv-runtime-env`, `python=3.11`, `run_id_inputs=authored-config-only`, `horizon_features=computed-at-future-dates` |
| 10 | `10_mixed_runtimes.json` | Spark + Ray + BigQuery families concurrently under one run_id | CURRENT | 2026-09-02 | `smoke-10-mixed-runtimes-f6f98f70eb80` | `ray_deps=stock-image+uv-runtime-env`, `serverless_deps=container-image`, `native_source_pin=unpinned-all-sources`, `python=3.11`, `fleet_sizing=derived-overlay`, `horizon_features=computed-at-future-dates`, `run_id_inputs=authored-config-only` |
| 11 | `11_ensemble_barrier.json` | Ensembling in barrier mode | CURRENT | 2026-09-02 | `smoke-11-ensemble-barrier-19926ef4b90f` | `serverless_deps=container-image`, `native_source_pin=unpinned-all-sources`, `python=3.11`, `fleet_sizing=derived-overlay`, `horizon_features=computed-at-future-dates`, `run_id_inputs=authored-config-only` |
| 12 | `12_ensemble_microbatch.json` | Ensembling in microbatch mode | CURRENT | 2026-09-02 | `smoke-12-ensemble-microbatch-f165a65d0b65` | `serverless_deps=container-image`, `native_source_pin=unpinned-all-sources`, `python=3.11`, `fleet_sizing=derived-overlay`, `horizon_features=computed-at-future-dates`, `run_id_inputs=authored-config-only` |
| 13 | `13_native_format.json` | Reading the native BigQuery source table | CURRENT | 2026-09-02 | `smoke-13-native-format-8e67fd137515` | `native_source_pin=unpinned-all-sources`, `serverless_deps=container-image`, `python=3.11`, `fleet_sizing=derived-overlay`, `horizon_features=computed-at-future-dates`, `run_id_inputs=authored-config-only` |
| 14 | `14_full_dag.json` | Flagship: all families + native + ensemble, one run_id (DL on Spark L4) | CURRENT | 2026-09-02 | `smoke-14-full-dag-c8664f7a2d23` | `serverless_deps=container-image`, `native_source_pin=unpinned-all-sources`, `python=3.11`, `fleet_sizing=derived-overlay`, `horizon_features=computed-at-future-dates`, `run_id_inputs=authored-config-only` |
| 15 | `15_airflow_multi_engine.json` | The whole DAG orchestrated by Composer/Airflow | NEVER_RUN | — | — | — |
| 16 | `16_cluster_split_hardware.json` | One run needing **two** Dataproc clusters at once — a CPU one and a GPU one | NEVER_RUN | — | — | — |

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

Not fixed here, for the same reason as the failover above: there is more than one defensible answer
(fold DL models into the CPU pool when `use_gpu` is false, so a CPU Ray run of a DL family simply
runs slowly; or reject a zero-worker plan before provisioning; or both), and the sizing path feeds
`run_id`-relevant config. Until it is fixed, **`hardware: cpu` on a `deep_learning` Ray family is a
hang, not a slower run.** The zero-worker plan is the detectable signal — no valid run ever wants
one.

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
| `per_family_runtimes_demo.json` | Per-family runtime split — deep learning to Ray GPU, the rest on Spark (50) | CURRENT | 2026-09-02 | `per-family-runtimes-demo-f1746911caf5` | `serverless_deps=container-image`, `ray_deps=stock-image+uv-runtime-env`, `native_source_pin=unpinned-all-sources`, `python=3.11`, `fleet_sizing=derived-overlay`, `horizon_features=computed-at-future-dates`, `run_id_inputs=authored-config-only` |
| `ray_cpu_demo.json` | Ray on Vertex, CPU, alongside the natives, backtested (6) | CURRENT | 2026-09-01 | `ray-cpu-demo-f6b6fbdb83a5` | `ray_deps=stock-image+uv-runtime-env`, `python=3.11`, `run_id_inputs=authored-config-only` |
| `ray_gpu_demo.json` | Ray on Vertex, GPU T4 (`neuralprophet`), alongside the natives (6) | CURRENT | 2026-09-02 | `ray-gpu-demo-e2dcbef4a373` | `ray_deps=stock-image+uv-runtime-env`, `python=3.11`, `native_source_pin=unpinned-all-sources`, `run_id_inputs=authored-config-only` |
| `ray_autoscale_demo.json` | **The shipped `ray_autoscale=true` default**, 1→8 CPU nodes at 10,000 series | CURRENT | 2026-09-01 | `ray-autoscale-demo-886a053c374c` | `ray_deps=stock-image+uv-runtime-env`, `python=3.11`, `run_id_inputs=authored-config-only`, `horizon_features=computed-at-future-dates` |
| `explode_100k.json` | The headline: Spark `explode` over 100,000 series | CURRENT | 2026-09-01 | `explode-100k-1c59265062aa` | `serverless_deps=container-image`, `python=3.11`, `fleet_sizing=derived-overlay`, `run_id_inputs=authored-config-only`, `horizon_features=computed-at-future-dates` |
| `ray_100k.json` | The same work on Ray — the runtime-parity half of the scale review. Attempted 2026-09-02 and never reached a job: blocked by the Ray provisioning outage below, not by GPU. That outage has since cleared, so this is runnable again | NEVER_RUN | — | — | — |
| `all_families_100k.json` | Every family at 100,000 series under one `run_id` (Ray + BigQuery, T4) | NEVER_RUN | — | — | — |
| `all_families_100k_full.json` | As above, plus backtesting and persisted artifacts | NEVER_RUN | — | — | — |

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
`per_family_runtimes_demo`, `all_families_100k` and `all_families_100k_full` all put a family on
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
notebook reflects the current path. **Seven were re-executed on 2026-09-02** against current code and
re-committed with their new outputs, clearing the last `STALE` row in this table. The executed
notebooks were diffed against the committed ones first: source cells were byte-identical in all
seven, so only outputs changed.

| Notebook | Status | Date | Axes at proof |
|----------|--------|------|---------------|
| `01_spark_via_connect.ipynb` | CURRENT | 2026-09-02 | `serverless_deps=container-image`, `python=3.11`, `horizon_features=computed-at-future-dates` |
| `02_bigquery_native.ipynb` | CURRENT | 2026-09-02 | `python=3.11` |
| `03_combo_and_ensemble.ipynb` | CURRENT | 2026-09-02 | `serverless_deps=container-image`, `python=3.11`, `fleet_sizing=derived-overlay` |
| `04_ray_on_vertex.ipynb` | CURRENT | 2026-08-28 | `ray_deps=stock-image+uv-runtime-env`, `python=3.11` |
| `07_scale_review.ipynb` | CURRENT | 2026-09-02 | `python=3.11` |
| `08_run_and_monitor.ipynb` | CURRENT | 2026-09-02 | `serverless_deps=container-image`, `python=3.11`, `fleet_sizing=derived-overlay` |
| `09_review_run.ipynb` | CURRENT | 2026-09-02 | `python=3.11` |
| `model_playground.ipynb` | CURRENT | 2026-09-02 | `python=3.11` |

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
| Airflow DAG emitter (`airflow_emit`) | NEVER_RUN | The renderer is offline-proven (emitted source compiles; `DagBag` parse test). No run has ever been orchestrated by Composer — that is smoke 15. |
| RuntimeProbe read path (P1–P4) | CURRENT | First live probe 2026-09-02 against `wave-62-mixed-runtimes-cpu-a7d04b6a9c8e` mid-flight: correct `TRUST_REGISTRY` + done/expected for the three terminal families, and a correct refusal on the running Ray one. `RayProbe.check` was then driven live out-of-process against that job and returned `RUNNING` — after the missing `_init_vertex` was fixed. The handle fix then landed and was re-proven live the same day: `--probe` against `ray-dl-on-cpu-probe-2e8a9f3f5c8d`, a single-family ephemeral Ray run, escalated out-of-process and returned `RUNNING_CONFIRMED`. Scope is now the whole verb, on every runtime. |
| RuntimeProbe cancel (P5) | CURRENT | A real Ray job was stopped live 2026-09-02 (`RayProbe.cancel` → `stopped: True`, job reached `STOPPED`, the run's own poll loop saw it and unwound). The **data-integrity property is proven by a genuine failure**: when `--cancel --force` could not reach that family, the registry was *not* marked CANCELLED. **Re-run live 2026-09-02 against a purpose-built single-family Ray job and the verb now reaches it** (`deep_learning cancelled — ray job stop issued`, count line correct at `1 of 1`, launcher unwound and tore the cluster down, teardown REST-verified). **New scope: the cancellation does not survive.** The launcher finalized the run `FAILED` 17 s after the cancel wrote `header=CANCELLED`, so the registry cannot distinguish a deliberate stop from a crash. See below. |
| Custom IAM roles (P6) | CURRENT | Applied live 2026-09-01: `projects/statmike-scale-forecasting/roles/sfProbeReader` and `roles/sfJobCanceller` now exist. Until then they had only ever been `validate`-clean. Creation is not use — that the permission sets are *sufficient* for a probe or a cancel is the P1–P5 rows below, not this one. |
| Registry ops (`registry.ops`) | CURRENT | All six `@gcp` tests in `tests/integration/test_registry_ops_live.py` pass 2026-09-02 — artifact-prefix delete correctly scoped in real GCS, `CREATE SNAPSHOT TABLE` valid against the real schema (native `JSON` columns included), `doctor`, `drop_run` preview, `drop_run` execute across every tier. One of the six had rotted and had to be repaired first — see below. **Scope: six of the seven verbs.** |
| Registry ops — `close_runs` (7th verb) | CURRENT | Executed live 2026-09-02 against the real registry: closed 9 of the 10 stuck headers to `FAILED` and skipped the tenth with its reason, leaving `doctor` reporting exactly one in-flight run. **The first live call failed** on a column that does not exist, which no offline test could have caught — see below. |
| Run audit principal (P6) | NEEDS_RECHECK | It has now executed, live, under ADC — and produced `actor=None`. The audit line for a real cancel attempt carried no principal. Whether that is a resolver defect or the expected ADC answer for this credential type is unresolved; either way the audit trail was empty when it mattered. See below. |

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
ADC — and returned nothing, so the audit line for a real cancel attempt names no one. Recorded as
`NEEDS_RECHECK` rather than a defect because it has not been established whether ADC user
credentials are expected to yield a principal here; what *is* established is that the audit trail was
empty on the one occasion it was exercised.

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
hiding it.** Recorded as a gap below rather than patched mid-campaign.

`cancelled_by: null` in the persisted telemetry is the same P6 `actor=None` finding as above,
now confirmed to propagate into stored state rather than only into the console line.

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

## Known validation gaps

Things that are true today and that no entry above covers. Keep this list short and act on it.

- **A run that needs both a CPU and a GPU Dataproc cluster has never been executed.** A Dataproc
  cluster has one worker machine type, so as of 2026-09-02 a run's ephemeral cluster families are
  grouped by hardware and get one right-sized cluster each — `sf-cluster-<run_id>-cpu` alongside
  `sf-cluster-<run_id>-gpu`. **No row above goes stale, and until smoke 16 none of them reached the
  new branch either**: smoke 04 was the only config with two ephemeral cluster families and both are
  CPU, so it takes the single-group path and is byte-identical to what it was — same one cluster,
  same unsuffixed name, same sizing. `16_cluster_split_hardware.json` was added to close the config
  half of the gap (a `statistical` CPU family and a `deep_learning` T4 family, both
  `spark_mode: cluster`), and `tests/smokes/test_smoke_configs.py` now fails if the library ever
  stops containing one. What remains unproven is the live half: the second create, the two distinct
  names, the per-cluster region after a capacity hop, and the two teardowns. Offline tests pin all
  of it, including the partial-create unwind; that is not the same as having run it.
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
- **A cancellation does not survive the launcher's unwind: the run ends `FAILED`, not `CANCELLED`.**
  Found live 2026-09-02 on `ray-cancel-probe-e22e6fe9a830`, analysed above. `--cancel --force` stops
  the job and writes `header=CANCELLED`; the launching process then sees its own job go `STOPPED`,
  treats that as death, and finalizes the run `FAILED` 17 s later. The evidence is not destroyed —
  `job_telemetry.cancel` keeps `cancelled_at`, `native_state_at_cancel` and `n_done_at_cancel` — but
  it is demoted out of every surface that reads `status`, so a deliberate stop is indistinguishable
  from a crash in the header, in `doctor`, and in any status filter. The fix is one rule —
  **a cancellation is sticky against the failure it caused** — and it is offline-testable once
  stated, though not offline-*findable*: it needs two processes disagreeing about one run, and until
  the handle fix landed the CLI could not reach far enough to create the disagreement.
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
