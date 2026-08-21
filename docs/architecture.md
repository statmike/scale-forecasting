# Software architecture — how the system works

This is a code-reading guide for data scientists: how a run flows through the system, how the modules
call each other, how a model file gets discovered and executed, and how each runtime fans the work
out. Every module link points at the file you'd open next; the spine is small enough to read in an
afternoon.

The design bet is **one capability per file** and **one unit of work everywhere**. A single function —
[`worker.run_cell`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/worker.py) — fits, backtests, and predicts one
`(series, model)` cell, and it runs *identically* on your laptop, inside a Spark task, and inside a
Ray task. Everything else is plumbing that decides *which* cells run where and *where the results go*.

For the *what/why* of each config knob see
[configuration_reference.md](./configuration_reference.md); for the *tables* a run writes see
[output_schemas.md](./output_schemas.md). This doc is the *how*.

---

## The one-paragraph mental model

A run is **one JSON config**. [`main.run(cfg)`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/main.py) computes a
deterministic `run_id`, then resolves the config into an **execution DAG**: one **job per model
family** present in the config — `statistical`, `ml`, `deep_learning`, `native` — plus a downstream
`ensemble` node. Each Python family runs on **its own resolved runtime** (Spark *xor* Ray, chosen
*per family*); the `native` family runs as SQL in BigQuery. All the family jobs launch **in parallel
under one run header**, so a run's wall-clock is the *slowest* family, not the sum. Whichever runtime
a family lands on, it fans out that family's cells and calls the **same** `worker.run_cell` for each,
then writes results to BigQuery via the Storage Write API. When every family job has landed its base
predictions, the `ensemble` node blends them. Three analyst views read the tables back. That's the
whole system.

```
                         one JSON config
                               │
                        main.run(cfg)  ─────────  registry: one run_id, one header row
                               │
              dag.plan_dag(cfg): one job per family, all parallel
                               │
   ┌───────────┬───────────────┼───────────────┬───────────────┐
statistical    ml         deep_learning       native      (each on its
(spark|ray)  (spark|ray)   (spark|ray)       (BigQuery)    resolved runtime)
   │           │               │                │
   └─────── engine fans out cells ───────┐  bigquery_engine (BQML SQL)
                                         │      │
                         worker.run_cell ◄── THE unit of work
                                         │      │
                                         ▼      ▼
                    registry.bq.write_cells (Storage Write API)
       run_registry · run_jobs · forecast_metadata · forecast_predictions · backtest_oof
                               │
                    (all families joined & green)
                               │
                        ensemble node  ── ensemble_run  ── blends base predictions
                               │
          v_run_summary · v_run_jobs · v_model_leaderboard  (analyst views)
```

---

## Layer 1 — entrypoints (who starts a run)

There are three ways a run begins, all converging on the same engines.

| Entrypoint | File | What it is |
|-----------|------|------------|
| `main.run(cfg)` | [`main.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/main.py) | **The spine.** In-process orchestrator — owns the `run_id` and the header, plans the DAG, launches every family job in parallel, then runs the ensemble node. |
| `submit` | [`submit.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/submit.py) | Submit-side launcher for a **Spark** family: zip the code, stage the config to GCS, build + submit a Dataproc Serverless batch, stamp telemetry back. |
| `ray_submit` | [`ray_submit.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/ray_submit.py) | Submit-side launcher for a **Ray** family: plan a per-pool autoscaling cluster, create it (with region fallback), submit the Ray job, poll, tear down. |
| `playground` | [`playground.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/playground.py) | Local single-cell path — one `run_cell` on the driver, no cluster, no registry. The fastest way to see a model run. |

`main.run` orchestrates the DAG ([`main.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/main.py)):

1. **Plan the DAG** — `dag.plan_dag(cfg)` computes the `run_id` via
   [`registry.ids.make_run_id`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/registry/ids.py) and resolves the config into
   one `FamilyJob` per present family (each with its resolved per-family compute), plus whether the
   ensemble node runs.
2. **Write the header** once — `bq.run_header(..., manage=True)` writes one RUNNING row
   ([`registry/bq.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/registry/bq.py)) and finalizes it once at the end.
3. **Fan out in parallel** — a `ThreadPoolExecutor` launches each Python family job
   (`_launch_family_job`) on its own thread while the BigQuery-native family runs inline on the main
   thread (`_launch_native_job`). Every family job runs `manage_header=False` so **exactly one**
   header row exists per `run_id`, and each opens **its own** `run_jobs` row (contributor mode).
4. **Ensemble node** (if enabled, and only if every family job succeeded) —
   `_launch_ensemble_job(...)`, the run's final DAG node.
5. **Finalize** — `hdr.finalize(...)` writes the combined terminal status + wall-clock.

Each Python family's runtime dispatch lives in `_launch_family_job`
([`main.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/main.py)): it looks up the family's **resolved**
runtime (`job.compute.runtime` — Spark *xor* Ray, chosen per family) and calls
`get_submitter(runtime).launch(...)` (Layer 1½). An injected Spark session (e.g. notebook 01's Spark
Connect) makes a Spark family run **in-process** against that session instead of a remote batch,
using the identical engine code.

### Layer 1½ — the submitter seam

[`submitters.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/submitters.py) captures "how do I launch a Python family on
its runtime" as a `RuntimeSubmitter` protocol with one implementation per runtime — `SparkSubmitter`
(→ `submit.submit_batch`, a Dataproc batch) and `RaySubmitter` (→ `ray_submit.submit_ray`, a Vertex
Ray job). `get_submitter(runtime)` returns the right one, so `_launch_family_job` is a single dispatch
line — the family doesn't know how its runtime is provisioned.

---

## Layer 2 — the DAG planner (families → jobs)

[`dag.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/dag.py) is the pure, offline planner — no GCP, no clocks, so
the same config always plans the same DAG.

- `group_models_by_family(cfg)` walks `cfg.models`, asks each model `get_model(name).family`, and
  groups them into `statistical` / `ml` / `deep_learning` / `native` (config order preserved within a
  family). This is where the model list becomes a family map.
- `plan_dag(cfg)` turns that map into a `RunDag`: one `FamilyJob` per present family — each Python
  family carrying its **resolved** compute (`RunConfig.resolve_family_compute`), `native` carrying
  none (it always runs in BigQuery) — plus the shared `run_id` and the `ensemble_enabled` flag.
- `dag_nodes(run_dag)` resolves the DAG into its **nodes**: one `DagNode` per family job carrying its
  deterministic `job_key` (`registry.ids.make_job_key`, attempt 1) and resolved placement, plus — when
  ensembling is on — a downstream `ensemble` node that `depends_on` every family job. This is the
  offline "given a config, which jobs will run, under what ids, in what order" surface: the same
  `job_key`s the executor later stamps onto each platform job and its `run_jobs` row. The SDK's
  [`Forecaster.dag()`](./using_the_sdk.md) and the plan/stage manifest both expose it.

A model's **runtime** and its **family** are independent: `native` models declare
`runtime="bigquery"` and go to BigQuery; every other family runs its declared per-family runtime.
[`router.split_by_runtime`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/router.py) is the small helper that separates
Python models from BigQuery-native models (it keys off each model's own `.runtime`), used where a
plain Python-vs-native split is all that's needed.

---

## Layer 3 — engines (fanning out the cells)

Every engine reads the same source panel, fans out cells its own way, and calls the **same** unit of
work. The Spark and Ray engines even share the *exact same* per-cell driver — `spark_io.run_group` —
and the same writer — `bq.write_cells`. The only genuinely Ray-specific code is GPU/CPU routing,
cluster sizing, and chunking.

### Spark — the CPU workhorse

One on-cluster engine, sharing
[`spark_io.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/engines/spark_io.py):

| Engine | File | Fan-out unit |
|--------|------|-------------|
| `explode` | [`spark_explode.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/engines/spark_explode.py) | One Spark task per **`(series, model)` cell** — series are cross-joined with the model list, then bucketed on `[ts_id, model]`. A slow deep-learning cell occupies its own bucket while the series' fast cells run concurrently. The hero scale path. |

The common Spark shape (see `spark_explode.run`,
[`spark_explode.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/engines/spark_explode.py)):
`read_source_series` → `resolve_fleetwide_hpo` → `cross_join_models` → `add_bucket` →
`groupBy(bucket).applyInPandas(group_runner, …)` → `aggregate_status` → `update_header`. The
`group_runner` closure ([`spark_io.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/engines/spark_io.py)) is what each
Spark task actually executes: it calls `run_group` (which loops `run_cell` over the cells in the
bucket) and then `bq.write_cells` to persist them.

### Ray — the fractional-GPU path

[`ray_engine.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/engines/ray_engine.py) is the structural twin of
`spark_explode`, and it **re-exports** `spark_io.run_group` and `aggregate_status` verbatim
([`ray_io.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/engines/ray_io.py)) — the cell logic is identical; only the
fan-out mechanism differs. `run()` reads the panel to the driver → splits models into a **GPU pool**
(NeuralProphet) and a **CPU pool** (everything else) → calibrates the T4 `gpu_fraction` → chunks
cells → fans one `@ray.remote` task per chunk (`num_gpus=fraction` for GPU cells, `num_cpus=1`
otherwise) → `ray.get` → aggregate → update header. Each worker pool **autoscales** by default
between an independent `[min, max]` ([`ray_io.plan_cluster`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/engines/ray_io.py)
resolves the bounds; `ray_submit` attaches a Vertex `AutoscalingSpec` per pool) — so the CPU pool
grows to work through the queue and the expensive T4 pool shrinks when idle. Determinism is preserved
a level up: the *initial* size is a pure function of the fan-out (clamped into the bounds) and the
whole spec is hashed into `run_id` and stamped to telemetry. `ray_autoscale=false` restores the
fixed-size path.

When more than one family resolves to Ray in the same run, the orchestrator provisions **one shared
Ray cluster** for the launch block (`main._shared_ray_cluster`) and each Ray family submits its job
to it, instead of each family self-provisioning.

### BigQuery-native — SQL only

[`bigquery_engine.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/engines/bigquery_engine.py) runs `arima_plus` /
`arima_plus_xreg` / `timesfm` entirely inside BigQuery as BQML SQL — no Python compute — and writes
its metrics and predictions through the **same** Storage Write API path as the Python cells (it
reuses `bq._proto_for` / `_encode_rows` / `_append_via_write_api`). It honors holidays (for BQML
parity via `features.holiday_frame`) but not the Python target transform. It is the `native` family
job, running in parallel with the Python family jobs under the same `run_id`.

### The pure / I-O split

Both `spark_io.py` and `ray_io.py` are deliberately split so the *interesting* logic is
offline-testable without a cluster:

- **Pure** (no Spark, no Ray, no GCP): `run_group` (the per-cell loop), `aggregate_status` (COMPLETED
  / PARTIAL / FAILED roll-up), `bucket_target` / `plan_cluster` / `chunk_cells` / `calibrate_gpu_fraction`
  sizing math.
- **I-O**: reading the source table, cross-join, bucketing, and the group-runner closure that writes
  cells.

---

## Layer 4 — the unit of work (the heart)

[`worker.run_cell(series, model_name, cfg, params=None)`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/worker.py) is the
one function every engine calls, and the one to read first. It **never raises** — a failure becomes an
error `CellResult`, so one bad series can't sink a 100k run. Its steps
([`worker.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/worker.py)):

1. **Identity** — `make_run_id(cfg)` + `make_model_hash(...)` from
   [`registry/ids.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/registry/ids.py).
2. **Look up the model** — `get_model(model_name)` from
   [`models/`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/models/__init__.py) (see Layer 5).
3. **Resolve the transform** — `features.fit_transform_lambda` (fits per-series boxcox λ if asked),
   build a `ModelContext`.
4. **Resolve hyperparameters** — `_resolve_params`, which may call
   [`hpo.tune_model`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/hpo.py) for per-series HPO.
5. **Backtest** (if `cfg.backtest.enabled`) — `backtest.backtest_cell(...)` lays out the folds and
   fits a *fresh* model per fold ([`backtest.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/backtest.py)), scoring each
   with [`metrics.compute_metrics`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/metrics.py).
6. **Fit + predict** — `features.build_features` → `model.fit(y, X)` → `model.predict(horizon, …)`.
7. **Persist** (if `cfg.compute.persist_models`) — `model.serialize()` to a GCS artifact.

It returns a `CellResult` ([`worker.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/worker.py)) carrying the
predictions, OOF rows, metrics, best params, and fit time — the raw material the registry writes.

Its pure downstream helpers, each a single-capability file:
[`backtest.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/backtest.py) (fold layout),
[`features.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/features.py) (transform + feature matrix),
[`metrics.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/metrics.py) (the metric panel, shared by worker, backtest,
*and* the BigQuery engine).

---

## Layer 5 — the model system (add one file, it appears)

This is the part most data scientists will extend, so it's worth understanding precisely. See
[adding_a_model.md](./adding_a_model.md) for the how-to; this is the mechanism.

**The contract** — every model subclasses `BaseModel`
([`models/base_model.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/models/base_model.py)) and sets a few ClassVars
(`name`, `runtime`, `family`, `supports_exog`, …) and implements two methods:

- `fit(y, X)` — fit on the (transformed) target and optional feature frame.
- `predict(horizon, X, quantiles)` — return the forecast frame with intervals.
- *(optional)* `search_space(trial)` — declare the HPO space; `serialize()` — override the default
  pickle.

**The registry** — a module-level dict `_REGISTRY` and a `register(model_cls)` function
([`base_model.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/models/base_model.py)). Each model file ends by calling
`register(MyModel)`, which stores it by its `name` (rejecting duplicates).

**Discovery** — [`models/__init__.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/models/__init__.py) imports every
model module for its `register()` **side effect**, then exposes `get_model(name)` and `list_models()`.
So dropping a new `models/foo.py` with one import line in `__init__.py` makes `foo` show up everywhere:
`playground --list`, the DAG planner's family grouping, the leaderboard — no other wiring.

**Who reads what** off a model class:

- `dag.group_models_by_family` and `router.split_by_runtime` read `.family` and `.runtime` to decide
  which family job a model lands in and whether that job runs in Python or BigQuery.
- `ray_io.split_gpu_cpu_models` reads `.family` (`deep_learning` → the GPU pool).
- `worker.run_cell` instantiates it and calls `fit`/`predict`.

**BigQuery-native models are metadata shims** —
[`bigquery_native.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/models/bigquery_native.py) registers `arima_plus`,
`arima_plus_xreg`, and `timesfm` as `BaseModel` subclasses with `runtime="bigquery"` and
`family="native"` whose `fit`/`predict` *raise* (they never run in Python). Their registration exists
purely so the planner can *see* them and route them to `bigquery_engine`. This is why the families
compose so cleanly — a native model is just a model with a different declared runtime.

---

## Layer 6 — the registry (where results land)

[`registry/`](https://github.com/statmike/scale-forecasting/tree/main/src/scale_forecasting/registry) is the persistence boundary — pure row assemblers
plus Storage Write API appends. Five native-BigQuery tables, written idempotently (append +
dedupe-on-read):

| Table | Tier | Written by |
|-------|------|-----------|
| `run_registry` | the header — config + telemetry | `bq.run_header` / `update_header` (once per run) |
| `run_jobs` | per-family-job row — runtime, hardware, system job id, status, telemetry | `bq.run_job` (once per family job + the ensemble) |
| `forecast_metadata` | per-cell metrics + artifact links | `bq.write_cells` (executor-side) |
| `forecast_predictions` | the forecast values | `bq.write_cells` |
| `backtest_oof` | out-of-fold rows for learned ensembling | `bq.write_cells` |

The files:

- [`bq.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/registry/bq.py) — the write path: pure `assemble_*_rows`
  functions, the Storage Write API encoder (`_proto_for` / `_encode_rows` / `_append_via_write_api`),
  and the header, `run_job`, and `write_cells` lifecycles. **The reusable seam**: `write_cells` is
  called executor-side by *both* the Spark group-runner and the Ray chunk-runner, so results stream to
  BigQuery in bulk from the workers — parallelism is bounded by compute, not a tracking server's QPS.
- [`ddl.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/registry/ddl.py) — the table definitions (single source of truth
  for the schema), rendered and executed by `bq.ensure_tables` at run time. Terraform owns the
  *containers*; the app owns the *tables*.
- [`ids.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/registry/ids.py) — `make_run_id(cfg)` = `<run-name-slug>-<12-hex
  digest of the canonical config>` and `make_job_key(run_id, family, attempt)` = the canonical
  per-family job id. Deterministic: the same config always yields the same `run_id`, so re-runs and
  multi-family runs collide by design (idempotency).
- [`artifacts.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/registry/artifacts.py) — GCS upload for serialized models
  (lineage).
- [`views.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/registry/views.py) — the three analyst views: `v_run_summary`
  (per-run scaling/efficiency, unpacking the telemetry JSON), `v_run_jobs` (the per-family-job trace —
  latest attempt per family, its runtime/hardware/system job id/status/telemetry), and
  `v_model_leaderboard` (per-model accuracy). These are what notebook 07 reads.

---

## Layer 7 — per-job identity (one run, many systems)

A run fans across several platforms — Dataproc, Vertex Ray, BigQuery — but every family job keeps one
identity that ties its platform job, its `run_jobs` row, and its offline plan together:

- **The canonical key** — `registry.ids.make_job_key(run_id, family, attempt)` →
  `sf-<run_id>-<family>-a<n>`. This is the one name the DAG plans (`dag_nodes`), the executor stamps,
  and a trace keys on. It lands in `run_jobs.job_id`.
- **The system id** — `main._system_job_id(job_key, runtime)` maps the canonical key to each
  platform's legal charset/length: `dataproc_job_id` (Spark), `ray_submission_id` (Ray),
  `bigquery_job_id` (native / ensemble). It lands in `run_jobs.system_job_id`, so you can jump from a
  run's trace straight to the platform console.
- **Attempts** — `bq.next_job_attempt(run_id, family, force=…)` bumps the attempt so a `--force`
  re-run is a fresh, distinctly-keyed job under the same `run_id`; `v_run_jobs` surfaces the latest
  attempt per family.

The offline `dag_nodes` and the executed `run_jobs`/`v_run_jobs` are two views of the *same* map: what
*will* run (from the config alone) and what *did* run (from BigQuery). The SDK exposes both —
[`Forecaster.dag()`](./using_the_sdk.md) (planned nodes) and `Forecaster.jobs()` (executed
`JobTrace`s).

---

## Layer 8 — the identity seam (same code, everywhere)

The "same code local ↔ cluster" guarantee rests on three small files:

- [`settings.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/settings.py) — `Settings.resolve()` reads the `SF_*` env
  (`SF_PROJECT_ID`, `SF_CONNECTION`, `SF_WAREHOUSE_URI`, …) into one frozen object. Every engine and
  the registry resolve identity the same way, whether on your laptop (ADC) or on a cluster.
- [`config.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/config.py) — loads, validates (strict, `extra="forbid"`), and
  freezes the `RunConfig`, and resolves each family's compute (`resolve_family_compute`). The
  normalized config is logged verbatim to `run_registry.raw_config`, so the config *is* the experiment
  record.
- [`_infra_args.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/_infra_args.py) — carries the `Settings` across the
  process boundary as `--sf-*` CLI args (Dataproc/Ray reject driver env), then re-exports them to env
  on the cluster *before* `Settings.resolve`. `infra_args_from(settings)` builds them submit-side;
  `export_infra_env(ns)` reads them on-cluster.

On-cluster, the batch/job driver lands in a thin entrypoint —
[`spark_entry.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/spark_entry.py) or
[`ray_entry.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/ray_entry.py) — which exports the infra env, loads the config
from its GCS URI, and dispatches to the named engine's `run()`. That's the whole boundary: the same
`run()` you can call in a notebook is what the cluster calls.

---

## The full call tree

```
CLI / notebook / Airflow / SDK
  main.run(cfg)                                            main.py
    dag.plan_dag → make_run_id (ids.py) + group_models_by_family + resolve_family_compute
    bq.run_header (RUNNING, one shared header)             registry/bq.py   (ddl.py, views.py)
    ├─ [thread] per Python family — _launch_family_job → get_submitter(runtime).launch
    │    ├─ SparkSubmitter → submit.submit_batch → spark_entry.main → spark_explode.run
    │    │      spark_io: cross_join_models · add_bucket · make_group_runner
    │    │      applyInPandas → spark_io.run_group → worker.run_cell → bq.write_cells
    │    ├─ RaySubmitter → ray_submit.submit_ray → ray_entry.main → ray_engine.run
    │    │      ray_io: split_gpu_cpu_models · plan_cluster · chunk_cells · make_chunk_runner
    │    │      make_chunk_runner → spark_io.run_group → worker.run_cell → bq.write_cells
    │    └─ injected Spark session → spark_explode.run   (in-process, e.g. NB01)
    │        (each family opens its own run_jobs row via bq.run_job)
    ├─ [inline] native family — _launch_native_job → bigquery_engine.run   (BQML SQL → bq write-api)
    └─ (all families join & green) ensemble node — _launch_ensemble_job → ensemble_run.run_ensembles
    hdr.finalize (combined status + wall-clock)            registry/bq.py

worker.run_cell  ── THE unit of work ──                     worker.py
    ├─ features.fit_transform_lambda / build_features       features.py
    ├─ hpo.tune_model            (per-series HPO)           hpo.py
    ├─ backtest.backtest_cell → model.fit/predict + metrics.compute_metrics
    ├─ models.get_model → BaseModel.fit / predict / serialize
    └─ registry.ids.make_run_id / make_model_hash → CellResult

playground.run_model → worker.run_cell                      (local, no cluster, no registry)
```

**The reuse seams to notice:** `spark_io.run_group`, `bq.write_cells`, and `aggregate_status` are
shared **verbatim** by Spark and Ray (`ray_io` re-exports them), and the `manage=True` header opened
by `main.run` threads `manage_header=False` into every family job so exactly one header row exists per
`run_id` while each family keeps its own `run_jobs` row. Those facts are what make "same code
everywhere, one job per family, one run" real.

---

## Where to start reading

1. [`config.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/config.py) — the run contract (what a run *is*).
2. [`worker.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/worker.py) — the unit of work (what actually runs per cell).
3. [`models/base_model.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/models/base_model.py) — the model interface + registry.
4. [`dag.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/dag.py) — how a config becomes a set of parallel family jobs.
5. [`main.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/main.py) — how it's all orchestrated.
6. Then one engine — [`spark_explode.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/engines/spark_explode.py) — to see
   the fan-out, and [`registry/bq.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/registry/bq.py) to see the write path.

## See also

- [configuration_reference.md](./configuration_reference.md) — every config field and option value.
- [adding_a_model.md](./adding_a_model.md) — add a model in one file (the Layer 5 how-to).
- [output_schemas.md](./output_schemas.md) — the registry tables' column-by-column layout.
- [running_and_reviewing.md](./running_and_reviewing.md) — submit, watch, and review a run.
- [editing_code_without_rebuilding.md](./editing_code_without_rebuilding.md) — why a code edit ships on
  the next run with no image rebuild (the runtime code-delivery seam).
</content>
</invoke>
