# Software architecture — how the system works

This is a code-reading guide for data scientists: how a run flows through the system, how the modules
call each other, how a model file gets discovered and executed, and how each runtime fans the work
out. Every module link points at the file you'd open next; the spine is small enough to read in an
afternoon.

The design bet is **one capability per file** and **one unit of work everywhere**. A single function —
[`worker.run_cell`](../src/scale_forecasting/worker.py) — fits, backtests, and predicts one
`(series, model)` cell, and it runs *identically* on your laptop, inside a Spark task, and inside a
Ray task. Everything else is plumbing that decides *which* cells run where and *where the results go*.

For the *what/why* of each config knob see
[configuration_reference.md](./configuration_reference.md); for the *tables* a run writes see
[output_schemas.md](./output_schemas.md). This doc is the *how*.

---

## The one-paragraph mental model

A run is **one JSON config**. [`main.run(cfg)`](../src/scale_forecasting/main.py) computes a
deterministic `run_id`, splits the model list into *Python* models and *BigQuery-native* models, and
launches both **in parallel under one run header**. The Python models go to **one** runtime (Spark
*xor* Ray); the BigQuery-native models run as SQL in BigQuery. Whichever engine runs, it fans out the
work and calls the **same** `worker.run_cell` for each cell, then writes results to three BigQuery
tables via the Storage Write API. Two analyst views read those tables back. That's the whole system.

```
                         one JSON config
                               │
                        main.run(cfg)  ──────────  registry: one run_id, one header row
                               │
             ┌─────────────────┴─────────────────┐
      Python models                         BigQuery-native models
   (Spark XOR Ray — pick one)              (arima_plus, timesfm — SQL)
             │                                     │
     engine fans out cells                  bigquery_engine (BQML)
             │                                     │
      worker.run_cell  ◄── THE unit of work        │
             │                                     │
             └──────────────┬──────────────────────┘
                            ▼
             registry.bq.write_cells (Storage Write API)
             run_registry · forecast_metadata · forecast_predictions · backtest_oof
                            │
                   v_run_summary · v_model_leaderboard  (analyst views)
```

---

## Layer 1 — entrypoints (who starts a run)

There are three ways a run begins, all converging on the same engines.

| Entrypoint | File | What it is |
|-----------|------|------------|
| `main.run(cfg)` | [`main.py`](../src/scale_forecasting/main.py) | **The spine.** In-process orchestrator — owns the `run_id` and the header, launches the Python runtime and BigQuery in parallel, runs ensembles at the end. |
| `submit` / `submit_multi` | [`submit.py`](../src/scale_forecasting/submit.py) | Submit-side launcher for **Spark**: zip the code, stage the config to GCS, build + submit a Dataproc Serverless batch, stamp telemetry back. |
| `ray_submit` | [`ray_submit.py`](../src/scale_forecasting/ray_submit.py) | Submit-side launcher for **Ray**: plan a fixed-size cluster, create it (with region fallback), submit the Ray job, poll, tear down. |
| `playground` | [`playground.py`](../src/scale_forecasting/playground.py) | Local single-cell path — one `run_cell` on the driver, no cluster, no registry. The fastest way to see a model run. |

`main.run` does the routing ([`main.py:173`](../src/scale_forecasting/main.py)):

1. **Plan** — `_plan(cfg)` computes the `run_id` via
   [`registry.ids.make_run_id`](../src/scale_forecasting/registry/ids.py) and splits the models with
   [`router.split_by_runtime`](../src/scale_forecasting/router.py).
2. **Write the header** once — `bq.ensure_tables` + `bq.write_header`
   ([`registry/bq.py`](../src/scale_forecasting/registry/bq.py)).
3. **Fan out in parallel** — a `ThreadPoolExecutor` runs the Python runtime while
   `bigquery_engine.run(...)` runs inline. Both are passed `manage_header=False` so **exactly one**
   header row exists per `run_id` across all runtimes.
4. **Ensemble** (if enabled) — `ensemble_run.run_ensembles(...)`.
5. **Finalize** — `bq.update_header` with the terminal status.

The Python-runtime dispatch inside `_launch_python_runtime`
([`main.py:117`](../src/scale_forecasting/main.py)):

- `python_runtime == "ray"` → `ray_submit.submit_ray(...)`
- an **injected** Spark session (e.g. notebook 01's Spark Connect) → `spark_explode.run` /
  `spark_naive.run` in-process
- otherwise → `submit.submit_batch(...)` (a Dataproc batch)

---

## Layer 2 — the router (splitting the model list)

[`router.py`](../src/scale_forecasting/router.py) is 41 lines and does exactly one thing:
`split_by_runtime(cfg)` walks `cfg.models`, asks each model
`get_model(name).runtime` ([`router.py:37`](../src/scale_forecasting/router.py)), and returns
`(python_models, bq_models)`. A model whose `runtime == "bigquery"` goes to the SQL engine; everything
else is a Python model. This is the *only* place the runtime split happens — and it keys off the
**model's own declared runtime**, not the config, which is why `arima_plus` and `theta` can sit in one
`models` list and transparently run in two engines under one `run_id`.

---

## Layer 3 — engines (fanning out the cells)

Every engine reads the same source panel, fans out cells its own way, and calls the **same** unit of
work. The Spark and Ray engines even share the *exact same* per-cell driver — `spark_io.run_group` —
and the same writer — `bq.write_cells`. The only genuinely Ray-specific code is GPU/CPU routing,
cluster sizing, and chunking.

### Spark — the CPU workhorse

Two on-cluster engines plus a submit-side orchestrator, all sharing
[`spark_io.py`](../src/scale_forecasting/engines/spark_io.py):

| Engine | File | Fan-out unit |
|--------|------|-------------|
| `explode` | [`spark_explode.py`](../src/scale_forecasting/engines/spark_explode.py) | One Spark task per **`(series, model)` cell** — series are cross-joined with the model list, then bucketed on `[ts_id, model]`. A slow deep-learning cell occupies its own bucket while the series' fast cells run concurrently. The hero scale path. |
| `naive` | [`spark_naive.py`](../src/scale_forecasting/engines/spark_naive.py) | One task per **whole series**; every model for that series runs sequentially in the task. One slow model blocks the rest — the deliberate straggler demo. |
| `multi` | [`spark_multi.py`](../src/scale_forecasting/engines/spark_multi.py) | Not an on-cluster engine — `run()` just raises. `multi` is orchestrated **submit-side** by [`submit.submit_multi`](../src/scale_forecasting/submit.py), which fires one child `explode` batch per model family under one shared `run_id`. |

The common Spark shape (see `spark_explode.run`,
[`spark_explode.py:35`](../src/scale_forecasting/engines/spark_explode.py)):
`read_source_series` → `resolve_fleetwide_hpo` → `cross_join_models` → `add_bucket` →
`groupBy(bucket).applyInPandas(group_runner, …)` → `aggregate_status` → `update_header`. The
`group_runner` closure ([`spark_io.py:344`](../src/scale_forecasting/engines/spark_io.py)) is what each
Spark task actually executes: it calls `run_group` (which loops `run_cell` over the cells in the
bucket) and then `bq.write_cells` to persist them.

### Ray — the fractional-GPU path

[`ray_engine.py`](../src/scale_forecasting/engines/ray_engine.py) is the structural twin of
`spark_explode`, and it **re-exports** `spark_io.run_group` and `aggregate_status` verbatim
([`ray_io.py:44`](../src/scale_forecasting/engines/ray_io.py)) — the cell logic is identical; only the
fan-out mechanism differs. `run()` ([`ray_engine.py:223`](../src/scale_forecasting/engines/ray_engine.py)):
read the panel to the driver → split models into a **GPU pool** (NeuralProphet) and a **CPU pool**
(everything else) → calibrate the T4 `gpu_fraction` → chunk cells → fan one `@ray.remote` task per
chunk (`num_gpus=fraction` for GPU cells, `num_cpus=1` otherwise) → `ray.get` → aggregate → update
header. The cluster is **fixed-size** (planned up front by
[`ray_io.plan_cluster`](../src/scale_forecasting/engines/ray_io.py)), not autoscaling — a forecast's
fan-out is known before it starts.

### BigQuery-native — SQL only

[`bigquery_engine.py`](../src/scale_forecasting/engines/bigquery_engine.py) runs `arima_plus` /
`timesfm` entirely inside BigQuery as BQML SQL — no Python compute — and writes its metrics and
predictions through the **same** Storage Write API path as the Python cells (it reuses `bq._proto_for`
/ `_encode_rows` / `_append_via_write_api`). It honors holidays (for BQML parity via
`features.holiday_frame`) but not the Python target transform. It always runs in parallel with the
Python runtime, under the same `run_id`.

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

[`worker.run_cell(series, model_name, cfg, params=None)`](../src/scale_forecasting/worker.py) is the
one function every engine calls, and the one to read first. It **never raises** — a failure becomes an
error `CellResult`, so one bad series can't sink a 100k run. Its steps
([`worker.py:129`](../src/scale_forecasting/worker.py)):

1. **Identity** — `make_run_id(cfg)` + `make_model_hash(...)` from
   [`registry/ids.py`](../src/scale_forecasting/registry/ids.py).
2. **Look up the model** — `get_model(model_name)` from
   [`models/`](../src/scale_forecasting/models/__init__.py) (see Layer 5).
3. **Resolve the transform** — `features.fit_transform_lambda` (fits per-series boxcox λ if asked),
   build a `ModelContext`.
4. **Resolve hyperparameters** — `_resolve_params`, which may call
   [`hpo.tune_model`](../src/scale_forecasting/hpo.py) for per-series HPO.
5. **Backtest** (if `cfg.backtest.enabled`) — `backtest.backtest_cell(...)` lays out the folds and
   fits a *fresh* model per fold ([`backtest.py`](../src/scale_forecasting/backtest.py)), scoring each
   with [`metrics.compute_metrics`](../src/scale_forecasting/metrics.py).
6. **Fit + predict** — `features.build_features` → `model.fit(y, X)` → `model.predict(horizon, …)`.
7. **Persist** (if `cfg.compute.persist_models`) — `model.serialize()` to a GCS artifact.

It returns a `CellResult` ([`worker.py:36`](../src/scale_forecasting/worker.py)) carrying the
predictions, OOF rows, metrics, best params, and fit time — the raw material the registry writes.

Its pure downstream helpers, each a single-capability file:
[`backtest.py`](../src/scale_forecasting/backtest.py) (fold layout),
[`features.py`](../src/scale_forecasting/features.py) (transform + feature matrix),
[`metrics.py`](../src/scale_forecasting/metrics.py) (the 11-metric panel, shared by worker, backtest,
*and* the BigQuery engine).

---

## Layer 5 — the model system (add one file, it appears)

This is the part most data scientists will extend, so it's worth understanding precisely. See
[adding_a_model.md](./adding_a_model.md) for the how-to; this is the mechanism.

**The contract** — every model subclasses `BaseModel`
([`models/base_model.py`](../src/scale_forecasting/models/base_model.py)) and sets a few ClassVars
(`name`, `runtime`, `family`, `supports_exog`, …) and implements two methods:

- `fit(y, X)` — fit on the (transformed) target and optional feature frame.
- `predict(horizon, X, quantiles)` — return the forecast frame with intervals.
- *(optional)* `search_space(trial)` — declare the HPO space; `serialize()` — override the default
  pickle.

**The registry** — a module-level dict `_REGISTRY` and a `register(model_cls)` function
([`base_model.py:62`](../src/scale_forecasting/models/base_model.py)). Each model file ends by calling
`register(MyModel)`, which stores it by its `name` (rejecting duplicates).

**Discovery** — [`models/__init__.py`](../src/scale_forecasting/models/__init__.py) imports every
model module for its `register()` **side effect**, then exposes `get_model(name)` and `list_models()`.
So dropping a new `models/foo.py` with one import line in `__init__.py` makes `foo` show up everywhere:
`playground --list`, the router, the family split, the leaderboard — no other wiring.

**Who reads what** off a model class:

- `router.split_by_runtime` reads `.runtime` (`"bigquery"` vs Python).
- `submit.split_models_by_family` and `ray_io.split_gpu_cpu_models` read `.family`
  (`deep_learning` → the GPU pool / its own `multi` batch).
- `worker.run_cell` instantiates it and calls `fit`/`predict`.

**BigQuery-native models are metadata shims** —
[`bigquery_native.py`](../src/scale_forecasting/models/bigquery_native.py) registers `arima_plus` and
`timesfm` as `BaseModel` subclasses with `runtime="bigquery"` whose `fit`/`predict` *raise* (they
never run in Python). Their registration exists purely so the router can *see* them and route them to
`bigquery_engine`. This is why the two runtimes compose so cleanly — a native model is just a model
with a different declared runtime.

---

## Layer 6 — the registry (where results land)

[`registry/`](../src/scale_forecasting/registry/) is the persistence boundary — pure row assemblers
plus Storage Write API appends. Four native-BigQuery tables, written idempotently (append +
dedupe-on-read):

| Table | Tier | Written by |
|-------|------|-----------|
| `run_registry` | the header — config + telemetry | `bq.write_header` / `update_header` (once per run) |
| `forecast_metadata` | per-cell metrics + artifact links | `bq.write_cells` (executor-side) |
| `forecast_predictions` | the forecast values | `bq.write_cells` |
| `backtest_oof` | out-of-fold rows for learned ensembling | `bq.write_cells` |

The files:

- [`bq.py`](../src/scale_forecasting/registry/bq.py) — the write path: pure `assemble_*_rows`
  functions, the Storage Write API encoder (`_proto_for` / `_encode_rows` / `_append_via_write_api`),
  and the header + `write_cells` lifecycle. **The reusable seam**: `write_cells` is called executor-side
  by *both* the Spark group-runner and the Ray chunk-runner, so results stream to BigQuery in bulk from
  the workers — parallelism is bounded by compute, not a tracking server's QPS.
- [`ddl.py`](../src/scale_forecasting/registry/ddl.py) — the table definitions (single source of truth
  for the schema), rendered and executed by `bq.ensure_tables` at run time. Terraform owns the
  *containers*; the app owns the *tables*.
- [`ids.py`](../src/scale_forecasting/registry/ids.py) — `make_run_id(cfg)` = `<run-name-slug>-<12-hex
  digest of the canonical config>`. Deterministic: the same config always yields the same `run_id`, so
  re-runs and mixed-runtime runs collide by design (idempotency).
- [`artifacts.py`](../src/scale_forecasting/registry/artifacts.py) — GCS upload for serialized models
  (lineage).
- [`views.py`](../src/scale_forecasting/registry/views.py) — the two analyst views: `v_run_summary`
  (per-run scaling/efficiency, unpacking the telemetry JSON) and `v_model_leaderboard` (per-model
  accuracy). These are what notebook 07 reads.

---

## Layer 7 — the identity seam (same code, everywhere)

The "same code local ↔ cluster" guarantee rests on three small files:

- [`settings.py`](../src/scale_forecasting/settings.py) — `Settings.resolve()` reads the `SF_*` env
  (`SF_PROJECT_ID`, `SF_CONNECTION`, `SF_WAREHOUSE_URI`, …) into one frozen object. Every engine and
  the registry resolve identity the same way, whether on your laptop (ADC) or on a cluster.
- [`config.py`](../src/scale_forecasting/config.py) — loads, validates (strict, `extra="forbid"`), and
  freezes the `RunConfig`. The normalized config is logged verbatim to `run_registry.raw_config`, so
  the config *is* the experiment record.
- [`_infra_args.py`](../src/scale_forecasting/_infra_args.py) — carries the `Settings` across the
  process boundary as `--sf-*` CLI args (Dataproc/Ray reject driver env), then re-exports them to env
  on the cluster *before* `Settings.resolve`. `infra_args_from(settings)` builds them submit-side;
  `export_infra_env(ns)` reads them on-cluster.

On-cluster, the batch/job driver lands in a thin entrypoint —
[`spark_entry.py`](../src/scale_forecasting/spark_entry.py) or
[`ray_entry.py`](../src/scale_forecasting/ray_entry.py) — which exports the infra env, loads the config
from its GCS URI, and dispatches to the named engine's `run()`. That's the whole boundary: the same
`run()` you can call in a notebook is what the cluster calls.

---

## The full call tree

```
CLI / notebook / Airflow
  main.run(cfg)                                            main.py:173
    _plan → make_run_id (ids.py) + router.split_by_runtime (router.py)
    bq.ensure_tables / bq.write_header                     registry/bq.py   (ddl.py, views.py)
    ├─ [thread] Python runtime — one of:
    │    ├─ ray_submit.submit_ray → ray_entry.main → ray_engine.run          ray_engine.py:223
    │    │      ray_io: split_gpu_cpu_models · plan_cluster · chunk_cells · make_chunk_runner
    │    │      make_chunk_runner → spark_io.run_group → worker.run_cell → bq.write_cells
    │    ├─ submit.submit_batch → spark_entry.main → spark_explode.run / spark_naive.run
    │    │      spark_io: cross_join_models · add_bucket · make_group_runner
    │    │      applyInPandas → spark_io.run_group → worker.run_cell → bq.write_cells
    │    │      (submit.submit_multi: one explode batch per model family, one run_id)
    │    └─ injected Spark session → spark_explode.run / spark_naive.run  (in-process, e.g. NB01)
    └─ [inline] bigquery_engine.run(cfg, bq_models)         bigquery_engine.py:528  (BQML SQL → bq write-api)
    if ensemble.enabled → ensemble_run.run_ensembles        main.py:269
    bq.update_header                                        registry/bq.py

worker.run_cell  ── THE unit of work ──                     worker.py:129
    ├─ features.fit_transform_lambda / build_features       features.py
    ├─ hpo.tune_model            (per-series HPO)           hpo.py
    ├─ backtest.backtest_cell → model.fit/predict + metrics.compute_metrics
    ├─ models.get_model → BaseModel.fit / predict / serialize
    └─ registry.ids.make_run_id / make_model_hash → CellResult

playground.run_model → worker.run_cell                      (local, no cluster, no registry)
```

**The two reuse seams to notice:** `spark_io.run_group`, `bq.write_cells`, and `aggregate_status` are
shared **verbatim** by Spark and Ray (`ray_io` re-exports them), and the `manage_header=False` flag
threads from `main.run` into every engine so exactly one header row exists per `run_id` across all
three runtimes. Those two facts are what make "same code everywhere, three runtimes, one run" real.

---

## Where to start reading

1. [`config.py`](../src/scale_forecasting/config.py) — the run contract (what a run *is*).
2. [`worker.py`](../src/scale_forecasting/worker.py) — the unit of work (what actually runs per cell).
3. [`models/base_model.py`](../src/scale_forecasting/models/base_model.py) — the model interface + registry.
4. [`main.py`](../src/scale_forecasting/main.py) — how it's all orchestrated.
5. Then one engine — [`spark_explode.py`](../src/scale_forecasting/engines/spark_explode.py) — to see
   the fan-out, and [`registry/bq.py`](../src/scale_forecasting/registry/bq.py) to see the write path.

## See also

- [configuration_reference.md](./configuration_reference.md) — every config field and option value.
- [adding_a_model.md](./adding_a_model.md) — add a model in one file (the Layer 5 how-to).
- [output_schemas.md](./output_schemas.md) — the registry tables' column-by-column layout.
- [running_and_reviewing.md](./running_and_reviewing.md) — submit, watch, and review a run.
- [editing_code_without_rebuilding.md](./editing_code_without_rebuilding.md) — why a code edit ships on
  the next run with no image rebuild (the runtime code-delivery seam).
</content>
</invoke>
