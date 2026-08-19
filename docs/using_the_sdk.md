# Using the SDK

There are two ways to use `scale_forecasting` from Python, and they share one engine underneath:

1. **The easy path** — the `Forecaster` facade. Point it at a config, call `run()`. It wraps the
   exact same orchestration the CLI and Composer run, so an SDK run is byte-for-byte a CLI run.
2. **The direct path** — drive Spark or Ray yourself and call the model machinery directly, bypassing
   the SDK entirely while reusing the *same* per-cell code. This is for when you already have a
   `SparkSession` or a Ray cluster and want to embed forecasting into your own job.

Both paths run the identical unit of work (`run_cell`), so results are the same whichever door you
come in. Everything below imports from the top-level package:

```python
import scale_forecasting as sf
```

`import scale_forecasting` is near-instant — the heavy model modules (statsmodels, xgboost, …) load
lazily, only when you first touch a name that runs models (`Forecaster`, `run`, `run_cell`, …).
Touching `RunConfig`/`Settings` alone pays no model-import cost.

---

## The easy path: `Forecaster`

```python
import scale_forecasting as sf

# Build from a JSON config file, a dict, or an in-memory RunConfig.
forecaster = sf.Forecaster.from_file("configs/explode_demo.json")

# See what it would do — no GCP calls, no compute launched.
plan = forecaster.dry_run()
print(plan.run_id)          # deterministic config hash — the id the real run lands under
print(plan.fanout)          # estimated series × models × folds = cells
print(plan.python_models)   # models routed to the Spark/Ray runtime
print(plan.bq_models)       # models routed to BigQuery-native

# Run it. Spark/Ray and BigQuery-native run in parallel under one run_id.
result = forecaster.run()
print(result.run_id, result.dataset_ref)
print(result.views)         # v_run_summary, v_model_leaderboard — query these, filtered by run_id
```

`Forecaster` construction:

| Constructor | Source |
|-------------|--------|
| `sf.Forecaster(cfg)` | an in-memory `sf.RunConfig` |
| `sf.Forecaster.from_file(path)` | a JSON config file (raises `sf.ConfigError` on a bad file) |
| `sf.Forecaster.from_dict(data)` | an already-parsed dict (raises `sf.ConfigError` on a bad schema) |

Useful methods and properties:

- `.run_id` — the deterministic run_id for this config (pure hash; no GCP call).
- `.dry_run() -> DryRunResult` — validate + report the planned fan-out and runtime split, offline.
- `.run(*, spark=None) -> RunResult` — execute; pass a `SparkSession` (incl. a Spark Connect
  `DataprocSparkSession`) to run the Spark models in-process instead of as a remote batch.
- `.review() -> RunResult` — a pointer to where a *past* run's results live, computed offline from
  the deterministic run_id. `dataset_ref` is `None` if the GCP identity can't be resolved.

### Threading an explicit GCP identity

By default the run resolves its project/dataset from the `SF_*` environment variables. To pass an
explicit identity instead (e.g. from Terraform outputs), inject `Settings`:

```python
settings = sf.Settings.from_terraform_outputs(tf_outputs)   # or sf.Settings.resolve()
forecaster = sf.Forecaster.from_file("configs/explode_demo.json", settings=settings)
result = forecaster.run()
```

---

## The direct path: drive Spark or Ray yourself

When you already have your own Spark or Ray job, you can call the model machinery directly and skip
the SDK. The stable surface is:

| Name | What it is |
|------|-----------|
| `run_cell(series, model, cfg, params=None)` | the unit of work — fit + predict one series×model into a `CellResult`; never raises (a failure comes back as `status="error"`) |
| `run_group(pdf, cfg, models=None, params_by_model=None)` | pure: run every cell in one pandas frame; returns `(list[CellResult], status_frame)` |
| `make_group_runner(cfg, settings, ...)` | builds the `applyInPandas` closure: `run_group` + write results, returns the status frame |
| `make_chunk_runner(cfg, settings, ...)` | the Ray twin of `make_group_runner`, for a `@ray.remote` task |
| `chunk_cells(source, cfg, models, n_chunks)` | pure: shuffle `(series × models)` into task-sized frames (advanced/explode embedding) |

There are two embedding modes:

- **Naive (recommended):** hand `run_group` a chunk of whole series plus `models=[...]`; it loops
  every model per series. This is the simplest shape for your own `applyInPandas` / `@ray.remote`.
- **Explode (advanced):** pre-tag with `chunk_cells(...)` to get cell-level fan-out. Each tagged
  frame carries one `(ts_id, model)` cell per group, so `run_group` takes its per-cell branch.

### Spark — your own `applyInPandas`

`make_group_runner` builds the executor function for you: it runs the group and writes the results
(once per bucket, executor-side) so no forecast payload crosses back to the driver.

```python
import scale_forecasting as sf

cfg = sf.load_config("configs/explode_demo.json")
settings = sf.Settings.resolve()

# Read your source into a Spark DataFrame with columns [ts_id, ds, y], add a bucket column, then:
runner = sf.make_group_runner(cfg, settings, models=cfg.models)
status = (
    df.groupBy("bucket")
      .applyInPandas(runner, schema="ts_id string, model_type string, status string, "
                                    "fit_seconds double")
)
status.show()   # one row per cell: ts_id, model_type, status, fit_seconds
```

Prefer to control the write yourself? Call the pure core and the writer separately:

```python
from scale_forecasting.registry import bq

def my_runner(pdf):
    results, status = sf.run_group(pdf, cfg, models=cfg.models)
    if results:
        bq.write_cells(results, settings=settings)   # append-only + dedupe-on-read; safe per-partition
    return status
```

### Ray — your own `@ray.remote`

```python
import ray
import scale_forecasting as sf

cfg = sf.load_config("configs/ray_cpu_demo.json")
settings = sf.Settings.resolve()

# Shuffle (series × models) into task-sized chunks, then run one task per chunk.
source_pdf = ...  # a pandas frame with columns [ts_id, ds, y]
chunks = sf.chunk_cells(source_pdf, cfg, cfg.models, n_chunks=64)

runner = sf.make_chunk_runner(cfg, settings)
run_remote = ray.remote(runner)
statuses = ray.get([run_remote.remote(chunk) for chunk in chunks])
```

Each task runs the shared `run_group` and appends its results with the same writer the Spark path
uses — appends compose, so per-task writes are safe (results dedupe on read by `run_id`).

---

## Which door should I use?

- **Just run a config** → `Forecaster` (or the CLI, `python -m scale_forecasting.main`).
- **Embed forecasting into a Spark/Ray job you already own** → the direct path.

Either way the per-cell logic is identical, so you can prototype with `Forecaster`, then drop down to
the direct path without changing model behavior. See [architecture.md](./architecture.md) for how the
pieces fit and [configuration_reference.md](./configuration_reference.md) for every config knob.
