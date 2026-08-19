# Output schemas — the registry tables

Every run writes to the same **three-tier registry** plus a backtest table, and reads back through
two curated views. This page documents the layout of each — what every column collects and how the
tiers link — so you can query the results directly, not just through the notebooks.

Two facts hold for all of them:

- **Always native BigQuery.** The four run-collection tables are native (never Iceberg), so
  `raw_config` / `job_telemetry` / `quantiles` / `best_params` are the real `JSON` column type and a
  reseed is a clean `WRITE_TRUNCATE`. (The *input* table is the one that ships in both Iceberg and
  native — see [configuration_reference.md](./configuration_reference.md).)
- **Written via the Storage Write API.** Engines return data, not RPCs; results are streamed into
  these tables in bulk through the BigQuery **Storage Write API** for high-speed updates, so
  throughput is bounded by compute, not a tracking server's QPS. Writes are **append-only**, and the
  views **dedupe on read** (idempotency) — re-running the same `run_id` never corrupts a table,
  it just appends rows the views collapse back to one.

The schema below is rendered from a single source of truth,
[`src/scale_forecasting/registry/ddl.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/registry/ddl.py); the views from
[`src/scale_forecasting/registry/views.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/registry/views.py).
`registry/bq.ensure_tables` / `ensure_views` execute exactly what those render.

## How the tiers link

```
run_registry            (1 row  per run)          ── the config + run-level telemetry
   │  run_id
   ├── forecast_metadata (1 row  per run × series × model)  ── per-cell metrics + artifact link
   ├── forecast_predictions (N rows per run × series × model) ── the forecast values (one per date)
   └── backtest_oof      (rows   per run × series × model × fold) ── out-of-fold truth vs prediction
```

Everything joins on **`run_id`** (the config digest — see
[configuration_reference.md](./configuration_reference.md)); the per-series tiers additionally share
`ts_id` and `model_type`.

---

## `run_registry` — one row per run

The config *is* the experiment record. This tier stores the whole config verbatim plus how the run
went. Partitioned by `DATE(created_at)`, clustered by `run_id`.

| Column | Type | Collects |
|--------|------|----------|
| `run_id` | `STRING` | The run identity — a digest of the full config (`make_run_id`). Same config → same id (idempotent). |
| `created_at` | `TIMESTAMP` | When the header row was written. |
| `user_id` | `STRING` | Who/what launched the run (identity of the writer). |
| `git_sha` | `STRING` | The code revision that produced the run (lineage). |
| `python_runtime` | `STRING` | `spark` or `ray` — which runtime ran the Python models. |
| `spark_method` | `STRING` | `explode` / `multi` / `naive` fan-out (NULL when `python_runtime=ray`). |
| `bq_models` | `ARRAY<STRING>` | Which BigQuery-native models ran in parallel (e.g. `arima_plus`). |
| `backtest_on` | `BOOL` | Whether backtesting was enabled for the run. |
| `decision_metric` | `STRING` | The metric folds were judged on (when backtesting). |
| `ensemble_strategies` | `ARRAY<STRING>` | Which consensus strategies ran (e.g. `median`, `nnls`). |
| `raw_config` | `JSON` | **The entire validated config, verbatim** — the experiment record. |
| `status` | `STRING` | `RUNNING` → `COMPLETED` / `FAILED`. |
| `n_series` | `INT64` | Series count actually run. |
| `n_models` | `INT64` | Model count actually run. |
| `runtime_seconds` | `FLOAT64` | The engine's own compute time (excludes cluster stand-up). |
| `job_telemetry` | `JSON` | Dataproc/Ray overlay: `total_wall_s`, executor sizing, `dcu_milli_seconds`, `runtime_version`. Unpacked by `v_run_summary`. |

## `forecast_metadata` — one row per (run, series, model) cell

The metrics tier: how each model did on each series, and where its persisted artifact lives.
Partitioned by `DATE(created_at)`, clustered by `run_id, model_type`.

| Column | Type | Collects |
|--------|------|----------|
| `run_id` | `STRING` | Joins to `run_registry`. |
| `ts_id` | `STRING` | The series. |
| `model_type` | `STRING` | The model (e.g. `theta`, `arima_plus`). |
| `compute_engine` | `STRING` | Where the cell ran (`spark` / `ray` / `bigquery`). |
| `model_hash` | `STRING` | Content hash of the fitted model (lineage / cache key). |
| `ensemble_id` | `STRING` | NULL for base models; the `EnsembleConfig` digest for ensemble pseudo-models (so two ensemble configs under one `run_id` stay distinct). |
| `fold_id` | `INT64` | NULL for the final (full-fit) row; set for a backtest fold's metrics. |
| `mae`, `rmse`, `mse`, `mape`, `smape`, `wape`, `mase`, `rmsse`, `bias`, `coverage`, `pinball` | `FLOAT64` | The metric panel. Populated only when a backtest produced out-of-fold predictions to score; otherwise NULL. |
| `fit_seconds` | `FLOAT64` | Wall-clock to fit this cell (surfaces the straggler under `naive`). |
| `best_params` | `JSON` | Winning hyperparameters when HPO ran (else NULL). |
| `model_artifact` | `STRING` | GCS ObjectRef to the persisted model (`persist_models=true`), else NULL. `no_artifact_rate=1.0` in the leaderboard = no cell produced an artifact. |
| `created_at` | `TIMESTAMP` | When the row was written. |

## `forecast_predictions` — the forecast values

The values tier: one row per (run, series, model, **date**) over the horizon. Partitioned by
`forecast_date`, clustered by `run_id, ts_id`.

| Column | Type | Collects |
|--------|------|----------|
| `run_id` | `STRING` | Joins to `run_registry`. |
| `ts_id` | `STRING` | The series. |
| `model_type` | `STRING` | The model that produced this point (base model name or `ensemble_<strategy>`). |
| `compute_engine` | `STRING` | Where it ran (`spark` / `ray` / `bigquery`). |
| `ensemble_id` | `STRING` | NULL for base models; the ensemble digest for ensemble rows. |
| `forecast_date` | `DATE` | The future date this point forecasts (partition key). |
| `yhat` | `FLOAT64` | The point forecast. |
| `yhat_lower` | `FLOAT64` | Lower prediction-interval bound. |
| `yhat_upper` | `FLOAT64` | Upper prediction-interval bound. |
| `quantiles` | `JSON` | Full quantile forecast when a model emits one (e.g. `{"0.1": ..., "0.9": ...}`), else NULL. |

## `backtest_oof` — out-of-fold predictions (learned ensembling)

The evidence learned ensembles (`nnls`/`ridge`/`xgb`) train on: the held-out truth vs. each base
model's prediction, per fold. Written only when `backtest.enabled`. Partitioned by `forecast_date`,
clustered by `run_id, ts_id`.

| Column | Type | Collects |
|--------|------|----------|
| `run_id` | `STRING` | Joins to `run_registry`. |
| `ts_id` | `STRING` | The series. |
| `model_type` | `STRING` | The base model whose OOF prediction this is. |
| `fold_id` | `INT64` | Which backtest fold. |
| `forecast_date` | `DATE` | The held-out date. |
| `y_true` | `FLOAT64` | The actual value (held out of training that fold). |
| `yhat` | `FLOAT64` | The base model's prediction for it. |

---

## The read surface — two views

You rarely query the raw tables. Two `CREATE OR REPLACE VIEW`s are the curated read surface (and they
apply the dedupe-on-read). Full operator loop in
[running_and_reviewing.md](./running_and_reviewing.md).

### `v_run_summary` — how did each run go, and how efficiently?

One row per run: the scaling knobs (`spark_method`, `n_series`, `n_models`) plus the `job_telemetry`
JSON unpacked into scalars — `total_wall_s`, `overhead_seconds` (`total_wall_s − runtime_seconds`),
`overhead_fraction`, `executor_instances` / `executor_cores` / `max_executors`, `dcu_milli_seconds`,
`runtime_version`. This is the scaling-and-efficiency story: explode's flat curve vs. naive's
straggler, with provisioning overhead that amortizes at scale.

### `v_model_leaderboard` — which model won, per run?

One row per `(run_id, model_type, ensemble_id)`: `n_cells`, `n_no_artifact` /
`no_artifact_rate` (a model failing every cell shows as `no_artifact_rate = 1.0`),
`median_fit_seconds`, and `mean_wape` / `mean_mae` where a backtest populated them. The entry point
for "is this model worth keeping" before ensembling. Reads only the final rows (`fold_id IS NULL`),
so per-fold metrics don't double-count.

---

See also: [configuration_reference.md](./configuration_reference.md) (the run *is* the config that
lands in `raw_config`) · [running_and_reviewing.md](./running_and_reviewing.md) (submit, watch, and
review through these views) · [`registry/ddl.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/registry/ddl.py) /
[`registry/views.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/registry/views.py) (the source of truth).
