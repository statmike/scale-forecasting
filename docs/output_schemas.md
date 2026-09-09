# Output schemas — the registry tables

Every run writes to the same **registry tables** plus a backtest table, and reads back through
three curated views. This page documents the layout of each — what every column collects and how the
tiers link — so you can query the results directly, not just through the notebooks.

Two facts hold for all of them:

- **Always native BigQuery.** The five run-collection tables are native (never Iceberg), so
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
`registry.tables.ensure_tables` / `ensure_views` execute exactly what those render.

## How the tiers link

```
run_registry            (1 row  per run)          ── the config + run-level telemetry
   │  run_id
   ├── run_jobs          (1 row  per run × family) ── per-family-job runtime/hardware + telemetry
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
| `python_runtime` | `STRING` | `spark` or `ray` — the run-level default runtime for the Python model families (a family can override it). |
| `bq_models` | `ARRAY<STRING>` | Which BigQuery-native models ran in parallel (e.g. `arima_plus`). |
| `backtest_on` | `BOOL` | Whether backtesting was enabled for the run. |
| `decision_metric` | `STRING` | The metric folds were judged on (when backtesting). |
| `ensemble_strategies` | `ARRAY<STRING>` | Which consensus strategies ran (e.g. `median`, `nnls`). |
| `raw_config` | `JSON` | **The entire validated config, verbatim** — the experiment record. |
| `status` | `STRING` | `RUNNING` → `COMPLETED` / `FAILED`. |
| `n_series` | `INT64` | Series count actually run. |
| `n_models` | `INT64` | Model count actually run. |
| `runtime_seconds` | `FLOAT64` | The engine's own compute time (excludes cluster stand-up). |
| `job_telemetry` | `JSON` | Dataproc/Ray overlay: `total_wall_s`, executor sizing, `dcu_milli_seconds`, `runtime_version`, plus `sizing.<family>` — the whole sizing decision per family job. Unpacked by `v_run_summary`. |

## `run_jobs` — one row per family job

A run resolves into one job per model family (`statistical` / `ml` / `deep_learning` / `native`),
plus the downstream `ensemble` node — each launched in parallel under the shared `run_id`. This tier
records what each of those jobs actually ran on and how it fared, so a run's DAG is queryable as
executed. A `--force` re-run appends a higher-`attempt` job under the same `(run_id, family)`.
Partitioned by `DATE(created_at)`, clustered by `run_id, family`.

| Column | Type | Collects |
|--------|------|----------|
| `job_id` | `STRING` | The canonical per-family job key (`make_job_key`) — `sf-<run_id>-<family>-a<attempt>`. |
| `run_id` | `STRING` | Joins to `run_registry`. |
| `family` | `STRING` | The model family this job ran (`statistical` / `ml` / `deep_learning` / `native`, or `ensemble`). |
| `attempt` | `INT64` | Attempt number — a `--force` re-run bumps it so re-runs are distinctly keyed under one `run_id`. |
| `runtime` | `STRING` | The resolved runtime for this family (`spark` / `ray` / `bigquery`). |
| `spark_mode` | `STRING` | The resolved Spark launch mode when `runtime=spark` (else NULL). |
| `hardware` | `STRING` | The resolved hardware profile for this family (else NULL). |
| `gpu_type` | `STRING` | The GPU type when the family ran on GPUs (e.g. Ray deep-learning), else NULL. |
| `system_job_id` | `STRING` | The platform's own job id (`dataproc_job_id` / `ray_submission_id` / `bigquery_job_id`) — jump straight to the platform console. |
| `status` | `STRING` | `RUNNING` → `COMPLETED` / `FAILED` for this job. |
| `created_at` | `TIMESTAMP` | When the job row was written. |
| `runtime_seconds` | `FLOAT64` | The job's own compute time (excludes cluster stand-up). |
| `job_telemetry` | `JSON` | Per-job overlay: `total_wall_s`, `dcu_milli_seconds`, sizing, and — for a GPU family — `device_use`, the verdict on whether the accelerator did anything ([below](#was-the-accelerator-you-paid-for-actually-used)). Unpacked by `v_run_jobs`. |

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
| `mae`, `rmse`, `mse`, `mape`, `smape`, `wape`, `mase`, `rmsse`, `bias`, `coverage`, `pinball`, `mase_seasonal`, `maape`, `interval_score`, `interval_width` | `FLOAT64` | The metric panel — every metric, every run, so choosing a different `decision_metric` never means re-running. Populated only when a backtest produced out-of-fold predictions to score; otherwise NULL. Definitions in [configuration_reference.md](./configuration_reference.md#decision_metric). |
| `fit_seconds` | `FLOAT64` | Wall-clock to fit this cell (per-cell fit time — surfaces the straggler cells). |
| `best_params` | `JSON` | Winning hyperparameters when HPO ran (else NULL). |
| `model_artifact` | `STRING` | GCS ObjectRef to the persisted model (`persist_models=true`), else NULL. `no_artifact_rate=1.0` in the leaderboard = no cell produced an artifact. |
| `created_at` | `TIMESTAMP` | When the row was written. |
| `worker_id` | `STRING` | `hostname:pid` of the worker that ran the cell — the trace's lane. |
| `cell_started_at`, `cell_ended_at` | `TIMESTAMP` | The cell's wall-clock bracket (Gantt/waterfall). |
| `cpu_seconds` | `FLOAT64` | CPU time the fit consumed, summed across threads. With `fit_seconds` this gives `effective_cores` — how much parallelism the library actually used. |
| `process_rss_bytes` | `INT64` | The worker process's **absolute** memory high-water while the cell ran — not the cell's increment. This is the number that sizes an executor slot. |
| `peak_gpu_bytes` | `INT64` | Peak device bytes allocated, never zero. **NULL is ambiguous on its own** — it covers no tensor library, a CPU-only build, no device, and `measure="off"` alike. Read it beside `device_used`, which says which. |
| `intraop_threads` | `INT64` | The native-thread cap in force (`OMP_NUM_THREADS`). Without it `cpu_seconds / fit_seconds` is uninterpretable — under a cap the ratio just reports the cap back. |
| `n_obs` | `INT64` | Rows fed to the fit — the data signature a later run matches against. |
| `device_requested` | `STRING` | What the cell was *told* to use: `auto` / `cpu` / `gpu`. Set from the job's provisioned hardware and the family's `hardware`, not from config intent. |
| `device_available` | `STRING` | What the worker could actually *see*: `cuda` / `cpu` / `unknown`. `unknown` means no tensor library was importable, so nobody asked — a different fact from "there is no card". |
| `device_used` | `STRING` | Where the fitted weights actually *landed*, read off a parameter tensor: `cuda` / `cpu`. NULL means the model has no device concept (everything but NeuralProphet). |
| `device_name` | `STRING` | The visible device, e.g. `Tesla T4`. NULL when none is. |
| `cell_status` | `STRING` | How the *cell* went: `ok` or `error`. |
| `error_class` | `STRING` | Which kind of failure it was, from a fixed vocabulary you can `GROUP BY` (see below). NULL on an `ok` cell. |
| `error_detail` | `STRING` | The exception itself, as text, truncated at 2,000 characters. NULL on an `ok` cell. |
| `interval_source` | `STRING` | Where the prediction interval came from **as the model produced it**: `native` (the model computed its own) or `residual` (built from the spread of its in-sample residuals). NULL on ensemble rows, which do not carry an interval. |
| `point_forecast_source` | `STRING` | Which arm the shipped `yhat` is: `raw` (the model's own number) or `median` / `mean` (that number plus the corresponding residual shift). |
| `point_forecast_decision` | `STRING` | *How* that arm was picked, which the column above cannot say on its own: `configured` (the config named it), `auto-corrected` / `auto-raw` (`output.point_forecast: "auto"`, and the cell's held-out folds went that way), `auto-few-folds` / `auto-no-backtest` (selection was asked for but there was not enough held out to decide, so the fleetwide arm applied), or `engine-native` (BigQuery-native rows, where only one arm exists). |
| `interval_calibration` | `STRING` | What happened to the band *after* the model produced it: `oof-per-step` (re-estimated per horizon step from out-of-fold residuals), `oof-flat` (one pooled out-of-fold band, too few residuals to resolve per step), `in-sample` (no backtest ran; the model's own band shipped unchanged), or `native` (BigQuery-native rows). |
| `point_forecast_margin` | `FLOAT64` | How much the corrected arm beat the raw one by on this cell, as a fraction of the raw arm's loss in the run's `decision_metric`. Positive means the correction helped. NULL when there was no backtest to grade it on. |
| `backtest_status` | `STRING` | How the *scoring* went, which is not how the cell went: `full` / `reduced` / `unscored` / `failed`. `reduced` covers any geometry other than the one the config asked for, not just a shorter fold list. NULL means backtesting was never asked for. |
| `n_folds_achieved` | `INT64` | Folds actually scored (`0` on `unscored`/`failed`). |
| `backtest_note` | `STRING` | Why the backtest was not `full` — the shortfall arithmetic, or the `short_series` policy and what it traded away, or the exception. NULL when it was. |

### Why a cell failed, in a word you can group by

A run of a million cells does not fail; some fraction of it does, and the only useful question is
*of what*. Three thousand rows of `error_detail` will not answer that — every message is phrased by
whatever library raised it, so counting them counts wording. `error_class` is the same failure
reduced to one token from a fixed list, which is what makes `GROUP BY error_class` a report:

| `error_class` | What it means | Whose problem it is |
|---------------|---------------|---------------------|
| `OOM` | The worker ran out of memory. | Sizing — a bigger slot or fewer cells per slot. |
| `CAPACITY` | The platform had no room: quota, stockout, a preflight refusal. | Capacity — retry elsewhere or later. |
| `TRANSIENT_INFRA` | A 5xx, a reset connection, a deadline. | Nobody's — the same cell would likely succeed on a retry. |
| `SHORT_HISTORY` | The series did not have enough observations. | **Ours.** A series short enough to defeat a model should have been detected before the fit, not during it. |
| `CONFIG_REPAIRABLE` | The config asked for something that does not exist, e.g. an unregistered model name. | The config — and it is fixable without touching data or code. |
| `BAD_DATA` | Duplicate timestamps, a gap, a NaN, a value off the declared frequency. | The source table. |
| `MODEL_ERROR` | The fit itself failed: no convergence, a singular matrix. | The model/series pairing. |
| `UNKNOWN` | Nothing in the table matched. | Read `error_detail`. |

`UNKNOWN` is a real answer rather than a gap in the table. A classifier that guesses is worse than
one that admits it does not know: a wrong token sends the reader to the wrong fix, and an honest
`UNKNOWN` with the text beside it sends them to the message. If a class of failure keeps arriving as
`UNKNOWN`, that is the signal to add a row — not to widen an existing one.

Classification happens in the worker, at the moment the exception is caught, because that is the
last point at which the exception *object* exists; by the time the row reaches BigQuery it is a
string, and the exception's type is the strongest evidence the table has. The vocabulary lives in
`worker.ERROR_CLASSES`, the one module both Python runtimes share, so a Spark cell and a Ray cell
classify the same failure identically.

Native (BigQuery) and ensemble rows carry `cell_status = 'ok'` with both error columns NULL. A row
only exists on those paths because the model was built and forecast — a native failure takes the
whole job down and writes nothing — so there is never a native error row to describe. They are
filled rather than left NULL so that `WHERE cell_status = 'ok'` does not quietly skip every native
and ensemble model in the run.

### Where the prediction interval came from

Every model in the suite emits `yhat_lower` / `yhat_upper`, but they do not all mean the same thing,
and the two columns look identical either way. Some models compute an interval as part of their own
arithmetic — Theta, SARIMAX, Prophet, UCM, STL-bagging, NeuralProphet, and the BigQuery-native
models all do. The rest have no notion of uncertainty at all, so `BaseModel.residual_intervals`
manufactures a band for them from the spread of their in-sample residuals.

A residual band is a reasonable fallback and a poor measurement. It is fitted on data the model has
already seen, so it is optimistic; and it is one constant width applied to the whole horizon, so it
cannot widen as the forecast gets further from what the model knows. Comparing its `coverage` or
`interval_score` against a model that computed its own interval is not a like-for-like comparison,
and until this column existed there was nothing in the row to tell you which kind you were looking
at. `WHERE interval_source = 'native'` makes that comparison honest.

BigQuery-native rows say `native` because `ML.FORECAST` returns real bounds. Ensemble rows leave it
NULL: combining base-model point forecasts produces no interval, so there is no provenance to
record.

### What happened to the band afterwards

`interval_source` describes what the *model* handed over. `interval_calibration` describes what the
system did with it, and the two are independent — when a backtest ran, the band that ships is
re-estimated from out-of-fold residuals regardless of which kind the model started with.

That re-estimation is the fix for both weaknesses above at once. The residuals come from held-out
folds rather than from data the model has already seen, so the band is no longer optimistic by
construction; and they are bucketed by `horizon_step`, so the band is allowed to widen with distance
instead of being one width for the whole horizon.

Measured on ten models × 24 series, scored leave-one-fold-out so nothing is graded on the residuals
it was fitted from: the in-sample band achieved **0.601** coverage against a nominal 0.8, the
recalibrated band **0.790**. The average is the smaller half of the result. Per-model coverage ran
from 0.053 to 0.837 before and from 0.755 to 0.821 after — so a column that meant something
different for every model now means the same thing across the leaderboard, which is what makes
`coverage` usable as a `decision_metric` at all.

| `interval_calibration` | What you are looking at |
|---|---|
| `oof-per-step` | Re-estimated per horizon step from out-of-fold residuals. The band widens with distance because the measured error does. |
| `oof-flat` | Out-of-fold, but too few residuals to resolve per step, so one pooled band covers the horizon. Honest about being flat rather than dressed up as per-step. |
| `in-sample` | No backtest ran, so the model's own band shipped untouched — whatever `interval_source` says it was. |
| `native` | A BigQuery-native row. `ML.FORECAST` computes its own bounds and nothing downstream rewrites them. |

Per-step estimation needs enough residuals per step to be worth the name; below that floor the
window over neighbouring steps widens until it has them, and if it swallows the whole horizon the
row says `oof-flat` rather than claiming a resolution it does not have.

### Which number `yhat` is

Every model emits one number per future date, and something has to decide what that number is. Three
columns record the decision rather than leaving it implicit:

- **`yhat_raw`** — the model's own output, untouched.
- **`yhat_adjusted`** — that output plus a residual shift, either the median (minimises absolute
  error) or the mean (minimises squared error, and drives `bias` to zero by construction).
- **`yhat`** — whichever of the two the run shipped, chosen by `output.point_forecast` — either once
  for the whole run, or per series and model when that is set to `auto`.

Both arms are always written, so the choice is never destructive: a run that shipped the corrected
arm can still be scored on the raw one months later without re-fitting anything.
`forecast_metadata.point_forecast_source` says which arm `yhat` is,
`point_forecast_decision` says how that arm was picked, and `point_forecast_margin` says
what the choice was worth on that cell. The margin is always measured corrected-minus-raw whichever
arm shipped, so averaging it over cells that chose differently still means something. `sf.calibration_report(run_id)` rolls both up per model
alongside the coverage panel — the win rate matters more than the average margin, because a
correction that helps half the series a lot and hurts the other half a lot is a different
proposition from one that helps everything a little.

On BigQuery-native rows all three columns hold the same number: `ML.FORECAST` returns one forecast,
there is no second arm, and writing it three times keeps `y_true - yhat_raw` a valid residual on
every engine so a cross-engine query needs no special case.

### A series too short to score still has a forecast

Backtesting scores a model; it does not produce the forecast. Those three columns exist because the
two used to be welded together: a series shorter than `min_train + horizon + (n_folds−1)·step` raised,
the cell caught it as an error, and a forecast that had not even been attempted was thrown away.
Short history was the single largest error class in the registry, and none of it was a modelling
failure.

Now the fold grid adapts to what the series supports — by default the oldest folds are dropped,
survivors keeping their original `fold_id` — and the cell fits and forecasts either way. What the
grid does instead of raising is `backtest.short_series`'s to decide; the outcome is recorded rather
than inferred, because through the metric columns alone these are indistinguishable:

| Situation | Metrics | `backtest_status` |
|-----------|---------|-------------------|
| Backtesting switched off | all NULL | NULL |
| Scored on exactly the grid the config asked for | populated | `full` |
| Scored on an adjusted grid — fewer folds, a narrowed `step`, or a shrunken `min_train` | populated | `reduced` |
| Too short to score at all, or the policy declined to score it | all NULL | `unscored` |
| Scoring raised | all NULL | `failed` |

`reduced` is the broad one: it means "not the geometry the config asked for", whichever way the
geometry moved. Under `short_series: "overlap"` or `"shrink_train"` that produces a pair which is
otherwise impossible — `reduced` with `n_folds_achieved` equal to the requested `n_folds` — and
that pair is precisely the signal that the full fold count was bought with fold independence or
with training history rather than with folds. `backtest_note` says which.

`n_folds_achieved` is what makes a leaderboard readable across a ragged panel: two series with the
same WAPE are not comparable if one was scored on five folds and the other on one.

Base-model rows fill these whichever engine wrote them — a Python cell and a BigQuery-native cell
answer the same question the same way. **Ensemble rows are the exception**: they leave all three
NULL even when the ensemble was scored, because an ensemble is scored on the base models' folds
rather than on folds of its own. That gets its own column, `ensemble_scoring`, described next.

### Was this number earned on a fold the fit had already seen?

Two columns answer that, and they exist because a metric column cannot answer it by itself. An
in-sample number and an honest one look identical once they are sitting side by side under
`wape`.

The rule the run enforces is one sentence: **the newest fold — the one with the smallest step-back,
`n_folds - 1` — is never used to fit anything.** Not stacker weights, not `inverse_error` weights,
not a hyperparameter search. Everything that learns from the backtest learns from the older
*inner* folds and is then judged on the newest one. Scoring is unaffected; every fold is still
scored. Only *fitting* is restricted.

| Column | On which rows | Values |
|--------|---------------|--------|
| `ensemble_scoring` | `ensemble_*` rows | `holdout`, `in_sample`, or NULL for `mean`/`median`, which fit nothing |
| `hpo_scoring` | base-model rows whose params came from a search | `holdout`, `in_sample`, or NULL when no search ran |

`in_sample` is not an error. There are two ordinary ways the split fails to exist: the run asked
for `n_folds: 1`, or a short series in a ragged panel achieved only the newest fold. In both cases
the fit falls back to using everything — refusing would cost the run a whole ensemble over a fold
count — and the column records that it did. That is the entire point: the fallback is *said*, not
inferred.

The run header carries the same claim once for the whole run, on
`run_registry.job_telemetry.$.scoring`: `hpo` is `off`, `holdout`, or `in_sample` from the run's
fold geometry, and `ensemble` is the **weakest** answer across the strategies that fitted anything
(one `in_sample` strategy makes the run's ensemble metrics partly in-sample, and a header that
reported the best case would be exactly the reassurance these columns exist to withhold).

### Was the accelerator you paid for actually used?

Those four columns exist because that question used to be unanswerable. Every GPU run in this
project's history was, in substance, a CPU run: the card attached, the forecasts were correct, and
nothing recorded that the arithmetic had happened somewhere else. `device_requested` is the intent,
`device_available` is the environment, and `device_used` is the receipt — the three disagree exactly
when something is wrong.

Each family job also gets a one-line verdict on `run_jobs.job_telemetry.device_use`, drawn from
these columns on the driver once the job's cells are written:

| Verdict | Means |
|---------|-------|
| `MISSING_DEVICE` | The family asked for a device and no cell reports having run on one. The run is correct and the accelerator was billed for nothing. |
| `ENGAGED_IDLE` | Cells ran on the device and barely touched it. A cost finding, not a fault — **this is the expected verdict today** (NeuralProphet peaks at 50–78 KB on a 17 GB T4 unless `n_lags > 0`). |
| `ENGAGED_UTILISED` | Cells ran on the device and used at least 1% of it. |
| absent | The family never asked for a device, so there was nothing to judge. |

The verdict warns and never fails a job. It is filed with the counts it was drawn from
(`cells`, `cells_on_device`, `cells_no_device`, `max_peak_gpu_bytes`, `device_name`) so it can be
re-checked later rather than taken on trust.

Two places read it back. `v_run_jobs.device_verdict` is the word as a column, so a fleet-wide "which
jobs wasted their card" is one `WHERE` clause; and `review.monitor_run` carries it on each
`FamilyProgress`, so `plot_progress` prints `gpu used` / `gpu idle` / `no gpu` at the end of that
family's bar. A GPU family's bar is otherwise indistinguishable from a CPU family's, which is how
the accelerator went unnoticed for twenty-one jobs in the first place.

### Columns that exist but are not filled yet

`SELECT *` on this table also returns
`achieved_step`, `achieved_min_train`, `first_val_date`, `last_val_date`, `n_fits`, and
`train_rows_total`. **They are all NULL today.** They are
declared ahead of the code that writes them because adding a column to a deployed table is a
migration every deployment has to run, and doing that once is better than doing it five times.
Don't build a reader on them yet — `NULL` here means "not recorded", not "no".

### Sizing a future run from a past one

The last five columns are the **compute harvest**. Every cell records what it cost, on the
hardware it really ran on, so a completed `run_id` doubles as a measured cost model: point
`profiling.cost.harvest_profile` at those rows and it aggregates them into the same `ComputeProfile` the
deliberate pre-pass produces, which the fleet translators then size from. Nothing extra is stored
and nothing extra is versioned — the profile is a query over a run.

Harvest is on by default (`compute.profile.measure`, see
[configuration_reference.md](./configuration_reference.md)); it costs three cheap probes per fit.
All five read NULL when it is off — which is also how rows written before these columns existed read
back, so both mean "no evidence" rather than "zero". A cell that errored never reached the probes,
so its `cpu_seconds` is NULL and the harvest reader skips it on that alone; `cell_status = 'error'`
is the readable version of the same fact, and the two must not be allowed to disagree — an error
row counted into a cost model would size slots off fits that never happened.

Consumption is on by default too: `compute.profile.source = "auto"` makes the next run look for the
newest harvest matching its data signature and size itself from it, stamping the `run_id` it chose
into the staged config so the choice is on the record. Set `source` to a specific `run_id` to pin
one, or to `"none"` to ignore the harvest entirely.

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
| `yhat` | `FLOAT64` | The point forecast that shipped — whichever arm `output.point_forecast` selected. |
| `yhat_raw` | `FLOAT64` | The model's own output, before any residual correction. |
| `yhat_adjusted` | `FLOAT64` | The corrected arm: `yhat_raw` plus the residual shift. Equal to `yhat_raw` on BigQuery-native rows, which have only one arm. |
| `yhat_lower` | `FLOAT64` | Lower prediction-interval bound. |
| `yhat_upper` | `FLOAT64` | Upper prediction-interval bound. |
| `quantiles` | `JSON` | Full quantile forecast when a model emits one (e.g. `{"0.1": ..., "0.9": ...}`), else NULL. |
| `created_at` | `TIMESTAMP` | **Declared, not yet written — NULL today.** Reserved for telling two generations of rows apart under one `run_id`; see the note under `forecast_metadata`. |

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
| `yhat` | `FLOAT64` | The base model's prediction for it, on the arm the run shipped. |
| `yhat_raw` | `FLOAT64` | The model's own output for that date, uncorrected. This is the column the calibration reads: `y_true - yhat_raw` is the residual that both the point-forecast shift and the per-step band are estimated from. |
| `yhat_adjusted` | `FLOAT64` | The corrected arm for that date. |
| `yhat_lower` | `FLOAT64` | Lower bound of the prediction interval for that held-out date. |
| `yhat_upper` | `FLOAT64` | Upper bound of the same interval. |
| `cutoff_date` | `DATE` | The last date the model was allowed to see when it made this prediction — the fold's origin, and the column that identifies a fold across engines. |
| `horizon_step` | `INT64` | How far ahead of `cutoff_date` this row is, counting from 1. |
| `created_at` | `TIMESTAMP` | **Declared, not yet written — NULL today.** The write timestamp; same reasoning as the note under `forecast_metadata`. |
| `ensemble_id` | `STRING` | NULL on a base model's row; the `EnsembleConfig` digest on a blended row. Two ensemble configs scored under one `run_id` stay distinct instead of overwriting each other. |

### Ensembles land here too

An ensemble is a model like any other, so it writes its blended out-of-fold predictions back into
this same table — one row per `(ts_id, fold_id, forecast_date)` under `model_type =
'ensemble_<strategy>'`, with `ensemble_id` set. Only the blended columns are filled: `y_true`,
`yhat`, `cutoff_date` and `horizon_step` carry through from the base rows the blend was computed
from, and `yhat_raw` / `yhat_adjusted` / the two bounds stay NULL, because a weighted combination of
corrected predictions has no separate "uncorrected arm" to report.

The reason this matters is comparability. `v_model_leaderboard_comparable` pools error over rows of
*this* table, so before ensembles wrote here they were simply **absent** from that ranking — not
ranked poorly, missing. Writing them here puts base models and ensembles on one fold, one pooling
rule and one column.

Two consequences worth knowing. The reads that feed the ensembler filter to the run's base models
(`model_type IN (…)`), so a re-ensemble or a microbatch drain never blends a previous consensus back
into the next one. And every per-horizon query — anything grouping by `horizon_step` — now sees
ensembles alongside the models they were built from.

### Error by how far ahead you asked

`horizon_step` exists so that "how fast does this model decay?" is a `GROUP BY`, not a second run.
A model that is excellent one day out and useless four weeks out has the *same* `wape` in the
leaderboard as one that is mediocre throughout, because the leaderboard averages the whole horizon.
Grouping the OOF rows by `horizon_step` separates them.

It is also the column the interval calibration is built on. A band estimated once for the whole
horizon is the **same width at every step** — as wide one day out as twenty-eight days out. Real
uncertainty grows with distance, so such a band is too wide early and too narrow late, and the late
under-coverage is invisible in a single averaged `coverage` number. Bucketing the out-of-fold
residuals by `horizon_step` is what lets the shipped band widen with distance instead; a run whose
`interval_calibration` says `oof-per-step` has had that done to it. Where it says `in-sample` or
`oof-flat`, the flat-band weakness is still there, and grouping by `horizon_step` shows it
immediately.

### Which fold is this, really

`fold_id` is an ordinal within one series' own backtest plan, and the two engines number from
different anchors. The Python engines count folds back from **each series' own last observation**;
the BigQuery-native path counts back from **one global `MAX(ds)`** across the whole panel. On a
panel where every series ends on the same date those agree exactly. On a ragged panel they do not,
and the disagreement is silent — the same `fold_id` names a different training window on each
engine, so a join on the ordinal pairs nothing and an ensemble quietly blends fewer models than it
claims to.

`cutoff_date` is the fix and the reason both engines now write it: two rows describe comparable
forecasts when the models saw the same history and predicted the same date, which is what the
cutoff says and the ordinal does not. The ensemble joins on `(ts_id, cutoff_date, forecast_date)`.
`fold_id` is still written — it is the ordinal a reader recognises, and it is what
`n_folds_achieved` counts — it is simply not what anything joins on.

---

## The read surface — five views

You rarely query the raw tables. Five `CREATE OR REPLACE VIEW`s are the curated read surface (and they
apply the dedupe-on-read). Full operator loop in
[running_and_reviewing.md](./running_and_reviewing.md).

### `v_run_summary` — how did each run go, and how efficiently?

One row per run: the scaling knobs (`n_series`, `n_models`, `python_runtime`) plus the `job_telemetry`
JSON unpacked into scalars — `total_wall_s`, `overhead_seconds` (`total_wall_s − runtime_seconds`),
`overhead_fraction`, `executor_instances` / `executor_cores` / `max_executors` /
`executor_memory` / `executor_memory_overhead`, `dcu_milli_seconds`, `runtime_version` — and the
raw `sizing` record. This is the run-level scaling-and-efficiency story: how wall-clock and overhead
move with the series and model counts and the chosen runtime, with provisioning overhead that
amortizes at scale. The per-family runtime/hardware breakdown that composes each run lives in
`v_run_jobs`.

The executor columns are the shape the platform was **told** (echoed back off the submitted job);
`sizing` is **why** it was that shape. Each family job of a run writes its own entry, so the column
holds one object per family:

```sql
SELECT
  JSON_VALUE(sizing, '$.deep_learning.plans[0].slot.memory_bytes')  AS dl_slot_memory,
  JSON_VALUE(sizing, '$.deep_learning.profile.provenance.run_id')   AS sized_off_run,
  JSON_VALUE(sizing, '$.statistical.translation.executor_cores')    AS stat_cores
FROM `PROJECT.DATASET.v_run_summary`
WHERE run_id = 'RUN_ID';
```

Three parts per entry: `plans` (the fleet the arithmetic asked for — one per fleet, so a Ray job
records its CPU **and** GPU pools), `translation` (what that became in platform settings, and the
ideals it snapped from — `null` on Ray, which sets task options rather than properties), and
`profile` (the measurements it was sized off, with the provenance naming whose run they came from —
`null` when the run sized from declared config alone). Absent for a run submitted before this
existed, or one that left the platform's own defaults standing.

### `v_run_jobs` — what jobs ran, on what runtime/hardware, and how did each fare?

One row per `(run_id, family)` = the run's DAG as executed: the deterministic `job_id`, the
`attempt`, the resolved `runtime` / `spark_mode` / `hardware` / `gpu_type`, the platform's own
`system_job_id`, the per-job `status` / `created_at` / `runtime_seconds`, and the per-job
`job_telemetry` unpacked into `total_wall_s` and `dcu_milli_seconds`. `device_verdict` is the one
word saying whether this row's accelerator did any work, with the counts behind it in `device_use`
beside it — both NULL for a CPU family, so `WHERE device_verdict != 'ENGAGED_UTILISED'` is the "what
did I pay for and not use" query ([below](#was-the-accelerator-you-paid-for-actually-used)). A
`--force` re-run appends a
higher-`attempt` job under the same `(run_id, family)`; the view keeps only the current one
(`QUALIFY ROW_NUMBER() … ORDER BY attempt DESC = 1`), so the `run_id → current job` map is one row
per family.

### `v_model_leaderboard` — which model won, per run?

One row per `(run_id, model_type, ensemble_id)`: `n_cells`, `n_no_artifact` /
`no_artifact_rate` (a model failing every cell shows as `no_artifact_rate = 1.0`),
`median_fit_seconds`, and `mean_wape` / `mean_mae` where a backtest populated them. The entry point
for "is this model worth keeping" before ensembling. Reads only the final rows (`fold_id IS NULL`),
so per-fold metrics don't double-count.

### `v_backtest_coverage` — how much of the panel did each model get scored on?

The question the leaderboard cannot answer, and the one that decides whether its ranking means
anything. A ragged panel does not give every series the same number of folds: `backtest.make_folds`
drops the oldest folds a short series cannot afford. So one model's `mean_wape` might be an average
over ten folds of two thousand series while its neighbour's is an average over one fold of two
hundred — two different questions, printed in the same column.

One row per `(run_id, model_type, ensemble_id, backtest_status, n_folds_achieved)`, with `n_series`
and `series_share` (that cohort's fraction of the model's panel). `backtest_status` is `full`,
`reduced`, `unscored` or `failed`, or NULL where the run never asked for a backtest at all. The
shape is long rather than one wide row per model because the achieved-fold histogram has no fixed
width, and because a wide shape would need a self-join that breaks on the NULL `ensemble_id` every
base model carries.

Read it beside the leaderboard. A model whose panel is mostly `reduced` won on an easier question.

### `v_model_leaderboard_comparable` — which model won, holding the question fixed?

The same ranking as `v_model_leaderboard`, rebuilt so the numbers are comparable across models
rather than merely present. Two differences, and both are the point.

First, it is restricted to the **holdout fold** — the newest one, `MAX(fold_id) OVER (PARTITION BY
run_id)`. Every series that achieved any fold at all achieved that one, because `make_folds` keeps a
survivor's original `fold_id` and drops from the oldest end. The fold is derived from the rows
rather than read from the config on purpose: a view has no config to read.

Second, the error is **pooled, not averaged**: `SUM(|y_true − yhat|) / SUM(|y_true|)` over every
series at once. That gives one WAPE for the whole panel, instead of the mean of per-series WAPEs
where a single near-zero series can dominate the average.

Columns: `holdout_fold_id`, `n_series`, `n_points`, `pooled_wape`, `pooled_mae`, and the
`first_forecast_date` / `last_forecast_date` bounding the window that was scored. Carry `n_series`
into any comparison you draw — equal `n_series` across the rows is the evidence that the models
answered the same question, and unequal `n_series` is itself a finding.

It reads `backtest_oof` rather than `forecast_metadata` because that is the only table holding
per-fold truth; `forecast_metadata` stores one rolled-up row per cell and cannot be restricted to a
fold after the fact. Ensembles appear here beside the base models, because they write their blended
out-of-fold rows into that same table.

---

See also: [configuration_reference.md](./configuration_reference.md) (the run *is* the config that
lands in `raw_config`) · [running_and_reviewing.md](./running_and_reviewing.md) (submit, watch, and
review through these views) · [`registry/ddl.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/registry/ddl.py) /
[`registry/views.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/registry/views.py) (the source of truth).
