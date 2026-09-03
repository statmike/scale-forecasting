# Configuration reference

One run is **one JSON config**. It is validated into a frozen `RunConfig`
(`src/scale_forecasting/config.py`), logged verbatim to the run registry, and *is* the experiment
record — behavior changes come from this file, not code edits. This page documents every
field.

Two invariants apply everywhere:

- **Immutable + strict.** Every section is frozen and `extra="forbid"` — an unknown key is a
  validation error, not a silent no-op. A typo fails fast with a clear message.
- **Config-derived identity.** The `run_id` is a digest of the config (`make_run_id`), so re-running
  the same config is idempotent at the logical level. Give each execution a fresh `run_name` (the
  notebooks timestamp it) if you want a distinct run.

Load + validate a file with `load_config(path)`; every failure mode (missing file, bad JSON, invalid
schema) surfaces as a single `ConfigError`.

## Top level — `RunConfig`

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `run_name` | `str` | *required* | Human name for the run. |
| `data` | `DataConfig` | *required* | Where the series come from and their shape. |
| `python_runtime` | `"spark"` \| `"ray"` | `"spark"` | Run-level **default** runtime for the Python model families; each family can override it (see below). |
| `models` | `list[str]` | *required* (≥1) | Model names to run (see `playground --list`). |
| `features` | `FeaturesConfig` | `{}` | Optional feature engineering. |
| `backtest` | `BacktestConfig` | `{}` | Time-series cross-validation. |
| `hpo` | `HpoConfig` | `{}` | Hyperparameter optimization. |
| `ensemble` | `EnsembleConfig` | `{}` | Consensus across base models. |
| `compute` | `ComputeConfig` | `{}` | Runtime scale + cost guardrails. |

**Cross-field rules** (enforced after parsing):

- Duplicate entries in `models` → error.
- `hpo.enabled` requires `backtest.enabled` → error otherwise (HPO tunes on folds).
- `ensemble.enabled` without `backtest.enabled` → the **learned** strategies (`nnls`/`ridge`/`xgb`)
  are dropped with a warning (they need OOF); calculated strategies remain. Not an error.

**`python_runtime` — the run-level default runtime for the Python model families** (the native family
always runs in parallel in BigQuery, regardless of this choice):

- `spark` (default) — Dataproc Serverless. The **100k CPU workhorse**; it fans out one task per
  `(series, model)` cell (series cross-joined with the family's models), so a family's job finishes in
  ~its slowest cell.
- `ray` — Ray on Vertex AI. Its reason to exist is **fractional-GPU packing** for NeuralProphet (many
  series share one T4); the Ray `compute` knobs apply.

A run resolves its models into **one job per family** (`statistical` / `ml` / `deep_learning`, plus
`native` in BigQuery), all running in parallel under one `run_id`. Each Python family runs on
`python_runtime` unless it is overridden **per family** via `compute.families` (below) — so one run
can put its statistical family on Spark and its deep-learning family on Ray. See the DAG model in
[architecture.md](./architecture.md).

## `data` — `DataConfig`

| Field | Type | Default | Constraint | Purpose |
|-------|------|---------|-----------|---------|
| `source_table` | `str` | *required* | — | Source table for series (e.g. `source_series_iceberg` / `source_series_native`). |
| `ts_id_col` | `str` | `"ts_id"` | — | Series-id column. |
| `date_col` | `str` | `"ds"` | — | Date column. |
| `target_col` | `str` | `"y"` | — | Target column. |
| `freq` | `str` | `"D"` | one of `D W MS ME h` | Series frequency (pandas ≥2.2 spellings). Sets the seasonal period used by the models — `D`=daily (period 7), `W`=weekly (52), `MS`/`ME`=month start/end (12), `h`=hourly (24). An unsupported freq is rejected at validation. |
| `horizon` | `int` | `28` | `> 0` | Forecast horizon (steps). |
| `series_limit` | `int` \| `null` | `null` | `> 0` when set | `null` = all series; an int subsets the shipped data (demo small → scale large). |

## `features` — `FeaturesConfig`

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `holidays` | `list[str]` | `[]` | Holiday country codes to add (e.g. `["US"]`). |
| `transform` | `"none"` \| `"log1p"` \| `"boxcox"` | `"none"` | Target transform, inverted on output. |
| `exog` | `list[str]` | `[]` | Exogenous driver columns — a **started-but-unexampled** seam: consumed by `sarimax`/`ucm`/`prophet`/`lightgbm`/`xgboost`, but the shipped source is univariate (bring your own table with these columns to use it). |
| `lags` | `list[int]` | `[]` | Lag features. |
| `fourier` | `bool` | `false` | Fourier seasonality terms. |
| `level_shift` | `bool` | `false` | Detect one abrupt regime change and add it as a `level_shift` step dummy. |

**What each option produces** ([`features.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/features.py)):

- **`transform`** — reshapes the target before fitting and inverts it on output:
  - `none` — identity (no constraint on `y`).
  - `log1p` — `log(1+y)` forward, `expm1` back. Tames multiplicative growth / right-skew. **Requires
    `y >= -1`.** Stateless.
  - `boxcox` — a **power transform** whose λ is fit **per series** by maximum likelihood; the *same* λ
    is reused across backtest folds and the final fit. Strongest variance-stabilizer of the three.
    **Requires strictly positive `y` (`y > 0`).**
- **`holidays`** — adds a single `is_holiday` flag (1.0 on holiday dates) from the `holidays` package
  for each ISO country code (e.g. `["US", "GB"]`). The same calendar feeds the BigQuery-native models,
  so holiday handling matches across runtimes. An unknown code fails fast.
- **`lags`** — for each integer `L`, adds a `lag_{L}` column (the target shifted back `L` steps).
  Gives the ML models (`lightgbm`/`xgboost`) autoregressive signal. Values must be positive.
- **`fourier`** — adds sine/cosine **yearly** seasonality terms (order 3 → 6 columns). Smooth periodic
  signal for the regression-based models.
- **`exog`** — passes named driver columns straight from the source table through to the models that
  accept exogenous regressors. See the univariate-shipped-data caveat above.
- **`level_shift`** — detects a single abrupt **regime change** in the series and adds one
  `level_shift` column: `0` before the changepoint, `1` from it onward, and `1` across the whole
  forecast horizon. It is a **step, not a spike** — that persistence is exactly what distinguishes a
  level shift from an outlier, and it lets a regression model absorb the jump as one coefficient
  instead of fitting the average of two regimes and staying biased for the entire horizon. Detection
  is a single-changepoint scan standardized by a robust (MAD-based) noise estimate, accepted only
  above 3σ; below that the column is all zeros, because a spurious regressor on one series in ten
  thousand is a bad forecast nobody reviews. Worth turning on when your history contains
  re-baselinings, store openings/closings, or a unit-of-measure change — the shipped example data
  contains them by construction (`data_gen.generator` plants one per series with archetype-specific
  probability).

**How these features are valued over the forecast horizon.** A model is fit on history and then
asked to predict dates it has never seen, so the same feature columns have to exist for those
dates too (`features.build_future_features`). Most of them are a deterministic function of the
date and are therefore **recomputed exactly** at the future dates — `is_holiday` reflects the
holidays that actually fall in the horizon, and the Fourier terms continue the real seasonal
phase. `level_shift` is carried forward as `1`. Configured `lag_L` columns are genuine
observations for the first `L` steps and then hold the last observed level.

The one exception is **`exog`**, which is genuinely unknown until the future arrives: those
columns fall back to the most recent `horizon` observed rows, so an exog-driven forecast is
*indicative* rather than authoritative. To get a real forward-looking exog path, extend your
source table past the target cutoff with the driver values (a price plan, a promo calendar, a
published index) — the read picks them up with no config change.

## `backtest` — `BacktestConfig`

Off by default (cheapest first run). Turn it on to get an OOF metric panel — and it's a prerequisite
for HPO and learned ensembles.

| Field | Type | Default | Constraint | Purpose |
|-------|------|---------|-----------|---------|
| `enabled` | `bool` | `false` | — | Turn backtesting on. |
| `scheme` | `"expanding"` \| `"sliding"` | `"expanding"` | — | CV window scheme. |
| `n_folds` | `int` | `3` | `≥ 1` | Number of folds. |
| `horizon` | `int` | `28` | `> 0` | Per-fold forecast horizon. |
| `step` | `int` | `28` | `> 0` | Step between folds. |
| `min_train` | `int` | `180` | `> 0` | Minimum training length. |
| `decision_metric` | see below | `"wape"` | — | Metric folds are judged on. |

**`scheme` — how the training window moves** ([`backtest.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/backtest.py)).
Folds are anchored from the **end** of each series: the latest fold validates on the final `horizon`
points, and each earlier fold steps its validation window back by `step`. The two schemes differ only
in where training *starts*:

- **`expanding`** (default) — training uses **all history** from the series start up to each fold's
  validation point. The train window grows fold to fold. Best default: every fold sees maximum
  history.
- **`sliding`** — training uses a **fixed-width** window of the last `min_train` observations
  immediately before each fold. Older history is dropped. Use when the series' behavior drifts and
  recent history is more representative than old history.

`n_folds`, `horizon`, `step`, and `min_train` lay the folds out together; a series needs at least
`min_train + horizon + (n_folds−1)·step` observations, or it's **skipped** for backtesting (it doesn't
sink the run). Features are built once and a **fresh** model is fit per fold, so no state leaks across
folds and `train_end == val_start` always (no leakage).

**`decision_metric` — what folds are judged on** (definitions in
[`metrics.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/metrics.py); `err = yhat − y_true`). This single choice drives
fold selection, HPO's objective, `inverse_error` weighting, and `prune_threshold`.

| Metric | Definition | Notes |
|--------|-----------|-------|
| `mae` | mean(\|err\|) | Mean absolute error. Scale-dependent. |
| `rmse` | √mean(err²) | Penalizes large misses more than `mae`. |
| `mse` | mean(err²) | Squared error; `rmse` without the root. |
| `mape` | mean(\|err\| / \|y_true\|) | % error. **NaN if any `y_true == 0`.** |
| `smape` | mean(2\|err\| / (\|y_true\|+\|yhat\|)) | Symmetric %; bounded, handles zeros gracefully. |
| `wape` | Σ\|err\| / Σ\|y_true\| | Weighted absolute % error — the scale-safe default. NaN only if the series sums to 0. |
| `mase` | mae / mae(naïve-1-step) | Scaled vs. a naïve forecast; **needs training history**. <1 beats naïve. |
| `rmsse` | rmse / rmse(naïve-1-step) | Squared analog of `mase`; **needs training history**. |
| `bias` | mean(err) | Mean error — sign shows over/under-forecast. HPO minimizes \|bias\|. |
| `coverage` | fraction of `y_true` inside [lower, upper] | **Needs prediction intervals**; want it near the nominal level. |
| `pinball` | avg quantile loss at the 0.1 / 0.9 bounds | **Needs prediction intervals**; scores interval sharpness+calibration. |

Pick `wape` (default) or `smape` for a robust scale-independent choice; `mase`/`rmsse` when you want
to beat a naïve baseline; `coverage`/`pinball` only when you care about the prediction intervals
(ensemble OOF has no intervals, so those two read NaN for ensembles).

## `hpo` — `HpoConfig`

Optional Optuna tuning on the aligned backtest. **Requires `backtest.enabled`.**

| Field | Type | Default | Constraint | Purpose |
|-------|------|---------|-----------|---------|
| `enabled` | `bool` | `false` | — | Turn HPO on. |
| `engine` | `"optuna"` | `"optuna"` | — | HPO engine. |
| `n_trials` | `int` | `20` | `> 0` | Trials per study. |
| `granularity` | `"fleetwide"` \| `"per_series"` | `"fleetwide"` | — | The cost knob — see below. |
| `sample_size` | `int` | `20` | `> 0` | Fleetwide only: how many series to tune on before applying the winner fleet-wide. |

**Fleetwide vs. per-series — the cost trade-off.** `fleetwide` (default) tunes each model **once**,
on a driver-side sample of `sample_size` series, then applies the winning hyperparameters to the
whole fleet — one study per model, cost independent of series count. `per_series` tunes **inside
every cell** — the best accuracy a model can reach on each series, but `n_trials` studies *per
series*, so cost scales with the fleet. Start fleetwide; reach for per-series only when a model's
optimal hyperparameters genuinely vary series to series and the accuracy is worth the spend.

Tuned hyperparameters flow to the workers through the engine, **not** through the config — so HPO
never shifts the config-derived `run_id`, keeping runs reproducible and idempotent.

## `ensemble` — `EnsembleConfig`

Consensus across the base models (scored onto the same leaderboard). See
[running_and_reviewing.md](./running_and_reviewing.md) for the re-ensemble workflow.

| Field | Type | Default | Constraint | Purpose |
|-------|------|---------|-----------|---------|
| `enabled` | `bool` | `false` | — | Turn ensembling on. |
| `strategies` | `list` | `["median"]` | each a known strategy | Consensus strategies (run several at once). |
| `prune_threshold` | `float` | `0.0` | `≥ 0.0` | Drop base models weaker than this before blending. |

`strategies` ∈ **calculated** `mean, median, inverse_error` (no backtest needed) and **learned**
`nnls, ridge, xgb` (need `backtest.enabled`). A bare `"strategy": "nnls"` is accepted as shorthand
for `"strategies": ["nnls"]`. Run several at once — each produces its own `ensemble_<strategy>` rows
and earns a line on the leaderboard next to the base models.

**What each strategy does** (implementation in
[`ensembler.py`](https://github.com/statmike/scale-forecasting/blob/main/src/scale_forecasting/ensembler.py)):

| Strategy | Kind | How the blend is formed | Needs |
|----------|------|-------------------------|-------|
| `mean` | calculated | Unweighted arithmetic mean of the base forecasts, row by row. Missing models are skipped and weights renormalize over those present. | — |
| `median` | calculated | Unweighted row-wise **median** — robust to a single wild base forecast. | — |
| `inverse_error` | calculated | Weighted mean where each model's weight ∝ **1 / its error**, normalized to sum to 1. For the future forecast the error is the mean `decision_metric` from `forecast_metadata`; for the on-fold (OOF) score it's each series' own WAPE. A zero-error model captures the weight; if no error is usable it degrades to `mean`. | (weights sharpen with `backtest`) |
| `nnls` | learned | **Non-negative least squares** meta-learner: solves for weights ≥ 0 that best reconstruct the truth from the base models' out-of-fold predictions. No intercept; weights can't go negative. | `backtest.enabled` |
| `ridge` | learned | **L2-regularized linear regression** (closed-form, α=1.0) over the same OOF matrix. Weights *may* be negative (a model can be a corrective term). | `backtest.enabled` |
| `xgb` | learned | **Gradient-boosted** meta-learner (`XGBRegressor`, 200 trees, depth 3) fit on the OOF matrix; captures non-linear interactions between base models. Reported "weights" are its normalized feature importances. | `backtest.enabled`, `xgboost` installed |

The three **learned** strategies train **only** on `backtest_oof` rows (never in-sample), so leakage
is structurally impossible — which is exactly why they require `backtest.enabled` (without it they're
dropped at config-load with a warning, not an error). Each fitted meta-learner is pickled to a GCS
artifact for lineage.

**`prune_threshold`** applies only to the **calculated** strategies: when `> 0`, any base model whose
mean `backtest.decision_metric` is *worse than* (greater than) the threshold is dropped from the blend
fleet-wide before combining. `0.0` (default) prunes nothing.

## `compute` — `ComputeConfig`

Runtime scale, dependency delivery, and cost guardrails. Defaults are tuned for a first run; the Ray
knobs only matter for a family that runs on Ray.

| Field | Type | Default | Constraint | Purpose |
|-------|------|---------|-----------|---------|
| `families` | `dict[family → FamilyCompute]` | `{}` | keys ∈ `statistical`/`ml`/`deep_learning` | Per-family runtime/hardware overrides (see below). |
| `max_parallelism` | `int` | `1000` | `> 0` | Max parallel tasks. |
| `bucket_target_cells` | `int` | `8` | `> 0` | Target cells per Spark bucket (shuffle-partition sizing). |
| `max_executors` | `int \| null` | `null` | `> 0` | Operator ceiling on the Spark fleet — most executors a batch may scale to, most workers a cluster may hold. `null` sizes to the fan-out alone, which at 100k series asks for hundreds and is rejected outright by a regional CPU quota. Budget for concurrency: a run's families submit together. |
| `machine_family` | `"auto"` \| `"n1"` \| `"n2"` \| `"n2d"` \| `"e2"` \| `"c2"` | `"auto"` | — | GCE machine family for a **Dataproc cluster's** master + CPU workers (`"auto"` = `n1`). No-op on Serverless and on GPU workers — see below. |
| `spark_deps` | `"packed_venv"` \| `"container"` | `"packed_venv"` | — | How a **Dataproc cluster** family gets its dependencies. `"container"` raises: it is a Serverless mechanism. See `cluster_deps._resolve_cluster_deps`. |
| `persist_models` | `bool` | `false` | — | Persist each fitted model as a GCS artifact (lineage). |
| `use_gpu` | `bool` | `false` | — | Enable GPU (Ray). |
| `gpu_type` | `str` | `"T4"` | — | GPU accelerator type. |
| `gpu_fraction` | `"auto"` \| `float` | `"auto"` | float ∈ `(0, 1]` | `"auto"` = profile-driven fractional GPU, else a fixed fraction. |
| `budget_usd` | `float` | `50.0` | `≥ 0.0` | Cost guardrail (USD). |
| `ray_cluster_name` | `str` \| `null` | `null` | — | Reuse a standing Ray cluster by name; `null` = ephemeral. |
| `ray_regions` | `list[str]` \| `null` | `null` | — | Priority-ordered candidate regions for the ephemeral cluster. |
| `ray_head_machine_type` | `str` | `"n1-standard-16"` | — | Head-node type (don't drop below, or job submit hangs). |
| `ray_cpu_machine_type` | `str` | `"n1-standard-8"` | — | CPU worker-pool machine type. |
| `ray_gpu_machine_type` | `str` | `"n1-standard-8"` | — | GPU worker-pool type (must be N1 for T4). |
| `accelerator_count` | `int` | `1` | T4 ∈ `{1,2,4}` | GPUs per GPU worker node. |
| `ray_target_cells_per_slot` | `int` | `8` | `> 0` | Cells one worker slot chews before a node is added. |
| `ray_max_nodes` | `int` | `16` | `> 0` | Shared per-pool ceiling; the fallback when a pool's own max is unset. |
| `ray_autoscale` | `bool` | `true` | — | Autoscale each worker pool between its min/max (default). `false` restores fixed-size sizing (the derived `node_count`, no autoscaling spec). |
| `ray_cpu_min_nodes` | `int` | `1` | `> 0` | CPU pool floor. Low = shrink when the queue drains. |
| `ray_cpu_max_nodes` | `int` \| `null` | `null` | `> 0` | CPU pool ceiling; `null` falls back to `ray_max_nodes`. Raise to grow under load. |
| `ray_gpu_min_nodes` | `int` | `1` | `> 0` | GPU pool floor. Low = shrink idle (expensive) T4s. |
| `ray_gpu_max_nodes` | `int` \| `null` | `null` | `> 0` | GPU pool ceiling; `null` falls back to `ray_max_nodes`. Cap independently for cost. |
| `gpu_calibration_samples` | `int` | `3` | `> 0` | Series to profile for auto `gpu_fraction`. |
| `gpu_safety_margin` | `float` | `1.3` | `> 1.0` | Headroom multiplier on measured peak GPU memory. |
| `ray_read_mode` | `"driver_collect"` \| `"ray_data"` | `"driver_collect"` | — | Ray source reader: the proven Storage Read client, or `ray.data.read_bigquery` (same Storage Read API, opt-in). |
| `read_max_streams` | `int` | `0` | `≥ 0` | Max Storage Read streams for the source read, shared by the Spark connector (`maxParallelism`) and Ray's `driver_collect` reader (`max_stream_count`). `0` lets the server size it from the table; a positive value caps read parallelism (e.g. to fit a slot/quota budget). Inert for the `ray_data` path and BigQuery-native models. See [reading_source_data.md](./reading_source_data.md). |

**`machine_family` — one knob, three deliberate boundaries.** It selects the GCE family for a
**Dataproc cluster's** master and CPU workers (`worker_machine_type` / `master_machine_type`), and
`"auto"` resolves to `n1` — today's shipped shape, so an existing config renders an identical
cluster. What it does *not* do is as important:

- **It does not pick a size.** Cores are fixed (master 4, workers 8) because the profiler derives
  the executor shape *from* the machine; letting you set both gives two knobs that can disagree.
- **It does not reach GPU workers.** The accelerator dictates the machine — a T4 is an add-on card
  that only attaches to `n1`, an L4 is bundled inside `g2` — so a family override there would ask
  GCE for a shape it does not sell. A run spanning both hardware kinds gets its CPU workers on your
  family and its GPU workers on the accelerator's.
- **It does nothing on Serverless or Ray.** Serverless has no machine concept at all (its shape is
  executor cores/memory properties); the Ray pools have their own explicit
  `ray_*_machine_type` knobs.

The offered families are exactly those `resources` can price (`_MEMORY_PER_CORE_GIB`), so the
sizing plan stays honest for whichever you pick — choosing `n2` moves the worker from 30 GiB to
32 GiB and the executor split follows. An unlisted family is rejected at config load rather than at
cluster create.

**Autoscaling (default):** each Ray worker pool scales between its own `[min, max]`; a `null` pool
max resolves to `ray_max_nodes`. Config validation requires `min ≤ resolved max` per pool. The
*initial* pool size stays a deterministic function of the config (fan-out ÷
`ray_target_cells_per_slot`, clamped into the bounds), so the run remains reproducible — the whole
spec is hashed into `run_id` and stamped to `run_registry.job_telemetry`. Set `ray_autoscale=false`
to opt out and get a fixed-size pool instead (no autoscaling spec at all) — worth reaching for when
you want a cluster whose cost is flat and predictable rather than demand-shaped. See the Ray runtime
in [architecture.md](./architecture.md).

### `compute.capacity` — how hard to look for room

"Resources are not available" is a **state**, not an exception. When a create fails for want of
machines the launcher walks its candidate places (Ray: `ray_regions`; a Dataproc cluster: the
zone/region fallback map; Serverless: the region), and if none has room it backs off and walks them
again until a budget runs out. While it waits the family's `run_jobs` row reads `AWAITING_CAPACITY`
and carries a ledger of every attempt; on exhaustion the row is `FAILED` with
`failure_reason = CAPACITY_EXHAUSTED`. See
[troubleshooting.md](./troubleshooting.md#capacity--the-cloud-has-no-room) for reading it.

`{"enabled": false}` (top level) collapses every service to a single pass with no back-off — the
pre-retry behaviour exactly. Otherwise each service takes a **partial** override: name only the
number you have an opinion about and inherit the rest.

| Field | Type | Default | Constraint | Purpose |
|-------|------|---------|-----------|---------|
| `enabled` | `bool` | `true` | — | `false` = one pass over the candidates, no back-off, all services. Beats an authored `max_passes`. |
| `ray` \| `dataproc_cluster` \| `dataproc_serverless` | `object` | `{}` | — | Per-service partial override; unset fields inherit the shipped default below. |

Per-service fields, all optional, `0` disables that bound:

| Field | Type | `ray` | `dataproc_cluster` | `dataproc_serverless` | Purpose |
|-------|------|-------|--------------------|-----------------------|---------|
| `max_attempts` | `int ≥ 0` | `6` | `8` | `10` | Total attempts across all candidates. |
| `max_wall_seconds` | `float ≥ 0` | `3600` | `2700` | `1800` | Clock budget for the whole walk. |
| `max_passes` | `int ≥ 0` | unbounded | unbounded | unbounded | Full sweeps of the candidate list. |
| `backoff_seconds` | `float ≥ 0` | `120` | `60` | `30` | First wait after a fruitless pass. |
| `backoff_multiplier` | `float ≥ 1.0` | `2.0` | `2.0` | `2.0` | Growth per pass. |
| `backoff_max_seconds` | `float ≥ 0` | `600` | `300` | `120` | Cap on the wait. |

The three differ because the services do. A Vertex Ray GPU provision costs ~12 minutes per attempt,
so it gets the fewest tries and the longest patience; a Serverless batch is rejected in seconds, so
retrying is nearly free and the clock is the bound that matters.

**BigQuery has no entry, deliberately.** Slot contention is resolved BigQuery-side and surfaces as
latency, not as a create that failed somewhere else it could be tried. There is no candidate list to
walk.

**None of this moves the `run_id`.** Patience is an operational knob, not part of what was asked
for — same rule as `compute.profile.source`. If it hashed into the digest, waiting longer for a GPU
would fork your run identity and break dedupe-on-read.

```json
"compute": {
  "capacity": {
    "ray": {"max_wall_seconds": 7200},
    "dataproc_serverless": {"max_attempts": 20}
  }
}
```

### `compute.profile` — measured compute profiling

Sizing is otherwise a pure cell **count** (`n_series × n_models × n_folds`, divided by a flat
cells-per-slot constant). That arithmetic cannot know that a deep-learning fit and a naive mean
differ by orders of magnitude, so the fleet is provisioned for the count rather than for the work.
`compute.profile` is the machinery for replacing that guess with a measurement.

Think of it as the general form of `gpu_calibration_samples` / `gpu_safety_margin` above, which
already do exactly this for one axis (GPU bytes), one model, one runtime.

#### What this actually does today — read this before setting anything

Two things are true at once, and conflating them will mislead you.

**Fleet shaping is on, everywhere, and it does not need a measurement.** Every Spark job now carries
a derived properties overlay: executor cores, heap and memoryOverhead, the dynamic-allocation band,
`spark.task.cpus` bounded by the accelerator, matching thread pins, and — on a Dataproc cluster — a
worker count derived from the run's fan-out. That arithmetic runs from the config alone. Setting
`mode = "off"` is what turns it off, and that is the escape hatch to reach for if a fleet shape ever
misbehaves in production.

**Measuring *inside the run that needs it* does not work on Spark, and structurally cannot.**
`spark.executor.cores` and `spark.task.cpus` are fixed at submit (Serverless) or at create (cluster),
before any of our code reaches the cluster; and the submit host is deliberately lean, carrying no
model stack to fit with. A same-run pre-pass has nowhere to run that is both early enough to matter
and equipped to measure anything. Only the Ray engine can do it — it requests per-task `num_cpus` /
`num_gpus` in-run against an autoscaling pool — and with the defaults below (`mode = "auto"`,
`min_cells = 1000`) it does not fire for a small run. So on Spark, `"auto"` and `"always"` size
identically today; only `"off"` changes anything.

**So measurement is decoupled from consumption: one run produces the evidence, a later run is sized
from it.** That is what `measure` is. Every cell of an ordinary run records what it cost — three
cheap probes around a fit that was happening anyway — onto `forecast_metadata`. A completed `run_id`
is therefore a measured cost model you can point a bigger run at. It is on by default, because the
run you wish you had measured is always the one you already finished.

**Three knobs, three questions.** `mode` is the master switch — whether the fleet arithmetic runs at
all, and the threshold the in-run Ray pre-pass compares against. `measure` decides what evidence this
run **produces**. `source` decides what evidence it **consumes**. They are orthogonal because the
questions are: a run can harvest without consuming (the first one ever), consume without harvesting
(a one-off resize), or do both (the ordinary case, and the default). `mode = "off"` vetoes all of it,
so there is still exactly one switch that makes the whole feature inert.

| Field | Type | Default | Constraint | Purpose |
|-------|------|---------|------------|---------|
| `mode` | `"off"` \| `"auto"` \| `"always"` | `"auto"` | — | `off` disables **both** the derived fleet overlay and any measurement — the escape hatch, and the only value that changes a Spark run today. `auto` measures when the cell count reaches `min_cells`; `always` measures unconditionally. Both currently affect the Ray path only. |
| `samples` | `int` | `8` | `> 0` | Series fitted in the pre-pass, spread across length/complexity strata. |
| `min_cells` | `int` | `1000` | `> 0` | The threshold `auto` compares the cell count against. |
| `memory_margin` | `float` | `1.3` | `> 1.0` | Headroom on the measured **max**, which sizes the slot. |
| `time_margin` | `float` | `1.2` | `> 1.0` | Headroom on the measured **median**, which sizes the fleet. |
| `measure` | `"off"` \| `"harvest"` \| `"controlled"` | `"harvest"` | — | What evidence this run **produces**. See the table below. |
| `source` | `"auto"` \| `"baseline"` \| `"none"` \| `"<run_id>"` | `"auto"` | keyword or a well-formed run_id | What evidence this run **consumes**. See the precedence chain below. |

| `measure` | What the run does | Use it when |
|-----------|-------------------|-------------|
| `"harvest"` | Records per-cell CPU time, absolute process memory, peak device bytes, the thread cap and `n_obs` onto `forecast_metadata`. Changes nothing about how the run executes. | Always — this is the default. |
| `"controlled"` | Harvest, **plus** the Spark fleet leaves the native thread pools uncapped so a fit's measured `effective_cores` reflects the library instead of the cap. | A deliberate calibration run, and never a production one: the executors are knowingly oversubscribed, so the run is slower and its shape is not a real run's shape. The translation carries a note saying so. |
| `"off"` | Writes NULL in all five columns. | You have a reason to skip three probes per fit. |

| `source` | Where the numbers come from | Provenance basis |
|----------|-----------------------------|------------------|
| `"auto"` | The newest completed run whose harvest matches this run's data signature; failing that, the shipped baseline; failing that, static config. Resolved **at plan time** and written into the staged config as a concrete `run_id`, so the run records what it actually sized from rather than a search that might resolve differently tomorrow. | `measured` / `reference` |
| `"<run_id>"` | That run's harvest, whatever its signature. Naming a run is a decision, so it is honoured — a drifted signature comes back as a warning, not a substitution. | `measured` / `reference` |
| `"baseline"` | The shipped, versioned reference measurements — see below. Real numbers, taken on reference hardware and reference data, never on yours. | `reference` |
| `"none"` | Nothing is consulted; size from declared config. | — |

The precedence, outside-in: **explicit `compute` settings > `compute.profile.source` > shipped
baseline > static config.** Anything you set by hand always wins; the evidence only fills what you
left to be derived, and the static floor is the behaviour this product shipped with.

Every resolved profile carries a `provenance` block naming its basis, the `run_id` and timestamp
behind it, the data signature it was measured on, and any drift warnings. `measured` means the
evidence matches this run's signature on every axis both sides can see; **`reference` means measured,
but not on your data**. That third value exists because a pinned profile from months ago on a
different table is worse than no profile at all — precisely because it looks authoritative. A
mismatch is never silent, and never fatal: sizing off drifted evidence still beats sizing off none.

If BigQuery is unreachable when the source is resolved, the run logs a warning and falls through the
rest of the chain — which today means the shipped baseline. Evidence is an optimisation; a registry
hiccup must not stop a run from submitting.

**What ships in the baseline.** It was cut from a real 100,000-series run on Ray (`ray_100k`,
recorded in [validation.md](validation.md)): daily series of 1,460 observations, four models across
the `statistical` and `ml` families. It therefore sizes those two families and **not**
`deep_learning`, and it carries no GPU bound — that run had neither. An unmeasured family resolves
to nothing rather than to a guess, so a run with a deep-learning family gets the static arithmetic
for it and measured numbers for the rest. The number it is really there for is
`slot_cores: 1`: every one of those fits measured single-threaded, and that is a property of the
libraries rather than of your panel, so it transfers in a way a memory bound does not.

**What the resolved profile actually changes.** Only the memory axis. The executor cores, the thread
pins, the warm `initialExecutors`, the device-aware `spark.task.cpus` and the worker/executor counts
all follow from the fan-out and the machine type alone and are emitted with or without evidence.
Memory is the one thing that cannot be derived — a Serverless executor's shape is fixed at *submit*
and a cluster's at *create*, both before any of our code runs — so with no profile the memory
properties are simply not emitted and the platform's defaults stand, exactly as before.

**Reading back what a run decided.** The whole decision is stamped into the run header's
`job_telemetry`, one entry per family job under `sizing.<family>`, and surfaced by `v_run_summary`
as its `sizing` column: the fleet the arithmetic asked for, what that became in platform settings,
and the profile it was sized off (with the provenance naming whose run supplied the evidence). So
"why was this run this shape" is a query against the registry rather than a hunt through driver
logs — see [output_schemas.md](./output_schemas.md).

**Why the two margins differ, and why they apply to different tails.** Over-estimating time buys
extra slots, which costs money; under-estimating memory OOM-kills the task, which costs the run.
Asymmetric risk, asymmetric margin — so memory carries the wider one. They also attach to different
statistics on purpose: a slot must hold the *worst* series that lands in it (max), while a fleet is
sized for *typical* work (median). Sizing the fleet off the worst case over-provisions every run;
sizing memory off the median OOM-kills it. Using one tail for both is the mistake the split exists
to prevent.

`compute.profile` is part of the `run_id` digest, like everything else under `compute`. It changes
the resource shape rather than the forecasts, so that is a deliberate choice: the config is the
experiment record, and a run whose fleet was sized differently is not the same run for performance
purposes. One practical consequence: adding these fields moved every pre-existing `run_id`, so a
config saved before the profiler existed no longer re-derives the id it originally produced — and
adding `measure` and then `source` moved them again. Re-running an older config produces a new
`run_id` and therefore a new run rather than a resumed one.

**None of the derived fleet arithmetic has run on live infrastructure yet.** It is offline-proven
self-consistent — the legal-value snapping, the AM reserve, the worker derivation all have unit
tests — but Dataproc has never been asked to accept it. See
[Validation ledger](./validation.md), where the Spark rows are currently `STALE` for exactly this
reason.

### `compute.families` — per-family runtime & hardware

By default every Python family runs on the run-level `python_runtime` on CPU. `compute.families` maps
a family name to a `FamilyCompute` that overrides that placement for **that family only** — the lever
behind "one run, a job per family, each on its own runtime". Every `FamilyCompute` field is optional;
an unset field inherits the run-level default. The `native` family is never listed here (it always
runs in BigQuery).

| Field | Type | Options | Purpose |
|-------|------|---------|---------|
| `runtime` | `str` | `"spark"` \| `"ray"` | Runtime for this family (overrides `python_runtime`). |
| `spark_mode` | `str` | `"serverless"` \| `"cluster"` | Spark launch mode (Spark only). `"cluster"` runs on a Dataproc cluster — needed for a T4 GPU on Spark. |
| `spark_cluster_name` | `str` | — | Reuse an existing Dataproc cluster by name (requires `spark_mode="cluster"`). |
| `hardware` | `str` | `"cpu"` \| `"gpu"` | Hardware profile for this family (GPU only for `deep_learning`). |
| `gpu_type` | `str` | `"T4"` \| `"L4"` | GPU type when `hardware="gpu"`. |

**Cross-field rules** (enforced at config-load):

- `spark_mode` / `spark_cluster_name` are valid only when `runtime="spark"`; `spark_cluster_name`
  requires `spark_mode="cluster"`.
- A GPU (`hardware="gpu"` or `gpu_type` set) is allowed **only** for the `deep_learning` family.
- A T4 on Spark requires `spark_mode="cluster"` (Serverless can't attach a T4) — or route the family
  to `runtime="ray"`.
- `hardware="cpu"` with a `gpu_type` set → error (drop `gpu_type` or set `hardware="gpu"`).

**How many clusters a run creates.** One per **hardware kind** among its ephemeral cluster families,
not one per run — a Dataproc cluster has exactly one worker machine type, so it is a CPU cluster or a
GPU cluster and cannot be both. A run whose cluster families are all CPU gets one cluster named
`sf-cluster-<run_id>`; a run mixing CPU and GPU families gets `sf-cluster-<run_id>-cpu` and
`sf-cluster-<run_id>-gpu`, each sized only for the families that land on it. That is why a GPU
`deep_learning` family no longer makes the rest of the run pay for accelerators it never uses.

Ray is deliberately different: a Vertex Ray cluster carries separate CPU and GPU worker *pools*, so
a mixed run shares **one** Ray cluster. The asymmetry is the hardware, not an inconsistency.

**Cluster lifetime.** A cluster the run creates (no `spark_cluster_name`) is deleted when the run
ends — every one of them, including on a failure part-way through creating the second. Each also
carries server-side bounds so it cannot outlive the orchestrator that made it: it
self-deletes after **30 min idle** or **24 h** total, whichever comes first. Those are backstops for
the case where the run process is killed before its teardown runs — when teardown works they never
fire. Override with `SF_CLUSTER_IDLE_TTL` / `SF_CLUSTER_MAX_AGE` (seconds; `0` disables a bound).
They are environment, not config, so changing them does **not** change your `run_id`.

A cluster named by `spark_cluster_name` gets neither — the run does not create it and does not own
when it ends, so reclaiming it is yours to do.

```json
"compute": {
  "families": {
    "deep_learning": { "runtime": "ray", "hardware": "gpu", "gpu_type": "T4" }
  }
}
```

That routes the deep-learning family (e.g. `neuralprophet`) to Ray-on-GPU while the statistical and ml
families stay on the default Spark runtime and the native family runs in BigQuery — four families,
three runtimes, one `run_id`. See [`configs/per_family_runtimes_demo.json`](https://github.com/statmike/scale-forecasting/blob/main/configs/per_family_runtimes_demo.json).

## A minimal config

```json
{
  "run_name": "my first run",
  "python_runtime": "spark",
  "data": { "source_table": "source_series_iceberg", "horizon": 28, "series_limit": 100 },
  "models": ["theta", "holtwinters", "arima_plus"],
  "features": { "holidays": ["US"] }
}
```

`theta`/`holtwinters` run on Spark; `arima_plus` runs in BigQuery — both under one `run_id`, in
parallel. See [`configs/`](https://github.com/statmike/scale-forecasting/tree/main/configs) for worked examples (demo and 100k),
[running_and_reviewing.md](./running_and_reviewing.md) to submit and review one, and
[output_schemas.md](./output_schemas.md) for the tables the run writes to (the whole config lands
verbatim in `run_registry.raw_config`).
