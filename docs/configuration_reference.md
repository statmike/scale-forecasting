# Configuration reference

One run is **one JSON config**. It is validated into a frozen `RunConfig`
(`src/scale_forecasting/config.py`), logged verbatim to the run registry, and *is* the experiment
record (G2/G3) — behavior changes come from this file, not code edits. This page documents every
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
| `python_runtime` | `"spark"` \| `"ray"` | `"spark"` | Which Python runtime runs the Python models. |
| `spark_method` | `"explode"` \| `"multi"` \| `"naive"` \| `null` | `null` | Spark fan-out strategy; only meaningful when `python_runtime="spark"`. |
| `models` | `list[str]` | *required* (≥1) | Model names to run (see `playground --list`). |
| `features` | `FeaturesConfig` | `{}` | Optional feature engineering. |
| `backtest` | `BacktestConfig` | `{}` | Time-series cross-validation. |
| `hpo` | `HpoConfig` | `{}` | Hyperparameter optimization. |
| `ensemble` | `EnsembleConfig` | `{}` | Consensus across base models. |
| `compute` | `ComputeConfig` | `{}` | Runtime scale + cost guardrails. |

**Cross-field rules** (enforced after parsing):

- `python_runtime="spark"` with no `spark_method` → defaults to `"explode"`. `python_runtime="ray"`
  with *any* `spark_method` set → error (Ray has no fan-out method).
- Duplicate entries in `models` → error.
- `hpo.enabled` requires `backtest.enabled` → error otherwise (HPO tunes on folds).
- `ensemble.enabled` without `backtest.enabled` → the **learned** strategies (`nnls`/`ridge`/`xgb`)
  are dropped with a warning (they need OOF); calculated strategies remain. Not an error.

## `data` — `DataConfig`

| Field | Type | Default | Constraint | Purpose |
|-------|------|---------|-----------|---------|
| `source_table` | `str` | *required* | — | Source table for series (e.g. `source_series_iceberg` / `source_series_native`). |
| `ts_id_col` | `str` | `"ts_id"` | — | Series-id column. |
| `date_col` | `str` | `"ds"` | — | Date column. |
| `target_col` | `str` | `"y"` | — | Target column. |
| `freq` | `str` | `"D"` | — | Series frequency (pandas 3 spellings: `D`, `W`, `MS`, `ME`, `h`). |
| `horizon` | `int` | `28` | `> 0` | Forecast horizon (steps). |
| `series_limit` | `int` \| `null` | `null` | `> 0` when set | `null` = all series; an int subsets the shipped data (demo small → scale large). |

## `features` — `FeaturesConfig`

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `holidays` | `list[str]` | `[]` | Holiday country codes to add (e.g. `["US"]`). |
| `transform` | `"none"` \| `"log1p"` \| `"boxcox"` | `"none"` | Target transform, inverted on output. `boxcox` fits its λ per series by MLE (requires strictly positive `y`); `log1p` needs `y >= -1`. |
| `exog` | `list[str]` | `[]` | Exogenous driver columns — a **started-but-unexampled** seam: consumed by `sarimax`/`ucm`/`prophet`/`lightgbm`/`xgboost`, but the shipped source is univariate (bring your own table with these columns to use it). |
| `lags` | `list[int]` | `[]` | Lag features. |
| `fourier` | `bool` | `false` | Fourier seasonality terms. |
| `level_shift` | `bool` | `false` | Level-shift feature. |

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

`decision_metric` ∈ `mae, rmse, mse, mape, smape, wape, mase, rmsse, bias, coverage, pinball`.

## `hpo` — `HpoConfig`

Optional Optuna tuning on the aligned backtest (C5). **Requires `backtest.enabled`.**

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
for `"strategies": ["nnls"]`.

## `compute` — `ComputeConfig`

Runtime scale, dependency delivery, and cost guardrails. Defaults are tuned for a first run; the Ray
knobs only matter when `python_runtime="ray"`.

| Field | Type | Default | Constraint | Purpose |
|-------|------|---------|-----------|---------|
| `max_parallelism` | `int` | `1000` | `> 0` | Max parallel tasks. |
| `bucket_target_cells` | `int` | `8` | `> 0` | Target cells per Spark bucket (shuffle-partition sizing). |
| `machine_family` | `str` | `"auto"` | — | Executor machine family. |
| `spark_deps` | `"packed_venv"` \| `"container"` | `"packed_venv"` | — | Dependency delivery mechanism. |
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
| `ray_max_nodes` | `int` | `16` | `> 0` | Hard ceiling on cluster node count. |
| `gpu_calibration_samples` | `int` | `3` | `> 0` | Series to profile for auto `gpu_fraction`. |
| `gpu_safety_margin` | `float` | `1.3` | `> 1.0` | Headroom multiplier on measured peak GPU memory. |
| `ray_read_mode` | `"driver_collect"` \| `"ray_data"` | `"driver_collect"` | — | Ray source reader: the proven Storage Read client, or `ray.data.read_bigquery` (same Storage Read API, opt-in). |

## A minimal config

```json
{
  "run_name": "my first run",
  "python_runtime": "spark",
  "spark_method": "explode",
  "data": { "source_table": "source_series_iceberg", "horizon": 28, "series_limit": 100 },
  "models": ["theta", "holtwinters", "arima_plus"],
  "features": { "holidays": ["US"] }
}
```

`theta`/`holtwinters` run on Spark; `arima_plus` runs in BigQuery — both under one `run_id`, in
parallel. See [`configs/`](../configs) for worked examples (demo and 100k),
[running_and_reviewing.md](./running_and_reviewing.md) to submit and review one, and
[output_schemas.md](./output_schemas.md) for the tables the run writes to (the whole config lands
verbatim in `run_registry.raw_config`).
