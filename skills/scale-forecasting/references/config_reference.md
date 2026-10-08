# `RunConfig` Schema & Validation Reference

> **Auto-generated from [`src/scale_forecasting/config.py`](../../../src/scale_forecasting/config.py) by `python -m scale_forecasting.agent_surfaces --write`.** Do not edit by hand; pre-commit (`test_agent_surfaces.py`) enforces zero drift.

- **JSON Schema URI:** `https://statmike.github.io/scale-forecasting/schemas/run_config.schema.json`
- **Local Schema Path:** [`docs/schemas/run_config.schema.json`](../../../docs/schemas/run_config.schema.json)
- **Strictness:** Every config block sets `ConfigDict(frozen=True, extra='forbid')`. Unknown keys fail immediately at load time (`ConfigError`).
- **Optional `$schema` Key:** Any config JSON file may include `"$schema": "https://statmike.github.io/scale-forecasting/schemas/run_config.schema.json"` at the top level. `RunConfig` validates and strips `$schema` before model construction so `model_dump()` and `run_id` hashes are completely unaffected.

---

## 1. All Config Blocks & Fields

### `RunConfig (top-level)`

Root run configuration object.

| Field | Type / Allowed Values | Default | Constraints |
| :--- | :--- | :--- | :--- |
| `run_name` | `str` | **required** | — |
| `data` | `DataConfig` | **required** | — |
| `python_runtime` | `'spark' \| 'ray' \| 'vertex' \| 'gce' \| 'gke' \| 'vertex_automl'` | `"spark"` | — |
| `models` | `list[str]` | **required** | `min_len= 1` |
| `features` | `FeaturesConfig` | `FeaturesConfig()` | — |
| `backtest` | `BacktestConfig` | `BacktestConfig()` | — |
| `output` | `OutputConfig` | `OutputConfig()` | — |
| `hpo` | `HpoConfig` | `HpoConfig()` | — |
| `ensemble` | `EnsembleConfig` | `EnsembleConfig()` | — |
| `hierarchy` | `HierarchyConfig` | `HierarchyConfig()` | — |
| `compute` | `ComputeConfig` | `ComputeConfig()` | — |
| `model_params` | `dict[str, dict[str, bool \| int \| float \| str \| None \| list[bool \| int \| float \| str \| None]]]` | `{}` | — |

### `data (DataConfig)`

Source BigQuery table, column names, frequency, and horizon.

| Field | Type / Allowed Values | Default | Constraints |
| :--- | :--- | :--- | :--- |
| `source_table` | `str` | **required** | — |
| `ts_id_col` | `str` | `"ts_id"` | — |
| `date_col` | `str` | `"ds"` | — |
| `target_col` | `str` | `"y"` | — |
| `freq` | `str` | `"D"` | — |
| `horizon` | `int` | `28` | `> 0` |
| `series_limit` | `int \| None` | `null` | `> 0` |

### `features (FeaturesConfig)`

Target transforms, country holidays, covariates (future/past/static), lags, and Fourier terms.

| Field | Type / Allowed Values | Default | Constraints |
| :--- | :--- | :--- | :--- |
| `holidays` | `list[str]` | `[]` | — |
| `transform` | `'none' \| 'log1p' \| 'boxcox'` | `"none"` | — |
| `exog` | `list[str]` | `[]` | — |
| `static_covariates` | `list[str]` | `[]` | — |
| `future_covariates` | `list[str]` | `[]` | — |
| `past_covariates` | `list[str]` | `[]` | — |
| `exog_lags` | `dict[str, list[int]]` | `{}` | — |
| `on_unsupported_covariates` | `'fallback' \| 'error'` | `"fallback"` | — |
| `fourier` | `bool` | `false` | — |
| `level_shift` | `bool` | `false` | — |

### `backtest (BacktestConfig)`

Expanding, sliding, frozen, or stale cross-validation geometry and decision metric.

| Field | Type / Allowed Values | Default | Constraints |
| :--- | :--- | :--- | :--- |
| `enabled` | `bool` | `false` | — |
| `scheme` | `'expanding' \| 'sliding' \| 'expanding_frozen' \| 'expanding_stale'` | `"expanding"` | — |
| `n_folds` | `int` | `3` | `>= 1` |
| `horizon` | `int` | `28` | `> 0` |
| `step` | `int` | `28` | `> 0` |
| `min_train` | `int` | `180` | `> 0` |
| `decision_metric` | `str` | `"wape"` | — |
| `short_series` | `'adapt' \| 'overlap' \| 'shrink_train' \| 'skip' \| 'error'` | `"adapt"` | — |
| `min_folds` | `int` | `1` | `>= 1` |
| `min_train_floor` | `int \| None` | `null` | `> 0` |
| `gap` | `int` | `0` | `>= 0` |
| `window` | `int \| None` | `null` | `> 0` |
| `control_arm` | `bool` | `false` | — |

### `output (OutputConfig)`

Point-forecast arm selection (`raw`, `median`, `mean`, or per-cell `auto`).

| Field | Type / Allowed Values | Default | Constraints |
| :--- | :--- | :--- | :--- |
| `point_forecast` | `'raw' \| 'median' \| 'mean' \| 'auto' \| None` | `null` | — |

### `hpo (HpoConfig)`

Optuna hyperparameter optimization (`fleetwide` or `per_series`).

| Field | Type / Allowed Values | Default | Constraints |
| :--- | :--- | :--- | :--- |
| `enabled` | `bool` | `false` | — |
| `engine` | `'optuna'` | `"optuna"` | — |
| `n_trials` | `int` | `20` | `> 0` |
| `granularity` | `'fleetwide' \| 'per_series'` | `"fleetwide"` | — |
| `sample_size` | `int` | `20` | `> 0` |

### `ensemble (EnsembleConfig)`

Calculated (`mean`, `median`, `inverse_error`) and learned (`nnls`, `ridge`, `xgb`) stackers.

| Field | Type / Allowed Values | Default | Constraints |
| :--- | :--- | :--- | :--- |
| `enabled` | `bool` | `false` | — |
| `strategies` | `list['mean' \| 'median' \| 'inverse_error' \| 'nnls' \| 'ridge' \| 'xgb']` | `["median"]` | — |
| `prune_threshold` | `float` | `0.0` | `>= 0.0` |

### `hierarchy (HierarchyConfig)`

Cross-sectional aggregation levels and FPP3 reconciliation (`bottom_up`, `mint_shrink`, ...).

| Field | Type / Allowed Values | Default | Constraints |
| :--- | :--- | :--- | :--- |
| `enabled` | `bool` | `false` | — |
| `levels` | `list[list[str]]` | `[]` | — |
| `reconciliation_methods` | `list['bottom_up' \| 'top_down' \| 'middle_out' \| 'ols' \| 'wls_struct' \| 'wls_var' \| 'mint_shrink']` | `["bottom_up", "wls_struct", "mint_shrink"]` | — |
| `middle_level` | `list[str] \| None` | `null` | — |

### `compute (ComputeConfig)`

Default runtime, GPU/VM shape, autoscaling bounds, profiling, capacity, and family overrides.

| Field | Type / Allowed Values | Default | Constraints |
| :--- | :--- | :--- | :--- |
| `max_parallelism` | `int` | `1000` | `> 0` |
| `bucket_target_cells` | `int` | `8` | `> 0` |
| `max_executors` | `int \| None` | `null` | `> 0` |
| `machine_family` | `'auto' \| 'n1' \| 'n2' \| 'n2d' \| 'e2' \| 'c2'` | `"auto"` | — |
| `spark_deps` | `'packed_venv' \| 'container'` | `"packed_venv"` | — |
| `persist_models` | `bool` | `false` | — |
| `use_gpu` | `bool` | `false` | — |
| `gpu_type` | `'T4' \| 'L4' \| 'A100' \| 'A100_80GB'` | `"T4"` | — |
| `gpu_fraction` | `'auto' \| float` | `"auto"` | — |
| `budget_usd` | `float` | `50.0` | `>= 0.0` |
| `machine_type` | `str` | `"auto"` | — |
| `workers` | `int` | `1` | `> 0` |
| `min_workers` | `int \| None` | `null` | `> 0` |
| `max_workers` | `int \| None` | `null` | `> 0` |
| `ray_cluster_name` | `str \| None` | `null` | — |
| `ray_mode` | `'vertex' \| 'gke'` | `"vertex"` | — |
| `gke_mode` | `'job' \| 'ray'` | `"job"` | — |
| `gke_cluster_name` | `str \| None` | `null` | — |
| `gke_namespace` | `str` | `"default"` | — |
| `automl_mode` | `'tabular_workflow' \| 'training_job'` | `"tabular_workflow"` | — |
| `ray_regions` | `list[str] \| None` | `null` | — |
| `ray_head_machine_type` | `str` | `"n1-standard-16"` | — |
| `ray_cpu_machine_type` | `str` | `"n1-standard-8"` | — |
| `ray_gpu_machine_type` | `str` | `"n1-standard-8"` | — |
| `accelerator_count` | `int` | `1` | `> 0` |
| `ray_target_cells_per_slot` | `int` | `8` | `> 0` |
| `ray_max_nodes` | `int` | `16` | `> 0` |
| `ray_autoscale` | `bool` | `true` | — |
| `ray_cpu_min_nodes` | `int` | `1` | `> 0` |
| `ray_gpu_min_nodes` | `int` | `1` | `> 0` |
| `ray_cpu_max_nodes` | `int \| None` | `null` | `> 0` |
| `ray_gpu_max_nodes` | `int \| None` | `null` | `> 0` |
| `gpu_calibration_samples` | `int` | `3` | `> 0` |
| `gpu_safety_margin` | `float` | `1.3` | `> 1.0` |
| `profile` | `ProfileConfig` | `ProfileConfig()` | — |
| `capacity` | `CapacityConfig` | `CapacityConfig()` | — |
| `ray_read_mode` | `'driver_collect' \| 'ray_data'` | `"driver_collect"` | — |
| `read_max_streams` | `int` | `0` | `>= 0` |
| `families` | `dict['statistical' \| 'ml' \| 'deep_learning' \| 'automl', FamilyCompute]` | `{}` | — |
| `ensemble` | `EnsembleCompute` | `EnsembleCompute()` | — |

### `compute.families.<family> (FamilyCompute)`

Per-family runtime and hardware override (`statistical`, `ml`, `deep_learning`, `automl`).

| Field | Type / Allowed Values | Default | Constraints |
| :--- | :--- | :--- | :--- |
| `runtime` | `'spark' \| 'ray' \| 'vertex' \| 'gce' \| 'gke' \| 'vertex_automl' \| None` | `null` | — |
| `spark_mode` | `'serverless' \| 'cluster' \| None` | `null` | — |
| `spark_cluster_name` | `str \| None` | `null` | — |
| `gke_mode` | `'job' \| 'ray' \| None` | `null` | — |
| `gke_cluster_name` | `str \| None` | `null` | — |
| `ray_mode` | `'vertex' \| 'gke' \| None` | `null` | — |
| `automl_mode` | `'tabular_workflow' \| 'training_job' \| None` | `null` | — |
| `hardware` | `'cpu' \| 'gpu' \| None` | `null` | — |
| `gpu_type` | `'T4' \| 'L4' \| 'A100' \| 'A100_80GB' \| None` | `null` | — |
| `accelerator_count` | `int \| None` | `null` | `> 0` |
| `machine_type` | `str \| None` | `null` | — |
| `workers` | `int \| None` | `null` | `> 0` |
| `min_workers` | `int \| None` | `null` | `> 0` |
| `max_workers` | `int \| None` | `null` | `> 0` |

### `compute.ensemble (EnsembleCompute)`

When the ensemble DAG node executes (`barrier` vs. `microbatch`).

| Field | Type / Allowed Values | Default | Constraints |
| :--- | :--- | :--- | :--- |
| `mode` | `'barrier' \| 'microbatch'` | `"barrier"` | — |
| `microbatch_interval_s` | `float` | `60.0` | `> 0` |

### `compute.profile (ProfileConfig)`

Empirical compute profiling (`mode`, `measure`, `source`, safety margins).

| Field | Type / Allowed Values | Default | Constraints |
| :--- | :--- | :--- | :--- |
| `mode` | `'off' \| 'auto' \| 'always'` | `"auto"` | — |
| `samples` | `int` | `8` | `> 0` |
| `min_cells` | `int` | `1000` | `> 0` |
| `memory_margin` | `float` | `1.3` | `> 1.0` |
| `time_margin` | `float` | `1.2` | `> 1.0` |
| `measure` | `'off' \| 'harvest' \| 'controlled'` | `"harvest"` | — |
| `source` | `str` | `"auto"` | — |

### `compute.capacity (CapacityConfig)`

Quota preflight and per-service regional capacity retry policies (excluded from `run_id`).

| Field | Type / Allowed Values | Default | Constraints |
| :--- | :--- | :--- | :--- |
| `enabled` | `bool` | `true` | — |
| `preflight` | `bool` | `true` | — |
| `ray` | `CapacityServicePolicy` | `CapacityServicePolicy()` | — |
| `vertex` | `CapacityServicePolicy` | `CapacityServicePolicy()` | — |
| `vertex_automl` | `CapacityServicePolicy` | `CapacityServicePolicy()` | — |
| `gce` | `CapacityServicePolicy` | `CapacityServicePolicy()` | — |
| `gke` | `CapacityServicePolicy` | `CapacityServicePolicy()` | — |
| `dataproc_cluster` | `CapacityServicePolicy` | `CapacityServicePolicy()` | — |
| `dataproc_serverless` | `CapacityServicePolicy` | `CapacityServicePolicy()` | — |
| `retry` | `RetryResources` | `RetryResources()` | — |

### `compute.capacity.<service> (CapacityServicePolicy)`

Per-service attempt, wall-clock, pass, and exponential backoff bounds.

| Field | Type / Allowed Values | Default | Constraints |
| :--- | :--- | :--- | :--- |
| `max_attempts` | `int \| None` | `null` | `>= 0` |
| `max_wall_seconds` | `float \| None` | `null` | `>= 0` |
| `max_passes` | `int \| None` | `null` | `>= 0` |
| `backoff_seconds` | `float \| None` | `null` | `>= 0` |
| `backoff_multiplier` | `float \| None` | `null` | `>= 1.0` |
| `backoff_max_seconds` | `float \| None` | `null` | `>= 0` |

### `compute.capacity.retry (RetryResources)`

Resource ceiling (`max_executors`) for surgical `--retry` repairs.

| Field | Type / Allowed Values | Default | Constraints |
| :--- | :--- | :--- | :--- |
| `max_executors` | `int \| None` | `null` | `> 0` |

---

## 2. Dynamic Vocabularies & Special Keys

| Field | Allowed Values / Rule |
| :--- | :--- |
| `models` | Non-empty list of registered model names (`34` available; see [`catalog_reference.md`](./catalog_reference.md)). No duplicates allowed. |
| `backtest.decision_metric` | Any of the `21` registered metrics: `mae`, `rmse`, `mse`, `mape`, `smape`, `wape`, `mase`, `rmsse`, `bias`, `coverage`, `pinball`, `mase_seasonal`, `maape`, `interval_score`, `interval_width`, `ope`, `rmsle`, `msse`, `msis`, `r2`, `cv`. |
| `ensemble.strategies` | Calculated: `mean`, `median`, `inverse_error` (work with or without backtest). Learned stackers: `nnls`, `ridge`, `xgb` (require `backtest.enabled=true`). Singular shorthand `strategy` is also accepted. |
| `hierarchy.reconciliation_methods` | Any subset of `bottom_up`, `middle_out`, `mint_shrink`, `ols`, `top_down`, `wls_struct`, `wls_var`. |
| `compute.families` | Keys must be in `statistical`, `ml`, `deep_learning`, `automl`. `native` models always run in BigQuery and never take a `compute.families` entry. |
| `compute.profile.source` | One of `none`, `auto`, `baseline` or an existing `<slug>-<12hex>` `run_id`. |
| `model_params` | Dict mapping `<model_name>` to `<param_dict>` of JSON-safe scalars or flat lists (`bool`, `int`, `float`, `str`, `None`). Non-finite floats (`NaN`, `Infinity`) are rejected. |

---

## 3. Cross-Field Validation Rules (`RunConfig._normalize`)

1. **HPO requires Backtesting:** `hpo.enabled = true` requires `backtest.enabled = true` (raises `ConfigError` otherwise).
2. **Learned Ensembles require Backtesting:** If `ensemble.enabled = true` and `backtest.enabled = false`, learned strategies (`nnls`, `ridge`, `xgb`) are automatically dropped with a warning; always set `backtest.enabled = true` when using learned stackers.
3. **Point Forecast Arm (`output.point_forecast`):**
   - Defaults to `"auto"` when `backtest.enabled = true` and `"median"` when `backtest.enabled = false`.
   - Setting `"mean"` or `"auto"` when `backtest.enabled = false` raises `ConfigError`.
4. **Short-Series Backtest Policies (`backtest.short_series`):**
   - `min_folds` cannot exceed `n_folds`.
   - `short_series = "shrink_train"` requires `min_train_floor` to be set (and `min_train_floor` is forbidden on other `short_series` policies).
   - `control_arm = true` is forbidden when `scheme = "expanding_stale"`.
5. **Covariate Hygiene (`features`):**
   - `future_covariates` and `past_covariates` must be disjoint.
   - `static_covariates` must be disjoint from all dynamic covariates (`exog`, `future_covariates`, `past_covariates`).
   - Every key in `exog_lags` must name a declared dynamic covariate, and lags must be positive integers.
6. **Hierarchy Hygiene (`hierarchy`):**
   - When `hierarchy.enabled = true`, both `levels` and `reconciliation_methods` must be non-empty, and `middle_level` (if set) must match one of the entries in `levels`.
7. **Per-Family Runtime & Hardware Constraints (`compute.families`):**
   - Only `deep_learning` and `automl` families may request `hardware = "gpu"` or `gpu_type`.
   - Family `automl` must use `runtime = "vertex_automl"`, and no other family may use `vertex_automl` or `automl_mode`.
   - `spark_mode` / `spark_cluster_name` are only valid when `runtime = "spark"`; Dataproc Serverless (`spark_mode = "serverless"`) supports `L4` GPUs only and forbids `machine_type`.
   - `gke_mode` is only valid when `runtime = "gke"`; `ray_mode` is only valid when `runtime = "ray"`.
   - `runtime = "gce"` is strictly single-VM (`workers = 1`).
   - `min_workers` / `max_workers` are only valid on autoscaling runtimes (`ray`, `spark`, `vertex_automl`, or `gke` with `gke_mode = "ray"`).

---

## 4. GPU-to-VM Machine Type Compatibility (`resolve_vm_machine_type`)

| `gpu_type` | Vertex Enum | VRAM | Allowed `accelerator_count` | Auto-Resolved `machine_type` | Valid Explicit `machine_type` Shapes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `T4` | `NVIDIA_TESLA_T4` | 16 GiB | `1`, `2`, `4` | `n1-standard-8 (1-2 GPUs) | n1-standard-16 (4 GPUs)` | Any n1-* shape (1-2 GPUs max 48 vCPUs; 4 GPUs up to 96 vCPUs) |
| `L4` | `NVIDIA_L4` | 24 GiB | `1`, `2`, `4`, `8` | `g2-standard-8 (1 GPU) | g2-standard-{24,48,96} (2/4/8 GPUs)` | 1 GPU: g2-standard-{4,8,12,16,32}; 2 GPUs: g2-standard-24; 4 GPUs: g2-standard-48; 8 GPUs: g2-standard-96 |
| `A100` | `NVIDIA_TESLA_A100` | 40 GiB | `1`, `2`, `4`, `8`, `16` | `a2-highgpu-{1,2,4,8}g | a2-megagpu-16g` | Strict 1:1 mapping with accelerator_count |
| `A100_80GB` | `NVIDIA_A100_80GB` | 80 GiB | `1`, `2`, `4`, `8` | `a2-ultragpu-{1,2,4,8}g` | Strict 1:1 mapping with accelerator_count |

---

## 5. Content-Addressed `run_id` Digest Rules (`registry/ids.py`)

- `run_id` is `<slug>-<12hex>` computed from the normalized `RunConfig` JSON dump.
- **Operational fields excluded from `run_id` (changing these keeps the same `run_id`):**
  - Infrastructure environment (`SF_PROJECT_ID`, `SF_DATASET_ID`, `SF_REGISTRY_DATASET_ID`, `SF_REGION`, `SF_WAREHOUSE_URI`, `SF_CONNECTION`).
  - `compute.capacity` (all regional retry policies and `compute.capacity.retry.max_executors`).
  - `compute.profile.source` (resolved profile pointer).
  - Default/unset `machine_type` (`"auto"`) and default `workers` (`1`) when left at their baseline defaults.
  - Top-level `"$schema"` URI key.
