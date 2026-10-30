"""Single-source generator and environment probe for AI agent, skill, schema, and LLM surfaces.

Generates and verifies the repository's machine-readable and agent-facing artifacts directly from
the canonical Python declarations (`RunConfig`, `list_models`, `METRIC_NAMES`, `VIEW_NAMES`,
`REGISTRY_TABLE_NAMES`, `SOURCE_TABLE_NAMES`, `EXTRA_MODULES`, and `configs/`):

1. ``docs/schemas/run_config.schema.json`` — JSON Schema (Draft 2020-12) for `RunConfig`.
2. ``skills/scale-forecasting/references/config_reference.md`` — complete field-by-field reference
   for `RunConfig` and every nested Pydantic model, cross-field rules, and ``run_id`` exclusions.
3. ``skills/scale-forecasting/references/catalog_reference.md`` — all 37 models, 21 metrics,
   7 compute runtimes, 4 GPU types, 13 dependency extras, BigQuery tables/views, and 62 configs.
4. ``skills/scale-forecasting/references/execution_paths_and_ops.md`` — all 5 execution paths,
   environment variables, monitoring/review/calibration, surgical repair, and registry operations.
5. ``docs/llms.txt`` — concise LLM index following the ``llms.txt`` convention.
6. ``docs/llms-full.txt`` — single-file comprehensive context bundle for LLM and RAG ingestion.

Also provides `probe_environment`, which inspects the active Python environment (installed extras,
locally available vs. missing-extra models, ``SF_*`` environment variables, `.env.infra`, and ADC
credentials) without making network calls or importing cloud SDKs.

CLI usage::

    python -m scale_forecasting.agent_surfaces --write
    python -m scale_forecasting.agent_surfaces --check
    python -m scale_forecasting.agent_surfaces --probe-env
    python -m scale_forecasting.agent_surfaces --bump-version X.Y.Z
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from .auto_policy import EXPOSURE_TIERS, SCALE_REGIMES
from .benchmarks import BENCHMARK_SPECS, PUBLIC_BENCHMARK_SPECS, list_benchmarks
from .config import (
    COMPUTE_FAMILIES,
    LEARNED_STRATEGIES,
    PROFILE_SOURCE_KEYWORDS,
    RECONCILIATION_METHODS,
    RUN_CONFIG_SCHEMA_URI,
    BacktestConfig,
    CapacityConfig,
    CapacityServicePolicy,
    ComputeConfig,
    DataConfig,
    EnsembleCompute,
    EnsembleConfig,
    FamilyCompute,
    FeaturesConfig,
    HierarchyConfig,
    HpoConfig,
    OutputConfig,
    ProfileConfig,
    RetryResources,
    RunConfig,
    Strategy,
)
from .errors import EXTRA_MODULES, configure_cli_logging, is_importable
from .metrics import METRIC_NAMES, get_metric
from .models import get_model, list_models
from .registry.ddl import REGISTRY_TABLE_NAMES, SOURCE_TABLE_NAMES
from .registry.views import CORE_VIEW_NAMES, DIAGNOSTIC_VIEW_NAMES, VIEW_NAMES

__all__ = [
    "ALL_EXTRA_MODULES",
    "EXTRA_DESCRIPTIONS",
    "GPU_CATALOG",
    "RUNTIME_CATALOG",
    "build_run_config_json_schema",
    "bump_version",
    "check_agent_surfaces",
    "expected_agent_surface_files",
    "main",
    "probe_environment",
    "render_llms_full_txt",
    "render_llms_txt",
    "render_release_notes",
    "render_run_config_json_schema",
    "render_skill_catalog_reference",
    "render_skill_config_reference",
    "render_skill_execution_paths_reference",
    "render_visual_architecture_inventory",
    "sync_visual_architecture_text",
    "write_agent_surfaces",
]

_SITE_BASE = "https://statmike.github.io/scale-forecasting"
_REPO_URL = "https://github.com/statmike/scale-forecasting"

_MODELS_STATS_MODULES: tuple[str, ...] = ("statsforecast",)
_MODELS_TREES_MODULES: tuple[str, ...] = ("xgboost", "lightgbm", "catboost")
_MODELS_PROPHET_MODULES: tuple[str, ...] = ("prophet",)
_MODELS_DL_MODULES: tuple[str, ...] = ("torch", "neuralprophet", "neuralforecast")
_MODELS_FOUNDATION_MODULES: tuple[str, ...] = ("torch", "chronos", "timesfm")
_MODELS_AUTOML_MODULES: tuple[str, ...] = EXTRA_MODULES["models-automl"]

ALL_EXTRA_MODULES: dict[str, tuple[str, ...]] = {
    "gcp": EXTRA_MODULES["gcp"],
    "notebook": EXTRA_MODULES["notebook"],
    "spark": EXTRA_MODULES["spark"],
    "ray": EXTRA_MODULES["ray"],
    "submit": EXTRA_MODULES["submit"],
    "models-stats": _MODELS_STATS_MODULES,
    "models-trees": _MODELS_TREES_MODULES,
    "models-prophet": _MODELS_PROPHET_MODULES,
    "models-dl": _MODELS_DL_MODULES,
    "models-foundation": _MODELS_FOUNDATION_MODULES,
    "models-automl": _MODELS_AUTOML_MODULES,
    "models": (
        *_MODELS_STATS_MODULES,
        *_MODELS_TREES_MODULES,
        *_MODELS_PROPHET_MODULES,
        *_MODELS_DL_MODULES,
        *_MODELS_FOUNDATION_MODULES,
        *_MODELS_AUTOML_MODULES,
    ),
    "all": tuple(
        dict.fromkeys(
            [
                *EXTRA_MODULES["gcp"],
                *EXTRA_MODULES["notebook"],
                *EXTRA_MODULES["spark"],
                *EXTRA_MODULES["ray"],
                *_MODELS_STATS_MODULES,
                *_MODELS_TREES_MODULES,
                *_MODELS_PROPHET_MODULES,
                *_MODELS_DL_MODULES,
                *_MODELS_FOUNDATION_MODULES,
                *_MODELS_AUTOML_MODULES,
            ]
        )
    ),
}

EXTRA_DESCRIPTIONS: dict[str, str] = {
    "gcp": "Google Cloud clients (BigQuery, Storage Read/Write APIs, GCS, Dataproc, Vertex AI)",
    "notebook": "Interactive notebook kernel and matplotlib plotting helpers",
    "spark": "PySpark runtime client plus [gcp]",
    "ray": "Ray cluster/job submission client plus [gcp]",
    "submit": "Alias for [ray] (thin launch-host client for all cloud runtimes)",
    "models-stats": "Nixtla statsforecast and pmdarima statistical models (5 models)",
    "models-trees": "Gradient-boosted tree models: XGBoost, LightGBM, CatBoost (3 models)",
    "models-prophet": "Prophet additive/multiplicative decomposable model (1 model)",
    "models-dl": "PyTorch, NeuralProphet, and NeuralForecast deep-learning models (5 models)",
    "models-foundation": (
        "Open-weight zero-shot foundation models: Chronos-Bolt, Chronos-2, TimesFM (3 models)"
    ),
    "models-automl": "Vertex AI Pipelines / KFP components for Tabular Workflows (4 models)",
    "models": "All 6 model family extras combined",
    "all": "Complete installation (all cloud clients, runtimes, notebooks, and model families)",
}

RUNTIME_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "runtime": "spark",
        "modes": "spark_mode: 'serverless' (default) | 'cluster' (plus interactive Spark Connect)",
        "families": ("statistical", "ml", "deep_learning"),
        "gpu_types": ("L4 (serverless/cluster)", "T4, A100, A100_80GB (cluster only)"),
        "scaling": "Dynamic executor allocation (max_executors / min_workers / max_workers)",
        "submitter": "scale_forecasting.submit",
        "summary": "Dataproc Serverless batches or ephemeral/existing Dataproc GCE clusters.",
    },
    {
        "runtime": "ray",
        "modes": "ray_mode: 'vertex' (default) | 'gke'",
        "families": ("statistical", "ml", "deep_learning"),
        "gpu_types": ("T4", "L4", "A100", "A100_80GB"),
        "scaling": "Autoscaling CPU + GPU worker pools (ray_autoscale, min_workers, max_workers)",
        "submitter": "scale_forecasting.ray_submit",
        "summary": "Vertex AI Managed Ray or KubeRay on GKE with fractional GPU task packing.",
    },
    {
        "runtime": "vertex",
        "modes": "Managed Vertex AI CustomJob worker pool",
        "families": ("statistical", "ml", "deep_learning"),
        "gpu_types": ("T4", "L4", "A100", "A100_80GB"),
        "scaling": "Fixed multi-VM worker pool (workers >= 1, auto-expands for multi-model DL)",
        "submitter": "scale_forecasting.vertex_submit",
        "summary": "Serverless multi-worker Vertex AI CustomJob with BQ Storage Read sharding.",
    },
    {
        "runtime": "gce",
        "modes": "Single-VM Container-Optimized OS instance",
        "families": ("statistical", "ml", "deep_learning"),
        "gpu_types": ("T4", "L4", "A100", "A100_80GB"),
        "scaling": "Strictly single-VM (workers = 1) with triple-redundant self-deletion",
        "submitter": "scale_forecasting.gce_submit",
        "summary": "Lowest-overhead single-VM execution with automatic guest + host teardown.",
    },
    {
        "runtime": "gke",
        "modes": "gke_mode: 'job' (default, K8s Indexed Job) | 'ray' (KubeRay)",
        "families": ("statistical", "ml", "deep_learning"),
        "gpu_types": ("T4", "L4", "A100", "A100_80GB"),
        "scaling": "Multi-pod Indexed Job (workers >= 1) or autoscaling Ray pods on GKE",
        "submitter": "scale_forecasting.gke_submit",
        "summary": "Runs on an existing or ephemeral Google Kubernetes Engine Standard cluster.",
    },
    {
        "runtime": "vertex_automl",
        "modes": "automl_mode: 'tabular_workflow' (default) | 'training_job'",
        "families": ("automl",),
        "gpu_types": ("T4", "L4", "A100", "A100_80GB"),
        "scaling": "Managed Vertex AI Pipelines / AutoML worker pools (min_workers, max_workers)",
        "submitter": "scale_forecasting.automl_submit",
        "summary": "Managed global cross-series AutoML (TiDE, TFT, Seq2Seq+, L2L) in Vertex AI.",
    },
    {
        "runtime": "bigquery",
        "modes": "In-warehouse BigQuery ML SQL",
        "families": ("native",),
        "gpu_types": (),
        "scaling": "Managed BigQuery slots (no VM provisioning)",
        "submitter": "scale_forecasting.engines.bigquery_engine",
        "summary": "Zero-copy in-warehouse SQL execution for arima_plus and timesfm.",
    },
)

GPU_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "gpu_type": "T4",
        "vertex_enum": "NVIDIA_TESLA_T4",
        "vram_gib": 16,
        "allowed_counts": (1, 2, 4),
        "auto_machine_type": "n1-standard-8 (1-2 GPUs) | n1-standard-16 (4 GPUs)",
        "compatible_machines": "Any n1-* shape (1-2 GPUs max 48 vCPUs; 4 GPUs up to 96 vCPUs)",
    },
    {
        "gpu_type": "L4",
        "vertex_enum": "NVIDIA_L4",
        "vram_gib": 24,
        "allowed_counts": (1, 2, 4, 8),
        "auto_machine_type": "g2-standard-8 (1 GPU) | g2-standard-{24,48,96} (2/4/8 GPUs)",
        "compatible_machines": (
            "1 GPU: g2-standard-{4,8,12,16,32}; 2 GPUs: g2-standard-24; "
            "4 GPUs: g2-standard-48; 8 GPUs: g2-standard-96"
        ),
    },
    {
        "gpu_type": "A100",
        "vertex_enum": "NVIDIA_TESLA_A100",
        "vram_gib": 40,
        "allowed_counts": (1, 2, 4, 8, 16),
        "auto_machine_type": "a2-highgpu-{1,2,4,8}g | a2-megagpu-16g",
        "compatible_machines": "Strict 1:1 mapping with accelerator_count",
    },
    {
        "gpu_type": "A100_80GB",
        "vertex_enum": "NVIDIA_A100_80GB",
        "vram_gib": 80,
        "allowed_counts": (1, 2, 4, 8),
        "auto_machine_type": "a2-ultragpu-{1,2,4,8}g",
        "compatible_machines": "Strict 1:1 mapping with accelerator_count",
    },
)

_ENV_VARS_DOC: tuple[tuple[str, bool, str, str], ...] = (
    (
        "SF_PROJECT_ID",
        True,
        "—",
        "Google Cloud project ID hosting BigQuery and compute runtimes.",
    ),
    (
        "SF_CONNECTION",
        True,
        "—",
        "BigLake connection reference (`project.region.connection_name`).",
    ),
    (
        "SF_WAREHOUSE_URI",
        True,
        "—",
        "Cloud Storage warehouse root (`gs://<bucket>/warehouse`).",
    ),
    (
        "SF_DATASET_ID",
        False,
        "scale_forecasting",
        "BigQuery dataset containing source series tables (and registry unless overridden).",
    ),
    (
        "SF_REGISTRY_DATASET_ID",
        False,
        "<SF_DATASET_ID>",
        "Optional separate BigQuery dataset for the 5 registry tables and 11 views.",
    ),
    (
        "SF_REGION",
        False,
        "us-central1",
        "Default Google Cloud region for BigQuery, Dataproc, Vertex AI, and GKE.",
    ),
    (
        "SF_SERVICE_ACCOUNT",
        False,
        "ADC / default SA",
        "Least-privilege runner service account email attached to cloud worker jobs.",
    ),
    (
        "SF_CONTAINER_IMAGE",
        False,
        "Derived from project/region",
        "Artifact Registry container URI for Ray, Vertex CustomJob, GCE, and GKE.",
    ),
    (
        "SF_GKE_CLUSTER",
        False,
        "Ephemeral per-run",
        "Existing GKE cluster name to reuse when `runtime='gke'` or `ray_mode='gke'`.",
    ),
    (
        "SF_SUBNET",
        False,
        "default",
        "VPC subnet name or self-link for private-IP Dataproc, Ray, Vertex, and GCE jobs.",
    ),
    (
        "SF_LOG_LEVEL",
        False,
        "INFO",
        "Root log level for CLI entry points (`DEBUG`, `INFO`, `WARNING`, `ERROR`).",
    ),
)

_CONFIG_MODELS: tuple[tuple[str, str, type[BaseModel]], ...] = (
    ("RunConfig (top-level)", "Root run configuration object.", RunConfig),
    (
        "data (DataConfig)",
        "Source BigQuery table, column names, frequency, and horizon.",
        DataConfig,
    ),
    (
        "features (FeaturesConfig)",
        (
            "Target transforms (including compositional `clr`/`ilr` simplex transforms), "
            "country holidays, covariates (future/past/static), lags, and Fourier terms."
        ),
        FeaturesConfig,
    ),
    (
        "backtest (BacktestConfig)",
        "Expanding, sliding, frozen, or stale cross-validation geometry and decision metric.",
        BacktestConfig,
    ),
    (
        "output (OutputConfig)",
        (
            "Point-forecast arm selection (`raw`, `median`, `mean`, or per-cell `auto`), physical "
            "bounds, Newsvendor underage/overage costs, and online Adaptive Conformal Inference "
            "(`calibration_method`, `aci_gamma`)."
        ),
        OutputConfig,
    ),
    (
        "hpo (HpoConfig)",
        "Optuna hyperparameter optimization (`fleetwide` or `per_series`).",
        HpoConfig,
    ),
    (
        "ensemble (EnsembleConfig)",
        (
            "Calculated (`mean`, `median`, `inverse_error`) and learned (`nnls`, `ridge`, `xgb`) "
            "stackers."
        ),
        EnsembleConfig,
    ),
    (
        "hierarchy (HierarchyConfig)",
        (
            "Cross-sectional aggregation levels, FPP3 reconciliation (`bottom_up`, "
            "`mint_shrink`, ...), temporal hierarchy reconciliation (`THieF`), and cross-temporal "
            "reconciliation (`cross_temporal`, `spatial_temporal_weight`)."
        ),
        HierarchyConfig,
    ),
    (
        "compute (ComputeConfig)",
        (
            "Default runtime, GPU/VM shape, autoscaling bounds, profiling, capacity, and family "
            "overrides."
        ),
        ComputeConfig,
    ),
    (
        "compute.families.<family> (FamilyCompute)",
        (
            "Per-family runtime and hardware override (`statistical`, `ml`, `deep_learning`, "
            "`automl`)."
        ),
        FamilyCompute,
    ),
    (
        "compute.ensemble (EnsembleCompute)",
        "When the ensemble DAG node executes (`barrier` vs. `microbatch`).",
        EnsembleCompute,
    ),
    (
        "compute.profile (ProfileConfig)",
        "Empirical compute profiling (`mode`, `measure`, `source`, safety margins).",
        ProfileConfig,
    ),
    (
        "compute.capacity (CapacityConfig)",
        (
            "Quota preflight and per-service regional capacity retry policies (excluded from "
            "`run_id`)."
        ),
        CapacityConfig,
    ),
    (
        "compute.capacity.<service> (CapacityServicePolicy)",
        "Per-service attempt, wall-clock, pass, and exponential backoff bounds.",
        CapacityServicePolicy,
    ),
    (
        "compute.capacity.retry (RetryResources)",
        "Resource ceiling (`max_executors`) for surgical `--retry` repairs.",
        RetryResources,
    ),
)

_VARIANT_SYNTAX_ROWS: tuple[tuple[str, str], ...] = (
    (
        "`<model>@local` / `<model>@global` / `<model>@hybrid`",
        "Explicitly select per-series local fitting, cross-series global pooling, or hybrid mode.",
    ),
    (
        (
            "`<model>@univariate` / `<model>@cov_none` / `<model>@cov_all` / "
            "`<model>@cov_future_only` / `<model>@cov_past_only` / `<model>@cov_static_only` / "
            "`<model>@cov_dynamic` / `<model>@cov_future_static` / `<model>@cov_past_static`"
        ),
        "Per-model covariate tier ablation override.",
    ),
    (
        "`<model>@direct` / `<model>@recursive`",
        "Select direct multi-horizon step-by-step regressors (`h=1..H`) vs. recursive rollouts.",
    ),
    (
        "`<model>@monotone`",
        (
            "Enforce causal covariate sign constraints (`features.monotonic_constraints`) on "
            "supported learners (`lightgbm`, `xgboost`, `catboost`, `regression_lags`)."
        ),
    ),
    (
        (
            "`<model>@transform_log1p` / `<model>@transform_boxcox` / "
            "`<model>@transform_clr` / `<model>@transform_ilr` / `<model>@transform_none`"
        ),
        (
            "Per-model target transform override (including compositional simplex `transform_clr` "
            "and `transform_ilr` isometric log-ratio transforms)."
        ),
    ),
    (
        (
            "`<model>@cold_start` / `<model>@analog_transfer` / "
            "`<model>@bass_diffusion` / `<model>@weibull_lifecycle` / "
            "`<model>@arps_decline` / `<model>@cold_start_arps`"
        ),
        (
            "Enable cold-start & NPI Empirical-Bayes shrinkage transfer and optionally pin the "
            "launch or depletion archetype (`analog_transfer`, `bass_diffusion`, "
            "`weibull_lifecycle`, or `arps_decline` with analytical `eur_total` / `eur_remaining`)."
        ),
    ),
    (
        "`<model>@aci` / `<model>@aci_norm` / `<model>@aci_normalized`",
        (
            "Per-model online Adaptive Conformal Inference (`aci`) or Locally Adaptive Normalized "
            "CQR (`aci_normalized`) interval calibration override."
        ),
    ),
    (
        "`<model>@similarity_global` / `<model>@similarity_cold_start`",
        (
            "Enable similarity-clustered global pooling (`@similarity_global`, injecting "
            "`__similarity_cluster_id__`) or similarity-driven cold-start donor transfer "
            "(`@similarity_cold_start`, `cold_start_matching = 'similarity'`)."
        ),
    ),
)

_NORMALIZATION_RULES: tuple[str, ...] = (
    (
        "1. **HPO requires Backtesting:** `hpo.enabled = true` requires `backtest.enabled = true` "
        "(raises `ConfigError` otherwise)."
    ),
    (
        "2. **Learned Ensembles require Backtesting:** If `ensemble.enabled = true` and "
        "`backtest.enabled = false`, learned strategies (`nnls`, `ridge`, `xgb`) are automatically "
        "dropped with a warning; always set `backtest.enabled = true` when using learned stackers."
    ),
    (
        '3. **Point Forecast Arm (`output.point_forecast`):** Defaults to `"auto"` when '
        '`backtest.enabled = true` and `"median"` when `backtest.enabled = false`. Setting '
        '`"mean"` or `"auto"` when `backtest.enabled = false` raises `ConfigError`.'
    ),
    (
        "4. **Online Adaptive Conformal Inference (`output.calibration_method`):** Supports "
        '`"conformal"`, `"aci"`, and `"aci_normalized"` with step size `output.aci_gamma` '
        "(`0 < aci_gamma <= 0.5`). Post-processing joint copula trajectories "
        "(`Forecaster.sample_trajectories` / `Forecaster.tail_risk_df`) support `copula_type` in "
        '`("auto", "gaussian", "student_t", "iman_conover", "empirical")`.'
    ),
    (
        "5. **Short-Series Backtest Policies (`backtest.short_series`):** `min_folds` cannot "
        'exceed `n_folds`. `short_series = "shrink_train"` requires `min_train_floor` to be set. '
        '`control_arm = true` is forbidden when `scheme = "expanding_stale"`.'
    ),
    (
        "6. **Covariate & Compositional Simplex Hygiene (`features`):** `future_covariates` and "
        "`past_covariates` must be disjoint; `static_covariates` must be disjoint from dynamic "
        "covariates; every key in `exog_lags` and `monotonic_constraints` must name a declared "
        "covariate; `transform` in `('clr', 'ilr')` automatically sets `simplex_sum = 1.0` when "
        "unset."
    ),
    (
        "7. **Hierarchy & Cross-Temporal Hygiene (`hierarchy`):** When `hierarchy.enabled = true`, "
        "both `levels` and `reconciliation_methods` must be non-empty. `hierarchy.cross_temporal = "
        "true` or `'cross_temporal_mint'` in `reconciliation_methods` automatically enables "
        "`reconcile_temporal = true` and validates `spatial_temporal_weight` (`'auto'` or float in "
        "`[0, 1]`)."
    ),
    (
        "8. **Per-Family Runtime & Hardware Constraints (`compute.families`):** Only "
        '`deep_learning` and `automl` families may request `hardware = "gpu"` or `gpu_type`. '
        'Family `automl` must use `runtime = "vertex_automl"`. `runtime = "gce"` is strictly '
        "single-VM (`workers = 1`)."
    ),
    (
        "9. **Time-Series Similarity, Multi-Modal Matching & Merchandising "
        "(`features`, `hierarchy`, `ensemble`):** "
        "`features.cold_start_matching = 'similarity'` uses 7-lens (`hybrid`, "
        "`physics_attractor`, `spectral_phase`, `elastic_dtw`, `minirocket`, "
        "`structural_signature`, `foundation_latent`) and multi-modal (`text_cols`, "
        "`attribute_cols`, `hierarchy_cols`) Day-0 to Day-14 donor matching "
        "(`similarity_method`). "
        "`features.similarity_cluster_feature = true` or `<model>@similarity_global` injects "
        "cluster labels into global models. `hierarchy.similarity_hierarchy = true` "
        "auto-discovers latent cluster hierarchies on flat panels, "
        "`ensemble.similarity_smoothing = true` applies Laplacian graph smoothing across "
        "per-series stacker weights, and `recommend_substitutes_df` / `detect_emerging_trends_df` "
        "provide inventory-aware SKU substitution and emerging trend breakout intelligence."
    ),
    (
        "10. **Cross-Industry Physics/Econometrics & Prescriptive Clearance/Contract MILP "
        "(`features`, `clearance_optimizer`):** "
        "`features.degree_day_cols` synthesizes Heating/Cooling Degree Days (`HDD`/`CDD`), "
        "`features.adstock_rate` applies geometric marketing adstock decay, "
        "`features.supersession_map` chains successor/predecessor SKU histories, and "
        "`<model>@arps_decline` fits hyperbolic/harmonic/exponential production decline curves. "
        "`optimize_clearance_df` and `optimize_contract_volume_df` automatically filter spatial, "
        "temporal (`THieF`), and cross-temporal aggregate nodes (`__agg__`) to prevent "
        "double-counting, partial out exogenous `covariate_cols` via Frisch–Waugh–Lovell (FWL) "
        "with Double-ML `promo_elasticity_df` priority, select per-SKU OOF champion models, "
        "and return bottom-up `hierarchy_rollup_df`. `optimize_assortment_df` solves SKU "
        "retention vs. delisting with similarity-driven demand transference and `walk_rate`; "
        "`optimize_prepacks_df` (`estimate_simplex_size_curves`) solves Aitchison CLR simplex "
        "size-curve and integer pre-pack bundle allocation; and `optimize_rebalancing_df` solves "
        "Hitchcock–Koopmans lateral store-to-store/depot inventory rebalancing."
    ),
)


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _adc_present() -> bool:
    """Check whether Application Default Credentials appear configured without network I/O."""
    env_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if env_creds and Path(env_creds).is_file():
        return True
    gcloud_adc = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    if gcloud_adc.is_file():
        return True
    return bool(
        os.environ.get("GCE_METADATA_HOST")
        or os.environ.get("K_SERVICE")
        or os.environ.get("CLOUD_RUN_JOB")
    )


def probe_environment(repo_root: Path | None = None) -> dict[str, Any]:
    """Inspect the active Python environment, installed extras, models, and GCP readiness.

    Pure local inspection — performs zero network calls and imports no cloud libraries, so it is
    safe to call in a bare ``pip install scale-forecasting`` environment.
    """
    root = repo_root or _default_repo_root()
    extras_status: dict[str, dict[str, Any]] = {}
    for extra_name, modules in ALL_EXTRA_MODULES.items():
        missing_mods = [m for m in modules if not is_importable(m)]
        extras_status[extra_name] = {
            "installed": len(missing_mods) == 0,
            "missing_modules": missing_mods,
            "install_command": f'pip install "scale-forecasting[{extra_name}]"',
        }

    available_models_list: list[str] = []
    missing_models_list: list[dict[str, str]] = []
    for name in list_models():
        cls = get_model(name)
        extra_needed = cls.optional_extra or ("gcp" if cls.runtime == "bigquery" else "core")
        if cls.is_available():
            available_models_list.append(name)
        else:
            missing_models_list.append(
                {
                    "model": name,
                    "family": cls.family,
                    "runtime": cls.runtime,
                    "package": cls.package,
                    "extra": extra_needed,
                    "install_command": f'pip install "scale-forecasting[{extra_needed}]"',
                }
            )

    env_vars: dict[str, dict[str, Any]] = {}
    for var_name, required, default_val, _desc in _ENV_VARS_DOC:
        raw = os.environ.get(var_name, "").strip()
        env_vars[var_name] = {
            "set": bool(raw),
            "required_for_cloud": required,
            "default": default_val,
            "value": raw if raw else None,
        }

    required_cloud_vars = ("SF_PROJECT_ID", "SF_CONNECTION", "SF_WAREHOUSE_URI")
    missing_cloud_vars = [k for k in required_cloud_vars if not env_vars[k]["set"]]
    settings_configured = len(missing_cloud_vars) == 0
    dotenv_present = (Path.cwd() / ".env.infra").is_file() or (root / ".env.infra").is_file()
    adc_ok = _adc_present()
    gcp_installed = bool(extras_status["gcp"]["installed"])

    cloud_ready = gcp_installed and settings_configured
    recommendations: list[str] = []
    if not gcp_installed:
        recommendations.append(
            "Install Google Cloud clients for BigQuery/Vertex/Dataproc: "
            'pip install "scale-forecasting[gcp]"'
        )
    if missing_cloud_vars:
        if dotenv_present:
            recommendations.append(
                f"Load infrastructure environment variables "
                f"({', '.join(missing_cloud_vars)} missing): "
                "set -a && source .env.infra && set +a"
            )
        else:
            recommendations.append(
                f"Export required cloud environment variables: {', '.join(missing_cloud_vars)}"
            )
    if missing_models_list:
        recommendations.append(
            f"{len(missing_models_list)} optional model(s) not installed locally; install all "
            'with: pip install "scale-forecasting[models]" (or pass --ignore-unavailable-models)'
        )

    return {
        "python_version": sys.version.split()[0],
        "package_version": _pkg_version("scale-forecasting"),
        "extras": extras_status,
        "models": {
            "total": len(list_models()),
            "available_count": len(available_models_list),
            "available": available_models_list,
            "monotonic_models": list_models(supports_monotonic_constraints=True),
            "missing": missing_models_list,
        },
        "settings_env": {
            "configured": settings_configured,
            "missing_required": missing_cloud_vars,
            "dotenv_infra_present": dotenv_present,
            "variables": env_vars,
        },
        "adc_configured": adc_ok,
        "execution_paths_ready": {
            "path_1_offline_playground_and_dry_run": True,
            "path_2_cloud_plan_feasibility_and_stage": cloud_ready,
            "path_3_cloud_multi_family_dag_run": cloud_ready,
            "path_4_direct_family_submitters": cloud_ready,
            "path_5_airflow_dag_emission": True,
        },
        "auto_forecaster": {
            "exposure_tiers": list(EXPOSURE_TIERS.keys()),
            "scale_regimes": list(SCALE_REGIMES.keys()),
            "where_options": ["auto", "local", "bigquery", "cloud_cpu", "cloud_all"],
            "optimization_goals": ["accuracy", "probabilistic", "service_level", "fast"],
            "benchmarks_count": len(list_benchmarks()),
            "practitioner_questions_count": 7,
            "calibration_methods": ["conformal", "aci", "aci_normalized"],
            "cold_start_archetypes": [
                "auto",
                "analog_transfer",
                "bass_diffusion",
                "weibull_lifecycle",
                "arps_decline",
            ],
            "similarity_methods": [
                "hybrid",
                "physics_attractor",
                "spectral_phase",
                "elastic_dtw",
                "minirocket",
                "structural_signature",
                "foundation_latent",
            ],
            "evolve_targets": [
                "features",
                "ensemble",
                "model",
                "policy",
                "similarity_kernel",
            ],
            "workbench_capabilities": [
                "promo_elasticity",
                "scenarios",
                "goal_seek",
                "temporal_reconciliation",
                "simplex",
                "cold_start",
                "aci_cqr_calibration",
                "trajectory_copula",
                "inventory_optimizer",
                "scenario_optimizer",
                "time_series_similarity",
                "sku_substitutes",
                "emerging_trends",
                "clearance_optimizer",
                "contract_volume_optimizer",
            ],
        },
        "recommended_actions": recommendations,
    }


def build_run_config_json_schema() -> dict[str, Any]:
    """Generate the canonical Draft 2020-12 JSON Schema for `RunConfig`."""
    raw = RunConfig.model_json_schema()
    props = dict(raw.get("properties", {}))
    props["$schema"] = {
        "description": "Optional JSON Schema URI for editor autocomplete and static validation.",
        "title": "$Schema",
        "type": "string",
    }
    schema: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": RUN_CONFIG_SCHEMA_URI,
        **raw,
        "properties": props,
    }
    return schema


def render_run_config_json_schema() -> str:
    """Serialize `build_run_config_json_schema` deterministically as formatted JSON."""
    return json.dumps(build_run_config_json_schema(), indent=2, sort_keys=True) + "\n"


def _format_type_annotation(annotation: Any) -> str:
    """Render a Python/Pydantic field annotation as a concise, table-safe string."""
    if annotation is None or annotation is type(None):
        return "None"
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is not None:
        origin_name = getattr(origin, "__name__", str(origin))
        if origin_name == "Literal":
            return " \\| ".join(repr(a) for a in args)
        if origin_name in ("Union", "UnionType") or "Union" in str(origin):
            return " \\| ".join(_format_type_annotation(a) for a in args)
        if origin in (list, tuple, set, frozenset):
            inner = ", ".join(_format_type_annotation(a) for a in args) if args else "Any"
            return f"{origin_name}[{inner}]"
        if origin is dict:
            if len(args) == 2:
                return (
                    f"dict[{_format_type_annotation(args[0])}, {_format_type_annotation(args[1])}]"
                )
            return "dict"
        if args:
            return _format_type_annotation(args[0])
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation).replace("typing.", "").replace("|", "\\|")


def _format_default(field: FieldInfo) -> str:
    """Render a Pydantic `FieldInfo` default value for a Markdown table cell."""
    if field.is_required():
        return "**required**"
    if field.default_factory is not None:
        factory_name = getattr(field.default_factory, "__name__", "")
        if factory_name in ("list", "dict"):
            return "`[]`" if factory_name == "list" else "`{}`"
        return f"`{factory_name}()`"
    if field.default is PydanticUndefined:
        return "**required**"
    return f"`{json.dumps(field.default)}`"


def _format_constraints(field: FieldInfo) -> str:
    """Extract numeric or length constraints from Pydantic field metadata."""
    parts: list[str] = []
    for meta in field.metadata:
        for attr, symbol in (
            ("gt", ">"),
            ("ge", ">="),
            ("lt", "<"),
            ("le", "<="),
            ("min_length", "min_len="),
        ):
            val = getattr(meta, attr, None)
            if val is not None:
                parts.append(f"`{symbol} {val}`")
    return ", ".join(parts) if parts else "—"


def render_skill_config_reference() -> str:
    """Render ``skills/scale-forecasting/references/config_reference.md`` from `RunConfig`."""
    lines: list[str] = [
        "# `RunConfig` Schema & Validation Reference",
        "",
        (
            "> **Auto-generated from [`src/scale_forecasting/config.py`]"
            "(../../../src/scale_forecasting/config.py) by "
            "`python -m scale_forecasting.agent_surfaces --write`.** "
            "Do not edit by hand; pre-commit (`test_agent_surfaces.py`) enforces zero drift."
        ),
        "",
        f"- **JSON Schema URI:** `{RUN_CONFIG_SCHEMA_URI}`",
        (
            "- **Local Schema Path:** [`docs/schemas/run_config.schema.json`]"
            "(../../../docs/schemas/run_config.schema.json)"
        ),
        (
            "- **Strictness:** Every config block sets `ConfigDict(frozen=True, extra='forbid')`. "
            "Unknown keys fail immediately at load time (`ConfigError`)."
        ),
        (
            '- **Optional `$schema` Key:** Any config JSON file may include `"$schema": "'
            + RUN_CONFIG_SCHEMA_URI
            + '"` at the top level. `RunConfig` validates and strips `$schema` before model '
            "construction so `model_dump()` and `run_id` hashes are completely unaffected."
        ),
        "",
        "---",
        "",
        "## 1. All Config Blocks & Fields",
        "",
    ]

    for section_title, summary, model_cls in _CONFIG_MODELS:
        lines.append(f"### `{section_title}`")
        lines.append("")
        lines.append(summary)
        lines.append("")
        lines.append("| Field | Type / Allowed Values | Default | Constraints |")
        lines.append("| :--- | :--- | :--- | :--- |")
        for fname, finfo in model_cls.model_fields.items():
            type_str = _format_type_annotation(finfo.annotation)
            default_str = _format_default(finfo)
            constraint_str = _format_constraints(finfo)
            lines.append(f"| `{fname}` | `{type_str}` | {default_str} | {constraint_str} |")
        lines.append("")

    metrics_inline = ", ".join(f"`{m}`" for m in METRIC_NAMES)
    learned_inline = ", ".join(f"`{s}`" for s in sorted(LEARNED_STRATEGIES))
    reconcil_inline = ", ".join(f"`{m}`" for m in sorted(RECONCILIATION_METHODS))
    families_inline = ", ".join(f"`{f}`" for f in COMPUTE_FAMILIES)
    profile_inline = ", ".join(f"`{k}`" for k in PROFILE_SOURCE_KEYWORDS)

    lines.extend(
        [
            "---",
            "",
            "## 2. Dynamic Vocabularies & Special Keys",
            "",
            "| Field | Allowed Values / Rule |",
            "| :--- | :--- |",
            (
                f"| `models` | Non-empty list of registered model names (`{len(list_models())}` "
                "available; see [`catalog_reference.md`](./catalog_reference.md)). "
                "No duplicates allowed. |"
            ),
            (
                f"| `backtest.decision_metric` | Any of the `{len(METRIC_NAMES)}` registered "
                f"metrics: {metrics_inline}. |"
            ),
            (
                "| `ensemble.strategies` | Calculated: `mean`, `median`, `inverse_error` (work "
                f"with or without backtest). Learned stackers: {learned_inline} (require "
                "`backtest.enabled=true`). Singular shorthand `strategy` is also accepted. |"
            ),
            f"| `hierarchy.reconciliation_methods` | Any subset of {reconcil_inline}. |",
            (
                f"| `compute.families` | Keys must be in {families_inline}. `native` models "
                "always run in BigQuery and never take a `compute.families` entry. |"
            ),
            (
                f"| `compute.profile.source` | One of {profile_inline} or an existing "
                "`<slug>-<12hex>` `run_id`. |"
            ),
            (
                "| `model_params` | Dict mapping `<model_name>` to `<param_dict>` of JSON-safe "
                "scalars or flat lists (`bool`, `int`, `float`, `str`, `None`). Non-finite floats "
                "(`NaN`, `Infinity`) are rejected. |"
            ),
            "",
            "### Model Variant Suffixes (`models` list)",
            "",
            "| Variant Syntax | Behavior |",
            "| :--- | :--- |",
        ]
    )
    for var_syn, var_desc in _VARIANT_SYNTAX_ROWS:
        lines.append(f"| {var_syn} | {var_desc} |")

    lines.extend(
        [
            "",
            "---",
            "",
            "## 3. Cross-Field Validation Rules (`RunConfig._normalize`)",
            "",
            *_NORMALIZATION_RULES,
            "",
            "---",
            "",
            "## 4. GPU-to-VM Machine Type Compatibility (`resolve_vm_machine_type`)",
            "",
            (
                "| `gpu_type` | Vertex Enum | VRAM | Allowed `accelerator_count` | "
                "Auto-Resolved `machine_type` | Valid Explicit `machine_type` Shapes |"
            ),
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
    )
    for g in GPU_CATALOG:
        counts = ", ".join(f"`{c}`" for c in g["allowed_counts"])
        lines.append(
            f"| `{g['gpu_type']}` | `{g['vertex_enum']}` | {g['vram_gib']} GiB | {counts} | "
            f"`{g['auto_machine_type']}` | {g['compatible_machines']} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 5. Content-Addressed `run_id` Digest Rules (`registry/ids.py`)",
            "",
            "- `run_id` is `<slug>-<12hex>` computed from the normalized `RunConfig` JSON dump.",
            (
                "- **Operational fields excluded from `run_id` "
                "(changing these keeps the same `run_id`):**"
            ),
            (
                "  - Infrastructure environment (`SF_PROJECT_ID`, `SF_DATASET_ID`, "
                "`SF_REGISTRY_DATASET_ID`, `SF_REGION`, `SF_WAREHOUSE_URI`, `SF_CONNECTION`)."
            ),
            (
                "  - `compute.capacity` (all regional retry policies and "
                "`compute.capacity.retry.max_executors`)."
            ),
            "  - `compute.profile.source` (resolved profile pointer).",
            (
                '  - Default/unset `machine_type` (`"auto"`) and default `workers` (`1`) when '
                "left at their baseline defaults."
            ),
            '  - Top-level `"$schema"` URI key.',
        ]
    )
    return "\n".join(lines) + "\n"


def render_skill_catalog_reference(repo_root: Path | None = None) -> str:
    """Render ``skills/scale-forecasting/references/catalog_reference.md`` from registries."""
    root = repo_root or _default_repo_root()
    lines: list[str] = [
        "# Canonical Platform Catalogs (Models, Metrics, Runtimes, Extras, Views & Configs)",
        "",
        (
            "> **Auto-generated from code registries by "
            "`python -m scale_forecasting.agent_surfaces --write`.** "
            "Do not edit by hand; pre-commit (`test_agent_surfaces.py`) enforces zero drift."
        ),
        "",
        "---",
        "",
        f"## 1. Forecasting Models ({len(list_models())} Models Across 5 Families)",
        "",
        (
            "| Model (`models` key) | Family | Runtime | Upstream Package | Install Extra | "
            "Future Cov. | Past Cov. | Static Cov. | Explainability | Monotone (`@monotone`) | "
            "Global / Hybrid | GPU Capable |"
        ),
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    model_names = sorted(
        list_models(),
        key=lambda n: (get_model(n).family, get_model(n).runtime, n),
    )
    for name in model_names:
        cls = get_model(name)
        extra = cls.optional_extra or ("gcp" if cls.runtime == "bigquery" else "core")
        modes = ["local"]
        if cls.supports_global:
            modes.append("global")
        if cls.supports_hybrid:
            modes.append("hybrid")
        lines.append(
            f"| `{name}` | `{cls.family}` | `{cls.runtime}` | `{cls.package}` | `{extra}` | "
            f"{'Yes' if cls.supports_future_covariates else 'No'} | "
            f"{'Yes' if cls.supports_past_covariates else 'No'} | "
            f"{'Yes' if cls.supports_static_covariates else 'No'} | "
            f"{'Yes' if cls.supports_explainability else 'No'} | "
            f"{'Yes' if cls.supports_monotonic_constraints else 'No'} | "
            f"`{'/'.join(modes)}` | "
            f"{'Yes' if cls.gpu_capable else 'No'} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            f"## 2. Evaluation Metrics ({len(METRIC_NAMES)} Metrics: 16 Point + 5 Interval)",
            "",
            (
                "| Metric (`decision_metric`) | Kind | Direction | Needs Intervals | "
                "Needs Train History | Needs Seasonal Period | Optimal Point Arm |"
            ),
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
    )
    for mname in METRIC_NAMES:
        mcls = get_metric(mname)
        kind = "interval" if mcls.needs_intervals else "point"
        arm = "mean" if mcls.mean_optimal else "median"
        lines.append(
            f"| `{mname}` | `{kind}` | `{mcls.direction}` | "
            f"{'Yes' if mcls.needs_intervals else 'No'} | "
            f"{'Yes' if mcls.needs_train_history else 'No'} | "
            f"{'Yes' if mcls.needs_seasonal_period else 'No'} | "
            f"`{arm}` |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            f"## 3. Compute Runtimes ({len(RUNTIME_CATALOG)} Runtimes)",
            "",
            (
                "| Runtime | Sub-Modes | Supported Families | GPU Support | Scaling Model | "
                "Submitter Module |"
            ),
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
    )
    for r in RUNTIME_CATALOG:
        fams = ", ".join(f"`{f}`" for f in r["families"])
        gpus = ", ".join(f"`{g}`" for g in r["gpu_types"]) if r["gpu_types"] else "CPU / SQL only"
        modes_escaped = str(r["modes"]).replace("|", "\\|")
        lines.append(
            f"| `{r['runtime']}` | {modes_escaped} | {fams} | {gpus} | {r['scaling']} | "
            f"`{r['submitter']}` |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            f"## 4. Dependency Extras ({len(ALL_EXTRA_MODULES)} Extras in `pyproject.toml`)",
            "",
            "| Extra | Install Command | Key Probe Modules | Purpose |",
            "| :--- | :--- | :--- | :--- |",
            (
                "| `core` (base) | `pip install scale-forecasting` | `pydantic`, `pandas`, "
                "`statsmodels`, `sklearn`, `optuna` | Pure offline layer: 14 core models, "
                "21 metrics, backtesting, ensembling, reconciliation, playground, dry-run, "
                "MCP server |"
            ),
        ]
    )
    for extra_name, mods in ALL_EXTRA_MODULES.items():
        mod_list = ", ".join(f"`{m}`" for m in mods[:4])
        if len(mods) > 4:
            mod_list += f" (+{len(mods) - 4} more)"
        lines.append(
            f'| `[{extra_name}]` | `pip install "scale-forecasting[{extra_name}]"` | '
            f"{mod_list} | {EXTRA_DESCRIPTIONS[extra_name]} |"
        )

    src_tables_inline = ", ".join(f"`{t}`" for t in SOURCE_TABLE_NAMES)
    reg_tables_inline = ", ".join(f"`{t}`" for t in REGISTRY_TABLE_NAMES)
    views_inline = ", ".join(f"`{v}`" for v in VIEW_NAMES)

    lines.extend(
        [
            "",
            "---",
            "",
            "## 5. BigQuery Source Tables, Registry Tables & Analytical SQL Views",
            "",
            "| Surface | Canonical Names | Description |",
            "| :--- | :--- | :--- |",
            (
                f"| **Source Tables (4)** | {src_tables_inline}, `source_series_wide_native`, "
                "`source_series_hierarchical_native` | Input time-series panels (Apache Iceberg "
                "on GCS via BigLake + native BigQuery tables). |"
            ),
            (
                f"| **Registry Tables ({len(REGISTRY_TABLE_NAMES)})** | {reg_tables_inline} | "
                "Append-only experiment header, per-family job ledger, cell metadata/metrics, "
                "forward predictions, and out-of-fold backtests. |"
            ),
            (
                f"| **Analytical SQL Views ({len(VIEW_NAMES)})** | {views_inline} | "
                "Self-documenting leaderboards, comparable cohort rankings, backtest fold "
                "coverage, run time & job ledgers, FVA waterfall, horizon decay, Trigg bias & "
                "exceptions, anomaly flags, feature attributions, and executive S&OP summary. |"
            ),
            "",
            "---",
            "",
            "## 6. Shipped Configuration Catalog (`configs/` & `configs/smokes/`)",
            "",
            "### Root Demonstration & Scale Configs (`configs/*.json`)",
            "",
            "| Config File | `run_name` | Default Runtime | Models | Backtest | Ensemble |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
    )

    demo_dir = root / "configs"
    if demo_dir.is_dir():
        for p in sorted(demo_dir.glob("*.json")):
            if p.name == "compute_fallback.json":
                continue
            cfg = RunConfig.model_validate_json(p.read_text(encoding="utf-8"))
            models_preview = ", ".join(f"`{m}`" for m in cfg.models[:4])
            if len(cfg.models) > 4:
                models_preview += f" (+{len(cfg.models) - 4})"
            ens = (
                ", ".join(f"`{s}`" for s in cfg.ensemble.strategies)
                if cfg.ensemble.enabled
                else "off"
            )
            bt = (
                f"`{cfg.backtest.scheme}` ({cfg.backtest.n_folds}f)"
                if cfg.backtest.enabled
                else "off"
            )
            lines.append(
                f"| `configs/{p.name}` | `{cfg.run_name}` | `{cfg.python_runtime}` | "
                f"{models_preview} | {bt} | {ens} |"
            )

    lines.extend(
        [
            "",
            "### Live Smoke Test Configs (`configs/smokes/*.json`)",
            "",
            "| Smoke Config | `run_name` | Default Runtime | Family Overrides | Models |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
    )
    smoke_dir = root / "configs" / "smokes"
    if smoke_dir.is_dir():
        for p in sorted(smoke_dir.glob("*.json")):
            cfg = RunConfig.model_validate_json(p.read_text(encoding="utf-8"))
            fam_ov = (
                ", ".join(
                    f"`{f}:{fc.runtime or cfg.python_runtime}`"
                    for f, fc in sorted(cfg.compute.families.items())
                )
                if cfg.compute.families
                else "—"
            )
            models_preview = ", ".join(f"`{m}`" for m in cfg.models[:4])
            if len(cfg.models) > 4:
                models_preview += f" (+{len(cfg.models) - 4})"
            lines.append(
                f"| `configs/smokes/{p.name}` | `{cfg.run_name}` | `{cfg.python_runtime}` | "
                f"{fam_ov} | {models_preview} |"
            )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 7. Auto-Forecaster Exposure Tiers, Scale Regimes & Synthetic DGP Benchmarks",
            "",
            "### Observational Exposure Tiers (`EXPOSURE_TIERS` in `auto_policy.py`)",
            "",
            "| Tier (`tier`) | Title & Typical Duration | Description |",
            "| :--- | :--- | :--- |",
        ]
    )
    for t_name, t_meta in EXPOSURE_TIERS.items():
        t_desc = str(t_meta["description"]).replace("|", "\\|")
        lines.append(f"| `{t_name}` | {t_meta['title']} | {t_desc} |")

    lines.extend(
        [
            "",
            "### Series-Count Scale Regimes (`SCALE_REGIMES` in `auto_policy.py`)",
            "",
            "| Scale Regime (`scale_regime`) | Series Count | Routing Strategy |",
            "| :--- | :--- | :--- |",
        ]
    )
    for s_name, s_meta in SCALE_REGIMES.items():
        s_strat = str(s_meta["strategy"]).replace("|", "\\|")
        lines.append(f"| `{s_name}` | {s_meta['title']} | {s_strat} |")

    bench_specs = list(BENCHMARK_SPECS.values())
    lines.extend(
        [
            "",
            (
                f"### Ground-Truth Synthetic DGP Benchmarks "
                f"({len(bench_specs)} Panels in `benchmarks.py`)"
            ),
            "",
            (
                "| Benchmark Name (`benchmark`) | Regime | Freq / H | "
                "Ground-Truth Description | Expected `autofocus` Actions |"
            ),
            "| :--- | :--- | :---: | :--- | :--- |",
        ]
    )
    for b_spec in bench_specs:
        desc = str(b_spec.description).replace("|", "\\|")
        acts = (
            ", ".join(f"`{a}`" for a in b_spec.expected_autofocus_actions)
            if b_spec.expected_autofocus_actions
            else "`stop_after_pass_1`"
        )
        lines.append(
            f"| `{b_spec.name}` | `{b_spec.expected_regime}` | "
            f"`{b_spec.freq}` / `{b_spec.horizon}` | {desc} | {acts} |"
        )

    pub_specs = list(PUBLIC_BENCHMARK_SPECS.values())
    lines.extend(
        [
            "",
            (
                f"### Public Open-Data Benchmarks "
                f"({len(pub_specs)} Real-World Datasets in `benchmarks.py` via "
                f"`fetch_public_benchmark`)"
            ),
            "",
            ("| Benchmark Name (`name`) | Domain | Freq / H | License | Official Source URL |"),
            "| :--- | :--- | :---: | :--- | :--- |",
        ]
    )
    for p_spec in pub_specs:
        lines.append(
            f"| `{p_spec.name}` | `{p_spec.domain}` | "
            f"`{p_spec.freq}` / `{p_spec.horizon}` | {p_spec.license} | {p_spec.homepage_url} |"
        )

    return "\n".join(lines) + "\n"


def render_skill_execution_paths_reference() -> str:
    """Render ``skills/scale-forecasting/references/execution_paths_and_ops.md``."""
    lines: list[str] = [
        "# Execution Paths, Environment Variables & Operations Reference",
        "",
        (
            "> **Auto-generated by `python -m scale_forecasting.agent_surfaces --write`.** "
            "Do not edit by hand; pre-commit (`test_agent_surfaces.py`) enforces zero drift."
        ),
        "",
        "---",
        "",
        "## 1. Environment Variables (`Settings.resolve`)",
        "",
        (
            "All cloud execution paths resolve their Google Cloud infrastructure target from "
            "`SF_*` environment variables (typically exported by `terraform/main` into "
            "`.env.infra`). `RunConfig` never embeds project IDs or bucket names so the same "
            "config is portable across environments."
        ),
        "",
        "| Variable | Required for Cloud? | Default | Purpose |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for var_name, req, default_val, desc in _ENV_VARS_DOC:
        lines.append(f"| `{var_name}` | {'**Yes**' if req else 'No'} | `{default_val}` | {desc} |")

    lines.extend(
        [
            "",
            "Load `.env.infra` in any shell before running cloud commands:",
            "",
            "```bash",
            "set -a && source .env.infra && set +a",
            "```",
            "",
            "---",
            "",
            "## 2. The 5 Execution Paths",
            "",
            "### Path 1: Offline Playground & Dry-Run (Zero GCP Required)",
            "",
            (
                "Works on a bare `pip install scale-forecasting` core install with no Google "
                "Cloud credentials."
            ),
            "",
            "```bash",
            "# Probe installed extras, available models, and environment readiness",
            "python -m scale_forecasting.agent_surfaces --probe-env",
            "",
            "# List all 37 models and 21 metrics with availability and capabilities",
            "python -m scale_forecasting.playground --list",
            "python -m scale_forecasting.playground --catalog",
            "",
            "# Run any installed Python model on synthetic panel data with 3-fold backtesting",
            "python -m scale_forecasting.playground --model holtwinters --horizon 14 --backtest",
            "",
            "# Validate a RunConfig, compute deterministic run_id, and estimate workload offline",
            "python -m scale_forecasting.main --config configs/ensemble_demo.json --dry-run",
            "```",
            "",
            "Python SDK equivalent:",
            "",
            "```python",
            "from scale_forecasting import Forecaster",
            "",
            'f = Forecaster.from_file("configs/ensemble_demo.json")',
            "dry = f.dry_run()          # DryRunResult(run_id, fanout, python_models, bq_models)",
            "dag_nodes = f.dag()        # Planned per-family DagNode tuple",
            "```",
            "",
            "### Path 2: Cloud Readiness, Feasibility, Quota Preflight & Staging (`[gcp]`)",
            "",
            (
                "Reads BigQuery source metadata or regional quota meters and stages artifacts to "
                "Cloud Storage without launching compute jobs."
            ),
            "",
            "```bash",
            "# Check achieved backtest fold cohorts against live BigQuery series history lengths",
            (
                "python -m scale_forecasting.main "
                "--config configs/ensemble_demo.json --dry-run --feasibility"
            ),
            "",
            "# Check regional CPU/GPU quota headroom and GPU usefulness warnings",
            "python -m scale_forecasting.main --config configs/ray_gpu_demo.json --quota",
            "",
            "# Upload config + code archive to GCS, write STAGED manifest, and print commands",
            "python -m scale_forecasting.main --config configs/ensemble_demo.json --stage-only",
            "```",
            "",
            "### Path 3: Unified Multi-Family DAG Execution (`main.run` / `Forecaster.run`)",
            "",
            (
                "Dispatches 1 independent job per active model family (`statistical`, `ml`, "
                "`deep_learning`, `automl`, `native`) in parallel under one shared `run_id`, "
                "followed by the `ensemble` node."
            ),
            "",
            "```bash",
            "# Run full multi-family DAG end-to-end",
            "python -m scale_forecasting.main --config configs/ensemble_demo.json",
            "",
            "# Force a fresh attempt of an already-completed config",
            ("python -m scale_forecasting.main --config configs/ensemble_demo.json --force"),
            "",
            "# Filter config to models whose optional Python packages are installed locally",
            (
                "python -m scale_forecasting.main "
                "--config configs/all_models.json --ignore-unavailable-models --dry-run"
            ),
            "```",
            "",
            "Python SDK equivalent:",
            "",
            "```python",
            "from scale_forecasting import Forecaster",
            "",
            'f = Forecaster.from_file("configs/ensemble_demo.json")',
            (
                "result = f.run(n_series=100)   "
                "# RunResult(run_id, dataset_ref, views, status, runtime_seconds)"
            ),
            "```",
            "",
            "### Path 4: Direct Per-Family Cloud Submitters",
            "",
            (
                "Submit a single family or runtime directly when driving jobs from an external "
                "scheduler:"
            ),
            "",
            "```bash",
            (
                "python -m scale_forecasting.submit "
                "--config configs/example_config.json --family statistical"
            ),
            (
                "python -m scale_forecasting.ray_submit "
                "--config configs/ray_gpu_demo.json --family deep_learning"
            ),
            (
                "python -m scale_forecasting.vertex_submit "
                "--config configs/vertex_demo.json --family ml"
            ),
            (
                "python -m scale_forecasting.gce_submit "
                "--config configs/gce_demo.json --family statistical"
            ),
            (
                "python -m scale_forecasting.gke_submit "
                "--config configs/gke_demo.json --family statistical"
            ),
            (
                "python -m scale_forecasting.automl_submit "
                "--config configs/automl_demo.json --family automl"
            ),
            "python -m scale_forecasting.ensemble_run --config configs/ensemble_demo.json",
            "```",
            "",
            "### Path 5: Cloud Composer 3 / Apache Airflow DAG Emission",
            "",
            (
                "Renders a self-contained Airflow DAG file (`dag_<run_id>.py`) offline with one "
                "task per active model family, optional surgical repair node (`--with-retry`), "
                "and downstream ensemble task:"
            ),
            "",
            "```bash",
            (
                "python -m scale_forecasting.main "
                "--config configs/ensemble_demo.json --emit-airflow --with-retry"
            ),
            (
                "make composer-sync   "
                "# syncs src/scale_forecasting to Composer plugins bucket when Composer is enabled"
            ),
            "```",
            "",
            "---",
            "",
            "## 3. Monitoring, Post-Run Review, Calibration & Surgical Repair",
            "",
            "| Goal | CLI Command | Python SDK / Function |",
            "| :--- | :--- | :--- |",
            (
                "| **Reconcile live job status** | "
                "`python -m scale_forecasting.main --config <cfg> --probe [--job <family>]` | "
                "`Forecaster(cfg).monitor(probe=True)` / `monitor_run(run_id, probe=True)` |"
            ),
            (
                "| **Preview stale row repair** | "
                "`python -m scale_forecasting.main --config <cfg> --settle` | "
                "`Forecaster(cfg).settle(yes=False)` |"
            ),
            (
                "| **Apply stale row repair** | "
                "`python -m scale_forecasting.main --config <cfg> --settle --force "
                '--reason "..."` | '
                '`Forecaster(cfg).settle(yes=True, reason="...")` |'
            ),
            (
                "| **Preview job cancellation** | "
                "`python -m scale_forecasting.main --config <cfg> --cancel` | "
                "`Forecaster(cfg).cancel(confirm=False)` |"
            ),
            (
                "| **Confirm job cancellation** | "
                "`python -m scale_forecasting.main --config <cfg> --cancel --force "
                '--reason "..."` | '
                '`Forecaster(cfg).cancel(confirm=True, reason="...")` |'
            ),
            (
                "| **Preview surgical cell repair** | "
                "`python -m scale_forecasting.main --run-id <run_id> --retry` | "
                "`Forecaster(cfg).retry(confirm=False)` / `retry_run(cfg, confirm=False)` |"
            ),
            (
                "| **Submit surgical cell repair** | "
                "`python -m scale_forecasting.main --run-id <run_id> --retry --force "
                '--reason "..."` | '
                '`Forecaster(cfg).retry(confirm=True, reason="...")` |'
            ),
            (
                "| **Full data-science run review** | — | "
                "`review_run(run_id)` / `Forecaster(cfg).review()` |"
            ),
            "| **Interval & arm calibration** | — | `calibration_report(run_id)` |",
            (
                "| **Forecaster's Workbench, Goal-Seek & Temporal Reconciliation** | — | "
                "`Forecaster(cfg).fva_df()` / `.horizon_decay_df()` / `.bias_report_df()` / "
                "`.exceptions_df()` / `.promo_elasticity_df()` / `.inventory_policy_df()` / "
                "`.allocate_target()` / `.temporal_reconcile_df()` |"
            ),
            (
                "| **Cold-Start NPI, Online ACI, Copula Tail-Risk & Prescriptive Optimization** | "
                "— | "
                "`Forecaster(cfg).cold_start_transfer_df()` / `.aci_df()` / "
                "`.sample_trajectories()` / `.tail_risk_df()` / "
                "`.optimize_inventory_df()` / `.optimize_scenario()` / "
                "`.optimize_clearance_df()` / `.optimize_contract_volume_df()` / "
                "`.optimize_assortment_df()` / `.optimize_prepacks_df()` / "
                "`.optimize_rebalancing_df()` |"
            ),
            (
                "| **Time-Series Similarity, Donor Matching, Substitutes & Emerging Trends** | — | "
                "`Forecaster(cfg).similarity_df()` / `.find_similar_series_df()` / "
                "`.cluster_by_similarity_df()` / `.recommend_substitutes_df()` / "
                "`.emerging_trends_df()` / `compute_time_series_similarity` |"
            ),
            (
                "| **Executive HTML + JSON report** | "
                "`python -m scale_forecasting.report --model holtwinters --output-dir ./report` | "
                "`Forecaster(cfg).export_report()` / `export_report(bench, './report')` |"
            ),
            (
                "| **Glass-Box Auto-Forecaster & `autofocus`** | "
                "`python -m scale_forecasting.main --auto --benchmark dgp_04_true_vs_decoy_covs "
                "--tier first_light --autofocus --max-passes 3` | "
                "`Forecaster.profile_data(panel)` / "
                "`Forecaster.build_config(panel, tier='wide_field')` / "
                "`Forecaster.auto(panel, tier='first_light', autofocus=True)` |"
            ),
            (
                "| **7-Question Practitioner Assessment & Model Comparison** | "
                "`python -m scale_forecasting.main --auto --benchmark dgp_04_true_vs_decoy_covs "
                "--autofocus` | "
                "`f.practitioner_assessment_df()` / `f.assess()` / `f.compare_models()` / "
                "`f.baseline_value_df()` / `f.hierarchy_value_df()` |"
            ),
            (
                "| **Auto-Forecaster Composer 3 / Airflow DAG Emission** | "
                "`python -m scale_forecasting.main --auto --table source_series_native "
                "--tier wide_field --autofocus --emit-out /tmp/dag_autofocus.py` | "
                "`bp.emit_airflow_dag(autofocus=True)` / `f.emit_airflow_dag()` |"
            ),
            (
                "| **Opt-In AlphaEvolve Code Evolution (`evolve`)** | "
                "`python -m scale_forecasting.evolve --target features --backend alphaevolve "
                "--model-name gemini-3.8-flash --dry-run` / `--from-run-ids <r1>,<r2>` | "
                "`Forecaster.evolve(panel, target='features', model_name='gemini-3.8-flash')` / "
                "`f.evolve(target='ensemble')` / `f.evolve(target='similarity_kernel')` / "
                "`Forecaster.evolve(from_run_ids=[...], target='ensemble')` |"
            ),
            (
                "| **Knowledge Catalog & Stateful BigQuery `DataAgent` (`bq_agent`)** | "
                "`python -m scale_forecasting.bq_agent --dataset scale_forecasting --ensure` / "
                '`--ask "Which model won on WAPE and why?"` | '
                '`Forecaster.ensure_bq_agent()` / `f.ask("...")` / `f.sop_summary_df()` / '
                '`Registry.ask("...")` |'
            ),
            "",
            "---",
            "",
            "## 4. Registry Lifecycle Operations (`python -m scale_forecasting.registry.ops`)",
            "",
            (
                "All mutating verbs default to **preview mode** and require `--yes` (CLI) or "
                "`yes=True` (Python) to execute:"
            ),
            "",
            "| Verb | CLI Invocation | What It Does |",
            "| :--- | :--- | :--- |",
            (
                "| `init` | `python -m scale_forecasting.registry.ops init` | "
                "Idempotently create the 5 registry tables and 11 analytical SQL views, and "
                "sync missing column/table descriptions. |"
            ),
            (
                "| `doctor` | `python -m scale_forecasting.registry.ops doctor` | "
                "Read-only health report: table row counts, stuck `RUNNING` runs, orphaned "
                "GCS artifacts. |"
            ),
            (
                "| `close-runs` | `python -m scale_forecasting.registry.ops close-runs [--yes]` | "
                "Finalize abandoned `RUNNING` headers whose `run_jobs` rows are all terminal. |"
            ),
            (
                "| `drop-run` | "
                "`python -m scale_forecasting.registry.ops drop-run <run_id> [--yes]` | "
                "Delete named run(s) across GCS artifacts, BigQuery rows, and BQML model "
                "objects. |"
            ),
            (
                "| `sweep-orphans` | "
                "`python -m scale_forecasting.registry.ops sweep-orphans [--yes]` | "
                "Delete GCS artifact prefixes under this registry with no `run_registry` "
                "header row. |"
            ),
            (
                "| `reap-clusters` | "
                "`python -m scale_forecasting.registry.ops reap-clusters [--yes]` | "
                "Delete orphaned Vertex AI Ray clusters whose owning run is already terminal. |"
            ),
            (
                "| `snapshot` | "
                "`python -m scale_forecasting.registry.ops snapshot --suffix <tag> [--yes]` | "
                "Create expirable BigQuery table snapshots of all 5 registry tables. |"
            ),
            (
                "| `export` | "
                "`python -m scale_forecasting.registry.ops export "
                "--Dest gs://... [--format PARQUET\\|JSON] [--yes]` | "
                "Export registry tables to Cloud Storage in Parquet or newline-delimited JSON. |"
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def render_llms_txt() -> str:
    """Render ``docs/llms.txt`` following the standard ``llms.txt`` specification."""
    return "\n".join(
        [
            "# scale-forecasting",
            "",
            (
                "> Declarative, multi-runtime enterprise time-series forecasting platform for "
                "Google Cloud. One frozen Pydantic `RunConfig` orchestrates 37 models across 5 "
                "families (`statistical`, `ml`, `deep_learning`, `automl`, `native`), 21 "
                "evaluation metrics, 7 compute runtimes (`spark`, `ray`, `vertex`, `gce`, `gke`, "
                "`vertex_automl`, `bigquery`), rolling-origin backtesting, Optuna HPO, "
                "conformal/quantile calibration (`conformal`, `aci`, `aci_normalized`), FPP3 "
                "spatial/temporal/cross-temporal hierarchical reconciliation, and stacked "
                "ensembling into a self-documenting 5-table, 11-view BigQuery registry with "
                "Knowledge Catalog and a stateful BigQuery Conversational Analytics `DataAgent`."
            ),
            "",
            "## Agent-First Surfaces (MCP, Skills & Schemas)",
            "",
            (
                f"- [AI Agents, Skills & MCP Guide]({_SITE_BASE}/agent_and_mcp_guide/): How to "
                "connect Google Antigravity (`agy` CLI & IDE), Claude Code, Cursor, VS Code "
                "Copilot, and Windsurf via the built-in MCP server (`python -m "
                "scale_forecasting.mcp`), `SKILL.md`, and JSON Schema."
            ),
            (
                f"- [RunConfig JSON Schema (Draft 2020-12)]"
                f"({_SITE_BASE}/schemas/run_config.schema.json): Machine-readable schema for "
                f'`RunConfig` with editor autocomplete (`"$schema": "{RUN_CONFIG_SCHEMA_URI}"`).'
            ),
            (
                f"- [Comprehensive LLM Context Bundle (`llms-full.txt`)]"
                f"({_SITE_BASE}/llms-full.txt): Single-file reference containing all `RunConfig` "
                "fields, all 37 models, 21 metrics, 7 runtimes, 13 extras, and 5 execution paths."
            ),
            (
                f"- [Repository Portable Skill (`skills/scale-forecasting/SKILL.md`)]"
                f"({_REPO_URL}/blob/main/skills/scale-forecasting/SKILL.md): Portable Agent Skill "
                "with Progressive Disclosure reference tables."
            ),
            "",
            "## Start Here & Positioning",
            "",
            f"- [Documentation Home]({_SITE_BASE}/): Navigation hub by role and goal.",
            (
                f"- [Platform Overview]({_SITE_BASE}/overview/): End-to-end architecture, "
                "capabilities summary, and registry model."
            ),
            (
                f"- [Visual System Walkthrough]({_SITE_BASE}/visual_architecture/): Progressive "
                "9-level (Levels 0-8 + Persona Map) color-coded diagram tour mapped directly to "
                "code, guides, notebooks, and tests."
            ),
            (
                f"- [Getting Started]({_SITE_BASE}/getting_started/): 5-minute offline sandbox "
                "(`pip install scale-forecasting`) to first cloud run."
            ),
            (
                f"- [Choosing a Runtime]({_SITE_BASE}/choosing_a_runtime/): Decision flow and fit "
                "matrix across all 7 Google Cloud runtimes."
            ),
            (
                f"- [Cost Estimates]({_SITE_BASE}/cost_estimates/): Service-by-service cost "
                "drivers, zero-idle vs. always-on resources, and run cost bands."
            ),
            (
                f"- [Why Google Cloud]({_SITE_BASE}/why_google_cloud/): Six architectural proof "
                "points with code and validation citations."
            ),
            (
                f"- [FAQ]({_SITE_BASE}/faq/): Direct answers on installation, runtimes, pricing, "
                "covariates, and operations."
            ),
            (
                f"- [Glossary]({_SITE_BASE}/glossary/): Definitions of platform, forecasting, and "
                "BigQuery registry terms."
            ),
            "",
            "## Reference & Operations",
            "",
            (
                f"- [Configuration Reference (`RunConfig`)]"
                f"({_SITE_BASE}/configuration_reference/): Complete JSON configuration guide."
            ),
            (
                f"- [Forecaster's Workbench]({_SITE_BASE}/forecaster_workbench/): Physical "
                "bounds, monotone quantiles, Newsvendor targets, SBC demand regimes, fold-safe "
                "hygiene, FVA, horizon decay, Trigg bias, AR(1) time rollup, THieF temporal & "
                "cross-temporal reconciliation, top-down Goal-Seek QP (`allocate_target`), "
                "compositional simplex (`clr`/`ilr`), Double-ML promo elasticity "
                "(`promo_elasticity_df`), cold-start NPI donor transfer "
                "(`cold_start_transfer_df`), online ACI conformal calibration (`aci_df`), joint "
                "copula trajectories & tail risk (`sample_trajectories`, `tail_risk_df`), "
                "multi-modal time-series similarity, donor matching, SKU substitutes & emerging "
                "trends (`similarity_df`, `find_similar_series_df`, `cluster_by_similarity_df`, "
                "`recommend_substitutes_df`, `detect_emerging_trends_df`, "
                "`compute_time_series_similarity`), portfolio inventory optimization "
                "(`optimize_inventory_df`), lifecycle markdown, space reclamation & vendor "
                "contract volume MILP (`optimize_clearance_df`, `optimize_contract_volume_df`), "
                "assortment delisting transference (`optimize_assortment_df`), simplex size-curve "
                "& integer pre-pack allocation (`optimize_prepacks_df`), lateral store-to-store "
                "rebalancing (`optimize_rebalancing_df`), "
                "prescriptive price & promo optimization (`optimize_scenario`), scenarios, and "
                "executive HTML/JSON reports."
            ),
            (
                f"- [Time-Series Similarity & Merchandising]"
                f"({_SITE_BASE}/time_series_similarity/): "
                "7-lens mathematical similarity (`hybrid`, `physics_attractor`, `spectral_phase`, "
                "`elastic_dtw`, `minirocket`, `structural_signature`, `foundation_latent`), "
                "multi-modal text/attribute/hierarchy fusion, Day-0 to Day-14 cold-start donor "
                "transfer, inventory-aware SKU substitutes (`recommend_substitutes_df`), and "
                "emerging trend breakouts (`detect_emerging_trends_df`)."
            ),
            (
                f"- [Temporal & Cross-Temporal Reconciliation]"
                f"({_SITE_BASE}/temporal_reconciliation_reference/): Mode 1 spatial (`BottomUp`, "
                "`WLS-Struct`, `MinT-Shrink`, `MinT-NonNeg`), Mode 2 temporal (`THieF`), and Mode "
                "3 cross-temporal (`cross_temporal_mint`) coherence plus compositional simplex "
                "(`@transform_clr`, `@transform_ilr`)."
            ),
            (
                f"- [Auto-Forecaster & Bounded `autofocus`]({_SITE_BASE}/auto_forecaster/): "
                "Step 0 data profiling (`Forecaster.profile_data`), observational exposure tiers "
                f"({', '.join(f'`{t}`' for t in EXPOSURE_TIERS)}), multi-branch `autofocus` "
                "refinement with cell borrowing, 7-Question Practitioner Assessment "
                "(`f.practitioner_assessment_df()`, `f.compare_models()`), 7-runtime parametric "
                f"wall-clock predictor, {len(list_benchmarks())} synthetic DGP benchmarks, and "
                "AlphaEvolve calibration."
            ),
            (
                f"- [Auto-Forecaster Benchmarks (29 Datasets & AlphaEvolve)]"
                f"({_SITE_BASE}/auto_forecaster_benchmarks/): Full 4-tier × `autofocus` "
                f"progression across {len(list_benchmarks())} synthetic DGP recipes, 8 "
                "AlphaEvolve evolution stages, and 9 public open-data benchmarks (`M5`, "
                "`Favorita`, `M4`, `Monash`, `ETT`)."
            ),
            (
                f"- [Auto-Forecaster Cloud Scale (7 Runtimes & AutoML)]"
                f"({_SITE_BASE}/auto_forecaster_cloud_scale/): 70+ live Google Cloud runs "
                "(`10` -> `100,000` series) across `bigquery`, `spark`, `vertex` (`L4` GPU & "
                "CPU), `gce`, `gke`, `ray`, and `vertex_automl` (`vertex_tide`, `vertex_l2l`), "
                "plus cross-run cell borrowing."
            ),
            (
                f"- [AlphaEvolve Runtime Harness (Opt-In)]({_SITE_BASE}/alphaevolve_runner/): "
                "Opt-in evolutionary code discovery (`Forecaster.evolve`, `python -m "
                "scale_forecasting.evolve`) across `# EVOLVE-BLOCK` targets (`features`, "
                "`ensemble`, `model`, `policy`, `similarity_kernel`) with AST security sandbox, "
                "temporal lookahead probe, and 3-Gate Anti-Overfitting Ratchet (default "
                "`backend='alphaevolve'` on Gemini Enterprise Discovery Engine + "
                "`backend='vertex_gemini'` backup on Vertex AI Gemini)."
            ),
            (
                f"- [Models Reference (37 Models)]({_SITE_BASE}/models_reference/): All 37 "
                "models, covariate support, explainability, and granular extras."
            ),
            (
                f"- [Metrics Reference (21 Metrics)]({_SITE_BASE}/metrics_reference/): All 16 "
                "point and 5 interval metrics."
            ),
            (
                f"- [Runtimes Reference (7 Runtimes)]({_SITE_BASE}/runtimes_reference/): "
                "Dataproc, Ray, Vertex CustomJob, GCE, GKE, Vertex AutoML, and BigQuery ML."
            ),
            (
                f"- [Using the Python SDK (`Forecaster`)]({_SITE_BASE}/using_the_sdk/): "
                "Programmatic execution, monitoring, review, and plotting."
            ),
            (
                f"- [Output Schemas & Views]({_SITE_BASE}/output_schemas/): DDL and query "
                "recipes for the 5 registry tables, 11 self-documenting analytical views, and "
                "Knowledge Catalog / BigQuery DataAgent."
            ),
            (
                f"- [Operations & Surgical Repair]({_SITE_BASE}/operations/): Probing, settling, "
                "canceling, and repairing runs (`retry_run`, `registry.ops`)."
            ),
            (
                f"- [Live Validation Ledger]({_SITE_BASE}/validation/): Proven `run_id` records "
                "across 42 smokes, 20 demo configs, and 13 notebooks."
            ),
            "",
        ]
    )


def render_llms_full_txt(repo_root: Path | None = None) -> str:
    """Render ``docs/llms-full.txt`` combining all generated agent reference surfaces."""
    root = repo_root or _default_repo_root()
    parts = [
        render_llms_txt().rstrip(),
        "",
        "================================================================================",
        "",
        render_skill_config_reference().rstrip(),
        "",
        "================================================================================",
        "",
        render_skill_catalog_reference(root).rstrip(),
        "",
        "================================================================================",
        "",
        render_skill_execution_paths_reference().rstrip(),
        "",
    ]
    return "\n".join(parts)


_VISUAL_INVENTORY_BEGIN = "<!-- BEGIN AUTO-SYNCED VISUAL INVENTORY -->"
_VISUAL_INVENTORY_END = "<!-- END AUTO-SYNCED VISUAL INVENTORY -->"


def _visual_inventory_counts(repo_root: Path | None = None) -> dict[str, int]:
    """Collect live repository counts via Python reflection for ``docs/visual_architecture.md``."""
    from .bq_agent.glossary import GLOSSARY_TERMS
    from .bq_agent.golden_queries import build_golden_queries
    from .config import VALID_VARIANT_TAGS
    from .mcp import McpServer
    from .registry.metadata import COLUMN_DESCRIPTIONS, TABLE_DESCRIPTIONS, VIEW_DESCRIPTIONS
    from .similarity import VALID_SIMILARITY_METHODS

    root = repo_root or _default_repo_root()
    fam_counts: dict[str, int] = dict.fromkeys(
        ("statistical", "ml", "deep_learning", "automl", "native"), 0
    )
    for model_name in list_models():
        fam = get_model(model_name).family
        fam_counts[fam] = fam_counts.get(fam, 0) + 1

    n_metrics = len(METRIC_NAMES)
    n_interval = sum(1 for name in METRIC_NAMES if get_metric(name).needs_intervals)
    srv = McpServer()
    notebooks_dir = root / "notebooks"
    smokes_dir = root / "configs" / "smokes"
    configs_dir = root / "configs"
    tf_mods_dir = root / "terraform" / "main" / "modules"

    n_notebooks = len(list(notebooks_dir.glob("*.ipynb"))) if notebooks_dir.is_dir() else 13
    n_smokes = len(list(smokes_dir.glob("*.json"))) if smokes_dir.is_dir() else 42
    n_demos = (
        len([p for p in configs_dir.glob("*.json") if p.name != "compute_fallback.json"])
        if configs_dir.is_dir()
        else 20
    )
    n_tf_mods = (
        len([p for p in tf_mods_dir.iterdir() if p.is_dir()]) if tf_mods_dir.is_dir() else 12
    )
    return {
        "n_models": len(list_models()),
        "n_stat": fam_counts["statistical"],
        "n_ml": fam_counts["ml"],
        "n_dl": fam_counts["deep_learning"],
        "n_automl": fam_counts["automl"],
        "n_native": fam_counts["native"],
        "n_metrics": n_metrics,
        "n_point_metrics": n_metrics - n_interval,
        "n_interval_metrics": n_interval,
        "n_runtimes": len(RUNTIME_CATALOG),
        "n_gpus": len(GPU_CATALOG),
        "n_run_fields": len(RunConfig.model_fields),
        "n_tiers": len(EXPOSURE_TIERS),
        "n_scale_regimes": len(SCALE_REGIMES),
        "n_dgps": len(BENCHMARK_SPECS),
        "n_pub_bench": len(PUBLIC_BENCHMARK_SPECS),
        "n_strategies": len(get_args(Strategy)),
        "n_reconcile": len(RECONCILIATION_METHODS),
        "n_variant_tags": len(VALID_VARIANT_TAGS),
        "n_sim_lenses": len(VALID_SIMILARITY_METHODS),
        "n_reg_tables": len(REGISTRY_TABLE_NAMES),
        "n_views": len(VIEW_NAMES),
        "n_core_views": len(CORE_VIEW_NAMES),
        "n_diag_views": len(DIAGNOSTIC_VIEW_NAMES),
        "n_meta_objs": len(TABLE_DESCRIPTIONS) + len(VIEW_DESCRIPTIONS),
        "n_meta_cols": sum(len(cols) for cols in COLUMN_DESCRIPTIONS.values()),
        "n_glossary": len(GLOSSARY_TERMS),
        "n_golden": len(build_golden_queries("proj", "ds")),
        "n_mcp_res": len(srv.list_resources()),
        "n_mcp_tools": len(srv.list_tools()),
        "n_extras": len(ALL_EXTRA_MODULES),
        "n_notebooks": n_notebooks,
        "n_smokes": n_smokes,
        "n_demos": n_demos,
        "n_tf_mods": n_tf_mods,
    }


def render_visual_architecture_inventory(repo_root: Path | None = None) -> str:
    """Render the auto-synced inventory table embedded in ``docs/visual_architecture.md``."""
    c = _visual_inventory_counts(repo_root)
    src_blob = f"{_REPO_URL}/blob/main/src/scale_forecasting"
    rows = [
        _VISUAL_INVENTORY_BEGIN,
        "| System Surface | Live Code Count & Breakdown | Canonical Source of Truth |",
        "| :--- | :--- | :--- |",
        (
            f"| **Forecasting Models** | **{c['n_models']} models** across **5 families** "
            f"(`{c['n_stat']}` `statistical`, `{c['n_ml']}` `ml`, `{c['n_dl']}` `deep_learning`, "
            f"`{c['n_automl']}` `automl`, `{c['n_native']}` `native`) + "
            f"**{c['n_variant_tags']} `@tag` modifiers** | "
            f"[`models/__init__.py`]({src_blob}/models/__init__.py), "
            f"[`config.py`]({src_blob}/config.py) |"
        ),
        (
            f"| **Evaluation Metrics** | **{c['n_metrics']} metrics** (`{c['n_point_metrics']}` "
            f"point accuracy + `{c['n_interval_metrics']}` prediction interval) | "
            f"[`metrics/__init__.py`]({src_blob}/metrics/__init__.py) |"
        ),
        (
            f"| **Cloud Runtimes & GPUs** | **{c['n_runtimes']} runtimes** (`spark`, `ray`, "
            "`vertex`, `gce`, `gke`, `vertex_automl`, `bigquery`) · "
            f"**{c['n_gpus']} GPU types** (`T4`, `L4`, `A100`, `A100_80GB`) | "
            f"[`config.py`]({src_blob}/config.py), "
            f"[`resources/catalog.py`]({src_blob}/resources/catalog.py) |"
        ),
        (
            f"| **Config, Auto-Compiler & Benchmarks** | **{c['n_run_fields']} root `RunConfig` "
            f"fields** · **{c['n_tiers']} exposure tiers** · **{c['n_scale_regimes']} scale "
            f"regimes** · **{c['n_dgps']} synthetic DGPs + "
            f"{c['n_pub_bench']} public benchmarks** | "
            f"[`auto.py`]({src_blob}/auto.py), "
            f"[`auto_policy.py`]({src_blob}/auto_policy.py), "
            f"[`benchmarks.py`]({src_blob}/benchmarks.py) |"
        ),
        (
            "| **Similarity, Ensembles & Coherence** | "
            f"**{c['n_sim_lenses']} similarity lenses** · "
            f"**{c['n_strategies']} ensemble strategies** · **{c['n_reconcile']} reconciliation "
            "methods** (3-mode `THieF` & `cross_temporal_mint` + `clr`/`ilr` simplex) | "
            f"[`similarity.py`]({src_blob}/similarity.py), "
            f"[`ensembler.py`]({src_blob}/ensembler.py), "
            f"[`temporal_reconciliation.py`]({src_blob}/temporal_reconciliation.py) |"
        ),
        (
            f"| **Always-On BigQuery Registry & Views** | **{c['n_reg_tables']} native registry "
            f"tables** · **{c['n_views']} always-created SQL views** (`{c['n_core_views']}` core + "
            f"`{c['n_diag_views']}` diagnostic/S&OP) · `{c['n_meta_objs']}` object & "
            f"`{c['n_meta_cols']}` column descriptions | "
            f"[`registry/ddl.py`]({src_blob}/registry/ddl.py), "
            f"[`registry/views.py`]({src_blob}/registry/views.py), "
            f"[`registry/metadata.py`]({src_blob}/registry/metadata.py) |"
        ),
        (
            "| **Opt-In Knowledge Catalog & `DataAgent`** | "
            f"**{c['n_glossary']} Business Glossary terms** · `DataProfile` scans · "
            f"**{c['n_golden']} Golden SQL queries** · **5 chat surfaces** | "
            f"[`bq_agent/`]({_REPO_URL}/tree/main/src/scale_forecasting/bq_agent) |"
        ),
        (
            "| **Forecaster Workbench & Solvers** | **6 Workbench pillars** · **7-Question "
            "`assess()`** · **Single-SKU `ROP`/`SS` + 7 SciPy/HiGHS LP & MILP optimizers** · "
            "**12-tab HTML/JSON report** | "
            f"[`workbench.py`]({src_blob}/workbench.py), "
            f"[`inventory_optimizer.py`]({src_blob}/inventory_optimizer.py), "
            f"[`clearance_optimizer.py`]({src_blob}/clearance_optimizer.py) |"
        ),
        (
            "| **AI Agent & MCP Ecosystem** | **6 generated agent files** · **1 portable "
            f"`SKILL.md`** · **MCP server (`{c['n_mcp_res']}` resources, `{c['n_mcp_tools']}` "
            "tools)** across 5 IDE/CLI clients | "
            f"[`agent_surfaces.py`]({src_blob}/agent_surfaces.py), "
            f"[`mcp.py`]({src_blob}/mcp.py) |"
        ),
        (
            "| **Infrastructure, Notebooks & Validation** | "
            f"**{c['n_tf_mods']} Terraform modules** "
            f"· **{c['n_extras']} packaging extras** · **{c['n_notebooks']} notebooks** · "
            f"**{c['n_smokes']} smoke + {c['n_demos']} demo configs** | "
            f"[`terraform/`]({_REPO_URL}/tree/main/terraform), "
            f"[`notebooks/`]({_REPO_URL}/tree/main/notebooks), "
            f"[`configs/`]({_REPO_URL}/tree/main/configs) |"
        ),
        _VISUAL_INVENTORY_END,
    ]
    return "\n".join(rows)


def sync_visual_architecture_text(text: str, repo_root: Path | None = None) -> str:
    """Sync the live inventory table and Mermaid counts in ``docs/visual_architecture.md``."""
    c = _visual_inventory_counts(repo_root)
    inv_block = render_visual_architecture_inventory(repo_root)
    updated = re.sub(
        rf"{re.escape(_VISUAL_INVENTORY_BEGIN)}[\s\S]*?{re.escape(_VISUAL_INVENTORY_END)}",
        lambda _: inv_block,
        text,
        count=1,
    )
    subs: tuple[tuple[str, str], ...] = (
        (r"\(\d+ Models Total\)", f"({c['n_models']} Models Total)"),
        (r"statistical \(\d+\)", f"statistical ({c['n_stat']})"),
        (r"ml \(\d+\)", f"ml ({c['n_ml']})"),
        (r"deep_learning \(\d+\)", f"deep_learning ({c['n_dl']})"),
        (r"automl \(\d+\)", f"automl ({c['n_automl']})"),
        (r"native \(\d+\)", f"native ({c['n_native']})"),
        (r"Across \d+ Metrics", f"Across {c['n_metrics']} Metrics"),
        (r"\d+ Ensemble Strategies", f"{c['n_strategies']} Ensemble Strategies"),
        (
            r"\(\d+ Strategies in ensembler\.py\)",
            f"({c['n_strategies']} Strategies in ensembler.py)",
        ),
        (r"\d+ Reconciliation Methods", f"{c['n_reconcile']} Reconciliation Methods"),
        (r"\d+ Composable @tag Modifiers", f"{c['n_variant_tags']} Composable @tag Modifiers"),
        (r"All \d+ Registered Metrics", f"All {c['n_metrics']} Registered Metrics"),
        (
            r"\d+ Core Operational & Leaderboard Views",
            f"{c['n_core_views']} Core Operational & Leaderboard Views",
        ),
        (
            r"\d+ Diagnostic & Executive S&OP Views",
            f"{c['n_diag_views']} Diagnostic & Executive S&OP Views",
        ),
        (r"\d+ Domain Terms", f"{c['n_glossary']} Domain Terms"),
        (
            r"\d+ Parameterized Golden SQL Queries",
            f"{c['n_golden']} Parameterized Golden SQL Queries",
        ),
        (r"MCP Tool #\d+", f"MCP Tool #{c['n_mcp_tools']}"),
        (
            r"\d+ Resources, \d+ Tools",
            f"{c['n_mcp_res']} Resources, {c['n_mcp_tools']} Tools",
        ),
        (r"\d+ Terraform Modules", f"{c['n_tf_mods']} Terraform Modules"),
        (r"\d+ Packaging Extras", f"{c['n_extras']} Packaging Extras"),
        (r"\d+-Smoke Live Suite", f"{c['n_smokes']}-Smoke Live Suite"),
    )
    for pattern, repl in subs:
        updated = re.sub(pattern, repl, updated)
    return updated


def expected_agent_surface_files(repo_root: Path | None = None) -> dict[Path, str]:
    """Return ``{path: expected_content}`` for all 6 code-generated agent files."""
    root = repo_root or _default_repo_root()
    refs_dir = root / "skills" / "scale-forecasting" / "references"
    return {
        root / "docs" / "schemas" / "run_config.schema.json": render_run_config_json_schema(),
        root / "docs" / "llms.txt": render_llms_txt(),
        root / "docs" / "llms-full.txt": render_llms_full_txt(root),
        refs_dir / "config_reference.md": render_skill_config_reference(),
        refs_dir / "catalog_reference.md": render_skill_catalog_reference(root),
        refs_dir / "execution_paths_and_ops.md": render_skill_execution_paths_reference(),
    }


def write_agent_surfaces(repo_root: Path | None = None) -> list[Path]:
    """Write all generated agent surface files to disk and sync ``docs/visual_architecture.md``."""
    root = repo_root or _default_repo_root()
    expected = expected_agent_surface_files(root)
    written: list[Path] = []
    for path, content in expected.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        written.append(path)
    visual_path = root / "docs" / "visual_architecture.md"
    if visual_path.is_file():
        raw = visual_path.read_text(encoding="utf-8")
        synced = sync_visual_architecture_text(raw, root)
        if synced != raw:
            visual_path.write_text(synced, encoding="utf-8")
    return written


def check_agent_surfaces(repo_root: Path | None = None) -> list[str]:
    """Return a list of drift errors if any generated agent surface file is missing or stale."""
    root = repo_root or _default_repo_root()
    expected = expected_agent_surface_files(root)
    drifted: list[str] = []
    for path, content in expected.items():
        rel = path.relative_to(root)
        if not path.is_file():
            drifted.append(f"{rel}: missing on disk (run `make agent-surfaces`)")
            continue
        actual = path.read_text(encoding="utf-8")
        if actual != content:
            drifted.append(
                f"{rel}: out of sync with canonical code declarations (run `make agent-surfaces`)"
            )
    visual_path = root / "docs" / "visual_architecture.md"
    if visual_path.is_file():
        actual_visual = visual_path.read_text(encoding="utf-8")
        if (
            _VISUAL_INVENTORY_BEGIN not in actual_visual
            or _VISUAL_INVENTORY_END not in actual_visual
            or actual_visual != sync_visual_architecture_text(actual_visual, root)
        ):
            drifted.append(
                "docs/visual_architecture.md: out of sync with canonical code declarations "
                "(run `make agent-surfaces`)"
            )
    return drifted


_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:[a-zA-Z0-9.\-+]+)?$")


def bump_version(
    new_version: str,
    repo_root: Path | None = None,
    *,
    release_date: str | None = None,
) -> list[Path]:
    """Atomically update the package version across all version-locked repository files.

    Updates:
    1. ``pyproject.toml`` (``[project].version``)
    2. ``src/scale_forecasting/__init__.py`` (``__version__`` fallback)
    3. ``plugin.json`` (``"version"``)
    4. ``skills/scale-forecasting/SKILL.md`` (``metadata.version``)
    5. ``CITATION.cff`` (``version`` and ``date-released``)
    6. ``CHANGELOG.md`` (inserts ``## [<version>] - <date>`` section if not already present)
    7. ``docker/cloudbuild.yaml`` (``_VERSION_TAG: "v<version>"`` when present)
    8. All 6 generated agent surface files via `write_agent_surfaces` when ``mkdocs.yml`` exists.

    Returns the list of modified file paths.
    """
    ver = new_version.strip().lstrip("v")
    if not _SEMVER_RE.match(ver):
        raise ValueError(f"Invalid semantic version {new_version!r}; expected 'MAJOR.MINOR.PATCH'.")
    root = repo_root or _default_repo_root()
    from datetime import UTC, datetime

    rel_date = release_date or datetime.now(UTC).strftime("%Y-%m-%d")
    updated: list[Path] = []

    pyproject_path = root / "pyproject.toml"
    if pyproject_path.is_file():
        text = pyproject_path.read_text(encoding="utf-8")
        new_text, n = re.subn(
            r'(?m)^version\s*=\s*"[^"]+"',
            f'version = "{ver}"',
            text,
            count=1,
        )
        if n:
            pyproject_path.write_text(new_text, encoding="utf-8")
            updated.append(pyproject_path)

    init_path = root / "src" / "scale_forecasting" / "__init__.py"
    if init_path.is_file():
        text = init_path.read_text(encoding="utf-8")
        new_text, n = re.subn(
            r'(?m)^(\s*__version__\s*=\s*)"[^"]+"',
            rf'\1"{ver}"',
            text,
            count=1,
        )
        if n:
            init_path.write_text(new_text, encoding="utf-8")
            updated.append(init_path)

    plugin_path = root / "plugin.json"
    if plugin_path.is_file():
        plugin_data = json.loads(plugin_path.read_text(encoding="utf-8"))
        plugin_data["version"] = ver
        plugin_path.write_text(json.dumps(plugin_data, indent=2) + "\n", encoding="utf-8")
        updated.append(plugin_path)

    skill_path = root / "skills" / "scale-forecasting" / "SKILL.md"
    if skill_path.is_file():
        text = skill_path.read_text(encoding="utf-8")
        new_text, n = re.subn(
            r"(?m)^(\s*version:\s*)\S+",
            rf"\g<1>{ver}",
            text,
            count=1,
        )
        if n:
            skill_path.write_text(new_text, encoding="utf-8")
            updated.append(skill_path)

    citation_path = root / "CITATION.cff"
    if citation_path.is_file():
        text = citation_path.read_text(encoding="utf-8")
        new_text = re.sub(
            r'(?m)^version:\s*"[^"]+"',
            f'version: "{ver}"',
            text,
            count=1,
        )
        new_text = re.sub(
            r'(?m)^date-released:\s*"[^"]+"',
            f'date-released: "{rel_date}"',
            new_text,
            count=1,
        )
        citation_path.write_text(new_text, encoding="utf-8")
        updated.append(citation_path)

    changelog_path = root / "CHANGELOG.md"
    if changelog_path.is_file():
        text = changelog_path.read_text(encoding="utf-8")
        if f"## [{ver}]" not in text:
            stub = (
                f"## [Unreleased]\n\n---\n\n"
                f"## [{ver}] - {rel_date}\n\n"
                f"### Changed\n\n"
                f"- Release `{ver}`.\n\n---"
            )
            if "## [Unreleased]\n\n---" in text:
                new_text = text.replace("## [Unreleased]\n\n---", stub, 1)
            else:
                new_text = text + f"\n## [{ver}] - {rel_date}\n"
            changelog_path.write_text(new_text, encoding="utf-8")
            updated.append(changelog_path)

    cloudbuild_path = root / "docker" / "cloudbuild.yaml"
    if cloudbuild_path.is_file():
        text = cloudbuild_path.read_text(encoding="utf-8")
        new_text, n = re.subn(
            r'(?m)^(\s*_VERSION_TAG:\s*)"[^"]+"',
            rf'\1"v{ver}"',
            text,
            count=1,
        )
        if n:
            cloudbuild_path.write_text(new_text, encoding="utf-8")
            updated.append(cloudbuild_path)

    if (root / "mkdocs.yml").is_file():
        updated.extend(write_agent_surfaces(root))

    return updated


def render_release_notes(version: str, repo_root: Path | None = None) -> str:
    """Render human-first GitHub Release notes for ``version`` from ``CHANGELOG.md``."""
    ver = version.strip().lstrip("v")
    root = repo_root or _default_repo_root()
    changelog_path = root / "CHANGELOG.md"
    section_body = ""
    if changelog_path.is_file():
        text = changelog_path.read_text(encoding="utf-8")
        pattern = re.compile(
            rf"(?ms)^## \[{re.escape(ver)}\][^\n]*\n(.*?)(?=^## \[|\Z)",
        )
        m = pattern.search(text)
        if m:
            section_body = m.group(1).strip().rstrip("-").strip()

    if not section_body:
        section_body = f"Release `v{ver}` of `scale-forecasting`."

    # Extract top-level bullet headlines for "Highlights at a Glance"
    highlights: list[str] = []
    for line in section_body.splitlines():
        if line.startswith("- **"):
            highlights.append(line)

    hl_block = (
        "\n".join(highlights[:6]) if highlights else f"- Release `v{ver}` of `scale-forecasting`."
    )

    lines = [
        f"# `scale-forecasting` v{ver}",
        "",
        "## Highlights at a Glance",
        "",
        hl_block,
        "",
        "## Installation & Upgrade",
        "",
        "```bash",
        (
            "# Core offline package (playground, 14 statistical/ML models, "
            "21 metrics, Auto-Forecaster, MCP server)"
        ),
        f'pip install --upgrade "scale-forecasting=={ver}"',
        "",
        "# With Google Cloud runtimes (BigQuery, Dataproc Spark, Vertex AI, Ray, GCE, GKE)",
        f'pip install --upgrade "scale-forecasting[gcp]=={ver}"',
        "",
        "# Complete installation (all 37 models, all 7 cloud runtimes, notebooks)",
        f'pip install --upgrade "scale-forecasting[all]=={ver}"',
        "```",
        "",
        "---",
        "",
        f"## Detailed Release Notes (`v{ver}`)",
        "",
        section_body,
        "",
        "---",
        "",
        "## Documentation & Quick Links",
        "",
        f"- **Documentation Site:** {_SITE_BASE}/",
        f"- **Auto-Forecaster & Bounded `autofocus`:** {_SITE_BASE}/auto_forecaster/",
        f"- **Forecaster's Workbench:** {_SITE_BASE}/forecaster_workbench/",
        f"- **AI Agents, Skills & MCP Guide:** {_SITE_BASE}/agent_and_mcp_guide/",
        f"- **Live Cloud Validation Ledger:** {_SITE_BASE}/validation/",
        f"- **Full Changelog:** {_REPO_URL}/blob/main/CHANGELOG.md",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for generating, checking, or probing agent surfaces."""
    configure_cli_logging()
    parser = argparse.ArgumentParser(
        prog="scale_forecasting.agent_surfaces",
        description="Generate, verify, or probe scale-forecasting AI agent surfaces.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--write",
        action="store_true",
        help="Regenerate JSON Schema, llms.txt, llms-full.txt, and skill reference tables.",
    )
    group.add_argument(
        "--check",
        action="store_true",
        help="Verify all generated agent surface files match code declarations (exit 1 on drift).",
    )
    group.add_argument(
        "--probe-env",
        action="store_true",
        help="Print a JSON report of installed extras, available models, and SF_* cloud readiness.",
    )
    group.add_argument(
        "--bump-version",
        metavar="X.Y.Z",
        help="Atomically bump package version across all version-locked files and agent surfaces.",
    )
    group.add_argument(
        "--release-notes",
        metavar="X.Y.Z",
        help="Render human-readable GitHub Release notes for version X.Y.Z from CHANGELOG.md.",
    )
    args = parser.parse_args(argv)

    if args.probe_env:
        print(json.dumps(probe_environment(), indent=2, sort_keys=True))
        return 0

    if args.bump_version:
        updated = bump_version(args.bump_version)
        for path in updated:
            print(f"updated {path}")
        return 0

    if args.release_notes:
        print(render_release_notes(args.release_notes), end="")
        return 0

    if args.write:
        written = write_agent_surfaces()
        for path in written:
            print(f"wrote {path}")
        return 0

    drifted = check_agent_surfaces()
    if drifted:
        for msg in drifted:
            print(f"DRIFT: {msg}", file=sys.stderr)
        return 1
    print("All agent surface files are in sync.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
