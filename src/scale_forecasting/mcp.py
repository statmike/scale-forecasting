"""Built-in Model Context Protocol (MCP) server for `scale-forecasting` over `stdio`.

Exposes the platform's configuration schema, canonical catalogs (37 models, 21 metrics, 7 runtimes,
4 GPU types, 13 extras, 11 views, 62 shipped configs), environment readiness probe, offline model
playground, `RunConfig` validator and DAG planner, Airflow DAG emitter, and live BigQuery registry
inspection/review/repair/conversational analytics tools to any MCP-compatible AI coding agent
(Google Antigravity `agy` CLI & IDE, Claude Code, Cursor, VS Code GitHub Copilot, Windsurf, OpenAI
Codex CLI).

Zero extra dependencies are required: the server runs on a bare ``pip install scale-forecasting``
core installation using the Python standard library and Pydantic. Tools that query or mutate Google
Cloud (`inspect_registry`, `review_run`, `launch_or_repair_run`, and cloud modes of
`plan_execution`) check for ``[gcp]`` and ``SF_*`` environment variables first and return structured
remediation steps if either is absent. Cloud job submission and mutating repair verbs are gated
behind the explicit ``--allow-launch`` CLI flag.

CLI usage::

    # Start stdio MCP server (read/plan/playground enabled; cloud launches safety-locked)
    python -m scale_forecasting.mcp

    # Start stdio MCP server with cloud job launch & repair unlocked
    python -m scale_forecasting.mcp --allow-launch

    # Print environment readiness JSON and exit
    python -m scale_forecasting.mcp --probe-env
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import logging
import sys
from collections.abc import Sequence
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any, BinaryIO

from .agent_surfaces import (
    ALL_EXTRA_MODULES,
    EXTRA_DESCRIPTIONS,
    GPU_CATALOG,
    RUNTIME_CATALOG,
    build_run_config_json_schema,
    probe_environment,
    render_run_config_json_schema,
)
from .auto_policy import EXPOSURE_TIERS, SCALE_REGIMES
from .benchmarks import benchmark_catalog
from .config import RunConfig, estimate_workload, load_config
from .dag import dag_nodes, group_models_by_family, plan_dag
from .errors import PACKAGE_LOGGER, MissingExtraError, ScaleForecastError
from .metrics import METRIC_NAMES, get_metric
from .models import get_model, list_models
from .registry.ddl import REGISTRY_TABLE_NAMES, SOURCE_TABLE_NAMES
from .registry.ids import make_run_id
from .registry.metadata import TABLE_DESCRIPTIONS, VIEW_DESCRIPTIONS
from .registry.reads import RECENT_RUNS_COLUMNS
from .registry.views import CORE_VIEW_NAMES, DIAGNOSTIC_VIEW_NAMES, VIEW_NAMES

__all__ = [
    "MCP_PROTOCOL_VERSION",
    "McpServer",
    "compute_time_series_similarity",
    "main",
]

MCP_PROTOCOL_VERSION = "2024-11-05"


def _json_default(obj: Any) -> Any:
    """Serialize dataclasses, dates/timestamps, paths, sets, and tuples cleanly to JSON."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, datetime.datetime | datetime.date):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set | frozenset | tuple):
        return list(obj)
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return str(obj)


def _dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=_json_default)


def _parse_run_config_input(raw_config: Any, repo_root: Path) -> RunConfig:
    """Load a `RunConfig` from a dict, a JSON string, or a file path string."""
    if isinstance(raw_config, dict):
        return RunConfig.model_validate(raw_config)
    if isinstance(raw_config, str):
        stripped = raw_config.strip()
        if stripped.startswith("{"):
            return RunConfig.model_validate_json(stripped)
        candidate = Path(stripped)
        if not candidate.is_file():
            candidate = repo_root / stripped
        return load_config(candidate)
    raise ValueError("config must be a JSON object (dict), JSON string, or path to a .json file")


def _build_models_catalog() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in sorted(list_models()):
        cls = get_model(name)
        extra = cls.optional_extra or ("gcp" if cls.runtime == "bigquery" else "core")
        rows.append(
            {
                "model": name,
                "family": cls.family,
                "runtime": cls.runtime,
                "package": cls.package,
                "package_url": cls.package_url,
                "optional_extra": extra,
                "install_command": (
                    "pip install scale-forecasting"
                    if extra == "core"
                    else f'pip install "scale-forecasting[{extra}]"'
                ),
                "available": cls.is_available(),
                "gpu_capable": cls.gpu_capable,
                "supports_future_covariates": cls.supports_future_covariates,
                "supports_past_covariates": cls.supports_past_covariates,
                "supports_static_covariates": cls.supports_static_covariates,
                "supports_explainability": cls.supports_explainability,
                "supports_monotonic_constraints": cls.supports_monotonic_constraints,
                "supports_global": cls.supports_global,
                "supports_hybrid": cls.supports_hybrid,
            }
        )
    return rows


def _build_metrics_catalog() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in METRIC_NAMES:
        cls = get_metric(name)
        rows.append(
            {
                "metric": name,
                "kind": "interval" if cls.needs_intervals else "point",
                "direction": cls.direction,
                "needs_intervals": cls.needs_intervals,
                "needs_train_history": cls.needs_train_history,
                "needs_seasonal_period": cls.needs_seasonal_period,
                "mean_optimal": cls.mean_optimal,
            }
        )
    return rows


def _build_configs_catalog(repo_root: Path) -> dict[str, list[dict[str, Any]]]:
    demo_rows: list[dict[str, Any]] = []
    smoke_rows: list[dict[str, Any]] = []
    demo_dir = repo_root / "configs"
    if demo_dir.is_dir():
        for p in sorted(demo_dir.glob("*.json")):
            if p.name == "compute_fallback.json":
                continue
            cfg = RunConfig.model_validate_json(p.read_text(encoding="utf-8"))
            demo_rows.append(
                {
                    "path": f"configs/{p.name}",
                    "run_name": cfg.run_name,
                    "run_id": make_run_id(cfg),
                    "python_runtime": cfg.python_runtime,
                    "models": list(cfg.models),
                    "backtest_enabled": cfg.backtest.enabled,
                    "ensemble_enabled": cfg.ensemble.enabled,
                }
            )
    smoke_dir = repo_root / "configs" / "smokes"
    if smoke_dir.is_dir():
        for p in sorted(smoke_dir.glob("*.json")):
            cfg = RunConfig.model_validate_json(p.read_text(encoding="utf-8"))
            smoke_rows.append(
                {
                    "path": f"configs/smokes/{p.name}",
                    "run_name": cfg.run_name,
                    "run_id": make_run_id(cfg),
                    "python_runtime": cfg.python_runtime,
                    "models": list(cfg.models),
                    "family_overrides": {
                        fam: fc.runtime or cfg.python_runtime
                        for fam, fc in sorted(cfg.compute.families.items())
                    },
                }
            )
    return {"demo_configs": demo_rows, "smoke_configs": smoke_rows}


def _build_views_catalog() -> dict[str, Any]:
    return {
        "source_tables": [
            *SOURCE_TABLE_NAMES,
            "source_series_wide_native",
            "source_series_hierarchical_native",
        ],
        "registry_tables": list(REGISTRY_TABLE_NAMES),
        "analytical_views": list(VIEW_NAMES),
        "core_views": list(CORE_VIEW_NAMES),
        "diagnostic_views": list(DIAGNOSTIC_VIEW_NAMES),
        "table_descriptions": dict(TABLE_DESCRIPTIONS),
        "view_descriptions": dict(VIEW_DESCRIPTIONS),
        "recent_runs_columns": list(RECENT_RUNS_COLUMNS),
    }


class McpServer:
    """JSON-RPC 2.0 Model Context Protocol server for `scale-forecasting`."""

    def __init__(self, *, allow_launch: bool = False, repo_root: Path | None = None) -> None:
        self.allow_launch = allow_launch
        self.repo_root = repo_root or Path(__file__).resolve().parents[2]

    def list_resources(self) -> list[dict[str, str]]:
        """Return the 7 MCP resources exposed by this server."""
        return [
            {
                "uri": "forecast://environment",
                "name": "Active Environment & Readiness Report",
                "description": (
                    "Installed extras (13), available vs. missing models (37), SF_* environment "
                    "variables, ADC status, and readiness across all 5 execution paths."
                ),
                "mimeType": "application/json",
            },
            {
                "uri": "forecast://schema/run-config",
                "name": "RunConfig JSON Schema (Draft 2020-12)",
                "description": "Complete JSON Schema for RunConfig and all nested blocks.",
                "mimeType": "application/schema+json",
            },
            {
                "uri": "forecast://catalog/models",
                "name": "Forecasting Models Catalog (37 Models)",
                "description": (
                    "All 37 registered models across 5 families with runtime, package, extra, "
                    "covariate flags, explainability, and local availability."
                ),
                "mimeType": "application/json",
            },
            {
                "uri": "forecast://catalog/metrics",
                "name": "Evaluation Metrics Catalog (21 Metrics)",
                "description": (
                    "All 16 point and 5 interval evaluation metrics with direction and "
                    "requirements."
                ),
                "mimeType": "application/json",
            },
            {
                "uri": "forecast://catalog/runtimes",
                "name": "Compute Runtimes & GPU Catalog (7 Runtimes, 4 GPUs)",
                "description": (
                    "All 7 Google Cloud runtimes, sub-modes, scaling models, and 4 GPU types."
                ),
                "mimeType": "application/json",
            },
            {
                "uri": "forecast://catalog/configs",
                "name": "Shipped Demo & Smoke Configurations (62 Configs)",
                "description": (
                    "All 20 root demo configs and 42 smoke configs with run_ids and models."
                ),
                "mimeType": "application/json",
            },
            {
                "uri": "forecast://catalog/views",
                "name": "BigQuery Source Tables, Registry Tables & Views",
                "description": (
                    "The 4 source tables, 5 registry tables, and 11 analytical SQL views."
                ),
                "mimeType": "application/json",
            },
        ]

    def read_resource(self, uri: str) -> dict[str, Any]:
        """Read one MCP resource by URI."""
        if uri == "forecast://environment":
            text = _dumps(probe_environment(self.repo_root))
            mime = "application/json"
        elif uri == "forecast://schema/run-config":
            text = render_run_config_json_schema()
            mime = "application/schema+json"
        elif uri == "forecast://catalog/models":
            text = _dumps(_build_models_catalog())
            mime = "application/json"
        elif uri == "forecast://catalog/metrics":
            text = _dumps(_build_metrics_catalog())
            mime = "application/json"
        elif uri == "forecast://catalog/runtimes":
            text = _dumps({"runtimes": list(RUNTIME_CATALOG), "gpus": list(GPU_CATALOG)})
            mime = "application/json"
        elif uri == "forecast://catalog/configs":
            text = _dumps(_build_configs_catalog(self.repo_root))
            mime = "application/json"
        elif uri == "forecast://catalog/views":
            text = _dumps(_build_views_catalog())
            mime = "application/json"
        else:
            raise ValueError(f"Unknown resource URI: {uri!r}")

        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": mime,
                    "text": text,
                }
            ]
        }

    def list_tools(self) -> list[dict[str, Any]]:
        """Return the 14 MCP tools exposed by this server."""
        from .benchmarks import list_benchmarks

        return [
            {
                "name": "probe_environment",
                "description": (
                    "Inspect the active Python environment: installed optional extras (out of 13), "
                    "locally available vs. missing-extra models (out of 37), SF_* environment "
                    "variables, .env.infra presence, ADC credentials, and readiness across all 5 "
                    "execution paths. Call this first to adapt recommendations to the machine."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
            {
                "name": "list_catalog",
                "description": (
                    "Query the canonical catalogs of models (37), metrics (21), runtimes (7), "
                    "GPUs (4), dependency extras (13), BigQuery views (11), shipped configs (62), "
                    "synthetic DGP benchmarks (20), or Auto-Forecaster exposure tiers & scale "
                    "regimes."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": [
                                "all",
                                "models",
                                "metrics",
                                "runtimes",
                                "gpus",
                                "extras",
                                "views",
                                "configs",
                                "benchmarks",
                                "auto_tiers",
                            ],
                            "default": "all",
                        },
                        "family": {
                            "type": "string",
                            "description": (
                                "Optional model family filter "
                                "(statistical, ml, deep_learning, automl, native)."
                            ),
                        },
                        "available_only": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "When true, filter models to those installed in the active "
                                "environment."
                            ),
                        },
                    },
                    "additionalProperties": False,
                },
            },
            {
                "name": "inspect_config_schema",
                "description": (
                    "Inspect the JSON Schema and validation rules for RunConfig or a specific "
                    "nested config block (DataConfig, FeaturesConfig, BacktestConfig, "
                    "OutputConfig, CalibrationConfig, HpoConfig, EnsembleConfig, HierarchyConfig, "
                    "ComputeConfig, FamilyCompute, EnsembleCompute, ProfileConfig, CapacityConfig, "
                    "CapacityServicePolicy, RetryResources)."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "section": {
                            "type": "string",
                            "description": (
                                "Optional section or Pydantic class name to inspect (e.g., "
                                "'compute', 'backtest', 'CalibrationConfig', 'FamilyCompute'). "
                                "Omit for overview."
                            ),
                        }
                    },
                    "additionalProperties": False,
                },
            },
            {
                "name": "validate_and_dry_run",
                "description": (
                    "Validate a RunConfig (dict, JSON string, or file path) offline, compute its "
                    "deterministic content-addressed run_id, estimate cell/fit workload, resolve "
                    "per-family compute placement and DAG nodes, and check whether any requested "
                    "models require an uninstalled optional extra."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "config": {
                            "description": (
                                "RunConfig dict, JSON string, or relative/absolute file path."
                            ),
                        },
                        "n_series": {
                            "type": "integer",
                            "description": "Optional series_limit override.",
                        },
                        "ignore_unavailable_models": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "Filter models to those installed locally before planning."
                            ),
                        },
                    },
                    "required": ["config"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "run_playground",
                "description": (
                    "Run a real forecasting model offline on deterministic synthetic time-series "
                    "panel data (`worker.run_cell`), optionally with expanding-window backtesting, "
                    "and return per-series evaluation metrics and sample forecast rows."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "model": {
                            "type": "string",
                            "default": "holtwinters",
                            "description": (
                                "Registered Python model name "
                                "(e.g., 'holtwinters', 'theta', 'naive_seasonal')."
                            ),
                        },
                        "models": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Optional list of model names for playground or bake-off runs."
                            ),
                        },
                        "horizon": {
                            "type": "integer",
                            "default": 14,
                        },
                        "n_series": {
                            "type": "integer",
                            "default": 3,
                        },
                        "backtest": {
                            "type": "boolean",
                            "default": True,
                        },
                        "n_folds": {
                            "type": "integer",
                            "default": 3,
                        },
                        "include_fva": {
                            "type": "boolean",
                            "default": False,
                            "description": "When true, include Forecast Value Added (FVA) summary.",
                        },
                        "scenarios": {
                            "type": "object",
                            "description": (
                                "Optional counterfactual covariate shock map "
                                "(e.g., {'promo_week': {'price_index': '*0.85'}})."
                            ),
                        },
                        "price_col": {
                            "type": "string",
                            "description": (
                                "Optional price column name for revenue impact simulation."
                            ),
                        },
                        "unit_cost": {
                            "type": ["number", "string"],
                            "description": (
                                "Optional unit cost (COGS) scalar or column name for gross-margin "
                                "counterfactual simulation."
                            ),
                        },
                        "include_inventory_impact": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "When true, compute operational inventory policy shifts "
                                "(delta_reorder_point, delta_safety_stock, delta_order_qty) for "
                                "each scenario."
                            ),
                        },
                        "include_promo_elasticity": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "When true, compute cross-fitted Double-ML promo & price "
                                "elasticity decomposition on the series."
                            ),
                        },
                        "include_inventory_policy": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "When true, compute operational (s, S) / (R, S) inventory & "
                                "replenishment policy (ROP, safety_stock, order_up_to_level, "
                                "recommended_order_qty, Newsvendor cost) from the forward "
                                "forecast and backtest residuals."
                            ),
                        },
                        "inventory_params": {
                            "type": "object",
                            "description": (
                                "Optional keyword overrides for compute_inventory_policy "
                                "(lead_time, lead_time_std, review_period, service_level, "
                                "fill_rate, on_hand, on_order, backorders, holding_cost, "
                                "stockout_cost, min_order_qty, order_multiple)."
                            ),
                        },
                        "include_goal_seek": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "When true, solve top-down constrained QP goal-seek target "
                                "allocation on the forward forecast."
                            ),
                        },
                        "target_total": {
                            "description": (
                                "Optional executive top-down target total (scalar or date map) "
                                "for goal-seek allocation."
                            ),
                        },
                        "reconcile_temporal": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "When true, apply THieF temporal hierarchy reconciliation across "
                                "aggregated calendar horizons."
                            ),
                        },
                        "temporal_reconciliation_method": {
                            "type": "string",
                            "enum": [
                                "ols",
                                "wls_struct",
                                "wls_var",
                                "mint_shrink",
                                "mint_nonneg",
                                "cross_temporal_mint",
                            ],
                            "default": "wls_struct",
                            "description": (
                                "Temporal reconciliation estimator when reconcile_temporal=True."
                            ),
                        },
                        "include_cold_start": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "When true, compute cold-start & NPI analog donor shrinkage or "
                                "Bass/Gompertz diffusion forecast."
                            ),
                        },
                        "cold_start_archetype": {
                            "type": "string",
                            "enum": [
                                "auto",
                                "analog_transfer",
                                "bass_diffusion",
                                "weibull_lifecycle",
                                "arps_decline",
                            ],
                            "default": "auto",
                            "description": "Cold-start NPI launch archetype.",
                        },
                        "include_aci": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "When true, compute online Adaptive Conformal Inference (ACI) "
                                "intervals and rolling empirical coverage tracking."
                            ),
                        },
                        "include_trajectories": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "When true, simulate joint multi-horizon copula sample "
                                "trajectories."
                            ),
                        },
                        "include_tail_risk": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "When true, compute portfolio Value-at-Risk (VaR) and Expected "
                                "Shortfall (CVaR) from joint copula trajectories."
                            ),
                        },
                        "copula_type": {
                            "type": "string",
                            "enum": [
                                "auto",
                                "gaussian",
                                "student_t",
                                "iman_conover",
                                "empirical",
                            ],
                            "default": "gaussian",
                            "description": "Copula family for joint trajectory sampling.",
                        },
                        "n_trajectories": {
                            "type": "integer",
                            "default": 50,
                            "description": "Number of joint copula trajectories to draw.",
                        },
                        "optimize_inventory_params": {
                            "type": "object",
                            "description": (
                                "Optional keyword arguments for portfolio-constrained inventory "
                                "optimization (max_budget, max_volume, max_total_units, "
                                "target_portfolio_fill_rate, vendor_moq, uncertainty_source)."
                            ),
                        },
                        "optimize_scenario_params": {
                            "type": "object",
                            "description": (
                                "Optional keyword arguments for prescriptive price & promo "
                                "calendar optimization (objective, price_bounds, "
                                "price_grid_steps, max_promo_periods, min_margin_pct)."
                            ),
                        },
                        "export_report_path": {
                            "type": "string",
                            "description": (
                                "Optional file path (.html) to export a self-contained "
                                "interactive HTML report."
                            ),
                        },
                        "include_similarity": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "When true, compute pairwise time-series similarity, digital-twin "
                                "pairs, and behavioral clusters across the playground panel."
                            ),
                        },
                        "similarity_method": {
                            "type": "string",
                            "default": "hybrid",
                            "description": "Similarity kernel lens when include_similarity=true.",
                        },
                        "include_substitutes": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "When true, compute inventory-aware SKU substitute / "
                                "cannibalization recommendations (recommend_substitutes_df)."
                            ),
                        },
                        "include_emerging_trends": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "When true, compute breakout velocity, acceleration, and "
                                "bellwether merchandising actions (detect_emerging_trends_df)."
                            ),
                        },
                        "text_cols": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional text columns for multi-modal similarity.",
                        },
                        "attribute_cols": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional attribute columns for multi-modal similarity.",
                        },
                        "hierarchy_cols": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional hierarchy columns for multi-modal similarity.",
                        },
                        "within_hierarchy_level": {
                            "type": "string",
                            "description": (
                                "Optional hierarchy column restricting substitute search."
                            ),
                        },
                        "inventory_records": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": (
                                "Optional inventory records for recommend_substitutes_df."
                            ),
                        },
                        "include_clearance": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "When true, solve lifecycle markdown & perishable clearance MILP "
                                "(optimize_clearance_df)."
                            ),
                        },
                        "optimize_clearance_params": {
                            "type": "object",
                            "description": (
                                "Optional keyword arguments for optimize_clearance "
                                "(initial_inventory, expiry_step, target_space_freed, "
                                "reclaim_by_step, salvage_value, target_sell_through_pct, "
                                "covariate_cols, hierarchy_cols)."
                            ),
                        },
                        "include_contract_volume": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "When true, solve vendor contract volume & tiered rebate / "
                                "Take-or-Pay MILP (optimize_contract_volume_df)."
                            ),
                        },
                        "optimize_contract_volume_params": {
                            "type": "object",
                            "description": (
                                "Optional keyword arguments for optimize_contract_volume "
                                "(contract_tiers, min_contract_volume, shortfall_penalty_per_unit, "
                                "risk_aversion_lambda, cvar_alpha, covariate_cols, hierarchy_cols)."
                            ),
                        },
                    },
                    "additionalProperties": False,
                },
            },
            {
                "name": "plan_execution",
                "description": (
                    "Plan or emit execution artifacts across the platform's execution paths: "
                    "'dry_run' (offline plan + CLI/gcloud launch commands + idempotency check when "
                    "SF_* is configured), 'emit_airflow' (render Cloud Composer 3 / Airflow DAG "
                    "Python source offline), 'feasibility' (query live BigQuery series lengths vs. "
                    "backtest geometry), 'quota' (query regional CPU/GPU quota meters), or "
                    "'stage_only' (upload config + code archive to GCS; requires --allow-launch)."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "config": {
                            "description": "RunConfig dict, JSON string, or file path.",
                        },
                        "mode": {
                            "type": "string",
                            "enum": [
                                "dry_run",
                                "emit_airflow",
                                "feasibility",
                                "quota",
                                "stage_only",
                            ],
                            "default": "dry_run",
                        },
                        "force": {
                            "type": "boolean",
                            "default": False,
                        },
                        "with_retry": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "For mode='emit_airflow': include the surgical retry node before "
                                "barrier ensembling."
                            ),
                        },
                        "emit_out": {
                            "type": "string",
                            "description": (
                                "Optional file path to write the rendered Airflow DAG when "
                                "mode='emit_airflow'."
                            ),
                        },
                    },
                    "required": ["config"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "inspect_registry",
                "description": (
                    "Read-only inspection of the live BigQuery run registry (`[gcp]` + `SF_*`): "
                    "'doctor' (table row counts, stuck RUNNING runs, orphaned GCS artifacts), "
                    "'recent_runs' (latest runs from v_run_summary with time ledger and "
                    "overhead_seconds), 'probe_run' (reconcile live runtime status for a run_id), "
                    "or 'retry_preview' (preview surgical cell-repair decision table)."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["doctor", "recent_runs", "probe_run", "retry_preview"],
                            "default": "doctor",
                        },
                        "run_id": {
                            "type": "string",
                            "description": (
                                "Target run_id (required for 'probe_run' and 'retry_preview')."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "default": 20,
                            "description": "Maximum rows for 'recent_runs'.",
                        },
                        "job": {
                            "type": "string",
                            "description": "Optional family filter for 'probe_run'.",
                        },
                    },
                    "additionalProperties": False,
                },
            },
            {
                "name": "review_run",
                "description": (
                    "Read-only data-science evaluation of a completed or partial run in BigQuery "
                    "(`review.review_run` + `review.calibration_report`): best model overall and "
                    "per family, ensemble lift over best base model, leaderboard metrics, and "
                    "point-forecast arm / prediction interval coverage calibration."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "run_id": {
                            "type": "string",
                            "description": "Content-addressed run_id (<slug>-<12hex>) to review.",
                        },
                        "include_calibration": {
                            "type": "boolean",
                            "default": True,
                        },
                        "include_leaderboard": {
                            "type": "boolean",
                            "default": True,
                        },
                        "include_fva": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "When true, include Forecast Value Added (FVA) & Diebold-Mariano "
                                "statistical significance table."
                            ),
                        },
                        "include_horizon_decay": {
                            "type": "boolean",
                            "default": False,
                            "description": "When true, include lead-time horizon accuracy decay.",
                        },
                        "include_bias_report": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "When true, include directional bias & Tracking Signal audit."
                            ),
                        },
                        "include_exceptions": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "When true, include Top-N operational exception triage queue."
                            ),
                        },
                        "include_promo_elasticity": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "When true, include Double-ML promotional & price elasticity "
                                "decomposition."
                            ),
                        },
                        "include_inventory_policy": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "When true, include operational (s, S) / (R, S) inventory & "
                                "replenishment policy table (ROP, safety_stock, "
                                "order_up_to_level, recommended_order_qty, Newsvendor cost)."
                            ),
                        },
                        "inventory_params": {
                            "type": "object",
                            "description": (
                                "Optional keyword overrides for compute_inventory_policy "
                                "(lead_time, lead_time_std, review_period, service_level, "
                                "fill_rate, on_hand, on_order, backorders, holding_cost, "
                                "stockout_cost, min_order_qty, order_multiple, model_type)."
                            ),
                        },
                        "include_goal_seek": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "When true, solve top-down constrained QP goal-seek target "
                                "allocation on the run's forward predictions."
                            ),
                        },
                        "target_total": {
                            "description": (
                                "Optional executive top-down target total (scalar or date map) "
                                "for goal-seek allocation."
                            ),
                        },
                        "reconcile_temporal": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "When true, apply THieF temporal hierarchy reconciliation across "
                                "the run's forward predictions."
                            ),
                        },
                        "temporal_reconciliation_method": {
                            "type": "string",
                            "enum": [
                                "ols",
                                "wls_struct",
                                "wls_var",
                                "mint_shrink",
                                "mint_nonneg",
                                "cross_temporal_mint",
                            ],
                            "default": "wls_struct",
                            "description": (
                                "Temporal reconciliation estimator when reconcile_temporal=True."
                            ),
                        },
                        "include_cold_start": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "When true, include cold-start NPI donor transfer / diffusion "
                                "forecast."
                            ),
                        },
                        "include_aci": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "When true, compute online Adaptive Conformal Inference (ACI) "
                                "intervals and rolling empirical coverage tracking."
                            ),
                        },
                        "include_trajectories": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "When true, include joint copula trajectory sample paths."
                            ),
                        },
                        "include_tail_risk": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "When true, compute portfolio Value-at-Risk (VaR) and Expected "
                                "Shortfall (CVaR) from joint copula trajectories."
                            ),
                        },
                        "copula_type": {
                            "type": "string",
                            "enum": [
                                "auto",
                                "gaussian",
                                "student_t",
                                "iman_conover",
                                "empirical",
                            ],
                            "default": "gaussian",
                            "description": "Copula family for joint trajectory sampling.",
                        },
                        "optimize_inventory_params": {
                            "type": "object",
                            "description": (
                                "Optional keyword arguments for portfolio-constrained inventory "
                                "optimization (max_budget, max_volume, max_total_units, "
                                "target_portfolio_fill_rate, vendor_moq, uncertainty_source)."
                            ),
                        },
                        "optimize_scenario_params": {
                            "type": "object",
                            "description": (
                                "Optional keyword arguments for prescriptive price & promo "
                                "calendar optimization."
                            ),
                        },
                        "export_report_path": {
                            "type": "string",
                            "description": (
                                "Optional file path (.html) to export a self-contained "
                                "interactive HTML report."
                            ),
                        },
                        "include_similarity": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "When true, compute pairwise time-series similarity and behavioral "
                                "clusters across the run's series."
                            ),
                        },
                        "similarity_method": {
                            "type": "string",
                            "default": "hybrid",
                            "description": "Similarity kernel lens when include_similarity=true.",
                        },
                        "include_substitutes": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "When true, compute inventory-aware SKU substitute recommendations "
                                "(recommend_substitutes_df)."
                            ),
                        },
                        "include_emerging_trends": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "When true, compute breakout velocity, acceleration, and "
                                "bellwether merchandising actions (detect_emerging_trends_df)."
                            ),
                        },
                        "text_cols": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional text columns for multi-modal similarity.",
                        },
                        "attribute_cols": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional attribute columns for multi-modal similarity.",
                        },
                        "hierarchy_cols": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional hierarchy columns for multi-modal similarity.",
                        },
                        "within_hierarchy_level": {
                            "type": "string",
                            "description": (
                                "Optional hierarchy column restricting substitute search."
                            ),
                        },
                        "inventory_records": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": (
                                "Optional inventory records for recommend_substitutes_df."
                            ),
                        },
                        "include_clearance": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "When true, solve lifecycle markdown & perishable clearance MILP "
                                "(optimize_clearance_df)."
                            ),
                        },
                        "optimize_clearance_params": {
                            "type": "object",
                            "description": (
                                "Optional keyword arguments for optimize_clearance "
                                "(initial_inventory, expiry_step, target_space_freed, "
                                "reclaim_by_step, salvage_value, target_sell_through_pct, "
                                "covariate_cols, hierarchy_cols)."
                            ),
                        },
                        "include_contract_volume": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "When true, solve vendor contract volume & tiered rebate / "
                                "Take-or-Pay MILP (optimize_contract_volume_df)."
                            ),
                        },
                        "optimize_contract_volume_params": {
                            "type": "object",
                            "description": (
                                "Optional keyword arguments for optimize_contract_volume "
                                "(contract_tiers, min_contract_volume, shortfall_penalty_per_unit, "
                                "risk_aversion_lambda, cvar_alpha, covariate_cols, hierarchy_cols)."
                            ),
                        },
                    },
                    "required": ["run_id"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "launch_or_repair_run",
                "description": (
                    "Safety-gated cloud execution and repair tool (requires the MCP server to be "
                    "started with `--allow-launch`). Actions: 'run' (launch multi-family DAG), "
                    "'retry' (submit surgical repair of failed/missing cells), 'settle' (repair "
                    "stale job status rows from runtime probes), 'cancel' (stop in-flight cloud "
                    "jobs), 'close_runs' (finalize abandoned RUNNING headers with terminal jobs)."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["run", "retry", "settle", "cancel", "close_runs"],
                        },
                        "config": {
                            "description": (
                                "RunConfig dict, JSON string, or file path (for 'run', or for "
                                "'retry'/'settle'/'cancel' when run_id is not passed)."
                            ),
                        },
                        "run_id": {
                            "type": "string",
                            "description": "Target run_id (for 'retry', 'settle', or 'cancel').",
                        },
                        "force": {
                            "type": "boolean",
                            "default": False,
                        },
                        "reason": {
                            "type": "string",
                            "default": "",
                        },
                        "n_series": {
                            "type": "integer",
                        },
                        "job": {
                            "type": "string",
                        },
                    },
                    "required": ["action"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "build_auto_config",
                "description": (
                    "Synthesize a glass-box RunConfig and ConfigBlueprint via the Auto-Forecaster "
                    "(`Forecaster.build_config`) across the 4 observational exposure tiers "
                    "('first_light', 'wide_field', 'deep_sky', 'deep_field') and 5 n_series "
                    f"scale regimes ('micro' to 'hyperscale'). Accepts any of the "
                    f"{len(list_benchmarks())} synthetic DGP benchmark names "
                    "('dgp_01_smooth_seasonal' .. 'dgp_20_monthly_regime_drift_damped'), "
                    "'playground' sample data, a local CSV/Parquet path, or a BigQuery table name. "
                    "Returns the Step 0 7-dimension data profile (`profile_table`), validated "
                    "RunConfig JSON, deterministic run_id, field-by-field rationale table, "
                    "4-anchor baseline ladder, and parametric wall-clock duration estimate."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "default": "playground",
                            "description": (
                                "Data source: 'playground', a synthetic benchmark name "
                                "('dgp_01_smooth_seasonal' .. "
                                "'dgp_20_monthly_regime_drift_damped'), a local CSV/Parquet file "
                                "path, or a BigQuery table name."
                            ),
                        },
                        "benchmark": {
                            "type": "string",
                            "description": (
                                "Optional alias for source when selecting a DGP benchmark."
                            ),
                        },
                        "source_table": {
                            "type": "string",
                            "description": "Optional BigQuery source table name.",
                        },
                        "n_series": {
                            "type": "integer",
                            "default": 4,
                            "description": (
                                "Series count when generating sample or DGP benchmark data."
                            ),
                        },
                        "horizon": {
                            "type": "integer",
                            "default": 14,
                            "description": "Forecast horizon H in native frequency steps.",
                        },
                        "tier": {
                            "type": "string",
                            "enum": [
                                "first_light",
                                "wide_field",
                                "deep_sky",
                                "deep_field",
                                "glimpse",
                                "starlight",
                                "survey",
                            ],
                            "default": "wide_field",
                            "description": "Observational exposure tier.",
                        },
                        "where": {
                            "type": "string",
                            "enum": ["auto", "local", "bigquery", "cloud_cpu", "cloud_all"],
                            "default": "auto",
                            "description": "Execution sandbox envelope.",
                        },
                        "goal": {
                            "type": "string",
                            "enum": ["accuracy", "probabilistic", "service_level", "fast"],
                            "default": "accuracy",
                            "description": "Optimization objective.",
                        },
                        "max_minutes": {
                            "type": "number",
                            "description": "Optional hard wall-clock budget in minutes.",
                        },
                        "max_wall_clock_minutes": {
                            "type": "number",
                            "description": "Optional hard wall-clock budget in minutes.",
                        },
                        "max_usd": {
                            "type": "number",
                            "description": "Optional cloud cost budget in USD.",
                        },
                        "only_available_models": {
                            "type": "boolean",
                            "default": True,
                            "description": (
                                "Filter synthesized models to those installed in the active "
                                "environment."
                            ),
                        },
                        "baselines": {
                            "description": (
                                "Baseline anchor policy ('auto', 'none', or array of model names)."
                            ),
                        },
                        "holidays": {
                            "description": (
                                "Optional ISO country code or array of country codes (e.g. ['US'])."
                            ),
                        },
                        "decision_metric": {
                            "type": "string",
                            "description": "Optional pinned decision metric (e.g. 'wape', 'mase').",
                        },
                        "n_folds": {
                            "type": "integer",
                            "description": "Optional pinned expanding-window backtest fold count.",
                        },
                        "allowed_families": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of allowed model families.",
                        },
                        "models": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional exact model shortlist to pin.",
                        },
                        "include_models": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional models to union into the tier shortlist.",
                        },
                        "exclude_models": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional models to remove from the tier shortlist.",
                        },
                        "write_config_path": {
                            "type": "string",
                            "description": (
                                "Optional file path (.json) to write the synthesized RunConfig."
                            ),
                        },
                        "emit_airflow_path": {
                            "type": "string",
                            "description": (
                                "Optional file path (.py) to write an Airflow / Composer 3 DAG."
                            ),
                        },
                        "autofocus": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "With emit_airflow_path: emit a multi-pass autofocus Airflow DAG."
                            ),
                        },
                    },
                    "additionalProperties": False,
                },
            },
            {
                "name": "run_auto_forecast",
                "description": (
                    "Run the Glass-Box Auto-Forecaster (`auto_forecast` / `Forecaster.auto`) "
                    f"on any of the {len(list_benchmarks())} synthetic DGP benchmarks "
                    "('dgp_01_smooth_seasonal' .. 'dgp_20_monthly_regime_drift_damped'), "
                    "'playground' sample panel, local CSV/Parquet file, or BigQuery table "
                    "(with --allow-launch), with multi-branch autofocus refinement "
                    "(`autofocus=True`, `autofocus_mode='dag'`). Returns the Step 0 data profile "
                    "(`profile_table`), pass-by-pass `autofocus_ledger`, 4-anchor `baseline_value` "
                    "ladder (plus beaten complex challengers when a univariate model wins), "
                    "per-level `hierarchy_value`, head-to-head `model_comparison`, and the "
                    "7-Question `practitioner_assessment`."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "default": "playground",
                            "description": (
                                "Offline data source: 'playground', a synthetic benchmark name "
                                "('dgp_01_smooth_seasonal' .. "
                                "'dgp_20_monthly_regime_drift_damped'), or a local CSV/Parquet "
                                "file path."
                            ),
                        },
                        "benchmark": {
                            "type": "string",
                            "description": (
                                "Optional DGP benchmark name ('dgp_01_smooth_seasonal' ..)."
                            ),
                        },
                        "source_table": {
                            "type": "string",
                            "description": (
                                "Optional BigQuery source table name (requires --allow-launch)."
                            ),
                        },
                        "horizon": {
                            "type": "integer",
                            "default": 14,
                            "description": "Forecast horizon H in native frequency steps.",
                        },
                        "tier": {
                            "type": "string",
                            "enum": [
                                "first_light",
                                "wide_field",
                                "deep_sky",
                                "deep_field",
                                "glimpse",
                                "starlight",
                                "survey",
                            ],
                            "default": "wide_field",
                            "description": "Initial observational exposure tier.",
                        },
                        "where": {
                            "type": "string",
                            "enum": ["auto", "local", "bigquery", "cloud_cpu", "cloud_all"],
                            "default": "local",
                            "description": "Execution sandbox envelope.",
                        },
                        "goal": {
                            "type": "string",
                            "enum": ["accuracy", "probabilistic", "service_level", "fast"],
                            "default": "accuracy",
                            "description": "Optimization objective.",
                        },
                        "autofocus": {
                            "type": "boolean",
                            "default": True,
                            "description": (
                                "When true, run bounded residual-driven autofocus refinement "
                                "(reusing passing cells from Pass 0)."
                            ),
                        },
                        "autofocus_mode": {
                            "type": "string",
                            "enum": ["dag", "linear"],
                            "default": "dag",
                            "description": "Multi-branch 'dag' or single-path 'linear' autofocus.",
                        },
                        "max_passes": {
                            "type": "integer",
                            "description": "Maximum total passes (including Pass 1).",
                        },
                        "max_autofocus_passes": {
                            "type": "integer",
                            "default": 1,
                            "description": "Maximum refinement passes after Pass 0.",
                        },
                        "baselines": {
                            "description": (
                                "Baseline anchor policy ('auto', 'none', or array of model names)."
                            ),
                        },
                        "holidays": {
                            "description": (
                                "Optional ISO country code or array of country codes (e.g. ['US'])."
                            ),
                        },
                        "decision_metric": {
                            "type": "string",
                            "description": "Optional pinned decision metric (e.g. 'wape', 'mase').",
                        },
                        "n_folds": {
                            "type": "integer",
                            "description": "Optional pinned expanding-window backtest fold count.",
                        },
                        "allowed_families": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of allowed model families.",
                        },
                        "models": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional exact model shortlist to pin.",
                        },
                        "include_models": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional models to union into the tier shortlist.",
                        },
                        "exclude_models": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional models to remove from the tier shortlist.",
                        },
                        "n_series": {
                            "type": "integer",
                            "default": 3,
                            "description": "Series count when source='playground' or benchmark.",
                        },
                        "export_report_path": {
                            "type": "string",
                            "description": (
                                "Optional file path (.html) to export a self-contained "
                                "interactive HTML report including the Autofocus & Blueprint tab."
                            ),
                        },
                        "evolve_target": {
                            "type": ["string", "null"],
                            "enum": [
                                "features",
                                "ensemble",
                                "model",
                                "policy",
                                "similarity_kernel",
                                None,
                            ],
                            "default": None,
                            "description": (
                                "Optional # EVOLVE-BLOCK target ('features', 'ensemble', 'model', "
                                "'policy', 'similarity_kernel') to trigger Forecaster.evolve(...) "
                                "with the 3-Gate Anti-Overfitting Ratchet."
                            ),
                        },
                        "evolve_backend": {
                            "type": "string",
                            "enum": ["alphaevolve", "vertex_gemini", "mock"],
                            "default": "alphaevolve",
                            "description": (
                                "Evolution backend when evolve_target is set: 'alphaevolve' "
                                "(default, Gemini Enterprise Discovery Engine), 'vertex_gemini' "
                                "(backup Vertex AI Gemini mutator), or 'mock' (offline test)."
                            ),
                        },
                        "evolve_max_evaluations": {
                            "type": "integer",
                            "default": 10,
                            "description": (
                                "Maximum candidate evaluations when evolve_target is set."
                            ),
                        },
                        "evolve_model_name": {
                            "type": "string",
                            "default": "gemini-3.8-flash",
                            "description": (
                                "Gemini / Vertex AI model identifier for code evolution "
                                "(default: 'gemini-3.8-flash')."
                            ),
                        },
                        "evolve_from_run_ids": {
                            "type": ["array", "null"],
                            "items": {"type": "string"},
                            "default": None,
                            "description": (
                                "Optional list of completed BigQuery run_ids to pool for "
                                "post-run super-ensembling via Forecaster.evolve(from_run_ids=...)."
                            ),
                        },
                    },
                    "additionalProperties": False,
                },
            },
            {
                "name": "compute_time_series_similarity",
                "description": (
                    "Compute multi-lens and multi-modal time-series similarity (`hybrid`, "
                    "`physics_attractor`, `spectral_phase`, `elastic_dtw`, `minirocket`, "
                    "`structural_signature`, `foundation_latent`), Takens phase-space / dynamical "
                    "invariants, Ward behavioral clusters, digital-twin analog matches "
                    "(`full_history`, `launch_prefix`, `recent_window`), stockout-aware assortment "
                    "substitutes (`substitutes`), and emerging trend breakouts (`emerging_trends`)."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": ["string", "null"],
                            "default": "playground",
                            "description": (
                                "Data source: 'playground', a DGP benchmark name, or a local "
                                "CSV/Parquet/JSON path."
                            ),
                        },
                        "source_path_or_table": {
                            "type": ["string", "null"],
                            "default": None,
                            "description": (
                                "Optional alias for source path, benchmark name, or BigQuery table."
                            ),
                        },
                        "preset": {
                            "type": "string",
                            "default": "retail_daily",
                            "description": "Playground preset when source='playground'.",
                        },
                        "records": {
                            "type": ["array", "null"],
                            "items": {"type": "object"},
                            "default": None,
                            "description": (
                                "Optional inline panel records (each with ts_id, ds, y)."
                            ),
                        },
                        "method": {
                            "type": "string",
                            "enum": [
                                "hybrid",
                                "physics_attractor",
                                "spectral_phase",
                                "elastic_dtw",
                                "minirocket",
                                "structural_signature",
                                "foundation_latent",
                            ],
                            "default": "hybrid",
                            "description": "Similarity kernel lens.",
                        },
                        "mode": {
                            "type": "string",
                            "enum": [
                                "full_history",
                                "launch_prefix",
                                "recent_window",
                                "substitutes",
                                "emerging_trends",
                            ],
                            "default": "full_history",
                            "description": (
                                "Digital-twin matching or retail sensing mode "
                                "('full_history', 'launch_prefix', 'recent_window', "
                                "'substitutes', 'emerging_trends')."
                            ),
                        },
                        "query_ts_id": {
                            "type": ["string", "null"],
                            "default": None,
                            "description": (
                                "Optional query series ID for targeted k-NN digital-twin or "
                                "substitute search."
                            ),
                        },
                        "top_k": {
                            "type": "integer",
                            "default": 5,
                            "description": "Number of top neighbor or substitute pairs per series.",
                        },
                        "n_clusters": {
                            "type": ["integer", "string"],
                            "default": "auto",
                            "description": (
                                "Target cluster count K >= 2 or 'auto' for silhouette-optimal K*."
                            ),
                        },
                        "max_lag": {
                            "type": "integer",
                            "default": 0,
                            "description": "Maximum lead-lag cross-correlation shift tau.",
                        },
                        "window": {
                            "type": "integer",
                            "default": 14,
                            "description": (
                                "Recent observation window for emerging trend velocity and "
                                "acceleration sensing."
                            ),
                        },
                        "include_substitutes": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "When true, compute stockout-aware assortment substitutes "
                                "(`recommend_substitutes_df`)."
                            ),
                        },
                        "include_emerging_trends": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "When true, detect emerging trend breakouts and bellwether "
                                "lead-lag propagation (`detect_emerging_trends_df`)."
                            ),
                        },
                        "text_cols": {
                            "type": ["array", "string", "null"],
                            "default": None,
                            "description": (
                                "Optional text columns for N-gram TF-IDF + SVD semantic similarity."
                            ),
                        },
                        "attribute_cols": {
                            "type": ["array", "null"],
                            "items": {"type": "string"},
                            "default": None,
                            "description": (
                                "Optional categorical/numeric attribute columns for Gower "
                                "similarity."
                            ),
                        },
                        "hierarchy_cols": {
                            "type": ["array", "null"],
                            "items": {"type": "string"},
                            "default": None,
                            "description": (
                                "Optional ordered hierarchy columns for Wu-Palmer LCA tree "
                                "proximity."
                            ),
                        },
                        "within_hierarchy_level": {
                            "type": ["string", "array", "null"],
                            "default": None,
                            "description": (
                                "Optional hierarchy column(s) restricting matches to the same "
                                "taxonomy node."
                            ),
                        },
                        "inventory_records": {
                            "type": ["array", "null"],
                            "items": {"type": "object"},
                            "default": None,
                            "description": (
                                "Optional inline inventory records (ts_id, on_hand, safety_stock) "
                                "for assortment substitution."
                            ),
                        },
                        "require_in_stock": {
                            "type": "boolean",
                            "default": True,
                            "description": (
                                "When true, exclude out-of-stock candidates in substitute "
                                "recommendations."
                            ),
                        },
                        "n_series": {
                            "type": "integer",
                            "default": 4,
                            "description": "Series count when generating playground data.",
                        },
                        "ts_id_col": {"type": "string", "default": "ts_id"},
                        "date_col": {"type": "string", "default": "ds"},
                        "target_col": {"type": "string", "default": "y"},
                        "freq": {"type": "string", "default": "D"},
                        "project_id": {"type": ["string", "null"], "default": None},
                    },
                    "additionalProperties": False,
                },
            },
            {
                "name": "optimize_inventory_clearance",
                "description": (
                    "Solve lifecycle markdown & perishable expiry clearance MILP "
                    "(`mode='clearance'`, `clearance_optimizer.optimize_clearance`), warehouse & "
                    "shelf space reclamation (`target_space_freed`, `unit_volume`, "
                    "`reclaim_by_step`), multi-SKU vendor contract volume & tiered rebate / "
                    "Take-or-Pay commitment optimization (`mode='contract_volume'`, "
                    "`clearance_optimizer.optimize_contract_volume`), assortment rationalization "
                    "& delisting demand transference (`mode='assortment'`, "
                    "`clearance_optimizer.optimize_assortment`), simplex size-curve & integer "
                    "pre-pack allocation (`mode='prepacks'`, "
                    "`inventory_optimizer.optimize_prepacks`), or lateral store-to-store / depot "
                    "rebalancing LP (`mode='rebalancing'`, "
                    "`inventory_optimizer.optimize_rebalancing`)."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "enum": [
                                "clearance",
                                "contract_volume",
                                "assortment",
                                "prepacks",
                                "rebalancing",
                            ],
                            "default": "clearance",
                            "description": (
                                "'clearance' for lifecycle markdown & space reclamation MILP; "
                                "'contract_volume' for vendor tiered rebate / Take-or-Pay MILP; "
                                "'assortment' for assortment rationalization & delisting "
                                "walk_rate; 'prepacks' for simplex size-curve & integer pre-packs; "
                                "'rebalancing' for lateral store-to-store transshipment LP."
                            ),
                        },
                        "records": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": (
                                "Optional forecast or historical panel rows "
                                "(ts_id, ds, yhat or y). "
                                "When omitted, runs offline on sample playground data."
                            ),
                        },
                        "history": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": (
                                "Optional historical panel rows for own-price elasticity and "
                                "digital-twin similarity estimation."
                            ),
                        },
                        "initial_inventory": {
                            "description": (
                                "Starting on-hand inventory (scalar float or {ts_id: units} map)."
                            ),
                        },
                        "expiry_step": {
                            "description": (
                                "Perishable shelf-life or end-of-season deadline step (int or "
                                "{ts_id: step} map)."
                            ),
                        },
                        "discount_ladder": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": (
                                "Allowed markdown discount rungs in [0, 0.95], e.g. "
                                "[0.0, 0.10, 0.20, 0.30, 0.50]."
                            ),
                        },
                        "monotone_markdown": {
                            "type": "boolean",
                            "default": True,
                            "description": (
                                "Enforce non-decreasing markdowns over time (d_{i,t} <= d_{i,t+1})."
                            ),
                        },
                        "base_price": {
                            "description": "Baseline unit selling price (float or {ts_id: price}).",
                        },
                        "unit_cost": {
                            "description": "Unit cost of goods sold (float or {ts_id: cost}).",
                        },
                        "salvage_value": {
                            "description": "Terminal salvage value per unsold unit at expiry.",
                        },
                        "spoilage_cost": {
                            "description": "Terminal disposal/spoilage cost per unsold unit.",
                        },
                        "target_sell_through_pct": {
                            "description": (
                                "Optional minimum sell-through fraction in [0, 1] by expiry_step."
                            ),
                        },
                        "unit_volume": {
                            "description": (
                                "Cubic volume / pallet / shelf-facing footprint per unit."
                            ),
                        },
                        "target_space_freed": {
                            "type": "number",
                            "description": (
                                "Minimum total warehouse/shelf space to free by reclaim_by_step."
                            ),
                        },
                        "reclaim_by_step": {
                            "type": "integer",
                            "description": (
                                "Horizon step by which target_space_freed must be reclaimed."
                            ),
                        },
                        "contract_tiers": {
                            "description": (
                                "For mode='contract_volume': array of [target_units, rebate_usd] "
                                "pairs or {target_units: rebate_usd} map."
                            ),
                        },
                        "min_contract_volume": {
                            "type": "number",
                            "description": (
                                "For mode='contract_volume': minimum Take-or-Pay portfolio volume."
                            ),
                        },
                        "shortfall_penalty_per_unit": {
                            "type": "number",
                            "default": 5.0,
                            "description": (
                                "Per-unit penalty for shortfall below min_contract_volume."
                            ),
                        },
                        "price_elasticity": {
                            "description": (
                                "Optional explicit own-price elasticity (float or {ts_id: float})."
                            ),
                        },
                        "use_digital_twin_elasticity": {
                            "type": "boolean",
                            "default": True,
                            "description": (
                                "Borrow deep-discount price elasticity from Track P digital-twin "
                                "neighbors when a SKU lacks markdown history."
                            ),
                        },
                        "cannibalization_penalty": {
                            "type": "number",
                            "default": 0.15,
                            "description": "Cross-SKU substitute cannibalization penalty weight.",
                        },
                        "oof": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": (
                                "Optional out-of-fold backtest rows (ts_id, y_true, yhat, "
                                "model_type) for per-SKU OOF champion selection and empirical "
                                "CVaR shortfall."
                            ),
                        },
                        "trajectories": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": (
                                "Optional joint copula trajectory rows (ts_id, trajectory_id, "
                                "yhat_traj) for mode='contract_volume' CVaR shortfall."
                            ),
                        },
                        "covariate_cols": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Optional exogenous covariate columns (e.g. promo_flag, holiday, "
                                "temperature) to partial out via Frisch-Waugh-Lovell (FWL) when "
                                "estimating own-price elasticity."
                            ),
                        },
                        "elasticity_records": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": (
                                "Optional precomputed Double-ML promo_elasticity_df rows."
                            ),
                        },
                        "model_type": {
                            "type": "string",
                            "description": "Optional model_type filter when records spans models.",
                        },
                        "reconciliation_method": {
                            "type": "string",
                            "description": (
                                "Optional spatial/temporal reconciliation_method filter."
                            ),
                        },
                        "prefer_reconciled": {
                            "type": "boolean",
                            "default": True,
                            "description": (
                                "When true, prefer reconciled leaf forecasts over base forecasts."
                            ),
                        },
                        "hierarchy_cols": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Optional hierarchy columns (e.g. ['category', 'store']) for "
                                "bottom-up hierarchy_rollup aggregation."
                            ),
                        },
                        "risk_aversion_lambda": {
                            "type": "number",
                            "default": 0.25,
                            "description": (
                                "For mode='contract_volume': weight in [0, 1] blending expected "
                                "shortfall and CVaR tail shortfall."
                            ),
                        },
                        "cvar_alpha": {
                            "type": "number",
                            "default": 0.90,
                            "description": (
                                "For mode='contract_volume': tail confidence level in (0.5, 0.999) "
                                "for CVaR volume shortfall."
                            ),
                        },
                        "horizon": {
                            "type": "integer",
                            "default": 14,
                        },
                        "n_series": {
                            "type": "integer",
                            "default": 3,
                        },
                        "max_skus": {
                            "type": "integer",
                            "description": (
                                "For mode='assortment': maximum retained SKUs after delisting."
                            ),
                        },
                        "max_shelf_space": {
                            "type": "number",
                            "description": "For mode='assortment': maximum shelf space / facings.",
                        },
                        "walk_rate": {
                            "description": (
                                "For mode='assortment': optional explicit shopper walk-away rate."
                            ),
                        },
                        "prepacks": {
                            "type": "object",
                            "description": (
                                "For mode='prepacks': map of {pack_id: {size_id: units}} bundles."
                            ),
                        },
                        "transfer_cost": {
                            "description": (
                                "For mode='rebalancing': unit lateral transfer cost between nodes."
                            ),
                        },
                        "max_transfer_lead_time": {
                            "type": "number",
                            "default": 5.0,
                            "description": (
                                "For mode='rebalancing': maximum eligible transfer transit time."
                            ),
                        },
                    },
                    "additionalProperties": False,
                },
            },
            {
                "name": "ask_forecast_registry",
                "description": (
                    "Ask a natural-language forecasting, diagnostic, FVA, horizon-decay, "
                    "anomaly, feature-attribution, or S&OP rollup question against either a "
                    "stateful BigQuery Conversational Analytics DataAgent (backed by Knowledge "
                    "Catalog glossary & the 11 registry SQL views) or an offline Playground run."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Natural-language question about forecast runs.",
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["auto", "bigquery_agent", "offline_playground", "probe"],
                            "default": "auto",
                            "description": (
                                "'bigquery_agent' queries the stateful BigQuery DataAgent; "
                                "'offline_playground' runs offline on synthetic panel data; "
                                "'probe' checks Conversational Analytics & Knowledge Catalog API "
                                "readiness without raising; 'auto' selects BigQuery when ready, "
                                "else offline_playground."
                            ),
                        },
                        "run_id": {
                            "type": "string",
                            "default": "latest",
                            "description": (
                                "Target run_id ('latest' for newest COMPLETED run, 'all' for "
                                "cross-run queries, or an exact/prefix run_id)."
                            ),
                        },
                        "project_id": {
                            "type": "string",
                            "description": "Optional GCP project ID override.",
                        },
                        "dataset_id": {
                            "type": "string",
                            "description": "Optional BigQuery registry dataset ID override.",
                        },
                        "conversation_id": {
                            "type": "string",
                            "description": "Optional conversation_id for stateful follow-up turns.",
                        },
                        "stateful": {
                            "type": "boolean",
                            "default": True,
                            "description": "Persist multi-turn conversation state in BigQuery.",
                        },
                        "auto_ensure_agent": {
                            "type": "boolean",
                            "default": True,
                            "description": (
                                "Provision the BigQuery DataAgent automatically if not yet created."
                            ),
                        },
                        "model": {
                            "type": "string",
                            "default": "holtwinters",
                            "description": "Model name when mode='offline_playground'.",
                        },
                        "horizon": {
                            "type": "integer",
                            "default": 14,
                            "description": "Forecast horizon when mode='offline_playground'.",
                        },
                        "n_series": {
                            "type": "integer",
                            "default": 3,
                            "description": "Series count when mode='offline_playground'.",
                        },
                    },
                    "additionalProperties": False,
                },
            },
        ]

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Dispatch a tool call and wrap the result in an MCP tool response."""
        args = arguments or {}
        try:
            payload = self._dispatch_tool(name, args)
            return {
                "content": [{"type": "text", "text": _dumps(payload)}],
                "isError": False,
            }
        except (ScaleForecastError, ValueError, KeyError, FileNotFoundError) as exc:
            err_payload = {
                "error": type(exc).__name__,
                "message": str(exc),
                "environment": probe_environment(self.repo_root),
            }
            return {
                "content": [{"type": "text", "text": _dumps(err_payload)}],
                "isError": True,
            }
        except Exception as exc:  # noqa: BLE001 - MCP tool boundary must return structured JSON-RPC result
            err_payload = {
                "error": type(exc).__name__,
                "message": str(exc),
            }
            return {
                "content": [{"type": "text", "text": _dumps(err_payload)}],
                "isError": True,
            }

    def _dispatch_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "probe_environment":
            return probe_environment(self.repo_root)

        if name == "list_catalog":
            category = str(args.get("category", "all"))
            family = args.get("family")
            available_only = bool(args.get("available_only", False))
            models = _build_models_catalog()
            if family:
                models = [m for m in models if m["family"] == family]
            if available_only:
                models = [m for m in models if m["available"]]

            extras_list = [
                {
                    "extra": ext,
                    "install_command": f'pip install "scale-forecasting[{ext}]"',
                    "probe_modules": list(mods),
                    "description": EXTRA_DESCRIPTIONS[ext],
                }
                for ext, mods in ALL_EXTRA_MODULES.items()
            ]
            catalog_map: dict[str, Any] = {
                "models": models,
                "metrics": _build_metrics_catalog(),
                "runtimes": list(RUNTIME_CATALOG),
                "gpus": list(GPU_CATALOG),
                "extras": extras_list,
                "views": _build_views_catalog(),
                "configs": _build_configs_catalog(self.repo_root),
                "benchmarks": benchmark_catalog().to_dict(orient="records"),
                "auto_tiers": {
                    "exposure_tiers": [{"tier": t, **meta} for t, meta in EXPOSURE_TIERS.items()],
                    "scale_regimes": [
                        {"scale_regime": s, **meta} for s, meta in SCALE_REGIMES.items()
                    ],
                    "where_options": ["auto", "local", "bigquery", "cloud_cpu", "cloud_all"],
                    "optimization_goals": ["accuracy", "probabilistic", "service_level", "fast"],
                },
            }
            if category == "all":
                return catalog_map
            if category not in catalog_map:
                raise ValueError(f"Unknown catalog category: {category!r}")
            return {category: catalog_map[category]}

        if name == "inspect_config_schema":
            schema = build_run_config_json_schema()
            section = args.get("section")
            if not section:
                return {
                    "$id": schema["$id"],
                    "top_level_properties": schema.get("properties", {}),
                    "required": schema.get("required", []),
                    "available_defs": sorted(schema.get("$defs", {}).keys()),
                }
            defs = schema.get("$defs", {})
            props = schema.get("properties", {})
            if section in defs:
                return {"section": section, "definition": defs[section]}
            if section in props:
                prop_def = props[section]
                ref = prop_def.get("$ref", "")
                def_name = ref.split("/")[-1] if ref.startswith("#/$defs/") else None
                return {
                    "section": section,
                    "property": prop_def,
                    "definition": defs.get(def_name) if def_name else None,
                }
            for def_key, def_val in defs.items():
                if def_key.lower() == str(section).lower():
                    return {"section": def_key, "definition": def_val}
            raise ValueError(
                f"Unknown schema section {section!r}; available properties={sorted(props)}, "
                f"$defs={sorted(defs)}"
            )

        if name == "validate_and_dry_run":
            cfg = _parse_run_config_input(args["config"], self.repo_root)
            n_series = args.get("n_series")
            if n_series is not None:
                cfg = cfg.with_series_limit(int(n_series))
            if args.get("ignore_unavailable_models"):
                cfg = cfg.with_available_models()
            run_id = make_run_id(cfg)
            workload = estimate_workload(cfg)
            run_dag = plan_dag(cfg)
            nodes = dag_nodes(run_dag)
            by_family = group_models_by_family(cfg)
            missing_local = []
            for mname in cfg.models:
                mcls = get_model(mname)
                if not mcls.is_available():
                    ext = mcls.optional_extra or ("gcp" if mcls.runtime == "bigquery" else "core")
                    missing_local.append(
                        {
                            "model": mname,
                            "family": mcls.family,
                            "extra": ext,
                            "install_command": f'pip install "scale-forecasting[{ext}]"',
                        }
                    )
            resolved_families = {
                job.family: (
                    dataclasses.asdict(job.compute)
                    if job.compute is not None
                    else {"family": "native", "runtime": "bigquery", "hardware": "cpu"}
                )
                for job in run_dag.jobs
            }
            return {
                "valid": True,
                "run_id": run_id,
                "workload": dataclasses.asdict(workload),
                "families": {fam: list(ms) for fam, ms in by_family.items()},
                "resolved_family_compute": resolved_families,
                "dag_nodes": [dataclasses.asdict(n) for n in nodes],
                "missing_local_model_extras": missing_local,
                "normalized_config": cfg.model_dump(mode="json"),
            }

        if name == "run_playground":
            from . import playground

            models_list = args.get("models")
            default_model = (
                str(models_list[0])
                if isinstance(models_list, list) and models_list
                else "holtwinters"
            )
            model_name = str(args.get("model", default_model))
            horizon = int(args.get("horizon", 14))
            n_series = int(args.get("n_series", 3))
            backtest = bool(args.get("backtest", True))

            mcls = get_model(model_name)
            if mcls.runtime != "python":
                raise ValueError(
                    f"Model {model_name!r} uses runtime={mcls.runtime!r} and cannot run in the "
                    "offline Python playground. Choose a Python model or use `plan_execution`."
                )
            if not mcls.is_available():
                ext = mcls.optional_extra or "models"
                raise MissingExtraError(
                    f"Model {model_name!r} requires package {mcls.package!r} from extra [{ext}]. "
                    f'Install with: pip install "scale-forecasting[{ext}]"'
                )

            scenarios_arg = args.get("scenarios")
            price_col = args.get("price_col")
            unit_cost_arg = args.get("unit_cost")
            inc_inv_impact = bool(args.get("include_inventory_impact", False))
            inc_promo_elast = bool(args.get("include_promo_elasticity", False))
            opt_scen_params = args.get("optimize_scenario_params")
            opt_clear_pg = args.get("optimize_clearance_params")
            opt_cv_pg = args.get("optimize_contract_volume_params")
            with_exog = bool(
                scenarios_arg
                or price_col
                or inc_promo_elast
                or isinstance(opt_scen_params, dict)
                or bool(args.get("include_clearance", False))
                or isinstance(opt_clear_pg, dict)
                or bool(args.get("include_contract_volume", False))
                or isinstance(opt_cv_pg, dict)
            )
            panel = playground.sample_data(n_series=n_series, with_exog=with_exog)
            run = playground.run_model(
                model_name,
                data=panel,
                horizon=horizon,
                backtest=backtest,
                with_exog=with_exog,
            )
            summary_text = playground.summarize(run)
            res = run.result
            sample_preds: list[dict[str, Any]] = []
            if not res.predictions.empty:
                head_df = res.predictions.head(5).copy()
                head_df["ts_id"] = res.ts_id
                sample_preds = [
                    {str(k): v for k, v in row.items()} for row in head_df.to_dict(orient="records")
                ]
            metrics_clean = {
                k: float(v) for k, v in res.metrics.items() if v is not None and v == v
            }
            out_pg: dict[str, Any] = {
                "model": model_name,
                "family": mcls.family,
                "ts_id": res.ts_id,
                "status": res.status,
                "n_series_in_panel": n_series,
                "horizon": horizon,
                "backtest": backtest,
                "demand_profile": res.demand_profile,
                "metrics": metrics_clean,
                "summary": summary_text,
                "sample_predictions": sample_preds,
            }
            if bool(args.get("include_fva", False)) and res.oof is not None and not res.oof.empty:
                from .diagnostics import compute_fva

                oof_fva = res.oof.assign(ts_id=res.ts_id, model_type=res.model_type)
                out_pg["fva"] = compute_fva(oof_fva).to_dict(orient="records")
            if scenarios_arg and isinstance(scenarios_arg, dict):
                from .scenarios import simulate_scenarios, summarize_scenarios

                cfg_pg = playground.build_config(
                    model_name, horizon=horizon, backtest=backtest, with_exog=True
                )
                sim_df = simulate_scenarios(
                    run.series,
                    cfg_pg,
                    scenarios_arg,
                    models=[model_name],
                    price_col=str(price_col) if price_col else None,
                    unit_cost=unit_cost_arg,
                    include_inventory_impact=inc_inv_impact,
                )
                out_pg["scenario_summary"] = summarize_scenarios(sim_df).to_dict(orient="records")
            if inc_promo_elast:
                out_pg["promo_elasticity"] = run.promo_elasticity_df(summary=True).to_dict(
                    orient="records"
                )
            inv_params_pg = args.get("inventory_params")
            if bool(args.get("include_inventory_policy", False)) or isinstance(inv_params_pg, dict):
                inv_kw_pg = dict(inv_params_pg) if isinstance(inv_params_pg, dict) else {}
                out_pg["inventory_policy"] = run.inventory_policy_df(**inv_kw_pg).to_dict(
                    orient="records"
                )
            if bool(args.get("reconcile_temporal", False)):
                t_method_pg = str(args.get("temporal_reconciliation_method", "wls_struct"))
                trec_pg = run.temporal_reconcile_df(method=t_method_pg)
                out_pg["temporal_reconciled_predictions"] = trec_pg.head(10).to_dict(
                    orient="records"
                )
            gs_df_pg = None
            if bool(args.get("include_goal_seek", False)) or args.get("target_total") is not None:
                from .goal_seek import summarize_goal_seek

                raw_target_pg = args.get("target_total")
                eff_target_pg = (
                    raw_target_pg
                    if raw_target_pg is not None
                    else float(run.predictions["yhat"].sum() * 1.15)
                )
                gs_df_pg = run.allocate_target(target_total=eff_target_pg)
                out_pg["goal_seek"] = gs_df_pg.to_dict(orient="records")
                out_pg["goal_seek_summary"] = summarize_goal_seek(gs_df_pg).to_dict(
                    orient="records"
                )
            cs_df_pg = None
            if bool(args.get("include_cold_start", False)):
                cs_arch_pg = str(args.get("cold_start_archetype") or "auto")
                cs_df_pg = run.cold_start_transfer_df(
                    donor_pool_df=panel if len(panel["ts_id"].unique()) > 1 else None,
                    method=cs_arch_pg,
                )
                out_pg["cold_start_transfer"] = cs_df_pg.head(10).to_dict(orient="records")
            aci_df_pg = None
            if bool(args.get("include_aci", False)):
                aci_df_pg = run.aci_df()
                out_pg["aci_coverage"] = aci_df_pg.head(10).to_dict(orient="records")
            copula_pg = str(args.get("copula_type", "gaussian"))
            n_traj_pg = int(args.get("n_trajectories", 50))
            traj_df_pg = None
            if bool(args.get("include_trajectories", False)) or bool(
                args.get("include_tail_risk", False)
            ):
                traj_df_pg = run.sample_trajectories(
                    n_paths=n_traj_pg,
                    copula_type=copula_pg,
                )
                if bool(args.get("include_trajectories", False)):
                    out_pg["trajectories_sample"] = traj_df_pg.head(10).to_dict(orient="records")
            tail_df_pg = None
            if bool(args.get("include_tail_risk", False)):
                tail_df_pg = run.tail_risk_df(
                    trajectories_df=traj_df_pg,
                    copula_type=copula_pg,
                    n_paths=n_traj_pg,
                )
                out_pg["tail_risk"] = tail_df_pg.to_dict(orient="records")
            opt_inv_params_pg = args.get("optimize_inventory_params")
            inv_opt_df_pg = None
            if isinstance(opt_inv_params_pg, dict):
                inv_opt_df_pg = run.optimize_inventory_df(**opt_inv_params_pg)
                out_pg["inventory_optimization"] = inv_opt_df_pg.to_dict(orient="records")
            scen_opt_df_pg = None
            if isinstance(opt_scen_params, dict):
                scen_opt_res_pg = run.optimize_scenario(**opt_scen_params)
                scen_opt_df_pg = (
                    scen_opt_res_pg.schedule_df
                    if hasattr(scen_opt_res_pg, "schedule_df")
                    else scen_opt_res_pg
                )
                out_pg["scenario_optimization"] = scen_opt_df_pg.head(15).to_dict(orient="records")
            subs_df_pg = None
            trends_df_pg = None
            want_sim_pg = bool(args.get("include_similarity", False))
            want_subs_pg = bool(args.get("include_substitutes", False))
            want_trends_pg = bool(args.get("include_emerging_trends", False))
            sim_m_pg = str(args.get("similarity_method", "hybrid"))
            inv_recs_pg = args.get("inventory_records")
            inv_df_pg = None
            if isinstance(inv_recs_pg, list) and inv_recs_pg:
                import pandas as pd

                inv_df_pg = pd.DataFrame.from_records(inv_recs_pg)
            if want_sim_pg:
                sim_df_pg = run.similarity_df(
                    method=sim_m_pg,
                    text_cols=args.get("text_cols"),
                    attribute_cols=args.get("attribute_cols"),
                    hierarchy_cols=args.get("hierarchy_cols"),
                    within_hierarchy_level=args.get("within_hierarchy_level"),
                )
                sim_cl_pg = run.similarity_clusters_df(
                    method=sim_m_pg,
                    text_cols=args.get("text_cols"),
                    attribute_cols=args.get("attribute_cols"),
                    hierarchy_cols=args.get("hierarchy_cols"),
                )
                out_pg["similarity"] = {
                    "method": sim_m_pg,
                    "pairs": sim_df_pg.head(15).to_dict(orient="records"),
                    "clusters": sim_cl_pg.to_dict(orient="records"),
                }
            if want_subs_pg:
                subs_df_pg = run.recommend_substitutes_df(
                    inventory_df=inv_df_pg,
                    method=sim_m_pg,
                    text_cols=args.get("text_cols"),
                    attribute_cols=args.get("attribute_cols"),
                    hierarchy_cols=args.get("hierarchy_cols"),
                    within_hierarchy_level=args.get("within_hierarchy_level"),
                )
                out_pg["substitutes"] = subs_df_pg.head(15).to_dict(orient="records")
            if want_trends_pg:
                trends_df_pg = run.emerging_trends_df(
                    method=sim_m_pg,
                    text_cols=args.get("text_cols"),
                    attribute_cols=args.get("attribute_cols"),
                    hierarchy_cols=args.get("hierarchy_cols"),
                )
                out_pg["emerging_trends"] = trends_df_pg.to_dict(orient="records")
            if bool(args.get("include_clearance", False)) or isinstance(opt_clear_pg, dict):
                clear_kw_pg = dict(opt_clear_pg) if isinstance(opt_clear_pg, dict) else {}
                clear_res_pg = run.optimize_clearance(**clear_kw_pg)
                out_pg["clearance_optimization"] = {
                    "status": clear_res_pg.status,
                    "is_feasible": clear_res_pg.is_feasible,
                    "total_profit_lift": clear_res_pg.total_profit_lift,
                    "total_space_freed": clear_res_pg.total_space_freed,
                    "overall_sell_through_pct": clear_res_pg.overall_sell_through_pct,
                    "relaxation_hint": clear_res_pg.relaxation_hint,
                    "summary": clear_res_pg.summary_df.to_dict(orient="records"),
                    "schedule": clear_res_pg.schedule_df.head(15).to_dict(orient="records"),
                    "hierarchy_rollup": (
                        clear_res_pg.hierarchy_rollup_df.to_dict(orient="records")
                        if clear_res_pg.hierarchy_rollup_df is not None
                        else None
                    ),
                }
            if bool(args.get("include_contract_volume", False)) or isinstance(opt_cv_pg, dict):
                cv_kw_pg = dict(opt_cv_pg) if isinstance(opt_cv_pg, dict) else {}
                cv_res_pg = run.optimize_contract_volume(**cv_kw_pg)
                out_pg["contract_volume_optimization"] = {
                    "status": cv_res_pg.status,
                    "is_feasible": cv_res_pg.is_feasible,
                    "baseline_portfolio_units": cv_res_pg.baseline_portfolio_units,
                    "optimal_portfolio_units": cv_res_pg.optimal_portfolio_units,
                    "target_contract_units": cv_res_pg.target_contract_units,
                    "achieved_tier_units": cv_res_pg.achieved_tier_units,
                    "rebate_earned": cv_res_pg.rebate_earned,
                    "shortfall_units": cv_res_pg.shortfall_units,
                    "cvar_shortfall_units": cv_res_pg.cvar_shortfall_units,
                    "net_value_lift": cv_res_pg.net_value_lift,
                    "relaxation_hint": cv_res_pg.relaxation_hint,
                    "summary": cv_res_pg.summary_df.to_dict(orient="records"),
                    "schedule": cv_res_pg.schedule_df.head(15).to_dict(orient="records"),
                    "hierarchy_rollup": (
                        cv_res_pg.hierarchy_rollup_df.to_dict(orient="records")
                        if cv_res_pg.hierarchy_rollup_df is not None
                        else None
                    ),
                }
            rep_path = args.get("export_report_path")
            if rep_path:
                written = run.export_report(
                    str(rep_path),
                    goal_seek_df=gs_df_pg,
                    aci_df=aci_df_pg,
                    tail_risk_df=tail_df_pg,
                    cold_start_df=cs_df_pg,
                    inventory_opt_df=inv_opt_df_pg,
                    scenario_opt_df=scen_opt_df_pg,
                )
                out_pg["report_path"] = str(written)
            return out_pg

        if name == "plan_execution":
            from . import airflow_emit, launch_plan

            cfg = _parse_run_config_input(args["config"], self.repo_root)
            mode = str(args.get("mode", "dry_run"))
            force = bool(args.get("force", False))

            if mode == "emit_airflow":
                with_retry = bool(args.get("with_retry", False))
                run_id = make_run_id(cfg)
                config_uri = f"gs://<warehouse>/staging/{run_id}/config.json"
                dag_src = airflow_emit.emit_airflow_dag(cfg, config_uri, with_retry=with_retry)
                emit_out = args.get("emit_out")
                written_path: str | None = None
                if emit_out:
                    out_p = Path(str(emit_out))
                    out_p.parent.mkdir(parents=True, exist_ok=True)
                    out_p.write_text(dag_src, encoding="utf-8")
                    written_path = str(out_p)
                return {
                    "mode": "emit_airflow",
                    "run_id": run_id,
                    "with_retry": with_retry,
                    "written_path": written_path,
                    "dag_source": dag_src,
                }

            if mode == "dry_run":
                plan = launch_plan.plan_run(cfg, force=force)
                return {
                    "mode": "dry_run",
                    "run_id": plan.run_id,
                    "python_runtime": plan.python_runtime,
                    "python_models": plan.python_models,
                    "bq_models": plan.bq_models,
                    "fanout": dataclasses.asdict(plan.fanout),
                    "staged": plan.staged,
                    "config_uri": plan.config_uri,
                    "idempotency": dataclasses.asdict(plan.idempotency),
                    "nodes": [dataclasses.asdict(n) for n in plan.nodes],
                    "commands": (
                        {k: dataclasses.asdict(v) for k, v in plan.commands.items()}
                        if plan.commands
                        else None
                    ),
                }

            if mode == "feasibility":
                lines = launch_plan.feasibility_report(cfg)
                return {
                    "mode": "feasibility",
                    "run_id": make_run_id(cfg),
                    "report_lines": list(lines),
                }

            if mode == "quota":
                from .dag import gpu_usefulness_report
                from .quota import report_for_run

                q_lines = report_for_run(cfg)
                gpu_lines = gpu_usefulness_report(cfg, plan_dag(cfg).jobs)
                return {
                    "mode": "quota",
                    "run_id": make_run_id(cfg),
                    "quota_report": list(q_lines),
                    "gpu_usefulness_warnings": list(gpu_lines),
                }

            if mode == "stage_only":
                if not self.allow_launch:
                    return {
                        "allowed": False,
                        "mode": "stage_only",
                        "run_id": make_run_id(cfg),
                        "message": (
                            "stage_only uploads artifacts to GCS and writes a STAGED row to "
                            "BigQuery. Start the MCP server with `--allow-launch` or run: "
                            "python -m scale_forecasting.main --config <path> --stage-only"
                        ),
                    }
                staged = launch_plan.stage_run(cfg, force=force)
                return {
                    "allowed": True,
                    "mode": "stage_only",
                    "run_id": staged.run_id,
                    "config_uri": staged.config_uri,
                    "commands": (
                        {k: dataclasses.asdict(v) for k, v in staged.commands.items()}
                        if staged.commands
                        else None
                    ),
                }

            raise ValueError(f"Unsupported plan_execution mode: {mode!r}")

        if name == "inspect_registry":
            from .errors import require_extra
            from .registry import ops, reads
            from .settings import Settings

            require_extra("gcp", purpose="inspect_registry")
            settings = Settings.resolve()
            action = str(args.get("action", "doctor"))

            if action == "doctor":
                doc = ops.doctor(settings=settings)
                return {
                    "action": "doctor",
                    "registry": doc.registry,
                    "artifact_root": doc.artifact_root,
                    "healthy": doc.healthy,
                    "missing_tables": list(doc.missing_tables),
                    "tables": [dataclasses.asdict(t) for t in doc.tables],
                    "views": list(doc.views),
                    "live_runs": [{"run_id": r_id, "status": r_st} for r_id, r_st in doc.live_runs],
                    "orphans": [dataclasses.asdict(p) for p in doc.orphans],
                    "formatted_report": ops.format_doctor(doc),
                }

            if action == "recent_runs":
                limit = int(args.get("limit", 20))
                rows = reads.read_recent_runs(limit=limit, settings=settings)
                return {
                    "action": "recent_runs",
                    "dataset": settings.registry_dataset_ref,
                    "count": len(rows),
                    "runs": rows,
                }

            if action == "probe_run":
                from .probes.reconcile import probe_run

                run_id = str(args.get("run_id") or "")
                if not run_id:
                    raise ValueError("run_id is required for action='probe_run'")
                probe_rep = probe_run(run_id, job=args.get("job"), settings=settings)
                return {
                    "action": "probe_run",
                    "report": dataclasses.asdict(probe_rep),
                }

            if action == "retry_preview":
                from .retry_run import config_for_run, format_retry_plan, retry_run

                run_id = str(args.get("run_id") or "")
                if not run_id:
                    raise ValueError("run_id is required for action='retry_preview'")
                cfg = config_for_run(run_id, settings=settings)
                rep = retry_run(cfg, confirm=False, settings=settings)
                return {
                    "action": "retry_preview",
                    "run_id": rep.run_id,
                    "executed": rep.executed,
                    "submittable": rep.plan.submittable,
                    "plan": dataclasses.asdict(rep.plan),
                    "formatted_plan": format_retry_plan(rep.plan),
                }

            raise ValueError(f"Unknown inspect_registry action: {action!r}")

        if name == "review_run":
            from . import review
            from .errors import require_extra
            from .settings import Settings

            require_extra("gcp", purpose="review_run")
            settings = Settings.resolve()
            run_id = str(args["run_id"])
            rev = review.review_run(run_id, settings=settings)
            out: dict[str, Any] = {
                "run_id": rev.run_id,
                "status": rev.status,
                "n_series": rev.n_series,
                "decision_metric": rev.decision_metric,
                "best_overall": dataclasses.asdict(rev.best_overall) if rev.best_overall else None,
                "best_per_family": {
                    k: dataclasses.asdict(v) for k, v in rev.best_per_family.items()
                },
                "ensemble_lift": [dataclasses.asdict(e) for e in rev.ensemble_lift],
            }
            if bool(args.get("include_leaderboard", True)):
                out["models"] = [dataclasses.asdict(m) for m in rev.models]
            if bool(args.get("include_calibration", True)):
                cal = review.calibration_report(run_id, settings=settings)
                out["calibration"] = dataclasses.asdict(cal)
            want_fva = bool(args.get("include_fva", False))
            want_decay = bool(args.get("include_horizon_decay", False))
            want_bias = bool(args.get("include_bias_report", False))
            want_exc = bool(args.get("include_exceptions", False))
            want_promo = bool(args.get("include_promo_elasticity", False))
            want_cs = bool(args.get("include_cold_start", False))
            inv_params_rev = args.get("inventory_params")
            want_inv = bool(args.get("include_inventory_policy", False)) or isinstance(
                inv_params_rev, dict
            )
            want_gs = bool(args.get("include_goal_seek", False)) or (
                args.get("target_total") is not None
            )
            want_trec = bool(args.get("reconcile_temporal", False))
            want_aci = bool(args.get("include_aci", False))
            want_traj = bool(args.get("include_trajectories", False))
            want_tail = bool(args.get("include_tail_risk", False))
            want_sim = bool(args.get("include_similarity", False))
            want_subs = bool(args.get("include_substitutes", False))
            want_trends = bool(args.get("include_emerging_trends", False))
            opt_inv_rev = args.get("optimize_inventory_params")
            opt_scen_rev = args.get("optimize_scenario_params")
            opt_clear_rev = args.get("optimize_clearance_params")
            want_clear = bool(args.get("include_clearance", False)) or isinstance(
                opt_clear_rev, dict
            )
            opt_cv_rev = args.get("optimize_contract_volume_params")
            want_cv = bool(args.get("include_contract_volume", False)) or isinstance(
                opt_cv_rev, dict
            )
            rep_path_rev = args.get("export_report_path")
            if (
                want_fva
                or want_decay
                or want_bias
                or want_exc
                or want_promo
                or want_cs
                or want_inv
                or want_gs
                or want_trec
                or want_aci
                or want_traj
                or want_tail
                or want_sim
                or want_subs
                or want_trends
                or want_clear
                or want_cv
                or isinstance(opt_inv_rev, dict)
                or isinstance(opt_scen_rev, dict)
                or rep_path_rev
            ):
                import pandas as pd

                from .aci import compute_aci_df
                from .clearance_optimizer import optimize_clearance, optimize_contract_volume
                from .cold_start import cold_start_transfer_df
                from .copula import compute_tail_risk, sample_trajectories
                from .diagnostics import (
                    compute_bias_report,
                    compute_exceptions_queue,
                    compute_fva,
                    compute_horizon_decay,
                    compute_inventory_policy,
                    compute_promo_elasticity,
                )
                from .goal_seek import allocate_target, summarize_goal_seek
                from .inventory_optimizer import optimize_inventory_portfolio
                from .registry.reads import read_oof, read_predictions
                from .report import export_report
                from .scenario_optimizer import optimize_scenario
                from .similarity import (
                    cluster_by_similarity_df,
                    compute_similarity_matrix,
                    detect_emerging_trends_df,
                    recommend_substitutes_df,
                )
                from .temporal_reconciliation import reconcile_temporal_forecasts

                oof_rows = read_oof(run_id, limit=20000, settings=settings)
                oof_df = pd.DataFrame.from_records(oof_rows) if oof_rows else pd.DataFrame()
                metric = rev.decision_metric or "wape"
                if want_fva:
                    out["fva"] = compute_fva(oof_df, metric=metric).to_dict(orient="records")
                if want_decay:
                    out["horizon_decay"] = compute_horizon_decay(oof_df, metric=metric).to_dict(
                        orient="records"
                    )
                if want_bias:
                    out["bias_report"] = compute_bias_report(oof_df).to_dict(orient="records")
                if want_exc:
                    out["exceptions"] = compute_exceptions_queue(
                        oof_df=oof_df, metric=metric
                    ).to_dict(orient="records")
                if want_promo and not oof_df.empty:
                    hist_promo = oof_df.rename(
                        columns={
                            c: "ds"
                            for c in ("val_date", "forecast_date")
                            if c in oof_df.columns and "ds" not in oof_df.columns
                        }
                    ).copy()
                    if "y" not in hist_promo.columns and "y_true" in hist_promo.columns:
                        hist_promo["y"] = hist_promo["y_true"]
                    out["promo_elasticity"] = compute_promo_elasticity(hist_promo).to_dict(
                        orient="records"
                    )
                pred_df: pd.DataFrame | None = None
                if (
                    want_inv
                    or want_gs
                    or want_trec
                    or want_cs
                    or want_aci
                    or want_traj
                    or want_tail
                    or want_sim
                    or want_subs
                    or want_trends
                    or want_clear
                    or want_cv
                    or isinstance(opt_inv_rev, dict)
                    or isinstance(opt_scen_rev, dict)
                    or rep_path_rev
                ):
                    pred_rows = read_predictions(run_id, limit=20000, settings=settings)
                    pred_df = pd.DataFrame.from_records(pred_rows) if pred_rows else pd.DataFrame()
                if want_clear and pred_df is not None and not pred_df.empty:
                    clear_kw_rev = dict(opt_clear_rev) if isinstance(opt_clear_rev, dict) else {}
                    clear_res_rev = optimize_clearance(
                        pred_df,
                        oof_df=oof_df if not oof_df.empty else None,
                        **clear_kw_rev,
                    )
                    out["clearance_optimization"] = {
                        "status": clear_res_rev.status,
                        "is_feasible": clear_res_rev.is_feasible,
                        "total_profit_lift": clear_res_rev.total_profit_lift,
                        "total_space_freed": clear_res_rev.total_space_freed,
                        "overall_sell_through_pct": clear_res_rev.overall_sell_through_pct,
                        "relaxation_hint": clear_res_rev.relaxation_hint,
                        "summary": clear_res_rev.summary_df.to_dict(orient="records"),
                        "schedule": clear_res_rev.schedule_df.head(20).to_dict(orient="records"),
                        "hierarchy_rollup": (
                            clear_res_rev.hierarchy_rollup_df.to_dict(orient="records")
                            if clear_res_rev.hierarchy_rollup_df is not None
                            else None
                        ),
                    }
                if want_cv and pred_df is not None and not pred_df.empty:
                    cv_kw_rev = dict(opt_cv_rev) if isinstance(opt_cv_rev, dict) else {}
                    cv_res_rev = optimize_contract_volume(
                        pred_df,
                        oof_df=oof_df if not oof_df.empty else None,
                        **cv_kw_rev,
                    )
                    out["contract_volume_optimization"] = {
                        "status": cv_res_rev.status,
                        "is_feasible": cv_res_rev.is_feasible,
                        "baseline_portfolio_units": cv_res_rev.baseline_portfolio_units,
                        "optimal_portfolio_units": cv_res_rev.optimal_portfolio_units,
                        "target_contract_units": cv_res_rev.target_contract_units,
                        "achieved_tier_units": cv_res_rev.achieved_tier_units,
                        "rebate_earned": cv_res_rev.rebate_earned,
                        "shortfall_units": cv_res_rev.shortfall_units,
                        "cvar_shortfall_units": cv_res_rev.cvar_shortfall_units,
                        "net_value_lift": cv_res_rev.net_value_lift,
                        "relaxation_hint": cv_res_rev.relaxation_hint,
                        "summary": cv_res_rev.summary_df.to_dict(orient="records"),
                        "schedule": cv_res_rev.schedule_df.head(20).to_dict(orient="records"),
                        "hierarchy_rollup": (
                            cv_res_rev.hierarchy_rollup_df.to_dict(orient="records")
                            if cv_res_rev.hierarchy_rollup_df is not None
                            else None
                        ),
                    }
                if want_sim or want_subs or want_trends:
                    sim_m_rev = str(args.get("similarity_method", "hybrid"))
                    sim_src = (
                        oof_df.rename(columns={"val_date": "ds", "y_true": "y"})
                        if not oof_df.empty
                        else (
                            pred_df.rename(columns={"forecast_date": "ds", "yhat": "y"})
                            if pred_df is not None and not pred_df.empty
                            else pd.DataFrame()
                        )
                    )
                    if not sim_src.empty:
                        rev_text_cols = (
                            [str(x) for x in args["text_cols"]]
                            if isinstance(args.get("text_cols"), list)
                            else None
                        )
                        rev_attr_cols = (
                            [str(x) for x in args["attribute_cols"]]
                            if isinstance(args.get("attribute_cols"), list)
                            else None
                        )
                        rev_hier_cols = (
                            [str(x) for x in args["hierarchy_cols"]]
                            if isinstance(args.get("hierarchy_cols"), list)
                            else None
                        )
                        sim_res_rev = None
                        if want_sim:
                            sim_res_rev = compute_similarity_matrix(
                                sim_src,
                                method=sim_m_rev,
                                text_cols=rev_text_cols,
                                attribute_cols=rev_attr_cols,
                                hierarchy_cols=rev_hier_cols,
                            )
                            sim_cl_rev = cluster_by_similarity_df(
                                sim_src, method=sim_m_rev, similarity_result=sim_res_rev
                            )
                            out["similarity"] = {
                                "method": sim_m_rev,
                                "pairs": sim_res_rev.pairs_df.head(20).to_dict(orient="records"),
                                "clusters": sim_cl_rev.to_dict(orient="records"),
                            }
                        if want_subs:
                            inv_recs_rev = args.get("inventory_records")
                            inv_df_rev = (
                                pd.DataFrame.from_records(inv_recs_rev)
                                if isinstance(inv_recs_rev, list) and inv_recs_rev
                                else None
                            )
                            out["substitutes"] = recommend_substitutes_df(
                                sim_src,
                                inventory_df=inv_df_rev,
                                method=sim_m_rev,
                                text_cols=rev_text_cols,
                                attribute_cols=rev_attr_cols,
                                hierarchy_cols=rev_hier_cols,
                                within_hierarchy_level=str(args["within_hierarchy_level"])
                                if args.get("within_hierarchy_level")
                                else None,
                                similarity_result=sim_res_rev,
                            ).to_dict(orient="records")
                        if want_trends:
                            out["emerging_trends"] = detect_emerging_trends_df(
                                sim_src,
                                method=sim_m_rev,
                                similarity_result=sim_res_rev,
                            ).to_dict(orient="records")
                if want_cs and pred_df is not None and not pred_df.empty:
                    hist_cs = (
                        oof_df.rename(
                            columns={
                                "val_date": "ds",
                                "y_true": "y",
                            }
                        )
                        if not oof_df.empty
                        else pred_df.rename(columns={"forecast_date": "ds", "yhat": "y"})
                    )
                    cs_rev = cold_start_transfer_df(
                        hist_cs,
                        oof_df=oof_df if not oof_df.empty else None,
                    )
                    assert isinstance(cs_rev, pd.DataFrame)
                    out["cold_start_transfer"] = cs_rev.head(20).to_dict(orient="records")
                if want_inv and pred_df is not None:
                    inv_kw_rev = dict(inv_params_rev) if isinstance(inv_params_rev, dict) else {}
                    out["inventory_policy"] = compute_inventory_policy(
                        pred_df,
                        oof_df=oof_df if not oof_df.empty else None,
                        **inv_kw_rev,
                    ).to_dict(orient="records")
                if isinstance(opt_inv_rev, dict) and pred_df is not None and not pred_df.empty:
                    inv_opt_res = optimize_inventory_portfolio(
                        pred_df,
                        oof_df=oof_df if not oof_df.empty else None,
                        **opt_inv_rev,
                    )
                    assert isinstance(inv_opt_res, pd.DataFrame)
                    out["inventory_optimization"] = inv_opt_res.to_dict(orient="records")
                if isinstance(opt_scen_rev, dict) and pred_df is not None and not pred_df.empty:
                    scen_opt_rev = optimize_scenario(
                        pred_df,
                        oof_df=oof_df if not oof_df.empty else None,
                        **opt_scen_rev,
                    )
                    scen_df_rev = (
                        scen_opt_rev.schedule_df
                        if hasattr(scen_opt_rev, "schedule_df")
                        else scen_opt_rev
                    )
                    out["scenario_optimization"] = scen_df_rev.head(20).to_dict(orient="records")
                if want_aci and pred_df is not None and not pred_df.empty:
                    aci_res_rev = compute_aci_df(
                        oof_df if not oof_df.empty else pred_df,
                        predictions_df=pred_df,
                    )
                    assert isinstance(aci_res_rev, pd.DataFrame)
                    out["aci_coverage"] = aci_res_rev.head(20).to_dict(orient="records")
                if (want_traj or want_tail) and pred_df is not None and not pred_df.empty:
                    cop_rev = str(args.get("copula_type", "gaussian"))
                    traj_rev = sample_trajectories(
                        pred_df,
                        oof_df=oof_df if not oof_df.empty else None,
                        copula_type=cop_rev,
                        n_paths=100,
                    )
                    assert isinstance(traj_rev, pd.DataFrame)
                    if want_traj:
                        out["trajectories_sample"] = traj_rev.head(20).to_dict(orient="records")
                    if want_tail:
                        tail_rev = compute_tail_risk(traj_rev)
                        assert isinstance(tail_rev, pd.DataFrame)
                        out["tail_risk"] = tail_rev.to_dict(orient="records")
                if want_trec and pred_df is not None and not pred_df.empty:
                    t_method_rev = str(args.get("temporal_reconciliation_method", "wls_struct"))
                    trec_rev = reconcile_temporal_forecasts(
                        pred_df,
                        oof_df=oof_df if not oof_df.empty else None,
                        method=t_method_rev,
                    )
                    assert isinstance(trec_rev, pd.DataFrame)
                    out["temporal_reconciled_predictions"] = trec_rev.head(20).to_dict(
                        orient="records"
                    )
                gs_df_rev: pd.DataFrame | None = None
                if want_gs and pred_df is not None and not pred_df.empty:
                    raw_target_rev = args.get("target_total")
                    eff_target_rev = (
                        raw_target_rev
                        if raw_target_rev is not None
                        else float(pd.to_numeric(pred_df["yhat"], errors="coerce").sum() * 1.15)
                    )
                    gs_res_rev = allocate_target(
                        pred_df,
                        target_total=eff_target_rev,
                        oof_df=oof_df if not oof_df.empty else None,
                    )
                    assert isinstance(gs_res_rev, pd.DataFrame)
                    gs_df_rev = gs_res_rev
                    out["goal_seek"] = gs_df_rev.to_dict(orient="records")
                    out["goal_seek_summary"] = summarize_goal_seek(gs_df_rev).to_dict(
                        orient="records"
                    )
                if rep_path_rev and pred_df is not None:
                    written_r = export_report(
                        {"run_id": run_id, "oof_df": oof_df, "predictions_df": pred_df},
                        str(rep_path_rev),
                        metric=metric,
                        goal_seek_df=gs_df_rev,
                    )
                    out["report_path"] = str(written_r)
            return out

        if name == "launch_or_repair_run":
            action = str(args["action"])
            if not self.allow_launch:
                return {
                    "allowed": False,
                    "action": action,
                    "message": (
                        "Cloud job submission and mutating registry repairs are locked by default. "
                        "Restart the MCP server with `python -m scale_forecasting.mcp "
                        "--allow-launch` or run the corresponding CLI command directly."
                    ),
                }
            from . import main as main_mod
            from .errors import require_extra
            from .probes.cancel import cancel_run
            from .probes.settle import settle_run
            from .registry import ops
            from .retry_run import config_for_run, retry_run
            from .settings import Settings

            require_extra("gcp", purpose=f"launch_or_repair_run ({action})")
            settings = Settings.resolve()
            force = bool(args.get("force", False))
            reason = str(args.get("reason", ""))

            if action == "run":
                if "config" not in args or args["config"] is None:
                    raise ValueError("config is required for action='run'")
                cfg = _parse_run_config_input(args["config"], self.repo_root)
                run_id = main_mod.run(
                    cfg,
                    settings=settings,
                    force=force,
                    n_series=args.get("n_series"),
                )
                return {"allowed": True, "action": "run", "run_id": run_id}

            if action == "retry":
                if args.get("config") is not None:
                    cfg = _parse_run_config_input(args["config"], self.repo_root)
                elif args.get("run_id"):
                    cfg = config_for_run(str(args["run_id"]), settings=settings)
                else:
                    raise ValueError("Either config or run_id is required for action='retry'")
                rep = retry_run(cfg, confirm=True, reason=reason, settings=settings)
                return {
                    "allowed": True,
                    "action": "retry",
                    "run_id": rep.run_id,
                    "executed": rep.executed,
                    "plan": dataclasses.asdict(rep.plan),
                    "outcome": dataclasses.asdict(rep.outcome) if rep.outcome else None,
                }

            if action == "settle":
                run_id = str(args.get("run_id") or "")
                if not run_id and args.get("config") is not None:
                    run_id = make_run_id(_parse_run_config_input(args["config"], self.repo_root))
                if not run_id:
                    raise ValueError("Either run_id or config is required for action='settle'")
                s_rep = settle_run(
                    run_id, job=args.get("job"), yes=True, reason=reason, settings=settings
                )
                return {
                    "allowed": True,
                    "action": "settle",
                    "report": dataclasses.asdict(s_rep),
                }

            if action == "cancel":
                run_id = str(args.get("run_id") or "")
                if not run_id and args.get("config") is not None:
                    run_id = make_run_id(_parse_run_config_input(args["config"], self.repo_root))
                if not run_id:
                    raise ValueError("Either run_id or config is required for action='cancel'")
                c_rep = cancel_run(
                    run_id, job=args.get("job"), confirm=True, reason=reason, settings=settings
                )
                return {
                    "allowed": True,
                    "action": "cancel",
                    "report": dataclasses.asdict(c_rep),
                }

            if action == "close_runs":
                closed = ops.close_runs(yes=True, settings=settings)
                return {
                    "allowed": True,
                    "action": "close_runs",
                    "plan": dataclasses.asdict(closed),
                }

            raise ValueError(f"Unknown launch_or_repair_run action: {action!r}")

        if name == "build_auto_config":
            import pandas as pd

            from . import playground
            from .auto import build_config as build_auto_cfg
            from .benchmarks import BENCHMARK_NAMES, load_benchmark

            src_arg = str(
                args.get("benchmark")
                or args.get("source_table")
                or args.get("source", "playground")
            )
            n_series_bp = int(args.get("n_series", 4))
            horizon_auto = int(args.get("horizon", 14))
            tier_auto = str(args.get("tier", "wide_field"))
            where_auto = str(args.get("where", "auto"))
            goal_auto = str(args.get("goal", "accuracy"))
            max_wc = args.get("max_minutes") or args.get("max_wall_clock_minutes")
            max_usd_auto = args.get("max_usd")
            only_avail = bool(args.get("only_available_models", True))

            data_input: Any
            if src_arg == "playground":
                data_input = playground.sample_data(n_series=n_series_bp, with_exog=True)
            elif src_arg in BENCHMARK_NAMES:
                ds_bench = load_benchmark(src_arg, n_series=n_series_bp)
                data_input = ds_bench.panel_df
                if "horizon" not in args:
                    horizon_auto = ds_bench.spec.horizon
            else:
                cand_path = Path(src_arg)
                if not cand_path.is_file():
                    cand_path = self.repo_root / src_arg
                if cand_path.is_file():
                    data_input = (
                        pd.read_parquet(cand_path)
                        if cand_path.suffix.lower() == ".parquet"
                        else pd.read_csv(cand_path)
                    )
                else:
                    data_input = src_arg

            bp = build_auto_cfg(
                data_input,
                horizon=horizon_auto,
                tier=tier_auto,
                where=where_auto,
                goal=goal_auto,
                baselines=args.get("baselines", "auto"),
                holidays=args.get("holidays"),
                decision_metric=args.get("decision_metric"),
                n_folds=int(args["n_folds"]) if args.get("n_folds") is not None else None,
                allowed_families=args.get("allowed_families"),
                models=args.get("models"),
                include_models=args.get("include_models"),
                exclude_models=args.get("exclude_models"),
                max_minutes=float(max_wc) if max_wc is not None else None,
                max_usd=float(max_usd_auto) if max_usd_auto is not None else None,
                only_available_models=only_avail,
            )
            written_cfg_path: str | None = None
            if args.get("write_config_path"):
                written_cfg_path = str(bp.to_file(str(args["write_config_path"])))
            emitted_airflow_path: str | None = None
            if args.get("emit_airflow_path"):
                emitted_airflow_path = str(args["emit_airflow_path"])
                bp.emit_airflow_dag(
                    written_cfg_path,
                    out_path=emitted_airflow_path,
                    autofocus=bool(args.get("autofocus", False)),
                )
            return {
                "run_id": bp.run_id,
                "tier": bp.tier,
                "where": bp.where,
                "goal": bp.goal,
                "estimated_minutes_p50": bp.estimated_minutes_p50,
                "estimated_minutes_p90": bp.estimated_minutes_p90,
                "duration_estimate": bp.duration_estimate.to_dict(),
                "profile_table": bp.profile_df().to_dict(orient="records"),
                "rationale": dict(bp.rationale),
                "explain": bp.explain(),
                "written_config_path": written_cfg_path,
                "emitted_airflow_path": emitted_airflow_path,
                "config": bp.config.model_dump(mode="json"),
            }

        if name in ("run_auto_forecast", "auto_forecast"):
            import pandas as pd

            from . import playground
            from .benchmarks import BENCHMARK_NAMES, load_benchmark
            from .sdk import Forecaster

            where_run = str(args.get("where", "local"))
            if (
                args.get("source_table") or where_run not in ("local", "auto")
            ) and not self.allow_launch:
                return {
                    "allowed": False,
                    "message": (
                        "Cloud auto-forecast execution is locked by default. Start the MCP "
                        "server with `python -m scale_forecasting.mcp --allow-launch` to submit "
                        "cloud runs."
                    ),
                }

            src_run = str(
                args.get("benchmark")
                or args.get("source_table")
                or args.get("source", "playground")
            )
            horizon_run = int(args.get("horizon", 14))
            tier_run = str(args.get("tier", "wide_field"))
            goal_run = str(args.get("goal", "accuracy"))
            autofocus_run = bool(args.get("autofocus", True))
            autofocus_mode_run = str(args.get("autofocus_mode", "dag"))
            max_passes_Arg = args.get("max_passes")
            max_af_passes_run = int(args.get("max_autofocus_passes", 1))
            n_series_run = int(args.get("n_series", 3))

            panel_run: Any
            fut_run: pd.DataFrame | None = None
            if src_run == "playground":
                panel_run = playground.sample_data(n_series=n_series_run, with_exog=True)
            elif src_run in BENCHMARK_NAMES:
                ds_b = load_benchmark(src_run, n_series=n_series_run)
                panel_run = ds_b.df
                fut_run = ds_b.future_df if not ds_b.future_df.empty else None
                if "horizon" not in args:
                    horizon_run = ds_b.spec.horizon
            elif args.get("source_table"):
                panel_run = str(args["source_table"])
            else:
                c_path = Path(src_run)
                if not c_path.is_file():
                    c_path = self.repo_root / src_run
                if not c_path.is_file():
                    raise FileNotFoundError(
                        f"Offline source {src_run!r} not found; pass 'playground', a benchmark "
                        f"in {list(BENCHMARK_NAMES)}, or a valid CSV/Parquet file path."
                    )
                panel_run = (
                    pd.read_parquet(c_path)
                    if c_path.suffix.lower() == ".parquet"
                    else pd.read_csv(c_path)
                )

            evolve_target = args.get("evolve_target")
            evolve_from_run_ids = args.get("evolve_from_run_ids")
            if evolve_target or evolve_from_run_ids:
                evolve_backend = str(args.get("evolve_backend", "alphaevolve"))
                evolve_max_evals = int(args.get("evolve_max_evaluations", 10))
                evolve_model_name = str(args.get("evolve_model_name", "gemini-3.8-flash"))
                eff_target = str(evolve_target or "ensemble")
                ev_res = Forecaster.evolve(
                    None if evolve_from_run_ids else panel_run,
                    from_run_ids=list(evolve_from_run_ids) if evolve_from_run_ids else None,
                    target=eff_target,
                    backend=evolve_backend,
                    tier=tier_run,
                    where=where_run,
                    horizon=horizon_run,
                    max_evaluations=evolve_max_evals,
                    model_name=evolve_model_name,
                )
                lb_records = (
                    ev_res.leaderboard_df.to_dict(orient="records")
                    if hasattr(ev_res, "leaderboard_df") and ev_res.leaderboard_df is not None
                    else []
                )
                return {
                    "target": ev_res.target,
                    "backend_used": ev_res.backend_used,
                    "model_name": getattr(ev_res, "model_name", evolve_model_name),
                    "promoted": ev_res.promoted,
                    "rejection_reason": ev_res.rejection_reason,
                    "baseline_score": ev_res.baseline_score,
                    "inner_cv_score": ev_res.inner_cv_score,
                    "lockdown_baseline_score": ev_res.lockdown_baseline_score,
                    "lockdown_candidate_score": ev_res.lockdown_candidate_score,
                    "fva_pct": ev_res.fva_pct,
                    "dm_pvalue": ev_res.dm_pvalue,
                    "worst_decile_fva_pct": ev_res.worst_decile_fva_pct,
                    "champion_code": ev_res.champion_code,
                    "export_path": ev_res.export_path,
                    "diagnostics": getattr(ev_res, "diagnostics", {}),
                    "leaderboard": lb_records,
                }

            f_res = Forecaster.auto(
                panel_run,
                future_df=fut_run,
                horizon=horizon_run,
                tier=tier_run,
                where=where_run,
                goal=goal_run,
                baselines=args.get("baselines", "auto"),
                holidays=args.get("holidays"),
                decision_metric=args.get("decision_metric"),
                n_folds=int(args["n_folds"]) if args.get("n_folds") is not None else None,
                allowed_families=args.get("allowed_families"),
                models=args.get("models"),
                include_models=args.get("include_models"),
                exclude_models=args.get("exclude_models"),
                autofocus=autofocus_run,
                autofocus_mode=autofocus_mode_run,
                max_passes=int(max_passes_Arg) if max_passes_Arg is not None else None,
                max_autofocus_passes=max_af_passes_run if max_passes_Arg is None else None,
                execute=True,
            )
            sess = f_res.auto_session
            assert sess is not None
            preview = f_res.preview_next_loop()
            report_written: str | None = None
            if args.get("export_report_path"):
                report_written = str(f_res.export_report(str(args["export_report_path"])))
            ledger_records = f_res.autofocus_ledger_df().to_dict(orient="records")
            baseline_records = f_res.baseline_value_df().to_dict(orient="records")
            hierarchy_records = f_res.hierarchy_value_df().to_dict(orient="records")
            assess_records = f_res.practitioner_assessment_df().to_dict(orient="records")
            compare_records = f_res.compare_models().to_dict(orient="records")
            return {
                "run_id": sess.run_id,
                "tier": sess.blueprint.tier,
                "best_model": sess.best_model,
                "best_metric": sess.best_metric,
                "decision_metric": sess.decision_metric,
                "profile_table": f_res.profile_df().to_dict(orient="records"),
                "autofocus_ledger": ledger_records,
                "autofocus_passes": [dataclasses.asdict(p) for p in sess.passes],
                "baseline_value": baseline_records,
                "hierarchy_value": hierarchy_records,
                "practitioner_assessment": assess_records,
                "model_comparison": compare_records,
                "next_loop_preview": preview.to_dict(),
                "leaderboard": sess.leaderboard_df.to_dict(orient="records"),
                "sample_predictions": sess.predictions_df.head(10).to_dict(orient="records"),
                "report_path": report_written,
            }

        if name == "promo_elasticity":
            import pandas as pd

            from .elasticity import (
                PROMO_ELASTICITY_COLUMNS,
                compute_promo_elasticity_decomposition,
                make_confounded_promo_dgp,
                summarize_promo_elasticity,
            )

            records_arg = args.get("records")
            if isinstance(records_arg, list) and records_arg:
                el_input = pd.DataFrame.from_records(records_arg)
            else:
                el_input, _ = make_confounded_promo_dgp(
                    n_series=int(args.get("n_series", 2)),
                    n_days=int(args.get("n_days", 120)),
                )
            decomp_df = compute_promo_elasticity_decomposition(
                el_input,
                promo_col=args.get("promo_col"),
                price_col=args.get("price_col"),
            )
            sum_df = summarize_promo_elasticity(decomp_df)
            return {
                "status": "ok",
                "columns": list(PROMO_ELASTICITY_COLUMNS),
                "summary": sum_df.to_dict(orient="records"),
                "sample_rows": decomp_df.head(10).to_dict(orient="records"),
            }

        if name == "simulate_scenarios":
            import pandas as pd

            from . import playground
            from .scenarios import simulate_scenarios, summarize_scenarios

            records_arg = args.get("records")
            if isinstance(records_arg, list) and records_arg:
                sim_panel = pd.DataFrame.from_records(records_arg)
            else:
                sim_panel = playground.sample_data(
                    n_series=int(args.get("n_series", 2)),
                    with_exog=True,
                )
            models_arg = args.get("models") or ["regression_lags@monotone"]
            if args.get("config"):
                sim_cfg = _parse_run_config_input(args["config"], self.repo_root)
            else:
                covs = [
                    c
                    for c in ("price", "promo", "price_index", "promo_flag")
                    if c in sim_panel.columns
                ]
                sim_cfg = RunConfig.model_validate(
                    {
                        "run_name": "mcp-scenarios",
                        "data": {
                            "source_table": "playground",
                            "horizon": int(args.get("horizon", 7)),
                            "freq": str(args.get("freq", "D")),
                        },
                        "features": {"future_covariates": covs},
                        "models": list(models_arg),
                        "ensemble": {"enabled": len(models_arg) >= 2},
                    }
                )
            scen_map = args.get("scenarios") or {
                "price_cut_10pct": {
                    ("price" if "price" in sim_panel.columns else "price_index"): "-10%"
                }
            }
            sim_df = simulate_scenarios(
                sim_panel,
                sim_cfg,
                scen_map,
                models=list(models_arg),
                price_col=args.get("price_col"),
                unit_cost=args.get("unit_cost"),
                include_inventory_impact=bool(args.get("include_inventory_impact", False)),
            )
            return {
                "status": "ok",
                "summary": summarize_scenarios(sim_df).to_dict(orient="records"),
                "sample_rows": sim_df.head(10).to_dict(orient="records"),
            }

        if name in ("allocate_target", "allocate_forecast_target"):
            import pandas as pd

            from . import playground
            from .goal_seek import GOAL_SEEK_COLUMNS, allocate_target, summarize_goal_seek

            records_gs = args.get("records") or args.get("predictions")
            if isinstance(records_gs, list) and records_gs:
                pred_gs = pd.DataFrame.from_records(records_gs)
                oof_gs = (
                    pd.DataFrame.from_records(args["oof"])
                    if isinstance(args.get("oof"), list) and args["oof"]
                    else None
                )
            else:
                pg_gs = playground.run_model(
                    str(args.get("model", "holtwinters")),
                    horizon=int(args.get("horizon", 7)),
                    backtest=True,
                )
                pred_gs = pg_gs.predictions.assign(
                    ts_id=pg_gs.result.ts_id, model_type=pg_gs.result.model_type
                )
                oof_gs = (
                    pg_gs.oof.assign(ts_id=pg_gs.result.ts_id, model_type=pg_gs.result.model_type)
                    if pg_gs.oof is not None and not pg_gs.oof.empty
                    else None
                )
            target_gs = args.get("target_total")
            if target_gs is None:
                target_gs = float(pd.to_numeric(pred_gs["yhat"], errors="coerce").sum() * 1.15)
            gs_out = allocate_target(
                pred_gs,
                target_total=target_gs,
                oof_df=oof_gs,
                weights=str(args.get("weights", "inverse_variance")),
                locked_nodes=args.get("locked_nodes") or args.get("locked_series"),
                bounds=args.get("bounds"),
                nonneg=bool(args.get("nonneg", args.get("non_negative", True))),
            )
            assert isinstance(gs_out, pd.DataFrame)
            return {
                "status": "ok",
                "columns": list(GOAL_SEEK_COLUMNS),
                "summary": summarize_goal_seek(gs_out).to_dict(orient="records"),
                "allocations": gs_out.to_dict(orient="records"),
            }

        if name == "temporal_reconcile":
            import pandas as pd

            from . import playground
            from .temporal_reconciliation import reconcile_temporal_forecasts

            records_tr = args.get("records") or args.get("predictions")
            method_tr = str(args.get("method", "wls_struct"))
            freq_tr = str(args.get("freq", "D"))
            if isinstance(records_tr, list) and records_tr:
                pred_tr = pd.DataFrame.from_records(records_tr)
                oof_tr = (
                    pd.DataFrame.from_records(args["oof"])
                    if isinstance(args.get("oof"), list) and args["oof"]
                    else None
                )
                trec_out = reconcile_temporal_forecasts(
                    pred_tr,
                    oof_df=oof_tr,
                    freq=freq_tr,
                    method=method_tr,
                    temporal_aggregations=args.get("aggregations"),
                )
            else:
                pg_tr = playground.run_model(
                    str(args.get("model", "holtwinters")),
                    horizon=int(args.get("horizon", 14)),
                    backtest=True,
                )
                trec_out = pg_tr.temporal_reconcile_df(
                    method=method_tr,
                    aggregations=args.get("aggregations"),
                )
            assert isinstance(trec_out, pd.DataFrame)
            return {
                "status": "ok",
                "method": method_tr,
                "rows": trec_out.to_dict(orient="records"),
            }

        if name in ("optimize_inventory", "sf_optimize_inventory", "optimize_inventory_df"):
            import pandas as pd

            from . import playground
            from .inventory_optimizer import optimize_inventory_portfolio

            records_inv = args.get("records") or args.get("predictions")
            inv_kwargs = {
                k: v
                for k, v in args.items()
                if k
                not in (
                    "records",
                    "predictions",
                    "oof",
                    "trajectories",
                    "model",
                    "horizon",
                )
            }
            if isinstance(records_inv, list) and records_inv:
                pred_inv = pd.DataFrame.from_records(records_inv)
                oof_inv = (
                    pd.DataFrame.from_records(args["oof"])
                    if isinstance(args.get("oof"), list) and args["oof"]
                    else None
                )
                traj_inv = (
                    pd.DataFrame.from_records(args["trajectories"])
                    if isinstance(args.get("trajectories"), list) and args["trajectories"]
                    else None
                )
                inv_df_out = optimize_inventory_portfolio(
                    pred_inv,
                    oof_df=oof_inv,
                    trajectories_df=traj_inv,
                    **inv_kwargs,
                )
            else:
                pg_inv = playground.run_model(
                    str(args.get("model", "holtwinters")),
                    horizon=int(args.get("horizon", 14)),
                    backtest=True,
                )
                inv_df_out = pg_inv.optimize_inventory_df(**inv_kwargs)
            assert isinstance(inv_df_out, pd.DataFrame)
            return {
                "status": "ok",
                "rows": inv_df_out.to_dict(orient="records"),
            }

        if name in ("optimize_scenario", "sf_optimize_scenario"):
            import pandas as pd

            from . import playground
            from .scenario_optimizer import optimize_scenario

            records_scen = args.get("records") or args.get("predictions")
            scen_kwargs = {
                k: v
                for k, v in args.items()
                if k
                not in (
                    "records",
                    "predictions",
                    "history",
                    "oof",
                    "model",
                    "horizon",
                )
            }
            if isinstance(records_scen, list) and records_scen:
                pred_scen = pd.DataFrame.from_records(records_scen)
                hist_scen = (
                    pd.DataFrame.from_records(args["history"])
                    if isinstance(args.get("history"), list) and args["history"]
                    else None
                )
                oof_scen = (
                    pd.DataFrame.from_records(args["oof"])
                    if isinstance(args.get("oof"), list) and args["oof"]
                    else None
                )
                scen_res_out = optimize_scenario(
                    pred_scen,
                    history_df=hist_scen,
                    oof_df=oof_scen,
                    **scen_kwargs,
                )
            else:
                pg_scen = playground.run_model(
                    str(args.get("model", "holtwinters")),
                    horizon=int(args.get("horizon", 14)),
                    backtest=True,
                )
                scen_res_out = pg_scen.optimize_scenario(**scen_kwargs)
            sched_df: pd.DataFrame = (
                scen_res_out.schedule_df
                if hasattr(scen_res_out, "schedule_df")
                else pd.DataFrame(scen_res_out)
            )
            return {
                "status": "ok",
                "objective": getattr(
                    scen_res_out, "objective", scen_kwargs.get("objective", "gross_profit")
                ),
                "optimal_total_gross_margin": getattr(
                    scen_res_out, "optimal_total_gross_margin", None
                ),
                "optimal_total_revenue": getattr(scen_res_out, "optimal_total_revenue", None),
                "optimal_total_units": getattr(scen_res_out, "optimal_total_units", None),
                "rows": sched_df.to_dict(orient="records"),
            }

        if name in ("compute_time_series_similarity", "sf_compute_time_series_similarity"):
            src_arg_sim = args.get("source_path_or_table") or args.get("source")
            return compute_time_series_similarity(
                src_arg_sim,
                preset=str(args.get("preset", "retail_daily")),
                method=str(args.get("method", "hybrid")),
                query_ts_id=args.get("query_ts_id"),
                mode=str(args.get("mode", "full_history")),
                n_clusters=args.get("n_clusters", "auto"),
                top_k=int(args.get("top_k", 5)),
                max_lag=int(args.get("max_lag", 0)),
                window=int(args.get("window", 14)),
                include_substitutes=bool(args.get("include_substitutes", False)),
                include_emerging_trends=bool(args.get("include_emerging_trends", False)),
                text_cols=args.get("text_cols"),
                attribute_cols=args.get("attribute_cols"),
                hierarchy_cols=args.get("hierarchy_cols"),
                within_hierarchy_level=args.get("within_hierarchy_level"),
                inventory_records=args.get("inventory_records"),
                require_in_stock=bool(args.get("require_in_stock", True)),
                ts_id_col=str(args.get("ts_id_col", "ts_id")),
                date_col=str(args.get("date_col", "ds")),
                target_col=str(args.get("target_col", "y")),
                freq=str(args.get("freq", "D")),
                project_id=args.get("project_id"),
                records=args.get("records"),
                n_series=int(args.get("n_series", 4)),
                repo_root=self.repo_root,
            )

        if name in (
            "optimize_inventory_clearance",
            "sf_optimize_inventory_clearance",
            "optimize_clearance_and_contracts",
            "optimize_clearance",
            "optimize_contract_volume",
            "optimize_assortment",
            "optimize_prepacks",
            "optimize_rebalancing",
        ):
            import pandas as pd

            from . import playground
            from .clearance_optimizer import (
                optimize_assortment,
                optimize_clearance,
                optimize_contract_volume,
            )
            from .inventory_optimizer import optimize_prepacks, optimize_rebalancing

            if name == "optimize_contract_volume":
                eff_mode = "contract_volume"
            elif name == "optimize_assortment":
                eff_mode = "assortment"
            elif name == "optimize_prepacks":
                eff_mode = "prepacks"
            elif name == "optimize_rebalancing":
                eff_mode = "rebalancing"
            else:
                eff_mode = str(args.get("mode", "clearance"))
            records_cl = args.get("records") or args.get("predictions")
            hist_cl_raw = args.get("history")
            hist_cl = (
                pd.DataFrame.from_records(hist_cl_raw)
                if isinstance(hist_cl_raw, list) and hist_cl_raw
                else None
            )
            oof_cl_raw = args.get("oof")
            oof_cl = (
                pd.DataFrame.from_records(oof_cl_raw)
                if isinstance(oof_cl_raw, list) and oof_cl_raw
                else None
            )
            traj_cl_raw = args.get("trajectories")
            traj_cl = (
                pd.DataFrame.from_records(traj_cl_raw)
                if isinstance(traj_cl_raw, list) and traj_cl_raw
                else None
            )
            elast_cl_raw = args.get("elasticity_records")
            elast_cl = (
                pd.DataFrame.from_records(elast_cl_raw)
                if isinstance(elast_cl_raw, list) and elast_cl_raw
                else None
            )
            if isinstance(records_cl, list) and records_cl:
                input_df = pd.DataFrame.from_records(records_cl)
            else:
                input_df = playground.sample_data(
                    n_series=int(args.get("n_series", 3)),
                    with_exog=True,
                )
                if hist_cl is None:
                    hist_cl = input_df

            common_kw: dict[str, Any] = {
                "history_df": hist_cl,
                "oof_df": oof_cl,
                "elasticity_df": elast_cl,
                "covariate_cols": args.get("covariate_cols"),
                "model_type": args.get("model_type"),
                "reconciliation_method": args.get("reconciliation_method"),
                "prefer_reconciled": bool(args.get("prefer_reconciled", True)),
                "hierarchy_cols": args.get("hierarchy_cols"),
                "base_price": args.get("base_price", 20.0),
                "unit_cost": args.get("unit_cost", 8.0),
                "initial_inventory": args.get("initial_inventory"),
                "price_elasticity": args.get("price_elasticity"),
                "use_digital_twin_elasticity": bool(args.get("use_digital_twin_elasticity", True)),
                "horizon": int(args.get("horizon", 14)),
            }
            if args.get("discount_ladder") is not None:
                common_kw["discount_ladder"] = list(args["discount_ladder"])

            if eff_mode == "assortment":
                as_res = optimize_assortment(
                    input_df,
                    history_df=hist_cl,
                    oof_df=oof_cl,
                    max_skus=(int(args["max_skus"]) if args.get("max_skus") is not None else None),
                    max_shelf_space=(
                        float(args["max_shelf_space"])
                        if args.get("max_shelf_space") is not None
                        else None
                    ),
                    base_price=args.get("base_price", 20.0),
                    unit_cost=args.get("unit_cost", 8.0),
                    walk_rate=args.get("walk_rate"),
                    hierarchy_cols=args.get("hierarchy_cols"),
                    model_type=args.get("model_type"),
                    reconciliation_method=args.get("reconciliation_method"),
                    prefer_reconciled=bool(args.get("prefer_reconciled", True)),
                    horizon=int(args.get("horizon", 14)),
                )
                return {
                    "mode": "assortment",
                    "status": as_res.status,
                    "is_feasible": as_res.is_feasible,
                    "total_candidate_skus": as_res.total_candidate_skus,
                    "retained_skus": as_res.retained_skus,
                    "delisted_skus": as_res.delisted_skus,
                    "baseline_category_units": as_res.baseline_category_units,
                    "post_assortment_units": as_res.post_assortment_units,
                    "transferred_units": as_res.transferred_units,
                    "lost_walk_units": as_res.lost_walk_units,
                    "category_retention_pct": as_res.category_retention_pct,
                    "mean_walk_rate": as_res.mean_walk_rate,
                    "baseline_net_profit": as_res.baseline_net_profit,
                    "optimal_net_profit": as_res.optimal_net_profit,
                    "net_profit_lift": as_res.net_profit_lift,
                    "shelf_space_used": as_res.shelf_space_used,
                    "relaxation_hint": as_res.relaxation_hint,
                    "assortment": as_res.assortment_df.to_dict(orient="records"),
                    "transference": as_res.transference_df.to_dict(orient="records"),
                }

            if eff_mode == "prepacks":
                pp_res = optimize_prepacks(
                    input_df,
                    history_df=hist_cl,
                    prepacks=args.get("prepacks"),
                    initial_inventory=args.get("initial_inventory", 0.0) or 0.0,
                    unit_cost=float(args.get("unit_cost", 1.0) or 1.0),
                )
                return {
                    "mode": "prepacks",
                    "status": pp_res.status,
                    "is_feasible": pp_res.is_feasible,
                    "total_demand_units": pp_res.total_demand_units,
                    "total_allocated_units": pp_res.total_allocated_units,
                    "total_under_units": pp_res.total_under_units,
                    "total_over_units": pp_res.total_over_units,
                    "fill_rate": pp_res.fill_rate,
                    "size_curve_l1_error": pp_res.size_curve_l1_error,
                    "total_cost": pp_res.total_cost,
                    "relaxation_hint": pp_res.relaxation_hint,
                    "allocation": pp_res.allocation_df.to_dict(orient="records"),
                    "pack_orders": pp_res.pack_orders_df.to_dict(orient="records"),
                    "size_curves": pp_res.size_curves_df.to_dict(orient="records"),
                }

            if eff_mode == "rebalancing":
                rb_res = optimize_rebalancing(
                    input_df,
                    oof_df=oof_cl,
                    initial_inventory=args.get("initial_inventory"),
                    transfer_cost=args.get("transfer_cost", 2.5),
                    max_transfer_lead_time=float(args.get("max_transfer_lead_time", 5.0)),
                )
                return {
                    "mode": "rebalancing",
                    "status": rb_res.status,
                    "is_feasible": rb_res.is_feasible,
                    "total_transferred_units": rb_res.total_transferred_units,
                    "total_transfer_cost": rb_res.total_transfer_cost,
                    "stockout_units_avoided": rb_res.stockout_units_avoided,
                    "baseline_network_cost": rb_res.baseline_network_cost,
                    "optimal_network_cost": rb_res.optimal_network_cost,
                    "net_cost_savings": rb_res.net_cost_savings,
                    "relaxation_hint": rb_res.relaxation_hint,
                    "transfers": rb_res.transfers_df.to_dict(orient="records"),
                    "node_summary": rb_res.node_summary_df.to_dict(orient="records"),
                }

            if eff_mode == "contract_volume":
                raw_tiers = args.get("contract_tiers")
                tiers_arg: Any = None
                if isinstance(raw_tiers, list):
                    tiers_arg = [
                        (float(pair[0]), float(pair[1])) for pair in raw_tiers if len(pair) >= 2
                    ]
                elif isinstance(raw_tiers, dict):
                    tiers_arg = {float(k): float(v) for k, v in raw_tiers.items()}
                cv_res = optimize_contract_volume(
                    input_df,
                    trajectories_df=traj_cl,
                    contract_tiers=tiers_arg,
                    min_contract_volume=(
                        float(args["min_contract_volume"])
                        if args.get("min_contract_volume") is not None
                        else None
                    ),
                    shortfall_penalty_per_unit=float(args.get("shortfall_penalty_per_unit", 5.0)),
                    risk_aversion_lambda=float(args.get("risk_aversion_lambda", 0.25)),
                    cvar_alpha=float(args.get("cvar_alpha", 0.90)),
                    **common_kw,
                )
                return {
                    "mode": "contract_volume",
                    "status": cv_res.status,
                    "is_feasible": cv_res.is_feasible,
                    "baseline_portfolio_units": cv_res.baseline_portfolio_units,
                    "optimal_portfolio_units": cv_res.optimal_portfolio_units,
                    "target_contract_units": cv_res.target_contract_units,
                    "achieved_tier_units": cv_res.achieved_tier_units,
                    "rebate_earned": cv_res.rebate_earned,
                    "shortfall_units": cv_res.shortfall_units,
                    "shortfall_penalty": cv_res.shortfall_penalty,
                    "cvar_shortfall_units": cv_res.cvar_shortfall_units,
                    "baseline_net_value": cv_res.baseline_net_value,
                    "optimal_net_value": cv_res.optimal_net_value,
                    "net_value_lift": cv_res.net_value_lift,
                    "relaxation_hint": cv_res.relaxation_hint,
                    "summary": cv_res.summary_df.to_dict(orient="records"),
                    "schedule": cv_res.schedule_df.head(30).to_dict(orient="records"),
                    "hierarchy_rollup": (
                        cv_res.hierarchy_rollup_df.to_dict(orient="records")
                        if cv_res.hierarchy_rollup_df is not None
                        else None
                    ),
                }

            cl_res = optimize_clearance(
                input_df,
                expiry_step=args.get("expiry_step"),
                monotone_markdown=bool(args.get("monotone_markdown", True)),
                salvage_value=args.get("salvage_value", 2.0),
                spoilage_cost=args.get("spoilage_cost", 0.0),
                target_sell_through_pct=args.get("target_sell_through_pct"),
                unit_volume=args.get("unit_volume", 1.0),
                target_space_freed=(
                    float(args["target_space_freed"])
                    if args.get("target_space_freed") is not None
                    else None
                ),
                reclaim_by_step=(
                    int(args["reclaim_by_step"])
                    if args.get("reclaim_by_step") is not None
                    else None
                ),
                cannibalization_penalty=float(args.get("cannibalization_penalty", 0.15)),
                **common_kw,
            )
            return {
                "mode": "clearance",
                "status": cl_res.status,
                "is_feasible": cl_res.is_feasible,
                "total_baseline_profit": cl_res.total_baseline_profit,
                "total_optimal_profit": cl_res.total_optimal_profit,
                "total_profit_lift": cl_res.total_profit_lift,
                "total_space_freed": cl_res.total_space_freed,
                "target_space_freed": cl_res.target_space_freed,
                "overall_sell_through_pct": cl_res.overall_sell_through_pct,
                "relaxation_hint": cl_res.relaxation_hint,
                "summary": cl_res.summary_df.to_dict(orient="records"),
                "schedule": cl_res.schedule_df.head(30).to_dict(orient="records"),
                "hierarchy_rollup": (
                    cl_res.hierarchy_rollup_df.to_dict(orient="records")
                    if cl_res.hierarchy_rollup_df is not None
                    else None
                ),
            }

        if name == "ask_forecast_registry":
            import os

            req_mode = str(args.get("mode", "auto")).strip().lower()
            proj_arg = args.get("project_id") or os.environ.get("SF_PROJECT_ID")
            ds_arg = (
                args.get("dataset_id")
                or os.environ.get("SF_REGISTRY_DATASET_ID")
                or os.environ.get("SF_DATASET_ID")
            )
            if req_mode == "probe":
                from .bq_agent.probe import probe_bq_agent_access

                return probe_bq_agent_access(
                    project_id=str(proj_arg) if proj_arg else None,
                    dataset_id=str(ds_arg) if ds_arg else None,
                ).to_dict()

            question = str(args.get("question", "")).strip()
            if not question:
                raise ValueError("question must be a non-empty string")
            conv_id = args.get("conversation_id")

            eff_mode = req_mode
            if eff_mode == "auto":
                if proj_arg and probe_environment(self.repo_root)["extras"]["gcp"]["installed"]:
                    from .bq_agent.probe import probe_bq_agent_access

                    eff_mode = (
                        "bigquery_agent"
                        if probe_bq_agent_access(
                            project_id=str(proj_arg),
                            dataset_id=str(ds_arg) if ds_arg else None,
                        ).ready
                        else "offline_playground"
                    )
                else:
                    eff_mode = "offline_playground"

            if eff_mode == "offline_playground":
                from . import playground
                from .bq_agent.offline import ask_offline

                model_name = str(args.get("model", "holtwinters"))
                horizon = int(args.get("horizon", 14))
                n_series = int(args.get("n_series", 3))
                panel = playground.sample_data(n_series=max(1, n_series), with_exog=True)
                bakeoff_models: list[str] = (
                    ["naive_seasonal", model_name]
                    if model_name != "naive_seasonal"
                    else ["naive_seasonal", "holtwinters"]
                )
                bo = playground.bakeoff(data=panel, models=bakeoff_models, horizon=horizon)
                resp = ask_offline(bo, question, conversation_id=conv_id)
                return resp.to_dict()

            if eff_mode == "bigquery_agent":
                from .bq_agent.client import ask_registry
                from .errors import require_extra

                require_extra("gcp", purpose="BigQuery Conversational Analytics Agent")
                resp = ask_registry(
                    question,
                    run_id=str(args.get("run_id", "latest")),
                    conversation_id=conv_id,
                    stateful=bool(args.get("stateful", True)),
                    auto_ensure_agent=bool(args.get("auto_ensure_agent", True)),
                    project_id=str(proj_arg) if proj_arg else None,
                    dataset_id=str(ds_arg) if ds_arg else None,
                )
                return resp.to_dict()

            raise ValueError(
                f"Unknown ask_forecast_registry mode {req_mode!r}; expected "
                "'auto', 'probe', 'bigquery_agent', or 'offline_playground'."
            )

        raise ValueError(f"Unknown tool: {name!r}")

    def handle_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Process one JSON-RPC 2.0 request or notification and return a response dict (or None)."""
        method = message.get("method", "")
        msg_id = message.get("id")
        params = message.get("params") or {}

        # Notifications have no `id` and expect no response.
        if msg_id is None:
            return None

        try:
            if method == "initialize":
                client_version = params.get("protocolVersion") or MCP_PROTOCOL_VERSION
                result: dict[str, Any] = {
                    "protocolVersion": client_version,
                    "capabilities": {
                        "resources": {"subscribe": False, "listChanged": False},
                        "tools": {"listChanged": False},
                    },
                    "serverInfo": {
                        "name": "scale-forecasting",
                        "version": _pkg_version("scale-forecasting"),
                    },
                }
            elif method == "ping":
                result = {}
            elif method == "resources/list":
                result = {"resources": self.list_resources()}
            elif method == "resources/read":
                uri = str(params.get("uri", ""))
                result = self.read_resource(uri)
            elif method == "tools/list":
                result = {"tools": self.list_tools()}
            elif method == "tools/call":
                tool_name = str(params.get("name", ""))
                tool_args = params.get("arguments") or {}
                result = self.call_tool(tool_name, tool_args)
            elif method == "prompts/list":
                result = {"prompts": []}
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}",
                    },
                }
            return {"jsonrpc": "2.0", "id": msg_id, "result": result}
        except Exception as exc:  # noqa: BLE001 - JSON-RPC protocol handler must return structured error object
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32602,
                    "message": str(exc),
                },
            }

    def serve_stdio(
        self,
        stdin: BinaryIO | None = None,
        stdout: BinaryIO | None = None,
    ) -> None:
        """Serve MCP JSON-RPC 2.0 requests over `stdio` (newline-delimited or Content-Length)."""
        in_stream: BinaryIO = stdin if stdin is not None else sys.stdin.buffer
        out_stream: BinaryIO = stdout if stdout is not None else sys.stdout.buffer

        while True:
            line_bytes = in_stream.readline()
            if not line_bytes:
                break
            stripped = line_bytes.strip()
            if not stripped:
                continue

            use_content_length = False
            if stripped.lower().startswith(b"content-length:"):
                use_content_length = True
                length = int(stripped.split(b":", 1)[1].strip())
                while True:
                    hdr_bytes = in_stream.readline()
                    if not hdr_bytes or not hdr_bytes.strip():
                        break
                body_bytes = in_stream.read(length)
            else:
                body_bytes = stripped

            try:
                request = json.loads(body_bytes.decode("utf-8"))
            except json.JSONDecodeError as exc:
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {exc}"},
                }
                _write_frame(out_stream, err_resp, use_content_length=use_content_length)
                continue

            response = self.handle_message(request)
            if response is not None:
                _write_frame(out_stream, response, use_content_length=use_content_length)


def _write_frame(
    out_stream: BinaryIO,
    payload: dict[str, Any],
    *,
    use_content_length: bool,
) -> None:
    encoded = json.dumps(payload, default=_json_default).encode("utf-8")
    if use_content_length:
        frame = f"Content-Length: {len(encoded)}\r\n\r\n".encode("ascii") + encoded
    else:
        frame = encoded + b"\n"
    out_stream.write(frame)
    out_stream.flush()


def compute_time_series_similarity(
    source_path_or_table: str | None = None,
    *,
    preset: str = "retail_daily",
    method: str = "hybrid",
    query_ts_id: str | None = None,
    mode: str = "full_history",
    n_clusters: int | str = "auto",
    top_k: int = 5,
    max_lag: int = 0,
    window: int = 14,
    include_substitutes: bool = False,
    include_emerging_trends: bool = False,
    text_cols: Sequence[str] | str | None = None,
    attribute_cols: Sequence[str] | None = None,
    hierarchy_cols: Sequence[str] | None = None,
    within_hierarchy_level: str | Sequence[str] | None = None,
    inventory_records: list[dict[str, Any]] | None = None,
    require_in_stock: bool = True,
    ts_id_col: str = "ts_id",
    date_col: str = "ds",
    target_col: str = "y",
    freq: str = "D",
    project_id: str | None = None,
    records: list[dict[str, Any]] | None = None,
    n_series: int = 4,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Compute time-series similarity, embeddings, behavioral clusters, analog matches,
    stockout-aware substitutes, and emerging trend breakouts."""
    import pandas as pd

    from . import playground
    from .benchmarks import BENCHMARK_NAMES, load_benchmark
    from .similarity import (
        cluster_by_similarity_df,
        compute_similarity_matrix,
        detect_emerging_trends_df,
        find_similar_series_df,
        recommend_substitutes_df,
    )

    _ = (preset, freq, project_id)
    if isinstance(records, list) and records:
        panel_df = pd.DataFrame.from_records(records)
    elif source_path_or_table and str(source_path_or_table).strip().lower() not in (
        "playground",
        "none",
        "",
    ):
        src_str = str(source_path_or_table).strip()
        if src_str in BENCHMARK_NAMES:
            panel_df = load_benchmark(src_str).df.copy()
        else:
            cand_p = Path(src_str)
            if not cand_p.is_file() and repo_root is not None:
                cand_p = repo_root / src_str
            if cand_p.is_file():
                sfx = cand_p.suffix.lower()
                if sfx == ".parquet":
                    panel_df = pd.read_parquet(cand_p)
                elif sfx == ".json":
                    panel_df = pd.read_json(cand_p)
                else:
                    panel_df = pd.read_csv(cand_p)
            else:
                panel_df = playground.sample_data(n_series=max(2, int(n_series)))
    else:
        panel_df = playground.sample_data(n_series=max(2, int(n_series)))

    sim_res = compute_similarity_matrix(
        panel_df,
        method=method,
        top_k=max(1, int(top_k)),
        max_lag=max(0, int(max_lag)),
        text_cols=text_cols,
        attribute_cols=attribute_cols,
        hierarchy_cols=hierarchy_cols,
        within_hierarchy_level=within_hierarchy_level,
        ts_id_col=ts_id_col,
        date_col=date_col,
        target_col=target_col,
    )
    eff_n_clusters: Any = (
        "auto"
        if isinstance(n_clusters, str) and n_clusters.strip().lower() == "auto"
        else int(n_clusters)
    )
    clusters_df = cluster_by_similarity_df(
        panel_df,
        n_clusters=eff_n_clusters,
        method=method,
        text_cols=text_cols,
        attribute_cols=attribute_cols,
        hierarchy_cols=hierarchy_cols,
        ts_id_col=ts_id_col,
        date_col=date_col,
        target_col=target_col,
        similarity_result=sim_res,
    )
    canon_mode = str(mode).strip().lower()
    inv_df = (
        pd.DataFrame.from_records(inventory_records)
        if isinstance(inventory_records, list) and inventory_records
        else None
    )
    subs_df: pd.DataFrame | None = None
    trends_df: pd.DataFrame | None = None

    if canon_mode == "substitutes" or include_substitutes:
        subs_df = recommend_substitutes_df(
            panel_df,
            inventory_df=inv_df,
            query_ts_id=query_ts_id,
            top_k=max(1, int(top_k)),
            text_cols=text_cols,
            attribute_cols=attribute_cols,
            hierarchy_cols=hierarchy_cols,
            within_hierarchy_level=within_hierarchy_level,
            require_in_stock=require_in_stock,
            method=method,
            ts_id_col=ts_id_col,
            date_col=date_col,
            target_col=target_col,
        )
    if canon_mode == "emerging_trends" or include_emerging_trends:
        trends_df = detect_emerging_trends_df(
            panel_df,
            window=max(3, int(window)),
            top_k_related=max(1, int(top_k)),
            text_cols=text_cols,
            attribute_cols=attribute_cols,
            hierarchy_cols=hierarchy_cols,
            max_lag=max(1, int(max_lag)) if int(max_lag) > 0 else 7,
            method=method,
            ts_id_col=ts_id_col,
            date_col=date_col,
            target_col=target_col,
        )

    if canon_mode == "substitutes" and subs_df is not None:
        matches_df = subs_df
    elif canon_mode == "emerging_trends" and trends_df is not None:
        matches_df = trends_df
    elif query_ts_id is not None or canon_mode != "full_history":
        matches_df = find_similar_series_df(
            panel_df,
            query_ts_id=query_ts_id,
            method=method,
            top_k=max(1, int(top_k)),
            mode=canon_mode,
            max_lag=max(0, int(max_lag)),
            text_cols=text_cols,
            attribute_cols=attribute_cols,
            hierarchy_cols=hierarchy_cols,
            within_hierarchy_level=within_hierarchy_level,
            ts_id_col=ts_id_col,
            date_col=date_col,
            target_col=target_col,
        )
    else:
        matches_df = sim_res.pairs_df

    opt_k = (
        int(clusters_df["similarity_cluster_id"].nunique())
        if not clusters_df.empty and "similarity_cluster_id" in clusters_df.columns
        else 1
    )
    sil_val = (
        float(clusters_df["silhouette_score"].mean())
        if not clusters_df.empty and "silhouette_score" in clusters_df.columns
        else 0.0
    )
    out_dict: dict[str, Any] = {
        "status": "ok",
        "method": sim_res.method,
        "mode": str(mode),
        "n_series": len(sim_res.matrix_df),
        "optimal_n_clusters": opt_k,
        "silhouette_score": round(sil_val, 6),
        "mean_similarity": round(
            float(sim_res.diagnostics.get("mean_pairwise_similarity", 1.0)), 6
        ),
        "top_pairs": sim_res.pairs_df.head(
            max(1, int(top_k) * max(1, len(sim_res.matrix_df)))
        ).to_dict(orient="records"),
        "clusters": clusters_df.to_dict(orient="records"),
        "embeddings": sim_res.embeddings_df.to_dict(orient="records"),
        "analog_matches": matches_df.to_dict(orient="records"),
        "diagnostics": dict(sim_res.diagnostics),
    }
    if subs_df is not None:
        out_dict["substitutes"] = subs_df.to_dict(orient="records")
    if trends_df is not None:
        out_dict["emerging_trends"] = trends_df.to_dict(orient="records")
    return out_dict


def _configure_stderr_logging() -> None:
    """Route package logs exclusively to `stderr` so `stdout` stays clean for JSON-RPC."""
    pkg_logger = logging.getLogger(PACKAGE_LOGGER)
    pkg_logger.propagate = False
    if not any(isinstance(h, logging.StreamHandler) for h in pkg_logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s")
        )
        pkg_logger.addHandler(handler)
    pkg_logger.setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``python -m scale_forecasting.mcp``."""
    _configure_stderr_logging()
    parser = argparse.ArgumentParser(
        prog="scale_forecasting.mcp",
        description="Run the scale-forecasting Model Context Protocol (MCP) server over stdio.",
    )
    parser.add_argument(
        "--allow-launch",
        action="store_true",
        help=(
            "Unlock cloud job submission and mutating registry repair tools "
            "(`launch_or_repair_run` and `stage_only`)."
        ),
    )
    parser.add_argument(
        "--probe-env",
        action="store_true",
        help="Print the environment readiness JSON report to stdout and exit.",
    )
    args = parser.parse_args(argv)

    if args.probe_env:
        print(_dumps(probe_environment()))
        return 0

    server = McpServer(allow_launch=args.allow_launch)
    server.serve_stdio()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
