"""Built-in Model Context Protocol (MCP) server for `scale-forecasting` over `stdio`.

Exposes the platform's configuration schema, canonical catalogs (34 models, 21 metrics, 7 runtimes,
4 GPU types, 12 extras, 5 views, 62 shipped configs), environment readiness probe, offline model
playground, `RunConfig` validator and DAG planner, Airflow DAG emitter, and live BigQuery registry
inspection/review/repair tools to any MCP-compatible AI coding agent (Gemini CLI, Claude Code,
Antigravity / Jetski, Cursor, VS Code Copilot, Windsurf).

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
from .config import RunConfig, estimate_workload, load_config
from .dag import dag_nodes, group_models_by_family, plan_dag
from .errors import PACKAGE_LOGGER, MissingExtraError, ScaleForecastError
from .metrics import METRIC_NAMES, get_metric
from .models import get_model, list_models
from .registry.ddl import REGISTRY_TABLE_NAMES, SOURCE_TABLE_NAMES
from .registry.ids import make_run_id
from .registry.reads import RECENT_RUNS_COLUMNS
from .registry.views import VIEW_NAMES

__all__ = [
    "MCP_PROTOCOL_VERSION",
    "McpServer",
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
                    "Installed extras (12), available vs. missing models (34), SF_* environment "
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
                "name": "Forecasting Models Catalog (34 Models)",
                "description": (
                    "All 34 registered models across 5 families with runtime, package, extra, "
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
                    "The 4 source tables, 5 registry tables, and 5 analytical SQL views."
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
        """Return the 9 MCP tools exposed by this server."""
        return [
            {
                "name": "probe_environment",
                "description": (
                    "Inspect the active Python environment: installed optional extras (out of 12), "
                    "locally available vs. missing-extra models (out of 34), SF_* environment "
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
                    "Query the canonical catalogs of models (34), metrics (21), runtimes (7), "
                    "GPUs (4), dependency extras (12), BigQuery views (5), or shipped configs (62)."
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
                    "OutputConfig, HpoConfig, EnsembleConfig, HierarchyConfig, ComputeConfig, "
                    "FamilyCompute, EnsembleCompute, ProfileConfig, CapacityConfig, "
                    "CapacityServicePolicy, RetryResources)."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "section": {
                            "type": "string",
                            "description": (
                                "Optional section or Pydantic class name to inspect (e.g., "
                                "'compute', 'backtest', 'FamilyCompute'). Omit for overview."
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

            model_name = str(args.get("model", "holtwinters"))
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

            panel = playground.sample_data(n_series=n_series)
            run = playground.run_model(
                model_name,
                data=panel,
                horizon=horizon,
                backtest=backtest,
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
            return {
                "model": model_name,
                "family": mcls.family,
                "ts_id": res.ts_id,
                "status": res.status,
                "n_series_in_panel": n_series,
                "horizon": horizon,
                "backtest": backtest,
                "metrics": metrics_clean,
                "summary": summary_text,
                "sample_predictions": sample_preds,
            }

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
