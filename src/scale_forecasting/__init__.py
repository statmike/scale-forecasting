"""scale-forecasting — massively-parallel time-series forecasting on Google Cloud.

The package is deliberately flat and readable: one capability per file. Start with the
architecture overview in the README, then read ``config.py`` (the run contract),
``worker.py`` (the unit of work), and one file under ``models/`` to see the whole shape.

Public surface — two ways in, one engine underneath:

* **The easy path.** `Forecaster` — point it at a config and call
  `dry_run` / `run`
  / `review`. Thin wrapper over `run`; no logic of its
  own.
* **The manage path.** `Registry` — the operator surface over the registry a deployment writes
  to: `doctor` (read-only health), `drop_run`, `sweep_orphans`, `snapshot`, `export`, `init`.
  Same verbs as ``python -m scale_forecasting.registry.ops``.
* **The direct path.** Drive Spark or Ray yourself and reuse the *same* model machinery:
  `run_group` (pure, per-bucket), `make_group_runner` / `make_chunk_runner` (the
  writer-attached ``applyInPandas`` / Ray-task closures), `chunk_cells`, and the unit of work
  `run_cell`. Both paths run byte-identical cell code. See ``docs/using_the_sdk.md``.

Import cost: ``import scale_forecasting`` is near-instant. Light names (config, settings, errors)
are eager; the heavy names above load the model modules and are therefore **lazy** — resolved on
first attribute access via PEP 562. Touching ``Forecaster``/``run``/``run_cell`` pays the model
import cost; touching ``RunConfig``/``Settings`` does not. ``test_sdk.py`` guards this contract.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import TYPE_CHECKING

# Eager (light): the config contract, infra identity, and the error hierarchy. None of these pull
# the model modules, so they keep ``import scale_forecasting`` cheap.
from .config import (
    Fanout,
    RunConfig,
    Workload,
    estimate_fanout,
    estimate_workload,
    load_config,
)
from .errors import (
    ConfigError,
    DataError,
    EngineError,
    GoalSeekInfeasibleError,
    ModelError,
    RegistryError,
    ScaleForecastError,
    ScaleForecastingError,
)
from .settings import Settings

try:
    __version__: str = _pkg_version("scale-forecasting")
except PackageNotFoundError:  # pragma: no cover - zip-staged cloud worker without .dist-info
    __version__ = "1.8.0"

# Lazy (heavy): name -> (module, attribute). Resolved on first access by ``__getattr__`` (PEP 562)
# so the model modules (statsmodels et al., ~5s) load only when a caller actually reaches for the
# run/model surface — never merely to ``import scale_forecasting``.
_LAZY: dict[str, tuple[str, str]] = {
    # Easy path.
    "Forecaster": (".sdk", "Forecaster"),
    "Registry": (".sdk", "Registry"),
    "DryRunResult": (".sdk", "DryRunResult"),
    "RunResult": (".sdk", "RunResult"),
    "ModelResult": (".sdk", "ModelResult"),
    "JobTrace": (".sdk", "JobTrace"),
    "build_explain_frame": (".sdk", "build_explain_frame"),
    "build_best_params_frame": (".sdk", "build_best_params_frame"),
    "build_trace_frame": (".sdk", "build_trace_frame"),
    "plot_trace": (".sdk", "plot_trace"),
    # Run inspection — monitor a running run, review a finished one.
    "monitor_run": (".review", "monitor_run"),
    "review_run": (".review", "review_run"),
    "RunProgress": (".review", "RunProgress"),
    "RunReview": (".review", "RunReview"),
    "calibration_report": (".review", "calibration_report"),
    "CalibrationReport": (".review", "CalibrationReport"),
    "build_leaderboard_frame": (".review", "build_leaderboard_frame"),
    "build_predictions_frame": (".review", "build_predictions_frame"),
    "build_hierarchy_frame": (".review", "build_hierarchy_frame"),
    "build_cohorts_frame": (".review", "build_cohorts_frame"),
    "build_calibration_frames": (".review", "build_calibration_frames"),
    "build_ensemble_weights_frame": (".review", "build_ensemble_weights_frame"),
    "build_attributions_frame": (".review", "build_attributions_frame"),
    "explain_forecast_frame": (".review", "explain_forecast_frame"),
    "plot_progress": (".review", "plot_progress"),
    "plot_leaderboard": (".review", "plot_leaderboard"),
    "plot_metric_distribution": (".review", "plot_metric_distribution"),
    "plot_forecasts_frame": (".review", "plot_forecasts_frame"),
    "plot_hierarchy_frame": (".review", "plot_hierarchy_frame"),
    "plot_calibration": (".review", "plot_calibration"),
    "plot_ensemble_weights": (".review", "plot_ensemble_weights"),
    "plot_attributions": (".review", "plot_attributions"),
    "plot_forecast_explanation": (".review", "plot_forecast_explanation"),
    "DagNode": (".dag", "DagNode"),
    # Cross-run ensembling.
    "merge_configs_for_ensemble": (".ensemble_run", "merge_configs_for_ensemble"),
    "ensemble_cross_runs": (".ensemble_run", "ensemble_cross_runs"),
    # Orchestration.
    "run": (".main", "run"),
    # Direct path — pure core + writer-attached runners + unit of work.
    "run_group": (".engines.spark_io", "run_group"),
    "make_group_runner": (".engines.spark_io", "make_group_runner"),
    "RunOutcome": (".engines.spark_io", "RunOutcome"),
    "aggregate_status": (".engines.spark_io", "aggregate_status"),
    "make_chunk_runner": (".engines.ray_io", "make_chunk_runner"),
    "chunk_cells": (".engines.ray_io", "chunk_cells"),
    "run_cell": (".worker", "run_cell"),
    "CellResult": (".worker", "CellResult"),
    # Model registry + runtime routing + catalogs.
    "get_model": (".models", "get_model"),
    "list_models": (".models", "list_models"),
    "model_catalog": (".playground", "model_catalog"),
    "metric_catalog": (".playground", "metric_catalog"),
    "split_by_runtime": (".router", "split_by_runtime"),
    # Hierarchical aggregation + coherent reconciliation.
    "HierarchySpec": (".reconciliation", "HierarchySpec"),
    "build_hierarchy": (".reconciliation", "build_hierarchy"),
    "materialize_similarity_hierarchy": (
        ".reconciliation",
        "materialize_similarity_hierarchy",
    ),
    "ensure_hierarchy_nodes_present": (
        ".reconciliation",
        "ensure_hierarchy_nodes_present",
    ),
    "reconcile_run": (".reconciliation", "reconcile_run"),
    "reconcile_matrix": (".reconciliation", "reconcile_matrix"),
    "reconcile_forecasts": (".reconciliation", "reconcile_forecasts"),
    "reconcile_oof": (".reconciliation", "reconcile_oof"),
    "reconcile_cells": (".reconciliation", "reconcile_cells"),
    "verify_coherence": (".reconciliation", "verify_coherence"),
    # Cross-temporal reconciliation, compositional simplex & goal-seek (Track K).
    "TEMPORAL_RECONCILIATION_METHODS": (
        ".temporal_reconciliation",
        "TEMPORAL_RECONCILIATION_METHODS",
    ),
    "DEFAULT_TEMPORAL_FACTORS": (
        ".temporal_reconciliation",
        "DEFAULT_TEMPORAL_FACTORS",
    ),
    "TemporalHierarchySpec": (".temporal_reconciliation", "TemporalHierarchySpec"),
    "default_temporal_aggregations": (
        ".temporal_reconciliation",
        "default_temporal_aggregations",
    ),
    "resolve_temporal_aggregations": (
        ".temporal_reconciliation",
        "resolve_temporal_aggregations",
    ),
    "build_temporal_hierarchy": (".temporal_reconciliation", "build_temporal_hierarchy"),
    "build_temporal_summing_matrix": (
        ".temporal_reconciliation",
        "build_temporal_summing_matrix",
    ),
    "reconcile_temporal_matrix": (".temporal_reconciliation", "reconcile_temporal_matrix"),
    "reconcile_series_temporal": (".temporal_reconciliation", "reconcile_series_temporal"),
    "reconcile_temporal_series": (".temporal_reconciliation", "reconcile_temporal_series"),
    "reconcile_cross_temporal": (".temporal_reconciliation", "reconcile_cross_temporal"),
    "cross_temporal_mint": (".temporal_reconciliation", "cross_temporal_mint"),
    "reconcile_forecasts_temporal": (
        ".temporal_reconciliation",
        "reconcile_forecasts_temporal",
    ),
    "reconcile_temporal_forecasts": (
        ".temporal_reconciliation",
        "reconcile_temporal_forecasts",
    ),
    "temporal_reconcile_df": (
        ".temporal_reconciliation",
        "temporal_reconcile_df",
    ),
    "sample_coherent_paths": (
        ".temporal_reconciliation",
        "sample_coherent_paths",
    ),
    "verify_temporal_coherence": (".temporal_reconciliation", "verify_temporal_coherence"),
    "verify_cross_temporal_coherence": (
        ".temporal_reconciliation",
        "verify_cross_temporal_coherence",
    ),
    "SIMPLEX_TRANSFORMS": (".simplex", "SIMPLEX_TRANSFORMS"),
    "SIMPLICIAL_TRANSFORMS": (".simplex", "SIMPLICIAL_TRANSFORMS"),
    "DEFAULT_ZERO_DELTA": (".simplex", "DEFAULT_ZERO_DELTA"),
    "close_simplex": (".simplex", "close_simplex"),
    "verify_simplex": (".simplex", "verify_simplex"),
    "verify_simplex_closure": (".simplex", "verify_simplex_closure"),
    "verify_simplex_coherence": (".simplex", "verify_simplex_coherence"),
    "forecast_simplex_panel": (".simplex", "forecast_simplex_panel"),
    "multiplicative_zero_replacement": (".simplex", "multiplicative_zero_replacement"),
    "replace_compositional_zeros": (".simplex", "replace_compositional_zeros"),
    "clr_transform": (".simplex", "clr_transform"),
    "clr_inverse": (".simplex", "clr_inverse"),
    "inverse_clr_transform": (".simplex", "inverse_clr_transform"),
    "helmert_sub_matrix": (".simplex", "helmert_sub_matrix"),
    "ilr_transform": (".simplex", "ilr_transform"),
    "ilr_inverse": (".simplex", "ilr_inverse"),
    "inverse_ilr_transform": (".simplex", "inverse_ilr_transform"),
    "transform_simplex_series": (".simplex", "transform_simplex_series"),
    "inverse_transform_simplex_series": (".simplex", "inverse_transform_simplex_series"),
    "apply_simplex_series_transform": (".simplex", "apply_simplex_series_transform"),
    "inverse_simplex_series_transform": (".simplex", "inverse_simplex_series_transform"),
    "close_simplex_predictions": (".simplex", "close_simplex_predictions"),
    "should_close_simplex": (".simplex", "should_close_simplex"),
    "GOAL_SEEK_COLUMNS": (".goal_seek", "GOAL_SEEK_COLUMNS"),
    "allocate_target": (".goal_seek", "allocate_target"),
    "allocate_forecast_target": (".goal_seek", "allocate_forecast_target"),
    "summarize_goal_seek": (".goal_seek", "summarize_goal_seek"),
    "plot_goal_seek": (".goal_seek", "plot_goal_seek"),
    # AI agent surfaces & MCP server.
    "probe_environment": (".agent_surfaces", "probe_environment"),
    "McpServer": (".mcp", "McpServer"),
    # Forecaster Workbench diagnostics, scenario simulation & self-contained reports.
    "compute_demand_profile": (".features", "compute_demand_profile"),
    "compute_fva": (".diagnostics", "compute_fva"),
    "diebold_mariano_test": (".diagnostics", "diebold_mariano_test"),
    "wilcoxon_pvalue": (".diagnostics", "wilcoxon_pvalue"),
    "compute_horizon_decay": (".diagnostics", "compute_horizon_decay"),
    "compute_bias_report": (".diagnostics", "compute_bias_report"),
    "compute_realized_accuracy": (".diagnostics", "compute_realized_accuracy"),
    "compute_revision_stability": (".diagnostics", "compute_revision_stability"),
    "compute_exceptions_queue": (".diagnostics", "compute_exceptions_queue"),
    "compute_anomalies": (".diagnostics", "compute_anomalies"),
    "compute_change_points": (".diagnostics", "compute_change_points"),
    "compute_inventory_policy": (".diagnostics", "compute_inventory_policy"),
    "rollup_time": (".diagnostics", "rollup_time"),
    "plot_fva": (".diagnostics", "plot_fva"),
    "plot_horizon_decay": (".diagnostics", "plot_horizon_decay"),
    "plot_inventory_policy": (".diagnostics", "plot_inventory_policy"),
    # Causal & structural econometric guardrails (Track J).
    "MONOTONE_SUPPORTED_MODELS": (".monotonicity", "MONOTONE_SUPPORTED_MODELS"),
    "MONOTONICITY_AUDIT_COLUMNS": (".monotonicity", "MONOTONICITY_AUDIT_COLUMNS"),
    "list_monotonic_models": (".monotonicity", "list_monotonic_models"),
    "audit_scenario_monotonicity": (".monotonicity", "audit_scenario_monotonicity"),
    "infer_monotonic_constraints": (".monotonicity", "infer_monotonic_constraints"),
    "resolve_monotonic_constraints": (".monotonicity", "resolve_monotonic_constraints"),
    "build_monotonic_constraint_vector": (
        ".monotonicity",
        "build_monotonic_constraint_vector",
    ),
    "resolve_model_monotonic_vector": (".monotonicity", "resolve_model_monotonic_vector"),
    "fit_sign_constrained_ridge": (".monotonicity", "fit_sign_constrained_ridge"),
    "HOLIDAY_WINDOW_COLUMNS": (".holiday_window", "HOLIDAY_WINDOW_COLUMNS"),
    "HolidayWindowConfig": (".holiday_window", "HolidayWindowConfig"),
    "build_holiday_window_features": (".holiday_window", "build_holiday_window_features"),
    "PROMO_ELASTICITY_COLUMNS": (".elasticity", "PROMO_ELASTICITY_COLUMNS"),
    "compute_promo_elasticity": (".elasticity", "compute_promo_elasticity"),
    "estimate_promo_elasticity": (".elasticity", "estimate_promo_elasticity"),
    "compute_promo_elasticity_decomposition": (
        ".elasticity",
        "compute_promo_elasticity_decomposition",
    ),
    "promo_elasticity_df": (".elasticity", "promo_elasticity_df"),
    "summarize_promo_elasticity": (".elasticity", "summarize_promo_elasticity"),
    "plot_promo_elasticity": (".elasticity", "plot_promo_elasticity"),
    "make_confounded_promo_dgp": (".elasticity", "make_confounded_promo_dgp"),
    "apply_scenario_shocks": (".scenarios", "apply_scenario_shocks"),
    "compare_scenario_margin_curves": (".scenarios", "compare_scenario_margin_curves"),
    "simulate_scenarios": (".scenarios", "simulate_scenarios"),
    "summarize_scenarios": (".scenarios", "summarize_scenarios"),
    "plot_scenarios": (".scenarios", "plot_scenarios"),
    "build_report_payload": (".report", "build_report_payload"),
    "render_html_report": (".report", "render_html_report"),
    "export_report": (".report", "export_report"),
    # Glass-Box Auto-Forecaster, Duration Predictor & DGP Benchmarks.
    "EXPOSURE_TIERS": (".auto", "EXPOSURE_TIERS"),
    "SCALE_REGIMES": (".auto", "SCALE_REGIMES"),
    "classify_scale_regime": (".auto", "classify_scale_regime"),
    "select_default_baselines": (".auto", "select_default_baselines"),
    "select_hierarchy_reconciliation_methods": (
        ".auto",
        "select_hierarchy_reconciliation_methods",
    ),
    "ConfigBlueprint": (".auto", "ConfigBlueprint"),
    "AutofocusPassRecord": (".auto", "AutofocusPassRecord"),
    "NextLoopPreview": (".auto", "NextLoopPreview"),
    "PanelAutoProfile": (".auto", "PanelAutoProfile"),
    "DataProfile": (".auto", "DataProfile"),
    "DatasetProfile": (".auto", "DatasetProfile"),
    "inspect_panel_for_auto": (".auto", "inspect_panel_for_auto"),
    "build_config": (".auto", "build_config"),
    "auto_forecast": (".auto", "auto_forecast"),
    "DurationEstimate": (".profiling.duration", "DurationEstimate"),
    "FamilyDurationBreakdown": (".profiling.duration", "FamilyDurationBreakdown"),
    "DurationCalibration": (".profiling.duration", "DurationCalibration"),
    "predict_run_duration": (".profiling.duration", "predict_run_duration"),
    "calibrate_duration_from_frames": (".profiling.duration", "calibrate_duration_from_frames"),
    "calibrate_duration_from_registry": (".profiling.duration", "calibrate_duration_from_registry"),
    "BENCHMARK_NAMES": (".benchmarks", "BENCHMARK_NAMES"),
    "BenchmarkSpec": (".benchmarks", "BenchmarkSpec"),
    "BenchmarkDataset": (".benchmarks", "BenchmarkDataset"),
    "list_benchmarks": (".benchmarks", "list_benchmarks"),
    "benchmark_catalog": (".benchmarks", "benchmark_catalog"),
    "load_benchmark": (".benchmarks", "load_benchmark"),
    "PUBLIC_BENCHMARK_NAMES": (".benchmarks", "PUBLIC_BENCHMARK_NAMES"),
    "PUBLIC_BENCHMARK_SPECS": (".benchmarks", "PUBLIC_BENCHMARK_SPECS"),
    "PublicBenchmarkSpec": (".benchmarks", "PublicBenchmarkSpec"),
    "list_public_benchmarks": (".benchmarks", "list_public_benchmarks"),
    "public_benchmark_catalog": (".benchmarks", "public_benchmark_catalog"),
    "fetch_public_benchmark": (".benchmarks", "fetch_public_benchmark"),
    # Opt-in evolutionary code discovery (Track H).
    "EvolveConfig": (".evolve", "EvolveConfig"),
    "EvolveResult": (".evolve", "EvolveResult"),
    "run_evolution": (".evolve", "run_evolution"),
    # Cold-start NPI, online ACI, copula risk, inventory & scenario optimization (Track L).
    "DonorProfile": (".cold_start", "DonorProfile"),
    "ColdStartDonorPool": (".cold_start", "ColdStartDonorPool"),
    "ColdStartDonorCatalog": (".cold_start", "ColdStartDonorCatalog"),
    "ColdStartForecastResult": (".cold_start", "ColdStartForecastResult"),
    "build_cold_start_donor_pool": (".cold_start", "build_cold_start_donor_pool"),
    "build_cold_start_donor_catalog": (".cold_start", "build_cold_start_donor_catalog"),
    "fit_bass_diffusion": (".cold_start", "fit_bass_diffusion"),
    "fit_weibull_lifecycle": (".cold_start", "fit_weibull_lifecycle"),
    "forecast_cold_start_series": (".cold_start", "forecast_cold_start_series"),
    "cold_start_transfer_df": (".cold_start", "cold_start_transfer_df"),
    "cold_start_diagnostic_df": (".cold_start", "cold_start_diagnostic_df"),
    "cold_start_report_df": (".cold_start", "cold_start_report_df"),
    "plot_cold_start_transfer": (".cold_start", "plot_cold_start_transfer"),
    "AciCalibrationState": (".aci", "AciCalibrationState"),
    "compute_aci_df": (".aci", "compute_aci_df"),
    "aci_df": (".aci", "aci_df"),
    "aci_trajectory_df": (".aci", "aci_trajectory_df"),
    "update_aci_online": (".aci", "update_aci_online"),
    "aci_report_df": (".aci", "aci_report_df"),
    "calibrate_aci_from_oof": (".aci", "calibrate_aci_from_oof"),
    "plot_aci": (".aci", "plot_aci"),
    "COPULA_TYPES": (".copula", "COPULA_TYPES"),
    "VALID_COPULA_TYPES": (".copula", "VALID_COPULA_TYPES"),
    "CopulaRiskSummary": (".copula", "CopulaRiskSummary"),
    "sample_trajectories": (".copula", "sample_trajectories"),
    "sample_copula_trajectories": (".copula", "sample_copula_trajectories"),
    "sample_joint_trajectories": (".copula", "sample_joint_trajectories"),
    "compute_tail_risk": (".copula", "compute_tail_risk"),
    "tail_risk_df": (".copula", "tail_risk_df"),
    "summarize_trajectory_risk": (".copula", "summarize_trajectory_risk"),
    "evaluate_trajectory_paths": (".copula", "evaluate_trajectory_paths"),
    "plot_trajectories": (".copula", "plot_trajectories"),
    "INVENTORY_OPTIMIZER_COLUMNS": (".inventory_optimizer", "INVENTORY_OPTIMIZER_COLUMNS"),
    "INVENTORY_POLICY_COLUMNS": (".inventory_optimizer", "INVENTORY_POLICY_COLUMNS"),
    "compute_inventory_frontier": (".inventory_optimizer", "compute_inventory_frontier"),
    "optimize_inventory_df": (".inventory_optimizer", "optimize_inventory_df"),
    "optimize_inventory_portfolio": (".inventory_optimizer", "optimize_inventory_portfolio"),
    "plot_inventory_frontier": (".inventory_optimizer", "plot_inventory_frontier"),
    "SCENARIO_OPTIMIZATION_COLUMNS": (
        ".scenario_optimizer",
        "SCENARIO_OPTIMIZATION_COLUMNS",
    ),
    "SCENARIO_OPTIMIZATION_SUMMARY_COLUMNS": (
        ".scenario_optimizer",
        "SCENARIO_OPTIMIZATION_SUMMARY_COLUMNS",
    ),
    "ScenarioOptimizationResult": (".scenario_optimizer", "ScenarioOptimizationResult"),
    "optimize_scenario": (".scenario_optimizer", "optimize_scenario"),
    "plot_optimal_scenario": (".scenario_optimizer", "plot_optimal_scenario"),
    # Time-series similarity, dynamical embeddings & similarity-regularized pipelines (Track P).
    "SIMILARITY_METHODS": (".similarity", "SIMILARITY_METHODS"),
    "VALID_SIMILARITY_METHODS": (".similarity", "VALID_SIMILARITY_METHODS"),
    "SimilarityMethod": (".similarity", "SimilarityMethod"),
    "SimilarityMatrixResult": (".similarity", "SimilarityMatrixResult"),
    "embed_series_df": (".similarity", "embed_series_df"),
    "compute_similarity_matrix": (".similarity", "compute_similarity_matrix"),
    "cluster_by_similarity_df": (".similarity", "cluster_by_similarity_df"),
    "find_similar_series_df": (".similarity", "find_similar_series_df"),
    "plot_similarity_map": (".similarity", "plot_similarity_map"),
    "extract_foundation_embeddings": (".similarity", "extract_foundation_embeddings"),
    "shrink_correlation_toward_similarity": (
        ".copula",
        "shrink_correlation_toward_similarity",
    ),
    "COLD_START_MATCHING_MODES": (".config", "COLD_START_MATCHING_MODES"),
    "ColdStartMatching": (".config", "ColdStartMatching"),
    "smooth_oof_losses_by_similarity": (
        ".ensembler",
        "smooth_oof_losses_by_similarity",
    ),
}

__all__ = [
    "__version__",
    # Eager.
    "RunConfig",
    "load_config",
    "estimate_workload",
    "Workload",
    "estimate_fanout",
    "Fanout",
    "Settings",
    "ScaleForecastError",
    "ScaleForecastingError",
    "ConfigError",
    "DataError",
    "GoalSeekInfeasibleError",
    "ModelError",
    "RegistryError",
    "EngineError",
    # Lazy.
    *_LAZY,
]

if TYPE_CHECKING:  # so IDEs / type-checkers see the lazy names as real imports (re-exported below)
    from .aci import (  # noqa: F401
        AciCalibrationState,
        aci_df,
        aci_report_df,
        aci_trajectory_df,
        calibrate_aci_from_oof,
        compute_aci_df,
        plot_aci,
        update_aci_online,
    )
    from .agent_surfaces import probe_environment  # noqa: F401
    from .auto import (  # noqa: F401
        EXPOSURE_TIERS,
        SCALE_REGIMES,
        AutofocusPassRecord,
        ConfigBlueprint,
        DataProfile,
        DatasetProfile,
        NextLoopPreview,
        PanelAutoProfile,
        auto_forecast,
        build_config,
        classify_scale_regime,
        inspect_panel_for_auto,
        select_default_baselines,
        select_hierarchy_reconciliation_methods,
    )
    from .benchmarks import (  # noqa: F401
        BENCHMARK_NAMES,
        PUBLIC_BENCHMARK_NAMES,
        PUBLIC_BENCHMARK_SPECS,
        BenchmarkDataset,
        BenchmarkSpec,
        PublicBenchmarkSpec,
        benchmark_catalog,
        fetch_public_benchmark,
        list_benchmarks,
        list_public_benchmarks,
        load_benchmark,
        public_benchmark_catalog,
    )
    from .cold_start import (  # noqa: F401
        ColdStartDonorCatalog,
        ColdStartDonorPool,
        ColdStartForecastResult,
        DonorProfile,
        build_cold_start_donor_catalog,
        build_cold_start_donor_pool,
        cold_start_diagnostic_df,
        cold_start_report_df,
        cold_start_transfer_df,
        fit_bass_diffusion,
        fit_weibull_lifecycle,
        forecast_cold_start_series,
        plot_cold_start_transfer,
    )
    from .config import (  # noqa: F401
        COLD_START_MATCHING_MODES,
        ColdStartMatching,
    )
    from .copula import (  # noqa: F401
        COPULA_TYPES,
        VALID_COPULA_TYPES,
        CopulaRiskSummary,
        compute_tail_risk,
        evaluate_trajectory_paths,
        plot_trajectories,
        sample_copula_trajectories,
        sample_joint_trajectories,
        sample_trajectories,
        shrink_correlation_toward_similarity,
        summarize_trajectory_risk,
        tail_risk_df,
    )
    from .dag import DagNode  # noqa: F401
    from .diagnostics import (  # noqa: F401
        compute_anomalies,
        compute_bias_report,
        compute_change_points,
        compute_exceptions_queue,
        compute_fva,
        compute_horizon_decay,
        compute_inventory_policy,
        compute_realized_accuracy,
        compute_revision_stability,
        diebold_mariano_test,
        plot_fva,
        plot_horizon_decay,
        plot_inventory_policy,
        rollup_time,
        wilcoxon_pvalue,
    )
    from .elasticity import (  # noqa: F401
        PROMO_ELASTICITY_COLUMNS,
        compute_promo_elasticity,
        compute_promo_elasticity_decomposition,
        estimate_promo_elasticity,
        make_confounded_promo_dgp,
        plot_promo_elasticity,
        promo_elasticity_df,
        summarize_promo_elasticity,
    )
    from .engines.ray_io import chunk_cells, make_chunk_runner  # noqa: F401
    from .engines.spark_io import (  # noqa: F401
        RunOutcome,
        aggregate_status,
        make_group_runner,
        run_group,
    )
    from .ensemble_run import ensemble_cross_runs, merge_configs_for_ensemble  # noqa: F401
    from .ensembler import smooth_oof_losses_by_similarity  # noqa: F401
    from .evolve import (  # noqa: F401
        EvolveConfig,
        EvolveResult,
        run_evolution,
    )
    from .features import compute_demand_profile  # noqa: F401
    from .goal_seek import (  # noqa: F401
        GOAL_SEEK_COLUMNS,
        allocate_forecast_target,
        allocate_target,
        plot_goal_seek,
        summarize_goal_seek,
    )
    from .holiday_window import (  # noqa: F401
        HOLIDAY_WINDOW_COLUMNS,
        HolidayWindowConfig,
        build_holiday_window_features,
    )
    from .inventory_optimizer import (  # noqa: F401
        INVENTORY_OPTIMIZER_COLUMNS,
        INVENTORY_POLICY_COLUMNS,
        compute_inventory_frontier,
        optimize_inventory_df,
        optimize_inventory_portfolio,
        plot_inventory_frontier,
    )
    from .main import run  # noqa: F401
    from .mcp import McpServer  # noqa: F401
    from .models import get_model, list_models  # noqa: F401
    from .monotonicity import (  # noqa: F401
        MONOTONE_SUPPORTED_MODELS,
        MONOTONICITY_AUDIT_COLUMNS,
        audit_scenario_monotonicity,
        build_monotonic_constraint_vector,
        fit_sign_constrained_ridge,
        infer_monotonic_constraints,
        list_monotonic_models,
        resolve_model_monotonic_vector,
        resolve_monotonic_constraints,
    )
    from .playground import metric_catalog, model_catalog  # noqa: F401
    from .profiling.duration import (  # noqa: F401
        DurationCalibration,
        DurationEstimate,
        FamilyDurationBreakdown,
        calibrate_duration_from_frames,
        calibrate_duration_from_registry,
        predict_run_duration,
    )
    from .reconciliation import (  # noqa: F401
        HierarchySpec,
        build_hierarchy,
        ensure_hierarchy_nodes_present,
        materialize_similarity_hierarchy,
        reconcile_cells,
        reconcile_forecasts,
        reconcile_matrix,
        reconcile_oof,
        reconcile_run,
        verify_coherence,
    )
    from .report import (  # noqa: F401
        build_report_payload,
        export_report,
        render_html_report,
    )
    from .review import (  # noqa: F401
        CalibrationReport,
        RunProgress,
        RunReview,
        build_attributions_frame,
        build_calibration_frames,
        build_cohorts_frame,
        build_ensemble_weights_frame,
        build_hierarchy_frame,
        build_leaderboard_frame,
        build_predictions_frame,
        calibration_report,
        explain_forecast_frame,
        monitor_run,
        plot_attributions,
        plot_calibration,
        plot_ensemble_weights,
        plot_forecast_explanation,
        plot_forecasts_frame,
        plot_hierarchy_frame,
        plot_leaderboard,
        plot_metric_distribution,
        plot_progress,
        review_run,
    )
    from .router import split_by_runtime  # noqa: F401
    from .scenario_optimizer import (  # noqa: F401
        SCENARIO_OPTIMIZATION_COLUMNS,
        SCENARIO_OPTIMIZATION_SUMMARY_COLUMNS,
        ScenarioOptimizationResult,
        optimize_scenario,
        plot_optimal_scenario,
    )
    from .scenarios import (  # noqa: F401
        apply_scenario_shocks,
        compare_scenario_margin_curves,
        plot_scenarios,
        simulate_scenarios,
        summarize_scenarios,
    )
    from .sdk import (  # noqa: F401
        DryRunResult,
        Forecaster,
        JobTrace,
        ModelResult,
        Registry,
        RunResult,
        build_best_params_frame,
        build_explain_frame,
        build_trace_frame,
        plot_trace,
    )
    from .similarity import (  # noqa: F401
        SIMILARITY_METHODS,
        VALID_SIMILARITY_METHODS,
        SimilarityMatrixResult,
        SimilarityMethod,
        cluster_by_similarity_df,
        compute_similarity_matrix,
        embed_series_df,
        extract_foundation_embeddings,
        find_similar_series_df,
        plot_similarity_map,
    )
    from .simplex import (  # noqa: F401
        DEFAULT_ZERO_DELTA,
        SIMPLEX_TRANSFORMS,
        SIMPLICIAL_TRANSFORMS,
        apply_simplex_series_transform,
        close_simplex,
        close_simplex_predictions,
        clr_inverse,
        clr_transform,
        forecast_simplex_panel,
        helmert_sub_matrix,
        ilr_inverse,
        ilr_transform,
        inverse_clr_transform,
        inverse_ilr_transform,
        inverse_simplex_series_transform,
        inverse_transform_simplex_series,
        multiplicative_zero_replacement,
        replace_compositional_zeros,
        should_close_simplex,
        transform_simplex_series,
        verify_simplex,
        verify_simplex_closure,
        verify_simplex_coherence,
    )
    from .temporal_reconciliation import (  # noqa: F401
        DEFAULT_TEMPORAL_FACTORS,
        TEMPORAL_RECONCILIATION_METHODS,
        TemporalHierarchySpec,
        build_temporal_hierarchy,
        build_temporal_summing_matrix,
        cross_temporal_mint,
        default_temporal_aggregations,
        reconcile_cross_temporal,
        reconcile_forecasts_temporal,
        reconcile_series_temporal,
        reconcile_temporal_forecasts,
        reconcile_temporal_matrix,
        reconcile_temporal_series,
        resolve_temporal_aggregations,
        sample_coherent_paths,
        temporal_reconcile_df,
        verify_cross_temporal_coherence,
        verify_temporal_coherence,
    )
    from .worker import CellResult, run_cell  # noqa: F401


def __getattr__(name: str) -> object:
    """PEP 562 lazy loader: resolve a heavy public name on first access, then cache it."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module_name, attr = target
    value = getattr(import_module(module_name, __name__), attr)
    globals()[name] = value  # cache so subsequent lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
