"""CLI entry point for ``python -m scale_forecasting.evolve``."""

from __future__ import annotations

import argparse
import json
import sys

from ..errors import configure_cli_logging
from .runner import EvolveConfig, run_evolution
from .targets import EVOLVE_TARGETS


def main(argv: list[str] | None = None) -> int:
    """Execute the ``scale_forecasting.evolve`` CLI and return exit code ``0`` on success."""
    configure_cli_logging()
    parser = argparse.ArgumentParser(
        prog="scale_forecasting.evolve",
        description=(
            "Opt-in AlphaEvolve & Vertex AI Gemini Evolutionary Harness built on "
            "Forecaster.auto() and the 3-Gate Anti-Overfitting Ratchet."
        ),
    )
    parser.add_argument(
        "--target",
        default="features",
        choices=list(EVOLVE_TARGETS),
        help="Evolutionary target (# EVOLVE-BLOCK): features, ensemble, model, or policy.",
    )
    parser.add_argument(
        "--backend",
        default="alphaevolve",
        choices=["alphaevolve", "vertex_gemini", "mock"],
        help=(
            "Evolution transport backend: 'alphaevolve' (default, requires Gemini Enterprise "
            "license + Discovery Engine AlphaEvolve entitlement), 'vertex_gemini' (backup on "
            "standard Vertex AI Gemini), or 'mock' (offline deterministic test backend)."
        ),
    )
    parser.add_argument(
        "--benchmark",
        default="dgp_01_smooth_seasonal",
        help="Canonical DGP benchmark panel name (default: dgp_01_smooth_seasonal).",
    )
    parser.add_argument(
        "--from-run-ids",
        default=None,
        help=(
            "Comma-separated list of existing BigQuery run_ids for post-run / cross-run "
            "evolution with zero base-model refitting (e.g. 'run_id_a,run_id_b')."
        ),
    )
    parser.add_argument(
        "--dataset-id",
        default="scale_forecasting",
        help=(
            "BigQuery registry dataset ID when --from-run-ids is used (default: scale_forecasting)."
        ),
    )
    parser.add_argument(
        "--tier",
        default="deep_sky",
        help=(
            "Observational Astronomy Exposure Tier (first_light, wide_field, deep_sky, deep_field)."
        ),
    )
    parser.add_argument(
        "--where",
        default="local",
        choices=["auto", "local", "bigquery", "cloud_cpu", "cloud_all"],
        help="Compute sandbox envelope (default: local).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Forecast and lockdown horizon H.",
    )
    parser.add_argument(
        "--max-evaluations",
        type=int,
        default=10,
        help="Maximum candidate evaluations (default: 10).",
    )
    parser.add_argument(
        "--significance-alpha",
        type=float,
        default=0.10,
        help="Diebold-Mariano significance threshold alpha (default: 0.10).",
    )
    parser.add_argument(
        "--project-id",
        default=None,
        help="Google Cloud project ID for 'alphaevolve' or 'vertex_gemini'.",
    )
    parser.add_argument(
        "--location",
        default="global",
        help="Discovery Engine AlphaEvolve location (default: global).",
    )
    parser.add_argument(
        "--region",
        default="us-central1",
        help="Vertex AI region for 'vertex_gemini' backup backend (default: us-central1).",
    )
    parser.add_argument(
        "--model-name",
        default="gemini-3.8-flash",
        help=(
            "Gemini model ID for evolution candidate synthesis "
            "(default: gemini-3.8-flash; accepts any Gemini model ID)."
        ),
    )
    parser.add_argument(
        "--export-path",
        default=None,
        help="Optional file path to export the verified champion Python module.",
    )
    parser.add_argument(
        "--emit-dag",
        default=None,
        help="Optional output path to emit a scheduled Cloud Composer 3 / Airflow DAG.",
    )
    parser.add_argument(
        "--schedule",
        default="@monthly",
        help="Cron/Airflow schedule when --emit-dag is used (default: @monthly).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate target seed AST and temporal causality offline without cloud calls.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON output.",
    )

    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(raw_argv)
    run_ids_tuple: tuple[str, ...] | None = None
    if args.from_run_ids:
        parsed_ids = [r.strip() for r in str(args.from_run_ids).split(",") if r.strip()]
        if parsed_ids:
            run_ids_tuple = tuple(parsed_ids)

    explicit_target = any(a == "--target" or a.startswith("--target=") for a in raw_argv)
    eff_target = "ensemble" if (run_ids_tuple and not explicit_target) else args.target

    cfg = EvolveConfig(
        target=eff_target,
        backend=args.backend,
        tier=args.tier,
        where=args.where,
        horizon=args.horizon,
        max_evaluations=args.max_evaluations,
        significance_alpha=args.significance_alpha,
        project_id=args.project_id,
        location=args.location,
        region=args.region,
        model_name=args.model_name,
        export_path=args.export_path,
        dry_run=args.dry_run,
        from_run_ids=run_ids_tuple,
        dataset_id=args.dataset_id,
    )
    result = run_evolution(
        None if run_ids_tuple else args.benchmark,
        config=cfg,
        from_run_ids=run_ids_tuple,
        dataset_id=args.dataset_id,
    )

    if args.emit_dag:
        result.emit_airflow_dag(args.emit_dag, schedule=args.schedule)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.summary())
        if not result.leaderboard_df.empty:
            print("\nCandidate Evaluation Leaderboard:")
            print(result.leaderboard_df.to_string(index=False))
    return 0


_main = main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
