"""CLI entrypoint for `scale_forecasting.bq_agent` (Knowledge Catalog & BigQuery Registry Agent).

Examples::

    # List the 32 verified Golden NL <-> SQL queries offline (zero GCP calls):
    python -m scale_forecasting.bq_agent --list-golden-queries

    # Provision/refresh the stateful BigQuery DataAgent + Knowledge Catalog Glossary:
    python -m scale_forecasting.bq_agent \\
        --project your-gcp-project-id --dataset scale_forecasting --ensure-agent

    # Ask a question against the latest completed run (or pass --run-id / --conversation-id):
    python -m scale_forecasting.bq_agent \\
        --project your-gcp-project-id --dataset scale_forecasting \\
        --ask "Which model won and what was the FVA lift?"
"""

from __future__ import annotations

import argparse
import json
import sys

from ..errors import configure_cli_logging
from .client import ask_registry, delete_registry_agent, ensure_registry_agent, get_registry_agent
from .golden_queries import build_golden_queries
from .probe import probe_bq_agent_access


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scale_forecasting.bq_agent",
        description=(
            "Provision Knowledge Catalog metadata & stateful BigQuery Conversational Analytics "
            "Agents over a scale-forecasting BigQuery Registry, or ask natural-language questions."
        ),
    )
    parser.add_argument(
        "--project", default=None, help="Google Cloud project ID (defaults to SF_PROJECT_ID)."
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="BigQuery registry dataset ID (defaults to SF_REGISTRY_DATASET_ID / SF_DATASET_ID).",
    )
    parser.add_argument(
        "--agent-id",
        default=None,
        help="Explicit DataAgent ID (defaults to sf-registry-<dataset>).",
    )
    parser.add_argument(
        "--location",
        default="global",
        help="Conversational Analytics API location (default: global).",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="Region for Knowledge Catalog DataProfile scans (default: SF_REGION or us-central1).",
    )
    parser.add_argument(
        "--include-sources",
        choices=("auto", "none", "explicit"),
        default="auto",
        help="Source table discovery policy for DataAgent scope (default: auto).",
    )
    parser.add_argument(
        "--source-table",
        action="append",
        default=[],
        help="Explicit source table FQN when --include-sources=explicit (repeatable).",
    )
    parser.add_argument(
        "--ensure-agent",
        "--ensure",
        dest="ensure_agent",
        action="store_true",
        help="Create or update the stateful BigQuery DataAgent.",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help=(
            "Check Conversational Analytics & Knowledge Catalog API readiness without raising "
            "(exits 0 with structured status JSON)."
        ),
    )
    parser.add_argument(
        "--no-glossary",
        action="store_true",
        help="Skip Knowledge Catalog Business Glossary sync during --ensure-agent.",
    )
    parser.add_argument(
        "--run-scans",
        action="store_true",
        help="Provision and trigger Knowledge Catalog DataProfile scans.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print the registered DataAgent metadata JSON if present.",
    )
    parser.add_argument(
        "--delete-agent", action="store_true", help="Delete the registered DataAgent."
    )
    parser.add_argument(
        "--ask", default=None, help="Natural-language question to ask the BigQuery Registry Agent."
    )
    parser.add_argument(
        "--run-id", default="latest", help="Target run_id, prefix, 'latest' (default), or 'all'."
    )
    parser.add_argument(
        "--conversation-id", default=None, help="Continue a stateful multi-turn conversation."
    )
    parser.add_argument(
        "--stateless",
        action="store_true",
        help="Send --ask without creating a persistent conversation.",
    )
    parser.add_argument(
        "--list-golden-queries",
        action="store_true",
        help="Print the 32 verified Golden NL <-> SQL pairs offline.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_cli_logging()
    args = _build_parser().parse_args(argv)

    if args.list_golden_queries:
        queries = build_golden_queries(
            args.project or "your-gcp-project-id",
            args.dataset or "scale_forecasting",
        )
        payload = [{"category": q.category, "question": q.question, "sql": q.sql} for q in queries]
        print(json.dumps(payload, indent=2))
        return 0

    if args.probe:
        probe_res = probe_bq_agent_access(
            project_id=args.project,
            dataset_id=args.dataset,
            agent_id=args.agent_id,
            location=args.location,
            region=args.region,
        )
        print(json.dumps(probe_res.to_dict(), indent=2))
        return 0

    if args.delete_agent:
        deleted = delete_registry_agent(
            project_id=args.project,
            dataset_id=args.dataset,
            agent_id=args.agent_id,
            location=args.location,
        )
        print(json.dumps({"deleted": deleted}, indent=2))
        return 0

    if args.status:
        info = get_registry_agent(
            project_id=args.project,
            dataset_id=args.dataset,
            agent_id=args.agent_id,
            location=args.location,
        )
        print(json.dumps(info or {"registered": False}, indent=2))
        return 0

    if args.ensure_agent or args.run_scans:
        reg = ensure_registry_agent(
            project_id=args.project,
            dataset_id=args.dataset,
            agent_id=args.agent_id,
            location=args.location,
            region=args.region,
            include_sources=args.include_sources,
            source_tables=args.source_table or None,
            sync_glossary=not args.no_glossary,
            run_scans=args.run_scans,
        )
        if args.json or not args.ask:
            print(json.dumps(reg.to_dict(), indent=2))

    if args.ask:
        resp = ask_registry(
            args.ask,
            run_id=args.run_id,
            conversation_id=args.conversation_id,
            stateful=not args.stateless,
            project_id=args.project,
            dataset_id=args.dataset,
            agent_id=args.agent_id,
            location=args.location,
        )
        if args.json:
            print(json.dumps(resp.to_dict(), indent=2, default=str))
        else:
            print(
                f"Agent: {resp.agent_name} "
                f"(conversation={resp.conversation_id}, run_id={resp.run_id})"
            )
            print(f"\n{resp.answer}\n")
            if resp.sql:
                print("--- Generated SQL ---")
                print(resp.sql)
            if not resp.df.empty:
                print("\n--- Result Preview ---")
                print(resp.df.head(20).to_string(index=False))
        return 0

    if not (args.ensure_agent or args.run_scans):
        _build_parser().print_help(sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
