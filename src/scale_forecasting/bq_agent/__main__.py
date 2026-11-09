"""CLI entrypoint for `scale_forecasting.bq_agent` (Knowledge Catalog & BigQuery Registry Agent).

Examples::

    # Print clickable Google Cloud Console & BigQuery Studio Agent deep links:
    python -m scale_forecasting.bq_agent --links

    # Interactive multi-turn stateful conversation in the terminal:
    python -m scale_forecasting.bq_agent --chat

    # List the 32 verified Golden NL <-> SQL queries offline (zero GCP calls):
    python -m scale_forecasting.bq_agent --list-golden-queries

    # Provision/refresh the stateful BigQuery DataAgent + Knowledge Catalog Glossary:
    python -m scale_forecasting.bq_agent \\
        --project your-gcp-project-id --dataset scale_forecasting --ensure-agent

    # Ask a single question against the latest completed run:
    python -m scale_forecasting.bq_agent \\
        --project your-gcp-project-id --dataset scale_forecasting \\
        --ask "Which model won and what was the FVA lift?"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from typing import Any

from ..errors import configure_cli_logging
from .client import (
    STARTER_QUESTIONS,
    BQChatSession,
    ask_registry,
    delete_registry_agent,
    ensure_registry_agent,
    get_registry_agent,
    resolve_registry_run_id,
    resolve_target_coordinates,
)
from .console_urls import ConsoleLinkSet, build_registry_console_urls
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
        "--links",
        action="store_true",
        help="Print clickable Google Cloud Console & BigQuery Studio Agent deep links.",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help=(
            "Launch an interactive multi-turn terminal chat session with slash helpers "
            "(/sql, /run, /starters, /links, /new, /quit)."
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


def _resolve_cli_links(
    *,
    project_id: str | None,
    dataset_id: str | None,
    region: str | None,
    run_id: str | None,
    agent_id: str | None,
) -> ConsoleLinkSet:
    """Resolve Console links from CLI args, `SF_*` environment, or project-neutral fallbacks."""
    eff_proj: str | None = project_id
    eff_ds = dataset_id or "scale_forecasting"
    eff_region = region or os.environ.get("SF_REGION") or "us-central1"
    warehouse_uri = os.environ.get("SF_WAREHOUSE_URI")
    try:
        eff_proj, eff_ds, _, eff_region = resolve_target_coordinates(
            project_id=project_id,
            dataset_id=dataset_id,
            region=region,
        )
    except Exception:  # noqa: BLE001 - allow offline link rendering without SF_* env
        eff_proj = (
            project_id
            or os.environ.get("SF_PROJECT_ID")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or None
        )
        eff_ds = (
            dataset_id
            or os.environ.get("SF_REGISTRY_DATASET_ID")
            or os.environ.get("SF_DATASET_ID")
            or "scale_forecasting"
        )

    resolved_run: str | None = None
    job_rows: list[dict[str, Any]] = []
    if eff_proj and run_id and run_id != "all":
        try:
            resolved_run = resolve_registry_run_id(
                run_id,
                project_id=eff_proj,
                dataset_id=eff_ds,
            )
            if resolved_run:
                from ..registry.jobs import read_run_jobs
                from ..settings import Settings as _Settings

                eff_settings = _Settings(
                    project_id=eff_proj,
                    dataset_id=eff_ds,
                    registry_dataset_id_override=eff_ds,
                    region=eff_region,
                    connection=f"{eff_proj}.{eff_region}.sf-iceberg",
                    warehouse_uri=warehouse_uri or f"gs://{eff_proj}-warehouse/warehouse",
                )
                job_rows = read_run_jobs(resolved_run, settings=eff_settings)
        except Exception:  # noqa: BLE001 - degrade gracefully when offline
            resolved_run = None if run_id == "latest" else run_id

    return build_registry_console_urls(
        project_id=eff_proj,
        dataset_id=eff_ds,
        region=eff_region,
        run_id=resolved_run,
        agent_id=agent_id,
        warehouse_uri=warehouse_uri,
        job_rows=job_rows,
    )


def _run_chat_repl(
    session: BQChatSession,
    *,
    region: str | None = None,
    input_fn: Callable[[str], str] = input,
) -> int:
    """Interactive stateful terminal chat loop with presenter slash commands."""
    links = _resolve_cli_links(
        project_id=session.project_id,
        dataset_id=session.dataset_id,
        region=region,
        run_id=session.run_id,
        agent_id=session.agent_id,
    )
    proj_label = links.project_id or "<auto>"
    print(
        f"=== scale-forecasting Registry Chat ({proj_label}.{links.dataset_id}, "
        f"run_id={session.run_id}) ==="
    )
    print(
        "Commands: /starters  /sql  /run <id|latest|all>  /links  /new  /quit  "
        "(or enter 1-4 after /starters)"
    )

    while True:
        try:
            raw = input_fn(f"\n[{session.run_id or 'all'}]> ")
        except (EOFError, KeyboardInterrupt):
            print("\nSession closed.")
            return 0

        text = raw.strip()
        if not text:
            continue
        lower = text.lower()
        if lower in {"/quit", "/exit", "quit", "exit"}:
            print("Session closed.")
            return 0
        if lower == "/starters":
            print("Starter questions (type 1-4 to ask directly):")
            for idx, starter in enumerate(STARTER_QUESTIONS, start=1):
                print(f"  [{idx}] {starter}")
            continue
        if lower == "/sql":
            if session.last is None or not session.last.sql:
                print("(No SQL recorded on the previous turn.)")
            else:
                print("--- Last Generated SQL ---")
                print(session.last.sql)
            continue
        if lower == "/links":
            cur_links = _resolve_cli_links(
                project_id=session.project_id,
                dataset_id=session.dataset_id,
                region=region,
                run_id=session.run_id,
                agent_id=session.agent_id,
            )
            print(cur_links.format_text(include_views=True))
            continue
        if lower == "/new":
            session.reset()
            print("Started a fresh conversation thread.")
            continue
        if lower.startswith("/run"):
            parts = text.split(maxsplit=1)
            if len(parts) == 1:
                print(
                    f"Current run scope: {session.run_id!r} "
                    "(pass `/run latest`, `/run all`, or an ID)"
                )
            else:
                target = parts[1].strip()
                session.switch_run(target)
                print(f"Switched default run scope to {session.run_id!r}.")
            continue

        if text.isdigit():
            choice = int(text)
            if 1 <= choice <= len(STARTER_QUESTIONS):
                text = STARTER_QUESTIONS[choice - 1]
                print(f"Asking [{choice}]: {text}")

        resp = session.ask(text)
        print(
            f"\nAgent (conversation={resp.conversation_id}, run_id={resp.run_id or 'cross-run'}):"
        )
        print(f"{resp.answer}\n")
        if not resp.df.empty:
            print("--- Result Preview (type /sql to inspect query) ---")
            print(resp.df.head(15).to_string(index=False))


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

    if args.links:
        link_set = _resolve_cli_links(
            project_id=args.project,
            dataset_id=args.dataset,
            region=args.region,
            run_id=args.run_id,
            agent_id=args.agent_id,
        )
        if args.json:
            print(json.dumps(link_set.to_dict(), indent=2))
        else:
            print(link_set.format_text(include_views=True))
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
        if args.json or not (args.ask or args.chat):
            print(json.dumps(reg.to_dict(), indent=2))

    if args.chat:
        chat_session = BQChatSession(
            run_id=args.run_id,
            conversation_id=args.conversation_id,
            stateful=not args.stateless,
            project_id=args.project,
            dataset_id=args.dataset,
            agent_id=args.agent_id,
            location=args.location,
        )
        return _run_chat_repl(chat_session, region=args.region)

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
