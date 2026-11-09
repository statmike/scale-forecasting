"""Isolated subpackage for Knowledge Catalog & Stateful BigQuery Conversational Analytics Agents.

Turns any `scale-forecasting` BigQuery Registry dataset (`5` registry tables + `11` analytical SQL
views + discovered `source_table`s) into a self-documenting, Knowledge-Catalog-governed,
multi-turn conversational forecasting workbench.
"""

from __future__ import annotations

from .client import (
    STARTER_QUESTIONS,
    BQAgentRegistration,
    BQAgentResponse,
    BQChatSession,
    ask_registry,
    default_agent_id,
    delete_registry_agent,
    ensure_registry_agent,
    get_registry_agent,
    resolve_registry_run_id,
)
from .console_urls import (
    ConsoleLink,
    ConsoleLinkSet,
    build_job_console_url,
    build_registry_console_urls,
)
from .context import (
    BQAgentScope,
    ScopedTableRef,
    build_published_context,
    build_schema_relationships,
    build_system_instruction,
    discover_registry_scope,
)
from .glossary import (
    GLOSSARY_TERMS,
    GlossaryTermSpec,
    build_context_glossary_terms,
    default_glossary_id,
    sync_knowledge_catalog_glossary,
)
from .golden_queries import GoldenQuery, build_golden_queries
from .offline import ask_offline, match_golden_query
from .probe import BQAgentProbeResult, probe_bq_agent_access
from .scans import default_profile_scan_id, ensure_knowledge_catalog_scans

__all__ = [
    "GLOSSARY_TERMS",
    "STARTER_QUESTIONS",
    "BQAgentProbeResult",
    "BQAgentRegistration",
    "BQAgentResponse",
    "BQAgentScope",
    "BQChatSession",
    "ConsoleLink",
    "ConsoleLinkSet",
    "GlossaryTermSpec",
    "GoldenQuery",
    "ScopedTableRef",
    "ask_offline",
    "ask_registry",
    "build_context_glossary_terms",
    "build_golden_queries",
    "build_job_console_url",
    "build_published_context",
    "build_registry_console_urls",
    "build_schema_relationships",
    "build_system_instruction",
    "default_agent_id",
    "default_glossary_id",
    "default_profile_scan_id",
    "delete_registry_agent",
    "discover_registry_scope",
    "ensure_knowledge_catalog_scans",
    "ensure_registry_agent",
    "get_registry_agent",
    "match_golden_query",
    "probe_bq_agent_access",
    "resolve_registry_run_id",
    "sync_knowledge_catalog_glossary",
]
