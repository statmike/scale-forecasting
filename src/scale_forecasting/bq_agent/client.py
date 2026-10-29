"""Stateful BigQuery `DataAgent` lifecycle & multi-turn conversational `ask_registry` client.

Talks to the GA ``v1`` REST surface of ``geminidataanalytics.googleapis.com`` to:
1. Register or update a stateful BigQuery Conversational Analytics Agent (`DataAgent`) scoped
   strictly to a specific `scale-forecasting` registry dataset (5 tables + 11 analytical views)
   and its discovered source tables.
2. Synchronize table/view/column descriptions and optional **Knowledge Catalog** Business Glossary
   terms and DataProfile scans.
3. Execute stateful multi-turn (`conversations`) or stateless natural-language analytics queries,
   returning `BQAgentResponse` (`.answer`, `.sql`, `.df`, `.chart_spec`, `.conversation_id`).
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from ..errors import ConfigError, RegistryError
from ..registry.ddl import REGISTRY_TABLE_NAMES, SOURCE_TABLE_NAMES
from ..registry.metadata import sync_existing_descriptions
from ..registry.reads import read_recent_runs
from ..registry.tables import bigquery_client, ensure_views, resolve_settings
from ..registry.views import VIEW_NAMES
from .context import (
    BQAgentScope,
    IncludeSourcesMode,
    build_published_context,
    discover_registry_scope,
)
from .glossary import sync_knowledge_catalog_glossary
from .rest import (
    GEMINI_ANALYTICS_BASE_URL,
    build_authorized_session,
    request_json,
    wait_for_lro,
)
from .scans import ensure_knowledge_catalog_scans

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

    from ..settings import Settings

logger = logging.getLogger(__name__)

_CROSS_RUN_PHRASES: tuple[str, ...] = (
    "recent runs",
    "all runs",
    "across runs",
    "how many runs",
    "list runs",
    "compare runs",
    "which source tables",
)


@dataclass(frozen=True)
class BQAgentRegistration:
    """Receipt returned by `ensure_registry_agent`."""

    agent_name: str
    agent_id: str
    project_id: str
    dataset_id: str
    location: str
    created: bool
    scoped_tables: tuple[str, ...]
    scoped_views: tuple[str, ...]
    scoped_sources: tuple[str, ...]
    descriptions_synced: tuple[str, ...] = ()
    glossary_name: str | None = None
    data_scans: tuple[str, ...] = ()

    @property
    def console_url(self) -> str:
        """Direct Google Cloud Console BigQuery Studio URL for this project."""
        return f"https://console.cloud.google.com/bigquery?project={self.project_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "agent_id": self.agent_id,
            "project_id": self.project_id,
            "dataset_id": self.dataset_id,
            "location": self.location,
            "created": self.created,
            "scoped_tables": list(self.scoped_tables),
            "scoped_views": list(self.scoped_views),
            "scoped_sources": list(self.scoped_sources),
            "descriptions_synced": list(self.descriptions_synced),
            "glossary_name": self.glossary_name,
            "data_scans": list(self.data_scans),
            "console_url": self.console_url,
        }


@dataclass(frozen=True)
class BQAgentResponse:
    """Structured answer from a BigQuery Registry `DataAgent` or offline workbench router."""

    question: str
    answer: str
    sql: str
    df: pd.DataFrame
    chart_spec: dict[str, Any] | None = None
    conversation_id: str | None = None
    agent_name: str | None = None
    run_id: str | None = None
    mode: Literal["bigquery_agent", "offline"] = "bigquery_agent"
    raw_messages: tuple[dict[str, Any], ...] = field(default=(), repr=False)

    def _repr_markdown_(self) -> str:
        """Rich Markdown rendering in Jupyter / Colab notebooks."""
        header = f"### Forecast Registry Agent (`{self.mode}`)"
        if self.run_id:
            header += f" — Run `{self.run_id}`"
        parts = [header, "", self.answer or "*(No narrative text returned.)*"]
        if self.sql:
            parts.extend(["", "```sql", self.sql.strip(), "```"])
        if not self.df.empty:
            try:
                preview = self.df.head(10).to_markdown(index=False)
            except Exception:  # noqa: BLE001 - optional tabulate dependency fallback
                preview = "```text\n" + self.df.head(10).to_string(index=False) + "\n```"
            if preview:
                parts.extend(["", preview])
        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "sql": self.sql,
            "rows": self.df.head(100).to_dict(orient="records"),
            "row_count": len(self.df),
            "has_chart_spec": self.chart_spec is not None,
            "conversation_id": self.conversation_id,
            "agent_name": self.agent_name,
            "run_id": self.run_id,
            "mode": self.mode,
        }


def default_agent_id(dataset_id: str) -> str:
    """Return the deterministic `DataAgent` resource ID for ``dataset_id``."""
    slug = re.sub(r"[^a-z0-9-]+", "-", dataset_id.lower()).strip("-")
    base = f"sf-registry-{slug}" if slug else "sf-registry-agent"
    if not base[0].isalpha():
        base = f"sf-{base}"
    return base[:62].rstrip("-")


def resolve_target_coordinates(
    *,
    project_id: str | None = None,
    dataset_id: str | None = None,
    region: str | None = None,
    settings: Settings | None = None,
) -> tuple[str, str, str, str]:
    """Resolve ``(project_id, registry_dataset_id, source_dataset_id, region)``."""
    if project_id and dataset_id:
        return (project_id, dataset_id, dataset_id, region or "us-central1")
    resolved = resolve_settings(settings)
    eff_proj = project_id or resolved.project_id
    eff_reg_ds = dataset_id or resolved.registry_dataset_id
    eff_src_ds = dataset_id or resolved.dataset_id
    eff_region = region or resolved.region
    return (eff_proj, eff_reg_ds, eff_src_ds, eff_region)


def resolve_registry_run_id(
    run_id: str | None = "latest",
    *,
    question: str = "",
    project_id: str | None = None,
    dataset_id: str | None = None,
    settings: Settings | None = None,
) -> str | None:
    """Resolve ``'latest'``, ``'all'``, an exact ``run_id``, or a ``run_name`` prefix.

    - ``run_id='all'`` returns ``None`` (registry-wide cross-run scope).
    - ``run_id=None`` or ``'latest'`` returns ``None`` if ``question`` explicitly asks a cross-run
      question, otherwise resolves the newest ``COMPLETED`` run in ``v_run_summary`` (falling back
      to the newest run of any status).
    - Any other string matches exact ``run_id`` first, then ``startswith(prefix)`` against recent
      runs in ``v_run_summary`` so callers can pass a short prefix like ``'ensemble-demo'``.
    """
    if run_id == "all":
        return None
    q_low = question.lower()
    if (run_id is None or run_id == "latest") and any(p in q_low for p in _CROSS_RUN_PHRASES):
        return None

    eff_settings = settings
    if eff_settings is None and project_id and dataset_id:
        from ..settings import Settings as _Settings

        eff_settings = _Settings(
            project_id=project_id,
            dataset_id=dataset_id,
            registry_dataset_id_override=dataset_id,
            connection=f"{project_id}.us-central1.sf-iceberg",
            warehouse_uri=f"gs://{project_id}-warehouse/warehouse",
        )

    try:
        recent = read_recent_runs(limit=100, settings=eff_settings)
    except Exception:  # noqa: BLE001 - registry may be empty or offline in unit tests
        return None if run_id in (None, "latest") else run_id

    if not recent:
        return None if run_id in (None, "latest") else run_id

    if run_id in (None, "latest"):
        for row in recent:
            if str(row.get("status", "")).upper() == "COMPLETED":
                return str(row["run_id"])
        return str(recent[0]["run_id"])

    hint = run_id.strip()
    for row in recent:
        if str(row.get("run_id")) == hint:
            return hint
    prefix_matches = [str(r["run_id"]) for r in recent if str(r.get("run_id", "")).startswith(hint)]
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    if len(prefix_matches) > 1:
        return prefix_matches[0]
    return hint


def ensure_registry_agent(
    *,
    project_id: str | None = None,
    dataset_id: str | None = None,
    agent_id: str | None = None,
    display_name: str | None = None,
    location: str = "global",
    region: str | None = None,
    include_sources: IncludeSourcesMode = "auto",
    source_tables: Sequence[str] | None = None,
    additional_registries: Sequence[str] = (),
    sync_descriptions: bool = True,
    sync_glossary: bool = True,
    run_scans: bool = False,
    settings: Settings | None = None,
    bq_client: Any = None,
    session: Any = None,
) -> BQAgentRegistration:
    """Idempotently provision/update the BigQuery Conversational Analytics `DataAgent`.

    1. Refreshes all 11 analytical SQL views and backfills table, view, and column descriptions on
       pre-existing registries when ``sync_descriptions=True``.
    2. Discovers the strict allowlisted `BQAgentScope` (excluding unrelated/staging tables).
    3. Optionally syncs the **Knowledge Catalog** Business Glossary & EntryLinks
       (``sync_glossary=True``) and DataProfile scans (``run_scans=True``).
    4. Creates or updates ``projects/{project}/locations/{location}/dataAgents/{agent_id}`` via
       ``dataAgents:createSync`` / ``:updateSync``.
    """
    eff_proj, eff_reg_ds, eff_src_ds, eff_region = resolve_target_coordinates(
        project_id=project_id,
        dataset_id=dataset_id,
        region=region,
        settings=settings,
    )
    eff_agent_id = agent_id or default_agent_id(eff_reg_ds)
    client = bq_client if bq_client is not None else bigquery_client(eff_proj)

    patched_descriptions: list[str] = []
    if sync_descriptions:
        from ..settings import Settings as _Settings

        view_settings = settings or _Settings(
            project_id=eff_proj,
            dataset_id=eff_src_ds,
            registry_dataset_id_override=eff_reg_ds if eff_reg_ds != eff_src_ds else "",
            region=eff_region,
            connection=f"{eff_proj}.{eff_region}.sf-iceberg",
            warehouse_uri=f"gs://{eff_proj}-warehouse/warehouse",
        )
        if bq_client is None:
            ensure_views(settings=view_settings)
        patched_descriptions.extend(
            sync_existing_descriptions(
                client,
                f"{eff_proj}.{eff_reg_ds}",
                (*REGISTRY_TABLE_NAMES, *VIEW_NAMES),
            )
        )
        patched_descriptions.extend(
            sync_existing_descriptions(
                client,
                f"{eff_proj}.{eff_src_ds}",
                (
                    *SOURCE_TABLE_NAMES,
                    "source_series_covariates_iceberg",
                    "source_series_covariates_native",
                ),
            )
        )

    scope: BQAgentScope = discover_registry_scope(
        eff_proj,
        eff_reg_ds,
        include_sources=include_sources,
        source_tables=source_tables,
        default_source_dataset_id=eff_src_ds,
        additional_registries=additional_registries,
        client=client,
    )
    if not scope.registry_tables and not scope.registry_views:
        raise ConfigError(
            f"No scale-forecasting registry tables or views found in `{eff_proj}.{eff_reg_ds}`. "
            "Run `Registry().init()` or `ensure_tables()` first."
        )

    # Also backfill descriptions on any discovered shipped source tables in cross-datasets.
    if sync_descriptions and scope.source_tables:
        by_ds: dict[str, list[str]] = {}
        for s_ref in scope.source_tables:
            by_ds.setdefault(f"{s_ref.project_id}.{s_ref.dataset_id}", []).append(s_ref.table_id)
        for ds_ref, tbls in by_ds.items():
            for patched in sync_existing_descriptions(client, ds_ref, tbls):
                if patched not in patched_descriptions:
                    patched_descriptions.append(patched)

    sess = session or build_authorized_session(eff_proj)

    glossary_name: str | None = None
    if sync_glossary:
        try:
            g_res = sync_knowledge_catalog_glossary(
                scope,
                bq_location=eff_region,
                session=sess,
            )
            glossary_name = str(g_res["glossary_name"])
        except Exception as exc:  # noqa: BLE001 - optional catalog permission degradation
            logger.warning("Knowledge Catalog glossary sync skipped: %s", exc)

    scan_names: list[str] = []
    if run_scans:
        try:
            s_res = ensure_knowledge_catalog_scans(scope, region=eff_region, session=sess)
            scan_names = list(s_res.get("scans_ensured", ()))
        except Exception as exc:  # noqa: BLE001 - optional DataScan permission degradation
            logger.warning("Knowledge Catalog DataScan provisioning skipped: %s", exc)

    published_ctx = build_published_context(scope)
    parent = f"projects/{eff_proj}/locations/{location}"
    agent_name = f"{parent}/dataAgents/{eff_agent_id}"
    agent_url = f"{GEMINI_ANALYTICS_BASE_URL}/{agent_name}"
    agent_payload: dict[str, Any] = {
        "displayName": display_name or f"scale-forecasting Registry Agent ({eff_reg_ds})",
        "description": (
            f"Stateful BigQuery Conversational Analytics Agent over `{eff_proj}.{eff_reg_ds}` "
            f"({len(scope.registry_tables)} registry tables, {len(scope.registry_views)} "
            f"analytical views, and {len(scope.source_tables)} discovered source table(s))."
        ),
        "dataAnalyticsAgent": {
            "publishedContext": published_ctx,
            "stagingContext": published_ctx,
        },
    }

    existing_agent = request_json(
        sess,
        "GET",
        agent_url,
        allow_not_found=True,
        service_label="BigQuery Conversational Analytics",
    )
    created = existing_agent is None
    if created:
        resp = request_json(
            sess,
            "POST",
            f"{GEMINI_ANALYTICS_BASE_URL}/{parent}/dataAgents:createSync",
            params={"dataAgentId": eff_agent_id},
            json_body=agent_payload,
            service_label="BigQuery Conversational Analytics",
        )
        wait_for_lro(
            sess,
            resp,
            base_url=GEMINI_ANALYTICS_BASE_URL,
            service_label="BigQuery Conversational Analytics",
        )
    else:
        resp = request_json(
            sess,
            "PATCH",
            f"{agent_url}:updateSync",
            params={
                "updateMask": (
                    "displayName,description,"
                    "dataAnalyticsAgent.publishedContext,"
                    "dataAnalyticsAgent.stagingContext"
                )
            },
            json_body=agent_payload,
            service_label="BigQuery Conversational Analytics",
        )
        wait_for_lro(
            sess,
            resp,
            base_url=GEMINI_ANALYTICS_BASE_URL,
            service_label="BigQuery Conversational Analytics",
        )

    return BQAgentRegistration(
        agent_name=agent_name,
        agent_id=eff_agent_id,
        project_id=eff_proj,
        dataset_id=eff_reg_ds,
        location=location,
        created=created,
        scoped_tables=tuple(r.fqn for r in scope.registry_tables),
        scoped_views=tuple(r.fqn for r in scope.registry_views),
        scoped_sources=tuple(r.fqn for r in scope.source_tables),
        descriptions_synced=tuple(patched_descriptions),
        glossary_name=glossary_name,
        data_scans=tuple(scan_names),
    )


def get_registry_agent(
    *,
    project_id: str | None = None,
    dataset_id: str | None = None,
    agent_id: str | None = None,
    location: str = "global",
    settings: Settings | None = None,
    session: Any = None,
) -> dict[str, Any] | None:
    """Fetch the registered BigQuery `DataAgent` metadata, or ``None`` if not yet registered."""
    eff_proj, eff_reg_ds, _, _ = resolve_target_coordinates(
        project_id=project_id,
        dataset_id=dataset_id,
        settings=settings,
    )
    eff_agent_id = agent_id or default_agent_id(eff_reg_ds)
    sess = session or build_authorized_session(eff_proj)
    url = (
        f"{GEMINI_ANALYTICS_BASE_URL}/projects/{eff_proj}"
        f"/locations/{location}/dataAgents/{eff_agent_id}"
    )
    res = request_json(
        sess,
        "GET",
        url,
        allow_not_found=True,
        service_label="BigQuery Conversational Analytics",
    )
    return dict(res) if isinstance(res, dict) else None


def delete_registry_agent(
    *,
    project_id: str | None = None,
    dataset_id: str | None = None,
    agent_id: str | None = None,
    location: str = "global",
    settings: Settings | None = None,
    session: Any = None,
) -> bool:
    """Delete the registered BigQuery `DataAgent` if it exists; returns ``True`` when deleted."""
    eff_proj, eff_reg_ds, _, _ = resolve_target_coordinates(
        project_id=project_id,
        dataset_id=dataset_id,
        settings=settings,
    )
    eff_agent_id = agent_id or default_agent_id(eff_reg_ds)
    sess = session or build_authorized_session(eff_proj)
    url = (
        f"{GEMINI_ANALYTICS_BASE_URL}/projects/{eff_proj}"
        f"/locations/{location}/dataAgents/{eff_agent_id}"
    )
    existing = request_json(
        sess,
        "GET",
        url,
        allow_not_found=True,
        service_label="BigQuery Conversational Analytics",
    )
    if existing is None:
        return False
    resp = request_json(
        sess,
        "DELETE",
        f"{url}:deleteSync",
        service_label="BigQuery Conversational Analytics",
    )
    if isinstance(resp, dict):
        wait_for_lro(
            sess,
            resp,
            base_url=GEMINI_ANALYTICS_BASE_URL,
            service_label="BigQuery Conversational Analytics",
        )
    return True


def parse_chat_messages(raw_payload: Any) -> list[dict[str, Any]]:
    """Normalize a `:chat` JSON array, dict, or NDJSON text stream into message dicts."""
    if isinstance(raw_payload, list):
        return [m for m in raw_payload if isinstance(m, dict)]
    if isinstance(raw_payload, dict):
        if "messages" in raw_payload and isinstance(raw_payload["messages"], list):
            return [m for m in raw_payload["messages"] if isinstance(m, dict)]
        return [raw_payload]
    if isinstance(raw_payload, str):
        stripped = raw_payload.strip()
        if not stripped:
            return []
        try:
            decoded = json.loads(stripped)
            return parse_chat_messages(decoded)
        except ValueError:
            msgs: list[dict[str, Any]] = []
            for line in stripped.splitlines():
                line = line.strip().rstrip(",")
                if not line or line in ("[", "]"):
                    continue
                try:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        msgs.append(item)
                except ValueError:
                    continue
            return msgs
    return []


def extract_response_artifacts(
    messages: Sequence[dict[str, Any]],
) -> tuple[str, str, pd.DataFrame, dict[str, Any] | None]:
    """Extract ``(answer_text, generated_sql, result_df, chart_spec)`` from `:chat` messages."""
    import pandas as pd

    final_parts: list[str] = []
    progress_parts: list[str] = []
    error_parts: list[str] = []
    sql = ""
    rows: list[dict[str, Any]] = []
    chart_spec: dict[str, Any] | None = None

    for msg in messages:
        sys_msg = msg.get("systemMessage")
        if not isinstance(sys_msg, dict):
            continue

        text_blk = sys_msg.get("text")
        if isinstance(text_blk, dict):
            text_type = str(text_blk.get("textType", "FINAL_RESPONSE"))
            joined = "".join(str(p) for p in text_blk.get("parts", ()) if p)
            if joined:
                if text_type == "FINAL_RESPONSE":
                    final_parts.append(joined)
                elif text_type != "THOUGHT":
                    progress_parts.append(joined)

        data_blk = sys_msg.get("data")
        if isinstance(data_blk, dict):
            gen_sql = data_blk.get("generatedSql")
            if isinstance(gen_sql, str) and gen_sql.strip():
                sql = gen_sql.strip()
            else:
                matched = data_blk.get("matchedQuery")
                if isinstance(matched, dict):
                    ex_q = matched.get("exampleQuery")
                    if isinstance(ex_q, dict) and isinstance(ex_q.get("sqlQuery"), str):
                        sql = ex_q["sqlQuery"].strip()
                if not sql and isinstance(data_blk.get("query"), dict):
                    q_sql = data_blk["query"].get("sql")
                    if isinstance(q_sql, str) and q_sql.strip():
                        sql = q_sql.strip()
            res_blk = data_blk.get("result")
            if isinstance(res_blk, dict) and isinstance(res_blk.get("data"), list):
                rows = [r for r in res_blk["data"] if isinstance(r, dict)]

        chart_blk = sys_msg.get("chart")
        if isinstance(chart_blk, dict):
            chart_res = chart_blk.get("result")
            if isinstance(chart_res, dict) and isinstance(chart_res.get("vegaConfig"), dict):
                chart_spec = dict(chart_res["vegaConfig"])

        err_blk = sys_msg.get("error")
        if isinstance(err_blk, dict) and err_blk.get("text"):
            error_parts.append(str(err_blk["text"]))

    answer = "\n\n".join(final_parts) if final_parts else "\n\n".join(progress_parts)
    if not answer and error_parts:
        answer = "Agent reported an error: " + "; ".join(error_parts)
    df = pd.DataFrame.from_records(rows) if rows else pd.DataFrame()
    return answer.strip(), sql, df, chart_spec


def ask_registry(
    question: str,
    *,
    run_id: str | None = "latest",
    conversation_id: str | None = None,
    stateful: bool = True,
    auto_ensure_agent: bool = True,
    project_id: str | None = None,
    dataset_id: str | None = None,
    agent_id: str | None = None,
    location: str = "global",
    settings: Settings | None = None,
    session: Any = None,
) -> BQAgentResponse:
    """Ask a natural-language question against a registered BigQuery Registry `DataAgent`.

    - Automatically resolves ``run_id`` (`'latest'` -> newest COMPLETED run, `'all'` -> cross-run,
      or prefix match against recent runs in ``v_run_summary``).
    - Auto-provisions the `DataAgent` on first use when ``auto_ensure_agent=True`` and the agent
      does not yet exist.
    - Maintains multi-turn conversation memory in Google Cloud when ``stateful=True``; pass back
      ``resp.conversation_id`` to ask follow-up questions in the same session.
    """
    if not question or not question.strip():
        raise ConfigError("question must be a non-empty string")

    eff_proj, eff_reg_ds, _, _ = resolve_target_coordinates(
        project_id=project_id,
        dataset_id=dataset_id,
        settings=settings,
    )
    eff_agent_id = agent_id or default_agent_id(eff_reg_ds)
    parent = f"projects/{eff_proj}/locations/{location}"
    agent_name = f"{parent}/dataAgents/{eff_agent_id}"
    sess = session or build_authorized_session(eff_proj)

    existing = get_registry_agent(
        project_id=eff_proj,
        dataset_id=eff_reg_ds,
        agent_id=eff_agent_id,
        location=location,
        settings=settings,
        session=sess,
    )
    if existing is None:
        if not auto_ensure_agent:
            raise RegistryError(
                f"BigQuery DataAgent `{agent_name}` is not registered yet. "
                "Call `ensure_bq_agent()` first or pass `auto_ensure_agent=True`."
            )
        ensure_registry_agent(
            project_id=eff_proj,
            dataset_id=eff_reg_ds,
            agent_id=eff_agent_id,
            location=location,
            settings=settings,
            session=sess,
        )

    resolved_run_id = resolve_registry_run_id(
        run_id,
        question=question,
        project_id=eff_proj,
        dataset_id=eff_reg_ds,
        settings=settings,
    )
    effective_prompt = question.strip()
    if resolved_run_id:
        effective_prompt = (
            f"[Context: Scope queries to run_id='{resolved_run_id}' unless this question "
            f"explicitly asks to compare multiple runs.]\n{effective_prompt}"
        )

    eff_conv_id = conversation_id
    chat_body: dict[str, Any] = {
        "parent": parent,
        "messages": [{"userMessage": {"text": effective_prompt}}],
    }

    if stateful:
        if not eff_conv_id:
            eff_conv_id = f"sf-conv-{uuid.uuid4().hex[:12]}"
            request_json(
                sess,
                "POST",
                f"{GEMINI_ANALYTICS_BASE_URL}/{parent}/conversations",
                params={"conversationId": eff_conv_id},
                json_body={"agents": [agent_name]},
                service_label="BigQuery Conversational Analytics",
            )
        conv_name = (
            eff_conv_id
            if eff_conv_id.startswith("projects/")
            else f"{parent}/conversations/{eff_conv_id}"
        )
        chat_body["conversationReference"] = {
            "conversation": conv_name,
            "dataAgentContext": {"dataAgent": agent_name},
        }
    else:
        chat_body["dataAgentContext"] = {"dataAgent": agent_name}

    raw_resp = sess.request(
        method="POST",
        url=f"{GEMINI_ANALYTICS_BASE_URL}/{parent}:chat",
        json=chat_body,
        timeout=180.0,
    )
    if raw_resp.status_code >= 400:
        raise RegistryError(
            f"BigQuery Conversational Analytics chat failed (HTTP {raw_resp.status_code}): "
            f"{raw_resp.text.strip()}"
        )

    messages = parse_chat_messages(raw_resp.text)
    answer, sql, df, chart_spec = extract_response_artifacts(messages)

    return BQAgentResponse(
        question=question.strip(),
        answer=answer,
        sql=sql,
        df=df,
        chart_spec=chart_spec,
        conversation_id=eff_conv_id,
        agent_name=agent_name,
        run_id=resolved_run_id,
        mode="bigquery_agent",
        raw_messages=tuple(messages),
    )
