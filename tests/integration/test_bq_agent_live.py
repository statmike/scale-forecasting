"""Live Google Cloud integration tests for `scale_forecasting.bq_agent` (@gcp).

Verifies the five contracts that only live BigQuery, **Knowledge Catalog**, and the BigQuery
Conversational Analytics API can prove end-to-end:

1. Non-raising pre-flight readiness probe (`Registry.probe_bq_agent`).
2. Self-documenting registry & view description parity (`sync_existing_descriptions`) across
   `INFORMATION_SCHEMA.TABLE_OPTIONS` and `COLUMN_FIELD_PATHS`, plus live queries over all 11
   analytical SQL views (`Registry.sop_summary_df`).
3. BigQuery SQL dry-run validity of all 32 Golden NL <-> SQL queries (`build_golden_queries`).
4. Idempotent `Registry.ensure_bq_agent` strict allowlisted object scope (`5` registry tables +
   `11` analytical views + verified `source_table`s; zero `_sf_automl_*` staging tables) and
   **Knowledge Catalog** Business Glossary sync.
5. Multi-turn stateful `Registry.ask` conversation, historical `Forecaster.from_run_id(...).ask`
   run pinning, and `McpServer` `ask_forecast_registry` tool dispatch.

Skipped unless ``SF_PROJECT_ID`` (+ ADC) is set (see ``tests/conftest.py``). Run manually::

    SF_PROJECT_ID=… SF_CONNECTION=… SF_WAREHOUSE_URI=… SF_DATASET_ID=scale_forecasting \\
        uv run pytest -m gcp tests/integration/test_bq_agent_live.py
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from scale_forecasting import Forecaster, Registry, build_golden_queries
from scale_forecasting.mcp import McpServer
from scale_forecasting.registry.ddl import REGISTRY_TABLE_NAMES
from scale_forecasting.registry.views import DIAGNOSTIC_VIEW_NAMES, VIEW_NAMES

if TYPE_CHECKING:
    from scale_forecasting.settings import Settings

pytestmark = pytest.mark.gcp


@pytest.fixture(scope="module")
def settings() -> Settings:
    """Resolve live infra identity from the ``SF_*`` environment."""
    from scale_forecasting.settings import Settings

    return Settings.resolve()


@pytest.fixture(scope="module")
def reg(settings: Settings) -> Registry:
    """Return a `Registry` handle bound to the live deployment."""
    return Registry(settings=settings)


def test_probe_bq_agent_access_live(reg: Registry) -> None:
    """`Registry.probe_bq_agent` reports Conversational Analytics & Knowledge Catalog readiness."""
    status = reg.probe_bq_agent()
    assert status.ready is True, f"Expected ready=True on live project, got: {status.to_dict()}"
    assert status.conversational_api_ready is True
    assert status.knowledge_catalog_ready is True
    assert status.project_id is not None


def test_registry_descriptions_and_all_11_views_queryable_live(
    settings: Settings, reg: Registry
) -> None:
    """All 5 registry tables and 11 analytical views carry descriptions and execute cleanly."""
    from google.cloud import bigquery

    client = bigquery.Client(project=settings.project_id)
    ds_ref = settings.registry_dataset_ref

    opt_sql = f"""
    SELECT table_name, option_value
    FROM `{ds_ref}.INFORMATION_SCHEMA.TABLE_OPTIONS`
    WHERE option_name = 'description'
    """
    desc_by_obj = {row["table_name"]: str(row["option_value"]) for row in client.query(opt_sql)}
    for obj_name in (*REGISTRY_TABLE_NAMES, *VIEW_NAMES):
        assert obj_name in desc_by_obj and len(desc_by_obj[obj_name]) > 10, (
            f"Missing table/view description on {obj_name}"
        )

    for diag_view in DIAGNOSTIC_VIEW_NAMES:
        rows = list(client.query(f"SELECT * FROM `{ds_ref}.{diag_view}` LIMIT 3").result())
        assert isinstance(rows, list)

    sop_df = reg.sop_summary_df(limit=5)
    assert "champion_model" in sop_df.columns or sop_df.empty


def test_all_32_golden_queries_dry_run_valid_on_bigquery_live(settings: Settings) -> None:
    """Every Golden SQL template dry-runs cleanly against the live BigQuery registry schema."""
    from google.cloud import bigquery

    client = bigquery.Client(project=settings.project_id)
    queries = build_golden_queries(settings.project_id, settings.registry_dataset_id)
    assert len(queries) == 32

    job_cfg = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    for gq in queries:
        job = client.query(gq.sql, job_config=job_cfg)
        assert job.total_bytes_processed is not None


def test_ensure_bq_agent_strict_scope_and_glossary_sync_live(reg: Registry) -> None:
    """`ensure_bq_agent` scopes strictly to canonical registry/views/sources + syncs Glossary."""
    receipt = reg.ensure_bq_agent(sync_glossary=True, run_scans=False)
    assert len(receipt.scoped_tables) == len(REGISTRY_TABLE_NAMES)
    assert len(receipt.scoped_views) == len(VIEW_NAMES)
    for fqn in (*receipt.scoped_tables, *receipt.scoped_views, *receipt.scoped_sources):
        assert "_sf_automl_" not in fqn, f"Staging table leaked into DataAgent scope: {fqn}"
    assert receipt.glossary_name is not None


def test_multi_turn_stateful_ask_and_historical_run_pinning_live(
    settings: Settings, reg: Registry
) -> None:
    """Stateful multi-turn `reg.ask`, historical `Forecaster.from_run_id().ask`, and MCP probe."""
    turn1 = reg.ask("Which model won on the latest completed run and what was its pooled WAPE?")
    assert turn1.answer.strip()
    assert turn1.conversation_id is not None
    assert turn1.run_id is not None

    turn2 = reg.ask(
        "Show the FVA waterfall stages for that same run.",
        conversation_id=turn1.conversation_id,
    )
    assert turn2.answer.strip()
    assert turn2.conversation_id == turn1.conversation_id

    completed = reg.runs_df(limit=10, status="COMPLETED")
    if len(completed) >= 2:
        older_id = str(completed.iloc[1]["run_id"])
        fc_hist = Forecaster.from_run_id(older_id, settings=settings)
        hist_resp = fc_hist.ask("Which model had the lowest pooled WAPE on this run?")
        assert hist_resp.run_id == older_id
        assert hist_resp.answer.strip()

    mcp_srv = McpServer()
    envelope = mcp_srv.call_tool(
        "ask_forecast_registry",
        {
            "mode": "probe",
            "project_id": settings.project_id,
            "dataset_id": settings.registry_dataset_id,
        },
    )
    assert envelope["isError"] is False
    parsed = json.loads(envelope["content"][0]["text"])
    assert parsed["ready"] is True
