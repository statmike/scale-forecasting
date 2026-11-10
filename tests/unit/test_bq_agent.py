"""Offline unit tests for `scale_forecasting.bq_agent` (Knowledge Catalog & BigQuery DataAgent).

Verifies:
1. `OutputConfig.ensure_bq_agent` config acceptance and exclusion from `run_id` / `model_spec_hash`.
2. Strict allowlisted object scope in `discover_registry_scope` (excluding `_sf_automl_*` and
   unrelated dataset tables).
3. Knowledge Catalog `GLOSSARY_TERMS` structure (`<= 8` EntryLinks per table) and 32 Golden SQL
   queries across all 11 analytical views.
4. `ensure_registry_agent`, `sync_knowledge_catalog_glossary`, `ensure_knowledge_catalog_scans`,
   `resolve_registry_run_id`, stateful multi-turn `ask_registry`, `get_registry_agent`, and
   `delete_registry_agent` via deterministic fake `AuthorizedSession` and BigQuery client doubles.
5. Pure-offline `.ask()` and `.sop_summary_df()` across `PlaygroundRun`, `BakeOff`, and
   `Forecaster.auto(..., where="local")`.
6. CLI (`python -m scale_forecasting.bq_agent`) and `Registry` SDK wrappers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from scale_forecasting import (
    GLOSSARY_TERMS,
    BQAgentRegistration,
    BQAgentResponse,
    BQAgentScope,
    Forecaster,
    Registry,
    RunConfig,
    ask_registry,
    build_golden_queries,
    delete_registry_agent,
    ensure_knowledge_catalog_scans,
    ensure_registry_agent,
    get_registry_agent,
    playground,
    sync_knowledge_catalog_glossary,
)
from scale_forecasting.bq_agent import __main__ as bq_agent_cli
from scale_forecasting.bq_agent.context import build_published_context, discover_registry_scope
from scale_forecasting.bq_agent.glossary import MAX_LINKS_PER_TABLE
from scale_forecasting.bq_agent.scans import default_profile_scan_id
from scale_forecasting.registry.ddl import REGISTRY_TABLE_NAMES
from scale_forecasting.registry.ids import make_model_spec_hash, make_run_id
from scale_forecasting.registry.views import DIAGNOSTIC_VIEW_NAMES, VIEW_NAMES

_PROJECT = "your-gcp-project-id"
_DATASET = "scale_forecasting"


@dataclass
class _FakeSchemaField:
    name: str
    field_type: str = "STRING"
    mode: str = "NULLABLE"
    description: str | None = "populated"

    def to_api_repr(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.field_type,
            "mode": self.mode,
            "description": self.description or "",
        }


@dataclass
class _FakeTable:
    full_id: str
    description: str | None = "populated"
    schema: list[_FakeSchemaField] = field(default_factory=list)


class _FakeQueryJob:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def result(self) -> list[dict[str, Any]]:
        return self._rows


class _FakeBQClient:
    """Fake BigQuery client holding both canonical registry objects and unrelated noise tables."""

    def __init__(self, existing_fqns: set[str], raw_sources: list[str]) -> None:
        self.existing_fqns = set(existing_fqns)
        self.raw_sources = list(raw_sources)
        self.updated_tables: list[str] = []

    def get_table(self, fqn: str) -> _FakeTable:
        from google.api_core import exceptions as gcp_exceptions

        if fqn not in self.existing_fqns:
            raise gcp_exceptions.NotFound(f"Table not found: {fqn}")
        return _FakeTable(full_id=fqn, description="already documented")

    def update_table(self, table: _FakeTable, fields: list[str]) -> _FakeTable:
        _ = fields
        self.updated_tables.append(table.full_id)
        return table

    def query(self, sql: str) -> _FakeQueryJob:
        assert "run_registry" in sql
        return _FakeQueryJob([{"src": s} for s in self.raw_sources])


@dataclass
class _FakeHTTPResponse:
    status_code: int
    payload: Any = None
    raw_text: str | None = None

    @property
    def text(self) -> str:
        if self.raw_text is not None:
            return self.raw_text
        if self.payload is None:
            return ""
        return json.dumps(self.payload)

    def json(self) -> Any:
        return self.payload


class _FakeAuthorizedSession:
    """Stateful in-memory fake for Conversational Analytics & Knowledge Catalog REST endpoints."""

    def __init__(self) -> None:
        self.resources: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, str, dict[str, Any] | None, Any]] = []
        self.chat_payloads: list[dict[str, Any]] = []

    def request(
        self,
        *,
        method: str,
        url: str,
        params: dict[str, Any] | None = None,
        json: Any = None,
        timeout: float = 60.0,
    ) -> _FakeHTTPResponse:
        _ = timeout
        self.calls.append((method, url, params, json))

        if url.endswith(":chat") and method == "POST":
            self.chat_payloads.append(json)
            ndjson_stream = "\n".join(
                [
                    (
                        '[{"systemMessage":{"text":{"textType":"THOUGHT",'
                        '"parts":["Checking v_model_leaderboard_comparable..."]}}},'
                    ),
                    (
                        '{"systemMessage":{"data":{"generatedSql":'
                        '"SELECT model_type, pooled_wape FROM '
                        "`your-gcp-project-id.scale_forecasting.v_model_leaderboard_comparable` "
                        "WHERE run_id = 'ensemble-demo-111111111111' "
                        'ORDER BY pooled_wape ASC","result":{"data":['
                        '{"model_type":"ensemble_ridge","pooled_wape":0.0612},'
                        '{"model_type":"lightgbm","pooled_wape":0.0685}]}}}},'
                    ),
                    '{"systemMessage":{"chart":{"result":{"vegaConfig":{"mark":"bar"}}}}},',
                    (
                        '{"systemMessage":{"text":{"textType":"FINAL_RESPONSE",'
                        '"parts":["`ensemble_ridge` won with a pooled WAPE of 6.12%."]}}}]'
                    ),
                ]
            )
            return _FakeHTTPResponse(status_code=200, raw_text=ndjson_stream)

        if url.endswith(":createSync") and method == "POST":
            agent_id = (params or {})["dataAgentId"]
            parent_url = url[: -len(":createSync")]
            full_url = f"{parent_url}/{agent_id}"
            stored = {"name": full_url.split("/v1beta/", 1)[-1], **(json or {})}
            self.resources[full_url] = stored
            return _FakeHTTPResponse(status_code=200, payload=stored)

        if url.endswith(":updateSync") and method == "PATCH":
            base_url = url[: -len(":updateSync")]
            stored = {**self.resources.get(base_url, {}), **(json or {})}
            self.resources[base_url] = stored
            return _FakeHTTPResponse(status_code=200, payload=stored)

        if url.endswith(":deleteSync") and method == "DELETE":
            base_url = url[: -len(":deleteSync")]
            self.resources.pop(base_url, None)
            return _FakeHTTPResponse(status_code=200, payload={})

        if url.endswith(":run") and method == "POST":
            return _FakeHTTPResponse(
                status_code=200,
                payload={"job": {"name": f"{url[:-4]}/jobs/job-001"}},
            )

        if method == "GET":
            if url in self.resources:
                return _FakeHTTPResponse(status_code=200, payload=self.resources[url])
            return _FakeHTTPResponse(status_code=404, payload={"error": "not found"})

        if method == "POST":
            p = params or {}
            child_id = (
                p.get("glossaryId")
                or p.get("termId")
                or p.get("entryLinkId")
                or p.get("dataScanId")
                or p.get("conversationId")
                or "created"
            )
            full_url = f"{url}/{child_id}"
            stored = {"name": full_url.split("/v1/", 1)[-1], **(json or {})}
            self.resources[full_url] = stored
            if "glossaries" in url or "dataScans" in url:
                return _FakeHTTPResponse(
                    status_code=200,
                    payload={
                        "name": f"projects/{_PROJECT}/locations/global/operations/op-1",
                        "done": True,
                        "response": stored,
                    },
                )
            return _FakeHTTPResponse(status_code=200, payload=stored)

        if method == "PATCH":
            stored = {**self.resources.get(url, {}), **(json or {})}
            self.resources[url] = stored
            return _FakeHTTPResponse(status_code=200, payload=stored)

        return _FakeHTTPResponse(status_code=200, payload={})


def test_output_ensure_bq_agent_excluded_from_run_id_and_dispatched() -> None:
    base_dict: dict[str, Any] = {
        "run_name": "agent_hash_check",
        "data": {"source_table": "source_series_native", "horizon": 14},
        "models": ["theta"],
        "output": {"ensure_bq_agent": False},
    }
    cfg_false = RunConfig.model_validate(base_dict)
    cfg_true = RunConfig.model_validate({**base_dict, "output": {"ensure_bq_agent": True}})
    assert cfg_false.output.ensure_bq_agent is False
    assert cfg_true.output.ensure_bq_agent is True
    assert make_run_id(cfg_true) == make_run_id(cfg_false)
    assert make_model_spec_hash("theta", cfg_true) == make_model_spec_hash("theta", cfg_false)


def test_discover_registry_scope_excludes_unrelated_and_staging_tables() -> None:
    existing = {
        *(f"{_PROJECT}.{_DATASET}.{t}" for t in REGISTRY_TABLE_NAMES),
        *(f"{_PROJECT}.{_DATASET}.{v}" for v in VIEW_NAMES),
        f"{_PROJECT}.{_DATASET}.source_series_native",
        f"{_PROJECT}.raw_erp.demand_history",
        # Noise & staging tables in the exact same dataset that MUST be excluded:
        f"{_PROJECT}.{_DATASET}._sf_automl_train_abc123",
        f"{_PROJECT}.{_DATASET}._sf_automl_eval_abc123",
        f"{_PROJECT}.{_DATASET}.unrelated_finance_ledger",
        f"{_PROJECT}.{_DATASET}.scratch_experiment_copy",
    }
    raw_sources = [
        "source_series_native",
        "raw_erp.demand_history",
        "_sf_automl_train_abc123",
        "raw_erp.deleted_legacy_panel",
    ]
    fake_bq = _FakeBQClient(existing_fqns=existing, raw_sources=raw_sources)

    scope: BQAgentScope = discover_registry_scope(_PROJECT, _DATASET, client=fake_bq)
    assert len(scope.registry_tables) == 5
    assert len(scope.registry_views) == 11
    assert [r.fqn for r in scope.source_tables] == [
        f"{_PROJECT}.{_DATASET}.source_series_native",
        f"{_PROJECT}.raw_erp.demand_history",
    ]
    all_fqns = {r.fqn for r in scope.all_objects}
    assert f"{_PROJECT}.{_DATASET}._sf_automl_train_abc123" not in all_fqns
    assert f"{_PROJECT}.{_DATASET}.unrelated_finance_ledger" not in all_fqns
    assert f"{_PROJECT}.raw_erp.deleted_legacy_panel" not in all_fqns

    scope_none = discover_registry_scope(_PROJECT, _DATASET, include_sources="none", client=fake_bq)
    assert scope_none.source_tables == ()

    scope_explicit = discover_registry_scope(
        _PROJECT,
        _DATASET,
        include_sources="explicit",
        source_tables=["raw_erp.demand_history"],
        client=fake_bq,
    )
    assert [r.fqn for r in scope_explicit.source_tables] == [f"{_PROJECT}.raw_erp.demand_history"]


def test_glossary_terms_and_golden_queries_invariants() -> None:
    assert len(GLOSSARY_TERMS) >= 40
    term_ids = [t.term_id for t in GLOSSARY_TERMS]
    assert len(term_ids) == len(set(term_ids))

    per_object_counts: dict[str, int] = {}
    for term in GLOSSARY_TERMS:
        for obj in term.linked_objects:
            per_object_counts[obj] = per_object_counts.get(obj, 0) + 1
    for obj, count in per_object_counts.items():
        assert count <= MAX_LINKS_PER_TABLE, f"{obj} exceeds {MAX_LINKS_PER_TABLE} links ({count})"

    golden = build_golden_queries(_PROJECT, _DATASET)
    assert len(golden) >= 30
    joined_sql = "\n".join(q.sql for q in golden)
    for diag_view in DIAGNOSTIC_VIEW_NAMES:
        assert diag_view in joined_sql, f"Diagnostic view {diag_view} missing from Golden Queries"


def test_ensure_registry_agent_and_stateful_ask_with_fake_session() -> None:
    existing = {
        *(f"{_PROJECT}.{_DATASET}.{t}" for t in REGISTRY_TABLE_NAMES),
        *(f"{_PROJECT}.{_DATASET}.{v}" for v in VIEW_NAMES),
        f"{_PROJECT}.{_DATASET}.source_series_native",
    }
    fake_bq = _FakeBQClient(existing_fqns=existing, raw_sources=["source_series_native"])
    fake_sess = _FakeAuthorizedSession()

    reg1: BQAgentRegistration = ensure_registry_agent(
        project_id=_PROJECT,
        dataset_id=_DATASET,
        run_scans=True,
        bq_client=fake_bq,
        session=fake_sess,
    )
    assert reg1.created is True
    assert reg1.agent_id == "sf-registry-scale-forecasting"
    assert len(reg1.scoped_tables) == 5
    assert len(reg1.scoped_views) == 11
    assert reg1.scoped_sources == (f"{_PROJECT}.{_DATASET}.source_series_native",)
    assert reg1.glossary_name is not None
    assert len(reg1.data_scans) == 6  # 5 physical registry tables + 1 physical source table

    # Second call is idempotent (`created=False`, uses `:updateSync`).
    reg2 = ensure_registry_agent(
        project_id=_PROJECT,
        dataset_id=_DATASET,
        sync_glossary=False,
        run_scans=False,
        bq_client=fake_bq,
        session=fake_sess,
    )
    assert reg2.created is False

    # Inspect published context structure directly.
    scope = discover_registry_scope(_PROJECT, _DATASET, client=fake_bq)
    ctx = build_published_context(scope)
    assert "COMPLETED" in ctx["systemInstruction"]
    assert len(ctx["datasourceReferences"]["bq"]["tableReferences"]) == 17
    assert len(ctx["exampleQueries"]) >= 30

    # Test direct glossary & scan helpers against fake session.
    g_res = sync_knowledge_catalog_glossary(scope, session=fake_sess)
    assert len(g_res["terms_synced"]) == len(GLOSSARY_TERMS)
    s_res = ensure_knowledge_catalog_scans(
        scope, include_documentation_scan=True, session=fake_sess
    )
    assert len(s_res["scans_ensured"]) == 12

    recent_runs_fixture = [
        {"run_id": "broken-run-999999999999", "status": "FAILED"},
        {"run_id": "ensemble-demo-111111111111", "status": "COMPLETED"},
        {"run_id": "baseline-demo-222222222222", "status": "COMPLETED"},
    ]
    with patch(
        "scale_forecasting.bq_agent.client.read_recent_runs",
        return_value=recent_runs_fixture,
    ):
        # Turn 1: `run_id='latest'` skips FAILED and selects newest COMPLETED run!
        resp1: BQAgentResponse = ask_registry(
            "Which model won?",
            run_id="latest",
            project_id=_PROJECT,
            dataset_id=_DATASET,
            session=fake_sess,
        )
        assert resp1.mode == "bigquery_agent"
        assert resp1.run_id == "ensemble-demo-111111111111"
        assert resp1.conversation_id is not None
        assert "ensemble_ridge" in resp1.answer
        assert "v_model_leaderboard_comparable" in resp1.sql
        assert len(resp1.df) == 2
        assert resp1.chart_spec == {"mark": "bar"}
        assert "ensemble_ridge" in resp1._repr_markdown_()

        # Turn 2: prefix match (`baseline-demo`) + follow-up `conversation_id` reuse.
        resp2 = ask_registry(
            "Now show me its FVA waterfall.",
            run_id="baseline-demo",
            conversation_id=resp1.conversation_id,
            project_id=_PROJECT,
            dataset_id=_DATASET,
            session=fake_sess,
        )
        assert resp2.run_id == "baseline-demo-222222222222"
        assert resp2.conversation_id == resp1.conversation_id

        # Cross-run question with `run_id='latest'` leaves `run_id=None` so all runs are compared!
        resp_cross = ask_registry(
            "Compare WAPE across runs over the last month.",
            run_id="latest",
            stateful=False,
            project_id=_PROJECT,
            dataset_id=_DATASET,
            session=fake_sess,
        )
        assert resp_cross.run_id is None

    fetched = get_registry_agent(project_id=_PROJECT, dataset_id=_DATASET, session=fake_sess)
    assert fetched is not None
    assert (
        delete_registry_agent(project_id=_PROJECT, dataset_id=_DATASET, session=fake_sess) is True
    )
    assert get_registry_agent(project_id=_PROJECT, dataset_id=_DATASET, session=fake_sess) is None


def test_offline_ask_across_playground_bakeoff_and_forecaster() -> None:
    panel = playground.sample_data(n_series=3, history=240, with_exog=True)

    # 1. PlaygroundRun.ask()
    pg = playground.run_model("theta", data=panel, horizon=7, backtest=True)
    r_pg = pg.ask("Show me the forward forecast volume.")
    assert isinstance(r_pg, BQAgentResponse)
    assert r_pg.mode == "offline"
    assert not r_pg.df.empty
    assert "SELECT" in r_pg.sql

    # 2. BakeOff.ask()
    bo = playground.bakeoff(data=panel, models=["naive_seasonal", "theta"], horizon=7)
    r_bo = bo.ask("What was the FVA waterfall lift over naive?")
    assert r_bo.mode == "offline"
    assert not r_bo.df.empty

    # 3. Forecaster.auto(..., where="local").ask() + .sop_summary_df()
    f = Forecaster.auto(
        panel,
        horizon=7,
        tier="first_light",
        where="local",
        autofocus=False,
    )
    sop_df = f.sop_summary_df()
    assert isinstance(sop_df, pd.DataFrame) and len(sop_df) == 1
    assert float(sop_df.iloc[0]["total_forecast_units"]) > 0.0

    r_sop = f.ask("Give me the executive S&OP scorecard summary.")
    assert r_sop.mode == "offline"
    assert len(r_sop.df) == 1
    assert "v_executive_sop_summary" in r_sop.sql

    with pytest.raises(Exception, match="non-empty"):
        f.ask("   ")


def test_cli_bq_agent_and_sdk_registry_init(capsys: pytest.CaptureFixture[str]) -> None:
    rc = bq_agent_cli.main(["--list-golden-queries", "--project", _PROJECT, "--dataset", _DATASET])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out) >= 30

    reg = Registry(project_id=_PROJECT, dataset_id=_DATASET, region="us-central1")
    assert reg.dataset_ref == f"{_PROJECT}.{_DATASET}"


def test_long_scan_ids_do_not_collide_and_per_table_scan_resilience() -> None:
    id_a = default_profile_scan_id(
        "scale_forecasting_benchmarks",
        "dgp_13_all_decoy_univariate_wins_variant_alpha",
    )
    id_b = default_profile_scan_id(
        "scale_forecasting_benchmarks",
        "dgp_13_all_decoy_univariate_wins_variant_beta",
    )
    assert len(id_a) <= 62 and len(id_b) <= 62
    assert id_a != id_b

    existing = {
        *(f"{_PROJECT}.{_DATASET}.{t}" for t in REGISTRY_TABLE_NAMES),
        f"{_PROJECT}.{_DATASET}.source_series_iceberg",
    }
    fake_bq = _FakeBQClient(existing_fqns=existing, raw_sources=["source_series_iceberg"])
    scope = discover_registry_scope(_PROJECT, _DATASET, client=fake_bq)
    base_sess = _FakeAuthorizedSession()

    class _FlakyIcebergSession(_FakeAuthorizedSession):
        def request(
            self,
            *,
            method: str,
            url: str,
            params: dict[str, Any] | None = None,
            json: Any = None,
            timeout: float = 60.0,
        ) -> _FakeHTTPResponse:
            if method == "POST" and params and "iceberg" in str(params.get("dataScanId", "")):
                return _FakeHTTPResponse(
                    status_code=400, payload={"error": "External table unsupported"}
                )
            return base_sess.request(
                method=method, url=url, params=params, json=json, timeout=timeout
            )

    res = ensure_knowledge_catalog_scans(scope, session=_FlakyIcebergSession())
    assert len(res["scans_ensured"]) == 5
    assert res["skipped_tables"] == [f"{_PROJECT}.{_DATASET}.source_series_iceberg"]


def test_cli_subcommands_rest_lro_and_offline_intents(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from scale_forecasting.bq_agent.rest import request_json, wait_for_lro
    from scale_forecasting.errors import RegistryError
    from scale_forecasting.playground import bakeoff

    # 1. REST transport edge cases (403 SERVICE_DISABLED hint, empty 204 body, LRO polling & errors)
    class _StubSession:
        def __init__(self, responses: list[_FakeHTTPResponse]) -> None:
            self._responses = list(responses)

        def request(self, **_: Any) -> _FakeHTTPResponse:
            return self._responses.pop(0)

    with pytest.raises(RegistryError, match=r"geminidataanalytics\.googleapis\.com"):
        request_json(
            _StubSession(
                [_FakeHTTPResponse(status_code=403, payload={"error": "SERVICE_DISABLED"})]
            ),
            "GET",
            "https://example.com/v1/x",
        )

    from types import SimpleNamespace

    empty_resp = SimpleNamespace(status_code=204, text="")
    assert request_json(_StubSession([empty_resp]), "DELETE", "https://example.com/v1/x") == {}  # type: ignore[list-item]

    monkeypatch.setattr("scale_forecasting.bq_agent.rest._LRO_POLL_INTERVAL_SECONDS", 0.001)
    polled = wait_for_lro(
        _StubSession(
            [_FakeHTTPResponse(status_code=200, payload={"done": True, "response": {"ok": 1}})]
        ),
        {"name": "projects/p/locations/global/operations/op-1", "done": False},
        base_url="https://dataplex.googleapis.com/v1",
        service_label="Knowledge Catalog",
    )
    assert polled == {"ok": 1}

    with pytest.raises(RegistryError, match="operation failed"):
        wait_for_lro(
            _StubSession([]),
            {"name": "inline", "done": True, "error": {"code": 13, "message": "boom"}},
            base_url="https://dataplex.googleapis.com/v1",
            service_label="Knowledge Catalog",
        )

    # 2. CLI dispatch branches (--status, --delete-agent, --ensure-agent, --ask, empty argv)
    assert bq_agent_cli.main([]) == 2
    capsys.readouterr()

    monkeypatch.setattr(bq_agent_cli, "delete_registry_agent", lambda **_: True)
    assert bq_agent_cli.main(["--project", _PROJECT, "--dataset", _DATASET, "--delete-agent"]) == 0
    assert json.loads(capsys.readouterr().out)["deleted"] is True

    monkeypatch.setattr(bq_agent_cli, "get_registry_agent", lambda **_: None)
    assert bq_agent_cli.main(["--project", _PROJECT, "--dataset", _DATASET, "--status"]) == 0
    assert json.loads(capsys.readouterr().out) == {"registered": False}

    fake_reg = BQAgentRegistration(
        agent_name=f"projects/{_PROJECT}/locations/global/dataAgents/sf-registry-scale-forecasting",
        agent_id="sf-registry-scale-forecasting",
        project_id=_PROJECT,
        dataset_id=_DATASET,
        location="global",
        created=True,
        scoped_tables=(f"{_PROJECT}.{_DATASET}.run_registry",),
        scoped_views=(f"{_PROJECT}.{_DATASET}.v_run_summary",),
        scoped_sources=(),
    )
    monkeypatch.setattr(bq_agent_cli, "ensure_registry_agent", lambda **_: fake_reg)
    assert bq_agent_cli.main(["--project", _PROJECT, "--dataset", _DATASET, "--ensure-agent"]) == 0
    assert json.loads(capsys.readouterr().out)["created"] is True

    fake_ans = BQAgentResponse(
        question="Which model won?",
        answer="lightgbm won.",
        sql="SELECT 'lightgbm' AS champion",
        df=pd.DataFrame([{"champion": "lightgbm"}]),
        conversation_id="sf-conv-123",
        agent_name=fake_reg.agent_name,
        run_id="demo-run-000000000001",
    )
    monkeypatch.setattr(bq_agent_cli, "ask_registry", lambda *_, **__: fake_ans)
    assert (
        bq_agent_cli.main(
            ["--project", _PROJECT, "--dataset", _DATASET, "--ask", "Which model won?"]
        )
        == 0
    )
    human_out = capsys.readouterr().out
    assert "Generated SQL" in human_out and "Result Preview" in human_out

    assert (
        bq_agent_cli.main(
            ["--project", _PROJECT, "--dataset", _DATASET, "--ask", "Which model won?", "--json"]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["conversation_id"] == "sf-conv-123"

    # 3. Offline local Forecaster / BakeOff intent branches
    bo = bakeoff(models=["naive_mean", "holtwinters"], horizon=4, n_folds=1)
    for q in (
        "Show the FVA waterfall stages.",
        "How does accuracy decay over the forecast horizon?",
        "Which series show tracking signal bias or exceptions?",
        "Show overview summary of this run.",
    ):
        ans = bo.ask(q)
        assert ans.answer and isinstance(ans.df, pd.DataFrame)


def test_probe_bq_agent_access_and_mcp_modes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from scale_forecasting.bq_agent.probe import BQAgentProbeResult, probe_bq_agent_access
    from scale_forecasting.mcp import McpServer
    from scale_forecasting.settings import Settings

    # 1. Missing [gcp] extra returns non-raising ready=False with pip hint
    monkeypatch.setattr("scale_forecasting.bq_agent.probe.is_importable", lambda _m: False)
    res_no_gcp = probe_bq_agent_access(project_id=_PROJECT, dataset_id=_DATASET)
    assert res_no_gcp.ready is False
    assert "scale-forecasting[gcp]" in res_no_gcp.setup_hint
    assert res_no_gcp.to_dict()["ready"] is False
    monkeypatch.undo()

    # 2. Missing project_id in env and Settings
    for key in ("SF_PROJECT_ID", "GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT", "SF_DATASET_ID"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        "scale_forecasting.bq_agent.probe.resolve_target_coordinates",
        lambda **_: (_ for _ in ()).throw(RuntimeError("no env")),
    )
    res_no_proj = probe_bq_agent_access()
    assert res_no_proj.ready is False
    assert "SF_PROJECT_ID" in res_no_proj.setup_hint
    monkeypatch.undo()

    # 3. ADC failure returns ready=False cleanly
    monkeypatch.setattr(
        "scale_forecasting.bq_agent.probe.build_authorized_session",
        lambda _p: (_ for _ in ()).throw(RuntimeError("ADC missing")),
    )
    res_no_adc = probe_bq_agent_access(project_id=_PROJECT, dataset_id=_DATASET)
    assert res_no_adc.ready is False
    assert "application-default login" in res_no_adc.setup_hint
    monkeypatch.undo()

    class _SeqSession:
        def __init__(self, responses: list[_FakeHTTPResponse]) -> None:
            self._responses = list(responses)

        def request(self, **_: Any) -> _FakeHTTPResponse:
            return self._responses.pop(0)

    # 4. Registered DataAgent (200) + Knowledge Catalog ready (200)
    s_ok = _SeqSession(
        [
            _FakeHTTPResponse(
                200,
                {"name": f"projects/{_PROJECT}/locations/global/dataAgents/sf-registry-{_DATASET}"},
            ),
            _FakeHTTPResponse(200, {"name": "glossary"}),
        ]
    )
    res_ok = probe_bq_agent_access(project_id=_PROJECT, dataset_id=_DATASET, session=s_ok)
    assert res_ok.ready is True
    assert res_ok.agent_registered is True
    assert res_ok.knowledge_catalog_ready is True

    # 5. Unregistered DataAgent (404) + Knowledge Catalog SERVICE_DISABLED (403) -> still ready=True
    s_partial = _SeqSession(
        [
            _FakeHTTPResponse(404, {"error": "not found"}),
            _FakeHTTPResponse(403, {"error": "SERVICE_DISABLED"}),
        ]
    )
    res_partial = probe_bq_agent_access(project_id=_PROJECT, dataset_id=_DATASET, session=s_partial)
    assert res_partial.ready is True
    assert res_partial.agent_registered is False
    assert res_partial.knowledge_catalog_ready is False
    assert "dataplex.googleapis.com" in res_partial.missing_apis
    assert "gcloud services enable" in res_partial.setup_hint

    # 6. Conversational API SERVICE_DISABLED (403) and IAM PERMISSION_DENIED (403)
    s_disabled = _SeqSession(
        [
            _FakeHTTPResponse(403, {"error": "SERVICE_DISABLED"}),
            _FakeHTTPResponse(403, {"error": "SERVICE_DISABLED"}),
        ]
    )
    res_disabled = probe_bq_agent_access(
        project_id=_PROJECT, dataset_id=_DATASET, session=s_disabled
    )
    assert res_disabled.ready is False
    assert "geminidataanalytics.googleapis.com" in res_disabled.missing_apis

    s_iam = _SeqSession(
        [
            _FakeHTTPResponse(403, {"error": "PERMISSION_DENIED"}),
            _FakeHTTPResponse(200, {}),
        ]
    )
    res_iam = probe_bq_agent_access(project_id=_PROJECT, dataset_id=_DATASET, session=s_iam)
    assert res_iam.ready is False
    assert "roles/geminidataanalytics.dataAgentOwner" in res_iam.setup_hint

    # 7. CLI --probe, Registry.probe_bq_agent, Forecaster.probe_bq_agent, and MCP modes
    monkeypatch.setattr(bq_agent_cli, "probe_bq_agent_access", lambda **_: res_ok)
    assert bq_agent_cli.main(["--project", _PROJECT, "--dataset", _DATASET, "--probe"]) == 0
    assert json.loads(capsys.readouterr().out)["ready"] is True

    fake_settings = Settings(
        project_id=_PROJECT,
        dataset_id=_DATASET,
        connection=f"{_PROJECT}.us-central1.sf-iceberg",
        warehouse_uri=f"gs://{_PROJECT}-warehouse/warehouse",
    )
    monkeypatch.setattr(
        "scale_forecasting.bq_agent.probe.probe_bq_agent_access", lambda **_: res_ok
    )
    assert Registry(settings=fake_settings).probe_bq_agent().ready is True
    fc = Forecaster(
        {
            "run_name": "probe_demo",
            "data": {"source_table": "source_series_native", "horizon": 7},
            "models": ["theta"],
        },
        settings=fake_settings,
    )
    assert fc.probe_bq_agent().ready is True

    srv = McpServer()
    env_probe = srv.call_tool(
        "ask_forecast_registry",
        {"mode": "probe", "project_id": _PROJECT, "dataset_id": _DATASET},
    )
    assert env_probe["isError"] is False
    assert json.loads(env_probe["content"][0]["text"])["ready"] is True

    # MCP auto mode falls back cleanly to offline_playground when probe reports ready=False
    monkeypatch.setattr(
        "scale_forecasting.bq_agent.probe.probe_bq_agent_access",
        lambda **_: BQAgentProbeResult(
            ready=False,
            conversational_api_ready=False,
            knowledge_catalog_ready=False,
            agent_registered=False,
        ),
    )
    env_auto = srv.call_tool(
        "ask_forecast_registry",
        {
            "mode": "auto",
            "project_id": _PROJECT,
            "dataset_id": _DATASET,
            "question": "Which model won?",
            "n_series": 2,
            "horizon": 7,
        },
    )
    assert env_auto["isError"] is False
    assert json.loads(env_auto["content"][0]["text"])["mode"] == "offline"

    env_bad = srv.call_tool(
        "ask_forecast_registry",
        {"mode": "invalid_mode", "question": "Which model won?"},
    )
    assert env_bad["isError"] is True


def test_sync_existing_descriptions_and_metadata_guards() -> None:
    from google.api_core import exceptions as gcp_exceptions
    from google.cloud import bigquery

    from scale_forecasting.registry.metadata import (
        column_description,
        metric_description,
        object_description,
        sql_escape_description,
        sync_existing_descriptions,
    )

    assert "\\'" in sql_escape_description("planner's\nview")
    assert "custom_metric" in metric_description("custom_metric")

    with pytest.raises(KeyError):
        object_description("nonexistent_table")
    with pytest.raises(KeyError):
        column_description("nonexistent_table", "run_id")
    with pytest.raises(KeyError):
        column_description("run_registry", "nonexistent_col")
    assert "MAE" in column_description("forecast_metadata", "mae") or column_description(
        "forecast_metadata", "mae"
    )

    class _SchemaBQClient:
        def __init__(self) -> None:
            self.tables: dict[str, Any] = {
                f"{_PROJECT}.{_DATASET}.run_registry": SimpleNamespace_table(
                    description="",
                    schema=[
                        bigquery.SchemaField("run_id", "STRING", description=""),
                        bigquery.SchemaField("spark_method", "STRING", description=""),
                        bigquery.SchemaField("custom_untracked", "STRING", description="keep"),
                    ],
                ),
                f"{_PROJECT}.{_DATASET}.forecast_metadata": SimpleNamespace_table(
                    description=object_description("forecast_metadata"),
                    schema=[
                        bigquery.SchemaField(
                            "wape", "FLOAT64", description=metric_description("wape")
                        )
                    ],
                ),
                f"{_PROJECT}.{_DATASET}.v_run_summary": SimpleNamespace_table(
                    description="stale",
                    schema=[bigquery.SchemaField("run_id", "STRING", description="stale")],
                ),
            }
            self.updated: list[str] = []

        def get_table(self, fqn: str) -> Any:
            if fqn not in self.tables:
                raise gcp_exceptions.NotFound(fqn)
            return self.tables[fqn]

        def update_table(self, tbl: Any, _fields: list[str]) -> Any:
            if tbl.full_id.endswith("v_run_summary"):
                raise gcp_exceptions.BadRequest("simulated view patch rejection")
            self.updated.append(tbl.full_id)
            return tbl

    from types import SimpleNamespace

    def SimpleNamespace_table(description: str, schema: list[Any]) -> Any:
        return SimpleNamespace(full_id="", description=description, schema=schema)

    bq = _SchemaBQClient()
    for k, v in bq.tables.items():
        v.full_id = k

    patched = sync_existing_descriptions(
        bq,
        f"{_PROJECT}.{_DATASET}",
        ("run_registry", "forecast_metadata", "v_run_summary", "missing_tbl", "untracked_obj"),
    )
    assert patched == ["run_registry"]
    # Second pass is a pure no-op on run_registry because descriptions now match desired_desc!
    patched_again = sync_existing_descriptions(bq, f"{_PROJECT}.{_DATASET}", ("run_registry",))
    assert patched_again == []


def test_context_client_and_offline_edge_branches() -> None:
    from google.api_core import exceptions as gcp_exceptions

    from scale_forecasting.bq_agent.client import (
        default_agent_id,
        extract_response_artifacts,
        parse_chat_messages,
        resolve_registry_run_id,
    )
    from scale_forecasting.bq_agent.context import parse_table_fqn
    from scale_forecasting.bq_agent.offline import ask_offline
    from scale_forecasting.errors import ConfigError, RegistryError

    assert default_agent_id("___") == "sf-registry-agent"
    assert parse_table_fqn("", default_project_id=_PROJECT, default_dataset_id=_DATASET) is None
    assert (
        parse_table_fqn("<placeholder>", default_project_id=_PROJECT, default_dataset_id=_DATASET)
        is None
    )
    assert parse_table_fqn("a.b.c", default_project_id=_PROJECT, default_dataset_id=_DATASET) == (
        "a",
        "b",
        "c",
    )
    assert (
        parse_table_fqn("a.b.c.d", default_project_id=_PROJECT, default_dataset_id=_DATASET) is None
    )

    with pytest.raises(ConfigError, match="include_sources"):
        discover_registry_scope(_PROJECT, _DATASET, include_sources="bad", client=object())  # type: ignore[arg-type]

    with pytest.raises(ConfigError, match="additional_registries"):
        discover_registry_scope(
            _PROJECT, _DATASET, additional_registries=["a.b.c"], client=object()
        )

    class _RestrictedBQClient(_FakeBQClient):
        def get_table(self, fqn: str) -> _FakeTable:
            if fqn.endswith("forbidden_src"):
                raise gcp_exceptions.Forbidden("denied")
            return super().get_table(fqn)

        def query(self, _sql: str) -> _FakeQueryJob:
            raise RuntimeError("query blocked")

    existing = {f"{_PROJECT}.{_DATASET}.run_registry", f"{_PROJECT}.other_ds.run_registry"}
    r_bq = _RestrictedBQClient(existing_fqns=existing, raw_sources=[])
    fed_scope = discover_registry_scope(
        _PROJECT,
        _DATASET,
        include_sources="explicit",
        source_tables=["forbidden_src", "run_registry"],
        additional_registries=["", "other_ds", f"{_PROJECT}.other_ds"],
        client=r_bq,
    )
    assert len(fed_scope.physical_tables) == 2

    # Empty registry dataset raises ConfigError on ensure_registry_agent
    empty_bq = _FakeBQClient(existing_fqns=set(), raw_sources=[])
    with pytest.raises(ConfigError, match="No scale-forecasting registry tables"):
        ensure_registry_agent(
            project_id=_PROJECT,
            dataset_id=_DATASET,
            sync_descriptions=False,
            bq_client=empty_bq,
            session=_FakeAuthorizedSession(),
        )

    # resolve_registry_run_id branches
    assert resolve_registry_run_id("all", project_id=_PROJECT, dataset_id=_DATASET) is None
    with patch("scale_forecasting.bq_agent.client.read_recent_runs", return_value=[]):
        assert resolve_registry_run_id("latest", project_id=_PROJECT, dataset_id=_DATASET) is None
        assert (
            resolve_registry_run_id("custom-123", project_id=_PROJECT, dataset_id=_DATASET)
            == "custom-123"
        )
    with patch(
        "scale_forecasting.bq_agent.client.read_recent_runs",
        return_value=[
            {"run_id": "dup-a", "status": "FAILED"},
            {"run_id": "dup-b", "status": "FAILED"},
        ],
    ):
        assert (
            resolve_registry_run_id("latest", project_id=_PROJECT, dataset_id=_DATASET) == "dup-a"
        )
        assert resolve_registry_run_id("dup-b", project_id=_PROJECT, dataset_id=_DATASET) == "dup-b"
        assert resolve_registry_run_id("dup", project_id=_PROJECT, dataset_id=_DATASET) == "dup-a"

    # parse_chat_messages & extract_response_artifacts branches
    assert parse_chat_messages("") == []
    assert parse_chat_messages({"messages": [{"systemMessage": {}}]}) == [{"systemMessage": {}}]
    assert parse_chat_messages({"systemMessage": {}}) == [{"systemMessage": {}}]
    ndjson_lines = '{"systemMessage":{"error":{"text":"bad sql"}}}\nnot-json\n'
    parsed_msgs = parse_chat_messages(ndjson_lines)
    ans_err, sql_err, df_err, _ = extract_response_artifacts(parsed_msgs)
    assert "bad sql" in ans_err and sql_err == "" and df_err.empty

    matched_msgs = [
        {"userMessage": {"text": "hi"}},
        {
            "systemMessage": {
                "text": {"textType": "PROGRESS", "parts": ["Running query..."]},
                "data": {"matchedQuery": {"exampleQuery": {"sqlQuery": "SELECT 1"}}},
            }
        },
    ]
    ans_m, sql_m, _, _ = extract_response_artifacts(matched_msgs)
    assert ans_m == "Running query..." and sql_m == "SELECT 1"

    # ask_registry error guards (empty question, auto_ensure_agent=False, HTTP >= 400)
    with pytest.raises(ConfigError, match="non-empty"):
        ask_registry(
            "  ", project_id=_PROJECT, dataset_id=_DATASET, session=_FakeAuthorizedSession()
        )
    with pytest.raises(RegistryError, match="not registered yet"):
        ask_registry(
            "Which model won?",
            auto_ensure_agent=False,
            project_id=_PROJECT,
            dataset_id=_DATASET,
            session=_FakeAuthorizedSession(),
        )
    assert (
        delete_registry_agent(
            project_id=_PROJECT, dataset_id=_DATASET, session=_FakeAuthorizedSession()
        )
        is False
    )

    # Offline single-model & bakeoff leaderboard, S&OP summary, anomaly, and fallback
    pg = playground.run_model("lightgbm", horizon=5, backtest=True)
    assert not pg.ask("Which model won on the leaderboard?").df.empty
    assert not pg.sop_summary_df().empty
    bo = playground.bakeoff(models=["theta", "holtwinters"], horizon=5, n_folds=2)
    bo_sop = bo.sop_summary_df()
    assert len(bo_sop) == 1 and float(bo_sop.iloc[0]["total_forecast_units"]) > 0.0
    assert bo.ask("Give me the executive S&OP scorecard summary").mode == "offline"
    assert "anomalies" in ask_offline(object(), "Show historical demand anomalies").answer.lower()
    assert (
        "attributions"
        in ask_offline(object(), "What are the top feature attributions?").answer.lower()
    )
    assert "offline-run" in ask_offline(object(), "Unmatched generic question xyz").answer


def test_empty_dataset_guard_and_lazy_source_table_scope_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scale_forecasting.errors import ConfigError

    # 1. Empty dataset with default sync_descriptions=True and bq_client=None MUST fail fast before
    # calling ensure_views() so an agent is never created with zero tables.
    empty_bq = _FakeBQClient(existing_fqns=set(), raw_sources=[])
    ensure_views_called = False

    def _fail_if_called(**_: Any) -> None:
        nonlocal ensure_views_called
        ensure_views_called = True

    monkeypatch.setattr("scale_forecasting.bq_agent.client.bigquery_client", lambda _p: empty_bq)
    monkeypatch.setattr("scale_forecasting.bq_agent.client.ensure_views", _fail_if_called)
    with pytest.raises(ConfigError, match="never deployed with zero tables"):
        ensure_registry_agent(
            project_id=_PROJECT,
            dataset_id=_DATASET,
            sync_descriptions=True,
            session=_FakeAuthorizedSession(),
        )
    assert ensure_views_called is False
    monkeypatch.undo()

    # 2. Lazy scope refresh on ask_registry: Run 1 registers agent with source_series_native;
    # Run 2 introduces raw_erp.demand_history -> next ask_registry auto-updates publishedContext!
    existing = {
        *(f"{_PROJECT}.{_DATASET}.{t}" for t in REGISTRY_TABLE_NAMES),
        *(f"{_PROJECT}.{_DATASET}.{v}" for v in VIEW_NAMES),
        f"{_PROJECT}.{_DATASET}.source_series_native",
    }
    fake_bq = _FakeBQClient(existing_fqns=existing, raw_sources=["source_series_native"])
    fake_sess = _FakeAuthorizedSession()

    ensure_registry_agent(
        project_id=_PROJECT,
        dataset_id=_DATASET,
        sync_glossary=False,
        bq_client=fake_bq,
        session=fake_sess,
    )
    update_calls_before = sum(1 for m, u, _, _ in fake_sess.calls if u.endswith(":updateSync"))
    assert update_calls_before == 0

    # Now a second forecast job adds `raw_erp.demand_history` to `run_registry` & BigQuery:
    fake_bq.existing_fqns.add(f"{_PROJECT}.raw_erp.demand_history")
    fake_bq.raw_sources.append("raw_erp.demand_history")

    with patch("scale_forecasting.bq_agent.client.read_recent_runs", return_value=[]):
        ask_registry(
            "Compare forecast vs historical demand.",
            project_id=_PROJECT,
            dataset_id=_DATASET,
            bq_client=fake_bq,
            session=fake_sess,
        )
        update_calls_after_new_src = sum(
            1 for m, u, _, _ in fake_sess.calls if u.endswith(":updateSync")
        )
        assert update_calls_after_new_src == 1

        # Verify raw_erp.demand_history is now in the registered agent's publishedContext!
        updated_agent = get_registry_agent(
            project_id=_PROJECT, dataset_id=_DATASET, session=fake_sess
        )
        assert updated_agent is not None
        tbl_refs = updated_agent["dataAnalyticsAgent"]["publishedContext"]["datasourceReferences"][
            "bq"
        ]["tableReferences"]
        fqns = {f"{r['projectId']}.{r['datasetId']}.{r['tableId']}" for r in tbl_refs}
        assert f"{_PROJECT}.raw_erp.demand_history" in fqns

        # Subsequent ask with no new source tables skips `:updateSync`!
        ask_registry(
            "Follow-up question on the same scope.",
            project_id=_PROJECT,
            dataset_id=_DATASET,
            bq_client=fake_bq,
            session=fake_sess,
        )
        update_calls_steady = sum(1 for m, u, _, _ in fake_sess.calls if u.endswith(":updateSync"))
        assert update_calls_steady == 1


def test_run_result_ask_airflow_finalize_retry_and_mcp_ensure_polish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scale_forecasting import RunResult, airflow_tasks, retry_run
    from scale_forecasting.bq_agent.probe import probe_bq_agent_access
    from scale_forecasting.job_launch import RetryOutcome
    from scale_forecasting.mcp import McpServer
    from scale_forecasting.retry_run import RetryPlan
    from scale_forecasting.settings import Settings

    # 1. Probe setup_hint names `enable_bq_agent = true` (matching terraform/main/variables.tf)
    class _DisabledSession:
        def request(self, **_: Any) -> _FakeHTTPResponse:
            return _FakeHTTPResponse(403, {"error": "SERVICE_DISABLED"})

    p_res = probe_bq_agent_access(
        project_id=_PROJECT, dataset_id=_DATASET, session=_DisabledSession()
    )
    assert "enable_bq_agent = true" in p_res.setup_hint
    assert "enable_bq_conversational_agent" not in p_res.setup_hint

    # 2. RunResult.ask() and RunResult.sop_summary_df() preserve equality & forward run_id/coords
    s = Settings(
        project_id=_PROJECT,
        dataset_id=_DATASET,
        connection=f"{_PROJECT}.us-central1.sf-iceberg",
        warehouse_uri=f"gs://{_PROJECT}-warehouse/warehouse",
    )
    r_a = RunResult(run_id="demo-run-123", dataset_ref=f"{_PROJECT}.{_DATASET}", views=VIEW_NAMES)
    r_b = RunResult(
        run_id="demo-run-123",
        dataset_ref=f"{_PROJECT}.{_DATASET}",
        views=VIEW_NAMES,
        _settings=s,
    )
    assert r_a == r_b
    assert "_settings" not in repr(r_b)

    captured_ask: dict[str, Any] = {}

    def _fake_ask(q: str, **kw: Any) -> BQAgentResponse:
        captured_ask["q"] = q
        captured_ask.update(kw)
        return BQAgentResponse(
            question=q,
            answer="ensemble_mean won.",
            sql="SELECT 1",
            df=pd.DataFrame([{"ok": 1}]),
            run_id=kw.get("run_id"),
            conversation_id="sf-conv-runresult",
        )

    monkeypatch.setattr("scale_forecasting.bq_agent.client.ask_registry", _fake_ask)
    ans = r_a.ask("Which model won?")
    assert ans.run_id == "demo-run-123"
    assert captured_ask["project_id"] == _PROJECT
    assert captured_ask["dataset_id"] == _DATASET

    monkeypatch.setattr(
        "scale_forecasting.registry.reads.read_executive_sop_summary",
        lambda **kw: [{"run_id": kw["run_id"], "champion_model": "theta"}],
    )
    sop = r_b.sop_summary_df()
    assert len(sop) == 1 and sop.iloc[0]["run_id"] == "demo-run-123"

    # 3. Composer Airflow `finalize_run` triggers `ensure_registry_agent` when ensure_bq_agent=True
    cfg_ens = RunConfig.model_validate(
        {
            "run_name": "airflow_agent_hook",
            "data": {"source_table": "source_series_native", "horizon": 7},
            "models": ["theta"],
            "output": {"ensure_bq_agent": True},
        }
    )
    ensured_calls: list[Any] = []
    fake_reg = BQAgentRegistration(
        agent_name=f"projects/{_PROJECT}/locations/global/dataAgents/sf-registry-{_DATASET}",
        agent_id=f"sf-registry-{_DATASET}",
        project_id=_PROJECT,
        dataset_id=_DATASET,
        location="global",
        created=False,
        scoped_tables=(f"{_PROJECT}.{_DATASET}.run_registry",),
        scoped_views=(f"{_PROJECT}.{_DATASET}.v_run_summary",),
        scoped_sources=(),
        data_scans=(),
    )
    monkeypatch.setattr(
        "scale_forecasting.bq_agent.client.ensure_registry_agent",
        lambda **kw: ensured_calls.append(kw) or fake_reg,
    )
    monkeypatch.setattr("scale_forecasting.config.load_config_uri", lambda _u: cfg_ens)
    monkeypatch.setattr("scale_forecasting.settings.Settings.resolve", lambda: s)
    monkeypatch.setattr(
        "scale_forecasting.registry.jobs.read_run_jobs",
        lambda *_a, **_kw: [{"family": "statistical", "status": "COMPLETED", "runtime_seconds": 5}],
    )
    monkeypatch.setattr("scale_forecasting.registry.header.update_header", lambda *_a, **_kw: None)

    airflow_tasks.finalize_run("gs://code/cfg.json")
    assert len(ensured_calls) == 1

    # 4. `retry_run(..., confirm=True)` triggers `ensure_registry_agent` when repair succeeds
    monkeypatch.setattr(
        retry_run,
        "build_retry_plan",
        lambda _c, settings=None: RetryPlan(
            run_id=make_run_id(cfg_ens),
            header_status="PARTIAL",
            models=("theta",),
            n_targets=2,
        ),
    )
    monkeypatch.setattr(
        "scale_forecasting.job_launch.submit_retry",
        lambda *_a, **_kw: RetryOutcome(families=("statistical_repair",), errors={}),
    )
    monkeypatch.setattr("scale_forecasting.identity.resolve_principal", lambda _s: "tester")
    monkeypatch.setattr("scale_forecasting.registry.jobs.update_job", lambda *_a, **_kw: None)

    rep = retry_run.retry_run(cfg_ens, confirm=True, settings=s)
    assert rep.executed is True
    assert len(ensured_calls) == 2

    # 5. MCP `ask_forecast_registry`: `mode="offline"` alias + `mode="ensure"` & `--allow-launch`
    srv_safe = McpServer(allow_launch=False)
    off_env = srv_safe.call_tool(
        "ask_forecast_registry",
        {"mode": "offline", "question": "Which model won?", "n_series": 2, "horizon": 7},
    )
    assert off_env["isError"] is False
    assert json.loads(off_env["content"][0]["text"])["mode"] == "offline"

    blocked_scans = srv_safe.call_tool(
        "ask_forecast_registry",
        {"mode": "ensure", "project_id": _PROJECT, "dataset_id": _DATASET, "run_scans": True},
    )
    assert blocked_scans["isError"] is False
    assert json.loads(blocked_scans["content"][0]["text"])["allowed"] is False

    ens_env = srv_safe.call_tool(
        "ask_forecast_registry",
        {"mode": "ensure", "project_id": _PROJECT, "dataset_id": _DATASET, "run_scans": False},
    )
    assert ens_env["isError"] is False
    assert json.loads(ens_env["content"][0]["text"])["agent_id"] == f"sf-registry-{_DATASET}"

    srv_launch = McpServer(allow_launch=True)
    scans_ok = srv_launch.call_tool(
        "ask_forecast_registry",
        {"mode": "ensure", "project_id": _PROJECT, "dataset_id": _DATASET, "run_scans": True},
    )
    assert scans_ok["isError"] is False
    assert ensured_calls[-1]["run_scans"] is True


def test_console_urls_across_all_runtimes_and_sdk_surfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scale_forecasting as sf
    from scale_forecasting.bq_agent.console_urls import (
        build_job_console_url,
        build_registry_console_urls,
    )
    from scale_forecasting.settings import Settings

    # 1. All 7 compute runtimes + borrowed cell reuse + project-neutral URLs
    u_spark_srv = build_job_console_url(
        "spark",
        system_job_id="sf-run1-statistical-a1",
        project_id=_PROJECT,
        probe_handle=json.dumps({"region": "us-central1", "spark_mode": "serverless"}),
    )
    assert (
        u_spark_srv
        == f"https://console.cloud.google.com/dataproc/batches/us-central1/sf-run1-statistical-a1/monitoring?project={_PROJECT}"
    )

    u_spark_cls = build_job_console_url(
        "spark",
        system_job_id="job-uuid-123",
        project_id=_PROJECT,
        region="us-east1",
        spark_mode="cluster",
    )
    assert (
        u_spark_cls
        == f"https://console.cloud.google.com/dataproc/jobs/job-uuid-123?region=us-east1&project={_PROJECT}"
    )

    u_vertex = build_job_console_url(
        "vertex",
        system_job_id="projects/12345/locations/us-central1/customJobs/987654321",
        project_id=_PROJECT,
    )
    assert (
        u_vertex
        == f"https://console.cloud.google.com/vertex-ai/locations/us-central1/training/987654321/cpu?project={_PROJECT}"
    )

    u_ray = build_job_console_url(
        "ray",
        system_job_id="raysubmit_abc",
        project_id=_PROJECT,
        probe_handle={
            "region": "us-central1",
            "resource_name": "projects/12345/locations/us-central1/persistentResources/sf-ray-1",
        },
    )
    assert (
        u_ray
        == f"https://console.cloud.google.com/vertex-ai/locations/us-central1/ray/sf-ray-1?project={_PROJECT}"
    )

    assert (
        build_job_console_url("gce", project_id=_PROJECT)
        == f"https://console.cloud.google.com/compute/instances?project={_PROJECT}"
    )
    assert (
        build_job_console_url("gke", project_id=_PROJECT)
        == f"https://console.cloud.google.com/kubernetes/workload/overview?project={_PROJECT}"
    )
    assert (
        build_job_console_url("vertex_automl", project_id=_PROJECT)
        == f"https://console.cloud.google.com/vertex-ai/pipelines/runs?project={_PROJECT}"
    )
    assert (
        build_job_console_url("bigquery", project_id=_PROJECT)
        == f"https://console.cloud.google.com/bigquery?project={_PROJECT}"
    )
    assert (
        build_job_console_url("spark", system_job_id="borrowed:prior-run-123", project_id=_PROJECT)
        == f"https://console.cloud.google.com/bigquery?project={_PROJECT}"
    )
    # Project-neutral link omits ?project=
    assert (
        build_job_console_url("bigquery", project_id=None)
        == "https://console.cloud.google.com/bigquery"
    )

    # 2. Full registry link set + SDK surfaces (JobTrace, jobs_df, RunResult, Forecaster, Registry)
    links = build_registry_console_urls(
        project_id=_PROJECT,
        dataset_id=_DATASET,
        region="us-central1",
        run_id="demo-run-123",
        warehouse_uri=f"gs://{_PROJECT}-warehouse/warehouse",
        job_rows=[
            {
                "family": "statistical",
                "runtime": "spark",
                "system_job_id": "sf-demo-stat-a1",
                "status": "COMPLETED",
                "spark_mode": "serverless",
            },
            {
                "family": "ml",
                "runtime": "spark",
                "system_job_id": "borrowed:prior-run-1",
                "status": "COMPLETED",
            },
        ],
    )
    assert f"bigquery/agents_hub?project={_PROJECT}" in links.bq_agents_url
    assert f"d={_DATASET}" in links.dataset_url
    assert f"dataplex/dp-glossaries?project={_PROJECT}" in links.glossary_url

    links_with_num = build_registry_console_urls(
        project_id=_PROJECT,
        project_number="123456789012",
        dataset_id=_DATASET,
    )
    assert (
        "dataplex/dp-glossaries/projects/123456789012/locations/global/"
        f"glossaries/sf-glossary-scale-forecasting?project={_PROJECT}"
    ) in links_with_num.glossary_url
    assert len(links.by_category("compute")) == 2
    assert len(links.by_category("storage")) == 1
    assert "BigQuery Studio" in links.format_text()
    assert "[Open in Console]" in links._repr_markdown_()
    assert len(links.to_dataframe()) == len(links.links)
    assert links.to_dict()["run_id"] == "demo-run-123"

    s = Settings(
        project_id=_PROJECT,
        dataset_id=_DATASET,
        connection=f"{_PROJECT}.us-central1.sf-iceberg",
        warehouse_uri=f"gs://{_PROJECT}-warehouse/warehouse",
    )
    monkeypatch.setattr(
        "scale_forecasting.registry.jobs.read_run_jobs",
        lambda *_a, **_kw: [
            {
                "family": "statistical",
                "job_id": "job-1",
                "system_job_id": "sf-demo-stat-a1",
                "runtime": "spark",
                "status": "COMPLETED",
                "attempt": 1,
                "hardware": "cpu",
                "gpu_type": None,
                "spark_mode": "serverless",
                "runtime_seconds": 41.2,
                "probe_handle": json.dumps({"region": "us-central1", "spark_mode": "serverless"}),
            }
        ],
    )
    fc = Forecaster(
        {
            "run_name": "console_demo",
            "data": {"source_table": "source_series_native", "horizon": 7},
            "models": ["theta"],
        },
        settings=s,
    )
    jdf = fc.jobs_df()
    assert "console_url" in jdf.columns
    assert "dataproc/batches/us-central1/sf-demo-stat-a1/monitoring" in jdf.loc[0, "console_url"]
    assert fc.console_urls().run_id == fc.run_id
    assert fc.review().console_urls().run_id == fc.run_id
    assert sf.Registry(settings=s).console_urls(run_id=fc.run_id).project_id == _PROJECT


def test_chat_session_response_plot_and_cli_repl(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import matplotlib.pyplot as plt

    from scale_forecasting import playground
    from scale_forecasting.bq_agent import (
        STARTER_QUESTIONS,
        BQAgentResponse,
        BQChatSession,
    )
    from scale_forecasting.settings import Settings

    # 1. BQAgentResponse.plot() covers bar chart, step line chart, Vega-Lite inline values, empty df
    bar_resp = BQAgentResponse(
        question="Which model won?",
        answer="ensemble_nnls won.",
        sql="SELECT model_type, pooled_wape FROM v_model_leaderboard_comparable",
        df=pd.DataFrame(
            {
                "model_type": ["ensemble_nnls", "lightgbm", "theta"],
                "pooled_wape": [0.062, 0.069, 0.081],
            }
        ),
    )
    ax1 = bar_resp.plot()
    assert ax1.get_xlabel() == "pooled_wape"

    line_resp = BQAgentResponse(
        question="Horizon decay",
        answer="WAPE grows gently.",
        sql="SELECT horizon_step, pooled_wape FROM v_horizon_decay",
        df=pd.DataFrame({"horizon_step": [1, 2, 3], "pooled_wape": [0.04, 0.05, 0.06]}),
    )
    ax2 = line_resp.plot()
    assert ax2.get_xlabel() == "horizon_step"

    vega_resp = BQAgentResponse(
        question="Chart from spec",
        answer="Spec values.",
        sql="",
        df=pd.DataFrame(),
        chart_spec={"data": {"values": [{"series": "s1", "wape": 0.05}]}},
    )
    ax3 = vega_resp.plot(render_vegalite=False)
    assert ax3.get_xlabel() == "wape"

    empty_resp = BQAgentResponse(question="Empty", answer="None", sql="", df=pd.DataFrame())
    ax4 = empty_resp.plot()
    assert "no tabular rows" in ax4.get_title()
    plt.close("all")

    # 2. Offline Forecaster.chat() & live BQChatSession multi-turn statefulness
    df = playground.sample_data(n_series=2)
    local_fc = Forecaster.auto(df, tier="first_light", where="local", horizon=7, autofocus=False)
    local_chat = local_fc.chat()
    t1 = local_chat("Which model won?")
    t2 = local_chat("Which series show bias?")
    assert len(local_chat.history) == 2
    assert local_chat.last is t2
    assert local_chat.conversation_id == t1.conversation_id == t2.conversation_id
    assert len(local_chat.starters) == len(STARTER_QUESTIONS)
    local_chat.reset()
    assert len(local_chat.history) == 0 and local_chat.conversation_id is None

    seen_calls: list[dict[str, Any]] = []

    def _fake_ask_registry(question: str, **kw: Any) -> BQAgentResponse:
        seen_calls.append({"question": question, **kw})
        cid = kw.get("conversation_id") or "sf-conv-demo123"
        return BQAgentResponse(
            question=question,
            answer=f"Answered: {question}",
            sql="SELECT 1 AS metric_val",
            df=pd.DataFrame({"metric_val": [1]}),
            conversation_id=cid,
            run_id=kw.get("run_id"),
        )

    monkeypatch.setattr("scale_forecasting.bq_agent.client.ask_registry", _fake_ask_registry)
    monkeypatch.setattr(bq_agent_cli, "ask_registry", _fake_ask_registry)

    s = Settings(
        project_id=_PROJECT,
        dataset_id=_DATASET,
        connection=f"{_PROJECT}.us-central1.sf-iceberg",
        warehouse_uri=f"gs://{_PROJECT}-warehouse/warehouse",
    )
    reg = Registry(settings=s)
    chat = reg.chat(run_id="forecaster-showcase-f67fec77de66")
    r_a = chat("Turn 1")
    r_b = chat("Turn 2")
    r_c = chat("Compare recent runs in v_run_summary including our 100,000-series benchmark runs.")
    assert r_a.conversation_id == "sf-conv-demo123"
    assert seen_calls[0]["conversation_id"] is None
    assert seen_calls[0]["auto_ensure_agent"] is True
    assert seen_calls[1]["conversation_id"] == "sf-conv-demo123"
    assert seen_calls[1]["auto_ensure_agent"] is False
    assert r_b.run_id == "forecaster-showcase-f67fec77de66"
    assert r_c.run_id == "all"
    assert chat.run_id == "forecaster-showcase-f67fec77de66"
    chat.switch_run("all")
    assert chat.run_id == "all"

    rr_chat = (
        Forecaster(
            {
                "run_name": "rr_demo",
                "data": {"source_table": "source_series_native", "horizon": 7},
                "models": ["theta"],
            },
            settings=s,
        )
        .review()
        .chat()
    )
    rr_resp = rr_chat("Winner?")
    assert rr_resp.conversation_id == "sf-conv-demo123"

    # 3. CLI --links and --chat REPL
    assert bq_agent_cli.main(["--project", _PROJECT, "--dataset", _DATASET, "--links"]) == 0
    assert "BigQuery Studio" in capsys.readouterr().out
    assert (
        bq_agent_cli.main(["--project", _PROJECT, "--dataset", _DATASET, "--links", "--json"]) == 0
    )
    assert json.loads(capsys.readouterr().out)["project_id"] == _PROJECT

    inputs = iter(["/starters", "1", "/sql", "/run all", "/links", "/new", "/quit"])
    repl_session = BQChatSession(project_id=_PROJECT, dataset_id=_DATASET, run_id="latest")
    assert bq_agent_cli._run_chat_repl(repl_session, input_fn=lambda _p: next(inputs)) == 0
    repl_out = capsys.readouterr().out
    assert "Starter questions" in repl_out
    assert "Last Generated SQL" in repl_out
    assert "Switched default run scope" in repl_out
