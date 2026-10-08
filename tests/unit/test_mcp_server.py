"""Unit and protocol tests for the built-in stdio MCP server (`scale_forecasting.mcp`).

Exercises JSON-RPC 2.0 protocol framing (newline-delimited and ``Content-Length`` headers),
all 7 ``forecast://*`` resources, all 9 tools, and the ``--allow-launch`` safety lock.
"""

from __future__ import annotations

import dataclasses
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scale_forecasting import mcp

REPO_ROOT = Path(__file__).resolve().parents[2]
MIXED_DEMO = REPO_ROOT / "configs" / "mixed_demo.json"


def _call_tool(
    server: mcp.McpServer,
    name: str,
    arguments: dict[str, Any] | None = None,
) -> tuple[bool, Any]:
    resp = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments or {}},
        }
    )
    assert resp is not None and "result" in resp, resp
    result = resp["result"]
    is_error = bool(result.get("isError", False))
    text = result["content"][0]["text"]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = text
    return is_error, parsed


def test_mcp_initialize_ping_notifications_and_errors() -> None:
    server = mcp.McpServer(allow_launch=False, repo_root=REPO_ROOT)

    init_resp = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-03-26"},
        }
    )
    assert init_resp is not None
    assert init_resp["result"]["protocolVersion"] == "2025-03-26"
    assert init_resp["result"]["serverInfo"]["name"] == "scale-forecasting"

    # Notifications (no id) return None.
    assert server.handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None

    ping_resp = server.handle_message({"jsonrpc": "2.0", "id": 2, "method": "ping"})
    assert ping_resp == {"jsonrpc": "2.0", "id": 2, "result": {}}

    prompts_resp = server.handle_message({"jsonrpc": "2.0", "id": 3, "method": "prompts/list"})
    assert prompts_resp == {"jsonrpc": "2.0", "id": 3, "result": {"prompts": []}}

    unknown_resp = server.handle_message({"jsonrpc": "2.0", "id": 4, "method": "no/such/method"})
    assert unknown_resp is not None and unknown_resp["error"]["code"] == -32601


def test_mcp_resources_list_and_read_all_uris() -> None:
    server = mcp.McpServer(allow_launch=False, repo_root=REPO_ROOT)
    list_resp = server.handle_message({"jsonrpc": "2.0", "id": 1, "method": "resources/list"})
    assert list_resp is not None
    resources = list_resp["result"]["resources"]
    uris = [r["uri"] for r in resources]
    assert uris == [
        "forecast://environment",
        "forecast://schema/run-config",
        "forecast://catalog/models",
        "forecast://catalog/metrics",
        "forecast://catalog/runtimes",
        "forecast://catalog/configs",
        "forecast://catalog/views",
    ]

    for uri in uris:
        read_resp = server.handle_message(
            {"jsonrpc": "2.0", "id": 2, "method": "resources/read", "params": {"uri": uri}}
        )
        assert read_resp is not None and "result" in read_resp, uri
        contents = read_resp["result"]["contents"]
        assert len(contents) == 1
        assert contents[0]["uri"] == uri
        payload = json.loads(contents[0]["text"])
        assert payload

    bad_uri = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "resources/read",
            "params": {"uri": "forecast://unknown"},
        }
    )
    assert bad_uri is not None and bad_uri["error"]["code"] == -32602


def test_mcp_tools_list_and_offline_tools(tmp_path: Path) -> None:
    server = mcp.McpServer(allow_launch=False, repo_root=REPO_ROOT)
    tools_resp = server.handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert tools_resp is not None
    tool_names = [t["name"] for t in tools_resp["result"]["tools"]]
    assert tool_names == [
        "probe_environment",
        "list_catalog",
        "inspect_config_schema",
        "validate_and_dry_run",
        "run_playground",
        "plan_execution",
        "inspect_registry",
        "review_run",
        "launch_or_repair_run",
    ]

    # 1. probe_environment
    is_err, env_payload = _call_tool(server, "probe_environment")
    assert not is_err
    assert env_payload["models"]["total"] == 34

    # 2. list_catalog across categories and filters
    for cat in ("all", "models", "metrics", "runtimes", "gpus", "configs", "views", "extras"):
        is_err, cat_payload = _call_tool(server, "list_catalog", {"category": cat})
        assert not is_err, cat_payload
        assert cat_payload
    is_err, dl_models = _call_tool(
        server,
        "list_catalog",
        {"category": "models", "family": "deep_learning", "available_only": True},
    )
    assert not is_err
    assert len(dl_models["models"]) <= 5

    # 3. inspect_config_schema (overview, property, class name, unknown)
    is_err, schema_overview = _call_tool(server, "inspect_config_schema")
    assert not is_err
    assert "top_level_properties" in schema_overview

    is_err, compute_sec = _call_tool(server, "inspect_config_schema", {"section": "compute"})
    assert not is_err
    assert compute_sec["section"] == "compute"

    is_err, by_cls = _call_tool(server, "inspect_config_schema", {"section": "BacktestConfig"})
    assert not is_err
    assert by_cls["section"] == "BacktestConfig"
    assert "definition" in by_cls

    is_err, err_msg = _call_tool(server, "inspect_config_schema", {"section": "nonexistent"})
    assert is_err and "Unknown schema section" in str(err_msg)

    # 4. validate_and_dry_run (path, inline JSON string, inline dict)
    is_err, dry_path = _call_tool(
        server,
        "validate_and_dry_run",
        {"config": "configs/mixed_demo.json", "n_series": 50, "ignore_unavailable_models": True},
    )
    assert not is_err
    assert dry_path["valid"] is True
    assert dry_path["run_id"].startswith("mixed-demo-")
    assert len(dry_path["dag_nodes"]) >= 1

    inline_cfg = {
        "run_name": "mcp_unit_test",
        "models": ["holtwinters"],
        "data": {"source_table": "source_series", "horizon": 7},
    }
    is_err, dry_dict = _call_tool(
        server,
        "validate_and_dry_run",
        {"config": inline_cfg, "ignore_unavailable_models": True},
    )
    assert not is_err and dry_dict["valid"] is True

    is_err, dry_json = _call_tool(
        server,
        "validate_and_dry_run",
        {"config": json.dumps(inline_cfg), "ignore_unavailable_models": True},
    )
    assert not is_err and dry_json["run_id"] == dry_dict["run_id"]

    # 5. run_playground (backtest=True and backtest=False, plus non-Python model rejection)
    is_err, pg_bt = _call_tool(
        server,
        "run_playground",
        {"model": "holtwinters", "horizon": 7, "n_series": 2, "backtest": True},
    )
    assert not is_err
    assert pg_bt["model"] == "holtwinters"
    assert len(pg_bt["sample_predictions"]) == 5
    assert "wape" in pg_bt["metrics"]

    is_err, pg_no_bt = _call_tool(
        server,
        "run_playground",
        {"model": "naive_mean", "horizon": 5, "n_series": 2, "backtest": False},
    )
    assert not is_err
    assert pg_no_bt["metrics"] == {}

    is_err, pg_bad = _call_tool(server, "run_playground", {"model": "arima_plus"})
    assert is_err and "cannot run in the offline Python playground" in str(pg_bad)

    # 6. plan_execution (dry_run, emit_airflow with file output, and stage_only safety lock)
    is_err, plan_dry = _call_tool(
        server,
        "plan_execution",
        {"config": inline_cfg, "mode": "dry_run"},
    )
    assert not is_err and plan_dry["mode"] == "dry_run"

    dag_out = tmp_path / "emitted_dag.py"
    is_err, plan_af = _call_tool(
        server,
        "plan_execution",
        {
            "config": inline_cfg,
            "mode": "emit_airflow",
            "with_retry": True,
            "emit_out": str(dag_out),
        },
    )
    assert not is_err
    assert dag_out.is_file()
    assert "statistical >> retry" in plan_af["dag_source"]

    is_err, stage_locked = _call_tool(
        server,
        "plan_execution",
        {"config": inline_cfg, "mode": "stage_only"},
    )
    assert not is_err
    assert stage_locked["allowed"] is False
    assert "--allow-launch" in stage_locked["message"]

    # 7. launch_or_repair_run safety lock when allow_launch=False
    is_err, launch_locked = _call_tool(
        server,
        "launch_or_repair_run",
        {"action": "run", "config": inline_cfg},
    )
    assert not is_err
    assert launch_locked["allowed"] is False
    assert "--allow-launch" in launch_locked["message"]


def test_mcp_cloud_tools_with_mocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SF_PROJECT_ID", "example-project")
    monkeypatch.setenv("SF_CONNECTION", "example-project.us-central1.biglake")
    monkeypatch.setenv("SF_WAREHOUSE_URI", "gs://example-bucket/warehouse")

    server = mcp.McpServer(allow_launch=True, repo_root=REPO_ROOT)

    @dataclasses.dataclass
    class _FakeTableStat:
        name: str = "run_registry"
        exists: bool = True
        rows: int = 10

    @dataclasses.dataclass
    class _FakeDoc:
        registry: str = "example-project.scale_forecasting"
        artifact_root: str = "gs://example-bucket/warehouse/registry"
        healthy: bool = True
        missing_tables: tuple[str, ...] = ()
        tables: tuple[_FakeTableStat, ...] = (_FakeTableStat(),)
        views: tuple[str, ...] = ("v_model_leaderboard",)
        live_runs: tuple[tuple[str, str], ...] = ()
        orphans: tuple[Any, ...] = ()

    @dataclasses.dataclass
    class _FakeRetryPlan:
        submittable: bool = True
        n_cells: int = 2

    @dataclasses.dataclass
    class _FakeRetryReport:
        run_id: str = "demo-123456789abc"
        executed: bool = True
        plan: _FakeRetryPlan = dataclasses.field(default_factory=_FakeRetryPlan)
        outcome: _FakeTableStat | None = None

    @dataclasses.dataclass
    class _FakeReview:
        run_id: str = "demo-123456789abc"
        status: str = "COMPLETED"
        n_series: int = 10
        decision_metric: str = "wape"
        best_overall: _FakeTableStat | None = None
        best_per_family: dict[str, _FakeTableStat] = dataclasses.field(default_factory=dict)
        ensemble_lift: tuple[_FakeTableStat, ...] = ()
        models: tuple[_FakeTableStat, ...] = ()

    import scale_forecasting.launch_plan as lp_mod
    import scale_forecasting.main as main_mod
    import scale_forecasting.probes.cancel as cancel_mod
    import scale_forecasting.probes.reconcile as rec_mod
    import scale_forecasting.probes.settle as settle_mod
    import scale_forecasting.quota as quota_mod
    import scale_forecasting.retry_run as retry_mod
    import scale_forecasting.review as rev_mod
    from scale_forecasting.registry import ops, reads

    monkeypatch.setattr(ops, "doctor", lambda **kw: _FakeDoc())
    monkeypatch.setattr(ops, "format_doctor", lambda doc: "Doctor OK")
    monkeypatch.setattr(ops, "close_runs", lambda **kw: _FakeTableStat())
    monkeypatch.setattr(
        reads,
        "read_recent_runs",
        lambda **kw: [{"run_id": "demo-123456789abc", "status": "COMPLETED"}],
    )
    monkeypatch.setattr(rec_mod, "probe_run", lambda run_id, **kw: _FakeTableStat())
    monkeypatch.setattr(
        retry_mod,
        "config_for_run",
        lambda run_id, **kw: mcp._parse_run_config_input(
            {
                "run_name": "t",
                "models": ["holtwinters"],
                "data": {"source_table": "source_series", "horizon": 7},
            },
            REPO_ROOT,
        ),
    )
    monkeypatch.setattr(retry_mod, "retry_run", lambda cfg, **kw: _FakeRetryReport())
    monkeypatch.setattr(retry_mod, "format_retry_plan", lambda plan: "Retry plan OK")
    monkeypatch.setattr(rev_mod, "review_run", lambda run_id, **kw: _FakeReview())
    monkeypatch.setattr(rev_mod, "calibration_report", lambda run_id, **kw: _FakeTableStat())
    monkeypatch.setattr(lp_mod, "feasibility_report", lambda cfg: ["Feasible"])
    monkeypatch.setattr(quota_mod, "report_for_run", lambda cfg: ["Quota OK"])

    @dataclasses.dataclass
    class _FakeStaged:
        run_id: str = "demo-123456789abc"
        config_uri: str = "gs://example-bucket/warehouse/staging/demo/config.json"
        commands: dict[str, _FakeTableStat] | None = None

    monkeypatch.setattr(lp_mod, "stage_run", lambda cfg, **kw: _FakeStaged())
    monkeypatch.setattr(main_mod, "run", lambda cfg, **kw: "demo-123456789abc")
    monkeypatch.setattr(settle_mod, "settle_run", lambda run_id, **kw: _FakeTableStat())
    monkeypatch.setattr(cancel_mod, "cancel_run", lambda run_id, **kw: _FakeTableStat())

    test_cfg = {
        "run_name": "t",
        "models": ["holtwinters"],
        "data": {"source_table": "source_series", "horizon": 7},
    }

    for mode in ("feasibility", "quota", "stage_only"):
        is_err, m_res = _call_tool(server, "plan_execution", {"config": test_cfg, "mode": mode})
        assert not is_err, m_res
        assert m_res["mode"] == mode

    for action in ("doctor", "recent_runs", "probe_run", "retry_preview"):
        is_err, i_res = _call_tool(
            server,
            "inspect_registry",
            {"action": action, "run_id": "demo-123456789abc"},
        )
        assert not is_err, i_res
        assert i_res["action"] == action

    is_err, rev_res = _call_tool(
        server,
        "review_run",
        {"run_id": "demo-123456789abc", "include_calibration": True, "include_leaderboard": True},
    )
    assert not is_err, rev_res
    assert rev_res["run_id"] == "demo-123456789abc"

    for action in ("run", "retry", "settle", "cancel", "close_runs"):
        is_err, l_res = _call_tool(
            server,
            "launch_or_repair_run",
            {"action": action, "config": test_cfg, "run_id": "demo-123456789abc"},
        )
        assert not is_err, l_res
        assert l_res["allowed"] is True
        assert l_res["action"] == action


def test_mcp_stdio_framing_and_cli_entrypoints() -> None:
    server = mcp.McpServer(allow_launch=False, repo_root=REPO_ROOT)

    # Test both Content-Length framing and newline-delimited JSON in a single stream.
    msg1 = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}).encode("utf-8")
    framed = f"Content-Length: {len(msg1)}\r\n\r\n".encode("ascii") + msg1
    line_msg = b'{"jsonrpc": "2.0", "id": 2, "method": "ping"}\n{invalid json}\n'

    in_stream = io.BytesIO(framed + line_msg)
    out_stream = io.BytesIO()
    server.serve_stdio(stdin=in_stream, stdout=out_stream)

    output_bytes = out_stream.getvalue()
    assert b"Content-Length:" in output_bytes
    assert b'"id": 2' in output_bytes
    assert b'"code": -32700' in output_bytes

    # Subprocess smoke test of `python -m scale_forecasting.mcp`
    proc = subprocess.run(
        [sys.executable, "-m", "scale_forecasting.mcp"],
        input='{"jsonrpc":"2.0","id":1,"method":"ping"}\n',
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert json.loads(proc.stdout.strip()) == {"jsonrpc": "2.0", "id": 1, "result": {}}

    probe_proc = subprocess.run(
        [sys.executable, "-m", "scale_forecasting.mcp", "--probe-env"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe_proc.returncode == 0
    assert json.loads(probe_proc.stdout)["models"]["total"] == 34
