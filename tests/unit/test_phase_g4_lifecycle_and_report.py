"""Unit and contract tests for Phase G.4: Realized Accuracy, Stability, Exceptions & Reports."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scale_forecasting import (
    McpServer,
    build_report_payload,
    compute_exceptions_queue,
    compute_realized_accuracy,
    compute_revision_stability,
    render_html_report,
)
from scale_forecasting.playground import bakeoff, run_model
from scale_forecasting.report import main as report_main


def test_compute_realized_accuracy_and_drift_detection() -> None:
    """compute_realized_accuracy joins predictions with new actuals and flags accuracy drift."""
    dates = pd.date_range("2026-02-01", periods=7, freq="D")
    oof_dates = pd.date_range("2026-01-01", periods=14, freq="D")

    # Backtest OOF had low ~3% error on both series
    oof_rows = []
    for tid in ("sku_stable", "sku_drifted"):
        for dt in oof_dates:
            oof_rows.append(
                {
                    "ts_id": tid,
                    "model_type": "theta",
                    "fold_id": 0,
                    "forecast_date": dt,
                    "y_true": 100.0,
                    "yhat": 103.0,
                }
            )
    oof_df = pd.DataFrame(oof_rows)

    # Published forward predictions
    pred_rows = []
    act_rows = []
    for dt in dates:
        # Stable series: ~3% live error, inside [90, 110] bounds
        pred_rows.append(
            {
                "ts_id": "sku_stable",
                "model_type": "theta",
                "forecast_date": dt,
                "yhat": 100.0,
                "yhat_lower": 90.0,
                "yhat_upper": 110.0,
            }
        )
        act_rows.append({"ts_id": "sku_stable", "ds": dt, "y": 103.0})

        # Drifted series: 35% live error (>10x backtest error), outside bounds
        pred_rows.append(
            {
                "ts_id": "sku_drifted",
                "model_type": "theta",
                "forecast_date": dt,
                "yhat": 100.0,
                "yhat_lower": 90.0,
                "yhat_upper": 110.0,
            }
        )
        act_rows.append({"ts_id": "sku_drifted", "ds": dt, "y": 135.0})

    real_df = compute_realized_accuracy(
        pd.DataFrame(pred_rows),
        pd.DataFrame(act_rows),
        oof_df=oof_df,
        metric="wape",
        drift_threshold=1.5,
    )

    assert len(real_df) == 2
    drifted = real_df[real_df["ts_id"] == "sku_drifted"].iloc[0]
    stable = real_df[real_df["ts_id"] == "sku_stable"].iloc[0]

    assert bool(drifted["drift_flagged"]) is True
    assert drifted["drift_ratio"] > 5.0
    assert drifted["realized_coverage_80"] == pytest.approx(0.0)

    assert bool(stable["drift_flagged"]) is False
    assert stable["drift_ratio"] == pytest.approx(1.0, rel=0.1)
    assert stable["realized_coverage_80"] == pytest.approx(1.0)


def test_compute_revision_stability_whipsaw() -> None:
    """compute_revision_stability measures MAR, CoV, and flags whipsaw jumps across runs."""
    dates = pd.date_range("2026-03-01", periods=5, freq="D")
    run_1 = pd.DataFrame(
        [
            {"ts_id": "sku_calm", "model_type": "theta", "forecast_date": d, "yhat": 100.0}
            for d in dates
        ]
        + [
            {"ts_id": "sku_whipsaw", "model_type": "theta", "forecast_date": d, "yhat": 100.0}
            for d in dates
        ]
    )
    run_2 = pd.DataFrame(
        [
            {"ts_id": "sku_calm", "model_type": "theta", "forecast_date": d, "yhat": 102.0}
            for d in dates
        ]
        + [
            {
                "ts_id": "sku_whipsaw",
                "model_type": "theta",
                "forecast_date": d,
                "yhat": 155.0 if i == 2 else 105.0,
            }
            for i, d in enumerate(dates)
        ]
    )

    stab = compute_revision_stability({"run_w1": run_1, "run_w2": run_2}, jump_threshold_pct=0.30)
    assert len(stab) == 2

    whip = stab[stab["ts_id"] == "sku_whipsaw"].iloc[0]
    calm = stab[stab["ts_id"] == "sku_calm"].iloc[0]

    assert bool(whip["whipsaw_flagged"]) is True
    assert whip["max_single_step_jump_pct"] == pytest.approx(0.55)
    assert bool(calm["whipsaw_flagged"]) is False
    assert calm["mar"] == pytest.approx(2.0)

    # Also test single DataFrame input with run_id column
    concat_df = pd.concat(
        [run_1.assign(run_id="run_w1"), run_2.assign(run_id="run_w2")], ignore_index=True
    )
    stab2 = compute_revision_stability(concat_df, jump_threshold_pct=0.30)
    assert len(stab2) == 2


def test_compute_exceptions_queue_all_five_rules() -> None:
    """compute_exceptions_queue ranks cells across all 5 operational triage rules."""
    dates = pd.date_range("2026-01-01", periods=14, freq="D")
    oof_rows = []
    for fold in range(3):
        for step, dt in enumerate(dates[:7], start=1):
            f_dt = dt + pd.Timedelta(days=fold * 7)
            # Baseline naive_seasonal has small error (2 units)
            oof_rows.append(
                {
                    "ts_id": "sku_bad",
                    "model_type": "naive_seasonal",
                    "fold_id": fold,
                    "horizon_step": step,
                    "forecast_date": f_dt,
                    "y_true": 100.0,
                    "yhat": 102.0,
                }
            )
            # Candidate theta has huge chronic positive bias (+25 -> worse_than_naive + bias)
            oof_rows.append(
                {
                    "ts_id": "sku_bad",
                    "model_type": "theta",
                    "fold_id": fold,
                    "horizon_step": step,
                    "forecast_date": f_dt,
                    "y_true": 100.0,
                    "yhat": 125.0,
                }
            )
    oof_df = pd.DataFrame(oof_rows)

    # Forward predictions hitting physical cap=200 on 5/5 steps
    pred_df = pd.DataFrame(
        [
            {
                "ts_id": "sku_bad",
                "model_type": "theta",
                "forecast_date": d,
                "yhat": 200.0,
            }
            for d in dates[:5]
        ]
    )
    realized_df = pd.DataFrame(
        [
            {
                "ts_id": "sku_bad",
                "model_type": "theta",
                "drift_ratio": 2.4,
                "drift_flagged": True,
            }
        ]
    )
    stability_df = pd.DataFrame(
        [
            {
                "ts_id": "sku_bad",
                "model_type": "theta",
                "max_single_step_jump_pct": 0.45,
                "whipsaw_flagged": True,
            }
        ]
    )

    exc = compute_exceptions_queue(
        oof_df=oof_df,
        predictions_df=pred_df,
        realized_df=realized_df,
        stability_df=stability_df,
        cap=200.0,
        metric="wape",
    )

    assert len(exc) == 1
    row = exc.iloc[0]
    assert row["ts_id"] == "sku_bad"
    assert row["model_type"] == "theta"
    assert int(row["n_flags"]) == 5
    flags = set(str(row["exception_flags"]).split(","))
    assert flags == {
        "worse_than_naive",
        "chronic_bias",
        "realized_drift",
        "whipsaw_jump",
        "capacity_ceiling_hit",
    }
    assert "worse than naive_seasonal" in str(row["exception_reasons"])
    assert "Chronic over-forecasting" in str(row["exception_reasons"])


def test_html_and_json_report_export_and_cli(tmp_path: Path) -> None:
    """BakeOff, PlaygroundRun, report CLI, and MCP export self-contained HTML + JSON reports."""
    bo = bakeoff(
        models=["naive_seasonal", "theta"],
        horizon=7,
        n_folds=2,
    )
    exc = bo.exceptions_df()
    assert isinstance(exc, pd.DataFrame)

    html_path = tmp_path / "bakeoff_report.html"
    written = bo.export_report(html_path, title="Unit Test Report", write_json=True)
    assert written == html_path
    assert html_path.is_file()
    json_path = html_path.with_suffix(".json")
    assert json_path.is_file()

    html_text = html_path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html_text
    assert "Unit Test Report" in html_text
    assert "<svg" in html_text
    assert "EXPLORER_DATA" in html_text

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["title"] == "Unit Test Report"
    assert "fva" in payload
    assert "explorer" in payload

    # Single-model PlaygroundRun export_report
    run = run_model("naive_seasonal", horizon=7)
    single_html = tmp_path / "single.html"
    run.export_report(single_html)
    assert single_html.is_file()

    # Direct render_html_report + build_report_payload on empty frames
    empty_html = render_html_report(title="Empty Report")
    assert "Empty Report" in empty_html
    empty_payload = build_report_payload(title="Empty Report")
    assert empty_payload["kpis"]["n_models"] == 0

    # CLI entry point: python -m scale_forecasting.report
    cli_out = tmp_path / "cli_report.html"
    rc = report_main(
        ["--out", str(cli_out), "--models", "naive_seasonal", "theta", "--horizon", "7", "--json"]
    )
    assert rc == 0
    assert cli_out.is_file()
    assert cli_out.with_suffix(".json").is_file()

    # MCP run_playground with export_report_path
    mcp = McpServer()
    mcp_html = tmp_path / "mcp_report.html"
    resp = mcp.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "run_playground",
                "arguments": {
                    "model": "theta",
                    "horizon": 7,
                    "export_report_path": str(mcp_html),
                },
            },
        }
    )
    assert resp is not None and "result" in resp
    res = json.loads(resp["result"]["content"][0]["text"])
    assert res["report_path"] == str(mcp_html)
    assert mcp_html.is_file()
