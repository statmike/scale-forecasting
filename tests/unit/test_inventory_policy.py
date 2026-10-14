"""Unit and contract tests for Track B: Operational Inventory & Replenishment Policy Translators."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.axes
import numpy as np
import pandas as pd
import pytest

from scale_forecasting import (
    Forecaster,
    McpServer,
    RunConfig,
    build_report_payload,
    compute_inventory_policy,
    plot_inventory_policy,
    render_html_report,
)
from scale_forecasting.diagnostics import _INVENTORY_POLICY_COLUMNS
from scale_forecasting.playground import bakeoff, run_model


def _make_predictions_df(
    *,
    n_steps: int = 14,
    daily_demand: float = 100.0,
    half_width_80: float = 20.0,
) -> pd.DataFrame:
    dates = pd.date_range("2026-04-01", periods=n_steps, freq="D")
    rows: list[dict[str, object]] = []
    for tid, scale in (("sku_fast", 1.0), ("sku_slow", 0.4)):
        for mtype, w_mult in (("theta", 1.0), ("naive_seasonal", 1.5)):
            for dt in dates:
                mu = daily_demand * scale
                hw = half_width_80 * scale * w_mult
                rows.append(
                    {
                        "ts_id": tid,
                        "model_type": mtype,
                        "forecast_date": dt,
                        "yhat": mu,
                        "yhat_lower": mu - hw,
                        "yhat_upper": mu + hw,
                    }
                )
    return pd.DataFrame(rows)


def _make_autocorrelated_oof_df(rho: float = 0.70) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    dates = pd.date_range("2026-02-01", periods=28, freq="D")
    rows: list[dict[str, object]] = []
    for tid in ("sku_fast", "sku_slow"):
        err = 0.0
        for step, dt in enumerate(dates, start=1):
            err = rho * err + float(rng.normal(0.0, 8.0))
            rows.append(
                {
                    "ts_id": tid,
                    "model_type": "theta",
                    "fold_id": 0,
                    "horizon_step": step,
                    "forecast_date": dt,
                    "y_true": 100.0,
                    "yhat": 100.0 - err,
                }
            )
    return pd.DataFrame(rows)


def test_compute_inventory_policy_empty_and_columns_contract() -> None:
    """Empty input returns an empty DataFrame with the canonical 30 columns."""
    empty_res = compute_inventory_policy(pd.DataFrame())
    assert empty_res.empty
    assert list(empty_res.columns) == list(_INVENTORY_POLICY_COLUMNS)


def test_continuous_vs_periodic_review_and_covariance_bounds() -> None:
    """Verify (s, S) continuous vs. (R, S) periodic review and AR(1) safety stock bounds."""
    pred_df = _make_predictions_df(n_steps=14, daily_demand=100.0, half_width_80=25.631)
    # Note: 25.631 / (2 * 1.28155) ~= 10.0 per-step sigma!

    # 1. Continuous review (R=0, L=7) at 95% CSL
    cont = compute_inventory_policy(
        pred_df,
        lead_time=7.0,
        review_period=0.0,
        service_level=0.95,
        model_type="theta",
    )
    assert len(cont) == 2
    fast = cont[cont["ts_id"] == "sku_fast"].iloc[0]
    assert fast["policy_type"] == "continuous_s_S"
    assert fast["target_mode"] == "cycle_service_level"
    assert fast["protection_steps"] == pytest.approx(7.0)
    assert fast["lead_time_demand"] == pytest.approx(700.0)
    assert fast["protection_demand"] == pytest.approx(700.0)

    # With default error_autocorr_rho = 0.35 in (0, 1):
    # naive_iid_safety_stock < safety_stock < naive_linear_safety_stock
    assert fast["naive_iid_safety_stock"] < fast["safety_stock"]
    assert fast["safety_stock"] < fast["naive_linear_safety_stock"]
    assert fast["reorder_point"] == pytest.approx(700.0 + fast["safety_stock"], rel=1e-4)
    assert fast["order_up_to_level"] == pytest.approx(fast["reorder_point"], rel=1e-4)

    # 2. Periodic review (R=7, L=7 -> P=14)
    per = compute_inventory_policy(
        pred_df,
        lead_time=7.0,
        review_period=7.0,
        service_level=0.95,
        model_type="theta",
    )
    fast_p = per[per["ts_id"] == "sku_fast"].iloc[0]
    assert fast_p["policy_type"] == "periodic_R_S"
    assert fast_p["protection_steps"] == pytest.approx(14.0)
    assert fast_p["lead_time_demand"] == pytest.approx(700.0)
    assert fast_p["protection_demand"] == pytest.approx(1400.0)
    assert fast_p["safety_stock"] > fast["safety_stock"]
    assert fast_p["order_up_to_level"] > fast_p["reorder_point"]


def test_service_target_modes_csl_fill_rate_and_newsvendor() -> None:
    """Verify Type-1 CSL, Type-2 Fill Rate normal-loss inversion, and Newsvendor fractile."""
    pred_df = _make_predictions_df(n_steps=14, daily_demand=100.0, half_width_80=30.0)

    # Mode A: Type-2 Fill Rate (beta = 0.985)
    fr_df = compute_inventory_policy(
        pred_df,
        lead_time=7.0,
        fill_rate=0.985,
        model_type="theta",
    )
    row_fr = fr_df[fr_df["ts_id"] == "sku_fast"].iloc[0]
    assert row_fr["target_mode"] == "fill_rate"
    assert row_fr["implied_fill_rate"] == pytest.approx(0.985, abs=2e-3)
    assert row_fr["expected_units_short"] == pytest.approx(
        (1.0 - 0.985) * row_fr["protection_demand"], rel=5e-2
    )

    # Mode B: Newsvendor critical fractile (service_level=None, fill_rate=None)
    # cu = 9.0, co = 1.0 -> alpha* = 9 / 10 = 0.90
    nv_df = compute_inventory_policy(
        pred_df,
        lead_time=7.0,
        service_level=None,
        fill_rate=None,
        holding_cost=1.0,
        stockout_cost=9.0,
        model_type="theta",
    )
    row_nv = nv_df[nv_df["ts_id"] == "sku_fast"].iloc[0]
    assert row_nv["target_mode"] == "newsvendor_critical_fractile"
    assert row_nv["target_service_level"] == pytest.approx(0.90, abs=1e-4)
    assert row_nv["safety_factor_z"] == pytest.approx(1.2816, abs=2e-3)


def test_lead_time_variability_and_empirical_oof_autocorrelation() -> None:
    """Lead-time standard deviation and OOF AR(1) autocorrelation increase safety stock."""
    pred_df = _make_predictions_df(n_steps=14, daily_demand=100.0, half_width_80=20.0)
    oof_df = _make_autocorrelated_oof_df(rho=0.75)

    base = compute_inventory_policy(pred_df, lead_time=7.0, lead_time_std=0.0, model_type="theta")
    with_lt_var = compute_inventory_policy(
        pred_df, lead_time=7.0, lead_time_std=1.5, model_type="theta"
    )
    with_oof = compute_inventory_policy(
        pred_df, oof_df=oof_df, lead_time=7.0, lead_time_std=0.0, model_type="theta"
    )

    b_fast = base[base["ts_id"] == "sku_fast"].iloc[0]
    lt_fast = with_lt_var[with_lt_var["ts_id"] == "sku_fast"].iloc[0]
    oof_fast = with_oof[with_oof["ts_id"] == "sku_fast"].iloc[0]

    assert lt_fast["lead_time_var_std"] == pytest.approx(100.0 * 1.5)
    assert lt_fast["total_protection_std"] > b_fast["total_protection_std"]
    assert lt_fast["safety_stock"] > b_fast["safety_stock"]

    # Empirical lag-1 autocorrelation from oof_df (~0.75) > default 0.35
    assert oof_fast["error_autocorr_rho"] > 0.45
    assert oof_fast["safety_stock"] > b_fast["safety_stock"]


def test_per_sku_overrides_moq_and_lot_sizing() -> None:
    """Verify per-SKU inventory_df (DataFrame and dict), MOQ, order_multiple, and triggers."""
    pred_df = _make_predictions_df(n_steps=10, daily_demand=50.0, half_width_80=10.0)

    # 1. DataFrame overrides
    inv_df = pd.DataFrame(
        [
            {
                "ts_id": "sku_fast",
                "on_hand": 100.0,
                "on_order": 50.0,
                "backorders": 10.0,
                "lead_time": 5.0,
                "min_order_qty": 200.0,
                "order_multiple": 25.0,
            },
            {
                "ts_id": "sku_slow",
                "on_hand": 500.0,  # Well above ROP -> no order needed
                "on_order": 0.0,
                "backorders": 0.0,
                "lead_time": 5.0,
            },
        ]
    )
    pol = compute_inventory_policy(pred_df, inventory_df=inv_df, model_type="theta")
    fast = pol[pol["ts_id"] == "sku_fast"].iloc[0]
    slow = pol[pol["ts_id"] == "sku_slow"].iloc[0]

    assert fast["inventory_position"] == pytest.approx(140.0)
    assert bool(fast["reorder_triggered"]) is True
    assert fast["recommended_order_qty"] >= 200.0
    assert fast["recommended_order_qty"] % 25.0 == pytest.approx(0.0)

    assert slow["inventory_position"] == pytest.approx(500.0)
    assert bool(slow["reorder_triggered"]) is False
    assert slow["recommended_order_qty"] == pytest.approx(0.0)

    # 2. Dict overrides
    pol_dict = compute_inventory_policy(
        pred_df,
        inventory_df={"sku_fast": {"on_hand": 9999.0}, "sku_slow": {"on_hand": 0.0}},
        model_type="theta",
    )
    fast_d = pol_dict[pol_dict["ts_id"] == "sku_fast"].iloc[0]
    slow_d = pol_dict[pol_dict["ts_id"] == "sku_slow"].iloc[0]
    assert bool(fast_d["reorder_triggered"]) is False
    assert bool(slow_d["reorder_triggered"]) is True


def test_quantiles_column_fallback_and_plot_inventory_policy() -> None:
    """Verify sigma extraction from quantiles JSON/dict when yhat_lower/upper are omitted."""
    dates = pd.date_range("2026-05-01", periods=7, freq="D")
    pred_q = pd.DataFrame(
        [
            {
                "ts_id": "sku_q",
                "model_type": "chronos2",
                "forecast_date": dt,
                "yhat": 80.0,
                "quantiles": json.dumps({"p10": 65.0, "p50": 80.0, "p90": 95.0}),
            }
            for dt in dates
        ]
    )
    pol = compute_inventory_policy(pred_q, lead_time=5.0)
    assert len(pol) == 1
    assert pol.iloc[0]["safety_stock"] > 0.0

    ax = plot_inventory_policy(pol)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_playground_bakeoff_forecaster_report_and_mcp_integration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify end-to-end integration across PlaygroundRun, BakeOff, Forecaster, report, and MCP."""
    # 1. PlaygroundRun
    run = run_model("theta", horizon=10, backtest=True)
    run_pol = run.inventory_policy_df(lead_time=5.0, service_level=0.95)
    assert not run_pol.empty
    assert list(run_pol.columns) == list(_INVENTORY_POLICY_COLUMNS)
    ax_run = run.plot_inventory_policy(lead_time=5.0)
    assert isinstance(ax_run, matplotlib.axes.Axes)

    # 2. BakeOff
    bo = bakeoff(models=["naive_seasonal", "theta"], horizon=10, n_folds=2)
    bo_pol = bo.inventory_policy_df(lead_time=5.0, fill_rate=0.98)
    assert len(bo_pol) >= 2
    ax_bo = bo.plot_inventory_policy(model_type="theta", lead_time=5.0)
    assert isinstance(ax_bo, matplotlib.axes.Axes)

    # 3. HTML & JSON Report export
    html_out = tmp_path / "inv_report.html"
    bo.export_report(html_out, write_json=True)
    html_txt = html_out.read_text(encoding="utf-8")
    assert "Inventory Policy (ROP / SS)" in html_txt
    assert "tab-inventory" in html_txt
    payload = json.loads(html_out.with_suffix(".json").read_text(encoding="utf-8"))
    assert "inventory_policy" in payload
    assert len(payload["inventory_policy"]) >= 2

    # Also verify direct build_report_payload / render_html_report
    direct_payload = build_report_payload(predictions_df=bo.predictions, oof_df=bo.oof)
    assert len(direct_payload["inventory_policy"]) >= 2
    direct_html = render_html_report(predictions_df=bo.predictions, oof_df=bo.oof)
    assert "tab-inventory" in direct_html

    # 4. Forecaster SDK wrapper (mocked registry reads)
    from scale_forecasting.registry import reads

    monkeypatch.setattr(
        reads,
        "read_predictions",
        lambda run_id, **kw: bo.predictions.to_dict(orient="records"),
    )
    monkeypatch.setattr(
        reads,
        "read_oof",
        lambda run_id, **kw: bo.oof.to_dict(orient="records") if bo.oof is not None else [],
    )
    cfg = RunConfig.model_validate(
        {
            "run_name": "test-inv-sdk",
            "data": {"source_table": "source_series", "horizon": 10},
            "models": ["theta"],
        }
    )
    f = Forecaster(cfg)
    f_pol = f.inventory_policy_df(lead_time=7.0, service_level=0.95)
    assert len(f_pol) >= 2
    ax_f = f.plot_inventory_policy(model_type="theta", lead_time=7.0)
    assert isinstance(ax_f, matplotlib.axes.Axes)

    # 5. MCP Server run_playground with include_inventory_policy=True
    mcp = McpServer()
    resp = mcp.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "run_playground",
                "arguments": {
                    "model": "theta",
                    "horizon": 10,
                    "n_series": 2,
                    "include_inventory_policy": True,
                    "inventory_params": {"lead_time": 5.0, "service_level": 0.95},
                },
            },
        }
    )
    assert resp is not None and not resp["result"]["isError"]
    mcp_data = json.loads(resp["result"]["content"][0]["text"])
    assert "inventory_policy" in mcp_data
    assert len(mcp_data["inventory_policy"]) == 1
    assert mcp_data["inventory_policy"][0]["reorder_point"] > 0.0
