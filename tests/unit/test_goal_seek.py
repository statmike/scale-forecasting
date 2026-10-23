"""Unit tests for top-down goal-seek / inverse-forecasting QP allocation (`goal_seek.py`)."""

from __future__ import annotations

import matplotlib.figure
import numpy as np
import pandas as pd
import pytest

from scale_forecasting.config import RunConfig
from scale_forecasting.data_gen.generator import GenConfig, generate_panel
from scale_forecasting.errors import ConfigError, GoalSeekInfeasibleError
from scale_forecasting.goal_seek import (
    GOAL_SEEK_COLUMNS,
    GoalSeekResult,
    allocate_target,
    plot_goal_seek,
    summarize_goal_seek,
)
from scale_forecasting.reconciliation import TOTAL_NODE_ID, build_hierarchy, verify_coherence


def _flat_forecasts_and_history() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    h_dates = pd.date_range("2024-01-01", periods=24, freq="MS")
    f_dates = pd.date_range("2026-01-01", periods=4, freq="MS")
    stores = ["Store_A", "Store_B", "Store_C"]
    means = {"Store_A": 20.0, "Store_B": 30.0, "Store_C": 50.0}
    stds = {"Store_A": 1.5, "Store_B": 3.0, "Store_C": 6.0}

    hist_rows = []
    pred_rows = []
    oof_rows = []
    for s in stores:
        for d in h_dates:
            y_val = means[s] + rng.normal(0.0, stds[s])
            hist_rows.append({"ts_id": s, "ds": d, "y": float(y_val)})
            oof_rows.append(
                {
                    "ts_id": s,
                    "model_type": "naive",
                    "fold_id": 0,
                    "cutoff_date": pd.Timestamp("2023-12-01"),
                    "ds": d,
                    "y_true": float(y_val),
                    "yhat": float(means[s]),
                }
            )
        for d in f_dates:
            pred_rows.append(
                {
                    "ts_id": s,
                    "model_type": "naive",
                    "forecast_date": d,
                    "yhat": float(means[s]),
                    "yhat_lower": float(means[s] - 2.0 * stds[s]),
                    "yhat_upper": float(means[s] + 2.0 * stds[s]),
                }
            )
    return pd.DataFrame(pred_rows), pd.DataFrame(hist_rows), pd.DataFrame(oof_rows)


@pytest.mark.parametrize(
    "method",
    ["mint_qp", "wls_var", "wls_struct", "proportional", "equal"],
)
def test_allocate_target_exact_sum_across_methods(method: str) -> None:
    preds_df, hist_df, oof_df = _flat_forecasts_and_history()
    # Base sum per period is 20 + 30 + 50 = 100; total across 4 periods is 400.
    # Allocate a scalar total horizon target of 480 (+20% lift).
    alloc = allocate_target(
        preds_df,
        target=480.0,
        history_df=hist_df,
        oof_df=oof_df,
        method=method,  # type: ignore[arg-type]
    )
    assert list(alloc.columns[: len(GOAL_SEEK_COLUMNS)]) == list(GOAL_SEEK_COLUMNS)
    # Each of the 4 dates had equal base sum (100), so each date gets target 120.
    for _dt, grp in alloc.groupby("forecast_date"):
        assert abs(float(grp["y_allocated"].sum()) - 120.0) < 1e-6
        assert abs(float(grp["allocation_share"].sum()) - 1.0) < 1e-6
    assert abs(float(alloc["y_allocated"].sum()) - 480.0) < 1e-6

    # Check return_result=True wrapper
    res = allocate_target(
        preds_df,
        target=[110.0, 120.0, 130.0, 140.0],
        history_df=hist_df,
        oof_df=oof_df,
        method=method,  # type: ignore[arg-type]
        return_result=True,
    )
    assert isinstance(res, GoalSeekResult)
    assert res.max_target_discrepancy < 1e-6
    summary = summarize_goal_seek(res)
    assert len(summary) == 3
    assert abs(float(summary["y_allocated_sum"].sum()) - 500.0) < 1e-6


def test_allocate_target_respects_box_constraints_and_shifts_intervals() -> None:
    preds_df, hist_df, oof_df = _flat_forecasts_and_history()
    # Base per period is Store_A=20, Store_B=30, Store_C=50 (sum=100).
    # Target per period = 130, but cap Store_C at <= 52 and floor Store_A at >= 28.
    f_dates = sorted(preds_df["forecast_date"].unique())
    target_by_date = dict.fromkeys(f_dates, 130.0)
    alloc = allocate_target(
        preds_df,
        target=target_by_date,
        history_df=hist_df,
        oof_df=oof_df,
        method="mint_qp",
        bounds={"Store_A": (28.0, 45.0), "Store_C": (40.0, 52.0)},
        floor=0.0,
        ceiling=100.0,
    )
    for _dt, grp in alloc.groupby("forecast_date"):
        assert abs(float(grp["y_allocated"].sum()) - 130.0) < 1e-6
    a_vals = alloc.loc[alloc["ts_id"] == "Store_A", "y_allocated"].to_numpy(dtype=float)
    c_vals = alloc.loc[alloc["ts_id"] == "Store_C", "y_allocated"].to_numpy(dtype=float)
    assert bool(np.all(a_vals >= 28.0 - 1e-6))
    assert bool(np.all(a_vals <= 45.0 + 1e-6))
    assert bool(np.all(c_vals >= 40.0 - 1e-6))
    assert bool(np.all(c_vals <= 52.0 + 1e-6))
    assert bool(alloc["bound_active"].any())

    # Verify intervals shifted and respect box bounds
    assert bool((alloc["yhat_lower"] <= alloc["y_allocated"] + 1e-6).all())
    assert bool((alloc["y_allocated"] <= alloc["yhat_upper"] + 1e-6).all())


def test_allocate_target_hierarchical_coherence_and_internal_node_target() -> None:
    raw = generate_panel(
        4,
        cfg=GenConfig(history=30, freq="MS", with_hierarchy=True),
        seed=11,
    )
    cfg = RunConfig.model_validate(
        {
            "run_name": "gs-hier",
            "data": {"source_table": "p.d.t", "freq": "MS", "horizon": 3},
            "models": ["naive"],
            "hierarchy": {
                "enabled": True,
                "levels": [["region"], ["region", "category"]],
            },
        }
    )
    panel, spec = build_hierarchy(raw, cfg)
    f_dates = pd.date_range("2025-01-01", periods=3, freq="MS")
    pred_rows = []
    for nid in spec.node_ids:
        last_y = float(panel.loc[panel["ts_id"] == nid, "y"].iloc[-1])
        for d in f_dates:
            pred_rows.append(
                {
                    "ts_id": nid,
                    "model_type": "naive",
                    "forecast_date": d,
                    "yhat": last_y,
                    "yhat_lower": last_y * 0.9,
                    "yhat_upper": last_y * 1.1,
                }
            )
    preds_df = pd.DataFrame(pred_rows)

    # Allocate a root __total__ target across the 3 horizon steps
    res = allocate_target(
        preds_df,
        target=[500.0, 520.0, 540.0],
        spec=spec,
        history_df=panel,
        target_node=TOTAL_NODE_ID,
        method="mint_qp",
        return_result=True,
    )
    assert isinstance(res, GoalSeekResult)
    assert res.max_target_discrepancy < 1e-6
    assert res.max_coherence_discrepancy < 1e-6
    assert verify_coherence(res.dataframe, spec) < 1e-6

    # Also test targeting an intermediate region node (`region=...`)
    region_nodes = [n for n in spec.node_ids if n.startswith("region=") and "/" not in n]
    assert region_nodes
    sub_target_node = region_nodes[0]
    sub_alloc = allocate_target(
        preds_df,
        target=600.0,
        spec=spec,
        history_df=panel,
        target_node=sub_target_node,
        method="wls_var",
    )
    sub_node_sum = float(sub_alloc.loc[sub_alloc["ts_id"] == sub_target_node, "y_allocated"].sum())
    assert abs(sub_node_sum - 600.0) < 1e-6
    assert verify_coherence(sub_alloc, spec) < 1e-6

    fig = plot_goal_seek(res)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_allocate_target_infeasible_bounds_raises_goal_seek_infeasible_error() -> None:
    preds_df, hist_df, _ = _flat_forecasts_and_history()
    assert issubclass(GoalSeekInfeasibleError, ConfigError)
    assert issubclass(GoalSeekInfeasibleError, ValueError)

    # Lower bounds sum to 3 * 50 = 150 per period > target 100 per period
    with pytest.raises(GoalSeekInfeasibleError):
        allocate_target(
            preds_df,
            target=[100.0, 100.0, 100.0, 100.0],
            history_df=hist_df,
            floor=50.0,
        )

    # Upper bounds sum to 3 * 20 = 60 per period < target 100 per period
    with pytest.raises(GoalSeekInfeasibleError):
        allocate_target(
            preds_df,
            target=[100.0, 100.0, 100.0, 100.0],
            history_df=hist_df,
            ceiling=20.0,
        )

    # Inverted per-series bounds (lower > upper)
    with pytest.raises(GoalSeekInfeasibleError):
        allocate_target(
            preds_df,
            target=400.0,
            bounds={"Store_A": (50.0, 30.0)},
        )


def test_sdk_playground_report_and_mcp_goal_seek_integration(tmp_path) -> None:
    import json

    import scale_forecasting as sf
    from scale_forecasting.mcp import McpServer
    from scale_forecasting.playground import bakeoff, run_model

    assert sf.ScaleForecastingError is sf.ScaleForecastError
    assert issubclass(sf.GoalSeekInfeasibleError, sf.ScaleForecastingError)

    preds_df, hist_df, oof_df = _flat_forecasts_and_history()
    cfg = RunConfig.model_validate(
        {
            "run_name": "gs-sdk",
            "data": {"source_table": "p.d.t", "freq": "MS", "horizon": 4},
            "models": ["naive"],
        }
    )
    f = sf.Forecaster(cfg, data=hist_df)

    # Calling convention 1: f.allocate_target(480.0, ...)
    gs_scalar = f.allocate_target(
        480.0,
        forecasts_df=preds_df,
        oof_df=oof_df,
        locked_nodes=["Store_A"],
        prices={"Store_A": 10.0, "Store_B": 12.0, "Store_C": 15.0},
        unit_cost=5.0,
        include_inventory_impact=True,
    )
    assert abs(float(gs_scalar["allocated_yhat"].sum()) - 480.0) < 1e-6
    assert bool(gs_scalar.loc[gs_scalar["ts_id"] == "Store_A", "is_locked"].all())
    assert "gross_margin_delta" in gs_scalar.columns
    assert "delta_reorder_point" in gs_scalar.columns

    # Calling convention 2: f.allocate_target(preds_df, targets={"__total__": 480.0}, ...)
    gs_df_first = f.allocate_target(
        preds_df,
        targets={"__total__": 480.0},
        oof_df=oof_df,
        method="mint_shrink",
    )
    assert abs(float(gs_df_first["allocated_yhat"].sum()) - 480.0) < 1e-6

    ax = f.plot_goal_seek(gs_scalar)
    assert ax is not None

    html_path = f.export_report(
        tmp_path / "gs_report.html",
        predictions_df=preds_df,
        oof_df=oof_df,
        goal_seek_df=gs_scalar,
        write_json=True,
    )
    assert html_path.is_file()
    html_text = html_path.read_text(encoding="utf-8")
    assert "Top-Down Goal-Seek Target Allocation" in html_text
    json_payload = json.loads(html_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert len(json_payload["goal_seek"]) > 0

    # PlaygroundRun and BakeOff integration
    pr = run_model("naive_mean", horizon=4, backtest=True)
    pr_gs = pr.allocate_target(target_total=500.0)
    assert abs(float(pr_gs["allocated_yhat"].sum()) - 500.0) < 1e-4
    assert pr.plot_goal_seek(pr_gs) is not None

    bo = bakeoff(models=["naive_drift", "naive_mean"], horizon=4, n_folds=1)
    bo_gs = bo.allocate_target(target_total=600.0)
    assert abs(float(bo_gs["allocated_yhat"].sum()) - 600.0) < 1e-4
    assert bo.plot_goal_seek(bo_gs) is not None
    bo_html = bo.export_report(tmp_path / "bo_gs.html", goal_seek_df=bo_gs)
    assert bo_html.is_file()

    # MCP server integration (`run_playground` + direct `allocate_target` alias)
    srv = McpServer()
    assert len(srv.list_tools()) == 12
    assert len(srv.list_resources()) == 7
    mcp_pg = srv._dispatch_tool(
        "run_playground",
        {
            "model": "naive_mean",
            "n_series": 2,
            "horizon": 4,
            "backtest": True,
            "include_goal_seek": True,
            "target_total": 500.0,
        },
    )
    assert "goal_seek" in mcp_pg
    assert len(mcp_pg["goal_seek"]) > 0

    mcp_direct = srv._dispatch_tool(
        "allocate_target",
        {"model": "naive_mean", "horizon": 4, "target_total": 450.0},
    )
    assert mcp_direct["status"] == "ok"
    assert len(mcp_direct["allocations"]) > 0


def test_allocate_target_smoothness_weight_reduces_temporal_roughness() -> None:
    import scale_forecasting as sf
    from scale_forecasting.playground import bakeoff, run_model

    dates = pd.date_range("2026-01-01", periods=6, freq="MS")
    stores = ["Store_A", "Store_B"]
    base_patterns = {
        "Store_A": [20.0, 60.0, 20.0, 60.0, 20.0, 60.0],
        "Store_B": [30.0, 90.0, 30.0, 90.0, 30.0, 90.0],
    }
    hw_patterns = {
        "Store_A": [2.0, 18.0, 2.0, 18.0, 2.0, 18.0],
        "Store_B": [3.0, 25.0, 3.0, 25.0, 3.0, 25.0],
    }
    rows = []
    for s in stores:
        for d, y_v, hw in zip(dates, base_patterns[s], hw_patterns[s], strict=True):
            rows.append(
                {
                    "ts_id": s,
                    "model_type": "naive",
                    "forecast_date": d,
                    "yhat": y_v,
                    "yhat_lower": max(0.0, y_v - hw),
                    "yhat_upper": y_v + hw,
                }
            )
    preds_df = pd.DataFrame(rows)

    # Base total is 240 + 360 = 600; allocate horizon target 780 (+180 lift)
    alloc_unsmoothed = allocate_target(
        preds_df,
        target_total=780.0,
        smoothness_weight=0.0,
        lower_bounds=10.0,
        upper_bounds=140.0,
    )
    alloc_smoothed = allocate_target(
        preds_df,
        target_total=780.0,
        smoothness_weight=10.0,
        lower_bounds=10.0,
        upper_bounds=140.0,
    )
    assert isinstance(alloc_unsmoothed, pd.DataFrame)
    assert isinstance(alloc_smoothed, pd.DataFrame)
    assert abs(float(alloc_unsmoothed["allocated_yhat"].sum()) - 780.0) < 1e-6
    assert abs(float(alloc_smoothed["allocated_yhat"].sum()) - 780.0) < 1e-6
    assert bool((alloc_smoothed["allocated_yhat"] >= 10.0 - 1e-6).all())
    assert bool((alloc_smoothed["allocated_yhat"] <= 140.0 + 1e-6).all())
    assert alloc_smoothed.attrs["smoothness_weight"] == 10.0

    def _roughness(df: pd.DataFrame, col: str) -> float:
        total_r = 0.0
        for _, grp in df.sort_values("forecast_date").groupby("ts_id"):
            vals = grp[col].to_numpy(dtype=float)
            total_r += float(np.sum(np.diff(vals) ** 2))
        return total_r

    assert _roughness(alloc_smoothed, "delta_yhat") < _roughness(alloc_unsmoothed, "delta_yhat")
    assert _roughness(alloc_smoothed, "allocated_yhat") < _roughness(
        alloc_unsmoothed, "allocated_yhat"
    )

    # Also verify uniform-weight method="ols" where unpenalized delta_yhat is constant
    alloc_ols_0 = allocate_target(preds_df, target_total=780.0, method="ols", smoothness_weight=0.0)
    alloc_ols_smooth = allocate_target(
        preds_df, target_total=780.0, method="ols", smoothness_weight=10.0
    )
    assert isinstance(alloc_ols_0, pd.DataFrame)
    assert isinstance(alloc_ols_smooth, pd.DataFrame)
    assert abs(float(alloc_ols_smooth["allocated_yhat"].sum()) - 780.0) < 1e-6
    assert _roughness(alloc_ols_smooth, "allocated_yhat") < _roughness(
        alloc_ols_0, "allocated_yhat"
    )

    # Verify Forecaster, PlaygroundRun, and BakeOff expose smoothness_weight
    cfg = RunConfig.model_validate(
        {
            "run_name": "gs-smooth-sdk",
            "data": {"source_table": "p.d.t", "freq": "MS", "horizon": 6},
            "models": ["naive"],
        }
    )
    f = sf.Forecaster(cfg)
    f_smooth = f.allocate_target(780.0, forecasts_df=preds_df, smoothness_weight=5.0)
    assert abs(float(f_smooth["allocated_yhat"].sum()) - 780.0) < 1e-6

    pr = run_model("naive_mean", horizon=4, backtest=False)
    pr_smooth = pr.allocate_target(target_total=500.0, smoothness_weight=5.0)
    assert abs(float(pr_smooth["allocated_yhat"].sum()) - 500.0) < 1e-4

    bo = bakeoff(models=["naive_drift", "naive_mean"], horizon=4, n_folds=1)
    bo_smooth = bo.allocate_target(target_total=600.0, smoothness_weight=5.0)
    assert abs(float(bo_smooth["allocated_yhat"].sum()) - 600.0) < 1e-4

    # Invalid smoothness_weight raises ConfigError
    for bad_sw in (-1.0, float("nan"), True):
        with pytest.raises(ConfigError, match="smoothness_weight"):
            allocate_target(preds_df, target_total=780.0, smoothness_weight=bad_sw)  # type: ignore[arg-type]


def test_allocate_target_with_similarity_hierarchy_and_cluster_node_target() -> None:
    """Verify F6 (`similarity_hierarchy=True`) in `allocate_target`: auto-discovers latent
    `similarity_cluster_id=cluster_*` nodes from a flat leaf panel and supports targeting a
    specific cluster node directly.
    """
    preds_df, hist_df, oof_df = _flat_forecasts_and_history()

    # 1. Total target allocation with auto-discovered similarity hierarchy
    res = allocate_target(
        preds_df,
        target=480.0,
        history_df=hist_df,
        oof_df=oof_df,
        similarity_hierarchy=True,
        similarity_n_clusters=2,
        return_result=True,
    )
    assert isinstance(res, GoalSeekResult)
    assert res.max_target_discrepancy < 1e-6
    assert res.max_coherence_discrepancy < 1e-6
    cluster_nodes = [
        nid
        for nid in res.dataframe["ts_id"].unique()
        if str(nid).startswith("similarity_cluster_id=")
    ]
    assert len(cluster_nodes) == 2

    # 2. Target a specific discovered cluster node (`similarity_cluster_id=cluster_0`) on flat input
    c0_node = sorted(cluster_nodes)[0]
    alloc_c0 = allocate_target(
        preds_df,
        target=240.0,
        node_id=c0_node,
        history_df=hist_df,
        oof_df=oof_df,
        similarity_hierarchy=True,
        similarity_n_clusters=2,
        return_result=True,
    )
    assert isinstance(alloc_c0, GoalSeekResult)
    c0_sum = float(
        alloc_c0.dataframe.loc[alloc_c0.dataframe["ts_id"] == c0_node, "y_allocated"].sum()
    )
    assert abs(c0_sum - 240.0) < 1e-6
    assert alloc_c0.max_coherence_discrepancy < 1e-6
