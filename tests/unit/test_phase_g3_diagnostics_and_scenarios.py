"""Unit and contract tests for Phase G.3: FVA, Horizon Decay, Bias, Rollup & Scenarios."""

from __future__ import annotations

import matplotlib.axes
import numpy as np
import pandas as pd
import pytest

from scale_forecasting import (
    McpServer,
    RunConfig,
    apply_scenario_shocks,
    compute_bias_report,
    compute_fva,
    compute_horizon_decay,
    diebold_mariano_test,
    plot_fva,
    plot_horizon_decay,
    plot_scenarios,
    rollup_time,
    simulate_scenarios,
    summarize_scenarios,
    wilcoxon_pvalue,
)
from scale_forecasting.playground import bakeoff, run_model


def _make_synthetic_oof() -> pd.DataFrame:
    """Build synthetic OOF DataFrame across complexity tiers."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2025-01-01", periods=14, freq="D")
    rows_oof: list[dict[str, object]] = []

    for ts_idx in range(6):
        ts_id = f"store_{ts_idx}"
        actuals = 100.0 + 10.0 * np.sin(np.arange(14) * 2 * np.pi / 7.0)

        # Baseline: naive_seasonal (medium-large errors, flat across horizon)
        naive_err = rng.normal(0.0, 12.0, size=14)
        # Statistical: theta (medium errors)
        theta_err = rng.normal(0.0, 5.0, size=14)
        # ML: lightgbm (very accurate on steps 1..10, degrades on steps 11..14)
        lgbm_err = np.concatenate(
            [
                rng.normal(0.0, 1.5, size=10),
                rng.normal(0.0, 22.0, size=4),
            ]
        )
        # Deep learning: tft (worse than baseline — negative FVA)
        tft_err = rng.normal(0.0, 18.0, size=14)

        model_errors = {
            "naive_seasonal": naive_err,
            "theta": theta_err,
            "lightgbm": lgbm_err,
            "tft": tft_err,
        }
        for model_type, errs in model_errors.items():
            for fold_idx in range(2):
                for step_idx, (dt, y_val, err_val) in enumerate(
                    zip(dates, actuals, errs, strict=True), start=1
                ):
                    yh = float(y_val + err_val)
                    rows_oof.append(
                        {
                            "ts_id": ts_id,
                            "model_type": model_type,
                            "fold_id": fold_idx,
                            "horizon_step": step_idx,
                            "forecast_date": dt + pd.Timedelta(days=fold_idx * 14),
                            "y_true": float(y_val),
                            "yhat": yh,
                            "yhat_lower": yh - 10.0,
                            "yhat_upper": yh + 10.0,
                        }
                    )

    return pd.DataFrame(rows_oof)


def test_diebold_mariano_and_wilcoxon_significance() -> None:
    """HLN modified Diebold-Mariano and Wilcoxon detect superior models and handle ties."""
    rng = np.random.default_rng(7)
    loss_benchmark = rng.uniform(8.0, 15.0, size=30)
    loss_candidate = rng.uniform(1.0, 3.0, size=30)

    dm_stat, dm_p = diebold_mariano_test(loss_candidate, loss_benchmark, h=3)
    w_p = wilcoxon_pvalue(loss_candidate, loss_benchmark)

    assert dm_stat < -2.0
    assert dm_p < 0.01
    assert w_p < 0.01

    # Identical losses -> p = 1.0
    stat_tie, p_tie = diebold_mariano_test(loss_benchmark, loss_benchmark, h=1)
    assert stat_tie == 0.0
    assert p_tie == 1.0
    assert wilcoxon_pvalue(loss_benchmark, loss_benchmark) == 1.0


def test_compute_fva_and_plot_fva() -> None:
    """compute_fva orders the complexity ladder, computes FVA %, and attaches p-values."""
    oof_df = _make_synthetic_oof()
    fva = compute_fva(oof_df, metric="wape")

    assert list(fva["model_type"]) == ["naive_seasonal", "theta", "lightgbm", "tft"]
    theta_row = fva[fva["model_type"] == "theta"].iloc[0]
    lgbm_row = fva[fva["model_type"] == "lightgbm"].iloc[0]
    tft_row = fva[fva["model_type"] == "tft"].iloc[0]

    assert theta_row["fva_pct"] > 0.0
    assert bool(theta_row["significant_at_05"]) is True
    assert lgbm_row["fva_pct"] > 0.0
    # TFT has higher error than naive_seasonal -> negative FVA
    assert tft_row["fva_pct"] < 0.0
    assert bool(tft_row["significant_at_05"]) is False

    ax = plot_fva(fva)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_compute_horizon_decay_and_crossover() -> None:
    """compute_horizon_decay computes step-by-step error and identifies crossover_step h*."""
    oof_df = _make_synthetic_oof()
    decay = compute_horizon_decay(oof_df, metric="mae", baseline="naive_seasonal")

    assert not decay.empty
    assert set(decay["horizon_step"].unique()) == set(range(1, 15))
    lgbm_decay = decay[decay["model_type"] == "lightgbm"].sort_values("horizon_step")
    # LightGBM is better than naive_seasonal on steps 1..10 and crosses over on steps 11..14
    crossover = lgbm_decay["crossover_step"].iloc[0]
    assert pd.notna(crossover)
    assert int(crossover) >= 11

    ax = plot_horizon_decay(decay)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_compute_bias_report_trigg_tracking_signal() -> None:
    """compute_bias_report flags chronic over/under-forecasting via Trigg TS and fold streaks."""
    rows: list[dict[str, object]] = []
    base_dt = pd.Timestamp("2025-01-01")
    for fold in range(4):
        for step in range(1, 8):
            dt = base_dt + pd.Timedelta(days=fold * 7 + step)
            # Biased series: yhat is consistently +15 above y_true
            rows.append(
                {
                    "ts_id": "chronic_over",
                    "model_type": "naive_seasonal",
                    "fold_id": fold,
                    "horizon_step": step,
                    "forecast_date": dt,
                    "y_true": 100.0,
                    "yhat": 115.0,
                }
            )
            # Unbiased series: alternating +5 / -5
            sign = 1.0 if (step + fold) % 2 == 0 else -1.0
            rows.append(
                {
                    "ts_id": "balanced",
                    "model_type": "naive_seasonal",
                    "fold_id": fold,
                    "horizon_step": step,
                    "forecast_date": dt,
                    "y_true": 100.0,
                    "yhat": 100.0 + 5.0 * sign,
                }
            )

    oof_df = pd.DataFrame(rows)
    report = compute_bias_report(oof_df)

    over_row = report[report["ts_id"] == "chronic_over"].iloc[0]
    bal_row = report[report["ts_id"] == "balanced"].iloc[0]

    assert over_row["bias_status"] == "over_forecasting"
    assert over_row["tracking_signal"] > 4.0
    assert int(over_row["max_consecutive_over_folds"]) == 4

    assert bal_row["bias_status"] == "unbiased"


def test_rollup_time_covariance_adjusted_intervals() -> None:
    """rollup_time aggregates daily forecasts to weekly/monthly with AR(1) intervals."""
    dates = pd.date_range("2026-01-05", periods=14, freq="D")  # two full ISO weeks
    rows = []
    for dt in dates:
        rows.append(
            {
                "ts_id": "sku_1",
                "model_type": "theta",
                "forecast_date": dt,
                "yhat": 50.0,
                "yhat_lower": 40.0,
                "yhat_upper": 60.0,
            }
        )
    pred_df = pd.DataFrame(rows)
    rolled = rollup_time(pred_df, freq="W")

    assert len(rolled) == 2
    for _, row in rolled.iterrows():
        assert row["n_steps"] == 7
        assert row["yhat"] == pytest.approx(350.0)
        # Naive sum of widths is 7 * 20 = 140 (350 +/- 70 -> [280, 420])
        assert row["naive_sum_lower"] == pytest.approx(280.0)
        assert row["naive_sum_upper"] == pytest.approx(420.0)
        # Covariance-adjusted width with default rho=0.35 is strictly narrower than naive sum
        adj_width = row["yhat_upper"] - row["yhat_lower"]
        naive_width = row["naive_sum_upper"] - row["naive_sum_lower"]
        assert adj_width < naive_width
        assert adj_width > 20.0 * np.sqrt(7)


def test_counterfactual_scenarios_and_revenue_impact() -> None:
    """simulate_scenarios runs baseline + counterfactual shocks and computes revenue."""
    rng = np.random.default_rng(123)
    n_hist = 56
    horizon = 7
    hist_dates = pd.date_range("2025-01-01", periods=n_hist, freq="D")
    fut_dates = pd.date_range(hist_dates[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")

    price_hist = rng.uniform(18.0, 22.0, size=n_hist)
    promo_hist = rng.integers(0, 2, size=n_hist).astype(float)
    # True demand model: higher promo increases units, lower price increases units
    y_hist = (
        120.0
        - 3.5 * price_hist
        + 25.0 * promo_hist
        + 5.0 * np.sin(np.arange(n_hist) * 2 * np.pi / 7.0)
    )

    history_df = pd.DataFrame(
        {
            "ts_id": "sku_promo",
            "ds": hist_dates,
            "y": y_hist,
            "price": price_hist,
            "promo": promo_hist,
        }
    )
    future_df = pd.DataFrame(
        {
            "ts_id": "sku_promo",
            "ds": fut_dates,
            "price": np.full(horizon, 20.0),
            "promo": np.zeros(horizon),
        }
    )

    # Test apply_scenario_shocks DSL directly
    shocked = apply_scenario_shocks(
        future_df,
        {
            "price": "-15%",
            "promo": {
                "op": "set",
                "value": 1.0,
                "start_date": str(fut_dates[0].date()),
                "end_date": str(fut_dates[2].date()),
            },
        },
    )
    assert np.allclose(shocked["price"], 17.0)
    assert list(shocked["promo"]) == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]

    cfg = RunConfig.model_validate(
        {
            "run_name": "test-scenarios",
            "data": {"source_table": "test", "freq": "D", "horizon": horizon},
            "features": {"future_covariates": ["price", "promo"]},
            "models": ["regression_lags"],
            "backtest": {"enabled": True, "n_folds": 2, "horizon": horizon, "step": horizon},
        }
    )

    sim_df = simulate_scenarios(
        history_df,
        cfg,
        scenarios={
            "promo_15pct_off": {"price": "-15%", "promo": "=1"},
            "price_hike_10pct": {"price": "+10%"},
        },
        future_covariates_df=future_df,
        price_col="price",
    )

    assert set(sim_df["scenario"].unique()) == {"baseline", "promo_15pct_off", "price_hike_10pct"}
    summary = summarize_scenarios(sim_df)
    promo_row = summary[summary["scenario"] == "promo_15pct_off"].iloc[0]
    hike_row = summary[summary["scenario"] == "price_hike_10pct"].iloc[0]

    # Promo + 15% discount should increase units; 10% price hike should decrease units
    assert promo_row["delta_units"] > 0.0
    assert hike_row["delta_units"] < 0.0
    assert "revenue_delta" in summary.columns
    assert "mean_implied_elasticity" in summary.columns
    assert hike_row["mean_implied_elasticity"] < 0.0

    ax = plot_scenarios(sim_df)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_playground_and_mcp_phase_g3_integration() -> None:
    """BakeOff, PlaygroundRun, and MCP run_playground expose FVA, decay, bias, and scenarios."""
    import json

    bo = bakeoff(
        models=["naive_seasonal", "holtwinters", "theta"],
        horizon=7,
        n_folds=2,
    )
    fva = bo.fva_df()
    decay = bo.horizon_decay_df()
    bias = bo.bias_report_df()
    rolled = bo.rollup_time(freq="W")

    assert not fva.empty
    assert "fva_pct" in fva.columns
    assert not decay.empty
    assert not bias.empty
    assert not rolled.empty

    run = run_model("naive_seasonal", horizon=7)
    assert not run.rollup_time(freq="W").empty

    # MCP run_playground with scenarios and demand_profile
    mcp = McpServer()
    resp = mcp.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "run_playground",
                "arguments": {
                    "model": "regression_lags",
                    "horizon": 7,
                    "scenarios": {"promo_15pct_off": {"price_index": "-15%"}},
                    "price_col": "price_index",
                },
            },
        }
    )
    assert resp is not None and "result" in resp
    res = json.loads(resp["result"]["content"][0]["text"])
    assert "demand_profile" in res
    assert "scenario_summary" in res
    assert len(res["scenario_summary"]) == 2
