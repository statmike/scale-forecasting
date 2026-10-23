"""Unit tests for `scale_forecasting.inventory_optimizer` (Track L — F5 / R1.4)."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

from scale_forecasting.diagnostics import INVENTORY_POLICY_COLUMNS, compute_inventory_policy
from scale_forecasting.errors import ConfigError, GoalSeekInfeasibleError
from scale_forecasting.inventory_optimizer import (
    INVENTORY_OPTIMIZER_COLUMNS,
    compute_inventory_frontier,
    optimize_inventory_df,
    optimize_inventory_portfolio,
    plot_inventory_frontier,
)

matplotlib.use("Agg")


def _make_multi_sku_predictions_and_oof(
    n_steps: int = 14,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a 3-SKU forecast and OOF residual panel with distinct demand and error profiles."""
    dates = pd.date_range("2026-04-01", periods=n_steps, freq="D")
    oof_dates = pd.date_range("2026-03-01", periods=28, freq="D")
    rng = np.random.default_rng(123)

    pred_rows: list[dict[str, object]] = []
    oof_rows: list[dict[str, object]] = []

    sku_specs = [
        ("SKU_HIGH_PRIORITY", 120.0, 20.0, 0.65),
        ("SKU_MEDIUM", 80.0, 15.0, 0.35),
        ("SKU_LOW_PRIORITY", 50.0, 12.0, 0.10),
    ]
    for tid, base_y, sigma, rho in sku_specs:
        for h, dt in enumerate(dates):
            yh = base_y + 4.0 * np.sin(2.0 * np.pi * h / 7.0)
            pred_rows.append(
                {
                    "ts_id": tid,
                    "model_type": "lightgbm",
                    "forecast_date": dt,
                    "ds": dt,
                    "yhat": yh,
                    "yhat_lower": max(0.0, yh - 1.28155 * sigma),
                    "yhat_upper": yh + 1.28155 * sigma,
                }
            )
        # Generate AR(1) OOF errors
        err = np.zeros(len(oof_dates), dtype=float)
        noise = rng.normal(0.0, sigma * np.sqrt(1.0 - rho**2), size=len(oof_dates))
        for t in range(1, len(oof_dates)):
            err[t] = rho * err[t - 1] + noise[t]
        for dt_o, e_val in zip(oof_dates, err, strict=True):
            oof_rows.append(
                {
                    "ts_id": tid,
                    "model_type": "lightgbm",
                    "fold_id": "fold_1",
                    "ds": dt_o,
                    "y_true": base_y + e_val,
                    "yhat": base_y,
                }
            )

    return pd.DataFrame.from_records(pred_rows), pd.DataFrame.from_records(oof_rows)


def test_unconstrained_matches_base_inventory_policy_and_schema() -> None:
    """When no portfolio caps bind, `optimize_inventory_portfolio` preserves all 30 base columns."""
    preds_df, oof_df = _make_multi_sku_predictions_and_oof()

    base_df = compute_inventory_policy(
        preds_df,
        oof_df=oof_df,
        lead_time=7.0,
        service_level=0.95,
        holding_cost=1.0,
        stockout_cost=9.0,
    )
    opt_df = optimize_inventory_portfolio(
        preds_df,
        oof_df=oof_df,
        lead_time=7.0,
        service_level=0.95,
        holding_cost=1.0,
        stockout_cost=9.0,
        uncertainty_source="analytical",
    )

    assert list(opt_df.columns[: len(INVENTORY_POLICY_COLUMNS)]) == list(INVENTORY_POLICY_COLUMNS)
    assert list(opt_df.columns) == list(INVENTORY_OPTIMIZER_COLUMNS)
    np.testing.assert_allclose(
        opt_df["recommended_order_qty"].to_numpy(dtype=float),
        base_df["recommended_order_qty"].to_numpy(dtype=float),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        opt_df["safety_stock"].to_numpy(dtype=float),
        base_df["safety_stock"].to_numpy(dtype=float),
        rtol=1e-6,
        atol=1e-6,
    )
    assert (opt_df["shadow_price_budget"] == 0.0).all()
    assert (opt_df["shadow_price_volume"] == 0.0).all()
    assert (opt_df["shadow_price_units"] == 0.0).all()
    assert (opt_df["binding_constraint"] == "none").all()
    np.testing.assert_allclose(opt_df["allocation_ratio"].to_numpy(dtype=float), 1.0, atol=1e-6)

    # Also verify when max_budget and max_volume are slack (> unconstrained spend/volume)
    slack_df = optimize_inventory_df(
        preds_df,
        oof_df=oof_df,
        max_budget=1e7,
        max_volume=1e7,
        uncertainty_source="analytical",
    )
    assert (slack_df["shadow_price_budget"] == 0.0).all()
    assert (slack_df["shadow_price_volume"] == 0.0).all()
    assert slack_df.attrs["portfolio_summary"]["binding_constraint"] == "none"


def test_working_capital_budget_constraint_and_kkt_shadow_price() -> None:
    """Binding `max_budget` yields `shadow_price_budget > 0` and prioritizes high-value SKUs."""
    preds_df, oof_df = _make_multi_sku_predictions_and_oof()

    unit_costs = {"SKU_HIGH_PRIORITY": 5.0, "SKU_MEDIUM": 10.0, "SKU_LOW_PRIORITY": 25.0}
    stockout_costs = {"SKU_HIGH_PRIORITY": 25.0, "SKU_MEDIUM": 10.0, "SKU_LOW_PRIORITY": 4.0}
    service_levels = {"SKU_HIGH_PRIORITY": 0.98, "SKU_MEDIUM": 0.95, "SKU_LOW_PRIORITY": 0.90}

    unconstrained = optimize_inventory_portfolio(
        preds_df,
        oof_df=oof_df,
        unit_cost=unit_costs,
        stockout_cost=stockout_costs,
        service_level=service_levels,
        on_hand=300.0,
        uncertainty_source="analytical",
    )
    unconstrained_spend = float(unconstrained["order_spend"].sum())
    assert unconstrained_spend > 0.0

    # Set binding budget at 70% of unconstrained spend
    target_budget = 0.70 * unconstrained_spend
    constrained = optimize_inventory_portfolio(
        preds_df,
        oof_df=oof_df,
        max_budget=target_budget,
        unit_cost=unit_costs,
        stockout_cost=stockout_costs,
        service_level=service_levels,
        on_hand=300.0,
        uncertainty_source="analytical",
    )

    total_spend = float(constrained["order_spend"].sum())
    assert total_spend <= target_budget + 1e-6
    assert abs(total_spend - target_budget) <= 1e-4
    assert float(constrained["shadow_price_budget"].iloc[0]) > 0.0
    assert float(constrained["shadow_price_volume"].iloc[0]) == 0.0
    assert (constrained["binding_constraint"] == "budget").all()

    by_sku = constrained.set_index("ts_id")
    # High-priority / low-unit-cost SKU must retain a strictly higher allocation ratio
    assert float(by_sku.loc["SKU_HIGH_PRIORITY", "allocation_ratio"]) > float(
        by_sku.loc["SKU_LOW_PRIORITY", "allocation_ratio"]
    )


def test_warehouse_volume_and_joint_budget_volume_constraints() -> None:
    """Verify `max_volume`, `max_total_units`, and joint `max_budget` + `max_volume` duals."""
    preds_df, oof_df = _make_multi_sku_predictions_and_oof()
    unit_costs = {"SKU_HIGH_PRIORITY": 20.0, "SKU_MEDIUM": 10.0, "SKU_LOW_PRIORITY": 5.0}
    unit_volumes = {"SKU_HIGH_PRIORITY": 0.5, "SKU_MEDIUM": 2.0, "SKU_LOW_PRIORITY": 6.0}

    base = optimize_inventory_portfolio(
        preds_df,
        oof_df=oof_df,
        unit_cost=unit_costs,
        unit_volume=unit_volumes,
        uncertainty_source="analytical",
    )
    unc_vol = float(base["order_volume"].sum())
    unc_units = float(base["recommended_order_qty"].sum())

    # 1. Volume-only binding
    vol_cap = 0.65 * unc_vol
    vol_df = optimize_inventory_portfolio(
        preds_df,
        oof_df=oof_df,
        max_volume=vol_cap,
        unit_cost=unit_costs,
        unit_volume=unit_volumes,
        uncertainty_source="analytical",
    )
    assert float(vol_df["order_volume"].sum()) <= vol_cap + 1e-6
    assert float(vol_df["shadow_price_volume"].iloc[0]) > 0.0
    assert float(vol_df["shadow_price_budget"].iloc[0]) == 0.0
    assert (vol_df["binding_constraint"] == "volume").all()

    # 2. Units-only binding
    unit_cap = 0.60 * unc_units
    unit_df = optimize_inventory_portfolio(
        preds_df,
        oof_df=oof_df,
        max_total_units=unit_cap,
        unit_cost=unit_costs,
        unit_volume=unit_volumes,
        uncertainty_source="analytical",
    )
    assert float(unit_df["recommended_order_qty"].sum()) <= unit_cap + 1e-6
    assert float(unit_df["shadow_price_units"].iloc[0]) > 0.0

    # 3. Joint budget + volume binding
    spend_at_vol_cap = float(vol_df["order_spend"].sum())
    joint_budget_cap = 0.85 * spend_at_vol_cap
    joint_df = optimize_inventory_portfolio(
        preds_df,
        oof_df=oof_df,
        max_budget=joint_budget_cap,
        max_volume=vol_cap,
        unit_cost=unit_costs,
        unit_volume=unit_volumes,
        uncertainty_source="analytical",
    )
    assert float(joint_df["order_spend"].sum()) <= joint_budget_cap + 1e-6
    assert float(joint_df["order_volume"].sum()) <= vol_cap + 1e-6
    assert float(joint_df["shadow_price_budget"].iloc[0]) > 0.0


def test_highs_milp_discrete_vendor_moq_and_order_multiple() -> None:
    """Stage 2 `HiGHS` MILP satisfies `vendor_moq`, `order_multiple`, and caps to `< 1e-6`."""
    preds_df, oof_df = _make_multi_sku_predictions_and_oof()

    moq_map = {"SKU_HIGH_PRIORITY": 200.0, "SKU_MEDIUM": 300.0, "SKU_LOW_PRIORITY": 400.0}
    mult_map = {"SKU_HIGH_PRIORITY": 25.0, "SKU_MEDIUM": 50.0, "SKU_LOW_PRIORITY": 50.0}
    cost_map = {"SKU_HIGH_PRIORITY": 10.0, "SKU_MEDIUM": 10.0, "SKU_LOW_PRIORITY": 10.0}
    vol_map = {"SKU_HIGH_PRIORITY": 1.0, "SKU_MEDIUM": 1.5, "SKU_LOW_PRIORITY": 2.0}

    # Budget allows ordering SKU_HIGH_PRIORITY and SKU_MEDIUM, but not all three at full demand
    budget_limit = 12_500.0
    volume_limit = 1_600.0

    milp_df = optimize_inventory_portfolio(
        preds_df,
        oof_df=oof_df,
        max_budget=budget_limit,
        max_volume=volume_limit,
        unit_cost=cost_map,
        unit_volume=vol_map,
        vendor_moq=moq_map,
        order_multiple=mult_map,
        uncertainty_source="analytical",
    )

    assert float(milp_df["order_spend"].sum()) <= budget_limit + 1e-6
    assert float(milp_df["order_volume"].sum()) <= volume_limit + 1e-6

    for row in milp_df.itertuples(index=False):
        q = float(row.recommended_order_qty)
        tid = str(row.ts_id)
        moq = moq_map[tid]
        mult = mult_map[tid]
        if q > 1e-9:
            assert q >= moq - 1e-6, f"SKU {tid} order {q} violated vendor_moq={moq}"
            rem = q % mult
            assert min(rem, mult - rem) < 1e-6, f"SKU {tid} order {q} not a multiple of {mult}"


def test_copula_uncertainty_source_and_target_portfolio_fill_rate() -> None:
    """`uncertainty_source='copula'` uses copula draws and supports `target_portfolio_fill_rate`."""
    preds_df, oof_df = _make_multi_sku_predictions_and_oof()

    cop_df = optimize_inventory_portfolio(
        preds_df,
        oof_df=oof_df,
        uncertainty_source="copula",
        copula_type="student_t",
        n_paths=300,
        seed=42,
    )
    assert (cop_df["uncertainty_source"] == "copula").all()
    assert (cop_df["recommended_order_qty"] > 0.0).all()
    assert cop_df.attrs["portfolio_summary"]["uncertainty_source"] == "copula"

    # Target portfolio fill rate
    fill_df = optimize_inventory_portfolio(
        preds_df,
        oof_df=oof_df,
        target_portfolio_fill_rate=0.97,
        on_hand=0.0,
        uncertainty_source="analytical",
    )
    assert abs(float(fill_df["portfolio_fill_rate"].iloc[0]) - 0.97) < 0.015

    # Infeasible target fill rate under tiny max_budget with strict_feasibility=True raises error
    with pytest.raises((GoalSeekInfeasibleError, ConfigError, ValueError)):
        optimize_inventory_portfolio(
            preds_df,
            oof_df=oof_df,
            target_portfolio_fill_rate=0.99,
            max_budget=10.0,
            strict_feasibility=True,
            uncertainty_source="analytical",
        )


def test_compute_and_plot_inventory_frontier_and_validation() -> None:
    """Verify `compute_inventory_frontier`, `plot_inventory_frontier`, and parameter validation."""
    preds_df, oof_df = _make_multi_sku_predictions_and_oof()

    frontier_df = compute_inventory_frontier(
        preds_df,
        oof_df=oof_df,
        n_points=5,
        uncertainty_source="analytical",
    )
    assert len(frontier_df) == 5
    # Fill rate should be non-decreasing as budget increases
    fills = frontier_df["portfolio_fill_rate"].to_numpy(dtype=float)
    assert np.all(np.diff(fills) >= -1e-6)

    ax1 = plot_inventory_frontier(frontier_df)
    assert ax1 is not None

    pol_df = optimize_inventory_portfolio(
        preds_df,
        oof_df=oof_df,
        max_budget=8_000.0,
        uncertainty_source="analytical",
    )
    ax2 = plot_inventory_frontier(pol_df)
    assert ax2 is not None

    empty_df = optimize_inventory_portfolio(pd.DataFrame())
    assert empty_df.empty
    assert list(empty_df.columns) == list(INVENTORY_OPTIMIZER_COLUMNS)

    with pytest.raises((ConfigError, ValueError)):
        optimize_inventory_portfolio(preds_df, max_budget=-100.0)
    with pytest.raises((ConfigError, ValueError)):
        optimize_inventory_portfolio(preds_df, min_service_level=1.2)
    with pytest.raises((ConfigError, ValueError)):
        optimize_inventory_portfolio(preds_df, unit_cost=-5.0)
