"""Unit tests for Track J Capability 4: Counterfactual Scenario Simulator (`scenarios.py`)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scale_forecasting.config import RunConfig
from scale_forecasting.scenarios import (
    apply_scenario_shocks,
    plot_scenarios,
    simulate_scenarios,
    summarize_scenarios,
)
from scale_forecasting.sdk import Forecaster


def _make_hierarchical_promo_panel(n_days: int = 90) -> pd.DataFrame:
    """Create a 2-region, 4-store hierarchical daily retail panel with price and promo."""
    rng = np.random.default_rng(2026)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    stores = [
        ("east", "east_s1"),
        ("east", "east_s2"),
        ("west", "west_s1"),
        ("west", "west_s2"),
    ]
    rows: list[dict[str, object]] = []
    for reg, store in stores:
        base_vol = 80.0 if reg == "east" else 60.0
        for t, d in enumerate(dates):
            promo = 1.0 if (t % 14 in (5, 6)) else 0.0
            price = 1.0 - 0.15 * promo + rng.normal(0.0, 0.01)
            seasonal = 10.0 * np.sin(2.0 * np.pi * (t % 7) / 7.0)
            y = base_vol + seasonal - 35.0 * (price - 1.0) + 25.0 * promo + rng.normal(0.0, 1.5)
            rows.append(
                {
                    "ts_id": store,
                    "region_id": reg,
                    "store_id": store,
                    "ds": d,
                    "y": float(max(5.0, y)),
                    "price_index": float(price),
                    "promo_flag": float(promo),
                }
            )
    return pd.DataFrame(rows)


def test_apply_scenario_shocks_operators() -> None:
    df = pd.DataFrame({"price_index": [1.0, 2.0], "promo_flag": [0.0, 0.0]})
    shocked = apply_scenario_shocks(df, {"price_index": "*0.85", "promo_flag": "+1"})
    assert np.allclose(shocked["price_index"], [0.85, 1.70])
    assert np.allclose(shocked["promo_flag"], [1.0, 1.0])


def test_full_chain_scenario_propagation_ensembles_hierarchy_margin_and_inventory() -> None:
    """Acceptance Criterion 4: `simulate_scenarios` propagates counterfactual shocks through:
    (1) base models (respecting `@<tag>` variants & `monotonic_constraints`),
    (2) calculated & learned ensembles (`ensemble_inverse_error`, `ensemble_nnls`),
    (3) hierarchy reconciliation (`mint_nonneg`) preserving cross-level additive coherence,
    (4) revenue & gross-margin delta (`unit_cost`, `gross_margin_delta`), and
    (5) operational inventory policy shifts (`delta_reorder_point`, `delta_safety_stock`,
        `delta_order_qty`).
    """
    panel = _make_hierarchical_promo_panel(n_days=84)
    cfg = RunConfig.model_validate(
        {
            "run_name": "full-chain-what-if",
            "data": {"source_table": "playground", "freq": "D", "horizon": 7},
            "models": ["regression_lags@monotone", "xgboost@monotone"],
            "features": {
                "future_covariates": ["price_index", "promo_flag"],
                "monotonic_constraints": {"price_index": -1, "promo_flag": 1},
            },
            "backtest": {"enabled": True, "n_folds": 2, "horizon": 7, "step": 7},
            "ensemble": {"enabled": True, "strategies": ["inverse_error", "nnls"]},
            "hierarchy": {
                "enabled": True,
                "levels": [["region_id"], ["region_id", "store_id"]],
                "reconciliation_methods": ["mint_nonneg"],
            },
        }
    )

    scenarios = {
        "promo_push": {"price_index": "*0.85", "promo_flag": 1.0},
        "price_hike": {"price_index": "*1.15", "promo_flag": 0.0},
    }
    sim_df = simulate_scenarios(
        panel,
        cfg,
        scenarios,
        price_col="price_index",
        unit_cost=0.55,
        reconcile=True,
        include_inventory_impact=True,
        lead_time=7.0,
        service_level=0.95,
    )
    assert not sim_df.empty

    # 1 & 2. Both base models and ensemble strategies are present in simulated output
    models_out = set(sim_df["model_type"].unique())
    assert "regression_lags@monotone" in models_out
    assert "xgboost@monotone" in models_out
    assert "ensemble_inverse_error" in models_out
    assert "ensemble_nnls" in models_out

    # 3. Hierarchy reconciliation: root node ('__total__') equals sum of leaf stores at every step
    leaf_ids = {"east_s1", "east_s2", "west_s1", "west_s2"}
    assert "__total__" in set(sim_df["ts_id"].unique())
    for (scen, mtype, fdate), grp in sim_df.groupby(["scenario", "model_type", "ds"], sort=False):
        root_row = grp[grp["ts_id"] == "__total__"]
        leaf_rows = grp[grp["ts_id"].isin(leaf_ids)]
        if not root_row.empty and len(leaf_rows) == 4:
            root_val = float(root_row["yhat"].iloc[0])
            leaf_sum = float(leaf_rows["yhat"].sum())
            assert abs(root_val - leaf_sum) < 1e-4, (
                f"Coherence violation for ({scen}, {mtype}, {fdate}): "
                f"root={root_val} vs leaf_sum={leaf_sum}"
            )
            assert (leaf_rows["yhat"] >= -1e-9).all()

    # 4. Revenue & Gross-Margin accounting identities
    leaf_sim = sim_df[sim_df["ts_id"].isin(leaf_ids)]
    expected_scen_gm = (leaf_sim["price"] - leaf_sim["unit_cost"]) * leaf_sim["yhat"]
    assert np.allclose(leaf_sim["scenario_gross_margin"], expected_scen_gm, atol=1e-6)
    assert np.allclose(
        leaf_sim["gross_margin_delta"],
        leaf_sim["scenario_gross_margin"] - leaf_sim["baseline_gross_margin"],
        atol=1e-6,
    )

    # 5. Operational inventory policy shifts (ROP, safety stock, order qty)
    for col in (
        "baseline_reorder_point",
        "scenario_reorder_point",
        "delta_reorder_point",
        "baseline_safety_stock",
        "scenario_safety_stock",
        "delta_safety_stock",
        "baseline_order_qty",
        "scenario_order_qty",
        "delta_order_qty",
    ):
        assert col in sim_df.columns
        assert sim_df[col].notna().all()

    promo_sub = leaf_sim[leaf_sim["scenario"] == "promo_push"]
    hike_sub = leaf_sim[leaf_sim["scenario"] == "price_hike"]
    assert float(promo_sub["delta_reorder_point"].mean()) > 0.0
    assert float(hike_sub["delta_reorder_point"].mean()) <= 0.0

    # Verify summarize_scenarios rolls up gross margin & inventory columns
    summary_df = summarize_scenarios(sim_df)
    assert not summary_df.empty
    for col in (
        "gross_margin_delta",
        "gross_margin_delta_pct",
        "delta_reorder_point",
        "delta_safety_stock",
        "delta_order_qty",
    ):
        assert col in summary_df.columns


def test_forecaster_auto_session_simulate_scenarios_and_plot() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panel = _make_hierarchical_promo_panel(n_days=70)
    f = Forecaster.auto(
        panel,
        tier="first_light",
        where="local",
        horizon=7,
        models=["regression_lags@monotone", "xgboost@monotone"],
        autofocus=False,
    )
    sim_df = f.simulate_scenarios(
        {"promo_boost": {"price_index": "*0.90", "promo_flag": 1.0}},
        price_col="price_index",
        unit_cost=0.50,
        include_inventory_impact=True,
    )
    assert not sim_df.empty
    assert "gross_margin_delta" in sim_df.columns
    assert "delta_reorder_point" in sim_df.columns

    ax = plot_scenarios(sim_df, show_revenue=True)
    assert ax is not None
    plt.close("all")


def test_hierarchy_financial_coherence_and_unit_cost_column_invariance() -> None:
    """C4-1 & C4-2 regression:
    (1) Upper hierarchy nodes (`__total__`, `cat=all`) equal the exact sum of bottom-leaf
        revenues and gross margins even under heterogeneous leaf prices and costs.
    (2) Passing `unit_cost` as a column name (`unit_cost="cost_col"`) does not inject `cost_col`
        into `future_covariates` or shift unit demand forecasts (`yhat`).
    """
    rng = np.random.default_rng(123)
    rows: list[dict[str, object]] = []
    for t_i, d in enumerate(pd.date_range("2025-01-01", periods=60, freq="D")):
        pr = float((t_i % 10) == 0)
        rows.append(
            {
                "ts_id": "s_cheap",
                "cat": "all",
                "ds": d,
                "y": 200.0 + 45.0 * pr,
                "price": 10.0 * (0.85 if pr else 1.0),
                "promo": pr,
                "cost_col": 9.0 + float(rng.normal(0.0, 0.5)),
            }
        )
        rows.append(
            {
                "ts_id": "s_luxury",
                "cat": "all",
                "ds": d,
                "y": 20.0 + 5.0 * pr,
                "price": 100.0 * (0.85 if pr else 1.0),
                "promo": pr,
                "cost_col": 20.0 + float(rng.normal(0.0, 0.5)),
            }
        )
    panel = pd.DataFrame(rows)
    cfg = RunConfig.model_validate(
        {
            "run_name": "verify-c4",
            "data": {
                "source_table": "local.c4",
                "ts_id_col": "ts_id",
                "date_col": "ds",
                "target_col": "y",
                "freq": "D",
                "horizon": 7,
            },
            "models": ["regression_lags@econ"],
            "features": {
                "future_covariates": ["price", "promo"],
                "monotonic_constraints": {"price": -1, "promo": 1},
            },
            "hierarchy": {
                "enabled": True,
                "levels": [["cat"]],
                "reconciliation_methods": ["mint_nonneg"],
            },
        }
    )
    s_dict = simulate_scenarios(
        panel,
        cfg,
        {"cut": {"price": "*0.85", "promo": 1.0}},
        price_col="price",
        unit_cost={"s_cheap": 9.0, "s_luxury": 20.0},
        reconcile=True,
    )
    s_col = simulate_scenarios(
        panel,
        cfg,
        {"cut": {"price": "*0.85", "promo": 1.0}},
        price_col="price",
        unit_cost="cost_col",
        reconcile=True,
    )

    for scen in ("baseline", "cut"):
        tot_rev = float(
            s_dict[(s_dict["scenario"] == scen) & (s_dict["ts_id"] == "__total__")][
                "scenario_revenue"
            ].sum()
        )
        leaf_rev = float(
            s_dict[(s_dict["scenario"] == scen) & (s_dict["ts_id"].isin(["s_cheap", "s_luxury"]))][
                "scenario_revenue"
            ].sum()
        )
        tot_gm = float(
            s_dict[(s_dict["scenario"] == scen) & (s_dict["ts_id"] == "__total__")][
                "scenario_gross_margin"
            ].sum()
        )
        leaf_gm = float(
            s_dict[(s_dict["scenario"] == scen) & (s_dict["ts_id"].isin(["s_cheap", "s_luxury"]))][
                "scenario_gross_margin"
            ].sum()
        )
        assert abs(tot_rev - leaf_rev) < 1e-6
        assert abs(tot_gm - leaf_gm) < 1e-6

    assert float((s_dict["yhat"] - s_col["yhat"]).abs().max()) == 0.0
