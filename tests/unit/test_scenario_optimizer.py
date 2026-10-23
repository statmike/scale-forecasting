"""Unit tests for `scale_forecasting.scenario_optimizer` (Track L — F6 / R1.5)."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

from scale_forecasting.elasticity import make_confounded_promo_dgp
from scale_forecasting.errors import ConfigError, GoalSeekInfeasibleError
from scale_forecasting.reconciliation import TOTAL_NODE_ID, HierarchySpec
from scale_forecasting.scenario_optimizer import (
    SCENARIO_OPTIMIZATION_COLUMNS,
    SCENARIO_OPTIMIZATION_SUMMARY_COLUMNS,
    ScenarioOptimizationResult,
    optimize_scenario,
    plot_optimal_scenario,
)

matplotlib.use("Agg")


def _make_hierarchical_promo_panel() -> tuple[pd.DataFrame, pd.DataFrame, HierarchySpec]:
    """Create a 2-SKU + `__total__` hierarchical panel with historical price/promo and forecasts."""
    hist_df, _ = make_confounded_promo_dgp(n_days=120, n_series=2, seed=42)
    hist_df["promo"] = hist_df["promo_flag"]
    # Add realistic varying prices negatively correlated with promo
    rng = np.random.default_rng(42)
    hist_df["price"] = np.where(
        hist_df["promo"] > 0.5,
        8.50 + rng.uniform(-0.25, 0.25, size=len(hist_df)),
        10.00 + rng.uniform(-0.40, 0.40, size=len(hist_df)),
    )

    bottom_ids = sorted(str(x) for x in hist_df["ts_id"].unique())
    horizon = 14
    fut_dates = pd.date_range("2026-05-01", periods=horizon, freq="D")

    pred_rows: list[dict[str, object]] = []
    for bid in bottom_ids:
        for h, dt in enumerate(fut_dates):
            yh = 100.0 + 10.0 * np.sin(2.0 * np.pi * h / 7.0)
            pred_rows.append(
                {
                    "ts_id": bid,
                    "model_type": "lightgbm@monotone",
                    "ds": dt,
                    "forecast_date": dt,
                    "yhat": yh,
                    "yhat_lower": yh - 15.0,
                    "yhat_upper": yh + 15.0,
                    "price": 10.0,
                    "promo": 0.0,
                }
            )
    # Add __total__ aggregate rows
    for h, dt in enumerate(fut_dates):
        yh_tot = 2.0 * (100.0 + 10.0 * np.sin(2.0 * np.pi * h / 7.0))
        pred_rows.append(
            {
                "ts_id": TOTAL_NODE_ID,
                "model_type": "lightgbm@monotone",
                "ds": dt,
                "forecast_date": dt,
                "yhat": yh_tot,
                "yhat_lower": yh_tot - 22.0,
                "yhat_upper": yh_tot + 22.0,
                "price": 10.0,
                "promo": 0.0,
            }
        )

    s_mat = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    hier_spec = HierarchySpec(
        node_ids=(TOTAL_NODE_ID, *bottom_ids),
        bottom_ids=tuple(bottom_ids),
        level_nodes={TOTAL_NODE_ID: (TOTAL_NODE_ID,), "__bottom__": tuple(bottom_ids)},
        summing_matrix=s_mat,
    )
    return hist_df, pd.DataFrame.from_records(pred_rows), hier_spec


def test_optimize_scenario_gross_profit_improves_over_status_quo() -> None:
    """`optimize_scenario` strictly improves objective value and delegates DataFrame methods."""
    hist_df, preds_df, hier_spec = _make_hierarchical_promo_panel()

    res = optimize_scenario(
        preds_df,
        history_df=hist_df,
        hier_spec=hier_spec,
        objective="gross_profit",
        unit_cost=4.50,
        price_bounds=(-0.15, 0.20),
        max_promo_days=3,
        min_cooldown_days=2,
        reconcile=True,
        include_inventory_impact=True,
    )

    assert isinstance(res, ScenarioOptimizationResult)
    assert res.feasibility_status == "optimal"
    assert res.optimal_objective_value > res.baseline_objective_value
    assert res.optimal_total_gross_margin > res.baseline_total_gross_margin
    assert res.objective_improvement > 0.0
    assert list(res.schedule_df.columns) == list(SCENARIO_OPTIMIZATION_COLUMNS)
    assert list(res.summary_df.columns) == list(SCENARIO_OPTIMIZATION_SUMMARY_COLUMNS)

    # DataFrame-like delegation on ScenarioOptimizationResult
    assert len(res) == len(res.schedule_df)
    assert res.dataframe is res.schedule_df
    assert "optimal_price" in res.columns
    assert len(res["optimal_price"]) == len(res.schedule_df)
    assert len(res.head(3)) == 3
    summary_dict = res.to_dict()
    assert summary_dict["optimal_objective_value"] == pytest.approx(res.optimal_objective_value)
    assert "ScenarioOptimizationResult" in res.summary_text()
    assert "optimal_scenario" in res.recommended_shocks


def test_promo_frequency_cap_cooldown_spacing_and_pull_forward_cannibalization() -> None:
    """Verify `max_promo_days`, `min_cooldown_days`, `pull_forward_dip`, and cannibalization."""
    hist_df, preds_df, hier_spec = _make_hierarchical_promo_panel()

    res = optimize_scenario(
        preds_df,
        history_df=hist_df,
        hier_spec=hier_spec,
        objective="units",
        unit_cost=4.50,
        price_bounds=(-0.10, 0.10),
        max_promo_days=3,
        min_cooldown_days=3,
    )

    for bid in hier_spec.bottom_ids:
        sub = res.schedule_df[res.schedule_df["ts_id"] == bid].sort_values("horizon_step")
        promos = (sub["optimal_promo"].to_numpy(dtype=float) >= 0.5).astype(int)
        # 1. Frequency cap: at most 3 promo days
        assert int(np.sum(promos)) <= 3
        assert int(np.sum(promos)) >= 1
        # 2. Cooldown spacing: any two active promo steps must be separated by > 3 steps
        active_steps = np.where(promos == 1)[0]
        if len(active_steps) >= 2:
            assert np.all(np.diff(active_steps) >= 4)
        # 3. Post-promo pull-forward dip is non-positive and strictly negative after promo pulses
        pf_dip = sub["pull_forward_dip"].to_numpy(dtype=float)
        assert np.all(pf_dip <= 1e-9)
        if active_steps[0] + 1 < len(pf_dip):
            assert pf_dip[active_steps[0] + 1] < 0.0
        # 4. Cross-SKU cannibalization is non-positive
        cannib = sub["cannibalization_effect"].to_numpy(dtype=float)
        assert np.all(cannib <= 1e-9)


def test_monotone_price_response_and_discrete_price_grid() -> None:
    """Verify `@monotone` downward-sloping demand on price changes and discrete `price_grid`."""
    hist_df, preds_df, _ = _make_hierarchical_promo_panel()
    leaf_preds = preds_df[preds_df["ts_id"] != TOTAL_NODE_ID].copy()

    grid = [-0.10, -0.05, 0.0, 0.05, 0.12]
    res = optimize_scenario(
        leaf_preds,
        history_df=hist_df,
        objective="gross_profit",
        unit_cost=4.50,
        price_grid=grid,
        max_promo_days=0,
    )

    rel_changes = res.schedule_df["price_change_pct"].to_numpy(dtype=float) / 100.0
    for rc in rel_changes:
        assert min(abs(rc - g) for g in grid) < 1e-6

    # Monotone check: wherever price increased (`price_change_pct > 0`), `price_lift < 0`
    price_lifts = res.schedule_df["price_lift"].to_numpy(dtype=float)
    pos_mask = rel_changes > 1e-6
    neg_mask = rel_changes < -1e-6
    if pos_mask.any():
        assert np.all(price_lifts[pos_mask] < 0.0)
    if neg_mask.any():
        assert np.all(price_lifts[neg_mask] > 0.0)


def test_guardrails_margin_floor_unit_caps_and_risk_adjusted_profit() -> None:
    """Verify `min_margin_pct`, `min_total_units`, `max_total_units`, and `risk_adjusted_profit`."""
    hist_df, preds_df, _ = _make_hierarchical_promo_panel()
    leaf_preds = preds_df[preds_df["ts_id"] != TOTAL_NODE_ID].copy()

    res_constrained = optimize_scenario(
        leaf_preds,
        history_df=hist_df,
        objective="units",
        unit_cost=6.00,
        price_bounds=(-0.20, 0.20),
        min_margin_pct=0.38,
        max_total_units=3100.0,
        max_promo_days=4,
    )
    tot_gm = res_constrained.optimal_total_gross_margin
    tot_rev = res_constrained.optimal_total_revenue
    assert (tot_gm / tot_rev) >= 0.38 - 1e-4
    assert res_constrained.optimal_total_units <= 3100.0 + 1e-4

    # Risk-adjusted profit (`CVaR`) vs risk-neutral profit
    res_neutral = optimize_scenario(
        leaf_preds,
        history_df=hist_df,
        objective="gross_profit",
        unit_cost=4.00,
        price_bounds=(-0.15, 0.30),
        risk_aversion_lambda=0.0,
    )
    res_risk_averse = optimize_scenario(
        leaf_preds,
        history_df=hist_df,
        objective="risk_adjusted_profit",
        unit_cost=4.00,
        price_bounds=(-0.15, 0.30),
        risk_aversion_lambda=0.80,
        cvar_alpha=0.95,
    )
    assert res_risk_averse.objective == "risk_adjusted_profit"
    assert np.isfinite(res_risk_averse.cvar_profit)
    # High risk aversion avoids extreme price hikes that inflate downside variance
    assert (
        float(res_risk_averse.schedule_df["optimal_price"].mean())
        <= float(res_neutral.schedule_df["optimal_price"].mean()) + 1e-6
    )


def test_hierarchy_coherence_and_inventory_deltas_and_history_only_input() -> None:
    """Verify `< 1e-6` hierarchical coherence, inventory deltas, and history-only input."""
    hist_df, preds_df, hier_spec = _make_hierarchical_promo_panel()

    res = optimize_scenario(
        preds_df,
        history_df=hist_df,
        hier_spec=hier_spec,
        objective="gross_profit",
        unit_cost=4.50,
        reconcile=True,
        include_inventory_impact=True,
    )
    assert res.coherence_max_abs_error < 1e-6
    assert TOTAL_NODE_ID in set(res.schedule_df["ts_id"])
    assert not res.inventory_delta_df.empty
    assert "delta_reorder_point" in res.inventory_delta_df.columns
    assert "delta_safety_stock" in res.inventory_delta_df.columns
    assert "delta_order_qty" in res.inventory_delta_df.columns

    # Passing ONLY historical DataFrame (`predictions_df=None`) works with zero-config defaults
    res_hist_only = optimize_scenario(hist_df, unit_cost=4.50, horizon=10)
    assert not res_hist_only.schedule_df.empty
    assert res_hist_only.optimal_objective_value >= res_hist_only.baseline_objective_value - 1e-6

    ax = plot_optimal_scenario(res)
    assert ax is not None

    with pytest.raises((GoalSeekInfeasibleError, ConfigError, ValueError)):
        optimize_scenario(
            leaf_df := preds_df[preds_df["ts_id"] != TOTAL_NODE_ID],
            min_total_units=5000.0,
            max_total_units=1000.0,
        )
    with pytest.raises((ConfigError, ValueError)):
        optimize_scenario(leaf_df, objective="invalid_obj")  # type: ignore[arg-type]


def test_scenario_optimizer_similarity_weighted_cross_sku_cannibalization() -> None:
    """Verify F8 (`use_similarity_priors=True` and explicit `similarity_matrix` in
    `optimize_scenario`): weights off-diagonal cross-SKU cannibalization by normalized positive
    trajectory similarity.
    """
    from scale_forecasting.scenario_optimizer import _estimate_sku_structural_params

    hist_df, _ = make_confounded_promo_dgp(n_days=90, n_series=3, seed=42)
    hist_df["promo"] = hist_df["promo_flag"]
    hist_df["price"] = np.where(hist_df["promo"] > 0.5, 8.5, 10.0)
    sku_ids = sorted(str(x) for x in hist_df["ts_id"].unique())

    sim_df = pd.DataFrame(
        [
            [1.0, 0.95, 0.05],
            [0.95, 1.0, 0.05],
            [0.05, 0.05, 1.0],
        ],
        index=sku_ids,
        columns=sku_ids,
    )
    _, _, _, cannib_sim = _estimate_sku_structural_params(
        bottom_ids=sku_ids,
        horizon=7,
        base_yhat=np.full((3, 7), 100.0),
        base_price=np.full((3, 7), 10.0),
        base_promo=np.zeros((3, 7)),
        history_df=hist_df,
        ts_id_col="ts_id",
        date_col="ds",
        target_col="y",
        promo_col="promo",
        price_col="price",
        pull_forward_lags=2,
        cannibalization_weight=1.0,
        use_similarity_priors=True,
        similarity_matrix=sim_df,
    )
    _, _, _, cannib_unif = _estimate_sku_structural_params(
        bottom_ids=sku_ids,
        horizon=7,
        base_yhat=np.full((3, 7), 100.0),
        base_price=np.full((3, 7), 10.0),
        base_promo=np.zeros((3, 7)),
        history_df=hist_df,
        ts_id_col="ts_id",
        date_col="ds",
        target_col="y",
        promo_col="promo",
        price_col="price",
        pull_forward_lags=2,
        cannibalization_weight=1.0,
        use_similarity_priors=False,
    )
    # Under similarity priors, twin SKUs (0, 1) exert ~19x stronger cannibalization than SKU 2
    assert abs(float(cannib_sim[0, 1])) > abs(float(cannib_sim[0, 2])) * 10.0
    assert abs(float(cannib_unif[0, 1]) - float(cannib_unif[0, 2])) < 1e-9

    res = optimize_scenario(
        hist_df,
        horizon=7,
        unit_cost=4.5,
        max_promo_days=2,
        use_similarity_priors=True,
        similarity_matrix=sim_df,
    )
    assert isinstance(res, ScenarioOptimizationResult)
    assert res.feasibility_status == "optimal"
    assert res.optimal_objective_value >= res.baseline_objective_value - 1e-6
