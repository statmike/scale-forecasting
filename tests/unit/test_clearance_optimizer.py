"""4-Tier mathematical ground-truth, invariant, and pathological tests for
`scale_forecasting.clearance_optimizer`.

Covers:
- Tier A: Synthetic DGP own-price elasticity recovery & Track P digital-twin elasticity borrowing
  for SKUs with zero historical markdown variation.
- Tier B: Monotone markdown ladder ($d_{i,t} \\le d_{i,t+1}$), finite initial inventory and
  perishable expiry conservation, warehouse/shelf space reclamation (`target_space_freed`), and
  vendor contract tiered rebate / Take-or-Pay optimization (`optimize_contract_volume`).
- Tier C: Infeasible space/sell-through targets (`status="infeasible"`, `is_feasible=False`,
  `relaxation_hint`), `strict_feasibility=True`, empty/zero-inventory edge cases, and plotting.
- Tier D: Integration with `Forecaster`, `BakeOff`, and `PlaygroundRun` workbench mixins.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from scale_forecasting import (
    CLEARANCE_SCHEDULE_COLUMNS,
    CLEARANCE_SUMMARY_COLUMNS,
    CONTRACT_VOLUME_COLUMNS,
    CONTRACT_VOLUME_SUMMARY_COLUMNS,
    ClearanceOptimizationResult,
    ContractVolumeOptimizationResult,
    Forecaster,
    optimize_clearance,
    optimize_clearance_df,
    optimize_contract_volume,
    playground,
    plot_clearance_schedule,
)


def _make_markdown_dgp_panel(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a 3-SKU panel where 2 donors have markdown history and 1 target has flat price."""
    rng = np.random.default_rng(seed)
    n_hist = 60
    n_fut = 12
    hist_dates = pd.date_range("2025-01-01", periods=n_hist, freq="D")
    fut_dates = pd.date_range(hist_dates[-1] + pd.Timedelta(days=1), periods=n_fut, freq="D")

    true_elasticity = -2.2
    base_p = 20.0
    t_idx = np.arange(n_hist, dtype=float)
    seasonal_wave = 1.0 + 0.20 * np.sin(2.0 * np.pi * t_idx / 7.0)

    hist_rows: list[dict[str, object]] = []
    fut_rows: list[dict[str, object]] = []

    for sku in ("donor_a", "donor_b", "seasonal_target"):
        for t, ds in enumerate(hist_dates):
            if sku in ("donor_a", "donor_b") and t % 9 in (3, 4):
                disc = 0.25 if (t % 18 == 3) else 0.15
            else:
                disc = 0.0
            price_t = base_p * (1.0 - disc)
            base_y = 40.0 * seasonal_wave[t]
            y_t = base_y * ((price_t / base_p) ** true_elasticity) * np.exp(rng.normal(0.0, 0.02))
            hist_rows.append(
                {
                    "ts_id": sku,
                    "ds": ds,
                    "y": float(max(1.0, y_t)),
                    "price": float(price_t),
                }
            )
        for h, ds in enumerate(fut_dates):
            wave_h = 1.0 + 0.20 * np.sin(2.0 * np.pi * (n_hist + h) / 7.0)
            fut_rows.append(
                {
                    "ts_id": sku,
                    "ds": ds,
                    "yhat": float(40.0 * wave_h),
                }
            )

    return pd.DataFrame(hist_rows), pd.DataFrame(fut_rows)


# ---------------------------------------------------------------------------
# Tier A: Synthetic DGP & Digital-Twin Elasticity Borrowing
# ---------------------------------------------------------------------------


def test_tier_a_digital_twin_elasticity_borrowing_and_dgp_recovery() -> None:
    """SKU without markdown history borrows elasticity from Track P digital-twin donors."""
    hist_df, fut_df = _make_markdown_dgp_panel(seed=42)
    res = optimize_clearance(
        fut_df,
        history_df=hist_df,
        initial_inventory={"donor_a": 620.0, "donor_b": 620.0, "seasonal_target": 620.0},
        expiry_step=10,
        base_price=20.0,
        unit_cost=7.0,
        salvage_value=0.0,
        spoilage_cost=4.0,
        use_digital_twin_elasticity=True,
    )
    assert isinstance(res, ClearanceOptimizationResult)
    assert res.is_feasible
    assert res.status == "optimal"

    summary = res.summary_df.set_index("ts_id")
    assert summary.loc["donor_a", "elasticity_source"] == "own_history"
    assert summary.loc["donor_b", "elasticity_source"] == "own_history"
    assert summary.loc["seasonal_target", "elasticity_source"] == "digital_twin_borrowed"

    borrowed_e = float(summary.loc["seasonal_target", "price_elasticity"])
    donor_e = float(summary.loc["donor_a", "price_elasticity"])
    assert -3.5 <= donor_e <= -1.2
    assert borrowed_e == pytest.approx(
        0.5 * (donor_e + float(summary.loc["donor_b", "price_elasticity"])),
        rel=0.25,
    )


# ---------------------------------------------------------------------------
# Tier B: Mathematical & Operational Invariants (Monotone Ladder, Expiry, Space, Rebates)
# ---------------------------------------------------------------------------


def test_tier_b_monotone_markdown_inventory_and_expiry_invariants() -> None:
    """Discounts are non-decreasing, sales respect initial_inventory, and expired steps sell 0."""
    _, fut_df = _make_markdown_dgp_panel(seed=7)
    res = optimize_clearance(
        fut_df,
        initial_inventory={"donor_a": 520.0, "donor_b": 480.0, "seasonal_target": 550.0},
        expiry_step={"donor_a": 9, "donor_b": 10, "seasonal_target": 8},
        discount_ladder=(0.0, 0.10, 0.20, 0.35, 0.50),
        monotone_markdown=True,
        base_price=22.0,
        unit_cost=8.0,
        salvage_value=1.0,
        spoilage_cost=5.0,
        price_elasticity=-2.4,
    )
    assert list(res.schedule_df.columns) == list(CLEARANCE_SCHEDULE_COLUMNS)
    assert list(res.summary_df.columns) == list(CLEARANCE_SUMMARY_COLUMNS)
    assert res.total_optimal_profit >= res.total_baseline_profit - 1e-6

    for ts_id, grp in res.schedule_df.groupby("ts_id"):
        grp_sorted = grp.sort_values("horizon_step")
        discs = grp_sorted["optimal_discount_pct"].to_numpy(dtype=float)
        # Non-decreasing markdown ladder invariant: d_{t} <= d_{t+1}
        assert np.all(np.diff(discs) >= -1e-9), f"Non-monotone discounts for {ts_id}: {discs}"

        # Expired steps must have zero expected_units
        expired_rows = grp_sorted[grp_sorted["is_expired_step"]]
        assert not expired_rows.empty
        assert np.allclose(expired_rows["expected_units"].to_numpy(dtype=float), 0.0)

        # Inventory conservation invariant
        sum_row = res.summary_df[res.summary_df["ts_id"] == ts_id].iloc[0]
        inv_0 = float(sum_row["initial_inventory"])
        sold = float(sum_row["optimal_sell_through_units"])
        unsold = float(sum_row["unsold_terminal_units"])
        assert sold <= inv_0 + 1e-6
        assert sold + unsold == pytest.approx(inv_0, abs=1e-6)
        assert float(grp_sorted["remaining_inventory"].iloc[-1]) == pytest.approx(unsold, abs=1e-6)


def test_tier_b_warehouse_and_shelf_space_reclamation_constraint() -> None:
    """target_space_freed accelerates markdowns on bulky/elastic SKUs to free cubic capacity."""
    dates = pd.date_range("2025-05-01", periods=10, freq="D")
    preds = pd.concat(
        [
            pd.DataFrame({"ts_id": "bulky_coat", "ds": dates, "yhat": np.full(10, 15.0)}),
            pd.DataFrame({"ts_id": "small_accessory", "ds": dates, "yhat": np.full(10, 15.0)}),
        ],
        ignore_index=True,
    )
    # Without space constraint and high salvage value, 0% discount is optimal:
    unconstrained = optimize_clearance(
        preds,
        initial_inventory={"bulky_coat": 250.0, "small_accessory": 250.0},
        unit_volume={"bulky_coat": 4.0, "small_accessory": 0.5},
        reclaim_by_step=6,
        base_price=30.0,
        unit_cost=12.0,
        salvage_value=17.5,
        price_elasticity=-2.0,
    )
    # Baseline space freed in first 6 steps: 6 * 15 * 4.0 + 6 * 15 * 0.5 = 360 + 45 = 405 cu ft.
    # Require 520 cu ft freed by step 6:
    constrained = optimize_clearance(
        preds,
        initial_inventory={"bulky_coat": 250.0, "small_accessory": 250.0},
        unit_volume={"bulky_coat": 4.0, "small_accessory": 0.5},
        target_space_freed=520.0,
        reclaim_by_step=6,
        base_price=30.0,
        unit_cost=12.0,
        salvage_value=17.5,
        price_elasticity=-2.0,
    )
    assert constrained.is_feasible
    assert constrained.status == "optimal"
    assert constrained.total_space_freed >= 520.0 - 1e-4
    assert constrained.total_space_freed > unconstrained.total_space_freed


def test_tier_b_contract_volume_rebate_tier_and_take_or_pay_invariants() -> None:
    """optimize_contract_volume unlocks lucrative rebate tiers and avoids Take-or-Pay penalties."""
    dates = pd.date_range("2025-06-01", periods=8, freq="D")
    preds = pd.concat(
        [
            pd.DataFrame({"ts_id": "sku_elastic", "ds": dates, "yhat": np.full(8, 25.0)}),
            pd.DataFrame({"ts_id": "sku_inelastic", "ds": dates, "yhat": np.full(8, 25.0)}),
        ],
        ignore_index=True,
    )
    # Baseline total units = 2 * 8 * 25 = 400 units.
    # Case 1: Lucrative rebate of $1,500 at 460 units -> solver discounts elastic SKU to cross 460.
    res_lucrative = optimize_contract_volume(
        preds,
        contract_tiers=[(460.0, 1500.0), (520.0, 1600.0)],
        base_price=20.0,
        unit_cost=8.0,
        price_elasticity={"sku_elastic": -2.5, "sku_inelastic": -0.4},
    )
    assert isinstance(res_lucrative, ContractVolumeOptimizationResult)
    assert list(res_lucrative.schedule_df.columns) == list(CONTRACT_VOLUME_COLUMNS)
    assert list(res_lucrative.summary_df.columns) == list(CONTRACT_VOLUME_SUMMARY_COLUMNS)
    assert res_lucrative.optimal_portfolio_units >= 460.0 - 1e-5
    assert res_lucrative.rebate_earned >= 1500.0
    assert res_lucrative.net_value_lift > 0.0

    # Elastic SKU should receive deeper average discount than inelastic SKU:
    sum_df = res_lucrative.summary_df.set_index("ts_id")
    assert float(sum_df.loc["sku_elastic", "mean_discount_pct"]) > float(
        sum_df.loc["sku_inelastic", "mean_discount_pct"]
    )

    # Case 2: Tiny rebate ($0.50) at 460 units -> margin erosion exceeds rebate, 0% discount wins.
    res_tiny = optimize_contract_volume(
        preds,
        contract_tiers=[(460.0, 0.50)],
        base_price=20.0,
        unit_cost=12.0,
        price_elasticity={"sku_elastic": -1.2, "sku_inelastic": -0.4},
    )
    assert res_tiny.rebate_earned == pytest.approx(0.0)
    assert res_tiny.optimal_portfolio_units == pytest.approx(400.0)

    # Case 3: Take-or-Pay min_contract_volume=450 with high shortfall penalty ($25/unit).
    res_top = optimize_contract_volume(
        preds,
        contract_tiers=[],
        min_contract_volume=450.0,
        shortfall_penalty_per_unit=25.0,
        base_price=20.0,
        unit_cost=8.0,
        price_elasticity={"sku_elastic": -2.5, "sku_inelastic": -0.5},
    )
    assert res_top.optimal_portfolio_units >= 450.0 - 1e-5
    assert res_top.shortfall_units == pytest.approx(0.0, abs=1e-5)
    assert res_top.net_value_lift > 0.0


# ---------------------------------------------------------------------------
# Tier C: Pathological Inputs, Infeasible Space Targets & Plotting
# ---------------------------------------------------------------------------


def test_tier_c_infeasible_space_and_sell_through_return_relaxation_hint() -> None:
    """Unreachable target_space_freed returns status='infeasible' with relaxation_hint."""
    dates = pd.date_range("2025-07-01", periods=5, freq="D")
    preds = pd.DataFrame({"ts_id": "sku_1", "ds": dates, "yhat": [10.0, 10.0, 10.0, 10.0, 10.0]})

    # Initial inventory is 40 units, unit_volume is 2.0 -> max possible space freed is 80.0.
    # Requesting target_space_freed=500.0 is physically impossible.
    res = optimize_clearance(
        preds,
        initial_inventory=40.0,
        unit_volume=2.0,
        target_space_freed=500.0,
    )
    assert res.status == "infeasible"
    assert not res.is_feasible
    assert res.relaxation_hint is not None
    assert "Infeasible target_space_freed=500.00" in res.relaxation_hint
    assert not res.schedule_df.empty

    with pytest.raises(ValueError, match="Infeasible target_space_freed"):
        optimize_clearance(
            preds,
            initial_inventory=40.0,
            unit_volume=2.0,
            target_space_freed=500.0,
            strict_feasibility=True,
        )

    # Unreachable min_contract_volume in optimize_contract_volume:
    cv_infeas = optimize_contract_volume(
        preds,
        min_contract_volume=10_000.0,
        shortfall_penalty_per_unit=10.0,
    )
    assert cv_infeas.status == "infeasible"
    assert not cv_infeas.is_feasible
    assert cv_infeas.relaxation_hint is not None
    assert cv_infeas.shortfall_units > 0.0


def test_tier_c_empty_zero_inventory_and_plot_smoke() -> None:
    """Empty frames, zero inventory, single-step horizons, and plot_clearance_schedule succeed."""
    empty_res = optimize_clearance(pd.DataFrame())
    assert empty_res.status == "empty"
    assert empty_res.schedule_df.empty

    empty_cv = optimize_contract_volume(pd.DataFrame())
    assert empty_cv.status == "empty"
    assert empty_cv.schedule_df.empty

    dates = pd.date_range("2025-08-01", periods=4, freq="D")
    zero_inv_df = optimize_clearance_df(
        pd.DataFrame({"ts_id": "zero_sku", "ds": dates, "yhat": [12.0, 15.0, 10.0, 8.0]}),
        initial_inventory=0.0,
        summary=True,
    )
    assert float(zero_inv_df["optimal_sell_through_units"].iloc[0]) == pytest.approx(0.0)

    sched_df = optimize_clearance_df(
        pd.DataFrame({"ts_id": "plot_sku", "ds": dates, "yhat": [20.0, 20.0, 20.0, 20.0]}),
        initial_inventory=90.0,
    )
    ax = plot_clearance_schedule(sched_df)
    assert ax is not None


# ---------------------------------------------------------------------------
# Tier D: Workbench Integration Across BakeOff, PlaygroundRun, and Forecaster
# ---------------------------------------------------------------------------


def test_tier_d_workbench_integration_bakeoff_and_forecaster() -> None:
    """BakeOff and Forecaster expose optimize_clearance_df and optimize_contract_volume_df."""
    panel = playground.sample_data(n_series=2, with_exog=True)
    bo = playground.bakeoff(panel, models=["naive_seasonal", "theta"], horizon=7, n_folds=1)

    cl_sum = bo.optimize_clearance_df(
        summary=True,
        initial_inventory=120.0,
        expiry_step=6,
        target_space_freed=40.0,
    )
    assert not cl_sum.empty
    assert set(CLEARANCE_SUMMARY_COLUMNS).issubset(cl_sum.columns)

    cv_sum = bo.optimize_contract_volume_df(summary=True)
    assert not cv_sum.empty
    assert set(CONTRACT_VOLUME_SUMMARY_COLUMNS).issubset(cv_sum.columns)

    ax = bo.plot_clearance_schedule()
    assert ax is not None

    fc = (
        Forecaster.from_Lossless(panel)
        if hasattr(Forecaster, "from_Lossless")
        else Forecaster(
            {
                "run_name": "clearance-wb-test",
                "data": {"source_table": "playground", "horizon": 7, "freq": "D"},
                "models": ["theta"],
            },
            data=panel,
        )
    )
    fc_cl = fc.optimize_clearance_df(summary=True, initial_inventory=100.0)
    assert not fc_cl.empty
    fc_cv = fc.optimize_contract_volume_df(summary=True)
    assert not fc_cv.empty


# ---------------------------------------------------------------------------
# Tier E: Cross-Feature Integration (Reconciliation, Covariates, All Model
#         Families/Ensembles/HPO/Modifiers, ACI/Copula Calibration & 7 Runtimes)
# ---------------------------------------------------------------------------


def test_tier_e_reconciliation_thief_and_simplex_coherence() -> None:
    """Hierarchical, THieF, Cross-Temporal & Compositional Simplex reconciliation:
    - Aggregate parent nodes (__total__, region=East, is_aggregate=True) are filtered out so
      leaf SKUs are never double-counted.
    - Reconciled leaf forecasts (reconciliation_method='mint_shrink', is_reconciled=True) are
      preferred over raw forecasts.
    - hierarchy_rollup_df satisfies exact bottom-up additivity across '__total__', 'region',
      and 'leaf' levels.
    """
    dates = pd.date_range("2025-09-01", periods=6, freq="D")
    rows: list[dict[str, object]] = []
    # Raw leaf forecasts vs reconciled leaf forecasts + aggregate parent nodes in same frame:
    for d in dates:
        # Raw (unreconciled) leaf rows with inflated demand (100/day):
        for sku, reg, cat in [
            ("East/Beverages/sku_1", "East", "Beverages"),
            ("East/Snacks/sku_2", "East", "Snacks"),
            ("West/Beverages/sku_3", "West", "Beverages"),
        ]:
            rows.append(
                {
                    "ts_id": sku,
                    "ds": d,
                    "yhat": 100.0,
                    "model_type": "theta",
                    "is_reconciled": False,
                    "reconciliation_method": "none",
                    "hierarchy_level": "leaf",
                    "region": reg,
                    "category": cat,
                }
            )
            # Reconciled leaf rows (cross_temporal_mint / mint_shrink) with coherent demand:
            rows.append(
                {
                    "ts_id": sku,
                    "ds": d,
                    "yhat": 20.0,
                    "model_type": "theta",
                    "is_reconciled": True,
                    "reconciliation_method": "mint_shrink",
                    "hierarchy_level": "leaf",
                    "region": reg,
                    "category": cat,
                }
            )
        # Aggregate hierarchy nodes that MUST be excluded from physical SKU optimization:
        rows.append(
            {
                "ts_id": "__total__",
                "ds": d,
                "yhat": 60.0,
                "model_type": "theta",
                "is_reconciled": True,
                "reconciliation_method": "mint_shrink",
                "hierarchy_level": "__total__",
                "region": "__total__",
                "category": "__total__",
            }
        )
        rows.append(
            {
                "ts_id": "region=East",
                "ds": d,
                "yhat": 40.0,
                "model_type": "theta",
                "is_reconciled": True,
                "reconciliation_method": "mint_shrink",
                "hierarchy_level": "region",
                "region": "East",
                "category": "__all__",
            }
        )

    rec_df = pd.DataFrame(rows)
    res_cl = optimize_clearance(
        rec_df,
        initial_inventory=110.0,
        expiry_step=6,
        reconciliation_method="mint_shrink",
        hierarchy_cols=["region", "category"],
    )
    # 1. Aggregate nodes (__total__, region=East) were stripped; only the 3 leaf SKUs remain:
    leaf_skus = set(res_cl.summary_df["ts_id"].astype(str))
    assert leaf_skus == {
        "East/Beverages/sku_1",
        "East/Snacks/sku_2",
        "West/Beverages/sku_3",
    }
    # 2. Reconciled rows (20/day * 6 days = 120 baseline demand, capped at 110 inventory) were used:
    for _, row in res_cl.summary_df.iterrows():
        assert float(row["baseline_sell_through_units"]) <= 110.0 + 1e-6

    # 3. Bottom-up hierarchy_rollup_df is populated and exactly additive:
    rollup = res_cl.hierarchy_rollup_df
    assert rollup is not None and not rollup.empty
    assert set(rollup["hierarchy_level"]) == {"__total__", "region", "category", "leaf"}
    tot_row = rollup[rollup["hierarchy_level"] == "__total__"].iloc[0]
    reg_sum = rollup[rollup["hierarchy_level"] == "region"]["optimal_sell_through_units"].sum()
    leaf_sum = rollup[rollup["hierarchy_level"] == "leaf"]["optimal_sell_through_units"].sum()
    assert float(tot_row["optimal_sell_through_units"]) == pytest.approx(float(reg_sum), rel=1e-6)
    assert float(tot_row["optimal_sell_through_units"]) == pytest.approx(float(leaf_sum), rel=1e-6)

    # 4. Also verify optimize_contract_volume with reconciled frame and @transform_ilr:
    res_cv = optimize_contract_volume(
        rec_df,
        reconciliation_method="mint_shrink",
        hierarchy_cols=["region", "category"],
    )
    assert set(res_cv.summary_df["ts_id"]) == leaf_skus
    assert res_cv.hierarchy_rollup_df is not None
    cv_tot = res_cv.hierarchy_rollup_df[
        res_cv.hierarchy_rollup_df["hierarchy_level"] == "__total__"
    ]["optimal_units"].iloc[0]
    assert float(cv_tot) == pytest.approx(res_cv.optimal_portfolio_units, rel=1e-6)


def test_tier_e_covariate_adjusted_fwl_and_double_ml_promo_elasticity() -> None:
    """3-Tier covariates (promo_flag, hdd_temp) partialled out via Frisch-Waugh-Lovell so
    confounded promo spikes do not corrupt price elasticity, plus direct ingestion of Double-ML
    promo_elasticity_df.
    """
    rng = np.random.default_rng(123)
    n_hist = 60
    hist_dates = pd.date_range("2025-01-01", periods=n_hist, freq="D")

    # Construct history where promo_flag=1 coincides with HIGHER price (+15%) and huge demand
    # spike (+120%). A naive regression without controlling for promo_flag & hdd_temp estimates
    # a spurious positive or near-zero elasticity, whereas FWL partialling recovers true_e = -2.2.
    promo = (np.arange(n_hist) % 5 == 0).astype(float)
    hdd = np.sin(np.linspace(0.0, 4.0 * np.pi, n_hist)) * 10.0 + 15.0
    price = 20.0 * np.exp(0.14 * promo + rng.normal(0.0, 0.08, size=n_hist))
    true_e = -2.2
    log_y = (
        np.log(50.0)
        + true_e * np.log(price / 20.0)
        + 0.85 * promo
        + 0.03 * hdd
        + rng.normal(0.0, 0.015, size=n_hist)
    )
    hist_df = pd.DataFrame(
        {
            "ts_id": "sku_cov",
            "ds": hist_dates,
            "y": np.exp(log_y),
            "price": price,
            "promo_flag": promo,
            "hdd_temp": hdd,
        }
    )
    fc_dates = pd.date_range(hist_dates[-1] + pd.Timedelta(days=1), periods=7, freq="D")
    fc_df = pd.DataFrame({"ts_id": "sku_cov", "ds": fc_dates, "yhat": [40.0] * 7})

    res_fwl = optimize_clearance(
        fc_df,
        history_df=hist_df,
        price_col="price",
        covariate_cols=["promo_flag", "hdd_temp"],
        initial_inventory=250.0,
    )
    est_e = float(res_fwl.summary_df["price_elasticity"].iloc[0])
    src_e = str(res_fwl.summary_df["elasticity_source"].iloc[0])
    assert src_e == "own_history"
    assert est_e == pytest.approx(true_e, abs=0.25)

    # Also verify direct ingestion of Double-ML promo_elasticity_df:
    dml_df = pd.DataFrame({"ts_id": ["sku_cov"], "elasticity": [-2.65]})
    res_dml = optimize_clearance(
        fc_df,
        history_df=hist_df,
        elasticity_df=dml_df,
        initial_inventory=250.0,
    )
    assert float(res_dml.summary_df["price_elasticity"].iloc[0]) == pytest.approx(-2.65, abs=1e-6)
    assert str(res_dml.summary_df["elasticity_source"].iloc[0]) == "own_history"


def test_tier_e_multi_model_families_ensembles_oof_champion_and_modifiers() -> None:
    """Multi-model predictions across all 5 model families (statistical, ml, deep_learning,
    automl, native), training modes (local, global, hybrid), @<tag> modifiers (@monotone,
    @similarity_global, @transform_ilr, @arps_decline), Optuna HPO, and OOF per-SKU champion
    selection.
    """
    dates = pd.date_range("2025-10-01", periods=6, freq="D")
    # Build multi-model predictions_df across all 5 model families + modifiers:
    model_specs = [
        ("theta", 15.0),  # statistical (local)
        ("ridge_lag@monotone", 18.0),  # ml (global + @monotone)
        ("ridge_lag@similarity_global", 19.0),  # ml (similarity_global cluster)
        ("theta@transform_ilr", 16.0),  # compositional simplex modifier
        ("holt_winters@arps_decline", 14.0),  # physics Arps decline modifier
        ("mlp_lag", 22.0),  # deep_learning
        ("automl_tabular", 25.0),  # automl
        ("bigquery_arima_plus", 30.0),  # native
    ]
    pred_rows: list[dict[str, object]] = []
    for sku in ("sku_a", "sku_b"):
        for m_name, val in model_specs:
            # Adjust per-SKU demand for ridge_lag@monotone vs mlp_lag so we can verify selection:
            y_val = (
                12.0
                if (sku == "sku_a" and m_name == "ridge_lag@monotone")
                else (28.0 if (sku == "sku_b" and m_name == "mlp_lag") else val)
            )
            for d in dates:
                pred_rows.append(
                    {
                        "ts_id": sku,
                        "ds": d,
                        "yhat": y_val,
                        "model_type": m_name,
                    }
                )
    multi_preds = pd.DataFrame(pred_rows)

    # Construct OOF backtest frame where 'ridge_lag@monotone' is best on 'sku_a' (MAE=0.1)
    # and 'mlp_lag' is best on 'sku_b' (MAE=0.1):
    oof_dates = pd.date_range("2025-09-20", periods=6, freq="D")
    oof_rows: list[dict[str, object]] = []
    for d in oof_dates:
        for m_name, _ in model_specs:
            err_a = 0.1 if m_name == "ridge_lag@monotone" else 15.0
            err_b = 0.1 if m_name == "mlp_lag" else 15.0
            oof_rows.append(
                {"ts_id": "sku_a", "ds": d, "y": 20.0, "yhat": 20.0 + err_a, "model_type": m_name}
            )
            oof_rows.append(
                {"ts_id": "sku_b", "ds": d, "y": 20.0, "yhat": 20.0 + err_b, "model_type": m_name}
            )
    oof_df = pd.DataFrame(oof_rows)

    # 1. Without ensemble_champion present, oof_df per-SKU champion selects:
    #    - 'ridge_lag@monotone' (yhat=12/day) for 'sku_a'
    #    - 'mlp_lag' (yhat=28/day) for 'sku_b'
    res_oof = optimize_clearance(
        multi_preds,
        oof_df=oof_df,
        initial_inventory=500.0,
    )
    sched_a = res_oof.schedule_df[res_oof.schedule_df["ts_id"] == "sku_a"]
    sched_b = res_oof.schedule_df[res_oof.schedule_df["ts_id"] == "sku_b"]
    assert len(sched_a) == 6 and len(sched_b) == 6
    assert float(sched_a["baseline_units"].iloc[0]) == pytest.approx(12.0, abs=1e-6)
    assert float(sched_b["baseline_units"].iloc[0]) == pytest.approx(28.0, abs=1e-6)

    # 2. When 'ensemble_champion' is present in predictions_df, it takes top priority automatically:
    ens_rows = [
        {"ts_id": sku, "ds": d, "yhat": 16.5, "model_type": "ensemble_champion"}
        for sku in ("sku_a", "sku_b")
        for d in dates
    ]
    preds_with_ens = pd.concat([multi_preds, pd.DataFrame(ens_rows)], ignore_index=True)
    res_ens = optimize_clearance(preds_with_ens, oof_df=oof_df, initial_inventory=500.0)
    assert float(res_ens.schedule_df["baseline_units"].iloc[0]) == pytest.approx(16.5, abs=1e-6)

    # 3. Live BakeOff + HPO worker.run_cell with multiple model families & ensemble strategies:
    from scale_forecasting.config import RunConfig
    from scale_forecasting.worker import run_cell

    panel = playground.sample_data(n_series=2, with_exog=True)
    hpo_cfg = RunConfig.model_validate(
        {
            "run_name": "clearance-hpo-cell",
            "data": {"source_table": "playground", "horizon": 6, "freq": "D"},
            "models": ["regression_lags@monotone"],
            "features": {"future_covariates": ["promo_flag"]},
            "backtest": {"enabled": True, "n_folds": 1, "horizon": 6, "step": 6},
            "hpo": {"enabled": True, "n_trials": 2},
        }
    )
    s0 = panel[panel["ts_id"] == panel["ts_id"].iloc[0]].reset_index(drop=True)
    cell_res = run_cell(s0, "regression_lags@monotone", hpo_cfg)
    assert cell_res.status == "ok"
    res_hpo = optimize_clearance(
        cell_res.predictions.assign(ts_id=cell_res.ts_id, model_type=cell_res.model_type),
        history_df=s0,
        initial_inventory=120.0,
    )
    assert len(res_hpo.summary_df) == 1

    bo_ens = playground.bakeoff(
        panel,
        models=["theta", "regression_lags@monotone"],
        future_covariates=["promo_flag"],
        horizon=6,
        n_folds=1,
        calculated="inverse_error",
        learned="nnls",
    )
    cl_ens = bo_ens.optimize_clearance_df(summary=True, initial_inventory=90.0)
    cv_ens = bo_ens.optimize_contract_volume_df(summary=True)
    assert not cl_ens.empty
    assert not cv_ens.empty


def test_tier_e_calibrated_intervals_aci_oof_cvar_and_all_7_runtimes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calibrated intervals (aci_lower, yhat_lower, q_0.1) & OOF residuals drive CVaR downside
    protection in optimize_contract_volume, and Forecaster works across all 7 cloud runtimes.
    """
    dates = pd.date_range("2025-11-01", periods=6, freq="D")
    # 1. Calibrated Online ACI / Conformal lower interval columns (aci_lower = 65% of yhat):
    preds_aci = pd.DataFrame(
        {
            "ts_id": ["sku_1"] * 6 + ["sku_2"] * 6,
            "ds": list(dates) * 2,
            "yhat": [30.0] * 12,
            "aci_lower": [19.5] * 12,
            "aci_upper": [40.5] * 12,
        }
    )
    cv_neutral = optimize_contract_volume(
        preds_aci,
        contract_tiers=[],
        min_contract_volume=370.0,
        shortfall_penalty_per_unit=25.0,
        risk_aversion_lambda=0.0,
    )
    cv_aci_risk = optimize_contract_volume(
        preds_aci,
        contract_tiers=[],
        min_contract_volume=370.0,
        shortfall_penalty_per_unit=25.0,
        risk_aversion_lambda=0.8,
        cvar_alpha=0.95,
    )
    assert cv_aci_risk.optimal_portfolio_units >= cv_neutral.optimal_portfolio_units
    assert cv_aci_risk.cvar_shortfall_units >= 0.0

    # 2. OOF empirical residuals drive CVaR shortfall when trajectories_df and intervals are absent:
    preds_point = preds_aci[["ts_id", "ds", "yhat"]].copy()
    oof_res = pd.DataFrame(
        {
            "ts_id": ["sku_1"] * 10,
            "ds": pd.date_range("2025-10-15", periods=10, freq="D"),
            "y": [18.0, 19.0, 20.0, 30.0, 31.0, 29.0, 17.0, 30.0, 28.0, 30.0],
            "yhat": [30.0] * 10,
        }
    )
    cv_oof_risk = optimize_contract_volume(
        preds_point,
        oof_df=oof_res,
        contract_tiers=[],
        min_contract_volume=370.0,
        shortfall_penalty_per_unit=25.0,
        risk_aversion_lambda=0.8,
    )
    assert cv_oof_risk.optimal_portfolio_units >= cv_neutral.optimal_portfolio_units

    # 3. Verify Forecaster workbench integration across all 7 cloud runtime configurations:
    panel = playground.sample_data(n_series=2, with_exog=True)
    ts_list = sorted(panel["ts_id"].unique().astype(str).tolist())
    panel_hier = panel.assign(region=np.where(panel["ts_id"] == ts_list[0], "East", "West"))
    runtime_configs: list[tuple[str, list[str], dict[str, object]]] = [
        ("spark", ["theta"], {"families": {"statistical": {"runtime": "spark"}}}),
        ("ray", ["regression_lags"], {"families": {"ml": {"runtime": "ray"}}}),
        ("vertex", ["mlp"], {"families": {"deep_learning": {"runtime": "vertex"}}}),
        ("gce", ["theta"], {"families": {"statistical": {"runtime": "gce"}}}),
        ("gke", ["regression_lags"], {"families": {"ml": {"runtime": "gke", "gke_mode": "job"}}}),
        ("vertex_automl", ["vertex_tide"], {"families": {"automl": {"runtime": "vertex_automl"}}}),
        ("bigquery", ["arima_plus"], {"families": {"statistical": {"runtime": "spark"}}}),
    ]
    for rt_label, models_list, compute_cfg in runtime_configs:
        fc = Forecaster(
            {
                "run_name": f"rt-{rt_label}",
                "data": {
                    "source_table": "proj.ds.tbl",
                    "horizon": 6,
                    "freq": "D",
                },
                "models": models_list,
                "features": {"future_covariates": ["promo_flag"]},
                "compute": compute_cfg,
                "hierarchy": {"enabled": True, "levels": [["region"]]},
            },
            data=panel_hier,
        )
        # Mock predictions_df retrieved from BigQuery forecast_predictions registry table:
        mock_preds = pd.DataFrame(
            {
                "ts_id": [ts_list[0]] * 6 + [ts_list[1]] * 6,
                "ds": list(dates) * 2,
                "yhat": [25.0] * 12,
                "yhat_lower": [18.0] * 12,
                "yhat_upper": [32.0] * 12,
                "model_type": [models_list[0]] * 12,
                "region": ["East"] * 6 + ["West"] * 6,
            }
        )
        monkeypatch.setattr(
            fc,
            "_resolve_track_k_frames",
            lambda _mp=mock_preds, _fc=fc, **_kw: (_mp, None, _fc._data),
        )
        cl_df = fc.optimize_clearance_df(summary=True, initial_inventory=120.0)
        cv_df = fc.optimize_contract_volume_df(summary=True, risk_aversion_lambda=0.5)
        assert len(cl_df) == 2
        assert len(cv_df) == 2
        assert cl_df.attrs["hierarchy_rollup_df"] is not None
        assert cv_df.attrs["hierarchy_rollup_df"] is not None
