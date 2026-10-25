"""Tier A DGP ground-truth recovery & integration tests for Track P.2 (`v1.8.1`).

Covers:
1. Positive Complementary Halo ($\\gamma_{ij} > 0$) & Geometric Ad-Stock
   ($A_{i,t} = D_{i,t} + \\lambda A_{i,t-1}$) in `elasticity.py` & `sdk.py`.
2. Piecewise Weather / Degree-Day Splines (`degree_day_cols`: `hdd_<col>` & `cdd_<col>`) in
   `features.py`, `monotonicity.py`, and `@monotone`.
3. Arps Decline Curve Analysis (`fit_arps_decline`, `@arps_decline`, `eur_total`, `eur_remaining`)
   across exponential ($b=0$), hyperbolic ($0 < b < 1$), and harmonic ($b=1$) regimes.
4. Multi-generation Product Supersession & Transition Splicing (`apply_supersession_map` &
   `features.supersession_map`).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from scale_forecasting import (
    Forecaster,
    RunConfig,
    apply_supersession_map,
    cold_start_transfer_df,
    compute_degree_days_df,
    compute_promo_elasticity_decomposition,
    fit_arps_decline,
    infer_monotonic_constraints,
    make_confounded_promo_dgp,
    run_cell,
    run_group,
    summarize_promo_elasticity,
)
from scale_forecasting.cold_start import _arps_cumulative_eur, _arps_rate_curve
from scale_forecasting.features import build_features, build_future_features, resolve_model_variant


def test_tier_a_dgp_positive_halo_cannibalization_and_adstock_recovery() -> None:
    """Recover own-promo lift, negative cannibalization, positive halo, and geometric ad-stock."""
    panel, gt = make_confounded_promo_dgp(
        n_series=4,
        n_days=180,
        true_promo_lift=45.0,
        true_pull_forward_dip=-10.0,
        true_cannibalization_dip=-12.0,
        true_halo_lift=18.0,
        adstock_decay=0.45,
        seed=42,
    )
    assert gt["true_halo_lift"] == 18.0
    assert gt["adstock_decay"] == 0.45

    decomp = compute_promo_elasticity_decomposition(
        panel,
        promo_col="promo_flag",
        adstock_decay=0.45,
        include_halo=True,
    )
    summary = summarize_promo_elasticity(decomp).set_index("ts_id")

    # 1. Verify new schema columns exist and are finite
    for col in (
        "adstock_effect",
        "halo_lift",
        "halo_effect",
        "top_halo_ts_id",
        "halo_similarity_score",
    ):
        assert col in decomp.columns

    # 2. Verify own-promo lift (> 0) recovery across all 4 SKUs
    assert (summary["mean_causal_promo_lift"] > 25.0).all()

    # 3. Substitute SKUs recover negative cannibalization dips
    assert (summary["total_cannibalization_dip"] < 0.0).any()
    assert (decomp["cannibalization_dip"] <= 0.0).all()

    # 4. Complementary SKU (sku_002 driven by sku_001 promos) recovers strictly positive halo lift
    assert float(summary.loc["sku_002", "total_halo_lift"]) > 20.0
    assert str(summary.loc["sku_002", "top_halo_ts_id"]) == "sku_001"
    assert float(summary.loc["sku_002", "halo_similarity_score"]) > 0.0
    assert (decomp["halo_lift"] >= 0.0).all()

    # 5. Geometric ad-stock carryover is strictly positive on post-promo decay days
    sku1 = decomp[decomp["ts_id"] == "sku_001"].reset_index(drop=True)
    promo_flags = panel[panel["ts_id"] == "sku_001"]["promo_flag"].reset_index(drop=True)
    post_promo_mask = (promo_flags == 0.0) & (promo_flags.shift(1, fill_value=0.0) > 0.0)
    assert post_promo_mask.any()
    assert (sku1.loc[post_promo_mask, "adstock_effect"] > 0.0).all()
    assert float(summary.loc["sku_001", "total_adstock_effect"]) > 0.0

    # 6. Auto ad-stock decay grid search via Forecaster.promo_elasticity_df selects positive decay
    cfg = RunConfig.model_validate(
        {
            "run_name": "halo-adstock-test",
            "data": {"source_table": "p.d.t", "horizon": 7, "freq": "D"},
            "models": ["naive_mean"],
            "features": {"future_covariates": ["promo_flag"]},
        }
    )
    f = Forecaster(cfg)
    auto_decomp = f.promo_elasticity_df(
        panel,
        promo_col="promo_flag",
        adstock_decay="auto",
        include_halo=True,
    )
    auto_summary = summarize_promo_elasticity(auto_decomp)
    assert (auto_summary["total_adstock_effect"] > 0.0).any()
    assert (auto_summary["total_halo_lift"] > 0.0).any()


def test_tier_a_dgp_piecewise_degree_days_and_monotone_recovery() -> None:
    """Recover V-shaped heating/cooling degree-day slopes and beat linear temperature baseline."""
    rng = np.random.default_rng(7)
    n = 90
    h = 14
    dates = pd.date_range("2025-01-01", periods=n + h, freq="D")
    # Smooth seasonal temperature wave swinging from 35F (winter) to 95F (summer)
    t_idx = np.arange(n + h, dtype=float)
    temp_f = 65.0 + 30.0 * np.sin(2.0 * np.pi * t_idx / 45.0)
    price = 10.0 + rng.uniform(-1.0, 1.0, size=n + h)

    hdd_true = np.maximum(0.0, 65.0 - temp_f)
    cdd_true = np.maximum(0.0, temp_f - 65.0)
    # True V-shaped energy/beverage demand DGP
    y_all = 120.0 + 3.5 * hdd_true + 4.5 * cdd_true - 2.0 * price + rng.normal(0.0, 0.5, size=n + h)

    full_df = pd.DataFrame(
        {
            "ts_id": "grid_zone_1",
            "ds": dates,
            "y": y_all,
            "temp_f": temp_f,
            "price": price,
        }
    )
    train_df = full_df.iloc[:n].reset_index(drop=True)
    fut_df = full_df.iloc[n:][["ts_id", "ds", "temp_f", "price"]].reset_index(drop=True)
    y_holdout = y_all[n:]

    # 1. Standalone compute_degree_days_df mathematical identities
    dd_df = compute_degree_days_df(train_df, {"temp_f": 65.0})
    assert {"hdd_temp_f", "cdd_temp_f"}.issubset(dd_df.columns)
    np.testing.assert_allclose(
        dd_df["cdd_temp_f"] - dd_df["hdd_temp_f"],
        dd_df["temp_f"] - 65.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(dd_df["hdd_temp_f"] * dd_df["cdd_temp_f"], 0.0, atol=1e-12)

    # 2. Monotonicity auto-inference assigns +1 to hdd_* and cdd_* and -1 to price
    inferred = infer_monotonic_constraints(["temp_f", "hdd_temp_f", "cdd_temp_f", "price"])
    assert inferred["hdd_temp_f"] == 1
    assert inferred["cdd_temp_f"] == 1
    assert inferred["price"] == -1
    assert "temp_f" not in inferred

    # 3. Declarative FeaturesConfig.degree_day_cols in build_features & build_future_features
    cfg_dd = RunConfig.model_validate(
        {
            "run_name": "degree-days-test",
            "data": {"source_table": "p.d.t", "horizon": h, "freq": "D"},
            "models": ["regression_lags@monotone"],
            "features": {
                "fourier": False,
                "future_covariates": ["temp_f", "price"],
                "degree_day_cols": {"temp_f": 65.0},
            },
            "backtest": {"enabled": False, "horizon": h, "n_folds": 1, "min_train": 30},
        }
    )
    y_tr, X_tr = build_features(train_df, cfg_dd)
    assert X_tr is not None
    assert {"hdd_temp_f", "cdd_temp_f"}.issubset(X_tr.columns)
    X_fut = build_future_features(y_tr, X_tr, cfg_dd, future_covariates_df=fut_df)
    assert X_fut is not None
    assert {"hdd_temp_f", "cdd_temp_f"}.issubset(X_fut.columns)

    # 4. Compare regression_lags@monotone with degree_day_cols vs linear temp_f only
    cfg_linear = cfg_dd.model_copy(
        update={
            "features": cfg_dd.features.model_copy(update={"degree_day_cols": {}}),
        }
    )
    cell_dd = run_cell(train_df, "regression_lags@monotone", cfg_dd, future_df=fut_df)
    cell_lin = run_cell(train_df, "regression_lags@monotone", cfg_linear, future_df=fut_df)
    assert cell_dd.status == "ok"
    assert cell_lin.status == "ok"

    wape_dd = float(np.sum(np.abs(y_holdout - cell_dd.predictions["yhat"])) / np.sum(y_holdout))
    wape_lin = float(np.sum(np.abs(y_holdout - cell_lin.predictions["yhat"])) / np.sum(y_holdout))
    assert wape_dd < 0.03
    assert wape_dd < 0.25 * wape_lin


def test_tier_a_dgp_arps_decline_and_eur_recovery() -> None:
    """Recover exact (q_i, b, D_i) and analytical EUR across exponential, hyperbolic & harmonic."""
    t_obs = np.arange(0, 40, dtype=float)
    t_fut = np.arange(40, 50, dtype=float)
    q_econ = 2.0

    regimes = [
        ("exponential", 500.0, 0.0, 0.04),
        ("hyperbolic", 800.0, 0.5, 0.06),
        ("harmonic", 600.0, 1.0, 0.08),
    ]
    for name, qi_true, b_true, di_true in regimes:
        y_obs = _arps_rate_curve(t_obs, qi_true, b_true, di_true)
        y_fut_true = _arps_rate_curve(t_fut, qi_true, b_true, di_true)
        eur_tot_true, eur_rem_true = _arps_cumulative_eur(
            qi_true, b_true, di_true, n_obs=len(t_obs), q_econ=q_econ
        )

        yhat_fut, params = fit_arps_decline(y_obs, horizon=10, q_econ=q_econ)
        assert math.isclose(params["q_i"], qi_true, rel_tol=1e-2), f"{name} q_i: {params}"
        assert math.isclose(params["b"], b_true, abs_tol=1e-2), f"{name} b: {params}"
        assert math.isclose(params["d_i"], di_true, rel_tol=1e-2), f"{name} d_i: {params}"
        assert math.isclose(params["eur_total"], eur_tot_true, rel_tol=1e-2), f"{name} eur_total"
        assert math.isclose(params["eur_remaining"], eur_rem_true, rel_tol=1e-2), f"{name} eur_rem"
        np.testing.assert_allclose(yhat_fut, y_fut_true, rtol=1e-2)

    # Verify @arps_decline variant tag and cold_start_transfer_df on a multi-well panel
    dates_mature = pd.date_range("2025-01-01", periods=30, freq="D")
    t30 = np.arange(0, 30, dtype=float)
    rows: list[dict[str, object]] = []
    for w_idx, qi_w in enumerate((900.0, 1000.0, 1100.0)):
        y_w = _arps_rate_curve(t30, qi_w, 0.5, 0.05)
        for dt, val in zip(dates_mature, y_w, strict=True):
            rows.append(
                {
                    "ts_id": f"donor_well_{w_idx}",
                    "ds": dt,
                    "y": float(val),
                    "basin": "Permian",
                }
            )
    # Early-life well with only 4 days of production
    dates_cs = dates_mature[-4:]
    y_cs = _arps_rate_curve(np.arange(0, 4, dtype=float), 950.0, 0.5, 0.05)
    for dt, val in zip(dates_cs, y_cs, strict=True):
        rows.append({"ts_id": "new_well_4obs", "ds": dt, "y": float(val), "basin": "Permian"})

    well_panel = pd.DataFrame(rows)
    report = cold_start_transfer_df(
        well_panel,
        method="arps_decline",
        min_obs=14,
        horizon=7,
        static_covariates=["basin"],
    )
    assert {"arps_qi", "arps_b", "arps_di", "eur_total", "eur_remaining"}.issubset(report.columns)
    cs_row = report[report["ts_id"] == "new_well_4obs"].iloc[0]
    assert bool(cs_row["is_cold_start"]) is True
    assert str(cs_row["selected_archetype"]) == "arps_decline"
    assert float(cs_row["eur_total"]) > float(cs_row["eur_remaining"]) > 0.0

    cfg = RunConfig.model_validate(
        {
            "run_name": "arps-decline-test",
            "data": {"source_table": "p.d.t", "horizon": 7, "freq": "D"},
            "models": ["naive_mean@arps_decline"],
            "features": {"cold_start_min_obs": 14, "static_covariates": ["basin"]},
        }
    )
    vspec = resolve_model_variant(cfg, "naive_mean@arps_decline")
    assert vspec.effective_cfg.features.cold_start_transfer is True
    assert vspec.effective_cfg.features.cold_start_method == "arps_decline"

    s_cs = well_panel[well_panel["ts_id"] == "new_well_4obs"].reset_index(drop=True)
    cell = run_cell(s_cs, "naive_mean@arps_decline", cfg, cold_start_donor_pool=well_panel)
    assert cell.status == "ok"
    assert "arps_decline" in cell.fit_diagnostics
    assert cell.fit_diagnostics["arps_decline"]["eur_total"] > 0.0


def test_tier_a_supersession_splicing_multi_generation_and_overlap() -> None:
    """Splice v1 -> v2 -> v3 predecessor histories and reconcile phase-out overlap windows."""
    dates_all = pd.date_range("2025-01-01", periods=50, freq="D")
    t_all = np.arange(50, dtype=float)
    true_continuous_y = 200.0 + 20.0 * np.sin(2.0 * np.pi * t_all / 7.0)

    # v1: Days 0..24 (Days 20..24 overlap with v2 at 40% share)
    y_v1 = true_continuous_y[:25].copy()
    y_v1[20:25] = 0.4 * true_continuous_y[20:25]
    df_v1 = pd.DataFrame({"ts_id": "SKU_v1", "ds": dates_all[:25], "y": y_v1, "generation": "gen1"})

    # v2: Days 20..44 (Days 20..24 overlap with v1 at 60% share)
    y_v2 = true_continuous_y[20:45].copy()
    y_v2[:5] = 0.6 * true_continuous_y[20:25]
    df_v2 = pd.DataFrame(
        {"ts_id": "SKU_v2", "ds": dates_all[20:45], "y": y_v2, "generation": "gen2"}
    )

    # v3: Days 45..49 (only 5 observations - would be cold-start without supersession)
    y_v3 = true_continuous_y[45:50].copy()
    df_v3 = pd.DataFrame(
        {"ts_id": "SKU_v3", "ds": dates_all[45:50], "y": y_v3, "generation": "gen3"}
    )

    raw_panel = pd.concat([df_v1, df_v2, df_v3], ignore_index=True)

    # Test both {succ: pred} and {pred: succ} orientations
    for sup_map in (
        {"SKU_v2": "SKU_v1", "SKU_v3": "SKU_v2"},
        {"SKU_v1": "SKU_v2", "SKU_v2": "SKU_v3"},
    ):
        spliced = apply_supersession_map(raw_panel, sup_map)
        v3_spliced = spliced[spliced["ts_id"] == "SKU_v3"].sort_values("ds").reset_index(drop=True)
        assert len(v3_spliced) == 50
        np.testing.assert_allclose(v3_spliced["y"].to_numpy(), true_continuous_y, atol=1e-9)
        assert (v3_spliced["generation"] == "gen3").all()
        chains = spliced.attrs["supersession_chains"]
        assert chains["SKU_v3"]["predecessors"] == ["SKU_v1", "SKU_v2"]
        assert chains["SKU_v3"]["spliced_n_obs"] == 50

    # Verify drop_predecessors=True keeps only terminal SKU_v3
    dropped = apply_supersession_map(
        raw_panel, {"SKU_v2": "SKU_v1", "SKU_v3": "SKU_v2"}, drop_predecessors=True
    )
    assert set(dropped["ts_id"].unique()) == {"SKU_v3"}

    # Verify end-to-end execution through run_group with features.supersession_map
    cfg = RunConfig.model_validate(
        {
            "run_name": "supersession-test",
            "data": {"source_table": "p.d.t", "horizon": 7, "freq": "D"},
            "models": ["naive_seasonal"],
            "features": {
                "cold_start_transfer": True,
                "cold_start_min_obs": 14,
                "supersession_map": {"SKU_v2": "SKU_v1", "SKU_v3": "SKU_v2"},
            },
            "backtest": {"enabled": True, "horizon": 7, "n_folds": 2, "min_train": 21, "step": 7},
        }
    )
    cells, _ = run_group(raw_panel, cfg)
    by_id = {c.ts_id: c for c in cells}
    v3_cell = by_id["SKU_v3"]
    assert v3_cell.status == "ok"
    assert v3_cell.n_folds_achieved == 2
    assert "supersession" in v3_cell.fit_diagnostics
    assert v3_cell.fit_diagnostics["supersession"]["predecessors"] == ["SKU_v1", "SKU_v2"]
    assert v3_cell.fit_diagnostics["supersession"]["spliced_n_obs"] == 50
