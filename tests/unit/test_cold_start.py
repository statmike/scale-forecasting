"""Unit tests for Cold-Start NPI & Hierarchical Analog Prior Transfer (`cold_start.py`)."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from scale_forecasting.cold_start import (
    build_cold_start_donor_catalog,
    cold_start_transfer_df,
    fit_bass_diffusion,
    fit_weibull_lifecycle,
    forecast_cold_start_series,
    plot_cold_start_transfer,
)
from scale_forecasting.config import FeaturesConfig, RunConfig
from scale_forecasting.features import resolve_model_variant
from scale_forecasting.worker import run_cell


def _make_npi_panel(seed: int = 42) -> pd.DataFrame:
    """Create a synthetic panel with mature donor series and cold-start NPI launch series."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    dates_mature = pd.date_range("2025-01-01", periods=60, freq="D")

    # 3 mature donor series in category "electronics" following a launch ramp + seasonality.
    t = np.arange(60, dtype=float)
    base_curve = (
        80.0 + 40.0 * (1.0 - np.exp(-0.08 * (t + 1.0))) + 10.0 * np.sin(2.0 * np.pi * t / 7.0)
    )
    for d_idx in range(3):
        scale = 1.0 + 0.15 * d_idx
        noise = rng.normal(0.0, 2.0, size=60)
        y_vals = np.maximum(1.0, scale * base_curve + noise)
        for dt, val in zip(dates_mature, y_vals, strict=True):
            rows.append(
                {
                    "ts_id": f"donor_{d_idx}",
                    "ds": dt,
                    "y": float(val),
                    "category": "electronics",
                    "store": f"S{d_idx}",
                }
            )

    # Cold-start series 1: n_obs = 5 days of launch history in "electronics".
    dates_cs5 = dates_mature[-5:]
    y_cs5 = 1.2 * base_curve[:5] + rng.normal(0.0, 1.0, size=5)
    for dt, val in zip(dates_cs5, y_cs5, strict=True):
        rows.append(
            {
                "ts_id": "npi_5obs",
                "ds": dt,
                "y": float(max(1.0, val)),
                "category": "electronics",
                "store": "S_new_1",
            }
        )

    # Cold-start series 2: n_obs = 1 day of launch history in "electronics".
    rows.append(
        {
            "ts_id": "npi_1obs",
            "ds": dates_mature[-1],
            "y": float(0.9 * base_curve[0]),
            "category": "electronics",
            "store": "S_new_2",
        }
    )

    return pd.DataFrame(rows)


def test_bass_diffusion_and_weibull_lifecycle_curve_recovery() -> None:
    """Bass diffusion and Weibull lifecycle curve fitters recover valid structural parameters."""
    t = np.arange(1, 31, dtype=float)
    # Synthetic Bass S-curve with p=0.04, q=0.35, m=5000.
    p_true, q_true, m_true = 0.04, 0.35, 5000.0
    exp_term = np.exp(-(p_true + q_true) * t)
    bass_y = (
        m_true
        * (((p_true + q_true) ** 2) / p_true)
        * exp_term
        / ((1.0 + (q_true / p_true) * exp_term) ** 2)
    )

    pred_bass, bass_params = fit_bass_diffusion(bass_y[:12], horizon=10)
    assert len(pred_bass) == 10
    assert np.all(np.isfinite(pred_bass))
    assert np.all(pred_bass >= 0.0)
    assert bass_params["p"] > 0.0
    assert bass_params["q"] >= bass_params["p"]
    assert bass_params["m"] > float(np.sum(bass_y[:12]))

    # Synthetic Weibull peak-and-decay curve with k=2.2, lambda=10.0, scale=2000.
    k_true, lam_true = 2.2, 10.0
    u = t / lam_true
    weibull_y = (
        2000.0 * (k_true / lam_true) * np.power(u, k_true - 1.0) * np.exp(-np.power(u, k_true))
    )
    pred_weib, weib_params = fit_weibull_lifecycle(weibull_y[:12], horizon=10)
    assert len(pred_weib) == 10
    assert np.all(np.isfinite(pred_weib))
    assert np.all(pred_weib >= 0.0)
    assert weib_params["k"] > 0.0
    assert weib_params["lambda"] > 0.0
    assert weib_params["amplitude"] > 0.0


def test_empirical_bayes_shrinkage_monotonicity_and_zero_obs() -> None:
    """Empirical-Bayes weight w_i = n_i / (n_i + kappa) scales monotonically from 0 at n_i=0."""
    panel = _make_npi_panel()
    pool = build_cold_start_donor_catalog(
        panel,
        min_obs=14,
        hierarchy_cols=["category"],
        horizon=7,
    )
    assert len(pool.donors) == 3

    # n_obs = 0 series (empty DataFrame) gets w_i = 0.0 and pure donor prior trajectory.
    empty_series = pd.DataFrame(
        {
            "ts_id": pd.Series(dtype=str),
            "ds": pd.Series(dtype="datetime64[ns]"),
            "y": pd.Series(dtype=float),
        }
    )
    res_0 = forecast_cold_start_series(
        empty_series,
        donor_pool=pool,
        horizon=7,
        min_obs=14,
        method="analog_transfer",
        ts_id="npi_0obs",
    )
    assert res_0.n_obs == 0
    assert res_0.is_cold_start is True
    assert res_0.empirical_bayes_weight == 0.0
    assert len(res_0.predictions) == 7
    assert (res_0.predictions["yhat_upper"] > res_0.predictions["yhat_lower"]).all()

    # n_obs = 1 series has 0 < w_1 < w_5 < 1.
    s1 = panel[panel["ts_id"] == "npi_1obs"]
    s5 = panel[panel["ts_id"] == "npi_5obs"]
    res_1 = forecast_cold_start_series(s1, donor_pool=pool, horizon=7, min_obs=14)
    res_5 = forecast_cold_start_series(s5, donor_pool=pool, horizon=7, min_obs=14)
    assert 0.0 < res_1.empirical_bayes_weight < res_5.empirical_bayes_weight < 1.0
    assert res_1.donor_ts_ids
    assert res_5.donor_ts_ids


def test_cold_start_transfer_df_columns_and_wape_lift() -> None:
    """cold_start_transfer_df populates all contract columns and beats naive baseline on NPIs."""
    panel = _make_npi_panel()
    report = cold_start_transfer_df(
        panel,
        min_obs=14,
        method="auto",
        hierarchy_cols=["category"],
        horizon=7,
    )
    required_cols = {
        "ts_id",
        "n_obs",
        "is_cold_start",
        "cold_start_flag",
        "donor_group",
        "matched_level",
        "selected_archetype",
        "method_used",
        "donor_ts_ids",
        "empirical_bayes_weight",
        "prior_scale",
        "bass_p",
        "bass_q",
        "bass_m",
        "weibull_k",
        "weibull_lambda",
        "pooled_q80_width",
        "yhat_mean",
        "interval_width_80",
        "oof_wape_lift_pct",
    }
    assert required_cols.issubset(report.columns)

    cs_rows = report[report["is_cold_start"]].set_index("ts_id")
    assert {"npi_5obs", "npi_1obs"}.issubset(cs_rows.index)
    assert (cs_rows["oof_wape_lift_pct"] > 0.0).all()
    assert (cs_rows["pooled_q80_width"] > 0.0).all()

    # Verify plotting helper returns a valid matplotlib Axes.
    ax = plot_cold_start_transfer(report, panel_df=panel, ts_id="npi_5obs")
    assert ax is not None


def test_cold_start_config_and_worker_pipeline() -> None:
    """FeaturesConfig cold_start_transfer and cold_start_method wire cleanly through run_cell."""
    panel = _make_npi_panel()
    for method in ("auto", "analog_transfer", "bass_diffusion", "weibull_lifecycle"):
        f_cfg = FeaturesConfig(
            cold_start_transfer=True,
            cold_start_min_obs=14,
            cold_start_method=method,
        )
        assert f_cfg.cold_start_transfer is True
        assert f_cfg.cold_start_method == method

    cfg = RunConfig.model_validate(
        {
            "run_name": "test_cold_start",
            "data": {"source_table": "p.d.t", "horizon": 7, "freq": "D"},
            "models": ["naive_mean@cold_start"],
            "features": {
                "cold_start_transfer": False,
                "cold_start_min_obs": 14,
                "cold_start_method": "analog_transfer",
            },
        }
    )
    variant = resolve_model_variant(cfg, "naive_mean@cold_start")
    assert variant.effective_cfg.features.cold_start_transfer is True

    pool = build_cold_start_donor_catalog(panel, cfg=cfg, min_obs=14, horizon=7)
    s5 = panel[panel["ts_id"] == "npi_5obs"][["ds", "y"]].reset_index(drop=True)
    cell = run_cell(
        "npi_5obs",
        s5,
        "naive_mean@cold_start",
        cfg,
        cold_start_donor_pool=pool,
    )
    assert cell.status == "ok"
    assert len(cell.predictions) == 7
    assert cell.fit_diagnostics.get("cold_start_transfer") is not None
    assert cell.fit_diagnostics["cold_start_transfer"]["is_cold_start"] is True


def test_cold_start_similarity_and_hybrid_matching() -> None:
    """Verify cold_start_matching='similarity'|'hybrid' and @similarity_cold_start transfer."""
    panel = _make_npi_panel()
    cfg = RunConfig.model_validate(
        {
            "run_name": "test_similarity_cold_start",
            "data": {"source_table": "p.d.t", "horizon": 7, "freq": "D"},
            "models": ["naive_mean@similarity_cold_start"],
            "features": {
                "cold_start_min_obs": 14,
                "cold_start_method": "analog_transfer",
                "cold_start_matching": "similarity",
                "similarity_method": "hybrid",
            },
        }
    )
    variant = resolve_model_variant(cfg, "naive_mean@similarity_cold_start")
    assert variant.effective_cfg.features.cold_start_transfer is True
    assert variant.effective_cfg.features.cold_start_matching == "similarity"

    s5 = panel[panel["ts_id"] == "npi_5obs"][["ts_id", "ds", "y"]].reset_index(drop=True)
    cell = run_cell(
        "npi_5obs",
        s5,
        "naive_mean@similarity_cold_start",
        cfg,
        cold_start_donor_pool=panel[["ts_id", "ds", "y"]],
    )
    assert cell.status == "ok"
    assert len(cell.predictions) == 7
    cs_diag = cell.fit_diagnostics["cold_start_transfer"]
    assert cs_diag["is_cold_start"] is True
    assert cs_diag["matching_mode_used"] == "similarity"
    assert float(cs_diag["donor_similarity_score"]) > 0.0
    assert len(cs_diag["top_donor_ts_ids"]) >= 1

    hybrid_report = cold_start_transfer_df(
        panel,
        horizon=7,
        freq="D",
        min_obs=14,
        matching="hybrid",
        similarity_method="hybrid",
        static_covariates=["category", "brand_tier"],
    )
    assert "matching_mode_used" in hybrid_report.columns
    assert "donor_similarity_score" in hybrid_report.columns
    assert "top_donor_ts_ids" in hybrid_report.columns
    npi_row = hybrid_report[hybrid_report["ts_id"] == "npi_5obs"].iloc[0]
    assert str(npi_row["matching_mode_used"]) == "hybrid"
    assert float(npi_row["donor_similarity_score"]) > 0.0


def test_multimodal_day0_to_day14_cold_start_handoff() -> None:
    """F17: Verify continuous Day-0 (0 obs) -> Day-5 (5 obs) -> Mature (28 obs) cold-start
    handoff using text_cols, attribute_cols, hierarchy_cols, and embedding_col."""
    dates = pd.date_range("2025-01-01", periods=28, freq="D")
    t = np.arange(28, dtype=float)
    oat_curve = 180.0 + 30.0 * np.sin(2.0 * np.pi * t / 7.0)
    soda_curve = 25.0 + 4.0 * np.cos(2.0 * np.pi * t / 7.0)

    panel = pd.concat(
        [
            pd.DataFrame(
                {
                    "ts_id": "donor_oat_orig",
                    "ds": dates,
                    "y": oat_curve,
                    "description": "Organic Oat Milk Original 64oz Carton",
                    "category": "Dairy",
                    "subcategory": "Plant Milk",
                    "brand": "OatCo",
                    "size_oz": 64.0,
                }
            ),
            pd.DataFrame(
                {
                    "ts_id": "donor_oat_choco",
                    "ds": dates,
                    "y": oat_curve * 0.92,
                    "description": "Organic Oat Milk Chocolate 64oz Carton",
                    "category": "Dairy",
                    "subcategory": "Plant Milk",
                    "brand": "OatCo",
                    "size_oz": 64.0,
                }
            ),
            pd.DataFrame(
                {
                    "ts_id": "donor_soda_can",
                    "ds": dates,
                    "y": soda_curve,
                    "description": "Sparkling Cola Soda 12oz Aluminum Can",
                    "category": "Beverages",
                    "subcategory": "Carbonated",
                    "brand": "FizzCo",
                    "size_oz": 12.0,
                }
            ),
            # Day-0 pre-launch SKU (n_obs == 0: y is NaN) with Oat Milk metadata
            pd.DataFrame(
                {
                    "ts_id": "npi_day0_oat_vanilla",
                    "ds": [pd.Timestamp("2025-01-28")],
                    "y": [np.nan],
                    "description": "Organic Oat Milk Vanilla 64oz Carton",
                    "category": "Dairy",
                    "subcategory": "Plant Milk",
                    "brand": "OatCo",
                    "size_oz": 64.0,
                }
            ),
            # Day-5 early-launch SKU (n_obs == 5)
            pd.DataFrame(
                {
                    "ts_id": "npi_day5_oat_barista",
                    "ds": dates[:5],
                    "y": oat_curve[:5] * 1.05,
                    "description": "Organic Oat Milk Barista Edition 64oz Carton",
                    "category": "Dairy",
                    "subcategory": "Plant Milk",
                    "brand": "OatCo",
                    "size_oz": 64.0,
                }
            ),
        ],
        ignore_index=True,
    )

    report = cold_start_transfer_df(
        panel,
        horizon=7,
        freq="D",
        min_obs=14,
        cold_start_matching="hybrid",
        text_cols=["description"],
        attribute_cols=["brand", "size_oz"],
        hierarchy_cols=["category", "subcategory"],
    )
    by_id = report.set_index("ts_id")

    # 1. Day-0 pre-launch SKU (n_obs == 0) matches Oat Milk donors purely from metadata
    day0 = by_id.loc["npi_day0_oat_vanilla"]
    assert int(day0["n_obs"]) == 0
    assert bool(day0["is_cold_start"]) is True
    assert str(day0["lifecycle_stage"]) == "pre_launch"
    assert abs(float(day0["empirical_bayes_weight"])) < 1e-9
    assert float(day0["donor_similarity_score"]) > 0.65
    assert str(day0["top_donor_ts_ids"]).startswith("donor_oat_")
    # Prior scale transfers from high-volume Oat Milk (~180) rather than Soda (~25)
    assert float(day0["prior_scale"]) > 120.0
    assert float(day0["yhat_mean"]) > 120.0

    # 2. Day-5 early-launch SKU (n_obs == 5) blends metadata + 5-day launch prefix
    day5 = by_id.loc["npi_day5_oat_barista"]
    assert int(day5["n_obs"]) == 5
    assert bool(day5["is_cold_start"]) is True
    assert str(day5["lifecycle_stage"]) == "early_launch"
    assert 0.0 < float(day5["empirical_bayes_weight"]) < 1.0
    assert float(day5["donor_similarity_score"]) > 0.65
    assert str(day5["top_donor_ts_ids"]).startswith("donor_oat_")

    # 3. Mature SKU (n_obs == 28) is not cold-start
    mature = by_id.loc["donor_oat_orig"]
    assert int(mature["n_obs"]) == 28
    assert bool(mature["is_cold_start"]) is False
    assert str(mature["lifecycle_stage"]) == "mature"
    assert str(mature["method_used"]) == "none"
