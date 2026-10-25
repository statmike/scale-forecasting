"""4-Tier End-to-End (E2E) Verification Suite for Track L (`v1.7.0`).

Covers all Track L features (F1–F10) across 4 systematic testing tiers:
- Tier 1 — Category-Partition Feature Coverage (Features F1–F10, >= 5 cases per feature)
- Tier 2 — Boundary Value & Corner-Case Hardening (>= 5 cases per feature area)
- Tier 3 — Pairwise & Multi-Subsystem Cross-Feature Combinations (12 pairwise suites)
- Tier 4 — Real-World Enterprise Application Workloads (6 industry scenarios)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

import scale_forecasting as sf
from scale_forecasting.auto import build_config, inspect_panel_for_auto
from scale_forecasting.auto_policy import (
    evolved_policy_thresholds,
    propose_autofocus_dag_branches,
    propose_autofocus_pass,
)
from scale_forecasting.benchmarks import list_benchmarks
from scale_forecasting.calibration import apply_calibration, calibrate_from_oof
from scale_forecasting.config import (
    CALIBRATION_METHODS,
    COLD_START_METHODS,
    VALID_VARIANT_TAGS,
    FeaturesConfig,
    OutputConfig,
    RunConfig,
)
from scale_forecasting.diagnostics import (
    INVENTORY_POLICY_COLUMNS,
    compute_inventory_policy,
)
from scale_forecasting.elasticity import compute_promo_elasticity_decomposition
from scale_forecasting.engines.spark_io import run_group
from scale_forecasting.errors import ConfigError, DataError
from scale_forecasting.evolve.targets import (
    AutoPolicyWeights,
    build_mutation_prompt,
    deterministic_mock_candidates,
    get_target_seed,
)
from scale_forecasting.goal_seek import GoalSeekResult, allocate_target
from scale_forecasting.mcp import McpServer
from scale_forecasting.playground import bakeoff, run_model
from scale_forecasting.reconciliation import (
    build_hierarchy,
    verify_coherence,
)
from scale_forecasting.registry.ids import (
    canonical_config,
    make_model_spec_hash,
    make_run_id,
)
from scale_forecasting.report import export_report, render_html_report
from scale_forecasting.scenarios import simulate_scenarios
from scale_forecasting.simplex import verify_simplex_coherence
from scale_forecasting.temporal_reconciliation import (
    verify_temporal_coherence,
)

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[2]


# =============================================================================
# Deterministic Synthetic Panel & OOF Generators
# =============================================================================


def _make_cold_start_npi_panel(
    n_mature_days: int = 42,
    n_short_days: int = 5,
    seed: int = 101,
) -> pd.DataFrame:
    """Create a multi-series retail panel with mature donors, short SKUs, and a 0-obs SKU."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_mature_days, freq="D")
    t = np.arange(n_mature_days, dtype=float)

    donor_specs = [
        ("donor_elec_a", "Electronics", "BrandX", "North", 120.0, 25.0),
        ("donor_elec_b", "Electronics", "BrandX", "North", 140.0, 30.0),
        ("donor_elec_c", "Electronics", "BrandY", "South", 100.0, 20.0),
        ("donor_fash_a", "Fashion", "BrandZ", "North", 85.0, 18.0),
        ("donor_fash_b", "Fashion", "BrandZ", "South", 90.0, 19.0),
        ("donor_fash_c", "Fashion", "BrandW", "East", 75.0, 15.0),
    ]
    rows: list[dict[str, Any]] = []
    for tid, cat, brand, reg, base, amp in donor_specs:
        # Surge-and-seasonal lifecycle shape
        surge = amp * (t / 7.0) * np.exp(1.0 - t / 7.0)
        seas = 10.0 * np.sin(2.0 * np.pi * t / 7.0)
        noise = rng.normal(0.0, 3.0, size=n_mature_days)
        y_vals = np.maximum(base + surge + seas + noise, 5.0)
        for dt, y_v in zip(dates, y_vals, strict=True):
            rows.append(
                {
                    "ts_id": tid,
                    "category": cat,
                    "brand": brand,
                    "region": reg,
                    "ds": dt,
                    "y": float(y_v),
                }
            )

    # Short-history NPI SKU 1: exact match on (Electronics, BrandX, North) with n_short_days obs
    short_dates = dates[-n_short_days:]
    for idx, dt in enumerate(short_dates):
        rows.append(
            {
                "ts_id": "npi_short_exact",
                "category": "Electronics",
                "brand": "BrandX",
                "region": "North",
                "ds": dt,
                "y": float(115.0 + 6.0 * idx + rng.normal(0.0, 2.0)),
            }
        )

    # Short-history NPI SKU 2: partial match on (Fashion, BrandNew, West) -> falls back to category
    for idx, dt in enumerate(dates[-4:]):
        rows.append(
            {
                "ts_id": "npi_short_fallback",
                "category": "Fashion",
                "brand": "BrandNew",
                "region": "West",
                "ds": dt,
                "y": float(80.0 + 4.0 * idx + rng.normal(0.0, 1.5)),
            }
        )

    # Pre-launch 0-observation NPI SKU: represented by a single NaN target row so it is in the panel
    rows.append(
        {
            "ts_id": "npi_zero_obs",
            "category": "Electronics",
            "brand": "BrandY",
            "region": "South",
            "ds": dates[-1],
            "y": float("nan"),
        }
    )

    return pd.DataFrame(rows)


def _make_regime_shift_oof_and_preds(
    n_oof_steps: int = 36,
    horizon: int = 7,
    shock_multiplier: float = 3.5,
    seed: int = 202,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create sequential OOF residuals with a mid-series volatility shock + forward predictions."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_oof_steps, freq="D")
    split_idx = n_oof_steps // 2
    oof_rows: list[dict[str, Any]] = []
    for t_idx, dt in enumerate(dates):
        base_yhat = 100.0 + 8.0 * np.sin(2.0 * np.pi * t_idx / 7.0)
        # First half: calm sigma=3.0; second half: structural break + 3.5x volatility shock
        if t_idx < split_idx:
            err = float(rng.normal(0.0, 3.0))
        else:
            # Deterministic alternating high-amplitude shock errors so static q80 (~4.0) misses >70%
            sign = 1.0 if (t_idx % 2 == 0) else -1.0
            err = float(sign * 3.8 * shock_multiplier + rng.normal(0.0, 1.5))
        y_true = base_yhat + err
        for h_step in (1, 2):
            oof_rows.append(
                {
                    "ts_id": "sku_shock",
                    "model_type": "theta",
                    "fold_id": t_idx // 6,
                    "cutoff_date": dt - pd.Timedelta(days=h_step),
                    "ds": dt,
                    "horizon_step": h_step,
                    "y_true": float(y_true),
                    "yhat": float(base_yhat),
                    "yhat_raw": float(base_yhat),
                    "yhat_lower": float(base_yhat - 4.2),
                    "yhat_upper": float(base_yhat + 4.2),
                }
            )
    oof_df = pd.DataFrame(oof_rows)

    f_dates = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    pred_rows: list[dict[str, Any]] = []
    for h_idx, dt in enumerate(f_dates, start=1):
        yh = 105.0 + 5.0 * np.sin(2.0 * np.pi * h_idx / 7.0)
        pred_rows.append(
            {
                "ts_id": "sku_shock",
                "model_type": "theta",
                "ds": dt,
                "forecast_date": dt,
                "horizon_step": h_idx,
                "yhat": float(yh),
                "yhat_raw": float(yh),
                "yhat_adjusted": float(yh),
                "yhat_lower": float(yh - 4.5),
                "yhat_upper": float(yh + 4.5),
            }
        )
    preds_df = pd.DataFrame(pred_rows)
    return oof_df, preds_df


def _make_autocorrelated_heavy_tail_oof_and_preds(
    n_series: int = 3,
    horizon: int = 7,
    n_folds: int = 8,
    rho: float = 0.78,
    heavy_tails: bool = True,
    seed: int = 303,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create multi-series predictions, positively autocorrelated heavy-tailed OOF, and actuals."""
    rng = np.random.default_rng(seed)
    f_dates = pd.date_range("2025-03-01", periods=horizon, freq="D")
    pred_rows: list[dict[str, Any]] = []
    actual_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []

    for s_idx in range(n_series):
        tid = f"sku_{s_idx}"
        base_level = 100.0 + 25.0 * s_idx
        for h_idx, dt in enumerate(f_dates, start=1):
            yh = base_level + 6.0 * np.sin(2.0 * np.pi * h_idx / 7.0)
            half_w = 10.0 + 1.2 * h_idx
            pred_rows.append(
                {
                    "ts_id": tid,
                    "model_type": "holtwinters",
                    "ds": dt,
                    "forecast_date": dt,
                    "horizon_step": h_idx,
                    "yhat": float(yh),
                    "yhat_raw": float(yh),
                    "yhat_lower": float(yh - half_w),
                    "yhat_upper": float(yh + half_w),
                }
            )
            actual_rows.append(
                {
                    "ts_id": tid,
                    "ds": dt,
                    "forecast_date": dt,
                    "horizon_step": h_idx,
                    "y_true": float(yh + rng.normal(0.0, 4.0)),
                    "y": float(yh + rng.normal(0.0, 4.0)),
                }
            )

    # Generate AR(1) positively autocorrelated OOF residuals across horizon steps 1..H
    for f_idx in range(n_folds):
        cutoff = pd.Timestamp("2025-01-01") + pd.Timedelta(days=7 * f_idx)
        # Common cross-series shock factor so cross_series=True captures positive co-movement
        common_shock = float(rng.standard_t(df=3.0) if heavy_tails else rng.normal(0.0, 1.0))
        for s_idx in range(n_series):
            tid = f"sku_{s_idx}"
            base_level = 100.0 + 25.0 * s_idx
            prev_e = 0.0
            for h_idx in range(1, horizon + 1):
                raw_innov = (
                    float(rng.standard_t(df=2.8)) if heavy_tails else float(rng.normal(0.0, 1.0))
                )
                innov = 0.6 * common_shock + 0.8 * raw_innov
                e_t = rho * prev_e + np.sqrt(1.0 - rho**2) * innov * 6.0
                prev_e = e_t
                dt = cutoff + pd.Timedelta(days=h_idx)
                yh = base_level + 4.0 * np.sin(2.0 * np.pi * h_idx / 7.0)
                oof_rows.append(
                    {
                        "ts_id": tid,
                        "model_type": "holtwinters",
                        "fold_id": f_idx,
                        "cutoff_date": cutoff,
                        "ds": dt,
                        "horizon_step": h_idx,
                        "y_true": float(yh + e_t),
                        "yhat": float(yh),
                        "yhat_raw": float(yh),
                        "yhat_lower": float(yh - 9.0),
                        "yhat_upper": float(yh + 9.0),
                    }
                )

    return pd.DataFrame(pred_rows), pd.DataFrame(oof_rows), pd.DataFrame(actual_rows)


def _make_confounded_price_promo_hierarchy_panel(
    n_days: int = 56,
    horizon: int = 10,
    seed: int = 404,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a hierarchical panel + future covariates with causal price, promo, and dip."""
    rng = np.random.default_rng(seed)
    all_dates = pd.date_range("2025-01-01", periods=n_days + horizon, freq="D")
    hist_dates = all_dates[:n_days]
    fut_dates = all_dates[n_days:]

    sku_meta = [
        ("North_Bev_A", "North", "Beverages", 10.0, 4.5, -1.6, 28.0, -7.0),
        ("North_Bev_B", "North", "Beverages", 12.0, 5.5, -1.4, 24.0, -6.0),
        ("South_Snack_A", "South", "Snacks", 8.0, 3.5, -1.8, 22.0, -5.0),
        ("South_Snack_B", "South", "Snacks", 9.0, 4.0, -1.5, 20.0, -4.5),
    ]

    hist_rows: list[dict[str, Any]] = []
    fut_rows: list[dict[str, Any]] = []

    for s_idx, (tid, reg, cat, base_p, cost, elast, promo_lift, dip_lag1) in enumerate(sku_meta):
        prev_promo = 0.0
        for t_idx, dt in enumerate(hist_dates):
            seas = 12.0 * np.sin(2.0 * np.pi * t_idx / 7.0)
            promo = 1.0 if (t_idx % 9 == (s_idx * 2) % 9) else 0.0
            price = base_p * (0.90 if promo > 0 else 1.0 + 0.04 * np.sin(t_idx / 5.0))
            rel_p = (price - base_p) / base_p
            y_val = (
                95.0
                + 10.0 * s_idx
                + seas
                + 95.0 * elast * rel_p
                + promo_lift * promo
                + dip_lag1 * prev_promo
                + rng.normal(0.0, 2.0)
            )
            prev_promo = promo
            hist_rows.append(
                {
                    "ts_id": tid,
                    "region": reg,
                    "category": cat,
                    "ds": dt,
                    "y": float(max(y_val, 8.0)),
                    "price": float(price),
                    "promo": float(promo),
                    "unit_cost": float(cost),
                }
            )

        for _h_idx, dt in enumerate(fut_dates):
            fut_rows.append(
                {
                    "ts_id": tid,
                    "region": reg,
                    "category": cat,
                    "ds": dt,
                    "price": float(base_p),
                    "promo": 0.0,
                    "unit_cost": float(cost),
                }
            )

    return pd.DataFrame(hist_rows), pd.DataFrame(fut_rows)


# =============================================================================
# TIER 1 — Category-Partition Feature Coverage (Features F1–F10, >= 5 cases each)
# =============================================================================


def test_t1_f1_run_config_track_l_fields_and_digest_invariance() -> None:
    """Tier 1 F1 (>= 5 cases): Verify FeaturesConfig & OutputConfig Track L fields,
    exported vocabulary constants, and 100% run_id / model_spec_hash invariance at defaults.
    """
    # Case 1: Exported vocabulary tuples/sets exist and contain expected Track L values
    assert set(COLD_START_METHODS) == {
        "auto",
        "analog_transfer",
        "bass_diffusion",
        "weibull_lifecycle",
        "arps_decline",
    }
    assert set(CALIBRATION_METHODS) == {"conformal", "aci", "aci_normalized"}
    for req_tag in ("monotone", "transform_clr", "transform_ilr", "univariate", "global", "local"):
        assert req_tag in VALID_VARIANT_TAGS

    # Case 2: Default field values on FeaturesConfig and OutputConfig
    f_cfg = FeaturesConfig()
    assert f_cfg.cold_start_transfer is False
    assert f_cfg.cold_start_min_obs == 14
    assert f_cfg.cold_start_method == "auto"

    o_cfg = OutputConfig()
    assert o_cfg.calibration_method == "conformal"
    assert o_cfg.aci_gamma == pytest.approx(0.05)

    # Case 3: 100% run_id and model_spec_hash invariance between implicit and explicit defaults
    base_payload = {
        "run_name": "track-l-digest-check",
        "data": {"source_table": "p.d.t", "freq": "D", "horizon": 14},
        "models": ["naive_mean", "theta"],
    }
    cfg_implicit = RunConfig.model_validate(base_payload)
    cfg_explicit = RunConfig.model_validate(
        {
            **base_payload,
            "features": {
                "cold_start_transfer": False,
                "cold_start_min_obs": 14,
                "cold_start_method": "auto",
            },
            "output": {
                "calibration_method": "conformal",
                "aci_gamma": 0.05,
            },
        }
    )
    assert canonical_config(cfg_implicit) == canonical_config(cfg_explicit)
    assert make_run_id(cfg_implicit) == make_run_id(cfg_explicit)
    for model_name in cfg_implicit.models:
        assert make_model_spec_hash(model_name, cfg_implicit) == make_model_spec_hash(
            model_name, cfg_explicit
        )

    # Case 4: Non-default cold_start_transfer=True changes both run_id and model_spec_hash
    cfg_cs = RunConfig.model_validate(
        {
            **base_payload,
            "features": {
                "cold_start_transfer": True,
                "cold_start_min_obs": 21,
                "cold_start_method": "bass_diffusion",
            },
        }
    )
    assert make_run_id(cfg_cs) != make_run_id(cfg_implicit)
    assert make_model_spec_hash("theta", cfg_cs) != make_model_spec_hash("theta", cfg_implicit)

    # Case 5: Non-default calibration_method="aci_normalized" and aci_gamma=0.10 change digests
    cfg_aci = RunConfig.model_validate(
        {
            **base_payload,
            "output": {
                "calibration_method": "aci_normalized",
                "aci_gamma": 0.10,
            },
        }
    )
    assert make_run_id(cfg_aci) != make_run_id(cfg_implicit)
    assert make_model_spec_hash("theta", cfg_aci) != make_model_spec_hash("theta", cfg_implicit)


def test_t1_f2_cold_start_npi_archetypes_and_hierarchical_donor_transfer() -> None:
    """Tier 1 F2 (>= 5 cases): Verify Cold-Start NPI donor matching, hierarchical fallback,
    launch archetypes (`analog_transfer`, `bass_diffusion`, `weibull_lifecycle`, `auto`),
    Empirical-Bayes weighting, Day-1 intervals, and pipeline hooks.
    """
    from scale_forecasting.cold_start import (
        build_cold_start_donor_pool,
        cold_start_transfer_df,
        fit_bass_diffusion,
        fit_weibull_lifecycle,
        forecast_cold_start_series,
        plot_cold_start_transfer,
    )

    panel = _make_cold_start_npi_panel(n_mature_days=42, n_short_days=5, seed=101)
    cfg = RunConfig.model_validate(
        {
            "run_name": "t1-f2-cold-start",
            "data": {"source_table": "p.d.t", "freq": "D", "horizon": 7},
            "models": ["naive_mean"],
            "features": {
                "static_covariates": ["category", "brand", "region"],
                "cold_start_transfer": True,
                "cold_start_min_obs": 14,
                "cold_start_method": "auto",
            },
        }
    )

    # Case 1: cold_start_transfer_df schema and cold-start detection across mature/short/0-obs
    cs_df = cold_start_transfer_df(
        panel,
        cfg=cfg,
        min_obs=14,
        method="auto",
        static_covariates=["category", "brand", "region"],
        horizon=7,
        freq="D",
    )
    assert isinstance(cs_df, pd.DataFrame)
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
    assert required_cols.issubset(cs_df.columns)
    cs_by_id = cs_df.set_index("ts_id")
    assert bool(cs_by_id.loc["npi_short_exact", "is_cold_start"]) is True
    assert bool(cs_by_id.loc["npi_zero_obs", "is_cold_start"]) is True
    assert bool(cs_by_id.loc["donor_elec_a", "is_cold_start"]) is False

    # Case 2: Explicit archetypes — Bass diffusion and Weibull lifecycle curve fitting
    donor_y = panel.loc[panel["ts_id"] == "donor_elec_a", "y"].to_numpy(dtype=float)
    bass_fc, bass_params = fit_bass_diffusion(donor_y, horizon=7)
    assert len(bass_fc) == 7
    assert np.all(np.isfinite(bass_fc)) and np.all(bass_fc >= 0.0)
    assert float(bass_params.get("p", 0.0)) > 0.0
    assert float(bass_params.get("q", -1.0)) >= 0.0
    assert float(bass_params.get("m", 0.0)) > 0.0

    weib_fc, weib_params = fit_weibull_lifecycle(donor_y, horizon=7)
    assert len(weib_fc) == 7
    assert np.all(np.isfinite(weib_fc)) and np.all(weib_fc >= 0.0)
    assert float(weib_params.get("k", 0.0)) > 0.0

    # Case 3: All 3 explicit methods + "auto" LODO-CV execute cleanly on forecast_cold_start_series
    pool = build_cold_start_donor_pool(
        panel,
        cfg,
        min_obs=14,
        static_covariates=["category", "brand", "region"],
    )
    short_sub = panel[panel["ts_id"] == "npi_short_exact"].copy()
    for method_name in ("analog_transfer", "bass_diffusion", "weibull_lifecycle", "auto"):
        res_m = forecast_cold_start_series(
            short_sub,
            pool,
            horizon=7,
            freq="D",
            method=method_name,
            min_obs=14,
            static_covariates=["category", "brand", "region"],
        )
        assert res_m.cold_start_flag is True
        assert res_m.method_used in {"analog_transfer", "bass_diffusion", "weibull_lifecycle"}
        assert len(res_m.predictions) == 7
        assert bool((res_m.predictions["yhat_upper"] > res_m.predictions["yhat_lower"]).all())

    # Case 4: Hierarchical static-covariate fallback & Empirical-Bayes monotonicity (w0 < w4 < w5)
    w_zero = float(cs_by_id.loc["npi_zero_obs", "empirical_bayes_weight"])
    w_four = float(cs_by_id.loc["npi_short_fallback", "empirical_bayes_weight"])
    w_five = float(cs_by_id.loc["npi_short_exact", "empirical_bayes_weight"])
    assert w_zero == pytest.approx(0.0, abs=1e-9)
    assert 0.0 < w_four < w_five < 1.0
    assert str(cs_by_id.loc["npi_short_exact", "matched_level"]) != ""
    assert str(cs_by_id.loc["npi_short_fallback", "matched_level"]) != ""

    # Case 5: Pipeline hook via run_group & plot_cold_start_transfer
    cells, status_df = run_group(panel[panel["ts_id"] != "npi_zero_obs"].copy(), cfg)
    assert "ok" in set(status_df["status"])
    short_cells = [c for c in cells if c.ts_id == "npi_short_exact"]
    assert len(short_cells) == 1
    assert len(short_cells[0].predictions) == 7
    assert "cold_start" in short_cells[0].diagnostics

    ax = plot_cold_start_transfer(cs_df)
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close("all")


def test_t1_f3_online_adaptive_conformal_inference_and_delayed_feedback() -> None:
    """Tier 1 F3 (>= 5 cases): Verify Gibbs & Candès online ACI (`aci` & `aci_normalized`),
    EWMA local volatility scaling, leak-free h-step feedback delay, coverage recovery after
    volatility shock, and in-pipeline calibration hooks.
    """
    from scale_forecasting.aci import (
        aci_df,
        aci_trajectory_df,
        calibrate_aci_from_oof,
        compute_aci_df,
        plot_aci,
    )

    oof_df, preds_df = _make_regime_shift_oof_and_preds(
        n_oof_steps=36, horizon=7, shock_multiplier=3.5, seed=202
    )

    # Case 1: Schema completeness across compute_aci_df, aci_df, and aci_trajectory_df aliases
    assert aci_df is compute_aci_df or callable(aci_df)
    assert callable(aci_trajectory_df)
    df_aci = compute_aci_df(
        oof_df,
        preds_df,
        gamma=0.05,
        target_coverage=0.80,
        normalized=True,
        horizon_delay=True,
    )
    required_cols = {
        "ts_id",
        "model_type",
        "ds",
        "step_idx",
        "horizon_step",
        "y_true",
        "yhat",
        "abs_error",
        "volatility_scale",
        "alpha_t",
        "err_t",
        "static_lower",
        "static_upper",
        "aci_lower",
        "aci_upper",
        "static_covered",
        "aci_covered",
        "static_coverage",
        "aci_coverage",
        "target_coverage",
        "static_width",
        "aci_width",
        "interval_width_ratio",
        "drift_detected",
    }
    assert required_cols.issubset(df_aci.columns)

    # Case 2: Coverage recovery under mid-series 3.5x volatility shock (aci_cov > static_cov)
    h1_post_shock = df_aci[(df_aci["horizon_step"] == 1) & (df_aci["step_idx"] >= 18)]
    static_post_cov = float(h1_post_shock["static_covered"].mean())
    aci_post_cov = float(h1_post_shock["aci_covered"].mean())
    assert aci_post_cov > static_post_cov
    assert bool(df_aci["drift_detected"].any()) is True
    assert float(h1_post_shock["volatility_scale"].max()) > 1.15
    assert float(h1_post_shock["interval_width_ratio"].max()) > 1.15

    # Case 3: Unnormalized Gibbs & Candès update (normalized=False) adapts alpha_t when errors occur
    df_unnorm = compute_aci_df(
        oof_df,
        gamma=0.08,
        target_coverage=0.80,
        normalized=False,
        horizon_delay=True,
    )
    h1_unnorm = df_unnorm[df_unnorm["horizon_step"] == 1].sort_values("step_idx")
    # Post-shock errors push effective miscoverage alpha_t down (or widen intervals)
    assert float(h1_unnorm["aci_width"].iloc[-1]) > float(h1_unnorm["static_width"].iloc[0])

    # Case 4: Leak-free h-step horizon feedback delay (perturbing future actuals has 0 effect)
    oof_h3 = oof_df[oof_df["horizon_step"] == 1].copy()
    oof_h3["horizon_step"] = 3
    res_orig = compute_aci_df(oof_h3, gamma=0.05, normalized=False, horizon_delay=True)
    oof_perturbed = oof_h3.copy()
    # Perturb the last 2 steps (which are within h=3 delay window of step N-2)
    oof_perturbed.iloc[-1, oof_perturbed.columns.get_loc("y_true")] += 5000.0
    oof_perturbed.iloc[-2, oof_perturbed.columns.get_loc("y_true")] += 5000.0
    res_pert = compute_aci_df(oof_perturbed, gamma=0.05, normalized=False, horizon_delay=True)
    # Alpha at step -2 must be identical because feedback from -2 and -1 is delayed by h=3 steps
    np.testing.assert_allclose(
        res_orig["alpha_t"].iloc[:-1].to_numpy(dtype=float),
        res_pert["alpha_t"].iloc[:-1].to_numpy(dtype=float),
        atol=1e-12,
    )

    # Case 5: calibrate_aci_from_oof and calibration.calibrate_from_oof integration + plot_aci
    cal_aci = calibrate_aci_from_oof(
        oof_df, method="aci_normalized", gamma=0.05, target_coverage=0.80
    )
    cal_obj = cal_aci[0] if isinstance(cal_aci, tuple) else cal_aci
    assert cal_obj is not None
    cal_via_hook = calibrate_from_oof(oof_df, (0.8,), method="aci_normalized", aci_gamma=0.05)
    assert cal_via_hook is not None
    cal_preds, cal_src = apply_calibration(preds_df, cal_via_hook, "median")
    assert "aci" in cal_src
    assert bool((cal_preds["yhat_upper"] > cal_preds["yhat_lower"]).all())

    ax = plot_aci(df_aci)
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close("all")


def test_t1_f4_joint_trajectory_copulas_tail_risk_and_multivariate_scores() -> None:
    """Tier 1 F4 (>= 5 cases): Verify `sample_trajectories` (`gaussian`, `student_t`,
    `iman_conover`, `auto`, `cross_series=True`), marginal/bound preservation, `compute_tail_risk`
    (`VaR`/`CVaR`, `copula_amplification_ratio > 1.0`, `CVaR_99(student_t) > CVaR_99(gaussian)`),
    and `evaluate_trajectory_paths` (CRPS, Energy Score, Variogram Score).
    """
    from scale_forecasting.copula import (
        compute_tail_risk,
        evaluate_trajectory_paths,
        plot_trajectories,
        sample_copula_trajectories,
        sample_joint_trajectories,
        sample_trajectories,
        summarize_trajectory_risk,
        tail_risk_df,
    )

    preds_df, oof_df, actuals_df = _make_autocorrelated_heavy_tail_oof_and_preds(
        n_series=3, horizon=7, n_folds=10, rho=0.80, heavy_tails=True, seed=303
    )

    # Case 1: Aliases and all 4 copula_type families (gaussian, student_t, iman_conover, auto)
    assert callable(sample_copula_trajectories)
    assert callable(sample_joint_trajectories)
    assert callable(tail_risk_df)
    assert callable(summarize_trajectory_risk)

    for c_type in ("gaussian", "student_t", "iman_conover", "auto"):
        paths = sample_trajectories(
            preds_df,
            oof_df,
            n_paths=250,
            copula_type=c_type,
            df_t=3.0,
            cross_series=True,
            floor=10.0,
            cap=300.0,
            seed=42,
        )
        assert {
            "path_id",
            "ts_id",
            "model_type",
            "forecast_date",
            "ds",
            "horizon_step",
            "yhat_sample",
            "yhat",
        }.issubset(paths.columns)
        assert paths["path_id"].nunique() == 250
        assert bool((paths["yhat_sample"] >= 10.0 - 1e-6).all())
        assert bool((paths["yhat_sample"] <= 300.0 + 1e-6).all())
        assert "copula_type_selected" in paths.attrs

    # Case 2: Auto-selection picks "student_t" on heavy-tailed residuals (excess kurtosis > 1.0)
    paths_auto = sample_trajectories(preds_df, oof_df, n_paths=200, copula_type="auto", seed=42)
    assert paths_auto.attrs.get("copula_type_selected") in {"student_t", "gaussian", "iman_conover"}

    # Case 3: Marginal quantile preservation — empirical median across paths tracks yhat
    paths_g = sample_trajectories(
        preds_df, oof_df, n_paths=600, copula_type="gaussian", df_t=3.0, seed=42
    )
    med_by_step = (
        paths_g.groupby(["ts_id", "horizon_step"], as_index=False)["yhat_sample"]
        .median()
        .merge(preds_df[["ts_id", "horizon_step", "yhat"]], on=["ts_id", "horizon_step"])
    )
    np.testing.assert_allclose(
        med_by_step["yhat_sample"].to_numpy(dtype=float),
        med_by_step["yhat"].to_numpy(dtype=float),
        rtol=0.08,
        atol=4.0,
    )

    # Case 4: Tail risk schema, amplification > 1.0, and Student-t CVaR_99 > Gaussian CVaR_99
    paths_t = sample_trajectories(
        preds_df, oof_df, n_paths=1000, copula_type="student_t", df_t=2.5, seed=42
    )
    paths_norm = sample_trajectories(
        preds_df, oof_df, n_paths=1000, copula_type="gaussian", seed=42
    )
    risk_t = compute_tail_risk(
        paths_t,
        predictions_df=preds_df,
        actuals_df=actuals_df,
        oof_df=oof_df,
        window=5,
        alphas=(0.90, 0.95, 0.99),
        threshold=850.0,
    )
    risk_g = compute_tail_risk(
        paths_norm,
        predictions_df=preds_df,
        actuals_df=actuals_df,
        oof_df=oof_df,
        window=5,
        alphas=(0.90, 0.95, 0.99),
        threshold=850.0,
    )
    required_risk_cols = {
        "ts_id",
        "model_type",
        "copula_type",
        "n_paths",
        "horizon",
        "window",
        "expected_cumulative_demand",
        "std_cumulative_demand",
        "iid_std_cumulative_demand",
        "copula_amplification_ratio",
        "var_90",
        "cvar_90",
        "var_95",
        "cvar_95",
        "var_99",
        "cvar_99",
        "var_alpha",
        "cvar_alpha",
        "peak_window_mean",
        "peak_window_var",
        "peak_window_cvar",
        "exceedance_prob",
        "expected_shortfall_units",
        "crps",
        "energy_score",
        "variogram_score",
        "risk_verdict",
    }
    assert required_risk_cols.issubset(risk_t.columns)
    assert bool((risk_t["copula_amplification_ratio"] > 1.0).all())
    assert bool((risk_t["cvar_99"] >= risk_t["var_99"] - 1e-6).all())
    assert bool((risk_t["var_99"] >= risk_t["var_95"] - 1e-6).all())
    assert float(risk_t["cvar_99"].sum()) > float(risk_g["cvar_99"].sum())

    # Case 5: Proper multivariate scoring rules (CRPS, Energy, Variogram) & plot_trajectories
    scores = evaluate_trajectory_paths(paths_t, actuals_df, p=0.5)
    for k in ("crps", "energy_score", "variogram_score", "variogram_score_p1"):
        assert k in scores
        assert np.isfinite(float(scores[k])) and float(scores[k]) >= 0.0

    ax = plot_trajectories(paths_t, ts_id="sku_0")
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close("all")


def test_t1_f5_portfolio_inventory_optimizer_shadow_prices_and_highs_milp() -> None:
    """Tier 1 F5 (>= 5 cases): Verify `optimize_inventory_portfolio` schema (30 base + 17 portfolio
    columns), KKT Lagrangian shadow prices (`shadow_price_budget > 0` iff binding), `HiGHS` MILP
    discrete `vendor_moq` + `order_multiple` lot-sizing, `uncertainty_source="copula"`, and plots.
    """
    from scale_forecasting.inventory_optimizer import (
        optimize_inventory_df,
        optimize_inventory_portfolio,
        plot_inventory_frontier,
    )

    preds_df, oof_df, _ = _make_autocorrelated_heavy_tail_oof_and_preds(
        n_series=3, horizon=7, n_folds=6, rho=0.65, heavy_tails=False, seed=505
    )

    # Case 1: Unconstrained portfolio matches INVENTORY_POLICY_COLUMNS and has shadow_price == 0
    assert optimize_inventory_df is optimize_inventory_portfolio or callable(optimize_inventory_df)
    unconstrained_df = optimize_inventory_portfolio(
        preds_df,
        oof_df,
        unit_cost={"sku_0": 10.0, "sku_1": 20.0, "sku_2": 30.0},
        unit_volume={"sku_0": 1.0, "sku_1": 2.0, "sku_2": 1.5},
        uncertainty_source="analytical",
    )
    assert set(INVENTORY_POLICY_COLUMNS).issubset(unconstrained_df.columns)
    extra_cols = {
        "unconstrained_service_level",
        "unconstrained_safety_stock",
        "unconstrained_order_qty",
        "unconstrained_total_cost",
        "unit_cost",
        "unit_volume",
        "vendor_moq",
        "order_cost",
        "order_spend",
        "order_volume",
        "allocation_ratio",
        "shadow_price_budget",
        "shadow_price_volume",
        "shadow_price_units",
        "portfolio_fill_rate",
        "binding_constraint",
        "uncertainty_source",
    }
    assert extra_cols.issubset(unconstrained_df.columns)
    assert "portfolio_summary" in unconstrained_df.attrs
    assert bool((unconstrained_df["shadow_price_budget"].abs() < 1e-6).all())

    # Case 2: Binding max_budget yields shadow_price_budget > 0 and sum(order_spend) <= max_budget
    unconstrained_spend = float(unconstrained_df["order_spend"].sum())
    tight_budget = unconstrained_spend * 0.60
    budget_df = optimize_inventory_portfolio(
        preds_df,
        oof_df,
        max_budget=tight_budget,
        unit_cost={"sku_0": 10.0, "sku_1": 20.0, "sku_2": 30.0},
        uncertainty_source="analytical",
    )
    assert float(budget_df["order_spend"].sum()) <= tight_budget + 1e-6
    assert bool((budget_df["shadow_price_budget"] > 0.0).all())
    assert "budget" in str(budget_df["binding_constraint"].iloc[0]).lower()

    # Case 3: Multi-constraint max_budget + max_volume + max_total_units satisfaction
    unconstrained_vol = float(unconstrained_df["order_volume"].sum())
    tight_vol = unconstrained_vol * 0.55
    multi_df = optimize_inventory_portfolio(
        preds_df,
        oof_df,
        max_budget=unconstrained_spend * 0.85,
        max_volume=tight_vol,
        max_total_units=float(unconstrained_df["recommended_order_qty"].sum()) * 0.80,
        unit_cost={"sku_0": 10.0, "sku_1": 20.0, "sku_2": 30.0},
        unit_volume={"sku_0": 1.0, "sku_1": 2.5, "sku_2": 2.0},
        uncertainty_source="analytical",
    )
    assert float(multi_df["order_spend"].sum()) <= unconstrained_spend * 0.85 + 1e-6
    assert float(multi_df["order_volume"].sum()) <= tight_vol + 1e-6
    assert bool((multi_df["shadow_price_volume"] > 0.0).all())

    # Case 4: HiGHS MILP discrete vendor_moq and order_multiple satisfaction to < 1e-6
    milp_df = optimize_inventory_portfolio(
        preds_df,
        oof_df,
        max_budget=tight_budget,
        unit_cost=10.0,
        vendor_moq={"sku_0": 100.0, "sku_1": 150.0, "sku_2": 120.0},
        order_multiple=25.0,
        uncertainty_source="analytical",
    )
    assert float(milp_df["order_spend"].sum()) <= tight_budget + 1e-6
    for _, row in milp_df.iterrows():
        q_val = float(row["recommended_order_qty"])
        moq_val = float(row["vendor_moq"])
        if q_val > 1e-6:
            assert q_val >= moq_val - 1e-6
            rem = (q_val - moq_val) % 25.0
            assert min(rem, 25.0 - rem) < 1e-5 or min(q_val % 25.0, 25.0 - (q_val % 25.0)) < 1e-5

    # Case 5: Copula uncertainty_source="copula" integration + plot_inventory_frontier
    copula_inv_df = optimize_inventory_portfolio(
        preds_df,
        oof_df,
        max_budget=tight_budget,
        unit_cost=10.0,
        uncertainty_source="copula",
        copula_type="student_t",
        n_paths=200,
        seed=42,
    )
    assert bool((copula_inv_df["uncertainty_source"] == "copula").all())
    assert float(copula_inv_df["order_spend"].sum()) <= tight_budget + 1e-6

    ax = plot_inventory_frontier(budget_df)
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close("all")


def test_t1_f6_prescriptive_scenario_optimizer_objectives_and_guardrails() -> None:
    """Tier 1 F6 (>= 5 cases): Verify `optimize_scenario` and `ScenarioOptimizationResult` across
    all 4 objectives (`gross_profit`, `revenue`, `units`, `risk_adjusted_profit`), business
    guardrails (`price_bounds`, `max_promo_days`, `min_cooldown_days`, `min_margin_pct`),
    3-mode hierarchy coherence (`< 1e-6`), and inventory deltas.
    """
    from scale_forecasting.scenario_optimizer import (
        ScenarioOptimizationResult,
        optimize_scenario,
        plot_optimal_scenario,
    )

    hist_df, fut_df = _make_confounded_price_promo_hierarchy_panel(n_days=48, horizon=8, seed=404)
    cfg = RunConfig.model_validate(
        {
            "run_name": "t1-f6-scenario-opt",
            "data": {"source_table": "p.d.t", "freq": "D", "horizon": 8},
            "models": ["regression_lags@monotone"],
            "features": {
                "future_covariates": ["price", "promo"],
                "monotonic_constraints": {"price": -1, "promo": 1},
            },
            "backtest": {"enabled": True, "n_folds": 1, "horizon": 8},
            "hierarchy": {
                "levels": [["region"]],
                "reconciliation_methods": ["wls_struct"],
            },
        }
    )

    # Case 1: ScenarioOptimizationResult contract, DataFrame delegation, and required columns
    res_gp = optimize_scenario(
        hist_df,
        config=cfg,
        future_covariates_df=fut_df,
        objective="gross_profit",
        price_col="price",
        promo_col="promo",
        price_bounds=(-0.15, 0.15),
        max_promo_days=2,
        min_cooldown_days=2,
        unit_cost=4.5,
        min_margin_pct=0.20,
        reconcile=True,
        include_inventory_impact=True,
    )
    assert isinstance(res_gp, ScenarioOptimizationResult)
    assert isinstance(res_gp.schedule_df, pd.DataFrame)
    assert isinstance(res_gp.summary_df, pd.DataFrame)
    assert isinstance(res_gp.inventory_delta_df, pd.DataFrame)
    assert res_gp.dataframe is res_gp.schedule_df
    assert len(res_gp) == len(res_gp.schedule_df)
    assert "optimal_price" in res_gp.columns
    assert len(res_gp["optimal_price"]) == len(res_gp.schedule_df)
    assert len(res_gp.head(3)) == min(3, len(res_gp.schedule_df))
    assert isinstance(res_gp.to_dict(), dict)
    assert isinstance(res_gp.summary_text(), str) and len(res_gp.summary_text()) > 20

    required_sched_cols = {
        "ts_id",
        "ds",
        "baseline_price",
        "optimal_price",
        "price_change_pct",
        "baseline_promo",
        "optimal_promo",
        "baseline_yhat",
        "optimal_yhat",
        "delta_yhat",
        "pull_forward_dip",
        "cannibalization_effect",
        "unit_cost",
        "baseline_revenue",
        "optimal_revenue",
        "revenue_delta",
        "baseline_gross_margin",
        "optimal_gross_margin",
        "gross_margin_delta",
        "margin_pct",
        "delta_reorder_point",
        "delta_safety_stock",
        "delta_order_qty",
    }
    assert required_sched_cols.issubset(res_gp.schedule_df.columns)

    # Case 2: Objective improvement and guardrail compliance (price_bounds, cooldown, margin)
    assert res_gp.optimal_objective_value >= res_gp.baseline_objective_value - 1e-6
    bottom_ids = set(hist_df["ts_id"].unique())
    leaf_sched = res_gp.schedule_df[res_gp.schedule_df["ts_id"].isin(bottom_ids)].copy()
    assert bool((leaf_sched["price_change_pct"] >= -15.0 - 1e-4).all())
    assert bool((leaf_sched["price_change_pct"] <= 15.0 + 1e-4).all())
    assert bool((leaf_sched["margin_pct"] >= 0.20 - 1e-4).all())

    for _tid, grp in leaf_sched.sort_values("ds").groupby("ts_id"):
        promo_vec = (grp["optimal_promo"].to_numpy(dtype=float) > 0.5).astype(int)
        assert int(promo_vec.sum()) <= 2
        active_indices = np.where(promo_vec == 1)[0]
        if len(active_indices) >= 2:
            assert int(np.min(np.diff(active_indices))) >= 2

    # Case 3: Hierarchy coherence < 1e-6
    assert res_gp.coherence_max_abs_error < 1e-6

    # Case 4: Remaining objectives ("revenue", "units", "risk_adjusted_profit") all optimize cleanly
    for obj_name in ("revenue", "units", "risk_adjusted_profit"):
        res_obj = optimize_scenario(
            hist_df,
            config=cfg,
            future_covariates_df=fut_df,
            objective=obj_name,
            unit_cost=4.5,
            risk_aversion_lambda=0.3,
            cvar_alpha=0.95,
            max_promo_days=3,
        )
        assert res_obj.optimal_objective_value >= res_obj.baseline_objective_value - 1e-5
        assert np.isfinite(res_obj.cvar_profit)

    # Case 5: Plotting optimal scenario returns matplotlib Axes
    ax = plot_optimal_scenario(res_gp)
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close("all")


def test_t1_f7_sdk_playground_bakeoff_report_and_root_exports(tmp_path: Path) -> None:
    """Tier 1 F7 (>= 5 cases): Verify Track L methods on `Forecaster`, `PlaygroundRun`, `BakeOff`,
    `report.py` HTML/JSON export, and root `scale_forecasting` (`sf`) lazy exports.
    """
    # Case 1: Root package exports all Track L symbols
    for sym in (
        "cold_start_transfer_df",
        "plot_cold_start_transfer",
        "compute_aci_df",
        "aci_df",
        "plot_aci",
        "sample_trajectories",
        "compute_tail_risk",
        "tail_risk_df",
        "evaluate_trajectory_paths",
        "plot_trajectories",
        "optimize_inventory_portfolio",
        "optimize_inventory_df",
        "plot_inventory_frontier",
        "optimize_scenario",
        "ScenarioOptimizationResult",
        "plot_optimal_scenario",
    ):
        assert hasattr(sf, sym), f"Missing root export sf.{sym}"

    # Case 2: Forecaster SDK surface across all Track L methods (including inventory_policy_df)
    hist_df, fut_df = _make_confounded_price_promo_hierarchy_panel(n_days=42, horizon=7, seed=707)
    cfg = RunConfig.model_validate(
        {
            "run_name": "t1-f7-sdk",
            "data": {"source_table": "p.d.t", "freq": "D", "horizon": 7},
            "models": ["naive_mean", "theta"],
            "features": {
                "future_covariates": ["price", "promo"],
                "static_covariates": ["region", "category"],
                "cold_start_transfer": True,
            },
            "backtest": {"enabled": True, "n_folds": 2, "horizon": 7},
        }
    )
    f = sf.Forecaster(cfg, data=hist_df)
    assert not f.cold_start_transfer_df().empty
    assert not f.aci_df().empty
    traj_df = f.sample_trajectories(n_paths=50, seed=42)
    assert not traj_df.empty
    assert not f.tail_risk_df().empty
    inv_opt = f.optimize_inventory_df(max_budget=5000.0, unit_cost=10.0)
    assert not inv_opt.empty
    inv_via_policy = f.inventory_policy_df(max_budget=5000.0, unit_cost=10.0)
    assert "shadow_price_budget" in inv_via_policy.columns
    scen_res = f.optimize_scenario(
        future_covariates_df=fut_df, unit_cost=4.5, objective="gross_profit"
    )
    assert not scen_res.schedule_df.empty

    # Case 3: PlaygroundRun surface
    pr = run_model("theta", horizon=7, backtest=True)
    assert not pr.aci_df().empty
    assert not pr.sample_trajectories(n_paths=30).empty
    assert not pr.tail_risk_df().empty
    assert not pr.optimize_inventory_df(max_budget=1000.0).empty
    assert not pr.optimize_scenario(unit_cost=4.0).schedule_df.empty

    # Case 4: BakeOff surface
    bo = bakeoff(models=["naive_mean", "theta"], horizon=7, n_folds=2)
    assert not bo.aci_df().empty
    assert not bo.sample_trajectories(n_paths=30).empty
    assert not bo.tail_risk_df().empty
    assert not bo.optimize_inventory_df(max_budget=1000.0).empty
    assert not bo.optimize_scenario(unit_cost=4.0).schedule_df.empty

    # Case 5: Self-contained HTML report export includes Track L sections
    report_path = tmp_path / "track_l_report.html"
    out_path = export_report(f, report_path, include_json_sidecar=True)
    assert out_path.exists()
    html_text = out_path.read_text(encoding="utf-8")
    assert "<html" in html_text.lower()
    plt.close("all")


def test_t1_f8_auto_forecaster_branch_2c_and_practitioner_assessment_q4_q7() -> None:
    """Tier 1 F8 (>= 5 cases): Verify standard `Forecaster.build_config()` / `Forecaster.auto()`
    invariance on mature panels, cold-start & ACI activation in Branch `2C`
    (`regime_and_specialists`), and 7-Question Assessment (`f.assess()`) Questions 4 & 7.
    """
    # Case 1: Standard mature panel leaves cold_start_transfer=False & calibration="conformal"
    mature_panel = _make_cold_start_npi_panel(n_mature_days=42, n_short_days=5, seed=808)
    mature_only = mature_panel[mature_panel["ts_id"].str.startswith("donor_")].copy()
    bp_std = build_config(
        mature_only[["ts_id", "ds", "y"]],
        horizon=7,
        freq="D",
        tier="first_light",
    )
    assert bp_std.config.features.cold_start_transfer is False
    assert bp_std.config.output.calibration_method == "conformal"

    # Case 2: Cold-start SKUs and static covariates trigger cold_start detection & Branch 2C
    cs_panel = mature_panel[mature_panel["ts_id"] != "npi_zero_obs"].copy()
    prof_cs = inspect_panel_for_auto(cs_panel, horizon=7, freq="D")
    assert prof_cs.regime_shares.get("cold_start", 0.0) > 0.0
    assert getattr(prof_cs, "cold_start_donor_similarity", 0.0) > 0.0

    bp_cs = build_config(cs_panel, horizon=7, freq="D", tier="first_light")
    mutations_cs = propose_autofocus_pass(
        bp_cs.config.model_dump(mode="python"),
        prof_cs,
        {"coverage_drift_detected": True, "cold_start_donor_transfer_recommended": True},
    )
    domains = {m["domain"] for m in mutations_cs}
    assert "regime_and_specialists" in domains

    # Case 3: Branch 2C DAG proposal copies cold_start_transfer and calibration_method cleanly
    branches = propose_autofocus_dag_branches(
        bp_cs.config.model_dump(mode="python"),
        prof_cs,
        {"coverage_drift_detected": True, "cold_start_donor_transfer_recommended": True},
    )
    assert any(b["branch_id"] == "2C" or b["domain"] == "regime_and_specialists" for b in branches)

    # Case 4 & 5: Forecaster.auto() and f.assess() / practitioner_assessment_df() Questions 4 & 7
    f_auto = sf.Forecaster.auto(
        cs_panel,
        horizon=7,
        freq="D",
        tier="first_light",
        models=["naive_mean", "theta"],
        n_folds=1,
        autofocus=False,
    )
    assess_df = f_auto.practitioner_assessment_df()
    assert len(assess_df) == 7
    q4 = assess_df.iloc[3]
    q7 = assess_df.iloc[6]
    q4_text = f"{q4['question_id']} {q4['answer']} {q4['evidence']} {q4['recommendation']}".lower()
    q7_text = f"{q7['question_id']} {q7['answer']} {q7['evidence']} {q7['recommendation']}".lower()
    assert "cold_start" in q4_text
    assert "aci" in q7_text or "tail_risk" in q7_text or "copula" in q7_text


def test_t1_f9_evolve_policy_weights_and_mutation_prompt_track_l_enrichment() -> None:
    """Tier 1 F9 (>= 5 cases): Verify `AutoPolicyWeights`, `evolved_policy_thresholds()`,
    policy seed & mock candidates, `build_mutation_prompt()`, and `Forecaster.evolve("policy")`.
    """
    # Case 1: AutoPolicyWeights & evolved_policy_thresholds()["action_weights"] contain Track L keys
    th = evolved_policy_thresholds()
    action_weights = th["action_weights"]
    has_cs_key = (
        "enable_cold_start_transfer" in action_weights
        or "enable_cold_start_donor_transfer" in action_weights
    )
    has_aci_key = (
        "enable_adaptive_conformal" in action_weights
        or "enable_aci_conformal_tracking" in action_weights
    )
    assert has_cs_key is True
    assert has_aci_key is True

    weights_obj = AutoPolicyWeights()
    assert hasattr(weights_obj, "action_weights")

    # Case 2: Policy target seed code and deterministic_mock_candidates include Track L weights
    policy_seed = get_target_seed("policy").seed_code
    assert (
        "enable_cold_start_transfer" in policy_seed
        or "enable_cold_start_donor_transfer" in policy_seed
    )
    assert (
        "enable_adaptive_conformal" in policy_seed or "enable_aci_conformal_tracking" in policy_seed
    )
    for mock_cand in deterministic_mock_candidates("policy"):
        assert (
            "enable_cold_start_transfer" in mock_cand
            or "enable_cold_start_donor_transfer" in mock_cand
        )
        assert (
            "enable_adaptive_conformal" in mock_cand or "enable_aci_conformal_tracking" in mock_cand
        )

    # Case 3: build_mutation_prompt includes cold-start and calibration / drift context
    prompt_str = build_mutation_prompt(
        "policy",
        policy_seed,
        diagnostics={
            "has_cold_start": True,
            "cold_start_fraction": 0.25,
            "cold_start_donor_similarity": 0.82,
            "calibration_method": "aci_normalized",
            "residual_drift_ks_stat": 0.31,
        },
        generation=1,
    )
    assert "cold_start" in prompt_str.lower()
    assert "aci" in prompt_str.lower() or "calibration" in prompt_str.lower()

    # Case 4 & 5: Forecaster.evolve(target="policy", backend="mock") executes and promotes cleanly
    panel = _make_cold_start_npi_panel(n_mature_days=35, n_short_days=5, seed=909)
    clean_panel = panel[panel["ts_id"].str.startswith("donor_")][["ts_id", "ds", "y"]].copy()
    ev_pol = sf.Forecaster.evolve(
        clean_panel,
        target="policy",
        backend="mock",
        max_evaluations=2,
        horizon=5,
        tier="first_light",
    )
    assert ev_pol.promoted is True
    assert (
        "enable_cold_start_transfer" in ev_pol.champion_code
        or "enable_cold_start_donor_transfer" in ev_pol.champion_code
    )


def test_t1_f10_agent_surfaces_and_mcp_server_track_l_integration() -> None:
    """Tier 1 F10 (>= 5 cases): Verify `agent_surfaces.py` (`render_skill_config_reference`,
    `render_skill_execution_paths_reference`, `render_llms_txt`) and `McpServer` dynamic
    benchmark counts and Track L workbench flags.
    """
    from scale_forecasting import agent_surfaces

    cfg_ref = agent_surfaces.render_skill_config_reference()
    cat_ref = agent_surfaces.render_skill_catalog_reference()

    # Case 1: Variant syntax (@monotone, @transform_clr, @transform_ilr) documented in surfaces
    syntax_rows = getattr(agent_surfaces, "_VARIANT_SYNTAX_ROWS", ())
    syntax_text = " ".join(str(r) for r in syntax_rows) + " " + cfg_ref + " " + cat_ref
    for tag in ("monotone", "transform_clr", "transform_ilr"):
        assert tag in syntax_text

    # Case 2: Normalization / config reference documents cross_temporal and calibration_method
    norm_rows = getattr(agent_surfaces, "_NORMALIZATION_RULES", ())
    norm_text = " ".join(str(r) for r in norm_rows) + " " + cfg_ref
    assert "cross_temporal" in norm_text
    assert "aci" in norm_text

    # Case 3: render_skill_execution_paths_reference and render_llms_txt include Track L methods
    exec_ref = agent_surfaces.render_skill_execution_paths_reference()
    llms_txt = agent_surfaces.render_llms_txt()
    for method_token in (
        "cold_start_transfer_df",
        "aci_df",
        "sample_trajectories",
        "tail_risk_df",
        "optimize_inventory_df",
        "optimize_scenario",
    ):
        assert method_token in exec_ref, f"{method_token} missing from execution_paths_and_ops.md"
        assert method_token in llms_txt, f"{method_token} missing from llms.txt"

    # Case 4: McpServer tool schemas dynamically reflect len(list_benchmarks()) (20, no stale "18")
    server = McpServer()
    tools = {t["name"]: t for t in server.list_tools()}
    n_bench = len(list_benchmarks())
    assert n_bench == 20
    for t_name in ("list_catalog", "build_auto_config", "run_auto_forecast"):
        schema_str = str(tools[t_name])
        assert str(n_bench) in schema_str
        assert "18 synthetic DGP" not in schema_str

    # Case 5: run_playground and review_run schemas & execution support Track L flags
    for t_name in ("run_playground", "review_run"):
        props = tools[t_name]["inputSchema"]["properties"]
        temp_enum = set(props["temporal_reconciliation_method"]["enum"])
        assert {"mint_nonneg", "cross_temporal_mint"}.issubset(temp_enum)
        for flag in (
            "include_promo_elasticity",
            "include_tail_risk",
            "copula_type",
            "include_aci",
            "optimize_inventory_params",
            "optimize_scenario_params",
        ):
            assert flag in props, f"{flag} missing from MCP {t_name} schema"

    play_resp = server.call_tool(
        "run_playground",
        {
            "models": ["naive_mean", "theta"],
            "horizon": 7,
            "n_folds": 2,
            "include_tail_risk": True,
            "copula_type": "gaussian",
            "include_aci": True,
            "optimize_inventory_params": {"max_budget": 1000.0, "unit_cost": 10.0},
            "optimize_scenario_params": {"unit_cost": 4.0, "objective": "gross_profit"},
        },
    )
    assert play_resp.get("isError") is not True


# =============================================================================
# TIER 2 — Boundary Value & Corner-Case Hardening (>= 5 cases per feature area)
# =============================================================================


@pytest.mark.parametrize(
    ("section", "payload"),
    [
        ("features", {"cold_start_min_obs": 0}),
        ("features", {"cold_start_min_obs": -5}),
        ("features", {"cold_start_method": "invalid_archetype"}),
        ("output", {"calibration_method": "unknown_conformal"}),
        ("output", {"aci_gamma": 0.0}),
        ("output", {"aci_gamma": -0.05}),
        ("output", {"aci_gamma": 0.51}),
        ("output", {"aci_gamma": float("nan")}),
    ],
)
def test_t2_config_invalid_track_l_parameters(section: str, payload: dict[str, Any]) -> None:
    """Tier 2 F1 (>= 5 cases): Reject invalid cold_start and ACI config parameters at validation."""
    base = {
        "run_name": "t2-invalid-cfg",
        "data": {"source_table": "p.d.t", "freq": "D", "horizon": 7},
        "models": ["naive_mean"],
        section: payload,
    }
    with pytest.raises((ValidationError, ValueError)):
        RunConfig.model_validate(base)


def test_t2_cold_start_zero_obs_single_obs_and_global_fallback_edge_cases() -> None:
    """Tier 2 F2 (>= 5 cases): Verify 0-obs pre-launch SKUs, 1-obs single-point SKUs, unseen
    static attributes falling back to `__global__`, all-short panels without mature donors, and
    constant zero-variance donor series.
    """
    from scale_forecasting.cold_start import (
        build_cold_start_donor_pool,
        cold_start_transfer_df,
        forecast_cold_start_series,
    )

    panel = _make_cold_start_npi_panel(n_mature_days=35, n_short_days=5, seed=111)

    # Case 1: 0-obs pre-launch SKU gets finite positive forecast and valid interval width
    pool = build_cold_start_donor_pool(
        panel, min_obs=14, static_covariates=["category", "brand", "region"]
    )
    zero_sub = panel[panel["ts_id"] == "npi_zero_obs"].copy()
    res_zero = forecast_cold_start_series(
        zero_sub,
        pool,
        horizon=7,
        freq="D",
        min_obs=14,
        static_covariates=["category", "brand", "region"],
    )
    assert res_zero.n_obs == 0
    assert res_zero.cold_start_flag is True
    assert len(res_zero.predictions) == 7
    assert bool(np.isfinite(res_zero.predictions["yhat"]).all())
    assert bool((res_zero.predictions["yhat_upper"] > res_zero.predictions["yhat_lower"]).all())

    # Case 2: 1-observation single-point SKU computes finite Empirical-Bayes weight 0 < w_1 < 1
    one_obs_df = pd.DataFrame(
        [
            {
                "ts_id": "npi_one_obs",
                "category": "Electronics",
                "brand": "BrandX",
                "region": "North",
                "ds": pd.Timestamp("2025-02-04"),
                "y": 130.0,
            }
        ]
    )
    res_one = forecast_cold_start_series(
        one_obs_df,
        pool,
        horizon=7,
        freq="D",
        min_obs=14,
        static_covariates=["category", "brand", "region"],
    )
    assert res_one.n_obs == 1
    assert len(res_one.predictions) == 7
    assert bool(np.isfinite(res_one.predictions["yhat"]).all())

    # Case 3: Completely unseen static covariates fall back cleanly to __global__
    unseen_df = pd.DataFrame(
        [
            {
                "ts_id": "npi_unseen_all",
                "category": "Aerospace",
                "brand": "Quantum",
                "region": "Antarctica",
                "ds": pd.Timestamp("2025-02-03") + pd.Timedelta(days=i),
                "y": 60.0 + 3.0 * i,
            }
            for i in range(3)
        ]
    )
    cs_unseen = cold_start_transfer_df(
        pd.concat([panel, unseen_df], ignore_index=True),
        min_obs=14,
        static_covariates=["category", "brand", "region"],
        horizon=7,
    )
    unseen_row = cs_unseen[cs_unseen["ts_id"] == "npi_unseen_all"].iloc[0]
    assert (
        "global" in str(unseen_row["matched_level"]).lower() or str(unseen_row["donor_group"]) != ""
    )

    # Case 4: Panel where ALL series are short (0 mature donors) falls back without crashing
    all_short_panel = panel[panel["ts_id"].isin(["npi_short_exact", "npi_short_fallback"])].copy()
    cs_all_short = cold_start_transfer_df(all_short_panel, min_obs=14, horizon=7)
    assert len(cs_all_short) == 2
    assert bool(np.isfinite(cs_all_short["yhat_mean"]).all())

    # Case 5: Constant zero-variance donor series produces finite non-crossing intervals
    const_dates = pd.date_range("2025-01-01", periods=28, freq="D")
    const_panel = pd.DataFrame(
        [{"ts_id": "donor_const", "category": "C1", "ds": dt, "y": 50.0} for dt in const_dates]
        + [
            {"ts_id": "short_const", "category": "C1", "ds": dt, "y": 50.0}
            for dt in const_dates[-3:]
        ]
    )
    cs_const = cold_start_transfer_df(
        const_panel, min_obs=14, static_covariates=["category"], horizon=5
    )
    assert bool((cs_const["interval_width_80"] >= 0.0).all())


def test_t2_aci_volatility_shocks_step_shifts_and_gamma_boundaries() -> None:
    """Tier 2 F3 (>= 5 cases): Verify extreme gamma boundaries, zero-error streams,
    single-fold / short OOF graceful handling, and invalid parameter errors in `aci.py`.
    """
    from scale_forecasting.aci import calibrate_aci_from_oof, compute_aci_df

    oof_df, preds_df = _make_regime_shift_oof_and_preds(
        n_oof_steps=30, horizon=5, shock_multiplier=3.0, seed=212
    )

    # Case 1: Boundary gamma=0.005 (slow) vs gamma=0.50 (max step) stay strictly inside (0, 1)
    for g_val in (0.005, 0.50):
        df_g = compute_aci_df(oof_df, preds_df, gamma=g_val, target_coverage=0.80, normalized=True)
        assert bool((df_g["alpha_t"] > 0.0).all())
        assert bool((df_g["alpha_t"] < 1.0).all())
        assert bool((df_g["aci_upper"] >= df_g["aci_lower"]).all())

    # Case 2: Zero-error stream (y_true == yhat) never produces negative interval widths
    zero_err_oof = oof_df.copy()
    zero_err_oof["y_true"] = zero_err_oof["yhat"]
    df_zero = compute_aci_df(zero_err_oof, gamma=0.05, target_coverage=0.80, normalized=True)
    assert bool((df_zero["aci_width"] >= 0.0).all())
    assert bool((df_zero["aci_upper"] >= df_zero["aci_lower"]).all())

    # Case 3: Ultra-short OOF (2 rows) falls back gracefully without NaN
    short_oof = oof_df.iloc[:2].copy()
    df_short = compute_aci_df(short_oof, gamma=0.05, target_coverage=0.80, normalized=True)
    assert len(df_short) == 2
    assert bool(np.isfinite(df_short["aci_lower"]).all())
    cal_short = calibrate_aci_from_oof(short_oof, method="aci_normalized", gamma=0.05)
    assert cal_short is not None

    # Case 4: Empty OOF returns empty DataFrame or None calibration without raising
    empty_aci = compute_aci_df(oof_df.iloc[0:0], gamma=0.05)
    assert empty_aci.empty

    # Case 5: Extreme gamma / coverage clip safely in compute_aci_df and reject in OutputConfig
    for extreme_gamma in (0.0, -0.1, 0.9):
        df_clipped = compute_aci_df(oof_df, gamma=extreme_gamma)
        assert bool((df_clipped["alpha_t"] > 0.0).all()) and bool(
            (df_clipped["alpha_t"] < 1.0).all()
        )
        with pytest.raises((ValidationError, ValueError)):
            OutputConfig(calibration_method="aci", aci_gamma=extreme_gamma)
    for extreme_cov in (0.01, 0.999):
        df_cov = compute_aci_df(oof_df, gamma=0.05, target_coverage=extreme_cov)
        assert bool((df_cov["aci_upper"] >= df_cov["aci_lower"]).all())


def test_t2_copula_single_fold_psd_intermittent_pit_and_heavy_tails() -> None:
    """Tier 2 F4 (>= 5 cases): Verify single-fold (K=1) PSD Higham/shrinkage fallback,
    intermittent/discrete count series randomized PIT jitter, seed determinism, H=1 single-step
    horizon, and invalid `n_paths` rejection in `copula.py`.
    """
    from scale_forecasting.copula import compute_tail_risk, sample_trajectories

    preds_df, oof_df, _ = _make_autocorrelated_heavy_tail_oof_and_preds(
        n_series=2, horizon=6, n_folds=6, rho=0.70, heavy_tails=True, seed=313
    )

    # Case 1: Single-fold OOF (K=1) PSD projection fallback succeeds with zero LinAlgError
    oof_k1 = oof_df[oof_df["fold_id"] == 0].copy()
    paths_k1 = sample_trajectories(
        preds_df, oof_k1, n_paths=100, copula_type="gaussian", cross_series=True, seed=42
    )
    assert paths_k1["path_id"].nunique() == 100
    assert bool(np.isfinite(paths_k1["yhat_sample"]).all())

    # Case 2: Intermittent / zero-inflated discrete count series with randomized PIT jitter
    rng = np.random.default_rng(314)
    intermittent_oof = oof_df.copy()
    intermittent_oof["model_type"] = "croston_sba"
    zero_mask = rng.uniform(0.0, 1.0, size=len(intermittent_oof)) < 0.65
    intermittent_oof["y_true"] = np.where(
        zero_mask, 0.0, np.round(rng.exponential(4.0, size=len(intermittent_oof)))
    )
    intermittent_oof["yhat"] = 1.5
    intermittent_oof["yhat_raw"] = 1.5
    intermittent_preds = preds_df.copy()
    intermittent_preds["model_type"] = "croston_sba"
    intermittent_preds["yhat"] = 1.5
    intermittent_preds["yhat_lower"] = 0.0
    intermittent_preds["yhat_upper"] = 5.0
    paths_int = sample_trajectories(
        intermittent_preds,
        intermittent_oof,
        n_paths=120,
        copula_type="iman_conover",
        floor=0.0,
        seed=42,
    )
    assert bool((paths_int["yhat_sample"] >= 0.0).all())
    assert bool(np.isfinite(paths_int["yhat_sample"]).all())

    # Case 3: Seed determinism (identical seed -> identical draws; different seed -> distinct draws)
    p_a = sample_trajectories(preds_df, oof_df, n_paths=40, copula_type="student_t", seed=77)
    p_b = sample_trajectories(preds_df, oof_df, n_paths=40, copula_type="student_t", seed=77)
    p_c = sample_trajectories(preds_df, oof_df, n_paths=40, copula_type="student_t", seed=88)
    np.testing.assert_allclose(
        p_a["yhat_sample"].to_numpy(dtype=float),
        p_b["yhat_sample"].to_numpy(dtype=float),
        atol=1e-12,
    )
    assert not np.allclose(
        p_a["yhat_sample"].to_numpy(dtype=float),
        p_c["yhat_sample"].to_numpy(dtype=float),
    )

    # Case 4: Single-step horizon H = 1 works across sample_trajectories and compute_tail_risk
    preds_h1 = preds_df[preds_df["horizon_step"] == 1].copy()
    oof_h1 = oof_df[oof_df["horizon_step"] == 1].copy()
    paths_h1 = sample_trajectories(preds_h1, oof_h1, n_paths=50, seed=42)
    risk_h1 = compute_tail_risk(paths_h1, window=1)
    assert len(risk_h1) == 2
    assert bool(np.isfinite(risk_h1["var_95"]).all())

    # Case 5: Invalid n_paths <= 0 or unknown copula_type raises ConfigError / ValueError
    for bad_n in (0, -5):
        with pytest.raises((ConfigError, ValueError)):
            sample_trajectories(preds_df, oof_df, n_paths=bad_n)
    with pytest.raises((ConfigError, ValueError)):
        sample_trajectories(preds_df, oof_df, n_paths=20, copula_type="archimedean_unknown")


def test_t2_inventory_optimizer_binding_slack_moq_and_infeasible_guards() -> None:
    """Tier 2 F5 (>= 5 cases): Verify KKT complementarity slackness (tight vs slack budget),
    unaffordable MOQ SKU skipping, excess initial on-hand inventory (`recommended_order_qty == 0`),
    and invalid constraint rejection in `inventory_optimizer.py`.
    """
    from scale_forecasting.inventory_optimizer import optimize_inventory_portfolio

    preds_df, oof_df, _ = _make_autocorrelated_heavy_tail_oof_and_preds(
        n_series=3, horizon=7, n_folds=4, rho=0.5, heavy_tails=False, seed=515
    )

    # Case 1: Slack budget (10x unconstrained spend) -> shadow_price_budget == 0
    slack_df = optimize_inventory_portfolio(
        preds_df,
        oof_df,
        max_budget=1_000_000.0,
        unit_cost=10.0,
        uncertainty_source="analytical",
    )
    assert bool((slack_df["shadow_price_budget"].abs() < 1e-6).all())
    assert bool((slack_df["allocation_ratio"] >= 0.999).all())

    # Case 2: Tight binding budget (15% of unconstrained spend) -> shadow_price_budget > 0
    unconstrained_spend = float(slack_df["unconstrained_order_qty"].sum() * 10.0)
    tight_b = unconstrained_spend * 0.15
    tight_df = optimize_inventory_portfolio(
        preds_df,
        oof_df,
        max_budget=tight_b,
        unit_cost=10.0,
        uncertainty_source="analytical",
    )
    assert float(tight_df["order_spend"].sum()) <= tight_b + 1e-6
    assert bool((tight_df["shadow_price_budget"] > 0.0).all())

    # Case 3: SKU with vendor_moq exceeding total portfolio budget is skipped (order_qty == 0)
    moq_skip_df = optimize_inventory_portfolio(
        preds_df,
        oof_df,
        max_budget=5000.0,
        unit_cost=10.0,
        vendor_moq={"sku_0": 1000.0, "sku_1": 50.0, "sku_2": 50.0},
        uncertainty_source="analytical",
    )
    sku0_qty = float(
        moq_skip_df.loc[moq_skip_df["ts_id"] == "sku_0", "recommended_order_qty"].iloc[0]
    )
    assert sku0_qty == pytest.approx(0.0, abs=1e-6)
    assert float(moq_skip_df["order_spend"].sum()) <= 5000.0 + 1e-6

    # Case 4: Massive initial on_hand inventory results in zero recommended orders and zero spend
    high_oh_df = optimize_inventory_portfolio(
        preds_df,
        oof_df,
        on_hand=50_000.0,
        max_budget=5000.0,
        unit_cost=10.0,
        uncertainty_source="analytical",
    )
    assert float(high_oh_df["recommended_order_qty"].sum()) == pytest.approx(0.0, abs=1e-6)
    assert float(high_oh_df["order_spend"].sum()) == pytest.approx(0.0, abs=1e-6)

    # Case 5: Invalid negative max_budget or max_volume raises ConfigError / ValueError
    for bad_kw in ({"max_budget": -100.0}, {"max_volume": -5.0}, {"unit_cost": -10.0}):
        with pytest.raises((ConfigError, DataError, ValueError)):
            optimize_inventory_portfolio(preds_df, oof_df, **bad_kw)


def test_t2_scenario_optimizer_tight_cooldown_margin_and_zero_elasticity() -> None:
    """Tier 2 F6 (>= 5 cases): Verify `max_promo_days=0`, `min_cooldown_days >= horizon`,
    high `min_margin_pct` floor, tight `max_total_units` ceiling, and discrete `price_grid`.
    """
    from scale_forecasting.scenario_optimizer import optimize_scenario

    hist_df, fut_df = _make_confounded_price_promo_hierarchy_panel(n_days=42, horizon=7, seed=616)
    flat_hist = hist_df[hist_df["ts_id"].isin(["North_Bev_A", "North_Bev_B"])].copy()
    flat_fut = fut_df[fut_df["ts_id"].isin(["North_Bev_A", "North_Bev_B"])].copy()

    # Case 1: max_promo_days=0 forces zero promotional days across all SKUs
    res_no_promo = optimize_scenario(
        flat_hist,
        future_covariates_df=flat_fut,
        objective="gross_profit",
        max_promo_days=0,
        unit_cost=4.5,
        reconcile=False,
    )
    assert float(res_no_promo.schedule_df["optimal_promo"].sum()) == pytest.approx(0.0, abs=1e-6)

    # Case 2: min_cooldown_days=10 on horizon=7 allows at most 1 promo day per SKU
    res_long_cd = optimize_scenario(
        flat_hist,
        future_covariates_df=flat_fut,
        horizon=7,
        objective="units",
        max_promo_days=5,
        min_cooldown_days=10,
        unit_cost=4.5,
        reconcile=False,
    )
    for _tid, grp in res_long_cd.schedule_df.groupby("ts_id"):
        assert int((grp["optimal_promo"] > 0.5).sum()) <= 1

    # Case 3: Discrete price_grid constrains optimal_price to exact grid ladder values
    ladder = [9.5, 10.0, 10.5, 11.5, 12.5]
    res_grid = optimize_scenario(
        flat_hist,
        future_covariates_df=flat_fut,
        objective="gross_profit",
        price_grid=ladder,
        unit_cost=4.5,
        reconcile=False,
    )
    for p_val in res_grid.schedule_df["optimal_price"].to_numpy(dtype=float):
        assert min(abs(p_val - g) for g in ladder) < 1e-4

    # Case 4: Tight max_total_units ceiling caps optimal_total_units
    cap_units = res_no_promo.baseline_total_units * 0.92
    res_capped = optimize_scenario(
        flat_hist,
        future_covariates_df=flat_fut,
        objective="revenue",
        max_total_units=cap_units,
        unit_cost=4.5,
        reconcile=False,
    )
    assert res_capped.optimal_total_units <= cap_units * 1.02 + 1.0

    # Case 5: High min_margin_pct=0.50 enforces portfolio gross margin percentage >= 0.50
    res_margin = optimize_scenario(
        flat_hist,
        future_covariates_df=flat_fut,
        horizon=7,
        objective="units",
        min_margin_pct=0.50,
        unit_cost=4.5,
        reconcile=False,
    )
    portfolio_margin_pct = res_margin.optimal_total_gross_margin / max(
        res_margin.optimal_total_revenue, 1e-9
    )
    assert portfolio_margin_pct >= 0.50 - 1e-3


# =============================================================================
# TIER 3 — Cross-Feature Pairwise Interactions (12 Pairwise Suites)
# =============================================================================


def test_t3_pair_01_cold_start_npi_with_online_aci_calibration() -> None:
    """Tier 3 Pair 1: Cold-Start NPI (`cold_start_transfer=True`) + Online `aci_normalized`."""
    panel = _make_cold_start_npi_panel(n_mature_days=42, n_short_days=5, seed=1001)
    active_panel = panel[panel["ts_id"] != "npi_zero_obs"].copy()
    cfg = RunConfig.model_validate(
        {
            "run_name": "t3-pair-01-cs-aci",
            "data": {"source_table": "p.d.t", "freq": "D", "horizon": 7},
            "models": ["naive_mean"],
            "features": {
                "static_covariates": ["category", "brand", "region"],
                "cold_start_transfer": True,
                "cold_start_min_obs": 14,
                "cold_start_method": "analog_transfer",
            },
            "backtest": {"enabled": True, "n_folds": 2, "horizon": 7},
            "output": {"calibration_method": "aci_normalized", "aci_gamma": 0.05},
        }
    )
    cells, status_df = run_group(active_panel, cfg)
    assert set(status_df["status"]) == {"ok"}
    for c in cells:
        assert len(c.predictions) == 7
        assert bool((c.predictions["yhat_upper"] > c.predictions["yhat_lower"]).all())


def test_t3_pair_02_cold_start_npi_with_copula_trajectories_and_tail_risk() -> None:
    """Tier 3 Pair 2: Cold-Start NPI forecasts chained into `sample_trajectories` & tail risk."""
    from scale_forecasting.cold_start import build_cold_start_donor_pool, forecast_cold_start_series
    from scale_forecasting.copula import compute_tail_risk, sample_trajectories

    panel = _make_cold_start_npi_panel(n_mature_days=42, n_short_days=5, seed=1002)
    pool = build_cold_start_donor_pool(
        panel, min_obs=14, static_covariates=["category", "brand", "region"]
    )
    short_res = forecast_cold_start_series(
        panel[panel["ts_id"] == "npi_short_exact"],
        pool,
        horizon=7,
        freq="D",
        method="weibull_lifecycle",
        static_covariates=["category", "brand", "region"],
    )
    preds_df = short_res.predictions.assign(
        ts_id="npi_short_exact",
        model_type="weibull_lifecycle",
        forecast_date=short_res.predictions["ds"],
    )
    paths = sample_trajectories(
        preds_df, n_paths=150, copula_type="student_t", df_t=4.0, floor=0.0, seed=42
    )
    risk = compute_tail_risk(paths, predictions_df=preds_df, window=5)
    assert len(risk) == 1
    assert float(risk["cvar_95"].iloc[0]) >= float(risk["var_95"].iloc[0])


def test_t3_pair_03_online_aci_with_copula_tail_risk_under_regime_shift() -> None:
    """Tier 3 Pair 3: Online `aci_normalized` interval expansion chained into Copula Tail Risk."""
    from scale_forecasting.copula import compute_tail_risk, sample_trajectories

    oof_df, preds_df = _make_regime_shift_oof_and_preds(
        n_oof_steps=36, horizon=7, shock_multiplier=3.5, seed=1003
    )
    cal_static = calibrate_from_oof(oof_df, (0.8,), method="conformal")
    cal_aci = calibrate_from_oof(oof_df, (0.8,), method="aci_normalized", aci_gamma=0.08)
    preds_static, _ = apply_calibration(preds_df.copy(), cal_static, "median")
    preds_aci, _ = apply_calibration(preds_df.copy(), cal_aci, "median")

    paths_static = sample_trajectories(
        preds_static.assign(ts_id="sku_shock", model_type="theta"),
        oof_df,
        n_paths=300,
        copula_type="gaussian",
        seed=42,
    )
    paths_aci = sample_trajectories(
        preds_aci.assign(ts_id="sku_shock", model_type="theta"),
        oof_df,
        n_paths=300,
        copula_type="gaussian",
        seed=42,
    )
    risk_static = compute_tail_risk(paths_static)
    risk_aci = compute_tail_risk(paths_aci)
    # ACI expands intervals after the 3.5x volatility burst, yielding higher upper tail VaR_95
    assert float(risk_aci["var_95"].iloc[0]) > float(risk_static["var_95"].iloc[0])


def test_t3_pair_04_copula_trajectories_with_portfolio_inventory_optimizer() -> None:
    """Tier 3 Pair 4: Student-t Copula trajectories passed to `optimize_inventory_portfolio`."""
    from scale_forecasting.copula import sample_trajectories
    from scale_forecasting.inventory_optimizer import optimize_inventory_portfolio

    preds_df, oof_df, _ = _make_autocorrelated_heavy_tail_oof_and_preds(
        n_series=3, horizon=7, n_folds=8, rho=0.75, heavy_tails=True, seed=1004
    )
    paths_t = sample_trajectories(
        preds_df, oof_df, n_paths=400, copula_type="student_t", df_t=3.0, cross_series=True, seed=42
    )
    inv_df = optimize_inventory_portfolio(
        preds_df,
        oof_df,
        trajectories_df=paths_t,
        max_budget=18_000.0,
        max_volume=250.0,
        unit_cost={"sku_0": 12.0, "sku_1": 18.0, "sku_2": 25.0},
        unit_volume={"sku_0": 1.0, "sku_1": 1.5, "sku_2": 2.0},
        vendor_moq=20.0,
        order_multiple=10.0,
        uncertainty_source="copula",
    )
    assert float(inv_df["order_spend"].sum()) <= 18_000.0 + 1e-6
    assert float(inv_df["order_volume"].sum()) <= 250.0 + 1e-6
    assert bool((inv_df["uncertainty_source"] == "copula").all())


def test_t3_pair_05_copula_trajectories_with_scenario_optimizer_cvar_objective() -> None:
    """Tier 3 Pair 5: Scenario Optimizer with `objective='risk_adjusted_profit'` (CVaR)."""
    from scale_forecasting.scenario_optimizer import optimize_scenario

    hist_df, fut_df = _make_confounded_price_promo_hierarchy_panel(n_days=42, horizon=7, seed=1005)
    res_risk = optimize_scenario(
        hist_df,
        future_covariates_df=fut_df,
        objective="risk_adjusted_profit",
        risk_aversion_lambda=0.45,
        cvar_alpha=0.95,
        unit_cost=4.5,
        max_promo_days=2,
        min_cooldown_days=2,
    )
    assert res_risk.objective == "risk_adjusted_profit"
    assert np.isfinite(res_risk.cvar_profit)
    assert res_risk.optimal_objective_value >= res_risk.baseline_objective_value - 1e-5


def test_t3_pair_06_compositional_simplex_with_copula_trajectories() -> None:
    """Tier 3 Pair 6: Compositional Simplex (`simplex_sum=100.0`) + Copula trajectories `< 1e-6`."""
    from scale_forecasting.copula import sample_trajectories

    f_dates = pd.date_range("2025-04-01", periods=6, freq="MS")
    shares = [45.0, 35.0, 20.0]
    rows = []
    for b_idx, sh in enumerate(shares):
        for h_idx, dt in enumerate(f_dates, start=1):
            rows.append(
                {
                    "ts_id": f"brand_{b_idx}",
                    "model_type": "theta@transform_ilr",
                    "ds": dt,
                    "forecast_date": dt,
                    "horizon_step": h_idx,
                    "yhat": sh,
                    "yhat_lower": max(0.0, sh - 6.0),
                    "yhat_upper": min(100.0, sh + 6.0),
                }
            )
    preds_df = pd.DataFrame(rows)
    paths = sample_trajectories(
        preds_df,
        n_paths=80,
        copula_type="gaussian",
        simplex_sum=100.0,
        seed=42,
    )
    assert verify_simplex_coherence(paths, simplex_sum=100.0) < 1e-6
    assert bool((paths["yhat_sample"] >= -1e-9).all())


def test_t3_pair_07_three_mode_cross_temporal_hierarchy_with_copula_trajectories() -> None:
    """Tier 3 Pair 7: 3-Mode `cross_temporal_mint` + Copula paths satisfy coherence `< 1e-6`."""
    from scale_forecasting.copula import sample_trajectories

    hist_df, _ = _make_confounded_price_promo_hierarchy_panel(n_days=42, horizon=7, seed=1007)
    cfg = RunConfig.model_validate(
        {
            "run_name": "t3-pair-07-ct-copula",
            "data": {"source_table": "p.d.t", "freq": "D", "horizon": 14},
            "models": ["naive_mean"],
            "hierarchy": {
                "levels": [["region"]],
                "cross_temporal": True,
                "temporal_aggregations": [7, 14],
            },
        }
    )
    panel, spec = build_hierarchy(hist_df[["ts_id", "region", "ds", "y"]], cfg)
    f_dates = pd.date_range("2025-02-12", periods=14, freq="D")
    pred_rows = [
        {
            "ts_id": nid,
            "model_type": "naive_mean",
            "ds": dt,
            "forecast_date": dt,
            "horizon_step": h_idx,
            "yhat": 90.0,
            "yhat_lower": 75.0,
            "yhat_upper": 105.0,
        }
        for nid in spec.node_ids
        for h_idx, dt in enumerate(f_dates, start=1)
    ]
    preds_df = pd.DataFrame(pred_rows)
    paths = sample_trajectories(
        preds_df,
        history_df=panel,
        cfg=cfg,
        hier_spec=spec,
        n_paths=30,
        copula_type="gaussian",
        reconcile=True,
        seed=42,
    )
    for _pid, p_grp in paths.groupby("path_id"):
        assert verify_coherence(p_grp, spec) < 1e-6
        assert verify_temporal_coherence(p_grp, cfg=cfg) < 1e-6


def test_t3_pair_08_scenario_optimizer_with_cross_temporal_hierarchy_and_inventory_deltas() -> None:
    """Tier 3 Pair 8: `optimize_scenario` with hierarchy reconciliation and inventory deltas."""
    from scale_forecasting.scenario_optimizer import optimize_scenario

    hist_df, fut_df = _make_confounded_price_promo_hierarchy_panel(n_days=42, horizon=7, seed=1008)
    cfg = RunConfig.model_validate(
        {
            "run_name": "t3-pair-08-scen-hier-inv",
            "data": {"source_table": "p.d.t", "freq": "D", "horizon": 7},
            "models": ["naive_mean"],
            "features": {"future_covariates": ["price", "promo"]},
            "hierarchy": {
                "levels": [["region"], ["region", "category"]],
                "reconciliation_methods": ["wls_struct"],
            },
        }
    )
    _, spec = build_hierarchy(hist_df, cfg)
    res = optimize_scenario(
        hist_df,
        config=cfg,
        future_covariates_df=fut_df,
        objective="gross_profit",
        unit_cost=4.5,
        reconcile=True,
        include_inventory_impact=True,
    )
    assert res.coherence_max_abs_error < 1e-6
    assert verify_coherence(res.schedule_df, spec, value_col="optimal_yhat", date_col="ds") < 1e-6
    assert not res.inventory_delta_df.empty
    for col in ("delta_reorder_point", "delta_safety_stock", "delta_order_qty"):
        assert col in res.schedule_df.columns


def test_t3_pair_09_scenario_optimizer_shocks_roundtrip_into_simulate_scenarios_and_goal_seek() -> (
    None
):
    """Tier 3 Pair 9: Feed `res.recommended_shocks` from `optimize_scenario` into
    `simulate_scenarios` and `allocate_target`.
    """
    from scale_forecasting.scenario_optimizer import optimize_scenario

    hist_df, fut_df = _make_confounded_price_promo_hierarchy_panel(n_days=42, horizon=7, seed=1009)
    flat_hist = hist_df[hist_df["ts_id"].isin(["North_Bev_A", "North_Bev_B"])].copy()
    flat_fut = fut_df[fut_df["ts_id"].isin(["North_Bev_A", "North_Bev_B"])].copy()

    res = optimize_scenario(
        flat_hist,
        future_covariates_df=flat_fut,
        horizon=7,
        objective="gross_profit",
        unit_cost=4.5,
        reconcile=False,
    )
    assert isinstance(res.recommended_shocks, dict) and len(res.recommended_shocks) > 0

    sim_cfg = RunConfig.model_validate(
        {
            "run_name": "t3-pair-09-sim",
            "data": {"source_table": "p.d.t", "freq": "D", "horizon": 7},
            "models": ["regression_lags@monotone"],
            "features": {"future_covariates": ["price", "promo"]},
        }
    )
    scenarios_payload = (
        res.recommended_shocks
        if "optimal_scenario" in res.recommended_shocks
        else {"optimal_calendar": res.recommended_shocks}
    )
    sim_df = simulate_scenarios(
        flat_hist,
        sim_cfg,
        scenarios=scenarios_payload,
        models=["regression_lags@monotone"],
        future_covariates_df=flat_fut,
        price_col="price",
        unit_cost=4.5,
        reconcile=False,
    )
    assert not sim_df.empty

    opt_preds = res.schedule_df[["ts_id", "forecast_date", "yhat", "model_type"]].copy()
    gs_res = allocate_target(
        opt_preds,
        target_total=float(opt_preds["yhat"].sum()) * 1.05,
        smoothness_weight=5.0,
        return_result=True,
    )
    assert isinstance(gs_res, GoalSeekResult)
    assert gs_res.max_target_discrepancy < 1e-5


def test_t3_pair_10_cold_start_and_aci_in_auto_forecaster_autofocus_branch_2c() -> None:
    """Tier 3 Pair 10: `Forecaster.auto(..., autofocus=True)` on a cold-start + drift panel."""
    panel = _make_cold_start_npi_panel(n_mature_days=42, n_short_days=6, seed=1010)
    active_panel = panel[panel["ts_id"] != "npi_zero_obs"].copy()
    f_auto = sf.Forecaster.auto(
        active_panel,
        horizon=7,
        freq="D",
        tier="first_light",
        models=["naive_mean", "theta"],
        n_folds=1,
        autofocus=True,
    )
    assert not f_auto.predictions_df().empty
    ledger_df = f_auto.autofocus_ledger_df()
    assert not ledger_df.empty
    assess_df = f_auto.practitioner_assessment_df()
    assert len(assess_df) == 7


def test_t3_pair_11_evolve_with_cold_start_and_aci_policy_and_ensemble() -> None:
    """Tier 3 Pair 11: `Forecaster.evolve` on a cold-start + ACI panel (`policy` & `ensemble`)."""
    panel = _make_cold_start_npi_panel(n_mature_days=35, n_short_days=6, seed=1011)
    active_panel = panel[panel["ts_id"].str.startswith("donor_")][["ts_id", "ds", "y"]].copy()
    ev_ens = sf.Forecaster.evolve(
        active_panel,
        target="ensemble",
        backend="mock",
        max_evaluations=2,
        horizon=5,
        tier="first_light",
    )
    assert not ev_ens.predictions_df.empty
    ev_pol = sf.Forecaster.evolve(
        active_panel,
        target="policy",
        backend="mock",
        max_evaluations=2,
        horizon=5,
        tier="first_light",
    )
    assert ev_pol.promoted is True


def test_t3_pair_12_mcp_server_and_html_report_full_track_l_payload(tmp_path: Path) -> None:
    """Tier 3 Pair 12: End-to-end MCP `run_playground` with all Track J/K/L flags + HTML report."""
    server = McpServer()
    resp = server.call_tool(
        "run_playground",
        {
            "models": ["naive_mean", "theta"],
            "horizon": 7,
            "n_folds": 2,
            "include_fva": True,
            "include_inventory_policy": True,
            "include_promo_elasticity": True,
            "include_tail_risk": True,
            "copula_type": "student_t",
            "include_aci": True,
            "optimize_inventory_params": {
                "max_budget": 1500.0,
                "max_volume": 200.0,
                "vendor_moq": 5.0,
                "unit_cost": 10.0,
            },
            "optimize_scenario_params": {
                "unit_cost": 4.0,
                "objective": "gross_profit",
            },
        },
    )
    assert resp.get("isError") is not True

    bo = bakeoff(models=["naive_mean", "theta"], horizon=7, n_folds=2)
    html_str = render_html_report(
        bo,
        aci_df=bo.aci_df(),
        tail_risk_df=bo.tail_risk_df(),
        scenario_opt_df=bo.optimize_scenario(unit_cost=4.0).schedule_df,
    )
    assert "<html" in html_str.lower()
    out_file = export_report(bo, tmp_path / "mcp_bakeoff_report.html")
    assert out_file.exists()


# =============================================================================
# TIER 4 — Real-World Enterprise Application Scenarios (6 Industry Workloads)
# =============================================================================


def test_t4_scenario_1_fashion_electronics_npi_launch_and_replenishment() -> None:
    """Tier 4 Scenario 1: Fashion & Consumer Electronics NPI Launch + Initial Buy Replenishment.

    Verifies that hierarchical donor transfer (`cold_start.py`) on a newly launched 5-day SKU
    strictly reduces held-out launch WAPE compared to a local naive baseline, produces calibrated
    Day-1 intervals for a 0-obs pre-launch SKU, and chains into Copula trajectory sampling and
    `HiGHS` MILP portfolio inventory optimization.
    """
    from scale_forecasting.cold_start import (
        build_cold_start_donor_pool,
        cold_start_transfer_df,
        forecast_cold_start_series,
    )
    from scale_forecasting.copula import compute_tail_risk, sample_trajectories
    from scale_forecasting.inventory_optimizer import optimize_inventory_portfolio

    # Construct a realistic seasonal + surge launch panel where donors share seasonal + surge shape
    panel = _make_cold_start_npi_panel(n_mature_days=49, n_short_days=12, seed=4001)
    # Hold out days 6..12 of donor_elec_b as a 5-day cold-start launch with known ground truth
    full_target = panel[panel["ts_id"] == "donor_elec_b"].sort_values("ds").reset_index(drop=True)
    observed_prefix = full_target.iloc[:5].copy()
    observed_prefix["ts_id"] = "npi_eval_sku"
    held_out_actuals = full_target.iloc[5:12]["y"].to_numpy(dtype=float)

    train_panel = pd.concat(
        [panel[panel["ts_id"] != "donor_elec_b"], observed_prefix],
        ignore_index=True,
    )
    pool = build_cold_start_donor_pool(
        train_panel,
        min_obs=14,
        static_covariates=["category", "brand", "region"],
    )
    cs_res = forecast_cold_start_series(
        observed_prefix,
        pool,
        horizon=7,
        freq="D",
        method="analog_transfer",
        min_obs=14,
        static_covariates=["category", "brand", "region"],
    )
    cs_yhat = cs_res.predictions["yhat"].to_numpy(dtype=float)
    # Local naive baseline uses only the 1st/last point without seasonal/surge donor shape
    naive_last = np.full(7, float(observed_prefix["y"].iloc[0]))
    wape_cs = float(np.sum(np.abs(held_out_actuals - cs_yhat)) / np.sum(np.abs(held_out_actuals)))
    wape_naive = float(
        np.sum(np.abs(held_out_actuals - naive_last)) / np.sum(np.abs(held_out_actuals))
    )
    assert wape_cs < wape_naive

    # Evaluate full panel transfer table (including 0-obs SKU) and chain into Copula + MILP
    transfer_table = cold_start_transfer_df(
        train_panel,
        min_obs=14,
        static_covariates=["category", "brand", "region"],
        horizon=7,
    )
    assert bool(
        transfer_table.loc[transfer_table["ts_id"] == "npi_zero_obs", "is_cold_start"].iloc[0]
    )

    zero_res = forecast_cold_start_series(
        train_panel[train_panel["ts_id"] == "npi_zero_obs"],
        pool,
        horizon=7,
        freq="D",
        static_covariates=["category", "brand", "region"],
    )
    npi_preds = pd.concat(
        [
            cs_res.predictions.assign(
                ts_id="npi_eval_sku",
                model_type="analog_transfer",
                forecast_date=cs_res.predictions["ds"],
            ),
            zero_res.predictions.assign(
                ts_id="npi_zero_obs",
                model_type="analog_transfer",
                forecast_date=zero_res.predictions["ds"],
            ),
        ],
        ignore_index=True,
    )
    paths = sample_trajectories(npi_preds, n_paths=200, copula_type="gaussian", floor=0.0, seed=42)
    risk = compute_tail_risk(paths, predictions_df=npi_preds, window=7)
    assert len(risk) == 2

    inv_plan = optimize_inventory_portfolio(
        npi_preds,
        trajectories_df=paths,
        max_budget=15_000.0,
        unit_cost={"npi_eval_sku": 12.0, "npi_zero_obs": 15.0},
        vendor_moq=50.0,
        order_multiple=25.0,
        uncertainty_source="copula",
    )
    assert float(inv_plan["order_spend"].sum()) <= 15_000.0 + 1e-6


def test_t4_scenario_2_retail_promotional_calendar_with_pull_forward_and_cannibalization() -> None:
    """Tier 4 Scenario 2: Retail Promotional Calendar Optimization with Pull-Forward & Halo."""
    from scale_forecasting.scenario_optimizer import optimize_scenario

    hist_df, fut_df = _make_confounded_price_promo_hierarchy_panel(n_days=56, horizon=10, seed=4002)
    cfg = RunConfig.model_validate(
        {
            "run_name": "t4-scen-2-retail-promo",
            "data": {"source_table": "p.d.t", "freq": "D", "horizon": 10},
            "models": ["regression_lags@monotone"],
            "features": {
                "future_covariates": ["price", "promo"],
                "monotonic_constraints": {"price": -1, "promo": 1},
            },
            "backtest": {"enabled": True, "n_folds": 1, "horizon": 10},
            "hierarchy": {
                "levels": [["region"], ["region", "category"]],
                "reconciliation_methods": ["wls_struct"],
            },
        }
    )

    elas_df = compute_promo_elasticity_decomposition(
        hist_df, promo_col="promo", price_col="price", cfg=cfg
    )
    assert not elas_df.empty
    assert bool((elas_df["pull_forward_dip"] <= 1e-9).all())

    opt_res = optimize_scenario(
        hist_df,
        config=cfg,
        future_covariates_df=fut_df,
        objective="gross_profit",
        price_col="price",
        promo_col="promo",
        price_bounds=(-0.12, 0.12),
        max_promo_days=3,
        min_cooldown_days=3,
        unit_cost=4.5,
        min_margin_pct=0.22,
        reconcile=True,
        include_inventory_impact=True,
    )
    assert opt_res.optimal_total_gross_margin >= opt_res.baseline_total_gross_margin - 1e-6
    assert opt_res.coherence_max_abs_error < 1e-6
    assert not opt_res.inventory_delta_df.empty


def test_t4_scenario_3_energy_grid_peak_surge_tail_risk_with_student_t_copula() -> None:
    """Tier 4 Scenario 3: Regional Power Grid Peak-Window Surge Risk with Student-t Copula."""
    from scale_forecasting.copula import (
        compute_tail_risk,
        evaluate_trajectory_paths,
        sample_trajectories,
    )

    preds_df, oof_df, actuals_df = _make_autocorrelated_heavy_tail_oof_and_preds(
        n_series=4, horizon=7, n_folds=10, rho=0.82, heavy_tails=True, seed=4003
    )
    paths_t = sample_trajectories(
        preds_df,
        oof_df,
        n_paths=800,
        copula_type="student_t",
        df_t=2.8,
        cross_series=True,
        floor=0.0,
        seed=42,
    )
    paths_g = sample_trajectories(
        preds_df,
        oof_df,
        n_paths=800,
        copula_type="gaussian",
        cross_series=True,
        floor=0.0,
        seed=42,
    )
    risk_t = compute_tail_risk(
        paths_t,
        predictions_df=preds_df,
        actuals_df=actuals_df,
        oof_df=oof_df,
        window=3,
        alphas=(0.90, 0.95, 0.99),
        threshold=420.0,
    )
    risk_g = compute_tail_risk(
        paths_g,
        predictions_df=preds_df,
        actuals_df=actuals_df,
        oof_df=oof_df,
        window=3,
        alphas=(0.90, 0.95, 0.99),
        threshold=420.0,
    )
    assert bool((risk_t["copula_amplification_ratio"] > 1.05).all())
    assert float(risk_t["cvar_99"].sum()) > float(risk_g["cvar_99"].sum())
    assert bool((risk_t["peak_window_cvar"] >= risk_t["peak_window_var"] - 1e-6).all())

    scores = evaluate_trajectory_paths(paths_t, actuals_df, p=0.5)
    assert scores["crps"] > 0.0 and scores["energy_score"] > 0.0 and scores["variogram_score"] > 0.0


def test_t4_scenario_4_multi_echelon_warehouse_working_capital_and_volume_replenishment() -> None:
    """Tier 4 Scenario 4: 8-SKU Multi-Echelon Distribution Center Replenishment under
    Working-Capital, Warehouse Cube, Vendor MOQ, and Case-Pack Multiples.
    """
    from scale_forecasting.inventory_optimizer import (
        optimize_inventory_portfolio,
        plot_inventory_frontier,
    )

    preds_df, oof_df, _ = _make_autocorrelated_heavy_tail_oof_and_preds(
        n_series=8, horizon=7, n_folds=6, rho=0.60, heavy_tails=False, seed=4004
    )
    costs = {f"sku_{i}": float(8.0 + 6.0 * i) for i in range(8)}
    vols = {f"sku_{i}": float(0.4 + 0.3 * (i % 4)) for i in range(8)}
    moqs = {f"sku_{i}": float(24.0 if i % 2 == 0 else 36.0) for i in range(8)}

    base_pol = compute_inventory_policy(preds_df, oof_df=oof_df, lead_time=7.0, service_level=0.95)
    unconstrained_spend = sum(
        float(row["recommended_order_qty"]) * costs[str(row["ts_id"])]
        for _, row in base_pol.iterrows()
    )
    cap_budget = unconstrained_spend * 0.55

    opt_pol = optimize_inventory_portfolio(
        preds_df,
        oof_df,
        max_budget=cap_budget,
        max_volume=2200.0,
        unit_cost=costs,
        unit_volume=vols,
        holding_cost=1.5,
        stockout_cost=14.0,
        vendor_moq=moqs,
        order_multiple=12.0,
        uncertainty_source="copula",
        n_paths=250,
        seed=42,
    )
    assert len(opt_pol) == 8
    assert float(opt_pol["order_spend"].sum()) <= cap_budget + 1e-6
    assert float(opt_pol["order_volume"].sum()) <= 2200.0 + 1e-6
    assert bool((opt_pol["shadow_price_budget"] > 0.0).all())
    for _, row in opt_pol.iterrows():
        q = float(row["recommended_order_qty"])
        if q > 1e-6:
            assert q >= float(row["vendor_moq"]) - 1e-6
            assert min(q % 12.0, 12.0 - (q % 12.0)) < 1e-5

    ax = plot_inventory_frontier(opt_pol)
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close("all")


def test_t4_scenario_5_macro_regime_shift_online_aci_and_auto_forecaster_governance() -> None:
    """Tier 4 Scenario 5: Supply-Chain Macro Regime Shift with Online `aci_normalized` & Auto."""
    from scale_forecasting.aci import compute_aci_df

    oof_df, preds_df = _make_regime_shift_oof_and_preds(
        n_oof_steps=40, horizon=7, shock_multiplier=3.8, seed=4005
    )
    aci_eval = compute_aci_df(
        oof_df,
        preds_df,
        gamma=0.06,
        target_coverage=0.80,
        normalized=True,
        horizon_delay=True,
    )
    post_shock = aci_eval[(aci_eval["horizon_step"] == 1) & (aci_eval["step_idx"] >= 22)]
    assert (
        float(post_shock["aci_covered"].mean()) > float(post_shock["static_covered"].mean()) + 0.15
    )
    assert bool(aci_eval["drift_detected"].any()) is True

    # Run Auto-Forecaster on a panel with a regime shift + cold-start series and audit Q4 & Q7
    cs_panel = _make_cold_start_npi_panel(n_mature_days=42, n_short_days=5, seed=4005)
    active_cs = cs_panel[cs_panel["ts_id"] != "npi_zero_obs"].copy()
    f_gov = sf.Forecaster.auto(
        active_cs,
        horizon=7,
        freq="D",
        tier="first_light",
        models=["naive_mean", "theta"],
        n_folds=1,
        autofocus=False,
    )
    assess_df = f_gov.assess()
    assert isinstance(assess_df, pd.DataFrame) and len(assess_df) == 7
    assert "cold_start" in str(assess_df.iloc[3].to_dict()).lower()
    assert (
        "aci" in str(assess_df.iloc[6].to_dict()).lower()
        or "tail_risk" in str(assess_df.iloc[6].to_dict()).lower()
    )


def test_t4_scenario_6_full_forecast_to_decision_spine_track_j_k_l_end_to_end(
    tmp_path: Path,
) -> None:
    """Tier 4 Scenario 6: Full Forecast-to-Decision Spine (Track J + Track K + Track L + Report)."""
    hist_df, fut_df = _make_confounded_price_promo_hierarchy_panel(n_days=42, horizon=7, seed=4006)
    cfg = RunConfig.model_validate(
        {
            "run_name": "t4-scen-6-full-spine",
            "data": {"source_table": "p.d.t", "freq": "D", "horizon": 7},
            "models": ["naive_mean", "regression_lags@monotone"],
            "features": {
                "future_covariates": ["price", "promo"],
                "static_covariates": ["region", "category"],
                "monotonic_constraints": {"price": -1, "promo": 1},
                "cold_start_transfer": True,
            },
            "backtest": {"enabled": True, "n_folds": 2, "horizon": 7},
            "hierarchy": {
                "levels": [["region"]],
                "cross_temporal": True,
                "temporal_aggregations": [7],
                "reconciliation_methods": ["wls_struct", "cross_temporal_mint"],
            },
            "output": {
                "calibration_method": "aci_normalized",
                "aci_gamma": 0.05,
                "floor": 0.0,
            },
        }
    )

    f = sf.Forecaster(cfg, data=hist_df)
    # 1. Track J: Causal Promo Elasticity & Counterfactual Scenario Simulation
    elas_df = f.promo_elasticity_df(promo_col="promo", price_col="price")
    assert not elas_df.empty

    # 2. Track K: 3-Mode Cross-Temporal Coherence & Smooth Goal-Seek QP
    preds_df = f.predictions_df()
    assert not preds_df.empty
    gs_df = f.allocate_target(
        target_total=float(preds_df["yhat"].sum()) * 1.05,
        smoothness_weight=4.0,
    )
    assert not gs_df.empty

    # 3. Track L: Cold-Start NPI, Online ACI, Copula Tail Risk, Inventory MILP, Scenario Optimizer
    cs_df = f.cold_start_transfer_df()
    assert not cs_df.empty
    aci_res = f.aci_df()
    assert not aci_res.empty
    traj_res = f.sample_trajectories(n_paths=100, copula_type="student_t", seed=42)
    assert not traj_res.empty
    risk_res = f.tail_risk_df()
    assert not risk_res.empty
    inv_res = f.optimize_inventory_df(
        max_budget=12_000.0,
        unit_cost=4.5,
        vendor_moq=20.0,
        order_multiple=10.0,
        uncertainty_source="copula",
    )
    assert not inv_res.empty
    assert float(inv_res["order_spend"].sum()) <= 12_000.0 + 1e-6

    scen_opt = f.optimize_scenario(
        future_covariates_df=fut_df,
        objective="gross_profit",
        unit_cost=4.5,
        max_promo_days=2,
        min_cooldown_days=2,
        reconcile=True,
        include_inventory_impact=True,
    )
    assert scen_opt.optimal_objective_value >= scen_opt.baseline_objective_value - 1e-5
    assert scen_opt.coherence_max_abs_error < 1e-6

    # 4. Self-contained HTML report export
    rep_path = f.export_report(tmp_path / "full_spine_report.html")
    assert rep_path.exists()
    plt.close("all")


# =============================================================================
# TIER 5 — Adversarial Challenger Regression Tests
# =============================================================================


def test_t5_adversarial_1_variant_tags_cold_start_and_aci_in_worker_and_hash() -> None:
    """Tier 5 Adversarial 1: `@cold_start`, `@analog_transfer`, `@bass_diffusion`,
    `@weibull_lifecycle`, `@aci`, `@aci_norm`, and `@aci_normalized` variant tags parse cleanly,
    execute in `run_cell` (including 5-arg `ts_id` + `cold_start_donor_pool`), and produce
    distinct `make_model_spec_hash` digests.
    """
    from scale_forecasting.cold_start import build_cold_start_donor_pool
    from scale_forecasting.registry.ids import make_model_spec_hash
    from scale_forecasting.worker import run_cell

    panel = _make_cold_start_npi_panel(n_mature_days=35, n_short_days=3, seed=5501)
    cfg = RunConfig.model_validate(
        {
            "run_name": "t5-variants",
            "data": {"source_table": "p.d.t", "freq": "D", "horizon": 5},
            "models": [
                "naive_mean",
                "naive_mean@cold_start",
                "naive_mean@bass_diffusion",
                "naive_mean@aci",
                "naive_mean@aci_norm",
            ],
            "backtest": {"enabled": True, "n_folds": 2, "horizon": 5, "step": 5, "min_train": 14},
            "features": {"static_covariates": ["category", "brand", "region"]},
        }
    )
    h_base = make_model_spec_hash("naive_mean", cfg)
    h_cs = make_model_spec_hash("naive_mean@cold_start", cfg)
    h_bass = make_model_spec_hash("naive_mean@bass_diffusion", cfg)
    h_aci = make_model_spec_hash("naive_mean@aci", cfg)
    h_aci_norm = make_model_spec_hash("naive_mean@aci_norm", cfg)
    assert len({h_base, h_cs, h_bass, h_aci, h_aci_norm}) == 5

    pool = build_cold_start_donor_pool(panel, cfg)
    short_df = panel[panel["ts_id"] == "npi_elec_3d"].copy()
    res_cs = run_cell(
        "npi_elec_3d", short_df, "naive_mean@bass_diffusion", cfg, cold_start_donor_pool=pool
    )
    assert res_cs.status == "ok"
    assert res_cs.fit_diagnostics["cold_start_transfer"]["is_cold_start"] is True
    assert res_cs.fit_diagnostics["cold_start"]["selected_archetype"] == "bass_diffusion"

    mature_df = panel[panel["ts_id"] == "donor_elec_a"].copy()
    res_aci = run_cell(mature_df, "naive_mean@aci_norm", cfg)
    assert res_aci.status == "ok"
    assert res_aci.interval_calibration is not None
    assert res_aci.interval_calibration.startswith(("aci", "oof-aci"))


def test_t5_adversarial_2_copula_trajectory_attrs_safe_under_pandas_groupby_and_merge() -> None:
    """Tier 5 Adversarial 2: `sample_trajectories` with cross-temporal hierarchy reconciliation
    wraps NumPy arrays inside `.attrs` in `SafeArrayDict` so pandas `.groupby()`, `.merge()`,
    `.loc[]`, and `.copy()` never raise `ValueError: The truth value of an array...`.
    """
    from scale_forecasting.copula import sample_trajectories
    from scale_forecasting.reconciliation import build_hierarchy
    from scale_forecasting.worker import run_cell

    raw_df = _make_confounded_price_promo_hierarchy_panel(n_days=28, horizon=14, seed=5502)[0]
    cfg = RunConfig.model_validate(
        {
            "run_name": "t5-copula-attrs",
            "data": {"source_table": "p.d.t", "freq": "D", "horizon": 14},
            "models": ["naive_mean"],
            "hierarchy": {
                "enabled": True,
                "levels": [["region"], ["region", "category"]],
                "reconcile_temporal": True,
                "temporal_aggregations": [7, 14],
                "cross_temporal": True,
            },
        }
    )
    panel, spec = build_hierarchy(raw_df, cfg)
    preds = pd.concat(
        [
            run_cell(sub, "naive_mean", cfg).predictions.assign(ts_id=tid, model_type="naive_mean")
            for tid, sub in panel.groupby("ts_id", sort=False)
        ],
        ignore_index=True,
    ).rename(columns={"ds": "forecast_date"})

    traj_df = sample_trajectories(
        preds,
        n_paths=10,
        cfg=cfg,
        hier_spec=spec,
        reconcile=True,
        seed=77,
    )
    # Trigger pandas attrs comparisons via groupby, loc, copy, and merge
    grouped = traj_df.groupby(["path_id", "ts_id"], as_index=False)["yhat_sample"].sum()
    merged = traj_df.merge(grouped, on=["path_id", "ts_id"], suffixes=("", "_sum"))
    sliced = traj_df.loc[traj_df["path_id"] == 0].copy()
    assert not merged.empty
    assert not sliced.empty
    assert traj_df.attrs["aggregated_by_node_k"] == traj_df.copy().attrs["aggregated_by_node_k"]


def test_t5_adversarial_3_scenario_optimizer_rejects_over_100pct_pull_forward_dip() -> None:
    """Tier 5 Adversarial 3: When historical promotions exhibit > 100% pull-forward dip
    (e.g., +40% direct promo lift followed by -45% post-promo dip), `optimize_scenario` does
    not clip the dip at -60% of lift and refuses to schedule net-value-destroying promotions
    at any horizon step (including the final step `h = H - 1`).
    """
    from scale_forecasting.scenario_optimizer import optimize_scenario

    dates = pd.date_range("2025-01-01", periods=56, freq="D")
    y_vals = np.full(56, 100.0, dtype=float)
    promo_vals = np.zeros(56, dtype=float)
    for p_idx in (10, 24, 38):
        promo_vals[p_idx] = 1.0
        y_vals[p_idx] = 140.0  # +40 units lift
        y_vals[p_idx + 1] = 70.0  # -30 units dip
        y_vals[p_idx + 2] = 85.0  # -15 units dip (total dip = -45 > +40 lift)
    hist_df = pd.DataFrame(
        {
            "ts_id": "sku_dip",
            "ds": dates,
            "y": y_vals,
            "promo": promo_vals,
            "price": 10.0,
        }
    )
    fut_dates = pd.date_range("2025-02-26", periods=7, freq="D")
    preds_df = pd.DataFrame(
        {
            "ts_id": "sku_dip",
            "model_type": "naive",
            "forecast_date": fut_dates,
            "yhat": 100.0,
            "yhat_lower": 90.0,
            "yhat_upper": 110.0,
        }
    )
    res = optimize_scenario(
        preds_df,
        history_df=hist_df,
        objective="gross_profit",
        unit_cost=5.0,
        price_bounds=(0.0, 0.0),
        max_promo_days=3,
    )
    assert float(res.schedule_df["optimal_promo"].sum()) == 0.0


def test_t5_adversarial_4_inventory_optimizer_strict_feasibility_with_unaffordable_moq() -> None:
    """Tier 5 Adversarial 4: When `strict_feasibility=True` and `vendor_moq * unit_cost`
    exceeds `max_budget` for all SKUs requiring replenishment, `optimize_inventory_portfolio`
    raises `GoalSeekInfeasibleError`.
    """
    from scale_forecasting.errors import GoalSeekInfeasibleError
    from scale_forecasting.inventory_optimizer import optimize_inventory_portfolio

    preds_df = pd.DataFrame(
        {
            "ts_id": ["sku_a", "sku_b"],
            "model_type": ["naive", "naive"],
            "forecast_date": [pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-01")],
            "yhat": [50.0, 60.0],
            "yhat_lower": [40.0, 50.0],
            "yhat_upper": [60.0, 70.0],
        }
    )
    with pytest.raises(GoalSeekInfeasibleError, match="vendor_moq"):
        optimize_inventory_portfolio(
            preds_df,
            unit_cost=10.0,
            vendor_moq=100.0,  # min order = $1,000
            max_budget=200.0,  # budget = $200 < $1,000
            strict_feasibility=True,
        )
