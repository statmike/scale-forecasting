"""Comprehensive 4-Tier End-to-End Verification Suite for Track J:
Causal & Structural Econometric Guardrails (Monotonicity, Holiday Windows,
Promo Elasticity & Full-Chain Counterfactual Scenario Simulator).

Tier 1: Feature Coverage (>=5 tests per feature area F1-F7, 41 tests total)
Tier 2: Boundary, Corner & Negative Cases (>=5 tests per feature area F1-F7, 36 tests total)
Tier 3: Cross-Feature Pairwise Interactions (8 pairwise tests)
Tier 4: Real-World Application Scenarios (5 end-to-end scenarios proving AC 1.1 - AC 1.4)
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import scale_forecasting as sf
from scale_forecasting.auto import (
    AutoForecastSession,
    auto_forecast,
    build_config,
    inspect_panel_for_auto,
)
from scale_forecasting.auto_policy import (
    propose_autofocus_dag_branches,
    propose_autofocus_pass,
)
from scale_forecasting.config import RunConfig
from scale_forecasting.elasticity import (
    PROMO_ELASTICITY_COLUMNS,
    compute_promo_elasticity,
    compute_promo_elasticity_decomposition,
    make_confounded_promo_dgp,
    plot_promo_elasticity,
    promo_elasticity_df,
    summarize_promo_elasticity,
)
from scale_forecasting.errors import ConfigError, DataError
from scale_forecasting.features import (
    build_features,
    build_future_features,
    effective_config_for_model,
    resolve_model_variant,
)
from scale_forecasting.holiday_window import (
    HOLIDAY_WINDOW_COLUMNS,
    HolidayWindowConfig,
    build_holiday_window_features,
    coerce_holiday_window_config,
    holiday_window_columns,
)
from scale_forecasting.mcp import McpServer
from scale_forecasting.models import get_model
from scale_forecasting.monotonicity import (
    MONOTONE_SUPPORTED_MODELS,
    build_monotonic_constraint_vector,
    fit_sign_constrained_ridge,
    infer_monotonic_constraints,
    resolve_model_monotonic_vector,
    validate_monotonic_constraints_map,
)
from scale_forecasting.playground import bakeoff, run_model
from scale_forecasting.reconciliation import build_hierarchy, verify_coherence
from scale_forecasting.registry.ids import make_model_spec_hash, make_run_id
from scale_forecasting.report import build_report_payload, export_report, render_html_report
from scale_forecasting.scenarios import (
    apply_scenario_shocks,
    compare_scenario_margin_curves,
    simulate_scenarios,
    summarize_scenarios,
)
from scale_forecasting.sdk import Forecaster
from scale_forecasting.worker import run_cell

# ==============================================================================
# Helpers & Synthetic DGPs
# ==============================================================================


def _make_collinear_price_promo_panel(
    n_days: int = 120,
    horizon: int = 14,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a collinear price/promo panel where correlation tricks unconstrained learners."""
    rng = np.random.default_rng(seed)
    total_days = n_days + horizon
    dates = pd.date_range("2025-01-01", periods=total_days, freq="D")

    rows: list[dict[str, object]] = []
    for s_idx, sku in enumerate(("SKU_A", "SKU_B")):
        seasonal = 25.0 * np.sin(2.0 * np.pi * np.arange(total_days) / 7.0)
        unobserved_boom = (rng.uniform(0.0, 1.0, size=total_days) > 0.72).astype(float)
        # Price is positively correlated with unobserved demand boom (+4.0) -> naive positive bias!
        price = 20.0 + 3.5 * unobserved_boom + rng.normal(0.0, 0.35, size=total_days)
        # Promo is independent with strong positive structural lift (+24.0)
        promo = (rng.uniform(0.0, 1.0, size=total_days) > 0.65).astype(float)
        # True structural DGP has negative price elasticity (-2.5) and positive promo lift (+24.0),
        # while unobserved boom (+45.0) confounds historical price correlation so OLS flips sign.
        y = (
            120.0
            + 10.0 * s_idx
            + seasonal
            + 45.0 * unobserved_boom
            - 2.5 * (price - 20.0)
            + 24.0 * promo
            + rng.normal(0.0, 1.5, size=total_days)
        )
        y = np.maximum(1.0, y)
        for t in range(total_days):
            rows.append(
                {
                    "ts_id": sku,
                    "ds": dates[t],
                    "y": float(y[t]),
                    "price": float(round(price[t], 3)),
                    "promo": float(promo[t]),
                    "region": "North" if s_idx == 0 else "South",
                }
            )
    full_df = pd.DataFrame(rows)
    hist_df = full_df.groupby("ts_id", sort=False).head(n_days).reset_index(drop=True)
    fut_df = (
        full_df.groupby("ts_id", sort=False)
        .tail(horizon)[["ts_id", "ds", "price", "promo", "region"]]
        .reset_index(drop=True)
    )
    return hist_df, fut_df


def _make_moving_holiday_dgp(
    n_days: int = 730,
    seed: int = 123,
) -> pd.DataFrame:
    """Synthetic daily retail DGP with 6-day pre-holiday stock-up and 3-day post-holiday slump."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    hw_dict = build_holiday_window_features(
        dates,
        ["US"],
        HolidayWindowConfig(
            enabled=True,
            pre_holiday_days=7,
            post_holiday_days=3,
            kernel="linear",
        ),
    )
    t_idx = np.arange(n_days, dtype=float)
    base = 100.0 + 12.0 * np.sin(2.0 * np.pi * t_idx / 7.0)
    pre_lift = 65.0 * np.asarray(hw_dict["pre_holiday_window"], dtype=float)
    post_slump = -35.0 * np.asarray(hw_dict["post_holiday_window"], dtype=float)
    y = np.maximum(5.0, base + pre_lift + post_slump + rng.normal(0.0, 2.0, size=n_days))
    return pd.DataFrame({"ts_id": "STORE_1", "ds": dates, "y": y})


# ==============================================================================
# TIER 1: Feature Coverage (>=5 tests per feature area F1-F7)
# ==============================================================================


# --- F1: Monotonicity Engine & Variant Resolution -----------------------------


def test_tier1_f1_infer_monotonic_constraints_all_synonyms() -> None:
    cols = [
        "price",
        "unit_price",
        "avg_price",
        "list_price",
        "reg_price",
        "cost_per_unit",
        "tariff_rate",
        "promo",
        "promo_flag",
        "onpromotion",
        "discount_pct",
        "rebate_depth",
        "markdown_active",
        "coupon_flag",
        "ad_spend",
        "marketing_impressions",
        "temperature",
        "dayofweek",
    ]
    inferred = infer_monotonic_constraints(cols)
    for neg_col in (
        "price",
        "unit_price",
        "avg_price",
        "list_price",
        "reg_price",
        "tariff_rate",
    ):
        assert inferred[neg_col] == -1
    for pos_col in (
        "promo",
        "promo_flag",
        "onpromotion",
        "discount_pct",
        "rebate_depth",
        "markdown_active",
        "coupon_flag",
        "ad_spend",
        "marketing_impressions",
    ):
        assert inferred[pos_col] == 1
    assert "cost_per_unit" not in inferred
    assert "temperature" not in inferred
    assert "dayofweek" not in inferred


def test_tier1_f1_resolve_model_variant_monotone_all_supported_models() -> None:
    for m_spec in (
        "lightgbm@monotone",
        "xgboost@monotone",
        "catboost@monotone",
        "regression_lags@monotone",
        "ridge_lags@monotone",
    ):
        cfg = RunConfig.model_validate(
            {
                "run_name": "f1-variants",
                "data": {"source_table": "t", "horizon": 7},
                "features": {"future_covariates": ["price", "promo"]},
                "models": [m_spec],
            }
        )
        resolved_name = cfg.models[0]
        spec = resolve_model_variant(cfg, resolved_name)
        assert spec.base_model in MONOTONE_SUPPORTED_MODELS
        assert spec.effective_cfg.features.monotonic_constraints == {"price": -1, "promo": 1}


def test_tier1_f1_composable_monotone_variant_tags() -> None:
    cfg = RunConfig.model_validate(
        {
            "run_name": "f1-composable",
            "data": {"source_table": "t", "horizon": 7},
            "features": {"future_covariates": ["price", "promo"]},
            "models": [
                "lightgbm@global+monotone",
                "xgboost@direct+monotone",
                "regression_lags@recursive+monotone",
            ],
        }
    )
    spec_lgb = resolve_model_variant(cfg, "lightgbm@global+monotone")
    assert spec_lgb.effective_cfg.features.monotonic_constraints == {"price": -1, "promo": 1}

    spec_xgb = resolve_model_variant(cfg, "xgboost@direct+monotone")
    assert spec_xgb.clean_params.get("strategy") == "direct"
    assert spec_xgb.effective_cfg.features.monotonic_constraints == {"price": -1, "promo": 1}

    spec_reg = resolve_model_variant(cfg, "regression_lags@recursive+monotone")
    assert spec_reg.clean_params.get("strategy") == "recursive"
    assert spec_reg.effective_cfg.features.monotonic_constraints == {"price": -1, "promo": 1}


def test_tier1_f1_sign_constrained_ridge_closed_form_enforcement() -> None:
    rng = np.random.default_rng(7)
    n = 80
    # Construct x where col 0 has positive raw correlation with y, col 1 has negative correlation
    z = rng.normal(0.0, 1.0, size=n)
    x_price = 10.0 + 2.0 * z + rng.normal(0.0, 0.2, size=n)
    x_promo = 1.0 - 1.5 * z + rng.normal(0.0, 0.2, size=n)
    y = 50.0 + 8.0 * z + rng.normal(0.0, 0.5, size=n)
    X = np.column_stack([x_price, x_promo])

    w_constrained = fit_sign_constrained_ridge(X, y, alpha=1.0, constraint_vec=[-1, 1])
    assert len(w_constrained) == 3
    assert w_constrained[1] <= 1e-9  # price weight <= 0
    assert w_constrained[2] >= -1e-9  # promo weight >= 0


def test_tier1_f1_tree_and_linear_learners_monotonic_forecast_ordering() -> None:
    hist_df, fut_base = _make_collinear_price_promo_panel(n_days=90, horizon=7, seed=19)
    s_df = hist_df[hist_df["ts_id"] == "SKU_A"].reset_index(drop=True)
    f_base = fut_base[fut_base["ts_id"] == "SKU_A"].reset_index(drop=True)

    f_high_price = f_base.copy()
    f_high_price["price"] = f_base["price"] * 1.35

    f_high_promo = f_base.copy()
    f_high_promo["promo"] = 1.0
    f_low_promo = f_base.copy()
    f_low_promo["promo"] = 0.0

    for model_spec in (
        "regression_lags@monotone",
        "lightgbm@monotone",
        "xgboost@monotone",
        "catboost@monotone",
    ):
        cls = get_model(model_spec)
        if not cls.is_available():
            continue
        cfg = RunConfig.model_validate(
            {
                "run_name": "mono-order",
                "data": {"source_table": "t", "horizon": 7, "freq": "D"},
                "features": {"future_covariates": ["price", "promo"]},
                "models": [model_spec],
                "backtest": {"enabled": False},
            }
        )
        res_base = run_cell(s_df, model_spec, cfg, future_df=f_base)
        res_hi_p = run_cell(s_df, model_spec, cfg, future_df=f_high_price)
        res_hi_pr = run_cell(s_df, model_spec, cfg, future_df=f_high_promo)
        res_lo_pr = run_cell(s_df, model_spec, cfg, future_df=f_low_promo)

        assert res_base.status == "ok"
        assert np.all(
            res_hi_p.predictions["yhat_raw"].to_numpy()
            <= res_base.predictions["yhat_raw"].to_numpy() + 1e-6
        ), f"{model_spec} violated price monotonicity"
        assert np.all(
            res_hi_pr.predictions["yhat_raw"].to_numpy()
            >= res_lo_pr.predictions["yhat_raw"].to_numpy() - 1e-6
        ), f"{model_spec} violated promo monotonicity"


def test_tier1_f1_baseline_isolation_with_monotone_sibling_in_run_cell() -> None:
    cfg = RunConfig.model_validate(
        {
            "run_name": "sibling-iso",
            "data": {"source_table": "t", "horizon": 7, "freq": "D"},
            "features": {
                "future_covariates": ["price", "promo"],
                "monotonic_constraints": {"price": -1, "promo": 1},
            },
            "models": ["regression_lags", "regression_lags@monotone"],
        }
    )
    eff_unconstrained = effective_config_for_model(
        cfg, get_model("regression_lags"), "regression_lags"
    )
    eff_constrained = effective_config_for_model(
        cfg, get_model("regression_lags@monotone"), "regression_lags@monotone"
    )
    assert eff_unconstrained.features.monotonic_constraints == {}
    assert eff_constrained.features.monotonic_constraints == {"price": -1, "promo": 1}


# --- F2: Asymmetric Holiday Lead/Lag Window Covariates ------------------------


def test_tier1_f2_holiday_window_config_canonical_and_alias_fields() -> None:
    hw1 = HolidayWindowConfig(pre_holiday_days=7, post_holiday_days=3, kernel="linear")
    assert hw1.pre_holiday_days == 7
    assert hw1.post_holiday_days == 3
    assert hw1.pre_days == 7
    assert hw1.post_days == 3
    assert hw1.decay == "linear"

    hw2 = HolidayWindowConfig(pre_days=5, post_days=2, decay="none")
    assert hw2.pre_holiday_days == 5
    assert hw2.post_holiday_days == 2
    assert hw2.pre_days == 5
    assert hw2.post_days == 2
    assert hw2.kernel == "indicator"
    assert hw2.decay == "none"
    assert hw2.to_dict()["pre_holiday_days"] == 5

    hw3 = HolidayWindowConfig.from_value((6, 4))
    assert hw3 is not None
    assert hw3.pre_holiday_days == 6 and hw3.post_holiday_days == 4


def test_tier1_f2_linear_indicator_and_exponential_kernel_ramps() -> None:
    dates = pd.date_range("2025-12-20", "2025-12-30", freq="D")
    holidays_df = pd.DataFrame({"ds": [pd.Timestamp("2025-12-25")], "holiday": ["Christmas"]})

    lin_df = pd.DataFrame(
        build_holiday_window_features(
            dates,
            holidays_df,
            HolidayWindowConfig(pre_holiday_days=4, post_holiday_days=2, kernel="linear"),
        ),
        index=dates,
    )
    ind_df = pd.DataFrame(
        build_holiday_window_features(
            dates,
            holidays_df,
            HolidayWindowConfig(pre_holiday_days=4, post_holiday_days=2, kernel="indicator"),
        ),
        index=dates,
    )
    exp_df = pd.DataFrame(
        build_holiday_window_features(
            dates,
            holidays_df,
            HolidayWindowConfig(pre_holiday_days=4, post_holiday_days=2, kernel="exponential"),
        ),
        index=dates,
    )

    assert list(lin_df.columns) == list(HOLIDAY_WINDOW_COLUMNS)
    # 1 day before Christmas (index 4: Dec 24) should have weight 1.0 across all 3 kernels
    assert lin_df.loc["2025-12-24", "pre_holiday_window"] == pytest.approx(1.0)
    assert ind_df.loc["2025-12-24", "pre_holiday_window"] == pytest.approx(1.0)
    assert exp_df.loc["2025-12-24", "pre_holiday_window"] == pytest.approx(1.0)
    # 4 days before Christmas (Dec 21): linear = 0.25, indicator = 1.0, exponential in (0, 1)
    assert lin_df.loc["2025-12-21", "pre_holiday_window"] == pytest.approx(0.25)
    assert ind_df.loc["2025-12-21", "pre_holiday_window"] == pytest.approx(1.0)
    assert 0.0 < exp_df.loc["2025-12-21", "pre_holiday_window"] < 1.0


def test_tier1_f2_build_features_and_future_features_column_parity() -> None:
    df = _make_moving_holiday_dgp(n_days=120)
    cfg = RunConfig.model_validate(
        {
            "run_name": "hw-parity",
            "data": {"source_table": "t", "horizon": 14, "freq": "D"},
            "features": {
                "holidays": ["US"],
                "holiday_window": {"pre_holiday_days": 7, "post_holiday_days": 3},
            },
            "models": ["regression_lags"],
        }
    )
    y, X = build_features(df, cfg)
    assert X is not None
    for col in HOLIDAY_WINDOW_COLUMNS:
        assert col in X.columns
    fut_X = build_future_features(y, X, cfg)
    assert fut_X is not None
    assert list(fut_X.columns) == list(X.columns)
    assert len(fut_X) == 14


def test_tier1_f2_step0_auto_enables_holiday_window_on_daily_and_hourly() -> None:
    df_daily = _make_moving_holiday_dgp(n_days=90)
    bp_daily = build_config(df_daily, holidays=["US"], horizon=7, where="local")
    assert bp_daily.config.features.holiday_window is not None
    assert bp_daily.config.features.holiday_window.enabled is True
    assert bp_daily.config.features.holiday_window.pre_holiday_days == 7
    assert bp_daily.config.features.holiday_window.post_holiday_days == 3

    dates_h = pd.date_range("2025-01-01", periods=120, freq="h")
    df_hourly = pd.DataFrame(
        {"ts_id": "H1", "ds": dates_h, "y": 50.0 + np.sin(np.arange(120) / 4.0)}
    )
    bp_hourly = build_config(df_hourly, holidays=["US"], horizon=24, where="local")
    assert bp_hourly.config.features.holiday_window is not None
    assert bp_hourly.config.features.holiday_window.enabled is True


def test_tier1_f2_multi_country_holiday_calendar_window_construction() -> None:
    dates = pd.date_range("2025-07-01", "2025-07-18", freq="D")
    hw_df = pd.DataFrame(
        build_holiday_window_features(
            dates,
            ["US", "FR"],
            HolidayWindowConfig(pre_holiday_days=3, post_holiday_days=2, kernel="linear"),
        ),
        index=dates,
    )
    # July 3 is 1 day before US Independence Day; July 13 is 1 day before Bastille Day
    assert hw_df.loc["2025-07-03", "pre_holiday_window"] == pytest.approx(1.0)
    assert hw_df.loc["2025-07-13", "pre_holiday_window"] == pytest.approx(1.0)


def test_tier1_f2_variant_override_holiday_window_in_model_params() -> None:
    cfg = RunConfig.model_validate(
        {
            "run_name": "hw-override",
            "data": {"source_table": "t", "horizon": 7, "freq": "D"},
            "features": {"holidays": ["US"]},
            "models": ["regression_lags", "regression_lags@hw_custom"],
            "model_params": {
                "regression_lags@hw_custom": {
                    "holiday_window": {"pre_days": 10, "post_days": 4, "decay": "exponential"}
                }
            },
        }
    )
    spec = resolve_model_variant(cfg, "regression_lags@hw_custom")
    hw = spec.effective_cfg.features.holiday_window
    assert hw is not None
    assert hw.pre_holiday_days == 10
    assert hw.post_holiday_days == 4
    assert hw.kernel == "exponential"
    assert "holiday_window" not in spec.clean_params


# --- F3: Promo & Price Elasticity Decomposition (`elasticity.py`) -------------


def test_tier1_f3_exact_ac12_schema_and_all_columns() -> None:
    df, _truth = make_confounded_promo_dgp(n_series=2, n_days=120, seed=42)
    decomp = compute_promo_elasticity_decomposition(df)
    required_ac12 = {
        "baseline_yhat",
        "incremental_promo_lift",
        "pull_forward_dip",
        "cannibalization_effect",
        "net_causal_elasticity",
    }
    assert required_ac12.issubset(decomp.columns)
    assert list(decomp.columns) == list(PROMO_ELASTICITY_COLUMNS)
    assert np.allclose(decomp["baseline_yhat"], decomp["baseline_demand"])
    assert np.allclose(decomp["incremental_promo_lift"], decomp["net_causal_lift"])
    assert np.allclose(decomp["cannibalization_effect"], decomp["cannibalization_dip"])


def test_tier1_f3_dml_partialing_out_removes_seasonal_confounding_bias() -> None:
    df, truth = make_confounded_promo_dgp(
        n_series=3,
        n_days=365,
        true_promo_lift=30.0,
        seasonal_amplitude=35.0,
        seed=42,
    )
    decomp = promo_elasticity_df(df)
    summary = summarize_promo_elasticity(decomp)
    dml_est = float(summary["mean_causal_promo_lift"].mean())
    ols_est = float(summary["mean_naive_ols_lift"].mean())
    true_val = float(truth["true_promo_lift"])
    # Naive OLS is heavily upward biased by seasonal confounding; DML is much closer to truth
    assert abs(dml_est - true_val) < abs(ols_est - true_val)
    assert abs(dml_est - true_val) <= 6.0
    assert float(summary["confounding_bias_pct"].mean()) > 10.0


def test_tier1_f3_post_promo_pull_forward_trough_recovery() -> None:
    df, _ = make_confounded_promo_dgp(
        n_series=2,
        n_days=240,
        true_promo_lift=32.0,
        true_pull_forward_dip=-14.0,
        seed=77,
    )
    decomp = compute_promo_elasticity(df, pull_forward_lags=3)
    assert (decomp["pull_forward_dip"] <= 0.0).all()
    assert float(decomp["pull_forward_dip"].min()) < -3.0


def test_tier1_f3_cross_sku_cannibalization_effect_estimation() -> None:
    df, _ = make_confounded_promo_dgp(
        n_series=3,
        n_days=240,
        true_promo_lift=30.0,
        true_cannibalization_dip=-12.0,
        seed=99,
    )
    decomp = compute_promo_elasticity_decomposition(df)
    assert (decomp["cannibalization_effect"] <= 0.0).all()
    assert float(decomp["cannibalization_effect"].sum()) < -10.0


def test_tier1_f3_summarize_and_plot_promo_elasticity() -> None:
    import matplotlib.pyplot as plt

    df, _ = make_confounded_promo_dgp(n_series=2, n_days=120, seed=11)
    decomp = compute_promo_elasticity_decomposition(df)
    summary = summarize_promo_elasticity(decomp)
    assert len(summary) == 2
    assert "net_to_gross_lift_ratio" in summary.columns
    fig = plot_promo_elasticity(decomp, ts_id="sku_001")
    assert fig is not None
    plt.close("all")


def test_tier1_f3_price_discount_and_explicit_confounders_support() -> None:
    df, _ = make_confounded_promo_dgp(n_series=2, n_days=150, seed=21)
    df["macro_index"] = 100.0 + np.linspace(0.0, 5.0, len(df))
    decomp = compute_promo_elasticity_decomposition(
        df,
        promo_col="promo",
        price_col="price",
        confounders=["macro_index"],
        holidays=["US"],
    )
    assert not decomp.empty
    assert (decomp["incremental_promo_lift"] >= 0.0).all()


# --- F4: Counterfactual What-If Scenario Simulator (`scenarios.py`) -----------


def test_tier1_f4_shock_dsl_operators_and_date_window_scoping() -> None:
    dates = pd.date_range("2025-05-01", periods=6, freq="D")
    fut_df = pd.DataFrame(
        {
            "ts_id": ["S1"] * 6,
            "ds": dates,
            "price": [10.0] * 6,
            "promo": [0.0] * 6,
        }
    )
    shocked = apply_scenario_shocks(
        fut_df,
        {
            "price": {
                "op": "pct",
                "value": -10.0,
                "start_date": "2025-05-02",
                "end_date": "2025-05-04",
            },
            "promo": {
                "op": "set",
                "value": 1.0,
                "start_date": "2025-05-03",
                "end_date": "2025-05-03",
            },
        },
    )
    assert shocked.loc[0, "price"] == pytest.approx(10.0)
    assert shocked.loc[1, "price"] == pytest.approx(9.0)
    assert shocked.loc[3, "price"] == pytest.approx(9.0)
    assert shocked.loc[4, "price"] == pytest.approx(10.0)
    assert shocked.loc[2, "promo"] == pytest.approx(1.0)
    assert shocked.loc[0, "promo"] == pytest.approx(0.0)


def test_tier1_f4_gross_margin_delta_with_scalar_mapping_and_column_unit_cost() -> None:
    hist_df, fut_df = _make_collinear_price_promo_panel(n_days=70, horizon=7, seed=31)
    cfg = RunConfig.model_validate(
        {
            "run_name": "f4-gm",
            "data": {"source_table": "t", "horizon": 7, "freq": "D"},
            "features": {"future_covariates": ["price", "promo"]},
            "models": ["regression_lags@monotone"],
            "ensemble": {"enabled": False},
        }
    )
    sim_scalar = simulate_scenarios(
        hist_df,
        cfg,
        {"promo_push": {"price": "*0.90", "promo": 1.0}},
        future_covariates_df=fut_df,
        unit_cost=12.0,
    )
    assert "gross_margin_delta" in sim_scalar.columns
    assert "scenario_gross_margin" in sim_scalar.columns
    assert "baseline_gross_margin" in sim_scalar.columns

    sim_map = simulate_scenarios(
        hist_df,
        cfg,
        {"promo_push": {"price": "*0.90", "promo": 1.0}},
        future_covariates_df=fut_df,
        unit_cost={"SKU_A": 11.0, "SKU_B": 13.0},
    )
    assert (sim_map.loc[sim_map["ts_id"] == "SKU_A", "unit_cost"] == 11.0).all()
    assert (sim_map.loc[sim_map["ts_id"] == "SKU_B", "unit_cost"] == 13.0).all()


def test_tier1_f4_inventory_policy_impact_both_column_naming_conventions() -> None:
    hist_df, fut_df = _make_collinear_price_promo_panel(n_days=70, horizon=7, seed=33)
    cfg = RunConfig.model_validate(
        {
            "run_name": "f4-inv",
            "data": {"source_table": "t", "horizon": 7, "freq": "D"},
            "features": {"future_covariates": ["price", "promo"]},
            "models": ["regression_lags@monotone"],
            "ensemble": {"enabled": False},
        }
    )
    sim_df = simulate_scenarios(
        hist_df,
        cfg,
        {"promo_boom": {"promo": 1.0, "price": "*0.85"}},
        future_covariates_df=fut_df,
        unit_cost=10.0,
        include_inventory_impact=True,
    )
    for col in (
        "gross_margin_delta",
        "delta_reorder_point",
        "delta_safety_stock",
        "delta_order_qty",
        "reorder_point_delta",
        "safety_stock_delta",
        "recommended_order_qty_delta",
    ):
        assert col in sim_df.columns
    assert np.allclose(sim_df["delta_reorder_point"], sim_df["reorder_point_delta"])
    assert np.allclose(sim_df["delta_safety_stock"], sim_df["safety_stock_delta"])
    assert np.allclose(sim_df["delta_order_qty"], sim_df["recommended_order_qty_delta"])
    promo_rows = sim_df[sim_df["scenario"] == "promo_boom"]
    assert (promo_rows["delta_reorder_point"] > 0.0).all()


def test_tier1_f4_ensemble_propagation_calculated_and_learned_strategies() -> None:
    hist_df, fut_df = _make_collinear_price_promo_panel(n_days=80, horizon=7, seed=35)
    cfg = RunConfig.model_validate(
        {
            "run_name": "f4-ens",
            "data": {"source_table": "t", "horizon": 7, "freq": "D"},
            "features": {"future_covariates": ["price", "promo"]},
            "models": ["regression_lags@monotone", "naive_seasonal"],
            "backtest": {"enabled": True, "n_folds": 2, "horizon": 7, "step": 7, "min_train": 40},
            "ensemble": {"enabled": True, "strategies": ["mean", "nnls"]},
        }
    )
    sim_df = simulate_scenarios(
        hist_df,
        cfg,
        {"cut_10": {"price": "-10%"}},
        future_covariates_df=fut_df,
    )
    models_out = set(sim_df["model_type"].unique())
    assert "ensemble_mean" in models_out
    assert "ensemble_nnls" in models_out


def test_tier1_f4_hierarchy_reconciliation_coherence_under_shocks() -> None:
    hist_df, fut_df = _make_collinear_price_promo_panel(n_days=75, horizon=7, seed=37)
    cfg = RunConfig.model_validate(
        {
            "run_name": "f4-hier",
            "data": {"source_table": "t", "horizon": 7, "freq": "D"},
            "features": {"future_covariates": ["price", "promo"]},
            "models": ["regression_lags@monotone"],
            "ensemble": {"enabled": False},
            "hierarchy": {
                "enabled": True,
                "levels": [["region"]],
                "reconciliation_methods": ["mint_nonneg"],
            },
        }
    )
    sim_df = simulate_scenarios(
        hist_df,
        cfg,
        {"promo_on": {"promo": 1.0}},
        future_covariates_df=fut_df,
        reconcile=True,
    )
    _, hier_spec = build_hierarchy(hist_df, cfg)
    for scen_name, s_grp in sim_df.groupby("scenario"):
        max_disc = verify_coherence(s_grp, hier_spec, date_col="ds")
        assert max_disc < 1e-6, f"Scenario {scen_name} failed coherence: {max_disc}"


def test_tier1_f4_summarize_and_compare_scenario_margin_curves() -> None:
    hist_df, fut_df = _make_collinear_price_promo_panel(n_days=70, horizon=7, seed=39)
    cfg = RunConfig.model_validate(
        {
            "run_name": "f4-sum",
            "data": {"source_table": "t", "horizon": 7, "freq": "D"},
            "features": {"future_covariates": ["price", "promo"]},
            "models": ["regression_lags@monotone"],
            "ensemble": {"enabled": False},
        }
    )
    sim_df = simulate_scenarios(
        hist_df,
        cfg,
        {
            "promo_only": {"promo": 1.0},
            "deep_discount": {"price": "*0.65", "promo": 1.0},
        },
        future_covariates_df=fut_df,
        unit_cost=15.0,
        include_inventory_impact=True,
    )
    summary = summarize_scenarios(sim_df)
    ranked = compare_scenario_margin_curves(sim_df, ts_id="SKU_A")
    assert not summary.empty
    assert not ranked.empty
    assert "gross_margin_delta" in ranked.columns
    assert "delta_reorder_point" in ranked.columns
    assert "reorder_point_delta" in ranked.columns
    assert ranked["gross_margin_delta"].is_monotonic_decreasing


# --- F5: Step 0 Auto-Forecaster & Branch 2B Integration -----------------------


def test_tier1_f5_inspect_panel_for_auto_detects_monotonic_drivers() -> None:
    hist_df, fut_df = _make_collinear_price_promo_panel(n_days=70, horizon=7, seed=41)
    profile = inspect_panel_for_auto(hist_df, future_df=fut_df, horizon=7)
    assert profile.monotonic_constraints == {"price": -1, "promo": 1}


def test_tier1_f5_build_config_populates_monotonic_constraints_and_holiday_window() -> None:
    hist_df, fut_df = _make_collinear_price_promo_panel(n_days=70, horizon=7, seed=43)
    bp = build_config(hist_df, future_df=fut_df, holidays=["US"], horizon=7, where="local")
    assert bp.config.features.monotonic_constraints == {"price": -1, "promo": 1}
    assert bp.config.features.holiday_window is not None
    assert bp.config.features.holiday_window.enabled is True


def test_tier1_f5_propose_autofocus_pass_injects_monotone_variants() -> None:
    hist_df, fut_df = _make_collinear_price_promo_panel(n_days=70, horizon=7, seed=45)
    bp = build_config(hist_df, future_df=fut_df, horizon=7, where="local")
    proposal = propose_autofocus_pass(
        bp.profile,
        bp.config.model_dump(mode="json"),
        2,
        {"positive_covariates": ["price", "promo"], "covariate_fva_pct": 2.5},
        tier="wide_field",
        where="local",
    )
    assert any("@monotone" in m for m in proposal["models"])
    assert any(m["action"] == "inject_monotone_guardrails" for m in proposal["mutations"])


def test_tier1_f5_propose_autofocus_dag_branches_2b_covariates_and_horizon() -> None:
    hist_df, fut_df = _make_collinear_price_promo_panel(n_days=70, horizon=7, seed=47)
    bp = build_config(hist_df, future_df=fut_df, horizon=7, where="local")
    branches = propose_autofocus_dag_branches(
        bp.profile,
        bp.config.model_dump(mode="json"),
        2,
        {
            "residual_heteroskedasticity": 0.45,
            "positive_covariates": ["price", "promo"],
            "covariate_fva_pct": 2.5,
        },
        tier="wide_field",
        where="local",
    )
    branch_2b = [b for b in branches if b["pass_id"] == "2B"]
    assert len(branch_2b) == 1
    assert any("@monotone" in m for m in branch_2b[0]["models"])


def test_tier1_f5_compile_autoforecast_plan_branch_2b_includes_monotone() -> None:
    hist_df, fut_df = _make_collinear_price_promo_panel(n_days=70, horizon=7, seed=49)
    f = auto_forecast(
        hist_df,
        future_df=fut_df,
        tier="first_light",
        where="local",
        horizon=7,
        autofocus=True,
        max_passes=2,
    )
    session = f.auto_session
    assert session is not None
    assert session.profile.monotonic_constraints == {"price": -1, "promo": 1}
    assert any(any("@monotone" in m for m in p.config.models) for p in session.passes)


def test_tier1_f5_auto_forecast_session_methods_promo_elasticity_and_scenarios() -> None:
    hist_df, fut_df = _make_collinear_price_promo_panel(n_days=70, horizon=7, seed=51)
    f = auto_forecast(
        hist_df,
        future_df=fut_df,
        tier="first_light",
        where="local",
        horizon=7,
        autofocus=False,
    )
    session = f.auto_session
    assert isinstance(session, AutoForecastSession)
    el_df = session.promo_elasticity_df()
    assert not el_df.empty
    assert "baseline_yhat" in el_df.columns
    sim_df = session.simulate_scenarios(
        {"price_cut": {"price": "-10%"}},
        unit_cost=10.0,
        include_inventory_impact=True,
    )
    assert not sim_df.empty
    assert "gross_margin_delta" in sim_df.columns


# --- F6: SDK, Playground & Report Surfaces ------------------------------------


def test_tier1_f6_forecaster_promo_elasticity_df_and_alias() -> None:
    df, _ = make_confounded_promo_dgp(n_series=2, n_days=120, seed=53)
    f = Forecaster(
        {
            "run_name": "f6-el",
            "data": {"source_table": "t", "horizon": 7, "freq": "D"},
            "models": ["naive_seasonal"],
        }
    )
    decomp1 = f.promo_elasticity_df(df)
    decomp2 = f.promo_elasticity(df)
    assert not decomp1.empty
    assert list(decomp1.columns) == list(PROMO_ELASTICITY_COLUMNS)
    assert decomp1.equals(decomp2)


def test_tier1_f6_forecaster_simulate_scenarios_auto_session_resolution() -> None:
    hist_df, fut_df = _make_collinear_price_promo_panel(n_days=70, horizon=7, seed=55)
    f = Forecaster.auto(
        hist_df,
        future_df=fut_df,
        tier="first_light",
        where="local",
        horizon=7,
        autofocus=False,
    )
    sim_df = f.simulate_scenarios(
        {"promo_push": {"promo": 1.0, "price": "-10%"}},
        unit_cost=10.0,
        include_inventory_impact=True,
    )
    assert not sim_df.empty
    assert {
        "gross_margin_delta",
        "delta_reorder_point",
        "delta_safety_stock",
        "delta_order_qty",
    }.issubset(sim_df.columns)


def test_tier1_f6_playground_run_promo_elasticity_and_simulate_scenarios() -> None:
    run = run_model("regression_lags", horizon=7, with_exog=True, future_covariates=["promo_flag"])
    el_df = run.promo_elasticity_df()
    assert not el_df.empty
    assert "baseline_yhat" in el_df.columns
    sim_df = run.simulate_scenarios(
        {"promo_active": {"promo_flag": 1.0}},
        unit_cost=5.0,
        include_inventory_impact=True,
    )
    assert not sim_df.empty
    assert "gross_margin_delta" in sim_df.columns


def test_tier1_f6_bakeoff_promo_elasticity_and_simulate_scenarios() -> None:
    bo = bakeoff(models=["naive_seasonal", "regression_lags"], horizon=7, n_folds=2, with_exog=True)
    el_df = bo.promo_elasticity_df()
    assert not el_df.empty
    sim_df = bo.simulate_scenarios(
        {"promo_active": {"promo_flag": 1.0}},
        unit_cost=5.0,
        include_inventory_impact=True,
    )
    assert not sim_df.empty
    assert "gross_margin_delta" in sim_df.columns


def test_tier1_f6_report_payload_and_html_include_promo_elasticity_and_scenarios(
    tmp_path: pathlib.Path,
) -> None:
    bo = bakeoff(models=["naive_seasonal", "regression_lags"], horizon=7, n_folds=2, with_exog=True)
    el_df = bo.promo_elasticity_df()
    sim_df = bo.simulate_scenarios(
        {"promo_active": {"promo_flag": 1.0}},
        unit_cost=5.0,
        include_inventory_impact=True,
    )
    payload = build_report_payload(bo, promo_elasticity_df=el_df, scenarios_df=sim_df)
    assert len(payload["promo_elasticity"]) > 0
    assert len(payload["scenarios"]) > 0
    html = render_html_report(payload)
    assert "Promo &amp; Price Elasticity" in html
    out_file = export_report(
        bo,
        tmp_path / "track_j_report.html",
        promo_elasticity_df=el_df,
        scenarios_df=sim_df,
    )
    assert out_file.exists()


def test_tier1_f6_top_level_package_lazy_exports_all_track_j_symbols() -> None:
    for sym in (
        "MONOTONE_SUPPORTED_MODELS",
        "infer_monotonic_constraints",
        "resolve_monotonic_constraints",
        "build_monotonic_constraint_vector",
        "resolve_model_monotonic_vector",
        "fit_sign_constrained_ridge",
        "HOLIDAY_WINDOW_COLUMNS",
        "HolidayWindowConfig",
        "build_holiday_window_features",
        "PROMO_ELASTICITY_COLUMNS",
        "compute_promo_elasticity",
        "compute_promo_elasticity_decomposition",
        "promo_elasticity_df",
        "summarize_promo_elasticity",
        "plot_promo_elasticity",
        "make_confounded_promo_dgp",
        "apply_scenario_shocks",
        "compare_scenario_margin_curves",
        "simulate_scenarios",
        "summarize_scenarios",
        "plot_scenarios",
    ):
        assert hasattr(sf, sym), f"Missing top-level export: {sym}"


# --- F7: MCP Server Tools (`mcp.py`) ------------------------------------------


def test_tier1_f7_mcp_list_tools_includes_promo_elasticity_and_simulate_scenarios() -> None:
    srv = McpServer()
    tools_by_name = {t["name"]: t for t in srv.list_tools()}
    assert "run_playground" in tools_by_name
    props = tools_by_name["run_playground"]["inputSchema"]["properties"]
    assert "include_promo_elasticity" in props
    assert "scenarios" in props


def test_tier1_f7_mcp_promo_elasticity_tool_execution_and_summary() -> None:
    srv = McpServer()
    res = srv.call_tool("promo_elasticity", {"n_series": 2, "n_days": 120})
    assert res["isError"] is False
    out = json.loads(res["content"][0]["text"])
    assert out["status"] == "ok"
    assert len(out["summary"]) == 2
    assert "baseline_yhat" in out["columns"]


def test_tier1_f7_mcp_simulate_scenarios_tool_with_unit_cost_and_inventory() -> None:
    srv = McpServer()
    res = srv.call_tool(
        "simulate_scenarios",
        {
            "models": ["regression_lags@monotone"],
            "unit_cost": 8.0,
            "include_inventory_impact": True,
        },
    )
    assert res["isError"] is False
    out = json.loads(res["content"][0]["text"])
    assert out["status"] == "ok"
    assert len(out["summary"]) > 0
    assert "gross_margin_delta" in out["summary"][0]
    assert "delta_reorder_point" in out["summary"][0]


def test_tier1_f7_mcp_json_rpc_dispatch_for_track_j_tools() -> None:
    srv = McpServer()
    resp = srv.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 101,
            "method": "tools/call",
            "params": {"name": "promo_elasticity", "arguments": {"n_series": 2, "n_days": 90}},
        }
    )
    assert resp is not None
    assert resp["result"]["isError"] is False


def test_tier1_f7_mcp_simulate_scenarios_with_base_config_and_custom_shocks() -> None:
    srv = McpServer()
    res = srv.call_tool(
        "simulate_scenarios",
        {
            "config": {
                "run_name": "mcp-custom",
                "data": {"source_table": "t", "horizon": 7, "freq": "D"},
                "features": {"future_covariates": ["price", "promo"]},
                "models": ["regression_lags@monotone"],
            },
            "scenarios": {"custom_shock": {"price": "-12%", "promo": 1.0}},
            "unit_cost": 5.0,
        },
    )
    assert res["isError"] is False
    out = json.loads(res["content"][0]["text"])
    assert out["status"] == "ok"
    assert any(r["scenario"] == "custom_shock" for r in out["summary"])


# ==============================================================================
# TIER 2: Boundary, Corner & Negative Cases (>=5 tests per feature area F1-F7)
# ==============================================================================


# --- F1 Boundary/Negative -----------------------------------------------------


def test_tier2_f1_invalid_direction_int_raises_value_error() -> None:
    with pytest.raises(ValueError, match="must be -1, 0, or 1"):
        validate_monotonic_constraints_map({"price": -5})


def test_tier2_f1_invalid_direction_string_raises_value_error() -> None:
    with pytest.raises(ValueError, match="must be an integer in"):
        validate_monotonic_constraints_map({"price": "sideways"})


def test_tier2_f1_boolean_direction_rejected() -> None:
    with pytest.raises(ValueError, match="must be an integer in"):
        validate_monotonic_constraints_map({"price": True})


def test_tier2_f1_empty_features_and_unmatched_columns_return_zero_vector() -> None:
    assert list(build_monotonic_constraint_vector([], {"price": -1})) == []
    assert list(build_monotonic_constraint_vector(["dow", "month"], {"price": -1})) == [0, 0]


def test_tier2_f1_collinear_singular_design_matrix_in_sign_constrained_ridge() -> None:
    x_col = np.linspace(1.0, 10.0, 30)
    X_singular = np.column_stack([x_col, x_col])  # rank-deficient identical columns
    y = 100.0 + 3.0 * x_col
    w = fit_sign_constrained_ridge(X_singular, y, alpha=1.0, constraint_vec=[-1, -1])
    assert np.all(np.isfinite(w))
    assert w[1] <= 1e-9 and w[2] <= 1e-9


def test_tier2_f1_direct_strategy_disables_recursive_lag_nonnegativity() -> None:
    vec_rec = resolve_model_monotonic_vector(
        ["lag_1", "lag_7", "price"],
        {"strategy": "recursive"},
        {"price": -1},
        recursive_lags_nonnegative=True,
    )
    vec_dir = resolve_model_monotonic_vector(
        ["lag_1", "lag_7", "price"],
        {"strategy": "direct"},
        {"price": -1},
        recursive_lags_nonnegative=False,
    )
    assert list(vec_rec) == [1, 1, -1]
    assert list(vec_dir) == [0, 0, -1]


# --- F2 Boundary/Negative -----------------------------------------------------


def test_tier2_f2_negative_or_excessive_holiday_window_days_rejected() -> None:
    with pytest.raises(ValueError):
        HolidayWindowConfig(pre_holiday_days=-1, post_holiday_days=3)
    with pytest.raises(ValueError):
        HolidayWindowConfig(pre_holiday_days=45, post_holiday_days=3)


def test_tier2_f2_invalid_tuple_length_or_type_in_coerce_holiday_window_raises() -> None:
    with pytest.raises(ValueError, match="must have length 2"):
        coerce_holiday_window_config((1, 2, 3))
    with pytest.raises(ValueError, match="must be a HolidayWindowConfig"):
        coerce_holiday_window_config(42)


def test_tier2_f2_zero_pre_and_post_days_or_disabled_returns_empty_columns() -> None:
    assert holiday_window_columns(None) == []
    assert holiday_window_columns(False) == []
    assert holiday_window_columns(HolidayWindowConfig(enabled=False)) == []
    assert (
        holiday_window_columns(HolidayWindowConfig(pre_holiday_days=0, post_holiday_days=0)) == []
    )


def test_tier2_f2_adjacent_overlapping_holidays_resolve_nearest_distance() -> None:
    dates = pd.date_range("2025-12-22", "2026-01-03", freq="D")
    holidays_df = pd.DataFrame(
        {
            "ds": [pd.Timestamp("2025-12-25"), pd.Timestamp("2026-01-01")],
            "holiday": ["Christmas", "NewYear"],
        }
    )
    hw = pd.DataFrame(
        build_holiday_window_features(
            dates,
            holidays_df,
            HolidayWindowConfig(pre_holiday_days=5, post_holiday_days=3, kernel="linear"),
        ),
        index=dates,
    )
    # Dec 26 is 1 day after Christmas AND 6 days before NewYear
    assert hw.loc["2025-12-26", "post_holiday_window"] == pytest.approx(1.0)
    # Dec 31 is 1 day before NewYear
    assert hw.loc["2025-12-31", "pre_holiday_window"] == pytest.approx(1.0)


def test_tier2_f2_empty_holiday_source_returns_zero_columns_gracefully() -> None:
    dates = pd.date_range("2025-03-01", periods=10, freq="D")
    hw = build_holiday_window_features(dates, [], HolidayWindowConfig())
    assert (hw["pre_holiday_window"] == 0.0).all()
    assert (hw["post_holiday_window"] == 0.0).all()


# --- F3 Boundary/Negative -----------------------------------------------------


def test_tier2_f3_empty_or_non_dataframe_returns_empty_schema_safely() -> None:
    empty_out = compute_promo_elasticity_decomposition(pd.DataFrame())
    assert empty_out.empty
    assert list(empty_out.columns) == list(PROMO_ELASTICITY_COLUMNS)


def test_tier2_f3_missing_required_columns_returns_empty_schema_safely() -> None:
    bad_df = pd.DataFrame({"ts_id": ["A"], "ds": ["2025-01-01"]})
    out = compute_promo_elasticity_decomposition(bad_df)
    assert out.empty
    assert list(out.columns) == list(PROMO_ELASTICITY_COLUMNS)


def test_tier2_f3_constant_zero_promo_or_short_series_returns_zero_lift_safely() -> None:
    short_df = pd.DataFrame(
        {
            "ts_id": ["S1"] * 8,
            "ds": pd.date_range("2025-01-01", periods=8, freq="D"),
            "y": [10.0] * 8,
            "promo": [0.0] * 8,
        }
    )
    decomp = compute_promo_elasticity_decomposition(short_df)
    assert len(decomp) == 8
    assert (decomp["incremental_promo_lift"] == 0.0).all()
    assert (decomp["net_causal_elasticity"] == 0.0).all()


def test_tier2_f3_single_series_panel_has_zero_cannibalization_effect() -> None:
    df, _ = make_confounded_promo_dgp(n_series=1, n_days=100, seed=61)
    decomp = compute_promo_elasticity_decomposition(df)
    assert (decomp["cannibalization_effect"] == 0.0).all()


def test_tier2_f3_nan_values_in_promo_or_price_handled_without_nan_leakage() -> None:
    df, _ = make_confounded_promo_dgp(n_series=2, n_days=100, seed=63)
    df.loc[df.index[:5], "promo"] = np.nan
    df.loc[df.index[5:10], "price"] = np.nan
    decomp = compute_promo_elasticity_decomposition(df)
    assert not decomp["baseline_yhat"].isna().any()
    assert not decomp["incremental_promo_lift"].isna().any()


# --- F4 Boundary/Negative -----------------------------------------------------


def test_tier2_f4_empty_panel_or_empty_scenarios_raises_config_error() -> None:
    cfg = RunConfig.model_validate(
        {
            "run_name": "err",
            "data": {"source_table": "t", "horizon": 7},
            "models": ["naive_seasonal"],
        }
    )
    with pytest.raises(ConfigError, match="non-empty"):
        simulate_scenarios(pd.DataFrame(), cfg, {"s1": {"price": 10.0}})
    df = pd.DataFrame(
        {
            "ts_id": ["A"] * 20,
            "ds": pd.date_range("2025-01-01", periods=20, freq="D"),
            "y": range(20),
        }
    )
    with pytest.raises(ConfigError, match="at least one named scenario"):
        simulate_scenarios(df, cfg, {})


def test_tier2_f4_unknown_shock_operator_or_malformed_spec_raises_config_error() -> None:
    fut_df = pd.DataFrame(
        {
            "ts_id": ["A"] * 5,
            "ds": pd.date_range("2025-01-01", periods=5, freq="D"),
            "price": [10.0] * 5,
        }
    )
    with pytest.raises((ConfigError, ValueError), match="Unsupported shock op"):
        apply_scenario_shocks(fut_df, {"price": {"op": "divide_by_zero", "value": 1.0}})
    with pytest.raises((ConfigError, ValueError, KeyError)):
        apply_scenario_shocks(fut_df, {"price": {"op": "set"}})


def test_tier2_f4_date_window_outside_horizon_leaves_trajectory_equal_to_baseline() -> None:
    hist_df, fut_df = _make_collinear_price_promo_panel(n_days=60, horizon=7, seed=65)
    cfg = RunConfig.model_validate(
        {
            "run_name": "out-win",
            "data": {"source_table": "t", "horizon": 7, "freq": "D"},
            "features": {"future_covariates": ["price", "promo"]},
            "models": ["regression_lags@monotone"],
            "ensemble": {"enabled": False},
        }
    )
    sim_df = simulate_scenarios(
        hist_df,
        cfg,
        {
            "past_window": {
                "price": {
                    "op": "mul",
                    "value": 0.5,
                    "start_date": "2020-01-01",
                    "end_date": "2020-01-05",
                }
            }
        },
        future_covariates_df=fut_df,
    )
    pw = sim_df[sim_df["scenario"] == "past_window"]
    assert np.allclose(pw["delta_yhat"], 0.0)


def test_tier2_f4_zero_price_change_sets_implied_elasticity_to_nan_without_div_zero() -> None:
    hist_df, fut_df = _make_collinear_price_promo_panel(n_days=60, horizon=7, seed=67)
    cfg = RunConfig.model_validate(
        {
            "run_name": "zero-p",
            "data": {"source_table": "t", "horizon": 7, "freq": "D"},
            "features": {"future_covariates": ["price", "promo"]},
            "models": ["regression_lags@monotone"],
            "ensemble": {"enabled": False},
        }
    )
    sim_df = simulate_scenarios(
        hist_df,
        cfg,
        {"promo_only": {"promo": 1.0}},
        future_covariates_df=fut_df,
    )
    po = sim_df[sim_df["scenario"] == "promo_only"]
    assert po["implied_elasticity"].isna().all()
    assert np.all(np.isfinite(po["revenue_delta"]))


def test_tier2_f4_summarize_and_compare_empty_dataframe_return_clean_schema() -> None:
    empty_sum = summarize_scenarios(pd.DataFrame())
    empty_cmp = compare_scenario_margin_curves(pd.DataFrame())
    assert empty_sum.empty and "scenario" in empty_sum.columns
    assert empty_cmp.empty and "scenario" in empty_cmp.columns


# --- F5 Boundary/Negative -----------------------------------------------------


def test_tier2_f5_weekly_and_monthly_panels_do_not_auto_enable_daily_holiday_window() -> None:
    dates_w = pd.date_range("2023-01-01", periods=60, freq="W-SUN")
    df_w = pd.DataFrame({"ts_id": "W1", "ds": dates_w, "y": np.linspace(10.0, 30.0, 60)})
    bp_w = build_config(df_w, holidays=["US"], horizon=4, where="local")
    assert bp_w.config.features.holiday_window is None


def test_tier2_f5_univariate_panel_without_price_promo_has_empty_monotonic_constraints() -> None:
    df_u = _make_moving_holiday_dgp(n_days=80)
    bp_u = build_config(df_u, horizon=7, where="local")
    assert bp_u.profile.monotonic_constraints == {}
    assert bp_u.config.features.monotonic_constraints == {}


def test_tier2_f5_propose_autofocus_pass_skips_monotone_when_already_present() -> None:
    hist_df, fut_df = _make_collinear_price_promo_panel(n_days=70, horizon=7, seed=71)
    bp = build_config(
        hist_df,
        future_df=fut_df,
        models=["lightgbm@monotone", "naive_seasonal"],
        horizon=7,
        where="local",
    )
    prop = propose_autofocus_pass(
        bp.profile,
        bp.config.model_dump(mode="json"),
        2,
        {"positive_covariates": ["price", "promo"], "covariate_fva_pct": 5.0},
        tier="wide_field",
        where="local",
    )
    assert not any(m["action"] == "inject_monotone_guardrails" for m in prop["mutations"])


def test_tier2_f5_auto_session_simulate_scenarios_without_data_raises_data_error() -> None:
    hist_df, fut_df = _make_collinear_price_promo_panel(n_days=60, horizon=7, seed=73)
    f = auto_forecast(
        hist_df, future_df=fut_df, tier="first_light", where="local", horizon=7, autofocus=False
    )
    session = f.auto_session
    assert session is not None
    session_no_data = AutoForecastSession(
        blueprint=session.blueprint,
        profile=session.profile,
        tier=session.tier,
        where=session.where,
        goal=session.goal,
        max_minutes=session.max_minutes,
        max_usd=session.max_usd,
        min_confidence=session.min_confidence,
        max_passes=1,
        passes=session.passes,
        predictions_df=session.predictions_df,
        leaderboard_df=session.leaderboard_df,
        oof_df=session.oof_df,
        fva_df=session.fva_df,
        horizon_decay_df=session.horizon_decay_df,
        bias_df=session.bias_df,
        attributions_df=session.attributions_df,
        data_df=None,
    )
    with pytest.raises(DataError, match="No in-memory panel"):
        session_no_data.simulate_scenarios({"s1": {"price": "*0.9"}})


def test_tier2_f5_auto_session_promo_elasticity_with_empty_data_returns_empty_df() -> None:
    hist_df, fut_df = _make_collinear_price_promo_panel(n_days=60, horizon=7, seed=75)
    f = auto_forecast(
        hist_df, future_df=fut_df, tier="first_light", where="local", horizon=7, autofocus=False
    )
    assert f.auto_session is not None
    assert f.auto_session.promo_elasticity_df(data=pd.DataFrame()).empty


# --- F6 Boundary/Negative -----------------------------------------------------


def test_tier2_f6_forecaster_simulate_scenarios_without_data_uses_sample_fallback() -> None:
    f = Forecaster(
        {
            "run_name": "f6-fallback",
            "data": {"source_table": "t", "horizon": 7, "freq": "D"},
            "features": {"future_covariates": ["price_index", "promo_flag"]},
            "models": ["regression_lags@monotone"],
            "ensemble": {"enabled": False},
        }
    )
    sim_df = f.simulate_scenarios({"promo_burst": {"promo_flag": 1.0}}, unit_cost=4.0)
    assert not sim_df.empty
    assert "gross_margin_delta" in sim_df.columns


def test_tier2_f6_report_payload_handles_missing_promo_or_scenario_columns_gracefully() -> None:
    bo = bakeoff(models=["naive_seasonal", "theta"], horizon=7, n_folds=2, with_exog=False)
    payload = build_report_payload(bo)
    assert payload["promo_elasticity"] == []
    assert payload["scenarios"] == []


def test_tier2_f6_malformed_at_tag_variant_raises_config_error() -> None:
    with pytest.raises(ValueError):
        RunConfig.model_validate(
            {
                "run_name": "bad-tag",
                "data": {"source_table": "t", "horizon": 7},
                "models": ["lightgbm@"],
            }
        )


def test_tier2_f6_elided_defaults_preserve_deterministic_run_id_stability() -> None:
    cfg1 = RunConfig.model_validate(
        {
            "run_name": "id-check",
            "data": {"source_table": "t", "horizon": 7},
            "models": ["naive_seasonal"],
        }
    )
    cfg2 = RunConfig.model_validate(
        {
            "run_name": "id-check",
            "data": {"source_table": "t", "horizon": 7},
            "features": {"monotonic_constraints": {}, "holiday_window": None},
            "models": ["naive_seasonal"],
        }
    )
    assert make_run_id(cfg1) == make_run_id(cfg2)


def test_tier2_f6_model_spec_hash_changes_when_monotonic_or_holiday_window_changes() -> None:
    cfg_base = RunConfig.model_validate(
        {
            "run_name": "hash-check",
            "data": {"source_table": "t", "horizon": 7, "freq": "D"},
            "features": {"future_covariates": ["price"], "holidays": ["US"]},
            "models": ["regression_lags", "regression_lags@monotone"],
        }
    )
    h_unconstrained = make_model_spec_hash("regression_lags", cfg_base)
    h_monotone = make_model_spec_hash("regression_lags@monotone", cfg_base)
    assert h_unconstrained != h_monotone

    cfg_hw = cfg_base.model_copy(
        update={
            "features": cfg_base.features.model_copy(
                update={
                    "holiday_window": HolidayWindowConfig(pre_holiday_days=7, post_holiday_days=3)
                }
            )
        }
    )
    assert make_model_spec_hash("regression_lags", cfg_hw) != h_unconstrained


# --- F7 Boundary/Negative -----------------------------------------------------


def test_tier2_f7_mcp_unknown_tool_returns_is_error_true() -> None:
    srv = McpServer()
    resp = srv.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 999,
            "method": "tools/call",
            "params": {"name": "nonexistent_tool", "arguments": {}},
        }
    )
    assert resp is not None
    assert resp["result"]["isError"] is True


def test_tier2_f7_mcp_promo_elasticity_with_custom_inline_records() -> None:
    df, _ = make_confounded_promo_dgp(n_series=1, n_days=60, seed=81)
    df["ds"] = df["ds"].dt.strftime("%Y-%m-%d")
    srv = McpServer()
    res = srv.call_tool("promo_elasticity", {"records": df.to_dict(orient="records")})
    assert res["isError"] is False
    out = json.loads(res["content"][0]["text"])
    assert out["status"] == "ok"
    assert len(out["summary"]) == 1


def test_tier2_f7_mcp_simulate_scenarios_with_custom_inline_records() -> None:
    hist_df, _ = _make_collinear_price_promo_panel(n_days=50, horizon=7, seed=83)
    sub = hist_df[hist_df["ts_id"] == "SKU_A"].copy()
    sub["ds"] = sub["ds"].dt.strftime("%Y-%m-%d")
    srv = McpServer()
    res = srv.call_tool(
        "simulate_scenarios",
        {
            "records": sub.to_dict(orient="records"),
            "models": ["regression_lags@monotone"],
            "scenarios": {"cut": {"price": "-10%"}},
        },
    )
    assert res["isError"] is False
    out = json.loads(res["content"][0]["text"])
    assert out["status"] == "ok"


def test_tier2_f7_mcp_simulate_scenarios_invalid_operator_returns_error_payload() -> None:
    srv = McpServer()
    resp = srv.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 888,
            "method": "tools/call",
            "params": {
                "name": "simulate_scenarios",
                "arguments": {"scenarios": {"bad": {"price": {"op": "invalid_op", "value": 1.0}}}},
            },
        }
    )
    assert resp is not None
    assert resp["result"]["isError"] is True


def test_tier2_f7_mcp_promo_elasticity_missing_column_returns_empty_summary_gracefully() -> None:
    srv = McpServer()
    resp = srv.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 889,
            "method": "tools/call",
            "params": {
                "name": "promo_elasticity",
                "arguments": {"records": [{"ts_id": "A", "ds": "2025-01-01"}]},
            },
        }
    )
    assert resp is not None
    assert resp["result"]["isError"] is False
    payload = json.loads(resp["result"]["content"][0]["text"])
    assert payload["summary"] == []


# ==============================================================================
# TIER 3: Cross-Feature Pairwise Interactions (8 pairwise tests)
# ==============================================================================


def test_tier3_pair1_monotone_variants_plus_holiday_window_in_run_cell() -> None:
    """Pair 1: F1 (`@monotone`) x F2 (`holiday_window`) inside `run_cell`."""
    hist_df, fut_df = _make_collinear_price_promo_panel(n_days=90, horizon=7, seed=91)
    s_df = hist_df[hist_df["ts_id"] == "SKU_A"].reset_index(drop=True)
    f_df = fut_df[fut_df["ts_id"] == "SKU_A"].reset_index(drop=True)
    cfg = RunConfig.model_validate(
        {
            "run_name": "pair1",
            "data": {"source_table": "t", "horizon": 7, "freq": "D"},
            "features": {
                "future_covariates": ["price", "promo"],
                "holidays": ["US"],
                "holiday_window": {"pre_holiday_days": 7, "post_holiday_days": 3},
            },
            "models": ["regression_lags@monotone"],
        }
    )
    res = run_cell(s_df, "regression_lags@monotone", cfg, future_df=f_df)
    assert res.status == "ok"
    attrs = res.diagnostics.get("feature_attributions", {})
    assert "pre_holiday_window" in attrs
    assert "price" in attrs


def test_tier3_pair2_monotone_variants_plus_simulate_scenarios_sign_coherent_lift() -> None:
    """Pair 2: F1 (`@monotone`) x F4 (`simulate_scenarios`) sign-coherent counterfactuals."""
    hist_df, fut_df = _make_collinear_price_promo_panel(n_days=90, horizon=7, seed=93)
    cfg = RunConfig.model_validate(
        {
            "run_name": "pair2",
            "data": {"source_table": "t", "horizon": 7, "freq": "D"},
            "features": {"future_covariates": ["price", "promo"]},
            "models": ["regression_lags@monotone", "lightgbm@monotone"],
            "ensemble": {"enabled": False},
        }
    )
    sim_df = simulate_scenarios(
        hist_df,
        cfg,
        {
            "price_hike_20pct": {"price": "+20%"},
            "price_cut_20pct_promo": {"price": "-20%", "promo": 1.0},
        },
        future_covariates_df=fut_df,
    )
    hike = sim_df[sim_df["scenario"] == "price_hike_20pct"]
    cut_promo = sim_df[sim_df["scenario"] == "price_cut_20pct_promo"]
    assert (hike["delta_yhat"] <= 1e-6).all()
    assert (cut_promo["delta_yhat"] >= -1e-6).all()


def test_tier3_pair3_holiday_window_plus_promo_elasticity_deconfounding_holiday_spikes() -> None:
    """Pair 3: F2 (`holiday_window`) x F3 (`promo_elasticity_df`) deconfounds holiday demand."""
    df, truth = make_confounded_promo_dgp(n_series=2, n_days=365, true_promo_lift=28.0, seed=95)
    decomp = compute_promo_elasticity_decomposition(df, holidays=["US"])
    summary = summarize_promo_elasticity(decomp)
    assert abs(float(summary["mean_causal_promo_lift"].mean()) - truth["true_promo_lift"]) <= 6.0


def test_tier3_pair4_monotone_variants_plus_ensemble_blending_in_bakeoff() -> None:
    """Pair 4: F1 (`@monotone`) x F6 (`bakeoff` + ensemble blending)."""
    bo = bakeoff(
        models=["regression_lags", "regression_lags@monotone", "lightgbm@monotone"],
        with_exog=True,
        future_covariates=["price_index", "promo_flag"],
        horizon=7,
        n_folds=2,
    )
    lb = bo.leaderboard
    m_col = "model" if "model" in lb.columns else "model_type"
    assert "regression_lags@monotone" in set(lb[m_col])
    assert "lightgbm@monotone" in set(lb[m_col])


def test_tier3_pair5_simulate_scenarios_plus_hierarchy_mint_nonneg_coherence() -> None:
    """Pair 5: F4 (`simulate_scenarios`) x Hierarchy (`mint_nonneg`) + F1 (`@monotone`)."""
    hist_df, fut_df = _make_collinear_price_promo_panel(n_days=80, horizon=7, seed=101)
    cfg = RunConfig.model_validate(
        {
            "run_name": "pair5",
            "data": {"source_table": "t", "horizon": 7, "freq": "D"},
            "features": {"future_covariates": ["price", "promo"]},
            "models": ["regression_lags@monotone", "lightgbm@monotone"],
            "backtest": {"enabled": True, "n_folds": 2, "horizon": 7, "step": 7, "min_train": 40},
            "ensemble": {"enabled": True, "strategies": ["inverse_error"]},
            "hierarchy": {
                "enabled": True,
                "levels": [["region"]],
                "reconciliation_methods": ["mint_nonneg"],
            },
        }
    )
    sim_df = simulate_scenarios(
        hist_df,
        cfg,
        {"promo_push": {"price": "-15%", "promo": 1.0}},
        future_covariates_df=fut_df,
        unit_cost=9.0,
        reconcile=True,
        include_inventory_impact=True,
    )
    _, hier_spec = build_hierarchy(hist_df, cfg)
    for _, grp in sim_df.groupby("scenario"):
        assert verify_coherence(grp, hier_spec, date_col="ds") < 1e-6
    assert (sim_df["yhat"] >= -1e-9).all()


def test_tier3_pair6_simulate_scenarios_plus_inventory_policy_and_gross_margin() -> None:
    """Pair 6: F4 (`simulate_scenarios`) x Inventory Policy x Margin."""
    hist_df, fut_df = _make_collinear_price_promo_panel(n_days=80, horizon=7, seed=103)
    cfg = RunConfig.model_validate(
        {
            "run_name": "pair6",
            "data": {"source_table": "t", "horizon": 7, "freq": "D"},
            "features": {"future_covariates": ["price", "promo"]},
            "models": ["regression_lags@monotone"],
            "ensemble": {"enabled": False},
        }
    )
    sim_df = simulate_scenarios(
        hist_df,
        cfg,
        {
            "moderate_promo": {"price": "-8%", "promo": 1.0},
            "below_cost_dump": {"price": "*0.40", "promo": 1.0},
        },
        future_covariates_df=fut_df,
        unit_cost=14.0,
        include_inventory_impact=True,
    )
    ranked = compare_scenario_margin_curves(sim_df, ts_id="SKU_A")
    # Below-cost dump sells more units (positive delta_reorder_point) but loses gross margin!
    dump_row = ranked[ranked["scenario"] == "below_cost_dump"].iloc[0]
    mod_row = ranked[ranked["scenario"] == "moderate_promo"].iloc[0]
    assert dump_row["delta_units"] >= mod_row["delta_units"]
    assert dump_row["gross_margin_delta"] < mod_row["gross_margin_delta"]
    assert dump_row["delta_reorder_point"] > 0.0


def test_tier3_pair7_auto_forecast_plus_report_export_with_elasticity_and_scenarios(
    tmp_path: pathlib.Path,
) -> None:
    """Pair 7: F5 (`auto_forecast`) x F3 (`promo_elasticity_df`) x F4 x F6 (`export_report`)."""
    hist_df, fut_df = _make_collinear_price_promo_panel(n_days=75, horizon=7, seed=105)
    f = auto_forecast(
        hist_df,
        future_df=fut_df,
        tier="first_light",
        where="local",
        horizon=7,
        autofocus=False,
    )
    session = f.auto_session
    assert session is not None
    el_df = session.promo_elasticity_df()
    sim_df = session.simulate_scenarios(
        {"promo_campaign": {"price": "-10%", "promo": 1.0}},
        unit_cost=10.0,
        include_inventory_impact=True,
    )
    payload = build_report_payload(session, promo_elasticity_df=el_df, scenarios_df=sim_df)
    assert len(payload["promo_elasticity"]) > 0
    assert len(payload["scenarios"]) > 0
    report_path = export_report(
        session,
        tmp_path / "auto_session_report.html",
        promo_elasticity_df=el_df,
        scenarios_df=sim_df,
    )
    assert report_path.exists()


def test_tier3_pair8_holiday_window_plus_simulate_scenarios_across_holiday_horizon() -> None:
    """Pair 8: F2 (`holiday_window`) x F4 (`simulate_scenarios`) across a holiday window."""
    hist_df, fut_df = _make_collinear_price_promo_panel(n_days=80, horizon=7, seed=107)
    cfg = RunConfig.model_validate(
        {
            "run_name": "pair8",
            "data": {"source_table": "t", "horizon": 7, "freq": "D"},
            "features": {
                "future_covariates": ["price", "promo"],
                "holidays": ["US"],
                "holiday_window": {"pre_holiday_days": 7, "post_holiday_days": 3},
            },
            "models": ["regression_lags@monotone"],
            "ensemble": {"enabled": False},
        }
    )
    sim_df = simulate_scenarios(
        hist_df,
        cfg,
        {"holiday_promo": {"promo": 1.0, "price": "-10%"}},
        future_covariates_df=fut_df,
        unit_cost=10.0,
    )
    assert not sim_df.empty
    assert (sim_df.loc[sim_df["scenario"] == "holiday_promo", "delta_yhat"] >= -1e-6).all()


# ==============================================================================
# TIER 4: Real-World Application Scenarios (5 Scenarios Proving AC 1.1 - AC 1.4)
# ==============================================================================


def test_tier4_scenario1_collinear_price_promo_monotonicity_and_auto_branch_2b() -> None:
    """Real-World Scenario 1 (AC 1.1):
    On a synthetic retail panel with collinear price & promo where unconstrained trees/ridge flip
    elasticity signs, `features.monotonic_constraints = {"price": -1, "promo": 1}` and `@monotone`
    model variants enforce 100% sign-correct counterfactual responses, and `Forecaster.auto()`
    automatically detects `price*`/`promo*` in Step 0 and evaluates `@monotone` in Branch 2B.
    """
    hist_df, fut_df = _make_collinear_price_promo_panel(n_days=100, horizon=7, seed=42)
    s_df = hist_df[hist_df["ts_id"] == "SKU_A"].reset_index(drop=True)
    f_base = fut_df[fut_df["ts_id"] == "SKU_A"].reset_index(drop=True)

    cfg = RunConfig.model_validate(
        {
            "run_name": "ac11-collinear",
            "data": {"source_table": "t", "horizon": 7, "freq": "D"},
            "features": {
                "future_covariates": ["price", "promo"],
                "monotonic_constraints": {"price": -1, "promo": 1},
            },
            "models": [
                "regression_lags",
                "regression_lags@monotone",
                "lightgbm",
                "lightgbm@monotone",
            ],
        }
    )

    f_price_up = f_base.copy()
    f_price_up["price"] = f_base["price"] + 5.0
    f_promo_on = f_base.copy()
    f_promo_on["promo"] = 1.0
    f_promo_off = f_base.copy()
    f_promo_off["promo"] = 0.0

    # Verify unconstrained model exhibits wrong-signed response on collinear data, while @monotone
    # strictly obeys d(yhat)/d(price) <= 0 and d(yhat)/d(promo) >= 0 across 100% of grid points
    res_uncon_base = run_cell(s_df, "regression_lags", cfg, future_df=f_base)
    res_uncon_pup = run_cell(s_df, "regression_lags", cfg, future_df=f_price_up)
    assert (
        res_uncon_pup.predictions["yhat_raw"].to_numpy()
        > res_uncon_base.predictions["yhat_raw"].to_numpy()
    ).any()

    for mono_m in ("regression_lags@monotone", "lightgbm@monotone"):
        r_base = run_cell(s_df, mono_m, cfg, future_df=f_base)
        r_pup = run_cell(s_df, mono_m, cfg, future_df=f_price_up)
        r_pon = run_cell(s_df, mono_m, cfg, future_df=f_promo_on)
        r_poff = run_cell(s_df, mono_m, cfg, future_df=f_promo_off)
        assert np.all(
            r_pup.predictions["yhat_raw"].to_numpy()
            <= r_base.predictions["yhat_raw"].to_numpy() + 1e-6
        )
        assert np.all(
            r_pon.predictions["yhat_raw"].to_numpy()
            >= r_poff.predictions["yhat_raw"].to_numpy() - 1e-6
        )

    # Verify Forecaster.auto() Step 0 detection and Branch 2B @monotone injection
    bp = build_config(hist_df, future_df=fut_df, tier="wide_field", where="local", horizon=7)
    assert bp.profile.monotonic_constraints == {"price": -1, "promo": 1}
    branches = propose_autofocus_dag_branches(
        bp.profile,
        bp.config.model_dump(mode="json"),
        2,
        {
            "residual_heteroskedasticity": 0.45,
            "positive_covariates": ["price", "promo"],
            "covariate_fva_pct": 2.5,
        },
        tier="wide_field",
        where="local",
    )
    branch_2b = next(b for b in branches if b["pass_id"] == "2B")
    assert any("@monotone" in m for m in branch_2b["models"])


def test_tier4_scenario2_confounded_seasonal_promo_pull_forward_and_cannibalization_dml() -> None:
    """Real-World Scenario 2 (AC 1.2):
    Calling `f.promo_elasticity_df()` on a seasonally confounded promotional panel decomposes
    demand into `baseline_yhat`, `incremental_promo_lift`, `pull_forward_dip`,
    `cannibalization_effect`, and `net_causal_elasticity`, recovering true structural promo lift
    and pull-forward dip within tolerance while beating naive uncontrolled OLS.
    """
    df, truth = make_confounded_promo_dgp(
        n_series=3,
        n_days=365,
        true_promo_lift=30.0,
        true_pull_forward_dip=-12.0,
        true_cannibalization_dip=-8.0,
        seasonal_amplitude=35.0,
        seed=42,
    )
    f = Forecaster(
        {
            "run_name": "ac12-elasticity",
            "data": {"source_table": "t", "horizon": 14, "freq": "D"},
            "features": {"future_covariates": ["promo", "price"]},
            "models": ["naive_seasonal"],
        }
    )
    decomp = f.promo_elasticity_df(df)
    for col in (
        "baseline_yhat",
        "incremental_promo_lift",
        "pull_forward_dip",
        "cannibalization_effect",
        "net_causal_elasticity",
    ):
        assert col in decomp.columns

    summary = summarize_promo_elasticity(decomp)
    est_lift = float(summary["mean_causal_promo_lift"].mean())
    naive_lift = float(summary["mean_naive_ols_lift"].mean())
    assert abs(est_lift - truth["true_promo_lift"]) <= 5.0
    assert abs(est_lift - truth["true_promo_lift"]) < abs(naive_lift - truth["true_promo_lift"])
    assert (decomp["pull_forward_dip"] <= 0.0).all()
    assert float(decomp["pull_forward_dip"].min()) < -3.0
    assert (decomp["cannibalization_effect"] <= 0.0).all()
    assert float(decomp["cannibalization_effect"].sum()) < -10.0


def test_tier4_scenario3_moving_holiday_pre_stockup_and_post_dip_window_wape_lift() -> None:
    """Real-World Scenario 3 (AC 1.3):
    Configuring `features.holiday_window = {"pre_holiday_days": 7, "post_holiday_days": 3}` on a
    daily retail panel with multi-day pre-holiday stock-up and post-holiday dip improves out-of-fold
    backtest WAPE over point-day-only `is_holiday` indicators, and Step 0 auto-enables it on daily
    and hourly panels.
    """
    df = _make_moving_holiday_dgp(n_days=540, seed=123)

    cfg_point = RunConfig.model_validate(
        {
            "run_name": "ac13-point-only",
            "data": {"source_table": "t", "horizon": 14, "freq": "D"},
            "features": {"holidays": ["US"], "holiday_window": None},
            "models": ["regression_lags"],
            "backtest": {
                "enabled": True,
                "n_folds": 4,
                "horizon": 14,
                "step": 30,
                "min_train": 365,
            },
        }
    )
    cfg_window = RunConfig.model_validate(
        {
            "run_name": "ac13-holiday-window",
            "data": {"source_table": "t", "horizon": 14, "freq": "D"},
            "features": {
                "holidays": ["US"],
                "holiday_window": {"pre_holiday_days": 7, "post_holiday_days": 3},
            },
            "models": ["regression_lags"],
            "backtest": {
                "enabled": True,
                "n_folds": 4,
                "horizon": 14,
                "step": 30,
                "min_train": 365,
            },
        }
    )

    res_point = run_cell(df, "regression_lags", cfg_point)
    res_window = run_cell(df, "regression_lags", cfg_window)
    assert res_point.status == "ok" and res_window.status == "ok"
    assert res_window.metrics["wape"] < res_point.metrics["wape"]


def test_tier4_scenario4_hierarchical_counterfactual_scenarios_margin_and_inventory_shift() -> None:
    """Real-World Scenario 4 (AC 1.4):
    Calling `f.simulate_scenarios(..., unit_cost=..., reconcile=True)` with inventory impact
    propagates counterfactual price/promo shocks through base models, the winning ensemble,
    `mint_nonneg` hierarchical reconciliation (`verify_coherence < 1e-6`), gross-margin delta
    (`gross_margin_delta`), and inventory policy shift (`delta_reorder_point`, `delta_safety_stock`,
    `delta_order_qty`).
    """
    hist_df, fut_df = _make_collinear_price_promo_panel(n_days=90, horizon=7, seed=202)
    f = Forecaster(
        {
            "run_name": "ac14-full-chain",
            "data": {"source_table": "t", "horizon": 7, "freq": "D"},
            "features": {
                "future_covariates": ["price", "promo"],
                "monotonic_constraints": {"price": -1, "promo": 1},
            },
            "models": ["regression_lags@monotone", "lightgbm@monotone"],
            "backtest": {"enabled": True, "n_folds": 2, "horizon": 7, "step": 7, "min_train": 50},
            "ensemble": {"enabled": True, "strategies": ["inverse_error"]},
            "hierarchy": {
                "enabled": True,
                "levels": [["region"]],
                "reconciliation_methods": ["mint_nonneg"],
            },
        }
    )
    sim_df = f.simulate_scenarios(
        {
            "promo_10pct_off": {"price": "-10%", "promo": 1.0},
            "price_increase_10pct": {"price": "+10%", "promo": 0.0},
        },
        data=hist_df,
        future_covariates_df=fut_df,
        unit_cost=11.0,
        reconcile=True,
        include_inventory_impact=True,
    )
    for col in (
        "gross_margin_delta",
        "delta_reorder_point",
        "delta_safety_stock",
        "delta_order_qty",
        "reorder_point_delta",
        "safety_stock_delta",
        "recommended_order_qty_delta",
    ):
        assert col in sim_df.columns

    _, hier_spec = build_hierarchy(hist_df, f.config)
    for scen, grp in sim_df.groupby("scenario"):
        disc = verify_coherence(grp, hier_spec, date_col="ds")
        assert disc < 1e-6, f"Coherence violated in {scen}: {disc}"


def test_tier4_scenario5_executive_glass_box_html_report_and_mcp_agent_roundtrip(
    tmp_path: pathlib.Path,
) -> None:
    """Real-World Scenario 5:
    End-to-end practitioner and AI agent workflow combining `Forecaster.auto()`,
    `f.promo_elasticity_df()`, `f.simulate_scenarios()`, `f.export_report()`, and `McpServer`
    JSON-RPC tool calls (`promo_elasticity` and `simulate_scenarios`).
    """
    hist_df, fut_df = _make_collinear_price_promo_panel(n_days=80, horizon=7, seed=303)
    f = Forecaster.auto(
        hist_df,
        future_df=fut_df,
        tier="first_light",
        where="local",
        horizon=7,
        autofocus=False,
    )
    el_df = f.promo_elasticity_df()
    sim_df = f.simulate_scenarios(
        {"q4_promo": {"price": "-12%", "promo": 1.0}},
        unit_cost=10.0,
        include_inventory_impact=True,
    )
    report_path = f.export_report(
        tmp_path / "exec_brief.html",
        promo_elasticity_df=el_df,
        scenarios_df=sim_df,
    )
    html_text = report_path.read_text(encoding="utf-8")
    assert "Promo &amp; Price Elasticity" in html_text
    assert "q4_promo" in html_text

    srv = McpServer()
    r1 = srv.call_tool("promo_elasticity", {"n_series": 2, "n_days": 90})
    r2 = srv.call_tool(
        "simulate_scenarios",
        {"unit_cost": 6.0, "include_inventory_impact": True},
    )
    assert r1["isError"] is False and r2["isError"] is False
    p1 = json.loads(r1["content"][0]["text"])
    p2 = json.loads(r2["content"][0]["text"])
    assert p1["status"] == "ok" and p2["status"] == "ok"


def test_track_j_f8_practitioner_assessment_q3_causal_guardrails_and_elasticity_rec() -> None:
    """F8: Question 3 reports monotonic_constraints, holiday_window, and f.promo_elasticity_df()."""
    hist_df, fut_df = _make_collinear_price_promo_panel(n_days=70, horizon=7, seed=404)
    f = Forecaster.auto(
        hist_df,
        future_df=fut_df,
        holidays=["US"],
        tier="first_light",
        where="local",
        horizon=7,
        autofocus=False,
    )
    assess = f.practitioner_assessment_df()
    q3 = assess[assess["question_id"] == "covariates_vs_overfitting"].iloc[0]
    assert "monotonic_constraints=" in str(q3["evidence"])
    assert "holiday_window=" in str(q3["evidence"])
    assert "f.promo_elasticity_df()" in str(q3["recommendation"])
    assert "f.attributions_df()" in str(q3["recommendation"])


def test_track_k_f9_practitioner_assessment_q5_reconciliation_modes_and_simplex() -> None:
    """F9: Question 5 distinguishes spatial_only, temporal_only, cross_temporal, and simplex."""
    hist_df, fut_df = _make_collinear_price_promo_panel(n_days=56, horizon=7, seed=505)

    # 1. Mode 2 (temporal_only on a flat panel)
    flat_df = hist_df.drop(columns=["region"]).copy()
    flat_fut = fut_df.drop(columns=["region"]).copy()
    f_temp = Forecaster.auto(
        flat_df,
        future_df=flat_fut,
        tier="first_light",
        where="local",
        horizon=7,
        autofocus=False,
        overrides={
            "hierarchy": {
                "enabled": False,
                "reconcile_temporal": True,
                "temporal_aggregations": [1, 7],
                "reconciliation_methods": ["wls_struct"],
            }
        },
    )
    q5_temp = (
        f_temp.practitioner_assessment_df()
        .query("question_id == 'hierarchy_reconciliation'")
        .iloc[0]
    )
    assert q5_temp["verdict"] == "temporal_coherent_and_verified"
    assert "mode='temporal_only'" in str(q5_temp["evidence"])
    assert "temporal_aggregations=[1, 7]" in str(q5_temp["evidence"])

    # 2. Mode 3 (cross_temporal on a hierarchical panel)
    f_ct = Forecaster.auto(
        hist_df,
        future_df=fut_df,
        tier="first_light",
        where="local",
        horizon=7,
        autofocus=False,
        overrides={
            "hierarchy": {
                "enabled": True,
                "levels": [["region"]],
                "reconcile_temporal": True,
                "cross_temporal": True,
                "temporal_aggregations": [1, 7],
                "reconciliation_methods": ["wls_struct", "cross_temporal_mint"],
            }
        },
    )
    q5_ct = (
        f_ct.practitioner_assessment_df().query("question_id == 'hierarchy_reconciliation'").iloc[0]
    )
    assert q5_ct["verdict"] == "coherent_and_verified"
    assert "mode='cross_temporal'" in str(q5_ct["evidence"])
    assert "spatial_weight_alpha=" in str(q5_ct["evidence"])
    assert "cross_temporal_beats_spatial=" in str(q5_ct["evidence"])

    # 3. Compositional simplex closure on a flat panel
    dates = pd.date_range("2025-01-01", periods=35, freq="D")
    s_rows = []
    for tid, share in (("seg_a", 0.55), ("seg_b", 0.30), ("seg_c", 0.15)):
        for idx, d in enumerate(dates):
            wobble = (
                0.02 * np.sin(idx * 2.0 * np.pi / 7.0)
                if tid == "seg_a"
                else (-0.02 * np.sin(idx * 2.0 * np.pi / 7.0) if tid == "seg_b" else 0.0)
            )
            s_rows.append({"ts_id": tid, "ds": d, "y": share + wobble})
    simplex_df = pd.DataFrame(s_rows)
    f_sim = Forecaster.auto(
        simplex_df,
        tier="first_light",
        where="local",
        horizon=7,
        autofocus=False,
    )
    q5_sim = (
        f_sim.practitioner_assessment_df()
        .query("question_id == 'hierarchy_reconciliation'")
        .iloc[0]
    )
    assert q5_sim["verdict"] == "coherent_and_verified"
    assert "is_compositional_simplex=True" in str(q5_sim["evidence"])
    assert "strictly_positive=True" in str(q5_sim["evidence"])
