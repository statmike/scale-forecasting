"""Unit tests for Online Adaptive Conformal Inference (`aci.py`)."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from scale_forecasting.aci import (
    aci_df,
    aci_trajectory_df,
    calibrate_aci_from_oof,
    compute_aci_df,
    plot_aci,
)
from scale_forecasting.calibration import apply_calibration, calibrate_from_oof
from scale_forecasting.config import OutputConfig, RunConfig
from scale_forecasting.features import resolve_model_variant


def _make_regime_shift_oof(n_calm: int = 80, n_shock: int = 120, seed: int = 42) -> pd.DataFrame:
    """Create an OOF residual stream with a 4x volatility regime shift after n_calm steps."""
    rng = np.random.default_rng(seed)
    n_total = n_calm + n_shock
    dates = pd.date_range("2025-01-01", periods=n_total, freq="D")
    yhat = 100.0 + 10.0 * np.sin(2.0 * np.pi * np.arange(n_total) / 7.0)
    sigma = np.concatenate([np.full(n_calm, 2.0), np.full(n_shock, 9.0)])
    errors = rng.normal(0.0, 1.0, size=n_total) * sigma
    y_true = yhat + errors
    # Static 80% interval built from the calm regime (sigma=2.0 -> half-width ~ 2.56).
    calm_hw = 2.0 * 1.28155
    return pd.DataFrame(
        {
            "ts_id": "sku_shock",
            "model_type": "ridge",
            "ds": dates,
            "horizon_step": 1,
            "y_true": y_true,
            "yhat": yhat,
            "yhat_lower": yhat - calm_hw,
            "yhat_upper": yhat + calm_hw,
        }
    )


def test_aci_recovers_target_coverage_under_volatility_regime_shift() -> None:
    """Online ACI widens intervals after a 4x volatility shock and restores empirical coverage."""
    oof = _make_regime_shift_oof()
    res = compute_aci_df(
        oof,
        gamma=0.05,
        target_coverage=0.80,
        normalized=True,
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
    assert required_cols.issubset(res.columns)
    assert len(res) == len(oof)

    # During the post-shock regime (rows 80..200), static intervals suffer severe under-coverage
    # while ACI adapts alpha_t downward and volatility_scale upward to recover target coverage.
    shock_slice = res.iloc[80:]
    static_shock_cov = float(shock_slice["static_covered"].mean())
    aci_overall_cov = float(res["aci_covered"].mean())
    assert static_shock_cov < 0.50
    assert aci_overall_cov > static_shock_cov + 0.25
    assert abs(aci_overall_cov - 0.80) <= 0.08
    assert float(shock_slice["aci_width"].mean()) > float(shock_slice["static_width"].mean()) * 1.5
    assert bool(shock_slice["drift_detected"].any()) is True

    # Aliases point to the same implementation.
    assert aci_df is compute_aci_df
    assert aci_trajectory_df is compute_aci_df

    ax = plot_aci(res)
    assert ax is not None


def test_aci_enforces_h_step_horizon_feedback_delay_without_lookahead() -> None:
    """At horizon_step=h, alpha_t cannot change in response to an error until h steps later."""
    dates = pd.date_range("2025-01-01", periods=20, freq="D")
    yhat = np.full(20, 50.0)
    # Errors are tiny (0.1) everywhere except at index t=10 where a massive spike (+100) occurs.
    y_true = yhat.copy()
    y_true[:10] += 0.1
    y_true[10] += 100.0
    y_true[11:] += 0.1

    df_h3 = pd.DataFrame(
        {
            "ts_id": "s_delay",
            "model_type": "theta",
            "ds": dates,
            "horizon_step": 3,
            "y_true": y_true,
            "yhat": yhat,
            "yhat_lower": yhat - 5.0,
            "yhat_upper": yhat + 5.0,
        }
    )
    res_delayed = compute_aci_df(
        df_h3,
        gamma=0.08,
        target_coverage=0.80,
        normalized=False,
        horizon_delay=True,
    )
    alphas = res_delayed["alpha_t"].to_numpy(dtype=float)
    # With h=3 feedback delay, the miss at t=10 is not revealed until step t + h = 13:
    # Steps 11 and 12 only observe calm hits from t-2 <= 9, so alpha_t keeps increasing on 11 and 12
    # and only drops at step 13 when the t=10 miss becomes mature!
    assert alphas[11] >= alphas[10]
    assert alphas[12] >= alphas[11]
    assert alphas[13] < alphas[12]


def test_aci_config_and_calibration_dispatch() -> None:
    """OutputConfig calibration_method ('conformal', 'aci', 'aci_normalized') dispatches cleanly."""
    oof = _make_regime_shift_oof(n_calm=30, n_shock=40)
    for method in ("conformal", "aci", "aci_normalized"):
        out_cfg = OutputConfig(calibration_method=method, aci_gamma=0.05)
        assert out_cfg.calibration_method == method

    cfg = RunConfig.model_validate(
        {
            "run_name": "test_aci",
            "data": {"source_table": "p.d.t", "horizon": 7, "freq": "D"},
            "models": ["regression_lags@aci", "regression_lags@aci_norm"],
            "output": {"calibration_method": "conformal", "aci_gamma": 0.05},
        }
    )
    v_aci = resolve_model_variant(cfg, "regression_lags@aci")
    v_norm = resolve_model_variant(cfg, "regression_lags@aci_norm")
    assert v_aci.effective_cfg.output.calibration_method == "aci"
    assert v_norm.effective_cfg.output.calibration_method == "aci_normalized"

    qs = (0.1, 0.5, 0.9)
    cal_conformal = calibrate_from_oof(oof, qs, residual_column="yhat", method="conformal")
    cal_aci = calibrate_from_oof(oof, qs, residual_column="yhat", method="aci", aci_gamma=0.05)
    cal_aci_norm = calibrate_from_oof(
        oof, qs, residual_column="yhat", method="aci_normalized", aci_gamma=0.05
    )
    direct_aci, state_aci = calibrate_aci_from_oof(
        oof, method="aci_normalized", gamma=0.05, quantiles=qs, residual_column="yhat"
    )

    assert cal_conformal is not None
    assert cal_aci is not None
    assert cal_aci_norm is not None
    assert direct_aci is not None
    assert state_aci.n_steps == len(oof)
    w_conf = cal_conformal.pooled.quantiles[0.9] - cal_conformal.pooled.quantiles[0.1]
    w_aci_norm = cal_aci_norm.pooled.quantiles[0.9] - cal_aci_norm.pooled.quantiles[0.1]
    w_direct = direct_aci.pooled.quantiles[0.9] - direct_aci.pooled.quantiles[0.1]
    assert w_aci_norm > w_conf
    assert np.isclose(w_aci_norm, w_direct)

    pred_df = pd.DataFrame(
        {
            "ds": pd.date_range("2025-03-15", periods=5, freq="D"),
            "yhat": np.full(5, 100.0),
            "yhat_raw": np.full(5, 100.0),
            "yhat_lower": np.full(5, 95.0),
            "yhat_upper": np.full(5, 105.0),
            "quantiles": ["{}"] * 5,
        }
    )
    recal, src = apply_calibration(pred_df, cal_aci_norm, "median")
    assert src.startswith(("aci", "oof-aci"))
    assert (recal["yhat_upper"] - recal["yhat_lower"] > 0.0).all()
