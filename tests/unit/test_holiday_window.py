"""Unit tests for Track J Capability 3: Asymmetric Holiday Windows (`holiday_window.py`)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scale_forecasting.config import RunConfig
from scale_forecasting.features import build_features
from scale_forecasting.holiday_window import (
    HOLIDAY_WINDOW_COLUMNS,
    HolidayWindowConfig,
    build_holiday_window_features,
    coerce_holiday_window_config,
)
from scale_forecasting.registry.ids import make_model_spec_hash, make_run_id
from scale_forecasting.worker import run_cell


def _make_holiday_ramp_series(n_days: int = 365 * 2) -> pd.DataFrame:
    """Create a 2-year daily retail series where major US holidays trigger a 7-day pre-holiday
    stock-up surge (+95 units peaking on the day before/of the holiday) followed by a 3-day
    post-holiday pantry-loading dip (-45 units).
    """
    rng = np.random.default_rng(77)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    hw_cfg = HolidayWindowConfig(
        enabled=True, pre_holiday_days=7, post_holiday_days=3, kernel="linear"
    )
    win_cols = build_holiday_window_features(dates, ["US"], hw_cfg)
    dow = np.asarray(dates.dayofweek, dtype=float)
    seasonal = 15.0 * np.sin(2.0 * np.pi * dow / 7.0)
    lead_ramp = np.asarray(win_cols["pre_holiday_window"], dtype=float)
    lag_dip = np.asarray(win_cols["post_holiday_window"], dtype=float)
    y = 100.0 + seasonal + 95.0 * lead_ramp - 45.0 * lag_dip + rng.normal(0.0, 2.5, size=n_days)
    return pd.DataFrame(
        {
            "ts_id": "store_holiday_1",
            "ds": dates,
            "y": np.maximum(5.0, y),
        }
    )


def test_holiday_window_kernels_and_enabled_toggle() -> None:
    """Exercise `enabled=True`/`False` and `kernel` in `('linear', 'indicator', 'exponential')`."""
    dates = pd.date_range("2025-12-15", "2026-01-06", freq="D")

    for kernel in ("linear", "indicator", "exponential"):
        cfg_on = HolidayWindowConfig(
            enabled=True,
            pre_holiday_days=7,
            post_holiday_days=3,
            kernel=kernel,
        )
        win_cols = build_holiday_window_features(dates, ["US"], cfg_on)
        assert list(win_cols.keys()) == list(HOLIDAY_WINDOW_COLUMNS)
        df_win = pd.DataFrame(win_cols, index=dates)
        assert df_win["pre_holiday_window"].between(0.0, 1.0).all()
        assert df_win["post_holiday_window"].between(0.0, 1.0).all()
        # Days leading up to Dec 25 (e.g. Dec 23) must have strictly positive pre_holiday_window
        assert float(df_win.loc[pd.Timestamp("2025-12-23"), "pre_holiday_window"]) > 0.0
        # Dec 26 (day after Christmas) must have strictly positive post_holiday_window
        assert float(df_win.loc[pd.Timestamp("2025-12-26"), "post_holiday_window"]) > 0.0

    cfg_off = HolidayWindowConfig(
        enabled=False,
        pre_holiday_days=7,
        post_holiday_days=3,
        kernel="linear",
    )
    win_off = build_holiday_window_features(dates, ["US"], cfg_off)
    assert win_off == {}
    assert coerce_holiday_window_config(False) is None
    coerced_true = coerce_holiday_window_config(True)
    assert coerced_true is not None and coerced_true.enabled is True


def test_holiday_window_run_id_invariance_and_build_features() -> None:
    base_raw = {
        "run_name": "holiday-win-check",
        "data": {"source_table": "t", "freq": "D", "horizon": 14},
        "models": ["regression_lags"],
        "features": {"holidays": ["US"]},
    }
    cfg_none = RunConfig.model_validate(base_raw)
    cfg_explicit_none = RunConfig.model_validate(
        {
            **base_raw,
            "features": {"holidays": ["US"], "holiday_window": None},
        }
    )
    assert make_run_id(cfg_none) == make_run_id(cfg_explicit_none)

    cfg_win = RunConfig.model_validate(
        {
            **base_raw,
            "features": {
                "holidays": ["US"],
                "holiday_window": {
                    "enabled": True,
                    "pre_holiday_days": 7,
                    "post_holiday_days": 3,
                    "kernel": "linear",
                },
            },
        }
    )
    assert make_run_id(cfg_none) != make_run_id(cfg_win)
    assert make_model_spec_hash("regression_lags", cfg_none) != make_model_spec_hash(
        "regression_lags", cfg_win
    )

    series = _make_holiday_ramp_series(n_days=120)
    _, x_df = build_features(series, cfg_win)
    assert x_df is not None
    for col in HOLIDAY_WINDOW_COLUMNS:
        assert col in x_df.columns


def test_holiday_window_reduces_backtest_wape_by_at_least_10_pct_over_point_holiday() -> None:
    """Acceptance Criterion 3: On a synthetic retail panel with a 7-day pre-holiday stock-up ramp
    and 3-day post-holiday dip, enabling `features.holiday_window` improves out-of-fold WAPE by
    >= 10% over point-day `is_holiday` alone.
    """
    series = _make_holiday_ramp_series(n_days=365 * 2)
    # Place backtest windows across Q4 holiday season (Thanksgiving + Christmas + New Year)
    common_bt = {"enabled": True, "n_folds": 3, "horizon": 21, "step": 21}

    cfg_point = RunConfig.model_validate(
        {
            "run_name": "holiday-point-only",
            "data": {"source_table": "playground", "freq": "D", "horizon": 21},
            "models": ["regression_lags"],
            "features": {"holidays": ["US"]},
            "backtest": common_bt,
        }
    )
    cfg_window = RunConfig.model_validate(
        {
            "run_name": "holiday-window-ramp",
            "data": {"source_table": "playground", "freq": "D", "horizon": 21},
            "models": ["regression_lags"],
            "features": {
                "holidays": ["US"],
                "holiday_window": {
                    "enabled": True,
                    "pre_holiday_days": 7,
                    "post_holiday_days": 3,
                    "kernel": "linear",
                },
            },
            "backtest": common_bt,
        }
    )

    res_point = run_cell(series, "regression_lags", cfg_point)
    res_window = run_cell(series, "regression_lags", cfg_window)
    assert res_point.status == "ok"
    assert res_window.status == "ok"

    wape_point = float(res_point.metrics["wape"])
    wape_window = float(res_window.metrics["wape"])
    rel_improvement = (wape_point - wape_window) / wape_point
    assert rel_improvement >= 0.10, (
        f"Expected >= 10% WAPE reduction from holiday_window, got {rel_improvement:.2%} "
        f"(point={wape_point:.4f}, window={wape_window:.4f})"
    )


def test_holiday_window_mixed_country_code_and_explicit_date_strings() -> None:
    """Challenger 1 polish: passing a mixed sequence of ISO country codes and explicit date strings
    (e.g. `['US', '2024-03-31']`) resolves both sources and unions their holiday ordinals.
    """
    dates = pd.date_range("2024-03-20", "2024-04-05", freq="D")
    cfg = HolidayWindowConfig(
        enabled=True, pre_holiday_days=5, post_holiday_days=2, kernel="linear"
    )
    win = build_holiday_window_features(dates, ["US", "2024-03-31"], cfg)
    df_win = pd.DataFrame(win, index=dates)
    assert float(df_win.loc[pd.Timestamp("2024-03-30"), "pre_holiday_window"]) == 1.0
    assert float(df_win.loc[pd.Timestamp("2024-04-01"), "post_holiday_window"]) == 1.0
