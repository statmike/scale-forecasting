"""Unit and contract tests for Phase G.2 Forecaster Workbench features.

Covers:
1. Monotone multi-percentile fans (`output.quantiles`) and Chernozhukov rearrangement.
2. Newsvendor asymmetric cost (`underage_cost`, `overage_cost`) & `target_quantile` decision rules.
3. Syntetos-Boylan-Croston (SBC) demand regime classification (`compute_demand_profile`) and
   opt-in `features.regime_routing`.
4. Pre-fit series grid regularization (`data.grid_policy`, `data.trim_leading_zeros`).
5. Trading-day calendar features (`features.trading_days`), censored demand recovery & validation
   masking (`features.censored_col`), and fold-safe outlier treatment (`features.outlier_policy`).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from scale_forecasting.calibration import rearrange_monotone_quantiles
from scale_forecasting.config import RunConfig
from scale_forecasting.ensembler import (
    calibrate_ensemble_predictions,
    combine_calculated,
    combine_oof,
)
from scale_forecasting.features import (
    build_features,
    build_future_features,
    compute_demand_profile,
    regularize_series_grid,
    select_models_for_series,
)
from scale_forecasting.registry.ids import make_run_id
from scale_forecasting.worker import run_cell


def _make_daily_series(
    n: int = 80,
    ts_id: str = "sku_1",
    *,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2025-01-01", periods=n, freq="D")
    t = np.arange(n, dtype=float)
    y = 50.0 + 0.2 * t + 8.0 * np.sin(2.0 * np.pi * t / 7.0) + rng.normal(0.0, 2.0, size=n)
    return pd.DataFrame({"ts_id": ts_id, "ds": ds, "y": np.maximum(y, 1.0)})


def test_monotone_multi_percentile_fans_and_rearrangement() -> None:
    # 1. Chernozhukov rearrangement repairs inverted quantiles.
    crossed = {0.10: 15.0, 0.25: 12.0, 0.50: 20.0, 0.75: 18.0, 0.90: 25.0}
    repaired = rearrange_monotone_quantiles(crossed)
    qs = sorted(repaired)
    vals = [repaired[q] for q in qs]
    assert vals == sorted(vals)

    # 2. Custom quantile fan flows through run_cell and ensemble calibration.
    df = _make_daily_series(70)
    custom_qs = [0.05, 0.25, 0.50, 0.75, 0.95]
    cfg = RunConfig.model_validate(
        {
            "run_name": "test-fan",
            "data": {"source_table": "p.d.t", "freq": "D", "horizon": 7},
            "models": ["naive_seasonal", "theta"],
            "backtest": {"enabled": True, "n_folds": 2, "horizon": 7, "step": 7, "min_train": 28},
            "ensemble": {"enabled": True, "strategies": ["mean"]},
            "output": {"quantiles": custom_qs},
        }
    )
    cells = [run_cell(df, m, cfg) for m in cfg.models]
    for cell in cells:
        assert cell.status == "ok"
        preds = cell.predictions
        for _, row in preds.iterrows():
            qmap = json.loads(row["quantiles"])
            assert [round(float(k), 2) for k in sorted(qmap, key=float)] == custom_qs
            qvals = [float(qmap[k]) for k in sorted(qmap, key=float)]
            assert qvals == sorted(qvals)
            assert np.isclose(row["yhat_lower"], qvals[0])
            assert np.isclose(row["yhat_upper"], qvals[-1])

    # Verify ensemble calibration also emits the custom monotone quantile fan.
    base_preds = pd.concat(
        [
            c.predictions.rename(columns={"ds": "forecast_date"}).assign(
                ts_id=c.ts_id, model_type=c.model_type
            )
            for c in cells
        ],
        ignore_index=True,
    )
    base_oof = pd.concat(
        [
            c.oof.rename(columns={"ds": "forecast_date"}).assign(
                ts_id=c.ts_id, model_type=c.model_type
            )
            for c in cells
            if c.oof is not None
        ],
        ignore_index=True,
    )
    ens_preds = combine_calculated(base_preds, cfg)
    ens_oof = combine_oof(base_oof, cfg)
    ens_cal = calibrate_ensemble_predictions(ens_preds, ens_oof, cfg)
    for row in ens_cal:
        qmap = json.loads(str(row["quantiles"]))
        assert [round(float(k), 2) for k in sorted(qmap, key=float)] == custom_qs
        qvals = [float(qmap[k]) for k in sorted(qmap, key=float)]
        assert qvals == sorted(qvals)


def test_newsvendor_asymmetric_cost_and_target_quantile() -> None:
    df = _make_daily_series(70)
    base_cfg = RunConfig.model_validate(
        {
            "run_name": "test-base",
            "data": {"source_table": "p.d.t", "freq": "D", "horizon": 7},
            "models": ["naive_seasonal"],
            "backtest": {"enabled": True, "n_folds": 2, "horizon": 7, "step": 7, "min_train": 28},
        }
    )
    nv_cfg = RunConfig.model_validate(
        {
            "run_name": "test-nv",
            "data": {"source_table": "p.d.t", "freq": "D", "horizon": 7},
            "models": ["naive_seasonal"],
            "backtest": {"enabled": True, "n_folds": 2, "horizon": 7, "step": 7, "min_train": 28},
            "output": {"underage_cost": 80.0, "overage_cost": 20.0},
        }
    )
    assert np.isclose(nv_cfg.output.effective_target_quantile or 0.0, 0.80)

    base_cell = run_cell(df, "naive_seasonal", base_cfg)
    nv_cell = run_cell(df, "naive_seasonal", nv_cfg)
    assert nv_cell.point_forecast_source == "quantile:0.80"
    assert nv_cell.point_forecast_decision == "configured"
    assert (
        nv_cell.predictions["yhat"].to_numpy() >= base_cell.predictions["yhat"].to_numpy()
    ).all()

    tq_cfg = RunConfig.model_validate(
        {
            "run_name": "test-tq",
            "data": {"source_table": "p.d.t", "freq": "D", "horizon": 7},
            "models": ["naive_seasonal"],
            "backtest": {"enabled": False},
            "output": {"target_quantile": 0.90},
        }
    )
    tq_cell = run_cell(df, "naive_seasonal", tq_cfg)
    assert tq_cell.point_forecast_source == "quantile:0.90"
    assert np.allclose(
        tq_cell.predictions["yhat"].to_numpy(),
        tq_cell.predictions["yhat_upper"].to_numpy(),
    )

    with pytest.raises(ValidationError):
        RunConfig.model_validate(
            {
                "run_name": "test-bad",
                "data": {"source_table": "p.d.t", "freq": "D", "horizon": 7},
                "models": ["naive_seasonal"],
                "output": {
                    "target_quantile": 0.80,
                    "underage_cost": 80.0,
                    "overage_cost": 20.0,
                },
            }
        )


def test_sbc_demand_profile_and_regime_routing() -> None:
    # 1. SBC classification across all 5 regimes.
    smooth_y = np.full(40, 25.0) + np.sin(np.arange(40))
    assert compute_demand_profile(smooth_y)["demand_class"] == "smooth"

    erratic_y = np.where(np.arange(40) % 2 == 0, 5.0, 80.0)
    assert compute_demand_profile(erratic_y)["demand_class"] == "erratic"

    intermittent_y = np.zeros(40)
    intermittent_y[::3] = 10.0
    assert compute_demand_profile(intermittent_y)["demand_class"] == "intermittent"

    lumpy_y = np.zeros(40)
    lumpy_y[0::4] = 5.0
    lumpy_y[2::4] = 90.0
    assert compute_demand_profile(lumpy_y)["demand_class"] == "lumpy"

    cold_y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert compute_demand_profile(cold_y)["demand_class"] == "cold_start"

    # 2. Default run_cell records demand_profile on CellResult while keeping diagnostics == {}.
    df_inter = pd.DataFrame(
        {
            "ts_id": "sku_inter",
            "ds": pd.date_range("2025-01-01", periods=40, freq="D"),
            "y": intermittent_y,
        }
    )
    cfg_default = RunConfig.model_validate(
        {
            "run_name": "test-sbc",
            "data": {"source_table": "p.d.t", "freq": "D", "horizon": 5},
            "models": ["theta", "croston"],
            "backtest": {"enabled": False},
        }
    )
    cell_def = run_cell(df_inter, "theta", cfg_default)
    assert cell_def.demand_profile is not None
    assert cell_def.demand_profile["demand_class"] == "intermittent"
    assert cell_def.diagnostics == {}

    # 3. With regime_routing=True, theta routes to croston on intermittent series.
    cfg_routed = cfg_default.model_copy(
        update={"features": cfg_default.features.model_copy(update={"regime_routing": True})}
    )
    selected = select_models_for_series(df_inter, cfg_routed)
    assert selected == ["croston"]
    cell_routed = run_cell(df_inter, "theta", cfg_routed)
    assert cell_routed.status == "ok"
    assert cell_routed.diagnostics["regime_routing"] == {
        "demand_class": "intermittent",
        "routed_to": "croston",
    }


def test_grid_regularization_and_trim_leading_zeros() -> None:
    ds_all = pd.date_range("2025-01-01", periods=30, freq="D")
    y_vals = np.concatenate([np.zeros(5), np.arange(1.0, 26.0)])
    full_df = pd.DataFrame({"ts_id": "s1", "ds": ds_all, "y": y_vals})
    # Drop 2 interior dates to create a gap.
    gapped_df = full_df.drop(index=[10, 11]).reset_index(drop=True)

    for policy in ("fill_zero", "zero_fill", "ffill", "linear"):
        cfg = RunConfig.model_validate(
            {
                "run_name": f"test-grid-{policy}",
                "data": {
                    "source_table": "p.d.t",
                    "freq": "D",
                    "horizon": 4,
                    "grid_policy": policy,
                    "trim_leading_zeros": True,
                },
                "models": ["naive_mean"],
                "backtest": {"enabled": False},
            }
        )
        reg = regularize_series_grid(gapped_df, cfg)
        # 30 days minus 5 leading zeros = 25 contiguous days.
        assert len(reg) == 25
        assert float(reg["y"].iloc[0]) == 1.0
        cell = run_cell(gapped_df, "naive_mean", cfg)
        assert cell.status == "ok"


def test_trading_days_censored_demand_and_outlier_hygiene() -> None:
    df = _make_daily_series(60)
    df["stockout"] = 0
    # Inject a training stockout and a validation stockout.
    df.loc[15, "y"] = 0.0
    df.loc[15, "stockout"] = 1
    df.loc[55, "y"] = 0.0
    df.loc[55, "stockout"] = 1
    # Inject an extreme outlier spike in training.
    df.loc[20, "y"] = 5000.0

    for policy in ("winsorize", "interpolate", "flag", "stl_resid"):
        cfg = RunConfig.model_validate(
            {
                "run_name": f"test-hygiene-{policy}",
                "data": {"source_table": "p.d.t", "freq": "D", "horizon": 7},
                "models": ["regression_lags"],
                "features": {
                    "trading_days": True,
                    "censored_col": "stockout",
                    "outlier_policy": policy,
                    "outlier_sigma": 3.5,
                },
                "backtest": {
                    "enabled": True,
                    "n_folds": 1,
                    "horizon": 7,
                    "step": 7,
                    "min_train": 35,
                },
            }
        )
        y_clean, X = build_features(df, cfg)
        assert X is not None
        assert {"days_in_period", "n_weekdays", "n_weekend_days"}.issubset(X.columns)
        # Censored training observation was recovered above 0.
        assert float(y_clean.iloc[15]) > 10.0
        if policy != "flag":
            assert float(y_clean.iloc[20]) < 500.0
        else:
            assert "is_outlier" in X.columns
            assert float(X["is_outlier"].iloc[20]) == 1.0

        X_fut = build_future_features(y_clean, X, cfg)
        assert X_fut is not None
        assert {"days_in_period", "n_weekdays", "n_weekend_days"}.issubset(X_fut.columns)

        cell = run_cell(df, "regression_lags", cfg)
        assert cell.status == "ok"
        assert cell.oof is not None
        # Validation row at index 55 (stockout=1) has y_true masked to NaN in OOF.
        assert cell.oof["y_true"].isna().sum() == 1


def test_phase_g2_default_run_id_stability() -> None:
    cfg = RunConfig.model_validate(
        {
            "run_name": "test-id",
            "data": {"source_table": "p.d.t", "freq": "D", "horizon": 7},
            "models": ["naive_mean"],
        }
    )
    cfg_explicit = RunConfig.model_validate(
        {
            "run_name": "test-id",
            "data": {
                "source_table": "p.d.t",
                "freq": "D",
                "horizon": 7,
                "grid_policy": "strict",
                "trim_leading_zeros": False,
            },
            "models": ["naive_mean"],
            "features": {
                "trading_days": False,
                "censored_col": None,
                "outlier_policy": "none",
                "outlier_sigma": 3.5,
                "regime_routing": False,
            },
            "output": {
                "quantiles": None,
                "underage_cost": None,
                "overage_cost": None,
                "target_quantile": None,
            },
        }
    )
    assert make_run_id(cfg) == make_run_id(cfg_explicit)
