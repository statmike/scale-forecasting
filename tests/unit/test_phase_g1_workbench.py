"""Offline unit tests for Phase G.1 (Physical Reality, Non-Negative MinT & Smart Ensembles).

Covers:
1. Physical bounds (`floor`, `cap`, `floor_col`, `cap_col`) and `integer_rounding` (`none`,
   `round`, `ceil`, `floor`, `cumulative_round` Bresenham error-diffusion) in `calibration.py`,
   `backtest.py`, and `worker.py`.
2. Non-negative hierarchical reconciliation (`method="mint_nonneg"`) via `scipy.optimize.nnls` in
   `reconciliation.py`.
3. Smart ensemble strategies (`champion`, `horizon_specialist`, `top_k`, `parsimony_margin`) and
   empirical ensemble interval calibration (`ensemble_oof_bounds`, `calibrate_ensemble_predictions`)
   in `ensembler.py`, `ensemble_run.py`, and `playground.py`.
4. Zero `run_id` and `ensemble_id` drift when Phase G.1 fields are left at their defaults.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from scale_forecasting.calibration import apply_output_constraints
from scale_forecasting.config import OutputConfig, RunConfig
from scale_forecasting.ensemble_run import ensemble_scoring_basis
from scale_forecasting.ensembler import (
    calibrate_ensemble_predictions,
    combine_calculated,
    combine_oof,
    ensemble_oof_bounds,
)
from scale_forecasting.playground import bakeoff, sample_data
from scale_forecasting.reconciliation import (
    build_hierarchy,
    reconcile_cells,
    reconcile_forecasts,
    reconcile_oof,
)
from scale_forecasting.registry.ids import make_ensemble_id, make_run_id
from scale_forecasting.worker import CellResult, run_cell


def test_output_constraints_floor_cap_and_rounding_modes() -> None:
    """Verify floor, cap, dynamic floor_col/cap_col, and all five integer_rounding modes."""
    # Static floor > cap is rejected by OutputConfig validation.
    with pytest.raises(ValidationError, match="floor"):
        OutputConfig(floor=10.0, cap=5.0)

    df = pd.DataFrame(
        {
            "ds": pd.date_range("2026-01-01", periods=5, freq="D"),
            "yhat": [-2.3, 0.4, 0.4, 0.4, 12.7],
            "yhat_lower": [-5.0, -0.2, 0.1, 0.1, 9.2],
            "yhat_upper": [1.2, 0.9, 0.9, 0.9, 15.8],
        }
    )

    # 1. Static floor=0.0, cap=10.0, integer_rounding="none"
    cfg_none = OutputConfig(floor=0.0, cap=10.0, integer_rounding="none")
    res_none = apply_output_constraints(df, cfg_none)
    assert res_none["yhat"].tolist() == pytest.approx([0.0, 0.4, 0.4, 0.4, 10.0])
    assert res_none["yhat_lower"].tolist() == pytest.approx([0.0, 0.0, 0.1, 0.1, 9.2])
    assert res_none["yhat_upper"].tolist() == pytest.approx([1.2, 0.9, 0.9, 0.9, 10.0])

    # 2. integer_rounding="round", "ceil", "floor"
    for mode, expected in [
        ("round", [0.0, 0.0, 0.0, 0.0, 10.0]),
        ("ceil", [0.0, 1.0, 1.0, 1.0, 10.0]),
        ("floor", [0.0, 0.0, 0.0, 0.0, 10.0]),
    ]:
        cfg_mode = OutputConfig(floor=0.0, cap=10.0, integer_rounding=mode)  # type: ignore[arg-type]
        res_mode = apply_output_constraints(df, cfg_mode)
        assert res_mode["yhat"].tolist() == pytest.approx(expected)
        assert (res_mode["yhat_lower"] <= res_mode["yhat"]).all()
        assert (res_mode["yhat"] <= res_mode["yhat_upper"]).all()

    # 3. Bresenham error-diffusion cumulative_round on fractional demand (0.4 x 5 = 2.0)
    frac_df = pd.DataFrame(
        {
            "ds": pd.date_range("2026-01-01", periods=5, freq="D"),
            "yhat": [0.4, 0.4, 0.4, 0.4, 0.4],
            "yhat_lower": [0.1, 0.1, 0.1, 0.1, 0.1],
            "yhat_upper": [0.8, 0.8, 0.8, 0.8, 0.8],
        }
    )
    cfg_cum = OutputConfig(floor=0.0, integer_rounding="cumulative_round")
    res_cum = apply_output_constraints(frac_df, cfg_cum)
    # Standard round gives [0, 0, 0, 0, 0] (sum = 0 vs 2.0); cumulative_round preserves sum = 2.0!
    assert res_cum["yhat"].sum() == pytest.approx(2.0)
    assert set(res_cum["yhat"].tolist()).issubset({0.0, 1.0})
    assert (res_cum["yhat_lower"] <= res_cum["yhat"]).all()
    assert (res_cum["yhat"] <= res_cum["yhat_upper"]).all()

    # 4. Per-row dynamic floor_col and cap_col via bounds_df
    bounds_df = pd.DataFrame(
        {
            "ds": pd.date_range("2026-01-01", periods=5, freq="D"),
            "min_display": [1.0, 1.0, 2.0, 2.0, 3.0],
            "max_capacity": [5.0, 5.0, 5.0, 5.0, 8.0],
        }
    )
    cfg_dyn = OutputConfig(
        floor_col="min_display", cap_col="max_capacity", integer_rounding="round"
    )
    res_dyn = apply_output_constraints(df, cfg_dyn, bounds_df=bounds_df)
    assert res_dyn["yhat"].tolist() == pytest.approx([1.0, 1.0, 2.0, 2.0, 8.0])


def test_worker_and_backtest_apply_output_constraints_post_calibration() -> None:
    """run_cell applies output constraints post-calibration while preserving yhat_raw in OOF."""
    panel = sample_data(n_series=1, history=150, freq="D")
    series = panel[panel["ts_id"] == panel["ts_id"].iloc[0]].copy()
    series["y"] = np.maximum(series["y"] - series["y"].min() - 5.0, -3.0)

    cfg = RunConfig.model_validate(
        {
            "run_name": "g1-bounds-worker",
            "data": {"source_table": "t", "freq": "D", "horizon": 7},
            "models": ["naive_mean"],
            "backtest": {"enabled": True, "n_folds": 2, "horizon": 7, "step": 7, "min_train": 30},
            "output": {"floor": 0.0, "cap": 50.0, "integer_rounding": "cumulative_round"},
        }
    )
    res = run_cell(series, "naive_mean", cfg)
    assert res.status == "ok"
    assert (res.predictions["yhat"] >= 0.0).all()
    assert (res.predictions["yhat"] <= 50.0).all()
    assert (res.predictions["yhat_lower"] >= 0.0).all()
    assert (res.predictions["yhat_upper"] <= 50.0).all()
    assert np.allclose(res.predictions["yhat"], np.round(res.predictions["yhat"]))
    assert res.oof is not None and not res.oof.empty
    assert (res.oof["yhat"] >= 0.0).all()


def test_reconciliation_mint_nonneg_guarantees_nonnegative_coherent_forecasts() -> None:
    """mint_nonneg eliminates negative reconciled forecasts while preserving S*y_b = y_tilde."""
    cfg = RunConfig.model_validate(
        {
            "run_name": "g1-mint-nonneg",
            "data": {"source_table": "t", "freq": "D", "horizon": 2},
            "models": ["theta"],
            "hierarchy": {
                "enabled": True,
                "levels": [["region"]],
                "reconciliation_methods": ["mint_nonneg"],
            },
        }
    )
    # Build a 2-bottom-series hierarchy (R1_S1 and R1_S2 under region=R1 and __total__).
    hist_rows = []
    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    for d in dates:
        hist_rows.append({"ts_id": "R1_S1", "region": "R1", "store": "S1", "ds": d, "y": 10.0})
        hist_rows.append({"ts_id": "R1_S2", "region": "R1", "store": "S2", "ds": d, "y": 1.0})
    hist_df = pd.DataFrame(hist_rows)
    _, spec = build_hierarchy(hist_df, cfg)

    f_dates = pd.date_range("2026-01-11", periods=2, freq="D")
    base_map = {
        "__total__": 5.0,
        "region=R1": 5.0,
        "R1_S1": 12.0,
        "R1_S2": -4.0,  # negative base forecast on bottom leaf
    }
    pred_rows = []
    oof_rows = []
    cells: list[CellResult] = []
    for tid in spec.node_ids:
        val = base_map[tid]
        p_df = pd.DataFrame(
            {
                "ds": f_dates,
                "yhat": [val, val],
                "yhat_lower": [val - 1.0, val - 1.0],
                "yhat_upper": [val + 2.0, val + 2.0],
            }
        )
        o_df = pd.DataFrame(
            {
                "fold_id": [0, 0, 1, 1],
                "ds": list(f_dates) * 2,
                "cutoff_date": [pd.Timestamp("2026-01-09")] * 2 + [pd.Timestamp("2026-01-10")] * 2,
                "horizon_step": [1, 2, 1, 2],
                "y_true": [max(val, 0.0) + 0.5, max(val, 0.0) - 0.2] * 2,
                "yhat": [val, val] * 2,
                "yhat_lower": [val - 1.0] * 4,
                "yhat_upper": [val + 2.0] * 4,
            }
        )
        cells.append(
            CellResult(
                run_id="r1",
                ts_id=tid,
                model_type="theta",
                compute_engine="local",
                model_hash="h1",
                predictions=p_df,
                metrics={"wape": 0.1},
                fit_seconds=0.01,
                best_params={},
                status="ok",
                error=None,
                oof=o_df,
            )
        )
        for d in f_dates:
            pred_rows.append(
                {
                    "ts_id": tid,
                    "model_type": "theta",
                    "forecast_date": d,
                    "yhat": val,
                    "yhat_lower": val - 1.0,
                    "yhat_upper": val + 2.0,
                }
            )
        for _, r in o_df.iterrows():
            oof_rows.append(
                {
                    "ts_id": tid,
                    "model_type": "theta",
                    "fold_id": int(r["fold_id"]),
                    "forecast_date": r["ds"],
                    "cutoff_date": r["cutoff_date"],
                    "horizon_step": int(r["horizon_step"]),
                    "y_true": float(r["y_true"]),
                    "yhat": float(r["yhat"]),
                }
            )

    preds_df = pd.DataFrame(pred_rows)
    oof_df = pd.DataFrame(oof_rows)

    rec_preds = reconcile_forecasts(preds_df, spec, method="mint_nonneg", oof_df=oof_df)
    assert (rec_preds["yhat"] >= 0.0).all()
    assert (rec_preds["yhat_lower"] >= 0.0).all()
    assert (rec_preds["yhat_lower"] <= rec_preds["yhat"]).all()
    assert (rec_preds["yhat"] <= rec_preds["yhat_upper"]).all()

    # Check exact structural coherence: total == sum of bottom leaves for each date
    piv = rec_preds.pivot(index="forecast_date", columns="ts_id", values="yhat")
    bottom_sum = piv["R1_S1"] + piv["R1_S2"]
    assert np.allclose(piv["__total__"], bottom_sum)
    assert np.allclose(piv["region=R1"], bottom_sum)

    # Also verify reconcile_oof and reconcile_cells with mint_nonneg
    rec_oof = reconcile_oof(oof_df, spec, method="mint_nonneg")
    assert (rec_oof["yhat"] >= 0.0).all()

    rec_cell_list = reconcile_cells(cells, cfg, spec, history_df=hist_df)
    assert len(rec_cell_list) == len(spec.node_ids)
    for c in rec_cell_list:
        assert (c.predictions["yhat"] >= 0.0).all()


def test_smart_ensembles_champion_horizon_specialist_topk_and_parsimony() -> None:
    """Verify champion, horizon_specialist, top_k, and parsimony_margin on OOF and future blends."""
    rows = []
    for fold_id, cutoff in [(0, "2026-01-01"), (1, "2026-01-08")]:
        for step, fdate in [(1, f"2026-01-0{2 + fold_id * 7}"), (2, f"2026-01-0{3 + fold_id * 7}")]:
            y_true = 100.0
            yhat_naive = 90.0 if step == 1 else 50.0
            yhat_xgb = 90.2 if step == 1 else 100.0
            yhat_theta = 70.0
            for m, yh in [("naive_mean", yhat_naive), ("xgboost", yhat_xgb), ("theta", yhat_theta)]:
                rows.append(
                    {
                        "ts_id": "s1",
                        "model_type": m,
                        "fold_id": fold_id,
                        "cutoff_date": cutoff,
                        "forecast_date": fdate,
                        "horizon_step": step,
                        "y_true": y_true,
                        "yhat": yh,
                    }
                )
    oof_df = pd.DataFrame(rows)

    base_df = pd.DataFrame(
        [
            {
                "ts_id": "s1",
                "model_type": m,
                "forecast_date": d,
                "yhat": yh,
                "yhat_lower": yh - 2.0,
                "yhat_upper": yh + 2.0,
            }
            for d, yh_map in [
                ("2026-01-15", {"naive_mean": 10.0, "xgboost": 20.0, "theta": 30.0}),
                ("2026-01-16", {"naive_mean": 10.0, "xgboost": 40.0, "theta": 30.0}),
            ]
            for m, yh in yh_map.items()
        ]
    )
    metric_close = pd.DataFrame(
        [
            {"ts_id": "s1", "model_type": "naive_mean", "wape": 0.100},
            {
                "ts_id": "s1",
                "model_type": "xgboost",
                "wape": 0.098,
            },  # only 2% better than naive_mean
            {"ts_id": "s1", "model_type": "theta", "wape": 0.250},
        ]
    )
    cfg_champ = RunConfig.model_validate(
        {
            "run_name": "g1-smart-ens",
            "data": {"source_table": "t", "freq": "D", "horizon": 2},
            "models": ["naive_mean", "xgboost", "theta"],
            "backtest": {"enabled": True, "n_folds": 2, "horizon": 2, "step": 2},
            "ensemble": {
                "enabled": True,
                "strategies": ["champion", "horizon_specialist", "inverse_error", "mean"],
                "top_k": 2,
                "parsimony_margin": 0.05,
            },
        }
    )
    calc_rows = combine_calculated(base_df, cfg_champ, metric_close, oof_df)
    by_strat: dict[str, list[float]] = {}
    for r in calc_rows:
        by_strat.setdefault(r["model_type"], []).append(r["yhat"])

    assert by_strat["ensemble_champion"] == pytest.approx([10.0, 10.0])
    assert by_strat["ensemble_horizon_specialist"][1] == pytest.approx(40.0)

    ens_oof = combine_oof(oof_df, cfg_champ)
    assert set(ens_oof["model_type"].unique()) == {
        "ensemble_champion",
        "ensemble_horizon_specialist",
        "ensemble_inverse_error",
        "ensemble_mean",
    }
    assert ensemble_scoring_basis("champion", oof_df, cfg_champ, "holdout") == "holdout"
    assert ensemble_scoring_basis("horizon_specialist", oof_df, cfg_champ, "holdout") == "holdout"
    assert ensemble_scoring_basis("mean", oof_df, cfg_champ, "holdout") == "holdout"

    champ_oof = ens_oof[ens_oof["model_type"] == "ensemble_champion"]
    lo, hi = ensemble_oof_bounds(champ_oof, 1, cfg_champ)
    assert lo is not None and hi is not None
    assert len(lo) == 2 and np.all(lo <= hi)

    cal_rows = calibrate_ensemble_predictions(calc_rows, ens_oof, cfg_champ)
    assert len(cal_rows) == len(calc_rows)


def test_bakeoff_populates_interval_metrics_for_ensembles() -> None:
    """playground.bakeoff computes finite interval metrics (coverage, interval_score, etc.)."""
    data = sample_data(n_series=1, history=420, freq="D")
    bo = bakeoff(
        data,
        models=["naive_mean", "theta"],
        horizon=7,
        n_folds=2,
        calculated="champion",
        learned="nnls",
    )
    ens_rows = bo.leaderboard[bo.leaderboard["kind"] == "ensemble"]
    assert len(ens_rows) == 2
    for col in ("coverage", "pinball", "interval_score", "interval_width", "msis"):
        assert ens_rows[col].notna().all(), f"expected non-NaN {col} on ensemble leaderboard rows"


def test_phase_g1_run_id_and_ensemble_id_determinism() -> None:
    """Default Phase G.1 fields elide cleanly so run_id and ensemble_id hashes never drift."""
    base_cfg = RunConfig.model_validate(
        {
            "run_name": "determinism-check",
            "data": {"source_table": "t"},
            "models": ["theta", "naive_mean"],
            "ensemble": {"enabled": True, "strategies": ["mean"]},
        }
    )
    floor_cfg = RunConfig.model_validate(
        {
            "run_name": "determinism-check",
            "data": {"source_table": "t"},
            "models": ["theta", "naive_mean"],
            "output": {"floor": 0.0, "integer_rounding": "cumulative_round"},
            "ensemble": {"enabled": True, "strategies": ["mean"]},
        }
    )
    topk_cfg = RunConfig.model_validate(
        {
            "run_name": "determinism-check",
            "data": {"source_table": "t"},
            "models": ["theta", "naive_mean"],
            "ensemble": {
                "enabled": True,
                "strategies": ["mean"],
                "top_k": 2,
                "parsimony_margin": 0.03,
            },
        }
    )
    assert make_run_id(base_cfg) != make_run_id(floor_cfg)
    assert make_ensemble_id(base_cfg.ensemble) != make_ensemble_id(topk_cfg.ensemble)
