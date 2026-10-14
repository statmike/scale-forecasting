"""Cross-version BigQuery schema, view, borrow, review, and diagnostic resilience tripwires.

Verifies that runs recorded under prior schema/package versions (``v1.0.0``, ``v1.1.0``, and
``v1.2.0``) remain 100% readable, reviewable, borrowable, and diagnosable by the current package:

1. ``Forecaster.from_run_id`` & ``retry_run.config_for_run``:
   - Loads stored ``raw_config`` JSON payloads written by ``v1.0.0`` (minimal config without newer
     blocks like ``reconciliation``, ``explainability``, ``inventory_policy``, or ``fva``),
     ``v1.1.0`` (Explainable Forecasting & Hierarchical Reconciliation), and ``v1.2.0``
     (Forecaster Workbench & Inventory Policy) while preserving the exact ``make_run_id`` digest.
2. ``registry.ops.match_reusable_models`` & ``render_borrow_model_sql``:
   - Matches reusable models across prior ``v1.0.0`` / ``v1.1.0`` / ``v1.2.0`` ``run_registry``
     header rows and renders valid ``INSERT INTO ... SELECT ...`` SQL for ``forecast_predictions``,
     ``backtest_oof``, and ``forecast_metadata`` (including ``COALESCE(fit_diagnostics, JSON '{}')``
     when borrowing legacy rows whose ``fit_diagnostics`` or ``explanations`` were ``NULL``).
3. ``review._assemble_review`` & DataFrame builders:
   - Assembles ``RunReview`` and tidy frames (`build_leaderboard_frame`, `build_cohorts_frame`,
     `build_predictions_frame`, `build_attributions_frame`, `explain_forecast_frame`) over legacy
     rows where optional post-``v1.0.0`` columns (``explanations``, ``fit_diagnostics``,
     ``pooled_wape``, ``staleness_gap``) are ``None`` or absent.
4. ``diagnostics.py`` (`compute_fva`, `compute_inventory_policy`, `compute_anomalies`,
   `compute_change_points`):
   - Computes Forecast Value Added (FVA), safety-stock / reorder-point inventory policy, residual
     anomalies, and structural change-points over ``v1.0.0`` / ``v1.1.0`` prediction & OOF frames
     that carry only point forecasts + ``yhat_lower`` / ``yhat_upper`` (no ``quantiles`` JSON).
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest

from scale_forecasting import review
from scale_forecasting.config import RunConfig
from scale_forecasting.diagnostics import (
    compute_anomalies,
    compute_change_points,
    compute_fva,
    compute_inventory_policy,
)
from scale_forecasting.registry.ids import make_model_spec_hash, make_run_id
from scale_forecasting.registry.ops import match_reusable_models, render_borrow_model_sql
from scale_forecasting.retry_run import config_for_run
from scale_forecasting.sdk import Forecaster
from scale_forecasting.settings import Settings

_SETTINGS = Settings(
    project_id="proj-x",
    connection="proj-x.us-central1.conn",
    warehouse_uri="gs://bkt/warehouse",
)


def _synthetic_version_configs() -> dict[str, dict[str, Any]]:
    """Return representative raw_config payloads as persisted by v1.0.0, v1.1.0, and v1.2.0."""
    v1_0_0 = {
        "run_name": "legacy-v100-run",
        "data": {
            "source_table": "iowa_liquor_weekly",
            "ts_id_col": "store_id",
            "date_col": "week_start",
            "target_col": "total_bottles",
            "freq": "W",
            "horizon": 8,
            "series_limit": 10,
        },
        "models": ["naive_mean", "holtwinters", "lightgbm"],
        "backtest": {
            "enabled": True,
            "n_folds": 2,
            "horizon": 8,
            "step": 4,
            "decision_metric": "wape",
        },
    }
    v1_1_0 = {
        **v1_0_0,
        "run_name": "prior-v110-run",
        "hierarchy": {
            "enabled": True,
            "levels": [["store_id"]],
            "reconciliation_methods": ["bottom_up", "mint_shrink"],
        },
    }
    v1_2_0 = {
        **v1_1_0,
        "run_name": "current-v120-run",
        "models": ["naive_mean", "naive_seasonal", "holtwinters", "lightgbm", "chronos2"],
        "output": {
            "quantiles": [0.1, 0.5, 0.9, 0.95],
            "underage_cost": 9.0,
            "overage_cost": 1.0,
        },
    }
    return {"v1.0.0": v1_0_0, "v1.1.0": v1_1_0, "v1.2.0": v1_2_0}


def test_from_run_id_and_config_for_run_across_v100_v110_v120(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`Forecaster.from_run_id` and `config_for_run` round-trip v1.0.0..v1.2.0 configs."""
    payloads = _synthetic_version_configs()
    for ver, raw in payloads.items():
        cfg = RunConfig.model_validate(raw)
        rid = make_run_id(cfg)
        monkeypatch.setattr(
            "scale_forecasting.registry.reads.read_run_config",
            lambda requested_id, settings=None, _raw=raw, _rid=rid: (
                _raw if requested_id == _rid else None
            ),
        )
        recovered_cfg = config_for_run(rid, settings=_SETTINGS)
        assert make_run_id(recovered_cfg) == rid, f"run_id drift for {ver}"
        fc = Forecaster.from_run_id(rid, settings=_SETTINGS)
        assert fc.run_id == rid
        assert fc.config.models == cfg.models


def test_cross_version_model_borrowing_matches_v100_and_v110_headers() -> None:
    """A v1.2.x run with `reuse_prior_runs=True` borrows models from a v1.0.0/v1.1.0 header."""
    payloads = _synthetic_version_configs()
    v100_cfg = RunConfig.model_validate(payloads["v1.0.0"])
    v100_run_id = make_run_id(v100_cfg)

    # New v1.2.x config adds extra models, sharing data/horizon/backtest
    # for `naive_mean` and `holtwinters`.
    new_raw = {
        **payloads["v1.0.0"],
        "run_name": "v121-borrow-from-v100",
        "models": ["naive_mean", "holtwinters", "theta"],
        "compute": {"reuse_prior_runs": True},
    }
    new_cfg = RunConfig.model_validate(new_raw)
    new_run_id = make_run_id(new_cfg)

    prior_headers = [
        {
            "run_id": v100_run_id,
            "status": "COMPLETED",
            "snapshot_millis": 1700000000000,
            "raw_config": json.dumps(payloads["v1.0.0"]),
        }
    ]
    available_by_run = {v100_run_id: {"naive_mean", "holtwinters", "lightgbm"}}

    borrowed = match_reusable_models(
        new_cfg,
        new_run_id,
        1700000000000,
        prior_headers,
        available_by_run,
    )
    assert set(borrowed.keys()) == {"naive_mean", "holtwinters"}
    for m in ("naive_mean", "holtwinters"):
        assert borrowed[m].source_run_id == v100_run_id
        assert borrowed[m].source_model_type == m
        assert borrowed[m].model_spec_hash == make_model_spec_hash(m, v100_cfg)

    # Verify the rendered borrow SQL safely handles NULL fit_diagnostics & explanations from v1.0.0
    stmts = render_borrow_model_sql("proj.ds")
    assert set(stmts.keys()) == {"forecast_predictions", "backtest_oof", "forecast_metadata"}
    assert "COALESCE(fit_diagnostics, JSON '{}')" in stmts["forecast_metadata"]
    assert "explanations" in stmts["forecast_predictions"]


def test_review_and_frames_degrade_cleanly_on_v100_rows_missing_newer_fields() -> None:
    """`review._assemble_review` and frame builders handle v1.0.0 rows missing newer fields."""
    summary = {"run_id": "legacy-v100", "status": "COMPLETED", "n_series": 2}
    leaderboard_rows = [
        {
            "model_type": "naive_mean",
            "is_ensemble": False,
            "n_series": 2,
            "mean_wape": 0.18,
            "mean_mae": 12.0,
            "mean_rmse": 15.0,
            "mean_mase": 1.0,
            "mean_smape": 0.19,
            "mean_mape": 0.20,
        },
        {
            "model_type": "holtwinters",
            "is_ensemble": False,
            "n_series": 2,
            "mean_wape": 0.12,
            "mean_mae": 8.0,
            "mean_rmse": 10.0,
            "mean_mase": 0.72,
            "mean_smape": 0.13,
            "mean_mape": 0.14,
        },
    ]
    rev = review._assemble_review(
        "legacy-v100",
        summary,
        "wape",
        2,
        leaderboard_rows,
        aggregate_rows=[],
        prediction_counts={"naive_mean": 16, "holtwinters": 16},
        coverage_rows=[],
        comparable_rows=[],
    )
    assert rev.best_overall is not None
    assert rev.best_overall.model_type == "holtwinters"

    lb_df = review.build_leaderboard_frame(rev)
    assert list(lb_df["model_type"]) == ["holtwinters", "naive_mean"]

    cohorts_df = review.build_cohorts_frame(rev)
    assert len(cohorts_df) == 2

    # v1.0.0 prediction rows have no `explanations` or `quantiles`
    v100_preds = [
        {
            "ts_id": "s1",
            "model_type": "holtwinters",
            "forecast_date": "2026-01-04",
            "yhat": 100.0,
            "yhat_lower": 85.0,
            "yhat_upper": 115.0,
            "quantiles": None,
            "explanations": None,
        },
        {
            "ts_id": "s1",
            "model_type": "holtwinters",
            "forecast_date": "2026-01-11",
            "yhat": 105.0,
            "yhat_lower": 88.0,
            "yhat_upper": 122.0,
            "quantiles": None,
            "explanations": None,
        },
    ]
    pred_df = review.build_predictions_frame(v100_preds)
    assert len(pred_df) == 2
    exp_df = review.explain_forecast_frame(pred_df)
    assert len(exp_df) == 2
    attr_df = review.build_attributions_frame([{"ts_id": "s1", "model_type": "holtwinters"}])
    assert attr_df.empty


def test_diagnostics_and_inventory_policy_over_v100_and_v110_outputs() -> None:
    """`compute_fva`, `compute_inventory_policy`, `compute_anomalies`, and `compute_change_points`
    work seamlessly on v1.0.0/v1.1.0 outputs that lack explicit `quantiles` JSON."""
    oof_df = pd.DataFrame(
        [
            {
                "ts_id": "s1",
                "model_type": "naive_mean",
                "fold_id": 0,
                "forecast_date": "2025-12-07",
                "y_true": 100.0,
                "yhat": 120.0,
                "yhat_lower": 90.0,
                "yhat_upper": 150.0,
            },
            {
                "ts_id": "s1",
                "model_type": "naive_mean",
                "fold_id": 0,
                "forecast_date": "2025-12-14",
                "y_true": 100.0,
                "yhat": 80.0,
                "yhat_lower": 50.0,
                "yhat_upper": 110.0,
            },
            {
                "ts_id": "s1",
                "model_type": "holtwinters",
                "fold_id": 0,
                "forecast_date": "2025-12-07",
                "y_true": 100.0,
                "yhat": 104.0,
                "yhat_lower": 95.0,
                "yhat_upper": 113.0,
            },
            {
                "ts_id": "s1",
                "model_type": "holtwinters",
                "fold_id": 0,
                "forecast_date": "2025-12-14",
                "y_true": 100.0,
                "yhat": 96.0,
                "yhat_lower": 87.0,
                "yhat_upper": 105.0,
            },
        ]
    )
    fva_df = compute_fva(oof_df, metric="wape")
    assert not fva_df.empty
    hw_row = fva_df[fva_df["model_type"] == "holtwinters"].iloc[0]
    assert float(hw_row["fva_pct"]) > 0.0

    # v1.0.0 predictions with yhat_lower / yhat_upper and no quantiles column
    preds_v100 = pd.DataFrame(
        [
            {
                "ts_id": "s1",
                "model_type": "holtwinters",
                "forecast_date": f"2026-01-{7 * i + 1:02d}",
                "yhat": 100.0 + i * 2.0,
                "yhat_lower": 80.0 + i * 2.0,
                "yhat_upper": 120.0 + i * 2.0,
            }
            for i in range(4)
        ]
    )
    inv_df = compute_inventory_policy(
        preds_v100,
        lead_time=2,
        service_level=0.90,
        holding_cost=1.0,
        stockout_cost=9.0,
    )
    assert len(inv_df) == 1
    assert float(inv_df.iloc[0]["reorder_point"]) > float(inv_df.iloc[0]["lead_time_demand"])
    assert float(inv_df.iloc[0]["safety_stock"]) > 0.0

    anom_df = compute_anomalies(oof_df)
    assert "is_anomaly" in anom_df.columns

    history_df = pd.DataFrame(
        {
            "ts_id": ["s1"] * 24,
            "ds": pd.date_range("2025-01-05", periods=24, freq="W"),
            "y": [50.0 + (i % 3) for i in range(12)] + [150.0 + (i % 3) for i in range(12)],
        }
    )
    cp_df = compute_change_points(history_df, freq="W")
    assert len(cp_df) >= 1
