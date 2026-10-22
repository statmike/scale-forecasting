"""Unit tests for temporal hierarchy aggregation and cross-temporal reconciliation (THieF)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scale_forecasting.config import RECONCILIATION_METHODS, RunConfig
from scale_forecasting.data_gen.generator import GenConfig, generate_panel
from scale_forecasting.engines.spark_io import run_group
from scale_forecasting.errors import ConfigError, DataError
from scale_forecasting.reconciliation import (
    build_hierarchy,
    reconcile_cells,
    reconcile_forecasts,
    reconcile_oof,
    verify_coherence,
)
from scale_forecasting.temporal_reconciliation import (
    TemporalHierarchySpec,
    aggregate_temporal_history,
    aggregate_temporal_series,
    build_temporal_hierarchy,
    build_temporal_hierarchy_spec,
    cross_temporal_mint,
    default_temporal_aggregations,
    reconcile_cross_temporal_forecasts,
    reconcile_series_temporal,
    reconcile_temporal_forecasts,
    reconcile_temporal_matrix,
    verify_cross_temporal_coherence,
    verify_temporal_coherence,
)
from scale_forecasting.worker import run_cell


def _monthly_series(n_months: int = 36, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_months, freq="MS")
    t = np.arange(n_months, dtype=float)
    y = (
        100.0
        + 1.5 * t
        + 15.0 * np.sin(2.0 * np.pi * t / 12.0)
        + rng.normal(0.0, 3.0, size=n_months)
    )
    return pd.DataFrame({"ts_id": "s_0", "ds": dates, "y": y})


def _monthly_hierarchy_panel(n_months: int = 36, seed: int = 42) -> pd.DataFrame:
    return generate_panel(
        4,
        cfg=GenConfig(
            history=n_months,
            freq="MS",
            with_hierarchy=True,
        ),
        seed=seed,
    )


def test_default_temporal_aggregations_and_spec_construction() -> None:
    assert "cross_temporal_mint" in RECONCILIATION_METHODS
    assert default_temporal_aggregations(12) == [2, 3, 4, 6, 12]
    assert default_temporal_aggregations(7) == [7]
    assert default_temporal_aggregations(4) == [2, 4]
    assert default_temporal_aggregations(24) == [2, 3, 4, 6, 8, 12, 24]
    assert default_temporal_aggregations(1) == []
    assert default_temporal_aggregations("D") == [7, 14, 28]
    assert default_temporal_aggregations("MS") == [3, 6, 12]

    spec = build_temporal_hierarchy(m=12)
    assert isinstance(spec, TemporalHierarchySpec)
    assert spec.m == 12
    assert spec.factors == (1, 2, 3, 4, 6, 12)
    # 1 + 2 + 3 + 4 + 6 + 12 = 28 rows, 12 bottom columns
    assert spec.S_temp.shape == (28, 12)
    assert spec.n_total == 28
    assert spec.n_bottom == 12
    # Top row is annual sum of all 12 months
    np.testing.assert_allclose(spec.S_temp[0], np.ones(12))
    # Bottom 12 rows are the identity matrix
    np.testing.assert_allclose(spec.S_temp[-12:], np.eye(12))

    # Custom aggregations auto-include 1 and m=12
    spec_custom = build_temporal_hierarchy_spec(freq="MS", aggregations=[3, 12])
    assert spec_custom.factors == (1, 3, 12)
    assert spec_custom.S_temp.shape == (1 + 4 + 12, 12)

    with pytest.raises(ConfigError):
        build_temporal_hierarchy(m=0)


def test_aggregate_temporal_series_and_history_modes() -> None:
    y = np.arange(1, 26, dtype=float)  # length 25 (non-divisible by 12 -> trims leading 1)
    agg_1 = aggregate_temporal_series(y, 1)
    agg_3 = aggregate_temporal_series(y, 3)
    agg_12 = aggregate_temporal_series(y, 12)
    assert len(agg_1) == 25
    assert len(agg_3) == 8
    assert len(agg_12) == 2
    np.testing.assert_allclose(agg_12[0], np.sum(y[1:13]))
    np.testing.assert_allclose(agg_12[1], np.sum(y[13:25]))

    df = _monthly_series(26)
    agg_df_12 = aggregate_temporal_series(df, 12, date_col="ds", target_col="y")
    assert isinstance(agg_df_12, pd.DataFrame)
    assert len(agg_df_12) == 2
    agg_df_3 = aggregate_temporal_series(df, 3, date_col="ds", target_col="y")
    assert len(agg_df_3) == 8
    hist_arr = aggregate_temporal_history(df, 12, target_col="y")
    assert len(hist_arr) == 2


@pytest.mark.parametrize(
    "method",
    ["ols", "wls_struct", "wls_var", "mint_shrink", "bottom_up", "top_down", "middle_out"],
)
def test_reconcile_temporal_matrix_exact_coherence(method: str) -> None:
    spec = build_temporal_hierarchy(m=12)
    rng = np.random.default_rng(7)
    # Build coherent truth + incoherent noise across 2 annual cycles
    y_bot_true = rng.uniform(10.0, 50.0, size=(2, 12))
    y_hat_all = (y_bot_true @ spec.S_temp.T) + rng.normal(0.0, 4.0, size=(2, spec.n_total))
    residuals_by_k = {
        int(k): rng.normal(0.0, 2.0, size=3 * (12 // int(k))) for k in spec.aggregations
    }
    history_by_k = {
        1: rng.uniform(10.0, 50.0, size=36),
    }

    g_temp = reconcile_temporal_matrix(
        spec,
        method=method,
        residuals_by_k=residuals_by_k,
        history_by_k=history_by_k,
    )
    assert g_temp.shape == (12, spec.n_total)
    p_temp = spec.S_temp @ g_temp
    y_tilde = y_hat_all @ p_temp.T
    assert y_tilde.shape == (2, spec.n_total)
    y_bot_rec = y_tilde[:, -spec.m :]
    reconstructed = y_bot_rec @ spec.S_temp.T
    max_err = float(np.max(np.abs(y_tilde - reconstructed)))
    assert max_err < 1e-6


def test_reconcile_series_temporal_and_non_divisible_horizon() -> None:
    df = _monthly_series(36)
    cfg = RunConfig.model_validate(
        {
            "run_name": "temp-rec",
            "data": {"source_table": "p.d.t", "freq": "MS", "horizon": 14},
            "models": ["naive_mean"],
            "hierarchy": {
                "reconcile_temporal": True,
                "temporal_aggregations": [3, 6, 12],
                "reconciliation_methods": ["mint_shrink"],
            },
        }
    )
    # Create base predictions for horizon=14 (not divisible by 12)
    dates = pd.date_range("2025-01-01", periods=14, freq="MS")
    base_preds = pd.DataFrame(
        {
            "forecast_date": dates,
            "yhat": np.linspace(110.0, 130.0, 14),
            "yhat_lower": np.linspace(100.0, 120.0, 14),
            "yhat_upper": np.linspace(120.0, 140.0, 14),
            "quantiles": [{"0.1": 100.0, "0.5": 110.0, "0.9": 120.0} for _ in range(14)],
        }
    )
    rec_preds, agg_by_k = reconcile_series_temporal(
        df,
        base_preds,
        cfg,
        method="mint_shrink",
    )
    assert len(rec_preds) == 14
    assert bool(agg_by_k)
    # Verify interval widths are preserved and ordering holds
    orig_width = base_preds["yhat_upper"] - base_preds["yhat_lower"]
    rec_width = rec_preds["yhat_upper"] - rec_preds["yhat_lower"]
    np.testing.assert_allclose(orig_width, rec_width, atol=1e-6)
    assert bool((rec_preds["yhat_lower"] <= rec_preds["yhat"] + 1e-9).all())
    assert bool((rec_preds["yhat"] <= rec_preds["yhat_upper"] + 1e-9).all())

    disc = verify_temporal_coherence(
        rec_preds,
        agg_by_k,
        m=12,
        aggregations=[3, 6, 12],
        atol=1e-6,
    )
    assert disc < 1e-6

    # Also test reconcile_temporal_forecasts with multiple series and return_aggregations=True
    panel_preds = pd.concat(
        [
            base_preds.assign(ts_id="s_0", model_type="naive_mean"),
            base_preds.assign(ts_id="s_1", model_type="naive_mean", yhat=base_preds["yhat"] * 1.2),
        ],
        ignore_index=True,
    )
    panel_hist = pd.concat([df, df.assign(ts_id="s_1")], ignore_index=True)
    rec_panel, agg_panel = reconcile_temporal_forecasts(
        panel_preds,
        history_df=panel_hist,
        cfg=cfg,
        method="wls_struct",
        return_aggregations=True,
    )
    assert len(rec_panel) == 28
    assert verify_temporal_coherence(rec_panel, agg_panel, cfg=cfg) < 1e-6


def test_cross_temporal_reconciliation_simultaneous_coherence() -> None:
    raw = _monthly_hierarchy_panel(n_months=36, seed=19)
    cfg = RunConfig.model_validate(
        {
            "run_name": "ct-test",
            "data": {"source_table": "p.d.t", "freq": "MS", "horizon": 12},
            "models": ["naive_mean"],
            "hierarchy": {
                "enabled": True,
                "levels": [["region"], ["region", "category"]],
                "reconcile_temporal": True,
                "temporal_aggregations": [3, 6, 12],
                "cross_temporal": True,
                "reconciliation_methods": ["cross_temporal_mint", "mint_shrink"],
            },
        }
    )
    panel, spec = build_hierarchy(raw, cfg)
    rng = np.random.default_rng(101)
    f_dates = pd.date_range("2025-01-01", periods=12, freq="MS")

    pred_rows = []
    oof_rows = []
    oof_dates = pd.date_range("2024-01-01", periods=12, freq="MS")
    for nid in spec.node_ids:
        base_val = rng.uniform(50.0, 200.0, size=12)
        for d, v in zip(f_dates, base_val, strict=True):
            pred_rows.append(
                {
                    "ts_id": nid,
                    "model_type": "naive_mean",
                    "forecast_date": d,
                    "yhat": float(v),
                    "yhat_lower": float(v - 10.0),
                    "yhat_upper": float(v + 10.0),
                }
            )
        for d, v in zip(oof_dates, base_val, strict=True):
            oof_rows.append(
                {
                    "ts_id": nid,
                    "model_type": "naive_mean",
                    "fold_id": 0,
                    "cutoff_date": pd.Timestamp("2023-12-01"),
                    "ds": d,
                    "y_true": float(v + rng.normal(0.0, 3.0)),
                    "yhat": float(v),
                }
            )
    preds_df = pd.DataFrame(pred_rows)
    oof_df = pd.DataFrame(oof_rows)

    rec_df, agg_df = cross_temporal_mint(
        preds_df,
        spec,
        history_df=panel,
        oof_df=oof_df,
        cfg=cfg,
        return_aggregations=True,
    )
    # Verify simultaneous cross-sectional AND temporal coherence < 1e-6
    assert verify_coherence(rec_df, spec) < 1e-6
    assert verify_cross_temporal_coherence(rec_df, spec, agg_df, cfg=cfg) < 1e-6

    # Also verify reconcile_forecasts and reconcile_oof dispatch for cross_temporal_mint
    rec_via_entry = reconcile_forecasts(
        preds_df,
        spec,
        "cross_temporal_mint",
        history_df=panel,
        oof_df=oof_df,
        cfg=cfg,
    )
    assert verify_coherence(rec_via_entry, spec) < 1e-6

    rec_oof_df = reconcile_oof(
        oof_df,
        spec,
        "cross_temporal_mint",
        cfg=cfg,
        history_df=panel,
    )
    assert verify_coherence(rec_oof_df, spec, date_col="ds") < 1e-6

    # Verify alias reconcile_cross_temporal_forecasts
    alias_df = reconcile_cross_temporal_forecasts(
        preds_df,
        spec,
        history_df=panel,
        cfg=cfg,
    )
    assert isinstance(alias_df, pd.DataFrame)
    assert verify_coherence(alias_df, spec) < 1e-6

    # Verify raise_on_error on incoherent predictions
    with pytest.raises(DataError):
        verify_cross_temporal_coherence(preds_df, spec, cfg=cfg, atol=1e-6, raise_on_error=True)


def test_worker_and_spark_io_temporal_reconciliation_integration() -> None:
    df = _monthly_series(36)
    cfg = RunConfig.model_validate(
        {
            "run_name": "worker-temp",
            "data": {"source_table": "p.d.t", "freq": "MS", "horizon": 12},
            "models": ["naive_mean"],
            "backtest": {"enabled": False},
            "hierarchy": {
                "reconcile_temporal": True,
                "temporal_aggregations": [3, 6, 12],
                "reconciliation_methods": ["wls_struct"],
            },
        }
    )
    res = run_cell(df, "naive_mean", cfg)
    assert res.status == "ok"
    assert len(res.predictions) == 12

    # Also test hierarchical + cross_temporal through spark_io.run_group and reconcile_cells
    raw_hier = _monthly_hierarchy_panel(n_months=36, seed=23)
    cfg_hier = RunConfig.model_validate(
        {
            "run_name": "spark-ct",
            "data": {"source_table": "p.d.t", "freq": "MS", "horizon": 12},
            "models": ["naive_mean"],
            "backtest": {"enabled": False},
            "hierarchy": {
                "enabled": True,
                "levels": [["region"], ["region", "category"]],
                "reconcile_temporal": True,
                "temporal_aggregations": [3, 6, 12],
                "cross_temporal": True,
                "reconciliation_methods": ["cross_temporal_mint"],
            },
        }
    )
    panel, spec = build_hierarchy(raw_hier, cfg_hier)
    base_cells = [
        run_cell(sub, "naive_mean", cfg_hier) for _, sub in panel.groupby("ts_id", sort=False)
    ]
    rec_cells = reconcile_cells(base_cells, cfg_hier, spec, history_df=panel)
    assert len(rec_cells) == len(spec.node_ids)
    assert all(c.model_type == "naive_mean_cross_temporal_mint" for c in rec_cells)

    combined_preds = pd.concat(
        [c.predictions.assign(ts_id=c.ts_id, model_type=c.model_type) for c in rec_cells],
        ignore_index=True,
    )
    assert verify_coherence(combined_preds, spec) < 1e-6

    # Verify spark_io.run_group runs cleanly
    group_cells, status_df = run_group(raw_hier, cfg_hier, models=["naive_mean"])
    assert "naive_mean_cross_temporal_mint" in set(status_df["model_type"])
    assert len(group_cells) >= len(spec.node_ids)


def test_sdk_playground_auto_and_mcp_temporal_reconciliation() -> None:
    import scale_forecasting as sf
    from scale_forecasting.mcp import McpServer
    from scale_forecasting.playground import bakeoff, run_model
    from scale_forecasting.temporal_reconciliation import (
        DEFAULT_TEMPORAL_FACTORS,
        build_temporal_summing_matrix,
        reconcile_temporal_series,
        resolve_temporal_aggregations,
        temporal_covariance_matrix,
    )

    assert DEFAULT_TEMPORAL_FACTORS["D"] == (1, 7, 14, 28)
    assert resolve_temporal_aggregations is default_temporal_aggregations
    assert callable(build_temporal_summing_matrix)
    assert reconcile_temporal_series is reconcile_series_temporal
    assert callable(temporal_covariance_matrix)

    df = _monthly_series(36)
    cfg = RunConfig.model_validate(
        {
            "run_name": "sdk-temp",
            "data": {"source_table": "p.d.t", "freq": "MS", "horizon": 12},
            "models": ["naive_mean"],
        }
    )
    f = sf.Forecaster(cfg, data=df)
    rec_df = f.temporal_reconcile_df(method="wls_struct", aggregations=[3, 6, 12])
    assert len(rec_df) == 12
    assert bool((rec_df["yhat_lower"] <= rec_df["yhat"] + 1e-9).all())

    pr = run_model("naive_mean", horizon=14, backtest=True)
    pr_temp = pr.temporal_reconcile_df(method="wls_struct", aggregations=[7, 14])
    assert len(pr_temp) == 14

    bo = bakeoff(models=["naive_mean", "theta"], horizon=14, n_folds=1)
    bo_temp = bo.temporal_reconcile_df(method="wls_struct", aggregations=[7, 14])
    assert len(bo_temp) == 56

    srv = McpServer()
    mcp_res = srv._dispatch_tool(
        "temporal_reconcile",
        {
            "model": "naive_mean",
            "horizon": 14,
            "method": "wls_struct",
        },
    )
    assert mcp_res["status"] == "ok"
    assert len(mcp_res["rows"]) == 14


def test_three_reconciliation_modes_and_spatial_temporal_weight_auto_learning() -> None:
    # F1 & F2: HierarchyConfig Mode 3 auto-normalization and validation
    cfg_m3 = RunConfig.model_validate(
        {
            "run_name": "mode3-norm",
            "data": {"source_table": "p.d.t", "freq": "MS", "horizon": 12},
            "models": ["naive_mean"],
            "hierarchy": {
                "levels": [["region"], ["region", "category"]],
                "cross_temporal": True,
                "reconciliation_methods": ["mint_shrink", "mint_nonneg"],
            },
        }
    )
    assert cfg_m3.hierarchy.enabled is True
    assert cfg_m3.hierarchy.reconcile_temporal is True
    assert cfg_m3.hierarchy.cross_temporal is True
    assert cfg_m3.hierarchy.spatial_temporal_weight == "auto"
    assert "cross_temporal_mint" in cfg_m3.hierarchy.reconciliation_methods

    with pytest.raises(Exception, match="cross_temporal=true requires at least one"):
        RunConfig.model_validate(
            {
                "run_name": "bad-ct",
                "data": {"source_table": "p.d.t", "freq": "MS", "horizon": 12},
                "models": ["naive_mean"],
                "hierarchy": {"cross_temporal": True, "levels": []},
            }
        )

    for bad_w in (-0.1, 1.5, True):
        with pytest.raises(Exception, match="spatial_temporal_weight"):
            RunConfig.model_validate(
                {
                    "run_name": "bad-w",
                    "data": {"source_table": "p.d.t", "freq": "MS", "horizon": 12},
                    "models": ["naive_mean"],
                    "hierarchy": {"spatial_temporal_weight": bad_w},
                }
            )

    # F3: Mode 2 flat-panel temporal OOF reconciliation & metrics recomputation
    s0 = _monthly_series(60, seed=10)
    s1 = _monthly_series(60, seed=20).assign(ts_id="s_1")
    flat_panel = pd.concat([s0, s1], ignore_index=True)
    cfg_m2 = RunConfig.model_validate(
        {
            "run_name": "mode2-flat-temp",
            "data": {"source_table": "p.d.t", "freq": "MS", "horizon": 12},
            "models": ["naive_mean"],
            "backtest": {"enabled": True, "n_folds": 1, "horizon": 12, "step": 12, "min_train": 24},
            "hierarchy": {
                "reconcile_temporal": True,
                "temporal_aggregations": [3, 6, 12],
                "reconciliation_methods": ["mint_shrink", "wls_struct"],
            },
        }
    )
    m2_cells, m2_status = run_group(flat_panel, cfg_m2, models=["naive_mean"])
    assert {"naive_mean", "naive_mean_mint_shrink", "naive_mean_wls_struct"}.issubset(
        set(m2_status["model_type"])
    )
    rec_m2_cells = [c for c in m2_cells if c.model_type == "naive_mean_mint_shrink"]
    assert len(rec_m2_cells) == 2
    for c in rec_m2_cells:
        assert c.oof is not None and not c.oof.empty
        assert np.isfinite(c.metrics["mase"])
        assert np.isfinite(c.metrics["wape"])

    # F4 & F5: Mode 3 side-by-side leaderboard preservation + alpha auto-learning + diagnostics
    raw_hier = _monthly_hierarchy_panel(n_months=36, seed=31)
    panel, spec = build_hierarchy(raw_hier, cfg_m3)
    base_cells = [
        run_cell(sub, "naive_mean", cfg_m3) for _, sub in panel.groupby("ts_id", sort=False)
    ]
    # Perturb base predictions so spatial-only and cross-temporal differ
    rng = np.random.default_rng(99)
    for c in base_cells:
        c.predictions["yhat"] = c.predictions["yhat"] + rng.normal(
            0.0, 5.0, size=len(c.predictions)
        )

    rec_m3_cells = reconcile_cells(base_cells, cfg_m3, spec, history_df=panel)
    m3_models = {c.model_type for c in rec_m3_cells}
    assert m3_models == {
        "naive_mean_mint_shrink",
        "naive_mean_mint_nonneg",
        "naive_mean_cross_temporal_mint",
    }
    for m_name in m3_models:
        m_df = pd.concat(
            [
                c.predictions.assign(ts_id=c.ts_id, model_type=c.model_type)
                for c in rec_m3_cells
                if c.model_type == m_name
            ],
            ignore_index=True,
        )
        assert verify_coherence(m_df, spec) < 1e-6

    # Check diagnostics across all 3 modes
    all_base_preds = pd.concat(
        [c.predictions.assign(ts_id=c.ts_id, model_type=c.model_type) for c in base_cells],
        ignore_index=True,
    )
    spatial_only_df = reconcile_forecasts(all_base_preds, spec, "mint_shrink", history_df=panel)
    diag_m1 = spatial_only_df.attrs["cross_temporal_diagnostics"]
    assert diag_m1["mode"] == "spatial_only"
    assert diag_m1["spatial_weight_alpha"] == 1.0
    assert diag_m1["final_cs_discrepancy"] < 1e-6

    temp_only_df = reconcile_temporal_forecasts(
        all_base_preds, history_df=panel, cfg=cfg_m2, method="wls_struct"
    )
    assert isinstance(temp_only_df, pd.DataFrame)
    diag_m2 = temp_only_df.attrs["cross_temporal_diagnostics"]
    assert diag_m2["mode"] == "temporal_only"
    assert diag_m2["spatial_weight_alpha"] == 0.0
    assert diag_m2["final_temp_discrepancy"] < 1e-6

    ct_auto_df = cross_temporal_mint(
        all_base_preds,
        spec,
        history_df=panel,
        cfg=cfg_m3,
        spatial_temporal_weight="auto",
    )
    assert isinstance(ct_auto_df, pd.DataFrame)
    diag_m3 = ct_auto_df.attrs["cross_temporal_diagnostics"]
    assert diag_m3["mode"] == "cross_temporal"
    assert 0.05 <= diag_m3["spatial_weight_alpha"] <= 0.95
    assert diag_m3["final_cs_discrepancy"] < 1e-6
    assert diag_m3["final_temp_discrepancy"] < 1e-6

    ct_fixed_df = cross_temporal_mint(
        all_base_preds,
        spec,
        history_df=panel,
        cfg=cfg_m3,
        spatial_temporal_weight=0.3,
    )
    assert isinstance(ct_fixed_df, pd.DataFrame)
    assert abs(ct_fixed_df.attrs["cross_temporal_diagnostics"]["spatial_weight_alpha"] - 0.3) < 1e-9
    assert ct_fixed_df.attrs["cross_temporal_diagnostics"]["final_cs_discrepancy"] < 1e-6
    assert ct_fixed_df.attrs["cross_temporal_diagnostics"]["final_temp_discrepancy"] < 1e-6


def test_sample_coherent_paths_and_verify_simplex_alias() -> None:
    import scale_forecasting as sf
    from scale_forecasting.playground import bakeoff, run_model
    from scale_forecasting.temporal_reconciliation import sample_coherent_paths

    raw_hier = _monthly_hierarchy_panel(n_months=36, seed=55)
    cfg = RunConfig.model_validate(
        {
            "run_name": "paths-ct",
            "data": {"source_table": "p.d.t", "freq": "MS", "horizon": 12},
            "models": ["naive_mean"],
            "hierarchy": {
                "enabled": True,
                "levels": [["region"], ["region", "category"]],
                "reconcile_temporal": True,
                "temporal_aggregations": [3, 6, 12],
                "cross_temporal": True,
            },
        }
    )
    panel, spec = build_hierarchy(raw_hier, cfg)
    base_cells = [run_cell(sub, "naive_mean", cfg) for _, sub in panel.groupby("ts_id", sort=False)]
    preds_df = pd.concat(
        [c.predictions.assign(ts_id=c.ts_id, model_type=c.model_type) for c in base_cells],
        ignore_index=True,
    )

    paths_df = sample_coherent_paths(
        preds_df,
        spec,
        history_df=panel,
        cfg=cfg,
        n_paths=15,
        seed=42,
        nonneg=True,
    )
    assert {"path_id", "ts_id", "forecast_date", "yhat_sample"}.issubset(paths_df.columns)
    assert paths_df["path_id"].nunique() == 15
    assert bool((paths_df["yhat_sample"] >= -1e-9).all())
    # verify_coherence and verify_temporal_coherence auto-detect yhat_sample and forecast_date
    assert verify_coherence(paths_df, spec) < 1e-6
    assert verify_temporal_coherence(paths_df, cfg=cfg) < 1e-6

    f = sf.Forecaster(cfg, data=raw_hier)
    f_paths = f.sample_coherent_paths(forecasts_df=preds_df, n_paths=8, seed=7)
    assert f_paths["path_id"].nunique() == 8
    assert verify_coherence(f_paths, spec) < 1e-6

    pr = run_model("naive_mean", horizon=14, backtest=True)
    pr_paths = pr.sample_coherent_paths(n_paths=6, seed=11)
    assert pr_paths["path_id"].nunique() == 6

    bo = bakeoff(models=["naive_mean", "theta"], horizon=14, n_folds=1)
    bo_paths = bo.sample_coherent_paths(n_paths=6, seed=11)
    assert bo_paths["path_id"].nunique() == 6

    # Verify verify_simplex alias with forecast_date column and custom simplex_sum
    simplex_df = pd.DataFrame(
        {
            "ts_id": ["a", "b", "a", "b"],
            "forecast_date": pd.to_datetime(
                ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"]
            ),
            "yhat": [60.0, 40.0, 25.0, 75.0],
        }
    )
    assert sf.verify_simplex(simplex_df, simplex_sum=100.0) < 1e-6
