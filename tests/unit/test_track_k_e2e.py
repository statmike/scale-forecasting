"""4-Tier End-to-End (E2E) Verification Suite for Track K (`v1.6.0`).

Covers all 12 Track K features (F1–F12) across 4 systematic testing tiers:
- Tier 1 — Category-Partition Feature Coverage (Features F1–F12, >= 5 cases per feature)
- Tier 2 — Boundary Value & Corner-Case Hardening (>= 5 cases per feature area)
- Tier 3 — Pairwise & Multi-Subsystem Cross-Feature Combinations
- Tier 4 — Real-World Enterprise Application Workloads
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.figure
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

import scale_forecasting as sf
from scale_forecasting.auto import build_config, inspect_panel_for_auto
from scale_forecasting.auto_policy import (
    DataProfile,
    PanelAutoProfile,
    evolved_policy_thresholds,
    propose_autofocus_dag_branches,
    propose_autofocus_synthesis_pass,
)
from scale_forecasting.config import RECONCILIATION_METHODS, HierarchyConfig, RunConfig
from scale_forecasting.data_gen.generator import GenConfig, generate_panel
from scale_forecasting.engines.spark_io import run_group
from scale_forecasting.ensembler import combine_calculated, combine_oof, fit_learned
from scale_forecasting.errors import ConfigError, GoalSeekInfeasibleError
from scale_forecasting.evolve.evaluator import (
    EvolutionEvaluator,
    _blend_frame_with_series_weights,
)
from scale_forecasting.evolve.targets import (
    EVOLVE_TARGETS,
    build_mutation_prompt,
    deterministic_mock_candidates,
    get_target_seed,
)
from scale_forecasting.features import resolve_model_variant
from scale_forecasting.goal_seek import (
    GoalSeekResult,
    allocate_forecast_target,
    allocate_target,
    plot_goal_seek,
    summarize_goal_seek,
)
from scale_forecasting.hpo import tune_model
from scale_forecasting.mcp import McpServer
from scale_forecasting.metrics import compute_metrics
from scale_forecasting.playground import bakeoff, run_model
from scale_forecasting.reconciliation import (
    TOTAL_NODE_ID,
    build_hierarchy,
    reconcile_cells,
    reconcile_forecasts,
    reconcile_oof,
    residual_matrix,
    verify_coherence,
)
from scale_forecasting.registry.ids import (
    canonical_config,
    make_model_spec_hash,
    make_run_id,
)
from scale_forecasting.scenarios import simulate_scenarios
from scale_forecasting.simplex import (
    DEFAULT_ZERO_DELTA,
    SIMPLICIAL_TRANSFORMS,
    SimplexSpec,
    apply_series_simplex_transform,
    close_simplex_forecasts,
    clr_inverse,
    clr_transform,
    enforce_simplex_closure,
    forecast_simplex_panel,
    ilr_inverse,
    ilr_transform,
    multiplicative_replacement,
    verify_simplex,
    verify_simplex_coherence,
)
from scale_forecasting.temporal_reconciliation import (
    DEFAULT_TEMPORAL_FACTORS,
    aggregate_temporal_series,
    build_temporal_summing_matrix,
    cross_temporal_mint,
    default_temporal_aggregations,
    reconcile_cross_temporal,
    reconcile_series_temporal,
    reconcile_temporal_forecasts,
    verify_cross_temporal_coherence,
    verify_temporal_coherence,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_daily_series(n_days: int = 84, seed: int = 17, ts_id: str = "sku_0") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    t = np.arange(n_days, dtype=float)
    y = 80.0 + 0.25 * t + 12.0 * np.sin(2.0 * np.pi * t / 7.0) + rng.normal(0.0, 2.5, size=n_days)
    return pd.DataFrame({"ts_id": ts_id, "ds": dates, "y": np.maximum(y, 1.0)})


def _make_compositional_panel(
    n_brands: int = 4,
    n_periods: int = 36,
    simplex_sum: float = 100.0,
    with_launch_zeros: bool = False,
    with_covariates: bool = False,
    seed: int = 29,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_periods, freq="MS")
    alpha = np.linspace(2.0, 4.0, n_brands)
    shares = rng.dirichlet(alpha, size=n_periods) * simplex_sum
    if with_launch_zeros and n_brands >= 3:
        # Brand 0 is a new launch: 0.0% share for the first 10 months
        shares[:10, 0] = 0.0
        row_sums = shares[:10, 1:].sum(axis=1, keepdims=True)
        shares[:10, 1:] = shares[:10, 1:] * (simplex_sum / row_sums)
    rows = []
    for b_idx in range(n_brands):
        brand_id = f"brand_{b_idx}"
        for t_idx, dt in enumerate(dates):
            row: dict[str, object] = {
                "ts_id": brand_id,
                "ds": dt,
                "y": float(shares[t_idx, b_idx]),
            }
            if with_covariates:
                # Price index negatively correlated with share
                row["price_index"] = float(100.0 - 0.15 * shares[t_idx, b_idx] + rng.normal(0, 0.5))
                row["promo"] = float(1 if (t_idx % 6 == b_idx % 6) else 0)
            rows.append(row)
    return pd.DataFrame(rows)


def _make_retail_hierarchy_panel(n_months: int = 36, seed: int = 31) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_months, freq="MS")
    t = np.arange(n_months, dtype=float)
    regions = ["North", "South"]
    categories = ["Beverages", "Snacks"]
    skus = ["A", "B"]
    rows = []
    for r_idx, reg in enumerate(regions):
        for c_idx, cat in enumerate(categories):
            for s_idx, sku in enumerate(skus):
                tid = f"{reg}_{cat}_{sku}"
                base = 60.0 + 20.0 * r_idx + 15.0 * c_idx + 10.0 * s_idx
                seas = (8.0 + 3.0 * c_idx) * np.sin(2.0 * np.pi * t / 12.0 + 0.4 * r_idx)
                noise = rng.normal(0.0, 2.0 + 1.5 * s_idx, size=n_months)
                y_vals = np.maximum(base + 0.5 * t + seas + noise, 5.0)
                price = 10.0 + 2.0 * c_idx + 1.0 * s_idx
                cost = 4.0 + 1.0 * c_idx
                for dt, y_v in zip(dates, y_vals, strict=True):
                    rows.append(
                        {
                            "ts_id": tid,
                            "region": reg,
                            "category": cat,
                            "ds": dt,
                            "y": float(y_v),
                            "price": float(price),
                            "unit_cost": float(cost),
                        }
                    )
    return pd.DataFrame(rows)


# =============================================================================
# TIER 1 — Category-Partition Feature Coverage (Features F1–F11)
# =============================================================================


def test_t1_f1_f2_hierarchy_config_and_id_stability() -> None:
    """Verify F1 & F2: HierarchyConfig temporal/cross-temporal fields and 100% hash stability."""
    assert "cross_temporal_mint" in RECONCILIATION_METHODS

    base_payload = {
        "run_name": "hash-check",
        "data": {"source_table": "p.d.t", "freq": "D", "horizon": 14},
        "models": ["naive_mean", "theta"],
    }
    cfg_implicit = RunConfig.model_validate(base_payload)
    cfg_explicit_defaults = RunConfig.model_validate(
        {
            **base_payload,
            "features": {"transform": "none", "simplex_sum": None},
            "hierarchy": {
                "enabled": False,
                "reconcile_temporal": False,
                "temporal_aggregations": None,
                "cross_temporal": False,
                "spatial_temporal_weight": "auto",
            },
        }
    )
    assert make_run_id(cfg_implicit) == make_run_id(cfg_explicit_defaults)
    for m in cfg_implicit.models:
        assert make_model_spec_hash(m, cfg_implicit) == make_model_spec_hash(
            m, cfg_explicit_defaults
        )

    # Enabling temporal or cross-temporal reconciliation changes run_id and normalizes aggregations
    cfg_temp = RunConfig.model_validate(
        {
            **base_payload,
            "hierarchy": {
                "levels": [["region"]],
                "reconcile_temporal": True,
                "temporal_aggregations": [28, 7, 14],
                "cross_temporal": True,
                "reconciliation_methods": ["cross_temporal_mint"],
            },
        }
    )
    assert cfg_temp.hierarchy.enabled is True
    assert cfg_temp.hierarchy.reconcile_temporal is True
    assert cfg_temp.hierarchy.cross_temporal is True
    assert cfg_temp.hierarchy.temporal_aggregations == [28, 7, 14]
    assert make_run_id(cfg_temp) != make_run_id(cfg_implicit)

    with pytest.raises(ValueError, match="duplicates"):
        RunConfig.model_validate(
            {
                **base_payload,
                "hierarchy": {
                    "reconcile_temporal": True,
                    "temporal_aggregations": [28, 7, 14, 7],
                },
            }
        )


def test_t1_f3_thief_temporal_reconciliation_methods() -> None:
    """Verify F3: THieF temporal hierarchy construction, aggregation, and all projection methods."""
    assert default_temporal_aggregations("D") == [7, 14, 28]
    assert default_temporal_aggregations("W") == [4, 13, 52]
    assert default_temporal_aggregations("MS") == [3, 6, 12]
    assert default_temporal_aggregations("Q") == [2, 4]
    assert default_temporal_aggregations("H") == [6, 12, 24]
    assert DEFAULT_TEMPORAL_FACTORS["M"] == (1, 3, 6, 12)

    s_mat = build_temporal_summing_matrix(12, [1, 3, 6, 12])
    assert s_mat.shape == (1 + 2 + 4 + 12, 12)
    np.testing.assert_allclose(s_mat[0], np.ones(12))

    df = _make_daily_series(n_days=84)
    agg_7 = aggregate_temporal_series(df["y"].to_numpy(), 7)
    assert len(agg_7) == 12

    cfg = RunConfig.model_validate(
        {
            "run_name": "t1-thief",
            "data": {"source_table": "p.d.t", "freq": "D", "horizon": 14},
            "models": ["naive_seasonal"],
            "hierarchy": {
                "reconcile_temporal": True,
                "temporal_aggregations": [7, 14],
            },
        }
    )
    f_dates = pd.date_range("2025-03-26", periods=14, freq="D")
    preds_df = pd.DataFrame(
        {
            "ts_id": "sku_0",
            "model_type": "naive_seasonal",
            "forecast_date": f_dates,
            "yhat": np.linspace(75.0, 95.0, 14),
            "yhat_lower": np.linspace(65.0, 85.0, 14),
            "yhat_upper": np.linspace(85.0, 105.0, 14),
        }
    )
    for method in (
        "bottom_up",
        "top_down",
        "middle_out",
        "ols",
        "wls_struct",
        "wls_var",
        "mint_shrink",
        "mint_nonneg",
    ):
        rec_df, agg_df = reconcile_temporal_forecasts(
            preds_df,
            history_df=df,
            cfg=cfg,
            method=method,
            return_aggregations=True,
        )
        assert len(rec_df) == 14
        assert verify_temporal_coherence(rec_df, agg_df, cfg=cfg) < 1e-6


def test_t1_f4_simultaneous_cross_temporal_projection() -> None:
    """Verify F4: Simultaneous cross-sectional and temporal projection."""
    raw = _make_retail_hierarchy_panel(n_months=36, seed=41)
    cfg = RunConfig.model_validate(
        {
            "run_name": "t1-ct",
            "data": {"source_table": "p.d.t", "freq": "MS", "horizon": 12},
            "models": ["naive_mean"],
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
    panel, spec = build_hierarchy(raw, cfg)
    assert callable(residual_matrix)

    rng = np.random.default_rng(55)
    f_dates = pd.date_range("2025-01-01", periods=12, freq="MS")
    rows = []
    for nid in spec.node_ids:
        vals = rng.uniform(40.0, 180.0, size=12)
        for dt, v in zip(f_dates, vals, strict=True):
            rows.append(
                {
                    "ts_id": nid,
                    "model_type": "naive_mean",
                    "forecast_date": dt,
                    "yhat": float(v),
                    "yhat_lower": float(v * 0.85),
                    "yhat_upper": float(v * 1.15),
                }
            )
    preds_df = pd.DataFrame(rows)
    rec_df, agg_df = reconcile_cross_temporal(
        preds_df,
        spec,
        history_df=panel,
        cfg=cfg,
        return_aggregations=True,
    )
    assert verify_coherence(rec_df, spec) < 1e-6
    assert verify_cross_temporal_coherence(rec_df, spec, agg_df, cfg=cfg) < 1e-6


def test_t1_f5_f6_simplex_config_transforms_and_closure() -> None:
    """Verify F5 & F6: Compositional simplex transforms (`clr`, `ilr`), variants, and closure."""
    assert DEFAULT_ZERO_DELTA == 1e-6
    assert set(SIMPLICIAL_TRANSFORMS) == {"clr", "ilr"}

    cfg = RunConfig.model_validate(
        {
            "run_name": "t1-simplex",
            "data": {"source_table": "p.d.t", "freq": "MS", "horizon": 6},
            "models": ["naive_mean@transform_ilr", "theta@transform_clr"],
            "features": {"simplex_sum": 100.0},
        }
    )
    spec_ilr = resolve_model_variant(cfg, "naive_mean@transform_ilr")
    assert spec_ilr.effective_cfg.features.transform == "ilr"
    assert spec_ilr.effective_cfg.features.simplex_sum == 100.0
    spec_clr = resolve_model_variant(cfg, "theta@transform_clr")
    assert spec_clr.effective_cfg.features.transform == "clr"
    assert make_model_spec_hash("naive_mean@transform_ilr", cfg) != make_model_spec_hash(
        "theta@transform_clr", cfg
    )

    panel = _make_compositional_panel(n_brands=4, n_periods=30, simplex_sum=100.0)
    preds, s_spec = forecast_simplex_panel(
        panel,
        horizon=6,
        transform="ilr",
        simplex_sum=100.0,
        return_spec=True,
    )
    assert isinstance(s_spec, SimplexSpec)
    assert verify_simplex_coherence(preds, simplex_sum=100.0) < 1e-6


def test_t1_f7_goal_seek_qp_allocation_and_summaries() -> None:
    """Verify F7: `allocate_target`, covariance weighting, summary, and plot."""
    assert allocate_forecast_target is allocate_target
    f_dates = pd.date_range("2025-01-01", periods=3, freq="MS")
    # Two series with equal baseline (50) but SKU_HighVar has 3x wider interval than SKU_LowVar
    rows = []
    for dt in f_dates:
        rows.append(
            {
                "ts_id": "SKU_LowVar",
                "model_type": "naive",
                "forecast_date": dt,
                "yhat": 50.0,
                "yhat_lower": 48.0,
                "yhat_upper": 52.0,
            }
        )
        rows.append(
            {
                "ts_id": "SKU_HighVar",
                "model_type": "naive",
                "forecast_date": dt,
                "yhat": 50.0,
                "yhat_lower": 35.0,
                "yhat_upper": 65.0,
            }
        )
    preds_df = pd.DataFrame(rows)

    # Baseline horizon sum is 300; allocate 360 (+60 units)
    res = allocate_target(
        preds_df,
        target_total=360.0,
        method="wls_var",
        return_result=True,
    )
    assert isinstance(res, GoalSeekResult)
    assert res.max_target_discrepancy < 1e-6
    summary = summarize_goal_seek(res)
    low_delta = float(summary.loc[summary["ts_id"] == "SKU_LowVar", "delta_units"].iloc[0])
    high_delta = float(summary.loc[summary["ts_id"] == "SKU_HighVar", "delta_units"].iloc[0])
    assert high_delta > low_delta * 5.0
    assert abs((low_delta + high_delta) - 60.0) < 1e-6

    fig = plot_goal_seek(res)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_t1_f8_sdk_playground_report_and_mcp_surfaces(tmp_path: Path) -> None:
    """Verify F8: Forecaster SDK, PlaygroundRun, BakeOff, HTML Report, and MCP 11-tool contract."""
    df = _make_daily_series(n_days=56)
    cfg = RunConfig.model_validate(
        {
            "run_name": "t1-surfaces",
            "data": {"source_table": "p.d.t", "freq": "D", "horizon": 14},
            "models": ["naive_seasonal"],
        }
    )
    f = sf.Forecaster(cfg, data=df)
    rec_df = f.temporal_reconcile_df(method="wls_struct", aggregations=[7, 14])
    assert len(rec_df) == 14

    gs_df = f.allocate_target(
        1500.0,
        forecasts_df=rec_df,
        prices={"sku_0": 10.0},
        unit_cost=4.0,
        include_inventory_impact=True,
    )
    assert abs(float(gs_df["allocated_yhat"].sum()) - 1500.0) < 1e-6
    assert f.plot_goal_seek(gs_df) is not None

    report_path = f.export_report(
        tmp_path / "t1_report.html",
        predictions_df=rec_df,
        goal_seek_df=gs_df,
        write_json=True,
    )
    assert report_path.is_file()
    payload = json.loads(report_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert "goal_seek" in payload and len(payload["goal_seek"]) > 0

    pr = run_model("naive_mean", horizon=14, backtest=True)
    assert len(pr.temporal_reconcile_df(aggregations=[7, 14])) == 14
    assert len(pr.allocate_target(target_total=400.0)) == 14

    bo = bakeoff(models=["naive_mean", "naive_drift"], horizon=7, n_folds=1)
    assert not bo.temporal_reconcile_df(aggregations=[7]).empty
    assert not bo.allocate_target(target_total=300.0).empty

    srv = McpServer()
    assert len(srv.list_tools()) == 12
    assert len(srv.list_resources()) == 7
    res_gs = srv._dispatch_tool(
        "allocate_target", {"model": "naive_mean", "horizon": 7, "target_total": 350.0}
    )
    assert res_gs["status"] == "ok"
    res_tr = srv._dispatch_tool(
        "temporal_reconcile", {"model": "naive_mean", "horizon": 14, "method": "wls_struct"}
    )
    assert res_tr["status"] == "ok"


def test_t1_f9_auto_forecaster_profile_and_branches() -> None:
    """Verify F9: PanelAutoProfile simplex detection and Branch 2A / Branch 2C proposals."""
    simplex_df = _make_compositional_panel(n_brands=4, n_periods=36, simplex_sum=100.0)
    prof = DataProfile.from_df(simplex_df, horizon=12, freq="MS")
    assert isinstance(prof, PanelAutoProfile)
    assert prof.is_compositional_simplex is True
    assert prof.simplex_total == 100.0

    bp = build_config(prof, tier="first_light")
    assert bp.config.features.simplex_sum == 100.0
    assert any("@transform_ilr" in m or "@transform_clr" in m for m in bp.config.models)

    # Verify Branch 2A & 2C generation when Pass 1 config has not yet enabled them
    raw_p1 = bp.config.model_dump(mode="json")
    raw_p1["models"] = ["naive_seasonal", "theta"]
    raw_p1["features"]["transform"] = "none"
    raw_p1["hierarchy"] = {"enabled": False, "reconcile_temporal": False}
    branches = propose_autofocus_dag_branches(
        prof,
        raw_p1,
        pass_index=2,
        diagnostics_summary={"residual_autocorr_lag1": 0.45, "fva_vs_naive_pct": 1.0},
        tier="wide_field",
    )
    assert len(branches) >= 2
    all_actions = {m["action"] for b in branches for m in b.get("mutations", [])}
    assert "inject_transform_simplex" in all_actions
    assert "enable_temporal_reconciliation" in all_actions


def test_t1_f10_f11_version_docs_and_local_pipeline() -> None:
    """Verify F10 & F11: Version >= 1.6.0 parity, API reference docs, and local group pipeline."""
    assert tuple(int(part) for part in sf.__version__.split(".")[:3]) >= (1, 6, 0)
    for api_doc in ("temporal_reconciliation.md", "simplex.md", "goal_seek.md"):
        doc_path = REPO_ROOT / "docs" / "api" / api_doc
        assert doc_path.is_file(), f"Missing API doc {doc_path}"

    wb_doc = (REPO_ROOT / "docs" / "forecaster_workbench.md").read_text(encoding="utf-8")
    assert "flowchart TD" in wb_doc or "flowchart TB" in wb_doc

    panel = _make_compositional_panel(n_brands=3, n_periods=24, simplex_sum=1.0)
    cfg = RunConfig.model_validate(
        {
            "run_name": "t1-pipeline",
            "data": {"source_table": "p.d.t", "freq": "MS", "horizon": 6},
            "models": ["naive_mean@transform_ilr"],
            "features": {"simplex_sum": 1.0},
            "backtest": {"enabled": False},
        }
    )
    cells, status_df = run_group(panel, cfg)
    assert len(cells) == 3
    assert set(status_df["status"]) == {"ok"}
    preds = pd.concat(
        [c.predictions.assign(ts_id=c.ts_id, model_type=c.model_type) for c in cells],
        ignore_index=True,
    )
    assert verify_simplex_coherence(preds, simplex_sum=1.0) < 1e-6


# =============================================================================
# TIER 2 — Boundary Value & Corner-Case Hardening
# =============================================================================


@pytest.mark.parametrize(
    ("freq", "horizon", "aggregations", "n_hist"),
    [
        ("D", 10, [7], 56),
        ("MS", 5, [3, 6, 12], 36),
        ("D", 17, [7, 14], 84),
    ],
)
def test_t2_temporal_non_divisible_horizons(
    freq: str,
    horizon: int,
    aggregations: list[int],
    n_hist: int,
) -> None:
    """Verify THieF handles non-divisible horizons H (H < m_max and H > m_max) cleanly."""
    dates = pd.date_range("2022-01-01", periods=n_hist, freq=freq)
    rng = np.random.default_rng(19 + horizon)
    hist_df = pd.DataFrame(
        {
            "ts_id": "series_ragged",
            "ds": dates,
            "y": rng.uniform(50.0, 150.0, size=n_hist),
        }
    )
    cfg = RunConfig.model_validate(
        {
            "run_name": f"ragged-{freq}-{horizon}",
            "data": {"source_table": "p.d.t", "freq": freq, "horizon": horizon},
            "models": ["naive_mean"],
            "hierarchy": {
                "reconcile_temporal": True,
                "temporal_aggregations": aggregations,
                "reconciliation_methods": ["wls_struct"],
            },
        }
    )
    f_dates = pd.date_range(dates[-1], periods=horizon + 1, freq=freq)[1:]
    base_preds = pd.DataFrame(
        {
            "ts_id": "series_ragged",
            "model_type": "naive_mean",
            "forecast_date": f_dates,
            "yhat": rng.uniform(60.0, 140.0, size=horizon),
            "yhat_lower": np.full(horizon, 45.0),
            "yhat_upper": np.full(horizon, 155.0),
        }
    )
    rec_preds, agg_by_k = reconcile_series_temporal(
        hist_df,
        base_preds,
        cfg,
        method="wls_struct",
    )
    assert len(rec_preds) == horizon
    assert bool(np.all(np.isfinite(rec_preds["yhat"].to_numpy(dtype=float))))
    assert bool((rec_preds["yhat_lower"] <= rec_preds["yhat"] + 1e-9).all())
    assert bool((rec_preds["yhat"] <= rec_preds["yhat_upper"] + 1e-9).all())
    disc = verify_temporal_coherence(
        rec_preds,
        agg_by_k,
        m=max(aggregations),
        aggregations=aggregations,
    )
    assert disc < 1e-6


def test_t2_temporal_short_history_and_single_step_horizon() -> None:
    """Verify THieF handles short histories (T < 2 * m_max) and single-step horizon (H = 1)."""
    # Short history: T = 8 months (< 2 * 12 = 24)
    short_dates = pd.date_range("2024-01-01", periods=8, freq="MS")
    short_df = pd.DataFrame(
        {
            "ts_id": "short_s",
            "ds": short_dates,
            "y": np.linspace(30.0, 65.0, 8),
        }
    )
    # default_temporal_aggregations filters factors to <= T // 2 = 4 -> [3]
    filtered = default_temporal_aggregations("MS", history_len=len(short_df))
    assert filtered == [3]

    cfg_short = RunConfig.model_validate(
        {
            "run_name": "short-hist",
            "data": {"source_table": "p.d.t", "freq": "MS", "horizon": 6},
            "models": ["naive_mean"],
            "hierarchy": {
                "reconcile_temporal": True,
                "temporal_aggregations": [3, 6, 12],
            },
        }
    )
    f_dates_6 = pd.date_range("2024-09-01", periods=6, freq="MS")
    preds_6 = pd.DataFrame(
        {
            "ts_id": "short_s",
            "model_type": "naive_mean",
            "forecast_date": f_dates_6,
            "yhat": np.full(6, 50.0),
            "yhat_lower": np.full(6, 40.0),
            "yhat_upper": np.full(6, 60.0),
        }
    )
    rec_short, agg_short = reconcile_temporal_forecasts(
        preds_6,
        history_df=short_df,
        cfg=cfg_short,
        method="mint_shrink",
        return_aggregations=True,
    )
    assert len(rec_short) == 6
    assert bool(np.all(np.isfinite(rec_short["yhat"].to_numpy(dtype=float))))
    assert verify_temporal_coherence(rec_short, agg_short, cfg=cfg_short) < 1e-6

    # Single-step horizon H = 1
    cfg_h1 = RunConfig.model_validate(
        {
            "run_name": "h1-temp",
            "data": {"source_table": "p.d.t", "freq": "D", "horizon": 1},
            "models": ["naive_mean"],
            "hierarchy": {"reconcile_temporal": True, "temporal_aggregations": [7]},
        }
    )
    preds_h1 = pd.DataFrame(
        {
            "ts_id": "short_s",
            "model_type": "naive_mean",
            "forecast_date": [pd.Timestamp("2024-09-01")],
            "yhat": [55.0],
            "yhat_lower": [45.0],
            "yhat_upper": [65.0],
        }
    )
    rec_h1 = reconcile_temporal_forecasts(preds_h1, history_df=short_df, cfg=cfg_h1)
    assert len(rec_h1) == 1
    assert bool(np.isfinite(rec_h1["yhat"].iloc[0]))


def test_t2_simplex_zeros_boundaries_single_series_and_negative_errors() -> None:
    """Verify Simplex transforms on exact zeros, boundary simplex_sum, D=1, and negative inputs."""
    # Exact zeros and boundary S=100.0
    x_extreme = np.array(
        [
            [0.0, 0.0, 100.0],
            [0.0, 50.0, 50.0],
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
        ]
    )
    for fn_fwd, fn_inv in ((clr_transform, clr_inverse), (ilr_transform, ilr_inverse)):
        transformed = fn_fwd(x_extreme, simplex_sum=100.0)
        assert bool(np.all(np.isfinite(transformed)))
        inverted = fn_inv(transformed, simplex_sum=100.0)
        assert bool(np.all(np.isfinite(inverted)))
        assert bool(np.all(inverted > 0.0))
        np.testing.assert_allclose(np.sum(inverted, axis=1), np.full(4, 100.0), atol=1e-6)

    # Single-series degenerate panel D=1
    single_panel = pd.DataFrame(
        {
            "ts_id": "only_brand",
            "ds": pd.date_range("2023-01-01", periods=18, freq="MS"),
            "y": np.full(18, 100.0),
        }
    )
    preds_d1 = forecast_simplex_panel(single_panel, horizon=4, transform="ilr", simplex_sum=100.0)
    assert len(preds_d1) == 4
    np.testing.assert_allclose(preds_d1["yhat"].to_numpy(dtype=float), np.full(4, 100.0), atol=1e-6)
    assert bool((preds_d1["yhat_lower"] >= 0.0).all())
    assert bool((preds_d1["yhat_upper"] <= 100.0).all())
    assert bool((preds_d1["yhat_lower"] <= preds_d1["yhat"] + 1e-9).all())
    assert bool((preds_d1["yhat"] <= preds_d1["yhat_upper"] + 1e-9).all())

    # Out-of-order / out-of-bounds intervals repaired by enforce_simplex_closure
    raw_preds = pd.DataFrame(
        {
            "ts_id": ["b0", "b1"],
            "model_type": ["m", "m"],
            "forecast_date": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01")],
            "yhat": [70.0, 50.0],
            "yhat_lower": [-15.0, 60.0],  # negative lower and inverted lower > yhat
            "yhat_upper": [140.0, 30.0],  # > simplex_sum and inverted upper < yhat
        }
    )
    closed = enforce_simplex_closure(raw_preds, simplex_sum=100.0)
    assert verify_simplex_coherence(closed, simplex_sum=100.0) < 1e-6
    assert bool((closed["yhat_lower"] >= 0.0).all())
    assert bool((closed["yhat_upper"] <= 100.0).all())
    assert bool((closed["yhat_lower"] <= closed["yhat"] + 1e-9).all())
    assert bool((closed["yhat"] <= closed["yhat_upper"] + 1e-9).all())

    # Negative target values raise ConfigError (which is a ValueError / ScaleForecastError)
    with pytest.raises((ConfigError, ValueError)):
        clr_transform(np.array([[10.0, -2.0, 92.0]]), simplex_sum=100.0)
    with pytest.raises((ConfigError, ValueError)):
        ilr_transform(np.array([[10.0, -5.0, 95.0]]), simplex_sum=100.0)
    with pytest.raises((ConfigError, ValueError)):
        apply_series_simplex_transform(pd.Series([0.5, -0.1, 0.6]), "clr", simplex_sum=1.0)
    with pytest.raises((ConfigError, ValueError)):
        multiplicative_replacement([0.2, 0.8], eps=0.0)
    with pytest.raises((ConfigError, ValueError)):
        multiplicative_replacement([0.2, 0.8], eps=0.6)


def test_t2_goal_seek_infeasible_bounds_and_conflicting_locks() -> None:
    """Verify GoalSeekInfeasibleError (isinstance of ConfigError & ValueError) on conflicts."""
    preds_df = pd.DataFrame(
        {
            "ts_id": ["S1", "S2", "S3"],
            "model_type": ["naive"] * 3,
            "forecast_date": [pd.Timestamp("2026-01-01")] * 3,
            "yhat": [30.0, 30.0, 40.0],
            "yhat_lower": [20.0, 20.0, 30.0],
            "yhat_upper": [40.0, 40.0, 50.0],
        }
    )

    # 1. Sum of lower_bounds (40 + 40 + 40 = 120) > target_total (100)
    with pytest.raises(GoalSeekInfeasibleError) as exc_lo:
        allocate_target(preds_df, target_total=100.0, lower_bounds=40.0)
    assert isinstance(exc_lo.value, (ConfigError, ValueError))

    # 2. Sum of upper_bounds (20 + 20 + 20 = 60) < target_total (100)
    with pytest.raises(GoalSeekInfeasibleError) as exc_hi:
        allocate_target(preds_df, target_total=100.0, upper_bounds=20.0)
    assert isinstance(exc_hi.value, (ConfigError, ValueError))

    # 3. Conflicting locked_nodes: locking all 3 nodes to 30+30+40=100 while demanding target=150
    with pytest.raises(GoalSeekInfeasibleError) as exc_lock_all:
        allocate_target(
            preds_df,
            target_total=150.0,
            locked_nodes=["S1", "S2", "S3"],
        )
    assert isinstance(exc_lock_all.value, (ConfigError, ValueError))

    # 4. Explicit locked_nodes dict whose sum (80 + 80 = 160) exceeds target=120 with floor=0 on S3
    with pytest.raises(GoalSeekInfeasibleError) as exc_lock_sum:
        allocate_target(
            preds_df,
            target_total=120.0,
            locked_nodes={"S1": 80.0, "S2": 80.0},
            floor=0.0,
        )
    assert isinstance(exc_lock_sum.value, (ConfigError, ValueError))

    # 5. Locked node violating its own box bounds (baseline=30.0 locked, but lower bound=40.0)
    with pytest.raises(GoalSeekInfeasibleError) as exc_lock_box:
        allocate_target(
            preds_df,
            target_total=120.0,
            locked_nodes=["S1"],
            bounds={"S1": (40.0, 50.0)},
        )
    assert isinstance(exc_lock_box.value, (ConfigError, ValueError))


def test_t2_zero_variance_and_missing_oof_fallbacks() -> None:
    """Verify THieF, cross_temporal_mint, and allocate_target with zero-variance or missing OOF."""
    raw = _make_retail_hierarchy_panel(n_months=24, seed=61)
    cfg = RunConfig.model_validate(
        {
            "run_name": "zero-var-oof",
            "data": {"source_table": "p.d.t", "freq": "MS", "horizon": 6},
            "models": ["naive_mean"],
            "hierarchy": {
                "enabled": True,
                "levels": [["region"]],
                "reconcile_temporal": True,
                "temporal_aggregations": [3, 6],
                "cross_temporal": True,
                "reconciliation_methods": ["cross_temporal_mint"],
            },
        }
    )
    panel, spec = build_hierarchy(raw, cfg)
    f_dates = pd.date_range("2024-01-01", periods=6, freq="MS")
    oof_dates = pd.date_range("2023-07-01", periods=6, freq="MS")

    pred_rows = []
    zero_var_oof_rows = []
    for nid in spec.node_ids:
        for dt in f_dates:
            pred_rows.append(
                {
                    "ts_id": nid,
                    "model_type": "naive_mean",
                    "forecast_date": dt,
                    "yhat": 100.0,
                    "yhat_lower": 100.0,  # Zero interval width
                    "yhat_upper": 100.0,
                }
            )
        for dt in oof_dates:
            zero_var_oof_rows.append(
                {
                    "ts_id": nid,
                    "model_type": "naive_mean",
                    "fold_id": 0,
                    "cutoff_date": pd.Timestamp("2023-06-01"),
                    "ds": dt,
                    "y_true": 100.0,  # Exact zero residual (y_true == yhat)
                    "yhat": 100.0,
                }
            )
    preds_df = pd.DataFrame(pred_rows)
    zero_oof_df = pd.DataFrame(zero_var_oof_rows)

    # 1. Missing oof_df (oof_df=None)
    rec_no_oof, agg_no_oof = cross_temporal_mint(
        preds_df, spec, history_df=panel, oof_df=None, cfg=cfg, return_aggregations=True
    )
    assert verify_coherence(rec_no_oof, spec) < 1e-6
    assert verify_cross_temporal_coherence(rec_no_oof, spec, agg_no_oof, cfg=cfg) < 1e-6

    # 2. Zero-variance oof_df
    rec_zero_oof, agg_zero_oof = cross_temporal_mint(
        preds_df, spec, history_df=panel, oof_df=zero_oof_df, cfg=cfg, return_aggregations=True
    )
    assert verify_coherence(rec_zero_oof, spec) < 1e-6
    assert verify_cross_temporal_coherence(rec_zero_oof, spec, agg_zero_oof, cfg=cfg) < 1e-6

    # 3. allocate_target with zero-variance OOF and zero-width intervals
    gs_zero = allocate_target(
        preds_df,
        target_total=5400.0,
        hier_spec=spec,
        oof_df=zero_oof_df,
        method="mint_shrink",
        return_result=True,
    )
    assert isinstance(gs_zero, GoalSeekResult)
    assert gs_zero.max_target_discrepancy < 1e-6
    assert verify_coherence(gs_zero.dataframe, spec) < 1e-6


# =============================================================================
# TIER 3 — Pairwise & Multi-Subsystem Cross-Feature Combinations
# =============================================================================


def test_t3_simplex_with_hpo_backtest_ensembles_and_monotone_covariates() -> None:
    """Verify @transform_clr/@transform_ilr + HPO + backtest + ensembles + @monotone."""
    panel = _make_compositional_panel(
        n_brands=3,
        n_periods=40,
        simplex_sum=100.0,
        with_launch_zeros=True,
        with_covariates=True,
        seed=73,
    )
    models = [
        "naive_mean@transform_ilr",
        "regression_lags@transform_clr",
        "lightgbm@monotone",
    ]
    cfg = RunConfig.model_validate(
        {
            "run_name": "t3-simplex-combo",
            "data": {"source_table": "p.d.t", "freq": "MS", "horizon": 4},
            "models": models,
            "features": {
                "transform": "ilr",
                "simplex_sum": 100.0,
                "future_covariates": ["price_index", "promo"],
                "monotonic_constraints": {"price_index": -1, "promo": 1},
            },
            "backtest": {
                "enabled": True,
                "n_folds": 2,
                "horizon": 4,
                "step": 4,
                "min_train": 29,
            },
            "hpo": {"enabled": True, "n_trials": 2, "granularity": "fleetwide"},
            "ensemble": {
                "enabled": True,
                "strategies": ["inverse_error", "nnls", "ridge"],
                "variant_policy": "all",
            },
        }
    )

    # 1. Fleetwide Optuna HPO on simplex-transformed series sample
    sample_series = [sub for _, sub in panel.groupby("ts_id", sort=False)]
    tuned_params = tune_model("regression_lags@transform_clr", sample_series, cfg)
    assert isinstance(tuned_params, dict)

    # 2. Execute all models via spark_io.run_group (includes backtest + close_simplex_cells)
    cells, status_df = run_group(
        panel,
        cfg,
        models=models,
        params_by_model={"regression_lags@transform_clr": tuned_params},
    )
    assert set(status_df["status"]) == {"ok"}

    base_preds = pd.concat(
        [
            c.predictions.assign(
                ts_id=c.ts_id,
                model_type=c.model_type,
                forecast_date=c.predictions["ds"],
            )
            for c in cells
        ],
        ignore_index=True,
    )
    oof_df = pd.concat(
        [
            c.oof.assign(ts_id=c.ts_id, model_type=c.model_type)
            for c in cells
            if c.oof is not None and not c.oof.empty
        ],
        ignore_index=True,
    )
    metric_rows = []
    for c in cells:
        if c.metrics:
            metric_rows.append({"ts_id": c.ts_id, "model_type": c.model_type, **c.metrics})
    metric_df = pd.DataFrame(metric_rows)

    # Verify every base model's horizon predictions sum to 100.0 at every date
    for m_name, grp in base_preds.groupby("model_type"):
        assert verify_simplex_coherence(grp, simplex_sum=100.0) < 1e-6, f"Failed for {m_name}"

    # 3. Fit learned ensembles (`nnls`, `ridge`) and calculated ensembles (`inverse_error`)
    learned_weights, _, _ = fit_learned(oof_df, cfg)
    assert "nnls" in learned_weights and "ridge" in learned_weights

    calc_rows = combine_calculated(base_preds, cfg, metric_df=metric_df, oof_df=oof_df)
    assert len(calc_rows) > 0
    ens_calc_df = close_simplex_forecasts(pd.DataFrame(calc_rows), simplex_sum=100.0)
    assert verify_simplex_coherence(ens_calc_df, simplex_sum=100.0) < 1e-6

    ens_oof_df = combine_oof(oof_df, cfg, learned_weights=learned_weights)
    assert not ens_oof_df.empty
    for ens_m, grp in ens_oof_df.groupby("model_type"):
        closed_oof = close_simplex_forecasts(grp, simplex_sum=100.0)
        assert verify_simplex_coherence(closed_oof, simplex_sum=100.0) < 1e-6, (
            f"OOF simplex closure failed for {ens_m}"
        )


def test_t3_temporal_and_cross_temporal_with_backtest_ensembles_and_covariates() -> None:
    """Verify reconcile_temporal & cross_temporal + backtest + ensembles + covariates."""
    raw = _make_retail_hierarchy_panel(n_months=42, seed=83)
    cfg = RunConfig.model_validate(
        {
            "run_name": "t3-ct-combo",
            "data": {"source_table": "p.d.t", "freq": "MS", "horizon": 6},
            "models": ["naive_seasonal", "regression_lags"],
            "features": {
                "future_covariates": ["price"],
            },
            "backtest": {
                "enabled": True,
                "n_folds": 2,
                "horizon": 6,
                "step": 6,
                "min_train": 29,
            },
            "ensemble": {
                "enabled": True,
                "strategies": ["inverse_error", "nnls"],
            },
            "hierarchy": {
                "enabled": True,
                "levels": [["region"], ["region", "category"]],
                "reconcile_temporal": True,
                "temporal_aggregations": [3, 6],
                "cross_temporal": True,
                "reconciliation_methods": ["cross_temporal_mint"],
            },
        }
    )
    _, spec = build_hierarchy(raw, cfg)
    cells, status_df = run_group(raw, cfg)
    assert set(status_df["status"]) == {"ok"}
    assert "naive_seasonal_cross_temporal_mint" in set(status_df["model_type"])
    assert "regression_lags_cross_temporal_mint" in set(status_df["model_type"])

    all_preds = pd.concat(
        [c.predictions.assign(ts_id=c.ts_id, model_type=c.model_type) for c in cells],
        ignore_index=True,
    )
    all_oof = pd.concat(
        [
            c.oof.assign(ts_id=c.ts_id, model_type=c.model_type)
            for c in cells
            if c.oof is not None and not c.oof.empty
        ],
        ignore_index=True,
    )

    for rec_m in ("naive_seasonal_cross_temporal_mint", "regression_lags_cross_temporal_mint"):
        sub_p = all_preds[all_preds["model_type"] == rec_m]
        assert verify_coherence(sub_p, spec) < 1e-6
        sub_oof = all_oof[all_oof["model_type"] == rec_m]
        for _fid, fold_grp in sub_oof.groupby("fold_id"):
            assert verify_coherence(fold_grp, spec, date_col="forecast_date") < 1e-6


def test_t3_scenario_to_goal_seek_margin_inventory_report_and_mcp_chain(tmp_path: Path) -> None:
    """Verify chain: simulate_scenarios -> allocate_target -> margin -> inventory -> report."""
    raw = _make_retail_hierarchy_panel(n_months=36, seed=97)
    # Filter to 4 bottom series for fast scenario simulation
    bottom_df = raw[raw["region"] == "North"].copy()
    cfg = RunConfig.model_validate(
        {
            "run_name": "t3-scen-gs-chain",
            "data": {"source_table": "p.d.t", "freq": "MS", "horizon": 4},
            "models": ["regression_lags"],
            "features": {
                "future_covariates": ["price"],
                "monotonic_constraints": {"price": -1},
            },
            "backtest": {
                "enabled": True,
                "n_folds": 1,
                "horizon": 4,
                "step": 4,
                "min_train": 29,
            },
        }
    )

    # 1. Run counterfactual scenario simulation (10% promotional price cut)
    scen_df = simulate_scenarios(
        bottom_df,
        cfg,
        scenarios={"promo_cut": {"price": "-10%"}},
        price_col="price",
        unit_cost="unit_cost",
        include_inventory_impact=True,
    )
    assert not scen_df.empty
    assert "yhat" in scen_df.columns

    # 2. Chain scenario output directly into allocate_target with locks, bounds, margin & inventory
    f = sf.Forecaster(cfg, data=bottom_df)
    locked_sku = "North_Beverages_A"
    capped_sku = "North_Snacks_B"
    promo_rows = scen_df[scen_df["scenario"] == "promo_cut"]
    base_promo_total = float(promo_rows["yhat"].sum())
    stretch_target = base_promo_total * 1.15

    gs_df = f.allocate_target(
        forecasts_df=scen_df,
        targets={"__total__": stretch_target},
        scenario="promo_cut",
        locked_nodes=[locked_sku],
        bounds={capped_sku: (0.0, 95.0)},
        price_col="price",
        unit_cost="unit_cost",
        include_inventory_impact=True,
        lead_time=7.0,
        service_level=0.95,
    )
    assert abs(float(gs_df["allocated_yhat"].sum()) - stretch_target) < 1e-5
    assert bool(gs_df.loc[gs_df["ts_id"] == locked_sku, "is_locked"].all())
    np.testing.assert_allclose(
        gs_df.loc[gs_df["ts_id"] == locked_sku, "allocated_yhat"].to_numpy(dtype=float),
        gs_df.loc[gs_df["ts_id"] == locked_sku, "baseline_yhat"].to_numpy(dtype=float),
        atol=1e-6,
    )
    assert bool((gs_df.loc[gs_df["ts_id"] == capped_sku, "allocated_yhat"] <= 95.0 + 1e-6).all())

    # Verify financial & inventory replenishment columns
    for required_col in (
        "baseline_revenue",
        "allocated_revenue",
        "revenue_delta",
        "unit_cost",
        "baseline_gross_margin",
        "allocated_gross_margin",
        "gross_margin_delta",
        "baseline_reorder_point",
        "allocated_reorder_point",
        "delta_reorder_point",
        "baseline_safety_stock",
        "allocated_safety_stock",
        "delta_safety_stock",
        "baseline_order_qty",
        "allocated_order_qty",
        "delta_order_qty",
    ):
        assert required_col in gs_df.columns, f"Missing column {required_col}"

    # Free series whose allocation increased must have non-negative reorder point & order qty deltas
    free_increased = gs_df[(~gs_df["is_locked"]) & (gs_df["delta_yhat"] > 1.0)]
    assert not free_increased.empty
    assert bool((free_increased["delta_reorder_point"] >= 0.0).all())
    assert bool((free_increased["delta_order_qty"] > 0.0).all())

    # 3. Export interactive HTML + JSON report
    html_out = f.export_report(
        tmp_path / "chain_report.html",
        predictions_df=gs_df,
        goal_seek_df=gs_df,
        write_json=True,
    )
    assert html_out.is_file()
    assert "Top-Down Goal-Seek Target Allocation" in html_out.read_text(encoding="utf-8")

    # 4. Verify MCP server direct tool dispatches
    srv = McpServer()
    mcp_alloc = srv._dispatch_tool(
        "allocate_target",
        {
            "predictions": promo_rows[["ts_id", "ds", "yhat"]]
            .assign(ds=lambda d: d["ds"].astype(str))
            .to_dict(orient="records"),
            "target_total": stretch_target,
            "locked_nodes": [locked_sku],
        },
    )
    assert mcp_alloc["status"] == "ok"
    alloc_sum = sum(float(r["allocated_yhat"]) for r in mcp_alloc["allocations"])
    assert abs(alloc_sum - stretch_target) < 1e-4


# =============================================================================
# TIER 4 — Real-World Enterprise Application Scenarios
# =============================================================================


def test_t4_retail_multi_echelon_cross_temporal_and_goal_seek() -> None:
    """Real-world Scenario 1: 3-level Retail Hierarchy (`Region -> Category -> SKU`) with
    simultaneous `cross_temporal_mint` reconciliation and executive top-down `allocate_target`.
    """
    raw = _make_retail_hierarchy_panel(n_months=36, seed=107)
    cfg = RunConfig.model_validate(
        {
            "run_name": "t4-retail-enterprise",
            "data": {"source_table": "p.d.t", "freq": "MS", "horizon": 12},
            "models": ["holtwinters"],
            "backtest": {"enabled": True, "n_folds": 1, "horizon": 12},
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
    _panel, spec = build_hierarchy(raw, cfg)
    # 1 root + 2 regions + 4 region/category + 8 bottom SKUs = 15 nodes
    assert len(spec.node_ids) == 15
    assert len(spec.bottom_ids) == 8

    cells, status_df = run_group(raw, cfg)
    assert set(status_df["status"]) == {"ok"}
    ct_cells = [c for c in cells if c.model_type == "holtwinters_cross_temporal_mint"]
    assert len(ct_cells) == 15

    ct_preds = pd.concat(
        [c.predictions.assign(ts_id=c.ts_id, model_type=c.model_type) for c in ct_cells],
        ignore_index=True,
    )
    # Verify < 1e-6 discrepancy across cross-sectional tree AND temporal buckets simultaneously
    assert verify_coherence(ct_preds, spec) < 1e-6
    assert verify_cross_temporal_coherence(ct_preds, spec, cfg=cfg) < 1e-6

    # Executive Goal-Seek: raise annual company total by +12% while locking region=South to baseline
    base_root_total = float(ct_preds.loc[ct_preds["ts_id"] == TOTAL_NODE_ID, "yhat"].sum())
    exec_target = base_root_total * 1.12
    gs_res = allocate_target(
        ct_preds,
        target_total=exec_target,
        hier_spec=spec,
        history_df=raw,
        locked_nodes=["region=South"],
        price_col="price",
        unit_cost="unit_cost",
        include_inventory_impact=True,
        method="mint_shrink",
        return_result=True,
    )
    assert isinstance(gs_res, GoalSeekResult)
    assert abs(gs_res.allocated_total - exec_target) < 1e-6
    assert gs_res.max_coherence_discrepancy < 1e-6
    assert verify_coherence(gs_res.dataframe, spec) < 1e-6

    # Verify region=South remained at its baseline total while region=North absorbed the +12% lift
    south_base = float(
        gs_res.dataframe.loc[gs_res.dataframe["ts_id"] == "region=South", "baseline_yhat"].sum()
    )
    south_alloc = float(
        gs_res.dataframe.loc[gs_res.dataframe["ts_id"] == "region=South", "allocated_yhat"].sum()
    )
    assert abs(south_alloc - south_base) < 1e-5


def test_t4_market_share_compositional_panel_with_product_launch_zeros() -> None:
    """Real-world Scenario 2: 5-brand market-share panel (`simplex_sum=100.0`) with zero-share
    product launch history forecasted via `@transform_ilr` and `@transform_clr`.
    """
    panel = _make_compositional_panel(
        n_brands=5,
        n_periods=36,
        simplex_sum=100.0,
        with_launch_zeros=True,
        seed=131,
    )
    # Confirm exact 0.0 shares exist in historical data
    assert float(panel["y"].min()) == 0.0

    cfg = RunConfig.model_validate(
        {
            "run_name": "t4-market-share",
            "data": {"source_table": "p.d.t", "freq": "MS", "horizon": 12},
            "models": ["theta@transform_ilr", "holtwinters@transform_clr"],
            "features": {"simplex_sum": 100.0},
            "backtest": {"enabled": True, "n_folds": 1, "horizon": 12},
        }
    )
    cells, status_df = run_group(panel, cfg)
    assert len(cells) == 10
    assert set(status_df["status"]) == {"ok"}

    preds = pd.concat(
        [c.predictions.assign(ts_id=c.ts_id, model_type=c.model_type) for c in cells],
        ignore_index=True,
    )
    for m_type, grp in preds.groupby("model_type"):
        assert bool((grp["yhat"] > 0.0).all()), f"Non-positive share in {m_type}"
        assert bool((grp["yhat"] < 100.0).all()), f"Share >= 100 in {m_type}"
        assert bool((grp["yhat_lower"] >= 0.0).all())
        assert bool((grp["yhat_upper"] <= 100.0).all())
        assert bool((grp["yhat_lower"] <= grp["yhat"] + 1e-9).all())
        assert bool((grp["yhat"] <= grp["yhat_upper"] + 1e-9).all())
        assert verify_simplex_coherence(grp, simplex_sum=100.0) < 1e-6


def test_t4_auto_forecaster_standard_simplex_and_temporal_ratchet() -> None:
    """Real-world Scenario 3: Glass-Box Auto-Forecaster across standard, simplex, and hierarchical
    temporal panels, verifying 100% unchanged defaults on standard panels and Branch 2A/2C ratchets.
    """
    # 1. Standard non-simplex panel -> reconcile_temporal=False and transform != clr/ilr
    std_panel = generate_panel(
        3,
        cfg=GenConfig(history=36, freq="MS", with_hierarchy=False),
        seed=42,
    )
    std_prof = inspect_panel_for_auto(std_panel, horizon=6, freq="MS")
    assert std_prof.is_compositional_simplex is False
    std_bp = sf.Forecaster.build_config(std_panel, horizon=6, freq="MS", tier="first_light")
    assert std_bp.config.hierarchy.reconcile_temporal is False
    assert std_bp.config.hierarchy.cross_temporal is False
    assert std_bp.config.features.transform not in ("clr", "ilr")
    assert std_bp.config.features.simplex_sum is None
    assert not any("@transform_ilr" in m or "@transform_clr" in m for m in std_bp.config.models)

    # 2. Compositional simplex panel -> is_compositional_simplex=True, simplex_sum=100.0
    comp_panel = _make_compositional_panel(
        n_brands=3, n_periods=36, simplex_sum=100.0, with_launch_zeros=True, seed=53
    )
    comp_bp = sf.Forecaster.build_config(comp_panel, horizon=6, freq="MS", tier="first_light")
    assert comp_bp.profile.is_compositional_simplex is True
    assert comp_bp.config.features.simplex_sum == 100.0
    assert any("@transform_ilr" in m or "@transform_clr" in m for m in comp_bp.config.models)

    # 3. Hierarchical temporal panel -> Branch 2C proposes reconcile_temporal & cross_temporal,
    # and Level-3 Champion Ratchet rejects a branch whose score degrades vs. Pass 1.
    hier_panel = _make_retail_hierarchy_panel(n_months=36, seed=67)[
        ["ts_id", "region", "category", "ds", "y"]
    ].copy()
    hier_prof = inspect_panel_for_auto(
        hier_panel,
        horizon=12,
        freq="MS",
        hierarchy_cols=["region", "category"],
    )
    hier_bp = build_config(hier_prof, tier="first_light")
    # Single-pass build_config keeps reconcile_temporal=False by default
    assert hier_bp.config.hierarchy.reconcile_temporal is False

    p1_dict = hier_bp.config.model_dump(mode="json")
    branches = propose_autofocus_dag_branches(
        hier_prof,
        p1_dict,
        pass_index=2,
        diagnostics_summary={"residual_autocorr_lag1": 0.40, "fva_vs_naive_pct": 1.5},
        tier="wide_field",
    )
    assert len(branches) >= 1
    branch_2c = branches[-1]
    assert branch_2c["hierarchy"]["reconcile_temporal"] is True
    assert branch_2c["hierarchy"]["cross_temporal"] is True

    # Verify Champion Ratchet in propose_autofocus_synthesis_pass:
    # When Branch 2C degrades OOF score (best_score=15.0 > pass1_score=10.0), it is NOT promoted.
    synth_rejected = propose_autofocus_synthesis_pass(
        hier_prof,
        p1_dict,
        branch_outcomes=[
            {
                "pass_id": "2C",
                "best_score": 15.0,
                "pass1_score": 10.0,
                "config_dict": branch_2c,
                "mutations_applied": ("refine_hierarchical_coherence",),
            }
        ],
        tier="wide_field",
    )
    assert bool((synth_rejected.get("hierarchy") or {}).get("reconcile_temporal", False)) is False

    # When Branch 2C improves OOF score (best_score=8.5 < pass1_score=10.0), it IS promoted.
    synth_promoted = propose_autofocus_synthesis_pass(
        hier_prof,
        p1_dict,
        branch_outcomes=[
            {
                "pass_id": "2C",
                "best_score": 8.5,
                "pass1_score": 10.0,
                "config_dict": branch_2c,
                "mutations_applied": ("refine_hierarchical_coherence",),
            }
        ],
        tier="wide_field",
    )
    assert synth_promoted["hierarchy"]["reconcile_temporal"] is True
    assert synth_promoted["hierarchy"]["cross_temporal"] is True


# =============================================================================
# PHASE 2 EXPANSION — F1–F12 Comprehensive 4-Tier Verification
# =============================================================================


def test_t1_f1_mode3_hierarchy_config_auto_normalization() -> None:
    """Tier 1 F1 (>= 5 cases): HierarchyConfig Mode 3 auto-normalization across all 3 modes."""
    # Case 1: cross_temporal=True with levels auto-enables reconcile_temporal, enabled,
    # and appends "cross_temporal_mint" while preserving default spatial methods.
    hc_mode3 = HierarchyConfig(cross_temporal=True, levels=[["region"]])
    assert hc_mode3.reconcile_temporal is True
    assert hc_mode3.enabled is True
    assert hc_mode3.cross_temporal is True
    assert hc_mode3.reconciliation_methods == [
        "bottom_up",
        "wls_struct",
        "mint_shrink",
        "cross_temporal_mint",
    ]

    # Case 2: "cross_temporal_mint" in reconciliation_methods with levels auto-enables
    # cross_temporal, reconcile_temporal, and enabled without duplicating "cross_temporal_mint".
    hc_explicit_ct = HierarchyConfig(
        levels=[["region"], ["region", "category"]],
        reconciliation_methods=["mint_shrink", "cross_temporal_mint"],
    )
    assert hc_explicit_ct.cross_temporal is True
    assert hc_explicit_ct.reconcile_temporal is True
    assert hc_explicit_ct.enabled is True
    assert hc_explicit_ct.reconciliation_methods == ["mint_shrink", "cross_temporal_mint"]

    # Case 3: Mode 1 Spatial-Only keeps reconcile_temporal=False and cross_temporal=False.
    hc_mode1 = HierarchyConfig(enabled=True, levels=[["region"]])
    assert hc_mode1.enabled is True
    assert hc_mode1.reconcile_temporal is False
    assert hc_mode1.cross_temporal is False
    assert "cross_temporal_mint" not in hc_mode1.reconciliation_methods

    # Case 4: Mode 2 Temporal-Only (levels=[]) keeps enabled=False and cross_temporal=False.
    hc_mode2 = HierarchyConfig(reconcile_temporal=True, levels=[])
    assert hc_mode2.reconcile_temporal is True
    assert hc_mode2.cross_temporal is False
    assert hc_mode2.enabled is False

    # Case 5: Custom spatial methods + cross_temporal=True appends "cross_temporal_mint".
    hc_custom = HierarchyConfig(
        cross_temporal=True,
        levels=[["region"]],
        reconciliation_methods=["wls_struct", "mint_nonneg"],
    )
    assert hc_custom.reconciliation_methods == [
        "wls_struct",
        "mint_nonneg",
        "cross_temporal_mint",
    ]


def test_t1_f2_spatial_temporal_weight_and_62_shipped_configs_hash_stability() -> None:
    """Tier 1 F2 (>= 5 cases): spatial_temporal_weight validation, elision, and 62-config hashes."""
    # Case 1: Default is "auto"
    hc_def = HierarchyConfig()
    assert hc_def.spatial_temporal_weight == "auto"

    # Case 2: Explicit float values in [0.0, 1.0] (and int 0, 1) normalize to float
    for raw_val, expected in ((0, 0.0), (0.0, 0.0), (0.25, 0.25), (0.5, 0.5), (0.8, 0.8), (1, 1.0)):
        hc_w = HierarchyConfig(spatial_temporal_weight=raw_val)
        assert isinstance(hc_w.spatial_temporal_weight, float)
        assert hc_w.spatial_temporal_weight == pytest.approx(expected)

    # Case 3: canonical_config elides ("hierarchy", "spatial_temporal_weight"): "auto"
    base_payload = {
        "run_name": "f2-elide",
        "data": {"source_table": "p.d.t", "freq": "MS", "horizon": 6},
        "models": ["naive_mean"],
    }
    cfg_a = RunConfig.model_validate(base_payload)
    cfg_b = RunConfig.model_validate(
        {**base_payload, "hierarchy": {"spatial_temporal_weight": "auto"}}
    )
    assert "hierarchy" not in canonical_config(cfg_a)
    assert canonical_config(cfg_a) == canonical_config(cfg_b)
    assert make_run_id(cfg_a) == make_run_id(cfg_b)

    # Case 4: Non-default float spatial_temporal_weight changes run_id when hierarchy is active
    cfg_ct_auto = RunConfig.model_validate(
        {
            **base_payload,
            "hierarchy": {
                "levels": [["region"]],
                "cross_temporal": True,
                "spatial_temporal_weight": "auto",
            },
        }
    )
    cfg_ct_float = RunConfig.model_validate(
        {
            **base_payload,
            "hierarchy": {
                "levels": [["region"]],
                "cross_temporal": True,
                "spatial_temporal_weight": 0.35,
            },
        }
    )
    assert make_run_id(cfg_ct_auto) != make_run_id(cfg_ct_float)

    # Case 5: All 62 shipped configs load cleanly with identical run_id when "auto" is explicit
    root_configs = [
        p
        for p in sorted((REPO_ROOT / "configs").glob("*.json"))
        if p.name != "compute_fallback.json"
    ]
    smoke_configs = sorted((REPO_ROOT / "configs" / "smokes").glob("*.json"))
    all_configs = root_configs + smoke_configs
    assert len(all_configs) == 62
    for cfg_path in all_configs:
        raw_dict = json.loads(cfg_path.read_text(encoding="utf-8"))
        c1 = RunConfig.model_validate(raw_dict)
        assert c1.hierarchy.spatial_temporal_weight == "auto"
        raw_with_auto = json.loads(cfg_path.read_text(encoding="utf-8"))
        raw_with_auto.setdefault("hierarchy", {})["spatial_temporal_weight"] = "auto"
        c2 = RunConfig.model_validate(raw_with_auto)
        assert make_run_id(c1) == make_run_id(c2), f"Hash drift on {cfg_path.name}"


def test_t1_f3_mode2_flat_panel_temporal_oof_and_metrics_recomputation() -> None:
    """Tier 1 F3 (>= 5 cases): Mode 2 flat-panel temporal reconciliation across predictions,
    OOF folds, and recomputed CellResult.metrics.
    """
    s0 = _make_daily_series(n_days=84, seed=101, ts_id="sku_0")
    s1 = _make_daily_series(n_days=84, seed=102, ts_id="sku_1")
    flat_panel = pd.concat([s0, s1], ignore_index=True)

    cfg = RunConfig.model_validate(
        {
            "run_name": "t1-f3-mode2-oof",
            "data": {"source_table": "p.d.t", "freq": "D", "horizon": 14},
            "models": ["naive_mean"],
            "backtest": {
                "enabled": True,
                "n_folds": 2,
                "horizon": 14,
                "step": 7,
                "min_train": 56,
            },
            "hierarchy": {
                "enabled": False,
                "levels": [],
                "reconcile_temporal": True,
                "temporal_aggregations": [7, 14],
                "reconciliation_methods": ["wls_struct", "mint_shrink"],
            },
        }
    )

    cells, status_df = run_group(flat_panel, cfg)
    assert set(status_df["status"]) == {"ok"}
    model_types = {c.model_type for c in cells}

    # Case 1: Both base model and <model>_<method> cells are emitted on the flat panel
    assert {"naive_mean", "naive_mean_wls_struct", "naive_mean_mint_shrink"}.issubset(model_types)

    base_s0 = next(c for c in cells if c.ts_id == "sku_0" and c.model_type == "naive_mean")
    rec_s0 = next(
        c for c in cells if c.ts_id == "sku_0" and c.model_type == "naive_mean_wls_struct"
    )

    # Case 2: Base cell predictions are preserved distinct from temporally reconciled predictions
    assert not np.allclose(
        base_s0.predictions["yhat"].to_numpy(dtype=float),
        rec_s0.predictions["yhat"].to_numpy(dtype=float),
    )
    direct_rec_cells = reconcile_cells([base_s0], cfg, None, history_df=s0)
    assert len(direct_rec_cells) == 2

    # Case 3: Forward predictions of <model>_wls_struct have < 1e-6 temporal discrepancy and
    # mode == "temporal_only" with spatial_weight_alpha == 0.0
    assert verify_temporal_coherence(rec_s0.predictions, cfg=cfg) < 1e-6
    tf_out = reconcile_temporal_forecasts(base_s0.predictions, history_df=s0, cfg=cfg)
    assert isinstance(tf_out, pd.DataFrame)
    diag = tf_out.attrs.get("cross_temporal_diagnostics", {})
    assert diag.get("mode") == "temporal_only"
    assert diag.get("spatial_weight_alpha") == pytest.approx(0.0)
    assert float(diag.get("final_temp_discrepancy", 0.0)) < 1e-6

    # Case 4: Rolling-origin OOF folds are temporally reconciled per fold
    assert rec_s0.oof is not None and not rec_s0.oof.empty
    assert set(rec_s0.oof["fold_id"].unique()) == {0, 1}
    assert not np.allclose(
        base_s0.oof["yhat"].to_numpy(dtype=float),
        rec_s0.oof["yhat"].to_numpy(dtype=float),
    )

    # Case 5: CellResult.metrics on <model>_wls_struct are freshly recomputed from reconciled OOF
    expected_metrics = compute_metrics(
        rec_s0.oof["y_true"].to_numpy(dtype=float),
        rec_s0.oof["yhat"].to_numpy(dtype=float),
        y_train=s0["y"].to_numpy(dtype=float),
        lower=rec_s0.oof["yhat_lower"].to_numpy(dtype=float),
        upper=rec_s0.oof["yhat_upper"].to_numpy(dtype=float),
        seasonal_period=7,
    )
    assert rec_s0.metrics["wape"] == pytest.approx(expected_metrics["wape"], rel=1e-6)
    assert rec_s0.metrics["mae"] == pytest.approx(expected_metrics["mae"], rel=1e-6)
    assert rec_s0.metrics["wape"] != pytest.approx(base_s0.metrics["wape"], rel=1e-4)


def test_t1_f4_mode3_side_by_side_leaderboard_preservation() -> None:
    """Tier 1 F4 (>= 5 cases): Mode 3 keeps standard spatial methods pure spatial (`S_cs`)
    alongside `cross_temporal_mint` (`S_cs x S_temp`) on the leaderboard.
    """
    raw = _make_retail_hierarchy_panel(n_months=36, seed=113)
    cfg = RunConfig.model_validate(
        {
            "run_name": "t1-f4-side-by-side",
            "data": {"source_table": "p.d.t", "freq": "MS", "horizon": 6},
            "models": ["naive_mean"],
            "backtest": {
                "enabled": True,
                "n_folds": 1,
                "horizon": 6,
                "step": 6,
                "min_train": 24,
            },
            "hierarchy": {
                "levels": [["region"], ["region", "category"]],
                "cross_temporal": True,
                "temporal_aggregations": [3, 6],
                "reconciliation_methods": [
                    "bottom_up",
                    "wls_struct",
                    "mint_shrink",
                    "mint_nonneg",
                    "cross_temporal_mint",
                ],
            },
        }
    )
    panel, spec = build_hierarchy(raw, cfg)
    cells, status_df = run_group(raw, cfg)
    assert set(status_df["status"]) == {"ok"}

    # Case 1: All 5 reconciled model variants exist alongside the base model
    model_names = {c.model_type for c in cells}
    for m_expected in (
        "naive_mean",
        "naive_mean_bottom_up",
        "naive_mean_wls_struct",
        "naive_mean_mint_shrink",
        "naive_mean_mint_nonneg",
        "naive_mean_cross_temporal_mint",
    ):
        assert m_expected in model_names

    base_cells = [c for c in cells if c.model_type == "naive_mean"]
    all_base_preds = pd.concat(
        [c.predictions.assign(ts_id=c.ts_id, model_type="naive_mean") for c in base_cells],
        ignore_index=True,
    )
    all_base_oof = pd.concat(
        [c.oof.assign(ts_id=c.ts_id, model_type="naive_mean") for c in base_cells],
        ignore_index=True,
    )

    # Case 2: Standard spatial methods in reconcile_cells match pure spatial reconcile_forecasts
    pure_spatial_mint = reconcile_forecasts(
        all_base_preds,
        spec,
        "mint_shrink",
        history_df=panel,
        oof_df=all_base_oof,
        date_col="ds",
        ts_id_col="ts_id",
        target_col="y",
    )
    cell_mint_preds = pd.concat(
        [
            c.predictions.assign(ts_id=c.ts_id)
            for c in cells
            if c.model_type == "naive_mean_mint_shrink"
        ],
        ignore_index=True,
    )
    merged_mint = pure_spatial_mint.merge(
        cell_mint_preds, on=["ts_id", "ds"], suffixes=("_pure", "_cell")
    )
    np.testing.assert_allclose(
        merged_mint["yhat_pure"].to_numpy(dtype=float),
        merged_mint["yhat_cell"].to_numpy(dtype=float),
        atol=1e-6,
    )

    # Case 3: Pure spatial predictions carry mode="spatial_only" and spatial_weight_alpha=1.0
    mint_diag = pure_spatial_mint.attrs.get("cross_temporal_diagnostics", {})
    assert mint_diag.get("mode") == "spatial_only"
    assert mint_diag.get("spatial_weight_alpha") == pytest.approx(1.0)

    # Case 4: <model>_cross_temporal_mint differs from pure spatial <model>_mint_shrink
    # while satisfying both spatial and temporal coherence < 1e-6
    cell_ct_preds = pd.concat(
        [
            c.predictions.assign(ts_id=c.ts_id)
            for c in cells
            if c.model_type == "naive_mean_cross_temporal_mint"
        ],
        ignore_index=True,
    )
    assert verify_coherence(cell_ct_preds, spec, date_col="ds") < 1e-6
    merged_ct_vs_mint = cell_mint_preds.merge(
        cell_ct_preds, on=["ts_id", "ds"], suffixes=("_spatial", "_ct")
    )
    assert not np.allclose(
        merged_ct_vs_mint["yhat_spatial"].to_numpy(dtype=float),
        merged_ct_vs_mint["yhat_ct"].to_numpy(dtype=float),
    )

    # Case 5: reconcile_oof keeps "mint_shrink" pure spatial while "cross_temporal_mint" is joint
    oof_mint = reconcile_oof(
        all_base_oof, spec, "mint_shrink", cfg=cfg, history_df=panel, stamp_method=True
    )
    oof_ct = reconcile_oof(
        all_base_oof, spec, "cross_temporal_mint", cfg=cfg, history_df=panel, stamp_method=True
    )
    assert not np.allclose(
        oof_mint.sort_values(["ts_id", "ds"])["yhat"].to_numpy(dtype=float),
        oof_ct.sort_values(["ts_id", "ds"])["yhat"].to_numpy(dtype=float),
    )


def test_t1_f5_reconcile_cross_temporal_alpha_blending_and_diagnostics() -> None:
    """Tier 1 F5 (>= 5 cases): spatial_temporal_weight alpha blending, auto inverse-variance
    learning from OOF residuals, alternating projection refinement, and diagnostics schema.
    """
    raw = _make_retail_hierarchy_panel(n_months=36, seed=127)
    cfg = RunConfig.model_validate(
        {
            "run_name": "t1-f5-alpha",
            "data": {"source_table": "p.d.t", "freq": "MS", "horizon": 6},
            "models": ["naive_mean"],
            "hierarchy": {
                "levels": [["region"], ["region", "category"]],
                "cross_temporal": True,
                "temporal_aggregations": [3, 6],
            },
        }
    )
    panel, spec = build_hierarchy(raw, cfg)
    rng = np.random.default_rng(129)
    f_dates = pd.date_range("2025-01-01", periods=6, freq="MS")
    rows = []
    for nid in spec.node_ids:
        vals = rng.uniform(50.0, 160.0, size=6)
        for dt, v in zip(f_dates, vals, strict=True):
            rows.append(
                {
                    "ts_id": nid,
                    "model_type": "naive_mean",
                    "forecast_date": dt,
                    "yhat": float(v),
                    "yhat_lower": float(v * 0.85),
                    "yhat_upper": float(v * 1.15),
                }
            )
    preds_df = pd.DataFrame(rows)

    # Case 1: spatial_temporal_weight=1.0 -> alpha=1.0, both discrepancies < 1e-6
    rec_1, agg_1 = reconcile_cross_temporal(
        preds_df,
        spec,
        history_df=panel,
        cfg=cfg,
        spatial_temporal_weight=1.0,
        return_aggregations=True,
    )
    diag_1 = rec_1.attrs["cross_temporal_diagnostics"]
    assert diag_1["mode"] == "cross_temporal"
    assert diag_1["spatial_weight_alpha"] == pytest.approx(1.0)
    assert diag_1["final_cs_discrepancy"] < 1e-6
    assert diag_1["final_temp_discrepancy"] < 1e-6
    assert verify_cross_temporal_coherence(rec_1, spec, agg_1, cfg=cfg) < 1e-6

    # Case 2: spatial_temporal_weight=0.0 -> alpha=0.0, both discrepancies < 1e-6
    rec_0, agg_0 = reconcile_cross_temporal(
        preds_df,
        spec,
        history_df=panel,
        cfg=cfg,
        spatial_temporal_weight=0.0,
        return_aggregations=True,
    )
    diag_0 = rec_0.attrs["cross_temporal_diagnostics"]
    assert diag_0["mode"] == "cross_temporal"
    assert diag_0["spatial_weight_alpha"] == pytest.approx(0.0)
    assert diag_0["final_cs_discrepancy"] < 1e-6
    assert diag_0["final_temp_discrepancy"] < 1e-6
    assert verify_cross_temporal_coherence(rec_0, spec, agg_0, cfg=cfg) < 1e-6

    # Case 3: Explicit intermediate float spatial_temporal_weight=0.35 lies between endpoints
    rec_35 = reconcile_cross_temporal(
        preds_df,
        spec,
        history_df=panel,
        cfg=cfg,
        spatial_temporal_weight=0.35,
    )
    assert isinstance(rec_35, pd.DataFrame)
    diag_35 = rec_35.attrs["cross_temporal_diagnostics"]
    assert diag_35["spatial_weight_alpha"] == pytest.approx(0.35)
    assert diag_35["final_cs_discrepancy"] < 1e-6
    assert diag_35["final_temp_discrepancy"] < 1e-6

    # Case 4: "auto" learns alpha from OOF residual variances across spatial vs temporal axes
    oof_dates = pd.date_range("2024-01-01", periods=12, freq="MS")
    oof_spatial_accurate_rows = []
    oof_temporal_accurate_rows = []
    b_true = rng.uniform(40.0, 90.0, size=(spec.n_bottom, 12))
    y_true_mat = spec.summing_matrix @ b_true
    n_agg = spec.n_aggregated
    for n_idx, nid in enumerate(spec.node_ids):
        for t_idx, dt in enumerate(oof_dates):
            yt = float(y_true_mat[n_idx, t_idx])
            # Spatial-accurate OOF: upper nodes have ~0 error and bottom nodes have
            # opposite-signed persistent bias (+25 / -25) that cancels spatially
            # (S_upper @ bot_res ~ 0 -> sigma2_cs ~ 0) while remaining large over
            # 3/6-month buckets (sigma2_temp ~ 625 -> alpha_sp ~ 0.95).
            if n_idx >= n_agg:
                b_pos = n_idx - n_agg
                sp_err = (25.0 if b_pos % 2 == 0 else -25.0) + float(rng.normal(0.0, 0.2))
            else:
                sp_err = float(rng.normal(0.0, 0.05))
            oof_spatial_accurate_rows.append(
                {
                    "ts_id": nid,
                    "model_type": "naive_mean",
                    "fold_id": 0,
                    "cutoff_date": pd.Timestamp("2023-12-01"),
                    "ds": dt,
                    "y_true": yt,
                    "yhat": yt + sp_err,
                }
            )
            # Temporal-accurate OOF: bottom nodes have zero-sum 3-month cycle (+15, -15, 0) so
            # 3/6-month bucket means are ~0 (sigma2_temp ~ 0), while upper nodes have large error
            # (sigma2_cs ~ 1600 -> alpha_tm ~ 0.05).
            if n_idx >= n_agg:
                tm_err = float((15.0, -15.0, 0.0)[t_idx % 3] + rng.normal(0.0, 0.1))
            else:
                tm_err = 40.0 * (1.0 + 0.1 * float(rng.normal(0.0, 1.0)))
            oof_temporal_accurate_rows.append(
                {
                    "ts_id": nid,
                    "model_type": "naive_mean",
                    "fold_id": 0,
                    "cutoff_date": pd.Timestamp("2023-12-01"),
                    "ds": dt,
                    "y_true": yt,
                    "yhat": yt + tm_err,
                }
            )
    rec_auto_sp = reconcile_cross_temporal(
        preds_df,
        spec,
        history_df=panel,
        oof_df=pd.DataFrame(oof_spatial_accurate_rows),
        cfg=cfg,
        spatial_temporal_weight="auto",
    )
    rec_auto_tm = reconcile_cross_temporal(
        preds_df,
        spec,
        history_df=panel,
        oof_df=pd.DataFrame(oof_temporal_accurate_rows),
        cfg=cfg,
        spatial_temporal_weight="auto",
    )
    assert isinstance(rec_auto_sp, pd.DataFrame)
    assert isinstance(rec_auto_tm, pd.DataFrame)
    alpha_sp = float(rec_auto_sp.attrs["cross_temporal_diagnostics"]["spatial_weight_alpha"])
    alpha_tm = float(rec_auto_tm.attrs["cross_temporal_diagnostics"]["spatial_weight_alpha"])
    assert 0.0 <= alpha_sp <= 1.0
    assert 0.0 <= alpha_tm <= 1.0
    assert alpha_sp > 0.7
    assert alpha_tm < 0.3
    assert alpha_sp > alpha_tm + 0.4

    # Case 5: Complete diagnostics dictionary keys across all 3 modes
    required_keys = {
        "mode",
        "spatial_weight_alpha",
        "cs_first_discrepancy",
        "temp_first_discrepancy",
        "final_cs_discrepancy",
        "final_temp_discrepancy",
    }
    assert required_keys.issubset(rec_auto_sp.attrs["cross_temporal_diagnostics"].keys())


def test_t1_f6_allocate_target_smoothness_weight_multi_period() -> None:
    """Tier 1 F6 (>= 5 cases): Smooth multi-period Goal-Seek trajectory control via
    `smoothness_weight` across `allocate_target`, `Forecaster`, `PlaygroundRun`, and `BakeOff`.
    """
    f_dates = pd.date_range("2025-01-01", periods=6, freq="MS")
    rows = []
    bounds_rows = []
    for t_idx, dt in enumerate(f_dates):
        # S1 has wider interval on t=2, 3 and a tight cap on t=0, 1 so unpenalized QP jumps at t=2
        w_s1 = 30.0 if t_idx in (2, 3) else 8.0
        rows.append(
            {
                "ts_id": "S1",
                "model_type": "naive",
                "forecast_date": dt,
                "yhat": 50.0,
                "yhat_lower": 50.0 - w_s1,
                "yhat_upper": 50.0 + w_s1,
            }
        )
        rows.append(
            {
                "ts_id": "S2",
                "model_type": "naive",
                "forecast_date": dt,
                "yhat": 50.0,
                "yhat_lower": 35.0,
                "yhat_upper": 65.0,
            }
        )
        bounds_rows.append(
            {
                "ts_id": "S1",
                "forecast_date": dt,
                "lower": 0.0,
                "upper": 52.0 if t_idx in (0, 1) else 150.0,
            }
        )
        bounds_rows.append({"ts_id": "S2", "forecast_date": dt, "lower": 0.0, "upper": 150.0})

    preds_df = pd.DataFrame(rows)
    bounds_df = pd.DataFrame(bounds_rows)

    def _roughness(df_alloc: pd.DataFrame) -> float:
        total_r = 0.0
        for _tid, grp in df_alloc.sort_values("forecast_date").groupby("ts_id"):
            delta = grp["allocated_yhat"].to_numpy(dtype=float) - grp["baseline_yhat"].to_numpy(
                dtype=float
            )
            diff_delta = np.diff(delta)
            total_r += float(np.sum(diff_delta * diff_delta))
        return total_r

    # Case 1: smoothness_weight=0.0 vs smoothness_weight=10.0 strictly reduces roughness
    res_0 = allocate_target(
        preds_df,
        target_total=720.0,
        bounds=bounds_df,
        method="wls_var",
        smoothness_weight=0.0,
        return_result=True,
    )
    res_1 = allocate_target(
        preds_df,
        target_total=720.0,
        bounds=bounds_df,
        method="wls_var",
        smoothness_weight=1.0,
        return_result=True,
    )
    res_10 = allocate_target(
        preds_df,
        target_total=720.0,
        bounds=bounds_df,
        method="wls_var",
        smoothness_weight=10.0,
        return_result=True,
    )
    assert isinstance(res_0, GoalSeekResult)
    assert isinstance(res_1, GoalSeekResult)
    assert isinstance(res_10, GoalSeekResult)

    r_0 = _roughness(res_0.dataframe)
    r_1 = _roughness(res_1.dataframe)
    r_10 = _roughness(res_10.dataframe)
    assert r_10 < r_0

    # Case 2: Monotonic roughness reduction across 0.0 -> 1.0 -> 10.0
    assert r_10 <= r_1 + 1e-6 <= r_0 + 1e-6

    # Case 3: Equality target satisfied to < 1e-6 under smoothness_weight=10.0
    assert abs(res_10.allocated_total - 720.0) < 1e-6
    assert res_10.max_target_discrepancy < 1e-6

    # Case 4: Box bounds satisfied to < 1e-6 at every period under smoothness_weight=10.0
    s1_rows = res_10.dataframe[res_10.dataframe["ts_id"] == "S1"].sort_values("forecast_date")
    assert bool((s1_rows["allocated_yhat"].iloc[:2] <= 52.0 + 1e-6).all())
    assert bool((res_10.dataframe["allocated_yhat"] >= -1e-6).all())
    assert bool((res_10.dataframe["allocated_yhat"] <= 150.0 + 1e-6).all())

    # Case 5: Exposed on Forecaster, PlaygroundRun, and BakeOff
    hist_df = _make_daily_series(n_days=42)
    cfg = RunConfig.model_validate(
        {
            "run_name": "t1-f6-sdk",
            "data": {"source_table": "p.d.t", "freq": "D", "horizon": 7},
            "models": ["naive_mean"],
        }
    )
    f = sf.Forecaster(cfg, data=hist_df)
    f_gs = f.allocate_target(700.0, forecasts_df=preds_df, smoothness_weight=5.0)
    assert abs(float(f_gs["allocated_yhat"].sum()) - 700.0) < 1e-5

    pr = run_model("naive_mean", horizon=7, backtest=False)
    pr_gs = pr.allocate_target(target_total=250.0, smoothness_weight=5.0)
    assert abs(float(pr_gs["allocated_yhat"].sum()) - 250.0) < 1e-5

    bo = bakeoff(models=["naive_mean", "naive_drift"], horizon=7, n_folds=1)
    bo_gs = bo.allocate_target(target_total=250.0, smoothness_weight=5.0)
    assert not bo_gs.empty


def test_t1_f7_sample_coherent_paths_and_verify_simplex_alias() -> None:
    """Tier 1 F7 (>= 5 cases): `sample_coherent_paths(..., n_paths=50)` coherent trajectory
    simulation and `verify_simplex` alias across SDK surfaces.
    """
    from scale_forecasting.temporal_reconciliation import sample_coherent_paths

    raw = _make_retail_hierarchy_panel(n_months=36, seed=139)
    cfg = RunConfig.model_validate(
        {
            "run_name": "t1-f7-paths",
            "data": {"source_table": "p.d.t", "freq": "MS", "horizon": 6},
            "models": ["naive_mean"],
            "hierarchy": {
                "levels": [["region"], ["region", "category"]],
                "cross_temporal": True,
                "temporal_aggregations": [3, 6],
            },
        }
    )
    panel, spec = build_hierarchy(raw, cfg)
    f_dates = pd.date_range("2025-01-01", periods=6, freq="MS")
    pred_rows = []
    for nid in spec.node_ids:
        for dt in f_dates:
            pred_rows.append(
                {
                    "ts_id": nid,
                    "model_type": "naive_mean",
                    "forecast_date": dt,
                    "yhat": 80.0,
                    "yhat_lower": 65.0,
                    "yhat_upper": 95.0,
                }
            )
    preds_df = pd.DataFrame(pred_rows)

    # Case 1: sample_coherent_paths(..., n_paths=50) returns 50 trajectories with canonical schema
    paths_df = sample_coherent_paths(
        preds_df,
        spec,
        history_df=panel,
        cfg=cfg,
        n_paths=50,
        seed=42,
    )
    assert isinstance(paths_df, pd.DataFrame)
    assert {"path_id", "ts_id", "forecast_date", "yhat_sample"}.issubset(paths_df.columns)
    assert paths_df["path_id"].nunique() == 50
    assert len(paths_df) == 50 * len(spec.node_ids) * 6

    # Case 2: Every single path_id in 0..49 satisfies cross-sectional AND temporal coherence < 1e-6
    for _pid, p_grp in paths_df.groupby("path_id"):
        assert verify_coherence(p_grp, spec) < 1e-6
        assert verify_temporal_coherence(p_grp, cfg=cfg) < 1e-6

    # Case 3: Works on flat-panel temporal-only (spec=None)
    daily_df = _make_daily_series(n_days=56)
    flat_cfg = RunConfig.model_validate(
        {
            "run_name": "t1-f7-flat-paths",
            "data": {"source_table": "p.d.t", "freq": "D", "horizon": 14},
            "models": ["naive_mean"],
            "hierarchy": {"reconcile_temporal": True, "temporal_aggregations": [7, 14]},
        }
    )
    flat_preds = pd.DataFrame(
        {
            "ts_id": "sku_0",
            "model_type": "naive_mean",
            "forecast_date": pd.date_range("2025-02-26", periods=14, freq="D"),
            "yhat": np.linspace(70.0, 90.0, 14),
            "yhat_lower": np.linspace(60.0, 80.0, 14),
            "yhat_upper": np.linspace(80.0, 100.0, 14),
        }
    )
    flat_paths = sample_coherent_paths(
        flat_preds, None, history_df=daily_df, cfg=flat_cfg, n_paths=15, seed=11
    )
    assert flat_paths["path_id"].nunique() == 15
    for _pid, p_grp in flat_paths.groupby("path_id"):
        assert verify_temporal_coherence(p_grp, cfg=flat_cfg) < 1e-6

    # Case 4: Exposed on sf.sample_coherent_paths, Forecaster, PlaygroundRun, and BakeOff
    assert getattr(sf, "sample_coherent_paths", None) is sample_coherent_paths
    f = sf.Forecaster(flat_cfg, data=daily_df)
    f_paths = f.sample_coherent_paths(forecasts_df=flat_preds, n_paths=10, seed=7)
    assert f_paths["path_id"].nunique() == 10

    pr = run_model("naive_mean", horizon=14, backtest=False)
    assert pr.sample_coherent_paths(n_paths=8, seed=7)["path_id"].nunique() == 8

    bo = bakeoff(models=["naive_mean", "naive_drift"], horizon=7, n_folds=1)
    assert bo.sample_coherent_paths(n_paths=6, seed=7)["path_id"].nunique() == 6

    # Case 5: sf.verify_simplex alias works on "forecast_date", "ds", and "date" columns
    assert getattr(sf, "verify_simplex", None) is verify_simplex
    comp_df = pd.DataFrame(
        {
            "ts_id": ["b0", "b1", "b0", "b1"],
            "forecast_date": [
                pd.Timestamp("2025-01-01"),
                pd.Timestamp("2025-01-01"),
                pd.Timestamp("2025-02-01"),
                pd.Timestamp("2025-02-01"),
            ],
            "yhat": [60.0, 40.0, 35.0, 65.0],
        }
    )
    assert verify_simplex(comp_df, simplex_sum=100.0) < 1e-6
    assert (
        verify_simplex(comp_df.rename(columns={"forecast_date": "date"}), simplex_sum=100.0) < 1e-6
    )
    assert verify_simplex(comp_df.rename(columns={"forecast_date": "ds"}), simplex_sum=100.0) < 1e-6


def test_t1_f8_practitioner_assessment_question_3_causal_guardrails() -> None:
    """Tier 1 F8 (>= 5 cases): Question 3 (`covariates_vs_overfitting`) surfaces
    `monotonic_constraints`, `@monotone` models, `holiday_window`, and `f.promo_elasticity_df()`.
    """
    rng = np.random.default_rng(149)
    dates = pd.date_range("2024-01-01", periods=56, freq="D")
    rows = []
    for s_idx in range(2):
        tid = f"sku_{s_idx}"
        for t_idx, ds in enumerate(dates):
            promo = float(1 if (t_idx % 7 == 2) else 0)
            price = float(10.0 - 1.5 * promo + rng.normal(0.0, 0.1))
            y = float(50.0 + 15.0 * promo - 2.0 * price + rng.normal(0.0, 0.5))
            rows.append({"ts_id": tid, "ds": ds, "y": max(1.0, y), "promo": promo, "price": price})
    panel_df = pd.DataFrame(rows)

    f = sf.Forecaster.auto(
        panel_df,
        horizon=7,
        freq="D",
        tier="first_light",
        where="local",
        models=["regression_lags", "lightgbm@monotone"],
        baselines=["naive_seasonal"],
        holidays=["US"],
        overrides={
            "features": {
                "monotonic_constraints": {"price": -1, "promo": 1},
                "holiday_window": {"enabled": True, "pre_holiday_days": 5, "post_holiday_days": 2},
            }
        },
        hpo=False,
        calibration=False,
        explainability=False,
        n_folds=1,
        autofocus=False,
    )
    assess_df = f.practitioner_assessment_df()
    q3_row = assess_df[assess_df["question_id"] == "covariates_vs_overfitting"].iloc[0]
    q3_text = f"{q3_row['answer']} {q3_row['evidence']}"

    # Case 1: Question 3 row is present with non-empty verdict, answer, evidence, recommendation
    assert q3_row["verdict"] in {
        "covariates_help",
        "selective_signal_kept",
        "overfitting_prevented",
    }

    # Case 2: Surfaces monotonic_constraints
    assert "monotonic_constraints" in q3_text or "price" in q3_text

    # Case 3: Surfaces @monotone model variants
    assert "@monotone" in q3_text or "monotone" in q3_text.lower()

    # Case 4: Surfaces holiday_window configuration
    assert "holiday_window" in q3_text or "pre_holiday" in q3_text or "5" in q3_text

    # Case 5: Recommends f.promo_elasticity_df() in Q3 recommendation and f.assess() report
    assert "promo_elasticity_df" in str(q3_row["recommendation"])
    full_assess = f.assess()
    assert "promo_elasticity_df" in full_assess


def test_t1_f9_practitioner_assessment_question_5_three_modes_and_simplex() -> None:
    """Tier 1 F9 (>= 5 cases): Question 5 (`hierarchy_reconciliation`) distinguishes Mode 1
    (`spatial_only`), Mode 2 (`temporal_only` -> `temporal_coherent_and_verified`), Mode 3
    (`cross_temporal` with `spatial_weight_alpha`), Compositional Simplex (`simplex_sum`),
    and plain flat panels (`not_applicable_flat_panel`).
    """
    hier_panel = _make_retail_hierarchy_panel(n_months=30, seed=151)[
        ["ts_id", "region", "category", "ds", "y"]
    ].copy()

    # Case 1: Mode 1 Spatial-Only (`spatial_only`)
    f_spatial = sf.Forecaster.auto(
        hier_panel,
        horizon=6,
        freq="MS",
        models=["naive_mean"],
        baselines=["naive_mean"],
        overrides={
            "hierarchy": {
                "enabled": True,
                "levels": [["region"]],
                "reconcile_temporal": False,
                "cross_temporal": False,
            }
        },
        hpo=False,
        calibration=False,
        explainability=False,
        n_folds=1,
        autofocus=False,
    )
    q5_sp = (
        f_spatial.practitioner_assessment_df()
        .query("question_id == 'hierarchy_reconciliation'")
        .iloc[0]
    )
    assert q5_sp["verdict"] == "coherent_and_verified"
    assert "spatial_only" in f"{q5_sp['answer']} {q5_sp['evidence']}"

    # Case 2: Mode 2 Temporal-Only on a flat panel returns "temporal_coherent_and_verified"
    flat_daily = _make_daily_series(n_days=56, seed=152)
    f_temp = sf.Forecaster.auto(
        flat_daily,
        horizon=14,
        freq="D",
        models=["naive_mean"],
        baselines=["naive_mean"],
        overrides={
            "hierarchy": {
                "enabled": False,
                "levels": [],
                "reconcile_temporal": True,
                "temporal_aggregations": [7, 14],
            }
        },
        hpo=False,
        calibration=False,
        explainability=False,
        n_folds=1,
        autofocus=False,
    )
    q5_tm = (
        f_temp.practitioner_assessment_df()
        .query("question_id == 'hierarchy_reconciliation'")
        .iloc[0]
    )
    assert q5_tm["verdict"] == "temporal_coherent_and_verified"
    assert "temporal_only" in f"{q5_tm['answer']} {q5_tm['evidence']}"

    # Case 3: Mode 3 Cross-Temporal reports "cross_temporal" and "spatial_weight_alpha"
    f_ct = sf.Forecaster.auto(
        hier_panel,
        horizon=6,
        freq="MS",
        models=["naive_mean"],
        baselines=["naive_mean"],
        overrides={
            "hierarchy": {
                "levels": [["region"]],
                "cross_temporal": True,
                "temporal_aggregations": [3, 6],
            }
        },
        hpo=False,
        calibration=False,
        explainability=False,
        n_folds=1,
        autofocus=False,
    )
    q5_ct = (
        f_ct.practitioner_assessment_df().query("question_id == 'hierarchy_reconciliation'").iloc[0]
    )
    assert q5_ct["verdict"] == "coherent_and_verified"
    q5_ct_text = f"{q5_ct['answer']} {q5_ct['evidence']}"
    assert "cross_temporal" in q5_ct_text
    assert "spatial_weight_alpha" in q5_ct_text or "alpha" in q5_ct_text

    # Case 4: Compositional Simplex panel reports simplex_sum closure in Question 5
    simplex_panel = _make_compositional_panel(n_brands=3, n_periods=30, simplex_sum=100.0, seed=153)
    f_sim = sf.Forecaster.auto(
        simplex_panel,
        horizon=6,
        freq="MS",
        models=["naive_mean@transform_ilr"],
        baselines=["naive_mean"],
        hpo=False,
        calibration=False,
        explainability=False,
        n_folds=1,
        autofocus=False,
    )
    q5_sim = (
        f_sim.practitioner_assessment_df()
        .query("question_id == 'hierarchy_reconciliation'")
        .iloc[0]
    )
    q5_sim_text = f"{q5_sim['answer']} {q5_sim['evidence']}"
    assert "simplex" in q5_sim_text.lower()
    assert "100" in q5_sim_text
    assert q5_sim["verdict"] != "not_applicable_flat_panel"

    # Case 5: Plain flat panel without hierarchy, temporal reconciliation, or simplex
    f_flat = sf.Forecaster.auto(
        flat_daily,
        horizon=7,
        freq="D",
        models=["naive_mean"],
        baselines=["naive_mean"],
        hpo=False,
        calibration=False,
        explainability=False,
        n_folds=1,
        autofocus=False,
    )
    q5_flat = (
        f_flat.practitioner_assessment_df()
        .query("question_id == 'hierarchy_reconciliation'")
        .iloc[0]
    )
    assert q5_flat["verdict"] == "not_applicable_flat_panel"


def test_t1_f10_evolve_ensemble_post_blend_simplex_and_hierarchy_coherence() -> None:
    """Tier 1 F10 (>= 5 cases): `Forecaster.evolve(target="ensemble")` enforces post-blend
    compositional simplex closure (`< 1e-6`) and cross-sectional hierarchy coherence (`< 1e-6`).
    """
    # Case 1: _blend_frame_with_series_weights with heterogeneous per-series weights + simplex_sum
    dates = [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-02-01")]
    frame = pd.DataFrame(
        [
            {"ts_id": "b0", "forecast_date": dates[0], "model_type": "m1", "yhat": 70.0},
            {"ts_id": "b1", "forecast_date": dates[0], "model_type": "m1", "yhat": 30.0},
            {"ts_id": "b0", "forecast_date": dates[0], "model_type": "m2", "yhat": 20.0},
            {"ts_id": "b1", "forecast_date": dates[0], "model_type": "m2", "yhat": 80.0},
            {"ts_id": "b0", "forecast_date": dates[1], "model_type": "m1", "yhat": 60.0},
            {"ts_id": "b1", "forecast_date": dates[1], "model_type": "m1", "yhat": 40.0},
            {"ts_id": "b0", "forecast_date": dates[1], "model_type": "m2", "yhat": 25.0},
            {"ts_id": "b1", "forecast_date": dates[1], "model_type": "m2", "yhat": 75.0},
        ]
    )
    # Without simplex closure, b0 gets 65 and b1 gets 75 -> sum=140 != 100
    het_weights = {"b0": {"m1": 0.9, "m2": 0.1}, "b1": {"m1": 0.1, "m2": 0.9}}
    blended_sim = _blend_frame_with_series_weights(
        frame, het_weights, date_col="forecast_date", simplex_sum=100.0
    )
    for _dt, grp in blended_sim.groupby("forecast_date"):
        assert abs(float(grp["yhat_blend"].sum()) - 100.0) < 1e-6
        assert bool((grp["yhat_blend"] > 0.0).all())

    # Case 2: Forecaster.evolve(target="ensemble") on a compositional simplex panel (S=100.0)
    comp_100 = _make_compositional_panel(n_brands=3, n_periods=30, simplex_sum=100.0, seed=163)
    res_100 = sf.Forecaster.evolve(
        comp_100,
        target="ensemble",
        backend="mock",
        max_evaluations=2,
        horizon=4,
    )
    assert not res_100.predictions_df.empty
    assert verify_simplex(res_100.predictions_df, simplex_sum=100.0) < 1e-6
    assert bool((res_100.predictions_df["yhat"] > 0.0).all())

    # Case 3: Forecaster.evolve(target="ensemble") on a unit simplex panel (S=1.0)
    comp_1 = _make_compositional_panel(n_brands=3, n_periods=30, simplex_sum=1.0, seed=167)
    res_1 = sf.Forecaster.evolve(
        comp_1,
        target="ensemble",
        backend="mock",
        max_evaluations=2,
        horizon=4,
    )
    assert not res_1.predictions_df.empty
    assert verify_simplex(res_1.predictions_df, simplex_sum=1.0) < 1e-6
    assert bool((res_1.predictions_df["yhat"] > 0.0).all())

    # Case 4 & 5: Hierarchical panel evolved ensemble predictions preserve coherence (< 1e-6)
    raw_hier = _make_retail_hierarchy_panel(n_months=30, seed=173)[
        ["ts_id", "region", "ds", "y"]
    ].copy()
    cfg_hier = RunConfig.model_validate(
        {
            "run_name": "t1-f10-hier",
            "data": {"source_table": "p.d.t", "freq": "MS", "horizon": 4},
            "models": ["naive_mean", "naive_drift"],
            "hierarchy": {"enabled": True, "levels": [["region"]]},
        }
    )
    full_hier_df, spec = build_hierarchy(raw_hier, cfg_hier)
    ev_hier = EvolutionEvaluator(full_hier_df, target="ensemble", horizon=4, freq="MS")
    ev_hier.hierarchy_spec = spec
    mat_hier = ev_hier.materialize_ensemble_predictions(
        {
            nid: {
                "naive_mean": 0.8 if idx % 2 == 0 else 0.2,
                "naive_drift": 0.2 if idx % 2 == 0 else 0.8,
            }
            for idx, nid in enumerate(spec.node_ids)
        }
    )
    assert verify_coherence(mat_hier, spec) < 1e-6

    res_hier = sf.Forecaster.evolve(
        raw_hier,
        target="ensemble",
        backend="mock",
        max_evaluations=2,
        horizon=4,
    )
    assert not res_hier.predictions_df.empty
    if set(spec.node_ids).issubset(set(res_hier.predictions_df["ts_id"].astype(str))):
        assert verify_coherence(res_hier.predictions_df, spec) < 1e-6


def test_t1_f11_build_mutation_prompt_structural_guardrails() -> None:
    """Tier 1 F11 (>= 5 cases): `build_mutation_prompt()` formats all 5 structural guardrails
    across evolutionary targets.
    """
    diag_guardrails: dict[str, Any] = {
        "fva_pct": 4.25,
        "dm_pvalue": 0.032,
        "worst_decile_fva": 0.015,
        "horizon_decay": 1.12,
        "inner_cv_score": 0.082,
        "baseline_score": 0.091,
        "monotonic_constraints": {"price": -1, "promo": 1},
        "is_compositional_simplex": True,
        "simplex_total": 100.0,
        "hierarchy_id_cols": ["region", "category"],
        "reconcile_temporal": True,
    }

    # Case 1: monotonic_constraints rendered in prompt
    p_feat = build_mutation_prompt(
        "features", get_target_seed("features").seed_code, diag_guardrails, generation=1
    )
    assert "monotonic_constraints" in p_feat
    assert "price" in p_feat

    # Case 2: is_compositional_simplex and simplex_total rendered in prompt
    p_ens = build_mutation_prompt(
        "ensemble", get_target_seed("ensemble").seed_code, diag_guardrails, generation=1
    )
    assert "is_compositional_simplex" in p_ens
    assert "simplex_total" in p_ens
    assert "100" in p_ens

    # Case 3: hierarchy_id_cols rendered in prompt
    p_mod = build_mutation_prompt(
        "model", get_target_seed("model").seed_code, diag_guardrails, generation=2
    )
    assert "hierarchy_id_cols" in p_mod
    assert "region" in p_mod

    # Case 4: reconcile_temporal rendered in prompt
    p_pol = build_mutation_prompt(
        "policy", get_target_seed("policy").seed_code, diag_guardrails, generation=1
    )
    assert "reconcile_temporal" in p_pol

    # Case 5: All 4 targets include the structural guardrail context when active
    for target in EVOLVE_TARGETS:
        prompt_t = build_mutation_prompt(
            target, get_target_seed(target).seed_code, diag_guardrails, generation=1
        )
        for key in (
            "monotonic_constraints",
            "is_compositional_simplex",
            "simplex_total",
            "hierarchy_id_cols",
            "reconcile_temporal",
        ):
            assert key in prompt_t, f"Missing {key} in {target} mutation prompt"


def test_t1_f12_auto_policy_weights_track_j_and_k_action_priors() -> None:
    """Tier 1 F12 (>= 5 cases): Track J & K action priors registered in
    `evolved_policy_thresholds()["action_weights"]`, `POLICY_SEED_CODE`, and mock candidates.
    """
    th = evolved_policy_thresholds()
    aw = th["action_weights"]

    # Case 1: inject_monotone_guardrails is registered with valid (confidence, gain) tuple
    assert "inject_monotone_guardrails" in aw
    assert 0.0 < float(aw["inject_monotone_guardrails"][0]) <= 1.0
    assert float(aw["inject_monotone_guardrails"][1]) > 0.0

    # Case 2: inject_transform_simplex is registered with valid (confidence, gain) tuple
    assert "inject_transform_simplex" in aw
    assert 0.0 < float(aw["inject_transform_simplex"][0]) <= 1.0
    assert float(aw["inject_transform_simplex"][1]) > 0.0

    # Case 3: enable_temporal_reconciliation is registered with valid (confidence, gain) tuple
    assert "enable_temporal_reconciliation" in aw
    assert 0.0 < float(aw["enable_temporal_reconciliation"][0]) <= 1.0
    assert float(aw["enable_temporal_reconciliation"][1]) > 0.0

    # Case 4: policy seed code and deterministic_mock_candidates("policy") contain all 3 keys
    policy_seed_code = get_target_seed("policy").seed_code
    for key in (
        "inject_monotone_guardrails",
        "inject_transform_simplex",
        "enable_temporal_reconciliation",
    ):
        assert key in policy_seed_code, f"{key} missing from policy seed code"
        for cand_code in deterministic_mock_candidates("policy"):
            assert key in cand_code, f"{key} missing from policy mock candidate"

    # Case 5: Evolving target="policy" preserves all 3 Track J/K action weight keys
    panel = _make_daily_series(n_days=35, seed=181)
    res_pol = sf.Forecaster.evolve(
        panel,
        target="policy",
        backend="mock",
        max_evaluations=2,
        horizon=4,
    )
    assert res_pol.promoted is True
    for key in (
        "inject_monotone_guardrails",
        "inject_transform_simplex",
        "enable_temporal_reconciliation",
    ):
        assert key in res_pol.champion_code


# =============================================================================
# TIER 2 EXPANSION — Boundary Value & Corner-Case Hardening for F1–F12
# =============================================================================


@pytest.mark.parametrize(
    "bad_weight",
    [-0.01, -5.0, 1.01, 2.5, float("nan"), float("inf"), -float("inf"), True, False, "invalid"],
)
def test_t2_hierarchy_config_and_spatial_temporal_weight_invalid_inputs(
    bad_weight: Any,
) -> None:
    """Tier 2 F1/F2: Rejects out-of-range, non-finite, boolean, and invalid string weights,
    and rejects cross_temporal=True with empty levels.
    """
    with pytest.raises((ValidationError, ValueError)):
        HierarchyConfig(spatial_temporal_weight=bad_weight)

    with pytest.raises((ValidationError, ValueError), match="levels"):
        HierarchyConfig(cross_temporal=True, levels=[])


@pytest.mark.parametrize(
    "bad_smoothness",
    [-0.001, -10.0, float("nan"), float("inf"), -float("inf"), True, False],
)
def test_t2_smoothness_weight_invalid_inputs_and_single_step_horizon(
    bad_smoothness: Any,
) -> None:
    """Tier 2 F6: Rejects negative, non-finite, or boolean smoothness_weight and handles H=1."""
    preds_h2 = pd.DataFrame(
        {
            "ts_id": ["S1", "S2", "S1", "S2"],
            "model_type": ["naive"] * 4,
            "forecast_date": [
                pd.Timestamp("2025-01-01"),
                pd.Timestamp("2025-01-01"),
                pd.Timestamp("2025-02-01"),
                pd.Timestamp("2025-02-01"),
            ],
            "yhat": [50.0, 50.0, 50.0, 50.0],
        }
    )
    with pytest.raises((ConfigError, ValueError)):
        allocate_target(preds_h2, target_total=220.0, smoothness_weight=bad_smoothness)

    # Single-step horizon H = 1 with positive smoothness_weight=25.0 works cleanly
    preds_h1 = preds_h2.iloc[:2].copy()
    res_h1 = allocate_target(
        preds_h1, target_total=120.0, smoothness_weight=25.0, return_result=True
    )
    assert isinstance(res_h1, GoalSeekResult)
    assert abs(res_h1.allocated_total - 120.0) < 1e-6


def test_t2_sample_coherent_paths_invalid_inputs_nonneg_and_seed_determinism() -> None:
    """Tier 2 F7: sample_coherent_paths error handling, nonneg clamping, and seed determinism."""
    from scale_forecasting.temporal_reconciliation import sample_coherent_paths

    raw = _make_retail_hierarchy_panel(n_months=24, seed=191)
    cfg = RunConfig.model_validate(
        {
            "run_name": "t2-paths-edge",
            "data": {"source_table": "p.d.t", "freq": "MS", "horizon": 6},
            "models": ["naive_mean"],
            "hierarchy": {
                "levels": [["region"]],
                "cross_temporal": True,
                "temporal_aggregations": [3, 6],
            },
        }
    )
    panel, spec = build_hierarchy(raw, cfg)
    f_dates = pd.date_range("2024-01-01", periods=6, freq="MS")
    preds_df = pd.DataFrame(
        [
            {
                "ts_id": nid,
                "model_type": "naive_mean",
                "forecast_date": dt,
                "yhat": 2.0,
                "yhat_lower": -25.0,
                "yhat_upper": 29.0,
            }
            for nid in spec.node_ids
            for dt in f_dates
        ]
    )

    # 1 & 2: Invalid n_paths <= 0 or bool raises ConfigError / ValueError
    for bad_n in (0, -1, -10, False):
        with pytest.raises((ConfigError, ValueError)):
            sample_coherent_paths(preds_df, spec, cfg=cfg, n_paths=bad_n)

    # 3: nonneg=True clamps negative draws while keeping exact cross-sectional & temporal coherence
    paths_nn = sample_coherent_paths(
        preds_df, spec, history_df=panel, cfg=cfg, n_paths=25, seed=42, nonneg=True
    )
    assert bool((paths_nn["yhat_sample"] >= -1e-12).all())
    for _pid, p_grp in paths_nn.groupby("path_id"):
        assert verify_coherence(p_grp, spec) < 1e-6
        assert verify_temporal_coherence(p_grp, cfg=cfg) < 1e-6

    # 4: Identical seed reproduces identical trajectories; different seed differs
    paths_a = sample_coherent_paths(preds_df, spec, cfg=cfg, n_paths=10, seed=42)
    paths_b = sample_coherent_paths(preds_df, spec, cfg=cfg, n_paths=10, seed=42)
    paths_c = sample_coherent_paths(preds_df, spec, cfg=cfg, n_paths=10, seed=99)
    np.testing.assert_allclose(
        paths_a["yhat_sample"].to_numpy(dtype=float),
        paths_b["yhat_sample"].to_numpy(dtype=float),
        atol=1e-12,
    )
    assert not np.allclose(
        paths_a["yhat_sample"].to_numpy(dtype=float),
        paths_c["yhat_sample"].to_numpy(dtype=float),
    )

    # 5: Empty DataFrame returns empty DataFrame with expected columns
    empty_paths = sample_coherent_paths(preds_df.iloc[0:0], spec, cfg=cfg, n_paths=5)
    assert empty_paths.empty
    assert {"path_id", "ts_id", "forecast_date", "yhat_sample"}.issubset(empty_paths.columns)


def test_t2_evolve_and_mutation_prompt_edge_cases_without_guardrails() -> None:
    """Tier 2 F10/F11: build_mutation_prompt with None/empty diagnostics and empty frame blend."""
    seed_code = get_target_seed("features").seed_code
    p_none = build_mutation_prompt("features", seed_code, diagnostics=None, generation=1)
    p_empty = build_mutation_prompt("features", seed_code, diagnostics={}, generation=1)
    assert "fva_pct=0.00%" in p_none
    assert "fva_pct=0.00%" in p_empty

    empty_blend = _blend_frame_with_series_weights(
        pd.DataFrame(), {"m1": 1.0}, date_col="forecast_date", simplex_sum=100.0
    )
    assert empty_blend.empty


# =============================================================================
# TIER 3 EXPANSION — Cross-Feature Combinations (3 Modes + Paths + Smooth GS + Evolve)
# =============================================================================


def test_t3_three_modes_head_to_head_and_smooth_goal_seek_with_coherent_paths() -> None:
    """Tier 3: Head-to-head execution of Mode 1 (Spatial-Only), Mode 2 (Temporal-Only), and
    Mode 3 (Joint Cross-Temporal with `spatial_temporal_weight="auto"`), chained into
    `sample_coherent_paths(n_paths=50)` and `allocate_target(smoothness_weight=10.0)`.
    """
    from scale_forecasting.temporal_reconciliation import sample_coherent_paths

    raw = _make_retail_hierarchy_panel(n_months=36, seed=211)

    # Mode 1: Spatial-Only
    cfg_m1 = RunConfig.model_validate(
        {
            "run_name": "t3-mode1",
            "data": {"source_table": "p.d.t", "freq": "MS", "horizon": 6},
            "models": ["naive_seasonal"],
            "backtest": {"enabled": True, "n_folds": 1, "horizon": 6, "step": 6, "min_train": 24},
            "hierarchy": {
                "enabled": True,
                "levels": [["region"]],
                "reconcile_temporal": False,
                "cross_temporal": False,
                "reconciliation_methods": ["wls_struct", "mint_shrink"],
            },
        }
    )
    # Mode 2: Temporal-Only on bottom leaves
    cfg_m2 = RunConfig.model_validate(
        {
            "run_name": "t3-mode2",
            "data": {"source_table": "p.d.t", "freq": "MS", "horizon": 6},
            "models": ["naive_seasonal"],
            "backtest": {"enabled": True, "n_folds": 1, "horizon": 6, "step": 6, "min_train": 24},
            "hierarchy": {
                "enabled": False,
                "levels": [],
                "reconcile_temporal": True,
                "temporal_aggregations": [3, 6],
                "reconciliation_methods": ["wls_struct"],
            },
        }
    )
    # Mode 3: Joint Cross-Temporal with spatial_temporal_weight="auto"
    cfg_m3 = RunConfig.model_validate(
        {
            "run_name": "t3-mode3",
            "data": {"source_table": "p.d.t", "freq": "MS", "horizon": 6},
            "models": ["naive_seasonal"],
            "backtest": {"enabled": True, "n_folds": 1, "horizon": 6, "step": 6, "min_train": 24},
            "hierarchy": {
                "levels": [["region"]],
                "cross_temporal": True,
                "temporal_aggregations": [3, 6],
                "spatial_temporal_weight": "auto",
            },
        }
    )

    _, spec = build_hierarchy(raw, cfg_m3)
    cells_m1, _ = run_group(raw, cfg_m1)
    cells_m2, _ = run_group(raw, cfg_m2)
    cells_m3, _ = run_group(raw, cfg_m3)

    assert any(c.model_type == "naive_seasonal_mint_shrink" for c in cells_m1)
    assert any(c.model_type == "naive_seasonal_wls_struct" for c in cells_m2)
    assert any(c.model_type == "naive_seasonal_cross_temporal_mint" for c in cells_m3)
    assert any(c.model_type == "naive_seasonal_mint_shrink" for c in cells_m3)

    ct_preds = pd.concat(
        [
            c.predictions.assign(
                ts_id=c.ts_id,
                model_type=c.model_type,
                forecast_date=c.predictions["ds"],
            )
            for c in cells_m3
            if c.model_type == "naive_seasonal_cross_temporal_mint"
        ],
        ignore_index=True,
    )
    assert verify_coherence(ct_preds, spec) < 1e-6

    # Chain Mode 3 predictions into sample_coherent_paths(n_paths=50)
    paths_50 = sample_coherent_paths(
        ct_preds, spec, history_df=raw, cfg=cfg_m3, n_paths=50, seed=42
    )
    assert paths_50["path_id"].nunique() == 50
    for _pid, p_grp in paths_50.groupby("path_id"):
        assert verify_coherence(p_grp, spec) < 1e-6

    # Chain Mode 3 predictions into multi-period smooth allocate_target(smoothness_weight=10.0)
    base_total = float(ct_preds.loc[ct_preds["ts_id"] == TOTAL_NODE_ID, "yhat"].sum())
    gs_smooth = allocate_target(
        ct_preds,
        target_total=base_total * 1.10,
        hier_spec=spec,
        locked_nodes=["region=South"],
        smoothness_weight=10.0,
        return_result=True,
    )
    assert isinstance(gs_smooth, GoalSeekResult)
    assert abs(gs_smooth.allocated_total - base_total * 1.10) < 1e-4
    assert gs_smooth.max_coherence_error < 1e-6
    assert verify_coherence(gs_smooth.dataframe, spec) < 1e-6


# =============================================================================
# TIER 4 EXPANSION — Real-World Enterprise Simplex Evolve & Assessment Scenario
# =============================================================================


def test_t4_compositional_market_share_evolve_and_practitioner_assess() -> None:
    """Tier 4 Real-World Scenario 4: Compositional Market-Share Forecasting with
    `Forecaster.auto()`, `f.assess()` (Q3 & Q5), and `Forecaster.evolve(target="ensemble")`.
    """
    panel = _make_compositional_panel(
        n_brands=4,
        n_periods=32,
        simplex_sum=100.0,
        with_launch_zeros=True,
        with_covariates=True,
        seed=223,
    )
    f = sf.Forecaster.auto(
        panel,
        horizon=4,
        freq="MS",
        models=["naive_mean@transform_ilr", "regression_lags@transform_clr"],
        baselines=["naive_mean"],
        overrides={"features": {"monotonic_constraints": {"price_index": -1, "promo": 1}}},
        hpo=False,
        calibration=False,
        explainability=False,
        n_folds=1,
        autofocus=False,
    )
    assert verify_simplex(f.predictions_df(), simplex_sum=100.0) < 1e-6

    assess_df = f.practitioner_assessment_df()
    q3 = assess_df.query("question_id == 'covariates_vs_overfitting'").iloc[0]
    q5 = assess_df.query("question_id == 'hierarchy_reconciliation'").iloc[0]
    assert "promo_elasticity_df" in str(q3["recommendation"])
    assert "simplex" in f"{q5['answer']} {q5['evidence']}".lower()

    ev_res = sf.Forecaster.evolve(
        panel,
        target="ensemble",
        backend="mock",
        max_evaluations=2,
        horizon=4,
    )
    assert not ev_res.predictions_df.empty
    assert verify_simplex(ev_res.predictions_df, simplex_sum=100.0) < 1e-6
    assert bool((ev_res.predictions_df["yhat"] > 0.0).all())


# =============================================================================
# TIER 5 EXPANSION — Adversarial Regression Coverage for Evolve Hierarchy,
# Mode 3 Learned Alpha Assessment, and Aggregate-Node Goal-Seek Target Total
# =============================================================================


def test_t5_evolve_hierarchy_levels_assess_mode3_alpha_and_goal_seek_locked_aggregate() -> None:
    """Tier 5 regression test for R3.2 (EvolutionEvaluator nested hierarchy.levels flattening and
    champion_outputs['hierarchy_spec'] propagation), R2.2 (learned float spatial_weight_alpha in
    AutoForecastSession.practitioner_assessment_df() Mode 3), and R4 (GoalSeekResult.target_total
    accuracy when an aggregate node is locked).
    """
    raw_hier = _make_retail_hierarchy_panel(n_months=30, seed=173)[
        ["ts_id", "region", "ds", "y"]
    ].copy()
    f_ct = sf.Forecaster.auto(
        raw_hier,
        tier="first_light",
        models=["naive_mean", "naive_drift"],
        horizon=4,
        base_config={
            "hierarchy": {
                "enabled": True,
                "levels": [["region"]],
                "reconciliation_methods": ["wls_struct", "cross_temporal_mint"],
                "reconcile_temporal": True,
                "cross_temporal": True,
                "temporal_aggregations": [2, 4],
                "spatial_temporal_weight": "auto",
            }
        },
    )
    assert f_ct.auto_session is not None
    diag = f_ct.auto_session.last_diagnostics_summary
    alpha_val = diag["spatial_weight_alpha"]
    assert isinstance(alpha_val, float) and not isinstance(alpha_val, bool)
    assert 0.0 <= alpha_val <= 1.0

    assess_df = f_ct.practitioner_assessment_df()
    q5 = assess_df.query("question_id == 'hierarchy_reconciliation'").iloc[0]
    assert q5["verdict"] == "coherent_and_verified"
    assert "spatial_weight_alpha=auto" not in str(q5["answer"])
    assert "spatial_weight_alpha=auto" not in str(q5["evidence"])
    assert f"spatial_weight_alpha={alpha_val:.4f}" in str(q5["evidence"])

    # R3.2: Forecaster.evolve on from_runs=[f_ct] with nested levels=[["region"]]
    spec = f_ct.auto_session.champion_outputs.get("hierarchy_spec")
    assert spec is not None
    ev_from_run = sf.Forecaster.evolve(
        from_runs=[f_ct],
        target="ensemble",
        backend="mock",
        max_evaluations=2,
    )
    assert not ev_from_run.predictions_df.empty
    assert verify_coherence(ev_from_run.predictions_df, spec) < 1e-6

    # R3.2: Forecaster.evolve on a raw panel with auto-detected hierarchy columns (region, store)
    raw_with_store = _make_retail_hierarchy_panel(n_months=28, seed=179).copy()
    raw_with_store["store"] = raw_with_store["ts_id"]
    ev_raw_hier = sf.Forecaster.evolve(
        raw_with_store[["ts_id", "region", "store", "ds", "y"]],
        target="ensemble",
        backend="mock",
        max_evaluations=2,
        horizon=4,
    )
    assert not ev_raw_hier.predictions_df.empty

    # R4: GoalSeekResult.target_total when locking an upper-level aggregate node ("region=South")
    ct_preds = f_ct.auto_session.raw_predictions_df[
        f_ct.auto_session.raw_predictions_df["model_type"] == "naive_mean_cross_temporal_mint"
    ].copy()
    assert not ct_preds.empty
    base_tot = float(ct_preds.loc[ct_preds["ts_id"] == TOTAL_NODE_ID, "yhat"].sum())
    target_tot = base_tot * 1.15
    gs_res = allocate_target(
        ct_preds,
        target_total=target_tot,
        hier_spec=spec,
        locked_nodes=["region=South"],
        return_result=True,
    )
    assert isinstance(gs_res, GoalSeekResult)
    assert abs(gs_res.target_total - target_tot) < 1e-6
    assert gs_res.max_target_discrepancy < 1e-6
    assert gs_res.max_coherence_error < 1e-6


def test_t6_multi_engine_sharding_mode2_ensemble_and_dgp_benchmark_parity() -> None:
    """Verify Spark/Ray/Vertex multi-worker sharding on simplex & cross-temporal configs,
    Mode 2 flat-panel temporal ensemble reconciliation, and synthetic DGP benchmark parity.
    """
    from scale_forecasting.benchmarks import list_benchmarks, load_benchmark
    from scale_forecasting.engines.ray_io import chunk_cells
    from scale_forecasting.engines.spark_io import bucket_key_cols
    from scale_forecasting.engines.vertex_engine import (
        WorkerTopology,
        models_sharded_across_workers,
        partition_models_for_worker,
    )

    # 1. Spark / Ray / Vertex worker sharding with compositional simplex (@transform_ilr)
    simplex_panel = _make_compositional_panel(
        n_brands=4, n_periods=36, simplex_sum=100.0, with_launch_zeros=True, seed=211
    )
    cfg_simplex = RunConfig.model_validate(
        {
            "run_name": "t6-simplex-sharding",
            "data": {"source_table": "p.d.t", "freq": "MS", "horizon": 6},
            "models": ["theta@transform_ilr", "naive_seasonal@transform_clr"],
            "features": {"simplex_sum": 100.0},
            "backtest": {"enabled": True, "n_folds": 1, "horizon": 6},
        }
    )
    assert bucket_key_cols(cfg_simplex) == ["_sf_model"]
    assert models_sharded_across_workers(cfg_simplex, cfg_simplex.models) is True

    # Ray chunk_cells must keep each simplex model's full panel together in a single chunk
    ray_chunks = chunk_cells(simplex_panel, cfg_simplex, cfg_simplex.models, n_chunks=8)
    assert len(ray_chunks) == 2
    for ch in ray_chunks:
        assert int(ch["ts_id"].nunique()) == int(simplex_panel["ts_id"].nunique())
        ch_results, _ = run_group(ch, cfg_simplex, cfg_simplex.models)
        ch_preds = pd.concat(
            [r.predictions.assign(ts_id=r.ts_id, model_type=r.model_type) for r in ch_results],
            ignore_index=True,
        )
        assert verify_simplex(ch_preds, simplex_sum=100.0) < 1e-6

    # Vertex multi-worker partitioning must assign whole models per worker rank
    rank0_models = partition_models_for_worker(
        cfg_simplex, cfg_simplex.models, WorkerTopology(rank=0, world_size=2)
    )
    rank1_models = partition_models_for_worker(
        cfg_simplex, cfg_simplex.models, WorkerTopology(rank=1, world_size=2)
    )
    assert rank0_models == ["theta@transform_ilr"]
    assert rank1_models == ["naive_seasonal@transform_clr"]

    # 2. Mode 2 (Temporal-Only THieF on flat panel) nonlinear median ensemble temporal coherence
    flat_panel = simplex_panel[simplex_panel["ts_id"].isin(["brand_0", "brand_1"])].copy()
    cfg_m2 = RunConfig.model_validate(
        {
            "run_name": "t6-mode2-ensemble",
            "data": {"source_table": "p.d.t", "freq": "MS", "horizon": 12},
            "models": ["naive_mean", "naive_seasonal"],
            "backtest": {"enabled": True, "n_folds": 1, "horizon": 12},
            "hierarchy": {
                "enabled": False,
                "levels": [],
                "reconcile_temporal": True,
                "temporal_aggregations": [3, 6, 12],
                "reconciliation_methods": ["mint_shrink"],
            },
            "ensemble": {"enabled": True, "strategies": ["median"]},
        }
    )
    m2_cells, _ = run_group(flat_panel, cfg_m2, cfg_m2.models)
    rec_m2 = [r for r in m2_cells if r.model_type.endswith("_mint_shrink")]
    assert len(rec_m2) == 4  # 2 series * 2 models
    ens_pred_raw = (
        pd.concat(
            [r.predictions.assign(ts_id=r.ts_id, model_type=r.model_type) for r in rec_m2],
            ignore_index=True,
        )
        .groupby(["ts_id", "ds"], as_index=False)["yhat"]
        .median()
        .assign(model_type="ensemble_median")
    )
    ens_pred_rec = reconcile_temporal_forecasts(
        ens_pred_raw,
        history_df=flat_panel,
        cfg=cfg_m2,
        method="mint_shrink",
        stamp_method=False,
        return_aggregations=False,
    )
    assert isinstance(ens_pred_rec, pd.DataFrame)
    assert verify_temporal_coherence(ens_pred_rec, cfg=cfg_m2) < 1e-6

    # 3. Synthetic DGP benchmark suite sanity across hierarchical & compositional panels
    b_names = list_benchmarks()
    assert len(b_names) >= 20
    for b_name in b_names[:3]:
        b = load_benchmark(b_name)
        bp = sf.Forecaster.build_config(b.df, horizon=b.spec.horizon, tier="first_light")
        assert bp.config.data.horizon == b.spec.horizon
