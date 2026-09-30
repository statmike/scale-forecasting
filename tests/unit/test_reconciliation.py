"""Unit tests for hierarchical aggregation and coherent forecast reconciliation (Hyndman FPP3)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from scale_forecasting.config import RECONCILIATION_METHODS, RunConfig
from scale_forecasting.data_gen.generator import GenConfig, generate_panel
from scale_forecasting.engines.ray_io import chunk_cells
from scale_forecasting.engines.spark_io import _needed_columns, bucket_key_cols, run_group
from scale_forecasting.errors import DataError
from scale_forecasting.reconciliation import (
    TOTAL_NODE_ID,
    build_hierarchy,
    reconcile_forecasts,
    reconcile_matrix,
    reconcile_oof,
    schafer_strimmer_shrinkage,
    verify_coherence,
)
from scale_forecasting.registry.ids import _canonical_config, make_run_id


def _hierarchy_panel(n_series: int = 6, n_days: int = 120) -> pd.DataFrame:
    return generate_panel(
        n_series,
        cfg=GenConfig(
            history=n_days,
            with_exog=True,
            with_hierarchy=True,
        ),
        seed=42,
    )


def _cfg(**overrides: object) -> RunConfig:
    base: dict[str, object] = {
        "run_name": "hier-test",
        "data": {"source_table": "p.d.t", "horizon": 7},
        "models": ["naive"],
        "hierarchy": {
            "enabled": True,
            "levels": [["region"], ["region", "category"]],
            "reconciliation_methods": [
                "bottom_up",
                "top_down",
                "middle_out",
                "ols",
                "wls_struct",
                "wls_var",
                "mint_shrink",
            ],
            "middle_level": ["region"],
        },
    }
    base.update(overrides)
    return RunConfig(**base)  # type: ignore[arg-type]


def test_hierarchy_config_elided_at_default_and_moves_run_id_when_enabled() -> None:
    plain = RunConfig(
        run_name="r",
        data={"source_table": "p.d.t"},
        models=["naive"],
    )
    assert "hierarchy" not in json.loads(_canonical_config(plain))
    explicit_default = RunConfig(
        run_name="r",
        data={"source_table": "p.d.t"},
        models=["naive"],
        hierarchy={"enabled": False},
    )
    assert make_run_id(plain) == make_run_id(explicit_default)

    enabled = RunConfig(
        run_name="r",
        data={"source_table": "p.d.t"},
        models=["naive"],
        hierarchy={"enabled": True, "levels": [["region"]]},
    )
    assert make_run_id(enabled) != make_run_id(plain)
    assert json.loads(_canonical_config(enabled))["hierarchy"]["enabled"] is True


def test_hierarchy_config_validation_rejects_invalid_specs() -> None:
    with pytest.raises(ValidationError, match="requires at least one aggregation level"):
        RunConfig(
            run_name="r",
            data={"source_table": "p.d.t"},
            models=["naive"],
            hierarchy={"enabled": True, "levels": []},
        )

    with pytest.raises(ValidationError, match="contains duplicates"):
        RunConfig(
            run_name="r",
            data={"source_table": "p.d.t"},
            models=["naive"],
            hierarchy={
                "enabled": True,
                "levels": [["region"]],
                "reconciliation_methods": ["bottom_up", "bottom_up"],
            },
        )

    with pytest.raises(ValidationError, match="must be one of hierarchy.levels"):
        RunConfig(
            run_name="r",
            data={"source_table": "p.d.t"},
            models=["naive"],
            hierarchy={
                "enabled": True,
                "levels": [["region"]],
                "middle_level": ["category"],
            },
        )


def test_build_hierarchy_nested_and_grouped_coherence() -> None:
    panel = _hierarchy_panel(n_series=6, n_days=60)
    cfg_nested = _cfg(
        features={
            "future_covariates": ["promo_flag", "price_index"],
            "past_covariates": ["temperature"],
        }
    )
    hier_df, spec = build_hierarchy(panel, cfg_nested)

    assert spec.node_ids[0] == TOTAL_NODE_ID
    assert spec.n_bottom == 6
    assert (
        spec.n_nodes
        == 1 + len(spec.level_nodes["region"]) + len(spec.level_nodes["region/category"]) + 6
    )
    assert spec.summing_matrix.shape == (spec.n_nodes, 6)
    # Historical target values are strictly additive across the hierarchy.
    err = verify_coherence(hier_df, spec, value_col="y", date_col="ds", atol=1e-9)
    assert err <= 1e-9
    for cov in ["promo_flag", "price_index", "temperature"]:
        assert cov in hier_df.columns
        assert hier_df[cov].notna().all()

    # Cross-classified (grouped) hierarchy: [["region"], ["category"]]
    cfg_grouped = _cfg(
        hierarchy={
            "enabled": True,
            "levels": [["region"], ["category"]],
            "reconciliation_methods": ["wls_struct", "mint_shrink"],
        }
    )
    grouped_df, grouped_spec = build_hierarchy(panel, cfg_grouped)
    assert (
        verify_coherence(grouped_df, grouped_spec, value_col="y", date_col="ds", atol=1e-9) <= 1e-9
    )


def test_schafer_strimmer_shrinkage_positive_definite_when_n_exceeds_t() -> None:
    rng = np.random.default_rng(123)
    # High-dimensional regime: n_series=40 >> T_obs=8 (sample covariance is rank <= 8).
    residuals = rng.normal(loc=0.0, scale=2.0, size=(8, 40))
    w_shrink, lam = schafer_strimmer_shrinkage(residuals)
    assert 0.0 < lam <= 1.0
    assert w_shrink.shape == (40, 40)
    eigvals = np.linalg.eigvalsh(w_shrink)
    assert np.all(eigvals > 0.0)


@pytest.mark.parametrize("method", sorted(RECONCILIATION_METHODS))
def test_all_seven_reconciliation_methods_produce_coherent_forecasts_and_intervals(
    method: str,
) -> None:
    panel = _hierarchy_panel(n_series=6, n_days=90)
    cfg = _cfg(
        hierarchy={
            "enabled": True,
            "levels": [["region"], ["region", "category"]],
            "reconciliation_methods": [method],
            "middle_level": ["region"],
        }
    )
    hier_df, spec = build_hierarchy(panel, cfg)

    rng = np.random.default_rng(99)
    dates = pd.date_range("2025-05-01", periods=7, freq="D")
    pred_rows: list[dict[str, object]] = []
    oof_rows: list[dict[str, object]] = []

    for nid in spec.node_ids:
        for dt in dates:
            yhat = float(rng.uniform(50.0, 250.0))
            pred_rows.append(
                {
                    "ts_id": nid,
                    "model_type": "naive",
                    "forecast_date": dt,
                    "yhat": yhat,
                    "yhat_raw": yhat - 1.5,
                    "yhat_lower": yhat - 15.0,
                    "yhat_upper": yhat + 15.0,
                }
            )
        for fold_id, cutoff in [(0, pd.Timestamp("2025-04-10")), (1, pd.Timestamp("2025-04-17"))]:
            for step in range(1, 8):
                fdt = cutoff + pd.Timedelta(days=step)
                y_true = float(rng.uniform(60.0, 220.0))
                yhat_oof = y_true + float(rng.normal(0.0, 8.0))
                oof_rows.append(
                    {
                        "ts_id": nid,
                        "model_type": "naive",
                        "fold_id": fold_id,
                        "cutoff_date": cutoff,
                        "forecast_date": fdt,
                        "horizon_step": step,
                        "y_true": y_true,
                        "yhat": yhat_oof,
                        "yhat_lower": yhat_oof - 12.0,
                        "yhat_upper": yhat_oof + 12.0,
                    }
                )

    preds_df = pd.DataFrame(pred_rows)
    oof_df = pd.DataFrame(oof_rows)

    # Incoherent base forecasts fail verify_coherence.
    with pytest.raises(DataError, match="not coherent"):
        verify_coherence(preds_df, spec, value_col="yhat", date_col="forecast_date", atol=1e-3)

    rec_preds = reconcile_forecasts(
        preds_df,
        spec,
        method,  # type: ignore[arg-type]
        history_df=hier_df,
        oof_df=oof_df,
        middle_level=["region"],
    )
    assert set(rec_preds["model_type"].unique()) == {f"naive_{method}"}
    assert (
        verify_coherence(rec_preds, spec, value_col="yhat", date_col="forecast_date", atol=1e-8)
        <= 1e-8
    )
    assert (
        verify_coherence(rec_preds, spec, value_col="yhat_raw", date_col="forecast_date", atol=1e-8)
        <= 1e-8
    )
    assert (rec_preds["yhat_lower"] <= rec_preds["yhat"]).all()
    assert (rec_preds["yhat"] <= rec_preds["yhat_upper"]).all()

    rec_oof = reconcile_oof(
        oof_df,
        spec,
        method,  # type: ignore[arg-type]
        cfg=cfg,
        history_df=hier_df,
        middle_level=["region"],
    )
    for _cutoff, grp in rec_oof.groupby("cutoff_date"):
        assert (
            verify_coherence(grp, spec, value_col="yhat", date_col="forecast_date", atol=1e-8)
            <= 1e-8
        )

    # Structural invariants per method:
    g_mat = reconcile_matrix(
        spec,
        method,  # type: ignore[arg-type]
        history_df=hier_df,
        residuals_df=oof_df,
        middle_level=["region"],
    )
    if method in {"bottom_up", "ols", "wls_struct", "wls_var", "mint_shrink"}:
        # Projection idempotence: S @ G @ S == S (already-coherent forecasts stay unchanged).
        np.testing.assert_allclose(
            spec.summing_matrix @ g_mat @ spec.summing_matrix,
            spec.summing_matrix,
            atol=1e-8,
        )
    if method == "bottom_up":
        b_orig = (
            preds_df[preds_df["ts_id"].isin(spec.bottom_ids)]
            .sort_values(["ts_id", "forecast_date"])["yhat"]
            .to_numpy()
        )
        b_rec = (
            rec_preds[rec_preds["ts_id"].isin(spec.bottom_ids)]
            .sort_values(["ts_id", "forecast_date"])["yhat"]
            .to_numpy()
        )
        np.testing.assert_allclose(b_orig, b_rec, atol=1e-9)
    elif method == "top_down":
        t_orig = (
            preds_df[preds_df["ts_id"] == TOTAL_NODE_ID]
            .sort_values("forecast_date")["yhat"]
            .to_numpy()
        )
        t_rec = (
            rec_preds[rec_preds["ts_id"] == TOTAL_NODE_ID]
            .sort_values("forecast_date")["yhat"]
            .to_numpy()
        )
        np.testing.assert_allclose(t_orig, t_rec, atol=1e-9)
    elif method == "middle_out":
        mid_ids = spec.level_nodes["region"]
        m_orig = (
            preds_df[preds_df["ts_id"].isin(mid_ids)]
            .sort_values(["ts_id", "forecast_date"])["yhat"]
            .to_numpy()
        )
        m_rec = (
            rec_preds[rec_preds["ts_id"].isin(mid_ids)]
            .sort_values(["ts_id", "forecast_date"])["yhat"]
            .to_numpy()
        )
        np.testing.assert_allclose(m_orig, m_rec, atol=1e-9)


def test_run_group_and_engine_wiring_with_hierarchy_reconciliation() -> None:
    panel = _hierarchy_panel(n_series=4, n_days=100)
    cfg = _cfg(
        models=["naive_mean"],
        backtest={
            "enabled": True,
            "n_folds": 2,
            "horizon": 7,
            "step": 7,
            "min_train": 60,
        },
        hierarchy={
            "enabled": True,
            "levels": [["region"], ["region", "category"]],
            "reconciliation_methods": ["bottom_up", "wls_struct", "mint_shrink"],
        },
    )
    assert bucket_key_cols(cfg) == ["_sf_model"]
    needed = _needed_columns(cfg)
    assert "region" in needed and "category" in needed

    chunks = chunk_cells(panel, cfg, ["naive_mean"], n_chunks=4)
    assert len(chunks) == 1

    results, status = run_group(panel, cfg)
    assert (status["status"] == "ok").all()

    _, spec = build_hierarchy(panel, cfg)
    expected_models = {
        "naive_mean",
        "naive_mean_bottom_up",
        "naive_mean_wls_struct",
        "naive_mean_mint_shrink",
    }
    assert set(status["model_type"].unique()) == expected_models
    assert len(results) == spec.n_nodes * len(expected_models)

    for rec_model in ["naive_mean_bottom_up", "naive_mean_wls_struct", "naive_mean_mint_shrink"]:
        m_cells = [r for r in results if r.model_type == rec_model]
        assert len(m_cells) == spec.n_nodes
        pred_df = pd.concat(
            [c.predictions.assign(ts_id=c.ts_id, model_type=c.model_type) for c in m_cells],
            ignore_index=True,
        )
        assert verify_coherence(pred_df, spec, value_col="yhat", date_col="ds", atol=1e-8) <= 1e-8
        for c in m_cells:
            assert np.isfinite(c.metrics["wape"])
            assert np.isfinite(c.metrics["mase"])
