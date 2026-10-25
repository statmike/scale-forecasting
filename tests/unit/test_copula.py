"""Unit tests for Joint Multi-Horizon & Cross-Series Trajectory Copulas (`copula.py`)."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from scale_forecasting.config import RunConfig
from scale_forecasting.copula import (
    compute_tail_risk,
    evaluate_trajectory_paths,
    plot_trajectories,
    sample_copula_trajectories,
    sample_trajectories,
    tail_risk_df,
)
from scale_forecasting.reconciliation import build_hierarchy, verify_coherence
from scale_forecasting.simplex import verify_simplex_coherence
from scale_forecasting.temporal_reconciliation import (
    build_temporal_hierarchy,
    verify_cross_temporal_coherence,
    verify_temporal_coherence,
)


def _make_ar1_oof_and_predictions(
    rho: float = 0.75,
    horizon: int = 14,
    heavy_tails: bool = False,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create synthetic OOF residuals with AR(1) correlation rho and horizon predictions."""
    rng = np.random.default_rng(seed)
    n_hist = 120
    eps = np.empty(n_hist, dtype=float)
    innov = (
        rng.standard_t(df=3.2, size=n_hist) * 4.0
        if heavy_tails
        else rng.normal(0.0, 5.0, size=n_hist)
    )
    eps[0] = innov[0]
    for t in range(1, n_hist):
        eps[t] = rho * eps[t - 1] + np.sqrt(max(1.0 - rho**2, 0.05)) * innov[t]

    oof_dates = pd.date_range("2025-01-01", periods=n_hist, freq="D")
    yhat_oof = np.full(n_hist, 100.0)
    oof_df = pd.DataFrame(
        {
            "ts_id": "sku_A",
            "model_type": "theta",
            "ds": oof_dates,
            "y_true": yhat_oof + eps,
            "yhat": yhat_oof,
        }
    )

    f_dates = pd.date_range("2025-05-01", periods=horizon, freq="D")
    yhat_f = 100.0 + 5.0 * np.sin(2.0 * np.pi * np.arange(horizon) / 7.0)
    hw_80 = np.full(horizon, 10.0)
    pred_df = pd.DataFrame(
        {
            "ts_id": "sku_A",
            "model_type": "theta",
            "forecast_date": f_dates,
            "ds": f_dates,
            "yhat": yhat_f,
            "yhat_lower": yhat_f - hw_80,
            "yhat_upper": yhat_f + hw_80,
        }
    )
    return pred_df, oof_df


def test_copula_amplification_and_heavy_tail_student_t_vs_gaussian() -> None:
    """Positive AR(1) residuals produce amplification > 1 and Student-t > Gaussian CVaR_99."""
    pred_df, oof_df = _make_ar1_oof_and_predictions(rho=0.80, horizon=14, heavy_tails=True)

    traj_gauss = sample_trajectories(
        pred_df,
        oof_df,
        n_paths=1000,
        copula_type="gaussian",
        seed=42,
    )
    traj_t = sample_trajectories(
        pred_df,
        oof_df,
        n_paths=1000,
        copula_type="student_t",
        df_t=3.5,
        seed=42,
    )
    traj_ic = sample_trajectories(
        pred_df,
        oof_df,
        n_paths=500,
        copula_type="iman_conover",
        seed=42,
    )
    traj_auto = sample_trajectories(
        pred_df,
        oof_df,
        n_paths=300,
        copula_type="auto",
        seed=42,
    )
    assert traj_auto.attrs["copula_type_selected"] == "student_t"
    assert len(traj_ic) == 500 * 14

    # Marginal quantiles at q=0.10, 0.50, 0.90 closely match [yhat_lower, yhat, yhat_upper].
    piv_g = traj_gauss.pivot(index="path_id", columns="horizon_step", values="yhat_sample")
    q10_emp = np.quantile(piv_g.to_numpy(), 0.10, axis=0)
    q50_emp = np.quantile(piv_g.to_numpy(), 0.50, axis=0)
    q90_emp = np.quantile(piv_g.to_numpy(), 0.90, axis=0)
    np.testing.assert_allclose(q50_emp, pred_df["yhat"].to_numpy(), atol=1.5)
    np.testing.assert_allclose(q10_emp, pred_df["yhat_lower"].to_numpy(), atol=2.0)
    np.testing.assert_allclose(q90_emp, pred_df["yhat_upper"].to_numpy(), atol=2.0)

    risk_gauss = compute_tail_risk(traj_gauss, predictions_df=pred_df)
    risk_t = compute_tail_risk(traj_t, predictions_df=pred_df)

    r_g = risk_gauss.iloc[0]
    r_t = risk_t.iloc[0]

    # Positive autocorrelation amplifies multi-day cumulative demand std above i.i.d. sum std.
    assert float(r_g["copula_amplification_ratio"]) > 1.25
    assert float(r_t["copula_amplification_ratio"]) > 1.25

    # Student-t copula yields strictly higher extreme 99% CVaR than Gaussian copula.
    assert float(r_t["cvar_99"]) > float(r_g["cvar_99"])

    # Exact quantile monotonicity: CVaR_99 >= VaR_99 >= VaR_95 >= VaR_90 >= expected_cum_demand.
    for row in (r_g, r_t):
        assert (
            float(row["cvar_99"])
            >= float(row["var_99"])
            >= float(row["var_95"])
            >= float(row["var_90"])
            >= float(row["expected_cumulative_demand"])
        )

    assert sample_copula_trajectories is sample_trajectories
    assert tail_risk_df is compute_tail_risk

    ax = plot_trajectories(traj_t, tail_risk=risk_t)
    assert ax is not None


def test_copula_cross_series_floor_cap_and_simplex_closure() -> None:
    """sample_trajectories enforces floor, cap, cross_series correlation, and simplex_sum."""
    dates = pd.date_range("2025-06-01", periods=7, freq="D")
    rows: list[dict[str, object]] = []
    shares = [45.0, 35.0, 20.0]
    for idx, s_val in enumerate(shares):
        for dt in dates:
            rows.append(
                {
                    "ts_id": f"comp_{idx}",
                    "model_type": "theta",
                    "forecast_date": dt,
                    "ds": dt,
                    "yhat": s_val,
                    "yhat_lower": max(1.0, s_val - 8.0),
                    "yhat_upper": min(99.0, s_val + 8.0),
                }
            )
    pred_df = pd.DataFrame(rows)

    traj = sample_trajectories(
        pred_df,
        n_paths=120,
        copula_type="gaussian",
        cross_series=True,
        simplex_sum=100.0,
        floor=0.0,
        cap=100.0,
        seed=7,
    )
    assert (traj["yhat_sample"] >= 0.0).all()
    assert (traj["yhat_sample"] <= 100.0).all()
    disc = verify_simplex_coherence(traj, simplex_sum=100.0, raise_on_error=True)
    assert disc < 1e-6


def test_copula_three_mode_hierarchical_temporal_and_cross_temporal_coherence() -> None:
    """Sampled trajectories satisfy cross-sectional, temporal, and cross-temporal coherence."""
    cfg = RunConfig.model_validate(
        {
            "run_name": "test_copula",
            "data": {"source_table": "p.d.t", "horizon": 14, "freq": "D"},
            "models": ["theta"],
            "hierarchy": {
                "enabled": True,
                "levels": [["region"], ["region", "store"]],
                "reconcile_temporal": True,
                "cross_temporal": True,
                "temporal_aggregations": [7, 14],
            },
            "output": {"floor": 0.0},
        }
    )
    hist_dates = pd.date_range("2025-01-01", periods=28, freq="D")
    hist_rows: list[dict[str, object]] = []
    for reg, stores in [("East", ["E1", "E2"]), ("West", ["W1", "W2"])]:
        for st in stores:
            for dt in hist_dates:
                hist_rows.append(
                    {
                        "ts_id": f"{reg}/{st}",
                        "region": reg,
                        "store": st,
                        "ds": dt,
                        "y": 30.0,
                    }
                )
    raw_hist = pd.DataFrame(hist_rows)
    full_hist, hier_spec = build_hierarchy(raw_hist, cfg)
    temp_spec = build_temporal_hierarchy("D", 14, [7, 14])

    f_dates = pd.date_range("2025-01-29", periods=14, freq="D")
    pred_rows: list[dict[str, object]] = []
    for nid in hier_spec.node_ids:
        base_v = 120.0 if nid == "Total" else (60.0 if nid in ("East", "West") else 30.0)
        for dt in f_dates:
            pred_rows.append(
                {
                    "ts_id": nid,
                    "model_type": "theta",
                    "forecast_date": dt,
                    "ds": dt,
                    "yhat": base_v,
                    "yhat_lower": base_v * 0.8,
                    "yhat_upper": base_v * 1.2,
                }
            )
    pred_df = pd.DataFrame(pred_rows)

    traj = sample_trajectories(
        pred_df,
        history_df=full_hist,
        cfg=cfg,
        hier_spec=hier_spec,
        temp_spec=temp_spec,
        n_paths=40,
        copula_type="student_t",
        reconcile=True,
        floor=0.0,
        seed=11,
    )
    assert (traj["yhat_sample"] >= 0.0).all()
    assert verify_coherence(traj, hier_spec) < 1e-6
    assert verify_temporal_coherence(traj) < 1e-6
    assert verify_cross_temporal_coherence(traj, hier_spec, temp_spec) < 1e-6

    # Evaluate trajectory paths against realized actuals.
    actuals = pred_df[["ts_id", "forecast_date", "yhat"]].rename(columns={"yhat": "y"})
    scores = evaluate_trajectory_paths(traj, actuals)
    assert scores["crps"] >= 0.0
    assert scores["energy_score"] >= 0.0
    assert scores["variogram_score"] >= 0.0


def test_shrink_correlation_toward_similarity_and_regularized_copula_sampling() -> None:
    """Verify F7 (`shrink_correlation_toward_similarity` and `similarity_shrinkage=True` in
    `sample_trajectories`): regularizes noisy short-OOF cross-series correlations toward the PSD
    pairwise trajectory similarity matrix.
    """
    from scale_forecasting.copula import shrink_correlation_toward_similarity

    emp = np.array(
        [
            [1.0, -0.35, 0.10],
            [-0.35, 1.0, -0.20],
            [0.10, -0.20, 1.0],
        ],
        dtype=float,
    )
    sim = pd.DataFrame(
        [
            [1.0, 0.90, 0.25],
            [0.90, 1.0, 0.30],
            [0.25, 0.30, 1.0],
        ],
        index=["sku_0", "sku_1", "sku_2"],
        columns=["sku_0", "sku_1", "sku_2"],
    )

    shrunk_auto = shrink_correlation_toward_similarity(
        emp,
        sim,
        n_samples=3,
        weight="auto",
        series_ids=["sku_0", "sku_1", "sku_2"],
    )
    assert shrunk_auto.shape == (3, 3)
    np.testing.assert_allclose(np.diag(shrunk_auto), 1.0, atol=1e-7)
    np.testing.assert_allclose(shrunk_auto, shrunk_auto.T, atol=1e-7)
    eigvals = np.linalg.eigvalsh(shrunk_auto)
    assert np.all(eigvals > 0.0)
    # With n_samples=3, p=3 -> auto weight = 3 / (3 + 3) = 0.50, pulling (0, 1) above emp[0, 1]
    assert float(shrunk_auto[0, 1]) > float(emp[0, 1]) + 0.40

    shrunk_zero = shrink_correlation_toward_similarity(emp, sim, weight=0.0)
    shrunk_one = shrink_correlation_toward_similarity(
        emp,
        sim,
        weight=1.0,
        series_ids=["sku_0", "sku_1", "sku_2"],
    )
    assert abs(float(shrunk_zero[0, 1]) - float(emp[0, 1])) < 1e-4
    assert abs(float(shrunk_one[0, 1]) - 0.90) < 1e-4

    # Verify sample_trajectories with cross_series=True and similarity_shrinkage=True vs False
    hist_dates = pd.date_range("2025-01-01", periods=30, freq="D")
    f_dates = pd.date_range("2025-01-31", periods=7, freq="D")
    t_idx = np.arange(len(hist_dates), dtype=float)
    hist_rows: list[dict[str, object]] = []
    pred_rows: list[dict[str, object]] = []
    for i, sid in enumerate(["sku_0", "sku_1", "sku_2"]):
        base_wave = 50.0 + 12.0 * np.sin(2.0 * np.pi * t_idx / 7.0) if i < 2 else 30.0 + 0.8 * t_idx
        for dt, yv in zip(hist_dates, base_wave, strict=True):
            hist_rows.append({"ts_id": sid, "ds": dt, "y": float(yv)})
        for dt in f_dates:
            pred_rows.append(
                {
                    "ts_id": sid,
                    "model_type": "theta",
                    "forecast_date": dt,
                    "ds": dt,
                    "yhat": 50.0,
                    "yhat_lower": 40.0,
                    "yhat_upper": 60.0,
                }
            )
    hist_df = pd.DataFrame(hist_rows)
    pred_df = pd.DataFrame(pred_rows)

    traj_shrunk = sample_trajectories(
        pred_df,
        history_df=hist_df,
        n_paths=100,
        cross_series=True,
        similarity_shrinkage=True,
        similarity_weight=0.55,
        seed=42,
    )
    traj_unshrunk = sample_trajectories(
        pred_df,
        history_df=hist_df,
        n_paths=100,
        cross_series=True,
        similarity_shrinkage=False,
        seed=42,
    )
    assert abs(float(traj_shrunk.attrs["similarity_shrinkage_weight"]) - 0.55) < 1e-6
    assert float(traj_unshrunk.attrs["similarity_shrinkage_weight"]) == 0.0
    corr_shrunk = np.asarray(traj_shrunk.attrs["cross_series_correlation"], dtype=float)
    assert corr_shrunk.shape == (3, 3)
    assert float(corr_shrunk[0, 1]) > float(corr_shrunk[0, 2])
