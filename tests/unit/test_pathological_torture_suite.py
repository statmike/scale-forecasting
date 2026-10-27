"""12-Case Pathological Input & 100k-Series Canopy Torture Suite (Track F.2 / v1.8.2).

Verifies that real-world malformed, degenerate, or extreme time-series panels never trigger
unhandled division-by-zero, ``LinAlgError``, ``NaN`` propagation, or out-of-memory failures:

1. All-zero series (``y = 0`` for all ``t``) across ``naive_last``, ``croston``, ``sba``,
   ``tsb``, ``theta``, and ``lightgbm``.
2. Single-observation ($T = 1$) and two-observation ($T = 2$) series via ``run_cell``.
3. Interior and trailing ``NaN`` / ``±inf`` blocks sanitized by ``regularize_series_grid``
   and ``clean_training_target``.
4. Constant non-zero series (``y = 42.0``, $\\sigma = 0$) across z-score, STL invariants,
   FFT/Hilbert similarity, and conformal calibration.
5. Extreme dynamic range ($y \\in [10^{-9}, 10^{12}]$) across ``rmsle``, ``wape``, ``mase``,
   and ``lightgbm``.
6. Negative target values ($y < 0$, e.g., net returns or negative electricity prices) across
   transforms (``boxcox``, ``log1p``, ``none``), models, and metrics.
7. Duplicate timestamps (``(ts_id, ds)`` collisions) and unsorted timestamp ordering.
8. Gapped / irregular timestamps reconstructed by ``grid_policy`` (``fill_zero``, ``ffill``,
   ``linear``).
9. Collinear exogenous covariates ($X_2 = 2 X_1$, rank-deficient design matrix) across
   ``regression_lags@monotone``, ``compute_promo_elasticity_decomposition``, and
   ``sample_trajectories`` copula covariance.
10. Degenerate compositional simplex (100% share on one component, 0% on all others) in
    ``@transform_ilr`` and ``@transform_clr``.
11. Unachievable / tight Goal-Seek QP bounds and zero-budget Inventory optimization.
12. 100k-series ($N = 100,000, T = 24$) Canopy Similarity scaling in $< 30\\text{s}$ and
    $< 250\\text{ MB}$ peak memory.
"""

from __future__ import annotations

import time
import tracemalloc

import numpy as np
import pandas as pd
import pytest

from scale_forecasting.calibration import calibrate_from_oof
from scale_forecasting.config import RunConfig
from scale_forecasting.copula import sample_trajectories
from scale_forecasting.elasticity import compute_promo_elasticity_decomposition
from scale_forecasting.errors import ConfigError, GoalSeekInfeasibleError
from scale_forecasting.features import (
    apply_transform,
    build_features,
    clean_training_target,
    fit_transform_lambda,
    regularize_series_grid,
)
from scale_forecasting.goal_seek import allocate_target
from scale_forecasting.inventory_optimizer import optimize_inventory_df
from scale_forecasting.metrics import compute_metrics
from scale_forecasting.similarity import (
    cluster_by_similarity_df,
    compute_similarity_matrix,
    embed_series_df,
)
from scale_forecasting.simplex import (
    clr_inverse,
    clr_transform,
    enforce_simplex_closure,
    ilr_inverse,
    ilr_transform,
)
from scale_forecasting.worker import run_cell


def _make_base_cfg(
    models: list[str] | None = None,
    *,
    horizon: int = 4,
    freq: str = "D",
    transform: str = "none",
    grid_policy: str = "strict",
    outlier_policy: str = "none",
    exog: list[str] | None = None,
    monotonic_constraints: dict[str, int] | None = None,
) -> RunConfig:
    payload: dict[str, object] = {
        "run_name": "torture-test",
        "data": {
            "source_table": "proj.ds.tbl",
            "ts_id_col": "ts_id",
            "date_col": "ds",
            "target_col": "y",
            "freq": freq,
            "horizon": horizon,
            "grid_policy": grid_policy,
        },
        "features": {
            "transform": transform,
            "outlier_policy": outlier_policy,
            "exog": exog or [],
            "monotonic_constraints": monotonic_constraints or {},
        },
        "models": models or ["naive_mean"],
        "backtest": {"enabled": False},
    }
    return RunConfig.model_validate(payload)


# ---------------------------------------------------------------------------
# Case 1: All-Zero Series (y = 0 for all t)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model_name", "extra_params"),
    [
        ("naive_mean", {}),
        ("croston", {"variant": "classic"}),
        ("croston", {"variant": "sba"}),
        ("croston", {"variant": "tsb"}),
        ("theta", {}),
        ("lightgbm", {}),
    ],
)
def test_case_01_all_zero_series(model_name: str, extra_params: dict[str, object]) -> None:
    """All-zero series must return finite non-NaN predictions without ZeroDivisionError."""
    dates = pd.date_range("2025-01-01", periods=32, freq="D")
    series = pd.DataFrame({"ts_id": "zero_sku", "ds": dates, "y": np.zeros(len(dates))})
    cfg = RunConfig.model_validate(
        {
            "run_name": "torture-zero",
            "data": {
                "source_table": "proj.ds.tbl",
                "ts_id_col": "ts_id",
                "date_col": "ds",
                "target_col": "y",
                "freq": "D",
                "horizon": 4,
            },
            "models": [model_name],
            "model_params": {model_name: extra_params} if extra_params else {},
            "backtest": {"enabled": False},
        }
    )
    res = run_cell(series, model_name, cfg)
    assert res.status == "ok", (
        f"Unexpected failure on all-zero series for {model_name}: {res.error}"
    )
    assert len(res.predictions) == 4
    for col in ("yhat", "yhat_lower", "yhat_upper"):
        vals = res.predictions[col].to_numpy(dtype=float)
        assert np.all(np.isfinite(vals)), f"Non-finite {col} for {model_name}: {vals}"


# ---------------------------------------------------------------------------
# Case 2: Single-Observation (T = 1) & Two-Observation (T = 2) Series
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_obs", [1, 2])
@pytest.mark.parametrize("model_name", ["naive_last", "naive_mean", "theta", "lightgbm"])
def test_case_02_ultra_short_series_never_crashes_worker(n_obs: int, model_name: str) -> None:
    """T=1 and T=2 series must produce finite forecasts or a clean CellResult(status='error')."""
    dates = pd.date_range("2025-01-01", periods=n_obs, freq="D")
    series = pd.DataFrame(
        {"ts_id": f"short_{n_obs}", "ds": dates, "y": np.full(n_obs, 15.0, dtype=float)}
    )
    cfg = _make_base_cfg([model_name], horizon=3)
    res = run_cell(series, model_name, cfg)
    assert res.status in ("ok", "error")
    if res.status == "ok":
        assert len(res.predictions) == 3
        assert np.all(np.isfinite(res.predictions["yhat"].to_numpy(dtype=float)))
    else:
        assert res.error is not None and len(res.error) > 0


# ---------------------------------------------------------------------------
# Case 3: Interior and Trailing NaN / ±inf Blocks
# ---------------------------------------------------------------------------


def test_case_03_interior_and_trailing_nan_inf_blocks() -> None:
    """regularize_series_grid and clean_training_target must sanitize NaN and ±inf blocks."""
    dates = pd.date_range("2025-01-01", periods=20, freq="D")
    y_vals = np.linspace(10.0, 20.0, 20)
    y_vals[3] = np.nan
    y_vals[4] = np.nan
    y_vals[8] = np.inf
    y_vals[9] = -np.inf
    y_vals[-1] = np.nan
    series = pd.DataFrame({"ts_id": "dirty_sku", "ds": dates, "y": y_vals})

    cfg = _make_base_cfg(
        ["naive_mean", "theta"],
        horizon=4,
        grid_policy="linear",
        outlier_policy="winsorize",
    )
    reg = regularize_series_grid(series, cfg)
    assert np.all(np.isfinite(reg["y"].to_numpy(dtype=float)))

    cleaned_y, _ = clean_training_target(
        pd.Series(y_vals, index=dates, name="y"),
        series.set_index("ds"),
        cfg,
    )
    assert np.all(np.isfinite(cleaned_y.to_numpy(dtype=float)))

    for m in ("naive_mean", "theta"):
        res = run_cell(series, m, cfg)
        assert res.status == "ok"
        assert np.all(np.isfinite(res.predictions["yhat"].to_numpy(dtype=float)))


# ---------------------------------------------------------------------------
# Case 4: Constant Non-Zero Series (y = 42.0, zero variance sigma = 0)
# ---------------------------------------------------------------------------


def test_case_04_constant_nonzero_series_zero_variance() -> None:
    """Zero-variance constant series must not divide by zero in invariants or calibration."""
    dates = pd.date_range("2025-01-01", periods=30, freq="D")
    panel = pd.concat(
        [
            pd.DataFrame({"ts_id": "const_a", "ds": dates, "y": np.full(30, 42.0)}),
            pd.DataFrame({"ts_id": "const_b", "ds": dates, "y": np.full(30, 42.0)}),
            pd.DataFrame({"ts_id": "const_c", "ds": dates, "y": np.full(30, 100.0)}),
        ],
        ignore_index=True,
    )
    inv_df = embed_series_df(panel, method="structural_signature", freq="D")
    assert len(inv_df) == 3
    num_cols = [
        c for c in inv_df.columns if c != "ts_id" and pd.api.types.is_numeric_dtype(inv_df[c])
    ]
    assert len(num_cols) > 0
    for col in num_cols:
        assert np.all(np.isfinite(inv_df[col].to_numpy(dtype=float)))

    for lens in ("hybrid", "physics_attractor", "spectral_phase", "elastic_dtw", "minirocket"):
        sim_res = compute_similarity_matrix(panel, method=lens, freq="D")
        mat = sim_res.matrix_df.to_numpy(dtype=float)
        assert np.all(np.isfinite(mat)), f"Non-finite similarity for lens={lens}"
        assert np.allclose(np.diag(mat), 1.0)

    # Conformal calibration on zero-error constant OOF (10 rows >= MIN_RESIDUALS_TOTAL)
    oof = pd.DataFrame(
        {
            "fold_id": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
            "ds": dates[:10],
            "y_true": np.full(10, 42.0),
            "yhat": np.full(10, 42.0),
            "yhat_raw": np.full(10, 42.0),
            "yhat_lower": np.full(10, 42.0),
            "yhat_upper": np.full(10, 42.0),
            "horizon_step": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        }
    )
    cal = calibrate_from_oof(oof, quantiles=(0.1, 0.9))
    assert cal is not None
    assert np.isfinite(cal.pooled.mean)
    for q_val in cal.pooled.quantiles.values():
        assert np.isfinite(q_val)


# ---------------------------------------------------------------------------
# Case 5: Extreme Dynamic Range (y in [1e-9, 1e12])
# ---------------------------------------------------------------------------


def test_case_05_extreme_dynamic_range() -> None:
    """Extreme dynamic range [1e-9, 1e12] must not overflow float64 in metrics or LightGBM."""
    dates = pd.date_range("2025-01-01", periods=36, freq="D")
    y_micro = np.full(36, 1e-9, dtype=float) + np.linspace(0.0, 1e-9, 36)
    y_macro = np.full(36, 1e11, dtype=float) + np.linspace(0.0, 9e11, 36)

    for scale_label, y_arr in (("micro", y_micro), ("macro", y_macro)):
        metrics = compute_metrics(
            y_true=y_arr[-6:],
            yhat=y_arr[-6:] * 1.02,
            y_train=y_arr[:-6],
            lower=y_arr[-6:] * 0.90,
            upper=y_arr[-6:] * 1.10,
            seasonal_period=7,
        )
        for m_name in ("wape", "smape", "mae", "rmse", "rmsle", "mase"):
            assert np.isfinite(metrics[m_name]), f"Non-finite {m_name} at {scale_label}"

        series = pd.DataFrame({"ts_id": scale_label, "ds": dates, "y": y_arr})
        cfg = _make_base_cfg(["lightgbm"], horizon=4)
        res = run_cell(series, "lightgbm", cfg)
        assert res.status == "ok"
        assert np.all(np.isfinite(res.predictions["yhat"].to_numpy(dtype=float)))


# ---------------------------------------------------------------------------
# Case 6: Negative Target Values (y < 0)
# ---------------------------------------------------------------------------


def test_case_06_negative_target_values() -> None:
    """Negative y raises ConfigError on boxcox/log1p(y<-1) and succeeds with transform='none'."""
    dates = pd.date_range("2025-01-01", periods=30, freq="D")
    y_neg = np.sin(np.linspace(0.0, 4.0 * np.pi, 30)) * 25.0 - 10.0  # includes values < -1
    y_series = pd.Series(y_neg, index=dates, name="y")

    with pytest.raises(ConfigError, match="boxcox"):
        fit_transform_lambda(y_series, "boxcox")

    with pytest.raises(ConfigError, match="log1p"):
        apply_transform(y_series, "log1p")

    series = pd.DataFrame({"ts_id": "neg_price", "ds": dates, "y": y_neg})
    cfg = _make_base_cfg(["naive_mean", "theta", "lightgbm"], horizon=4, transform="none")
    for m in ("naive_mean", "theta", "lightgbm"):
        res = run_cell(series, m, cfg)
        assert res.status == "ok"
        assert np.all(np.isfinite(res.predictions["yhat"].to_numpy(dtype=float)))

    metrics = compute_metrics(
        y_true=y_neg[-4:],
        yhat=y_neg[-4:] + 1.5,
        y_train=y_neg[:-4],
        lower=y_neg[-4:] - 5.0,
        upper=y_neg[-4:] + 5.0,
    )
    for m_name in ("wape", "smape", "mae", "rmse"):
        assert np.isfinite(metrics[m_name])


# ---------------------------------------------------------------------------
# Case 7: Duplicate Timestamps & Unsorted Timestamp Ordering
# ---------------------------------------------------------------------------


def test_case_07_duplicate_and_unsorted_timestamps() -> None:
    """regularize_series_grid sorts unsorted timestamps and deduplicates (ts_id, ds) collisions."""
    dates = pd.date_range("2025-01-01", periods=15, freq="D")
    df = pd.DataFrame({"ts_id": "dup_sku", "ds": dates, "y": np.arange(1.0, 16.0)})
    # Inject a duplicate date at 2025-01-05 with updated value 99.0 and scramble row order
    dup_row = pd.DataFrame({"ts_id": ["dup_sku"], "ds": [pd.Timestamp("2025-01-05")], "y": [99.0]})
    scrambled = pd.concat([df.iloc[::-1], dup_row], ignore_index=True)

    cfg = _make_base_cfg(["naive_last"], horizon=3, grid_policy="ffill")
    reg = regularize_series_grid(scrambled, cfg)
    assert len(reg) == 15
    assert reg["ds"].is_monotonic_increasing
    assert not reg["ds"].duplicated().any()
    val_jan5 = float(reg.loc[reg["ds"] == pd.Timestamp("2025-01-05"), "y"].iloc[0])
    assert val_jan5 == pytest.approx(99.0)


# ---------------------------------------------------------------------------
# Case 8: Gapped / Irregular Timestamps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("policy", ["fill_zero", "ffill", "linear"])
def test_case_08_gapped_irregular_timestamps(policy: str) -> None:
    """grid_policy must reconstruct a contiguous date_range across missing interior dates."""
    full_dates = pd.date_range("2025-01-01", periods=20, freq="D")
    # Drop 4 interior dates
    keep_mask = np.ones(20, dtype=bool)
    keep_mask[[4, 5, 11, 12]] = False
    gapped = pd.DataFrame(
        {
            "ts_id": "gap_sku",
            "ds": full_dates[keep_mask],
            "y": np.linspace(10.0, 29.0, 20)[keep_mask],
        }
    )
    cfg = _make_base_cfg(["naive_last"], horizon=3, grid_policy=policy)
    y_built, _ = build_features(gapped, cfg)
    assert len(y_built) == 20
    assert y_built.index.is_monotonic_increasing
    assert np.all(np.isfinite(y_built.to_numpy(dtype=float)))


# ---------------------------------------------------------------------------
# Case 9: Collinear Exogenous Covariates (X_2 = 2 * X_1)
# ---------------------------------------------------------------------------


def test_case_09_collinear_exogenous_covariates() -> None:
    """Rank-deficient collinear covariates (X2 = 2*X1) must not raise LinAlgError."""
    rng = np.random.default_rng(99)
    dates = pd.date_range("2025-01-01", periods=48, freq="D")
    x1 = rng.uniform(5.0, 15.0, size=48)
    x2 = 2.0 * x1  # Exact collinearity
    promo = (rng.uniform(0.0, 1.0, size=48) > 0.7).astype(float)
    y = 100.0 - 2.0 * x1 + 15.0 * promo + rng.normal(0.0, 0.5, size=48)

    series = pd.DataFrame(
        {"ts_id": "collinear_sku", "ds": dates, "y": y, "x1": x1, "x2": x2, "promo": promo}
    )
    cfg = _make_base_cfg(
        ["regression_lags@monotone"],
        horizon=4,
        exog=["x1", "x2", "promo"],
        monotonic_constraints={"x1": -1, "x2": -1, "promo": 1},
    )
    res = run_cell(series, "regression_lags@monotone", cfg)
    assert res.status == "ok", f"regression_lags@monotone failed on collinear X: {res.error}"
    assert np.all(np.isfinite(res.predictions["yhat"].to_numpy(dtype=float)))

    # Double-ML elasticity with collinear controls
    elast_df = compute_promo_elasticity_decomposition(
        series,
        cfg=cfg,
        price_col="x1",
        promo_col="promo",
        control_cols=["x2"],
    )
    assert len(elast_df) == 48
    assert np.all(np.isfinite(elast_df["net_causal_lift"].to_numpy(dtype=float)))
    assert np.all(np.isfinite(elast_df["incremental_promo_lift"].to_numpy(dtype=float)))

    # Copula trajectories with collinear identical residual series
    preds = pd.concat(
        [
            pd.DataFrame(
                {
                    "ts_id": uid,
                    "ds": pd.date_range("2025-02-18", periods=4, freq="D"),
                    "yhat": [50.0, 52.0, 51.0, 53.0],
                    "yhat_lower": [45.0, 47.0, 46.0, 48.0],
                    "yhat_upper": [55.0, 57.0, 56.0, 58.0],
                }
            )
            for uid in ("col_a", "col_b")
        ],
        ignore_index=True,
    )
    hist_col = pd.concat(
        [
            pd.DataFrame({"ts_id": "col_a", "ds": dates, "y": y}),
            pd.DataFrame({"ts_id": "col_b", "ds": dates, "y": y}),  # 100% collinear history
        ],
        ignore_index=True,
    )
    traj_df = sample_trajectories(
        preds,
        history_df=hist_col,
        n_paths=25,
        copula_type="gaussian",
        cross_series=True,
        random_seed=42,
    )
    assert not traj_df.empty
    assert np.all(np.isfinite(traj_df["yhat_sample"].to_numpy(dtype=float)))


# ---------------------------------------------------------------------------
# Case 10: Degenerate Compositional Simplex (100% share on 1 component, 0% on others)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("transform", ["ilr", "clr"])
def test_case_10_degenerate_compositional_simplex_boundary_zeros(transform: str) -> None:
    """100% share on one component and 0% on all others must not trigger log(0) = -inf."""
    dates = pd.date_range("2025-01-01", periods=10, freq="D")
    comp_mat = np.column_stack(
        [
            np.ones(10, dtype=float),
            np.zeros(10, dtype=float),
            np.zeros(10, dtype=float),
        ]
    )
    if transform == "ilr":
        z_mat = ilr_transform(comp_mat, simplex_sum=1.0)
        assert np.all(np.isfinite(z_mat))
        rec_mat = ilr_inverse(z_mat, simplex_sum=1.0)
    else:
        z_mat = clr_transform(comp_mat, simplex_sum=1.0)
        assert np.all(np.isfinite(z_mat))
        rec_mat = clr_inverse(z_mat, simplex_sum=1.0)

    assert np.all(np.isfinite(rec_mat))
    assert np.allclose(np.sum(rec_mat, axis=1), 1.0, atol=1e-6)

    panel = pd.concat(
        [
            pd.DataFrame({"ts_id": "brand_dom", "ds": dates, "yhat": np.ones(10, dtype=float)}),
            pd.DataFrame({"ts_id": "brand_zero_1", "ds": dates, "yhat": np.zeros(10, dtype=float)}),
            pd.DataFrame({"ts_id": "brand_zero_2", "ds": dates, "yhat": np.zeros(10, dtype=float)}),
        ],
        ignore_index=True,
    )
    closed = enforce_simplex_closure(panel, simplex_sum=1.0)
    sums_by_ds = closed.groupby("ds")["yhat"].sum().to_numpy(dtype=float)
    assert np.allclose(sums_by_ds, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Case 11: Contradictory Goal-Seek Bounds & Zero-Budget Inventory Optimization
# ---------------------------------------------------------------------------


def test_case_11_contradictory_goal_seek_and_zero_budget_inventory() -> None:
    """Goal-Seek QP with infeasible bounds raises GoalSeekInfeasibleError; budget=0 yields 0."""
    dates = pd.date_range("2025-01-01", periods=2, freq="D")
    bottom_df = pd.DataFrame(
        {
            "ts_id": ["Store_A", "Store_A", "Store_B", "Store_B"],
            "ds": [dates[0], dates[1], dates[0], dates[1]],
            "yhat": [40.0, 40.0, 60.0, 60.0],
            "yhat_lower": [30.0, 30.0, 50.0, 50.0],
            "yhat_upper": [50.0, 50.0, 70.0, 70.0],
        }
    )

    # Target (500.0) exceeds sum of upper bounds across the 2 days (2 * (50 + 70) = 240.0)
    with pytest.raises(GoalSeekInfeasibleError):
        allocate_target(
            bottom_df,
            target_total=500.0,
            bounds={"Store_A": (10.0, 50.0), "Store_B": (10.0, 70.0)},
        )

    # Inventory optimization under zero budget (max_budget = 0.0)
    inv_df = optimize_inventory_df(
        bottom_df,
        holding_cost=1.0,
        stockout_cost=10.0,
        unit_cost=5.0,
        max_budget=0.0,
    )
    assert not inv_df.empty
    assert np.all(np.isfinite(inv_df["recommended_order_qty"].to_numpy(dtype=float)))
    assert np.allclose(inv_df["recommended_order_qty"].to_numpy(dtype=float), 0.0)

    # Clearance space-reclamation optimization under physically unreachable target_space_freed
    from scale_forecasting import optimize_clearance

    cl_infeas = optimize_clearance(
        bottom_df,
        initial_inventory=50.0,
        unit_volume=1.0,
        target_space_freed=1000.0,
    )
    assert cl_infeas.status == "infeasible"
    assert not cl_infeas.is_feasible
    assert cl_infeas.relaxation_hint is not None


# ---------------------------------------------------------------------------
# Case 12: 100k-Series Canopy Similarity Scaling (< 30s, < 250 MB Peak Memory)
# ---------------------------------------------------------------------------


def test_case_12_100k_series_canopy_similarity_scaling() -> None:
    """N=100,000 series (T=24, 2.4M rows) must cluster in <30s and <250 MB peak memory."""
    rng = np.random.default_rng(42)
    n_series = 100_000
    t_len = 24
    ts_ids = np.repeat(np.arange(n_series, dtype=np.int32), t_len)
    base_dates = pd.date_range("2024-01-01", periods=t_len, freq="MS").to_numpy()
    ds = np.tile(base_dates, n_series)
    t_grid = np.linspace(0.0, 2.0 * np.pi, t_len, dtype=np.float32)
    archetypes = np.stack(
        [
            100.0 + 20.0 * np.sin(t_grid),
            100.0 + 20.0 * np.cos(t_grid),
            50.0 + 5.0 * np.arange(t_len, dtype=np.float32),
            200.0 - 4.0 * np.arange(t_len, dtype=np.float32),
        ]
    )
    labels = rng.integers(0, 4, size=n_series)
    y_mat = archetypes[labels] + rng.normal(0.0, 1.5, size=(n_series, t_len)).astype(np.float32)
    panel_df = pd.DataFrame({"ts_id": ts_ids, "ds": ds, "y": y_mat.ravel()})
    del y_mat, ts_ids, ds

    tracemalloc.start()
    t0 = time.perf_counter()
    clusters_df = cluster_by_similarity_df(
        panel_df,
        n_clusters=8,
        method="hybrid",
        freq="MS",
    )
    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak_bytes / (1024 * 1024)
    assert len(clusters_df) == n_series
    assert elapsed < 30.0, f"100k canopy clustering took {elapsed:.2f}s (limit: 30.0s)"
    assert peak_mb < 250.0, f"100k canopy clustering used {peak_mb:.1f} MB (limit: 250.0 MB)"
