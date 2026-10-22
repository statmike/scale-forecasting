"""Unit tests for compositional simplex transforms (`clr`, `ilr`) and cross-series closure."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scale_forecasting.config import RunConfig
from scale_forecasting.engines.ray_io import chunk_cells
from scale_forecasting.engines.spark_io import bucket_key_cols, run_group
from scale_forecasting.errors import ConfigError, DataError
from scale_forecasting.features import (
    apply_transform,
    fit_transform_lambda,
    invert_transform,
    resolve_model_variant,
)
from scale_forecasting.simplex import (
    SimplexSpec,
    apply_series_simplex_transform,
    close_simplex_cells,
    close_simplex_forecasts,
    clr_inverse,
    clr_transform,
    detect_simplex_sum,
    forecast_simplex_panel,
    has_simplex_models,
    helmert_contrast_matrix,
    ilr_inverse,
    ilr_transform,
    invert_series_simplex_transform,
    multiplicative_replacement,
    verify_simplex,
    verify_simplex_coherence,
)
from scale_forecasting.worker import run_cell, run_panel


def _simplex_panel(
    n_parts: int = 4,
    n_periods: int = 36,
    simplex_sum: float = 1.0,
    with_zeros: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_periods, freq="MS")
    alpha = np.linspace(1.5, 3.5, n_parts)
    shares = rng.dirichlet(alpha, size=n_periods) * simplex_sum
    if with_zeros:
        shares[0, 0] = 0.0
        shares[0, 1:] = shares[0, 1:] * (simplex_sum / np.sum(shares[0, 1:]))
        shares[5, -1] = 0.0
        shares[5, :-1] = shares[5, :-1] * (simplex_sum / np.sum(shares[5, :-1]))
    rows = []
    for j in range(n_parts):
        tid = f"part_{j}"
        for t_idx, d in enumerate(dates):
            rows.append({"ts_id": tid, "ds": d, "y": float(shares[t_idx, j])})
    return pd.DataFrame(rows)


def test_detect_simplex_sum_and_helmert_orthonormality() -> None:
    assert detect_simplex_sum([0.2, 0.3, 0.5]) == 1.0
    assert detect_simplex_sum([20.0, 30.0, 50.0]) == 100.0
    assert detect_simplex_sum([0.2, 0.8], simplex_sum=10.0) == 10.0

    with pytest.raises(ConfigError):
        detect_simplex_sum([0.2, 0.8], simplex_sum=0.0)
    with pytest.raises(ConfigError):
        detect_simplex_sum([0.2, 0.8], simplex_sum=-5.0)

    for d in (2, 3, 5, 8):
        v = helmert_contrast_matrix(d)
        assert v.shape == (d, d - 1)
        # Orthonormal columns: V^T V = I_{D-1}
        np.testing.assert_allclose(v.T @ v, np.eye(d - 1), atol=1e-12)
        # Zero column sums: 1^T V = 0
        np.testing.assert_allclose(np.sum(v, axis=0), np.zeros(d - 1), atol=1e-12)
        # Centering projector: V V^T = I_D - (1/D) 1 1^T
        centering = np.eye(d) - np.ones((d, d)) / float(d)
        np.testing.assert_allclose(v @ v.T, centering, atol=1e-12)

    with pytest.raises(ConfigError):
        helmert_contrast_matrix(1)


def test_multiplicative_zero_replacement_preserves_sums_and_positivity() -> None:
    x = np.array(
        [
            [0.0, 0.4, 0.6],
            [0.2, 0.5, 0.3],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )
    replaced = multiplicative_replacement(x, simplex_sum=1.0, eps=1e-6)
    assert replaced.shape == x.shape
    assert bool(np.all(replaced > 0.0))
    np.testing.assert_allclose(np.sum(replaced, axis=1), np.ones(4), atol=1e-12)
    # Row 1 has no zeros -> unchanged
    np.testing.assert_allclose(replaced[1], x[1], atol=1e-12)
    # Row 0 has c=1 zero -> replaced by eps, non-zeros scaled by (1 - eps)
    assert pytest.approx(replaced[0, 0], abs=1e-12) == 1e-6
    assert pytest.approx(replaced[0, 1], abs=1e-12) == 0.4 * (1.0 - 1e-6)
    # Row 3 all zeros -> uniform 1/3
    np.testing.assert_allclose(replaced[3], np.full(3, 1.0 / 3.0), atol=1e-12)


@pytest.mark.parametrize("simplex_sum", [1.0, 100.0])
def test_clr_and_ilr_exact_round_trip_with_zeros(simplex_sum: float) -> None:
    rng = np.random.default_rng(12)
    x_interior = rng.dirichlet([2.0, 3.0, 1.5, 2.5], size=20) * simplex_sum

    # Exact interior round-trip
    z_clr = clr_transform(x_interior, simplex_sum=simplex_sum)
    assert z_clr.shape == (20, 4)
    np.testing.assert_allclose(np.sum(z_clr, axis=1), np.zeros(20), atol=1e-10)
    x_clr_back = clr_inverse(z_clr, simplex_sum=simplex_sum)
    np.testing.assert_allclose(x_clr_back, x_interior, atol=1e-9)

    z_ilr, basis = ilr_transform(x_interior, simplex_sum=simplex_sum, return_basis=True)
    assert z_ilr.shape == (20, 3)
    x_ilr_back = ilr_inverse(z_ilr, simplex_sum=simplex_sum, basis=basis)
    np.testing.assert_allclose(x_ilr_back, x_interior, atol=1e-9)

    # Boundary zeros (0.0 and simplex_sum)
    x_boundary = x_interior.copy()
    x_boundary[0] = np.array([0.0, 0.0, 0.4 * simplex_sum, 0.6 * simplex_sum])
    x_boundary[1] = np.array([simplex_sum, 0.0, 0.0, 0.0])

    x_b_clr = clr_inverse(
        clr_transform(x_boundary, simplex_sum=simplex_sum), simplex_sum=simplex_sum
    )
    assert bool(np.all(x_b_clr > 0.0))
    np.testing.assert_allclose(np.sum(x_b_clr, axis=1), np.full(20, simplex_sum), atol=1e-9)
    np.testing.assert_allclose(x_b_clr, x_boundary, atol=1e-3 * simplex_sum)

    x_b_ilr = ilr_inverse(
        ilr_transform(x_boundary, simplex_sum=simplex_sum), simplex_sum=simplex_sum
    )
    assert bool(np.all(x_b_ilr > 0.0))
    np.testing.assert_allclose(np.sum(x_b_ilr, axis=1), np.full(20, simplex_sum), atol=1e-9)
    np.testing.assert_allclose(x_b_ilr, x_boundary, atol=1e-3 * simplex_sum)


def test_single_series_simplex_fallback_and_features_integration() -> None:
    s = pd.Series([0.0, 0.25, 0.5, 0.75, 1.0])
    z = apply_series_simplex_transform(s, transform="ilr", simplex_sum=1.0)
    assert bool(np.all(np.isfinite(z)))
    s_back = invert_series_simplex_transform(z, transform="ilr", simplex_sum=1.0)
    assert bool(np.all((s_back > 0.0) & (s_back < 1.0)))
    np.testing.assert_allclose(s_back[1:4], s.iloc[1:4].to_numpy(), atol=1e-5)

    # Verify features.py integration (`fit_transform_lambda`, `apply_transform`, `invert_transform`)
    lam = fit_transform_lambda(pd.Series([20.0, 50.0, 80.0]), "clr", simplex_sum=100.0)
    assert lam == 100.0
    transformed = apply_transform(pd.Series([20.0, 50.0, 80.0]), "clr", lam)
    recovered = invert_transform(transformed, "clr", lam)
    np.testing.assert_allclose(recovered, [20.0, 50.0, 80.0], atol=1e-5)

    # Verify variant tag parsing (`@transform_ilr`, `@transform_clr`, `simplex_sum`)
    cfg = RunConfig.model_validate(
        {
            "run_name": "var-simplex",
            "data": {"source_table": "p.d.t", "horizon": 6},
            "models": ["naive_mean@transform_ilr", "regression_lags@transform_clr"],
            "features": {"simplex_sum": 100.0},
        }
    )
    v1 = resolve_model_variant(cfg, "naive_mean@transform_ilr")
    assert v1.effective_cfg.features.transform == "ilr"
    v2 = resolve_model_variant(cfg, "regression_lags@transform_clr")
    assert v2.effective_cfg.features.transform == "clr"
    assert v2.effective_cfg.features.simplex_sum == 100.0
    assert has_simplex_models(cfg) is True


@pytest.mark.parametrize("transform", ["clr", "ilr"])
@pytest.mark.parametrize("simplex_sum", [1.0, 100.0])
def test_forecast_simplex_panel_and_closure_invariants(
    transform: str,
    simplex_sum: float,
) -> None:
    panel = _simplex_panel(n_parts=4, n_periods=30, simplex_sum=simplex_sum, with_zeros=True)
    preds, spec = forecast_simplex_panel(
        panel,
        horizon=6,
        transform=transform,  # type: ignore[arg-type]
        simplex_sum=simplex_sum,
        return_spec=True,
    )
    assert isinstance(spec, SimplexSpec)
    assert spec.simplex_sum == simplex_sum
    assert len(preds) == 4 * 6
    assert bool((preds["yhat"] > 0.0).all())
    assert bool((preds["yhat"] < simplex_sum).all())
    assert bool((preds["yhat_lower"] >= 0.0).all())
    assert bool((preds["yhat_upper"] <= simplex_sum).all())
    assert bool((preds["yhat_lower"] <= preds["yhat"] + 1e-9).all())
    assert bool((preds["yhat"] <= preds["yhat_upper"] + 1e-9).all())

    disc = verify_simplex_coherence(preds, simplex_sum=simplex_sum)
    assert disc < 1e-6
    assert verify_simplex(preds, simplex_sum=simplex_sum) < 1e-6

    # Check raise_on_error when simplex sum is violated
    bad = preds.copy()
    bad["yhat"] = bad["yhat"] * 1.5
    with pytest.raises(DataError):
        verify_simplex_coherence(bad, simplex_sum=simplex_sum, atol=1e-6, raise_on_error=True)

    # Check enforce_simplex_closure / close_simplex_forecasts repairs bad predictions
    repaired = close_simplex_forecasts(bad, simplex_sum=simplex_sum)
    assert verify_simplex_coherence(repaired, simplex_sum=simplex_sum) < 1e-6


def test_worker_and_engine_simplex_closure_end_to_end() -> None:
    panel = _simplex_panel(n_parts=3, n_periods=30, simplex_sum=100.0, with_zeros=True)
    cfg = RunConfig.model_validate(
        {
            "run_name": "simplex-e2e",
            "data": {"source_table": "p.d.t", "freq": "MS", "horizon": 6},
            "models": ["naive_mean", "regression_lags"],
            "features": {
                "transform": "ilr",
                "simplex_sum": 100.0,
            },
            "backtest": {"enabled": True, "n_folds": 1, "horizon": 6},
        }
    )
    assert bucket_key_cols(cfg) == ["_sf_model"]
    ray_chunks = chunk_cells(panel, cfg, ["naive_mean", "regression_lags"], n_chunks=8)
    assert len(ray_chunks) == 2

    # Local model cells + close_simplex_cells
    local_cells = [
        run_cell(sub, "naive_mean", cfg) for _, sub in panel.groupby("ts_id", sort=False)
    ]
    closed_local = close_simplex_cells(local_cells, cfg, history_df=panel)
    local_preds = pd.concat(
        [c.predictions.assign(ts_id=c.ts_id, model_type=c.model_type) for c in closed_local],
        ignore_index=True,
    )
    assert verify_simplex_coherence(local_preds, simplex_sum=100.0) < 1e-6

    # Global/panel model (`run_panel` applies `close_simplex_cells` automatically)
    panel_cells = run_panel(panel, "regression_lags", cfg)
    panel_preds = pd.concat(
        [c.predictions.assign(ts_id=c.ts_id, model_type=c.model_type) for c in panel_cells],
        ignore_index=True,
    )
    assert verify_simplex_coherence(panel_preds, simplex_sum=100.0) < 1e-6

    # Spark run_group end-to-end
    group_cells, status_df = run_group(panel, cfg, models=["naive_mean"])
    assert len(status_df) == 3
    assert set(status_df["status"]) == {"ok"}
    group_preds = pd.concat(
        [c.predictions.assign(ts_id=c.ts_id, model_type=c.model_type) for c in group_cells],
        ignore_index=True,
    )
    assert verify_simplex_coherence(group_preds, simplex_sum=100.0) < 1e-6


def test_auto_forecaster_compositional_simplex_detection_and_branch_2a() -> None:
    import scale_forecasting as sf
    from scale_forecasting.auto import inspect_panel_for_auto
    from scale_forecasting.simplex import (
        SIMPLICIAL_TRANSFORMS,
        apply_simplex_series_transform,
        close_simplex_predictions,
        inverse_clr_transform,
        inverse_ilr_transform,
        inverse_simplex_series_transform,
        replace_compositional_zeros,
        should_close_simplex,
        verify_simplex_closure,
    )

    assert set(SIMPLICIAL_TRANSFORMS) == {"clr", "ilr"}
    assert verify_simplex_closure is verify_simplex_coherence
    assert replace_compositional_zeros is multiplicative_replacement
    assert inverse_clr_transform is clr_inverse
    assert inverse_ilr_transform is ilr_inverse
    assert apply_simplex_series_transform is apply_series_simplex_transform
    assert inverse_simplex_series_transform is invert_series_simplex_transform
    assert close_simplex_predictions is close_simplex_forecasts

    panel = _simplex_panel(n_parts=3, n_periods=36, simplex_sum=100.0, with_zeros=True)
    profile = inspect_panel_for_auto(panel, horizon=6, freq="MS")
    assert isinstance(profile, sf.DataProfile)
    assert profile.is_compositional_simplex is True
    assert profile.simplex_total == 100.0
    prof_df = profile.to_dataframe()
    assert "compositional_simplex" in set(prof_df["signal"])

    bp = sf.Forecaster.build_config(panel, horizon=6, freq="MS", tier="first_light")
    assert bp.config.features.simplex_sum == 100.0
    assert any("@transform_ilr" in m or "@transform_clr" in m for m in bp.config.models)
    assert should_close_simplex(bp.config) is True

    f_auto = sf.Forecaster.auto(
        panel,
        horizon=6,
        freq="MS",
        tier="first_light",
        autofocus=True,
        max_passes=2,
    )
    assert f_auto._auto_session is not None
    preds = f_auto._auto_session.predictions_df
    assert not preds.empty
    for _m, grp in preds.groupby("model_type"):
        assert verify_simplex_coherence(grp, simplex_sum=100.0) < 1e-5
