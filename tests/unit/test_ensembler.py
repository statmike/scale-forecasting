"""Tests for the ensembler — calculated combine math, learned meta-learners, pandas blend.

Covers CONTRACTS §6 / DESIGN §5.2 / BUILD 6a: mean/median exact, inverse-error weights sum to
1, NNLS weights ≥ 0, the leakage guard (learned strategies refuse to run without backtest),
multi-strategy dispatch, and the pandas :func:`combine_calculated` blend that replaced the retired
``INSERT…SELECT`` SQL (C4 / Q4 fix — every append-only cell write now goes through the Write API).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from scale_forecasting.config import RunConfig
from scale_forecasting.ensembler import (
    combine_calculated,
    fit_learned,
    inverse_error_weights,
    mean_combine,
    median_combine,
)
from scale_forecasting.errors import ConfigError


def _cfg(strategies: list[str], *, backtest: bool = True, prune: float = 0.0) -> RunConfig:
    over: dict[str, Any] = {
        "run_name": "ens test",
        "data": {"source_table": "t"},
        "models": ["theta", "sarimax", "xgboost"],
        "ensemble": {"enabled": True, "strategies": strategies, "prune_threshold": prune},
        "backtest": {"enabled": backtest, "n_folds": 2, "decision_metric": "wape"},
    }
    return RunConfig(**over)


def _oof(models: list[str], n_per: int = 20, seed: int = 0) -> pd.DataFrame:
    """Synthetic long-format OOF: truth + each model's noisy forecast of it."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_per, freq="D")
    truth = np.linspace(10, 30, n_per) + rng.normal(0, 1, n_per)
    rows = []
    # model quality decreasing: theta best, then sarimax, then xgboost noisiest.
    noise = {"theta": 0.5, "sarimax": 1.5, "xgboost": 3.0}
    for m in models:
        yhat = truth + rng.normal(0, noise.get(m, 1.0), n_per)
        for d, yt, yh in zip(dates, truth, yhat, strict=True):
            rows.append(
                {"ts_id": "s1", "model_type": m, "fold_id": 0, "ds": d, "y_true": yt, "yhat": yh}
            )
    return pd.DataFrame(rows)


# --- calculated combine math ---------------------------------------------------


def test_mean_combine_exact() -> None:
    yhats = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    assert np.allclose(mean_combine(yhats), [2.0, 3.0, 4.0])


def test_median_combine_exact_and_robust() -> None:
    # third model is a wild outlier; median ignores it, mean would not.
    yhats = np.array([[10.0, 10.0], [12.0, 12.0], [1000.0, 1000.0]])
    assert np.allclose(median_combine(yhats), [12.0, 12.0])


def test_inverse_error_weights_sum_to_one() -> None:
    w = inverse_error_weights(np.array([1.0, 2.0, 4.0]))
    assert w.sum() == pytest.approx(1.0)
    # smaller error → larger weight
    assert w[0] > w[1] > w[2]


def test_inverse_error_zero_error_dominates() -> None:
    w = inverse_error_weights(np.array([0.0, 2.0, 4.0]))
    assert np.allclose(w, [1.0, 0.0, 0.0])


def test_inverse_error_all_nonfinite_is_uniform() -> None:
    w = inverse_error_weights(np.array([np.nan, np.inf]))
    assert np.allclose(w, [0.5, 0.5])


def test_inverse_error_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        inverse_error_weights(np.array([]))


# --- learned meta-learners -----------------------------------------------------


def test_nnls_weights_are_nonnegative() -> None:
    cfg = _cfg(["nnls"])
    weights, artifacts = fit_learned(_oof(cfg.models), cfg)
    assert set(weights) == {"nnls"}
    assert all(w >= 0.0 for w in weights["nnls"].values())
    assert "nnls" in artifacts and isinstance(artifacts["nnls"], bytes)


def test_learned_trusts_the_better_model_more() -> None:
    # theta is the least-noisy base model → should earn the largest nnls weight.
    cfg = _cfg(["nnls"])
    weights, _ = fit_learned(_oof(cfg.models, seed=7), cfg)
    w = weights["nnls"]
    assert w["theta"] >= w["sarimax"]
    assert w["theta"] >= w["xgboost"]


def test_ridge_returns_weight_per_model() -> None:
    cfg = _cfg(["ridge"])
    weights, _ = fit_learned(_oof(cfg.models), cfg)
    assert set(weights["ridge"]) == set(cfg.models)


def test_multi_strategy_fits_each_learned() -> None:
    cfg = _cfg(["nnls", "ridge"])
    weights, artifacts = fit_learned(_oof(cfg.models), cfg)
    assert set(weights) == {"nnls", "ridge"}
    assert set(artifacts) == {"nnls", "ridge"}


def test_calculated_only_config_fits_nothing() -> None:
    cfg = _cfg(["mean", "median"])
    weights, artifacts = fit_learned(_oof(cfg.models), cfg)
    assert weights == {} and artifacts == {}


# --- leakage guard -------------------------------------------------------------


def test_learned_without_backtest_is_rejected() -> None:
    # NOTE: RunConfig drops learned strategies when backtest is off, so construct the guard
    # scenario by asking fit_learned directly with a backtest-off config that still lists one.
    cfg = _cfg(["nnls"], backtest=True)
    off = cfg.model_copy(update={"backtest": cfg.backtest.model_copy(update={"enabled": False})})
    with pytest.raises(ConfigError, match="require backtest"):
        fit_learned(_oof(cfg.models), off)


def test_learned_with_empty_oof_is_rejected() -> None:
    cfg = _cfg(["nnls"])
    empty = pd.DataFrame(columns=["ts_id", "model_type", "fold_id", "ds", "y_true", "yhat"])
    with pytest.raises(ConfigError, match="empty"):
        fit_learned(empty, cfg)


def test_learned_missing_a_base_model_is_rejected() -> None:
    cfg = _cfg(["nnls"])
    partial = _oof(["theta", "sarimax"])  # xgboost absent from OOF
    with pytest.raises(ConfigError, match="missing base models"):
        fit_learned(partial, cfg)


# --- combine_calculated: the pandas blend (Write-API path, C4 / Q4 fix) --------


def _base_df(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    """Long-format base predictions (ts_id, model_type, forecast_date, yhat); bounds mirror yhat."""
    return pd.DataFrame(
        [
            {
                "ts_id": t,
                "model_type": m,
                "forecast_date": d,
                "yhat": y,
                "yhat_lower": y - 1.0,
                "yhat_upper": y + 1.0,
            }
            for (t, m, d, y) in rows
        ]
    )


def _metric_df(rows: list[tuple[str, float]], *, metric: str = "wape") -> pd.DataFrame:
    """Per-model run-level metric rows (model_type, <metric>) — the forecast_metadata subset."""
    return pd.DataFrame([{"model_type": m, metric: v} for (m, v) in rows])


def test_no_calculated_strategy_yields_no_rows() -> None:
    # a learned-only config produces nothing from the calculated blender.
    assert combine_calculated(_base_df([("s1", "theta", "d1", 10.0)]), _cfg(["nnls"])) == []


def test_empty_base_yields_no_rows() -> None:
    assert combine_calculated(_base_df([]), _cfg(["mean"])) == []


def test_mean_blends_base_predictions() -> None:
    base = _base_df([("s1", "theta", "d1", 10.0), ("s1", "sarimax", "d1", 20.0)])
    rows = combine_calculated(base, _cfg(["mean"]))
    assert len(rows) == 1
    r = rows[0]
    assert r["model_type"] == "ensemble_mean"
    assert r["yhat"] == pytest.approx(15.0)  # (10 + 20) / 2
    assert r["yhat_lower"] == pytest.approx(14.0)  # bounds blend the same way (yhat ± 1)
    assert r["yhat_upper"] == pytest.approx(16.0)
    # run_id / ensemble_id are stamped by the orchestrator, not the pure blender.
    assert "run_id" not in r and "ensemble_id" not in r


def test_median_is_robust_to_a_wild_base_forecast() -> None:
    base = _base_df(
        [
            ("s1", "theta", "d1", 10.0),
            ("s1", "sarimax", "d1", 12.0),
            ("s1", "xgboost", "d1", 1000.0),
        ]
    )
    rows = combine_calculated(base, _cfg(["median"]))
    assert rows[0]["yhat"] == pytest.approx(12.0)  # median ignores the 1000 outlier


def test_inverse_error_weights_by_run_metric() -> None:
    # theta far better than sarimax (lower wape) → blend pulled toward theta's 10.
    base = _base_df([("s1", "theta", "d1", 10.0), ("s1", "sarimax", "d1", 30.0)])
    metric = _metric_df([("theta", 0.1), ("sarimax", 0.9)])
    rows = combine_calculated(base, _cfg(["inverse_error"]), metric)
    # weights ∝ 1/0.1 : 1/0.9 = 9 : 1 → (9*10 + 1*30)/10 = 12.0
    assert rows[0]["yhat"] == pytest.approx(12.0)


def test_inverse_error_without_metric_frame_degrades_to_mean() -> None:
    # no metadata → uniform weights (the old SQL's NULL-tolerant SAFE_DIVIDE behavior).
    base = _base_df([("s1", "theta", "d1", 10.0), ("s1", "sarimax", "d1", 30.0)])
    rows = combine_calculated(base, _cfg(["inverse_error"]), None)
    assert rows[0]["yhat"] == pytest.approx(20.0)  # (10 + 30) / 2


def test_prune_threshold_drops_weak_base_models_fleetwide() -> None:
    # sarimax's mean wape (0.8) exceeds the 0.5 threshold → dropped; mean is theta alone.
    base = _base_df([("s1", "theta", "d1", 10.0), ("s1", "sarimax", "d1", 30.0)])
    metric = _metric_df([("theta", 0.1), ("sarimax", 0.8)])
    rows = combine_calculated(base, _cfg(["mean"], prune=0.5), metric)
    assert rows[0]["yhat"] == pytest.approx(10.0)  # sarimax pruned → theta only


def test_blend_renormalizes_over_present_models_per_key() -> None:
    # d2 has only theta present → its mean blend is theta alone.
    base = _base_df(
        [
            ("s1", "theta", "d1", 10.0),
            ("s1", "sarimax", "d1", 20.0),
            ("s1", "theta", "d2", 40.0),
        ]
    )
    rows = combine_calculated(base, _cfg(["mean"]))
    by_date = {r["forecast_date"]: r["yhat"] for r in rows if r["model_type"] == "ensemble_mean"}
    assert by_date["d1"] == pytest.approx(15.0)
    assert by_date["d2"] == pytest.approx(40.0)


def test_multi_strategy_emits_each_calculated_family() -> None:
    base = _base_df([("s1", "theta", "d1", 10.0), ("s1", "sarimax", "d1", 20.0)])
    metric = _metric_df([("theta", 0.5), ("sarimax", 0.5)])
    rows = combine_calculated(base, _cfg(["mean", "median", "inverse_error"]), metric)
    assert {r["model_type"] for r in rows} == {
        "ensemble_mean",
        "ensemble_median",
        "ensemble_inverse_error",
    }


def test_learned_strategies_are_ignored_by_calculated_blender() -> None:
    # a mixed config blends only the calculated members; learned ones are fit_learned's job.
    base = _base_df([("s1", "theta", "d1", 10.0), ("s1", "sarimax", "d1", 20.0)])
    rows = combine_calculated(base, _cfg(["mean", "nnls"]))
    assert {r["model_type"] for r in rows} == {"ensemble_mean"}
