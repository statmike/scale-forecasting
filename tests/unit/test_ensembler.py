"""Tests for the ensembler — calculated combine math, learned meta-learners, SQL builders.

Covers CONTRACTS §6 / DESIGN §5.2 / BUILD 6a: mean/median exact, inverse-error weights sum to
1, NNLS weights ≥ 0, the leakage guard (learned strategies refuse to run without backtest),
multi-strategy dispatch, and a snapshot of the generated BigQuery SQL.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scale_forecasting.config import RunConfig
from scale_forecasting.ensembler import (
    build_ensemble_sql,
    fit_learned,
    inverse_error_weights,
    mean_combine,
    median_combine,
)
from scale_forecasting.errors import ConfigError

SNAPSHOT = Path(__file__).parent / "snapshots" / "ensemble_sql.sql"


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


# --- SQL builders --------------------------------------------------------------


def test_no_calculated_strategy_yields_empty_sql() -> None:
    assert build_ensemble_sql(_cfg(["nnls"])) == ""


def test_mean_sql_aggregates_base_predictions() -> None:
    sql = build_ensemble_sql(_cfg(["mean"]), dataset="proj.ds")
    assert "AVG(yhat)" in sql
    assert "'ensemble_mean'" in sql
    assert "`proj.ds.forecast_predictions`" in sql
    assert "@run_id" in sql


def test_median_sql_uses_approx_quantiles() -> None:
    sql = build_ensemble_sql(_cfg(["median"]))
    assert "APPROX_QUANTILES(yhat, 2)[OFFSET(1)]" in sql


def test_inverse_error_sql_weights_by_metric() -> None:
    sql = build_ensemble_sql(_cfg(["inverse_error"]))
    assert "SAFE_DIVIDE(1, NULLIF(AVG(wape), 0))" in sql
    assert "'ensemble_inverse_error'" in sql


def test_prune_threshold_filters_models() -> None:
    sql = build_ensemble_sql(_cfg(["mean"], prune=0.5))
    assert "forecast_metadata" in sql
    assert "wape > 0.5" in sql


def test_multi_strategy_sql_emits_each() -> None:
    sql = build_ensemble_sql(_cfg(["mean", "median", "inverse_error"]))
    assert "'ensemble_mean'" in sql
    assert "'ensemble_median'" in sql
    assert "'ensemble_inverse_error'" in sql


def test_ensemble_sql_snapshot() -> None:
    cfg = _cfg(["mean", "median", "inverse_error"], prune=0.3)
    rendered = build_ensemble_sql(cfg, dataset="proj.scale_forecasting")
    if os.environ.get("SF_UPDATE_SNAPSHOTS") == "1":
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(rendered)
    assert SNAPSHOT.exists(), "snapshot missing; run with SF_UPDATE_SNAPSHOTS=1 to create"
    assert rendered == SNAPSHOT.read_text()
