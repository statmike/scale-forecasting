"""Offline tests for the HPO execution path (``scale_forecasting.hpo``).

Covers an Optuna study over the aligned backtest that *actually* varies params and
records a winner (vs an earlier no-op ``{}``), the two granularities (fleetwide + per-series),
the no-search-space / native / disabled short-circuits, and the ``require_backtest`` guard. It is
deterministic (fixed-seed TPE) and fully offline — no GCP, no Spark.

The engine threading (driver pre-pass → ``run_group`` → ``run_cell`` closure) is exercised in
``test_worker.py`` / ``test_spark_engines.py``; here we test the pure tuning core.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from scale_forecasting.config import RunConfig
from scale_forecasting.errors import ConfigError
from scale_forecasting.hpo import (
    _has_search_space,
    _minimize_scalar,
    require_backtest,
    resolve_fleetwide,
    tune_model,
)
from scale_forecasting.models import get_model


def _cfg(**over: Any) -> RunConfig:
    base: dict[str, Any] = {
        "run_name": "hpo test",
        "data": {"source_table": "t", "horizon": 7, "freq": "D"},
        "models": ["xgboost"],
        "backtest": {"enabled": True, "n_folds": 2, "horizon": 7, "step": 7, "min_train": 60},
        "hpo": {"enabled": True, "n_trials": 5, "granularity": "fleetwide", "sample_size": 3},
    }
    base.update(over)
    return RunConfig(**base)


def _series(tid: str = "s0", n: int = 120, seed: int = 0) -> pd.DataFrame:
    """A synthetic weekly-seasonal + trend series long enough for the fold geometry."""
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2022-01-01", periods=n, freq="D")
    y = 50 + 10 * np.sin(np.arange(n) * 2 * np.pi / 7) + np.linspace(0, 20, n) + rng.normal(0, 2, n)
    return pd.DataFrame({"ts_id": tid, "ds": ds, "y": y})


def _sample(k: int = 3) -> list[pd.DataFrame]:
    return [_series(f"s{i}", seed=i) for i in range(k)]


# --- require_backtest guard ----------------------------------------------------


def test_require_backtest_raises_when_hpo_on_but_backtest_off() -> None:
    # Construct the guard scenario directly (RunConfig itself also rejects this at validation).
    cfg = _cfg()
    off = cfg.model_copy(update={"backtest": cfg.backtest.model_copy(update={"enabled": False})})
    with pytest.raises(ConfigError, match="requires backtest"):
        require_backtest(off)


def test_require_backtest_is_noop_when_hpo_off() -> None:
    cfg = _cfg(hpo={"enabled": False}, backtest={"enabled": False})
    require_backtest(cfg)  # no raise: nothing to tune, so backtest isn't required


def test_config_rejects_hpo_without_backtest_at_validation() -> None:
    # The fail-fast at load (config._normalize) — a stranger's typo fails before any GCP touch.
    with pytest.raises(Exception, match="requires backtest"):
        _cfg(backtest={"enabled": False})


# --- _minimize_scalar direction mapping ----------------------------------------


def test_minimize_scalar_error_metrics_pass_through() -> None:
    assert _minimize_scalar("wape", 0.3) == pytest.approx(0.3)
    assert _minimize_scalar("rmse", 12.0) == pytest.approx(12.0)


def test_minimize_scalar_coverage_is_negated() -> None:
    # higher coverage is better → minimize its negation
    assert _minimize_scalar("coverage", 0.9) == pytest.approx(-0.9)


def test_minimize_scalar_bias_uses_magnitude() -> None:
    # bias is best near zero → a large negative bias is as bad as a large positive one
    assert _minimize_scalar("bias", -5.0) == pytest.approx(5.0)
    assert _minimize_scalar("bias", 5.0) == pytest.approx(5.0)


def test_minimize_scalar_nan_is_infinite() -> None:
    assert _minimize_scalar("wape", float("nan")) == float("inf")


# --- _has_search_space ---------------------------------------------------------


def test_has_search_space_true_for_models_that_override() -> None:
    assert _has_search_space(get_model("xgboost")) is True
    assert _has_search_space(get_model("theta")) is True


def test_has_search_space_false_for_models_without_one() -> None:
    # holtwinters inherits the base (empty) search_space → nothing to tune.
    assert _has_search_space(get_model("holtwinters")) is False


def test_has_search_space_false_for_native_models() -> None:
    # BigQuery-native models tune in BQML, not here — excluded regardless.
    assert _has_search_space(get_model("arima_plus")) is False


# --- tune_model: the study actually runs and picks a winner --------------------


def test_tune_model_returns_params_from_the_search_space() -> None:
    out = tune_model("xgboost", _sample(), _cfg())
    # the winner is a concrete point in xgboost's search space
    assert set(out) == {"n_estimators", "max_depth", "learning_rate"}
    assert 100 <= out["n_estimators"] <= 600
    assert 3 <= out["max_depth"] <= 10


def test_tune_model_is_deterministic() -> None:
    # fixed-seed TPE → same sample + config → same winner (reproducible).
    a = tune_model("xgboost", _sample(), _cfg())
    b = tune_model("xgboost", _sample(), _cfg())
    assert a == b


def test_tune_model_empty_search_space_returns_empty_no_study() -> None:
    # holtwinters has no space → {} without constructing a study (costs nothing).
    assert tune_model("holtwinters", _sample(), _cfg(models=["holtwinters"])) == {}


def test_tune_model_native_model_returns_empty() -> None:
    assert tune_model("arima_plus", _sample(), _cfg(models=["arima_plus"])) == {}


def test_tune_model_honors_n_trials() -> None:
    # more trials must not error and should still land a valid point (smoke on the trial count).
    cfg = _cfg(models=["theta"], hpo={"enabled": True, "n_trials": 8})
    out = tune_model("theta", _sample(), cfg)
    assert set(out) == {"deseasonalize"}


def test_tune_model_all_series_too_short_scores_inf_but_still_returns_a_point() -> None:
    # every series is too short for the fold geometry → each trial scores +inf, yet the study still
    # returns *a* point from the space (it never crashes; the objective is fault-tolerant).
    short = [_series("x", n=40)]  # < min_train + horizon + step
    out = tune_model("theta", short, _cfg(models=["theta"]))
    assert set(out) == {"deseasonalize"}


# --- resolve_fleetwide: one winner per tunable model ---------------------------


def test_resolve_fleetwide_maps_each_tunable_model() -> None:
    cfg = _cfg(models=["xgboost", "theta"])
    resolved = resolve_fleetwide(_sample(), cfg)
    assert set(resolved) == {"xgboost", "theta"}
    assert set(resolved["theta"]) == {"deseasonalize"}


def test_resolve_fleetwide_omits_models_with_no_search_space() -> None:
    # holtwinters has nothing to tune → absent from the map (it keeps {} defaults in run_cell).
    cfg = _cfg(models=["xgboost", "holtwinters"])
    resolved = resolve_fleetwide(_sample(), cfg)
    assert set(resolved) == {"xgboost"}


def test_resolve_fleetwide_requires_backtest() -> None:
    cfg = _cfg()
    off = cfg.model_copy(update={"backtest": cfg.backtest.model_copy(update={"enabled": False})})
    with pytest.raises(ConfigError, match="requires backtest"):
        resolve_fleetwide(_sample(), off)


def test_resolve_fleetwide_all_calculated_models_is_empty_map() -> None:
    # a config of only no-search-space models → empty resolution (every cell uses {} defaults).
    cfg = _cfg(models=["holtwinters"])
    assert resolve_fleetwide(_sample(), cfg) == {}
