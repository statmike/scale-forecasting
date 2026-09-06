"""``cfg.model_params`` — the precedence ladder, and the plan-time checks that guard it.

Three places a model's params resolve, and an authored block has to reach all three or it is
silently dropped somewhere: the cell (`worker._resolve_params`), each HPO trial's objective, and the
winner HPO returns (`hpo.tune_model`). Under the default ``fleetwide`` granularity the params the
cell fits with come *from* HPO, so a merge missing in `tune_model` would not be visible at the cell
at all — which is why each site is asserted separately here.

The rule is one sentence: **HPO wins on the keys its search space names, the authored block fills in
everything else.** A study that scored one value and shipped another would publish a metric that
does not belong to the fitted model, so a pin on a searched key loses; a pin on an unsearched key is
the common case and survives untouched.

The second half covers `dag.check_model_params`, which refuses a typo'd model name and lets a model
refuse a block it cannot honour — at plan time, before anything is provisioned.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from scale_forecasting import dag
from scale_forecasting.config import RunConfig
from scale_forecasting.errors import ConfigError
from scale_forecasting.hpo import tune_model
from scale_forecasting.models import get_model
from scale_forecasting.models.base_model import BaseModel
from scale_forecasting.worker import _model_context, _resolve_params, run_cell

HORIZON = 7


def _cfg(**over: Any) -> RunConfig:
    base: dict[str, Any] = {
        "run_name": "model params test",
        "data": {"source_table": "t", "freq": "D", "horizon": HORIZON},
        "models": ["theta"],
    }
    base.update(over)
    return RunConfig(**base)


def _series(n: int = 160, ts_id: str = "s0") -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    y = np.linspace(10.0, 30.0, n) + 3.0 * np.sin(np.arange(n) * 2 * np.pi / 7)
    return pd.DataFrame({"ts_id": ts_id, "ds": idx, "y": y})


# --- the ladder at the cell ----------------------------------------------------


def test_an_authored_block_reaches_the_cell_with_hpo_off() -> None:
    cfg = _cfg(model_params={"theta": {"deseasonalize": False}})
    assert _resolve_params(_series(), "theta", cfg, _model_context(cfg), None) == {
        "deseasonalize": False
    }


def test_a_model_this_run_did_not_author_gets_nothing() -> None:
    cfg = _cfg(models=["theta", "naive_mean"], model_params={"theta": {"deseasonalize": False}})
    assert _resolve_params(_series(), "naive_mean", cfg, _model_context(cfg), None) == {}


def test_pre_resolved_fleetwide_params_land_on_top_of_the_authored_block() -> None:
    """The fleetwide pre-pass already merged; the cell must not undo it, or re-drop the rest."""
    cfg = _cfg(model_params={"theta": {"deseasonalize": False, "alpha": 0.1}})
    resolved = _resolve_params(
        _series(), "theta", cfg, _model_context(cfg), {"alpha": 0.9, "beta": 2}
    )
    assert resolved == {"deseasonalize": False, "alpha": 0.9, "beta": 2}


def test_the_params_a_cell_actually_fits_with_are_the_ones_it_reports() -> None:
    """End-to-end through ``run_cell``: what lands in ``best_params`` is the merged block."""
    cfg = _cfg(model_params={"theta": {"deseasonalize": False}})
    result = run_cell(_series(), "theta", cfg)
    assert result.status == "ok"
    assert result.best_params.get("deseasonalize") is False


# --- the ladder inside HPO -----------------------------------------------------


class _SpyModel(BaseModel):
    """Records every params dict it is constructed with, so a merge can be seen at each site."""

    name = "_spy_model"
    runtime = "python"
    family = "statistical"
    seen: list[dict[str, Any]] = []

    def __init__(self, params: dict[str, Any], ctx: Any) -> None:
        super().__init__(params, ctx)
        type(self).seen.append(dict(params))

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        self._mean = float(y.mean())
        self._last = y.index[-1]

    def predict(
        self,
        horizon: int,
        X: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> pd.DataFrame:
        yhat = np.full(horizon, self._mean)
        return self._assemble_frame(
            self._future_index(self._last, horizon), {q: yhat for q in quantiles}
        )

    @classmethod
    def search_space(cls, trial: Any) -> dict[str, Any]:
        return {"alpha": trial.suggest_categorical("alpha", [0.25, 0.75])}


@pytest.fixture
def spy() -> Any:
    """Register the spy for one test and take it back out of the global registry after."""
    from scale_forecasting.models import base_model

    _SpyModel.seen = []
    base_model._REGISTRY[_SpyModel.name] = _SpyModel
    try:
        yield _SpyModel
    finally:
        base_model._REGISTRY.pop(_SpyModel.name, None)


def _hpo_cfg(**over: Any) -> RunConfig:
    return _cfg(
        models=[_SpyModel.name],
        backtest={
            "enabled": True,
            "n_folds": 2,
            "horizon": HORIZON,
            "step": HORIZON,
            "min_train": 60,
        },
        hpo={"enabled": True, "n_trials": 2, "granularity": "fleetwide", "sample_size": 1},
        **over,
    )


def test_every_trial_is_scored_with_the_authored_block_underneath_it(spy: Any) -> None:
    """A study that tuned a model the run will not fit optimizes the wrong thing."""
    cfg = _hpo_cfg(model_params={_SpyModel.name: {"n_lags": 28}})
    tune_model(_SpyModel.name, [_series()], cfg)
    assert spy.seen, "the study never constructed the model"
    assert all(p.get("n_lags") == 28 for p in spy.seen)


def test_hpo_returns_the_winner_with_the_authored_block_still_beneath_it(spy: Any) -> None:
    cfg = _hpo_cfg(model_params={_SpyModel.name: {"n_lags": 28}})
    best = tune_model(_SpyModel.name, [_series()], cfg)
    assert best["n_lags"] == 28
    assert best["alpha"] in (0.25, 0.75)


def test_hpo_wins_on_a_key_its_search_space_names(spy: Any) -> None:
    """The pin loses on purpose: the published metric has to belong to the fitted model."""
    cfg = _hpo_cfg(model_params={_SpyModel.name: {"alpha": 0.5}})
    best = tune_model(_SpyModel.name, [_series()], cfg)
    assert best["alpha"] in (0.25, 0.75)
    assert all(p["alpha"] in (0.25, 0.75) for p in spy.seen)


def test_a_model_with_no_search_space_tunes_nothing_rather_than_echoing_the_block() -> None:
    """``{}`` means "nothing was tuned"; the cell applies the authored layer either way."""
    cfg = _cfg(
        models=["naive_mean"],
        backtest={
            "enabled": True,
            "n_folds": 2,
            "horizon": HORIZON,
            "step": HORIZON,
            "min_train": 60,
        },
        hpo={"enabled": True, "n_trials": 2},
        model_params={"naive_mean": {"whatever": 1}},
    )
    assert tune_model("naive_mean", [_series()], cfg) == {}


# --- plan-time checks ----------------------------------------------------------


def test_a_typo_in_a_model_name_is_refused_rather_than_silently_ignored() -> None:
    cfg = _cfg(model_params={"nueralprophet": {"n_lags": 7}})
    with pytest.raises(ConfigError, match="not registered models"):
        dag.check_model_params(cfg)


def test_a_block_for_an_unselected_model_warns_but_runs(caplog: Any) -> None:
    cfg = _cfg(models=["theta"], model_params={"naive_mean": {"whatever": 1}})
    with caplog.at_level("WARNING"):
        dag.check_model_params(cfg)
    assert "no effect" in caplog.text


def test_a_clean_block_passes() -> None:
    dag.check_model_params(_cfg(model_params={"theta": {"deseasonalize": False}}))


def test_plan_dag_stays_total_so_pure_inspection_never_raises() -> None:
    """The SDK's ``dag``, a dry run and the tests all call ``plan_dag``; it must not refuse."""
    dag.plan_dag(_cfg(model_params={"not_a_model": {"x": 1}}))


def test_the_default_validate_params_accepts_everything() -> None:
    get_model("theta").validate_params({"anything": 1}, max_horizon=999)


def test_autoregression_below_the_horizon_is_refused_before_anything_is_provisioned() -> None:
    """NeuralProphet emits exactly ``n_forecasts`` direct steps and does not recurse."""
    cfg = _cfg(
        models=["neuralprophet"],
        model_params={"neuralprophet": {"n_lags": 28, "n_forecasts": 1}},
    )
    with pytest.raises(ConfigError, match="n_forecasts"):
        dag.check_model_params(cfg)


def test_the_backtest_horizon_counts_too_when_it_is_the_longer_one() -> None:
    cfg = _cfg(
        models=["neuralprophet"],
        backtest={"enabled": True, "n_folds": 2, "horizon": 21, "step": 21, "min_train": 60},
        model_params={"neuralprophet": {"n_lags": 28, "n_forecasts": 14}},
    )
    with pytest.raises(ConfigError, match="longest horizon of 21"):
        dag.check_model_params(cfg)
    # ...and a disabled backtest does not get a vote.
    ok = cfg.model_copy(update={"backtest": cfg.backtest.model_copy(update={"enabled": False})})
    dag.check_model_params(ok)


def test_autoregression_with_enough_heads_is_accepted() -> None:
    cfg = _cfg(
        models=["neuralprophet"],
        model_params={"neuralprophet": {"n_lags": 28, "n_forecasts": HORIZON}},
    )
    dag.check_model_params(cfg)


def test_the_shipped_default_never_trips_the_autoregression_check() -> None:
    """``n_lags`` unset is one head and no autoregression — the check must stay silent."""
    dag.check_model_params(_cfg(models=["neuralprophet"]))
