"""The frozen-backtest seam, exercised against every registered Python model.

Two operations move a fitted model's forecast origin without re-estimating it, and the whole
value of ``backtest.scheme`` in {``expanding_frozen``, ``expanding_stale``} rests on both being
*exactly* what they claim:

* ``advance_origin(k)`` — walk ``k`` steps forward **blind**, no new actuals. Every model supports
  it, so the staleness question is asked of all sixteen in identical terms.
* ``recondition(y_new)`` — take in the observations that arrived after the origin, parameters held
  fixed. Ten models support it; the other six say so rather than approximate.

The invariant that makes the sixteen-file edit safe is the first test here: **at the default offset
of zero, every model's ``predict`` is byte-identical to what it produced before this seam existed.**
Nothing on the ordinary path may move.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scale_forecasting.errors import ModelError
from scale_forecasting.models import get_model, list_models
from scale_forecasting.models.base_model import DEFAULT_QUANTILES, BaseModel, ModelContext

HORIZON = 14
# Deliberately not a multiple of the weekly seasonal period: a seasonal model that restarted its
# cycle at the advanced origin instead of keeping phase against the fit would pass at 7 and fail
# here, which is the bug worth catching.
OFFSET = 6
SPAN = OFFSET + HORIZON

# Optional third-party dep required by each model (None = core-only).
_MODEL_DEP: dict[str, str] = {
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "prophet": "prophet",
    "neuralprophet": "neuralprophet",
}

# Cheap-but-equivalent params: this file asserts *where* a forecast is anchored, never how good it
# is, so a two-epoch network exercises the same code path as a fifty-epoch one in a fraction of the
# time.
_PARAMS: dict[str, dict[str, Any]] = {"neuralprophet": {"epochs": 2}}

# Models whose point forecast provably *must* move when the history steps up by a constant, because
# nothing in them is estimated (or, for the two state-space models, because the Kalman filter
# carries the level). The tree models are excluded on purpose: a tree cannot extrapolate past its
# training range, so a large upward shift saturates it at its top leaf and the direction of the
# change is not something the seam can promise.
_LEVEL_SHIFT_RAISES_THE_FORECAST = {
    "sarimax",
    "ucm",
    "croston",
    "naive_mean",
    "naive_drift",
    "naive_seasonal",
    "naive_moving_average",
}


def _python_models() -> list[str]:
    return [n for n in list_models() if get_model(n).runtime == "python"]


def _ctx(**over: Any) -> ModelContext:
    base: dict[str, Any] = {"freq": "D", "horizon": HORIZON, "seed": 7}
    base.update(over)
    return ModelContext(**base)


def _golden(n: int, with_exog: bool) -> tuple[pd.Series, pd.DataFrame | None]:
    """Deterministic trend + weekly seasonality + mild noise, ds-indexed."""
    rng = np.random.default_rng(1234)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    y = pd.Series(
        np.linspace(10.0, 40.0, n)
        + 4.0 * np.sin(np.arange(n) * 2 * np.pi / 7)
        + rng.normal(0, 0.5, n),
        index=idx,
        name="y",
    )
    X = None
    if with_exog:
        X = pd.DataFrame({"price_index": np.cos(np.arange(n) * 2 * np.pi / 30)}, index=idx)
    return y, X


def _fit_holding_back(name: str, span: int) -> tuple[BaseModel, pd.Series, pd.DataFrame | None]:
    """Fit on all but the last ``span`` observations; hand back the held-out exog for them."""
    cls = get_model(name)
    dep = _MODEL_DEP.get(name)
    if dep is not None and importlib.util.find_spec(dep) is None:
        pytest.skip(f"optional dependency '{dep}' not installed for model '{name}'")
    inst = cls(_PARAMS.get(name, {}), _ctx())
    y, X = _golden(400, with_exog=inst.supports_exog)
    y_fit, x_fit = y.iloc[:-span], (X.iloc[:-span] if X is not None else None)
    inst.fit(y_fit, x_fit)
    return inst, y_fit, (X.iloc[-span:] if X is not None else None)


def _window(exog: pd.DataFrame | None, start: int, count: int) -> pd.DataFrame | None:
    return None if exog is None else exog.iloc[start : start + count]


@pytest.fixture(params=_python_models())
def model_name(request: pytest.FixtureRequest) -> str:
    return str(request.param)


# --- advancing the origin -------------------------------------------------------


def test_the_default_offset_changes_nothing(model_name: str) -> None:
    """Offset 0 — and an explicit ``advance_origin(0)`` — reproduce the pre-seam frame exactly.

    This is the safety net for the whole edit. Sixteen ``predict`` bodies were rewritten to route
    their dates through `_forecast_index` and their step counts through `_forecast_steps`; if any of
    them got the arithmetic wrong, an ordinary forecast — every run in the ledger — would move.
    """
    model, y_fit, exog = _fit_holding_back(model_name, SPAN)
    before = model.predict(HORIZON, _window(exog, 0, HORIZON))

    assert before["ds"].iloc[0] == y_fit.index[-1] + pd.Timedelta(days=1)

    model.advance_origin(0)
    after = model.predict(HORIZON, _window(exog, 0, HORIZON))
    pd.testing.assert_frame_equal(before, after)


def test_an_advanced_origin_is_the_tail_of_a_longer_forecast(model_name: str) -> None:
    """Advancing ``k`` and forecasting ``h`` must equal steps ``k+1..k+h`` of one long forecast.

    That is the definition of "same fitted model, later window", and it is what makes a stale fold's
    score comparable to a refit fold's: both are read off the same date grid.

    Asserted on ``yhat_raw`` — the model's own point forecast — rather than ``yhat``. The two
    forecasts are of different lengths, and ``stl_bagging`` draws one bootstrap sample per requested
    step, so its *band* legitimately differs between a 20-step call and a 14-step one. The point
    forecast does not, and that is the number the origin arithmetic is responsible for.
    """
    model, _, exog = _fit_holding_back(model_name, SPAN)
    long = model.predict(SPAN, _window(exog, 0, SPAN))

    model.advance_origin(OFFSET, X_gap=_window(exog, 0, OFFSET))
    advanced = model.predict(HORIZON, _window(exog, OFFSET, HORIZON))

    assert list(advanced["ds"]) == list(long["ds"].iloc[-HORIZON:])
    assert np.allclose(
        advanced["yhat_raw"].to_numpy(), long["yhat_raw"].to_numpy()[-HORIZON:], atol=1e-8
    ), f"{model_name}: advanced forecast is not the tail of the long one"


def test_the_offset_is_absolute_so_setting_it_twice_is_idempotent(model_name: str) -> None:
    """``advance_origin`` counts from the *fit*, not from wherever the origin currently sits.

    The backtest driver sets it once per fold on a model it is reusing, so a relative offset would
    silently accumulate — fold three would forecast from six times the intended gap. Setting the
    same value twice, and resetting to 0, both have to be no-ops.
    """
    model, _, exog = _fit_holding_back(model_name, SPAN)
    at_zero = model.predict(HORIZON, _window(exog, 0, HORIZON))

    model.advance_origin(OFFSET, X_gap=_window(exog, 0, OFFSET))
    once = model.predict(HORIZON, _window(exog, OFFSET, HORIZON))
    model.advance_origin(OFFSET, X_gap=_window(exog, 0, OFFSET))
    twice = model.predict(HORIZON, _window(exog, OFFSET, HORIZON))
    pd.testing.assert_frame_equal(once, twice)

    model.advance_origin(0)
    pd.testing.assert_frame_equal(at_zero, model.predict(HORIZON, _window(exog, 0, HORIZON)))


def test_a_negative_offset_is_refused(model_name: str) -> None:
    model, _, _ = _fit_holding_back(model_name, SPAN)
    with pytest.raises(ModelError, match="must be >= 0"):
        model.advance_origin(-1)


def test_every_python_model_can_be_walked_forward_blind() -> None:
    """`expanding_stale` asks all sixteen one identical question — so all sixteen must opt in.

    If a model ever declines, the stale leaderboard silently becomes a leaderboard of *some* models,
    which is the confounding this scheme exists to remove.
    """
    declining = [n for n in _python_models() if not get_model(n).supports_extrapolate]
    assert declining == []


# --- re-conditioning ------------------------------------------------------------


def test_recondition_sees_a_level_shift_that_a_blind_advance_cannot(model_name: str) -> None:
    """The two estimands, side by side on one date grid — which is exactly how they get scored.

    Both models below are fitted identically and forecast the same fourteen dates. One is walked
    forward blind; the other is handed the six observations that arrived in between, with its
    parameters held fixed. Those observations step up by 100, so a working ``recondition`` cannot
    produce the blind model's number — and a ``recondition`` that quietly did nothing would produce
    exactly it, with a perfectly plausible frame and no other symptom. Hence the comparison rather
    than a bare "the call returned".
    """
    if not get_model(model_name).supports_recondition:
        pytest.skip(f"{model_name} declines re-conditioning")

    blind_model, y_fit, exog = _fit_holding_back(model_name, SPAN)
    blind_model.advance_origin(OFFSET, X_gap=_window(exog, 0, OFFSET))
    blind = blind_model.predict(HORIZON, _window(exog, OFFSET, HORIZON))

    new_index = pd.date_range(y_fit.index[-1] + pd.Timedelta(days=1), periods=OFFSET, freq="D")
    y_new = pd.Series(y_fit.to_numpy()[-OFFSET:] + 100.0, index=new_index, name="y")
    cond_model, _, _ = _fit_holding_back(model_name, SPAN)
    cond_model.recondition(y_new, _window(exog, 0, OFFSET))
    conditioned = cond_model.predict(HORIZON, _window(exog, OFFSET, HORIZON))

    assert conditioned["ds"].iloc[0] == new_index[-1] + pd.Timedelta(days=1)
    assert list(conditioned["ds"]) == list(blind["ds"]), "both arms must score the same dates"
    assert not np.allclose(conditioned["yhat_raw"].to_numpy(), blind["yhat_raw"].to_numpy()), (
        f"{model_name}: re-conditioning left the forecast untouched"
    )
    if model_name in _LEVEL_SHIFT_RAISES_THE_FORECAST:
        assert conditioned["yhat_raw"].mean() > blind["yhat_raw"].mean()


def test_a_model_that_declines_says_so_rather_than_approximating(model_name: str) -> None:
    """Six models have no seam that absorbs an observation short of re-estimating.

    They raise, the cell falls back to a fresh fit, and ``backtest_refit`` records
    ``"unsupported"`` — a fact a reader can filter on. Silently substituting a weaker estimand is
    the failure mode this test exists to prevent.
    """
    cls = get_model(model_name)
    if cls.supports_recondition:
        pytest.skip(f"{model_name} supports re-conditioning")

    model, y_fit, _ = _fit_holding_back(model_name, SPAN)
    new_index = pd.date_range(y_fit.index[-1] + pd.Timedelta(days=1), periods=OFFSET, freq="D")
    with pytest.raises(ModelError, match="must be refit"):
        model.recondition(pd.Series(np.ones(OFFSET), index=new_index, name="y"))


def test_the_re_condition_tier_is_the_one_the_plan_names() -> None:
    """Which models can be walked forward on real actuals is a published fact, not an accident.

    ``docs/backtesting.md`` and the leaderboard's ``backtest_refit`` column both rest on this split,
    so a model quietly gaining or losing the capability has to fail here first.
    """
    tier = sorted(n for n in _python_models() if get_model(n).supports_recondition)
    assert tier == [
        "croston",
        "lightgbm",
        "naive_drift",
        "naive_mean",
        "naive_moving_average",
        "naive_seasonal",
        "regression_lags",
        "sarimax",
        "ucm",
        "xgboost",
    ]


def test_the_base_class_refuses_to_advance_a_model_that_has_not_opted_in() -> None:
    """`supports_extrapolate` defaults to False so an out-of-tree model is never *assumed* capable.

    Every model in this repo opts in, so the default path needs its own model to exercise.
    """

    class NotOptedIn(BaseModel):
        name = "not_opted_in"
        runtime = "python"
        family = "statistical"

        def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
            self._last_date = y.index[-1]

        def predict(
            self,
            horizon: int,
            X: pd.DataFrame | None = None,
            quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        ) -> pd.DataFrame:
            raise NotImplementedError

    model = NotOptedIn({}, _ctx())
    with pytest.raises(ModelError, match="cannot advance its forecast origin"):
        model.advance_origin(3)
    with pytest.raises(ModelError, match="must be refit"):
        model.recondition(pd.Series([1.0], index=pd.DatetimeIndex(["2023-01-01"])))
