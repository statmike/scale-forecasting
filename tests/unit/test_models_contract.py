"""Shared model contract test.

Parametrized over every registered Python model: fit a deterministic golden series,
then assert ``predict`` returns the canonical frame — right columns/dtypes, length =
horizon, bounds ordered (lower ≤ yhat ≤ upper), original units, deterministic under a
fixed seed, and ``supports_exog`` honored. BigQuery-runtime models are skipped (they
execute as SQL, not Python); models whose optional dep isn't installed are skipped.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scale_forecasting.errors import ModelError
from scale_forecasting.models import get_model, list_models
from scale_forecasting.models.base_model import PREDICTION_COLUMNS, BaseModel, ModelContext

HORIZON = 14

# Optional third-party dep required by each model (None = core-only).
_MODEL_DEP: dict[str, str] = {
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "prophet": "prophet",
    "neuralprophet": "neuralprophet",
}


def _python_models() -> list[str]:
    names = []
    for name in list_models():
        cls = get_model(name)
        if cls.runtime != "python":
            continue
        names.append(name)
    return names


def _golden_series(n: int = 400, with_exog: bool = False) -> tuple[pd.Series, pd.DataFrame | None]:
    """Deterministic trend + weekly seasonality + mild noise, ds-indexed."""
    rng = np.random.default_rng(1234)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    trend = np.linspace(10.0, 40.0, n)
    weekly = 4.0 * np.sin(np.arange(n) * 2 * np.pi / 7)
    noise = rng.normal(0, 0.5, n)
    y = pd.Series(trend + weekly + noise, index=idx, name="y")
    X = None
    if with_exog:
        X = pd.DataFrame({"price_index": np.cos(np.arange(n) * 2 * np.pi / 30)}, index=idx)
    return y, X


def _ctx(**over: Any) -> ModelContext:
    base: dict[str, Any] = {"freq": "D", "horizon": HORIZON, "seed": 7}
    base.update(over)
    return ModelContext(**base)


def _make(name: str) -> BaseModel:
    dep = _MODEL_DEP.get(name)
    if dep is not None and importlib.util.find_spec(dep) is None:
        pytest.skip(f"optional dependency '{dep}' not installed for model '{name}'")
    return get_model(name)({}, _ctx())


@pytest.fixture(params=_python_models())
def model_name(request: pytest.FixtureRequest) -> str:
    return str(request.param)


# --- the contract --------------------------------------------------------------


def test_predict_returns_canonical_frame(model_name: str) -> None:
    m = _make(model_name)
    y, X = _golden_series(with_exog=m.supports_exog)
    m.fit(y, X)
    fx = X.iloc[:HORIZON] if (X is not None and m.supports_exog) else None
    df = m.predict(HORIZON, fx)

    assert list(df.columns) == list(PREDICTION_COLUMNS)
    assert len(df) == HORIZON
    assert df["ds"].dtype == np.dtype("datetime64[ns]")
    for col in ("yhat", "yhat_lower", "yhat_upper"):
        assert df[col].dtype == np.float64
        assert df[col].notna().all()


def test_bounds_are_ordered(model_name: str) -> None:
    m = _make(model_name)
    y, X = _golden_series(with_exog=m.supports_exog)
    m.fit(y, X)
    fx = X.iloc[:HORIZON] if (X is not None and m.supports_exog) else None
    df = m.predict(HORIZON, fx)
    assert (df["yhat_lower"] <= df["yhat"] + 1e-6).all()
    assert (df["yhat"] <= df["yhat_upper"] + 1e-6).all()


def test_forecast_dates_follow_history(model_name: str) -> None:
    m = _make(model_name)
    y, X = _golden_series(with_exog=m.supports_exog)
    m.fit(y, X)
    fx = X.iloc[:HORIZON] if (X is not None and m.supports_exog) else None
    df = m.predict(HORIZON, fx)
    assert df["ds"].iloc[0] > y.index[-1]
    assert df["ds"].is_monotonic_increasing


def test_original_units_after_log1p(model_name: str) -> None:
    # Fit on a positive series with log1p; forecasts must return to original scale
    # (roughly the level of the data, not the ~log level).
    m = get_model(model_name)
    dep = _MODEL_DEP.get(model_name)
    if dep is not None and importlib.util.find_spec(dep) is None:
        pytest.skip(f"optional dependency '{dep}' not installed")
    inst = m({}, _ctx(transform="log1p"))
    y, X = _golden_series(with_exog=inst.supports_exog)
    inst.fit(np.log1p(y).rename("y"), X)
    fx = X.iloc[:HORIZON] if (X is not None and inst.supports_exog) else None
    df = inst.predict(HORIZON, fx)
    # original data sits in ~[10, 45]; inverted forecast should be in a sane band, not log.
    assert df["yhat"].median() > 5.0


def test_original_units_after_boxcox(model_name: str) -> None:
    # Box-Cox is stateful: λ is fit per cell and handed to the model on ctx. Every model must
    # invert it in predict() and return original-scale forecasts. This exercises *every*
    # registered Python model, so a model that forgets ctx.transform_lambda fails loudly here.
    from scale_forecasting.features import apply_transform, fit_transform_lambda

    m = get_model(model_name)
    dep = _MODEL_DEP.get(model_name)
    if dep is not None and importlib.util.find_spec(dep) is None:
        pytest.skip(f"optional dependency '{dep}' not installed")
    y, X = _golden_series(with_exog=m({}, _ctx()).supports_exog)
    lam = fit_transform_lambda(y, "boxcox")
    inst = m({}, _ctx(transform="boxcox", transform_lambda=lam))
    inst.fit(apply_transform(y, "boxcox", lam).rename("y"), X)
    fx = X.iloc[:HORIZON] if (X is not None and inst.supports_exog) else None
    df = inst.predict(HORIZON, fx)
    # original data sits in ~[10, 45]; inverted forecast should be in a sane band, not boxcox-space.
    assert df["yhat"].median() > 5.0
    assert np.isfinite(df["yhat"].to_numpy()).all()


def test_deterministic_under_seed(model_name: str) -> None:
    """Same config, same data, same numbers — including the interval, not just the point.

    The bounds used to be outside this check, and one model was quietly failing it: Prophet does not
    compute its interval, it draws a thousand posterior samples and takes their quantiles, so two
    identical runs disagreed in the third decimal. Nothing noticed while the interval metrics were
    NaN on every Python cell. They are scored now, so an irreproducible bound is an irreproducible
    `coverage` in the registry, and the contract has to cover the whole frame.
    """
    y, X0 = _golden_series(with_exog=get_model(model_name).supports_exog)

    def run() -> pd.DataFrame:
        m = _make(model_name)
        m.fit(y, X0)
        fx = X0.iloc[:HORIZON] if (X0 is not None and m.supports_exog) else None
        return m.predict(HORIZON, fx)

    a, b = run(), run()
    for col in ("yhat", "yhat_lower", "yhat_upper"):
        assert np.allclose(a[col].to_numpy(), b[col].to_numpy()), f"{model_name}: {col} not stable"


def test_at_least_theta_registered() -> None:
    assert "theta" in list_models()


# --- NeuralProphet in autoregressive mode --------------------------------------
#
# `n_lags > 0` puts NeuralProphet into a *different output shape* than the shipped default, and
# reading the wrong one is silent rather than loud: it emits `n_forecasts` direct heads on a
# diagonal (the row for step i fills `yhat{i}` and leaves every other yhat column NaN), and
# `make_future_dataframe` returns `n_lags + n_forecasts` rows whatever `periods` it was asked for.
# Read naively — `yhat1` straight down, `.tail(horizon)` for the rows — the first gives one number
# followed by NaNs and the second gives the wrong steps. Both are fitted here rather than mocked,
# because the shape is the library's behaviour and a fake would just restate our belief about it.
# `epochs=2` keeps it to a few seconds; accuracy is not what is being asserted.

_AR = {"n_lags": 14, "n_forecasts": 7, "epochs": 2}


def _fit_ar(**over: Any) -> BaseModel:
    if importlib.util.find_spec("neuralprophet") is None:
        pytest.skip("optional dependency 'neuralprophet' not installed")
    y, _ = _golden_series(n=200)
    model = get_model("neuralprophet")({**_AR, **over}, _ctx(horizon=7))
    model.fit(y)
    return model


def test_autoregression_returns_a_finite_value_for_every_step() -> None:
    """Reading `yhat1` down the diagonal frame would give one value and six NaNs."""
    df = _fit_ar().predict(7)
    assert len(df) == 7
    assert np.isfinite(df["yhat"].to_numpy()).all()
    assert np.isfinite(df["yhat_lower"].to_numpy()).all()
    assert np.isfinite(df["yhat_upper"].to_numpy()).all()


def test_a_horizon_shorter_than_n_forecasts_reads_the_first_steps_not_the_last() -> None:
    """`make_future_dataframe` clamps to n_lags + n_forecasts, so the tail is steps 5-7."""
    model = _fit_ar()
    short, full = model.predict(3), model.predict(7)
    assert list(short["ds"]) == list(full["ds"][:3])
    assert np.allclose(short["yhat"].to_numpy(), full["yhat"].to_numpy()[:3])


def test_too_few_heads_for_the_horizon_is_an_error_not_a_frame_of_nans() -> None:
    """`dag.check_model_params` refuses this at plan time; predict must not paper over it either."""
    model = _fit_ar(n_forecasts=2)
    with pytest.raises(ModelError, match="does not recurse"):
        model.predict(7)
