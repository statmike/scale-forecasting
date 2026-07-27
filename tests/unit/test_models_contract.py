"""Shared model contract test (CONTRACTS §1, §2.1, BUILD 2.5).

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


def test_deterministic_under_seed(model_name: str) -> None:
    y, X0 = _golden_series(with_exog=get_model(model_name).supports_exog)

    def run() -> np.ndarray:
        m = _make(model_name)
        m.fit(y, X0)
        fx = X0.iloc[:HORIZON] if (X0 is not None and m.supports_exog) else None
        return m.predict(HORIZON, fx)["yhat"].to_numpy()

    a, b = run(), run()
    assert np.allclose(a, b)


def test_at_least_theta_registered() -> None:
    assert "theta" in list_models()
