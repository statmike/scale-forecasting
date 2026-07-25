"""Tests for the model ABC, registry, and shared helpers (CONTRACTS §1, BUILD 2.2)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scale_forecasting.errors import ModelError
from scale_forecasting.models.base_model import (
    DEFAULT_QUANTILES,
    PREDICTION_COLUMNS,
    BaseModel,
    ModelContext,
    register,
)


def _ctx(**over: Any) -> ModelContext:
    base: dict[str, Any] = {"freq": "D", "horizon": 3, "seed": 0}
    base.update(over)
    return ModelContext(**base)


class _Dummy(BaseModel):
    """Minimal concrete model that uses the residual-interval helper."""

    name = "_dummy"
    runtime = "python"
    family = "statistical"
    supports_exog = False

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        self._level = float(y.mean())
        self._set_residuals(y.to_numpy() - self._level)

    def predict(
        self, horizon: int, X: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        ds = pd.date_range("2026-01-01", periods=horizon, freq="D")
        yhat = np.full(horizon, self._level)
        return self._assemble_frame(ds, self.residual_intervals(yhat, quantiles))


# --- ABC enforcement -----------------------------------------------------------


def test_cannot_instantiate_abstract_base() -> None:
    with pytest.raises(TypeError):
        BaseModel({}, _ctx())  # type: ignore[abstract]


def test_concrete_subclass_instantiates() -> None:
    m = _Dummy({"a": 1}, _ctx())
    assert m.get_params() == {"a": 1}
    assert m.ctx.horizon == 3


def test_default_search_space_is_empty() -> None:
    assert _Dummy.search_space(object()) == {}  # type: ignore[arg-type]


# --- registry ------------------------------------------------------------------


def test_register_adds_class_and_returns_it() -> None:
    from scale_forecasting.models.base_model import _REGISTRY

    returned = register(_Dummy)
    assert returned is _Dummy
    assert _REGISTRY["_dummy"] is _Dummy


def test_register_is_idempotent_for_same_class() -> None:
    register(_Dummy)
    register(_Dummy)  # no error on re-register of the identical class


def test_register_rejects_missing_name() -> None:
    class _NoName(_Dummy):
        name = ""

    with pytest.raises(ModelError, match="must set a class-level 'name'"):
        register(_NoName)


def test_register_rejects_duplicate_name() -> None:
    class _Clash(_Dummy):
        pass

    _Clash.name = "_dummy"  # same name, different class
    with pytest.raises(ModelError, match="duplicate model name"):
        register(_Clash)


# --- residual-interval helper --------------------------------------------------


def test_residual_intervals_are_ordered() -> None:
    m = _Dummy({}, _ctx())
    m.fit(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))
    qm = m.residual_intervals(np.array([3.0, 3.0]), (0.1, 0.5, 0.9))
    assert np.all(qm[0.1] <= qm[0.5])
    assert np.all(qm[0.5] <= qm[0.9])


def test_residual_intervals_center_on_median_residual() -> None:
    m = _Dummy({}, _ctx())
    # symmetric residuals → median residual ~0 → 0.5 band ~= point forecast
    m.fit(pd.Series([2.0, 4.0, 6.0]))  # mean 4 → residuals [-2, 0, 2]
    qm = m.residual_intervals(np.array([10.0]), (0.5,))
    assert qm[0.5][0] == pytest.approx(10.0)


def test_residual_intervals_fallback_without_residuals() -> None:
    m = _Dummy({}, _ctx())  # never fit → no residuals
    qm = m.residual_intervals(np.array([5.0, 6.0]))
    for q in DEFAULT_QUANTILES:
        assert np.allclose(qm[q], [5.0, 6.0])


# --- canonical frame assembly --------------------------------------------------


def test_assemble_frame_shape_and_dtypes() -> None:
    m = _Dummy({}, _ctx())
    m.fit(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))
    df = m.predict(3)
    assert list(df.columns) == list(PREDICTION_COLUMNS)
    assert len(df) == 3
    assert df["ds"].dtype == np.dtype("datetime64[ns]")
    assert df["yhat"].dtype == np.float64
    assert str(df["quantiles"].dtype) == "string"
    # bounds ordered on every row
    assert (df["yhat_lower"] <= df["yhat"]).all()
    assert (df["yhat"] <= df["yhat_upper"]).all()


def test_assemble_frame_quantiles_json_roundtrips() -> None:
    m = _Dummy({}, _ctx())
    m.fit(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))
    df = m.predict(1)
    parsed = json.loads(df["quantiles"].iloc[0])
    assert set(parsed) == {"0.1", "0.5", "0.9"}
    assert parsed["0.1"] <= parsed["0.5"] <= parsed["0.9"]
