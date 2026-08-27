"""Factory gate.

The factory is the whole point of one-model-one-file: importing ``models`` registers every
model by name, ``get_model`` resolves a name to its class, and ``list_models`` enumerates
them. This proves the full suite registered (all Python + BigQuery-native models), that
resolution round-trips, and that an unknown name raises ``ModelError`` listing what's known.
"""

from __future__ import annotations

import pytest

from scale_forecasting.errors import ModelError
from scale_forecasting.models import get_model, list_models
from scale_forecasting.models.base_model import BaseModel

# Every model in the suite, by runtime. Kept explicit (not derived from
# list_models) so the test fails loudly if a model silently stops registering.
_PYTHON_MODELS = {
    "theta",
    "holtwinters",
    "sarimax",
    "ucm",
    "xgboost",
    "lightgbm",
    "stl_bagging",
    "prophet",
    "neuralprophet",
    "naive_seasonal",
    "naive_drift",
    "naive_mean",
    "naive_moving_average",
    "croston",
    "autoets",
    "regression_lags",
}
_BIGQUERY_MODELS = {"arima_plus", "timesfm"}
_ALL_MODELS = _PYTHON_MODELS | _BIGQUERY_MODELS


def test_all_models_registered() -> None:
    assert set(list_models()) == _ALL_MODELS


def test_list_models_is_sorted() -> None:
    names = list_models()
    assert names == sorted(names)


@pytest.mark.parametrize("name", sorted(_ALL_MODELS))
def test_get_model_resolves_to_subclass(name: str) -> None:
    cls = get_model(name)
    assert issubclass(cls, BaseModel)
    assert cls.name == name


@pytest.mark.parametrize("name", sorted(_PYTHON_MODELS))
def test_python_models_have_python_runtime(name: str) -> None:
    assert get_model(name).runtime == "python"


@pytest.mark.parametrize("name", sorted(_BIGQUERY_MODELS))
def test_bigquery_models_have_bigquery_runtime(name: str) -> None:
    assert get_model(name).runtime == "bigquery"


def test_unknown_model_raises_listing_known() -> None:
    with pytest.raises(ModelError) as excinfo:
        get_model("does_not_exist")
    msg = str(excinfo.value)
    assert "does_not_exist" in msg
    # Error is actionable: it names the models that *are* registered.
    assert "theta" in msg
