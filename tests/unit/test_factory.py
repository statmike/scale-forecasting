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
    "auto_arima",
    "auto_ces",
    "auto_theta",
    "autoets",
    "catboost",
    "croston",
    "fft",
    "holtwinters",
    "kalman",
    "lightgbm",
    "naive_drift",
    "naive_mean",
    "naive_moving_average",
    "naive_seasonal",
    "neuralprophet",
    "prophet",
    "random_forest",
    "regression_lags",
    "sarimax",
    "stl_bagging",
    "tbats",
    "theta",
    "ucm",
    "xgboost",
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
    assert isinstance(cls.package, str) and cls.package
    assert isinstance(cls.package_url, str) and cls.package_url.startswith("https://")
    assert isinstance(cls.is_available(), bool)


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


def test_config_accepts_every_registered_model() -> None:
    from scale_forecasting.config import RunConfig

    MODEL_NAMES = list_models()
    for name in MODEL_NAMES:
        cfg = RunConfig(run_name="test", data={"source_table": "p.d.t"}, models=[name])
        assert cfg.models == [name]


def test_filter_available_models_omits_uninstalled_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scale_forecasting.config import RunConfig
    from scale_forecasting.models import filter_available_models

    catboost_cls = get_model("catboost")
    monkeypatch.setattr(catboost_cls, "is_available", classmethod(lambda cls: False))
    assert "catboost" not in list_models(available_only=True)
    assert filter_available_models(["theta", "catboost", "random_forest"]) == [
        "theta",
        "random_forest",
    ]
    cfg = RunConfig(
        run_name="test",
        data={"source_table": "p.d.t"},
        models=["theta", "catboost", "random_forest"],
    ).with_available_models()
    assert cfg.models == ["theta", "random_forest"]
    with pytest.raises(ModelError, match="requires optional package 'catboost'"):
        catboost_cls.require_available()
