"""Tests for the runtime router — partitioning a run's models by execution runtime.

Covers CONTRACTS §6: ``split_by_runtime`` sends the BigQuery-native models
(``arima_plus``/``timesfm``) to the BQ engine and everything else to the Python
runtime, preserving input order, and surfaces unknown names as ``ModelError``.
"""

from __future__ import annotations

from typing import Any

import pytest

from scale_forecasting.config import RunConfig
from scale_forecasting.errors import ModelError
from scale_forecasting.router import split_by_runtime


def _cfg(models: list[str]) -> RunConfig:
    over: dict[str, Any] = {
        "run_name": "router test",
        "data": {"source_table": "t"},
        "models": models,
    }
    return RunConfig(**over)


def test_split_mixed_partitions_by_runtime() -> None:
    py, bq = split_by_runtime(_cfg(["theta", "arima_plus", "xgboost", "timesfm"]))
    assert py == ["theta", "xgboost"]
    assert bq == ["arima_plus", "timesfm"]


def test_split_preserves_input_order_within_each_list() -> None:
    py, bq = split_by_runtime(_cfg(["timesfm", "sarimax", "theta", "arima_plus"]))
    assert py == ["sarimax", "theta"]
    assert bq == ["timesfm", "arima_plus"]


def test_split_all_python() -> None:
    py, bq = split_by_runtime(_cfg(["theta", "sarimax", "xgboost"]))
    assert py == ["theta", "sarimax", "xgboost"]
    assert bq == []


def test_split_all_bigquery() -> None:
    py, bq = split_by_runtime(_cfg(["arima_plus", "timesfm"]))
    assert py == []
    assert bq == ["arima_plus", "timesfm"]


def test_split_unknown_model_raises() -> None:
    with pytest.raises(ModelError):
        split_by_runtime(_cfg(["theta", "not_a_model"]))
