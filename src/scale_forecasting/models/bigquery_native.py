"""BigQuery-native models — arima_plus / timesfm.

These register through the *same* factory (``runtime="bigquery"``) so the router and registry
treat them uniformly with the Python models, but they are **executed as SQL** by
``engines/bigquery_engine.py`` — never by this Python code. That is by design, not a gap: the
classes exist so fan-out, routing, and the registry see them as first-class models, while the
actual ``CREATE MODEL`` / ``ML.FORECAST`` / ``AI.FORECAST`` runs in BigQuery. The in-process
``fit``/``predict`` therefore raise :class:`BigQueryNativeExecutionError` — reaching them means a
native model was mistakenly dispatched to the Python worker path instead of the BigQuery engine.

Two models, two classes, one file — they share nothing but a runtime and an identical
"executed in BigQuery" stance, so a thin in-file base keeps each a real registered model while
avoiding copies of the same guard (the one-model-one-file rule is about *authorship locality*, and
both native models are authored here against the same BigQuery contract).
"""

from __future__ import annotations

import pandas as pd

from .base_model import BaseModel, register

_EXECUTED_IN_BIGQUERY = (
    "BigQuery-native models execute as SQL in engines/bigquery_engine.py, not in the Python "
    "worker — this method should never be called. Route native models through the BigQuery engine."
)


class BigQueryNativeExecutionError(NotImplementedError):
    """Raised if a BigQuery-native model's in-process fit/predict is called.

    Native models run as SQL in :mod:`engines.bigquery_engine`; hitting this means one was
    dispatched to the Python worker path by mistake. Subclasses :class:`NotImplementedError` so
    existing ``except NotImplementedError`` handlers and tests keep working.
    """


class _BigQueryNativeModel(BaseModel):
    """Shared stance for native models: registered here, executed as SQL in BigQuery."""

    runtime = "bigquery"
    family = "native"
    supports_native_intervals = True  # ML.FORECAST returns prediction-interval bounds

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        raise BigQueryNativeExecutionError(_EXECUTED_IN_BIGQUERY)

    def predict(
        self,
        horizon: int,
        X: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> pd.DataFrame:
        raise BigQueryNativeExecutionError(_EXECUTED_IN_BIGQUERY)


class ArimaPlus(_BigQueryNativeModel):
    """BigQuery ML ``ARIMA_PLUS`` (univariate; custom holidays for parity with features.py)."""

    name = "arima_plus"
    supports_exog = False


class TimesFm(_BigQueryNativeModel):
    """BigQuery ``AI.FORECAST`` with TimesFM (pretrained; no training step)."""

    name = "timesfm"
    supports_exog = False


register(ArimaPlus)
register(TimesFm)
