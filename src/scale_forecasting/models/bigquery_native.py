"""BigQuery-native models — arima_plus / timesfm (CONTRACTS §1, §5).

These register through the *same* factory (``runtime="bigquery"``) so the router and registry
treat them uniformly with the Python models, but they are **executed as SQL** by
``engines/bigquery_engine.py`` — not by this Python code. In Arc A they are metadata-only:
they exist in the factory (so fan-out, routing, and the registry see them), and their
``fit``/``predict`` raise :class:`NotImplementedError` pointing at BUILD step B3 where the SQL
templates land.

Two models, two classes, one file — they share nothing but a runtime and an identical
"executed elsewhere" stance, so a thin in-file base keeps each a real registered model while
avoiding copies of the same stub (the one-model-one-file rule is about *authorship locality*, and
both native models are authored here against the same BigQuery contract).
"""

from __future__ import annotations

import pandas as pd

from .base_model import BaseModel, register

_ARC_B = "BigQuery-native models execute as SQL in engines/bigquery_engine.py (BUILD step B3)"


class _BigQueryNativeModel(BaseModel):
    """Shared stance for native models: registered here, executed as SQL in Arc B."""

    runtime = "bigquery"
    family = "native"
    supports_native_intervals = True  # ML.FORECAST returns prediction-interval bounds

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        raise NotImplementedError(_ARC_B)

    def predict(
        self,
        horizon: int,
        X: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> pd.DataFrame:
        raise NotImplementedError(_ARC_B)


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
