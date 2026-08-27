"""Seasonal-naive baseline — repeat the last observed seasonal cycle.

One model, one file. Runtime python, statistical family. No native intervals; uses the
base-class residual-quantile helper for its PI. A strong, dependency-free baseline: the
forecast for step ``h`` is the observation one full season earlier. When the frequency has
no usable seasonal period (or history is shorter than a season) it degrades to the plain
naive forecast (repeat the last value).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..errors import ModelError
from ..features import invert_transform
from ..seasonality import seasonal_period
from .base_model import DEFAULT_QUANTILES, BaseModel, register


class NaiveSeasonal(BaseModel):
    """Repeat the last full season (falls back to the last value when non-seasonal)."""

    name = "naive_seasonal"
    runtime = "python"
    family = "statistical"
    supports_exog = False
    supports_native_intervals = False

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        if len(y) < 1:
            raise ModelError("naive_seasonal requires at least 1 observation")
        vals = y.astype(float).to_numpy()
        period = seasonal_period(self.ctx.freq)
        # No usable seasonality (e.g. yearly freq), or history shorter than a full season → naive.
        self._period = period if period > 1 and len(vals) >= period else 1
        self._season = vals[-self._period :]
        self._last_date = y.index[-1]
        # In-sample one-season-ahead residuals feed the interval helper.
        if len(vals) > self._period:
            self._set_residuals(vals[self._period :] - vals[: -self._period])

    def predict(
        self,
        horizon: int,
        X: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        mean = self._season[np.arange(horizon) % self._period]
        qmap_t = self.residual_intervals(mean, quantiles)
        t, lam = self.ctx.transform, self.ctx.transform_lambda
        qmap = {q: invert_transform(v, t, lam) for q, v in qmap_t.items()}
        ds = self._future_index(self._last_date, horizon)
        return self._assemble_frame(ds, qmap)


register(NaiveSeasonal)
