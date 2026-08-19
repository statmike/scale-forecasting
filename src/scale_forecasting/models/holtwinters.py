"""Holt-Winters exponential smoothing (statsmodels).

One model, one file. Runtime python, statistical family. No native
intervals — uses the base-class residual-quantile helper for its PI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from ..errors import ModelError
from ..features import invert_transform
from ..seasonality import seasonal_period
from .base_model import DEFAULT_QUANTILES, BaseModel, register


class HoltWinters(BaseModel):
    """Holt-Winters (additive trend + seasonality) exponential smoothing."""

    name = "holtwinters"
    runtime = "python"
    family = "statistical"
    supports_exog = False
    supports_native_intervals = False

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        if len(y) < 2:
            raise ModelError("holtwinters requires at least 2 observations")
        period = seasonal_period(self.ctx.freq)
        seasonal = "add" if len(y) >= 2 * period else None
        self._last_date = y.index[-1]
        self._fitted = ExponentialSmoothing(
            y.astype(float),
            trend="add",
            seasonal=seasonal,
            seasonal_periods=period if seasonal else None,
            initialization_method="estimated",
        ).fit()
        self._set_residuals(self._fitted.resid.to_numpy())

    def predict(
        self,
        horizon: int,
        X: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        mean = np.asarray(self._fitted.forecast(horizon), dtype=float)
        qmap_t = self.residual_intervals(mean, quantiles)
        t, lam = self.ctx.transform, self.ctx.transform_lambda
        qmap = {q: invert_transform(v, t, lam) for q, v in qmap_t.items()}
        ds = self._future_index(self._last_date, horizon)
        return self._assemble_frame(ds, qmap)


register(HoltWinters)
