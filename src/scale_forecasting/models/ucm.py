"""Unobserved Components (structural) model (statsmodels).

One model, one file (CONTRACTS §1). Runtime python, statistical family. A state-space
structural model (local linear trend + optional seasonal); native Gaussian intervals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.statespace.structural import UnobservedComponents

from ..errors import ModelError
from ..features import invert_transform
from ..seasonality import seasonal_period
from .base_model import DEFAULT_QUANTILES, BaseModel, register


class Ucm(BaseModel):
    """Unobserved-components structural time-series model."""

    name = "ucm"
    runtime = "python"
    family = "statistical"
    supports_exog = True
    supports_native_intervals = True

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        if len(y) < 3:
            raise ModelError("ucm requires at least 3 observations")
        period = seasonal_period(self.ctx.freq)
        seasonal = period if len(y) >= 2 * period else None
        self._last_date = y.index[-1]
        self._fitted = UnobservedComponents(
            y.astype(float),
            exog=X,
            level="local linear trend",
            seasonal=seasonal,
        ).fit(disp=False)

    def predict(
        self,
        horizon: int,
        X: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        fc = self._fitted.get_forecast(horizon, exog=X)
        mean = np.asarray(fc.predicted_mean, dtype=float)
        sigma = np.asarray(fc.se_mean, dtype=float)
        t, lam = self.ctx.transform, self.ctx.transform_lambda
        qmap = {q: invert_transform(mean + norm.ppf(q) * sigma, t, lam) for q in quantiles}
        ds = self._future_index(self._last_date, horizon)
        return self._assemble_frame(ds, qmap)


register(Ucm)
