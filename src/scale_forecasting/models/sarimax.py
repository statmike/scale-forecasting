"""SARIMAX — seasonal ARIMA with exogenous regressors (statsmodels).

One model, one file (CONTRACTS §1). Runtime python, statistical family. Supports exog
and emits native prediction intervals (Gaussian, from the forecast standard error).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.statespace.sarimax import SARIMAX

from ..errors import ModelError
from ..features import invert_transform
from .base_model import DEFAULT_QUANTILES, BaseModel, register

if TYPE_CHECKING:
    import optuna

# Seasonal period per frequency (the SARIMA "m"). Kept modest so a 100k-series batch stays
# tractable; HPO can widen the (p,d,q) orders later.
_PERIOD: dict[str, int] = {"D": 7, "W": 52, "M": 12, "MS": 12, "H": 24}


class Sarimax(BaseModel):
    """Seasonal ARIMA with optional exogenous regressors."""

    name = "sarimax"
    runtime = "python"
    family = "statistical"
    supports_exog = True
    supports_native_intervals = True

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        if len(y) < 3:
            raise ModelError("sarimax requires at least 3 observations")
        period = _PERIOD.get(self.ctx.freq, 7)
        order = tuple(self.params.get("order", (1, 1, 1)))
        seasonal = len(y) >= 2 * period
        default_seasonal = (0, 1, 1, period) if seasonal else (0, 0, 0, 0)
        seasonal_order = tuple(self.params.get("seasonal_order", default_seasonal))
        self._last_date = y.index[-1]
        self._fitted = SARIMAX(
            y.astype(float),
            exog=X,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
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
        t = self.ctx.transform
        qmap = {q: invert_transform(mean + norm.ppf(q) * sigma, t) for q in quantiles}
        ds = self._future_index(self._last_date, horizon)
        return self._assemble_frame(ds, qmap)

    @classmethod
    def search_space(cls, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "order": (
                trial.suggest_int("p", 0, 3),
                trial.suggest_int("d", 0, 2),
                trial.suggest_int("q", 0, 3),
            )
        }


register(Sarimax)
