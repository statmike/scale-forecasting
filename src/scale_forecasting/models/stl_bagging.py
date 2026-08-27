"""STL decomposition + bagged base forecasts.

One model, one file. Runtime python, statistical family. STL splits the
series into trend + seasonal + remainder; the deseasonalized series is forecast with a
simple ARIMA and the seasonal component is projected forward periodically. Prediction
intervals come from **bagging**: block-bootstrap the STL remainder to build an ensemble of
forecast paths, then read empirical quantiles off the ensemble (so intervals are native).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..errors import ModelError
from ..features import invert_transform
from ..seasonality import seasonal_period
from .base_model import DEFAULT_QUANTILES, BaseModel, register

if TYPE_CHECKING:
    import optuna

_N_BAG = 40  # bootstrap replicates for the predictive distribution


class StlBagging(BaseModel):
    """STL decomposition with a bagged remainder for native intervals."""

    name = "stl_bagging"
    runtime = "python"
    family = "statistical"
    supports_exog = False
    supports_native_intervals = True

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        # Lazy import: keep the model stack off the module top (lean launch point).
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.seasonal import STL

        period = seasonal_period(self.ctx.freq)
        if len(y) < 2 * period:
            raise ModelError(f"stl_bagging requires at least {2 * period} observations")
        self._period = period
        self._last_date = y.index[-1]

        # HPO knobs (search_space): STL robustness + the deseasonalized ARIMA (p, q); the
        # differencing order d is held at 1. Defaults reproduce robust STL + ARIMA(1, 1, 1).
        robust = bool(self.params.get("stl_robust", True))
        p = int(self.params.get("arima_p", 1))
        q = int(self.params.get("arima_q", 1))
        stl = STL(y.astype(float), period=period, robust=robust).fit()
        self._seasonal = np.asarray(stl.seasonal, dtype=float)
        self._resid = np.asarray(stl.resid, dtype=float)
        deseasonalized = y.astype(float).to_numpy() - self._seasonal
        # Forecast the (trend + remainder) deseasonalized series with a light ARIMA.
        self._arima = ARIMA(deseasonalized, order=(p, 1, q)).fit()

    def predict(
        self,
        horizon: int,
        X: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        base = np.asarray(self._arima.forecast(horizon), dtype=float)
        # Project the seasonal component forward by repeating the last full period.
        last_season = self._seasonal[-self._period :]
        seasonal_future = np.resize(last_season, horizon)
        center = base + seasonal_future

        # Bagging: add block-bootstrapped remainder draws to the point path.
        rng = np.random.default_rng(self.ctx.seed)
        clean_resid = self._resid[~np.isnan(self._resid)]
        paths = np.empty((_N_BAG, horizon), dtype=float)
        for b in range(_N_BAG):
            draw = rng.choice(clean_resid, size=horizon, replace=True)
            paths[b] = center + draw

        # Empirical quantiles off the bagged ensemble — monotonic in q, so bounds stay
        # ordered by construction.
        t, lam = self.ctx.transform, self.ctx.transform_lambda
        qmap = {q: invert_transform(np.quantile(paths, q, axis=0), t, lam) for q in quantiles}
        ds = self._future_index(self._last_date, horizon)
        return self._assemble_frame(ds, qmap)

    @classmethod
    def search_space(cls, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "stl_robust": trial.suggest_categorical("stl_robust", [True, False]),
            "arima_p": trial.suggest_int("arima_p", 0, 2),
            "arima_q": trial.suggest_int("arima_q", 0, 2),
        }


register(StlBagging)
