"""Unobserved Components (structural) model (statsmodels).

One model, one file. Runtime python, statistical family. A state-space
structural model (local linear trend + optional seasonal); native Gaussian intervals.
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

_LEVELS = ("local level", "local linear trend", "smooth trend")


class Ucm(BaseModel):
    """Unobserved-components structural time-series model."""

    name = "ucm"
    runtime = "python"
    family = "statistical"
    supports_exog = True
    supports_native_intervals = True

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        # Lazy import: keep the model stack off the module top (lean launch point).
        from statsmodels.tsa.statespace.structural import UnobservedComponents

        if len(y) < 3:
            raise ModelError("ucm requires at least 3 observations")
        period = seasonal_period(self.ctx.freq)
        # HPO knobs (search_space): trend spec, AR order, seasonal on/off. Defaults reproduce
        # the local-linear-trend + auto-seasonal, no-AR behavior.
        level = str(self.params.get("level", "local linear trend"))
        ar = int(self.params.get("autoregressive", 0))
        seasonal_on = bool(self.params.get("seasonal", True))
        seasonal = period if (seasonal_on and len(y) >= 2 * period) else None
        self._last_date = y.index[-1]
        self._fitted = UnobservedComponents(
            y.astype(float),
            exog=X,
            level=level,
            seasonal=seasonal,
            autoregressive=ar if ar > 0 else None,
        ).fit(disp=False)

    def predict(
        self,
        horizon: int,
        X: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        from scipy.stats import norm  # lazy: keep scipy off the module top (lean launch point)

        fc = self._fitted.get_forecast(horizon, exog=X)
        mean = np.asarray(fc.predicted_mean, dtype=float)
        sigma = np.asarray(fc.se_mean, dtype=float)
        t, lam = self.ctx.transform, self.ctx.transform_lambda
        qmap = {q: invert_transform(mean + norm.ppf(q) * sigma, t, lam) for q in quantiles}
        ds = self._future_index(self._last_date, horizon)
        return self._assemble_frame(ds, qmap)

    @classmethod
    def search_space(cls, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "level": trial.suggest_categorical("level", list(_LEVELS)),
            "autoregressive": trial.suggest_int("autoregressive", 0, 1),
            "seasonal": trial.suggest_categorical("seasonal", [True, False]),
        }


register(Ucm)
