"""Theta method — simple, strong baseline; proves the model pattern first.

One model, one file: a ``BaseModel`` subclass ending in ``register(...)``.
Runtime python, statistical family, native prediction intervals (statsmodels ThetaModel
emits its own PI, so ``supports_native_intervals = True``).

This file is the template every other Python model follows: fit on the transformed target,
predict the canonical frame in *original* units (transform inverted here), build the
quantile map, and hand it to ``_assemble_frame``.
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


class ThetaModel(BaseModel):
    """Theta forecaster (statsmodels)."""

    name = "theta"
    runtime = "python"
    family = "statistical"
    supports_exog = False
    supports_native_intervals = True

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        # Lazy import: keep the model stack off the module top (lean launch point).
        from statsmodels.tsa.forecasting.theta import ThetaModel as _StatsmodelsTheta

        if len(y) < 2:
            raise ModelError("theta requires at least 2 observations")
        period = seasonal_period(self.ctx.freq)
        # HPO knob (search_space): an explicit `deseasonalize` param overrides the default
        # data-driven rule (deseasonalize only with ≥2 full seasons). Absent → the default.
        deseasonalize = bool(self.params.get("deseasonalize", len(y) >= 2 * period))
        self._last_date = y.index[-1]
        self._fitted = _StatsmodelsTheta(
            y.astype(float), period=period, deseasonalize=deseasonalize
        ).fit()

    def predict(
        self,
        horizon: int,
        X: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        from scipy.stats import norm  # lazy: keep scipy off the module top (lean launch point)

        mean = np.asarray(self._fitted.forecast(horizon), dtype=float)
        # Theta's forecast SE ≈ (upper − lower) / (2 z) from a symmetric PI; use it to
        # place arbitrary requested quantiles, so the frame honors any quantile set.
        pi = self._fitted.prediction_intervals(horizon, alpha=0.2)  # ~10/90
        z90 = norm.ppf(0.9)
        sigma = (pi["upper"].to_numpy() - pi["lower"].to_numpy()) / (2.0 * z90)

        t, lam = self.ctx.transform, self.ctx.transform_lambda
        qmap = {q: invert_transform(mean + norm.ppf(q) * sigma, t, lam) for q in quantiles}
        ds = self._future_index(self._last_date, horizon)
        return self._assemble_frame(ds, qmap)

    @classmethod
    def search_space(cls, trial: optuna.Trial) -> dict[str, Any]:
        return {"deseasonalize": trial.suggest_categorical("deseasonalize", [True, False])}


register(ThetaModel)
