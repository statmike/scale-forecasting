"""Theta method — simple, strong baseline; proves the model pattern first.

One model, one file (CONTRACTS §1): a ``BaseModel`` subclass ending in ``register(...)``.
Runtime python, statistical family, native prediction intervals (statsmodels ThetaModel
emits its own PI, so ``supports_native_intervals = True``).

This file is the template every other Python model follows: fit on the transformed target,
predict the canonical §2.1 frame in *original* units (transform inverted here), build the
quantile map, and hand it to ``_assemble_frame``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.forecasting.theta import ThetaModel

from ..errors import ModelError
from ..features import invert_transform
from .base_model import DEFAULT_QUANTILES, BaseModel, register

if TYPE_CHECKING:
    import optuna

# Seasonal period per frequency; Theta needs a period for its seasonal decomposition.
_PERIOD: dict[str, int] = {"D": 7, "W": 52, "M": 12, "MS": 12, "H": 24}


class ThetaModelWrapper(BaseModel):
    """Theta forecaster (statsmodels)."""

    name = "theta"
    runtime = "python"
    family = "statistical"
    supports_exog = False
    supports_native_intervals = True

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        if len(y) < 2:
            raise ModelError("theta requires at least 2 observations")
        period = _PERIOD.get(self.ctx.freq, 7)
        deseasonalize = len(y) >= 2 * period
        self._last_date = y.index[-1]
        self._fitted = ThetaModel(
            y.astype(float), period=period, deseasonalize=deseasonalize
        ).fit()

    def predict(
        self,
        horizon: int,
        X: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        mean = np.asarray(self._fitted.forecast(horizon), dtype=float)
        # Theta's forecast SE ≈ (upper − lower) / (2 z) from a symmetric PI; use it to
        # place arbitrary requested quantiles, so the frame honors any quantile set.
        pi = self._fitted.prediction_intervals(horizon, alpha=0.2)  # ~10/90
        z90 = norm.ppf(0.9)
        sigma = (pi["upper"].to_numpy() - pi["lower"].to_numpy()) / (2.0 * z90)

        t = self.ctx.transform
        qmap = {
            q: invert_transform(mean + norm.ppf(q) * sigma, t) for q in quantiles
        }
        ds = self._future_index(self._last_date, horizon)
        return self._assemble_frame(ds, qmap)

    @classmethod
    def search_space(cls, trial: optuna.Trial) -> dict[str, Any]:
        return {"deseasonalize": trial.suggest_categorical("deseasonalize", [True, False])}


register(ThetaModelWrapper)
