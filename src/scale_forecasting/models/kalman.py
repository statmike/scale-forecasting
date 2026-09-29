"""Kalman — state-space linear Kalman filter with harmonic seasonality (statsmodels).

One model, one file. Runtime python, statistical family. Fits a linear Gaussian
state-space model with local linear trend, trigonometric Fourier seasonal states,
and autoregressive state dynamics of order ``dim_x``, estimated by maximum
likelihood and filtered via exact Kalman recursions. Supports exogenous
covariates, native Gaussian prediction intervals, and zero-refit state
reconditioning (``append(refit=False)``).
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


class KalmanForecaster(BaseModel):
    """State-space linear Kalman filter forecaster (statsmodels)."""

    name = "kalman"
    runtime = "python"
    family = "statistical"
    supports_exog = True
    supports_native_intervals = True
    # `append(refit=False)` advances the Kalman state filter over new observations while holding
    # the state transition and covariance matrices fixed.
    supports_recondition = True
    supports_extrapolate = True
    package = "statsmodels"
    package_url = "https://www.statsmodels.org/"

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        from statsmodels.tsa.statespace.structural import UnobservedComponents

        if len(y) < 4:
            raise ModelError("kalman requires at least 4 observations")
        period = seasonal_period(self.ctx.freq)
        dim_x = int(self.params.get("dim_x", 1))
        harmonics_req = int(self.params.get("harmonics", min(3, max(1, period // 2))))
        has_seasonal = period > 1 and len(y) >= 2 * period
        freq_seasonal = (
            [{"period": period, "harmonics": min(harmonics_req, period // 2)}]
            if has_seasonal
            else None
        )

        self._last_date = y.index[-1]
        self._fitted = UnobservedComponents(
            y.astype(float),
            exog=X,
            level="local linear trend",
            freq_seasonal=freq_seasonal,
            autoregressive=max(0, dim_x),
        ).fit(disp=False)

    def predict(
        self,
        horizon: int,
        X: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        from scipy.stats import norm

        fc = self._fitted.get_forecast(self._forecast_steps(horizon), exog=self._forecast_exog(X))
        mean = np.asarray(fc.predicted_mean, dtype=float)[-horizon:]
        sigma = np.asarray(fc.se_mean, dtype=float)[-horizon:]
        sigma = np.where(np.isfinite(sigma) & (sigma >= 0.0), sigma, 0.0)
        t, lam = self.ctx.transform, self.ctx.transform_lambda
        qmap = {q: invert_transform(mean + norm.ppf(q) * sigma, t, lam) for q in quantiles}
        ds = self._forecast_index(horizon)
        return self._assemble_frame(ds, qmap, raw=invert_transform(mean, t, lam))

    def recondition(self, y_new: pd.Series, X_new: pd.DataFrame | None = None) -> None:
        self._fitted = self._fitted.append(y_new.astype(float), exog=X_new, refit=False)
        self._last_date = y_new.index[-1]

    @classmethod
    def search_space(cls, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "dim_x": trial.suggest_int("dim_x", 0, 3),
            "harmonics": trial.suggest_int("harmonics", 1, 4),
        }


register(KalmanForecaster)
