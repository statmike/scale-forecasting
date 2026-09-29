"""AutoARIMA — automatic seasonal ARIMA order selection (statsforecast).

One model, one file. Runtime python, statistical family. Uses Nixtla's
Hyndman-Khandakar stepwise search (KPSS unit-root tests + AICc minimization)
from ``statsforecast.models.AutoARIMA``. Supports exogenous covariates and
emits native Gaussian prediction intervals.
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


class AutoArima(BaseModel):
    """Automatic ARIMA/SARIMAX order selection (statsforecast)."""

    name = "auto_arima"
    runtime = "python"
    family = "statistical"
    supports_exog = True
    supports_native_intervals = True
    # `AutoARIMA.forward` applies the fitted ARIMA state-space filter to the extended series
    # without re-running the stepwise order search or re-estimating parameters.
    supports_recondition = True
    supports_extrapolate = True
    package = "statsforecast"
    package_url = "https://nixtlaverse.nixtla.io/statsforecast/"
    optional_import = "statsforecast"
    optional_extra = "models-stats"

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        try:
            from statsforecast.models import AutoARIMA
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            self.require_available()
            raise ModelError("statsforecast not installed; install the 'models' extra") from e
        if len(y) < 3:
            raise ModelError("auto_arima requires at least 3 observations")

        period = seasonal_period(self.ctx.freq)
        seasonal_requested = bool(self.params.get("seasonal", True))
        eff_period = period if (seasonal_requested and len(y) >= 2 * period) else 1
        max_p = int(self.params.get("max_p", 3))
        max_q = int(self.params.get("max_q", 3))
        max_d = int(self.params.get("max_d", 2))
        stepwise = bool(self.params.get("stepwise", True))

        self._last_date = y.index[-1]
        self._y_history = y.astype(float).copy()
        self._x_history = X.astype(float).copy() if X is not None else None
        self._reconditioned = False

        x_arr = self._x_history.to_numpy(dtype=float) if self._x_history is not None else None
        self._fitted = AutoARIMA(
            season_length=eff_period,
            max_p=max_p,
            max_q=max_q,
            max_d=max_d,
            max_P=1,
            max_Q=1,
            seasonal=eff_period > 1,
            stepwise=stepwise,
        ).fit(y=self._y_history.to_numpy(dtype=float), X=x_arr)

    def predict(
        self,
        horizon: int,
        X: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        from scipy.stats import norm

        steps = self._forecast_steps(horizon)
        exog_future = self._forecast_exog(X)
        x_fut = exog_future.to_numpy(dtype=float) if exog_future is not None else None
        if self._reconditioned:
            x_hist = self._x_history.to_numpy(dtype=float) if self._x_history is not None else None
            res = self._fitted.forward(
                y=self._y_history.to_numpy(dtype=float),
                h=steps,
                X=x_hist,
                X_future=x_fut,
                level=[80],
            )
        else:
            res = self._fitted.predict(h=steps, X=x_fut, level=[80])

        mean = np.asarray(res["mean"], dtype=float)[-horizon:]
        z90 = float(norm.ppf(0.9))
        hi = np.asarray(res["hi-80"], dtype=float)[-horizon:]
        lo = np.asarray(res["lo-80"], dtype=float)[-horizon:]
        sigma = np.maximum(0.0, (hi - lo) / (2.0 * z90))
        sigma = np.where(np.isfinite(sigma), sigma, 0.0)

        t, lam = self.ctx.transform, self.ctx.transform_lambda
        qmap = {q: invert_transform(mean + norm.ppf(q) * sigma, t, lam) for q in quantiles}
        ds = self._forecast_index(horizon)
        return self._assemble_frame(ds, qmap, raw=invert_transform(mean, t, lam))

    def recondition(self, y_new: pd.Series, X_new: pd.DataFrame | None = None) -> None:
        self._y_history = pd.concat([self._y_history, y_new.astype(float)])
        if self._x_history is not None and X_new is not None:
            self._x_history = pd.concat([self._x_history, X_new.astype(float)])
        self._last_date = y_new.index[-1]
        self._reconditioned = True

    def diagnostics(self) -> dict[str, Any]:
        model_dict = getattr(self._fitted, "model_", None)
        if not isinstance(model_dict, dict):
            return {}
        out: dict[str, Any] = {}
        arma = model_dict.get("arma")
        if isinstance(arma, (list, tuple)) and len(arma) >= 7:
            out["arima_order"] = f"({arma[0]},{arma[5]},{arma[1]})"
            out["seasonal_order"] = f"({arma[2]},{arma[6]},{arma[3]})[{arma[4]}]"
        for key in ("aic", "aicc", "bic"):
            val = model_dict.get(key)
            if isinstance(val, (int, float, np.floating)) and np.isfinite(float(val)):
                out[key] = float(val)
        return out

    @classmethod
    def search_space(cls, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "max_p": trial.suggest_int("max_p", 1, 4),
            "max_q": trial.suggest_int("max_q", 1, 4),
            "seasonal": trial.suggest_categorical("seasonal", [True, False]),
        }


register(AutoArima)
