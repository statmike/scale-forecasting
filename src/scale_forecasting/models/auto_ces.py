"""AutoCES — automatic Complex Exponential Smoothing (statsforecast).

One model, one file. Runtime python, statistical family. Fits Svetunkov &
Kourentzes' Complex Exponential Smoothing via ``statsforecast.models.AutoCES``,
which models non-stationary level and seasonal dynamics using complex-valued
smoothing parameters. Emits native prediction intervals.
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

_CES_MODELS = ("Z", "N", "S", "P")


class AutoCes(BaseModel):
    """Complex Exponential Smoothing with automatic model selection (statsforecast)."""

    name = "auto_ces"
    runtime = "python"
    family = "statistical"
    supports_exog = False
    supports_native_intervals = True
    # `AutoCES.forward` filters the updated series through the already-selected CES state model.
    supports_recondition = True
    supports_extrapolate = True
    package = "statsforecast"
    package_url = "https://nixtlaverse.nixtla.io/statsforecast/"
    optional_import = "statsforecast"
    optional_extra = "models-stats"

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        try:
            from statsforecast.models import AutoCES
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            self.require_available()
            raise ModelError("statsforecast not installed; install the 'models' extra") from e
        if len(y) < 4:
            raise ModelError("auto_ces requires at least 4 observations")

        period = seasonal_period(self.ctx.freq)
        eff_period = period if len(y) >= 2 * period else 1
        model_spec = str(self.params.get("model", "Z"))
        if model_spec not in _CES_MODELS:
            raise ModelError(f"auto_ces model must be one of {_CES_MODELS}, got {model_spec!r}")
        if eff_period == 1 and model_spec in ("S", "P"):
            model_spec = "N"

        self._last_date = y.index[-1]
        self._y_history = y.astype(float).copy()
        self._reconditioned = False
        self._fitted = AutoCES(season_length=eff_period, model=model_spec).fit(
            y=self._y_history.to_numpy(dtype=float)
        )

    def predict(
        self,
        horizon: int,
        X: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        from scipy.stats import norm

        steps = self._forecast_steps(horizon)
        if self._reconditioned:
            res = self._fitted.forward(y=self._y_history.to_numpy(dtype=float), h=steps, level=[80])
        else:
            res = self._fitted.predict(h=steps, level=[80])

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
        self._last_date = y_new.index[-1]
        self._reconditioned = True

    @classmethod
    def search_space(cls, trial: optuna.Trial) -> dict[str, Any]:
        return {"model": trial.suggest_categorical("model", ["Z", "N", "S"])}


register(AutoCes)
