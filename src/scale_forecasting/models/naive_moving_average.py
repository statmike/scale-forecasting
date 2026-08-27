"""Moving-average baseline — forecast the mean of the last ``window`` observations.

One model, one file. Runtime python, statistical family. No native intervals; uses the
residual-quantile helper. The point forecast is flat at the trailing-window mean; the
window is HPO-tunable (``search_space``) and defaults to the seasonal period (or a week).
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


class NaiveMovingAverage(BaseModel):
    """Flat forecast at the mean of the last ``window`` observations."""

    name = "naive_moving_average"
    runtime = "python"
    family = "statistical"
    supports_exog = False
    supports_native_intervals = False

    def _resolve_window(self, n: int) -> int:
        period = seasonal_period(self.ctx.freq)
        default = period if period > 1 else 7
        window = int(self.params.get("window", default))
        return max(1, min(window, n))

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        vals = y.astype(float).to_numpy()
        if len(vals) < 1:
            raise ModelError("naive_moving_average requires at least 1 observation")
        window = self._resolve_window(len(vals))
        self._level = float(np.mean(vals[-window:]))
        self._last_date = y.index[-1]
        # In-sample one-step residuals: actual[t] minus the mean of the ``window`` values
        # immediately before t, for every t where a full window exists.
        if len(vals) > window:
            prefix = np.concatenate([[0.0], np.cumsum(vals)])
            t = np.arange(window, len(vals))
            preds = (prefix[t] - prefix[t - window]) / window
            self._set_residuals(vals[window:] - preds)

    def predict(
        self,
        horizon: int,
        X: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        mean = np.full(horizon, self._level)
        qmap_t = self.residual_intervals(mean, quantiles)
        t, lam = self.ctx.transform, self.ctx.transform_lambda
        qmap = {q: invert_transform(v, t, lam) for q, v in qmap_t.items()}
        ds = self._future_index(self._last_date, horizon)
        return self._assemble_frame(ds, qmap)

    @classmethod
    def search_space(cls, trial: optuna.Trial) -> dict[str, Any]:
        return {"window": trial.suggest_int("window", 2, 30)}


register(NaiveMovingAverage)
