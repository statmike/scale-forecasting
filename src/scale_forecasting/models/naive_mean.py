"""Historical-mean baseline — forecast the mean of all observations.

One model, one file. Runtime python, statistical family. No native intervals; uses the
residual-quantile helper. A flat forecast at the sample mean; the natural floor every
other model should beat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..errors import ModelError
from ..features import invert_transform
from .base_model import DEFAULT_QUANTILES, BaseModel, register


class NaiveMean(BaseModel):
    """Forecast the historical mean at every step."""

    name = "naive_mean"
    runtime = "python"
    family = "statistical"
    supports_exog = False
    supports_native_intervals = False
    # Nothing is estimated — the mean *is* the data. Taking in new observations is one line and
    # re-running it is not a refit, so this model can answer both frozen questions honestly.
    supports_recondition = True
    supports_extrapolate = True

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        vals = y.astype(float).to_numpy()
        if len(vals) < 1:
            raise ModelError("naive_mean requires at least 1 observation")
        self._history = vals
        self._mean = float(np.mean(vals))
        self._last_date = y.index[-1]
        self._set_residuals(vals - self._mean)

    def predict(
        self,
        horizon: int,
        X: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        mean = np.full(horizon, self._mean)
        qmap_t = self.residual_intervals(mean, quantiles)
        t, lam = self.ctx.transform, self.ctx.transform_lambda
        qmap = {q: invert_transform(v, t, lam) for q, v in qmap_t.items()}
        ds = self._forecast_index(horizon)
        return self._assemble_frame(ds, qmap, raw=invert_transform(mean, t, lam))

    def recondition(self, y_new: pd.Series, X_new: pd.DataFrame | None = None) -> None:
        self._history = np.concatenate([self._history, y_new.astype(float).to_numpy()])
        self._mean = float(np.mean(self._history))
        self._last_date = y_new.index[-1]
        self._set_residuals(self._history - self._mean)


register(NaiveMean)
