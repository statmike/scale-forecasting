"""Drift baseline — extrapolate the straight line from the first to the last point.

One model, one file. Runtime python, statistical family. No native intervals; uses the
residual-quantile helper. The forecast continues at the average per-step change over the
history: ``slope = (y[-1] - y[0]) / (n - 1)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..errors import ModelError
from ..features import invert_transform
from .base_model import DEFAULT_QUANTILES, BaseModel, register


class NaiveDrift(BaseModel):
    """Linear drift from the first to the last observation."""

    name = "naive_drift"
    runtime = "python"
    family = "statistical"
    supports_exog = False
    supports_native_intervals = False
    # The slope is arithmetic on the endpoints, not an estimate — recomputing it over a longer
    # history is conditioning forward, not refitting.
    supports_recondition = True
    supports_extrapolate = True

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        vals = y.astype(float).to_numpy()
        if len(vals) < 2:
            raise ModelError("naive_drift requires at least 2 observations")
        n = len(vals)
        self._history = vals
        self._last = float(vals[-1])
        self._slope = (vals[-1] - vals[0]) / (n - 1)
        self._last_date = y.index[-1]
        # In-sample fit is the line through (0, y[0]) at this slope.
        fitted = vals[0] + self._slope * np.arange(n)
        self._set_residuals(vals - fitted)

    def predict(
        self,
        horizon: int,
        X: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        # Steps are counted from the *fit's* last point, so an advanced origin keeps sliding down
        # the same line rather than restarting it — the drift does not reset when the origin moves.
        steps = self._forecast_steps(horizon)
        mean = (self._last + self._slope * np.arange(1, steps + 1))[-horizon:]
        qmap_t = self.residual_intervals(mean, quantiles)
        t, lam = self.ctx.transform, self.ctx.transform_lambda
        qmap = {q: invert_transform(v, t, lam) for q, v in qmap_t.items()}
        ds = self._forecast_index(horizon)
        return self._assemble_frame(ds, qmap, raw=invert_transform(mean, t, lam))

    def recondition(self, y_new: pd.Series, X_new: pd.DataFrame | None = None) -> None:
        self._history = np.concatenate([self._history, y_new.astype(float).to_numpy()])
        n = len(self._history)
        self._last = float(self._history[-1])
        self._slope = (self._history[-1] - self._history[0]) / (n - 1)
        self._last_date = y_new.index[-1]
        self._set_residuals(self._history - (self._history[0] + self._slope * np.arange(n)))


register(NaiveDrift)
