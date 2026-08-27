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

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        vals = y.astype(float).to_numpy()
        if len(vals) < 2:
            raise ModelError("naive_drift requires at least 2 observations")
        n = len(vals)
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
        mean = self._last + self._slope * np.arange(1, horizon + 1)
        qmap_t = self.residual_intervals(mean, quantiles)
        t, lam = self.ctx.transform, self.ctx.transform_lambda
        qmap = {q: invert_transform(v, t, lam) for q, v in qmap_t.items()}
        ds = self._future_index(self._last_date, horizon)
        return self._assemble_frame(ds, qmap)


register(NaiveDrift)
