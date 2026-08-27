"""Croston's method for intermittent demand (classic / SBA / TSB).

One model, one file. Runtime python, statistical family. No native intervals; uses the
residual-quantile helper. Croston decomposes a sparse (many-zero) series into demand
*sizes* and inter-arrival *intervals* and smooths each with a single exponential filter;
the flat forecast is ``size / interval``. Variants (``variant`` param, HPO-tunable):

* ``classic`` — Croston's original estimator.
* ``sba`` — Syntetos-Boylan Approximation: multiplies by ``(1 - alpha/2)`` to debias.
* ``tsb`` — Teunter-Syntetos-Babai: smooths demand *probability* instead of interval, so
  the forecast decays toward zero during long gaps (obsolescence-aware).

On a dense (non-intermittent) series every step has demand, so the estimator reduces to a
plain exponential smoother of the level.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..errors import ModelError
from ..features import invert_transform
from .base_model import DEFAULT_QUANTILES, BaseModel, register

if TYPE_CHECKING:
    import optuna

_VARIANTS = ("classic", "sba", "tsb")


class Croston(BaseModel):
    """Intermittent-demand forecaster (Croston / SBA / TSB)."""

    name = "croston"
    runtime = "python"
    family = "statistical"
    supports_exog = False
    supports_native_intervals = False

    def _estimate(self, vals: np.ndarray) -> tuple[float, np.ndarray]:
        """Return ``(level, in_sample_one_step)`` for the configured variant."""
        alpha = self._alpha
        n = len(vals)
        fitted = np.zeros(n)
        if self._variant == "tsb":
            nz = vals[vals > 0]
            z = float(nz[0]) if nz.size else 0.0
            p = float(np.mean(vals > 0))
            for t in range(n):
                fitted[t] = p * z
                d = vals[t]
                if d > 0:
                    z += alpha * (d - z)
                    p += alpha * (1.0 - p)
                else:
                    p += alpha * (0.0 - p)
            return p * z, fitted

        # classic / sba: smooth demand size z and interval p, forecast z / p.
        correction = 1.0 - alpha / 2.0 if self._variant == "sba" else 1.0
        z: float | None = None
        p: float | None = None
        q = 1  # periods since the last non-zero demand
        for t in range(n):
            fitted[t] = (z / p) * correction if (z is not None and p) else 0.0
            d = vals[t]
            if d > 0:
                if z is None:
                    z, p = d, float(q)
                else:
                    z += alpha * (d - z)
                    p += alpha * (q - p)
                q = 1
            else:
                q += 1
        level = (z / p) * correction if (z is not None and p) else 0.0
        return level, fitted

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        vals = y.astype(float).to_numpy()
        if len(vals) < 1:
            raise ModelError("croston requires at least 1 observation")
        self._alpha = float(self.params.get("alpha", 0.1))
        self._variant = str(self.params.get("variant", "classic"))
        if self._variant not in _VARIANTS:
            raise ModelError(f"croston variant must be one of {_VARIANTS}, got '{self._variant}'")
        self._level, fitted = self._estimate(vals)
        self._last_date = y.index[-1]
        self._set_residuals(vals - fitted)

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
        return {
            "alpha": trial.suggest_float("alpha", 0.01, 0.4),
            "variant": trial.suggest_categorical("variant", list(_VARIANTS)),
        }


register(Croston)
