"""AutoETS — automatic exponential-smoothing model selection.

One model, one file. Runtime python, statistical family. No native intervals; uses the
residual-quantile helper. Fits every additive ETS configuration (trend on/off, damped
on/off, seasonal on/off) and keeps the one with the best information criterion (AICc by
default). Multiplicative error/trend/seasonal components are deliberately excluded: the
target may be modeled in a transformed, possibly non-positive space where they are
undefined. Where a fixed ``holtwinters`` picks one structure, this searches for it.
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

_CRITERIA = ("aic", "aicc", "bic")


class AutoETS(BaseModel):
    """Exponential smoothing with information-criterion model selection."""

    name = "autoets"
    runtime = "python"
    family = "statistical"
    supports_exog = False
    supports_native_intervals = False

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        # Lazy import: keep the model stack off the module top (lean launch point).
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        vals = y.astype(float)
        if len(vals) < 2:
            raise ModelError("autoets requires at least 2 observations")
        period = seasonal_period(self.ctx.freq)
        can_season = period > 1 and len(vals) >= 2 * period
        criterion = str(self.params.get("criterion", "aicc"))
        allow_damped = bool(self.params.get("allow_damped", True))

        best_score = np.inf
        best_fit = None
        for trend in (None, "add"):
            damped_opts = (False, True) if (trend and allow_damped) else (False,)
            for damped in damped_opts:
                for seasonal in ((None, "add") if can_season else (None,)):
                    try:
                        candidate = ExponentialSmoothing(
                            vals,
                            trend=trend,
                            damped_trend=damped,
                            seasonal=seasonal,
                            seasonal_periods=period if seasonal else None,
                            initialization_method="estimated",
                        ).fit()
                    except Exception:  # noqa: BLE001 - a bad config is skipped, not fatal
                        continue
                    score = float(getattr(candidate, criterion, candidate.aic))
                    if np.isfinite(score) and score < best_score:
                        best_score, best_fit = score, candidate
        if best_fit is None:
            raise ModelError("autoets: no configuration could be fit")
        self._fitted = best_fit
        self._last_date = y.index[-1]
        self._set_residuals(np.asarray(self._fitted.resid, dtype=float))

    def predict(
        self,
        horizon: int,
        X: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        mean = np.asarray(self._fitted.forecast(horizon), dtype=float)
        qmap_t = self.residual_intervals(mean, quantiles)
        t, lam = self.ctx.transform, self.ctx.transform_lambda
        qmap = {q: invert_transform(v, t, lam) for q, v in qmap_t.items()}
        ds = self._future_index(self._last_date, horizon)
        return self._assemble_frame(ds, qmap)

    @classmethod
    def search_space(cls, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "criterion": trial.suggest_categorical("criterion", list(_CRITERIA)),
            "allow_damped": trial.suggest_categorical("allow_damped", [True, False]),
        }


register(AutoETS)
