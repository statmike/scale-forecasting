"""TBATS — Trigonometric seasonality, Box-Cox, ARMA errors, Trend, Seasonal (statsforecast).

One model, one file. Runtime python, statistical family. Uses Nixtla's
Numba-accelerated ``statsforecast.models.TBATS`` implementation of De Livera,
Hyndman & Snyder (2011) with trigonometric Fourier seasonal representations,
damped trend, and ARMA error structure. Emits native prediction intervals.
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


class TbatsModel(BaseModel):
    """TBATS harmonic state-space forecaster (statsforecast)."""

    name = "tbats"
    runtime = "python"
    family = "statistical"
    supports_exog = False
    supports_native_intervals = True
    supports_recondition = False
    supports_extrapolate = True
    package = "statsforecast"
    package_url = "https://nixtlaverse.nixtla.io/statsforecast/"
    optional_import = "statsforecast"
    optional_extra = "models-stats"

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        try:
            from statsforecast.models import TBATS
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            self.require_available()
            raise ModelError("statsforecast not installed; install the 'models' extra") from e
        if len(y) < 4:
            raise ModelError("tbats requires at least 4 observations")

        period = seasonal_period(self.ctx.freq)
        eff_period = period if len(y) >= 2 * period else 1
        y_arr = y.to_numpy(dtype=float)
        # Box-Cox inside TBATS requires strictly positive values and is redundant when the
        # framework has already applied a target transform.
        can_boxcox = bool(np.all(y_arr > 0.0)) and self.ctx.transform == "none"
        use_boxcox_param = self.params.get("use_boxcox")
        use_boxcox = (
            (bool(use_boxcox_param) and can_boxcox) if use_boxcox_param is not None else None
        )
        use_trend = self.params.get("use_trend")
        use_damped_trend = self.params.get("use_damped_trend")
        use_arma_errors = bool(self.params.get("use_arma_errors", True))

        self._last_date = y.index[-1]
        self._fitted = TBATS(
            season_length=[eff_period] if eff_period > 1 else [1],
            use_boxcox=use_boxcox if can_boxcox else False,
            use_trend=use_trend,
            use_damped_trend=use_damped_trend,
            use_arma_errors=use_arma_errors,
        ).fit(y=y_arr)

    def predict(
        self,
        horizon: int,
        X: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        from scipy.stats import norm

        steps = self._forecast_steps(horizon)
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

    @classmethod
    def search_space(cls, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "use_trend": trial.suggest_categorical("use_trend", [True, False]),
            "use_arma_errors": trial.suggest_categorical("use_arma_errors", [True, False]),
        }


register(TbatsModel)
