"""AutoTheta — optimized / dynamic Theta family (statsforecast).

One model, one file. Runtime python, statistical family. Fits Fiorucci et al.'s
generalized Theta family (STM, OTM, DSTM, DOTM) via
``statsforecast.models.AutoTheta``, selecting the best specification and
emitting native prediction intervals.
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

_DECOMP_TYPES = ("additive", "multiplicative")


class AutoThetaModel(BaseModel):
    """Generalized AutoTheta forecaster (STM / OTM / DSTM / DOTM, statsforecast)."""

    name = "auto_theta"
    runtime = "python"
    family = "statistical"
    supports_exog = False
    supports_native_intervals = True
    # `AutoTheta.forward` evaluates the selected Theta state specification on the extended series.
    supports_recondition = True
    supports_extrapolate = True
    package = "statsforecast"
    package_url = "https://nixtlaverse.nixtla.io/statsforecast/"
    optional_import = "statsforecast"
    optional_extra = "models-stats"

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        try:
            from statsforecast.models import AutoTheta
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            self.require_available()
            raise ModelError("statsforecast not installed; install the 'models' extra") from e
        if len(y) < 4:
            raise ModelError("auto_theta requires at least 4 observations")

        period = seasonal_period(self.ctx.freq)
        eff_period = period if len(y) >= 2 * period else 1
        decomp = str(self.params.get("decomposition_type", "additive"))
        if decomp not in _DECOMP_TYPES:
            raise ModelError(
                f"auto_theta decomposition_type must be one of {_DECOMP_TYPES}, got {decomp!r}"
            )
        # Multiplicative decomposition is undefined on non-positive targets (e.g. under log1p or
        # boxcox transforms or zero-inflated series); fall back to additive when y has <= 0 values.
        y_arr = y.to_numpy(dtype=float)
        if decomp == "multiplicative" and np.any(y_arr <= 0.0):
            decomp = "additive"

        self._last_date = y.index[-1]
        self._y_history = y.astype(float).copy()
        self._reconditioned = False
        self._fitted = AutoTheta(
            season_length=eff_period,
            decomposition_type=decomp,
        ).fit(y=y_arr)

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
        return {
            "decomposition_type": trial.suggest_categorical(
                "decomposition_type", ["additive", "multiplicative"]
            )
        }


register(AutoThetaModel)
