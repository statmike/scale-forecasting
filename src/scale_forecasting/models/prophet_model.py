"""Prophet — additive trend/seasonality/holiday model.

One model, one file. Runtime python, statistical family. Prophet is an
optional dependency, imported lazily in ``fit`` so the model registers without it. It emits
its own uncertainty interval; we read the symmetric band once and place arbitrary requested
quantiles from it (same trick as ``theta``), so any quantile set is honored. Exogenous
regressors are wired through ``add_regressor``; ``ctx.holidays`` feeds Prophet's holiday
frame when present.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from ..errors import ModelError
from ..features import invert_transform
from .base_model import DEFAULT_QUANTILES, BaseModel, register

if TYPE_CHECKING:
    import optuna

_INTERVAL_WIDTH = 0.8  # ~10/90 band; sigma is backed out from it for arbitrary quantiles


class ProphetModel(BaseModel):
    """Facebook Prophet forecaster."""

    name = "prophet"
    runtime = "python"
    family = "statistical"
    supports_exog = True
    supports_native_intervals = True

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        try:
            from prophet import Prophet
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise ModelError("prophet not installed; install the 'models' extra") from e
        if len(y) < 2:
            raise ModelError("prophet requires at least 2 observations")

        self._last_date = y.index[-1]
        self._exog_cols = list(X.columns) if X is not None else []

        holidays = self.ctx.holidays if self.ctx.holidays is not None else None
        model = Prophet(interval_width=_INTERVAL_WIDTH, holidays=holidays)
        for col in self._exog_cols:
            model.add_regressor(col)

        train = pd.DataFrame({"ds": pd.DatetimeIndex(y.index), "y": y.astype(float).to_numpy()})
        for col in self._exog_cols:
            train[col] = X[col].to_numpy(dtype=float)  # type: ignore[index]
        self._model = model.fit(train)

    def predict(
        self,
        horizon: int,
        X: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        ds = self._future_index(self._last_date, horizon)
        future = pd.DataFrame({"ds": ds})
        for col in self._exog_cols:
            if X is None or col not in X.columns:
                raise ModelError(f"prophet: exog column '{col}' missing at predict time")
            future[col] = X[col].to_numpy(dtype=float)[:horizon]

        from scipy.stats import norm  # lazy: keep scipy off the module top (lean launch point)

        fc = self._model.predict(future)
        mean = fc["yhat"].to_numpy(dtype=float)
        # Back out sigma from Prophet's symmetric interval, then place any quantile.
        z = norm.ppf(0.5 + _INTERVAL_WIDTH / 2.0)
        sigma = (fc["yhat_upper"].to_numpy() - fc["yhat_lower"].to_numpy()) / (2.0 * z)

        t, lam = self.ctx.transform, self.ctx.transform_lambda
        qmap = {q: invert_transform(mean + norm.ppf(q) * sigma, t, lam) for q in quantiles}
        return self._assemble_frame(ds, qmap)

    @classmethod
    def search_space(cls, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "changepoint_prior_scale": trial.suggest_float(
                "changepoint_prior_scale", 0.001, 0.5, log=True
            ),
            "seasonality_prior_scale": trial.suggest_float(
                "seasonality_prior_scale", 0.01, 10.0, log=True
            ),
        }


register(ProphetModel)
