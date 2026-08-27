"""Ridge linear regression on lag/calendar features.

One model, one file. Runtime python, ml family. A dependency-free linear counterpart to
the tree models: it shares the recursive multi-step forecasting + design matrix with
LightGBM/XGBoost via ``_lag_forecaster``, but fits a closed-form ridge regression (no
optional dependency). Residual-quantile intervals; the ridge penalty is HPO-tunable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..errors import ModelError
from ..features import invert_transform
from . import _lag_forecaster as lf
from .base_model import DEFAULT_QUANTILES, BaseModel, register

if TYPE_CHECKING:
    import optuna


class _Ridge:
    """Closed-form ridge regression with an unpenalized intercept.

    Exposes ``fit``/``predict`` so it drops straight into ``_lag_forecaster``'s recursive
    step loop, which only relies on ``estimator.predict(row)``.
    """

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self._w: np.ndarray | None = None

    @staticmethod
    def _augment(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return np.hstack([np.ones((x.shape[0], 1)), x])

    def fit(self, x: np.ndarray, y: np.ndarray) -> _Ridge:
        xb = self._augment(x)
        d = xb.shape[1]
        penalty = self.alpha * np.eye(d)
        penalty[0, 0] = 0.0  # never penalize the intercept
        self._w = np.linalg.solve(xb.T @ xb + penalty, xb.T @ np.asarray(y, dtype=float))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._w is None:
            raise ModelError("regression_lags: estimator used before fit")
        return self._augment(x) @ self._w


class RegressionLags(BaseModel):
    """Ridge regression on lag + calendar features with recursive forecasting."""

    name = "regression_lags"
    runtime = "python"
    family = "ml"
    supports_exog = True
    supports_native_intervals = False

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        if len(y) <= max(lf.LAGS):
            raise ModelError(f"regression_lags requires more than {max(lf.LAGS)} observations")
        design, y_aligned, self._features = lf.build_design(y, X)
        self._history = y.astype(float)
        self._last_date = y.index[-1]
        self._model = _Ridge(float(self.params.get("alpha", 1.0)))
        self._model.fit(design.to_numpy(), y_aligned.to_numpy())
        fitted = self._model.predict(design.to_numpy())
        self._set_residuals(y_aligned.to_numpy() - np.asarray(fitted, dtype=float))

    def predict(
        self,
        horizon: int,
        X: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        ds = self._future_index(self._last_date, horizon)
        mean = lf.recursive_predict(self._model, self._history, ds, self._features, X)
        qmap_t = self.residual_intervals(mean, quantiles)
        t, lam = self.ctx.transform, self.ctx.transform_lambda
        qmap = {q: invert_transform(v, t, lam) for q, v in qmap_t.items()}
        return self._assemble_frame(ds, qmap)

    @classmethod
    def search_space(cls, trial: optuna.Trial) -> dict[str, Any]:
        return {"alpha": trial.suggest_float("alpha", 0.01, 100.0, log=True)}


register(RegressionLags)
