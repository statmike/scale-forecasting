"""LightGBM regressor on lag/calendar features.

One model, one file (CONTRACTS §1). Runtime python, ml family. LightGBM is an optional
dependency, imported lazily in ``fit`` so the model registers without it. Shares the
recursive multi-step forecasting + design matrix with XGBoost via ``_lag_forecaster``;
residual-quantile intervals.
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


class LightgbmModel(BaseModel):
    """Gradient-boosted trees (LightGBM) on lag + calendar features."""

    name = "lightgbm"
    runtime = "python"
    family = "ml"
    supports_exog = True
    supports_native_intervals = False

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        try:
            from lightgbm import LGBMRegressor
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise ModelError("lightgbm not installed; install the 'models' extra") from e
        if len(y) <= max(lf.LAGS):
            raise ModelError(f"lightgbm requires more than {max(lf.LAGS)} observations")

        design, y_aligned, self._features = lf.build_design(y, X)
        self._history = y.astype(float)
        self._last_date = y.index[-1]
        self._model = LGBMRegressor(
            n_estimators=int(self.params.get("n_estimators", 300)),
            max_depth=int(self.params.get("max_depth", -1)),
            learning_rate=float(self.params.get("learning_rate", 0.05)),
            subsample=0.9,
            random_state=self.ctx.seed,
            n_jobs=1,
            verbose=-1,
        )
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
        t = self.ctx.transform
        qmap = {q: invert_transform(v, t) for q, v in qmap_t.items()}
        return self._assemble_frame(ds, qmap)

    @classmethod
    def search_space(cls, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        }


register(LightgbmModel)
