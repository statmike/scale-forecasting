"""CatBoost regressor on lag/calendar features.

One model, one file. Runtime python, ml family. CatBoost is an optional
dependency (``scale-forecasting[models-trees]`` or ``scale-forecasting[models]``),
imported lazily in ``fit`` so the model registers without it. Shares recursive
multi-step forecasting and the lag/calendar design matrix with XGBoost and
LightGBM via ``_lag_forecaster``; uses empirical residual-quantile intervals.
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


class CatboostModel(BaseModel):
    """Symmetric (oblivious) gradient-boosted trees (CatBoost) on lag + calendar features."""

    name = "catboost"
    runtime = "python"
    family = "ml"
    supports_exog = True
    supports_native_intervals = False
    supports_recondition = True
    supports_extrapolate = True
    package = "catboost"
    package_url = "https://catboost.ai/"
    optional_import = "catboost"
    optional_extra = "models-trees"

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        try:
            from catboost import CatBoostRegressor
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            self.require_available()
            raise ModelError("catboost not installed; install the 'models' extra") from e
        if len(y) <= max(lf.LAGS):
            raise ModelError(f"catboost requires more than {max(lf.LAGS)} observations")

        design, y_aligned, self._features = lf.build_design(y, X)
        self._history = y.astype(float)
        self._last_date = y.index[-1]
        self._model = CatBoostRegressor(
            iterations=int(self.params.get("iterations", self.params.get("n_estimators", 300))),
            depth=int(self.params.get("depth", 6)),
            learning_rate=float(self.params.get("learning_rate", 0.05)),
            random_seed=self.ctx.seed,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
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
        full_index = self._future_index(self._last_date, self._forecast_steps(horizon))
        mean = lf.recursive_predict(
            self._model, self._history, full_index, self._features, self._forecast_exog(X)
        )[-horizon:]
        ds = full_index[-horizon:]
        qmap_t = self.residual_intervals(mean, quantiles)
        t, lam = self.ctx.transform, self.ctx.transform_lambda
        qmap = {q: invert_transform(v, t, lam) for q, v in qmap_t.items()}
        return self._assemble_frame(ds, qmap, raw=invert_transform(mean, t, lam))

    def recondition(self, y_new: pd.Series, X_new: pd.DataFrame | None = None) -> None:
        self._history = pd.concat([self._history, y_new.astype(float)])
        self._last_date = y_new.index[-1]

    @classmethod
    def search_space(cls, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "iterations": trial.suggest_int("iterations", 100, 600),
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        }


register(CatboostModel)
