"""Random Forest regressor on lag/calendar features (scikit-learn).

One model, one file. Runtime python, ml family. Fits an ensemble of bagged
decision trees via ``sklearn.ensemble.RandomForestRegressor`` on the shared
``_lag_forecaster`` design matrix. Because ``scikit-learn`` is in core
dependencies, ``random_forest`` is always available without optional extras.
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


class RandomForestModel(BaseModel):
    """Bagged decision-tree ensemble (RandomForestRegressor) on lag + calendar features."""

    name = "random_forest"
    runtime = "python"
    family = "ml"
    supports_exog = True
    supports_native_intervals = False
    supports_recondition = True
    supports_extrapolate = True
    package = "scikit-learn"
    package_url = "https://scikit-learn.org/"

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        from sklearn.ensemble import RandomForestRegressor

        if len(y) <= max(lf.LAGS):
            raise ModelError(f"random_forest requires more than {max(lf.LAGS)} observations")

        design, y_aligned, self._features = lf.build_design(y, X)
        self._history = y.astype(float)
        self._last_date = y.index[-1]
        max_depth_raw = self.params.get("max_depth", 10)
        max_depth = int(max_depth_raw) if max_depth_raw is not None else None
        self._model = RandomForestRegressor(
            n_estimators=int(self.params.get("n_estimators", 200)),
            max_depth=max_depth,
            min_samples_leaf=int(self.params.get("min_samples_leaf", 2)),
            random_state=self.ctx.seed,
            n_jobs=1,
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
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 4, 16),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        }


register(RandomForestModel)
