"""XGBoost regressor on lag/calendar features.

One model, one file (CONTRACTS §1). Runtime python, ml family. XGBoost is an optional
dependency, so it is imported lazily inside ``fit`` — the model still *registers* without
it (the factory stays complete), and only fails if actually used without the extra
installed. Recursive multi-step forecasting + design matrix come from the shared
``_lag_forecaster`` helper; intervals are residual-quantile based.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..errors import ModelError
from ..features import invert_transform
from . import _lag_forecaster as lf
from .base_model import DEFAULT_QUANTILES, BaseModel, register


class XgboostModel(BaseModel):
    """Gradient-boosted trees on lag + calendar features."""

    name = "xgboost"
    runtime = "python"
    family = "ml"
    supports_exog = True
    supports_native_intervals = False

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        try:
            from xgboost import XGBRegressor
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise ModelError("xgboost not installed; install the 'models' extra") from e
        if len(y) <= max(lf.LAGS):
            raise ModelError(f"xgboost requires more than {max(lf.LAGS)} observations")

        design, y_aligned, self._features = lf.build_design(y, X)
        self._history = y.astype(float)
        self._last_date = y.index[-1]
        self._model = XGBRegressor(
            n_estimators=int(self.params.get("n_estimators", 300)),
            max_depth=int(self.params.get("max_depth", 6)),
            learning_rate=float(self.params.get("learning_rate", 0.05)),
            subsample=0.9,
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
        ds = self._future_index(self._last_date, horizon)
        mean = lf.recursive_predict(self._model, self._history, ds, self._features, X)
        qmap_t = self.residual_intervals(mean, quantiles)
        t = self.ctx.transform
        qmap = {q: invert_transform(v, t) for q, v in qmap_t.items()}
        return self._assemble_frame(ds, qmap)

    @classmethod
    def search_space(cls, trial: Any) -> dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        }


register(XgboostModel)
