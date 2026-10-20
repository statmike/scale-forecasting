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
from ..monotonicity import fit_sign_constrained_ridge, resolve_model_monotonic_vector
from . import _lag_forecaster as lf
from .base_model import DEFAULT_QUANTILES, BaseModel, register

if TYPE_CHECKING:
    import optuna


class _Ridge:
    """Closed-form or sign-constrained ridge regression with an unpenalized intercept.

    Exposes ``fit``/``predict`` so it drops straight into ``_lag_forecaster``'s recursive
    step loop, which only relies on ``estimator.predict(row)``.
    """

    def __init__(self, alpha: float, constraint_vec: list[int] | None = None) -> None:
        self.alpha = alpha
        self.constraint_vec = list(constraint_vec) if constraint_vec is not None else None
        self._w: np.ndarray | None = None

    @staticmethod
    def _augment(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return np.hstack([np.ones((x.shape[0], 1)), x])

    def fit(self, x: np.ndarray, y: np.ndarray) -> _Ridge:
        if self.constraint_vec is not None and any(c != 0 for c in self.constraint_vec):
            self._w = fit_sign_constrained_ridge(
                x, np.asarray(y, dtype=float), self.alpha, self.constraint_vec
            )
            return self
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
    # Same seam as the tree models: the ridge coefficients are the estimate, the lag buffer is not.
    supports_recondition = True
    supports_extrapolate = True

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        lags = lf.resolve_lags(self.params)
        strategy = lf.resolve_strategy(self.params)
        if len(y) <= max(lags):
            raise ModelError(f"regression_lags requires more than {max(lags)} observations")
        design, y_aligned, self._features = lf.build_design(y, X, lags=lags)
        self._lags = lags
        self._strategy = strategy
        self._design = design.to_numpy()
        self._design_mean = self._design.mean(axis=0)
        self._baseline_score = float(y_aligned.mean())
        self._history = y.astype(float)
        self._last_date = y.index[-1]
        alpha = float(self.params.get("alpha", 1.0))
        mono_vec = resolve_model_monotonic_vector(
            self._features,
            self.params,
            self.ctx.monotonic_constraints,
            recursive_lags_nonnegative=(strategy == "recursive"),
        )
        self._monotonic_vector = mono_vec

        def _make_est() -> _Ridge:
            return _Ridge(alpha, constraint_vec=mono_vec)

        self._model = _make_est()
        self._model.fit(self._design, y_aligned.to_numpy())
        fitted = self._model.predict(self._design)
        self._set_residuals(y_aligned.to_numpy() - np.asarray(fitted, dtype=float))
        self._direct_models: list[_Ridge] | None = (
            lf.fit_direct_models(_make_est, self._model, y, X, self.ctx.horizon, lags=lags)
            if strategy == "direct"
            else None
        )

    def predict(
        self,
        horizon: int,
        X: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        # Rolled from the fit's last observation across any skipped span, then tailed — see the
        # note in `xgboost`; the three lag models share this recursion.
        lags = getattr(self, "_lags", lf.LAGS)
        full_index = self._future_index(self._last_date, self._forecast_steps(horizon))
        direct_models = getattr(self, "_direct_models", None)
        if getattr(self, "_strategy", "recursive") == "direct" and direct_models:
            mean = lf.direct_predict(
                direct_models,
                self._history,
                full_index,
                self._features,
                self._forecast_exog(X),
                lags=lags,
            )[-horizon:]
        else:
            mean = lf.recursive_predict(
                self._model,
                self._history,
                full_index,
                self._features,
                self._forecast_exog(X),
                lags=lags,
            )[-horizon:]
        ds = full_index[-horizon:]
        qmap_t = self.residual_intervals(mean, quantiles)
        t, lam = self.ctx.transform, self.ctx.transform_lambda
        qmap = {q: invert_transform(v, t, lam) for q, v in qmap_t.items()}
        return self._assemble_frame(ds, qmap, raw=invert_transform(mean, t, lam))

    def diagnostics(self) -> dict[str, Any]:
        attrs = lf.global_feature_attributions(self._model, self._features, self._design)
        return {"feature_attributions": attrs} if attrs else {}

    def explain(
        self, horizon: int, X: pd.DataFrame | None = None
    ) -> list[dict[str, Any] | None] | None:
        lags = getattr(self, "_lags", lf.LAGS)
        full_index = self._future_index(self._last_date, self._forecast_steps(horizon))
        direct_models = getattr(self, "_direct_models", None)
        if getattr(self, "_strategy", "recursive") == "direct" and direct_models:
            return lf.direct_explain(
                direct_models,
                self._history,
                full_index,
                self._features,
                self._forecast_exog(X),
                design_mean=self._design_mean,
                baseline_score=self._baseline_score,
                lags=lags,
            )[-horizon:]
        return lf.recursive_explain(
            self._model,
            self._history,
            full_index,
            self._features,
            self._forecast_exog(X),
            design_mean=self._design_mean,
            baseline_score=self._baseline_score,
            lags=lags,
        )[-horizon:]

    def recondition(self, y_new: pd.Series, X_new: pd.DataFrame | None = None) -> None:
        self._history = pd.concat([self._history, y_new.astype(float)])
        self._last_date = y_new.index[-1]

    @classmethod
    def search_space(cls, trial: optuna.Trial) -> dict[str, Any]:
        return {"alpha": trial.suggest_float("alpha", 0.01, 100.0, log=True)}


register(RegressionLags)
