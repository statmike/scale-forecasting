"""Shared recursive lag-forecasting for tree models (internal helper, not a model).

XGBoost and LightGBM are point regressors: to forecast a horizon they need engineered
lag/calendar features and must roll forward one step at a time, feeding each prediction
back in as the next lag. That machinery is identical for both, so it lives here (a helper
module, like ``base_model``) and each model file stays thin — honoring one-model-one-file
while not duplicating the loop.

Not a public API: the leading underscore marks it internal to ``models``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

import numpy as np
import pandas as pd

from ..errors import ModelError


class SupportsPredict(Protocol):
    """The one method the recursive forecaster needs of a fitted regressor — the sklearn /
    XGBoost / LightGBM / CatBoost estimators all have it, none of them share a typed base."""

    def predict(self, X: Any) -> Any: ...


# Default lag depths and calendar features used by the tree models when `lags` is not overridden.
LAGS: tuple[int, ...] = (1, 2, 3, 7, 14, 28)

_CALENDAR_COLS: tuple[str, ...] = ("dow", "dom", "month", "doy")

# Every column name `build_design` produces on its own under default lags.
_RESERVED: frozenset[str] = frozenset([f"lag_{lag}" for lag in LAGS] + list(_CALENDAR_COLS))


def resolve_lags(params: Mapping[str, Any] | None = None) -> tuple[int, ...]:
    """Return sorted positive integer lags from ``params['lags']``, or default `LAGS`."""
    if not params or params.get("lags") is None:
        return LAGS
    raw = params["lags"]
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ModelError(f"lags must be a sequence of positive integers; got {raw!r}")
    parsed = sorted({int(x) for x in raw})
    if not parsed or any(x <= 0 for x in parsed):
        raise ModelError(f"lags must be a non-empty sequence of positive integers; got {raw!r}")
    return tuple(parsed)


def resolve_strategy(params: Mapping[str, Any] | None = None) -> str:
    """Return ``'recursive'`` (default) or ``'direct'`` from ``params['strategy']``."""
    if not params or params.get("strategy") is None:
        return "recursive"
    strategy = str(params["strategy"]).strip().lower()
    if strategy not in ("recursive", "direct"):
        raise ModelError(
            f"unknown lag forecasting strategy {strategy!r}; expected 'recursive' or 'direct'"
        )
    return strategy


def _check_not_reserved(
    exog: pd.DataFrame | None, *, lags: Sequence[int] = LAGS
) -> pd.DataFrame | None:
    """Refuse an exog frame that would overwrite a feature this model owns.

    `build_design` merges exog into the same flat column dict as the target lags and the
    calendar terms, so a covariate sharing one of those names does not sit alongside it — it
    replaces it, and the recursion below then feeds a stale value into a coefficient fitted on
    the real one. Nothing about the output looks wrong when that happens, which is why this
    raises instead of dropping.

    The predecessor to this check silently dropped ``lag_*`` columns, because the config used
    to be able to build target lags and those genuinely had to be kept out. That field is gone
    — target lags are model-owned (see ``FeaturesConfig``) — so the only way to land on one of
    these names now is to have a source column called ``lag_7`` or ``month``, which is a
    collision worth a message rather than a silent drop. Lagged *covariates* are named
    ``<column>_lag_<k>`` and never collide, which is why they pass straight through to the
    tree models like any other regressor.
    """
    if exog is None:
        return None
    reserved = frozenset([f"lag_{lag}" for lag in lags] + list(_CALENDAR_COLS))
    if clash := sorted(set(exog.columns) & reserved):
        raise ModelError(
            f"exog column(s) {clash} collide with features this model builds itself "
            f"({sorted(reserved)}); rename them in the source table"
        )
    return exog


def _calendar(index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Deterministic calendar features from a datetime index."""
    return {
        "dow": index.dayofweek.to_numpy(dtype=float),
        "dom": index.day.to_numpy(dtype=float),
        "month": index.month.to_numpy(dtype=float),
        "doy": index.dayofyear.to_numpy(dtype=float),
    }


def build_design(
    y: pd.Series,
    exog: pd.DataFrame | None,
    *,
    lags: Sequence[int] = LAGS,
    step_offset: int = 0,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Build the training design matrix from lags + calendar (+ exog).

    Returns ``(X, y_aligned, feature_names)`` with the first ``max(lags) + step_offset`` rows
    dropped (their lags are undefined). When ``step_offset > 0`` (direct multi-step training for
    horizon step ``step_offset + 1``), ``lag_k`` is taken at shift ``k + step_offset`` so the
    feature vector matches the forecast-origin lag buffer ``history.iloc[-k]``.
    """
    idx = pd.DatetimeIndex(y.index)
    cols: dict[str, np.ndarray] = {
        f"lag_{lag}": y.shift(lag + step_offset).to_numpy() for lag in lags
    }
    cols.update(_calendar(idx))
    exog = _check_not_reserved(exog, lags=lags)  # a covariate must not overwrite a feature we build
    if exog is not None:
        for c in exog.columns:
            cols[c] = exog[c].to_numpy(dtype=float)
    design = pd.DataFrame(cols, index=idx)
    valid = design.dropna()
    feature_names = list(design.columns)
    return valid, y.loc[valid.index], feature_names


def fit_direct_models(
    make_estimator: Callable[[], Any],
    base_model: Any,
    y: pd.Series,
    exog: pd.DataFrame | None,
    horizon: int,
    *,
    lags: Sequence[int] = LAGS,
) -> list[Any]:
    """Fit one direct estimator per horizon step ``h in 0..horizon-1``
    (step 0 reuses ``base_model``).
    """
    models: list[Any] = [base_model]
    n_steps = max(1, int(horizon))
    for step_idx in range(1, n_steps):
        design_h, y_h, _ = build_design(y, exog, lags=lags, step_offset=step_idx)
        if len(design_h) < 5:
            models.append(models[-1])
            continue
        est_h = make_estimator()
        est_h.fit(design_h.to_numpy(), y_h.to_numpy())
        models.append(est_h)
    return models


def recursive_predict(
    estimator: SupportsPredict,
    history: pd.Series,
    future_index: pd.DatetimeIndex,
    feature_names: list[str],
    future_exog: pd.DataFrame | None,
    *,
    lags: Sequence[int] = LAGS,
) -> np.ndarray:
    """Roll the fitted ``estimator`` forward over ``future_index`` one step at a time.

    Each step assembles the same feature row (lags from the growing history + calendar +
    exog), predicts, and appends the prediction to the history for the next step.
    """
    series = history.copy()
    future_exog = _check_not_reserved(future_exog, lags=lags)
    preds: list[float] = []
    for i, ts in enumerate(future_index):
        row: dict[str, float] = {f"lag_{lag}": float(series.iloc[-lag]) for lag in lags}
        cal = _calendar(pd.DatetimeIndex([ts]))
        row.update({k: float(v[0]) for k, v in cal.items()})
        if future_exog is not None:
            for c in future_exog.columns:
                row[c] = float(future_exog.iloc[i][c])
        x = np.array([[row[name] for name in feature_names]], dtype=float)
        raw = estimator.predict(x)
        yhat = float(np.asarray(raw, dtype=float).ravel()[0])
        preds.append(yhat)
        series = pd.concat([series, pd.Series([yhat], index=[ts])])
    return np.asarray(preds, dtype=float)


def direct_predict(
    models: Sequence[SupportsPredict],
    history: pd.Series,
    future_index: pd.DatetimeIndex,
    feature_names: list[str],
    future_exog: pd.DataFrame | None,
    *,
    lags: Sequence[int] = LAGS,
) -> np.ndarray:
    """Predict each step ``i`` over ``future_index`` directly from the fixed
    forecast-origin lags.
    """
    future_exog = _check_not_reserved(future_exog, lags=lags)
    origin_lags: dict[str, float] = {f"lag_{lag}": float(history.iloc[-lag]) for lag in lags}
    preds: list[float] = []
    for i, ts in enumerate(future_index):
        est = models[min(i, len(models) - 1)]
        row: dict[str, float] = dict(origin_lags)
        cal = _calendar(pd.DatetimeIndex([ts]))
        row.update({k: float(v[0]) for k, v in cal.items()})
        if future_exog is not None:
            for c in future_exog.columns:
                row[c] = float(future_exog.iloc[i][c])
        x = np.array([[row[name] for name in feature_names]], dtype=float)
        raw = est.predict(x)
        yhat = float(np.asarray(raw, dtype=float).ravel()[0])
        preds.append(yhat)
    return np.asarray(preds, dtype=float)


def global_feature_attributions(
    estimator: Any,
    feature_names: list[str],
    design: pd.DataFrame | np.ndarray | None = None,
) -> dict[str, float]:
    """Compute normalized global feature importance (`{feature_name: float}`) for Tier 1.

    Supports tree estimators (`feature_importances_`) and linear ridge estimators (`_w` scaled by
    feature standard deviation). Returns JSON-safe finite floats normalized to sum to 1.0 when
    total importance is positive.
    """
    if not feature_names:
        return {}
    raw_imp: np.ndarray | None = None
    if hasattr(estimator, "feature_importances_"):
        raw_imp = np.abs(np.asarray(estimator.feature_importances_, dtype=float).ravel())
    elif hasattr(estimator, "_w") and getattr(estimator, "_w", None) is not None:
        w = np.asarray(estimator._w, dtype=float).ravel()
        coef = (
            np.abs(w[1:]) if len(w) == len(feature_names) + 1 else np.abs(w[: len(feature_names)])
        )
        if design is not None:
            x_arr = np.asarray(design, dtype=float)
            std = np.nanstd(x_arr, axis=0)
            std = np.where(np.isfinite(std) & (std > 0.0), std, 1.0)
            raw_imp = coef * std
        else:
            raw_imp = coef
    if raw_imp is None or len(raw_imp) != len(feature_names):
        return {}
    raw_imp = np.where(np.isfinite(raw_imp), raw_imp, 0.0)
    total = float(raw_imp.sum())
    if total > 0.0:
        raw_imp = raw_imp / total
    return {name: round(float(val), 6) for name, val in zip(feature_names, raw_imp, strict=True)}


def _step_shap_or_linear_contribs(
    estimator: Any,
    x: np.ndarray,
    feature_names: list[str],
    design_mean: np.ndarray | None = None,
    baseline_score: float = 0.0,
) -> dict[str, Any] | None:
    """Compute one step's `{"baseline_score": float, "attributions": {feature: float}}`."""
    d = len(feature_names)
    contribs: np.ndarray | None = None
    base_val: float | None = None

    try:
        cls_name = type(estimator).__name__
        if cls_name == "XGBRegressor" and hasattr(estimator, "get_booster"):
            import xgboost as xgb

            mat = xgb.DMatrix(x)
            raw_c = np.asarray(
                estimator.get_booster().predict(mat, pred_contribs=True), dtype=float
            ).ravel()
            if len(raw_c) == d + 1:
                contribs, base_val = raw_c[:d], float(raw_c[d])
        elif cls_name == "LGBMRegressor":
            raw_c = np.asarray(estimator.predict(x, pred_contrib=True), dtype=float).ravel()
            if len(raw_c) == d + 1:
                contribs, base_val = raw_c[:d], float(raw_c[d])
        elif cls_name == "CatBoostRegressor" and hasattr(estimator, "get_feature_importance"):
            from catboost import Pool

            raw_c = np.asarray(
                estimator.get_feature_importance(Pool(x), type="ShapValues"), dtype=float
            ).ravel()
            if len(raw_c) == d + 1:
                contribs, base_val = raw_c[:d], float(raw_c[d])
        elif hasattr(estimator, "_w") and getattr(estimator, "_w", None) is not None:
            w = np.asarray(estimator._w, dtype=float).ravel()
            if len(w) == d + 1:
                intercept = float(w[0])
                coef = w[1:]
                mean_x = (
                    np.asarray(design_mean, dtype=float).ravel()
                    if design_mean is not None and len(design_mean) == d
                    else np.zeros(d, dtype=float)
                )
                base_val = float(intercept + np.dot(mean_x, coef))
                contribs = coef * (x.ravel() - mean_x)
        elif hasattr(estimator, "feature_importances_"):
            yhat = float(np.asarray(estimator.predict(x), dtype=float).ravel()[0])
            imp = np.abs(np.asarray(estimator.feature_importances_, dtype=float).ravel())
            mean_x = (
                np.asarray(design_mean, dtype=float).ravel()
                if design_mean is not None and len(design_mean) == d
                else np.zeros(d, dtype=float)
            )
            diff = np.abs(x.ravel() - mean_x)
            weights = imp * (diff + 1e-9)
            w_sum = float(weights.sum())
            delta = yhat - float(baseline_score)
            contribs = (
                delta * (weights / w_sum)
                if w_sum > 0.0
                else np.full(d, delta / max(1, d), dtype=float)
            )
            base_val = float(baseline_score)
    except Exception:  # noqa: BLE001 - explainability is best-effort
        return None

    if contribs is None or base_val is None or not math.isfinite(base_val):
        return None
    attr_map = {
        name: round(float(val), 6)
        for name, val in zip(feature_names, contribs, strict=True)
        if math.isfinite(float(val))
    }
    return {"baseline_score": round(float(base_val), 6), "attributions": attr_map}


def recursive_explain(
    estimator: SupportsPredict,
    history: pd.Series,
    future_index: pd.DatetimeIndex,
    feature_names: list[str],
    future_exog: pd.DataFrame | None,
    *,
    design_mean: np.ndarray | None = None,
    baseline_score: float = 0.0,
    lags: Sequence[int] = LAGS,
) -> list[dict[str, Any] | None]:
    """Roll ``estimator`` forward over ``future_index`` and return per-step local attributions."""
    series = history.copy()
    future_exog = _check_not_reserved(future_exog, lags=lags)
    explanations: list[dict[str, Any] | None] = []
    for i, ts in enumerate(future_index):
        row: dict[str, float] = {f"lag_{lag}": float(series.iloc[-lag]) for lag in lags}
        cal = _calendar(pd.DatetimeIndex([ts]))
        row.update({k: float(v[0]) for k, v in cal.items()})
        if future_exog is not None:
            for c in future_exog.columns:
                row[c] = float(future_exog.iloc[i][c])
        x = np.array([[row[name] for name in feature_names]], dtype=float)
        raw = estimator.predict(x)
        yhat = float(np.asarray(raw, dtype=float).ravel()[0])
        explanations.append(
            _step_shap_or_linear_contribs(
                estimator,
                x,
                feature_names,
                design_mean=design_mean,
                baseline_score=baseline_score,
            )
        )
        series = pd.concat([series, pd.Series([yhat], index=[ts])])
    return explanations


def direct_explain(
    models: Sequence[SupportsPredict],
    history: pd.Series,
    future_index: pd.DatetimeIndex,
    feature_names: list[str],
    future_exog: pd.DataFrame | None,
    *,
    design_mean: np.ndarray | None = None,
    baseline_score: float = 0.0,
    lags: Sequence[int] = LAGS,
) -> list[dict[str, Any] | None]:
    """Compute per-step local attributions across ``future_index`` using direct step estimators."""
    future_exog = _check_not_reserved(future_exog, lags=lags)
    origin_lags: dict[str, float] = {f"lag_{lag}": float(history.iloc[-lag]) for lag in lags}
    explanations: list[dict[str, Any] | None] = []
    for i, ts in enumerate(future_index):
        est = models[min(i, len(models) - 1)]
        row: dict[str, float] = dict(origin_lags)
        cal = _calendar(pd.DatetimeIndex([ts]))
        row.update({k: float(v[0]) for k, v in cal.items()})
        if future_exog is not None:
            for c in future_exog.columns:
                row[c] = float(future_exog.iloc[i][c])
        x = np.array([[row[name] for name in feature_names]], dtype=float)
        explanations.append(
            _step_shap_or_linear_contribs(
                est,
                x,
                feature_names,
                design_mean=design_mean,
                baseline_score=baseline_score,
            )
        )
    return explanations
