"""Shared recursive lag-forecasting for tree models (internal helper, not a model).

XGBoost and LightGBM are point regressors: to forecast a horizon they need engineered
lag/calendar features and must roll forward one step at a time, feeding each prediction
back in as the next lag. That machinery is identical for both, so it lives here (a helper
module, like ``base_model``) and each model file stays thin — honoring one-model-one-file
while not duplicating the loop.

Not a public API: the leading underscore marks it internal to ``models``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Lag depths and calendar features used by the tree models. Fixed here (not config-driven)
# because the recursion depends on knowing them; HPO tunes the estimator, not the lags.
LAGS: tuple[int, ...] = (1, 2, 3, 7, 14, 28)


def _calendar(index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Deterministic calendar features from a datetime index."""
    return {
        "dow": index.dayofweek.to_numpy(dtype=float),
        "dom": index.day.to_numpy(dtype=float),
        "month": index.month.to_numpy(dtype=float),
        "doy": index.dayofyear.to_numpy(dtype=float),
    }


def build_design(
    y: pd.Series, exog: pd.DataFrame | None
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Build the training design matrix from lags + calendar (+ exog).

    Returns ``(X, y_aligned, feature_names)`` with the first ``max(LAGS)`` rows dropped
    (their lags are undefined).
    """
    idx = pd.DatetimeIndex(y.index)
    cols: dict[str, np.ndarray] = {f"lag_{lag}": y.shift(lag).to_numpy() for lag in LAGS}
    cols.update(_calendar(idx))
    if exog is not None:
        for c in exog.columns:
            cols[c] = exog[c].to_numpy(dtype=float)
    design = pd.DataFrame(cols, index=idx)
    valid = design.dropna()
    feature_names = list(design.columns)
    return valid, y.loc[valid.index], feature_names


def recursive_predict(
    estimator: object,
    history: pd.Series,
    future_index: pd.DatetimeIndex,
    feature_names: list[str],
    future_exog: pd.DataFrame | None,
) -> np.ndarray:
    """Roll the fitted ``estimator`` forward over ``future_index`` one step at a time.

    Each step assembles the same feature row (lags from the growing history + calendar +
    exog), predicts, and appends the prediction to the history for the next step.
    """
    series = history.copy()
    preds: list[float] = []
    for i, ts in enumerate(future_index):
        row: dict[str, float] = {f"lag_{lag}": float(series.iloc[-lag]) for lag in LAGS}
        cal = _calendar(pd.DatetimeIndex([ts]))
        row.update({k: float(v[0]) for k, v in cal.items()})
        if future_exog is not None:
            for c in future_exog.columns:
                row[c] = float(future_exog.iloc[i][c])
        x = np.array([[row[name] for name in feature_names]], dtype=float)
        raw = estimator.predict(x)  # type: ignore[attr-defined]
        yhat = float(np.asarray(raw, dtype=float).ravel()[0])
        preds.append(yhat)
        series = pd.concat([series, pd.Series([yhat], index=[ts])])
    return np.asarray(preds, dtype=float)
