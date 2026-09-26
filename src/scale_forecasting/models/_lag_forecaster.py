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

from ..errors import ModelError

# Lag depths and calendar features used by the tree models. Fixed here (not config-driven)
# because the recursion depends on knowing them; HPO tunes the estimator, not the lags.
LAGS: tuple[int, ...] = (1, 2, 3, 7, 14, 28)

# Every column name `build_design` produces on its own. An exog column that lands on one of
# these would replace it rather than join it — see `_check_not_reserved`.
_RESERVED: frozenset[str] = frozenset(
    [f"lag_{lag}" for lag in LAGS] + ["dow", "dom", "month", "doy"]
)


def _check_not_reserved(exog: pd.DataFrame | None) -> pd.DataFrame | None:
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
    if clash := sorted(set(exog.columns) & _RESERVED):
        raise ModelError(
            f"exog column(s) {clash} collide with features this model builds itself "
            f"({sorted(_RESERVED)}); rename them in the source table"
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
    y: pd.Series, exog: pd.DataFrame | None
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Build the training design matrix from lags + calendar (+ exog).

    Returns ``(X, y_aligned, feature_names)`` with the first ``max(LAGS)`` rows dropped
    (their lags are undefined).
    """
    idx = pd.DatetimeIndex(y.index)
    cols: dict[str, np.ndarray] = {f"lag_{lag}": y.shift(lag).to_numpy() for lag in LAGS}
    cols.update(_calendar(idx))
    exog = _check_not_reserved(exog)  # a covariate must not overwrite a feature we build
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
    future_exog = _check_not_reserved(future_exog)  # match build_design: same reserved names
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
