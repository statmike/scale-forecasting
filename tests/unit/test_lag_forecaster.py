"""Tests for the shared tree-model lag machinery (_lag_forecaster.py).

The tree models (xgboost, lightgbm) own their lags via the recursion, so config-driven
``features.lags`` — which arrive as ``lag_*`` columns in the exog frame — must not leak in
and overwrite the recursively-computed values. These tests pin that boundary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scale_forecasting.models import _lag_forecaster as lf


def _series(n: int = 60) -> pd.Series:
    idx = pd.date_range("2021-01-01", periods=n, freq="D")
    return pd.Series(np.arange(n, dtype=float) + 10.0, index=idx, name="y")


def test_build_design_ignores_lag_columns_from_exog() -> None:
    y = _series()
    # An exog frame carrying a genuine driver plus a colliding lag_7 (as config would emit).
    exog = pd.DataFrame(
        {"price_index": np.linspace(1.0, 2.0, len(y)), "lag_7": np.zeros(len(y))},
        index=y.index,
    )
    design, _, feature_names = lf.build_design(y, exog)

    # The real driver survives; the injected lag_7 does not overwrite the recursion's own.
    assert "price_index" in feature_names
    # Exactly one lag_7 column, and it holds the true shifted target (not the zeros).
    assert feature_names.count("lag_7") == 1
    expected_lag7 = y.shift(7).loc[design.index]
    np.testing.assert_allclose(design["lag_7"].to_numpy(), expected_lag7.to_numpy())


def test_build_design_keeps_true_exog_only() -> None:
    y = _series()
    exog = pd.DataFrame({"lag_1": np.ones(len(y)), "lag_28": np.ones(len(y))}, index=y.index)
    # All exog columns were lag_* → nothing genuine to add; only the recursion's own lags
    # and calendar remain, each present exactly once.
    _, _, feature_names = lf.build_design(y, exog)
    for lag in lf.LAGS:
        assert feature_names.count(f"lag_{lag}") == 1


def test_recursive_predict_ignores_injected_lag_columns() -> None:
    y = _series()
    design, y_aligned, feature_names = lf.build_design(y, None)

    class _Mean:
        """Trivial estimator: predicts the mean of the training target, ignoring features."""

        def __init__(self, value: float) -> None:
            self.value = value

        def predict(self, x: np.ndarray) -> np.ndarray:
            return np.full(x.shape[0], self.value, dtype=float)

    est = _Mean(float(y_aligned.mean()))
    future = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=5, freq="D")

    baseline = lf.recursive_predict(est, y, future, feature_names, None)
    # Feeding a lag_* laden exog must not change the result (they're filtered out).
    poison = pd.DataFrame({"lag_7": np.full(5, 1e6)}, index=future)
    with_poison = lf.recursive_predict(est, y, future, feature_names, poison)

    np.testing.assert_allclose(baseline, with_poison)
