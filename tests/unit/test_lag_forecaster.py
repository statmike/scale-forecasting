"""Tests for the shared tree-model lag machinery (_lag_forecaster.py).

The tree models (xgboost, lightgbm) own their target lags via the recursion, and they build
their own calendar terms. `build_design` merges exog into the same flat column dict as both,
so a covariate sharing one of those names would *replace* it rather than sit beside it, and
the recursion would then feed a stale value into a coefficient fitted on the real one. These
tests pin that boundary.

Lagged *covariates* are a different thing and are meant to get through: `features.exog_lags`
names them ``<column>_lag_<k>``, which collides with nothing, so they reach the tree models
as ordinary regressors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scale_forecasting.errors import ModelError
from scale_forecasting.models import _lag_forecaster as lf


def _series(n: int = 60) -> pd.Series:
    idx = pd.date_range("2021-01-01", periods=n, freq="D")
    return pd.Series(np.arange(n, dtype=float) + 10.0, index=idx, name="y")


@pytest.mark.parametrize("clashing", ["lag_7", "month", "dow"])
def test_build_design_refuses_an_exog_column_that_would_overwrite_its_own(clashing: str) -> None:
    """A collision raises rather than being dropped, because nothing downstream looks wrong.

    ``month``/``dow`` are here deliberately: the predecessor guard only screened ``lag_*``, so
    a source column called ``month`` silently replaced the calendar term of the same name.
    """
    y = _series()
    exog = pd.DataFrame(
        {"price_index": np.linspace(1.0, 2.0, len(y)), clashing: np.zeros(len(y))},
        index=y.index,
    )
    with pytest.raises(ModelError, match="collide with features this model builds itself"):
        lf.build_design(y, exog)


def test_build_design_passes_a_lagged_covariate_straight_through() -> None:
    """``features.exog_lags`` output is a covariate, not a target lag, so it is not withheld.

    The tree models' recursion owns lags of ``y``. It has nothing to say about lags of someone
    else's column, so those reach the design matrix intact and unduplicated.
    """
    y = _series()
    values = np.linspace(1.0, 2.0, len(y))
    exog = pd.DataFrame({"promo": values, "promo_lag_1": values - 0.5}, index=y.index)
    design, _, feature_names = lf.build_design(y, exog)

    assert "promo_lag_1" in feature_names
    assert feature_names.count("promo_lag_1") == 1
    np.testing.assert_allclose(design["promo_lag_1"].to_numpy(), (values - 0.5)[-len(design) :])
    # The recursion's own lags are untouched and still hold the shifted target.
    for lag in lf.LAGS:
        assert feature_names.count(f"lag_{lag}") == 1
    np.testing.assert_allclose(design["lag_7"].to_numpy(), y.shift(7).loc[design.index].to_numpy())


def test_recursive_predict_refuses_the_same_collisions_build_design_does() -> None:
    y = _series()
    _design, y_aligned, feature_names = lf.build_design(y, None)

    class _Mean:
        """Trivial estimator: predicts the mean of the training target, ignoring features."""

        def __init__(self, value: float) -> None:
            self.value = value

        def predict(self, x: np.ndarray) -> np.ndarray:
            return np.full(x.shape[0], self.value, dtype=float)

    est = _Mean(float(y_aligned.mean()))
    future = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=5, freq="D")

    # A horizon frame must clear the same bar the training frame did, or the recursion would
    # read a poisoned lag_7 at predict time having fitted against the real one.
    lf.recursive_predict(est, y, future, feature_names, None)
    poison = pd.DataFrame({"lag_7": np.full(5, 1e6)}, index=future)
    with pytest.raises(ModelError, match="collide with features this model builds itself"):
        lf.recursive_predict(est, y, future, feature_names, poison)
