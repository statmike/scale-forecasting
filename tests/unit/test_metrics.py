"""Tests for the metric panel.

Each metric is checked against a hand-computed value on a tiny fixed array, plus the
edge cases worth calling out: MAPE with zeros → NaN, MASE/RMSSE need y_train,
coverage in [0,1], pinball ≥ 0.
"""

from __future__ import annotations

import math
from typing import get_args

import numpy as np
import pytest

from scale_forecasting.config import DecisionMetric
from scale_forecasting.metrics import METRIC_NAMES, compute_metrics

# A tiny fixed window with easy-to-verify arithmetic.
#   y_true = [10, 20, 30, 40];  yhat = [12, 18, 33, 36]
#   err    = [ 2, -2,  3, -4];  |err| = [2, 2, 3, 4]
_YT = [10.0, 20.0, 30.0, 40.0]
_YH = [12.0, 18.0, 33.0, 36.0]
_YTRAIN = [2.0, 4.0, 6.0, 8.0, 10.0]  # constant step of 2 → naive mae = 2, rmse = 2


def _m() -> dict[str, float]:
    return compute_metrics(_YT, _YH, y_train=_YTRAIN, lower=[9, 17, 28, 34], upper=[13, 23, 35, 45])


# --- panel completeness --------------------------------------------------------


def test_panel_has_every_metric() -> None:
    m = compute_metrics(_YT, _YH)
    assert set(m) == set(METRIC_NAMES)
    assert all(isinstance(v, float) for v in m.values())


def test_metric_names_match_config_decision_metric() -> None:
    # The panel metrics.py produces must be exactly the DecisionMetric vocabulary in the
    # config — same order, one source of truth (metrics ↔ config ↔ DDL).
    assert METRIC_NAMES == get_args(DecisionMetric)


# --- point-error metrics vs hand-computed --------------------------------------


def test_mae() -> None:
    assert _m()["mae"] == pytest.approx((2 + 2 + 3 + 4) / 4)  # 2.75


def test_mse_and_rmse() -> None:
    mse = (4 + 4 + 9 + 16) / 4  # 8.25
    assert _m()["mse"] == pytest.approx(mse)
    assert _m()["rmse"] == pytest.approx(math.sqrt(mse))


def test_bias_is_mean_signed_error() -> None:
    # err = [2, -2, 3, -4] → mean = -0.25
    assert _m()["bias"] == pytest.approx(-0.25)


def test_mape() -> None:
    # mean(|err|/|y|) = mean(0.2, 0.1, 0.1, 0.1) = 0.125
    assert _m()["mape"] == pytest.approx(0.125)


def test_smape() -> None:
    # 2|e|/(|y|+|yhat|) per term
    terms = [2 * 2 / 22, 2 * 2 / 38, 2 * 3 / 63, 2 * 4 / 76]
    assert _m()["smape"] == pytest.approx(sum(terms) / 4)


def test_wape() -> None:
    # sum|err| / sum|y| = 11 / 100
    assert _m()["wape"] == pytest.approx(11 / 100)


# --- scaled metrics need y_train -----------------------------------------------


def test_mase_uses_naive_scale() -> None:
    # naive one-step mae on y_train (step 2) = 2 → mase = mae / 2 = 2.75 / 2
    assert _m()["mase"] == pytest.approx(2.75 / 2)


def test_rmsse_uses_naive_scale() -> None:
    rmse = math.sqrt(8.25)
    assert _m()["rmsse"] == pytest.approx(rmse / 2)  # naive rmse = 2


def test_mase_rmsse_nan_without_train() -> None:
    m = compute_metrics(_YT, _YH)
    assert math.isnan(m["mase"])
    assert math.isnan(m["rmsse"])


def test_scaled_nan_when_train_flat() -> None:
    m = compute_metrics(_YT, _YH, y_train=[5.0, 5.0, 5.0])
    assert math.isnan(m["mase"])
    assert math.isnan(m["rmsse"])


# --- edge cases ----------------------------------------------------------------


def test_mape_nan_with_zeros() -> None:
    m = compute_metrics([0.0, 10.0], [1.0, 9.0])
    assert math.isnan(m["mape"])
    # but wape/smape stay finite
    assert not math.isnan(m["wape"])
    assert not math.isnan(m["smape"])


def test_wape_nan_when_all_actuals_zero() -> None:
    m = compute_metrics([0.0, 0.0], [1.0, 2.0])
    assert math.isnan(m["wape"])


def test_smape_zero_when_both_zero() -> None:
    m = compute_metrics([0.0, 0.0], [0.0, 0.0])
    assert m["smape"] == pytest.approx(0.0)


def test_perfect_forecast_is_zero_error() -> None:
    m = compute_metrics(_YT, _YT, y_train=_YTRAIN)
    for k in ("mae", "rmse", "mse", "mape", "smape", "wape", "mase", "rmsse", "bias"):
        assert m[k] == pytest.approx(0.0)


# --- intervals: coverage & pinball ---------------------------------------------


def test_coverage_in_unit_interval_and_counts_inside() -> None:
    # bounds chosen so all 4 actuals fall inside → coverage 1.0
    m = _m()
    assert 0.0 <= m["coverage"] <= 1.0
    assert m["coverage"] == pytest.approx(1.0)


def test_coverage_partial() -> None:
    # y_true=[10,20]; put the second actual outside its band
    m = compute_metrics([10.0, 20.0], [10.0, 20.0], lower=[9, 25], upper=[11, 30])
    assert m["coverage"] == pytest.approx(0.5)


def test_pinball_nonnegative() -> None:
    assert _m()["pinball"] >= 0.0


def test_coverage_pinball_nan_without_intervals() -> None:
    m = compute_metrics(_YT, _YH)
    assert math.isnan(m["coverage"])
    assert math.isnan(m["pinball"])


# --- guards --------------------------------------------------------------------


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_metrics([1.0, 2.0], [1.0])


def test_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        compute_metrics([], [])


def test_accepts_numpy_arrays() -> None:
    m = compute_metrics(np.array(_YT), np.array(_YH))
    assert m["mae"] == pytest.approx(2.75)
