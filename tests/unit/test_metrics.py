"""Tests for the metric panel.

Each metric is checked against a hand-computed value on a tiny fixed array, plus the
edge cases worth calling out: MAPE with zeros → NaN, MASE/RMSSE need y_train,
coverage in [0,1], pinball ≥ 0.

The NaN cases are as much of the contract as the arithmetic. A metric that cannot be computed
for a cell returns NaN rather than raising, because a 100k-series batch must not die on one
pathological series — but that only holds if NaN means "undefined here", never "computed from
a substituted default". So each new metric is pinned both ways: the value when it is defined,
and NaN in every case where the inputs don't support it.
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
    for k in ("mae", "rmse", "mse", "mape", "smape", "wape", "mase", "rmsse", "bias", "maape"):
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


# --- mase_seasonal: MASE against the seasonal naive ----------------------------


def test_mase_seasonal_scales_by_the_seasonal_naive() -> None:
    # _YTRAIN steps by 2, so the m=2 naive error is a constant 4 (two steps' worth).
    m = compute_metrics(_YT, _YH, y_train=_YTRAIN, seasonal_period=2)
    assert m["mase_seasonal"] == pytest.approx(2.75 / 4.0)


def test_mase_seasonal_at_period_one_is_plain_mase() -> None:
    """The seasonal naive at m=1 *is* the one-step naive — the two must agree there, or one of
    the two denominators is computed differently from the other."""
    m = compute_metrics(_YT, _YH, y_train=_YTRAIN, seasonal_period=1)
    assert m["mase_seasonal"] == pytest.approx(m["mase"])


def test_mase_seasonal_nan_without_a_period() -> None:
    # The default: callers that don't know the run frequency get NaN, not a wrong-cycle score.
    assert math.isnan(compute_metrics(_YT, _YH, y_train=_YTRAIN)["mase_seasonal"])


def test_mase_seasonal_nan_when_history_is_shorter_than_a_cycle() -> None:
    # 5 points of history cannot compare y_t to y_{t-7}; the alternative to NaN is a fabricated
    # denominator from whatever partial cycle happens to exist.
    m = compute_metrics(_YT, _YH, y_train=_YTRAIN, seasonal_period=7)
    assert math.isnan(m["mase_seasonal"])


def test_mase_seasonal_nan_when_the_seasonal_naive_is_perfect() -> None:
    # A perfectly periodic history makes the seasonal naive error 0 — nothing to scale by.
    m = compute_metrics(_YT, _YH, y_train=[1.0, 5.0, 1.0, 5.0, 1.0, 5.0], seasonal_period=2)
    assert math.isnan(m["mase_seasonal"])


# --- maape: the percentage metric that survives zeros --------------------------


def test_maape_is_mean_arctan_of_the_ratio() -> None:
    # ratios = [2/10, 2/20, 3/30, 4/40] = [0.2, 0.1, 0.1, 0.1]
    expected = (math.atan(0.2) + 3 * math.atan(0.1)) / 4
    assert compute_metrics(_YT, _YH)["maape"] == pytest.approx(expected)


def test_maape_is_a_real_number_where_mape_is_nan() -> None:
    """The whole reason it exists: one zero actual NaNs MAPE for the entire window, so an
    intermittent-demand series has no percentage metric to rank on at all."""
    m = compute_metrics([0.0, 20.0], [5.0, 18.0])
    assert math.isnan(m["mape"])
    assert m["maape"] == pytest.approx((math.pi / 2 + math.atan(0.1)) / 2)


def test_maape_treats_a_zero_forecast_of_a_zero_actual_as_perfect() -> None:
    # 0/0 is a match, not an infinite error — otherwise a correctly-forecast idle series scores
    # the same as one that missed by everything.
    assert compute_metrics([0.0, 0.0], [0.0, 0.0])["maape"] == pytest.approx(0.0)


def test_maape_is_bounded_by_half_pi() -> None:
    assert compute_metrics([0.0], [1e12])["maape"] == pytest.approx(math.pi / 2)


# --- interval_score / interval_width -------------------------------------------


def test_interval_width_is_the_mean_band_width() -> None:
    # widths = [13-9, 23-17, 35-28, 45-34] = [4, 6, 7, 11]
    assert _m()["interval_width"] == pytest.approx(28.0 / 4)


def test_interval_score_is_just_the_width_when_nothing_misses() -> None:
    m = _m()
    assert m["coverage"] == pytest.approx(1.0)
    assert m["interval_score"] == pytest.approx(m["interval_width"])


def test_interval_score_penalises_a_miss_in_proportion_to_the_distance() -> None:
    # Second point: y=20 sits 5 below a lower bound of 25 → width 5 + (2/0.2)·5 = 55.
    m = compute_metrics([10.0, 20.0], [10.0, 20.0], lower=[9, 25], upper=[11, 30])
    assert m["interval_score"] == pytest.approx((2.0 + 55.0) / 2)
    assert m["interval_width"] == pytest.approx((2.0 + 5.0) / 2)


def test_a_wider_band_scores_worse_once_it_already_covers() -> None:
    """Coverage alone is gamed by widening; the interval score is what stops that."""
    tight = compute_metrics(_YT, _YT, lower=[9, 19, 29, 39], upper=[11, 21, 31, 41])
    loose = compute_metrics(_YT, _YT, lower=[0, 0, 0, 0], upper=[99, 99, 99, 99])
    assert tight["coverage"] == loose["coverage"] == pytest.approx(1.0)
    assert tight["interval_score"] < loose["interval_score"]


def test_interval_metrics_nan_without_intervals() -> None:
    m = compute_metrics(_YT, _YH)
    assert math.isnan(m["interval_score"])
    assert math.isnan(m["interval_width"])


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
