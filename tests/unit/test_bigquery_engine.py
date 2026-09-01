"""Offline tests for the BigQuery-native engine's pure surface (`engines.bigquery_engine`).

`run` itself is live-only (it is one long BigQuery conversation, proven by the ``@gcp`` smoke), but
the decisions inside it are not. `_score_fold` is the one that matters most: it turns a fold's eval
frame into the ``backtest_oof`` rows and the metric panel a native model is *ranked on*. A mistake
there — an unsorted horizon, the wrong series' history as the scale denominator, dropped interval
bounds — does not fail a run. It just publishes a different number, which is precisely the kind of
bug a live smoke cannot see. So it is asserted here, with no cloud.

The row builders `_meta_row` / `_oof_row` are checked against the column specs they feed in
``test_registry_assembly.py``, alongside the other two producers of the same tables.
"""

from __future__ import annotations

import math

import pandas as pd

from scale_forecasting.engines.bigquery_engine import _score_fold


def _eval_df() -> pd.DataFrame:
    """Two series x three horizon steps, deliberately shuffled out of forecast_date order."""
    return pd.DataFrame(
        {
            "ts_id": ["a", "a", "a", "b", "b", "b"],
            "forecast_date": [
                "2026-02-03",
                "2026-02-01",
                "2026-02-02",
                "2026-02-02",
                "2026-02-03",
                "2026-02-01",
            ],
            "y_true": [12.0, 10.0, 11.0, 101.0, 102.0, 100.0],
            "yhat": [12.5, 9.5, 11.2, 99.0, 103.0, 101.0],
            "yhat_lower": [11.0, 8.0, 10.0, 95.0, 99.0, 97.0],
            "yhat_upper": [14.0, 11.0, 12.5, 104.0, 107.0, 105.0],
        }
    )


# Different *volatility*, not just different level: MASE divides by the mean absolute step of
# y_train, so two histories that merely sit at different levels would produce the same scale and the
# swap below would prove nothing. "a" steps by 1, "b" by 10.
_HIST = {
    "a": pd.Series([8.0, 9.0, 10.0]).to_numpy(),
    "b": pd.Series([80.0, 90.0, 100.0]).to_numpy(),
}


def test_a_fold_yields_one_oof_row_per_observation_and_one_panel_per_series() -> None:
    oof, panels = _score_fold(_eval_df(), _HIST, run_id="r", model_name="arima_plus", fold_id=2)
    assert len(oof) == 6
    assert set(panels) == {"a", "b"}
    assert {r["fold_id"] for r in oof} == {2}
    assert {r["model_type"] for r in oof} == {"arima_plus"}


def test_the_oof_rows_come_out_in_horizon_order_within_each_series() -> None:
    # The eval frame arrives shuffled; scoring sorts it, and the rows written must reflect that or
    # the OOF table records a horizon that never happened in that order.
    oof, _ = _score_fold(_eval_df(), _HIST, run_id="r", model_name="arima_plus", fold_id=0)
    per_series: dict[str, list[str]] = {}
    for row in oof:
        per_series.setdefault(row["ts_id"], []).append(row["forecast_date"])
    assert per_series["a"] == ["2026-02-01", "2026-02-02", "2026-02-03"]
    assert per_series["b"] == ["2026-02-01", "2026-02-02", "2026-02-03"]


def test_each_series_is_scaled_by_its_own_history_not_a_shared_one() -> None:
    # MASE/RMSSE divide by a scale computed from y_train. The two histories step at different rates,
    # so feeding either series the other's history moves its scaled metrics by that ratio.
    _, panels = _score_fold(_eval_df(), _HIST, run_id="r", model_name="arima_plus", fold_id=0)
    correct = panels["b"]["mase"]
    _, swapped = _score_fold(
        _eval_df(), {"a": _HIST["b"], "b": _HIST["a"]}, run_id="r", model_name="m", fold_id=0
    )
    assert not math.isclose(correct, swapped["b"]["mase"])


def test_a_series_with_no_history_still_scores_the_unscaled_metrics() -> None:
    # `hist_by_id.get` returns None for a series the history read did not cover; that must degrade
    # to NaN scaled metrics rather than dropping the series out of the leaderboard entirely.
    _, panels = _score_fold(_eval_df(), {}, run_id="r", model_name="arima_plus", fold_id=0)
    assert set(panels) == {"a", "b"}
    assert not math.isnan(panels["a"]["mae"])
    assert math.isnan(panels["a"]["mase"])


def test_the_interval_bounds_reach_the_panel_so_coverage_is_a_real_number() -> None:
    # The native eval query returns interval bounds; the Python worker's OOF path does not, and its
    # coverage/pinball are NaN. Passing them through is what makes the native numbers different --
    # and dropping them would silently NaN two columns for native models only.
    _, panels = _score_fold(_eval_df(), _HIST, run_id="r", model_name="arima_plus", fold_id=0)
    assert not math.isnan(panels["a"]["coverage"])
    assert not math.isnan(panels["a"]["pinball"])
    # Every actual for "a" falls inside its band, so coverage is total.
    assert panels["a"]["coverage"] == 1.0


def test_an_empty_fold_scores_nothing_rather_than_raising() -> None:
    empty = _eval_df().iloc[0:0]
    oof, panels = _score_fold(empty, _HIST, run_id="r", model_name="arima_plus", fold_id=0)
    assert oof == []
    assert panels == {}
