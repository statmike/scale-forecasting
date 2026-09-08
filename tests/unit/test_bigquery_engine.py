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
from typing import Any

import pandas as pd

from scale_forecasting.config import RunConfig
from scale_forecasting.engines.bigquery_engine import _score_fold


def _cfg(**over: Any) -> RunConfig:
    """A daily run (seasonal period 7) with the default expanding backtest scheme."""
    base: dict[str, Any] = {
        "run_name": "bq engine test",
        "data": {"source_table": "source_series_native", "horizon": 3, "freq": "D"},
        "models": ["arima_plus"],
    }
    base.update(over)
    return RunConfig(**base)


def _eval_df(cutoff: str = "2026-01-31") -> pd.DataFrame:
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
            # One cutoff for the whole fold — the native path trains every series to the same
            # global origin — with the step counted from it, so the geometry travels with the row
            # even though the frame arrives shuffled.
            "cutoff_date": [cutoff] * 6,
            "horizon_step": [3, 1, 2, 2, 3, 1],
        }
    )


def _hist(ts_id: str, values: list[float], start: str = "2026-01-01") -> tuple[Any, Any]:
    """One series' history as the ``(ds, y)`` pair `_score_fold` looks up, daily from ``start``."""
    ds = pd.date_range(start, periods=len(values), freq="D")
    return (ds.to_numpy(), pd.Series(values, name=ts_id).to_numpy())


# Different *volatility*, not just different level: MASE divides by the mean absolute step of
# y_train, so two histories that merely sit at different levels would produce the same scale and the
# swap below would prove nothing. "a" steps by 1, "b" by 10. Both run to 2026-01-31, the fold's
# cutoff, so the whole of each is inside the training window unless a test moves the cutoff.
_HIST = {
    "a": _hist("a", [float(8 + i) for i in range(31)]),
    "b": _hist("b", [float(80 + 10 * i) for i in range(31)]),
}


def test_a_fold_yields_one_oof_row_per_observation_and_one_panel_per_series() -> None:
    oof, panels = _score_fold(
        _eval_df(), _HIST, run_id="r", model_name="arima_plus", fold_id=2, cfg=_cfg()
    )
    assert len(oof) == 6
    assert set(panels) == {"a", "b"}
    assert {r["fold_id"] for r in oof} == {2}
    assert {r["model_type"] for r in oof} == {"arima_plus"}


def test_the_oof_rows_come_out_in_horizon_order_within_each_series() -> None:
    # The eval frame arrives shuffled; scoring sorts it, and the rows written must reflect that or
    # the OOF table records a horizon that never happened in that order.
    oof, _ = _score_fold(
        _eval_df(), _HIST, run_id="r", model_name="arima_plus", fold_id=0, cfg=_cfg()
    )
    per_series: dict[str, list[str]] = {}
    for row in oof:
        per_series.setdefault(row["ts_id"], []).append(row["forecast_date"])
    assert per_series["a"] == ["2026-02-01", "2026-02-02", "2026-02-03"]
    assert per_series["b"] == ["2026-02-01", "2026-02-02", "2026-02-03"]


def test_each_series_is_scaled_by_its_own_history_not_a_shared_one() -> None:
    # MASE/RMSSE divide by a scale computed from y_train. The two histories step at different rates,
    # so feeding either series the other's history moves its scaled metrics by that ratio.
    _, panels = _score_fold(
        _eval_df(), _HIST, run_id="r", model_name="arima_plus", fold_id=0, cfg=_cfg()
    )
    correct = panels["b"]["mase"]
    _, swapped = _score_fold(
        _eval_df(),
        {"a": _HIST["b"], "b": _HIST["a"]},
        run_id="r",
        model_name="m",
        fold_id=0,
        cfg=_cfg(),
    )
    assert not math.isclose(correct, swapped["b"]["mase"])


def test_the_scale_denominator_stops_at_the_folds_cutoff() -> None:
    # The whole point of 3.2: the fold's own training window is the denominator, not the series.
    # "a" steps by 1 up to 2026-01-21 and then by 100 — so an earlier cutoff that excludes the
    # volatile tail has to produce a *different* (much larger) MASE than one that includes it. If
    # the cut were dropped, both calls would score against the same full history and agree.
    quiet_then_wild = [float(8 + i) for i in range(21)] + [
        float(28 + 100 * i) for i in range(1, 11)
    ]
    hist = {"a": _hist("a", quiet_then_wild), "b": _HIST["b"]}
    _, early = _score_fold(
        _eval_df(cutoff="2026-01-21"), hist, run_id="r", model_name="m", fold_id=0, cfg=_cfg()
    )
    _, late = _score_fold(
        _eval_df(cutoff="2026-01-31"), hist, run_id="r", model_name="m", fold_id=0, cfg=_cfg()
    )
    assert early["a"]["mase"] > late["a"]["mase"] * 10
    # The unscaled metrics are the same forecast either way — only the denominator moved.
    assert math.isclose(early["a"]["mae"], late["a"]["mae"])


def test_the_sliding_scheme_scales_by_a_fixed_width_window() -> None:
    # `sliding` fits on the last `min_train` observations, so its denominator has to be the same
    # slice — the leading quiet stretch is not training data and must not dilute the scale.
    quiet_then_wild = [float(8 + i) for i in range(21)] + [
        float(28 + 100 * i) for i in range(1, 11)
    ]
    hist = {"a": _hist("a", quiet_then_wild), "b": _HIST["b"]}
    expanding = _cfg(backtest={"scheme": "expanding", "min_train": 10})
    sliding = _cfg(backtest={"scheme": "sliding", "min_train": 10})
    _, wide = _score_fold(_eval_df(), hist, run_id="r", model_name="m", fold_id=0, cfg=expanding)
    _, narrow = _score_fold(_eval_df(), hist, run_id="r", model_name="m", fold_id=0, cfg=sliding)
    # The last 10 observations are all in the volatile stretch, so the sliding scale is larger and
    # the MASE correspondingly smaller.
    assert narrow["a"]["mase"] < wide["a"]["mase"]


def test_a_series_with_no_history_still_scores_the_unscaled_metrics() -> None:
    # `hist_by_id.get` returns None for a series the history read did not cover; that must degrade
    # to NaN scaled metrics rather than dropping the series out of the leaderboard entirely.
    _, panels = _score_fold(
        _eval_df(), {}, run_id="r", model_name="arima_plus", fold_id=0, cfg=_cfg()
    )
    assert set(panels) == {"a", "b"}
    assert not math.isnan(panels["a"]["mae"])
    assert math.isnan(panels["a"]["mase"])


def test_the_run_frequencys_seasonal_period_reaches_the_native_panel() -> None:
    # `mase_seasonal` divides by the m-step naive, so the period is a real input to the number the
    # leaderboard ranks on. It comes from `cfg.data.freq` rather than a default; if that threading
    # broke, the column would still be a plausible float against the wrong cycle.
    _, weekly = _score_fold(_eval_df(), _HIST, run_id="r", model_name="m", fold_id=0, cfg=_cfg())
    _, monthly = _score_fold(
        _eval_df(),
        _HIST,
        run_id="r",
        model_name="m",
        fold_id=0,
        cfg=_cfg(data={"source_table": "source_series_native", "horizon": 3, "freq": "MS"}),
    )
    assert not math.isnan(weekly["a"]["mase_seasonal"])
    assert not math.isclose(weekly["a"]["mase_seasonal"], monthly["a"]["mase_seasonal"])


def test_the_interval_bounds_reach_the_panel_so_coverage_is_a_real_number() -> None:
    # The native eval query returns interval bounds; the Python worker's OOF path does not, and its
    # coverage/pinball are NaN. Passing them through is what makes the native numbers different --
    # and dropping them would silently NaN two columns for native models only.
    _, panels = _score_fold(
        _eval_df(), _HIST, run_id="r", model_name="arima_plus", fold_id=0, cfg=_cfg()
    )
    assert not math.isnan(panels["a"]["coverage"])
    assert not math.isnan(panels["a"]["pinball"])
    # Every actual for "a" falls inside its band, so coverage is total.
    assert panels["a"]["coverage"] == 1.0


def test_an_empty_fold_scores_nothing_rather_than_raising() -> None:
    empty = _eval_df().iloc[0:0]
    oof, panels = _score_fold(
        empty, _HIST, run_id="r", model_name="arima_plus", fold_id=0, cfg=_cfg()
    )
    assert oof == []
    assert panels == {}


def test_the_folds_geometry_travels_from_the_eval_frame_onto_the_oof_rows() -> None:
    # `fold_id` is an ordinal, and the two engines number their folds from different anchors: the
    # Python path counts back from each series' own last observation, this path from one global
    # MAX(ds). On a ragged panel the same ordinal is therefore a different training window per
    # engine, so an ensemble joining on it pairs nothing and silently blends one model. The cutoff
    # is what makes the two comparable, which is why it has to reach the table rather than stay in
    # the query. See `ensembler._fold_key`.
    oof, _ = _score_fold(
        _eval_df(), _HIST, run_id="r", model_name="arima_plus", fold_id=2, cfg=_cfg()
    )
    assert {r["cutoff_date"] for r in oof} == {"2026-01-31"}
    by_date = {(r["ts_id"], r["forecast_date"]): r["horizon_step"] for r in oof}
    assert by_date[("a", "2026-02-01")] == 1
    assert by_date[("a", "2026-02-03")] == 3
    # The step is the row's own, not the position it happened to land at after the sort.
    assert by_date[("b", "2026-02-02")] == 2


def test_a_fold_without_the_geometry_columns_still_produces_rows() -> None:
    # A run whose OOF rows predate the two columns, or a hand-built frame, must not raise here --
    # the ensemble's join key falls back to the ordinal when the cutoff is missing, and the scale
    # denominator falls back to the whole history it had before the cut existed.
    bare = _eval_df().drop(columns=["cutoff_date", "horizon_step"])
    oof, panels = _score_fold(bare, _HIST, run_id="r", model_name="m", fold_id=0, cfg=_cfg())
    assert len(oof) == 6
    assert {r["cutoff_date"] for r in oof} == {None}
    assert {r["horizon_step"] for r in oof} == {None}
    assert not math.isnan(panels["a"]["mase"])
