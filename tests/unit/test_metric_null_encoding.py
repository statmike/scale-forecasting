"""One encoding for "no metric" across every writer of ``forecast_metadata``.

Three code paths append to that table — the Python worker (`registry.rows.assemble_metadata_row`),
the BigQuery-native engine (`engines.bigquery_engine._meta_row`) and the ensemble orchestrator
(`ensemble_run._ensemble_meta_row`) — and all three represent an unscored metric in memory the same
way, as ``float("nan")``. Only the first used to convert it on the way out. The other two wrote the
NaN through, so a backtest-off run landed NULL for its Python models and NaN for its native ones, in
the same column of the same table.

That is not cosmetic. BigQuery sorts NaN *before* every real number, so ``ORDER BY wape`` puts an
unscored model at the head of the leaderboard — which is exactly what smoke 13 printed on
2026-09-02, with `arima_plus` and `timesfm` above two models that had actually been scored.

These tests are written against the row builders rather than the write path because that is where
the divergence lived, and they cover all three together on purpose: the invariant is registry-wide,
so a fourth writer added later should fail here rather than reintroduce the split quietly.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from scale_forecasting.config import RunConfig
from scale_forecasting.engines.bigquery_engine import _meta_row
from scale_forecasting.ensemble_run import _ensemble_meta_row
from scale_forecasting.metrics import METRIC_NAMES

_CREATED_AT = datetime(2026, 9, 2, tzinfo=UTC)


def _cfg() -> RunConfig:
    return RunConfig(
        **{
            "run_name": "metric null encoding",
            "data": {"source_table": "source_series_native", "horizon": 7},
            "models": ["theta", "arima_plus"],
        }
    )


def _unscored_panel() -> dict[str, float]:
    """What every engine builds when there is nothing to score: NaN in every metric."""
    return {name: float("nan") for name in METRIC_NAMES}


def _non_finite_panel() -> dict[str, float]:
    """The other way a metric goes bad — a runaway series overflowing an inverse transform."""
    return {name: float("inf") for name in METRIC_NAMES}


def _native_row(panel: dict[str, float]) -> dict[str, Any]:
    return _meta_row("run-1", "s_0", "arima_plus", panel, "{}", _CREATED_AT, _cfg())


def _ensemble_row(panel: dict[str, float]) -> dict[str, Any]:
    return _ensemble_meta_row(
        run_id="run-1",
        ts_id="s_0",
        model_type="ensemble_mean",
        panel=panel,
        ensemble_id="ens-1",
        weights=None,
        artifact_uri=None,
        created_at=_CREATED_AT,
        cfg=_cfg(),
    )


def test_the_native_engine_writes_an_unscored_metric_as_null_not_nan() -> None:
    row = _native_row(_unscored_panel())
    assert [row[name] for name in METRIC_NAMES] == [None] * len(METRIC_NAMES)


def test_the_ensemble_writer_writes_an_unscored_metric_as_null_not_nan() -> None:
    row = _ensemble_row(_unscored_panel())
    assert [row[name] for name in METRIC_NAMES] == [None] * len(METRIC_NAMES)


def test_infinities_are_nulled_too_not_just_nans() -> None:
    # A non-finite metric is the case `_as_float` was written for in the first place: the Storage
    # Write API rejects it for a FLOAT64 column, and one rejected row fails the whole append.
    for row in (_native_row(_non_finite_panel()), _ensemble_row(_non_finite_panel())):
        assert [row[name] for name in METRIC_NAMES] == [None] * len(METRIC_NAMES)


def test_a_real_metric_still_comes_through_as_a_float() -> None:
    # The coercion must not swallow scores — this is the case the leaderboard is actually about.
    panel = {**_unscored_panel(), "wape": 0.25}
    for row in (_native_row(panel), _ensemble_row(panel)):
        assert row["wape"] == 0.25
