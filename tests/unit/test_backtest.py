"""Tests for backtest folds + out-of-fold predictions.

Covers fold geometry, the no-leakage invariant (train_end == val_start), expanding vs
sliding schemes, the short-series clamp (fewer folds, never an exception), and OOF frame
shape/units.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from scale_forecasting.backtest import Fold, achievable_folds, backtest_cell, make_folds
from scale_forecasting.config import RunConfig
from scale_forecasting.features import invert_transform
from scale_forecasting.models.base_model import DEFAULT_QUANTILES, BaseModel, ModelContext


def _cfg(
    backtest: dict[str, Any] | None = None, features: dict[str, Any] | None = None
) -> RunConfig:
    kw: dict[str, Any] = {
        "run_name": "r",
        "data": {"source_table": "p.d.s"},
        "models": ["theta"],
    }
    if backtest is not None:
        kw["backtest"] = {"enabled": True, **backtest}
    if features is not None:
        kw["features"] = features
    return RunConfig(**kw)


def _series(n: int) -> pd.DataFrame:
    ds = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.DataFrame({"ds": ds, "y": np.arange(1.0, n + 1.0)})


class _LastValue(BaseModel):
    """Deterministic: forecast = last training value, flat over the horizon."""

    name = "_lastval"
    runtime = "python"
    family = "statistical"

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> None:
        self._last = float(y.iloc[-1])

    def predict(
        self,
        horizon: int,
        X: pd.DataFrame | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        ds = pd.date_range("2026-01-01", periods=horizon, freq="D")
        # predict returns original units: invert the transform on the way out.
        yhat = invert_transform(
            np.full(horizon, self._last), self.ctx.transform, self.ctx.transform_lambda
        )
        return self._assemble_frame(ds, {q: yhat for q in quantiles})


def _factory(transform: str = "none") -> Any:
    def make() -> BaseModel:
        return _LastValue({}, ModelContext(freq="D", horizon=3, transform=transform))

    return make


# --- fold geometry -------------------------------------------------------------


def test_folds_count_and_ids() -> None:
    folds = make_folds(100, _cfg({"n_folds": 3, "horizon": 5, "step": 5, "min_train": 10}))
    assert len(folds) == 3
    assert [f.fold_id for f in folds] == [0, 1, 2]


def test_no_leakage_train_end_equals_val_start() -> None:
    folds = make_folds(100, _cfg({"n_folds": 4, "horizon": 7, "step": 7, "min_train": 20}))
    for f in folds:
        assert f.train_end == f.val_start  # training strictly precedes validation
        assert f.train_start < f.train_end
        assert f.val_start < f.val_end


def test_last_fold_validates_final_window() -> None:
    n = 100
    folds = make_folds(n, _cfg({"n_folds": 3, "horizon": 5, "step": 5, "min_train": 10}))
    assert folds[-1].val_end == n
    assert folds[-1].val_start == n - 5


def test_val_windows_are_horizon_sized_and_stepped() -> None:
    folds = make_folds(100, _cfg({"n_folds": 3, "horizon": 5, "step": 5, "min_train": 10}))
    for f in folds:
        assert f.val_size == 5
    # consecutive folds step by `step`
    assert folds[1].val_start - folds[0].val_start == 5


def test_expanding_scheme_grows_train_from_zero() -> None:
    folds = make_folds(100, _cfg({"n_folds": 3, "horizon": 5, "step": 5, "min_train": 10}))
    assert all(f.train_start == 0 for f in folds)
    # train grows fold to fold
    assert folds[0].train_size < folds[1].train_size < folds[2].train_size


def test_sliding_scheme_fixed_window() -> None:
    folds = make_folds(
        100, _cfg({"scheme": "sliding", "n_folds": 3, "horizon": 5, "step": 5, "min_train": 20})
    )
    for f in folds:
        assert f.train_size == 20


def test_a_series_too_short_for_every_fold_gets_the_folds_it_can_support() -> None:
    """It used to raise, and `run_cell` turned that into an error cell — losing the forecast.

    Backtesting scores a model; it does not produce the forecast. A series with 15 observations can
    be fit and forecast perfectly well, so a shortfall in *scoring* must cost only the score.
    """
    cfg = _cfg({"n_folds": 3, "horizon": 5, "step": 5, "min_train": 10})
    folds = make_folds(15, cfg)  # 15 - 5 - 10 = 0 slack: room for exactly one fold
    assert len(folds) == 1
    assert folds[0].train_size == 10  # exactly min_train, the tightest legal fold


def test_the_folds_dropped_are_the_oldest_and_the_survivors_keep_their_original_ids() -> None:
    """Both halves matter, and renumbering is the tempting mistake.

    Dropping the oldest keeps every series scored on the most recent window it can reach — the
    window a leaderboard is about. Keeping the original ids is what lets ``fold_id`` mean the same
    thing across a ragged panel: renumber the survivors 0..k and a short series' fold 0 silently
    lines up against a long series' fold 0 covering a completely different stretch of history.
    """
    cfg = _cfg({"n_folds": 3, "horizon": 5, "step": 5, "min_train": 10})
    full = make_folds(100, cfg)
    clamped = make_folds(20, cfg)  # slack 5 → two folds
    assert [f.fold_id for f in clamped] == [1, 2]  # fold 0, the oldest, is the one dropped
    # The survivors are the same folds they would have been, measured from the end of the series.
    assert [f.val_end - 20 for f in clamped] == [f.val_end - 100 for f in full[1:]]


def test_a_series_that_cannot_support_one_fold_gets_no_folds_rather_than_an_exception() -> None:
    cfg = _cfg({"n_folds": 3, "horizon": 5, "step": 5, "min_train": 10})
    assert make_folds(14, cfg) == []  # 14 - 5 = 9 < min_train
    assert achievable_folds(14, cfg) == 0


def test_achievable_folds_saturates_at_the_requested_count() -> None:
    """A long series does not get bonus folds — ``n_folds`` is what was asked for."""
    cfg = _cfg({"n_folds": 3, "horizon": 5, "step": 5, "min_train": 10})
    assert achievable_folds(10_000, cfg) == 3
    assert achievable_folds(15, cfg) == 1
    assert achievable_folds(20, cfg) == 2


def test_achievable_folds_agrees_with_the_folds_actually_built() -> None:
    """The count and the list must not drift; the cell records one and scores the other."""
    cfg = _cfg({"n_folds": 4, "horizon": 7, "step": 3, "min_train": 12})
    for n in range(0, 60):
        assert achievable_folds(n, cfg) == len(make_folds(n, cfg)), f"disagreed at n={n}"


def test_every_clamped_fold_still_honours_the_no_leakage_and_min_train_invariants() -> None:
    """Clamping must not buy folds by relaxing the geometry it was protecting."""
    cfg = _cfg({"n_folds": 4, "horizon": 7, "step": 3, "min_train": 12})
    for n in range(0, 60):
        for f in make_folds(n, cfg):
            assert f.train_end == f.val_start  # no leakage
            assert f.train_size >= 12  # min_train respected
            assert f.val_size == 7  # full-width validation window
            assert f.val_end <= n  # never reads past the series


# --- backtest_cell -------------------------------------------------------------


def test_oof_frame_shape_and_columns() -> None:
    cfg = _cfg({"n_folds": 3, "horizon": 4, "step": 4, "min_train": 10})
    oof, fold_metrics = backtest_cell(_series(40), _factory(), cfg)
    assert list(oof.columns) == ["ds", "fold_id", "y_true", "yhat"]
    assert len(oof) == 3 * 4  # n_folds × horizon
    assert oof["ds"].dtype == np.dtype("datetime64[ns]")
    assert sorted(oof["fold_id"].unique()) == [0, 1, 2]
    assert len(fold_metrics) == 3


def test_oof_values_match_lastvalue_model() -> None:
    # series is 1..40; last-value model on fold 0 (val at positions 32..36 for the
    # earliest window) predicts the value at the split point, flat.
    cfg = _cfg({"n_folds": 1, "horizon": 4, "step": 4, "min_train": 10})
    oof, _ = backtest_cell(_series(40), _factory(), cfg)
    # last training value is y at position val_start-1 = 35 → value 36.0
    assert np.allclose(oof["yhat"].to_numpy(), 36.0)
    # y_true is the actual future window: positions 36..39 → values 37..40
    assert np.allclose(oof["y_true"].to_numpy(), [37.0, 38.0, 39.0, 40.0])


def test_oof_in_original_units_under_log1p() -> None:
    # With log1p, yhat and y_true must both be back in original units.
    cfg = _cfg(
        backtest={"n_folds": 1, "horizon": 4, "step": 4, "min_train": 10},
        features={"transform": "log1p"},
    )
    oof, _ = backtest_cell(_series(40), _factory("log1p"), cfg)
    assert np.allclose(oof["y_true"].to_numpy(), [37.0, 38.0, 39.0, 40.0])
    # last-value model fit on log1p target, inverted → original last value 36.0
    assert np.allclose(oof["yhat"].to_numpy(), 36.0)


def test_fold_metrics_have_full_panel() -> None:
    from scale_forecasting.metrics import METRIC_NAMES

    cfg = _cfg({"n_folds": 2, "horizon": 4, "step": 4, "min_train": 10})
    _, fold_metrics = backtest_cell(_series(40), _factory(), cfg)
    for m in fold_metrics:
        assert set(m) == set(METRIC_NAMES)


def test_fold_dataclass_helpers() -> None:
    f = Fold(fold_id=0, train_start=0, train_end=30, val_start=30, val_end=35)
    assert f.train_size == 30
    assert f.val_size == 5
