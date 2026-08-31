"""Tests for feature engineering.

Covers: log1p round-trips (apply→invert is identity), holiday parity to the `holidays`
package, exog pass-through, lag/Fourier/holiday-flag columns, the (y, X) shape/index,
level-shift detection, and the forecast-horizon design frame (`build_future_features`) —
whose whole job is that deterministic columns are recomputed at the *future* dates.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scale_forecasting.config import RunConfig
from scale_forecasting.errors import ConfigError
from scale_forecasting.features import (
    _fourier_terms,
    apply_transform,
    build_features,
    build_future_features,
    fit_transform_lambda,
    holiday_frame,
    invert_transform,
    level_shift_step,
)


def _cfg(features: dict[str, Any] | None = None, data: dict[str, Any] | None = None) -> RunConfig:
    base_data = {"source_table": "p.d.s"}
    if data:
        base_data.update(data)
    kw: dict[str, Any] = {"run_name": "r", "data": base_data, "models": ["theta"]}
    if features is not None:
        kw["features"] = features
    return RunConfig(**kw)


def _series(n: int = 10, with_exog: bool = False) -> pd.DataFrame:
    ds = pd.date_range("2026-01-01", periods=n, freq="D")
    frame = {"ds": ds, "y": np.arange(1.0, n + 1.0)}
    if with_exog:
        frame["price_index"] = np.arange(100.0, 100.0 + n)
    return pd.DataFrame(frame)


# --- transforms ----------------------------------------------------------------


def test_log1p_roundtrips() -> None:
    y = pd.Series([0.0, 1.0, 9.0, 99.0])
    fwd = apply_transform(y, "log1p")
    back = invert_transform(fwd.to_numpy(), "log1p")
    assert np.allclose(back, y.to_numpy())


def test_none_transform_is_identity() -> None:
    y = pd.Series([1.0, 2.0, 3.0])
    assert apply_transform(y, "none") is y
    assert np.allclose(invert_transform(y.to_numpy(), "none"), y.to_numpy())


def test_log1p_rejects_below_neg_one() -> None:
    with pytest.raises(ConfigError, match="log1p"):
        apply_transform(pd.Series([-2.0, 0.0]), "log1p")


def test_boxcox_roundtrips_with_fitted_lambda() -> None:
    # λ fit on the series drives both directions; apply→invert is identity.
    y = pd.Series([10.0, 12.0, 15.0, 11.0, 14.0, 20.0, 25.0, 18.0, 22.0, 30.0])
    lam = fit_transform_lambda(y, "boxcox")
    assert lam is not None
    fwd = apply_transform(y, "boxcox", lam)
    back = invert_transform(fwd.to_numpy(), "boxcox", lam)
    assert np.allclose(back, y.to_numpy())


def test_boxcox_lambda_is_deterministic() -> None:
    y = pd.Series([3.0, 7.0, 2.0, 9.0, 5.0, 11.0, 4.0])
    assert fit_transform_lambda(y, "boxcox") == fit_transform_lambda(y, "boxcox")


def test_fit_transform_lambda_none_for_stateless() -> None:
    y = pd.Series([1.0, 2.0, 3.0])
    assert fit_transform_lambda(y, "none") is None
    assert fit_transform_lambda(y, "log1p") is None


def test_boxcox_requires_positive_y() -> None:
    with pytest.raises(ConfigError, match="strictly positive"):
        fit_transform_lambda(pd.Series([1.0, 0.0, 3.0]), "boxcox")
    with pytest.raises(ConfigError, match="strictly positive"):
        fit_transform_lambda(pd.Series([1.0, -2.0]), "boxcox")


def test_boxcox_without_lambda_raises() -> None:
    # A caller that forgets to fit λ gets a clear error, not a silent mis-transform.
    with pytest.raises(ConfigError, match="fitted lambda"):
        apply_transform(pd.Series([1.0, 2.0]), "boxcox")
    with pytest.raises(ConfigError, match="fitted lambda"):
        invert_transform(np.array([1.0, 2.0]), "boxcox")


def test_build_features_applies_boxcox_with_lambda() -> None:
    s = _series(8)  # y = 1..8, strictly positive
    y_raw = s["y"].astype(float)
    lam = fit_transform_lambda(y_raw, "boxcox")
    y, _ = build_features(s, _cfg(features={"transform": "boxcox"}), lam)
    # forward-transformed values differ from raw but invert back to raw.
    assert not np.allclose(y.to_numpy(), y_raw.to_numpy())
    assert np.allclose(invert_transform(y.to_numpy(), "boxcox", lam), y_raw.to_numpy())


def test_unknown_transform_raises() -> None:
    with pytest.raises(ConfigError, match="unknown transform"):
        apply_transform(pd.Series([1.0]), "sqrt")


# --- holidays ------------------------------------------------------------------


def test_holiday_frame_empty_when_unconfigured() -> None:
    hf = holiday_frame(_cfg())
    assert list(hf.columns) == ["ds", "holiday"]
    assert len(hf) == 0


def test_holiday_frame_matches_holidays_package() -> None:
    import holidays as holidays_pkg

    hf = holiday_frame(_cfg(features={"holidays": ["US"]}))
    days = set(hf["ds"].dt.date)
    us = holidays_pkg.country_holidays("US", years=range(2015, 2036))
    # July 4th 2026 and New Year 2026 are US holidays and must be present.
    assert dt.date(2026, 7, 4) in days
    assert dt.date(2026, 1, 1) in days
    # exact parity of the set within the window
    assert days == set(us.keys())


def test_holiday_frame_unknown_code_raises() -> None:
    with pytest.raises(ConfigError, match="unknown holiday country code"):
        holiday_frame(_cfg(features={"holidays": ["ZZ"]}))


# --- build_features ------------------------------------------------------------


def test_build_features_bare_returns_y_and_none() -> None:
    y, X = build_features(_series(), _cfg())
    assert X is None
    assert y.name == "y"
    assert y.index.name == "ds"
    assert y.index.dtype == np.dtype("datetime64[ns]")
    assert len(y) == 10


def test_build_features_sorts_by_date() -> None:
    s = _series(5).iloc[::-1]  # reversed
    y, _ = build_features(s, _cfg())
    assert y.index.is_monotonic_increasing


def test_build_features_applies_transform_to_y() -> None:
    y, _ = build_features(_series(4), _cfg(features={"transform": "log1p"}))
    assert np.allclose(y.to_numpy(), np.log1p(np.arange(1.0, 5.0)))


def test_build_features_exog_passthrough() -> None:
    y, X = build_features(_series(6, with_exog=True), _cfg(features={"exog": ["price_index"]}))
    assert X is not None
    assert "price_index" in X.columns
    assert np.allclose(X["price_index"].to_numpy(), np.arange(100.0, 106.0))


def test_build_features_missing_exog_raises() -> None:
    with pytest.raises(ConfigError, match="exog column 'nope'"):
        build_features(_series(), _cfg(features={"exog": ["nope"]}))


def test_build_features_holiday_flag() -> None:
    # series spanning US New Year's Day 2026-01-01
    y, X = build_features(_series(10), _cfg(features={"holidays": ["US"]}))
    assert X is not None
    assert "is_holiday" in X.columns
    # 2026-01-01 is a holiday, 2026-01-02 is not
    assert X["is_holiday"].iloc[0] == 1.0
    assert X["is_holiday"].iloc[1] == 0.0


def test_build_features_lags() -> None:
    y, X = build_features(_series(6), _cfg(features={"lags": [1, 2]}))
    assert X is not None
    assert {"lag_1", "lag_2"} <= set(X.columns)
    # lag_1 of a 1..6 series: first is NaN, then y[t-1]
    assert np.isnan(X["lag_1"].iloc[0])
    assert X["lag_1"].iloc[1] == pytest.approx(1.0)


def test_build_features_fourier_terms() -> None:
    y, X = build_features(_series(8), _cfg(features={"fourier": True}))
    assert X is not None
    fcols = [c for c in X.columns if c.startswith("fourier_")]
    assert len(fcols) == 6  # order 3 → sin+cos × 3
    assert (X[fcols].abs() <= 1.0 + 1e-9).all().all()


def test_build_features_X_aligned_to_y() -> None:
    cfg = _cfg(features={"exog": ["price_index"], "lags": [1]})
    y, X = build_features(_series(7, with_exog=True), cfg)
    assert X is not None
    assert X.index.equals(y.index)


def test_build_features_missing_target_raises() -> None:
    bad = _series().rename(columns={"y": "value"})
    with pytest.raises(ConfigError, match="missing required columns"):
        build_features(bad, _cfg())


# --- level shift ----------------------------------------------------------------


def _shifted(n: int = 60, cut: int = 30, jump: float = 50.0) -> pd.Series:
    """A flat, low-noise series with one abrupt additive jump at ``cut`` — the shape
    `data_gen.generator` plants via ``level_shift_prob``."""
    rng = np.random.default_rng(0)
    values = 100.0 + rng.normal(0.0, 1.0, n)
    values[cut:] += jump
    return pd.Series(values, index=pd.date_range("2026-01-01", periods=n, freq="D"))


def test_level_shift_step_finds_the_planted_changepoint() -> None:
    step = level_shift_step(_shifted(cut=30))
    assert step[:30].sum() == 0.0, "nothing flagged before the jump"
    assert step[30:].all(), "every observation from the jump onward is in the new regime"


def test_level_shift_step_is_a_step_not_a_spike() -> None:
    """The whole point of the encoding: a regime change persists, an outlier does not."""
    step = level_shift_step(_shifted())
    transitions = np.flatnonzero(np.diff(step) != 0)
    assert transitions.size == 1, f"a step changes value exactly once, saw {transitions.size}"


def test_level_shift_step_stays_silent_on_pure_noise() -> None:
    """A false positive hands a model a spurious regressor on a forecast nobody reviews."""
    rng = np.random.default_rng(7)
    quiet = pd.Series(100.0 + rng.normal(0.0, 1.0, 200))
    assert not level_shift_step(quiet).any()


def test_level_shift_step_zero_for_series_too_short_to_split() -> None:
    assert level_shift_step(pd.Series([1.0, 2.0, 3.0, 4.0])).tolist() == [0.0, 0.0, 0.0, 0.0]


def test_level_shift_step_zero_when_series_is_constant() -> None:
    """Zero noise scale must degrade to 'no shift', not divide by zero."""
    assert not level_shift_step(pd.Series(np.full(40, 5.0))).any()


def test_build_features_level_shift_column_is_opt_in() -> None:
    frame = pd.DataFrame({"ds": _shifted().index, "y": _shifted().to_numpy()})
    _, off = build_features(frame, _cfg(features={"fourier": True}))
    assert off is not None and "level_shift" not in off.columns
    _, on = build_features(frame, _cfg(features={"level_shift": True}))
    assert on is not None and on["level_shift"].tolist() == level_shift_step(_shifted()).tolist()


# --- the forecast-horizon design frame -------------------------------------------


def _future_cfg(features: dict[str, Any], horizon: int = 5) -> RunConfig:
    return _cfg(features=features, data={"horizon": horizon})


def test_build_future_features_none_when_no_features_configured() -> None:
    cfg = _future_cfg({})
    y, X = build_features(_series(20), cfg)
    assert X is None
    assert build_future_features(y, X, cfg) is None


def test_build_future_features_matches_training_columns_exactly() -> None:
    """Column *order* is load-bearing: `_lag_forecaster.recursive_predict` reads exog
    positionally, so a reordered frame feeds the wrong column to the wrong coefficient."""
    cfg = _future_cfg(
        {"exog": ["price_index"], "holidays": ["US"], "fourier": True, "lags": [1, 3]}
    )
    y, X = build_features(_series(30, with_exog=True), cfg)
    future = build_future_features(y, X, cfg)
    assert future is not None and X is not None
    assert list(future.columns) == list(X.columns)


def test_build_future_features_is_indexed_by_the_future() -> None:
    cfg = _future_cfg({"fourier": True}, horizon=5)
    y, X = build_features(_series(30), cfg)
    future = build_future_features(y, X, cfg)
    assert future is not None
    assert len(future) == 5
    assert future.index[0] == y.index[-1] + pd.Timedelta(days=1)
    assert (future.index > y.index[-1]).all()


def test_build_future_features_continues_the_fourier_phase() -> None:
    """The bug this frame exists to fix: handing a model the *first* horizon rows of history
    gave it the seasonal phase of four years ago for the dates it is forecasting."""
    cfg = _future_cfg({"fourier": True}, horizon=5)
    y, X = build_features(_series(400), cfg)
    future = build_future_features(y, X, cfg)
    assert future is not None and X is not None
    expected = _fourier_terms(pd.DatetimeIndex(future.index), cfg.data.freq, order=3)
    for name, values in expected.items():
        assert future[name].to_numpy() == pytest.approx(values)
    assert not np.allclose(future["fourier_sin_1"].to_numpy(), X["fourier_sin_1"].to_numpy()[:5])


def test_build_future_features_recomputes_holidays_at_the_future_dates() -> None:
    # A history ending 2025-12-30 puts New Year's Day inside a 5-day horizon.
    ds = pd.date_range("2025-11-01", periods=60, freq="D")
    frame = pd.DataFrame({"ds": ds, "y": np.arange(1.0, 61.0)})
    cfg = _future_cfg({"holidays": ["US"]}, horizon=5)
    y, X = build_features(frame, cfg)
    future = build_future_features(y, X, cfg)
    assert future is not None
    assert future.loc[pd.Timestamp("2026-01-01"), "is_holiday"] == 1.0


def test_build_future_features_carries_the_level_shift_forward() -> None:
    """A regime change is still in force over the horizon — that is what makes it a shift."""
    series = _shifted(n=60, cut=30)
    frame = pd.DataFrame({"ds": series.index, "y": series.to_numpy()})
    cfg = _future_cfg({"level_shift": True}, horizon=5)
    y, X = build_features(frame, cfg)
    future = build_future_features(y, X, cfg)
    assert future is not None
    assert (future["level_shift"] == 1.0).all()


def test_build_future_features_level_shift_stays_zero_when_none_detected() -> None:
    rng = np.random.default_rng(3)
    ds = pd.date_range("2026-01-01", periods=80, freq="D")
    frame = pd.DataFrame({"ds": ds, "y": 100.0 + rng.normal(0.0, 1.0, 80)})
    cfg = _future_cfg({"level_shift": True}, horizon=5)
    y, X = build_features(frame, cfg)
    future = build_future_features(y, X, cfg)
    assert future is not None and X is not None
    assert not X["level_shift"].any()
    assert not future["level_shift"].any()


def test_build_future_features_lags_are_real_observations_then_persist() -> None:
    cfg = _future_cfg({"lags": [3]}, horizon=5)
    y, X = build_features(_series(20), cfg)
    future = build_future_features(y, X, cfg)
    assert future is not None
    lag3 = future["lag_3"].to_numpy()
    # Steps 1..3 look back into real history; beyond that the history is persistence-extended.
    assert lag3[:3] == pytest.approx(y.to_numpy()[-3:])
    assert lag3[3:] == pytest.approx(np.full(2, float(y.iloc[-1])))


def test_build_future_features_exog_falls_back_to_the_most_recent_rows() -> None:
    """True exog is genuinely unknown; the stand-in should reflect the current regime, not
    the oldest one in the history."""
    cfg = _future_cfg({"exog": ["price_index"]}, horizon=5)
    y, X = build_features(_series(20, with_exog=True), cfg)
    future = build_future_features(y, X, cfg)
    assert future is not None and X is not None
    assert future["price_index"].to_numpy() == pytest.approx(X["price_index"].to_numpy()[-5:])


def test_build_future_features_handles_history_shorter_than_the_horizon() -> None:
    cfg = _future_cfg({"exog": ["price_index"]}, horizon=10)
    y, X = build_features(_series(4, with_exog=True), cfg)
    future = build_future_features(y, X, cfg)
    assert future is not None
    assert len(future) == 10, "length follows the horizon, never the history"
