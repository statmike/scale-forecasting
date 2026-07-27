"""Tests for feature engineering (CONTRACTS §6, DESIGN §4, BUILD 2.3).

Covers: log1p round-trips (apply→invert is identity), holiday parity to the `holidays`
package, exog pass-through, lag/Fourier/holiday-flag columns, and the (y, X) shape/index.
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
    apply_transform,
    build_features,
    holiday_frame,
    invert_transform,
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


def test_boxcox_rejected_for_now() -> None:
    with pytest.raises(ConfigError, match="boxcox"):
        apply_transform(pd.Series([1.0]), "boxcox")
    with pytest.raises(ConfigError, match="boxcox"):
        invert_transform(np.array([1.0]), "boxcox")


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
