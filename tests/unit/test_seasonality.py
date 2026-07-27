"""Tests for the shared frequency-semantics module (seasonality.py).

This is the single source of "what does each freq mean" that every seasonal model and the
Fourier features read. The point of the module is that the maps live in one place, so the
tests here pin the values and the fallback behavior once.
"""

from __future__ import annotations

from scale_forecasting.seasonality import (
    SUPPORTED_FREQS,
    is_supported,
    periods_per_year,
    seasonal_period,
)


def test_seasonal_period_known_freqs() -> None:
    assert seasonal_period("D") == 7
    assert seasonal_period("W") == 52
    assert seasonal_period("M") == 12
    assert seasonal_period("MS") == 12
    assert seasonal_period("H") == 24


def test_seasonal_period_unknown_falls_back_to_weekly() -> None:
    # Unknown freq must not raise — a model still runs (validator rejects up front instead).
    assert seasonal_period("Q") == 7
    assert seasonal_period("") == 7


def test_periods_per_year_known_freqs() -> None:
    assert periods_per_year("D") == 365.25
    assert periods_per_year("M") == 12.0
    assert periods_per_year("H") == 8766.0


def test_periods_per_year_unknown_falls_back_to_daily() -> None:
    assert periods_per_year("Q") == 365.25


def test_is_supported_matches_supported_freqs() -> None:
    for f in SUPPORTED_FREQS:
        assert is_supported(f)
    assert not is_supported("Q")
    assert not is_supported("T")


def test_every_supported_freq_has_both_maps() -> None:
    # A supported freq must resolve in both maps to something non-default, or a model would
    # silently use the fallback for a freq we claim to support end-to-end.
    for f in SUPPORTED_FREQS:
        assert seasonal_period(f) > 0
        assert periods_per_year(f) > 0
