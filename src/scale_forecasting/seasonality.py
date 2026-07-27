"""Frequency semantics — the one place that knows what each ``freq`` *means*.

Every model that decomposes seasonality needs "how many steps in one seasonal cycle"
(the SARIMA ``m``, Holt-Winters ``seasonal_periods``, STL ``period`` …), and the Fourier
features need "how many steps in one year". Both are pure functions of the run frequency,
so they live here once instead of being copy-pasted into every model file — a mismatch
would otherwise be invisible until a model silently used the wrong period.

Supported frequencies (``SUPPORTED_FREQS``) are the pandas offset aliases the product
understands end-to-end: daily, weekly, month-start, month-end, and hourly. Unknown
frequencies fall back to a weekly/daily-ish default rather than raising, so a model still
runs on an exotic freq — but :func:`is_supported` lets callers (the input validator)
reject one up front with a clear message.

Public surface: ``SUPPORTED_FREQS``, ``seasonal_period``, ``periods_per_year``,
``is_supported``.
"""

from __future__ import annotations

# Steps in one dominant seasonal cycle, per frequency (the SARIMA "m", STL period, …).
# Weekly-for-daily, yearly-for-the-rest — the cycle each model can actually estimate from
# a few years of history. Kept modest so a 100k-series batch stays tractable.
_SEASONAL_PERIOD: dict[str, int] = {"D": 7, "W": 52, "M": 12, "MS": 12, "H": 24}

# Steps in one year, per frequency — the period for the yearly Fourier terms.
_PERIODS_PER_YEAR: dict[str, float] = {
    "D": 365.25,
    "W": 52.18,
    "M": 12.0,
    "MS": 12.0,
    "H": 8766.0,  # 365.25 × 24
}

# The frequencies the product understands end-to-end (config → features → models → axis).
SUPPORTED_FREQS: tuple[str, ...] = ("D", "W", "M", "MS", "H")

_DEFAULT_PERIOD = 7
_DEFAULT_PERIODS_PER_YEAR = 365.25


def seasonal_period(freq: str) -> int:
    """Steps in one seasonal cycle for ``freq`` (e.g. 7 for daily, 12 for monthly).

    Falls back to a weekly-ish default for an unrecognized frequency so a model still
    runs; use :func:`is_supported` to reject unknown frequencies up front instead.
    """
    return _SEASONAL_PERIOD.get(freq, _DEFAULT_PERIOD)


def periods_per_year(freq: str) -> float:
    """Steps in one year for ``freq`` — the period for yearly Fourier features."""
    return _PERIODS_PER_YEAR.get(freq, _DEFAULT_PERIODS_PER_YEAR)


def is_supported(freq: str) -> bool:
    """True when ``freq`` is a frequency the product understands end-to-end."""
    return freq in SUPPORTED_FREQS
