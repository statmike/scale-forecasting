"""Pre-flight input-contract validator — pure, offline (CONTRACTS §2, DESIGN §9).

The product **specifies** the data shape (via the config) and **validates** that the input
conforms — it never *prepares* data. Resampling, aggregation, timezone/DST handling, and
bucket cutoffs are out of scope (a separate upstream project); here we only confirm the
source already matches what the config declares, and fail fast with a message that names
the offender if it doesn't.

Why fail fast here instead of in the worker: a shape problem (a missing column, a gap in
one series, the wrong ``freq``) would otherwise surface as thousands of failed cells deep
inside a distributed run. One pass over the panel up front turns that into a single clear
``DataError`` — *which* series, *what's* wrong — before any compute is scheduled.

Checks, in order (cheapest/most-fundamental first):
  1. the panel has rows,
  2. ``data.freq`` is a frequency the product understands end-to-end,
  3. the columns the config names (id / timestamp / target + declared exog) are present,
  4. the timestamp column parses to datetime and the target is numeric,
  5. per series: no duplicate timestamps, regular spacing at ``freq`` (no gaps / off-grid
     points) naming the first offender, and enough history for the run (and its backtest).

Public surface: ``ValidationReport``, ``validate_panel``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import RunConfig
from .errors import DataError
from .seasonality import SUPPORTED_FREQS, is_supported


@dataclass(frozen=True)
class ValidationReport:
    """A clean bill of health for a validated panel — handy to print in the dev loop."""

    n_series: int
    n_rows: int
    freq: str
    first_date: pd.Timestamp
    last_date: pd.Timestamp
    min_history: int  # shortest per-series length (rows)


def _required_min_history(cfg: RunConfig) -> tuple[int, str]:
    """The minimum per-series length this run needs, and a human reason for it.

    When backtesting is on, the binding constraint is the earliest fold's train window —
    the same arithmetic ``backtest.make_folds`` uses, checked here per series so a short
    series is caught up front rather than as a failed cell. Otherwise a series only needs
    enough points to fit and leave room for the forecast horizon.
    """
    bt = cfg.backtest
    if bt.enabled:
        need = bt.min_train + bt.horizon + (bt.n_folds - 1) * bt.step
        reason = (
            f"backtest needs min_train={bt.min_train} + horizon={bt.horizon} + "
            f"(n_folds={bt.n_folds}-1)*step={bt.step}"
        )
        return need, reason
    # No backtest: need at least a few points to fit and a horizon to forecast into.
    return cfg.data.horizon + 2, f"horizon={cfg.data.horizon} + 2 to fit"


def validate_panel(
    df: pd.DataFrame, cfg: RunConfig, *, min_history: int | None = None
) -> ValidationReport:
    """Validate that ``df`` conforms to the shape ``cfg`` declares (CONTRACTS §2).

    ``df`` is the long-format panel using the *business* column names the config maps
    (``data.ts_id_col`` / ``date_col`` / ``target_col`` and ``features.exog``) — the same
    columns ``features.build_features`` reads. Returns a :class:`ValidationReport` on
    success; raises :class:`DataError` naming the first offender on any violation.

    ``min_history`` overrides the auto-derived per-series minimum (from horizon/backtest)
    when a caller wants a stricter floor.
    """
    d = cfg.data

    # 1. Non-empty. --------------------------------------------------------------
    if df.empty:
        raise DataError("input panel has no rows")

    # 2. Frequency understood end-to-end. ---------------------------------------
    if not is_supported(d.freq):
        raise DataError(
            f"unsupported freq '{d.freq}'; supported: {', '.join(SUPPORTED_FREQS)}"
        )

    # 3. Declared columns present. ----------------------------------------------
    required = [d.ts_id_col, d.date_col, d.target_col, *cfg.features.exog]
    have = list(df.columns)
    for col in required:
        if col not in df.columns:
            raise DataError(f"missing column '{col}'; panel has {have}")

    # 4. Timestamp parses, target (and exog) numeric. ---------------------------
    ds = _parse_dates(df[d.date_col], d.date_col)
    _require_numeric(df[d.target_col], d.target_col)
    for col in cfg.features.exog:
        _require_numeric(df[col], col)

    # 5. Per-series: duplicates, spacing, history. ------------------------------
    need, reason = _required_min_history(cfg)
    if min_history is not None:
        need, reason = min_history, f"caller-requested min_history={min_history}"

    work = pd.DataFrame({"ts_id": df[d.ts_id_col].to_numpy(), "ds": ds.to_numpy()})
    shortest = None
    # Group order follows first appearance so the "first offender" is stable/reproducible.
    for ts_id, group in work.groupby("ts_id", sort=False):
        dates = pd.DatetimeIndex(group["ds"]).sort_values()
        _check_series_spacing(str(ts_id), dates, d.freq)
        if len(dates) < need:
            raise DataError(
                f"series '{ts_id}' has only {len(dates)} observations, needs >= {need} "
                f"({reason})"
            )
        shortest = len(dates) if shortest is None else min(shortest, len(dates))

    return ValidationReport(
        n_series=int(work["ts_id"].nunique()),
        n_rows=len(df),
        freq=d.freq,
        first_date=pd.Timestamp(ds.min()),
        last_date=pd.Timestamp(ds.max()),
        min_history=int(shortest or 0),
    )


# --- column-level helpers ------------------------------------------------------


def _parse_dates(col: pd.Series, name: str) -> pd.Series:
    """Parse a column to datetime64[ns], naming the first unparseable value on failure."""
    parsed = pd.to_datetime(col, errors="coerce")
    bad = parsed.isna() & col.notna()
    if bad.any():
        first = col[bad].iloc[0]
        raise DataError(f"column '{name}' has a non-date value: {first!r}")
    if parsed.isna().any():
        raise DataError(f"column '{name}' has missing (null) timestamps")
    return parsed.astype("datetime64[ns]")


def _require_numeric(col: pd.Series, name: str) -> None:
    """Raise if ``col`` isn't numeric, naming the first offending value."""
    coerced = pd.to_numeric(col, errors="coerce")
    bad = coerced.isna() & col.notna()
    if bad.any():
        first = col[bad].iloc[0]
        raise DataError(f"column '{name}' must be numeric; got {first!r}")


# --- per-series spacing --------------------------------------------------------


def _check_series_spacing(ts_id: str, dates: pd.DatetimeIndex, freq: str) -> None:
    """Confirm one series' sorted timestamps are regular at ``freq`` (CONTRACTS §2).

    Duplicates, gaps, and off-grid points are all reported against the *expected* grid
    ``date_range(first, last, freq)``, naming the first place the actual dates diverge.
    """
    # Duplicates first — they'd otherwise masquerade as a length mismatch below.
    dup = dates[dates.duplicated()]
    if len(dup) > 0:
        raise DataError(
            f"series '{ts_id}' has duplicate timestamp {dup[0].date()} for freq='{freq}'"
        )

    expected = pd.date_range(start=dates[0], end=dates[-1], freq=freq)
    if dates.equals(expected):
        return

    # Off-grid points (a timestamp that doesn't land on the expected grid) are reported
    # before gaps, since an off-grid point also *looks* like a missing grid point.
    off_grid = dates[~dates.isin(expected)]
    if len(off_grid) > 0:
        raise DataError(
            f"series '{ts_id}': timestamp {off_grid[0]} is not on the freq='{freq}' grid"
        )

    # Gaps: expected grid points missing from the actual timestamps.
    missing = expected[~expected.isin(dates)]
    if len(missing) > 0:
        raise DataError(
            f"series '{ts_id}': gap at {missing[0].date()} — missing from the "
            f"freq='{freq}' grid"
        )

    # Membership matches but the index still differs (shouldn't happen post dup-check).
    raise DataError(
        f"series '{ts_id}' does not form a regular freq='{freq}' grid "
        f"({len(dates)} points, expected {len(expected)})"
    )
