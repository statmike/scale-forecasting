"""Tests for the pre-flight input-contract validator (validation.py).

The validator's whole job is to fail fast with a message that names the offender, so each
test asserts both that the right inputs pass and that a bad input raises ``DataError`` with
a diagnostic that points at the specific series/column/value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scale_forecasting.config import RunConfig
from scale_forecasting.errors import DataError
from scale_forecasting.validation import ValidationReport, validate_panel


def _cfg(**data_overrides: object) -> RunConfig:
    data = {"source_table": "proj.ds.source_series", **data_overrides}
    return RunConfig(run_name="t", data=data, models=["theta"])


def _panel(
    n_series: int = 2,
    n_days: int = 60,
    freq: str = "D",
    start: str = "2024-01-01",
    with_exog: bool = False,
) -> pd.DataFrame:
    """A clean, well-formed long panel keyed by the default ts_id/ds/y columns."""
    dates = pd.date_range(start, periods=n_days, freq=freq)
    frames = []
    for s in range(n_series):
        cols: dict[str, object] = {
            "ts_id": f"s_{s:03d}",
            "ds": dates,
            "y": np.linspace(10, 20, n_days) + s,
        }
        if with_exog:
            cols["price_index"] = np.linspace(100, 110, n_days)
        frames.append(pd.DataFrame(cols))
    return pd.concat(frames, ignore_index=True)


# --- happy path ----------------------------------------------------------------


def test_clean_panel_passes_and_reports() -> None:
    rep = validate_panel(_panel(n_series=3, n_days=60), _cfg(horizon=7))
    assert isinstance(rep, ValidationReport)
    assert rep.n_series == 3
    assert rep.n_rows == 180
    assert rep.freq == "D"
    assert rep.min_history == 60
    assert rep.first_date == pd.Timestamp("2024-01-01")


def test_weekly_and_monthly_grids_pass() -> None:
    validate_panel(_panel(n_days=40, freq="W", start="2024-01-07"), _cfg(freq="W", horizon=4))
    validate_panel(_panel(n_days=24, freq="MS", start="2024-01-01"), _cfg(freq="MS", horizon=3))


def test_exog_column_validated_when_declared() -> None:
    cfg = RunConfig(
        run_name="t",
        data={"source_table": "t", "horizon": 7},
        models=["sarimax"],
        features={"exog": ["price_index"]},
    )
    validate_panel(_panel(with_exog=True), cfg)


# --- structural failures -------------------------------------------------------


def test_empty_panel_raises() -> None:
    with pytest.raises(DataError, match="no rows"):
        validate_panel(pd.DataFrame({"ts_id": [], "ds": [], "y": []}), _cfg())


def test_unsupported_freq_raises() -> None:
    with pytest.raises(DataError, match="unsupported freq 'Q'"):
        validate_panel(_panel(), _cfg(freq="Q"))


def test_missing_target_column_lists_columns() -> None:
    df = _panel().drop(columns=["y"])
    with pytest.raises(DataError, match="missing column 'y'"):
        validate_panel(df, _cfg())


def test_missing_declared_exog_raises() -> None:
    cfg = RunConfig(
        run_name="t", data={"source_table": "t"}, models=["sarimax"],
        features={"exog": ["price_index"]},
    )
    with pytest.raises(DataError, match="missing column 'price_index'"):
        validate_panel(_panel(with_exog=False), cfg)


# --- dtype failures ------------------------------------------------------------


def test_non_date_timestamp_names_value() -> None:
    df = _panel()
    df["ds"] = df["ds"].astype(object)  # a real source (CSV/BQ) can hand us strings
    df.loc[5, "ds"] = "not-a-date"
    with pytest.raises(DataError, match="non-date value.*not-a-date"):
        validate_panel(df, _cfg())


def test_non_numeric_target_names_value() -> None:
    df = _panel().astype({"y": object})
    df.loc[3, "y"] = "oops"
    with pytest.raises(DataError, match="must be numeric.*oops"):
        validate_panel(df, _cfg())


# --- per-series spacing --------------------------------------------------------


def test_duplicate_timestamp_names_series() -> None:
    df = _panel(n_series=2, n_days=60)
    # Duplicate a date inside the second series.
    dup_row = df[df["ts_id"] == "s_001"].iloc[10:11]
    df = pd.concat([df, dup_row], ignore_index=True)
    with pytest.raises(DataError, match="series 's_001' has duplicate timestamp"):
        validate_panel(df, _cfg(horizon=7))


def test_gap_in_series_names_first_offender() -> None:
    df = _panel(n_series=2, n_days=60)
    # Drop one interior day from s_001 → a gap on the daily grid.
    mask = ~((df["ts_id"] == "s_001") & (df["ds"] == pd.Timestamp("2024-01-15")))
    df = df[mask]
    with pytest.raises(DataError, match="series 's_001': gap at 2024-01-15"):
        validate_panel(df, _cfg(horizon=7))


def test_off_grid_timestamp_named() -> None:
    df = _panel(n_series=1, n_days=60)
    # Shift one point half a day off the daily grid.
    df.loc[20, "ds"] = pd.Timestamp("2024-01-21 06:00:00")
    with pytest.raises(DataError, match="not on the freq='D' grid"):
        validate_panel(df, _cfg(horizon=7))


def test_first_offending_series_reported_in_appearance_order() -> None:
    # s_000 appears first and is well-formed; s_001 has the gap → it must be the one named.
    df = _panel(n_series=2, n_days=60)
    mask = ~((df["ts_id"] == "s_001") & (df["ds"] == pd.Timestamp("2024-01-15")))
    df = df[mask]
    with pytest.raises(DataError, match="s_001"):
        validate_panel(df, _cfg(horizon=7))


# --- history sufficiency -------------------------------------------------------


def test_too_short_for_horizon_raises() -> None:
    df = _panel(n_series=1, n_days=5)
    with pytest.raises(DataError, match="only 5 observations, needs >= 12"):
        validate_panel(df, _cfg(horizon=10))


def test_too_short_for_backtest_raises() -> None:
    # backtest needs min_train + horizon + (n_folds-1)*step.
    cfg = RunConfig(
        run_name="t",
        data={"source_table": "t", "horizon": 7},
        models=["theta"],
        backtest={"enabled": True, "n_folds": 3, "horizon": 7, "step": 7, "min_train": 60},
    )
    df = _panel(n_series=1, n_days=60)  # need 60 + 7 + 14 = 81
    with pytest.raises(DataError, match="needs >= 81.*backtest"):
        validate_panel(df, cfg)


def test_backtest_history_sufficient_passes() -> None:
    cfg = RunConfig(
        run_name="t",
        data={"source_table": "t", "horizon": 7},
        models=["theta"],
        backtest={"enabled": True, "n_folds": 3, "horizon": 7, "step": 7, "min_train": 60},
    )
    validate_panel(_panel(n_series=1, n_days=81), cfg)


def test_min_history_override() -> None:
    df = _panel(n_series=1, n_days=30)
    with pytest.raises(DataError, match="needs >= 100.*caller-requested"):
        validate_panel(df, _cfg(horizon=7), min_history=100)
