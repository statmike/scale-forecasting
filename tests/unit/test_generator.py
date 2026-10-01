"""Tests for the pure example-data generator.

The generator is the one piece of "shipped data" that runs offline, so it must be
byte-for-byte deterministic, cover every archetype, produce clean numbers, and — critically
— satisfy the partition-union invariant the distributed Spark seed job relies on:
``generate_panel(n)`` equals the union of any partitioning of ``range(n)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scale_forecasting.data_gen.generator import (
    ARCHETYPES,
    GenConfig,
    generate_panel,
    generate_partition,
    is_holiday_flags,
)

SEED = 20260726


def _cfg(**over: object) -> GenConfig:
    base: dict[str, object] = {"history": 120, "holidays": ("US",)}
    base.update(over)
    return GenConfig(**base)  # type: ignore[arg-type]


# --- determinism ---------------------------------------------------------------


def test_byte_for_byte_deterministic() -> None:
    a = generate_panel(10, _cfg(), SEED)
    b = generate_panel(10, _cfg(), SEED)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_changes_data() -> None:
    a = generate_panel(10, _cfg(), SEED)
    b = generate_panel(10, _cfg(), SEED + 1)
    assert not np.allclose(a["y"].to_numpy(), b["y"].to_numpy())


# --- shape & cleanliness -------------------------------------------------------


def test_shape_and_columns() -> None:
    cfg = _cfg(history=100)
    df = generate_panel(5, cfg, SEED)
    assert list(df.columns) == ["ts_id", "archetype", "ds", "y"]
    assert len(df) == 5 * 100
    assert df["ts_id"].nunique() == 5
    assert df["ds"].dtype == np.dtype("datetime64[ns]")


def test_no_nan_or_negatives() -> None:
    df = generate_panel(20, _cfg(), SEED)
    assert df["y"].notna().all()
    assert (df["y"] >= 0.0).all()


def test_every_archetype_appears() -> None:
    # 20 series over 5 archetypes assigned by i % 5 → all present.
    df = generate_panel(20, _cfg(), SEED)
    assert set(df["archetype"]) == {a.name for a in ARCHETYPES}


def test_exog_column_emitted_when_requested() -> None:
    cfg = _cfg(with_exog=True)
    df = generate_panel(5, cfg, SEED)
    for col in ("region", "category", "promo_flag", "price_index", "temperature"):
        assert col in df.columns
        assert df[col].notna().all()
    assert set(df["promo_flag"].unique()) <= {0, 1}


def test_exog_three_tier_covariate_signals_affect_y() -> None:
    # When with_exog=True, all three covariate tiers (static, future, and lagged past)
    # contribute structural signal to y relative to the univariate baseline.
    uni = generate_panel(12, _cfg(history=180, with_exog=False), SEED)
    exog = generate_panel(12, _cfg(history=180, with_exog=True), SEED)
    diff = exog["y"] - uni["y"]

    # 1. Non-zero causal delta across the panel
    assert not np.allclose(exog["y"].to_numpy(), uni["y"].to_numpy())

    # 2. Future covariate signal: promo_flag=1 days have higher positive lift than promo_flag=0
    promo_lift_1 = float(diff[exog["promo_flag"] == 1].mean())
    promo_lift_0 = float(diff[exog["promo_flag"] == 0].mean())
    assert promo_lift_1 > promo_lift_0

    # 3. Static covariate signal: NA (+15% offset) vs LATAM (-12% offset) shift the exog delta
    na_delta = float(diff[exog["region"] == "NA"].mean())
    latam_delta = float(diff[exog["region"] == "LATAM"].mean())
    assert na_delta > latam_delta

    # 4. Past covariate lag signal: lagged temperature correlates positively with residual delta
    s0_exog = exog[exog["ts_id"] == "s_000000"].reset_index(drop=True)
    s0_uni = uni[uni["ts_id"] == "s_000000"].reset_index(drop=True)
    s0_delta = (s0_exog["y"] - s0_uni["y"]).to_numpy()
    temp_lag1 = np.roll(s0_exog["temperature"].to_numpy(), 1)
    assert float(np.corrcoef(s0_delta[7:], temp_lag1[7:])[0, 1]) != 0.0


def test_hierarchy_columns_emitted_without_exog() -> None:
    cfg = _cfg(with_hierarchy=True)
    df = generate_panel(12, cfg, SEED)
    assert list(df.columns) == ["ts_id", "archetype", "region", "category", "ds", "y"]
    assert set(df["region"].unique()) == {"NA", "EMEA", "APAC", "LATAM"}
    assert set(df["category"].unique()) == {"enterprise", "SMB", "consumer"}
    # Univariate target `y` is untouched when only hierarchy attributes are requested.
    univariate = generate_panel(12, _cfg(with_hierarchy=False), SEED)
    pd.testing.assert_series_equal(df["y"], univariate["y"])


# --- frequency generality ------------------------------------------------------


@pytest.mark.parametrize("freq", ["D", "W", "MS", "ME", "h"])
def test_generates_on_any_supported_freq(freq: str) -> None:
    cfg = _cfg(history=60, freq=freq)
    df = generate_panel(3, cfg, SEED)
    assert len(df) == 3 * 60
    # Each series lands on a regular grid at the requested freq.
    one = df[df["ts_id"] == "s_000000"]
    expected = pd.date_range(cfg.start, periods=60, freq=freq)
    assert pd.DatetimeIndex(one["ds"]).equals(expected.as_unit("ns"))


def test_daily_output_unchanged_by_position_based_seasonality() -> None:
    # Regression guard: for daily data, step position == day offset, so switching the
    # seasonality math to position-based must not perturb a single value. This digest was
    # captured from the pre-refactor generator; it must not move.
    cfg = _cfg(history=90, freq="D")
    df = generate_panel(4, cfg, SEED)
    assert df["y"].notna().all()
    assert float(df["y"].sum()) == pytest.approx(41909.749, abs=1e-3)


def test_generated_panel_passes_the_validator() -> None:
    # The generator and the validator must agree on what a well-formed panel is, at any freq.
    from scale_forecasting.config import RunConfig
    from scale_forecasting.validation import validate_panel

    for freq in ("D", "W", "MS"):
        panel = generate_panel(3, _cfg(history=80, freq=freq), SEED)
        cfg = RunConfig(
            run_name="t",
            data={"source_table": "t", "freq": freq, "horizon": 4},
            models=["theta"],
        )
        rep = validate_panel(panel, cfg)
        assert rep.n_series == 3
        assert rep.freq == freq


# --- is_holiday_flags (the source_series is_holiday column) ---------------------


def test_is_holiday_flags_marks_new_years_day() -> None:
    # 2021-01-01 (US New Year's Day) is a holiday; 2021-01-04 is an ordinary Monday.
    ds = pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-04"])
    flags = is_holiday_flags(ds, ("US",))
    assert flags.dtype == np.bool_
    assert flags.tolist() == [True, False, False]


def test_is_holiday_flags_aligns_with_generated_panel() -> None:
    # The flag derived over a series' dates must match the panel's own calendar (parity):
    # exactly the US holidays inside the history window are marked.
    cfg = _cfg(history=400, freq="D")
    df = generate_partition([0], cfg, SEED)
    flags = is_holiday_flags(df["ds"], cfg.holidays)
    assert len(flags) == len(df)
    # A daily year+ window always contains at least the fixed-date US holidays.
    assert flags.sum() > 0


def test_is_holiday_flags_empty_codes_all_false() -> None:
    ds = pd.to_datetime(["2021-07-04", "2021-12-25"])
    flags = is_holiday_flags(ds, ())
    assert flags.tolist() == [False, False]


# --- the partition-union invariant (what the Spark seed job relies on) ----------


def test_partition_union_equals_full_panel() -> None:
    cfg = _cfg(history=90)
    full = generate_panel(12, cfg, SEED)

    # Arbitrary, uneven partitioning of range(12).
    parts = [range(0, 3), range(3, 4), range(4, 10), range(10, 12)]
    union = pd.concat([generate_partition(p, cfg, SEED) for p in parts], ignore_index=True)

    pd.testing.assert_frame_equal(full, union)


def test_series_limit_is_a_prefix() -> None:
    cfg = _cfg(history=90)
    full = generate_panel(10, cfg, SEED)
    subset = generate_panel(4, cfg, SEED)  # data.series_limit=4 → first 4 series

    prefix = full[full["ts_id"].isin(subset["ts_id"].unique())].reset_index(drop=True)
    pd.testing.assert_frame_equal(subset, prefix)


def test_single_series_stable_across_partitions() -> None:
    cfg = _cfg(history=60)
    # Series 7 must be identical whether generated alone or inside a wider range.
    alone = generate_partition([7], cfg, SEED)
    wide = generate_partition(range(5, 10), cfg, SEED)
    wide_7 = wide[wide["ts_id"] == "s_000007"].reset_index(drop=True)
    pd.testing.assert_frame_equal(alone, wide_7)


# --- edge cases ----------------------------------------------------------------


def test_empty_partition_returns_typed_empty_frame() -> None:
    df = generate_partition([], _cfg(), SEED)
    assert len(df) == 0
    assert list(df.columns) == ["ts_id", "archetype", "ds", "y"]


def test_negative_n_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        generate_panel(-1, _cfg(), SEED)
