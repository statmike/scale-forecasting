"""Tests for the pure example-data generator (CONTRACTS §6, DESIGN §13.1, BUILD 2.5a).

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
)

SEED = 20260726


def _cfg(**over: object) -> GenConfig:
    base: dict[str, object] = {"n_series": 20, "history_days": 120, "holidays": ("US",)}
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
    cfg = _cfg(n_series=5, history_days=100)
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
    assert "price_index" in df.columns
    assert df["price_index"].notna().all()


# --- the partition-union invariant (what the Spark seed job relies on) ----------


def test_partition_union_equals_full_panel() -> None:
    cfg = _cfg(n_series=12, history_days=90)
    full = generate_panel(12, cfg, SEED)

    # Arbitrary, uneven partitioning of range(12).
    parts = [range(0, 3), range(3, 4), range(4, 10), range(10, 12)]
    union = pd.concat([generate_partition(p, cfg, SEED) for p in parts], ignore_index=True)

    pd.testing.assert_frame_equal(full, union)


def test_series_limit_is_a_prefix() -> None:
    cfg = _cfg(n_series=10, history_days=90)
    full = generate_panel(10, cfg, SEED)
    subset = generate_panel(4, cfg, SEED)  # data.series_limit=4 → first 4 series

    prefix = full[full["ts_id"].isin(subset["ts_id"].unique())].reset_index(drop=True)
    pd.testing.assert_frame_equal(subset, prefix)


def test_single_series_stable_across_partitions() -> None:
    cfg = _cfg(n_series=50, history_days=60)
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
