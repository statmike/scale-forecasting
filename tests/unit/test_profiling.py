"""Offline tests for measured compute profiling (``scale_forecasting.profiling``).

No torch, no GPU, no Ray, no Spark, no BigQuery: every function under test here is pure, and the
one I/O function (`measure_fit`) is exercised only through its constructed-record arithmetic. That
is the whole point of the module's pure/I-O split — the numbers that decide how much hardware a run
gets have to be checkable on a laptop with no accelerator.

The load-bearing properties these tests defend:

* **The sample is a pure function of the panel.** Same stats in, byte-identical sample out, for a
  shuffled input and for a repeated call. Sizing is snapshotted for audit, so a sample that varies
  run-to-run is a correctness bug, not a nit. Every tie resolves on ``ts_id``.
* **The longest series is always in the sample, and always first.** A memory bound derived from a
  sample that excludes the longest series is not a bound. This holds at every budget, including 1,
  and regardless of where the longest series' id sorts or how boring its complexity is.
* **The sample spans the length range** rather than exhausting itself in one bucket, and reaches the
  complexity *extremes* inside a stratum rather than its medians.
* **Both tails are kept, and each drives the axis it is supposed to.** ``max`` governs safety
  (memory: the slot must hold the worst case), ``median`` governs throughput (time: the fleet is
  sized for typical work). These tests use deliberately skewed measurement sets so max and median
  are far apart and a swap would be visible.
* **No axis is ever reported as a number without evidence.** An unmeasured axis is ``None``, never
  ``0`` — because ``0 x 1.3`` is still a plan, and it is wrong. Absence must survive every roll-up.
* **A failing probe widens sizing, it never sinks the aggregation.** Failed measurements are
  excluded from every aggregate and still counted, so "we sized off 6 of 8 fits" stays visible.

`measure_fit` itself is the only function not exercised offline (its live branches carry
``# pragma: no cover``); its pure ``effective_cores`` arithmetic is tested against constructed
`MeasuredFit` records.
"""

from __future__ import annotations

import json
import math
import os
import random
from typing import Any

import pandas as pd
import pytest

from scale_forecasting import profiling
from scale_forecasting.config import RunConfig
from scale_forecasting.errors import DataError
from scale_forecasting.profiling import (
    MeasuredFit,
    SeriesStats,
    build_profile,
    select_profile_sample,
    series_stats,
)

_GIB = 1024**3


# --- helpers -------------------------------------------------------------------


def _cfg(**over: Any) -> RunConfig:
    base: dict[str, Any] = {
        "run_name": "profiling test",
        "data": {"source_table": "source_series_native"},
        "models": ["theta"],
    }
    base.update(over)
    return RunConfig(**base)


def _frame(
    ts_id: Any,
    values: list[Any],
    *,
    start: str = "2024-01-01",
    dates: Any = None,
    **extra: Any,
) -> pd.DataFrame:
    """One series in long format: ``ts_id`` / ``ds`` / ``y``, plus any passthrough columns."""
    n = len(values)
    frame = pd.DataFrame(
        {
            "ts_id": [ts_id] * n,
            "ds": pd.date_range(start, periods=n) if dates is None else dates,
            "y": list(values),
        }
    )
    for name, column in extra.items():
        frame[name] = column
    return frame


def _panel(*frames: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True)


def _stats(
    ts_id: str,
    n_obs: int,
    *,
    zero_fraction: float = 0.0,
    distinct_values: int = 9,
    diff_cv: float = 1.0,
    acf_at_season: float = 0.0,
    n_exog: int = 0,
) -> SeriesStats:
    """A `SeriesStats` with plausible, non-degenerate defaults, so a test varies one axis."""
    return SeriesStats(
        ts_id=ts_id,
        n_obs=n_obs,
        zero_fraction=zero_fraction,
        distinct_values=distinct_values,
        diff_cv=diff_cv,
        acf_at_season=acf_at_season,
        n_exog=n_exog,
    )


def _fit(
    model_type: str = "theta",
    *,
    ts_id: str = "s1",
    family: str = "statistical",
    n_obs: int = 100,
    wall_s: float = 1.0,
    cpu_s: float = 1.0,
    peak_rss_bytes: int = _GIB,
    peak_gpu_bytes: int | None = None,
    ok: bool = True,
    error: str | None = None,
    process_rss_bytes: int | None = -1,
) -> MeasuredFit:
    """An injected measurement — the offline seam. No fit runs, exactly as with GPU calibration.

    ``process_rss_bytes`` defaults to mirroring ``peak_rss_bytes`` (the ``-1`` sentinel) so a
    test that cares only about "the memory axis" sets one number and gets a consistent record.
    Pass it explicitly — including ``None`` — when the point of the test is that the two
    differ, which is the whole reason there are two of them.
    """
    return MeasuredFit(
        ts_id=ts_id,
        model_type=model_type,
        family=family,
        n_obs=n_obs,
        wall_s=wall_s,
        cpu_s=cpu_s,
        peak_rss_bytes=peak_rss_bytes,
        peak_gpu_bytes=peak_gpu_bytes,
        ok=ok,
        error=error,
        process_rss_bytes=peak_rss_bytes if process_rss_bytes == -1 else process_rss_bytes,
    )


# --- series_stats: determinism and ordering ------------------------------------


def test_stats_are_ordered_by_ts_id_not_panel_order() -> None:
    # The defect this replaces sampled an alphabetical prefix of whatever order the reader emitted.
    # Output order must be a property of the ids, never of row arrival.
    panel = _panel(_frame("b", [1, 2, 3]), _frame("a", [4, 5, 6]), _frame("c", [7, 8, 9]))
    assert [s.ts_id for s in series_stats(panel, _cfg())] == ["a", "b", "c"]


def test_stats_are_invariant_to_shuffled_rows() -> None:
    # Engines do not promise row order. If a statistic moved with it, two identical runs would size
    # two different fleets.
    panel = _panel(
        _frame("a", [1.0, 5.0, 2.0, 9.0, 0.0, 3.0]),
        _frame("b", [10.0, 11.0, 10.0, 12.0, 0.0, 14.0]),
    )
    assert series_stats(panel, _cfg()) == series_stats(panel.sample(frac=1, random_state=7), _cfg())


def test_stats_are_invariant_to_unsorted_dates_within_a_series() -> None:
    # diff_cv and the lag-m ACF are order-dependent by nature, so the sort by (ts_id, timestamp)
    # has to happen before them — otherwise a reverse-ordered read reports a different volatility.
    ordered = _frame("a", [1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0])
    reversed_rows = ordered.iloc[::-1].reset_index(drop=True)
    assert series_stats(ordered, _cfg()) == series_stats(reversed_rows, _cfg())


# --- series_stats: structural input --------------------------------------------


def test_empty_panel_yields_no_stats() -> None:
    # Zero rows carry zero information. Profiling must degrade to "no basis", not fail the run.
    empty = _frame("a", [1.0]).iloc[:0]
    assert series_stats(empty, _cfg()) == []


def test_bare_empty_frame_with_no_columns_yields_no_stats_not_a_raise() -> None:
    # The emptiness check comes *before* the column check: a frame with no columns at all is still
    # zero rows, and the column names are irrelevant when there is nothing to compute.
    assert series_stats(pd.DataFrame(), _cfg()) == []


@pytest.mark.parametrize("missing", ["ts_id", "ds", "y"])
def test_missing_id_date_or_target_column_raises_dataerror_naming_it(missing: str) -> None:
    # Structural violation, not degenerate content: the config points at the wrong table and every
    # number downstream would be computed from the wrong bytes. Same wording as the pre-flight
    # validator, so an operator sees one message vocabulary.
    panel = _panel(_frame("a", [1.0, 2.0, 3.0])).drop(columns=[missing])
    with pytest.raises(DataError, match=f"missing column '{missing}'"):
        series_stats(panel, _cfg())


def test_missing_declared_exog_degrades_and_is_not_counted_in_n_exog() -> None:
    # Profiling is a sizing pre-pass; it must never change whether a run starts. Turning a run that
    # today limps into a hard pre-flight failure would be a behaviour change made sideways.
    # Understating the feature width is bounded and absorbed by the memory margin.
    panel = _panel(_frame("a", [1.0, 2.0, 3.0], promo=[0, 1, 0]))
    cfg = _cfg(features={"exog": ["promo", "weather"]})
    stats = series_stats(panel, cfg)
    assert [s.n_exog for s in stats] == [1]  # 'promo' present, 'weather' declared but absent


def test_unparseable_dates_degrade_and_leave_n_obs_intact() -> None:
    # An unusable timestamp costs the secondary spread axes some fidelity. It does not change n_obs,
    # and n_obs is the axis that actually bounds memory.
    panel = _frame("a", [1.0, 2.0, 3.0, 4.0], dates=["nope", "also-nope", "still-nope", "nope"])
    (stats,) = series_stats(panel, _cfg())
    assert stats.n_obs == 4
    assert math.isfinite(stats.diff_cv)


def test_non_numeric_target_is_coerced_and_treated_as_gaps() -> None:
    # Same reasoning as the dates: length is intact, only the value statistics see the gap.
    panel = _frame("a", [1.0, "oops", 3.0])
    (stats,) = series_stats(panel, _cfg())
    assert stats.n_obs == 3  # the fit is handed all three rows
    assert stats.distinct_values == 2  # only two of them carry a value


def test_duplicate_timestamps_are_counted_not_deduped() -> None:
    # run_cell receives exactly these rows. The cost driver is what the fit sees, not what a clean
    # panel would have had, so deduping here would understate the work.
    stamps = pd.to_datetime(["2024-01-01"] * 4)
    (stats,) = series_stats(_frame("a", [1.0, 2.0, 3.0, 4.0], dates=stamps), _cfg())
    assert stats.n_obs == 4


def test_non_string_ts_ids_are_coerced_like_the_rest_of_the_codebase() -> None:
    # One id vocabulary everywhere: the chunker and the worker both coerce with str(), so the
    # profiler's sample ids line up with the ids the registry records.
    panel = _panel(_frame(2, [1.0, 2.0]), _frame(10, [3.0, 4.0]))
    stats = series_stats(panel, _cfg())
    assert all(isinstance(s.ts_id, str) for s in stats)
    assert [s.ts_id for s in stats] == ["10", "2"]  # string ordering, not numeric


# --- series_stats: degenerate content ------------------------------------------


def test_zero_fraction_and_distinct_values_flag_a_degenerate_series() -> None:
    # Intermittency is one of the three complexity axes; it has to be measured off the non-null
    # values, not off the row count.
    (stats,) = series_stats(_frame("a", [0.0, 0.0, 0.0, 5.0]), _cfg())
    assert stats.zero_fraction == pytest.approx(0.75)
    assert stats.distinct_values == 2


def test_diff_cv_of_a_flat_zero_series_is_zero_not_infinite() -> None:
    # The divide-by-level guard. An inf or NaN here would become a sort key in the sampler and
    # destroy the determinism the whole module rests on.
    (stats,) = series_stats(_frame("a", [0.0] * 10), _cfg())
    assert stats.diff_cv == 0.0
    assert stats.zero_fraction == 1.0
    assert stats.distinct_values == 1


def test_all_nan_series_yields_finite_zeroed_stats() -> None:
    # "No values at all" is a real panel state. Every field must still be a finite number.
    (stats,) = series_stats(_frame("a", [float("nan")] * 6), _cfg())
    assert stats.n_obs == 6  # the rows exist and the fit will be handed them
    assert stats.distinct_values == 0
    assert (stats.zero_fraction, stats.diff_cv, stats.acf_at_season) == (0.0, 0.0, 0.0)


def test_single_row_series_yields_finite_stats() -> None:
    # Cannot difference a single value; 0.0 is the honest answer, NaN is not.
    (stats,) = series_stats(_frame("a", [7.0]), _cfg())
    assert stats.n_obs == 1
    assert stats.diff_cv == 0.0
    assert stats.acf_at_season == 0.0


def test_acf_at_season_is_high_for_a_seasonal_series_and_low_for_noise() -> None:
    # Seasonal strength is the axis that separates "a model has structure to fit" from "it does
    # not". Daily freq means lag 7, so a period-7 repeat correlates perfectly with its own shift.
    rng = random.Random(0)
    panel = _panel(
        _frame("seasonal", [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0] * 6),
        _frame("noise", [rng.random() for _ in range(42)]),
    )
    by_id = {s.ts_id: s for s in series_stats(panel, _cfg())}
    assert by_id["seasonal"].acf_at_season == pytest.approx(1.0)
    assert abs(by_id["noise"].acf_at_season) < 0.5


def test_acf_at_season_is_zero_when_the_series_is_shorter_than_the_season() -> None:
    # No lagged pairs exist, so there is no correlation to estimate. 0.0 reads as "no evidence of
    # seasonality", which is true; NaN would poison a sort key.
    (stats,) = series_stats(_frame("a", [1.0, 2.0, 3.0, 4.0, 5.0]), _cfg())  # 5 rows, lag 7
    assert stats.acf_at_season == 0.0


def test_every_stat_field_is_finite_for_every_degenerate_panel() -> None:
    # The post-condition the sampler's determinism depends on: no inf, no NaN, ever, whatever the
    # content. Each of these is a panel a real source can produce.
    panels = {
        "all_zero": _frame("a", [0.0] * 8),
        "all_nan": _frame("a", [float("nan")] * 8),
        "constant": _frame("a", [3.5] * 8),
        "single_row": _frame("a", [1.0]),
        "infinite_value": _frame("a", [1.0, float("inf"), 2.0, float("-inf")]),
        "huge": _frame("a", [1e300, -1e300, 1e300]),
        "tiny_level": _frame("a", [1e-18, -1e-18, 1e-18, -1e-18]),
        "mixed_gaps": _frame("a", [0.0, float("nan"), 0.0, 4.0, float("nan")]),
    }
    for label, panel in panels.items():
        (stats,) = series_stats(panel, _cfg())
        assert stats.n_obs >= 1, label
        assert math.isfinite(stats.zero_fraction), label
        assert math.isfinite(stats.diff_cv), label
        assert math.isfinite(stats.acf_at_season), label
        assert -1.0 <= stats.acf_at_season <= 1.0, label
        assert 0.0 <= stats.zero_fraction <= 1.0, label
        assert stats.diff_cv >= 0.0, label
        assert stats.distinct_values >= 0, label


def test_diff_cv_is_the_population_sigma_of_the_differences_over_the_level() -> None:
    # Pinning the definition, not just its finiteness: sigma(diff(y), ddof=0) over
    # max(|mean(y)|, guard). A ddof=1 or a mean-of-diffs denominator would silently rescale the
    # volatility axis and move which series the sampler calls complex.
    (stats,) = series_stats(_frame("a", [1.0, 2.0, 4.0, 8.0]), _cfg())
    diffs = [1.0, 2.0, 4.0]
    mean_diff = sum(diffs) / 3
    sigma = (sum((d - mean_diff) ** 2 for d in diffs) / 3) ** 0.5
    assert stats.diff_cv == pytest.approx(sigma / 3.75)  # mean(y) == 3.75


def test_zero_fraction_is_measured_over_non_null_values_not_over_rows() -> None:
    # A gap is not a zero. Dividing by n_obs would make a sparse-but-nonzero series look
    # intermittent and pull the sampler towards the wrong extreme.
    (stats,) = series_stats(_frame("a", [0.0, float("nan"), 0.0, 5.0]), _cfg())
    assert stats.n_obs == 4
    assert stats.zero_fraction == pytest.approx(2 / 3)
    assert stats.distinct_values == 2


def test_acf_keeps_the_sign_of_a_negative_seasonal_correlation() -> None:
    # A negative lag-m correlation is a real and *different* phenomenon from no correlation, so the
    # stored value is signed. Collapsing it to a magnitude here would destroy information the
    # record is meant to preserve; taking the magnitude is the sampler's job, not the statistic's.
    base = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    flipping = (base + [-v for v in base]) * 3  # y[t] == -y[t-7], exactly
    (stats,) = series_stats(_frame("a", flipping), _cfg())
    assert stats.acf_at_season == pytest.approx(-1.0)


def test_acf_at_season_is_zero_at_exactly_the_season_length() -> None:
    # n_valid == m leaves no lagged pairs at all, and n_valid == m + 1 leaves exactly one, which is
    # not a correlation either. Both must read 0.0 rather than NaN or a spurious +/-1.
    for n in (7, 8):
        (stats,) = series_stats(_frame("a", [float(i) for i in range(n)]), _cfg())
        assert stats.acf_at_season == 0.0, n


def test_the_seasonal_lag_follows_the_configured_frequency() -> None:
    # The lag is m = seasonal_period(freq), so the same values read as perfectly seasonal on a
    # monthly run (m = 12) and as something else entirely on a daily one (m = 7).
    values = [float(v) for v in range(1, 13)] * 4
    monthly = _cfg(data={"source_table": "s", "freq": "MS"})
    (as_monthly,) = series_stats(_frame("a", values), monthly)
    (as_daily,) = series_stats(_frame("a", values), _cfg())
    assert as_monthly.acf_at_season == pytest.approx(1.0)
    assert as_daily.acf_at_season != pytest.approx(1.0)


def test_an_unsupported_frequency_does_not_raise() -> None:
    # Rejecting a frequency is the pre-flight validator's job. A sizing pre-pass that refused an
    # exotic freq would block a run that would otherwise have completed.
    exotic = _cfg(data={"source_table": "s", "freq": "QS-JAN"})
    (stats,) = series_stats(_frame("a", [1.0, 2.0, 3.0, 4.0]), exotic)
    assert math.isfinite(stats.acf_at_season)


def test_ids_sort_by_string_comparison_not_by_case_folded_or_numeric_order() -> None:
    # The ordering contract is plain ascending string comparison, the same rule the tie-breaks use.
    # A case-insensitive or numeric-aware sort would make the sample depend on the id vocabulary.
    panel = _panel(_frame("a", [1.0, 2.0]), _frame("B", [3.0, 4.0]), _frame("C", [5.0, 6.0]))
    assert [s.ts_id for s in series_stats(panel, _cfg())] == ["B", "C", "a"]


def test_stats_have_no_duplicate_ids_and_stay_sorted_on_a_wide_panel() -> None:
    # The two post-conditions the sampler assumes and never re-checks (a duplicate id would make it
    # raise). One groupby, so this holds however many series arrive.
    panel = _panel(*[_frame(f"s{i:03d}", [float(i), float(i + 1), 0.0]) for i in range(50)])
    stats = series_stats(panel.sample(frac=1, random_state=3), _cfg())
    ids = [s.ts_id for s in stats]
    assert ids == sorted(ids)
    assert len(set(ids)) == 50


# --- select_profile_sample: the headline invariant ------------------------------


def test_sample_always_contains_the_longest_series_first() -> None:
    # The single most important property in the module: a memory bound derived from a sample that
    # excludes the longest series is not a bound. It must hold at every budget, and be readable
    # off index 0 in one line.
    stats = [_stats(f"s{i:02d}", n_obs=10 + 7 * i) for i in range(12)]
    for budget in range(1, 14):
        sample = select_profile_sample(stats, samples=budget)
        assert sample[0].ts_id == "s11"
        assert sample[0].reason == "longest"
        assert sample[0].n_obs == max(s.n_obs for s in stats)
        assert len(sample) == min(budget, len(stats))


def test_longest_wins_even_when_its_id_sorts_last_and_its_complexity_is_boring() -> None:
    # Everything else about this series argues against picking it: last-sorting id, zero on all
    # three complexity axes. Length still wins, because length is what bounds memory.
    stats = [
        _stats("aaa", n_obs=20, diff_cv=9.0, zero_fraction=0.9, acf_at_season=0.9),
        _stats("mmm", n_obs=25, diff_cv=5.0, zero_fraction=0.5, acf_at_season=0.5),
        _stats("zzz", n_obs=10_000, diff_cv=0.0, zero_fraction=0.0, acf_at_season=0.0),
    ]
    assert select_profile_sample(stats, samples=8)[0].ts_id == "zzz"
    assert select_profile_sample(stats, samples=1)[0].ts_id == "zzz"


def test_longest_is_included_at_a_budget_of_one() -> None:
    # The longest series is claimed *before* any budget arithmetic, so a budget of one spends it
    # on the series that bounds memory rather than on the first stratum.
    stats = [_stats(f"s{i}", n_obs=n) for i, n in enumerate([5, 900, 12, 60])]
    sample = select_profile_sample(stats, samples=1)
    assert [(s.ts_id, s.reason) for s in sample] == [("s1", "longest")]


# --- select_profile_sample: determinism -----------------------------------------


def test_sample_is_deterministic_across_repeated_calls() -> None:
    # No RNG, no clock, no set/dict iteration-order dependence. The sample is snapshotted for
    # audit, so a second call on the same panel must produce the same audit record.
    stats = [
        _stats(f"s{i:02d}", n_obs=3 + i * i, diff_cv=(i % 5) / 2, zero_fraction=(i % 3) / 3)
        for i in range(20)
    ]
    first = select_profile_sample(stats)
    for _ in range(5):
        assert select_profile_sample(stats) == first


def test_sample_is_independent_of_input_order() -> None:
    # The function sorts by ts_id before doing anything, so a Spark aggregation emitting rows in
    # partition order and a pandas groupby emitting them sorted must size the same fleet.
    stats = [
        _stats(f"s{i:02d}", n_obs=4 + i * 3, diff_cv=(i * 7 % 11) / 3, zero_fraction=(i % 4) / 4)
        for i in range(15)
    ]
    shuffled = list(stats)
    random.Random(11).shuffle(shuffled)
    assert select_profile_sample(shuffled) == select_profile_sample(stats)


def test_sample_ties_resolve_on_ts_id() -> None:
    # Ties are where nondeterminism hides, and there is no coin flip anywhere here. Several series
    # tied for longest: the smallest ts_id takes the label, the rest stay ordinary candidates.
    tied_longest = [_stats(i, n_obs=100) for i in ("b", "a", "c")]
    sample = select_profile_sample(tied_longest, samples=1)
    assert [(s.ts_id, s.reason) for s in sample] == [("a", "longest")]


def test_complexity_ranks_break_a_full_tie_on_ts_id() -> None:
    # Four series identical on every complexity axis. Rank normalization breaks the tie on ts_id,
    # so the composite is a total order with no ambiguity left for a sort to resolve differently on
    # another pandas/numpy build: ranks 0..3 scaled by 1/(n-1).
    sample = select_profile_sample([_stats(i, n_obs=50) for i in ("d", "b", "c", "a")], samples=4)
    complexity = {s.ts_id: s.complexity for s in sample}
    assert complexity == pytest.approx({"a": 0.0, "b": 1 / 3, "c": 2 / 3, "d": 1.0})
    reasons = {s.ts_id: s.reason for s in sample}
    assert reasons["a"] == "longest"  # smallest id wins the tie for longest
    assert reasons["d"] == "complexity_max"  # highest composite
    assert reasons["b"] == "complexity_min"  # lowest composite among the unclaimed


def test_complexity_ranks_the_magnitude_of_a_negative_seasonal_correlation() -> None:
    # The record stores a signed ACF; the composite ranks abs(). A strong negative lag-m
    # correlation is as much structure for a seasonal model to fit as a strong positive one, so
    # flipping the sign must not move a series in the complexity ordering.
    def panel(acf: float) -> list[SeriesStats]:
        return [
            _stats("a", n_obs=50, diff_cv=0.1, zero_fraction=0.1, acf_at_season=acf),
            _stats("b", n_obs=50, diff_cv=0.5, zero_fraction=0.5, acf_at_season=0.2),
            _stats("c", n_obs=60, diff_cv=0.9, zero_fraction=0.9, acf_at_season=0.3),
        ]

    positive = {s.ts_id: s.complexity for s in select_profile_sample(panel(0.9), samples=3)}
    negative = {s.ts_id: s.complexity for s in select_profile_sample(panel(-0.9), samples=3)}
    flat = {s.ts_id: s.complexity for s in select_profile_sample(panel(0.0), samples=3)}
    assert positive == negative  # sign is irrelevant to the ranking
    # ...but magnitude is not: a strong seasonal signal lifts an otherwise-calm series, and it
    # lifts it by exactly as much whichever way the correlation points.
    assert negative["a"] > flat["a"]
    assert positive["a"] == negative["a"]


def test_non_finite_stat_does_not_crash_or_reorder_the_sample() -> None:
    # A Spark aggregation over an all-null series returns NaN readily, and a NaN sort key produces
    # an order that varies with the pandas/numpy version. Non-finite values coerce to 0.0 — the
    # sample must therefore match the one built from an explicitly-zeroed input.
    poisoned = [
        _stats("a", n_obs=10, diff_cv=float("nan"), acf_at_season=float("inf")),
        _stats("b", n_obs=20, diff_cv=2.0, acf_at_season=0.4),
        _stats("c", n_obs=30, diff_cv=float("-inf"), zero_fraction=float("nan")),
        _stats("d", n_obs=40, diff_cv=1.0, acf_at_season=-0.8),
    ]
    zeroed = [
        _stats("a", n_obs=10, diff_cv=0.0, acf_at_season=0.0),
        _stats("b", n_obs=20, diff_cv=2.0, acf_at_season=0.4),
        _stats("c", n_obs=30, diff_cv=0.0, zero_fraction=0.0),
        _stats("d", n_obs=40, diff_cv=1.0, acf_at_season=-0.8),
    ]
    sample = select_profile_sample(poisoned, samples=4)
    assert [s.ts_id for s in sample] == [s.ts_id for s in select_profile_sample(zeroed, samples=4)]
    assert all(math.isfinite(s.complexity) for s in sample)


# --- select_profile_sample: coverage and budget ---------------------------------


def test_sample_spans_length_strata_rather_than_one_bucket() -> None:
    # The prototype's actual defect: a small budget exhausted in one length bucket, so the fleet
    # was sized off four near-identical series. A budget of 4 must reach four different bands.
    stats = [_stats(f"s{i:02d}", n_obs=10 * (i + 1)) for i in range(20)]  # 10 .. 200
    sample = select_profile_sample(stats, samples=4)
    assert len({s.stratum for s in sample}) == 4
    assert {s.stratum_label for s in sample} == {"p50", "p90", "p99", "max"}


def test_sample_includes_a_short_violently_intermittent_series() -> None:
    # The complexity axis earning its place. A short, mostly-zero, wildly-varying series can cost
    # more to fit than a long smooth one, and length-only stratification would never reach it.
    stats = [
        _stats("short_spiky", n_obs=10, diff_cv=50.0, zero_fraction=0.9, acf_at_season=-0.95),
        _stats("short_calm_a", n_obs=12, diff_cv=0.1, zero_fraction=0.0, acf_at_season=0.0),
        _stats("short_calm_b", n_obs=14, diff_cv=0.2, zero_fraction=0.0, acf_at_season=0.05),
        _stats("mid_a", n_obs=100, diff_cv=0.3, zero_fraction=0.0, acf_at_season=0.1),
        _stats("mid_b", n_obs=110, diff_cv=0.4, zero_fraction=0.0, acf_at_season=0.1),
        _stats("long", n_obs=120, diff_cv=0.5, zero_fraction=0.0, acf_at_season=0.2),
    ]
    sample = select_profile_sample(stats, samples=3)
    picked = {s.ts_id: s.reason for s in sample}
    assert picked["short_spiky"] == "complexity_max"
    assert picked["long"] == "longest"


def test_complexity_phase_picks_the_extremes_not_the_middle() -> None:
    # Inside one stratum the point is spread, not a representative average: the cheapest and the
    # most expensive series bracket the cost, a median one tells you nothing about either bound.
    stats = [
        _stats("calm", n_obs=50, diff_cv=0.0, zero_fraction=0.0, acf_at_season=0.0),
        _stats("middling", n_obs=50, diff_cv=1.0, zero_fraction=0.5, acf_at_season=0.5),
        _stats("wild", n_obs=50, diff_cv=99.0, zero_fraction=0.99, acf_at_season=-0.99),
        _stats("aaa_longest", n_obs=90, diff_cv=1.0, zero_fraction=0.5, acf_at_season=0.5),
    ]
    sample = select_profile_sample(stats, samples=3)
    picked = {s.ts_id: s.reason for s in sample}
    assert picked["aaa_longest"] == "longest"
    assert picked["wild"] == "complexity_max"
    assert picked["calm"] == "complexity_min"
    assert "middling" not in picked


def test_sample_size_is_capped_by_the_panel_and_by_the_hard_ceiling() -> None:
    # The pre-pass is samples x models real fits on the driver. An operator asking for 500 has
    # turned a sizing pre-pass into the run, so the ceiling clamps silently — it is a budget cap,
    # not bad input.
    stats = [_stats(f"s{i:03d}", n_obs=5 + i) for i in range(200)]
    assert len(select_profile_sample(stats, samples=500)) == profiling._MAX_PROFILE_SAMPLES
    assert len(select_profile_sample(stats, samples=5)) == 5
    assert len(select_profile_sample(stats[:2], samples=5)) == 2


def test_sample_never_repeats_a_series_when_the_panel_is_smaller_than_the_budget() -> None:
    # Duplicating a series to fill the budget would double-weight it in the median and quietly
    # narrow the basis the fleet is sized from.
    stats = [_stats(i, n_obs=n) for i, n in zip("abc", (10, 20, 30), strict=True)]
    sample = select_profile_sample(stats, samples=8)
    assert len(sample) == 3
    assert len({s.ts_id for s in sample}) == 3


def test_all_same_length_panel_collapses_to_one_stratum_without_crashing() -> None:
    # This is the case that produces four identical quantile edges and a rank/(n-1) divide with
    # n == 1. Edges dedupe to a single band and the lowest quantile's label survives.
    stats = [_stats(f"s{i}", n_obs=50) for i in range(5)]
    sample = select_profile_sample(stats, samples=4)
    assert len(sample) == 4
    assert {s.stratum for s in sample} == {0}
    assert {s.stratum_label for s in sample} == {"p50"}


def test_empty_strata_are_skipped_and_the_budget_spills_forward() -> None:
    # A stratum holding only the already-claimed longest series is exhausted on the first pass.
    # The budget must roll into the remaining bands instead of returning a short sample.
    stats = [_stats(i, n_obs=n) for i, n in zip("abcd", (10, 10, 10, 100), strict=True)]
    sample = select_profile_sample(stats, samples=4)
    assert len(sample) == 4
    assert {s.ts_id for s in sample} == {"a", "b", "c", "d"}
    assert sample[0].ts_id == "d"


def test_all_flat_panel_still_yields_a_sample() -> None:
    # Flat series are demoted, not banned. A panel where *every* series is flat is a real (boring)
    # panel, and returning nothing for it would read as "we have no basis" when we do.
    stats = [_stats(f"s{i}", n_obs=20 + i, distinct_values=1) for i in range(4)]
    sample = select_profile_sample(stats, samples=3)
    assert len(sample) == 3
    assert sample[0].reason == "longest"


def test_flat_series_are_demoted_when_non_flat_candidates_exist() -> None:
    # A flat series short-circuits most fits, so sizing a fleet on one under-provisions. It stays
    # eligible for the longest pick and for backfill, but never for a complexity extreme.
    stats = [
        _stats("a", n_obs=10, distinct_values=1),
        _stats("b", n_obs=10, distinct_values=1),
        _stats("c", n_obs=10, distinct_values=5),
        _stats("d", n_obs=12, distinct_values=5),
    ]
    sample = select_profile_sample(stats, samples=3)
    picked = {s.ts_id: s.reason for s in sample}
    assert picked["d"] == "longest"
    assert picked["c"] == "complexity_max"
    assert {ts_id for ts_id, reason in picked.items() if reason.startswith("complexity")} == {"c"}
    assert all(picked.get(flat) in (None, "fill") for flat in ("a", "b"))


def test_backfill_takes_the_longest_unclaimed_series_first() -> None:
    # Once the complexity phase is exhausted the remaining budget still has to buy the most
    # informative evidence left, and that is length — the axis that bounds memory. Taking the
    # shortest leftovers would spend real fits on the series that tell you least.
    stats = [
        _stats("a", n_obs=10, distinct_values=1),
        _stats("b", n_obs=30, distinct_values=1),
        _stats("c", n_obs=20, distinct_values=1),
        _stats("d", n_obs=15),
        _stats("e", n_obs=40),
    ]
    sample = select_profile_sample(stats, samples=4)
    filled = [s.ts_id for s in sample if s.reason == "fill"]
    assert sorted(filled) == ["b", "c"]  # the two longest demoted series, not the 10-row one
    assert "a" not in {s.ts_id for s in sample}


def test_result_is_longest_first_then_ordered_by_stratum_reason_and_id() -> None:
    # The headline invariant is readable off index 0; everything after it is sorted so two audit
    # records of the same sample diff to nothing.
    stats = [
        _stats("a", n_obs=10, distinct_values=1),
        _stats("b", n_obs=30, distinct_values=1),
        _stats("c", n_obs=20, distinct_values=1),
        _stats("d", n_obs=15),
        _stats("e", n_obs=40),
    ]
    sample = select_profile_sample(stats, samples=4)
    assert sample[0].reason == "longest"
    tail = sample[1:]
    ranks = {"complexity_max": 0, "complexity_min": 1, "fill": 2}
    keys = [(s.stratum, ranks[s.reason], s.ts_id) for s in tail]
    assert keys == sorted(keys)
    assert [s.ts_id for s in sample] == ["e", "d", "c", "b"]


def test_sample_of_an_empty_stats_list_is_empty() -> None:
    # Legitimate: an empty panel, or a series_limit that matched nothing. Never a raise.
    assert select_profile_sample([]) == []


def test_panel_below_the_minimum_profilable_length_yields_an_empty_sample() -> None:
    # An empty sample reads as "no basis". A sample of unfittable series would read as "we measured
    # and it cost nothing" — a lie that under-sizes the fleet.
    assert select_profile_sample([_stats(f"s{i}", n_obs=1) for i in range(5)]) == []


def test_a_series_too_short_for_the_models_is_still_sampled() -> None:
    # Minimum history is model policy (horizon, backtest.min_train), not sampler policy. Duplicating
    # that arithmetic here would put it in two places; instead the fit fails and build_profile skips
    # the measurement.
    stats = [_stats("tiny", n_obs=2), _stats("big", n_obs=500)]
    assert {s.ts_id for s in select_profile_sample(stats, samples=2)} == {"tiny", "big"}


@pytest.mark.parametrize("samples", [0, -1, -100])
def test_samples_below_one_raises_valueerror(samples: int) -> None:
    # A caller-argument violation, not data. Returning [] would present "no basis" as "nothing to
    # profile" — two very different things to a consumer deciding whether to fall back.
    with pytest.raises(ValueError, match="samples"):
        select_profile_sample([_stats("a", n_obs=10)], samples=samples)


def test_duplicate_ts_id_in_stats_raises_valueerror_naming_it() -> None:
    # series_stats cannot produce this, so it means two stat sources were merged. Silently deduping
    # would shrink the real budget and measure one series twice, biasing the median.
    stats = [_stats("a", n_obs=10), _stats("b", n_obs=20), _stats("a", n_obs=30)]
    with pytest.raises(ValueError, match="a"):
        select_profile_sample(stats)


def test_sample_carries_the_stats_it_was_chosen_from() -> None:
    # A ts_id alone does not explain a sizing decision six weeks later. The audit record has to
    # carry the numbers the choice was made on.
    stat = _stats("only", n_obs=40, zero_fraction=0.25)
    (spec,) = select_profile_sample([stat], samples=1)
    assert spec.stats == stat
    assert spec.n_obs == 40
    assert 0.0 <= spec.complexity <= 1.0
    assert spec.stratum_label in profiling._LENGTH_LABELS


# --- MeasuredFit.effective_cores (constructed records, no fit) -------------------


def test_effective_cores_floors_at_one_and_detects_threading() -> None:
    # The whole reason to measure rather than declare: cpu_s / wall_s answers "does this model use
    # more than one core" by observation. A single-threaded fit lands near 1.0; a threaded one above
    # it. Below 1.0 is floored, because nothing can be scheduled on less than one core.
    assert _fit(wall_s=2.0, cpu_s=2.0).effective_cores == pytest.approx(1.0)
    assert _fit(wall_s=2.0, cpu_s=7.0).effective_cores == pytest.approx(3.5)
    assert _fit(wall_s=2.0, cpu_s=0.5).effective_cores == pytest.approx(1.0)


def test_effective_cores_of_an_untimeable_fit_is_one_not_a_division_error() -> None:
    # A trivially fast statistical fit can genuinely read 0.0 wall seconds. No ZeroDivisionError and
    # no absurd ratio — the reading is simply meaningless below the clock's resolution.
    assert _fit(wall_s=0.0, cpu_s=0.0).effective_cores == profiling._MIN_EFFECTIVE_CORES
    assert _fit(wall_s=0.0, cpu_s=5.0).effective_cores == profiling._MIN_EFFECTIVE_CORES
    assert _fit(wall_s=1e-12, cpu_s=1.0).effective_cores == profiling._MIN_EFFECTIVE_CORES


# --- build_profile: the two tails ------------------------------------------------


def test_profile_uses_max_for_memory_and_median_for_time() -> None:
    # The mistake this split exists to prevent. The set is deliberately skewed so max and median are
    # far apart: if memory took the median the executor would OOM on the 8 GiB fit, and if time took
    # the max the fleet would be sized for a 100 s outlier that happens once.
    measurements = [
        _fit(ts_id="a", peak_rss_bytes=1 * _GIB, wall_s=1.0),
        _fit(ts_id="b", peak_rss_bytes=1 * _GIB, wall_s=1.0),
        _fit(ts_id="c", peak_rss_bytes=8 * _GIB, wall_s=100.0),
    ]
    cost = build_profile(measurements).models["theta"]
    assert cost.max_peak_rss_bytes == 8 * _GIB  # safety: the worst case, not the typical one
    assert cost.median_wall_s == pytest.approx(1.0)  # throughput: the typical one, not the worst
    assert cost.max_peak_rss_bytes != 1 * _GIB
    assert cost.median_wall_s != pytest.approx(100.0)


def test_profile_rolls_family_peaks_up_as_the_max_across_its_models() -> None:
    # One slot has to hold whichever model of the family lands in it, so the family peak is the max
    # over its models — never their average or their median.
    measurements = [
        _fit("theta", peak_rss_bytes=1 * _GIB, peak_gpu_bytes=None),
        _fit("holtwinters", peak_rss_bytes=6 * _GIB, peak_gpu_bytes=None),
        _fit("arima", peak_rss_bytes=2 * _GIB, peak_gpu_bytes=None),
    ]
    family = build_profile(measurements).for_family("statistical")
    assert family is not None
    assert family.max_peak_rss_bytes == 6 * _GIB
    assert family.models == ("arima", "holtwinters", "theta")  # sorted, for a stable audit diff


def test_profile_family_total_time_sums_its_models_and_median_does_not() -> None:
    # Two different questions, and one field cannot answer both: "what does a typical cell of this
    # family cost" (median) versus "what does running the whole family for one series cost" (sum).
    measurements = [
        _fit("m1", wall_s=1.0),
        _fit("m2", wall_s=2.0),
        _fit("m3", wall_s=6.0),
    ]
    family = build_profile(measurements).for_family("statistical")
    assert family is not None
    assert family.median_wall_s == pytest.approx(2.0)
    assert family.total_wall_s_per_series == pytest.approx(9.0)


def test_the_two_headline_questions_answer_in_one_expression_each() -> None:
    # The seam is at the right altitude only if a consumer reads a byte count and a second count
    # straight off the record — no walking, no re-derivation, no runtime vocabulary.
    profile = build_profile(
        [
            _fit("np", family="deep_learning", peak_rss_bytes=3 * _GIB, wall_s=30.0),
            _fit("theta", family="statistical", peak_rss_bytes=1 * _GIB, wall_s=0.5),
        ],
        memory_margin=2.0,
    )
    assert profile.for_family("deep_learning").slot_rss_bytes == 6 * _GIB
    assert profile.for_family("statistical").median_wall_s == pytest.approx(0.5)


# --- build_profile: failure and absence ------------------------------------------


def test_profile_skips_failed_measurements_but_counts_them() -> None:
    # A flaky probe widens sizing to nominal; it never sinks the run and never poisons the numbers.
    # But "we sized off 1 of 2 fits" has to stay visible, or a thin basis looks like a solid one.
    measurements = [
        _fit(ts_id="good", peak_rss_bytes=2 * _GIB, wall_s=4.0),
        _fit(ts_id="bad", peak_rss_bytes=99 * _GIB, wall_s=999.0, ok=False, error="boom"),
    ]
    profile = build_profile(measurements)
    cost = profile.models["theta"]
    assert cost.max_peak_rss_bytes == 2 * _GIB  # the failed fit's 99 GiB never reaches the slot
    assert cost.median_wall_s == pytest.approx(4.0)
    assert (cost.n_fits, cost.n_ok) == (2, 1)
    assert (profile.n_measurements, profile.n_ok, profile.n_failed) == (2, 1, 1)
    assert profile.sample_ts_ids == ("good",)  # only ids that actually backed a number


def test_unmeasured_gpu_axis_is_none_not_zero() -> None:
    # The profile runs on the driver at submit time, where there is usually no accelerator. A 0 here
    # would let a consumer compute a minimum GPU fraction and pack ten tasks onto a device that fits
    # two. A measured 0 (the model allocated nothing) reaches the same destination honestly.
    profile = build_profile(
        [
            _fit("np", family="deep_learning", peak_gpu_bytes=None),
            _fit("np", family="deep_learning", peak_gpu_bytes=0),
        ]
    )
    cost = profile.models["np"]
    assert cost.max_peak_gpu_bytes is None
    family = profile.for_family(cost.family)
    assert family is not None
    assert family.max_peak_gpu_bytes is None
    assert family.slot_gpu_bytes is None


def test_all_zero_rss_collapses_to_none_rather_than_sizing_at_zero() -> None:
    # ru_maxrss is a monotonic process-wide high-water mark, so a fit that needs less than an
    # earlier one reports 0. That is "no evidence", not "free" — never size an executor at 0 bytes.
    profile = build_profile(
        [_fit(peak_rss_bytes=0, wall_s=2.0), _fit(peak_rss_bytes=0, wall_s=3.0)]
    )
    cost = profile.models["theta"]
    assert cost.max_peak_rss_bytes is None
    assert cost.median_wall_s == pytest.approx(2.5)  # the time axis is still perfectly good
    assert profile.for_family("statistical").slot_rss_bytes is None


def test_a_measurement_can_be_usable_for_time_and_unusable_for_memory() -> None:
    # Usability is decided per axis, not per record. Filtering whole records throws away the good
    # time reading; trusting whole records fabricates the missing memory one.
    profile = build_profile([_fit(peak_rss_bytes=0, peak_gpu_bytes=None, wall_s=5.0, cpu_s=9.0)])
    cost = profile.models["theta"]
    assert cost.max_peak_rss_bytes is None
    assert cost.max_peak_gpu_bytes is None
    assert cost.median_wall_s == pytest.approx(5.0)
    assert cost.max_effective_cores == pytest.approx(1.8)


def test_negative_or_non_finite_injected_values_are_skipped_per_axis() -> None:
    # Injected numbers come from tests today and a Spark aggregation later. A poisoned axis must
    # collapse to "no basis" rather than propagate an inf or a NaN into a resource request.
    poisoned = [
        _fit(ts_id="a", peak_rss_bytes=-5, wall_s=float("nan"), cpu_s=float("inf")),
        _fit(ts_id="b", peak_rss_bytes=2 * _GIB, wall_s=float("inf"), cpu_s=1.0),
    ]
    cost = build_profile(poisoned).models["theta"]
    assert cost.max_peak_rss_bytes == 2 * _GIB  # the -5 never enters the max
    assert cost.median_wall_s is None  # NaN and inf are both no evidence
    assert cost.median_cpu_s == pytest.approx(1.0)
    assert cost.max_effective_cores is None  # no fit had a timeable wall clock


def test_family_with_no_usable_evidence_is_absent_not_zeroed() -> None:
    # Absence is the signal to fall back to static config. A FamilyCost of zeros would be consumed
    # as a real size, which is how a fleet ends up requesting nothing and OOMing.
    profile = build_profile(
        [
            _fit("theta", family="statistical", peak_rss_bytes=_GIB, wall_s=1.0),
            _fit("np", family="deep_learning", peak_rss_bytes=0, wall_s=0.0, cpu_s=0.0),
        ]
    )
    assert "statistical" in profile.families
    assert "deep_learning" not in profile.families
    assert profile.for_family("deep_learning") is None
    assert "np" not in profile.models  # counted, but nothing usable came out of it
    assert profile.n_measurements == 2  # the audit still shows the fit happened


def test_empty_or_all_failed_measurements_yield_an_empty_profile() -> None:
    # "Nothing measured" is a first-class state the caller handles by staying on static config —
    # never a raise, and never an empty-looking profile that hides how much was attempted.
    empty = build_profile([])
    assert empty.is_empty is True
    assert empty.families == {} and empty.models == {}
    assert (empty.n_measurements, empty.n_ok, empty.n_failed) == (0, 0, 0)
    assert empty.sample_ts_ids == ()

    all_failed = build_profile([_fit(ts_id=f"s{i}", ok=False, error="boom") for i in range(8)])
    assert all_failed.is_empty is True
    assert (all_failed.n_measurements, all_failed.n_ok, all_failed.n_failed) == (8, 0, 8)


# --- build_profile: margins, ordering, serialization -----------------------------


def test_slot_values_apply_each_margin_exactly_once() -> None:
    # Raw and sized travel together on one record precisely so a margin cannot be applied twice on
    # the way through — an audit reads "measured 1 GiB" and "sized 1.5 GiB" off the same object.
    profile = build_profile(
        [_fit(peak_rss_bytes=1024, peak_gpu_bytes=2048, wall_s=2.0, cpu_s=2.0)],
        memory_margin=1.5,
        time_margin=1.5,
    )
    family = profile.for_family("statistical")
    assert family is not None
    assert family.max_peak_rss_bytes == 1024  # raw, no margin
    assert family.slot_rss_bytes == 1536  # 1024 x 1.5, once — not 2304
    assert family.slot_gpu_bytes == 3072  # 2048 x 1.5, once
    assert family.planning_wall_s == pytest.approx(3.0)  # 2.0 x 1.5, once
    assert family.planning_total_wall_s_per_series == pytest.approx(3.0)
    assert family.slot_cores == 1  # no margin: a core count is already discrete


def test_slot_cores_rounds_a_threaded_fit_up_to_whole_cores() -> None:
    # A slot is bought in whole cores. Under-requesting oversubscribes the box, so this rounds up
    # and never below one.
    profile = build_profile([_fit(wall_s=2.0, cpu_s=6.5), _fit(wall_s=2.0, cpu_s=1.0)])
    family = profile.for_family("statistical")
    assert family is not None
    assert family.max_effective_cores == pytest.approx(3.25)  # max, not median: a peak-like axis
    assert family.slot_cores == 4


def test_derived_properties_are_none_exactly_when_their_raw_basis_is_none() -> None:
    # The invariant a consumer relies on: None means "fall back to your own static default", and
    # there is no fabricated number anywhere in between.
    profile = build_profile(
        [_fit(peak_rss_bytes=0, peak_gpu_bytes=None, wall_s=4.0, cpu_s=4.0)],
        memory_margin=2.0,
    )
    family = profile.for_family("statistical")
    assert family is not None
    pairs = [
        (family.max_peak_rss_bytes, family.slot_rss_bytes),
        (family.max_peak_gpu_bytes, family.slot_gpu_bytes),
        (family.max_effective_cores, family.slot_cores),
        (family.median_wall_s, family.planning_wall_s),
        (family.total_wall_s_per_series, family.planning_total_wall_s_per_series),
    ]
    for raw, derived in pairs:
        assert (raw is None) == (derived is None)
    assert family.slot_rss_bytes is None  # the RSS axis had no evidence
    assert family.planning_wall_s == pytest.approx(4.0 * profiling._DEFAULT_TIME_MARGIN)


@pytest.mark.parametrize("margin", [0.9, 0.0, -1.0, float("inf"), float("nan")])
def test_margin_below_one_raises_valueerror(margin: float) -> None:
    # A margin below 1 asks for less headroom than the measurement — never valid, and cheap to
    # catch offline before it becomes an under-sized executor.
    with pytest.raises(ValueError, match="margin"):
        build_profile([_fit()], memory_margin=margin)
    with pytest.raises(ValueError, match="margin"):
        build_profile([_fit()], time_margin=margin)


def test_a_margin_of_exactly_one_is_allowed_and_is_a_no_op() -> None:
    # 1.0 is "size exactly what was measured" — unwise, but coherent, and the boundary has to be
    # inclusive or an operator who wants no headroom cannot express it.
    family = build_profile(
        [_fit(peak_rss_bytes=4096, wall_s=3.0)], memory_margin=1.0, time_margin=1.0
    ).for_family("statistical")
    assert family is not None
    assert family.slot_rss_bytes == 4096
    assert family.planning_wall_s == pytest.approx(3.0)


def test_default_margins_are_recorded_on_the_profile_it_built() -> None:
    # A sizing decision that cannot be re-derived from its own record is not auditable, so the
    # margins ride along with the numbers rather than living only in the caller.
    profile = build_profile([_fit()])
    assert profile.memory_margin == profiling._DEFAULT_MEMORY_MARGIN
    assert profile.time_margin == profiling._DEFAULT_TIME_MARGIN
    family = profile.for_family("statistical")
    assert family is not None
    assert (family.memory_margin, family.time_margin) == (
        profiling._DEFAULT_MEMORY_MARGIN,
        profiling._DEFAULT_TIME_MARGIN,
    )


def test_single_measurement_makes_max_equal_median() -> None:
    # A thin basis is allowed, but the audit has to show how thin: one fit, both tails identical.
    cost = build_profile([_fit(n_obs=365, peak_rss_bytes=3 * _GIB, wall_s=7.0)]).models["theta"]
    assert cost.max_peak_rss_bytes == 3 * _GIB
    assert cost.median_wall_s == pytest.approx(7.0)
    assert (cost.n_fits, cost.n_ok, cost.max_n_obs) == (1, 1, 365)


def test_an_even_number_of_values_medians_the_two_middles() -> None:
    # statistics.median averages the two middle values, which is deterministic and order-free —
    # a "pick the lower middle" rule would make the result depend on the input sequence.
    cost = build_profile([_fit(wall_s=w) for w in (1.0, 2.0, 3.0, 10.0)]).models["theta"]
    assert cost.median_wall_s == pytest.approx(2.5)


def test_repeat_measurements_of_the_same_series_both_contribute() -> None:
    # A legitimate repeat measurement of one (ts_id, model) is evidence, not a duplicate to dedupe.
    cost = build_profile(
        [_fit(ts_id="a", wall_s=2.0), _fit(ts_id="a", wall_s=4.0)]
    ).models["theta"]
    assert cost.n_fits == 2
    assert cost.median_wall_s == pytest.approx(3.0)


def test_a_fit_that_used_no_measurable_cpu_time_still_reports_one_core() -> None:
    # cpu_s == 0 with a real wall clock is a fit too cheap to register on process_time, not a fit
    # that needs zero cores. The floor makes that unrepresentable.
    cost = build_profile([_fit(wall_s=2.0, cpu_s=0.0)]).models["theta"]
    assert cost.max_effective_cores == pytest.approx(1.0)
    assert cost.median_cpu_s is None  # zero seconds of CPU is not evidence about CPU time


def test_build_profile_does_not_mutate_its_input() -> None:
    # The pre-pass hands the same measurement list to the profile builder and to telemetry. A
    # sort-in-place or a pop here would change what the audit record says was measured.
    measurements = [_fit("theta", ts_id="b", wall_s=2.0), _fit("arima", ts_id="a", wall_s=1.0)]
    snapshot = list(measurements)
    build_profile(measurements)
    assert measurements == snapshot
    assert [m.ts_id for m in measurements] == ["b", "a"]


def test_a_family_keeps_only_the_models_that_produced_usable_evidence() -> None:
    # One model of a family can be pure noise while another is solid. The family stays — sized off
    # the model that gave a number — and the useless one simply is not listed as a contributor.
    profile = build_profile(
        [
            _fit("theta", family="statistical", peak_rss_bytes=2 * _GIB, wall_s=3.0),
            _fit("arima", family="statistical", peak_rss_bytes=0, wall_s=0.0, cpu_s=0.0),
        ]
    )
    family = profile.for_family("statistical")
    assert family is not None
    assert family.models == ("theta",)
    assert family.max_peak_rss_bytes == 2 * _GIB
    assert "arima" not in profile.models
    assert profile.n_measurements == 2  # the attempt is still on the record


def test_profile_reports_only_the_models_that_were_measured() -> None:
    # The profile reports what was measured; deciding whether that coverage is good enough belongs
    # to the consumer, not here.
    profile = build_profile([_fit("theta")])
    assert list(profile.models) == ["theta"]


def test_profile_is_independent_of_measurement_order() -> None:
    # A pure function of the measurement *set*. Anything else means two identical pre-passes could
    # size two different fleets depending on which fit finished first.
    measurements = [
        _fit("theta", ts_id="a", peak_rss_bytes=1 * _GIB, wall_s=1.0, cpu_s=2.0),
        _fit("theta", ts_id="b", peak_rss_bytes=4 * _GIB, wall_s=3.0, cpu_s=3.0),
        _fit("xgboost", family="ml", ts_id="a", peak_rss_bytes=2 * _GIB, wall_s=8.0),
        _fit("np", family="deep_learning", ts_id="b", peak_gpu_bytes=5 * _GIB, wall_s=40.0),
        _fit("np", family="deep_learning", ts_id="c", ok=False, error="oom"),
    ]
    shuffled = list(measurements)
    random.Random(5).shuffle(shuffled)
    assert build_profile(shuffled) == build_profile(measurements)
    # Dict equality ignores key order, so compare the serialized form too: telemetry is diffed
    # byte-for-byte and an insertion-order drift would be invisible to == alone.
    assert json.dumps(build_profile(shuffled).to_dict()) == json.dumps(
        build_profile(measurements).to_dict()
    )


def test_profile_orders_families_and_models_deterministically() -> None:
    # The profile is stamped into run telemetry, so byte-stable JSON is what makes an audit diff
    # meaningful. Insertion order of a dict is its serialization order.
    measurements = [
        _fit("theta", family="statistical"),
        _fit("np", family="deep_learning"),
        _fit("xgboost", family="ml"),
        _fit("arima", family="statistical"),
    ]
    profile = build_profile(measurements)
    assert list(profile.models) == ["arima", "np", "theta", "xgboost"]
    assert list(profile.families) == ["deep_learning", "ml", "statistical"]
    assert list(profile.to_dict()["models"]) == ["arima", "np", "theta", "xgboost"]
    assert list(profile.to_dict()["families"]) == ["deep_learning", "ml", "statistical"]


def test_profile_is_json_serializable_for_telemetry() -> None:
    # The whole seam this module owes the telemetry work: json.dumps with no custom encoder, so no
    # numpy scalar, pd.Timestamp or dataclass instance may leak into the record.
    profile = build_profile(
        [
            _fit("theta", ts_id="a", peak_rss_bytes=_GIB, wall_s=1.0),
            # The deep-learning fits report a GPU peak but no host-RSS evidence (the high-water
            # mark had already been raised), so the family must serialize one axis and null the
            # other rather than filling the gap.
            _fit(
                "np",
                family="deep_learning",
                ts_id="b",
                peak_rss_bytes=0,
                peak_gpu_bytes=2 * _GIB,
                wall_s=30.0,
            ),
            _fit("np", family="deep_learning", ts_id="c", ok=False, error="oom"),
        ],
        memory_margin=1.5,
        time_margin=1.5,
    )
    payload = json.loads(json.dumps(profile.to_dict()))

    assert payload["n_measurements"] == 3
    assert payload["n_ok"] == 2
    assert payload["n_failed"] == 1
    assert payload["sample_ts_ids"] == ["a", "b"]
    assert payload["memory_margin"] == 1.5
    assert set(payload["models"]["theta"]) == {
        "model_type",
        "family",
        "n_fits",
        "n_ok",
        "max_n_obs",
        "max_peak_rss_bytes",
        "max_process_rss_bytes",
        "max_peak_gpu_bytes",
        "median_wall_s",
        "median_cpu_s",
        "max_effective_cores",
    }
    assert set(payload["families"]["deep_learning"]) == {
        "family",
        "models",
        "n_fits",
        "n_ok",
        "max_peak_rss_bytes",
        "max_process_rss_bytes",
        "max_peak_gpu_bytes",
        "max_effective_cores",
        "median_wall_s",
        "total_wall_s_per_series",
        "memory_margin",
        "time_margin",
        "slot_rss_bytes",
        "slot_gpu_bytes",
        "slot_cores",
        "planning_wall_s",
        "planning_total_wall_s_per_series",
    }
    # Raw and sized both present, and an unmeasured axis survives the round trip as null.
    assert payload["families"]["deep_learning"]["max_peak_gpu_bytes"] == 2 * _GIB
    assert payload["families"]["deep_learning"]["slot_gpu_bytes"] == 3 * _GIB
    assert payload["families"]["deep_learning"]["max_peak_rss_bytes"] is None
    assert payload["families"]["deep_learning"]["slot_rss_bytes"] is None


def test_empty_profile_is_also_json_serializable() -> None:
    # The fall-back state is stamped into telemetry too — "we measured nothing" is a result worth
    # recording, not a reason to skip the record.
    assert json.loads(json.dumps(build_profile([]).to_dict()))["models"] == {}


# --- end-to-end over the pure seam ----------------------------------------------


def test_panel_to_sample_to_profile_composes_without_a_fit() -> None:
    # The composition the caller actually performs, minus the one I/O step: stats -> sample ->
    # (measure_fit, live only) -> profile. Proves the seam is injectable, which is the reason the
    # module is split this way at all.
    rng = random.Random(4)
    panel = _panel(
        *[
            _frame(f"s{i:02d}", [rng.random() * (i + 1) for _ in range(10 + i * 9)])
            for i in range(12)
        ]
    )
    stats = series_stats(panel, _cfg())
    sample = select_profile_sample(stats, samples=5)
    assert len(sample) == 5
    assert sample[0].ts_id == "s11"  # the longest series, first

    # Stand in for measure_fit with injected numbers, exactly as the offline tests above do.
    measurements = [
        _fit("theta", ts_id=spec.ts_id, n_obs=spec.n_obs, wall_s=1.0 + spec.n_obs / 100)
        for spec in sample
    ]
    profile = build_profile(measurements)
    assert profile.is_empty is False
    assert profile.sample_ts_ids == tuple(sorted(spec.ts_id for spec in sample))
    assert profile.models["theta"].max_n_obs == sample[0].n_obs


# --- regressions: what a review found after the first implementation --------------
#
# Every test below pins a defect that a code review caught and that the original suite could not
# have caught, because each one hides in a case the happy path never reaches: out-of-contract
# input, floating-point noise, a platform artefact, or a number that is only wrong in telemetry.


def test_stats_are_reproducible_when_timestamps_tie() -> None:
    # Was: the sort was stable on (ts_id, ds) only, so rows sharing a timestamp kept panel order —
    # and diff_cv/acf_at_season are computed over that order. Duplicate (ts_id, timestamp) rows are
    # out of contract (validate_panel rejects them) but reach here anyway from a restatement or a
    # re-landed partition, and no engine promises row order. Two reads of one table could size two
    # different fleets.
    rows = _frame("a", [1.0, 9.0, 2.0, 8.0], dates=["2024-01-01"] * 3 + ["2024-01-02"])
    forward = series_stats(rows, _cfg())
    reversed_rows = rows.iloc[::-1].reset_index(drop=True)
    assert series_stats(reversed_rows, _cfg()) == forward


def test_stats_are_reproducible_when_no_timestamp_parses() -> None:
    # The broader half of the same defect: when the whole date column fails to parse, every row is
    # NaT, so the *entire* series ordering — not just a tie — falls back to arrival order.
    values = [1.0, 50.0, 2.0, 49.0, 3.0, 48.0, 4.0, 47.0]
    rows = _frame("a", values, dates=["2024/13/45"] * len(values))
    forward = series_stats(rows, _cfg())
    shuffled = rows.sample(frac=1.0, random_state=7).reset_index(drop=True)
    assert series_stats(shuffled, _cfg()) == forward
    # n_obs is unaffected by any of this — it is a row count, and it is the axis bounding memory.
    assert forward[0].n_obs == len(values)


def test_two_lagged_pairs_are_not_evidence_of_seasonality() -> None:
    # Was: `estimable` required only 2 lagged pairs, but a Pearson correlation over two points is
    # +/-1 by construction. Every series with exactly season+2 observations therefore reported
    # maximum seasonal strength regardless of content — and the sampler ranks that magnitude, so
    # those series were preferentially picked as their stratum's complexity extreme.
    period = 7  # daily freq
    nine_rows = [1.0, 5.0, 2.0, 9.0, 3.0, 7.0, 4.0, 100.0, 6.0]
    at_two_pairs = series_stats(_frame("a", nine_rows), _cfg())
    assert at_two_pairs[0].n_obs == period + 2
    assert at_two_pairs[0].acf_at_season == 0.0

    # Three pairs is a real (if small) estimate, so the axis still switches on with enough data.
    seasonal = [float(v) for v in [1, 2, 3, 4, 5, 6, 7] * 2] + [1.0, 2.0, 3.0]
    assert series_stats(_frame("a", seasonal), _cfg())[0].acf_at_season > 0.9


def test_seasonality_survives_a_large_baseline() -> None:
    # Was: the one-pass form n*sum(x^2) - (sum x)^2 subtracts two nearly equal large numbers, so it
    # lost all significance once the coefficient of variation fell below ~sqrt(eps). A revenue line
    # near 1e9 that moves by tens is exactly that regime, and the failure was not a NaN the guard
    # would catch — it was a plausible wrong number decided by rounding, used as a sort key.
    seasonal = [float(v) for v in [1, 2, 3, 4, 5, 6, 7] * 6]
    plain = series_stats(_frame("a", seasonal), _cfg())[0].acf_at_season
    offset = series_stats(_frame("a", [v + 1e9 for v in seasonal]), _cfg())[0].acf_at_season
    # Adding a constant to every value cannot change a correlation.
    assert offset == pytest.approx(plain, abs=1e-6)
    assert offset > 0.9


def test_a_single_threaded_fit_does_not_round_up_to_two_cores() -> None:
    # Was: slot_cores was an unguarded ceil() over a ratio of two clocks. For a genuinely
    # single-threaded fit the two differ by scheduling jitter and which lands higher is a coin
    # flip, so ceil(1.0000005) requested 2 cores and halved fleet density — with an audit record
    # reading 1.0000005, which looks right to two decimal places.
    noise = build_profile([_fit(wall_s=1.0, cpu_s=1.0000005)])
    assert noise.families["statistical"].max_effective_cores == pytest.approx(1.0000005)
    assert noise.families["statistical"].slot_cores == 1

    # A real second thread is still counted — the snap is tolerance, not truncation.
    assert build_profile([_fit(wall_s=1.0, cpu_s=1.2)]).families["statistical"].slot_cores == 2
    assert build_profile([_fit(wall_s=1.0, cpu_s=3.4)]).families["statistical"].slot_cores == 4


def test_a_family_counts_the_fits_of_a_model_that_produced_nothing() -> None:
    # Was: family n_fits/n_ok were summed over the surviving ModelCosts, so a model whose every fit
    # failed vanished from the counters. A family that was 2-of-4 reported a clean 2-of-2 — not a
    # missing number but a wrong one, and the deep-learning member that OOM'd on every fit is
    # exactly the signal that the slot must be bigger.
    profile = build_profile(
        [
            _fit("nhits", family="deep_learning", peak_gpu_bytes=2 * _GIB),
            _fit("nhits", family="deep_learning", peak_gpu_bytes=2 * _GIB),
            _fit("nbeats", family="deep_learning", ok=False, error="CUDA out of memory"),
            _fit("nbeats", family="deep_learning", ok=False, error="CUDA out of memory"),
        ]
    )
    family = profile.families["deep_learning"]
    assert family.models == ("nhits",)  # only nhits produced a usable number
    assert (family.n_fits, family.n_ok) == (4, 2)  # but all four fits are on the record


def test_a_dropped_model_and_its_error_survive_into_the_profile() -> None:
    # The other half: absence must not be silent. Without this the profile can say a family was
    # sized off one model, but never which model it lost or why — and the reason is the only thing
    # that tells an operator whether to re-run or to raise the slot.
    profile = build_profile(
        [
            _fit("nhits", family="deep_learning", peak_gpu_bytes=2 * _GIB),
            _fit("nbeats", family="deep_learning", ok=False, error="CUDA out of memory"),
        ]
    )
    assert profile.dropped_models == ("nbeats",)
    assert profile.first_error_by_model == {"nbeats": "CUDA out of memory"}
    blob = profile.to_dict()
    assert blob["dropped_models"] == ["nbeats"]
    assert "CUDA out of memory" in blob["first_error_by_model"]["nbeats"]


def test_the_sample_explains_itself_in_telemetry() -> None:
    # Was: the sample reached telemetry as a bare list of ids. `reason` and `stats` are on the
    # contract precisely because an id alone does not explain a sizing decision six weeks later,
    # and every field carrying that explanation was dropped at the serialization boundary.
    stats = [_stats("a", 10), _stats("b", 40, diff_cv=9.0), _stats("c", 90, zero_fraction=0.8)]
    sample = select_profile_sample(stats, samples=3)
    profile = build_profile([_fit(ts_id=spec.ts_id) for spec in sample], sample=sample)

    blob = profile.to_dict()
    json.dumps(blob)  # must survive telemetry with no custom encoder
    assert [entry["ts_id"] for entry in blob["sample"]] == [spec.ts_id for spec in sample]
    first = blob["sample"][0]
    assert first["reason"] == "longest"
    # The panel properties the sizing rests on travel with the pick, not just the id.
    assert set(first) >= {"stratum_label", "complexity", "zero_fraction", "diff_cv", "n_exog"}

    # And it stays optional: a caller that does not pass a sample gets an empty one, not a crash.
    assert build_profile([_fit()]).to_dict()["sample"] == []


def test_a_measurement_records_how_it_was_taken() -> None:
    # Both host axes are otherwise measurements of the machine, not the model: ru_maxrss is
    # monotonic (so an unreset mark makes a fit's memory read as whatever earlier fits left) and an
    # unpinned fit takes every idle core (so cpu_s/wall_s reads as the driver's nproc). What the
    # probe achieved has to be on the record, because both attempts are platform-dependent.
    unknown = _fit()
    assert (unknown.intraop_threads, unknown.host_cpu_count, unknown.rss_peak_reset) == (
        None,
        None,
        False,
    )
    pinned = MeasuredFit(
        ts_id="s1",
        model_type="theta",
        family="statistical",
        n_obs=100,
        wall_s=1.0,
        cpu_s=1.0,
        peak_rss_bytes=_GIB,
        peak_gpu_bytes=None,
        ok=True,
        error=None,
        intraop_threads=1,
        host_cpu_count=32,
        rss_peak_reset=True,
    )
    # The provenance fields are metadata: they annotate the ratio, they never alter it.
    assert pinned.effective_cores == 1.0
    assert build_profile([pinned]).families["statistical"].slot_cores == 1


# --- regressions: what live measurement found that constructed records could not ----
#
# The two axes below were each implemented in the obvious way first, passed every unit test
# written against them, and then turned out to be wrong by 17x and by the host's core count
# when a real fit ran through them. These tests pin the *corrected* semantics so the obvious
# implementation cannot come back. They stay offline — what they defend is the arithmetic and
# the plumbing, which is what a constructed record can reach.


def test_the_slot_is_sized_from_the_absolute_footprint_not_the_per_fit_delta():
    """Live measurement: the delta swings 17x on run order, the absolute lands within 0.6%.

    So a slot must be sized from ``process_rss_bytes``. Here the delta is deliberately tiny
    (a warm heap serving a fit from resident pages — measured as 0.00 MB for real models)
    while the true footprint is 4 GiB. Sizing off the delta would ask for 130 MB.
    """
    warm_fit = _fit(peak_rss_bytes=100 * 1024**2, process_rss_bytes=4 * _GIB)
    family = build_profile([warm_fit]).families["statistical"]

    assert family.max_peak_rss_bytes == 100 * 1024**2  # kept, as a diagnostic
    assert family.max_process_rss_bytes == 4 * _GIB
    assert family.slot_rss_bytes == math.ceil(4 * _GIB * family.memory_margin)


def test_a_fit_that_reused_every_page_does_not_size_a_slot_at_zero():
    """The measured 0.00 MB case. ``peak_rss_bytes=0`` is "no evidence", never "it was free"."""
    reused = _fit(peak_rss_bytes=0, process_rss_bytes=3 * _GIB)
    family = build_profile([reused]).families["statistical"]

    assert family.max_peak_rss_bytes is None  # non-positive delta is discarded, not folded in
    assert family.slot_rss_bytes == math.ceil(3 * _GIB * family.memory_margin)


def test_an_unmeasured_absolute_footprint_yields_no_slot_size_rather_than_a_guess():
    """No ``resource`` module means NOT MEASURED. The consumer must fall back, not size on air."""
    unmeasured = _fit(peak_rss_bytes=_GIB, process_rss_bytes=None)
    family = build_profile([unmeasured]).families["statistical"]

    assert family.max_process_rss_bytes is None
    assert family.slot_rss_bytes is None
    assert family.max_peak_rss_bytes == _GIB  # the diagnostic survives on its own


def test_both_memory_axes_reach_telemetry_so_a_reader_can_tell_them_apart():
    """An audit that cannot see both numbers cannot tell an over-size from a bad measurement."""
    payload = build_profile([_fit(peak_rss_bytes=_GIB, process_rss_bytes=4 * _GIB)]).to_dict()
    json.dumps(payload)  # the surface is telemetry; it must serialize with no custom encoder

    assert payload["models"]["theta"]["max_process_rss_bytes"] == 4 * _GIB
    assert payload["models"]["theta"]["max_peak_rss_bytes"] == _GIB
    assert payload["families"]["statistical"]["max_process_rss_bytes"] == 4 * _GIB
    assert payload["families"]["statistical"]["max_peak_rss_bytes"] == _GIB


def test_the_family_slot_holds_the_heaviest_member_on_the_absolute_axis():
    """Peaks take the max across a family — one slot must hold whichever model lands in it."""
    light = _fit("theta", peak_rss_bytes=_GIB, process_rss_bytes=2 * _GIB)
    heavy = _fit("sarimax", peak_rss_bytes=_GIB, process_rss_bytes=6 * _GIB)
    family = build_profile([light, heavy]).families["statistical"]

    assert family.max_process_rss_bytes == 6 * _GIB


def test_the_thread_pin_sets_and_restores_every_native_pool_variable(monkeypatch):
    """The pre-pass runs in the driver, which goes on to do real work.

    Leaving the fleet's BLAS pinned to one thread would be a silent performance regression far
    larger than the pre-pass that caused it — so the variables must be restored exactly, and a
    variable that was previously *unset* must go back to unset rather than to an empty string.
    """
    monkeypatch.setenv("OMP_NUM_THREADS", "8")  # pre-existing: must be restored to "8"
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)  # unset: must go back to unset

    with profiling._pinned_intraop_threads(1):
        inside = {name: os.environ.get(name) for name in profiling._INTRAOP_ENV_VARS}

    assert inside == dict.fromkeys(profiling._INTRAOP_ENV_VARS, "1"), "every pool must be capped"
    assert os.environ["OMP_NUM_THREADS"] == "8"
    assert "MKL_NUM_THREADS" not in os.environ


def test_the_thread_pin_restores_the_environment_even_when_the_fit_raises():
    """A probe fit that raises must not leave the driver single-threaded for the rest of the run."""
    monkeypatch_free_before = os.environ.get("OMP_NUM_THREADS")

    with pytest.raises(RuntimeError, match="fit exploded"):
        with profiling._pinned_intraop_threads(1):
            raise RuntimeError("fit exploded")

    assert os.environ.get("OMP_NUM_THREADS") == monkeypatch_free_before


def test_an_environment_that_cannot_be_pinned_reports_none_rather_than_claiming_a_pin(monkeypatch):
    """Env vars alone leave the already-loaded pool at ``nproc`` — the dominant contamination.

    That is not a pin, so it must not be recorded as one: ``intraop_threads=None`` is what tells
    a reader the ratio is an upper bound contaminated by ``host_cpu_count``.
    """
    import builtins

    real_import = builtins.__import__

    def no_threadpoolctl(name, *args, **kwargs):
        if name == "threadpoolctl":
            raise ImportError("stripped environment")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_threadpoolctl)

    with profiling._pinned_intraop_threads(1) as pinned:
        assert pinned is None, "a half-applied pin is reported as no pin"
        assert os.environ["OMP_NUM_THREADS"] == "1", "the half we can still do, we still do"

    assert "OMP_NUM_THREADS" not in os.environ, "restored even on the degraded path"
