"""Which series to profile, and why — a deterministic function of the panel statistics.

The pre-pass runs ``samples x models`` real fits sequentially on the driver, so the sample
is a budget, and spending it on an alphabetical prefix wastes it: no length spread, no
complexity spread, and the longest series — the one that bounds memory — almost never in
it. Series are stratified by length first (the axis that provably bounds memory) and then
spread across the non-length cost axes, so a small budget still spans the panel.

Selection is deterministic: the same panel and the same budget always pick the same series,
which is what makes a measured profile reproducible and a `SampleSpec` worth recording.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .stats import SeriesStats

# Series to measure in the pre-pass. Three (today's calibration sample size) is an
# alphabetical prefix of the panel: no length spread, no complexity spread, and the longest
# series — the one that bounds memory — almost never in it. Eight is the smallest budget
# that spans four length strata *and* both complexity extremes without the pre-pass cost
# mattering for statistical models.
_DEFAULT_PROFILE_SAMPLES = 8  # series

# Hard ceiling on the budget. The pre-pass is ``samples x models`` real fits, run
# sequentially on the driver; an operator asking for 500 samples has turned a sizing
# pre-pass into the run. A request above this clamps (it is a budget cap, not bad input).
_MAX_PROFILE_SAMPLES = 64  # series

# Rows below which a series cannot be differenced, fit, or meaningfully timed.
_MIN_PROFILABLE_OBS = 2  # rows

# Length quantiles that define the strata, with their labels. Length is the primary
# stratification axis because it is the one that provably bounds memory. Upper-tail
# weighted because the interesting cost lives in the tail, and the 1.00 stratum guarantees
# the longest series is representable.
_LENGTH_QUANTILES = (0.50, 0.90, 0.99, 1.00)
_LENGTH_LABELS = ("p50", "p90", "p99", "max")

# The cost axes that are *not* length: volatility, intermittency, seasonal strength.
# Equal-weighted after rank normalization — no weights to tune, and a weight table would be
# an unmeasured assumption dressed up as a constant.
_COMPLEXITY_AXES = ("diff_cv", "zero_fraction", "acf_at_season")


@dataclass(frozen=True)
class SampleSpec:
    """One series chosen for the profiling pre-pass, with the reason it was chosen (pure).

    The reason is part of the contract, not a debug string. It is what makes a sizing
    decision reviewable after the fact ("this fleet was sized off the longest series and
    the two most intermittent ones"), and it is what a test asserts on instead of asserting
    on an opaque list of ids. ``stats`` is embedded rather than re-looked-up because the
    sample is snapshotted for audit, and a ``ts_id`` alone does not explain a sizing
    decision six weeks later.
    """

    ts_id: str
    n_obs: int
    stratum: int  # 0-based index into the deduped length strata, 0 == shortest band
    stratum_label: str  # "p50" | "p90" | "p99" | "max" — the quantile band it came from
    reason: str  # "longest" | "complexity_max" | "complexity_min" | "fill"
    complexity: float  # [0,1] equal-weighted mean of the rank-normalized _COMPLEXITY_AXES
    stats: SeriesStats

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict of the pick *and* the panel properties behind it.

        Without this the explanation stops at the telemetry boundary: a bare list of ids says
        which series were measured but not why, and "why" is the whole reason `reason` and
        `stats` are on the contract. Plain scalars only, same rule as
        `ComputeProfile.to_dict`.
        """
        return {
            "ts_id": self.ts_id,
            "n_obs": self.n_obs,
            "stratum": self.stratum,
            "stratum_label": self.stratum_label,
            "reason": self.reason,
            "complexity": self.complexity,
            "zero_fraction": self.stats.zero_fraction,
            "distinct_values": self.stats.distinct_values,
            "diff_cv": self.stats.diff_cv,
            "acf_at_season": self.stats.acf_at_season,
            "n_exog": self.stats.n_exog,
        }


def _length_strata(n_obs_values: list[int]) -> list[tuple[int, str]]:
    """The deduped ascending ``(edge, label)`` length bands for a candidate set (pure).

    Edges come from **nearest rank** on the sorted lengths — ``sorted[floor(q * (n - 1))]`` —
    not from an interpolating quantile: no numpy, no dtype-dependent midpoint, so the
    boundaries are exactly reproducible on any platform. The ``1.00`` edge is the longest
    series by construction, which is what makes the top band always representable.

    Duplicate edges collapse, keeping the **lowest** quantile that produced each one so its
    label survives. A panel whose series are all the same length therefore collapses to a
    single ``"p50"`` band instead of four identical ones.
    """
    ordered = sorted(n_obs_values)
    n = len(ordered)
    strata: list[tuple[int, str]] = []
    for quantile, label in zip(_LENGTH_QUANTILES, _LENGTH_LABELS, strict=True):
        edge = ordered[math.floor(quantile * (n - 1))]
        if not strata or edge != strata[-1][0]:
            strata.append((edge, label))
    return strata


def _rank_normalized(values: list[tuple[str, float]]) -> dict[str, float]:
    """Rank each ``(ts_id, value)`` ascending and scale the rank into ``[0, 1]`` (pure).

    Rank, not min-max: ranks are scale-free, so one outlier cannot flatten an axis for every
    other series (a legitimately huge ``diff_cv`` from a series whose level sits on the guard
    floor would otherwise collapse the axis to two useful values). Ties break on ``ts_id``, so
    every candidate gets a distinct rank and the composite is a total order. A single
    candidate ranks ``0.0`` rather than dividing by zero.
    """
    n = len(values)
    if n <= 1:
        return {ts_id: 0.0 for ts_id, _value in values}
    ordered = sorted(values, key=lambda pair: (pair[1], pair[0]))
    return {ts_id: rank / (n - 1) for rank, (ts_id, _value) in enumerate(ordered)}


def _complexity_scores(candidates: list[SeriesStats]) -> dict[str, float]:
    """Equal-weighted mean of the rank-normalized complexity axes, per ``ts_id`` (pure).

    The axes are volatility, intermittency and seasonal strength — the cost drivers that are
    *not* length. ``acf_at_season`` enters as a magnitude: a strong negative lag-m correlation
    is as much structure for a seasonal model to fit as a strong positive one.

    A non-finite value in a caller-supplied `SeriesStats` (a Spark aggregation over an all-null
    series returns ``NaN`` readily) is coerced to ``0.0`` rather than raised on, because a
    ``NaN`` sort key produces an order that varies with the pandas/numpy version — the exact
    class of nondeterminism this module cannot tolerate.

    Consequence worth stating: ``complexity`` is a property of the **panel**, not of the series
    alone. The same series in a different panel gets a different score.
    """
    ranked: list[dict[str, float]] = []
    for axis in _COMPLEXITY_AXES:
        raw: list[tuple[str, float]] = []
        for stats in candidates:
            value = float(getattr(stats, axis))
            value = value if math.isfinite(value) else 0.0
            raw.append((stats.ts_id, abs(value) if axis == "acf_at_season" else value))
        ranked.append(_rank_normalized(raw))
    return {
        stats.ts_id: sum(axis_ranks[stats.ts_id] for axis_ranks in ranked) / len(ranked)
        for stats in candidates
    }


def select_profile_sample(
    stats: Sequence[SeriesStats],
    *,
    samples: int = _DEFAULT_PROFILE_SAMPLES,
) -> list[SampleSpec]:
    """Choose which series to profile: stratified on length, spread on complexity (pure).

    Takes statistics, not a panel: `series_stats` is the pandas path, and a Spark aggregation
    can build the identical rows for a panel too big to collect. Takes no ``cfg`` either, which
    is what makes "same stats in, same sample out" trivially true — there is no clock, no RNG,
    and every intermediate collection is an explicitly sorted list, so the result is
    byte-identical for a shuffled input.

    The algorithm, in order:

    1. Candidates are the series with at least `_MIN_PROFILABLE_OBS` rows, sorted by ``ts_id``.
       A series too short for the *models* is deliberately kept — minimum history is model
       policy (``horizon``, ``backtest.min_train``), and duplicating that arithmetic here would
       put it in two places. Its fit fails, `measure_fit` records ``ok=False``, and
       `build_profile` skips it.
    2. Flat series (``distinct_values <= 1``) are demoted out of the complexity phase: they
       short-circuit most fits, so sizing a fleet on one under-provisions. They stay eligible
       for the longest pick and for backfill, and the demotion is skipped entirely when *every*
       candidate is flat — a real all-flat panel must still yield a sample.
    3. **The longest series is claimed first**, before any budget arithmetic. A memory bound
       derived from a sample that excludes the longest series is not a bound. Ties go to the
       smallest ``ts_id``.
    4. The remaining budget goes round-robin over the non-empty length strata, alternating the
       complexity **max** then **min** inside each — so a small budget spans the length range
       instead of exhausting itself in one bucket, and a short, violently intermittent series
       stays reachable.
    5. Anything left over backfills from the unclaimed candidates, longest first.

    ``samples`` below 1 raises `ValueError` — it is a caller-argument violation, and returning
    ``[]`` would present "no basis" as "nothing to profile". Above `_MAX_PROFILE_SAMPLES` it
    clamps silently, because that ceiling is a budget cap rather than bad input. A duplicate
    ``ts_id`` also raises: `series_stats` cannot produce one, so it means two stat sources were
    merged, and silently deduping would shrink the real budget and measure one series twice,
    biasing the median.

    ``result[0]`` is always the ``"longest"`` pick; the rest is ordered by
    ``(stratum, reason, ts_id)``.
    """
    if samples < 1:
        raise ValueError(f"samples must be >= 1, got {samples}")

    seen: set[str] = set()
    for stat in stats:
        if stat.ts_id in seen:
            raise ValueError(f"duplicate ts_id '{stat.ts_id}' in stats")
        seen.add(stat.ts_id)

    candidates = sorted(
        (stat for stat in stats if stat.n_obs >= _MIN_PROFILABLE_OBS), key=lambda s: s.ts_id
    )
    if not candidates:
        return []

    budget = min(samples, _MAX_PROFILE_SAMPLES)
    complexity = _complexity_scores(candidates)
    strata = _length_strata([stat.n_obs for stat in candidates])
    # First band whose edge covers the length; the last edge is the longest series, so this
    # always finds one.
    stratum_of = {
        stat.ts_id: next(i for i, (edge, _label) in enumerate(strata) if edge >= stat.n_obs)
        for stat in candidates
    }

    def spec(stat: SeriesStats, reason: str) -> SampleSpec:
        index = stratum_of[stat.ts_id]
        return SampleSpec(
            ts_id=stat.ts_id,
            n_obs=stat.n_obs,
            stratum=index,
            stratum_label=strata[index][1],
            reason=reason,
            complexity=complexity[stat.ts_id],
            stats=stat,
        )

    longest = min(candidates, key=lambda s: (-s.n_obs, s.ts_id))
    claimed = {longest.ts_id}
    picked = [spec(longest, "longest")]

    # Flat series are poor evidence, but "every series is flat" is a panel, not an error.
    eligible = [stat for stat in candidates if stat.distinct_values > 1] or candidates
    buckets: dict[int, list[SeriesStats]] = {}
    for stat in eligible:
        buckets.setdefault(stratum_of[stat.ts_id], []).append(stat)

    # One pick per stratum per pass, ascending; each stratum alternates max/min on its own turn
    # counter so a stratum that runs dry does not shift the alternation of the others.
    turns = dict.fromkeys(buckets, 0)
    while len(picked) < budget:
        progressed = False
        for index in sorted(buckets):
            if len(picked) >= budget:
                break
            pool = [stat for stat in buckets[index] if stat.ts_id not in claimed]
            if not pool:
                continue
            want_max = turns[index] % 2 == 0
            turns[index] += 1
            chosen = min(
                pool,
                key=(
                    (lambda s: (-complexity[s.ts_id], s.ts_id))
                    if want_max
                    else (lambda s: (complexity[s.ts_id], s.ts_id))
                ),
            )
            claimed.add(chosen.ts_id)
            picked.append(spec(chosen, "complexity_max" if want_max else "complexity_min"))
            progressed = True
        if not progressed:
            break

    # Backfill from whatever is left (demoted flat series included), longest first.
    if len(picked) < budget:
        rest = sorted(
            (stat for stat in candidates if stat.ts_id not in claimed),
            key=lambda s: (-s.n_obs, s.ts_id),
        )
        for stat in rest[: budget - len(picked)]:
            claimed.add(stat.ts_id)
            picked.append(spec(stat, "fill"))

    # The headline invariant is readable off index 0; the tail is sorted for a stable audit diff.
    reason_rank = {"complexity_max": 0, "complexity_min": 1, "fill": 2}
    tail = sorted(picked[1:], key=lambda s: (s.stratum, reason_rank[s.reason], s.ts_id))
    return [picked[0], *tail]
