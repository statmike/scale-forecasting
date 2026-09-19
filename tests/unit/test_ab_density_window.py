"""The repaired steady-state density rule, as an executable specification (offline, pure).

`docs/sql/ab_density_window.sql` is what actually runs during an A/B campaign. This file is the
reference implementation of the same rule in Python, plus the synthetic fleets that show it handles
the shapes the original rule got wrong. The SQL and the reference are two implementations of one
rule, kept in step by hand — so the last test here pins the SQL's declared constants against the
defaults below, which is the only part of the correspondence that can be checked without BigQuery.

**Why this exists.** Control 4 of the NeuralProphet A/B asks whether the two arms' fleets were
equally busy. Its denominator is right — fit-seconds over bucket-seconds times the nodes that ran a
cell in the bucket — but it chose its steady-state buckets by aligning them to the wall clock and
dropping exactly one from each end. That assumes the ramp and the drain each fit in one bucket and
that they line up with the top of the hour, and neither assumption survived contact:

* A cluster run whose last two cells landed 73 seconds past the hour grew an extra bucket. The
  extra one absorbed the "drop the last" rule, the real drain was promoted into the steady-state
  set, and a pair of identical fleets returned a 10.6% gap against a 10% threshold.
* On an autoscaled fleet the ramp spans two buckets, because workers arrive over several minutes.
  Both arms of the Ray A/B carried a ramp bucket *and* a drain bucket, and the control passed only
  because the contamination happened to be nearly symmetric. Its recorded agreement of 1.0E-4 is
  the difference between two wrong numbers.

Both pre-registered SQL files are deliberately left exactly as written; `docs/validation.md` records
why. The repair lives in a new file so the next A/B can pre-register it.

`_wall_clock_rule` below is the original, reimplemented here for one purpose: several tests assert
that it gets a fixture wrong and the repaired rule gets it right. A regression guard that only shows
the new rule working cannot tell you whether the fixture was ever hard.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pytest

# The rule's parameters. These are the values `docs/sql/ab_density_window.sql` declares, and the
# last test in this file asserts they have not drifted apart.
_BUCKET_SECONDS = 30 * 60
_DRAIN_SECONDS = 30 * 60
_MIN_STEADY_BUCKETS = 2
_DENSITY_TOLERANCE = 0.10

_SQL = Path(__file__).resolve().parents[2] / "docs" / "sql" / "ab_density_window.sql"


class Cell:
    """One fitted cell, as the three columns the rule reads. Times are integer epoch seconds."""

    __slots__ = ("arm", "worker", "started", "ended", "fit_seconds")

    def __init__(self, arm: str, worker: str, started: int, ended: int, fit_seconds: float):
        self.arm = arm
        self.worker = worker
        self.started = started
        self.ended = ended
        self.fit_seconds = fit_seconds


class Bucket(NamedTuple):
    """One steady-state bucket: what the SQL's `buckets` temp table holds, one row."""

    index: int
    fit_seconds: float
    active_nodes: int
    cells: int

    @property
    def density(self) -> float:
        return self.fit_seconds / (_BUCKET_SECONDS * self.active_nodes)


def _tally(placed: dict[int, list[Cell]]) -> list[Bucket]:
    return [
        Bucket(
            index=i,
            fit_seconds=sum(c.fit_seconds for c in group),
            active_nodes=len({c.worker for c in group}),
            cells=len(group),
        )
        for i, group in sorted(placed.items())
    ]


def steady_state_buckets(cells: list[Cell]) -> dict[str, list[Bucket]]:
    """The repaired rule: window from the last arrival to one drain before the last cell end.

    The window *start* is measured rather than assumed — it is the moment the fleet stopped growing,
    which is a fact about the run and can be computed without looking at a single density. The
    window *end* is still an assumption, because the drain has no equally clean observable: the
    symmetric definition, the first worker to stop working, reads a churned worker that ran two
    cells and was reclaimed as a drain beginning hours early.

    Only whole buckets count. A cell belongs to the bucket its fit ended in, which is what the
    original did and is not the part that was wrong.
    """
    out: dict[str, list[Bucket]] = {}
    for arm in sorted({c.arm for c in cells}):
        mine = [c for c in cells if c.arm == arm]
        joined: dict[str, int] = {}
        for c in mine:
            joined[c.worker] = min(joined.get(c.worker, c.started), c.started)
        window_start = max(joined.values())
        window_end = max(c.ended for c in mine) - _DRAIN_SECONDS

        # Clamped at zero so a window shorter than one bucket yields no buckets rather than a
        # negative count. BigQuery's DIV truncates toward zero where Python floors, and this is the
        # only place the two could disagree; both end up with an empty bucket set either way.
        whole = max(0, (window_end - window_start) // _BUCKET_SECONDS)

        placed: dict[int, list[Cell]] = {}
        for c in mine:
            if not window_start <= c.ended < window_end:
                continue
            i = (c.ended - window_start) // _BUCKET_SECONDS
            if i < whole:
                placed.setdefault(i, []).append(c)
        out[arm] = _tally(placed)
    return out


def _wall_clock_rule(cells: list[Cell]) -> dict[str, list[Bucket]]:
    """The original control 4, for contrast only: clock-aligned buckets, drop one at each end."""
    out: dict[str, list[Bucket]] = {}
    for arm in sorted({c.arm for c in cells}):
        placed: dict[int, list[Cell]] = {}
        for c in cells:
            if c.arm == arm:
                placed.setdefault(c.ended // _BUCKET_SECONDS, []).append(c)
        ranked = _tally(placed)
        out[arm] = ranked[1:-1]
    return out


def verdict(buckets: dict[str, list[Bucket]]) -> str:
    """PASS, or the reason it is INCONCLUSIVE. Too few buckets fails before the gap is looked at."""
    if set(buckets) != {"gpu", "cpu"} or not all(buckets.values()):
        return "INCONCLUSIVE - an arm produced no steady-state buckets at all"
    if min(len(b) for b in buckets.values()) < _MIN_STEADY_BUCKETS:
        return "INCONCLUSIVE - too few whole buckets between the last arrival and the drain"
    means = {arm: sum(b.density for b in bs) / len(bs) for arm, bs in buckets.items()}
    gap = abs(means["gpu"] - means["cpu"]) / means["cpu"]
    return "PASS" if gap <= _DENSITY_TOLERANCE else "INCONCLUSIVE - the arms were not equally busy"


# --------------------------------------------------------------------------------------------
# Synthetic fleets. A worker fits back-to-back 60-second cells between its join and its stop, so a
# bucket it fully covers contributes exactly `_BUCKET_SECONDS` of fit time and density comes out at
# 1.0 by construction. Anything below 1.0 in a test below is a bucket the rule should not have kept.
# --------------------------------------------------------------------------------------------

_CELL = 60
_PER_BUCKET = _BUCKET_SECONDS // _CELL

# A fit already in flight when the window opens finishes outside it, so bucket zero is short by up
# to one cell per worker — exactly one for each worker whose fit straddles the opening, none for a
# worker that happened to finish on it. That is the rule's one known bias; the SQL header says why
# it is left uncorrected. Every later bucket is saturated exactly, which is what the tests pin.
_FIRST_FLOOR = (_PER_BUCKET - 1) / _PER_BUCKET


def _fleet(
    arm: str, joins: list[int], stops: list[int], *, duty: float = 1.0, tag: str = "w"
) -> list[Cell]:
    """`duty` under 1.0 spaces the cells out, which is how a genuinely half-idle arm is built."""
    step = int(_CELL / duty)
    cells: list[Cell] = []
    for i, (join, stop) in enumerate(zip(joins, stops, strict=True)):
        t = join
        while t + _CELL <= stop:
            cells.append(Cell(arm, f"{arm}-{tag}{i}", t, t + _CELL, float(_CELL)))
            t += step
    return cells


def _fixed_fleet(arm: str, size: int, start: int, stop: int) -> list[Cell]:
    """A cluster-shaped fleet: every worker present from the first cell to the last."""
    return _fleet(arm, [start] * size, [stop] * size)


def test_a_run_that_spills_past_the_hour_no_longer_promotes_its_drain() -> None:
    """The recorded cluster failure, rebuilt: four identical workers, one of them finishing late.

    Three workers stop at 7800 and the fourth keeps going to 9660, which is past the 9000 boundary.
    Under the original rule those trailing cells form an extra clock bucket, the extra bucket takes
    the "drop the last" rule, and the arm's real drain is averaged in as if it were steady state.
    """
    cells = _fixed_fleet("cpu", 3, 600, 7800) + _fleet("cpu", [600], [9660], tag="late")

    old = _wall_clock_rule(cells)["cpu"]
    assert min(b.density for b in old) < 0.6, (
        "the fixture is not reproducing the defect — the original rule should be keeping the drain"
    )

    new = steady_state_buckets(cells)["cpu"]
    assert len(new) == 4
    assert _FIRST_FLOOR <= new[0].density <= 1.0
    assert all(b.density == pytest.approx(1.0) for b in new[1:])
    assert all(b.active_nodes == 4 for b in new)


def test_the_repaired_rule_does_not_care_where_the_run_falls_on_the_clock() -> None:
    """The sharpest statement of the defect: the original's answer moves when only the clock does.

    Same fleet, same work, same durations — shifted by a few minutes so the run lands differently
    against the top of the hour. A measurement of how busy two fleets were must not depend on that,
    and the original's does.
    """
    base = _fixed_fleet("cpu", 3, 600, 7800) + _fleet("cpu", [600], [9660], tag="late")

    def shifted(by: int) -> list[Cell]:
        return [Cell(c.arm, c.worker, c.started + by, c.ended + by, c.fit_seconds) for c in base]

    def mean(rule, cells) -> float:  # noqa: ANN001 - local helper over two rule functions
        bs = rule(cells)["cpu"]
        return sum(b.density for b in bs) / len(bs)

    old_answers = {round(mean(_wall_clock_rule, shifted(by)), 6) for by in (0, 300, 900, 1500)}
    new_answers = {round(mean(steady_state_buckets, shifted(by)), 6) for by in (0, 300, 900, 1500)}

    assert len(old_answers) > 1, "the original rule was supposed to be clock-dependent"
    assert len(new_answers) == 1, f"the repaired rule moved with the clock: {sorted(new_answers)}"


def test_an_autoscaled_ramp_wider_than_one_bucket_is_excluded_whole() -> None:
    """Workers arriving over 40 minutes, which is the shape that made the Ray arms agree by luck.

    The original drops one bucket and leaves the second half of the ramp inside the comparison. The
    repaired rule starts at the last arrival, so every kept bucket has the whole fleet in it — and
    `active_nodes` landing on the fleet size is the check that says so, independently of density.

    Note what the ramp does *not* look like: every worker has run at least one cell by the end of
    the second clock bucket, so `active_nodes` is already at its full six there and the bucket is
    unsaturated only in density. That is the same reason trimming to the maximum node count was
    rejected on the real data — the count is at full strength while the fleet is still filling up.
    """
    joins = [600, 900, 1500, 2100, 2700, 3000]
    cells = _fleet("gpu", joins, [12000] * len(joins))

    old = _wall_clock_rule(cells)["gpu"]
    assert old[0].active_nodes == len(joins), "the node count is no help in spotting this ramp"
    assert old[0].density < 0.8, "the original rule should be keeping an unsaturated ramp bucket"

    new = steady_state_buckets(cells)["gpu"]
    assert len(new) == 4
    assert all(b.active_nodes == len(joins) for b in new)
    assert _FIRST_FLOOR <= new[0].density <= 1.0
    assert all(b.density == pytest.approx(1.0) for b in new[1:])


def test_a_churned_worker_marks_only_the_bucket_it_worked_in() -> None:
    """Ray's autoscaler overshot to 108 workers for a steady 84, and the extras ran a cell or two.

    The denominator counts nodes that ran a cell *in that bucket*, so a node that appears once and
    is reclaimed depresses that one bucket and leaves every later one alone. That is inherited from
    the original rule unchanged — the repair moved the window, not the denominator — and it is the
    reason the alternative fix of trimming to the run's maximum node count was rejected: the maximum
    on the real autoscaled arm was 108, and it occurred during the ramp.

    Stated as a test because it is a real limitation and not a rounding detail: churn *inside* the
    steady window does bias its bucket down. On the arms measured it all happened before the window
    opened, which is why the window is the thing that had to be fixed.
    """
    joins = [600, 900, 1500]
    cells = _fleet("gpu", joins, [12000] * len(joins))
    cells += _fleet("gpu", [1600], [1720], tag="churn")  # two cells, inside the first bucket

    new = steady_state_buckets(cells)["gpu"]
    assert len(new) >= 4, "the churn worker should not have collapsed the window"
    assert new[0].active_nodes == len(joins) + 1
    assert new[0].density < _FIRST_FLOOR, "a one-cell node in the bucket must show up as slack"
    assert all(b.active_nodes == len(joins) for b in new[1:])
    assert all(b.density == pytest.approx(1.0) for b in new[1:])
    assert all(b.cells == len(joins) * _PER_BUCKET for b in new[1:])


def test_a_late_arrival_makes_the_control_refuse_rather_than_answer_from_one_bucket() -> None:
    """The repair's own failure mode, and it fails in the safe direction.

    Measuring the window start from the last arrival means one straggler can swallow the run. The
    rule returns INCONCLUSIVE instead of averaging whatever is left, because a mean over one bucket
    is that bucket, and a control that answers from one bucket is worse than one that declines.
    """
    steady = {"gpu": [600, 900, 1500], "cpu": [600, 900, 1500]}
    cells = [c for arm, j in steady.items() for c in _fleet(arm, j, [12000] * len(j))]
    assert verdict(steady_state_buckets(cells)) == "PASS"

    # One node joins with 2,000 seconds of run left: enough for a single whole bucket, not two.
    late = cells + _fleet("gpu", [8200], [12000], tag="late")
    assert len(steady_state_buckets(late)["gpu"]) == 1
    assert verdict(steady_state_buckets(late)).startswith("INCONCLUSIVE - too few whole buckets")

    # And one that joins later still leaves no whole bucket at all.
    later = cells + _fleet("gpu", [10000], [12000], tag="late")
    assert steady_state_buckets(later)["gpu"] == []
    assert verdict(steady_state_buckets(later)).startswith("INCONCLUSIVE - an arm produced no")


def test_the_repair_did_not_cost_the_control_its_power() -> None:
    """A wider window that passes everything is not a control. One arm half-idle must still fail.

    This is the test that would catch the tempting mistake — widening or softening the rule until
    the inconvenient result goes away — because the failure it asserts is one the rule has to keep.
    """
    joins = [600, 900, 1500]
    busy = _fleet("gpu", joins, [12000] * len(joins))
    idle = _fleet("cpu", joins, [12000] * len(joins), duty=0.5)

    buckets = steady_state_buckets(busy + idle)
    assert all(b.density == pytest.approx(1.0) for b in buckets["gpu"][1:])
    assert all(b.density == pytest.approx(0.5) for b in buckets["cpu"])
    assert verdict(buckets) == "INCONCLUSIVE - the arms were not equally busy"

    # And the threshold is a threshold: a 5% difference is still a pass.
    near = _fleet("cpu", joins, [12000] * len(joins), duty=0.95)
    assert verdict(steady_state_buckets(busy + near)) == "PASS"


def test_the_sql_and_this_reference_declare_the_same_rule() -> None:
    """Two implementations of one rule drift silently. Only the constants can be checked offline.

    The arithmetic cannot be — the SQL runs in BigQuery and this does not — so this asserts the
    parameters match and that the shipped file does not carry the two pieces of the original that
    were wrong: clock-aligned bucket starts and a positional trim.
    """
    text = _SQL.read_text(encoding="utf-8")
    body = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("--"))

    assert f"DEFAULT {_BUCKET_SECONDS // 60};" in text
    assert f"DEFAULT {_DRAIN_SECONDS // 60};" in text
    for decl in (
        f"DECLARE bucket_minutes INT64 DEFAULT {_BUCKET_SECONDS // 60};",
        f"DECLARE drain_minutes INT64 DEFAULT {_DRAIN_SECONDS // 60};",
        f"DECLARE min_steady_buckets INT64 DEFAULT {_MIN_STEADY_BUCKETS};",
        "DECLARE density_tolerance FLOAT64 DEFAULT 0.10;",
    ):
        assert decl in text, f"{_SQL.name} no longer declares: {decl}"
    assert _DENSITY_TOLERANCE == 0.10

    assert "bucket_rank" not in body, "the positional trim is the thing this file replaced"
    assert "UNIX_SECONDS(c.cell_ended_at) - UNIX_SECONDS(w.window_start)" in body, (
        "buckets must be tiled from the measured window start, not from the epoch"
    )
    assert "MAX(j.joined_at)" in body, "the window start must still be the last worker's arrival"


def test_the_original_sql_files_were_not_edited() -> None:
    """The point of the new file is that the pre-registered ones stay as they were.

    A pre-registered analysis that gets improved after the numbers land is no longer pre-registered,
    and `docs/validation.md` records a decision to re-run an arm rather than touch this rule. This
    asserts nobody later takes the cheaper option and edits control 4 in place.
    """
    for name in ("neuralprophet_ab.sql", "neuralprophet_ab_cluster.sql"):
        text = (_SQL.parent / name).read_text(encoding="utf-8")
        assert "bucket_rank > 1 AND bucket_rank_desc > 1" in text, (
            f"{name} is pre-registered and must keep the trim it was registered with, defect and "
            "all — the repaired rule lives in ab_density_window.sql"
        )
