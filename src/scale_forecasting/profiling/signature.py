"""What a profile was measured on — and whether this run still resembles it.

A `ComputeProfile` is only as good as the data it was measured against. A profile harvested
from a 1k-series daily panel says nothing useful about a 100k-series hourly one, but nothing
in the numbers themselves reveals that. `DataSignature` is the small set of panel properties
recorded alongside every profile so the mismatch is detectable rather than silent.

`signature_from_config` reads the intent, `signature_from_rows` reads what actually landed,
and `compare_signatures` reports the axes that have drifted far enough to matter. Drift is a
*warning*, never an error: a stale profile is still better evidence than no evidence, so the
consumer is told what moved and left to decide.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .numbers import as_optional_int

if TYPE_CHECKING:
    from ..config import RunConfig


@dataclass(frozen=True)
class DataSignature:
    """What a set of measurements was taken *on* — the basis for "is this evidence yours?" (pure).

    Deliberately coarse. The question a signature has to answer is not "is this the same data"
    (nothing is) but "is this close enough that its costs transfer" — a fit's cost tracks the shape
    of the panel, not its values. Four fields cover that: a different **table** is different work, a
    10x change in **series count** changes nothing per-cell but everything about the fleet, a 10x
    change in **history length** changes the per-fit cost directly, and a different **frequency**
    changes the seasonality every model estimates.

    Every field is optional because the two sides are read from different places and neither is
    complete: a config knows its table and frequency but not how long its series are until it
    reads them, while a harvest knows the lengths it measured but not the table it read
    (``forecast_metadata`` records no source). `compare_signatures` compares what both sides have.
    """

    source_table: str | None = None
    n_series: int | None = None
    median_n_obs: int | None = None
    freq: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict, for the telemetry stamp."""
        return {
            "source_table": self.source_table,
            "n_series": self.n_series,
            "median_n_obs": self.median_n_obs,
            "freq": self.freq,
        }


def signature_from_config(cfg: RunConfig) -> DataSignature:
    """What this run is about to read, as far as the config alone can say (pure).

    ``median_n_obs`` is always ``None`` here: history length is a property of the data, and at plan
    time nobody has read it. That asymmetry is deliberate rather than a gap — `compare_signatures`
    skips what one side cannot know, so a config-side signature checks the table, the series count
    and the frequency, and lets the length axis go unchecked rather than guessing at it.
    """
    return DataSignature(
        source_table=cfg.data.source_table,
        n_series=cfg.data.series_limit,
        median_n_obs=None,
        freq=cfg.data.freq,
    )


def signature_from_rows(
    rows: Iterable[Mapping[str, Any]], *, source_table: str | None = None
) -> DataSignature:
    """What a harvest was measured on, read off the rows themselves (pure).

    ``source_table`` is passed in because ``forecast_metadata`` does not record one — the run's
    header does. A caller that has the header supplies it; one that does not leaves it ``None``
    and the table axis simply goes unchecked.
    """
    ts_ids: set[str] = set()
    lengths: list[int] = []
    for row in rows:
        if row.get("fold_id") is not None or row.get("ensemble_id") is not None:
            continue
        ts_id = row.get("ts_id")
        if ts_id is not None:
            ts_ids.add(str(ts_id))
        n_obs = as_optional_int(row.get("n_obs"))
        if n_obs:
            lengths.append(n_obs)
    return DataSignature(
        source_table=source_table,
        n_series=len(ts_ids) or None,
        median_n_obs=int(statistics.median(lengths)) if lengths else None,
        freq=None,
    )


# How far two signatures may drift on a *scale* axis before the evidence stops transferring. An
# order of magnitude, because that is the point at which per-fit cost and fleet width stop being
# the same problem — and because a tighter band would warn on every ordinary week-to-week change
# in a panel and train operators to ignore the warning, which is worse than not emitting one.
_SIGNATURE_DRIFT_FACTOR = 10.0


def compare_signatures(want: DataSignature, have: DataSignature) -> tuple[str, ...]:
    """Human-readable reasons ``have``'s evidence may not describe ``want``'s run (pure).

    Warnings, never errors. A profile is a *hint* about hardware; sizing off drifted evidence is
    still better than sizing off nothing, and a hard failure here would turn a convenience into a
    thing that blocks runs. But it is never silent either — §3.10's rule is that a pinned profile
    from months ago on different data is worse than no profile precisely *because it looks
    authoritative*, and an unnamed mismatch is how it gets to look that way.

    Axes either side leaves ``None`` are skipped: an unchecked axis is honest, an axis compared
    against a placeholder is not.
    """
    out: list[str] = []
    if want.source_table and have.source_table and want.source_table != have.source_table:
        out.append(f"measured on a different table ({have.source_table} vs {want.source_table})")
    if want.freq and have.freq and want.freq != have.freq:
        out.append(f"measured at a different frequency ({have.freq} vs {want.freq})")
    for label, mine, theirs in (
        ("series count", want.n_series, have.n_series),
        ("history length", want.median_n_obs, have.median_n_obs),
    ):
        if not mine or not theirs:
            continue
        ratio = max(mine, theirs) / min(mine, theirs)
        if ratio >= _SIGNATURE_DRIFT_FACTOR:
            out.append(f"{label} differs by {ratio:.0f}x ({theirs} measured vs {mine} planned)")
    return tuple(out)
