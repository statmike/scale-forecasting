"""Absence-aware numeric coercion — the helpers that keep ``None`` from becoming ``0``.

Every aggregated axis in this package is ``| None``, and ``None`` means "we have no basis
for this number" — never "zero". A CPU-only family reports ``peak_gpu_bytes=None``, because
``0 * 1.3`` is still ``0`` bytes of GPU, and that is a plan with no basis behind it. Making
the absence type-level is what stops a consumer silently sizing off nothing.

These five functions are where that discipline is mechanised: `usable` decides what counts
as evidence, `safe_max` / `safe_median` aggregate only over evidence and return ``None``
when there is none, and `as_number` / `as_optional_int` coerce untrusted JSON without
inventing a value for a missing key. Kept in their own module because both `cost` and
`signature` need them and `cost` already depends on `signature`.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from typing import Any


def usable(value: float | int | None) -> bool:
    """True when a measured value is real evidence: present, finite, and strictly positive.

    Zero is not evidence on any axis here — an RSS delta of 0 is the high-water-mark artefact,
    a wall time of 0 is a fit too fast to time, and a GPU peak of 0 is a fit that allocated
    nothing. All three must collapse to ``None`` rather than be maxed or medianed into a size.
    """
    return value is not None and math.isfinite(value) and value > 0


def safe_max(values: Sequence[float | int | None]) -> float | None:
    """Max over the usable values, or None when none of them is usable (pure)."""
    evidence = [float(v) for v in values if usable(v)]
    return max(evidence) if evidence else None


def safe_median(values: Sequence[float | int | None]) -> float | None:
    """Median over the usable values, or None when none of them is usable (pure).

    ``statistics.median`` averages the two middle values on an even count, which is
    deterministic and does not depend on the input order.
    """
    evidence = [float(v) for v in values if usable(v)]
    return statistics.median(evidence) if evidence else None


def as_number(value: Any) -> float:
    """A row cell as a finite float, with NULL / non-numeric / non-finite all reading as ``0.0``.

    Zero is already this module's "no evidence" value on every axis `usable` guards, so a missing
    reading needs no second representation and no branch at every call site.
    """
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) else 0.0


def as_optional_int(value: Any) -> int | None:
    """A row cell as an int, preserving ``None`` — the axes where NULL means NOT MEASURED.

    ``peak_gpu_bytes`` and ``process_rss_bytes`` must not collapse a missing reading to ``0``: a
    consumer reading zero device bytes would pack ten tasks onto a device that fits two.
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
