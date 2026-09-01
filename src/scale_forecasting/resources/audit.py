"""What was decided, and off what evidence — the sizing record the registry keeps.

The top of the package, and the only module that has to see all three algebras at once: the
plans (`fleet`), whichever translation produced them (`serverless` or `cluster`), and the
profile they were sized from (`profiling`). Nothing here decides anything; it assembles what
was already decided into one JSON-safe blob.

It exists because a run's fleet is chosen once, before anything starts, off evidence that may
have come from a different run entirely — and until this is written down, the only trace of
that choice is a driver log line nobody keeps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..profiling.cost import ComputeProfile
    from .cluster import ClusterTranslation
    from .fleet import RuntimeResourcePlan
    from .serverless import ServerlessTranslation


def sizing_telemetry(
    *plans: RuntimeResourcePlan | None,
    translation: ServerlessTranslation | ClusterTranslation | None = None,
    profile: ComputeProfile | None = None,
    family: str | None = None,
) -> dict[str, Any]:
    """The whole sizing decision as one JSON-safe blob, for the registry stamp (pure).

    A run's fleet is chosen once, before anything starts, off evidence that may have come from a
    different run entirely — and until this is recorded, the only trace of it is a driver log line
    that nobody keeps. Three parts, because an auditor asks three questions:

    * ``plans`` — *what did we ask for?* One `RuntimeResourcePlan` per fleet the job runs
      (Spark spells one; a Ray run spells its CPU and GPU pools separately). ``None`` entries are
      dropped, so a Ray run with no GPU pool records one plan rather than a null.
    * ``translation`` — *what did that become on this platform?* The properties actually set, plus
      the ideals they snapped from — the gap between "the arithmetic wanted 6.4 cores" and "the
      legal table granted 8" is exactly what a surprising bill is explained by. ``None`` on Ray,
      which sets task options rather than properties and carries them on the plan.
    * ``profile`` — *off what evidence?* The measurements, their margins, and (via
      `ProfileProvenance`) whose run they came from and whether the data still matches. ``None``
      when nothing sized the memory axis, which is itself the answer to "why is this fleet the
      shape the static arithmetic gives".

    ``family`` is what the record is *filed under*: a run has one job per family, each sized
    independently, and a header field that held only the last one to finish would be worse than
    none. It defaults to the first plan's label, which is right whenever the job has one fleet
    (Spark) — a caller whose fleets disagree passes the job's own label instead, because a Ray
    deep-learning job's *CPU* pool is labelled ``"cpu"`` and filing that job under ``cpu`` would
    hide it. ``None`` when there is neither an override nor a plan to take one from.

    Pure and total: every input is optional, so a caller that has only some of the three still
    records what it has instead of recording nothing.
    """
    kept = [plan for plan in plans if plan is not None]
    return {
        "family": family or (kept[0].family if kept else None),
        "plans": [plan.to_dict() for plan in kept],
        "translation": translation.to_dict() if translation is not None else None,
        "profile": profile.to_dict() if profile is not None else None,
    }
