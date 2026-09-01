"""Measurements -> a per-family cost model. Pure arithmetic, no accelerator required.

`build_profile` folds a list of `MeasuredFit` records into a `ComputeProfile`: per-model
costs, rolled up per family, with safety margins applied. This is the pure half of the
package's pure/I-O seam — the arithmetic that decides how much hardware a run gets must be
testable with no accelerator, no cluster, and no cloud, exactly as ``calibrate_gpu_fraction``
is.

**Why both tails.** `build_profile` keeps the **max** of the peaks and the **median** of the
times. Max governs *safety* — how many tasks may share a device or an executor without an
OOM. Median governs *throughput* — how much work the fleet has, i.e. how wide it needs to
be. Sizing a fleet off the worst case over-provisions every run; sizing memory off the
median OOM-kills it. Using one tail for both is the mistake this split exists to prevent.

**Absence is a value.** Every aggregated axis is ``| None``, and ``None`` means "we have no
basis for this number" — never "zero"; see `numbers` for the helpers that enforce it.

**This module reports bytes, seconds and cores. It never emits a runtime knob** — no GPU
fraction, no executor cores, no node count, no autoscaling bound. Turning a `ComputeProfile`
into those is three different translations of the same numbers and lands later as
``plan_resources``, which consumes a `ComputeProfile` plus a cell count and nothing else.
The GPU-memory denominator stays in ``ray_io.device_memory_bytes``; this module never
re-declares a device table.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from .measure import _MIN_WALL_S, MeasuredFit
from .numbers import as_number, as_optional_int, safe_max, safe_median, usable
from .sampling import SampleSpec
from .signature import DataSignature

# Headroom on measured *peaks*. Matches the existing ``compute.gpu_safety_margin`` default
# so the two safety factors do not disagree once they are unified in config.
_DEFAULT_MEMORY_MARGIN = 1.3  # ratio, applied to max()

# Headroom on measured *times*. Deliberately smaller than the memory margin: over-estimating
# time buys extra slots (money), under-estimating memory kills the task (correctness).
# Asymmetric risk, asymmetric margin.
_DEFAULT_TIME_MARGIN = 1.2  # ratio, applied to median()

# Below this a ``cpu_s / wall_s`` ratio is clock noise, not a second thread. For a genuinely
# single-threaded fit the two clocks differ only by scheduling jitter, and which one lands
# higher is a coin flip — so an unguarded ``ceil()`` turns 1.0000005 into a 2-core request and
# halves fleet density. Applied before rounding up, never to the reported raw ratio.
_CORE_SNAP_TOLERANCE = 0.05  # cores


@dataclass(frozen=True)
class ModelCost:
    """Aggregated cost of one model across the profiled sample — RAW, no margin (pure).

    Every axis is ``| None`` and independently derived: a measurement can be good evidence
    for wall time and no evidence at all for GPU memory, so usability is decided per axis,
    not per record. ``None`` means no usable measurement contributed to that axis.

    No margins are applied here. They ride on `FamilyCost` and `ComputeProfile`, which is
    what a consumer reads, so a margin can never be applied twice on the way through.
    """

    model_type: str
    family: str
    n_fits: int  # measurements supplied for this model
    n_ok: int  # of those, how many had ok=True
    max_n_obs: int  # length of the longest series actually measured, 0 if none
    max_peak_rss_bytes: int | None  # bytes, max over usable values — DIAGNOSTIC (marginal)
    max_peak_gpu_bytes: int | None  # bytes, max over usable values
    median_wall_s: float | None  # seconds, median over usable values — the throughput tail
    median_cpu_s: float | None  # seconds, median over usable values
    max_effective_cores: float | None  # cores, max over usable per-fit ratios
    max_process_rss_bytes: int | None = None  # bytes, max absolute footprint — SIZES THE SLOT


@dataclass(frozen=True)
class FamilyCost:
    """One family's slot shape and workload, rolled up from its models (pure).

    Roll-up rule, and the reason it differs per axis: **peaks take the max** across the
    family's models, because one slot must hold whichever model lands in it; **times take
    the median** for the per-cell question and the **sum** for the per-series question,
    because a family's models all run — they do not compete for one cell.

    The margins are carried on the record rather than applied by the caller so the sized
    numbers and the raw measurements travel together: an audit reads both "measured 3.1
    GiB" and "sized 4.1 GiB" off one object, and a consumer cannot accidentally apply the
    margin twice.

    **Invariant: every derived property is ``None`` if and only if its raw basis is
    ``None``.** There is no fabricated fallback here — a consumer that gets ``None`` has
    been told, unambiguously, that it must fall back to its own static default (for
    ``slot_cores`` that default is 1, today's hardcoded value).
    """

    family: str
    models: tuple[str, ...]  # sorted model_types that contributed a usable number
    # Counted over every measurement tagged with this family — including those from a model
    # that produced nothing usable and is therefore absent from ``models``. Scoping these to
    # the surviving models instead would report a family that was 2-of-4 as a clean 2-of-2,
    # which is not a missing number but a wrong one: a family whose heavyweight member OOM'd
    # on every fit would present as a fully successful measurement.
    n_fits: int  # measurements supplied for this family
    n_ok: int  # of those, how many had ok=True
    max_peak_rss_bytes: int | None  # bytes, max over models — RAW; DIAGNOSTIC (marginal)
    max_peak_gpu_bytes: int | None  # bytes, max over models — RAW; None == not measured
    max_effective_cores: float | None  # cores, max over models — RAW
    median_wall_s: float | None  # seconds, median over models' median_wall_s — RAW
    total_wall_s_per_series: float | None  # seconds, SUM over models' median_wall_s — RAW
    memory_margin: float  # the ratio the slot_* properties apply
    time_margin: float  # the ratio the planning_* properties apply
    max_process_rss_bytes: int | None = None  # bytes, max over models — RAW; SIZES THE SLOT

    # --- sized values: what a runtime translation actually consumes ---------------
    @property
    def slot_rss_bytes(self) -> int | None:
        """Host memory one cell of this family needs, margin applied (bytes, rounded up).

        Built on ``max_process_rss_bytes`` — the **absolute** footprint — not on the marginal
        ``max_peak_rss_bytes``. A slot holds an interpreter with the model stack imported, not
        just the incremental allocation of one fit, and the marginal number is in any case
        unstable: it swings 17x on the order the sample was measured in and reads 0.00 MB for a
        model whose fit is served entirely from already-resident pages. See `MeasuredFit`.

        ``None`` when nothing measured the absolute footprint (no ``resource`` module, or every
        fit failed), which tells the consumer to fall back to its own static default rather
        than to size against a number that was never taken.
        """
        if self.max_process_rss_bytes is None:
            return None
        return math.ceil(self.max_process_rss_bytes * self.memory_margin)

    @property
    def slot_gpu_bytes(self) -> int | None:
        """Device memory one cell needs, margin applied; None when the axis is unmeasured."""
        if self.max_peak_gpu_bytes is None:
            return None
        return math.ceil(self.max_peak_gpu_bytes * self.memory_margin)

    @property
    def slot_cores(self) -> int | None:
        """Whole cores one cell needs: ``ceil(max_effective_cores)``, at least 1.

        No margin — a core count is already discrete and already the max over the family — but
        a `_CORE_SNAP_TOLERANCE` snap before rounding up, because the input is a ratio of two
        clocks. A single-threaded fit lands on either side of 1.0 at random, and since this is
        the **max** over the family, the chance that at least one member lands a hair above
        grows with family size: five single-threaded models trip it ~97% of the time. Without
        the snap that reads as ``slot_cores=2`` and halves fleet density, with an audit record
        showing ``1.0000005`` that looks correct to two decimal places.
        """
        if self.max_effective_cores is None:
            return None
        return max(1, math.ceil(self.max_effective_cores - _CORE_SNAP_TOLERANCE))

    @property
    def planning_wall_s(self) -> float | None:
        """Expected seconds for the median cell of this family, time margin applied."""
        if self.median_wall_s is None:
            return None
        return self.median_wall_s * self.time_margin

    @property
    def planning_total_wall_s_per_series(self) -> float | None:
        """Expected seconds to run every model of this family for one series, margin applied."""
        if self.total_wall_s_per_series is None:
            return None
        return self.total_wall_s_per_series * self.time_margin


@dataclass(frozen=True)
class ProfileProvenance:
    """Where a profile's numbers came from, carried with the numbers (pure).

    A sizing decision an operator cannot attribute is one they cannot argue with. The load-bearing
    field is ``basis``:

    * ``measured`` — taken on this run's own data. The strongest claim.
    * ``reference`` — measured, genuinely, but **not on your data**: another run's harvest, or the
      baseline shipped with the product. Real evidence with a caveat, and without a name for that
      state an operator reading a resolved fleet shape cannot tell whose evidence produced it.
    * ``assumed`` — no measurement; the static arithmetic. Recorded so "we had nothing" is a stated
      outcome rather than an absent field.
    """

    basis: Literal["measured", "reference", "assumed"]
    source: str  # the `compute.profile.source` value that produced this — the audit key
    run_id: str | None = None  # the harvest's run, when the evidence came from one
    baseline_version: str | None = None  # the shipped baseline's version, when it came from that
    measured_at: str | None = None  # ISO-8601, when the evidence was recorded
    signature: DataSignature | None = None  # what it was measured on
    warnings: tuple[str, ...] = ()  # signature mismatches; see `compare_signatures`

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict, for the telemetry stamp."""
        return {
            "basis": self.basis,
            "source": self.source,
            "run_id": self.run_id,
            "baseline_version": self.baseline_version,
            "measured_at": self.measured_at,
            "signature": self.signature.to_dict() if self.signature else None,
            "warnings": list(self.warnings),
        }


# How many contributing series ids `ComputeProfile.to_dict` carries into a telemetry blob. Enough
# to spot-check which series the evidence came from, few enough that the blob stays readable and
# bounded no matter how wide the harvest was; the full count travels beside them.
_TELEMETRY_SAMPLE_IDS = 50


@dataclass(frozen=True)
class ComputeProfile:
    """The measured cost model for one run: per-model and per-family, both tails kept (pure).

    The whole object is stamped into run telemetry, so it stays JSON-representable and
    carries the margins it was built with — a sizing decision that cannot be re-derived
    from its own record is not auditable. Both mappings are built from sorted keys so the
    serialized form is byte-stable and an audit diff is meaningful.

    A family or model is **absent** when nothing usable was measured for it. Absence is the
    signal to fall back to static config; a zero-valued entry would be consumed as a real
    size. Absence is deliberately *not* silent: ``dropped_models`` and ``first_error_by_model``
    name what fell out and why, so "this family was sized off one of its two models, because
    the other one OOM'd" is readable straight off the record instead of being inferred from a
    gap in it.
    """

    families: dict[str, FamilyCost]  # keyed by family: statistical | ml | deep_learning
    models: dict[str, ModelCost]  # keyed by model_type, flat — no nesting to walk
    memory_margin: float
    time_margin: float
    n_measurements: int  # measurements supplied
    n_ok: int  # of those, how many had ok=True
    sample_ts_ids: tuple[str, ...]  # sorted unique ids that contributed usable evidence
    # Measured but contributed nothing: every fit failed, or every axis was unusable. These are
    # the models missing from ``models``/``FamilyCost.models``, named so the gap is legible.
    dropped_models: tuple[str, ...] = ()
    # model_type -> first error text seen for it. The one place a failure *reason* survives
    # aggregation; without it a profile can only say a fit produced nothing, never why.
    first_error_by_model: dict[str, str] | None = None
    # The pre-pass sample, when the caller passes it — which series were measured and why.
    sample: tuple[SampleSpec, ...] = ()
    # Whose evidence this is. ``None`` on a profile built straight from measurements by a caller
    # that has not decided yet — `resolve_profile_source` is what stamps it.
    provenance: ProfileProvenance | None = None

    @property
    def n_failed(self) -> int:
        """``n_measurements - n_ok`` — how much of the sample we could not use."""
        return self.n_measurements - self.n_ok

    @property
    def is_empty(self) -> bool:
        """True when nothing usable was measured — the caller must fall back to static config."""
        return not self.models

    def for_family(self, family: str) -> FamilyCost | None:
        """This family's cost, or None when it was never measured (fall back, don't guess)."""
        return self.families.get(family)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict of the whole profile — raw aggregates *and* the sized values.

        The sized values are serialized alongside the raw ones so telemetry answers both
        "what did we measure" and "what did we ask the platform for" without the reader
        re-deriving a margin. Plain ints/floats/strings/None only: ``json.dumps`` must
        succeed with no custom encoder.

        **Bounded by construction**, because this is what lands in a registry JSON column. Every
        field above is sized by the model list except ``sample_ts_ids``, which is sized by the
        *panel*: a profile resolved from a harvest (`resolve_profile_source`) can carry tens of
        thousands of ids, and a telemetry blob that grows with the data is one nobody can read and
        one that eventually will not write. The count is kept whole (``n_sample_series``) and the
        ids are truncated to `_TELEMETRY_SAMPLE_IDS`; a reader sees the truncation as
        ``len(sample_ts_ids) < n_sample_series`` without needing a flag for it.
        """
        return {
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "memory_margin": self.memory_margin,
            "time_margin": self.time_margin,
            "n_measurements": self.n_measurements,
            "n_ok": self.n_ok,
            "n_failed": self.n_failed,
            "n_sample_series": len(self.sample_ts_ids),
            "sample_ts_ids": list(self.sample_ts_ids[:_TELEMETRY_SAMPLE_IDS]),
            "dropped_models": list(self.dropped_models),
            "first_error_by_model": dict(self.first_error_by_model or {}),
            "sample": [spec.to_dict() for spec in self.sample],
            "models": {
                name: {
                    "model_type": cost.model_type,
                    "family": cost.family,
                    "n_fits": cost.n_fits,
                    "n_ok": cost.n_ok,
                    "max_n_obs": cost.max_n_obs,
                    "max_peak_rss_bytes": cost.max_peak_rss_bytes,
                    "max_process_rss_bytes": cost.max_process_rss_bytes,
                    "max_peak_gpu_bytes": cost.max_peak_gpu_bytes,
                    "median_wall_s": cost.median_wall_s,
                    "median_cpu_s": cost.median_cpu_s,
                    "max_effective_cores": cost.max_effective_cores,
                }
                for name, cost in self.models.items()
            },
            "families": {
                name: {
                    "family": cost.family,
                    "models": list(cost.models),
                    "n_fits": cost.n_fits,
                    "n_ok": cost.n_ok,
                    "max_peak_rss_bytes": cost.max_peak_rss_bytes,
                    "max_process_rss_bytes": cost.max_process_rss_bytes,
                    "max_peak_gpu_bytes": cost.max_peak_gpu_bytes,
                    "max_effective_cores": cost.max_effective_cores,
                    "median_wall_s": cost.median_wall_s,
                    "total_wall_s_per_series": cost.total_wall_s_per_series,
                    "memory_margin": cost.memory_margin,
                    "time_margin": cost.time_margin,
                    "slot_rss_bytes": cost.slot_rss_bytes,
                    "slot_gpu_bytes": cost.slot_gpu_bytes,
                    "slot_cores": cost.slot_cores,
                    "planning_wall_s": cost.planning_wall_s,
                    "planning_total_wall_s_per_series": cost.planning_total_wall_s_per_series,
                }
                for name, cost in self.families.items()
            },
        }


def _model_cost(model_type: str, records: list[MeasuredFit]) -> ModelCost | None:
    """Roll one model's measurements up: max for peaks, median for times (pure).

    Returns ``None`` when no axis has usable evidence — the model is then absent from the
    profile entirely, because a `ModelCost` of zeros would be consumed as a real size while an
    absent one is unambiguously "fall back to static config". The measurements are still
    counted at the profile level, so the audit shows that fits happened and produced nothing.

    ``effective_cores`` has its own usability rule: a fit contributes iff it succeeded, was
    slow enough to time (`_MIN_WALL_S`), and reported a finite non-negative CPU time. The ratio
    is already floored at one core by `MeasuredFit.effective_cores`.
    """
    ok = [record for record in records if record.ok]
    cores = [
        record.effective_cores
        for record in ok
        if math.isfinite(record.wall_s)
        and record.wall_s >= _MIN_WALL_S
        and math.isfinite(record.cpu_s)
        and record.cpu_s >= 0
    ]

    max_rss = safe_max([record.peak_rss_bytes for record in ok])
    max_process_rss = safe_max([record.process_rss_bytes for record in ok])
    max_gpu = safe_max([record.peak_gpu_bytes for record in ok])
    median_wall = safe_median([record.wall_s for record in ok])
    median_cpu = safe_median([record.cpu_s for record in ok])
    max_cores = safe_max(cores)

    axes = (max_rss, max_process_rss, max_gpu, median_wall, median_cpu, max_cores)
    if all(axis is None for axis in axes):
        return None

    return ModelCost(
        model_type=model_type,
        family=min(record.family for record in ok),
        n_fits=len(records),
        n_ok=len(ok),
        max_n_obs=max((record.n_obs for record in ok), default=0),
        max_peak_rss_bytes=int(max_rss) if max_rss is not None else None,
        max_peak_gpu_bytes=int(max_gpu) if max_gpu is not None else None,
        median_wall_s=median_wall,
        median_cpu_s=median_cpu,
        max_effective_cores=max_cores,
        max_process_rss_bytes=int(max_process_rss) if max_process_rss is not None else None,
    )


def _family_cost(
    family: str,
    costs: list[ModelCost],
    memory_margin: float,
    time_margin: float,
    *,
    n_fits: int,
    n_ok: int,
) -> FamilyCost:
    """Roll a family's models up into one slot shape and one workload figure (pure).

    Peaks and cores take the **max** — any model in the family may land in the slot, so the
    slot must hold the widest of them. ``median_wall_s`` takes the median of the members'
    medians (what does a typical cell of this family cost?) while
    ``total_wall_s_per_series`` takes their **sum** (what does running the whole family for one
    series cost?). Those are different questions and a single field cannot answer both.

    ``n_fits`` / ``n_ok`` are passed in rather than summed from ``costs`` because a model that
    produced no usable axis has no `ModelCost` to sum — see the field comments.
    """
    member_walls: list[float | None] = [cost.median_wall_s for cost in costs]
    usable_walls = [wall for wall in member_walls if usable(wall)]

    max_rss = safe_max([cost.max_peak_rss_bytes for cost in costs])
    max_process_rss = safe_max([cost.max_process_rss_bytes for cost in costs])
    max_gpu = safe_max([cost.max_peak_gpu_bytes for cost in costs])

    return FamilyCost(
        family=family,
        models=tuple(sorted(cost.model_type for cost in costs)),
        n_fits=n_fits,
        n_ok=n_ok,
        max_peak_rss_bytes=int(max_rss) if max_rss is not None else None,
        max_peak_gpu_bytes=int(max_gpu) if max_gpu is not None else None,
        max_effective_cores=safe_max([cost.max_effective_cores for cost in costs]),
        median_wall_s=safe_median(member_walls),
        total_wall_s_per_series=sum(usable_walls) if usable_walls else None,
        memory_margin=memory_margin,
        time_margin=time_margin,
        max_process_rss_bytes=int(max_process_rss) if max_process_rss is not None else None,
    )


def build_profile(
    measurements: Sequence[MeasuredFit],
    *,
    sample: Sequence[SampleSpec] = (),
    memory_margin: float = _DEFAULT_MEMORY_MARGIN,
    time_margin: float = _DEFAULT_TIME_MARGIN,
) -> ComputeProfile:
    """Aggregate measurements into a per-family cost model: max for peaks, median for times (pure).

    The whole point of the split: ``max(peak) x memory_margin`` governs **safety** — how many
    tasks may share a device or an executor without an OOM — while ``median(time) x
    time_margin`` governs **throughput** — how many slots the load needs. Using the max for
    both systematically over-provisions every run; using the median for both OOM-kills it. Both
    tails are therefore kept and both are exposed.

    Usability is decided **per axis, not per record**: a measurement contributes to an axis iff
    it succeeded and that axis's value is finite and positive. One fit can be perfectly good
    evidence for wall time and no evidence at all for GPU memory, so filtering whole records
    would throw the first away and trusting whole records would fabricate the second. An axis
    with no usable value is ``None``, and a model or family with no usable axis is **absent**
    from the profile — absence is what tells a consumer to fall back to its static default.

    Margins are validated (below 1.0 asks for less headroom than the measurement, which is
    never valid), recorded on every record, and applied **only** in the ``slot_*`` /
    ``planning_*`` properties, so they can never be applied twice. Input is never mutated,
    grouping keys are iterated sorted, and the result is a pure function of the measurement
    *set* — a shuffled input yields an equal profile.

    ``sample`` is optional and carried through verbatim for the audit record: the profile then
    answers "what did this cost" and "which series was that measured on, and why those" from
    one object. It is not read by any arithmetic here.
    """
    for name, margin in (("memory_margin", memory_margin), ("time_margin", time_margin)):
        if not math.isfinite(margin) or margin < 1.0:
            raise ValueError(f"{name} must be a finite ratio >= 1.0, got {margin}")

    by_model: dict[str, list[MeasuredFit]] = {}
    for record in measurements:
        by_model.setdefault(record.model_type, []).append(record)

    models: dict[str, ModelCost] = {}
    for model_type in sorted(by_model):
        cost = _model_cost(model_type, by_model[model_type])
        if cost is not None:
            models[model_type] = cost

    by_family: dict[str, list[ModelCost]] = {}
    for cost in models.values():
        by_family.setdefault(cost.family, []).append(cost)

    # Counted off the measurements, not off the surviving ModelCosts, so a family that lost a
    # whole model still reports how many fits were really spent on it.
    family_fits: dict[str, int] = {}
    family_ok: dict[str, int] = {}
    for record in measurements:
        family_fits[record.family] = family_fits.get(record.family, 0) + 1
        family_ok[record.family] = family_ok.get(record.family, 0) + int(record.ok)

    families = {
        family: _family_cost(
            family,
            by_family[family],
            memory_margin,
            time_margin,
            n_fits=family_fits.get(family, 0),
            n_ok=family_ok.get(family, 0),
        )
        for family in sorted(by_family)
    }

    # What fell out, and the first reason given for it — the failure text stops here otherwise.
    dropped = tuple(sorted(name for name in by_model if name not in models))
    first_errors: dict[str, str] = {}
    for model_type in sorted(by_model):
        for record in by_model[model_type]:
            if not record.ok and record.error and model_type not in first_errors:
                first_errors[model_type] = record.error

    # The ids that actually backed a number, not merely the ids we tried — "we sized off these
    # six series" is the auditable claim.
    contributed = {
        record.ts_id
        for record in measurements
        if record.ok
        and (
            usable(record.peak_rss_bytes)
            or usable(record.peak_gpu_bytes)
            or usable(record.wall_s)
            or usable(record.cpu_s)
        )
    }

    return ComputeProfile(
        families=families,
        models=models,
        memory_margin=memory_margin,
        time_margin=time_margin,
        n_measurements=len(measurements),
        n_ok=sum(1 for record in measurements if record.ok),
        sample_ts_ids=tuple(sorted(contributed)),
        dropped_models=dropped,
        first_error_by_model=first_errors,
        sample=tuple(sample),
    )


def _harvest_family(model_type: str) -> str:
    """The compute family of ``model_type``, or ``"unknown"`` when it cannot be resolved.

    Resolved from the model registry rather than persisted per row: family is a property of the
    code, and a model that has since been re-homed should aggregate where it lives *now*. An
    unresolvable name (a model deleted since the run) lands in ``"unknown"``, which no translator
    consumes, so it is dropped from sizing without being hidden from the counts.
    """
    from ..models import get_model

    try:
        return str(get_model(model_type).family)
    except Exception:  # noqa: BLE001 - a stale model name must not sink a profile read
        return "unknown"


def harvest_profile(
    rows: Iterable[Mapping[str, Any]],
    *,
    memory_margin: float = _DEFAULT_MEMORY_MARGIN,
    time_margin: float = _DEFAULT_TIME_MARGIN,
) -> ComputeProfile:
    """Aggregate persisted ``forecast_metadata`` rows into a `ComputeProfile` (pure).

    **The second producer, and the one that scales.** `build_profile` aggregates a pre-pass that
    deliberately fits a small sample; this aggregates the fits a completed run already performed.
    Both feed the identical aggregation, so a harvested profile and a measured one are the same
    object and every translator consumes them the same way. The difference is only in how the
    evidence was obtained — and this way it is obtained for free, from every cell rather than
    from eight, on the real hardware rather than on a driver.

    That is what makes "size this run like run X" a **query** instead of an artifact store: a
    completed ``run_id`` *is* a profile, with its config, data signature and lineage already
    recorded next to it in the registry. Nothing new to version, expire, or garbage-collect.

    ``rows`` are mappings with ``forecast_metadata`` column names — from a BigQuery read, a test
    fixture, or a committed baseline file; the function never touches BigQuery itself. Missing
    keys read as NULL, so rows written before the measurement columns existed degrade to
    wall-time-only evidence rather than raising.

    **Two filters, both structural.** Backtest fold rows (``fold_id`` set) are skipped because
    ``fit_seconds`` on the full-fit row already brackets the whole cell, folds included — counting
    folds too would double-count the same work. Ensemble rows (``ensemble_id`` set) are skipped
    because an ensemble is arithmetic over predictions, not a fit whose cost sizes a slot.

    **How success is inferred, since ``forecast_metadata`` carries no status column.** A cell that
    errored returns before its wall clock is recorded and lands with ``fit_seconds`` of zero, so a
    usable wall time is exactly the signal that a fit happened. Those rows still count toward
    ``n_measurements`` and ``n_failed``, so "we sized off 940 of 1000 cells" stays visible — the
    same honesty `build_profile` gives a pre-pass, at run scale.
    """
    measurements: list[MeasuredFit] = []
    family_of: dict[str, str] = {}
    for row in rows:
        if row.get("fold_id") is not None or row.get("ensemble_id") is not None:
            continue
        model_type = str(row.get("model_type") or "")
        if not model_type:
            continue
        if model_type not in family_of:
            family_of[model_type] = _harvest_family(model_type)
        wall_s = as_number(row.get("fit_seconds"))
        measurements.append(
            MeasuredFit(
                ts_id=str(row.get("ts_id") or ""),
                model_type=model_type,
                family=family_of[model_type],
                n_obs=int(as_number(row.get("n_obs"))),
                wall_s=wall_s,
                cpu_s=as_number(row.get("cpu_seconds")),
                # The RSS *delta* axis is not harvested: it is order-dependent to the point of
                # uselessness (see `MeasuredFit`), and 0 already means "no evidence" there. The
                # absolute high-water — the number that actually sizes a slot — arrives on
                # ``process_rss_bytes`` below.
                peak_rss_bytes=0,
                peak_gpu_bytes=as_optional_int(row.get("peak_gpu_bytes")),
                ok=usable(wall_s),
                error=None,
                intraop_threads=as_optional_int(row.get("intraop_threads")),
                host_cpu_count=None,
                rss_peak_reset=False,
                process_rss_bytes=as_optional_int(row.get("process_rss_bytes")),
            )
        )
    return build_profile(measurements, memory_margin=memory_margin, time_margin=time_margin)
