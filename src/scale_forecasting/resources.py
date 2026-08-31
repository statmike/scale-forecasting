"""Translation — one measured profile into the knobs a runtime actually accepts.

`profiling` reports bytes, seconds and cores and deliberately stops there. This module
is the other half: it turns a `ComputeProfile` plus a cell count into
the concrete request a runtime understands — a Ray ``.options()`` dict, and (in the
Serverless/cluster work that follows) a set of ``spark.*`` properties. One measurement,
three algebras.

**The seam is the point.** Sizing arithmetic that lives inside an engine can only be
tested with that engine running. Everything here is pure: no Ray, no Spark, no GPU, no
GCP. An engine hands over a `ComputeProfile` and the shape of one schedulable
unit (`UnitShape`), and gets back a `RuntimeResourcePlan` that carries both the
decision and the evidence for it.

**Two layers, because they answer different questions.**

* `resource_slot` — *how big is one cell of this family?* Cores, host memory, GPU
  fraction. Runtime-neutral: the same three numbers whether the cell runs as a Ray task
  or a Spark partition.
* `plan_resources` — *how much hardware does this load need?* Wraps the slot with the
  fleet math: slots per unit, units for the fan-out, the autoscaling ``[min, max]``.

**What measurement actually changed, honestly.** Of the three slot axes, the interesting
result is that **cores is usually 1, and that is a finding rather than a null one.** The
probe pins every native thread pool to one thread (`profiling._pinned_intraop_threads`),
which is exactly the environment a Ray task runs in — Ray exports ``OMP_NUM_THREADS`` =
the task's ``num_cpus``. So the measurement says the hardcoded ``num_cpus=1`` was right,
and now says it from evidence instead of from assumption. The axes that move a run are the
other two: a **memory** request that Ray had never been given at all, and a GPU fraction
derived from a real footprint rather than from `ray_io._NOMINAL_AUTO_FRACTION`.
(A Spark executor has no such automatic pin — its Python workers inherit the executor
environment and each one grabs the whole machine. `translate_serverless` closes that by
exporting the caps itself, which is also what makes a profile measured under a pin
describe the executor it is being used to size.)

**The knobs are discrete.** Serverless takes ``executor.cores`` from ``{4, 8, 16}`` — a
different set again on GPU — and rejects anything else at submit, minutes after the
operator walked away. So the Spark side never emits a raw arithmetic result: it computes
the ideal, snaps to a legal neighbour in whichever direction is *safe* for that knob
(memory up, tasks-per-device down), and keeps both numbers so an audit can see what the
arithmetic wanted and what the platform allowed.

**Absence propagates; it is never filled in silently.** A `ResourceSlot` records
which axes came from measurement (``measured``) and which fell back to a static default
(``assumed``), plus every clamp that was applied (``notes``). A plan whose memory axis was
never measured requests no memory — today's behaviour — rather than requesting a number
nobody took. This is the same contract `FamilyCost` keeps, carried one layer out
so that "we sized this off nothing" survives into telemetry.

**Density is bounded by memory, not only by cores.** ``slots_per_unit = unit.cores //
slot.cores`` is the design's formula and it is right up to the point where the cells do
not fit: eight NeuralProphet cells at 4 GiB each do not run on a 30 GiB node no matter how
many cores it has. When both the unit's memory and the slot's memory are known the density
takes the **min** of the two bounds. This mirrors the Serverless side of the design, where
memory-bound concurrency is explicit (``floor(usable_python_mem / peak_rss)``); it is the
same rule, stated on the runtime that had been ignoring it.

Public surface: ``ResourceSlot``, ``UnitShape``, ``RuntimeResourcePlan``,
``ServerlessTranslation``, ``ClusterTranslation``, ``resource_slot``, ``merge_slots``,
``plan_resources``, ``plan_fleet``, ``plan_serverless``, ``plan_dataproc_cluster``,
``machine_cores``, ``machine_memory_bytes``, ``schedulable_memory_bytes``,
``spark_tasks_per_executor``, ``serverless_unit``, ``cluster_unit``, ``slots_per_unit``,
``snap_to_legal``, ``tasks_for_ceiling``, ``translate_serverless``, ``translate_cluster``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .profiling import ComputeProfile

__all__ = [
    "ClusterTranslation",
    "ResourceSlot",
    "RuntimeResourcePlan",
    "ServerlessTranslation",
    "UnitShape",
    "cluster_unit",
    "machine_cores",
    "machine_memory_bytes",
    "merge_slots",
    "plan_dataproc_cluster",
    "plan_fleet",
    "plan_resources",
    "plan_serverless",
    "resource_slot",
    "schedulable_memory_bytes",
    "serverless_unit",
    "slots_per_unit",
    "snap_to_legal",
    "spark_tasks_per_executor",
    "tasks_for_ceiling",
    "translate_cluster",
    "translate_serverless",
]

# --- fallbacks: what a slot is worth when nothing measured it ------------------

# Cores per cell when the profile has no basis. One, because that is what every engine
# hardcodes today: falling back must reproduce current behaviour exactly, so enabling the
# profiler can never make an unmeasured run worse than a profiler-less one.
_DEFAULT_SLOT_CORES = 1

# GPU-fraction band, mirrored from ``ray_io``. Below the floor a task barely uses the
# device and packing overhead dominates; above 1.0 is meaningless. Duplicated rather than
# imported because this module must not depend on an engine — a drift test pins the two
# together instead (`tests/unit/test_resources.py`).
_MIN_GPU_FRACTION = 0.1
_NOMINAL_GPU_FRACTION = 0.5

# Cells we want to flow through one slot before widening the fleet, mirroring
# ``compute.ray_target_cells_per_slot``. Only a default for direct callers; every engine
# passes the configured value.
_DEFAULT_TARGET_CELLS_PER_SLOT = 8


# --- machine shapes: memory a GCE machine type implies -------------------------

# GiB of RAM per vCPU, by ``<family>-<class>`` prefix. Concurrency is bounded by memory as
# well as by cores, and neither Ray nor Vertex tells us a node's size — the machine type is
# the only handle we have. Values are the published GCE ratios; the *class* is what
# carries them (a "highmem" node holds roughly seven times the cells of a "highcpu" one at
# the same core count, which is the whole reason this is a table and not a constant).
_MEMORY_PER_CORE_GIB = {
    "n1-standard": 3.75,
    "n1-highmem": 6.5,
    "n1-highcpu": 0.9,
    "n2-standard": 4.0,
    "n2-highmem": 8.0,
    "n2-highcpu": 1.0,
    "n2d-standard": 4.0,
    "n2d-highmem": 8.0,
    "n2d-highcpu": 1.0,
    "e2-standard": 4.0,
    "e2-highmem": 8.0,
    "e2-highcpu": 1.0,
    "c2-standard": 4.0,
    "g2-standard": 4.0,  # L4 machines; the card is bundled into the machine type
}

# Unrecognised machine type → assume the smallest *standard* ratio. Same asymmetry as
# ``ray_io._DEFAULT_DEVICE_MEMORY_BYTES``: guessing low under-packs the node (we pay for
# capacity we don't use), guessing high over-packs it (the run OOMs). Only one of those is
# recoverable.
_DEFAULT_MEMORY_PER_CORE_GIB = 3.75

# Cores assumed for a machine type whose name does not carry a count. Unlike the memory
# axis there is no "unknown" answer available here: every caller divides by this number, and
# a unit with zero cores holds zero slots. Eight is the N1 default and the shape both the Ray
# CPU pool and the Dataproc worker already default to, so the fallback stands in for the
# concrete case it is most likely covering.
_DEFAULT_MACHINE_CORES = 8

# ``<family>-<class>-<cores>``. Both machine-shape readers below match against this one
# pattern so they never disagree about whether a name is legible.
_MACHINE_TYPE_RE = r"^([a-z0-9]+-[a-z]+)-(\d+)$"

# Share of a node's RAM that is actually schedulable. Ray reserves ~30% of available
# memory for the plasma object store by default and subtracts it from the node's ``memory``
# resource, so sizing against the machine's nameplate RAM over-packs by roughly that much.
# Also absorbs the OS and the Ray runtime itself.
_SCHEDULABLE_MEMORY_FRACTION = 0.7

_GIB = 1024**3


def machine_cores(machine_type: str) -> int:
    """vCPUs a GCE machine type implies — ``n1-standard-8`` → 8 (pure; unparseable → 8).

    The count suffix of a ``<family>-<class>-<cores>`` name — the same shape
    `machine_memory_bytes` parses, so the two axes agree about which names they understand.
    Matching the whole shape rather than just a trailing number is what keeps a custom type
    (``n1-custom-8-16384``, where the trailing number is megabytes) from being read as a
    16384-core machine.
    """
    match = re.match(_MACHINE_TYPE_RE, machine_type)
    return int(match.group(2)) if match else _DEFAULT_MACHINE_CORES


def machine_memory_bytes(machine_type: str) -> int:
    """RAM a GCE machine type implies, in bytes (pure; unknown → the smallest standard ratio).

    Derived as ``cores x GiB-per-vCPU`` from `_MEMORY_PER_CORE_GIB`, keyed on the
    ``<family>-<class>`` prefix — so ``n1-standard-8`` is 30 GiB and ``n1-highmem-8`` is 52
    GiB. Nameplate RAM, not schedulable RAM: `plan_resources` applies
    `_SCHEDULABLE_MEMORY_FRACTION` on top.

    Two different kinds of "we don't know", kept distinct because they warrant different
    answers. A name whose *class* is untabulated but whose shape parses (some future
    ``n4-standard-8``) falls back to the smallest standard ratio — it under-counts memory
    and therefore under-packs the node, the safe direction. A name that does not parse at
    all (``n1-custom-8-16384``, a bare alias) returns **0**, meaning *unknown*: callers
    treat that as no memory bound rather than as a machine with no memory, so an
    unrecognised type degrades to today's cores-only packing instead of to one slot.
    """
    match = re.match(_MACHINE_TYPE_RE, machine_type)
    if match is None:
        return 0
    prefix, cores = match.group(1), int(match.group(2))
    per_core = _MEMORY_PER_CORE_GIB.get(prefix, _DEFAULT_MEMORY_PER_CORE_GIB)
    return int(cores * per_core * _GIB)


# --- the slot: how big is one cell of this family ------------------------------


@dataclass(frozen=True)
class ResourceSlot:
    """What one cell of one family needs, and how much of that was actually measured (pure).

    Runtime-neutral by construction: three numbers plus their provenance. ``cores`` is
    always present (it has a defensible static default); ``memory_bytes`` and
    ``gpu_fraction`` are ``None`` when there is no basis, and ``None`` means *request
    nothing on this axis* — not zero, and not a guess.

    ``measured`` and ``assumed`` partition the axes that carry a value, so a reader of the
    telemetry can tell a 4 GiB request that came from a fit apart from one that came from a
    table. ``notes`` records every clamp in plain words; a slot that was silently trimmed to
    fit the machine is a sizing decision, and sizing decisions have to be auditable.
    """

    family: str
    cores: int  # >= 1, whole cores per cell
    memory_bytes: int | None  # bytes of host RAM per cell; None == no basis, request none
    gpu_fraction: float | None  # share of one device per cell; None == CPU-only or no basis
    device_bytes: int | None  # the denominator the fraction was computed against
    measured: tuple[str, ...] = ()  # axis names taken from a fit
    assumed: tuple[str, ...] = ()  # axis names that fell back to a static default
    notes: tuple[str, ...] = ()  # clamps applied, most-recent last

    @property
    def basis(self) -> str:
        """``"measured"`` if any axis came from a fit, else ``"static"`` — the headline."""
        return "measured" if self.measured else "static"

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict for telemetry — plain scalars and lists, no custom encoder."""
        return {
            "family": self.family,
            "basis": self.basis,
            "cores": self.cores,
            "memory_bytes": self.memory_bytes,
            "gpu_fraction": self.gpu_fraction,
            "device_bytes": self.device_bytes,
            "measured": list(self.measured),
            "assumed": list(self.assumed),
            "notes": list(self.notes),
        }


def _clamp_gpu_fraction(fraction: float) -> float:
    """Clamp a GPU fraction to ``[_MIN_GPU_FRACTION, 1.0]`` (pure)."""
    return max(_MIN_GPU_FRACTION, min(1.0, fraction))


def resource_slot(
    profile: ComputeProfile | None,
    family: str,
    *,
    use_gpu: bool = False,
    device_bytes: int | None = None,
    static_gpu_fraction: float | None = None,
    max_cores: int | None = None,
    max_memory_bytes: int | None = None,
) -> ResourceSlot:
    """Size one cell of ``family`` from the profile, falling back per-axis (pure).

    Each axis resolves independently, because they fail independently: a family can have a
    solid wall-time and RSS measurement and no GPU number at all (nothing ran on a device),
    or the reverse. Resolving them as a unit would throw away good evidence because of a
    neighbouring gap.

    * **cores** — ``FamilyCost.slot_cores``, else `_DEFAULT_SLOT_CORES`. Clamped to
      ``max_cores`` when given: a task asking for more cores than any node has is not slow,
      it is *unschedulable*, and Ray will sit on it forever rather than fail. Clamping and
      recording beats hanging.
    * **memory** — ``FamilyCost.slot_rss_bytes`` (the absolute footprint x the profile's
      memory margin), else ``None``. Clamped to ``max_memory_bytes`` for the same
      unschedulable-task reason.
    * **gpu_fraction** — ``None`` unless ``use_gpu``. When measured *and* ``device_bytes``
      is known: ``slot_gpu_bytes / device_bytes``, clamped to the band. Otherwise
      ``static_gpu_fraction`` (the operator's pin), and failing that the nominal.

    ``profile`` may be ``None`` (profiling off, or the pre-pass produced nothing) and the
    family may simply be absent from it; both take every fallback, which reproduces the
    pre-profiler behaviour exactly. That equivalence is the safety property: turning the
    profiler on can add information, never remove it.

    The margin already lives inside the ``slot_*`` properties, so nothing here multiplies
    again. Note that the measured GPU path applies the profile's ``memory_margin`` while the
    legacy `ray_io.calibrate_gpu_fraction` path applies ``compute.gpu_safety_margin``;
    both default to 1.3, so the two agree unless an operator moves one of them.
    """
    cost = profile.for_family(family) if profile is not None else None
    measured: list[str] = []
    assumed: list[str] = []
    notes: list[str] = []

    cores = cost.slot_cores if cost is not None else None
    if cores is None:
        cores = _DEFAULT_SLOT_CORES
        assumed.append("cores")
    else:
        measured.append("cores")
    if max_cores is not None and cores > max_cores:
        notes.append(f"cores {cores} exceeded the unit's {max_cores}; clamped")
        cores = max_cores
    cores = max(1, cores)

    memory_bytes = cost.slot_rss_bytes if cost is not None else None
    if memory_bytes is None:
        assumed.append("memory_bytes")
    else:
        measured.append("memory_bytes")
        if max_memory_bytes is not None and memory_bytes > max_memory_bytes:
            notes.append(
                f"memory {memory_bytes} exceeded the unit's schedulable "
                f"{max_memory_bytes}; clamped"
            )
            memory_bytes = max_memory_bytes

    gpu_fraction = _resolve_gpu_fraction(
        cost,
        use_gpu=use_gpu,
        device_bytes=device_bytes,
        static_gpu_fraction=static_gpu_fraction,
        measured=measured,
        assumed=assumed,
    )

    return ResourceSlot(
        family=family,
        cores=cores,
        memory_bytes=memory_bytes,
        gpu_fraction=gpu_fraction,
        device_bytes=device_bytes if use_gpu else None,
        measured=tuple(measured),
        assumed=tuple(assumed),
        notes=tuple(notes),
    )


def merge_slots(slots: Sequence[ResourceSlot], *, family: str) -> ResourceSlot:
    """Collapse several families' slots into the one slot a shared pool needs (pure).

    A Ray CPU pool runs whatever lands on it — statistical cells and ML cells go through the
    same worker — so its slot has to hold the heaviest of them. Same roll-up rule
    `FamilyCost` applies across a family's models, one level out: **max per axis**,
    because the slot must fit whichever cell arrives, not the average one.

    Provenance is resolved per axis by asking *where the winning number came from*: an axis is
    ``measured`` iff a contributor that measured it supplied the max. That is the honest
    answer — a 4 GiB max taken from a real fit is measured evidence even if a lighter family
    beside it had none. What the lighter family's gap does earn is a **note**, so a reader can
    see that the pool was sized off a subset of the families that will run on it. Without that
    note "measured" would over-claim; with it, both facts are on the record.

    ``family`` is the merged label (``"statistical+ml"``). Raises on an empty sequence: a pool
    with no families is a caller bug, not a slot.
    """
    if not slots:
        raise ValueError("merge_slots needs at least one slot")

    def pick(values: list[tuple[float | None, bool]]) -> tuple[float | None, bool]:
        """The max across contributors, plus whether a *measuring* contributor supplied it."""
        present = [(value, was_measured) for value, was_measured in values if value is not None]
        if not present:
            return None, False
        best = max(value for value, _ in present)
        return best, any(was_measured for value, was_measured in present if value == best)

    def axis_of(slot: ResourceSlot, axis: str) -> tuple[float | None, bool]:
        raw = getattr(slot, axis)
        return (None if raw is None else float(raw)), axis in slot.measured

    cores, cores_measured = pick([axis_of(s, "cores") for s in slots])
    memory, memory_measured = pick([axis_of(s, "memory_bytes") for s in slots])
    fraction, fraction_measured = pick([axis_of(s, "gpu_fraction") for s in slots])

    measured: list[str] = []
    assumed: list[str] = []
    for axis, value, was_measured in (
        ("cores", cores, cores_measured),
        ("memory_bytes", memory, memory_measured),
        ("gpu_fraction", fraction, fraction_measured),
    ):
        if axis == "gpu_fraction" and value is None:
            continue  # a CPU pool has no device axis at all, measured or otherwise
        (measured if was_measured else assumed).append(axis)

    notes = [note for slot in slots for note in slot.notes]
    for axis in ("cores", "memory_bytes", "gpu_fraction"):
        if axis == "gpu_fraction" and fraction is None:
            continue
        blind = sorted(s.family for s in slots if axis not in s.measured)
        if blind and len(blind) < len(slots):
            notes.append(f"{axis} sized without a measurement for {', '.join(blind)}")

    return ResourceSlot(
        family=family,
        cores=max(1, int(cores or _DEFAULT_SLOT_CORES)),
        memory_bytes=None if memory is None else int(memory),
        gpu_fraction=fraction,
        device_bytes=next((s.device_bytes for s in slots if s.device_bytes is not None), None),
        measured=tuple(measured),
        assumed=tuple(assumed),
        notes=tuple(notes),
    )


def _resolve_gpu_fraction(
    cost: Any,
    *,
    use_gpu: bool,
    device_bytes: int | None,
    static_gpu_fraction: float | None,
    measured: list[str],
    assumed: list[str],
) -> float | None:
    """The device share one cell needs: measured, else pinned, else nominal (pure).

    Split out of `resource_slot` because it is the one axis with three sources rather
    than two, and inlining its ladder buried the other two. ``None`` when no GPU is
    provisioned — a CPU-only family must not carry a fraction, or a consumer will schedule
    against a device that isn't there.
    """
    if not use_gpu:
        return None
    slot_gpu = cost.slot_gpu_bytes if cost is not None else None
    if slot_gpu is not None and device_bytes:
        measured.append("gpu_fraction")
        return _clamp_gpu_fraction(slot_gpu / device_bytes)
    assumed.append("gpu_fraction")
    if static_gpu_fraction is not None:
        return _clamp_gpu_fraction(static_gpu_fraction)
    return _NOMINAL_GPU_FRACTION


# --- the fleet: how much hardware the load needs -------------------------------


@dataclass(frozen=True)
class UnitShape:
    """One schedulable unit of a fleet — a Ray worker node, later a Spark executor (pure).

    The runtime-neutral description of the thing slots are packed into. ``memory_bytes`` is
    the unit's *nameplate* RAM (`machine_memory_bytes`); the schedulable share is
    applied during packing, not here, so the raw number stays legible in telemetry.
    ``accelerators`` is devices per unit — ``compute.accelerator_count`` for a Ray GPU pool,
    ``0`` for a CPU pool.
    """

    cores: int
    memory_bytes: int | None = None
    accelerators: int = 0

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict for telemetry."""
        return {
            "cores": self.cores,
            "memory_bytes": self.memory_bytes,
            "accelerators": self.accelerators,
        }


@dataclass(frozen=True)
class RuntimeResourcePlan:
    """One family's sizing decision on one runtime, with the evidence attached (pure).

    Everything a runtime needs to launch the family's work, plus everything an auditor
    needs to reconstruct why: the slot and its provenance, the unit it packs into, the load
    it was sized for, and the resulting fleet bounds. Stamped whole into ``job_telemetry``.

    ``derived_units`` is the fan-out's implied unit count *after* the ceiling clamp — the
    number the fixed path provisions and the autoscaling path uses as its reference size.
    ``saturating_units`` is the count that would run every cell at once with no ceiling at
    all; the gap between the two is exactly how much the ceiling is throttling this run, and
    it is a number worth being able to read off a record rather than infer from a wall
    clock.
    """

    runtime: str  # "ray" | "serverless" | "cluster"
    family: str
    slot: ResourceSlot
    unit: UnitShape
    n_cells: int
    slots_per_unit: int  # concurrent cells one unit holds, >= 1
    derived_units: int  # units the fan-out implies, clamped into [min, max]; 0 == unused
    saturating_units: int  # units to run every cell at once, unclamped — diagnostic
    min_units: int
    max_units: int
    target_cells_per_slot: int

    @property
    def total_slots(self) -> int:
        """Concurrent cells the derived fleet can hold — ``slots_per_unit x derived_units``."""
        return self.slots_per_unit * self.derived_units

    @property
    def slots_at_ceiling(self) -> int:
        """Concurrent cells the fleet could hold if it scaled all the way to ``max_units``.

        The demand an autoscaler must actually see before it will grow to its ceiling. See
        `tasks_for_ceiling` for why that matters.
        """
        return self.slots_per_unit * self.max_units

    @property
    def task_options(self) -> dict[str, float]:
        """The Ray ``@ray.remote.options(**...)`` mapping this plan implies.

        A GPU slot requests ``num_gpus`` and lets Ray default ``num_cpus`` to 1, exactly as
        the engine did before — several cells pack onto one device by summing fractions
        against its capacity of 1.0. A CPU slot requests ``num_cpus`` explicitly. Either way
        ``memory`` is included **only when it was measured**: Ray treats it as a hard
        scheduling resource, so requesting a number nobody took could leave tasks
        permanently unschedulable.

        Only meaningful for ``runtime == "ray"``; the Spark translations emit properties,
        not options, and will carry their own accessor.
        """
        options: dict[str, float] = {}
        if self.slot.gpu_fraction is not None:
            options["num_gpus"] = self.slot.gpu_fraction
        else:
            options["num_cpus"] = self.slot.cores
        if self.slot.memory_bytes is not None:
            options["memory"] = self.slot.memory_bytes
        return options

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict of the whole decision — the record stamped into telemetry."""
        return {
            "runtime": self.runtime,
            "family": self.family,
            "n_cells": self.n_cells,
            "slot": self.slot.to_dict(),
            "unit": self.unit.to_dict(),
            "slots_per_unit": self.slots_per_unit,
            "derived_units": self.derived_units,
            "saturating_units": self.saturating_units,
            "min_units": self.min_units,
            "max_units": self.max_units,
            "target_cells_per_slot": self.target_cells_per_slot,
            "total_slots": self.total_slots,
            "slots_at_ceiling": self.slots_at_ceiling,
        }


def schedulable_memory_bytes(unit: UnitShape) -> int | None:
    """The share of a unit's RAM a scheduler will actually hand out (pure; unknown → ``None``).

    ``_SCHEDULABLE_MEMORY_FRACTION`` of nameplate. ``None`` when the unit's memory is unknown
    (an unparseable machine type), which callers must read as *no memory bound* rather than
    as a unit with no memory. Exposed because every caller that clamps a slot has to clamp
    against the same ceiling the packing arithmetic uses.
    """
    return int(unit.memory_bytes * _SCHEDULABLE_MEMORY_FRACTION) if unit.memory_bytes else None


def _memory_bound(slot: ResourceSlot, unit: UnitShape) -> int | None:
    """Cells one unit's schedulable RAM holds, or ``None`` when either side is unknown (pure)."""
    schedulable = schedulable_memory_bytes(unit)
    if schedulable is None or not slot.memory_bytes:
        return None
    return math.floor(schedulable / slot.memory_bytes)


def slots_per_unit(slot: ResourceSlot, unit: UnitShape) -> int:
    """Concurrent cells one unit holds — the min of its primary bound and its memory bound (pure).

    The primary bound is whichever resource the slot is *defined* by:

    * **GPU slot** — ``accelerators x floor(1 / gpu_fraction)``. The device is the scarce
      resource; a GPU node's cores and RAM are sized around its cards, so they usually do
      not bind first. A unit with a fraction but no accelerators holds one cell (whatever
      provisioned it believed there was a device).
    * **CPU slot** — ``floor(cores / slot.cores)``.

    Then, on **either** kind of slot, when both memory numbers are known, also
    ``floor(schedulable_memory / slot.memory_bytes)`` — taking the smaller. The primary
    bound alone is the design's formula and it silently over-packs a memory-heavy family;
    the memory bound is what stops eight 4 GiB cells landing on a 30 GiB node. It has to
    apply to the GPU slot too, because `RuntimeResourcePlan.task_options` requests
    ``memory`` alongside ``num_gpus`` and Ray enforces it: a device bound this function
    reported but Ray will not honour is a density the pool never reaches.

    Always at least 1. A slot too big for its unit has already been clamped to fit by
    `resource_slot`, so the floor here is a belt-and-braces guard against a caller
    that assembled a slot by hand.
    """
    if slot.gpu_fraction is not None:
        packed = max(1, math.floor(1.0 / slot.gpu_fraction))
        primary = max(1, unit.accelerators * packed) if unit.accelerators else 1
    else:
        primary = math.floor(unit.cores / slot.cores) if slot.cores > 0 else 1

    by_memory = _memory_bound(slot, unit)
    return max(1, primary if by_memory is None else min(primary, by_memory))


def tasks_for_ceiling(plan: RuntimeResourcePlan) -> int:
    """Concurrent tasks the fan-out must produce before the autoscaler will reach its ceiling.

    An autoscaler grows on *pending demand*. Ray adds nodes because tasks are queued and
    cannot be placed; if the run only ever submits as many tasks as the current fleet can
    hold, nothing is ever pending and the pool sits at its minimum — the "we enabled
    autoscaling and nothing scaled" failure, which looks like a platform problem and is
    actually an arithmetic one.

    So a run that wants to be able to reach ``max_units`` must split its work into at least
    ``slots_per_unit x max_units`` tasks. Callers compare this against their chunk/bucket
    count and raise it if it falls short. Returns 0 for an unused pool (no cells, no
    ceiling worth reaching).
    """
    if plan.n_cells <= 0:
        return 0
    return plan.slots_at_ceiling


def plan_resources(
    profile: ComputeProfile | None,
    family: str,
    runtime: str,
    n_cells: int,
    *,
    unit: UnitShape,
    use_gpu: bool = False,
    device_bytes: int | None = None,
    static_gpu_fraction: float | None = None,
    target_cells_per_slot: int = _DEFAULT_TARGET_CELLS_PER_SLOT,
    min_units: int = 1,
    max_units: int = 1,
) -> RuntimeResourcePlan:
    """Size one family's fleet on one runtime from its measured profile (pure).

    Three steps, each independently testable: size the slot (`resource_slot`), pack
    slots into a unit (`slots_per_unit`), then widen the fleet until the load flows
    through at the target density.

    ``derived_units = ceil(n_cells / (slots_per_unit x target_cells_per_slot))``, clamped
    into ``[min_units, max_units]`` — the same shape as the node-count arithmetic the Ray
    engine already used, now over a *measured* slot rather than an assumed one. An empty
    pool (``n_cells <= 0``) derives 0 units and is never floored to the minimum: a family
    with no work should not provision hardware.

    ``saturating_units`` answers the different question of how wide the fleet would have to
    be to run every cell simultaneously, and is deliberately left unclamped so the record
    shows when the ceiling — not the work — is what bounded the run.

    ``runtime`` is carried rather than branched on: the slot and the fleet math are the same
    for every runtime, and what differs is only how the numbers are *spelled* when handed
    over (`RuntimeResourcePlan.task_options` for Ray; ``spark.*`` properties for the
    Spark runtimes). Keeping the divergence at the edge is what makes one profile usable
    three ways.

    The slot is clamped to the unit before packing, so a family measured larger than any
    available node yields a schedulable — if inefficient — plan with the clamp recorded,
    instead of a task the scheduler will never place.
    """
    slot = resource_slot(
        profile,
        family,
        use_gpu=use_gpu,
        device_bytes=device_bytes,
        static_gpu_fraction=static_gpu_fraction,
        max_cores=unit.cores if unit.cores > 0 else None,
        max_memory_bytes=schedulable_memory_bytes(unit),
    )
    return plan_fleet(
        slot,
        runtime=runtime,
        n_cells=n_cells,
        unit=unit,
        target_cells_per_slot=target_cells_per_slot,
        min_units=min_units,
        max_units=max_units,
    )


def plan_fleet(
    slot: ResourceSlot,
    *,
    runtime: str,
    n_cells: int,
    unit: UnitShape,
    target_cells_per_slot: int = _DEFAULT_TARGET_CELLS_PER_SLOT,
    min_units: int = 1,
    max_units: int = 1,
    density: int | None = None,
) -> RuntimeResourcePlan:
    """The fleet half of `plan_resources`, over a slot the caller already sized (pure).

    Split out because a *pool* is not a family: a shared Ray CPU pool runs several families
    through one worker, so its slot comes from `merge_slots` rather than from a single
    `resource_slot` call. Both entry points then need the identical fleet arithmetic, and
    duplicating it is how the two paths would quietly drift apart.

    The slot is taken as given — a caller that assembles one by hand is responsible for
    having clamped it to the unit (`schedulable_memory_bytes` is the ceiling to clamp
    against). `slots_per_unit` still floors the density at 1, so an unclamped slot
    yields an inefficient plan rather than a stalled pool.

    ``density`` overrides the cells-per-unit figure for a runtime whose own scheduler is the
    authority on it: Serverless divides an executor's cores by ``spark.task.cpus`` and honours
    nothing else, so a fleet sized off `slots_per_unit`'s device arithmetic would be sized off a
    density the platform never grants (`spark_tasks_per_executor`). Left ``None`` — every Ray
    caller — the derivation stands.
    """
    per_unit = max(1, density) if density is not None else slots_per_unit(slot, unit)
    cells_per_unit = max(1, per_unit * max(1, target_cells_per_slot))
    saturating = math.ceil(n_cells / per_unit) if n_cells > 0 else 0
    if n_cells <= 0:
        derived = 0
    else:
        derived = max(min_units, min(math.ceil(n_cells / cells_per_unit), max_units))

    return RuntimeResourcePlan(
        runtime=runtime,
        family=slot.family,
        slot=slot,
        unit=unit,
        n_cells=max(0, n_cells),
        slots_per_unit=per_unit,
        derived_units=derived,
        saturating_units=saturating,
        min_units=min_units,
        max_units=max_units,
        target_cells_per_slot=target_cells_per_slot,
    )


# --- Serverless Spark: the legal-value tables (design 2.3a) --------------------

# Every Serverless knob picks from a list, so the translation may never emit a raw
# arithmetic result: it computes the ideal, snaps to a legal neighbour, and records both.
# An illegal pair is rejected by the service at submit — minutes after the operator walked
# away — so snapping offline is the difference between a typo and a wasted afternoon.

_SERVERLESS_CPU_CORES = (4, 8, 16)

# The GPU core set is *different*, and each entry carries a fixed device count. Cores and
# devices together are what set the per-task share: Serverless applies
# ``task.resource.gpu.amount = 1/cores`` against ``executor.resource.gpu.amount`` devices,
# so tasks-per-device is ``cores / gpus``. Choosing cores IS choosing the fraction.
_SERVERLESS_L4_CORES = (4, 8, 12, 16, 24, 48, 96)
_SERVERLESS_L4_GPUS_PER_EXECUTOR = {4: 1, 8: 1, 12: 1, 16: 1, 24: 2, 48: 4, 96: 8}

# ``(spark.executor.memory + spark.executor.memoryOverhead) / cores`` must land in this band.
_SERVERLESS_MIN_MB_PER_CORE = 1024
_SERVERLESS_MAX_MB_PER_CORE = {"standard": 7424, "premium": 24576}

# The GPU table replaces that ceiling with a *per-config maximum* on ``spark.executor.memory``
# alone, and Google publishes one worked figure rather than a table: 13384 MB at L4/4-core.
# 13384/4 = 3346 MB per core is the ratio it implies — roughly 82% of a g2-standard's 4 GiB
# per vCPU, the rest being host overhead — so that is what we extrapolate to the wider shapes.
# Extrapolated, not published: we clamp to it and say so in a note rather than quietly
# trusting it. Overshooting here is an INVALID_ARGUMENT at submit, minutes into an
# unattended run, which is the whole reason these tables exist.
_SERVERLESS_L4_MB_PER_CORE = 3346

# JVM heap per core. We do no JVM-side work — every fit runs in the Python worker, which is
# charged to memoryOverhead — so this is deliberately a floor rather than a share: enough for
# the shuffle machinery and the Arrow batches crossing the boundary, and no more.
_SPARK_JVM_MB_PER_CORE = 512

# Serverless' own PySpark default: overhead is 40% *of executor memory* (10% for JVM-only
# workloads). We can set overhead explicitly on the CPU path; on the GPU path the service
# rejects it, so there the only handle is ``spark.executor.memory`` and this ratio is what
# converts a Python-memory requirement back into the memory number we are allowed to set.
_SERVERLESS_PYSPARK_OVERHEAD_RATIO = 0.4

_SERVERLESS_MIN_EXECUTORS = 2  # platform floor; asking for 1 is rejected
_SERVERLESS_MAX_EXECUTORS = 2000

# The service default fills only 30% of the demand gap per scaling round — the exact
# signature of "we enabled autoscaling and it ramped slowly". Our fan-out is embarrassingly
# parallel and known up front, so there is nothing to ease into.
_SERVERLESS_DEFAULT_ALLOCATION_RATIO = 0.3
_SERVERLESS_FAST_ALLOCATION_RATIO = 1.0

_MIB = 1024**2

# Native thread-pool caps. A Ray task inherits ``OMP_NUM_THREADS = num_cpus`` for free; a
# Spark executor pins nothing, so N concurrent Python workers each grab the whole executor
# and the machine thrashes on N x cores threads. The profile was measured with these pinned
# to one (`profiling._pinned_intraop_threads`), so exporting them is also what makes the
# measurement describe the environment it is being used to size.
_INTRAOP_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def snap_to_legal(value: float, choices: Sequence[int], *, up: bool) -> int:
    """Nearest legal value to ``value`` in the given direction (pure).

    ``up=True`` returns the smallest choice ``>= value`` (falling back to the largest when
    the ideal is off the top of the table); ``up=False`` the largest ``<= value`` (falling
    back to the smallest). Direction is the caller's call because it encodes which way is
    *safe*, and that differs per knob: memory snaps up, tasks-per-device snaps down.
    """
    legal = sorted(choices)
    if not legal:
        raise ValueError("snap_to_legal needs at least one legal value")
    if up:
        return next((c for c in legal if c >= value), legal[-1])
    return next((c for c in reversed(legal) if c <= value), legal[0])


@dataclass(frozen=True)
class ServerlessTranslation:
    """One plan spelled as Dataproc Serverless properties, with the ideals it snapped from.

    ``properties`` is merged straight into ``RuntimeConfig.properties``. Everything beside it
    exists so an audit can answer *why*: what the arithmetic wanted (``ideal``), what the
    platform's legal-value table allowed (the properties), and where the two disagreed
    enough to matter (``notes``).
    """

    executor_cores: int
    ideal_executor_cores: float
    tasks_per_device: int | None  # concurrent cells sharing one L4; None on a CPU batch
    properties: dict[str, str]
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe record of the translation — what telemetry stamps alongside the plan."""
        return {
            "executor_cores": self.executor_cores,
            "ideal_executor_cores": self.ideal_executor_cores,
            "tasks_per_device": self.tasks_per_device,
            "properties": dict(self.properties),
            "notes": list(self.notes),
        }


def _serverless_gpu_cores(fraction: float) -> tuple[int, float, int]:
    """Legal L4 core count for a measured device fraction: (cores, ideal, tasks/device).

    ``1 / fraction`` is how many cells fit on one card. Cores are then chosen so that
    ``cores / gpus_per_executor`` does not exceed it — the largest legal shape that still
    packs the device safely, i.e. snapped **down**, because overshooting here is a device
    OOM rather than an idle core.
    """
    ideal = 1.0 / max(fraction, _MIN_GPU_FRACTION)
    fits = [c for c in _SERVERLESS_L4_CORES if c / _SERVERLESS_L4_GPUS_PER_EXECUTOR[c] <= ideal]
    cores = max(fits) if fits else min(_SERVERLESS_L4_CORES)
    return cores, ideal, cores // _SERVERLESS_L4_GPUS_PER_EXECUTOR[cores]


def spark_tasks_per_executor(slot: ResourceSlot, cores: int) -> int:
    """Cells one Serverless executor runs at once (pure) — the number Spark will really honour.

    ``spark.task.cpus`` divides an executor's cores into task slots, and that division is the
    *only* concurrency a Spark executor enforces. `slots_per_unit` answers the same question in
    the plan's vocabulary and, on a GPU slot, answers it in device terms —
    ``accelerators x floor(1 / gpu_fraction)`` — which is what the measurement *asked* for, not
    what the legal core table *granted*: `_serverless_gpu_cores` snaps down, so the granted
    packing is the smaller of the two. Sizing a fleet against the ask leaves it short of the
    hardware it will actually get, so the Serverless planner and the translation both size
    against this one number instead of each deriving their own.
    """
    return max(1, cores // max(1, min(slot.cores, cores)))


def _executor_counts(plan: RuntimeResourcePlan) -> tuple[int, int, int]:
    """``(min, initial, max)`` executors, clamped into the platform's ``[2, 2000]`` (pure).

    ``initialExecutors`` is the one that is usually left on the floor, and it is the one that
    costs wall clock: starting at 2 and ramping means the first minutes of a 100k run are
    served by two executors. We start at the derived count instead — the fan-out is known
    before the job starts, so there is nothing to discover.
    """
    lo = max(_SERVERLESS_MIN_EXECUTORS, plan.min_units)
    hi = max(lo, min(_SERVERLESS_MAX_EXECUTORS, plan.max_units))
    initial = max(lo, min(plan.derived_units or lo, hi))
    return lo, initial, hi


def translate_serverless(
    plan: RuntimeResourcePlan, *, tier: str = "standard"
) -> ServerlessTranslation:
    """Spell one `RuntimeResourcePlan` as Dataproc Serverless properties (pure).

    The same three measured numbers as the Ray translation, solved backwards. Ray is told a
    fraction directly; Serverless derives the fraction from ``spark.executor.cores``, so here
    we solve for the cores that *yield* the fraction we measured and snap to a legal value.

    Four things this sets that the submit path never has:

    1. **``executor.cores`` from evidence.** GPU: the largest legal shape whose
       tasks-per-device fits the measured footprint (`_serverless_gpu_cores`). CPU: the
       slot's core requirement snapped up to a legal value — which lands on the 4-core
       default whenever the cores axis measures 1, and says so from evidence.
    2. **Memory in ``memoryOverhead``, not ``spark.executor.memory``.** Every fit runs in the
       Python worker, outside the JVM heap, and PySpark charges that to overhead — which is
       why the service defaults it to 40% rather than 10%. Sizing ``executor.memory`` from
       ``peak_rss`` would hand the JVM memory it never touches while the Python worker got
       OOM-killed. On the **GPU** path the service rejects an explicit overhead, so the
       control is inverted: set ``executor.memory`` such that the 40% derivation lands on
       the Python memory we need — then clamp, because dividing by 0.4 puts an ordinary
       deep-learning footprint past the per-config maximum.
    3. **A warm, fast autoscaler.** ``initialExecutors`` at the derived count instead of the
       floor of 2, and ``executorAllocationRatio`` at 1.0 instead of the default 0.3.
    4. **Concurrency stated once, consistently.** ``spark.task.cpus`` (when the family really
       is threaded), the `_INTRAOP_ENV_VARS` pin, and the memory budget all derive from the
       same ``tasks_per_executor``. Ray gets the pin for free; a Spark executor pins nothing,
       so without it the sizing is sound and the run still thrashes.

    Unmeasured axes stay absent, as everywhere else: with no memory measurement the memory
    properties are simply not emitted and Serverless' own defaults apply, which is exactly
    today's behaviour. ``tier`` is ``"standard"`` or ``"premium"`` and only widens the
    legal per-core memory band.
    """
    slot = plan.slot
    notes: list[str] = []
    properties: dict[str, str] = {}

    if slot.gpu_fraction is not None:
        cores, ideal_cores, per_device = _serverless_gpu_cores(slot.gpu_fraction)
        wanted = 1.0 / max(slot.gpu_fraction, _MIN_GPU_FRACTION)
        if per_device > wanted:
            notes.append(
                f"serverless packs {per_device} cells per L4 at its smallest legal shape; "
                f"the measurement allows {wanted:.1f} — expect device pressure"
            )
    else:
        cores = snap_to_legal(slot.cores, _SERVERLESS_CPU_CORES, up=True)
        ideal_cores = float(slot.cores)
        per_device = None
    properties["spark.executor.cores"] = str(cores)
    properties["spark.driver.cores"] = str(snap_to_legal(4, _SERVERLESS_CPU_CORES, up=True))

    # Cores per task, and therefore cells running at once inside one executor. Spark's default
    # is one task per core, which is right for the single-threaded case the probe measures and
    # wrong for a genuinely threaded family: without ``spark.task.cpus`` Spark would hand that
    # family one core and then watch it spawn threads it was never given room for. These three
    # numbers have to agree — tasks, the thread pin, and the memory budgeted for them — or the
    # translation over-packs on one axis while under-packing another.
    task_cpus = max(1, min(slot.cores, cores))
    tasks_per_executor = spark_tasks_per_executor(slot, cores)
    if task_cpus > 1:
        properties["spark.task.cpus"] = str(task_cpus)
    for name in _INTRAOP_ENV_VARS:
        properties[f"spark.executorEnv.{name}"] = str(task_cpus)

    if slot.memory_bytes:
        python_mb = math.ceil(tasks_per_executor * slot.memory_bytes / _MIB)
        jvm_mb = cores * _SPARK_JVM_MB_PER_CORE
        if per_device is None:
            per_core = (python_mb + jvm_mb) / cores
            ceiling = _SERVERLESS_MAX_MB_PER_CORE.get(tier, _SERVERLESS_MAX_MB_PER_CORE["standard"])
            if per_core > ceiling:
                notes.append(
                    f"{per_core:.0f}m per core exceeds the {tier}-tier ceiling of {ceiling}m; "
                    "clamped — use premium compute or a smaller slot"
                )
                python_mb = max(1, int(cores * ceiling) - jvm_mb)
            elif per_core < _SERVERLESS_MIN_MB_PER_CORE:
                python_mb = cores * _SERVERLESS_MIN_MB_PER_CORE - jvm_mb
            properties["spark.executor.memory"] = f"{jvm_mb}m"
            properties["spark.executor.memoryOverhead"] = f"{python_mb}m"
        else:
            # GPU: overhead is service-owned at 40% of memory, so ask for the memory whose
            # 40% is the Python footprint we need, and let the service derive the rest. The
            # result is still bounded — the inversion divides by 0.4, so an ordinary
            # deep-learning footprint lands well past the per-config maximum if left raw.
            memory_mb = math.ceil(python_mb / _SERVERLESS_PYSPARK_OVERHEAD_RATIO)
            ceiling = cores * _SERVERLESS_L4_MB_PER_CORE
            if memory_mb > ceiling:
                notes.append(
                    f"{memory_mb}m exceeds the extrapolated L4 maximum of {ceiling}m at "
                    f"{cores} cores; clamped — the cells will contend for host memory"
                )
                memory_mb = ceiling
            memory_mb = max(memory_mb, cores * _SERVERLESS_MIN_MB_PER_CORE)
            properties["spark.executor.memory"] = f"{memory_mb}m"
            notes.append("gpu batch: memoryOverhead is service-owned, sized via executor.memory")
    else:
        notes.append("memory unmeasured; serverless memory defaults left in place")

    lo, initial, hi = _executor_counts(plan)
    properties["spark.dynamicAllocation.enabled"] = "true"
    properties["spark.dynamicAllocation.minExecutors"] = str(lo)
    properties["spark.dynamicAllocation.initialExecutors"] = str(initial)
    properties["spark.dynamicAllocation.maxExecutors"] = str(hi)
    ratio = _SERVERLESS_FAST_ALLOCATION_RATIO if hi > initial else None
    if ratio is not None:
        properties["spark.dynamicAllocation.executorAllocationRatio"] = str(ratio)
    if plan.max_units > _SERVERLESS_MAX_EXECUTORS:
        notes.append(f"executor ceiling clamped to the platform max of {_SERVERLESS_MAX_EXECUTORS}")

    return ServerlessTranslation(
        executor_cores=cores,
        ideal_executor_cores=ideal_cores,
        tasks_per_device=per_device,
        properties=properties,
        notes=tuple(notes) + slot.notes,
    )


def serverless_unit(cores: int, *, gpu: bool) -> UnitShape:
    """The executor a given core count buys, as a `UnitShape` (pure).

    Serverless bills the executor as a shape rather than a machine type, so memory follows
    the tier band rather than a GCE ratio: the standard-tier per-core maximum is what an
    executor can actually be given. ``accelerators`` comes from the L4 table, where the
    device count is a property of the core count and not a separate knob.
    """
    per_core = _SERVERLESS_L4_MB_PER_CORE if gpu else _SERVERLESS_MAX_MB_PER_CORE["standard"]
    return UnitShape(
        cores=cores,
        memory_bytes=cores * per_core * _MIB,
        accelerators=_SERVERLESS_L4_GPUS_PER_EXECUTOR.get(cores, 0) if gpu else 0,
    )


def plan_serverless(
    profile: ComputeProfile | None,
    families: Sequence[str],
    n_cells: int,
    *,
    gpu: bool = False,
    device_bytes: int | None = None,
    static_gpu_fraction: float | None = None,
    target_cells_per_slot: int = _DEFAULT_TARGET_CELLS_PER_SLOT,
    max_executors: int | None = None,
    tier: str = "standard",
) -> tuple[RuntimeResourcePlan, ServerlessTranslation]:
    """Size a Serverless batch and spell it as properties, in one call (pure).

    Two passes, because the executor's shape and the fleet's size each depend on the other.
    The slot fixes ``executor.cores`` on its own — that is the whole point of the inversion,
    the cores come from the measured fraction rather than from the fleet — so pass one
    translates against the *widest* legal executor purely to learn the core count, and pass two
    builds the real `serverless_unit` from it and plans the fleet properly. The same shape
    `ray_io.plan_cluster` uses for its autoscaling ceiling, and for the same reason.

    Pass one deliberately clamps against the widest shape rather than a nominal one: clamping
    to a small executor first would cap the measured slot *before* it had a chance to ask for a
    bigger executor, and the batch would then be sized to the clamp instead of to the work. The
    slot is re-derived against the chosen shape in pass two, so the plan that comes out is still
    clamped to the executor it will actually run on.

    ``max_executors`` is the operator's explicit cap (``--max-executors``). Left ``None`` the
    ceiling is the **saturating** count — enough executors to run every cell at once — rather
    than a derived guess. That asymmetry is deliberate: Serverless already defaults
    ``maxExecutors`` to 1000, so a ceiling we invent can only ever make a large run *slower*
    than leaving it alone, while a floor we raise can only make it faster.

    ``families`` is every family the batch will run, merged the same way a shared Ray pool's
    is (`merge_slots`): one executor shape has to hold whichever cell arrives.
    """
    label = "+".join(families) or "cpu"

    def sized_for(unit: UnitShape) -> ResourceSlot:
        return merge_slots(
            [
                resource_slot(
                    profile,
                    family,
                    use_gpu=gpu,
                    device_bytes=device_bytes,
                    static_gpu_fraction=static_gpu_fraction,
                    max_cores=unit.cores,
                    max_memory_bytes=schedulable_memory_bytes(unit),
                )
                for family in (families or ["cpu"])
            ],
            family=label,
        )

    widest = serverless_unit(
        (_SERVERLESS_L4_CORES if gpu else _SERVERLESS_CPU_CORES)[-1], gpu=gpu
    )
    first = translate_serverless(
        plan_fleet(sized_for(widest), runtime="serverless", n_cells=n_cells, unit=widest),
        tier=tier,
    )

    unit = serverless_unit(first.executor_cores, gpu=gpu)
    slot = sized_for(unit)
    # Density comes from the executor's task slots, not from `slots_per_unit`'s device
    # arithmetic — see `spark_tasks_per_executor`. Both the ceiling and the plan use it, so
    # the fleet and the properties describe one machine rather than two.
    per_unit = spark_tasks_per_executor(slot, unit.cores)
    ceiling = max_executors or (math.ceil(n_cells / per_unit) if n_cells > 0 else 1)
    plan = plan_fleet(
        slot,
        runtime="serverless",
        n_cells=n_cells,
        unit=unit,
        target_cells_per_slot=target_cells_per_slot,
        min_units=_SERVERLESS_MIN_EXECUTORS,
        max_units=max(_SERVERLESS_MIN_EXECUTORS, ceiling),
        density=per_unit,
    )
    return plan, translate_serverless(plan, tier=tier)


# --- Dataproc cluster: the worker is the unit, and it is billed whole ----------

# **One executor per worker, and the executor is (almost) the worker.** Everything this
# product runs on Spark happens inside the Python worker — the JVM is a shuttle for Arrow
# batches — so a second executor on the same machine buys a second JVM's overhead and splits
# the Python memory pool in two. On the GPU path it is not merely wasteful but wrong: Dataproc
# leaves YARN's GPU isolation off, so the device is not a schedulable resource and two
# executors on one worker would each believe they own the whole card.
#
# "Almost", because YARN still has to place the job's ApplicationMaster somewhere, and
# Dataproc's capacity scheduler counts vcores as well as memory (DominantResourceCalculator).
# An executor sized to the *whole* worker leaves the AM unplaceable and the job sits in
# ACCEPTED forever — a hang rather than an error, the worst shape of failure. So one core and
# a small slice of memory stay unclaimed on every worker.
_CLUSTER_AM_CORES = 1
_CLUSTER_AM_RESERVE_MB = 2048

# Dataproc's own floor for a standard cluster (HDFS wants two datanodes), and equal to
# ``dataproc_cluster._DEFAULT_WORKER_COUNT`` — so a run small enough to derive nothing keeps
# exactly the cluster it has today.
_CLUSTER_MIN_WORKERS = 2

# The ceiling on a *derived* worker count. Unlike a Serverless batch — billed per executor
# second, so an unused ceiling is free — a cluster's workers are billed from create to delete
# whether or not a task ever lands on one. A fan-out big enough to want hundreds of workers is
# therefore a spend decision, not an arithmetic one: we derive the number, clamp it here, and
# say in the notes what the arithmetic actually wanted. Operators raise it explicitly
# (``max_workers``), which is the same shape as ``--max-executors`` on the batch path.
_CLUSTER_MAX_WORKERS = 10


@dataclass(frozen=True)
class ClusterTranslation:
    """One plan spelled as Dataproc **cluster** job properties plus a worker count.

    The cluster analog of `ServerlessTranslation`, and it carries one thing more: on a batch
    the fleet *is* the properties, while a cluster's ceiling is a physical worker count fixed
    at create. So ``worker_count`` is a separate output that the create path consumes and
    ``properties`` is what the job carries.

    ``ideal_workers`` is what the fan-out asked for before `_CLUSTER_MAX_WORKERS` (or the
    operator's cap) applied. When the two differ the run is throttled by a spend ceiling
    rather than by its own shape, and that is worth being able to read off a record.
    """

    executor_cores: int
    tasks_per_executor: int
    worker_count: int
    ideal_workers: int
    properties: dict[str, str]
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe record of the translation — what telemetry stamps alongside the plan."""
        return {
            "executor_cores": self.executor_cores,
            "tasks_per_executor": self.tasks_per_executor,
            "worker_count": self.worker_count,
            "ideal_workers": self.ideal_workers,
            "properties": dict(self.properties),
            "notes": list(self.notes),
        }


def cluster_unit(machine_type: str, *, accelerators: int = 0) -> UnitShape:
    """The worker a GCE machine type buys, as a `UnitShape` (pure).

    Straight from the machine name, because on a cluster that name *is* the knob — there is no
    legal-value table to snap to as there is on Serverless, and no tier band: an
    ``n1-standard-8`` worker has eight cores and 30 GiB, and the sizing question is only how to
    divide them. ``accelerators`` is devices per worker (`AcceleratorConfig.accelerator_count`,
    1 on every GPU cluster we build today).
    """
    return UnitShape(
        cores=machine_cores(machine_type),
        memory_bytes=machine_memory_bytes(machine_type),
        accelerators=accelerators,
    )


def _cluster_executor(unit: UnitShape) -> tuple[int, int, int | None]:
    """``(cores, JVM MiB, python MiB)`` for the one executor a worker hosts (pure).

    The worker minus the AM's reserve, split into the heap we barely use and the Python pool
    that does all the work. The Python figure is ``None`` when the worker's machine type did
    not parse — the same *unknown* `machine_memory_bytes` reports, carried through so the
    translation emits no memory property at all rather than one derived from a guess.
    """
    cores = max(1, unit.cores - _CLUSTER_AM_CORES)
    schedulable = schedulable_memory_bytes(unit)
    if schedulable is None:
        return cores, 0, None
    container_mb = max(1, schedulable // _MIB - _CLUSTER_AM_RESERVE_MB)
    jvm_mb = min(cores * _SPARK_JVM_MB_PER_CORE, container_mb)
    return cores, jvm_mb, max(1, container_mb - jvm_mb)


def _cluster_density(slot: ResourceSlot, unit: UnitShape) -> tuple[int, int, int]:
    """``(cores, spark.task.cpus, cells at once)`` for a worker's executor (pure).

    The one place the cluster's density is decided, because the planner and the translation
    both need it and a fleet sized off a density the properties do not spell is the exact bug
    `spark_tasks_per_executor` exists to prevent on the batch path.
    """
    cores, _jvm_mb, pool_mb = _cluster_executor(unit)
    task_cpus = _cluster_task_cpus(slot, unit, cores, pool_mb)
    return cores, task_cpus, max(1, cores // task_cpus)


def _cluster_task_cpus(
    slot: ResourceSlot, unit: UnitShape, cores: int, python_mb: int | None
) -> int:
    """``spark.task.cpus`` for a whole-worker executor — the only lever on its density (pure).

    On Serverless, choosing ``executor.cores`` *is* choosing the packing, so ``task.cpus`` only
    has to widen a genuinely threaded family. Here the cores are fixed by the machine, so this
    one integer carries every bound the slot has: a GPU cluster's card, a memory-heavy family's
    footprint, and the family's own thread count. Left at the Serverless rule
    (``min(slot.cores, cores)``) a 7-core executor would run seven NeuralProphet cells on one
    T4 whatever fraction was measured, because nothing else on a cluster limits them.

    So: compute how many cells the executor may hold (the device bound, the memory bound, and
    the trivial core bound, whichever is smallest), then pick the **narrowest** ``task.cpus``
    whose ``cores // task.cpus`` does not exceed it. Narrowest rather than
    ``ceil(cores / limit)`` because integer division floors: at 7 cores and a limit of 2,
    ``ceil`` gives 4 and one task, while stepping up from 1 finds 3 and the two tasks the
    limit actually allowed. Snapping **down** past the limit is never considered — an extra
    cell on a full device is an OOM, an idle core is a rounding loss.
    """
    limit = cores
    if slot.gpu_fraction is not None:
        packed = max(1, math.floor(1.0 / slot.gpu_fraction))
        limit = min(limit, max(1, unit.accelerators * packed) if unit.accelerators else 1)
    if slot.memory_bytes and python_mb:
        limit = min(limit, math.floor(python_mb * _MIB / slot.memory_bytes))
    limit = max(1, limit)
    narrowest = next(c for c in range(1, cores + 1) if cores // c <= limit)
    return max(1, min(cores, max(slot.cores, narrowest)))


def translate_cluster(plan: RuntimeResourcePlan) -> ClusterTranslation:
    """Spell one `RuntimeResourcePlan` as Dataproc **cluster** job properties (pure).

    The third algebra over the same three measured numbers. The Serverless inversion does not
    apply here — a cluster's executor shape is not chosen from a legal table, it is carved out
    of a machine we already picked — so this is the simplest of the three and differs from
    `translate_serverless` in exactly four places:

    1. **Cores come from the worker, not from the slot.** One executor per worker (see
       `_CLUSTER_AM_CORES` for why "almost"), so ``spark.executor.cores`` is a property of
       the machine type. The slot still decides ``spark.task.cpus``, and therefore how many
       cells run at once inside that executor — the number that actually matters.
    2. **Memory is always stated, measured or not.** Everywhere else an unmeasured axis
       requests nothing, and that rule is right when the platform's own default is shaped for
       the request we are making. Here it is not: Dataproc bakes ``spark.executor.memory``
       into the cluster's ``spark-defaults`` at *create*, sized for the default executor
       shape. Change the shape at job level and leave memory alone and the pairing is stale —
       a 7-core executor holding a 4-core executor's heap. So the container is sized from the
       worker (a machine fact, not a measurement) and the *split* between JVM heap and Python
       overhead is what measurement refines.
    3. **The GPU path uses ``memoryOverhead`` like everything else.** Serverless forbids
       setting it on a GPU batch and forces the awkward ``executor.memory`` inversion; a
       cluster has no such restriction, so the GPU and CPU paths are the same code.
    4. **The fleet is workers, not executors.** ``dynamicAllocation`` still runs (it is what
       lets a job release executors it is not using) but ``initialExecutors`` equals
       ``maxExecutors``: the workers are already paid for, so there is no ramp worth taking.

    **What is deliberately not emitted:** ``spark.executor.resource.gpu.amount`` and
    ``spark.task.resource.gpu.amount``, the design's GPU-aware scheduling pair. Spark validates
    an executor resource request against what the NodeManager advertises, and Dataproc does not
    turn YARN GPU isolation on — no ``yarn.resource-types``, no resource plugin, no discovery
    script. Emitting the Spark half alone fails every executor at launch. One executor per
    worker already gives the device a single owner, and ``spark.task.cpus`` is what shares it;
    turning the YARN half on is a cluster-create change that has to be proven live, not an
    offline one.
    """
    slot = plan.slot
    unit = plan.unit
    notes: list[str] = []
    properties: dict[str, str] = {}

    _cores, jvm_mb, pool_mb = _cluster_executor(unit)
    cores, task_cpus, tasks = _cluster_density(slot, unit)
    properties["spark.executor.cores"] = str(cores)

    # Tasks, the native-thread pin, and the memory budgeted for them all derive from this one
    # number, exactly as on the Serverless side — otherwise the executor is over-packed on one
    # axis while under-packed on another.
    if task_cpus > 1:
        properties["spark.task.cpus"] = str(task_cpus)
    for name in _INTRAOP_ENV_VARS:
        properties[f"spark.executorEnv.{name}"] = str(task_cpus)

    if pool_mb is None:
        notes.append("worker machine type unparseable; cluster memory defaults left in place")
    else:
        python_mb = pool_mb
        if slot.memory_bytes:
            wanted = math.ceil(tasks * slot.memory_bytes / _MIB)
            if wanted > python_mb:
                notes.append(
                    f"one cell needs {slot.memory_bytes} bytes and the worker leaves "
                    f"{python_mb}m for {tasks} of them — clamped, expect host memory pressure"
                )
            else:
                python_mb = max(1, wanted)
        properties["spark.executor.memory"] = f"{jvm_mb}m"
        properties["spark.executor.memoryOverhead"] = f"{python_mb}m"

    workers = plan.derived_units or _CLUSTER_MIN_WORKERS
    ideal = (
        math.ceil(plan.n_cells / max(1, plan.slots_per_unit * plan.target_cells_per_slot))
        if plan.n_cells > 0
        else 0
    )
    if ideal > plan.max_units:
        notes.append(
            f"the fan-out wants {ideal} workers; clamped to {plan.max_units} because a "
            "cluster's workers bill whether or not they are busy — raise max_workers to spend it"
        )
    properties["spark.dynamicAllocation.enabled"] = "true"
    properties["spark.dynamicAllocation.minExecutors"] = "1"
    properties["spark.dynamicAllocation.initialExecutors"] = str(workers)
    properties["spark.dynamicAllocation.maxExecutors"] = str(workers)

    if unit.accelerators:
        notes.append(
            "gpu cluster: spark.{executor,task}.resource.gpu.amount withheld — dataproc leaves "
            "yarn gpu isolation off, so a resource request the nodemanager does not advertise "
            "fails every executor at launch; the single executor owns the device instead"
        )

    return ClusterTranslation(
        executor_cores=cores,
        tasks_per_executor=tasks,
        worker_count=workers,
        ideal_workers=ideal,
        properties=properties,
        notes=tuple(notes) + slot.notes,
    )


def plan_dataproc_cluster(
    profile: ComputeProfile | None,
    families: Sequence[str],
    n_cells: int,
    *,
    machine_type: str,
    accelerators: int = 0,
    gpu: bool = False,
    device_bytes: int | None = None,
    static_gpu_fraction: float | None = None,
    target_cells_per_slot: int = _DEFAULT_TARGET_CELLS_PER_SLOT,
    max_workers: int | None = None,
) -> tuple[RuntimeResourcePlan, ClusterTranslation]:
    """Size a Dataproc cluster and spell it as job properties, in one call (pure).

    One pass, where `plan_serverless` needs two: there is no chicken-and-egg to break, because
    the executor's shape follows from the worker's machine type and the caller already chose
    that (the accelerator decides it — a T4 rides an ``n1-standard-8``, an L4 comes bundled in
    a ``g2-standard-8``). So the shape is known before the slot is sized, and the slot can be
    clamped to the executor it will actually run in on the first and only attempt.

    ``n_cells`` is the **task** count, not the cell count — the bucket count the engine will
    fan out into (`engines.spark_io.default_bucket_count`), for the same reason the batch path
    sizes against it: each bucket holds `compute.bucket_target_cells` cells that run
    sequentially inside one pandas frame, so sizing against cells would ask for that many times
    more workers than the fan-out can keep busy.

    ``max_workers`` is the operator's explicit ceiling. Left ``None`` it is
    `_CLUSTER_MAX_WORKERS` — a real clamp rather than the batch path's "saturating count",
    because the two runtimes have opposite cost shapes: an unused Serverless executor costs
    nothing, an unused worker costs a VM-hour.

    ``families`` is every family whose work lands on this cluster, merged the way a shared Ray
    pool's is (`merge_slots`): one executor shape has to hold whichever cell arrives.
    """
    label = "+".join(families) or "cpu"
    unit = cluster_unit(machine_type, accelerators=accelerators)
    cores, _jvm_mb, pool_mb = _cluster_executor(unit)

    slot = merge_slots(
        [
            resource_slot(
                profile,
                family,
                use_gpu=gpu,
                device_bytes=device_bytes,
                static_gpu_fraction=static_gpu_fraction,
                max_cores=cores,
                max_memory_bytes=pool_mb * _MIB if pool_mb is not None else None,
            )
            for family in (families or ["cpu"])
        ],
        family=label,
    )
    # The density the properties will actually spell (`_cluster_density`), not the one the
    # device arithmetic asks for — the fleet and the job have to describe one machine. One
    # executor per worker, so a worker's density *is* an executor's density.
    _cores, _task_cpus, per_unit = _cluster_density(slot, unit)
    ceiling = max(_CLUSTER_MIN_WORKERS, max_workers or _CLUSTER_MAX_WORKERS)
    plan = plan_fleet(
        slot,
        runtime="cluster",
        n_cells=n_cells,
        unit=unit,
        target_cells_per_slot=target_cells_per_slot,
        min_units=_CLUSTER_MIN_WORKERS,
        max_units=ceiling,
        density=per_unit,
    )
    return plan, translate_cluster(plan)
