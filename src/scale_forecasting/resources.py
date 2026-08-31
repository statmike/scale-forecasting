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
environment and each one grabs the whole machine. That is a real oversubscription bug and
it belongs to the Serverless translation, not here.)

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
``resource_slot``, ``plan_resources``, ``machine_memory_bytes``, ``tasks_for_ceiling``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .profiling import ComputeProfile

__all__ = [
    "ResourceSlot",
    "RuntimeResourcePlan",
    "UnitShape",
    "machine_memory_bytes",
    "plan_resources",
    "resource_slot",
    "slots_per_unit",
    "tasks_for_ceiling",
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

# Share of a node's RAM that is actually schedulable. Ray reserves ~30% of available
# memory for the plasma object store by default and subtracts it from the node's ``memory``
# resource, so sizing against the machine's nameplate RAM over-packs by roughly that much.
# Also absorbs the OS and the Ray runtime itself.
_SCHEDULABLE_MEMORY_FRACTION = 0.7

_GIB = 1024**3


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
    match = re.match(r"^([a-z0-9]+-[a-z]+)-(\d+)$", machine_type)
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


def slots_per_unit(slot: ResourceSlot, unit: UnitShape) -> int:
    """Concurrent cells one unit holds — the min of its core bound and its memory bound (pure).

    * **GPU slot** — ``accelerators x floor(1 / gpu_fraction)``. The device is the scarce
      resource; a GPU node's cores and RAM are sized around its cards, so they do not bind
      first. A unit with a fraction but no accelerators holds one cell (whatever provisioned
      it believed there was a device).
    * **CPU slot** — ``floor(cores / slot.cores)``, and when both memory numbers are known
      also ``floor(schedulable_memory / slot.memory_bytes)``, taking the smaller. Cores
      alone is the design's formula and it silently over-packs a memory-heavy family; the
      memory bound is what stops eight 4 GiB cells being scheduled onto a 30 GiB node.

    Always at least 1. A slot too big for its unit has already been clamped to fit by
    `resource_slot`, so the floor here is a belt-and-braces guard against a caller
    that assembled a slot by hand.
    """
    if slot.gpu_fraction is not None:
        packed = max(1, math.floor(1.0 / slot.gpu_fraction))
        return max(1, unit.accelerators * packed) if unit.accelerators else 1

    by_cores = math.floor(unit.cores / slot.cores) if slot.cores > 0 else 1
    bounds = [by_cores]
    if unit.memory_bytes and slot.memory_bytes:
        schedulable = unit.memory_bytes * _SCHEDULABLE_MEMORY_FRACTION
        bounds.append(math.floor(schedulable / slot.memory_bytes))
    return max(1, min(bounds))


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
    schedulable_memory = (
        int(unit.memory_bytes * _SCHEDULABLE_MEMORY_FRACTION) if unit.memory_bytes else None
    )
    slot = resource_slot(
        profile,
        family,
        use_gpu=use_gpu,
        device_bytes=device_bytes,
        static_gpu_fraction=static_gpu_fraction,
        max_cores=unit.cores if unit.cores > 0 else None,
        max_memory_bytes=schedulable_memory,
    )
    per_unit = slots_per_unit(slot, unit)

    cells_per_unit = max(1, per_unit * max(1, target_cells_per_slot))
    saturating = math.ceil(n_cells / per_unit) if n_cells > 0 else 0
    if n_cells <= 0:
        derived = 0
    else:
        derived = max(min_units, min(math.ceil(n_cells / cells_per_unit), max_units))

    return RuntimeResourcePlan(
        runtime=runtime,
        family=family,
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
