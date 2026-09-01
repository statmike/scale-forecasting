"""How much hardware does this load need? — the second layer, wrapping the first.

`slot` sizes one cell; this sizes the fleet that runs all of them. Slots per unit, units for
the fan-out, the autoscaling ``[min, max]``. Still runtime-neutral: a caller supplies the
shape of one schedulable unit (`UnitShape` — a Ray worker node, a Spark executor, a Dataproc
worker) and gets back a `RuntimeResourcePlan` carrying both the decision and its evidence.

**Density is bounded by memory, not only by cores.** ``slots_per_unit = unit.cores //
slot.cores`` is the design's formula and it is right up to the point where the cells do not
fit: eight NeuralProphet cells at 4 GiB each do not run on a 30 GiB node no matter how many
cores it has. When both the unit's memory and the slot's memory are known the density takes
the **min** of the two bounds — the same rule the Serverless translation states explicitly
(``floor(usable_python_mem / peak_rss)``), applied on the runtime that had been ignoring it.

Nameplate RAM is not schedulable RAM, which is why `schedulable_memory_bytes` exists and why
every memory bound here goes through it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .catalog import _DEFAULT_TARGET_CELLS_PER_SLOT, _SCHEDULABLE_MEMORY_FRACTION
from .slot import ResourceSlot, resource_slot

if TYPE_CHECKING:
    from ..profiling.cost import ComputeProfile


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
