"""How much hardware does this load need? — the second layer, wrapping the first.

`slot` sizes one cell; this sizes the fleet that runs all of them. Slots per unit, units for
the fan-out, the autoscaling ``[min, max]``. Still runtime-neutral: a caller supplies the
shape of one schedulable unit (`UnitShape` — a Ray worker node, a Spark executor, a Dataproc
worker) and gets back a `RuntimeResourcePlan` carrying both the decision and its evidence.

**Three axes bound a density, and the smallest one wins.** ``slots_per_unit = unit.cores //
slot.cores`` is the design's formula and it is right up to the point where the cells do not
fit: eight NeuralProphet cells at 4 GiB each do not run on a 30 GiB node no matter how many
cores it has, and a node's *card* stops being the scarce resource as soon as the fraction is
small enough that its cores run out first. So the density is ``min(device, cores, memory)``
over whichever of the three have a basis — the same rule the Serverless translation states
explicitly (``floor(usable_python_mem / peak_rss)``), applied on the runtime that had been
taking one axis at a time.

The device axis in particular used to stand alone: a GPU slot's density was
``accelerators x floor(1 / gpu_fraction)`` with **no core term at all**, so an
``n1-standard-8`` with one T4 at the 0.1 fraction floor reported ten concurrent cells onto
seven usable cores. Ten cells were never going to run; the arithmetic just never said so.

**Nameplate is not schedulable, on either axis.** `schedulable_memory_bytes` takes the
plasma store and the OS off the RAM; `schedulable_cores` takes `_RESERVED_CORES_PER_UNIT`
off the vCPUs for the node's own agents. Every bound here goes through one of the two.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .catalog import (
    _DEFAULT_TARGET_CELLS_PER_SLOT,
    _MAX_SLOT_MEMORY_FRACTION,
    _RESERVED_CORES_PER_UNIT,
    _SCHEDULABLE_MEMORY_FRACTION,
    intraop_env_vars,
)
from .slot import ResourceSlot, resource_slot

if TYPE_CHECKING:
    from ..profiling.cost import ComputeProfile

_GIB = 1024**3


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
    def binding_axis(self) -> str:
        """Which resource holds this pool's density down — ``memory``/``device``/``cores``.

        ``"scheduler"`` when the stored density did not come from this module's arithmetic at
        all (`plan_fleet`'s ``density`` override — Serverless), because then no axis here is
        the one that decided.
        """
        if slots_per_unit(self.slot, self.unit) != self.slots_per_unit:
            return "scheduler"
        return _binding_axis(self.slot, self.unit)

    @property
    def density_note(self) -> str | None:
        """One line, only when *memory* is what holds the density down (pure; else ``None``).

        A pool bound by cores is a pool doing the obvious thing, and saying so on every run is
        noise. A pool bound by memory is the one that goes wrong silently: `slots_per_unit`
        takes the min of the two bounds and returns a single integer, so a slot sized at 97% of
        a node collapses the pool to one cell per unit and reports it as a density with no
        indication that the other axis had room to spare. That is exactly what a live 10,000-
        series Ray run did — one cell per node, seven of eight cores and 90% of every T4 idle,
        and nothing in the record said why (see `profiling.source._without_driver_rss` for the
        cause).

        So the memory axis has to speak up. It names both sides of the comparison and what the
        *tightest other* axis would have packed, which is the difference between "this family is
        genuinely memory-heavy, buy bigger nodes" and "the slot is mis-measured."
        """
        schedulable = schedulable_memory_bytes(self.unit)
        if schedulable is None or not self.slot.memory_bytes:
            return None
        if self.binding_axis != "memory":
            return None
        others = {a: v for a, v in _bounds(self.slot, self.unit).items() if a != "memory"}
        axis, packed = _tightest(others, _defining_axis(self.slot))
        return (
            f"memory is holding this {self.family} pool to {self.slots_per_unit} concurrent "
            f"cells per unit: {self.slot.memory_bytes / _GIB:.2f} GiB per cell against "
            f"{schedulable / _GIB:.2f} GiB schedulable, where "
            f"{'devices' if axis == 'device' else axis} alone would have packed {packed}."
        )

    @property
    def assigned_cores(self) -> int:
        """Cores Ray will actually hand this plan's task — what its thread pools may use.

        Not always ``slot.cores``. A GPU task requests ``num_gpus`` and no ``num_cpus``, so Ray
        assigns it the default of one core however wide the slot's core figure happens to be.
        Capping the thread pools at what the task is *given* rather than at what was planned for
        it is what keeps this number equal to the ``OMP_NUM_THREADS`` Ray sets beside it.
        """
        return 1 if self.slot.gpu_fraction is not None else self.slot.cores

    @property
    def task_options(self) -> dict[str, Any]:
        """The Ray ``@ray.remote.options(**...)`` mapping this plan implies.

        A GPU slot requests ``num_gpus`` and lets Ray default ``num_cpus`` to 1, exactly as
        the engine did before — several cells pack onto one device by summing fractions
        against its capacity of 1.0. A CPU slot requests ``num_cpus`` explicitly. Either way
        ``memory`` is included **only when it was measured**: Ray treats it as a hard
        scheduling resource, so requesting a number nobody took could leave tasks
        permanently unschedulable.

        **The task also carries a per-pool thread cap, and it has to ride here rather than on the
        job.** Ray already sets ``OMP_NUM_THREADS`` per task, to that task's assigned cores, but it
        sets only that one — so OpenBLAS, MKL, NumExpr and vecLib fall back to counting the
        machine's cores and each cell claims the whole node. Seven cells packed onto seven cores
        then run forty-nine threads against seven, which costs throughput and, worse, makes
        ``cpu_seconds / fit_seconds`` report clean occupancy on a fleet that is thrashing. The
        remedy has to reach the worker *before* it imports numpy, because these pools are sized at
        import and cannot be resized afterwards, which rules out setting them inside the cell.

        A task-level ``runtime_env`` is the seam that works: Ray merges its ``env_vars`` over the
        job's per key and inherits every other field, so ``working_dir`` and the ``uv`` install are
        untouched and neither is re-staged. Putting it here rather than in
        `code_delivery.build_runtime_env` is what lets the CPU and GPU pools carry *different*
        caps, which a single job-level value cannot express — the cost is that the two pools stop
        sharing worker processes.

        Only meaningful for ``runtime == "ray"``; the Spark translations emit properties,
        not options, and will carry their own accessor.
        """
        options: dict[str, Any] = {}
        if self.slot.gpu_fraction is not None:
            options["num_gpus"] = self.slot.gpu_fraction
        else:
            options["num_cpus"] = self.slot.cores
        if self.slot.memory_bytes is not None:
            options["memory"] = self.slot.memory_bytes
        options["runtime_env"] = {
            "env_vars": intraop_env_vars(self.assigned_cores, include_omp=False)
        }
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
            "binding_axis": self.binding_axis,
            # What the runtime was actually asked for, beside what was decided. `binding_axis`
            # says which bound won; these two say what that turned into and, when memory won,
            # how much of another axis it left on the floor. Both were previously derivable only
            # by re-running the arithmetic against a plan nobody had kept.
            "task_options": self.task_options,
            "density_note": self.density_note,
        }


def schedulable_memory_bytes(unit: UnitShape) -> int | None:
    """The share of a unit's RAM a scheduler will actually hand out (pure; unknown → ``None``).

    ``_SCHEDULABLE_MEMORY_FRACTION`` of nameplate. ``None`` when the unit's memory is unknown
    (an unparseable machine type), which callers must read as *no memory bound* rather than
    as a unit with no memory. Exposed because every caller that clamps a slot has to clamp
    against the same ceiling the packing arithmetic uses.
    """
    return int(unit.memory_bytes * _SCHEDULABLE_MEMORY_FRACTION) if unit.memory_bytes else None


def max_slot_memory_bytes(unit: UnitShape) -> int | None:
    """The largest memory ask one slot may make of a unit (pure; unknown → ``None``).

    `schedulable_memory_bytes` with `_MAX_SLOT_MEMORY_FRACTION` of headroom on top, and the
    distinction between the two is the difference between a fleet that runs and one that hangs.
    The schedulable figure is an *estimate* of the scheduler's ceiling, derived from a machine
    type's nameplate RAM; the scheduler's real ceiling is derived from what the container's OS
    reports, which is lower. Clamping a slot to exactly the estimate therefore produces a task
    request that is plausibly just above what any node can offer, and an unplaceable Ray task does
    not fail — it queues forever while the autoscaler explains, once per second, that no node type
    can fulfil it.

    Use this wherever a slot is being *clamped to fit*; use `schedulable_memory_bytes` where a
    unit's capacity is being *divided up* (`_memory_bound`, container sizing). Same node, two
    different questions.
    """
    schedulable = schedulable_memory_bytes(unit)
    return int(schedulable * _MAX_SLOT_MEMORY_FRACTION) if schedulable else None


def schedulable_cores(unit: UnitShape) -> int:
    """vCPUs a unit will actually run cells on — nameplate minus the reserve (pure; >= 1).

    The core-axis twin of `schedulable_memory_bytes`, and it exists for the same reason: the
    number on the machine type is what Google bills, not what the scheduler has left to give
    once the node's own agents are running. See `_RESERVED_CORES_PER_UNIT` for why the reserve
    is a flat core rather than a share.

    Floored at 1 so a single-core unit still holds a cell. A unit that holds zero cells is not
    a conservative plan, it is a pool that never starts.
    """
    return max(1, unit.cores - _RESERVED_CORES_PER_UNIT)


def _memory_bound(slot: ResourceSlot, unit: UnitShape) -> int | None:
    """Cells one unit's schedulable RAM holds, or ``None`` when either side is unknown (pure)."""
    schedulable = schedulable_memory_bytes(unit)
    if schedulable is None or not slot.memory_bytes:
        return None
    return math.floor(schedulable / slot.memory_bytes)


def _core_bound(slot: ResourceSlot, unit: UnitShape) -> int:
    """Cells one unit's schedulable cores hold — ``floor(cores / slot.cores)`` (pure; >= 1)."""
    if slot.cores <= 0:
        return 1
    return max(1, math.floor(schedulable_cores(unit) / slot.cores))


def _device_bound(slot: ResourceSlot, unit: UnitShape) -> int | None:
    """Cells one unit's accelerators hold, or ``None`` when the slot has no device axis (pure).

    ``accelerators x floor(1 / gpu_fraction)``: Ray schedules a fractional device by summing
    each task's ``num_gpus`` against a capacity of 1.0 per card. A slot carrying a fraction on
    a unit with **no** accelerators holds one cell — whatever provisioned it believed there was
    a device, and refusing to schedule is worse than packing conservatively.
    """
    if slot.gpu_fraction is None:
        return None
    packed = max(1, math.floor(1.0 / slot.gpu_fraction))
    return max(1, unit.accelerators * packed) if unit.accelerators else 1


def _bounds(slot: ResourceSlot, unit: UnitShape) -> dict[str, int]:
    """Every axis that has a basis, as cells-per-unit, keyed by axis name (pure).

    Insertion order is ``device``, ``cores``, ``memory`` — scarcest-first by convention, and it
    is what breaks a tie between two axes that are *both* non-defining.
    """
    bounds: dict[str, int] = {}
    device = _device_bound(slot, unit)
    if device is not None:
        bounds["device"] = device
    bounds["cores"] = _core_bound(slot, unit)
    memory = _memory_bound(slot, unit)
    if memory is not None:
        bounds["memory"] = memory
    return bounds


def _defining_axis(slot: ResourceSlot) -> str:
    """The axis the slot is *defined* by — ``device`` when it carries a fraction, else ``cores``."""
    return "device" if slot.gpu_fraction is not None else "cores"


def _tightest(bounds: dict[str, int], defining: str) -> tuple[str, int]:
    """The axis holding the density down, and by how much (pure).

    Ties go to the slot's **defining** axis. That is a reporting rule, not an arithmetic one —
    the number is the same either way — and it is the honest attribution: when a GPU slot's card
    and its cores both allow seven cells, the pool is a GPU pool that happens to be balanced,
    not a pool that discovered it was core-bound. Reporting the incidental axis would send an
    operator to widen the machine type when the fraction is what moves the density.
    """
    smallest = min(bounds.values())
    if bounds.get(defining) == smallest:
        return defining, smallest
    return next((axis, value) for axis, value in bounds.items() if value == smallest)


def _binding_axis(slot: ResourceSlot, unit: UnitShape) -> str:
    """Which of the three bounds in `slots_per_unit` is the one that decided (pure)."""
    return _tightest(_bounds(slot, unit), _defining_axis(slot))[0]


def slots_per_unit(slot: ResourceSlot, unit: UnitShape) -> int:
    """Concurrent cells one unit holds — the smallest of its three bounds (pure).

    * **device** — ``accelerators x floor(1 / gpu_fraction)``, only when the slot carries a
      fraction. A GPU node's cores and RAM are sized around its cards, so this usually binds
      first; "usually" is exactly why it cannot be the only term.
    * **cores** — ``floor(schedulable_cores / slot.cores)``, on **every** slot including a GPU
      one. A cell needs a core to run on whether or not it also needs a card, and the card does
      not supply one: an ``n1-standard-8`` + 1 T4 at the 0.1 fraction floor is ten cells by the
      device bound and seven by this one, and seven is the number the node can actually run.
    * **memory** — ``floor(schedulable_memory / slot.memory_bytes)``, when both sides are known.
      What stops eight 4 GiB cells landing on a 30 GiB node. It applies to a GPU slot too,
      because `RuntimeResourcePlan.task_options` requests ``memory`` alongside ``num_gpus`` and
      Ray enforces it: a density this function reports but Ray will not honour is a density the
      pool never reaches.

    An axis with no basis is *absent*, not zero — an unmeasured memory footprint must not shrink
    a fleet it knows nothing about, which is the property that keeps turning the profiler on from
    ever making a run worse.

    Always at least 1. A slot too big for its unit has already been clamped to fit by
    `resource_slot`, so the floor here is a belt-and-braces guard against a caller
    that assembled a slot by hand.
    """
    return max(1, min(_bounds(slot, unit).values()))


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
        max_memory_bytes=max_slot_memory_bytes(unit),
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
