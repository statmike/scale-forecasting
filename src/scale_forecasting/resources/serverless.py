"""Serverless Spark — the arithmetic, snapped to the knobs the service will accept.

The first of the two Spark translators (`cluster` is the other; they are peers and share
their platform facts through `catalog`, never through each other).

**The knobs are discrete.** Serverless takes ``executor.cores`` from ``{4, 8, 16}`` — a
different set again on GPU — and rejects anything else *at submit*, minutes after the
operator walked away. So this module never emits a raw arithmetic result: it computes the
ideal, snaps to a legal neighbour in whichever direction is safe for that knob (memory up,
tasks-per-device down), and keeps **both** numbers, so an audit can see what the arithmetic
wanted and what the platform allowed.

It also exports the thread-pin caps itself. A Ray task inherits ``OMP_NUM_THREADS`` for free;
a Spark executor pins nothing, so N concurrent Python workers each grab the whole executor.
Setting the caps is what makes a profile measured under a pin describe the executor it is
being used to size.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .catalog import (
    _DEFAULT_TARGET_CELLS_PER_SLOT,
    _INTRAOP_ENV_VARS,
    _MIB,
    _MIN_GPU_FRACTION,
    _SPARK_JVM_MB_PER_CORE,
)
from .fleet import RuntimeResourcePlan, UnitShape, max_slot_memory_bytes, plan_fleet
from .slot import ResourceSlot, merge_slots, resource_slot

if TYPE_CHECKING:
    from ..profiling.cost import ComputeProfile


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
    plan: RuntimeResourcePlan, *, tier: str = "standard", pin_threads: bool = True
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
    if pin_threads:
        for name in _INTRAOP_ENV_VARS:
            properties[f"spark.executorEnv.{name}"] = str(task_cpus)
    else:
        notes.append(
            "native thread pools left uncapped so effective_cores can be measured; "
            "executors are deliberately oversubscribed — do not size a real run from this shape"
        )

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
    pin_threads: bool = True,
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
                    max_memory_bytes=max_slot_memory_bytes(unit),
                )
                for family in (families or ["cpu"])
            ],
            family=label,
        )

    widest = serverless_unit((_SERVERLESS_L4_CORES if gpu else _SERVERLESS_CPU_CORES)[-1], gpu=gpu)
    first = translate_serverless(
        plan_fleet(sized_for(widest), runtime="serverless", n_cells=n_cells, unit=widest),
        tier=tier,
        pin_threads=pin_threads,
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
    return plan, translate_serverless(plan, tier=tier, pin_threads=pin_threads)
