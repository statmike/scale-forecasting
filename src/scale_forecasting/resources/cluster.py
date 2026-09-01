"""Dataproc cluster — the worker is the unit, and it is billed whole.

The second Spark translator, and the differences from `serverless` are all consequences of
one fact: here we choose the machine, and we pay for the whole of it whether or not the
arithmetic filled it. Serverless rents executors from a legal-value table; a cluster rents
machines from GCE, so the shapes come from `catalog`'s machine tables and the knobs are
continuous within what the worker actually has.

The two translators are peers. Everything they must agree about — the JVM heap floor, the
thread-pin variable names, the mebibyte — lives one layer down in `catalog`, so neither
imports the other and neither can drift.
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
    _SPARK_JVM_MB_PER_CORE,
    machine_cores,
    machine_memory_bytes,
)
from .fleet import RuntimeResourcePlan, UnitShape, plan_fleet, schedulable_memory_bytes
from .slot import ResourceSlot, merge_slots, resource_slot

if TYPE_CHECKING:
    from ..profiling.cost import ComputeProfile


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


def translate_cluster(plan: RuntimeResourcePlan, *, pin_threads: bool = True) -> ClusterTranslation:
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
    if pin_threads:
        for name in _INTRAOP_ENV_VARS:
            properties[f"spark.executorEnv.{name}"] = str(task_cpus)
    else:
        notes.append(
            "native thread pools left uncapped so effective_cores can be measured; "
            "executors are deliberately oversubscribed — do not size a real run from this shape"
        )

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
    pin_threads: bool = True,
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
    return plan, translate_cluster(plan, pin_threads=pin_threads)
