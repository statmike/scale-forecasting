"""Shared Ray-engine plumbing — deterministic cluster sizing, GPU/CPU routing, cell chunking.

The Ray-on-Vertex analog of `spark_io`. Split along the same pure/I-O seam so
the interesting logic is offline-testable without a cluster, a GPU, or BigQuery:

* **Pure** (no Ray, no Vertex, no GPU): `split_gpu_cpu_models` (which models want a GPU),
  `plan_cluster` (size an *autoscaling* cluster to the run's fan-out),
  `calibrate_gpu_fraction` (profile-driven ``num_gpus`` per NeuralProphet task,
  unit-tested with injected memory numbers), `chunk_cells` (shuffle cells into task-sized
  pandas frames), `make_chunk_runner` (the body one Ray task runs).
* **Reuse, not re-implementation.** The executor-side work is the *exact* Spark core:
  `run_group` runs each cell, and the status roll-up is
  `aggregate_status`. A Ray "chunk" is the Spark "bucket"
  by another name — same pandas shape, same `run_cell`. This module owns only what is
  genuinely Ray-specific: the deterministic sizing and the heterogeneous GPU/CPU split.

**Autoscaling by default.** Each worker pool is created with a Vertex ``AutoscalingSpec(min, max)``
and scales with Ray's pending-task demand, so a bursty, embarrassingly-parallel fleet can grow to
chew a deep task queue and shrink the expensive T4 pool when idle. Determinism is preserved:
`plan_cluster` stays a pure function of the config — the autoscale flag, the per-pool
``[min, max]``, and the fixed-size-equivalent node count the fan-out implies are all derived offline
and snapshotted into ``run_id`` + ``job_telemetry``. ``ray_autoscale=False`` selects a fixed-size
mode instead (a fixed ``node_count`` and **no** ``autoscaling_spec``). NOTE: under autoscaling the
Vertex SDK ignores ``node_count`` (the pool starts at ``min`` and scales to ``max``), so the derived
count is the *initial* size only for the fixed path.

**Why heterogeneous routing.** Only NeuralProphet (``family == "deep_learning"``) benefits from a
GPU, and Spark can't share a GPU fractionally across tasks — which is the whole
reason Ray is in the design. So NeuralProphet cells run in ``@ray.remote(num_gpus=<fraction>)``
tasks that pack several onto one T4, while every other model runs in ``@ray.remote(num_cpus=1)``.
The routing decision (which models, which fraction, how many nodes) lives here; the decorators that
act on it live in `ray_engine`.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

# The measured-profile → runtime-knobs translation. It lives at the top level rather than under
# ``engines/`` and depends on no engine, so importing it here cannot cycle.
from ..resources.catalog import machine_cores, machine_memory_bytes
from ..resources.fleet import (
    RuntimeResourcePlan,
    UnitShape,
    plan_fleet,
    schedulable_memory_bytes,
)
from ..resources.slot import merge_slots, resource_slot

# The pure Spark core is engine-agnostic — reuse it verbatim rather than duplicating.
# ``_MODEL_COL`` is the internal per-cell model tag ``run_group`` reads to take its
# explode branch (one cell per ``(ts_id, model)``); a Ray chunk carries it exactly like a Spark
# bucket does.
from .spark_io import _MODEL_COL, aggregate_status, run_group

if TYPE_CHECKING:
    import pandas as pd

    from ..config import RunConfig
    from ..profiling.cost import ComputeProfile
    from ..settings import Settings

__all__ = [
    "RayClusterPlan",
    "aggregate_status",
    "calibrate_gpu_fraction",
    "chunk_cells",
    "cluster_name",
    "device_memory_bytes",
    "make_chunk_runner",
    "plan_cluster",
    "plan_pool",
    "pool_families",
    "split_gpu_cpu_models",
]

# The model family that benefits from a GPU (only NeuralProphet today). Everything else is CPU work.
_GPU_FAMILY = "deep_learning"

# Device memory per supported accelerator — the denominator when auto-calibration turns a measured
# peak-memory footprint into a GPU fraction. Per *device*, not per node (a node may carry several,
# which is `accelerator_count`). Getting this wrong is silently expensive in both directions: too
# small under-packs the device (paying for GPU we don't use), too large over-packs it (OOM).
_DEVICE_MEMORY_BYTES = {
    "T4": 16 * 1024**3,  # NVIDIA Tesla T4 — 16 GiB
    "L4": 24 * 1024**3,  # NVIDIA L4 — 24 GiB
}
# Fallback for an unrecognised accelerator: assume the smallest device we know, so an unknown GPU
# under-packs (wastes capacity) rather than over-packs (OOMs the run).
_DEFAULT_DEVICE_MEMORY_BYTES = min(_DEVICE_MEMORY_BYTES.values())

# Accelerator type strings Vertex expects, keyed by the config's short ``gpu_type``.
_ACCELERATOR_TYPES = {"T4": "NVIDIA_TESLA_T4", "L4": "NVIDIA_L4"}

# Each accelerator attaches to one machine family: a T4 is an add-on card on an N1 VM, while an L4
# is only offered on G2 VMs (the card is bundled into the machine type). Sizing/creation must pair
# the two correctly, so the gpu machine type is validated against the chosen ``gpu_type`` — a T4 on
# a g2 machine (or an L4 on an n1) is a create-time error, caught here at plan time instead.
_GPU_MACHINE_PREFIX = {"T4": "n1-", "L4": "g2-"}

# When ``gpu_fraction == "auto"`` we can't run the live calibration at *submit* time (no cluster
# yet) to size the pool, so sizing uses this nominal fraction (→ 2 NeuralProphet slots per T4). The
# on-cluster `calibrate_gpu_fraction` refines the *actual* ``num_gpus`` per task once a T4 is
# available; the node count is fixed at create time, so only the sizing math uses this.
_NOMINAL_AUTO_FRACTION = 0.5

# Clamp calibrated fractions to a sane band: below this a single task barely uses the GPU (packing
# overhead dominates), above 1.0 is meaningless (one task can't want more than a whole device).
_MIN_FRACTION = 0.1

# Safety ceiling on chunk count, mirroring spark_io's bucket ceiling: even a huge run shouldn't
# shatter into an unbounded number of tiny Ray tasks (scheduler overhead, tiny writes).
_MAX_CHUNKS = 100_000


# --- pure: which models want a GPU ---------------------------------------------


def split_gpu_cpu_models(
    cfg: RunConfig, models: list[str] | None = None, *, use_gpu: bool | None = None
) -> tuple[list[str], list[str]]:
    """Partition the executed models into ``(gpu_models, cpu_models)`` by family (pure).

    A model routes to the GPU pool iff its registered ``family`` is ``deep_learning``
    (NeuralProphet) — the only family a GPU helps — **and this job actually has a GPU pool**.
    Everything else (statistical/ml) is CPU. ``models`` is the executed subset (`main.run`);
    ``None`` means ``cfg.models``. ``use_gpu`` overrides the flat ``compute.use_gpu`` with the
    family's resolved hardware, the same way `plan_cluster` takes it.

    **The ``use_gpu`` half is what keeps a GPU-less job from planning a cluster with no workers.**
    A deep-learning model still runs without a GPU — it falls back to CPU inside the cell — so with
    no GPU pool its cells belong to the CPU pool, not to a pool that will not exist. Splitting on
    family alone put them in ``gpu_models``, whose cells were then zeroed because ``use_gpu`` was
    False, while ``cpu_models`` was empty because every executed model was deep-learning: both pools
    derived 0 nodes and the run hung on a head-only cluster with nothing to schedule on. That is not
    an exotic config — ``deep_learning`` resolves to ``hardware="cpu"`` whenever ``compute.use_gpu``
    is left at its default, so ``{"python_runtime": "ray", "models": ["neuralprophet"]}`` reached
    it.

    Order is preserved within each list so logs and chunking stay deterministic. Unknown names raise
    `ModelError` via the factory — the same up-front validation the
    router does.
    """
    from ..models import get_model

    executed = models if models is not None else cfg.models
    effective_use_gpu = cfg.compute.use_gpu if use_gpu is None else use_gpu
    gpu_models: list[str] = []
    cpu_models: list[str] = []
    for name in executed:
        if effective_use_gpu and get_model(name).family == _GPU_FAMILY:
            gpu_models.append(name)
        else:
            cpu_models.append(name)
    return gpu_models, cpu_models


# --- pure: auto-fraction calibration -------------------------------------------


def _clamp_fraction(fraction: float) -> float:
    """Clamp a GPU fraction to ``[_MIN_FRACTION, 1.0]`` (pure)."""
    return max(_MIN_FRACTION, min(1.0, fraction))


def gpu_slots_per_device(fraction: float) -> int:
    """How many fractional tasks pack onto one GPU: ``floor(1 / fraction)`` (≥ 1) (pure).

    Ray schedules by summing each task's ``num_gpus`` against a device's capacity of 1.0, so a
    fraction of ``0.25`` packs 4 tasks; ``0.5`` packs 2. Always at least one (a task can't be
    smaller than a whole device if the fraction rounds it there).
    """
    return max(1, math.floor(1.0 / fraction))


def device_memory_bytes(gpu_type: str | None) -> int:
    """Device memory for one accelerator of ``gpu_type`` (pure; unknown → the smallest known).

    The denominator of the auto-fraction. Kept a lookup rather than a constant because the two
    supported devices differ by 50% (T4 16 GiB, L4 24 GiB): sizing an L4 against the T4 constant
    packs only two-thirds of the tasks the device could actually hold.
    """
    return _DEVICE_MEMORY_BYTES.get(gpu_type or "", _DEFAULT_DEVICE_MEMORY_BYTES)


def calibrate_gpu_fraction(
    cfg: RunConfig,
    *,
    sample_series: list[pd.DataFrame] | None = None,
    measured_peaks_bytes: list[int] | None = None,
    gpu_type: str | None = None,
) -> float:
    """Resolve the ``num_gpus`` fraction each NeuralProphet task requests.

    Two paths, mirroring the config's ``compute.gpu_fraction``:

    * **fixed float** → return it unchanged (the operator pinned it; no profiling).
    * **``"auto"``** → size the fraction to the model's real footprint: fit NeuralProphet on a few
      sample series measuring peak GPU memory, take the worst case, add a safety margin, and divide
      by **the device's** memory — so ``fraction ≈ peak × margin / device_bytes`` and
      ``floor(1/fraction)`` tasks pack without an OOM. Clamped to ``[_MIN_FRACTION, 1.0]``.

    ``gpu_type`` picks that denominator (`device_memory_bytes`); ``None`` falls back to
    ``compute.gpu_type``. It is an argument rather than read from ``cfg`` because a family's
    accelerator is resolved per-job and deliberately kept out of the config (the ``run_id`` digest
    must stay identical across every family in a run) — the same reason `plan_cluster` takes it.

    The measurement is injectable so the sizing math is unit-testable **without a GPU** in the
    offline gate: pass ``measured_peaks_bytes`` to skip the live fit entirely. On a real cluster
    (`run`) the peaks are measured live via
    `_measure_np_peak_bytes` over ``sample_series``. With nothing to measure it falls back to
    `_NOMINAL_AUTO_FRACTION`. The chosen fraction + measurements are logged to the registry so
    the sizing decision is auditable (done by the caller).
    """
    fraction = cfg.compute.gpu_fraction
    if isinstance(fraction, float):
        return float(fraction)

    # auto: profile peak memory (injected for offline tests, measured live on the cluster).
    if measured_peaks_bytes is None:
        series = (sample_series or [])[: cfg.compute.gpu_calibration_samples]
        measured_peaks_bytes = [_measure_np_peak_bytes(s, cfg) for s in series]
    if not measured_peaks_bytes:
        return _NOMINAL_AUTO_FRACTION

    peak = max(measured_peaks_bytes)
    raw = (peak * cfg.compute.gpu_safety_margin) / device_memory_bytes(
        gpu_type or cfg.compute.gpu_type
    )
    return _clamp_fraction(raw)


def _measure_np_peak_bytes(
    series: pd.DataFrame, cfg: RunConfig
) -> int:  # pragma: no cover - live GPU path, exercised only by the @gpu smoke
    """Fit NeuralProphet on one series and return the peak CUDA bytes it allocated.

    Live-only (needs a real GPU): resets the torch allocator's high-water mark, fits one cell via
    the shared `run_cell`, and reads
    ``torch.cuda.max_memory_allocated``. Any failure degrades to 0 (the caller's ``max`` skips it),
    so a flaky probe never sinks the run — it just widens to the nominal fraction.
    """
    try:
        import torch

        from ..worker import run_cell

        torch.cuda.reset_peak_memory_stats()
        run_cell(series, "neuralprophet", cfg)
        return int(torch.cuda.max_memory_allocated())
    except Exception:  # noqa: BLE001 - calibration is best-effort; fall back to nominal
        return 0


# --- pure: deterministic per-pool autoscaling cluster sizing -------------------


@dataclass(frozen=True)
class RayClusterPlan:
    """An autoscaling Vertex Ray cluster spec, sized to a run's fan-out (pure product of config).

    Autoscaling by default: when ``autoscale`` each worker pool is created
    with a Vertex ``AutoscalingSpec`` bounded by its resolved ``[cpu|gpu]_min_nodes`` /
    ``[cpu|gpu]_max_nodes`` and starts at its min; when ``autoscale`` is False both pools are fixed
    at ``cpu_node_count`` / ``gpu_node_count`` (a fixed-size mode, no ``autoscaling_spec``).

    ``cpu_node_count`` / ``gpu_node_count`` are the deterministic fixed-size-equivalent the fan-out
    implies — the actual node count on the fixed path, and the initial/reference size on the
    autoscaling path (where the SDK starts the pool at ``min`` instead). ``reuse=True`` means an
    existing cluster is targeted by name (skip create + skip teardown); the sizing fields then
    describe what it *should* be. ``sizing_gpu_fraction`` is the fraction used to size the GPU pool;
    the on-cluster calibration may request a different actual ``num_gpus`` per task.
    ``n_gpu_cells`` / ``n_cpu_cells`` are the per-pool task counts the sizing derived from.
    """

    cluster_name: str
    reuse: bool
    head_machine_type: str
    cpu_machine_type: str
    cpu_node_count: int
    gpu_machine_type: str
    gpu_node_count: int
    accelerator_type: str
    accelerator_count: int
    sizing_gpu_fraction: float
    n_gpu_cells: int
    n_cpu_cells: int
    # Autoscaling spec. ``autoscale`` gates whether the pools carry an ``AutoscalingSpec``; the
    # resolved per-pool ``[min, max]`` bounds it. The max is derived from this run's fan-out
    # (`_resolve_pool_max`) unless the pool was explicitly pinned. All pure products of the config
    # + the run's cell counts → snapshotted for audit.
    autoscale: bool
    cpu_min_nodes: int
    cpu_max_nodes: int
    gpu_min_nodes: int
    gpu_max_nodes: int
    # The full sizing decision behind each pool's node count, with its evidence attached — which
    # axes were measured, which fell back to a constant, what was clamped. The node counts above
    # are `derived_units` off these; keeping the whole record on the plan is what makes a
    # sizing choice auditable after the run rather than only reproducible from the config.
    # ``None`` on a plan built before the pool plans existed (nothing constructs one that way
    # today, but the default keeps the dataclass constructible field-by-field in tests).
    cpu_pool: RuntimeResourcePlan | None = None
    gpu_pool: RuntimeResourcePlan | None = None

    @property
    def total_worker_nodes(self) -> int:
        """Fixed-size-equivalent worker count across both pools — the number the fan-out implies.

        On the fixed path this is the actual provisioned worker count; under autoscaling it is the
        reference size (each pool actually starts at its min and scales toward its max).
        """
        return self.cpu_node_count + self.gpu_node_count


def _accelerator_type(gpu_type: str) -> str:
    """Map the config's short ``gpu_type`` (``T4``/``L4``) to the Vertex accelerator enum (pure)."""
    try:
        return _ACCELERATOR_TYPES[gpu_type]
    except KeyError:
        raise ValueError(
            f"unsupported gpu_type '{gpu_type}'; supported: {sorted(_ACCELERATOR_TYPES)}"
        ) from None


def _check_gpu_machine(gpu_type: str, gpu_machine_type: str) -> None:
    """Fail if the gpu machine type doesn't match the accelerator's required family (pure).

    A T4 is an N1 add-on card; an L4 is only offered on G2 machines. Pairing them wrong is rejected
    at create by Vertex, so it's caught here at plan time with an actionable message.
    """
    prefix = _GPU_MACHINE_PREFIX.get(gpu_type)
    if prefix is not None and not gpu_machine_type.startswith(prefix):
        raise ValueError(
            f"gpu_type '{gpu_type}' requires a '{prefix}' machine type, "
            f"but ray_gpu_machine_type is '{gpu_machine_type}'"
        )


def cluster_name(cfg: RunConfig, run_id: str) -> str:
    """The cluster name: the reuse target if set, else ``sf-ray-<run_id>`` (Vertex-legal, ≤ 63).

    Vertex cluster display names must be lowercase alnum + hyphens and start with a letter; the
    ``run_id`` is already a slug + hex digest, so the ``sf-ray-`` prefix keeps it legal. Clamped to
    63 chars with no trailing hyphen.

    Public because the name is *knowable before the cluster exists*, and one caller needs exactly
    that: `job_launch` builds a single-family Ray job's probe handle before submit, when nothing has
    created a cluster yet. Being derivable from the ``run_id`` is what makes a running Ray job
    reachable at all.
    """
    if cfg.compute.ray_cluster_name:
        return cfg.compute.ray_cluster_name
    return f"sf-ray-{run_id}"[:63].rstrip("-")


def _sizing_fraction(cfg: RunConfig) -> float:
    """The GPU fraction used to *size* the pool: the fixed float, or the nominal when ``auto``.

    Sizing happens offline at submit time (no cluster to calibrate against), so an ``auto`` fraction
    sizes with `_NOMINAL_AUTO_FRACTION`; the live calibration later refines the per-task
    request but not the (already-created) node count.
    """
    fraction = cfg.compute.gpu_fraction
    return float(fraction) if isinstance(fraction, float) else _NOMINAL_AUTO_FRACTION


def pool_families(models: list[str]) -> list[str]:
    """The distinct model families landing on one pool, in first-seen order (pure).

    A pool is not a family: everything that isn't ``deep_learning`` shares the CPU pool, so its
    worker has to hold a statistical cell and an ML cell alike. Order is first-seen so the merged
    label is deterministic and the digest it feeds stays stable.
    """
    from ..models import get_model

    families: list[str] = []
    for name in models:
        family = get_model(name).family
        if family not in families:
            families.append(family)
    return families


def plan_pool(
    cfg: RunConfig,
    models: list[str],
    n_cells: int,
    *,
    gpu: bool,
    gpu_type: str | None = None,
    profile: ComputeProfile | None = None,
    gpu_fraction: float | None = None,
    max_units: int | None = None,
) -> RuntimeResourcePlan:
    """Size one Ray worker pool from the measured profile, or from the old constants (pure).

    The single place the Ray runtime turns a `ComputeProfile` into hardware. Builds the
    pool's `UnitShape` from its configured machine type (cores from the name, RAM from
    `machine_memory_bytes`, devices from ``accelerator_count``), sizes one slot per family
    that lands on the pool, `merge_slots` them into the one slot a shared worker needs, and
    hands the result to `plan_fleet`.

    **``profile=None`` reproduces the pre-profiler arithmetic exactly**, which is the property
    that lets this replace the old inline sizing rather than sit beside it. With no measurement a
    slot is one core, no memory request, and — on the GPU pool — `_sizing_fraction`, so
    ``slots_per_unit`` collapses to `resources.catalog.machine_cores` on the CPU side and to
    ``accelerator_count x gpu_slots_per_device(fraction)`` on the GPU side: the two expressions
    `plan_cluster` used to compute inline.

    ``gpu_fraction`` overrides the sizing fraction with one that is better known — on the cluster
    `ray_engine` has already run `calibrate_gpu_fraction` against a real device, and
    that live number beats the submit-time nominal. It is only a *fallback* either way: a profile
    that measured the device footprint wins over both.

    ``max_units`` overrides the hard ceiling. `plan_cluster` leaves it unset on the first pass (it
    needs the ceiling-clamped count *before* it can resolve the autoscaling ceiling from it), while
    `ray_engine` passes the cluster's already-resolved ``[cpu|gpu]_max_nodes`` so that
    `tasks_for_ceiling` counts against the ceiling the pool can really reach.
    """
    machine_type = cfg.compute.ray_gpu_machine_type if gpu else cfg.compute.ray_cpu_machine_type
    unit = UnitShape(
        cores=machine_cores(machine_type),
        memory_bytes=machine_memory_bytes(machine_type),
        accelerators=cfg.compute.accelerator_count if gpu else 0,
    )
    ceiling = max_units if max_units is not None else _pool_ceiling(cfg, gpu=gpu)
    floor_nodes = cfg.compute.ray_gpu_min_nodes if gpu else cfg.compute.ray_cpu_min_nodes

    families = pool_families(models) or [_GPU_FAMILY if gpu else "cpu"]
    slots = [
        resource_slot(
            profile,
            family,
            use_gpu=gpu,
            device_bytes=device_memory_bytes(gpu_type or cfg.compute.gpu_type) if gpu else None,
            static_gpu_fraction=(
                gpu_fraction if gpu_fraction is not None else _sizing_fraction(cfg)
            ),
            max_cores=unit.cores if unit.cores > 0 else None,
            max_memory_bytes=schedulable_memory_bytes(unit),
        )
        for family in families
    ]
    return plan_fleet(
        merge_slots(slots, family="+".join(families)),
        runtime="ray",
        n_cells=n_cells,
        unit=unit,
        target_cells_per_slot=cfg.compute.ray_target_cells_per_slot,
        min_units=floor_nodes,
        max_units=ceiling,
    )


def _pool_ceiling(cfg: RunConfig, *, gpu: bool) -> int:
    """The hard node ceiling for one pool: its explicit override, else the shared max (pure)."""
    explicit = cfg.compute.ray_gpu_max_nodes if gpu else cfg.compute.ray_cpu_max_nodes
    return explicit or cfg.compute.ray_max_nodes


# Floor on a *derived* autoscaling ceiling. Even a run whose fan-out implies a single node gets
# room to burst to a second, so a small run isn't accidentally pinned to a fixed pool by an
# autoscaling spec of [1, 1]. Above this, the run's own size sets the ceiling.
_AUTOSCALE_MAX_FLOOR = 2


def _resolve_pool_max(
    explicit: int | None, derived_nodes: int, ceiling: int, min_nodes: int
) -> int:
    """The autoscaling ceiling for one pool — an explicit pin, else derived from the run (pure).

    * **``explicit`` set** — the operator pinned this pool's ceiling; honour it verbatim.
      ``ComputeConfig`` has already rejected a pin below the pool's floor, so no check here.
    * **unset** — derive from the run: the fan-out's own node count (already clamped to ``ceiling``
      inside `plan_pool`), floored at `_AUTOSCALE_MAX_FLOOR` and at ``min_nodes``.
    * **unused pool** (``derived_nodes == 0``) — the floors still apply, but the value is inert: a
      zero-node pool is omitted at create, so no ``AutoscalingSpec`` is built from it.

    This is the difference between an elastic pool that can actually absorb its run and one capped
    at a constant: before, an unset ceiling meant the shared ``ray_max_nodes`` (16) regardless of
    whether the run implied 2 nodes or 200.
    """
    if explicit is not None:
        return explicit
    return min(ceiling, max(_AUTOSCALE_MAX_FLOOR, min_nodes, derived_nodes))


def plan_cluster(
    cfg: RunConfig,
    models: list[str] | None = None,
    *,
    run_id: str,
    use_gpu: bool | None = None,
    gpu_type: str | None = None,
    profile: ComputeProfile | None = None,
) -> RayClusterPlan:
    """Size an autoscaling Vertex Ray cluster to this run's fan-out (pure).

    ``use_gpu``/``gpu_type`` override the flat ``compute`` defaults for one family's job (the DAG
    orchestrator passes the family's resolved hardware). They're kept **out of** ``cfg`` on purpose:
    the ``run_id`` is a digest of ``cfg`` and must stay identical across every family under one run,
    so a per-family GPU decision flows as an argument, not a config mutation. ``None`` falls back to
    ``compute.use_gpu`` / ``compute.gpu_type``.

    Deterministic function of the config — no GCP, no GPU. Splits the executed models into GPU
    (NeuralProphet, and only when this job has a GPU at all) and CPU pools, counts the cells each
    pool must run (``series × models``, using
    ``max_parallelism`` as the basis when ``series_limit`` is unbounded), and derives each pool's
    fixed-size-equivalent ``node_count`` from those cell counts (`plan_pool`), clamped
    into the pool's resolved ``[min, max]``. Folds are *not* a factor in the node count — a cell
    runs all its backtest folds internally in one `run_cell`, so folds add per-cell time, not
    more tasks.

    Per-pool autoscaling bounds are resolved here and carried on the plan: the floor comes from
    config (``ray_[cpu|gpu]_min_nodes``), and the **ceiling is derived from this run's own fan-out**
    — the pool's derived node count, floored at `_AUTOSCALE_MAX_FLOOR` and capped by the hard
    ceiling (the per-pool override, else the shared ``ray_max_nodes``). An explicitly pinned
    ``ray_[cpu|gpu]_max_nodes`` is honoured verbatim instead. So a run implying two nodes scales to
    two, not to a constant 16, while the hard ceiling still guards a runaway fan-out.
    When ``ray_autoscale`` (default) the
    launcher gives each pool an ``AutoscalingSpec(min, max)`` and it starts at ``min`` and scales to
    ``max`` with Ray's task demand; when False both pools are fixed at the derived ``node_count``.
    Either way the whole spec is a pure product of the config — a bigger ``series_limit`` implies
    more cells → a higher derived count (and, on the fixed path, more nodes); the plan is the whole
    sizing decision, logged and stamped to the run for audit.

    ``profile`` is an optional `ComputeProfile` from the driver-side measurement pre-pass
    (`profiling.source.resolve_profile`). When given, each pool's slot is sized from what the models
    actually cost — cores, host memory, and the GPU fraction — instead of from the constants this
    function used to inline; ``None`` reproduces those constants exactly, so an unprofiled run is
    byte-identical to one planned before any of this existed. It is an argument rather than a config
    field for the same reason ``use_gpu``/``gpu_type`` are: ``run_id`` digests ``cfg``, and a
    measurement taken at submit time must not move it.
    """
    # Per-family overrides fall back to the flat compute defaults (kept out of cfg to hold run_id).
    # Resolved *before* the split, because the split depends on it: without a GPU pool the
    # deep-learning models are CPU work and have to be sized into the CPU pool (see
    # `split_gpu_cpu_models`), not dropped between the two.
    effective_use_gpu = cfg.compute.use_gpu if use_gpu is None else use_gpu
    effective_gpu_type = gpu_type or cfg.compute.gpu_type
    if effective_use_gpu:
        _check_gpu_machine(effective_gpu_type, cfg.compute.ray_gpu_machine_type)

    gpu_models, cpu_models = split_gpu_cpu_models(cfg, models, use_gpu=effective_use_gpu)

    n_series = cfg.data.series_limit
    basis = n_series if n_series is not None else cfg.compute.max_parallelism
    n_gpu_cells = basis * len(gpu_models)
    n_cpu_cells = basis * len(cpu_models)

    sizing_fraction = _sizing_fraction(cfg)

    # Hard ceiling per pool — the explicit per-pool override, else the shared ray_max_nodes. This
    # bounds the *derived* node count below; the autoscaling ceiling is resolved from that count
    # afterwards (`_resolve_pool_max`), so an unpinned pool scales to the size of the run rather
    # than to a constant.
    cpu_ceiling = _pool_ceiling(cfg, gpu=False)
    gpu_ceiling = _pool_ceiling(cfg, gpu=True)
    cpu_min = cfg.compute.ray_cpu_min_nodes
    gpu_min = cfg.compute.ray_gpu_min_nodes

    # Derived fixed-size-equivalent node counts, each capped by its pool max and (when the pool is
    # used) floored at its pool min so the fixed path and the autoscale reference size agree with
    # the bounds. A pool with zero cells stays at 0 nodes (omitted at create), never bumped to min.
    # `plan_pool` owns the arithmetic for both pools now: with ``profile=None`` it reproduces the
    # constants this function used inline, and with a profile it sizes the slot from measurement.
    # ``n_gpu_cells`` is passed unconditionally: the split above is already hardware-aware, so a
    # GPU-less job has no ``gpu_models`` and no GPU cells to zero out — and the cells it *does* have
    # are counted against the CPU pool rather than dropped.
    gpu_pool = plan_pool(
        cfg,
        gpu_models,
        n_gpu_cells,
        gpu=True,
        gpu_type=effective_gpu_type,
        profile=profile,
    )
    cpu_pool = plan_pool(cfg, cpu_models, n_cpu_cells, gpu=False, profile=profile)
    gpu_nodes = gpu_pool.derived_units
    cpu_nodes = cpu_pool.derived_units

    # The autoscaling ceilings, derived from those node counts unless a pool was explicitly pinned.
    cpu_max = _resolve_pool_max(cfg.compute.ray_cpu_max_nodes, cpu_nodes, cpu_ceiling, cpu_min)
    gpu_max = _resolve_pool_max(cfg.compute.ray_gpu_max_nodes, gpu_nodes, gpu_ceiling, gpu_min)

    # Re-plan each pool against the ceiling it can *actually* reach. The first pass had to use the
    # hard ceiling because the autoscaling one is derived from its answer; this pass makes the
    # stored plan's ``slots_at_ceiling`` — and therefore `tasks_for_ceiling` — describe the
    # real pool rather than the guardrail. The derived node count is unchanged by construction
    # (the resolved max is never below it), so the cluster spec above is unaffected.
    cpu_pool = plan_pool(
        cfg, cpu_models, n_cpu_cells, gpu=False, profile=profile, max_units=cpu_max
    )
    gpu_pool = plan_pool(
        cfg,
        gpu_models,
        n_gpu_cells,
        gpu=True,
        gpu_type=effective_gpu_type,
        profile=profile,
        max_units=gpu_max,
    )

    return RayClusterPlan(
        cluster_name=cluster_name(cfg, run_id),
        reuse=cfg.compute.ray_cluster_name is not None,
        head_machine_type=cfg.compute.ray_head_machine_type,
        cpu_machine_type=cfg.compute.ray_cpu_machine_type,
        cpu_node_count=cpu_nodes,
        gpu_machine_type=cfg.compute.ray_gpu_machine_type,
        gpu_node_count=gpu_nodes,
        accelerator_type=_accelerator_type(effective_gpu_type),
        accelerator_count=cfg.compute.accelerator_count,
        sizing_gpu_fraction=sizing_fraction,
        n_gpu_cells=n_gpu_cells,
        n_cpu_cells=n_cpu_cells,
        autoscale=cfg.compute.ray_autoscale,
        cpu_min_nodes=cpu_min,
        cpu_max_nodes=cpu_max,
        gpu_min_nodes=gpu_min,
        gpu_max_nodes=gpu_max,
        cpu_pool=cpu_pool,
        gpu_pool=gpu_pool,
    )


# --- pure: cell chunking (the Ray task unit) -----------------------------------


def chunk_cells(
    source: pd.DataFrame, cfg: RunConfig, models: list[str], n_chunks: int
) -> list[pd.DataFrame]:
    """Shuffle ``(series × models)`` cells into ``n_chunks`` task-sized pandas frames (pure).

    The Ray analog of Spark's cross-join + bucket, done in pandas on the driver: replicate the
    source once per model (tagging each copy with `_MODEL_COL`), then assign every
    ``(ts_id, model)`` cell to a chunk by a stable CRC32 of its key so a cell's whole history lands
    in one chunk. Each returned frame carries `_MODEL_COL`, so `run_group` takes its
    per-cell explode branch over it — identical to what a Spark bucket feeds. Empty chunks are
    dropped; an empty ``models`` or empty source yields ``[]``.

    ``n_chunks`` is clamped to ``[1, _MAX_CHUNKS]``. Cross-joining in memory is bounded by
    ``series_limit`` for demos; Ray is not the 100k hero path (that's Spark), so a
    driver-side replicate is acceptable here.
    """
    import pandas as pd

    if source.empty or not models:
        return []

    id_col = cfg.data.ts_id_col
    n_chunks = max(1, min(n_chunks, _MAX_CHUNKS))

    # Cross-join: one tagged copy of the source per model (a handful of models, so this is cheap).
    tagged = pd.concat(
        [source.assign(**{_MODEL_COL: model}) for model in models], ignore_index=True
    )

    # Stable per-cell chunk index: CRC32 of "<ts_id>\x00<model>" (deterministic across processes,
    # unlike hash()). Keeps a cell's full history in one chunk so run_group sees a whole series.
    keys = tagged[id_col].astype(str) + "\x00" + tagged[_MODEL_COL].astype(str)
    tagged["_sf_chunk"] = keys.map(lambda k: zlib.crc32(k.encode("utf-8")) % n_chunks)

    chunks: list[pd.DataFrame] = []
    for _idx, frame in tagged.groupby("_sf_chunk", sort=True):
        chunks.append(frame.drop(columns=["_sf_chunk"]).reset_index(drop=True))
    return chunks


def make_chunk_runner(
    cfg: RunConfig,
    settings: Settings,
    models: list[str] | None = None,
    params_by_model: dict[str, dict[str, Any]] | None = None,
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Build the function one Ray task runs: run a chunk's cells, write them, return status.

    The Ray analog of `make_group_runner`. Closes over
    the picklable ``cfg`` + ``settings`` (both frozen → cross the Ray task boundary as plain data,
    the single local/cloud seam without a second env path), calls the shared pure `run_group` on the
    chunk, appends the results with the writer (`write_cells`, task-side, once per chunk — appends
    compose), and returns only the compact status frame so no forecast payload crosses back to the
    driver.

    ``models`` is forwarded to `run_group` for parity with the Spark path; since chunks always
    carry `_MODEL_COL`, ``run_group`` takes its explode branch and the subset only matters if
    a chunk ever arrived without the tag.

    ``params_by_model`` is the fleetwide-HPO resolution, captured in the closure like ``cfg`` /
    ``settings`` and forwarded to `run_group` — the Ray twin of the Spark group runner's
    fleetwide threading, so tuned params reach every task without entering ``cfg`` (run_id stable).
    """

    def _run(chunk: pd.DataFrame) -> pd.DataFrame:
        from ..registry.cells import write_cells

        results, status = run_group(chunk, cfg, models, params_by_model)
        if results:
            write_cells(results, settings=settings)
        return status

    return _run
