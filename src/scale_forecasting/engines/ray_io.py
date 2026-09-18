"""Shared Ray-engine plumbing — deterministic cluster sizing, GPU/CPU routing, cell chunking.

The Ray-on-Vertex analog of `spark_io`. Split along the same pure/I-O seam so
the interesting logic is offline-testable without a cluster, a GPU, or BigQuery:

* **Pure** (no Ray, no Vertex, no GPU): `split_gpu_cpu_models` (which models want a GPU),
  `plan_cluster` (size an *autoscaling* cluster to the run's fan-out),
  `calibrate_gpu_fraction` (profile-driven ``num_gpus`` per NeuralProphet task,
  unit-tested with injected memory numbers), `chunk_cells` (shard the panel by series into
  task-sized pandas frames), `make_chunk_runner` (the body one Ray task runs).
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
from ..resources.catalog import device_floor_fraction, machine_cores, machine_memory_bytes
from ..resources.fleet import (
    RuntimeResourcePlan,
    UnitShape,
    max_slot_memory_bytes,
    plan_fleet,
    schedulable_cores,
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
    "EPHEMERAL_PREFIX",
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
    "run_id_prefix_from_cluster_name",
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
#
# It is also the answer when the calibration cannot measure — which, until the probe was moved onto
# a GPU worker (`_measure_np_peaks_on_device`), was *every run*. Treat a change to this number as a
# change to real fleet density, not as a change to a default nobody reaches.
_NOMINAL_AUTO_FRACTION = 0.5

# How long to wait for the calibration probes before giving up and using the nominal fraction. The
# probes are three NeuralProphet fits on whole devices; generous, because the alternative to waiting
# is sizing the entire run off a guess, and bounded, because a GPU pool that never scales must not
# hang the driver.
_CALIBRATION_TIMEOUT_S = 600.0

# Clamp calibrated fractions to a sane band: below this a single task barely uses the GPU (packing
# overhead dominates), above 1.0 is meaningless (one task can't want more than a whole device).
#
# The floor is a *default*. `catalog.device_floor_fraction` lowers it on a node whose cores would
# have run more cells than ``0.1`` permits, because a card split ten ways on a sixteen-core node is
# the device axis idling cores — read that function before changing this number.
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


def resolve_job_gpu(cfg: RunConfig) -> tuple[bool, str | None]:
    """This Ray job's GPU decision, from the *resolved* per-family compute (pure).

    The single answer to "does this job have a device, and which one" — the same
    `config.RunConfig.resolve_family_compute` the submitter provisions from and the DAG
    orchestrator plans from. The engine must ask this rather than read ``compute.use_gpu``
    directly, because that flat field is only one of the two ways a run can request a GPU.

    **This exists because reading the flat field shipped a real defect.** A config that asks
    per-family — ``compute.families.deep_learning.hardware: "gpu"``, which is the documented
    way — leaves ``compute.use_gpu`` at ``False``. The submitter resolved the family and
    provisioned accelerators; the engine read the flat field, sent every deep-learning cell to
    the CPU pool, and the devices sat idle for the whole run with nothing reporting it. It also
    ran the reverse: flat ``use_gpu: true`` with a family override of ``hardware: "cpu"``
    provisioned no GPU nodes while the engine still asked Ray for ``num_gpus``, leaving tasks
    permanently unschedulable. Both disappear once provisioning and routing read one function.

    Returns ``(has_gpu, gpu_type)``. Only ``deep_learning`` can resolve to a GPU, so that is the
    family asked; a config with no deep-learning model still answers, and the answer costs
    nothing because the resolver is pure.
    """
    dl = cfg.resolve_family_compute(_GPU_FAMILY)
    return dl.hardware == "gpu", dl.gpu_type


# --- pure: auto-fraction calibration -------------------------------------------


def _clamp_fraction(fraction: float, floor: float | None = None) -> float:
    """Clamp a GPU fraction to ``[floor, 1.0]``, defaulting to `_MIN_FRACTION` (pure)."""
    return max(_MIN_FRACTION if floor is None else floor, min(1.0, fraction))


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


def pool_unit_shape(cfg: RunConfig, *, gpu: bool) -> UnitShape:
    """One worker node of the CPU or GPU pool: cores and RAM from the machine type (pure).

    Shared by `plan_pool` and `calibrate_gpu_fraction` because the two have to agree about the
    node. The calibration decides how finely to split a card, `plan_pool` decides how many cells
    that card then holds, and both answers are bounded by the same cores — sizing them off two
    independently-built shapes is how the two drift apart.
    """
    machine_type = cfg.compute.ray_gpu_machine_type if gpu else cfg.compute.ray_cpu_machine_type
    return UnitShape(
        cores=machine_cores(machine_type),
        memory_bytes=machine_memory_bytes(machine_type),
        accelerators=cfg.compute.accelerator_count if gpu else 0,
    )


def calibrate_gpu_fraction(
    cfg: RunConfig,
    *,
    sample_series: list[pd.DataFrame] | None = None,
    measured_peaks_bytes: list[int | None] | None = None,
    gpu_type: str | None = None,
) -> float:
    """Resolve the ``num_gpus`` fraction each NeuralProphet task requests.

    Two paths, mirroring the config's ``compute.gpu_fraction``:

    * **fixed float** → return it unchanged (the operator pinned it; no profiling).
    * **``"auto"``** → size the fraction to the model's real footprint: fit NeuralProphet on a few
      sample series measuring peak GPU memory, take the worst case, add a safety margin, and divide
      by **the device's** memory — so ``fraction ≈ peak × margin / device_bytes`` and
      ``floor(1/fraction)`` tasks pack without an OOM. Clamped to ``[floor, 1.0]``, where the
      floor is `_MIN_FRACTION` or lower — see `catalog.device_floor_fraction`, which lets it drop
      on a node whose cores would have run more cells than a flat ``0.1`` allows.

    ``gpu_type`` picks that denominator (`device_memory_bytes`); ``None`` falls back to
    ``compute.gpu_type``. It is an argument rather than read from ``cfg`` because a family's
    accelerator is resolved per-job and deliberately kept out of the config (the ``run_id`` digest
    must stay identical across every family in a run) — the same reason `plan_cluster` takes it.

    The measurement is injectable so the sizing math is unit-testable **without a GPU** in the
    offline gate: pass ``measured_peaks_bytes`` to skip the live fit entirely. On a real cluster
    (`run`) the peaks are measured live by `_measure_np_peaks_on_device`, which dispatches
    ``sample_series`` to GPU workers. With nothing to measure it falls back to
    `_NOMINAL_AUTO_FRACTION`. The chosen fraction + measurements are logged to the registry so
    the sizing decision is auditable (done by the caller).

    **A failed probe is not a measurement of zero, and treating it as one inverted the result.**
    Earlier code returned 0 from a probe that raised; ``max`` of three zeros is zero; and
    `_clamp_fraction` turns a zero raw fraction into exactly ``_MIN_FRACTION`` — the *smallest
    legal* fraction, i.e. the *densest* packing, on the strength of a measurement that never
    happened. Non-positive samples are therefore dropped before the ``max``, and a sample list that
    empties out lands on the nominal fraction, the same as no samples at all.

    **That fix was correct and it was not sufficient, because the probe could not run at all.** It
    was being called on the head node, which has no accelerator, so "no usable samples" was not an
    edge case — it was every run, and the function's real output was whichever constant the
    fallback held. `_measure_np_peaks_on_device` puts the probe on a worker that has a card. The
    fallback is now what it always claimed to be: the answer when measurement is impossible, rather
    than the answer.
    """
    fraction = cfg.compute.gpu_fraction
    if isinstance(fraction, float):
        return float(fraction)

    # auto: profile peak memory (injected for offline tests, measured live on the cluster).
    if measured_peaks_bytes is None:
        series = (sample_series or [])[: cfg.compute.gpu_calibration_samples]
        measured_peaks_bytes = _measure_np_peaks_on_device(series, cfg)
    usable = [peak for peak in measured_peaks_bytes if peak and peak > 0]
    if not usable:
        return _NOMINAL_AUTO_FRACTION

    raw = (max(usable) * cfg.compute.gpu_safety_margin) / device_memory_bytes(
        gpu_type or cfg.compute.gpu_type
    )
    unit = pool_unit_shape(cfg, gpu=True)
    return _clamp_fraction(raw, device_floor_fraction(schedulable_cores(unit), unit.accelerators))


def _measure_np_peaks_on_device(
    series: list[pd.DataFrame], cfg: RunConfig
) -> list[int | None]:  # pragma: no cover - live GPU path, exercised only by the @gpu smoke
    """Measure each sample's peak device memory **on a node that has a device**.

    `_measure_np_peak_bytes` needs a card. `calibrate_gpu_fraction` is called from the driver
    section of `ray_engine.run`, and that driver is the Ray **job entrypoint — it runs on the head
    node, which carries no accelerator.** Called inline, therefore, the probe raised on
    ``torch.cuda.reset_peak_memory_stats`` for every sample on every run that has ever been made,
    and ``"auto"`` silently collapsed to whatever the no-samples fallback happened to be. It was
    never a calibration; it was a constant with a measurement's name on it. Measured live
    2026-09-16 on a 10,000-series run: the fallback of the day put two NeuralProphet cells on each
    T4 where cores would have allowed seven, and cost 44 % of the fleet's per-device throughput.

    The fix is only to run the probe somewhere else. Each sample is dispatched as a Ray task
    requesting a whole device, so the scheduler places it on the GPU pool. Nothing about the
    measurement changes — it is the same module-level function, called with the same arguments.

    **Dispatch is best-effort in exactly the way the measurement is.** Ray is already connected by
    this point (`ray_engine.run` calls ``ray.init()`` before sizing) and the GPU pool is already
    provisioned (the cluster is created before the job is submitted), so the common case is a
    straightforward remote call. But a pool that never scales would leave the tasks pending
    forever, and a calibration is not worth hanging a run over — hence the bounded wait, after
    which we return "not measured" and the caller lands on the nominal fraction, which is precisely
    the behaviour this function replaces. Degrading to the old answer is acceptable; blocking is
    not.
    """
    if not series:
        return []
    try:
        import ray

        probe = ray.remote(num_gpus=1)(_measure_np_peak_bytes)
        futures = [probe.remote(s, cfg) for s in series]
        return list(ray.get(futures, timeout=_CALIBRATION_TIMEOUT_S))
    except Exception:  # noqa: BLE001 - see above: a probe that cannot run is not a measurement
        return [None] * len(series)


def _measure_np_peak_bytes(
    series: pd.DataFrame, cfg: RunConfig
) -> int | None:  # pragma: no cover - live GPU path, exercised only by the @gpu smoke
    """Fit NeuralProphet on one series and return the peak CUDA bytes it allocated.

    Live-only (needs a real GPU): resets the torch allocator's high-water mark, fits one cell via
    the shared `run_cell`, and reads ``torch.cuda.max_memory_allocated``. Dispatched onto a GPU
    worker by `_measure_np_peaks_on_device` — calling it on the driver measures nothing, which is
    the bug that function exists to fix.

    Any failure returns ``None`` — *not measured* — and so does a genuine zero, because a fit that
    allocated nothing on the device is a fit that did not run on the device. Both are dropped by
    the caller rather than being maxed as if they were footprints. The distinction matters because
    returning 0 made `calibrate_gpu_fraction` size every cell at the *minimum* fraction it is
    allowed to request, which is the *densest* packing, on the strength of a probe that had failed.
    """
    try:
        import torch

        from ..worker import run_cell

        torch.cuda.reset_peak_memory_stats()
        run_cell(series, "neuralprophet", cfg)
        return int(torch.cuda.max_memory_allocated()) or None
    except Exception:  # noqa: BLE001 - calibration is best-effort; fall back to nominal
        return None


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


#: What every *ephemeral* Ray cluster's name starts with, and the only place a ``run_id`` is written
#: down on the compute side: `ray_reaper.classify_clusters` reads it back out to find the run that
#: owns a standing cluster, and will not consider deleting a cluster whose name lacks it.
EPHEMERAL_PREFIX = "sf-ray-"


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
    return f"{EPHEMERAL_PREFIX}{run_id}"[:63].rstrip("-")


def run_id_prefix_from_cluster_name(name: str) -> str | None:
    """Read the ``run_id`` back out of an ephemeral cluster name, or ``None`` if it isn't one.

    The inverse of `cluster_name`, and deliberately named a *prefix* rather than a run id: the clamp
    to 63 characters means a run whose ``run_name`` is long enough loses the tail of its id here, so
    what comes back is only guaranteed to be the start of the real one. Callers must match it
    against known run ids by prefix, never assume equality — `ray_reaper.classify_clusters` does
    exactly that, and treats an ambiguous prefix as a reason to leave the cluster alone.
    """
    if not name.startswith(EPHEMERAL_PREFIX):
        return None
    return name[len(EPHEMERAL_PREFIX) :] or None


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

    The single place the Ray runtime turns a `ComputeProfile` into hardware. Takes the pool's
    `UnitShape` from `pool_unit_shape`, sizes one slot per family that lands on the pool,
    `merge_slots` them into the one slot a shared worker needs, and hands the result to
    `plan_fleet`.

    **``profile=None`` asks for nothing the measurement would have asked for**: a slot is one
    core, no memory request, and — on the GPU pool — `_sizing_fraction`. What that slot then packs
    onto a node is `resources.fleet`'s business, and its answer is the smallest of the three
    bounds. So the CPU pool lands on `resources.fleet.schedulable_cores`, one cell per node below
    the nameplate count the inline arithmetic used, because the reserved core is the one the raylet
    reports progress on. The GPU pool usually still binds on devices at the coarse submit-time
    fraction, and binds on cores once a calibrated fraction packs more slots per device than the
    node has cores to run them on.

    ``gpu_fraction`` overrides the sizing fraction with one that is better known — on the cluster
    `ray_engine` has already run `calibrate_gpu_fraction` against a real device, and
    that live number beats the submit-time nominal. It is only a *fallback* either way: a profile
    that measured the device footprint wins over both.

    ``max_units`` overrides the hard ceiling. `plan_cluster` leaves it unset on the first pass (it
    needs the ceiling-clamped count *before* it can resolve the autoscaling ceiling from it), while
    `ray_engine` passes the cluster's already-resolved ``[cpu|gpu]_max_nodes`` so that
    `tasks_for_ceiling` counts against the ceiling the pool can really reach.
    """
    unit = pool_unit_shape(cfg, gpu=gpu)
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
            min_gpu_fraction=device_floor_fraction(schedulable_cores(unit), unit.accelerators),
            max_cores=unit.cores if unit.cores > 0 else None,
            max_memory_bytes=max_slot_memory_bytes(unit),
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
    """Shard the panel into ``n_chunks`` task-sized pandas frames, one Ray task each (pure).

    **Chunks are assigned by series, not by cell.** Each distinct ``ts_id`` gets a chunk index from
    a stable CRC32 of the id (deterministic across processes, unlike ``hash()``), the index is
    mapped back onto the panel, and one ``groupby`` splits it. The frames come back **untagged** —
    no `_MODEL_COL` — so `run_group` takes its per-series loop over the executed model list. The
    caller must therefore build one runner per pool, with that pool's models, or a chunk would run
    models that belong to the other pool.

    The reason this is not a cross-join. Tagging cells means materializing one copy of the whole
    panel per model on the driver before anything is shipped anywhere: at 2,000 series × 1,460 rows
    × 4 models that measured 8.39 s and +2,416 MB of driver RAM, against 0.52 s and +185 MB for the
    per-series shard — 16× the time and ~13× the incremental memory, for frames that then cost 4×
    the object-store bytes because every row crosses the wire once per model. The cell set is
    identical either way; only who does the replication changes, and a Ray task replicating its own
    handful of series is free.

    **The fallback.** When ``n_chunks > n_series`` there are not enough series to fill the requested
    task count, and the count is not decoration — it is what keeps enough tasks pending for the pool
    to reach its autoscaling ceiling (`tasks_for_ceiling`). So that case cross-joins as before and
    returns *tagged* frames, buying finer-than-series granularity at the cost that made the default
    path expensive. It only fires on small panels, where the cost does not matter.

    Empty chunks are dropped; an empty ``models`` or empty source yields ``[]``. ``n_chunks`` is
    clamped to ``[1, _MAX_CHUNKS]``. **Chunk composition is not part of the contract** — which
    series land together, and whether frames arrive tagged, are implementation details that have
    already changed once. What is contractual: the union of the chunks covers every
    ``(series, model)`` cell exactly once, and each series' full history is in exactly one chunk.
    """
    import pandas as pd

    if source.empty or not models:
        return []

    id_col = cfg.data.ts_id_col
    n_chunks = max(1, min(n_chunks, _MAX_CHUNKS))
    ids = source[id_col].astype(str)
    distinct = ids.drop_duplicates()

    if n_chunks > len(distinct):
        # Fallback: more tasks wanted than there are series. Cross-join to cell granularity — one
        # tagged copy of the source per model — and shard on the (ts_id, model) key.
        tagged = pd.concat(
            [source.assign(**{_MODEL_COL: model}) for model in models], ignore_index=True
        )
        keys = tagged[id_col].astype(str) + "\x00" + tagged[_MODEL_COL].astype(str)
        assignment = keys.map(lambda k: zlib.crc32(k.encode("utf-8")) % n_chunks)
        return _split_on(tagged, assignment)

    by_id = {ts_id: zlib.crc32(ts_id.encode("utf-8")) % n_chunks for ts_id in distinct}
    return _split_on(source, ids.map(by_id))


def _split_on(frame: pd.DataFrame, assignment: pd.Series) -> list[pd.DataFrame]:
    """Split ``frame`` into one frame per distinct value of ``assignment``, in index order (pure).

    The assignment rides alongside rather than in a column so the returned frames carry exactly the
    columns they arrived with — a chunk is handed straight to `run_group`, which would otherwise
    have to know to drop a helper it never asked for.
    """
    chunks: list[pd.DataFrame] = []
    for _idx, rows in frame.groupby(assignment.to_numpy(), sort=True):
        chunks.append(rows.reset_index(drop=True))
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

    ``models`` is **load-bearing on the Ray path**, not the parity nicety it is on Spark. Chunks
    from `chunk_cells` arrive untagged, so ``run_group`` takes its per-series loop and runs exactly
    the models in this list — which means a runner must be built per pool, with that pool's models.
    (The one exception is the small-panel fallback, where chunks do carry `_MODEL_COL` and the tag
    wins; passing the pool's list is right either way.)

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
