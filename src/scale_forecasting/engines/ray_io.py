"""Shared Ray-engine plumbing — deterministic cluster sizing, GPU/CPU routing, cell chunking.

The Ray-on-Vertex analog of :mod:`spark_io`. Split along the same pure/I-O seam (CONTRACTS §0) so
the interesting logic is offline-testable without a cluster, a GPU, or BigQuery:

* **Pure** (no Ray, no Vertex, no GPU): :func:`split_gpu_cpu_models` (which models want a GPU),
  :func:`plan_cluster` (size an *autoscaling* cluster to the run's fan-out, DESIGN §11.1 / D17),
  :func:`calibrate_gpu_fraction` (profile-driven ``num_gpus`` per NeuralProphet task,
  unit-tested with injected memory numbers), :func:`chunk_cells` (shuffle cells into task-sized
  pandas frames), :func:`make_chunk_runner` (the body one Ray task runs).
* **Reuse, not re-implementation.** The executor-side work is the *exact* Spark core:
  :func:`~scale_forecasting.engines.spark_io.run_group` runs each cell, and the status roll-up is
  :func:`~scale_forecasting.engines.spark_io.aggregate_status`. A Ray "chunk" is the Spark "bucket"
  by another name — same pandas shape, same :func:`run_cell`. This module owns only what is
  genuinely Ray-specific: the deterministic sizing and the heterogeneous GPU/CPU split.

**Why autoscaling by default (D17, reversed post-demo).** The B4 design shipped a *fixed*-size pool
on the reasoning that a run's fan-out is known up front. The overnight 100k run showed that is the
wrong default for a bursty, embarrassingly-parallel fleet: a fixed pool can neither grow to chew a
deep task queue nor shrink the expensive T4 pool when idle. So each pool is now created with a
Vertex ``AutoscalingSpec(min, max)`` and scales with Ray's pending-task demand. Determinism is kept
a better way — :func:`plan_cluster` stays a pure function of the config: the autoscale flag, the
per-pool ``[min, max]``, and the fixed-size-equivalent node count the fan-out implies are all
derived offline and snapshotted into ``run_id`` + ``job_telemetry``. ``ray_autoscale=False`` gives
the fixed path (a fixed ``node_count`` and **no** ``autoscaling_spec``). NOTE: under autoscaling the
Vertex SDK ignores ``node_count`` (the pool starts at ``min`` and scales to ``max``), so the derived
count is the *initial* size only for the fixed path.

**Why heterogeneous routing.** Only NeuralProphet (``family == "deep_learning"``) benefits from a
GPU, and Spark can't share a GPU fractionally across tasks (DESIGN §11.2) — which is the whole
reason Ray is in the design. So NeuralProphet cells run in ``@ray.remote(num_gpus=<fraction>)``
tasks that pack several onto one T4, while every other model runs in ``@ray.remote(num_cpus=1)``.
The routing decision (which models, which fraction, how many nodes) lives here; the decorators that
act on it live in :mod:`.ray_engine`.
"""

from __future__ import annotations

import math
import re
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

# The pure Spark core is engine-agnostic — reuse it verbatim rather than duplicating (D17 / plan
# decision 1). ``_MODEL_COL`` is the internal per-cell model tag ``run_group`` reads to take its
# explode branch (one cell per ``(ts_id, model)``); a Ray chunk carries it exactly like a Spark
# bucket does.
from .spark_io import _MODEL_COL, aggregate_status, run_group

if TYPE_CHECKING:
    import pandas as pd

    from ..config import RunConfig
    from ..settings import Settings

__all__ = [
    "RayClusterPlan",
    "aggregate_status",
    "calibrate_gpu_fraction",
    "chunk_cells",
    "make_chunk_runner",
    "plan_cluster",
    "split_gpu_cpu_models",
]

# The model family that benefits from a GPU (only NeuralProphet today). Everything else is CPU work.
_GPU_FAMILY = "deep_learning"

# T4 device memory (16 GiB). The denominator when auto-calibration turns a measured peak-memory
# footprint into a GPU fraction. A T4 is the design's GPU (cheap, ubiquitous — DESIGN §11).
_T4_MEMORY_BYTES = 16 * 1024**3

# Accelerator type strings Vertex expects, keyed by the config's short ``gpu_type``.
_ACCELERATOR_TYPES = {"T4": "NVIDIA_TESLA_T4"}

# When ``gpu_fraction == "auto"`` we can't run the live calibration at *submit* time (no cluster
# yet) to size the pool, so sizing uses this nominal fraction (→ 2 NeuralProphet slots per T4). The
# on-cluster :func:`calibrate_gpu_fraction` refines the *actual* ``num_gpus`` per task once a T4 is
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
    cfg: RunConfig, models: list[str] | None = None
) -> tuple[list[str], list[str]]:
    """Partition the executed models into ``(gpu_models, cpu_models)`` by family (pure).

    A model routes to the GPU pool iff its registered ``family`` is ``deep_learning``
    (NeuralProphet) — the only family a GPU helps (§11.2). Everything else (statistical/ml) is CPU.
    ``models`` is the executed subset (Arc B / :func:`main.run`); ``None`` means ``cfg.models``.
    Order is preserved within each list so logs and chunking stay deterministic. Unknown names raise
    :class:`~scale_forecasting.errors.ModelError` via the factory — the same up-front validation the
    router does.
    """
    from ..models import get_model

    executed = models if models is not None else cfg.models
    gpu_models: list[str] = []
    cpu_models: list[str] = []
    for name in executed:
        if get_model(name).family == _GPU_FAMILY:
            gpu_models.append(name)
        else:
            cpu_models.append(name)
    return gpu_models, cpu_models


# --- pure: auto-fraction calibration (DESIGN §11.1) ----------------------------


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


def calibrate_gpu_fraction(
    cfg: RunConfig,
    *,
    sample_series: list[pd.DataFrame] | None = None,
    measured_peaks_bytes: list[int] | None = None,
) -> float:
    """Resolve the ``num_gpus`` fraction each NeuralProphet task requests (DESIGN §11.1).

    Two paths, mirroring the config's ``compute.gpu_fraction``:

    * **fixed float** → return it unchanged (the operator pinned it; no profiling).
    * **``"auto"``** → size the fraction to the model's real footprint: fit NeuralProphet on a few
      sample series measuring peak GPU memory, take the worst case, add a safety margin, and divide
      by the T4's 16 GiB — so ``fraction ≈ peak × margin / 16GiB`` and ``floor(1/fraction)`` tasks
      pack without an OOM. Clamped to ``[_MIN_FRACTION, 1.0]``.

    The measurement is injectable so the sizing math is unit-testable **without a GPU** (BUILD B4's
    offline gate): pass ``measured_peaks_bytes`` to skip the live fit entirely. On a real cluster
    (:func:`~scale_forecasting.engines.ray_engine.run`) the peaks are measured live via
    :func:`_measure_np_peak_bytes` over ``sample_series``. With nothing to measure it falls back to
    :data:`_NOMINAL_AUTO_FRACTION`. The chosen fraction + measurements are logged to the registry so
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
    raw = (peak * cfg.compute.gpu_safety_margin) / _T4_MEMORY_BYTES
    return _clamp_fraction(raw)


def _measure_np_peak_bytes(
    series: pd.DataFrame, cfg: RunConfig
) -> int:  # pragma: no cover - live GPU path, exercised only by the @gpu smoke
    """Fit NeuralProphet on one series and return the peak CUDA bytes it allocated.

    Live-only (needs a real GPU): resets the torch allocator's high-water mark, fits one cell via
    the shared :func:`~scale_forecasting.worker.run_cell`, and reads
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


# --- pure: deterministic (fixed-size) cluster sizing (D17) ---------------------


@dataclass(frozen=True)
class RayClusterPlan:
    """An autoscaling Vertex Ray cluster spec, sized to a run's fan-out (pure product of config).

    Autoscaling by default (D17, reversed post-demo): when ``autoscale`` each worker pool is created
    with a Vertex ``AutoscalingSpec`` bounded by its resolved ``[cpu|gpu]_min_nodes`` /
    ``[cpu|gpu]_max_nodes`` and starts at its min; when ``autoscale`` is False both pools are fixed
    at ``cpu_node_count`` / ``gpu_node_count`` (the pre-reversal path, no ``autoscaling_spec``).

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
    # Autoscaling spec (D17 reversal). ``autoscale`` gates whether the pools carry an
    # ``AutoscalingSpec``; the resolved per-pool ``[min, max]`` bounds it (max already defaulted
    # from ``ray_max_nodes`` when unset). All pure products of the config → snapshotted for audit.
    autoscale: bool
    cpu_min_nodes: int
    cpu_max_nodes: int
    gpu_min_nodes: int
    gpu_max_nodes: int

    @property
    def total_worker_nodes(self) -> int:
        """Fixed-size-equivalent worker count across both pools — the number the fan-out implies.

        On the fixed path this is the actual provisioned worker count; under autoscaling it is the
        reference size (each pool actually starts at its min and scales toward its max).
        """
        return self.cpu_node_count + self.gpu_node_count


def _machine_cores(machine_type: str) -> int:
    """Cores implied by a GCE machine type like ``n1-standard-8`` → 8 (pure; default 8).

    Reads the trailing integer of a ``<family>-<class>-<cores>`` name. Custom/typeless names or an
    unparseable suffix fall back to 8 (a reasonable N1 default) so sizing never divides by zero.
    """
    m = re.search(r"-(\d+)$", machine_type)
    return int(m.group(1)) if m else 8


def _accelerator_type(gpu_type: str) -> str:
    """Map the config's short ``gpu_type`` (``T4``) to the Vertex accelerator enum (pure)."""
    try:
        return _ACCELERATOR_TYPES[gpu_type]
    except KeyError:
        raise ValueError(
            f"unsupported gpu_type '{gpu_type}'; supported: {sorted(_ACCELERATOR_TYPES)}"
        ) from None


def _cluster_name(cfg: RunConfig, run_id: str) -> str:
    """The cluster name: the reuse target if set, else ``sf-ray-<run_id>`` (Vertex-legal, ≤ 63).

    Vertex cluster display names must be lowercase alnum + hyphens and start with a letter; the
    ``run_id`` is already a slug + hex digest, so the ``sf-ray-`` prefix keeps it legal. Clamped to
    63 chars with no trailing hyphen.
    """
    if cfg.compute.ray_cluster_name:
        return cfg.compute.ray_cluster_name
    return f"sf-ray-{run_id}"[:63].rstrip("-")


def _sizing_fraction(cfg: RunConfig) -> float:
    """The GPU fraction used to *size* the pool: the fixed float, or the nominal when ``auto``.

    Sizing happens offline at submit time (no cluster to calibrate against), so an ``auto`` fraction
    sizes with :data:`_NOMINAL_AUTO_FRACTION`; the live calibration later refines the per-task
    request but not the (already-created) node count.
    """
    fraction = cfg.compute.gpu_fraction
    return float(fraction) if isinstance(fraction, float) else _NOMINAL_AUTO_FRACTION


def _pool_node_count(
    n_cells: int, slots_per_node: int, target_per_slot: int, max_nodes: int
) -> int:
    """Fixed nodes for one pool: ``ceil(cells / (slots × target))`` clamped to ``[1, max]`` (pure).

    Each node offers ``slots_per_node`` concurrent task slots; we want roughly ``target_per_slot``
    cells to flow through each slot before adding another node (so per-node warm-up amortizes). Zero
    cells → zero nodes (the pool isn't needed); otherwise at least one node, never more than
    ``max_nodes`` (the guardrail against a runaway fan-out requesting an unbounded cluster).
    """
    if n_cells <= 0:
        return 0
    per_node = max(1, slots_per_node * target_per_slot)
    return max(1, min(math.ceil(n_cells / per_node), max_nodes))


def _clamp_pool_nodes(nodes: int, min_nodes: int) -> int:
    """Floor a used pool's derived node count at ``min_nodes``; leave an unused pool at 0 (pure).

    A zero-cell pool is omitted at create (``nodes == 0`` stays 0), so the min floor applies only
    when the pool is actually used. Keeps the fixed-path node count and the autoscaling reference
    size consistent with the resolved ``[min, max]`` bounds. The max clamp already happened inside
    :func:`_pool_node_count` (its ``max_nodes`` arg is the pool max).
    """
    return max(nodes, min_nodes) if nodes > 0 else 0


def plan_cluster(cfg: RunConfig, models: list[str] | None = None, *, run_id: str) -> RayClusterPlan:
    """Size an autoscaling Vertex Ray cluster to this run's fan-out (pure; D17 reversed post-demo).

    Deterministic function of the config — no GCP, no GPU. Splits the executed models into GPU
    (NeuralProphet) and CPU pools, counts the cells each pool must run (``series × models``, using
    ``max_parallelism`` as the basis when ``series_limit`` is unbounded), and derives each pool's
    fixed-size-equivalent ``node_count`` from those cell counts (:func:`_pool_node_count`), clamped
    into the pool's resolved ``[min, max]``. Folds are *not* a factor in the node count — a cell
    runs all its backtest folds internally in one :func:`run_cell`, so folds add per-cell time, not
    more tasks.

    Per-pool autoscaling bounds are resolved here (min from config; max from the per-pool override,
    else the shared ``ray_max_nodes``) and carried on the plan. When ``ray_autoscale`` (default) the
    launcher gives each pool an ``AutoscalingSpec(min, max)`` and it starts at ``min`` and scales to
    ``max`` with Ray's task demand; when False both pools are fixed at the derived ``node_count``.
    Either way the whole spec is a pure product of the config — a bigger ``series_limit`` implies
    more cells → a higher derived count (and, on the fixed path, more nodes); the plan is the whole
    sizing decision, logged and stamped to the run for audit.
    """
    gpu_models, cpu_models = split_gpu_cpu_models(cfg, models)

    n_series = cfg.data.series_limit
    basis = n_series if n_series is not None else cfg.compute.max_parallelism
    n_gpu_cells = basis * len(gpu_models)
    n_cpu_cells = basis * len(cpu_models)

    sizing_fraction = _sizing_fraction(cfg)
    gpu_slots_per_node = cfg.compute.accelerator_count * gpu_slots_per_device(sizing_fraction)
    cpu_slots_per_node = _machine_cores(cfg.compute.ray_cpu_machine_type)

    # Resolve per-pool autoscaling bounds (max defaults to the shared ray_max_nodes when unset).
    cpu_max = cfg.compute.ray_cpu_max_nodes or cfg.compute.ray_max_nodes
    gpu_max = cfg.compute.ray_gpu_max_nodes or cfg.compute.ray_max_nodes
    cpu_min = cfg.compute.ray_cpu_min_nodes
    gpu_min = cfg.compute.ray_gpu_min_nodes

    # Derived fixed-size-equivalent node counts, each capped by its pool max and (when the pool is
    # used) floored at its pool min so the fixed path and the autoscale reference size agree with
    # the bounds. A pool with zero cells stays at 0 nodes (omitted at create), never bumped to min.
    gpu_nodes = (
        _clamp_pool_nodes(
            _pool_node_count(
                n_gpu_cells, gpu_slots_per_node, cfg.compute.ray_target_cells_per_slot, gpu_max
            ),
            gpu_min,
        )
        if cfg.compute.use_gpu
        else 0
    )
    cpu_nodes = _clamp_pool_nodes(
        _pool_node_count(
            n_cpu_cells, cpu_slots_per_node, cfg.compute.ray_target_cells_per_slot, cpu_max
        ),
        cpu_min,
    )

    return RayClusterPlan(
        cluster_name=_cluster_name(cfg, run_id),
        reuse=cfg.compute.ray_cluster_name is not None,
        head_machine_type=cfg.compute.ray_head_machine_type,
        cpu_machine_type=cfg.compute.ray_cpu_machine_type,
        cpu_node_count=cpu_nodes,
        gpu_machine_type=cfg.compute.ray_gpu_machine_type,
        gpu_node_count=gpu_nodes,
        accelerator_type=_accelerator_type(cfg.compute.gpu_type),
        accelerator_count=cfg.compute.accelerator_count,
        sizing_gpu_fraction=sizing_fraction,
        n_gpu_cells=n_gpu_cells,
        n_cpu_cells=n_cpu_cells,
        autoscale=cfg.compute.ray_autoscale,
        cpu_min_nodes=cpu_min,
        cpu_max_nodes=cpu_max,
        gpu_min_nodes=gpu_min,
        gpu_max_nodes=gpu_max,
    )


# --- pure: cell chunking (the Ray task unit) -----------------------------------


def chunk_cells(
    source: pd.DataFrame, cfg: RunConfig, models: list[str], n_chunks: int
) -> list[pd.DataFrame]:
    """Shuffle ``(series × models)`` cells into ``n_chunks`` task-sized pandas frames (pure).

    The Ray analog of Spark's cross-join + bucket, done in pandas on the driver: replicate the
    source once per model (tagging each copy with :data:`_MODEL_COL`), then assign every
    ``(ts_id, model)`` cell to a chunk by a stable CRC32 of its key so a cell's whole history lands
    in one chunk. Each returned frame carries :data:`_MODEL_COL`, so :func:`run_group` takes its
    per-cell explode branch over it — identical to what a Spark bucket feeds. Empty chunks are
    dropped; an empty ``models`` or empty source yields ``[]``.

    ``n_chunks`` is clamped to ``[1, _MAX_CHUNKS]``. Cross-joining in memory is bounded by
    ``series_limit`` for demos; Ray is not the 100k hero path (that's Spark, DESIGN §11.2), so a
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

    The Ray analog of :func:`~scale_forecasting.engines.spark_io.make_group_runner`. Closes over
    the picklable ``cfg`` + ``settings`` (both frozen → cross the Ray task boundary as plain data,
    the G1 seam without a second env path), calls the shared pure :func:`run_group` on the chunk,
    appends the results with the B1 writer (:func:`~scale_forecasting.registry.bq.write_cells`,
    task-side, once per chunk — appends compose, §3.4), and returns only the compact status frame so
    no forecast payload crosses back to the driver.

    ``models`` is forwarded to :func:`run_group` for parity with the Spark path; since chunks always
    carry :data:`_MODEL_COL`, ``run_group`` takes its explode branch and the subset only matters if
    a chunk ever arrived without the tag.

    ``params_by_model`` is the fleetwide-HPO resolution (C5), captured in the closure like ``cfg`` /
    ``settings`` and forwarded to :func:`run_group` — the Ray twin of the Spark group runner's
    fleetwide threading, so tuned params reach every task without entering ``cfg`` (run_id stable).
    """

    def _run(chunk: pd.DataFrame) -> pd.DataFrame:
        from ..registry import bq

        results, status = run_group(chunk, cfg, models, params_by_model)
        if results:
            bq.write_cells(results, settings=settings)
        return status

    return _run
