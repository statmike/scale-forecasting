"""Vertex AI ``CustomJob`` & GCE VM engine — single-VM default + worker-pool sharding + sizing.

Executes any combination of model families on a dedicated Vertex AI ``CustomJob`` VM, across a
multi-worker ``worker_pool_specs`` pool (or GKE Indexed Job), or on a single Compute Engine VM
(`runtime="gce"`) without the head-node overhead of a Ray cluster or the JVM shuffle overhead of
Spark:

1. **Dedicated Per-Model VMs for Global & Deep-Learning Models**:
   Global/hybrid panel models (`tide`, `tft`, `tsmixer`, `patchtst`, and `neuralprophet` with
   `global_fit=True`) fit across the entire series panel simultaneously. When multiple global or
   `deep_learning` models run in one job, `effective_worker_count` automatically allocates **1
   dedicated VM per model** (even when `workers` is left at its default `1`), and
   `partition_models_for_worker` assigns each model to its own replica (`idx % world_size == rank`)
   on the full series panel.

2. **Shared ThreadPoolExecutor + Multi-VM Sharding for Local (`statistical` / `ml`) Models**:
   Local per-series models share each VM via a profile-bounded `ThreadPoolExecutor`
   (`slots_per_unit` from `plan_vertex_pool`, enforcing `min(device, cores, memory)` with native
   thread-pool pinning via `intraop_env_vars` and Longest Processing Time first (`LPT`) ordering).
   When `workers > 1`:
   - If `hierarchy.enabled` is False, unique `ts_id` series are sorted and partitioned into
     contiguous ranges per worker (`shard_series_for_worker`) pushed directly into the BigQuery
     Storage Read API `row_restriction`.
   - If `hierarchy.enabled` is True, models are modulo-partitioned across VMs
     (`partition_models_for_worker`) so each VM holds the full series tree required to reconcile its
     assigned model(s) via FPP3 `reconcile_panel_results`.

3. **Hardware Sizing & Telemetry (`profiling/` + `resources/`)**:
   `plan_vertex_pool` translates `ComputeProfile` measurements (or static defaults) into a
   `RuntimeResourcePlan` over the VM's `UnitShape`, stamped at submit time (`$.sizing.<family>`) and
   after the on-VM measurement pre-pass (`$.sizing_executed.<family>`).

Public surface: ``WorkerTopology``, ``resolve_worker_topology``, ``effective_worker_count``,
``models_sharded_across_workers``, ``partition_models_for_worker``, ``shard_series_for_worker``,
``vertex_unit_shape``, ``plan_vertex_pool``, ``execute_panel``, ``run``.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from ..config import resolve_vm_machine_type
from ..errors import ConfigError, EngineError, get_logger
from ..profiling.source import resolve_profile
from ..resources.catalog import (
    device_floor_fraction,
    intraop_env_vars,
    machine_cores,
    machine_memory_bytes,
)
from ..resources.fleet import (
    RuntimeResourcePlan,
    UnitShape,
    max_slot_memory_bytes,
    plan_fleet,
    schedulable_cores,
)
from ..resources.slot import merge_slots, resource_slot
from ..worker import CellResult, is_panel_model
from . import ray_io
from .ray_engine import (
    _assert_source_supports_folds,
    _chunk_count,
    _create_read_session,
    _failed_chunk_status,
    _limit_series,
    _needed_columns,
    _read_source_series,
    _read_streams,
    _resolve_fleetwide_hpo,
    _stamp_executed_sizing,
)
from .spark_io import STATUS_COLUMNS, RunOutcome, aggregate_status, run_group

if TYPE_CHECKING:
    from ..config import RunConfig
    from ..profiling.cost import ComputeProfile
    from ..settings import Settings

_log = get_logger(__name__)

_WORKERPOOL_RE = re.compile(r"^workerpool(\d+)$")
_RAY_READ_SOURCE_SERIES = _read_source_series

# Relative per-model cost weights for intra-VM Longest-Processing-Time-First (LPT) chunk ordering
# when unprofiled. Heavy iterative / tree models start first in ThreadPoolExecutor so slow tail fits
# never serialize at the end of a VM run.
_MODEL_LPT_WEIGHTS: dict[str, float] = {
    "sarimax": 10.0,
    "auto_arima": 9.0,
    "prophet": 8.0,
    "neuralprophet": 8.0,
    "tft": 8.0,
    "patchtst": 7.0,
    "tide": 6.0,
    "tsmixer": 6.0,
    "xgboost": 4.0,
    "lightgbm": 3.0,
    "ets": 2.5,
    "theta": 1.5,
    "seasonal_naive": 1.0,
    "naive_mean": 1.0,
}


@dataclass(frozen=True)
class WorkerTopology:
    """One worker's 0-based ``rank`` inside a ``world_size`` worker pool (pure)."""

    rank: int = 0
    world_size: int = 1
    source: str = "single"

    def __post_init__(self) -> None:
        if self.world_size < 1:
            raise ConfigError(f"world_size must be >= 1, got {self.world_size}")
        if not (0 <= self.rank < self.world_size):
            raise ConfigError(
                f"worker rank {self.rank} is out of bounds for world_size={self.world_size}"
            )

    @property
    def is_primary(self) -> bool:
        """True for rank 0."""
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        """True when more than one worker participates in the job."""
        return self.world_size > 1

    def to_dict(self) -> dict[str, Any]:
        return {"rank": self.rank, "world_size": self.world_size, "source": self.source}


def _pool_sort_key(pool_name: str) -> tuple[int, int, str]:
    """Order ``workerpool0``, ``workerpool1``, ... numerically before non-standard pool names."""
    match = _WORKERPOOL_RE.match(pool_name)
    if match is not None:
        return (0, int(match.group(1)), pool_name)
    role_order = {"chief": 0, "master": 1, "worker": 2, "ps": 3, "evaluator": 4}
    if pool_name in role_order:
        return (1, role_order[pool_name], pool_name)
    return (2, 0, pool_name)


def resolve_worker_topology(
    worker_rank: int | None = None,
    worker_count: int | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> WorkerTopology:
    """Resolve ``(rank, world_size)`` from args, env vars, or Vertex AI ``CLUSTER_SPEC`` (pure).

    Precedence:
    1. Explicit ``worker_rank`` / ``worker_count`` arguments (`--worker-rank` / `--worker-count`).
    2. Explicit environment variables ``SF_WORKER_RANK`` / ``SF_WORKER_COUNT``.
    3. Kubernetes Indexed Job environment variables ``JOB_COMPLETION_INDEX`` /
       ``JOB_COMPLETION_TOTAL`` (or ``JOB_COMPLETION_COUNT`` / ``SF_WORKER_COUNT``).
    4. Vertex AI ``CLUSTER_SPEC`` JSON (`{"cluster": {"workerpool0": [...], "workerpool1": [...]},
       "task": {"type": "workerpool1", "index": 0}}`).
    5. Single-machine fallback ``WorkerTopology(rank=0, world_size=1, source="single")``.
    """
    env = os.environ if environ is None else environ

    if worker_rank is not None or worker_count is not None:
        resolved_count = int(worker_count) if worker_count is not None else 1
        resolved_rank = int(worker_rank) if worker_rank is not None else 0
        return WorkerTopology(rank=resolved_rank, world_size=resolved_count, source="cli")

    if "SF_WORKER_RANK" in env or "SF_WORKER_COUNT" in env:
        try:
            r = int(env["SF_WORKER_RANK"]) if "SF_WORKER_RANK" in env else 0
            w = int(env["SF_WORKER_COUNT"]) if "SF_WORKER_COUNT" in env else max(1, r + 1)
            return WorkerTopology(rank=r, world_size=w, source="env")
        except ValueError as exc:
            raise ConfigError(
                f"invalid worker topology in environment "
                f"(SF_WORKER_RANK={env.get('SF_WORKER_RANK')!r}, "
                f"SF_WORKER_COUNT={env.get('SF_WORKER_COUNT')!r}): {exc}"
            ) from exc

    if "JOB_COMPLETION_INDEX" in env:
        try:
            r = int(env["JOB_COMPLETION_INDEX"])
            raw_w = (
                env.get("JOB_COMPLETION_TOTAL")
                or env.get("JOB_COMPLETION_COUNT")
                or env.get("SF_WORKER_COUNT")
            )
            w = int(raw_w) if raw_w is not None else max(1, r + 1)
            return WorkerTopology(rank=r, world_size=w, source="k8s_indexed_job")
        except ValueError as exc:
            raise ConfigError(
                f"invalid Kubernetes Indexed Job topology "
                f"(JOB_COMPLETION_INDEX={env.get('JOB_COMPLETION_INDEX')!r}): {exc}"
            ) from exc

    raw_spec = env.get("CLUSTER_SPEC")
    if raw_spec:
        try:
            spec = json.loads(raw_spec)
            cluster = spec.get("cluster") or {}
            task = spec.get("task") or {}
            task_type = str(task.get("type", "workerpool0"))
            task_index = int(task.get("index", 0))
            if isinstance(cluster, dict) and cluster:
                ordered_pools = sorted(cluster.keys(), key=_pool_sort_key)
                world_size = sum(len(cluster.get(p) or []) for p in ordered_pools)
                if world_size >= 1 and task_type in cluster:
                    offset = sum(
                        len(cluster.get(p) or [])
                        for p in ordered_pools[: ordered_pools.index(task_type)]
                    )
                    rank = offset + task_index
                    return WorkerTopology(
                        rank=rank, world_size=world_size, source="vertex_cluster_spec"
                    )
        except (ValueError, TypeError, KeyError) as exc:
            _log.warning("could not parse CLUSTER_SPEC (%r); defaulting to single worker", exc)

    return WorkerTopology(rank=0, world_size=1, source="single")


def _is_dedicated_vm_model_set(cfg: RunConfig, models: Sequence[str]) -> bool:
    """True when every model in ``models`` is a panel model or belongs to ``deep_learning`` (pure).

    Global/hybrid panel models (`tide`, `tft`, `tsmixer`, `patchtst`, and `neuralprophet` with
    `global_fit=True`) fit across the entire series panel simultaneously, and `deep_learning`
    models (PyTorch Lightning trainers) get their own dedicated VM when multiple models run in one
    job rather than sharing a single VM sequentially.
    """
    from ..models import get_model

    if not models:
        return False
    return all(is_panel_model(m, cfg) or get_model(m).family == "deep_learning" for m in models)


def models_sharded_across_workers(cfg: RunConfig, models: Sequence[str]) -> bool:
    """True when multi-VM workers partition ``models`` by model index (each on the full panel).

    Happens in three cases:
    1. ``cfg.hierarchy.enabled`` is True — `reconcile_panel_results` reconciles each `model_type`
       independently across the entire series tree, so different VMs can run different models on the
       full panel in parallel.
    2. Every model in ``models`` is a panel (`global`/`hybrid`) model.
    3. Multiple `deep_learning` / panel models run in the same job (`len(models) > 1` and
       `_is_dedicated_vm_model_set`), giving each deep-learning model its own dedicated VM on the
       full panel.
    """
    if not models:
        return False
    if cfg.hierarchy.enabled:
        return True
    if all(is_panel_model(m, cfg) for m in models):
        return True
    return len(models) > 1 and _is_dedicated_vm_model_set(cfg, models)


def effective_worker_count(
    cfg: RunConfig,
    models: Sequence[str],
    requested_workers: int,
    *,
    n_series: int | None = None,
) -> int:
    """Resolve the effective VM replica count so each worker has non-empty work (pure).

    * **Dedicated per-model VMs for global / deep-learning sets**: when `models` contains $K > 1$
      global or `deep_learning` models (`_is_dedicated_vm_model_set`), a default
      `requested_workers == 1` automatically scales to `len(models)` so each model gets its own
      dedicated VM; an explicit `requested_workers > 1` is capped at `len(models)` when all models
      are panel models (or `models_sharded_across_workers` is True).
    * **Hierarchical reconciliation (`cfg.hierarchy.enabled`)**: models are partitioned across VMs
      while keeping the full hierarchy tree on each VM, so up to `min(requested_workers,
      len(models))` VMs can run in parallel.
    * **Local per-series models (`statistical`, `ml`, or single per-series `neuralprophet`)**:
      series are sharded across VMs, bounded by `n_series` when known.
    """
    if not models:
        return 1
    asked = max(1, int(requested_workers))
    if models_sharded_across_workers(cfg, models):
        if _is_dedicated_vm_model_set(cfg, models) and asked == 1 and len(models) > 1:
            return len(models)
        return min(asked, max(1, len(models)))
    if asked == 1:
        return 1
    if n_series is not None and n_series > 0:
        return min(asked, max(len(models), n_series))
    return asked


def partition_models_for_worker(
    cfg: RunConfig, models: Sequence[str], topology: WorkerTopology
) -> list[str]:
    """Deterministically assign the model subset that ``topology.rank`` should execute (pure).

    * Single worker (`world_size == 1`): executes all ``models``.
    * Model-sharded jobs (`models_sharded_across_workers` — hierarchy enabled, all panel models, or
      multiple deep-learning models): modulo-assigns `models` across workers (`idx % world_size ==
      rank`), with each worker running its assigned model(s) on the full series panel.
    * Mixed / local jobs: any panel models in `models` receive modulo assignment (`panel_idx %
      world_size == rank`), while local per-series models are kept on all active workers and sharded
      by `ts_id` via `shard_series_for_worker`.
    """
    if topology.world_size <= 1:
        return list(models)
    if models_sharded_across_workers(cfg, models):
        return [m for idx, m in enumerate(models) if idx % topology.world_size == topology.rank]
    assigned: list[str] = []
    panel_idx = 0
    for m in models:
        if is_panel_model(m, cfg):
            if panel_idx % topology.world_size == topology.rank:
                assigned.append(m)
            panel_idx += 1
        else:
            assigned.append(m)
    return assigned


def worker_series_slice(sorted_ids: Sequence[str], topology: WorkerTopology) -> list[str]:
    """Return the contiguous slice of ``sorted_ids`` assigned to ``topology.rank`` (pure).

    Partitions ``n = len(sorted_ids)`` into ``world_size`` contiguous, non-overlapping slices
    ``[(n * rank) // world_size : (n * (rank + 1)) // world_size]``. Contiguous slicing enables
    pushing ``ts_id >= '{start_id}' AND ts_id <= '{end_id}'`` directly into the BigQuery Storage
    Read API ``row_restriction`` so each VM in a multi-VM worker pool streams only its own 1/W
    slice of the table over Arrow gRPC.
    """
    if topology.world_size <= 1:
        return list(sorted_ids)
    n = len(sorted_ids)
    start_idx = (n * topology.rank) // topology.world_size
    end_idx = (n * (topology.rank + 1)) // topology.world_size
    return list(sorted_ids[start_idx:end_idx])


def shard_series_for_worker(
    series_or_panel: Sequence[str] | pd.DataFrame,
    topology: WorkerTopology,
    *,
    ts_id_col: str = "ts_id",
) -> list[str] | pd.DataFrame:
    """Deterministically shard a sequence of series IDs or panel into a contiguous slice (pure)."""
    if isinstance(series_or_panel, pd.DataFrame):
        panel = series_or_panel
        if topology.world_size <= 1 or panel.empty:
            return panel
        sorted_ids = sorted(panel[ts_id_col].astype(str).unique())
        assigned = set(worker_series_slice(sorted_ids, topology))
        if not assigned:
            return panel.iloc[0:0].copy()
        return panel[panel[ts_id_col].astype(str).isin(assigned)].reset_index(drop=True)

    sorted_ids = sorted(str(s) for s in series_or_panel)
    return worker_series_slice(sorted_ids, topology)


def _escape_sql_literal(val: str) -> str:
    return val.replace("\\", "\\\\").replace("'", "\\'")


def build_worker_series_range(
    ids: Iterable[Any],
    limit: int | None,
    id_col: str,
    topology: WorkerTopology,
    *,
    hpo_sample_n: int = 0,
) -> tuple[str | None, set[str], int]:
    """Build the BigQuery Storage Read API ``row_restriction`` for ``topology.rank`` (pure).

    Returns ``(row_restriction, worker_ids_set, n_series_total)``:
    - Because ``limited = sorted_ids[:limit]`` is a contiguous prefix of all distinct ``ts_id``s in
      the table and ``worker_list = worker_series_slice(limited, topology)`` is a contiguous
      sub-slice of ``limited``, there is no ``ts_id`` in the source table between ``worker_list[0]``
      and ``worker_list[-1]`` outside ``worker_list``. Thus ``ts_id >= '{start}' AND ts_id <=
      '{end}'`` pushes both ``series_limit`` and the worker's 1/W shard into a single server-side
      Storage Read API filter.
    - When fleetwide HPO is active (``hpo_sample_n > 0``) on a non-zero rank, ``OR ts_id <=
      '{hpo_bound}'`` is included so every worker tunes on the identical first-K series sample
      before filtering to ``worker_ids_set``.
    """
    distinct = sorted({str(i) for i in ids})
    if not distinct:
        return None, set(), 0
    limited = distinct[:limit] if limit is not None else distinct
    n_series_total = len(limited)
    worker_list = worker_series_slice(limited, topology)
    worker_set = set(worker_list)
    if not worker_list:
        return None, set(), n_series_total

    if topology.world_size <= 1:
        if limit is None or limit >= len(distinct):
            return None, worker_set, n_series_total
        end_esc = _escape_sql_literal(limited[-1])
        return f"{id_col} <= '{end_esc}'", worker_set, n_series_total

    start_esc = _escape_sql_literal(worker_list[0])
    end_esc = _escape_sql_literal(worker_list[-1])
    range_clause = f"{id_col} >= '{start_esc}' AND {id_col} <= '{end_esc}'"
    if hpo_sample_n > 0 and topology.rank > 0:
        hpo_idx = min(hpo_sample_n, len(limited)) - 1
        hpo_esc = _escape_sql_literal(limited[hpo_idx])
        range_clause = f"({range_clause}) OR {id_col} <= '{hpo_esc}'"
    return range_clause, worker_set, n_series_total


def _read_worker_source_series(
    cfg: RunConfig,
    settings: Settings,
    topology: WorkerTopology,
    assigned_models: Sequence[str],
) -> tuple[pd.DataFrame, set[str] | None, int]:  # pragma: no cover - live GCP I/O
    """Read only ``topology.rank``'s contiguous ``ts_id`` slice via BigQuery Storage Read API.

    For single-VM runs, model-sharded runs (hierarchy / global panel / multi-DL), or custom
    monkeypatched readers, falls back to `_read_source_series` (`worker_ids = None`).
    For multi-VM series-sharded local runs (`topology.world_size > 1`), performs a lightweight
    1-column `ts_id` scan to resolve `[start_id, end_id]` and streams only this worker's 1/W slice
    from BigQuery Storage Read API.
    """
    id_col = cfg.data.ts_id_col
    needs_full_panel = (
        topology.world_size <= 1
        or models_sharded_across_workers(cfg, assigned_models)
        or any(is_panel_model(m, cfg) for m in assigned_models)
        or cfg.compute.ray_read_mode == "ray_data"
        or _read_source_series is not _RAY_READ_SOURCE_SERIES
    )
    if needs_full_panel:
        source = _read_source_series(cfg, settings)
        n_total = int(source[id_col].nunique()) if not source.empty else 0
        return source, None, n_total

    from google.cloud.bigquery_storage_v1 import BigQueryReadClient

    read_client = BigQueryReadClient()
    id_session = _create_read_session(
        read_client, cfg, settings, fields=[id_col], row_restriction=None
    )
    id_frame = _read_streams(read_client, id_session, [id_col])
    hpo_sample_n = (
        cfg.hpo.sample_size if (cfg.hpo.enabled and cfg.hpo.granularity == "fleetwide") else 0
    )
    restriction, worker_ids, n_total = build_worker_series_range(
        id_frame[id_col],
        cfg.data.series_limit,
        id_col,
        topology,
        hpo_sample_n=hpo_sample_n,
    )
    if not worker_ids:
        return pd.DataFrame(columns=_needed_columns(cfg)), set(), n_total

    _log.info(
        "vertex read: rank %d/%d pushing contiguous shard row_restriction (%d/%d series): %s",
        topology.rank,
        topology.world_size,
        len(worker_ids),
        n_total,
        restriction,
    )
    cols = _needed_columns(cfg)
    session = _create_read_session(
        read_client, cfg, settings, fields=cols, row_restriction=restriction
    )
    panel = _read_streams(read_client, session, cols)
    if hpo_sample_n == 0:
        panel = panel[panel[id_col].astype(str).isin(worker_ids)].reset_index(drop=True)
    else:
        panel = _limit_series(panel, cfg)
    return panel, worker_ids, n_total


def order_chunks_lpt(
    chunks: Sequence[pd.DataFrame],
    models: Sequence[str] = (),
) -> list[pd.DataFrame]:
    """Order intra-VM chunks Longest-Processing-Time-First (LPT) to minimize tail latency (pure).

    Scores each chunk by its row count weighted by `_MODEL_LPT_WEIGHTS` so heavy models (`sarimax`,
    `auto_arima`, `prophet`, `xgboost`) and longer series histories start first in
    `ThreadPoolExecutor`, while fast chunks (`theta`, `naive_mean`) backfill around them.
    """
    if len(chunks) <= 1:
        return list(chunks)

    default_model_weight = sum(_MODEL_LPT_WEIGHTS.get(m, 2.0) for m in models) or 2.0

    def _chunk_weight(item: tuple[int, pd.DataFrame]) -> tuple[float, int]:
        idx, ch = item
        if ch.empty:
            return (0.0, idx)
        if ray_io._MODEL_COL in ch.columns:
            counts = ch[ray_io._MODEL_COL].astype(str).value_counts()
            score = sum(
                float(cnt) * _MODEL_LPT_WEIGHTS.get(str(m), 2.0) for m, cnt in counts.items()
            )
        else:
            score = float(len(ch)) * default_model_weight
        return (-score, idx)

    return [ch for _, ch in sorted(enumerate(chunks), key=_chunk_weight)]


def vertex_unit_shape(
    cfg: RunConfig,
    *,
    gpu: bool = False,
    gpu_type: str | None = None,
    machine_type: str | None = None,
) -> UnitShape:
    """One VM worker of a Vertex CustomJob or GCE instance: cores, RAM, and accelerators (pure)."""
    resolved_machine = resolve_vm_machine_type(
        "gpu" if gpu else "cpu",
        gpu_type,
        cfg.compute.machine_type,
        machine_type,
        accelerator_count=cfg.compute.accelerator_count if gpu else 1,
    )
    return UnitShape(
        cores=machine_cores(resolved_machine),
        memory_bytes=machine_memory_bytes(resolved_machine),
        accelerators=cfg.compute.accelerator_count if gpu else 0,
    )


def plan_vertex_pool(
    cfg: RunConfig,
    models: Sequence[str],
    n_cells: int,
    *,
    runtime: str = "vertex",
    gpu: bool = False,
    gpu_type: str | None = None,
    machine_type: str | None = None,
    worker_count: int = 1,
    profile: ComputeProfile | None = None,
) -> RuntimeResourcePlan:
    """Size the per-VM slot density and worker fleet for a Vertex CustomJob or GCE VM (pure).

    Uses the shared `resources.slot.resource_slot` + `resources.fleet.plan_fleet` algebra:
    - Each cell's `ResourceSlot` is derived from `profile` (or static defaults when unprofiled).
    - Because `"vertex"` and `"gce"` are not in `_UNENFORCED_AXES`, `plan_fleet` enforces all three
      axes (`min(device, cores, memory)`) against `schedulable_cores(unit)` and
      `schedulable_memory_bytes(unit)`.
    - `plan.slots_per_unit` bounds `ThreadPoolExecutor(max_workers=...)` on each VM, and
      `intraop_env_vars(plan.slot.cores, include_omp=True)` pins native BLAS/OMP threads per slot.
    """
    unit = vertex_unit_shape(cfg, gpu=gpu, gpu_type=gpu_type, machine_type=machine_type)
    effective_gpu_type = (gpu_type or cfg.compute.gpu_type).upper() if gpu else None
    families = ray_io.pool_families(list(models)) or ["deep_learning" if gpu else "cpu"]
    static_frac = (
        ray_io._NOMINAL_AUTO_FRACTION
        if cfg.compute.gpu_fraction == "auto"
        else float(cfg.compute.gpu_fraction)
    )
    slots = [
        resource_slot(
            profile,
            family,
            use_gpu=gpu,
            device_bytes=ray_io.device_memory_bytes(effective_gpu_type) if gpu else None,
            static_gpu_fraction=static_frac,
            min_gpu_fraction=device_floor_fraction(schedulable_cores(unit), unit.accelerators),
            max_cores=unit.cores if unit.cores > 0 else None,
            max_memory_bytes=max_slot_memory_bytes(unit),
        )
        for family in families
    ]
    units = max(1, worker_count)
    return plan_fleet(
        merge_slots(slots, family="+".join(families)),
        runtime=runtime,
        n_cells=n_cells,
        unit=unit,
        target_cells_per_slot=cfg.compute.ray_target_cells_per_slot,
        min_units=units,
        max_units=units,
    )


def execute_panel(
    source: pd.DataFrame,
    cfg: RunConfig,
    models: list[str] | None = None,
    *,
    topology: WorkerTopology | None = None,
    params_by_model: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[CellResult], pd.DataFrame]:
    """Pure, in-memory execution of one worker's slice of ``source`` (no GCP I/O).

    Partitions models and/or shards series according to ``topology`` and delegates to the shared
    `spark_io.run_group` kernel so local, global/hybrid, and hierarchical reconciliation behave
    identically to Spark and Ray.
    """
    topo = topology or WorkerTopology(rank=0, world_size=1, source="single")
    executed = models if models is not None else list(cfg.models)
    assigned_models = partition_models_for_worker(cfg, executed, topo)
    if not assigned_models or source.empty:
        return [], pd.DataFrame(columns=list(STATUS_COLUMNS))

    if models_sharded_across_workers(cfg, executed):
        return run_group(source, cfg, assigned_models, params_by_model)

    by_model = params_by_model or {}
    panel_models = [m for m in assigned_models if is_panel_model(m, cfg, by_model.get(m))]
    local_models = [m for m in assigned_models if m not in panel_models]

    all_results: list[CellResult] = []
    status_frames: list[pd.DataFrame] = []

    if panel_models:
        res_p, stat_p = run_group(source, cfg, panel_models, params_by_model)
        all_results.extend(res_p)
        status_frames.append(stat_p)

    if local_models:
        sharded = shard_series_for_worker(source, topo, ts_id_col=cfg.data.ts_id_col)
        assert isinstance(sharded, pd.DataFrame)
        if not sharded.empty:
            res_l, stat_l = run_group(sharded, cfg, local_models, params_by_model)
            all_results.extend(res_l)
            status_frames.append(stat_l)

    if not status_frames:
        return [], pd.DataFrame(columns=list(STATUS_COLUMNS))
    return all_results, pd.concat(status_frames, ignore_index=True)


def _run_chunks_on_vm(
    chunks: list[pd.DataFrame],
    models: list[str],
    cfg: RunConfig,
    settings: Settings,
    params_by_model: dict[str, dict[str, Any]] | None,
    *,
    parallel: bool,
    max_concurrency: int | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Run ``chunks`` on the current VM using `ray_io.make_chunk_runner` and collect statuses."""
    if not chunks or not models:
        return pd.DataFrame(columns=list(STATUS_COLUMNS)), []

    ordered_chunks = order_chunks_lpt(chunks, models)
    runner = ray_io.make_chunk_runner(cfg, settings, models, params_by_model)
    n_cpus = max(1, os.cpu_count() or 1)
    slot_cap = max_concurrency if max_concurrency is not None else max(1, n_cpus - 1)
    frames: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []

    if not parallel or len(ordered_chunks) <= 1 or slot_cap <= 1:
        for chunk in ordered_chunks:
            try:
                frames.append(runner(chunk))
            except Exception as exc:  # noqa: BLE001
                _log.warning("vertex chunk failed (cells recorded as error): %r", exc)
                failures.append(
                    {
                        "error": repr(exc),
                        "n_series": int(chunk[cfg.data.ts_id_col].nunique()),
                        "models": list(models),
                    }
                )
                frames.append(_failed_chunk_status(chunk, cfg, models, exc))
    else:
        max_workers = max(1, min(len(ordered_chunks), slot_cap))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_chunk = {pool.submit(runner, c): c for c in ordered_chunks}
            for fut in as_completed(future_to_chunk):
                chunk = future_to_chunk[fut]
                try:
                    frames.append(fut.result())
                except Exception as exc:  # noqa: BLE001
                    _log.warning("vertex chunk failed (cells recorded as error): %r", exc)
                    failures.append(
                        {
                            "error": repr(exc),
                            "n_series": int(chunk[cfg.data.ts_id_col].nunique()),
                            "models": list(models),
                        }
                    )
                    frames.append(_failed_chunk_status(chunk, cfg, models, exc))

    if not frames:
        return pd.DataFrame(columns=list(STATUS_COLUMNS)), failures
    return pd.concat(frames, ignore_index=True), failures


def _executed_vm_sizing_patch(
    plan: RuntimeResourcePlan,
    models: Sequence[str],
    topology: WorkerTopology,
) -> dict[str, Any]:
    """Header patch describing the VM pool sized on the worker after reading the panel (pure)."""
    from ..registry.header import executed_sizing_path

    families = ray_io.pool_families(list(models))
    if not families:
        return {}
    path = executed_sizing_path("+".join(families))
    return {
        path: {
            "vm_pool": plan.to_dict(),
            "topology": topology.to_dict(),
        }
    }


def _barrier_gcs_parts(
    settings: Settings,
    run_id: str,
    barrier_id: str,
    rank: int,
) -> tuple[str, str]:
    """Return `(bucket_name, object_path)` for a worker's barrier marker in `artifact_root`."""
    root = settings.artifact_root
    if root.startswith("gs://"):
        without = root[5:]
        bucket, _, prefix = without.partition("/")
        obj = f"{prefix.rstrip('/')}/{run_id}/_vertex_barrier/{barrier_id}/rank_{rank}.json"
        return bucket, obj.lstrip("/")
    return "", ""


def _write_worker_barrier_marker(
    settings: Settings,
    run_id: str,
    barrier_id: str,
    topology: WorkerTopology,
    payload: dict[str, Any],
) -> None:  # pragma: no cover - live GCS I/O
    """Write this worker's completion marker JSON to GCS so rank 0 waits for all replicas."""
    bucket_name, obj_path = _barrier_gcs_parts(settings, run_id, barrier_id, topology.rank)
    if not bucket_name or not obj_path:
        return
    try:
        from google.cloud import storage

        client = storage.Client(project=settings.project_id)
        blob = client.bucket(bucket_name).blob(obj_path)
        blob.upload_from_string(
            json.dumps(payload, sort_keys=True),
            content_type="application/json",
        )
    except Exception as exc:  # noqa: BLE001 - best-effort barrier write
        _log.warning(
            "vertex barrier marker upload failed for rank=%d/%d (%s): %r",
            topology.rank,
            topology.world_size,
            barrier_id,
            exc,
        )


def _wait_for_worker_pool_barrier(
    settings: Settings,
    run_id: str,
    barrier_id: str,
    topology: WorkerTopology,
    *,
    timeout_s: float = 1800.0,
    poll_interval_s: float = 3.0,
) -> list[dict[str, Any]]:  # pragma: no cover - live GCS I/O
    """Wait on rank 0 until all secondary workers (`1 .. world_size - 1`) upload their markers.

    In Vertex AI ``CustomJob``, ``worker_pool_specs[0]`` (rank 0) is the primary replica whose
    container exit immediately terminates ``worker_pool_specs[1]`` (ranks ``1 .. world_size - 1``).
    Without this barrier, a fast model on rank 0 (e.g. ``theta``) would exit and kill a slower
    sibling model on rank 1 (e.g. ``sarimax``) before rank 1 finishes writing to BigQuery.
    """
    if not topology.is_distributed or not topology.is_primary:
        return []
    bucket_name, _ = _barrier_gcs_parts(settings, run_id, barrier_id, 0)
    if not bucket_name:
        return []

    from google.cloud import storage

    client = storage.Client(project=settings.project_id)
    bucket = client.bucket(bucket_name)
    pending_ranks = set(range(1, topology.world_size))
    collected: dict[int, dict[str, Any]] = {}
    deadline = time.monotonic() + timeout_s

    _log.info(
        "vertex barrier wait: rank 0/%d waiting for peer ranks %s (barrier_id=%s)",
        topology.world_size,
        sorted(pending_ranks),
        barrier_id,
    )
    while pending_ranks and time.monotonic() < deadline:
        for r in list(pending_ranks):
            _, obj_path = _barrier_gcs_parts(settings, run_id, barrier_id, r)
            blob = bucket.blob(obj_path)
            try:
                if blob.exists():
                    doc = json.loads(blob.download_as_text())
                    if isinstance(doc, dict):
                        collected[r] = doc
                        pending_ranks.discard(r)
                        _log.info(
                            "vertex barrier: received marker from rank %d/%d (status=%s cells=%s)",
                            r,
                            topology.world_size,
                            doc.get("status"),
                            doc.get("n_cells"),
                        )
            except Exception as exc:  # noqa: BLE001 - transient GCS read
                _log.warning("vertex barrier poll error for rank %d: %r", r, exc)
        if pending_ranks:
            time.sleep(poll_interval_s)

    if pending_ranks:
        raise EngineError(
            f"Vertex worker-pool barrier timed out after {timeout_s:.0f}s waiting for peer "
            f"rank(s) {sorted(pending_ranks)} in {barrier_id}"
        )

    for r, doc in sorted(collected.items()):
        if doc.get("error"):
            raise EngineError(
                f"Vertex worker rank {r}/{topology.world_size} failed in {barrier_id}: "
                f"{doc['error']}"
            )
    return [collected[r] for r in sorted(collected)]


def _should_use_worker_barrier(
    topology: WorkerTopology,
    *,
    manage_header: bool = True,
) -> bool:
    """True when rank 0 must block at a GCS barrier until peer workers (`1..W-1`) finish.

    - On Vertex AI ``CustomJob`` (``source == "vertex_cluster_spec"``), ``worker_pool_specs[0]``
      (rank 0) is the chief replica whose exit immediately terminates ``worker_pool_specs[1]``
      (ranks ``1..W-1``), so rank 0 must always wait for peers.
    - On Kubernetes Indexed Jobs (``source == "k8s_indexed_job"``), each pod (``0..W-1``) is
      tracked independently by the Kubernetes ``batch/v1`` ``Job`` controller until
      ``succeeded == completions``. When ``manage_header=False`` (orchestrated via ``main.run`` /
      ``gke_submit``), pods exit immediately as soon as their own model or shard finishes so GKE
      Cluster Autoscaler can scale down completed nodes while slower pods continue running.
    """
    if not topology.is_distributed:
        return False
    if topology.source in ("k8s_indexed_job", "JOB_COMPLETION_INDEX") and not manage_header:
        return False
    return topology.source in (
        "CLUSTER_SPEC",
        "JOB_COMPLETION_INDEX",
        "vertex_cluster_spec",
        "k8s_indexed_job",
    ) or bool(os.environ.get("SF_VERTEX_JOB_ID"))


def run(
    cfg: RunConfig,
    models: list[str] | None = None,
    *,
    manage_header: bool = True,
    settings: Settings | None = None,
    worker_rank: int | None = None,
    worker_count: int | None = None,
) -> RunOutcome:
    """Execute a Vertex AI ``CustomJob``, GCE VM, or GKE Indexed Job worker end-to-end.

    Structural twin of `ray_engine.run` and `spark_explode.run`, invoked by `vertex_entry`.
    """
    from ..registry.ids import make_run_id
    from ..registry.lifecycle import run_header
    from ..settings import Settings as _Settings

    settings = settings or _Settings.resolve()
    run_id = make_run_id(cfg)
    executed = models if models is not None else list(cfg.models)
    topology = resolve_worker_topology(worker_rank=worker_rank, worker_count=worker_count)
    raw_gpu, job_gpu_type = ray_io.resolve_job_gpu(cfg)
    job_gpu = raw_gpu and ("deep_learning" in ray_io.pool_families(list(executed)))
    runtime_name = os.environ.get("SF_VM_RUNTIME", "vertex")
    use_barrier = _should_use_worker_barrier(topology, manage_header=manage_header)
    barrier_id = (
        os.environ.get("SF_VERTEX_JOB_ID")
        or os.environ.get("CLOUD_ML_JOB_ID")
        or f"{run_id}-{'+'.join(ray_io.pool_families(list(executed)) or ['default'])}"
    )

    # Secondary workers in a multi-worker pool never manage the shared run_registry header.
    owns_header = manage_header and topology.is_primary
    _log.info(
        "%s run start: run_id=%s rank=%d/%d (%s) models=%s use_gpu=%s manage_header=%s",
        runtime_name,
        run_id,
        topology.rank,
        topology.world_size,
        topology.source,
        executed,
        job_gpu,
        owns_header,
    )

    with run_header(cfg, run_id, settings=settings, manage=owns_header) as hdr:
        started = time.perf_counter()
        try:
            assigned_models = partition_models_for_worker(cfg, executed, topology)
            if not assigned_models:
                _log.info(
                    "%s worker %d/%d has no assigned models; standing down without reading source",
                    runtime_name,
                    topology.rank,
                    topology.world_size,
                )
                outcome = aggregate_status(pd.DataFrame(columns=list(STATUS_COLUMNS)))
                if use_barrier:
                    _write_worker_barrier_marker(
                        settings,
                        run_id,
                        barrier_id,
                        topology,
                        {
                            "rank": topology.rank,
                            "world_size": topology.world_size,
                            "status": outcome.status,
                            "n_cells": 0,
                            "n_ok": 0,
                            "n_error": 0,
                        },
                    )
                    if topology.is_primary:
                        _wait_for_worker_pool_barrier(settings, run_id, barrier_id, topology)
                return outcome

            source, worker_ids, n_series_total = _read_worker_source_series(
                cfg, settings, topology, assigned_models
            )
            _assert_source_supports_folds(source, cfg)

            if source.empty:
                _log.info(
                    "%s worker %d/%d has empty source slice; standing down cleanly",
                    runtime_name,
                    topology.rank,
                    topology.world_size,
                )
                outcome = aggregate_status(pd.DataFrame(columns=list(STATUS_COLUMNS)))
                if use_barrier:
                    _write_worker_barrier_marker(
                        settings,
                        run_id,
                        barrier_id,
                        topology,
                        {
                            "rank": topology.rank,
                            "world_size": topology.world_size,
                            "status": outcome.status,
                            "n_cells": 0,
                            "n_ok": 0,
                            "n_error": 0,
                        },
                    )
                    if topology.is_primary:
                        _wait_for_worker_pool_barrier(settings, run_id, barrier_id, topology)
                return outcome

            params_by_model = _resolve_fleetwide_hpo(source, cfg, assigned_models)

            # Measure per-family slot cost (gated by compute.profile) and size the VM concurrency.
            profile = resolve_profile(source, cfg, assigned_models, params_by_model=params_by_model)
            if n_series_total <= 0:
                n_series_total = int(source[cfg.data.ts_id_col].nunique())
            exec_plan = plan_vertex_pool(
                cfg,
                executed,
                n_series_total * len(executed),
                runtime=runtime_name,
                gpu=job_gpu,
                gpu_type=job_gpu_type,
                worker_count=topology.world_size,
                profile=profile,
            )
            if cfg.compute.profile.measure != "controlled":
                for env_key, env_val in intraop_env_vars(
                    exec_plan.slot.cores, include_omp=True
                ).items():
                    os.environ.setdefault(env_key, env_val)

            _log.info("%s sizing: %s", runtime_name, exec_plan.to_dict())
            if exec_plan.density_note:
                _log.warning("%s sizing: %s", runtime_name, exec_plan.density_note)
            if topology.is_primary:
                _stamp_executed_sizing(
                    run_id,
                    _executed_vm_sizing_patch(exec_plan, executed, topology),
                    settings,
                )

            target = cfg.compute.bucket_target_cells
            status_frames: list[pd.DataFrame] = []
            all_failures: list[dict[str, Any]] = []
            max_concurrency = exec_plan.slots_per_unit

            if models_sharded_across_workers(cfg, executed):
                # Each worker holds the full series panel for its assigned model(s) (hierarchy
                # reconciliation, global panel models, or dedicated per-model deep_learning VMs).
                chunks = ray_io.chunk_cells(source, cfg, assigned_models, len(assigned_models))
                stat_df, fails = _run_chunks_on_vm(
                    chunks,
                    assigned_models,
                    cfg,
                    settings,
                    params_by_model,
                    parallel=(not job_gpu and len(assigned_models) > 1),
                    max_concurrency=max_concurrency,
                )
                status_frames.append(stat_df)
                all_failures.extend(fails)
            else:
                by_model = params_by_model or {}
                panel_models = [
                    m for m in assigned_models if is_panel_model(m, cfg, by_model.get(m))
                ]
                local_models = [m for m in assigned_models if m not in panel_models]

                if panel_models:
                    panel_chunks = ray_io.chunk_cells(source, cfg, panel_models, len(panel_models))
                    stat_p, fails_p = _run_chunks_on_vm(
                        panel_chunks,
                        panel_models,
                        cfg,
                        settings,
                        params_by_model,
                        parallel=False,
                        max_concurrency=max_concurrency,
                    )
                    status_frames.append(stat_p)
                    all_failures.extend(fails_p)

                if local_models:
                    if worker_ids is not None:
                        sharded = source[
                            source[cfg.data.ts_id_col].astype(str).isin(worker_ids)
                        ].reset_index(drop=True)
                    else:
                        sharded_res = shard_series_for_worker(
                            source, topology, ts_id_col=cfg.data.ts_id_col
                        )
                        assert isinstance(sharded_res, pd.DataFrame)
                        sharded = sharded_res
                    if not sharded.empty:
                        n_series_shard = int(sharded[cfg.data.ts_id_col].nunique())
                        gpu_local, cpu_local = ray_io.split_gpu_cpu_models(
                            cfg, local_models, use_gpu=job_gpu
                        )
                        if gpu_local:
                            gpu_chunks = ray_io.chunk_cells(
                                sharded,
                                cfg,
                                gpu_local,
                                _chunk_count(n_series_shard * len(gpu_local), target),
                            )
                            stat_g, fails_g = _run_chunks_on_vm(
                                gpu_chunks,
                                gpu_local,
                                cfg,
                                settings,
                                params_by_model,
                                parallel=False,
                                max_concurrency=max_concurrency,
                            )
                            status_frames.append(stat_g)
                            all_failures.extend(fails_g)
                        if cpu_local:
                            cpu_chunks = ray_io.chunk_cells(
                                sharded,
                                cfg,
                                cpu_local,
                                _chunk_count(n_series_shard * len(cpu_local), target),
                            )
                            stat_c, fails_c = _run_chunks_on_vm(
                                cpu_chunks,
                                cpu_local,
                                cfg,
                                settings,
                                params_by_model,
                                parallel=True,
                                max_concurrency=max_concurrency,
                            )
                            status_frames.append(stat_c)
                            all_failures.extend(fails_c)

            if all_failures and topology.is_primary:
                _stamp_executed_sizing(run_id, {"chunk_failures": all_failures}, settings)

            status_pdf = (
                pd.concat(status_frames, ignore_index=True)
                if status_frames
                else pd.DataFrame(columns=list(STATUS_COLUMNS))
            )
            outcome = aggregate_status(status_pdf)
            if use_barrier:
                _write_worker_barrier_marker(
                    settings,
                    run_id,
                    barrier_id,
                    topology,
                    {
                        "rank": topology.rank,
                        "world_size": topology.world_size,
                        "status": outcome.status,
                        "n_cells": outcome.n_cells,
                        "n_ok": outcome.n_ok,
                        "n_error": outcome.n_error,
                    },
                )
                if topology.is_primary:
                    _wait_for_worker_pool_barrier(settings, run_id, barrier_id, topology)
        except Exception as exc:
            if use_barrier:
                _write_worker_barrier_marker(
                    settings,
                    run_id,
                    barrier_id,
                    topology,
                    {
                        "rank": topology.rank,
                        "world_size": topology.world_size,
                        "status": "FAILED",
                        "error": repr(exc),
                    },
                )
            raise

        runtime_seconds = time.perf_counter() - started
        hdr.finalize(status=outcome.status, n_series=outcome.n_series)

    _log.info(
        "%s run done: run_id=%s rank=%d/%d status=%s cells=%d ok=%d error=%d runtime=%.1fs",
        runtime_name,
        run_id,
        topology.rank,
        topology.world_size,
        outcome.status,
        outcome.n_cells,
        outcome.n_ok,
        outcome.n_error,
        runtime_seconds,
    )
    return outcome
