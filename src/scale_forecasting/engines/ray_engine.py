"""Ray on Vertex — the on-cluster driver: read → route GPU/CPU → fan chunks → close.

The Ray analog of `run`, and its structural twin:
header → fan cells across the cluster → aggregate statuses → close header. Everything reusable —
the per-cell work (`run_group`), the executor-side write
(`write_cells`), and the run-level roll-up
(`aggregate_status`) — is shared verbatim through
`ray_io`; this module owns only the Ray-specific driver shell.

**What's different from Spark, and why Ray is in the design.** The models are split
into a GPU pool (NeuralProphet — ``family == "deep_learning"``) and a CPU pool (everything else).
GPU cells run in ``@ray.remote(num_gpus=<fraction>)`` tasks that *pack several onto one T4* — the
fractional-GPU sharing Spark can't do — while CPU cells run in ``@ray.remote(num_cpus=1)`` tasks.
Both pools run the exact same chunk runner. The cluster they land on autoscales per pool by default,
planned at submit time by `plan_cluster`; this driver just fans work across
whatever the cluster is.

Runs on the Ray cluster head via `ray_entry` (the Jobs API entrypoint).

Public surface: ``run(cfg, models=None, *, manage_header=True) -> None``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from ..errors import get_logger
from ..profiling.source import resolve_profile
from ..resources.fleet import RuntimeResourcePlan, tasks_for_ceiling
from . import ray_io
from .spark_io import (
    _MODEL_COL,
    STATUS_COLUMNS,
    _needed_columns,
    _resolve_source_table,
    _snapshot_millis,
)

if TYPE_CHECKING:
    import pandas as pd

    from ..config import RunConfig
    from ..profiling.cost import ComputeProfile
    from ..settings import Settings

_log = get_logger(__name__)

# How many Storage Read API streams `_read_driver_collect` consumes at once. The reads are network-
# and Arrow-decode-bound rather than compute-bound, so this is a concurrency cap, not a core count;
# it exists to keep a large stream count from opening an unbounded number of gRPC channels on the
# driver. Deliberately a constant and not a config field: `compute.read_max_streams` already caps
# how many streams the *session* asks for, and every ComputeConfig field moves the run_id digest.
_MAX_READ_THREADS = 16

# Ray-level retries for a chunk task, and only for the two failures where no Python `except` ever
# runs (see the call site). Small on purpose: a crash that repeats twice is a real defect, and each
# replay re-runs a chunk's whole model fit.
_CHUNK_MAX_RETRIES = 2


def _storage_table_path(cfg: RunConfig, settings: Settings) -> str:
    """Resolve the source to a Storage Read API path ``projects/P/datasets/D/tables/T`` (pure).

    `_resolve_source_table` yields the BigQuery
    ``project.dataset.table`` form (qualifying a bare name against the deployment dataset); the
    Storage Read API wants the resource-path form. A two-part ``dataset.table`` (a caller-qualified
    source in another dataset of the same project) is prefixed with ``settings.project_id``.
    """
    ref = _resolve_source_table(cfg, settings)
    parts = ref.split(".")
    if len(parts) == 3:
        project, dataset, table = parts
    elif len(parts) == 2:
        project, (dataset, table) = settings.project_id, parts
    else:  # pragma: no cover - _resolve_source_table always yields a qualified ref
        raise ValueError(f"cannot resolve source table to a storage path: {ref!r}")
    return f"projects/{project}/datasets/{dataset}/tables/{table}"


def _limit_series(source: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    """Keep the first ``series_limit`` ts_ids (ordered); pass-through when unset (pure).

    The pandas twin of `spark_io._limit_series`: distinct ts_ids → ordered → first N → filter, so
    Ray and Spark subset the *same* series at every scale — the property that makes the
    "10 vs 100 vs 100k" runtime comparison apples-to-apples.

    Still applied client-side, but no longer as the *only* limiter. `build_series_bound` now pushes
    an equivalent bound into the read as a ``row_restriction`` (see there for why a range and not
    an ``IN`` list), so on the default reader this usually has nothing left to drop and costs one
    pass over an already-subset frame. It stays because it is the cheap idempotent check that the
    two rules agree, and because the ``ray_data`` reader has no restriction knob on its table-scan
    form — there, this is still what enforces the limit.
    """
    limit = cfg.data.series_limit
    if limit is None:
        return source
    id_col = cfg.data.ts_id_col
    keep = sorted(source[id_col].unique())[:limit]
    return source[source[id_col].isin(keep)].reset_index(drop=True)


def build_series_bound(ids: Iterable[Any], limit: int | None, id_col: str) -> str | None:
    """The Storage Read ``row_restriction`` that keeps the first ``limit`` series ids (pure).

    ``None`` means "read everything": no limit was asked for, the source is empty, or the limit
    already covers every id there is — a filter that excludes nothing is worse than no filter.

    **A range bound, not a set.** The obvious encoding of "these N series" is
    ``ts_id IN ('a','b',…)``, and it does not survive contact with the sizes this runs at: 10,000
    ids is roughly 130 KB of filter text, well past what the service will accept in a read session.
    A bound is one comparison however many series it selects. It works because the subset rule is
    already an *ordered* first-N — ``sorted(distinct)[:N]`` — so "the first N ids" and "every id
    ``<=`` the Nth" describe the same set. Python orders strings by code point and BigQuery
    compares STRING by UTF-8 bytes, which is the same order, so the boundary the driver picks is
    the boundary the service applies.

    The value is a SQL string literal, so a backslash or a quote in a series id has to be escaped
    or the filter is a syntax error at best.
    """
    if limit is None:
        return None
    distinct = sorted({str(i) for i in ids})
    if not distinct or limit >= len(distinct):
        return None
    boundary = distinct[limit - 1]
    escaped = boundary.replace("\\", "\\\\").replace("'", "\\'")
    return f"{id_col} <= '{escaped}'"


def _read_source_series(cfg: RunConfig, settings: Settings) -> pd.DataFrame:
    """Read the source series to a driver-side pandas panel, then apply the ``series_limit`` subset.

    The Ray analog of `read_source_series`. Two readers
    chosen by ``cfg.compute.ray_read_mode`` — both hit the **same** BigQuery Storage Read API (no
    query slots, matching Spark) and return the **same** column-projected pandas panel, so the
    downstream fan-out is byte-identical whichever runs:

    * ``driver_collect`` (default) — `_read_driver_collect`, the ``BigQueryReadClient`` path
      the @gpu smoke and the 100k run are proven on.
    * ``ray_data`` — `_read_ray_data`, the Ray-native ``ray.data.read_bigquery`` reader.

    The whole panel lands on the driver either way, then `chunk_cells` shards it into
    task-sized frames — acceptable because Ray is the GPU path for modest scales, not the 100k hero
    (that's Spark). The deterministic ``series_limit`` subset (`_limit_series`) is
    applied here so both readers subset identically.
    """
    reader = _read_ray_data if cfg.compute.ray_read_mode == "ray_data" else _read_driver_collect
    return _limit_series(reader(cfg, settings), cfg)


def _read_driver_collect(
    cfg: RunConfig, settings: Settings
) -> pd.DataFrame:  # pragma: no cover - GCP I/O, exercised by the @gpu smoke
    """Read the source panel via the BigQuery Storage Read API (``BigQueryReadClient``).

    Like the Spark connector, this reads through the **Storage Read API**, *not* ``client.query()``:
    a direct columnar table read over the storage layer, so it consumes no BigQuery query slots and
    streams Arrow straight to the driver (matching Spark). The read is column-projected
    to only what a cell needs (`_needed_columns` → ``selected_fields``), and **row-restricted** to
    the ``series_limit`` subset (`_series_bound`) so a 100-series run off a 100k-series table
    transfers 100 series rather than reading the table and throwing 99.9% of it away on the driver.

    Pinned to the run's input snapshot (`_snapshot_millis`) via the read session's
    ``table_modifiers.snapshot_time`` — the Storage Read API's native time-travel field, so the
    read consumes no query slots yet still sees the identical source state every other job in the
    run does. Unset snapshot → an un-pinned live read (the pre-snapshot behavior).
    """
    from google.cloud.bigquery_storage_v1 import BigQueryReadClient

    read_client = BigQueryReadClient()
    restriction = _series_bound(read_client, cfg, settings)
    session = _create_read_session(
        read_client, cfg, settings, fields=_needed_columns(cfg), row_restriction=restriction
    )
    return _read_streams(read_client, session, _needed_columns(cfg))


def _create_read_session(
    read_client: Any,
    cfg: RunConfig,
    settings: Settings,
    *,
    fields: list[str],
    row_restriction: str | None,
) -> Any:  # pragma: no cover - GCP I/O, exercised by the @gpu smoke
    """One Storage Read session over the source table, column-projected and optionally filtered.

    Pinned to the run's input snapshot (`_snapshot_millis`) when the run has one, so every session
    this module opens — the boundary pass and the panel read alike — time-travels to the identical
    instant. That matters more than it looks: a bound resolved against a table that then gains a
    series would otherwise select a different set than the driver thinks it did.
    """
    from google.cloud.bigquery_storage_v1 import types

    options = types.ReadSession.TableReadOptions(selected_fields=fields)
    if row_restriction:
        options.row_restriction = row_restriction
    requested = types.ReadSession(
        table=_storage_table_path(cfg, settings),
        data_format=types.DataFormat.ARROW,
        read_options=options,
    )
    ms = _snapshot_millis(cfg, settings)
    if ms is not None:
        # proto-plus surfaces the Timestamp field as a datetime, so set the whole modifiers
        # sub-message from a UTC datetime rather than mutating a Timestamp in place.
        from datetime import UTC, datetime

        requested.table_modifiers = types.ReadSession.TableModifiers(
            snapshot_time=datetime.fromtimestamp(ms / 1000, tz=UTC)
        )
    return read_client.create_read_session(
        parent=f"projects/{settings.project_id}",
        read_session=requested,
        # 0 (default) lets the server pick the stream count from the table size;
        # compute.read_max_streams caps it to bound read parallelism (shared with the Spark reader).
        max_stream_count=cfg.compute.read_max_streams,
    )


def _read_streams(
    read_client: Any, session: Any, columns: list[str]
) -> pd.DataFrame:  # pragma: no cover - GCP I/O, exercised by the @gpu smoke
    """Drain a read session's streams concurrently into one pandas frame, in stream order.

    The service splits the table into N independent streams precisely so they can be consumed
    concurrently; reading them one at a time serializes the parallelism it just handed us. Each
    ``read_rows()`` is gRPC transfer plus Arrow decode and both release the GIL, so threads are the
    right tool. ``map`` yields in input order, so the concatenated panel is byte-identical to the
    serial read — which matters because `_limit_series` subsets off this frame's row order.
    """
    # Runtime import: pandas is TYPE_CHECKING-only at module scope (offline import parity), so every
    # function that touches pandas at runtime must import it locally.
    from concurrent.futures import ThreadPoolExecutor

    import pandas as pd

    stream_names = [stream.name for stream in session.streams]
    if not stream_names:  # empty table → an empty, correctly-typed frame from the session schema
        return pd.DataFrame(columns=columns)

    def _stream_frame(name: str) -> pd.DataFrame:
        return read_client.read_rows(name).to_dataframe(session)

    if len(stream_names) == 1:
        return _stream_frame(stream_names[0])
    with ThreadPoolExecutor(max_workers=min(len(stream_names), _MAX_READ_THREADS)) as pool:
        frames = list(pool.map(_stream_frame, stream_names))
    return pd.concat(frames, ignore_index=True)


def _series_bound(
    read_client: Any, cfg: RunConfig, settings: Settings
) -> str | None:  # pragma: no cover - GCP I/O, exercised by the @gpu smoke
    """Resolve ``series_limit`` to a ``row_restriction``, or ``None`` if the read needs no filter.

    Costs one extra read session, and it is worth being honest about when: the boundary pass reads
    a *single column* but the table's full height, so it pays roughly ``1/n_columns`` of a scan to
    avoid ``1 - series_limit/table_series`` of the real one. Reading 100 series out of 100,000 is
    an enormous win; reading 90,000 out of 100,000 is a small loss. Runs that subset at all
    normally subset hard, and a run with no ``series_limit`` skips this entirely.
    """
    if cfg.data.series_limit is None:
        return None
    id_col = cfg.data.ts_id_col
    session = _create_read_session(
        read_client, cfg, settings, fields=[id_col], row_restriction=None
    )
    ids = _read_streams(read_client, session, [id_col])
    bound = build_series_bound(ids[id_col], cfg.data.series_limit, id_col)
    _log.info("ray read: series_limit pushed into the read as %s", bound or "no filter (limit ≥ n)")
    return bound


def _read_ray_data(
    cfg: RunConfig, settings: Settings
) -> pd.DataFrame:  # pragma: no cover - GCP I/O + live Ray, exercised by the @raylive smoke
    """Read the source panel with the Ray-native ``ray.data.read_bigquery`` reader (opt-in).

    ``ray.data.read_bigquery`` reads over the **same** BigQuery Storage Read API underneath (no
    query slots, matching Spark), returning a distributed `ray.data.Dataset`. We pass ``dataset=``
    (not ``query=``) so the read stays a pure table scan — matching `_read_driver_collect` and
    then materialize to a single driver-side pandas panel with ``.to_pandas()`` so the rest of the
    fan-out is identical to the default path. Column projection is applied in pandas after the read
    (the reader takes no ``selected_fields``), keeping the two readers' outputs the same shape.

    No ``series_limit`` pushdown here, for the same reason as the column projection: the
    ``dataset=`` table-scan form exposes no row-restriction knob. This path reads the whole table
    and lets `_limit_series` drop the excess on the driver, which is one more reason it stays
    opt-in — a subsetting run is cheaper on the default reader.

    When the run pins an input snapshot (`_snapshot_millis`) we instead pass ``query=`` with a
    ``FOR SYSTEM_TIME AS OF TIMESTAMP_MILLIS(...)`` clause — the reader's ``dataset=`` table-scan
    form has no snapshot-time knob, and a time-travel read must go through a query. That trades a
    pure scan for query slots on the pinned path only; the un-pinned default stays a slot-free scan.

    Kept off by default (``ray_read_mode == "driver_collect"``): this is the Ray-native ingest path,
    the same Storage Read API as the proven reader, but a live Ray run should vet it before it
    becomes the default.

    Keeping the panel distributed as ``ray.data`` blocks all the way into the fan-out (never
    calling ``.to_pandas()``) is the change that would remove the driver as a memory ceiling. It is
    **not** a small follow-up to this function: it replaces `chunk_cells` with a block-level
    ``map_groups`` and changes what a worker is handed, so it is gated on a live Ray run at a scale
    where the driver panel actually binds — which is precisely the scale this deployment sends to
    Spark. See README / NB04.
    """
    import ray

    ms = _snapshot_millis(cfg, settings)
    if ms is not None:
        cols = ", ".join(_needed_columns(cfg))
        table = _resolve_source_table(cfg, settings)
        query = f"SELECT {cols} FROM `{table}` FOR SYSTEM_TIME AS OF TIMESTAMP_MILLIS({ms})"
        ds = ray.data.read_bigquery(project_id=settings.project_id, query=query)
    else:
        dataset_ref = _storage_dataset_path(cfg, settings)
        ds = ray.data.read_bigquery(project_id=settings.project_id, dataset=dataset_ref)
    frame = ds.to_pandas()
    needed = _needed_columns(cfg)
    # The reader has no server-side column projection, so project in pandas to match the default
    # path's shape. Guard on presence so a lean source table (only the needed columns) still works.
    projected = [c for c in needed if c in frame.columns]
    return frame[projected] if projected else frame


def _storage_dataset_path(cfg: RunConfig, settings: Settings) -> str:
    """Resolve the source to the ``dataset.table`` form ``ray.data.read_bigquery`` wants (pure).

    ``read_bigquery(dataset=...)`` takes ``<dataset>.<table>`` (the project is passed separately as
    ``project_id``). `_storage_table_path` already resolves the full resource path; reuse it
    and drop the ``projects/P/datasets/`` / ``/tables/`` scaffolding back to ``D.T``.
    """
    path = _storage_table_path(cfg, settings)  # projects/P/datasets/D/tables/T
    _, _, _, dataset, _, table = path.split("/")
    return f"{dataset}.{table}"


def _sample_series(source: pd.DataFrame, cfg: RunConfig) -> list[pd.DataFrame]:
    """The first few per-series frames, in ts_id order, for live GPU calibration (auto only).

    `calibrate_gpu_fraction` fits NeuralProphet on these to measure peak GPU memory;
    ``gpu_calibration_samples`` caps how many so calibration is a few fits, not the whole panel.

    **Ordered by ts_id, not by arrival.** These few series decide ``gpu_fraction``, which decides
    how many cells share a device for the whole run — so taking whichever series the reader
    happened to return first made the fleet's density a function of Storage Read API stream
    ordering. Two runs of the same config on the same data could size differently, and neither
    would be wrong in a way anything could catch. Sorting is the same deterministic "first k
    ts_ids" subset `_limit_series`, `sample_series_to_driver` and `_resolve_fleetwide_hpo` already
    use, so every sample this codebase takes off a panel is the same sample.
    """
    id_col = cfg.data.ts_id_col
    n = cfg.compute.gpu_calibration_samples
    ids = sorted(dict.fromkeys(source[id_col].tolist()))[:n]
    return [source[source[id_col] == tid] for tid in ids]


def _assert_source_supports_folds(source: pd.DataFrame, cfg: RunConfig) -> None:
    """Backstop for ``backtest.short_series="error"`` — the Ray twin of the Spark one.

    Same contract as `spark_io.assert_source_supports_folds` and the same reason for existing: the
    submit-time check in `launch_plan.preflight_short_series` is not on every path into this
    engine. Cheaper here than there — the panel is already a driver-side pandas frame, so this is a
    ``groupby.size()`` rather than a distributed aggregation — but still skipped outright under
    every other policy, which have already decided what a short series gets.
    """
    from ..backtest import assert_panel_supports_folds

    bt = cfg.backtest
    if not (bt.enabled and bt.short_series == "error"):
        return
    assert_panel_supports_folds([int(n) for n in source.groupby(cfg.data.ts_id_col).size()], cfg)


def _resolve_fleetwide_hpo(
    source: pd.DataFrame, cfg: RunConfig, executed: list[str]
) -> dict[str, dict[str, object]] | None:
    """Driver-side fleetwide-HPO pre-pass over the collected pandas panel (the Ray twin).

    Returns ``None`` unless HPO is enabled at ``fleetwide`` granularity. When it is, takes the first
    ``hpo.sample_size`` series (deterministically, matching `_limit_series`) and tunes
    the executed model subset on them (`resolve_fleetwide`), scoping
    the study to the models that will actually run. The pandas analog of
    `resolve_fleetwide_hpo` — the Spark path samples from a Spark DataFrame, this
    one from the panel already on the driver.
    """
    if not (cfg.hpo.enabled and cfg.hpo.granularity == "fleetwide"):
        return None
    from ..hpo import resolve_fleetwide

    id_col = cfg.data.ts_id_col
    ids = sorted(dict.fromkeys(source[id_col].tolist()))[: cfg.hpo.sample_size]
    sample = [source[source[id_col] == tid].reset_index(drop=True) for tid in ids]
    tuning_cfg = cfg.model_copy(update={"models": executed})
    return resolve_fleetwide(sample, tuning_cfg)


def _chunk_count(n_cells: int, target_cells: int) -> int:
    """Chunks (Ray tasks) for a pool: ``ceil(cells / target)`` (≥ 1), or 0 for an empty pool.

    Mirrors Spark's bucket count (`default_bucket_count`):
    each chunk carries ~``target_cells`` cells so per-task memory stays bounded and the scheduler
    has many units to pack onto the fixed nodes. Clamped to `_MAX_CHUNKS` downstream
    by `chunk_cells`.
    """
    if n_cells <= 0:
        return 0
    return max(1, math.ceil(n_cells / target_cells))


def run(
    cfg: RunConfig,
    models: list[str] | None = None,
    *,
    manage_header: bool = True,
    settings: Settings | None = None,
) -> None:
    """Execute a Ray run end-to-end: header → route + fan chunks across the cluster → close header.

    Driver-side lifecycle, the structural twin of `spark_explode.run`:

    1. Resolve infra `Settings`, derive the ``run_id`` from
       the *full* ``cfg`` (so a mixed run shares one id across runtimes), and — in owner mode —
       ``ensure_tables`` + ``write_header`` (RUNNING).
    2. Read the source panel to the driver, split the executed models into GPU/CPU pools
       (`split_gpu_cpu_models`), calibrate the per-task GPU fraction
       (`calibrate_gpu_fraction` — live NeuralProphet memory profiling when ``auto``),
       measure what the models cost (`profiling.source.resolve_profile`) and size each pool from
       that measurement (`_pool_plans`), shard the panel per pool (`chunk_cells`),
       and dispatch one Ray task per chunk — GPU chunks as ``@ray.remote(num_gpus=fraction)``
       (packed onto T4s), CPU chunks as ``num_cpus=1`` plus, when it was measured, the host
       ``memory`` the family needs. Each pool has **its own** chunk runner
       (`make_chunk_runner`), carrying that pool's model list, because chunks arrive
       untagged and the runner's list is what a chunk runs. Every runner calls the exact
       `run_group` + `write_cells` and returns only the compact status frame.
    3. Concatenate the statuses, `aggregate_status`, and — in owner mode —
       ``update_header`` (COMPLETED/PARTIAL/FAILED, wall-clock, ``n_series``).

    ``models`` is the executed subset: ``None`` runs every model in ``cfg.models``;
    `main.run` passes only the Python-runtime models of a mixed config so the BigQuery-native
    ones run in BigQuery, not as Ray tasks. ``manage_header=False`` is contributor mode — the engine
    skips the header lifecycle because `main.run` owns the single shared header (parity with
    the Spark contributor mode). ``settings`` may pass an already-resolved `Settings` (the infra
    identity) to reuse a caller's; ``None`` resolves it from the environment — parity with the Spark
    engines' contract. Idempotent by construction: the config-derived ``run_id`` + append/
    dedupe-on-read writes mean a re-run of the same config lands byte-identical rows.

    Assumes Ray is reachable: connects with a plain ``ray.init()`` only if not already connected
    (the `ray_entry` Jobs entrypoint normally owns the session), and tears
    down only a session it opened — so a caller-managed session (e.g. the local-mode test) is left
    intact.
    """
    import time

    import ray

    from ..registry.ids import make_run_id
    from ..registry.lifecycle import run_header
    from ..settings import Settings

    settings = settings or Settings.resolve()
    run_id = make_run_id(cfg)
    executed = models if models is not None else cfg.models
    # One GPU decision for this job, from the resolved per-family compute the submitter
    # provisioned from. Never the flat compute.use_gpu — see ray_io.resolve_job_gpu.
    job_gpu, job_gpu_type = ray_io.resolve_job_gpu(cfg)
    gpu_models, cpu_models = ray_io.split_gpu_cpu_models(cfg, executed, use_gpu=job_gpu)
    _log.info(
        "ray run start: run_id=%s series_limit=%s gpu_models=%s cpu_models=%s manage_header=%s",
        run_id,
        cfg.data.series_limit,
        gpu_models,
        cpu_models,
        manage_header,
    )

    # 1. Header first (run_header): RUNNING on entry so a run is visible even if the cluster dies
    #    mid-flight, finalized on a clean exit; a crash records FAILED first. Contributor mode
    #    (main.run owns the shared header) is a no-op wrapper.
    with run_header(cfg, run_id, settings=settings, manage=manage_header) as hdr:
        owns_ray = not ray.is_initialized()
        if owns_ray:
            ray.init()
        started = time.perf_counter()
        try:
            source = _read_source_series(cfg, settings)

            # The `short_series="error"` backstop, before the job costs anything. No-op otherwise.
            _assert_source_supports_folds(source, cfg)

            # Fleetwide HPO resolves once on the driver over a small sample, before fan-out — the
            # Ray twin of spark_io.resolve_fleetwide_hpo, over the already-collected pandas panel.
            # None unless HPO is enabled at fleetwide granularity. Tuned params flow through the
            # chunk-runner closure to every task (not cfg → run_id stable).
            params_by_model = _resolve_fleetwide_hpo(source, cfg, executed)

            # One runner per pool, not one for the job. Chunks arrive untagged, so the runner's
            # model list *is* what a chunk runs — hand the CPU pool the full executed list and it
            # would run the GPU models on CPU hardware as well, twice-running every deep-learning
            # cell. The pool lists are already disjoint and together exactly ``executed``.
            cpu_runner = ray_io.make_chunk_runner(cfg, settings, cpu_models, params_by_model)
            gpu_runner = ray_io.make_chunk_runner(cfg, settings, gpu_models, params_by_model)

            # The per-task GPU fraction: fixed float passthrough, or live NeuralProphet profiling
            # when "auto". Sample series only when auto (profiling costs) and a GPU is present.
            sample = (
                _sample_series(source, cfg)
                if (gpu_models and job_gpu and cfg.compute.gpu_fraction == "auto")
                else None
            )
            gpu_fraction = ray_io.calibrate_gpu_fraction(
                cfg, sample_series=sample, gpu_type=job_gpu_type
            )

            # Measure what the models actually cost before deciding what to ask Ray for. Driver-side
            # and short (`compute.profile` gates it; "off" and a too-small fan-out both return
            # None), and it never enters cfg — the run_id must not move because a probe ran.
            profile = resolve_profile(source, cfg, executed, params_by_model=params_by_model)
            cpu_plan, gpu_plan = _pool_plans(
                source, cfg, run_id, cpu_models, gpu_models, profile, gpu_fraction
            )
            _log.info("ray sizing: cpu=%s gpu=%s", cpu_plan.to_dict(), gpu_plan.to_dict())
            # Whichever way memory and the scheduler disagree, the disagreement is the thing worth
            # saying out loud: the pool that under-runs silently, or the footprint the pool is
            # about to ignore. `fleet.RuntimeResourcePlan.density_note` words both cases.
            for note in (cpu_plan.density_note, gpu_plan.density_note):
                if note:
                    _log.warning("ray sizing: %s", note)
            _stamp_executed_sizing(
                run_id,
                _executed_sizing_patch(cpu_plan, gpu_plan, cpu_models, gpu_models),
                settings,
            )

            # Chunk counts come from the true cell counts (series in the panel × pool models),
            # floored so the pool can actually reach its autoscaling ceiling (`tasks_for_ceiling`).
            target = cfg.compute.bucket_target_cells
            gpu_chunks = ray_io.chunk_cells(source, cfg, gpu_models, _pool_chunks(gpu_plan, target))
            cpu_chunks = ray_io.chunk_cells(source, cfg, cpu_models, _pool_chunks(cpu_plan, target))

            # One Ray task per chunk. The remote closes over the picklable runner (cloudpickle
            # handles the cfg/settings closure — the single local/cloud seam, no second env path).
            # GPU tasks request a fraction of a T4 so several pack onto one device; when no GPU is
            # provisioned there are no GPU chunks at all — those cells are in ``cpu_chunks``, and
            # NeuralProphet falls back to CPU inside the task.
            @ray.remote
            def _cpu_task(chunk: pd.DataFrame) -> pd.DataFrame:
                return cpu_runner(chunk)

            @ray.remote
            def _gpu_task(chunk: pd.DataFrame) -> pd.DataFrame:
                return gpu_runner(chunk)

            # Retry only the failures where no Python `except` ever runs: the worker process died
            # or the node went away, which on a preemptible autoscaling pool is a scheduling event
            # rather than a bug. An application exception is deliberately NOT retried — the task
            # writes its cells before returning, so replaying it would duplicate durable work to
            # reach the same exception.
            from ray.exceptions import NodeDiedError, WorkerCrashedError

            retry = {
                "max_retries": _CHUNK_MAX_RETRIES,
                "retry_exceptions": [WorkerCrashedError, NodeDiedError],
            }
            cpu_opts = {**cpu_plan.task_options, **retry}
            gpu_opts = {**gpu_plan.task_options, **retry}
            pending: dict[Any, tuple[pd.DataFrame, list[str]]] = {}
            for chunk in cpu_chunks:
                pending[_cpu_task.options(**cpu_opts).remote(chunk)] = (chunk, cpu_models)
            for chunk in gpu_chunks:
                pending[_gpu_task.options(**gpu_opts).remote(chunk)] = (chunk, gpu_models)

            status_pdf, failures = _collect_chunks(ray, pending, cfg)
            if failures:
                _log.warning("ray run: %d chunk(s) failed; run will close PARTIAL", len(failures))
                _stamp_executed_sizing(run_id, {"chunk_failures": failures}, settings)
        finally:
            if owns_ray:
                ray.shutdown()

        # Close the header from the collected statuses (owner mode; contributor → main.run).
        outcome = ray_io.aggregate_status(status_pdf)
        runtime_seconds = time.perf_counter() - started
        hdr.finalize(status=outcome.status, n_series=outcome.n_series)
    _log.info(
        "ray run done: run_id=%s status=%s cells=%d ok=%d error=%d gpu_fraction=%s runtime=%.1fs",
        run_id,
        outcome.status,
        outcome.n_cells,
        outcome.n_ok,
        outcome.n_error,
        gpu_fraction,
        runtime_seconds,
    )


def _pool_cells(source: pd.DataFrame, cfg: RunConfig, pool_models: list[str]) -> int:
    """Cells one pool must run: distinct series in the panel × models in the pool (pure)."""
    if not pool_models or source.empty:
        return 0
    n_series = int(source[cfg.data.ts_id_col].nunique())
    return n_series * len(pool_models)


def _pool_plans(
    source: pd.DataFrame,
    cfg: RunConfig,
    run_id: str,
    cpu_models: list[str],
    gpu_models: list[str],
    profile: ComputeProfile | None,
    gpu_fraction: float,
) -> tuple[RuntimeResourcePlan, RuntimeResourcePlan]:
    """Size both worker pools for the *actual* panel — the heterogeneous-routing decision (pure).

    Two plans, one per pool, each carrying what a task should request and how wide the pool can
    grow. When no device is provisioned there is no GPU pool to plan against: `split_gpu_cpu_models`
    has already put the deep-learning models in ``cpu_models`` (NeuralProphet falls back to CPU
    inside the cell — the run still finishes, just slower), so ``gpu_models`` is empty and the GPU
    plan is a zero-cell placeholder.

    The GPU decision comes from `ray_io.resolve_job_gpu` — the resolved per-family compute, which
    is what the submitter provisioned from — and **not** from the flat ``compute.use_gpu``. Reading
    the flat field here is what let a run buy accelerators and then size a pool that asked for
    none. Re-resolving rather than taking it as an argument is deliberate: the function is pure and
    cheap, and an argument is one more thing that can be threaded through inconsistently.

    The cell counts come from the panel that was actually read, not from ``series_limit``, so the
    sizing reflects the run rather than its upper bound. The autoscaling ceilings, however, come
    from `plan_cluster` — the cluster was created from those bounds at submit time and the
    engine cannot widen them now; re-deriving them here would let the chunk count chase a ceiling
    the pool can never reach. Pure (no Ray, no GPU) so the routing stays unit-testable.
    """
    job_gpu, job_gpu_type = ray_io.resolve_job_gpu(cfg)
    cluster = ray_io.plan_cluster(
        cfg,
        cpu_models + gpu_models,
        run_id=run_id,
        use_gpu=job_gpu,
        gpu_type=job_gpu_type,
        profile=profile,
    )
    cpu_plan = ray_io.plan_pool(
        cfg,
        cpu_models,
        _pool_cells(source, cfg, cpu_models),
        gpu=False,
        profile=profile,
        max_units=cluster.cpu_max_nodes,
    )
    gpu_plan = ray_io.plan_pool(
        cfg,
        gpu_models,
        _pool_cells(source, cfg, gpu_models),
        gpu=job_gpu,
        gpu_type=job_gpu_type,
        profile=profile,
        gpu_fraction=gpu_fraction,
        max_units=cluster.gpu_max_nodes,
    )
    return cpu_plan, gpu_plan


def _collect_chunks(
    ray_mod: Any, pending: dict[Any, tuple[pd.DataFrame, list[str]]], cfg: RunConfig
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Collect chunk statuses one at a time; a chunk that raises costs its cells, not the run.

    ``ray.get(futures)`` on the whole list is all-or-nothing: the first exception propagates and the
    driver never sees the frames the other tasks already returned. That is the wrong shape here,
    because a chunk task writes its cells to BigQuery *before* it returns. By the time one chunk
    raises, every other chunk's forecasts are already durable in the registry — aborting the driver
    over them means the run closes FAILED and the header disowns work that is sitting in the table.

    So: wait for one future at a time, and treat a raising future as a per-chunk outcome rather
    than a control-flow event. The chunk's cells are recorded as ``status="error"`` so
    `aggregate_status` sees them, which is what makes the run close **PARTIAL** — some cells landed,
    some did not — instead of FAILED. The exception text comes back as the second return value for
    the caller to file on the header.

    ``ray_mod`` is passed in rather than imported so this is testable with a fake: everything here
    is scheduling logic, and none of it needs a real cluster to be wrong.
    """
    import pandas as pd

    frames: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []
    remaining = dict(pending)
    while remaining:
        done, _not_done = ray_mod.wait(list(remaining), num_returns=1)
        for ref in done:
            chunk, models = remaining.pop(ref)
            try:
                frames.append(ray_mod.get(ref))
            except Exception as exc:  # noqa: BLE001 - one chunk's failure is one chunk's outcome
                _log.warning("ray chunk failed (cells recorded as error): %r", exc)
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


def _failed_chunk_status(
    chunk: pd.DataFrame, cfg: RunConfig, models: list[str], exc: BaseException
) -> pd.DataFrame:
    """One ``status="error"`` row per cell the failed chunk was carrying (pure).

    The cells are reconstructed the same way `run_group` would have expanded them — from the model
    tag when the chunk has one, otherwise the pool's model list once per series — so the roll-up
    counts exactly the cells that were lost, not an approximation of them. ``STATUS_COLUMNS`` is
    deliberately not widened to carry the exception: it doubles as the Spark UDF's ``StructType``,
    so a column added here changes a schema on the other engine. The text goes on the header.
    """
    import pandas as pd

    id_col = cfg.data.ts_id_col
    if _MODEL_COL in chunk.columns:
        pairs = chunk[[id_col, _MODEL_COL]].drop_duplicates().to_numpy()
        cells = [(str(ts_id), str(model)) for ts_id, model in pairs]
    else:
        cells = [(str(ts_id), m) for ts_id in chunk[id_col].drop_duplicates() for m in models]
    return pd.DataFrame(
        {
            "ts_id": pd.Series([c[0] for c in cells], dtype="object"),
            "model_type": pd.Series([c[1] for c in cells], dtype="object"),
            "status": pd.Series(["error"] * len(cells), dtype="object"),
            "fit_seconds": pd.Series([0.0] * len(cells), dtype="float64"),
        },
        columns=list(STATUS_COLUMNS),
    )


def _executed_sizing_patch(
    cpu_plan: RuntimeResourcePlan,
    gpu_plan: RuntimeResourcePlan,
    cpu_models: list[str],
    gpu_models: list[str],
) -> dict[str, Any]:
    """The header patch describing the pools this driver actually sized, by family (pure).

    Sizing happens twice for a Ray run, and only the first half was ever written down. The submitter
    plans a cluster from the config's bounds and files that under ``sizing.<family>``; then the
    driver re-plans on the head node against the panel it really read, and *that* is the shape the
    run executed on. The two disagree whenever the panel is smaller than ``series_limit``, whenever
    profiling changed a slot width, or whenever GPU calibration landed on a different fraction than
    the plan assumed — which is to say, on most real runs. Filing the executed plan under its own
    key (``sizing_executed.<family>``) rather than merging it into the decided one keeps the pair
    readable: the disagreement is the finding, so overwriting would erase exactly the evidence
    somebody would go looking for.

    One entry per pool that has work. A pool with no models is not a pool — `split_gpu_cpu_models`
    puts deep-learning cells on the CPU side when no device was provisioned, so an absent GPU entry
    means "there was no GPU pool", not "the record is missing". Both pools file under the family
    label of the models on them, joined with ``+`` the same way `plan_cluster` and
    `resources.audit.sizing_telemetry` label a shared fleet, so the decided and executed records sit
    on matching header segments. Nesting under ``cpu_pool``/``gpu_pool`` keeps the two apart even in
    the case where a single family lands on both.
    """
    from ..registry.header import executed_sizing_path

    patch: dict[str, Any] = {}
    pools = (("cpu_pool", cpu_plan, cpu_models), ("gpu_pool", gpu_plan, gpu_models))
    for pool, plan, models in pools:
        if not models:
            continue
        path = executed_sizing_path("+".join(ray_io.pool_families(models)))
        patch.setdefault(path, {})[pool] = plan.to_dict()
    return patch


def _stamp_executed_sizing(
    run_id: str, patch: dict[str, Any], settings: Settings
) -> None:  # pragma: no cover - GCP I/O, exercised by the @gcp smokes
    """File the executed pool plans on the run header (best-effort).

    Best-effort in the same sense as every other telemetry write, and for the same reason: this runs
    inside a live run on the cluster head. A header that will not take an overlay costs a record,
    never a run.
    """
    from ..registry.header import merge_header_telemetry

    try:
        merge_header_telemetry(run_id, patch, settings=settings)
    except Exception as exc:  # noqa: BLE001 - telemetry is an overlay, never fatal
        _log.warning("executed sizing capture failed (non-fatal): %r", exc)


def _pool_chunks(plan: RuntimeResourcePlan, target_cells: int) -> int:
    """Chunks for one pool: the target-density count, floored so the autoscaler can reach its max.

    `_chunk_count` sizes chunks for bounded per-task memory. That is necessary but not
    sufficient under autoscaling: Ray grows a pool only while tasks are *pending*, so a run that
    submits no more tasks than the current fleet can hold leaves the pool at its minimum forever —
    the "we turned on autoscaling and nothing scaled" failure, which is arithmetic rather than a
    platform problem. `tasks_for_ceiling` is the demand the ceiling needs to see, so take
    the larger of the two. Overshooting is free: `chunk_cells` clamps the count and drops
    the empty chunks, so a pool never gets more tasks than it has cells.
    """
    return max(_chunk_count(plan.n_cells, target_cells), tasks_for_ceiling(plan))
