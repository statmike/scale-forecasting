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
from typing import TYPE_CHECKING

from ..errors import get_logger
from . import ray_io
from .spark_io import STATUS_COLUMNS, _needed_columns, _resolve_source_table, _snapshot_millis

if TYPE_CHECKING:
    import pandas as pd

    from ..config import RunConfig
    from ..settings import Settings

_log = get_logger(__name__)


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

    The pandas twin of `_limit_series`: distinct ts_ids →
    ordered → first N → filter, so Ray and Spark subset the *same* series at every scale — the
    property that makes the "10 vs 100 vs 100k" runtime comparison apples-to-apples.
    Applied client-side (not as a Storage Read ``row_restriction``, which can't express an ordered
    first-N over distinct ids), exactly as the Spark connector applies its limit after the read.
    """
    limit = cfg.data.series_limit
    if limit is None:
        return source
    id_col = cfg.data.ts_id_col
    keep = sorted(source[id_col].unique())[:limit]
    return source[source[id_col].isin(keep)].reset_index(drop=True)


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
    to only what a cell needs (`_needed_columns` → ``selected_fields``). Returns the raw
    panel; the caller (`_read_source_series`) applies the ``series_limit`` subset.

    Pinned to the run's input snapshot (`_snapshot_millis`) via the read session's
    ``table_modifiers.snapshot_time`` — the Storage Read API's native time-travel field, so the
    read consumes no query slots yet still sees the identical source state every other job in the
    run does. Unset snapshot → an un-pinned live read (the pre-snapshot behavior).
    """
    # Runtime import: pandas is TYPE_CHECKING-only at module scope (offline import parity), so every
    # function that touches pandas at runtime must import it locally.
    import pandas as pd
    from google.cloud.bigquery_storage_v1 import BigQueryReadClient, types

    read_client = BigQueryReadClient()
    requested = types.ReadSession(
        table=_storage_table_path(cfg, settings),
        data_format=types.DataFormat.ARROW,
        read_options=types.ReadSession.TableReadOptions(selected_fields=_needed_columns(cfg)),
    )
    ms = _snapshot_millis(cfg, settings)
    if ms is not None:
        # proto-plus surfaces the Timestamp field as a datetime, so set the whole modifiers
        # sub-message from a UTC datetime rather than mutating a Timestamp in place.
        from datetime import UTC, datetime

        requested.table_modifiers = types.ReadSession.TableModifiers(
            snapshot_time=datetime.fromtimestamp(ms / 1000, tz=UTC)
        )
    session = read_client.create_read_session(
        parent=f"projects/{settings.project_id}",
        read_session=requested,
        # 0 (default) lets the server pick the stream count from the table size;
        # compute.read_max_streams caps it to bound read parallelism (shared with the Spark reader).
        max_stream_count=cfg.compute.read_max_streams,
    )

    frames = [
        read_client.read_rows(stream.name).to_dataframe(session) for stream in session.streams
    ]
    if not frames:  # empty table → an empty, correctly-typed frame from the session schema
        return pd.DataFrame(columns=_needed_columns(cfg))
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


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

    When the run pins an input snapshot (`_snapshot_millis`) we instead pass ``query=`` with a
    ``FOR SYSTEM_TIME AS OF TIMESTAMP_MILLIS(...)`` clause — the reader's ``dataset=`` table-scan
    form has no snapshot-time knob, and a time-travel read must go through a query. That trades a
    pure scan for query slots on the pinned path only; the un-pinned default stays a slot-free scan.

    Kept off by default (``ray_read_mode == "driver_collect"``): this is the Ray-native ingest path,
    the same Storage Read API as the proven reader, but a live Ray run should vet it before it
    becomes the default. Distributing the panel as ``ray.data`` blocks straight into the fan-out
    (skipping the driver round-trip) is the documented next step from here — see README / NB04.
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
    """The first few per-series frames, for live GPU-memory calibration (auto fraction only).

    `calibrate_gpu_fraction` fits NeuralProphet on these to measure peak GPU memory;
    ``gpu_calibration_samples`` caps how many so calibration is a few fits, not the whole panel.
    """
    id_col = cfg.data.ts_id_col
    n = cfg.compute.gpu_calibration_samples
    ids = list(dict.fromkeys(source[id_col].tolist()))[:n]
    return [source[source[id_col] == tid] for tid in ids]


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
       chunk each pool's cells (`chunk_cells`), and dispatch one Ray task per chunk —
       GPU chunks as ``@ray.remote(num_gpus=fraction)`` (packed onto T4s), CPU chunks as
       ``num_cpus=1``. Every task runs the shared chunk runner (`make_chunk_runner`),
       which calls the exact `run_group` + `write_cells`
       and returns only the compact status frame.
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

    import pandas as pd
    import ray

    from ..registry import bq
    from ..registry.ids import make_run_id
    from ..settings import Settings

    settings = settings or Settings.resolve()
    run_id = make_run_id(cfg)
    executed = models if models is not None else cfg.models
    gpu_models, cpu_models = ray_io.split_gpu_cpu_models(cfg, executed)
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
    with bq.run_header(cfg, run_id, settings=settings, manage=manage_header) as hdr:
        owns_ray = not ray.is_initialized()
        if owns_ray:
            ray.init()
        started = time.perf_counter()
        try:
            source = _read_source_series(cfg, settings)

            # Fleetwide HPO resolves once on the driver over a small sample, before fan-out — the
            # Ray twin of spark_io.resolve_fleetwide_hpo, over the already-collected pandas panel.
            # None unless HPO is enabled at fleetwide granularity. Tuned params flow through the
            # chunk-runner closure to every task (not cfg → run_id stable).
            params_by_model = _resolve_fleetwide_hpo(source, cfg, executed)
            runner = ray_io.make_chunk_runner(cfg, settings, executed, params_by_model)

            # The per-task GPU fraction: fixed float passthrough, or live NeuralProphet profiling
            # when "auto". Sample series only when auto (profiling costs) and a GPU is present.
            sample = (
                _sample_series(source, cfg)
                if (gpu_models and cfg.compute.use_gpu and cfg.compute.gpu_fraction == "auto")
                else None
            )
            gpu_fraction = ray_io.calibrate_gpu_fraction(
                cfg, sample_series=sample, gpu_type=cfg.compute.gpu_type
            )

            # Chunk counts come from the true cell counts (series in the panel × pool models).
            target = cfg.compute.bucket_target_cells
            gpu_chunks = ray_io.chunk_cells(
                source, cfg, gpu_models, _chunk_count(_pool_cells(source, cfg, gpu_models), target)
            )
            cpu_chunks = ray_io.chunk_cells(
                source, cfg, cpu_models, _chunk_count(_pool_cells(source, cfg, cpu_models), target)
            )

            # One Ray task per chunk. The remote closes over the picklable runner (cloudpickle
            # handles the cfg/settings closure — the single local/cloud seam, no second env path).
            # GPU tasks request a fraction of a T4 so several pack onto one device; when no GPU is
            # provisioned, NeuralProphet cells fall back to CPU inside the task, so route them as
            # CPU work too (see _task_options).
            @ray.remote
            def _task(chunk: pd.DataFrame) -> pd.DataFrame:
                return runner(chunk)

            cpu_opts = _task_options(cfg, gpu_fraction, gpu=False)
            gpu_opts = _task_options(cfg, gpu_fraction, gpu=True)
            futures = [_task.options(**cpu_opts).remote(c) for c in cpu_chunks]
            futures += [_task.options(**gpu_opts).remote(c) for c in gpu_chunks]

            status_frames = ray.get(futures) if futures else []
            status_pdf = (
                pd.concat(status_frames, ignore_index=True)
                if status_frames
                else pd.DataFrame(columns=list(STATUS_COLUMNS))
            )
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


def _task_options(cfg: RunConfig, gpu_fraction: float, *, gpu: bool) -> dict[str, float]:
    """The ``@ray.remote.options`` for one pool's tasks — the heterogeneous-routing decision (pure).

    CPU-pool tasks (``gpu=False``) always request ``num_cpus=1``. GPU-pool tasks (``gpu=True``)
    request ``num_gpus=gpu_fraction`` **only when a GPU is actually provisioned** (``use_gpu``), so
    several NeuralProphet tasks pack onto one T4 (Ray sums fractions against the device's 1.0). When
    ``use_gpu`` is off there is no device to schedule against, so a GPU-model chunk runs as a plain
    ``num_cpus=1`` task and NeuralProphet falls back to CPU inside the cell — the run still
    finishes, just slower. Kept pure (no Ray) so the routing is unit-testable without a GPU.
    """
    if gpu and cfg.compute.use_gpu:
        return {"num_gpus": gpu_fraction}
    return {"num_cpus": 1}
