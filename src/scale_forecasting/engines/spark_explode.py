"""Spark method A — cross-join series × model, grouped Pandas UDF (CONTRACTS §6, DESIGN §6).

The hero fan-out. Each ``(ts_id, model_type)`` cell is an independent unit of work: the source
series are cross-joined with the (small) model list, hashed into per-cell buckets, and run one
Spark task per bucket via ``groupBy(bucket).applyInPandas``. A slow ``(series, deep-model)`` cell
occupies its own bucket while that series' fast cells run concurrently in other buckets, so the
autoscaler spreads work and no single cell blocks the batch — the property ``spark_naive``
deliberately lacks (DESIGN §2.1). This is the method that carries the 10 → 100 → 1k → 100k scale-up.

Runs on the Dataproc Serverless driver via ``spark_entry`` (the ``gs://`` launcher). All the
reusable mechanics — connector read + deterministic ``series_limit`` subset, cross-join, bucketing,
the executor-side write of each bucket's :class:`~scale_forecasting.worker.CellResult`s through the
B1-validated writer, and the status roll-up — live in :mod:`.spark_io`; this module is just the
driver shell that wires them into a run with a proper registry header lifecycle.

Public surface: ``run(cfg) -> None``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..errors import get_logger
from . import spark_io

if TYPE_CHECKING:
    from ..config import RunConfig

_log = get_logger(__name__)


def run(cfg: RunConfig) -> None:
    """Execute an explode run end-to-end: header → fan cells across Spark → close header.

    Driver-side lifecycle (CONTRACTS §3.4, §8.2):

    1. Resolve infra :class:`~scale_forecasting.settings.Settings` from the environment (G1),
       ``ensure_tables``, and ``write_header`` (status RUNNING) with a ``run_id`` derived from the
       config — computed once here so every executor's ``write_cells`` shares it.
    2. Read + subset the source series, cross-join the model list, hash into per-cell buckets, and
       ``groupBy(bucket).applyInPandas`` the group runner (:func:`spark_io.make_group_runner`),
       which runs each cell and appends its results executor-side. Only the compact status frame
       returns to the driver.
    3. Aggregate the statuses and ``update_header`` (COMPLETED/PARTIAL/FAILED, wall-clock
       ``runtime_seconds``, ``n_series``).

    Idempotent by construction: the config-derived ``run_id`` + append-only/dedupe-on-read writes
    mean a re-run of the same config lands byte-identical rows (§3.4).
    """
    import time

    from pyspark.sql import SparkSession

    from ..registry import bq
    from ..registry.ids import make_run_id
    from ..settings import Settings

    settings = Settings.resolve()
    run_id = make_run_id(cfg)
    n_buckets = spark_io.default_bucket_count(cfg)
    _log.info(
        "explode run start: run_id=%s series_limit=%s models=%d buckets=%d",
        run_id,
        cfg.data.series_limit,
        len(cfg.models),
        n_buckets,
    )

    # 1. Header first, so a run is visible in the registry even if the Spark job dies mid-flight.
    bq.ensure_tables(cfg, settings=settings)
    bq.write_header(cfg, run_id, settings=settings)

    spark = SparkSession.builder.appName(f"scale-forecasting-explode-{run_id}").getOrCreate()
    started = time.perf_counter()
    try:
        # 2. Fan cells across the cluster. Settings is broadcast (picklable frozen dataclass) so
        #    every executor's write_cells resolves the same infra without a second env path (G1).
        settings_bc = spark.sparkContext.broadcast(settings)
        cells = spark_io.read_source_series(spark, cfg, settings)
        cells = spark_io.cross_join_models(cells, cfg, spark)
        cells = spark_io.add_bucket(cells, cfg, n_buckets)

        runner = spark_io.make_group_runner(cfg, settings_bc)
        status_sdf = cells.groupBy(spark_io._BUCKET_COL).applyInPandas(
            runner, schema=spark_io.status_schema()
        )
        status_pdf = status_sdf.toPandas()  # compact: 4 cols × n_cells, no forecast payload
    finally:
        spark.stop()

    # 3. Close the header from the collected statuses.
    outcome = spark_io.aggregate_status(status_pdf)
    runtime_seconds = time.perf_counter() - started
    bq.update_header(
        run_id,
        settings=settings,
        status=outcome.status,
        runtime_seconds=runtime_seconds,
        n_series=outcome.n_series,
    )
    _log.info(
        "explode run done: run_id=%s status=%s cells=%d ok=%d error=%d runtime=%.1fs",
        run_id,
        outcome.status,
        outcome.n_cells,
        outcome.n_ok,
        outcome.n_error,
        runtime_seconds,
    )
