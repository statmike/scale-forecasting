"""Spark method C — group by ts_id, sequential model loop: the anti-pattern (DESIGN §2.1).

The deliberate straggler baseline that makes explode's per-cell fan-out *visible* in the registry.
Where :mod:`.spark_explode` cross-joins series × model so every ``(ts_id, model_type)`` cell is an
independent unit, naive keeps a whole series in one Spark task and runs *all* its models back to
back. One slow ``(series, deep-model)`` fit therefore blocks every other model for that series, and
— throttled with ``--max-executors`` — a handful of long series set the batch wall-clock while most
executors sit idle. That poor autoscaling is the point: the run's ``runtime_seconds`` in
``run_registry`` lands materially higher than explode's at the same scale, so the scaling story is a
single query, not a slide. Small scales only (10 / 100); it is not meant to reach 100k.

Mechanically this is :mod:`.spark_explode` minus the cross-join: :func:`spark_io.bucket_key_cols`
returns ``[ts_id]`` for the naive method (so a series' whole history shares one bucket) and
:func:`spark_io.run_group` sees no model column and loops ``cfg.models`` per series. Everything else
— the connector read + deterministic subset, executor-side batched write of each bucket's
:class:`~scale_forecasting.worker.CellResult`s, and the driver's header lifecycle + status roll-up —
is the shared code, unchanged.

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
    """Execute a naive run end-to-end: header → fan *series* across Spark → close header.

    Same driver-side lifecycle as :func:`spark_explode.run` (CONTRACTS §3.4, §8.2): resolve
    :class:`~scale_forecasting.settings.Settings`, derive the ``run_id`` once, ``ensure_tables`` +
    ``write_header`` (RUNNING), fan the work across the cluster, then ``update_header`` with the
    aggregated status + wall-clock ``runtime_seconds`` + ``n_series``.

    The one structural difference is the unit of parallelism: no cross-join, so the bucket key is
    ``ts_id`` alone and each Spark task owns a whole series and runs every model in ``cfg.models``
    sequentially (:func:`spark_io.run_group`). The throttle that exposes the straggler
    (``spark.dynamicAllocation.maxExecutors``) is applied at submit time (``--max-executors``), not
    here — the engine code is identical to the un-throttled case, which keeps the G1 seam clean.

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
        "naive run start: run_id=%s series_limit=%s models=%d buckets=%d",
        run_id,
        cfg.data.series_limit,
        len(cfg.models),
        n_buckets,
    )

    # 1. Header first, so a run is visible in the registry even if the Spark job dies mid-flight.
    bq.ensure_tables(cfg, settings=settings)
    bq.write_header(cfg, run_id, settings=settings)

    spark = SparkSession.builder.appName(f"scale-forecasting-naive-{run_id}").getOrCreate()
    started = time.perf_counter()
    try:
        # 2. Fan whole series across the cluster — NO cross-join, so each task runs all models for
        #    its series sequentially. Settings is broadcast (picklable frozen dataclass) so every
        #    executor's write_cells resolves the same infra without a second env path (G1).
        settings_bc = spark.sparkContext.broadcast(settings)
        cells = spark_io.read_source_series(spark, cfg, settings)
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
        "naive run done: run_id=%s status=%s cells=%d ok=%d error=%d runtime=%.1fs",
        run_id,
        outcome.status,
        outcome.n_cells,
        outcome.n_ok,
        outcome.n_error,
        runtime_seconds,
    )
