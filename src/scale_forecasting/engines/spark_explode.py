"""The Spark engine — cross-join series × model, grouped Pandas UDF.

The per-cell fan-out. Each ``(ts_id, model_type)`` cell is an independent unit of work: the source
series are cross-joined with the (small) model list, hashed into per-cell buckets, and run one
Spark task per bucket via ``groupBy(bucket).applyInPandas``. A slow ``(series, deep-model)`` cell
occupies its own bucket while that series' fast cells run concurrently in other buckets, so the
autoscaler spreads work and no single cell blocks the batch. This is what carries the
10 → 100 → 1k → 100k scale-up.

Runs on the Dataproc Serverless driver via ``spark_entry`` (the ``gs://`` launcher). All the
reusable mechanics — connector read + deterministic ``series_limit`` subset, cross-join, bucketing,
the executor-side write of each bucket's `CellResult`s through the
writer, and the status roll-up — live in `spark_io`; this module is just the
driver shell that wires them into a run with a proper registry header lifecycle.

Public surface: ``run(cfg) -> None``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..errors import get_logger
from . import spark_io

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

    from ..config import RunConfig
    from ..settings import Settings

_log = get_logger(__name__)


def _conf_int(spark: SparkSession, key: str) -> int | None:
    """One Spark conf entry as an int, or ``None`` when it is unset or unparseable.

    ``None`` is the honest answer for "the platform default applies" — we did not choose that
    number and should not size fan-out against a guess at it.
    """
    try:
        raw = spark.conf.get(key, None)
    except Exception:  # a Connect session may reject an unknown key outright
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _widen_fanout(cfg: RunConfig, spark: SparkSession, n_buckets: int) -> int:
    """Reconcile the bucket count with the fleet it is about to run on; return the final count.

    Two halves of one identity, and both are needed. `spark_io.reachable_bucket_count` raises the
    count until it can create the pending demand the autoscaler grows on, reading the ceiling off
    the live conf so it holds however that number got onto the batch — `submit.sizing_properties`,
    ``--max-executors``, or the operator's own property. `spark_io.fanout_properties` then pins the
    shuffle width to the result, because the group count and the task count are otherwise
    unrelated numbers and it is the task count the scheduler acts on.

    ``profile.mode == "off"`` returns the count untouched. That is the documented escape hatch
    back to the pre-profiler run, and it has to be checked *here* rather than at the call site
    because Serverless materializes its own dynamic-allocation defaults into the driver conf:
    read without the gate, an unsized batch reports a 1000-executor ceiling we never chose, and
    a run nobody asked to reshape gets reshaped around it.
    """
    if cfg.compute.profile.mode == "off":
        return n_buckets
    reachable = spark_io.reachable_bucket_count(
        n_buckets,
        max_executors=_conf_int(spark, "spark.dynamicAllocation.maxExecutors"),
        executor_cores=_conf_int(spark, "spark.executor.cores"),
        task_cpus=_conf_int(spark, "spark.task.cpus") or 1,
    )
    if reachable != n_buckets:
        _log.info(
            "buckets raised %d -> %d so the executor ceiling is reachable",
            n_buckets,
            reachable,
        )
    for key, value in spark_io.fanout_properties(reachable).items():
        spark.conf.set(key, value)
    return reachable


def run(
    cfg: RunConfig,
    models: list[str] | None = None,
    *,
    manage_header: bool = True,
    settings: Settings | None = None,
    spark: SparkSession | None = None,
) -> None:
    """Execute an explode run end-to-end: header → fan cells across Spark → close header.

    Driver-side lifecycle:

    1. Resolve infra `Settings` from the environment,
       ``ensure_tables``, and ``write_header`` (status RUNNING) with a ``run_id`` derived from the
       config — computed once here so every executor's ``write_cells`` shares it.
    2. Read + subset the source series, cross-join the model list, hash into per-cell buckets, and
       ``groupBy(bucket).applyInPandas`` the group runner (`spark_io.make_group_runner`),
       which runs each cell and appends its results executor-side. Only the compact status frame
       returns to the driver.
    3. Aggregate the statuses and ``update_header`` (COMPLETED/PARTIAL/FAILED, wall-clock
       ``runtime_seconds``, ``n_series``).

    ``models`` is the executed subset: ``None`` (the default, standalone) runs every model
    in ``cfg.models``; `main.run` passes only the Python-runtime models of a mixed config so
    the BigQuery-native ones don't become Spark cells. run_id is always derived from the *full*
    ``cfg`` so both runtimes share it (`make_run_id`).

    ``manage_header=False`` puts the engine in **contributor mode**: `main.run` owns
    the single shared header, so the engine skips ``ensure_tables`` / ``write_header`` /
    ``update_header`` and only fans cells + writes results. The default ``True`` preserves the
    self-contained standalone lifecycle every existing caller (CLI, ``@spark`` smoke) relies on.

    ``spark`` is an **optional injected session** (a `SparkSession`, incl. a Spark Connect
    ``DataprocSparkSession``). When ``None`` — the Dataproc batch path (``spark_entry`` passes
    none) — the engine self-creates a session via ``getOrCreate()`` and ``stop()``s it in
    ``finally``, exactly as before. When a session is injected — the notebook/Connect path — the
    engine uses it and does **not** stop it (the caller owns its lifecycle). The fan-out code is
    identical either way, so Connect and batch share one engine. ``settings`` similarly lets a
    caller pass an already-resolved `Settings`; ``None`` resolves it from the environment.

    Idempotent by construction: the config-derived ``run_id`` + append-only/dedupe-on-read writes
    mean a re-run of the same config lands byte-identical rows.
    """
    import time

    from pyspark.sql import SparkSession

    from ..registry import bq
    from ..registry.ids import make_run_id
    from ..settings import Settings

    settings = settings or Settings.resolve()
    run_id = make_run_id(cfg)
    executed = models if models is not None else cfg.models
    n_buckets = spark_io.default_bucket_count(cfg, executed)
    _log.info(
        "explode run start: run_id=%s series_limit=%s models=%d buckets=%d manage_header=%s",
        run_id,
        cfg.data.series_limit,
        len(executed),
        n_buckets,
        manage_header,
    )

    # 1. Header first (run_header): write RUNNING on entry so a run is visible even if the Spark job
    #    dies mid-flight, and finalize the collected status on a clean exit. Contributor mode
    #    (main.run owns the shared header) is a no-op wrapper. A crash inside records FAILED first.
    with bq.run_header(cfg, run_id, settings=settings, manage=manage_header) as hdr:
        # An injected session (notebook / Spark Connect) is caller-owned — use it, don't stop it.
        # Only a self-created session (the Dataproc batch path) is stopped here.
        owns_session = spark is None
        if spark is None:
            spark = SparkSession.builder.appName(
                f"scale-forecasting-explode-{run_id}"
            ).getOrCreate()
        # Fan-out width is only decidable once there is a session to ask: the fleet's ceiling is
        # set on the batch and read back from the live conf. See `_widen_fanout`.
        n_buckets = _widen_fanout(cfg, spark, n_buckets)

        started = time.perf_counter()
        try:
            # 2. Fan cells across the cluster. The frozen Settings is captured directly in the group
            #    runner's closure (no sparkContext.broadcast — Connect has no such API);
            #    applyInPandas cloudpickles it to every executor so write_cells resolves the infra.
            source = spark_io.read_source_series(spark, cfg, settings)

            # Fleetwide HPO resolves once on the driver over a small sample, before fan-out. The
            # tuned params flow to executors through the group-runner closure (not cfg → run_id
            # stable).
            params_by_model = spark_io.resolve_fleetwide_hpo(source, cfg, executed)

            cells = spark_io.cross_join_models(source, cfg, spark, executed)
            cells = spark_io.add_bucket(cells, cfg, n_buckets)

            runner = spark_io.make_group_runner(cfg, settings, executed, params_by_model)
            status_sdf = cells.groupBy(spark_io._BUCKET_COL).applyInPandas(
                runner, schema=spark_io.status_schema()
            )
            status_pdf = status_sdf.toPandas()  # compact: 4 cols × n_cells, no forecast payload
        finally:
            if owns_session:
                spark.stop()

        # Close the header from the collected statuses (owner mode; contributor → main.run).
        outcome = spark_io.aggregate_status(status_pdf)
        runtime_seconds = time.perf_counter() - started
        hdr.finalize(status=outcome.status, n_series=outcome.n_series)
    _log.info(
        "explode run done: run_id=%s status=%s cells=%d ok=%d error=%d runtime=%.1fs",
        run_id,
        outcome.status,
        outcome.n_cells,
        outcome.n_ok,
        outcome.n_error,
        runtime_seconds,
    )
