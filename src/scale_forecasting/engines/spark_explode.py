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

from typing import TYPE_CHECKING, Any

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


def _estimated_series(cfg: RunConfig, source: Any) -> int | None:
    """Roughly how many distinct series the source holds, or ``None`` when it need not be asked.

    Only the unbounded run has to ask: with ``series_limit`` set the count is already known offline
    and `spark_io.default_bucket_count` uses it directly. Without one, the alternative to asking is
    sizing the fan-out against a parallelism cap, which is the arithmetic that OOMs at scale.

    ``approx_count_distinct`` rather than ``countDistinct`` because the number decides a frame
    size, not a result: HyperLogLog's few-percent error moves the bucket count by a few percent
    and nothing else, while an exact count is a full shuffle of the very column the run is about
    to shuffle again. ``None`` on any failure — a fan-out that cannot be sized well should still
    be sized, not refused.
    """
    if cfg.data.series_limit is not None:
        return None
    from pyspark.sql import functions as F

    try:
        row = source.select(F.approx_count_distinct(cfg.data.ts_id_col).alias("n")).first()
    except Exception:  # noqa: BLE001 - an estimate is an optimisation; the fallback is a cap
        _log.warning("could not estimate the series count; sizing buckets off max_parallelism")
        return None
    n = row["n"] if row is not None else None
    return int(n) if n else None


def _fanout_family(executed: list[str]) -> str:
    """The family label the fan-out record files under — ``+``-joined, in first-seen order (pure).

    One explode job may carry several families through the same buckets, and the shape it settled
    on belongs to all of them. Joining rather than picking the first matches what
    `resources.audit.sizing_telemetry` does for a shared cluster, so the decided record and the
    executed one land under the same header segment and can be read as a pair.
    """
    from ..models import get_model

    families: list[str] = []
    for name in executed:
        family = get_model(name).family
        if family not in families:
            families.append(family)
    return "+".join(families)


def _stamp_executed_fanout(
    run_id: str, executed: list[str], fanout: dict[str, Any], settings: Settings
) -> None:  # pragma: no cover - GCP I/O, exercised by the @gcp smokes
    """File the executed fan-out under ``sizing_executed.<family>`` (best-effort).

    Best-effort in the same sense as every other telemetry write: this runs inside a live run, and
    a header that will not take an overlay is a lost record, never a lost run.
    """
    from ..registry.header import executed_sizing_path, merge_header_telemetry

    try:
        path = executed_sizing_path(_fanout_family(executed))
        merge_header_telemetry(run_id, {path: {"fanout": fanout}}, settings=settings)
    except Exception as exc:  # noqa: BLE001 - telemetry is an overlay, never fatal
        _log.warning("executed fan-out capture failed (non-fatal): %r", exc)


def _widen_fanout(cfg: RunConfig, spark: SparkSession, n_buckets: int) -> dict[str, Any]:
    """Reconcile the bucket count with the fleet it is about to run on; return what it settled on.

    The record, not just the number, because this is the only place the *executed* Spark shape is
    knowable. Everything else about the batch is decided at submit and echoed back by the API
    (`batch_telemetry.extract_job_telemetry`); the fan-out is decided here, on a live session,
    against confs that may have arrived from three different places. ``buckets_policy`` vs
    ``buckets`` is the widening itself, ``shuffle_partitions`` is the number that decides how many
    tasks actually ran, and the three conf reads are the evidence the widening used — a run whose
    ceiling came from somewhere nobody expected says so here rather than in a driver log that
    outlives nothing.

    Two halves of one identity, and both are needed. `spark_io.reachable_bucket_count` raises the
    count until it can create the pending demand the autoscaler grows on, reading the ceiling off
    the live conf so it holds however that number got onto the batch — `submit.sizing_properties`,
    ``--max-executors``, or the operator's own property. `spark_io.fanout_properties` then pins the
    shuffle width to the result, because the group count and the task count are otherwise
    unrelated numbers and it is the task count the scheduler acts on.

    ``profile.mode == "off"`` gates **the widening only**, and the gate has to be checked *here*
    rather than at the call site because Serverless materializes its own dynamic-allocation
    defaults into the driver conf: read without the gate, an unsized batch reports a
    1000-executor ceiling we never chose, and a run nobody asked to reshape gets reshaped
    around it.

    The shuffle-width pin is applied either way, because it is not a profiling decision. Whatever
    bucket count this run ended up with — widened or the caller's own — Spark plans the shuffle at
    ``spark.sql.shuffle.partitions`` and AQE coalesces below that, so leaving the pin off does not
    restore some earlier behaviour; it hands 200 tasks to a run that asked for N. Gating it here
    made ``profile.mode="off"`` quietly mean "and also stop fanning out", which is a second
    behaviour nobody opted into by turning measurement off.
    """
    max_executors = _conf_int(spark, "spark.dynamicAllocation.maxExecutors")
    executor_cores = _conf_int(spark, "spark.executor.cores")
    task_cpus = _conf_int(spark, "spark.task.cpus")
    if cfg.compute.profile.mode == "off":
        reachable = n_buckets
    else:
        reachable = spark_io.reachable_bucket_count(
            n_buckets,
            max_executors=max_executors,
            executor_cores=executor_cores,
            task_cpus=task_cpus or 1,
        )
        if reachable != n_buckets:
            _log.info(
                "buckets raised %d -> %d so the executor ceiling is reachable",
                n_buckets,
                reachable,
            )
    properties = spark_io.fanout_properties(reachable)
    for key, value in properties.items():
        spark.conf.set(key, value)
    return {
        "buckets_policy": n_buckets,
        "buckets": reachable,
        "shuffle_partitions": int(properties["spark.sql.shuffle.partitions"]),
        "max_executors": max_executors,
        "executor_cores": executor_cores,
        "task_cpus": task_cpus,
        "widened": reachable != n_buckets,
    }


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

    from ..registry.ids import make_run_id
    from ..registry.lifecycle import run_header
    from ..settings import Settings

    settings = settings or Settings.resolve()
    run_id = make_run_id(cfg)
    executed = models if models is not None else cfg.models
    _log.info(
        "explode run start: run_id=%s series_limit=%s models=%d manage_header=%s",
        run_id,
        cfg.data.series_limit,
        len(executed),
        manage_header,
    )

    # 1. Header first (run_header): write RUNNING on entry so a run is visible even if the Spark job
    #    dies mid-flight, and finalize the collected status on a clean exit. Contributor mode
    #    (main.run owns the shared header) is a no-op wrapper. A crash inside records FAILED first.
    with run_header(cfg, run_id, settings=settings, manage=manage_header) as hdr:
        # An injected session (notebook / Spark Connect) is caller-owned — use it, don't stop it.
        # Only a self-created session (the Dataproc batch path) is stopped here.
        owns_session = spark is None
        if spark is None:
            spark = SparkSession.builder.appName(
                f"scale-forecasting-explode-{run_id}"
            ).getOrCreate()
        started = time.perf_counter()
        try:
            # 2. Fan cells across the cluster. The frozen Settings is captured directly in the group
            #    runner's closure (no sparkContext.broadcast — Connect has no such API);
            #    applyInPandas cloudpickles it to every executor so write_cells resolves the infra.
            source = spark_io.read_source_series(spark, cfg, settings)

            # Fan-out width needs both a session and the source: the fleet's ceiling is set on the
            # batch and read back from the live conf, and an unbounded run's series count can only
            # come off the data. The confs `_widen_fanout` sets govern the applyInPandas shuffle
            # below, so landing them here rather than before the read changes nothing about it.
            fanout = _widen_fanout(
                cfg,
                spark,
                spark_io.default_bucket_count(
                    cfg, executed, n_series=_estimated_series(cfg, source)
                ),
            )
            n_buckets = fanout["buckets"]
            _log.info("explode fan-out: run_id=%s %s", run_id, fanout)
            _stamp_executed_fanout(run_id, executed, fanout, settings)

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
