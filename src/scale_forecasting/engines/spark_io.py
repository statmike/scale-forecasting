"""Shared Spark-engine plumbing — the pieces the ``spark_explode`` engine is built from.

Split along the pure/I-O seam so the interesting logic is offline-testable:

* **Pure** (no Spark, no BigQuery): `run_group` — the body of the grouped-Pandas UDF, which
  runs `run_cell` for every cell in one group's pandas frame and
  returns ``(results, status_frame)``; `aggregate_status` — fold the per-cell status frame
  into a run-level `RunOutcome`; `default_bucket_count`, `reachable_bucket_count`,
  `fanout_properties`, `bucket_key_cols`.
* **I/O / Spark shell** (pyspark imported lazily, parity with the seed job):
  `read_source_series` (connector read + deterministic ``series_limit`` subset),
  `add_bucket`, `cross_join_models`, `status_schema`, `make_group_runner`.

**Fan-out mechanics.** The engine shuffles cells into *buckets* and runs one Spark task per bucket
(``groupBy(bucket).applyInPandas``, with the shuffle width pinned to the bucket count by
`fanout_properties` — Spark would otherwise plan its default 200 tasks whatever the bucket count
is, and the fan-out would be a fiction). The bucket key is the natural unit of independence
(`bucket_key_cols`): a cell — ``(ts_id, model_type)``. A slow ``(series, deep-model)`` cell lands in
its own bucket while that same series' fast cells sit in *other* buckets and run concurrently, so
the scheduler spreads cells and autoscales. Every history row of a given cell shares both keys, so a
cell's full series stays intact in one bucket.

**Writes are executor-side, once per bucket.** `make_group_runner` wraps `run_group`
and calls `write_cells` on each bucket's
results (all three cell tables + artifact upload). Append-only + dedupe-on-read
makes per-partition appends safe. The UDF returns only the compact status frame; the driver
owns the header. Infra identity reaches executors by capturing the frozen `Settings`
dataclass (picklable, tiny, read-only) directly in the group-runner closure, which ``applyInPandas``
cloudpickles to each executor — preserving the single local/cloud seam without a second env-based
delivery path, and without the ``sparkContext.broadcast`` call that a Spark Connect session does not
expose (so the same engine runs both as a Dataproc batch and driven over a Connect endpoint).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd
    from pyspark.sql import DataFrame, SparkSession

    from ..config import RunConfig
    from ..settings import Settings
    from ..worker import CellResult

# Internal helper columns added to the working DataFrame. Underscore-prefixed + ``_sf_`` namespaced
# so they can't collide with a real source column, and dropped before a frame reaches ``run_cell``.
_MODEL_COL = "_sf_model"
_BUCKET_COL = "_sf_bucket"

# Safety ceiling on bucket count: even a huge run shouldn't shatter into an unbounded number of
# tiny shuffle partitions (scheduler overhead, tiny writes). At the 100k×N hero scale with the
# default target this is not reached; it only guards a pathological config.
_MAX_BUCKETS = 100_000

# The compact per-cell status frame the UDF returns to the driver (no forecast payload).
STATUS_COLUMNS: tuple[str, ...] = ("ts_id", "model_type", "status", "fit_seconds")


# --- pure: bucketing policy ----------------------------------------------------


def bucket_key_cols(cfg: RunConfig) -> list[str]:
    """The columns whose hash defines a bucket — the engine's unit of independence.

    ``[ts_id, model_type]`` — a whole cell per bucket, so cells spread across tasks and a slow cell
    can't block fast ones. See the module docstring for why this is the crux of the scaling story.
    """
    return [cfg.data.ts_id_col, _MODEL_COL]


def default_bucket_count(cfg: RunConfig, models: list[str] | None = None) -> int:
    """Bucket count that keeps each ``applyInPandas`` frame bounded as scale grows.

    Buckets are *groups*, not executor concurrency (that's
    ``spark.dynamicAllocation.maxExecutors``): each bucket is materialized as one pandas frame in
    one ``run_group`` call, so its size — not the cluster width — is what OOMs an executor. (A
    group only becomes a *task* because `fanout_properties` pins the shuffle width to match;
    Spark's own default would run the whole fan-out in 200 tasks whatever this returns.) We size
    buckets to hold ~``compute.bucket_target_cells`` cells each: ``buckets = ceil(cells / target)``.
    That holds ~``target`` series-histories per frame at *every* scale (1k and 100k alike), instead
    of the old ``min(cells, max_parallelism)`` which silently fattened frames past the executor
    memory budget once ``cells`` outgrew the cap (the 100k OOM). Clamped to ``[1, _MAX_BUCKETS]``.

    With ``series_limit`` set the cell count is known offline (series × models); an unbounded run
    falls back to ``max_parallelism`` buckets (best guess without a known cell count). ``models`` is
    the executed subset; ``None`` means ``cfg.models`` — so a standalone run and a subset run size
    buckets to the work they actually fan out.

    This is the *policy* number. `reachable_bucket_count` then raises it if the cluster it is
    about to run on is wider than the policy would keep busy.
    """
    executed = models if models is not None else cfg.models
    target = cfg.compute.bucket_target_cells
    limit = cfg.data.series_limit
    if limit is None:
        return max(1, min(cfg.compute.max_parallelism, _MAX_BUCKETS))
    cells = limit * len(executed)
    return max(1, min(math.ceil(cells / target), _MAX_BUCKETS))


def reachable_bucket_count(
    buckets: int,
    *,
    max_executors: int | None,
    executor_cores: int | None,
    task_cpus: int = 1,
) -> int:
    """Raise a bucket count until the autoscaler's ceiling is actually reachable (pure).

    Dynamic allocation grows on *pending demand*: Spark adds an executor because tasks are
    queued and cannot be placed. A run that never submits more tasks than the current fleet
    can hold has nothing pending, so the fleet sits at its minimum for the whole run — the
    "we enabled autoscaling and nothing scaled" report, which reads as a platform problem and
    is arithmetic. One executor runs ``executor.cores / task.cpus`` tasks at once, so reaching
    ``maxExecutors`` takes at least that many tasks times the ceiling.

    Raising the count is the cheap side of the trade: extra buckets mean smaller pandas frames
    (further from the OOM `default_bucket_count` exists to avoid), while too few caps the run's
    width for its entire duration. Any argument left ``None`` — the property is unset, so the
    platform default applies and we are not the ones who chose it — leaves the count alone.
    """
    if max_executors is None or executor_cores is None:
        return buckets
    per_executor = max(1, executor_cores // max(1, task_cpus))
    return max(1, min(max(buckets, max_executors * per_executor), _MAX_BUCKETS))


def fanout_properties(buckets: int) -> dict[str, str]:
    """The Spark confs that make one bucket one *task* (pure).

    ``groupBy(bucket).applyInPandas`` is a shuffle, and a shuffle stage's task count is
    ``spark.sql.shuffle.partitions`` — **not** the number of distinct groups. Nothing about
    hashing cells into 4000 buckets makes Spark run 4000 tasks: left alone it plans Spark's
    default 200 and AQE then coalesces *below* that, so a run carefully fanned out for a wide
    fleet arrives at the scheduler as a couple of hundred tasks and the fleet it was sized for
    never fills. That is the same "we enabled autoscaling and nothing scaled" arithmetic
    `reachable_bucket_count` guards from the other side, and raising the bucket count alone
    would not have moved it.

    So the width is stated outright. The AQE floor goes with it because coalescing would undo
    the pin from underneath: each partition here holds a whole bucket's cells, and merging
    partitions serialises cells that were meant to run side by side. It costs nothing when the
    partitions are genuinely small — that is what `default_bucket_count` is for.
    """
    return {
        "spark.sql.shuffle.partitions": str(buckets),
        "spark.sql.adaptive.coalescePartitions.minPartitionNum": str(buckets),
    }


# --- pure: the grouped-UDF body ------------------------------------------------


def run_group(
    pdf: pd.DataFrame,
    cfg: RunConfig,
    models: list[str] | None = None,
    params_by_model: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[CellResult], pd.DataFrame]:
    """Run every cell in one group's pandas frame; pure — no Spark, no BigQuery.

    The frame is one bucket's rows. If it carries the internal model column (the series were
    cross-joined with the model list, as the Spark/Ray fan-out does), each ``(ts_id, model_type)``
    sub-frame is one cell. Otherwise — an untagged frame, e.g. direct SDK use — every model in the
    executed list is run for each ``ts_id`` in a loop. Helper columns are dropped so each sub-frame
    is a clean series frame for `run_cell` (which derives the run_id from ``cfg`` itself, so no
    id needs threading here). Returns the `CellResult` list (for the writer) and the compact
    `STATUS_COLUMNS` frame (for the driver's header roll-up). Never raises per cell —
    ``run_cell`` maps a failure to a ``status="error"`` result.

    ``models`` is the executed subset: under `main.run` a mixed config routes only its
    Python-runtime models here while the BigQuery-native models run elsewhere, so the untagged-frame
    loop must iterate the subset, not the full ``cfg.models`` (which would feed a native model into
    ``run_cell`` → ``NotImplementedError``). ``None`` means ``cfg.models`` (standalone). A tagged
    frame takes its models from the cross-join column, so the subset only affects the untagged loop.

    ``params_by_model`` is the fleetwide-HPO resolution: ``{model: tuned params}`` computed
    once on the driver and passed to every cell of that model, so the tuned params apply fleet-wide
    without entering ``cfg`` (which would shift the run_id). A model absent from the map (or a None
    map) resolves inside ``run_cell`` — per-series HPO if configured, else the ``{}`` default.
    """
    import pandas as pd

    from ..worker import run_cell

    id_col = cfg.data.ts_id_col
    helper_cols = [c for c in (_MODEL_COL, _BUCKET_COL) if c in pdf.columns]
    executed = models if models is not None else cfg.models
    by_model = params_by_model or {}

    results: list[CellResult] = []
    if _MODEL_COL in pdf.columns:
        # tagged frame: the cross-join tagged each row with its model; one cell per (ts_id, model).
        for (_ts_id, model_name), sub in pdf.groupby([id_col, _MODEL_COL], sort=False):
            series = sub.drop(columns=helper_cols)
            results.append(run_cell(series, str(model_name), cfg, by_model.get(str(model_name))))
    else:
        # untagged frame: one group per series, every executed model run for it in a loop.
        for _ts_id, sub in pdf.groupby(id_col, sort=False):
            series = sub.drop(columns=helper_cols)
            for model_name in executed:
                results.append(run_cell(series, model_name, cfg, by_model.get(model_name)))

    status = pd.DataFrame(
        {
            "ts_id": pd.Series([r.ts_id for r in results], dtype="object"),
            "model_type": pd.Series([r.model_type for r in results], dtype="object"),
            "status": pd.Series([r.status for r in results], dtype="object"),
            "fit_seconds": pd.Series([r.fit_seconds for r in results], dtype="float64"),
        },
        columns=list(STATUS_COLUMNS),
    )
    return results, status


# --- pure: run-level roll-up ---------------------------------------------------


@dataclass(frozen=True)
class RunOutcome:
    """The driver's roll-up of a run's per-cell statuses (feeds ``update_header``)."""

    n_series: int
    n_cells: int
    n_ok: int
    n_error: int
    status: str  # COMPLETED | PARTIAL | FAILED


def aggregate_status(status_pdf: pd.DataFrame) -> RunOutcome:
    """Fold the collected per-cell status frame into a `RunOutcome` (pure).

    ``COMPLETED`` = every cell ok; ``PARTIAL`` = a mix of ok and error; ``FAILED`` = no ok cells
    (all errored, or nothing ran at all). This is what the driver writes to
    ``run_registry.status`` after the Spark job returns.
    """
    n_cells = int(len(status_pdf))
    if n_cells == 0:
        return RunOutcome(n_series=0, n_cells=0, n_ok=0, n_error=0, status="FAILED")
    n_error = int((status_pdf["status"] == "error").sum())
    n_ok = n_cells - n_error
    n_series = int(status_pdf["ts_id"].nunique())
    if n_error == 0:
        status = "COMPLETED"
    elif n_ok == 0:
        status = "FAILED"
    else:
        status = "PARTIAL"
    return RunOutcome(n_series=n_series, n_cells=n_cells, n_ok=n_ok, n_error=n_error, status=status)


# --- Spark shell: source read + subset -----------------------------------------


def _resolve_source_table(cfg: RunConfig, settings: Settings) -> str:
    """Fully-qualify the source table: use ``source_table`` as-is if it already has a dataset
    qualifier, else resolve a bare name against the deployment's dataset."""
    src = cfg.data.source_table
    return src if "." in src else settings.table_ref(src)


def _snapshot_millis(cfg: RunConfig, settings: Settings) -> int | None:
    """The input-data snapshot this run pins its source read to, or ``None`` (unpinned).

    Shared by every reader (this Spark path and the Ray engine's): derive the ``run_id`` from the
    config (`registry.ids.make_run_id`, pure) and fetch the snapshot the run recorded on its header
    (`registry.header.snapshot_millis_for`), so all family jobs time-travel to the identical
    instant. Best-effort — a missing/NULL snapshot returns ``None`` and the read stays unpinned.
    """
    from ..registry.header import snapshot_millis_for
    from ..registry.ids import make_run_id

    return snapshot_millis_for(make_run_id(cfg), settings=settings)


def _needed_columns(cfg: RunConfig) -> list[str]:
    """Project the read to only what a cell needs — ts_id, date, target, and configured exog.

    Trims the ``ts_id × model`` cross-join's shuffle to the essential columns (explode duplicates
    each series once per model, so narrow rows matter). Order-preserving + de-duplicated.
    """
    wanted = [cfg.data.ts_id_col, cfg.data.date_col, cfg.data.target_col, *cfg.features.exog]
    seen: set[str] = set()
    out: list[str] = []
    for col in wanted:
        if col not in seen:
            seen.add(col)
            out.append(col)
    return out


def read_source_series(spark: SparkSession, cfg: RunConfig, settings: Settings) -> DataFrame:
    """Read the source series via the spark-bigquery connector, projected + optionally subset.

    Applies ``data.series_limit`` deterministically (distinct ts_ids → ordered → first N →
    semi-join) so every scale in the demo runs the *same* series — the property that
    makes "10 vs 100 vs 100k" a clean apples-to-apples runtime comparison.

    Pins the read to the run's input snapshot (`_snapshot_millis`) when one is recorded, so all of
    a run's family jobs read byte-identical source data even if the table is written to mid-run.

    Reads through the BigQuery Storage Read API in **Arrow** format (``readDataFormat=ARROW``, set
    explicitly rather than relying on the connector default) — the columnar, zero-copy path into the
    executor-side pandas frames the cells consume. ``compute.read_max_streams`` (when > 0) caps the
    number of Storage Read streams via the connector's ``maxParallelism`` option; 0 lets the server
    size it from the table.
    """
    table = _resolve_source_table(cfg, settings)
    reader = spark.read.format("bigquery").option("table", table)
    # Arrow is the columnar Storage Read path; set it explicitly so the format never silently
    # depends on a connector default.
    reader = reader.option("readDataFormat", "ARROW")
    if cfg.compute.read_max_streams > 0:
        reader = reader.option("maxParallelism", str(cfg.compute.read_max_streams))
    ms = _snapshot_millis(cfg, settings)
    if ms is not None:
        # Pin the connector read to the run's snapshot so every family job time-travels to the
        # identical input instant. ``snapshotTimeMillis`` is the connector's BigQuery time-travel
        # option (native + managed-Iceberg both read through the same table interface).
        reader = reader.option("snapshotTimeMillis", str(ms))
    df = reader.load().select(*_needed_columns(cfg))
    return _limit_series(df, cfg)


def _limit_series(df: DataFrame, cfg: RunConfig) -> DataFrame:
    """Keep the first ``series_limit`` ts_ids (ordered) via a semi-join; pass-through when unset."""
    limit = cfg.data.series_limit
    if limit is None:
        return df
    id_col = cfg.data.ts_id_col
    keep = df.select(id_col).distinct().orderBy(id_col).limit(limit)
    return df.join(keep, on=id_col, how="leftsemi")


def sample_series_to_driver(df: DataFrame, cfg: RunConfig, k: int) -> list[pd.DataFrame]:
    """Collect the first ``k`` series (deterministically) to the driver as per-series frames.

    The fleetwide-HPO pre-pass tunes on a small sample *before* the cluster fan-out; that sample
    must live on the driver (Optuna runs there, not in an executor). Reuses the same deterministic
    "first ``k`` ts_ids, ordered" subset as `_limit_series` so the tuning sample is stable and
    apples-to-apples across scales. Returns one pandas frame per ts_id (the shape ``run_cell`` /
    ``backtest_cell`` expect); an empty list if the source has no rows. Tiny by construction —
    ``k`` is ``hpo.sample_size`` (default 20), not the fleet.
    """
    id_col = cfg.data.ts_id_col
    keep = df.select(id_col).distinct().orderBy(id_col).limit(k)
    pdf = df.join(keep, on=id_col, how="leftsemi").toPandas()
    return [g.reset_index(drop=True) for _, g in pdf.groupby(id_col, sort=True)]


def resolve_fleetwide_hpo(
    source: DataFrame, cfg: RunConfig, executed: list[str]
) -> dict[str, dict[str, Any]] | None:
    """Driver-side fleetwide-HPO pre-pass: sample series → tune each model → ``{model: params}``.

    Returns ``None`` (no fleetwide params — cells resolve per-series/off in ``run_cell``) unless HPO
    is enabled at ``fleetwide`` granularity. When it is, collects ``hpo.sample_size`` series to the
    driver (`sample_series_to_driver`) and tunes the executed model subset on them
    (`resolve_fleetwide`), scoping the tuning to the models that will
    actually run. Kept in ``spark_io`` as shared engine plumbing (one pre-pass); the Ray engine has
    its own analog over its already-collected pandas source.
    """
    if not (cfg.hpo.enabled and cfg.hpo.granularity == "fleetwide"):
        return None
    from ..hpo import resolve_fleetwide

    sample = sample_series_to_driver(source, cfg, cfg.hpo.sample_size)
    tuning_cfg = cfg.model_copy(update={"models": executed})
    return resolve_fleetwide(sample, tuning_cfg)


def cross_join_models(
    df: DataFrame, cfg: RunConfig, spark: SparkSession, models: list[str] | None = None
) -> DataFrame:
    """Cross-join the series with the (small) model list → one row-set per ``(series, model)``.

    The model list is broadcast (a handful of rows), so this is a map-side replication of each
    series, not a shuffle join. The resulting `_MODEL_COL` is the cell's model and part of
    the explode bucket key.

    ``models`` is the executed subset: a mixed config under `main.run` cross-joins
    only its Python-runtime models — cross-joining a BigQuery-native model would create Spark cells
    whose ``run_cell`` raises ``NotImplementedError``. ``None`` means ``cfg.models`` (standalone).
    """
    from pyspark.sql import functions as F

    executed = models if models is not None else cfg.models
    models_df = spark.createDataFrame([(m,) for m in executed], [_MODEL_COL])
    return df.crossJoin(F.broadcast(models_df))


def add_bucket(df: DataFrame, cfg: RunConfig, n_buckets: int) -> DataFrame:
    """Add ``_BUCKET_COL`` = ``pmod(hash(<key cols>), n_buckets)`` (see `bucket_key_cols`).

    Spark's ``hash`` keeps identical keys together (so a cell's full history shares a bucket) and
    ``pmod`` folds it into ``[0, n_buckets)`` non-negative. The bucket is the ``groupBy`` key the
    engines run one task per.
    """
    from pyspark.sql import functions as F

    key = [F.col(c) for c in bucket_key_cols(cfg)]
    return df.withColumn(_BUCKET_COL, F.pmod(F.hash(*key), F.lit(n_buckets)))


def status_schema() -> Any:
    """The Spark ``StructType`` for the UDF's return frame (`STATUS_COLUMNS`)."""
    from pyspark.sql.types import (
        DoubleType,
        StringType,
        StructField,
        StructType,
    )

    return StructType(
        [
            StructField("ts_id", StringType(), True),
            StructField("model_type", StringType(), True),
            StructField("status", StringType(), True),
            StructField("fit_seconds", DoubleType(), True),
        ]
    )


def make_group_runner(
    cfg: RunConfig,
    settings: Settings,
    models: list[str] | None = None,
    params_by_model: dict[str, dict[str, Any]] | None = None,
) -> Any:
    """Build the ``applyInPandas`` function: run one bucket's cells, write them, return status.

    Closes over the picklable ``cfg`` and the frozen `Settings` dataclass **directly** — not
    a Spark broadcast. ``applyInPandas`` cloudpickles the whole closure to each executor, so
    capturing the small immutable ``Settings`` by value is equivalent to a broadcast here (the
    object is tiny and read-only) while dropping the ``sparkContext.broadcast`` call that a **Spark
    Connect** session does not expose (Connect has no RDD/``sparkContext`` API). This is what lets
    the *same* engine run both as a classic Dataproc batch and driven over a Connect endpoint from a
    notebook (one code path, no per-environment fork). On each bucket it calls the pure
    `run_group`, appends the results with the writer
    (`write_cells`, executor-side, once per bucket), and
    returns only the compact status frame — so no forecast payload ever crosses back to the driver.

    ``models`` is the executed subset, forwarded to `run_group` so the untagged-frame loop runs
    only the Python-runtime models under a mixed `main.run` config. ``None`` → ``cfg.models``.

    ``params_by_model`` is the fleetwide-HPO resolution, captured in the closure exactly like
    ``settings``/``models`` and forwarded to `run_group` — so the driver tunes once and every
    executor builds its cells with the tuned params, without those params entering ``cfg`` (which
    would shift the run_id). ``None`` = no fleetwide params (per-series/off resolves in
    ``run_cell``).
    """

    def _run(pdf: pd.DataFrame) -> pd.DataFrame:
        from ..registry.cells import write_cells

        results, status = run_group(pdf, cfg, models, params_by_model)
        if results:
            write_cells(results, settings=settings)
        return status

    return _run
