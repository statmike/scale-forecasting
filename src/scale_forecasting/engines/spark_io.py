"""Shared Spark-engine plumbing — the pieces ``spark_explode`` and ``spark_naive`` both need.

Split along the pure/I-O seam (CONTRACTS §0) so the interesting logic is offline-testable:

* **Pure** (no Spark, no BigQuery): :func:`run_group` — the body of the grouped-Pandas UDF, which
  runs :func:`~scale_forecasting.worker.run_cell` for every cell in one group's pandas frame and
  returns ``(results, status_frame)``; :func:`aggregate_status` — fold the per-cell status frame
  into a run-level :class:`RunOutcome`; :func:`default_bucket_count`, :func:`bucket_key_cols`.
* **I/O / Spark shell** (pyspark imported lazily, parity with the seed job):
  :func:`read_source_series` (connector read + deterministic ``series_limit`` subset),
  :func:`add_bucket`, :func:`cross_join_models`, :func:`status_schema`, :func:`make_group_runner`.

**Fan-out mechanics.** Both engines shuffle cells into *buckets* and run one Spark task per bucket
(``groupBy(bucket).applyInPandas``). The bucket key is the natural unit of independence for the
method (:func:`bucket_key_cols`):

* ``explode`` buckets on ``(ts_id, model_type)`` — the whole *cell*. A slow ``(series, deep-model)``
  cell lands in its own bucket while that same series' fast cells sit in *other* buckets and run
  concurrently, so the scheduler spreads cells and autoscales. Every history row of a given cell
  shares both keys, so a cell's full series stays intact in one bucket.
* ``naive`` buckets on ``ts_id`` alone — all of a series' models land together and run sequentially
  in one task (the loop in :func:`run_group`). One slow model holds an executor for the whole
  series → the straggler anti-pattern the demo exists to show.

**Writes are executor-side, once per bucket.** :func:`make_group_runner` wraps :func:`run_group`
and calls the B1-validated :func:`~scale_forecasting.registry.bq.write_cells` on each bucket's
results (all three cell tables + artifact upload, unchanged). Append-only + dedupe-on-read
(§3.4) makes per-partition appends safe. The UDF returns only the compact status frame; the driver
owns the header. Infra identity reaches executors by capturing the frozen :class:`Settings`
dataclass (picklable, tiny, read-only) directly in the group-runner closure, which ``applyInPandas``
cloudpickles to each executor — preserving the G1 seam without a second env-based delivery path, and
without the ``sparkContext.broadcast`` call that a Spark Connect session does not expose (so the
same engine runs both as a Dataproc batch and driven over a Connect endpoint).
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
    """The columns whose hash defines a bucket — the method's unit of independence.

    ``naive`` → ``[ts_id]`` (a whole series per task, models run sequentially → stragglers).
    ``explode``/``multi`` → ``[ts_id, model_type]`` (a whole cell per bucket, cells spread so a
    slow cell can't block fast ones). See the module docstring for why this is the crux of the
    scaling demo.
    """
    id_col = cfg.data.ts_id_col
    if cfg.spark_method == "naive":
        return [id_col]
    return [id_col, _MODEL_COL]


def default_bucket_count(cfg: RunConfig, models: list[str] | None = None) -> int:
    """Bucket count that keeps each ``applyInPandas`` frame bounded as scale grows.

    Buckets are *shuffle partitions*, not executor concurrency (that's
    ``spark.dynamicAllocation.maxExecutors``): each bucket is materialized as one pandas frame in
    one task, so its size — not the cluster width — is what OOMs an executor. We therefore size
    buckets to hold ~``compute.bucket_target_cells`` cells each: ``buckets = ceil(cells / target)``.
    That holds ~``target`` series-histories per frame at *every* scale (1k and 100k alike), instead
    of the old ``min(cells, max_parallelism)`` which silently fattened frames past the executor
    memory budget once ``cells`` outgrew the cap (the 100k OOM). Clamped to ``[1, _MAX_BUCKETS]``.

    With ``series_limit`` set the cell count is known offline (series × models for explode, series
    for naive); an unbounded run falls back to ``max_parallelism`` buckets (best guess without a
    known cell count). ``models`` is the executed subset (Arc B); ``None`` means ``cfg.models`` — so
    a standalone run and a subset run size buckets to the work they actually fan out.
    """
    executed = models if models is not None else cfg.models
    target = cfg.compute.bucket_target_cells
    limit = cfg.data.series_limit
    if limit is None:
        return max(1, min(cfg.compute.max_parallelism, _MAX_BUCKETS))
    cells = limit if cfg.spark_method == "naive" else limit * len(executed)
    return max(1, min(math.ceil(cells / target), _MAX_BUCKETS))


# --- pure: the grouped-UDF body ------------------------------------------------


def run_group(
    pdf: pd.DataFrame,
    cfg: RunConfig,
    models: list[str] | None = None,
    params_by_model: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[CellResult], pd.DataFrame]:
    """Run every cell in one group's pandas frame; pure — no Spark, no BigQuery.

    The frame is one bucket's rows. If it carries the internal model column (explode/multi: the
    series were cross-joined with the model list), each ``(ts_id, model_type)`` sub-frame is one
    cell. Otherwise (naive) every model in the executed list is run for each ``ts_id`` in a
    sequential loop. Helper columns are dropped so each sub-frame is a clean series frame for
    :func:`~scale_forecasting.worker.run_cell` (which derives the run_id from ``cfg`` itself, so no
    id needs threading here). Returns the :class:`CellResult` list (for the writer) and the compact
    :data:`STATUS_COLUMNS` frame (for the driver's header roll-up). Never raises per cell —
    ``run_cell`` maps a failure to a ``status="error"`` result (CONTRACTS §3.3).

    ``models`` is the executed subset (Arc B): under :func:`main.run` a mixed config routes only its
    Python-runtime models here while the BigQuery-native models run elsewhere, so the naive loop
    must iterate the subset, not the full ``cfg.models`` (which would feed a native model into
    ``run_cell`` → ``NotImplementedError``). ``None`` means ``cfg.models`` (standalone). The explode
    path takes its models from the cross-join column, so the subset only affects the naive loop.

    ``params_by_model`` is the fleetwide-HPO resolution (C5): ``{model: tuned params}`` computed
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
        # explode/multi: the cross-join tagged each row with its model; one cell per (ts_id, model).
        for (_ts_id, model_name), sub in pdf.groupby([id_col, _MODEL_COL], sort=False):
            series = sub.drop(columns=helper_cols)
            results.append(run_cell(series, str(model_name), cfg, by_model.get(str(model_name))))
    else:
        # naive: one task per series, all models sequentially — the deliberate anti-pattern.
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
    """Fold the collected per-cell status frame into a :class:`RunOutcome` (pure).

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
    qualifier, else resolve a bare name against the deployment's dataset (G1)."""
    src = cfg.data.source_table
    return src if "." in src else settings.table_ref(src)


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
    semi-join) so every scale in the demo runs the *same* series (DESIGN §13.1) — the property that
    makes "10 vs 100 vs 100k" a clean apples-to-apples runtime comparison.
    """
    table = _resolve_source_table(cfg, settings)
    df = spark.read.format("bigquery").option("table", table).load().select(*_needed_columns(cfg))
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
    """Collect the first ``k`` series (deterministically) to the driver as per-series frames (C5).

    The fleetwide-HPO pre-pass tunes on a small sample *before* the cluster fan-out; that sample
    must live on the driver (Optuna runs there, not in an executor). Reuses the same deterministic
    "first ``k`` ts_ids, ordered" subset as :func:`_limit_series` so the tuning sample is stable and
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
    driver (:func:`sample_series_to_driver`) and tunes the executed model subset on them
    (:func:`~scale_forecasting.hpo.resolve_fleetwide`), scoping the tuning to the models that will
    actually run (Arc B). Kept in ``spark_io`` so all Spark engines (explode/naive) share one
    pre-pass; the Ray engine has its own analog over its already-collected pandas source.
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
    series, not a shuffle join. The resulting :data:`_MODEL_COL` is the cell's model and part of
    the explode bucket key.

    ``models`` is the executed subset (Arc B): a mixed config under :func:`main.run` cross-joins
    only its Python-runtime models — cross-joining a BigQuery-native model would create Spark cells
    whose ``run_cell`` raises ``NotImplementedError``. ``None`` means ``cfg.models`` (standalone).
    """
    from pyspark.sql import functions as F

    executed = models if models is not None else cfg.models
    models_df = spark.createDataFrame([(m,) for m in executed], [_MODEL_COL])
    return df.crossJoin(F.broadcast(models_df))


def add_bucket(df: DataFrame, cfg: RunConfig, n_buckets: int) -> DataFrame:
    """Add ``_BUCKET_COL`` = ``pmod(hash(<key cols>), n_buckets)`` (see :func:`bucket_key_cols`).

    Spark's ``hash`` keeps identical keys together (so a cell's full history shares a bucket) and
    ``pmod`` folds it into ``[0, n_buckets)`` non-negative. The bucket is the ``groupBy`` key the
    engines run one task per.
    """
    from pyspark.sql import functions as F

    key = [F.col(c) for c in bucket_key_cols(cfg)]
    return df.withColumn(_BUCKET_COL, F.pmod(F.hash(*key), F.lit(n_buckets)))


def status_schema() -> Any:
    """The Spark ``StructType`` for the UDF's return frame (:data:`STATUS_COLUMNS`)."""
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

    Closes over the picklable ``cfg`` and the frozen :class:`Settings` dataclass **directly** — not
    a Spark broadcast. ``applyInPandas`` cloudpickles the whole closure to each executor, so
    capturing the small immutable ``Settings`` by value is equivalent to a broadcast here (the
    object is tiny and read-only) while dropping the ``sparkContext.broadcast`` call that a **Spark
    Connect** session does not expose (Connect has no RDD/``sparkContext`` API). This is what lets
    the *same* engine run both as a classic Dataproc batch and driven over a Connect endpoint from a
    notebook (G1: one code path, no per-environment fork). On each bucket it calls the pure
    :func:`run_group`, appends the results with the B1 writer
    (:func:`~scale_forecasting.registry.bq.write_cells`, executor-side, once per bucket), and
    returns only the compact status frame — so no forecast payload ever crosses back to the driver.

    ``models`` is the executed subset (Arc B), forwarded to :func:`run_group` so the naive loop runs
    only the Python-runtime models under a mixed :func:`main.run` config. ``None`` → ``cfg.models``.

    ``params_by_model`` is the fleetwide-HPO resolution (C5), captured in the closure exactly like
    ``settings``/``models`` and forwarded to :func:`run_group` — so the driver tunes once and every
    executor builds its cells with the tuned params, without those params entering ``cfg`` (which
    would shift the run_id). ``None`` = no fleetwide params (per-series/off resolves in
    ``run_cell``).
    """

    def _run(pdf: pd.DataFrame) -> pd.DataFrame:
        from ..registry import bq

        results, status = run_group(pdf, cfg, models, params_by_model)
        if results:
            bq.write_cells(results, settings=settings)
        return status

    return _run
