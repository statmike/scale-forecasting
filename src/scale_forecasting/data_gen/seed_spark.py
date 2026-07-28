"""PySpark serverless seed entrypoint (Arc B) — CONTRACTS §6, DESIGN §13.1-seed.

Generates the shipped example dataset — ``n_series`` deterministic synthetic series — and writes
it to the managed-Iceberg ``source_series`` table with a **Dataproc Serverless Spark** job. This
is deliberately the platform's own core pattern (parallel Spark + high-throughput BigQuery
writes), so the very first thing a deployment does also serves as a **Spark scale smoke** for the
write path before any forecast runs.

Pure/shell split (mirrors worker vs engines, CONTRACTS §0):

* :mod:`data_gen.generator` — pure panel math. Each series is seeded by its own index, so
  ``generate_panel(n)`` equals the union of any partitioning of ``range(n)`` — the invariant
  this job relies on to fan generation across executors.
* This module — the Spark shell: partition ``range(n_series)`` across executors, call the pure
  generator per partition, reconcile to the ``source_series`` schema (:func:`_to_source_rows`,
  pure and offline-tested), and write via the spark-bigquery connector.

**Write path (B0.4 decision).** Primary is the connector's ``writeMethod=direct`` (BigQuery
Storage Write API, no temp bucket) — pre-installed on Dataproc Serverless. ``indirect`` (Spark
writes Parquet to GCS, then a BigQuery load) is the documented fallback if managed-Iceberg
direct-write has a rough edge; select it with ``--write-method indirect``. Both write in APPEND
mode (managed Iceberg rejects truncate); replace-on-reseed is a driver-side ``DELETE ... WHERE
TRUE`` before the write (the B0.3-proven delete-then-append shape).

**Infra identity** is resolved from the ``SF_*`` environment via :class:`~scale_forecasting.
settings.Settings` (G1). Dataproc Serverless rejects driver-env Spark properties, so the batch
passes the identity as ``--sf-*`` job args, which :func:`main` exports into ``os.environ`` on the
driver before resolution — keeping env-based ``Settings`` the single G1 seam.

Public surface: ``main(argv)``. ``pyspark`` and GCP clients import lazily inside the functions
that need them, so this module imports cleanly offline (parity with the engines) and
:func:`_to_source_rows` is unit-testable without Spark.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..errors import get_logger

if TYPE_CHECKING:
    import pandas as pd

    from ..settings import Settings

_log = get_logger(__name__)

# The source_series column order, verbatim from the DDL (registry/ddl.py). Both the pandas
# reconciliation and the Spark schema below follow this order so a positional write is safe.
_SOURCE_COLUMNS: tuple[str, ...] = ("ts_id", "ds", "y", "archetype", "price_index", "is_holiday")


@dataclass(frozen=True)
class SeedArgs:
    """Parsed seed knobs — the *shape* of the example dataset (not infra; that's ``Settings``)."""

    n_series: int
    master_seed: int
    history: int
    freq: str
    start: str
    holidays: tuple[str, ...]
    with_exog: bool
    write_method: str  # "direct" | "indirect"
    num_partitions: int  # Spark parallelism for generation


# The (--flag → SF_* env var) mapping for the infra-identity args. One place so the parser, the
# exporter, and the Terraform module stay in agreement.
_INFRA_ARG_ENV: tuple[tuple[str, str], ...] = (
    ("sf_project_id", "SF_PROJECT_ID"),
    ("sf_connection", "SF_CONNECTION"),
    ("sf_warehouse_uri", "SF_WAREHOUSE_URI"),
    ("sf_dataset_id", "SF_DATASET_ID"),
    ("sf_region", "SF_REGION"),
)


def _export_infra_env(ns: argparse.Namespace) -> None:
    """Copy any provided ``--sf-*`` args into ``os.environ`` before ``Settings.resolve()``.

    Dataproc Serverless rejects driver-env Spark properties, so the batch delivers the infra
    identity as args; exporting them here (only when set) keeps env-based resolution the single G1
    seam without forking a "resolve from args" path. Local runs pass no ``--sf-*`` and use the
    ambient environment untouched.
    """
    import os

    for attr, env_name in _INFRA_ARG_ENV:
        value = getattr(ns, attr, None)
        if value:
            os.environ[env_name] = value


def _parse_args(argv: list[str] | None) -> SeedArgs:
    """Parse the CLI knobs the Terraform ``seed`` module passes to the batch."""
    p = argparse.ArgumentParser(prog="seed_spark", description="Seed the source_series table.")
    p.add_argument("--n-series", type=int, default=100_000)
    p.add_argument("--master-seed", type=int, default=20260726)
    p.add_argument("--history", type=int, default=1460)
    p.add_argument("--freq", type=str, default="D")
    p.add_argument("--start", type=str, default="2021-01-01")
    # Comma-separated country codes, e.g. "US" or "US,CA". Empty string → no holidays.
    p.add_argument("--holidays", type=str, default="US")
    p.add_argument("--with-exog", action="store_true")
    p.add_argument("--write-method", type=str, choices=("direct", "indirect"), default="direct")
    # 0 → let the driver derive a sensible default from n_series.
    p.add_argument("--num-partitions", type=int, default=0)
    # Infra identity delivered as args (not env): Dataproc Serverless allowlists Spark property
    # prefixes and rejects driver-env, so the batch passes SF_* here and main() exports them to
    # os.environ before Settings.resolve() — keeping env-based resolution the single G1 seam. When
    # unset (local runs), the ambient SF_* environment is used as-is.
    p.add_argument("--sf-project-id", type=str, default=None)
    p.add_argument("--sf-connection", type=str, default=None)
    p.add_argument("--sf-warehouse-uri", type=str, default=None)
    p.add_argument("--sf-dataset-id", type=str, default=None)
    p.add_argument("--sf-region", type=str, default=None)
    ns = p.parse_args(argv)

    _export_infra_env(ns)
    holidays = tuple(c.strip() for c in ns.holidays.split(",") if c.strip())
    num_partitions = ns.num_partitions or _default_partitions(ns.n_series)
    return SeedArgs(
        n_series=ns.n_series,
        master_seed=ns.master_seed,
        history=ns.history,
        freq=ns.freq,
        start=ns.start,
        holidays=holidays,
        with_exog=ns.with_exog,
        write_method=ns.write_method,
        num_partitions=num_partitions,
    )


def _default_partitions(n_series: int) -> int:
    """A reasonable partition count: ~2k series per task, clamped to [1, 512].

    Keeps each task's generated frame modest (a few hundred MB at daily history) while giving
    the autoscaler enough parallelism for a real scale smoke. A dedicated knob (``--num-
    partitions``) overrides this when tuning.
    """
    return max(1, min(512, -(-n_series // 2000)))


def _to_source_rows(df: pd.DataFrame, holidays: tuple[str, ...]) -> pd.DataFrame:
    """Reconcile one generator partition to the ``source_series`` schema (pure, no Spark).

    The generator emits ``ts_id, archetype, ds(datetime64[ns]), y [, price_index]``; the table
    is ``ts_id STRING, ds DATE, y FLOAT64, archetype STRING, price_index FLOAT64, is_holiday
    BOOL``. This casts ``ds`` to python ``date``, derives ``is_holiday`` from the same calendar
    as the panel's holiday bump (parity, via :func:`~data_gen.generator.is_holiday_flags`), fills
    ``price_index`` with NA when the generator didn't emit it (``with_exog=False``), and projects
    to the DDL column order. Pure → unit-tested offline against a tiny generator call.
    """
    import pandas as pd

    from .generator import is_holiday_flags

    if df.empty:
        empty = {c: pd.Series(dtype="object") for c in _SOURCE_COLUMNS}
        return pd.DataFrame(empty)

    price = (
        df["price_index"].astype("float64")
        if "price_index" in df.columns
        else pd.Series([pd.NA] * len(df), dtype="Float64")
    )
    out = pd.DataFrame(
        {
            "ts_id": df["ts_id"].astype("string"),
            "ds": pd.to_datetime(df["ds"]).dt.date,
            "y": df["y"].astype("float64"),
            "archetype": df["archetype"].astype("string"),
            "price_index": price,
            "is_holiday": pd.Series(is_holiday_flags(df["ds"], holidays), dtype="boolean"),
        }
    )
    return out[list(_SOURCE_COLUMNS)]


def _source_series_schema() -> object:
    """Explicit Spark ``StructType`` matching the ``source_series`` DDL (STRING/DATE/DOUBLE/BOOL).

    Declared explicitly (not inferred) so an all-NULL ``price_index`` partition can't collapse
    the column to the wrong type and so the write matches the managed-Iceberg table exactly.
    """
    from pyspark.sql.types import (
        BooleanType,
        DateType,
        DoubleType,
        StringType,
        StructField,
        StructType,
    )

    return StructType(
        [
            StructField("ts_id", StringType(), False),
            StructField("ds", DateType(), False),
            StructField("y", DoubleType(), True),
            StructField("archetype", StringType(), True),
            StructField("price_index", DoubleType(), True),
            StructField("is_holiday", BooleanType(), True),
        ]
    )


def _clear_existing(settings: Settings) -> None:
    """DELETE all rows for a clean re-seed (managed Iceberg rejects WRITE_TRUNCATE, B0.3).

    A no-op on the first seed (empty table). NOTE: right after a ``direct`` write the rows sit in
    the Storage Write API buffer (~90 min) and cannot be DELETE-d during that window; an immediate
    re-seed should use ``--write-method indirect`` or wait out the buffer (see the B0.4 NOTES).
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    table = settings.table_ref("source_series")
    client = bigquery.Client(project=settings.project_id)
    try:
        client.query(f"DELETE FROM `{table}` WHERE TRUE").result()
    except Exception as exc:  # noqa: BLE001 - re-raised with table context
        raise RegistryError(f"seed clear of {table} failed: {exc}") from exc


def main(argv: list[str] | None = None) -> None:
    """Run the seed job: generate ``n_series`` series and write ``source_series``.

    Entrypoint for the Dataproc Serverless PySpark batch (invoked via ``seed_entry.py``).
    """
    from pyspark.sql import SparkSession

    from ..registry import bq
    from ..settings import Settings
    from .generator import GenConfig, generate_partition

    args = _parse_args(argv)
    settings = Settings.resolve()
    table = settings.table_ref("source_series")
    _log.info(
        "seed start: n_series=%d partitions=%d write_method=%s -> %s",
        args.n_series,
        args.num_partitions,
        args.write_method,
        table,
    )

    # Driver-side: guarantee the managed-Iceberg table exists, then clear for a clean re-seed.
    bq.ensure_tables(settings=settings)
    _clear_existing(settings)

    gen_cfg = GenConfig(
        history=args.history,
        freq=args.freq,
        start=args.start,
        holidays=args.holidays,
        with_exog=args.with_exog,
    )
    # Bind loop-invariants into locals so the executor closure captures values, not `args`.
    holidays = args.holidays
    master_seed = args.master_seed

    def partition_to_rows(ids: object) -> object:
        # Runs on executors: generate this id-slice's panel and yield source_series row dicts.
        id_list = list(ids)  # type: ignore[call-overload]
        if not id_list:
            return iter(())
        frame = generate_partition(id_list, gen_cfg, master_seed)
        rows = _to_source_rows(frame, holidays)
        # pd.NA → None so Spark writes SQL NULL (e.g. price_index when with_exog=False).
        records = rows.astype(object).where(rows.notna(), None).to_dict("records")
        return iter(records)

    spark = SparkSession.builder.appName("scale-forecasting-seed").getOrCreate()
    try:
        ids_rdd = spark.sparkContext.parallelize(
            range(args.n_series), numSlices=args.num_partitions
        )
        row_rdd = ids_rdd.mapPartitions(partition_to_rows)
        sdf = spark.createDataFrame(row_rdd, schema=_source_series_schema())

        writer = (
            sdf.write.format("bigquery")
            .option("table", table)
            .option("writeMethod", args.write_method)
        )
        if args.write_method == "indirect":
            # indirect stages Parquet to GCS before the BQ load; give it a temp location in the
            # warehouse bucket (the connector cleans it up after the load).
            bucket = settings.warehouse_uri.removeprefix("gs://").split("/", 1)[0]
            writer = writer.option("temporaryGcsBucket", bucket)
        writer.mode("append").save()
        _log.info("seed complete: wrote %d series to %s", args.n_series, table)
    finally:
        spark.stop()


if __name__ == "__main__":  # pragma: no cover - cluster entrypoint
    main()
