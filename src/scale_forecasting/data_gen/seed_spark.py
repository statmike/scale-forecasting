"""PySpark serverless seed entrypoint.

Generates the shipped example dataset — ``n_series`` deterministic synthetic series — and writes
it to the source table(s) with a **Dataproc Serverless Spark** job. The example input ships in
**both** storage formats: ``source_series_iceberg`` (managed Apache Iceberg on GCS) and
``source_series_native`` (plain native BigQuery), seeded from the *same* generated panel so a
deployment can benchmark the identical series on either storage. This is deliberately the
platform's own core pattern (parallel Spark + high-throughput BigQuery writes), so the very first
thing a deployment does also serves as a **Spark scale smoke** for the write path before any
forecast runs.

Pure/shell split (mirrors worker vs engines):

* `data_gen.generator` — pure panel math. Each series is seeded by its own index, so
  ``generate_panel(n)`` equals the union of any partitioning of ``range(n)`` — the invariant
  this job relies on to fan generation across executors.
* This module — the Spark shell: partition ``range(n_series)`` across executors, call the pure
  generator per partition, reconcile to the ``source_series`` schema (`_to_source_rows`,
  pure and offline-tested), and write via the spark-bigquery connector.

**Write path.** Primary is the connector's ``writeMethod=direct`` (BigQuery
Storage Write API, no temp bucket) — pre-installed on Dataproc Serverless. ``indirect`` (Spark
writes Parquet to GCS, then a BigQuery load) is the documented fallback if direct-write has a
rough edge; select it with ``--write-method indirect``. Both variants are written in APPEND mode;
replace-on-reseed differs by format: the **native** table is cleared with ``TRUNCATE TABLE`` (a
metadata op that clears even the streaming buffer), while **managed Iceberg** rejects truncate so
its reseed is a driver-side ``DELETE ... WHERE TRUE`` (the delete-then-append shape;
subject to the ~90-min buffer window on an immediate re-seed).

Which variant(s) to seed is selected with ``--variant {iceberg,native,both}`` (default ``both``).
Both are generated from one panel pass, so the series are identical across formats.

**Infra identity** is resolved from the ``SF_*`` environment via `Settings`. Dataproc
Serverless rejects driver-env Spark properties, so the batch passes the identity as ``--sf-*``
job args, which `main` exports into ``os.environ`` on the driver before resolution — keeping
env-based ``Settings`` the single seam.

Public surface: ``main(argv)``. ``pyspark`` and GCP clients import lazily inside the functions
that need them, so this module imports cleanly offline (parity with the engines) and
`_to_source_rows` is unit-testable without Spark.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .._infra_args import add_infra_args, export_infra_env
from ..errors import get_logger

if TYPE_CHECKING:
    import pandas as pd

    from ..settings import Settings

_log = get_logger(__name__)

# The source_series column order, verbatim from the DDL (registry/ddl.py). Both the pandas
# reconciliation and the Spark schema below follow this order so a positional write is safe.
_SOURCE_COLUMNS: tuple[str, ...] = ("ts_id", "ds", "y", "archetype", "is_holiday")


@dataclass(frozen=True)
class SeedArgs:
    """Parsed seed knobs — the *shape* of the example dataset (not infra; that's ``Settings``)."""

    n_series: int
    master_seed: int
    history: int
    freq: str
    start: str
    holidays: tuple[str, ...]
    write_method: str  # "direct" | "indirect"
    num_partitions: int  # Spark parallelism for generation
    variant: str  # "iceberg" | "native" | "both"
    include_covariates: bool = False
    driver_load: bool = False


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
    p.add_argument("--write-method", type=str, choices=("direct", "indirect"), default="direct")
    # 0 → let the driver derive a sensible default from n_series.
    p.add_argument("--num-partitions", type=int, default=0)
    # Which storage format(s) of the example input to seed (both share one generated panel).
    p.add_argument("--variant", type=str, choices=("iceberg", "native", "both"), default="both")
    p.add_argument(
        "--include-covariates",
        action="store_true",
        default=False,
        help="Include hierarchy (region, category) and three-tier covariate columns.",
    )
    p.add_argument(
        "--driver-load",
        action="store_true",
        default=False,
        help="Generate in-process and load via BigQuery Load API instead of PySpark.",
    )
    # Infra identity delivered as args (not env): Dataproc Serverless allowlists Spark property
    # prefixes and rejects driver-env, so the batch passes SF_* here and main() exports them to
    # os.environ before Settings.resolve() — keeping env-based resolution the single seam. The
    # --sf-* flags + the exporter live in _infra_args so every entrypoint shares one mapping.
    add_infra_args(p)
    ns = p.parse_args(argv)

    export_infra_env(ns)
    holidays = tuple(c.strip() for c in ns.holidays.split(",") if c.strip())
    num_partitions = ns.num_partitions or _default_partitions(ns.n_series)
    return SeedArgs(
        n_series=ns.n_series,
        master_seed=ns.master_seed,
        history=ns.history,
        freq=ns.freq,
        start=ns.start,
        holidays=holidays,
        write_method=ns.write_method,
        num_partitions=num_partitions,
        variant=ns.variant,
        include_covariates=bool(ns.include_covariates),
        driver_load=bool(ns.driver_load),
    )


def _default_partitions(n_series: int) -> int:
    """A reasonable partition count: ~2k series per task, clamped to [1, 512].

    Keeps each task's generated frame modest (a few hundred MB at daily history) while giving
    the autoscaler enough parallelism for a real scale smoke. A dedicated knob (``--num-
    partitions``) overrides this when tuning.
    """
    return max(1, min(512, -(-n_series // 2000)))


_COVARIATE_COLUMNS: tuple[str, ...] = (
    "region",
    "category",
    "promo_flag",
    "price_index",
    "temperature",
)


def _to_source_rows(
    df: pd.DataFrame,
    holidays: tuple[str, ...],
    *,
    include_covariates: bool = False,
) -> pd.DataFrame:
    """Reconcile one generator partition to the ``source_series`` schema (pure, no Spark).

    The generator emits ``ts_id, archetype, ds(datetime64[ns]), y`` (plus optional hierarchy and
    covariate columns); the default table is ``ts_id STRING, ds DATE, y FLOAT64, archetype STRING,
    is_holiday BOOL``. This casts ``ds`` to python ``date``, derives ``is_holiday`` from the same
    calendar as the panel's holiday bump (parity, via `is_holiday_flags`), and projects to the DDL
    column order. Pass ``include_covariates=True`` to retain any hierarchy/covariate columns
    present on ``df`` (`region, category, promo_flag, price_index, temperature`).
    """
    import pandas as pd

    from .generator import is_holiday_flags

    extra_cols = [c for c in _COVARIATE_COLUMNS if include_covariates and c in df.columns]
    cols_order = [*_SOURCE_COLUMNS, *extra_cols]

    if df.empty:
        empty = {c: pd.Series(dtype="object") for c in cols_order}
        return pd.DataFrame(empty)

    data: dict[str, object] = {
        "ts_id": df["ts_id"].astype("string"),
        "ds": pd.to_datetime(df["ds"]).dt.date,
        "y": df["y"].astype("float64"),
        "archetype": df["archetype"].astype("string"),
        "is_holiday": pd.Series(is_holiday_flags(df["ds"], holidays), dtype="boolean"),
    }
    for c in extra_cols:
        if c in ("region", "category"):
            data[c] = df[c].astype("string")
        elif c == "promo_flag":
            data[c] = df[c].astype("int64")
        else:
            data[c] = df[c].astype("float64")
    out = pd.DataFrame(data)
    return out[cols_order]


def _source_series_schema(*, include_covariates: bool = False) -> object:
    """Explicit Spark ``StructType`` matching the ``source_series`` DDL (STRING/DATE/DOUBLE/BOOL).

    Declared explicitly (not inferred) so a type never collapses on an all-NULL partition and so
    the write matches the managed-Iceberg table exactly.
    """
    from pyspark.sql.types import (
        BooleanType,
        DateType,
        DoubleType,
        LongType,
        StringType,
        StructField,
        StructType,
    )

    fields = [
        StructField("ts_id", StringType(), False),
        StructField("ds", DateType(), False),
        StructField("y", DoubleType(), True),
        StructField("archetype", StringType(), True),
        StructField("is_holiday", BooleanType(), True),
    ]
    if include_covariates:
        fields.extend(
            [
                StructField("region", StringType(), True),
                StructField("category", StringType(), True),
                StructField("promo_flag", LongType(), True),
                StructField("price_index", DoubleType(), True),
                StructField("temperature", DoubleType(), True),
            ]
        )
    return StructType(fields)


def _clear_existing(settings: Settings, table_name: str, *, iceberg: bool) -> None:
    """Clear a source table's rows for a clean re-seed, format-appropriately.

    The **native** table uses ``TRUNCATE TABLE`` — a metadata op that clears the table including
    any rows still in the Storage Write API streaming buffer, so an immediate re-seed is clean. The
    **managed-Iceberg** table rejects truncate, so it falls back to ``DELETE ... WHERE
    TRUE``; right after a ``direct`` write those rows sit in the ~90-min buffer and can't be
    DELETE-d during that window (use ``--write-method indirect`` or wait it out). A no-op on the
    first seed (empty table).
    """
    from google.cloud import bigquery

    from ..errors import RegistryError

    table = settings.table_ref(table_name)
    stmt = f"DELETE FROM `{table}` WHERE TRUE" if iceberg else f"TRUNCATE TABLE `{table}`"
    client = bigquery.Client(project=settings.project_id)
    try:
        client.query(stmt).result()
    except Exception as exc:  # noqa: BLE001 - re-raised with table context
        raise RegistryError(f"seed clear of {table} failed: {exc}") from exc


SOURCE_COVARIATES_ICEBERG = "source_series_covariates_iceberg"
SOURCE_COVARIATES_NATIVE = "source_series_covariates_native"


def _variant_tables(
    variant: str,
    *,
    include_covariates: bool = False,
) -> tuple[tuple[str, bool], ...]:
    """Resolve ``--variant`` (and ``--include-covariates``) to ``(table_name, iceberg)`` pairs."""
    from ..registry.ddl import SOURCE_TABLE_ICEBERG, SOURCE_TABLE_NATIVE

    if include_covariates:
        iceberg = (SOURCE_COVARIATES_ICEBERG, True)
        native = (SOURCE_COVARIATES_NATIVE, False)
    else:
        iceberg = (SOURCE_TABLE_ICEBERG, True)
        native = (SOURCE_TABLE_NATIVE, False)
    return {
        "iceberg": (iceberg,),
        "native": (native,),
        "both": (iceberg, native),
    }[variant]


def _ensure_covariate_tables(
    settings: Settings,
    targets: tuple[tuple[str, bool], ...],
) -> None:
    """Create the 10-column covariate + hierarchy source tables if they do not yet exist."""
    from google.cloud import bigquery

    dataset = f"{settings.project_id}.{settings.dataset_id}"
    body = """\
CREATE TABLE IF NOT EXISTS `{dataset}.{table}` (
  ts_id        STRING NOT NULL,
  ds           DATE NOT NULL,
  y            FLOAT64,
  archetype    STRING,
  is_holiday   BOOL,
  region       STRING,
  category     STRING,
  promo_flag   INT64,
  price_index  FLOAT64,
  temperature  FLOAT64
)
"""
    client = bigquery.Client(project=settings.project_id)
    for table_name, is_iceberg in targets:
        if is_iceberg:
            storage_uri = f"{settings.warehouse_uri.rstrip('/')}/{table_name}"
            stmt = (
                body.format(dataset=dataset, table=table_name)
                + "CLUSTER BY ts_id\n"
                + f"WITH CONNECTION `{settings.connection}`\n"
                + "OPTIONS (\n"
                + "  file_format = 'PARQUET',\n"
                + "  table_format = 'ICEBERG',\n"
                + f"  storage_uri = '{storage_uri}'\n"
                + ");"
            )
        else:
            stmt = (
                body.format(dataset=dataset, table=table_name)
                + "PARTITION BY ds\nCLUSTER BY ts_id;"
            )
        client.query(stmt).result()


def _write_variant(sdf: object, settings: Settings, table_name: str, write_method: str) -> None:
    """Append the seeded DataFrame to one source table via the spark-bigquery connector."""
    table = settings.table_ref(table_name)
    writer = (
        sdf.write.format("bigquery")  # type: ignore[attr-defined]
        .option("table", table)
        .option("writeMethod", write_method)
    )
    if write_method == "indirect":
        # indirect stages Parquet to GCS before the BQ load; give it a temp location in the
        # warehouse bucket (the connector cleans it up after the load).
        bucket = settings.warehouse_uri.removeprefix("gs://").split("/", 1)[0]
        writer = writer.option("temporaryGcsBucket", bucket)
    writer.mode("append").save()


def _driver_load_variants(
    rows_df: pd.DataFrame,
    settings: Settings,
    targets: tuple[tuple[str, bool], ...],
    *,
    include_covariates: bool,
) -> None:
    """Load an in-memory pandas DataFrame directly into the target BigQuery / Iceberg table(s)."""
    from google.cloud import bigquery

    schema = [
        bigquery.SchemaField("ts_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("ds", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("y", "FLOAT64"),
        bigquery.SchemaField("archetype", "STRING"),
        bigquery.SchemaField("is_holiday", "BOOL"),
    ]
    if include_covariates:
        schema.extend(
            [
                bigquery.SchemaField("region", "STRING"),
                bigquery.SchemaField("category", "STRING"),
                bigquery.SchemaField("promo_flag", "INT64"),
                bigquery.SchemaField("price_index", "FLOAT64"),
                bigquery.SchemaField("temperature", "FLOAT64"),
            ]
        )
    client = bigquery.Client(project=settings.project_id)
    job_cfg = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
    )
    for name, _iceberg in targets:
        table_ref = settings.table_ref(name)
        client.load_table_from_dataframe(rows_df, table_ref, job_config=job_cfg).result()
        _log.info("driver-load complete: wrote %d rows to %s", len(rows_df), name)


def _empty_baseline_targets(
    settings: Settings,
    variant: str,
) -> tuple[tuple[str, bool], ...]:
    """Return baseline ``source_series_*`` tables for ``variant`` that currently have 0 rows.

    When ``--include-covariates`` is the default on a fresh ``terraform apply``, the primary
    targets are ``source_series_covariates_*`` (10 columns). Also backfilling any empty baseline
    ``source_series_*`` tables (5 columns, projected from the same cached panel) ensures that
    ``module.smoke``, notebooks, and univariate configs targeting ``source_series_native`` /
    ``source_series_iceberg`` work out of the box without overwriting already-seeded baseline
    tables on an existing deployment.
    """
    from google.cloud import bigquery

    client = bigquery.Client(project=settings.project_id)
    empty: list[tuple[str, bool]] = []
    for name, is_iceberg in _variant_tables(variant, include_covariates=False):
        table_ref = settings.table_ref(name)
        try:
            rows = list(client.query(f"SELECT 1 FROM `{table_ref}` LIMIT 1").result())
            if not rows:
                empty.append((name, is_iceberg))
        except Exception:  # noqa: BLE001 - skip backfill safely if check fails
            continue
    return tuple(empty)


def main(argv: list[str] | None = None) -> None:
    """Run the seed job: generate ``n_series`` series and write the source table variant(s).

    Entrypoint for the Dataproc Serverless PySpark batch (invoked via ``seed_entry.py``) or
    local driver-side seeding (``--driver-load``). Both source variants (when ``--variant both``)
    are seeded from one generated panel so the series are byte-identical across storage formats.
    """
    from ..registry.tables import ensure_tables
    from ..settings import Settings
    from .generator import GenConfig, generate_panel, generate_partition

    args = _parse_args(argv)
    settings = Settings.resolve()
    include_covs = args.include_covariates
    targets = _variant_tables(args.variant, include_covariates=include_covs)
    _log.info(
        "seed start: n_series=%d partitions=%d write_method=%s variant=%s covariates=%s -> %s",
        args.n_series,
        args.num_partitions,
        args.write_method,
        args.variant,
        include_covs,
        ", ".join(settings.table_ref(name) for name, _ in targets),
    )

    # Driver-side: guarantee the tables exist, then clear each target for a clean re-seed.
    ensure_tables(settings=settings)
    baseline_backfill: tuple[tuple[str, bool], ...] = ()
    if include_covs:
        _ensure_covariate_tables(settings, targets)
        baseline_backfill = _empty_baseline_targets(settings, args.variant)
    for name, iceberg in targets:
        _clear_existing(settings, name, iceberg=iceberg)

    gen_cfg = GenConfig(
        history=args.history,
        freq=args.freq,
        start=args.start,
        holidays=args.holidays,
        with_exog=include_covs,
        with_hierarchy=include_covs,
    )
    if args.driver_load:
        raw_df = generate_panel(args.n_series, gen_cfg, args.master_seed)
        rows_df = _to_source_rows(raw_df, args.holidays, include_covariates=include_covs)
        _driver_load_variants(rows_df, settings, targets, include_covariates=include_covs)
        if baseline_backfill:
            base_df = rows_df[list(_SOURCE_COLUMNS)]
            _driver_load_variants(base_df, settings, baseline_backfill, include_covariates=False)
        return

    from pyspark.sql import SparkSession

    # Bind loop-invariants into locals so the executor closure captures values, not `args`.
    holidays = args.holidays
    master_seed = args.master_seed

    def partition_to_rows(ids: object) -> object:
        # Runs on executors: generate this id-slice's panel and yield source_series row dicts.
        id_list = list(ids)  # type: ignore[call-overload]
        if not id_list:
            return iter(())
        frame = generate_partition(id_list, gen_cfg, master_seed)
        rows = _to_source_rows(frame, holidays, include_covariates=include_covs)
        # pd.NA → None so Spark writes SQL NULL for any missing cell.
        # (pandas-stubs<3 has no .where(cond, None) overload though it's valid at runtime.)
        records = (
            rows.astype(object)
            .where(rows.notna(), None)  # type: ignore[call-overload]
            .to_dict("records")
        )
        return iter(records)

    spark = SparkSession.builder.appName("scale-forecasting-seed").getOrCreate()
    try:
        ids_rdd = spark.sparkContext.parallelize(
            range(args.n_series), numSlices=args.num_partitions
        )
        row_rdd = ids_rdd.mapPartitions(partition_to_rows)
        sdf = spark.createDataFrame(
            row_rdd, schema=_source_series_schema(include_covariates=include_covs)
        )
        # Generate once, write to each target format. cache() so the second write reuses the same
        # rows instead of recomputing the panel (and so the two variants are provably identical).
        if len(targets) + len(baseline_backfill) > 1:
            sdf = sdf.cache()
        for name, _iceberg in targets:
            _write_variant(sdf, settings, name, args.write_method)
            _log.info("seed complete: wrote %d series to %s", args.n_series, name)
        if baseline_backfill:
            base_sdf = sdf.select(*_SOURCE_COLUMNS)
            for name, _iceberg in baseline_backfill:
                _write_variant(base_sdf, settings, name, args.write_method)
                _log.info("seed baseline backfill: wrote %d series to %s", args.n_series, name)
    finally:
        spark.stop()


if __name__ == "__main__":  # pragma: no cover - cluster entrypoint
    main()
