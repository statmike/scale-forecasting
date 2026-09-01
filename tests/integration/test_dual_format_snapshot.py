"""Dual source-format read parity + one-snapshot-per-run pinning, against live BigQuery (``@gcp``).

Two data-plane guarantees that only a live run can prove, exercised here with the **driver-side**
readers (no Spark/Ray cluster submit needed — every read runs on the driver against live BigQuery):

1. **Both source formats read identically.** The same panel is seeded into a native BigQuery table
   *and* a BigQuery-managed Apache Iceberg table, and each reader returns the *same* projected panel
   from either — proving the uniform "read through BigQuery's table interface" design (no per-format
   read fork). Parametrized over ``fmt in {"native", "iceberg"}``.

2. **One snapshot per run.** A run resolves its input snapshot **once** (on its header) and every
   reader pins to it, so all of a run's family jobs read byte-identical source data even if the
   table is written to mid-run. The test seeds N series, writes the header (which pins the
   snapshot), *then* appends M more series, and asserts every pinned reader still sees exactly N —
   while an un-pinned control read sees N+M (so the mutation demonstrably landed).

The readers under test are the driver-runnable ones, including the two forms that had no live
coverage before:

* ``ray_engine._read_driver_collect`` — the ``BigQueryReadClient`` + Arrow path, snapshot-pinned via
  ``table_modifiers.snapshot_time`` (the proto-plus datetime form).
* ``ray_engine._read_ray_data`` — the ``ray.data.read_bigquery`` path, snapshot-pinned via a
  ``FOR SYSTEM_TIME AS OF`` query (needs the ``[ray]`` extra → additionally ``@ray``-gated).
* the BigQuery-native family's ``FOR SYSTEM_TIME AS OF`` clause
  (``bigquery_sql._snapshot_clause``), exercised as a direct ``SELECT``.

The Spark connector reader (``spark_io.read_source_series``) needs a live Spark session and is
covered by ``test_spark_connect_smoke``; its option wiring is unit-tested in ``test_spark_engines``.
It is out of scope for this driver-local test.

Skipped unless ``SF_PROJECT_ID`` (+ ADC) is set (see ``tests/conftest.py``). The Iceberg variant
also needs ``SF_CONNECTION`` + ``SF_WAREHOUSE_URI``. Run manually::

    SF_PROJECT_ID=statmike-scale-forecasting \\
    SF_CONNECTION=statmike-scale-forecasting.us-central1.sf-iceberg \\
    SF_WAREHOUSE_URI=gs://statmike-scale-forecasting-warehouse/warehouse \\
    SF_DATASET_ID=scale_forecasting \\
        uv run pytest -m gcp tests/integration/test_dual_format_snapshot.py

**Self-contained data.** Each test owns its data: it seeds its own uniquely-named scratch source
table(s) via the same generator the production seed uses, and drops them (and the run header rows)
afterward — the "test owns its data" discipline of the other integration smokes. Append-only headers
can't be DELETE-d while buffered, so ``write_header`` uses a query-INSERT (deletable), and each
invocation varies ``run_name`` so its ``run_id`` is unique.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import pytest

from scale_forecasting.config import RunConfig
from scale_forecasting.engines import ray_engine
from scale_forecasting.engines.bigquery_sql import _snapshot_clause
from scale_forecasting.registry import ddl
from scale_forecasting.registry.header import (
    _SNAPSHOT_SAFETY_MARGIN_MS,
    snapshot_millis_for,
    write_header,
)
from scale_forecasting.registry.ids import make_run_id
from scale_forecasting.settings import Settings

pytestmark = pytest.mark.gcp

_N_PRE = 6  # series present at the snapshot instant
_N_POST = 4  # series appended *after* the header pins the snapshot (invisible to a pinned read)
_HISTORY = 120  # short daily history — this test reads, it doesn't fit, so cents-cheap
# The header pins snapshot = now − _SNAPSHOT_SAFETY_MARGIN_MS (2000ms). Pre-rows must be committed
# strictly before that instant to be visible at the snapshot, so we wait out the margin (+slack)
# between seeding the pre-rows and writing the header.
_MARGIN_SLACK_S = 4


@pytest.fixture(scope="module")
def settings() -> Settings:
    """Resolve live infra identity from the ``SF_*`` environment."""
    return Settings.resolve()


def _create_source_table(client: Any, settings: Settings, name: str, *, iceberg: bool) -> None:
    """Create an empty scratch source table (native or managed-Iceberg) with the source schema.

    Mirrors ``ddl._render_one`` for the source body but with a scratch name: fill ``{name}`` from
    the shared source template, then ``{d}`` with the dataset, and wrap the Iceberg OPTIONS when
    asked — so the scratch table's schema/partitioning matches a real source variant.
    """
    body = ddl._SOURCE_BODY_TEMPLATE.format(name=name).format(d=settings.dataset_ref)
    stmt = body
    if iceberg:
        if not (settings.connection and settings.warehouse_uri):
            pytest.skip("iceberg variant needs SF_CONNECTION + SF_WAREHOUSE_URI")
        opts = ddl._iceberg_options(name, settings.warehouse_uri)
        stmt = f"{body}\nWITH CONNECTION `{settings.connection}`\n{opts}"
    client.query(stmt + ";").result()


def _seed_rows(n_series: int, seed: int) -> Any:
    """Generate ``n_series`` coherent source rows via the production generator (schema-matching)."""
    from scale_forecasting.data_gen.generator import GenConfig, generate_panel
    from scale_forecasting.data_gen.seed_spark import _to_source_rows

    gen = GenConfig(history=_HISTORY, freq="D", start="2023-01-01", holidays=("US",))
    panel = generate_panel(n_series, gen, seed=seed)
    rows = _to_source_rows(panel, ("US",))
    return rows.astype({"y": "float64", "is_holiday": "bool"})


def _insert_via_staging(client: Any, settings: Settings, dest: str, rows: Any) -> None:
    """Append ``rows`` into ``dest`` (native or Iceberg) via a native staging table + INSERT…SELECT.

    Loading a dataframe straight into a managed-Iceberg table isn't uniformly supported, but DML
    ``INSERT`` is — so we land the rows in a throwaway native staging table with a load job, then
    ``INSERT INTO dest SELECT …`` (works identically for both destination formats), then drop the
    staging table. This is what makes the seed/append path format-agnostic.
    """
    from google.cloud import bigquery

    staging = f"{dest}_staging_{int(time.time() * 1000)}"
    staging_ref = settings.table_ref(staging)
    schema = [
        bigquery.SchemaField("ts_id", "STRING"),
        bigquery.SchemaField("ds", "DATE"),
        bigquery.SchemaField("y", "FLOAT"),
        bigquery.SchemaField("archetype", "STRING"),
        bigquery.SchemaField("is_holiday", "BOOL"),
    ]
    job_config = bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_TRUNCATE")
    client.load_table_from_dataframe(rows, staging_ref, job_config=job_config).result()
    try:
        cols = "ts_id, ds, y, archetype, is_holiday"
        client.query(
            f"INSERT INTO `{settings.table_ref(dest)}` ({cols}) "
            f"SELECT {cols} FROM `{staging_ref}`"
        ).result()
    finally:
        client.delete_table(staging_ref, not_found_ok=True)


@pytest.fixture(params=["native", "iceberg"])
def scratch_source(request: Any, settings: Settings) -> Iterator[tuple[str, str]]:
    """Yield ``(fmt, scratch_table_name)`` for an empty scratch source table of the given format.

    Dropped (best-effort) after the test. The name is unique per invocation so parallel/rerun
    invocations don't collide, and — for Iceberg — its storage_uri is likewise unique.
    """
    from google.cloud import bigquery

    fmt = request.param
    client = bigquery.Client(project=settings.project_id)
    name = f"df_snap_{fmt}_{int(time.time() * 1000)}"
    _create_source_table(client, settings, name, iceberg=(fmt == "iceberg"))
    try:
        yield fmt, name
    finally:
        client.delete_table(settings.table_ref(name), not_found_ok=True)


def _cfg(source_table: str, **over: Any) -> RunConfig:
    # ``data`` is merged (source_table always kept) so callers can add fields without dropping it.
    data = {"source_table": source_table, **over.pop("data", {})}
    return RunConfig(
        run_name=f"dual-format snapshot {int(time.time() * 1000)}",
        data=data,
        models=["theta"],
        **over,
    )


def _distinct_ids(frame: Any) -> int:
    return int(frame["ts_id"].nunique())


# --- 1. dual-format read parity (un-pinned) -----------------------------------


def test_dual_format_read_parity(settings: Settings, scratch_source: tuple[str, str]) -> None:
    """Both formats read to the same projected panel via the ``BigQueryReadClient`` reader."""
    from google.cloud import bigquery

    fmt, table = scratch_source
    client = bigquery.Client(project=settings.project_id)
    rows = _seed_rows(_N_PRE, seed=11)
    _insert_via_staging(client, settings, table, rows)

    cfg = _cfg(table)  # no header written → un-pinned read (snapshot_millis_for → None)
    frame = ray_engine._read_driver_collect(cfg, settings)

    # Column projection: exactly the columns a cell needs (ts_id, date, target), no more.
    assert list(frame.columns) == ["ts_id", "ds", "y"], f"{fmt}: {list(frame.columns)}"
    assert _distinct_ids(frame) == _N_PRE, fmt
    assert len(frame) == _N_PRE * _HISTORY, fmt


@pytest.mark.ray
def test_dual_format_read_parity_ray_data(
    settings: Settings, scratch_source: tuple[str, str]
) -> None:
    """The ``ray.data.read_bigquery`` reader returns the same panel from either format."""
    from google.cloud import bigquery

    fmt, table = scratch_source
    client = bigquery.Client(project=settings.project_id)
    _insert_via_staging(client, settings, table, _seed_rows(_N_PRE, seed=11))

    cfg = _cfg(table, compute={"ray_read_mode": "ray_data"})
    frame = ray_engine._read_ray_data(cfg, settings)
    assert set(frame.columns) >= {"ts_id", "ds", "y"}, f"{fmt}: {list(frame.columns)}"
    assert _distinct_ids(frame) == _N_PRE, fmt


# --- 2. one-snapshot-per-run pinning ------------------------------------------


def _seed_pre_pin_then_append_post(
    client: Any, settings: Settings, table: str, cfg: RunConfig, run_id: str
) -> None:
    """Seed N_PRE series, wait out the safety margin, pin the snapshot (header), then append N_POST.

    After this returns: the header's snapshot instant sits strictly between the pre-rows' commit
    and the post-rows' commit, so a pinned read sees only the N_PRE series; an unpinned read, all.
    """
    _insert_via_staging(client, settings, table, _seed_rows(_N_PRE, seed=21))
    # Pre-rows must be committed before snapshot = now − 2000ms, so wait out the margin (+slack).
    time.sleep(_SNAPSHOT_SAFETY_MARGIN_MS / 1000 + _MARGIN_SLACK_S)
    write_header(cfg, run_id, settings=settings)  # pins snapshot_millis on the header
    # Distinct ids so the appended series are genuinely new (seed offset avoids id overlap).
    _insert_via_staging(client, settings, table, _seed_rows(_N_POST, seed=99))


def test_snapshot_pins_driver_collect(
    settings: Settings, scratch_source: tuple[str, str]
) -> None:
    """The ``BigQueryReadClient`` reader pins to the run snapshot — post-header appends unseen."""
    from google.cloud import bigquery

    fmt, table = scratch_source
    client = bigquery.Client(project=settings.project_id)
    cfg = _cfg(table)
    run_id = make_run_id(cfg)
    try:
        _seed_pre_pin_then_append_post(client, settings, table, cfg, run_id)

        # Control: an un-pinned read (a config whose run_id has no header) sees every series.
        unpinned = ray_engine._read_driver_collect(_cfg(table), settings)
        assert _distinct_ids(unpinned) == _N_PRE + _N_POST, f"{fmt}: mutation didn't land"

        # Pinned: the reader looks up this run_id's snapshot and time-travels — only the pre-rows.
        pinned = ray_engine._read_driver_collect(cfg, settings)
        assert _distinct_ids(pinned) == _N_PRE, f"{fmt}: snapshot pin leaked post-header rows"
    finally:
        client.query(
            f"DELETE FROM `{settings.table_ref('run_registry')}` WHERE run_id=@r",
            job_config=bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("r", "STRING", run_id)]
            ),
        ).result()


def test_snapshot_pins_native_clause(
    settings: Settings, scratch_source: tuple[str, str]
) -> None:
    """The BigQuery-native family's ``FOR SYSTEM_TIME AS OF`` clause pins to the same snapshot."""
    from google.cloud import bigquery

    fmt, table = scratch_source
    client = bigquery.Client(project=settings.project_id)
    cfg = _cfg(table)
    run_id = make_run_id(cfg)
    try:
        _seed_pre_pin_then_append_post(client, settings, table, cfg, run_id)

        ref = settings.table_ref(table)
        live = next(iter(client.query(f"SELECT COUNT(DISTINCT ts_id) c FROM `{ref}`").result())).c
        assert live == _N_PRE + _N_POST, f"{fmt}: mutation didn't land"

        clause = _snapshot_clause(snapshot_millis_for(run_id, settings=settings))
        assert clause, f"{fmt}: run recorded no snapshot"
        pinned = next(
            iter(client.query(f"SELECT COUNT(DISTINCT ts_id) c FROM `{ref}`{clause}").result())
        ).c
        assert pinned == _N_PRE, f"{fmt}: native snapshot clause leaked post-header rows"
    finally:
        client.query(
            f"DELETE FROM `{settings.table_ref('run_registry')}` WHERE run_id=@r",
            job_config=bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("r", "STRING", run_id)]
            ),
        ).result()


@pytest.mark.ray
def test_snapshot_pins_ray_data(settings: Settings, scratch_source: tuple[str, str]) -> None:
    """The ``ray.data.read_bigquery`` query form pins to the run snapshot (post-header unseen)."""
    from google.cloud import bigquery

    fmt, table = scratch_source
    client = bigquery.Client(project=settings.project_id)
    cfg = _cfg(table, compute={"ray_read_mode": "ray_data"})
    run_id = make_run_id(cfg)
    try:
        _seed_pre_pin_then_append_post(client, settings, table, cfg, run_id)
        pinned = ray_engine._read_ray_data(cfg, settings)
        assert _distinct_ids(pinned) == _N_PRE, f"{fmt}: ray_data snapshot query leaked rows"
    finally:
        client.query(
            f"DELETE FROM `{settings.table_ref('run_registry')}` WHERE run_id=@r",
            job_config=bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("r", "STRING", run_id)]
            ),
        ).result()
