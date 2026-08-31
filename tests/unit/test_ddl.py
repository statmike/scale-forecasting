"""Tests for the registry DDL renderer.

Offline snapshot test: rendering is a pure string op, so we pin the exact SQL. If the
DDL changes intentionally, regenerate the snapshot with SF_UPDATE_SNAPSHOTS=1.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from scale_forecasting.registry.ddl import (
    REGISTRY_TABLE_NAMES,
    SOURCE_TABLE_ICEBERG,
    SOURCE_TABLE_NAMES,
    SOURCE_TABLE_NATIVE,
    TABLE_NAMES,
    additive_columns,
    render_create_tables,
    render_deployment_ddl,
    render_drop_tables,
    render_migrations,
)

SNAPSHOT = Path(__file__).parent / "snapshots" / "ddl_deployment.sql"
DROP_SNAPSHOT = Path(__file__).parent / "snapshots" / "ddl_drop.sql"

_KW = {"connection": "proj.us-central1.sf-conn", "warehouse_uri": "gs://proj-wh/warehouse"}

# The five run-collection tables are always native; the source table ships in both formats.
# Spelled out literally rather than re-exported, so a rename or a table added to the wrong family
# fails here instead of being rubber-stamped by the constant it is supposed to be checking.
_REGISTRY_TABLES = (
    "run_registry",
    "run_jobs",
    "forecast_metadata",
    "forecast_predictions",
    "backtest_oof",
)


def _render_all() -> str:
    stmts = render_deployment_ddl("proj.scale_forecasting", **_KW)
    return "\n\n".join(stmts[name] for name in TABLE_NAMES)


def test_all_tables_rendered() -> None:
    stmts = render_deployment_ddl("proj.scale_forecasting", **_KW)
    assert set(stmts) == set(TABLE_NAMES)
    assert len(TABLE_NAMES) == 7
    # five registry + two source variants
    assert set(TABLE_NAMES) == {*_REGISTRY_TABLES, SOURCE_TABLE_ICEBERG, SOURCE_TABLE_NATIVE}


def test_the_two_families_partition_all_tables() -> None:
    # The split is what lets a deployment address registry and source separately; if the two
    # families ever overlap or leave a table unclaimed, a subset-scoped drop/migrate silently
    # misses (or doubles up on) it.
    assert REGISTRY_TABLE_NAMES == _REGISTRY_TABLES
    assert SOURCE_TABLE_NAMES == (SOURCE_TABLE_ICEBERG, SOURCE_TABLE_NATIVE)
    assert not set(REGISTRY_TABLE_NAMES) & set(SOURCE_TABLE_NAMES)
    assert set(REGISTRY_TABLE_NAMES) | set(SOURCE_TABLE_NAMES) == set(TABLE_NAMES)


def test_render_deployment_ddl_can_split_the_two_datasets() -> None:
    # SF_REGISTRY_DATASET_ID: registry in one dataset, source panel in another. The renderer is the
    # single place that knows which family goes where.
    stmts = render_deployment_ddl("proj.reg", source_dataset="proj.src", **_KW)
    for name in REGISTRY_TABLE_NAMES:
        assert f"`proj.reg.{name}`" in stmts[name]
        assert "proj.src" not in stmts[name]
    for name in SOURCE_TABLE_NAMES:
        assert f"`proj.src.{name}`" in stmts[name]
        assert "proj.reg." not in stmts[name]


def test_render_deployment_ddl_defaults_source_to_the_registry_dataset() -> None:
    # Zero behaviour change for a deployment that never sets SF_REGISTRY_DATASET_ID.
    assert render_deployment_ddl("proj.ds", **_KW) == render_deployment_ddl(
        "proj.ds", source_dataset="proj.ds", **_KW
    )


def test_subset_renderers_restrict_to_the_named_family() -> None:
    assert set(render_drop_tables("d", tables=REGISTRY_TABLE_NAMES)) == set(REGISTRY_TABLE_NAMES)
    assert set(render_migrations("d", tables=SOURCE_TABLE_NAMES)) <= set(SOURCE_TABLE_NAMES)
    assert set(render_create_tables("d", iceberg=False, tables=SOURCE_TABLE_NAMES)) == set(
        SOURCE_TABLE_NAMES
    )


def test_every_statement_is_idempotent_and_terminated() -> None:
    for stmt in render_deployment_ddl("d", **_KW).values():
        assert "CREATE TABLE IF NOT EXISTS" in stmt
        assert stmt.rstrip().endswith(";")


def test_registry_tables_are_native_in_deployment() -> None:
    # The run-collection tables are always native BigQuery — no Iceberg wrapping, so they
    # can carry the native JSON column type and be reseeded with WRITE_TRUNCATE.
    stmts = render_deployment_ddl("d", **_KW)
    for name in _REGISTRY_TABLES:
        assert "ICEBERG" not in stmts[name], name
        assert "WITH CONNECTION" not in stmts[name], name


def test_registry_json_columns_use_native_json_type() -> None:
    stmts = render_deployment_ddl("d", **_KW)
    assert "raw_config        JSON NOT NULL" in stmts["run_registry"]
    assert "job_telemetry     JSON" in stmts["run_registry"]
    assert "best_params    JSON" in stmts["forecast_metadata"]
    assert "quantiles     JSON" in stmts["forecast_predictions"]


def test_run_jobs_is_native_with_job_identity_columns() -> None:
    stmts = render_deployment_ddl("d", **_KW)
    job = stmts["run_jobs"]
    # native (no Iceberg wrapping) so it carries the native JSON telemetry column
    assert "ICEBERG" not in job
    assert "WITH CONNECTION" not in job
    # the identity + resolved-compute columns the trace and re-run policy key on
    assert "job_id           STRING NOT NULL" in job
    assert "run_id           STRING NOT NULL" in job
    assert "family           STRING NOT NULL" in job
    assert "attempt          INT64 NOT NULL" in job
    assert "job_telemetry    JSON" in job
    assert "CLUSTER BY run_id, family" in job


def test_source_iceberg_variant_is_iceberg_wrapped() -> None:
    stmt = render_deployment_ddl("d", **_KW)[SOURCE_TABLE_ICEBERG]
    assert "table_format = 'ICEBERG'" in stmt
    assert "WITH CONNECTION `proj.us-central1.sf-conn`" in stmt
    assert f"storage_uri = 'gs://proj-wh/warehouse/{SOURCE_TABLE_ICEBERG}'" in stmt


def test_source_native_variant_has_no_iceberg_clause() -> None:
    stmt = render_deployment_ddl("d", **_KW)[SOURCE_TABLE_NATIVE]
    assert "ICEBERG" not in stmt
    assert "WITH CONNECTION" not in stmt


def test_both_source_variants_share_the_same_columns() -> None:
    # Same panel seeds both, so schemas match column-for-column (only the wrapping differs).
    stmts = render_deployment_ddl("d", **_KW)

    def _cols(stmt: str) -> str:
        return stmt[stmt.index("(") : stmt.index("\n)")]

    assert _cols(stmts[SOURCE_TABLE_ICEBERG]) == _cols(stmts[SOURCE_TABLE_NATIVE])


def test_deployment_dataset_ref_is_substituted() -> None:
    stmt = render_deployment_ddl("myproj.myds", **_KW)["forecast_predictions"]
    assert "`myproj.myds.forecast_predictions`" in stmt


# --- the low-level render_create_tables primitive (per-table iceberg flag) ------


def test_native_fallback_has_no_iceberg_clause() -> None:
    stmt = render_create_tables("d", iceberg=False)["run_registry"]
    assert "ICEBERG" not in stmt
    assert "WITH CONNECTION" not in stmt


def test_iceberg_requires_connection_and_warehouse() -> None:
    with pytest.raises(ValueError, match="requires both"):
        render_create_tables("d", iceberg=True)


# --- drop tables (reset path) --------------------------------------------------


def test_render_drop_tables_covers_all_tables() -> None:
    drops = render_drop_tables("proj.ds")
    assert set(drops) == set(TABLE_NAMES)
    for name, stmt in drops.items():
        assert stmt == f"DROP TABLE IF EXISTS `proj.ds.{name}`;"


def test_drop_snapshot() -> None:
    rendered = "\n".join(render_drop_tables("proj.scale_forecasting")[n] for n in TABLE_NAMES)
    if os.environ.get("SF_UPDATE_SNAPSHOTS") == "1":
        DROP_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        DROP_SNAPSHOT.write_text(rendered)
    assert DROP_SNAPSHOT.exists(), "snapshot missing; run with SF_UPDATE_SNAPSHOTS=1 to create"
    assert rendered == DROP_SNAPSHOT.read_text()


# --- additive schema evolution (migrations) ------------------------------------


def test_additive_columns_excludes_not_null_keys() -> None:
    cols = dict(additive_columns("run_registry"))
    # NOT NULL columns can't be added to a populated table, so they're not migration candidates.
    assert "run_id" not in cols
    assert "created_at" not in cols
    assert "raw_config" not in cols  # NOT NULL, and now JSON
    # the nullable columns are, with their types intact.
    assert cols["status"] == "STRING"
    assert cols["n_series"] == "INT64"
    assert cols["job_telemetry"] == "JSON"
    # snapshot_millis is nullable, so it auto-migrates onto an older run_registry.
    assert cols["snapshot_millis"] == "INT64"


def test_additive_columns_include_trace_timing_columns() -> None:
    # The trace timing columns are nullable, so they auto-migrate onto older tables.
    jobs = dict(additive_columns("run_jobs"))
    assert jobs["started_at"] == "TIMESTAMP"
    assert jobs["ended_at"] == "TIMESTAMP"
    meta = dict(additive_columns("forecast_metadata"))
    assert meta["worker_id"] == "STRING"
    assert meta["cell_started_at"] == "TIMESTAMP"
    assert meta["cell_ended_at"] == "TIMESTAMP"


def test_the_measurement_columns_auto_migrate_onto_an_existing_forecast_metadata() -> None:
    # All five are nullable, so a deployment that predates them picks them up from
    # `render_migrations` without a hand-written ALTER — which is the whole point of deriving
    # the migration from the same table body the CREATE renders.
    meta = dict(additive_columns("forecast_metadata"))
    assert meta["cpu_seconds"] == "FLOAT64"
    assert meta["process_rss_bytes"] == "INT64"
    assert meta["peak_gpu_bytes"] == "INT64"
    assert meta["intraop_threads"] == "INT64"
    assert meta["n_obs"] == "INT64"
    stmt = render_migrations("proj.ds")["forecast_metadata"]
    assert "ADD COLUMN IF NOT EXISTS cpu_seconds FLOAT64" in stmt


def test_additive_columns_parse_array_type() -> None:
    # a comma-free composite type (ARRAY<STRING>) survives the comma-split of the column block.
    cols = dict(additive_columns("run_registry"))
    assert cols["bq_models"] == "ARRAY<STRING>"


def test_render_migrations_adds_job_telemetry_idempotently() -> None:
    stmt = render_migrations("proj.ds")["run_registry"]
    assert stmt.startswith("ALTER TABLE `proj.ds.run_registry`")
    assert "ADD COLUMN IF NOT EXISTS job_telemetry JSON" in stmt
    assert stmt.rstrip().endswith(";")


def test_render_migrations_covers_every_table_with_nullable_columns() -> None:
    migrations = render_migrations("d")
    # every table has at least one nullable column, so each gets a migration statement.
    assert set(migrations) == set(TABLE_NAMES)


def test_ddl_snapshot() -> None:
    rendered = _render_all()
    if os.environ.get("SF_UPDATE_SNAPSHOTS") == "1":
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(rendered)
    assert SNAPSHOT.exists(), "snapshot missing; run with SF_UPDATE_SNAPSHOTS=1 to create"
    assert rendered == SNAPSHOT.read_text()
