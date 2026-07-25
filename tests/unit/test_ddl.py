"""Tests for the registry DDL renderer (CONTRACTS §4, DESIGN §8).

Offline snapshot test: rendering is a pure string op, so we pin the exact SQL. If the
DDL changes intentionally, regenerate the snapshot with SF_UPDATE_SNAPSHOTS=1.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from scale_forecasting.registry.ddl import TABLE_NAMES, render_create_tables

SNAPSHOT = Path(__file__).parent / "snapshots" / "ddl_iceberg.sql"

_KW = {"connection": "proj.us-central1.sf-conn", "warehouse_uri": "gs://proj-wh/warehouse"}


def _render_all() -> str:
    stmts = render_create_tables("proj.scale_forecasting", **_KW)
    return "\n\n".join(stmts[name] for name in TABLE_NAMES)


def test_all_five_tables_rendered() -> None:
    stmts = render_create_tables("proj.scale_forecasting", **_KW)
    assert set(stmts) == set(TABLE_NAMES)
    assert len(TABLE_NAMES) == 5


def test_every_statement_is_idempotent_and_terminated() -> None:
    for stmt in render_create_tables("d", **_KW).values():
        assert "CREATE TABLE IF NOT EXISTS" in stmt
        assert stmt.rstrip().endswith(";")


def test_iceberg_wrapping_present() -> None:
    stmt = render_create_tables("d", **_KW)["run_registry"]
    assert "table_format = 'ICEBERG'" in stmt
    assert "WITH CONNECTION `proj.us-central1.sf-conn`" in stmt
    assert "storage_uri = 'gs://proj-wh/warehouse/run_registry'" in stmt


def test_native_fallback_has_no_iceberg_clause() -> None:
    stmt = render_create_tables("d", iceberg=False)["run_registry"]
    assert "ICEBERG" not in stmt
    assert "WITH CONNECTION" not in stmt


def test_iceberg_requires_connection_and_warehouse() -> None:
    with pytest.raises(ValueError, match="requires both"):
        render_create_tables("d", iceberg=True)


def test_dataset_ref_is_substituted() -> None:
    stmt = render_create_tables("myproj.myds", **_KW)["forecast_predictions"]
    assert "`myproj.myds.forecast_predictions`" in stmt


def test_ddl_snapshot() -> None:
    rendered = _render_all()
    if os.environ.get("SF_UPDATE_SNAPSHOTS") == "1":
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(rendered)
    assert SNAPSHOT.exists(), "snapshot missing; run with SF_UPDATE_SNAPSHOTS=1 to create"
    assert rendered == SNAPSHOT.read_text()
