"""Tests for the analyst-view renderer.

Offline snapshot test: rendering is a pure string op, so we pin the exact SQL. If the view
definitions change intentionally, regenerate the snapshot with SF_UPDATE_SNAPSHOTS=1.
"""

from __future__ import annotations

import os
from pathlib import Path

from scale_forecasting.registry.views import VIEW_NAMES, render_create_views

SNAPSHOT = Path(__file__).parent / "snapshots" / "views.sql"


def _render_all() -> str:
    stmts = render_create_views("proj.scale_forecasting")
    return "\n\n".join(stmts[name] for name in VIEW_NAMES)


def test_all_views_rendered() -> None:
    stmts = render_create_views("proj.scale_forecasting")
    assert set(stmts) == set(VIEW_NAMES)
    assert set(VIEW_NAMES) == {"v_run_summary", "v_run_jobs", "v_model_leaderboard"}


def test_every_statement_is_replace_and_terminated() -> None:
    for stmt in render_create_views("d").values():
        assert "CREATE OR REPLACE VIEW" in stmt
        assert stmt.rstrip().endswith(";")


def test_dataset_ref_substituted_in_name_and_sources() -> None:
    stmt = render_create_views("myproj.myds")["v_run_summary"]
    # both the view name and the table it reads carry the dataset ref
    assert "`myproj.myds.v_run_summary`" in stmt
    assert "`myproj.myds.run_registry`" in stmt


def test_run_summary_unpacks_telemetry_and_derives_overhead() -> None:
    stmt = render_create_views("d")["v_run_summary"]
    # telemetry is read out of the JSON STRING, and overhead is total_wall − our runtime
    assert "JSON_VALUE(job_telemetry, '$.total_wall_s')" in stmt
    assert "overhead_seconds" in stmt
    assert "overhead_fraction" in stmt


def test_run_summary_exposes_the_shape_that_ran_and_the_decision_behind_it() -> None:
    stmt = render_create_views("d")["v_run_summary"]
    # The resolved executor shape, as scalars — "how wide, how much memory" without opening JSON.
    for column in (
        "executor_cores",
        "max_executors",
        "executor_memory",
        "executor_memory_overhead",
    ):
        assert f"AS {column}" in stmt
    # And the whole decision, per family, left as JSON: its interesting parts are nested, so
    # unpacking it into columns would pick a family for the reader.
    assert "JSON_QUERY(job_telemetry, '$.sizing') AS sizing" in stmt


def test_run_summary_keeps_one_row_per_run_after_a_forced_rerun() -> None:
    stmt = render_create_views("d")["v_run_summary"]
    # A forced re-run appends a second header under the same run_id; keep only the latest so one
    # run is always one row.
    assert "QUALIFY ROW_NUMBER() OVER (PARTITION BY run_id ORDER BY created_at DESC) = 1" in stmt


def test_leaderboard_is_per_run_model_full_fit_only() -> None:
    stmt = render_create_views("d")["v_model_leaderboard"]
    assert "GROUP BY run_id, model_type" in stmt
    # full-fit summary rows only (fold_id IS NULL), so per-fold rows don't inflate counts
    assert "fold_id IS NULL" in stmt
    # a model that failed every cell surfaces as a high no-artifact rate
    assert "no_artifact_rate" in stmt


def test_leaderboard_dedupes_cells_before_aggregating() -> None:
    stmt = render_create_views("d")["v_model_leaderboard"]
    # Writes are append-only + at-least-once, so a task retry or a --force re-run can re-append a
    # cell. Like the two views above, the leaderboard collapses to one row per cell (latest write
    # wins) BEFORE the roll-up, or a duplicated cell would double-count into mean_wape/mean_mae.
    assert "ROW_NUMBER() OVER (" in stmt
    assert "PARTITION BY run_id, ts_id, model_type, fold_id, ensemble_id" in stmt
    assert "ORDER BY created_at DESC" in stmt
    # dedup happens in a CTE that the aggregate reads from — so it precedes GROUP BY, not after it.
    assert "WITH deduped AS (" in stmt
    dedup_at = stmt.index("QUALIFY ROW_NUMBER()")
    group_at = stmt.index("GROUP BY run_id, model_type, ensemble_id")
    assert dedup_at < group_at
    assert "FROM deduped" in stmt
    # ensemble_id + fold_id are in the grain: base vs ensemble rows, and final vs per-fold rows,
    # must not collapse into each other.
    assert "ensemble_id" in stmt.split("PARTITION BY")[1].split("\n")[0]
    assert "fold_id" in stmt.split("PARTITION BY")[1].split("\n")[0]


def test_run_jobs_view_keeps_current_attempt_per_family() -> None:
    stmt = render_create_views("d")["v_run_jobs"]
    # one row per (run_id, family) = the current job; a forced re-run's higher attempt wins
    assert "QUALIFY ROW_NUMBER() OVER (" in stmt
    assert "PARTITION BY run_id, family ORDER BY attempt DESC, created_at DESC" in stmt
    # surfaces the deterministic job id + the resolved runtime/hardware for the trace
    assert "job_id" in stmt
    assert "runtime" in stmt and "hardware" in stmt and "gpu_type" in stmt
    assert "FROM `d.run_jobs`" in stmt


def test_run_jobs_view_exposes_job_timing_for_the_trace() -> None:
    stmt = render_create_views("d")["v_run_jobs"]
    # the wall-clock bracket the SDK trace() reads to place each job on a timeline
    assert "started_at" in stmt
    assert "ended_at" in stmt


def test_run_jobs_view_projects_probe_handle() -> None:
    # The probe handle (runtime coordinates for reconciliation) is projected out of the per-job
    # job_telemetry JSON so a reader can parse it without unpacking the whole column.
    stmt = render_create_views("d")["v_run_jobs"]
    assert "JSON_QUERY(job_telemetry, '$.probe_handle') AS probe_handle" in stmt


def test_views_snapshot() -> None:
    rendered = _render_all()
    if os.environ.get("SF_UPDATE_SNAPSHOTS") == "1":
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(rendered)
    assert SNAPSHOT.exists(), "snapshot missing; run with SF_UPDATE_SNAPSHOTS=1 to create"
    assert rendered == SNAPSHOT.read_text()
