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
    assert set(VIEW_NAMES) == {
        "v_run_summary",
        "v_run_jobs",
        "v_model_leaderboard",
        "v_backtest_coverage",
        "v_model_leaderboard_comparable",
    }


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
    assert (
        "QUALIFY ROW_NUMBER() OVER (PARTITION BY run_id ORDER BY created_at DESC NULLS LAST) = 1"
        in stmt
    )


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


def test_run_jobs_view_surfaces_why_a_job_failed_and_what_it_tried() -> None:
    # `failure_reason` is a column so "show me every job that ran out of regions" is a WHERE
    # clause; the ledger stays JSON because its interesting part is a per-attempt list.
    stmt = render_create_views("d")["v_run_jobs"]
    assert "\n  failure_reason,\n" in stmt
    assert "JSON_QUERY(job_telemetry, '$.capacity') AS capacity" in stmt


def test_run_jobs_view_says_whether_the_accelerator_did_anything() -> None:
    # The verdict is a scalar column so "which of my GPU jobs wasted the card" is a WHERE clause
    # against a word, not a JSON walk; the blob beside it keeps the counts and the peak byte
    # figure that word was decided from. A CPU family has neither, and reads NULL.
    stmt = render_create_views("d")["v_run_jobs"]
    assert "JSON_VALUE(job_telemetry, '$.device_use.verdict') AS device_verdict" in stmt
    assert "JSON_QUERY(job_telemetry, '$.device_use') AS device_use" in stmt


def test_run_summary_view_projects_the_shared_clusters_capacity_ledger() -> None:
    # The run-level half of the same story: a shared cluster is provisioned before any job row
    # exists, so its walk is recorded on the header instead — see `shared_capacity_path`.
    stmt = render_create_views("d")["v_run_summary"]
    assert "JSON_QUERY(job_telemetry, '$.capacity') AS capacity" in stmt


# --- the two cohort views: is the ranking comparable at all? --------------------


def test_backtest_coverage_is_a_long_cohort_table_over_full_fit_rows() -> None:
    stmt = render_create_views("d")["v_backtest_coverage"]
    assert "FROM `d.forecast_metadata`" in stmt
    # Same full-fit restriction the leaderboard uses: one row per cell, not one per fold, or the
    # cohort counts would be fold counts wearing a series count's name.
    assert "fold_id IS NULL" in stmt
    # Long-format grain: the achieved-fold histogram has no fixed width, so it is rows, not columns.
    assert "GROUP BY run_id, model_type, ensemble_id, backtest_status, n_folds_achieved" in stmt
    assert "COUNT(*) AS n_series" in stmt
    # Each cohort's share of that model's own panel — a window over the group, not over the run.
    assert "SUM(COUNT(*)) OVER (PARTITION BY run_id, model_type, ensemble_id)" in stmt


def test_backtest_coverage_dedupes_cells_before_counting_them() -> None:
    stmt = render_create_views("d")["v_backtest_coverage"]
    # A re-appended cell would be counted twice as two series. Dedupe first, like every other view.
    assert "WITH deduped AS (" in stmt
    assert "PARTITION BY run_id, ts_id, model_type, fold_id, ensemble_id" in stmt
    assert stmt.index("QUALIFY ROW_NUMBER()") < stmt.index("GROUP BY run_id, model_type")


def test_comparable_leaderboard_restricts_to_the_holdout_fold() -> None:
    stmt = render_create_views("d")["v_model_leaderboard_comparable"]
    # The whole point of the view: every model scored on the one fold every series that backtested
    # at all achieved. Derived from the rows (`make_folds` drops from the oldest end, keeping the
    # survivor's original fold_id) because a view has no config to read the holdout out of.
    assert "QUALIFY fold_id = MAX(fold_id) OVER (PARTITION BY run_id)" in stmt
    assert "ANY_VALUE(h.fold_id) AS holdout_fold_id" in stmt


def test_comparable_leaderboard_pools_the_error_and_reports_its_panel() -> None:
    stmt = render_create_views("d")["v_model_leaderboard_comparable"]
    # Pooled, not averaged: one WAPE of the whole panel, where a near-zero series cannot dominate
    # the way it does in a mean of per-series WAPEs.
    assert "SAFE_DIVIDE(SUM(ABS(h.y_true - h.yhat)), SUM(ABS(h.y_true))) AS pooled_wape" in stmt
    # And the panel it was pooled over, so a reader can see whether two rows answered the same
    # question. Equal n_series across models is the evidence; unequal n_series is the finding.
    assert "COUNT(DISTINCT h.ts_id) AS n_series" in stmt
    # Unscorable rows are excluded rather than half-counted: a NULL yhat would drop out of the
    # numerator while its y_true stayed in the denominator, quietly flattering the model.
    assert "WHERE h.y_true IS NOT NULL AND h.yhat IS NOT NULL" in stmt


def test_comparable_leaderboard_reads_oof_and_keeps_ensembles_distinct() -> None:
    stmt = render_create_views("d")["v_model_leaderboard_comparable"]
    # The ranking comes from backtest_oof, not forecast_metadata: only the OOF table holds
    # per-fold truth, and forecast_metadata's rolled-up rows cannot be restricted to a fold after
    # the fact. (forecast_metadata is still read, but only by the `refit` CTE, which contributes a
    # label and never a number.)
    assert "FROM `d.backtest_oof`" in stmt
    assert "GROUP BY h.run_id, h.model_type, h.ensemble_id" in stmt
    # Two ensemble configs under one run_id stay apart, here as everywhere else.
    assert "ensemble_id" in stmt.split("PARTITION BY")[1].split("\n")[0]
    # Dedupe precedes the holdout restriction, which precedes the aggregate.
    assert stmt.index("WITH deduped AS (") < stmt.index("holdout AS (")
    assert stmt.index("holdout AS (") < stmt.index("GROUP BY h.run_id, h.model_type, h.ensemble_id")


def test_both_leaderboards_say_whether_the_ranking_was_scored_one_way() -> None:
    # A frozen run whose models could not all freeze produces a ranking that mixes two questions.
    # `refit_modes` is how a reader sees that without the view splitting one model into two rows:
    # "recondition" is a clean cohort, "recondition,unsupported" is a warning.
    for name in ("v_model_leaderboard", "v_model_leaderboard_comparable"):
        stmt = render_create_views("d")[name]
        assert "STRING_AGG(DISTINCT backtest_refit ORDER BY backtest_refit) AS refit_modes" in stmt


def test_the_comparable_leaderboard_joins_refit_modes_without_dropping_base_models() -> None:
    stmt = render_create_views("d")["v_model_leaderboard_comparable"]
    # ensemble_id is NULL on every base model, and NULL never equals NULL — an equality join would
    # silently hand back a leaderboard of ensembles only. COALESCE is what keeps the base models in.
    assert "COALESCE(h.ensemble_id, '') = COALESCE(r.ensemble_id, '')" in stmt
    # LEFT, so a model whose metadata row never landed still ranks with a NULL label rather than
    # disappearing: the ranking is the product here, and the label is the annotation.
    assert "LEFT JOIN refit AS r" in stmt
    # The label is joined in at the cell grain the metadata table actually has.
    assert "  FROM `d.forecast_metadata`\n  WHERE fold_id IS NULL" in stmt


def test_views_snapshot() -> None:
    rendered = _render_all()
    if os.environ.get("SF_UPDATE_SNAPSHOTS") == "1":
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(rendered)
    assert SNAPSHOT.exists(), "snapshot missing; run with SF_UPDATE_SNAPSHOTS=1 to create"
    assert rendered == SNAPSHOT.read_text()
