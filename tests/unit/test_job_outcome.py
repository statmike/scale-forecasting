"""Offline tests for the terminal statuses a run earns (``scale_forecasting.job_outcome``).

Two surfaces, both pure or injectable: `job_status` / `audit_cells` decide what one family job's
``run_jobs`` row should say from the cells it wrote, and `combined_run_status` folds a run's job
statuses into the header's. The BigQuery read behind the audit is injected (``read=``), exactly as
`device_audit`'s is, so nothing here touches GCP.

The case that motivates the whole module is `test_a_job_that_wrote_nothing_is_failed` and its
all-errored sibling: on 2026-09-10 a Dataproc cluster job whose six cells all raised, and a Ray job
that wrote no cells at all, both closed ``COMPLETED`` on both registry tiers.
"""

from __future__ import annotations

from typing import Any

from scale_forecasting import job_outcome

# --- job_status: the three-way fold over one family's cell tallies (pure) -------


def test_every_cell_ok_is_completed() -> None:
    assert job_outcome.job_status(cells=100, errors=0) == "COMPLETED"


def test_some_cells_failed_is_partial() -> None:
    # The surviving forecasts are real and usable, so the run is not a failure — it is incomplete.
    assert job_outcome.job_status(cells=100, errors=1) == "PARTIAL"
    assert job_outcome.job_status(cells=100, errors=99) == "PARTIAL"


def test_every_cell_failed_is_failed() -> None:
    # Smoke 18, 2026-09-10: six cells, six `_require_device` refusals, zero predictions written.
    assert job_outcome.job_status(cells=6, errors=6) == "FAILED"


def test_a_job_that_wrote_nothing_is_failed() -> None:
    # Smoke 19, 2026-09-10: the Ray workers crashed before a single cell could record itself, so
    # the aggregate is empty. Zero cells is the absence of evidence that the job ran at all, and
    # reading it as "no errors, therefore COMPLETED" is how that run closed green.
    assert job_outcome.job_status(cells=0, errors=0) == "FAILED"


def test_more_errors_than_cells_is_still_failed() -> None:
    # Defensive: the two counts come from one aggregate and cannot disagree, but the fold must not
    # fall through to COMPLETED if they ever do.
    assert job_outcome.job_status(cells=2, errors=5) == "FAILED"


# --- audit_cells: read the aggregate, judge it, file the numbers ----------------


def _reader(agg: dict[str, Any]) -> Any:
    """A stand-in for `read_cell_counts` that returns ``agg`` and records how it was called."""

    def read(run_id: str, models: list[str], **kwargs: Any) -> dict[str, Any]:
        read.seen = {"run_id": run_id, "models": models, **kwargs}  # type: ignore[attr-defined]
        return agg

    return read


def test_audit_returns_the_status_and_the_numbers_behind_it() -> None:
    status, blob = job_outcome.audit_cells(
        "rid-0", "statistical", ["theta"], read=_reader({"cells": 10, "errors": 3})
    )
    assert status == "PARTIAL"
    assert blob == {"cells": 10, "errors": 3, "status": "PARTIAL"}


def test_an_unreadable_aggregate_is_no_opinion_rather_than_a_failure() -> None:
    # BigQuery being briefly unreachable must not fail a job that ran fine; the caller then leaves
    # the row's default COMPLETED alone.
    status, blob = job_outcome.audit_cells("rid-0", "ml", ["xgboost"], read=_reader({}))
    assert status is None
    assert blob == {}


def test_missing_counts_read_as_zero_and_so_as_failed() -> None:
    # A row of NULLs is what an aggregate over no rows looks like coming back from BigQuery.
    status, _ = job_outcome.audit_cells(
        "rid-0", "deep_learning", ["neuralprophet"], read=_reader({"cells": None, "errors": None})
    )
    assert status == "FAILED"


def test_the_audit_is_scoped_to_this_family_and_this_attempt() -> None:
    # Both scopes matter and neither is optional: `models` keeps a run's other families out of the
    # tally, and `since` keeps the *previous attempt's* rows out of it — `forecast_metadata` is
    # append-only and has no attempt column.
    from datetime import UTC, datetime

    since = datetime(2026, 9, 10, tzinfo=UTC)
    read = _reader({"cells": 4, "errors": 0})
    job_outcome.audit_cells("rid-0", "ml", ["xgboost", "lightgbm"], since=since, read=read)
    assert read.seen["run_id"] == "rid-0"  # type: ignore[attr-defined]
    assert read.seen["models"] == ["xgboost", "lightgbm"]  # type: ignore[attr-defined]
    assert read.seen["since"] == since  # type: ignore[attr-defined]


def test_the_launch_window_starts_before_now() -> None:
    # The margin exists because the worker stamping `created_at` is not the process reading it.
    from datetime import UTC, datetime

    start = job_outcome.launch_window_start()
    assert start.tzinfo is not None
    assert 0 < (datetime.now(UTC) - start).total_seconds() < 300


# --- combined_run_status: the header roll-up over those statuses (pure) ---------
#
# One definition, called by both `main.run` (over the statuses its launch calls returned) and
# `airflow_tasks.finalize_run` (over the statuses it re-read from `run_jobs`). It used to be two —
# a copy here and a mirror in main that rolled up *exceptions* instead of statuses — and the two
# tiers disagreeing is precisely how a job that produced nothing closed a run green.


def test_all_families_completed_is_completed() -> None:
    statuses = {"statistical": "COMPLETED", "ml": "COMPLETED"}
    assert job_outcome.combined_run_status(statuses, ensemble_enabled=False) == "COMPLETED"


def test_all_families_failed_is_failed() -> None:
    statuses = {"statistical": "FAILED", "ml": "FAILED"}
    assert job_outcome.combined_run_status(statuses, ensemble_enabled=False) == "FAILED"


def test_mixed_families_is_partial() -> None:
    statuses = {"statistical": "COMPLETED", "ml": "FAILED"}
    assert job_outcome.combined_run_status(statuses, ensemble_enabled=False) == "PARTIAL"


def test_missing_or_running_family_counts_as_failed() -> None:
    # a family row still RUNNING (its task died before finalizing) or absent is not COMPLETED
    assert (
        job_outcome.combined_run_status(
            {"statistical": "COMPLETED", "ml": "RUNNING"}, ensemble_enabled=False
        )
        == "PARTIAL"
    )
    assert (
        job_outcome.combined_run_status({"statistical": "RUNNING"}, ensemble_enabled=False)
        == "FAILED"
    )


def test_no_base_families_is_completed() -> None:
    # degenerate: nothing to fail → COMPLETED
    assert job_outcome.combined_run_status({}, ensemble_enabled=False) == "COMPLETED"


def test_ensemble_incomplete_downgrades_completed_run() -> None:
    statuses = {"statistical": "COMPLETED", "ml": "COMPLETED", "ensemble": "FAILED"}
    assert job_outcome.combined_run_status(statuses, ensemble_enabled=True) == "FAILED"


def test_completed_ensemble_keeps_completed() -> None:
    statuses = {"statistical": "COMPLETED", "ensemble": "COMPLETED"}
    assert job_outcome.combined_run_status(statuses, ensemble_enabled=True) == "COMPLETED"


def test_ensemble_never_masks_a_family_failure() -> None:
    # a base-family PARTIAL is not upgraded by a completed ensemble; the ensemble key is excluded
    # from the base roll-up
    statuses = {"statistical": "COMPLETED", "ml": "FAILED", "ensemble": "COMPLETED"}
    assert job_outcome.combined_run_status(statuses, ensemble_enabled=True) == "PARTIAL"


def test_a_family_that_failed_without_raising_still_fails_the_run() -> None:
    # The whole point of the cell audit reaching the header: nothing raised, every launch call
    # returned, and the run is still not COMPLETED.
    statuses = {"statistical": "COMPLETED", "deep_learning": "FAILED"}
    assert job_outcome.combined_run_status(statuses, ensemble_enabled=False) == "PARTIAL"
    assert job_outcome.combined_run_status({"deep_learning": "FAILED"}, ensemble_enabled=False) == (
        "FAILED"
    )


def test_a_failed_repair_row_leaves_the_run_partial() -> None:
    # A repair only exists because cells were missing, so its own row counts like any other job:
    # `statistical` completed but its repair did not, and the run is genuinely incomplete.
    statuses = {"statistical": "COMPLETED", "statistical_repair": "FAILED"}
    assert job_outcome.combined_run_status(statuses, ensemble_enabled=False) == "PARTIAL"


def test_a_repair_never_reports_the_family_it_repaired_as_completed() -> None:
    # The other direction, and the reason the repair has a row of its own: a forty-cell repair that
    # succeeded must not close a hundred-thousand-cell family that failed.
    statuses = {"statistical": "FAILED", "statistical_repair": "COMPLETED"}
    assert job_outcome.combined_run_status(statuses, ensemble_enabled=False) == "PARTIAL"
