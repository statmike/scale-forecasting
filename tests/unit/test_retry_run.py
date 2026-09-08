"""The repair orchestrator's pure half: assembling, classifying, narrowing, and reporting.

`retry_policy` decides; `retry_run` is what feeds it real rows and prints the answer. Everything
tested here runs without a project: the registry reads are represented by the dicts they return,
and the two GCP-touching functions (`_expected_series`, `build_retry_plan`) are exercised only
through the seam that matters — that a preview writes nothing.

The load-bearing cases are the arithmetic that has no registry row behind it (a cell that never ran
is *expected minus observed*, and getting the denominator wrong invents or hides work) and the
report's honesty about models it refuses to submit.
"""

from __future__ import annotations

from typing import Any

import pytest

# `airflow_tasks` is imported here rather than inside the one test that uses it because that test
# reads its log output through `caplog`. `errors.get_logger` turns propagation off, and pytest's
# logging plugin can only attach its capture handler directly to a non-propagating logger that
# already exists when collection ends — a module first imported inside a test body logs into the
# void. Both this import and the module-level logger in `airflow_tasks` are needed for that.
from scale_forecasting import airflow_tasks, retry_run
from scale_forecasting.retry_policy import (
    RETRY_AS_IS,
    SKIP_ALREADY_DONE,
    SKIP_DETERMINISTIC,
    SKIP_NOT_FINISHED,
    CellState,
    FamilyState,
)


def _group(
    model: str,
    *,
    n: int,
    status: str | None = "ok",
    error: str | None = None,
    preds: bool = True,
    example: str = "series-a",
) -> dict[str, Any]:
    """One row shaped like `registry.reads.read_cell_groups` returns."""
    return {
        "model_type": model,
        "cell_status": status,
        "error_class": error,
        "has_predictions": preds,
        "n_cells": n,
        "example_ts_id": example,
    }


# --- the family axis, joined from two reads ------------------------------------


def test_family_states_join_the_job_row_to_the_probe_verdict() -> None:
    states = retry_run.family_states(
        [
            {"family": "ml", "status": "FAILED", "failure_reason": "CAPACITY_EXHAUSTED"},
            {"family": "native", "status": "COMPLETED", "failure_reason": None},
        ],
        {"ml": "ABANDONED_WAIT"},
    )
    assert states["ml"] == FamilyState(
        family="ml",
        status="FAILED",
        failure_reason="CAPACITY_EXHAUSTED",
        probe_verdict="ABANDONED_WAIT",
    )
    # A family the probe never escalated keeps its registry reading and simply has no verdict.
    assert states["native"].probe_verdict is None
    assert states["native"].failure_reason is None


def test_family_states_survive_a_row_with_no_family() -> None:
    """A malformed row costs one family's context, never the whole report."""
    assert retry_run.family_states([{"family": None, "status": "RUNNING"}]) == {}


def test_an_in_flight_repair_is_what_the_family_it_repairs_reads_as() -> None:
    """The guard against a second ``--retry`` re-submitting cells a first repair is already fitting.

    Cells always carry the *base* family, so if the repair row stayed keyed under its own token the
    classifier would find only the base family's stale FAILED and target those cells again.
    """
    states = retry_run.family_states(
        [
            {"family": "statistical", "status": "FAILED", "failure_reason": None},
            {"family": "statistical_repair", "status": "RUNNING", "failure_reason": None},
        ]
    )
    assert set(states) == {"statistical"}
    assert states["statistical"].status == "RUNNING" and states["statistical"].is_live


def test_the_repair_wins_regardless_of_which_row_arrives_first() -> None:
    rows = [
        {"family": "statistical_repair", "status": "RUNNING"},
        {"family": "statistical", "status": "FAILED"},
    ]
    assert retry_run.family_states(rows)["statistical"].status == "RUNNING"


def test_a_repair_verdict_is_read_under_the_token_the_probe_used() -> None:
    """The probe reports per job row, so its key is the repair token; the fold must not lose it."""
    states = retry_run.family_states(
        [{"family": "ml_repair", "status": "RUNNING"}], {"ml_repair": "ABANDONED_WAIT"}
    )
    assert states["ml"].probe_verdict == "ABANDONED_WAIT"


def test_a_finished_repair_leaves_its_cells_classifiable_again() -> None:
    """A repair that ended is not a reason to stop repairing — a second pass must be allowed."""
    states = retry_run.family_states(
        [
            {"family": "ml", "status": "FAILED"},
            {"family": "ml_repair", "status": "COMPLETED"},
        ]
    )
    assert states["ml"].status == "COMPLETED" and not states["ml"].is_live


# --- assembling states: the grouped rows, and the cells with no row at all ------


def test_a_grouped_row_becomes_one_weighted_cell_state() -> None:
    (state,) = retry_run.assemble_cell_states(
        [_group("theta", n=40, status="FAILED", error="OOM", preds=False, example="s-7")],
        family_of={"theta": "statistical"},
    )
    assert state == CellState(
        ts_id="s-7",
        model_type="theta",
        has_metadata=True,
        has_predictions=False,
        cell_status="FAILED",
        error_class="OOM",
        family="statistical",
        n_cells=40,
    )


def test_the_never_ran_cells_are_expected_minus_observed() -> None:
    """The whole point of the exercise: cells whose job died before reaching them wrote no row.

    They cannot be read, only derived, so a wrong denominator either invents work or hides it.
    """
    states = retry_run.assemble_cell_states(
        [_group("theta", n=900)],
        family_of={"theta": "statistical"},
        expected_series=1000,
    )
    never_ran = [s for s in states if not s.has_metadata]
    assert len(never_ran) == 1
    assert never_ran[0].n_cells == 100
    assert never_ran[0].ts_id == retry_run.NEVER_RAN
    assert never_ran[0].family == "statistical"


def test_a_model_the_run_planned_but_never_touched_is_entirely_never_ran() -> None:
    states = retry_run.assemble_cell_states(
        [_group("theta", n=1000)],
        family_of={"theta": "statistical", "xgboost": "ml"},
        expected_series=1000,
    )
    xgb = [s for s in states if s.model_type == "xgboost"]
    assert len(xgb) == 1 and xgb[0].n_cells == 1000 and not xgb[0].has_metadata


def test_more_observed_than_expected_contributes_no_negative_shortfall() -> None:
    """A source that grew, or a second attempt's rows, must not subtract from the worklist."""
    states = retry_run.assemble_cell_states(
        [_group("theta", n=1200)],
        family_of={"theta": "statistical"},
        expected_series=1000,
    )
    assert all(s.has_metadata for s in states)


def test_no_expected_count_means_no_synthesized_cells_at_all() -> None:
    """An unreadable denominator under-counts visibly rather than guessing."""
    states = retry_run.assemble_cell_states(
        [_group("theta", n=900)], family_of={"theta": "statistical"}, expected_series=None
    )
    assert len(states) == 1 and states[0].has_metadata


def test_a_model_missing_from_the_family_map_still_classifies() -> None:
    (state,) = retry_run.assemble_cell_states([_group("mystery", n=3)], family_of={})
    assert state.family is None


# --- the plan: classification, narrowing, and the weighted headline ------------


def _plan(**kw: Any) -> retry_run.RetryPlan:
    defaults: dict[str, Any] = {
        "header_status": "FAILED",
        "states": (),
        "families": {},
        "landed_counts": {},
        "expected_series": 1000,
        "universe_source": "source table at the run's pinned snapshot (17)",
    }
    return retry_run.assemble_retry_plan("rid-1", **{**defaults, **kw})


def test_the_headline_counts_cells_not_grouped_rows() -> None:
    """A grouped read and a cell-at-a-time read of the same run must agree on the headline."""
    states = retry_run.assemble_cell_states(
        [_group("theta", n=40, status="FAILED", error="TRANSIENT_INFRA", preds=False)],
        family_of={"theta": "statistical"},
        expected_series=100,
    )
    plan = _plan(states=states, expected_series=100)
    assert plan.counts == {RETRY_AS_IS: 100}  # 40 transient + 60 never-ran
    assert plan.n_targets == 100
    assert plan.observed_cells == 40


def test_the_plan_separates_what_it_will_submit_from_what_it_refuses() -> None:
    """``blocked`` exists so the gap is reported; a silent drop is the misleading version."""
    states = (
        CellState("s1", "theta", has_metadata=True, error_class="TRANSIENT_INFRA", n_cells=40),
        CellState("s2", "xgboost", has_metadata=True, error_class="OOM", n_cells=5),
    )
    plan = _plan(states=states, landed_counts={"theta": 99_000, "xgboost": 0})
    assert plan.models == ("xgboost",)
    assert plan.blocked == ("theta",)
    assert plan.submittable is True


def test_a_plan_with_nothing_submittable_says_so() -> None:
    states = (CellState("s1", "theta", has_metadata=True, has_predictions=True, n_cells=1000),)
    plan = _plan(states=states, expected_series=1000)
    assert plan.counts == {SKIP_ALREADY_DONE: 1000}
    assert plan.models == () and plan.blocked == () and plan.submittable is False


def test_a_still_running_family_holds_its_missing_cells_back() -> None:
    """The family axis is why a repair does not duplicate work that simply has not finished."""
    states = retry_run.assemble_cell_states(
        [], family_of={"theta": "statistical"}, expected_series=500
    )
    plan = _plan(
        states=states,
        families={"statistical": FamilyState("statistical", status="RUNNING")},
    )
    assert plan.counts == {SKIP_NOT_FINISHED: 500}
    assert plan.models == ()


def test_the_plan_rows_carry_the_classifier_input_beside_its_output() -> None:
    states = (CellState("s-9", "theta", has_metadata=True, error_class="SHORT_HISTORY", n_cells=7),)
    plan = _plan(states=states, expected_series=None)
    (row,) = plan.rows
    assert (row.verdict, row.error_class, row.n_cells, row.example_ts_id) == (
        SKIP_DETERMINISTIC,
        "SHORT_HISTORY",
        7,
        "s-9",
    )


def test_the_plan_carries_its_own_denominator_and_provenance() -> None:
    plan = _plan(expected_series=None, universe_source="run header carries no input snapshot")
    assert plan.expected_cells is None
    assert "no input snapshot" in plan.universe_source


# --- the report an operator reads ----------------------------------------------


def test_the_report_states_where_the_expected_cell_set_came_from() -> None:
    """A count of missing work is only as trustworthy as its denominator, so the text says it."""
    text = retry_run.format_retry_plan(_plan(universe_source="pinned snapshot (17)"))
    assert "universe: expected=1000 series/model" in text
    assert "pinned snapshot (17)" in text


def test_the_report_names_the_models_it_will_not_submit_and_why() -> None:
    states = (
        CellState("s1", "theta", has_metadata=True, error_class="TRANSIENT_INFRA", n_cells=40),
    )
    text = retry_run.format_retry_plan(_plan(states=states, landed_counts={"theta": 99_000}))
    assert "would submit: (nothing)" in text
    assert "NOT submittable: theta" in text
    assert "new run_id" in text


def test_the_report_lists_every_verdict_that_occurred_not_only_the_retryable_ones() -> None:
    states = (
        CellState("s1", "theta", has_metadata=True, error_class="TRANSIENT_INFRA", n_cells=40),
        CellState("s2", "theta", has_metadata=True, error_class="SHORT_HISTORY", n_cells=3),
        CellState("s3", "theta", has_metadata=True, has_predictions=True, n_cells=900),
    )
    text = retry_run.format_retry_plan(_plan(states=states, expected_series=None))
    for verdict in (RETRY_AS_IS, SKIP_DETERMINISTIC, SKIP_ALREADY_DONE):
        assert verdict in text


# --- the audit blob ------------------------------------------------------------


def test_the_audit_blob_records_the_decision_and_the_evidence_it_rests_on() -> None:
    from datetime import UTC, datetime

    states = (
        CellState("s1", "theta", has_metadata=True, error_class="TRANSIENT_INFRA", n_cells=40),
    )
    plan = _plan(states=states, expected_series=None)
    blob = retry_run._retry_audit(
        plan, actor="someone", at=datetime(2026, 1, 2, tzinfo=UTC), reason="driver died"
    )
    assert blob["retried_by"] == "someone" and blob["reason"] == "driver died"
    assert blob["retried_at"].startswith("2026-01-02")
    assert blob["counts"] == {RETRY_AS_IS: 40} and blob["n_targets"] == 40
    assert blob["models"] == ["theta"] and blob["blocked"] == []
    assert "universe_source" in blob


# --- preview submits nothing ---------------------------------------------------


def test_a_preview_reads_the_plan_and_launches_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default is the whole safety story: no launcher, no registry write, on any path."""
    import scale_forecasting.job_launch as job_launch
    import scale_forecasting.registry.jobs as jobs

    plan = _plan(
        states=(CellState("s1", "theta", has_metadata=True, error_class="OOM", n_cells=5),)
    )
    monkeypatch.setattr(retry_run, "build_retry_plan", lambda cfg, settings=None: plan)
    monkeypatch.setattr(
        job_launch, "submit_retry", lambda *a, **k: pytest.fail("preview submitted a job")
    )
    monkeypatch.setattr(jobs, "update_job", lambda *a, **k: pytest.fail("preview wrote a row"))

    report = retry_run.retry_run(object())  # type: ignore[arg-type]
    assert report.executed is False and report.plan is plan and report.outcome is None


def test_a_confirmed_call_with_nothing_submittable_still_launches_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not a failure — the correct answer for a run whose every gap is a SKIP_*."""
    import scale_forecasting.job_launch as job_launch

    plan = _plan(
        states=(CellState("s1", "theta", has_metadata=True, has_predictions=True, n_cells=5),)
    )
    monkeypatch.setattr(retry_run, "build_retry_plan", lambda cfg, settings=None: plan)
    monkeypatch.setattr(
        job_launch, "submit_retry", lambda *a, **k: pytest.fail("launched an empty repair")
    )
    report = retry_run.retry_run(object(), confirm=True)  # type: ignore[arg-type]
    assert report.executed is False and report.plan.models == ()


# --- one decision, two call sites ----------------------------------------------


def test_the_dag_node_and_the_cli_verb_repair_from_the_same_worklist(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``--retry --force`` and the Airflow retry node are one implementation, not two.

    A repair can be launched by an operator at a terminal or, unattended, by the DAG node
    (`airflow_tasks.retry_families`). If those two ever classified differently, the decision table
    an operator reviewed would stop being evidence for what Composer does at 3am — and the whole
    point of the preview is that it *is* that evidence. So both are driven here off one set of
    fixtures, through the real `retry_run.retry_run`, and asked to submit the same thing.
    """
    import json

    import scale_forecasting.identity as identity
    import scale_forecasting.job_launch as job_launch
    import scale_forecasting.registry.jobs as jobs
    from scale_forecasting import main
    from scale_forecasting.dag import RunDag
    from scale_forecasting.job_launch import RetryOutcome
    from scale_forecasting.settings import Settings

    path = tmp_path / "run.json"
    path.write_text(
        json.dumps(
            {
                "run_name": "two call sites",
                "data": {"source_table": "source_series_native", "horizon": 7},
                "models": ["theta", "xgboost"],
            }
        )
    )

    # theta's cells already landed, so nothing asks for them; xgboost's OOM cells are the repair.
    plan = _plan(
        states=(
            CellState("s1", "theta", has_metadata=True, has_predictions=True, n_cells=1000),
            CellState("s2", "xgboost", has_metadata=True, error_class="OOM", n_cells=40),
        ),
        landed_counts={"theta": 1000, "xgboost": 0},
    )
    assert plan.models == ("xgboost",)

    submitted: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []

    def _record(cfg: Any, retry_dag: RunDag, run_id: str, *a: Any, **k: Any) -> RetryOutcome:
        models = tuple(m for job in retry_dag.jobs for m in job.models)
        submitted.append((run_id, tuple(retry_dag.families), models))
        return RetryOutcome(families=tuple(retry_dag.families))

    monkeypatch.setattr(retry_run, "build_retry_plan", lambda cfg, settings=None: plan)
    monkeypatch.setattr(job_launch, "submit_retry", _record)
    monkeypatch.setattr(jobs, "read_run_jobs", lambda *a, **k: [])
    monkeypatch.setattr(identity, "resolve_principal", lambda settings: "tester")
    monkeypatch.setattr(Settings, "resolve", classmethod(lambda cls: object()))

    main._main(["--config", str(path), "--retry", "--force"])
    cli_output = capsys.readouterr().out
    with caplog.at_level("INFO"):
        dag_summary = airflow_tasks.retry_families(str(path))

    # The same run, the same repair family, the same models — from the same plan.
    assert len(submitted) == 2
    assert submitted[0] == submitted[1] == (plan.run_id, ("ml_repair",), ("xgboost",))
    # And both surface the table they acted on, so the audit trail reads the same either way.
    table = retry_run.format_retry_plan(plan)
    assert table in cli_output
    assert table in caplog.text
    assert "ml_repair" in dag_summary
