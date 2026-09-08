"""Offline tests for the pure repair classifier (``scale_forecasting.retry_policy``).

The module has no I/O, so everything it does is testable here and nothing about it needs GCP. Two
things carry the weight: the **no-overlap invariant** (a cell with predictions is never retried,
whatever else is true of it) and **totality** (every reachable input produces a verdict from the
fixed vocabulary, never an exception and never a token nobody has seen).
"""

from __future__ import annotations

import ast
import itertools
import subprocess
import sys
from pathlib import Path

import pytest

from scale_forecasting import retry_policy
from scale_forecasting.capacity import AWAITING_CAPACITY
from scale_forecasting.capacity import CAPACITY_EXHAUSTED as _CAPACITY_EXHAUSTED
from scale_forecasting.errors import RegistryError
from scale_forecasting.probes import vocabulary as probe_vocabulary
from scale_forecasting.probes.vocabulary import (
    VERDICT_ABANDONED_WAIT,
    VERDICT_LOST,
    VERDICT_RUNNING,
    VERDICT_STALE_REGISTRY,
    VERDICT_TRUST_REGISTRY,
)
from scale_forecasting.retry_policy import (
    CAPACITY_EXHAUSTED,
    CONFIG_REPAIRABLE,
    ERROR_CLASS_VERDICTS,
    LIVE_JOB_STATUSES,
    PROBE_FINISHED_VERDICTS,
    PROBE_RUNNING_VERDICT,
    RETRY_AS_IS,
    RETRY_LATER,
    RETRY_VERDICTS,
    RETRY_WITH_MORE_MEMORY,
    SKIP_ALREADY_DONE,
    SKIP_DETERMINISTIC,
    SKIP_NOT_FINISHED,
    UNKNOWN,
    VERDICTS,
    CellState,
    FamilyState,
    RetryTargets,
    Worklist,
    build_worklist,
    classify_cell,
    narrow_to_submittable,
)
from scale_forecasting.sdk import _TERMINAL_STATUSES
from scale_forecasting.worker import ERROR_CLASSES

# Every value each input can take, including the ones a healthy run never produces. The
# cross-product below is the "property test" the plan asks for: it is small enough to enumerate
# exhaustively, so there is no reason to sample it.
_ERROR_CLASSES: tuple[str | None, ...] = (None, *(t for t, _, _ in ERROR_CLASSES), "UNKNOWN")
_STATUSES: tuple[str | None, ...] = (None, "ok", "error")
_FLAGS = (False, True)


def _every_cell_state() -> list[CellState]:
    return [
        CellState(
            ts_id="s1",
            model_type="theta",
            has_metadata=meta,
            has_predictions=preds,
            cell_status=status,
            error_class=err,
        )
        for meta, preds, status, err in itertools.product(_FLAGS, _FLAGS, _STATUSES, _ERROR_CLASSES)
    ]


# --- the invariant -------------------------------------------------------------


def test_a_cell_with_predictions_is_never_retried_whatever_else_is_true_of_it() -> None:
    # The whole cross-product, filtered to cells that already landed a forecast. A retry is an
    # append: it cannot replace the row, only add a second one beside it, and the write clock then
    # decides which the run means. That is a new run's job, not a repair's.
    landed = [c for c in _every_cell_state() if c.has_predictions]
    assert landed, "the cross-product should contain cells with predictions"
    assert {classify_cell(c) for c in landed} == {SKIP_ALREADY_DONE}


def test_the_worklist_refuses_to_return_a_target_that_already_landed() -> None:
    # Enforced as a postcondition, not only as a table row -- because the quiet version of this bug
    # is silent. `classify_cell` is monkeypatched into disagreeing with itself, which is exactly
    # the edit the postcondition exists to catch.
    import scale_forecasting.retry_policy as rp

    original = rp.classify_cell
    rp.classify_cell = lambda state, family=None: RETRY_AS_IS  # type: ignore[assignment]
    try:
        with pytest.raises(RegistryError, match="already have predictions"):
            build_worklist([CellState("s1", "theta", has_metadata=True, has_predictions=True)])
    finally:
        rp.classify_cell = original  # type: ignore[assignment]


# --- totality ------------------------------------------------------------------


def test_every_reachable_cell_state_gets_a_verdict_from_the_fixed_vocabulary() -> None:
    # A classifier that raises on one unrecognised cell tells an operator nothing about the other
    # ninety-nine thousand, so the fallthrough is a verdict rather than an exception.
    assert {classify_cell(c) for c in _every_cell_state()} <= set(VERDICTS)


def test_every_error_class_the_worker_can_emit_has_a_verdict() -> None:
    # The two vocabularies are written in different files and drift silently: a new row in
    # `worker.ERROR_CLASSES` would classify as UNKNOWN and quietly stop being repaired.
    assert {t for t, _, _ in ERROR_CLASSES} <= set(ERROR_CLASS_VERDICTS)


def test_the_error_class_table_only_uses_verdicts_that_exist() -> None:
    assert set(ERROR_CLASS_VERDICTS.values()) <= set(VERDICTS)


# --- the individual rows -------------------------------------------------------


def _errored(error_class: str | None) -> CellState:
    return CellState("s1", "theta", has_metadata=True, cell_status="error", error_class=error_class)


@pytest.mark.parametrize(
    ("error_class", "expected"),
    [
        ("OOM", RETRY_WITH_MORE_MEMORY),
        ("CAPACITY", RETRY_LATER),
        ("TRANSIENT_INFRA", RETRY_AS_IS),
        ("SHORT_HISTORY", SKIP_DETERMINISTIC),
        ("BAD_DATA", SKIP_DETERMINISTIC),
        ("MODEL_ERROR", SKIP_DETERMINISTIC),
        ("CONFIG_REPAIRABLE", CONFIG_REPAIRABLE),
        ("UNKNOWN", UNKNOWN),
        (None, UNKNOWN),
        ("A_TOKEN_FROM_THE_FUTURE", UNKNOWN),
    ],
)
def test_an_error_cell_is_classified_by_what_went_wrong(error_class: str, expected: str) -> None:
    assert classify_cell(_errored(error_class)) == expected


def test_a_cell_that_never_ran_is_the_case_retry_exists_for() -> None:
    # No metadata row and no predictions: the job that should have covered this cell did not get
    # there. Nothing is known to be wrong with the cell itself.
    assert classify_cell(CellState("s1", "theta")) == RETRY_AS_IS


def test_a_cell_that_claims_success_but_wrote_no_forecast_is_not_repaired_quietly() -> None:
    # The worker recorded ok and there are no prediction rows -- a contradiction about the run, not
    # a fact about the cell. Re-fitting into the same hole would hide it; UNKNOWN surfaces it.
    ok_but_empty = CellState("s1", "theta", has_metadata=True, cell_status="ok")
    assert classify_cell(ok_but_empty) == UNKNOWN


def test_a_short_series_is_not_asked_twice() -> None:
    # The dominant real error class. It is deterministic in everything a retry can change: the
    # series does not grow by being re-submitted, so the honest verdict refuses the work.
    assert classify_cell(_errored("SHORT_HISTORY")) not in RETRY_VERDICTS


def test_a_broken_config_is_never_resubmitted() -> None:
    # Repairable, but not by this machinery -- the fix is an edit, and an edit moves the run_id.
    assert CONFIG_REPAIRABLE not in RETRY_VERDICTS


# --- the worklist --------------------------------------------------------------


def test_a_clean_run_produces_nothing_to_do() -> None:
    states = [
        CellState(f"s{i}", m, has_metadata=True, has_predictions=True, cell_status="ok")
        for i in range(5)
        for m in ("theta", "xgboost")
    ]
    worklist = build_worklist(states)
    assert worklist.targets == ()
    assert worklist.models == ()
    assert worklist.counts == {SKIP_ALREADY_DONE: 10}


def test_an_empty_run_produces_an_empty_worklist() -> None:
    worklist = build_worklist([])
    assert worklist.targets == () and worklist.counts == {}


def test_the_worklist_narrows_to_the_models_that_actually_need_resubmitting() -> None:
    # This is the grain v1 submits at: `FamilyJob.models` -> `--models`. A model whose every cell
    # landed must not appear, or the repair re-runs a hundred thousand finished cells to fix forty.
    states = [
        CellState("s1", "theta", has_metadata=True, has_predictions=True, cell_status="ok"),
        CellState("s2", "theta"),  # never ran
        CellState("s3", "xgboost", has_metadata=True, has_predictions=True, cell_status="ok"),
        CellState("s4", "prophet", has_metadata=True, cell_status="error", error_class="OOM"),
    ]
    worklist = build_worklist(states)
    assert worklist.models == ("prophet", "theta")
    assert worklist.counts == {RETRY_AS_IS: 1, RETRY_WITH_MORE_MEMORY: 1, SKIP_ALREADY_DONE: 2}


def test_the_worklist_files_every_cell_under_exactly_one_verdict() -> None:
    states = _every_cell_state()
    worklist = build_worklist(states)
    filed = sum(len(cells) for cells in worklist.by_verdict.values())
    assert filed == len(states)


def test_targets_read_in_a_stable_order_whatever_order_the_rows_arrive_in() -> None:
    # A report an operator diffs between two invocations must not reshuffle itself; the registry
    # read makes no ordering promise.
    a = CellState("s1", "theta")  # RETRY_AS_IS
    b = CellState("s2", "theta", has_metadata=True, cell_status="error", error_class="OOM")
    forward = build_worklist([a, b]).targets
    backward = build_worklist([b, a]).targets
    assert forward == backward == (a, b)


def test_a_hand_built_worklist_has_no_targets_by_default() -> None:
    # The dataclass default has to be the safe one: an empty worklist submits nothing.
    assert Worklist().targets == () and Worklist().models == ()


# --- the family axis -----------------------------------------------------------


def _never_ran(family: str | None = "statistical") -> CellState:
    return CellState("s1", "theta", family=family)


def test_a_cell_under_a_running_family_is_pending_not_missing() -> None:
    # The single most expensive mistake this classifier could make: an operator runs the report
    # mid-flight, sees a hundred thousand "never ran" cells, and submits a duplicate of the run
    # that is at that moment producing them.
    running = FamilyState("statistical", status="RUNNING")
    assert classify_cell(_never_ran(), running) == SKIP_NOT_FINISHED


def test_a_family_still_waiting_for_hardware_has_not_finished_either() -> None:
    # AWAITING_CAPACITY is not a failure -- the job has not started. Same answer as RUNNING.
    waiting = FamilyState("deep_learning", status=AWAITING_CAPACITY)
    assert classify_cell(_never_ran("deep_learning"), waiting) == SKIP_NOT_FINISHED


def test_the_probe_overrules_a_registry_row_that_never_caught_up() -> None:
    # A killed job's last word was RUNNING and it will stay RUNNING forever. Without the probe
    # taking precedence, repair would be frozen on every run that ended by being killed.
    stale = FamilyState("ml", status="RUNNING", probe_verdict=VERDICT_STALE_REGISTRY)
    assert stale.is_live is False
    assert classify_cell(_never_ran("ml"), stale) == RETRY_AS_IS


def test_the_probe_also_overrules_a_registry_row_that_gave_up_too_early() -> None:
    # The mirror case: the registry says the job is gone, the runtime says it is still there.
    live = FamilyState("ml", status="FAILED", probe_verdict=VERDICT_RUNNING)
    assert live.is_live is True
    assert classify_cell(_never_ran("ml"), live) == SKIP_NOT_FINISHED


@pytest.mark.parametrize("verdict", sorted(PROBE_FINISHED_VERDICTS))
def test_every_finished_probe_verdict_lets_the_cell_speak_for_itself(verdict: str) -> None:
    assert FamilyState("ml", status="RUNNING", probe_verdict=verdict).is_live is False


def test_a_capacity_wall_turns_a_resubmission_into_a_later_one() -> None:
    # The cell's own row cannot see this: it never ran, so it reads RETRY_AS_IS, and resubmitting
    # it immediately walks straight back into the wall the family just hit.
    walled = FamilyState("deep_learning", status="FAILED", failure_reason=CAPACITY_EXHAUSTED)
    assert classify_cell(_never_ran("deep_learning"), walled) == RETRY_LATER


def test_an_abandoned_capacity_walk_is_read_the_same_way() -> None:
    abandoned = FamilyState("deep_learning", probe_verdict=VERDICT_ABANDONED_WAIT)
    assert classify_cell(_never_ran("deep_learning"), abandoned) == RETRY_LATER


def test_the_capacity_downgrade_never_turns_a_refusal_into_work() -> None:
    # It only ever moves a retry to a later retry. A short series under a capacity-bound family is
    # still a short series.
    walled = FamilyState("statistical", status="FAILED", failure_reason=CAPACITY_EXHAUSTED)
    short = CellState(
        "s1",
        "theta",
        has_metadata=True,
        cell_status="error",
        error_class="SHORT_HISTORY",
        family="statistical",
    )
    assert classify_cell(short, walled) == SKIP_DETERMINISTIC


def test_a_cancelled_family_is_not_quietly_resurrected() -> None:
    # Someone stopped that work on purpose. Offering to undo a human decision without saying so is
    # the kind of surprise a repair tool cannot afford; UNKNOWN reports it and submits nothing.
    cancelled = FamilyState("ml", status="CANCELLED")
    assert classify_cell(_never_ran("ml"), cancelled) == UNKNOWN


def test_a_landed_cell_outranks_every_family_reading() -> None:
    # The invariant is checked before the family is consulted, in both directions: a running family
    # cannot make a landed cell pending, and a dead one cannot make it retryable.
    landed = CellState("s1", "theta", has_metadata=True, has_predictions=True, family="statistical")
    for family in (
        FamilyState("statistical", status="RUNNING"),
        FamilyState("statistical", status="FAILED", failure_reason=CAPACITY_EXHAUSTED),
        FamilyState("statistical", status="CANCELLED", probe_verdict=VERDICT_LOST),
    ):
        assert classify_cell(landed, family) == SKIP_ALREADY_DONE


def test_a_family_with_nothing_known_about_it_leaves_the_cell_reading_alone() -> None:
    assert classify_cell(_never_ran(), FamilyState("statistical")) == RETRY_AS_IS
    assert classify_cell(_never_ran(), None) == RETRY_AS_IS


def test_the_worklist_joins_each_cell_to_its_own_family() -> None:
    states = [
        CellState("s1", "theta", family="statistical"),
        CellState("s1", "neuralprophet", family="deep_learning"),
    ]
    families = {
        "statistical": FamilyState("statistical", status="COMPLETED"),
        "deep_learning": FamilyState("deep_learning", status="RUNNING"),
    }
    worklist = build_worklist(states, families)
    assert worklist.models == ("theta",)
    assert worklist.counts == {RETRY_AS_IS: 1, SKIP_NOT_FINISHED: 1}


def test_a_cell_whose_family_is_missing_from_the_map_is_still_classified() -> None:
    # Dropping it would silently shrink the report; the per-cell rules still have an answer.
    worklist = build_worklist([_never_ran("statistical")], {"ml": FamilyState("ml")})
    assert worklist.counts == {RETRY_AS_IS: 1}


def test_the_family_axis_stays_total_across_its_own_cross_product() -> None:
    statuses = (None, "RUNNING", AWAITING_CAPACITY, "COMPLETED", "FAILED", "PARTIAL", "CANCELLED")
    reasons = (None, CAPACITY_EXHAUSTED, "SOMETHING_ELSE")
    verdicts = (None, *sorted(PROBE_FINISHED_VERDICTS), VERDICT_RUNNING, "UNKNOWN")
    seen = {
        classify_cell(cell, FamilyState("statistical", s, r, v))
        for cell in _every_cell_state()
        for s, r, v in itertools.product(statuses, reasons, verdicts)
    }
    assert seen <= set(VERDICTS)


# --- the vocabularies this module restates rather than imports -----------------


def test_retry_policy_imports_neither_the_probes_package_nor_a_gcp_extra() -> None:
    # It is imported by the CLI, the SDK and an Airflow task. Dragging `probes` (or anything that
    # pulls a GCP extra) into those paths would reverse a decision made deliberately elsewhere --
    # which is why the tokens below are copied rather than imported, and why this test exists to
    # make the copies safe.
    tree = ast.parse(Path(retry_policy.__file__).read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
        elif isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
    # `from .errors import ...` parses with an empty module head plus level=1; the names that
    # matter here are the absolute ones.
    banned = {"probes", "google", "pandas", "numpy", "pyarrow", "ray", "pyspark", "torch"}
    assert not imported & banned, sorted(imported & banned)


def test_nothing_retry_policy_imports_drags_the_probes_package_in_behind_it() -> None:
    """The line above reads this module's own import statements; this one reads the whole
    transitive closure, in a fresh interpreter where ``sys.modules`` starts empty. Only the second
    can catch a `probes` import that arrives two hops away through something innocuous."""
    # Baselined against a bare interpreter, because the `google` namespace package is already in
    # ``sys.modules`` before line one runs (its distributions install a ``.pth``). Only what the
    # import *adds* is this module's doing.
    script = (
        "import sys\n"
        "before = set(sys.modules)\n"
        "import scale_forecasting.retry_policy\n"
        "added = set(sys.modules) - before\n"
        "heavy = ('probes', 'google', 'pandas', 'numpy', 'pyarrow', 'ray', 'pyspark', 'torch')\n"
        "pulled = sorted(m for m in added if m.split('.')[0] in heavy or 'probes' in m)\n"
        "assert not pulled, f'retry_policy pulled: {pulled}'\n"
        "print('ok')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_the_live_job_statuses_are_words_the_registry_actually_writes() -> None:
    assert AWAITING_CAPACITY in LIVE_JOB_STATUSES
    assert LIVE_JOB_STATUSES.isdisjoint(_TERMINAL_STATUSES)


def test_the_capacity_token_still_matches_the_one_capacity_publishes() -> None:
    assert CAPACITY_EXHAUSTED == _CAPACITY_EXHAUSTED


def test_every_probe_verdict_is_accounted_for_on_exactly_one_side() -> None:
    # A new probe verdict must be filed as finished, as live, or as deliberately-neither. Left out
    # of all three, it would fall through to the registry status and quietly stop overruling it.
    # Read off the module by reflection so adding a `VERDICT_*` constant reaches this test without
    # anyone remembering to widen a list here.
    published = {
        value
        for name, value in vars(probe_vocabulary).items()
        if name.startswith("VERDICT_") and isinstance(value, str)
    }
    filed = PROBE_FINISHED_VERDICTS | {PROBE_RUNNING_VERDICT, VERDICT_TRUST_REGISTRY, "UNKNOWN"}
    assert published == filed
    assert PROBE_RUNNING_VERDICT not in PROBE_FINISHED_VERDICTS


def test_trust_registry_is_deliberately_on_neither_side() -> None:
    # It is the probe declining to have an opinion -- the registry status is authoritative, which
    # is exactly what happens when neither set matches.
    trusted = FamilyState("ml", status="RUNNING", probe_verdict=VERDICT_TRUST_REGISTRY)
    assert trusted.is_live is True
    assert FamilyState("ml", status="COMPLETED", probe_verdict=VERDICT_TRUST_REGISTRY).is_live is (
        False
    )


# --- the submission grain: what v1 can safely re-ask ---------------------------


def _worklist_over(*models: str) -> Worklist:
    return build_worklist([CellState(f"s{i}", m) for i, m in enumerate(models)])


def test_a_model_that_produced_nothing_is_the_case_v1_repairs() -> None:
    targets = narrow_to_submittable(_worklist_over("theta", "xgboost"), {})
    assert targets.models == ("theta", "xgboost") and targets.blocked == ()


def test_a_single_landed_forecast_blocks_the_whole_model() -> None:
    # Stricter than the per-cell invariant, and it has to be: v1 submits `--models theta`, which
    # re-runs theta across the run's whole series universe. Repairing forty cells that way would
    # append a second forecast beside every one that already landed.
    targets = narrow_to_submittable(_worklist_over("theta", "xgboost"), {"theta": 1})
    assert targets.models == ("xgboost",)
    assert targets.blocked == ("theta",)


def test_a_model_with_no_landed_rows_at_all_is_not_blocked_by_a_zero() -> None:
    # A model whose every cell failed writes metadata but zero predictions, so it shows up in the
    # count map with 0 -- which is exactly the model repair exists for.
    targets = narrow_to_submittable(_worklist_over("theta"), {"theta": 0, "xgboost": 900})
    assert targets.models == ("theta",) and targets.blocked == ()


def test_a_landed_model_nobody_asked_about_changes_nothing() -> None:
    targets = narrow_to_submittable(_worklist_over("theta"), {"sarimax": 500})
    assert targets.models == ("theta",) and targets.blocked == ()


def test_what_the_grain_cannot_reach_is_reported_rather_than_dropped() -> None:
    # The silent version of this is the bad one: an operator told "40 cells need repair" who then
    # watches nothing happen has been misled about a system behaving correctly.
    targets = narrow_to_submittable(_worklist_over("theta", "xgboost"), {"theta": 1, "xgboost": 1})
    assert targets.models == ()
    assert targets.blocked == ("theta", "xgboost")


def test_a_clean_run_asks_for_no_models_and_blocks_none() -> None:
    clean = build_worklist([CellState("s1", "theta", has_metadata=True, has_predictions=True)])
    targets = narrow_to_submittable(clean, {"theta": 400})
    assert targets == RetryTargets()


def test_nothing_is_submittable_that_the_worklist_did_not_ask_for() -> None:
    # The narrowing only ever subtracts. A model absent from the worklist cannot enter the
    # submission by way of the count map.
    worklist = _worklist_over("theta")
    targets = narrow_to_submittable(worklist, {"xgboost": 0})
    assert set(targets.models) <= set(worklist.models)
