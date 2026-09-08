"""Offline tests for the pure repair classifier (``scale_forecasting.retry_policy``).

The module has no I/O, so everything it does is testable here and nothing about it needs GCP. Two
things carry the weight: the **no-overlap invariant** (a cell with predictions is never retried,
whatever else is true of it) and **totality** (every reachable input produces a verdict from the
fixed vocabulary, never an exception and never a token nobody has seen).
"""

from __future__ import annotations

import itertools

import pytest

from scale_forecasting.errors import RegistryError
from scale_forecasting.retry_policy import (
    CONFIG_REPAIRABLE,
    ERROR_CLASS_VERDICTS,
    RETRY_AS_IS,
    RETRY_LATER,
    RETRY_VERDICTS,
    RETRY_WITH_MORE_MEMORY,
    SKIP_ALREADY_DONE,
    SKIP_DETERMINISTIC,
    UNKNOWN,
    VERDICTS,
    CellState,
    Worklist,
    build_worklist,
    classify_cell,
)
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
    rp.classify_cell = lambda state: RETRY_AS_IS  # type: ignore[assignment]
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
