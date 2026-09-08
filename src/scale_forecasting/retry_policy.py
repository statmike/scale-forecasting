"""What a failed run's next command should be — the pure repair classifier.

A run of a hundred thousand cells rarely fails all at once. It fails in patches: one family hit a
capacity wall, one model OOM'd on the fat tail of the panel, three thousand series were too short
for the fold geometry. Re-running the whole thing costs the same as the first run and throws away
everything that worked. This module is the alternative — it looks at what landed and decides, per
cell, whether re-asking the question could produce a different answer.

Everything here is **pure**: dataclasses in, verdicts out, no client and no imports beyond the
standard library. The registry read that assembles `CellState` values lives in `registry.reads`,
the submit side in `job_launch`, and the operator-facing report in `main` / `sdk`. Keeping the
decision separate from both is what lets the CLI verb and the Airflow task reach the same worklist
from the same rows — a property that is asserted rather than assumed.

**The one invariant, and why it outranks everything else.** A cell that already has prediction rows
is never retried, whatever its status says and whatever went wrong around it. The reason is that a
retry is an *append*: it cannot replace a forecast, only add a second one beside it, and the
serving views then resolve the pair by write time (`registry.rows.cell_dedup_key`). That is the
right rule for a cell that never landed and the wrong one for a cell that landed badly — silently
swapping a reviewed forecast for a fresh fit is not a repair, it is a different run. So a
wrong-but-present forecast is explicitly **out of scope for retry**; changing it means changing the
config, which changes the ``run_id``, which is the honest way to say the answer changed.
`build_worklist` enforces this as a postcondition and raises rather than returning a worklist that
violates it.

**Verdicts.** Three ask for work, four refuse it, and the refusals are the interesting half:

- ``RETRY_AS_IS`` — the same submission could plausibly succeed (a transient infrastructure fault,
  or a cell that never ran at all because its job died before reaching it).
- ``RETRY_WITH_MORE_MEMORY`` — it will fail the same way unless the fleet is resized first.
- ``RETRY_LATER`` — nothing is wrong with the work; the capacity to do it was not there.
- ``SKIP_ALREADY_DONE`` — the invariant above.
- ``SKIP_DETERMINISTIC`` — the same input through the same code fails the same way. A series too
  short for the fold geometry does not grow by being asked twice.
- ``CONFIG_REPAIRABLE`` — fixable, but not by this machinery: the config named something that does
  not exist. Submits nothing, and says what to edit.
- ``UNKNOWN`` — the default, and a real answer rather than a gap. Doing nothing on a failure nobody
  has classified is safer than guessing, and a rising ``UNKNOWN`` share is the signal that
  `worker.ERROR_CLASSES` needs a row.

**What this classifier does not see yet.** Its inputs are the per-cell registry rows alone. The
family job's own status and the runtime probe's verdict are two more inputs that can overturn a
per-cell reading — a whole family that never started produces cells indistinguishable from cells
whose job ran and skipped them. Those rows widen `classify_cell` in a later step; the vocabulary
above is already the full one, so widening adds inputs rather than moving answers.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from .errors import RegistryError

RETRY_AS_IS = "RETRY_AS_IS"
RETRY_WITH_MORE_MEMORY = "RETRY_WITH_MORE_MEMORY"
RETRY_LATER = "RETRY_LATER"
SKIP_ALREADY_DONE = "SKIP_ALREADY_DONE"
SKIP_DETERMINISTIC = "SKIP_DETERMINISTIC"
CONFIG_REPAIRABLE = "CONFIG_REPAIRABLE"
UNKNOWN = "UNKNOWN"

#: Every verdict, in the order a report reads best: the work first, then the refusals.
VERDICTS: tuple[str, ...] = (
    RETRY_AS_IS,
    RETRY_WITH_MORE_MEMORY,
    RETRY_LATER,
    SKIP_ALREADY_DONE,
    SKIP_DETERMINISTIC,
    CONFIG_REPAIRABLE,
    UNKNOWN,
)

#: The verdicts that put a cell on the worklist. Everything else submits nothing.
RETRY_VERDICTS: frozenset[str] = frozenset({RETRY_AS_IS, RETRY_WITH_MORE_MEMORY, RETRY_LATER})

#: `worker.ERROR_CLASSES` token → verdict. The mapping is deliberately explicit rather than
#: derived: an error class is a statement about what went wrong, and a verdict is a statement about
#: what to do next, and the second does not follow from the first without a judgement call.
#:
#: ``MODEL_ERROR`` is the row worth arguing about. A solver that did not converge on this series
#: will not converge on a re-run of the same series with the same hyperparameters — the fit is
#: deterministic in everything the retry could change, so retrying it burns a fleet to reproduce a
#: failure. It is filed as deterministic and surfaces in the report, where the fix is a config
#: change (different model, different search space) rather than a resubmission.
ERROR_CLASS_VERDICTS: dict[str, str] = {
    "OOM": RETRY_WITH_MORE_MEMORY,
    "CAPACITY": RETRY_LATER,
    "TRANSIENT_INFRA": RETRY_AS_IS,
    "SHORT_HISTORY": SKIP_DETERMINISTIC,
    "BAD_DATA": SKIP_DETERMINISTIC,
    "MODEL_ERROR": SKIP_DETERMINISTIC,
    "CONFIG_REPAIRABLE": CONFIG_REPAIRABLE,
    "UNKNOWN": UNKNOWN,
}


@dataclass(frozen=True)
class CellState:
    """What the registry knows about one ``(ts_id, model_type)`` cell of a run.

    Assembled by the caller, not read here — a synthesized cell for a series the run was *supposed*
    to cover but never wrote a row for is just ``CellState(ts_id, model_type)``, all flags false.
    That case is the point of the whole exercise and it has no registry row to be read from, so the
    caller derives the expected cell set from the run's snapshot-pinned source rather than from a
    registry query. ``has_metadata`` false with ``has_predictions`` false means "this cell never
    happened".
    """

    ts_id: str
    model_type: str
    has_metadata: bool = False
    has_predictions: bool = False
    cell_status: str | None = None
    error_class: str | None = None


@dataclass(frozen=True)
class Worklist:
    """The classification of a whole run, split into what to submit and what to explain.

    ``by_verdict`` is every cell filed under its answer — the whole picture, and the thing the
    report prints. ``targets`` is the subset a retry would re-ask, and ``models`` is the narrowed
    model subset those targets imply: the grain v1 actually submits at (``FamilyJob.models`` →
    ``--models``), derived once here rather than re-derived by each caller. Empty ``targets`` on a
    non-empty ``by_verdict`` is the common and correct outcome of a healthy run.
    """

    by_verdict: dict[str, tuple[CellState, ...]] = field(default_factory=dict)

    @property
    def targets(self) -> tuple[CellState, ...]:
        """Every cell a retry would resubmit, in `VERDICTS` order (so a report reads stably)."""
        return tuple(c for v in VERDICTS if v in RETRY_VERDICTS for c in self.by_verdict.get(v, ()))

    @property
    def models(self) -> tuple[str, ...]:
        """The distinct model types on the worklist, sorted — what a retry narrows ``--models`` to.

        A model whose every cell landed must not appear here, or the repair resubmits a hundred
        thousand finished cells to fix forty.
        """
        return tuple(sorted({c.model_type for c in self.targets}))

    @property
    def counts(self) -> dict[str, int]:
        """Verdict → cell count, over every verdict that occurred. The report's headline."""
        return {v: len(cells) for v, cells in self.by_verdict.items() if cells}


def classify_cell(state: CellState) -> str:
    """What to do about one cell — one of `VERDICTS` (pure, total).

    Total by construction: every path returns, and the fallthrough is ``UNKNOWN`` rather than an
    exception, because a repair report that raises on one unrecognised cell tells an operator
    nothing about the other ninety-nine thousand.

    The order of the checks is the argument:

    1. **Predictions present → ``SKIP_ALREADY_DONE``**, before anything else is consulted. This is
       the invariant, and putting it first is what makes it one — a status column that disagrees
       with the rows on disk does not get to overrule the rows.
    2. **No metadata row → ``RETRY_AS_IS``.** The cell never ran. Nothing is known to be wrong with
       it; the job that should have covered it did not get there.
    3. **Metadata says ``ok``, but no predictions.** A contradiction: the worker recorded success
       and wrote no forecast. Something is wrong with the *run*, not with this cell, and a retry
       that re-fits into the same hole is not a diagnosis — ``UNKNOWN``, no action.
    4. **Otherwise it is an error cell** — dispatch on `ERROR_CLASS_VERDICTS`. An error row with no
       ``error_class`` at all predates the column or came from a writer that does not fill it, and
       is ``UNKNOWN`` for the same reason as an unrecognised token.
    """
    if state.has_predictions:
        return SKIP_ALREADY_DONE
    if not state.has_metadata:
        return RETRY_AS_IS
    if state.cell_status == "ok":
        return UNKNOWN
    if state.error_class is None:
        return UNKNOWN
    return ERROR_CLASS_VERDICTS.get(state.error_class, UNKNOWN)


def build_worklist(states: Iterable[CellState]) -> Worklist:
    """Classify every cell and split it into a `Worklist` (pure).

    Raises `RegistryError` if the no-overlap postcondition is broken — a targeted cell that already
    has predictions. That can only happen if `classify_cell` is edited into disagreeing with itself,
    which is exactly the edit worth failing loudly on: the quiet version of that bug appends a
    second forecast beside a reviewed one and lets the write clock decide which the run means.
    """
    grouped: dict[str, list[CellState]] = {}
    for state in states:
        grouped.setdefault(classify_cell(state), []).append(state)
    worklist = Worklist(by_verdict={v: tuple(cells) for v, cells in grouped.items()})

    overlapping = [c for c in worklist.targets if c.has_predictions]
    if overlapping:
        raise RegistryError(
            f"retry worklist would re-run {len(overlapping)} cell(s) that already have "
            f"predictions, e.g. {overlapping[0].ts_id}/{overlapping[0].model_type}; "
            "a landed forecast is replaced by a new run, never by a retry"
        )
    return worklist
