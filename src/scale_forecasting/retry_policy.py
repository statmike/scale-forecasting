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

**Verdicts.** Three ask for work, five refuse it, and the refusals are the interesting half:

- ``RETRY_AS_IS`` — the same submission could plausibly succeed (a transient infrastructure fault,
  or a cell that never ran at all because its job died before reaching it).
- ``RETRY_WITH_MORE_MEMORY`` — it will fail the same way unless the fleet is resized first.
- ``RETRY_LATER`` — nothing is wrong with the work; the capacity to do it was not there.
- ``SKIP_ALREADY_DONE`` — the invariant above.
- ``SKIP_DETERMINISTIC`` — the same input through the same code fails the same way. A series too
  short for the fold geometry does not grow by being asked twice.
- ``SKIP_NOT_FINISHED`` — the family that owned this cell is still running. The cell is not
  missing, it is pending, and an impatient operator who cannot tell the two apart produces a
  duplicate submission.
- ``CONFIG_REPAIRABLE`` — fixable, but not by this machinery: the config named something that does
  not exist. Submits nothing, and says what to edit.
- ``UNKNOWN`` — the default, and a real answer rather than a gap. Doing nothing on a failure nobody
  has classified is safer than guessing, and a rising ``UNKNOWN`` share is the signal that
  `worker.ERROR_CLASSES` needs a row.

**Two grains, because a cell cannot see its own job.** `CellState` is what the registry knows about
one ``(ts_id, model_type)``; `FamilyState` is what it knows about the job that owned it — status,
failure reason, and the runtime probe's verdict where one has run. The second exists because a cell
with no metadata row looks identical whether its job never started, is running right now, died
halfway, or ran to completion and skipped it, and only the job says which. The family reading is
optional throughout: a report built without it is a weaker report, not a broken one.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from .errors import RegistryError

RETRY_AS_IS = "RETRY_AS_IS"
RETRY_WITH_MORE_MEMORY = "RETRY_WITH_MORE_MEMORY"
RETRY_LATER = "RETRY_LATER"
SKIP_ALREADY_DONE = "SKIP_ALREADY_DONE"
SKIP_DETERMINISTIC = "SKIP_DETERMINISTIC"
SKIP_NOT_FINISHED = "SKIP_NOT_FINISHED"
CONFIG_REPAIRABLE = "CONFIG_REPAIRABLE"
UNKNOWN = "UNKNOWN"

#: Every verdict, in the order a report reads best: the work first, then the refusals.
VERDICTS: tuple[str, ...] = (
    RETRY_AS_IS,
    RETRY_WITH_MORE_MEMORY,
    RETRY_LATER,
    SKIP_ALREADY_DONE,
    SKIP_DETERMINISTIC,
    SKIP_NOT_FINISHED,
    CONFIG_REPAIRABLE,
    UNKNOWN,
)

#: The verdicts that put a cell on the worklist. Everything else submits nothing.
RETRY_VERDICTS: frozenset[str] = frozenset({RETRY_AS_IS, RETRY_WITH_MORE_MEMORY, RETRY_LATER})

# --- the family-level vocabularies, restated rather than imported --------------
# `retry_policy` stays free of `probes` and of anything that pulls a GCP extra: it is imported by
# the CLI, by the SDK and by an Airflow task, and dragging the probe package into those import
# paths would reverse a decision made deliberately elsewhere. The cost is two token lists written
# twice — paid for by `tests/unit/test_retry_policy.py`, which imports both sides and asserts they
# still agree. Copying the words is cheap; copying them silently is not.

#: Registry job statuses that mean the family has not finished. `capacity.AWAITING_CAPACITY` is one
#: of them: a job waiting for a GPU has not failed, it has not started. So is
#: `registry.rows.EMITTED` — a staged command whose job id has been handed out but which this
#: process never ran; treating it as finished would let a repair submit over a command somebody
#: pasted thirty seconds ago. The probe overrules both, which is how an emitted command that was
#: never actually run stops holding a repair back.
LIVE_JOB_STATUSES: frozenset[str] = frozenset({"RUNNING", "AWAITING_CAPACITY", "EMITTED"})

#: `probes.vocabulary` verdicts that settle a family as *finished*, whatever the registry says. The
#: probe reads the runtime directly, so when the two disagree the probe is the one that looked.
PROBE_FINISHED_VERDICTS: frozenset[str] = frozenset(
    {"STALE_REGISTRY", "LIKELY_COMPLETED", "LOST", "ABANDONED_WAIT"}
)

#: The probe verdict that settles a family as still live.
PROBE_RUNNING_VERDICT = "RUNNING_CONFIRMED"

#: `capacity.CAPACITY_EXHAUSTED` — the family ran out of room rather than out of correctness.
CAPACITY_EXHAUSTED = "CAPACITY_EXHAUSTED"

#: The probe's reading of the same thing: an `AWAITING_CAPACITY` walk nobody is walking any more.
PROBE_ABANDONED_WAIT = "ABANDONED_WAIT"

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

    ``family`` is stamped by the caller too, from `dag.group_models_by_family` — the mapping lives
    in the config, and reading a config here would put a heavyweight import in the CLI, SDK and
    Airflow paths for one lookup. It is what joins a cell to its `FamilyState`; ``None`` means the
    family reading is simply unavailable and the per-cell rules stand alone.

    **``n_cells`` is how one value stands for many.** A hundred-thousand-cell run cannot be read
    back a row at a time to be classified, and it does not need to be: every field above is an
    input to `classify_cell`, so two cells that agree on all of them get the same verdict by
    construction. The registry read therefore groups by exactly this tuple and returns one
    ``CellState`` per distinct combination with ``n_cells`` set to the size of its group — lossless
    with respect to the classifier, and a report of a few dozen rows instead of a million. ``ts_id``
    on a grouped value is one arbitrary member of the group, kept because an operator reading
    "3,140 cells skipped as SHORT_HISTORY" wants an example to go look at. A caller that really
    does hold single cells just leaves ``n_cells`` at 1 and ``ts_id`` means what it says.
    """

    ts_id: str
    model_type: str
    has_metadata: bool = False
    has_predictions: bool = False
    cell_status: str | None = None
    error_class: str | None = None
    family: str | None = None
    n_cells: int = 1


@dataclass(frozen=True)
class FamilyState:
    """What is known about the *job* that owned a cell — the context a per-cell row cannot carry.

    A cell with no metadata row looks identical whether its job never started, is running right
    now, died halfway, or ran to completion and skipped it. Only the job says which, and the
    difference decides between "resubmit", "wait", and "something is wrong upstream of this cell".

    ``status`` is the ``run_jobs`` status, ``failure_reason`` its first token (see `capacity`), and
    ``probe_verdict`` the reconciled reading from `probes.reconcile` when a probe has run. All three
    are optional: a repair report built without probing is a weaker report, not a broken one.
    """

    family: str
    status: str | None = None
    failure_reason: str | None = None
    probe_verdict: str | None = None

    @property
    def is_live(self) -> bool:
        """Has this family *not* finished? The probe wins over the registry when they disagree.

        That precedence is the whole reason the probe exists: the registry records what a job said
        about itself last, and a job that was killed says nothing at all, so a stale ``RUNNING``
        row would otherwise freeze repair on a run that ended hours ago.
        """
        if self.probe_verdict in PROBE_FINISHED_VERDICTS:
            return False
        if self.probe_verdict == PROBE_RUNNING_VERDICT:
            return True
        return self.status in LIVE_JOB_STATUSES

    @property
    def is_capacity_bound(self) -> bool:
        """Did this family stop because there was no room, rather than because of the work?"""
        return (
            self.failure_reason == CAPACITY_EXHAUSTED or self.probe_verdict == PROBE_ABANDONED_WAIT
        )


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
        """Verdict → **cell** count, over every verdict that occurred. The report's headline.

        Cells, not rows: each value is weighted by `CellState.n_cells`, so a grouped read and a
        cell-at-a-time read of the same run produce the same headline. On ungrouped states every
        weight is 1 and this is just the row count.
        """
        return {v: sum(c.n_cells for c in cells) for v, cells in self.by_verdict.items() if cells}

    @property
    def n_targets(self) -> int:
        """How many cells a retry would resubmit — `targets` weighted the same way as `counts`."""
        return sum(c.n_cells for c in self.targets)


@dataclass(frozen=True)
class RetryTargets:
    """The worklist's model wish-list, split by what v1's submission grain can actually honour.

    ``models`` is what a retry submits — ``FamilyJob.models`` narrowed, which becomes ``--models``.
    ``blocked`` is the difference between that and what the worklist asked for, and it exists so the
    gap is *reported* rather than silently dropped: an operator who is told "40 cells need repair"
    and then watches nothing happen has been misled about a system that was working correctly.
    """

    models: tuple[str, ...] = ()
    blocked: tuple[str, ...] = ()


def narrow_to_submittable(worklist: Worklist, landed_counts: Mapping[str, int]) -> RetryTargets:
    """Split the worklist's models into what v1 can safely resubmit and what it cannot (pure).

    **Why a model can be blocked while its cells are legitimately retryable.** The no-overlap
    invariant is per *cell*, but v1 submits per *model*: `FamilyJob.models` narrowed to ``--models``
    re-runs that model across the run's whole series universe. So a model that landed ninety-nine
    thousand forecasts and lost forty cannot be repaired at this grain at all — resubmitting it to
    fix the forty would append a second forecast beside each of the ninety-nine thousand, which is
    exactly what `build_worklist` refuses to let happen one cell at a time. The rule is therefore
    stricter than the invariant: **any** landed prediction blocks the whole model.

    That leaves v1 useful for the failure it was built for — a family or a model that produced
    *nothing*, because its job died, never started, or never found room — and honest about the one
    it cannot reach yet. The per-cell manifest that would fix the rest is step two, and the
    ``blocked`` list is the measurement that says how much it is worth.

    ``landed_counts`` is ``{model_type: prediction row count}`` — what
    `registry.reads.read_prediction_counts` returns from a single GROUP BY rather than a per-cell
    scan. It is read from the table rather than inferred from the worklist on purpose: the
    worklist's view of what landed is only as complete as the cell set the caller assembled, and
    the table's is not an inference at all.
    """
    wanted = worklist.models
    blocked = tuple(m for m in wanted if landed_counts.get(m, 0) > 0)
    return RetryTargets(models=tuple(m for m in wanted if m not in set(blocked)), blocked=blocked)


def classify_cell(state: CellState, family: FamilyState | None = None) -> str:
    """What to do about one cell — one of `VERDICTS` (pure, total).

    Total by construction: every path returns, and the fallthrough is ``UNKNOWN`` rather than an
    exception, because a repair report that raises on one unrecognised cell tells an operator
    nothing about the other ninety-nine thousand.

    The order of the checks is the argument:

    1. **Predictions present → ``SKIP_ALREADY_DONE``**, before anything else is consulted. This is
       the invariant, and putting it first is what makes it one — neither a status column nor a
       probe verdict gets to overrule the rows on disk.
    2. **The family has not finished → ``SKIP_NOT_FINISHED``.** A cell missing from a job that is
       still running is not missing, it is pending, and classifying it any other way turns an
       impatient operator into a duplicate submission. This is also where a cancelled family is
       caught: someone stopped that work on purpose, and quietly offering to resurrect it is not a
       repair — ``UNKNOWN`` says so without deciding for them.
    3. **No metadata row → ``RETRY_AS_IS``.** The cell never ran and the family is finished.
       Nothing is known to be wrong with the cell; the job that should have covered it did not get
       there.
    4. **Metadata says ``ok``, but no predictions.** A contradiction: the worker recorded success
       and wrote no forecast. Something is wrong with the *run*, not with this cell, and a retry
       that re-fits into the same hole is not a diagnosis — ``UNKNOWN``, no action.
    5. **Otherwise it is an error cell** — dispatch on `ERROR_CLASS_VERDICTS`. An error row with no
       ``error_class`` at all predates the column or came from a writer that does not fill it, and
       is ``UNKNOWN`` for the same reason as an unrecognised token.
    6. **Finally, a capacity-bound family downgrades any retry to ``RETRY_LATER``.** The per-cell
       row cannot see this: a cell that never ran under a family that hit a capacity wall reads as
       ``RETRY_AS_IS``, and resubmitting it immediately walks into the same wall. The downgrade
       only ever moves a retry to a *later* retry, so it can never turn a refusal into work.
    """
    if state.has_predictions:
        return SKIP_ALREADY_DONE
    if family is not None:
        if family.is_live:
            return SKIP_NOT_FINISHED
        if family.status == "CANCELLED":
            return UNKNOWN
    verdict = _verdict_from_cell(state)
    if verdict in RETRY_VERDICTS and family is not None and family.is_capacity_bound:
        return RETRY_LATER
    return verdict


def _verdict_from_cell(state: CellState) -> str:
    """Steps 3–5 of `classify_cell` — everything the cell's own row can decide (pure)."""
    if not state.has_metadata:
        return RETRY_AS_IS
    if state.cell_status == "ok":
        return UNKNOWN
    if state.error_class is None:
        return UNKNOWN
    return ERROR_CLASS_VERDICTS.get(state.error_class, UNKNOWN)


def build_worklist(
    states: Iterable[CellState], families: Mapping[str, FamilyState] | None = None
) -> Worklist:
    """Classify every cell and split it into a `Worklist` (pure).

    ``families`` maps a family name to what is known about its job, joined to each cell by
    `CellState.family`. Omitting it is legitimate — a report built without the job rows is a weaker
    report, not a broken one — and a cell whose family is absent from the mapping is classified on
    its own row alone rather than being dropped.

    Raises `RegistryError` if the no-overlap postcondition is broken — a targeted cell that already
    has predictions. That can only happen if `classify_cell` is edited into disagreeing with itself,
    which is exactly the edit worth failing loudly on: the quiet version of that bug appends a
    second forecast beside a reviewed one and lets the write clock decide which the run means.
    """
    grouped: dict[str, list[CellState]] = {}
    for state in states:
        family = None if families is None or state.family is None else families.get(state.family)
        grouped.setdefault(classify_cell(state, family), []).append(state)
    worklist = Worklist(by_verdict={v: tuple(cells) for v, cells in grouped.items()})

    overlapping = [c for c in worklist.targets if c.has_predictions]
    if overlapping:
        raise RegistryError(
            f"retry worklist would re-run {len(overlapping)} cell(s) that already have "
            f"predictions, e.g. {overlapping[0].ts_id}/{overlapping[0].model_type}; "
            "a landed forecast is replaced by a new run, never by a retry"
        )
    return worklist
