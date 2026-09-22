"""Is this job actually still running? — fuse registry, artifacts and live runtime into verdicts.

The question a reader asks when a progress bar stops moving, and the answer is not any one
source's alone. A ``run_jobs`` row records a last-written status that a crashed driver never got
to update; the landed-cell counts say what work survived; the runtime says whether the job still
exists. One `FamilyVerdict` per family is what you get from fusing all three.

Split pure from I/O, as the rest of this codebase is: `_verdict_for_family`,
`_assemble_probe_report` and `_narrow_to_job` are pure and unit-tested per matrix row, and
`_read_and_probe` is the one function that reads BigQuery and calls the probes before handing the
assembled inputs over.

**Escalation is the cost control.** A terminal family is never probed — the registry is
authoritative once the work is done — so a routine poll of a finished run makes zero native calls.
Only non-terminal jobs are escalated, and `_is_stale` decides whether a *vanished* one is
genuinely LOST or merely still-starting. That startup grace is why a normal launch window does not
cry wolf.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ..errors import ConfigError
from .runtimes import get_probe
from .vocabulary import (
    _AWAITING_CAPACITY,
    _EMITTED,
    _REGISTRY_RUNNING,
    _TERMINAL,
    NATIVE_FAILED,
    NATIVE_NOT_FOUND,
    NATIVE_RUNNING,
    NATIVE_SUCCEEDED,
    VERDICT_ABANDONED_WAIT,
    VERDICT_LIKELY_COMPLETED,
    VERDICT_LOST,
    VERDICT_RUNNING,
    VERDICT_STALE_REGISTRY,
    VERDICT_TRUST_REGISTRY,
    VERDICT_UNKNOWN,
    ProbeHandle,
    ProbeResult,
)

if TYPE_CHECKING:
    from ..review import FamilyProgress, RunProgress
    from ..settings import Settings


# How long a RUNNING row must have gone quiet before a *vanished* runtime job (native NOT_FOUND with
# its artifacts still incomplete) is judged LOST rather than merely still-starting: below this floor
# a not-yet-created runtime job reads as UNKNOWN, not a false LOST (the startup grace). Tunable per
# call via `probe_run(stale_after_s=...)` (G2).
_DEFAULT_STALE_S = 900.0

# The same idea for the other non-terminal status, on a far longer clock. An ``AWAITING_CAPACITY``
# row is quiet *on purpose*, so it needs a window bounded by the walk's own patience rather than by
# a launch grace: `capacity.CapacityPolicy.max_wall_seconds` ships at 3600s, and its docstring puts
# the real ceiling at that plus one in-flight attempt (~12 min for a Ray GPU provision). Past double
# that, a living walk would already have written ``FAILED``/``CAPACITY_EXHAUSTED`` — so the silence
# is evidence the *driver* is gone, not that it is still being patient. Generous on purpose: reading
# too early puts a false FAILED on a run that was fine, while reading too late only leaves a dead
# row sitting a while longer. Tunable per call via `probe_run(abandoned_after_s=...)` for a
# deployment that raised its own budget (G2).
_DEFAULT_ABANDONED_WAIT_S = 7200.0

# The statuses the startup grace above applies to — see `_is_stale` for why these two and not the
# third non-terminal one.
_STALEABLE = frozenset({_REGISTRY_RUNNING, _EMITTED})

# --- reconciliation (pure) ----------------------------------------------------
# The layer above the probes: fuse a run's registry+artifact progress (`review.RunProgress`) with
# the live native readings (`vocabulary.ProbeResult`, only for families that were escalated) into
# one verdict per family. Pure and unit-tested per matrix row — the I/O caller (`probe_run`) does
# the reads and calls the probes, then hands the assembled inputs here.


@dataclass(frozen=True)
class FamilyVerdict:
    """The reconciled truth for one family: what the registry says vs. what the runtime says.

    ``registry_status`` is the family's last-written job status; ``native_state`` / ``exists`` are
    the live `vocabulary.ProbeResult` reading (``None`` for a family that wasn't escalated —
    terminal or never launched). ``verdict`` is one of the ``VERDICT_*`` set and ``disagreement`` is
    ``True`` only when the runtime contradicts the registry (a reader's "look here" flag).
    ``n_done`` / ``n_expected`` are the family's landed-vs-expected cell counts (the artifact
    evidence that splits a vanished job into ``LIKELY_COMPLETED`` vs. ``LOST``). ``detail`` is a
    short human reason; ``telemetry`` carries anything the probe cheaply gathered.
    """

    family: str
    runtime: str | None
    registry_status: str | None
    native_state: str | None
    exists: bool | None
    verdict: str
    disagreement: bool
    n_done: int
    n_expected: int | None
    detail: str
    telemetry: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProbeReport:
    """A run's reconciled snapshot: header status + one `FamilyVerdict` per family.

    ``escalated`` is ``True`` when at least one family was probed against its runtime (``False`` for
    a routine poll of an all-terminal run — the terminal short-circuit that keeps polling native-
    call-free). ``disagreement`` is the run-wide roll-up: ``True`` when any family's runtime
    contradicts its registry status.
    """

    run_id: str
    status: str | None
    escalated: bool
    families: tuple[FamilyVerdict, ...]
    disagreement: bool


def _is_stale(fp: FamilyProgress, stale_after_s: float | None) -> bool:
    """Whether a pre-terminal family has gone quiet long enough to be past its startup grace.

    A family is stale when its status is one of `_STALEABLE` and its ``quiet_seconds`` — the age of
    its last registry signal, measured once by `review._assemble_progress` so every family in a
    report is judged against the same instant — exceeds ``stale_after_s`` (default
    `_DEFAULT_STALE_S`). A stale family whose runtime job has vanished (native NOT_FOUND, artifacts
    incomplete) is judged LOST; a *young* one is still-starting, so it reads UNKNOWN (the grace that
    stops a probe crying wolf during a normal launch window). Any other status is never stale. Pure
    and defensive: a family whose timestamps didn't parse has no ``quiet_seconds`` and is treated as
    *not* stale, never raising.

    Two statuses qualify and they need the same grace for the same reason. ``RUNNING`` is the
    original case: the row is written before the platform job exists, so a fresh 404 means
    "starting", not "gone". ``EMITTED`` is the staged-command case (`registry.rows.EMITTED`): the
    row is written when `launch_plan.stage_run` hands out the job id, and the operator pasting the
    command takes a little while to do it, so an immediate 404 means "not pasted yet". Past the same
    grace, both readings mean the job is not coming — and `settle._settle_decision` splits them into
    ``RUNTIME_LOST`` vs ``NEVER_LAUNCHED``, because a job that ran and vanished is worth
    investigating and one that never started is not.

    What stays out is ``AWAITING_CAPACITY``, and it should: a family waiting on a stocked-out region
    is quiet *on purpose* and can legitimately stay quiet for an hour, so measuring its silence
    against a launch-window grace would report every capacity wait as a lost job. It has its own,
    far longer clock in `_is_abandoned_wait`.

    This is the judgement half of a two-part split: the monitor reports the age (a fact anyone can
    read off a frozen bar), and the threshold that turns it into an escalation lives here, with the
    probe that acts on it.
    """
    if (fp.status or "").upper() not in _STALEABLE or fp.quiet_seconds is None:
        return False
    threshold = _DEFAULT_STALE_S if stale_after_s is None else stale_after_s
    return fp.quiet_seconds > threshold


def _is_abandoned_wait(fp: FamilyProgress, abandoned_after_s: float | None) -> bool:
    """Whether an ``AWAITING_CAPACITY`` family's walk has outlived any walk's own patience (pure).

    The counterpart to `_is_stale` for the *other* non-terminal status, and deliberately a separate
    function on a separate clock rather than a widening of that one — the two windows answer
    different questions and share nothing but their shape. `_is_stale` asks "has a job that should
    be writing gone quiet?" against a launch grace. This asks "is anyone still walking?" against
    `_DEFAULT_ABANDONED_WAIT_S`, which is sized from the capacity policy's own wall-clock budget.

    The reasoning that makes this safe without a heartbeat: a walk that is still alive terminates
    *itself* at its budget, writing ``FAILED``/``CAPACITY_EXHAUSTED``. So a row that is still
    ``AWAITING_CAPACITY`` at twice the shipped budget was not written by a walk that is deciding
    slowly — it was written by one that stopped existing. Defensive in the same way as `_is_stale`:
    an unparseable timestamp yields no ``quiet_seconds`` and is treated as *not* abandoned.
    """
    if (fp.status or "").upper() != _AWAITING_CAPACITY or fp.quiet_seconds is None:
        return False
    threshold = _DEFAULT_ABANDONED_WAIT_S if abandoned_after_s is None else abandoned_after_s
    return fp.quiet_seconds > threshold


def _has_submitted_handle(row: Mapping[str, Any]) -> bool:
    """Whether a job row's handle names a runtime job at all — something to ask about (pure).

    This is a "worth an API call" test, not a "was submitted" test, and the distinction is worth
    stating because it was originally read the other way. A non-empty ``native_id`` looks like
    evidence of a submit and is not: every runtime stamps the id it is *about to* submit under into
    the entry handle before launching, so the id is there from the moment the row is written. What
    it means is narrower — that we know a name to ask about, so escalating this row buys a real
    answer rather than a guaranteed 404 on the empty string.

    Deciding what that answer is allowed to *do* to an ``AWAITING_CAPACITY`` row is a separate
    question, and it belongs with the reading rather than with the handle:
    `_capacity_reading_is_informative`.
    """
    handle = ProbeHandle.from_job_row(dict(row))
    return handle is not None and bool(handle.native_id)


def _capacity_reading_is_informative(result: ProbeResult | None, artifacts_complete: bool) -> bool:
    """Whether a runtime reading may overrule an ``AWAITING_CAPACITY`` row (pure).

    `_probe_targets` escalates a capacity wait whose handle carries a job id, and this decides
    whether the answer is worth anything. Only two kinds of reading are: the runtime naming a job
    it has (``RUNNING``/``FAILED``/``SUCCEEDED`` — a job that exists was submitted, whatever the
    status says), or a complete set of artifacts (work that landed was done by something).

    Everything else has to yield, and the reason is that it cannot tell the two halves of the
    status apart. A family walking regions has no runtime job *because it has not submitted one*;
    a family whose launcher died after submitting has no runtime job *because the job is gone*.
    Both read ``NOT_FOUND`` with incomplete cells, so treating that as evidence would report every
    healthy capacity wait in the fleet as ``UNKNOWN`` — the one verdict an operator is told to go
    investigate — while the registry could have said exactly what was happening.

    Why this guard is here and not in `_has_submitted_handle`, which is where the split was
    originally drawn: that helper infers "a job was submitted" from a non-empty ``native_id``, and
    the inference does not hold. Every runtime stamps the id it is *about to* submit under into the
    entry handle before launching, precisely so the probe can address the job during the window
    before the submit call returns. So the id is present from the start and says nothing about
    whether a submit happened. The reading does.

    A capacity wait that genuinely lost its job is not abandoned here — it falls back to the
    registry and the abandoned-wait clock picks it up, which is the witness of last resort that
    clock exists to be.

    Note the order of the two tests. Completeness **corroborates** a reading; it is not one. A row
    that was never escalated has complete cells and no runtime answer, and it must still reach the
    capacity branch — that is where the abandoned-wait clock lives, and where `settle`'s
    completeness guard refuses to write ``FAILED`` over a run whose every cell landed.
    """
    if result is None:
        return False
    if result.native_state in (NATIVE_RUNNING, NATIVE_FAILED, NATIVE_SUCCEEDED):
        return True
    return artifacts_complete


def _probe_targets(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The job rows worth escalating to a runtime, in order (pure).

    A row is skipped when its status is terminal — the registry is authoritative and the runtime
    job is very likely gone. Everything else is escalated. Pure so the "don't spend an API call on
    a job that provably isn't there" rule is testable offline — it lives out here for the same
    reason `_narrow_to_job` does.

    ``AWAITING_CAPACITY`` is skipped when its handle names no job to ask about, because then the
    call is guaranteed to tell us nothing. With a name it is escalated, for the row shape seen live
    in ``reaper-orphan-c-5873d5cca61a``: a launcher that died seconds *after* submitting left its
    row at ``AWAITING_CAPACITY`` with the runtime job id sitting right there in it, while the job
    ran to completion and landed all its cells.

    That escalation reaches healthy capacity waits as well — the id in the handle is written before
    the submit, not by it — so the cost of this rule is one live call per waiting family. It is
    bounded (a probe is invoked by an operator, not by a loop) and it buys the only reading that
    can find an orphan. What must *not* follow is the reading overruling the registry on no
    evidence, and that is `_capacity_reading_is_informative`'s job, over in `_verdict_for_family`.
    """
    out: list[dict[str, Any]] = []
    for r in rows:
        status = (r.get("status") or "").upper()
        if status in _TERMINAL:
            continue
        if status == _AWAITING_CAPACITY and not _has_submitted_handle(r):
            continue
        out.append(dict(r))
    return out


def _verdict_for_family(
    fp: FamilyProgress,
    native: dict[str, ProbeResult],
    no_handle: frozenset[str],
    stale: frozenset[str],
    abandoned: frozenset[str] = frozenset(),
) -> FamilyVerdict:
    """The verdict matrix for one `review.FamilyProgress`, as a single if-ladder.

    ``fp`` is a `review.FamilyProgress`; ``native`` holds a `vocabulary.ProbeResult` only for
    families that were escalated; ``no_handle`` names families escalated but with no usable
    `vocabulary.ProbeHandle`; ``stale`` names families past their startup grace (a vanished job in
    this set is LOST, a young one is still-starting → UNKNOWN); ``abandoned`` names capacity waits
    past any walk's own budget (`_is_abandoned_wait`).

    Two statuses short-circuit before any native reading is consulted: a terminal one (the work is
    done) and ``AWAITING_CAPACITY`` (the work has not started). Both trust the registry, for
    opposite reasons — except for the one capacity wait that has outlived the walk that was
    supposed to end it, which is the single reading here taken from the clock alone.

    The capacity short-circuit yields to a native reading, but only to one that proves a job was
    submitted — the runtime naming a job it has, or a complete set of artifacts
    (`_capacity_reading_is_informative`). A runtime's answer about a job that really exists outranks
    the status a dead launcher left behind; a bare "no such job" does not, because a family still
    walking regions gives exactly the same answer. Such a family is never in ``stale`` either —
    `_is_stale` deliberately leaves ``AWAITING_CAPACITY`` out, because a capacity wait has its own
    far longer clock — so nothing on this path can read LOST. From a pre-launch-looking row we
    cannot tell a job that died from one that never started, and where no runtime reading settles
    it the abandoned-wait clock is the witness of last resort.

    ``EMITTED`` takes no short-circuit at all: it reconciles exactly like ``RUNNING``, and the one
    place the two part company is `settle`, which reads a vanished ``EMITTED`` job as a command
    nobody ran rather than as a loss.
    """
    common: dict[str, Any] = {
        "family": fp.family,
        "runtime": fp.runtime,
        "registry_status": fp.status,
        "n_done": fp.n_done,
        "n_expected": fp.n_expected,
    }
    # Terminal → the registry is authoritative; the family was never probed (short-circuit).
    if (fp.status or "") in _TERMINAL:
        return FamilyVerdict(
            **common,
            native_state=None,
            exists=None,
            verdict=VERDICT_TRUST_REGISTRY,
            disagreement=False,
            detail="",
        )
    # Waiting between capacity attempts → in flight, but deliberately not progressing. There is no
    # runtime job to reconcile against and that is the expected state, not a gap in our knowledge:
    # UNKNOWN would be a lie ("we couldn't tell") about the one status where we can tell exactly.
    #
    # A row that really did submit gets past this short-circuit, but only on a reading that proves
    # it — see `_capacity_reading_is_informative`. Escalation alone is not enough: every runtime now
    # writes its job id into the handle *before* submitting, so `_probe_targets` escalates healthy
    # capacity waits too, and letting a bare "no such job" through would answer UNKNOWN for a family
    # that is simply hopping regions.
    if (fp.status or "").upper() == _AWAITING_CAPACITY and not _capacity_reading_is_informative(
        native.get(fp.family),
        fp.n_expected is not None and fp.n_done >= fp.n_expected,
    ):
        # ...unless it has been waiting longer than any walk is allowed to. A live walk ends itself
        # at its budget; one that did not is a walk with nobody left to walk it, and saying
        # TRUST_REGISTRY there tells an operator to keep waiting for a decision that will never
        # come. `disagreement=True` because this *is* the registry being contradicted — by the
        # clock rather than by a runtime, which is the only witness a pre-launch row can have.
        if fp.family in abandoned:
            quiet = f"{fp.quiet_seconds / 3600:.1f}h" if fp.quiet_seconds else "long"
            return FamilyVerdict(
                **common,
                native_state=None,
                exists=None,
                verdict=VERDICT_ABANDONED_WAIT,
                disagreement=True,
                detail=f"awaiting capacity for {quiet}, past any walk's budget; driver is gone",
            )
        return FamilyVerdict(
            **common,
            native_state=None,
            exists=None,
            verdict=VERDICT_TRUST_REGISTRY,
            disagreement=False,
            detail="awaiting capacity; no runtime job exists yet",
        )
    # Non-terminal but escalated with no usable handle (pre-feature / malformed blob) → can't tell.
    if fp.family in no_handle:
        return FamilyVerdict(
            **common,
            native_state=None,
            exists=None,
            verdict=VERDICT_UNKNOWN,
            disagreement=False,
            detail="no handle recorded",
        )
    result = native.get(fp.family)
    # Non-terminal and not escalated (no job row yet — never launched) → nothing to reconcile.
    if result is None:
        return FamilyVerdict(
            **common,
            native_state=None,
            exists=None,
            verdict=VERDICT_TRUST_REGISTRY,
            disagreement=False,
            detail="",
        )
    ns = result.native_state
    artifacts_complete = fp.n_expected is not None and fp.n_done >= fp.n_expected
    detail = result.detail
    if ns == NATIVE_RUNNING:
        verdict, disagreement = VERDICT_RUNNING, False
    elif ns == NATIVE_FAILED:
        # A failed runtime job is authoritative — the registry never recorded the terminal state.
        verdict, disagreement = VERDICT_STALE_REGISTRY, True
    elif ns == NATIVE_SUCCEEDED:
        # "Succeeded" is only trustworthy when the artifacts corroborate it. A native family's
        # BigQuery statements go DONE one-by-one, so an all-DONE reading mid-run is a lull between
        # statements, not the end: complete artifacts ⇒ the registry is genuinely stale, otherwise
        # it's ambiguous (don't overrule the registry on a transient all-DONE).
        if artifacts_complete:
            verdict, disagreement = VERDICT_STALE_REGISTRY, True
        else:
            verdict, disagreement = VERDICT_UNKNOWN, False
            detail = detail or "runtime reports success but artifacts are incomplete"
    elif ns == NATIVE_NOT_FOUND:
        # The job/cluster is gone. Complete artifacts ⇒ it finished; otherwise a job past its
        # startup grace (stale) is LOST, while a young one is just still starting (a RUNNING row is
        # written before the native job exists, so a fresh probe legitimately 404s → UNKNOWN, not a
        # false LOST).
        if artifacts_complete:
            verdict, disagreement = VERDICT_LIKELY_COMPLETED, True
        elif fp.family in stale:
            verdict, disagreement = VERDICT_LOST, True
        else:
            verdict, disagreement = VERDICT_UNKNOWN, False
            detail = detail or "runtime has no record yet; job may still be starting"
    else:  # NATIVE_UNKNOWN — the probe degraded; don't overrule the registry.
        verdict, disagreement = VERDICT_UNKNOWN, False
    return FamilyVerdict(
        **common,
        native_state=ns,
        exists=result.exists,
        verdict=verdict,
        disagreement=disagreement,
        detail=detail,
        telemetry=result.telemetry,
    )


def _assemble_probe_report(
    progress: RunProgress,
    native: dict[str, ProbeResult],
    no_handle: frozenset[str],
    stale: frozenset[str] = frozenset(),
    abandoned: frozenset[str] = frozenset(),
) -> ProbeReport:
    """Fuse registry+artifact `review.RunProgress` with live native readings into a `ProbeReport`.

    Pure: ``native`` carries a `vocabulary.ProbeResult` only for the escalated families and
    ``no_handle`` names the escalated ones that had no usable handle; ``stale`` names families past
    their startup grace (used to split a vanished job into LOST vs. still-starting); ``abandoned``
    names capacity waits past any walk's own budget; every other family reconciles from the registry
    alone. ``escalated`` reflects whether any family was probed; ``disagreement`` rolls up the
    per-family flags.
    """
    families = tuple(
        _verdict_for_family(fp, native, no_handle, stale, abandoned) for fp in progress.families
    )
    return ProbeReport(
        run_id=progress.run_id,
        status=progress.status,
        escalated=bool(native),
        families=families,
        disagreement=any(f.disagreement for f in families),
    )


def _narrow_to_job(
    progress: RunProgress, job_rows: Sequence[Mapping[str, Any]], job: str | None, run_id: str
) -> tuple[RunProgress, list[dict[str, Any]]]:
    """Narrow a run's progress **and** its job rows to one family — or both untouched (pure).

    Extracted from `_read_and_probe`'s GCP body precisely because the two filters have to agree.
    They feed different consumers: the narrowed ``progress`` becomes the `ProbeReport` an operator
    reads, while the narrowed ``rows`` are what `cancel.cancel_run` stops and finalizes. Filtering
    one and not the other would let ``--job statistical`` print a one-family preview and then cancel
    every family in the run, so the agreement is a data-integrity property, not tidiness — and this
    is the seam that makes it checkable with no cloud.

    An unknown ``job`` raises `errors.ConfigError` naming the run's actual families. A typo has to
    fail loudly: filtering to nothing would otherwise report an empty, healthy-looking run.
    """
    if job is None:
        return progress, [dict(r) for r in job_rows]
    known = {f.family for f in progress.families}
    if job not in known:
        raise ConfigError(f"unknown family {job!r}; run {run_id} has: {sorted(known)}")
    narrowed = replace(progress, families=tuple(f for f in progress.families if f.family == job))
    return narrowed, [dict(r) for r in job_rows if r["family"] == job]


# --- I/O caller ----------------------------------------------------------------
# The thin reader that turns a run_id into a ProbeReport: read the registry (header + config + job
# rows + landed-cell counts), escalate only the incomplete/stale jobs to their runtime, then hand
# the assembled inputs to the pure `_assemble_probe_report`. It *is* `review.monitor_run`'s
# read-then-assemble body plus the escalation, which is why `monitor_run(probe=True)` delegates
# here rather than reading the registry a second time; all GCP imports stay lazy in the function.


def _read_and_probe(
    run_id: str,
    *,
    job: str | None,
    settings: Settings,
    stale_after_s: float | None,
    abandoned_after_s: float | None = None,
) -> tuple[RunProgress, ProbeReport, list[dict[str, Any]]]:  # pragma: no cover - GCP I/O
    """Read the registry, escalate the incomplete jobs, reconcile → `(progress, report, rows)`.

    The shared read+probe body behind three callers: `probe_run` (which keeps just the report),
    `cancel.cancel_run` (which also needs the ``v_run_jobs`` rows for their ``job_id`` +
    ``probe_handle`` to finalize), and `review.monitor_run` with ``probe=True`` (which keeps the
    progress and attaches the report) — so a monitor that escalates pays for the native calls, not
    for a second set of registry queries. ``rows`` is the run's job rows filtered to ``job`` (all
    families when ``None``); ``progress`` and the reconciled ``report`` cover the same set.
    ``abandoned_after_s`` overrides the capacity-wait window (`_DEFAULT_ABANDONED_WAIT_S`).
    """
    from ..config import RunConfig
    from ..registry.jobs import read_run_jobs
    from ..registry.reads import read_progress, read_run_config, read_run_summary
    from ..review import _assemble_progress

    summary = read_run_summary(run_id, settings=settings)
    raw = read_run_config(run_id, settings=settings)
    cfg = RunConfig.model_validate(raw) if raw else None
    job_rows = read_run_jobs(run_id, settings=settings) if cfg else []
    progress_rows = read_progress(run_id, settings=settings) if cfg else []
    progress = _assemble_progress(
        run_id, summary, cfg, job_rows, progress_rows, now=datetime.now(UTC)
    )

    # --job narrows the report and the rows together, or neither — see `_narrow_to_job`.
    progress, rows = _narrow_to_job(progress, job_rows, job, run_id)
    # Escalate every non-terminal job to its runtime; terminal rows short-circuit to the registry,
    # and so do AWAITING_CAPACITY rows that still look pre-launch — a family between capacity
    # attempts has no runtime job to address, so escalating it would spend a live API call to be
    # told NOT_FOUND, which we already knew. One carrying a runtime job id *is* escalated, because
    # it has provably left the walk. `_probe_targets` is the pure half, checkable with no cloud.
    to_probe = _probe_targets(rows)
    # A RUNNING family quiet longer than the floor is "stale" — past its startup grace, so a
    # vanished runtime job is judged LOST rather than still-starting (see `_verdict_for_family`).
    # Read off the assembled progress, so the monitor's reported age and the probe's escalation
    # decision can never disagree about how quiet a family has been.
    stale = frozenset(f.family for f in progress.families if _is_stale(f, stale_after_s))
    # And an AWAITING_CAPACITY family quiet past any walk's own budget is "abandoned" — the one
    # reading here that comes from the clock with no runtime to ask, because a pre-launch row has
    # no runtime to ask (see `_is_abandoned_wait`). It deliberately does not widen `to_probe`: the
    # clock is the witness of last resort, used only where a runtime reading was unavailable.
    abandoned = frozenset(
        f.family for f in progress.families if _is_abandoned_wait(f, abandoned_after_s)
    )
    native: dict[str, ProbeResult] = {}
    no_handle: set[str] = set()
    for r in to_probe:
        handle = ProbeHandle.from_job_row(r)
        if handle is None:
            no_handle.add(r["family"])
            continue
        native[r["family"]] = get_probe(handle.runtime).check(handle, settings=settings)
    report = _assemble_probe_report(progress, native, frozenset(no_handle), stale, abandoned)
    return progress, report, rows


def probe_run(
    run_id: str,
    *,
    job: str | None = None,
    settings: Settings | None = None,
    stale_after_s: float | None = None,
    abandoned_after_s: float | None = None,
) -> ProbeReport:  # pragma: no cover - GCP I/O
    """Reconcile a run's registry state against live runtime state → a `ProbeReport`.

    Reads the run's header + config + job rows + landed-cell counts (`registry.reads`,
    `registry.jobs`), assembles the registry-side progress (`review._assemble_progress`), then
    escalates **only** the non-terminal
    jobs to their runtime — a routine poll of an already-terminal run touches no runtime (empty
    ``to_probe`` ⇒ ``escalated=False``). ``job`` narrows *both* the escalation and the report to one
    family (the per-family drill-down; an unknown name raises `ConfigError` listing the valid ones);
    ``settings`` is the GCP identity (from the ``SF_*`` env when ``None``); ``stale_after_s``
    overrides the startup-grace floor (`_DEFAULT_STALE_S`) that decides whether a vanished young job
    reads LOST or still-starting, and ``abandoned_after_s`` the far longer window
    (`_DEFAULT_ABANDONED_WAIT_S`) past which a capacity wait reads as a walk nobody is walking. A
    family whose handle can't be parsed (a pre-feature or malformed row) degrades to registry-only
    via ``no_handle`` rather than raising.
    """
    from ..settings import Settings

    s = settings if settings is not None else Settings.resolve()
    _progress, report, _rows = _read_and_probe(
        run_id,
        job=job,
        settings=s,
        stale_after_s=stale_after_s,
        abandoned_after_s=abandoned_after_s,
    )
    return report
