"""Stop a run's jobs — the blast-radius plan, the confirmation gate, and the audit trail.

The only destructive, outward-facing path in this package, so it is built around **honesty about
what stops and what survives** (§7 of the design). Three properties hold it together:

*Preview by default.* Without ``confirm``, `cancel_run` probes, builds the `CancelPlan`, and
touches no runtime and no registry. You see the blast radius before anything happens to it.

*Cancel never deletes.* Partial results that already landed are retained; they are labelled by the
CANCELLED status and the cell counts, never removed. A cancelled run stays readable.

*Every stop is auditable.* Each finalized row carries who cancelled it, when, why, and what the
runtime state was at the moment of the stop, merged into ``job_telemetry.$.cancel`` alongside the
handle that addressed the job — so a cancelled attempt stays reconcilable afterwards.

The pure pieces — plan, audit blob, header roll-up, and the plan→rows join (`_cancel_steps`) that
decides which families a confirmed cancel actually addresses — are unit-tested per row; `cancel_run`
is the I/O orchestrator that sequences them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from .reconcile import ProbeReport, _read_and_probe
from .runtimes import get_probe
from .vocabulary import (
    _CANCELLED,
    _TERMINAL,
    NATIVE_NOT_FOUND,
    ProbeHandle,
    _parse_ts,
)

if TYPE_CHECKING:
    from ..settings import Settings


# --- cancel: blast-radius plan, audit, header roll-up (pure) ------------------
# Cancel is destructive and outward-facing, so it is built around *honesty about what stops and what
# survives* (§7 of the design). The pure pieces here — the blast-radius plan, the audit blob, the
# header roll-up — are unit-tested per row; the I/O orchestrator (`cancel_run`) probes first, shows
# the plan, gates on explicit confirmation, then stops each runtime job and finalizes the registry.


@dataclass(frozen=True)
class CancelPlanItem:
    """One family's line in the blast-radius preview: what will happen and what data is retained.

    ``cancellable`` is keyed on the *registry* status (non-terminal ⇒ will cancel) so the plan is
    robust even when the live probe degraded; ``native_state`` enriches the note when we have it.
    ``n_done``/``n_expected`` are the landed-vs-expected cells — the partial results a cancel
    **retains** (never deletes). ``note`` is the human one-liner shown in the preview.
    """

    family: str
    runtime: str | None
    registry_status: str | None
    native_state: str | None
    cancellable: bool
    n_done: int
    n_expected: int | None
    note: str


@dataclass(frozen=True)
class CancelPlan:
    """The full blast radius of a cancel: one `CancelPlanItem` per family + the ensemble impact.

    ``ensemble_suppressed`` is ``True`` when cancelling a base family will cause the run's ensemble
    node to be skipped (§7.4 — a cancelled base family suppresses the ensemble rather than producing
    a lopsided one). ``n_cancellable`` is how many jobs a confirmed cancel would actually stop.
    """

    run_id: str
    items: tuple[CancelPlanItem, ...]
    ensemble_suppressed: bool

    @property
    def n_cancellable(self) -> int:
        return sum(1 for i in self.items if i.cancellable)


@dataclass(frozen=True)
class CancelOutcome:
    """What actually happened to one family's job on a confirmed cancel.

    ``requested`` is ``True`` when a stop was issued to the runtime (``False`` when we couldn't even
    address the job — no handle). ``cancelled`` is ``True`` when the registry row was finalized to
    ``CANCELLED`` (the runtime confirmed the stop, or the job was already gone). ``stopped`` /
    ``already_gone`` are the raw `vocabulary.CancelResult` flags; ``detail`` is the short reason.
    """

    family: str
    job_key: str
    requested: bool
    cancelled: bool
    stopped: bool
    already_gone: bool
    detail: str


@dataclass(frozen=True)
class CancelReport:
    """The result of a cancel call — a preview (``executed=False``) or the executed outcome.

    ``plan`` is the blast radius (always present — it is what a no-``confirm`` call returns).
    ``outcomes`` is empty for a preview and one `CancelOutcome` per attempted family otherwise.
    ``header_status`` is the run's rolled-up status after the cancel (``None`` when the header was
    not changed — e.g. a preview, or a cancel whose stops all failed). ``actor`` / ``reason`` are
    the audit identity recorded on the cancelled rows.
    """

    run_id: str
    plan: CancelPlan
    executed: bool
    outcomes: tuple[CancelOutcome, ...]
    header_status: str | None
    actor: str | None
    reason: str


def _assemble_cancel_plan(report: ProbeReport) -> CancelPlan:
    """Turn a reconciled `reconcile.ProbeReport` into the blast-radius plan (pure).

    A family is cancellable when its registry status is non-terminal (`vocabulary._TERMINAL`
    includes ``CANCELLED``, so an already-cancelled job is a no-op). The note states what a cancel
    retains (landed cells) and, when the runtime already vanished, says so. The ensemble node is
    flagged as suppressed when any *base* family will be cancelled.
    """
    base_cancelled = any(
        fv.family != "ensemble" and (fv.registry_status or "") not in _TERMINAL
        for fv in report.families
    )
    has_ensemble = any(fv.family == "ensemble" for fv in report.families)
    ensemble_suppressed = base_cancelled and has_ensemble
    items: list[CancelPlanItem] = []
    for fv in report.families:
        cancellable = (fv.registry_status or "") not in _TERMINAL
        exp = fv.n_expected if fv.n_expected is not None else "?"
        if not cancellable:
            note = f"already {fv.registry_status or 'terminal'}, untouched"
        elif fv.family == "ensemble" and ensemble_suppressed:
            note = "will be SKIPPED (a base family is cancelled)"
        else:
            note = f"will cancel; {fv.n_done}/{exp} series landed (retained)"
            if fv.native_state == NATIVE_NOT_FOUND:
                note += "; runtime job already gone"
        items.append(
            CancelPlanItem(
                family=fv.family,
                runtime=fv.runtime,
                registry_status=fv.registry_status,
                native_state=fv.native_state,
                cancellable=cancellable,
                n_done=fv.n_done,
                n_expected=fv.n_expected,
                note=note,
            )
        )
    return CancelPlan(
        run_id=report.run_id, items=tuple(items), ensemble_suppressed=ensemble_suppressed
    )


def _build_cancel_audit(
    *,
    actor: str | None,
    cancelled_at: datetime,
    reason: str,
    native_state: str | None,
    n_done: int,
) -> dict[str, Any]:
    """The audit blob stamped under ``run_jobs.job_telemetry.$.cancel`` (pure).

    Captures *who / when / why* plus the *observed reality* at cancel time (the live native state
    and how many series had landed) so the trail records both intent and what was stopped (§9).
    """
    return {
        "cancelled_by": actor,
        "cancelled_at": cancelled_at.isoformat(),
        "reason": reason,
        "native_state_at_cancel": native_state,
        "n_done_at_cancel": n_done,
    }


def _roll_header_after_cancel(statuses: list[str | None]) -> str | None:
    """Roll a run's family statuses into its post-cancel header status (pure).

    Every family ``CANCELLED`` ⇒ the whole run is ``CANCELLED``; a mix of ``CANCELLED`` with other
    terminals ⇒ ``PARTIAL`` (some work finished, some was stopped). If any family is still
    non-terminal (a stop that failed, or an untouched live job) the header is left unchanged
    (``None``) — we never finalize a run whose jobs aren't all settled.
    """
    if not statuses:
        return None
    if any((s or "") not in _TERMINAL for s in statuses):
        return None
    normalized = [(s or "").upper() for s in statuses]
    if _CANCELLED not in normalized:
        return None
    if all(s == _CANCELLED for s in normalized):
        return _CANCELLED
    return "PARTIAL"


@dataclass(frozen=True)
class _CancelTarget:
    """One family a confirmed cancel will actually stop: its plan line, job row and handle."""

    item: CancelPlanItem
    row: dict[str, Any]
    handle: ProbeHandle


def _cancel_steps(
    plan: CancelPlan, rows: Sequence[Mapping[str, Any]]
) -> tuple[_CancelTarget | CancelOutcome, ...]:
    """Join the blast-radius plan to the run's job rows → what to stop, in plan order (pure).

    The plan is derived from the run's *families* and the rows from ``v_run_jobs``; they are two
    independently-assembled collections that this join has to reconcile before anything destructive
    happens, which is why it lives out here where it can be checked with no cloud. Each element is
    either a `_CancelTarget` (address the runtime, then finalize the row) or a ready-made
    `CancelOutcome` for a family that cannot be addressed at all — interleaved in plan order, so the
    report reads in the same order as the preview the operator just approved.

    Three families are deliberately *not* targets. A non-cancellable one is already terminal. One
    with no job row never launched, so there is no runtime job to stop — it is skipped silently, and
    the caller's re-read of every family is what settles the header afterwards. One whose row has no
    parseable handle (a pre-feature or malformed blob) yields the ``requested=False`` outcome: we
    know it is live and we are saying plainly that we could not reach it, rather than dropping it.
    """
    rows_by_family = {r["family"]: dict(r) for r in rows}
    steps: list[_CancelTarget | CancelOutcome] = []
    for item in plan.items:
        if not item.cancellable:
            continue
        row = rows_by_family.get(item.family)
        if row is None:
            continue
        handle = ProbeHandle.from_job_row(row)
        if handle is None:
            steps.append(
                CancelOutcome(
                    family=item.family,
                    job_key=row["job_id"],
                    requested=False,
                    cancelled=False,
                    stopped=False,
                    already_gone=False,
                    detail="no handle recorded; cannot address the runtime job",
                )
            )
            continue
        steps.append(_CancelTarget(item=item, row=row, handle=handle))
    return tuple(steps)


# --- cancel I/O orchestrator ---------------------------------------------------


def _finalize_cancelled(
    row: dict[str, Any],
    handle: ProbeHandle,
    item: CancelPlanItem,
    *,
    actor: str | None,
    cancelled_at: datetime,
    reason: str,
    settings: Settings,
) -> None:  # pragma: no cover - GCP I/O
    """Finalize one job row to ``CANCELLED`` with the audit blob merged into ``job_telemetry``.

    A non-terminal job row's ``job_telemetry`` holds only its ``probe_handle`` (sizing telemetry
    goes to the header, not the job row), so we rebuild it from the handle we parsed plus the new
    ``$.cancel`` audit blob — preserving the handle so the cancelled attempt stays reconcilable.
    """
    from ..registry.jobs import update_job

    audit = _build_cancel_audit(
        actor=actor,
        cancelled_at=cancelled_at,
        reason=reason,
        native_state=item.native_state,
        n_done=item.n_done,
    )
    telemetry: dict[str, Any] = {"probe_handle": handle.to_blob(), "cancel": audit}
    started = _parse_ts(row.get("started_at"))
    fields: dict[str, Any] = {
        "status": _CANCELLED,
        "ended_at": cancelled_at,
        "job_telemetry": telemetry,
    }
    if started is not None:
        fields["runtime_seconds"] = (cancelled_at - started).total_seconds()
    update_job(row["job_id"], settings=settings, **fields)


def cancel_run(
    run_id: str,
    *,
    job: str | None = None,
    confirm: bool = False,
    reason: str = "",
    actor: str | None = None,
    settings: Settings | None = None,
    stale_after_s: float | None = None,
) -> CancelReport:  # pragma: no cover - GCP I/O
    """Cancel a run (or one family) — preview unless ``confirm``; finalize registry to CANCELLED.

    Probes the run first (`reconcile._read_and_probe`) to reconcile live state, builds the blast-
    radius `CancelPlan`, and — **only when ``confirm``** — stops each cancellable family's runtime
    job and finalizes its row to ``CANCELLED`` with an audit blob (`_finalize_cancelled`), then
    rolls the run header up (`_roll_header_after_cancel`). Without ``confirm`` it returns the plan,
    touches no runtime and no registry (the CLI prints the preview and exits). ``job`` narrows to
    one family; ``reason`` and ``actor`` (default: resolved from ADC) are recorded for audit. Cancel
    **never deletes** — landed partial results are retained and labelled by the CANCELLED status +
    counts.
    """
    from ..identity import resolve_principal
    from ..settings import Settings

    s = settings if settings is not None else Settings.resolve()
    _progress, report, rows = _read_and_probe(
        run_id, job=job, settings=s, stale_after_s=stale_after_s
    )
    plan = _assemble_cancel_plan(report)
    if not confirm:
        return CancelReport(
            run_id=run_id,
            plan=plan,
            executed=False,
            outcomes=(),
            header_status=None,
            actor=None,
            reason=reason,
        )

    resolved_actor = actor if actor is not None else resolve_principal(s)
    cancelled_at = datetime.now(UTC)
    outcomes: list[CancelOutcome] = []
    # Which families this actually addresses is decided out here, in the open — see `_cancel_steps`.
    for step in _cancel_steps(plan, rows):
        if isinstance(step, CancelOutcome):  # unaddressable; nothing to stop, but say so
            outcomes.append(step)
            continue
        result = get_probe(step.handle.runtime).cancel(step.handle, settings=s)
        finalized = result.stopped or result.already_gone
        if finalized:
            _finalize_cancelled(
                step.row,
                step.handle,
                step.item,
                actor=resolved_actor,
                cancelled_at=cancelled_at,
                reason=reason,
                settings=s,
            )
        outcomes.append(
            CancelOutcome(
                family=step.item.family,
                job_key=step.row["job_id"],
                requested=True,
                cancelled=finalized,
                stopped=result.stopped,
                already_gone=result.already_gone,
                detail=result.detail,
            )
        )

    # Re-read every family (not just the --job subset) so the header reflects the true post-cancel
    # state — a per-family cancel makes a multi-family run PARTIAL, a whole-run cancel CANCELLED.
    from ..registry.header import update_header
    from ..registry.jobs import read_run_jobs

    all_statuses = [r.get("status") for r in read_run_jobs(run_id, settings=s)]
    header_status = _roll_header_after_cancel(all_statuses)
    if header_status is not None:
        update_header(run_id, settings=s, status=header_status)
    return CancelReport(
        run_id=run_id,
        plan=plan,
        executed=True,
        outcomes=tuple(outcomes),
        header_status=header_status,
        actor=resolved_actor,
        reason=reason,
    )
