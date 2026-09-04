"""Write back the verdict the probe already computed — the fourth registry-repair verb.

``--probe`` reads a run, reconciles it against its runtimes, decides that the deep-learning family
whose row still says RUNNING actually finished forty minutes ago — and then throws that answer
away. Nothing in the system could act on it: ``close-runs`` refuses a run with a non-terminal job
row (correctly — only a probe can tell a live job from a stale row), ``--cancel --force`` stamps
CANCELLED over work that succeeded, and ``drop-run`` destroys real predictions to fix a status
field. `settle_run` is the missing move: **``--probe`` that writes**, at the job tier, taking the
probe's own per-family verdict rather than a roll-up.

Three properties hold it together, and they are the same three that hold `cancel` together because
they are the properties a repair verb needs:

*Preview by default.* Without ``yes``, `settle_run` probes, builds the `SettlePlan`, and writes
nothing. ``yes=`` rather than ``confirm=`` is deliberate: ``confirm=`` gates stopping running work
in the cloud, ``yes=`` gates repairing a status field in our own registry — the same gate
`registry.ops.close_runs` and `registry.ops.sweep_orphans` use, and settle belongs with them.

*Refusal is the feature.* Only five (verdict, native-state) combinations settle. ``UNKNOWN`` — the
probe degraded, no handle was recorded, the runtime says SUCCEEDED but the cells have not landed —
writes nothing, ever. A verb that guessed would be worse than no verb, because its guesses would
be indistinguishable from measurements afterwards.

*Settle never deletes, and never invents time.* One UPDATE per row: the status, a
``failure_reason`` when there is a token for it, and an audit blob under ``job_telemetry.$.settle``
recording who settled it, from what, on what evidence. It does **not** stamp ``ended_at`` or
``runtime_seconds`` — a row settled three days after the fact would report three days of runtime
for a ten-minute job, and every consumer already tolerates those being NULL. It does not touch the
run header: `registry.ops.close_runs` owns that tier, and settling the job rows is what unblocks
it (the report prints the hint).

The pure pieces — the decision table, the plan, the audit blob, and the plan→rows join — are unit
tested per row; `settle_run` is the I/O orchestrator that sequences them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from .reconcile import ProbeReport, _read_and_probe
from .vocabulary import (
    _TERMINAL,
    CAPACITY_ABANDONED,
    NATIVE_FAILED,
    NATIVE_NOT_FOUND,
    NATIVE_SUCCEEDED,
    RUNTIME_FAILED,
    RUNTIME_LOST,
    VERDICT_ABANDONED_WAIT,
    VERDICT_LIKELY_COMPLETED,
    VERDICT_LOST,
    VERDICT_STALE_REGISTRY,
)

if TYPE_CHECKING:
    from ..settings import Settings
    from .reconcile import FamilyVerdict


# The statuses a settle write must not overwrite. Wider than `lifecycle._STICKY_STATUSES` (which
# protects a cancellation from the failure it caused) because settle runs *later* than anything
# else: between the probe read and the UPDATE, the process that owns the job may have finished and
# written its own terminal status, and that writer was there. Sorted for a stable rendered guard.
_SETTLE_GUARD: tuple[str, ...] = tuple(sorted(_TERMINAL))


# --- the decision table (pure) -------------------------------------------------
# Total over the verdict set, and it says "no" far more often than "yes". Each arm below is a
# reading the probe already committed to in `reconcile._verdict_for_family`; settle adds no new
# inference, it only decides which of those readings are firm enough to write down.


@dataclass(frozen=True)
class SettleDecision:
    """What one family's row will be set to, and the one-line reason it can be (pure).

    ``failure_reason`` is the short token stamped alongside a FAILED status (`vocabulary`'s
    ``RUNTIME_FAILED`` / ``RUNTIME_LOST``); ``None`` for a COMPLETED settle, which needs no excuse.
    """

    status: str
    failure_reason: str | None
    reason: str


def _settle_decision(fv: FamilyVerdict) -> SettleDecision | None:
    """The status a family's stale row should be repaired to, or ``None`` to leave it alone (pure).

    Five arms write, everything else refuses:

    * ``STALE_REGISTRY`` + runtime ``SUCCEEDED`` + every expected cell landed → ``COMPLETED``.
    * ``STALE_REGISTRY`` + runtime ``FAILED`` → ``FAILED`` / ``RUNTIME_FAILED``. The runtime is
      authoritative about its own failure; the registry simply never got the update.
    * ``LIKELY_COMPLETED`` (runtime gone, every expected cell landed) → ``COMPLETED``. This arm has
      to exist or the verb cannot repair the runtime that needs it most: a Ray cluster is garbage-
      collected when its job finishes, so "no record of the job, and all the work is in BigQuery"
      is the *normal* trace of a successful Ray run whose driver died before it closed the row.
    * ``LOST`` (runtime gone, cells missing, past the startup grace) → ``FAILED`` /
      ``RUNTIME_LOST``.
    * ``ABANDONED_WAIT`` (still ``AWAITING_CAPACITY`` past any walk's own budget) → ``FAILED`` /
      ``CAPACITY_ABANDONED``. The one arm with no runtime reading behind it, because a row that
      never launched has no runtime to read. Its witness is the clock instead
      (`reconcile._is_abandoned_wait`), and without this arm the row is unreachable by every verb
      we have: settle had nothing to probe, and ``close-runs`` refuses the header for as long as a
      non-terminal job row exists.

    ``RUNNING_CONFIRMED`` is live and must be left alone. ``TRUST_REGISTRY`` is already terminal or
    deliberately waiting. ``UNKNOWN`` is the whole point of refusing: we could not tell, so we do
    not write. The completeness re-check on the two COMPLETED arms is belt-and-braces — the verdict
    already encodes it — but this is the function that turns a reading into a permanent row, and
    the cheap second look is worth more here than the deduplication.
    """
    if (fv.registry_status or "") in _TERMINAL:
        return None
    complete = fv.n_expected is not None and fv.n_done >= fv.n_expected
    exp = fv.n_expected if fv.n_expected is not None else "?"
    if fv.verdict == VERDICT_STALE_REGISTRY and fv.native_state == NATIVE_SUCCEEDED and complete:
        return SettleDecision(
            "COMPLETED", None, f"runtime succeeded; {fv.n_done}/{exp} series landed"
        )
    if fv.verdict == VERDICT_STALE_REGISTRY and fv.native_state == NATIVE_FAILED:
        return SettleDecision(
            "FAILED", RUNTIME_FAILED, f"runtime failed; {fv.n_done}/{exp} series landed"
        )
    if fv.verdict == VERDICT_LIKELY_COMPLETED and fv.native_state == NATIVE_NOT_FOUND and complete:
        return SettleDecision(
            "COMPLETED", None, f"runtime job gone; all {fv.n_done}/{exp} series landed"
        )
    if fv.verdict == VERDICT_LOST:
        return SettleDecision(
            "FAILED", RUNTIME_LOST, f"runtime job gone; only {fv.n_done}/{exp} series landed"
        )
    if fv.verdict == VERDICT_ABANDONED_WAIT:
        return SettleDecision(
            "FAILED",
            CAPACITY_ABANDONED,
            f"capacity walk abandoned; {fv.n_done}/{exp} series landed",
        )
    return None


# --- the plan, the audit blob, and the plan→rows join (pure) -------------------


@dataclass(frozen=True)
class SettleItem:
    """One family's line in the settle preview: what it says now, what it would say, and why.

    ``decision`` is ``None`` for a family that will be left alone; ``note`` is the human one-liner
    shown in the preview either way, because *why a row is being refused* is the part an operator
    most needs — "UNKNOWN: the probe degraded" means go look, and burying it under a count of what
    was settled is how the one row that mattered gets missed.
    """

    family: str
    runtime: str | None
    registry_status: str | None
    verdict: str
    native_state: str | None
    n_done: int
    n_expected: int | None
    decision: SettleDecision | None
    note: str

    @property
    def settleable(self) -> bool:
        return self.decision is not None


@dataclass(frozen=True)
class SettlePlan:
    """Every family of a run with its settle verdict — one `SettleItem` each, in report order."""

    run_id: str
    header_status: str | None
    items: tuple[SettleItem, ...]

    @property
    def n_settleable(self) -> int:
        return sum(1 for i in self.items if i.settleable)


@dataclass(frozen=True)
class SettleOutcome:
    """What actually happened to one family's row, read back rather than assumed.

    ``final_status`` is the row's status **re-read after the write**, not the status we asked for.
    A settle UPDATE carries a status guard (`_SETTLE_GUARD`), and a guarded skip is silent — the
    statement succeeds and touches nothing — so the only honest way to report the outcome is to
    look. ``settled`` is ``True`` when ``final_status`` is what the decision asked for.
    """

    family: str
    job_key: str
    from_status: str | None
    to_status: str
    final_status: str | None
    settled: bool
    detail: str


@dataclass(frozen=True)
class SettleReport:
    """The result of a settle call — a preview (``executed=False``) or the executed outcome.

    ``plan`` is always present (it is what a no-``yes`` call returns). ``header_hint`` is the
    follow-up line when the run's header is still non-terminal after the job rows were settled:
    settle deliberately does not close headers, so it says who does.
    """

    run_id: str
    plan: SettlePlan
    executed: bool
    outcomes: tuple[SettleOutcome, ...]
    header_hint: str
    actor: str | None
    reason: str


def _assemble_settle_plan(report: ProbeReport) -> SettlePlan:
    """Turn a reconciled `reconcile.ProbeReport` into the settle preview (pure).

    One item per family, in the report's order, whether or not it will be written — a plan that
    listed only the settleable rows would answer "what will change" while hiding "what won't, and
    why", and the second question is the one an operator is actually asking when a run is stuck.
    """
    items: list[SettleItem] = []
    for fv in report.families:
        decision = _settle_decision(fv)
        if decision is not None:
            note = f"{fv.registry_status} -> {decision.status} ({decision.reason})"
        elif (fv.registry_status or "") in _TERMINAL:
            note = f"already {fv.registry_status or 'terminal'}, untouched"
        else:
            note = f"left alone: verdict {fv.verdict}" + (f" — {fv.detail}" if fv.detail else "")
        items.append(
            SettleItem(
                family=fv.family,
                runtime=fv.runtime,
                registry_status=fv.registry_status,
                verdict=fv.verdict,
                native_state=fv.native_state,
                n_done=fv.n_done,
                n_expected=fv.n_expected,
                decision=decision,
                note=note,
            )
        )
    return SettlePlan(run_id=report.run_id, header_status=report.status, items=tuple(items))


def _build_settle_audit(
    item: SettleItem,
    *,
    actor: str | None,
    settled_at: datetime,
    reason: str,
) -> dict[str, Any]:
    """The audit blob merged under ``run_jobs.job_telemetry.$.settle`` (pure).

    A settled row is the one row in the registry whose terminal status was written by somebody who
    was not running the job, so the blob records both the intent (*who / when / why*) and the whole
    of the evidence it was written on (*the verdict, the runtime's reading, the cell counts, and
    the status it replaced*). That is enough for a reader a week later to re-derive the decision
    and disagree with it — which is the only real check on a verb like this.
    """
    decision = item.decision
    return {
        "settled_by": actor,
        "settled_at": settled_at.isoformat(),
        "from_status": item.registry_status,
        "verdict": item.verdict,
        "native_state": item.native_state,
        "n_done": item.n_done,
        "n_expected": item.n_expected,
        "reason": reason or (decision.reason if decision else ""),
    }


@dataclass(frozen=True)
class _SettleTarget:
    """One family a confirmed settle will write: its plan line and the row carrying the job id."""

    item: SettleItem
    row: dict[str, Any]


def _settle_steps(plan: SettlePlan, rows: Sequence[Mapping[str, Any]]) -> tuple[_SettleTarget, ...]:
    """Join the plan to the run's job rows → what to write, in plan order (pure).

    The same two-collection reconciliation `cancel._cancel_steps` does and for the same reason: the
    plan comes from the run's *families* and the rows from ``v_run_jobs``, and only a join can say
    which plan line owns which ``job_id``. Simpler than the cancel join because settle addresses no
    runtime — it needs the row's id, not a handle, so a row whose ``probe_handle`` never parsed can
    still be repaired. A settleable family with no job row is skipped: there is no row to write.
    """
    rows_by_family = {r["family"]: dict(r) for r in rows}
    steps: list[_SettleTarget] = []
    for item in plan.items:
        if not item.settleable:
            continue
        row = rows_by_family.get(item.family)
        if row is None:
            continue
        steps.append(_SettleTarget(item=item, row=row))
    return tuple(steps)


def _header_hint(plan: SettlePlan, statuses: Mapping[str, str | None]) -> str:
    """The follow-up line for the run header settle deliberately did not touch (pure).

    Settling job rows is what *unblocks* `registry.ops.close_runs` — which refuses any run with a
    non-terminal job row — so the useful thing to say once the rows are all terminal is "now close
    the header", and the useful thing to say when they are not is which families are still open.
    An already-terminal header needs no hint at all.
    """
    header = plan.header_status
    if (header or "") in _TERMINAL:
        return ""
    still_open = [i.family for i in plan.items if (statuses.get(i.family) or "") not in _TERMINAL]
    if still_open:
        return f"header left at {header}; still non-terminal: {', '.join(still_open)}"
    return f"header left at {header}; every job row is terminal now — close_runs can close it"


def format_settle_plan(plan: SettlePlan) -> str:
    """Render a `SettlePlan` as the operator-facing preview — the exact text a dry run prints."""
    lines = [f"settle run {plan.run_id} (header {plan.header_status or 'unknown'})"]
    for item in plan.items:
        mark = "  ->" if item.settleable else "    "
        lines.append(f"{mark} {item.family:<16} {item.note}")
    lines.append(f"  {plan.n_settleable} of {len(plan.items)} job row(s) would be settled")
    return "\n".join(lines)


# --- settle I/O orchestrator ---------------------------------------------------


def settle_run(
    run_id: str,
    *,
    job: str | None = None,
    yes: bool = False,
    reason: str = "",
    actor: str | None = None,
    settings: Settings | None = None,
    stale_after_s: float | None = None,
    abandoned_after_s: float | None = None,
) -> SettleReport:  # pragma: no cover - GCP I/O
    """Repair a run's stale job rows from the probe's own verdicts — preview unless ``yes``.

    Probes the run (`reconcile._read_and_probe`), builds the `SettlePlan`, and — **only when
    ``yes``** — writes each settleable family's row: the decided status, the ``failure_reason``
    token when there is one, and the audit blob merged under ``job_telemetry.$.settle``. Without
    ``yes`` it returns the plan and writes nothing. ``job`` narrows to one family; ``reason`` and
    ``actor`` (default: resolved from ADC) are recorded in the audit. ``stale_after_s`` and
    ``abandoned_after_s`` pass through to the probe, where they set the two clocks that decide the
    ``LOST`` and ``ABANDONED_WAIT`` verdicts respectively.

    Every write carries a status guard against `_SETTLE_GUARD`, so a job that reached a terminal
    status of its own between the probe read and the UPDATE keeps it — the process that ran the
    job outranks a repair verb. That skip is silent, so the rows are re-read once afterwards and
    each outcome reports the status it actually ended up with.
    """
    from ..identity import resolve_principal
    from ..registry.jobs import read_run_jobs, update_job
    from ..settings import Settings

    s = settings if settings is not None else Settings.resolve()
    _progress, report, rows = _read_and_probe(
        run_id,
        job=job,
        settings=s,
        stale_after_s=stale_after_s,
        abandoned_after_s=abandoned_after_s,
    )
    plan = _assemble_settle_plan(report)
    if not yes:
        return SettleReport(
            run_id=run_id,
            plan=plan,
            executed=False,
            outcomes=(),
            header_hint="",
            actor=None,
            reason=reason,
        )

    resolved_actor = actor if actor is not None else resolve_principal(s)
    settled_at = datetime.now(UTC)
    written: list[tuple[_SettleTarget, SettleDecision]] = []
    for step in _settle_steps(plan, rows):
        decision = step.item.decision
        if decision is None:  # `_settle_steps` yields settleable items only; belt-and-braces
            continue
        fields: dict[str, Any] = {"status": decision.status}
        if decision.failure_reason is not None:
            fields["failure_reason"] = decision.failure_reason
        update_job(
            step.row["job_id"],
            settings=s,
            merge_telemetry={
                "settle": _build_settle_audit(
                    step.item, actor=resolved_actor, settled_at=settled_at, reason=reason
                )
            },
            unless_status_in=_SETTLE_GUARD,
            **fields,
        )
        written.append((step, decision))

    # Re-read rather than assume: a guarded skip is silent, so the plan's intent is not evidence.
    after = {r["family"]: r.get("status") for r in read_run_jobs(run_id, settings=s)}
    outcomes = tuple(
        SettleOutcome(
            family=step.item.family,
            job_key=step.row["job_id"],
            from_status=step.item.registry_status,
            to_status=decision.status,
            final_status=after.get(step.item.family),
            settled=after.get(step.item.family) == decision.status,
            detail=decision.reason,
        )
        for step, decision in written
    )
    hint = _header_hint(plan, after)
    return SettleReport(
        run_id=run_id,
        plan=plan,
        executed=True,
        outcomes=outcomes,
        header_hint=hint,
        actor=resolved_actor,
        reason=reason,
    )
