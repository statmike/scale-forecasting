"""Assemble a failed run's repair, preview it, and — only on confirmation — submit it.

`retry_policy` is **pure**: cells and families in, verdicts out. This module is the I/O half that
makes those verdicts answerable — it reads what the run actually did, derives the cells that are
missing entirely, classifies the lot, narrows the result to what v1's submission grain can honour,
and hands the narrowed DAG to `job_launch.submit_retry`. The same pairing as `ensembler` /
`ensemble_run`: one file that decides and one that touches GCP, so the decision can be tested
without a project and the launch can be tested without a classifier.

**Preview is the default and confirmation is explicit.** ``retry_run(cfg)`` reads, classifies,
prints, and submits nothing; ``retry_run(cfg, confirm=True)`` submits. That asymmetry is the point
of the verb — a repair is a *re-submission* against a run that already has rows in it, and an
operator who has not yet seen the decision table has no basis for launching one.

**Where the numbers come from, and why the report says so.** Two of the three inputs are ordinary
registry reads:

* `registry.reads.read_cell_groups` — every cell that wrote something, grouped by the four facts
  the classifier reads, so a hundred-thousand-cell run arrives as a few dozen rows.
* `probes.reconcile.probe_run` + `registry.jobs.read_run_jobs` — the family axis: status, failure
  token, and the reconciled runtime verdict. This is what separates "this cell is missing" from
  "this cell has not happened *yet*", and it is worth the probe's cost because the registry alone
  can be stale — the exact condition a repair is most likely to be run under.

The third is not a registry read at all. A cell whose job died before reaching it wrote **no row**,
so the only way to count those is *expected minus observed*, and the expected side has to be
re-derived from the run's own source at the run's own snapshot
(`registry.header.snapshot_millis_for` → `engines.bigquery_sql.build_series_count_query`). Counting
the source as it is *today* would report a series that arrived after the run as a hole in it. When
the snapshot or the count cannot be read the plan says so in ``universe_source`` and reports zero
never-ran cells rather than guessing — an under-count an operator can see beats an over-count they
cannot.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from . import retry_policy
from .errors import get_logger
from .registry.ids import base_family, is_repair_family
from .retry_policy import (
    CellState,
    FamilyState,
    RetryTargets,
    Worklist,
    build_worklist,
    narrow_to_submittable,
)

if TYPE_CHECKING:
    from .config import RunConfig
    from .job_launch import RetryOutcome
    from .settings import Settings

_log = get_logger(__name__)

# The placeholder `ts_id` on a synthesized never-ran group. These cells have no registry row, so
# there is no example id to show; the token says that plainly instead of showing a misleading one.
NEVER_RAN = "<no row>"


@dataclass(frozen=True)
class RetryPlanRow:
    """One line of the decision table: a group of identical cells and the verdict they earned.

    ``n_cells`` is how many cells this line stands for and ``example_ts_id`` is one of them (or
    `NEVER_RAN` for the synthesized never-ran groups, which have no row to point at). Everything
    else is the classifier's input, printed beside its output so a reader can check the reasoning
    rather than trust it.
    """

    verdict: str
    model_type: str
    family: str | None
    cell_status: str | None
    error_class: str | None
    has_predictions: bool
    n_cells: int
    example_ts_id: str


@dataclass(frozen=True)
class RetryPlan:
    """What a repair of this run would do — the whole preview, and a pure value.

    ``counts`` is the verdict headline and ``rows`` the table behind it. ``models`` is what would
    actually be submitted and ``blocked`` the models the worklist wanted but v1's grain cannot
    reach (see `retry_policy.RetryTargets`) — both are printed, because a targeted count with no
    submission behind it is the one way this verb can mislead.

    ``expected_cells`` / ``observed_cells`` are the two sides of the never-ran arithmetic and
    ``universe_source`` is the sentence that says where the expected side came from. It is part of
    the plan rather than the printer because a report of missing work is only as trustworthy as its
    denominator, and a caller reading `RetryPlan` programmatically needs the caveat as much as a
    caller reading the text does.
    """

    run_id: str
    header_status: str | None
    counts: dict[str, int] = field(default_factory=dict)
    rows: tuple[RetryPlanRow, ...] = ()
    families: tuple[FamilyState, ...] = ()
    models: tuple[str, ...] = ()
    blocked: tuple[str, ...] = ()
    n_targets: int = 0
    expected_cells: int | None = None
    observed_cells: int = 0
    universe_source: str = ""

    @property
    def submittable(self) -> bool:
        """Is there anything a confirmed call would actually launch?"""
        return bool(self.models)


@dataclass(frozen=True)
class RetryReport:
    """A retry call's result — the preview (``executed=False``) or the submitted outcome.

    ``plan`` is always present and is identical in both modes: the preview an operator confirmed
    and the record of what was confirmed must not be able to disagree. ``outcome`` is
    `job_launch.RetryOutcome` once something was launched, ``None`` otherwise.
    """

    run_id: str
    plan: RetryPlan
    executed: bool
    outcome: RetryOutcome | None = None
    actor: str | None = None
    reason: str = ""


# --- assembling the plan (pure) ------------------------------------------------


def family_states(
    rows: Iterable[Mapping[str, Any]], verdicts: Mapping[str, str] | None = None
) -> dict[str, FamilyState]:
    """Join ``v_run_jobs`` rows to the probe's verdicts → ``{family: FamilyState}`` (pure).

    Two sources because neither is complete: the job row carries the status and the failure token,
    the probe carries the reconciled reading of whether that status is still true. A family the
    probe did not escalate simply has no verdict and its registry status stands alone.

    **A repair row folds onto the family it repairs, and wins.** After one ``--retry`` the run holds
    a ``statistical`` row *and* a ``statistical_repair`` row (`registry.ids.REPAIR_JOB_FAMILIES`),
    and the classifier keys on the family a *cell* belongs to, which is always the base one. Keyed
    raw, a second ``--retry`` fired while the first repair is still in flight would find only the
    base family's stale ``FAILED`` and submit the same cells again — duplicating live work, which is
    the one thing the family axis exists to prevent. Folded, the in-flight repair is what the base
    family reads as, and those cells come back ``SKIP_NOT_FINISHED``.

    Rows arrive newest-family-last in no guaranteed order, so the win is explicit rather than
    positional: a repair row overwrites a base row, and a base row never overwrites a repair.
    """
    verdicts = verdicts or {}
    states: dict[str, FamilyState] = {}
    for row in rows:
        token = str(row.get("family") or "")
        if not token:
            continue
        family = base_family(token)
        if family in states and not is_repair_family(token):
            continue
        reason = row.get("failure_reason")
        states[family] = FamilyState(
            family=family,
            status=(str(row["status"]) if row.get("status") else None),
            failure_reason=(str(reason).strip() or None if reason else None),
            probe_verdict=verdicts.get(token),
        )
    return states


def assemble_cell_states(
    groups: Iterable[Mapping[str, Any]],
    *,
    family_of: Mapping[str, str],
    expected_series: int | None = None,
) -> tuple[CellState, ...]:
    """Turn grouped registry rows (+ the never-ran remainder) into `CellState` values (pure).

    Each group row becomes one weighted `CellState`. Then, for every model the run planned, the
    shortfall between ``expected_series`` and the cells that model actually wrote becomes a second
    synthesized state with ``has_metadata`` false — the cells whose job died before reaching them,
    which exist nowhere in the registry and are the whole reason a repair is worth running. A model
    that wrote *more* cells than expected (a re-run's extra attempt, a source that grew) contributes
    no shortfall rather than a negative one, and ``expected_series=None`` skips the synthesis
    entirely.

    ``family_of`` maps model → family (`dag.group_models_by_family`, inverted). A model missing from
    it gets ``family=None``, which costs the family axis for that model and nothing else.
    """
    states: list[CellState] = []
    observed: dict[str, int] = {}
    for row in groups:
        model = str(row["model_type"])
        n = int(row.get("n_cells") or 0)
        observed[model] = observed.get(model, 0) + n
        states.append(
            CellState(
                ts_id=str(row.get("example_ts_id") or NEVER_RAN),
                model_type=model,
                has_metadata=True,
                has_predictions=bool(row.get("has_predictions")),
                cell_status=(str(row["cell_status"]) if row.get("cell_status") else None),
                error_class=(str(row["error_class"]) if row.get("error_class") else None),
                family=family_of.get(model),
                n_cells=n,
            )
        )
    if expected_series is None:
        return tuple(states)
    for model in family_of:
        missing = expected_series - observed.get(model, 0)
        if missing > 0:
            states.append(
                CellState(
                    ts_id=NEVER_RAN,
                    model_type=model,
                    family=family_of.get(model),
                    n_cells=missing,
                )
            )
    return tuple(states)


def _plan_rows(worklist: Worklist) -> tuple[RetryPlanRow, ...]:
    """Flatten a `Worklist` into the printable decision table, in verdict order (pure)."""
    rows: list[RetryPlanRow] = []
    for verdict in retry_policy.VERDICTS:
        for cell in worklist.by_verdict.get(verdict, ()):
            rows.append(
                RetryPlanRow(
                    verdict=verdict,
                    model_type=cell.model_type,
                    family=cell.family,
                    cell_status=cell.cell_status,
                    error_class=cell.error_class,
                    has_predictions=cell.has_predictions,
                    n_cells=cell.n_cells,
                    example_ts_id=cell.ts_id,
                )
            )
    return tuple(rows)


def assemble_retry_plan(
    run_id: str,
    *,
    header_status: str | None,
    states: Sequence[CellState],
    families: Mapping[str, FamilyState],
    landed_counts: Mapping[str, int],
    expected_series: int | None,
    universe_source: str,
) -> RetryPlan:
    """Classify the assembled states and narrow them to a submittable plan (pure).

    The whole decision, in one call and with no I/O: `build_worklist` files every cell under its
    verdict, `narrow_to_submittable` splits the resulting model wish-list into what v1 can re-ask
    and what it cannot, and the counts are carried through weighted so the headline is in cells
    rather than in grouped rows.
    """
    worklist = build_worklist(states, families)
    targets: RetryTargets = narrow_to_submittable(worklist, landed_counts)
    observed = sum(c.n_cells for c in states if c.has_metadata)
    return RetryPlan(
        run_id=run_id,
        header_status=header_status,
        counts=worklist.counts,
        rows=_plan_rows(worklist),
        families=tuple(families[f] for f in sorted(families)),
        models=targets.models,
        blocked=targets.blocked,
        n_targets=worklist.n_targets,
        expected_cells=expected_series,
        observed_cells=observed,
        universe_source=universe_source,
    )


def format_retry_plan(plan: RetryPlan) -> str:
    """Render a `RetryPlan` as the operator-facing block (pure; the one rendering of these fields).

    Printed identically in preview and after execution, so the thing an operator confirmed and the
    record of what was confirmed cannot drift.
    """
    lines = [
        f"Retry run {plan.run_id} (header={plan.header_status or '-'}): "
        f"{plan.n_targets} cell(s) on the worklist",
        f"  universe: expected={plan.expected_cells if plan.expected_cells is not None else '?'} "
        f"series/model, observed={plan.observed_cells} cell(s) — {plan.universe_source}",
    ]
    if plan.counts:
        lines.append("  verdicts: " + ", ".join(f"{v}={n}" for v, n in sorted(plan.counts.items())))
    row_fmt = "  %-22s %-14s %-14s %-16s %8s  %s"
    lines.append(row_fmt % ("verdict", "model", "family", "error_class", "cells", "example"))
    for row in plan.rows:
        lines.append(
            row_fmt
            % (
                row.verdict,
                row.model_type,
                row.family or "-",
                row.error_class or (row.cell_status or "-"),
                row.n_cells,
                row.example_ts_id,
            )
        )
    lines.append(f"  would submit: {', '.join(plan.models) if plan.models else '(nothing)'}")
    if plan.blocked:
        lines.append(
            f"  NOT submittable: {', '.join(plan.blocked)} — these models already have landed "
            "predictions, and v1 re-submits a whole model over the whole series universe, so a "
            "repair would append a duplicate beside every finished cell. Re-run under a new "
            "config (a new run_id) to redo them."
        )
    return "\n".join(lines)


# --- the I/O half --------------------------------------------------------------


def _retry_audit(
    plan: RetryPlan, *, actor: str | None, at: datetime, reason: str
) -> dict[str, Any]:
    """The blob merged under ``run_jobs.job_telemetry.$.retry`` (pure).

    A repaired job is one whose successor attempt exists because somebody looked at a decision
    table and agreed with it, so the row records both the intent (*who / when / why*) and the
    evidence (*the verdict counts, the denominator, what was submitted and what was refused*) —
    enough for a reader later to re-derive the call and disagree with it, the same standard
    `probes.settle._build_settle_audit` is held to.
    """
    return {
        "retried_by": actor,
        "retried_at": at.isoformat(),
        "reason": reason,
        "counts": dict(plan.counts),
        "n_targets": plan.n_targets,
        "models": list(plan.models),
        "blocked": list(plan.blocked),
        "expected_cells": plan.expected_cells,
        "observed_cells": plan.observed_cells,
        "universe_source": plan.universe_source,
    }


def _expected_series(
    cfg: RunConfig, run_id: str, settings: Settings | None
) -> tuple[int | None, str]:  # pragma: no cover - GCP I/O
    """Count the run's series universe at the run's own snapshot → ``(count, provenance)``.

    Never raises: a repair that cannot read the denominator should report a smaller worklist and
    say why, not fail. Both failure modes return ``(None, <why>)`` and the plan then contains no
    never-ran cells at all.
    """
    from google.cloud import bigquery

    from .engines.bigquery_sql import build_series_count_query
    from .registry.header import snapshot_millis_for
    from .registry.tables import _resolve_settings

    resolved = _resolve_settings(settings)
    try:
        millis = snapshot_millis_for(run_id, settings=resolved)
    except Exception as exc:  # noqa: BLE001 - a missing denominator is reported, not raised
        return None, f"snapshot unreadable ({exc}); never-ran cells NOT counted"
    if millis is None:
        return None, "run header carries no input snapshot; never-ran cells NOT counted"
    sql = build_series_count_query(cfg, resolved.dataset_ref, snapshot_millis=millis)
    try:
        rows = list(bigquery.Client(project=resolved.project_id).query(sql).result())
        n = int(rows[0]["n_series"])
    except Exception as exc:  # noqa: BLE001 - as above
        return None, f"source count failed ({exc}); never-ran cells NOT counted"
    return n, f"source table at the run's pinned snapshot ({millis})"


def build_retry_plan(
    cfg: RunConfig, *, settings: Settings | None = None
) -> RetryPlan:  # pragma: no cover - GCP I/O
    """Read everything a repair decision needs and return the plan. Reads only; writes nothing."""
    from .dag import group_models_by_family
    from .probes.reconcile import probe_run
    from .registry.ids import make_run_id
    from .registry.jobs import read_run_jobs
    from .registry.reads import read_cell_groups, read_prediction_counts

    run_id = make_run_id(cfg)
    by_family = group_models_by_family(cfg)
    family_of = {model: family for family, models in by_family.items() for model in models}

    probe = probe_run(run_id, settings=settings)
    families = family_states(
        read_run_jobs(run_id, settings=settings),
        {fv.family: fv.verdict for fv in probe.families},
    )
    expected, provenance = _expected_series(cfg, run_id, settings)
    states = assemble_cell_states(
        read_cell_groups(run_id, settings=settings),
        family_of=family_of,
        expected_series=expected,
    )
    return assemble_retry_plan(
        run_id,
        header_status=probe.status,
        states=states,
        families=families,
        landed_counts=read_prediction_counts(run_id, settings=settings),
        expected_series=expected,
        universe_source=provenance,
    )


def config_for_run(run_id: str, *, settings: Settings | None = None) -> RunConfig:
    """Load back the config a run landed under, so a repair can be planned from its ``run_id``.

    An operator who inherits a broken run has its id — from a pager, a dashboard, a ledger row —
    and often not its config file. The config is still the planning input (the DAG to narrow, the
    family each model belongs to, the source table and the series subset all come from it); this is
    a second way to *obtain* it, not a second way to plan. The run's header carries it, because the
    config **is** the experiment record (`registry.reads.read_run_config`).

    **It refuses loudly rather than planning small,** and that asymmetry is the whole design. Every
    failure here has a plausible-looking wrong answer available — an empty config plans an empty
    repair, a partially-valid one plans a subset — and a repair that quietly does less than the
    operator asked for is worse than one that does nothing, because nothing is visible. So all three
    raise `errors.ConfigError` with the id in the message:

    * **no stored config** — the run never wrote a header, or is not this deployment's run;
    * **a config that no longer validates** — the schema moved under a run written by older code, so
      what the file *meant* is no longer knowable and guessing at it is not a repair;
    * **a config whose own digest is not the id asked for** — `registry.ids.make_run_id` is pure, so
      this can only mean the row was edited or the digest rule changed. Planning from it would
      repair a *different* run under this one's name.
    """
    from .config import RunConfig
    from .errors import ConfigError
    from .registry.ids import make_run_id
    from .registry.reads import read_run_config

    raw = read_run_config(run_id, settings=settings)
    if raw is None:
        raise ConfigError(
            f"no stored config for run {run_id}: the registry has no header row carrying one. "
            "Pass the config file instead, or check the run id and the deployment."
        )
    try:
        cfg = RunConfig.model_validate(raw)
    except Exception as exc:  # pydantic ValidationError → the package's ConfigError
        raise ConfigError(
            f"the stored config for run {run_id} no longer validates: {exc}. It was written by a "
            "different version of this package; repairing from a config we cannot read would "
            "guess at what the run asked for."
        ) from exc
    resolved = make_run_id(cfg)
    if resolved != run_id:
        raise ConfigError(
            f"the stored config for run {run_id} digests to {resolved}. make_run_id is pure, so "
            "either the header row was edited or the digest rule has changed; planning from it "
            "would repair a different run under this one's name."
        )
    return cfg


def retry_run(
    cfg: RunConfig,
    *,
    confirm: bool = False,
    reason: str = "",
    actor: str | None = None,
    settings: Settings | None = None,
    max_executors: int | None = None,
) -> RetryReport:  # pragma: no cover - GCP I/O
    """Preview (default) or submit (``confirm=True``) the repair of ``cfg``'s run.

    Without ``confirm`` this is a pure read: it returns the plan and launches nothing. With it, the
    run's DAG is narrowed to the submittable models (`dag.narrow_to_models` — which also drops the
    ensemble node, since a partial re-blend is a different question) and handed to
    `job_launch.submit_retry`, which forces a fresh attempt number on every family it touches.

    The audit blob lands on the **repair job's own rows** — ``narrow_to_models`` files each of them
    under a repair family token (``statistical_repair``), so ``read_run_jobs`` returns them beside
    the families they repair rather than in place of them. That is the right row: the record answers
    "why does this job exist", and the attempt it was launched to fix keeps its own row and its own
    failure reason, which is what ``run_jobs`` being append-only is for.
    """
    # Resolve once, up front, exactly as the sibling verbs do (`probes.cancel.cancel_run`,
    # `probes.settle.settle_run`, `probes.reconcile`). Deferring it is what broke the launch on
    # 2026-09-11: every *read* below tolerates ``None`` because the registry helpers resolve it
    # themselves, so the preview looked perfect while the submit path handed ``None`` down to
    # `job_launch.launch_family_job` and died on ``settings.region``. Resolving here means the
    # preview and the launch are planned against the same settings object rather than two
    # independently resolved ones, which is the property that actually matters.
    from .settings import Settings  # module-level import is TYPE_CHECKING-only

    s = settings if settings is not None else Settings.resolve()

    plan = build_retry_plan(cfg, settings=s)
    if not confirm:
        return RetryReport(run_id=plan.run_id, plan=plan, executed=False, reason=reason)
    if not plan.submittable:
        _log.info("retry %s: nothing submittable; not launching", plan.run_id)
        return RetryReport(run_id=plan.run_id, plan=plan, executed=False, reason=reason)

    # Imported here rather than at the top of the function: a preview is the default and it must
    # not pay for the launch stack it is not going to use.
    from .dag import narrow_to_models, plan_dag
    from .identity import resolve_principal
    from .job_launch import submit_retry
    from .registry.jobs import read_run_jobs, update_job

    retry_dag = narrow_to_models(plan_dag(cfg), plan.models)
    outcome = submit_retry(cfg, retry_dag, plan.run_id, s, max_executors=max_executors)

    resolved_actor = actor if actor is not None else resolve_principal(s)
    audit = _retry_audit(plan, actor=resolved_actor, at=datetime.now(UTC), reason=reason)
    repaired = set(retry_dag.families)
    for row in read_run_jobs(plan.run_id, settings=s):
        if row.get("family") in repaired and row.get("job_id"):
            update_job(row["job_id"], settings=s, merge_telemetry={"retry": audit})
    return RetryReport(
        run_id=plan.run_id,
        plan=plan,
        executed=True,
        outcome=outcome,
        actor=resolved_actor,
        reason=reason,
    )
