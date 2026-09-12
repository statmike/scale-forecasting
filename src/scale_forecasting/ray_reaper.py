"""Reclaim Ray clusters whose run is over — the backstop for a launcher that never cleaned up.

Every Ray cluster this product creates is torn down in a ``finally`` by the process that created it.
That is not a guarantee, it is a hope that nothing kills the process: ``kill -9``, a preempted VM, a
Composer worker eviction, an OOM and a laptop that sleeps all defeat it, and each one leaves a head
node plus workers — possibly an accelerator — billing against a run that no longer exists. This was
not theoretical. A dev machine restarted mid-provision on 2026-09-12 and left a head node, a CPU
worker and a T4 running for twenty-five minutes with nobody watching.

**The Dataproc path does not have this problem**, which is what makes the gap worth a module.
`dataproc_cluster.build_lifecycle_config` attaches an idle TTL and a max age to every cluster we
build, so the identical kill leaves something the *service* reclaims without us. Vertex's
``PersistentResource`` has no such field to set — there is no TTL, no idle timeout, no auto-delete
anywhere on the resource — so the only mechanism available is to come back later and look. That is
this module.

**What makes "later" safe is that garbage is decidable, not guessed at.** Three independent facts
have to line up before anything is deleted:

* the cluster carries our ``app`` label **and** a ``registry`` label naming *this* registry
  (`ray_cluster.cluster_labels`), so a second deployment sharing the project is invisible to us;
* its name starts with ``sf-ray-`` (`engines.ray_io.EPHEMERAL_PREFIX`), which every run-derived
  ephemeral cluster has and nothing else is checked against;
* the ``run_id`` embedded in that name resolves to a run header that is **terminal**, or to no
  header at all *and* the cluster is older than `DEFAULT_MIN_AGE_SECONDS`.

**A standing reuse target is safe because of the first fact, not the second.** The reuse path skips
create *and* teardown (`ray_submit.submit_ray`), so a cluster an operator provisioned themselves was
never labelled by us and `ray_cluster.list_clusters` does not even return it — which holds however
they chose to name it. The name check is the second line, for a labelled cluster of ours that a
config then pointed a reuse run at.

The age floor exists for exactly one race: a cluster that has just been created by a run whose
header is not yet written reads as "unknown run" for a few seconds. It applies only to the no-header
case — a cluster whose run is already ``COMPLETED`` is garbage the moment it is seen, however young,
because the only thing that could have produced it is a teardown that did not happen.

Name-to-run matching is by **prefix**, not equality, because `ray_io.cluster_name` clamps to 63
characters and a long enough ``run_name`` loses the tail of its id. A prefix that matches several
runs is treated as matching all of them: one live match keeps the cluster. Ambiguity resolves toward
leaving a machine running, never toward deleting one.

Preview is the default and ``yes=True`` executes, the same shape every destructive verb in
`registry.ops` uses — and the plan prints the kept clusters as prominently as the doomed ones,
because "why is this one still here?" is the question an operator actually arrives with.

Split along the usual pure/I-O seam: `classify_clusters` and `format_reap_plan` are pure and decide
everything, while `plan_reap_clusters` and `reap_clusters` only fetch and act.

Public surface: ``reap_clusters``, ``plan_reap_clusters``, ``classify_clusters``,
``format_reap_plan``, ``ReapPlan``, ``ReapCandidate``, ``DEFAULT_MIN_AGE_SECONDS``. Reachable as
``registry-ops reap-clusters`` and as `sdk.Registry.reap_clusters`; the Vertex listing and the
delete it drives are `ray_cluster.list_clusters` / `ray_cluster.teardown_shared_cluster`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .engines.ray_io import run_id_prefix_from_cluster_name
from .errors import get_logger
from .ray_cluster import REGISTRY_LABEL_KEY, RayCluster
from .registry.ops import LIVE_STATUSES

if TYPE_CHECKING:
    from datetime import datetime

    from .settings import Settings

_log = get_logger(__name__)

#: How old a cluster with no run header must be before it counts as garbage. Matches
#: `batch_infra._DEFAULT_CLUSTER_IDLE_TTL_SECONDS`, the Dataproc idle TTL, so both compute services
#: abandon a cluster on the same clock — and comfortably longer than the gap between a run creating
#: its cluster and its header landing in BigQuery.
DEFAULT_MIN_AGE_SECONDS = 1800.0

#: Vertex is already taking this one down; deleting it again achieves nothing and logs a scary
#: error.
_GOING_AWAY = "STOPPING"


@dataclass(frozen=True)
class ReapCandidate:
    """One cluster the reaper looked at, and why it lived or died.

    ``matched_runs`` is every run id in the registry whose name the cluster's embedded id is a
    prefix of — usually one, empty when the run is unknown, and occasionally several when a clamped
    name is ambiguous. ``reason`` is written for an operator reading a preview, so it names the
    run and the status rather than restating the rule.
    """

    cluster: RayCluster
    run_id_prefix: str | None
    matched_runs: tuple[str, ...]
    reap: bool
    reason: str


@dataclass(frozen=True)
class ReapPlan:
    """Exactly which Ray clusters `reap_clusters` would delete, and what it is leaving alone."""

    registry: str
    regions: tuple[str, ...] = ()
    candidates: tuple[ReapCandidate, ...] = ()
    min_age_seconds: float = DEFAULT_MIN_AGE_SECONDS

    @property
    def reapable(self) -> tuple[ReapCandidate, ...]:
        return tuple(c for c in self.candidates if c.reap)

    @property
    def kept(self) -> tuple[ReapCandidate, ...]:
        return tuple(c for c in self.candidates if not c.reap)

    @property
    def is_empty(self) -> bool:
        return not self.reapable


# --- pure: the policy -------------------------------------------------------------


def _age_seconds(cluster: RayCluster, now: datetime) -> float | None:
    """How many seconds old the cluster is, or ``None`` if Vertex gave us no create time."""
    if cluster.create_time is None:
        return None
    return (now - cluster.create_time).total_seconds()


def _verdict(
    matched: tuple[str, ...],
    status_by_run: Mapping[str, str | None],
    *,
    age: float | None,
    min_age_seconds: float,
) -> tuple[bool, str]:
    """``(reap?, reason)`` for one cluster that has already passed the label and name checks."""
    live = tuple(r for r in matched if (status_by_run.get(r) or "").upper() in LIVE_STATUSES)
    if live:
        statuses = ", ".join(f"{r} is {status_by_run.get(r)}" for r in live)
        return False, f"still in flight: {statuses}"
    if matched:
        statuses = ", ".join(f"{r} is {status_by_run.get(r)}" for r in matched)
        return True, f"run finished but the cluster is still up: {statuses}"
    if age is None:
        return False, "no run header, and Vertex reported no create time — cannot age it"
    if age < min_age_seconds:
        return False, (
            f"no run header yet, but only {age:.0f}s old "
            f"(under the {min_age_seconds:.0f}s floor — the run may still be starting)"
        )
    return True, f"no run header in this registry, and {age / 60:.0f} minutes old"


def classify_clusters(
    clusters: Sequence[RayCluster],
    status_by_run: Mapping[str, str | None],
    *,
    registry_dataset_id: str,
    now: datetime,
    min_age_seconds: float = DEFAULT_MIN_AGE_SECONDS,
) -> tuple[ReapCandidate, ...]:
    """Decide each cluster's fate (pure). Every cluster comes back with a verdict and a reason.

    The checks run cheapest-and-most-exclusionary first — registry label, then ephemeral name, then
    the ``STOPPING`` short-circuit, then the run header — so a cluster that is not ours exits before
    anything has to reason about run ids. ``status_by_run`` is the whole registry's latest header
    per run; a run id missing from it is a run this registry has never heard of.
    """
    out: list[ReapCandidate] = []
    for cluster in clusters:
        prefix = run_id_prefix_from_cluster_name(cluster.name)
        matched = tuple(sorted(r for r in status_by_run if prefix and r.startswith(prefix)))

        owner = cluster.labels.get(REGISTRY_LABEL_KEY)
        if owner is not None and owner != registry_dataset_id:
            reap, reason = False, f"belongs to registry '{owner}', not this one"
        elif prefix is None:
            reap, reason = False, "not a run-derived cluster — a reuse target we must never delete"
        elif cluster.state == _GOING_AWAY:
            reap, reason = False, "already STOPPING"
        else:
            reap, reason = _verdict(
                matched,
                status_by_run,
                age=_age_seconds(cluster, now),
                min_age_seconds=min_age_seconds,
            )
        out.append(
            ReapCandidate(
                cluster=cluster,
                run_id_prefix=prefix,
                matched_runs=matched,
                reap=reap,
                reason=reason,
            )
        )
    return tuple(out)


def format_reap_plan(plan: ReapPlan) -> str:
    """Render a `ReapPlan` as the operator-facing preview — the exact text a dry run prints.

    Kept clusters are listed too, and by design: a reaper that prints only what it will delete
    cannot answer "so why is that T4 still running?", which is the question that brings an operator
    here in the first place.
    """
    lines = [f"reap-clusters against registry {plan.registry}"]
    lines.append(f"  regions: {', '.join(plan.regions) or '(none)'}")
    lines.append(f"  unknown-run age floor: {plan.min_age_seconds:.0f}s")
    if plan.reapable:
        lines.append(f"  will DELETE ({len(plan.reapable)}):")
        for c in plan.reapable:
            lines.append(f"    {c.cluster.name}  [{c.cluster.region} {c.cluster.state}]")
            lines.append(f"      {c.reason}")
    if plan.kept:
        lines.append(f"  leaving alone ({len(plan.kept)}):")
        for c in plan.kept:
            lines.append(f"    {c.cluster.name}  [{c.cluster.region} {c.cluster.state}]")
            lines.append(f"      {c.reason}")
    if not plan.candidates:
        lines.append("  no scale-forecasting Ray clusters found")
    return "\n".join(lines)


# --- I/O: the verb ----------------------------------------------------------------


def _settings(settings: Settings | None) -> Settings:
    """The passed settings, or a fresh resolve from the ``SF_*`` environment."""
    from .settings import Settings as _Settings

    return settings if settings is not None else _Settings.resolve()


def plan_reap_clusters(
    *,
    settings: Settings | None = None,
    regions: Sequence[str] | None = None,
    min_age_seconds: float = DEFAULT_MIN_AGE_SECONDS,
) -> ReapPlan:  # pragma: no cover - GCP I/O, @gcp smoke
    """Which Ray clusters `reap_clusters` would delete — without deleting anything.

    ``regions`` defaults to the data-plane region alone. Widen it when a run's capacity walk may
    have hopped the cluster elsewhere; a region left off the list is one the sweep cannot clean,
    and `ray_cluster.list_clusters` raises rather than quietly skipping one it cannot read.
    """
    from datetime import UTC, datetime

    from .ray_cluster import list_clusters
    from .registry.ops import all_header_statuses

    resolved = _settings(settings)
    where = tuple(regions or (resolved.region,))
    clusters = list_clusters(resolved, where)
    # Read the headers only when there is something to judge — the usual answer is zero clusters,
    # and a preview that costs a BigQuery query every time is a preview nobody runs on a schedule.
    statuses = all_header_statuses(resolved) if clusters else {}
    return ReapPlan(
        registry=resolved.registry_dataset_ref,
        regions=where,
        candidates=classify_clusters(
            clusters,
            statuses,
            registry_dataset_id=resolved.registry_dataset_id,
            now=datetime.now(UTC),
            min_age_seconds=min_age_seconds,
        ),
        min_age_seconds=min_age_seconds,
    )


def reap_clusters(
    *,
    settings: Settings | None = None,
    regions: Sequence[str] | None = None,
    min_age_seconds: float = DEFAULT_MIN_AGE_SECONDS,
    yes: bool = False,
) -> ReapPlan:  # pragma: no cover - GCP I/O, @gcp smoke
    """Delete the Ray clusters whose run is over. Preview by default; ``yes=True`` executes.

    Deletes through `ray_cluster.teardown_shared_cluster`, which is the same verified teardown a
    normal run uses: the SDK's success line is ignored and the resource is polled until it reads
    ``NOT_FOUND``. A cluster that refuses to go is logged by name as still billing rather than
    counted as reclaimed — a reaper that reports a delete it did not achieve is worse than none at
    all, because it also removes the operator's reason to go and look.

    Every cluster is attempted even if one delete fails, for the reason
    `shared_clusters.teardown_spark_clusters` gives: stopping at the first failure strands all of
    the ones behind it, which is precisely the outcome this module exists to prevent.
    """
    from .ray_cluster import teardown_shared_cluster

    resolved = _settings(settings)
    plan = plan_reap_clusters(settings=resolved, regions=regions, min_age_seconds=min_age_seconds)
    _log.warning("%s", format_reap_plan(plan))
    if plan.is_empty:
        _log.warning("nothing to reap in %s", plan.registry)
        return plan
    if not yes:
        _log.warning("DRY RUN — no cluster deleted. Re-run with yes=True to execute.")
        return plan

    for candidate in plan.reapable:
        cluster = candidate.cluster
        try:
            teardown_shared_cluster(cluster.name, cluster.region, resolved)
        except Exception as exc:  # noqa: BLE001 - one stuck delete must not strand the rest
            _log.warning(
                "could not delete Ray cluster %s in %s — it may still be billing: %r",
                cluster.name,
                cluster.region,
                exc,
            )
    _log.warning("reaped %d Ray cluster(s) in %s", len(plan.reapable), plan.registry)
    return plan
