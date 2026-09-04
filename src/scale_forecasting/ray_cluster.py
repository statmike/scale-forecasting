"""The Vertex Ray cluster — plan it into a spec, provision it, find it, tear it down.

Everything that talks to Vertex's ``vertex_ray`` control plane about a *cluster* (as opposed to a
job running on one, which is `ray_jobs`). The Ray sibling of `dataproc_cluster`, and it carries the
same two responsibilities that module does: turn a ``RayClusterPlan`` into the SDK's resource shape,
and own the resource's whole lifetime.

The interesting part is the **region fallback**. GPU capacity and accelerator quota are both granted
per-region, so a create that fails for either reason says nothing about the next region's odds —
`_create_cluster_across_regions` walks the configured regions until one provisions. The walk itself
(classification, back-off, both budgets, the attempt ledger) is not here: it is
`scale_forecasting.capacity`, shared with the Dataproc path so both cluster services answer to one
rule and one vocabulary. This module supplies the two things only Vertex knows — how to attempt a
region, and where the reason for a failure actually lives.

That second one is the whole difficulty. The SDK's exception is a generic "Cluster … returned an
error" while the actionable text sits on the ``PersistentResource.error`` field
(`_cluster_error_message`), and that read has to hit the *regional* endpoint or the reason is
silently lost. So the walk is handed a ``describe_failure`` that reads the resource before tearing
it down, and classifies on that.

Vertex frequently fails a provision without saying why at all, which is why the shared classifier
treats an unrecognised message as transient rather than as a diagnosed fault (see `capacity`).

Only the cluster hops. The data plane — config staging, registry writes — stays pinned to
``settings.region``, which is why `cluster_resource_path` takes an explicit region rather than
assuming one.

The other thing worth knowing is that **teardown verifies rather than assumes**. The SDK reports
success the moment a delete is accepted, so a stuck resource and a deleted one produce the identical
log line; `_delete_cluster` polls the resource until it reads ``NOT_FOUND`` and says so plainly when
it does not. `_clear_stale_resource` is the other half — a run-derived name means a previous
attempt's ``ERROR``-state wreckage sits on the exact path the next create wants.

Public surface: ``cluster_resource_path``, ``provision_shared_cluster``,
``teardown_shared_cluster``. The lifecycle verbs `_create_cluster_across_regions` / `_get_cluster` /
`_delete_cluster` are driven by `ray_submit.submit_ray`, which owns the create→run→teardown
ordering for a single-family run.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .capacity import (
    DEFAULT_POLICIES,
    HARD_CEILING,
    CapacityExhausted,
    CapacityLedger,
    CapacityPolicy,
    current_publisher,
)
from .capacity import walk as capacity_walk
from .engines import ray_io
from .errors import get_logger
from .ray_infra import RayInfra

if TYPE_CHECKING:
    from .config import RunConfig
    from .settings import Settings

_log = get_logger(__name__)


def _worker_resources(plan: ray_io.RayClusterPlan) -> list[Any]:
    """Build the worker ``Resources`` list — one entry per non-empty pool.

    A GPU pool (``accelerator_type``/``accelerator_count``) for NeuralProphet and a CPU pool for
    everything else. When ``plan.autoscale`` (the default) each pool carries a Vertex
    ``AutoscalingSpec(min, max)`` from the plan's resolved per-pool bounds and Ray grows/shrinks it
    with task demand — note the SDK ignores ``node_count`` here (the pool starts at ``min``), but we
    still pass the derived count as the documented fixed-size-equivalent. When ``autoscale``
    is False both pools are fixed at their derived ``node_count`` with **no** ``autoscaling_spec``
    (a deterministic fixed-size path). A pool with zero planned nodes is omitted (Vertex rejects
    a zero-node worker type). No ``custom_image`` is set — Ray runs on Vertex's prebuilt image and
    the uv ``runtime_env`` delivers the deps (see `code_delivery.build_runtime_env`).
    """
    from google.cloud.aiplatform import vertex_ray
    from google.cloud.aiplatform.vertex_ray.util.resources import AutoscalingSpec

    def _spec(min_nodes: int, max_nodes: int) -> Any:
        return AutoscalingSpec(min_replica_count=min_nodes, max_replica_count=max_nodes)

    workers: list[Any] = []
    if plan.cpu_node_count > 0:
        workers.append(
            vertex_ray.Resources(
                machine_type=plan.cpu_machine_type,
                node_count=plan.cpu_node_count,
                autoscaling_spec=(
                    _spec(plan.cpu_min_nodes, plan.cpu_max_nodes) if plan.autoscale else None
                ),
            )
        )
    if plan.gpu_node_count > 0:
        workers.append(
            vertex_ray.Resources(
                machine_type=plan.gpu_machine_type,
                node_count=plan.gpu_node_count,
                accelerator_type=plan.accelerator_type,
                accelerator_count=plan.accelerator_count,
                autoscaling_spec=(
                    _spec(plan.gpu_min_nodes, plan.gpu_max_nodes) if plan.autoscale else None
                ),
            )
        )
    return workers


def _init_vertex(
    settings: Settings, region: str
) -> None:  # pragma: no cover - thin SDK call, live smoke covers
    """Pin the Vertex SDK to the configured project + region before a ``vertex_ray`` call.

    ``vertex_ray.create_ray_cluster`` (and the get/delete helpers) take no explicit project or
    location — they read them from the SDK's global config, which else falls back to the ambient
    ``GOOGLE_CLOUD_PROJECT`` / gcloud default. That would silently provision the cluster in the
    wrong project when the deployment's project differs from the environment's (Composer, local dev,
    any multi-project setup). Binding it from `Settings` here keeps the same code targeting
    the configured project everywhere — never whatever project the shell happens to point at.

    ``region`` is explicit (not ``settings.region``) because the *cluster* may hop across regions on
    a capacity stockout while the *data plane* stays pinned to ``settings.region`` — so every
    cluster-region-scoped call re-inits the SDK to the region actually being attempted.
    """
    from google.cloud import aiplatform

    aiplatform.init(project=settings.project_id, location=region)


def _resolve_regions(cfg: RunConfig, settings: Settings) -> list[str]:
    """Priority-ordered cluster regions to attempt (pure): configured list, else [settings.region].

    ``settings.region`` (the data-plane region) is always appended as a final fallback if it isn't
    already listed, so a config that lists only remote regions still ends up trying home.
    """
    regions = list(cfg.compute.ray_regions or [])
    if not regions:
        return [settings.region]
    if settings.region not in regions:
        regions.append(settings.region)
    return regions


def _create_cluster(
    plan: ray_io.RayClusterPlan, infra: RayInfra, name: str
) -> str:  # pragma: no cover - live Vertex I/O, exercised by the @gpu smoke
    """Create the Vertex Ray cluster (autoscaling per pool by default) and return its
    ``cluster_resource_name``.

    Head node is a single small CPU box (no accelerator, never autoscaled); workers are the planned
    GPU/CPU pools (`_worker_resources`), each with a Vertex ``AutoscalingSpec`` by default
    or a fixed ``node_count`` when ``ray_autoscale=False``. Labels tag the
    run.

    Connectivity follows `RayInfra`'s three modes (first set wins): a PSC-I network
    attachment (``psc_interface_config`` — the supported private path, the only mode whose managed
    dashboard/``JobSubmissionClient`` handshake is reachable off-cluster on this org), else VPC
    peering (``network=``), else a public endpoint (both unset — Vertex's default). PSC-I and
    ``network`` are mutually exclusive at the API, so we pass exactly one.
    """
    from google.cloud.aiplatform import vertex_ray

    head = vertex_ray.Resources(
        machine_type=plan.head_machine_type,
        node_count=1,
    )

    # PSC-I takes precedence over peering; only one of psc_interface_config / network may be set.
    psc_config = None
    network = infra.network
    if infra.network_attachment:
        from google.cloud.aiplatform.vertex_ray.util.resources import PscIConfig

        psc_config = PscIConfig(network_attachment=infra.network_attachment)
        network = None  # mutually exclusive — never pass both
        endpoint = "psc-i"
    elif infra.network:
        endpoint = "peering"
    else:
        endpoint = "public"

    _log.info(
        "creating Ray cluster %s: autoscale=%s cpu[min=%d,max=%d] gpu[min=%d,max=%d] "
        "cpu_nodes=%d gpu_nodes=%d accel=%s x%d endpoint=%s",
        name,
        plan.autoscale,
        plan.cpu_min_nodes,
        plan.cpu_max_nodes,
        plan.gpu_min_nodes,
        plan.gpu_max_nodes,
        plan.cpu_node_count,
        plan.gpu_node_count,
        plan.accelerator_type,
        plan.accelerator_count,
        endpoint,
    )
    return vertex_ray.create_ray_cluster(
        head_node_type=head,
        worker_node_types=_worker_resources(plan),
        cluster_name=name,
        network=network,
        psc_interface_config=psc_config,
        service_account=infra.compute_sa,
        ray_version=infra.ray_version,
        python_version=infra.python_version,
        labels={"app": "scale-forecasting"},
        # NOTE: no explicit location — the region is bound via _init_vertex before this call, which
        # is what the region-fallback loop re-pins per attempt.
    )


def _region_from_resource_name(resource_name: str) -> str | None:
    """Parse the region out of a ``.../locations/<region>/...`` resource path, or ``None``.

    Persistent-resource reads are *regional* — the service client must target
    ``<region>-aiplatform.googleapis.com``, so we recover the region from the resource name the
    create returned rather than assuming the data-plane region (the cluster may have hopped).
    """
    match = re.search(r"/locations/([^/]+)/", resource_name)
    return match.group(1) if match else None


def cluster_resource_path(settings: Settings, name: str, region: str | None = None) -> str:
    """The Vertex persistent-resource path for a cluster display name (reuse targeting; pure).

    ``region`` defaults to ``settings.region`` (the reuse case — a standing cluster lives in the
    data-plane region); the ephemeral fallback path passes the region actually being attempted.
    """
    loc = region or settings.region
    return f"projects/{settings.project_id}/locations/{loc}/persistentResources/{name}"


def _cluster_error_message(
    resource_name: str,
) -> str:  # pragma: no cover - live Vertex I/O, exercised by the @gpu smoke
    """The failed cluster's ``error.message`` (where the *real* reason lives), or ``""`` if none.

    A create that fails to reach RUNNING surfaces only a generic
    ``RuntimeError("Cluster ... returned an error.")`` from the SDK — the actionable text
    ("Resources are insufficient in region: …") is on the ``PersistentResource.error`` field, not
    in the exception. So the fallback classifier reads it here rather than trusting ``str(exc)``.

    The read must hit the resource's *regional* endpoint (``<region>-aiplatform.googleapis.com``) —
    against the global default the ``get`` fails and the reason is lost, which silently defeats the
    capacity classifier. Best-effort: any read failure returns ``""`` (the caller falls back to the
    exception string).
    """
    try:
        from google.cloud import aiplatform_v1

        region = _region_from_resource_name(resource_name)
        client_options = {"api_endpoint": f"{region}-aiplatform.googleapis.com"} if region else None
        client = aiplatform_v1.PersistentResourceServiceClient(client_options=client_options)
        pr = client.get_persistent_resource(name=resource_name)
        return pr.error.message or ""
    except Exception as exc:  # noqa: BLE001 - diagnostic read; never fatal
        _log.debug("could not read cluster error for %s: %r", resource_name, exc)
        return ""


def _attempt_cluster_in_region(
    plan: ray_io.RayClusterPlan, infra: RayInfra, name: str, settings: Settings, region: str
) -> tuple[str, str]:  # pragma: no cover - live Vertex I/O; @gpu smoke exercises it
    """One attempt: pin the SDK to ``region``, clear any wreckage, create. Raises on failure.

    Only the cluster hops — the data plane (config staging, registry writes) stays in
    ``settings.region``. `_init_vertex` re-pins the SDK to the region being attempted, so
    ``vertex_ray`` provisions there.
    """
    _init_vertex(settings, region)
    # The name is run-derived, so an earlier attempt's wreckage sits on this exact path and would
    # fail the create with AlreadyExists. Clear it first (see `_clear_stale_resource`). This is also
    # what makes a *second pass* over the same region possible at all.
    _clear_stale_resource(cluster_resource_path(settings, name, region))
    _log.info("attempting Ray cluster %s in region %s", name, region)
    resource_name = _create_cluster(plan, infra, name)
    _log.info("Ray cluster %s created in region %s", name, region)
    return resource_name, region


def _describe_region_failure(
    name: str, settings: Settings, region: str, exc: Exception
) -> str:  # pragma: no cover - live Vertex I/O; @gpu smoke exercises it
    """The richest text available for a failed create, then tear the failed resource down.

    Reads the resource's own ``error.message`` **before** teardown — that is where the reason lives;
    the SDK's exception is only a generic "returned an error", so classifying on it alone would
    never detect a stockout. A create that errors mid-provision still leaves a resource behind, and
    it has to go before this region can be attempted again on a later pass.
    """
    resource_path = cluster_resource_path(settings, name, region)
    detail = _cluster_error_message(resource_path)
    _delete_cluster(resource_path)
    return f"{exc} | {detail}".strip(" |")


def _apply_quota_preflight(
    plan: ray_io.RayClusterPlan,
    infra: RayInfra,
    settings: Settings,
    regions: list[str],
    ledger: CapacityLedger,
) -> dict[str, ray_io.RayClusterPlan]:
    """Rule out the regions that cannot work, and return the per-region plan for the ones that can.

    Two checks, cheapest first. `quota.regions_without_attachment` is a regex over a string and
    needs no network at all: a PSC-I network attachment is a *regional* resource and Terraform
    builds exactly one, so every region other than its home is unreachable no matter what its quota
    says. Only the survivors are worth an API read.

    Then three effects, which are the whole point of preflighting rather than discovering:

    * a region that cannot host the fleet is recorded in ``ledger`` as a `capacity.HARD_CEILING`
      with ``elapsed_seconds=0``, which both explains the omission in telemetry and puts the region
      into `CapacityLedger.dead_candidates` so the walk never revisits it;
    * a region that can host a smaller fleet gets a plan clamped to what its allowance grants,
      keyed by region because the answer genuinely differs between them (12 T4s in ``us-central1``,
      2 in ``us-east1``);
    * every non-trivial finding is logged, including the wall clock the ceiling implies.

    Returns ``{}`` and logs at debug if the read itself fails — a preflight is a diagnostic, and a
    diagnostic that can block a launch is worse than no diagnostic. The zero elapsed time on the
    recorded attempts is deliberate and honest: nothing was spent establishing them.
    """
    from . import quota

    # getattr, not attribute access: the preflight is a diagnostic layered over the launch path and
    # must degrade to "I know nothing" rather than raise on an infra object it did not expect.
    attachment = getattr(infra, "network_attachment", None)
    unreachable = quota.regions_without_attachment(attachment, regions)
    for region, reason in unreachable.items():
        _log.warning("preflight rules out %s without attempting a create: %s", region, reason)
        ledger.record(region, HARD_CEILING, f"preflight: {reason}", 0.0)
    candidates = [region for region in regions if region not in unreachable]
    if not candidates:
        return {}

    try:
        preflights = quota.preflight_ray(plan, candidates, settings.project_id)
    except Exception as exc:  # noqa: BLE001 - a diagnostic must never sink a launch
        _log.debug("quota preflight unavailable (non-fatal): %r", exc)
        return {}

    plans: dict[str, ray_io.RayClusterPlan] = {}
    for region, preflight in preflights.items():
        ledger.preflight.append(preflight.to_dict())
        for line in preflight.render():
            _log.info("%s", line)
        if preflight.blocked:
            _log.warning(
                "quota preflight rules out %s without attempting a create: %s",
                region,
                preflight.block_reason,
            )
            ledger.record(region, HARD_CEILING, f"quota preflight: {preflight.block_reason}", 0.0)
            continue
        plans[region] = quota.apply_to_ray_plan(plan, preflight)
    return plans


def _publish_ledger(
    on_state: Callable[[CapacityLedger], None] | None, ledger: CapacityLedger
) -> None:  # pragma: no cover - one-line delegation to a tested helper
    """Publish the ledger through the caller's channel, or the ambient one."""
    publish = on_state if on_state is not None else current_publisher()
    if publish is not None:
        publish(ledger)


def _create_cluster_across_regions(
    plan: ray_io.RayClusterPlan,
    infra: RayInfra,
    name: str,
    settings: Settings,
    regions: list[str],
    *,
    policy: CapacityPolicy | None = None,
    ledger: CapacityLedger | None = None,
    on_state: Callable[[CapacityLedger], None] | None = None,
    preflight: bool = True,
) -> tuple[str, str]:  # pragma: no cover - orchestrates live Vertex I/O; @gpu smoke exercises it
    """Create the cluster, walking ``regions`` until one can provision it.

    Returns ``(cluster_resource_name, region)`` for the region that succeeded.

    Before the first create, `_apply_quota_preflight` drops every region with no PSC-I network
    attachment (a regex, no API call) and reads the rest's allowance. Regions that cannot host the
    fleet are recorded as `capacity.HARD_CEILING` and never attempted, a region that can host a
    *smaller* fleet gets a plan clamped to what it will grant, and the throttle is logged in
    wall-clock terms. ``preflight=False`` (``compute.capacity.preflight``) skips both checks; the
    walk then behaves exactly as it did before they existed.

    The loop is `capacity.walk`, shared with the Dataproc path: try each live region with no wait
    between them, then back off and try them again, until an attempt or wall-clock budget runs out.
    A `capacity.HARD_CEILING` region (a quota this project is not allowed past) is dropped from
    later passes — waiting cannot raise a quota. A `capacity.CONFIG_FAULT` re-raises the original
    exception at once. Exhaustion raises `capacity.CapacityExhausted`, an `EngineError` carrying the
    ledger of every attempt.

    ``policy`` defaults to the shipped Ray patience; callers with a config pass
    ``cfg.compute.capacity.policy_for("ray")``. ``ledger`` is caller-owned so the attempt log
    survives *success* too — landing in ``us-east1`` after two stockouts is worth recording — and
    ``on_state`` publishes ``AWAITING_CAPACITY`` while the walk is still running, defaulting to
    whatever `capacity.publishing_to` installed for this family (`job_launch` does).
    """
    ledger = ledger if ledger is not None else CapacityLedger(service="ray")
    plans = _apply_quota_preflight(plan, infra, settings, regions, ledger) if preflight else {}
    live = [region for region in regions if region not in ledger.dead_candidates]
    if not live:
        ledger.exhausted = True
        _publish_ledger(on_state, ledger)
        raise CapacityExhausted(
            f"ray: preflight ruled out every candidate region "
            f"({', '.join(regions)}); no create was attempted",
            ledger,
        )

    return capacity_walk(
        live,
        lambda region: _attempt_cluster_in_region(
            plans.get(region, plan), infra, name, settings, region
        ),
        ledger=ledger,
        policy=policy or DEFAULT_POLICIES["ray"],
        describe_failure=lambda region, exc: _describe_region_failure(name, settings, region, exc),
        on_state=on_state if on_state is not None else current_publisher(),
    )


def _get_cluster(
    cluster_resource_name: str,
) -> Any:  # pragma: no cover - live Vertex I/O, exercised by the @gpu smoke
    """Fetch a Vertex Ray cluster by resource name (for its ``dashboard_address`` + telemetry)."""
    from google.cloud.aiplatform import vertex_ray

    return vertex_ray.get_ray_cluster(cluster_resource_name=cluster_resource_name)


# Two answers that end a wait: the resource is gone (what teardown is trying to establish), and the
# read could not be completed (the admission that we cannot establish it). Both are distinct from
# every real Vertex state name, so they can share the state channel without ambiguity.
_ABSENT = "NOT_FOUND"
_UNREADABLE = "UNREADABLE"

# How long to wait for a delete to actually take effect, and how often to ask. A Ray cluster takes
# a minute or two to disappear; five minutes is generous enough that a timeout means something.
_TEARDOWN_TIMEOUT_S = 300.0
_TEARDOWN_POLL_S = 10.0


def _resource_state(
    resource_name: str,
) -> str:  # pragma: no cover - live Vertex I/O, exercised by the @gpu smoke
    """The resource's state name, ``_ABSENT`` if it is gone, ``_UNREADABLE`` if the read failed.

    Like `_cluster_error_message`, this must hit the resource's *regional* endpoint — against the
    global default the ``get`` fails, and a teardown check that cannot read is a teardown check that
    always says "unverifiable".
    """
    try:
        from google.api_core.exceptions import NotFound
        from google.cloud import aiplatform_v1

        region = _region_from_resource_name(resource_name)
        client_options = {"api_endpoint": f"{region}-aiplatform.googleapis.com"} if region else None
        client = aiplatform_v1.PersistentResourceServiceClient(client_options=client_options)
        try:
            resource = client.get_persistent_resource(name=resource_name)
        except NotFound:
            return _ABSENT
        return str(aiplatform_v1.PersistentResource.State(resource.state).name)
    except Exception as exc:  # noqa: BLE001 - a failed read is an answer, not a fatal error
        _log.debug("could not read state of %s: %r", resource_name, exc)
        return _UNREADABLE


def _await_absent(
    read_state: Any,
    *,
    timeout_s: float = _TEARDOWN_TIMEOUT_S,
    poll_s: float = _TEARDOWN_POLL_S,
    sleep: Any = None,
    clock: Any = None,
) -> str:
    """Poll ``read_state`` until the resource is gone; return the last state observed.

    Ends on ``_ABSENT`` (gone), on ``_UNREADABLE`` (no point re-asking a question we cannot ask —
    returning at once beats stalling for the whole timeout on a permission fault), or on the
    deadline. The seams are injected so the loop is unit-testable without sleeping.
    """
    import time

    sleep = sleep or time.sleep
    clock = clock or time.monotonic

    deadline = clock() + timeout_s
    state = read_state()
    while state not in (_ABSENT, _UNREADABLE) and clock() < deadline:
        sleep(poll_s)
        state = read_state()
    return str(state)


def _delete_cluster(
    cluster_resource_name: str, *, verify: bool = True
) -> bool:  # pragma: no cover - live Vertex I/O, exercised by the @gpu smoke
    """Tear down an ephemeral cluster and *confirm it is gone*; True if confirmed absent.

    **The SDK's success line is not evidence.** ``vertex_ray`` prints ``Successfully deleted the
    cluster`` and returns as soon as the delete is accepted, so a resource that is stuck — or that
    the delete never really reached — reads exactly like one that went away. This campaign watched a
    teardown log success two seconds after a provisioning error and leave the cluster standing, and
    the only way to tell the two apart was to go and ask. So we ask: poll the resource until it
    reads ``NOT_FOUND``.

    Still best-effort in the sense that matters — a failure is logged and never raised, because a
    teardown that throws would mask the run's real outcome. What changes is that an unverified
    teardown now says so, loudly, naming a resource that is still billing.
    """
    from google.cloud.aiplatform import vertex_ray

    try:
        vertex_ray.delete_ray_cluster(cluster_resource_name=cluster_resource_name)
    except Exception as exc:  # noqa: BLE001 - teardown is best-effort; verify below regardless
        _log.warning("Ray cluster delete call failed (non-fatal): %r", exc)
    if not verify:
        return False

    final = _await_absent(lambda: _resource_state(cluster_resource_name))
    if final == _ABSENT:
        _log.info("deleted ephemeral Ray cluster %s (verified gone)", cluster_resource_name)
        return True
    _log.warning(
        "Ray cluster %s did not go away after %.0fs (last state %s) — it may still be running and "
        "billing; check it by hand",
        cluster_resource_name,
        _TEARDOWN_TIMEOUT_S,
        final,
    )
    return False


def _clear_stale_resource(resource_path: str) -> None:
    """Clear a same-named leftover before a create, so old wreckage cannot block a new attempt.

    The cluster name derives from ``run_id``, so every attempt at the same config targets the *same*
    resource path. A create that failed mid-provision can leave the resource behind in ``ERROR``,
    and the next attempt then fails with ``AlreadyExists`` while the delete that would fix it
    returns ``FAILED_PRECONDITION`` — the config becomes unrunnable until the state changes on its
    own. That turns an intermittent leak into a permanent one, and it is the reason this runs before
    every create rather than only on retry.

    Only ``ERROR`` is deleted: nothing is using it, and it is the state that blocks. A ``RUNNING``
    or ``PROVISIONING`` resource of the same name is **left alone** and allowed to collide — it
    may be a concurrent run of this same config, and taking its cluster out from under it would be
    a far worse failure than the collision. ``STOPPING`` is simply waited out.
    """
    state = _resource_state(resource_path)
    if state in (_ABSENT, _UNREADABLE):
        return
    if state == "STOPPING":
        _log.info("same-named Ray cluster %s is STOPPING; waiting for it to go", resource_path)
        _await_absent(lambda: _resource_state(resource_path))
        return
    if state == "ERROR":
        _log.warning(
            "clearing a leftover Ray cluster %s left in ERROR by an earlier attempt", resource_path
        )
        _delete_cluster(resource_path)
        return
    _log.info("same-named Ray cluster %s is %s; leaving it alone", resource_path, state)


def provision_shared_cluster(
    cfg: RunConfig,
    *,
    models: list[str],
    run_id: str,
    use_gpu: bool,
    gpu_type: str | None = None,
    settings: Settings | None = None,
    infra: RayInfra | None = None,
) -> tuple[str, str]:  # pragma: no cover - live Vertex I/O, exercised by the @gpu smoke
    """Create one shared ephemeral Ray cluster for a run's Ray families; return ``(name, region)``.

    The multi-family analog of `submit_ray`'s create step. When a run has more than one ephemeral
    Ray family the DAG orchestrator provisions **one** cluster here rather than letting each family
    create its own (which would collide on the run-derived ``sf-ray-<run_id>`` name and waste a
    second cluster). The cluster is sized for the **union** of those families' ``models`` (its CPU
    pool covers every Ray CPU-family model; it gets a GPU pool when ``use_gpu`` — any Ray family
    needs one) at the run's scale; autoscaling (the default) then absorbs the combined demand.

    Returns the cluster's display name and the region it actually landed in (a capacity hop may move
    it off the data-plane region). The caller threads both into every Ray family's `submit_ray`
    (``cluster_name`` + ``cluster_region`` → the reuse path, so each family submits its own
    failure-isolated Ray job to the shared cluster) and tears it down once via
    `teardown_shared_cluster` after all families join.
    """
    from .profiling.source import profile_for_run
    from .settings import Settings

    settings = settings or Settings.resolve()
    infra = infra or RayInfra.resolve()
    plan = ray_io.plan_cluster(
        cfg,
        models,
        run_id=run_id,
        use_gpu=use_gpu,
        gpu_type=gpu_type,
        # Same evidence the per-family `submit_ray` calls will size off (memoized), so the shared
        # cluster is shaped for the union of families by the same rule each family would apply.
        profile=profile_for_run(cfg, settings=settings),
    )
    regions = _resolve_regions(cfg, settings)
    _log.info(
        "provisioning shared Ray cluster %s: %d union models use_gpu=%s cpu_nodes=%d gpu_nodes=%d",
        plan.cluster_name,
        len(models),
        use_gpu,
        plan.cpu_node_count,
        plan.gpu_node_count,
    )
    _resource, region = _create_cluster_across_regions(
        plan,
        infra,
        plan.cluster_name,
        settings,
        regions,
        policy=cfg.compute.capacity.policy_for("ray"),
        preflight=cfg.compute.capacity.preflight,
    )
    return plan.cluster_name, region


def teardown_shared_cluster(
    name: str, region: str, settings: Settings
) -> None:  # pragma: no cover - live Vertex I/O, exercised by the @gpu smoke
    """Tear down the run's shared ephemeral Ray cluster (best-effort, like `submit_ray`'s teardown).

    Deletes by the deterministic resource path in the region the cluster landed in
    (`provision_shared_cluster` returns it). `_delete_cluster` swallows any error, so a cluster that
    never fully materialized is a harmless no-op.
    """
    _delete_cluster(cluster_resource_path(settings, name, region))
