"""The Vertex Ray cluster — plan it into a spec, provision it, find it, tear it down.

Everything that talks to Vertex's ``vertex_ray`` control plane about a *cluster* (as opposed to a
job running on one, which is `ray_jobs`). The Ray sibling of `dataproc_cluster`, and it carries the
same two responsibilities that module does: turn a ``RayClusterPlan`` into the SDK's resource shape,
and own the resource's whole lifetime.

The interesting part is the **region fallback**. GPU capacity and accelerator quota are both granted
per-region, so a create that fails for either reason says nothing about the next region's odds —
`_create_cluster_across_regions` walks the configured regions until one provisions. Two things make
that harder than it sounds and are the reason the error classifiers are their own pure functions:
the SDK's exception is a generic "Cluster … returned an error" while the actionable text lives on
the ``PersistentResource.error`` field (`_cluster_error_message`), and that read has to hit the
*regional* endpoint or the reason is silently lost. Misclassifying here means either burning every
region on a config typo or giving up on the first stockout.

Those two costs are not symmetric, and the walk is biased accordingly: it continues by default and
stops only on a cause that is the same everywhere. Vertex frequently fails a provision without
saying why at all, so a classifier that had to *recognise* a reason before continuing spent most of
its life not continuing.

Only the cluster hops. The data plane — config staging, registry writes — stays pinned to
``settings.region``, which is why `cluster_resource_path` takes an explicit region rather than
assuming one.

Public surface: ``cluster_resource_path``, ``provision_shared_cluster``,
``teardown_shared_cluster``. The lifecycle verbs `_create_cluster_across_regions` / `_get_cluster` /
`_delete_cluster` are driven by `ray_submit.submit_ray`, which owns the create→run→teardown
ordering for a single-family run.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from .engines import ray_io
from .errors import EngineError, get_logger
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


# Substrings that mark a *regional capacity* failure (retry a different region) vs. a config/quota
# error (retrying elsewhere won't help). Matched case-insensitively against the cluster's error
# message. Kept as data so the classifier stays a pure, unit-testable function.
_CAPACITY_ERROR_MARKERS = (
    "resources are insufficient in region",
    "try a different region",
    "does not have enough resources",
    "insufficient resources",
    "resource exhausted",
)


def _is_capacity_error(message: str) -> bool:
    """True if a cluster-create error message signals a *regional capacity* shortage (pure).

    Capacity errors are worth retrying in another region; a bad machine type or permission fault is
    not. Quota is handled separately by `_is_quota_error` (also region-hoppable, different reason).
    """
    low = message.lower()
    return any(marker in low for marker in _CAPACITY_ERROR_MARKERS)


# What marks a *regional quota* ceiling. GPU/accelerator quota on Vertex is granted
# per-region, so a region that is over its quota says nothing about the next region's ceiling — the
# fallback advances on these just as it does on capacity stockouts. A quota error is distinct from a
# capacity stockout (the region has room, this project is simply not allowed more), so it gets its
# own classifier rather than widening the capacity markers.
#
# Matched *compositionally* — "quota" near a word of exhaustion — rather than as fixed phrases.
# Every phrase the fixed list once held ("quota exceeded", "exceeds quota", "quota limit", …) says
# quota and says exceed-or-limit, so nothing is lost; what is gained is the wordings nobody
# enumerated. Found live 2026-09-01: `us-east1` answered "The following quotas are exceeded:
# CustomModelTrainingT4GPUsPerProjectPerRegion" — plural, and in an order no marker matched — so a
# textbook hoppable quota ceiling was misread as a config fault and the third region never tried.
_QUOTA_WORDS = ("quota", "quotas")
_EXHAUSTION_WORDS = ("exceed", "exceeds", "exceeded", "limit", "limits")


def _is_quota_error(message: str) -> bool:
    """True if a cluster-create error message signals a *regional quota* ceiling (pure).

    Vertex accelerator quota is per-region, so a quota-exhausted region is worth retrying elsewhere:
    another region carries its own independent ceiling. (A capacity stockout is a different reason
    with the same remedy — hop — and is classified by `_is_capacity_error`.)

    Note that the relevant ceiling is often *not* the Compute Engine one. `NVIDIA_T4_GPUS` can read
    4-of-4 free in a region while `CustomModelTrainingT4GPUsPerProjectPerRegion` — the Vertex-side
    quota a Ray cluster actually spends — is zero. Checking Compute Engine quota before a Ray GPU
    run tells you nothing.
    """
    low = message.lower()
    return any(q in low for q in _QUOTA_WORDS) and any(e in low for e in _EXHAUSTION_WORDS)


# Substrings marking a cause that is the *same in every region*, and so the only reason to stop
# walking. Everything here names something about the request rather than the place: who is asking,
# what they asked for, whether the API is even on. Trying `us-east1` will not change any of them.
#
# This list is the *whole* stop condition — see `_is_region_invariant_error` for why the fallback is
# an allowlist of reasons to give up rather than an allowlist of reasons to continue.
_REGION_INVARIANT_MARKERS = (
    "permission denied",
    "permission_denied",
    "does not have permission",
    "not authorized",
    "unauthorized",
    "iam",
    "service account",
    "invalid argument",
    "invalid_argument",
    # Broad on purpose: any complaint *about the machine type* is a complaint about what was asked
    # for. Guarded below by the capacity/quota check, which wins — "machine type X is unavailable in
    # this zone" is about the place, and reads as capacity.
    "machine type",
    "unsupported accelerator",
    "has not been used in project",  # API disabled
    "api is not enabled",
    "billing",
)


def _is_region_invariant_error(message: str) -> bool:
    """True if the failure would recur identically in every region, so walking is pointless (pure).

    **This is the stop condition, and it is deliberately an allowlist of reasons to give up.** The
    fallback used to work the other way — hop only on reasons we recognised, re-raise on everything
    else — and that inverted default cost the feature three times in one afternoon (2026-09-01),
    each time to a different contentless string: "An internal error occurred on your cluster",
    "Unexpected response.", and a quota message phrased in an order no marker matched. Each read as
    a diagnosed config fault and re-raised in the first region, so a config naming three regions
    tried one.

    The asymmetry justifies the inversion. Hopping when we should not have costs a few minutes of
    provisioning per extra region, and the caller still ends up with an `EngineError` naming every
    region tried and carrying the last error — the diagnosis is not lost, only delayed. Not hopping
    when we should have costs the entire feature, silently, and only shows up as a live failure in a
    region that ran out. So: give up only when the message names a cause that travels with the
    *request*, not with the *place*.

    Capacity and quota keep their own classifiers (`_is_capacity_error`, `_is_quota_error`) — no
    longer to decide whether to continue, but to say *why* in the log, which is worth keeping.
    """
    low = message.lower()
    return any(marker in low for marker in _REGION_INVARIANT_MARKERS)


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


def _create_cluster_across_regions(
    plan: ray_io.RayClusterPlan,
    infra: RayInfra,
    name: str,
    settings: Settings,
    regions: list[str],
) -> tuple[str, str]:  # pragma: no cover - orchestrates live Vertex I/O; @gpu smoke exercises it
    """Create the cluster, walking ``regions`` in order until one can provision it.

    Returns ``(cluster_resource_name, region)`` for the region that succeeded. On a *regional
    **The default is to keep walking.** The failed attempt's (deterministic) resource is torn down
    and the next region tried unless the message names a cause that travels with the request rather
    than the place — permission, a bad machine type, an API that isn't on — for which
    `_is_region_invariant_error` re-raises at once. Capacity and quota are still recognised, but now
    only to say *why* in the log. Exhausting every region raises `EngineError` naming the regions
    tried and carrying the last error. See `_is_region_invariant_error` for why the stop condition
    is an allowlist rather than the continue condition.

    The failure signal is read from the failed resource's ``error.message`` (via
    `_cluster_error_message`) *and* the raised exception string — the SDK's exception is a
    generic "returned an error" while the "Resources are insufficient in region" / quota text lives
    only on the resource, so classifying on the exception alone would never detect a stockout.

    Only the cluster hops — the data plane (config staging, registry writes) stays in
    ``settings.region``. The SDK is re-pinned to each attempted region via `_init_vertex`
    just before the create, so ``vertex_ray`` provisions there.
    """
    last_exc: Exception | None = None
    for region in regions:
        _init_vertex(settings, region)
        try:
            _log.info("attempting Ray cluster %s in region %s", name, region)
            resource_name = _create_cluster(plan, infra, name)
            _log.info("Ray cluster %s created in region %s", name, region)
            return resource_name, region
        except Exception as exc:  # noqa: BLE001 - classify, then either advance or re-raise
            # Read the resource's own error text *before* teardown — that's where the capacity
            # reason lives; the exception string is only a generic "returned an error".
            resource_path = cluster_resource_path(settings, name, region)
            detail = _cluster_error_message(resource_path)
            message = f"{exc} | {detail}".strip(" |")
            _delete_cluster(
                resource_path
            )  # a create that errors mid-provision still leaves a resource
            # Hop when the reason reads as a per-region condition — capacity stockout or quota
            # ceiling — OR when we couldn't read the reason but the SDK raised its generic
            # post-provision "returned an error" (which only fires after polling to ERROR state — in
            # practice a stockout). A specific exception with none of those signals is a real
            # config/permission fault: another region won't help, so re-raise.
            # Capacity and quota win the tie. A message can name a machine type *and* say the
            # region ran out of it ("machine type X unavailable in zone Y"); that is about the
            # place, and hopping is exactly right.
            regional = _is_capacity_error(message) or _is_quota_error(message)
            if _is_region_invariant_error(message) and not regional:
                raise
            if _is_quota_error(message) and not _is_capacity_error(message):
                reason = "quota ceiling"
            elif _is_capacity_error(message):
                reason = "insufficient capacity"
            else:
                # The common case, and the one the old allowlist kept getting wrong: Vertex
                # failed and would not say why. Name it as unexplained rather than guessing.
                reason = "an unexplained provisioning failure"
            _log.warning(
                "region %s hit %s (%s); trying next region",
                region,
                reason,
                detail or exc,
            )
            last_exc = exc
    raise EngineError(
        f"Ray cluster {name} could not be created in any of {regions} "
        f"(no capacity or quota available): last error {last_exc!r}"
    )


def _get_cluster(
    cluster_resource_name: str,
) -> Any:  # pragma: no cover - live Vertex I/O, exercised by the @gpu smoke
    """Fetch a Vertex Ray cluster by resource name (for its ``dashboard_address`` + telemetry)."""
    from google.cloud.aiplatform import vertex_ray

    return vertex_ray.get_ray_cluster(cluster_resource_name=cluster_resource_name)


def _delete_cluster(
    cluster_resource_name: str,
) -> None:  # pragma: no cover - live Vertex I/O, exercised by the @gpu smoke
    """Tear down an ephemeral cluster (best-effort: a teardown failure is logged, never fatal)."""
    from google.cloud.aiplatform import vertex_ray

    try:
        vertex_ray.delete_ray_cluster(cluster_resource_name=cluster_resource_name)
        _log.info("deleted ephemeral Ray cluster %s", cluster_resource_name)
    except Exception as exc:  # noqa: BLE001 - teardown is best-effort; surface it, don't re-raise
        _log.warning("Ray cluster teardown failed (non-fatal): %r", exc)


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
        plan, infra, plan.cluster_name, settings, regions
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
