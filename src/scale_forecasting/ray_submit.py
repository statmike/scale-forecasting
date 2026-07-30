"""Submit a forecast run to Ray on Vertex AI (BUILD B4) — the fixed-size-cluster launcher.

The Ray analog of :mod:`~scale_forecasting.submit`: the ``[ray]``-extra, ADC-authenticated helper
that turns a validated :class:`~scale_forecasting.config.RunConfig` into a run on a **fixed-size**
Vertex Ray cluster (DESIGN §11.1 / D17 — deterministic sizing, *not* autoscaling). It owns the
cluster lifecycle the way ``submit`` owns the Dataproc batch; the on-cluster compute is
:func:`~scale_forecasting.engines.ray_engine.run`, reached through the
:mod:`~scale_forecasting.ray_entry` Jobs entrypoint.

What :func:`submit_ray` does:

1. **Size the cluster to the run's fan-out** — :func:`.ray_io.plan_cluster` turns the config into a
   fixed ``RayClusterPlan`` (a GPU worker pool for NeuralProphet + a CPU worker pool for everything
   else, each a fixed ``node_count``, no ``autoscaling_spec``). "Resize for a bigger/smaller scale"
   is just a different ``series_limit`` yielding a different plan — the sizing decision is logged
   and stamped to the run.
2. **Stage the run config** — write the validated config to ``gs://<code>/runs/<run_id>.json`` and
   pass it as ``--config-uri`` (the lossless reproducibility record, G3 — same contract as Spark).
3. **Provision (ephemeral default) or target (reuse opt-in) the cluster** — ephemeral:
   ``create_ray_cluster`` at the planned fixed size, run, then ``delete_ray_cluster`` in a
   ``finally`` (teardown guaranteed even on failure); reuse: ``compute.ray_cluster_name`` /
   ``cluster_name=`` targets a standing cluster by name and skips both create and delete.
4. **Submit the on-cluster driver** — a Ray Job via
   :class:`~ray.job_submission.JobSubmissionClient` (``vertex_ray://<dashboard>``) whose entrypoint
   is ``python -m scale_forecasting.ray_entry`` with the same ``--config-uri`` / ``--models`` /
   ``--manage-header`` / ``--sf-*`` contract the Spark entry uses. Current ``src/`` ships as the
   job's ``runtime_env`` working dir (runtime code delivery, never baked into the image — the G1
   seam), with ``requirements.txt`` for the on-cluster deps.
5. **Poll to terminal + stamp telemetry** — with ``wait``, block until the job is terminal, stamp a
   Ray analog of Spark's ``job_telemetry`` (cluster name, node counts, machine/accelerator types,
   calibrated-vs-sizing GPU fraction, wall-clock, job id) into ``run_registry.job_telemetry`` via
   ``bq.update_header`` — **no schema change** (the STRING column already exists) — and raise on a
   non-SUCCEEDED terminal state so a failed run never exits 0.

Public surface: ``RayInfra``, ``submit_ray``, ``build_entrypoint``, ``build_runtime_env``,
``extract_ray_telemetry``, ``main``.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._infra_args import infra_args_from
from .engines import ray_io
from .errors import ConfigError, EngineError, get_logger

if TYPE_CHECKING:
    from .config import RunConfig
    from .settings import Settings

_log = get_logger(__name__)

# The package root shipped as the Ray job's runtime_env working dir (contains scale_forecasting/, so
# `python -m scale_forecasting.ray_entry` resolves on the cluster). The locked cluster deps live at
# docker/requirements.txt (the same file the container image pins) — reused for the on-cluster pip.
_SRC_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _SRC_DIR.parent
_REQUIREMENTS = _REPO_ROOT / "docker" / "requirements.txt"

# Ray-cluster infra env vars (beyond the SF_* identity Settings resolves). Kept together so the
# docstring, resolve(), and any tooling agree. code_bucket + compute_sa are shared with the Spark
# batch; network (optional) is a VPC for a private endpoint; container_image is optional.
_ENV_NETWORK = "SF_RAY_NETWORK"
_ENV_COMPUTE_SA = "SF_COMPUTE_SA"
_ENV_CODE_BUCKET = "SF_CODE_BUCKET"
_ENV_CONTAINER_IMAGE = "SF_CONTAINER_IMAGE"
_ENV_RAY_VERSION = "SF_RAY_VERSION"

# Vertex Ray's supported Ray version + our runtime Python. Overridable via SF_RAY_VERSION so a newer
# cluster image can be selected without a code change (the client submits jobs over HTTP, which is
# tolerant of a minor client/cluster Ray skew).
_DEFAULT_RAY_VERSION = "2.47"
_DEFAULT_PYTHON_VERSION = "3.11"

# Poll cadence + terminal Ray job states (the Jobs API reports these on get_job_status).
_POLL_SECONDS = 15
_TERMINAL_STATES = frozenset({"SUCCEEDED", "FAILED", "STOPPED"})

# Dashboard warm-up race: a cluster reaches RUNNING *before* its Ray dashboard is reachable through
# Vertex's public-endpoint proxy, so the first JobSubmissionClient handshake (GET /api/version) can
# get a proxy gateway timeout / connection refusal. We retry the *connection only* with backoff up
# to this budget before giving up — well within the ~few-minutes the dashboard takes to serve.
_DASHBOARD_CONNECT_ATTEMPTS = 20
_DASHBOARD_CONNECT_BACKOFF_SECONDS = 15


@dataclass(frozen=True)
class RayInfra:
    """Vertex-Ray infra identity — what launching a cluster needs beyond :class:`Settings`.

    Resolved from ``SF_*`` env (parity with ``Settings`` / ``BatchInfra``) or ``terraform output``.
    ``network`` is an **optional** VPC: set it (with a private-services connection in place) for a
    private endpoint; leave it unset and the cluster gets a public endpoint — no VPC peering / PSA
    required, which is the simplest way to run when the deployment has no private-services access.
    ``compute_sa`` is the runtime SA the cluster runs as; ``code_bucket`` where the run config JSON
    is staged. ``container_image`` is an optional custom node image (the shared runtime that bundles
    ``[ray]`` + ``[models]``); when unset, Vertex's prebuilt Ray image is used and ``runtime_env``
    installs ``requirements.txt`` on top.
    """

    compute_sa: str
    code_bucket: str
    network: str | None = None
    container_image: str | None = None
    ray_version: str = _DEFAULT_RAY_VERSION
    python_version: str = _DEFAULT_PYTHON_VERSION

    @classmethod
    def resolve(cls) -> RayInfra:
        """Build from the ``SF_*`` Ray-infra environment; raise naming the first missing var.

        ``SF_COMPUTE_SA`` and ``SF_CODE_BUCKET`` are required; ``SF_RAY_NETWORK`` is optional
        (unset → public endpoint, no VPC peering needed).
        """
        required = {
            "compute_sa": _ENV_COMPUTE_SA,
            "code_bucket": _ENV_CODE_BUCKET,
        }
        values: dict[str, str] = {}
        for field_name, env_name in required.items():
            raw = os.environ.get(env_name)
            if not raw:
                raise ConfigError(
                    f"missing required environment variable {env_name} "
                    f"(set it, or use RayInfra.from_terraform_outputs for local dev)"
                )
            values[field_name] = raw
        return cls(
            compute_sa=values["compute_sa"],
            code_bucket=values["code_bucket"],
            network=os.environ.get(_ENV_NETWORK) or None,
            container_image=os.environ.get(_ENV_CONTAINER_IMAGE) or None,
            ray_version=os.environ.get(_ENV_RAY_VERSION) or _DEFAULT_RAY_VERSION,
        )

    @classmethod
    def from_terraform_outputs(
        cls, outputs: dict[str, str], image_tag: str | None = None
    ) -> RayInfra:
        """Build from a ``terraform output -json`` value map (local dev/tests).

        Reads the keys the ``terraform/main`` stage emits — ``compute_sa``, ``code_bucket``, an
        optional ``network_id`` (absent → public endpoint), and (when ``image_tag`` is given)
        ``runtime_image_repo`` for a custom node image. Omitting ``image_tag`` leaves
        ``container_image`` unset (Vertex prebuilt image + ``requirements.txt``).
        """
        try:
            image = (
                f"{outputs['runtime_image_repo']}:{image_tag}" if image_tag is not None else None
            )
            return cls(
                compute_sa=outputs["compute_sa"],
                code_bucket=outputs["code_bucket"],
                network=outputs.get("network_id") or None,
                container_image=image,
            )
        except KeyError as exc:
            raise ConfigError(f"terraform outputs missing key: {exc.args[0]}") from exc


# --- pure: job spec assembly (no network) --------------------------------------


def build_entrypoint(
    config_uri: str,
    settings: Settings,
    *,
    models: list[str] | None = None,
    manage_header: bool = True,
) -> str:
    """The Ray Job entrypoint shell command — ``python -m scale_forecasting.ray_entry ...`` (pure).

    The Ray analog of :func:`~scale_forecasting.submit.build_batch`'s arg list, as a single shell
    string (what the Jobs API runs on the cluster head). Carries the same contract: ``--config-uri``
    (the staged config, whose digest is the shared ``run_id``), the ``--sf-*`` infra identity, and
    — only when non-default — ``--models m1,m2`` (executed subset, Arc B) and ``--manage-header
    false`` (contributor mode; :func:`main.run` owns the shared header). Defaults omit the Arc B
    flags so a standalone submit builds the plain command.
    """
    parts = ["python", "-m", "scale_forecasting.ray_entry", "--config-uri", config_uri]
    parts += infra_args_from(settings)
    if models is not None:
        parts += ["--models", ",".join(models)]
    if not manage_header:
        parts += ["--manage-header", "false"]
    return " ".join(parts)


# Packages the cluster already provides — never reinstall these via runtime_env or a pip-installed
# version would fight the one baked into the Vertex Ray image (the cluster's Ray is pinned at create
# via ``ray_version``, and requirements.txt may pin a newer Ray than Vertex supports, so swapping it
# out from under the running head/workers breaks the job). Matched on the PEP-508 project name.
_CLUSTER_PROVIDED = frozenset({"ray"})


def _requirements_packages() -> list[str]:
    """Parse ``docker/requirements.txt`` into a package-spec list, dropping cluster-provided deps.

    The uv-exported file is ``name==version [; marker]`` lines interleaved with ``# via`` comment
    blocks; we keep only the requirement lines and skip anything whose project name is in
    :data:`_CLUSTER_PROVIDED` (see its note — Ray must come from the image, not pip).
    """
    packages: list[str] = []
    for raw in _REQUIREMENTS.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Project name = everything before the first version/extra/marker delimiter.
        name = re.split(r"[<>=!~;\[ ]", line, maxsplit=1)[0].lower()
        if name in _CLUSTER_PROVIDED:
            continue
        packages.append(line)
    return packages


def build_runtime_env() -> dict[str, Any]:
    """The Ray ``runtime_env`` shipping current ``src/`` + on-cluster deps (pure; local paths).

    Delivers code at RUNTIME the way the Spark path uploads a ``src/`` zip: ``working_dir`` is the
    package root (so ``python -m scale_forecasting.ray_entry`` imports the code that was just
    submitted, not anything baked into the image — the G1 "same code" seam), and ``pip`` is the
    requirements package **list minus Ray** (:func:`_requirements_packages`): a Vertex prebuilt
    image gains our deps, while Ray itself stays the image's version rather than being swapped by a
    conflicting pip pin. With a custom node image that already bundles them the pip step is a fast
    no-op, so it is always kept for safety.
    """
    return {
        "working_dir": str(_SRC_DIR),
        "pip": _requirements_packages(),
    }


def extract_ray_telemetry(
    plan: ray_io.RayClusterPlan,
    *,
    cluster: object,
    job_id: str,
    job_status: str,
    total_wall_s: float | None,
    reuse: bool,
) -> dict[str, Any]:
    """Flatten the plan + cluster into the JSON-able telemetry dict stamped on the header (pure).

    The Ray analog of :func:`~scale_forecasting.submit.extract_job_telemetry`, answering the same
    operability questions for a *fixed-size* cluster — *how big was the pool, what did it cost in
    wall-clock, and what sizing produced it* — so a Ray run is as auditable on ``v_run_summary`` as
    a Spark one. Reads only fields already on the ``plan`` and the ``cluster`` object; every cluster
    field is optional (a missing attr degrades to None, never a raise) so this is safe on any object
    ``get_ray_cluster`` returns.
    """
    return {
        "runtime": "ray",
        "cluster_name": plan.cluster_name,
        "reuse": reuse,
        "job_id": job_id,
        "job_status": job_status,
        "total_wall_s": total_wall_s,
        "cpu_node_count": plan.cpu_node_count,
        "gpu_node_count": plan.gpu_node_count,
        "total_worker_nodes": plan.total_worker_nodes,
        "head_machine_type": plan.head_machine_type,
        "cpu_machine_type": plan.cpu_machine_type,
        "gpu_machine_type": plan.gpu_machine_type,
        "accelerator_type": plan.accelerator_type,
        "accelerator_count": plan.accelerator_count,
        "sizing_gpu_fraction": plan.sizing_gpu_fraction,
        "n_gpu_cells": plan.n_gpu_cells,
        "n_cpu_cells": plan.n_cpu_cells,
        "ray_version": getattr(cluster, "ray_version", None) or None,
        "python_version": getattr(cluster, "python_version", None) or None,
        "dashboard_address": getattr(cluster, "dashboard_address", None) or None,
    }


# --- I/O: config staging -------------------------------------------------------


def _stage_config(cfg: RunConfig, run_id: str, infra: RayInfra) -> str:
    """Write the validated config to ``gs://<code>/runs/<run_id>.json`` and return the URI (G3).

    Byte-for-byte the Spark staging contract (:func:`~scale_forecasting.submit._stage_config`) so a
    mixed run stages one config the same way regardless of runtime — the JSON *is* the shared
    reproducibility record, and its digest is the shared ``run_id``.
    """
    import json

    from google.cloud import storage

    client = storage.Client()
    payload = json.dumps(cfg.model_dump(mode="json"), sort_keys=True, indent=2)
    name = f"runs/{run_id}.json"
    storage.Client.bucket(client, infra.code_bucket).blob(name).upload_from_string(
        payload, content_type="application/json"
    )
    return f"gs://{infra.code_bucket}/{name}"


# --- I/O: Vertex Ray cluster lifecycle -----------------------------------------


def _worker_resources(plan: ray_io.RayClusterPlan, infra: RayInfra) -> list[Any]:
    """Build the fixed-size worker ``Resources`` list — one entry per non-empty pool (D17).

    A GPU pool (``accelerator_type``/``accelerator_count``, ``gpu_node_count`` nodes) for
    NeuralProphet and a CPU pool (``cpu_node_count`` nodes) for everything else, each a fixed
    ``node_count`` and **no** ``autoscaling_spec`` — the whole determinism guarantee. A pool with
    zero planned nodes is omitted (Vertex rejects a zero-node worker type). The optional custom node
    image is applied to every pool so the on-cluster code sees the bundled deps.
    """
    from google.cloud.aiplatform import vertex_ray

    image = infra.container_image
    workers: list[Any] = []
    if plan.cpu_node_count > 0:
        workers.append(
            vertex_ray.Resources(
                machine_type=plan.cpu_machine_type,
                node_count=plan.cpu_node_count,
                custom_image=image,
            )
        )
    if plan.gpu_node_count > 0:
        workers.append(
            vertex_ray.Resources(
                machine_type=plan.gpu_machine_type,
                node_count=plan.gpu_node_count,
                accelerator_type=plan.accelerator_type,
                accelerator_count=plan.accelerator_count,
                custom_image=image,
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
    any multi-project setup). Binding it from :class:`Settings` here keeps the same code targeting
    the configured project everywhere (G1) — never whatever project the shell happens to point at.

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

    Capacity errors are worth retrying in another region; anything else (bad machine type, missing
    quota, permission) is not — so the fallback loop only advances on these, else fails fast.
    """
    low = message.lower()
    return any(marker in low for marker in _CAPACITY_ERROR_MARKERS)


def _is_generic_cluster_error(message: str) -> bool:
    """True for the SDK's opaque post-provision "Cluster ... returned an error." (pure).

    The Vertex SDK raises exactly this after polling a create to ERROR state, with the real reason
    only on the resource (not the exception). When the resource read also fails we can't see the
    reason — but this string only appears *after* a cluster provisioned and then failed, which in
    practice is a capacity stockout, so the fallback treats it as retryable rather than fatal.
    """
    low = message.lower()
    return "returned an error" in low and "cluster" in low


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
    """Create the fixed-size Vertex Ray cluster and return its ``cluster_resource_name``.

    Head node is a single small CPU box (no accelerator); workers are the planned GPU/CPU pools
    (:func:`_worker_resources`). No ``autoscaling_spec`` anywhere — a fixed pool is the honest model
    for a fixed-scale batch (D17). ``infra.network`` is passed through: a VPC (with a
    private-services connection) gives a private endpoint, and ``None`` gives a public endpoint —
    Vertex's own default — so a deployment without VPC peering can still run. Labels tag the run.
    """
    from google.cloud.aiplatform import vertex_ray

    head = vertex_ray.Resources(
        machine_type=plan.head_machine_type,
        node_count=1,
        custom_image=infra.container_image,
    )
    _log.info(
        "creating fixed-size Ray cluster %s: cpu_nodes=%d gpu_nodes=%d accel=%s x%d endpoint=%s",
        name,
        plan.cpu_node_count,
        plan.gpu_node_count,
        plan.accelerator_type,
        plan.accelerator_count,
        "private" if infra.network else "public",
    )
    return vertex_ray.create_ray_cluster(
        head_node_type=head,
        worker_node_types=_worker_resources(plan, infra),
        cluster_name=name,
        network=infra.network,
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
    """Create the fixed-size cluster, walking ``regions`` in order until one has T4 capacity.

    Returns ``(cluster_resource_name, region)`` for the region that succeeded. On a *regional
    capacity* failure (:func:`_is_capacity_error`) the stocked-out attempt's (deterministic)
    resource is torn down and the next region tried; any *other* error (bad machine type, missing
    quota, permission) is re-raised at once because another region won't fix it. Exhausting every
    region raises :class:`EngineError` naming the regions tried.

    The capacity signal is read from the failed resource's ``error.message`` (via
    :func:`_cluster_error_message`) *and* the raised exception string — the SDK's exception is a
    generic "returned an error" while the "Resources are insufficient in region" text lives only on
    the resource, so classifying on the exception alone would never detect a stockout.

    Only the cluster hops — the data plane (config staging, registry writes) stays in
    ``settings.region``. The SDK is re-pinned to each attempted region via :func:`_init_vertex`
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
            resource_path = _resource_name(settings, name, region)
            detail = _cluster_error_message(resource_path)
            message = f"{exc} | {detail}".strip(" |")
            _delete_cluster(
                resource_path
            )  # a create that errors mid-provision still leaves a resource
            # Hop when the reason reads as capacity, OR when we couldn't read the reason but the SDK
            # raised its generic post-provision "returned an error" (which only fires after polling
            # to ERROR state — in practice a stockout). A specific exception with no capacity signal
            # is a real config/quota/permission fault: another region won't help, so re-raise.
            capacity = _is_capacity_error(message)
            generic_provision_error = not detail and _is_generic_cluster_error(str(exc))
            if not (capacity or generic_provision_error):
                raise
            _log.warning(
                "region %s lacks T4 capacity (%s); trying next region",
                region,
                detail or exc,
            )
            last_exc = exc
    raise EngineError(
        f"Ray cluster {name} could not be created in any of {regions} "
        f"(no T4 capacity): last error {last_exc!r}"
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


def _is_dashboard_warmup_error(exc: Exception) -> bool:
    """True if ``exc`` looks like the dashboard-not-yet-serving race (retryable), not a real fault.

    The JobSubmissionClient version handshake fails during warm-up with a proxy gateway timeout
    (HTTP 5xx — 502/503/504, and Cloudflare's 524) or a bare connection error, all transient. A
    4xx / auth / version-mismatch is a genuine fault and must *not* be retried, so we match on the
    known-transient shapes only.
    """
    low = str(exc).lower()
    transient_markers = (
        " 502",
        " 503",
        " 504",
        " 524",
        "gateway",
        "timeout",
        "timed out",
        "connection",
        "temporarily unavailable",
        "max retries",
    )
    return any(marker in low for marker in transient_markers)


def _connect_job_client(
    dashboard_address: str,
) -> Any:  # pragma: no cover - live Ray Jobs I/O, exercised by the @gpu smoke
    """Open a ``JobSubmissionClient`` to the dashboard, retrying past the warm-up race.

    ``JobSubmissionClient.__init__`` does a GET ``/api/version`` handshake; right after the cluster
    hits RUNNING that endpoint may not be reachable through Vertex's public-endpoint proxy yet, so
    the first attempts can raise a proxy gateway timeout (524/504/…). We back off and retry the
    *connection only* (never a partial submit) until it succeeds or the budget is spent, then let
    the last error propagate.
    """
    import time

    from ray.job_submission import JobSubmissionClient

    last_exc: Exception | None = None
    for attempt in range(1, _DASHBOARD_CONNECT_ATTEMPTS + 1):
        try:
            return JobSubmissionClient(f"vertex_ray://{dashboard_address}")
        except Exception as exc:  # noqa: BLE001 - classify, retry transients, re-raise real faults
            if not _is_dashboard_warmup_error(exc):
                raise
            last_exc = exc
            _log.info(
                "Ray dashboard not ready yet (attempt %d/%d): %r; retrying in %ds",
                attempt,
                _DASHBOARD_CONNECT_ATTEMPTS,
                exc,
                _DASHBOARD_CONNECT_BACKOFF_SECONDS,
            )
            time.sleep(_DASHBOARD_CONNECT_BACKOFF_SECONDS)
    assert last_exc is not None
    raise last_exc


def _submit_and_poll(
    cluster: object,
    entrypoint: str,
    runtime_env: dict[str, Any],
    *,
    wait: bool,
) -> tuple[str, str]:  # pragma: no cover - live Ray Jobs I/O, exercised by the @gpu smoke
    """Submit the on-cluster driver as a Ray Job and (when ``wait``) poll to a terminal state.

    Connects the Jobs client to the cluster's dashboard (``vertex_ray://<dashboard_address>``,
    retrying past the dashboard warm-up race), submits ``entrypoint`` with ``runtime_env`` (current
    ``src/`` + requirements), and returns ``(job_id, status)``. Without ``wait`` the status is the
    immediate post-submit state (the caller skips telemetry + the terminal-state check).
    """
    import time

    client = _connect_job_client(cluster.dashboard_address)  # type: ignore[attr-defined]
    job_id = client.submit_job(entrypoint=entrypoint, runtime_env=runtime_env)
    _log.info("submitted Ray job %s", job_id)
    if not wait:
        return job_id, str(client.get_job_status(job_id))

    status = str(client.get_job_status(job_id))
    while status not in _TERMINAL_STATES:
        time.sleep(_POLL_SECONDS)
        status = str(client.get_job_status(job_id))
    _log.info("Ray job %s finished: status=%s", job_id, status)
    return job_id, status


def _stamp_ray_telemetry(telemetry: dict[str, Any], run_id: str, settings: Settings) -> None:
    """Write the Ray telemetry dict to the run header as a JSON string (best-effort).

    The pure :func:`extract_ray_telemetry` output → ``update_header(job_telemetry=<json>)``. Wrapped
    so any failure (API error, header not yet written) is logged and swallowed: telemetry is a
    nice-to-have overlay on an already-complete run, never a reason to fail it (CONTRACTS §3.3).
    """
    import json

    from .registry import bq

    try:
        bq.update_header(
            run_id, settings=settings, job_telemetry=json.dumps(telemetry, sort_keys=True)
        )
        _log.info("Ray telemetry stamped for run %s: %s", run_id, telemetry)
    except Exception as exc:  # noqa: BLE001 - telemetry is best-effort, never fatal
        _log.warning("Ray telemetry capture failed (non-fatal): %r", exc)


def submit_ray(
    cfg: RunConfig,
    *,
    models: list[str] | None = None,
    manage_header: bool = True,
    settings: Settings | None = None,
    infra: RayInfra | None = None,
    cluster_name: str | None = None,
    n_series: int | None = None,
    wait: bool = True,
) -> str:
    """Size, provision, run, and (ephemeral) tear down a Ray-on-Vertex forecast run; return job id.

    The Ray analog of :func:`~scale_forecasting.submit.submit_batch`. Resolves infra from the
    environment when not passed (G1), sizes a **fixed** cluster to the run's fan-out
    (:func:`.ray_io.plan_cluster` — no autoscaling, D17), stages the full config to GCS (so its
    ``run_id`` matches :func:`main.run`'s), then runs the lifecycle:

    * **ephemeral (default):** create the planned cluster → submit the on-cluster driver as a Ray
      Job → (with ``wait``) poll to terminal + stamp telemetry → ``delete_ray_cluster`` in a
      ``finally`` so teardown happens even if the job raises.
    * **reuse (opt-in):** ``cluster_name`` (or ``compute.ray_cluster_name``) targets a standing
      cluster by name — skip create *and* skip delete; the plan still records the size it *should*
      be.

    ``n_series`` overrides ``series_limit`` at submit time (the scale knob — a different scale is a
    different fixed plan *and* a distinct ``run_id``/header, so each scale is its own queryable run;
    this is how "resize for a larger/smaller scale" is driven). ``models`` / ``manage_header`` carry
    the Arc B contract: the full ``cfg`` is always staged (shared ``run_id``) while ``models``
    restricts the on-cluster executed subset and ``manage_header=False`` runs the engine in
    contributor mode (:func:`main.run` owns the shared header). With ``wait`` a non-SUCCEEDED
    terminal state raises so a failed run never exits 0; the telemetry stamp precedes the raise so a
    failed run still records its sizing.
    """
    from .registry.ids import make_run_id
    from .settings import Settings

    settings = settings or Settings.resolve()
    infra = infra or RayInfra.resolve()
    if n_series is not None:
        cfg = cfg.model_copy(
            update={"data": cfg.data.model_copy(update={"series_limit": n_series})}
        )
    run_id = make_run_id(cfg)
    plan = ray_io.plan_cluster(cfg, models, run_id=run_id)

    # Reuse when the config names a standing cluster or the caller overrides the name; else create.
    reuse = plan.reuse or cluster_name is not None
    name = cluster_name or plan.cluster_name

    config_uri = _stage_config(cfg, run_id, infra)
    entrypoint = build_entrypoint(config_uri, settings, models=models, manage_header=manage_header)
    runtime_env = build_runtime_env()
    regions = _resolve_regions(cfg, settings)
    _log.info(
        "ray submit: run_id=%s cluster=%s reuse=%s cpu_nodes=%d gpu_nodes=%d regions=%s",
        run_id,
        name,
        reuse,
        plan.cpu_node_count,
        plan.gpu_node_count,
        regions,
    )

    cluster_resource_name: str | None = None
    # For an ephemeral run, the teardown target is the *deterministic* resource path. The create
    # helper cleans up each stocked-out region as it advances; this final target covers the region
    # that actually got a cluster (torn down after the job) or a create that reached ERROR after the
    # loop settled on a region. Reuse leaves the standing cluster alone.
    teardown_target: str | None = None
    try:
        if reuse:
            # A standing cluster lives in the data-plane region; pin the SDK there and target it.
            _init_vertex(settings, settings.region)
            cluster_resource_name = _resource_name(settings, name)
        else:
            cluster_resource_name, cluster_region = _create_cluster_across_regions(
                plan, infra, name, settings, regions
            )
            teardown_target = _resource_name(settings, name, cluster_region)

        import time

        cluster = _get_cluster(cluster_resource_name)
        started = time.perf_counter()
        job_id, status = _submit_and_poll(cluster, entrypoint, runtime_env, wait=wait)
        wall_s = round(time.perf_counter() - started, 1) if wait else None

        if wait:
            telemetry = extract_ray_telemetry(
                plan,
                cluster=cluster,
                job_id=job_id,
                job_status=status,
                total_wall_s=wall_s,
                reuse=reuse,
            )
            _stamp_ray_telemetry(telemetry, run_id, settings)
            if status != "SUCCEEDED":
                raise EngineError(f"ray job {job_id} terminal state {status}")
        return job_id
    finally:
        # Guaranteed teardown of an ephemeral cluster — even on a raised job *or a create that
        # errored mid-provision* — mirroring the §10 all_done intent. teardown_target is set (to the
        # deterministic path) for every ephemeral run and None for reuse, so a reused cluster is
        # left standing while a half-created ephemeral one is still cleaned up. _delete_cluster is
        # best-effort, so a target that never actually materialized is a harmless no-op.
        if teardown_target is not None:
            _delete_cluster(teardown_target)


def _resource_name(settings: Settings, name: str, region: str | None = None) -> str:
    """The Vertex persistent-resource path for a cluster display name (reuse targeting; pure).

    ``region`` defaults to ``settings.region`` (the reuse case — a standing cluster lives in the
    data-plane region); the ephemeral fallback path passes the region actually being attempted.
    """
    loc = region or settings.region
    return f"projects/{settings.project_id}/locations/{loc}/persistentResources/{name}"


def main(argv: list[str] | None = None) -> None:
    """CLI: ``python -m scale_forecasting.ray_submit --config run.json [--cluster-name ...]``."""
    from .config import load_config

    p = argparse.ArgumentParser(
        prog="ray_submit", description="Submit a forecast run to Vertex Ray."
    )
    p.add_argument("--config", required=True, help="path to the run config JSON")
    p.add_argument("--n-series", type=int, default=None, help="override series_limit (scale knob)")
    p.add_argument(
        "--cluster-name",
        default=None,
        help="reuse a standing cluster by name (skip create + teardown); else ephemeral",
    )
    p.add_argument("--no-wait", action="store_true", help="return once submitted (don't block)")
    ns = p.parse_args(argv)

    cfg = load_config(ns.config)
    job_id = submit_ray(
        cfg, cluster_name=ns.cluster_name, n_series=ns.n_series, wait=not ns.no_wait
    )
    _log.info("submitted: %s", job_id)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
