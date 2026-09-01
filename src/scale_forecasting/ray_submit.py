"""Submit a forecast run to Ray on Vertex AI — the autoscaling-cluster launcher.

The Ray analog of `submit`: the ``[ray]``-extra, ADC-authenticated helper
that turns a validated `RunConfig` into a run on an **autoscaling**
Vertex Ray cluster (``ray_autoscale=False`` selects fixed
sizing). It owns the cluster lifecycle the way ``submit`` owns the Dataproc batch; the on-cluster
compute is
`run`, reached through the
`ray_entry` Jobs entrypoint.

What `submit_ray` does:

1. **Size the cluster to the run's fan-out** — `plan_cluster` turns the config into a
   ``RayClusterPlan`` (a GPU worker pool for NeuralProphet + a CPU worker pool for everything else).
   By default each pool carries a Vertex ``AutoscalingSpec(min, max)`` and scales with Ray's task
   demand; ``ray_autoscale=False`` gives each pool a fixed ``node_count`` and no spec. Either way
   the whole spec is a pure product of the config — logged and stamped to the run.
2. **Stage the run config** — write the validated config to ``gs://<code>/runs/<run_id>.json`` and
   pass it as ``--config-uri`` (the lossless reproducibility record — same contract as Spark).
3. **Provision (ephemeral default) or target (reuse opt-in) the cluster** — ephemeral:
   ``create_ray_cluster`` from the planned spec, run, then ``delete_ray_cluster`` in a
   ``finally`` (teardown guaranteed even on failure); reuse: ``compute.ray_cluster_name`` /
   ``cluster_name=`` targets a standing cluster by name and skips both create and delete.
4. **Submit the on-cluster driver** — a Ray Job via
   `JobSubmissionClient` (``vertex_ray://<dashboard>``) whose entrypoint
   is ``python -m scale_forecasting.ray_entry`` with the same ``--config-uri`` / ``--models`` /
   ``--manage-header`` / ``--sf-*`` contract the Spark entry uses. Current ``src/`` ships as the
   job's ``runtime_env`` working dir (runtime code delivery, never baked into the image — the same
   code runs locally and in the cloud), with ``requirements.txt`` for the on-cluster deps.
5. **Poll to terminal + stamp telemetry** — with ``wait``, block until the job is terminal, stamp a
   Ray analog of Spark's ``job_telemetry`` (cluster name, node counts, machine/accelerator types,
   calibrated-vs-sizing GPU fraction, wall-clock, job id) **plus the whole sizing decision** (both
   pool plans and the profile they were sized off, filed under ``$.sizing.<family>``) into
   ``run_registry.job_telemetry`` via ``bq.merge_header_telemetry`` — **no schema change** (the JSON
   column already exists), and a merge rather than a whole-column write so the several family jobs
   of one run don't overwrite each other — and raise on a non-SUCCEEDED terminal state so a failed
   run never exits 0.

Public surface: ``RayInfra``, ``submit_ray``, ``build_entrypoint``, ``build_runtime_env``,
``extract_ray_telemetry``, ``main``.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .commands import build_driver_args
from .engines import ray_io
from .errors import ConfigError, EngineError, get_logger
from .resources.audit import sizing_telemetry
from .staging import stage_config

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

# torch's x86_64/linux pin is the CUDA-12.6 local build (``torch==2.13.0+cu126``) for the Vertex T4
# driver — that ``+cuXXX`` local version exists ONLY on the PyTorch index, never on PyPI. The
# on-cluster runtime_env pip install must therefore add the SAME ``--extra-index-url`` the image
# build uses (docker/Dockerfile), or the pin 404s ("No matching distribution for torch==…+cu126")
# and the whole Ray job fails at env setup. ``--extra-index-url`` (not ``--index-url``) keeps PyPI
# primary for every other package; pip honors the option line when it appears in the requirements
# list Ray materializes. Source of truth for the URL is docker/Dockerfile — keep them in lockstep.
_TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu126"

# Ray-cluster infra env vars (beyond the SF_* identity Settings resolves). Kept together so the
# docstring, resolve(), and any tooling agree. code_bucket + compute_sa are shared with the Spark
# batch; network (optional) is a VPC for a private endpoint. There is deliberately no custom-image
# var: Ray always runs on Vertex's prebuilt image + a uv runtime_env (a custom node image fails
# Vertex Ray GPU-node provisioning — see build_runtime_env). SF_CONTAINER_IMAGE stays Spark-only.
_ENV_NETWORK = "SF_RAY_NETWORK"
_ENV_NETWORK_ATTACHMENT = "SF_RAY_NETWORK_ATTACHMENT"
_ENV_COMPUTE_SA = "SF_COMPUTE_SA"
_ENV_CODE_BUCKET = "SF_CODE_BUCKET"
_ENV_RAY_VERSION = "SF_RAY_VERSION"

# Vertex Ray's supported Ray version + our runtime Python. Vertex AI accepts only a fixed set of Ray
# versions for the cluster image (2.9.3 / 2.33.0 / 2.42.0 / 2.47.1; on Python 3.11 only 2.42 or
# 2.47), and the client-side Ray MUST match the cluster's: the JobSubmissionClient handshake (GET
# /api/version) hangs on a version-skewed dashboard rather than erroring cleanly. So the [ray] extra
# is capped to a supported range (see pyproject.toml) and this default matches. Overridable via
# SF_RAY_VERSION to select a different *supported* image without a code change — but the client Ray
# must still equal it.
_DEFAULT_RAY_VERSION = "2.47"
_DEFAULT_PYTHON_VERSION = "3.11"

# Poll cadence + terminal Ray job states (the Jobs API reports these on get_job_status).
_POLL_SECONDS = 15
_TERMINAL_STATES = frozenset({"SUCCEEDED", "FAILED", "STOPPED"})
# On a FAILED job, how many trailing driver-log lines to capture into our log + the raised error —
# enough to carry a Python traceback without dumping the whole (potentially huge) driver stdout.
_FAILURE_LOG_TAIL_LINES = 60

# Dashboard warm-up race: a cluster reaches RUNNING *before* its Ray dashboard is reachable through
# Vertex's public-endpoint proxy, so the first JobSubmissionClient handshake (GET /api/version) can
# get a proxy gateway timeout / connection refusal. We retry the *connection only* with backoff up
# to this budget before giving up — well within the ~few-minutes the dashboard takes to serve.
_DASHBOARD_CONNECT_ATTEMPTS = 20
_DASHBOARD_CONNECT_BACKOFF_SECONDS = 15

# The vertex_ray:// Jobs client mints an OAuth Bearer token (~60-min TTL) at construction and never
# refreshes it (see `_is_auth_expiry_error`). A long GPU run (NeuralProphet) can outlive it, so we
# proactively rebuild the client — minting a fresh token — once it reaches this age, comfortably
# under the TTL, rather than waiting to absorb the 401 the reactive poll path handles as a backstop.
_CLIENT_MAX_AGE_SECONDS = 2700  # 45 min


@dataclass(frozen=True)
class RayInfra:
    """Vertex-Ray infra identity — what launching a cluster needs beyond `Settings`.

    Resolved from ``SF_*`` env (parity with ``Settings`` / ``BatchInfra``) or ``terraform output``.

    Connectivity is one of three modes, in precedence order — the first that is set wins:

    * ``network_attachment`` (**PSC-I**, the supported private path): a network-attachment
      resource name. Vertex's tenant attaches an interface into the VPC through it, and — critically
      — this is the *only* mode under which the managed Ray dashboard / ``JobSubmissionClient``
      handshake (``GET /api/version``) is reachable off-cluster on this org; both public and VPC
      peering leave the proxy→head-node hop dead (a 30s hang → HTTP 524). Excludes ``network``.
    * ``network`` (VPC peering): a VPC (with a private-services connection) for a peered private
      endpoint. Kept for deployments that already run this way, but note the dashboard-handshake
      caveat above — prefer ``network_attachment``.
    * neither set: a public endpoint (Vertex's default; same handshake caveat).

    ``compute_sa`` is the runtime SA the cluster runs as; ``code_bucket`` where the run config JSON
    is staged. There is no custom-image field: Ray always runs on Vertex's prebuilt image and the
    uv ``runtime_env`` installs the deps on top. Unlike the Spark path, Ray never uses a custom node
    image — one fails Vertex Ray GPU-node provisioning (see ``build_runtime_env``), so
    ``SF_CONTAINER_IMAGE`` stays Spark-only and is never read here.
    """

    compute_sa: str
    code_bucket: str
    network: str | None = None
    network_attachment: str | None = None
    ray_version: str = _DEFAULT_RAY_VERSION
    python_version: str = _DEFAULT_PYTHON_VERSION

    @classmethod
    def resolve(cls) -> RayInfra:
        """Build from the ``SF_*`` Ray-infra environment; raise naming the first missing var.

        ``SF_COMPUTE_SA`` and ``SF_CODE_BUCKET`` are required; ``SF_RAY_NETWORK_ATTACHMENT``
        (PSC-I, preferred) and ``SF_RAY_NETWORK`` (VPC peering) are optional — set at most one; if
        both are set the attachment wins. Neither set → public endpoint. Ray always runs on Vertex's
        prebuilt image + a uv ``runtime_env`` (no custom-image var).
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
            network_attachment=os.environ.get(_ENV_NETWORK_ATTACHMENT) or None,
            ray_version=os.environ.get(_ENV_RAY_VERSION) or _DEFAULT_RAY_VERSION,
        )

    @classmethod
    def from_terraform_outputs(cls, outputs: dict[str, str]) -> RayInfra:
        """Build from a ``terraform output -json`` value map (local dev/tests).

        Reads the keys the ``terraform/main`` stage emits — ``compute_sa``, ``code_bucket``, an
        optional ``network_attachment_id`` (PSC-I, preferred) and/or ``network_id`` (VPC peering);
        both absent → public endpoint, and if both are present the attachment wins. Ray always runs
        on Vertex's prebuilt image + a uv ``runtime_env``, so no image is read here.
        """
        try:
            return cls(
                compute_sa=outputs["compute_sa"],
                code_bucket=outputs["code_bucket"],
                network=outputs.get("network_id") or None,
                network_attachment=outputs.get("network_attachment_id") or None,
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

    The Ray analog of `build_batch`'s arg list, as a single shell
    string (what the Jobs API runs on the cluster head). Carries the same contract: ``--config-uri``
    (the staged config, whose digest is the shared ``run_id``), the ``--sf-*`` infra identity, and
    — only when non-default — ``--models m1,m2`` (executed subset) and ``--manage-header
    false`` (contributor mode; `main.run` owns the shared header). Defaults omit these
    flags so a standalone submit builds the plain command.
    """
    parts = ["python", "-m", "scale_forecasting.ray_entry"]
    parts += build_driver_args(config_uri, settings, models=models, manage_header=manage_header)
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
    `_CLUSTER_PROVIDED` (see its note — Ray must come from the image, not pip).
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
    """The Ray ``runtime_env``: current ``src/`` + on-cluster deps installed by uv (pure).

    Delivers code at RUNTIME the way the Spark path uploads a ``src/`` zip: ``working_dir`` is the
    package root, so ``python -m scale_forecasting.ray_entry`` imports the code that was just
    submitted, not anything baked into the image — the "same code local and in the cloud" seam.

    Ray always runs on Vertex's prebuilt image (a custom node image fails Vertex Ray GPU-node
    provisioning), which lacks our deps — so **uv** installs the requirements package **list minus
    Ray** into the per-job virtualenv (`_requirements_packages` — Ray stays the image's pinned
    version rather than being swapped by a conflicting pin). uv resolves from the same pinned
    requirements export the container is built from, so the on-cluster env is byte-aligned with
    every other surface, and it installs markedly faster than pip; Ray 2.47's runtime_env uv plugin
    self-bootstraps uv into the prebuilt image if absent, so nothing has to preinstall it.
    ``--extra-index-url`` adds the PyTorch CUDA wheels (`_TORCH_CUDA_INDEX`, mirroring
    docker/Dockerfile): the x86_64/linux torch pin is a ``+cu126`` local build that only resolves
    from that index, and ``--index-strategy unsafe-best-match`` lets uv pick it from the extra index
    even though the same name exists on PyPI (uv's default first-index strategy would stop at PyPI
    and never find the ``+cu126`` build). PyPI stays the primary index.
    """
    return {
        "working_dir": str(_SRC_DIR),
        "uv": {
            "packages": _requirements_packages(),
            # Passed through to ``uv pip install`` — this REPLACES the plugin default
            # ``["--no-cache"]``, so re-list it. See the docstring for why the extra index +
            # unsafe-best-match are needed for the ``+cu126`` torch build.
            "uv_pip_install_options": [
                "--no-cache",
                "--extra-index-url",
                _TORCH_CUDA_INDEX,
                "--index-strategy",
                "unsafe-best-match",
            ],
            # Run ``uv pip check`` after install so dependency drift fails loudly at env setup
            # rather than as a confusing runtime import error — the byte-alignment guarantee.
            "uv_check": True,
        },
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

    The Ray analog of `extract_job_telemetry`, answering the same
    operability questions — *how big was the pool (and its elastic bounds), what did it cost in
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
        # Elastic spec: the flag + per-pool bounds the cluster was created with, so
        # v_run_summary shows whether/how the pools autoscaled. node_count above is the derived
        # fixed-size-equivalent (the reference size; under autoscaling the pool starts at min).
        "autoscale": plan.autoscale,
        "cpu_min_nodes": plan.cpu_min_nodes,
        "cpu_max_nodes": plan.cpu_max_nodes,
        "gpu_min_nodes": plan.gpu_min_nodes,
        "gpu_max_nodes": plan.gpu_max_nodes,
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
    """Stage the run config to GCS and return its URI (see `staging.stage_config`).

    Shares the one staging helper with the Spark path, so a mixed run stages one config the same
    way regardless of runtime — the JSON *is* the shared reproducibility record, and its digest is
    the shared ``run_id``.
    """
    return stage_config(cfg, run_id, infra.code_bucket)


# --- I/O: Vertex Ray cluster lifecycle -----------------------------------------


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
    the uv ``runtime_env`` delivers the deps (see ``build_runtime_env``).
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


# Substrings that mark a *regional quota* ceiling. GPU/accelerator quota on Vertex is granted
# per-region, so a region that is over its quota says nothing about the next region's ceiling — the
# fallback advances on these just as it does on capacity stockouts. A quota error is distinct from a
# capacity stockout (the region has room, this project is simply not allowed more), so it gets its
# own classifier rather than widening the capacity markers.
_QUOTA_ERROR_MARKERS = (
    "quota exceeded",
    "exceeds quota",
    "exceed quota",
    "exceeded quota",
    "quota limit",
)


def _is_quota_error(message: str) -> bool:
    """True if a cluster-create error message signals a *regional quota* ceiling (pure).

    Vertex accelerator quota is per-region, so a quota-exhausted region is worth retrying elsewhere:
    another region carries its own independent ceiling. (A capacity stockout is a different reason
    with the same remedy — hop — and is classified by `_is_capacity_error`.)
    """
    low = message.lower()
    return any(marker in low for marker in _QUOTA_ERROR_MARKERS)


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
    capacity* failure (`_is_capacity_error`) or a *regional quota* ceiling (`_is_quota_error`) the
    failed attempt's (deterministic) resource is torn down and the next region tried — both are
    per-region conditions another region may not share. Any *other* error (bad machine type,
    permission, bad config) is re-raised at once because another region won't fix it. Exhausting
    every region raises `EngineError` naming the regions tried.

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
            resource_path = _resource_name(settings, name, region)
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
            capacity = _is_capacity_error(message)
            quota = _is_quota_error(message)
            generic_provision_error = not detail and _is_generic_cluster_error(str(exc))
            if not (capacity or quota or generic_provision_error):
                raise
            reason = "quota ceiling" if quota and not capacity else "insufficient capacity"
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


def _is_auth_expiry_error(exc: Exception) -> bool:
    """True if ``exc`` is an expired-credential ``401`` from the dashboard proxy (refresh & retry).

    Distinct from `_is_dashboard_warmup_error`, which treats a 401 as a *connect-time* fault
    that won't fix itself by waiting (right — spinning the warm-up loop on bad auth is pointless).
    During a *long poll*, however, a 401 means something different: the ``vertex_ray://`` Jobs
    client caches an OAuth Bearer token minted at construction (~60-min TTL), so a run outliving the
    token gets a 401 on the next ``get_job_status`` even though nothing is wrong — rebuilding the
    client mints a fresh token and the poll resumes. Match the 401 shapes the proxy returns.
    """
    low = str(exc).lower()
    return " 401" in low or "unauthorized" in low


def _client_needs_refresh(
    born_monotonic: float,
    now_monotonic: float,
    max_age_s: float = _CLIENT_MAX_AGE_SECONDS,
) -> bool:
    """True once the Jobs client is old enough that its cached OAuth token may be nearing expiry.

    The ``vertex_ray://`` client mints a Bearer token (~60-min TTL) at construction and never
    refreshes it; rebuilding *before* the TTL keeps a long poll authenticated. Pure and
    time-injected so the age policy is unit-testable without any live Ray I/O.
    """
    return (now_monotonic - born_monotonic) >= max_age_s


def _connect_job_client(
    cluster_resource_name: str,
) -> Any:  # pragma: no cover - live Ray Jobs I/O, exercised by the @gpu smoke
    """Open a ``JobSubmissionClient`` to the cluster, retrying past the warm-up race.

    We address the cluster by its **resource name**
    (``vertex_ray://projects/<num>/locations/<region>/persistentResources/<name>``): the
    ``[ray]``-extra resolver discovers the dashboard endpoint and authenticates the connection
    itself. Submission routes through a Google-managed dashboard proxy.

    Importing ``vertex_ray`` here is **load-bearing, not cosmetic**: the plugin registers the
    ``vertex_ray://`` address handler *and* injects the OAuth Bearer token into the dashboard
    handshake — Google's docs state it is "required to obtain authentication automatically." Without
    it in the process that builds the client, the ``GET /api/version`` request reaches the proxy
    without valid auth and the proxy holds it open until it times out (HTTP 524) instead of
    returning a clean 401. The SDK's project/location is already bound upstream (``_init_vertex`` on
    both the create and reuse paths); the import is idempotent, so this is belt-and-suspenders.

    ``JobSubmissionClient.__init__`` does a GET ``/api/version`` handshake; right after the cluster
    hits RUNNING that endpoint may not be reachable yet, so the first attempts can raise a proxy
    gateway timeout (524/504/…). We back off and retry the *connection only* (never a partial
    submit) until it succeeds or the budget is spent, then let the last error propagate.
    """
    from google.cloud.aiplatform import vertex_ray  # noqa: F401 - registers vertex_ray:// + auth
    from ray.job_submission import JobSubmissionClient

    last_exc: Exception | None = None
    for attempt in range(1, _DASHBOARD_CONNECT_ATTEMPTS + 1):
        try:
            return JobSubmissionClient(f"vertex_ray://{cluster_resource_name}")
        except Exception as exc:  # noqa: BLE001 - classify, retry transients, re-raise faults
            if not _is_dashboard_warmup_error(exc):
                raise
            last_exc = exc
            _log.info(
                "Ray dashboard not ready yet (attempt %d/%d): %r",
                attempt,
                _DASHBOARD_CONNECT_ATTEMPTS,
                exc,
            )
        _log.info("retrying Ray dashboard connect in %ds", _DASHBOARD_CONNECT_BACKOFF_SECONDS)
        time.sleep(_DASHBOARD_CONNECT_BACKOFF_SECONDS)
    assert last_exc is not None
    raise last_exc


def _submit_and_poll(
    cluster_resource_name: str,
    entrypoint: str,
    runtime_env: dict[str, Any],
    *,
    wait: bool,
    submission_id: str | None = None,
) -> tuple[str, str, str]:  # pragma: no cover - live Ray Jobs I/O, exercised by the @gpu smoke
    """Submit the on-cluster driver as a Ray Job and (when ``wait``) poll to a terminal state.

    Connects the Jobs client to the cluster by resource name (``vertex_ray://<resource_name>``,
    retrying past the dashboard warm-up race), submits ``entrypoint`` with ``runtime_env`` (current
    ``src/`` + requirements), and returns ``(job_id, status, detail)``. ``submission_id``, when set,
    is passed to ``submit_job`` so the Ray job's own id is the deterministic ``job_key`` rather than
    a random auto-assigned one; the returned ``job_id`` then equals it. ``detail`` is empty except
    on a ``FAILED`` terminal state, where it carries the driver's error message + log tail
    (`_fetch_job_failure_detail`) captured at the moment of failure — so the cause is recorded
    even after the ``ml_job`` log stream ages out. Without ``wait`` the status is the immediate
    post-submit state (the caller skips telemetry + the terminal-state check).
    """
    client = _connect_job_client(cluster_resource_name)
    client_born = time.monotonic()
    submit_kwargs: dict[str, Any] = {"entrypoint": entrypoint, "runtime_env": runtime_env}
    if submission_id is not None:
        submit_kwargs["submission_id"] = submission_id
    job_id = client.submit_job(**submit_kwargs)
    _log.info("submitted Ray job %s", job_id)

    def _fresh_client() -> Any:
        # Proactively re-mint the OAuth token BEFORE it dies (see `_client_needs_refresh`): the
        # vertex_ray:// client caches a Bearer token (~60-min TTL) at construction, and a long GPU
        # run (NeuralProphet) can outlive it. Rebuilding at 45 min keeps every poll authenticated,
        # so we never even take the 401 the reactive branch below would otherwise absorb.
        nonlocal client, client_born
        if _client_needs_refresh(client_born, time.monotonic()):
            _log.info("Ray Jobs client nearing token TTL; proactively refreshing")
            client = _connect_job_client(cluster_resource_name)
            client_born = time.monotonic()
        return client

    def _status() -> str:
        # Backstop: if the proactive refresh ever misses (clock skew / a rebuild that lands late), a
        # 401 is still recoverable — rebuild the client (fresh token) and retry once, so a long run
        # polls to completion instead of aborting.
        nonlocal client, client_born
        try:
            return str(_fresh_client().get_job_status(job_id))
        except Exception as exc:  # noqa: BLE001 - only a 401 is recoverable here; re-raise the rest
            if not _is_auth_expiry_error(exc):
                raise
            _log.info("Ray job poll hit auth expiry (%r); refreshing client and retrying", exc)
            client = _connect_job_client(cluster_resource_name)
            client_born = time.monotonic()
            return str(client.get_job_status(job_id))

    if not wait:
        return job_id, _status(), ""

    status = _status()
    while status not in _TERMINAL_STATES:
        time.sleep(_POLL_SECONDS)
        status = _status()
    _log.info("Ray job %s finished: status=%s", job_id, status)
    # Use the age-checked client here too: a token dying right at terminal-FAILED would otherwise
    # cost us the driver diagnosis (`_fetch_job_failure_detail` is best-effort and unwrapped).
    detail = _fetch_job_failure_detail(_fresh_client(), job_id) if status == "FAILED" else ""
    if detail:
        _log.error("Ray job %s FAILED — driver diagnosis:\n%s", job_id, detail)
    return job_id, status, detail


def _fetch_job_failure_detail(
    client: Any, job_id: str
) -> str:  # pragma: no cover - live Ray Jobs I/O, exercised by the @gpu smoke
    """Best-effort driver error message + log tail for a FAILED Ray job (operability).

    A terminal ``FAILED`` status alone says *nothing* about the cause; the driver's Python
    traceback lives in the Ray dashboard and Cloud Logging's ``ml_job`` stream, which ages out
    of the default freshness window within ~90 min — so a failure diagnosed later is a failure
    diagnosed by archaeology. The Jobs client already holds both facts: ``get_job_info().message``
    (the terminal error line) and ``get_job_logs()`` (the full driver stdout/stderr). Pull them at
    the moment of failure so the cause is captured in *our* log and folded into the raised
    `EngineError` — never dependent on a still-warm log stream.
    Every step is defensive: a diagnosis that itself fails must not mask the underlying job failure.
    """
    parts: list[str] = []
    try:
        info = client.get_job_info(job_id)
        message = getattr(info, "message", None)
        if message:
            parts.append(f"message: {message}")
    except Exception as exc:  # noqa: BLE001 - diagnosis is best-effort, never fatal
        _log.warning("could not fetch Ray job info for %s: %r", job_id, exc)
    try:
        logs = client.get_job_logs(job_id) or ""
        tail = "\n".join(logs.splitlines()[-_FAILURE_LOG_TAIL_LINES:]).strip()
        if tail:
            parts.append(f"driver log tail:\n{tail}")
    except Exception as exc:  # noqa: BLE001 - diagnosis is best-effort, never fatal
        _log.warning("could not fetch Ray job logs for %s: %r", job_id, exc)
    return "\n".join(parts)


def _stamp_ray_telemetry(
    telemetry: dict[str, Any],
    run_id: str,
    settings: Settings,
    *,
    sizing: dict[str, Any] | None = None,
) -> None:
    """Write the Ray telemetry dict to the run header's native JSON column (best-effort).

    The pure `extract_ray_telemetry` output, merged into ``job_telemetry`` a key at a time
    (`registry.bq.merge_header_telemetry`) rather than written whole — several family jobs of one
    run each land here, and a whole-column write would leave only whichever finished last. The
    column is a native ``JSON`` type whose query parameter serializes the value itself, so we pass
    **dicts** (not pre-serialized strings, which would double-encode).

    ``sizing`` (`resources.audit.sizing_telemetry` over the two pool plans) is filed under
    ``$.sizing.<family>``: what the pools were sized to hold, and off whose measurements.

    Wrapped so any failure (API error, header not yet written) is logged and swallowed: telemetry
    is a nice-to-have overlay on an already-complete run, never a reason to fail it.
    """
    from .registry import bq

    patch = dict(telemetry)
    if sizing:
        patch[bq.sizing_telemetry_path(sizing)] = sizing
    try:
        bq.merge_header_telemetry(run_id, patch, settings=settings)
        _log.info("Ray telemetry stamped for run %s: %s", run_id, telemetry)
    except Exception as exc:  # noqa: BLE001 - telemetry is best-effort, never fatal
        _log.warning("Ray telemetry capture failed (non-fatal): %r", exc)


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
    _delete_cluster(_resource_name(settings, name, region))


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
    submission_id: str | None = None,
    use_gpu: bool | None = None,
    gpu_type: str | None = None,
    cluster_region: str | None = None,
) -> tuple[str, str, str]:
    """Size, provision, run, and (ephemeral) tear down a Ray-on-Vertex forecast run.

    Returns ``(job_id, cluster_resource_name, region)`` — the Ray job's id plus the coordinates a
    reader needs to probe it later: the cluster's Vertex persistent-resource path and the region it
    actually landed in (the reuse target's region, or the region an ephemeral create settled on
    after any capacity hop).

    The Ray analog of `submit_batch`. Resolves infra from the
    environment when not passed, sizes an **autoscaling** cluster to the run's fan-out
    (`plan_cluster`; ``ray_autoscale=False`` for fixed),
    stages the full config to GCS (so its ``run_id`` matches `main.run`'s), then runs the
    lifecycle:

    * **ephemeral (default):** create the planned cluster → submit the on-cluster driver as a Ray
      Job → (with ``wait``) poll to terminal + stamp telemetry → ``delete_ray_cluster`` in a
      ``finally`` so teardown happens even if the job raises.
    * **reuse (opt-in):** ``cluster_name`` (or ``compute.ray_cluster_name``) targets a standing
      cluster by name — skip create *and* skip delete; the plan still records the size it *should*
      be.

    ``n_series`` overrides ``series_limit`` at submit time (the scale knob — a different scale is a
    different fixed plan *and* a distinct ``run_id``/header, so each scale is its own queryable run;
    this is how "resize for a larger/smaller scale" is driven). ``models`` / ``manage_header`` carry
    the on-cluster contract: the full ``cfg`` is always staged (shared ``run_id``) while ``models``
    restricts the on-cluster executed subset and ``manage_header=False`` runs the engine in
    contributor mode (`main.run` owns the shared header). With ``wait`` a non-SUCCEEDED
    terminal state raises so a failed run never exits 0; the telemetry stamp precedes the raise so a
    failed run still records its sizing.

    ``submission_id``, when set (the orchestrator passes the family's deterministic ``job_key`` via
    `registry.ids.ray_submission_id`), becomes the Ray job's own id — so families sharing a
    ``run_id`` get distinct, directly-queryable Ray jobs instead of random auto-assigned ids. When
    ``None`` Ray assigns one (the standalone path).

    ``use_gpu``/``gpu_type`` override the flat ``compute`` GPU defaults for this family's job (the
    DAG orchestrator passes the family's resolved hardware — e.g. the deep-learning family gets a
    GPU pool even when the flat default is CPU). They size the cluster only; they're kept out of
    ``cfg`` so the staged config's ``run_id`` stays identical across every family in the run.

    ``cluster_region`` pins where a *reused* cluster is targeted (SDK init + resource path). It
    defaults to ``settings.region`` — the standing-cluster case — but the DAG orchestrator passes
    the region a shared ephemeral cluster landed in (which may differ after a capacity hop), so
    every Ray family's job finds the one shared cluster by name in the right region.
    """
    from .profiling.source import profile_for_run
    from .registry.ids import make_run_id
    from .settings import Settings

    settings = settings or Settings.resolve()
    infra = infra or RayInfra.resolve()
    cfg = cfg.with_series_limit(n_series)
    run_id = make_run_id(cfg)
    # A past run's measurements, when `compute.profile.source` points at any. The pools are sized
    # here, before any node exists, so the only evidence available is somebody else's run — the
    # same position both Spark submit paths are in. The engine's own in-run `resolve_profile` sizes
    # *tasks* on the cluster this produces; it cannot reach back and change the cluster.
    profile = profile_for_run(cfg, settings=settings)
    plan = ray_io.plan_cluster(
        cfg, models, run_id=run_id, use_gpu=use_gpu, gpu_type=gpu_type, profile=profile
    )

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
            # A reused cluster (a standing one, or the run's shared ephemeral one) lives in a known
            # region — the data-plane region by default, or the region a shared cluster landed in
            # after a capacity hop (passed as cluster_region). Pin the SDK there and target it.
            region = cluster_region or settings.region
            _init_vertex(settings, region)
            cluster_resource_name = _resource_name(settings, name, region)
        else:
            cluster_resource_name, cluster_region = _create_cluster_across_regions(
                plan, infra, name, settings, regions
            )
            region = cluster_region
            teardown_target = _resource_name(settings, name, cluster_region)

        cluster = _get_cluster(cluster_resource_name)
        started = time.perf_counter()
        job_id, status, detail = _submit_and_poll(
            cluster_resource_name, entrypoint, runtime_env, wait=wait, submission_id=submission_id
        )
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
            _stamp_ray_telemetry(
                telemetry,
                run_id,
                settings,
                sizing=sizing_telemetry(
                    plan.cpu_pool,
                    plan.gpu_pool,
                    profile=profile,
                    # This job's own label, not a pool's: a deep-learning job still has a (fallback)
                    # CPU pool, and filing the record under that pool's ``"cpu"`` would bury it.
                    family="+".join(ray_io.pool_families(models or cfg.models)) or "cpu",
                ),
            )
            if status != "SUCCEEDED":
                # Fold the driver diagnosis (captured before the log stream ages out) into the
                # raised error so the *cause* — not just "FAILED" — reaches the operator.
                suffix = f"\n{detail}" if detail else ""
                raise EngineError(f"ray job {job_id} terminal state {status}{suffix}")
        return job_id, cluster_resource_name, region
    finally:
        # Guaranteed teardown of an ephemeral cluster — even on a raised job *or a create that
        # errored mid-provision*. teardown_target is set (to the
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
    from .config import load_config_uri

    p = argparse.ArgumentParser(
        prog="ray_submit", description="Submit a forecast run to Vertex Ray."
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", help="path to the run config JSON")
    src.add_argument("--config-uri", help="gs:// URI of a staged config (portable source)")
    p.add_argument("--n-series", type=int, default=None, help="override series_limit (scale knob)")
    p.add_argument(
        "--cluster-name",
        default=None,
        help="reuse a standing cluster by name (skip create + teardown); else ephemeral",
    )
    p.add_argument("--no-wait", action="store_true", help="return once submitted (don't block)")
    ns = p.parse_args(argv)

    cfg = load_config_uri(ns.config or ns.config_uri)
    job_id, _, _ = submit_ray(
        cfg, cluster_name=ns.cluster_name, n_series=ns.n_series, wait=not ns.no_wait
    )
    _log.info("submitted: %s", job_id)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
