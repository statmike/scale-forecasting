"""Submit a run (or family slice) to Google Kubernetes Engine (`runtime="gke"` / `ray_mode="gke"`).

Why a Google Kubernetes Engine runtime beside Spark, Vertex AI, and Compute Engine:
1. **Kubernetes Indexed Jobs (``gke_mode="job"``, default)**:
   - Dispatches a Kubernetes ``batch/v1`` Indexed Job (``completionMode: Indexed``,
     ``completions = W``, ``parallelism = W``) running ``SF_CONTAINER_IMAGE`` with dynamic GCS code
     delivery (``VERTEX_BOOTSTRAP_CODE`` -> ``scale_forecasting.vertex_entry``).
   - Kubernetes automatically injects ``JOB_COMPLETION_INDEX`` (``0 .. W-1``) into each pod, which
     `engines.vertex_engine.resolve_worker_topology` parses into ``WorkerTopology(rank=r,
     world_size=W, source="k8s_indexed_job")``.
   - Supports single-pod execution (``workers = 1``), multi-pod local series-range pushdown
     (``workers = N`` via `engines.vertex_engine.build_worker_series_range`), $1$-pod-per-model
     parallel global/deep-learning training (`engines.vertex_engine.partition_models_for_worker`),
     and GCS completion barriers (`SF_VERTEX_JOB_ID`).
2. **Ray on GKE (``gke_mode="ray"`` or ``runtime="ray"`` with ``ray_mode="gke"``)**:
   - Runs `engines.ray_engine.run` on GKE using ``SF_CONTAINER_IMAGE`` (``ray==2.59.0``),
     bypassing Vertex Managed Ray's custom-image and Ray-version constraints while preserving
     fractional GPU packing and dynamic chunk scheduling.
   - Emits both native Kubernetes Ray manifests (Headless Service + Worker Deployment + Driver Job)
     and KubeRay ``ray.io/v1`` ``RayCluster`` specifications.
3. **Zero-Idle Standing Clusters (Autopilot / Node Autoprovisioning) & Ephemeral Lifecycle**:
   - Targets an existing GKE cluster when ``compute.gke_cluster_name`` or ``SF_GKE_CLUSTER`` is set
     (allowing sub-15-second warm pod startup and scale-to-zero node pools), or provisions an
     ephemeral GKE cluster per run and tears it down unconditionally in ``finally``.
   - Communicates with both the GKE Control Plane API (``https://container.googleapis.com/v1``) and
     the Kubernetes API Server (``https://<cluster.endpoint>``) directly via
     ``google.auth.transport.requests.AuthorizedSession`` + the cluster's CA certificate — requiring
     zero ``kubectl``, ``gke-gcloud-auth-plugin``, or third-party ``kubernetes`` client binaries.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import os
import re
import tempfile
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from . import capacity, quota, staging
from .batch_infra import BatchInfra
from .commands import build_driver_args
from .compute_fallback import US_ZONES, Candidate, resolve_candidates
from .config import RunConfig, resolve_vm_machine_type
from .engines import ray_io
from .engines.ray_io import RayClusterPlan, pool_families, resolve_job_gpu
from .engines.vertex_engine import effective_worker_count, plan_vertex_pool
from .errors import ConfigError, EngineError, configure_cli_logging, get_logger, require_extra
from .job_outcome import launch_window_start
from .job_wait import is_stalled
from .probes.vocabulary import ProbeHandle
from .profiling.source import profile_for_run
from .registry.header import merge_header_telemetry, sizing_telemetry_path
from .registry.ids import gke_job_id, make_run_id
from .registry.jobs import update_job
from .resources.audit import sizing_telemetry
from .resources.catalog import intraop_env_vars, machine_cores, machine_memory_bytes
from .resources.fleet import RuntimeResourcePlan
from .router import split_by_runtime
from .settings import Settings
from .vertex_submit import VERTEX_BOOTSTRAP_CODE

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .profiling.cost import ComputeProfile

_log = get_logger(__name__)

_GKE_API = "https://container.googleapis.com/v1"
_DEFAULT_BOOT_DISK_GB = 100
_POLL_INTERVAL_SECONDS = 10.0
_WATCHDOG_CHECK_SECONDS = 300.0
_HTTP_TIMEOUT_S = 30.0
_GKE_OP_TIMEOUT_S = 1800.0

_GKE_ACCELERATOR_TYPES = {
    "T4": "nvidia-tesla-t4",
    "L4": "nvidia-l4",
    "A100": "nvidia-tesla-a100",
    "A100_80GB": "nvidia-a100-80gb",
}

# Bootstrap snippet for Ray-on-GKE driver/head pods. Downloads `scale_forecasting-<hash>.zip`,
# extracts it into a local directory added to `PYTHONPATH` so all Ray worker processes spawned by
# `raylet` inherit the staged package, starts a local Ray head daemon when `RAY_ADDRESS` is not
# pre-configured, waits for any companion worker pods (`SF_RAY_EXPECTED_NODES`), and invokes
# `scale_forecasting.ray_entry.main`.
RAY_GKE_BOOTSTRAP_CODE = (
    "import os, subprocess, sys, tempfile, time, zipfile, google.cloud.storage as gcs\n"
    "td = tempfile.mkdtemp(prefix='sf_ray_gke_')\n"
    "os.environ.setdefault('MPLCONFIGDIR', os.path.join(td, 'mpl'))\n"
    "if not os.access(os.getcwd(), os.W_OK):\n"
    "    os.chdir(td)\n"
    "argv = sys.argv[1:]\n"
    "idx = argv.index('--package-uri')\n"
    "uri = argv[idx + 1]\n"
    "rest = argv[:idx] + argv[idx + 2:]\n"
    "b, p = uri[5:].split('/', 1)\n"
    "zp = os.path.join(td, 'scale_forecasting.zip')\n"
    "gcs.Client().bucket(b).blob(p).download_to_filename(zp)\n"
    "code_dir = os.path.join(td, 'code')\n"
    "with zipfile.ZipFile(zp, 'r') as zf:\n"
    "    zf.extractall(code_dir)\n"
    "py_path = os.environ.get('PYTHONPATH', '')\n"
    "os.environ['PYTHONPATH'] = f'{code_dir}:{py_path}' if py_path else code_dir\n"
    "sys.path.insert(0, code_dir)\n"
    "started_local_head = False\n"
    "if not os.environ.get('RAY_ADDRESS'):\n"
    "    cmd = [\n"
    "        '/opt/venv/bin/ray', 'start', '--head', '--port=6379',\n"
    "        '--include-dashboard=false', '--disable-usage-stats',\n"
    "    ]\n"
    "    subprocess.run(cmd, check=True)\n"
    "    os.environ['RAY_ADDRESS'] = 'auto'\n"
    "    started_local_head = True\n"
    "expected = int(os.environ.get('SF_RAY_EXPECTED_NODES', '1'))\n"
    "if expected > 1:\n"
    "    import ray\n"
    "    ray.init(address=os.environ['RAY_ADDRESS'])\n"
    "    deadline = time.time() + 600\n"
    "    while time.time() < deadline:\n"
    "        alive = [n for n in ray.nodes() if n.get('Alive')]\n"
    "        if len(alive) >= expected:\n"
    "            break\n"
    "        time.sleep(2)\n"
    "    ray.shutdown()\n"
    "from scale_forecasting.ray_entry import main\n"
    "try:\n"
    "    main(rest)\n"
    "finally:\n"
    "    if started_local_head:\n"
    "        subprocess.run(['/opt/venv/bin/ray', 'stop'], check=False)\n"
)

# Bootstrap snippet for Ray-on-GKE companion worker pods when `worker_count > 1`.
RAY_GKE_WORKER_BOOTSTRAP_CODE = (
    "import os, socket, subprocess, sys, tempfile, time, zipfile, google.cloud.storage as gcs\n"
    "td = tempfile.mkdtemp(prefix='sf_ray_worker_')\n"
    "os.environ.setdefault('MPLCONFIGDIR', os.path.join(td, 'mpl'))\n"
    "if not os.access(os.getcwd(), os.W_OK):\n"
    "    os.chdir(td)\n"
    "uri = os.environ['SF_PACKAGE_URI']\n"
    "b, p = uri[5:].split('/', 1)\n"
    "zp = os.path.join(td, 'scale_forecasting.zip')\n"
    "gcs.Client().bucket(b).blob(p).download_to_filename(zp)\n"
    "code_dir = os.path.join(td, 'code')\n"
    "with zipfile.ZipFile(zp, 'r') as zf:\n"
    "    zf.extractall(code_dir)\n"
    "py_path = os.environ.get('PYTHONPATH', '')\n"
    "os.environ['PYTHONPATH'] = f'{code_dir}:{py_path}' if py_path else code_dir\n"
    "head_addr = os.environ['SF_RAY_HEAD_ADDRESS']\n"
    "host, port_s = head_addr.rsplit(':', 1)\n"
    "port = int(port_s)\n"
    "resolved_addr = head_addr\n"
    "deadline = time.time() + 600\n"
    "while time.time() < deadline:\n"
    "    try:\n"
    "        with socket.create_connection((host, port), timeout=5):\n"
    "            resolved_addr = f'{socket.gethostbyname(host)}:{port}'\n"
    "            break\n"
    "    except OSError:\n"
    "        time.sleep(2)\n"
    "subprocess.run(\n"
    "    [\n"
    "        '/opt/venv/bin/ray', 'start', f'--address={resolved_addr}',\n"
    "        '--block', '--disable-usage-stats',\n"
    "    ],\n"
    "    check=True,\n"
    ")\n"
)


def _ephemeral_cluster_name(job_name: str) -> str:
    """Derive a GKE-legal cluster name (<= 40 chars, RFC-1035) from ``job_name`` (pure)."""
    cleaned = re.sub(r"[^a-z0-9-]+", "-", job_name.lower()).strip("-")
    if not cleaned or not cleaned[0].isalpha():
        cleaned = f"sf-{cleaned}"
    return cleaned[:40].rstrip("-")


@dataclass(frozen=True)
class GkeJobPlan:
    """Resolved execution plan for a Google Kubernetes Engine run (pure)."""

    job_name: str
    gke_mode: str
    cluster_name: str
    ephemeral_cluster: bool
    namespace: str
    machine_type: str
    accelerator_type: str
    accelerator_count: int
    worker_count: int
    requested_workers: int
    hardware: str
    gpu_type: str | None
    image_uri: str = ""
    package_uri: str = ""
    config_uri: str = ""
    service_account: str = ""
    subnetwork_uri: str = ""
    boot_disk_gb: int = _DEFAULT_BOOT_DISK_GB
    resource_plan: RuntimeResourcePlan | None = None
    ray_cluster_plan: RayClusterPlan | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe summary for `run_jobs.job_telemetry.$.gke_job`."""
        return {
            "job_name": self.job_name,
            "gke_mode": self.gke_mode,
            "cluster_name": self.cluster_name,
            "ephemeral_cluster": self.ephemeral_cluster,
            "namespace": self.namespace,
            "machine_type": self.machine_type,
            "accelerator_type": self.accelerator_type,
            "accelerator_count": self.accelerator_count,
            "worker_count": self.worker_count,
            "requested_workers": self.requested_workers,
            "hardware": self.hardware,
            "gpu_type": self.gpu_type,
            "image_uri": self.image_uri,
            "package_uri": self.package_uri,
            "config_uri": self.config_uri,
            "subnetwork_uri": self.subnetwork_uri,
            "boot_disk_gb": self.boot_disk_gb,
            "resource_plan": (
                self.resource_plan.to_dict() if self.resource_plan is not None else None
            ),
            "ray_cluster_plan": (
                {
                    "cluster_name": self.ray_cluster_plan.cluster_name,
                    "cpu_machine_type": self.ray_cluster_plan.cpu_machine_type,
                    "cpu_node_count": self.ray_cluster_plan.cpu_node_count,
                    "gpu_machine_type": self.ray_cluster_plan.gpu_machine_type,
                    "gpu_node_count": self.ray_cluster_plan.gpu_node_count,
                    "total_worker_nodes": self.ray_cluster_plan.total_worker_nodes,
                }
                if self.ray_cluster_plan is not None
                else None
            ),
        }


def plan_gke_job(
    cfg: RunConfig,
    models: Sequence[str] | None = None,
    *,
    run_id: str | None = None,
    job_id: str | None = None,
    gke_mode: str | None = None,
    cluster_name: str | None = None,
    namespace: str | None = None,
    hardware: str | None = None,
    gpu_type: str | None = None,
    machine_type: str | None = None,
    worker_count: int | None = None,
    accelerator_count: int | None = None,
    image_uri: str = "",
    package_uri: str = "",
    config_uri: str = "",
    service_account: str = "",
    subnetwork_uri: str = "",
    profile: ComputeProfile | None = None,
    infra: BatchInfra | None = None,
) -> GkeJobPlan:
    """Resolve a `GkeJobPlan` from `cfg` and per-family overrides (pure)."""
    rid = run_id or make_run_id(cfg)
    python_models, _ = split_by_runtime(cfg)
    selected_models = list(models) if models is not None else python_models

    resolved_mode = gke_mode or cfg.compute.gke_mode
    if resolved_mode not in ("job", "ray"):
        raise ConfigError(f"gke_mode must be 'job' or 'ray' (got {resolved_mode!r})")

    if hardware is None or gpu_type is None:
        inferred_gpu, inferred_type = resolve_job_gpu(cfg)
        resolved_hw = hardware or ("gpu" if inferred_gpu else "cpu")
        resolved_gpu_type = (gpu_type or inferred_type) if resolved_hw == "gpu" else None
    else:
        resolved_hw = hardware
        resolved_gpu_type = gpu_type if resolved_hw == "gpu" else None

    if resolved_gpu_type:
        resolved_gpu_type = resolved_gpu_type.upper()

    eff_accel = (
        (accelerator_count if accelerator_count is not None else cfg.compute.accelerator_count)
        if resolved_hw == "gpu"
        else 0
    )

    resolved_machine = resolve_vm_machine_type(
        resolved_hw,
        resolved_gpu_type,
        cfg.compute.machine_type,
        machine_type,
        accelerator_count=eff_accel if resolved_hw == "gpu" else 1,
    )

    if resolved_hw == "gpu" and resolved_gpu_type:
        accel_type = _GKE_ACCELERATOR_TYPES.get(
            resolved_gpu_type, f"nvidia-{resolved_gpu_type.lower()}"
        )
        accel_count = eff_accel
    else:
        accel_type = ""
        accel_count = 0

    resolved_job_name = gke_job_id(job_id) if job_id else gke_job_id(f"sf-{rid}-gke")
    standing_cluster = (
        cluster_name
        or cfg.compute.gke_cluster_name
        or (infra.gke_cluster_name if infra is not None else None)
        or (cfg.compute.ray_cluster_name if resolved_mode == "ray" else None)
    )
    ephemeral = standing_cluster is None
    resolved_cluster = (
        _ephemeral_cluster_name(resolved_job_name) if ephemeral else str(standing_cluster)
    )
    resolved_ns = namespace or cfg.compute.gke_namespace or "default"

    ray_plan: RayClusterPlan | None = None
    if resolved_mode == "ray":
        ray_plan = ray_io.plan_cluster(
            cfg,
            selected_models,
            run_id=rid,
            use_gpu=(resolved_hw == "gpu"),
            gpu_type=resolved_gpu_type,
            profile=profile,
        )
        requested = (
            worker_count
            if worker_count is not None
            else max(1, ray_plan.cpu_node_count + ray_plan.gpu_node_count)
        )
        effective = requested
        resource_plan = (
            ray_plan.gpu_pool
            if (resolved_hw == "gpu" and ray_plan.gpu_pool is not None)
            else ray_plan.cpu_pool
        )
    else:
        requested = worker_count if worker_count is not None else cfg.compute.workers
        effective = effective_worker_count(
            cfg,
            selected_models,
            requested,
            n_series=cfg.data.series_limit,
        )
        n_series = cfg.data.series_limit
        basis = n_series if n_series is not None else cfg.compute.max_parallelism
        resource_plan = plan_vertex_pool(
            cfg,
            selected_models,
            basis * len(selected_models),
            runtime="gke",
            gpu=(resolved_hw == "gpu"),
            gpu_type=resolved_gpu_type,
            machine_type=resolved_machine,
            worker_count=effective,
            profile=profile,
        )

    return GkeJobPlan(
        job_name=resolved_job_name,
        gke_mode=resolved_mode,
        cluster_name=resolved_cluster,
        ephemeral_cluster=ephemeral,
        namespace=resolved_ns,
        machine_type=resolved_machine,
        accelerator_type=accel_type,
        accelerator_count=accel_count,
        worker_count=effective,
        requested_workers=requested,
        hardware=resolved_hw,
        gpu_type=resolved_gpu_type,
        image_uri=image_uri,
        package_uri=package_uri,
        config_uri=config_uri,
        service_account=service_account,
        subnetwork_uri=subnetwork_uri,
        resource_plan=resource_plan,
        ray_cluster_plan=ray_plan,
    )


def _pod_resource_requests(plan: GkeJobPlan) -> dict[str, Any]:
    """Compute Kubernetes container `resources` (`requests` / `limits`) that fit inside the node's
    allocatable capacity (pure).

    GKE reserves ~100-400m CPU and ~1-2 GiB RAM per node for kubelet, system daemons, and
    kube-system DaemonSets. Requesting 60% of the node's cores (floored at 1 vCPU) and 60% of node
    memory (floored at 2 GiB) guarantees 1 pod per worker node schedules cleanly without
    `Insufficient cpu` or `Insufficient memory` deadlocks while still triggering Cluster Autoscaler
    / NAP scale-up.
    """
    cores = machine_cores(plan.machine_type) or 4
    mem_bytes = machine_memory_bytes(plan.machine_type) or (8 * 1024**3)
    req_cpu = max(1, int(cores * 0.6))
    req_mem_gi = max(2, int((mem_bytes / (1024**3)) * 0.6))

    requests: dict[str, str] = {
        "cpu": str(req_cpu),
        "memory": f"{req_mem_gi}Gi",
    }
    limits: dict[str, str] = {}
    if plan.accelerator_count > 0:
        requests["nvidia.com/gpu"] = str(plan.accelerator_count)
        limits["nvidia.com/gpu"] = str(plan.accelerator_count)

    res: dict[str, Any] = {"requests": requests}
    if limits:
        res["limits"] = limits
    return res


def _pod_env_items(
    plan: GkeJobPlan,
    *,
    settings: Settings | None = None,
    extra_env: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """Build the Kubernetes container `env` list for a GKE pod (pure)."""
    threads = (
        plan.resource_plan.slot.cores
        if plan.resource_plan is not None and plan.resource_plan.slot is not None
        else 1
    )
    env_map: dict[str, str] = {
        "SF_PROVISIONED_HARDWARE": plan.hardware,
        "SF_VM_RUNTIME": "gke",
        "JOB_COMPLETION_TOTAL": str(plan.worker_count),
        "SF_VERTEX_JOB_ID": plan.job_name,
        "PYTHONUNBUFFERED": "1",
    }
    if settings is not None:
        env_map["SF_PROJECT_ID"] = settings.project_id
        env_map["SF_DATASET_ID"] = settings.dataset_id
        if settings.registry_dataset_id_override:
            env_map["SF_REGISTRY_DATASET_ID"] = settings.registry_dataset_id_override
        env_map["SF_CONNECTION"] = settings.connection
        env_map["SF_WAREHOUSE_URI"] = settings.warehouse_uri
        env_map["SF_REGION"] = settings.region
    env_map.update(intraop_env_vars(threads))
    if extra_env:
        env_map.update(extra_env)
    return [{"name": k, "value": str(v)} for k, v in env_map.items()]


def _pod_tolerations(plan: GkeJobPlan) -> list[dict[str, str]]:
    """Standard Kubernetes tolerations for GKE GPU and scale-forecasting node pools (pure)."""
    tolerations = [
        {"key": "scale-forecasting/pool", "operator": "Exists", "effect": "NoSchedule"},
    ]
    if plan.accelerator_count > 0:
        tolerations.append({"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"})
    return tolerations


def _pod_node_selector(plan: GkeJobPlan) -> dict[str, str]:
    """Return GKE ``nodeSelector`` labels when specific accelerator hardware is requested (pure)."""
    if plan.accelerator_count > 0 and plan.accelerator_type:
        return {"cloud.google.com/gke-accelerator": plan.accelerator_type}
    return {}


def build_k8s_indexed_job_manifest(
    plan: GkeJobPlan,
    driver_args: Sequence[str] = (),
    *,
    settings: Settings | None = None,
    labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a Kubernetes ``batch/v1`` Indexed ``Job`` manifest for ``gke_mode="job"`` (pure).

    Kubernetes sets ``JOB_COMPLETION_INDEX`` in ``[0, plan.worker_count - 1]`` on each pod;
    `engines.vertex_engine.resolve_worker_topology` reads ``JOB_COMPLETION_INDEX`` and
    ``JOB_COMPLETION_TOTAL`` to shard series or models across pods without a coordinator node.
    """
    merged_labels = {
        "app": "scale-forecasting",
        "managed-by": "scale-forecasting",
        "sf-job": plan.job_name,
        "sf-mode": "job",
        **(labels or {}),
    }
    container_args = ["--package-uri", plan.package_uri, *driver_args]
    pod_spec: dict[str, Any] = {
        "restartPolicy": "Never",
        "tolerations": _pod_tolerations(plan),
        "volumes": [{"name": "dshm", "emptyDir": {"medium": "Memory"}}],
        "containers": [
            {
                "name": "worker",
                "image": plan.image_uri,
                "command": ["/opt/venv/bin/python", "-c", VERTEX_BOOTSTRAP_CODE],
                "args": container_args,
                "env": _pod_env_items(plan, settings=settings),
                "resources": _pod_resource_requests(plan),
                "volumeMounts": [{"name": "dshm", "mountPath": "/dev/shm"}],
            }
        ],
    }
    node_sel = _pod_node_selector(plan)
    if node_sel:
        pod_spec["nodeSelector"] = node_sel
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": plan.job_name,
            "namespace": plan.namespace,
            "labels": merged_labels,
        },
        "spec": {
            "completions": plan.worker_count,
            "parallelism": plan.worker_count,
            "completionMode": "Indexed",
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 3600,
            "template": {
                "metadata": {
                    "labels": merged_labels,
                },
                "spec": pod_spec,
            },
        },
    }


def build_kuberay_cluster_manifest(
    plan: GkeJobPlan,
    *,
    settings: Settings | None = None,
    labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a KubeRay ``ray.io/v1`` ``RayCluster`` custom resource manifest (pure)."""
    merged_labels = {
        "app": "scale-forecasting",
        "managed-by": "scale-forecasting",
        "sf-job": plan.job_name,
        "sf-mode": "ray",
        **(labels or {}),
    }
    worker_replicas = max(0, plan.worker_count - 1)
    node_sel = _pod_node_selector(plan)
    head_spec: dict[str, Any] = {
        "tolerations": _pod_tolerations(plan),
        "volumes": [{"name": "dshm", "emptyDir": {"medium": "Memory"}}],
        "containers": [
            {
                "name": "ray-head",
                "image": plan.image_uri,
                "env": _pod_env_items(plan, settings=settings),
                "resources": _pod_resource_requests(plan),
                "volumeMounts": [{"name": "dshm", "mountPath": "/dev/shm"}],
            }
        ],
    }
    worker_spec: dict[str, Any] = {
        "tolerations": _pod_tolerations(plan),
        "volumes": [{"name": "dshm", "emptyDir": {"medium": "Memory"}}],
        "containers": [
            {
                "name": "ray-worker",
                "image": plan.image_uri,
                "env": _pod_env_items(plan, settings=settings),
                "resources": _pod_resource_requests(plan),
                "volumeMounts": [{"name": "dshm", "mountPath": "/dev/shm"}],
            }
        ],
    }
    if node_sel:
        head_spec["nodeSelector"] = node_sel
        worker_spec["nodeSelector"] = node_sel
    return {
        "apiVersion": "ray.io/v1",
        "kind": "RayCluster",
        "metadata": {
            "name": f"{plan.job_name}-ray",
            "namespace": plan.namespace,
            "labels": merged_labels,
        },
        "spec": {
            "rayVersion": "2.59.0",
            "headGroupSpec": {
                "rayStartParams": {
                    "port": "6379",
                    "dashboard-host": "0.0.0.0",
                    "disable-usage-stats": "true",
                },
                "template": {
                    "metadata": {"labels": {**merged_labels, "sf-ray-role": "head"}},
                    "spec": head_spec,
                },
            },
            "workerGroupSpecs": [
                {
                    "groupName": "default-group",
                    "replicas": worker_replicas,
                    "minReplicas": worker_replicas,
                    "maxReplicas": max(worker_replicas, plan.worker_count),
                    "rayStartParams": {"disable-usage-stats": "true"},
                    "template": {
                        "metadata": {"labels": {**merged_labels, "sf-ray-role": "worker"}},
                        "spec": worker_spec,
                    },
                }
            ],
        },
    }


def build_k8s_ray_manifests(
    plan: GkeJobPlan,
    driver_args: Sequence[str] = (),
    *,
    settings: Settings | None = None,
    labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build Kubernetes manifests for ``gke_mode="ray"`` (pure).

    Returns a dictionary containing:
    * ``job``: The Kubernetes ``batch/v1`` ``Job`` running ``RAY_GKE_BOOTSTRAP_CODE`` ->
      ``scale_forecasting.ray_entry``.
    * ``head_service``: Headless ``v1/Service`` exposing port ``6379`` when
      ``plan.worker_count > 1``, else ``None``.
    * ``worker_deployment``: ``apps/v1/Deployment`` of ``plan.worker_count - 1`` Ray worker pods
      joining ``<job_name>-head:6379`` when ``plan.worker_count > 1``, else ``None``.
    * ``kuberay_cluster``: The KubeRay ``ray.io/v1`` ``RayCluster`` specification.
    """
    merged_labels = {
        "app": "scale-forecasting",
        "managed-by": "scale-forecasting",
        "sf-job": plan.job_name,
        "sf-mode": "ray",
        **(labels or {}),
    }
    head_svc_name = f"{plan.job_name}-head"
    worker_deploy_name = f"{plan.job_name}-workers"
    head_labels = {**merged_labels, "sf-ray-role": "head"}
    worker_labels = {**merged_labels, "sf-ray-role": "worker"}
    node_sel = _pod_node_selector(plan)

    container_args = ["--package-uri", plan.package_uri, *driver_args]
    driver_pod_spec: dict[str, Any] = {
        "restartPolicy": "Never",
        "tolerations": _pod_tolerations(plan),
        "volumes": [{"name": "dshm", "emptyDir": {"medium": "Memory"}}],
        "containers": [
            {
                "name": "ray-driver",
                "image": plan.image_uri,
                "command": ["/opt/venv/bin/python", "-c", RAY_GKE_BOOTSTRAP_CODE],
                "args": container_args,
                "env": _pod_env_items(
                    plan,
                    settings=settings,
                    extra_env={"SF_RAY_EXPECTED_NODES": str(plan.worker_count)},
                ),
                "resources": _pod_resource_requests(plan),
                "volumeMounts": [{"name": "dshm", "mountPath": "/dev/shm"}],
            }
        ],
    }
    if node_sel:
        driver_pod_spec["nodeSelector"] = node_sel
    job_manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": plan.job_name,
            "namespace": plan.namespace,
            "labels": merged_labels,
        },
        "spec": {
            "completions": 1,
            "parallelism": 1,
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 3600,
            "template": {
                "metadata": {
                    "labels": head_labels,
                },
                "spec": driver_pod_spec,
            },
        },
    }

    head_service: dict[str, Any] | None = None
    worker_deployment: dict[str, Any] | None = None
    if plan.worker_count > 1:
        head_service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": head_svc_name,
                "namespace": plan.namespace,
                "labels": merged_labels,
            },
            "spec": {
                "clusterIP": "None",
                "selector": {
                    "sf-job": plan.job_name,
                    "sf-ray-role": "head",
                },
                "ports": [
                    {"name": "gcs", "port": 6379, "targetPort": 6379},
                ],
            },
        }
        worker_pod_spec: dict[str, Any] = {
            "tolerations": _pod_tolerations(plan),
            "volumes": [{"name": "dshm", "emptyDir": {"medium": "Memory"}}],
            "containers": [
                {
                    "name": "ray-worker",
                    "image": plan.image_uri,
                    "command": [
                        "/opt/venv/bin/python",
                        "-c",
                        RAY_GKE_WORKER_BOOTSTRAP_CODE,
                    ],
                    "env": _pod_env_items(
                        plan,
                        settings=settings,
                        extra_env={
                            "SF_PACKAGE_URI": plan.package_uri,
                            "SF_RAY_HEAD_ADDRESS": f"{head_svc_name}:6379",
                        },
                    ),
                    "resources": _pod_resource_requests(plan),
                    "volumeMounts": [{"name": "dshm", "mountPath": "/dev/shm"}],
                }
            ],
        }
        if node_sel:
            worker_pod_spec["nodeSelector"] = node_sel
        worker_deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": worker_deploy_name,
                "namespace": plan.namespace,
                "labels": merged_labels,
            },
            "spec": {
                "replicas": plan.worker_count - 1,
                "selector": {
                    "matchLabels": {
                        "sf-job": plan.job_name,
                        "sf-ray-role": "worker",
                    }
                },
                "template": {
                    "metadata": {
                        "labels": worker_labels,
                    },
                    "spec": worker_pod_spec,
                },
            },
        }

    return {
        "job": job_manifest,
        "head_service": head_service,
        "worker_deployment": worker_deployment,
        "kuberay_cluster": build_kuberay_cluster_manifest(plan, settings=settings, labels=labels),
    }


def build_gke_cluster_spec(
    plan: GkeJobPlan,
    *,
    location: str,
    labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build the GKE ``projects.locations.clusters.create`` request body (pure)."""
    node_config: dict[str, Any] = {
        "machineType": plan.machine_type,
        "diskSizeGb": plan.boot_disk_gb,
        "diskType": "pd-balanced",
        "oauthScopes": ["https://www.googleapis.com/auth/cloud-platform"],
    }
    if plan.service_account:
        node_config["serviceAccount"] = plan.service_account
    if plan.accelerator_count > 0 and plan.accelerator_type:
        node_config["accelerators"] = [
            {
                "acceleratorType": plan.accelerator_type,
                "acceleratorCount": str(plan.accelerator_count),
                "gpuDriverInstallationConfig": {"gpuDriverVersion": "DEFAULT"},
            }
        ]

    cluster: dict[str, Any] = {
        "name": plan.cluster_name,
        "initialNodeCount": max(1, plan.worker_count),
        "nodeConfig": node_config,
        "autoscaling": {
            "enableNodeAutoprovisioning": False,
        },
        "nodePools": [
            {
                "name": "default-pool",
                "initialNodeCount": max(1, plan.worker_count),
                "config": node_config,
                "autoscaling": {
                    "enabled": plan.worker_count > 1,
                    "minNodeCount": 1,
                    "maxNodeCount": max(1, plan.worker_count),
                },
            }
        ],
        "ipAllocationPolicy": {"useIpAliases": True},
        "controlPlaneEndpointsConfig": {
            "dnsEndpointConfig": {"allowExternalTraffic": True},
        },
        "resourceLabels": {
            "managed-by": "scale-forecasting",
            **(labels or {}),
        },
        "deletionProtection": False,
    }
    if plan.subnetwork_uri:
        cluster["subnetwork"] = plan.subnetwork_uri
        cluster["privateClusterConfig"] = {"enablePrivateNodes": True}
        if "/networks/" in plan.subnetwork_uri:
            cluster["network"] = plan.subnetwork_uri.split("/subnetworks/")[0]
        elif "/subnetworks/" in plan.subnetwork_uri:
            subnet_name = plan.subnetwork_uri.rsplit("/subnetworks/", 1)[-1]
            if subnet_name.endswith("-compute"):
                cluster["network"] = subnet_name.removesuffix("-compute")
    return {"cluster": cluster}


# --- GKE Control Plane & Kubernetes API Server REST Clients ---------------------


def _authorized_session() -> Any:  # pragma: no cover - live GCP auth
    """Build an ADC-authenticated `AuthorizedSession` for GKE and K8s REST API calls."""
    import google.auth
    from google.auth.transport.requests import AuthorizedSession

    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    return AuthorizedSession(creds)


def get_gke_cluster(
    project_id: str,
    location: str,
    cluster_name: str,
    *,
    session: Any | None = None,
) -> dict[str, Any] | None:  # pragma: no cover - live GKE I/O
    """Fetch a GKE cluster resource by ``(project_id, location, cluster_name)`` (`None` on 404).

    When ``location`` is a region (e.g. ``us-central1``) and the direct regional lookup returns 404,
    also searches clusters across ``locations/-`` in the project so a standing zonal cluster in that
    region (e.g. ``us-central1-a``) is resolved automatically.
    """
    sess = session or _authorized_session()
    url = (
        f"{_GKE_API}/projects/{quote(project_id)}/locations/{quote(location)}"
        f"/clusters/{quote(cluster_name)}"
    )
    resp = sess.get(url, timeout=_HTTP_TIMEOUT_S)
    if resp.status_code == 200:
        return dict(resp.json())
    if resp.status_code != 404:
        raise EngineError(
            f"GKE clusters.get({cluster_name!r}, location={location!r}) failed "
            f"(HTTP {resp.status_code}): {resp.text[:500]}"
        )
    # Fallback search across project locations when location was a region name.
    all_url = f"{_GKE_API}/projects/{quote(project_id)}/locations/-/clusters"
    all_resp = sess.get(all_url, timeout=_HTTP_TIMEOUT_S)
    if all_resp.status_code == 200:
        for c in all_resp.json().get("clusters", []):
            if c.get("name") == cluster_name:
                return dict(c)
    return None


def _wait_gke_operation(
    project_id: str,
    location: str,
    op_name: str,
    *,
    session: Any | None = None,
    timeout_s: float = _GKE_OP_TIMEOUT_S,
) -> None:  # pragma: no cover - live GKE I/O
    """Block until a GKE Control Plane operation reaches ``DONE``."""
    sess = session or _authorized_session()
    op_id = op_name.rsplit("/", 1)[-1]
    url = (
        f"{_GKE_API}/projects/{quote(project_id)}/locations/{quote(location)}"
        f"/operations/{quote(op_id)}"
    )
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        resp = sess.get(url, timeout=_HTTP_TIMEOUT_S)
        if resp.status_code == 404:
            return
        if resp.status_code >= 400:
            raise EngineError(
                f"GKE operations.get({op_id}) failed (HTTP {resp.status_code}): {resp.text[:500]}"
            )
        body = resp.json()
        if body.get("status") == "DONE":
            err = body.get("error") or body.get("statusMessage")
            if body.get("error"):
                raise EngineError(f"GKE operation {op_id} in {location} failed: {err}")
            return
        time.sleep(5.0)
    raise EngineError(f"GKE operation {op_id} in {location} timed out after {timeout_s:.0f}s")


def create_gke_cluster(
    project_id: str,
    location: str,
    plan: GkeJobPlan,
    *,
    labels: dict[str, str] | None = None,
    session: Any | None = None,
) -> dict[str, Any]:  # pragma: no cover - live GKE I/O
    """Create a GKE cluster and wait until its status is ``RUNNING``."""
    sess = session or _authorized_session()
    existing = get_gke_cluster(project_id, location, plan.cluster_name, session=sess)
    if existing is not None:
        status = existing.get("status", "")
        if status == "RUNNING":
            return existing
        if status == "PROVISIONING":
            deadline = time.monotonic() + _GKE_OP_TIMEOUT_S
            while time.monotonic() < deadline:
                cur = get_gke_cluster(project_id, location, plan.cluster_name, session=sess)
                if cur is not None and cur.get("status") == "RUNNING":
                    return cur
                time.sleep(10.0)
    url = f"{_GKE_API}/projects/{quote(project_id)}/locations/{quote(location)}/clusters"
    body = build_gke_cluster_spec(plan, location=location, labels=labels)
    if (
        plan.subnetwork_uri
        and "network" not in body["cluster"]
        and plan.subnetwork_uri.startswith("https://")
    ):
        with contextlib.suppress(Exception):
            sub_resp = sess.get(plan.subnetwork_uri, timeout=_HTTP_TIMEOUT_S)
            if sub_resp.status_code == 200 and sub_resp.json().get("network"):
                body["cluster"]["network"] = sub_resp.json()["network"]
    resp = sess.post(url, json=body, timeout=_HTTP_TIMEOUT_S)
    if resp.status_code >= 400:
        raise EngineError(
            f"GKE clusters.create({plan.cluster_name!r}, location={location!r}) failed "
            f"(HTTP {resp.status_code}): {resp.text[:500]}"
        )
    op = resp.json()
    _wait_gke_operation(project_id, location, op.get("name", ""), session=sess)
    cluster = get_gke_cluster(project_id, location, plan.cluster_name, session=sess)
    if cluster is None or cluster.get("status") != "RUNNING":
        raise EngineError(
            f"GKE cluster {plan.cluster_name!r} in {location} did not reach RUNNING after create"
        )
    return cluster


def delete_gke_cluster(
    project_id: str,
    location: str,
    cluster_name: str,
    *,
    wait: bool = True,
    session: Any | None = None,
) -> None:  # pragma: no cover - live GKE I/O
    """Delete a GKE cluster (idempotent on 404). Blocks until deleted when ``wait=True``."""
    sess = session or _authorized_session()
    cluster = get_gke_cluster(project_id, location, cluster_name, session=sess)
    if cluster is None:
        return
    actual_loc = cluster.get("location") or location
    url = (
        f"{_GKE_API}/projects/{quote(project_id)}/locations/{quote(actual_loc)}"
        f"/clusters/{quote(cluster_name)}"
    )
    resp = sess.delete(url, timeout=_HTTP_TIMEOUT_S)
    if resp.status_code == 404:
        return
    if resp.status_code >= 400:
        _log.warning(
            "GKE clusters.delete(%s, location=%s) returned HTTP %d: %s",
            cluster_name,
            actual_loc,
            resp.status_code,
            resp.text[:300],
        )
        return
    if wait:
        op = resp.json()
        if op.get("name"):
            _wait_gke_operation(project_id, actual_loc, op["name"], session=sess)


class GkeK8sClient:
    """Direct REST client for a GKE cluster's Kubernetes API server using `AuthorizedSession`."""

    def __init__(self, cluster: dict[str, Any], *, session: Any | None = None) -> None:
        self.cluster_name = cluster.get("name", "")
        self.location = cluster.get("location", "")
        dns_cfg = (cluster.get("controlPlaneEndpointsConfig") or {}).get("dnsEndpointConfig") or {}
        dns_endpoint = dns_cfg.get("endpoint", "") if dns_cfg.get("allowExternalTraffic") else ""
        endpoint = dns_endpoint or cluster.get("endpoint", "")
        if not endpoint:
            raise EngineError(f"GKE cluster {self.cluster_name!r} has no control-plane endpoint")
        self.base_url = f"https://{endpoint}"
        self.session = session or _authorized_session()
        ca_b64 = (
            "" if dns_endpoint else cluster.get("masterAuth", {}).get("clusterCaCertificate", "")
        )
        self._ca_file: str | None = None
        if ca_b64:
            fd, path = tempfile.mkstemp(prefix="sf_gke_ca_", suffix=".pem")
            with os.fdopen(fd, "wb") as f:
                f.write(base64.b64decode(ca_b64))
            self._ca_file = path

    @property
    def verify(self) -> bool | str:
        return self._ca_file if self._ca_file else True

    def close(self) -> None:
        if self._ca_file and os.path.exists(self._ca_file):
            with contextlib.suppress(OSError):
                os.remove(self._ca_file)
            self._ca_file = None

    def __enter__(self) -> GkeK8sClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def create_job(self, namespace: str, manifest: dict[str, Any]) -> dict[str, Any]:
        name = manifest["metadata"]["name"]
        url = f"{self.base_url}/apis/batch/v1/namespaces/{quote(namespace)}/jobs"
        resp = self.session.post(url, json=manifest, timeout=_HTTP_TIMEOUT_S, verify=self.verify)
        if resp.status_code == 409:
            existing = self.get_job(namespace, name)
            if existing is not None:
                return existing
        if resp.status_code >= 400:
            raise EngineError(
                f"Kubernetes Job create ({name!r}) failed (HTTP {resp.status_code}): "
                f"{resp.text[:500]}"
            )
        return dict(resp.json())

    def get_job(self, namespace: str, job_name: str) -> dict[str, Any] | None:
        url = f"{self.base_url}/apis/batch/v1/namespaces/{quote(namespace)}/jobs/{quote(job_name)}"
        resp = self.session.get(url, timeout=_HTTP_TIMEOUT_S, verify=self.verify)
        if resp.status_code == 404:
            return None
        if resp.status_code >= 400:
            raise EngineError(
                f"Kubernetes Job get ({job_name!r}) failed (HTTP {resp.status_code}): "
                f"{resp.text[:500]}"
            )
        return dict(resp.json())

    def delete_job(self, namespace: str, job_name: str) -> None:
        url = (
            f"{self.base_url}/apis/batch/v1/namespaces/{quote(namespace)}"
            f"/jobs/{quote(job_name)}?propagationPolicy=Background"
        )
        resp = self.session.delete(url, timeout=_HTTP_TIMEOUT_S, verify=self.verify)
        if resp.status_code not in (200, 202, 404) and resp.status_code >= 400:
            _log.warning(
                "Kubernetes Job delete (%s) returned HTTP %d: %s",
                job_name,
                resp.status_code,
                resp.text[:300],
            )

    def create_service(self, namespace: str, manifest: dict[str, Any]) -> dict[str, Any]:
        name = manifest["metadata"]["name"]
        url = f"{self.base_url}/api/v1/namespaces/{quote(namespace)}/services"
        resp = self.session.post(url, json=manifest, timeout=_HTTP_TIMEOUT_S, verify=self.verify)
        if resp.status_code == 409:
            return manifest
        if resp.status_code >= 400:
            raise EngineError(
                f"Kubernetes Service create ({name!r}) failed (HTTP {resp.status_code}): "
                f"{resp.text[:500]}"
            )
        return dict(resp.json())

    def delete_service(self, namespace: str, name: str) -> None:
        url = f"{self.base_url}/api/v1/namespaces/{quote(namespace)}/services/{quote(name)}"
        self.session.delete(url, timeout=_HTTP_TIMEOUT_S, verify=self.verify)

    def create_deployment(self, namespace: str, manifest: dict[str, Any]) -> dict[str, Any]:
        name = manifest["metadata"]["name"]
        url = f"{self.base_url}/apis/apps/v1/namespaces/{quote(namespace)}/deployments"
        resp = self.session.post(url, json=manifest, timeout=_HTTP_TIMEOUT_S, verify=self.verify)
        if resp.status_code == 409:
            return manifest
        if resp.status_code >= 400:
            raise EngineError(
                f"Kubernetes Deployment create ({name!r}) failed (HTTP {resp.status_code}): "
                f"{resp.text[:500]}"
            )
        return dict(resp.json())

    def delete_deployment(self, namespace: str, name: str) -> None:
        url = (
            f"{self.base_url}/apis/apps/v1/namespaces/{quote(namespace)}"
            f"/deployments/{quote(name)}?propagationPolicy=Background"
        )
        self.session.delete(url, timeout=_HTTP_TIMEOUT_S, verify=self.verify)

    def list_job_pods(self, namespace: str, job_name: str) -> list[dict[str, Any]]:
        url = (
            f"{self.base_url}/api/v1/namespaces/{quote(namespace)}"
            f"/pods?labelSelector=sf-job%3D{quote(job_name)}"
        )
        resp = self.session.get(url, timeout=_HTTP_TIMEOUT_S, verify=self.verify)
        if resp.status_code != 200:
            return []
        return list(resp.json().get("items", []))

    def get_pod_logs(self, namespace: str, pod_name: str, *, tail_lines: int = 80) -> str:
        url = (
            f"{self.base_url}/api/v1/namespaces/{quote(namespace)}"
            f"/pods/{quote(pod_name)}/log?tailLines={tail_lines}"
        )
        resp = self.session.get(url, timeout=_HTTP_TIMEOUT_S, verify=self.verify)
        if resp.status_code != 200:
            return ""
        return str(resp.text)


def k8s_job_state(job_obj: dict[str, Any] | None) -> tuple[str, str]:
    """Classify a Kubernetes ``batch/v1`` ``Job`` status into ``(state, detail)`` (pure).

    Returns ``("SUCCEEDED" | "FAILED" | "RUNNING" | "NOT_FOUND", detail)``.
    """
    if job_obj is None:
        return "NOT_FOUND", "Kubernetes Job not found"
    spec = job_obj.get("spec") or {}
    status = job_obj.get("status") or {}
    completions = int(spec.get("completions") or 1)
    succeeded = int(status.get("succeeded") or 0)
    failed = int(status.get("failed") or 0)
    active = int(status.get("active") or 0)

    for cond in status.get("conditions") or []:
        ctype = cond.get("type")
        cstatus = cond.get("status")
        if ctype == "Complete" and cstatus == "True":
            return "SUCCEEDED", f"succeeded={succeeded}/{completions}"
        if ctype == "Failed" and cstatus == "True":
            reason = cond.get("reason") or "JobFailed"
            msg = cond.get("message") or ""
            return "FAILED", f"{reason}: {msg}".strip(": ")

    if succeeded >= completions:
        return "SUCCEEDED", f"succeeded={succeeded}/{completions}"
    if failed > 0 and active == 0:
        return "FAILED", f"failed={failed} active=0"
    return "RUNNING", f"active={active} succeeded={succeeded}/{completions}"


def resolve_gke_candidates(
    cfg: RunConfig,
    *,
    settings: Settings,
    infra: BatchInfra,
) -> list[Candidate]:
    """Explicit-zone candidates for ephemeral GKE cluster creation (pure)."""
    if cfg.compute.ray_regions:
        out: list[Candidate] = []
        for reg in cfg.compute.ray_regions:
            # "" is the plan's "default network" value (`GkeJobPlan.subnetwork_uri`) for a
            # cross-region candidate, whose subnet the deployment's cannot serve.
            sub = infra.subnetwork_uri if reg == settings.region else ""
            zones = US_ZONES.get(reg, [f"{reg}-a"])
            for z in zones:
                out.append(Candidate(region=reg, zone=z, subnetwork_uri=sub))
        return out
    raw = resolve_candidates(settings=settings, infra=infra)
    explicit = [c for c in raw if c.zone is not None]
    if explicit:
        return explicit
    zones = US_ZONES.get(settings.region, [f"{settings.region}-a"])
    return [
        Candidate(region=settings.region, zone=z, subnetwork_uri=infra.subnetwork_uri)
        for z in zones
    ]


def submit_gke(
    cfg: RunConfig,
    *,
    settings: Settings | None = None,
    infra: BatchInfra | None = None,
    wait: bool = True,
    models: Sequence[str] | None = None,
    job_id: str | None = None,
    gke_mode: str | None = None,
    cluster_name: str | None = None,
    cluster_location: str | None = None,
    keep_cluster: bool = False,
    manage_header: bool = True,
    hardware: str | None = None,
    gpu_type: str | None = None,
    machine_type: str | None = None,
    worker_count: int | None = None,
    accelerator_count: int | None = None,
    wait_timeout_seconds: int | None = None,
) -> tuple[str, str, ProbeHandle]:
    """Stage artifacts, resolve/create GKE cluster, dispatch K8s Job, wait, and clean up.

    Returns ``(run_id, job_name, ProbeHandle)``.
    """
    from .job_outcome import cells_written

    settings = settings or Settings.resolve()
    infra = infra or BatchInfra.resolve()
    if not infra.container_image:
        raise ConfigError(
            "GKE runtime requires SF_CONTAINER_IMAGE to be set (see docker/Dockerfile)."
        )
    run_id = make_run_id(cfg)

    python_models, _ = split_by_runtime(cfg)
    if models is not None:
        allow = set(models)
        selected_models = [m for m in python_models if m in allow]
    else:
        selected_models = python_models
    if not selected_models:
        raise ConfigError("submit_gke called with no Python-runtime models to execute")

    package_uri, _ = staging.stage_code(infra.code_bucket)
    config_uri = staging.stage_config(cfg, run_id, infra.code_bucket)
    profile = profile_for_run(cfg, settings=settings)

    plan = plan_gke_job(
        cfg,
        selected_models,
        run_id=run_id,
        job_id=job_id,
        gke_mode=gke_mode,
        cluster_name=cluster_name,
        hardware=hardware,
        gpu_type=gpu_type,
        machine_type=machine_type,
        worker_count=worker_count,
        accelerator_count=accelerator_count,
        image_uri=infra.container_image,
        package_uri=package_uri,
        config_uri=config_uri,
        service_account=infra.compute_sa,
        subnetwork_uri=infra.subnetwork_uri,
        profile=profile,
        infra=infra,
    )

    # Record sizing telemetry on the run header (best-effort).
    if plan.resource_plan is not None:
        family_key = "+".join(pool_families(selected_models)) or "cpu"
        sizing = sizing_telemetry(
            plan.resource_plan,
            profile=profile,
            family=family_key,
        )
        with contextlib.suppress(Exception):
            merge_header_telemetry(
                run_id,
                {
                    sizing_telemetry_path(sizing): sizing,
                    f"gke_job.{family_key}": plan.to_dict(),
                },
                settings=settings,
            )

    driver_args = build_driver_args(
        config_uri,
        settings,
        models=list(models) if models is not None else None,
        manage_header=manage_header,
        provisioned_hardware=plan.hardware,
    )
    labels = {
        "sf-run-id": re.sub(r"[^a-z0-9_-]", "-", run_id.lower())[:63].strip("-"),
    }

    session = _authorized_session()
    created_ephemeral_cluster = False
    landed_location = cluster_location or settings.region
    cluster_obj: dict[str, Any] | None = None

    if not plan.ephemeral_cluster:
        cluster_obj = get_gke_cluster(
            settings.project_id,
            landed_location,
            plan.cluster_name,
            session=session,
        )
        if cluster_obj is None:
            raise EngineError(
                f"Standing GKE cluster {plan.cluster_name!r} not found in project "
                f"{settings.project_id!r} (checked location {landed_location!r} and project-wide)"
            )
        landed_location = cluster_obj.get("location") or landed_location
    else:
        candidates = resolve_gke_candidates(cfg, settings=settings, infra=infra)
        regions = list(dict.fromkeys(c.region for c in candidates))
        policy = cfg.compute.capacity.policy_for("gke")
        ledger = capacity.CapacityLedger(service="gke")

        preflights: dict[str, quota.QuotaPreflight] = {}
        if cfg.compute.capacity.preflight:
            preflights = quota.preflight_gke(plan, regions, settings.project_id)
            for region in regions:
                pf = preflights.get(region)
                if pf is not None:
                    ledger.preflight.append(pf.to_dict())
                    for line in pf.render():
                        _log.info("%s", line)

        usable_candidates = [
            c for c in candidates if not (preflights.get(c.region) and preflights[c.region].blocked)
        ]
        if not usable_candidates:
            usable_candidates = candidates

        def _try_create_cluster(cand: Candidate) -> tuple[dict[str, Any], str]:
            cand_plan = (
                plan
                if cand.subnetwork_uri == plan.subnetwork_uri
                else replace(plan, subnetwork_uri=cand.subnetwork_uri or "")
            )
            c = create_gke_cluster(
                settings.project_id,
                cand.zone or cand.region,
                cand_plan,
                labels=labels,
                session=session,
            )
            return c, (c.get("location") or cand.zone or cand.region)

        try:
            cluster_obj, landed_location = capacity.walk(
                usable_candidates,
                _try_create_cluster,
                policy=policy,
                ledger=ledger,
                on_state=capacity.current_publisher(),
            )
        finally:
            if job_id is not None and (ledger.attempts or ledger.preflight):
                with contextlib.suppress(Exception):
                    update_job(
                        job_id,
                        settings=settings,
                        merge_telemetry={"capacity": ledger.to_json()},
                    )
        created_ephemeral_cluster = True

    landed_region = (
        landed_location.rsplit("-", 1)[0] if landed_location.count("-") >= 2 else landed_location
    )
    resource_name = (
        f"projects/{settings.project_id}/locations/{landed_location}"
        f"/clusters/{plan.cluster_name}/namespaces/{plan.namespace}/jobs/{plan.job_name}"
    )
    handle = ProbeHandle(
        "gke",
        native_id=plan.job_name,
        region=landed_region,
        resource_name=resource_name,
    )

    if job_id:
        with contextlib.suppress(Exception):
            update_job(
                job_id,
                settings=settings,
                merge_telemetry={
                    "probe_handle": handle.to_blob(),
                    "gke_job": plan.to_dict(),
                },
            )

    _log.info(
        "gke submit: cluster=%s location=%s ephemeral=%s mode=%s job=%s machine=%s workers=%d",
        plan.cluster_name,
        landed_location,
        plan.ephemeral_cluster,
        plan.gke_mode,
        plan.job_name,
        plan.machine_type,
        plan.worker_count,
    )

    head_svc_name: str | None = None
    worker_deploy_name: str | None = None

    try:
        assert cluster_obj is not None
        with GkeK8sClient(cluster_obj, session=session) as k8s:
            if plan.gke_mode == "ray":
                ray_manifests = build_k8s_ray_manifests(
                    plan,
                    driver_args,
                    settings=settings,
                    labels=labels,
                )
                if ray_manifests["head_service"] is not None:
                    head_svc_name = ray_manifests["head_service"]["metadata"]["name"]
                    k8s.create_service(plan.namespace, ray_manifests["head_service"])
                if ray_manifests["worker_deployment"] is not None:
                    worker_deploy_name = ray_manifests["worker_deployment"]["metadata"]["name"]
                    k8s.create_deployment(plan.namespace, ray_manifests["worker_deployment"])
                k8s.create_job(plan.namespace, ray_manifests["job"])
            else:
                job_manifest = build_k8s_indexed_job_manifest(
                    plan,
                    driver_args,
                    settings=settings,
                    labels=labels,
                )
                k8s.create_job(plan.namespace, job_manifest)

            if not wait:
                return run_id, plan.job_name, handle

            timeout_s = (
                wait_timeout_seconds
                if wait_timeout_seconds is not None
                else infra.batch_job_wait_seconds
            )
            grace_s = int(infra.stall_grace_seconds)
            deadline = time.monotonic() + timeout_s
            since = launch_window_start()
            started_wait = time.monotonic()
            last_watchdog = started_wait
            watching = grace_s > 0

            while time.monotonic() < deadline:
                cur_job = k8s.get_job(plan.namespace, plan.job_name)
                state, detail = k8s_job_state(cur_job)
                if state == "SUCCEEDED":
                    _log.info(
                        "gke job %s SUCCEEDED on cluster %s (%s)",
                        plan.job_name,
                        plan.cluster_name,
                        detail,
                    )
                    return run_id, plan.job_name, handle
                if state == "FAILED":
                    pod_logs: list[str] = []
                    for pod in k8s.list_job_pods(plan.namespace, plan.job_name):
                        pname = pod.get("metadata", {}).get("name", "")
                        if pname:
                            log_tail = k8s.get_pod_logs(plan.namespace, pname, tail_lines=60)
                            if log_tail:
                                pod_logs.append(f"--- pod {pname} ---\n{log_tail[-1500:]}")
                    tail_msg = ("\n" + "\n".join(pod_logs)) if pod_logs else ""
                    raise EngineError(
                        f"GKE Job {plan.job_name!r} on cluster {plan.cluster_name!r} FAILED "
                        f"({detail}){tail_msg}"
                    )

                now = time.monotonic()
                if (
                    watching
                    and (now - started_wait) >= grace_s
                    and (now - last_watchdog) >= _WATCHDOG_CHECK_SECONDS
                ):
                    last_watchdog = now
                    elapsed = now - started_wait
                    cells = cells_written(run_id, since=since)
                    if cells:
                        watching = False
                    elif is_stalled(elapsed_s=elapsed, grace_s=grace_s, cells=cells):
                        k8s.delete_job(plan.namespace, plan.job_name)
                        raise EngineError(
                            f"GKE Job {plan.job_name!r} stalled: zero cells landed after "
                            f"{elapsed:.0f}s (stall_grace_seconds={grace_s})"
                        )
                time.sleep(_POLL_INTERVAL_SECONDS)

            k8s.delete_job(plan.namespace, plan.job_name)
            raise EngineError(
                f"GKE Job {plan.job_name!r} on cluster {plan.cluster_name!r} timed out after "
                f"{timeout_s}s"
            )
    finally:
        if wait and cluster_obj is not None:
            with (
                contextlib.suppress(Exception),
                GkeK8sClient(cluster_obj, session=session) as k8s_clean,
            ):
                k8s_clean.delete_job(plan.namespace, plan.job_name)
                if worker_deploy_name:
                    k8s_clean.delete_deployment(plan.namespace, worker_deploy_name)
                if head_svc_name:
                    k8s_clean.delete_service(plan.namespace, head_svc_name)
            if created_ephemeral_cluster and not keep_cluster:
                _log.info(
                    "gke teardown: deleting ephemeral cluster %s in %s",
                    plan.cluster_name,
                    landed_location,
                )
                delete_gke_cluster(
                    settings.project_id,
                    landed_location,
                    plan.cluster_name,
                    wait=True,
                    session=session,
                )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scale_forecasting.gke_submit",
        description="Submit a forecast run (or family slice) to Google Kubernetes Engine (GKE).",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", help="Path to local RunConfig JSON file.")
    src.add_argument("--config-uri", help="gs:// URI of a staged RunConfig JSON.")
    p.add_argument("--models", nargs="+", help="Optional subset of models to execute.")
    p.add_argument("--job-id", help="Explicit Kubernetes Job name.")
    p.add_argument(
        "--gke-mode",
        choices=["job", "ray"],
        help="GKE execution mode ('job' for K8s Indexed Job, 'ray' for Ray on GKE).",
    )
    p.add_argument("--cluster-name", help="Target an existing GKE cluster by name.")
    p.add_argument("--cluster-location", help="GKE cluster region or zone.")
    p.add_argument(
        "--keep-cluster",
        action="store_true",
        help="Do not tear down an ephemeral GKE cluster on exit.",
    )
    p.add_argument("--hardware", choices=["cpu", "gpu"], help="Hardware override (cpu or gpu).")
    p.add_argument("--gpu-type", help="GPU type override (T4, L4, A100, A100_80GB).")
    p.add_argument("--machine-type", help="GCE machine type override.")
    p.add_argument("--workers", type=int, help="Number of worker pods/nodes.")
    p.add_argument("--accelerator-count", type=int, help="GPUs per worker pod/node.")
    p.add_argument(
        "--no-wait",
        action="store_true",
        help="Return immediately after creating the Kubernetes Job.",
    )
    p.add_argument(
        "--no-manage-header",
        action="store_true",
        help="Contributor mode: do not manage the shared run_registry header.",
    )
    p.add_argument("--wait-timeout", type=int, help="Max seconds to wait for completion.")
    return p


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - CLI entrypoint
    from .config import load_config, load_config_uri

    configure_cli_logging()
    require_extra("gcp", purpose="Submitting to GKE")
    args = _build_parser().parse_args(argv)
    cfg = load_config_uri(args.config_uri) if args.config_uri else load_config(args.config)
    submit_gke(
        cfg,
        wait=not args.no_wait,
        models=args.models,
        job_id=args.job_id,
        gke_mode=args.gke_mode,
        cluster_name=args.cluster_name,
        cluster_location=args.cluster_location,
        keep_cluster=args.keep_cluster,
        manage_header=not args.no_manage_header,
        hardware=args.hardware,
        gpu_type=args.gpu_type,
        machine_type=args.machine_type,
        worker_count=args.workers,
        accelerator_count=args.accelerator_count,
        wait_timeout_seconds=args.wait_timeout,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
