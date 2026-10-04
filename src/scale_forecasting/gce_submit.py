"""Submit a run (or family slice) to a single Google Compute Engine VM (``runtime="gce"``).

Why a direct Compute Engine single-VM runtime beside Vertex ``CustomJob``, Spark, and Ray:
1. **Direct GCE Compute Quota & Zonal Placement**: Runs directly on standard Compute Engine VM
   quotas (`CPUS`, `N2_CPUS`, `NVIDIA_T4_GPUS`, `NVIDIA_L4_GPUS`) across candidate zones via
   `compute_fallback.resolve_candidates`, without needing Vertex AI `CustomJob` training quota or a
   multi-node Dataproc/Ray cluster.
2. **Identical Container + Dynamic GCS Code Delivery**: Boots Google's Container-Optimized OS
   (`projects/cos-cloud/global/images/family/cos-stable`), pulls `SF_CONTAINER_IMAGE` from Artifact
   Registry, downloads `gs://<code_bucket>/runs/scale_forecasting-<code_hash>.zip` via
   `VERTEX_BOOTSTRAP_CODE`, and executes `scale_forecasting.vertex_entry` (`vertex_engine.run`).
3. **Triple-Redundant Zero-Orphan VM Lifecycle**:
   - **Layer 1 — GCE Hypervisor Hard TTL**: Every VM is created with
     `scheduling.maxRunDuration` + `scheduling.instanceTerminationAction = "DELETE"` and
     `automaticRestart = False`, so the Compute Engine hypervisor unconditionally deletes the VM
     when its TTL expires even if both the client process and the guest OS crash.
   - **Layer 2 — Guest OS `trap cleanup EXIT` Self-Deletion**: The Container-Optimized OS startup
     script registers `trap cleanup EXIT` on line 1. As soon as the container exits (whether exit 0
     or non-zero), `cleanup()` writes a JSON status marker (`SUCCEEDED` or `FAILED` + container tail
     logs) to `gs://<code_bucket>/runs/gce-status/<instance_name>.json` and calls the Compute Engine
     REST API (`DELETE .../instances/<instance_name>`) using the VM's metadata OAuth token.
   - **Layer 3 — Client-Side `try ... finally` Teardown & `GceProbe.cancel()`**: `submit_gce`
     deletes the instance in a `finally` block (idempotent on `404 Not Found` if the guest already
     self-deleted) and cancels + deletes on stall-watchdog or wait timeout.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shlex
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from . import capacity, quota, staging
from .batch_infra import BatchInfra
from .commands import build_driver_args
from .compute_fallback import US_ZONES, Candidate, resolve_candidates
from .config import RunConfig, resolve_vm_machine_type
from .engines.ray_io import pool_families, resolve_job_gpu
from .engines.vertex_engine import plan_vertex_pool
from .errors import ConfigError, EngineError, get_logger
from .job_outcome import launch_window_start
from .job_wait import is_stalled
from .probes.vocabulary import ProbeHandle
from .profiling.source import profile_for_run
from .registry.header import merge_header_telemetry, sizing_telemetry_path
from .registry.ids import gce_instance_id, make_run_id
from .registry.jobs import update_job
from .resources.audit import sizing_telemetry
from .resources.catalog import intraop_env_vars
from .resources.fleet import RuntimeResourcePlan
from .router import split_by_runtime
from .settings import Settings
from .vertex_submit import VERTEX_BOOTSTRAP_CODE

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .profiling.cost import ComputeProfile

_log = get_logger(__name__)

_COMPUTE_API = "https://compute.googleapis.com/compute/v1"
_COS_IMAGE = "projects/cos-cloud/global/images/family/cos-stable"
_DEFAULT_BOOT_DISK_GB = 100
_POLL_INTERVAL_SECONDS = 10.0
_WATCHDOG_CHECK_SECONDS = 300.0
_HTTP_TIMEOUT_S = 30.0

_GCE_ACCELERATOR_TYPES = {
    "T4": "nvidia-tesla-t4",
    "L4": "nvidia-l4",
    "A100": "nvidia-tesla-a100",
    "A100_80GB": "nvidia-a100-80gb",
}

_ACTIVE_INSTANCE_STATES = frozenset({"PROVISIONING", "STAGING", "RUNNING"})
_TERMINATED_INSTANCE_STATES = frozenset({"STOPPING", "SUSPENDING", "SUSPENDED", "TERMINATED"})


def status_object_path(instance_name: str) -> str:
    """GCS object path holding the terminal status marker for a GCE VM (pure)."""
    return f"runs/gce-status/{instance_name}.json"


@dataclass(frozen=True)
class GceInstancePlan:
    """Resolved execution plan for a single-VM Compute Engine run (pure)."""

    instance_name: str
    machine_type: str
    accelerator_type: str
    accelerator_count: int
    hardware: str
    gpu_type: str | None
    image_uri: str = ""
    package_uri: str = ""
    config_uri: str = ""
    service_account: str = ""
    subnetwork_uri: str = ""
    status_bucket: str = ""
    status_object: str = ""
    max_run_duration_seconds: int = 7200
    boot_disk_gb: int = _DEFAULT_BOOT_DISK_GB
    resource_plan: RuntimeResourcePlan | None = None

    @property
    def status_uri(self) -> str:
        """Full `gs://` URI of the status marker object, or empty string when unset."""
        if self.status_bucket and self.status_object:
            return f"gs://{self.status_bucket}/{self.status_object}"
        return ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe summary for `run_jobs.job_telemetry.$.gce_instance`."""
        return {
            "instance_name": self.instance_name,
            "machine_type": self.machine_type,
            "accelerator_type": self.accelerator_type,
            "accelerator_count": self.accelerator_count,
            "hardware": self.hardware,
            "gpu_type": self.gpu_type,
            "image_uri": self.image_uri,
            "package_uri": self.package_uri,
            "config_uri": self.config_uri,
            "subnetwork_uri": self.subnetwork_uri,
            "status_uri": self.status_uri,
            "max_run_duration_seconds": self.max_run_duration_seconds,
            "boot_disk_gb": self.boot_disk_gb,
            "resource_plan": (
                self.resource_plan.to_dict() if self.resource_plan is not None else None
            ),
        }


def plan_gce_job(
    cfg: RunConfig,
    models: Sequence[str] | None = None,
    *,
    run_id: str | None = None,
    job_id: str | None = None,
    instance_name: str | None = None,
    hardware: str | None = None,
    gpu_type: str | None = None,
    machine_type: str | None = None,
    accelerator_count: int | None = None,
    image_uri: str = "",
    package_uri: str = "",
    config_uri: str = "",
    service_account: str = "",
    subnetwork_uri: str = "",
    status_bucket: str = "",
    max_run_duration_seconds: int = 7200,
    ttl_seconds: int | None = None,
    profile: ComputeProfile | None = None,
    settings: Settings | None = None,
    infra: BatchInfra | None = None,
) -> GceInstancePlan:
    """Resolve a single-VM `GceInstancePlan` from `cfg` and per-family overrides (pure)."""
    rid = run_id or make_run_id(cfg)
    python_models, _ = split_by_runtime(cfg)
    selected_models = list(models) if models is not None else python_models

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
        accel_type = _GCE_ACCELERATOR_TYPES.get(
            resolved_gpu_type, f"nvidia-{resolved_gpu_type.lower()}"
        )
        accel_count = eff_accel
    else:
        accel_type = ""
        accel_count = 0

    raw_id = instance_name or job_id
    resolved_instance = gce_instance_id(raw_id) if raw_id else gce_instance_id(f"sf-{rid}-gce")

    resolved_bucket = status_bucket
    if not resolved_bucket and infra is not None:
        resolved_bucket = infra.code_bucket
    if not resolved_bucket and package_uri.startswith("gs://"):
        resolved_bucket = package_uri.removeprefix("gs://").split("/", 1)[0]
    if not resolved_bucket and settings is not None and settings.warehouse_uri.startswith("gs://"):
        resolved_bucket = settings.warehouse_uri.removeprefix("gs://").split("/", 1)[0]

    effective_ttl = ttl_seconds if ttl_seconds is not None else max_run_duration_seconds

    n_series = cfg.data.series_limit
    basis = n_series if n_series is not None else cfg.compute.max_parallelism
    resource_plan = plan_vertex_pool(
        cfg,
        selected_models,
        basis * len(selected_models),
        runtime="gce",
        gpu=(resolved_hw == "gpu"),
        gpu_type=resolved_gpu_type,
        machine_type=resolved_machine,
        worker_count=1,
        profile=profile,
    )

    return GceInstancePlan(
        instance_name=resolved_instance,
        machine_type=resolved_machine,
        accelerator_type=accel_type,
        accelerator_count=accel_count,
        hardware=resolved_hw,
        gpu_type=resolved_gpu_type,
        image_uri=image_uri,
        package_uri=package_uri,
        config_uri=config_uri,
        service_account=service_account,
        subnetwork_uri=subnetwork_uri,
        status_bucket=resolved_bucket,
        status_object=status_object_path(resolved_instance),
        max_run_duration_seconds=max(600, int(effective_ttl)),
        resource_plan=resource_plan,
    )


def build_gce_startup_script(
    plan: GceInstancePlan,
    driver_args: Sequence[str] = (),
    *,
    project_id: str = "project",
    zone: str = "us-central1-a",
) -> str:
    """Render the Container-Optimized OS startup script with `trap cleanup EXIT` self-delete (pure).

    Uses only POSIX shell + `curl` + `sed` + `docker` so it has zero dependency on a host Python
    installation on Container-Optimized OS (`cos-stable`).
    """
    registry_host = (
        plan.image_uri.split("/", 1)[0] if "/" in plan.image_uri else "us-docker.pkg.dev"
    )
    threads = plan.resource_plan.slot.cores if plan.resource_plan is not None else 1
    env_flags = [
        "-e",
        "SF_VM_RUNTIME=gce",
        "-e",
        "SF_WORKER_RANK=0",
        "-e",
        "SF_WORKER_COUNT=1",
    ]
    for k, v in sorted(intraop_env_vars(threads, include_omp=True).items()):
        env_flags.extend(["-e", f"{k}={v}"])

    gpu_setup = ""
    gpu_docker_flags: list[str] = []
    if plan.accelerator_count > 0:
        gpu_setup = (
            "cos-extensions install gpu\n"
            "mount --bind /var/lib/nvidia /var/lib/nvidia\n"
            "mount -o remount,exec /var/lib/nvidia\n"
        )
        gpu_docker_flags = [
            "--volume",
            "/var/lib/nvidia/lib64:/usr/local/nvidia/lib64",
            "--volume",
            "/var/lib/nvidia/bin:/usr/local/nvidia/bin",
            "--device",
            "/dev/nvidia0:/dev/nvidia0",
            "--device",
            "/dev/nvidiactl:/dev/nvidiactl",
            "--device",
            "/dev/nvidia-uvm:/dev/nvidia-uvm",
            "-e",
            "LD_LIBRARY_PATH=/usr/local/nvidia/lib64",
        ]

    cmd_tokens = [
        "docker",
        "run",
        "--rm",
        *env_flags,
        *gpu_docker_flags,
        plan.image_uri,
        "/opt/venv/bin/python",
        "-c",
        VERTEX_BOOTSTRAP_CODE,
        "--package-uri",
        plan.package_uri,
        *driver_args,
    ]
    docker_cmd = " ".join(shlex.quote(tok) for tok in cmd_tokens)
    encoded_obj = quote(plan.status_object, safe="")
    registry_url_q = shlex.quote("https://" + registry_host)

    return (
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "export HOME=/tmp\n"
        "export DOCKER_CONFIG=/tmp/.docker\n"
        f"PROJECT_ID={shlex.quote(project_id)}\n"
        f"ZONE={shlex.quote(zone)}\n"
        f"INSTANCE_NAME={shlex.quote(plan.instance_name)}\n"
        f"STATUS_BUCKET={shlex.quote(plan.status_bucket)}\n"
        f"STATUS_OBJECT_ENC={shlex.quote(encoded_obj)}\n"
        "LOG_FILE=/tmp/sf_container.log\n"
        ': > "$LOG_FILE"\n'
        "\n"
        "get_token() {\n"
        "  curl -sf -H 'Metadata-Flavor: Google' \\\n"
        "    'http://metadata.google.internal/computeMetadata/v1/"
        "instance/service-accounts/default/token' \\\n"
        '    | sed -n \'s/.*"access_token"[[:space:]]*:[[:space:]]*"\\([^"]*\\)".*/\\1/p\'\n'
        "}\n"
        "\n"
        "upload_status() {\n"
        '  local st="$1"\n'
        '  local rc="$2"\n'
        '  local detail="$3"\n'
        '  if [ -n "$STATUS_BUCKET" ]; then\n'
        "    local tok\n"
        "    tok=$(get_token || true)\n"
        '    printf \'{"status":"%s","exit_code":%s,"instance":"%s",'
        '"zone":"%s","detail":"%s"}\\n\' \\\n'
        '      "$st" "$rc" "$INSTANCE_NAME" "$ZONE" "$detail" \\\n'
        '      | curl -sf -X POST -H "Authorization: Bearer ${tok}" \\\n'
        "        -H 'Content-Type: application/json' --data-binary @- \\\n"
        '        "https://storage.googleapis.com/upload/storage/v1/b/'
        '${STATUS_BUCKET}/o?uploadType=media&name=${STATUS_OBJECT_ENC}" >/dev/null || true\n'
        "  fi\n"
        "}\n"
        "\n"
        "cleanup() {\n"
        "  local rc=$?\n"
        "  trap - EXIT\n"
        "  local st='FAILED'\n"
        '  if [ "$rc" -eq 0 ]; then\n'
        "    st='SUCCEEDED'\n"
        "  fi\n"
        "  local tail_msg=''\n"
        '  if [ -f "$LOG_FILE" ]; then\n'
        '    tail_msg=$(tail -n 25 "$LOG_FILE" | '
        "tr '\\n\\r\"\\\\' '    ' | cut -c 1-800 || true)\n"
        "  fi\n"
        '  upload_status "$st" "$rc" "$tail_msg"\n'
        "  local tok\n"
        "  tok=$(get_token || true)\n"
        '  curl -sf -X DELETE -H "Authorization: Bearer ${tok}" \\\n'
        '    "https://compute.googleapis.com/compute/v1/projects/'
        '${PROJECT_ID}/zones/${ZONE}/instances/${INSTANCE_NAME}" >/dev/null || true\n'
        "  shutdown -h now >/dev/null 2>&1 || poweroff >/dev/null 2>&1 || true\n"
        "}\n"
        "trap cleanup EXIT\n"
        "\n"
        "upload_status 'RUNNING' 'null' ''\n"
        f"{gpu_setup}"
        "TOKEN=$(get_token)\n"
        'echo "$TOKEN" | docker login -u oauth2accesstoken '
        f'--password-stdin {registry_url_q} >>"$LOG_FILE" 2>&1\n'
        f'docker pull {shlex.quote(plan.image_uri)} >>"$LOG_FILE" 2>&1\n'
        f'{docker_cmd} >>"$LOG_FILE" 2>&1\n'
    )


def build_gce_instance_spec_dict(
    plan: GceInstancePlan,
    driver_args: Sequence[str] = (),
    *,
    project_id: str = "project",
    zone: str = "us-central1-a",
    labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build the Compute Engine ``instances.insert`` REST request body (pure).

    Enforces triple-redundant zero-orphan lifecycle guarantees:
    1. ``scheduling.maxRunDuration`` + ``scheduling.instanceTerminationAction = "DELETE"`` +
       ``scheduling.automaticRestart = False`` (hypervisor-level hard TTL deletion).
    2. ``metadata.items[startup-script]`` with ``trap cleanup EXIT`` self-deletion via GCE REST API.
    3. ``disks[0].autoDelete = True`` so the boot persistent disk never outlives the VM.
    """
    startup_script = build_gce_startup_script(plan, driver_args, project_id=project_id, zone=zone)
    spec: dict[str, Any] = {
        "name": plan.instance_name,
        "machineType": f"zones/{zone}/machineTypes/{plan.machine_type}",
        "scheduling": {
            "automaticRestart": False,
            "onHostMaintenance": "TERMINATE",
            "maxRunDuration": {"seconds": str(plan.max_run_duration_seconds)},
            "instanceTerminationAction": "DELETE",
        },
        "disks": [
            {
                "boot": True,
                "autoDelete": True,
                "initializeParams": {
                    "sourceImage": _COS_IMAGE,
                    "diskSizeGb": str(plan.boot_disk_gb),
                    "diskType": f"zones/{zone}/diskTypes/pd-balanced",
                },
            }
        ],
        "networkInterfaces": [
            {
                "subnetwork": plan.subnetwork_uri,
            }
        ],
        "metadata": {
            "items": [
                {"key": "startup-script", "value": startup_script},
                {"key": "google-logging-enabled", "value": "true"},
            ]
        },
    }
    if plan.service_account:
        spec["serviceAccounts"] = [
            {
                "email": plan.service_account,
                "scopes": ["https://www.googleapis.com/auth/cloud-platform"],
            }
        ]
    if plan.accelerator_count > 0 and plan.accelerator_type:
        spec["guestAccelerators"] = [
            {
                "acceleratorType": f"zones/{zone}/acceleratorTypes/{plan.accelerator_type}",
                "acceleratorCount": plan.accelerator_count,
            }
        ]
    if labels:
        spec["labels"] = dict(labels)
    return spec


def _authorized_session() -> Any:  # pragma: no cover - live GCP I/O
    import google.auth
    from google.auth.transport.requests import AuthorizedSession

    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    return AuthorizedSession(creds)


def _instance_url(project_id: str, zone: str, instance_name: str) -> str:
    return f"{_COMPUTE_API}/projects/{project_id}/zones/{zone}/instances/{instance_name}"


def _write_status_marker(
    bucket_name: str,
    object_name: str,
    payload: dict[str, Any],
) -> None:  # pragma: no cover - live GCS I/O
    if not bucket_name or not object_name:
        return
    from google.cloud import storage

    blob = storage.Client().bucket(bucket_name).blob(object_name)
    blob.upload_from_string(json.dumps(payload), content_type="application/json")


def read_status_marker(
    bucket_name: str,
    object_or_instance_name: str,
    *,
    project_id: str | None = None,
) -> dict[str, Any] | None:  # pragma: no cover - live GCS I/O
    """Read the GCS status marker written by `submit_gce` / the guest startup script, or `None`."""
    if not bucket_name or not object_or_instance_name:
        return None
    from google.api_core.exceptions import NotFound
    from google.cloud import storage

    object_name = (
        object_or_instance_name
        if object_or_instance_name.startswith("runs/")
        else status_object_path(object_or_instance_name)
    )
    client = storage.Client(project=project_id) if project_id else storage.Client()
    blob = client.bucket(bucket_name).blob(object_name)
    try:
        raw = blob.download_as_text()
    except NotFound:
        return None
    except Exception as exc:  # noqa: BLE001
        _log.debug("could not read GCE status marker gs://%s/%s: %r", bucket_name, object_name, exc)
        return None
    try:
        doc = json.loads(raw)
        return doc if isinstance(doc, dict) else None
    except ValueError:
        return None


def get_instance(
    project_id: str,
    zone: str,
    instance_name: str,
    *,
    session: Any = None,
    timeout: float = _HTTP_TIMEOUT_S,
) -> dict[str, Any] | None:  # pragma: no cover - live GCE I/O
    """Return the Compute Engine instance resource dict, or ``None`` on 404."""
    sess = session or _authorized_session()
    resp = sess.get(_instance_url(project_id, zone, instance_name), timeout=timeout)
    if resp.status_code == 404:
        return None
    if resp.status_code >= 400:
        raise EngineError(
            f"GCE instances.get({instance_name} in {zone}) failed ({resp.status_code}): {resp.text}"
        )
    data = resp.json()
    return data if isinstance(data, dict) else None


def delete_instance(
    project_id: str,
    zone: str,
    instance_name: str,
    *,
    session: Any = None,
    timeout: float = _HTTP_TIMEOUT_S,
) -> bool:  # pragma: no cover - live GCE I/O
    """Best-effort delete of a GCE instance; returns True if a delete was accepted, False on 404."""
    try:
        sess = session or _authorized_session()
        resp = sess.delete(_instance_url(project_id, zone, instance_name), timeout=timeout)
        if resp.status_code == 404:
            return False
        if resp.status_code >= 400:
            _log.warning(
                "GCE instances.delete(%s in %s) returned %d: %s",
                instance_name,
                zone,
                resp.status_code,
                resp.text[:300],
            )
            return False
        return True
    except Exception as exc:  # noqa: BLE001 - best-effort teardown
        _log.warning("GCE instances.delete(%s in %s) failed: %r", instance_name, zone, exc)
        return False


def _insert_and_wait_running(
    session: Any,
    project_id: str,
    zone: str,
    spec_dict: dict[str, Any],
    *,
    poll_interval_s: float = 5.0,
    provision_timeout_s: float = 300.0,
) -> dict[str, Any]:  # pragma: no cover - live GCE I/O
    """Insert a GCE instance and wait until its zonal operation completes and VM reaches RUNNING."""
    url = f"{_COMPUTE_API}/projects/{project_id}/zones/{zone}/instances"
    resp = session.post(url, json=spec_dict, timeout=_HTTP_TIMEOUT_S)
    if resp.status_code >= 400:
        raise EngineError(
            f"GCE instances.insert({spec_dict.get('name')} in {zone}) failed "
            f"({resp.status_code}): {resp.text}"
        )
    op = resp.json()
    op_name = op.get("name")
    instance_name = str(spec_dict["name"])
    deadline = time.monotonic() + provision_timeout_s

    while op_name and time.monotonic() < deadline:
        op_url = f"{_COMPUTE_API}/projects/{project_id}/zones/{zone}/operations/{op_name}"
        op_resp = session.get(op_url, timeout=_HTTP_TIMEOUT_S)
        if op_resp.status_code < 400:
            op_doc = op_resp.json()
            if op_doc.get("error"):
                errors = op_doc["error"].get("errors") or [op_doc["error"]]
                msg = "; ".join(f"{e.get('code', 'ERROR')}: {e.get('message', '')}" for e in errors)
                delete_instance(project_id, zone, instance_name, session=session)
                raise EngineError(
                    f"GCE instances.insert({instance_name} in {zone}) operation failed: {msg}"
                )
            if op_doc.get("status") == "DONE":
                break
        time.sleep(poll_interval_s)

    while time.monotonic() < deadline:
        inst = get_instance(project_id, zone, instance_name, session=session)
        if inst is None:
            # Fast run that already self-deleted after writing its status marker
            return {"name": instance_name, "status": "TERMINATED"}
        st = str(inst.get("status", ""))
        if st == "RUNNING" or st in _TERMINATED_INSTANCE_STATES:
            return inst
        time.sleep(poll_interval_s)

    delete_instance(project_id, zone, instance_name, session=session)
    raise EngineError(
        f"GCE instance {instance_name} in {zone} did not reach RUNNING within "
        f"{provision_timeout_s:.0f}s"
    )


def resolve_gce_candidates(
    *,
    settings: Settings,
    infra: BatchInfra,
) -> list[Candidate]:
    """Explicit-zone candidates for GCE VM creation (pure).

    Compute Engine `instances.insert` is zonal, so auto-zone (`zone=None`) entries from
    `resolve_candidates` are expanded to explicit zones (`US_ZONES` fallback when needed).
    """
    raw = resolve_candidates(settings=settings, infra=infra)
    explicit = [c for c in raw if c.zone is not None]
    if explicit:
        return explicit
    zones = US_ZONES.get(settings.region, [f"{settings.region}-a"])
    return [
        Candidate(region=settings.region, zone=z, subnetwork_uri=infra.subnetwork_uri)
        for z in zones
    ]


def _poll_gce_instance(
    session: Any,
    project_id: str,
    zone: str,
    plan: GceInstancePlan,
    *,
    run_id: str,
    wait_timeout: float,
    grace_s: int,
    since: Any,
    poll_interval_s: float = _POLL_INTERVAL_SECONDS,
) -> dict[str, Any]:  # pragma: no cover - live GCE I/O
    """Poll the GCS status marker + GCE VM state until completion, enforcing the stall watchdog."""
    from .job_outcome import cells_written

    started = time.monotonic()
    deadline = started + wait_timeout
    last_watchdog = started
    watching = grace_s > 0
    saw_terminated_at: float | None = None

    while True:
        marker = read_status_marker(plan.status_bucket, plan.status_object)
        if marker is not None and marker.get("status") in {"SUCCEEDED", "FAILED"}:
            return marker

        inst = get_instance(project_id, zone, plan.instance_name, session=session)
        inst_state = str(inst.get("status", "")) if inst is not None else "NOT_FOUND"

        if inst is None or inst_state in _TERMINATED_INSTANCE_STATES:
            # Give the guest cleanup trap a brief grace window to finish uploading the marker
            if saw_terminated_at is None:
                saw_terminated_at = time.monotonic()
            elif (time.monotonic() - saw_terminated_at) >= 15.0:
                final_marker = read_status_marker(plan.status_bucket, plan.status_object)
                if final_marker is not None and final_marker.get("status") in {
                    "SUCCEEDED",
                    "FAILED",
                }:
                    return final_marker
                return {
                    "status": "FAILED",
                    "exit_code": -1,
                    "instance": plan.instance_name,
                    "zone": zone,
                    "detail": (
                        f"GCE instance {plan.instance_name} reached {inst_state} before writing "
                        f"a terminal status marker"
                    ),
                }

        now = time.monotonic()
        if now >= deadline:
            delete_instance(project_id, zone, plan.instance_name, session=session)
            raise EngineError(
                f"GCE instance {plan.instance_name} in {zone} did not finish within "
                f"{wait_timeout:.0f}s (last VM state: {inst_state})"
            )

        if (
            watching
            and (now - started) >= grace_s
            and (now - last_watchdog) >= _WATCHDOG_CHECK_SECONDS
        ):
            last_watchdog = now
            elapsed = now - started
            cells = cells_written(run_id, since=since)
            if cells:
                _log.info(
                    "GCE instance %s has written %d cell(s); watchdog stands down",
                    plan.instance_name,
                    cells,
                )
                watching = False
            elif is_stalled(elapsed_s=elapsed, grace_s=grace_s, cells=cells):
                _log.error(
                    "GCE instance %s wrote no cells in %.0f min; deleting VM",
                    plan.instance_name,
                    elapsed / 60,
                )
                delete_instance(project_id, zone, plan.instance_name, session=session)
                raise EngineError(
                    f"GCE instance {plan.instance_name} wrote no forecast rows in "
                    f"{elapsed / 60:.0f} minutes and was deleted (stall watchdog)"
                )

        time.sleep(poll_interval_s)


def submit_gce(
    cfg: RunConfig,
    settings: Settings | None = None,
    infra: BatchInfra | None = None,
    *,
    wait: bool = True,
    models: list[str] | None = None,
    job_id: str | None = None,
    instance_name: str | None = None,
    manage_header: bool = True,
    hardware: str | None = None,
    gpu_type: str | None = None,
    machine_type: str | None = None,
    accelerator_count: int | None = None,
) -> tuple[str, str, ProbeHandle]:
    """Stage config + code and run on a single Compute Engine VM with zero-orphan teardown.

    Returns ``(run_id, instance_resource_path, probe_handle)``.
    """
    settings = settings or Settings.resolve()
    infra = infra or BatchInfra.resolve()
    if not infra.container_image:
        raise ConfigError(
            "GCE runtime requires SF_CONTAINER_IMAGE to be set (see docker/Dockerfile)."
        )

    run_id = make_run_id(cfg)
    python_models, _ = split_by_runtime(cfg)
    if models is not None:
        allow = set(models)
        python_models = [m for m in python_models if m in allow]
    if not python_models:
        raise ConfigError(
            "submit_gce called with a config that has no Python-runtime models to execute."
        )

    code_bucket = infra.code_bucket
    package_uri, _ = staging.stage_code(code_bucket)
    config_uri = staging.stage_config(cfg, run_id, code_bucket)
    profile = profile_for_run(cfg, settings=settings)

    effective_job_id = instance_name or job_id
    max_run_s = max(600, min(86400, int(infra.batch_job_wait_seconds) + 600))
    base_plan = plan_gce_job(
        cfg,
        python_models,
        run_id=run_id,
        job_id=effective_job_id,
        hardware=hardware,
        gpu_type=gpu_type,
        machine_type=machine_type,
        accelerator_count=accelerator_count,
        image_uri=infra.container_image,
        package_uri=package_uri,
        config_uri=config_uri,
        service_account=infra.compute_sa,
        subnetwork_uri=infra.subnetwork_uri,
        status_bucket=code_bucket,
        max_run_duration_seconds=max_run_s,
        profile=profile,
    )
    sizing = sizing_telemetry(
        base_plan.resource_plan,
        profile=profile,
        family="+".join(pool_families(python_models)) or "cpu",
    )
    with contextlib.suppress(Exception):
        merge_header_telemetry(
            run_id,
            {sizing_telemetry_path(sizing): sizing},
            settings=settings,
        )

    driver_args = build_driver_args(
        config_uri,
        settings,
        models=models,
        manage_header=manage_header,
        provisioned_hardware=base_plan.hardware,
    )

    candidates = resolve_gce_candidates(settings=settings, infra=infra)
    regions = list(dict.fromkeys(c.region for c in candidates))
    policy = cfg.compute.capacity.policy_for("gce")
    ledger = capacity.CapacityLedger(service="gce")

    preflights: dict[str, quota.QuotaPreflight] = {}
    if cfg.compute.capacity.preflight:
        preflights = quota.preflight_gce(base_plan, regions, settings.project_id)
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

    session = _authorized_session()
    chosen_zone: str | None = None

    def _attempt(cand: Candidate) -> tuple[str, ProbeHandle, Any, str]:
        assert cand.zone is not None
        zone = cand.zone
        plan = (
            base_plan
            if cand.subnetwork_uri == base_plan.subnetwork_uri
            else GceInstancePlan(**{**base_plan.__dict__, "subnetwork_uri": cand.subnetwork_uri})
        )
        _write_status_marker(
            plan.status_bucket,
            plan.status_object,
            {
                "status": "PROVISIONING",
                "exit_code": None,
                "instance": plan.instance_name,
                "zone": zone,
                "detail": "",
            },
        )
        spec_dict = build_gce_instance_spec_dict(
            plan,
            driver_args,
            project_id=settings.project_id,
            zone=zone,
            labels={"run_id": run_id[:63], "runtime": "gce"},
        )
        _log.info(
            "Creating GCE VM %s in %s: %s (%s, TTL=%ds)",
            plan.instance_name,
            zone,
            plan.machine_type,
            plan.accelerator_type if plan.accelerator_count > 0 else "CPU",
            plan.max_run_duration_seconds,
        )
        since = launch_window_start()
        _insert_and_wait_running(session, settings.project_id, zone, spec_dict)
        resource_path = (
            f"projects/{settings.project_id}/zones/{zone}/instances/{plan.instance_name}"
        )
        handle = ProbeHandle(
            runtime="gce",
            native_id=plan.instance_name,
            region=cand.region,
            resource_name=resource_path,
        )
        if effective_job_id is not None:
            with contextlib.suppress(Exception):
                update_job(
                    effective_job_id,
                    settings=settings,
                    merge_telemetry={
                        "probe_handle": handle.to_blob(),
                        "gce_instance": {
                            **plan.to_dict(),
                            "region": cand.region,
                            "zone": zone,
                            "resource_path": resource_path,
                        },
                    },
                )
        return resource_path, handle, since, zone

    try:
        resource_path, handle, since, chosen_zone = capacity.walk(
            usable_candidates,
            _attempt,
            policy=policy,
            ledger=ledger,
            on_state=capacity.current_publisher(),
        )
    finally:
        if effective_job_id is not None and (ledger.attempts or ledger.preflight):
            with contextlib.suppress(Exception):
                update_job(
                    effective_job_id,
                    settings=settings,
                    merge_telemetry={"capacity": ledger.to_json()},
                )

    if wait:
        assert chosen_zone is not None
        try:
            marker = _poll_gce_instance(
                session,
                settings.project_id,
                chosen_zone,
                base_plan,
                run_id=run_id,
                wait_timeout=float(infra.batch_job_wait_seconds),
                grace_s=int(infra.stall_grace_seconds),
                since=since,
            )
            if marker.get("status") != "SUCCEEDED":
                detail = marker.get("detail") or f"exit_code={marker.get('exit_code')}"
                raise EngineError(
                    f"GCE instance {base_plan.instance_name} in {chosen_zone} failed: {detail}"
                )
        finally:
            # Layer 3 zero-orphan guarantee: unconditionally delete the VM on exit (404-safe if
            # the guest startup-script trap already self-deleted the instance).
            delete_instance(
                settings.project_id, chosen_zone, base_plan.instance_name, session=session
            )

    return run_id, resource_path, handle


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scale_forecasting.gce_submit",
        description="Stage config + code and run on a single Compute Engine VM.",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", help="Path to the run config JSON.")
    src.add_argument("--config-uri", help="gs:// URI of a staged config (portable source).")
    p.add_argument(
        "--n-series",
        type=int,
        default=None,
        help="Override data.series_limit.",
    )
    p.add_argument(
        "--models",
        default=None,
        help="Comma-separated model subset for per-family DAG jobs.",
    )
    p.add_argument(
        "--job-id",
        default=None,
        help="Deterministic job_key (sf-<run_id>-<family>-a<n>).",
    )
    p.add_argument(
        "--instance-name",
        default=None,
        help="Explicit GCE instance name override.",
    )
    p.add_argument(
        "--no-manage-header",
        action="store_true",
        help="Do not start/complete the run_registry header (caller owns it).",
    )
    p.add_argument(
        "--hardware",
        choices=("cpu", "gpu"),
        default=None,
        help="Per-family hardware override ('cpu' or 'gpu').",
    )
    p.add_argument(
        "--gpu-type",
        choices=("T4", "L4", "A100", "A100_80GB"),
        default=None,
        help="Per-family GPU type override ('T4', 'L4', 'A100', or 'A100_80GB').",
    )
    p.add_argument(
        "--machine-type",
        default=None,
        help="Per-family GCE machine type override (e.g. 'n2-standard-8', 'g2-standard-8').",
    )
    p.add_argument(
        "--accelerator-count",
        type=int,
        default=None,
        help="Per-family GPU count per VM override.",
    )
    p.add_argument(
        "--no-wait",
        action="store_true",
        help="Return as soon as the GCE VM reaches RUNNING.",
    )
    return p


def main(argv: list[str] | None = None) -> str:
    from .config import load_config_uri

    args = _build_parser().parse_args(argv)
    cfg = load_config_uri(args.config or args.config_uri).with_series_limit(args.n_series)
    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else None
    run_id, resource_path, _ = submit_gce(
        cfg,
        wait=not args.no_wait,
        models=models,
        job_id=args.job_id,
        instance_name=args.instance_name,
        manage_header=not args.no_manage_header,
        hardware=args.hardware,
        gpu_type=args.gpu_type,
        machine_type=args.machine_type,
        accelerator_count=args.accelerator_count,
    )
    _log.info("GCE VM run complete: run_id=%s instance=%s", run_id, resource_path)
    return run_id


if __name__ == "__main__":  # pragma: no cover
    main()
