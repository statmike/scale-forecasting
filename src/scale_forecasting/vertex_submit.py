"""Submit a run (or family slice) to Vertex AI ``CustomJob`` (single-VM or worker pool).

Why a third Python runtime beside Spark and Ray:
1. **Zero Head-Node Tax**: Vertex Ray requires a persistent `n1-standard-16` head node (16 vCPUs,
   60 GB RAM) that executes zero cells just to host the Ray dashboard proxy. A Vertex AI
   `CustomJob` (`google.cloud.aiplatform_v1.JobServiceClient.create_custom_job`) provisions only
   the worker VM(s) requested (`worker_pool_specs`), runs the container, and tears down immediately.
2. **Custom Container Support on Both CPU and GPU**: Unlike Vertex Ray (which rejects custom images
   on GPU pools), Vertex AI `CustomJob` natively runs `SF_CONTAINER_IMAGE` (`docker/Dockerfile`) on
   both CPU (`n2-standard-*`, `n1-standard-*`, `c2-*`) and GPU (`NVIDIA_TESLA_T4` on
   `n1-standard-*`, `NVIDIA_L4` on `g2-standard-*`, `NVIDIA_TESLA_A100` / `NVIDIA_A100_80GB` on
   `a2-*`).
3. **Dynamic GCS Code Delivery (Zero Image Rebuilds)**: Following the repository's code-delivery
   invariant (`tests/unit/test_code_delivery.py`), `src/scale_forecasting` is never baked into
   `SF_CONTAINER_IMAGE`. At submit time, `staging.stage_code` zips `src/scale_forecasting` to
   `gs://<code_bucket>/runs/scale_forecasting-<code_hash>.zip`, and `VERTEX_BOOTSTRAP_CODE`
   downloads and prepends that zip to `sys.path` before invoking `scale_forecasting.vertex_entry`.
4. **Single-Machine Default + Multi-Worker Pool Distribution**: Defaults to a 1-replica
   `worker_pool_specs` (`workers = 1`). When `workers > 1`, provisions primary pool 0
   (`replica_count = 1`) + secondary pool 1 (`replica_count = effective_workers - 1`); Vertex AI
   injects `CLUSTER_SPEC` into every container so `vertex_engine.resolve_worker_topology`
   deterministically shards series or models across workers without a coordinator node.
"""

from __future__ import annotations

import argparse
import contextlib
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from . import capacity, quota, staging
from .batch_infra import BatchInfra
from .commands import build_driver_args
from .config import RunConfig, resolve_vm_machine_type
from .engines.ray_io import _ACCELERATOR_TYPES, pool_families, resolve_job_gpu
from .engines.vertex_engine import effective_worker_count, plan_vertex_pool
from .errors import ConfigError, EngineError, configure_cli_logging, get_logger, require_extra
from .job_outcome import launch_window_start
from .job_wait import is_stalled
from .probes.vocabulary import ProbeHandle
from .profiling.source import profile_for_run
from .ray_cluster import _resolve_regions
from .registry.header import merge_header_telemetry, sizing_telemetry_path
from .registry.ids import make_run_id, vertex_job_id
from .registry.jobs import update_job
from .resources.audit import sizing_telemetry
from .resources.catalog import intraop_env_vars
from .resources.fleet import RuntimeResourcePlan
from .router import split_by_runtime
from .settings import Settings

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .profiling.cost import ComputeProfile

_log = get_logger(__name__)

# Self-contained bootstrap snippet passed to `/opt/venv/bin/python -c` inside the CustomJob
# container. Downloads the content-addressed `scale_forecasting-<code_hash>.zip` from GCS using the
# pre-baked `google-cloud-storage` client in `/opt/venv`, switches to a writable scratch working
# directory when the image's WORKDIR (`/opt/scale-forecasting`) is owned by root (container runs as
# `USER spark`), inserts the zip at `sys.path[0]`, and hands the remaining driver arguments to
# `scale_forecasting.vertex_entry.main`.
VERTEX_BOOTSTRAP_CODE = (
    "import os, sys, tempfile, google.cloud.storage as gcs\n"
    "td = tempfile.mkdtemp(prefix='sf_vertex_')\n"
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
    "sys.path.insert(0, zp)\n"
    "from scale_forecasting.vertex_entry import main\n"
    "main(rest)\n"
)

_POLL_INTERVAL_SECONDS = 15.0
_WATCHDOG_CHECK_SECONDS = 300.0

_TERMINAL_JOB_STATES = frozenset(
    {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }
)


@dataclass(frozen=True)
class VertexCustomJobPlan:
    """Resolved execution plan for a Vertex AI ``CustomJob`` submission (pure)."""

    display_name: str
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
    resource_plan: RuntimeResourcePlan | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe summary for `run_jobs.job_telemetry.$.vertex_custom_job`."""
        return {
            "display_name": self.display_name,
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
            "resource_plan": (
                self.resource_plan.to_dict() if self.resource_plan is not None else None
            ),
        }


def plan_vertex_job(
    cfg: RunConfig,
    models: Sequence[str] | None = None,
    *,
    run_id: str | None = None,
    job_id: str | None = None,
    hardware: str | None = None,
    gpu_type: str | None = None,
    machine_type: str | None = None,
    worker_count: int | None = None,
    accelerator_count: int | None = None,
    image_uri: str = "",
    package_uri: str = "",
    config_uri: str = "",
    service_account: str = "",
    profile: ComputeProfile | None = None,
) -> VertexCustomJobPlan:
    """Resolve a `VertexCustomJobPlan` from `cfg` and per-family overrides (pure)."""
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
        accel_type = _ACCELERATOR_TYPES.get(resolved_gpu_type, f"NVIDIA_{resolved_gpu_type}")
        accel_count = eff_accel
    else:
        accel_type = "ACCELERATOR_TYPE_UNSPECIFIED"
        accel_count = 0

    requested = worker_count if worker_count is not None else cfg.compute.workers
    effective = effective_worker_count(
        cfg,
        selected_models,
        requested,
        n_series=cfg.data.series_limit,
    )
    display_name = vertex_job_id(job_id) if job_id else f"sf-{rid}-vertex"

    n_series = cfg.data.series_limit
    basis = n_series if n_series is not None else cfg.compute.max_parallelism
    resource_plan = plan_vertex_pool(
        cfg,
        selected_models,
        basis * len(selected_models),
        runtime="vertex",
        gpu=(resolved_hw == "gpu"),
        gpu_type=resolved_gpu_type,
        machine_type=resolved_machine,
        worker_count=effective,
        profile=profile,
    )

    return VertexCustomJobPlan(
        display_name=display_name,
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
        resource_plan=resource_plan,
    )


def build_custom_job_spec_dict(
    plan: VertexCustomJobPlan,
    driver_args: Sequence[str],
    *,
    labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build the Vertex AI ``CustomJob`` protobuf-compatible dictionary (pure).

    Vertex AI requires ``worker_pool_specs[0].replica_count == 1`` (the primary pool). When
    ``plan.worker_count > 1``, a second pool (``worker_pool_specs[1]``) carries the remaining
    ``plan.worker_count - 1`` replicas with an identical machine and container specification.
    Vertex AI automatically populates ``CLUSTER_SPEC`` across both pools so each replica resolves
    its unique ``(rank, world_size)`` in `vertex_engine.resolve_worker_topology`.
    """
    machine_spec: dict[str, Any] = {"machine_type": plan.machine_type}
    if plan.accelerator_count > 0 and plan.accelerator_type != "ACCELERATOR_TYPE_UNSPECIFIED":
        machine_spec["accelerator_type"] = plan.accelerator_type
        machine_spec["accelerator_count"] = plan.accelerator_count

    threads = plan.resource_plan.slot.cores if plan.resource_plan is not None else 1
    env_items = [
        {"name": "SF_VERTEX_JOB_ID", "value": plan.display_name},
        *[
            {"name": k, "value": v}
            for k, v in sorted(intraop_env_vars(threads, include_omp=True).items())
        ],
    ]

    container_spec: dict[str, Any] = {
        "image_uri": plan.image_uri,
        "command": ["/opt/venv/bin/python", "-c", VERTEX_BOOTSTRAP_CODE],
        "args": ["--package-uri", plan.package_uri, *driver_args],
        "env": env_items,
    }

    worker_pool_specs: list[dict[str, Any]] = [
        {
            "machine_spec": dict(machine_spec),
            "replica_count": 1,
            "container_spec": dict(container_spec),
        }
    ]
    if plan.worker_count > 1:
        worker_pool_specs.append(
            {
                "machine_spec": dict(machine_spec),
                "replica_count": plan.worker_count - 1,
                "container_spec": dict(container_spec),
            }
        )

    job_spec: dict[str, Any] = {"worker_pool_specs": worker_pool_specs}
    if plan.service_account:
        job_spec["service_account"] = plan.service_account

    custom_job: dict[str, Any] = {
        "display_name": plan.display_name,
        "job_spec": job_spec,
    }
    if labels:
        custom_job["labels"] = dict(labels)
    return custom_job


def _job_client(region: str) -> Any:  # pragma: no cover - live Vertex I/O
    """Regional Vertex AI `JobServiceClient`."""
    from google.cloud import aiplatform_v1

    return aiplatform_v1.JobServiceClient(
        client_options={"api_endpoint": f"{region}-aiplatform.googleapis.com"}
    )


def _state_name(state: Any) -> str:
    """Normalize a gapic `JobState` enum or int/string to its `"JOB_STATE_*"` string name."""
    name = getattr(state, "name", None)
    if isinstance(name, str):
        return name
    return str(state)


_PROVISIONING_STATES = frozenset({"JOB_STATE_QUEUED", "JOB_STATE_PENDING"})


def _wait_for_provisioning(
    client: Any,
    job_name: str,
    *,
    region: str,
    poll_interval_s: float = _POLL_INTERVAL_SECONDS,
) -> Any:
    """Wait until `job_name` leaves `QUEUED`/`PENDING` so stockouts trigger region hop."""
    while True:
        job = client.get_custom_job(name=job_name)
        state = _state_name(getattr(job, "state", ""))
        if state in _PROVISIONING_STATES:
            time.sleep(poll_interval_s)
            continue
        if state in _TERMINAL_JOB_STATES and state != "JOB_STATE_SUCCEEDED":
            err_msg = getattr(getattr(job, "error", None), "message", "") or state
            raise EngineError(
                f"Vertex CustomJob {job_name} in {region} failed during provisioning "
                f"({state}): {err_msg}"
            )
        return job


def _poll_custom_job(
    client: Any,
    job_name: str,
    *,
    run_id: str,
    wait_timeout: float,
    grace_s: int,
    since: Any,
    poll_interval_s: float = _POLL_INTERVAL_SECONDS,
) -> Any:
    """Poll `client.get_custom_job` until terminal, enforcing the output stall watchdog."""
    from .job_outcome import cells_written

    started = time.monotonic()
    deadline = started + wait_timeout
    last_watchdog = started
    watching = grace_s > 0

    while True:
        job = client.get_custom_job(name=job_name)
        state = _state_name(getattr(job, "state", ""))
        if state in _TERMINAL_JOB_STATES:
            return job

        now = time.monotonic()
        if now >= deadline:
            with contextlib.suppress(Exception):
                client.cancel_custom_job(name=job_name)
            raise EngineError(
                f"Vertex CustomJob {job_name} did not reach a terminal state within "
                f"{wait_timeout:.0f}s (last state: {state})"
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
                    "Vertex CustomJob %s has written %d cell(s); watchdog stands down",
                    job_name,
                    cells,
                )
                watching = False
            elif is_stalled(elapsed_s=elapsed, grace_s=grace_s, cells=cells):
                _log.error(
                    "Vertex CustomJob %s wrote no cells in %.0f min; cancelling",
                    job_name,
                    elapsed / 60,
                )
                with contextlib.suppress(Exception):
                    client.cancel_custom_job(name=job_name)
                raise EngineError(
                    f"Vertex CustomJob {job_name} wrote no forecast rows in {elapsed / 60:.0f} "
                    f"minutes and was cancelled (stall watchdog; raise SF_STALL_GRACE_S, or set "
                    f"it to 0, if this run legitimately takes that long to produce its first cell)"
                )

        time.sleep(poll_interval_s)


def submit_vertex(
    cfg: RunConfig,
    settings: Settings | None = None,
    infra: BatchInfra | None = None,
    *,
    wait: bool = True,
    models: list[str] | None = None,
    job_id: str | None = None,
    manage_header: bool = True,
    hardware: str | None = None,
    gpu_type: str | None = None,
    machine_type: str | None = None,
    worker_count: int | None = None,
    accelerator_count: int | None = None,
) -> tuple[str, str, ProbeHandle]:
    """Stage config + code and submit a Vertex AI ``CustomJob`` across candidate regions.

    Returns ``(run_id, custom_job_resource_name, probe_handle)``.
    """
    settings = settings or Settings.resolve()
    infra = infra or BatchInfra.resolve()
    if not infra.container_image:
        raise ConfigError(
            "Vertex AI CustomJob requires SF_CONTAINER_IMAGE to be set (see docker/Dockerfile)."
        )

    run_id = make_run_id(cfg)
    python_models, _ = split_by_runtime(cfg)
    if models is not None:
        allow = set(models)
        python_models = [m for m in python_models if m in allow]
    if not python_models:
        raise ConfigError(
            "submit_vertex called with a config that has no Python-runtime models to execute."
        )

    code_bucket = infra.code_bucket
    package_uri, _ = staging.stage_code(code_bucket)
    config_uri = staging.stage_config(cfg, run_id, code_bucket)
    profile = profile_for_run(cfg, settings=settings)

    base_plan = plan_vertex_job(
        cfg,
        python_models,
        run_id=run_id,
        job_id=job_id,
        hardware=hardware,
        gpu_type=gpu_type,
        machine_type=machine_type,
        worker_count=worker_count,
        accelerator_count=accelerator_count,
        image_uri=infra.container_image,
        package_uri=package_uri,
        config_uri=config_uri,
        service_account=infra.compute_sa,
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
    regions = _resolve_regions(cfg, settings)
    policy = cfg.compute.capacity.policy_for("vertex")
    ledger = capacity.CapacityLedger(service="vertex")

    preflights: dict[str, quota.QuotaPreflight] = {}
    if cfg.compute.capacity.preflight:
        preflights = quota.preflight_vertex(base_plan, regions, settings.project_id)
        for region in regions:
            pf = preflights.get(region)
            if pf is not None:
                ledger.preflight.append(pf.to_dict())
                for line in pf.render():
                    _log.info("%s", line)

    usable_regions = [r for r in regions if not (preflights.get(r) and preflights[r].blocked)]
    if not usable_regions:
        usable_regions = regions

    def _attempt(region: str) -> tuple[Any, str, ProbeHandle, Any, str]:
        pf = preflights.get(region)
        granted_workers = (
            quota.clamp_worker_count(base_plan.worker_count, pf)
            if pf is not None
            else base_plan.worker_count
        )
        regional_plan = (
            replace(base_plan, worker_count=granted_workers)
            if granted_workers != base_plan.worker_count
            else base_plan
        )

        client = _job_client(region)
        parent = f"projects/{settings.project_id}/locations/{region}"
        spec_dict = build_custom_job_spec_dict(
            regional_plan,
            driver_args,
            labels={"run_id": run_id[:63], "runtime": "vertex"},
        )
        _log.info(
            "Submitting Vertex CustomJob %s in %s: %d x %s (%s)",
            regional_plan.display_name,
            region,
            regional_plan.worker_count,
            regional_plan.machine_type,
            regional_plan.accelerator_type if regional_plan.accelerator_count > 0 else "CPU",
        )
        since = launch_window_start()
        created = client.create_custom_job(parent=parent, custom_job=spec_dict)
        job_name = created.name
        handle = ProbeHandle(
            runtime="vertex",
            native_id=job_name,
            region=region,
            resource_name=job_name,
        )

        if job_id is not None:
            with contextlib.suppress(Exception):
                update_job(
                    job_id,
                    settings=settings,
                    merge_telemetry={
                        "probe_handle": handle.to_blob(),
                        "vertex_custom_job": {
                            **regional_plan.to_dict(),
                            "region": region,
                            "job_name": job_name,
                        },
                    },
                )

        if wait:
            _wait_for_provisioning(client, job_name, region=region)
        return client, job_name, handle, since, region

    try:
        client, job_name, handle, since, chosen_region = capacity.walk(
            usable_regions,
            _attempt,
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

    if wait:
        finished = _poll_custom_job(
            client,
            job_name,
            run_id=run_id,
            wait_timeout=float(infra.batch_job_wait_seconds),
            grace_s=int(infra.stall_grace_seconds),
            since=since,
        )
        state = _state_name(getattr(finished, "state", ""))
        if state != "JOB_STATE_SUCCEEDED":
            err_msg = getattr(getattr(finished, "error", None), "message", "") or state
            raise EngineError(
                f"Vertex CustomJob {job_name} in {chosen_region} ended with {state}: {err_msg}"
            )

    return run_id, job_name, handle


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scale_forecasting.vertex_submit",
        description="Stage config + code and submit a Vertex AI CustomJob.",
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
        help="Per-family Vertex machine type override (e.g. 'n2-standard-8', 'g2-standard-8').",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Per-family Vertex worker count override.",
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
        help="Return as soon as the CustomJob is created.",
    )
    return p


def main(argv: list[str] | None = None) -> str:
    from .config import load_config_uri

    configure_cli_logging()
    require_extra("gcp", purpose="Submitting a Vertex AI CustomJob")
    args = _build_parser().parse_args(argv)
    cfg = load_config_uri(args.config or args.config_uri).with_series_limit(args.n_series)
    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else None
    run_id, job_name, _ = submit_vertex(
        cfg,
        wait=not args.no_wait,
        models=models,
        job_id=args.job_id,
        manage_header=not args.no_manage_header,
        hardware=args.hardware,
        gpu_type=args.gpu_type,
        machine_type=args.machine_type,
        worker_count=args.workers,
        accelerator_count=args.accelerator_count,
    )
    _log.info("Vertex CustomJob complete: run_id=%s job_name=%s", run_id, job_name)
    return run_id


if __name__ == "__main__":  # pragma: no cover
    main()
