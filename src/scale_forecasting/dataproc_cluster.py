"""Dataproc **cluster** submitter: a PySpark job on a managed Dataproc cluster (Spark, not batch).

The cluster analog of `submit`'s Serverless batch. Same on-cluster code + contract (the staged
config whose digest is the shared ``run_id``, the ``--models`` subset, contributor-mode header), but
the job runs on a Dataproc **cluster** the caller can size and — for the T4 path — a cluster is the
only Spark surface that offers T4 (Serverless is L4-only). Lifecycle mirrors the Ray path:

* **ephemeral (default):** create a cluster sized for the run → submit the PySpark job → wait to
  terminal → ``delete_cluster`` in a ``finally`` so a crashed submit never leaks a billing cluster.
* **reuse (opt-in):** ``spark_cluster_name`` targets a standing cluster by name — skip create *and*
  skip delete, for a warm cluster across runs (directly analogous to Ray's ``ray_cluster_name``).

The GCP imports stay lazy inside the functions that touch the network so importing this module never
pulls the ``[spark]`` extra; the pure spec builders (`build_cluster` / `build_job`) are import-free
and unit-tested. The exact GPU init-action + accelerator wiring is validated against a live cluster
(deferred); the message shapes are pinned here so the specs and their tests stay deterministic.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any

from .commands import build_driver_args
from .compute_fallback import Candidate, is_capacity_error, resolve_candidates
from .errors import ConfigError, EngineError, get_logger
from .submit import _ENV_VENV_ARCHIVE, BatchInfra, _stage_code, _stage_config

if TYPE_CHECKING:
    from .config import RunConfig
    from .profiling import ComputeProfile
    from .settings import Settings

_log = get_logger(__name__)

# Dataproc *cluster* image version (distinct from the Serverless runtime version): the 2.2 line on
# Debian 12, matching the Serverless runtime so on-cluster and batch code run the same stack.
_DEFAULT_IMAGE_VERSION = "2.2-debian12"

# The cluster a run gets when nothing sized it — `cluster_sizing` derives a worker count from the
# fan-out and only falls back here when profiling is switched off. A cluster's workers bill from
# create to delete, so unlike a Serverless batch the ceiling *is* this number: there is no
# Dataproc autoscaling policy behind it, and the honest statement of that is a derived, clamped
# count rather than a policy we do not have. Kept off the config so the run_id digest is unchanged.
_DEFAULT_WORKER_COUNT = 2
_DEFAULT_MASTER_MACHINE = "n1-standard-4"
_DEFAULT_WORKER_MACHINE = "n1-standard-8"

# GPU worker machine + Vertex/Compute accelerator enum per short gpu_type. A T4 is an add-on card on
# an n1 worker; an L4 is bundled into a g2 machine. (Both still declare an AcceleratorConfig so
# Dataproc exposes the device to Spark.)
_GPU_WORKER_MACHINE = {"T4": "n1-standard-8", "L4": "g2-standard-8"}
_GPU_ACCELERATOR_TYPE = {"T4": "nvidia-tesla-t4", "L4": "nvidia-l4"}

# The stock Dataproc GPU-driver install action — installs the NVIDIA driver on each node at create,
# so the executor's Python worker can use the GPU for the torch fit inside the pandas UDF.
_GPU_INIT_ACTION = (
    "gs://goog-dataproc-initialization-actions-us-central1/gpu/install_gpu_driver.sh"
)
# The stock driver install compiles the NVIDIA kernel modules from source, which exceeds Dataproc's
# 10-minute default per-init-action timeout. Give it a generous ceiling so the build finishes (still
# well under the client-side create wait below).
_GPU_INIT_TIMEOUT = timedelta(minutes=30)

# How long ``wait=True`` blocks on the job before giving up (parity with the batch wait ceiling).
_WAIT_TIMEOUT_SECONDS = 7200.0

# Packed-venv delivery: a Dataproc cluster can't use the Serverless custom container, so the locked
# dependency env is shipped as a self-contained venv archive and unpacked to a fixed absolute path
# on every node by a cluster **init action** (below). Job ``archive_uris`` are localized only to the
# *executors'* working dirs — never the client-mode *driver's* CWD (the driver runs on the master) —
# so a relative ``./env/bin/python`` fails for the driver with ``error=2, No such file or dir``.
# The init action sidesteps that: it lands the venv at ``/opt/sf-venv`` on master + workers alike,
# so both driver and executors point at the same absolute interpreter and run the exact same locked
# env as the container path — the model libraries (statsmodels, xgboost, …) live inside it.
_VENV_DIR = "/opt/sf-venv"
_VENV_PYTHON = f"{_VENV_DIR}/bin/python"
_VENV_JOB_PROPERTIES = {
    "spark.pyspark.python": _VENV_PYTHON,
    "spark.pyspark.driver.python": _VENV_PYTHON,
}

# Cluster metadata keys the init action reads to know what to fetch + where to unpack it. Metadata
# rides on the cluster (not the job), so it's available to the init action at create time on every
# node; the archive URI is the same ``venv_archive_uri`` a forecast job would otherwise attach.
_VENV_ARCHIVE_METADATA_KEY = "sf-venv-archive-uri"
_VENV_DIR_METADATA_KEY = "sf-venv-dir"

# The init-action script: on every node at cluster create, download the venv archive named in
# cluster metadata and unpack it to the absolute venv dir. The archive is a plain tar of the venv's
# *contents* (packed with ``tar -C /opt/venv .``), so it extracts straight into the target dir; the
# bundled interpreter + relative ``bin/python`` symlink make it runnable at any absolute path. Fails
# the node (``set -e``) if the metadata is missing or the fetch/unpack errors, so a broken env
# surfaces at create rather than as a silent bare-Python job later.
_CLUSTER_INIT_SCRIPT = f"""#!/bin/bash
set -euo pipefail
ARCHIVE_URI="$(/usr/share/google/get_metadata_value attributes/{_VENV_ARCHIVE_METADATA_KEY})"
VENV_DIR="$(/usr/share/google/get_metadata_value attributes/{_VENV_DIR_METADATA_KEY})"
if [[ -z "${{ARCHIVE_URI}}" || -z "${{VENV_DIR}}" ]]; then
  echo "sf venv init: missing venv cluster metadata" >&2
  exit 1
fi
mkdir -p "${{VENV_DIR}}"
gsutil -q cp "${{ARCHIVE_URI}}" /tmp/sf-venv.tar.gz
tar xzf /tmp/sf-venv.tar.gz -C "${{VENV_DIR}}"
rm -f /tmp/sf-venv.tar.gz
"""


def cluster_name(run_id: str, spark_cluster_name: str | None) -> str:
    """The cluster name: the reuse target if set, else ``sf-cluster-<run_id>`` (Dataproc-legal).

    Dataproc cluster names must be lowercase alnum + hyphens, start with a letter, ≤ 51 chars, no
    trailing hyphen. The ``run_id`` is already a slug + hex digest, so the ``sf-cluster-`` prefix
    keeps it legal; clamp to 51 with no trailing hyphen.
    """
    if spark_cluster_name:
        return spark_cluster_name
    return f"sf-cluster-{run_id}"[:51].rstrip("-")


def _resolve_cluster_deps(cfg: RunConfig, infra: BatchInfra) -> str:
    """The packed-venv archive URI a cluster job must attach, per ``compute.spark_deps`` (pure).

    A Dataproc cluster can't use the Serverless custom container, so ``packed_venv`` (the default)
    is the only viable dependency mechanism on a cluster — it requires ``infra.venv_archive_uri``
    (the ``SF_VENV_ARCHIVE`` env / terraform ``venv_archive_uri`` output). ``container`` is a
    Serverless-only mechanism, so requesting it for a cluster is a config error rather than a
    silent run with no model libraries. Raises `ConfigError` on ``container`` or a missing URI.
    """
    spark_deps = cfg.compute.spark_deps
    if spark_deps == "container":
        raise ConfigError(
            "compute.spark_deps='container' is a Dataproc Serverless mechanism, not available on a "
            "Dataproc cluster; use spark_deps='packed_venv' for cluster families"
        )
    if not infra.venv_archive_uri:
        raise ConfigError(
            "a Dataproc cluster forecast job needs the packed-venv archive but none is configured; "
            f"set {_ENV_VENV_ARCHIVE} (or the terraform 'venv_archive_uri' output). Without it the "
            "cluster runs bare Python with no model libraries and every fit fails"
        )
    return infra.venv_archive_uri


def _stage_cluster_init(infra: BatchInfra) -> str:
    """Upload the venv init-action script to the code bucket; return its ``gs://`` URI.

    Mirrors `submit._stage_code`'s staging pattern. The object name carries the script's md5 so an
    edit to the script is a new object (no in-place-overwrite races) and an unchanged script re-uses
    the same URI across runs. `build_cluster` points a `NodeInitializationAction` at the returned
    URI.
    """
    import hashlib

    from google.cloud import storage

    data = _CLUSTER_INIT_SCRIPT.encode("utf-8")
    digest = hashlib.md5(data, usedforsecurity=False).hexdigest()[:8]
    name = f"init/sf-venv-init-{digest}.sh"
    client = storage.Client()
    bucket = client.bucket(infra.code_bucket)
    bucket.blob(name).upload_from_string(data, content_type="text/x-shellscript")
    return f"gs://{infra.code_bucket}/{name}"


# --- pure: cluster + job spec assembly (no network) ----------------------------


def build_cluster(
    *,
    infra: BatchInfra,
    settings: Settings,
    project_id: str,
    name: str,
    hardware: str = "cpu",
    gpu_type: str | None = None,
    worker_count: int = _DEFAULT_WORKER_COUNT,
    venv_archive_uri: str | None = None,
    venv_init_uri: str | None = None,
    gpu_image_uri: str | None = None,
    zone: str | None = None,
    subnetwork_uri: str | None = None,
) -> object:
    """Assemble the ``dataproc_v1.Cluster`` message for one run (pure — builds the message only).

    A master + ``worker_count`` workers on the resolved subnet, running as the compute SA with
    internal-IP-only networking (the same isolation the batch uses). ``hardware="gpu"`` puts an
    accelerator on each worker (`_GPU_WORKER_MACHINE`/`_GPU_ACCELERATOR_TYPE` by ``gpu_type``) so
    the executor's Python worker can use the device. A CPU cluster attaches no accelerator and runs
    the default worker machine.

    ``zone`` and ``subnetwork_uri`` are the capacity-failover overrides (see `compute_fallback`):
    ``zone=None`` (default) lets Dataproc auto-place within the subnet's region — the pre-failover
    behavior — while an explicit zone pins the create to it; ``subnetwork_uri=None`` uses the
    deployment subnet (``infra.subnetwork_uri``); a value overrides it for a cross-region attempt.

    The GPU **driver** is delivered one of two ways: with ``gpu_image_uri`` the cluster boots from a
    custom image that has the driver pre-baked (fast, repeatable — no per-create compile), so no
    GPU-driver init action is added and `image_version` is dropped in favour of that image. Without
    it (the fallback) the stock GPU-driver init action installs the driver on each node at create.

    ``venv_archive_uri`` + ``venv_init_uri`` wire the packed-venv delivery: the archive URI (and the
    target dir) ride as cluster metadata, and `venv_init_uri` (the staged `_stage_cluster_init`
    script) becomes a `NodeInitializationAction` that unpacks the venv to `_VENV_DIR` on every node
    at create — so driver + executors share the absolute interpreter (`build_job` points Spark's
    Python there). Both are required together for a forecast cluster; omitting them (the
    pure-builder tests) yields a bare cluster with no venv metadata or init action.
    """
    from google.cloud import dataproc_v1 as dataproc

    zone_uri = zone or ""  # empty = Dataproc auto-places within the subnet's region
    subnet = subnetwork_uri or infra.subnetwork_uri

    metadata: dict[str, str] = {}
    if venv_archive_uri:
        metadata[_VENV_ARCHIVE_METADATA_KEY] = venv_archive_uri
        metadata[_VENV_DIR_METADATA_KEY] = _VENV_DIR

    # A GPU cluster boots from the pre-baked custom image when one is supplied; otherwise it
    # installs the driver at create time (the fallback). This gates several GPU-only knobs below.
    use_gpu_image = hardware == "gpu" and bool(gpu_image_uri)

    gce_kwargs: dict[str, Any] = {}
    if hardware == "gpu":
        if not use_gpu_image:
            # Fallback (no pre-baked image): the stock init action installs the driver at create.
            # Skip its cuDNN + NCCL install — both compile from source (the whole GPU-stack build
            # can run ~150 min on small nodes and is what stalls the cluster create), and we don't
            # need the system copies (the deep-learning wheel bundles its own cuDNN/NCCL). The
            # action gates both on a non-empty cuDNN version; an empty metadata value reads back as
            # empty, so it installs the base driver + CUDA only.
            metadata["cudnn-version"] = ""
        # The NVIDIA kernel modules are unsigned, which Secure Boot (on by default in the Dataproc
        # image) refuses to load — whether they were baked into the custom image or built by the
        # init action. Turn Secure Boot off on the GPU cluster's VMs so the driver loads, keeping
        # vTPM + integrity monitoring on. CPU clusters keep the stronger default (Secure Boot on).
        gce_kwargs["shielded_instance_config"] = dataproc.ShieldedInstanceConfig(
            enable_secure_boot=False,
            enable_vtpm=True,
            enable_integrity_monitoring=True,
        )

    gce = dataproc.GceClusterConfig(
        subnetwork_uri=subnet,
        service_account=infra.compute_sa,
        internal_ip_only=True,
        zone_uri=zone_uri,
        metadata=metadata,
        **gce_kwargs,
    )
    # A custom image is set per instance group (master + workers), not on SoftwareConfig; it carries
    # its own Dataproc version, so `image_version` is omitted when one is used.
    image_kwargs: dict[str, Any] = {"image_uri": gpu_image_uri} if use_gpu_image else {}

    master = dataproc.InstanceGroupConfig(
        num_instances=1,
        machine_type_uri=_DEFAULT_MASTER_MACHINE,
        **image_kwargs,
    )

    init_actions: list[Any] = []
    # Unpack the venv first so it's in place before any downstream action; the GPU driver install
    # (below) is independent and appended after.
    if venv_init_uri:
        init_actions.append(dataproc.NodeInitializationAction(executable_file=venv_init_uri))
    if hardware == "gpu":
        worker_machine, accelerators = _gpu_worker(gpu_type)
        if not use_gpu_image:
            # No pre-baked driver image: install the driver on each node at create time. (With a
            # pre-baked image the driver is already on disk, so no init action is needed.)
            init_actions.append(
                dataproc.NodeInitializationAction(
                    executable_file=_GPU_INIT_ACTION,
                    execution_timeout=_GPU_INIT_TIMEOUT,
                )
            )
    else:
        worker_machine, accelerators = _DEFAULT_WORKER_MACHINE, []

    worker = dataproc.InstanceGroupConfig(
        num_instances=worker_count,
        machine_type_uri=worker_machine,
        accelerators=accelerators,
        **image_kwargs,
    )
    software_kwargs: dict[str, Any] = {
        # Dynamic allocation lets the job scale executors within the cluster to the run's fan-out.
        "properties": {"spark:spark.dynamicAllocation.enabled": "true"},
    }
    if not use_gpu_image:
        # A *cluster* image version (2.2-debian12), distinct from the Serverless *runtime* version.
        # Omitted on the custom-image path (the image pins its own version).
        software_kwargs["image_version"] = _DEFAULT_IMAGE_VERSION
    software = dataproc.SoftwareConfig(**software_kwargs)

    return dataproc.Cluster(
        project_id=project_id,
        cluster_name=name,
        config=dataproc.ClusterConfig(
            gce_cluster_config=gce,
            master_config=master,
            worker_config=worker,
            software_config=software,
            initialization_actions=init_actions,
        ),
    )


def worker_machine_type(hardware: str, gpu_type: str | None = None) -> str:
    """The GCE machine type this run's workers get (pure).

    The one place the answer lives, because `build_cluster` provisions against it and
    `cluster_sizing` sizes the executor that will run on it — two readings of the same fact,
    and a fleet sized for a machine other than the one created is a silent mis-shape. Raises on
    an unknown ``gpu_type``: a Dataproc cluster supports T4 or L4 (unlike Serverless, L4-only).
    """
    if hardware != "gpu":
        return _DEFAULT_WORKER_MACHINE
    gt = gpu_type or "T4"
    try:
        return _GPU_WORKER_MACHINE[gt]
    except KeyError:
        raise ConfigError(
            f"unsupported gpu_type '{gt}' for a Dataproc cluster; "
            f"supported: {sorted(_GPU_WORKER_MACHINE)}"
        ) from None


def _gpu_worker(gpu_type: str | None) -> tuple[str, list[Any]]:
    """The (worker machine type, [AcceleratorConfig]) for a GPU cluster worker (pure)."""
    from google.cloud import dataproc_v1 as dataproc

    machine = worker_machine_type("gpu", gpu_type)
    accel = dataproc.AcceleratorConfig(
        accelerator_type_uri=_GPU_ACCELERATOR_TYPE[gpu_type or "T4"], accelerator_count=1
    )
    return machine, [accel]


def cluster_sizing(
    cfg: RunConfig,
    models: list[str] | None = None,
    *,
    hardware: str = "cpu",
    gpu_type: str | None = None,
    max_workers: int | None = None,
    profile: ComputeProfile | None = None,
) -> tuple[int | None, dict[str, str], dict[str, Any]]:
    """``(worker count, job properties, audit record)`` this run's shape implies (pure).

    The cluster analog of `submit.plan_sizing`, and it returns one thing more: a batch's
    fleet is entirely a property, while a cluster's ceiling is a physical worker count fixed at
    create — so the caller feeds the count to `build_cluster` and the properties to `build_job`.
    `resources.plan_dataproc_cluster` does the arithmetic; this only assembles its inputs.

    **The unit sized against is a task, not a cell** — the bucket count
    (`engines.spark_io.default_bucket_count`), each bucket holding
    ``compute.bucket_target_cells`` cells that run *sequentially* in one pandas frame. Sizing
    against cells would ask for that many times more workers than the fan-out can keep busy, and
    on a cluster an idle worker is a billed VM rather than an unclaimed quota slot.

    **Most of the win needs no measurement**: the executor shape, the thread pins, the
    device-aware ``spark.task.cpus`` and the worker count all follow from the machine type and the
    fan-out alone. The memory *split* is what a profile buys, and — as on the batch path — it
    cannot be measured here, because the cluster's shape is fixed at *create*, before any of our
    code runs on it. So ``profile`` is a previous run's measurement, resolved by
    `profiling.profile_for_run` and handed in; this function stays pure and never goes looking.

    The third element is the audit record (`resources.sizing_telemetry`) — plan, translation and
    the evidence behind them — stamped onto the run header by `submit_cluster_job` so a cluster
    run's sizing is as readable after the fact as a batch's.

    ``max_workers`` is the operator's ceiling; ``compute.profile.mode == "off"`` returns
    ``(None, {}, {})``, the documented escape hatch back to the pre-profiler two-worker cluster —
    nothing is decided, so there is nothing to record.
    """
    if cfg.compute.profile.mode == "off":
        return None, {}, {}

    # `device_memory_bytes` lives with the Ray engine because that is where the device table is
    # maintained; one table, consulted by every runtime, beats a second copy that drifts.
    from .engines.ray_io import device_memory_bytes
    from .engines.spark_io import default_bucket_count
    from .models import get_model
    from .resources import plan_dataproc_cluster, sizing_telemetry

    executed = models if models is not None else cfg.models
    families: list[str] = []
    for name in executed:
        family = get_model(name).family
        if family not in families:
            families.append(family)

    gpu = hardware == "gpu"
    fraction = cfg.compute.gpu_fraction
    plan, translation = plan_dataproc_cluster(
        profile,
        families,
        default_bucket_count(cfg, executed),
        machine_type=worker_machine_type(hardware, gpu_type),
        # Every GPU cluster we build attaches exactly one card per worker (`_gpu_worker`).
        accelerators=1 if gpu else 0,
        gpu=gpu,
        device_bytes=device_memory_bytes(gpu_type or cfg.compute.gpu_type) if gpu else None,
        static_gpu_fraction=float(fraction) if isinstance(fraction, float) else None,
        max_workers=max_workers,
        # See the same call in `submit.serverless_properties`: a controlled-measurement run
        # unpins the native thread pools so `effective_cores` measures the library, not the cap.
        pin_threads=not cfg.compute.profile.unpins_threads,
    )
    _log.info("cluster sizing: %s", translation.to_dict())
    return (
        translation.worker_count,
        translation.properties,
        sizing_telemetry(plan, translation=translation, profile=profile),
    )


def build_job(
    *,
    cluster: str,
    launcher_uri: str,
    package_uri: str,
    config_uri: str,
    settings: Settings,
    models: list[str] | None = None,
    manage_header: bool = True,
    use_venv: bool = False,
    properties: dict[str, str] | None = None,
) -> object:
    """Assemble the ``dataproc_v1.Job`` (a PySpark job placed on ``cluster``) (pure).

    Same launcher shim + package zip + driver args as the Serverless batch (`build_driver_args`), so
    on-cluster code and its contract (``--config-uri``/``--models``/``--manage-header``) are
    identical between the batch and cluster surfaces.

    ``use_venv`` points the job's ``spark.pyspark.python`` / ``spark.pyspark.driver.python`` at the
    absolute ``_VENV_PYTHON`` the cluster's venv init action lands on every node (see
    `build_cluster`), so on-cluster code runs the exact same locked env as the container path. The
    interpreter is delivered by the *cluster* (init action), not attached to the *job* — job
    ``archive_uris`` reach only executors, never the client-mode driver. Without ``use_venv`` the
    job runs the cluster's bare Python (no model libraries), so it's required for a forecast job.

    ``properties`` is the sizing overlay (`cluster_sizing`), applied over the interpreter pins so
    a shape decision can never displace the interpreter the venv init action landed. Left
    ``None`` — every pure-builder test, and any run with profiling off — the job carries exactly
    the properties it carried before, and the cluster's own ``spark-defaults`` stand.
    """
    from google.cloud import dataproc_v1 as dataproc

    args = build_driver_args(config_uri, settings, models=models, manage_header=manage_header)
    job_properties: dict[str, str] = dict(properties or {})
    if use_venv:
        job_properties.update(_VENV_JOB_PROPERTIES)
    return dataproc.Job(
        placement=dataproc.JobPlacement(cluster_name=cluster),
        pyspark_job=dataproc.PySparkJob(
            main_python_file_uri=launcher_uri,
            python_file_uris=[package_uri],
            args=args,
            properties=job_properties,
        ),
    )


# --- I/O: clients + lifecycle --------------------------------------------------


def _cluster_client(region: str) -> object:  # pragma: no cover - thin client factory
    from google.api_core.client_options import ClientOptions
    from google.cloud import dataproc_v1 as dataproc

    return dataproc.ClusterControllerClient(
        client_options=ClientOptions(api_endpoint=f"{region}-dataproc.googleapis.com:443")
    )


def _job_client(region: str) -> object:  # pragma: no cover - thin client factory
    from google.api_core.client_options import ClientOptions
    from google.cloud import dataproc_v1 as dataproc

    return dataproc.JobControllerClient(
        client_options=ClientOptions(api_endpoint=f"{region}-dataproc.googleapis.com:443")
    )


def get_cluster_job(
    region: str, job_id: str, *, settings: Settings | None = None, timeout: float | None = None
) -> tuple[str, str]:  # pragma: no cover - live Dataproc I/O, exercised by the @gcp smoke
    """Read a Dataproc **cluster** job's current state without blocking; return ``(state, detail)``.

    The non-blocking read the probe path needs: today the only cluster job-state access is inside
    `_submit_job_and_wait`, which blocks to terminal. A `JobControllerClient.get_job` fetches the
    live ``JobStatus.State`` (its ``.name``) plus the status message for one already-submitted job,
    so a reader can reconcile the registry against the runtime. Raises
    ``google.api_core.exceptions.NotFound`` when the job id is unknown (the cluster was torn down,
    or the id never existed) — the caller maps that to a NOT_FOUND probe result. ``timeout`` caps
    the RPC (the probe passes a short ceiling so a slow control plane can't hang the reader).
    """
    from .settings import Settings

    settings = settings or Settings.resolve()
    result = _job_client(region).get_job(
        request={"project_id": settings.project_id, "region": region, "job_id": job_id},
        timeout=timeout,
    )
    state = result.status.state
    state_name = getattr(state, "name", str(state))
    detail = getattr(result.status, "details", "") or ""
    return state_name, detail


def cancel_cluster_job(
    region: str, job_id: str, *, settings: Settings | None = None, timeout: float | None = None
) -> None:  # pragma: no cover - live Dataproc I/O, exercised by the @gcp cancel path
    """Cancel a Dataproc **cluster** job — the write counterpart to `get_cluster_job`.

    A `JobControllerClient.cancel_job` requests that one already-submitted job stop; the cluster
    winds it down (the job's state moves through ``CANCEL_PENDING``/``CANCEL_STARTED`` to
    ``CANCELLED``). Raises ``google.api_core.exceptions.NotFound`` when the job id is unknown (the
    cluster was torn down, or the id never existed) — the cancel caller maps that to "already gone".
    ``timeout`` caps the RPC so a slow control plane can't hang the caller.
    """
    from .settings import Settings

    settings = settings or Settings.resolve()
    _job_client(region).cancel_job(
        request={"project_id": settings.project_id, "region": region, "job_id": job_id},
        timeout=timeout,
    )


def _create_cluster(
    client: Any, project_id: str, region: str, cluster: object
) -> None:  # pragma: no cover - live Dataproc I/O, exercised by the @gcp smoke
    """Create the cluster and block until it's RUNNING (raises on a create error)."""
    op = client.create_cluster(
        request={"project_id": project_id, "region": region, "cluster": cluster}
    )
    op.result(timeout=_WAIT_TIMEOUT_SECONDS)


def _create_cluster_across_candidates(
    candidates: list[Candidate],
    *,
    project_id: str,
    name: str,
    build_kwargs: dict[str, Any],
) -> Candidate:  # pragma: no cover - orchestrates live Dataproc I/O, exercised by the @gcp smoke
    """Create the cluster, walking ``candidates`` until one has capacity; return the one that won.

    Compute capacity is zonal and stocks out transiently (`compute_fallback`): the first candidate
    is the deployment region with auto-zone placement — identical to the pre-failover single attempt
    — so a run that would have succeeded takes the same first step. On a *capacity* failure
    (`is_capacity_error`: a Compute Engine ``ServiceUnavailable``/``ResourceExhausted`` or an
    "insufficient resources"/"does not have enough resources" message) the partial cluster is torn
    down and the next candidate tried; any *other* error (bad machine type, missing quota,
    permission) is re-raised at once because another zone/region won't fix it. Each attempt targets
    its candidate's region (the regional cluster client) and pins its zone + subnet via
    `build_cluster`. Exhausting every candidate raises `EngineError` naming how many were tried.
    """
    last_exc: Exception | None = None
    for cand in candidates:
        client = _cluster_client(cand.region)
        cluster = build_cluster(
            name=name, zone=cand.zone, subnetwork_uri=cand.subnetwork_uri, **build_kwargs
        )
        try:
            _log.info("attempting cluster %s at %s", name, cand.label)
            _create_cluster(client, project_id, cand.region, cluster)
            _log.info("cluster %s created at %s", name, cand.label)
            return cand
        except Exception as exc:  # noqa: BLE001 - classify, then either advance or re-raise
            if not is_capacity_error(exc):
                raise
            _log.warning(
                "no capacity for cluster %s at %s (%s); trying next candidate",
                name,
                cand.label,
                exc,
            )
            # A create that errors mid-provision can still leave a resource — clean it before hop.
            _delete_cluster(client, project_id, cand.region, name)
            last_exc = exc
    raise EngineError(
        f"cluster {name} could not be created in any of {len(candidates)} candidate "
        f"zone(s)/region(s) (no capacity): last error {last_exc!r}"
    )


def _delete_cluster(
    client: Any, project_id: str, region: str, name: str
) -> None:  # pragma: no cover - live Dataproc I/O, exercised by the @gcp smoke
    """Delete the cluster, block until gone (best-effort — a delete error is logged, not raised)."""
    try:
        op = client.delete_cluster(
            request={"project_id": project_id, "region": region, "cluster_name": name}
        )
        op.result(timeout=_WAIT_TIMEOUT_SECONDS)
    except Exception as exc:  # noqa: BLE001 - teardown is best-effort; never mask the run's outcome
        _log.warning("cluster %s delete failed (non-fatal, may need manual cleanup): %r", name, exc)


def _submit_job_and_wait(
    client: Any, project_id: str, region: str, job: object, *, wait: bool
) -> tuple[str, str, str]:  # pragma: no cover - live Dataproc I/O, exercised by the @gcp smoke
    """Submit the PySpark job; return ``(job_id, state_name, detail)``.

    With ``wait`` block to terminal and return the terminal state; without it return the immediate
    post-submit state. ``detail`` carries the driver status message on a non-DONE terminal state.
    """
    op = client.submit_job_as_operation(
        request={"project_id": project_id, "region": region, "job": job}
    )
    submitted = op.metadata.job_id if not wait else None
    if not wait:
        return (submitted or "", "SUBMITTED", "")
    result = op.result(timeout=_WAIT_TIMEOUT_SECONDS)
    job_id = result.reference.job_id
    state = result.status.state
    state_name = getattr(state, "name", str(state))
    detail = getattr(result.status, "details", "") or ""
    return (job_id, state_name, detail)


def submit_cluster_job(
    cfg: RunConfig,
    *,
    settings: Settings | None = None,
    infra: BatchInfra | None = None,
    models: list[str] | None = None,
    manage_header: bool = True,
    hardware: str = "cpu",
    gpu_type: str | None = None,
    spark_cluster_name: str | None = None,
    spark_cluster_region: str | None = None,
    job_id: str | None = None,
    worker_count: int | None = None,
    max_workers: int | None = None,
    wait: bool = True,
) -> tuple[str, str]:
    """Stage code + config, run a PySpark job on a Dataproc cluster; return ``(job id, region)``.

    The cluster analog of `submit.submit_batch`. Stages the **full** ``cfg`` (so its ``run_id``
    matches `main.run`'s) and the shared launcher + package, then runs the lifecycle:

    * **ephemeral (default):** create a sized cluster → submit → (with ``wait``) poll to terminal →
      ``delete_cluster`` in a ``finally`` so teardown happens even if the job raises.
    * **reuse:** ``spark_cluster_name`` targets a standing cluster — skip create *and* delete.

    ``models``/``manage_header`` carry the on-cluster contract; ``hardware``/``gpu_type`` size the
    workers' accelerator (a cluster is the T4 Spark path). ``job_id`` is the deterministic
    per-family id (the orchestrator's `registry.ids.dataproc_job_id`), used only as a fallback: the
    returned id is Dataproc's own server-assigned ``reference.job_id`` (the console-resolvable one)
    when we waited. The returned ``region`` is where the job actually ran (the reuse target's
    region, or the region an ephemeral create landed in after any capacity failover) so the caller
    can record a probe-able coordinate. With ``wait`` a non-DONE terminal state raises so a failed
    job never exits 0.

    The ephemeral create walks zone/region capacity candidates (`compute_fallback`): the deployment
    region's auto-zone first (unchanged), then its other zones, then opt-in cross-region — hopping
    on a transient stockout so a scarce-GPU run isn't lost to one stocked-out zone. The job then
    submits to (and the cluster is torn down in) the region the create landed in. On the reuse path
    ``spark_cluster_region`` names the region the reuse target lives in (a shared cluster may have
    hopped; defaults to the deployment region for a standing cluster); no failover, no teardown.

    ``worker_count`` left ``None`` (every caller today) derives the cluster's size from the run's
    fan-out (`cluster_sizing`), bounded by ``max_workers``; an explicit number overrides the
    derivation outright. The job carries the matching executor/task overlay either way — on the
    reuse path the count is moot (the cluster exists) but the overlay still applies, which is what
    keeps a family's own shape correct on a cluster sized for the union of several.
    """
    from .profiling import profile_for_run
    from .registry.ids import make_run_id
    from .settings import Settings

    settings = settings or Settings.resolve()
    infra = infra or BatchInfra.resolve()
    run_id = make_run_id(cfg)
    name = cluster_name(run_id, spark_cluster_name)
    reuse = spark_cluster_name is not None
    venv_archive_uri = _resolve_cluster_deps(cfg, infra)
    derived_workers, job_properties, sizing = cluster_sizing(
        cfg,
        models,
        hardware=hardware,
        gpu_type=gpu_type,
        max_workers=max_workers,
        profile=profile_for_run(cfg, settings=settings),
    )
    workers = worker_count if worker_count is not None else derived_workers
    workers = workers if workers is not None else _DEFAULT_WORKER_COUNT

    package_uri, launcher_uri = _stage_code(infra)
    config_uri = _stage_config(cfg, run_id, infra)
    # The venv init-action script is only needed when we create the cluster; a reuse target already
    # carries the init action (it was created with one), so skip the upload on the reuse path.
    venv_init_uri = None if reuse else _stage_cluster_init(infra)
    project_id = settings.project_id

    _log.info(
        "cluster submit: run_id=%s cluster=%s reuse=%s hardware=%s workers=%d",
        run_id,
        name,
        reuse,
        hardware,
        workers,
    )

    if reuse:
        # A named cluster (standing, or the run's shared ephemeral): submit to its region — a shared
        # cluster may have hopped on a capacity failover, so trust the region the caller threads.
        region = spark_cluster_region or settings.region
    else:
        # Ephemeral: create with zone/region capacity failover; the create lands in some candidate's
        # region and the rest of the lifecycle (submit, teardown) follows it there.
        candidates = resolve_candidates(settings=settings, infra=infra)
        landed = _create_cluster_across_candidates(
            candidates,
            project_id=project_id,
            name=name,
            build_kwargs={
                "infra": infra,
                "settings": settings,
                "project_id": project_id,
                "hardware": hardware,
                "gpu_type": gpu_type,
                "worker_count": workers,
                "venv_archive_uri": venv_archive_uri,
                "venv_init_uri": venv_init_uri,
                "gpu_image_uri": infra.gpu_image_uri,
            },
        )
        region = landed.region

    cluster_client = _cluster_client(region)
    job_client = _job_client(region)

    try:
        job = build_job(
            cluster=name,
            launcher_uri=launcher_uri,
            package_uri=package_uri,
            config_uri=config_uri,
            settings=settings,
            models=models,
            manage_header=manage_header,
            use_venv=True,
            properties=job_properties,
        )
        submitted_id, state_name, detail = _submit_job_and_wait(
            job_client, project_id, region, job, wait=wait
        )
        # A cluster job id is server-assigned (not client-set), so return the *real* id Dataproc
        # assigned (``reference.job_id``) — that's what resolves in the console for reverse-trace.
        # Fall back to the deterministic per-family id only when we didn't wait and no id came back.
        final_id = submitted_id or job_id or ""
        if wait and state_name != "DONE":
            raise EngineError(
                f"cluster job {final_id} terminal state {state_name}: {detail or '(no detail)'}"
            )
        return final_id, region
    finally:
        # Stamp the sizing decision even on the failure path: a cluster job that OOM'd is exactly
        # the one whose executor split someone needs to read, and the header is the only place it
        # would survive the teardown below. Best-effort, like every other telemetry write.
        _stamp_cluster_telemetry(run_id, sizing, settings)
        if not reuse:
            _delete_cluster(cluster_client, project_id, region, name)


def _stamp_cluster_telemetry(
    run_id: str, sizing: dict[str, Any], settings: Settings
) -> None:  # pragma: no cover - GCP I/O
    """Write a cluster job's sizing record to the run header's ``job_telemetry`` (best-effort).

    The cluster path's answer to `submit._stamp_job_telemetry`. It stamps *less*: a Dataproc job
    has no ``approximate_usage`` and no runtime-config echo, so there is no DCU figure and no
    resolved-shape read-back — only what we decided and why. That is still the half of the record
    nobody could see before, and a cluster run that previously left ``v_run_summary`` blank now
    at least says what it asked for.

    Wrapped so any failure (API error, header not written) is logged and swallowed: telemetry is
    an overlay on an already-finished job, never a reason to fail one.
    """
    if not sizing:
        return
    from .registry import bq

    try:
        bq.merge_header_telemetry(
            run_id, {bq.sizing_telemetry_path(sizing): sizing}, settings=settings
        )
        _log.info("cluster sizing stamped for run %s", run_id)
    except Exception as exc:  # noqa: BLE001 - telemetry is best-effort, never fatal
        _log.warning("cluster sizing capture failed (non-fatal): %r", exc)


def provision_shared_cluster(
    cfg: RunConfig,
    *,
    run_id: str,
    use_gpu: bool,
    gpu_type: str | None = None,
    settings: Settings | None = None,
    infra: BatchInfra | None = None,
    models: list[str] | None = None,
    worker_count: int | None = None,
    max_workers: int | None = None,
) -> tuple[str, str]:  # pragma: no cover - live Dataproc I/O, exercised by the @gcp smoke
    """Create one shared ephemeral Dataproc cluster for a run's families; return ``(name, region)``.

    The multi-family analog of `submit_cluster_job`'s create step. When a run has more than one
    ephemeral cluster family the DAG orchestrator provisions **one** cluster here rather than let
    each family create its own: every family derives the same ``sf-cluster-<run_id>`` name and would
    each create *and* tear it down, so a family finishing first deletes the cluster out from under
    the others (``cluster is in state DELETING and cannot accept jobs``). Sized for the **union** of
    those families' hardware (GPU workers when ``use_gpu`` — any cluster family needs one; CPU
    families run on the same cluster, their jobs simply not using the GPU), mirroring the shared Ray
    cluster. The caller threads the name **and region** into every cluster family's
    `submit_cluster_job` as the ``spark_cluster_name``/``spark_cluster_region`` reuse target (each
    submits its own failure-isolated job to the region the cluster landed in, no per-family
    create/delete) and tears it down once via `teardown_shared_cluster` after all families join.
    Like the single-family path the create walks zone/region candidates (`compute_fallback`), so
    the returned region may differ from the deployment region on a capacity hop.

    Sized like the single-family path: ``models`` is the union of the cluster families' models —
    the only ones that will ever land here — so the worker count follows *their* fan-out rather
    than the whole run's (a run whose native/Ray families dwarf its Spark ones would otherwise buy
    idle VMs). ``worker_count`` overrides the derivation; ``max_workers`` caps it.
    """
    from .profiling import profile_for_run
    from .settings import Settings

    settings = settings or Settings.resolve()
    infra = infra or BatchInfra.resolve()
    name = cluster_name(run_id, None)
    venv_archive_uri = _resolve_cluster_deps(cfg, infra)
    venv_init_uri = _stage_cluster_init(infra)
    derived_workers, _properties, _sizing = cluster_sizing(
        cfg,
        models,
        hardware="gpu" if use_gpu else "cpu",
        gpu_type=gpu_type,
        max_workers=max_workers,
        profile=profile_for_run(cfg, settings=settings),
    )
    workers = worker_count if worker_count is not None else derived_workers
    workers = workers if workers is not None else _DEFAULT_WORKER_COUNT
    _log.info(
        "provisioning shared Dataproc cluster %s: use_gpu=%s workers=%d",
        name,
        use_gpu,
        workers,
    )
    candidates = resolve_candidates(settings=settings, infra=infra)
    landed = _create_cluster_across_candidates(
        candidates,
        project_id=settings.project_id,
        name=name,
        build_kwargs={
            "infra": infra,
            "settings": settings,
            "project_id": settings.project_id,
            "hardware": "gpu" if use_gpu else "cpu",
            "gpu_type": gpu_type,
            "worker_count": workers,
            "venv_archive_uri": venv_archive_uri,
            "venv_init_uri": venv_init_uri,
            "gpu_image_uri": infra.gpu_image_uri,
        },
    )
    return name, landed.region


def teardown_shared_cluster(
    name: str, region: str, settings: Settings | None = None
) -> None:  # pragma: no cover - live Dataproc I/O, exercised by the @gcp smoke
    """Delete the shared ephemeral Dataproc cluster (best-effort — a delete error is logged).

    Deletes in ``region`` — the one `provision_shared_cluster` landed in, which a capacity hop may
    have moved off the deployment region.
    """
    from .settings import Settings

    settings = settings or Settings.resolve()
    _delete_cluster(_cluster_client(region), settings.project_id, region, name)
