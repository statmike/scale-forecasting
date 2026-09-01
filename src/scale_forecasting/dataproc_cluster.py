"""The Dataproc **cluster** a Spark forecast run gets: its shape, its size, its lifetime.

Not the job that runs on it — that is `cluster_submit`. This module answers "what cluster, and how
does one come to exist", which has callers that never submit anything: `main` and `airflow_tasks`
provision **one** shared cluster for a run whose families would otherwise each create and delete
their own (a family finishing first would delete the cluster out from under the rest).

Three layers, read top to bottom:

* **shape** — `build_cluster` assembles the ``dataproc_v1.Cluster`` message; `worker_machine_type` /
  `master_machine_type` are the hardware catalogue behind it. They stay beside `cluster_sizing`
  because that sizes the executor for the machine `build_cluster` will provision, and a fleet sized
  for a machine other than the one created is a silent mis-shape.
* **size** — `cluster_sizing`, the cluster analog of `submit.plan_sizing`, returning a worker count
  as well as properties because a cluster's ceiling is physical and fixed at *create*.
* **lifetime** — create (walking zone/region capacity candidates), delete, and the shared-cluster
  provision/teardown pair the DAG orchestrator calls.

The GCP imports stay lazy inside the functions that touch the network so importing this module never
pulls the ``[spark]`` extra; the pure spec builders are import-free and unit-tested. The exact GPU
init-action + accelerator wiring is validated against a live cluster (deferred); the message shapes
are pinned here so the specs and their tests stay deterministic.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any

from .batch_infra import BatchInfra
from .cluster_deps import (
    _VENV_ARCHIVE_METADATA_KEY,
    _VENV_DIR,
    _VENV_DIR_METADATA_KEY,
    _resolve_cluster_deps,
    _stage_cluster_init,
)
from .compute_fallback import Candidate, is_capacity_error, resolve_candidates
from .errors import ConfigError, EngineError, get_logger

if TYPE_CHECKING:
    from .config import RunConfig
    from .profiling.cost import ComputeProfile
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

# Core counts for the two instance groups. Fixed, not config: the profiler derives the *executor*
# shape from the machine type, and letting a user pick both the machine and the shape gives two
# knobs that can disagree. `compute.machine_family` picks the family; these pick the size.
_MASTER_CORES = 4
_WORKER_CORES = 8
# What `machine_family="auto"` resolves to — today's shipped behaviour, so an existing config
# renders the identical cluster (and, since the field's default is unchanged, the identical run_id).
_AUTO_MACHINE_FAMILY = "n1"
_DEFAULT_MASTER_MACHINE = f"{_AUTO_MACHINE_FAMILY}-standard-{_MASTER_CORES}"
_DEFAULT_WORKER_MACHINE = f"{_AUTO_MACHINE_FAMILY}-standard-{_WORKER_CORES}"

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


def cluster_name(run_id: str, spark_cluster_name: str | None) -> str:
    """The cluster name: the reuse target if set, else ``sf-cluster-<run_id>`` (Dataproc-legal).

    Dataproc cluster names must be lowercase alnum + hyphens, start with a letter, ≤ 51 chars, no
    trailing hyphen. The ``run_id`` is already a slug + hex digest, so the ``sf-cluster-`` prefix
    keeps it legal; clamp to 51 with no trailing hyphen.
    """
    if spark_cluster_name:
        return spark_cluster_name
    return f"sf-cluster-{run_id}"[:51].rstrip("-")


# --- pure: cluster spec + sizing (no network) ----------------------------------


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
    machine_family: str = "auto",
) -> object:
    """Assemble the ``dataproc_v1.Cluster`` message for one run (pure — builds the message only).

    A master + ``worker_count`` workers on the resolved subnet, running as the compute SA with
    internal-IP-only networking (the same isolation the batch uses). ``hardware="gpu"`` puts an
    accelerator on each worker (`_GPU_WORKER_MACHINE`/`_GPU_ACCELERATOR_TYPE` by ``gpu_type``) so
    the executor's Python worker can use the device. A CPU cluster attaches no accelerator and runs
    the default worker machine. ``machine_family`` (``compute.machine_family``) picks the GCE family
    for the master and the CPU workers; a GPU worker's machine is fixed by its accelerator (see
    `worker_machine_type`).

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
        machine_type_uri=master_machine_type(machine_family),
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
        worker_machine, accelerators = worker_machine_type("cpu", None, machine_family), []

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


def master_machine_type(machine_family: str = "auto") -> str:
    """The GCE machine type for the cluster's master (pure). Always CPU — it schedules."""
    return f"{_resolve_machine_family(machine_family)}-standard-{_MASTER_CORES}"


def _resolve_machine_family(machine_family: str) -> str:
    """``compute.machine_family`` → a GCE family prefix, with ``"auto"`` resolved (pure)."""
    return _AUTO_MACHINE_FAMILY if machine_family == "auto" else machine_family


def worker_machine_type(
    hardware: str, gpu_type: str | None = None, machine_family: str = "auto"
) -> str:
    """The GCE machine type this run's workers get (pure).

    The one place the answer lives, because `build_cluster` provisions against it and
    `cluster_sizing` sizes the executor that will run on it — two readings of the same fact,
    and a fleet sized for a machine other than the one created is a silent mis-shape. Raises on
    an unknown ``gpu_type``: a Dataproc cluster supports T4 or L4 (unlike Serverless, L4-only).

    ``machine_family`` (``compute.machine_family``) selects the family for a **CPU** worker; the
    size stays `_WORKER_CORES`, because the profiler derives the executor shape from the machine
    rather than the other way round. It is **ignored on GPU**, and that is a platform fact, not an
    oversight: the accelerator dictates the machine (a T4 is an add-on card that only attaches to
    n1, an L4 comes bundled inside g2), so honouring a family here would ask for a shape GCE will
    not sell. A run whose families span both hardware kinds therefore gets its CPU workers on the
    chosen family and its GPU workers on the accelerator's — the honest answer for each.
    """
    if hardware != "gpu":
        return f"{_resolve_machine_family(machine_family)}-standard-{_WORKER_CORES}"
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
    `resources.cluster.plan_dataproc_cluster` does the arithmetic; this only assembles its inputs.

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
    `profiling.source.profile_for_run` and handed in; this function stays pure and never goes
    looking.

    The third element is the audit record (`resources.audit.sizing_telemetry`) — plan, translation
    and the evidence behind them — stamped onto the run header by `submit_cluster_job` so a cluster
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
    from .resources.audit import sizing_telemetry
    from .resources.cluster import plan_dataproc_cluster

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
        machine_type=worker_machine_type(hardware, gpu_type, cfg.compute.machine_family),
        # Every GPU cluster we build attaches exactly one card per worker (`_gpu_worker`).
        accelerators=1 if gpu else 0,
        gpu=gpu,
        device_bytes=device_memory_bytes(gpu_type or cfg.compute.gpu_type) if gpu else None,
        static_gpu_fraction=float(fraction) if isinstance(fraction, float) else None,
        max_workers=max_workers,
        # See the same call in `submit.sizing_properties`: a controlled-measurement run
        # unpins the native thread pools so `effective_cores` measures the library, not the cap.
        pin_threads=not cfg.compute.profile.unpins_threads,
    )
    _log.info("cluster sizing: %s", translation.to_dict())
    return (
        translation.worker_count,
        translation.properties,
        sizing_telemetry(plan, translation=translation, profile=profile),
    )


# --- I/O: the cluster's lifetime -----------------------------------------------


def _cluster_client(region: str) -> object:  # pragma: no cover - thin client factory
    from google.api_core.client_options import ClientOptions
    from google.cloud import dataproc_v1 as dataproc

    return dataproc.ClusterControllerClient(
        client_options=ClientOptions(api_endpoint=f"{region}-dataproc.googleapis.com:443")
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
    from .profiling.source import profile_for_run
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
            "machine_family": cfg.compute.machine_family,
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
