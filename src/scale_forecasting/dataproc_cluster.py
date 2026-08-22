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

from typing import TYPE_CHECKING, Any

from .commands import build_driver_args
from .errors import ConfigError, EngineError, get_logger
from .submit import _ENV_VENV_ARCHIVE, BatchInfra, _stage_code, _stage_config

if TYPE_CHECKING:
    from .config import RunConfig
    from .settings import Settings

_log = get_logger(__name__)

# Dataproc *cluster* image version (distinct from the Serverless runtime version): the 2.2 line on
# Debian 12, matching the Serverless runtime so on-cluster and batch code run the same stack.
_DEFAULT_IMAGE_VERSION = "2.2-debian12"

# A small default cluster — dynamic allocation scales executors within it, so the node count is a
# ceiling, not a fixed fan-out. Tunable later; kept off the config for now so the run_id digest is
# unchanged (a per-run cluster size is a config change, deferred with shared-cluster sizing).
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

# How long ``wait=True`` blocks on the job before giving up (parity with the batch wait ceiling).
_WAIT_TIMEOUT_SECONDS = 7200.0

# Packed-venv delivery: a Dataproc cluster can't use the Serverless custom container, so the locked
# dependency env is shipped as a venv-pack archive attached to the job via ``archive_uris`` with the
# ``#env`` fragment (Dataproc localizes + unpacks it to ``./env`` in each task's working dir). Point
# the driver *and* executors at that interpreter so on-cluster code runs the exact same locked env
# as the container path — the model libraries (statsmodels, xgboost, …) live inside it.
_VENV_ARCHIVE_FRAGMENT = "#env"
_VENV_PYTHON = "./env/bin/python"
_VENV_JOB_PROPERTIES = {
    "spark.pyspark.python": _VENV_PYTHON,
    "spark.pyspark.driver.python": _VENV_PYTHON,
}


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
) -> object:
    """Assemble the ``dataproc_v1.Cluster`` message for one run (pure — builds the message only).

    A master + ``worker_count`` workers on the resolved subnet, running as the compute SA with
    internal-IP-only networking (the same isolation the batch uses). ``hardware="gpu"`` puts an
    accelerator on each worker (`_GPU_WORKER_MACHINE`/`_GPU_ACCELERATOR_TYPE` by ``gpu_type``) and
    adds the GPU-driver init action, so the executor's Python worker can use the device. A CPU
    cluster attaches no accelerator and runs the default worker machine.
    """
    from google.cloud import dataproc_v1 as dataproc

    zone_uri = ""  # empty = Dataproc auto-places within the subnet's region

    gce = dataproc.GceClusterConfig(
        subnetwork_uri=infra.subnetwork_uri,
        service_account=infra.compute_sa,
        internal_ip_only=True,
        zone_uri=zone_uri,
    )
    master = dataproc.InstanceGroupConfig(
        num_instances=1,
        machine_type_uri=_DEFAULT_MASTER_MACHINE,
    )

    init_actions: list[Any] = []
    if hardware == "gpu":
        worker_machine, accelerators = _gpu_worker(gpu_type)
        init_actions.append(dataproc.NodeInitializationAction(executable_file=_GPU_INIT_ACTION))
    else:
        worker_machine, accelerators = _DEFAULT_WORKER_MACHINE, []

    worker = dataproc.InstanceGroupConfig(
        num_instances=worker_count,
        machine_type_uri=worker_machine,
        accelerators=accelerators,
    )
    software = dataproc.SoftwareConfig(
        # A *cluster* image version (2.2-debian12), distinct from the Serverless *runtime* version.
        image_version=_DEFAULT_IMAGE_VERSION,
        # Dynamic allocation lets the job scale executors within the cluster to the run's fan-out.
        properties={"spark:spark.dynamicAllocation.enabled": "true"},
    )

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


def _gpu_worker(gpu_type: str | None) -> tuple[str, list[Any]]:
    """The (worker machine type, [AcceleratorConfig]) for a GPU cluster worker (pure).

    Raises on an unknown ``gpu_type`` — a Dataproc cluster supports T4 or L4 (unlike Serverless,
    which is L4-only).
    """
    from google.cloud import dataproc_v1 as dataproc

    gt = gpu_type or "T4"
    try:
        machine = _GPU_WORKER_MACHINE[gt]
        accel_type = _GPU_ACCELERATOR_TYPE[gt]
    except KeyError:
        raise ConfigError(
            f"unsupported gpu_type '{gt}' for a Dataproc cluster; "
            f"supported: {sorted(_GPU_WORKER_MACHINE)}"
        ) from None
    accel = dataproc.AcceleratorConfig(accelerator_type_uri=accel_type, accelerator_count=1)
    return machine, [accel]


def build_job(
    *,
    cluster: str,
    launcher_uri: str,
    package_uri: str,
    config_uri: str,
    settings: Settings,
    engine: str,
    models: list[str] | None = None,
    manage_header: bool = True,
    venv_archive_uri: str | None = None,
) -> object:
    """Assemble the ``dataproc_v1.Job`` (a PySpark job placed on ``cluster``) (pure).

    Same launcher shim + package zip + driver args as the Serverless batch (`build_driver_args`), so
    on-cluster code and its contract (``--engine``/``--config-uri``/``--models``/
    ``--manage-header``) are identical between the batch and cluster surfaces.

    ``venv_archive_uri`` (packed-venv delivery) attaches the locked-dependency archive to the job so
    the cluster runs the exact same env as the container path: it's added to ``archive_uris`` with a
    ``#env`` fragment (Dataproc unpacks it to ``./env``) and the job's ``spark.pyspark.python`` /
    ``spark.pyspark.driver.python`` point at that interpreter. Without it the job runs the cluster's
    bare Python — which lacks the model libraries — so it's required for a cluster forecast job.
    """
    from google.cloud import dataproc_v1 as dataproc

    args = build_driver_args(
        config_uri, settings, engine=engine, models=models, manage_header=manage_header
    )
    archive_uris: list[str] = []
    properties: dict[str, str] = {}
    if venv_archive_uri:
        archive_uris.append(f"{venv_archive_uri}{_VENV_ARCHIVE_FRAGMENT}")
        properties.update(_VENV_JOB_PROPERTIES)
    return dataproc.Job(
        placement=dataproc.JobPlacement(cluster_name=cluster),
        pyspark_job=dataproc.PySparkJob(
            main_python_file_uri=launcher_uri,
            python_file_uris=[package_uri],
            args=args,
            archive_uris=archive_uris,
            properties=properties,
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


def _create_cluster(
    client: Any, project_id: str, region: str, cluster: object
) -> None:  # pragma: no cover - live Dataproc I/O, exercised by the @gcp smoke
    """Create the cluster and block until it's RUNNING (raises on a create error)."""
    op = client.create_cluster(
        request={"project_id": project_id, "region": region, "cluster": cluster}
    )
    op.result(timeout=_WAIT_TIMEOUT_SECONDS)


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
    engine: str = "explode",
    settings: Settings | None = None,
    infra: BatchInfra | None = None,
    models: list[str] | None = None,
    manage_header: bool = True,
    hardware: str = "cpu",
    gpu_type: str | None = None,
    spark_cluster_name: str | None = None,
    job_id: str | None = None,
    worker_count: int = _DEFAULT_WORKER_COUNT,
    wait: bool = True,
) -> str:
    """Stage code + config, run a PySpark job on a Dataproc cluster; return its job id.

    The cluster analog of `submit.submit_batch`. Stages the **full** ``cfg`` (so its ``run_id``
    matches `main.run`'s) and the shared launcher + package, then runs the lifecycle:

    * **ephemeral (default):** create a sized cluster → submit → (with ``wait``) poll to terminal →
      ``delete_cluster`` in a ``finally`` so teardown happens even if the job raises.
    * **reuse:** ``spark_cluster_name`` targets a standing cluster — skip create *and* delete.

    ``models``/``manage_header`` carry the on-cluster contract; ``hardware``/``gpu_type`` size the
    workers' accelerator (a cluster is the T4 Spark path). ``job_id`` is the deterministic
    per-family id (the orchestrator's `registry.ids.dataproc_job_id`), used only as a fallback: the
    returned id is Dataproc's own server-assigned ``reference.job_id`` (the console-resolvable one)
    when we waited. With ``wait`` a non-DONE terminal state raises so a failed job never exits 0.
    """
    from .registry.ids import make_run_id
    from .settings import Settings

    settings = settings or Settings.resolve()
    infra = infra or BatchInfra.resolve()
    run_id = make_run_id(cfg)
    name = cluster_name(run_id, spark_cluster_name)
    reuse = spark_cluster_name is not None
    venv_archive_uri = _resolve_cluster_deps(cfg, infra)

    package_uri, launcher_uri = _stage_code(infra)
    config_uri = _stage_config(cfg, run_id, infra)
    project_id = settings.project_id
    region = settings.region

    cluster_client = _cluster_client(region)
    job_client = _job_client(region)
    _log.info(
        "cluster submit: run_id=%s cluster=%s reuse=%s hardware=%s workers=%d",
        run_id,
        name,
        reuse,
        hardware,
        worker_count,
    )

    try:
        if not reuse:
            cluster = build_cluster(
                infra=infra,
                settings=settings,
                project_id=project_id,
                name=name,
                hardware=hardware,
                gpu_type=gpu_type,
                worker_count=worker_count,
            )
            _create_cluster(cluster_client, project_id, region, cluster)

        job = build_job(
            cluster=name,
            launcher_uri=launcher_uri,
            package_uri=package_uri,
            config_uri=config_uri,
            settings=settings,
            engine=engine,
            models=models,
            manage_header=manage_header,
            venv_archive_uri=venv_archive_uri,
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
        return final_id
    finally:
        if not reuse:
            _delete_cluster(cluster_client, project_id, region, name)


def provision_shared_cluster(
    cfg: RunConfig,
    *,
    run_id: str,
    use_gpu: bool,
    gpu_type: str | None = None,
    settings: Settings | None = None,
    infra: BatchInfra | None = None,
    worker_count: int = _DEFAULT_WORKER_COUNT,
) -> str:  # pragma: no cover - live Dataproc I/O, exercised by the @gcp smoke
    """Create one shared ephemeral Dataproc cluster for a run's cluster families; return its name.

    The multi-family analog of `submit_cluster_job`'s create step. When a run has more than one
    ephemeral cluster family the DAG orchestrator provisions **one** cluster here rather than let
    each family create its own: every family derives the same ``sf-cluster-<run_id>`` name and would
    each create *and* tear it down, so a family finishing first deletes the cluster out from under
    the others (``cluster is in state DELETING and cannot accept jobs``). Sized for the **union** of
    those families' hardware (GPU workers when ``use_gpu`` — any cluster family needs one; CPU
    families run on the same cluster, their jobs simply not using the GPU), mirroring the shared Ray
    cluster. The caller threads the name into every cluster family's `submit_cluster_job` as the
    ``spark_cluster_name`` reuse target (each submits its own failure-isolated job, no per-family
    create/delete) and tears it down once via `teardown_shared_cluster` after all families join.
    """
    from .settings import Settings

    settings = settings or Settings.resolve()
    infra = infra or BatchInfra.resolve()
    name = cluster_name(run_id, None)
    cluster = build_cluster(
        infra=infra,
        settings=settings,
        project_id=settings.project_id,
        name=name,
        hardware="gpu" if use_gpu else "cpu",
        gpu_type=gpu_type,
        worker_count=worker_count,
    )
    _log.info(
        "provisioning shared Dataproc cluster %s: use_gpu=%s workers=%d",
        name,
        use_gpu,
        worker_count,
    )
    _create_cluster(_cluster_client(settings.region), settings.project_id, settings.region, cluster)
    return name


def teardown_shared_cluster(
    name: str, settings: Settings | None = None
) -> None:  # pragma: no cover - live Dataproc I/O, exercised by the @gcp smoke
    """Delete the shared ephemeral Dataproc cluster (best-effort — a delete error is logged)."""
    from .settings import Settings

    settings = settings or Settings.resolve()
    _delete_cluster(_cluster_client(settings.region), settings.project_id, settings.region, name)
