"""Run a PySpark forecast job on a Dataproc **cluster**.

The cluster analog of `submit`: same on-cluster code and the same contract (the staged config whose
digest is the shared ``run_id``, the ``--models`` subset, contributor-mode header), but the job is
placed on a cluster rather than handed to Serverless. A cluster is also the only Spark surface that
offers T4 — Serverless is L4-only — which is why the path exists at all.

`submit_cluster_job` owns the *job's* lifecycle and borrows the cluster's:

* **ephemeral (default):** `dataproc_cluster` creates a sized cluster → submit the job → wait to
  terminal → delete in a ``finally``, so a crashed submit never leaks a billing cluster.
* **reuse (opt-in):** ``spark_cluster_name`` targets a standing cluster (or the run's shared one)
  by name — skip create *and* skip delete, directly analogous to Ray's ``ray_cluster_name``.

Its three neighbours: `dataproc_cluster` is the cluster itself (spec, sizing, create, delete),
`cluster_deps` is how the locked environment gets onto it, and `cluster_telemetry` is how anyone
looks at the job afterwards.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .batch_infra import BatchInfra
from .cluster_deps import _VENV_JOB_PROPERTIES, _resolve_cluster_deps, _stage_cluster_init
from .cluster_telemetry import _job_client, _stamp_cluster_telemetry
from .commands import build_driver_args
from .compute_fallback import resolve_candidates
from .dataproc_cluster import (
    _DEFAULT_WORKER_COUNT,
    _WAIT_TIMEOUT_SECONDS,
    _cluster_client,
    _create_cluster_across_candidates,
    _delete_cluster,
    cluster_name,
    cluster_sizing,
)
from .errors import EngineError, get_logger
from .staging import stage_code, stage_config

if TYPE_CHECKING:
    from .config import RunConfig
    from .settings import Settings

_log = get_logger(__name__)


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
    `dataproc_cluster.build_cluster`), so on-cluster code runs the exact same locked env as the
    container path. The
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
    from .profiling.source import profile_for_run
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

    package_uri, launcher_uri = stage_code(infra.code_bucket)
    config_uri = stage_config(cfg, run_id, infra.code_bucket)
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
                "machine_family": cfg.compute.machine_family,
            },
            policy=cfg.compute.capacity.policy_for("dataproc_cluster"),
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
