"""Orchestrate one run end-to-end — Spark + BigQuery-native in parallel, one run_id.

A single config can mix Python-runtime models (run
by the Spark engine as per-cell fan-out) and BigQuery-native models (run as SQL inside BigQuery);
`run` executes **both runtimes in parallel under one shared ``run_id`` and one
``run_registry`` header row**, so a native model and a Spark model land in the same run and are
directly comparable on ``v_model_leaderboard`` (the "wall-clock ≈ max(python, bq), not
sum" thesis).

Two invariants make the single-run parallelism work (see `ids`):

1. **One run_id from the full config.** ``make_run_id`` is a pure digest over the *whole* config
   incl. ``cfg.models``; both engines receive the full ``cfg`` so they derive the same id. Each is
   handed only its own executed subset (``python_models`` / ``bq_models``) via the ``models``
   argument, so the BigQuery-native models never become Spark cells (which would raise
   ``NotImplementedError`` in ``worker.run_cell``) and vice-versa.
2. **One header owner.** `run` alone writes the header (RUNNING) up front and finalizes it;
   both engines run in **contributor mode** (``manage_header=False``), skipping the header lifecycle
   and only writing their cell rows. No two writers ever touch the header, so there is no UPDATE
   race — the only in-window header write is ``submit_batch``'s best-effort telemetry stamp, which
   completes inside the joined future before `run`'s finalize.

**Parallelism.** The remote Spark batch is launched on a worker thread (``submit_batch(wait=True)``)
while the in-process BigQuery engine runs on the main thread; the BQ work (minutes, in-process)
overlaps the Spark provisioning floor. `run` joins both, rolls the two outcomes into one
combined status (COMPLETED iff both green, else FAILED — finalized *before* re-raising so the run
stays queryable and the CLI exits non-zero), and returns the shared ``run_id``.

**Coarsening (documented).** A remote contributor batch can't return its run-level PARTIAL (some
cells errored) to the orchestrator, so a SUCCEEDED batch is reported COMPLETED; per-model failure
stays visible on ``v_model_leaderboard`` (a failed model → NULL metric AVGs).

Both Python runtimes are supported and dispatched by ``cfg.python_runtime``: ``"spark"`` launches a
Dataproc Serverless batch (`submit_batch`), ``"ray"`` an autoscaling
Vertex Ray cluster (`submit_ray`) — either way on the worker
thread, in contributor mode, in parallel with the in-process BigQuery engine under one run_id.

Public surface: ``run(cfg, *, dry_run=False) -> run_id``, the offline ``plan_run(cfg) ->
LaunchPlan`` (id + fanout + launch-command templates), ``stage_run(cfg) -> LaunchPlan`` (upload
artifacts + runnable commands + reproducibility manifest, no submit), and
``python -m scale_forecasting.main (--config … | --config-uri gs://…) [--dry-run | --stage-only]``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .errors import ConfigError, get_logger
from .registry.ids import make_run_id
from .router import split_by_runtime

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from .commands import LaunchCommands
    from .config import Fanout, RunConfig
    from .dag import DagNode, FamilyJob, RunDag
    from .probes import ProbeReport
    from .settings import Settings

_log = get_logger(__name__)


@dataclass(frozen=True)
class Idempotency:
    """Whether a run's config has already been submitted (the pre-submit existence check).

    ``run_id`` is a pure digest of the config, so an unchanged config re-derives the same id — this
    verdict says whether that id already has a header in the registry. ``checked`` is ``False`` when
    the registry couldn't be consulted (no ``SF_*`` env, or no reachable registry): the verdict is
    simply *unknown*, not "new". ``prior_status`` is the existing run's status when ``exists`` (e.g.
    ``COMPLETED``), else ``None``.
    """

    checked: bool
    exists: bool
    prior_status: str | None


@dataclass(frozen=True)
class LaunchPlan:
    """A run resolved to launch-ready form: id, runtime split, fanout, and the launch commands.

    Produced offline by `plan_run` (``staged=False`` — URIs are the *templates* where artifacts will
    land; no GCS is touched) or by `stage_run` (``staged=True`` — artifacts uploaded, URIs real and
    the commands runnable). ``commands`` maps a tier name (``"main"``/``"spark"``/``"ray"``) to its
    `LaunchCommands`; it is ``None`` only when the GCP infra identity can't be resolved offline (no
    ``SF_*`` env), in which case the run_id + fanout + runtime split are still returned.

    ``idempotency`` is the exists-vs-new verdict for this config's id; ``force`` records that the
    caller intends to re-run an already-run config (it only shapes the emitted guidance — a re-run
    is idempotent regardless, via dedupe-on-read under the shared ``run_id``).

    ``nodes`` is the run's execution DAG resolved offline (`dag.dag_nodes`): one node per family job
    plus the ensemble, each with its deterministic ``job_key`` and upstream dependencies — the
    plan's cross-system identity map, before anything runs.
    """

    run_id: str
    python_runtime: str
    python_models: list[str]
    bq_models: list[str]
    fanout: Fanout
    staged: bool
    config_uri: str | None
    commands: dict[str, LaunchCommands] | None
    idempotency: Idempotency
    force: bool
    nodes: tuple[DagNode, ...]


@dataclass(frozen=True)
class _RunPlan:
    """The offline-resolvable shape of a run: its id and the per-runtime executed subsets.

    Pure product of the config (`_plan`) — no GCP.
    """

    run_id: str
    python_models: list[str]
    bq_models: list[str]


def _plan(cfg: RunConfig) -> _RunPlan:
    """Resolve the run_id + per-runtime model split (pure and offline).

    Computes ``make_run_id(cfg)`` and `router.split_by_runtime`. Both ``python_runtime="spark"``
    and ``python_runtime="ray"`` are supported (dispatched in `run`); an all-BigQuery config plans
    and runs regardless of ``python_runtime``.
    """
    run_id = make_run_id(cfg)
    python_models, bq_models = split_by_runtime(cfg)
    return _RunPlan(
        run_id=run_id,
        python_models=python_models,
        bq_models=bq_models,
    )


def _system_job_id(job_key: str, runtime: str) -> str:
    """Map a family's canonical ``job_key`` to its runtime's platform-legal job id (pure).

    Each platform stamps the ``job_key`` as its *own* job id so the platform job and the
    ``run_jobs`` row share an identity, but platforms differ on the legal charset/length — so the
    key is mapped to a legal form per runtime (`registry.ids`): a Dataproc batch/job id for
    ``spark``, a Ray ``submission_id`` for ``ray``, a BigQuery parent job id for ``bigquery``. The
    canonical key stays in ``run_jobs.job_id`` (this mapped id lands in ``system_job_id``), so a
    trace never reverses a lossy mapping.
    """
    from .registry.ids import bigquery_job_id, dataproc_job_id, ray_submission_id

    if runtime == "spark":
        return dataproc_job_id(job_key)
    if runtime == "ray":
        return ray_submission_id(job_key)
    return bigquery_job_id(job_key)


def _shared_ray_inputs(python_jobs: list[FamilyJob]) -> tuple[list[str], bool, str | None] | None:
    """The union sizing inputs for a run's ephemeral Ray families, or ``None`` if fewer than two.

    Sharing one cluster only matters when **more than one** family resolves to Ray — that's the case
    that would otherwise collide on the run-derived ``sf-ray-<run_id>`` name (and waste a second
    cluster). Returns the union of those families' models, whether **any** needs a GPU pool, and the
    GPU type to size it (the first GPU family's) — the inputs to one shared cluster. A single Ray
    family (or none) returns ``None`` and keeps the proven self-provisioning path.
    """
    ray_jobs = [j for j in python_jobs if j.runtime == "ray"]
    if len(ray_jobs) < 2:
        return None
    models: list[str] = []
    any_gpu = False
    gpu_type: str | None = None
    for j in ray_jobs:
        assert j.compute is not None  # a Python family always resolves compute
        models.extend(j.models)
        if j.compute.hardware == "gpu":
            any_gpu = True
            gpu_type = gpu_type or j.compute.gpu_type
    return models, any_gpu, gpu_type


@contextmanager
def _shared_ray_cluster(
    cfg: RunConfig, run_dag: RunDag, run_id: str, settings: Settings
) -> Iterator[tuple[str, str] | None]:
    """Provision one shared ephemeral Ray cluster when a run has several ephemeral Ray families.

    Yields ``(cluster_name, cluster_region)`` to thread into each Ray family's job as a reuse target
    (so each submits its own failure-isolated Ray job to the one cluster), or ``None`` when sharing
    doesn't apply — a single Ray family, no Ray family, or a config already reusing a standing
    cluster (``compute.ray_cluster_name`` set: every family targets it, no orchestrator create). The
    cluster is torn down once in a ``finally`` so a family failure never leaks it.
    """
    inputs = None
    if cfg.compute.ray_cluster_name is None:
        inputs = _shared_ray_inputs(run_dag.python_jobs)
    if inputs is None:
        yield None
        return
    from . import ray_submit

    models, any_gpu, gpu_type = inputs
    name, region = ray_submit.provision_shared_cluster(
        cfg, models=models, run_id=run_id, use_gpu=any_gpu, gpu_type=gpu_type, settings=settings
    )
    try:
        yield (name, region)
    finally:
        ray_submit.teardown_shared_cluster(name, region, settings)


def _shared_spark_inputs(python_jobs: list[FamilyJob]) -> tuple[bool, str | None] | None:
    """The union sizing inputs for a run's ephemeral Dataproc cluster families, or ``None`` if fewer
    than two.

    Sharing one cluster only matters when **more than one** family resolves to an ephemeral Spark
    cluster (``spark_mode="cluster"`` with no standing ``spark_cluster_name``) — the case that would
    otherwise have each family both create *and* tear down the shared run-derived
    ``sf-cluster-<run_id>`` name, so a family finishing first deletes the cluster out from under the
    others. Returns whether **any** of them needs a GPU pool and the GPU type to size it (the first
    GPU family's) — the inputs to one shared cluster. Fewer than two ephemeral cluster families (or
    none) returns ``None`` and keeps the proven per-family lifecycle (a single family has no
    collision risk; a family naming a standing cluster already reuses).
    """
    cluster_jobs = [
        j
        for j in python_jobs
        if j.runtime == "spark"
        and j.compute is not None
        and j.compute.spark_mode == "cluster"
        and j.compute.spark_cluster_name is None
    ]
    if len(cluster_jobs) < 2:
        return None
    any_gpu = False
    gpu_type: str | None = None
    for j in cluster_jobs:
        assert j.compute is not None  # a Python family always resolves compute
        if j.compute.hardware == "gpu":
            any_gpu = True
            gpu_type = gpu_type or j.compute.gpu_type
    return any_gpu, gpu_type


@contextmanager
def _shared_spark_cluster(
    cfg: RunConfig, run_dag: RunDag, run_id: str, settings: Settings
) -> Iterator[tuple[str, str] | None]:
    """Provision one shared ephemeral Dataproc cluster when a run has several ephemeral cluster
    families.

    Yields ``(cluster_name, cluster_region)`` to thread into each cluster family's job as a reuse
    target (so each submits its own failure-isolated job to the one cluster, skipping the per-family
    create/delete that would otherwise race — a family finishing first would tear down the shared
    cluster out from under the others), or ``None`` when sharing doesn't apply — fewer than two
    cluster families. The region is returned because a capacity failover may have moved the cluster
    off the deployment region, and each family's job must submit to where it actually landed. The
    cluster is torn down once in a ``finally`` so a family failure never leaks it. The Dataproc
    analog of `_shared_ray_cluster`.
    """
    inputs = _shared_spark_inputs(run_dag.python_jobs)
    if inputs is None:
        yield None
        return
    from .dataproc_cluster import provision_shared_cluster, teardown_shared_cluster

    any_gpu, gpu_type = inputs
    name, region = provision_shared_cluster(
        cfg, run_id=run_id, use_gpu=any_gpu, gpu_type=gpu_type, settings=settings
    )
    try:
        yield (name, region)
    finally:
        teardown_shared_cluster(name, region, settings)


def _launch_family_job(
    cfg: RunConfig,
    job: FamilyJob,
    run_id: str,
    settings: Settings,
    spark: object | None = None,
    *,
    force: bool = False,
    max_executors: int | None = None,
    ray_cluster: tuple[str, str] | None = None,
    spark_cluster: tuple[str, str] | None = None,
) -> None:
    """Run one Python family's job on its resolved runtime, wrapped in its ``run_jobs`` row.

    Called on a worker thread — one per Python family (statistical / ml / deep_learning), so the
    families run in parallel under one shared header. Resolves this family's attempt
    (`registry.bq.next_job_attempt`, bumped by ``--force``), opens the per-job lifecycle
    (`registry.bq.run_job`, which writes the row RUNNING and finalizes its terminal status +
    wall-clock), then dispatches to the `RuntimeSubmitter` for the family's **resolved** runtime
    (`get_submitter` on ``job.compute.runtime`` — Spark *xor* Ray, chosen per family, not per run)
    with ``manage_header=False`` (this orchestrator owns the single shared header). The submitter
    blocks until terminal, so the caller joins one future per family.

    ``spark`` is an optional injected `SparkSession`, passed through to the submitter: the Spark
    submitter, given one, runs **in-process against that session** (the injectable-session seam)
    instead of submitting a remote batch; other runtimes ignore it. ``max_executors`` caps the
    remote Spark batch's dynamic-allocation ceiling (ignored by the in-process and Ray paths). Kept
    a plain module function (not a lambda) so a worker thread's traceback names it, and the
    submitter's imports stay lazy (Ray/Spark extras load only for the chosen path).

    The family's deterministic ``job_key`` (`registry.ids.make_job_key`) is mapped to its runtime's
    platform-legal id (`_system_job_id`) and threaded both onto the ``run_jobs`` row and into the
    submitter — so several families under one shared ``run_id`` submit distinct, traceable platform
    jobs (a Dataproc ``batch_id`` / Ray ``submission_id``) instead of colliding on a run-derived id.
    A Dataproc *cluster* job id is server-assigned rather than accepted from us, so when the
    submitter returns the real id it is stamped back onto the row (`JobFinalizer`) for the trace.

    ``ray_cluster``, when set, is the run's shared ephemeral Ray cluster ``(name, region)``
    (`_shared_ray_cluster`): a Ray family reuses it instead of self-provisioning; every other
    runtime ignores it. ``spark_cluster``, when set, is the run's shared ephemeral Dataproc cluster
    ``(name, region)`` (`_shared_spark_cluster`): an ephemeral cluster family reuses it (submits its
    own job to that region — a capacity failover may have moved the cluster off the deployment
    region — with no per-family create/delete); a family naming its own standing cluster keeps that,
    and every other runtime/mode ignores it.
    """
    from .probes import ProbeHandle
    from .registry import bq
    from .registry.ids import make_job_key
    from .submitters import get_submitter

    compute = job.compute
    assert compute is not None  # a Python family always resolves compute (native is handled inline)
    attempt, _ = bq.next_job_attempt(run_id, job.family, force=force, settings=settings)
    system_job_id = _system_job_id(make_job_key(run_id, job.family, attempt), compute.runtime)
    # A shared Ray cluster (provisioned by the orchestrator for a multi-Ray-family run) is targeted
    # only by Ray families; every other runtime ignores it.
    ray_cluster_name = ray_cluster[0] if ray_cluster and compute.runtime == "ray" else None
    ray_cluster_region = ray_cluster[1] if ray_cluster and compute.runtime == "ray" else None
    # The shared Dataproc cluster is targeted only by ephemeral Spark cluster families (spark_mode
    # cluster, no standing cluster of their own) as a reuse target; a family naming its own standing
    # cluster keeps it, and serverless/Ray families ignore it.
    is_shared_spark_family = (
        spark_cluster is not None
        and compute.runtime == "spark"
        and compute.spark_mode == "cluster"
        and compute.spark_cluster_name is None
    )
    shared_spark_name = spark_cluster[0] if is_shared_spark_family and spark_cluster else None
    shared_spark_region = spark_cluster[1] if is_shared_spark_family and spark_cluster else None
    # The ENTRY probe handle, built from coordinates known before submit — the handle the probe
    # actually reads while a job is RUNNING. It never asserts an id it doesn't truly have (a cluster
    # job's real id is server-assigned, so native_id is empty until the stamp-back refresh below),
    # so a probe degrades to registry-only rather than emitting a false NOT_FOUND.
    if compute.runtime == "ray":
        resource_name = None
        if ray_cluster_name is not None:
            from .ray_submit import _resource_name

            resource_name = _resource_name(settings, ray_cluster_name, ray_cluster_region)
        entry_handle = ProbeHandle(
            "ray",
            native_id=system_job_id,
            region=ray_cluster_region or settings.region,
            resource_name=resource_name,
        )
    elif compute.spark_mode == "cluster":
        entry_handle = ProbeHandle(
            "spark",
            native_id="",  # real id is server-assigned; filled in at the stamp-back refresh
            region=shared_spark_region or settings.region,
            spark_mode="cluster",
        )
    else:
        entry_handle = ProbeHandle(
            "spark",
            native_id=system_job_id,
            region=settings.region,
            spark_mode="serverless",
        )
    with bq.run_job(
        run_id,
        job.family,
        attempt,
        runtime=compute.runtime,
        spark_mode=compute.spark_mode,
        hardware=compute.hardware,
        gpu_type=compute.gpu_type,
        system_job_id=system_job_id,
        probe_handle=entry_handle.to_blob(),
        settings=settings,
    ) as fin:
        handle = get_submitter(compute.runtime).launch(
            cfg,
            models=list(job.models),
            manage_header=False,
            settings=settings,
            spark=spark,
            max_executors=max_executors,
            system_job_id=system_job_id,
            hardware=compute.hardware,
            gpu_type=compute.gpu_type,
            spark_mode=compute.spark_mode,
            spark_cluster_name=compute.spark_cluster_name or shared_spark_name,
            spark_cluster_region=shared_spark_region,
            ray_cluster_name=ray_cluster_name,
            ray_cluster_region=ray_cluster_region,
        )
        # Stamp-back refresh: replace the entry handle with post-submit truths (a cluster's real id,
        # the landed region + Ray resource path). A cluster job's id is server-assigned, so when the
        # returned native_id differs from system_job_id, also stamp the real id back for
        # reverse-trace. The in-process session submits nothing and returns None (no refresh).
        if handle is not None:
            fields: dict[str, Any] = {"job_telemetry": {"probe_handle": handle.to_blob()}}
            if handle.native_id != system_job_id:
                fields["system_job_id"] = handle.native_id
            fin.finalize(**fields)


def _launch_native_job(
    cfg: RunConfig,
    job: FamilyJob,
    run_id: str,
    settings: Settings,
    *,
    force: bool = False,
) -> object:
    """Run the BigQuery-native family inline (main thread), wrapped in its ``run_jobs`` row.

    Native models execute as SQL in BigQuery — no Python runtime, no worker thread — so this runs on
    `run`'s main thread, overlapping the Python family jobs. Like `_launch_family_job` it resolves
    the ``native`` attempt, maps the deterministic ``job_key`` to the BigQuery job id
    (`_system_job_id`), and opens the per-job lifecycle (`registry.bq.run_job`, ``runtime`` fixed to
    ``"bigquery"``, carrying that id), then runs the engine in contributor mode. Returns the
    engine's `BqOutcome` so the caller can stamp the observed ``n_series`` onto the header.
    """
    from .engines import bigquery_engine
    from .probes import ProbeHandle
    from .registry import bq
    from .registry.ids import make_job_key

    attempt, _ = bq.next_job_attempt(run_id, "native", force=force, settings=settings)
    system_job_id = _system_job_id(make_job_key(run_id, "native", attempt), "bigquery")
    # BigQuery coordinates are fully known up front (jobs share the deterministic id prefix), so the
    # entry handle is the only one — there is no stamp-back site for the native family.
    native_handle = ProbeHandle(
        "bigquery",
        native_id=f"{system_job_id}-",
        region=settings.region,
        id_kind="prefix",
    )
    with bq.run_job(
        run_id,
        "native",
        attempt,
        runtime="bigquery",
        system_job_id=system_job_id,
        probe_handle=native_handle.to_blob(),
        settings=settings,
    ):
        # Prefix the family's BigQuery jobs with its deterministic id so they resolve in the console
        # under system_job_id (reverse-trace); a BQ job id is server-assigned, not client-set.
        return bigquery_engine.run(
            cfg,
            list(job.models),
            manage_header=False,
            settings=settings,
            job_id_prefix=f"{system_job_id}-",
        )


def _launch_ensemble_job(
    cfg: RunConfig,
    run_id: str,
    settings: Settings,
    *,
    force: bool = False,
    upstream_done: Callable[[], bool] | None = None,
) -> None:
    """Run the ensemble DAG node inline (driver), wrapped in its own ``run_jobs`` row.

    The ensemble is the run's final DAG node: it blends every family's base predictions into the
    consensus pseudo-models and scores them onto the leaderboard (`ensemble_run`). Like the family
    jobs it gets a deterministic identity — ``job_key`` (`registry.ids.make_job_key` on the
    ``"ensemble"`` family) mapped to its BigQuery job id (`_system_job_id`) — and its own
    ``run_jobs`` row (`registry.bq.run_job`, ``runtime="bigquery"``: the node reads/writes BigQuery
    and blends in driver pandas, taking no Spark/Ray cluster), so a run's cross-system trace shows
    the ensemble beside the base jobs under the shared ``run_id``.

    ``cfg.compute.ensemble.mode`` selects *when* the consensus is computed: ``"barrier"`` (default)
    blends once over every base prediction; ``"microbatch"`` drains series incrementally as each
    one's full base set lands (`ensemble_run.run_ensembles_microbatch`), polling every
    ``cfg.compute.ensemble.microbatch_interval_s``. Both execute on the driver and land the same
    rows; only the batching differs.

    ``upstream_done`` (microbatch only) lets the caller run this node **concurrently** with the base
    jobs: the drain loop keeps polling for ready series until ``upstream_done()`` reports the base
    jobs finished *and* no ready series remain. ``None`` (the default) means "already done" — the
    post-join trigger, draining every ready series in a single pass. A series is *ready* only once
    **every** configured base model has landed for it, so a failed family leaves no series ready and
    the concurrent node produces nothing — preserving the "no ensembles for a failed run" contract.
    """
    from .ensemble_run import run_ensembles, run_ensembles_microbatch
    from .probes import ProbeHandle
    from .registry import bq
    from .registry.ids import make_job_key

    attempt, _ = bq.next_job_attempt(run_id, "ensemble", force=force, settings=settings)
    system_job_id = _system_job_id(make_job_key(run_id, "ensemble", attempt), "bigquery")
    # BigQuery coordinates are fully known up front (jobs share the deterministic id prefix), so the
    # entry handle is the only one — there is no stamp-back site for the ensemble node.
    ensemble_handle = ProbeHandle(
        "bigquery",
        native_id=f"{system_job_id}-",
        region=settings.region,
        id_kind="prefix",
    )
    with bq.run_job(
        run_id,
        "ensemble",
        attempt,
        runtime="bigquery",
        system_job_id=system_job_id,
        probe_handle=ensemble_handle.to_blob(),
        settings=settings,
    ):
        # Prefix the ensemble's BigQuery jobs with its deterministic id so they resolve in the
        # console under system_job_id (reverse-trace), mirroring the native family.
        prefix = f"{system_job_id}-"
        if cfg.compute.ensemble.mode == "microbatch":
            run_ensembles_microbatch(
                cfg,
                run_id,
                settings=settings,
                poll_interval_s=cfg.compute.ensemble.microbatch_interval_s,
                upstream_done=upstream_done,
                job_id_prefix=prefix,
            )
        else:
            run_ensembles(cfg, run_id, settings=settings, job_id_prefix=prefix)


def run(
    cfg: RunConfig,
    *,
    dry_run: bool = False,
    spark: object | None = None,
    settings: Settings | None = None,
    force: bool = False,
    n_series: int | None = None,
    max_executors: int | None = None,
) -> str:
    """Execute one run as a DAG: every model family in parallel under one run_id; return that id.

    Plans the run's DAG (`dag.plan_dag`) — one job per model family present in the config
    (statistical / ml / deep_learning each on its resolved runtime, native in BigQuery) plus the
    downstream ensemble node — then:

    * ``dry_run=True`` → delegate to `plan_run` (the offline "what would this schedule" path):
      resolve the run_id + `estimate_fanout` + emit the launch-command templates, touching no GCP,
      and return the run_id.
    * otherwise → resolve `Settings`, ``ensure_tables`` + ``write_header`` (RUNNING, the one shared
      header), launch each Python family's job on its own worker thread (`_launch_family_job`) and
      run the BigQuery-native family inline (`_launch_native_job`) so they all overlap — each in
      contributor mode with its own ``run_jobs`` row. Once every family job joins and *all* of them
      succeeded, and ``cfg.ensemble.enabled``, run the ensemble DAG node (`_launch_ensemble_job`,
      which reads the just-written base predictions/OOF and scores each consensus onto the
      leaderboard under its own ``run_jobs`` row), then finalize the header with the combined status
      + wall-clock
      ``runtime_seconds`` + ``bq_models`` (+ the native engine's observed ``n_series`` when it ran).
      On any family *or* ensemble failure the header is finalized FAILED/PARTIAL before the first
      error re-raises, so the run stays queryable and the CLI exits non-zero.

    ``spark`` is an optional injected `SparkSession` (incl. a Spark Connect
    ``DataprocSparkSession``). When supplied, a Spark-runtime family runs **in-process against that
    session** instead of a remote Dataproc batch — the notebook / Connect demo path — using the
    identical engine code (the injectable-session seam). The default (``None``) keeps the
    remote-batch behavior every CLI/Composer caller relies on.

    ``settings`` optionally injects a resolved `Settings` (the GCP infra identity); the default
    (``None``) resolves it from the ``SF_*`` environment. The SDK ``Forecaster`` uses this to thread
    an explicit identity instead of relying on process env. ``force`` bumps each family's
    ``run_jobs`` attempt (a fresh, distinctly-keyed job under the same run_id) and shapes the
    ``dry_run`` plan's re-run guidance.

    ``n_series`` overrides ``data.series_limit`` (the 10→100→1k→100k scale knob) before anything
    else, so it changes the ``run_id`` and every family sees the same limit. ``max_executors`` caps
    a remote Spark batch's dynamic-allocation ceiling (ignored by the in-process/Ray paths).

    Idempotent: the config-pinned run_id + append-only/dedupe-on-read cell writes mean re-running
    the same config lands byte-identical rows.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from .dag import plan_dag
    from .registry import bq
    from .settings import Settings

    # The series-limit override is applied first so it flows into the run_id and every family — a
    # different scale is a distinct, independently-queryable run.
    cfg = cfg.with_series_limit(n_series)

    if dry_run:
        # Single-source the offline plan: plan_run resolves the id + fanout + runtime split, reports
        # the exists-vs-new verdict, and emits the launch-command templates. It returns the run_id,
        # so this contract is unchanged.
        return plan_run(cfg, settings=settings, force=force).run_id

    run_dag = plan_dag(cfg)
    run_id = run_dag.run_id

    settings = settings or Settings.resolve()

    # Idempotency guard: the run_id is config-pinned, so re-running the same config is a no-op once
    # it has COMPLETED — return without relaunching. Relaunching would resubmit each family's
    # deterministic platform job id (a Dataproc batch_id / Ray submission_id) and collide
    # (AlreadyExists), and the reused-attempt terminal write would clobber the completed run's job
    # rows. ``force`` re-executes as a fresh attempt (distinct job ids); a run that never completed
    # (never ran, or a prior FAILED/PARTIAL) falls through and runs.
    if not force and bq.header_status(run_id, settings=settings) == "COMPLETED":
        _log.info("run %s already COMPLETED; skipping relaunch (pass force=True to re-run)", run_id)
        return run_id

    _log.info(
        "run %s start: families=%s ensemble=%s",
        run_id,
        run_dag.families,
        run_dag.ensemble_enabled,
    )

    bq_outcome = None
    # One error slot per family job (keyed by family name), plus the ensemble node's.
    job_errors: dict[str, BaseException] = {}
    ensemble_error: BaseException | None = None
    native = run_dag.native_job
    python_jobs = run_dag.python_jobs

    # Microbatch ensemble overlaps the base jobs: it drains series as each one's full base set lands
    # (rather than waiting for the join like the barrier), so it runs as a concurrent pool task with
    # an ``upstream_done`` predicate the join flips. ``base_done`` is set once every family (Python
    # + native) has joined, telling the drain loop no further base predictions will land. A family
    # that fails leaves its models absent, so no series ever reaches the full-base-set readiness bar
    # and the concurrent node drains nothing — same "no ensembles when a base family failed" outcome
    # as the barrier's post-join skip, just reached by readiness rather than an up-front gate.
    ensemble_concurrent = run_dag.ensemble_enabled and cfg.compute.ensemble.mode == "microbatch"
    base_done = threading.Event()

    # One header owner: run_header writes RUNNING on entry and finalizes once, after every family
    # job joins, with the combined status computed below. Every job runs with manage_header=False so
    # nothing else touches this row. Per-family errors are captured (not raised through the block)
    # so the finalize records the right status; the first is re-raised after, for a non-zero exit.
    with bq.run_header(cfg, run_id, settings=settings, manage=True) as hdr:
        # Launch each Python family on its own worker thread, and run the BigQuery-native family
        # inline on the main thread, so all families overlap. Each family carries the same
        # contributor-mode contract (its model subset + shared header owned here) and its own
        # run_jobs row, so N heterogeneous families run under one run_id.
        # When the run has several ephemeral Ray (or Dataproc-cluster) families, provision one
        # shared cluster per runtime for the duration of the launch block (each family submits its
        # own job to it, torn down once on exit); otherwise these yield None and each family
        # self-provisions as before.
        # +1 pool worker for the concurrent microbatch ensemble so it never contends with a family
        # for a thread; barrier mode keeps the exact family-only pool it always had.
        max_workers = max(1, len(python_jobs)) + (1 if ensemble_concurrent else 0)
        with (
            _shared_ray_cluster(cfg, run_dag, run_id, settings) as ray_cluster,
            _shared_spark_cluster(cfg, run_dag, run_id, settings) as spark_cluster,
            ThreadPoolExecutor(max_workers=max_workers) as pool,
        ):
            futures = {
                pool.submit(
                    _launch_family_job,
                    cfg,
                    job,
                    run_id,
                    settings,
                    spark,
                    force=force,
                    max_executors=max_executors,
                    ray_cluster=ray_cluster,
                    spark_cluster=spark_cluster,
                ): job
                for job in python_jobs
            }
            # Microbatch: fire the ensemble now, concurrently with the base jobs, draining ready
            # series until the join flips base_done. Barrier: it stays a post-join step (below).
            ensemble_future = (
                pool.submit(
                    _launch_ensemble_job,
                    cfg,
                    run_id,
                    settings,
                    force=force,
                    upstream_done=base_done.is_set,
                )
                if ensemble_concurrent
                else None
            )
            if native is not None:
                try:
                    bq_outcome = _launch_native_job(cfg, native, run_id, settings, force=force)
                except Exception as exc:  # noqa: BLE001 - captured, finalized below, re-raised
                    job_errors["native"] = exc
            for future, job in futures.items():
                try:
                    future.result()
                except Exception as exc:  # noqa: BLE001 - captured, finalized below, re-raised
                    job_errors[job.family] = exc
            # Every base family has joined: no more base predictions will land, so the concurrent
            # drain loop can stop after its final ready-series pass. Set before the ensemble join.
            base_done.set()
            if ensemble_future is not None:
                try:
                    ensemble_future.result()
                except Exception as exc:  # noqa: BLE001 - captured, finalized below, re-raised
                    ensemble_error = exc

        # Barrier ensemble: it reads every family's base predictions / backtest_oof, so it runs
        # strictly after the join and only when every family succeeded. (Microbatch already ran
        # concurrently above.) A failure here is captured like a family error — the ensembles are
        # part of the run's success contract.
        if not ensemble_concurrent and not job_errors and run_dag.ensemble_enabled:
            try:
                _launch_ensemble_job(cfg, run_id, settings, force=force)
            except Exception as exc:  # noqa: BLE001 - captured, finalized below, re-raised
                ensemble_error = exc

        # Combined status across the family jobs that ran: all green → COMPLETED, all failed →
        # FAILED, some but not all → PARTIAL (surviving families' forecasts stay usable). An
        # ensemble failure on top of all-green families fails the run: full output undelivered.
        status = _combined_status(run_dag, job_errors, ensemble_error)
        fields: dict[str, object] = {"bq_models": list(native.models) if native else []}
        if bq_outcome is not None:
            fields["n_series"] = bq_outcome.n_series
        hdr.finalize(status=status, **fields)

    first_error = next(iter(job_errors.values()), None) or ensemble_error
    if first_error is not None:
        # Re-raise the first failure so the CLI exits non-zero; the header already records the
        # combined status (FAILED or PARTIAL).
        raise first_error
    _log.info("run %s done: status=%s", run_id, status)
    return run_id


def _combined_status(
    run_dag: RunDag,
    job_errors: dict[str, BaseException],
    ensemble_error: BaseException | None,
) -> str:
    """Roll the per-family job outcomes into one run status (pure).

    Over the family jobs that ran (one per family in the DAG): every job green → ``COMPLETED``;
    every job failed → ``FAILED``; a mix → ``PARTIAL``. An ensemble failure downgrades an
    otherwise-``COMPLETED`` run to ``FAILED`` (the requested output is incomplete); it never masks a
    family ``PARTIAL``/``FAILED``.
    """
    n_jobs = len(run_dag.jobs)
    n_failed = len(job_errors)
    if n_failed == 0:
        engine_status = "COMPLETED"
    elif n_failed == n_jobs:
        engine_status = "FAILED"
    else:
        engine_status = "PARTIAL"

    if engine_status == "COMPLETED" and ensemble_error is not None:
        return "FAILED"
    return engine_status


def _check_idempotency(run_id: str, settings: Settings) -> Idempotency:
    """Best-effort pre-submit existence check: has this exact config already run?

    Queries the registry for ``run_id`` (`registry.bq.header_status`). Any failure — no registry
    table yet, no reachable BigQuery — degrades to ``checked=False`` (unknown), so a plain offline
    dry-run still returns a plan. The check is advisory: it warns before an accidental duplicate run
    but never blocks one (a re-run is idempotent via dedupe-on-read).
    """
    from .registry import bq

    try:
        status = bq.header_status(run_id, settings=settings)
    except Exception:  # noqa: BLE001 - advisory check; unknown on any failure, never fatal
        return Idempotency(checked=False, exists=False, prior_status=None)
    return Idempotency(checked=True, exists=status is not None, prior_status=status)


def _resolve_infra(cfg: RunConfig, infra: object | None) -> object:
    """The runtime's infra identity: the injected ``infra``, else resolved from the ``SF_*`` env.

    Spark resolves a `BatchInfra`, Ray a `RayInfra`; both carry the ``code_bucket`` the config
    stages to. Raises `ConfigError` (from ``resolve``) when the env is unset — plan emission is
    best-effort, so `plan_run` catches that and returns a plan without commands.
    """
    if infra is not None:
        return infra
    if cfg.python_runtime == "ray":
        from .ray_submit import RayInfra

        return RayInfra.resolve()
    from .submit import BatchInfra

    return BatchInfra.resolve()


def _template_uris(
    cfg: RunConfig, plan: _RunPlan, code_bucket: str
) -> tuple[str, str | None, str | None]:
    """The ``gs://`` URIs a run's artifacts *will* land at (pure — mirrors the staging scheme).

    Returns ``(config_uri, package_uri, launcher_uri)``. The config URI always exists; the Spark
    package/launcher URIs are set only for a Spark run with Python models (Ray delivers code via its
    ``runtime_env`` working dir, not a staged zip). The package name carries the code hash from
    `code_delivery.build_package_zip` — a deterministic local build, no network — so the template is
    byte-faithful to what `submit._stage_code` would upload.
    """
    config_uri = f"gs://{code_bucket}/runs/{plan.run_id}.json"
    if cfg.python_runtime == "ray" or not plan.python_models:
        return config_uri, None, None
    from .code_delivery import build_package_zip

    _, code_hash = build_package_zip()
    package_uri = f"gs://{code_bucket}/runs/scale_forecasting-{code_hash}.zip"
    launcher_uri = f"gs://{code_bucket}/runs/spark_main.py"
    return config_uri, package_uri, launcher_uri


def _assemble_commands(
    cfg: RunConfig,
    plan: _RunPlan,
    settings: Settings,
    infra: object,
    *,
    config_uri: str,
    package_uri: str | None,
    launcher_uri: str | None,
) -> dict[str, LaunchCommands]:
    """Build the launch commands for a run, keyed by tier (pure — assembles strings only).

    Always emits ``"main"`` — ``python -m scale_forecasting.main --config-uri …``, the orchestrator
    that reproduces the *full* run (both engines under one run_id). When there are Python-runtime
    models it adds the per-runtime tier: ``"ray"`` (universal only) or ``"spark"`` (native
    ``gcloud`` + universal). The Spark tier restricts to ``--models`` **only** for a mixed run (so
    the batch runs just its subset while BigQuery runs the rest); a Python-only config emits no
    ``--models`` so the standalone batch runs the whole config under its own header.
    """
    from .commands import build_main_command, build_ray_commands, build_spark_commands

    commands: dict[str, LaunchCommands] = {"main": build_main_command(config_uri)}
    if not plan.python_models:
        return commands
    if cfg.python_runtime == "ray":
        commands["ray"] = build_ray_commands(
            config_uri=config_uri, cluster_name=cfg.compute.ray_cluster_name
        )
        return commands

    from .submit import BatchInfra, _batch_id

    assert isinstance(infra, BatchInfra)  # spark runtime → BatchInfra (resolved above)
    models_arg = plan.python_models if plan.bq_models else None
    commands["spark"] = build_spark_commands(
        settings=settings,
        infra=infra,
        batch_id=_batch_id(plan.run_id),
        package_uri=package_uri or "",
        launcher_uri=launcher_uri or "",
        config_uri=config_uri,
        models=models_arg,
        manage_header=True,
    )
    return commands


def _emit_idempotency(result: LaunchPlan) -> None:
    """Log the exists-vs-new verdict and, when re-running, the ``--force`` guidance."""
    idem = result.idempotency
    if not idem.checked:
        return  # registry not consulted (offline / unreachable) — verdict unknown, say nothing
    if not idem.exists:
        _log.info("  new run — this config has not run before")
        return
    if result.force:
        _log.info(
            "  re-run (--force): this config already ran (%s); it appends to the same run_id "
            "(idempotent, dedupe-on-read)",
            idem.prior_status,
        )
    else:
        _log.warning(
            "  already ran: this config ran before (%s). A re-run is idempotent (dedupe-on-read "
            "under the same run_id); pass --force to acknowledge re-running it.",
            idem.prior_status,
        )


def _emit_plan(result: LaunchPlan) -> None:
    """Log the resolved plan, its DAG nodes, and each launch-command tier (dry-run/stage emit)."""
    verb = "stage" if result.staged else "dry-run"
    _log.info(
        "%s %s: runtime=%s python=%s bq=%s fanout=%s",
        verb,
        result.run_id,
        result.python_runtime,
        result.python_models,
        result.bq_models,
        result.fanout,
    )
    _emit_idempotency(result)
    for node in result.nodes:
        after = f" after [{', '.join(node.depends_on)}]" if node.depends_on else ""
        _log.info("  node %s: %s on %s%s", node.family, node.job_key, node.runtime, after)
    if result.commands is None:
        _log.info(
            "%s %s: infra unresolved (no SF_* env) — commands not emitted", verb, result.run_id
        )
        return
    for name, lc in result.commands.items():
        _log.info("  [%s] %s", name, lc.universal)
        if lc.native is not None:
            _log.info("  [%s:native] %s", name, lc.native)


def plan_run(
    cfg: RunConfig,
    *,
    settings: Settings | None = None,
    infra: object | None = None,
    force: bool = False,
) -> LaunchPlan:
    """Resolve a run offline to its id, fanout, runtime split, and launch-command *templates*.

    The "plan" verb: pure and GCP-free. Computes `_plan` and `estimate_fanout`, then — best-effort —
    resolves the infra identity, consults the registry for the exists-vs-new verdict
    (`_check_idempotency`), and
    builds the two-tier launch commands with the URIs the artifacts *will* land at (nothing is
    uploaded). When the infra can't be resolved offline (no ``SF_*`` env), the commands are omitted
    (``commands=None``) and the verdict is *unknown*, but the id/fanout/split are still returned, so
    a plain ``--dry-run`` works with no environment. ``force`` shapes only the emitted re-run
    guidance. Emits the plan to the log and returns it.
    """
    from .config import estimate_fanout
    from .dag import dag_nodes, plan_dag

    plan = _plan(cfg)
    fanout = estimate_fanout(cfg)
    nodes = dag_nodes(plan_dag(cfg))
    config_uri: str | None = None
    commands: dict[str, LaunchCommands] | None = None
    idempotency = Idempotency(checked=False, exists=False, prior_status=None)
    try:
        settings = settings or _resolve_settings()
        idempotency = _check_idempotency(plan.run_id, settings)
        resolved_infra = _resolve_infra(cfg, infra)
        code_bucket = resolved_infra.code_bucket  # type: ignore[attr-defined]
        config_uri, package_uri, launcher_uri = _template_uris(cfg, plan, code_bucket)
        commands = _assemble_commands(
            cfg,
            plan,
            settings,
            resolved_infra,
            config_uri=config_uri,
            package_uri=package_uri,
            launcher_uri=launcher_uri,
        )
    except ConfigError:
        # No SF_* env (or no injected settings/infra): return the plan without commands.
        config_uri = None
        commands = None
    result = LaunchPlan(
        run_id=plan.run_id,
        python_runtime=cfg.python_runtime,
        python_models=plan.python_models,
        bq_models=plan.bq_models,
        fanout=fanout,
        staged=False,
        config_uri=config_uri,
        commands=commands,
        idempotency=idempotency,
        force=force,
        nodes=nodes,
    )
    _emit_plan(result)
    return result


def _manifest_dict(result: LaunchPlan, *, created_at: str) -> dict[str, object]:
    """The reproducibility-manifest payload for a staged run (pure — ``created_at`` is caller-set).

    Records the config digest, fan-out, runtime split, both command tiers, the staged config URI,
    and the execution ``dag`` (one entry per family job + the ensemble, each with its deterministic
    ``job_key`` and ``depends_on``) — everything needed to answer "what command produced run X, and
    which jobs did it schedule under what ids?". The timestamp is passed in (not read here) so this
    stays a pure function with no wall-clock.
    """
    commands = {
        name: {"runtime": lc.runtime, "universal": lc.universal, "native": lc.native}
        for name, lc in (result.commands or {}).items()
    }
    dag = [
        {
            "job_key": n.job_key,
            "family": n.family,
            "runtime": n.runtime,
            "models": list(n.models),
            "hardware": n.hardware,
            "gpu_type": n.gpu_type,
            "spark_mode": n.spark_mode,
            "depends_on": list(n.depends_on),
        }
        for n in result.nodes
    ]
    return {
        "run_id": result.run_id,
        "created_at": created_at,
        "dag": dag,
        "python_runtime": result.python_runtime,
        "python_models": result.python_models,
        "bq_models": result.bq_models,
        "fanout": {
            "n_series": result.fanout.n_series,
            "n_models": result.fanout.n_models,
            "n_folds": result.fanout.n_folds,
            "n_cells": result.fanout.n_cells,
        },
        "config_uri": result.config_uri,
        "commands": commands,
        "force": result.force,
        "idempotency": {
            "checked": result.idempotency.checked,
            "exists": result.idempotency.exists,
            "prior_status": result.idempotency.prior_status,
        },
    }


def stage_run(
    cfg: RunConfig,
    *,
    settings: Settings | None = None,
    infra: object | None = None,
    force: bool = False,
) -> LaunchPlan:
    """Stage a run's artifacts to GCS and return the *runnable* launch commands — no submit.

    The "stage" verb: uploads the config (and, for Spark, the code zip + launcher shim) to the code
    bucket, builds the launch commands against those **real** URIs (so they run as-is from any ADC
    box), and writes the reproducibility manifest ``runs/<run_id>.plan.json`` next to the config.
    Unlike `plan_run`, the infra identity is required — staging touches GCS — so a missing ``SF_*``
    env raises rather than degrading; the exists-vs-new verdict (`_check_idempotency`) is therefore
    always resolved. ``force`` shapes only the emitted re-run guidance. Returns the `LaunchPlan`
    with ``staged=True``.
    """
    from datetime import UTC, datetime

    from .config import estimate_fanout
    from .dag import dag_nodes, plan_dag
    from .staging import stage_config, stage_manifest

    plan = _plan(cfg)
    fanout = estimate_fanout(cfg)
    nodes = dag_nodes(plan_dag(cfg))
    settings = settings or _resolve_settings()
    idempotency = _check_idempotency(plan.run_id, settings)
    resolved_infra = _resolve_infra(cfg, infra)
    code_bucket: str = resolved_infra.code_bucket  # type: ignore[attr-defined]

    config_uri = stage_config(cfg, plan.run_id, code_bucket)
    package_uri: str | None = None
    launcher_uri: str | None = None
    if cfg.python_runtime != "ray" and plan.python_models:
        from .submit import BatchInfra, _stage_code

        assert isinstance(resolved_infra, BatchInfra)  # spark runtime → BatchInfra
        package_uri, launcher_uri = _stage_code(resolved_infra)

    commands = _assemble_commands(
        cfg,
        plan,
        settings,
        resolved_infra,
        config_uri=config_uri,
        package_uri=package_uri,
        launcher_uri=launcher_uri,
    )
    result = LaunchPlan(
        run_id=plan.run_id,
        python_runtime=cfg.python_runtime,
        python_models=plan.python_models,
        bq_models=plan.bq_models,
        fanout=fanout,
        staged=True,
        config_uri=config_uri,
        commands=commands,
        idempotency=idempotency,
        force=force,
        nodes=nodes,
    )
    manifest_uri = stage_manifest(
        _manifest_dict(result, created_at=datetime.now(UTC).isoformat()), plan.run_id, code_bucket
    )
    _log.info("wrote run manifest: %s", manifest_uri)
    _emit_plan(result)
    return result


def _resolve_settings() -> Settings:
    """Resolve `Settings` from the ``SF_*`` env (raises `ConfigError` when unset)."""
    from .settings import Settings

    return Settings.resolve()


def _emit_airflow(cfg: RunConfig, config_uri: str, *, out_path: str | None = None) -> str:
    """Render this run's Airflow DAG to a local file and return the path (offline — touches no GCP).

    The "emit" verb: resolves the run's DAG and renders it to a standalone ``dag_<run_id>.py``
    (`airflow_emit.emit_airflow_dag`) whose tasks load the config from ``config_uri`` — the same
    ``--config``/``--config-uri`` value passed in, embedded verbatim. Pass a ``gs://``
    ``--config-uri`` (a staged config, digest == run_id) to emit a DAG a Composer environment can
    run directly; a local ``--config`` path emits a DAG suitable for local parse/compile checks and
    inspection. ``out_path`` overrides the default ``./dag_<run_id>.py`` destination. Writing is the
    only side effect.
    """
    from pathlib import Path

    from .airflow_emit import emit_airflow_dag

    run_id = make_run_id(cfg)
    source = emit_airflow_dag(cfg, config_uri)
    out = Path(out_path) if out_path else Path(f"dag_{run_id}.py")
    out.write_text(source, encoding="utf-8")
    return str(out)


def _print_probe_report(report: ProbeReport) -> None:
    """Print a `ProbeReport` as one compact per-family table — one job or a whole run, same shape.

    A header line (``run <id>  status=…  escalated=…  disagreement=…``) then one row per family:
    ``family · runtime · registry · native · verdict · n_done/n_expected · detail``. Written to
    stdout (not the logger) because the report *is* the ``--probe`` verb's output — it must show
    regardless of the log level — and it stays plain text (no plots, no colour) so it reads
    identically in a terminal, a notebook, or a Composer task log.
    """
    row_fmt = "  %-14s %-8s %-10s %-10s %-17s %-9s %s"
    print(
        f"run {report.run_id}  status={report.status}  "
        f"escalated={report.escalated}  disagreement={report.disagreement}"
    )
    print(row_fmt % ("family", "runtime", "registry", "native", "verdict", "done/exp", "detail"))
    for fv in report.families:
        expected = fv.n_expected if fv.n_expected is not None else "?"
        print(
            row_fmt
            % (
                fv.family,
                fv.runtime or "-",
                fv.registry_status or "-",
                fv.native_state or "-",
                fv.verdict,
                f"{fv.n_done}/{expected}",
                fv.detail or "",
            )
        )


def _main(argv: list[str] | None = None) -> None:
    """CLI: ``main (--config … | --config-uri …) [--dry-run | --stage-only | --emit-airflow]``."""
    import argparse

    from .config import load_config_uri

    p = argparse.ArgumentParser(prog="main", description="Run a forecast (Spark + BigQuery).")
    # Accept either a local path (--config, the interactive UX) or a gs:// URI (--config-uri, what
    # the emitted portable "main" command references — the staged config, digest == run_id). Exactly
    # one is required; load_config_uri resolves both forms.
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", help="path to the run config JSON")
    src.add_argument("--config-uri", help="gs:// (or local) URI of a staged run config JSON")
    verbs = p.add_mutually_exclusive_group()
    verbs.add_argument(
        "--dry-run", action="store_true", help="resolve + estimate fanout offline; touch no GCP"
    )
    verbs.add_argument(
        "--stage-only",
        action="store_true",
        help="stage artifacts to GCS + emit the runnable command + manifest; do not submit",
    )
    verbs.add_argument(
        "--emit-airflow",
        action="store_true",
        help="render this run's Airflow DAG to a local dag_<run_id>.py; touch no GCP",
    )
    verbs.add_argument(
        "--probe",
        action="store_true",
        help="registry-first reconciled status of this config's run; escalate incomplete/stale "
        "jobs to their runtime; touch no runtime if the run is already terminal",
    )
    p.add_argument(
        "--job",
        help="with --probe: narrow to one family "
        "(statistical/ml/deep_learning/native/ensemble)",
    )
    p.add_argument(
        "--emit-out",
        help="where to write the emitted DAG (default: ./dag_<run_id>.py); implies --emit-airflow",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="acknowledge re-running an already-run config (shapes the exists-vs-new guidance)",
    )
    ns = p.parse_args(argv)

    cfg = load_config_uri(ns.config or ns.config_uri)
    if ns.emit_airflow or ns.emit_out:
        out = _emit_airflow(cfg, ns.config or ns.config_uri, out_path=ns.emit_out)
        _log.info("wrote Airflow DAG: %s", out)
        return
    if ns.stage_only:
        result = stage_run(cfg, force=ns.force)
        _log.info("staged: %s", result.run_id)
        return
    if ns.probe:
        from .probes import probe_run

        report = probe_run(make_run_id(cfg), job=ns.job)
        _print_probe_report(report)
        return
    run_id = run(cfg, dry_run=ns.dry_run, force=ns.force)
    _log.info("%s: %s", "planned" if ns.dry_run else "submitted", run_id)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    _main()
