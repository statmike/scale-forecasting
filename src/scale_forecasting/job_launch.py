"""Run one DAG node as a registered job — the five-step recipe every node follows.

A run's execution DAG (`dag.plan_dag`) resolves to nodes: one per Python family
(statistical / ml / deep_learning), one for the BigQuery-native family, and the ensemble. This
module runs a node. It is deliberately *not* the thing that decides which nodes exist, what order
they go in, or when the run is finished — that is `main.run` locally and the emitted Airflow DAG
under Composer. **Both drivers call these same three functions**, which is what makes
"same code local ↔ Composer" true at the node level rather than merely claimed.

Every node, whatever it dispatches to, follows the same five steps:

1. **Resolve the attempt** — `registry.jobs.next_job_attempt`, bumped by ``force``.
2. **Derive the identity** — the deterministic ``job_key`` (`registry.ids.make_job_key`) mapped to
   the runtime's platform-legal id (``_system_job_id``), so the platform's own job and the
   ``run_jobs`` row share one identity and a trace never has to reverse a lossy mapping.
3. **Build the ENTRY probe handle** — the coordinates a probe reads *while the job is running*,
   asserting only what is truly known before submit (a Dataproc cluster job's id is server-assigned,
   so its ``native_id`` stays empty rather than emitting a false NOT_FOUND).
4. **Open the lifecycle** — `registry.lifecycle.run_job` writes the row RUNNING and finalizes its
   terminal status + wall-clock, whatever happens inside.
5. **Dispatch**, in contributor mode (``manage_header=False``) — the run's driver owns the single
   shared header, so no two writers ever touch it.

The three differ only in step 5 and in whether step 3 has a post-submit correction:
``launch_family_job`` hands off to the family's `RuntimeSubmitter` on a worker thread and stamps
the real id back once the platform assigns one; ``launch_native_job`` runs the BigQuery engine
inline on the driver; ``launch_ensemble_job`` runs the blend inline on the driver. The BigQuery
pair need no stamp-back — their coordinates are fully known up front.

Public surface: ``launch_family_job``, ``launch_native_job``, ``launch_ensemble_job``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .capacity import AWAITING_CAPACITY, CAPACITY_EXHAUSTED, CapacityExhausted, publishing_to

if TYPE_CHECKING:
    from collections.abc import Callable

    from .capacity import CapacityLedger
    from .config import RunConfig
    from .dag import FamilyJob
    from .settings import Settings


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


def _capacity_publisher(
    job_id: str, probe_handle: dict[str, Any], settings: Settings
) -> Callable[[CapacityLedger], None]:
    """A callback that writes a walk's live attempt ledger onto this family's ``run_jobs`` row.

    Installed by `launch_family_job` for the duration of one launch (`capacity.publishing_to`) and
    called by `capacity.walk` after every attempt, so a family stuck hopping regions reads as
    ``AWAITING_CAPACITY`` *while it is happening* rather than only in the post-mortem. That is the
    whole point of the state: a run whose deep-learning family has spent twenty minutes failing to
    find a T4 previously looked exactly like one that was computing.

    Two details are load-bearing. The telemetry is written as a **merge**, so the ledger accretes
    onto the row rather than replacing whatever else is on it; the entry probe handle is re-sent
    alongside it anyway, cheaply, so a row written by an older code path still comes out of a long
    wait probe-able. And the write is guarded against `lifecycle._STICKY_STATUSES`: an operator who
    cancels a family mid-walk must not have the next attempt write ``AWAITING_CAPACITY`` back over
    their ``CANCELLED``.

    `capacity._publish` swallows anything this raises — a failed telemetry write must not sink a
    walk that might still find room.
    """
    from .registry.jobs import update_job
    from .registry.lifecycle import _STICKY_STATUSES

    def publish(ledger: CapacityLedger) -> None:
        update_job(
            job_id,
            settings=settings,
            status=AWAITING_CAPACITY,
            merge_telemetry={"probe_handle": probe_handle, "capacity": ledger.to_json()},
            unless_status_in=_STICKY_STATUSES,
        )

    return publish


def launch_family_job(
    cfg: RunConfig,
    job: FamilyJob,
    run_id: str,
    settings: Settings,
    spark: object | None = None,
    *,
    force: bool = False,
    max_executors: int | None = None,
    ray_cluster: tuple[str, str] | None = None,
    spark_cluster: dict[str, tuple[str, str]] | None = None,
) -> None:
    """Run one Python family's job on its resolved runtime, wrapped in its ``run_jobs`` row.

    Called on a worker thread — one per Python family (statistical / ml / deep_learning), so the
    families run in parallel under one shared header. Resolves this family's attempt
    (`registry.jobs.next_job_attempt`, bumped by ``--force``), opens the per-job lifecycle
    (`registry.lifecycle.run_job`, which writes the row RUNNING and finalizes its terminal status +
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
    (`shared_clusters.shared_ray_cluster`): a Ray family reuses it instead of self-provisioning;
    every other runtime ignores it. ``spark_cluster``, when set, is the run's shared ephemeral
    Dataproc cluster(s) keyed by hardware — ``{"cpu": (name, region), ...}``
    (`shared_clusters.shared_spark_cluster`): an ephemeral cluster family looks itself up by its own
    ``compute.hardware`` and reuses that cluster (submitting to *its* region — a capacity failover
    may have moved one cluster off the deployment region while the other landed at home — with no
    per-family create/delete). Keyed rather than a single pair because a Dataproc cluster has one
    worker machine type, so a mixed run gets one cluster per hardware kind. A family naming its own
    standing cluster keeps that, and every other runtime/mode ignores the dict.
    """
    from .probes.vocabulary import ProbeHandle
    from .registry.ids import make_job_key
    from .registry.jobs import next_job_attempt
    from .registry.lifecycle import run_job
    from .submitters import get_submitter

    compute = job.compute
    assert compute is not None  # a Python family always resolves compute (native is handled inline)
    attempt, _ = next_job_attempt(run_id, job.family, force=force, settings=settings)
    system_job_id = _system_job_id(make_job_key(run_id, job.family, attempt), compute.runtime)
    # A shared Ray cluster (provisioned by the orchestrator for a multi-Ray-family run) is targeted
    # only by Ray families; every other runtime ignores it.
    ray_cluster_name = ray_cluster[0] if ray_cluster and compute.runtime == "ray" else None
    ray_cluster_region = ray_cluster[1] if ray_cluster and compute.runtime == "ray" else None
    # The shared Dataproc cluster is targeted only by ephemeral Spark cluster families (spark_mode
    # cluster, no standing cluster of their own) as a reuse target; a family naming its own standing
    # cluster keeps it, and serverless/Ray families ignore it. The lookup is by this family's own
    # hardware, because a run's cluster families split across one cluster per hardware kind — and a
    # miss means self-provision, which is correct rather than fatal: the orchestrator only builds
    # clusters for the hardware its planned families asked for.
    is_shared_spark_family = (
        spark_cluster is not None
        and compute.runtime == "spark"
        and compute.spark_mode == "cluster"
        and compute.spark_cluster_name is None
    )
    shared_spark = (
        spark_cluster.get(compute.hardware) if is_shared_spark_family and spark_cluster else None
    )
    shared_spark_name = shared_spark[0] if shared_spark else None
    shared_spark_region = shared_spark[1] if shared_spark else None
    # The ENTRY probe handle, built from coordinates known before submit — the handle the probe
    # actually reads while a job is RUNNING. It never asserts an id it doesn't truly have (a cluster
    # job's real id is server-assigned, so native_id is empty until the stamp-back refresh below),
    # so a probe degrades to registry-only rather than emitting a false NOT_FOUND.
    if compute.runtime == "ray":
        from .engines.ray_io import cluster_name as ray_cluster_name_for
        from .ray_cluster import cluster_resource_path

        # A shared cluster hands us its name and landed region; a *single*-family Ray run has no
        # shared cluster, and `submit_ray` creates one only once we call it — so the name is derived
        # here from the same pure rule the submitter will use. It has to be: `launch` blocks until
        # the job finishes, so the stamp-back below lands only at the very end, and a handle without
        # a resource_name leaves the probe and the cancel with nothing to reach for during the whole
        # window they exist to serve.
        #
        # The region is the one guess in it. A capacity hop would move the cluster off
        # ``settings.region`` and the predicted path would miss — the probe degrades to
        # registry-only, exactly as it does today, and the stamp-back corrects the record at the
        # end. A path that is right in the common case beats one that is never populated at all.
        resource_name = cluster_resource_path(
            settings,
            ray_cluster_name or ray_cluster_name_for(cfg, run_id),
            ray_cluster_region,
        )
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
    entry_blob = entry_handle.to_blob()
    job_id = make_job_key(run_id, job.family, attempt)
    # The publisher is installed *around* the dispatch, not passed into it: the capacity walk runs
    # several frames down inside the submitter, and this is the only frame that knows which row it
    # belongs to. See `capacity.publishing_to` for why it is ambient rather than a parameter.
    with (
        run_job(
            run_id,
            job.family,
            attempt,
            runtime=compute.runtime,
            spark_mode=compute.spark_mode,
            hardware=compute.hardware,
            gpu_type=compute.gpu_type,
            system_job_id=system_job_id,
            probe_handle=entry_blob,
            settings=settings,
        ) as fin,
        publishing_to(_capacity_publisher(job_id, entry_blob, settings)),
    ):
        try:
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
        except CapacityExhausted as exc:
            # The one failure worth re-running unchanged, and previously indistinguishable from a
            # broken import. `run_job`'s handler will write FAILED on the way out; what it cannot
            # know is *why*, so the reason and the finished ledger are attached here. Recorded via
            # the finalizer rather than a second write, so the row goes terminal exactly once.
            fin.finalize(
                failure_reason=CAPACITY_EXHAUSTED,
                telemetry={"probe_handle": entry_blob, "capacity": exc.ledger.to_json()},
            )
            raise
        # Stamp-back refresh: replace the entry handle with post-submit truths (a cluster's real id,
        # the landed region + Ray resource path). A cluster job's id is server-assigned, so when the
        # returned native_id differs from system_job_id, also stamp the real id back for
        # reverse-trace. The in-process session submits nothing and returns None (no refresh).
        #
        # The handle refresh is a *merge*: a family that walked regions before it got a cluster has
        # its whole attempt ledger under ``$.capacity`` by now, and replacing the column here would
        # erase — at the moment of success — the record of what it took to succeed.
        if handle is not None:
            fields: dict[str, Any] = {}
            if handle.native_id != system_job_id:
                fields["system_job_id"] = handle.native_id
            fin.finalize(telemetry={"probe_handle": handle.to_blob()}, **fields)


def launch_native_job(
    cfg: RunConfig,
    job: FamilyJob,
    run_id: str,
    settings: Settings,
    *,
    force: bool = False,
) -> object:
    """Run the BigQuery-native family inline (main thread), wrapped in its ``run_jobs`` row.

    Native models execute as SQL in BigQuery — no Python runtime, no worker thread — so this runs on
    the run driver's main thread, overlapping the Python family jobs. Like `launch_family_job` it
    resolves the ``native`` attempt, maps the deterministic ``job_key`` to the BigQuery job id
    (`_system_job_id`), and opens the per-job lifecycle (`registry.lifecycle.run_job`, ``runtime``
    fixed to ``"bigquery"``, carrying that id), then runs the engine in contributor mode. Returns
    the engine's `BqOutcome` so the caller can stamp the observed ``n_series`` onto the header.
    """
    from .engines import bigquery_engine
    from .probes.vocabulary import ProbeHandle
    from .registry.ids import make_job_key
    from .registry.jobs import next_job_attempt
    from .registry.lifecycle import run_job

    attempt, _ = next_job_attempt(run_id, "native", force=force, settings=settings)
    system_job_id = _system_job_id(make_job_key(run_id, "native", attempt), "bigquery")
    # BigQuery coordinates are fully known up front (jobs share the deterministic id prefix), so the
    # entry handle is the only one — there is no stamp-back site for the native family.
    native_handle = ProbeHandle(
        "bigquery",
        native_id=f"{system_job_id}-",
        region=settings.region,
        id_kind="prefix",
    )
    with run_job(
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


def launch_ensemble_job(
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
    ``run_jobs`` row (`registry.lifecycle.run_job`, ``runtime="bigquery"``: the node reads/writes
    BigQuery and blends in driver pandas, taking no Spark/Ray cluster), so a run's cross-system
    trace shows the ensemble beside the base jobs under the shared ``run_id``.

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
    from .probes.vocabulary import ProbeHandle
    from .registry.ids import make_job_key
    from .registry.jobs import next_job_attempt
    from .registry.lifecycle import run_job

    attempt, _ = next_job_attempt(run_id, "ensemble", force=force, settings=settings)
    system_job_id = _system_job_id(make_job_key(run_id, "ensemble", attempt), "bigquery")
    # BigQuery coordinates are fully known up front (jobs share the deterministic id prefix), so the
    # entry handle is the only one — there is no stamp-back site for the ensemble node.
    ensemble_handle = ProbeHandle(
        "bigquery",
        native_id=f"{system_job_id}-",
        region=settings.region,
        id_kind="prefix",
    )
    with run_job(
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
