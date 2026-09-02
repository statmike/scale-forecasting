"""The task callables an emitted Airflow DAG invokes — one thin wrapper per DAG node.

`airflow_emit` renders a ``dag_<run_id>.py`` whose every task is a ``PythonOperator`` pointing at a
callable here; this module is the seam between Airflow's scheduler and the *same* run building
blocks `main.run` uses locally. Each task re-derives everything it needs from the staged config URI
(`config.load_config_uri`) and the ``SF_*`` environment, so the DAG file carries no state beyond
that URI — a run is reproducible from its config alone, on Composer exactly as locally (no forked
logic).

The node set mirrors `dag.plan_dag`:

* ``begin_run`` — ensure the registry tables exist and write the run header (RUNNING).
* ``run_family`` / ``run_native`` — run one model family on its resolved runtime, each
  **self-owning its own ``run_jobs`` row** (via `job_launch.launch_family_job` /
  `job_launch.launch_native_job`, which wrap the launch in `registry.lifecycle.run_job`) — so the
  per-job trace and wall-clock are byte-identical to a live `main.run`, and a concurrent
  (microbatch) ensemble can watch the rows flip in real time.
* ``run_ensemble`` — the ensemble node, ``barrier`` (post-join) or ``microbatch`` (concurrent, its
  cross-process stop-signal polling ``run_jobs`` for base-family completion).
* ``create_ray_cluster`` / ``delete_ray_cluster`` (and the Dataproc-cluster pair) — the shared
  ephemeral-cluster bracket for a run with several ephemeral Ray (or Spark-cluster) families, the
  DAG-task form of `shared_clusters.shared_ray_cluster` / ``shared_spark_cluster`` split into
  create/delete. One task pair either way, but the Dataproc one manages a cluster **per hardware
  kind** (one worker machine type per cluster), so its XCom is a dict where Ray's is a pair.
* ``finalize_run`` — read every family's ``run_jobs`` outcome and finalize the header with the
  combined run status (the DAG's terminal join, ``trigger_rule="all_done"``).

Imports stay lazy inside each function so importing this module (which the DAG file does at parse
time, on every scheduler heartbeat) never pulls the GCP/engine extras.
"""

from __future__ import annotations

from typing import Any

# A family/ensemble ``run_jobs`` row is "done" once it reaches one of these; a row still RUNNING (or
# absent) when finalize reads it means the task died before finalizing, which the run counts as
# FAILED. The same set is the microbatch ensemble's cross-process "base family finished" signal.
# CANCELLED counts as done (a stopped job won't produce more) so a drained ensemble stops waiting.
_TERMINAL_STATUSES = frozenset({"COMPLETED", "FAILED", "PARTIAL", "CANCELLED"})


def _xcom_cluster(ti: Any, task_id: str) -> tuple[str, str] | None:
    """Pull a shared cluster's ``[name, region]`` from an upstream create task's XCom, or ``None``.

    The create-cluster task returns ``[name, region]`` (a capacity failover may have moved the
    cluster off the deployment region, so the region is discovered at create time, not emit time).
    Returns ``None`` when there is no task instance (a direct unit call) or the create task did not
    run (no shared cluster for this run) — in which case the family self-provisions as usual.
    """
    if ti is None:
        return None
    value = ti.xcom_pull(task_ids=task_id)
    if not value:
        return None
    return (value[0], value[1])


def _xcom_spark_clusters(ti: Any) -> dict[str, tuple[str, str]] | None:
    """Pull the shared Dataproc clusters ``{hardware: (name, region)}`` from XCom, or ``None``.

    The Dataproc counterpart of `_xcom_cluster`, keyed rather than a single pair because a run's
    cluster families split across one cluster per hardware kind (`shared_clusters
    .shared_spark_inputs`). XCom round-trips through JSON, so the create task's ``[name, region]``
    lists come back as lists; they are normalized to tuples here so callers see the same shape the
    local path yields.
    """
    if ti is None:
        return None
    value = ti.xcom_pull(task_ids="create_spark_cluster")
    if not value:
        return None
    return {hardware: (pair[0], pair[1]) for hardware, pair in value.items()}


def combined_run_status(job_statuses: dict[str, str | None], *, ensemble_enabled: bool) -> str:
    """Roll the per-family ``run_jobs`` statuses into one header status (pure — mirrors
    `main._combined_status`).

    Over the base families (every key but ``"ensemble"``): all ``COMPLETED`` → ``COMPLETED``; all
    non-``COMPLETED`` → ``FAILED``; a mix → ``PARTIAL`` (surviving families' forecasts stay usable).
    A missing or still-RUNNING family row counts as failed (its task died before finalizing). An
    ensemble that did not complete downgrades an otherwise-``COMPLETED`` run to ``FAILED`` (the
    requested output is incomplete); it never masks a family ``PARTIAL``/``FAILED``.
    """
    base = {family: status for family, status in job_statuses.items() if family != "ensemble"}
    n_jobs = len(base)
    n_failed = sum(1 for status in base.values() if status != "COMPLETED")
    if n_jobs == 0 or n_failed == 0:
        engine_status = "COMPLETED"
    elif n_failed == n_jobs:
        engine_status = "FAILED"
    else:
        engine_status = "PARTIAL"

    if ensemble_enabled and engine_status == "COMPLETED":
        if job_statuses.get("ensemble") != "COMPLETED":
            return "FAILED"
    return engine_status


def begin_run(config_uri: str) -> str:
    """Open the run: ensure the registry tables exist, then write the header RUNNING; return run_id.

    The owner-mode entry half of `registry.lifecycle.run_header`, split out so the DAG can write
    the header up front and finalize it in a separate task process (`finalize_run`). Idempotent:
    re-running the
    same config re-derives the same ``run_id`` and appends a fresh header row (latest wins),
    matching the local re-run behavior.
    """
    from .config import load_config_uri
    from .registry.header import write_header
    from .registry.ids import make_run_id
    from .registry.tables import ensure_tables
    from .settings import Settings

    cfg = load_config_uri(config_uri)
    run_id = make_run_id(cfg)
    settings = Settings.resolve()
    ensure_tables(cfg, settings=settings)
    write_header(cfg, run_id, settings=settings)
    return run_id


def run_family(config_uri: str, family: str, ti: Any = None) -> None:
    """Run one Python family's job on its resolved runtime (the call `main.run` makes per family).

    Re-plans the run's DAG from the staged config (`dag.plan_dag`), selects this ``family``'s
    `dag.FamilyJob`, and dispatches through `job_launch.launch_family_job` — which opens the
    family's ``run_jobs`` row (RUNNING → terminal + wall-clock), maps its deterministic ``job_key``
    to the platform job id, and submits to the family's runtime submitter in contributor mode (the
    shared header is owned by `begin_run`/`finalize_run`). When the run shares an ephemeral cluster,
    the ``(name, region)`` is pulled from the create-cluster task's XCom and threaded in; a family
    that does not share one self-provisions.
    """
    from . import job_launch
    from .config import load_config_uri
    from .dag import plan_dag
    from .errors import ConfigError
    from .registry.ids import make_run_id
    from .settings import Settings

    cfg = load_config_uri(config_uri)
    run_id = make_run_id(cfg)
    settings = Settings.resolve()
    run_dag = plan_dag(cfg)
    job = next((j for j in run_dag.python_jobs if j.family == family), None)
    if job is None:
        raise ConfigError(f"run_family: no Python family {family!r} in run {run_id}")
    job_launch.launch_family_job(
        cfg,
        job,
        run_id,
        settings,
        ray_cluster=_xcom_cluster(ti, "create_ray_cluster"),
        spark_cluster=_xcom_spark_clusters(ti),
    )


def run_native(config_uri: str) -> None:
    """Run the BigQuery-native family's job inline (the same call `main.run` makes for native).

    Selects the run's native `dag.FamilyJob` and dispatches through `job_launch.launch_native_job`,
    which opens the native ``run_jobs`` row and runs the BigQuery engine in contributor mode under
    the shared header. Raises `errors.ConfigError` if the config has no native models (the DAG would
    not have emitted this task).
    """
    from . import job_launch
    from .config import load_config_uri
    from .dag import plan_dag
    from .errors import ConfigError
    from .registry.ids import make_run_id
    from .settings import Settings

    cfg = load_config_uri(config_uri)
    run_id = make_run_id(cfg)
    settings = Settings.resolve()
    job = plan_dag(cfg).native_job
    if job is None:
        raise ConfigError(f"run_native: run {run_id} has no BigQuery-native family")
    job_launch.launch_native_job(cfg, job, run_id, settings)


def run_ensemble(config_uri: str) -> None:
    """Run the ensemble node — ``barrier`` (post-join) or ``microbatch`` (concurrent) per config.

    Dispatches through `job_launch.launch_ensemble_job`, which opens the ensemble's own ``run_jobs``
    row and blends the base predictions into the consensus pseudo-models. In ``microbatch`` mode the
    DAG runs this task **in parallel** with the base families (gated only on `begin_run`); its
    ``upstream_done`` predicate is the cross-process equivalent of the in-process ``base_done``
    event — it polls ``run_jobs`` (`registry.jobs.read_run_jobs`) and reports the base jobs finished
    once every base family has reached a terminal status, so the drain loop stops after its final
    ready-series pass. ``barrier`` mode ignores the predicate (the task already runs after the
    join).
    """
    from . import job_launch
    from .config import load_config_uri
    from .dag import plan_dag
    from .registry.ids import make_run_id
    from .registry.jobs import read_run_jobs
    from .settings import Settings

    cfg = load_config_uri(config_uri)
    run_id = make_run_id(cfg)
    settings = Settings.resolve()

    if cfg.compute.ensemble.mode == "microbatch":
        base_families = plan_dag(cfg).families  # every base family; excludes the ensemble node

        def upstream_done() -> bool:
            rows = read_run_jobs(run_id, settings=settings)
            statuses = {r["family"]: r.get("status") for r in rows}
            return all(statuses.get(family) in _TERMINAL_STATUSES for family in base_families)

        job_launch.launch_ensemble_job(cfg, run_id, settings, upstream_done=upstream_done)
    else:
        job_launch.launch_ensemble_job(cfg, run_id, settings)


def finalize_run(config_uri: str) -> None:
    """Close the run: read every family's ``run_jobs`` outcome and finalize the header status.

    The DAG's terminal join (``trigger_rule="all_done"``, so it runs even when a family failed). The
    owner-mode exit half of `registry.lifecycle.run_header`, split out to a separate task: reads the
    per-job rows (`registry.jobs.read_run_jobs`), rolls them into the combined status
    (`combined_run_status`), and stamps ``status`` + a whole-run ``runtime_seconds`` (the slowest
    parallel job's wall-clock) + the ``bq_models`` list onto the header
    (`registry.header.update_header`). Each family's own row was
    already finalized by its task, so this only reconciles the header the base jobs run *under*.
    """
    from .config import load_config_uri
    from .dag import plan_dag
    from .registry.header import update_header
    from .registry.ids import make_run_id
    from .registry.jobs import read_run_jobs
    from .settings import Settings

    cfg = load_config_uri(config_uri)
    run_id = make_run_id(cfg)
    settings = Settings.resolve()
    run_dag = plan_dag(cfg)

    rows = read_run_jobs(run_id, settings=settings)
    statuses = {r["family"]: r.get("status") for r in rows}
    status = combined_run_status(statuses, ensemble_enabled=run_dag.ensemble_enabled)
    runtime_seconds = max((r.get("runtime_seconds") or 0.0 for r in rows), default=0.0)
    native = run_dag.native_job
    update_header(
        run_id,
        settings=settings,
        status=status,
        runtime_seconds=runtime_seconds,
        bq_models=list(native.models) if native else [],
    )


def create_ray_cluster(config_uri: str) -> list[str]:
    """Provision the run's one shared ephemeral Ray cluster; return ``[name, region]`` for XCom.

    The create half of `shared_clusters.shared_ray_cluster`, split to a task. Sizes one cluster for
    the union of the run's ephemeral Ray families (`shared_clusters.shared_ray_inputs`) so each
    family submits its own
    failure-isolated job to it instead of colliding on the run-derived name. The region is returned
    alongside the name because a capacity failover may move the cluster off the deployment region;
    the family tasks and `delete_ray_cluster` read both from this task's XCom.
    """
    from . import ray_cluster, shared_clusters
    from .config import load_config_uri
    from .dag import plan_dag
    from .errors import ConfigError
    from .registry.ids import make_run_id
    from .settings import Settings

    cfg = load_config_uri(config_uri)
    run_id = make_run_id(cfg)
    settings = Settings.resolve()
    inputs = shared_clusters.shared_ray_inputs(plan_dag(cfg).python_jobs)
    if inputs is None:
        raise ConfigError(f"create_ray_cluster: run {run_id} has no shared Ray families")
    models, any_gpu, gpu_type = inputs
    name, region = ray_cluster.provision_shared_cluster(
        cfg, models=models, run_id=run_id, use_gpu=any_gpu, gpu_type=gpu_type, settings=settings
    )
    return [name, region]


def delete_ray_cluster(config_uri: str, ti: Any = None) -> None:
    """Tear down the run's shared ephemeral Ray cluster (``trigger_rule="all_done"`` in the DAG).

    The teardown half of `shared_clusters.shared_ray_cluster`, split to a task that always runs —
    so a family
    failure never leaks the cluster. Reads the ``[name, region]`` the create task returned via XCom;
    a no-op when there was none (nothing to tear down).
    """
    from . import ray_cluster
    from .settings import Settings

    cluster = _xcom_cluster(ti, "create_ray_cluster")
    if cluster is None:
        return
    ray_cluster.teardown_shared_cluster(cluster[0], cluster[1], Settings.resolve())


def create_spark_cluster(config_uri: str) -> dict[str, list[str]]:
    """Provision the run's shared ephemeral Dataproc cluster **per hardware kind**; return
    ``{hardware: [name, region]}`` for XCom.

    The Dataproc analog of `create_ray_cluster` — the create half of
    `shared_clusters.shared_spark_cluster`. Sizes a cluster for each hardware kind among the run's
    ephemeral ``spark_mode="cluster"`` families (`shared_clusters.shared_spark_inputs`) so each
    submits its own job to the right-sized one rather than racing to create and tear down a shared
    name. One cluster per kind because a Dataproc cluster has exactly one worker machine type — a
    mixed CPU/GPU run cannot be served by one, and serving it with one GPU cluster would put
    accelerators under the CPU families' work. Each value carries its own region, because a capacity
    failover may have moved one cluster and not the other.

    **A partial create tears itself down before raising.** Unlike the local path's ``ExitStack``,
    this task has no ``finally`` reaching into `delete_spark_cluster` — a raising task pushes no
    XCom, so the downstream teardown would find nothing and the already-created cluster would bill
    on unnoticed. Cleaning up here is the only place that can see it.
    """
    from . import shared_clusters
    from .config import load_config_uri
    from .dag import plan_dag
    from .dataproc_cluster import provision_shared_cluster, teardown_shared_cluster
    from .errors import ConfigError
    from .registry.ids import make_run_id
    from .settings import Settings

    cfg = load_config_uri(config_uri)
    run_id = make_run_id(cfg)
    settings = Settings.resolve()
    inputs = shared_clusters.shared_spark_inputs(plan_dag(cfg).python_jobs)
    if inputs is None:
        raise ConfigError(
            f"create_spark_cluster: run {run_id} has no shared Dataproc-cluster families"
        )
    suffixed = len(inputs) > 1
    clusters: dict[str, list[str]] = {}
    try:
        for hardware, (models, gpu_type) in inputs.items():
            name, region = provision_shared_cluster(
                cfg,
                run_id=run_id,
                use_gpu=hardware == "gpu",
                gpu_type=gpu_type,
                settings=settings,
                models=models,
                name_suffix=hardware if suffixed else None,
            )
            clusters[hardware] = [name, region]
    except Exception:
        for name, region in clusters.values():
            teardown_shared_cluster(name, region, settings)
        raise
    return clusters


def delete_spark_cluster(config_uri: str, ti: Any = None) -> None:
    """Tear down every shared ephemeral Dataproc cluster the run created
    (``trigger_rule="all_done"`` in DAG).

    The Dataproc analog of `delete_ray_cluster` — the teardown half of
    `shared_clusters.shared_spark_cluster`, always run so a family failure never leaks a cluster.
    Reads the ``{hardware: [name, region]}`` the create task returned via XCom; a no-op when there
    was none.

    Every entry is attempted even if one teardown raises, and the first error is re-raised only
    after the rest have been tried: bailing on the first failure would leak the cluster behind it,
    which is exactly the outcome this task exists to prevent.
    """
    from .dataproc_cluster import teardown_shared_cluster
    from .settings import Settings

    clusters = _xcom_spark_clusters(ti)
    if not clusters:
        return
    settings = Settings.resolve()
    first_error: Exception | None = None
    for name, region in clusters.values():
        try:
            teardown_shared_cluster(name, region, settings)
        except Exception as exc:  # noqa: BLE001 - keep tearing down; re-raised below
            first_error = first_error or exc
    if first_error is not None:
        raise first_error
