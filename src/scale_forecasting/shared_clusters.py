"""One shared ephemeral cluster for a whole run, instead of one per family.

A run's Python families each resolve their own runtime and hardware, and each knows how to
provision what it needs. That independence is right at the *node* level and wrong at the *run*
level: two families that both resolve to an ephemeral cluster derive the **same** run-scoped
name — ``sf-ray-<run_id>`` or ``sf-cluster-<run_id>`` — so they collide on create, and the first
one to finish tears the cluster down out from under the others. This module is the run-level
answer: provision **one** cluster before the fan-out, hand every eligible family its
``(name, region)`` as a reuse target, and tear it down once when the block exits.

Two symmetric pairs, one per runtime — a pure predicate and the bracket that acts on it:

* ``shared_ray_inputs`` / ``shared_ray_cluster`` — Vertex Ray.
* ``shared_spark_inputs`` / ``shared_spark_cluster`` — ephemeral Dataproc clusters.

The **inputs** half is pure and offline: given the run's planned Python jobs it answers "does
sharing apply, and if so what does the one cluster have to be big enough for" — the union of the
eligible families' models, whether any of them needs a GPU pool, and which GPU type to size it
with. Fewer than two eligible families returns ``None``, which keeps the proven self-provisioning
path rather than inventing a shared one for a run that cannot collide.

That the predicate is *pure and separable* is what lets the Airflow surface reuse it. `airflow_emit`
calls the same two functions at DAG-emit time to decide whether to emit the ``create_ray_cluster`` /
``delete_ray_cluster`` (and Dataproc) task pair at all, and `airflow_tasks` calls them again inside
those tasks to size the cluster — so a Composer run brackets its fan-out with exactly the cluster
`main.run` would have provisioned locally. The context manager and the DAG-task pair are two
spellings of one decision, and the decision itself lives here once.

The **bracket** half is a context manager that yields ``(cluster_name, cluster_region)`` or
``None``. The region is yielded, not assumed, because a capacity failover may have moved the
cluster off the deployment region and each family's job has to submit to where it actually landed.
Teardown is in a ``finally``, so a family failure never leaks a cluster.

Public surface: ``shared_ray_inputs``, ``shared_ray_cluster``, ``shared_spark_inputs``,
``shared_spark_cluster``.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .config import RunConfig
    from .dag import FamilyJob, RunDag
    from .settings import Settings


def shared_ray_inputs(python_jobs: list[FamilyJob]) -> tuple[list[str], bool, str | None] | None:
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
def shared_ray_cluster(
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
        inputs = shared_ray_inputs(run_dag.python_jobs)
    if inputs is None:
        yield None
        return
    from . import ray_cluster

    models, any_gpu, gpu_type = inputs
    name, region = ray_cluster.provision_shared_cluster(
        cfg, models=models, run_id=run_id, use_gpu=any_gpu, gpu_type=gpu_type, settings=settings
    )
    try:
        yield (name, region)
    finally:
        ray_cluster.teardown_shared_cluster(name, region, settings)


def shared_spark_inputs(
    python_jobs: list[FamilyJob],
) -> tuple[list[str], bool, str | None] | None:
    """The union sizing inputs for a run's ephemeral Dataproc cluster families, or ``None`` if fewer
    than two.

    Sharing one cluster only matters when **more than one** family resolves to an ephemeral Spark
    cluster (``spark_mode="cluster"`` with no standing ``spark_cluster_name``) — the case that would
    otherwise have each family both create *and* tear down the shared run-derived
    ``sf-cluster-<run_id>`` name, so a family finishing first deletes the cluster out from under the
    others. Returns the union of those families' models, whether **any** of them needs a GPU pool,
    and the GPU type to size it (the first GPU family's) — the inputs to one shared cluster. The
    models are only the *cluster* families': a run whose Ray or BigQuery-native families dwarf its
    Spark ones must not buy workers for work that never lands here. Fewer than two ephemeral cluster
    families (or none) returns ``None`` and keeps the proven per-family lifecycle (a single family
    has no collision risk; a family naming a standing cluster already reuses).
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
    models: list[str] = []
    any_gpu = False
    gpu_type: str | None = None
    for j in cluster_jobs:
        assert j.compute is not None  # a Python family always resolves compute
        models.extend(j.models)
        if j.compute.hardware == "gpu":
            any_gpu = True
            gpu_type = gpu_type or j.compute.gpu_type
    return models, any_gpu, gpu_type


@contextmanager
def shared_spark_cluster(
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
    analog of `shared_ray_cluster`.
    """
    inputs = shared_spark_inputs(run_dag.python_jobs)
    if inputs is None:
        yield None
        return
    from .dataproc_cluster import provision_shared_cluster, teardown_shared_cluster

    models, any_gpu, gpu_type = inputs
    name, region = provision_shared_cluster(
        cfg,
        run_id=run_id,
        use_gpu=any_gpu,
        gpu_type=gpu_type,
        settings=settings,
        models=models,
    )
    try:
        yield (name, region)
    finally:
        teardown_shared_cluster(name, region, settings)
