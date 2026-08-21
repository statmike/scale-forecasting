"""The runtime-submitter spine: one interface, one implementation per Python runtime.

`main.run` drives exactly one Python-runtime job (Spark *xor* Ray) in parallel with the inline
BigQuery engine, under one shared ``run_id``. Which runtime — and whether it runs in-process against
an injected session or submits a remote job — is the only thing that varies between them; everything
else (the executed-subset contract, contributor-mode header ownership, block-until-terminal) is
shared. This module captures that seam as a `RuntimeSubmitter` protocol with a per-runtime
implementation, registered by ``cfg.python_runtime``, so `main._launch_python_runtime` is a single
registry dispatch and adding a runtime later (GKE, Cloud Run, Batch-on-GCE) is one class + one
registry entry — mirroring the one-model-one-file factory.

The GCP/engine imports stay lazy inside `launch` so importing this module (and `main`) never pulls
the Ray/Spark extras — only the chosen runtime's path loads them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from .errors import ConfigError

if TYPE_CHECKING:
    from .config import RunConfig
    from .settings import Settings


class RuntimeSubmitter(Protocol):
    """Launch a Python-runtime job for one run. ``name`` is the ``cfg.python_runtime`` key."""

    name: str

    def launch(
        self,
        cfg: RunConfig,
        *,
        models: list[str],
        manage_header: bool,
        settings: Settings,
        spark: object | None = None,
        wait: bool = True,
        max_executors: int | None = None,
        system_job_id: str | None = None,
        hardware: str = "cpu",
        gpu_type: str | None = None,
        spark_mode: str | None = None,
        spark_cluster_name: str | None = None,
    ) -> None:
        """Run ``models`` on this runtime, blocking until terminal when ``wait``.

        ``manage_header`` threads the header-ownership contract (``False`` = contributor mode, the
        caller owns the shared header). ``spark`` is an optional injected `SparkSession`, honored
        only by the runtimes that can run in-process against one (Spark); others ignore it.
        ``max_executors`` caps the remote Spark batch's dynamic-allocation ceiling; runtimes that
        autoscale on their own (Ray) or run in-process (an injected session) ignore it.

        ``system_job_id`` is the deterministic platform id the orchestrator derived for this
        family's job (`registry.ids.dataproc_job_id` / `ray_submission_id`). When set, the submitter
        hands it to the platform as the job's own id — a Dataproc ``batch_id`` or a Ray
        ``submission_id`` — so several families under one shared ``run_id`` get distinct, traceable
        jobs instead of colliding on a run-derived id. When ``None`` the platform assigns/derives an
        id (the standalone path).

        ``hardware``/``gpu_type`` carry the family's resolved accelerator (`ResolvedFamilyCompute`):
        ``hardware="gpu"`` provisions the runtime's GPU (a Serverless L4 executor, a Ray GPU pool);
        ``gpu_type`` names it. They default to CPU, so a CPU family launches exactly as before.

        ``spark_mode``/``spark_cluster_name`` select the Spark sub-runtime (``serverless`` batch xor
        a Dataproc ``cluster`` job, the latter reusing a standing cluster by name when given); other
        runtimes ignore them.
        """
        ...


class SparkSubmitter:
    """Dataproc Spark: a Serverless batch xor a cluster job, or in-process over an injected
    session."""

    name = "spark"

    def launch(
        self,
        cfg: RunConfig,
        *,
        models: list[str],
        manage_header: bool,
        settings: Settings,
        spark: object | None = None,
        wait: bool = True,
        max_executors: int | None = None,
        system_job_id: str | None = None,
        hardware: str = "cpu",
        gpu_type: str | None = None,
        spark_mode: str | None = None,
        spark_cluster_name: str | None = None,
    ) -> None:
        if spark is not None:
            # In-process Spark over an injected (Connect or local) session — no remote submit.
            # max_executors is a remote-batch dynamic-allocation cap; an injected session skips it.
            # system_job_id names a remote job; an in-process session submits none, so unused.
            # hardware/gpu_type/spark_mode shape a remote job's compute; a session runs local.
            from .engines import spark_explode

            spark_explode.run(
                cfg, models=models, manage_header=manage_header, settings=settings, spark=spark
            )
            return
        if spark_mode == "cluster":
            # A Dataproc cluster job (the T4 Spark path; ephemeral unless a cluster is named).
            from .dataproc_cluster import submit_cluster_job

            submit_cluster_job(
                cfg,
                engine="explode",
                models=models,
                manage_header=manage_header,
                settings=settings,
                wait=wait,
                hardware=hardware,
                gpu_type=gpu_type,
                spark_cluster_name=spark_cluster_name,
                job_id=system_job_id,
            )
            return
        from .submit import submit_batch

        submit_batch(
            cfg,
            engine="explode",
            models=models,
            manage_header=manage_header,
            settings=settings,
            wait=wait,
            max_executors=max_executors,
            batch_id=system_job_id,
            hardware=hardware,
            gpu_type=gpu_type,
        )


class RaySubmitter:
    """Autoscaling Ray on Vertex: submit a job to a cluster (no in-process path)."""

    name = "ray"

    def launch(
        self,
        cfg: RunConfig,
        *,
        models: list[str],
        manage_header: bool,
        settings: Settings,
        spark: object | None = None,
        wait: bool = True,
        max_executors: int | None = None,
        system_job_id: str | None = None,
        hardware: str = "cpu",
        gpu_type: str | None = None,
        spark_mode: str | None = None,
        spark_cluster_name: str | None = None,
    ) -> None:
        # spark, max_executors, and the spark_* args are ignored — there is no in-process Ray path
        # from the orchestrator, the Ray cluster autoscales on its own (no fixed executor cap), and
        # spark_mode/spark_cluster_name are Spark-only. system_job_id becomes the Ray submission_id
        # so the job's own id is deterministic; hardware="gpu" provisions the Ray GPU pool for this
        # family (kept out of cfg for run_id).
        from .ray_submit import submit_ray

        submit_ray(
            cfg,
            models=models,
            manage_header=manage_header,
            settings=settings,
            wait=wait,
            submission_id=system_job_id,
            use_gpu=(hardware == "gpu"),
            gpu_type=gpu_type,
        )


# Registered by cfg.python_runtime. A new runtime = one class + one entry here.
_SUBMITTERS: dict[str, RuntimeSubmitter] = {
    SparkSubmitter.name: SparkSubmitter(),
    RaySubmitter.name: RaySubmitter(),
}


def get_submitter(python_runtime: str) -> RuntimeSubmitter:
    """The `RuntimeSubmitter` for ``cfg.python_runtime`` (config already restricts the values)."""
    try:
        return _SUBMITTERS[python_runtime]
    except KeyError:
        raise ConfigError(
            f"no runtime submitter for python_runtime={python_runtime!r}; "
            f"known: {sorted(_SUBMITTERS)}"
        ) from None
