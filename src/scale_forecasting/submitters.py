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
    ) -> None:
        """Run ``models`` on this runtime, blocking until terminal when ``wait``.

        ``manage_header`` threads the header-ownership contract (``False`` = contributor mode, the
        caller owns the shared header). ``spark`` is an optional injected `SparkSession`, honored
        only by the runtimes that can run in-process against one (Spark); others ignore it.
        ``max_executors`` caps the remote Spark batch's dynamic-allocation ceiling; runtimes that
        autoscale on their own (Ray) or run in-process (an injected session) ignore it.
        """
        ...


class SparkSubmitter:
    """Dataproc Serverless Spark: a remote batch, or in-process over an injected session."""

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
    ) -> None:
        if spark is not None:
            # In-process Spark over an injected (Connect or local) session — no remote batch submit.
            # max_executors is a remote-batch dynamic-allocation cap; an injected session skips it.
            from .engines import spark_explode

            spark_explode.run(
                cfg, models=models, manage_header=manage_header, settings=settings, spark=spark
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
    ) -> None:
        # spark and max_executors are ignored — there is no in-process Ray path from the
        # orchestrator, and the Ray cluster autoscales on its own (no fixed executor cap).
        from .ray_submit import submit_ray

        submit_ray(
            cfg, models=models, manage_header=manage_header, settings=settings, wait=wait
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
