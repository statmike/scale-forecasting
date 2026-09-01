"""Reach a Dataproc **cluster** job, read what it says, stop it.

The cluster counterpart to `batch_telemetry`, and its own module for the same reason: observing a
job is a different capability from launching one, with a different caller. `probes.runtimes`
reconciles the registry against the live runtime and cancels in-flight work — it never submits
anything, and before this split it had to import two private names out of the cluster submitter to
do it.

`_job_client` is the handle (a regional `JobControllerClient`, distinct from the *cluster*
controller that creates and deletes clusters); `get_cluster_job` and `cancel_cluster_job` are the
read and the write against one already-submitted job; `_stamp_cluster_telemetry` files what the
submitter decided onto the run header. That last one stays here rather than with the submitter
because it is the cluster path's answer to `batch_telemetry._stamp_job_telemetry`, and the two
telemetry stories should be read side by side.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .errors import get_logger

if TYPE_CHECKING:
    from .settings import Settings

_log = get_logger(__name__)


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


def _stamp_cluster_telemetry(
    run_id: str, sizing: dict[str, Any], settings: Settings
) -> None:  # pragma: no cover - GCP I/O
    """Write a cluster job's sizing record to the run header's ``job_telemetry`` (best-effort).

    The cluster path's answer to `batch_telemetry._stamp_job_telemetry`. It stamps *less*: a
    Dataproc job has no ``approximate_usage`` and no runtime-config echo, so there is no DCU figure
    and no resolved-shape read-back — only what we decided and why. That is still the half of the
    record nobody could see before, and a cluster run that previously left ``v_run_summary`` blank
    now at least says what it asked for.

    Wrapped so any failure (API error, header not written) is logged and swallowed: telemetry is
    an overlay on an already-finished job, never a reason to fail one.
    """
    if not sizing:
        return
    from .registry.header import merge_header_telemetry, sizing_telemetry_path

    try:
        merge_header_telemetry(run_id, {sizing_telemetry_path(sizing): sizing}, settings=settings)
        _log.info("cluster sizing stamped for run %s", run_id)
    except Exception as exc:  # noqa: BLE001 - telemetry is best-effort, never fatal
        _log.warning("cluster sizing capture failed (non-fatal): %r", exc)
