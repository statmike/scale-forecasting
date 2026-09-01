"""One live-state probe per runtime — the read (and stop) seam, and its native state maps.

Mirrors `submitters.py`'s `RuntimeSubmitter` spine, deliberately: all probes in one file, **one
class + one registry entry per runtime**, dispatched by `get_probe`. Adding a runtime is adding a
class and a line to `_PROBES`, and keeping them together is what makes that obvious.

Each `check` maps the platform-native job state into the closed ``NATIVE_*`` set from
`vocabulary`, so the layer above reconciles against one word list regardless of who ran the job.

**A probe is advisory.** Every native call is capped at `_PROBE_TIMEOUT_S` and every method
swallows its exceptions — `check` degrades to ``UNKNOWN``, `cancel` reports a failed
`vocabulary.CancelResult`. Neither ever raises and neither ever hangs, so a probe failure can
never take down the reader that called it. The GCP/engine imports stay lazy inside each method:
importing this module loads no cloud client, and only the probed runtime's path pulls its extra.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any

from ..errors import ConfigError
from .vocabulary import (
    NATIVE_FAILED,
    NATIVE_NOT_FOUND,
    NATIVE_RUNNING,
    NATIVE_SUCCEEDED,
    NATIVE_UNKNOWN,
    CancelResult,
    ProbeHandle,
    ProbeResult,
    RuntimeProbe,
    _parse_ts,
)

if TYPE_CHECKING:
    from ..settings import Settings


# A probe is advisory and must never block the reader that called it: cap every native client call
# that accepts a timeout at this ceiling, and degrade to UNKNOWN when one still fails.
_PROBE_TIMEOUT_S = 20.0

# BigQuery's per-run jobs share an id prefix, but a bare list_jobs walks the whole project history.
# Bound the scan by the job's start time (see `ProbeHandle.created_at`, minus a skew margin) and a
# hard ceiling, so a probe of a busy project stays within its advisory time budget instead of paging
# through unrelated history.
_BQ_MAX_JOBS_SCAN = 2000
_BQ_SCAN_SKEW = timedelta(minutes=5)

# --- native → normalized state maps -------------------------------------------
# The platform enum *names* (``State.name`` strings) map to the closed set. Kept as plain string
# dicts (no SDK import) so this module stays import-light; an unrecognized name falls through to
# UNKNOWN via ``.get(..., NATIVE_UNKNOWN)``.

# Dataproc Serverless ``Batch.State``. CANCELLING is still winding down → treat as RUNNING; a
# CANCELLED batch reached a non-success terminal state → FAILED.
_SPARK_BATCH_STATES = {
    "PENDING": NATIVE_RUNNING,
    "RUNNING": NATIVE_RUNNING,
    "CANCELLING": NATIVE_RUNNING,
    "SUCCEEDED": NATIVE_SUCCEEDED,
    "FAILED": NATIVE_FAILED,
    "CANCELLED": NATIVE_FAILED,
}

# Dataproc **cluster** ``JobStatus.State``. DONE is the only success; ERROR / CANCELLED /
# ATTEMPT_FAILURE are non-success terminals; everything pre-terminal is RUNNING.
_SPARK_CLUSTER_STATES = {
    "PENDING": NATIVE_RUNNING,
    "SETUP_DONE": NATIVE_RUNNING,
    "RUNNING": NATIVE_RUNNING,
    "CANCEL_PENDING": NATIVE_RUNNING,
    "CANCEL_STARTED": NATIVE_RUNNING,
    "DONE": NATIVE_SUCCEEDED,
    "ERROR": NATIVE_FAILED,
    "CANCELLED": NATIVE_FAILED,
    "ATTEMPT_FAILURE": NATIVE_FAILED,
}

# Ray Jobs API ``JobStatus``. STOPPED (a job stopped before completing) is a non-success terminal.
_RAY_JOB_STATES = {
    "PENDING": NATIVE_RUNNING,
    "RUNNING": NATIVE_RUNNING,
    "SUCCEEDED": NATIVE_SUCCEEDED,
    "FAILED": NATIVE_FAILED,
    "STOPPED": NATIVE_FAILED,
}


def _short_detail(exc: Exception) -> str:
    """A concise, scannable degrade reason — the exception type + its first message line,
    truncated — so the probe table shows an actionable hint, not a multi-line repr/traceback."""
    first = (str(exc).strip().splitlines() or [""])[0]
    return f"{type(exc).__name__}: {first}"[:160] if first else type(exc).__name__


def _cancel_failure(exc: Exception) -> CancelResult:
    """Map a swallowed cancel exception to a failed `vocabulary.CancelResult`, with an IAM hint.

    A `PermissionDenied` (or any error whose message names a missing permission) is the common
    enterprise case — a read-only *probe-reader* principal trying to *cancel* without the
    *job-canceller* role (§9 of the design). We translate it to an actionable one-liner rather than
    surfacing a raw stack trace; every other error degrades to a `_short_detail` summary. Either way
    ``stopped=already_gone=False`` so the caller does not finalize the registry to CANCELLED."""
    from google.api_core.exceptions import Forbidden, PermissionDenied

    if isinstance(exc, PermissionDenied | Forbidden):
        return CancelResult(
            stopped=False,
            already_gone=False,
            detail="permission denied — cancel needs the job-canceller role "
            "(dataproc.batches.delete / dataproc.jobs.cancel / bigquery.jobs.update / Ray stop)",
        )
    return CancelResult(stopped=False, already_gone=False, detail=_short_detail(exc))


class SparkProbe:
    """Probe a Dataproc Spark job — a Serverless batch xor a cluster job, by ``handle.spark_mode``.

    Serverless reuses `batch_telemetry._batch_client` + ``get_batch`` (with
    `batch_telemetry.extract_job_telemetry`
    for the usage overlay); cluster reuses `cluster_telemetry.get_cluster_job`, the non-blocking
    read. A missing batch/job (``NotFound``) is NOT_FOUND (``exists=False``); any other error
    degrades to UNKNOWN.
    """

    name = "spark"

    def check(self, handle: ProbeHandle, *, settings: Settings) -> ProbeResult:
        if handle.spark_mode == "cluster":
            return self._check_cluster(handle, settings=settings)
        return self._check_serverless(handle, settings=settings)

    def _check_serverless(self, handle: ProbeHandle, *, settings: Settings) -> ProbeResult:
        try:
            from google.api_core.exceptions import NotFound

            from ..batch_telemetry import _batch_client, extract_job_telemetry

            client = _batch_client(handle.region)
            parent = f"projects/{settings.project_id}/locations/{handle.region}"
            try:
                batch = client.get_batch(
                    name=f"{parent}/batches/{handle.native_id}", timeout=_PROBE_TIMEOUT_S
                )
            except NotFound:
                return ProbeResult(NATIVE_NOT_FOUND, exists=False, detail="batch not found")
            state = getattr(batch, "state", None)
            state_name = getattr(state, "name", str(state))
            native = _SPARK_BATCH_STATES.get(state_name, NATIVE_UNKNOWN)
            detail = getattr(batch, "state_message", "") or ""
            return ProbeResult(
                native, exists=True, detail=detail, telemetry=extract_job_telemetry(batch)
            )
        except Exception as exc:  # noqa: BLE001 - a probe is advisory: degrade, never raise
            return ProbeResult(NATIVE_UNKNOWN, exists=True, detail=_short_detail(exc))

    def _check_cluster(self, handle: ProbeHandle, *, settings: Settings) -> ProbeResult:
        # A cluster job's real id is server-assigned and only stamped back after submission, so the
        # entry handle carries native_id="" for the launch window. Without an id we can't address
        # the job — report UNKNOWN (exists=True) rather than a false NOT_FOUND, honouring the entry-
        # handle contract that a probe never asserts an id it doesn't truly have.
        if not handle.native_id:
            return ProbeResult(
                NATIVE_UNKNOWN, exists=True, detail="cluster job id not yet assigned"
            )
        try:
            from google.api_core.exceptions import NotFound

            from ..cluster_telemetry import get_cluster_job

            try:
                state_name, detail = get_cluster_job(
                    handle.region, handle.native_id, settings=settings, timeout=_PROBE_TIMEOUT_S
                )
            except NotFound:
                return ProbeResult(NATIVE_NOT_FOUND, exists=False, detail="cluster job not found")
            native = _SPARK_CLUSTER_STATES.get(state_name, NATIVE_UNKNOWN)
            return ProbeResult(native, exists=True, detail=detail)
        except Exception as exc:  # noqa: BLE001 - a probe is advisory: degrade, never raise
            return ProbeResult(NATIVE_UNKNOWN, exists=True, detail=_short_detail(exc))

    def cancel(self, handle: ProbeHandle, *, settings: Settings) -> CancelResult:
        if handle.spark_mode == "cluster":
            return self._cancel_cluster(handle, settings=settings)
        return self._cancel_serverless(handle, settings=settings)

    def _cancel_serverless(self, handle: ProbeHandle, *, settings: Settings) -> CancelResult:
        # Dataproc Serverless has no separate "cancel" — deleting a running batch stops it.
        try:
            from google.api_core.exceptions import NotFound

            from ..batch_telemetry import _batch_client

            client = _batch_client(handle.region)
            parent = f"projects/{settings.project_id}/locations/{handle.region}"
            try:
                client.delete_batch(
                    name=f"{parent}/batches/{handle.native_id}", timeout=_PROBE_TIMEOUT_S
                )
            except NotFound:
                return CancelResult(stopped=False, already_gone=True, detail="batch already gone")
            return CancelResult(stopped=True, already_gone=False, detail="batch delete issued")
        except Exception as exc:  # noqa: BLE001 - cancel is advisory: report failure, never raise
            return _cancel_failure(exc)

    def _cancel_cluster(self, handle: ProbeHandle, *, settings: Settings) -> CancelResult:
        if not handle.native_id:
            # No server-assigned id yet (launch window) → nothing addressable to cancel.
            return CancelResult(
                stopped=False, already_gone=False, detail="cluster job id not yet assigned"
            )
        try:
            from google.api_core.exceptions import NotFound

            from ..cluster_telemetry import cancel_cluster_job

            try:
                cancel_cluster_job(
                    handle.region, handle.native_id, settings=settings, timeout=_PROBE_TIMEOUT_S
                )
            except NotFound:
                return CancelResult(
                    stopped=False, already_gone=True, detail="cluster job already gone"
                )
            return CancelResult(stopped=True, already_gone=False, detail="cluster job cancelled")
        except Exception as exc:  # noqa: BLE001 - cancel is advisory: report failure, never raise
            return _cancel_failure(exc)


class RayProbe:
    """Probe a Ray-on-Vertex job via its cluster's persistent-resource path.

    Checks the cluster first (`ray_cluster._get_cluster`): a gone cluster (``NotFound``) is
    NOT_FOUND (``exists=False``, "cluster torn down") — and short-circuits before the dashboard
    connect, which would otherwise retry through its warm-up budget against a dead endpoint. When
    the cluster is alive it reuses `ray_jobs._connect_job_client` + ``get_job_status`` (and
    ``get_job_info`` for the failure message). Any error degrades to UNKNOWN.
    """

    name = "ray"

    def check(self, handle: ProbeHandle, *, settings: Settings) -> ProbeResult:
        try:
            from google.api_core.exceptions import NotFound

            from ..ray_cluster import _get_cluster
            from ..ray_jobs import _connect_job_client

            resource_name = handle.resource_name
            if not resource_name:
                # No persistent-resource path → can't address the cluster; can't tell live state.
                return ProbeResult(
                    NATIVE_UNKNOWN, exists=True, detail="handle missing resource_name"
                )
            try:
                _get_cluster(resource_name)
            except NotFound:
                return ProbeResult(NATIVE_NOT_FOUND, exists=False, detail="ray cluster torn down")
            client = _connect_job_client(resource_name)
            status = str(client.get_job_status(handle.native_id))
            native = _RAY_JOB_STATES.get(status, NATIVE_UNKNOWN)
            detail = ""
            try:
                info = client.get_job_info(handle.native_id)
                detail = getattr(info, "message", None) or ""
            except Exception:  # noqa: BLE001 - job-info is a best-effort enrichment, not the state
                pass
            return ProbeResult(native, exists=True, detail=detail)
        except Exception as exc:  # noqa: BLE001 - a probe is advisory: degrade, never raise
            return ProbeResult(NATIVE_UNKNOWN, exists=True, detail=_short_detail(exc))

    def cancel(self, handle: ProbeHandle, *, settings: Settings) -> CancelResult:
        try:
            from google.api_core.exceptions import NotFound

            from ..ray_cluster import _get_cluster
            from ..ray_jobs import _connect_job_client

            resource_name = handle.resource_name
            if not resource_name:
                return CancelResult(
                    stopped=False, already_gone=False, detail="handle missing resource_name"
                )
            try:
                _get_cluster(resource_name)
            except NotFound:
                # The cluster is gone → the job is not running; nothing to stop (§4.1 NOT_FOUND).
                return CancelResult(
                    stopped=False, already_gone=True, detail="ray cluster already torn down"
                )
            client = _connect_job_client(resource_name)
            client.stop_job(handle.native_id)
            return CancelResult(stopped=True, already_gone=False, detail="ray job stop issued")
        except Exception as exc:  # noqa: BLE001 - cancel is advisory: report failure, never raise
            return _cancel_failure(exc)


class BigQueryProbe:
    """Probe the BigQuery-native family: the run's jobs share a deterministic id *prefix*.

    A native run submits several BigQuery jobs (one per statement) whose ids all start with the
    handle's ``native_id`` prefix (``id_kind="prefix"``), so a bare ``get_job(prefix)`` 404s — we
    ``list_jobs`` in the handle's region (``all_users=True`` so a cross-principal reader still sees
    them, bounded by the job's start time + a hard ceiling) and match the prefix, then roll the
    group up (`_rollup_bigquery_states`). No matches → NOT_FOUND (``exists=False``); err → UNKNOWN.
    """

    name = "bigquery"

    def check(self, handle: ProbeHandle, *, settings: Settings) -> ProbeResult:
        try:
            from google.cloud import bigquery

            client = bigquery.Client(project=settings.project_id, location=handle.region)
            # all_users=True so a probe run by a different principal than the submitter (e.g. a
            # laptop reattaching to a run the Composer runner SA launched) still sees the jobs.
            # Bounded by the job's start time (when we have it) and a hard ceiling, so a busy
            # project's history doesn't blow the advisory time budget.
            started = _parse_ts(handle.created_at)
            min_creation_time = (started - _BQ_SCAN_SKEW) if started is not None else None
            matched = [
                job
                for job in client.list_jobs(
                    all_users=True,
                    min_creation_time=min_creation_time,
                    max_results=_BQ_MAX_JOBS_SCAN,
                    timeout=_PROBE_TIMEOUT_S,
                )
                if (job.job_id or "").startswith(handle.native_id)
            ]
            if not matched:
                return ProbeResult(
                    NATIVE_NOT_FOUND, exists=False, detail="no matching bigquery jobs"
                )
            native = _rollup_bigquery_states(matched)
            return ProbeResult(native, exists=True, telemetry={"statement_count": len(matched)})
        except Exception as exc:  # noqa: BLE001 - a probe is advisory: degrade, never raise
            return ProbeResult(NATIVE_UNKNOWN, exists=True, detail=_short_detail(exc))

    def cancel(self, handle: ProbeHandle, *, settings: Settings) -> CancelResult:
        # A native family runs as several BigQuery jobs under a shared id prefix (no single id to
        # cancel), so resolve the live ones by prefix and cancel each. cancel_job is idempotent on a
        # job that already finished, so cancelling a just-completed statement is harmless.
        try:
            from google.cloud import bigquery

            client = bigquery.Client(project=settings.project_id, location=handle.region)
            started = _parse_ts(handle.created_at)
            min_creation_time = (started - _BQ_SCAN_SKEW) if started is not None else None
            live = [
                job
                for job in client.list_jobs(
                    all_users=True,
                    min_creation_time=min_creation_time,
                    max_results=_BQ_MAX_JOBS_SCAN,
                    timeout=_PROBE_TIMEOUT_S,
                )
                if (job.job_id or "").startswith(handle.native_id)
                and (getattr(job, "state", "") or "").upper() != "DONE"
            ]
            if not live:
                return CancelResult(
                    stopped=False, already_gone=True, detail="no live bigquery jobs"
                )
            for job in live:
                client.cancel_job(job.job_id, location=handle.region, timeout=_PROBE_TIMEOUT_S)
            return CancelResult(
                stopped=True, already_gone=False, detail=f"cancelled {len(live)} bigquery job(s)"
            )
        except Exception as exc:  # noqa: BLE001 - cancel is advisory: report failure, never raise
            return _cancel_failure(exc)


def _rollup_bigquery_states(jobs: list[Any]) -> str:
    """Collapse a BigQuery statement group (one job per statement) into one normalized state.

    Any statement still live (not ``DONE``) means the group is RUNNING; once all are terminal, a
    single ``error_result`` fails the whole group (worst-terminal-wins), else it SUCCEEDED.
    """
    any_failed = False
    for job in jobs:
        state = (getattr(job, "state", "") or "").upper()
        if state != "DONE":
            return NATIVE_RUNNING
        if getattr(job, "error_result", None):
            any_failed = True
    return NATIVE_FAILED if any_failed else NATIVE_SUCCEEDED


# Registered by ``runtime`` (a `ProbeHandle.runtime`). A new probe = one class + one entry here,
# mirroring `submitters._SUBMITTERS`.
_PROBES: dict[str, RuntimeProbe] = {
    SparkProbe.name: SparkProbe(),
    RayProbe.name: RayProbe(),
    BigQueryProbe.name: BigQueryProbe(),
}


def get_probe(runtime: str) -> RuntimeProbe:
    """The `vocabulary.RuntimeProbe` for a handle's ``runtime``.

    Raises `ConfigError` on an unknown one.
    """
    try:
        return _PROBES[runtime]
    except KeyError:
        raise ConfigError(
            f"no runtime probe for runtime={runtime!r}; known: {sorted(_PROBES)}"
        ) from None
