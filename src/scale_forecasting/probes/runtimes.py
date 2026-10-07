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

# The batch states past which there is nothing left to stop. Read by `_cancel_serverless` so a
# batch that finished between the plan read and the cancel is reported as gone, not as a failure.
_BATCH_TERMINAL = frozenset({NATIVE_SUCCEEDED, NATIVE_FAILED})

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

# Vertex AI ``CustomJob`` ``JobState``.
_VERTEX_CUSTOM_JOB_STATES = {
    "JOB_STATE_QUEUED": NATIVE_RUNNING,
    "JOB_STATE_PENDING": NATIVE_RUNNING,
    "JOB_STATE_RUNNING": NATIVE_RUNNING,
    "JOB_STATE_CANCELLING": NATIVE_RUNNING,
    "JOB_STATE_PAUSED": NATIVE_RUNNING,
    "JOB_STATE_UPDATING": NATIVE_RUNNING,
    "JOB_STATE_SUCCEEDED": NATIVE_SUCCEEDED,
    "JOB_STATE_FAILED": NATIVE_FAILED,
    "JOB_STATE_CANCELLED": NATIVE_FAILED,
    "JOB_STATE_EXPIRED": NATIVE_FAILED,
    "JOB_STATE_PARTIALLY_SUCCEEDED": NATIVE_FAILED,
}

# Vertex AI ``PipelineJob`` ``PipelineState``.
_VERTEX_PIPELINE_STATES = {
    "PIPELINE_STATE_QUEUED": NATIVE_RUNNING,
    "PIPELINE_STATE_PENDING": NATIVE_RUNNING,
    "PIPELINE_STATE_RUNNING": NATIVE_RUNNING,
    "PIPELINE_STATE_CANCELLING": NATIVE_RUNNING,
    "PIPELINE_STATE_PAUSED": NATIVE_RUNNING,
    "PIPELINE_STATE_SUCCEEDED": NATIVE_SUCCEEDED,
    "PIPELINE_STATE_FAILED": NATIVE_FAILED,
    "PIPELINE_STATE_CANCELLED": NATIVE_FAILED,
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
    *job-canceller* role (§9 of the runtime-probe design). We translate it to an actionable
    one-liner rather than surfacing a raw stack trace; every other error degrades to a
    `_short_detail` summary. Either way ``stopped=already_gone=False`` so the caller does not
    finalize the registry to CANCELLED."""
    from google.api_core.exceptions import Forbidden, PermissionDenied

    if isinstance(exc, PermissionDenied | Forbidden):
        return CancelResult(
            stopped=False,
            already_gone=False,
            detail="permission denied — cancel needs the job-canceller role "
            "(dataproc.batches.delete / dataproc.jobs.cancel / aiplatform.customJobs.cancel "
            "/ bigquery.jobs.update / Ray stop)",
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
        # An empty id used to be the normal state of a cluster row for the whole launch window,
        # because the path let Dataproc name the job and only stamped the id back afterwards. It
        # names its own job now (`cluster_submit.build_job`), so a handle written by
        # `job_launch.launch_family_job` always carries one and the read below is reachable during
        # the provisioning window — which is the window that matters, since that is where a killed
        # launcher leaves a row behind.
        #
        # The guard stays for the rows that predate the change and for any caller that builds a
        # handle by hand. Without an id we cannot address the job, and UNKNOWN is the honest answer:
        # a probe must never assert an id it does not truly have. Note what that costs, and why it
        # was worth removing from the common path — UNKNOWN is a permanent refusal by design, so a
        # row that lands here never ages into anything a repair verb will accept.
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
        # Stopping a Serverless batch means cancelling the *long-running operation* that created it,
        # not touching the batch resource. There is no `cancel_batch` RPC — `BatchControllerClient`
        # offers only create/get/list/delete, and REST `.../batches/{id}:cancel` is a 404 — so what
        # `gcloud dataproc batches cancel` really does is POST `operations/{id}:cancel` against the
        # operation the batch names in its own `operation` field. Deleting instead, as this did
        # until 2026-09-11, is wrong in both directions: a live batch refuses it outright ("Cannot
        # delete non-terminal batch", so cancel could never stop anything) and a finished one would
        # be destroyed along with the telemetry the registry still reads from it.
        try:
            from google.api_core.exceptions import NotFound

            from ..batch_telemetry import _batch_client

            client = _batch_client(handle.region)
            parent = f"projects/{settings.project_id}/locations/{handle.region}"
            try:
                batch = client.get_batch(
                    name=f"{parent}/batches/{handle.native_id}", timeout=_PROBE_TIMEOUT_S
                )
            except NotFound:
                return CancelResult(stopped=False, already_gone=True, detail="batch already gone")

            # The plan layer filters terminal families out before we get here, but a batch can
            # finish in the gap between that read and this one. Nothing to stop → already_gone,
            # which the caller reads as an effective cancel.
            state_name = getattr(getattr(batch, "state", None), "name", "")
            if _SPARK_BATCH_STATES.get(state_name, NATIVE_UNKNOWN) in _BATCH_TERMINAL:
                return CancelResult(
                    stopped=False, already_gone=True, detail=f"batch already {state_name.lower()}"
                )

            operation = getattr(batch, "operation", "") or ""
            if not operation:
                # The service stamps `operation` when it accepts the batch; it is empty only in the
                # sliver before that lands, where there is no running work to stop yet.
                return CancelResult(
                    stopped=False, already_gone=False, detail="batch operation not yet assigned"
                )
            try:
                client.transport.operations_client.cancel_operation(name=operation)
            except NotFound:
                return CancelResult(
                    stopped=False, already_gone=True, detail="batch operation already gone"
                )
            return CancelResult(stopped=True, already_gone=False, detail="batch cancel issued")
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
    ``get_job_info`` for the failure message). Errors degrade to UNKNOWN, with one exception below.

    **A living cluster that has no such job is NOT_FOUND, not UNKNOWN,** and the distinction is the
    difference between a hole that heals and one that never does. A launcher killed while its
    cluster provisions leaves the cluster up — so the "torn down" arm cannot fire — and a job id
    that was never submitted, so the dashboard 404s. Read as UNKNOWN that is a permanent refusal:
    the run header stays RUNNING because no verb may close it, and the reaper keeps sparing the
    cluster *because* the header says RUNNING. Read as NOT_FOUND (`ray_jobs._is_job_absent_error`,
    which demands two independent markers before it will say so) the existing ladder runs on its
    own: LOST past the startup grace → ``settle`` writes FAILED → ``close-runs`` closes the header
    → the reaper collects the cluster. Nothing about the verdict set changes; the adapter simply
    stops calling a fact an uncertainty.

    **`_init_vertex` first, always.** ``vertex_ray.get_ray_cluster`` takes no project or location —
    it reads the SDK's global config, which a probe process has never set. The launching process
    happens to have pinned it while creating the cluster, so this is invisible in-process and fails
    out-of-process, which is the only way a probe is ever actually run: the regional endpoint is
    unset, Vertex answers 404, and the ``except`` below turns a knowable state into UNKNOWN. Found
    live 2026-09-02 — the same call went 404 → ``RUNNING`` with nothing changed but the init.
    The handle's own region is used, not ``settings.region``, because a cluster may have hopped.
    """

    name = "ray"

    def check(self, handle: ProbeHandle, *, settings: Settings) -> ProbeResult:
        try:
            from google.api_core.exceptions import NotFound

            from ..ray_cluster import _get_cluster, _init_vertex
            from ..ray_jobs import _connect_job_client, _is_job_absent_error

            resource_name = handle.resource_name
            if not resource_name:
                # No persistent-resource path → can't address the cluster; can't tell live state.
                return ProbeResult(
                    NATIVE_UNKNOWN, exists=True, detail="handle missing resource_name"
                )
            _init_vertex(settings, handle.region or settings.region)
            try:
                _get_cluster(resource_name)
            except NotFound:
                return ProbeResult(NATIVE_NOT_FOUND, exists=False, detail="ray cluster torn down")
            client = _connect_job_client(resource_name)
            try:
                status = str(client.get_job_status(handle.native_id))
            except Exception as exc:  # noqa: BLE001 - one shape is a verdict, the rest degrade
                if not _is_job_absent_error(exc):
                    return ProbeResult(NATIVE_UNKNOWN, exists=True, detail=_short_detail(exc))
                return ProbeResult(
                    NATIVE_NOT_FOUND, exists=False, detail="cluster alive; ray job not on it"
                )
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

            from ..ray_cluster import _get_cluster, _init_vertex
            from ..ray_jobs import _connect_job_client

            resource_name = handle.resource_name
            if not resource_name:
                return CancelResult(
                    stopped=False, already_gone=False, detail="handle missing resource_name"
                )
            # Same reason as `check` — without this the SDK has no regional endpoint out-of-process,
            # and a cancel that cannot reach the cluster reports failure instead of stopping a job.
            _init_vertex(settings, handle.region or settings.region)
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


class VertexProbe:
    """Probe a Vertex AI ``CustomJob`` via ``JobServiceClient``.

    Before stamp-back (or during ``_attempt_free_of_taken_ids``), ``handle.native_id`` is the
    deterministic ``display_name`` (``sf-r-...``); after stamp-back, ``handle.native_id`` (and
    ``handle.resource_name``) is the server-assigned ``projects/.../locations/.../customJobs/...``
    resource name. ``_resolve_custom_job`` handles both: a full resource name is fetched directly
    via ``get_custom_job``, while a ``display_name`` is resolved via ``list_custom_jobs``.
    """

    name = "vertex"

    @staticmethod
    def _resolve_custom_job(client: Any, handle: ProbeHandle, settings: Settings) -> Any:
        from google.api_core.exceptions import NotFound

        target = handle.resource_name or handle.native_id
        if not target:
            raise NotFound("vertex custom job id not yet assigned")
        if target.startswith("projects/"):
            return client.get_custom_job(name=target, timeout=_PROBE_TIMEOUT_S)
        region = handle.region or settings.region
        parent = f"projects/{settings.project_id}/locations/{region}"
        matches = list(
            client.list_custom_jobs(
                request={"parent": parent, "filter": f'display_name="{target}"'},
                timeout=_PROBE_TIMEOUT_S,
            )
        )
        if not matches:
            raise NotFound("vertex custom job not found")
        return matches[0]

    def check(self, handle: ProbeHandle, *, settings: Settings) -> ProbeResult:
        try:
            from google.api_core.exceptions import NotFound

            from ..vertex_submit import _job_client

            region = handle.region or settings.region
            client = _job_client(region)
            try:
                job = self._resolve_custom_job(client, handle, settings)
            except NotFound:
                return ProbeResult(
                    NATIVE_NOT_FOUND, exists=False, detail="vertex custom job not found"
                )
            state = getattr(job, "state", None)
            state_name = getattr(state, "name", str(state))
            native = _VERTEX_CUSTOM_JOB_STATES.get(state_name, NATIVE_UNKNOWN)
            error = getattr(job, "error", None)
            detail = getattr(error, "message", "") or "" if error else ""
            return ProbeResult(native, exists=True, detail=detail)
        except Exception as exc:  # noqa: BLE001 - a probe is advisory: degrade, never raise
            return ProbeResult(NATIVE_UNKNOWN, exists=True, detail=_short_detail(exc))

    def cancel(self, handle: ProbeHandle, *, settings: Settings) -> CancelResult:
        try:
            from google.api_core.exceptions import NotFound

            from ..vertex_submit import _job_client

            region = handle.region or settings.region
            client = _job_client(region)
            try:
                job = self._resolve_custom_job(client, handle, settings)
            except NotFound:
                return CancelResult(
                    stopped=False, already_gone=True, detail="vertex custom job already gone"
                )
            state_name = getattr(getattr(job, "state", None), "name", "")
            if _VERTEX_CUSTOM_JOB_STATES.get(state_name, NATIVE_UNKNOWN) in _BATCH_TERMINAL:
                return CancelResult(
                    stopped=False,
                    already_gone=True,
                    detail=f"vertex custom job already {state_name.lower()}",
                )
            try:
                client.cancel_custom_job(name=job.name, timeout=_PROBE_TIMEOUT_S)
            except NotFound:
                return CancelResult(
                    stopped=False, already_gone=True, detail="vertex custom job already gone"
                )
            return CancelResult(
                stopped=True, already_gone=False, detail="vertex custom job cancel issued"
            )
        except Exception as exc:  # noqa: BLE001 - cancel is advisory: report failure, never raise
            return _cancel_failure(exc)


_GCE_RUNNING_STATES = frozenset({"PROVISIONING", "STAGING", "RUNNING", "REPAIRING"})
_GCE_TERMINAL_STATES = frozenset({"STOPPING", "STOPPED", "SUSPENDING", "SUSPENDED", "TERMINATED"})


class GceProbe:
    """Probe a Compute Engine single-VM job via its GCS status marker and GCE instance state.

    Because GCE VMs self-delete on container exit (`trap cleanup EXIT`), a completed GCE job's
    authoritative terminal state lives in its GCS status marker
    (`gs://<code_bucket>/runs/gce-status/<instance_name>.json`), while a running or provisioning VM
    is visible via ``instances.get``.
    """

    name = "gce"

    @staticmethod
    def _resolve_zone_and_name(handle: ProbeHandle, settings: Settings) -> tuple[str, str]:
        resource = handle.resource_name or ""
        if "/zones/" in resource and "/instances/" in resource:
            parts = resource.split("/")
            zone_idx = parts.index("zones") + 1
            inst_idx = parts.index("instances") + 1
            if zone_idx < len(parts) and inst_idx < len(parts):
                return parts[zone_idx], parts[inst_idx]
        region_or_zone = handle.region or settings.region
        zone = region_or_zone if region_or_zone.count("-") >= 2 else f"{region_or_zone}-a"
        return zone, handle.native_id

    @staticmethod
    def _resolve_status_bucket(settings: Settings) -> str:
        try:
            from ..batch_infra import BatchInfra

            return BatchInfra.resolve().code_bucket
        except Exception:  # noqa: BLE001
            w = getattr(settings, "warehouse_uri", "") or ""
            if w.startswith("gs://"):
                return w.removeprefix("gs://").split("/", 1)[0]
            return ""

    def check(self, handle: ProbeHandle, *, settings: Settings) -> ProbeResult:
        if not handle.native_id:
            return ProbeResult(NATIVE_NOT_FOUND, exists=False, detail="gce instance id not set")
        try:
            from ..gce_submit import get_instance, read_status_marker

            status_bucket = self._resolve_status_bucket(settings)
            marker = (
                read_status_marker(status_bucket, handle.native_id, project_id=settings.project_id)
                if status_bucket
                else None
            )
            if marker is not None:
                m_state = str(marker.get("status") or marker.get("state") or "").upper()
                exit_code = marker.get("exit_code")
                if m_state == "SUCCEEDED":
                    return ProbeResult(
                        NATIVE_SUCCEEDED,
                        exists=True,
                        detail="gce container exited 0",
                        telemetry=marker,
                    )
                if m_state == "FAILED":
                    return ProbeResult(
                        NATIVE_FAILED,
                        exists=True,
                        detail=f"gce container exited {exit_code}",
                        telemetry=marker,
                    )

            zone, instance_name = self._resolve_zone_and_name(handle, settings)
            inst = get_instance(settings.project_id, zone, instance_name, timeout=_PROBE_TIMEOUT_S)
            if inst is None:
                if marker is not None:
                    # Marker said RUNNING/BOOTING, but the VM is now gone without writing SUCCEEDED
                    # (e.g. preempted, OOM-killed, or TTL-deleted).
                    return ProbeResult(
                        NATIVE_FAILED,
                        exists=True,
                        detail="gce instance terminated before writing terminal marker",
                        telemetry=marker,
                    )
                return ProbeResult(NATIVE_NOT_FOUND, exists=False, detail="gce instance not found")

            status = str(inst.get("status") or "").upper()
            if status in _GCE_RUNNING_STATES:
                return ProbeResult(NATIVE_RUNNING, exists=True, detail=f"gce status={status}")
            if status in _GCE_TERMINAL_STATES:
                return ProbeResult(
                    NATIVE_FAILED,
                    exists=True,
                    detail=f"gce instance {status.lower()} without success marker",
                )
            return ProbeResult(NATIVE_UNKNOWN, exists=True, detail=f"gce status={status}")
        except Exception as exc:  # noqa: BLE001 - a probe is advisory: degrade, never raise
            return ProbeResult(NATIVE_UNKNOWN, exists=True, detail=_short_detail(exc))

    def cancel(self, handle: ProbeHandle, *, settings: Settings) -> CancelResult:
        if not handle.native_id:
            return CancelResult(stopped=False, already_gone=False, detail="gce instance id not set")
        try:
            from ..gce_submit import delete_instance

            zone, instance_name = self._resolve_zone_and_name(handle, settings)
            deleted = delete_instance(
                settings.project_id, zone, instance_name, timeout=_PROBE_TIMEOUT_S
            )
            if not deleted:
                return CancelResult(
                    stopped=False, already_gone=True, detail="gce instance already gone"
                )
            return CancelResult(
                stopped=True, already_gone=False, detail="gce instance deletion issued"
            )
        except Exception as exc:  # noqa: BLE001 - cancel is advisory: report failure, never raise
            return _cancel_failure(exc)


class GkeProbe:
    """Probe a Google Kubernetes Engine Job via the GKE cluster & K8s ``batch/v1`` Job API."""

    name = "gke"

    @staticmethod
    def _resolve_coordinates(handle: ProbeHandle, settings: Settings) -> tuple[str, str, str, str]:
        """Return ``(location, cluster_name, namespace, job_name)`` from ``handle`` (pure)."""
        from ..gke_submit import _ephemeral_cluster_name

        resource = handle.resource_name or ""
        parts = resource.split("/") if resource else []
        if "locations" in parts and "clusters" in parts:
            loc_idx = parts.index("locations") + 1
            cl_idx = parts.index("clusters") + 1
            ns_idx = parts.index("namespaces") + 1 if "namespaces" in parts else -1
            job_idx = parts.index("jobs") + 1 if "jobs" in parts else -1
            loc = parts[loc_idx] if loc_idx < len(parts) else (handle.region or settings.region)
            cl = parts[cl_idx] if cl_idx < len(parts) else _ephemeral_cluster_name(handle.native_id)
            ns = parts[ns_idx] if 0 < ns_idx < len(parts) else "default"
            jb = parts[job_idx] if 0 < job_idx < len(parts) else handle.native_id
            return loc, cl, ns, jb

        location = handle.region or settings.region
        cluster_name = ""
        try:
            from ..batch_infra import BatchInfra

            cluster_name = BatchInfra.resolve().gke_cluster_name or ""
        except Exception:  # noqa: BLE001
            cluster_name = ""
        if not cluster_name:
            cluster_name = _ephemeral_cluster_name(handle.native_id)
        return location, cluster_name, "default", handle.native_id

    def check(self, handle: ProbeHandle, *, settings: Settings) -> ProbeResult:
        if not handle.native_id:
            return ProbeResult(NATIVE_NOT_FOUND, exists=False, detail="gke job id not set")
        try:
            from ..gke_submit import GkeK8sClient, get_gke_cluster, k8s_job_state

            location, cluster_name, namespace, job_name = self._resolve_coordinates(
                handle, settings
            )
            cluster = get_gke_cluster(settings.project_id, location, cluster_name)
            if cluster is None:
                return ProbeResult(NATIVE_NOT_FOUND, exists=False, detail="gke cluster not found")
            c_status = str(cluster.get("status") or "").upper()
            if c_status == "PROVISIONING":
                return ProbeResult(NATIVE_RUNNING, exists=True, detail="gke cluster provisioning")
            if c_status in {"STOPPING", "ERROR", "DEGRADED"} and not cluster.get("endpoint"):
                return ProbeResult(
                    NATIVE_FAILED, exists=True, detail=f"gke cluster status={c_status}"
                )

            with GkeK8sClient(cluster) as k8s:
                job_obj = k8s.get_job(namespace, job_name)
            state, detail = k8s_job_state(job_obj)
            if state == "NOT_FOUND":
                return ProbeResult(NATIVE_NOT_FOUND, exists=False, detail=detail)
            if state == "SUCCEEDED":
                return ProbeResult(NATIVE_SUCCEEDED, exists=True, detail=detail)
            if state == "FAILED":
                return ProbeResult(NATIVE_FAILED, exists=True, detail=detail)
            return ProbeResult(NATIVE_RUNNING, exists=True, detail=detail)
        except Exception as exc:  # noqa: BLE001 - a probe is advisory: degrade, never raise
            return ProbeResult(NATIVE_UNKNOWN, exists=True, detail=_short_detail(exc))

    def cancel(self, handle: ProbeHandle, *, settings: Settings) -> CancelResult:
        if not handle.native_id:
            return CancelResult(stopped=False, already_gone=False, detail="gke job id not set")
        try:
            from ..gke_submit import GkeK8sClient, get_gke_cluster

            location, cluster_name, namespace, job_name = self._resolve_coordinates(
                handle, settings
            )
            cluster = get_gke_cluster(settings.project_id, location, cluster_name)
            if cluster is None:
                return CancelResult(
                    stopped=False, already_gone=True, detail="gke cluster already gone"
                )
            with GkeK8sClient(cluster) as k8s:
                existing = k8s.get_job(namespace, job_name)
                if existing is None:
                    return CancelResult(
                        stopped=False, already_gone=True, detail="gke job already gone"
                    )
                k8s.delete_job(namespace, job_name)
            return CancelResult(stopped=True, already_gone=False, detail="gke job deletion issued")
        except Exception as exc:  # noqa: BLE001 - cancel is advisory: report failure, never raise
            return _cancel_failure(exc)


class VertexAutoMLProbe:
    """Probe a Vertex AI AutoML / Tabular Workflow ``PipelineJob`` via ``PipelineServiceClient``."""

    name = "vertex_automl"

    @staticmethod
    def _resolve_pipeline_jobs(client: Any, handle: ProbeHandle, settings: Settings) -> list[Any]:
        from google.api_core.exceptions import NotFound

        target = handle.resource_name or handle.native_id
        if not target:
            raise NotFound("vertex automl pipeline job id not yet assigned")
        region = handle.region or settings.region
        parent = f"projects/{settings.project_id}/locations/{region}"
        if target.startswith("projects/"):
            if "/pipelineJobs/" in target:
                return [client.get_pipeline_job(name=target, timeout=_PROBE_TIMEOUT_S)]
            raise NotFound("vertex automl pipeline job not found")

        # Try exact pipelineJob name first, then prefix match on display_name.
        with_exact_name = f"{parent}/pipelineJobs/{target}"
        try:
            return [client.get_pipeline_job(name=with_exact_name, timeout=_PROBE_TIMEOUT_S)]
        except NotFound:
            pass

        matches = [
            job
            for job in client.list_pipeline_jobs(
                request={"parent": parent},
                timeout=_PROBE_TIMEOUT_S,
            )
            if (getattr(job, "display_name", "") or "").startswith(target)
            or (getattr(job, "name", "") or "").rsplit("/", 1)[-1].startswith(target)
        ]
        if not matches:
            raise NotFound("vertex automl pipeline job not found")
        return matches

    def check(self, handle: ProbeHandle, *, settings: Settings) -> ProbeResult:
        try:
            from google.api_core.exceptions import NotFound

            from ..automl_submit import _pipeline_client

            region = handle.region or settings.region
            client = _pipeline_client(region)
            try:
                jobs = self._resolve_pipeline_jobs(client, handle, settings)
            except NotFound:
                return ProbeResult(
                    NATIVE_NOT_FOUND,
                    exists=False,
                    detail="vertex automl pipeline job not found",
                )

            any_failed = False
            any_running = False
            detail = ""
            for job in jobs:
                state = getattr(job, "state", None)
                state_name = getattr(state, "name", str(state))
                norm = _VERTEX_PIPELINE_STATES.get(state_name, NATIVE_UNKNOWN)
                if norm == NATIVE_RUNNING:
                    any_running = True
                elif norm == NATIVE_FAILED:
                    any_failed = True
                    error = getattr(job, "error", None)
                    detail = getattr(error, "message", "") or detail if error else detail
            if any_running:
                return ProbeResult(NATIVE_RUNNING, exists=True, detail=detail)
            if any_failed:
                return ProbeResult(NATIVE_FAILED, exists=True, detail=detail)
            return ProbeResult(NATIVE_SUCCEEDED, exists=True, detail=detail)
        except Exception as exc:  # noqa: BLE001 - a probe is advisory: degrade, never raise
            return ProbeResult(NATIVE_UNKNOWN, exists=True, detail=_short_detail(exc))

    def cancel(self, handle: ProbeHandle, *, settings: Settings) -> CancelResult:
        try:
            from google.api_core.exceptions import NotFound

            from ..automl_submit import _pipeline_client

            region = handle.region or settings.region
            client = _pipeline_client(region)
            try:
                jobs = self._resolve_pipeline_jobs(client, handle, settings)
            except NotFound:
                return CancelResult(
                    stopped=False,
                    already_gone=True,
                    detail="vertex automl pipeline job already gone",
                )
            live = [
                job
                for job in jobs
                if _VERTEX_PIPELINE_STATES.get(
                    getattr(getattr(job, "state", None), "name", ""), NATIVE_UNKNOWN
                )
                not in _BATCH_TERMINAL
            ]
            if not live:
                return CancelResult(
                    stopped=False,
                    already_gone=True,
                    detail="vertex automl pipeline job already terminal",
                )
            for job in live:
                client.cancel_pipeline_job(name=job.name, timeout=_PROBE_TIMEOUT_S)
            return CancelResult(
                stopped=True,
                already_gone=False,
                detail=f"cancelled {len(live)} vertex automl pipeline job(s)",
            )
        except Exception as exc:  # noqa: BLE001 - cancel is advisory: report failure, never raise
            return _cancel_failure(exc)


# Registered by ``runtime`` (a `ProbeHandle.runtime`). A new probe = one class + one entry here,
# mirroring `submitters._SUBMITTERS`.
_PROBES: dict[str, RuntimeProbe] = {
    SparkProbe.name: SparkProbe(),
    RayProbe.name: RayProbe(),
    VertexProbe.name: VertexProbe(),
    VertexAutoMLProbe.name: VertexAutoMLProbe(),
    GceProbe.name: GceProbe(),
    GkeProbe.name: GkeProbe(),
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
