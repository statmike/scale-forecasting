"""Runtime probe coordinates + per-runtime live-state probes: the seam that lets a run reconcile
registry state against live runtime state.

A ``run_jobs`` row records a job's deterministic identity and its last-written ``status``, but a
crashed driver or a torn-down cluster can leave that status stale. To reconcile it, a reader needs
the *runtime* coordinates — which platform (Spark / Ray / BigQuery), the platform-native job id,
the region it landed in — captured while the job is launched. ``ProbeHandle`` is that coordinate
blob: built at launch, serialized under ``run_jobs.job_telemetry.$.probe_handle``, and parsed back
out when a reader wants to check a job against its runtime.

`RuntimeProbe` is the read seam that consumes a handle: one implementation per runtime
(`SparkProbe` / `RayProbe` / `BigQueryProbe`), registered by ``runtime`` and dispatched through
`get_probe` — mirroring `submitters.py`'s `RuntimeSubmitter` spine (all probes in one file, one
class + one registry entry per runtime). Each `check` maps the platform-native job state into a
closed normalized set (``RUNNING`` / ``SUCCEEDED`` / ``FAILED`` / ``NOT_FOUND`` / ``UNKNOWN``) so a
reader reconciles against one vocabulary. A probe is *advisory*: every ``check`` caps its native
calls with a short timeout and degrades to ``UNKNOWN`` on any error — it never raises and never
hangs, so a probe failure can never take down the reader that called it.

`ProbeHandle` stays a pure dataclass with dict (de)serialization only (no imports), and the
GCP/engine imports for `check` stay lazy inside each method, so importing this module never pulls
the Ray/Spark/BigQuery extras — only the probed runtime's path loads them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol

from .errors import ConfigError

if TYPE_CHECKING:
    from .review import FamilyProgress, RunProgress
    from .settings import Settings


# --- normalized runtime states (closed set) -----------------------------------
# Every probe maps its platform-native job state into this closed set, so a reader reconciles a job
# against the registry using one vocabulary regardless of which runtime ran it. NOT_FOUND is
# distinct from FAILED (the job/cluster is gone, not failed) and from UNKNOWN (we couldn't tell).
NATIVE_RUNNING = "RUNNING"
NATIVE_SUCCEEDED = "SUCCEEDED"
NATIVE_FAILED = "FAILED"
NATIVE_NOT_FOUND = "NOT_FOUND"
NATIVE_UNKNOWN = "UNKNOWN"

# A probe is advisory and must never block the reader that called it: cap every native client call
# that accepts a timeout at this ceiling, and degrade to UNKNOWN when one still fails.
_PROBE_TIMEOUT_S = 20.0

# BigQuery's per-run jobs share an id prefix, but a bare list_jobs walks the whole project history.
# Bound the scan by the job's start time (see `ProbeHandle.created_at`, minus a skew margin) and a
# hard ceiling, so a probe of a busy project stays within its advisory time budget instead of paging
# through unrelated history.
_BQ_MAX_JOBS_SCAN = 2000
_BQ_SCAN_SKEW = timedelta(minutes=5)

# How long a RUNNING row must have gone quiet before a *vanished* runtime job (native NOT_FOUND with
# its artifacts still incomplete) is judged LOST rather than merely still-starting: below this floor
# a not-yet-created runtime job reads as UNKNOWN, not a false LOST (the startup grace). Tunable per
# call via `probe_run(stale_after_s=...)` (G2).
_DEFAULT_STALE_S = 900.0

# Registry-side status literals — deliberately kept distinct from the normalized runtime-state
# vocabulary above (they share the "RUNNING" spelling but mean different things: a registry row's
# last-written status vs. a live native reading). `_TERMINAL` is a local dup of
# `sdk._TERMINAL_STATUSES` kept here to avoid an import cycle (probes is imported low, sdk high). A
# family in one of these is trusted outright — its work is done, so it is never escalated. CANCELLED
# is a terminal literal too (a stopped job is done): it short-circuits the probe and makes a
# re-cancel of an already-cancelled job a no-op.
_REGISTRY_RUNNING = "RUNNING"
_CANCELLED = "CANCELLED"
_TERMINAL = frozenset({"COMPLETED", "FAILED", "PARTIAL", _CANCELLED})


# --- reconciled verdicts (closed set) -----------------------------------------
# One verdict per family after fusing registry status + landed artifacts + (only when the family was
# escalated) the live native reading. When the native truth contradicts the registry, the verdict
# also carries a `disagreement` flag so a reader can surface exactly the rows that need attention.
VERDICT_TRUST_REGISTRY = "TRUST_REGISTRY"  # terminal / never-probed: registry is authoritative
VERDICT_RUNNING = "RUNNING_CONFIRMED"  # runtime confirms the job is still live
VERDICT_STALE_REGISTRY = "STALE_REGISTRY"  # runtime is terminal but the registry never caught up
VERDICT_LIKELY_COMPLETED = "LIKELY_COMPLETED"  # job gone + expected artifacts all landed
VERDICT_LOST = "LOST"  # job gone with artifacts missing / denominator unknown
VERDICT_UNKNOWN = "UNKNOWN"  # couldn't tell (no handle, or the probe itself degraded)


@dataclass(frozen=True)
class ProbeHandle:
    """The runtime coordinates for one family's job — the write-blob and read-parse shared type.

    ``native_id`` is the platform-native id (a Serverless ``batch_id``, a Ray ``submission_id``, a
    Dataproc *cluster* job's server-assigned id, or a BigQuery job-id prefix); it is ``""`` when the
    real id isn't known yet (a cluster job pre-stamp-back). ``region`` is where the job runs.
    ``id_kind`` is ``"prefix"`` for BigQuery (whose jobs share a deterministic id prefix) and
    ``"exact"`` otherwise; it is carried on the persisted blob so a reader (and the P5 cancel path)
    can tell a prefix match from an exact id without re-deriving it per runtime. ``spark_mode``
    (``"serverless"`` | ``"cluster"``) is set for Spark only; ``resource_name`` (the Ray persistent-
    resource path) for Ray only. ``created_at`` is *not* a persisted coordinate — it is hydrated
    from the job row at read time (`from_job_row`) purely to lower-bound the BigQuery history scan;
    it is excluded from `to_blob` and is ``None`` for non-BQ handles and pre-feature rows.
    """

    runtime: str
    native_id: str
    region: str
    id_kind: str = "exact"
    spark_mode: str | None = None
    resource_name: str | None = None
    created_at: Any = None

    def to_blob(self) -> dict[str, Any]:
        """The compact dict stored under ``run_jobs.job_telemetry.$.probe_handle``.

        Omits the runtime-specific fields (``spark_mode``/``resource_name``) when unset so a handle
        carries only the coordinates its runtime actually has.
        """
        blob: dict[str, Any] = {
            "runtime": self.runtime,
            "native_id": self.native_id,
            "region": self.region,
            "id_kind": self.id_kind,
        }
        if self.spark_mode is not None:
            blob["spark_mode"] = self.spark_mode
        if self.resource_name is not None:
            blob["resource_name"] = self.resource_name
        return blob

    @classmethod
    def from_job_row(cls, row: dict[str, Any]) -> ProbeHandle | None:
        """Parse the handle out of a ``v_run_jobs`` row, or ``None`` when there isn't a usable one.

        ``row["probe_handle"]`` is the ``JSON_QUERY`` string the view projects (or an already-parsed
        dict); ``None`` (a pre-feature run) or a malformed/incomplete blob returns ``None`` so a
        reader degrades to registry-only rather than raising.
        """
        raw = row.get("probe_handle")
        if not raw:
            return None
        blob = json.loads(raw) if isinstance(raw, str) else raw
        try:
            return cls(
                runtime=blob["runtime"],
                native_id=blob["native_id"],
                region=blob["region"],
                id_kind=blob.get("id_kind", "exact"),
                spark_mode=blob.get("spark_mode"),
                resource_name=blob.get("resource_name"),
                # Hydrated from the row (not the blob) to bound the BigQuery job-history scan.
                created_at=row.get("started_at") or row.get("created_at"),
            )
        except (KeyError, TypeError):
            return None


@dataclass(frozen=True)
class ProbeResult:
    """One runtime's live answer for a job: the normalized state + whether it still exists.

    ``native_state`` is one of the closed ``NATIVE_*`` set. ``exists`` is ``False`` only for
    ``NATIVE_NOT_FOUND`` (the job / cluster is gone) — a reader uses it to tell "torn down" apart
    from "failed". ``detail`` is a short human reason (a status message, or — when a probe
    degrades — a one-line ``Type: message`` summary of the swallowed exception, via
    `_short_detail`). ``telemetry`` carries any extra fields the
    probe cheaply gathered (Dataproc batch usage, statement counts); it is always a plain JSON-able
    dict, empty when the probe has nothing to add.
    """

    native_state: str
    exists: bool
    detail: str = ""
    telemetry: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CancelResult:
    """One runtime's answer to a *cancel* request for a job — the write-boundary evidence.

    ``stopped`` is ``True`` when the runtime accepted the stop (the job was live and is now being
    torn down). ``already_gone`` is ``True`` when there was nothing to stop — the batch/job/cluster
    was already ``NotFound`` (a Ray cluster GC'd on completion is the common case), which the caller
    treats as an effective cancel (the job is not running). Either flag ``True`` means the registry
    row may be finalized to ``CANCELLED``; when *both* are ``False`` the stop genuinely failed
    (permission denied, timeout) and the caller leaves the registry as-is and reports honestly.
    ``detail`` is a short human reason (a permission hint, a NotFound note, or a `_short_detail`
    of the swallowed exception). Like `check`, cancel is advisory: it never raises.
    """

    stopped: bool
    already_gone: bool
    detail: str = ""


class RuntimeProbe(Protocol):
    """Read (and, on request, stop) one family's runtime job from its `ProbeHandle`.

    ``name`` is the ``runtime`` key. ``check`` maps the platform-native job state into the closed
    ``NATIVE_*`` set and never raises: a probe is advisory, so any error (auth, timeout, a torn-down
    cluster the native call chokes on) degrades to ``ProbeResult(NATIVE_UNKNOWN, exists=True, ...)``
    rather than propagating. ``cancel`` issues the runtime's stop and returns a `CancelResult`; it
    likewise never raises (a failed stop is reported, not thrown).
    """

    name: str

    def check(self, handle: ProbeHandle, *, settings: Settings) -> ProbeResult: ...

    def cancel(self, handle: ProbeHandle, *, settings: Settings) -> CancelResult: ...


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
    """Map a swallowed cancel exception to a failed `CancelResult`, with a clear IAM hint.

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

    Serverless reuses `submit._batch_client` + ``get_batch`` (with `submit.extract_job_telemetry`
    for the usage overlay); cluster reuses the new `dataproc_cluster.get_cluster_job` non-blocking
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

            from .submit import _batch_client, extract_job_telemetry

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

            from .dataproc_cluster import get_cluster_job

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

            from .submit import _batch_client

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

            from .dataproc_cluster import cancel_cluster_job

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

    Checks the cluster first (`ray_submit._get_cluster`): a gone cluster (``NotFound``) is
    NOT_FOUND (``exists=False``, "cluster torn down") — and short-circuits before the dashboard
    connect, which would otherwise retry through its warm-up budget against a dead endpoint. When
    the cluster is alive it reuses `ray_submit._connect_job_client` + ``get_job_status`` (and
    ``get_job_info`` for the failure message). Any error degrades to UNKNOWN.
    """

    name = "ray"

    def check(self, handle: ProbeHandle, *, settings: Settings) -> ProbeResult:
        try:
            from google.api_core.exceptions import NotFound

            from .ray_submit import _connect_job_client, _get_cluster

            resource_name = handle.resource_name
            if not resource_name:
                # No persistent-resource path → can't address the cluster; can't tell live state.
                return ProbeResult(
                    NATIVE_UNKNOWN, exists=True, detail="handle missing resource_name"
                )
            try:
                _get_cluster(resource_name)
            except NotFound:
                return ProbeResult(
                    NATIVE_NOT_FOUND, exists=False, detail="ray cluster torn down"
                )
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

            from .ray_submit import _connect_job_client, _get_cluster

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
            return ProbeResult(
                native, exists=True, telemetry={"statement_count": len(matched)}
            )
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
    """The `RuntimeProbe` for a handle's ``runtime`` (raises `ConfigError` on an unknown one)."""
    try:
        return _PROBES[runtime]
    except KeyError:
        raise ConfigError(
            f"no runtime probe for runtime={runtime!r}; known: {sorted(_PROBES)}"
        ) from None


# --- reconciliation (pure) ----------------------------------------------------
# The layer above the probes: fuse a run's registry+artifact progress (`review.RunProgress`) with
# the live native readings (`ProbeResult`, only for families that were escalated) into one verdict
# per family. Pure and unit-tested per matrix row — the I/O caller (`probe_run`) does the reads and
# calls the probes, then hands the assembled inputs here.


@dataclass(frozen=True)
class FamilyVerdict:
    """The reconciled truth for one family: what the registry says vs. what the runtime says.

    ``registry_status`` is the family's last-written job status; ``native_state`` / ``exists`` are
    the live `ProbeResult` reading (``None`` for a family that wasn't escalated — terminal or never
    launched). ``verdict`` is one of the ``VERDICT_*`` set and ``disagreement`` is ``True`` only
    when the runtime contradicts the registry (a reader's "look here" flag). ``n_done`` /
    ``n_expected`` are the family's landed-vs-expected cell counts (the artifact evidence that
    splits a vanished job into ``LIKELY_COMPLETED`` vs. ``LOST``). ``detail`` is a short human
    reason; ``telemetry`` carries anything the probe cheaply gathered.
    """

    family: str
    runtime: str | None
    registry_status: str | None
    native_state: str | None
    exists: bool | None
    verdict: str
    disagreement: bool
    n_done: int
    n_expected: int | None
    detail: str
    telemetry: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProbeReport:
    """A run's reconciled snapshot: header status + one `FamilyVerdict` per family.

    ``escalated`` is ``True`` when at least one family was probed against its runtime (``False`` for
    a routine poll of an all-terminal run — the terminal short-circuit that keeps polling native-
    call-free). ``disagreement`` is the run-wide roll-up: ``True`` when any family's runtime
    contradicts its registry status.
    """

    run_id: str
    status: str | None
    escalated: bool
    families: tuple[FamilyVerdict, ...]
    disagreement: bool


def _parse_ts(value: Any) -> datetime | None:
    """Coerce a registry timestamp (a ``datetime`` from the BigQuery client, or an ISO string) to a
    timezone-aware ``datetime`` in UTC, or ``None`` when it isn't parseable — so staleness math
    never trips on a malformed value (it just declines to escalate)."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def _is_stale(row: dict[str, Any], now: datetime, stale_after_s: float | None) -> bool:
    """Whether a ``RUNNING`` job row has gone quiet long enough to be past its startup grace.

    A row is stale when its status is ``RUNNING`` and its last signal (``ended_at`` → ``started_at``
    → ``created_at``, first present wins) is older than ``stale_after_s`` (default
    `_DEFAULT_STALE_S`). A stale family whose runtime job has vanished (native NOT_FOUND, artifacts
    incomplete) is judged LOST; a *young* one is still-starting, so it reads UNKNOWN (the grace that
    stops a probe crying wolf during a normal launch window). Any non-``RUNNING`` status is never
    stale. Pure and defensive: an unparseable timestamp is treated as *not* stale, never raising.
    """
    if (row.get("status") or "").upper() != _REGISTRY_RUNNING:
        return False
    threshold = _DEFAULT_STALE_S if stale_after_s is None else stale_after_s
    ts = _parse_ts(row.get("ended_at") or row.get("started_at") or row.get("created_at"))
    if ts is None:
        return False
    return (now - ts).total_seconds() > threshold


def _verdict_for_family(
    fp: FamilyProgress,
    native: dict[str, ProbeResult],
    no_handle: frozenset[str],
    stale: frozenset[str],
) -> FamilyVerdict:
    """The verdict matrix for one `review.FamilyProgress`, as a single if-ladder.

    ``fp`` is a `review.FamilyProgress`; ``native`` holds a `ProbeResult` only for families that
    were escalated; ``no_handle`` names families escalated but with no usable `ProbeHandle`;
    ``stale`` names families past their startup grace (a vanished job in this set is LOST, a young
    one is still-starting → UNKNOWN).
    """
    common: dict[str, Any] = {
        "family": fp.family,
        "runtime": fp.runtime,
        "registry_status": fp.status,
        "n_done": fp.n_done,
        "n_expected": fp.n_expected,
    }
    # Terminal → the registry is authoritative; the family was never probed (short-circuit).
    if (fp.status or "") in _TERMINAL:
        return FamilyVerdict(
            **common, native_state=None, exists=None,
            verdict=VERDICT_TRUST_REGISTRY, disagreement=False, detail="",
        )
    # Non-terminal but escalated with no usable handle (pre-feature / malformed blob) → can't tell.
    if fp.family in no_handle:
        return FamilyVerdict(
            **common, native_state=None, exists=None,
            verdict=VERDICT_UNKNOWN, disagreement=False, detail="no handle recorded",
        )
    result = native.get(fp.family)
    # Non-terminal and not escalated (no job row yet — never launched) → nothing to reconcile.
    if result is None:
        return FamilyVerdict(
            **common, native_state=None, exists=None,
            verdict=VERDICT_TRUST_REGISTRY, disagreement=False, detail="",
        )
    ns = result.native_state
    artifacts_complete = fp.n_expected is not None and fp.n_done >= fp.n_expected
    detail = result.detail
    if ns == NATIVE_RUNNING:
        verdict, disagreement = VERDICT_RUNNING, False
    elif ns == NATIVE_FAILED:
        # A failed runtime job is authoritative — the registry never recorded the terminal state.
        verdict, disagreement = VERDICT_STALE_REGISTRY, True
    elif ns == NATIVE_SUCCEEDED:
        # "Succeeded" is only trustworthy when the artifacts corroborate it. A native family's
        # BigQuery statements go DONE one-by-one, so an all-DONE reading mid-run is a lull between
        # statements, not the end: complete artifacts ⇒ the registry is genuinely stale, otherwise
        # it's ambiguous (don't overrule the registry on a transient all-DONE).
        if artifacts_complete:
            verdict, disagreement = VERDICT_STALE_REGISTRY, True
        else:
            verdict, disagreement = VERDICT_UNKNOWN, False
            detail = detail or "runtime reports success but artifacts are incomplete"
    elif ns == NATIVE_NOT_FOUND:
        # The job/cluster is gone. Complete artifacts ⇒ it finished; otherwise a job past its
        # startup grace (stale) is LOST, while a young one is just still starting (a RUNNING row is
        # written before the native job exists, so a fresh probe legitimately 404s → UNKNOWN, not a
        # false LOST).
        if artifacts_complete:
            verdict, disagreement = VERDICT_LIKELY_COMPLETED, True
        elif fp.family in stale:
            verdict, disagreement = VERDICT_LOST, True
        else:
            verdict, disagreement = VERDICT_UNKNOWN, False
            detail = detail or "runtime has no record yet; job may still be starting"
    else:  # NATIVE_UNKNOWN — the probe degraded; don't overrule the registry.
        verdict, disagreement = VERDICT_UNKNOWN, False
    return FamilyVerdict(
        **common, native_state=ns, exists=result.exists,
        verdict=verdict, disagreement=disagreement,
        detail=detail, telemetry=result.telemetry,
    )


def _assemble_probe_report(
    progress: RunProgress,
    native: dict[str, ProbeResult],
    no_handle: frozenset[str],
    stale: frozenset[str] = frozenset(),
) -> ProbeReport:
    """Fuse registry+artifact `review.RunProgress` with live native readings into a `ProbeReport`.

    Pure: ``native`` carries a `ProbeResult` only for the escalated families and ``no_handle`` names
    the escalated ones that had no usable handle; ``stale`` names families past their startup grace
    (used to split a vanished job into LOST vs. still-starting); every other family reconciles from
    the registry alone. ``escalated`` reflects whether any family was probed; ``disagreement`` rolls
    up the per-family flags.
    """
    families = tuple(
        _verdict_for_family(fp, native, no_handle, stale) for fp in progress.families
    )
    return ProbeReport(
        run_id=progress.run_id,
        status=progress.status,
        escalated=bool(native),
        families=families,
        disagreement=any(f.disagreement for f in families),
    )


# --- I/O caller ----------------------------------------------------------------
# The thin reader that turns a run_id into a ProbeReport: read the registry (header + config + job
# rows + landed-cell counts), escalate only the incomplete/stale jobs to their runtime, then hand
# the assembled inputs to the pure `_assemble_probe_report`. Mirrors `review.monitor_run`'s
# read-then-assemble seam; all GCP imports stay lazy inside the function.


def _read_and_probe(
    run_id: str,
    *,
    job: str | None,
    settings: Settings,
    stale_after_s: float | None,
) -> tuple[ProbeReport, list[dict[str, Any]]]:  # pragma: no cover - GCP I/O
    """Read a run's registry state, escalate the incomplete jobs, and reconcile → `(report, rows)`.

    The shared read+probe body behind both `probe_run` (which returns just the report) and
    `cancel_run` (which also needs the ``v_run_jobs`` rows for their ``job_id`` + ``probe_handle``
    to finalize). ``rows`` is the run's job rows filtered to ``job`` (all families when ``None``);
    the report is the reconciled `ProbeReport` over the same set.
    """
    from .config import RunConfig
    from .registry import bq
    from .review import _assemble_progress

    summary = bq.read_run_summary(run_id, settings=settings)
    raw = bq.read_run_config(run_id, settings=settings)
    cfg = RunConfig.model_validate(raw) if raw else None
    job_rows = bq.read_run_jobs(run_id, settings=settings) if cfg else []
    progress_rows = bq.read_progress(run_id, settings=settings) if cfg else []
    progress = _assemble_progress(run_id, summary, cfg, job_rows, progress_rows)

    # --job narrows both the escalation and the report; a typo must fail loudly (else it would
    # silently report nothing) — validate against the run's actual families before filtering.
    if job is not None:
        known = {f.family for f in progress.families}
        if job not in known:
            raise ConfigError(f"unknown family {job!r}; run {run_id} has: {sorted(known)}")
        progress = replace(
            progress, families=tuple(f for f in progress.families if f.family == job)
        )

    rows = [r for r in job_rows if job is None or r["family"] == job]
    now = datetime.now(UTC)
    # Escalate every non-terminal job to its runtime; terminal rows short-circuit to the registry.
    to_probe = [r for r in rows if r.get("status") not in _TERMINAL]
    # A RUNNING row quiet longer than the floor is "stale" — past its startup grace, so a vanished
    # runtime job is judged LOST rather than still-starting (see `_verdict_for_family`).
    stale = frozenset(r["family"] for r in to_probe if _is_stale(r, now, stale_after_s))
    native: dict[str, ProbeResult] = {}
    no_handle: set[str] = set()
    for r in to_probe:
        handle = ProbeHandle.from_job_row(r)
        if handle is None:
            no_handle.add(r["family"])
            continue
        native[r["family"]] = get_probe(handle.runtime).check(handle, settings=settings)
    report = _assemble_probe_report(progress, native, frozenset(no_handle), stale)
    return report, rows


def probe_run(
    run_id: str,
    *,
    job: str | None = None,
    settings: Settings | None = None,
    stale_after_s: float | None = None,
) -> ProbeReport:  # pragma: no cover - GCP I/O
    """Reconcile a run's registry state against live runtime state → a `ProbeReport`.

    Reads the run's header + config + job rows + landed-cell counts (`registry.bq`), assembles the
    registry-side progress (`review._assemble_progress`), then escalates **only** the non-terminal
    jobs to their runtime — a routine poll of an already-terminal run touches no runtime (empty
    ``to_probe`` ⇒ ``escalated=False``). ``job`` narrows *both* the escalation and the report to one
    family (the per-family drill-down; an unknown name raises `ConfigError` listing the valid ones);
    ``settings`` is the GCP identity (from the ``SF_*`` env when ``None``); ``stale_after_s``
    overrides the startup-grace floor (`_DEFAULT_STALE_S`) that decides whether a vanished young job
    reads LOST or still-starting. A family whose handle can't be parsed (a pre-feature or malformed
    row) degrades to registry-only via ``no_handle`` rather than raising.
    """
    from .settings import Settings

    s = settings if settings is not None else Settings.resolve()
    report, _rows = _read_and_probe(run_id, job=job, settings=s, stale_after_s=stale_after_s)
    return report


# --- cancel: blast-radius plan, audit, header roll-up (pure) ------------------
# Cancel is destructive and outward-facing, so it is built around *honesty about what stops and what
# survives* (§7 of the design). The pure pieces here — the blast-radius plan, the audit blob, the
# header roll-up — are unit-tested per row; the I/O orchestrator (`cancel_run`) probes first, shows
# the plan, gates on explicit confirmation, then stops each runtime job and finalizes the registry.


@dataclass(frozen=True)
class CancelPlanItem:
    """One family's line in the blast-radius preview: what will happen and what data is retained.

    ``cancellable`` is keyed on the *registry* status (non-terminal ⇒ will cancel) so the plan is
    robust even when the live probe degraded; ``native_state`` enriches the note when we have it.
    ``n_done``/``n_expected`` are the landed-vs-expected cells — the partial results a cancel
    **retains** (never deletes). ``note`` is the human one-liner shown in the preview.
    """

    family: str
    runtime: str | None
    registry_status: str | None
    native_state: str | None
    cancellable: bool
    n_done: int
    n_expected: int | None
    note: str


@dataclass(frozen=True)
class CancelPlan:
    """The full blast radius of a cancel: one `CancelPlanItem` per family + the ensemble impact.

    ``ensemble_suppressed`` is ``True`` when cancelling a base family will cause the run's ensemble
    node to be skipped (§7.4 — a cancelled base family suppresses the ensemble rather than producing
    a lopsided one). ``n_cancellable`` is how many jobs a confirmed cancel would actually stop.
    """

    run_id: str
    items: tuple[CancelPlanItem, ...]
    ensemble_suppressed: bool

    @property
    def n_cancellable(self) -> int:
        return sum(1 for i in self.items if i.cancellable)


@dataclass(frozen=True)
class CancelOutcome:
    """What actually happened to one family's job on a confirmed cancel.

    ``requested`` is ``True`` when a stop was issued to the runtime (``False`` when we couldn't even
    address the job — no handle). ``cancelled`` is ``True`` when the registry row was finalized to
    ``CANCELLED`` (the runtime confirmed the stop, or the job was already gone). ``stopped`` /
    ``already_gone`` are the raw `CancelResult` flags; ``detail`` is the short reason.
    """

    family: str
    job_key: str
    requested: bool
    cancelled: bool
    stopped: bool
    already_gone: bool
    detail: str


@dataclass(frozen=True)
class CancelReport:
    """The result of a cancel call — a preview (``executed=False``) or the executed outcome.

    ``plan`` is the blast radius (always present — it is what a no-``confirm`` call returns).
    ``outcomes`` is empty for a preview and one `CancelOutcome` per attempted family otherwise.
    ``header_status`` is the run's rolled-up status after the cancel (``None`` when the header was
    not changed — e.g. a preview, or a cancel whose stops all failed). ``actor`` / ``reason`` are
    the audit identity recorded on the cancelled rows.
    """

    run_id: str
    plan: CancelPlan
    executed: bool
    outcomes: tuple[CancelOutcome, ...]
    header_status: str | None
    actor: str | None
    reason: str


def _assemble_cancel_plan(report: ProbeReport) -> CancelPlan:
    """Turn a reconciled `ProbeReport` into the blast-radius plan (pure).

    A family is cancellable when its registry status is non-terminal (`_TERMINAL` includes
    ``CANCELLED``, so an already-cancelled job is a no-op). The note states what a cancel retains
    (landed cells) and, when the runtime already vanished, says so. The ensemble node is flagged as
    suppressed when any *base* family will be cancelled.
    """
    base_cancelled = any(
        fv.family != "ensemble" and (fv.registry_status or "") not in _TERMINAL
        for fv in report.families
    )
    has_ensemble = any(fv.family == "ensemble" for fv in report.families)
    ensemble_suppressed = base_cancelled and has_ensemble
    items: list[CancelPlanItem] = []
    for fv in report.families:
        cancellable = (fv.registry_status or "") not in _TERMINAL
        exp = fv.n_expected if fv.n_expected is not None else "?"
        if not cancellable:
            note = f"already {fv.registry_status or 'terminal'}, untouched"
        elif fv.family == "ensemble" and ensemble_suppressed:
            note = "will be SKIPPED (a base family is cancelled)"
        else:
            note = f"will cancel; {fv.n_done}/{exp} series landed (retained)"
            if fv.native_state == NATIVE_NOT_FOUND:
                note += "; runtime job already gone"
        items.append(
            CancelPlanItem(
                family=fv.family,
                runtime=fv.runtime,
                registry_status=fv.registry_status,
                native_state=fv.native_state,
                cancellable=cancellable,
                n_done=fv.n_done,
                n_expected=fv.n_expected,
                note=note,
            )
        )
    return CancelPlan(
        run_id=report.run_id, items=tuple(items), ensemble_suppressed=ensemble_suppressed
    )


def _build_cancel_audit(
    *,
    actor: str | None,
    cancelled_at: datetime,
    reason: str,
    native_state: str | None,
    n_done: int,
) -> dict[str, Any]:
    """The audit blob stamped under ``run_jobs.job_telemetry.$.cancel`` (pure).

    Captures *who / when / why* plus the *observed reality* at cancel time (the live native state
    and how many series had landed) so the trail records both intent and what was stopped (§9).
    """
    return {
        "cancelled_by": actor,
        "cancelled_at": cancelled_at.isoformat(),
        "reason": reason,
        "native_state_at_cancel": native_state,
        "n_done_at_cancel": n_done,
    }


def _roll_header_after_cancel(statuses: list[str | None]) -> str | None:
    """Roll a run's family statuses into its post-cancel header status (pure).

    Every family ``CANCELLED`` ⇒ the whole run is ``CANCELLED``; a mix of ``CANCELLED`` with other
    terminals ⇒ ``PARTIAL`` (some work finished, some was stopped). If any family is still
    non-terminal (a stop that failed, or an untouched live job) the header is left unchanged
    (``None``) — we never finalize a run whose jobs aren't all settled.
    """
    if not statuses:
        return None
    if any((s or "") not in _TERMINAL for s in statuses):
        return None
    normalized = [(s or "").upper() for s in statuses]
    if _CANCELLED not in normalized:
        return None
    if all(s == _CANCELLED for s in normalized):
        return _CANCELLED
    return "PARTIAL"


# --- cancel I/O orchestrator ---------------------------------------------------


def _resolve_actor(settings: Settings) -> str | None:  # pragma: no cover - ADC identity I/O
    """Best-effort ADC principal for the cancel audit trail — a service-account email when the
    credentials carry one, else ``None`` (a user-ADC laptop has no cheap email without a token
    call). P6 expands this; here it never raises and never blocks a cancel."""
    try:
        import google.auth

        creds, _ = google.auth.default()
        return getattr(creds, "service_account_email", None) or None
    except Exception:  # noqa: BLE001 - actor resolution is best-effort; a cancel proceeds without it
        return None


def _finalize_cancelled(
    row: dict[str, Any],
    handle: ProbeHandle,
    item: CancelPlanItem,
    *,
    actor: str | None,
    cancelled_at: datetime,
    reason: str,
    settings: Settings,
) -> None:  # pragma: no cover - GCP I/O
    """Finalize one job row to ``CANCELLED`` with the audit blob merged into ``job_telemetry``.

    A non-terminal job row's ``job_telemetry`` holds only its ``probe_handle`` (sizing telemetry
    goes to the header, not the job row), so we rebuild it from the handle we parsed plus the new
    ``$.cancel`` audit blob — preserving the handle so the cancelled attempt stays reconcilable.
    """
    from .registry import bq

    audit = _build_cancel_audit(
        actor=actor,
        cancelled_at=cancelled_at,
        reason=reason,
        native_state=item.native_state,
        n_done=item.n_done,
    )
    telemetry: dict[str, Any] = {"probe_handle": handle.to_blob(), "cancel": audit}
    started = _parse_ts(row.get("started_at"))
    fields: dict[str, Any] = {
        "status": _CANCELLED, "ended_at": cancelled_at, "job_telemetry": telemetry
    }
    if started is not None:
        fields["runtime_seconds"] = (cancelled_at - started).total_seconds()
    bq.update_job(row["job_id"], settings=settings, **fields)


def cancel_run(
    run_id: str,
    *,
    job: str | None = None,
    confirm: bool = False,
    reason: str = "",
    actor: str | None = None,
    settings: Settings | None = None,
    stale_after_s: float | None = None,
) -> CancelReport:  # pragma: no cover - GCP I/O
    """Cancel a run (or one family) — preview unless ``confirm``; finalize registry to CANCELLED.

    Probes the run first (`_read_and_probe`) to reconcile live state, builds the blast-radius
    `CancelPlan`, and — **only when ``confirm``** — stops each cancellable family's runtime job and
    finalizes its row to ``CANCELLED`` with an audit blob (`_finalize_cancelled`), then rolls the
    run header up (`_roll_header_after_cancel`). Without ``confirm`` it returns the plan, touches no
    runtime and no registry (the CLI prints the preview and exits). ``job`` narrows to one family;
    ``reason`` and ``actor`` (default: resolved from ADC) are recorded for audit. Cancel **never
    deletes** — landed partial results are retained and labelled by the CANCELLED status + counts.
    """
    from .settings import Settings

    s = settings if settings is not None else Settings.resolve()
    report, rows = _read_and_probe(run_id, job=job, settings=s, stale_after_s=stale_after_s)
    plan = _assemble_cancel_plan(report)
    if not confirm:
        return CancelReport(
            run_id=run_id, plan=plan, executed=False, outcomes=(),
            header_status=None, actor=None, reason=reason,
        )

    resolved_actor = actor if actor is not None else _resolve_actor(s)
    cancelled_at = datetime.now(UTC)
    rows_by_family = {r["family"]: r for r in rows}
    outcomes: list[CancelOutcome] = []
    for item in plan.items:
        if not item.cancellable:
            continue
        r = rows_by_family.get(item.family)
        if r is None:
            continue
        handle = ProbeHandle.from_job_row(r)
        if handle is None:
            outcomes.append(CancelOutcome(
                family=item.family, job_key=r["job_id"], requested=False, cancelled=False,
                stopped=False, already_gone=False,
                detail="no handle recorded; cannot address the runtime job",
            ))
            continue
        result = get_probe(handle.runtime).cancel(handle, settings=s)
        finalized = result.stopped or result.already_gone
        if finalized:
            _finalize_cancelled(
                r, handle, item, actor=resolved_actor,
                cancelled_at=cancelled_at, reason=reason, settings=s,
            )
        outcomes.append(CancelOutcome(
            family=item.family, job_key=r["job_id"], requested=True, cancelled=finalized,
            stopped=result.stopped, already_gone=result.already_gone, detail=result.detail,
        ))

    # Re-read every family (not just the --job subset) so the header reflects the true post-cancel
    # state — a per-family cancel makes a multi-family run PARTIAL, a whole-run cancel CANCELLED.
    from .registry import bq

    all_statuses = [r.get("status") for r in bq.read_run_jobs(run_id, settings=s)]
    header_status = _roll_header_after_cancel(all_statuses)
    if header_status is not None:
        bq.update_header(run_id, settings=s, status=header_status)
    return CancelReport(
        run_id=run_id, plan=plan, executed=True, outcomes=tuple(outcomes),
        header_status=header_status, actor=resolved_actor, reason=reason,
    )
