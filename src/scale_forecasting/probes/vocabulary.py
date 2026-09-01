"""The closed vocabularies every probe layer speaks — states, verdicts, handles, results.

The lowest layer of this package, and it exists because all three layers above it must agree on
exactly these words. `runtimes` produces them, `reconcile` fuses them into verdicts, and `cancel`
acts on those verdicts; putting any of this in one of those modules would make the other two
import it and cycle.

Two vocabularies live here and they are deliberately distinct even where they share a spelling.
The ``NATIVE_*`` set is what a *runtime* says about a job right now. The registry literals
(`_REGISTRY_RUNNING`, `_CANCELLED`, `_TERMINAL`) are what a ``run_jobs`` row last *wrote*. Both
spell "RUNNING", and conflating them is precisely the bug this package exists to catch: a stale
registry row that still says RUNNING for a job whose runtime is long gone.

`ProbeHandle` is the coordinate blob that makes reconciliation possible at all — built at launch,
serialized under ``run_jobs.job_telemetry.$.probe_handle``, parsed back out by a reader. It stays
a pure dataclass with dict (de)serialization only, no GCP imports, so importing this module never
pulls a cloud extra.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from ..settings import Settings


# --- normalized runtime states (closed set) -----------------------------------
# Every probe maps its platform-native job state into this closed set, so a reader reconciles a job
# against the registry using one vocabulary regardless of which runtime ran it. NOT_FOUND is
# distinct from FAILED (the job/cluster is gone, not failed) and from UNKNOWN (we couldn't tell).
NATIVE_RUNNING = "RUNNING"
NATIVE_SUCCEEDED = "SUCCEEDED"
NATIVE_FAILED = "FAILED"
NATIVE_NOT_FOUND = "NOT_FOUND"
NATIVE_UNKNOWN = "UNKNOWN"

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
    `runtimes._short_detail`). ``telemetry`` carries any extra fields the
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
    ``detail`` is a short human reason (a permission hint, a NotFound note, or a
    `runtimes._short_detail` of the swallowed exception). Like `check`, cancel is advisory: it never
    raises.
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
