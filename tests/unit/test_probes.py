"""Offline tests for the runtime-probe seam (``scale_forecasting.probes``).

`ProbeHandle` is the coordinate blob captured at launch, written under
``run_jobs.job_telemetry.$.probe_handle``, and parsed back out of a ``v_run_jobs`` row; the first
block pins its serialization round-trip and defensive parse. `RuntimeProbe` / `SparkProbe` /
`RayProbe` / `BigQueryProbe` are the read seam that consumes a handle and returns a normalized
`ProbeResult`; the second block stubs each runtime's lazily-imported client factory so nothing
touches GCP, and pins the native→normalized state maps, the NOT_FOUND (`exists=False`) path, and the
degrade-to-UNKNOWN-never-raise contract.
"""

from __future__ import annotations

import json
import types
from datetime import UTC, datetime
from typing import Any

import pytest

from scale_forecasting.errors import ConfigError
from scale_forecasting.probes.cancel import (
    CancelOutcome,
    _assemble_cancel_plan,
    _build_cancel_audit,
    _cancel_steps,
    _roll_header_after_cancel,
)
from scale_forecasting.probes.reconcile import (
    _assemble_probe_report,
    _is_stale,
    _narrow_to_job,
)
from scale_forecasting.probes.runtimes import (
    _BQ_MAX_JOBS_SCAN,
    BigQueryProbe,
    RayProbe,
    SparkProbe,
    get_probe,
)
from scale_forecasting.probes.vocabulary import (
    NATIVE_FAILED,
    NATIVE_NOT_FOUND,
    NATIVE_RUNNING,
    NATIVE_SUCCEEDED,
    NATIVE_UNKNOWN,
    VERDICT_LIKELY_COMPLETED,
    VERDICT_LOST,
    VERDICT_RUNNING,
    VERDICT_STALE_REGISTRY,
    VERDICT_TRUST_REGISTRY,
    VERDICT_UNKNOWN,
    ProbeHandle,
    ProbeResult,
)
from scale_forecasting.review import FamilyProgress, RunProgress
from scale_forecasting.settings import Settings

_SETTINGS = Settings(
    project_id="proj-x",
    connection="proj-x.us-central1.conn",
    warehouse_uri="gs://bkt/warehouse",
)


def test_to_blob_spark_serverless_round_trips() -> None:
    handle = ProbeHandle(
        "spark",
        native_id="sf-run-abc-statistical-a1",
        region="us-central1",
        spark_mode="serverless",
    )
    blob = handle.to_blob()
    assert blob == {
        "runtime": "spark",
        "native_id": "sf-run-abc-statistical-a1",
        "region": "us-central1",
        "id_kind": "exact",
        "spark_mode": "serverless",
    }
    # A round-trip through the parse rebuilds the same handle.
    assert ProbeHandle.from_job_row({"probe_handle": blob}) == handle


def test_to_blob_ray_round_trips_with_resource_name() -> None:
    handle = ProbeHandle(
        "ray",
        native_id="job-1",
        region="us-west1",
        resource_name="projects/p/locations/us-west1/persistentResources/c",
    )
    blob = handle.to_blob()
    assert blob["resource_name"] == "projects/p/locations/us-west1/persistentResources/c"
    assert "spark_mode" not in blob  # omitted when None
    assert ProbeHandle.from_job_row({"probe_handle": blob}) == handle


def test_to_blob_bigquery_carries_prefix_id_kind() -> None:
    handle = ProbeHandle(
        "bigquery", native_id="sf-run-abc-native-a1-", region="us", id_kind="prefix"
    )
    blob = handle.to_blob()
    assert blob["id_kind"] == "prefix"
    # BigQuery has neither a spark mode nor a resource path.
    assert "spark_mode" not in blob
    assert "resource_name" not in blob


def test_to_blob_omits_none_spark_mode_and_resource_name() -> None:
    blob = ProbeHandle("ray", native_id="j", region="us-central1").to_blob()
    assert set(blob) == {"runtime", "native_id", "region", "id_kind"}


def test_from_job_row_parses_json_string_blob() -> None:
    # v_run_jobs projects the handle as a JSON_QUERY string; from_job_row parses it.
    raw = json.dumps(
        {"runtime": "ray", "native_id": "j", "region": "us-west1", "resource_name": "rn"}
    )
    handle = ProbeHandle.from_job_row({"probe_handle": raw})
    assert handle == ProbeHandle("ray", native_id="j", region="us-west1", resource_name="rn")


def test_from_job_row_parses_already_parsed_dict() -> None:
    blob = {"runtime": "spark", "native_id": "b1", "region": "us-central1", "spark_mode": "cluster"}
    handle = ProbeHandle.from_job_row({"probe_handle": blob})
    assert handle == ProbeHandle(
        "spark", native_id="b1", region="us-central1", spark_mode="cluster"
    )


def test_from_job_row_missing_handle_returns_none() -> None:
    assert ProbeHandle.from_job_row({}) is None  # pre-feature run: no column
    assert ProbeHandle.from_job_row({"probe_handle": None}) is None


def test_from_job_row_malformed_blob_returns_none() -> None:
    # Missing a required key → degrade to None (registry-only) rather than raise.
    assert ProbeHandle.from_job_row({"probe_handle": {"runtime": "spark"}}) is None
    # A non-mapping blob → None (the .get on a list would raise KeyError/TypeError path).
    assert ProbeHandle.from_job_row({"probe_handle": json.dumps(["not", "a", "dict"])}) is None


# --- get_probe registry --------------------------------------------------------


def test_get_probe_maps_runtime_to_implementation() -> None:
    assert isinstance(get_probe("spark"), SparkProbe)
    assert isinstance(get_probe("ray"), RayProbe)
    assert isinstance(get_probe("bigquery"), BigQueryProbe)


def test_get_probe_unknown_runtime_raises() -> None:
    with pytest.raises(ConfigError, match="no runtime probe"):
        get_probe("nope")


# --- SparkProbe: Serverless batch ----------------------------------------------


class _FakeBatch:
    """A minimal Dataproc ``Batch`` stand-in: only the fields the probe + telemetry reader touch."""

    def __init__(self, state_name: str, message: str = "") -> None:
        self.state = types.SimpleNamespace(name=state_name)
        self.state_message = message


class _FakeBatchClient:
    def __init__(self, batch: _FakeBatch | None = None, exc: Exception | None = None) -> None:
        self._batch = batch
        self._exc = exc
        self.seen: dict[str, Any] = {}
        self.deleted = False

    def get_batch(self, *, name: str, timeout: float) -> _FakeBatch:
        self.seen["name"] = name
        self.seen["timeout"] = timeout
        if self._exc is not None:
            raise self._exc
        assert self._batch is not None
        return self._batch

    def delete_batch(self, *, name: str, timeout: float) -> None:
        self.seen["delete_name"] = name
        self.seen["delete_timeout"] = timeout
        if self._exc is not None:
            raise self._exc
        self.deleted = True


def _serverless_handle() -> ProbeHandle:
    return ProbeHandle(
        "spark",
        native_id="sf-run-abc-statistical-a1",
        region="us-central1",
        spark_mode="serverless",
    )


def _patch_batch_client(monkeypatch: pytest.MonkeyPatch, client: _FakeBatchClient) -> None:
    import scale_forecasting.batch_telemetry as telemetry_mod

    monkeypatch.setattr(telemetry_mod, "_batch_client", lambda region: client)


@pytest.mark.parametrize(
    ("state_name", "expected"),
    [
        ("PENDING", NATIVE_RUNNING),
        ("RUNNING", NATIVE_RUNNING),
        ("CANCELLING", NATIVE_RUNNING),
        ("SUCCEEDED", NATIVE_SUCCEEDED),
        ("FAILED", NATIVE_FAILED),
        ("CANCELLED", NATIVE_FAILED),
        ("STATE_UNSPECIFIED", NATIVE_UNKNOWN),  # unrecognized → UNKNOWN, never a raise
    ],
)
def test_spark_serverless_maps_batch_state(
    monkeypatch: pytest.MonkeyPatch, state_name: str, expected: str
) -> None:
    client = _FakeBatchClient(batch=_FakeBatch(state_name, message="hi"))
    _patch_batch_client(monkeypatch, client)

    result = SparkProbe().check(_serverless_handle(), settings=_SETTINGS)

    assert result.native_state == expected
    assert result.exists is True
    # The probe addresses the batch by its regional resource name and caps the call with a timeout.
    assert client.seen["name"] == (
        "projects/proj-x/locations/us-central1/batches/sf-run-abc-statistical-a1"
    )
    assert client.seen["timeout"] > 0


def test_spark_serverless_not_found_sets_exists_false(monkeypatch: pytest.MonkeyPatch) -> None:
    from google.api_core.exceptions import NotFound

    _patch_batch_client(monkeypatch, _FakeBatchClient(exc=NotFound("gone")))

    result = SparkProbe().check(_serverless_handle(), settings=_SETTINGS)

    assert result.native_state == NATIVE_NOT_FOUND
    assert result.exists is False


def test_spark_serverless_error_degrades_to_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    # Any non-NotFound error (auth, transport) degrades to UNKNOWN — a probe never raises.
    _patch_batch_client(monkeypatch, _FakeBatchClient(exc=RuntimeError("boom")))

    result = SparkProbe().check(_serverless_handle(), settings=_SETTINGS)

    assert result.native_state == NATIVE_UNKNOWN
    assert result.exists is True
    assert "boom" in result.detail


# --- SparkProbe: cluster job ---------------------------------------------------


def _cluster_handle() -> ProbeHandle:
    return ProbeHandle(
        "spark", native_id="real-dataproc-job-id", region="us-west1", spark_mode="cluster"
    )


@pytest.mark.parametrize(
    ("state_name", "expected"),
    [
        ("PENDING", NATIVE_RUNNING),
        ("SETUP_DONE", NATIVE_RUNNING),
        ("RUNNING", NATIVE_RUNNING),
        ("DONE", NATIVE_SUCCEEDED),
        ("ERROR", NATIVE_FAILED),
        ("CANCELLED", NATIVE_FAILED),
        ("ATTEMPT_FAILURE", NATIVE_FAILED),
        ("STATE_UNSPECIFIED", NATIVE_UNKNOWN),
    ],
)
def test_spark_cluster_maps_job_state(
    monkeypatch: pytest.MonkeyPatch, state_name: str, expected: str
) -> None:
    import scale_forecasting.cluster_telemetry as telemetry_mod

    seen: dict[str, Any] = {}

    def _fake_get_cluster_job(region: str, job_id: str, **kw: Any) -> tuple[str, str]:
        seen["region"] = region
        seen["job_id"] = job_id
        seen["timeout"] = kw.get("timeout")
        return state_name, "detail-msg"

    monkeypatch.setattr(telemetry_mod, "get_cluster_job", _fake_get_cluster_job)

    result = SparkProbe().check(_cluster_handle(), settings=_SETTINGS)

    assert result.native_state == expected
    assert result.exists is True
    assert seen == {
        "region": "us-west1",
        "job_id": "real-dataproc-job-id",
        "timeout": pytest.approx(20.0),
    }


def test_spark_cluster_not_found_sets_exists_false(monkeypatch: pytest.MonkeyPatch) -> None:
    from google.api_core.exceptions import NotFound

    import scale_forecasting.cluster_telemetry as telemetry_mod

    def _raise(*a: Any, **kw: Any) -> tuple[str, str]:
        raise NotFound("no such job")

    monkeypatch.setattr(telemetry_mod, "get_cluster_job", _raise)

    result = SparkProbe().check(_cluster_handle(), settings=_SETTINGS)

    assert result.native_state == NATIVE_NOT_FOUND
    assert result.exists is False


def test_spark_cluster_error_degrades_to_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    import scale_forecasting.cluster_telemetry as telemetry_mod

    def _raise(*a: Any, **kw: Any) -> tuple[str, str]:
        raise RuntimeError("transport down")

    monkeypatch.setattr(telemetry_mod, "get_cluster_job", _raise)

    result = SparkProbe().check(_cluster_handle(), settings=_SETTINGS)

    assert result.native_state == NATIVE_UNKNOWN
    assert result.exists is True


def test_spark_cluster_empty_native_id_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    # A cluster job's id is server-assigned and only stamped back after submission, so the entry
    # handle carries native_id="" for the launch window. The probe must NOT call get_cluster_job("")
    # (that would 404 → a false NOT_FOUND/LOST); it reports UNKNOWN(exists=True) without any I/O.
    import scale_forecasting.cluster_telemetry as telemetry_mod

    def _must_not_call(*a: Any, **kw: Any) -> tuple[str, str]:
        raise AssertionError("get_cluster_job must not be called for an empty native_id")

    monkeypatch.setattr(telemetry_mod, "get_cluster_job", _must_not_call)
    handle = ProbeHandle("spark", native_id="", region="us-west1", spark_mode="cluster")

    result = SparkProbe().check(handle, settings=_SETTINGS)

    assert result.native_state == NATIVE_UNKNOWN
    assert result.exists is True
    assert "not yet assigned" in result.detail


# --- RayProbe ------------------------------------------------------------------


class _FakeRayJobClient:
    def __init__(self, status: str, message: str = "") -> None:
        self._status = status
        self._message = message
        self.stopped_id: str | None = None

    def get_job_status(self, job_id: str) -> str:
        return self._status

    def get_job_info(self, job_id: str) -> Any:
        return types.SimpleNamespace(message=self._message)

    def stop_job(self, job_id: str) -> bool:
        self.stopped_id = job_id
        return True


def _ray_handle() -> ProbeHandle:
    return ProbeHandle(
        "ray",
        native_id="job-1",
        region="us-west1",
        resource_name="projects/p/locations/us-west1/persistentResources/c",
    )


def _patch_ray(
    monkeypatch: pytest.MonkeyPatch,
    *,
    client: _FakeRayJobClient | None = None,
    cluster_exc: Exception | None = None,
) -> None:
    import scale_forecasting.ray_cluster as ray_cluster
    import scale_forecasting.ray_jobs as ray_jobs

    def _get_cluster(resource_name: str) -> Any:
        if cluster_exc is not None:
            raise cluster_exc
        return types.SimpleNamespace(name=resource_name)

    monkeypatch.setattr(ray_cluster, "_get_cluster", _get_cluster)
    if client is not None:
        monkeypatch.setattr(ray_jobs, "_connect_job_client", lambda rn: client)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("PENDING", NATIVE_RUNNING),
        ("RUNNING", NATIVE_RUNNING),
        ("SUCCEEDED", NATIVE_SUCCEEDED),
        ("FAILED", NATIVE_FAILED),
        ("STOPPED", NATIVE_FAILED),
        ("WEIRD", NATIVE_UNKNOWN),
    ],
)
def test_ray_maps_job_status(monkeypatch: pytest.MonkeyPatch, status: str, expected: str) -> None:
    _patch_ray(monkeypatch, client=_FakeRayJobClient(status, message="driver msg"))

    result = RayProbe().check(_ray_handle(), settings=_SETTINGS)

    assert result.native_state == expected
    assert result.exists is True
    if expected == NATIVE_FAILED:
        assert result.detail == "driver msg"


def test_ray_cluster_gone_is_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    from google.api_core.exceptions import NotFound

    _patch_ray(monkeypatch, cluster_exc=NotFound("cluster gone"))

    result = RayProbe().check(_ray_handle(), settings=_SETTINGS)

    assert result.native_state == NATIVE_NOT_FOUND
    assert result.exists is False


def test_ray_missing_resource_name_is_unknown() -> None:
    # A handle with no persistent-resource path can't address the cluster → UNKNOWN, not a crash.
    handle = ProbeHandle("ray", native_id="job-1", region="us-west1")
    result = RayProbe().check(handle, settings=_SETTINGS)
    assert result.native_state == NATIVE_UNKNOWN
    assert result.exists is True


def test_ray_connect_error_degrades_to_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    import scale_forecasting.ray_cluster as ray_cluster
    import scale_forecasting.ray_jobs as ray_jobs

    monkeypatch.setattr(ray_cluster, "_get_cluster", lambda rn: object())

    def _boom(rn: str) -> Any:
        raise RuntimeError("dashboard 524")

    monkeypatch.setattr(ray_jobs, "_connect_job_client", _boom)

    result = RayProbe().check(_ray_handle(), settings=_SETTINGS)

    assert result.native_state == NATIVE_UNKNOWN
    assert result.exists is True


# --- BigQueryProbe -------------------------------------------------------------


class _FakeBqJob:
    def __init__(self, job_id: str, state: str, error_result: Any = None) -> None:
        self.job_id = job_id
        self.state = state
        self.error_result = error_result


class _FakeBqClient:
    def __init__(self, jobs: list[_FakeBqJob]) -> None:
        self._jobs = jobs
        self.seen: dict[str, Any] = {}
        self.cancelled: list[str] = []

    def list_jobs(self, **kwargs: Any) -> list[_FakeBqJob]:
        self.seen.update(kwargs)
        return self._jobs

    def cancel_job(self, job_id: str, *, location: str, timeout: float | None = None) -> None:
        self.cancelled.append(job_id)
        self.seen["cancel_location"] = location


def _bq_handle() -> ProbeHandle:
    return ProbeHandle("bigquery", native_id="sf-run-abc-native-a1-", region="us", id_kind="prefix")


def _patch_bq(monkeypatch: pytest.MonkeyPatch, client: _FakeBqClient) -> None:
    from google.cloud import bigquery

    monkeypatch.setattr(bigquery, "Client", lambda project, location: client)


def test_bigquery_all_statements_done_is_succeeded(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeBqClient(
        [
            _FakeBqJob("sf-run-abc-native-a1-0", "DONE"),
            _FakeBqJob("sf-run-abc-native-a1-1", "DONE"),
            _FakeBqJob("other-run-xyz", "RUNNING"),  # different prefix → excluded
        ]
    )
    _patch_bq(monkeypatch, client)

    result = BigQueryProbe().check(_bq_handle(), settings=_SETTINGS)

    assert result.native_state == NATIVE_SUCCEEDED
    assert result.exists is True
    assert result.telemetry["statement_count"] == 2  # only the two prefix matches
    assert client.seen["timeout"] > 0
    # Cross-principal-visible + bounded: a probe from a non-submitter principal still sees the jobs,
    # and the scan is capped so a busy project's history can't blow the advisory time budget.
    assert client.seen["all_users"] is True
    assert client.seen["max_results"] == _BQ_MAX_JOBS_SCAN


def test_bigquery_any_live_statement_is_running(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_bq(
        monkeypatch,
        _FakeBqClient(
            [
                _FakeBqJob("sf-run-abc-native-a1-0", "DONE"),
                _FakeBqJob("sf-run-abc-native-a1-1", "RUNNING"),
            ]
        ),
    )

    result = BigQueryProbe().check(_bq_handle(), settings=_SETTINGS)

    assert result.native_state == NATIVE_RUNNING


def test_bigquery_one_failed_statement_fails_the_group(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_bq(
        monkeypatch,
        _FakeBqClient(
            [
                _FakeBqJob("sf-run-abc-native-a1-0", "DONE"),
                _FakeBqJob("sf-run-abc-native-a1-1", "DONE", error_result={"reason": "invalid"}),
            ]
        ),
    )

    result = BigQueryProbe().check(_bq_handle(), settings=_SETTINGS)

    assert result.native_state == NATIVE_FAILED


def test_bigquery_no_matching_jobs_is_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_bq(monkeypatch, _FakeBqClient([_FakeBqJob("some-other-job", "DONE")]))

    result = BigQueryProbe().check(_bq_handle(), settings=_SETTINGS)

    assert result.native_state == NATIVE_NOT_FOUND
    assert result.exists is False


def test_bigquery_scan_lower_bounded_by_handle_created_at(monkeypatch: pytest.MonkeyPatch) -> None:
    # created_at (hydrated from the row) lower-bounds the history scan, minus a skew margin; a
    # handle without it (non-BQ / pre-feature) passes min_creation_time=None (fall back to the cap).
    from datetime import UTC, datetime

    started = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    handle = ProbeHandle(
        "bigquery",
        native_id="sf-run-abc-native-a1-",
        region="us",
        id_kind="prefix",
        created_at=started,
    )
    client = _FakeBqClient([_FakeBqJob("sf-run-abc-native-a1-0", "DONE")])
    _patch_bq(monkeypatch, client)

    BigQueryProbe().check(handle, settings=_SETTINGS)
    assert client.seen["min_creation_time"] < started  # skewed slightly earlier for clock drift

    bare = _FakeBqClient([_FakeBqJob("sf-run-abc-native-a1-0", "DONE")])
    _patch_bq(monkeypatch, bare)
    BigQueryProbe().check(_bq_handle(), settings=_SETTINGS)  # no created_at
    assert bare.seen["min_creation_time"] is None


def test_bigquery_error_degrades_to_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    from google.cloud import bigquery

    def _boom(project: str, location: str) -> Any:
        raise RuntimeError("bq down")

    monkeypatch.setattr(bigquery, "Client", _boom)

    result = BigQueryProbe().check(_bq_handle(), settings=_SETTINGS)

    assert result.native_state == NATIVE_UNKNOWN
    assert result.exists is True


# --- P3: staleness (_is_stale) ------------------------------------------------
#
# _is_stale marks the end of a RUNNING job's startup grace: only a RUNNING family whose
# `quiet_seconds` exceeds the threshold is stale (a vanished stale job is LOST; a young one is
# still-starting -> UNKNOWN). Every non-RUNNING status is never stale, and a family with no
# parseable timestamp (quiet_seconds None) declines rather than raising.
#
# The *age* itself is derived once by `review._assemble_progress` (row parsing, latest-signal-wins
# and the unparseable case are tested there) so the monitor's reported quiet time and the probe's
# escalation decision can never disagree; what is tested here is the threshold.

_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def _quiet(status: str | None, quiet_seconds: float | None) -> FamilyProgress:
    """A `FamilyProgress` carrying only what `_is_stale` reads: status + quiet time."""
    return _fp("statistical", status, quiet_seconds=quiet_seconds)


def test_is_stale_fresh_running_is_not_stale() -> None:
    assert _is_stale(_quiet("RUNNING", 30.0), None) is False


def test_is_stale_old_running_is_stale() -> None:
    assert _is_stale(_quiet("RUNNING", 3600.0), None) is True


def test_is_stale_respects_explicit_threshold() -> None:
    fp = _quiet("RUNNING", 120.0)
    assert _is_stale(fp, 60.0) is True
    assert _is_stale(fp, 600.0) is False


def test_is_stale_terminal_family_is_never_stale() -> None:
    # An old COMPLETED family must never escalate - this preserves the terminal short-circuit.
    assert _is_stale(_quiet("COMPLETED", 7 * 86400.0), None) is False


def test_is_stale_unknown_quiet_time_is_not_stale() -> None:
    # No job row, or timestamps that did not parse: no evidence of silence, so no escalation.
    assert _is_stale(_quiet("RUNNING", None), None) is False
    assert _is_stale(_quiet(None, None), None) is False


# --- P3: reconciliation matrix (_assemble_probe_report) -----------------------
#
# One test per row of the §4 verdict matrix over a single family, plus the terminal short-circuit,
# the never-launched fall-through, and the run-wide disagreement roll-up. Pure — no cloud.


def _fp(
    family: str,
    status: str | None,
    *,
    runtime: str | None = "spark",
    n_done: int = 0,
    n_expected: int | None = None,
    quiet_seconds: float | None = None,
) -> FamilyProgress:
    """A minimal `FamilyProgress` carrying only the fields the reconciliation reads."""
    return FamilyProgress(
        family=family,
        runtime=runtime,
        hardware="cpu",
        status=status,
        models=("m1",),
        n_expected=n_expected,
        n_done=n_done,
        fraction=None,
        avg_fit_seconds=None,
        runtime_seconds=None,
        quiet_seconds=quiet_seconds,
    )


def _progress(*families: FamilyProgress, status: str | None = "RUNNING") -> RunProgress:
    return RunProgress(
        run_id="sf-run-abc",
        status=status,
        n_series=None,
        families=tuple(families),
        n_done=sum(f.n_done for f in families),
        n_expected=None,
        fraction=None,
    )


def _only(report: Any) -> Any:
    assert len(report.families) == 1
    return report.families[0]


def test_report_terminal_family_trusts_registry() -> None:
    progress = _progress(_fp("statistical", "COMPLETED"), status="COMPLETED")
    report = _assemble_probe_report(progress, {}, frozenset())
    fv = _only(report)
    assert fv.verdict == VERDICT_TRUST_REGISTRY
    assert fv.disagreement is False
    assert fv.native_state is None and fv.exists is None
    # Terminal short-circuit: nothing was probed.
    assert report.escalated is False
    assert report.disagreement is False


def test_report_no_handle_family_is_unknown() -> None:
    progress = _progress(_fp("statistical", "RUNNING"))
    report = _assemble_probe_report(progress, {}, frozenset({"statistical"}))
    fv = _only(report)
    assert fv.verdict == VERDICT_UNKNOWN
    assert fv.disagreement is False
    assert fv.detail == "no handle recorded"
    assert report.escalated is False  # no native call ran


def test_report_running_native_confirms_running() -> None:
    progress = _progress(_fp("statistical", "RUNNING"))
    native = {"statistical": ProbeResult(NATIVE_RUNNING, exists=True, detail="in flight")}
    report = _assemble_probe_report(progress, native, frozenset())
    fv = _only(report)
    assert fv.verdict == VERDICT_RUNNING
    assert fv.disagreement is False
    assert fv.native_state == NATIVE_RUNNING and fv.exists is True
    assert report.escalated is True


def test_report_failed_native_over_live_registry_is_stale() -> None:
    # A failed runtime job is authoritative regardless of artifact counts.
    progress = _progress(_fp("statistical", "RUNNING"))
    native = {"statistical": ProbeResult(NATIVE_FAILED, exists=True)}
    report = _assemble_probe_report(progress, native, frozenset())
    fv = _only(report)
    assert fv.verdict == VERDICT_STALE_REGISTRY
    assert fv.disagreement is True
    assert report.disagreement is True


def test_report_succeeded_native_with_complete_artifacts_is_stale() -> None:
    # SUCCEEDED is trusted only when the artifacts corroborate it → the registry is genuinely stale.
    progress = _progress(_fp("statistical", "RUNNING", n_done=10, n_expected=10))
    native = {"statistical": ProbeResult(NATIVE_SUCCEEDED, exists=True)}
    report = _assemble_probe_report(progress, native, frozenset())
    fv = _only(report)
    assert fv.verdict == VERDICT_STALE_REGISTRY
    assert fv.disagreement is True


def test_report_succeeded_native_with_incomplete_artifacts_is_unknown() -> None:
    # A native family's BigQuery statements go DONE one-by-one; an all-DONE reading mid-run is a
    # lull between statements, not the end — incomplete artifacts ⇒ ambiguous, don't overrule.
    progress = _progress(_fp("native", "RUNNING", runtime="bigquery", n_done=3, n_expected=10))
    native = {"native": ProbeResult(NATIVE_SUCCEEDED, exists=True)}
    report = _assemble_probe_report(progress, native, frozenset())
    fv = _only(report)
    assert fv.verdict == VERDICT_UNKNOWN
    assert fv.disagreement is False


def test_report_not_found_with_all_artifacts_is_likely_completed() -> None:
    progress = _progress(_fp("statistical", "RUNNING", n_done=10, n_expected=10))
    native = {"statistical": ProbeResult(NATIVE_NOT_FOUND, exists=False)}
    report = _assemble_probe_report(progress, native, frozenset())
    fv = _only(report)
    assert fv.verdict == VERDICT_LIKELY_COMPLETED
    assert fv.disagreement is True
    assert fv.exists is False


def test_report_not_found_stale_with_partial_artifacts_is_lost() -> None:
    # Past the startup grace (family in `stale`): a vanished job with missing artifacts is LOST.
    progress = _progress(_fp("statistical", "RUNNING", n_done=3, n_expected=10))
    native = {"statistical": ProbeResult(NATIVE_NOT_FOUND, exists=False)}
    report = _assemble_probe_report(progress, native, frozenset(), frozenset({"statistical"}))
    fv = _only(report)
    assert fv.verdict == VERDICT_LOST
    assert fv.disagreement is True


def test_report_not_found_stale_with_unknown_denominator_is_lost() -> None:
    # n_expected is None (series count unknown) → we cannot claim completion → LOST (once stale).
    progress = _progress(_fp("statistical", "RUNNING", n_done=99, n_expected=None))
    native = {"statistical": ProbeResult(NATIVE_NOT_FOUND, exists=False)}
    report = _assemble_probe_report(progress, native, frozenset(), frozenset({"statistical"}))
    assert _only(report).verdict == VERDICT_LOST


def test_report_not_found_young_job_is_unknown_not_lost() -> None:
    # Startup grace: a RUNNING row is written before the native job exists, so a fresh probe
    # legitimately 404s. Not yet stale + incomplete artifacts ⇒ UNKNOWN (still starting), never a
    # false LOST — the probe must not cry wolf during a normal launch window.
    progress = _progress(_fp("statistical", "RUNNING", n_done=0, n_expected=10))
    native = {"statistical": ProbeResult(NATIVE_NOT_FOUND, exists=False)}
    report = _assemble_probe_report(progress, native, frozenset())  # not in `stale`
    fv = _only(report)
    assert fv.verdict == VERDICT_UNKNOWN
    assert fv.disagreement is False
    assert "still be starting" in fv.detail


def test_report_unknown_probe_does_not_overrule_registry() -> None:
    progress = _progress(_fp("statistical", "RUNNING"))
    native = {"statistical": ProbeResult(NATIVE_UNKNOWN, exists=True, detail="auth blip")}
    report = _assemble_probe_report(progress, native, frozenset())
    fv = _only(report)
    assert fv.verdict == VERDICT_UNKNOWN
    assert fv.disagreement is False


def test_report_non_terminal_never_launched_trusts_registry() -> None:
    # A configured family with no job row (never launched) is neither probed nor in no_handle.
    progress = _progress(_fp("ensemble", "PENDING"))
    report = _assemble_probe_report(progress, {}, frozenset())
    fv = _only(report)
    assert fv.verdict == VERDICT_TRUST_REGISTRY
    assert fv.disagreement is False


def test_report_disagreement_rolls_up_across_families() -> None:
    progress = _progress(
        _fp("statistical", "COMPLETED"),  # trust registry
        _fp("ml", "RUNNING", n_done=5, n_expected=5),  # NOT_FOUND + complete → likely-completed
        _fp("native", "RUNNING"),  # running-confirmed
    )
    native = {
        "ml": ProbeResult(NATIVE_NOT_FOUND, exists=False),
        "native": ProbeResult(NATIVE_RUNNING, exists=True),
    }
    report = _assemble_probe_report(progress, native, frozenset())
    verdicts = {f.family: f.verdict for f in report.families}
    assert verdicts == {
        "statistical": VERDICT_TRUST_REGISTRY,
        "ml": VERDICT_LIKELY_COMPLETED,
        "native": VERDICT_RUNNING,
    }
    assert report.escalated is True
    assert report.disagreement is True  # driven by the LIKELY_COMPLETED family
    # Per-family carries the registry status + counts straight through.
    ml = next(f for f in report.families if f.family == "ml")
    assert ml.registry_status == "RUNNING"
    assert (ml.n_done, ml.n_expected) == (5, 5)


# --- P5: CANCELLED is terminal (short-circuit + no-op re-cancel) ---------------


def test_report_cancelled_family_trusts_registry() -> None:
    # CANCELLED joins the terminal set: a cancelled job is settled → trusted, never re-escalated.
    progress = _progress(_fp("ml", "CANCELLED"), status="CANCELLED")
    report = _assemble_probe_report(progress, {}, frozenset())
    fv = _only(report)
    assert fv.verdict == VERDICT_TRUST_REGISTRY
    assert fv.native_state is None
    assert report.escalated is False


# --- P5: cancel blast-radius plan (_assemble_cancel_plan, pure) ----------------


def _report(*families: FamilyProgress, native: dict[str, ProbeResult] | None = None) -> Any:
    """Reconcile some families (default: nothing escalated) into a `ProbeReport` for plan tests."""
    return _assemble_probe_report(_progress(*families), native or {}, frozenset())


def test_cancel_plan_marks_running_family_cancellable() -> None:
    report = _report(
        _fp("ml", "RUNNING", runtime="spark", n_done=312, n_expected=500),
        native={"ml": ProbeResult(NATIVE_RUNNING, exists=True)},
    )
    plan = _assemble_cancel_plan(report)
    item = plan.items[0]
    assert item.cancellable is True
    assert "312/500" in item.note and "retained" in item.note
    assert plan.n_cancellable == 1
    assert plan.ensemble_suppressed is False


def test_cancel_plan_terminal_family_untouched() -> None:
    report = _assemble_probe_report(
        _progress(_fp("native", "COMPLETED", runtime="bigquery"), status="COMPLETED"),
        {},
        frozenset(),
    )
    plan = _assemble_cancel_plan(report)
    assert plan.items[0].cancellable is False
    assert "already COMPLETED" in plan.items[0].note
    assert plan.n_cancellable == 0


def test_cancel_plan_already_cancelled_is_noop() -> None:
    # An already-CANCELLED family is terminal → not cancellable (re-cancel is a no-op).
    report = _assemble_probe_report(
        _progress(_fp("ml", "CANCELLED"), status="CANCELLED"), {}, frozenset()
    )
    plan = _assemble_cancel_plan(report)
    assert plan.items[0].cancellable is False
    assert "already CANCELLED" in plan.items[0].note


def test_cancel_plan_suppresses_ensemble_when_base_cancelled() -> None:
    report = _report(
        _fp("ml", "RUNNING"),
        _fp("ensemble", "RUNNING", runtime="bigquery"),
        native={
            "ml": ProbeResult(NATIVE_RUNNING, exists=True),
            "ensemble": ProbeResult(NATIVE_RUNNING, exists=True),
        },
    )
    plan = _assemble_cancel_plan(report)
    assert plan.ensemble_suppressed is True
    ens = next(i for i in plan.items if i.family == "ensemble")
    assert "SKIPPED" in ens.note


def test_cancel_plan_no_ensemble_not_suppressed() -> None:
    report = _report(_fp("ml", "RUNNING"), native={"ml": ProbeResult(NATIVE_RUNNING, exists=True)})
    assert _assemble_cancel_plan(report).ensemble_suppressed is False


def test_cancel_plan_flags_vanished_runtime() -> None:
    # Cancellability is keyed on the registry status, so a family whose runtime job already vanished
    # is still "cancellable" (we finalize the registry) — the note says the runtime is gone.
    report = _assemble_probe_report(
        _progress(_fp("dl", "RUNNING", runtime="ray", n_done=0, n_expected=500)),
        {"dl": ProbeResult(NATIVE_NOT_FOUND, exists=False)},
        frozenset(),
        frozenset({"dl"}),
    )
    item = _assemble_cancel_plan(report).items[0]
    assert item.cancellable is True
    assert "already gone" in item.note


# --- P5: header roll-up after cancel (_roll_header_after_cancel, pure) ---------


def test_roll_header_all_cancelled_is_cancelled() -> None:
    assert _roll_header_after_cancel(["CANCELLED", "CANCELLED"]) == "CANCELLED"


def test_roll_header_mix_of_terminals_is_partial() -> None:
    assert _roll_header_after_cancel(["CANCELLED", "COMPLETED"]) == "PARTIAL"
    assert _roll_header_after_cancel(["CANCELLED", "FAILED"]) == "PARTIAL"


def test_roll_header_live_job_leaves_header_unchanged() -> None:
    # A stop that failed (job still RUNNING) must not finalize the run — leave the header alone.
    assert _roll_header_after_cancel(["CANCELLED", "RUNNING"]) is None


def test_roll_header_no_cancelled_leaves_header_unchanged() -> None:
    assert _roll_header_after_cancel(["COMPLETED", "FAILED"]) is None


def test_roll_header_empty_is_none() -> None:
    assert _roll_header_after_cancel([]) is None


# --- P5: cancel audit blob (_build_cancel_audit, pure) ------------------------


def test_build_cancel_audit_shape() -> None:
    ts = datetime(2026, 8, 29, 12, 0, 0, tzinfo=UTC)
    audit = _build_cancel_audit(
        actor="sa@proj.iam",
        cancelled_at=ts,
        reason="stuck job",
        native_state=NATIVE_RUNNING,
        n_done=7,
    )
    assert audit == {
        "cancelled_by": "sa@proj.iam",
        "cancelled_at": ts.isoformat(),
        "reason": "stuck job",
        "native_state_at_cancel": NATIVE_RUNNING,
        "n_done_at_cancel": 7,
    }


# --- P5: per-engine cancel() (stubbed clients, never touch GCP) ----------------


def test_spark_serverless_cancel_issues_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeBatchClient(batch=_FakeBatch("RUNNING"))
    _patch_batch_client(monkeypatch, client)

    result = SparkProbe().cancel(_serverless_handle(), settings=_SETTINGS)

    assert result.stopped is True and result.already_gone is False
    assert client.deleted is True
    assert client.seen["delete_name"].endswith("/batches/sf-run-abc-statistical-a1")


def test_spark_serverless_cancel_not_found_is_already_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    from google.api_core.exceptions import NotFound

    _patch_batch_client(monkeypatch, _FakeBatchClient(exc=NotFound("gone")))

    result = SparkProbe().cancel(_serverless_handle(), settings=_SETTINGS)

    assert result.already_gone is True and result.stopped is False


def test_spark_serverless_cancel_permission_denied_gives_iam_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from google.api_core.exceptions import PermissionDenied

    _patch_batch_client(monkeypatch, _FakeBatchClient(exc=PermissionDenied("nope")))

    result = SparkProbe().cancel(_serverless_handle(), settings=_SETTINGS)

    assert result.stopped is False and result.already_gone is False
    assert "job-canceller" in result.detail  # actionable IAM message, not a stack trace


def test_spark_cluster_cancel_calls_cancel_job(monkeypatch: pytest.MonkeyPatch) -> None:
    import scale_forecasting.cluster_telemetry as telemetry_mod

    seen: dict[str, Any] = {}

    def _fake_cancel(region: str, job_id: str, **kw: Any) -> None:
        seen.update(region=region, job_id=job_id, timeout=kw.get("timeout"))

    monkeypatch.setattr(telemetry_mod, "cancel_cluster_job", _fake_cancel)

    result = SparkProbe().cancel(_cluster_handle(), settings=_SETTINGS)

    assert result.stopped is True
    assert seen == {
        "region": "us-west1",
        "job_id": "real-dataproc-job-id",
        "timeout": pytest.approx(20.0),
    }


def test_spark_cluster_cancel_no_id_reports_failure() -> None:
    # No server-assigned id yet → nothing addressable to cancel (and no false "already gone").
    handle = ProbeHandle("spark", native_id="", region="us-west1", spark_mode="cluster")
    result = SparkProbe().cancel(handle, settings=_SETTINGS)
    assert result.stopped is False and result.already_gone is False
    assert "not yet assigned" in result.detail


def test_ray_cancel_stops_job(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeRayJobClient("RUNNING")
    _patch_ray(monkeypatch, client=client)

    result = RayProbe().cancel(_ray_handle(), settings=_SETTINGS)

    assert result.stopped is True and result.already_gone is False
    assert client.stopped_id == "job-1"


def test_ray_cancel_cluster_gone_is_already_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    from google.api_core.exceptions import NotFound

    _patch_ray(monkeypatch, cluster_exc=NotFound("cluster gone"))

    result = RayProbe().cancel(_ray_handle(), settings=_SETTINGS)

    assert result.already_gone is True and result.stopped is False


def test_bigquery_cancel_cancels_live_statements(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeBqClient(
        [
            _FakeBqJob("sf-run-abc-native-a1-0", "DONE"),  # already done → skipped
            _FakeBqJob("sf-run-abc-native-a1-1", "RUNNING"),  # live → cancelled
            _FakeBqJob("other-run", "RUNNING"),  # different prefix → excluded
        ]
    )
    _patch_bq(monkeypatch, client)

    result = BigQueryProbe().cancel(_bq_handle(), settings=_SETTINGS)

    assert result.stopped is True
    assert client.cancelled == ["sf-run-abc-native-a1-1"]
    assert client.seen["cancel_location"] == "us"


def test_bigquery_cancel_no_live_jobs_is_already_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_bq(monkeypatch, _FakeBqClient([_FakeBqJob("sf-run-abc-native-a1-0", "DONE")]))

    result = BigQueryProbe().cancel(_bq_handle(), settings=_SETTINGS)

    assert result.already_gone is True and result.stopped is False


def test_cancel_error_degrades_without_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    # A non-permission error is reported (not raised) with stopped=already_gone=False.
    _patch_batch_client(monkeypatch, _FakeBatchClient(exc=RuntimeError("transport down")))
    result = SparkProbe().cancel(_serverless_handle(), settings=_SETTINGS)
    assert result.stopped is False and result.already_gone is False
    assert "transport down" in result.detail


# --- --job narrowing (_narrow_to_job) -----------------------------------------
#
# The report an operator reads and the rows `cancel_run` actually stops come out of the same
# filter. If they could disagree, `--job statistical` would preview one family and cancel every
# family in the run, so these tests assert the two move together — including in the untouched
# `job=None` case, where the pair must be the full run on both sides.


def _job_rows() -> list[dict[str, Any]]:
    return [
        {"family": "statistical", "job_id": "j-stat", "status": "RUNNING"},
        {"family": "ml", "job_id": "j-ml", "status": "RUNNING"},
        {"family": "native", "job_id": "j-native", "status": "COMPLETED"},
    ]


def _both_families() -> RunProgress:
    return _progress(
        _fp("statistical", "RUNNING"), _fp("ml", "RUNNING"), _fp("native", "COMPLETED")
    )


def test_narrow_to_job_filters_the_report_and_the_rows_together() -> None:
    progress, rows = _narrow_to_job(_both_families(), _job_rows(), "ml", "sf-run-abc")
    assert [f.family for f in progress.families] == ["ml"]
    assert [r["job_id"] for r in rows] == ["j-ml"]


def test_narrow_to_job_none_touches_neither_side() -> None:
    progress, rows = _narrow_to_job(_both_families(), _job_rows(), None, "sf-run-abc")
    assert [f.family for f in progress.families] == ["statistical", "ml", "native"]
    assert [r["family"] for r in rows] == ["statistical", "ml", "native"]


def test_narrow_to_job_rejects_a_family_this_run_does_not_have() -> None:
    # A typo must fail loudly: filtering to nothing would report an empty, healthy-looking run --
    # and on the cancel path, a plan with no items reads as "nothing to do".
    with pytest.raises(ConfigError, match="unknown family 'statistcal'"):
        _narrow_to_job(_both_families(), _job_rows(), "statistcal", "sf-run-abc")


def test_narrow_to_job_names_the_runs_actual_families_in_the_error() -> None:
    with pytest.raises(ConfigError, match=r"\['ml', 'native', 'statistical'\]"):
        _narrow_to_job(_both_families(), _job_rows(), "deep_learning", "sf-run-abc")


def test_narrow_to_job_copies_rows_rather_than_aliasing_them() -> None:
    # The rows go on to be mutated into audit blobs; the caller's list must not be reached through.
    source = _job_rows()
    _, rows = _narrow_to_job(_both_families(), source, None, "sf-run-abc")
    rows[0]["status"] = "CANCELLED"
    assert source[0]["status"] == "RUNNING"


# --- what a confirmed cancel addresses (_cancel_steps) ------------------------
#
# The join between the blast-radius plan (built from the run's families) and the job rows (from
# `v_run_jobs`) decides which runtime jobs a confirmed cancel actually stops. It sits on the
# destructive path, so every skip in it is asserted here rather than inferred from the live run:
# a terminal family, a never-launched one, and an unaddressable one each have to behave the way
# the preview implied, and in the order the operator just approved.


def _handle_row(family: str, job_id: str, *, handle: dict[str, Any] | None) -> dict[str, Any]:
    row: dict[str, Any] = {"family": family, "job_id": job_id, "status": "RUNNING"}
    if handle is not None:
        row["probe_handle"] = handle
    return row


_SPARK_BLOB = {
    "runtime": "spark",
    "native_id": "b1",
    "region": "us-central1",
    "spark_mode": "serverless",
}


def test_cancel_steps_targets_every_cancellable_family_that_has_a_handle() -> None:
    plan = _assemble_cancel_plan(_report(_fp("statistical", "RUNNING"), _fp("ml", "RUNNING")))
    steps = _cancel_steps(
        plan,
        [
            _handle_row("statistical", "j-stat", handle=_SPARK_BLOB),
            _handle_row("ml", "j-ml", handle=_SPARK_BLOB),
        ],
    )
    assert [s.item.family for s in steps] == ["statistical", "ml"]
    assert all(s.handle.runtime == "spark" for s in steps)


def test_cancel_steps_never_targets_a_terminal_family() -> None:
    # The row is present and perfectly addressable -- it is the plan's `cancellable` flag, not the
    # row, that keeps a COMPLETED family from being stopped.
    report = _assemble_probe_report(
        _progress(_fp("native", "COMPLETED", runtime="bigquery"), status="COMPLETED"),
        {},
        frozenset(),
    )
    steps = _cancel_steps(
        _assemble_cancel_plan(report), [_handle_row("native", "j-native", handle=_SPARK_BLOB)]
    )
    assert steps == ()


def test_cancel_steps_skips_a_planned_family_that_never_launched() -> None:
    # A family the config plans but that has no `run_jobs` row yet has no runtime job to stop, so
    # it is skipped outright -- the header re-read after the loop is what settles its status.
    plan = _assemble_cancel_plan(_report(_fp("statistical", "RUNNING"), _fp("ml", None)))
    steps = _cancel_steps(plan, [_handle_row("statistical", "j-stat", handle=_SPARK_BLOB)])
    assert [s.item.family for s in steps] == ["statistical"]


def test_cancel_steps_reports_a_live_family_it_cannot_address() -> None:
    # No handle (a pre-feature or malformed row) is the one skip that must be *visible*: the job is
    # live and we failed to reach it, which is not the same as there being nothing to stop.
    plan = _assemble_cancel_plan(_report(_fp("ml", "RUNNING")))
    (step,) = _cancel_steps(plan, [_handle_row("ml", "j-ml", handle=None)])
    assert isinstance(step, CancelOutcome)
    assert step.requested is False and step.cancelled is False
    assert step.job_key == "j-ml"
    assert "no handle" in step.detail


def test_cancel_steps_interleaves_outcomes_in_plan_order() -> None:
    # The report has to read in the order of the preview the operator approved, so an unaddressable
    # family keeps its position rather than being collected at one end.
    plan = _assemble_cancel_plan(
        _report(
            _fp("statistical", "RUNNING"), _fp("ml", "RUNNING"), _fp("deep_learning", "RUNNING")
        )
    )
    steps = _cancel_steps(
        plan,
        [
            _handle_row("statistical", "j-stat", handle=_SPARK_BLOB),
            _handle_row("ml", "j-ml", handle=None),
            _handle_row("deep_learning", "j-dl", handle=_SPARK_BLOB),
        ],
    )
    assert [type(s) is CancelOutcome for s in steps] == [False, True, False]
    assert [s.family if isinstance(s, CancelOutcome) else s.item.family for s in steps] == [
        "statistical",
        "ml",
        "deep_learning",
    ]


def test_cancel_steps_copies_the_row_it_hands_to_the_finalizer() -> None:
    plan = _assemble_cancel_plan(_report(_fp("ml", "RUNNING")))
    source = [_handle_row("ml", "j-ml", handle=_SPARK_BLOB)]
    (step,) = _cancel_steps(plan, source)
    step.row["status"] = "CANCELLED"
    assert source[0]["status"] == "RUNNING"
