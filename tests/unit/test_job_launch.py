"""Offline tests for the per-job launchers (``scale_forecasting.job_launch``).

Three variants of one recipe — resolve the attempt, derive the deterministic identity, open the
``run_jobs`` lifecycle, dispatch in contributor mode — so the assertions here are about *identity
and lifecycle*, not about forecasting: which family and attempt the row is opened for, what
``system_job_id`` and probe handle get stamped on it, and which submitter (or engine) the body
reaches. Every registry and submit seam is faked; nothing touches GCP.

The same three functions are what an emitted Airflow DAG's task callables invoke
(`airflow_tasks.run_family` / ``run_native`` / ``run_ensemble``), so these tests cover the Composer
node behaviour as well as the local `main.run` one — that is what "same code local ↔ Composer"
means at the node level.
"""

from __future__ import annotations

from typing import Any

import pytest

from scale_forecasting import dag, job_launch
from scale_forecasting.config import RunConfig
from scale_forecasting.settings import Settings

# Model names by runtime: theta is a Python/Spark model; arima_plus / timesfm are the
# BigQuery-native models (runtime == "bigquery").
_SPARK = "theta"
_NATIVE = ["arima_plus", "timesfm"]

# A resolved Settings for the dispatch tests (never used to touch GCP — the submit fns are faked).
_SETTINGS = Settings(
    project_id="proj-x",
    connection="proj-x.us-central1.conn",
    warehouse_uri="gs://bkt/warehouse",
)


def _cfg(**over: Any) -> RunConfig:
    base: dict[str, Any] = {
        "run_name": "main test",
        "data": {"source_table": "source_series_native", "horizon": 7, "series_limit": 5},
        "models": [_SPARK, *_NATIVE],
    }
    base.update(over)
    return RunConfig(**base)


# --- launch_family_job / launch_native_job: per-job lifecycle + dispatch --------


def _fake_job_lifecycle(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Fake the run_jobs attempt+lifecycle seams so a family launch runs offline.

    Records the ``run_job`` call args under ``"job"`` and yields a real `JobFinalizer` so the body
    can finalize normally. ``next_job_attempt`` is pinned to ``(1, True)`` (first attempt, new job).
    """
    from contextlib import contextmanager

    from scale_forecasting.registry import jobs, lifecycle

    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        jobs, "next_job_attempt", lambda run_id, family, *, force=False, settings=None: (1, True)
    )

    @contextmanager
    def _fake_run_job(run_id: str, family: str, attempt: int, **kw: Any) -> Any:
        seen["job"] = {"run_id": run_id, "family": family, "attempt": attempt, **kw}
        fin = lifecycle.JobFinalizer()
        seen["fin"] = fin  # expose it so tests can assert what the body finalized
        yield fin

    monkeypatch.setattr(lifecycle, "run_job", _fake_run_job)
    return seen


def test_launch_family_job_dispatches_to_resolved_submitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scale_forecasting.submitters as submitters_mod

    seen = _fake_job_lifecycle(monkeypatch)
    captured: dict[str, Any] = {}

    class _FakeSubmitter:
        def launch(self, cfg: RunConfig, **kw: Any) -> None:
            captured.update(kw)
            captured["cfg"] = cfg

    def _fake_get(runtime: str) -> Any:
        captured["runtime"] = runtime
        return _FakeSubmitter()

    monkeypatch.setattr(submitters_mod, "get_submitter", _fake_get)

    cfg = _cfg(models=[_SPARK])
    job = dag.plan_dag(cfg).python_jobs[0]  # the statistical family, on the default Spark runtime
    job_launch.launch_family_job(cfg, job, "rid-0", _SETTINGS, max_executors=8)

    # Dispatch is by the family's *resolved* runtime, in contributor mode, with its model subset.
    assert captured["runtime"] == "spark"
    assert captured["models"] == [_SPARK]
    assert captured["manage_header"] is False
    assert captured["max_executors"] == 8
    # The per-job row is opened for this family's resolved compute + attempt.
    assert seen["job"]["family"] == "statistical"
    assert seen["job"]["attempt"] == 1
    assert seen["job"]["runtime"] == "spark"
    # The deterministic per-family platform id is threaded onto both the row and the submitter,
    # so a Spark family under a shared run_id gets its own batch id.
    from scale_forecasting.registry.ids import dataproc_job_id, make_job_key

    expected_id = dataproc_job_id(make_job_key("rid-0", "statistical", 1))
    assert seen["job"]["system_job_id"] == expected_id
    assert captured["system_job_id"] == expected_id
    # The ENTRY probe handle is stamped into the RUNNING row: a serverless spark job knows its id
    # and single region up front, so native_id is the system id and id_kind stays "exact".
    assert seen["job"]["probe_handle"] == {
        "runtime": "spark",
        "native_id": expected_id,
        "region": "us-central1",
        "id_kind": "exact",
        "spark_mode": "serverless",
    }
    # The default fake submitter returns None (its id == system_job_id), so nothing is stamped back.
    assert "system_job_id" not in seen["fin"].extra


def test_launch_family_job_stamps_real_id_when_submitter_returns_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cluster submitter returns a handle with a server-assigned id → it's finalized onto the row,
    and the entry handle is refreshed with the post-submit truths."""
    import scale_forecasting.submitters as submitters_mod
    from scale_forecasting.probes.vocabulary import ProbeHandle

    seen = _fake_job_lifecycle(monkeypatch)

    class _ClusterSubmitter:
        def launch(self, cfg: RunConfig, **kw: Any) -> ProbeHandle:
            # A cluster job's id is server-assigned (differs from the deterministic system_job_id);
            # the handle also carries the region the job actually landed in.
            return ProbeHandle(
                "spark", native_id="real-dataproc-job-id", region="us-west1", spark_mode="cluster"
            )

    monkeypatch.setattr(submitters_mod, "get_submitter", lambda runtime: _ClusterSubmitter())

    cfg = _cfg(models=[_SPARK])
    job = dag.plan_dag(cfg).python_jobs[0]
    job_launch.launch_family_job(cfg, job, "rid-0", _SETTINGS)

    # The real (server-assigned) id is stamped back onto the run_jobs row for reverse-trace.
    assert seen["fin"].extra["system_job_id"] == "real-dataproc-job-id"
    # The stamp-back also refreshes the probe handle with the post-submit truths (real id + region)
    # — as a merge, so a family that walked regions before it got this cluster does not have the
    # record of what it took to succeed erased at the moment of success.
    assert "job_telemetry" not in seen["fin"].extra
    assert seen["fin"].telemetry == {
        "probe_handle": {
            "runtime": "spark",
            "native_id": "real-dataproc-job-id",
            "region": "us-west1",
            "id_kind": "exact",
            "spark_mode": "cluster",
        }
    }


def test_launch_family_job_dispatches_ray_for_ray_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scale_forecasting.submitters as submitters_mod

    seen = _fake_job_lifecycle(monkeypatch)
    captured: dict[str, Any] = {}

    class _FakeSubmitter:
        def launch(self, cfg: RunConfig, **kw: Any) -> None:
            captured.update(kw)

    monkeypatch.setattr(
        submitters_mod,
        "get_submitter",
        lambda runtime: captured.__setitem__("runtime", runtime) or _FakeSubmitter(),
    )

    cfg = _cfg(models=[_SPARK], python_runtime="ray")
    job = dag.plan_dag(cfg).python_jobs[0]
    job_launch.launch_family_job(cfg, job, "rid-0", _SETTINGS)
    assert captured["runtime"] == "ray"
    # Ray keeps the canonical key verbatim as its submission id.
    from scale_forecasting.registry.ids import make_job_key, ray_submission_id

    submission_id = ray_submission_id(make_job_key("rid-0", "statistical", 1))
    assert captured["system_job_id"] == submission_id
    # The ENTRY probe handle for a self-provisioning Ray family. There is no shared cluster, and
    # `submit_ray` has not created one yet — but the name is a pure function of the run_id, so the
    # handle predicts it rather than shipping without one. Without this the probe and the cancel
    # have nothing to reach for during the entire window a single-family Ray job is running, since
    # the stamp-back only lands after the (blocking) job finishes.
    assert seen["job"]["probe_handle"] == {
        "runtime": "ray",
        "native_id": submission_id,
        "region": "us-central1",
        "id_kind": "exact",
        "resource_name": "projects/proj-x/locations/us-central1/persistentResources/sf-ray-rid-0",
    }


def test_launch_family_job_cluster_entry_handle_omits_unresolved_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A cluster job's real id is server-assigned, unknown at entry. The ENTRY handle must not assert
    # an id it doesn't have (that would risk a false NOT_FOUND on probe), so native_id is empty
    # until the stamp-back refresh fills it in.
    import scale_forecasting.submitters as submitters_mod

    seen = _fake_job_lifecycle(monkeypatch)

    class _NoOpSubmitter:
        def launch(self, cfg: RunConfig, **kw: Any) -> None:
            return None  # nothing to stamp back for this entry-handle assertion

    monkeypatch.setattr(submitters_mod, "get_submitter", lambda runtime: _NoOpSubmitter())

    cfg = _cfg(models=[_SPARK], compute={"families": {"statistical": {"spark_mode": "cluster"}}})
    job = dag.plan_dag(cfg).python_jobs[0]
    assert job.compute is not None and job.compute.spark_mode == "cluster"
    job_launch.launch_family_job(cfg, job, "rid-0", _SETTINGS)

    assert seen["job"]["probe_handle"] == {
        "runtime": "spark",
        "native_id": "",
        "region": "us-central1",
        "id_kind": "exact",
        "spark_mode": "cluster",
    }


def test_launch_native_job_runs_bigquery_engine_inline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scale_forecasting.engines import bigquery_engine

    seen = _fake_job_lifecycle(monkeypatch)
    captured: dict[str, Any] = {}

    def _fake_bq_run(cfg: RunConfig, models: list[str], **kw: Any) -> Any:
        captured["models"] = models
        captured["manage_header"] = kw.get("manage_header")
        return bigquery_engine.BqOutcome(status="COMPLETED", n_series=3, models=models)

    monkeypatch.setattr(bigquery_engine, "run", _fake_bq_run)

    cfg = _cfg()
    native = dag.plan_dag(cfg).native_job
    assert native is not None
    outcome = job_launch.launch_native_job(cfg, native, "rid-0", _SETTINGS)

    assert captured["models"] == _NATIVE
    assert captured["manage_header"] is False
    assert outcome.n_series == 3
    # The native family's entry probe handle: BigQuery coordinates are fully known up front, so the
    # handle is a job-id *prefix* (id_kind="prefix") ending in the "-" the engine prefixes its jobs.
    handle = seen["job"]["probe_handle"]
    assert handle["runtime"] == "bigquery"
    assert handle["id_kind"] == "prefix"
    assert handle["native_id"] == f"{seen['job']['system_job_id']}-"
    assert handle["region"] == "us-central1"
    # The native family's row is opened with the BigQuery runtime.
    assert seen["job"]["family"] == "native"
    assert seen["job"]["runtime"] == "bigquery"
    from scale_forecasting.registry.ids import bigquery_job_id, make_job_key

    assert seen["job"]["system_job_id"] == bigquery_job_id(make_job_key("rid-0", "native", 1))


def test_launch_ensemble_job_stamps_bigquery_prefix_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The ensemble node runs in BigQuery like the native family, so its entry handle is the only one
    # (no stamp-back site) and carries a BigQuery job-id prefix.
    import scale_forecasting.ensemble_run as ensemble_run

    seen = _fake_job_lifecycle(monkeypatch)
    monkeypatch.setattr(ensemble_run, "run_ensembles", lambda *a, **k: None)

    cfg = _cfg()
    job_launch.launch_ensemble_job(cfg, "rid-0", _SETTINGS)

    handle = seen["job"]["probe_handle"]
    assert handle["runtime"] == "bigquery"
    assert handle["id_kind"] == "prefix"
    assert handle["native_id"] == f"{seen['job']['system_job_id']}-"
    assert handle["region"] == "us-central1"
    assert seen["job"]["family"] == "ensemble"


# --- running out of regions: AWAITING_CAPACITY while it waits, FAILED with a reason after ---


def test_a_launching_family_installs_a_publisher_the_submitter_can_reach(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The install is *around* the dispatch, so a walk several frames down finds it.

    Only this frame knows which ``run_jobs`` row a walk belongs to, and it does not pass that down
    — see `capacity.publishing_to`. The submitter stands in for the walk here; the full chain
    (``submit_ray`` → ``_create_cluster_across_regions`` → ``walk``) is proven in test_ray_submit.
    """
    import scale_forecasting.submitters as submitters_mod
    from scale_forecasting import capacity

    _fake_job_lifecycle(monkeypatch)
    inside: list[Any] = []

    class _LooksAtAmbient:
        def launch(self, cfg: RunConfig, **kw: Any) -> None:
            inside.append(capacity.current_publisher())

    monkeypatch.setattr(submitters_mod, "get_submitter", lambda runtime: _LooksAtAmbient())

    cfg = _cfg(models=[_SPARK])
    job = dag.plan_dag(cfg).python_jobs[0]
    job_launch.launch_family_job(cfg, job, "rid-0", _SETTINGS)

    assert inside and inside[0] is not None  # the submitter saw a publisher...
    assert capacity.current_publisher() is None  # ...and it did not outlive the launch


def test_the_publisher_writes_awaiting_capacity_with_the_ledger_and_the_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mid-walk write: a live status, the attempt ledger, and the handle re-sent beside it.

    The telemetry is merged, so the ledger accretes onto whatever the row already holds. The probe
    handle is re-sent anyway, cheaply: a row written by an older code path that has no
    ``$.probe_handle`` yet would otherwise stay unprobeable for the entire wait, which is precisely
    the window an operator is most likely to be looking.
    """
    from scale_forecasting import capacity
    from scale_forecasting.registry import jobs as jobs_mod

    written: list[dict[str, Any]] = []
    monkeypatch.setattr(
        jobs_mod, "update_job", lambda job_id, **kw: written.append({"job_id": job_id, **kw})
    )

    handle = {"runtime": "ray", "native_id": "job-1", "region": "us-central1"}
    publish = job_launch._capacity_publisher("rid-0-statistical-1", handle, _SETTINGS)
    ledger = capacity.CapacityLedger(service="ray")
    ledger.record("us-east1", capacity.TRANSIENT_CAPACITY, "Resources are insufficient", 1.0)
    publish(ledger)

    assert len(written) == 1
    wrote = written[0]
    assert wrote["job_id"] == "rid-0-statistical-1"
    assert wrote["status"] == capacity.AWAITING_CAPACITY
    assert "job_telemetry" not in wrote
    assert wrote["merge_telemetry"]["probe_handle"] == handle
    assert wrote["merge_telemetry"]["capacity"]["attempts"][0]["candidate"] == "us-east1"


def test_the_publisher_will_not_write_over_a_cancel_that_landed_mid_walk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An operator who stops a waiting family must not have the next attempt undo it.

    The walk is a loop inside the submitting process, so a cancel cannot interrupt it — the next
    attempt comes around regardless. Without the guard it would write AWAITING_CAPACITY straight
    back over the CANCELLED, and the row would read as still waiting for a job nobody wants.
    """
    from scale_forecasting import capacity
    from scale_forecasting.registry import jobs as jobs_mod
    from scale_forecasting.registry.lifecycle import _STICKY_STATUSES

    written: list[dict[str, Any]] = []
    monkeypatch.setattr(
        jobs_mod, "update_job", lambda job_id, **kw: written.append({"job_id": job_id, **kw})
    )

    publish = job_launch._capacity_publisher("j-1", {}, _SETTINGS)
    publish(capacity.CapacityLedger(service="ray"))

    assert written[0]["unless_status_in"] == _STICKY_STATUSES


def test_a_family_that_runs_out_of_regions_records_why_before_it_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CAPACITY_EXHAUSTED plus the finished ledger, attached on the way out through `run_job`.

    Every other launch failure is a bug to fix; this one is a run to re-submit, possibly unchanged.
    A bare FAILED row cannot tell those apart, and before this the two were indistinguishable — a
    stocked-out region looked exactly like a broken import.
    """
    import scale_forecasting.submitters as submitters_mod
    from scale_forecasting import capacity

    seen = _fake_job_lifecycle(monkeypatch)
    ledger = capacity.CapacityLedger(service="ray")
    ledger.record("us-central1", capacity.TRANSIENT_CAPACITY, "no room", 1.0)
    ledger.record("us-east1", capacity.HARD_CEILING, "Quota exceeded for NVIDIA_T4_GPUS", 2.0)

    class _AllStockedOut:
        def launch(self, cfg: RunConfig, **kw: Any) -> None:
            raise capacity.CapacityExhausted("no capacity after 2 attempts", ledger=ledger)

    monkeypatch.setattr(submitters_mod, "get_submitter", lambda runtime: _AllStockedOut())

    cfg = _cfg(models=[_SPARK])
    job = dag.plan_dag(cfg).python_jobs[0]
    with pytest.raises(capacity.CapacityExhausted):
        job_launch.launch_family_job(cfg, job, "rid-0", _SETTINGS)

    # Re-raised, so the run's combined status still goes non-green — the reason is an addition to
    # the failure, not a softening of it.
    extra = seen["fin"].extra
    assert extra["failure_reason"] == capacity.CAPACITY_EXHAUSTED
    recorded = seen["fin"].telemetry["capacity"]["attempts"]
    assert [a["candidate"] for a in recorded] == ["us-central1", "us-east1"]
    assert [a["verdict"] for a in recorded] == [capacity.TRANSIENT_CAPACITY, capacity.HARD_CEILING]
    # The handle stays alongside it: the row a reconciler reads must not lose its coordinates just
    # because the launch failed.
    assert seen["fin"].telemetry["probe_handle"]["runtime"] == "spark"
    # And nothing else in the row is invented — `run_job`'s own handler owns the FAILED status.
    assert "status" not in extra


def test_an_ordinary_launch_failure_records_no_capacity_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reason has to stay rare to stay meaningful — a broken import is not a stockout."""
    import scale_forecasting.submitters as submitters_mod

    seen = _fake_job_lifecycle(monkeypatch)

    class _Broken:
        def launch(self, cfg: RunConfig, **kw: Any) -> None:
            raise ModuleNotFoundError("statsmodels")

    monkeypatch.setattr(submitters_mod, "get_submitter", lambda runtime: _Broken())

    cfg = _cfg(models=[_SPARK])
    job = dag.plan_dag(cfg).python_jobs[0]
    with pytest.raises(ModuleNotFoundError):
        job_launch.launch_family_job(cfg, job, "rid-0", _SETTINGS)
    assert "failure_reason" not in seen["fin"].extra


# --- ensemble DAG node: identity + mode dispatch -------------------------------


def _patch_ensemble_seams(monkeypatch: pytest.MonkeyPatch, calls: dict[str, Any]) -> None:
    import contextlib

    from scale_forecasting import ensemble_run
    from scale_forecasting.registry import jobs, lifecycle

    monkeypatch.setattr(jobs, "next_job_attempt", lambda *a, **k: (1, None))

    @contextlib.contextmanager
    def _fake_run_job(run_id: str, family: str, attempt: int, **k: Any) -> Any:
        calls["run_job"] = {"family": family, "runtime": k.get("runtime")}
        yield None

    monkeypatch.setattr(lifecycle, "run_job", _fake_run_job)
    monkeypatch.setattr(
        ensemble_run, "run_ensembles", lambda *a, **k: calls.__setitem__("barrier", True)
    )
    monkeypatch.setattr(
        ensemble_run,
        "run_ensembles_microbatch",
        lambda *a, **k: calls.__setitem__("microbatch", True),
    )


def test_launch_ensemble_job_barrier_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}
    _patch_ensemble_seams(monkeypatch, calls)
    cfg = _cfg(ensemble={"enabled": True, "strategies": ["mean"]})
    job_launch.launch_ensemble_job(cfg, "run-abc", _SETTINGS)
    assert calls.get("barrier") is True
    assert "microbatch" not in calls
    # It opens its own run_jobs row as the "ensemble" family, executed on the driver (bigquery).
    assert calls["run_job"] == {"family": "ensemble", "runtime": "bigquery"}


def test_launch_ensemble_job_microbatch_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}
    _patch_ensemble_seams(monkeypatch, calls)
    cfg = _cfg(
        ensemble={"enabled": True, "strategies": ["mean"]},
        compute={"ensemble": {"runtime": "spark", "mode": "microbatch"}},
    )
    job_launch.launch_ensemble_job(cfg, "run-abc", _SETTINGS)
    assert calls.get("microbatch") is True
    assert "barrier" not in calls
