"""Offline tests for the Airflow task-callable seam (``scale_forecasting.airflow_tasks``).

Covers the module's pure, GCP-free surface — the repair node and the XCom cluster pull — without
importing Airflow or touching a registry. The task callables themselves (``begin_run``,
``run_family``, …) are thin wrappers over `main`'s launch building blocks, exercised live by the
orchestrator tests and the ``@gcp`` smoke; here we only pin the logic that lives in this module.

The header-status roll-up `finalize_run` applies is **not** here: it is one shared function,
`job_outcome.combined_run_status`, and `test_job_outcome` pins it once for both this task and
`main.run`.
"""

from __future__ import annotations

from typing import Any

import pytest

from scale_forecasting import airflow_tasks
from scale_forecasting.settings import Settings

# A resolved Settings for the teardown tests (never used to touch GCP — teardown is faked).
_SETTINGS = Settings(
    project_id="proj-x",
    connection="proj-x.us-central1.conn",
    warehouse_uri="gs://bkt/warehouse",
)

# --- retry_families: the repair node's task callable ----------------------------------------------
#
# The node is the unattended form of ``--retry --force`` and delegates to the same
# `retry_run.retry_run`, so these pin the seam rather than the decision: what it passes, what it
# returns to XCom, and — most of the point — that it cannot take the ensemble down with it.


def _plan(run_id: str = "run-abc") -> Any:
    from scale_forecasting.retry_run import RetryPlan

    return RetryPlan(run_id=run_id, header_status="PARTIAL", counts={"RETRY_AS_IS": 12})


def _report(*, executed: bool, families: tuple[str, ...] = (), errors: Any = None) -> Any:
    from scale_forecasting.job_launch import RetryOutcome
    from scale_forecasting.retry_run import RetryReport

    outcome = RetryOutcome(families=families, errors=errors or {}) if executed else None
    return RetryReport(run_id="run-abc", plan=_plan(), executed=executed, outcome=outcome)


def _patch_retry(monkeypatch: pytest.MonkeyPatch, result: Any) -> list[dict[str, Any]]:
    """Stub out the repair call and the config/settings loads; return the recorded call list.

    ``result`` is either the `retry_run.RetryReport` to hand back or an exception to raise.
    """
    from scale_forecasting import config as config_module
    from scale_forecasting import retry_run as retry_run_module

    calls: list[dict[str, Any]] = []

    def _fake_retry_run(cfg: Any, **kw: Any) -> Any:
        calls.append({"cfg": cfg, **kw})
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(config_module, "load_config_uri", lambda uri: f"cfg<{uri}>")
    monkeypatch.setattr(Settings, "resolve", classmethod(lambda cls: _SETTINGS))
    monkeypatch.setattr(retry_run_module, "retry_run", _fake_retry_run)
    return calls


def test_the_retry_node_confirms_the_repair_it_previews(monkeypatch: pytest.MonkeyPatch) -> None:
    # Unattended by definition: there is no operator at a terminal to type "yes", so the node
    # submits. Everything that makes that safe (a landed model is never re-asked) is in the plan.
    calls = _patch_retry(monkeypatch, _report(executed=True, families=("statistical_repair",)))
    summary = airflow_tasks.retry_families("gs://bkt/run.json")
    assert calls == [
        {
            "cfg": "cfg<gs://bkt/run.json>",
            "confirm": True,
            "reason": "airflow retry node",
            "settings": _SETTINGS,
        }
    ]
    assert "statistical_repair" in summary


def test_the_retry_node_logs_the_table_it_acted_on(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from scale_forecasting.retry_run import format_retry_plan

    _patch_retry(monkeypatch, _report(executed=True, families=("ml_repair",)))
    with caplog.at_level("INFO"):
        airflow_tasks.retry_families("gs://bkt/run.json")
    assert format_retry_plan(_plan()) in caplog.text


def test_a_run_with_nothing_to_repair_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_retry(monkeypatch, _report(executed=False))
    assert "nothing to repair" in airflow_tasks.retry_families("gs://bkt/run.json")


def test_a_repair_that_failed_again_is_reported_not_hidden(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _patch_retry(
        monkeypatch,
        _report(
            executed=True,
            families=("statistical_repair", "ml_repair"),
            errors={"ml_repair": RuntimeError("no capacity")},
        ),
    )
    with caplog.at_level("ERROR"):
        summary = airflow_tasks.retry_families("gs://bkt/run.json")
    assert "still failing: ml_repair" in summary
    assert "ml_repair" in caplog.text


def test_a_broken_repair_never_takes_the_ensemble_down_with_it(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The node sits upstream of the ensemble, so raising here would *skip* a run's ensemble.

    A repair is an attempt to improve a run's outcome; failing to make one must never leave the run
    worse off than not trying. The failure still lands in the log, and any row the repair opened
    still carries its own status.
    """
    _patch_retry(monkeypatch, RuntimeError("registry unreachable"))
    with caplog.at_level("ERROR"):
        summary = airflow_tasks.retry_families("gs://bkt/run.json")
    assert "registry unreachable" in summary
    assert "could not be attempted" in caplog.text


# --- _xcom_cluster: the shared-cluster (name, region) pull ----------------------------------------


def test_xcom_cluster_none_without_task_instance() -> None:
    # a direct unit call (no Airflow ti) → no shared cluster; the family self-provisions
    assert airflow_tasks._xcom_cluster(None, "create_ray_cluster") is None


class _FakeTI:
    def __init__(self, value: object) -> None:
        self._value = value

    def xcom_pull(self, task_ids: str) -> object:
        return self._value


def test_xcom_cluster_returns_name_region_pair() -> None:
    ti = _FakeTI(["sf-ray-abc", "us-east1"])
    assert airflow_tasks._xcom_cluster(ti, "create_ray_cluster") == ("sf-ray-abc", "us-east1")


def test_xcom_cluster_none_when_create_task_did_not_run() -> None:
    # the create task was not in the DAG (no shared cluster) → XCom pull yields nothing
    assert airflow_tasks._xcom_cluster(_FakeTI(None), "create_ray_cluster") is None


# --- _xcom_spark_clusters: the same pull, keyed by hardware ---------------------------------------
# Dataproc's create task returns one cluster per hardware kind (a Dataproc cluster has a single
# worker machine type), so its XCom is a dict where Ray's is a pair.


def test_xcom_spark_clusters_returns_pairs_keyed_by_hardware() -> None:
    # XCom round-trips through JSON, so the create task's lists come back as lists; the helper
    # normalizes them to the tuples the local path yields, so job_launch sees one shape.
    ti = _FakeTI(
        {"cpu": ["sf-cluster-abc-cpu", "us-central1"], "gpu": ["sf-cluster-abc-gpu", "us-east4"]}
    )
    assert airflow_tasks._xcom_spark_clusters(ti) == {
        "cpu": ("sf-cluster-abc-cpu", "us-central1"),
        "gpu": ("sf-cluster-abc-gpu", "us-east4"),
    }


def test_xcom_spark_clusters_none_without_task_instance_or_create_task() -> None:
    assert airflow_tasks._xcom_spark_clusters(None) is None
    assert airflow_tasks._xcom_spark_clusters(_FakeTI(None)) is None


def test_delete_spark_cluster_tears_down_every_cluster(monkeypatch: pytest.MonkeyPatch) -> None:
    from scale_forecasting import dataproc_cluster
    from scale_forecasting import settings as settings_mod

    torn: list[str] = []
    monkeypatch.setattr(
        dataproc_cluster,
        "teardown_shared_cluster",
        lambda name, region, settings: torn.append(name),
    )
    monkeypatch.setattr(settings_mod.Settings, "resolve", staticmethod(lambda: _SETTINGS))
    ti = _FakeTI({"cpu": ["c-cpu", "us-central1"], "gpu": ["c-gpu", "us-central1"]})
    airflow_tasks.delete_spark_cluster("gs://cfg.json", ti)
    assert torn == ["c-cpu", "c-gpu"]


def test_delete_spark_cluster_keeps_going_after_a_failed_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A first teardown that raises must not leave the second cluster billing.

    Bailing on the first error is the failure this task exists to prevent — so every cluster is
    attempted, and the error surfaces only once there is nothing left to reclaim.
    """
    from scale_forecasting import dataproc_cluster
    from scale_forecasting import settings as settings_mod

    torn: list[str] = []

    def _teardown(name: str, region: str, settings: object) -> None:
        torn.append(name)
        if name == "c-cpu":
            raise RuntimeError("delete refused")

    monkeypatch.setattr(dataproc_cluster, "teardown_shared_cluster", _teardown)
    monkeypatch.setattr(settings_mod.Settings, "resolve", staticmethod(lambda: _SETTINGS))
    ti = _FakeTI({"cpu": ["c-cpu", "us-central1"], "gpu": ["c-gpu", "us-central1"]})
    with pytest.raises(RuntimeError, match="delete refused"):
        airflow_tasks.delete_spark_cluster("gs://cfg.json", ti)
    assert torn == ["c-cpu", "c-gpu"]
