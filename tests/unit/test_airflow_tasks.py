"""Offline tests for the Airflow task-callable seam (``scale_forecasting.airflow_tasks``).

Covers the module's pure, GCP-free surface — the header-status roll-up and the XCom cluster pull —
without importing Airflow or touching a registry. The task callables themselves (``begin_run``,
``run_family``, …) are thin wrappers over `main`'s launch building blocks, exercised live by the
orchestrator tests and the ``@gcp`` smoke; here we only pin the logic that lives in this module.
"""

from __future__ import annotations

import pytest

from scale_forecasting import airflow_tasks
from scale_forecasting.settings import Settings

# A resolved Settings for the teardown tests (never used to touch GCP — teardown is faked).
_SETTINGS = Settings(
    project_id="proj-x",
    connection="proj-x.us-central1.conn",
    warehouse_uri="gs://bkt/warehouse",
)

# --- combined_run_status: the header roll-up (mirror of main._combined_status) --------------------


def test_all_families_completed_is_completed() -> None:
    statuses = {"statistical": "COMPLETED", "ml": "COMPLETED"}
    assert airflow_tasks.combined_run_status(statuses, ensemble_enabled=False) == "COMPLETED"


def test_all_families_failed_is_failed() -> None:
    statuses = {"statistical": "FAILED", "ml": "FAILED"}
    assert airflow_tasks.combined_run_status(statuses, ensemble_enabled=False) == "FAILED"


def test_mixed_families_is_partial() -> None:
    statuses = {"statistical": "COMPLETED", "ml": "FAILED"}
    assert airflow_tasks.combined_run_status(statuses, ensemble_enabled=False) == "PARTIAL"


def test_missing_or_running_family_counts_as_failed() -> None:
    # a family row still RUNNING (its task died before finalizing) or absent is not COMPLETED
    assert (
        airflow_tasks.combined_run_status(
            {"statistical": "COMPLETED", "ml": "RUNNING"}, ensemble_enabled=False
        )
        == "PARTIAL"
    )
    assert (
        airflow_tasks.combined_run_status({"statistical": "RUNNING"}, ensemble_enabled=False)
        == "FAILED"
    )


def test_no_base_families_is_completed() -> None:
    # degenerate: nothing to fail → COMPLETED (matches main._combined_status n_failed==0 branch)
    assert airflow_tasks.combined_run_status({}, ensemble_enabled=False) == "COMPLETED"


def test_ensemble_incomplete_downgrades_completed_run() -> None:
    statuses = {"statistical": "COMPLETED", "ml": "COMPLETED", "ensemble": "FAILED"}
    assert airflow_tasks.combined_run_status(statuses, ensemble_enabled=True) == "FAILED"


def test_completed_ensemble_keeps_completed() -> None:
    statuses = {"statistical": "COMPLETED", "ensemble": "COMPLETED"}
    assert airflow_tasks.combined_run_status(statuses, ensemble_enabled=True) == "COMPLETED"


def test_ensemble_never_masks_a_family_failure() -> None:
    # a base-family PARTIAL is not upgraded by a completed ensemble; the ensemble key is excluded
    # from the base roll-up
    statuses = {"statistical": "COMPLETED", "ml": "FAILED", "ensemble": "COMPLETED"}
    assert airflow_tasks.combined_run_status(statuses, ensemble_enabled=True) == "PARTIAL"


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
