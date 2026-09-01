"""Offline tests for the Airflow task-callable seam (``scale_forecasting.airflow_tasks``).

Covers the module's pure, GCP-free surface — the header-status roll-up and the XCom cluster pull —
without importing Airflow or touching a registry. The task callables themselves (``begin_run``,
``run_family``, …) are thin wrappers over `main`'s launch building blocks, exercised live by the
orchestrator tests and the ``@gcp`` smoke; here we only pin the logic that lives in this module.
"""

from __future__ import annotations

from scale_forecasting import airflow_tasks

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
