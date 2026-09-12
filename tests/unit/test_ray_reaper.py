"""Offline tests for the Ray cluster reaper (`ray_reaper`) — the whole decision, none of the I/O.

Every rule that can delete a running machine lives in `ray_reaper.classify_clusters`, which is pure:
it takes clusters, a map of run ids to header statuses and a clock, and returns a verdict plus a
reason for each. So the tests here are the real coverage for this feature — the Vertex listing and
the delete it drives are thin wrappers the ``@gcp`` smoke exercises.

They are written the way the risk runs. A reaper that fails to delete costs money; a reaper that
deletes the wrong thing kills a running job and loses hours of GPU work, so most of what follows
pins the *refusals*: the reuse target, the other deployment's cluster, the run that is still going,
and the few-seconds-old cluster whose run has not written its header yet.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from scale_forecasting import ray_reaper
from scale_forecasting.ray_cluster import APP_LABEL, REGISTRY_LABEL_KEY, RayCluster, cluster_labels
from scale_forecasting.settings import Settings

NOW = datetime(2026, 9, 12, 18, 0, 0, tzinfo=UTC)
REGISTRY = "scale_forecasting"


def _cluster(
    name: str,
    *,
    state: str = "RUNNING",
    age_s: float = 7200.0,
    labels: dict[str, str] | None = None,
    region: str = "us-central1",
) -> RayCluster:
    """A cluster as `ray_cluster.list_clusters` hands it over — ours by default, two hours old."""
    ours = {APP_LABEL[0]: APP_LABEL[1], REGISTRY_LABEL_KEY: REGISTRY}
    return RayCluster(
        name=name,
        region=region,
        resource_name=f"projects/p/locations/{region}/persistentResources/{name}",
        state=state,
        labels=ours if labels is None else labels,
        create_time=None if age_s is None else NOW - timedelta(seconds=age_s),
    )


def _classify(clusters, statuses, **kw):
    return ray_reaper.classify_clusters(
        clusters, statuses, registry_dataset_id=REGISTRY, now=NOW, **kw
    )


# --- the two answers that matter ---------------------------------------------------


@pytest.mark.parametrize("status", ["COMPLETED", "FAILED", "PARTIAL", "CANCELLED"])
def test_a_cluster_whose_run_is_over_is_garbage(status):
    """The leak this module exists for: the run ended, teardown never ran, the T4 kept billing."""
    (verdict,) = _classify([_cluster("sf-ray-demo-abc123")], {"demo-abc123": status})
    assert verdict.reap
    assert "demo-abc123" in verdict.reason


@pytest.mark.parametrize("status", ["RUNNING", "PENDING", "AWAITING_CAPACITY", "running"])
def test_a_cluster_whose_run_is_still_going_is_left_alone(status):
    """Deleting this one takes a live job's cluster out from under it — the worst outcome."""
    (verdict,) = _classify([_cluster("sf-ray-demo-abc123")], {"demo-abc123": status})
    assert not verdict.reap
    assert "still in flight" in verdict.reason


def test_an_awaiting_capacity_run_keeps_its_cluster():
    """Guard on the borrowed status set: `LIVE_STATUSES` is what makes this a refusal, not a delete.

    A run between capacity attempts has no running job and no recent signal, which reads exactly
    like an abandoned one. If the reaper ever stopped consulting the same live-status set the
    destructive registry verbs use, this is the case that would silently start deleting.
    """
    (verdict,) = _classify([_cluster("sf-ray-x-1")], {"x-1": "AWAITING_CAPACITY"})
    assert not verdict.reap


# --- the refusals -------------------------------------------------------------------


def test_a_named_reuse_cluster_is_never_a_candidate():
    """An operator's standing cluster has no run to be over, and is not ours to delete."""
    (verdict,) = _classify([_cluster("team-standing-ray")], {"demo-abc123": "COMPLETED"})
    assert not verdict.reap
    assert verdict.run_id_prefix is None
    assert "reuse target" in verdict.reason


def test_another_deployments_cluster_is_invisible_to_us():
    """Two registries in one project: each one's runs are unknown to the other, so each would read
    the other's live clusters as headerless garbage. The registry label is what prevents that."""
    other = _cluster(
        "sf-ray-demo-abc123",
        labels={APP_LABEL[0]: APP_LABEL[1], REGISTRY_LABEL_KEY: "someone_elses_registry"},
    )
    (verdict,) = _classify([other], {})
    assert not verdict.reap
    assert "someone_elses_registry" in verdict.reason


def test_a_cluster_already_stopping_is_left_to_finish():
    (verdict,) = _classify([_cluster("sf-ray-demo-abc123", state="STOPPING")], {})
    assert not verdict.reap
    assert "STOPPING" in verdict.reason


def test_a_brand_new_cluster_with_no_header_yet_survives_the_race():
    """The one race the age floor is for: created seconds ago, header not yet in BigQuery.

    Without the floor the reaper would delete the cluster of a run that is still starting up, which
    is the same catastrophic outcome as deleting a live one — just harder to reproduce.
    """
    (verdict,) = _classify([_cluster("sf-ray-demo-abc123", age_s=20.0)], {})
    assert not verdict.reap
    assert "20s old" in verdict.reason


def test_an_old_cluster_with_no_header_at_all_is_garbage():
    """Past the floor, an unknown run means the header never landed — nothing will ever claim it."""
    (verdict,) = _classify([_cluster("sf-ray-demo-abc123", age_s=7200.0)], {})
    assert verdict.reap
    assert "no run header" in verdict.reason


def test_the_age_floor_does_not_protect_a_cluster_whose_run_already_finished():
    """A finished run plus a standing cluster is a failed teardown, and waiting changes nothing.

    The floor exists only to cover the not-yet-written-header window. Applying it to a run that is
    demonstrably over would keep an accelerator billing for another half hour for no reason.
    """
    young = _cluster("sf-ray-demo-abc123", age_s=5.0)
    (verdict,) = _classify([young], {"demo-abc123": "COMPLETED"})
    assert verdict.reap


def test_a_cluster_vertex_gave_no_create_time_for_is_left_alone():
    """Unaged is unjudgeable. Refusing costs money; guessing zero would delete a starting run."""
    (verdict,) = _classify([_cluster("sf-ray-demo-abc123", age_s=None)], {})
    assert not verdict.reap
    assert "no create time" in verdict.reason


# --- clamped names and ambiguity -----------------------------------------------------


def test_a_clamped_cluster_name_still_finds_its_run():
    """`ray_io.cluster_name` truncates at 63 chars, so the name can hold only a prefix of the id.

    Matching by equality would call every long-named run "unknown" and reap its cluster while the
    run was still going — the prefix match is load-bearing, not a nicety.
    """
    run_id = "a" * 70 + "-abc123def456"
    from scale_forecasting.engines.ray_io import EPHEMERAL_PREFIX

    clamped = f"{EPHEMERAL_PREFIX}{run_id}"[:63].rstrip("-")
    (verdict,) = _classify([_cluster(clamped)], {run_id: "RUNNING"})
    assert verdict.matched_runs == (run_id,)
    assert not verdict.reap


def test_an_ambiguous_prefix_with_one_live_run_keeps_the_cluster():
    """When a clamped name matches several runs, one live match leaves the machine up."""
    (verdict,) = _classify(
        [_cluster("sf-ray-demo")], {"demo-abc123": "COMPLETED", "demo-def456": "RUNNING"}
    )
    assert verdict.matched_runs == ("demo-abc123", "demo-def456")
    assert not verdict.reap


def test_an_ambiguous_prefix_whose_runs_all_finished_is_still_garbage():
    (verdict,) = _classify(
        [_cluster("sf-ray-demo")], {"demo-abc123": "COMPLETED", "demo-def456": "FAILED"}
    )
    assert verdict.reap


# --- the plan and its preview ---------------------------------------------------------


def _plan(candidates):
    return ray_reaper.ReapPlan(
        registry="proj.scale_forecasting", regions=("us-central1",), candidates=tuple(candidates)
    )


def test_the_plan_splits_what_it_will_delete_from_what_it_is_keeping():
    candidates = _classify(
        [_cluster("sf-ray-done-1"), _cluster("sf-ray-live-2")],
        {"done-1": "COMPLETED", "live-2": "RUNNING"},
    )
    plan = _plan(candidates)
    assert [c.cluster.name for c in plan.reapable] == ["sf-ray-done-1"]
    assert [c.cluster.name for c in plan.kept] == ["sf-ray-live-2"]
    assert not plan.is_empty


def test_a_plan_that_deletes_nothing_is_empty_even_when_it_saw_clusters():
    plan = _plan(_classify([_cluster("sf-ray-live-2")], {"live-2": "RUNNING"}))
    assert plan.is_empty
    assert plan.kept


def test_the_preview_says_why_each_kept_cluster_was_kept():
    """ "Why is that T4 still up?" is what an operator arrives asking, so the answer is printed.

    A preview listing only the doomed clusters would show an empty delete list and nothing else,
    which reads as "there is nothing running" — the opposite of the truth.
    """
    text = ray_reaper.format_reap_plan(
        _plan(_classify([_cluster("sf-ray-live-2")], {"live-2": "RUNNING"}))
    )
    assert "leaving alone (1)" in text
    assert "sf-ray-live-2" in text
    assert "still in flight" in text


def test_the_preview_names_the_region_and_the_floor():
    text = ray_reaper.format_reap_plan(_plan(()))
    assert "us-central1" in text
    assert "1800s" in text
    assert "no scale-forecasting Ray clusters found" in text


# --- the labels the whole thing hangs off -----------------------------------------------


def _settings(dataset_id: str) -> Settings:
    return Settings(
        project_id="p",
        dataset_id=dataset_id,
        connection="p.us-central1.c",
        warehouse_uri="gs://b/w",
    )


def test_our_clusters_are_labelled_with_the_registry_that_owns_them():
    labels = cluster_labels(_settings("scale_forecasting"))
    assert labels == {"app": "scale-forecasting", "registry": "scale_forecasting"}


def test_a_dataset_id_vertex_would_reject_is_folded_into_a_legal_label():
    """BigQuery allows uppercase in a dataset id; Vertex label values do not. Reject-on-create would
    break every run in a deployment that used one, so the value is sanitized rather than passed."""
    labels = cluster_labels(_settings("Scale.Forecasting"))
    assert labels["registry"] == "scale-forecasting"


def test_a_cluster_from_before_the_registry_label_is_still_judged_on_its_run():
    """Older clusters carry only the ``app`` label. They must not become permanently unreapable."""
    legacy = _cluster("sf-ray-demo-abc123", labels={APP_LABEL[0]: APP_LABEL[1]})
    (verdict,) = _classify([legacy], {"demo-abc123": "COMPLETED"})
    assert verdict.reap
