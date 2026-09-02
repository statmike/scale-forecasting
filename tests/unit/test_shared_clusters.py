"""Offline tests for the run-level shared ephemeral cluster (``scale_forecasting.shared_clusters``).

Two symmetric pairs — Ray and Dataproc — each a pure predicate plus the context manager that acts on
it. The predicate tests are the interesting half: sharing engages at **two or more** eligible
families and nowhere else, and the cluster it sizes covers the union of exactly those families'
models (never a run's Ray or BigQuery-native work, which lands elsewhere). The bracket tests prove
engage / skip / tear-down-even-on-exception, and the last few prove the yielded ``(name, region)``
actually reaches `job_launch.launch_family_job` for the runtimes that can use it — and is ignored by
the ones that cannot.

All offline: provision and teardown are faked, so nothing touches Vertex or Dataproc.
"""

from __future__ import annotations

from typing import Any

import pytest

from scale_forecasting import dag, job_launch, shared_clusters
from scale_forecasting.config import RunConfig
from scale_forecasting.settings import Settings

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
        "models": ["theta", "arima_plus", "timesfm"],
    }
    base.update(over)
    return RunConfig(**base)


# --- shared ephemeral Ray cluster across families ------------------------------
# A run with two or more ephemeral Ray families shares ONE cluster: the orchestrator provisions it,
# each family submits its own failure-isolated job to it, and it's torn down once. These tests cover
# the pure sizing helper, the context manager's engage/skip/teardown behavior, and the per-family
# threading — all offline (provision/teardown are faked; nothing touches Vertex).


def _ray_cfg(**over: Any) -> RunConfig:
    # theta (statistical) + xgboost (ml) both resolve to Ray → two ephemeral Ray families.
    base: dict[str, Any] = {
        "run_name": "shared ray test",
        "data": {"source_table": "source_series_native", "horizon": 7, "series_limit": 5},
        "models": ["theta", "xgboost"],
        "python_runtime": "ray",
    }
    base.update(over)
    return RunConfig(**base)


def test_shared_ray_inputs_none_for_single_ray_family() -> None:
    # One Ray family self-provisions (no collision risk); sharing doesn't apply.
    run_dag = dag.plan_dag(_ray_cfg(models=["theta"]))
    assert shared_clusters.shared_ray_inputs(run_dag.python_jobs) is None


def test_shared_ray_inputs_none_when_no_ray_family() -> None:
    run_dag = dag.plan_dag(_cfg(models=["theta", "holtwinters"]))  # default spark
    assert shared_clusters.shared_ray_inputs(run_dag.python_jobs) is None


def test_shared_ray_inputs_unions_models_cpu() -> None:
    run_dag = dag.plan_dag(_ray_cfg())
    inputs = shared_clusters.shared_ray_inputs(run_dag.python_jobs)
    assert inputs is not None
    models, any_gpu, gpu_type = inputs
    assert sorted(models) == ["theta", "xgboost"]
    assert any_gpu is False
    assert gpu_type is None


def test_shared_ray_inputs_flags_gpu_from_deep_learning() -> None:
    # theta (statistical, cpu) + neuralprophet (deep_learning, gpu) both on Ray.
    run_dag = dag.plan_dag(
        _ray_cfg(models=["theta", "neuralprophet"], compute={"use_gpu": True, "gpu_type": "T4"})
    )
    inputs = shared_clusters.shared_ray_inputs(run_dag.python_jobs)
    assert inputs is not None
    models, any_gpu, gpu_type = inputs
    assert sorted(models) == ["neuralprophet", "theta"]
    assert any_gpu is True
    assert gpu_type == "T4"


def _patch_shared_cluster(monkeypatch: pytest.MonkeyPatch, calls: dict[str, Any]) -> None:
    from scale_forecasting import ray_cluster

    def _provision(cfg: RunConfig, **kw: Any) -> tuple[str, str]:
        calls["provision"] = kw
        return "sf-ray-shared", "us-west1"

    def _teardown(name: str, region: str, settings: Settings) -> None:
        calls["teardown"] = (name, region)

    monkeypatch.setattr(ray_cluster, "provision_shared_cluster", _provision)
    monkeypatch.setattr(ray_cluster, "teardown_shared_cluster", _teardown)


def test_shared_ray_cluster_engages_and_tears_down(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}
    _patch_shared_cluster(monkeypatch, calls)
    cfg = _ray_cfg()
    run_dag = dag.plan_dag(cfg)
    with shared_clusters.shared_ray_cluster(cfg, run_dag, "run-abc", _SETTINGS) as ray_cluster:
        assert ray_cluster == ("sf-ray-shared", "us-west1")
        assert "provision" in calls
        assert "teardown" not in calls  # not yet — torn down on exit
    assert calls["teardown"] == ("sf-ray-shared", "us-west1")


def test_shared_ray_cluster_tears_down_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}
    _patch_shared_cluster(monkeypatch, calls)
    cfg = _ray_cfg()
    run_dag = dag.plan_dag(cfg)
    with pytest.raises(RuntimeError, match="boom"):
        with shared_clusters.shared_ray_cluster(cfg, run_dag, "run-abc", _SETTINGS):
            raise RuntimeError("boom")
    assert calls["teardown"] == ("sf-ray-shared", "us-west1")  # finally still ran


def test_shared_ray_cluster_skips_single_ray_family(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}
    _patch_shared_cluster(monkeypatch, calls)
    cfg = _ray_cfg(models=["theta"])
    run_dag = dag.plan_dag(cfg)
    with shared_clusters.shared_ray_cluster(cfg, run_dag, "run-abc", _SETTINGS) as ray_cluster:
        assert ray_cluster is None
    assert calls == {}  # never provisioned, never torn down


def test_shared_ray_cluster_skips_spark_run(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}
    _patch_shared_cluster(monkeypatch, calls)
    cfg = _cfg(models=["theta", "holtwinters"])  # default spark
    run_dag = dag.plan_dag(cfg)
    with shared_clusters.shared_ray_cluster(cfg, run_dag, "run-abc", _SETTINGS) as ray_cluster:
        assert ray_cluster is None
    assert calls == {}


def test_shared_ray_cluster_skips_when_standing_cluster_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A config reusing a standing cluster targets it directly; the orchestrator provisions nothing.
    calls: dict[str, Any] = {}
    _patch_shared_cluster(monkeypatch, calls)
    cfg = _ray_cfg(compute={"ray_cluster_name": "my-standing-ray"})
    run_dag = dag.plan_dag(cfg)
    with shared_clusters.shared_ray_cluster(cfg, run_dag, "run-abc", _SETTINGS) as ray_cluster:
        assert ray_cluster is None
    assert calls == {}


class _CapturingSubmitter:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def launch(self, cfg: RunConfig, **kw: Any) -> None:
        self.kwargs = kw


def _patch_launch_seams(monkeypatch: pytest.MonkeyPatch, sub: _CapturingSubmitter) -> None:
    import contextlib

    from scale_forecasting import submitters
    from scale_forecasting.registry import jobs, lifecycle

    monkeypatch.setattr(jobs, "next_job_attempt", lambda *a, **k: (1, None))

    @contextlib.contextmanager
    def _fake_run_job(*a: Any, **k: Any) -> Any:
        yield None

    monkeypatch.setattr(lifecycle, "run_job", _fake_run_job)
    monkeypatch.setattr(submitters, "get_submitter", lambda runtime: sub)


def test_launch_family_job_threads_shared_cluster_for_ray(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sub = _CapturingSubmitter()
    _patch_launch_seams(monkeypatch, sub)
    run_dag = dag.plan_dag(_ray_cfg())
    ray_job = next(j for j in run_dag.python_jobs if j.runtime == "ray")
    job_launch.launch_family_job(
        _ray_cfg(), ray_job, "run-abc", _SETTINGS, ray_cluster=("sf-ray-shared", "us-west1")
    )
    assert sub.kwargs["ray_cluster_name"] == "sf-ray-shared"
    assert sub.kwargs["ray_cluster_region"] == "us-west1"


def test_launch_family_job_ignores_shared_cluster_for_spark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sub = _CapturingSubmitter()
    _patch_launch_seams(monkeypatch, sub)
    cfg = _cfg(models=["theta", "holtwinters"])  # default spark
    run_dag = dag.plan_dag(cfg)
    spark_job = next(j for j in run_dag.python_jobs if j.runtime == "spark")
    job_launch.launch_family_job(
        cfg, spark_job, "run-abc", _SETTINGS, ray_cluster=("sf-ray-shared", "us-west1")
    )
    assert sub.kwargs["ray_cluster_name"] is None
    assert sub.kwargs["ray_cluster_region"] is None


# --- shared ephemeral Dataproc cluster across families -------------------------
# The Dataproc analog of the shared-Ray tests above: a run with two or more ephemeral cluster
# families shares clusters the orchestrator provisions (each family submits its own job to one as a
# reuse target, torn down once), so no family's teardown races another's job. One cluster per
# *hardware kind* rather than one per run, because a Dataproc cluster has a single worker machine
# type — the difference from Ray, whose one cluster carries both a CPU and a GPU worker pool.
# Offline — provision/teardown are faked; nothing touches Dataproc.


def _spark_cluster_cfg(**over: Any) -> RunConfig:
    # theta (statistical) + xgboost (ml), both forced to spark_mode=cluster → two ephemeral
    # cluster families sharing one run-derived cluster.
    base: dict[str, Any] = {
        "run_name": "shared spark cluster test",
        "data": {"source_table": "source_series_native", "horizon": 7, "series_limit": 5},
        "models": ["theta", "xgboost"],
        "compute": {
            "families": {
                "statistical": {"spark_mode": "cluster"},
                "ml": {"spark_mode": "cluster"},
            }
        },
    }
    base.update(over)
    return RunConfig(**base)


def test_shared_spark_inputs_none_for_single_cluster_family() -> None:
    # One ephemeral cluster family self-manages its own cluster (no collision); no sharing.
    cfg = _spark_cluster_cfg(
        models=["theta"], compute={"families": {"statistical": {"spark_mode": "cluster"}}}
    )
    run_dag = dag.plan_dag(cfg)
    assert shared_clusters.shared_spark_inputs(run_dag.python_jobs) is None


def test_shared_spark_inputs_none_for_serverless_run() -> None:
    run_dag = dag.plan_dag(_cfg(models=["theta", "holtwinters"]))  # default serverless spark
    assert shared_clusters.shared_spark_inputs(run_dag.python_jobs) is None


def test_shared_spark_inputs_engages_for_two_cluster_families_cpu() -> None:
    run_dag = dag.plan_dag(_spark_cluster_cfg())
    inputs = shared_clusters.shared_spark_inputs(run_dag.python_jobs)
    assert inputs is not None
    assert list(inputs) == ["cpu"]  # one group, so one cluster — the common case
    models, gpu_type = inputs["cpu"]  # only the cluster families' models
    assert sorted(models) == sorted(_spark_cluster_cfg().models)
    assert gpu_type is None


def _mixed_hardware_cfg() -> RunConfig:
    # statistical (cpu) + deep_learning (gpu T4), both on spark_mode=cluster.
    return _spark_cluster_cfg(
        models=["theta", "neuralprophet"],
        compute={
            "families": {
                "statistical": {"spark_mode": "cluster"},
                "deep_learning": {"spark_mode": "cluster", "hardware": "gpu", "gpu_type": "T4"},
            }
        },
    )


def test_shared_spark_inputs_splits_cpu_and_gpu_rather_than_unioning() -> None:
    """A mixed run sizes two clusters, not one GPU cluster for everyone.

    The behaviour this replaced collapsed the run to a single "does anyone need a GPU" flag, which
    is right for Ray (separate worker pools per cluster) and wrong here: a Dataproc cluster has one
    worker machine type, so a single GPU cluster would put an accelerator under theta's fits.
    """
    run_dag = dag.plan_dag(_mixed_hardware_cfg())
    assert shared_clusters.shared_spark_inputs(run_dag.python_jobs) == {
        "cpu": (["theta"], None),
        "gpu": (["neuralprophet"], "T4"),
    }


def test_shared_spark_inputs_orders_cpu_before_gpu_whatever_the_dag_order() -> None:
    # Fixed order, not insertion order: the provisioning sequence must not depend on which family
    # the planner happened to list first.
    run_dag = dag.plan_dag(_mixed_hardware_cfg())
    reversed_jobs = list(reversed(run_dag.python_jobs))
    inputs = shared_clusters.shared_spark_inputs(reversed_jobs)
    assert inputs is not None
    assert list(inputs) == ["cpu", "gpu"]


def test_shared_spark_inputs_ignores_family_with_standing_cluster() -> None:
    # A family naming its own standing cluster is already reuse; only one *ephemeral* family is
    # left, so sharing doesn't engage.
    cfg = _spark_cluster_cfg(
        compute={
            "families": {
                "statistical": {"spark_mode": "cluster"},
                "ml": {"spark_mode": "cluster", "spark_cluster_name": "my-standing"},
            }
        }
    )
    run_dag = dag.plan_dag(cfg)
    assert shared_clusters.shared_spark_inputs(run_dag.python_jobs) is None


def _patch_shared_spark(
    monkeypatch: pytest.MonkeyPatch, calls: dict[str, Any], *, fail_on: str | None = None
) -> None:
    """Fake provision/teardown, recording every call in order.

    ``fail_on`` makes the create for that hardware kind raise, which is how the partial-create
    unwind is exercised: the group that already came up must still be torn down.
    """
    from scale_forecasting import dataproc_cluster

    calls.setdefault("provision", [])
    calls.setdefault("teardown", [])

    def _provision(cfg: RunConfig, **kw: Any) -> tuple[str, str]:
        calls["provision"].append(kw)
        if fail_on is not None and kw["use_gpu"] == (fail_on == "gpu"):
            raise RuntimeError(f"capacity: {fail_on}")
        suffix = kw.get("name_suffix")
        return f"sf-cluster-shared{'-' + suffix if suffix else ''}", "us-central1"

    def _teardown(name: str, region: str, settings: Settings) -> None:
        calls["teardown"].append((name, region))

    monkeypatch.setattr(dataproc_cluster, "provision_shared_cluster", _provision)
    monkeypatch.setattr(dataproc_cluster, "teardown_shared_cluster", _teardown)


def test_shared_spark_cluster_engages_and_tears_down(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}
    _patch_shared_spark(monkeypatch, calls)
    cfg = _spark_cluster_cfg()
    run_dag = dag.plan_dag(cfg)
    with shared_clusters.shared_spark_cluster(cfg, run_dag, "run-abc", _SETTINGS) as spark_cluster:
        assert spark_cluster == {"cpu": ("sf-cluster-shared", "us-central1")}
        assert calls["provision"] == [
            {
                "run_id": "run-abc",
                "use_gpu": False,
                "gpu_type": None,
                "settings": _SETTINGS,
                # Sized against the cluster families' models only — see shared_spark_inputs.
                "models": ["theta", "xgboost"],
                # One hardware kind, so no suffix: the name is what it has always been.
                "name_suffix": None,
            }
        ]
        assert calls["teardown"] == []  # not yet — torn down on exit
    assert calls["teardown"] == [("sf-cluster-shared", "us-central1")]


def test_shared_spark_cluster_provisions_one_cluster_per_hardware_kind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}
    _patch_shared_spark(monkeypatch, calls)
    cfg = _mixed_hardware_cfg()
    run_dag = dag.plan_dag(cfg)
    with shared_clusters.shared_spark_cluster(cfg, run_dag, "run-abc", _SETTINGS) as spark_cluster:
        assert spark_cluster == {
            "cpu": ("sf-cluster-shared-cpu", "us-central1"),
            "gpu": ("sf-cluster-shared-gpu", "us-central1"),
        }
        # Each cluster is sized for its own group's models and its own hardware — the CPU cluster
        # never sees the GPU family's fan-out, and buys no accelerators for it.
        assert [(c["use_gpu"], c["models"], c["name_suffix"]) for c in calls["provision"]] == [
            (False, ["theta"], "cpu"),
            (True, ["neuralprophet"], "gpu"),
        ]
    # Both torn down, in reverse creation order — the ExitStack unwinds LIFO. Order is incidental
    # here (the clusters are independent); what matters is that neither is left behind.
    assert calls["teardown"] == [
        ("sf-cluster-shared-gpu", "us-central1"),
        ("sf-cluster-shared-cpu", "us-central1"),
    ]


def test_shared_spark_cluster_tears_down_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}
    _patch_shared_spark(monkeypatch, calls)
    cfg = _spark_cluster_cfg()
    run_dag = dag.plan_dag(cfg)
    with pytest.raises(RuntimeError, match="boom"):
        with shared_clusters.shared_spark_cluster(cfg, run_dag, "run-abc", _SETTINGS):
            raise RuntimeError("boom")
    assert calls["teardown"] == [("sf-cluster-shared", "us-central1")]  # unwind still ran


def test_shared_spark_cluster_unwinds_a_partial_create(monkeypatch: pytest.MonkeyPatch) -> None:
    """A GPU create that fails after the CPU cluster came up must not leak the CPU cluster.

    The reason the bracket is an ``ExitStack`` and not a single ``finally``: with two clusters there
    is a window where one exists and the other is still being created, and the failure that closes
    that window is exactly the one nobody is watching for.
    """
    calls: dict[str, Any] = {}
    _patch_shared_spark(monkeypatch, calls, fail_on="gpu")
    cfg = _mixed_hardware_cfg()
    run_dag = dag.plan_dag(cfg)
    with pytest.raises(RuntimeError, match="capacity: gpu"):
        with shared_clusters.shared_spark_cluster(cfg, run_dag, "run-abc", _SETTINGS):
            pytest.fail("body must not run — the second create failed")
    assert calls["teardown"] == [("sf-cluster-shared-cpu", "us-central1")]


def test_shared_spark_cluster_skips_single_family(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}
    _patch_shared_spark(monkeypatch, calls)
    cfg = _spark_cluster_cfg(
        models=["theta"], compute={"families": {"statistical": {"spark_mode": "cluster"}}}
    )
    run_dag = dag.plan_dag(cfg)
    with shared_clusters.shared_spark_cluster(cfg, run_dag, "run-abc", _SETTINGS) as spark_cluster:
        assert spark_cluster is None
    assert calls == {"provision": [], "teardown": []}  # never provisioned, never torn down


def test_launch_family_job_threads_shared_spark_cluster(monkeypatch: pytest.MonkeyPatch) -> None:
    sub = _CapturingSubmitter()
    _patch_launch_seams(monkeypatch, sub)
    cfg = _spark_cluster_cfg()
    run_dag = dag.plan_dag(cfg)
    cluster_job = next(
        j for j in run_dag.python_jobs if j.compute and j.compute.spark_mode == "cluster"
    )
    job_launch.launch_family_job(
        cfg,
        cluster_job,
        "run-abc",
        _SETTINGS,
        spark_cluster={"cpu": ("sf-cluster-shared", "us-east4")},
    )
    assert sub.kwargs["spark_cluster_name"] == "sf-cluster-shared"
    assert sub.kwargs["spark_cluster_region"] == "us-east4"


def test_launch_family_job_picks_the_cluster_matching_its_own_hardware(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two clusters offered; the GPU family must land on the GPU one — including its region, which
    # may differ if one cluster capacity-hopped and the other did not.
    sub = _CapturingSubmitter()
    _patch_launch_seams(monkeypatch, sub)
    cfg = _mixed_hardware_cfg()
    run_dag = dag.plan_dag(cfg)
    gpu_job = next(j for j in run_dag.python_jobs if j.compute and j.compute.hardware == "gpu")
    job_launch.launch_family_job(
        cfg,
        gpu_job,
        "run-abc",
        _SETTINGS,
        spark_cluster={
            "cpu": ("sf-cluster-shared-cpu", "us-central1"),
            "gpu": ("sf-cluster-shared-gpu", "us-east4"),
        },
    )
    assert sub.kwargs["spark_cluster_name"] == "sf-cluster-shared-gpu"
    assert sub.kwargs["spark_cluster_region"] == "us-east4"


def test_launch_family_job_self_provisions_when_no_cluster_matches_its_hardware(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A miss is not fatal: the family falls back to self-provisioning, which is what it did before
    # sharing existed. Threading a mismatched cluster instead would put the work on wrong hardware.
    sub = _CapturingSubmitter()
    _patch_launch_seams(monkeypatch, sub)
    cfg = _mixed_hardware_cfg()
    run_dag = dag.plan_dag(cfg)
    gpu_job = next(j for j in run_dag.python_jobs if j.compute and j.compute.hardware == "gpu")
    job_launch.launch_family_job(
        cfg,
        gpu_job,
        "run-abc",
        _SETTINGS,
        spark_cluster={"cpu": ("sf-cluster-shared-cpu", "us-central1")},
    )
    assert sub.kwargs["spark_cluster_name"] is None
    assert sub.kwargs["spark_cluster_region"] is None


def test_launch_family_job_ignores_shared_spark_for_serverless(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sub = _CapturingSubmitter()
    _patch_launch_seams(monkeypatch, sub)
    cfg = _cfg(models=["theta", "holtwinters"])  # default serverless spark
    run_dag = dag.plan_dag(cfg)
    spark_job = next(j for j in run_dag.python_jobs if j.runtime == "spark")
    job_launch.launch_family_job(
        cfg,
        spark_job,
        "run-abc",
        _SETTINGS,
        spark_cluster={"cpu": ("sf-cluster-shared", "us-central1")},
    )
    assert sub.kwargs["spark_cluster_name"] is None
    assert sub.kwargs["spark_cluster_region"] is None


def test_launch_family_job_keeps_standing_cluster_over_shared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A family naming its own standing cluster keeps it even when a shared cluster is offered.
    sub = _CapturingSubmitter()
    _patch_launch_seams(monkeypatch, sub)
    cfg = _spark_cluster_cfg(
        compute={
            "families": {"statistical": {"spark_mode": "cluster", "spark_cluster_name": "standing"}}
        },
        models=["theta"],
    )
    run_dag = dag.plan_dag(cfg)
    cluster_job = next(j for j in run_dag.python_jobs if j.compute)
    job_launch.launch_family_job(
        cfg,
        cluster_job,
        "run-abc",
        _SETTINGS,
        spark_cluster={"cpu": ("sf-cluster-shared", "us-central1")},
    )
    assert sub.kwargs["spark_cluster_name"] == "standing"
    # A standing-cluster family isn't the shared-cluster reuser, so no shared region is threaded.
    assert sub.kwargs["spark_cluster_region"] is None
