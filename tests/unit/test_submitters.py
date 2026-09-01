"""Offline tests for the runtime-submitter spine (``scale_forecasting.submitters``).

The registry maps ``cfg.python_runtime`` to a `RuntimeSubmitter`, and each submitter's ``launch``
forwards the executed subset + contributor-mode header flag to the right path. The live submit paths
(remote Dataproc batch / Ray job) are the ``@gcp`` smokes; here every path a launch reaches is
faked, so no GCP or Spark/Ray extra is touched.
"""

from __future__ import annotations

from typing import Any

import pytest

from scale_forecasting.config import RunConfig
from scale_forecasting.errors import ConfigError
from scale_forecasting.probes.vocabulary import ProbeHandle
from scale_forecasting.settings import Settings
from scale_forecasting.submitters import (
    RaySubmitter,
    SparkSubmitter,
    get_submitter,
)

_SPARK = "theta"
_SETTINGS = Settings(
    project_id="proj-x",
    connection="proj-x.us-central1.conn",
    warehouse_uri="gs://bkt/warehouse",
)


def _cfg(**over: Any) -> RunConfig:
    base: dict[str, Any] = {
        "run_name": "submitter test",
        "data": {"source_table": "source_series_native", "horizon": 7, "series_limit": 5},
        "models": [_SPARK],
    }
    base.update(over)
    return RunConfig(**base)


def test_get_submitter_maps_runtime_to_implementation() -> None:
    assert isinstance(get_submitter("spark"), SparkSubmitter)
    assert isinstance(get_submitter("ray"), RaySubmitter)


def test_get_submitter_unknown_runtime_raises() -> None:
    with pytest.raises(ConfigError, match="no runtime submitter"):
        get_submitter("nope")


def test_spark_launch_no_session_submits_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    import scale_forecasting.submit as submit_mod

    seen: dict[str, Any] = {}

    def _fake_submit_batch(cfg: RunConfig, **kw: Any) -> str:
        seen.update(kw)
        return "batch-1"

    monkeypatch.setattr(submit_mod, "submit_batch", _fake_submit_batch)

    handle = SparkSubmitter().launch(
        _cfg(),
        models=[_SPARK],
        manage_header=False,
        settings=_SETTINGS,
    )
    assert seen["models"] == [_SPARK]
    assert seen["manage_header"] is False
    assert "engine" not in seen  # the Spark engine is built in — no method flag threaded through
    assert seen["batch_id"] is None  # standalone: no per-family id → submit derives one from run_id
    # A Serverless launch reports a single-region spark handle for later probing.
    assert handle == ProbeHandle(
        "spark", native_id=None, region="us-central1", spark_mode="serverless"
    )


def test_spark_launch_threads_system_job_id_as_batch_id(monkeypatch: pytest.MonkeyPatch) -> None:
    # The orchestrator's deterministic per-family id becomes the Dataproc batch_id, so families
    # sharing one run_id submit distinct batches instead of colliding on a run-derived id.
    import scale_forecasting.submit as submit_mod

    seen: dict[str, Any] = {}
    monkeypatch.setattr(submit_mod, "submit_batch", lambda cfg, **kw: seen.update(kw) or "b")

    handle = SparkSubmitter().launch(
        _cfg(),
        models=[_SPARK],
        manage_header=False,
        settings=_SETTINGS,
        system_job_id="sf-run-abc-statistical-a1",
    )
    assert seen["batch_id"] == "sf-run-abc-statistical-a1"
    # Serverless batch id == system_job_id (we set it): the handle's native_id is that same id.
    assert handle == ProbeHandle(
        "spark",
        native_id="sf-run-abc-statistical-a1",
        region="us-central1",
        spark_mode="serverless",
    )


def test_spark_launch_threads_gpu_to_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    import scale_forecasting.submit as submit_mod

    seen: dict[str, Any] = {}
    monkeypatch.setattr(submit_mod, "submit_batch", lambda cfg, **kw: seen.update(kw) or "b")

    SparkSubmitter().launch(
        _cfg(),
        models=[_SPARK],
        manage_header=False,
        settings=_SETTINGS,
        hardware="gpu",
        gpu_type="L4",
    )
    assert seen["hardware"] == "gpu"
    assert seen["gpu_type"] == "L4"


def test_spark_launch_cluster_mode_submits_cluster_job(monkeypatch: pytest.MonkeyPatch) -> None:
    # spark_mode="cluster" routes to the Dataproc cluster submitter (not the Serverless batch),
    # threading the per-family id, the GPU sizing, and the reuse target through.
    import scale_forecasting.cluster_submit as cluster_mod

    seen: dict[str, Any] = {}
    # The fake returns (server-assigned id, landed region); the id is distinct from the
    # deterministic one passed in, and the region is where the job actually ran.
    monkeypatch.setattr(
        cluster_mod,
        "submit_cluster_job",
        lambda cfg, **kw: seen.update(kw) or ("real-dataproc-job-id", "us-west1"),
    )

    handle = SparkSubmitter().launch(
        _cfg(),
        models=[_SPARK],
        manage_header=False,
        settings=_SETTINGS,
        system_job_id="sf-run-abc-statistical-a1",
        hardware="gpu",
        gpu_type="T4",
        spark_mode="cluster",
        spark_cluster_name="warm-cluster",
    )
    assert seen["job_id"] == "sf-run-abc-statistical-a1"
    assert seen["hardware"] == "gpu"
    assert seen["gpu_type"] == "T4"
    assert seen["spark_cluster_name"] == "warm-cluster"
    # The cluster path carries the real server-assigned id + landed region in the handle so the
    # orchestrator can stamp the real id back and probe the job later.
    assert handle == ProbeHandle(
        "spark", native_id="real-dataproc-job-id", region="us-west1", spark_mode="cluster"
    )


def test_spark_launch_with_session_ignores_system_job_id(monkeypatch: pytest.MonkeyPatch) -> None:
    # An in-process session submits no remote batch, so the batch id has nowhere to go.
    from scale_forecasting.engines import spark_explode

    seen: dict[str, Any] = {}
    monkeypatch.setattr(spark_explode, "run", lambda cfg, **kw: seen.update(kw))

    handle = SparkSubmitter().launch(
        _cfg(),
        models=[_SPARK],
        manage_header=False,
        settings=_SETTINGS,
        spark=object(),
        system_job_id="sf-run-abc-statistical-a1",
    )
    assert "batch_id" not in seen
    assert "system_job_id" not in seen
    assert handle is None  # in-process session submits nothing → no handle


def test_ray_launch_threads_system_job_id_as_submission_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scale_forecasting.ray_submit as ray_submit_mod

    seen: dict[str, Any] = {}
    resource = "projects/p/locations/us-west1/pr/c"
    monkeypatch.setattr(
        ray_submit_mod,
        "submit_ray",
        lambda cfg, **kw: seen.update(kw) or ("j", resource, "us-west1"),
    )

    handle = RaySubmitter().launch(
        _cfg(python_runtime="ray"),
        models=[_SPARK],
        manage_header=False,
        settings=_SETTINGS,
        system_job_id="sf-run-abc-ml-a1",
    )
    assert seen["submission_id"] == "sf-run-abc-ml-a1"
    # Ray submission_id == system_job_id (we set it); the handle carries the resource path + region.
    assert handle == ProbeHandle(
        "ray",
        native_id="j",
        region="us-west1",
        resource_name=resource,
    )


def test_ray_launch_maps_hardware_to_use_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    import scale_forecasting.ray_submit as ray_submit_mod

    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        ray_submit_mod, "submit_ray", lambda cfg, **kw: seen.update(kw) or ("j", "rn", "us-west1")
    )

    RaySubmitter().launch(
        _cfg(python_runtime="ray"),
        models=[_SPARK],
        manage_header=False,
        settings=_SETTINGS,
        hardware="gpu",
        gpu_type="L4",
    )
    assert seen["use_gpu"] is True
    assert seen["gpu_type"] == "L4"


def test_ray_launch_threads_shared_cluster_target(monkeypatch: pytest.MonkeyPatch) -> None:
    # A shared ephemeral cluster's (name, region) are passed through as the reuse target.
    import scale_forecasting.ray_submit as ray_submit_mod

    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        ray_submit_mod, "submit_ray", lambda cfg, **kw: seen.update(kw) or ("j", "rn", "us-west1")
    )

    RaySubmitter().launch(
        _cfg(python_runtime="ray"),
        models=[_SPARK],
        manage_header=False,
        settings=_SETTINGS,
        ray_cluster_name="sf-ray-shared",
        ray_cluster_region="us-west1",
    )
    assert seen["cluster_name"] == "sf-ray-shared"
    assert seen["cluster_region"] == "us-west1"


def test_spark_launch_with_session_runs_in_process(monkeypatch: pytest.MonkeyPatch) -> None:
    # An injected session picks the in-process engine (explode by default) — no remote batch submit.
    from scale_forecasting.engines import spark_explode

    seen: dict[str, Any] = {}

    def _fake_run(cfg: RunConfig, **kw: Any) -> None:
        seen.update(kw)

    monkeypatch.setattr(spark_explode, "run", _fake_run)

    sentinel = object()
    SparkSubmitter().launch(
        _cfg(),
        models=[_SPARK],
        manage_header=False,
        settings=_SETTINGS,
        spark=sentinel,
    )
    assert seen["spark"] is sentinel
    assert seen["models"] == [_SPARK]
    assert seen["manage_header"] is False


def test_ray_launch_submits_ray_ignoring_session(monkeypatch: pytest.MonkeyPatch) -> None:
    import scale_forecasting.ray_submit as ray_submit_mod

    seen: dict[str, Any] = {}

    def _fake_submit_ray(cfg: RunConfig, **kw: Any) -> tuple[str, str, str]:
        seen.update(kw)
        return "job-1", "rn", "us-west1"

    monkeypatch.setattr(ray_submit_mod, "submit_ray", _fake_submit_ray)

    RaySubmitter().launch(
        _cfg(python_runtime="ray"),
        models=[_SPARK],
        manage_header=False,
        settings=_SETTINGS,
        spark=object(),  # ignored by Ray
    )
    assert "engine" not in seen  # ray takes no spark engine arg
    assert seen["models"] == [_SPARK]
    assert seen["manage_header"] is False
