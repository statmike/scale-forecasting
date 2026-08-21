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

    SparkSubmitter().launch(
        _cfg(),
        models=[_SPARK],
        manage_header=False,
        settings=_SETTINGS,
    )
    assert seen["engine"] == "explode"
    assert seen["models"] == [_SPARK]
    assert seen["manage_header"] is False
    assert seen["batch_id"] is None  # standalone: no per-family id → submit derives one from run_id


def test_spark_launch_threads_system_job_id_as_batch_id(monkeypatch: pytest.MonkeyPatch) -> None:
    # The orchestrator's deterministic per-family id becomes the Dataproc batch_id, so families
    # sharing one run_id submit distinct batches instead of colliding on a run-derived id.
    import scale_forecasting.submit as submit_mod

    seen: dict[str, Any] = {}
    monkeypatch.setattr(submit_mod, "submit_batch", lambda cfg, **kw: seen.update(kw) or "b")

    SparkSubmitter().launch(
        _cfg(),
        models=[_SPARK],
        manage_header=False,
        settings=_SETTINGS,
        system_job_id="sf-run-abc-statistical-a1",
    )
    assert seen["batch_id"] == "sf-run-abc-statistical-a1"


def test_spark_launch_with_session_ignores_system_job_id(monkeypatch: pytest.MonkeyPatch) -> None:
    # An in-process session submits no remote batch, so the batch id has nowhere to go.
    from scale_forecasting.engines import spark_explode

    seen: dict[str, Any] = {}
    monkeypatch.setattr(spark_explode, "run", lambda cfg, **kw: seen.update(kw))

    SparkSubmitter().launch(
        _cfg(),
        models=[_SPARK],
        manage_header=False,
        settings=_SETTINGS,
        spark=object(),
        system_job_id="sf-run-abc-statistical-a1",
    )
    assert "batch_id" not in seen
    assert "system_job_id" not in seen


def test_ray_launch_threads_system_job_id_as_submission_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scale_forecasting.ray_submit as ray_submit_mod

    seen: dict[str, Any] = {}
    monkeypatch.setattr(ray_submit_mod, "submit_ray", lambda cfg, **kw: seen.update(kw) or "j")

    RaySubmitter().launch(
        _cfg(python_runtime="ray"),
        models=[_SPARK],
        manage_header=False,
        settings=_SETTINGS,
        system_job_id="sf-run-abc-ml-a1",
    )
    assert seen["submission_id"] == "sf-run-abc-ml-a1"


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

    def _fake_submit_ray(cfg: RunConfig, **kw: Any) -> str:
        seen.update(kw)
        return "job-1"

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
