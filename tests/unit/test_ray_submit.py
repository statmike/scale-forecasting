"""Offline tests for the Vertex Ray submit helper (BUILD B4, ``scale_forecasting.ray_submit``).

No network: the pure job-spec assembly (:func:`build_entrypoint`, :func:`build_runtime_env`,
:func:`extract_ray_telemetry`), infra resolution (:class:`RayInfra`), and the whole cluster
lifecycle with ``vertex_ray`` + ``JobSubmissionClient`` monkeypatched — asserting the two load-
bearing lifecycle properties: **ephemeral creates then deletes (even when the job raises)**, and
**reuse skips both create and delete**. The live T4 path is the ``@gpu`` smoke in
``tests/integration/test_ray_gpu_smoke.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from scale_forecasting import ray_submit
from scale_forecasting.config import RunConfig
from scale_forecasting.engines import ray_io
from scale_forecasting.errors import ConfigError, EngineError
from scale_forecasting.registry.ids import make_run_id
from scale_forecasting.settings import Settings

_CPU = "theta"
_GPU = "neuralprophet"


def _cfg(**over: Any) -> RunConfig:
    base: dict[str, Any] = {
        "run_name": "ray submit test",
        "python_runtime": "ray",
        "data": {"source_table": "source_series", "horizon": 7, "series_limit": 8},
        "models": [_CPU, _GPU],
        "compute": {"use_gpu": True},
    }
    base.update(over)
    return RunConfig(**base)


def _settings() -> Settings:
    return Settings(
        project_id="proj-x",
        connection="proj-x.us-central1.conn",
        warehouse_uri="gs://bkt/warehouse",
        dataset_id="ds_x",
        region="us-central1",
    )


def _infra(**over: Any) -> ray_submit.RayInfra:
    base: dict[str, Any] = {
        "network": "projects/proj-x/global/networks/scale-forecasting",
        "compute_sa": "compute@proj-x.iam.gserviceaccount.com",
        "code_bucket": "code-bkt",
    }
    base.update(over)
    return ray_submit.RayInfra(**base)


# --- build_entrypoint: the on-cluster command ----------------------------------


def test_build_entrypoint_wires_config_and_infra_args() -> None:
    ep = ray_submit.build_entrypoint("gs://code-bkt/runs/run-abc.json", _settings())
    assert ep.startswith("python -m scale_forecasting.ray_entry --config-uri ")
    assert "gs://code-bkt/runs/run-abc.json" in ep
    assert "--sf-project-id proj-x" in ep


def test_build_entrypoint_defaults_omit_arc_b_flags() -> None:
    # Standalone submit (no subset, header-owning) omits both Arc B flags, parity with build_batch.
    ep = ray_submit.build_entrypoint("gs://c/r.json", _settings())
    assert "--models" not in ep
    assert "--manage-header" not in ep


def test_build_entrypoint_appends_arc_b_flags_when_non_default() -> None:
    ep = ray_submit.build_entrypoint(
        "gs://c/r.json", _settings(), models=[_CPU, _GPU], manage_header=False
    )
    assert "--models theta,neuralprophet" in ep
    assert "--manage-header false" in ep


# --- build_runtime_env: runtime code delivery ----------------------------------


def test_build_runtime_env_ships_src_and_requirements() -> None:
    import os

    env = ray_submit.build_runtime_env()
    # working_dir is the package root (so `python -m scale_forecasting.ray_entry` resolves), pip the
    # locked cluster deps — both must be real local paths that exist for the upload to succeed.
    assert env["working_dir"].endswith("/src")
    assert os.path.isdir(env["working_dir"])
    assert env["pip"].endswith("docker/requirements.txt")
    assert os.path.isfile(env["pip"])


# --- RayInfra resolution -------------------------------------------------------


def test_ray_infra_resolve_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SF_RAY_NETWORK", "projects/p/global/networks/n")
    monkeypatch.setenv("SF_COMPUTE_SA", "sa@x.iam")
    monkeypatch.setenv("SF_CODE_BUCKET", "code-bkt")
    monkeypatch.delenv("SF_CONTAINER_IMAGE", raising=False)
    infra = ray_submit.RayInfra.resolve()
    assert infra.network.endswith("/networks/n")
    assert infra.code_bucket == "code-bkt"
    assert infra.container_image is None  # unset → Vertex prebuilt image + requirements
    assert infra.ray_version == ray_submit._DEFAULT_RAY_VERSION


def test_ray_infra_resolve_missing_var_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("SF_RAY_NETWORK", "SF_COMPUTE_SA", "SF_CODE_BUCKET"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ConfigError, match="SF_RAY_NETWORK"):
        ray_submit.RayInfra.resolve()


def test_ray_infra_from_terraform_outputs_with_and_without_image() -> None:
    outputs = {
        "network_id": "projects/p/global/networks/scale-forecasting",
        "compute_sa": "sa@x.iam",
        "code_bucket": "code-bkt",
        "runtime_image_repo": "us-docker.pkg.dev/p/repo/runtime",
    }
    no_image = ray_submit.RayInfra.from_terraform_outputs(outputs)
    assert no_image.container_image is None
    with_image = ray_submit.RayInfra.from_terraform_outputs(outputs, image_tag="v1")
    assert with_image.container_image == "us-docker.pkg.dev/p/repo/runtime:v1"


def test_ray_infra_from_terraform_outputs_missing_key_raises() -> None:
    with pytest.raises(ConfigError, match="network_id"):
        ray_submit.RayInfra.from_terraform_outputs({"compute_sa": "s", "code_bucket": "b"})


# --- extract_ray_telemetry: the header overlay ---------------------------------


class _FakeCluster:
    """Mirrors the vertex_ray Cluster shape (only the fields telemetry + submit read)."""

    dashboard_address = "1.2.3.4:8265"
    ray_version = "2.47"
    python_version = "3.11"


def test_extract_ray_telemetry_flattens_plan_and_cluster() -> None:
    cfg = _cfg()
    plan = ray_io.plan_cluster(cfg, run_id="rid")
    tel = ray_submit.extract_ray_telemetry(
        plan,
        cluster=_FakeCluster(),
        job_id="job-1",
        job_status="SUCCEEDED",
        total_wall_s=123.4,
        reuse=False,
    )
    assert tel["runtime"] == "ray"
    assert tel["cluster_name"] == plan.cluster_name
    assert tel["reuse"] is False
    assert tel["job_id"] == "job-1"
    assert tel["total_wall_s"] == 123.4
    assert tel["cpu_node_count"] == plan.cpu_node_count
    assert tel["gpu_node_count"] == plan.gpu_node_count
    assert tel["accelerator_type"] == "NVIDIA_TESLA_T4"
    assert tel["ray_version"] == "2.47"


def test_extract_ray_telemetry_is_json_serializable() -> None:
    import json

    plan = ray_io.plan_cluster(_cfg(), run_id="rid")
    tel = ray_submit.extract_ray_telemetry(
        plan,
        cluster=_FakeCluster(),
        job_id="j",
        job_status="SUCCEEDED",
        total_wall_s=1.0,
        reuse=False,
    )
    assert json.loads(json.dumps(tel, sort_keys=True))["runtime"] == "ray"


def test_extract_ray_telemetry_degrades_on_bare_cluster() -> None:
    # A cluster object missing ray/python/dashboard attrs yields None for them, never a raise.
    plan = ray_io.plan_cluster(_cfg(), run_id="rid")
    tel = ray_submit.extract_ray_telemetry(
        plan, cluster=object(), job_id="j", job_status="STOPPED", total_wall_s=None, reuse=True
    )
    assert tel["ray_version"] is None
    assert tel["dashboard_address"] is None
    assert tel["reuse"] is True


# --- submit_ray: the lifecycle (vertex_ray + JobSubmissionClient monkeypatched) --
#
# The helpers that touch the network (_create_cluster/_get_cluster/_delete_cluster/_submit_and_poll)
# are monkeypatched to record calls, so the lifecycle *ordering* is what's under test — the pure
# plan/entrypoint/telemetry are covered above.


@pytest.fixture
def _stubbed_lifecycle(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub the network seams + staging + telemetry; record the lifecycle calls in order."""
    calls: dict[str, Any] = {
        "created": 0,
        "deleted": 0,
        "submitted": 0,
        "telemetry": None,
        "order": [],
    }

    def _fake_stage(cfg: RunConfig, run_id: str, infra: ray_submit.RayInfra) -> str:
        return f"gs://code-bkt/runs/{run_id}.json"

    def _fake_create(plan: Any, infra: Any, name: str) -> str:
        calls["created"] += 1
        calls["order"].append("create")
        calls["create_name"] = name
        return f"projects/proj-x/locations/us-central1/persistentResources/{name}"

    def _fake_get(resource_name: str) -> Any:
        calls["order"].append("get")
        return _FakeCluster()

    def _fake_delete(resource_name: str) -> None:
        calls["deleted"] += 1
        calls["order"].append("delete")
        calls["delete_name"] = resource_name

    def _fake_submit(
        cluster: Any, entrypoint: str, runtime_env: dict, *, wait: bool
    ) -> tuple[str, str]:
        calls["submitted"] += 1
        calls["order"].append("submit")
        calls["entrypoint"] = entrypoint
        calls["runtime_env"] = runtime_env
        calls["wait"] = wait
        return "job-xyz", "SUCCEEDED"

    def _fake_stamp(telemetry: dict, run_id: str, settings: Settings) -> None:
        calls["telemetry"] = telemetry

    monkeypatch.setattr(ray_submit, "_stage_config", _fake_stage)
    monkeypatch.setattr(ray_submit, "_create_cluster", _fake_create)
    monkeypatch.setattr(ray_submit, "_get_cluster", _fake_get)
    monkeypatch.setattr(ray_submit, "_delete_cluster", _fake_delete)
    monkeypatch.setattr(ray_submit, "_submit_and_poll", _fake_submit)
    monkeypatch.setattr(ray_submit, "_stamp_ray_telemetry", _fake_stamp)
    return calls


def test_submit_ray_ephemeral_creates_submits_and_deletes(
    _stubbed_lifecycle: dict[str, Any],
) -> None:
    calls = _stubbed_lifecycle
    job_id = ray_submit.submit_ray(_cfg(), settings=_settings(), infra=_infra(), wait=True)
    assert job_id == "job-xyz"
    assert calls["created"] == 1
    assert calls["submitted"] == 1
    assert calls["deleted"] == 1  # ephemeral tears down
    assert calls["order"] == ["create", "get", "submit", "delete"]
    assert calls["telemetry"]["cluster_name"].startswith("sf-ray-")


def test_submit_ray_deletes_even_when_job_raises(
    _stubbed_lifecycle: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The teardown-in-finally guarantee: a non-SUCCEEDED terminal state raises, but the ephemeral
    # cluster is still deleted (no orphaned T4s billing forever).
    calls = _stubbed_lifecycle

    def _failing_submit(
        cluster: Any, entrypoint: str, runtime_env: dict, *, wait: bool
    ) -> tuple[str, str]:
        calls["order"].append("submit")
        return "job-fail", "FAILED"

    monkeypatch.setattr(ray_submit, "_submit_and_poll", _failing_submit)

    with pytest.raises(EngineError, match="FAILED"):
        ray_submit.submit_ray(_cfg(), settings=_settings(), infra=_infra(), wait=True)
    assert calls["created"] == 1
    assert calls["deleted"] == 1  # deleted despite the raise
    # telemetry is stamped before the raise, so a failed run still records its sizing.
    assert calls["telemetry"] is not None


def test_submit_ray_reuse_skips_create_and_delete(_stubbed_lifecycle: dict[str, Any]) -> None:
    calls = _stubbed_lifecycle
    ray_submit.submit_ray(
        _cfg(compute={"use_gpu": True, "ray_cluster_name": "standing-cluster"}),
        settings=_settings(),
        infra=_infra(),
        wait=True,
    )
    assert calls["created"] == 0  # reuse never creates
    assert calls["deleted"] == 0  # reuse never deletes
    assert calls["submitted"] == 1
    assert calls["order"] == ["get", "submit"]
    assert calls["telemetry"]["reuse"] is True


def test_submit_ray_cluster_name_override_forces_reuse(_stubbed_lifecycle: dict[str, Any]) -> None:
    # A CLI --cluster-name reuses even when the config didn't name one.
    calls = _stubbed_lifecycle
    ray_submit.submit_ray(
        _cfg(), settings=_settings(), infra=_infra(), cluster_name="adhoc-cluster", wait=True
    )
    assert calls["created"] == 0
    assert calls["deleted"] == 0
    assert calls["telemetry"]["reuse"] is True


def test_submit_ray_carries_arc_b_contract_to_entrypoint(
    _stubbed_lifecycle: dict[str, Any],
) -> None:
    calls = _stubbed_lifecycle
    ray_submit.submit_ray(
        _cfg(),
        models=[_GPU],
        manage_header=False,
        settings=_settings(),
        infra=_infra(),
        wait=True,
    )
    assert "--models neuralprophet" in calls["entrypoint"]
    assert "--manage-header false" in calls["entrypoint"]
    # the runtime_env ships current src + requirements
    assert calls["runtime_env"]["working_dir"].endswith("/src")


def test_submit_ray_n_series_override_resizes_and_changes_run_id(
    _stubbed_lifecycle: dict[str, Any],
) -> None:
    # n_series is the scale knob: it changes the staged config (hence run_id) AND the fixed plan.
    calls = _stubbed_lifecycle
    base_run_id = make_run_id(
        _cfg(data={"source_table": "source_series", "horizon": 7, "series_limit": 8})
    )
    ray_submit.submit_ray(_cfg(), settings=_settings(), infra=_infra(), n_series=1000, wait=True)
    # the ephemeral cluster name embeds the (new-scale) run_id, distinct from the base scale's.
    assert not calls["create_name"].endswith(base_run_id)


def test_submit_ray_no_wait_skips_poll_telemetry_and_teardown_check(
    _stubbed_lifecycle: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # wait=False returns the job id right after submit: no telemetry stamp, no terminal-state raise,
    # but the ephemeral cluster is still torn down in the finally (fire-and-forget would orphan it).
    calls = _stubbed_lifecycle

    def _immediate_submit(
        cluster: Any, entrypoint: str, runtime_env: dict, *, wait: bool
    ) -> tuple[str, str]:
        calls["order"].append("submit")
        calls["wait"] = wait
        return "job-nw", "PENDING"

    monkeypatch.setattr(ray_submit, "_submit_and_poll", _immediate_submit)
    job_id = ray_submit.submit_ray(_cfg(), settings=_settings(), infra=_infra(), wait=False)
    assert job_id == "job-nw"
    assert calls["wait"] is False
    assert calls["telemetry"] is None  # no telemetry without wait
    assert calls["deleted"] == 1  # still torn down
