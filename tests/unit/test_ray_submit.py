"""Offline tests for the Vertex Ray submit helper (``scale_forecasting.ray_submit``).

No network: the pure job-spec assembly (:func:`build_entrypoint`, :func:`build_runtime_env`,
:func:`extract_ray_telemetry`), infra resolution (:class:`RayInfra`), and the whole cluster
lifecycle with ``vertex_ray`` + ``JobSubmissionClient`` monkeypatched — asserting the two load-
bearing lifecycle properties: **ephemeral creates then deletes (even when the job raises)**, and
**reuse skips both create and delete**. The live T4 path is the ``@gpu`` smoke in
``tests/integration/test_ray_gpu_smoke.py``.
"""

from __future__ import annotations

import re
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
        "data": {"source_table": "source_series_native", "horizon": 7, "series_limit": 8},
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


def test_build_entrypoint_defaults_omit_oncluster_flags() -> None:
    # Standalone submit (no subset, header-owning) omits both coordination flags, like build_batch.
    ep = ray_submit.build_entrypoint("gs://c/r.json", _settings())
    assert "--models" not in ep
    assert "--manage-header" not in ep


def test_build_entrypoint_appends_oncluster_flags_when_non_default() -> None:
    ep = ray_submit.build_entrypoint(
        "gs://c/r.json", _settings(), models=[_CPU, _GPU], manage_header=False
    )
    assert "--models theta,neuralprophet" in ep
    assert "--manage-header false" in ep


# --- build_runtime_env: runtime code delivery ----------------------------------


def test_build_runtime_env_prebuilt_image_ships_src_and_requirements() -> None:
    import os

    # No custom image (Vertex prebuilt Ray image): the image lacks our deps but has pip, so the
    # runtime_env carries the requirements list.
    env = ray_submit.build_runtime_env(None)
    # working_dir is the package root (so `python -m scale_forecasting.ray_entry` resolves) — a real
    # local dir that exists for the upload to succeed.
    assert env["working_dir"].endswith("/src")
    assert os.path.isdir(env["working_dir"])
    # pip is the locked deps as a package LIST (pinned "name==version" specs), not a file path, so
    # we can drop cluster-provided packages from it.
    pip = env["pip"]
    assert isinstance(pip, list) and pip
    # The list LEADS with the PyTorch CUDA extra index (mirrors docker/Dockerfile) so the x86_64
    # torch "+cu126" pin resolves on the cluster instead of 404-ing the whole job at env setup.
    assert pip[0] == "--extra-index-url https://download.pytorch.org/whl/cu126"
    specs = pip[1:]
    assert specs and all("==" in spec for spec in specs)
    # neuralprophet (a real [models] dep) is shipped; Ray is NOT (the cluster image provides it, and
    # a pip pin could clash with the version Vertex booted).
    names = {re.split(r"[<>=!~;\[ ]", spec, maxsplit=1)[0].lower() for spec in specs}
    assert "neuralprophet" in names
    assert "ray" not in names


def test_build_runtime_env_custom_image_omits_pip() -> None:
    import os

    # A custom node image already bundles the full dep set, so no pip key is emitted: Ray then skips
    # its runtime_env pip plugin (which, on the image's pip-less self-contained venv, would fail env
    # setup with "No module named pip"). Only the working_dir (code delivery) ships.
    env = ray_submit.build_runtime_env("us-docker.pkg.dev/p/repo/spark-runtime:latest")
    assert env["working_dir"].endswith("/src")
    assert os.path.isdir(env["working_dir"])
    assert "pip" not in env


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


def test_ray_infra_resolve_without_network_is_public_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # SF_RAY_NETWORK is optional: unset → network is None → public endpoint (no VPC peering needed).
    monkeypatch.delenv("SF_RAY_NETWORK", raising=False)
    monkeypatch.delenv("SF_RAY_NETWORK_ATTACHMENT", raising=False)
    monkeypatch.setenv("SF_COMPUTE_SA", "sa@x.iam")
    monkeypatch.setenv("SF_CODE_BUCKET", "code-bkt")
    infra = ray_submit.RayInfra.resolve()
    assert infra.network is None
    assert infra.network_attachment is None
    assert infra.compute_sa == "sa@x.iam"


def test_ray_infra_resolve_reads_network_attachment(monkeypatch: pytest.MonkeyPatch) -> None:
    # SF_RAY_NETWORK_ATTACHMENT (PSC-I) resolves onto network_attachment — the preferred path.
    monkeypatch.delenv("SF_RAY_NETWORK", raising=False)
    monkeypatch.setenv(
        "SF_RAY_NETWORK_ATTACHMENT",
        "projects/123/regions/us-central1/networkAttachments/scale-forecasting-ray",
    )
    monkeypatch.setenv("SF_COMPUTE_SA", "sa@x.iam")
    monkeypatch.setenv("SF_CODE_BUCKET", "code-bkt")
    infra = ray_submit.RayInfra.resolve()
    assert infra.network_attachment.endswith("/networkAttachments/scale-forecasting-ray")
    assert infra.network is None


def test_ray_infra_resolve_missing_var_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("SF_RAY_NETWORK", "SF_COMPUTE_SA", "SF_CODE_BUCKET"):
        monkeypatch.delenv(var, raising=False)
    # network is optional now, so the first *required* var that's missing is SF_COMPUTE_SA.
    with pytest.raises(ConfigError, match="SF_COMPUTE_SA"):
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
    assert no_image.network == "projects/p/global/networks/scale-forecasting"
    with_image = ray_submit.RayInfra.from_terraform_outputs(outputs, image_tag="v1")
    assert with_image.container_image == "us-docker.pkg.dev/p/repo/runtime:v1"


def test_ray_infra_from_terraform_outputs_without_network_is_public() -> None:
    # network_id is optional (a deployment without private-services access omits it) → public.
    infra = ray_submit.RayInfra.from_terraform_outputs({"compute_sa": "s", "code_bucket": "b"})
    assert infra.network is None
    assert infra.network_attachment is None


def test_ray_infra_from_terraform_outputs_reads_network_attachment() -> None:
    # network_attachment_id (PSC-I) is the supported private path — carried through from TF outputs.
    outputs = {
        "compute_sa": "sa@x.iam",
        "code_bucket": "code-bkt",
        "network_attachment_id": (
            "projects/123/regions/us-central1/networkAttachments/scale-forecasting-ray"
        ),
    }
    infra = ray_submit.RayInfra.from_terraform_outputs(outputs)
    assert infra.network_attachment.endswith("/networkAttachments/scale-forecasting-ray")


def test_ray_infra_from_terraform_outputs_missing_key_raises() -> None:
    with pytest.raises(ConfigError, match="code_bucket"):
        ray_submit.RayInfra.from_terraform_outputs({"compute_sa": "s"})


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
    # Elastic spec shows up for auditability on v_run_summary.
    assert tel["autoscale"] is True
    assert tel["cpu_min_nodes"] == plan.cpu_min_nodes
    assert tel["cpu_max_nodes"] == plan.cpu_max_nodes
    assert tel["gpu_min_nodes"] == plan.gpu_min_nodes
    assert tel["gpu_max_nodes"] == plan.gpu_max_nodes


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


# --- _worker_resources: the AutoscalingSpec wiring ------------------------------
#
# The pool builder imports vertex_ray lazily, so we inject fakes via sys.modules — this keeps the
# test offline and independent of whether the [ray] extra is installed. The fakes record the kwargs
# each Resources gets so we can assert whether an AutoscalingSpec was attached.


@pytest.fixture
def _fake_vertex_ray(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject fake ``vertex_ray`` + ``AutoscalingSpec`` modules that record Resources kwargs."""
    import sys
    import types

    class _FakeResources:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class _FakeAutoscalingSpec:
        def __init__(self, *, min_replica_count: int, max_replica_count: int) -> None:
            self.min_replica_count = min_replica_count
            self.max_replica_count = max_replica_count

    vr_mod = types.ModuleType("google.cloud.aiplatform.vertex_ray")
    vr_mod.Resources = _FakeResources  # type: ignore[attr-defined]
    res_mod = types.ModuleType("google.cloud.aiplatform.vertex_ray.util.resources")
    res_mod.AutoscalingSpec = _FakeAutoscalingSpec  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google.cloud.aiplatform.vertex_ray", vr_mod)
    monkeypatch.setitem(
        sys.modules, "google.cloud.aiplatform.vertex_ray.util.resources", res_mod
    )


def test_worker_resources_attaches_autoscaling_spec_per_pool(_fake_vertex_ray: None) -> None:
    plan = ray_io.plan_cluster(
        _cfg(compute={"use_gpu": True, "ray_cpu_max_nodes": 20, "ray_gpu_max_nodes": 4}),
        run_id="rid",
    )
    workers = ray_submit._worker_resources(plan, _infra())
    assert len(workers) == 2  # CPU + GPU pool
    specs = {}
    for w in workers:
        spec = w.kwargs["autoscaling_spec"]
        assert spec is not None
        # Distinguish CPU vs GPU pool by the accelerator kwarg.
        pool = "gpu" if w.kwargs.get("accelerator_count") else "cpu"
        specs[pool] = (spec.min_replica_count, spec.max_replica_count)
    assert specs["cpu"] == (plan.cpu_min_nodes, 20)
    assert specs["gpu"] == (plan.gpu_min_nodes, 4)


def test_worker_resources_omits_spec_when_autoscale_off(_fake_vertex_ray: None) -> None:
    plan = ray_io.plan_cluster(
        _cfg(compute={"use_gpu": True, "ray_autoscale": False}), run_id="rid"
    )
    workers = ray_submit._worker_resources(plan, _infra())
    assert workers  # pools still built
    for w in workers:
        assert w.kwargs["autoscaling_spec"] is None
        assert w.kwargs["node_count"] >= 1  # fixed path keeps the derived node count


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
        calls.setdefault("delete_names", []).append(resource_name)

    def _fake_submit(
        cluster_resource_name: str,
        entrypoint: str,
        runtime_env: dict,
        *,
        wait: bool,
        submission_id: str | None = None,
    ) -> tuple[str, str]:
        calls["submitted"] += 1
        calls["order"].append("submit")
        calls["submit_resource_name"] = cluster_resource_name
        calls["entrypoint"] = entrypoint
        calls["runtime_env"] = runtime_env
        calls["wait"] = wait
        calls["submission_id"] = submission_id
        return "job-xyz", "SUCCEEDED", ""

    def _fake_stamp(telemetry: dict, run_id: str, settings: Settings) -> None:
        calls["telemetry"] = telemetry

    def _fake_init(settings: Settings, region: str) -> None:
        calls["init_project"] = settings.project_id
        calls.setdefault("init_regions", []).append(region)

    def _fake_cluster_error(resource_name: str) -> str:
        # Default: no resource-side error text (tests that need one override this).
        return calls.get("cluster_error", "")

    monkeypatch.setattr(ray_submit, "_init_vertex", _fake_init)
    monkeypatch.setattr(ray_submit, "_cluster_error_message", _fake_cluster_error)
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
    job_id = ray_submit.submit_ray(
        _cfg(), settings=_settings(), infra=_infra(), wait=True, submission_id="sf-rid-ml-a1"
    )
    assert job_id == "job-xyz"
    assert calls["created"] == 1
    assert calls["submitted"] == 1
    assert calls["deleted"] == 1  # ephemeral tears down
    assert calls["order"] == ["create", "get", "submit", "delete"]
    assert calls["telemetry"]["cluster_name"].startswith("sf-ray-")
    # Vertex SDK is pinned to the configured project (never the ambient GOOGLE_CLOUD_PROJECT).
    assert calls["init_project"] == "proj-x"
    # The caller-supplied id is threaded to the Ray job so its own submission id is deterministic.
    assert calls["submission_id"] == "sf-rid-ml-a1"


def test_submit_ray_deletes_even_when_job_raises(
    _stubbed_lifecycle: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The teardown-in-finally guarantee: a non-SUCCEEDED terminal state raises, but the ephemeral
    # cluster is still deleted (no orphaned T4s billing forever).
    calls = _stubbed_lifecycle

    def _failing_submit(
        cluster_resource_name: str,
        entrypoint: str,
        runtime_env: dict,
        *,
        wait: bool,
        submission_id: str | None = None,
    ) -> tuple[str, str, str]:
        calls["order"].append("submit")
        detail = "message: boom\ndriver log tail:\nTraceback ... RuntimeError: boom"
        return "job-fail", "FAILED", detail

    monkeypatch.setattr(ray_submit, "_submit_and_poll", _failing_submit)

    with pytest.raises(EngineError, match="RuntimeError: boom") as excinfo:
        ray_submit.submit_ray(_cfg(), settings=_settings(), infra=_infra(), wait=True)
    # The driver diagnosis (message + log tail) is folded into the raised error, not just "FAILED",
    # so the *cause* survives even after the ml_job log stream ages out.
    assert "FAILED" in str(excinfo.value)
    assert "message: boom" in str(excinfo.value)
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


def test_submit_ray_carries_oncluster_contract_to_entrypoint(
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
        _cfg(data={"source_table": "source_series_native", "horizon": 7, "series_limit": 8})
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
        cluster_resource_name: str,
        entrypoint: str,
        runtime_env: dict,
        *,
        wait: bool,
        submission_id: str | None = None,
    ) -> tuple[str, str, str]:
        calls["order"].append("submit")
        calls["wait"] = wait
        return "job-nw", "PENDING", ""

    monkeypatch.setattr(ray_submit, "_submit_and_poll", _immediate_submit)
    job_id = ray_submit.submit_ray(_cfg(), settings=_settings(), infra=_infra(), wait=False)
    assert job_id == "job-nw"
    assert calls["wait"] is False
    assert calls["telemetry"] is None  # no telemetry without wait
    assert calls["deleted"] == 1  # still torn down


# --- region fallback: capacity classifier + resolution + the multi-region create loop ----------


@pytest.mark.parametrize(
    "message",
    [
        "Resources are insufficient in region: us-central1. Please try a different region.",
        "The zone does not have enough resources available",
        "RESOURCE EXHAUSTED",
    ],
)
def test_is_capacity_error_true_for_stockout_messages(message: str) -> None:
    assert ray_submit._is_capacity_error(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "machine type's memory is too small",
        "Permission denied on service account",
        "Quota exceeded for aiplatform.googleapis.com",  # a quota error, not a *capacity* one
    ],
)
def test_is_capacity_error_false_for_non_capacity_messages(message: str) -> None:
    # Capacity is a classifier distinct from quota; a bad machine type / permission is neither.
    assert ray_submit._is_capacity_error(message) is False


@pytest.mark.parametrize(
    "message",
    [
        "Quota exceeded for quota metric 'Nvidia T4 GPUs' of service 'aiplatform.googleapis.com'",
        "The request exceeds quota for the region",
        "resource creation would exceed quota limit for NVIDIA_T4_GPUS",
    ],
)
def test_is_quota_error_true_for_quota_messages(message: str) -> None:
    # Vertex accelerator quota is per-region, so a quota ceiling in one region is worth hopping.
    assert ray_submit._is_quota_error(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "Resources are insufficient in region: us-central1",  # capacity, not quota
        "machine type's memory is too small",
        "Permission denied on service account",
    ],
)
def test_is_quota_error_false_for_non_quota_messages(message: str) -> None:
    assert ray_submit._is_quota_error(message) is False


def test_is_generic_cluster_error_matches_sdk_opaque_error() -> None:
    # The SDK's post-provision opaque error → retryable when the reason can't be read.
    msg = "[Ray on Vertex AI]: Cluster projects/.../persistentResources/x returned an error."
    assert ray_submit._is_generic_cluster_error(msg) is True


@pytest.mark.parametrize(
    "message",
    ["Permission denied on service account", "some other RuntimeError", "returned successfully"],
)
def test_is_generic_cluster_error_false_otherwise(message: str) -> None:
    assert ray_submit._is_generic_cluster_error(message) is False


@pytest.mark.parametrize(
    ("resource_name", "expected"),
    [
        ("projects/307701787156/locations/us-central1/persistentResources/x", "us-central1"),
        ("projects/p/locations/us-east1/persistentResources/sf-ray", "us-east1"),
        ("not-a-resource-name", None),
    ],
)
def test_region_from_resource_name(resource_name: str, expected: str | None) -> None:
    # The regional endpoint for the error read is derived from the resource path, not assumed.
    assert ray_submit._region_from_resource_name(resource_name) == expected


# --- dashboard warm-up classifier: retry the connection race, not real faults ------------------


@pytest.mark.parametrize(
    "message",
    [
        "524 Server Error: status code 524 for url: https://x.../api/version",
        "504 Gateway Timeout for url: https://.../api/version",
        "503 Server Error: Service Temporarily Unavailable",
        "Connection refused",
        "HTTPConnectionPool: Max retries exceeded",
        "Read timed out",
    ],
)
def test_is_dashboard_warmup_error_true_for_transient(message: str) -> None:
    # The dashboard isn't serving through the proxy yet — retrying the connection is right.
    assert ray_submit._is_dashboard_warmup_error(Exception(message)) is True


@pytest.mark.parametrize(
    "message",
    [
        "401 Client Error: Unauthorized",
        "403 Client Error: Forbidden",
        "Ray cluster version 2.9 is incompatible with client 2.47",
    ],
)
def test_is_dashboard_warmup_error_false_for_real_faults(message: str) -> None:
    # Auth / version-mismatch won't fix themselves by waiting — must propagate, not spin.
    assert ray_submit._is_dashboard_warmup_error(Exception(message)) is False


def test_connect_job_client_uses_resource_name_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # We address the cluster by its resource name; the [ray] resolver discovers and authenticates
    # the dashboard itself. (Dashboard reachability is an execution-context property, not something
    # the addressing form controls.)
    resource_name = "projects/proj-x/locations/us-central1/persistentResources/sf-ray-abc"
    seen: list[str] = []

    class _FakeJobClient:
        def __init__(self, address: str) -> None:
            seen.append(address)

    monkeypatch.setattr("ray.job_submission.JobSubmissionClient", _FakeJobClient, raising=False)
    client = ray_submit._connect_job_client(resource_name)
    assert isinstance(client, _FakeJobClient)
    assert seen == [f"vertex_ray://{resource_name}"]


def test_connect_job_client_reraises_non_transient_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A real fault (auth) must propagate at once — not spin the warm-up retry loop.
    resource_name = "projects/proj-x/locations/us-central1/persistentResources/sf-ray-abc"
    seen: list[str] = []

    class _FakeJobClient:
        def __init__(self, address: str) -> None:
            seen.append(address)
            raise Exception("403 Client Error: Forbidden")

    monkeypatch.setattr("ray.job_submission.JobSubmissionClient", _FakeJobClient, raising=False)
    with pytest.raises(Exception, match="403"):
        ray_submit._connect_job_client(resource_name)
    assert seen == [f"vertex_ray://{resource_name}"]


# --- poll-loop auth expiry: refresh the client on a 401, don't abort a long run ----------------


@pytest.mark.parametrize(
    "message",
    [
        "401 Client Error: Unauthorized",
        "Request failed with status code 401: <html>...Unauthorized...</html>",
        "Unauthorized",
    ],
)
def test_is_auth_expiry_error_true_for_401(message: str) -> None:
    # A 401 during the poll loop = the client's OAuth token expired — refresh & retry, not fail.
    assert ray_submit._is_auth_expiry_error(Exception(message)) is True


@pytest.mark.parametrize(
    "message",
    [
        "403 Client Error: Forbidden",
        "500 Internal Server Error",
        "Ray job failed",
    ],
)
def test_is_auth_expiry_error_false_for_non_401(message: str) -> None:
    # Anything that isn't a 401 is not a recoverable token expiry — must propagate.
    assert ray_submit._is_auth_expiry_error(Exception(message)) is False


def test_submit_and_poll_refreshes_client_on_401(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 401 mid-poll rebuilds the Jobs client (fresh token) and polls to a terminal state.

    Simulates a run outliving the ~60-min OAuth TTL: the first client returns RUNNING then raises a
    401; the poll must reconnect (a *second* client) and finish on SUCCEEDED — not abort the run.
    """
    resource_name = "projects/proj-x/locations/us-central1/persistentResources/sf-ray-abc"
    connects: list[int] = []

    class _FakeClient:
        def __init__(self, idx: int) -> None:
            self.idx = idx
            self.calls = 0

        def submit_job(self, *, entrypoint: str, runtime_env: dict) -> str:
            return "job-1"

        def get_job_status(self, job_id: str) -> str:
            self.calls += 1
            # First client: RUNNING once, then the token expires → 401 on the next poll.
            if self.idx == 0:
                if self.calls == 1:
                    return "RUNNING"
                raise RuntimeError("Request failed with status code 401: Unauthorized")
            # Second client (post-refresh) reports the job finished.
            return "SUCCEEDED"

    def _fake_connect(_name: str) -> _FakeClient:
        idx = len(connects)
        connects.append(idx)
        return _FakeClient(idx)

    monkeypatch.setattr(ray_submit, "_connect_job_client", _fake_connect)
    monkeypatch.setattr(ray_submit.time, "sleep", lambda _s: None)

    job_id, status, detail = ray_submit._submit_and_poll(
        resource_name, "python -m x", {"working_dir": "/src"}, wait=True
    )
    assert (job_id, status) == ("job-1", "SUCCEEDED")
    assert detail == ""  # no failure detail on a SUCCEEDED run
    assert len(connects) == 2  # connected once to submit/poll, reconnected once after the 401


def test_submit_and_poll_captures_driver_detail_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A FAILED terminal state pulls the driver's error message + log tail at the moment of failure.

    The Jobs client holds both facts (``get_job_info().message`` + ``get_job_logs()``); capturing
    them here means the cause survives even after the ``ml_job`` log stream ages out of Cloud
    Logging's freshness window — no post-hoc archaeology.
    """
    resource_name = "projects/proj-x/locations/us-central1/persistentResources/sf-ray-abc"

    class _Info:
        message = "Job entrypoint command failed with exit code 1"

    class _FakeClient:
        def submit_job(self, *, entrypoint: str, runtime_env: dict) -> str:
            return "job-1"

        def get_job_status(self, job_id: str) -> str:
            return "FAILED"

        def get_job_info(self, job_id: str) -> _Info:
            return _Info()

        def get_job_logs(self, job_id: str) -> str:
            return "line1\nline2\nTraceback (most recent call last):\nValueError: bad series"

    monkeypatch.setattr(ray_submit, "_connect_job_client", lambda _n: _FakeClient())
    monkeypatch.setattr(ray_submit.time, "sleep", lambda _s: None)

    job_id, status, detail = ray_submit._submit_and_poll(
        resource_name, "python -m x", {"working_dir": "/src"}, wait=True
    )
    assert (job_id, status) == ("job-1", "FAILED")
    assert "Job entrypoint command failed" in detail
    assert "ValueError: bad series" in detail


def test_fetch_job_failure_detail_is_defensive(monkeypatch: pytest.MonkeyPatch) -> None:
    # Diagnosis is best-effort: if the info/logs calls themselves raise, the helper must swallow
    # and return what it could gather — never mask the job failure with a diagnosis crash.
    class _BrokenClient:
        def get_job_info(self, job_id: str) -> Any:
            raise RuntimeError("info endpoint down")

        def get_job_logs(self, job_id: str) -> str:
            raise RuntimeError("logs endpoint down")

    detail = ray_submit._fetch_job_failure_detail(_BrokenClient(), "job-1")
    assert detail == ""  # both sources failed → empty, no exception escapes


def test_submit_and_poll_reraises_non_401_poll_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # A non-401 poll error is a real fault — it must propagate, not trigger a client refresh.
    resource_name = "projects/proj-x/locations/us-central1/persistentResources/sf-ray-abc"
    connects: list[int] = []

    class _FakeClient:
        def submit_job(self, *, entrypoint: str, runtime_env: dict) -> str:
            return "job-1"

        def get_job_status(self, job_id: str) -> str:
            raise RuntimeError("500 Internal Server Error")

    def _fake_connect(_name: str) -> _FakeClient:
        connects.append(len(connects))
        return _FakeClient()

    monkeypatch.setattr(ray_submit, "_connect_job_client", _fake_connect)
    monkeypatch.setattr(ray_submit.time, "sleep", lambda _s: None)

    with pytest.raises(RuntimeError, match="500"):
        ray_submit._submit_and_poll(
            resource_name, "python -m x", {"working_dir": "/src"}, wait=True
        )
    assert len(connects) == 1  # no refresh on a non-401


def test_resolve_regions_defaults_to_settings_region() -> None:
    # No ray_regions configured → just the data-plane region.
    assert ray_submit._resolve_regions(_cfg(), _settings()) == ["us-central1"]


def test_resolve_regions_appends_home_region_last() -> None:
    # A configured list that omits home still ends up trying home as the final fallback.
    cfg = _cfg(compute={"use_gpu": True, "ray_regions": ["us-east1", "us-west1"]})
    assert ray_submit._resolve_regions(cfg, _settings()) == ["us-east1", "us-west1", "us-central1"]


def test_resolve_regions_keeps_order_when_home_already_listed() -> None:
    cfg = _cfg(compute={"use_gpu": True, "ray_regions": ["us-central1", "us-east1"]})
    assert ray_submit._resolve_regions(cfg, _settings()) == ["us-central1", "us-east1"]


def test_submit_ray_falls_back_to_next_region_on_capacity_stockout(
    _stubbed_lifecycle: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # region 1 stocks out (capacity error) → its partial resource is torn down and region 2 is
    # tried, which succeeds. The winning region's cluster is torn down after the job.
    calls = _stubbed_lifecycle

    def _stockout_then_ok(plan: Any, infra: Any, name: str) -> str:
        calls["created"] += 1
        calls["order"].append("create")
        if calls["created"] == 1:
            raise RuntimeError("Resources are insufficient in region: us-east1.")
        calls["create_name"] = name
        return f"projects/proj-x/locations/us-west1/persistentResources/{name}"

    monkeypatch.setattr(ray_submit, "_create_cluster", _stockout_then_ok)
    cfg = _cfg(compute={"use_gpu": True, "ray_regions": ["us-east1", "us-west1"]})
    job_id = ray_submit.submit_ray(cfg, settings=_settings(), infra=_infra(), wait=True)

    assert job_id == "job-xyz"
    assert calls["created"] == 2  # first region failed, second succeeded
    assert calls["init_regions"][:2] == ["us-east1", "us-west1"]  # tried in order
    # teardown count = 1 stocked-out region + 1 successful region after the job.
    assert calls["deleted"] == 2
    # the stocked-out region's resource path was among the deletes.
    assert any("us-east1" in name for name in calls["delete_names"])


def test_submit_ray_falls_back_when_capacity_reason_only_on_resource_error(
    _stubbed_lifecycle: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The real Vertex behavior: create raises a GENERIC "returned an error" while the capacity
    # reason lives only on the resource's error.message. The classifier must read the resource
    # error (not just the exception string) and still hop — this is the bug the live smoke caught.
    calls = _stubbed_lifecycle
    calls["cluster_error"] = (
        "Resources are insufficient in region: us-east1. Try a different region."
    )

    def _generic_then_ok(plan: Any, infra: Any, name: str) -> str:
        calls["created"] += 1
        calls["order"].append("create")
        if calls["created"] == 1:
            raise RuntimeError("[Ray on Vertex AI]: Cluster ... returned an error.")
        # region 2 succeeds → clear the resource-error so it isn't misread on a later call.
        calls["cluster_error"] = ""
        return f"projects/proj-x/locations/us-west1/persistentResources/{name}"

    monkeypatch.setattr(ray_submit, "_create_cluster", _generic_then_ok)
    cfg = _cfg(compute={"use_gpu": True, "ray_regions": ["us-east1", "us-west1"]})
    job_id = ray_submit.submit_ray(cfg, settings=_settings(), infra=_infra(), wait=True)

    assert job_id == "job-xyz"
    assert calls["created"] == 2  # hopped despite the generic exception text
    assert calls["init_regions"][:2] == ["us-east1", "us-west1"]


def test_submit_ray_fails_fast_on_non_capacity_error_without_trying_more_regions(
    _stubbed_lifecycle: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # A non-capacity error (e.g. bad machine type) must NOT hop regions — fail on the first attempt.
    calls = _stubbed_lifecycle

    def _bad_config(plan: Any, infra: Any, name: str) -> str:
        calls["created"] += 1
        raise RuntimeError("machine type's memory is too small")

    monkeypatch.setattr(ray_submit, "_create_cluster", _bad_config)
    cfg = _cfg(compute={"use_gpu": True, "ray_regions": ["us-east1", "us-west1"]})
    with pytest.raises(RuntimeError, match="memory is too small"):
        ray_submit.submit_ray(cfg, settings=_settings(), infra=_infra(), wait=True)
    assert calls["created"] == 1  # did not try the second region


def test_submit_ray_raises_when_all_regions_stock_out(
    _stubbed_lifecycle: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stubbed_lifecycle

    def _always_stockout(plan: Any, infra: Any, name: str) -> str:
        calls["created"] += 1
        raise RuntimeError("Resources are insufficient in region: x. try a different region")

    monkeypatch.setattr(ray_submit, "_create_cluster", _always_stockout)
    cfg = _cfg(compute={"use_gpu": True, "ray_regions": ["us-east1", "us-west1"]})
    with pytest.raises(EngineError, match="could not be created in any"):
        ray_submit.submit_ray(cfg, settings=_settings(), infra=_infra(), wait=True)
    assert calls["created"] == 3  # us-east1, us-west1, then home us-central1
