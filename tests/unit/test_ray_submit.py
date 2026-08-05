"""Offline tests for the Vertex Ray submit helper (BUILD B4, ``scale_forecasting.ray_submit``).

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
    # working_dir is the package root (so `python -m scale_forecasting.ray_entry` resolves) — a real
    # local dir that exists for the upload to succeed.
    assert env["working_dir"].endswith("/src")
    assert os.path.isdir(env["working_dir"])
    # pip is the locked deps as a package LIST (pinned "name==version" specs), not a file path, so
    # we can drop cluster-provided packages from it.
    pip = env["pip"]
    assert isinstance(pip, list) and pip
    assert all("==" in spec for spec in pip)
    # neuralprophet (a real [models] dep) is shipped; Ray is NOT (the cluster image provides it, and
    # a pip pin could clash with the version Vertex booted).
    names = {re.split(r"[<>=!~;\[ ]", spec, maxsplit=1)[0].lower() for spec in pip}
    assert "neuralprophet" in names
    assert "ray" not in names


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
        calls.setdefault("delete_names", []).append(resource_name)

    def _fake_submit(
        cluster_resource_name: str, entrypoint: str, runtime_env: dict, *, wait: bool
    ) -> tuple[str, str]:
        calls["submitted"] += 1
        calls["order"].append("submit")
        calls["submit_resource_name"] = cluster_resource_name
        calls["entrypoint"] = entrypoint
        calls["runtime_env"] = runtime_env
        calls["wait"] = wait
        return "job-xyz", "SUCCEEDED"

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
    job_id = ray_submit.submit_ray(_cfg(), settings=_settings(), infra=_infra(), wait=True)
    assert job_id == "job-xyz"
    assert calls["created"] == 1
    assert calls["submitted"] == 1
    assert calls["deleted"] == 1  # ephemeral tears down
    assert calls["order"] == ["create", "get", "submit", "delete"]
    assert calls["telemetry"]["cluster_name"].startswith("sf-ray-")
    # Vertex SDK is pinned to the configured project (never the ambient GOOGLE_CLOUD_PROJECT).
    assert calls["init_project"] == "proj-x"


def test_submit_ray_deletes_even_when_job_raises(
    _stubbed_lifecycle: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The teardown-in-finally guarantee: a non-SUCCEEDED terminal state raises, but the ephemeral
    # cluster is still deleted (no orphaned T4s billing forever).
    calls = _stubbed_lifecycle

    def _failing_submit(
        cluster_resource_name: str, entrypoint: str, runtime_env: dict, *, wait: bool
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
        cluster_resource_name: str, entrypoint: str, runtime_env: dict, *, wait: bool
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
        "Quota exceeded for aiplatform.googleapis.com",
    ],
)
def test_is_capacity_error_false_for_non_capacity_messages(message: str) -> None:
    # A config/quota/permission error must NOT trigger a region hop (retrying elsewhere won't help).
    assert ray_submit._is_capacity_error(message) is False


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
    # the addressing form controls — see NOTES.md for the handshake status.)
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

    job_id, status = ray_submit._submit_and_poll(
        resource_name, "python -m x", {"working_dir": "/src"}, wait=True
    )
    assert (job_id, status) == ("job-1", "SUCCEEDED")
    assert len(connects) == 2  # connected once to submit/poll, reconnected once after the 401


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
