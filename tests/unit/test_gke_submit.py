"""Unit and contract tests for the Google Kubernetes Engine (`gke`) runtime (`gke_submit.py`)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from scale_forecasting.batch_infra import BatchInfra
from scale_forecasting.commands import build_gke_commands
from scale_forecasting.config import RunConfig
from scale_forecasting.dag import dag_nodes, plan_dag, preflight
from scale_forecasting.gke_submit import (
    RAY_GKE_BOOTSTRAP_CODE,
    GkeK8sClient,
    build_gke_cluster_spec,
    build_k8s_indexed_job_manifest,
    build_k8s_ray_manifests,
    build_kuberay_cluster_manifest,
    k8s_job_state,
    plan_gke_job,
    resolve_gke_candidates,
    submit_gke,
)
from scale_forecasting.job_launch import _entry_handle, _system_job_id
from scale_forecasting.probes.runtimes import GkeProbe, get_probe
from scale_forecasting.probes.vocabulary import (
    NATIVE_NOT_FOUND,
    NATIVE_RUNNING,
    NATIVE_SUCCEEDED,
    ProbeHandle,
)
from scale_forecasting.quota import gke_demands
from scale_forecasting.registry.ids import gke_job_id, make_job_key, make_run_id
from scale_forecasting.settings import Settings
from scale_forecasting.submitters import get_submitter
from scale_forecasting.vertex_submit import VERTEX_BOOTSTRAP_CODE


def _test_settings() -> Settings:
    return Settings(
        project_id="test-proj",
        connection="test-proj.us-central1.conn",
        warehouse_uri="gs://test-bkt/warehouse",
        dataset_id="forecasting",
        region="us-central1",
    )


def _test_infra(*, gke_cluster_name: str | None = "sf-standing-gke") -> BatchInfra:
    return BatchInfra(
        code_bucket="test-code-bkt",
        container_image="us-central1-docker.pkg.dev/test-proj/repo/sf:latest",
        compute_sa="sf-runner@test-proj.iam.gserviceaccount.com",
        subnetwork_uri="projects/test-proj/regions/us-central1/subnetworks/sf-sub",
        gke_cluster_name=gke_cluster_name,
    )


def test_gke_job_id_bounds_and_determinism() -> None:
    key = "gke-smoke-1234567890ab:deep_learning:1"
    jid = gke_job_id(key)
    assert jid[0].isalpha() and jid[-1].isalnum()
    assert len(jid) <= 52
    assert jid == gke_job_id(key)
    assert jid != gke_job_id("gke-smoke-1234567890ab:statistical:1")


def test_plan_gke_job_indexed_mode_cpu_and_gpu() -> None:
    settings = _test_settings()
    infra = _test_infra()
    cfg = RunConfig(
        run_name="test-gke-job",
        data={"source_table": "p.d.t"},
        python_runtime="gke",
        models=["theta", "autoets"],
        compute={
            "gke_mode": "job",
            "gke_cluster_name": "my-gke",
            "gke_namespace": "forecasting-ns",
            "machine_type": "n2-standard-8",
            "workers": 2,
        },
    )
    run_id = make_run_id(cfg)
    job_key = make_job_key(run_id, "statistical", 1)
    jname = gke_job_id(job_key)

    plan = plan_gke_job(
        cfg,
        ["theta", "autoets"],
        run_id=run_id,
        job_id=jname,
        gke_mode="job",
        cluster_name="my-gke",
        namespace="forecasting-ns",
        hardware="cpu",
        machine_type="n2-standard-8",
        worker_count=2,
        image_uri=infra.container_image,
        package_uri="gs://test-bkt/runs/pkg.zip",
        config_uri="gs://test-bkt/runs/cfg.json",
        service_account=infra.compute_sa,
        subnetwork_uri=infra.subnetwork_uri or "",
        infra=infra,
    )
    assert plan.gke_mode == "job"
    assert plan.cluster_name == "my-gke"
    assert plan.ephemeral_cluster is False
    assert plan.namespace == "forecasting-ns"
    assert plan.machine_type == "n2-standard-8"
    assert plan.worker_count == 2
    assert plan.hardware == "cpu"
    assert plan.accelerator_count == 0
    assert plan.resource_plan is not None
    assert plan.resource_plan.max_units == 2

    manifest = build_k8s_indexed_job_manifest(
        plan, driver_args=["--config-uri", plan.config_uri], settings=settings
    )
    assert manifest["apiVersion"] == "batch/v1"
    assert manifest["kind"] == "Job"
    assert manifest["metadata"]["name"] == jname
    assert manifest["metadata"]["namespace"] == "forecasting-ns"
    spec = manifest["spec"]
    assert spec["completionMode"] == "Indexed"
    assert spec["completions"] == 2
    assert spec["parallelism"] == 2
    assert spec["backoffLimit"] == 0
    container = spec["template"]["spec"]["containers"][0]
    assert container["command"][2] == VERTEX_BOOTSTRAP_CODE
    env_map = {e["name"]: e["value"] for e in container["env"]}
    assert env_map["JOB_COMPLETION_TOTAL"] == "2"
    assert env_map["SF_VERTEX_JOB_ID"] == jname
    assert env_map["SF_PROJECT_ID"] == "test-proj"
    assert "OMP_NUM_THREADS" in env_map
    # 60% of 8 cores = 4 CPU, 60% of 32 GiB = 19 GiB
    assert container["resources"]["requests"]["cpu"] == "4"
    assert container["resources"]["requests"]["memory"] == "19Gi"

    # GPU Indexed Job
    plan_gpu = plan_gke_job(
        cfg,
        ["tide"],
        run_id=run_id,
        job_id=jname,
        gke_mode="job",
        hardware="gpu",
        gpu_type="L4",
        accelerator_count=1,
        worker_count=1,
        image_uri=infra.container_image,
        package_uri="gs://test-bkt/runs/pkg.zip",
        config_uri="gs://test-bkt/runs/cfg.json",
        infra=infra,
    )
    assert plan_gpu.machine_type == "g2-standard-8"
    assert plan_gpu.accelerator_type == "nvidia-l4"
    assert plan_gpu.accelerator_count == 1
    gpu_manifest = build_k8s_indexed_job_manifest(
        plan_gpu, driver_args=["--config-uri", plan_gpu.config_uri], settings=settings
    )
    gpu_pod_spec = gpu_manifest["spec"]["template"]["spec"]
    gpu_container = gpu_pod_spec["containers"][0]
    assert gpu_container["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert any(t.get("key") == "nvidia.com/gpu" for t in gpu_pod_spec.get("tolerations", []))


def test_plan_gke_job_ray_mode_and_kuberay_manifests() -> None:
    settings = _test_settings()
    infra = _test_infra()
    cfg = RunConfig(
        run_name="test-gke-ray",
        data={"source_table": "p.d.t"},
        python_runtime="gke",
        models=["theta", "lightgbm"],
        compute={
            "gke_mode": "ray",
            "workers": 2,
            "machine_type": "n2-standard-8",
        },
    )
    run_id = make_run_id(cfg)
    jname = gke_job_id(make_job_key(run_id, "statistical", 1))
    plan = plan_gke_job(
        cfg,
        ["theta", "lightgbm"],
        run_id=run_id,
        job_id=jname,
        gke_mode="ray",
        worker_count=2,
        machine_type="n2-standard-8",
        image_uri=infra.container_image,
        package_uri="gs://test-bkt/runs/pkg.zip",
        config_uri="gs://test-bkt/runs/cfg.json",
        infra=infra,
    )
    assert plan.gke_mode == "ray"
    assert plan.ray_cluster_plan is not None
    manifests = build_k8s_ray_manifests(
        plan, driver_args=["--config-uri", plan.config_uri], settings=settings
    )
    assert manifests["head_service"] is not None
    assert manifests["head_service"]["kind"] == "Service"
    assert manifests["worker_deployment"] is not None
    assert manifests["worker_deployment"]["kind"] == "Deployment"
    assert manifests["worker_deployment"]["spec"]["replicas"] == 1
    head_job = manifests["job"]
    head_cmd = head_job["spec"]["template"]["spec"]["containers"][0]["command"]
    assert head_cmd[2] == RAY_GKE_BOOTSTRAP_CODE

    # Single-node Ray-on-GKE omits Service and Deployment
    plan_single = plan_gke_job(
        cfg,
        ["theta"],
        run_id=run_id,
        job_id=jname,
        gke_mode="ray",
        worker_count=1,
        machine_type="n2-standard-8",
        image_uri=infra.container_image,
        package_uri="gs://test-bkt/runs/pkg.zip",
        config_uri="gs://test-bkt/runs/cfg.json",
        infra=infra,
    )
    single_manifests = build_k8s_ray_manifests(
        plan_single, driver_args=["--config-uri", plan_single.config_uri], settings=settings
    )
    assert single_manifests["head_service"] is None
    assert single_manifests["worker_deployment"] is None

    # KubeRay RayCluster CRD manifest
    kuberay = build_kuberay_cluster_manifest(plan, settings=settings)
    assert kuberay["apiVersion"] == "ray.io/v1"
    assert kuberay["kind"] == "RayCluster"
    assert kuberay["spec"]["workerGroupSpecs"][0]["replicas"] == 1


def test_build_gke_cluster_spec_and_quota_demands() -> None:
    infra = _test_infra(gke_cluster_name=None)
    cfg = RunConfig(
        run_name="test-gke-cluster",
        data={"source_table": "p.d.t"},
        python_runtime="gke",
        models=["tide", "tsmixer"],
        model_params={
            "tide": {"training_mode": "global", "max_steps": 5},
            "tsmixer": {"training_mode": "global", "max_steps": 5},
        },
    )
    plan_gpu = plan_gke_job(
        cfg,
        ["tide", "tsmixer"],
        run_id="r-123",
        job_id="sf-r-123-dl-a1",
        gke_mode="job",
        hardware="gpu",
        gpu_type="L4",
        worker_count=2,
        image_uri=infra.container_image,
        package_uri="gs://test-bkt/runs/pkg.zip",
        config_uri="gs://test-bkt/runs/cfg.json",
        service_account=infra.compute_sa,
        subnetwork_uri=infra.subnetwork_uri or "",
        infra=infra,
    )
    assert plan_gpu.ephemeral_cluster is True
    cluster_spec = build_gke_cluster_spec(
        plan_gpu,
        location="us-central1-a",
        labels={"sf-run-id": "r-123"},
    )
    cluster = cluster_spec["cluster"]
    assert cluster["name"] == plan_gpu.cluster_name
    assert cluster["initialNodeCount"] == 2
    assert cluster["nodeConfig"]["accelerators"][0]["acceleratorType"] == "nvidia-l4"

    demands = gke_demands(plan_gpu, "us-central1")
    metrics = {d.metric.metric: d.amount(d.max_units) for d in demands}
    assert metrics["compute.googleapis.com/cpus"] == 16  # 2 * g2-standard-8
    assert metrics["compute.googleapis.com/nvidia_l4_gpus"] == 2


def test_k8s_job_state_parsing() -> None:
    assert k8s_job_state(None) == ("NOT_FOUND", "Kubernetes Job not found")
    assert k8s_job_state({"spec": {"completions": 2}, "status": {"succeeded": 2}})[0] == "SUCCEEDED"
    assert (
        k8s_job_state(
            {
                "status": {
                    "conditions": [{"type": "Complete", "status": "True"}],
                }
            }
        )[0]
        == "SUCCEEDED"
    )
    state_fail, detail = k8s_job_state(
        {
            "status": {
                "conditions": [
                    {
                        "type": "Failed",
                        "status": "True",
                        "reason": "BackoffLimitExceeded",
                        "message": "Pod failed",
                    }
                ]
            }
        }
    )
    assert state_fail == "FAILED"
    assert "BackoffLimitExceeded" in detail
    assert k8s_job_state({"spec": {"completions": 2}, "status": {"active": 1}})[0] == "RUNNING"


def test_gke_k8s_client_and_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _test_settings()

    class _FakeResp:
        def __init__(self, status_code: int, payload: Any = None, text: str = "") -> None:
            self.status_code = status_code
            self._payload = payload if payload is not None else {}
            self.text = text

        def json(self) -> Any:
            return self._payload

    class _FakeSession:
        def get(self, url: str, **kwargs: Any) -> _FakeResp:
            if url.endswith("/jobs/sf-job-ok"):
                return _FakeResp(
                    200,
                    {"spec": {"completions": 2}, "status": {"succeeded": 2}},
                )
            if url.endswith("/jobs/sf-job-running"):
                return _FakeResp(
                    200,
                    {"spec": {"completions": 2}, "status": {"active": 1, "succeeded": 1}},
                )
            if url.endswith("/jobs/sf-job-missing"):
                return _FakeResp(404, text="not found")
            if "/pods?" in url:
                return _FakeResp(200, {"items": [{"metadata": {"name": "pod-0"}}]})
            if url.endswith("/pods/pod-0/log?tailLines=80"):
                return _FakeResp(200, text="worker finished ok")
            return _FakeResp(200, {})

        def post(self, url: str, **kwargs: Any) -> _FakeResp:
            return _FakeResp(201, kwargs.get("json") or {})

        def delete(self, url: str, **kwargs: Any) -> _FakeResp:
            return _FakeResp(200, {})

    fake_session = _FakeSession()
    cluster_dict = {
        "name": "sf-standing-gke",
        "location": "us-central1-a",
        "status": "RUNNING",
        "endpoint": "10.0.0.1",
        "masterAuth": {"clusterCaCertificate": ""},
    }
    with GkeK8sClient(cluster_dict, session=fake_session) as client:
        assert client.create_job("default", {"metadata": {"name": "sf-job-ok"}}) == {
            "metadata": {"name": "sf-job-ok"}
        }
        assert client.get_job("default", "sf-job-ok") is not None
        assert client.get_job("default", "sf-job-missing") is None
        pods = client.list_job_pods("default", "sf-job-ok")
        assert len(pods) == 1
        assert client.get_pod_logs("default", "pod-0") == "worker finished ok"
        client.delete_job("default", "sf-job-ok")
        client.delete_deployment("default", "sf-job-ok-workers")
        client.delete_service("default", "sf-job-ok-head")

    # Test GkeProbe
    monkeypatch.setattr(
        "scale_forecasting.gke_submit.get_gke_cluster",
        lambda project, location, name, **kw: cluster_dict,
    )
    monkeypatch.setattr(
        "scale_forecasting.gke_submit._authorized_session",
        lambda: fake_session,
    )

    probe = GkeProbe()
    res_name = (
        "projects/test-proj/locations/us-central1-a/clusters/sf-standing-gke/"
        "namespaces/default/jobs/sf-job-ok"
    )
    assert probe._resolve_coordinates(
        ProbeHandle("gke", native_id="sf-job-ok", region="us-central1", resource_name=res_name),
        settings,
    ) == ("us-central1-a", "sf-standing-gke", "default", "sf-job-ok")

    h_ok = ProbeHandle("gke", native_id="sf-job-ok", region="us-central1", resource_name=res_name)
    assert probe.check(h_ok, settings=settings).native_state == NATIVE_SUCCEEDED

    h_missing = ProbeHandle(
        "gke",
        native_id="sf-job-missing",
        region="us-central1",
        resource_name=res_name.replace("sf-job-ok", "sf-job-missing"),
    )
    assert probe.check(h_missing, settings=settings).native_state == NATIVE_NOT_FOUND

    h_run = ProbeHandle(
        "gke",
        native_id="sf-job-running",
        region="us-central1",
        resource_name=res_name.replace("sf-job-ok", "sf-job-running"),
    )
    assert probe.check(h_run, settings=settings).native_state == NATIVE_RUNNING
    cancel_res = probe.cancel(h_run, settings=settings)
    assert cancel_res.stopped is True


def test_submit_gke_standing_and_ephemeral_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _test_settings()
    infra_standing = _test_infra(gke_cluster_name="sf-standing-gke")
    infra_ephemeral = _test_infra(gke_cluster_name=None)
    cfg = RunConfig(
        run_name="test-gke-submit",
        data={"source_table": "p.d.t"},
        python_runtime="gke",
        models=["theta"],
    )
    deleted_clusters: list[str] = []
    deleted_jobs: list[str] = []

    monkeypatch.setattr(
        "scale_forecasting.gke_submit.staging.stage_code",
        lambda bkt: ("gs://test-code-bkt/runs/pkg.zip", "abc"),
    )
    monkeypatch.setattr(
        "scale_forecasting.gke_submit.staging.stage_config",
        lambda c, rid, bkt: "gs://test-code-bkt/runs/cfg.json",
    )
    monkeypatch.setattr("scale_forecasting.gke_submit.profile_for_run", lambda *a, **k: None)
    monkeypatch.setattr("scale_forecasting.gke_submit.merge_header_telemetry", lambda *a, **k: None)
    monkeypatch.setattr("scale_forecasting.gke_submit.update_job", lambda *a, **k: None)
    monkeypatch.setattr("scale_forecasting.gke_submit._authorized_session", lambda: MagicMock())
    monkeypatch.setattr("scale_forecasting.gke_submit.quota.preflight_gke", lambda *a, **k: {})
    monkeypatch.setattr(
        "scale_forecasting.gke_submit.get_gke_cluster",
        lambda project, loc, name, **kw: {
            "name": name,
            "location": "us-central1-a",
            "status": "RUNNING",
            "endpoint": "10.0.0.1",
            "masterAuth": {"clusterCaCertificate": ""},
        },
    )
    monkeypatch.setattr(
        "scale_forecasting.gke_submit.create_gke_cluster",
        lambda project, loc, plan, **kw: {
            "name": plan.cluster_name,
            "location": loc,
            "status": "RUNNING",
            "endpoint": "10.0.0.1",
            "masterAuth": {"clusterCaCertificate": ""},
        },
    )
    monkeypatch.setattr(
        "scale_forecasting.gke_submit.delete_gke_cluster",
        lambda project, loc, name, **kw: deleted_clusters.append(name),
    )

    class _FakeK8s:
        def __init__(self, cluster: dict[str, Any], *, session: Any = None) -> None:
            self.cluster = cluster

        def __enter__(self) -> _FakeK8s:
            return self

        def __exit__(self, *exc: object) -> None:
            pass

        def create_job(self, namespace: str, manifest: dict[str, Any]) -> dict[str, Any]:
            return manifest

        def create_service(self, namespace: str, manifest: dict[str, Any]) -> dict[str, Any]:
            return manifest

        def create_deployment(self, namespace: str, manifest: dict[str, Any]) -> dict[str, Any]:
            return manifest

        def get_job(self, namespace: str, job_name: str) -> dict[str, Any]:
            return {"spec": {"completions": 1}, "status": {"succeeded": 1}}

        def list_job_pods(self, namespace: str, job_name: str) -> list[dict[str, Any]]:
            return []

        def delete_job(self, namespace: str, job_name: str) -> None:
            deleted_jobs.append(job_name)

        def delete_deployment(self, namespace: str, name: str) -> None:
            pass

        def delete_service(self, namespace: str, name: str) -> None:
            pass

    monkeypatch.setattr("scale_forecasting.gke_submit.GkeK8sClient", _FakeK8s)

    # 1. Standing cluster: does NOT delete cluster on exit
    rid1, jname1, handle1 = submit_gke(
        cfg,
        settings=settings,
        infra=infra_standing,
        models=["theta"],
        job_id="sf-test-standing",
    )
    assert rid1 == make_run_id(cfg)
    assert jname1 == "sf-test-standing"
    assert "clusters/sf-standing-gke/namespaces/default/jobs/sf-test-standing" in (
        handle1.resource_name or ""
    )
    assert deleted_clusters == []
    assert deleted_jobs == ["sf-test-standing"]

    # 2. Ephemeral cluster: deletes cluster in finally
    _rid2, jname2, handle2 = submit_gke(
        cfg,
        settings=settings,
        infra=infra_ephemeral,
        models=["theta"],
        job_id="sf-test-ephemeral",
    )
    assert jname2 == "sf-test-ephemeral"
    assert "jobs/sf-test-ephemeral" in (handle2.resource_name or "")
    assert len(deleted_clusters) == 1


def test_dag_submitters_and_commands_for_gke_and_ray_mode_gke() -> None:
    settings = _test_settings()
    infra = _test_infra()
    cfg = RunConfig(
        run_name="test-gke-modes",
        data={"source_table": "p.d.t"},
        python_runtime="gke",
        models=["theta", "lightgbm", "tide"],
        model_params={"tide": {"training_mode": "global", "max_steps": 5}},
        compute={
            "gke_mode": "job",
            "families": {
                "ml": {"runtime": "gke", "gke_mode": "ray", "workers": 2},
                "deep_learning": {
                    "runtime": "ray",
                    "ray_mode": "gke",
                    "hardware": "gpu",
                    "gpu_type": "L4",
                },
            },
        },
    )
    preflight(cfg)
    nodes = {n.family: n for n in dag_nodes(plan_dag(cfg))}
    assert nodes["statistical"].runtime == "gke"
    assert nodes["ml"].runtime == "gke"
    assert nodes["deep_learning"].runtime == "ray"

    fc_stat = cfg.resolve_family_compute("statistical")
    fc_ml = cfg.resolve_family_compute("ml")
    fc_dl = cfg.resolve_family_compute("deep_learning")
    assert fc_stat.gke_mode == "job"
    assert fc_ml.gke_mode == "ray"
    assert fc_dl.ray_mode == "gke"

    cfg_ray_vertex = RunConfig(
        run_name="test-ray-mode-vertex",
        data={"source_table": "p.d.t"},
        python_runtime="ray",
        models=["tide"],
        compute={"families": {"deep_learning": {"runtime": "ray", "ray_mode": "vertex"}}},
    )
    assert cfg_ray_vertex.resolve_family_compute("deep_learning").ray_mode == "vertex"

    run_id = make_run_id(cfg)
    dl_key = make_job_key(run_id, "deep_learning", 1)
    dl_sys_id = _system_job_id(dl_key, "gke")
    assert dl_sys_id == gke_job_id(dl_key)
    dl_handle = _entry_handle(cfg, run_id, fc_dl, _system_job_id(dl_key, "ray"), settings)
    assert dl_handle.runtime == "gke"
    assert dl_handle.native_id == gke_job_id(_system_job_id(dl_key, "ray"))

    assert get_submitter("gke").name == "gke"
    assert get_probe("gke").name == "gke"

    lc = build_gke_commands(
        config_uri="gs://test-bkt/runs/cfg.json",
        package_uri="gs://test-bkt/runs/pkg.zip",
        settings=settings,
        job_id=dl_sys_id,
        models=["tide"],
        gke_mode="ray",
        hardware="gpu",
        gpu_type="L4",
        machine_type="g2-standard-8",
        worker_count=1,
    )
    assert lc.runtime == "gke"
    assert "scale_forecasting.gke_submit" in lc.universal
    assert "--gke-mode ray" in lc.universal

    # Fallback candidate expansion
    cfg_fb = cfg.model_copy(
        update={
            "compute": cfg.compute.model_copy(
                update={
                    "ray_regions": ["us-central1", "us-east1"],
                }
            )
        }
    )
    cands = resolve_gke_candidates(cfg_fb, settings=settings, infra=infra)
    assert len(cands) >= 2
    assert cands[0].region == "us-central1"
    assert any(c.region == "us-east1" for c in cands)
