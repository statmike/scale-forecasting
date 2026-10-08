"""Unit and contract tests for the Vertex AI ``CustomJob`` runtime (`runtime="vertex"`)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scale_forecasting.commands import build_vertex_commands
from scale_forecasting.config import RunConfig, resolve_vertex_machine_type
from scale_forecasting.dag import dag_nodes, plan_dag, preflight
from scale_forecasting.engines.vertex_engine import (
    WorkerTopology,
    effective_worker_count,
    execute_panel,
    partition_models_for_worker,
    resolve_worker_topology,
    shard_series_for_worker,
)
from scale_forecasting.errors import ConfigError
from scale_forecasting.job_launch import _entry_handle, _system_job_id
from scale_forecasting.probes.runtimes import VertexProbe, get_probe
from scale_forecasting.probes.vocabulary import (
    NATIVE_FAILED,
    NATIVE_NOT_FOUND,
    NATIVE_RUNNING,
    NATIVE_SUCCEEDED,
    ProbeHandle,
)
from scale_forecasting.quota import vertex_demands
from scale_forecasting.registry.ids import make_job_key, make_run_id, vertex_job_id
from scale_forecasting.settings import Settings
from scale_forecasting.submitters import get_submitter
from scale_forecasting.vertex_submit import (
    VERTEX_BOOTSTRAP_CODE,
    build_custom_job_spec_dict,
    plan_vertex_job,
)


def _synthetic_panel(n_series: int = 4, n_steps: int = 24) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2024-01-01", periods=n_steps, freq="MS")
    for idx in range(n_series):
        ts_id = f"S{idx:02d}"
        for step, dt in enumerate(dates):
            rows.append(
                {
                    "ts_id": ts_id,
                    "ds": dt,
                    "y": float(100.0 + idx * 10.0 + step * 1.5),
                    "region": "North" if idx < n_series // 2 else "South",
                }
            )
    return pd.DataFrame(rows)


def test_resolve_worker_topology_single_default() -> None:
    topo = resolve_worker_topology(environ={})
    assert topo == WorkerTopology(rank=0, world_size=1, source="single")
    assert not topo.is_distributed


def test_resolve_worker_topology_cli_and_env() -> None:
    topo_cli = resolve_worker_topology(worker_rank=1, worker_count=4, environ={})
    assert topo_cli == WorkerTopology(rank=1, world_size=4, source="cli")
    assert topo_cli.is_distributed

    topo_env = resolve_worker_topology(environ={"SF_WORKER_RANK": "2", "SF_WORKER_COUNT": "4"})
    assert topo_env == WorkerTopology(rank=2, world_size=4, source="env")

    topo_k8s = resolve_worker_topology(
        environ={"JOB_COMPLETION_INDEX": "1", "JOB_COMPLETION_TOTAL": "3"}
    )
    assert topo_k8s == WorkerTopology(rank=1, world_size=3, source="k8s_indexed_job")

    with pytest.raises(ConfigError, match="out of bounds"):
        resolve_worker_topology(worker_rank=3, worker_count=3, environ={})


def test_resolve_worker_topology_vertex_cluster_spec() -> None:
    cluster_spec = {
        "cluster": {
            "workerpool0": ["10.0.0.1:2222"],
            "workerpool1": ["10.0.0.2:2222", "10.0.0.3:2222"],
        },
        "task": {"type": "workerpool1", "index": 1},
    }
    topo = resolve_worker_topology(environ={"CLUSTER_SPEC": json.dumps(cluster_spec)})
    assert topo == WorkerTopology(rank=2, world_size=3, source="vertex_cluster_spec")

    cluster_spec_primary = {
        **cluster_spec,
        "task": {"type": "workerpool0", "index": 0},
    }
    topo0 = resolve_worker_topology(environ={"CLUSTER_SPEC": json.dumps(cluster_spec_primary)})
    assert topo0 == WorkerTopology(rank=0, world_size=3, source="vertex_cluster_spec")


def test_effective_worker_count_and_sharding_rules() -> None:
    cfg_local = RunConfig(
        run_name="test-vertex",
        data={"source_table": "p.d.t"},
        python_runtime="vertex",
        models=["naive_mean", "theta"],
    )
    assert effective_worker_count(cfg_local, ["naive_mean", "theta"], 4) == 4
    topo0 = WorkerTopology(rank=0, world_size=2, source="cli")
    topo1 = WorkerTopology(rank=1, world_size=2, source="cli")
    assert partition_models_for_worker(cfg_local, ["naive_mean", "theta"], topo0) == [
        "naive_mean",
        "theta",
    ]
    assert partition_models_for_worker(cfg_local, ["naive_mean", "theta"], topo1) == [
        "naive_mean",
        "theta",
    ]
    series = ["S00", "S01", "S02", "S03", "S04"]
    s0 = shard_series_for_worker(series, topo0)
    s1 = shard_series_for_worker(series, topo1)
    assert set(s0).isdisjoint(set(s1))
    assert sorted(s0 + s1) == series

    # Global / deep-learning models: automatically allocate 1 dedicated VM per model even when
    # requested_workers=1, and cap at len(models) when requested_workers > len(models).
    cfg_global = RunConfig(
        run_name="test-vertex",
        data={"source_table": "p.d.t"},
        python_runtime="vertex",
        models=["tide", "tsmixer"],
        model_params={
            "tide": {"training_mode": "global"},
            "tsmixer": {"training_mode": "global"},
        },
    )
    assert effective_worker_count(cfg_global, ["tide", "tsmixer"], 1) == 2
    assert effective_worker_count(cfg_global, ["tide", "tsmixer"], 4) == 2
    assert partition_models_for_worker(
        cfg_global, ["tide", "tsmixer"], WorkerTopology(rank=0, world_size=4)
    ) == ["tide"]
    assert partition_models_for_worker(
        cfg_global, ["tide", "tsmixer"], WorkerTopology(rank=1, world_size=4)
    ) == ["tsmixer"]
    assert (
        partition_models_for_worker(
            cfg_global, ["tide", "tsmixer"], WorkerTopology(rank=2, world_size=4)
        )
        == []
    )

    # Mixed hybrid + global deep_learning models (e.g. neuralprophet + tide): each gets its own VM
    # and runs on the full series panel.
    from scale_forecasting.engines.vertex_engine import models_sharded_across_workers

    cfg_dl_mixed = RunConfig(
        run_name="test-vertex-dl-mixed",
        data={"source_table": "p.d.t"},
        python_runtime="vertex",
        models=["neuralprophet", "tide"],
    )
    assert effective_worker_count(cfg_dl_mixed, ["neuralprophet", "tide"], 1) == 2
    assert models_sharded_across_workers(cfg_dl_mixed, ["neuralprophet", "tide"]) is True

    # Hierarchical reconciliation: 1 model -> 1 worker; 2 models -> up to 2 workers (model-sharded).
    cfg_hier = RunConfig(
        run_name="test-vertex",
        data={"source_table": "p.d.t"},
        python_runtime="vertex",
        models=["naive_mean"],
        hierarchy={
            "enabled": True,
            "levels": [["region"]],
            "reconciliation_methods": ["bottom_up"],
        },
    )
    assert effective_worker_count(cfg_hier, ["naive_mean"], 4) == 1
    assert partition_models_for_worker(
        cfg_hier, ["naive_mean"], WorkerTopology(rank=0, world_size=4)
    ) == ["naive_mean"]
    assert (
        partition_models_for_worker(cfg_hier, ["naive_mean"], WorkerTopology(rank=1, world_size=4))
        == []
    )

    cfg_hier_multi = cfg_hier.model_copy(update={"models": ["naive_mean", "theta"]})
    assert effective_worker_count(cfg_hier_multi, ["naive_mean", "theta"], 4) == 2
    assert partition_models_for_worker(
        cfg_hier_multi, ["naive_mean", "theta"], WorkerTopology(rank=0, world_size=2)
    ) == ["naive_mean"]
    assert partition_models_for_worker(
        cfg_hier_multi, ["naive_mean", "theta"], WorkerTopology(rank=1, world_size=2)
    ) == ["theta"]


def test_plan_vertex_job_and_custom_job_spec() -> None:
    cfg = RunConfig(
        run_name="test-vertex",
        data={"source_table": "p.d.t"},
        python_runtime="vertex",
        models=["naive_mean", "theta"],
        compute={"vertex_machine_type": "n2-standard-8", "vertex_workers": 3},
    )
    plan = plan_vertex_job(
        cfg,
        ["naive_mean", "theta"],
        run_id="r-20261002-abcd1234",
        image_uri="us-docker.pkg.dev/p/r/img:latest",
        package_uri="gs://bkt/runs/pkg.zip",
        config_uri="gs://bkt/runs/cfg.json",
        service_account="compute@p.iam.gserviceaccount.com",
    )
    assert plan.worker_count == 3
    assert plan.machine_type == "n2-standard-8"
    assert plan.accelerator_count == 0
    assert plan.resource_plan is not None
    assert plan.resource_plan.slots_per_unit >= 1
    assert plan.resource_plan.max_units == 3

    spec = build_custom_job_spec_dict(plan, driver_args=["--config-uri", plan.config_uri])
    pools = spec["job_spec"]["worker_pool_specs"]
    assert len(pools) == 2
    assert pools[0]["replica_count"] == 1
    assert pools[1]["replica_count"] == 2
    assert pools[0]["container_spec"]["command"][2] == VERTEX_BOOTSTRAP_CODE
    env_names = {e["name"] for e in pools[0]["container_spec"].get("env", [])}
    assert {"OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"} <= env_names

    # GPU machine auto-resolution (T4 -> n1-standard-8, L4 -> g2-standard-8)
    assert resolve_vertex_machine_type("gpu", "T4", "n2-standard-8", "auto") == "n1-standard-8"
    assert resolve_vertex_machine_type("gpu", "L4", "n2-standard-8", "auto") == "g2-standard-8"
    plan_gpu = plan_vertex_job(
        cfg,
        ["naive_mean"],
        run_id="r-20261002-abcd1234",
        hardware="gpu",
        gpu_type="L4",
        worker_count=1,
        image_uri="us-docker.pkg.dev/p/r/img:latest",
        package_uri="gs://bkt/runs/pkg.zip",
        config_uri="gs://bkt/runs/cfg.json",
    )
    assert plan_gpu.machine_type == "g2-standard-8"
    assert plan_gpu.accelerator_type == "NVIDIA_L4"
    assert plan_gpu.accelerator_count == 1
    demands = vertex_demands(plan_gpu, "us-central1")
    metrics = {d.metric.metric: d.amount(d.max_units) for d in demands}
    assert metrics["aiplatform.googleapis.com/custom_model_training_g2_cpus"] == 8
    assert metrics["aiplatform.googleapis.com/custom_model_training_nvidia_l4_gpus"] == 1


def test_dag_and_commands_support_vertex_runtime() -> None:
    cfg = RunConfig(
        run_name="test-vertex",
        data={"source_table": "p.d.t"},
        python_runtime="vertex",
        models=["theta", "tide"],
        model_params={"tide": {"training_mode": "global", "max_steps": 5}},
        compute={
            "families": {
                "deep_learning": {
                    "runtime": "vertex",
                    "hardware": "gpu",
                    "gpu_type": "L4",
                    "vertex_workers": 1,
                }
            }
        },
    )
    preflight(cfg)
    nodes = dag_nodes(plan_dag(cfg))
    by_family = {n.family: n for n in nodes}
    assert by_family["statistical"].runtime == "vertex"
    assert by_family["deep_learning"].runtime == "vertex"
    assert by_family["deep_learning"].hardware == "gpu"

    run_id = make_run_id(cfg)
    job_key = make_job_key(run_id, "deep_learning", 1)
    sys_id = _system_job_id(job_key, "vertex")
    assert sys_id == vertex_job_id(job_key)
    settings = Settings(
        project_id="p",
        connection="p.us-central1.c",
        warehouse_uri="gs://w",
        dataset_id="ds",
        region="us-central1",
    )
    handle = _entry_handle(
        cfg, run_id, cfg.resolve_family_compute("deep_learning"), sys_id, settings
    )
    assert handle.runtime == "vertex"
    assert handle.native_id == sys_id
    assert get_submitter("vertex").name == "vertex"
    assert get_probe("vertex").name == "vertex"

    lc = build_vertex_commands(
        config_uri="gs://b/runs/r.json",
        package_uri="gs://b/runs/pkg.zip",
        settings=settings,
        display_name=sys_id,
        models=["tide"],
        job_id=sys_id,
        hardware="gpu",
        gpu_type="L4",
        machine_type="g2-standard-8",
        worker_count=1,
    )
    assert lc.runtime == "vertex"
    assert "scale_forecasting.vertex_submit" in lc.universal
    assert "--hardware gpu" in lc.universal


def test_vertex_engine_executes_single_and_sharded_workers_and_hierarchy() -> None:
    panel = _synthetic_panel(n_series=4, n_steps=20)
    cfg = RunConfig(
        run_name="test-vertex",
        data={"source_table": "p.d.t", "freq": "MS", "horizon": 3},
        python_runtime="vertex",
        backtest={"enabled": True, "n_folds": 1, "step": 1, "horizon": 3, "min_train": 12},
        models=["naive_mean"],
    )
    # 2-worker sharded execution: rank 0 and rank 1 process disjoint halves of the panel.
    res0, st0 = execute_panel(
        panel,
        cfg,
        models=["naive_mean"],
        topology=WorkerTopology(rank=0, world_size=2, source="cli"),
    )
    res1, st1 = execute_panel(
        panel,
        cfg,
        models=["naive_mean"],
        topology=WorkerTopology(rank=1, world_size=2, source="cli"),
    )
    assert set(st0["ts_id"]).isdisjoint(set(st1["ts_id"]))
    assert sorted(list(st0["ts_id"]) + list(st1["ts_id"])) == ["S00", "S01", "S02", "S03"]
    assert sum(len(r.predictions) for r in res0) + sum(len(r.predictions) for r in res1) == 4 * 3
    assert len(res0) + len(res1) == 4

    # Hierarchical reconciliation on Vertex CustomJob (rank 0 reconciles tree; rank 1 stands down).
    cfg_hier = RunConfig(
        run_name="test-vertex",
        data={"source_table": "p.d.t", "freq": "MS", "horizon": 2},
        python_runtime="vertex",
        backtest={"enabled": True, "n_folds": 1, "step": 1, "horizon": 2, "min_train": 12},
        models=["naive_mean"],
        hierarchy={
            "enabled": True,
            "levels": [["region"]],
            "reconciliation_methods": ["bottom_up"],
        },
    )
    res_h0, _st_h0 = execute_panel(
        panel,
        cfg_hier,
        models=["naive_mean"],
        topology=WorkerTopology(rank=0, world_size=2, source="cli"),
    )
    res_h1, st_h1 = execute_panel(
        panel,
        cfg_hier,
        models=["naive_mean"],
        topology=WorkerTopology(rank=1, world_size=2, source="cli"),
    )
    assert len(res_h1) == 0 and st_h1.empty
    reconciled = [r for r in res_h0 if r.model_type == "naive_mean_bottom_up"]
    assert len(reconciled) == 7  # 1 total + 2 regions + 4 leaves
    by_id = {r.ts_id: r.predictions["yhat"].to_numpy() for r in reconciled}
    np.testing.assert_allclose(
        by_id["__total__"],
        by_id["region=North"] + by_id["region=South"],
    )


def test_vertex_probe_check_and_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    from google.api_core.exceptions import NotFound

    settings = Settings(
        project_id="p",
        connection="p.us-central1.c",
        warehouse_uri="gs://w",
        dataset_id="ds",
        region="us-central1",
    )
    fake_job = SimpleNamespace(
        name="projects/p/locations/us-central1/customJobs/999",
        state=SimpleNamespace(name="JOB_STATE_RUNNING"),
        error=None,
    )

    class _FakeClient:
        def __init__(self) -> None:
            self.cancelled: list[str] = []

        def get_custom_job(self, *, name: str, timeout: float) -> object:
            if name.endswith("/404"):
                raise NotFound("missing")
            return fake_job

        def list_custom_jobs(self, *, request: dict[str, str], timeout: float) -> list[object]:
            if "missing" in request.get("filter", ""):
                return []
            return [fake_job]

        def cancel_custom_job(self, *, name: str, timeout: float) -> None:
            self.cancelled.append(name)

    fake_client = _FakeClient()
    monkeypatch.setattr("scale_forecasting.vertex_submit._job_client", lambda region: fake_client)

    probe = VertexProbe()
    # 1. Check by display_name (pre-stamp-back)
    res_disp = probe.check(
        ProbeHandle("vertex", native_id="sf-r-20261002-abcd-stat-a1", region="us-central1"),
        settings=settings,
    )
    assert res_disp.native_state == NATIVE_RUNNING
    assert res_disp.exists is True

    # 2. Check by resource name (post-stamp-back)
    res_name = probe.check(
        ProbeHandle(
            "vertex",
            native_id="projects/p/locations/us-central1/customJobs/999",
            region="us-central1",
        ),
        settings=settings,
    )
    assert res_name.native_state == NATIVE_RUNNING

    # 3. NotFound by resource name
    res_404 = probe.check(
        ProbeHandle(
            "vertex",
            native_id="projects/p/locations/us-central1/customJobs/404",
            region="us-central1",
        ),
        settings=settings,
    )
    assert res_404.native_state == NATIVE_NOT_FOUND
    assert res_404.exists is False

    # 4. Cancel running job
    cancel_res = probe.cancel(
        ProbeHandle("vertex", native_id="sf-r-20261002-abcd-stat-a1", region="us-central1"),
        settings=settings,
    )
    assert cancel_res.stopped is True
    assert fake_client.cancelled == ["projects/p/locations/us-central1/customJobs/999"]

    # 5. Cancel already-succeeded job
    fake_job.state = SimpleNamespace(name="JOB_STATE_SUCCEEDED")
    assert (
        probe.check(
            ProbeHandle(
                "vertex",
                native_id="projects/p/locations/us-central1/customJobs/999",
                region="us-central1",
            ),
            settings=settings,
        ).native_state
        == NATIVE_SUCCEEDED
    )
    cancel_done = probe.cancel(
        ProbeHandle(
            "vertex",
            native_id="projects/p/locations/us-central1/customJobs/999",
            region="us-central1",
        ),
        settings=settings,
    )
    assert cancel_done.stopped is False
    assert cancel_done.already_gone is True

    # 6. Failed job state mapping
    fake_job.state = SimpleNamespace(name="JOB_STATE_FAILED")
    fake_job.error = SimpleNamespace(message="OOM")
    res_fail = probe.check(
        ProbeHandle(
            "vertex",
            native_id="projects/p/locations/us-central1/customJobs/999",
            region="us-central1",
        ),
        settings=settings,
    )
    assert res_fail.native_state == NATIVE_FAILED
    assert res_fail.detail == "OOM"


def test_vertex_bootstrap_and_entry_switch_out_of_readonly_workdir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Container WORKDIR (/opt/scale-forecasting) is root-owned while running as USER spark."""
    import os

    from scale_forecasting import vertex_entry
    from scale_forecasting.vertex_submit import VERTEX_BOOTSTRAP_CODE

    assert "os.access(os.getcwd(), os.W_OK)" in VERTEX_BOOTSTRAP_CODE
    assert "os.chdir(td)" in VERTEX_BOOTSTRAP_CODE

    recorded_cwd: list[str] = []
    monkeypatch.setattr(os, "access", lambda path, mode: False)
    monkeypatch.setattr(os, "chdir", lambda path: recorded_cwd.append(str(path)))
    monkeypatch.setattr(
        vertex_entry,
        "run_entry",
        lambda argv, **kw: None,
    )
    vertex_entry.main(["--config-uri", "gs://b/c.json"])
    assert len(recorded_cwd) == 1


def test_gce_runtime_planning_spec_and_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify single-VM GCE runtime (`runtime="gce"`), anti-orphan spec, and GceProbe."""
    from scale_forecasting.commands import build_gce_commands
    from scale_forecasting.gce_submit import (
        build_gce_instance_spec_dict,
        build_gce_startup_script,
        plan_gce_job,
    )
    from scale_forecasting.probes.runtimes import GceProbe
    from scale_forecasting.quota import gce_demands
    from scale_forecasting.registry.ids import gce_instance_id

    cfg = RunConfig(
        run_name="test-gce",
        data={"source_table": "p.d.t"},
        python_runtime="gce",
        models=["naive_mean", "lightgbm"],
        compute={
            "families": {
                "ml": {
                    "runtime": "gce",
                    "hardware": "cpu",
                    "vertex_machine_type": "n2-standard-8",
                    "vertex_workers": 1,
                }
            },
        },
    )
    preflight(cfg)
    fc_stat = cfg.resolve_family_compute("statistical")
    assert fc_stat.runtime == "gce"
    assert fc_stat.vertex_workers == 1

    # Reject multi-VM on GCE
    with pytest.raises(ValueError, match="single-VM execution only"):
        RunConfig(
            run_name="test-gce-invalid",
            data={"source_table": "p.d.t"},
            python_runtime="gce",
            models=["naive_mean"],
            compute={"vertex_workers": 2},
        ).resolve_family_compute("statistical")

    settings = Settings(
        project_id="p",
        connection="p.us-central1.c",
        warehouse_uri="gs://bkt/warehouse",
        dataset_id="ds",
        region="us-central1",
    )
    run_id = make_run_id(cfg)
    job_key = make_job_key(run_id, "statistical", 1)
    inst_id = _system_job_id(job_key, "gce")
    assert inst_id == gce_instance_id(job_key)

    plan = plan_gce_job(
        cfg,
        ["naive_mean"],
        run_id=run_id,
        instance_name=inst_id,
        hardware="cpu",
        machine_type="n2-standard-8",
        image_uri="us-docker.pkg.dev/p/r/img:latest",
        package_uri="gs://bkt/runs/pkg.zip",
        config_uri="gs://bkt/runs/cfg.json",
        service_account="compute@p.iam.gserviceaccount.com",
        subnetwork_uri="projects/p/regions/us-central1/subnetworks/sub",
        ttl_seconds=3600,
        settings=settings,
    )
    assert plan.instance_name == inst_id
    assert plan.resource_plan is not None
    assert plan.resource_plan.max_units == 1

    script = build_gce_startup_script(plan, driver_args=["--config-uri", plan.config_uri])
    assert "trap cleanup EXIT" in script
    assert "DELETE" in script
    assert plan.status_uri == f"gs://bkt/runs/gce-status/{inst_id}.json"
    assert plan.status_bucket in script

    spec = build_gce_instance_spec_dict(
        plan, project_id="p", zone="us-central1-a", driver_args=["--config-uri", plan.config_uri]
    )
    assert spec["scheduling"]["maxRunDuration"] == {"seconds": "3600"}
    assert spec["scheduling"]["instanceTerminationAction"] == "DELETE"
    assert spec["scheduling"]["automaticRestart"] is False
    assert "accessConfigs" not in spec["networkInterfaces"][0]

    demands = gce_demands(plan, "us-central1")
    assert any(d.metric.metric == "compute.googleapis.com/cpus" for d in demands)

    lc = build_gce_commands(
        config_uri=plan.config_uri,
        package_uri=plan.package_uri,
        settings=settings,
        instance_name=inst_id,
        models=["naive_mean"],
        hardware="cpu",
        machine_type="n2-standard-8",
    )
    assert lc.runtime == "gce"
    assert "scale_forecasting.gce_submit" in lc.universal
    assert get_submitter("gce").name == "gce"
    assert get_probe("gce").name == "gce"

    # Probe tests
    probe = GceProbe()
    marker_state: dict[str, object] | None = {"state": "SUCCEEDED", "exit_code": 0}
    inst_state: dict[str, object] | None = None
    monkeypatch.setattr(
        "scale_forecasting.gce_submit.read_status_marker",
        lambda bucket, name, **kw: marker_state,
    )
    monkeypatch.setattr(
        "scale_forecasting.gce_submit.get_instance",
        lambda project, zone, name, **kw: inst_state,
    )
    monkeypatch.setattr(
        "scale_forecasting.gce_submit.delete_instance",
        lambda project, zone, name, **kw: True,
    )
    handle = ProbeHandle(
        "gce",
        native_id=inst_id,
        region="us-central1",
        resource_name=f"projects/p/zones/us-central1-a/instances/{inst_id}",
    )
    assert probe.check(handle, settings=settings).native_state == NATIVE_SUCCEEDED

    marker_state = {"state": "FAILED", "exit_code": 2}
    assert probe.check(handle, settings=settings).native_state == NATIVE_FAILED

    marker_state = None
    inst_state = {"status": "RUNNING"}
    assert probe.check(handle, settings=settings).native_state == NATIVE_RUNNING

    inst_state = None
    assert probe.check(handle, settings=settings).native_state == NATIVE_NOT_FOUND

    cancel_res = probe.cancel(handle, settings=settings)
    assert cancel_res.stopped is True


def test_vertex_worker_barrier_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `_barrier_gcs_parts` and `_should_use_worker_barrier` for multi-worker CustomJobs."""
    from scale_forecasting.engines.vertex_engine import (
        _barrier_gcs_parts,
        _should_use_worker_barrier,
    )

    settings = Settings(
        project_id="p",
        connection="p.us-central1.c",
        warehouse_uri="gs://wh-bucket/warehouse",
        dataset_id="ds",
        region="us-central1",
    )
    bucket, obj = _barrier_gcs_parts(settings, "run-123", "sf-run-123-stat-a1", 1)
    assert bucket == "wh-bucket"
    assert obj == "warehouse/artifacts/p/ds/run-123/_vertex_barrier/sf-run-123-stat-a1/rank_1.json"

    assert _should_use_worker_barrier(WorkerTopology(0, 1, "CLUSTER_SPEC")) is False
    assert _should_use_worker_barrier(WorkerTopology(0, 2, "CLUSTER_SPEC")) is True
    assert _should_use_worker_barrier(WorkerTopology(0, 2, "JOB_COMPLETION_INDEX")) is True
    monkeypatch.delenv("SF_VERTEX_JOB_ID", raising=False)
    assert _should_use_worker_barrier(WorkerTopology(0, 2, "explicit")) is False
    monkeypatch.setenv("SF_VERTEX_JOB_ID", "sf-run-123-stat-a1")
    assert _should_use_worker_barrier(WorkerTopology(0, 2, "explicit")) is True


def test_gpu_machine_type_resolution_and_validation() -> None:
    """Verify GCP GPU-to-VM compatibility matrix (T4, L4, A100, A100_80GB) & catalog sizing."""
    from scale_forecasting.config import resolve_vm_machine_type
    from scale_forecasting.resources.catalog import machine_cores, machine_memory_bytes

    # Auto defaults across GPU families and counts
    assert resolve_vm_machine_type("gpu", "T4", accelerator_count=1) == "n1-standard-8"
    assert resolve_vm_machine_type("gpu", "T4", accelerator_count=2) == "n1-standard-8"
    assert resolve_vm_machine_type("gpu", "T4", accelerator_count=4) == "n1-standard-16"
    assert resolve_vm_machine_type("gpu", "L4", accelerator_count=1) == "g2-standard-8"
    assert resolve_vm_machine_type("gpu", "L4", accelerator_count=2) == "g2-standard-24"
    assert resolve_vm_machine_type("gpu", "L4", accelerator_count=4) == "g2-standard-48"
    assert resolve_vm_machine_type("gpu", "L4", accelerator_count=8) == "g2-standard-96"
    assert resolve_vm_machine_type("gpu", "A100", accelerator_count=1) == "a2-highgpu-1g"
    assert resolve_vm_machine_type("gpu", "A100", accelerator_count=2) == "a2-highgpu-2g"
    assert resolve_vm_machine_type("gpu", "A100", accelerator_count=4) == "a2-highgpu-4g"
    assert resolve_vm_machine_type("gpu", "A100", accelerator_count=8) == "a2-highgpu-8g"
    assert resolve_vm_machine_type("gpu", "A100", accelerator_count=16) == "a2-megagpu-16g"
    assert resolve_vm_machine_type("gpu", "A100_80GB", accelerator_count=1) == "a2-ultragpu-1g"
    assert resolve_vm_machine_type("gpu", "A100_80GB", accelerator_count=4) == "a2-ultragpu-4g"

    # Catalog core and memory parsing for A2 shapes
    gib = 1024**3
    assert machine_cores("a2-highgpu-1g") == 12
    assert machine_memory_bytes("a2-highgpu-1g") == 85 * gib
    assert machine_cores("a2-megagpu-16g") == 96
    assert machine_memory_bytes("a2-megagpu-16g") == 1360 * gib
    assert machine_cores("a2-ultragpu-2g") == 24
    assert machine_memory_bytes("a2-ultragpu-2g") == 340 * gib

    # FamilyCompute with A100 and A100_80GB
    cfg_a100 = RunConfig(
        run_name="test-a100",
        data={"source_table": "p.d.t"},
        python_runtime="vertex",
        models=["tide"],
        model_params={"tide": {"training_mode": "global", "max_steps": 5}},
        compute={
            "families": {
                "deep_learning": {
                    "runtime": "vertex",
                    "hardware": "gpu",
                    "gpu_type": "A100",
                    "accelerator_count": 2,
                    "workers": 1,
                }
            }
        },
    )
    fc_a100 = cfg_a100.resolve_family_compute("deep_learning")
    assert fc_a100.gpu_type == "A100"
    assert fc_a100.accelerator_count == 2
    assert fc_a100.machine_type == "a2-highgpu-2g"

    cfg_a100_80 = RunConfig(
        run_name="test-a100-80gb",
        data={"source_table": "p.d.t"},
        python_runtime="vertex",
        models=["tide"],
        model_params={"tide": {"training_mode": "global", "max_steps": 5}},
        compute={
            "gpu_type": "A100_80GB",
            "families": {
                "deep_learning": {
                    "runtime": "vertex",
                    "hardware": "gpu",
                    "gpu_type": "A100_80GB",
                    "accelerator_count": 1,
                    "workers": 1,
                }
            },
        },
    )
    fc_a100_80 = cfg_a100_80.resolve_family_compute("deep_learning")
    assert fc_a100_80.gpu_type == "A100_80GB"
    assert fc_a100_80.machine_type == "a2-ultragpu-1g"

    # Universal compute parameters (machine_type, workers, min_workers, max_workers) across runtimes
    cfg_ray_uni = RunConfig(
        run_name="test-ray-uni",
        data={"source_table": "p.d.t"},
        python_runtime="ray",
        models=["theta", "neuralprophet"],
        compute={
            "gpu_type": "A100",
            "families": {
                "statistical": {
                    "runtime": "ray",
                    "machine_type": "n2-standard-8",
                    "min_workers": 2,
                    "max_workers": 6,
                },
                "deep_learning": {
                    "runtime": "ray",
                    "hardware": "gpu",
                    "gpu_type": "L4",
                    "accelerator_count": 2,
                    "machine_type": "auto",
                    "min_workers": 1,
                    "max_workers": 4,
                },
            },
        },
    )
    fc_ray_stat = cfg_ray_uni.resolve_family_compute("statistical")
    assert fc_ray_stat.machine_type == "n2-standard-8"
    assert fc_ray_stat.min_workers == 2 and fc_ray_stat.max_workers == 6
    fc_ray_dl = cfg_ray_uni.resolve_family_compute("deep_learning")
    assert fc_ray_dl.machine_type == "g2-standard-24"
    assert fc_ray_dl.min_workers == 1 and fc_ray_dl.max_workers == 4

    from scale_forecasting.engines.ray_io import plan_cluster

    ray_plan = plan_cluster(cfg_ray_uni, ["theta", "neuralprophet"], run_id="r", use_gpu=True)
    assert ray_plan.cpu_machine_type == "n2-standard-8"
    assert ray_plan.gpu_machine_type == "g2-standard-24"
    assert ray_plan.accelerator_count == 2
    assert ray_plan.cpu_min_nodes == 2 and ray_plan.cpu_max_nodes == 6
    assert ray_plan.gpu_min_nodes == 1 and ray_plan.gpu_max_nodes == 4

    # Reject min_workers/max_workers on non-autoscaling vertex/gce and machine_type on serverless
    with pytest.raises(ValueError, match="min_workers/max_workers are only"):
        RunConfig(
            run_name="bad-vertex-min",
            data={"source_table": "p.d.t"},
            python_runtime="vertex",
            models=["theta"],
            compute={"min_workers": 2},
        ).resolve_family_compute("statistical")
    with pytest.raises(ValueError, match="Dataproc Serverless does not use VM machine_type"):
        RunConfig(
            run_name="bad-serverless-mt",
            data={"source_table": "p.d.t"},
            python_runtime="spark",
            models=["theta"],
            compute={"families": {"statistical": {"machine_type": "n2-standard-8"}}},
        )

    # Reject invalid pairings
    with pytest.raises(ValueError, match="T4 with accelerator_count=1 allows at most 48 vCPUs"):
        resolve_vm_machine_type("gpu", "T4", "n1-standard-64", accelerator_count=1)
    with pytest.raises(ValueError, match="2x L4 GPUs require machine_type='g2-standard-24'"):
        resolve_vm_machine_type("gpu", "L4", "g2-standard-8", accelerator_count=2)
    with pytest.raises(ValueError, match="1x A100 GPUs require machine_type='a2-highgpu-1g'"):
        resolve_vm_machine_type("gpu", "A100", "n1-standard-8", accelerator_count=1)
    with pytest.raises(ValueError, match="CPU workloads cannot use GPU-attached machine family"):
        resolve_vm_machine_type("cpu", None, "auto", "g2-standard-8")


def test_storage_read_api_contiguous_sharding_and_lpt_ordering() -> None:
    """Verify contiguous ts_id range pushdown and LPT chunk scheduling on Vertex/GCE."""
    from scale_forecasting.engines.vertex_engine import (
        build_worker_series_range,
        order_chunks_lpt,
        shard_series_for_worker,
        worker_series_slice,
    )

    ids = [f"S{i:02d}" for i in range(10)]
    assert worker_series_slice(ids, WorkerTopology(0, 3, "cli")) == ["S00", "S01", "S02"]
    assert worker_series_slice(ids, WorkerTopology(1, 3, "cli")) == ["S03", "S04", "S05"]
    assert worker_series_slice(ids, WorkerTopology(2, 3, "cli")) == ["S06", "S07", "S08", "S09"]

    panel = _synthetic_panel(n_series=10, n_steps=5)
    s0 = shard_series_for_worker(panel, WorkerTopology(0, 3, "cli"))
    s1 = shard_series_for_worker(panel, WorkerTopology(1, 3, "cli"))
    s2 = shard_series_for_worker(panel, WorkerTopology(2, 3, "cli"))
    assert list(s0["ts_id"].unique()) == ["S00", "S01", "S02"]
    assert list(s1["ts_id"].unique()) == ["S03", "S04", "S05"]
    assert list(s2["ts_id"].unique()) == ["S06", "S07", "S08", "S09"]

    r0, w0, n0 = build_worker_series_range(ids, None, "ts_id", WorkerTopology(0, 3, "cli"))
    r1, w1, n1 = build_worker_series_range(ids, None, "ts_id", WorkerTopology(1, 3, "cli"))
    r2, w2, n2 = build_worker_series_range(
        ids, 6, "ts_id", WorkerTopology(1, 2, "cli"), hpo_sample_n=2
    )
    assert r0 == "ts_id >= 'S00' AND ts_id <= 'S02'" and w0 == {"S00", "S01", "S02"} and n0 == 10
    assert r1 == "ts_id >= 'S03' AND ts_id <= 'S05'" and w1 == {"S03", "S04", "S05"} and n1 == 10
    assert r2 == "(ts_id >= 'S03' AND ts_id <= 'S05') OR ts_id <= 'S01'"
    assert w2 == {"S03", "S04", "S05"} and n2 == 6

    # LPT chunk ordering places heavier chunks first
    c_small = _synthetic_panel(n_series=1, n_steps=10)
    c_large = _synthetic_panel(n_series=3, n_steps=50)
    c_med = _synthetic_panel(n_series=2, n_steps=20)
    ordered = order_chunks_lpt([c_small, c_large, c_med], ["naive_mean"])
    assert [len(c) for c in ordered] == [150, 40, 10]
