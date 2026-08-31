"""Offline tests for the Dataproc cluster submitter (``scale_forecasting.dataproc_cluster``).

No network: the pure spec builders (`cluster_name`, `build_cluster`, `_gpu_worker`, `build_job`) are
exercised against real ``RunConfig``/``Settings``/``BatchInfra`` objects. The create→submit→delete
lifecycle is the ``@gcp`` cluster smoke; here nothing touches GCP.
"""

from __future__ import annotations

from typing import Any

import pytest

from scale_forecasting.config import RunConfig
from scale_forecasting.errors import ConfigError
from scale_forecasting.settings import Settings
from scale_forecasting.submit import BatchInfra

# build_cluster/build_job assemble google.cloud.dataproc_v1 messages — skip the module cleanly when
# the [spark] extra is absent (parity with test_submit).
pytest.importorskip("google.cloud.dataproc_v1")

from scale_forecasting import dataproc_cluster  # noqa: E402


def _cfg(**over: Any) -> RunConfig:
    base: dict[str, Any] = {
        "run_name": "cluster test",
        "data": {"source_table": "source_series_native", "horizon": 7},
        "models": ["theta", "holtwinters"],
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


def _infra() -> BatchInfra:
    return BatchInfra(
        code_bucket="code-bkt",
        container_image="us-docker.pkg.dev/proj-x/repo/runtime:latest",
        compute_sa="compute@proj-x.iam.gserviceaccount.com",
        subnetwork_uri="projects/proj-x/regions/us-central1/subnetworks/sf",
    )


# --- cluster_name --------------------------------------------------------------


def test_cluster_name_derives_from_run_id() -> None:
    assert dataproc_cluster.cluster_name("run-abc123", None) == "sf-cluster-run-abc123"


def test_cluster_name_reuse_target_wins() -> None:
    assert dataproc_cluster.cluster_name("run-abc", "my-standing-cluster") == "my-standing-cluster"


def test_cluster_name_clamped_no_trailing_hyphen() -> None:
    name = dataproc_cluster.cluster_name("x" * 100, None)
    assert len(name) <= 51
    assert not name.endswith("-")


# --- build_cluster: CPU + GPU wire spec ----------------------------------------


def test_build_cluster_cpu_default_attaches_no_accelerator() -> None:
    cluster = dataproc_cluster.build_cluster(
        infra=_infra(),
        settings=_settings(),
        project_id="proj-x",
        name="sf-cluster-run-abc",
    )
    cfg = cluster.config
    assert cluster.cluster_name == "sf-cluster-run-abc"
    assert cfg.gce_cluster_config.internal_ip_only is True
    assert cfg.gce_cluster_config.service_account == "compute@proj-x.iam.gserviceaccount.com"
    assert cfg.gce_cluster_config.subnetwork_uri.endswith("/subnetworks/sf")
    assert cfg.worker_config.machine_type_uri == dataproc_cluster._DEFAULT_WORKER_MACHINE
    assert list(cfg.worker_config.accelerators) == []
    assert list(cfg.initialization_actions) == []
    assert dict(cfg.gce_cluster_config.metadata) == {}  # no venv wired → no venv metadata
    assert cfg.software_config.image_version == dataproc_cluster._DEFAULT_IMAGE_VERSION


def test_build_cluster_wires_venv_init_action_and_metadata() -> None:
    cluster = dataproc_cluster.build_cluster(
        infra=_infra(),
        settings=_settings(),
        project_id="proj-x",
        name="sf-cluster-run-abc",
        venv_archive_uri="gs://code-bkt/envs/deadbeef.tar.gz",
        venv_init_uri="gs://code-bkt/init/sf-venv-init-abcd1234.sh",
    )
    cfg = cluster.config
    # The archive URI + target dir ride as cluster metadata for the init action to read at create.
    meta = dict(cfg.gce_cluster_config.metadata)
    assert meta["sf-venv-archive-uri"] == "gs://code-bkt/envs/deadbeef.tar.gz"
    assert meta["sf-venv-dir"] == "/opt/sf-venv"
    # The staged init script is a node init action so the venv lands on master + workers alike.
    init = list(cfg.initialization_actions)
    assert len(init) == 1
    assert init[0].executable_file == "gs://code-bkt/init/sf-venv-init-abcd1234.sh"


def test_build_cluster_gpu_with_venv_runs_both_init_actions() -> None:
    cluster = dataproc_cluster.build_cluster(
        infra=_infra(),
        settings=_settings(),
        project_id="proj-x",
        name="sf-cluster-run-abc",
        hardware="gpu",
        gpu_type="T4",
        venv_archive_uri="gs://code-bkt/envs/deadbeef.tar.gz",
        venv_init_uri="gs://code-bkt/init/sf-venv-init-abcd1234.sh",
    )
    # Venv unpack first, then the GPU-driver install — both present, venv before GPU.
    init = [a.executable_file for a in cluster.config.initialization_actions]
    assert init == [
        "gs://code-bkt/init/sf-venv-init-abcd1234.sh",
        dataproc_cluster._GPU_INIT_ACTION,
    ]


def test_build_cluster_gpu_t4_attaches_n1_accelerator_and_init_action() -> None:
    cluster = dataproc_cluster.build_cluster(
        infra=_infra(),
        settings=_settings(),
        project_id="proj-x",
        name="sf-cluster-run-abc",
        hardware="gpu",
        gpu_type="T4",
    )
    cfg = cluster.config
    assert cfg.worker_config.machine_type_uri == "n1-standard-8"
    accels = list(cfg.worker_config.accelerators)
    assert len(accels) == 1
    assert accels[0].accelerator_type_uri == "nvidia-tesla-t4"
    assert accels[0].accelerator_count == 1
    # The GPU-driver install action must be present so the executor's worker can use the card.
    init = list(cfg.initialization_actions)
    assert len(init) == 1
    assert init[0].executable_file == dataproc_cluster._GPU_INIT_ACTION
    # It compiles the driver from source, so it carries a longer-than-default execution timeout.
    assert init[0].execution_timeout == dataproc_cluster._GPU_INIT_TIMEOUT
    assert dataproc_cluster._GPU_INIT_TIMEOUT.total_seconds() > 600  # above the 10-min default
    # An empty cuDNN version tells the init action to skip the cuDNN + NCCL source builds (the
    # deep-learning wheel bundles its own), leaving just the base driver + CUDA.
    assert dict(cfg.gce_cluster_config.metadata)["cudnn-version"] == ""


def test_build_cluster_gpu_disables_secure_boot_only_on_gpu() -> None:
    # The GPU-driver install action loads unsigned NVIDIA kernel modules, which Secure Boot blocks;
    # a GPU cluster turns Secure Boot off (vTPM + integrity monitoring stay on).
    gpu = dataproc_cluster.build_cluster(
        infra=_infra(),
        settings=_settings(),
        project_id="proj-x",
        name="sf-cluster-run-abc",
        hardware="gpu",
        gpu_type="T4",
    )
    shielded = gpu.config.gce_cluster_config.shielded_instance_config
    assert shielded.enable_secure_boot is False
    assert shielded.enable_vtpm is True
    assert shielded.enable_integrity_monitoring is True
    # A CPU cluster builds no kernel modules, so it keeps the image default (no override set).
    cpu = dataproc_cluster.build_cluster(
        infra=_infra(),
        settings=_settings(),
        project_id="proj-x",
        name="sf-cluster-run-abc",
    )
    assert "shielded_instance_config" not in cpu.config.gce_cluster_config


def test_build_cluster_gpu_with_custom_image_bakes_driver_no_init_action() -> None:
    # A pre-baked GPU image delivers the driver, so the cluster boots from it (image_uri on master +
    # workers), drops the create-time GPU-driver init action, and omits image_version (the custom
    # image pins its own Dataproc version). The accelerator + Secure-Boot-off still apply.
    image = "projects/proj-x/global/images/sf-dataproc-gpu-abcd1234"
    cluster = dataproc_cluster.build_cluster(
        infra=_infra(),
        settings=_settings(),
        project_id="proj-x",
        name="sf-cluster-run-abc",
        hardware="gpu",
        gpu_type="T4",
        gpu_image_uri=image,
    )
    cfg = cluster.config
    assert cfg.master_config.image_uri == image
    assert cfg.worker_config.image_uri == image
    # No GPU-driver init action (the driver is baked in); no cuDNN-skip metadata (nothing installs).
    assert list(cfg.initialization_actions) == []
    assert "cudnn-version" not in dict(cfg.gce_cluster_config.metadata)
    # Custom image carries its own version, so image_version is unset.
    assert not cfg.software_config.image_version
    # The physical card is still attached, and unsigned modules still need Secure Boot off.
    assert list(cfg.worker_config.accelerators)[0].accelerator_type_uri == "nvidia-tesla-t4"
    assert cfg.gce_cluster_config.shielded_instance_config.enable_secure_boot is False


def test_build_cluster_gpu_custom_image_keeps_venv_init_action() -> None:
    # The custom image bakes only the OS + driver; the venv (deps) still ships via its init action.
    image = "projects/proj-x/global/images/sf-dataproc-gpu-abcd1234"
    cluster = dataproc_cluster.build_cluster(
        infra=_infra(),
        settings=_settings(),
        project_id="proj-x",
        name="sf-cluster-run-abc",
        hardware="gpu",
        gpu_type="T4",
        gpu_image_uri=image,
        venv_archive_uri="gs://code-bkt/envs/deadbeef.tar.gz",
        venv_init_uri="gs://code-bkt/init/sf-venv-init-abcd1234.sh",
    )
    init = [a.executable_file for a in cluster.config.initialization_actions]
    assert init == ["gs://code-bkt/init/sf-venv-init-abcd1234.sh"]  # venv only, no GPU driver


def test_build_cluster_cpu_ignores_gpu_image_uri() -> None:
    # A stray gpu_image_uri on a CPU cluster is ignored: stock image_version, no per-group image.
    cluster = dataproc_cluster.build_cluster(
        infra=_infra(),
        settings=_settings(),
        project_id="proj-x",
        name="sf-cluster-run-abc",
        gpu_image_uri="projects/proj-x/global/images/sf-dataproc-gpu-abcd1234",
    )
    cfg = cluster.config
    assert cfg.software_config.image_version == dataproc_cluster._DEFAULT_IMAGE_VERSION
    assert not cfg.master_config.image_uri
    assert not cfg.worker_config.image_uri


def test_build_cluster_gpu_l4_attaches_g2_machine() -> None:
    cluster = dataproc_cluster.build_cluster(
        infra=_infra(),
        settings=_settings(),
        project_id="proj-x",
        name="sf-cluster-run-abc",
        hardware="gpu",
        gpu_type="L4",
    )
    cfg = cluster.config
    assert cfg.worker_config.machine_type_uri == "g2-standard-8"
    assert list(cfg.worker_config.accelerators)[0].accelerator_type_uri == "nvidia-l4"


def test_build_cluster_honours_worker_count() -> None:
    cluster = dataproc_cluster.build_cluster(
        infra=_infra(),
        settings=_settings(),
        project_id="proj-x",
        name="sf-cluster-run-abc",
        worker_count=5,
    )
    assert cluster.config.worker_config.num_instances == 5


# --- _gpu_worker ---------------------------------------------------------------


def test_gpu_worker_defaults_to_t4() -> None:
    machine, accels = dataproc_cluster._gpu_worker(None)
    assert machine == "n1-standard-8"
    assert accels[0].accelerator_type_uri == "nvidia-tesla-t4"


def test_gpu_worker_unknown_type_raises() -> None:
    with pytest.raises(ConfigError, match="unsupported gpu_type"):
        dataproc_cluster._gpu_worker("A100")


# --- worker_machine_type / cluster_sizing ---------------------------------------


def test_the_worker_machine_is_read_from_one_place_by_both_readers() -> None:
    """`build_cluster` provisions against it and `cluster_sizing` sizes the executor for it."""
    assert dataproc_cluster.worker_machine_type("cpu") == "n1-standard-8"
    assert dataproc_cluster.worker_machine_type("gpu") == "n1-standard-8"  # a T4 rides an n1
    assert dataproc_cluster.worker_machine_type("gpu", "L4") == "g2-standard-8"
    with pytest.raises(ConfigError, match="unsupported gpu_type"):
        dataproc_cluster.worker_machine_type("gpu", "A100")


def test_profiling_off_sizes_nothing_and_falls_back_to_the_old_cluster() -> None:
    # The documented escape hatch: mode="off" restores the pre-profiler two-worker cluster with
    # no job overlay at all.
    cfg = _cfg(
        data={"source_table": "t", "horizon": 7, "series_limit": 100},
        compute={"profile": {"mode": "off"}},
    )
    assert dataproc_cluster.cluster_sizing(cfg) == (None, {}, {})


def test_a_handed_in_profile_reshapes_the_cluster_executor() -> None:
    """The consumer half on the cluster path — same seam, same reason as the batch path.

    A cluster's shape is fixed at *create*, so the only measurement available here is a previous
    run's. `cluster_sizing` is handed one (by `profiling.profile_for_run`) rather than fetching it,
    which is what keeps this function pure and offline-testable.
    """
    from scale_forecasting.profiling import MeasuredFit, build_profile

    cfg = _cfg(data={"source_table": "t", "horizon": 7, "series_limit": 100})
    profile = build_profile(
        [
            MeasuredFit(
                ts_id=f"s{i}",
                model_type="theta",
                family="statistical",
                n_obs=400,
                wall_s=2.0,
                cpu_s=2.0,
                peak_rss_bytes=0,
                peak_gpu_bytes=None,
                ok=True,
                error=None,
                intraop_threads=1,
                process_rss_bytes=6 * 1024**3,
            )
            for i in range(4)
        ]
    )
    assert dataproc_cluster.cluster_sizing(cfg, profile=profile) != dataproc_cluster.cluster_sizing(
        cfg
    )


def test_the_cluster_sizing_shapes_the_executor_to_the_worker_it_will_run_on() -> None:
    cfg = _cfg(data={"source_table": "t", "horizon": 7, "series_limit": 100})
    workers, props, _sizing = dataproc_cluster.cluster_sizing(cfg)
    # One executor per n1-standard-8 worker, minus the core YARN keeps for the AppMaster.
    assert props["spark.executor.cores"] == "7"
    # Unlike the batch overlay, memory is always stated — Dataproc bakes a stale default
    # otherwise (see resources.translate_cluster).
    assert props["spark.executor.memory"] == "3584m"
    assert props["spark.executorEnv.OMP_NUM_THREADS"] == "1"
    assert props["spark.dynamicAllocation.maxExecutors"] == str(workers)


def test_the_cluster_sizing_also_hands_back_the_record_of_how_it_decided() -> None:
    """The cluster path had no header telemetry at all before this — a cluster run read blank.

    The third element is what `_stamp_cluster_telemetry` files under ``$.sizing.<family>``, so a
    worker split someone questions after an OOM is answerable from the registry.
    """
    cfg = _cfg(data={"source_table": "t", "horizon": 7, "series_limit": 100})
    workers, props, sizing = dataproc_cluster.cluster_sizing(cfg)
    assert sizing["family"] == "statistical"
    assert sizing["profile"] is None  # nothing was handed in; the record says so
    # The record describes the job that was actually submitted, not a second computation of it.
    assert sizing["translation"]["properties"] == props
    assert sizing["translation"]["worker_count"] == workers


def test_the_cluster_sizing_sizes_against_tasks_not_cells() -> None:
    # 100 series x 2 models = 200 cells -> ceil(200/8) = 25 buckets. Seven cells at a time per
    # worker x 8 target each = 56, so one worker suffices and Dataproc's own floor of two stands.
    cfg = _cfg(
        data={"source_table": "t", "horizon": 7, "series_limit": 100},
        compute={"bucket_target_cells": 8},
    )
    workers, _props, _sizing = dataproc_cluster.cluster_sizing(cfg)
    assert workers == 2


def test_the_cluster_sizing_widens_with_the_fan_out_and_stops_at_the_cap() -> None:
    cfg = _cfg(
        data={"source_table": "t", "horizon": 7, "series_limit": 20_000},
        compute={"bucket_target_cells": 8},
    )
    wide, _props, _sizing = dataproc_cluster.cluster_sizing(cfg)
    capped, _props, _sizing = dataproc_cluster.cluster_sizing(cfg, max_workers=3)
    assert wide > 2
    assert capped == 3


def test_the_cluster_sizing_sizes_to_the_executed_subset_not_the_whole_config() -> None:
    # Below the spend cap on both sides, so the comparison is the arithmetic and not the clamp.
    cfg = _cfg(data={"source_table": "t", "horizon": 7, "series_limit": 1_000})
    full, _props, _sizing = dataproc_cluster.cluster_sizing(cfg)
    subset, _props, _sizing = dataproc_cluster.cluster_sizing(cfg, ["theta"])
    assert 2 < subset < full < 10


def test_a_gpu_cluster_bounds_the_cells_that_share_the_card() -> None:
    # The whole point of the cluster translation: without spark.task.cpus a 7-core executor would
    # run seven neuralprophet cells on one T4 whatever fraction the config asked for.
    cfg = _cfg(
        models=["neuralprophet"],
        data={"source_table": "t", "horizon": 7, "series_limit": 100},
        compute={"gpu_fraction": 0.5},
    )
    _workers, props, _sizing = dataproc_cluster.cluster_sizing(cfg, hardware="gpu", gpu_type="T4")
    assert props["spark.task.cpus"] == "3"  # 7 // 3 = 2 cells on the card
    # Withheld deliberately: Dataproc leaves YARN GPU isolation off, so a resource request the
    # NodeManager never advertises fails every executor at launch.
    assert "spark.executor.resource.gpu.amount" not in props
    assert "spark.task.resource.gpu.amount" not in props


# --- build_job -----------------------------------------------------------------


def test_build_job_places_pyspark_job_on_cluster_with_contract() -> None:
    job = dataproc_cluster.build_job(
        cluster="sf-cluster-run-abc",
        launcher_uri="gs://code-bkt/runs/spark_main.py",
        package_uri="gs://code-bkt/runs/pkg-1234.zip",
        config_uri="gs://code-bkt/runs/run-abc.json",
        settings=_settings(),
        models=["theta", "holtwinters"],
        manage_header=False,
    )
    assert job.placement.cluster_name == "sf-cluster-run-abc"
    ps = job.pyspark_job
    assert ps.main_python_file_uri == "gs://code-bkt/runs/spark_main.py"
    assert list(ps.python_file_uris) == ["gs://code-bkt/runs/pkg-1234.zip"]
    args = list(ps.args)
    assert args[:2] == ["--config-uri", "gs://code-bkt/runs/run-abc.json"]
    # The same on-cluster contract as the Serverless batch (subset + contributor-mode header).
    assert args[args.index("--models") + 1] == "theta,holtwinters"
    assert args[args.index("--manage-header") + 1] == "false"


def test_build_job_defaults_omit_oncluster_flags() -> None:
    job = dataproc_cluster.build_job(
        cluster="sf-cluster-run-abc",
        launcher_uri="gs://c/spark_main.py",
        package_uri="gs://c/pkg.zip",
        config_uri="gs://c/run.json",
        settings=_settings(),
    )
    args = list(job.pyspark_job.args)
    assert "--models" not in args
    assert "--manage-header" not in args


# --- packed-venv delivery (cluster dependency mechanism) -----------------------


def test_build_job_use_venv_points_python_at_absolute_venv() -> None:
    job = dataproc_cluster.build_job(
        cluster="sf-cluster-run-abc",
        launcher_uri="gs://c/spark_main.py",
        package_uri="gs://c/pkg.zip",
        config_uri="gs://c/run.json",
        settings=_settings(),
        use_venv=True,
    )
    ps = job.pyspark_job
    # The venv is delivered by the cluster init action to an absolute path (not attached to the job:
    # job archives reach only executors, never the client-mode driver), so no archive_uris here and
    # both driver + executor Python point at the same absolute interpreter.
    assert list(ps.archive_uris) == []
    assert ps.properties["spark.pyspark.python"] == "/opt/sf-venv/bin/python"
    assert ps.properties["spark.pyspark.driver.python"] == "/opt/sf-venv/bin/python"


def test_build_job_carries_the_sizing_overlay() -> None:
    job = dataproc_cluster.build_job(
        cluster="sf-cluster-run-abc",
        launcher_uri="gs://c/spark_main.py",
        package_uri="gs://c/pkg.zip",
        config_uri="gs://c/run.json",
        settings=_settings(),
        properties={"spark.executor.cores": "7", "spark.task.cpus": "3"},
    )
    props = dict(job.pyspark_job.properties)
    assert props["spark.executor.cores"] == "7"
    assert props["spark.task.cpus"] == "3"


def test_the_venv_interpreter_pins_win_over_any_sizing_overlay() -> None:
    """A shape decision must never displace the interpreter — that would run bare Python."""
    job = dataproc_cluster.build_job(
        cluster="sf-cluster-run-abc",
        launcher_uri="gs://c/spark_main.py",
        package_uri="gs://c/pkg.zip",
        config_uri="gs://c/run.json",
        settings=_settings(),
        use_venv=True,
        properties={"spark.pyspark.python": "/usr/bin/python3", "spark.executor.cores": "7"},
    )
    props = dict(job.pyspark_job.properties)
    assert props["spark.pyspark.python"] == "/opt/sf-venv/bin/python"
    assert props["spark.executor.cores"] == "7"


def test_build_job_without_venv_stays_bare() -> None:
    job = dataproc_cluster.build_job(
        cluster="sf-cluster-run-abc",
        launcher_uri="gs://c/spark_main.py",
        package_uri="gs://c/pkg.zip",
        config_uri="gs://c/run.json",
        settings=_settings(),
    )
    ps = job.pyspark_job
    assert list(ps.archive_uris) == []
    assert "spark.pyspark.python" not in dict(ps.properties)


def _infra_with_venv() -> BatchInfra:
    return BatchInfra(
        code_bucket="code-bkt",
        container_image="us-docker.pkg.dev/proj-x/repo/runtime:latest",
        compute_sa="compute@proj-x.iam.gserviceaccount.com",
        subnetwork_uri="projects/proj-x/regions/us-central1/subnetworks/sf",
        venv_archive_uri="gs://code-bkt/envs/deadbeef.tar.gz",
    )


def test_resolve_cluster_deps_returns_archive_for_packed_venv() -> None:
    cfg = _cfg(compute={"spark_deps": "packed_venv"})
    assert (
        dataproc_cluster._resolve_cluster_deps(cfg, _infra_with_venv())
        == "gs://code-bkt/envs/deadbeef.tar.gz"
    )


def test_resolve_cluster_deps_rejects_container_on_cluster() -> None:
    cfg = _cfg(compute={"spark_deps": "container"})
    with pytest.raises(ConfigError, match="Serverless mechanism"):
        dataproc_cluster._resolve_cluster_deps(cfg, _infra_with_venv())


def test_resolve_cluster_deps_requires_archive_uri() -> None:
    cfg = _cfg(compute={"spark_deps": "packed_venv"})
    # infra without a venv archive configured — a cluster forecast job can't run bare.
    with pytest.raises(ConfigError, match="packed-venv archive"):
        dataproc_cluster._resolve_cluster_deps(cfg, _infra())


# --- get_cluster_job: non-blocking state read (probe path) ---------------------


class _FakeJobResult:
    def __init__(self, state_name: str, details: str = "") -> None:
        import types as _types

        self.status = _types.SimpleNamespace(
            state=_types.SimpleNamespace(name=state_name), details=details
        )


class _FakeJobClient:
    def __init__(self, result: Any = None, exc: Exception | None = None) -> None:
        self._result = result
        self._exc = exc
        self.seen: dict[str, Any] = {}

    def get_job(self, *, request: dict[str, Any], timeout: Any) -> Any:
        self.seen["request"] = request
        self.seen["timeout"] = timeout
        if self._exc is not None:
            raise self._exc
        return self._result


def test_get_cluster_job_returns_state_and_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeJobClient(result=_FakeJobResult("RUNNING", details="in progress"))
    monkeypatch.setattr(dataproc_cluster, "_job_client", lambda region: client)

    state, detail = dataproc_cluster.get_cluster_job(
        "us-west1", "job-abc", settings=_settings(), timeout=20.0
    )

    assert (state, detail) == ("RUNNING", "in progress")
    # Addresses the job by (project, region, job_id) and forwards the timeout ceiling.
    assert client.seen["request"] == {
        "project_id": "proj-x",
        "region": "us-west1",
        "job_id": "job-abc",
    }
    assert client.seen["timeout"] == 20.0


def test_get_cluster_job_missing_detail_is_empty_string(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeJobClient(result=_FakeJobResult("DONE"))
    monkeypatch.setattr(dataproc_cluster, "_job_client", lambda region: client)

    state, detail = dataproc_cluster.get_cluster_job("us-central1", "j", settings=_settings())

    assert state == "DONE"
    assert detail == ""


def test_get_cluster_job_propagates_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    # An unknown job id raises NotFound so the probe caller can map it to NOT_FOUND (not a crash).
    from google.api_core.exceptions import NotFound

    monkeypatch.setattr(
        dataproc_cluster, "_job_client", lambda region: _FakeJobClient(exc=NotFound("gone"))
    )

    with pytest.raises(NotFound):
        dataproc_cluster.get_cluster_job("us-central1", "j", settings=_settings())


def test_cluster_init_script_reads_metadata_and_unpacks_to_absolute_dir() -> None:
    script = dataproc_cluster._CLUSTER_INIT_SCRIPT
    # Reads both metadata keys the cluster carries, and unpacks to the absolute venv dir the job's
    # Python points at — so driver + executors resolve the same interpreter.
    assert "attributes/sf-venv-archive-uri" in script
    assert "attributes/sf-venv-dir" in script
    assert 'tar xzf /tmp/sf-venv.tar.gz -C "${VENV_DIR}"' in script
    assert "set -euo pipefail" in script  # a fetch/unpack failure fails the node, not silently bare
    assert dataproc_cluster._VENV_PYTHON == "/opt/sf-venv/bin/python"
