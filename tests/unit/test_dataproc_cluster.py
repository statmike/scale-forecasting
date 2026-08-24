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


# --- build_job -----------------------------------------------------------------


def test_build_job_places_pyspark_job_on_cluster_with_contract() -> None:
    job = dataproc_cluster.build_job(
        cluster="sf-cluster-run-abc",
        launcher_uri="gs://code-bkt/runs/spark_main.py",
        package_uri="gs://code-bkt/runs/pkg-1234.zip",
        config_uri="gs://code-bkt/runs/run-abc.json",
        settings=_settings(),
        engine="explode",
        models=["theta", "holtwinters"],
        manage_header=False,
    )
    assert job.placement.cluster_name == "sf-cluster-run-abc"
    ps = job.pyspark_job
    assert ps.main_python_file_uri == "gs://code-bkt/runs/spark_main.py"
    assert list(ps.python_file_uris) == ["gs://code-bkt/runs/pkg-1234.zip"]
    args = list(ps.args)
    assert args[:4] == ["--engine", "explode", "--config-uri", "gs://code-bkt/runs/run-abc.json"]
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
        engine="explode",
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
        engine="explode",
        use_venv=True,
    )
    ps = job.pyspark_job
    # The venv is delivered by the cluster init action to an absolute path (not attached to the job:
    # job archives reach only executors, never the client-mode driver), so no archive_uris here and
    # both driver + executor Python point at the same absolute interpreter.
    assert list(ps.archive_uris) == []
    assert ps.properties["spark.pyspark.python"] == "/opt/sf-venv/bin/python"
    assert ps.properties["spark.pyspark.driver.python"] == "/opt/sf-venv/bin/python"


def test_build_job_without_venv_stays_bare() -> None:
    job = dataproc_cluster.build_job(
        cluster="sf-cluster-run-abc",
        launcher_uri="gs://c/spark_main.py",
        package_uri="gs://c/pkg.zip",
        config_uri="gs://c/run.json",
        settings=_settings(),
        engine="explode",
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


def test_cluster_init_script_reads_metadata_and_unpacks_to_absolute_dir() -> None:
    script = dataproc_cluster._CLUSTER_INIT_SCRIPT
    # Reads both metadata keys the cluster carries, and unpacks to the absolute venv dir the job's
    # Python points at — so driver + executors resolve the same interpreter.
    assert "attributes/sf-venv-archive-uri" in script
    assert "attributes/sf-venv-dir" in script
    assert 'tar xzf /tmp/sf-venv.tar.gz -C "${VENV_DIR}"' in script
    assert "set -euo pipefail" in script  # a fetch/unpack failure fails the node, not silently bare
    assert dataproc_cluster._VENV_PYTHON == "/opt/sf-venv/bin/python"
