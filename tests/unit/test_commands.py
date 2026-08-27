"""Offline tests for the portable launch-command builder (``scale_forecasting.commands``).

The point of this module is that an emitted command cannot drift from what actually runs, so the
tests are mostly *anti-drift* assertions: the driver args shared by every tier equal what
`submit.build_batch` and `ray_submit.build_entrypoint` really submit, and the native ``gcloud``
command reconstructs the exact batch fields. Pure assembly, no network.
"""

from __future__ import annotations

import shlex
from typing import Any

import pytest

from scale_forecasting.commands import (
    LaunchCommands,
    build_driver_args,
    build_ray_commands,
    build_spark_commands,
    shell_join,
)
from scale_forecasting.config import RunConfig
from scale_forecasting.settings import Settings

# The native-command faithfulness test compares against a real dataproc_v1.Batch; skip cleanly when
# the [spark] extra is absent (parity with test_submit).
pytest.importorskip("google.cloud.dataproc_v1")

from scale_forecasting.ray_submit import build_entrypoint  # noqa: E402
from scale_forecasting.submit import BatchInfra, build_batch  # noqa: E402


def _cfg(**over: Any) -> RunConfig:
    base: dict[str, Any] = {
        "run_name": "cmd test",
        "data": {"source_table": "source_series_native", "horizon": 28},
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


# --- shell_join ----------------------------------------------------------------


def test_shell_join_quotes_only_what_needs_it() -> None:
    line = shell_join(["gcloud", "batches", "submit", "--batch=sf-abc"])
    assert line == "gcloud batches submit --batch=sf-abc"
    # a token with spaces/specials is quoted so the line stays copy-pasteable and re-parses cleanly.
    joined = shell_join(["--label", "run one", "--filter", "a=b;c"])
    assert shlex.split(joined) == ["--label", "run one", "--filter", "a=b;c"]
    assert "'run one'" in joined


# --- build_driver_args ---------------------------------------------------------


def test_driver_args_default_is_config_uri_plus_infra_only() -> None:
    args = build_driver_args("gs://c/runs/r.json", _settings())
    assert args[:2] == ["--config-uri", "gs://c/runs/r.json"]
    assert "--sf-project-id" in args and "proj-x" in args
    # defaults omit the optional flags entirely
    assert "--models" not in args
    assert "--manage-header" not in args


def test_driver_args_config_uri_leads_and_optionals_appended_when_non_default() -> None:
    args = build_driver_args(
        "gs://c/runs/r.json",
        _settings(),
        models=["theta", "holtwinters"],
        manage_header=False,
    )
    # config-uri always leads (there is no method flag); the optionals append when non-default.
    assert args[:2] == ["--config-uri", "gs://c/runs/r.json"]
    assert args[-2:] == ["--manage-header", "false"]
    i = args.index("--models")
    assert args[i + 1] == "theta,holtwinters"


# --- build_spark_commands: anti-drift vs build_batch ---------------------------


def test_spark_native_command_reconstructs_the_exact_batch() -> None:
    settings, infra = _settings(), _infra()
    package_uri = "gs://code-bkt/runs/pkg-1234.zip"
    launcher_uri = "gs://code-bkt/runs/spark_main.py"
    config_uri = "gs://code-bkt/runs/run-abc.json"

    cmds = build_spark_commands(
        settings=settings,
        infra=infra,
        batch_id="sf-run-abc",
        package_uri=package_uri,
        launcher_uri=launcher_uri,
        config_uri=config_uri,
    )
    assert isinstance(cmds, LaunchCommands)
    assert cmds.runtime == "spark"
    native = shlex.split(cmds.native)

    # the batch the launcher would actually submit, for a field-by-field comparison.
    batch = build_batch(
        infra=infra,
        settings=settings,
        package_uri=package_uri,
        launcher_uri=launcher_uri,
        config_uri=config_uri,
    )
    ps = batch.pyspark_batch
    rc = batch.runtime_config
    ec = batch.environment_config.execution_config

    assert native[:5] == ["gcloud", "dataproc", "batches", "submit", "pyspark"]
    assert native[5] == ps.main_python_file_uri == launcher_uri
    assert f"--project={settings.project_id}" in native
    assert f"--region={settings.region}" in native
    assert "--batch=sf-run-abc" in native
    assert f"--py-files={ps.python_file_uris[0]}" in native
    assert f"--version={rc.version}" in native
    assert f"--container-image={rc.container_image}" in native
    assert f"--service-account={ec.service_account}" in native
    assert f"--subnet={ec.subnetwork_uri}" in native
    assert f"--ttl={infra.ttl_seconds}s" in native

    # the driver args after "--" ARE the batch's args, byte for byte (the anti-drift guarantee).
    dash = native.index("--")
    assert native[dash + 1 :] == list(ps.args)


def test_spark_max_executors_sets_native_property_and_universal_flag() -> None:
    cmds = build_spark_commands(
        settings=_settings(),
        infra=_infra(),
        batch_id="sf-x",
        package_uri="gs://c/p.zip",
        launcher_uri="gs://c/e.py",
        config_uri="gs://c/r.json",
        max_executors=4,
    )
    assert "--properties=spark.dynamicAllocation.maxExecutors=4" in shlex.split(cmds.native)
    universal = shlex.split(cmds.universal)
    assert universal[:3] == ["python", "-m", "scale_forecasting.submit"]
    assert universal[3:] == [
        "--config-uri",
        "gs://c/r.json",
        "--max-executors",
        "4",
    ]


def test_spark_universal_omits_max_executors_when_unset() -> None:
    cmds = build_spark_commands(
        settings=_settings(),
        infra=_infra(),
        batch_id="sf-x",
        package_uri="gs://c/p.zip",
        launcher_uri="gs://c/e.py",
        config_uri="gs://c/r.json",
    )
    assert "--properties=" not in cmds.native
    assert shlex.split(cmds.universal) == [
        "python",
        "-m",
        "scale_forecasting.submit",
        "--config-uri",
        "gs://c/r.json",
    ]


# --- build_ray_commands: universal-only, anti-drift vs build_entrypoint --------


def test_ray_has_no_native_form_and_a_portable_universal_one() -> None:
    cmds = build_ray_commands(config_uri="gs://c/r.json")
    assert cmds.runtime == "ray"
    assert cmds.native is None
    assert shlex.split(cmds.universal) == [
        "python",
        "-m",
        "scale_forecasting.ray_submit",
        "--config-uri",
        "gs://c/r.json",
    ]


def test_ray_universal_emits_cluster_name_only_when_reusing() -> None:
    cmds = build_ray_commands(config_uri="gs://c/r.json", cluster_name="standing-1")
    assert shlex.split(cmds.universal)[-2:] == ["--cluster-name", "standing-1"]


def test_ray_entrypoint_shares_the_driver_args() -> None:
    # the on-cluster ray_entry command must carry exactly build_driver_args (no engine for Ray).
    settings = _settings()
    entry = build_entrypoint("gs://c/r.json", settings, models=["theta"], manage_header=False)
    parts = entry.split(" ")
    assert parts[:3] == ["python", "-m", "scale_forecasting.ray_entry"]
    assert parts[3:] == build_driver_args(
        "gs://c/r.json", settings, models=["theta"], manage_header=False
    )
