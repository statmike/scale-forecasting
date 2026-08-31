"""Offline tests for the Dataproc submit helper (``scale_forecasting.submit``).

No network: the pure batch-spec assembly (:func:`build_batch`), the ``n_series`` scale override,
infra resolution (:class:`BatchInfra`), and the batch-id slug are all exercised against real
``RunConfig``/``Settings`` objects. The live ``create_batch`` path is covered by the ``@gcp``
serverless smoke.
"""

from __future__ import annotations

from typing import Any

import pytest

from scale_forecasting.config import RunConfig
from scale_forecasting.errors import ConfigError
from scale_forecasting.profiling import MeasuredFit, build_profile
from scale_forecasting.settings import Settings
from scale_forecasting.submit import (
    BatchInfra,
    _batch_id,
    build_batch,
    sizing_properties,
)

# pyspark is not needed here, but google-cloud-dataproc (the [spark] extra) is — build_batch
# imports dataproc_v1. Skip the whole module cleanly when the extra is absent (parity with @spark).
pytest.importorskip("google.cloud.dataproc_v1")


def _cfg(**over: Any) -> RunConfig:
    base: dict[str, Any] = {
        "run_name": "submit test",
        "data": {"source_table": "source_series_native", "horizon": 28},
        "models": ["theta", "holtwinters", "xgboost", "lightgbm"],
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


# --- batch id ------------------------------------------------------------------


def test_batch_id_is_sf_prefixed_and_clamped() -> None:
    bid = _batch_id("run-abc123")
    assert bid == "sf-run-abc123"
    assert len(bid) <= 63
    # a very long run_id is trimmed, never leaving a trailing hyphen (Dataproc rejects it).
    long = _batch_id("x" * 100)
    assert len(long) == 63
    assert not long.endswith("-")


# --- build_batch: the wire spec ------------------------------------------------


def test_build_batch_wires_launcher_package_and_args() -> None:
    batch = build_batch(
        infra=_infra(),
        settings=_settings(),
        package_uri="gs://code-bkt/runs/pkg-1234.zip",
        launcher_uri="gs://code-bkt/runs/spark_main.py",
        config_uri="gs://code-bkt/runs/run-abc.json",
    )
    ps = batch.pyspark_batch
    assert ps.main_python_file_uri == "gs://code-bkt/runs/spark_main.py"
    assert list(ps.python_file_uris) == ["gs://code-bkt/runs/pkg-1234.zip"]
    # config-uri leads; the --sf-* infra args follow (the Dataproc delivery path).
    assert ps.args[:2] == ["--config-uri", "gs://code-bkt/runs/run-abc.json"]
    assert "--sf-project-id" in ps.args
    assert "proj-x" in ps.args
    rc = batch.runtime_config
    assert rc.container_image == "us-docker.pkg.dev/proj-x/repo/runtime:latest"
    ec = batch.environment_config.execution_config
    assert ec.service_account == "compute@proj-x.iam.gserviceaccount.com"
    assert ec.subnetwork_uri.endswith("/subnetworks/sf")


def test_build_batch_sets_explicit_ttl_over_dataproc_default() -> None:
    # An explicit ttl must be on the batch — Dataproc's silent 4h default would cancel a healthy
    # long 100k run mid-flight (before it writes its run_registry summary). Default is 24h.
    from datetime import timedelta

    from scale_forecasting import submit

    batch = build_batch(
        infra=_infra(),
        settings=_settings(),
        package_uri="gs://c/p.zip",
        launcher_uri="gs://c/e.py",
        config_uri="gs://c/r.json",
    )
    ec = batch.environment_config.execution_config
    assert ec.ttl == timedelta(seconds=submit._DEFAULT_TTL_SECONDS)
    assert ec.ttl > timedelta(hours=4)  # the point: strictly beyond the platform default


def test_build_batch_honours_custom_ttl() -> None:
    from dataclasses import replace
    from datetime import timedelta

    batch = build_batch(
        infra=replace(_infra(), ttl_seconds=10800),
        settings=_settings(),
        package_uri="gs://c/p.zip",
        launcher_uri="gs://c/e.py",
        config_uri="gs://c/r.json",
    )
    assert batch.environment_config.execution_config.ttl == timedelta(seconds=10800)


def test_build_batch_without_max_executors_sets_no_cap() -> None:
    batch = build_batch(
        infra=_infra(),
        settings=_settings(),
        package_uri="gs://c/p.zip",
        launcher_uri="gs://c/e.py",
        config_uri="gs://c/r.json",
    )
    assert "spark.dynamicAllocation.maxExecutors" not in dict(batch.runtime_config.properties)


def test_build_batch_defaults_omit_oncluster_flags() -> None:
    # Standalone submit (no models subset, header-owning) must build the exact arg list it always
    # did — no --models / --manage-header — so existing batches and callers are byte-stable.
    batch = build_batch(
        infra=_infra(),
        settings=_settings(),
        package_uri="gs://c/p.zip",
        launcher_uri="gs://c/e.py",
        config_uri="gs://c/r.json",
    )
    args = list(batch.pyspark_batch.args)
    assert "--models" not in args
    assert "--manage-header" not in args


def test_build_batch_appends_oncluster_flags_when_non_default() -> None:
    # main.run's contributor launch: restrict the executed subset + hand header ownership to main.
    batch = build_batch(
        infra=_infra(),
        settings=_settings(),
        package_uri="gs://c/p.zip",
        launcher_uri="gs://c/e.py",
        config_uri="gs://c/r.json",
        models=["theta", "holtwinters"],
        manage_header=False,
    )
    args = list(batch.pyspark_batch.args)
    assert args[args.index("--models") + 1] == "theta,holtwinters"
    assert args[args.index("--manage-header") + 1] == "false"


def test_build_batch_max_executors_caps_dynamic_allocation() -> None:
    # The executor throttle: capping executors is what makes a straggler visible.
    batch = build_batch(
        infra=_infra(),
        settings=_settings(),
        package_uri="gs://c/p.zip",
        launcher_uri="gs://c/e.py",
        config_uri="gs://c/r.json",
        max_executors=2,
    )
    assert dict(batch.runtime_config.properties)["spark.dynamicAllocation.maxExecutors"] == "2"


def test_build_batch_cpu_default_adds_no_accelerator_properties() -> None:
    batch = build_batch(
        infra=_infra(),
        settings=_settings(),
        package_uri="gs://c/p.zip",
        launcher_uri="gs://c/e.py",
        config_uri="gs://c/r.json",
    )
    props = dict(batch.runtime_config.properties)
    assert not any(k.startswith("spark.dataproc.executor.resource.accelerator") for k in props)


def test_build_batch_gpu_attaches_l4_executor() -> None:
    batch = build_batch(
        infra=_infra(),
        settings=_settings(),
        package_uri="gs://c/p.zip",
        launcher_uri="gs://c/e.py",
        config_uri="gs://c/r.json",
        hardware="gpu",
        gpu_type="L4",
    )
    props = dict(batch.runtime_config.properties)
    assert props["spark.dataproc.executor.resource.accelerator.type"] == "l4"
    assert props["spark.dataproc.executor.compute.tier"] == "premium"
    # L4 on Serverless mandates the premium disk tier; the Spark-level GPU resource-scheduling
    # properties are unsupported on Serverless and must not be set.
    assert props["spark.dataproc.executor.disk.tier"] == "premium"
    assert not any(k.startswith("spark.executor.resource.gpu") for k in props)
    assert "spark.task.resource.gpu.amount" not in props


def test_build_batch_gpu_rejects_non_l4_on_serverless() -> None:
    with pytest.raises(ConfigError, match="Serverless supports L4 only"):
        build_batch(
            infra=_infra(),
            settings=_settings(),
                package_uri="gs://c/p.zip",
            launcher_uri="gs://c/e.py",
            config_uri="gs://c/r.json",
            hardware="gpu",
            gpu_type="T4",
        )


# --- BatchInfra resolution -----------------------------------------------------


def test_batch_infra_resolve_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SF_CODE_BUCKET", "code-bkt")
    monkeypatch.setenv("SF_CONTAINER_IMAGE", "img:tag")
    monkeypatch.setenv("SF_COMPUTE_SA", "sa@x.iam")
    monkeypatch.setenv("SF_SUBNETWORK_URI", "projects/p/regions/r/subnetworks/s")
    infra = BatchInfra.resolve()
    assert infra.code_bucket == "code-bkt"
    assert infra.container_image == "img:tag"
    assert infra.runtime_version == "2.2"  # default
    assert infra.venv_archive_uri is None  # optional, absent by default
    assert infra.gpu_image_uri is None  # optional, absent by default


def test_batch_infra_resolve_reads_venv_archive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SF_CODE_BUCKET", "code-bkt")
    monkeypatch.setenv("SF_CONTAINER_IMAGE", "img:tag")
    monkeypatch.setenv("SF_COMPUTE_SA", "sa@x.iam")
    monkeypatch.setenv("SF_SUBNETWORK_URI", "projects/p/regions/r/subnetworks/s")
    monkeypatch.setenv("SF_VENV_ARCHIVE", "gs://code-bkt/envs/deadbeef.tar.gz")
    infra = BatchInfra.resolve()
    assert infra.venv_archive_uri == "gs://code-bkt/envs/deadbeef.tar.gz"


def test_batch_infra_resolve_reads_gpu_image(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SF_CODE_BUCKET", "code-bkt")
    monkeypatch.setenv("SF_CONTAINER_IMAGE", "img:tag")
    monkeypatch.setenv("SF_COMPUTE_SA", "sa@x.iam")
    monkeypatch.setenv("SF_SUBNETWORK_URI", "projects/p/regions/r/subnetworks/s")
    monkeypatch.setenv("SF_GPU_IMAGE", "projects/p/global/images/sf-dataproc-gpu-abcd1234")
    infra = BatchInfra.resolve()
    assert infra.gpu_image_uri == "projects/p/global/images/sf-dataproc-gpu-abcd1234"


def test_batch_infra_resolve_missing_var_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("SF_CODE_BUCKET", "SF_CONTAINER_IMAGE", "SF_COMPUTE_SA", "SF_SUBNETWORK_URI"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ConfigError, match="SF_CODE_BUCKET"):
        BatchInfra.resolve()


def test_batch_infra_from_terraform_outputs() -> None:
    infra = BatchInfra.from_terraform_outputs(
        {
            "code_bucket": "code-bkt",
            "runtime_image_repo": "us-docker.pkg.dev/p/repo/runtime",
            "compute_sa": "sa@x.iam",
            "subnetwork_uri": "projects/p/regions/r/subnetworks/s",
        },
        image_tag="v1",
    )
    assert infra.container_image == "us-docker.pkg.dev/p/repo/runtime:v1"
    assert infra.venv_archive_uri is None  # absent from outputs → None


def test_batch_infra_from_terraform_outputs_reads_venv_archive() -> None:
    infra = BatchInfra.from_terraform_outputs(
        {
            "code_bucket": "code-bkt",
            "runtime_image_repo": "us-docker.pkg.dev/p/repo/runtime",
            "compute_sa": "sa@x.iam",
            "subnetwork_uri": "projects/p/regions/r/subnetworks/s",
            "venv_archive_uri": "gs://code-bkt/envs/deadbeef.tar.gz",
            "gpu_image_uri": "projects/p/global/images/sf-dataproc-gpu-abcd1234",
        }
    )
    assert infra.venv_archive_uri == "gs://code-bkt/envs/deadbeef.tar.gz"
    assert infra.gpu_image_uri == "projects/p/global/images/sf-dataproc-gpu-abcd1234"


def test_batch_infra_from_terraform_outputs_missing_key_raises() -> None:
    with pytest.raises(ConfigError, match="subnetwork_uri"):
        BatchInfra.from_terraform_outputs(
            {"code_bucket": "b", "runtime_image_repo": "r", "compute_sa": "s"}
        )


# --- submit_batch: n_series override + client wiring (mocked) -------------------


def test_submit_batch_applies_n_series_and_wires_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """submit_batch overrides series_limit, stages code+config, and calls create_batch once."""
    from scale_forecasting import submit

    staged: dict[str, Any] = {}

    def _fake_stage_code(infra: BatchInfra) -> tuple[str, str]:
        return ("gs://code-bkt/runs/pkg.zip", "gs://code-bkt/runs/spark_main.py")

    def _fake_stage_config(cfg: RunConfig, run_id: str, infra: BatchInfra) -> str:
        staged["series_limit"] = cfg.data.series_limit
        staged["run_id"] = run_id
        return f"gs://code-bkt/runs/{run_id}.json"

    class _FakeOp:
        def result(self, timeout: float | None = None) -> Any:
            staged["wait_timeout"] = timeout
            return type("R", (), {"state": "SUCCEEDED"})()

    class _FakeClient:
        def create_batch(self, *, parent: str, batch: Any, batch_id: str) -> _FakeOp:
            staged["parent"] = parent
            staged["batch_id"] = batch_id
            staged["engine_arg"] = list(batch.pyspark_batch.args)
            return _FakeOp()

        def get_batch(self, *, name: str) -> Any:  # telemetry fetch after terminal state
            staged["get_batch_name"] = name
            return type("B", (), {})()

    monkeypatch.setattr(submit, "_stage_code", _fake_stage_code)
    monkeypatch.setattr(submit, "_stage_config", _fake_stage_config)
    monkeypatch.setattr(submit, "_batch_client", lambda region: _FakeClient())
    # Telemetry stamping calls the live update_header; stub it so this stays offline. (It's
    # best-effort and would be swallowed anyway, but stubbing keeps the test off BigQuery.)
    monkeypatch.setattr(submit, "_stamp_job_telemetry", lambda *a, **k: None)

    batch_id = submit.submit_batch(
        _cfg(models=["theta"]),
        n_series=1000,
        settings=_settings(),
        infra=_infra(),
        wait=True,
    )
    assert staged["series_limit"] == 1000  # the scale override reached the staged config
    assert staged["parent"] == "projects/proj-x/locations/us-central1"
    assert batch_id.startswith("sf-")
    assert staged["batch_id"] == batch_id
    # The blocking wait must use the long timeout (a 100k batch exceeds api-core's 900s default),
    # not the bare no-arg result() that regressed to it.
    assert staged["wait_timeout"] == submit._WAIT_TIMEOUT_SECONDS


def test_submit_batch_raises_on_failed_terminal_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """A FAILED batch must raise (not exit 0) — else the header stays RUNNING, failure silent."""
    from scale_forecasting import submit
    from scale_forecasting.errors import EngineError

    class _State:
        name = "FAILED"

    class _FakeResult:
        state = _State()
        state_message = "ImportError: attempted relative import with no known parent package"

    class _FakeOp:
        def result(self, timeout: float | None = None) -> Any:
            return _FakeResult()

    class _FakeClient:
        def create_batch(self, *, parent: str, batch: Any, batch_id: str) -> _FakeOp:
            return _FakeOp()

    monkeypatch.setattr(
        submit, "_stage_code", lambda infra: ("gs://c/p.zip", "gs://c/spark_main.py")
    )
    monkeypatch.setattr(submit, "_stage_config", lambda cfg, run_id, infra: "gs://c/r.json")
    monkeypatch.setattr(submit, "_batch_client", lambda region: _FakeClient())
    # Telemetry is stamped even on a FAILED batch (so its sizing is recorded), before the raise —
    # stub it out so this stays offline and the raise is what the test observes.
    monkeypatch.setattr(submit, "_stamp_job_telemetry", lambda *a, **k: None)

    with pytest.raises(EngineError, match="FAILED"):
        submit.submit_batch(
            _cfg(models=["theta"]),
                settings=_settings(),
            infra=_infra(),
            wait=True,
        )


# --- extract_job_telemetry: flatten the Dataproc Batch into the header overlay ---
#
# Fakes mirror the google.cloud.dataproc_v1.Batch shape (nested attrs / a properties dict / proto
# Timestamps with .timestamp()); values are the real ones observed on the n=10 serverless smoke.


class _FakeTs:
    def __init__(self, epoch: float) -> None:
        self._epoch = epoch

    def timestamp(self) -> float:
        return self._epoch


class _FakeUsage:
    milli_dcu_seconds = 39723675
    shuffle_storage_gb_seconds = 3997200


class _FakeRuntimeInfo:
    approximate_usage = _FakeUsage()


class _FakeRuntimeConfig:
    version = "2.2.82"
    container_image = "us-central1-docker.pkg.dev/p/repo/spark-runtime:latest"
    properties = {
        "spark.driver.cores": "4",
        "spark.executor.cores": "4",
        "spark.executor.instances": "2",
        "spark.dynamicAllocation.maxExecutors": "2",
    }


class _FakeExecConfig:
    service_account = "compute@p.iam.gserviceaccount.com"
    subnetwork_uri = "projects/p/regions/us-central1/subnetworks/sf"


class _FakeEnvConfig:
    execution_config = _FakeExecConfig()


class _FakeBatch:
    # create→state span of 562s (the n=10 provision→terminal wall-clock).
    create_time = _FakeTs(1_000_000.0)
    state_time = _FakeTs(1_000_562.0)
    runtime_info = _FakeRuntimeInfo()
    runtime_config = _FakeRuntimeConfig()
    environment_config = _FakeEnvConfig()


def test_extract_job_telemetry_full_batch() -> None:
    from scale_forecasting.submit import extract_job_telemetry

    tel = extract_job_telemetry(_FakeBatch())
    assert tel["total_wall_s"] == 562.0
    assert tel["dcu_milli_seconds"] == 39723675
    assert tel["shuffle_storage_gb_seconds"] == 3997200
    assert tel["driver_cores"] == 4
    assert tel["executor_cores"] == 4
    assert tel["executor_instances"] == 2
    assert tel["max_executors"] == 2
    assert tel["runtime_version"] == "2.2.82"
    assert tel["container_image"].endswith("spark-runtime:latest")
    assert tel["service_account"] == "compute@p.iam.gserviceaccount.com"
    assert tel["subnetwork_uri"].endswith("/subnetworks/sf")


def test_extract_job_telemetry_is_json_serializable() -> None:
    import json

    from scale_forecasting.submit import extract_job_telemetry

    # The header stores it as a JSON STRING (Iceberg rejects native JSON) — it must round-trip.
    tel = extract_job_telemetry(_FakeBatch())
    assert json.loads(json.dumps(tel, sort_keys=True))["total_wall_s"] == 562.0


def test_extract_job_telemetry_degrades_on_empty_batch() -> None:
    # A batch object missing every sub-message must yield all-None keys, never raise (best-effort).
    from scale_forecasting.submit import extract_job_telemetry

    tel = extract_job_telemetry(type("Empty", (), {})())
    assert tel["total_wall_s"] is None
    assert tel["dcu_milli_seconds"] is None
    assert tel["executor_instances"] is None
    assert tel["runtime_version"] is None
    assert tel["service_account"] is None


def test_extract_job_telemetry_no_executor_cap_when_unset() -> None:
    # No dynamicAllocation.maxExecutors property (unthrottled explode) → max_executors is None.
    from scale_forecasting.submit import extract_job_telemetry

    class _RC:
        version = "2.2"
        container_image = "img:tag"
        properties = {"spark.executor.instances": "8"}

    class _B:
        runtime_config = _RC()

    tel = extract_job_telemetry(_B())
    assert tel["executor_instances"] == 8
    assert tel["max_executors"] is None


# --- sizing overlay ------------------------------------------------------------


def test_build_batch_applies_the_sizing_overlay_verbatim() -> None:
    batch = build_batch(
        infra=_infra(),
        settings=_settings(),
        package_uri="gs://c/p.zip",
        launcher_uri="gs://c/e.py",
        config_uri="gs://c/r.json",
        properties={"spark.executor.cores": "8", "spark.task.cpus": "2"},
    )
    props = dict(batch.runtime_config.properties)
    assert props["spark.executor.cores"] == "8"
    assert props["spark.task.cpus"] == "2"


def test_an_explicit_max_executors_overrides_the_overlays_ceiling() -> None:
    # The overlay is a derived default; --max-executors is the operator saying a number out loud.
    batch = build_batch(
        infra=_infra(),
        settings=_settings(),
        package_uri="gs://c/p.zip",
        launcher_uri="gs://c/e.py",
        config_uri="gs://c/r.json",
        max_executors=2,
        properties={"spark.dynamicAllocation.maxExecutors": "500"},
    )
    props = dict(batch.runtime_config.properties)
    assert props["spark.dynamicAllocation.maxExecutors"] == "2"


def test_the_gpu_attachment_wins_over_anything_the_overlay_said() -> None:
    batch = build_batch(
        infra=_infra(),
        settings=_settings(),
        package_uri="gs://c/p.zip",
        launcher_uri="gs://c/e.py",
        config_uri="gs://c/r.json",
        hardware="gpu",
        gpu_type="L4",
        properties={"spark.dataproc.executor.compute.tier": "standard"},
    )
    props = dict(batch.runtime_config.properties)
    assert props["spark.dataproc.executor.compute.tier"] == "premium"


def test_no_overlay_leaves_the_batch_with_the_pre_profiler_empty_properties() -> None:
    # The pre-profiler CPU batch set no properties at all — every spark.* knob was the service's.
    # Asserting the concrete emptiness (not "equals itself with properties=None", which cannot
    # fail) is what would catch the overlay leaking in through a default.
    batch = build_batch(
        infra=_infra(),
        settings=_settings(),
        package_uri="gs://c/p.zip",
        launcher_uri="gs://c/e.py",
        config_uri="gs://c/r.json",
    )
    assert dict(batch.runtime_config.properties) == {}


def test_profiling_off_emits_no_sizing_properties_at_all() -> None:
    # The documented escape hatch: mode="off" restores the pre-profiler batch exactly.
    cfg = _cfg(
        data={"source_table": "t", "series_limit": 100},
        compute={"profile": {"mode": "off"}},
    )
    assert sizing_properties(cfg) == {}


def test_the_overlay_sizes_against_tasks_not_cells() -> None:
    # 100 series × 4 models = 400 cells → ceil(400/8) = 50 buckets. One 4-core executor runs 4
    # single-core tasks, so 13 executors is the widest fleet the fan-out can keep busy — sizing
    # against the 400 cells instead would ask for 100 and leave most of them idle.
    cfg = _cfg(data={"source_table": "t", "series_limit": 100}, compute={"bucket_target_cells": 8})
    props = sizing_properties(cfg)
    assert props["spark.executor.cores"] == "4"
    assert props["spark.dynamicAllocation.maxExecutors"] == "13"
    assert props["spark.dynamicAllocation.minExecutors"] == "2"


def test_the_overlay_pins_native_thread_pools_so_tasks_do_not_thrash() -> None:
    cfg = _cfg(data={"source_table": "t", "series_limit": 100})
    props = sizing_properties(cfg)
    # One task per core, so every task gets exactly one BLAS/OMP thread.
    assert props["spark.executorEnv.OMP_NUM_THREADS"] == "1"
    assert props["spark.executorEnv.MKL_NUM_THREADS"] == "1"


def test_the_overlay_asks_for_no_memory_when_nothing_measured_it() -> None:
    # There is no submit-side probe — the shape is fixed before any of our code runs on the
    # cluster — so with no profile handed in, the memory axis is absent and the platform's own
    # defaults stand. "Absence is a value."
    cfg = _cfg(data={"source_table": "t", "series_limit": 100})
    props = sizing_properties(cfg)
    assert "spark.executor.memory" not in props
    assert "spark.executor.memoryOverhead" not in props


def test_a_handed_in_profile_is_what_turns_the_memory_axis_on() -> None:
    """The consumer half, at the seam that matters: evidence in, a sized executor out.

    `sizing_properties` stays pure — it is *given* a profile (resolved by
    `profiling.profile_for_run` from `compute.profile.source`), it never goes and finds one. This
    is the whole payoff of the harvest: a previous run's measurements are the only way this call
    can know how much memory a fit needs, because the executor's shape is fixed at submit.
    """
    cfg = _cfg(data={"source_table": "t", "series_limit": 100})
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
                process_rss_bytes=3 * 1024**3,
            )
            for i in range(4)
        ]
    )
    props = sizing_properties(cfg, profile=profile)
    assert "spark.executor.memory" in props
    assert props != sizing_properties(cfg)


def test_the_overlay_respects_an_explicit_executor_cap() -> None:
    cfg = _cfg(data={"source_table": "t", "series_limit": 100})
    props = sizing_properties(cfg, max_executors=4)
    assert props["spark.dynamicAllocation.maxExecutors"] == "4"


def test_the_overlay_sizes_to_the_executed_subset_not_the_whole_config() -> None:
    cfg = _cfg(data={"source_table": "t", "series_limit": 100})
    full = sizing_properties(cfg)
    subset = sizing_properties(cfg, ["theta"])
    assert int(subset["spark.dynamicAllocation.maxExecutors"]) < int(
        full["spark.dynamicAllocation.maxExecutors"]
    )


def test_a_gpu_overlay_uses_the_configs_own_gpu_fraction_not_the_nominal_one() -> None:
    # On the GPU path executor.cores IS the per-task device share, so a config that says a cell
    # takes a tenth of a card must not be sized as if it took half: 8 cells per L4 rather than 4.
    common: dict[str, Any] = dict(data={"source_table": "t", "series_limit": 100})
    half = sizing_properties(_cfg(**common, compute={"gpu_fraction": 0.5}), hardware="gpu")
    tenth = sizing_properties(_cfg(**common, compute={"gpu_fraction": 0.1}), hardware="gpu")
    assert half["spark.executor.cores"] == "4"
    assert tenth["spark.executor.cores"] == "8"


def _submit_capturing_properties(
    monkeypatch: pytest.MonkeyPatch, cfg: RunConfig, **kwargs: Any
) -> dict[str, str]:
    """Run ``submit_batch`` against stubbed staging/client and return the batch's properties."""
    from scale_forecasting import submit

    captured: dict[str, str] = {}

    class _FakeOp:
        def result(self, timeout: float | None = None) -> Any:
            return type("R", (), {"state": "SUCCEEDED"})()

    class _FakeClient:
        def create_batch(self, *, parent: str, batch: Any, batch_id: str) -> _FakeOp:
            captured.update(dict(batch.runtime_config.properties))
            return _FakeOp()

        def get_batch(self, *, name: str) -> Any:
            return type("B", (), {})()

    monkeypatch.setattr(submit, "_stage_code", lambda infra: ("gs://c/p.zip", "gs://c/e.py"))
    monkeypatch.setattr(submit, "_stage_config", lambda cfg, run_id, infra: "gs://c/r.json")
    monkeypatch.setattr(submit, "_batch_client", lambda region: _FakeClient())
    monkeypatch.setattr(submit, "_stamp_job_telemetry", lambda *a, **k: None)
    submit.submit_batch(cfg, settings=_settings(), infra=_infra(), **kwargs)
    return captured


def test_submit_batch_puts_the_sizing_overlay_on_the_batch_it_creates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The unit above proves the overlay is *computed* correctly; this proves it is *wired*.
    # Without it, dropping `properties=sizing_properties(...)` from submit_batch is invisible.
    cfg = _cfg(data={"source_table": "t", "series_limit": 100})
    props = _submit_capturing_properties(monkeypatch, cfg)
    assert props == sizing_properties(cfg)
    assert props  # and it is not the vacuously-passing empty dict


def test_submit_batch_with_profiling_off_creates_the_pre_profiler_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(
        data={"source_table": "t", "series_limit": 100},
        compute={"profile": {"mode": "off"}},
    )
    assert _submit_capturing_properties(monkeypatch, cfg) == {}


# --- plan_sizing: the same overlay, plus the record of how it was chosen ---------


def test_the_sizing_call_hands_back_the_overlay_and_the_audit_record_together() -> None:
    """One computation, two consumers — the batch gets properties, the header gets the reason.

    Computing them separately would let the record drift from the batch it claims to describe.
    """
    from scale_forecasting.submit import plan_sizing

    cfg = _cfg(data={"source_table": "t", "series_limit": 100})
    props, sizing = plan_sizing(cfg)
    assert props == sizing_properties(cfg)  # the submitting caller's half is unchanged
    assert sizing["family"] == "statistical+ml"
    assert sizing["plans"] and sizing["translation"]
    # No evidence handed in → the record says so rather than omitting the question.
    assert sizing["profile"] is None
    assert props["spark.executor.cores"] == str(sizing["translation"]["executor_cores"])


def test_profiling_off_records_no_sizing_either() -> None:
    # Nothing was decided, so there is nothing to file — and an empty record is what the stamp
    # checks to skip the write entirely.
    from scale_forecasting.submit import plan_sizing

    cfg = _cfg(
        data={"source_table": "t", "series_limit": 100},
        compute={"profile": {"mode": "off"}},
    )
    assert plan_sizing(cfg) == ({}, {})


def test_the_record_names_the_evidence_it_sized_from() -> None:
    from scale_forecasting.submit import plan_sizing

    cfg = _cfg(data={"source_table": "t", "series_limit": 100})
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
                process_rss_bytes=3 * 1024**3,
            )
            for i in range(4)
        ]
    )
    _props, sizing = plan_sizing(cfg, profile=profile)
    assert sizing["profile"] is not None
    assert sizing["profile"]["n_measurements"] == 4
    assert sizing["profile"]["models"]["theta"]["n_fits"] == 4


def test_the_echoed_telemetry_carries_the_memory_the_batch_was_actually_given() -> None:
    """The other half of "what shape ran": cores were already echoed, memory was not.

    Read back off the submitted batch rather than off our own plan, so a plan/platform
    disagreement is visible instead of assumed away.
    """
    from scale_forecasting.submit import extract_job_telemetry

    class _RC:
        version = "2.2"
        container_image = "img:tag"
        properties = {"spark.executor.memory": "3891m", "spark.executor.memoryOverhead": "1024m"}

    tel = extract_job_telemetry(type("_B", (), {"runtime_config": _RC()})())
    assert tel["executor_memory"] == "3891m"
    assert tel["executor_memory_overhead"] == "1024m"
    # Absent when the batch left Serverless' own defaults standing — itself the answer to
    # "what sized this".
    assert extract_job_telemetry(_FakeBatch())["executor_memory"] is None
