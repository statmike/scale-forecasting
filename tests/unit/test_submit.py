"""Offline tests for the Dataproc submit helper (BUILD B2, ``scale_forecasting.submit``).

No network: the pure batch-spec assembly (:func:`build_batch`), the family-split that drives the
``multi`` method (:func:`split_models_by_family`), the ``n_series`` scale override, infra
resolution (:class:`BatchInfra`), and the batch-id slug are all exercised against real
``RunConfig``/``Settings`` objects. The live ``create_batch`` path is covered by the ``@gcp``
serverless smoke in B2.2.
"""

from __future__ import annotations

from typing import Any

import pytest

from scale_forecasting.config import RunConfig
from scale_forecasting.errors import ConfigError
from scale_forecasting.settings import Settings
from scale_forecasting.submit import (
    BatchInfra,
    _batch_id,
    build_batch,
    split_models_by_family,
)

# pyspark is not needed here, but google-cloud-dataproc (the [spark] extra) is — build_batch
# imports dataproc_v1. Skip the whole module cleanly when the extra is absent (parity with @spark).
pytest.importorskip("google.cloud.dataproc_v1")


def _cfg(**over: Any) -> RunConfig:
    base: dict[str, Any] = {
        "run_name": "submit test",
        "data": {"source_table": "source_series", "horizon": 28},
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


def test_batch_id_is_engine_prefixed_and_clamped() -> None:
    bid = _batch_id("run-abc123", "explode")
    assert bid.startswith("sf-explode-")
    assert len(bid) <= 63
    # a very long run_id is trimmed, never leaving a trailing hyphen (Dataproc rejects it).
    long = _batch_id("x" * 100, "naive")
    assert len(long) == 63
    assert not long.endswith("-")


# --- build_batch: the wire spec ------------------------------------------------


def test_build_batch_wires_launcher_package_and_args() -> None:
    batch = build_batch(
        infra=_infra(),
        settings=_settings(),
        engine="explode",
        package_uri="gs://code-bkt/runs/pkg-1234.zip",
        launcher_uri="gs://code-bkt/runs/spark_main.py",
        config_uri="gs://code-bkt/runs/run-abc.json",
    )
    ps = batch.pyspark_batch
    assert ps.main_python_file_uri == "gs://code-bkt/runs/spark_main.py"
    assert list(ps.python_file_uris) == ["gs://code-bkt/runs/pkg-1234.zip"]
    # engine + config-uri lead; the --sf-* infra args follow (the Dataproc delivery path).
    assert ps.args[:4] == ["--engine", "explode", "--config-uri", "gs://code-bkt/runs/run-abc.json"]
    assert "--sf-project-id" in ps.args
    assert "proj-x" in ps.args
    rc = batch.runtime_config
    assert rc.container_image == "us-docker.pkg.dev/proj-x/repo/runtime:latest"
    ec = batch.environment_config.execution_config
    assert ec.service_account == "compute@proj-x.iam.gserviceaccount.com"
    assert ec.subnetwork_uri.endswith("/subnetworks/sf")


def test_build_batch_without_max_executors_sets_no_cap() -> None:
    batch = build_batch(
        infra=_infra(),
        settings=_settings(),
        engine="explode",
        package_uri="gs://c/p.zip",
        launcher_uri="gs://c/e.py",
        config_uri="gs://c/r.json",
    )
    assert "spark.dynamicAllocation.maxExecutors" not in dict(batch.runtime_config.properties)


def test_build_batch_max_executors_caps_dynamic_allocation() -> None:
    # The naive-demo throttle: capping executors is what makes the straggler visible.
    batch = build_batch(
        infra=_infra(),
        settings=_settings(),
        engine="naive",
        package_uri="gs://c/p.zip",
        launcher_uri="gs://c/e.py",
        config_uri="gs://c/r.json",
        max_executors=2,
    )
    assert dict(batch.runtime_config.properties)["spark.dynamicAllocation.maxExecutors"] == "2"


# --- family split (the multi method) -------------------------------------------


def test_split_models_by_family_groups_and_preserves_order() -> None:
    cfg = _cfg(models=["theta", "xgboost", "holtwinters", "lightgbm"])
    families = split_models_by_family(cfg)
    assert families == {
        "statistical": ["theta", "holtwinters"],
        "ml": ["xgboost", "lightgbm"],
    }


def test_split_single_family_is_one_group() -> None:
    cfg = _cfg(models=["theta", "holtwinters"])
    assert split_models_by_family(cfg) == {"statistical": ["theta", "holtwinters"]}


def test_multi_on_cluster_engine_is_guarded() -> None:
    # multi is submit-side only; the on-cluster engine must refuse loudly, not run un-split.
    from scale_forecasting.engines import spark_multi
    from scale_forecasting.errors import EngineError

    with pytest.raises(EngineError, match="submit_multi"):
        spark_multi.run(_cfg(spark_method="multi"))


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
        def result(self) -> Any:
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
        engine="explode",
        n_series=1000,
        settings=_settings(),
        infra=_infra(),
        wait=True,
    )
    assert staged["series_limit"] == 1000  # the scale override reached the staged config
    assert staged["parent"] == "projects/proj-x/locations/us-central1"
    assert batch_id.startswith("sf-explode-")
    assert staged["batch_id"] == batch_id


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
        def result(self) -> Any:
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
            engine="explode",
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
