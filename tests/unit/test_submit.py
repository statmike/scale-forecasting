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


def test_build_batch_defaults_omit_arc_b_flags() -> None:
    # Standalone submit (no models subset, header-owning) must build the exact arg list it always
    # did — no --models / --manage-header — so existing batches and callers are byte-stable.
    batch = build_batch(
        infra=_infra(),
        settings=_settings(),
        engine="explode",
        package_uri="gs://c/p.zip",
        launcher_uri="gs://c/e.py",
        config_uri="gs://c/r.json",
    )
    args = list(batch.pyspark_batch.args)
    assert "--models" not in args
    assert "--manage-header" not in args


def test_build_batch_appends_arc_b_flags_when_non_default() -> None:
    # main.run's contributor launch: restrict the executed subset + hand header ownership to main.
    batch = build_batch(
        infra=_infra(),
        settings=_settings(),
        engine="explode",
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


# --- submit_multi: one shared run_id + header (C3) ------------------------------


def _patch_multi(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub submit_multi's GCP seams (header I/O + per-child submit); capture every call.

    Returns a dict the test inspects: ``header`` (write/finalize calls), ``children`` (one entry per
    submit_batch call with the args C3 cares about).
    """
    from scale_forecasting import submit
    from scale_forecasting.registry import bq

    captured: dict[str, Any] = {"header": [], "children": []}

    monkeypatch.setattr(bq, "ensure_tables", lambda cfg, *, settings: None)
    monkeypatch.setattr(
        bq,
        "write_header",
        lambda cfg, run_id, *, settings: captured["header"].append(("write", run_id)),
    )
    monkeypatch.setattr(
        bq,
        "update_header",
        lambda run_id, *, settings, **fields: captured["header"].append(
            ("update", run_id, fields.get("status"))
        ),
    )

    def _fake_submit_batch(cfg: RunConfig, **kw: Any) -> str:
        from scale_forecasting.registry.ids import make_run_id

        captured["children"].append(
            {
                # derived from the *staged* cfg → proves the child got the full cfg
                "run_id": make_run_id(cfg),
                "models": kw.get("models"),
                "manage_header": kw.get("manage_header"),
                "batch_id": kw.get("batch_id"),
                "engine": kw.get("engine"),
                "cfg_models": list(cfg.models),
            }
        )
        return kw["batch_id"]

    monkeypatch.setattr(submit, "submit_batch", _fake_submit_batch)
    return captured


def test_submit_multi_shares_one_run_id_and_header(monkeypatch: pytest.MonkeyPatch) -> None:
    from scale_forecasting import submit
    from scale_forecasting.registry.ids import make_run_id

    captured = _patch_multi(monkeypatch)
    cfg = _cfg(models=["theta", "holtwinters", "xgboost", "lightgbm"], spark_method="multi")
    expected_run_id = make_run_id(cfg)

    ids = submit.submit_multi(cfg, settings=_settings(), infra=_infra(), wait=False)

    # two families (statistical / ml) → two children, both under the ONE run_id from the full cfg.
    assert len(captured["children"]) == 2
    assert {c["run_id"] for c in captured["children"]} == {expected_run_id}
    # each child stages the FULL cfg (all four models); models= restricts the executed subset.
    assert all(c["cfg_models"] == cfg.models for c in captured["children"])
    assert [c["models"] for c in captured["children"]] == [
        ["theta", "holtwinters"],
        ["xgboost", "lightgbm"],
    ]
    # contributor mode: submit_multi owns the header, so no child touches it.
    assert all(c["manage_header"] is False for c in captured["children"])
    # per-family batch ids are distinct (same run_id collides without the multi-<family> prefix).
    assert len(set(ids)) == 2
    assert all(bid.startswith("sf-multi-") for bid in ids)

    # exactly one header written (RUNNING) then finalized COMPLETED, both on the shared run_id.
    assert captured["header"] == [
        ("write", expected_run_id),
        ("update", expected_run_id, "COMPLETED"),
    ]


def test_submit_multi_finalizes_failed_and_reraises(monkeypatch: pytest.MonkeyPatch) -> None:
    from scale_forecasting import submit
    from scale_forecasting.errors import EngineError
    from scale_forecasting.registry.ids import make_run_id

    captured = _patch_multi(monkeypatch)

    # First family's submit raises; the header must still finalize FAILED and the error re-raise.
    def _boom(cfg: RunConfig, **kw: Any) -> str:
        raise EngineError("batch sf-multi-statistical-... terminal state FAILED")

    monkeypatch.setattr(submit, "submit_batch", _boom)
    cfg = _cfg(models=["theta", "xgboost"], spark_method="multi")
    expected_run_id = make_run_id(cfg)

    with pytest.raises(EngineError, match="FAILED"):
        submit.submit_multi(cfg, settings=_settings(), infra=_infra(), wait=False)

    assert ("write", expected_run_id) in captured["header"]
    assert ("update", expected_run_id, "FAILED") in captured["header"]


def test_submit_multi_n_series_keeps_one_run_id(monkeypatch: pytest.MonkeyPatch) -> None:
    # The scale override is applied once before hashing, so all children still share one run_id
    # (and it differs from the un-overridden id — the scale is part of the config, G3).
    from scale_forecasting import submit
    from scale_forecasting.registry.ids import make_run_id

    captured = _patch_multi(monkeypatch)
    cfg = _cfg(models=["theta", "xgboost"], spark_method="multi")
    scaled = cfg.model_copy(update={"data": cfg.data.model_copy(update={"series_limit": 1000})})
    expected_run_id = make_run_id(scaled)

    submit.submit_multi(cfg, n_series=1000, settings=_settings(), infra=_infra(), wait=False)

    assert {c["run_id"] for c in captured["children"]} == {expected_run_id}
    assert expected_run_id != make_run_id(cfg)
    assert all(c["cfg_models"] == cfg.models for c in captured["children"])


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
