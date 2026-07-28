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
        launcher_uri="gs://code-bkt/runs/spark_entry.py",
        config_uri="gs://code-bkt/runs/run-abc.json",
    )
    ps = batch.pyspark_batch
    assert ps.main_python_file_uri == "gs://code-bkt/runs/spark_entry.py"
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
        return ("gs://code-bkt/runs/pkg.zip", "gs://code-bkt/runs/spark_entry.py")

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

    monkeypatch.setattr(submit, "_stage_code", _fake_stage_code)
    monkeypatch.setattr(submit, "_stage_config", _fake_stage_config)
    monkeypatch.setattr(submit, "_batch_client", lambda region: _FakeClient())

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
