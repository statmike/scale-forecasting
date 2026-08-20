"""Offline tests for the run orchestrator (``scale_forecasting.main``).

No GCP: the pure plan (:func:`main._plan`) — run_id parity, the per-runtime model split, and the
ray/multi rejections — plus the ``dry_run`` path and the CLI's dispatch of it. The live parallel
launch (Spark batch + BigQuery engine under one run_id) is the ``@gcp`` smoke in
``tests/integration/test_main_orchestration_smoke.py``; here the GCP seams are never reached because
``dry_run`` returns before them and the rejection tests raise first.
"""

from __future__ import annotations

from typing import Any

import pytest

from scale_forecasting import main
from scale_forecasting.config import RunConfig
from scale_forecasting.errors import ConfigError
from scale_forecasting.registry.ids import make_run_id
from scale_forecasting.settings import Settings

# Model names by runtime: theta is a Python/Spark model; arima_plus / timesfm are the
# BigQuery-native models (runtime == "bigquery").
_SPARK = "theta"
_NATIVE = ["arima_plus", "timesfm"]

# A resolved Settings for the dispatch tests (never used to touch GCP — the submit fns are faked).
_SETTINGS = Settings(
    project_id="proj-x",
    connection="proj-x.us-central1.conn",
    warehouse_uri="gs://bkt/warehouse",
)


def _cfg(**over: Any) -> RunConfig:
    base: dict[str, Any] = {
        "run_name": "main test",
        "data": {"source_table": "source_series_native", "horizon": 7, "series_limit": 5},
        "models": [_SPARK, *_NATIVE],
    }
    base.update(over)
    return RunConfig(**base)


@pytest.fixture(autouse=True)
def _no_live_header_check(monkeypatch: pytest.MonkeyPatch) -> None:
    # The exists-vs-new verdict queries the registry; default it to "new run" so offline plan/stage
    # tests never touch BigQuery. The idempotency tests below override this explicitly.
    from scale_forecasting.registry import bq

    monkeypatch.setattr(bq, "header_status", lambda *a, **k: None)


# --- _plan: run_id parity + the per-runtime split ------------------------------


def test_plan_run_id_matches_full_config_digest() -> None:
    # Both engines must derive the same id, so the plan's run_id is the digest over the WHOLE cfg
    # (incl. every model), not over either executed subset.
    cfg = _cfg()
    assert main._plan(cfg).run_id == make_run_id(cfg)


def test_plan_splits_models_by_runtime() -> None:
    plan = main._plan(_cfg())
    assert plan.python_models == [_SPARK]
    assert plan.bq_models == _NATIVE
    assert plan.spark_method == "explode"  # normalized default


def test_plan_all_bigquery_has_no_python_models() -> None:
    plan = main._plan(_cfg(models=_NATIVE))
    assert plan.python_models == []
    assert plan.bq_models == _NATIVE


def test_plan_all_python_has_no_bq_models() -> None:
    plan = main._plan(_cfg(models=[_SPARK, "holtwinters"]))
    assert plan.python_models == [_SPARK, "holtwinters"]
    assert plan.bq_models == []


# --- _plan: ray is accepted; the out-of-scope multi shape is rejected ----------


def test_plan_accepts_ray_when_python_models_present() -> None:
    # The Ray engine is built, so main.run now dispatches ray — _plan must NOT reject it.
    plan = main._plan(_cfg(models=[_SPARK, *_NATIVE], python_runtime="ray"))
    assert plan.python_models == [_SPARK]
    assert plan.bq_models == _NATIVE


def test_plan_rejects_multi_when_python_models_present() -> None:
    with pytest.raises(ConfigError, match="submit --engine multi"):
        main._plan(_cfg(models=[_SPARK, *_NATIVE], spark_method="multi"))


def test_ray_runtime_cannot_carry_spark_method() -> None:
    # multi is a Spark-only method, so main._plan's multi guard is gated on python_runtime="spark".
    # It never has to fire for a ray config because the config layer forbids ray + any spark_method
    # outright — so a ray config that names one fails to construct, well before _plan sees it.
    with pytest.raises(ValueError, match="spark_method is only valid"):
        _cfg(models=[_SPARK], python_runtime="ray", spark_method="multi")


def test_plan_allows_ray_config_when_only_bigquery_models() -> None:
    # An all-native config never uses the Python runtime, so runtime choice doesn't apply.
    plan = main._plan(_cfg(models=_NATIVE, python_runtime="ray"))
    assert plan.python_models == []
    assert plan.bq_models == _NATIVE


# --- run(dry_run=True): offline, no GCP ----------------------------------------


def test_dry_run_returns_run_id_and_estimates_fanout(monkeypatch: pytest.MonkeyPatch) -> None:
    # dry_run must return the shared run_id and call estimate_fanout, without touching any GCP seam.
    import scale_forecasting.config as config_mod

    called: dict[str, Any] = {}
    real_estimate = config_mod.estimate_fanout

    def _spy(cfg: RunConfig) -> Any:
        called["cfg"] = cfg
        return real_estimate(cfg)

    monkeypatch.setattr(config_mod, "estimate_fanout", _spy)

    cfg = _cfg()
    run_id = main.run(cfg, dry_run=True)
    assert run_id == make_run_id(cfg)
    assert called["cfg"] is cfg


def test_plan_run_without_env_returns_plan_but_no_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No SF_* identity → command emission is skipped, but the id/fanout/split still resolve offline.
    import scale_forecasting.settings as settings_mod

    def _no_env() -> Settings:
        raise ConfigError("no SF_* env")

    monkeypatch.setattr(settings_mod.Settings, "resolve", staticmethod(_no_env))

    result = main.plan_run(_cfg())
    assert result.run_id == make_run_id(_cfg())
    assert result.staged is False
    assert result.commands is None
    assert result.config_uri is None
    assert result.fanout.n_cells > 0


def _batch_infra() -> Any:
    from scale_forecasting.submit import BatchInfra

    return BatchInfra(
        code_bucket="bkt-code",
        container_image="us-docker.pkg.dev/proj-x/sf/runtime:tag",
        compute_sa="sf-compute@proj-x.iam.gserviceaccount.com",
        subnetwork_uri="projects/proj-x/regions/us-central1/subnetworks/sf",
    )


def test_plan_run_spark_emits_main_and_spark_command_templates() -> None:
    result = main.plan_run(_cfg(models=[_SPARK]), settings=_SETTINGS, infra=_batch_infra())
    assert result.commands is not None
    assert set(result.commands) == {"main", "spark"}
    config_uri = f"gs://bkt-code/runs/{result.run_id}.json"
    assert result.config_uri == config_uri
    # main = the orchestrator command; spark = native gcloud + universal, both referencing the URI.
    assert result.commands["main"].universal.endswith(config_uri)
    assert result.commands["main"].native is None
    spark = result.commands["spark"]
    assert spark.native is not None and "gcloud" in spark.native and config_uri in spark.native
    # A Python-only config emits no --models (the standalone batch runs the whole config).
    assert "--models" not in spark.native


def test_plan_run_mixed_spark_restricts_the_spark_subset() -> None:
    result = main.plan_run(
        _cfg(models=[_SPARK, *_NATIVE]), settings=_SETTINGS, infra=_batch_infra()
    )
    assert result.commands is not None
    spark = result.commands["spark"]
    # A mixed run restricts the Spark batch to just its Python model(s) via --models.
    assert spark.native is not None and "--models" in spark.native and _SPARK in spark.native


def test_plan_run_ray_emits_universal_only_ray_command() -> None:
    from scale_forecasting.ray_submit import RayInfra

    infra = RayInfra(compute_sa="sf-compute@proj-x.iam", code_bucket="bkt-code")
    result = main.plan_run(
        _cfg(models=[_SPARK, *_NATIVE], python_runtime="ray"), settings=_SETTINGS, infra=infra
    )
    assert result.commands is not None
    assert set(result.commands) == {"main", "ray"}
    ray = result.commands["ray"]
    assert ray.native is None  # no gcloud verb submits a Ray job
    assert "ray_submit" in ray.universal and result.run_id in ray.universal


def test_plan_run_reports_new_run_when_config_never_ran() -> None:
    # The autouse fixture makes header_status return None → the config has not run before.
    result = main.plan_run(_cfg(models=[_SPARK]), settings=_SETTINGS, infra=_batch_infra())
    assert result.idempotency.checked is True
    assert result.idempotency.exists is False
    assert result.idempotency.prior_status is None


def test_plan_run_reports_existing_run_when_config_already_ran(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scale_forecasting.registry import bq

    monkeypatch.setattr(bq, "header_status", lambda *a, **k: "COMPLETED")
    result = main.plan_run(_cfg(models=[_SPARK]), settings=_SETTINGS, infra=_batch_infra())
    assert result.idempotency.checked is True
    assert result.idempotency.exists is True
    assert result.idempotency.prior_status == "COMPLETED"


def test_plan_run_verdict_unknown_when_registry_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A registry read failure (no table yet / unreachable) degrades to an unknown verdict, never
    # fatal — the plan still returns.
    from scale_forecasting.errors import RegistryError
    from scale_forecasting.registry import bq

    def _boom(*a: Any, **k: Any) -> str:
        raise RegistryError("no such table")

    monkeypatch.setattr(bq, "header_status", _boom)
    result = main.plan_run(_cfg(models=[_SPARK]), settings=_SETTINGS, infra=_batch_infra())
    assert result.idempotency.checked is False
    assert result.idempotency.exists is False


def test_plan_run_without_env_leaves_verdict_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    # No SF_* env → settings never resolve, so the registry is never consulted: verdict unknown.
    import scale_forecasting.settings as settings_mod

    def _no_env() -> Settings:
        raise ConfigError("no SF_* env")

    monkeypatch.setattr(settings_mod.Settings, "resolve", staticmethod(_no_env))
    result = main.plan_run(_cfg())
    assert result.idempotency.checked is False


def test_emit_idempotency_warns_on_existing_and_notes_force(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from scale_forecasting.registry import bq

    monkeypatch.setattr(bq, "header_status", lambda *a, **k: "COMPLETED")

    with caplog.at_level("WARNING"):
        main.plan_run(_cfg(models=[_SPARK]), settings=_SETTINGS, infra=_batch_infra())
    assert any("already ran" in r.message and "--force" in r.message for r in caplog.records)

    caplog.clear()
    with caplog.at_level("INFO"):
        main.plan_run(_cfg(models=[_SPARK]), settings=_SETTINGS, infra=_batch_infra(), force=True)
    assert any("re-run (--force)" in r.message for r in caplog.records)


def test_stage_run_spark_uploads_and_builds_runnable_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scale_forecasting.staging as staging_mod
    import scale_forecasting.submit as submit_mod

    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        staging_mod, "stage_config", lambda cfg, rid, bkt: f"gs://{bkt}/runs/{rid}.json"
    )
    monkeypatch.setattr(
        submit_mod,
        "_stage_code",
        lambda infra: (
            f"gs://{infra.code_bucket}/runs/pkg.zip",
            f"gs://{infra.code_bucket}/runs/spark_main.py",
        ),
    )

    def _fake_manifest(manifest: dict[str, Any], rid: str, bkt: str) -> str:
        captured["manifest"] = manifest
        return f"gs://{bkt}/runs/{rid}.plan.json"

    monkeypatch.setattr(staging_mod, "stage_manifest", _fake_manifest)

    result = main.stage_run(_cfg(models=[_SPARK]), settings=_SETTINGS, infra=_batch_infra())
    assert result.staged is True
    assert result.config_uri == f"gs://bkt-code/runs/{result.run_id}.json"
    assert result.commands is not None
    spark = result.commands["spark"]
    assert spark.native is not None and "gs://bkt-code/runs/pkg.zip" in spark.native

    manifest = captured["manifest"]
    assert manifest["run_id"] == result.run_id
    assert manifest["config_uri"] == result.config_uri
    assert set(manifest["commands"]) == {"main", "spark"}
    assert manifest["fanout"]["n_cells"] == result.fanout.n_cells
    assert "created_at" in manifest  # caller-stamped timestamp
    # The manifest records the re-run intent and the exists-vs-new verdict it was staged under.
    assert manifest["force"] is False
    assert manifest["idempotency"]["checked"] is True
    assert manifest["idempotency"]["exists"] is False


def test_stage_run_requires_infra(monkeypatch: pytest.MonkeyPatch) -> None:
    # stage_run touches GCS, so a missing SF_* identity raises rather than degrading (unlike plan).
    import scale_forecasting.settings as settings_mod

    def _no_env() -> Settings:
        raise ConfigError("no SF_* env")

    monkeypatch.setattr(settings_mod.Settings, "resolve", staticmethod(_no_env))
    with pytest.raises(ConfigError):
        main.stage_run(_cfg())


def test_cli_dispatches_stage_only(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    from types import SimpleNamespace

    seen: dict[str, Any] = {}

    def _fake_stage(cfg: RunConfig, *, force: bool = False) -> Any:
        seen["run_name"] = cfg.run_name
        seen["force"] = force
        return SimpleNamespace(run_id="rid-staged")

    monkeypatch.setattr(main, "stage_run", _fake_stage)

    path = tmp_path / "run.json"
    path.write_text(
        json.dumps(
            {
                "run_name": "cli stage test",
                "data": {"source_table": "source_series_native", "horizon": 7},
                "models": [_SPARK],
            }
        )
    )
    main._main(["--config", str(path), "--stage-only"])
    assert seen == {"run_name": "cli stage test", "force": False}


def test_dry_run_still_rejects_multi() -> None:
    # The plan (and its rejection) runs before the dry_run short-circuit, so bad shapes fail fast.
    with pytest.raises(ConfigError, match="submit --engine multi"):
        main.run(_cfg(models=[_SPARK, *_NATIVE], spark_method="multi"), dry_run=True)


def test_dry_run_allows_ray() -> None:
    # Ray is a supported runtime now; a ray config plans + dry-runs like any other.
    run_id = main.run(_cfg(models=[_SPARK, *_NATIVE], python_runtime="ray"), dry_run=True)
    assert run_id == make_run_id(_cfg(models=[_SPARK, *_NATIVE], python_runtime="ray"))


# --- _launch_python_runtime: dispatch by python_runtime ------------------------


def test_launch_python_runtime_dispatches_spark(monkeypatch: pytest.MonkeyPatch) -> None:
    import scale_forecasting.submit as submit_mod

    seen: dict[str, Any] = {}

    def _fake_submit_batch(cfg: RunConfig, **kw: Any) -> str:
        seen.update(kw)
        seen["cfg"] = cfg
        return "batch-1"

    monkeypatch.setattr(submit_mod, "submit_batch", _fake_submit_batch)

    cfg = _cfg(models=[_SPARK], python_runtime="spark")
    plan = main._plan(cfg)
    main._launch_python_runtime(cfg, plan, _SETTINGS)
    assert seen["engine"] == "explode"
    assert seen["models"] == [_SPARK]
    assert seen["manage_header"] is False


def test_run_n_series_override_changes_run_id() -> None:
    # n_series overrides series_limit before planning, so the dry-run id matches the adjusted cfg.
    from scale_forecasting.registry.ids import make_run_id as _mri

    base = _cfg(models=[_SPARK])
    overridden = base.with_series_limit(1000)
    run_id = main.run(base, dry_run=True, n_series=1000)
    assert run_id == _mri(overridden)
    assert run_id != _mri(base)


def test_launch_python_runtime_threads_max_executors(monkeypatch: pytest.MonkeyPatch) -> None:
    import scale_forecasting.submit as submit_mod

    seen: dict[str, Any] = {}

    def _fake_submit_batch(cfg: RunConfig, **kw: Any) -> str:
        seen.update(kw)
        return "batch-1"

    monkeypatch.setattr(submit_mod, "submit_batch", _fake_submit_batch)

    cfg = _cfg(models=[_SPARK], python_runtime="spark")
    plan = main._plan(cfg)
    main._launch_python_runtime(cfg, plan, _SETTINGS, max_executors=8)
    assert seen["max_executors"] == 8


def test_launch_python_runtime_dispatches_ray(monkeypatch: pytest.MonkeyPatch) -> None:
    import scale_forecasting.ray_submit as ray_submit_mod

    seen: dict[str, Any] = {}

    def _fake_submit_ray(cfg: RunConfig, **kw: Any) -> str:
        seen.update(kw)
        seen["cfg"] = cfg
        return "job-1"

    monkeypatch.setattr(ray_submit_mod, "submit_ray", _fake_submit_ray)

    cfg = _cfg(models=[_SPARK], python_runtime="ray")
    plan = main._plan(cfg)
    main._launch_python_runtime(cfg, plan, _SETTINGS)
    assert "engine" not in seen  # ray takes no spark engine arg
    assert seen["models"] == [_SPARK]
    assert seen["manage_header"] is False


# --- run(): ensemble orchestration after the engine join -----------------------


def _patch_run_seams(
    monkeypatch: pytest.MonkeyPatch, *, bq_error: Exception | None = None
) -> dict[str, Any]:
    """Fake every GCP seam main.run touches so the ensemble gating is exercised offline.

    Records what happened in the returned dict: header status finalized, whether the BigQuery engine
    and the Spark launch ran. The Python-runtime launch is faked to a no-op success. ``bq_error``
    makes the BigQuery engine raise, to prove ensembles are skipped when an engine fails.
    """
    import scale_forecasting.ensemble_run as ensemble_mod
    from scale_forecasting.engines import bigquery_engine
    from scale_forecasting.registry import bq

    seen: dict[str, Any] = {"ensemble_called": False}

    monkeypatch.setattr(Settings, "resolve", classmethod(lambda cls: _SETTINGS))
    monkeypatch.setattr(bq, "ensure_tables", lambda *a, **k: None)
    monkeypatch.setattr(bq, "write_header", lambda *a, **k: None)

    def _fake_update(run_id: str, *, settings: Any = None, **fields: Any) -> None:
        seen["status"] = fields.get("status")

    monkeypatch.setattr(bq, "update_header", _fake_update)

    def _fake_bq_run(cfg: RunConfig, models: list[str], **kw: Any) -> Any:
        seen["bq_ran"] = True
        if bq_error is not None:
            raise bq_error
        return bigquery_engine.BqOutcome(status="COMPLETED", n_series=3, models=models)

    monkeypatch.setattr(bigquery_engine, "run", _fake_bq_run)
    monkeypatch.setattr(
        main, "_launch_python_runtime", lambda *a, **k: seen.__setitem__("spark_ran", True)
    )

    def _fake_ensembles(cfg: RunConfig, run_id: str, *, settings: Any) -> None:
        seen["ensemble_called"] = True
        seen["ensemble_run_id"] = run_id

    monkeypatch.setattr(ensemble_mod, "run_ensembles", _fake_ensembles)
    return seen


def test_run_invokes_ensembles_when_enabled_and_engines_ok(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _patch_run_seams(monkeypatch)
    cfg = _cfg(ensemble={"enabled": True, "strategies": ["mean"]})
    run_id = main.run(cfg)
    assert seen["ensemble_called"] is True
    assert seen["ensemble_run_id"] == run_id
    assert seen["status"] == "COMPLETED"


def test_run_skips_ensembles_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _patch_run_seams(monkeypatch)
    main.run(_cfg())  # ensemble.enabled defaults False
    assert seen["ensemble_called"] is False
    assert seen["status"] == "COMPLETED"


def test_run_skips_ensembles_when_engine_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    # A BigQuery engine failure must skip ensembles (they'd read incomplete predictions). Here the
    # Python track succeeded and BigQuery failed — a mixed outcome — so the shared header finalizes
    # PARTIAL, and the first failure is still re-raised for a non-zero exit.
    seen = _patch_run_seams(monkeypatch, bq_error=RuntimeError("bq boom"))
    cfg = _cfg(ensemble={"enabled": True, "strategies": ["mean"]})
    with pytest.raises(RuntimeError, match="bq boom"):
        main.run(cfg)
    assert seen["ensemble_called"] is False
    assert seen["status"] == "PARTIAL"


def test_run_ensemble_failure_finalizes_header_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    import scale_forecasting.ensemble_run as ensemble_mod

    seen = _patch_run_seams(monkeypatch)

    def _boom(cfg: RunConfig, run_id: str, *, settings: Any) -> None:
        seen["ensemble_called"] = True
        raise RuntimeError("ensemble boom")

    monkeypatch.setattr(ensemble_mod, "run_ensembles", _boom)
    cfg = _cfg(ensemble={"enabled": True, "strategies": ["mean"]})
    with pytest.raises(RuntimeError, match="ensemble boom"):
        main.run(cfg)
    assert seen["ensemble_called"] is True
    assert seen["status"] == "FAILED"


# --- _combined_status: the multi-engine roll-up (pure) -------------------------


def _boom() -> RuntimeError:
    return RuntimeError("x")


def test_combined_status_all_engines_green_is_completed() -> None:
    plan = main._plan(_cfg())  # both python + bq tracks
    assert main._combined_status(plan, None, None, None) == "COMPLETED"


def test_combined_status_mixed_is_partial() -> None:
    plan = main._plan(_cfg())
    # python green, bq failed → some but not all → PARTIAL (and the mirror case).
    assert main._combined_status(plan, None, _boom(), None) == "PARTIAL"
    assert main._combined_status(plan, _boom(), None, None) == "PARTIAL"


def test_combined_status_all_engines_failed_is_failed() -> None:
    plan = main._plan(_cfg())
    assert main._combined_status(plan, _boom(), _boom(), None) == "FAILED"


def test_combined_status_single_engine_has_no_partial() -> None:
    bq_only = main._plan(_cfg(models=_NATIVE))
    assert main._combined_status(bq_only, None, None, None) == "COMPLETED"
    assert main._combined_status(bq_only, None, _boom(), None) == "FAILED"


def test_combined_status_ensemble_failure_fails_a_green_run() -> None:
    plan = main._plan(_cfg())
    # engines green but the ensemble step failed → the run didn't deliver full output → FAILED.
    assert main._combined_status(plan, None, None, _boom()) == "FAILED"
    # an ensemble error never masks an engine PARTIAL/FAILED (engine status already non-COMPLETED).
    assert main._combined_status(plan, None, _boom(), _boom()) == "PARTIAL"


# --- CLI: dispatches dry_run ---------------------------------------------------


def test_cli_dispatches_dry_run(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    seen: dict[str, Any] = {}

    def _fake_run(cfg: RunConfig, *, dry_run: bool = False, force: bool = False) -> str:
        seen["dry_run"] = dry_run
        seen["force"] = force
        seen["run_name"] = cfg.run_name
        return "rid-123"

    monkeypatch.setattr(main, "run", _fake_run)

    path = tmp_path / "run.json"
    path.write_text(
        json.dumps(
            {
                "run_name": "cli main test",
                "data": {"source_table": "source_series_native", "horizon": 7},
                "models": [_SPARK],
            }
        )
    )
    main._main(["--config", str(path), "--dry-run"])
    assert seen == {"dry_run": True, "force": False, "run_name": "cli main test"}


def test_cli_force_flag_threads_through(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    seen: dict[str, Any] = {}

    def _fake_run(cfg: RunConfig, *, dry_run: bool = False, force: bool = False) -> str:
        seen["force"] = force
        return "rid-123"

    monkeypatch.setattr(main, "run", _fake_run)

    path = tmp_path / "run.json"
    path.write_text(
        json.dumps(
            {
                "run_name": "cli force test",
                "data": {"source_table": "source_series_native", "horizon": 7},
                "models": [_SPARK],
            }
        )
    )
    main._main(["--config", str(path), "--dry-run", "--force"])
    assert seen["force"] is True
