"""Offline tests for the run orchestrator (``scale_forecasting.main``).

No GCP: the pure plan (:func:`main._plan`) — run_id parity and the per-runtime model split — plus
the ``dry_run`` path and the CLI's dispatch of it. The live parallel
launch (Spark batch + BigQuery engine under one run_id) is the ``@gcp`` smoke in
``tests/integration/test_main_orchestration_smoke.py``; here the GCP seams are never reached because
``dry_run`` returns before them.
"""

from __future__ import annotations

from typing import Any

import pytest

from scale_forecasting import dag, job_launch, main
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
    from scale_forecasting.registry import header

    monkeypatch.setattr(header, "header_status", lambda *a, **k: None)


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


def test_plan_all_bigquery_has_no_python_models() -> None:
    plan = main._plan(_cfg(models=_NATIVE))
    assert plan.python_models == []
    assert plan.bq_models == _NATIVE


def test_plan_all_python_has_no_bq_models() -> None:
    plan = main._plan(_cfg(models=[_SPARK, "holtwinters"]))
    assert plan.python_models == [_SPARK, "holtwinters"]
    assert plan.bq_models == []


# --- _plan: ray is accepted ----------------------------------------------------


def test_plan_accepts_ray_when_python_models_present() -> None:
    # The Ray engine is built, so main.run now dispatches ray — _plan must NOT reject it.
    plan = main._plan(_cfg(models=[_SPARK, *_NATIVE], python_runtime="ray"))
    assert plan.python_models == [_SPARK]
    assert plan.bq_models == _NATIVE


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
    from scale_forecasting.batch_infra import BatchInfra

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
    from scale_forecasting.ray_infra import RayInfra

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
    from scale_forecasting.registry import header

    monkeypatch.setattr(header, "header_status", lambda *a, **k: "COMPLETED")
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
    from scale_forecasting.registry import header

    def _boom(*a: Any, **k: Any) -> str:
        raise RegistryError("no such table")

    monkeypatch.setattr(header, "header_status", _boom)
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
    from scale_forecasting.registry import header

    monkeypatch.setattr(header, "header_status", lambda *a, **k: "COMPLETED")

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

    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        staging_mod, "stage_config", lambda cfg, rid, bkt: f"gs://{bkt}/runs/{rid}.json"
    )
    monkeypatch.setattr(
        staging_mod,
        "stage_code",
        lambda bkt: (f"gs://{bkt}/runs/pkg.zip", f"gs://{bkt}/runs/spark_main.py"),
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
    # The manifest is a DAG manifest: one node per family job carrying its deterministic job_key.
    dag = manifest["dag"]
    assert [n["family"] for n in dag] == ["statistical"]  # a pure-Spark config → one family
    assert dag[0]["runtime"] == "spark"
    assert dag[0]["job_key"].endswith("-statistical-a1")
    assert dag[0]["depends_on"] == []


def test_plan_run_resolves_dag_nodes_offline() -> None:
    # A mixed config plans one node per family + the ensemble node depending on them all.
    from scale_forecasting.registry.ids import make_job_key

    result = main.plan_run(
        _cfg(
            models=[_SPARK, "arima_plus"],
            backtest={"enabled": True},
            ensemble={"enabled": True, "strategies": ["mean", "median"]},
        ),
        settings=_SETTINGS,
        infra=_batch_infra(),
    )
    families = [n.family for n in result.nodes]
    assert families == ["statistical", "native", "ensemble"]
    ensemble = result.nodes[-1]
    assert ensemble.job_key == make_job_key(result.run_id, "ensemble", 1)
    assert set(ensemble.depends_on) == {n.job_key for n in result.nodes if n.family != "ensemble"}


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


def test_dry_run_allows_ray() -> None:
    # Ray is a supported runtime now; a ray config plans + dry-runs like any other.
    run_id = main.run(_cfg(models=[_SPARK, *_NATIVE], python_runtime="ray"), dry_run=True)
    assert run_id == make_run_id(_cfg(models=[_SPARK, *_NATIVE], python_runtime="ray"))


def test_run_n_series_override_changes_run_id() -> None:
    # n_series overrides series_limit before planning, so the dry-run id matches the adjusted cfg.
    from scale_forecasting.registry.ids import make_run_id as _mri

    base = _cfg(models=[_SPARK])
    overridden = base.with_series_limit(1000)
    run_id = main.run(base, dry_run=True, n_series=1000)
    assert run_id == _mri(overridden)
    assert run_id != _mri(base)


# --- run(): ensemble orchestration after the engine join -----------------------


def _patch_run_seams(
    monkeypatch: pytest.MonkeyPatch, *, bq_error: Exception | None = None
) -> dict[str, Any]:
    """Fake every GCP seam main.run touches so the ensemble gating is exercised offline.

    Records what happened in the returned dict: the header status finalized, whether the native
    (BigQuery) family and the Python family jobs ran. Each family launch is faked at the
    `job_launch.launch_family_job` / `job_launch.launch_native_job` seam to a no-op success (so the
    per-job run_jobs lifecycle and submitters are never reached offline). ``bq_error`` makes the
    native job
    raise, to prove ensembles are skipped when a family fails.
    """
    import scale_forecasting.ensemble_run as ensemble_mod
    from scale_forecasting.engines import bigquery_engine
    from scale_forecasting.registry import header, jobs, lifecycle, tables

    seen: dict[str, Any] = {"ensemble_called": False}

    monkeypatch.setattr(Settings, "resolve", classmethod(lambda cls: _SETTINGS))
    monkeypatch.setattr(tables, "ensure_tables", lambda *a, **k: None)
    monkeypatch.setattr(header, "write_header", lambda *a, **k: None)
    # Default: this config has not run before, so the idempotency guard falls through to launch.
    # A test overrides this to "COMPLETED" to exercise the no-op re-run guard.
    monkeypatch.setattr(header, "header_status", lambda *a, **k: None)

    def _fake_update(run_id: str, *, settings: Any = None, **fields: Any) -> None:
        seen["status"] = fields.get("status")

    monkeypatch.setattr(header, "update_header", _fake_update)

    def _fake_native(cfg: RunConfig, job: Any, run_id: str, settings: Any, *, force: bool = False):
        seen["bq_ran"] = True
        if bq_error is not None:
            raise bq_error
        return bigquery_engine.BqOutcome(status="COMPLETED", n_series=3, models=list(job.models))

    monkeypatch.setattr(job_launch, "launch_native_job", _fake_native)
    monkeypatch.setattr(
        job_launch, "launch_family_job", lambda *a, **k: seen.__setitem__("spark_ran", True)
    )

    # The ensemble DAG node opens its own run_jobs row before running the consensus; fake that
    # per-job lifecycle so launch_ensemble_job runs for real down to the run_ensembles call.
    import contextlib

    monkeypatch.setattr(jobs, "next_job_attempt", lambda *a, **k: (1, None))

    @contextlib.contextmanager
    def _fake_run_job(*a: Any, **k: Any) -> Any:
        yield None

    monkeypatch.setattr(lifecycle, "run_job", _fake_run_job)

    def _fake_ensembles(cfg: RunConfig, run_id: str, *, settings: Any, **_: Any) -> None:
        seen["ensemble_called"] = True
        seen["ensemble_run_id"] = run_id
        seen["ensemble_mode"] = "barrier"

    monkeypatch.setattr(ensemble_mod, "run_ensembles", _fake_ensembles)

    def _fake_microbatch(cfg: RunConfig, run_id: str, *, settings: Any, **kw: Any) -> None:
        seen["ensemble_called"] = True
        seen["ensemble_run_id"] = run_id
        seen["ensemble_mode"] = "microbatch"
        seen["poll_interval_s"] = kw.get("poll_interval_s")
        # The concurrent trigger passes a live upstream_done predicate. By the time main.run
        # joins this future the base jobs have finished, so it must report done.
        seen["upstream_done_at_join"] = bool(kw.get("upstream_done") and kw["upstream_done"]())

    monkeypatch.setattr(ensemble_mod, "run_ensembles_microbatch", _fake_microbatch)
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


def test_run_microbatch_ensemble_runs_concurrently_with_configured_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # mode="microbatch" routes to run_ensembles_microbatch (not the barrier run_ensembles),
    # threads the config's poll interval, and hands it a live upstream_done predicate that reads
    # done once the base jobs have joined.
    seen = _patch_run_seams(monkeypatch)
    cfg = _cfg(
        ensemble={"enabled": True, "strategies": ["mean"]},
        compute={"ensemble": {"mode": "microbatch", "microbatch_interval_s": 12.5}},
    )
    run_id = main.run(cfg)
    assert seen["ensemble_called"] is True
    assert seen["ensemble_run_id"] == run_id
    assert seen["ensemble_mode"] == "microbatch"
    assert seen["poll_interval_s"] == 12.5
    assert seen["upstream_done_at_join"] is True
    assert seen["status"] == "COMPLETED"


def test_run_barrier_ensemble_uses_barrier_trigger(monkeypatch: pytest.MonkeyPatch) -> None:
    # The default (mode="barrier") stays the post-join single-pass run_ensembles call.
    seen = _patch_run_seams(monkeypatch)
    cfg = _cfg(ensemble={"enabled": True, "strategies": ["mean"]})
    main.run(cfg)
    assert seen["ensemble_mode"] == "barrier"


def test_run_microbatch_ensemble_still_launches_when_a_family_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Unlike the barrier's up-front skip, the concurrent microbatch node is already running when a
    # base family fails; it stays launched but (live) drains nothing because the failed family's
    # models never satisfy the full-base-set readiness bar. The run still finalizes PARTIAL and
    # re-raises the family error.
    seen = _patch_run_seams(monkeypatch, bq_error=RuntimeError("bq boom"))
    cfg = _cfg(
        ensemble={"enabled": True, "strategies": ["mean"]},
        compute={"ensemble": {"mode": "microbatch"}},
    )
    with pytest.raises(RuntimeError, match="bq boom"):
        main.run(cfg)
    assert seen["ensemble_called"] is True
    assert seen["ensemble_mode"] == "microbatch"
    assert seen["status"] == "PARTIAL"


def test_run_noops_when_config_already_completed(monkeypatch: pytest.MonkeyPatch) -> None:
    # A plain re-run of an already-COMPLETED config is a no-op: it must NOT relaunch any family
    # (relaunching resubmits the deterministic per-family job id and collides), and returns the
    # config-pinned run_id unchanged.
    from scale_forecasting.registry import header

    seen = _patch_run_seams(monkeypatch)
    monkeypatch.setattr(header, "header_status", lambda *a, **k: "COMPLETED")
    cfg = _cfg(ensemble={"enabled": True, "strategies": ["mean"]})
    run_id = main.run(cfg)
    assert run_id == dag.plan_dag(cfg).run_id
    assert "spark_ran" not in seen
    assert "bq_ran" not in seen
    assert seen["ensemble_called"] is False
    assert "status" not in seen  # header never re-finalized


def test_run_force_reexecutes_even_when_completed(monkeypatch: pytest.MonkeyPatch) -> None:
    # force=True re-executes a COMPLETED config (a fresh, distinctly-keyed attempt) — the guard
    # applies only to unforced re-runs.
    from scale_forecasting.registry import header

    seen = _patch_run_seams(monkeypatch)
    monkeypatch.setattr(header, "header_status", lambda *a, **k: "COMPLETED")
    main.run(_cfg(), force=True)
    assert seen.get("spark_ran") is True
    assert seen["status"] == "COMPLETED"


def test_run_reexecutes_when_prior_run_not_completed(monkeypatch: pytest.MonkeyPatch) -> None:
    # A prior FAILED run is not COMPLETED, so an unforced re-run falls through and launches (a retry
    # path), rather than no-op'ing.
    from scale_forecasting.registry import header

    seen = _patch_run_seams(monkeypatch)
    monkeypatch.setattr(header, "header_status", lambda *a, **k: "FAILED")
    main.run(_cfg())
    assert seen.get("spark_ran") is True


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

    def _boom(cfg: RunConfig, run_id: str, *, settings: Any, **_: Any) -> None:
        seen["ensemble_called"] = True
        raise RuntimeError("ensemble boom")

    monkeypatch.setattr(ensemble_mod, "run_ensembles", _boom)
    cfg = _cfg(ensemble={"enabled": True, "strategies": ["mean"]})
    with pytest.raises(RuntimeError, match="ensemble boom"):
        main.run(cfg)
    assert seen["ensemble_called"] is True
    assert seen["status"] == "FAILED"


# --- _combined_status: the per-family roll-up (pure) ---------------------------


def _boom() -> RuntimeError:
    return RuntimeError("x")


def test_combined_status_all_jobs_green_is_completed() -> None:
    d = dag.plan_dag(_cfg())  # statistical + native jobs
    assert main._combined_status(d, {}, None) == "COMPLETED"


def test_combined_status_mixed_is_partial() -> None:
    d = dag.plan_dag(_cfg())
    # one family failed, the other green → some but not all → PARTIAL (and the mirror case).
    assert main._combined_status(d, {"native": _boom()}, None) == "PARTIAL"
    assert main._combined_status(d, {"statistical": _boom()}, None) == "PARTIAL"


def test_combined_status_all_jobs_failed_is_failed() -> None:
    d = dag.plan_dag(_cfg())
    assert main._combined_status(d, {"statistical": _boom(), "native": _boom()}, None) == "FAILED"


def test_combined_status_single_job_has_no_partial() -> None:
    bq_only = dag.plan_dag(_cfg(models=_NATIVE))  # native family only
    assert main._combined_status(bq_only, {}, None) == "COMPLETED"
    assert main._combined_status(bq_only, {"native": _boom()}, None) == "FAILED"


def test_combined_status_ensemble_failure_fails_a_green_run() -> None:
    d = dag.plan_dag(_cfg())
    # families green but the ensemble step failed → the run didn't deliver full output → FAILED.
    assert main._combined_status(d, {}, _boom()) == "FAILED"
    # an ensemble error never masks a family PARTIAL/FAILED (status already non-COMPLETED).
    assert main._combined_status(d, {"native": _boom()}, _boom()) == "PARTIAL"


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


def test_cli_dispatches_probe(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    import scale_forecasting.probes.reconcile as probes_mod
    from scale_forecasting.config import load_config_uri

    seen: dict[str, Any] = {}

    def _fake_probe_run(run_id: str, *, job: str | None = None, settings: Any = None) -> str:
        seen["run_id"] = run_id
        seen["job"] = job
        return "REPORT"

    monkeypatch.setattr(probes_mod, "probe_run", _fake_probe_run)
    printed: dict[str, Any] = {}
    monkeypatch.setattr(main, "_print_probe_report", lambda r: printed.__setitem__("report", r))

    path = tmp_path / "run.json"
    path.write_text(
        json.dumps(
            {
                "run_name": "cli probe test",
                "data": {"source_table": "source_series_native", "horizon": 7},
                "models": [_SPARK],
            }
        )
    )
    # --probe resolves the config's run_id offline, escalates via probe_run, and prints — no run().
    main._main(["--config", str(path), "--probe", "--job", "statistical"])

    assert seen["run_id"] == make_run_id(load_config_uri(str(path)))
    assert seen["job"] == "statistical"
    assert printed["report"] == "REPORT"


def test_print_probe_report_formats_header_and_rows(capsys: pytest.CaptureFixture[str]) -> None:
    from scale_forecasting.probes.reconcile import FamilyVerdict, ProbeReport
    from scale_forecasting.probes.vocabulary import VERDICT_RUNNING, VERDICT_STALE_REGISTRY

    report = ProbeReport(
        run_id="sf-run-xyz",
        status="RUNNING",
        escalated=True,
        families=(
            FamilyVerdict(
                family="statistical", runtime="spark", registry_status="RUNNING",
                native_state="RUNNING", exists=True, verdict=VERDICT_RUNNING,
                disagreement=False, n_done=3, n_expected=10, detail="in flight",
            ),
            FamilyVerdict(
                family="native", runtime="bigquery", registry_status="RUNNING",
                native_state="SUCCEEDED", exists=True, verdict=VERDICT_STALE_REGISTRY,
                disagreement=True, n_done=5, n_expected=None, detail="",
            ),
        ),
        disagreement=True,
    )
    main._print_probe_report(report)
    text = capsys.readouterr().out
    # Header line carries the run-wide roll-up.
    assert "run sf-run-xyz" in text
    assert "status=RUNNING" in text and "escalated=True" in text and "disagreement=True" in text
    # One row per family with its verdict + done/expected (unknown denominator renders as ?).
    assert "statistical" in text and "RUNNING_CONFIRMED" in text and "3/10" in text
    assert "native" in text and "STALE_REGISTRY" in text and "5/?" in text


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


def test_cli_accepts_config_uri(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--config-uri`` (what the emitted portable "main" command uses) loads a config too.

    ``load_config_uri`` treats a non-``gs://`` value as a local path, so this exercises the flag
    wiring offline without touching GCS.
    """
    import json

    seen: dict[str, Any] = {}

    def _fake_run(cfg: RunConfig, *, dry_run: bool = False, force: bool = False) -> str:
        seen["run_name"] = cfg.run_name
        return "rid-123"

    monkeypatch.setattr(main, "run", _fake_run)

    path = tmp_path / "run.json"
    path.write_text(
        json.dumps(
            {
                "run_name": "cli config-uri test",
                "data": {"source_table": "source_series_native", "horizon": 7},
                "models": [_SPARK],
            }
        )
    )
    main._main(["--config-uri", str(path), "--dry-run"])
    assert seen["run_name"] == "cli config-uri test"


def test_cli_requires_exactly_one_config_source() -> None:
    """Exactly one of ``--config`` / ``--config-uri`` is required (neither, or both, exits)."""
    with pytest.raises(SystemExit):
        main._main(["--dry-run"])  # neither source
    with pytest.raises(SystemExit):
        main._main(["--config", "a.json", "--config-uri", "gs://b/c.json"])  # both sources


# --- lock_profile_source: pinning the reference before the digest ---------------


def _lock_cfg(**profile: Any) -> RunConfig:
    return _cfg(compute={"profile": profile}) if profile else _cfg()


def _fake_discover(monkeypatch: pytest.MonkeyPatch, result: Any) -> list[dict[str, Any]]:
    """Stand in for the BigQuery discovery query; return the kwargs it was called with."""
    seen: list[dict[str, Any]] = []

    def discover(**kwargs: Any) -> Any:
        seen.append(kwargs)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr("scale_forecasting.registry.harvest.discover_harvest_run", discover)
    return seen


def test_auto_is_pinned_to_the_run_it_resolves_to(monkeypatch: pytest.MonkeyPatch) -> None:
    """The lockfile trick: what actually sized this run is written in, never re-searched."""
    seen = _fake_discover(monkeypatch, "prior-run-0123456789ab")
    locked = main.lock_profile_source(_lock_cfg(), settings=_SETTINGS)
    assert locked.compute.profile.source == "prior-run-0123456789ab"
    # Only the identity axes are filtered in SQL; the scale axes are warnings, checked after load.
    assert set(seen[0]) == {"source_table", "freq", "settings"}


def test_pinning_moves_the_run_id_because_the_fleet_moved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run sized off last week's evidence is not the same run — §2.6's rule, applied."""
    _fake_discover(monkeypatch, "prior-run-0123456789ab")
    cfg = _lock_cfg()
    assert make_run_id(main.lock_profile_source(cfg, settings=_SETTINGS)) != make_run_id(cfg)


def test_finding_nothing_pins_the_baseline_rather_than_leaving_auto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rest of the chain is deterministic, so pinning it is what makes the plan reproducible."""
    _fake_discover(monkeypatch, None)
    assert main.lock_profile_source(_lock_cfg(), settings=_SETTINGS).compute.profile.source == (
        "baseline"
    )


@pytest.mark.parametrize("source", ["none", "baseline", "some-run-0123456789ab"])
def test_an_already_concrete_source_is_never_re_resolved(
    monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    """Only `auto` is a search. Anything else is already the operator's answer."""
    seen = _fake_discover(monkeypatch, "should-not-be-used-0123456789ab")
    cfg = _lock_cfg(source=source)
    assert main.lock_profile_source(cfg, settings=_SETTINGS) is cfg
    assert seen == []


def test_profiling_off_is_never_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    """`mode="off"` is the one switch that makes the whole feature inert, plan time included."""
    seen = _fake_discover(monkeypatch, "should-not-be-used-0123456789ab")
    cfg = _lock_cfg(mode="off")
    assert main.lock_profile_source(cfg, settings=_SETTINGS) is cfg
    assert seen == []


def test_an_unreachable_registry_leaves_the_plan_unpinned_rather_than_failing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plan with no SF_* environment is a preview; it must still produce an id and a fanout."""
    _fake_discover(monkeypatch, RuntimeError("no credentials"))
    cfg = _lock_cfg()
    assert main.lock_profile_source(cfg, settings=_SETTINGS).compute.profile.source == "auto"
