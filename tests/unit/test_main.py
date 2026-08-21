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

from scale_forecasting import dag, main
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


# --- _launch_family_job / _launch_native_job: per-job lifecycle + dispatch ------


def _fake_job_lifecycle(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Fake the run_jobs attempt+lifecycle seams so a family launch runs offline.

    Records the ``run_job`` call args under ``"job"`` and yields a real `JobFinalizer` so the body
    can finalize normally. ``next_job_attempt`` is pinned to ``(1, True)`` (first attempt, new job).
    """
    from contextlib import contextmanager

    from scale_forecasting.registry import bq

    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        bq, "next_job_attempt", lambda run_id, family, *, force=False, settings=None: (1, True)
    )

    @contextmanager
    def _fake_run_job(run_id: str, family: str, attempt: int, **kw: Any) -> Any:
        seen["job"] = {"run_id": run_id, "family": family, "attempt": attempt, **kw}
        fin = bq.JobFinalizer()
        seen["fin"] = fin  # expose it so tests can assert what the body finalized
        yield fin

    monkeypatch.setattr(bq, "run_job", _fake_run_job)
    return seen


def test_launch_family_job_dispatches_to_resolved_submitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scale_forecasting.submitters as submitters_mod

    seen = _fake_job_lifecycle(monkeypatch)
    captured: dict[str, Any] = {}

    class _FakeSubmitter:
        def launch(self, cfg: RunConfig, **kw: Any) -> None:
            captured.update(kw)
            captured["cfg"] = cfg

    def _fake_get(runtime: str) -> Any:
        captured["runtime"] = runtime
        return _FakeSubmitter()

    monkeypatch.setattr(submitters_mod, "get_submitter", _fake_get)

    cfg = _cfg(models=[_SPARK])
    job = dag.plan_dag(cfg).python_jobs[0]  # the statistical family, on the default Spark runtime
    main._launch_family_job(cfg, job, "rid-0", _SETTINGS, max_executors=8)

    # Dispatch is by the family's *resolved* runtime, in contributor mode, with its model subset.
    assert captured["runtime"] == "spark"
    assert captured["models"] == [_SPARK]
    assert captured["manage_header"] is False
    assert captured["max_executors"] == 8
    # The per-job row is opened for this family's resolved compute + attempt.
    assert seen["job"]["family"] == "statistical"
    assert seen["job"]["attempt"] == 1
    assert seen["job"]["runtime"] == "spark"
    # The deterministic per-family platform id is threaded onto both the row and the submitter,
    # so a Spark family under a shared run_id gets its own batch id.
    from scale_forecasting.registry.ids import dataproc_job_id, make_job_key

    expected_id = dataproc_job_id(make_job_key("rid-0", "statistical", 1))
    assert seen["job"]["system_job_id"] == expected_id
    assert captured["system_job_id"] == expected_id
    # The default fake submitter returns None (its id == system_job_id), so nothing is stamped back.
    assert "system_job_id" not in seen["fin"].extra


def test_launch_family_job_stamps_real_id_when_submitter_returns_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cluster submitter returns a server-assigned id → it's finalized onto the row."""
    import scale_forecasting.submitters as submitters_mod

    seen = _fake_job_lifecycle(monkeypatch)

    class _ClusterSubmitter:
        def launch(self, cfg: RunConfig, **kw: Any) -> str:
            return "real-dataproc-job-id"  # differs from the deterministic system_job_id

    monkeypatch.setattr(submitters_mod, "get_submitter", lambda runtime: _ClusterSubmitter())

    cfg = _cfg(models=[_SPARK])
    job = dag.plan_dag(cfg).python_jobs[0]
    main._launch_family_job(cfg, job, "rid-0", _SETTINGS)

    # The real (server-assigned) id is stamped back onto the run_jobs row for reverse-trace.
    assert seen["fin"].extra["system_job_id"] == "real-dataproc-job-id"


def test_launch_family_job_dispatches_ray_for_ray_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scale_forecasting.submitters as submitters_mod

    _fake_job_lifecycle(monkeypatch)
    captured: dict[str, Any] = {}

    class _FakeSubmitter:
        def launch(self, cfg: RunConfig, **kw: Any) -> None:
            captured.update(kw)

    monkeypatch.setattr(
        submitters_mod,
        "get_submitter",
        lambda runtime: captured.__setitem__("runtime", runtime) or _FakeSubmitter(),
    )

    cfg = _cfg(models=[_SPARK], python_runtime="ray")
    job = dag.plan_dag(cfg).python_jobs[0]
    main._launch_family_job(cfg, job, "rid-0", _SETTINGS)
    assert captured["runtime"] == "ray"
    # Ray keeps the canonical key verbatim as its submission id.
    from scale_forecasting.registry.ids import make_job_key, ray_submission_id

    assert captured["system_job_id"] == ray_submission_id(make_job_key("rid-0", "statistical", 1))


def test_launch_native_job_runs_bigquery_engine_inline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scale_forecasting.engines import bigquery_engine

    seen = _fake_job_lifecycle(monkeypatch)
    captured: dict[str, Any] = {}

    def _fake_bq_run(cfg: RunConfig, models: list[str], **kw: Any) -> Any:
        captured["models"] = models
        captured["manage_header"] = kw.get("manage_header")
        return bigquery_engine.BqOutcome(status="COMPLETED", n_series=3, models=models)

    monkeypatch.setattr(bigquery_engine, "run", _fake_bq_run)

    cfg = _cfg()
    native = dag.plan_dag(cfg).native_job
    assert native is not None
    outcome = main._launch_native_job(cfg, native, "rid-0", _SETTINGS)

    assert captured["models"] == _NATIVE
    assert captured["manage_header"] is False
    assert outcome.n_series == 3
    # The native family's row is opened with the BigQuery runtime.
    assert seen["job"]["family"] == "native"
    assert seen["job"]["runtime"] == "bigquery"
    from scale_forecasting.registry.ids import bigquery_job_id, make_job_key

    assert seen["job"]["system_job_id"] == bigquery_job_id(make_job_key("rid-0", "native", 1))


# --- run(): ensemble orchestration after the engine join -----------------------


def _patch_run_seams(
    monkeypatch: pytest.MonkeyPatch, *, bq_error: Exception | None = None
) -> dict[str, Any]:
    """Fake every GCP seam main.run touches so the ensemble gating is exercised offline.

    Records what happened in the returned dict: the header status finalized, whether the native
    (BigQuery) family and the Python family jobs ran. Each family launch is faked at the
    `main._launch_family_job` / `main._launch_native_job` seam to a no-op success (so the per-job
    run_jobs lifecycle and submitters are never reached offline). ``bq_error`` makes the native job
    raise, to prove ensembles are skipped when a family fails.
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

    def _fake_native(cfg: RunConfig, job: Any, run_id: str, settings: Any, *, force: bool = False):
        seen["bq_ran"] = True
        if bq_error is not None:
            raise bq_error
        return bigquery_engine.BqOutcome(status="COMPLETED", n_series=3, models=list(job.models))

    monkeypatch.setattr(main, "_launch_native_job", _fake_native)
    monkeypatch.setattr(
        main, "_launch_family_job", lambda *a, **k: seen.__setitem__("spark_ran", True)
    )

    # The ensemble DAG node opens its own run_jobs row before running the consensus; fake that
    # per-job lifecycle so _launch_ensemble_job runs for real down to the run_ensembles call.
    import contextlib

    monkeypatch.setattr(bq, "next_job_attempt", lambda *a, **k: (1, None))

    @contextlib.contextmanager
    def _fake_run_job(*a: Any, **k: Any) -> Any:
        yield None

    monkeypatch.setattr(bq, "run_job", _fake_run_job)

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


# --- shared ephemeral Ray cluster across families ------------------------------
# A run with two or more ephemeral Ray families shares ONE cluster: the orchestrator provisions it,
# each family submits its own failure-isolated job to it, and it's torn down once. These tests cover
# the pure sizing helper, the context manager's engage/skip/teardown behavior, and the per-family
# threading — all offline (provision/teardown are faked; nothing touches Vertex).


def _ray_cfg(**over: Any) -> RunConfig:
    # theta (statistical) + xgboost (ml) both resolve to Ray → two ephemeral Ray families.
    base: dict[str, Any] = {
        "run_name": "shared ray test",
        "data": {"source_table": "source_series_native", "horizon": 7, "series_limit": 5},
        "models": ["theta", "xgboost"],
        "python_runtime": "ray",
    }
    base.update(over)
    return RunConfig(**base)


def test_shared_ray_inputs_none_for_single_ray_family() -> None:
    # One Ray family self-provisions (no collision risk); sharing doesn't apply.
    run_dag = dag.plan_dag(_ray_cfg(models=["theta"]))
    assert main._shared_ray_inputs(run_dag.python_jobs) is None


def test_shared_ray_inputs_none_when_no_ray_family() -> None:
    run_dag = dag.plan_dag(_cfg(models=["theta", "holtwinters"]))  # default spark
    assert main._shared_ray_inputs(run_dag.python_jobs) is None


def test_shared_ray_inputs_unions_models_cpu() -> None:
    run_dag = dag.plan_dag(_ray_cfg())
    inputs = main._shared_ray_inputs(run_dag.python_jobs)
    assert inputs is not None
    models, any_gpu, gpu_type = inputs
    assert sorted(models) == ["theta", "xgboost"]
    assert any_gpu is False
    assert gpu_type is None


def test_shared_ray_inputs_flags_gpu_from_deep_learning() -> None:
    # theta (statistical, cpu) + neuralprophet (deep_learning, gpu) both on Ray.
    run_dag = dag.plan_dag(
        _ray_cfg(models=["theta", "neuralprophet"], compute={"use_gpu": True, "gpu_type": "T4"})
    )
    inputs = main._shared_ray_inputs(run_dag.python_jobs)
    assert inputs is not None
    models, any_gpu, gpu_type = inputs
    assert sorted(models) == ["neuralprophet", "theta"]
    assert any_gpu is True
    assert gpu_type == "T4"


def _patch_shared_cluster(monkeypatch: pytest.MonkeyPatch, calls: dict[str, Any]) -> None:
    from scale_forecasting import ray_submit

    def _provision(cfg: RunConfig, **kw: Any) -> tuple[str, str]:
        calls["provision"] = kw
        return "sf-ray-shared", "us-west1"

    def _teardown(name: str, region: str, settings: Settings) -> None:
        calls["teardown"] = (name, region)

    monkeypatch.setattr(ray_submit, "provision_shared_cluster", _provision)
    monkeypatch.setattr(ray_submit, "teardown_shared_cluster", _teardown)


def test_shared_ray_cluster_engages_and_tears_down(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}
    _patch_shared_cluster(monkeypatch, calls)
    cfg = _ray_cfg()
    run_dag = dag.plan_dag(cfg)
    with main._shared_ray_cluster(cfg, run_dag, "run-abc", _SETTINGS) as ray_cluster:
        assert ray_cluster == ("sf-ray-shared", "us-west1")
        assert "provision" in calls
        assert "teardown" not in calls  # not yet — torn down on exit
    assert calls["teardown"] == ("sf-ray-shared", "us-west1")


def test_shared_ray_cluster_tears_down_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}
    _patch_shared_cluster(monkeypatch, calls)
    cfg = _ray_cfg()
    run_dag = dag.plan_dag(cfg)
    with pytest.raises(RuntimeError, match="boom"):
        with main._shared_ray_cluster(cfg, run_dag, "run-abc", _SETTINGS):
            raise RuntimeError("boom")
    assert calls["teardown"] == ("sf-ray-shared", "us-west1")  # finally still ran


def test_shared_ray_cluster_skips_single_ray_family(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}
    _patch_shared_cluster(monkeypatch, calls)
    cfg = _ray_cfg(models=["theta"])
    run_dag = dag.plan_dag(cfg)
    with main._shared_ray_cluster(cfg, run_dag, "run-abc", _SETTINGS) as ray_cluster:
        assert ray_cluster is None
    assert calls == {}  # never provisioned, never torn down


def test_shared_ray_cluster_skips_spark_run(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}
    _patch_shared_cluster(monkeypatch, calls)
    cfg = _cfg(models=["theta", "holtwinters"])  # default spark
    run_dag = dag.plan_dag(cfg)
    with main._shared_ray_cluster(cfg, run_dag, "run-abc", _SETTINGS) as ray_cluster:
        assert ray_cluster is None
    assert calls == {}


def test_shared_ray_cluster_skips_when_standing_cluster_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A config reusing a standing cluster targets it directly; the orchestrator provisions nothing.
    calls: dict[str, Any] = {}
    _patch_shared_cluster(monkeypatch, calls)
    cfg = _ray_cfg(compute={"ray_cluster_name": "my-standing-ray"})
    run_dag = dag.plan_dag(cfg)
    with main._shared_ray_cluster(cfg, run_dag, "run-abc", _SETTINGS) as ray_cluster:
        assert ray_cluster is None
    assert calls == {}


class _CapturingSubmitter:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def launch(self, cfg: RunConfig, **kw: Any) -> None:
        self.kwargs = kw


def _patch_launch_seams(monkeypatch: pytest.MonkeyPatch, sub: _CapturingSubmitter) -> None:
    import contextlib

    from scale_forecasting import submitters
    from scale_forecasting.registry import bq

    monkeypatch.setattr(bq, "next_job_attempt", lambda *a, **k: (1, None))

    @contextlib.contextmanager
    def _fake_run_job(*a: Any, **k: Any) -> Any:
        yield None

    monkeypatch.setattr(bq, "run_job", _fake_run_job)
    monkeypatch.setattr(submitters, "get_submitter", lambda runtime: sub)


def test_launch_family_job_threads_shared_cluster_for_ray(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sub = _CapturingSubmitter()
    _patch_launch_seams(monkeypatch, sub)
    run_dag = dag.plan_dag(_ray_cfg())
    ray_job = next(j for j in run_dag.python_jobs if j.runtime == "ray")
    main._launch_family_job(
        _ray_cfg(), ray_job, "run-abc", _SETTINGS, ray_cluster=("sf-ray-shared", "us-west1")
    )
    assert sub.kwargs["ray_cluster_name"] == "sf-ray-shared"
    assert sub.kwargs["ray_cluster_region"] == "us-west1"


def test_launch_family_job_ignores_shared_cluster_for_spark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sub = _CapturingSubmitter()
    _patch_launch_seams(monkeypatch, sub)
    cfg = _cfg(models=["theta", "holtwinters"])  # default spark
    run_dag = dag.plan_dag(cfg)
    spark_job = next(j for j in run_dag.python_jobs if j.runtime == "spark")
    main._launch_family_job(
        cfg, spark_job, "run-abc", _SETTINGS, ray_cluster=("sf-ray-shared", "us-west1")
    )
    assert sub.kwargs["ray_cluster_name"] is None
    assert sub.kwargs["ray_cluster_region"] is None


# --- ensemble DAG node: identity + mode dispatch -------------------------------


def _patch_ensemble_seams(monkeypatch: pytest.MonkeyPatch, calls: dict[str, Any]) -> None:
    import contextlib

    from scale_forecasting import ensemble_run
    from scale_forecasting.registry import bq

    monkeypatch.setattr(bq, "next_job_attempt", lambda *a, **k: (1, None))

    @contextlib.contextmanager
    def _fake_run_job(run_id: str, family: str, attempt: int, **k: Any) -> Any:
        calls["run_job"] = {"family": family, "runtime": k.get("runtime")}
        yield None

    monkeypatch.setattr(bq, "run_job", _fake_run_job)
    monkeypatch.setattr(
        ensemble_run, "run_ensembles", lambda *a, **k: calls.__setitem__("barrier", True)
    )
    monkeypatch.setattr(
        ensemble_run,
        "run_ensembles_microbatch",
        lambda *a, **k: calls.__setitem__("microbatch", True),
    )


def test_launch_ensemble_job_barrier_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}
    _patch_ensemble_seams(monkeypatch, calls)
    cfg = _cfg(ensemble={"enabled": True, "strategies": ["mean"]})
    main._launch_ensemble_job(cfg, "run-abc", _SETTINGS)
    assert calls.get("barrier") is True
    assert "microbatch" not in calls
    # It opens its own run_jobs row as the "ensemble" family, executed on the driver (bigquery).
    assert calls["run_job"] == {"family": "ensemble", "runtime": "bigquery"}


def test_launch_ensemble_job_microbatch_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}
    _patch_ensemble_seams(monkeypatch, calls)
    cfg = _cfg(
        ensemble={"enabled": True, "strategies": ["mean"]},
        compute={"ensemble": {"runtime": "spark", "mode": "microbatch"}},
    )
    main._launch_ensemble_job(cfg, "run-abc", _SETTINGS)
    assert calls.get("microbatch") is True
    assert "barrier" not in calls
