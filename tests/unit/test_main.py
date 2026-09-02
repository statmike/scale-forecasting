"""Offline tests for the run orchestrator (``scale_forecasting.main``).

No GCP: the ``dry_run`` path, the fan-out `main.run` drives over the DAG's nodes, the combined
run status it rolls the per-job outcomes into, and the CLI that dispatches all of it. Resolving a
config into a plan and staging it are `scale_forecasting.launch_plan`'s and are tested in
``test_launch_plan.py``. The live parallel launch (Spark batch + BigQuery engine under one run_id)
is the ``@gcp`` smoke in ``tests/integration/test_main_orchestration_smoke.py``; here the GCP seams
are either faked or never reached, because ``dry_run`` returns before them.
"""

from __future__ import annotations

from typing import Any

import pytest

from scale_forecasting import dag, job_launch, launch_plan, main
from scale_forecasting.config import RunConfig
from scale_forecasting.errors import ConfigError
from scale_forecasting.registry.ids import make_run_id
from scale_forecasting.settings import Settings

# Model names by runtime: theta is a Python/Spark model; arima_plus / timesfm are the
# BigQuery-native models (runtime == "bigquery").
_SPARK = "theta"
_NATIVE = ["arima_plus", "timesfm"]

# One ``forecast_metadata`` measurement row, enough for a pinned profile source to resolve to a
# real profile (and so to a real signature comparison). The full harvest arithmetic is
# ``test_profiling.py``'s; here it only has to load.
_HARVEST_ROW = {
    "ts_id": "series-a",
    "model_type": "theta",
    "fit_seconds": 2.0,
    "cpu_seconds": 2.0,
    "process_rss_bytes": 1024**3,
    "n_obs": 400,
}

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
    from scale_forecasting.registry import harvest, header

    monkeypatch.setattr(header, "header_status", lambda *a, **k: None)
    # Locking `profile.source: "auto"` before the digest is also a registry query, on every verb.
    # Offline there is nothing to discover, so it pins "baseline" — the deterministic remainder of
    # the chain — and every id in this file is the id of a baseline-pinned config.
    monkeypatch.setattr(harvest, "discover_harvest_run", lambda **k: None)


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


def test_cli_dispatches_stage_only(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    from types import SimpleNamespace

    seen: dict[str, Any] = {}

    def _fake_stage(cfg: RunConfig, *, force: bool = False) -> Any:
        seen["run_name"] = cfg.run_name
        seen["force"] = force
        return SimpleNamespace(run_id="rid-staged")

    monkeypatch.setattr(launch_plan, "stage_run", _fake_stage)

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
    assert run_id == main.run(cfg, dry_run=True)  # and the three verbs agree on the id
    assert "spark_ran" not in seen
    assert "bq_ran" not in seen
    assert seen["ensemble_called"] is False
    assert "status" not in seen  # header never re-finalized


def test_every_verb_resolves_the_same_id_whether_or_not_it_locks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Four entry points, one id — that is the contract, and it used to hold only because three of
    # them locked identically. `plan_dag` never locked, so its id was the odd one out; discovery
    # finding a prior run was enough to make `--dry-run` print an id the real run then never used.
    # With the resolved source out of the digest, agreement no longer depends on every verb
    # remembering to lock, which is why `plan_dag` is asserted equal here rather than unequal.
    from scale_forecasting.registry import harvest, header

    _patch_run_seams(monkeypatch)
    monkeypatch.setattr(harvest, "discover_harvest_run", lambda **k: "prior-run-0123456789ab")
    monkeypatch.setattr(header, "header_status", lambda *a, **k: "COMPLETED")  # return early
    cfg = _cfg()

    run_id = main.run(cfg)
    assert run_id == main.run(cfg, dry_run=True)
    assert run_id == launch_plan.plan_run(cfg).run_id
    assert run_id == dag.plan_dag(cfg).run_id


def test_run_refuses_a_pinned_profile_source_whose_data_moved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A hand-pinned source is an assertion that that run's measurements apply here. When they no
    # longer do, the run stops at the entry point rather than sizing the whole fleet off evidence
    # the operator did not mean to use.
    from scale_forecasting.registry import harvest

    _patch_run_seams(monkeypatch)
    monkeypatch.setattr(
        harvest, "read_compute_harvest", lambda run_id, **k: ([_HARVEST_ROW], "another_table")
    )
    cfg = _cfg(compute={"profile": {"source": "prior-run-0123456789ab"}})

    with pytest.raises(ConfigError, match="prior-run-0123456789ab"):
        main.run(cfg)
    main.run(cfg, force=True)  # the documented override


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


# --- CLI: says something at all -------------------------------------------------


def test_cli_installs_a_log_handler_so_its_output_is_not_swallowed(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every verb reports through ``_log.info``, so a CLI with no handler prints nothing.

    Found live: the workshop's documented first command
    (``main --config … --dry-run``) exited 0 in total silence, because nothing in the
    package calls `logging.basicConfig` and the root logger ships with no handler at
    WARNING. Correct for a library, useless for a CLI. The assertion is on the *handler*
    and the *level*, not on captured text — pytest attaches its own handler, so a caplog
    test would pass against the broken code.
    """
    import json
    import logging

    monkeypatch.setattr(main, "run", lambda cfg, *, dry_run=False, force=False: "rid-123")

    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    root.handlers = []
    try:
        path = tmp_path / "run.json"
        path.write_text(
            json.dumps(
                {
                    "run_name": "cli logging test",
                    "data": {"source_table": "source_series_native", "horizon": 7},
                    "models": [_SPARK],
                }
            )
        )
        main._main(["--config", str(path), "--dry-run"])
        assert root.handlers, "the CLI left the root logger with no handler"
        assert root.level <= logging.INFO, f"INFO is swallowed at level {root.level}"
    finally:
        root.handlers, root.level = saved_handlers, saved_level


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
                family="statistical",
                runtime="spark",
                registry_status="RUNNING",
                native_state="RUNNING",
                exists=True,
                verdict=VERDICT_RUNNING,
                disagreement=False,
                n_done=3,
                n_expected=10,
                detail="in flight",
            ),
            FamilyVerdict(
                family="native",
                runtime="bigquery",
                registry_status="RUNNING",
                native_state="SUCCEEDED",
                exists=True,
                verdict=VERDICT_STALE_REGISTRY,
                disagreement=True,
                n_done=5,
                n_expected=None,
                detail="",
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
