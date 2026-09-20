"""Offline tests for the plan/stage half of a run (``scale_forecasting.launch_plan``).

Nothing here launches anything, which is the point: these prove that resolving a config offline
gives the *same* answer a live run would — the same ``run_id``, the same per-runtime model split,
the same fanout — and that the artifacts `stage_run` uploads land where the launch will look for
them. The exists-vs-new verdict, the two-tier command emit, and the ``profile.source: "auto"``
pinning that has to happen before the digest are all covered here.

No GCP: `Settings` is injected, the deployment envelope is a stub, and the staging seams are faked.
"""

from __future__ import annotations

from typing import Any

import pytest

from scale_forecasting import launch_plan
from scale_forecasting.config import RunConfig
from scale_forecasting.errors import ConfigError
from scale_forecasting.registry.ids import make_run_id
from scale_forecasting.settings import Settings

# Model names by runtime: theta is a Python/Spark model; arima_plus / timesfm are the
# BigQuery-native models (runtime == "bigquery").
_SPARK = "theta"
_NATIVE = ["arima_plus", "timesfm"]

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
    assert launch_plan._plan(cfg).run_id == make_run_id(cfg)


def test_plan_splits_models_by_runtime() -> None:
    plan = launch_plan._plan(_cfg())
    assert plan.python_models == [_SPARK]
    assert plan.bq_models == _NATIVE


def test_plan_all_bigquery_has_no_python_models() -> None:
    plan = launch_plan._plan(_cfg(models=_NATIVE))
    assert plan.python_models == []
    assert plan.bq_models == _NATIVE


def test_plan_all_python_has_no_bq_models() -> None:
    plan = launch_plan._plan(_cfg(models=[_SPARK, "holtwinters"]))
    assert plan.python_models == [_SPARK, "holtwinters"]
    assert plan.bq_models == []


# --- _plan: ray is accepted ----------------------------------------------------


def test_plan_accepts_ray_when_python_models_present() -> None:
    # The Ray engine is built, so main.run now dispatches ray — _plan must NOT reject it.

    plan = launch_plan._plan(_cfg(models=[_SPARK, *_NATIVE], python_runtime="ray"))
    assert plan.python_models == [_SPARK]
    assert plan.bq_models == _NATIVE


def test_plan_allows_ray_config_when_only_bigquery_models() -> None:
    # An all-native config never uses the Python runtime, so runtime choice doesn't apply.
    plan = launch_plan._plan(_cfg(models=_NATIVE, python_runtime="ray"))
    assert plan.python_models == []
    assert plan.bq_models == _NATIVE


def test_plan_run_without_env_returns_plan_but_no_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No SF_* identity → command emission is skipped, but the id/fanout/split still resolve offline.
    import scale_forecasting.settings as settings_mod

    def _no_env() -> Settings:
        raise ConfigError("no SF_* env")

    monkeypatch.setattr(settings_mod.Settings, "resolve", staticmethod(_no_env))

    result = launch_plan.plan_run(_cfg())
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


def test_plan_run_spark_emits_main_and_a_command_per_family() -> None:
    result = launch_plan.plan_run(_cfg(models=[_SPARK]), settings=_SETTINGS, infra=_batch_infra())
    assert result.commands is not None
    # theta is the statistical family, and the Spark tier is keyed per family — not one "spark".
    assert set(result.commands) == {"main", "spark:statistical"}
    config_uri = f"gs://bkt-code/runs/{result.run_id}.json"
    assert result.config_uri == config_uri
    # main = the orchestrator command; the family = native gcloud + universal, both on the URI.
    assert result.commands["main"].universal.endswith(config_uri)
    assert result.commands["main"].native is None
    spark = result.commands["spark:statistical"]
    assert spark.native is not None and "gcloud" in spark.native and config_uri in spark.native
    # Each family names its own subset, even when it is the only one — the run submits it that way.
    assert "--models" in spark.native and _SPARK in spark.native


def test_plan_run_family_commands_carry_the_batch_ids_the_run_will_use() -> None:
    # The printed command is only worth printing if it submits the batch `run` submits. Its --batch
    # id therefore has to be the planned node's id, not the derived single-batch `sf-<run_id>`.
    from scale_forecasting.registry.ids import dataproc_job_id

    result = launch_plan.plan_run(_cfg(models=[_SPARK]), settings=_SETTINGS, infra=_batch_infra())
    assert result.commands is not None
    node = next(n for n in result.nodes if n.family == "statistical")
    expected = dataproc_job_id(node.job_key)
    spark = result.commands["spark:statistical"]
    assert spark.native is not None and f"--batch={expected}" in spark.native
    assert f"--batch-id {expected}" in spark.universal


def test_plan_run_mixed_spark_restricts_the_spark_subset() -> None:
    result = launch_plan.plan_run(
        _cfg(models=[_SPARK, *_NATIVE]), settings=_SETTINGS, infra=_batch_infra()
    )
    assert result.commands is not None
    # The BigQuery-native models run in BigQuery, so no Spark command mentions them.
    assert set(result.commands) == {"main", "spark:statistical"}
    spark = result.commands["spark:statistical"]
    assert spark.native is not None and "--models" in spark.native and _SPARK in spark.native
    assert not any(m in spark.native for m in _NATIVE)


def test_plan_run_gpu_family_emits_a_gpu_shaped_command_and_the_cpu_one_stays_cpu() -> None:
    # Two Spark families on different devices is exactly the case one command cannot describe: a
    # Serverless executor's shape is fixed at batch creation, so the deep-learning line has to buy
    # an L4 and the statistical line beside it must not.
    result = launch_plan.plan_run(
        _cfg(
            models=[_SPARK, "neuralprophet"],
            compute={"families": {"deep_learning": {"hardware": "gpu", "gpu_type": "L4"}}},
        ),
        settings=_SETTINGS,
        infra=_batch_infra(),
    )
    assert result.commands is not None
    assert set(result.commands) == {"main", "spark:statistical", "spark:deep_learning"}
    dl = result.commands["spark:deep_learning"]
    assert dl.native is not None
    assert "--provisioned-hardware gpu" in dl.native
    assert "spark.dataproc.executor.resource.accelerator.type=l4" in dl.native
    assert "--hardware gpu" in dl.universal and "--gpu-type L4" in dl.universal
    stat = result.commands["spark:statistical"]
    assert stat.native is not None and "--provisioned-hardware" not in stat.native
    assert "--hardware" not in stat.universal
    # Different families, different batches — the ids the run fans out under must not collide.
    assert dl.universal != stat.universal


def test_plan_run_skips_a_cluster_mode_family_rather_than_misdescribe_it() -> None:
    # `gcloud dataproc batches submit` is the Serverless form. A cluster family's job goes to a
    # cluster the run creates, so there is no standalone line to copy — and printing the Serverless
    # one would give a reader a command that runs and runs the wrong thing.
    result = launch_plan.plan_run(
        _cfg(
            models=[_SPARK, "neuralprophet"],
            compute={
                "families": {
                    "deep_learning": {"spark_mode": "cluster", "hardware": "gpu", "gpu_type": "T4"}
                }
            },
        ),
        settings=_SETTINGS,
        infra=_batch_infra(),
    )
    assert result.commands is not None
    assert set(result.commands) == {"main", "spark:statistical"}
    # The node is still planned and still in the DAG — only its *command* is withheld.
    assert any(n.family == "deep_learning" and n.spark_mode == "cluster" for n in result.nodes)


def test_plan_run_ray_emits_universal_only_ray_command() -> None:
    from scale_forecasting.ray_infra import RayInfra

    infra = RayInfra(compute_sa="sf-compute@proj-x.iam", code_bucket="bkt-code")
    result = launch_plan.plan_run(
        _cfg(models=[_SPARK, *_NATIVE], python_runtime="ray"), settings=_SETTINGS, infra=infra
    )
    assert result.commands is not None
    assert set(result.commands) == {"main", "ray"}
    ray = result.commands["ray"]
    assert ray.native is None  # no gcloud verb submits a Ray job
    assert "ray_submit" in ray.universal and result.run_id in ray.universal


def test_plan_run_reports_new_run_when_config_never_ran() -> None:
    # The autouse fixture makes header_status return None → the config has not run before.
    result = launch_plan.plan_run(_cfg(models=[_SPARK]), settings=_SETTINGS, infra=_batch_infra())
    assert result.idempotency.checked is True
    assert result.idempotency.exists is False
    assert result.idempotency.prior_status is None


def test_plan_run_reports_existing_run_when_config_already_ran(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scale_forecasting.registry import header

    monkeypatch.setattr(header, "header_status", lambda *a, **k: "COMPLETED")
    result = launch_plan.plan_run(_cfg(models=[_SPARK]), settings=_SETTINGS, infra=_batch_infra())
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
    result = launch_plan.plan_run(_cfg(models=[_SPARK]), settings=_SETTINGS, infra=_batch_infra())
    assert result.idempotency.checked is False
    assert result.idempotency.exists is False


def test_plan_run_without_env_leaves_verdict_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    # No SF_* env → settings never resolve, so the registry is never consulted: verdict unknown.
    import scale_forecasting.settings as settings_mod

    def _no_env() -> Settings:
        raise ConfigError("no SF_* env")

    monkeypatch.setattr(settings_mod.Settings, "resolve", staticmethod(_no_env))
    result = launch_plan.plan_run(_cfg())
    assert result.idempotency.checked is False


def test_emit_idempotency_warns_on_existing_and_notes_force(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from scale_forecasting.registry import header

    monkeypatch.setattr(header, "header_status", lambda *a, **k: "COMPLETED")

    with caplog.at_level("WARNING"):
        launch_plan.plan_run(_cfg(models=[_SPARK]), settings=_SETTINGS, infra=_batch_infra())
    assert any("already ran" in r.message and "--force" in r.message for r in caplog.records)

    caplog.clear()
    with caplog.at_level("INFO"):
        launch_plan.plan_run(
            _cfg(models=[_SPARK]), settings=_SETTINGS, infra=_batch_infra(), force=True
        )
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

    result = launch_plan.stage_run(_cfg(models=[_SPARK]), settings=_SETTINGS, infra=_batch_infra())
    assert result.staged is True
    assert result.config_uri == f"gs://bkt-code/runs/{result.run_id}.json"
    assert result.commands is not None
    spark = result.commands["spark:statistical"]
    assert spark.native is not None and "gs://bkt-code/runs/pkg.zip" in spark.native

    manifest = captured["manifest"]
    assert manifest["run_id"] == result.run_id
    assert manifest["config_uri"] == result.config_uri
    assert set(manifest["commands"]) == {"main", "spark:statistical"}
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

    result = launch_plan.plan_run(
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
        launch_plan.stage_run(_cfg())


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
    locked = launch_plan.lock_profile_source(_lock_cfg(), settings=_SETTINGS)
    assert locked.compute.profile.source == "prior-run-0123456789ab"
    # The identity axes filter in SQL; scale and runtime do not filter but do *rank*, so both have
    # to reach discovery rather than only being checked after the rows are loaded.
    assert set(seen[0]) == {
        "source_table",
        "freq",
        "target_series",
        "target_runtime",
        "settings",
    }


def test_pinning_does_not_move_the_run_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Identity is what was asked for; the resolved source is provenance, and is excluded.

    The regression this guards was seen live (smoke 01, 2026-09-01) rather than reasoned about:
    with the source in the digest, run N pinned run N-1's harvest, so re-running one config
    produced a new id every time and a "re-run" executed a whole second run instead of deduping.
    """
    _fake_discover(monkeypatch, "prior-run-0123456789ab")
    cfg = _lock_cfg()
    assert make_run_id(launch_plan.lock_profile_source(cfg, settings=_SETTINGS)) == make_run_id(cfg)


def test_the_id_is_the_same_whether_or_not_discovery_reached_the_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other half: a transient registry error must not fork identity either.

    Discovery failing leaves the config unpinned (the test below), so before the exclusion a
    timeout or a permissions blip silently produced a *different* `run_id` for the same file —
    which is exactly how smoke 02 forked on 2026-09-01, when the discovery query ran a moment
    before its own schema migration.
    """
    _fake_discover(monkeypatch, RuntimeError("no credentials"))
    unreachable = make_run_id(launch_plan.lock_profile_source(_lock_cfg(), settings=_SETTINGS))
    _fake_discover(monkeypatch, "prior-run-0123456789ab")
    reachable = make_run_id(launch_plan.lock_profile_source(_lock_cfg(), settings=_SETTINGS))
    assert unreachable == reachable


def test_finding_nothing_pins_the_baseline_rather_than_leaving_auto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rest of the chain is deterministic, so pinning it is what makes the plan reproducible."""
    _fake_discover(monkeypatch, None)
    locked = launch_plan.lock_profile_source(_lock_cfg(), settings=_SETTINGS)
    assert locked.compute.profile.source == "baseline"


@pytest.mark.parametrize("source", ["none", "baseline", "some-run-0123456789ab"])
def test_an_already_concrete_source_is_never_re_resolved(
    monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    """Only `auto` is a search. Anything else is already the operator's answer."""
    seen = _fake_discover(monkeypatch, "should-not-be-used-0123456789ab")
    cfg = _lock_cfg(source=source)
    assert launch_plan.lock_profile_source(cfg, settings=_SETTINGS) is cfg
    assert seen == []


def test_profiling_off_is_never_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    """`mode="off"` is the one switch that makes the whole feature inert, plan time included."""
    seen = _fake_discover(monkeypatch, "should-not-be-used-0123456789ab")
    cfg = _lock_cfg(mode="off")
    assert launch_plan.lock_profile_source(cfg, settings=_SETTINGS) is cfg
    assert seen == []


def test_an_unreachable_registry_leaves_the_plan_unpinned_rather_than_failing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plan with no SF_* environment is a preview; it must still produce an id and a fanout."""
    _fake_discover(monkeypatch, RuntimeError("no credentials"))
    cfg = _lock_cfg()
    assert launch_plan.lock_profile_source(cfg, settings=_SETTINGS).compute.profile.source == "auto"


# --- feasibility: the one planning question the config cannot answer -----------


def _bt_cfg(**backtest: Any) -> RunConfig:
    bt = {"enabled": True, "n_folds": 2, "horizon": 28, "step": 28, **backtest}
    return _cfg(
        data={"source_table": "source_series_native", "horizon": 28},
        models=[_SPARK],
        backtest=bt,
    )


def test_feasibility_reports_fits_and_the_backtest_multiplier() -> None:
    lines = launch_plan.feasibility_lines(_bt_cfg(), [1460, 1460])
    joined = "\n".join(lines)
    assert "2 series x 1 models = 2 cells" in joined
    assert "fits: 6" in joined
    assert "2.942 whole-history fits per cell" in joined


def test_feasibility_names_every_fold_cohort_including_the_unscored_one() -> None:
    lines = launch_plan.feasibility_lines(_bt_cfg(), [1460, 1460, 220, 100])
    joined = "\n".join(lines)
    assert "2 of 2 folds: 2 series (50.0%) — all folds" in joined
    assert "1 of 2 folds: 1 series (25.0%) — reduced" in joined
    assert "0 of 2 folds: 1 series (25.0%) — UNSCORED" in joined
    assert "1 series get no folds at all" in joined


def test_feasibility_says_min_train_is_within_budget_when_the_panel_is_long() -> None:
    joined = "\n".join(launch_plan.feasibility_lines(_bt_cfg(), [1460] * 10))
    assert "is within budget" in joined and "moves the run_id" not in joined


def test_feasibility_suggests_a_lower_min_train_when_the_panel_is_short() -> None:
    # Nine 1460-row series and one 300-row: min_train=180 leaves the short one short.
    joined = "\n".join(launch_plan.feasibility_lines(_bt_cfg(min_train=250), [300] * 10))
    assert "is above what this panel supports" in joined
    assert "moves the run_id" in joined


def test_feasibility_says_change_the_geometry_when_no_min_train_helps() -> None:
    # 50 observations against horizon 28 + one step of 28: the geometry alone eats the series,
    # so no min_train — not even 1 — buys a second fold. Saying "lower min_train" here would be
    # advice that cannot work.
    joined = "\n".join(launch_plan.feasibility_lines(_bt_cfg(), [50] * 10))
    assert "reduce n_folds" in joined


def test_feasibility_with_backtesting_off_reports_the_cells_and_stops() -> None:
    joined = "\n".join(launch_plan.feasibility_lines(_cfg(models=[_SPARK]), [1460, 900]))
    assert "1.000 whole-history fits per cell" in joined
    assert "backtest: off" in joined
    assert "folds" not in joined.split("backtest: off")[0].split("\n")[-1]


def test_feasibility_report_degrades_to_a_line_when_the_panel_cannot_be_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A reporting verb that raises is one an operator stops running. No SF_* env is the common
    # case (a laptop with no environment), and it must not cost the caller their plan.
    def _boom(cfg: RunConfig, *, settings: Any = None) -> list[int]:
        raise ConfigError("no SF_* env")

    monkeypatch.setattr(launch_plan, "read_series_lengths", _boom)
    lines = launch_plan.feasibility_report(_bt_cfg())
    assert len(lines) == 1 and "could not read the source panel" in lines[0]


def test_feasibility_report_says_so_when_the_panel_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launch_plan, "read_series_lengths", lambda cfg, settings=None: [])
    assert launch_plan.feasibility_report(_bt_cfg()) == [
        "feasibility: the source panel returned no series"
    ]


# --- preflight_short_series: the one policy that refuses a run -----------------


def _counted(monkeypatch: pytest.MonkeyPatch, counts: list[int]) -> list[int]:
    """Patch the panel read to return ``counts``, and record that it was called."""
    calls: list[int] = []

    def _read(cfg: RunConfig, *, settings: Any = None) -> list[int]:
        calls.append(1)
        return counts

    monkeypatch.setattr(launch_plan, "read_series_lengths", _read)
    return calls


def test_the_preflight_refuses_the_run_when_a_series_cannot_hold_the_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The point of ``error``: an operator who asked for a comparable panel finds out at submit."""
    _counted(monkeypatch, [1460, 100])
    with pytest.raises(ConfigError, match="short_series='error'"):
        launch_plan.preflight_short_series(_bt_cfg(short_series="error"))


def test_the_preflight_passes_a_panel_that_holds_the_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _counted(monkeypatch, [1460, 1460])
    assert launch_plan.preflight_short_series(_bt_cfg(short_series="error")) is None


@pytest.mark.parametrize("policy", ["adapt", "overlap", "skip"])
def test_the_preflight_costs_nothing_under_every_other_policy(
    monkeypatch: pytest.MonkeyPatch, policy: str
) -> None:
    """A guard that queried BigQuery on every run would be a tax on the policies that never fire."""
    calls = _counted(monkeypatch, [100])
    assert launch_plan.preflight_short_series(_bt_cfg(short_series=policy)) is None
    assert calls == []


def test_the_preflight_costs_nothing_when_backtesting_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _counted(monkeypatch, [100])
    cfg = _cfg(data={"source_table": "source_series_native", "horizon": 28}, models=[_SPARK])
    assert launch_plan.preflight_short_series(cfg) is None
    assert calls == []


def test_an_unreadable_panel_warns_and_defers_rather_than_refusing_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``error`` means "stop if the panel is too short", not "stop if I could not tell".

    Refusing on a transient BigQuery failure would be a worse outcome than the one this guards
    against, and it would be unfixable from the config. The engine drivers hold the backstop, on a
    panel they have by then definitely read.
    """

    def _boom(cfg: RunConfig, *, settings: Any = None) -> list[int]:
        raise ConfigError("no SF_* env")

    monkeypatch.setattr(launch_plan, "read_series_lengths", _boom)
    assert launch_plan.preflight_short_series(_bt_cfg(short_series="error")) is None
