"""Offline tests for the registry-operations surface (`registry.ops`).

Everything here is the pure half — set arithmetic, SQL strings, plan formatting, and the BQML
model-name matcher `drop_run` depends on. The artifact-path arithmetic the destructive verbs
run on moved to `registry.artifacts` with the layout itself; its tests went with it, into
``test_registry_assembly.py``. The seven verbs themselves are GCP I/O and are
covered by the `@gcp` smokes (the artifact-prefix delete and `snapshot`); what is tested here is
every decision those verbs make *before* they touch anything.
"""

from __future__ import annotations

import pytest

from scale_forecasting.engines.bigquery_names import model_object_matches_run
from scale_forecasting.registry import artifacts, ops
from scale_forecasting.registry.ddl import REGISTRY_TABLE_NAMES, SOURCE_TABLE_NAMES

# --- blocking_runs -----------------------------------------------------------------


@pytest.mark.parametrize(
    "status", ["RUNNING", "PENDING", "running", "pending", "AWAITING_CAPACITY", "awaiting_capacity"]
)
def test_live_statuses_block(status):
    assert ops.blocking_runs({"r1": status}) == ("r1",)


def test_a_run_awaiting_capacity_cannot_be_dropped():
    """`LIVE_STATUSES` is a deny-list, so a live status left out of it reads as safe to delete.

    A run waiting for a stocked-out region has no runtime job and no recent signal, which is
    exactly what a stuck run looks like — and `drop_run` is destructive. The status has to be in
    the set on purpose; the default for anything unrecognised is "terminal and safe".
    """
    assert "AWAITING_CAPACITY" in ops.LIVE_STATUSES


@pytest.mark.parametrize("status", ["COMPLETED", "FAILED", "PARTIAL", "CANCELLED"])
def test_terminal_statuses_do_not_block(status):
    assert ops.blocking_runs({"r1": status}) == ()


def test_an_unknown_run_is_not_blocking():
    """No header row means the run cannot be in flight; `DropPlan.unknown` reports it instead."""
    assert ops.blocking_runs({"r1": None, "r2": "RUNNING"}) == ("r2",)


# --- render_delete_rows -----------------------------------------------------------


def test_delete_covers_every_registry_table_and_no_source_table():
    sql = ops.render_delete_rows("proj.reg")
    assert set(sql) == set(REGISTRY_TABLE_NAMES)
    assert not set(sql) & set(SOURCE_TABLE_NAMES)


def test_delete_binds_run_ids_as_a_parameter():
    """Ids are bound, never interpolated — the statement text carries no run id at all."""
    sql = ops.render_delete_rows("proj.reg")["run_registry"]
    assert sql == "DELETE FROM `proj.reg.run_registry` WHERE run_id IN UNNEST(@run_ids);"


def test_delete_accepts_a_table_subset():
    sql = ops.render_delete_rows("proj.reg", tables=["run_jobs"])
    assert set(sql) == {"run_jobs"}


# --- render_snapshot_sql ----------------------------------------------------------


def test_snapshot_clones_each_table_with_the_suffix():
    sql = ops.render_snapshot_sql("proj.reg", "proj.reg", "20260831")
    assert set(sql) == set(REGISTRY_TABLE_NAMES)
    assert sql["run_registry"] == (
        "CREATE SNAPSHOT TABLE IF NOT EXISTS `proj.reg.run_registry_20260831`\n"
        "CLONE `proj.reg.run_registry`;"
    )


def test_snapshot_can_target_another_dataset():
    sql = ops.render_snapshot_sql("proj.reg", "proj.archive", "before_migration")
    assert "`proj.archive.run_jobs_before_migration`" in sql["run_jobs"]
    assert "CLONE `proj.reg.run_jobs`" in sql["run_jobs"]


def test_snapshot_expiration_is_relative_to_creation():
    sql = ops.render_snapshot_sql("proj.reg", "proj.reg", "tmp", expiration_days=7)
    assert "TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)" in sql["run_registry"]


@pytest.mark.parametrize("days", [0, -1])
def test_snapshot_rejects_a_nonpositive_expiration(days):
    with pytest.raises(ValueError):
        ops.render_snapshot_sql("proj.reg", "proj.reg", "tmp", expiration_days=days)


# --- render_export_sql ------------------------------------------------------------


def test_export_defaults_to_parquet_under_a_per_table_prefix():
    sql = ops.render_export_sql("proj.reg", "gs://bucket/dump/")
    assert set(sql) == set(REGISTRY_TABLE_NAMES)
    body = sql["forecast_predictions"]
    assert "uri = 'gs://bucket/dump/forecast_predictions/forecast_predictions-*.parquet'" in body
    assert "format = 'PARQUET'" in body
    assert "overwrite = true" in body
    assert body.endswith("SELECT * FROM `proj.reg.forecast_predictions`;")


def test_export_json_uses_the_newline_delimited_form():
    body = ops.render_export_sql("proj.reg", "gs://bucket/dump", fmt="json")["run_registry"]
    assert "format = 'NEWLINE_DELIMITED_JSON'" in body
    assert "run_registry-*.json'" in body


def test_export_rejects_an_unknown_format():
    with pytest.raises(ValueError, match="unsupported export format"):
        ops.render_export_sql("proj.reg", "gs://bucket/dump", fmt="AVRO")


# --- plans and formatting ---------------------------------------------------------


def _prefix(run_id: str, n: int = 2, b: int = 1024) -> artifacts.ArtifactPrefix:
    return artifacts.ArtifactPrefix(run_id=run_id, object_count=n, byte_total=b)


def test_drop_plan_totals_roll_up_its_prefixes():
    plan = ops.DropPlan(
        registry="proj.reg",
        run_ids=("r1", "r2"),
        prefixes=(_prefix("r1", 3, 100), _prefix("r2", 4, 200)),
    )
    assert plan.object_count == 7
    assert plan.byte_total == 300
    assert not plan.is_empty


def test_a_plan_naming_only_unknown_runs_is_empty():
    plan = ops.DropPlan(registry="proj.reg", unknown=("nope",))
    assert plan.is_empty


def test_format_plan_names_every_run_and_the_gcs_total():
    plan = ops.DropPlan(
        registry="proj.reg",
        run_ids=("r1",),
        prefixes=(_prefix("r1", 2, 2048),),
        models=("sf_model_arima_plus_r1",),
        unknown=("gone",),
    )
    text = ops.format_plan(plan)
    assert "drop-run against registry proj.reg" in text
    assert "runs (1): r1" in text
    assert "sf_model_arima_plus_r1" in text
    assert "not in this registry (skipped): gone" in text
    assert "2 objects, 2.0 KiB" in text


def test_format_plan_shouts_about_an_in_flight_run():
    plan = ops.DropPlan(registry="proj.reg", blocked=("r1",))
    assert "IN FLIGHT — refusing: r1" in ops.format_plan(plan)


def test_format_sweep_plan_reports_the_scope_it_searched():
    plan = ops.SweepPlan(
        registry="proj.reg",
        root="gs://b/warehouse/artifacts/proj/reg",
        prefixes=(_prefix("orphan_1"),),
        known_runs=12,
    )
    text = ops.format_plan(plan)
    assert "artifact root: gs://b/warehouse/artifacts/proj/reg" in text
    assert "runs known to the registry: 12" in text
    assert "orphan prefixes (1): orphan_1" in text


def test_format_sweep_plan_with_nothing_to_do():
    plan = ops.SweepPlan(registry="proj.reg", root="gs://b/r", known_runs=3)
    assert "orphan prefixes (0): (none)" in ops.format_plan(plan)


def test_doctor_report_health_and_missing_tables():
    stats = tuple(ops.TableStat(t, 0) for t in REGISTRY_TABLE_NAMES)
    healthy = ops.DoctorReport(registry="proj.reg", artifact_root="gs://b/r", tables=stats)
    assert healthy.healthy
    assert healthy.missing_tables == ()
    assert "healthy" in ops.format_doctor(healthy)

    broken = ops.DoctorReport(
        registry="proj.reg",
        artifact_root="gs://b/r",
        tables=stats[:-1] + (ops.TableStat(stats[-1].table, None),),
    )
    assert not broken.healthy
    assert broken.missing_tables == (stats[-1].table,)
    assert "MISSING" in ops.format_doctor(broken)


def test_doctor_is_unhealthy_with_live_runs_or_orphans():
    stats = tuple(ops.TableStat(t, 1) for t in REGISTRY_TABLE_NAMES)
    live = ops.DoctorReport(
        registry="proj.reg", artifact_root="gs://b/r", tables=stats, live_runs=(("r1", "RUNNING"),)
    )
    assert not live.healthy
    assert "IN FLIGHT (1): r1 (RUNNING)" in ops.format_doctor(live)

    orphaned = ops.DoctorReport(
        registry="proj.reg", artifact_root="gs://b/r", tables=stats, orphans=(_prefix("x"),)
    )
    assert not orphaned.healthy
    assert "sweep_orphans" in ops.format_doctor(orphaned)


@pytest.mark.parametrize(
    ("n", "expected"), [(0, "0 B"), (512, "512 B"), (1536, "1.5 KiB"), (1024**3, "1.0 GiB")]
)
def test_human_bytes(n, expected):
    assert ops.human_bytes(n) == expected


# --- the BQML model matcher (the fourth orphan class) ------------------------------


def test_a_final_model_object_matches_its_run():
    assert model_object_matches_run("sf_model_arima_plus_run_abc", "run_abc")


def test_a_fold_model_object_matches_its_run():
    assert model_object_matches_run("sf_model_arima_plus_run_abc_f0", "run_abc")
    assert model_object_matches_run("sf_model_timesfm_run_abc_f12", "run_abc")


def test_a_model_object_from_another_run_does_not_match():
    assert not model_object_matches_run("sf_model_arima_plus_run_xyz", "run_abc")
    assert not model_object_matches_run("sf_model_arima_plus_run_abc2", "run_abc")


def test_a_non_product_model_object_never_matches():
    """Someone else's model in the same dataset must survive a drop-run."""
    assert not model_object_matches_run("their_model_run_abc", "run_abc")
    assert not model_object_matches_run("sf_model_arima_plus_run_abc", "run_ab")


def test_the_matcher_is_the_inverse_of_the_naming_rule():
    """Match what `_model_ref` actually renders, so the namer and the matcher cannot drift apart.

    This is the test that matters: `drop_run` finds a run's BQML objects by *name* (nothing records
    them), so a change to the naming rule that this matcher doesn't follow silently strands every
    model object a run creates.
    """
    from scale_forecasting.config import RunConfig
    from scale_forecasting.engines.bigquery_names import _model_ref
    from scale_forecasting.registry.ids import make_run_id

    cfg = RunConfig(
        run_name="ops matcher",
        data={"source_table": "source_series_native", "series_limit": 10},
        models=["arima_plus"],
    )
    other = RunConfig(
        run_name="ops matcher other",
        data={"source_table": "source_series_native", "series_limit": 10},
        models=["arima_plus"],
    )
    run_id, other_id = make_run_id(cfg), make_run_id(other)
    assert run_id != other_id

    for fold_id in (None, 0, 3):
        ref = _model_ref(cfg, "arima_plus", "proj.reg", fold_id=fold_id)
        model_id = ref.strip("`").rsplit(".", 1)[-1]
        assert model_object_matches_run(model_id, run_id)
        assert not model_object_matches_run(model_id, other_id)


# --- the CLI ⇄ SDK ⇄ ops seam ------------------------------------------------------
#
# One implementation, three entry points (G1). These tests hold the wiring: that each CLI
# subcommand reaches the verb it names with the flags the operator typed, and that the SDK class
# forwards the same way. The verbs' bodies are GCP I/O and are not called here — only the dispatch.


@pytest.fixture()
def spy(monkeypatch):
    """Replace every ops verb with a recorder, so `main`/`Registry` dispatch can be asserted."""
    calls: list[tuple[str, tuple, dict]] = []

    def record(name):
        def fn(*a, **kw):
            calls.append((name, a, kw))
            return {} if name in {"snapshot", "export"} else name

        return fn

    for verb in ("init", "doctor", "drop_run", "sweep_orphans", "snapshot", "export"):
        monkeypatch.setattr(ops, verb, record(verb))
    monkeypatch.setattr(ops, "format_doctor", lambda report: "ok")
    return calls


def test_cli_drop_run_previews_unless_told_otherwise(spy):
    ops.main(["drop-run", "r1", "r2"])
    assert spy == [("drop_run", (["r1", "r2"],), {"yes": False, "force": False})]


def test_cli_drop_run_passes_yes_and_force(spy):
    ops.main(["drop-run", "r1", "--yes", "--force"])
    assert spy[0][2] == {"yes": True, "force": True}


def test_cli_sweep_previews_unless_told_otherwise(spy):
    ops.main(["sweep-orphans"])
    assert spy == [("sweep_orphans", (), {"yes": False})]
    spy.clear()
    ops.main(["sweep-orphans", "--yes"])
    assert spy[0][2] == {"yes": True}


def test_cli_init_and_doctor(spy):
    ops.main(["init"])
    ops.main(["init", "--create-dataset"])
    ops.main(["doctor"])
    assert [c[0] for c in spy] == ["init", "init", "doctor"]
    assert spy[0][2] == {"create_dataset": False}
    assert spy[1][2] == {"create_dataset": True}


def test_cli_snapshot_and_export(spy):
    ops.main(["snapshot", "20260831", "--into", "proj.archive", "--expiration-days", "7"])
    ops.main(["export", "gs://bucket/dump", "--format", "JSON"])
    assert spy[0] == ("snapshot", ("20260831",), {"into": "proj.archive", "expiration_days": 7})
    assert spy[1] == ("export", ("gs://bucket/dump",), {"fmt": "JSON"})


def test_cli_rejects_an_unknown_export_format():
    """argparse `choices` catches it before any client is built."""
    with pytest.raises(SystemExit):
        ops.main(["export", "gs://bucket/dump", "--format", "AVRO"])


def test_cli_requires_a_verb():
    with pytest.raises(SystemExit):
        ops.main([])


def test_cli_has_no_wipe_verb():
    """Destructive-tier teardown deliberately left the product — it is `bq rm -r -f <dataset>`."""
    for absent in ("reset", "wipe", "drop-all"):
        with pytest.raises(SystemExit):
            ops.main([absent])


def test_the_product_ships_no_whole_registry_wipe_anywhere():
    """The tripwire for the decision, not just for this CLI.

    `reset.py` and a `drop_all` writer were removed when `registry.ops` landed: a whole-registry
    drop is `bq rm`, and giving it a product verb made it look supported and safe while it silently
    stranded every artifact it had just orphaned. Re-adding either would restore that trap quietly,
    so it fails here instead. The *pure* renderer stays — it is strings only and has no side effect.

    The sweep is over the whole `registry` package rather than one module, so splitting the writers
    up (or adding a new one) cannot smuggle the verb back in through a file the test never named.
    """
    import importlib
    import pkgutil

    from scale_forecasting import registry
    from scale_forecasting.registry import ddl

    for mod in pkgutil.iter_modules(registry.__path__):
        loaded = importlib.import_module(f"scale_forecasting.registry.{mod.name}")
        assert not hasattr(loaded, "drop_all"), f"{mod.name} ships a whole-registry wipe"
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("scale_forecasting.reset")
    assert callable(ddl.render_drop_tables), "the pure renderer is still wanted"


def test_sdk_registry_forwards_every_verb(spy):
    from scale_forecasting.sdk import Registry

    reg = Registry()
    reg.init(create_dataset=True)
    reg.doctor()
    reg.drop_run("r1", "r2", yes=True)
    reg.sweep_orphans()
    reg.snapshot("s1", expiration_days=3)
    reg.export("gs://b/d", fmt="JSON")

    assert [c[0] for c in spy] == [
        "init",
        "doctor",
        "drop_run",
        "sweep_orphans",
        "snapshot",
        "export",
    ]
    assert spy[2] == ("drop_run", (["r1", "r2"],), {"settings": None, "yes": True, "force": False})


def test_forecaster_hands_out_a_registry_on_the_same_settings():
    """`f.registry()` must carry the Forecaster's injected settings, not re-resolve from env."""
    from scale_forecasting.config import RunConfig
    from scale_forecasting.sdk import Forecaster, Registry
    from scale_forecasting.settings import Settings

    settings = Settings(
        project_id="proj",
        connection="proj.us.conn",
        warehouse_uri="gs://bucket/warehouse",
        dataset_id="source_ds",
        registry_dataset_id_override="registry_ds",
    )
    f = Forecaster(
        RunConfig(
            run_name="reg handle",
            data={"source_table": "source_series_native", "series_limit": 5},
            models=["theta"],
        ),
        settings=settings,
    )
    reg = f.registry()
    assert isinstance(reg, Registry)
    assert reg.dataset_ref == "proj.registry_ds"
    assert repr(reg) == "Registry(proj.registry_ds)"


# --- roll_up_job_statuses (the close-runs decision) ---------------------------------


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        (["COMPLETED", "COMPLETED"], "COMPLETED"),
        (["FAILED", "FAILED"], "FAILED"),
        (["CANCELLED", "CANCELLED"], "CANCELLED"),
        (["COMPLETED", "FAILED"], "PARTIAL"),
        (["COMPLETED", "CANCELLED"], "PARTIAL"),
        (["COMPLETED", "PARTIAL"], "PARTIAL"),
    ],
)
def test_roll_up_matches_the_status_a_finished_run_would_have_written(statuses, expected):
    # All green → COMPLETED, all failed → FAILED, mix → PARTIAL. This verb writes the status the run
    # itself failed to write, never a different one. The row-only reading is what close-runs falls
    # back to when the run's config is unreadable; `roll_up_against_plan` is the usual path.
    status, _reason = ops.roll_up_job_statuses(statuses)
    assert status == expected


@pytest.mark.parametrize("unsettled", ["RUNNING", "PENDING", "AWAITING_CAPACITY", "", None])
def test_roll_up_refuses_a_run_with_a_non_terminal_job(unsettled):
    """A live-looking job row means "go probe it", never "close it" (the load-bearing refusal).

    Only a runtime probe can tell a live job from a stale row. Closing on a non-terminal row is how
    an operator tidying the registry would silently mark a *running* job's run as finished.
    """
    status, reason = ops.roll_up_job_statuses(["COMPLETED", unsettled])
    assert status is None
    assert "not terminal" in reason


def test_roll_up_closes_a_header_with_no_job_rows_as_failed():
    # The common stuck shape: the driver wrote a header and died in the submit path before any
    # family row landed. Nothing completed, so not COMPLETED; nobody stopped it, so not CANCELLED.
    status, reason = ops.roll_up_job_statuses([])
    assert status == "FAILED"
    assert "never recorded a family" in reason


def test_roll_up_is_case_insensitive():
    assert ops.roll_up_job_statuses(["completed", "Completed"])[0] == "COMPLETED"


def test_close_runs_will_not_settle_a_run_that_is_still_waiting_for_capacity():
    """Stated on purpose, not left to fall out of a frozenset.

    ``AWAITING_CAPACITY`` refuses because it is simply not in `_TERMINAL_JOB_STATUSES` — free, and
    correct for the right reason: the family has neither finished nor failed, it is queued to try
    again, so closing the run would write a verdict on work still scheduled to happen. If someone
    ever "tidies" the terminal set by adding it, this fails and says why.
    """
    status, reason = ops.roll_up_job_statuses(["COMPLETED", "AWAITING_CAPACITY"])
    assert status is None
    assert "AWAITING_CAPACITY" in reason


# --- roll_up_against_plan (the same decision, with the run's own plan in hand) ------
#
# The row-only roll-up can only count rows that exist. A family that was planned and never submitted
# leaves none, so an abandoned run whose last family happened to finish reads as "every job
# COMPLETED". Two live runs closed that way on 2026-09-20, each claiming an ensemble that never ran.
# These pin the repair, and pin it to `job_outcome.combined_run_status` rather than to a second
# opinion about what the statuses mean.

_PLAN = ("statistical", "native", "ensemble")


def test_a_run_that_never_submitted_its_ensemble_does_not_close_as_completed():
    """The defect, written down. Both base families landed; the ensemble node never started.

    FAILED rather than PARTIAL because that is what the run itself would have written: an ensemble
    was requested and does not exist, so the requested output is incomplete. The reason names the
    family, because "FAILED" on a run whose forecasts are all present needs an explanation.
    """
    status, reason = ops.roll_up_against_plan(
        [("statistical", "COMPLETED"), ("native", "COMPLETED")],
        planned_families=_PLAN,
        ensemble_enabled=True,
    )
    assert status == "FAILED"
    assert "never recorded a row: ensemble" in reason


def test_a_missing_base_family_closes_partial_because_what_landed_is_still_usable():
    status, reason = ops.roll_up_against_plan(
        [("statistical", "COMPLETED"), ("ensemble", "COMPLETED")],
        planned_families=_PLAN,
        ensemble_enabled=True,
    )
    assert status == "PARTIAL"
    assert "never recorded a row: native" in reason


def test_an_ensemble_that_failed_fails_the_run_even_though_every_forecast_landed():
    """The second place the two readings had drifted, found while fixing the first.

    On the rows alone this is a mix of terminals and therefore PARTIAL. The run itself would have
    written FAILED, and for the same reason a *missing* ensemble does: an ensemble was asked for and
    does not exist. Whether the node failed or never started is not a difference the header cares
    about.
    """
    status, _reason = ops.roll_up_against_plan(
        [("statistical", "COMPLETED"), ("native", "COMPLETED"), ("ensemble", "FAILED")],
        planned_families=_PLAN,
        ensemble_enabled=True,
    )
    assert status == "FAILED"
    assert ops.roll_up_job_statuses(["COMPLETED", "COMPLETED", "FAILED"])[0] == "PARTIAL"


def test_the_plan_aware_roll_up_agrees_with_what_the_run_would_have_written():
    """One policy, in one place — the anti-drift test.

    The row-only roll-up used to cite a ``main._combined_status`` that no longer existed, and the
    two readings had quietly diverged. This asserts the equivalence directly rather than describing
    it in a docstring, so the next divergence fails a test instead of mis-closing a header.
    """
    from scale_forecasting.job_outcome import combined_run_status

    for rows in (
        [("statistical", "COMPLETED"), ("native", "COMPLETED")],
        [("statistical", "COMPLETED"), ("native", "FAILED")],
        [("statistical", "FAILED"), ("native", "FAILED")],
        [("statistical", "COMPLETED"), ("native", "COMPLETED"), ("ensemble", "FAILED")],
        [("statistical", "COMPLETED"), ("ensemble", "COMPLETED")],
    ):
        landed = dict(rows)
        expected = combined_run_status(
            landed | {f: None for f in _PLAN if f not in landed}, ensemble_enabled=True
        )
        status, _reason = ops.roll_up_against_plan(
            rows, planned_families=_PLAN, ensemble_enabled=True
        )
        assert status == expected, rows


def test_a_run_that_recorded_every_planned_family_is_unaffected_by_the_plan():
    rows = [("statistical", "COMPLETED"), ("native", "COMPLETED"), ("ensemble", "COMPLETED")]
    assert ops.roll_up_against_plan(rows, planned_families=_PLAN, ensemble_enabled=True) == (
        ops.roll_up_job_statuses([s for _f, s in rows])
    )


def test_with_no_plan_in_hand_it_is_exactly_the_row_only_roll_up():
    """The fallback path: a header whose ``raw_config`` is missing or no longer validates.

    An unreadable plan is not a reason to refuse to close an abandoned header — it only means we are
    back to the situation the row-only roll-up always handled alone.
    """
    rows = [("statistical", "COMPLETED"), ("native", "FAILED")]
    assert ops.roll_up_against_plan(rows, planned_families=(), ensemble_enabled=False) == (
        ops.roll_up_job_statuses([s for _f, s in rows])
    )


def test_a_present_non_terminal_row_still_refuses_even_with_a_family_missing():
    """The load-bearing refusal is unchanged and comes first — the plan never overrides it."""
    status, reason = ops.roll_up_against_plan(
        [("statistical", "RUNNING")], planned_families=_PLAN, ensemble_enabled=True
    )
    assert status is None
    assert "not terminal" in reason


def test_a_missing_row_never_refuses_to_close():
    """Missing and non-terminal must stay different answers, or the verb stops working.

    Refusing on a *missing* family would strand exactly the abandoned runs close-runs exists to
    repair: their whole signature is a family that never wrote anything at all.
    """
    status, _reason = ops.roll_up_against_plan(
        [("statistical", "COMPLETED")], planned_families=_PLAN, ensemble_enabled=True
    )
    assert status is not None


def test_a_cancelled_run_stays_cancelled_even_though_it_skipped_its_ensemble():
    """Cancelling is the ordinary way a planned family ends up with no row.

    `combined_run_status` would call this FAILED — it never sees a cancelled run, because cancelling
    skips the finalizer that calls it. CANCELLED is the truer word and the one the operator stopped
    the run to get.
    """
    status, _reason = ops.roll_up_against_plan(
        [("statistical", "CANCELLED"), ("native", "CANCELLED")],
        planned_families=_PLAN,
        ensemble_enabled=True,
    )
    assert status == "CANCELLED"


def test_a_header_with_no_job_rows_is_still_failed_when_a_plan_exists():
    status, reason = ops.roll_up_against_plan([], planned_families=_PLAN, ensemble_enabled=True)
    assert status == "FAILED"
    assert "never recorded a family" in reason


def test_a_second_attempt_never_upgrades_a_family_off_its_first_row():
    """Retries share the family and differ only in ``job_id``, so a family can hold two rows.

    Folding them the other way — letting a later COMPLETED erase an earlier FAILED — is the same lie
    the repair token exists to prevent, just arriving through a different door.
    """
    status, _reason = ops.roll_up_against_plan(
        [("statistical", "FAILED"), ("statistical", "COMPLETED"), ("native", "COMPLETED")],
        planned_families=("statistical", "native"),
        ensemble_enabled=False,
    )
    assert status == "PARTIAL"


# --- a repair must not close a family it only partly repaired ----------------------
#
# ``roll_up_job_statuses`` is deliberately untouched by the repair work: its contract is that it
# writes the status the run itself would have written, and "a repair succeeded" is not one of those.
# What changes is *which rows it is handed*, and that is decided upstream by `v_run_jobs`' grain.
# These tests pin the grain, because the whole repair-token design rests on it.


def _v_run_jobs(rows):
    """The rows `v_run_jobs` yields from ``run_jobs`` — one per (run_id, family), top attempt wins.

    Mirrors the view's ``QUALIFY ROW_NUMBER() OVER (PARTITION BY run_id, family ORDER BY attempt
    DESC …) = 1``. Reproduced rather than queried so the arithmetic that makes a repair token
    necessary is checkable offline; `test_the_view_still_partitions_on_family` keeps the two honest.
    """
    top = {}
    for family, attempt, status in rows:
        if family not in top or attempt > top[family][0]:
            top[family] = (attempt, status)
    return {f: s for f, (_a, s) in top.items()}


def test_the_view_still_partitions_on_family() -> None:
    """If the view's grain ever loses ``family``, the helper above is a lie and so is the design."""
    from scale_forecasting.registry.views import _VIEW_BODIES

    assert "PARTITION BY run_id, family ORDER BY attempt DESC" in _VIEW_BODIES["v_run_jobs"]


@pytest.mark.parametrize(
    ("base_status", "repair_status"),
    [("FAILED", "COMPLETED"), ("COMPLETED", "FAILED")],
)
def test_a_repair_leaves_the_status_of_the_family_it_repaired_alone(base_status, repair_status):
    """Both directions, because the guarantee is about the *row*, not about which outcome is nicer.

    A repair that succeeds must not promote a family that failed; a repair that fails must not
    demote a family that succeeded. Under a distinct token neither can happen — they are different
    rows of the view — and the run rolls up to PARTIAL, which is what a partly-repaired run is.
    """
    visible = _v_run_jobs(
        [("statistical", 1, base_status), ("statistical_repair", 1, repair_status)]
    )
    assert visible["statistical"] == base_status
    assert visible["statistical_repair"] == repair_status
    assert ops.roll_up_job_statuses(list(visible.values()))[0] == "PARTIAL"


def test_without_the_token_a_forty_cell_repair_would_close_a_hundred_thousand_cell_family():
    """The bug the repair token exists to prevent, written down so it stays prevented.

    Filed as attempt 2 of ``statistical``, a successful repair is the only ``statistical`` row the
    view returns — the FAILED attempt disappears and the run reads COMPLETED, claiming work that was
    never produced.
    """
    visible = _v_run_jobs([("statistical", 1, "FAILED"), ("statistical", 2, "COMPLETED")])
    assert visible == {"statistical": "COMPLETED"}
    assert ops.roll_up_job_statuses(list(visible.values()))[0] == "COMPLETED"


def test_a_repair_still_in_flight_keeps_the_run_open():
    """A RUNNING repair row is a non-terminal job like any other — close-runs refuses the whole run.

    Free, and correct for the right reason: the repair's cells are neither landed nor abandoned, so
    a header written now would be a verdict on work still happening.
    """
    visible = _v_run_jobs([("statistical", 1, "FAILED"), ("statistical_repair", 1, "RUNNING")])
    status, reason = ops.roll_up_job_statuses(list(visible.values()))
    assert status is None and "not terminal" in reason


# --- ClosePlan and its preview -----------------------------------------------------


def _candidate(run_id, new_status, reason, *, jobs=("COMPLETED",)):
    return ops.CloseCandidate(
        run_id=run_id,
        header_status="RUNNING",
        job_statuses=tuple(jobs),
        new_status=new_status,
        reason=reason,
    )


def test_close_plan_splits_closable_from_skipped():
    plan = ops.ClosePlan(
        registry="p.d",
        candidates=(
            _candidate("a", "COMPLETED", "every job COMPLETED"),
            _candidate("b", None, "1 job status(es) not terminal: RUNNING"),
        ),
    )
    assert [c.run_id for c in plan.closable] == ["a"]
    assert [c.run_id for c in plan.skipped] == ["b"]
    assert not plan.is_empty


def test_close_plan_with_nothing_closable_is_empty():
    # "Empty" means nothing to *do* — a plan that is all skips must not execute, but it must still
    # print, because the skips are the part the operator has to act on.
    plan = ops.ClosePlan(
        registry="p.d", candidates=(_candidate("b", None, "not terminal: RUNNING"),)
    )
    assert plan.is_empty
    assert plan.skipped


def test_format_close_plan_shows_the_transition_and_the_reason():
    plan = ops.ClosePlan(
        registry="p.d",
        candidates=(
            _candidate("run-a", "COMPLETED", "every job COMPLETED"),
            _candidate("run-b", None, "1 job status(es) not terminal: RUNNING"),
        ),
        unknown=("run-c",),
    )
    text = ops.format_close_plan(plan)
    assert "run-a  RUNNING -> COMPLETED" in text
    assert "every job COMPLETED" in text
    # The skipped run and its reason must be visible, not summarized into a count.
    assert "left alone (1)" in text
    assert "run-b" in text and "not terminal: RUNNING" in text
    assert "not stuck (skipped): run-c" in text


def test_format_close_plan_with_no_stuck_headers():
    assert "no stuck headers" in ops.format_close_plan(ops.ClosePlan(registry="p.d"))
