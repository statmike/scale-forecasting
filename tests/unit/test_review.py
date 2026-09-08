"""Offline tests for the run-inspection layer (`scale_forecasting.review`).

Same shape as ``test_sdk.py``: the pure assembly/derivation functions are exercised directly with
hand-built reader dicts, the I/O entry points (`monitor_run`/`review_run`) are covered by
monkeypatching the `registry.reads` / `registry.jobs` readers, and the plots get a headless (Agg)
smoke check — populated and empty. No GCP, no matplotlib display.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

import scale_forecasting as sf
from scale_forecasting import review as R
from scale_forecasting.config import RunConfig
from scale_forecasting.settings import Settings

_SETTINGS = Settings(
    project_id="proj-x",
    connection="proj-x.us-central1.conn",
    warehouse_uri="gs://bkt/warehouse",
)


def _cfg(**over: Any) -> RunConfig:
    base: dict[str, Any] = {
        "run_name": "review test",
        "data": {"source_table": "source_series_native", "horizon": 7, "series_limit": 10},
        "models": ["theta", "xgboost", "arima_plus"],
        "ensemble": {"enabled": True, "strategies": ["mean", "median"]},
    }
    base.update(over)
    return RunConfig.model_validate(base)


def _model(
    model_type: str, family: str, *, ens: str | None = None, score: float | None = 0.2, **over: Any
) -> R.ModelReview:
    base: dict[str, Any] = dict(
        model_type=model_type,
        family=family,
        ensemble_id=ens,
        is_ensemble=ens is not None,
        compute_engine="spark",
        n_series=10,
        score=score,
    )
    base.update(over)
    return R.ModelReview(**base)


# --- family_of -----------------------------------------------------------------


def test_family_of_uses_ensemble_then_registry_then_unknown() -> None:
    assert R.family_of("ensemble_mean", "digest") == "ensemble"  # ensemble_id wins
    assert R.family_of("arima_plus") == "native"  # native model registry family
    assert R.family_of("theta") == "statistical"
    assert R.family_of("not_a_real_model") == "unknown"  # deregistered name → graceful


# --- bests + ensemble lift -----------------------------------------------------


def test_best_overall_and_per_family_ignore_ensembles_and_unscored() -> None:
    models = [
        _model("theta", "statistical", score=0.20),
        _model("holtwinters", "statistical", score=0.18),  # better statistical
        _model("xgboost", "ml", score=0.25),
        _model("ensemble_median", "ensemble", ens="d", score=0.10),  # best score, but ensemble
        _model("naive_mean", "statistical", score=None),  # unscored → ignored
    ]
    assert R.best_overall(models).model_type == "holtwinters"  # not the ensemble
    per = R.best_per_family(models)
    assert {k: v.model_type for k, v in per.items()} == {
        "statistical": "holtwinters",
        "ml": "xgboost",
    }


def test_best_overall_none_when_no_scored_base_model() -> None:
    only_ensembles = [_model("ensemble_mean", "ensemble", ens="d", score=0.1)]
    assert R.best_overall(only_ensembles) is None
    assert R.ensemble_lift(only_ensembles) == []  # no base champ to compare against


def test_ensemble_lift_measures_gain_over_best_base_model() -> None:
    models = [
        _model("theta", "statistical", score=0.20),  # best base = the champ
        _model("xgboost", "ml", score=0.30),
        _model("ensemble_mean", "ensemble", ens="d", score=0.15),  # beats base by 0.05
        _model("ensemble_worse", "ensemble", ens="d", score=0.25),  # worse than base
    ]
    lifts = R.ensemble_lift(models)
    assert [x.model_type for x in lifts] == ["ensemble_mean", "ensemble_worse"]  # best lift first
    top = lifts[0]
    assert top.best_base_model == "theta" and top.best_base_score == 0.20
    assert round(top.lift, 4) == 0.05 and round(top.lift_pct, 4) == 0.25
    assert lifts[1].lift < 0  # a worse ensemble reads as negative lift


# --- _assemble_progress (pure) -------------------------------------------------


def test_assemble_progress_rolls_cells_up_to_families_against_expected() -> None:
    cfg = _cfg()  # 10 series; statistical=[theta], ml=[xgboost], native=[arima_plus], ensemble x2
    summary = {"status": "RUNNING", "n_series": 10}
    jobs = [
        {
            "family": "statistical",
            "runtime": "spark",
            "hardware": "cpu",
            "status": "COMPLETED",
            "runtime_seconds": 12.0,
        },
        {
            "family": "ml",
            "runtime": "spark",
            "hardware": "cpu",
            "status": "RUNNING",
            "runtime_seconds": None,
        },
    ]
    progress = [
        {"model_type": "theta", "ensemble_id": None, "n_cells_done": 10, "mean_fit_seconds": 0.5},
        {"model_type": "xgboost", "ensemble_id": None, "n_cells_done": 4, "mean_fit_seconds": 2.0},
    ]
    rp = R._assemble_progress("rid", summary, cfg, jobs, progress)

    by_family = {f.family: f for f in rp.families}
    assert list(by_family) == ["statistical", "ml", "native", "ensemble"]  # DAG order, ens last
    stat = by_family["statistical"]
    assert stat.n_expected == 10 and stat.n_done == 10 and stat.fraction == 1.0
    assert stat.status == "COMPLETED" and stat.avg_fit_seconds == 0.5
    ml = by_family["ml"]
    assert ml.n_expected == 10 and ml.n_done == 4 and ml.fraction == 0.4 and ml.status == "RUNNING"
    ens = by_family["ensemble"]
    assert ens.n_expected == 20 and ens.n_done == 0  # 10 series x 2 strategies, none landed
    # roll-up: expected = 10 + 10 + 10 + 20 = 50; done = 14
    assert rp.n_expected == 50 and rp.n_done == 14 and rp.status == "RUNNING"


def test_assemble_progress_cell_weighted_mean_fit_across_models_in_a_family() -> None:
    cfg = _cfg(models=["theta", "holtwinters"], ensemble={"enabled": False})
    progress = [
        {"model_type": "theta", "ensemble_id": None, "n_cells_done": 10, "mean_fit_seconds": 1.0},
        {
            "model_type": "holtwinters",
            "ensemble_id": None,
            "n_cells_done": 30,
            "mean_fit_seconds": 2.0,
        },
    ]
    rp = R._assemble_progress("rid", {"n_series": 10}, cfg, [], progress)
    stat = next(f for f in rp.families if f.family == "statistical")
    # (1.0*10 + 2.0*30) / 40 = 1.75 — weighted by landed cells, not a flat average of 1.5
    assert stat.avg_fit_seconds == 1.75 and stat.n_done == 40


def test_assemble_progress_unknown_series_count_yields_none_fractions() -> None:
    cfg = _cfg(data={"source_table": "source_series_native", "horizon": 7})  # no series_limit
    rp = R._assemble_progress("rid", None, cfg, [], [])
    assert rp.n_series is None
    assert all(f.n_expected is None and f.fraction is None for f in rp.families)
    assert rp.n_expected is None and rp.fraction is None


def test_a_repair_job_is_listed_after_the_family_it_repairs() -> None:
    """`probes.reconcile` and ``--cancel`` read this snapshot and nothing else.

    Leaving a repair job out of it would leave a live job neither of them can see or stop, which is
    the data-integrity property the cancel path is built on.
    """
    cfg = _cfg(models=["theta", "xgboost"], ensemble={"enabled": False})
    jobs = [
        {"family": "statistical", "runtime": "spark", "status": "FAILED"},
        {"family": "statistical_repair", "runtime": "spark", "status": "RUNNING"},
        {"family": "ml", "runtime": "spark", "status": "COMPLETED"},
    ]
    rp = R._assemble_progress("rid", {"n_series": 10}, cfg, jobs, [])
    assert [f.family for f in rp.families] == ["statistical", "ml", "statistical_repair"]
    repair = rp.families[-1]
    assert repair.status == "RUNNING" and repair.runtime == "spark"
    assert repair.models == ("theta",)  # the family's models, so a readout can name the subject


def test_a_repair_job_carries_no_denominator_and_does_not_move_the_run_total() -> None:
    """A repair's row records no subset size, so any expected count for it would be invented.

    The whole family's count is the tempting wrong answer twice over: it would report a finished
    forty-cell repair as 0.04% done, and it would add a second copy of a denominator the base
    family already contributed to the run-level fraction.
    """
    cfg = _cfg(models=["theta"], ensemble={"enabled": False})
    base = [{"family": "statistical", "status": "FAILED"}]
    progress = [
        {"model_type": "theta", "ensemble_id": None, "n_cells_done": 6, "mean_fit_seconds": 1.0}
    ]
    without = R._assemble_progress("rid", {"n_series": 10}, cfg, base, progress)
    with_repair = R._assemble_progress(
        "rid", {"n_series": 10}, cfg, [*base, {"family": "statistical_repair"}], progress
    )
    repair = with_repair.families[-1]
    assert repair.n_expected is None and repair.n_done == 0 and repair.fraction is None
    assert (with_repair.n_expected, with_repair.n_done) == (without.n_expected, without.n_done)
    assert with_repair.fraction == without.fraction == 0.6


def test_assemble_progress_no_config_is_status_only_snapshot() -> None:
    rp = R._assemble_progress("rid", {"status": "PENDING"}, None, [], [])
    assert rp.status == "PENDING" and rp.families == () and rp.n_done == 0
    assert rp.n_expected is None and rp.fraction is None
    assert rp.probe is None  # nothing was escalated


# --- quiet time (the free half of the probe convergence) -----------------------
#
# The age of a family's last registry signal, derived from rows the monitor already reads — no
# runtime call. It is the only thing that distinguishes a dead job from a slow one on a frozen bar,
# and it is what `probes.reconcile._is_stale` thresholds, so the row-parsing lives here and only
# here.

_AT = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def _one_job(**row: Any) -> R.FamilyProgress:
    cfg = _cfg(models=["theta"], ensemble={"enabled": False})
    rp = R._assemble_progress("rid", None, cfg, [{"family": "statistical", **row}], [], now=_AT)
    return next(f for f in rp.families if f.family == "statistical")


def test_quiet_seconds_measures_age_of_the_last_signal() -> None:
    fp = _one_job(status="RUNNING", started_at=_AT - timedelta(seconds=90))
    assert fp.last_signal_at == _AT - timedelta(seconds=90)
    assert fp.quiet_seconds == 90.0


def test_quiet_seconds_prefers_the_latest_signal() -> None:
    # ended_at wins over the (older) started_at/created_at when present.
    fp = _one_job(
        status="RUNNING",
        created_at=_AT - timedelta(seconds=5000),
        started_at=_AT - timedelta(seconds=4000),
        ended_at=_AT - timedelta(seconds=10),
    )
    assert fp.quiet_seconds == 10.0


def test_quiet_seconds_parses_an_iso_string_and_assumes_utc() -> None:
    # A reader dict (or any JSON-shaped row) carries strings, not datetimes; a naive one is UTC.
    assert _one_job(status="RUNNING", started_at="2026-01-01T11:00:00+00:00").quiet_seconds == 3600
    assert _one_job(status="RUNNING", started_at="2026-01-01T11:00:00").quiet_seconds == 3600


def test_quiet_seconds_is_none_when_unknown() -> None:
    # No job row at all, an unparseable timestamp, and a non-timestamp value all mean "no evidence
    # of silence" — never a zero age, which would read as "signalled just now".
    assert _one_job(status="RUNNING").quiet_seconds is None
    assert _one_job(status="RUNNING", started_at="not-a-timestamp").quiet_seconds is None
    assert _one_job(status="RUNNING", started_at=17).quiet_seconds is None
    cfg = _cfg(models=["theta"], ensemble={"enabled": False})
    no_row = R._assemble_progress("rid", None, cfg, [], [], now=_AT).families[0]
    assert no_row.last_signal_at is None and no_row.quiet_seconds is None


def test_the_device_verdict_rides_along_from_the_job_row() -> None:
    # It comes off `v_run_jobs.device_verdict`, which the view unpacks from the job's telemetry.
    # A CPU family has no verdict and must read None rather than inventing a reassuring one.
    assert _one_job(status="COMPLETED", device_verdict="ENGAGED_IDLE").device_verdict == (
        "ENGAGED_IDLE"
    )
    assert _one_job(status="COMPLETED").device_verdict is None


# --- _assemble_review (pure) ---------------------------------------------------


def _agg(model_type: str, ens: str | None, wape: float, **over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "model_type": model_type,
        "ensemble_id": ens,
        "compute_engine": "spark",
        "n_series": 10,
        "mean_fit_seconds": 1.0,
        "mean_wape": wape,
        "p10_wape": wape - 0.05,
        "p50_wape": wape,
        "p90_wape": wape + 0.05,
    }
    base.update(over)
    return base


def test_assemble_review_from_aggregates_sorts_and_derives() -> None:
    aggs = [
        _agg("xgboost", None, 0.30),
        _agg("theta", None, 0.20),
        _agg("ensemble_mean", "d", 0.15),
    ]
    lb = [
        {
            "model_type": "theta",
            "ensemble_id": None,
            "median_fit_seconds": 1.1,
            "no_artifact_rate": 0.0,
        }
    ]
    rr = R._assemble_review(
        "rid", {"status": "COMPLETED"}, "wape", 10, lb, aggs, {"theta": 1400, "xgboost": 0}
    )

    assert [m.model_type for m in rr.models] == ["ensemble_mean", "theta", "xgboost"]  # best first
    assert rr.best_overall.model_type == "theta"  # best *base* (ensemble excluded)
    assert rr.decision_metric == "wape" and rr.status == "COMPLETED"
    theta = next(m for m in rr.models if m.model_type == "theta")
    assert theta.score == 0.20 and round(theta.metric_p10["wape"], 4) == 0.15  # panel populated
    assert theta.median_fit_seconds == 1.1 and theta.n_predictions == 1400  # merged lb + counts
    xgb = next(m for m in rr.models if m.model_type == "xgboost")
    assert xgb.n_predictions == 0  # scored metadata but no forecasts
    assert [e.model_type for e in rr.ensembles] == ["ensemble_mean"]
    assert round(rr.ensemble_lift[0].lift, 4) == 0.05


def test_assemble_review_falls_back_to_leaderboard_when_no_aggregates() -> None:
    lb = [
        {
            "model_type": "theta",
            "ensemble_id": None,
            "compute_engine": "spark",
            "n_cells": 10,
            "mean_wape": 0.22,
            "median_fit_seconds": 1.0,
            "no_artifact_rate": 0.0,
        },
        {
            "model_type": "naive_mean",
            "ensemble_id": None,
            "compute_engine": "spark",
            "n_cells": 10,
            "mean_wape": None,
            "median_fit_seconds": 0.1,
            "no_artifact_rate": 0.0,
        },
    ]
    rr = R._assemble_review("rid", {"status": "COMPLETED"}, "wape", 10, lb, [], {})
    assert [m.model_type for m in rr.models] == ["theta", "naive_mean"]  # unscored sorts last
    assert rr.best_overall.model_type == "theta" and rr.models[0].score == 0.22
    assert rr.models[0].metric_means == {}  # no panel without aggregates


# --- cohorts: the panel behind the score ---------------------------------------


def _cohort_row(
    model: str,
    status: str | None,
    folds: int | None,
    n: int,
    refit: str | None = None,
    gap: float | None = None,
) -> dict[str, Any]:
    """One `v_backtest_coverage` row: a cohort of ``n`` series for one model."""
    return {
        "model_type": model,
        "ensemble_id": None,
        "backtest_status": status,
        "n_folds_achieved": folds,
        "n_series": n,
        "backtest_refit": refit,
        "mean_staleness_gap": gap,
    }


def test_cohorts_sum_the_split_rows_back_into_one_per_model() -> None:
    # The view emits one row per (status, achieved-folds); a ragged panel therefore arrives split.
    cohorts = R._cohorts_by_model(
        [
            _cohort_row("theta", "full", 3, 40),
            _cohort_row("theta", "reduced", 2, 25),
            _cohort_row("theta", "reduced", 1, 10),
            _cohort_row("theta", "unscored", None, 5),
        ]
    )
    theta = cohorts[("theta", None)]
    assert theta.n_series == 80  # every cohort counts toward the panel
    assert (theta.n_full, theta.n_reduced, theta.n_unscored) == (40, 35, 5)
    # The histogram keeps the shape of the raggedness, ordered, and leaves out the NULL-fold rows.
    assert theta.fold_histogram == {1: 10, 2: 25, 3: 40}


def test_the_cohort_says_how_the_panel_was_scored_not_just_how_much_of_it() -> None:
    # A frozen run where two thirds of the panel really was frozen and the rest fell back to a
    # refit. Both facts have to survive the fold-up: the counts say the model is not answering one
    # question, and the gap says what freezing cost the part that was frozen.
    cohorts = R._cohorts_by_model(
        [
            _cohort_row("sarimax", "full", 3, 60, refit="recondition", gap=0.03),
            _cohort_row("sarimax", "full", 3, 30, refit="unsupported", gap=None),
        ]
    )
    sarimax = cohorts[("sarimax", None)]
    assert sarimax.refit_modes == {"recondition": 60, "unsupported": 30}
    assert sarimax.staleness_gap == pytest.approx(0.03)


def test_the_staleness_gap_is_weighted_by_cohort_size() -> None:
    # The view already averaged within each cohort, so a plain mean of the cohort means would let
    # a two-series row weigh as much as a two-thousand-series one.
    cohorts = R._cohorts_by_model(
        [
            _cohort_row("theta", "full", 3, 900, refit="extrapolate", gap=0.10),
            _cohort_row("theta", "reduced", 1, 100, refit="extrapolate", gap=0.50),
        ]
    )
    assert cohorts[("theta", None)].staleness_gap == pytest.approx(0.14)


def test_a_refit_run_has_no_staleness_gap_rather_than_a_zero_one() -> None:
    # Zero would read as "refitting buys nothing", which is a finding. No control arm ran.
    cohorts = R._cohorts_by_model([_cohort_row("theta", "full", 3, 40, refit="per_fold")])
    theta = cohorts[("theta", None)]
    assert theta.staleness_gap is None
    assert theta.refit_modes == {"per_fold": 40}


def test_a_null_backtest_status_means_never_requested_not_failed() -> None:
    # NULL is "the run did not ask for a backtest", which is not the same fact as "one was
    # attempted and produced nothing" — collapsing them would invent failures on every run that
    # simply had backtesting off.
    cohorts = R._cohorts_by_model([_cohort_row("theta", None, None, 12)])
    theta = cohorts[("theta", None)]
    assert theta.n_not_requested == 12
    assert (theta.n_failed, theta.n_unscored, theta.n_series) == (0, 0, 12)


def test_an_unrecognised_status_still_counts_toward_the_panel() -> None:
    # A status this code does not know about must not silently shrink n_series — the total is the
    # panel, and a panel that quietly loses series is worse than one with an unfamiliar bucket.
    cohorts = R._cohorts_by_model([_cohort_row("theta", "quarantined", 1, 7)])
    assert cohorts[("theta", None)].n_series == 7


def test_assemble_review_hangs_the_cohort_and_pooled_score_on_each_model() -> None:
    aggs = [_agg("theta", None, 0.20), _agg("xgboost", None, 0.30)]
    coverage = [_cohort_row("theta", "full", 3, 100), _cohort_row("xgboost", "reduced", 1, 20)]
    comparable = [
        {"model_type": "theta", "ensemble_id": None, "pooled_wape": 0.19, "n_series": 100},
        {"model_type": "xgboost", "ensemble_id": None, "pooled_wape": 0.11, "n_series": 20},
    ]
    rr = R._assemble_review(
        "rid", {"status": "COMPLETED"}, "wape", 100, [], aggs, {}, coverage, comparable
    )
    theta = next(m for m in rr.models if m.model_type == "theta")
    xgb = next(m for m in rr.models if m.model_type == "xgboost")
    # This is the whole reason the cohort rides along: xgboost has the better pooled score and the
    # worse claim to it — 20 series of one fold against theta's 100 of three. The ranking cannot
    # say that; the two numbers beside it can.
    assert xgb.pooled_wape < theta.pooled_wape
    assert (xgb.n_comparable_series, theta.n_comparable_series) == (20, 100)
    assert theta.cohort.n_full == 100 and xgb.cohort.n_reduced == 20


def test_a_model_with_no_coverage_row_keeps_none_rather_than_a_zero_cohort() -> None:
    # "No backtest was run" and "a cohort of zero series" are different facts, and only the first
    # is true of a run with backtesting off. A zeroed cohort would read as a total failure.
    rr = R._assemble_review(
        "rid", {"status": "COMPLETED"}, "wape", 10, [], [_agg("theta", None, 0.2)], {}
    )
    assert rr.models[0].cohort is None
    assert rr.models[0].pooled_wape is None and rr.models[0].n_comparable_series is None


def test_cohorts_key_on_the_ensemble_id_so_two_configs_stay_apart() -> None:
    rows = [
        {**_cohort_row("ensemble_mean", "full", 3, 50), "ensemble_id": "e1"},
        {**_cohort_row("ensemble_mean", "reduced", 1, 8), "ensemble_id": "e2"},
    ]
    cohorts = R._cohorts_by_model(rows)
    assert cohorts[("ensemble_mean", "e1")].n_full == 50
    assert cohorts[("ensemble_mean", "e2")].n_reduced == 8


# --- I/O entry points ----------------------------------------------------------


def test_monitor_run_composes_readers(monkeypatch: Any) -> None:
    from scale_forecasting.registry import jobs, reads

    cfg = _cfg()
    seen: dict[str, Any] = {}

    def _summary(rid: str, *, settings: Any = None) -> dict[str, Any]:
        seen["run_id"] = rid
        seen["settings"] = settings
        return {"status": "RUNNING", "n_series": 10}

    monkeypatch.setattr(reads, "read_run_summary", _summary)
    monkeypatch.setattr(reads, "read_run_config", lambda rid, *, settings=None: cfg.model_dump())
    monkeypatch.setattr(
        jobs,
        "read_run_jobs",
        lambda rid, *, settings=None: [
            {
                "family": "statistical",
                "runtime": "spark",
                "hardware": "cpu",
                "status": "RUNNING",
                "runtime_seconds": None,
            }
        ],
    )
    monkeypatch.setattr(
        reads,
        "read_progress",
        lambda rid, *, settings=None: [
            {"model_type": "theta", "ensemble_id": None, "n_cells_done": 5, "mean_fit_seconds": 0.5}
        ],
    )

    rp = R.monitor_run("rid", settings=_SETTINGS)
    assert seen == {"run_id": "rid", "settings": _SETTINGS}  # readers get id + injected settings
    assert rp.status == "RUNNING" and rp.n_series == 10
    stat = next(f for f in rp.families if f.family == "statistical")
    assert stat.n_done == 5 and stat.n_expected == 10


def test_monitor_run_status_only_when_config_missing(monkeypatch: Any) -> None:
    from scale_forecasting.registry import reads

    monkeypatch.setattr(
        reads, "read_run_summary", lambda rid, *, settings=None: {"status": "PENDING"}
    )
    monkeypatch.setattr(reads, "read_run_config", lambda rid, *, settings=None: None)
    rp = R.monitor_run("rid", settings=_SETTINGS)
    assert rp.status == "PENDING" and rp.families == ()


def test_monitor_run_with_probe_reuses_the_probe_reader_and_attaches_the_report(
    monkeypatch: Any,
) -> None:
    # probe=True must not re-read the registry: it delegates to the probe's single read+escalate
    # pass and keeps both halves — the progress it built and the report it reconciled.
    from scale_forecasting.probes import reconcile
    from scale_forecasting.registry import jobs, reads

    for module, name in (
        (reads, "read_run_summary"),
        (reads, "read_run_config"),
        (jobs, "read_run_jobs"),
        (reads, "read_progress"),
    ):
        monkeypatch.setattr(module, name, _never_called(name))

    progress = R.RunProgress("rid", "RUNNING", 10, (), 0, None, None)
    report = reconcile.ProbeReport("rid", "RUNNING", True, (), False)
    seen: dict[str, Any] = {}

    def _read_and_probe(rid: str, *, job: Any, settings: Any, stale_after_s: Any) -> Any:
        seen.update(run_id=rid, job=job, settings=settings, stale_after_s=stale_after_s)
        return progress, report, []

    monkeypatch.setattr(reconcile, "_read_and_probe", _read_and_probe)

    rp = R.monitor_run("rid", probe=True, stale_after_s=60.0, settings=_SETTINGS)
    assert rp.probe is report and rp.status == "RUNNING"
    assert seen == {"run_id": "rid", "job": None, "settings": _SETTINGS, "stale_after_s": 60.0}


def _never_called(name: str) -> Any:
    def _fail(*_a: Any, **_k: Any) -> Any:
        raise AssertionError(f"{name} must not be read twice when probing")

    return _fail


def test_review_run_composes_readers(monkeypatch: Any) -> None:
    from scale_forecasting.registry import reads

    cfg = _cfg(backtest={"enabled": True, "n_folds": 3, "decision_metric": "mae"})
    monkeypatch.setattr(
        reads,
        "read_run_summary",
        lambda rid, *, settings=None: {"status": "COMPLETED", "n_series": 10},
    )
    monkeypatch.setattr(reads, "read_run_config", lambda rid, *, settings=None: cfg.model_dump())
    monkeypatch.setattr(reads, "read_leaderboard", lambda rid, *, settings=None: [])
    monkeypatch.setattr(
        reads,
        "read_metric_aggregates",
        lambda rid, *, settings=None: [_agg("theta", None, 0.2, mean_mae=1.5, p50_mae=1.4)],
    )
    monkeypatch.setattr(
        reads, "read_prediction_counts", lambda rid, *, settings=None: {"theta": 70}
    )
    monkeypatch.setattr(
        reads,
        "read_backtest_coverage",
        lambda rid, *, settings=None: [
            {
                "model_type": "theta",
                "ensemble_id": None,
                "backtest_status": "full",
                "n_folds_achieved": 3,
                "n_series": 10,
            }
        ],
    )
    monkeypatch.setattr(
        reads,
        "read_comparable_leaderboard",
        lambda rid, *, settings=None: [
            {"model_type": "theta", "ensemble_id": None, "pooled_wape": 0.18, "n_series": 10}
        ],
    )

    rr = R.review_run("rid", settings=_SETTINGS)
    assert rr.decision_metric == "mae"  # taken from the run's own config
    assert rr.best_overall.model_type == "theta" and rr.best_overall.score == 1.5
    assert rr.models[0].n_predictions == 70
    # The cohort context rides along with the ranking, keyed on (model_type, ensemble_id).
    assert rr.models[0].cohort.n_full == 10
    assert rr.models[0].pooled_wape == 0.18
    assert rr.models[0].n_comparable_series == 10


def test_forecaster_monitor_and_review_run_delegate(monkeypatch: Any) -> None:
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        R,
        "monitor_run",
        lambda rid, *, probe=False, settings=None: seen.setdefault("mon", (rid, probe, settings)),
    )
    monkeypatch.setattr(
        R, "review_run", lambda rid, *, settings=None: seen.setdefault("rev", (rid, settings))
    )
    f = sf.Forecaster.from_dict(
        {
            "run_name": "x",
            "data": {"source_table": "source_series_native", "horizon": 7},
            "models": ["theta"],
        },
        settings=_SETTINGS,
    )
    f.monitor()
    f.review_run()
    assert seen["mon"] == (f.run_id, False, _SETTINGS)  # registry-only unless asked
    assert seen["rev"] == (f.run_id, _SETTINGS)


def test_forecaster_monitor_passes_probe_through(monkeypatch: Any) -> None:
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        R,
        "monitor_run",
        lambda rid, *, probe=False, settings=None: seen.setdefault("mon", (rid, probe)),
    )
    f = sf.Forecaster.from_dict(
        {
            "run_name": "x",
            "data": {"source_table": "source_series_native", "horizon": 7},
            "models": ["theta"],
        },
        settings=_SETTINGS,
    )
    f.monitor(probe=True)
    assert seen["mon"] == (f.run_id, True)


# --- calibration report --------------------------------------------------------


def _arm(model: str, **over: Any) -> dict[str, Any]:
    """One `read_arm_comparison` row, defaulting to a correction that won two series in three."""
    row: dict[str, Any] = {
        "model_type": model,
        "compute_engine": "spark",
        "interval_calibration": "oof-per-step",
        "n_series": 3,
        "n_raw_arm": 0,
        "n_auto_decided": 0,
        "n_compared": 3,
        "n_corrected_wins": 2,
        "mean_margin": 0.04,
        "median_margin": 0.05,
    }
    row.update(over)
    return row


def _cov(model: str, step: int, coverage: float | None, n: int = 100) -> dict[str, Any]:
    """One `read_coverage_by_step` row."""
    return {
        "model_type": model,
        "horizon_step": step,
        "n": n,
        "coverage": coverage,
        "mean_width": 2.0 + step,
    }


def test_calibration_assembles_both_halves() -> None:
    rep = R._assemble_calibration(
        "rid", "wape", 0.8, [_arm("theta"), _arm("xgboost")], [_cov("theta", 1, 0.79)]
    )
    assert rep.run_id == "rid" and rep.decision_metric == "wape" and rep.nominal_coverage == 0.8
    assert [a.model_type for a in rep.arms] == ["theta", "xgboost"]
    assert rep.coverage[0].horizon_step == 1 and rep.coverage[0].mean_width == 3.0


def test_win_rate_is_a_share_not_a_count() -> None:
    # The average margin can look healthy while most series lose; the win rate is the check.
    arm = R._assemble_calibration("rid", "wape", 0.8, [_arm("theta")], []).arms[0]
    assert arm.win_rate == pytest.approx(2 / 3)


def test_win_rate_is_none_when_nothing_was_compared() -> None:
    # A run with one fold has no out-of-fold estimate to grade, so there is no rate to report —
    # and reporting 0.0 would read as "the correction lost every time".
    arm = R._assemble_calibration(
        "rid", "wape", 0.8, [_arm("theta", n_compared=0, n_corrected_wins=0)], []
    ).arms[0]
    assert arm.win_rate is None


def test_arm_row_tolerates_missing_and_null_fields() -> None:
    # BigQuery hands back NULLs for a native-engine row that never ran an arm comparison.
    arm = R._assemble_calibration(
        "rid", "wape", 0.8, [{"model_type": "arima_plus", "mean_margin": None}], []
    ).arms[0]
    assert arm.compute_engine is None and arm.mean_margin is None
    assert arm.n_series == 0 and arm.n_compared == 0 and arm.win_rate is None


def test_mean_coverage_weights_by_row_count() -> None:
    # A step with ten residuals must not swing the fleet number as hard as one with a thousand.
    rep = R._assemble_calibration(
        "rid", "wape", 0.8, [], [_cov("theta", 1, 0.9, n=900), _cov("theta", 2, 0.5, n=100)]
    )
    assert rep.mean_coverage == pytest.approx(0.86)


def test_mean_coverage_ignores_null_and_empty_steps() -> None:
    rep = R._assemble_calibration(
        "rid",
        "wape",
        0.8,
        [],
        [_cov("theta", 1, 0.8), _cov("theta", 2, None), _cov("t", 3, 0.4, 0)],
    )
    assert rep.mean_coverage == pytest.approx(0.8)
    assert R._assemble_calibration("rid", "wape", 0.8, [], []).mean_coverage is None


def test_worst_step_is_the_furthest_from_nominal_in_either_direction() -> None:
    # Over-covering is a defect too: a band wide enough to always contain the truth says nothing.
    rep = R._assemble_calibration(
        "rid",
        "wape",
        0.8,
        [],
        [_cov("theta", 1, 0.78), _cov("theta", 28, 0.55), _cov("xgboost", 1, 1.0)],
    )
    worst = rep.worst_step
    assert worst is not None
    assert (worst.model_type, worst.horizon_step) == ("theta", 28)
    assert R._assemble_calibration("rid", "wape", 0.8, [], []).worst_step is None


def test_calibration_report_composes_readers(monkeypatch: Any) -> None:
    from scale_forecasting.registry import reads

    cfg = _cfg(backtest={"enabled": True, "n_folds": 3, "decision_metric": "mae"})
    monkeypatch.setattr(reads, "read_run_config", lambda rid, *, settings=None: cfg.model_dump())
    monkeypatch.setattr(reads, "read_arm_comparison", lambda rid, *, settings=None: [_arm("theta")])
    monkeypatch.setattr(
        reads, "read_coverage_by_step", lambda rid, *, settings=None: [_cov("theta", 1, 0.79)]
    )

    rep = R.calibration_report("rid", settings=_SETTINGS)
    assert rep.decision_metric == "mae"  # taken from the run's own config, like `review_run`
    assert rep.nominal_coverage == pytest.approx(0.8)  # span of DEFAULT_QUANTILES, not a constant
    assert rep.arms[0].win_rate == pytest.approx(2 / 3)
    assert rep.mean_coverage == pytest.approx(0.79)


def test_calibration_report_falls_back_when_the_run_has_no_config(monkeypatch: Any) -> None:
    from scale_forecasting.registry import reads

    monkeypatch.setattr(reads, "read_run_config", lambda rid, *, settings=None: None)
    monkeypatch.setattr(reads, "read_arm_comparison", lambda rid, *, settings=None: [])
    monkeypatch.setattr(reads, "read_coverage_by_step", lambda rid, *, settings=None: [])

    rep = R.calibration_report("rid", settings=_SETTINGS)
    assert rep.decision_metric == "wape" and rep.arms == () and rep.coverage == ()


# --- plots (headless smoke) ----------------------------------------------------


def _use_agg() -> None:
    import matplotlib

    matplotlib.use("Agg")  # headless: no display for the smoke check


def test_plot_progress_one_bar_per_family() -> None:
    _use_agg()
    cfg = _cfg(models=["theta", "xgboost"], ensemble={"enabled": False})
    rp = R._assemble_progress(
        "rid",
        {"status": "RUNNING", "n_series": 10},
        cfg,
        [{"family": "statistical", "status": "RUNNING"}],
        [],
    )
    ax = R.plot_progress(rp)
    assert len(ax.get_yticklabels()) == 2  # statistical + ml
    assert "rid" in ax.get_title()


def _bar_labels(ax: Any) -> list[str]:
    return [t.get_text() for t in ax.texts]


def test_plot_progress_labels_a_running_family_with_its_quiet_time() -> None:
    # The whole point: a bar that stopped moving must say how long ago it stopped. A pending family
    # (no job row) has no age to report, and a finished one is not waiting on anything.
    _use_agg()
    cfg = _cfg(models=["theta", "xgboost"], ensemble={"enabled": False})
    rp = R._assemble_progress(
        "rid",
        {"status": "RUNNING", "n_series": 10},
        cfg,
        [
            {
                "family": "statistical",
                "status": "RUNNING",
                "started_at": _AT - timedelta(seconds=1320),
            },
            {"family": "ml", "status": "COMPLETED", "ended_at": _AT - timedelta(seconds=1320)},
        ],
        [],
        now=_AT,
    )
    labels = _bar_labels(R.plot_progress(rp))
    assert any("quiet 22m" in t for t in labels)
    assert sum("quiet" in t for t in labels) == 1  # not the COMPLETED family


def test_plot_progress_prefers_a_probe_verdict_over_the_quiet_time() -> None:
    # Both families have been quiet 22m; the probe says one is dead and the other is alive. A live
    # reading supersedes the inference from silence, so neither bar reports its age.
    _use_agg()
    from scale_forecasting.probes import reconcile, vocabulary

    cfg = _cfg(models=["theta", "xgboost"], ensemble={"enabled": False})
    rp = R._assemble_progress(
        "rid",
        {"status": "RUNNING", "n_series": 10},
        cfg,
        [
            {
                "family": "statistical",
                "status": "RUNNING",
                "started_at": _AT - timedelta(seconds=1320),
            },
            {"family": "ml", "status": "RUNNING", "started_at": _AT - timedelta(seconds=1320)},
        ],
        [],
        now=_AT,
    )
    verdicts = (
        _verdict("statistical", vocabulary.VERDICT_LOST),
        _verdict("ml", vocabulary.VERDICT_RUNNING),
    )
    rp = replace(rp, probe=reconcile.ProbeReport("rid", "RUNNING", True, verdicts, True))
    labels = _bar_labels(R.plot_progress(rp))
    assert any("lost" in t for t in labels)
    assert any("running confirmed" in t for t in labels)
    assert not any("quiet" in t for t in labels)  # the age is superseded, not appended


def test_plot_progress_says_on_the_bar_whether_the_card_did_anything() -> None:
    # A GPU family's bar is otherwise pixel-identical to a CPU family's, which is how an
    # accelerator went twenty-one jobs doing nothing without anyone noticing. Two words, on the
    # family that has a verdict and only that one.
    _use_agg()
    cfg = _cfg(models=["theta", "xgboost"], ensemble={"enabled": False})
    rp = R._assemble_progress(
        "rid",
        {"status": "COMPLETED", "n_series": 10},
        cfg,
        [
            {"family": "statistical", "status": "COMPLETED", "device_verdict": "ENGAGED_IDLE"},
            {"family": "ml", "status": "COMPLETED"},
        ],
        [],
        now=_AT,
    )
    labels = _bar_labels(R.plot_progress(rp))
    assert sum("gpu idle" in t for t in labels) == 1


def test_plot_progress_does_not_print_a_verdict_word_nobody_defined() -> None:
    # A label is not where a reader should first meet a verdict string; an unknown one is dropped.
    _use_agg()
    cfg = _cfg(models=["theta"], ensemble={"enabled": False})
    rp = R._assemble_progress(
        "rid",
        {"status": "COMPLETED", "n_series": 10},
        cfg,
        [{"family": "statistical", "status": "COMPLETED", "device_verdict": "SOMETHING_NEW"}],
        [],
        now=_AT,
    )
    assert not any("SOMETHING_NEW" in t for t in _bar_labels(R.plot_progress(rp)))


def test_plot_progress_drops_a_trust_registry_verdict_as_noise() -> None:
    # TRUST_REGISTRY is what the bar's status colour already says (terminal, or never launched).
    # Printing it on every row would bury the two verdicts that matter.
    _use_agg()
    from scale_forecasting.probes import reconcile, vocabulary

    cfg = _cfg(models=["theta"], ensemble={"enabled": False})
    rp = R._assemble_progress(
        "rid",
        {"status": "COMPLETED", "n_series": 10},
        cfg,
        [
            {
                "family": "statistical",
                "status": "COMPLETED",
                "ended_at": _AT - timedelta(seconds=1320),
            }
        ],
        [],
        now=_AT,
    )
    verdict = (_verdict("statistical", vocabulary.VERDICT_TRUST_REGISTRY),)
    rp = replace(rp, probe=reconcile.ProbeReport("rid", "COMPLETED", False, verdict, False))
    labels = _bar_labels(R.plot_progress(rp))
    assert not any("trust registry" in t for t in labels)
    assert not any("quiet" in t for t in labels)  # a finished family is not waiting on anything


def _verdict(family: str, verdict: str) -> Any:
    from scale_forecasting.probes import reconcile

    return reconcile.FamilyVerdict(
        family=family,
        runtime="spark",
        registry_status="RUNNING",
        native_state=None,
        exists=None,
        verdict=verdict,
        disagreement=False,
        n_done=0,
        n_expected=None,
        detail="",
    )


def test_plot_leaderboard_and_distribution_render_scored_models() -> None:
    _use_agg()
    aggs = [_agg("theta", None, 0.2), _agg("ensemble_mean", "d", 0.15)]
    rr = R._assemble_review("rid", {"status": "COMPLETED"}, "wape", 10, [], aggs, {})
    lb = R.plot_leaderboard(rr)
    assert len(lb.get_yticklabels()) == 2
    dist = R.plot_metric_distribution(rr)
    assert len(dist.get_yticklabels()) == 2


def test_plots_handle_empty_inputs() -> None:
    _use_agg()
    empty_rev = R.RunReview("rid", "COMPLETED", "wape", 0, (), {}, None, (), ())
    assert "no scored models" in R.plot_leaderboard(empty_rev).get_title()
    assert "no aggregated percentiles" in R.plot_metric_distribution(empty_rev).get_title()
    empty_prog = R._assemble_progress("rid", None, None, [], [])
    assert "no families" in R.plot_progress(empty_prog).get_title()


# --- public surface ------------------------------------------------------------


def test_review_surface_is_exported_from_package() -> None:
    for name in (
        "monitor_run",
        "review_run",
        "RunProgress",
        "RunReview",
        "plot_progress",
        "plot_leaderboard",
        "plot_metric_distribution",
        "calibration_report",
        "CalibrationReport",
    ):
        assert hasattr(sf, name), name
