"""Offline tests for the run-inspection layer (`scale_forecasting.review`).

Same shape as ``test_sdk.py``: the pure assembly/derivation functions are exercised directly with
hand-built reader dicts, the I/O entry points (`monitor_run`/`review_run`) are covered by
monkeypatching the `registry.bq` readers, and the plots get a headless (Agg) smoke check —
populated and empty. No GCP, no matplotlib display.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

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


def _model(model_type: str, family: str, *, ens: str | None = None, score: float | None = 0.2,
           **over: Any) -> R.ModelReview:
    base: dict[str, Any] = dict(
        model_type=model_type, family=family, ensemble_id=ens, is_ensemble=ens is not None,
        compute_engine="spark", n_series=10, score=score,
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
        {"family": "statistical", "runtime": "spark", "hardware": "cpu",
         "status": "COMPLETED", "runtime_seconds": 12.0},
        {"family": "ml", "runtime": "spark", "hardware": "cpu", "status": "RUNNING",
         "runtime_seconds": None},
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
        {"model_type": "holtwinters", "ensemble_id": None, "n_cells_done": 30,
         "mean_fit_seconds": 2.0},
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


# --- _assemble_review (pure) ---------------------------------------------------


def _agg(model_type: str, ens: str | None, wape: float, **over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "model_type": model_type, "ensemble_id": ens, "compute_engine": "spark",
        "n_series": 10, "mean_fit_seconds": 1.0,
        "mean_wape": wape, "p10_wape": wape - 0.05, "p50_wape": wape, "p90_wape": wape + 0.05,
    }
    base.update(over)
    return base


def test_assemble_review_from_aggregates_sorts_and_derives() -> None:
    aggs = [
        _agg("xgboost", None, 0.30),
        _agg("theta", None, 0.20),
        _agg("ensemble_mean", "d", 0.15),
    ]
    lb = [{"model_type": "theta", "ensemble_id": None, "median_fit_seconds": 1.1,
           "no_artifact_rate": 0.0}]
    rr = R._assemble_review("rid", {"status": "COMPLETED"}, "wape", 10, lb, aggs,
                            {"theta": 1400, "xgboost": 0})

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
        {"model_type": "theta", "ensemble_id": None, "compute_engine": "spark", "n_cells": 10,
         "mean_wape": 0.22, "median_fit_seconds": 1.0, "no_artifact_rate": 0.0},
        {"model_type": "naive_mean", "ensemble_id": None, "compute_engine": "spark", "n_cells": 10,
         "mean_wape": None, "median_fit_seconds": 0.1, "no_artifact_rate": 0.0},
    ]
    rr = R._assemble_review("rid", {"status": "COMPLETED"}, "wape", 10, lb, [], {})
    assert [m.model_type for m in rr.models] == ["theta", "naive_mean"]  # unscored sorts last
    assert rr.best_overall.model_type == "theta" and rr.models[0].score == 0.22
    assert rr.models[0].metric_means == {}  # no panel without aggregates


# --- I/O entry points ----------------------------------------------------------


def test_monitor_run_composes_readers(monkeypatch: Any) -> None:
    from scale_forecasting.registry import bq

    cfg = _cfg()
    seen: dict[str, Any] = {}

    def _summary(rid: str, *, settings: Any = None) -> dict[str, Any]:
        seen["run_id"] = rid
        seen["settings"] = settings
        return {"status": "RUNNING", "n_series": 10}

    monkeypatch.setattr(bq, "read_run_summary", _summary)
    monkeypatch.setattr(bq, "read_run_config", lambda rid, *, settings=None: cfg.model_dump())
    monkeypatch.setattr(bq, "read_run_jobs", lambda rid, *, settings=None:
                        [{"family": "statistical", "runtime": "spark", "hardware": "cpu",
                          "status": "RUNNING", "runtime_seconds": None}])
    monkeypatch.setattr(bq, "read_progress", lambda rid, *, settings=None:
                        [{"model_type": "theta", "ensemble_id": None, "n_cells_done": 5,
                          "mean_fit_seconds": 0.5}])

    rp = R.monitor_run("rid", settings=_SETTINGS)
    assert seen == {"run_id": "rid", "settings": _SETTINGS}  # readers get id + injected settings
    assert rp.status == "RUNNING" and rp.n_series == 10
    stat = next(f for f in rp.families if f.family == "statistical")
    assert stat.n_done == 5 and stat.n_expected == 10


def test_monitor_run_status_only_when_config_missing(monkeypatch: Any) -> None:
    from scale_forecasting.registry import bq

    monkeypatch.setattr(bq, "read_run_summary", lambda rid, *, settings=None: {"status": "PENDING"})
    monkeypatch.setattr(bq, "read_run_config", lambda rid, *, settings=None: None)
    rp = R.monitor_run("rid", settings=_SETTINGS)
    assert rp.status == "PENDING" and rp.families == ()


def test_monitor_run_with_probe_reuses_the_probe_reader_and_attaches_the_report(
    monkeypatch: Any,
) -> None:
    # probe=True must not re-read the registry: it delegates to the probe's single read+escalate
    # pass and keeps both halves — the progress it built and the report it reconciled.
    from scale_forecasting.probes import reconcile
    from scale_forecasting.registry import bq

    for name in ("read_run_summary", "read_run_config", "read_run_jobs", "read_progress"):
        monkeypatch.setattr(bq, name, _never_called(name))

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
    from scale_forecasting.registry import bq

    cfg = _cfg(backtest={"enabled": True, "n_folds": 3, "decision_metric": "mae"})
    monkeypatch.setattr(bq, "read_run_summary", lambda rid, *, settings=None:
                        {"status": "COMPLETED", "n_series": 10})
    monkeypatch.setattr(bq, "read_run_config", lambda rid, *, settings=None: cfg.model_dump())
    monkeypatch.setattr(bq, "read_leaderboard", lambda rid, *, settings=None: [])
    monkeypatch.setattr(bq, "read_metric_aggregates", lambda rid, *, settings=None:
                        [_agg("theta", None, 0.2, mean_mae=1.5, p50_mae=1.4)])
    monkeypatch.setattr(bq, "read_prediction_counts", lambda rid, *, settings=None: {"theta": 70})

    rr = R.review_run("rid", settings=_SETTINGS)
    assert rr.decision_metric == "mae"  # taken from the run's own config
    assert rr.best_overall.model_type == "theta" and rr.best_overall.score == 1.5
    assert rr.models[0].n_predictions == 70


def test_forecaster_monitor_and_review_run_delegate(monkeypatch: Any) -> None:
    seen: dict[str, Any] = {}
    monkeypatch.setattr(R, "monitor_run",
                        lambda rid, *, probe=False, settings=None:
                        seen.setdefault("mon", (rid, probe, settings)))
    monkeypatch.setattr(R, "review_run",
                        lambda rid, *, settings=None: seen.setdefault("rev", (rid, settings)))
    f = sf.Forecaster.from_dict(
        {"run_name": "x", "data": {"source_table": "source_series_native", "horizon": 7},
         "models": ["theta"]},
        settings=_SETTINGS,
    )
    f.monitor()
    f.review_run()
    assert seen["mon"] == (f.run_id, False, _SETTINGS)  # registry-only unless asked
    assert seen["rev"] == (f.run_id, _SETTINGS)


def test_forecaster_monitor_passes_probe_through(monkeypatch: Any) -> None:
    seen: dict[str, Any] = {}
    monkeypatch.setattr(R, "monitor_run",
                        lambda rid, *, probe=False, settings=None:
                        seen.setdefault("mon", (rid, probe)))
    f = sf.Forecaster.from_dict(
        {"run_name": "x", "data": {"source_table": "source_series_native", "horizon": 7},
         "models": ["theta"]},
        settings=_SETTINGS,
    )
    f.monitor(probe=True)
    assert seen["mon"] == (f.run_id, True)


# --- plots (headless smoke) ----------------------------------------------------


def _use_agg() -> None:
    import matplotlib

    matplotlib.use("Agg")  # headless: no display for the smoke check


def test_plot_progress_one_bar_per_family() -> None:
    _use_agg()
    cfg = _cfg(models=["theta", "xgboost"], ensemble={"enabled": False})
    rp = R._assemble_progress("rid", {"status": "RUNNING", "n_series": 10}, cfg,
                              [{"family": "statistical", "status": "RUNNING"}], [])
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
        "rid", {"status": "RUNNING", "n_series": 10}, cfg,
        [{"family": "statistical", "status": "RUNNING",
          "started_at": _AT - timedelta(seconds=1320)},
         {"family": "ml", "status": "COMPLETED", "ended_at": _AT - timedelta(seconds=1320)}],
        [], now=_AT,
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
        "rid", {"status": "RUNNING", "n_series": 10}, cfg,
        [{"family": "statistical", "status": "RUNNING",
          "started_at": _AT - timedelta(seconds=1320)},
         {"family": "ml", "status": "RUNNING", "started_at": _AT - timedelta(seconds=1320)}],
        [], now=_AT,
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


def test_plot_progress_drops_a_trust_registry_verdict_as_noise() -> None:
    # TRUST_REGISTRY is what the bar's status colour already says (terminal, or never launched).
    # Printing it on every row would bury the two verdicts that matter.
    _use_agg()
    from scale_forecasting.probes import reconcile, vocabulary

    cfg = _cfg(models=["theta"], ensemble={"enabled": False})
    rp = R._assemble_progress(
        "rid", {"status": "COMPLETED", "n_series": 10}, cfg,
        [{"family": "statistical", "status": "COMPLETED",
          "ended_at": _AT - timedelta(seconds=1320)}], [], now=_AT,
    )
    verdict = (_verdict("statistical", vocabulary.VERDICT_TRUST_REGISTRY),)
    rp = replace(rp, probe=reconcile.ProbeReport("rid", "COMPLETED", False, verdict, False))
    labels = _bar_labels(R.plot_progress(rp))
    assert not any("trust registry" in t for t in labels)
    assert not any("quiet" in t for t in labels)  # a finished family is not waiting on anything


def _verdict(family: str, verdict: str) -> Any:
    from scale_forecasting.probes import reconcile

    return reconcile.FamilyVerdict(
        family=family, runtime="spark", registry_status="RUNNING", native_state=None,
        exists=None, verdict=verdict, disagreement=False, n_done=0, n_expected=None, detail="",
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
    for name in ("monitor_run", "review_run", "RunProgress", "RunReview",
                 "plot_progress", "plot_leaderboard", "plot_metric_distribution"):
        assert hasattr(sf, name), name
